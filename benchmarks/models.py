"""PyTorch models used by the common benchmark training pipeline."""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(input_dim: int, output_dim: int, hidden: Iterable[int], dropout: float = 0.0) -> nn.Sequential:
    layers = []
    dim = input_dim
    for width in hidden:
        layers.extend([nn.Linear(dim, int(width)), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = int(width)
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class DeterministicMLP(nn.Module):
    def __init__(self, cond_dim: int = 2, x_dim: int = 2, hidden=(256, 256, 256), dropout: float = 0.0):
        super().__init__()
        self.net = _mlp(cond_dim, x_dim, hidden, dropout)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)


class ConditionalVAE(nn.Module):
    def __init__(self, cond_dim: int = 2, x_dim: int = 2, latent_dim: int = 16, hidden=(256, 256)):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = _mlp(x_dim + cond_dim, 2 * latent_dim, hidden)
        self.decoder = _mlp(latent_dim + cond_dim, x_dim, tuple(reversed(hidden)))

    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(torch.cat([x, condition], dim=1)).chunk(2, dim=1)
        return mu, logvar.clamp(-12.0, 12.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, condition], dim=1))

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        mu, logvar = self.encode(x, condition)
        return self.decode(self.reparameterize(mu, logvar), condition), mu, logvar


class ConditionalGenerator(nn.Module):
    def __init__(self, cond_dim: int = 2, x_dim: int = 2, noise_dim: int = 16, hidden=(256, 256, 256)):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = _mlp(noise_dim + cond_dim, x_dim, hidden)

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([noise, condition], dim=1))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, cond_dim: int = 2, x_dim: int = 2, hidden=(256, 256, 256)):
        super().__init__()
        self.net = _mlp(x_dim + cond_dim, 1, hidden)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, condition], dim=1)).squeeze(1)


class MixtureDensityNetwork(nn.Module):
    """Full-covariance bivariate Gaussian mixture p(lat, lon | condition)."""

    def __init__(self, cond_dim: int = 2, x_dim: int = 2, components: int = 10, hidden=(256, 256, 256)):
        super().__init__()
        if x_dim != 2:
            raise ValueError("현재 MDN은 2차원 (lat, lon) 출력 전용입니다.")
        self.components = int(components)
        self.net = _mlp(cond_dim, self.components * 6, hidden)

    def parameters_for(self, condition: torch.Tensor):
        out = self.net(condition).view(-1, self.components, 6)
        logits = out[:, :, 0]
        means = out[:, :, 1:3]
        log_scales = out[:, :, 3:5].clamp(-7.0, 5.0)
        rho = torch.tanh(out[:, :, 5]) * 0.98
        return logits, means, log_scales, rho

    def nll(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        logits, means, log_scales, rho = self.parameters_for(condition)
        scales = torch.exp(log_scales)
        delta = (x[:, None, :] - means) / scales
        one_minus_rho2 = (1.0 - rho.square()).clamp_min(1e-5)
        quad = (delta[:, :, 0].square() + delta[:, :, 1].square() - 2.0 * rho * delta[:, :, 0] * delta[:, :, 1]) / one_minus_rho2
        log_prob = (
            -math.log(2.0 * math.pi)
            - log_scales.sum(dim=2)
            - 0.5 * torch.log(one_minus_rho2)
            - 0.5 * quad
        )
        return -torch.logsumexp(F.log_softmax(logits, dim=1) + log_prob, dim=1).mean()

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature는 0보다 커야 합니다.")
        logits, means, log_scales, rho = self.parameters_for(condition)
        component = torch.distributions.Categorical(logits=logits / temperature).sample()
        row = torch.arange(len(condition), device=condition.device)
        mean = means[row, component]
        scale = torch.exp(log_scales[row, component]) * temperature
        corr = rho[row, component]
        eps = torch.randn_like(mean)
        x1 = mean[:, 0] + scale[:, 0] * eps[:, 0]
        x2_noise = corr * eps[:, 0] + torch.sqrt((1.0 - corr.square()).clamp_min(1e-5)) * eps[:, 1]
        x2 = mean[:, 1] + scale[:, 1] * x2_noise
        return torch.stack([x1, x2], dim=1)

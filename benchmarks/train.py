"""Unified trainer for MLP, CVAE, continuous cGAN, and MDN baselines."""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from benchmarks.common import load_benchmark_data, scaler_bundle, set_seed
from benchmarks.models import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    ConditionalVAE,
    DeterministicMLP,
    MixtureDensityNetwork,
)


MODEL_NAMES = ("mlp", "cvae", "cgan", "mdn")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 torch.cuda.is_available()이 False입니다.")
    return torch.device(name)


def _build(model_name: str, args, device: torch.device):
    hidden = tuple(args.hidden)
    if model_name == "mlp":
        model = DeterministicMLP(hidden=hidden, dropout=args.dropout)
        config = {"hidden": hidden, "dropout": args.dropout}
        return model.to(device), None, config
    if model_name == "cvae":
        model = ConditionalVAE(latent_dim=args.latent_dim, hidden=hidden)
        config = {"hidden": hidden, "latent_dim": args.latent_dim}
        return model.to(device), None, config
    if model_name == "cgan":
        generator = ConditionalGenerator(noise_dim=args.noise_dim, hidden=hidden).to(device)
        discriminator = ConditionalDiscriminator(hidden=hidden).to(device)
        config = {"hidden": hidden, "noise_dim": args.noise_dim}
        return generator, discriminator, config
    if model_name == "mdn":
        model = MixtureDensityNetwork(components=args.components, hidden=hidden).to(device)
        config = {"hidden": hidden, "components": args.components}
        return model, None, config
    raise ValueError(model_name)


def _cvae_loss(model: ConditionalVAE, x, c, beta: float):
    recon, mu, logvar = model(x, c)
    reconstruction = F.mse_loss(recon, x, reduction="none").sum(dim=1).mean()
    kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    return reconstruction + beta * kl, reconstruction, kl


@torch.no_grad()
def _validate(model_name: str, model, loader, device: torch.device, beta: float) -> Optional[float]:
    if loader is None or model_name == "cgan":
        return None
    model.eval()
    total, count = 0.0, 0
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        if model_name == "mlp":
            loss = F.mse_loss(model(c), x)
        elif model_name == "cvae":
            loss, _, _ = _cvae_loss(model, x, c, beta)
        else:
            loss = model.nll(x, c)
        total += float(loss.item()) * len(x)
        count += len(x)
    model.train()
    return total / max(1, count)


def _checkpoint(model_name, model, discriminator, model_config, data, args, epoch, step, metric):
    state = {
        "format_version": 1,
        "model_type": model_name,
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "scalers": scaler_bundle(data.x_scaler, data.c_scaler),
        "condition_columns": ["pdr_mean", "N"],
        "output_columns": ["lat", "lon"],
        "training_csv": os.path.abspath(args.csv),
        "seed": args.seed,
        "epoch": epoch,
        "step": step,
        "validation_metric": metric,
    }
    if discriminator is not None:
        state["discriminator_state_dict"] = discriminator.state_dict()
    return state


def train(args) -> None:
    set_seed(args.seed)
    device = _device(args.device)
    data = load_benchmark_data(args.csv, args.batch_size, args.val_ratio, args.seed, args.max_train)
    model, discriminator, model_config = _build(args.model, args, device)
    os.makedirs(args.out, exist_ok=True)

    print(f"model={args.model} device={device} train={data.n_train} val={data.n_val}")
    print(f"condition=[pdr_mean, N], csv={args.csv}")

    if args.model == "cgan":
        optimizer_g = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_limit = args.epochs
    if args.max_steps > 0:
        epoch_limit = max(epoch_limit, int(math.ceil(args.max_steps / max(1, len(data.train_loader)))))

    best_metric = math.inf
    global_step = 0
    stop = False
    for epoch in range(1, epoch_limit + 1):
        model.train()
        if discriminator is not None:
            discriminator.train()
        running = 0.0
        running_aux = 0.0
        batches = 0

        for x, c in data.train_loader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break
            x, c = x.to(device), c.to(device)

            if args.model == "mlp":
                loss = F.mse_loss(model(c), x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                aux = 0.0
            elif args.model == "cvae":
                warmup = min(1.0, float(global_step + 1) / max(1, args.kl_warmup_steps))
                beta = args.beta * warmup
                loss, reconstruction, kl = _cvae_loss(model, x, c, beta)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                aux = float(kl.item())
            elif args.model == "mdn":
                loss = model.nll(x, c)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                aux = 0.0
            else:
                batch = len(x)
                real_target = torch.full((batch,), 1.0 - args.label_smoothing, device=device)
                fake_target = torch.zeros(batch, device=device)
                noise = torch.randn(batch, args.noise_dim, device=device)
                fake = model(noise, c)

                optimizer_d.zero_grad()
                d_real = F.binary_cross_entropy_with_logits(discriminator(x, c), real_target)
                d_fake = F.binary_cross_entropy_with_logits(discriminator(fake.detach(), c), fake_target)
                d_loss = 0.5 * (d_real + d_fake)
                d_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()
                noise = torch.randn(batch, args.noise_dim, device=device)
                generated = model(noise, c)
                g_loss = F.binary_cross_entropy_with_logits(discriminator(generated, c), torch.ones(batch, device=device))
                g_loss.backward()
                optimizer_g.step()
                loss, aux = g_loss, float(d_loss.item())

            global_step += 1
            batches += 1
            running += float(loss.item())
            running_aux += aux
            if args.log_every > 0 and global_step % args.log_every == 0:
                label = "D" if args.model == "cgan" else ("KL" if args.model == "cvae" else "aux")
                print(f"step={global_step} loss={running / batches:.6f} {label}={running_aux / batches:.6f}")
                running = running_aux = 0.0
                batches = 0

        metric = _validate(args.model, model, data.val_loader, device, args.beta)
        metric_text = "n/a" if metric is None else f"{metric:.6f}"
        print(f"epoch={epoch}/{epoch_limit} step={global_step} val={metric_text}")
        last = _checkpoint(args.model, model, discriminator, model_config, data, args, epoch, global_step, metric)
        torch.save(last, os.path.join(args.out, "last_model.pt"))
        if metric is not None and metric < best_metric:
            best_metric = metric
            torch.save(last, os.path.join(args.out, "best_model.pt"))
        if stop:
            break

    if args.model == "cgan" or data.val_loader is None:
        torch.save(last, os.path.join(args.out, "best_model.pt"))
    print(f"saved: {args.out}")


def build_parser(default_model: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MCI conditional generation benchmark trainer")
    if default_model is None:
        parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    else:
        parser.set_defaults(model=default_model)
    parser.add_argument("--csv", default=os.path.join(_ROOT, "src", "data", "national_all.csv"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=0, help="0이면 epochs 기준")
    parser.add_argument("--max_train", type=int, default=0, help="smoke test용; 0이면 전체")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--kl_warmup_steps", type=int, default=5000)
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(default_model: Optional[str] = None) -> None:
    parser = build_parser(default_model)
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(_ROOT, "outputs", "benchmarks", args.model)
    train(args)


if __name__ == "__main__":
    main()

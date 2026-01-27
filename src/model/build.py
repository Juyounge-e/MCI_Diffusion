import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple

# tab-ddpm 모듈 재사용 (경로 보정)
try:
    from tab_ddpm.modules import MLPDiffusion
except ImportError:
    this_dir = os.path.dirname(__file__)
    tab_ddpm_root = os.path.abspath(os.path.join(this_dir, "..", "..", "tab-ddpm"))
    if tab_ddpm_root not in sys.path:
        sys.path.append(tab_ddpm_root)
    from tab_ddpm.modules import MLPDiffusion


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class MLPDiffusionConfig:
    x_dim: int = 2
    cond_dim: int = 1
    dim_t: int = 128
    d_layers: List[int] = None
    dropout: float = 0.1
    num_timesteps: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: torch.device = None

    def __post_init__(self):
        if self.d_layers is None:
            self.d_layers = [128, 128, 128]
        if self.device is None:
            self.device = get_device()


def build_mlp_diffusion(cfg: MLPDiffusionConfig) -> nn.Module:
    rtdl_params = {
        "d_layers": cfg.d_layers,
        "dropout": cfg.dropout,
    }
    model = MLPDiffusion(
        d_in=cfg.x_dim,
        num_classes=0,
        is_y_cond=True,
        rtdl_params=rtdl_params,
        dim_t=cfg.dim_t,
    )
    return model.to(cfg.device)


def build_optimizer(model: nn.Module, cfg: MLPDiffusionConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_model_and_optimizer(cfg: MLPDiffusionConfig) -> Tuple[nn.Module, torch.optim.Optimizer]:
    model = build_mlp_diffusion(cfg)
    optim = build_optimizer(model, cfg)
    return model, optim

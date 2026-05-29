import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple
from src.model.spatial_embedding import SpatialValidityEmbedding
from tab_ddpm.modules import timestep_embedding

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
    cond_dim: int = 5
    dim_t: int = 256
    d_layers: List[int] = None
    dropout: float = 0.1
    num_timesteps: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: torch.device = None
    mask_path: Optional[str] = None   # build_spatial_mask.py 로 생성한 npz 경로

    def __post_init__(self):
        if self.d_layers is None:
            self.d_layers = [128, 128, 128]
        if self.device is None:
            self.device = get_device()
        if self.mask_path is None:
            raise ValueError(
                "mask_path 가 필요합니다.\n"
                "먼저 scripts/build_spatial_mask.py 를 실행하여 mask_cache.npz 를 생성하세요."
            )


class MLPDiffusionWithSpatial(MLPDiffusion):
    """
    tab_ddpm.MLPDiffusion 을 상속받아 forward 만 오버라이드.
    SpatialValidityEmbedding 을 time_embed, label_emb 와
    동일한 레벨(emb)에서 합산.
    """

    def __init__(self, cfg: MLPDiffusionConfig):
        super().__init__(
            d_in=cfg.x_dim,
            num_classes=0,
            is_y_cond=True,
            rtdl_params={
                "d_layers": cfg.d_layers,
                "dropout": cfg.dropout,
            },
            dim_t=cfg.dim_t,
            d_y_cond=cfg.cond_dim,
        )

        # Spatial Validity Embedding — npz 캐시로 초기화 (geopandas 불필요)
        self.spatial_emb = SpatialValidityEmbedding(
            mask_path=cfg.mask_path,
            dim_t=cfg.dim_t,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor = None,
    ) -> torch.Tensor:
       
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))

        # condition embedding
        if self.is_y_cond and y is not None:
            y = y.float()
            if y.dim() == 1:
                y = y.unsqueeze(1)
            emb += F.silu(self.label_emb(y))

        # spatial embedding: x[:, 0]=lat, x[:, 1]=lon (정규화된 값)
        emb += self.spatial_emb(x[:, 0], x[:, 1])  # (batch, dim_t)

        # proj + emb → mlp
        x = self.proj(x) + emb
        return self.mlp(x)


def build_mlp_diffusion(cfg: MLPDiffusionConfig) -> nn.Module:
    """모델만 반환 (샘플링 시 사용)."""
    return MLPDiffusionWithSpatial(cfg).to(cfg.device)


def build_model_and_optimizer(
    cfg: MLPDiffusionConfig,
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """모델 + 옵티마이저 반환 (학습 시 사용)."""
    model = build_mlp_diffusion(cfg)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    return model, optim

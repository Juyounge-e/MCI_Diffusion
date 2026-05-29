# src/model/spatial_embedding.py

import numpy as np
import torch
import torch.nn as nn


class SpatialValidityEmbedding(nn.Module):
    """
    SHP 기반 공간 유효성 임베딩.

    (lat, lon) 좌표가 유효한 재난 발생 가능 위치인지
    (육지=1, 바다=0) 를 격자 이진 마스크로 조회
    """

    def __init__(
        self,
        mask_path: str,
        dim_t: int = 256,
    ):
        super().__init__()

        data = np.load(mask_path)
        mask       = data["mask"].astype(np.float32)   # (n_lat, n_lon)
        self.lat_min    = float(data["lat_min"])
        self.lon_min    = float(data["lon_min"])
        self.resolution = float(data["resolution"])

        print(
            f"[SpatialValidityEmbedding] 마스크 로드: {mask_path}\n"
            f"  shape={mask.shape}  유효 셀 비율={mask.mean():.3f}"
        )

        # 학습 중 업데이트되지 않는 고정 버퍼
        self.register_buffer("mask", torch.tensor(mask))

        # x_scaler 역변환 버퍼 (기본값: 항등 변환)
        # → set_scaler() 로 fit_transform_scalers() 결과를 주입해야 올바르게 동작
        self.register_buffer("x_mean", torch.zeros(2))  # [lat_mean, lon_mean]
        self.register_buffer("x_std",  torch.ones(2))   # [lat_std,  lon_std]

        self.mlp = nn.Sequential(
            nn.Linear(1, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

    # ------------------------------------------------------------------
    def set_scaler(self, mean: np.ndarray, std: np.ndarray) -> None:
        """
        StandardScaler 파라미터를 역변환 버퍼에 주입.
        train_mlp.py 에서 fit_transform_scalers() 직후 호출해야 함.

        Args:
            mean : x_scaler.mean_   shape (2,)  — [lat_mean, lon_mean]
            std  : x_scaler.scale_  shape (2,)  — [lat_std,  lon_std]
        """
        self.x_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        self.x_std.copy_(torch.tensor(std,  dtype=torch.float32))

    # ------------------------------------------------------------------
    def _lookup(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        """
        정규화된 (lat, lon) → 유효성 값 (0.0 or 1.0).
        x_std / x_mean 으로 역변환 후 지리 좌표 기반 마스크를 인덱싱.

        Args:
            lat : (batch,) 정규화된 위도 텐서
            lon : (batch,) 정규화된 경도 텐서
        Returns:
            (batch,) 유효성 값
        """
        # scaled → geographic
        lat_geo = lat * self.x_std[0] + self.x_mean[0]
        lon_geo = lon * self.x_std[1] + self.x_mean[1]

        lat_idx = torch.clamp(
            ((lat_geo - self.lat_min) / self.resolution).long(),
            0, self.mask.shape[0] - 1,
        )
        lon_idx = torch.clamp(
            ((lon_geo - self.lon_min) / self.resolution).long(),
            0, self.mask.shape[1] - 1,
        )
        return self.mask[lat_idx, lon_idx]  # (batch,)

    # ------------------------------------------------------------------
    def forward(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lat : (batch,) 정규화된 위도 텐서
            lon : (batch,) 정규화된 경도 텐서
        Returns:
            (batch, dim_t) 공간 유효성 임베딩
        """
        validity = self._lookup(lat, lon).unsqueeze(-1)  # (batch, 1)
        return self.mlp(validity)                         # (batch, dim_t)

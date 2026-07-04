import numpy as np
import torch
from scipy.spatial import cKDTree


class RoadDistanceLoss:
    def __init__(self, road_points_npz: str, x_mean, x_std, device):
        d = np.load(road_points_npz)
        pts = d["points"].astype(np.float64)            # (M, 2) = lat, lon 
        mean = np.asarray(x_mean, dtype=np.float64).reshape(1, 2)
        std = np.asarray(x_std, dtype=np.float64).reshape(1, 2)

        pts_scaled = (pts - mean) / std                 # 정규화 공간
        self.tree = cKDTree(pts_scaled)                 # 최근접 조회용 
        self.pts_scaled = torch.tensor(
            pts_scaled, dtype=torch.float32, device=device
        )
        self.device = device
        print(f"[RoadDistanceLoss] 도로점 {len(pts):,}개 로드, KDTree 구성 완료")

    def __call__(self, x0_hat: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            q = x0_hat.detach().cpu().numpy()
            _, idx = self.tree.query(q, k=1)            # 최근접 도로 1개를 찾음 
        nearest = self.pts_scaled[torch.as_tensor(idx, device=self.device)]
        d2 = ((x0_hat - nearest) ** 2).sum(dim=1)        # (B,) 미분 가능
        if weight is not None:
            return (weight * d2).sum() / (weight.sum() + 1e-8)
        return d2.mean()

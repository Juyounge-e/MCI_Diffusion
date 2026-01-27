import os
import argparse
import pickle
import csv
import sys

import torch
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TABDDPM = os.path.join(_ROOT, "tab-ddpm")
for _p in (_ROOT, _TABDDPM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model.build import MLPDiffusionConfig, build_mlp_diffusion
from src.diffusion.scheduler import TabDDPMGaussianScheduler


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=os.path.join("outputs", "mlp_diffusion", "model_last.pt"))
    parser.add_argument("--out", type=str, default=os.path.join("outputs", "mlp_diffusion", "samples_q1.csv"))
    parser.add_argument("--scalers", type=str, default=os.path.join("outputs", "mlp_diffusion", "scalers.pkl"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--cond", type=float, default=0.0, help="pdr_mean 값")
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    # 로드: cfg, 모델 가중치
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise RuntimeError("Checkpoint에 cfg가 없습니다.")
    cfg = MLPDiffusionConfig(**cfg_dict)
    device = cfg.device

    model = build_mlp_diffusion(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 스케일러 로드 (조건/출력 역변환용)
    with open(args.scalers, "rb") as f:
        scalers = pickle.load(f)
    x_scaler = scalers.x_scaler
    c_scaler = scalers.c_scaler

    # 조건 준비(스케일링)
    cond_np = np.array([[args.cond]], dtype=np.float32)
    if c_scaler is not None:
        cond_np = c_scaler.transform(cond_np)
    cond = torch.from_numpy(cond_np).float().to(device)  # (1,1)
    cond = cond.repeat(args.n, 1)  # (N,1)

    # 샘플링 초기화
    N = args.n
    T = args.timesteps
    x = torch.randn((N, cfg.x_dim), device=device, dtype=torch.float32)
    scheduler = TabDDPMGaussianScheduler(
        num_classes=np.array([0]),
        num_numerical_features=cfg.x_dim,
        denoise_fn=model,
        num_timesteps=cfg.num_timesteps,
        gaussian_loss_type='mse',
        scheduler='cosine',
        device=device
    )

    # 역확산 루프
    for t_ in reversed(range(T)):
        t = torch.full((N,), t_, device=device, dtype=torch.long)
        eps = model(x, t, y=cond)
        x = scheduler.gaussian_p_sample(eps, x, t)

    # 역스케일 변환
    x_np = x.detach().cpu().numpy()
    x_np = x_scaler.inverse_transform(x_np)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])
        writer.writerows(x_np.tolist())
    print(f"Saved: {args.out} ({len(x_np)} rows)")


if __name__ == "__main__":
    main()


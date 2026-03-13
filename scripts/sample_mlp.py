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
    parser.add_argument("--ckpt", type=str, default=os.path.join("outputs", "mlp_diffusion", "1575", "model_last.pt"))
    parser.add_argument("--out", type=str, default=os.path.join("outputs", "mlp_diffusion",'resolution', "q10_0.042927.csv"))
    parser.add_argument("--scalers", type=str, default=os.path.join("outputs", "mlp_diffusion","1575", "scalers.pkl"))
    parser.add_argument("--sample_num", type=int, default=200, help="샘플링할 데이터 수")
    parser.add_argument("--cond", type=float, default=0.042927, help="pdr_mean 값")
    parser.add_argument("--normal", nargs=2, type=float, default=None,  metavar=("mean", "std"), help="pdr를 Normal(mean, std)에서 샘플링")
    parser.add_argument("--uniform", nargs=2, type=float, default=None,  metavar=("min", "max"), help="pdr를 Uniform(min, max)에서 샘플링")
    parser.add_argument("--N", type=int, default=30, help="N 값 (cond_dim=2일 때 사용, 미지정 시 30)")
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    # 로드: cfg, 모델 가중치
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise RuntimeError("Checkpoint에 cfg가 없습니다.")
    cfg = MLPDiffusionConfig(**cfg_dict)
    device = torch.device(cfg.device)
    device_str = f"{device.type}:{device.index}" if device.index is not None else device.type

    model = build_mlp_diffusion(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 스케일러 로드 (조건/출력 역변환용)
    with open(args.scalers, "rb") as f:
        scalers = pickle.load(f)
    x_scaler = scalers.x_scaler
    c_scaler = scalers.c_scaler

    # 조건 준비(스케일링)
    cond_dim = getattr(cfg, "cond_dim", 1)
    N_val = getattr(args, "N", 30)
    n_samples = args.sample_num

    # 샘플링 방식 (uniform, normal, fixed)          
    if args.uniform is not None:
        low, high = args.uniform
        if low >= high:
            raise ValueError("--uniform low는 high보다 작아야 합니다.")
        pdr_samples = np.random.uniform(low=low, high=high, size=(n_samples, 1)).astype(np.float32)
    elif args.normal is not None:
        mean, std = args.normal
        if std < 0:
            raise ValueError("--normal std는 0 이상이어야 합니다.")
        pdr_samples = np.random.normal(loc=mean, scale=std, size=(n_samples, 1)).astype(np.float32)
    else:
        pdr_samples = np.full((n_samples, 1), args.cond, dtype=np.float32)

    if cond_dim == 2:
        n_col = np.full((n_samples, 1), float(N_val), dtype=np.float32)
        cond_np = np.concatenate([pdr_samples, n_col], axis=1)
    else:
        cond_np = pdr_samples
    if c_scaler is not None:
        cond_np = c_scaler.transform(cond_np)

    cond = torch.from_numpy(cond_np).float().to(device)  # (n, cond_dim)

    # 샘플링 초기화
    N = n_samples
    T = args.timesteps
    x = torch.randn((N, cfg.x_dim), device=device, dtype=torch.float32)
    scheduler = TabDDPMGaussianScheduler(
        num_classes=np.array([0]),
        num_numerical_features=cfg.x_dim,
        denoise_fn=model,
        num_timesteps=cfg.num_timesteps,
        gaussian_loss_type='mse',
        scheduler='cosine',
        device=device_str
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
        writer.writerow(["lat", "lon", "N", "pdr"])
        for i in range(len(x_np)):
            row = (
                [float(x_np[i, 0]), float(x_np[i, 1]), float(N_val), round(float(pdr_samples[i, 0]), 6)]
                if cond_dim == 2
                else [float(x_np[i, 0]), float(x_np[i, 1]), "", round(float(pdr_samples[i, 0]), 6)]
            )
            writer.writerow(row)
    print(f"Saved: {args.out} ({len(x_np)} rows)")


if __name__ == "__main__":
    main()


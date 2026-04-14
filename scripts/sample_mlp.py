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


def sample_ratio_from_bank(ratio_bank, n_val, tol, rng):
    candidates = []
    for k, ratios in ratio_bank.items():
        if abs(int(k) - int(n_val)) <= tol:
            candidates.extend(ratios)

    if not candidates:
        for ratios in ratio_bank.values():
            candidates.extend(ratios)

    if not candidates:
        raise ValueError("ratio_bank에 사용 가능한 ratio가 없습니다.")

    ratio = np.asarray(candidates[rng.integers(len(candidates))], dtype=np.float32)
    ratio = ratio / ratio.sum()
    return ratio


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=os.path.join("outputs", "mlp_diffusion", "test_rygb", "model_last.pt"))
    parser.add_argument("--out", type=str, default=os.path.join("outputs", "mlp_diffusion", "test_rygb", "samples_n_only.csv"))
    parser.add_argument("--scalers", type=str, default=os.path.join("outputs", "mlp_diffusion", "test_rygb", "scalers.pkl"))
    parser.add_argument("--sample_num", type=int, default=50, help="샘플링할 데이터 수")
    parser.add_argument("--pdr", type=float, default=0.042927, help="pdr_mean 값")
    parser.add_argument("--normal", nargs=2, type=float, default=None, metavar=("mean", "std"), help="pdr를 Normal(mean, std)에서 샘플링")
    parser.add_argument("--uniform", nargs=2, type=float, default=None, metavar=("min", "max"), help="pdr를 Uniform(min, max)에서 샘플링")
    parser.add_argument("--N_only", type=int, default=30, help="N-only 입력값")
    parser.add_argument(
        "--rygb", nargs=4, type=int, default=None, metavar=("RED", "YELLOW", "GREEN", "BLACK"), help="red/yellow/green/black count 직접 입력")
    parser.add_argument("--ratio_bank", type=str, default=os.path.join("outputs", "mlp_diffusion", "test_rygb", "ratio_bank.pkl"))
    parser.add_argument("--n_tol", type=int, default=2, help="N-only에서 ratio 검색 허용 범 위")
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

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

    with open(args.scalers, "rb") as f:
        scalers = pickle.load(f)
    x_scaler = scalers.x_scaler
    c_scaler = scalers.c_scaler

    cond_dim = getattr(cfg, "cond_dim", 5)
    if cond_dim != 5:
        raise ValueError(f"현재 sample_mlp.py는 cond_dim=5 전용입니다. 현재 cond_dim={cond_dim}")

    n_samples = args.sample_num
    rng = np.random.default_rng()

    if args.rygb is not None:
        red, yellow, green, black = args.rygb
        if min(red, yellow, green, black) < 0:
            raise ValueError("rygb는 음수일 수 없습니다.")
        N_val = red + yellow + green + black
        rygb_cols = np.tile(
            np.array([red, yellow, green, black], dtype=np.float32),
            (n_samples, 1),
        )
    else:
        N_val = args.N_only
        if N_val <= 0:
            raise ValueError("N_only는 1 이상의 정수여야 합니다.")

        with open(args.ratio_bank, "rb") as f:
            ratio_bank = pickle.load(f)

        rygb_list = []
        for _ in range(n_samples):
            ratio = sample_ratio_from_bank(ratio_bank, N_val, args.n_tol, rng)
            red, yellow, green, black = rng.multinomial(N_val, ratio).tolist()
            rygb_list.append([red, yellow, green, black])

        rygb_cols = np.array(rygb_list, dtype=np.float32)

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
        pdr_samples = np.full((n_samples, 1), args.pdr, dtype=np.float32)
    
    cond_np = np.concatenate([pdr_samples, rygb_cols], axis=1)

    if c_scaler is not None:
        cond_np = c_scaler.transform(cond_np)

    cond = torch.from_numpy(cond_np).float().to(device)

    N = n_samples
    T = args.timesteps
    x = torch.randn((N, cfg.x_dim), device=device, dtype=torch.float32)
    scheduler = TabDDPMGaussianScheduler(
        num_classes=np.array([0]),
        num_numerical_features=cfg.x_dim,
        denoise_fn=model,
        num_timesteps=cfg.num_timesteps,
        gaussian_loss_type="mse",
        scheduler="cosine",
        device=device_str,
    )

    for t_ in reversed(range(T)):
        t = torch.full((N,), t_, device=device, dtype=torch.long)
        eps = model(x, t, y=cond)
        x = scheduler.gaussian_p_sample(eps, x, t)

    x_np = x.detach().cpu().numpy()
    x_np = x_scaler.inverse_transform(x_np)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "N", "pdr", "red", "yellow", "green", "black"])
        for i in range(len(x_np)):
            writer.writerow([
                float(x_np[i, 0]),
                float(x_np[i, 1]),
                int(np.sum(rygb_cols[i])),
                round(float(pdr_samples[i, 0]), 6),
                int(rygb_cols[i, 0]),
                int(rygb_cols[i, 1]),
                int(rygb_cols[i, 2]),
                int(rygb_cols[i, 3]),
            ])

    print(f"Saved: {args.out} ({len(x_np)} rows)")


if __name__ == "__main__":
    main()

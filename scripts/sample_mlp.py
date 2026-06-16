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
    parser.add_argument("--N", type=int, default=30, help="총 rygb 개수 (항상 입력)")
    parser.add_argument(
        "--rygb", nargs=4, type=float, default=None, metavar=("RED", "YELLOW", "GREEN", "BLACK"),
        help="red/yellow/green/black 비율 직접 입력 (합이 1이 되도록 자동 정규화). 생략 시 ratio_bank에서 샘플링.")
    parser.add_argument("--ratio_bank", type=str, default=os.path.join("outputs", "mlp_diffusion", "test_rygb", "ratio_bank.pkl"))
    parser.add_argument("--n_tol", type=int, default=2, help="ratio_bank N 검색 허용 범위")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument( "--mask_path",  type=str, default=os.path.join("MCI_ADV2", "scenarios", "mask_cache.npz"), help="사전 계산된 공간 마스크 npz 경로", )
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise RuntimeError("Checkpoint에 cfg가 없습니다.")

    # 구버전 체크포인트(shp_path) 또는 mask_path 미포함 시 현재 경로로 보완
    cfg_dict.pop("shp_path", None)
    cfg_dict.setdefault("mask_path", args.mask_path)

    # 구버전 체크포인트(spatial_emb 가중치 포함) 호환:
    # state_dict 에 spatial_emb.* 키가 있으면 use_spatial_emb 를 켜서 로딩이 깨지지 않게 함.
    has_spatial = any(k.startswith("spatial_emb.") for k in ckpt["model"].keys())
    cfg_dict["use_spatial_emb"] = bool(cfg_dict.get("use_spatial_emb", False) or has_spatial)

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

    # 공간 임베딩 역변환용 스케일러 주입 (use_spatial_emb=True 인 구버전 체크포인트만)
    if getattr(model, "use_spatial_emb", False) and hasattr(model, "spatial_emb"):
        model.spatial_emb.set_scaler(x_scaler.mean_, x_scaler.scale_)

    cond_dim = getattr(cfg, "cond_dim", 5)
    if cond_dim != 5:
        raise ValueError(f"현재 sample_mlp.py는 cond_dim=5 전용입니다. 현재 cond_dim={cond_dim}")

    n_samples = args.sample_num
    rng = np.random.default_rng()

    N_val = args.N
    if N_val <= 0:
        raise ValueError("--N은 1 이상의 정수여야 합니다.")

    if args.rygb is not None:
        ratios = np.array(args.rygb, dtype=np.float32)
        if np.any(ratios < 0):
            raise ValueError("--rygb 값은 음수일 수 없습니다.")
        ratios = ratios / ratios.sum()
        rygb_ratios = np.tile(ratios, (n_samples, 1))
    else:
        with open(args.ratio_bank, "rb") as f:
            ratio_bank = pickle.load(f)
        rygb_ratios = np.array(
            [sample_ratio_from_bank(ratio_bank, N_val, args.n_tol, rng) for _ in range(n_samples)],
            dtype=np.float32,
        )

    # ratio → count (모델은 counts로 학습됨)
    rygb_counts = rygb_ratios * float(N_val)

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
    
    cond_np = np.concatenate([pdr_samples, rygb_counts], axis=1)  # (n_samples, 5)

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
        writer.writerow(["lat", "lon", "N", "pdr", "r_ratio", "y_ratio", "g_ratio", "b_ratio"])
        for i in range(len(x_np)):
            writer.writerow([
                float(x_np[i, 0]),
                float(x_np[i, 1]),
                N_val,
                round(float(pdr_samples[i, 0]), 6),
                round(float(rygb_ratios[i, 0]), 6),
                round(float(rygb_ratios[i, 1]), 6),
                round(float(rygb_ratios[i, 2]), 6),
                round(float(rygb_ratios[i, 3]), 6),
            ])

    print(f"Saved: {args.out} ({len(x_np)} rows)")


if __name__ == "__main__":
    main()

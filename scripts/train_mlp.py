import argparse
import math
import time
from dataclasses import asdict
import pickle
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure project root and tab-ddpm are importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TABDDPM = os.path.join(_ROOT, "tab-ddpm")
for _p in (_ROOT, _TABDDPM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
# TensorBoard가 distutils 의존으로 깨질 수 있어 명시적으로 확인한다.
# (torch.utils.tensorboard 가 `distutils.version.LooseVersion`을 쓰는데, 최신 setuptools에서는
#  distutils.version 서브모듈이 자동으로 안 붙어있어서 AttributeError가 남 → 미리 import 해서 방지)
try:
    import distutils.version  # noqa: F401
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception as exc:
    raise RuntimeError(
        "TensorBoard를 불러오지 못했습니다. 현재 Python 환경에 tensorboard를 "
        "설치한 뒤 다시 실행하세요: pip install tensorboard"
    ) from exc

import numpy as np
import pandas as pd
import random
from src.data.data_module import load_csv, make_splits, fit_transform_scalers, make_loaders
from src.model.build import MLPDiffusionConfig, build_model_and_optimizer
from src.diffusion.scheduler import TabDDPMGaussianScheduler


def linear_warmdown(step: int, total_steps: int, base_lr: float) -> float:
    frac_done = step / max(1, total_steps)
    return base_lr * (1.0 - frac_done)


@torch.no_grad()
def evaluate(model, val_loader, scheduler, road_loss_fn, road_lambda, device):
    """val_loader 전체에 대한 평균 loss(gauss/road/total). best_model 선정 및 과적합 모니터링용."""
    model.eval()
    total_gauss, total_road, total_n = 0.0, 0.0, 0
    for xb, cb in val_loader:
        xb = xb.to(device)
        cb = cb.to(device)
        b = xb.shape[0]
        t, _ = scheduler.sample_time(b, device, method='uniform')
        noise = torch.randn_like(xb)
        xt = scheduler.gaussian_q_sample(xb, t, noise=noise)
        model_out = model(xt, t, y=cb)
        loss_gauss = scheduler.gaussian_loss(model_out, xb, xt, t, noise).mean()
        total_gauss += loss_gauss.item() * b

        if road_loss_fn is not None:
            x0_hat = scheduler.predict_xstart(model_out, xt, t)
            w = scheduler.ddpm.alphas_cumprod[t]
            l_road = road_loss_fn(x0_hat, weight=w)
            total_road += l_road.item() * b
        total_n += b
    model.train()

    val_gauss = total_gauss / max(1, total_n)
    val_road = (total_road / max(1, total_n)) if road_loss_fn is not None else None
    val_total = val_gauss + (road_lambda * val_road if val_road is not None else 0.0)
    return val_gauss, val_road, val_total

# CSV에서 r/y/g/b 비율 계산 (N-only 옵션에서 활용)
def build_ratio_bank(df: pd.DataFrame):
    ratio_bank = {}
    for _, row in df.iterrows():
        counts = np.array([row["red"], row["yellow"], row["green"], row["black"]], dtype=np.float32)
        total = counts.sum()
        if total <= 0:
            continue
        n_key = int(row['N']) 
        ratio = counts / total
        ratio_bank.setdefault(n_key, []).append(ratio)
    return ratio_bank

def main():
    parser = argparse.ArgumentParser(description="MLP Diffusion 학습 (조건부 lat,lon 생성)")
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join("src", "data", "test.csv"),
        help="학습용 CSV 경로 (lat, lon, pdr_mean, [r/y/g/b] 필요)",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=0,
        help="학습에 사용할 최대 train 샘플 수. default=0이면 전체 사용.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("outputs", "mlp_diffusion", "test_national"),
        help="학습 결과를 저장할 디렉토리 경로",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument(
        "--mask_path",
        type=str,
        default=os.path.join("MCI_ADV2", "scenarios", "mask_cache_0.005.npz"),
        help="사전 계산된 spatial mask npz 경로 (scripts/build_spatial_mask.py 로 생성)",
    )
    parser.add_argument(
        "--road_points",
        type=str,
        default="",
        help="도로점 npz 경로 (scripts/build_road_points.py). 지정 시 road-distance aux loss 사용.",
    )
    parser.add_argument(
        "--road_lambda",
        type=float,
        default=0.0,
        help="road-distance aux loss 가중치 (0이면 비활성). 정규화좌표 스케일상 보통 1e2~1e3.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=10000,
        help="총 학습 스텝 수 (빠른 실험 시 줄여서 사용).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="검증셋 비율 (0이면 검증 없이 기존과 동일하게 전체 train으로 사용).",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=500,
        help="검증 loss 측정 주기(step).",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="이어서 학습할 체크포인트 경로 (model_step*.pt 또는 last_model.pt).",
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=0,
        help="resume_ckpt의 학습 시작 스텝 (LR 스케줄 이어받기용). 보통 저장된 스텝 수와 동일하게 입력.",
    )

    args = parser.parse_args()

    csv_path = args.csv
    max_train_samples = args.max_train
    out_dir = args.out_dir
    seed = args.seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----------------
    # Config (CSV에 r/y/g/b 있으면 항상 사용)
    # ----------------
    x_cols = ["lat", "lon"]
    _df = pd.read_csv(csv_path, nrows=1)
    has_red = "red" in _df.columns
    has_yellow = "yellow" in _df.columns
    has_green = "green" in _df.columns
    has_black = "black" in _df.columns
    has_rygb = has_red and has_yellow and has_green and has_black
    if has_rygb:
        c_cols = ["pdr_mean", "red", "yellow", "green", "black"]
        cond_dim = 5  # pdr, r, y, g, b (raw counts)
    else:
        c_cols = ["pdr_mean"]
        cond_dim = 1
        print("  [WARN] CSV에 r/y/g/b 컬럼 없음. pdr_mean만 사용.")

    print(f"  cond_cols = {c_cols} -> cond_dim={cond_dim}")

    cfg = MLPDiffusionConfig(
        x_dim=2,
        cond_dim=cond_dim,
        dim_t=256,
        d_layers=[256, 256, 256],
        dropout=0.1,
        num_timesteps=1000,
        lr=1e-3,
        weight_decay=1e-4,
        device=torch.device("cpu"),  # CPU 강제 (GPU 비호환 회피)
        mask_path=args.mask_path,   
    )
    device = cfg.device

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = os.path.join(out_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir, flush_secs=30)
    print(f"[tensorboard] log_dir={os.path.abspath(tb_dir)}")

    # ---------------
    # ratio bank 구축 (N-only 모델에서 활용)
    # ---------------
    df = pd.read_csv(csv_path)
    ratio_bank = build_ratio_bank(df)

    # ----------------
    # Data
    # ----------------
    print(f"  data: {csv_path}")
    print(f"  saved outputs to: {out_dir}")
    x, c = load_csv(csv_path, x_cols, c_cols)
    train, val, test = make_splits(x, c, val_ratio=args.val_ratio, test_ratio=0, seed=seed)

    # 학습 데이터 랜덤 샘플링 
    if max_train_samples > 0:
        x_train, c_train = train
        n_train = len(x_train)
        if n_train > max_train_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_train, size=max_train_samples, replace=False)
            x_train = x_train[idx]
            c_train = c_train[idx]
            train = (x_train, c_train)
            print(f"  train 샘플링: {n_train} -> {max_train_samples}")

    print(f"  train n = {len(train[0])}, val n = {len(val[0])}")
    train_s, val_s, test_s, scalers = fit_transform_scalers(train, val, test, scale_condition=True)
    train_loader, val_loader, _ = make_loaders(train_s, val_s, test_s, batch_size=128, num_workers=0)

    # ----------------
    # Model / Optim
    # ----------------
    model, optim = build_model_and_optimizer(cfg)

    # ----------------
    # Resume (선택)
    # ----------------
    global_step = args.resume_step
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"  [resume] {args.resume_ckpt} 로드 완료 (resume_step={global_step})")

    if getattr(model, "use_spatial_emb", False) and hasattr(model, "spatial_emb"):
        model.spatial_emb.set_scaler(
            mean=scalers.x_scaler.mean_,
            std=scalers.x_scaler.scale_,
        )

    model.train()

    # ----------------
    # Road-distance aux loss (선택)
    # ----------------
    road_loss_fn = None
    if args.road_lambda > 0 and args.road_points:
        from src.diffusion.road_loss import RoadDistanceLoss
        road_loss_fn = RoadDistanceLoss(
            args.road_points,
            x_mean=scalers.x_scaler.mean_,
            x_std=scalers.x_scaler.scale_,
            device=device,
        )
        print(f"  road-distance aux loss 활성: lambda={args.road_lambda}")

    # ----------------
    # Diffusion Scheduler
    # ----------------
    # 현재는 연속형만 사용 → K=[0], num_numerical_features=x_dim
    # 추후 범주형 추가 시 K를 실제 카테고리 크기 배열로 교체하면 mixed_loss가 자동 확장됨.
    scheduler = TabDDPMGaussianScheduler(
        num_classes=np.array([0]),
        num_numerical_features=cfg.x_dim,
        denoise_fn=model,
        num_timesteps=cfg.num_timesteps,
        gaussian_loss_type='mse',
        scheduler='cosine',
        device=device
    )

    # ----------------
    # Train loop
    # ----------------
    total_steps = args.total_steps
    log_every = 100
    ckpt_every = 5000

    has_val = args.val_ratio > 0 and len(val_loader) > 0
    best_val_loss = float("inf")

    start_time = time.time()
    while global_step < total_steps:
        for xb, cb in train_loader:
            xb = xb.to(device)  # (B, 2)
            cb = cb.to(device)  # (B, 5)

            # 타임스텝 샘플링
            b = xb.shape[0]
            t, pt = scheduler.sample_time(b, device, method='uniform')
            
            # 노이즈 샘플링 및 forward noising
            noise = torch.randn_like(xb)
            xt = scheduler.gaussian_q_sample(xb, t, noise=noise)
            
            # 모델 forward pass (조건부 입력)
            model_out = model(xt, t, y=cb)
            
            # Loss 계산 (_gaussian_loss는 배치별 loss를 반환하므로 mean 필요)
            loss_gauss = scheduler.gaussian_loss(model_out, xb, xt, t, noise)
            loss = loss_gauss.mean()

            # road-distance aux loss: x̂₀(복원 좌표)를 도로 매니폴드로 당김.
            # 고-t의 garbage x̂₀ 가 폭주하지 않게 ᾱ_t(SNR)로 가중.
            l_road = None
            if road_loss_fn is not None:
                x0_hat = scheduler.predict_xstart(model_out, xt, t)
                w = scheduler.ddpm.alphas_cumprod[t]  # (B,) ∈ (0,1], 저-t≈1 고-t≈0
                l_road = road_loss_fn(x0_hat, weight=w)
                loss = loss + args.road_lambda * l_road

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            # cosine/linear LR decay (here simple linear decay)
            lr = linear_warmdown(global_step, total_steps, cfg.lr)
            for g in optim.param_groups:
                g["lr"] = lr

            if (global_step + 1) % log_every == 0:
                elapsed = time.time() - start_time
                road_str = f" road={l_road.item():.4f}" if l_road is not None else ""
                print(f"[{global_step+1}/{total_steps}] loss={loss.item():.4f} "
                      f"gauss={loss_gauss.mean().item():.4f}{road_str} lr={lr:.2e} ({elapsed:.1f}s)")
                writer.add_scalar("train/loss", loss.item(), global_step + 1)
                writer.add_scalar("train/gauss", loss_gauss.mean().item(), global_step + 1)
                if l_road is not None:
                    writer.add_scalar("train/road", l_road.item(), global_step + 1)
                writer.add_scalar("train/lr", lr, global_step + 1)
                writer.flush()
                start_time = time.time()

            if has_val and (global_step + 1) % args.val_every == 0:
                val_gauss, val_road, val_total = evaluate(
                    model, val_loader, scheduler, road_loss_fn, args.road_lambda, device)
                road_str = f" road={val_road:.4f}" if val_road is not None else ""
                print(f"  [val] step={global_step+1} gauss={val_gauss:.4f}{road_str} total={val_total:.4f}")
                writer.add_scalar("val/gauss", val_gauss, global_step + 1)
                if val_road is not None:
                    writer.add_scalar("val/road", val_road, global_step + 1)
                writer.add_scalar("val/total", val_total, global_step + 1)
                writer.flush()

                if val_total < best_val_loss:
                    best_val_loss = val_total
                    torch.save(
                        {"model": model.state_dict(), "cfg": asdict(cfg),
                         "step": global_step + 1, "val_loss": best_val_loss},
                        os.path.join(out_dir, "best_model.pt"),
                    )
                    print(f"  [val] best 갱신 → best_model.pt 저장 (val_total={best_val_loss:.4f})")

            if (global_step + 1) % ckpt_every == 0:
                ckpt_path = os.path.join(out_dir, f"model_step{global_step+1}.pt")
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, ckpt_path)

            global_step += 1
            if global_step >= total_steps:
                break

    # 최종 저장
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, os.path.join(out_dir, "last_model.pt"))
    # 스케일러 저장(샘플링 시 재사용)
    with open(os.path.join(out_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    writer.flush()
    writer.close()
    #ratio_bank 저장 (N-only 모델에서 활용)
    with open(os.path.join(out_dir, "ratio_bank.pkl"), "wb") as f:
        pickle.dump(ratio_bank, f)


if __name__ == "__main__":
    main()


import math
import time
from dataclasses import asdict
import pickle
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure project root and tab-ddpm are importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TABDDPM = os.path.join(_ROOT, "tab-ddpm")
for _p in (_ROOT, _TABDDPM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
# TensorBoard가 distutils 의존으로 깨질 수 있어 선택적 임포트
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    class SummaryWriter:  # 최소 no-op 대체
        def __init__(self, *args, **kwargs): 
            pass
        def add_scalar(self, *args, **kwargs): 
            pass
        def close(self): 
            pass

import numpy as np
from src.data.data_module import load_csv, make_splits, fit_transform_scalers, make_loaders
from src.model.build import MLPDiffusionConfig, build_model_and_optimizer
from src.diffusion.scheduler import TabDDPMGaussianScheduler


def linear_warmdown(step: int, total_steps: int, base_lr: float) -> float:
    frac_done = step / max(1, total_steps)
    return base_lr * (1.0 - frac_done)


def main():
    # ----------------
    # Config
    # ----------------
    csv_path = os.path.join("src", "data", "dataset.csv")
    x_cols = ["lat", "lon"]
    c_cols = ["pdr_mean"]

    cfg = MLPDiffusionConfig(
        x_dim=2,
        cond_dim=1,
        dim_t=128,
        d_layers=[128, 128, 128],
        dropout=0.1,
        num_timesteps=1000,
        lr=1e-3,
        weight_decay=1e-4,
        device=torch.device("cpu"),  # CPU 강제 (GPU 비호환 회피)
    )
    device = cfg.device

    out_dir = os.path.join("outputs", "mlp_diffusion")
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

    # ----------------
    # Data
    # ----------------
    x, c = load_csv(csv_path, x_cols, c_cols)
    train, val, test = make_splits(x, c, val_ratio=0.1, test_ratio=0.1, seed=42)
    train_s, val_s, test_s, scalers = fit_transform_scalers(train, val, test, scale_condition=False)
    train_loader, val_loader, _ = make_loaders(train_s, val_s, test_s, batch_size=256, num_workers=0)

    # ----------------
    # Model / Optim
    # ----------------
    model, optim = build_model_and_optimizer(cfg)
    model.train()

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
    total_steps = 10_000
    log_every = 100
    ckpt_every = 1000

    global_step = 0
    start_time = time.time()
    while global_step < total_steps:
        for xb, cb in train_loader:
            xb = xb.to(device)  # (B, 2)
            cb = cb.to(device)  # (B, 1)

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

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            # cosine/linear LR decay (here simple linear decay)
            lr = linear_warmdown(global_step, total_steps, cfg.lr)
            for g in optim.param_groups:
                g["lr"] = lr

            if (global_step + 1) % log_every == 0:
                elapsed = time.time() - start_time
                print(f"[{global_step+1}/{total_steps}] loss={loss.item():.4f} lr={lr:.2e} ({elapsed:.1f}s)")
                writer.add_scalar("train/loss", loss.item(), global_step + 1)
                writer.add_scalar("train/lr", lr, global_step + 1)
                start_time = time.time()

            if (global_step + 1) % ckpt_every == 0:
                ckpt_path = os.path.join(out_dir, f"model_step{global_step+1}.pt")
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, ckpt_path)

            global_step += 1
            if global_step >= total_steps:
                break

    # 최종 저장
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, os.path.join(out_dir, "model_last.pt"))
    # 스케일러 저장(샘플링 시 재사용)
    with open(os.path.join(out_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    writer.close()


if __name__ == "__main__":
    main()


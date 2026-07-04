"""
train_mlp.py가 기록한 TensorBoard 로그(events.out.tfevents...)를 읽어서
loss 곡선 PNG 2장을 같은 tb 폴더에 저장.

- tb_curves.png   : gauss/road/total loss (train vs val) + lr, raw 값 그대로
- tb_smoothed.png : gauss loss (train vs val) raw + 이동평균(smoothed) 비교
                    (diffusion 학습 특유의 timestep 샘플링 노이즈에 가려진 추세 확인용)

road loss 태그(train/road, val/road)는 road_lambda=0으로 학습한 run엔 없으므로,
없으면 해당 서브플롯만 비워둠.

사용 예:
    python eval/eval_tb.py --tb_dir outputs/mlp_diffusion/daejeon_roadloss_30000/tb
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _series(ea: EventAccumulator, tag: str):
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def _smooth(v: np.ndarray, k: int) -> np.ndarray:
    if len(v) < k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="valid")


def plot_curves(ea: EventAccumulator, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    s, v = _series(ea, "train/gauss")
    sv, vv = _series(ea, "val/gauss")
    axes[0, 0].plot(s, v, label="train/gauss", alpha=0.7)
    if len(sv):
        axes[0, 0].plot(sv, vv, label="val/gauss", marker="o", ms=3)
    axes[0, 0].set_title("gauss loss (train vs val)")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    s, v = _series(ea, "train/road")
    sv, vv = _series(ea, "val/road")
    if len(s):
        axes[0, 1].plot(s, v, label="train/road", alpha=0.7)
    if len(sv):
        axes[0, 1].plot(sv, vv, label="val/road", marker="o", ms=3)
    axes[0, 1].set_title("road loss (train vs val)")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    s, v = _series(ea, "train/loss")
    sv, vv = _series(ea, "val/total")
    axes[1, 0].plot(s, v, label="train/loss (gauss+lambda*road)", alpha=0.7)
    if len(sv):
        axes[1, 0].plot(sv, vv, label="val/total", marker="o", ms=3)
    axes[1, 0].set_title("total loss (train vs val)")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    s, v = _series(ea, "train/lr")
    axes[1, 1].plot(s, v, label="train/lr", color="green")
    axes[1, 1].set_title("learning rate")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_smoothed_pair(ax, ea: EventAccumulator, train_tag: str, val_tag: str,
                         title: str, smooth_k_train: int, smooth_k_val: int):
    s, v = _series(ea, train_tag)
    sv, vv = _series(ea, val_tag)

    if not len(s):
        ax.set_title(f"{title} (로그 없음)")
        return

    ax.plot(s, v, alpha=0.25, color="tab:blue", label=f"{train_tag} (raw)")
    sm = _smooth(v, smooth_k_train)
    ax.plot(s[len(s) - len(sm):], sm, color="tab:blue", lw=2,
             label=f"{train_tag} (smoothed, k={smooth_k_train})")

    if len(sv):
        ax.plot(sv, vv, alpha=0.3, color="tab:orange", label=f"{val_tag} (raw)")
        smv = _smooth(vv, smooth_k_val)
        ax.plot(sv[len(sv) - len(smv):], smv, color="tab:orange", lw=2,
                 label=f"{val_tag} (smoothed, k={smooth_k_val})")

    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(title)


def plot_smoothed(ea: EventAccumulator, out_path: str, smooth_k_train: int, smooth_k_val: int):
    """gauss / road / total 을 각각 raw+smoothed로 따로 비교 (스케일이 달라 한 그래프에 같이 보면 왜곡됨)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    _plot_smoothed_pair(axes[0], ea, "train/gauss", "val/gauss",
                         "gauss loss: raw vs smoothed", smooth_k_train, smooth_k_val)
    _plot_smoothed_pair(axes[1], ea, "train/road", "val/road",
                         "road loss: raw vs smoothed", smooth_k_train, smooth_k_val)
    _plot_smoothed_pair(axes[2], ea, "train/loss", "val/total",
                         "total loss: raw vs smoothed", smooth_k_train, smooth_k_val)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="train_mlp.py TensorBoard 로그 → loss 곡선 PNG 생성")
    ap.add_argument("--tb_dir", required=True, help="tb 로그 폴더 경로 (events.out.tfevents... 가 있는 곳)")
    ap.add_argument("--smooth_k_train", type=int, default=15, help="train 곡선 이동평균 윈도우")
    ap.add_argument("--smooth_k_val", type=int, default=5, help="val 곡선 이동평균 윈도우")
    args = ap.parse_args()

    ea = EventAccumulator(args.tb_dir)
    ea.Reload()
    if not ea.Tags().get("scalars"):
        print(f"[오류] {args.tb_dir} 에서 scalar 로그를 못 찾음."); return 1

    curves_path = os.path.join(args.tb_dir, "tb_curves.png")
    smoothed_path = os.path.join(args.tb_dir, "tb_smoothed.png")

    plot_curves(ea, curves_path)
    plot_smoothed(ea, smoothed_path, args.smooth_k_train, args.smooth_k_val)

    print(f"저장 완료: {curves_path}")
    print(f"저장 완료: {smoothed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

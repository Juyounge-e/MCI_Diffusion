"""
eval/eval_results/qbin_national_lam*.csv 로부터
  - Bias by q-bin (grouped bar, per experiment)
  - 생성 샘플 수 n: 의도(intended) vs 실제(eval) by q-bin
두 개의 figure로 시각화.

Usage:
    python eval/plot_bias_n_national.py
    python eval/plot_bias_n_national.py --out eval/figures/bias_n_national.png
"""
from __future__ import annotations
import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_GLOB = "eval/eval_results/qbin_national_lam*.csv"
INTENDED_DIR = "outputs/mlp_diffusion/resolution_national_lam1_last"  # 모든 실험 동일

# 실험명 → 표시 레이블 + 색 그룹
_LABEL_MAP = {
    "qbin_national_lam0_40k":        ("lam0 40k",        "lam0"),
    "qbin_national_lam0_50k":        ("lam0 50k",        "lam0"),
    "qbin_national_lam0_best_200k":  ("lam0 best 200k",  "lam0"),
    "qbin_national_lam0_last_200k":  ("lam0 last 200k",  "lam0"),
    "qbin_national_lam1_40k":        ("lam1 40k",        "lam1"),
    "qbin_national_lam1_50k":        ("lam1 50k",        "lam1"),
    "qbin_national_lam1_best_200k":  ("lam1 best 200k",  "lam1"),
    "qbin_national_lam1_last_200k":  ("lam1 last 200k",  "lam1"),
    "qbin_national_lam10_best_200k": ("lam10 best 200k", "lam10"),
    "qbin_national_lam10_last_200k": ("lam10 last 200k", "lam10"),
}

# 그룹별 색상 팔레트
_GROUP_COLORS = {
    "lam0":  ["#4e79a7", "#a0cbe8", "#76b7b2", "#59a14f"],
    "lam1":  ["#f28e2b", "#ffbe7d", "#e15759", "#ff9d9a"],
    "lam10": ["#b07aa1", "#d4a6c8"],
}


def _load_intended(intended_dir: str) -> dict[str, int]:
    """q-bin CSV 파일 행 수 → {bin_name: count}"""
    out = {}
    for p in Path(intended_dir).glob("q*.csv"):
        stem = p.stem  # e.g. q1_0.035898
        df = pd.read_csv(p)
        out[stem] = len(df)
    return dict(sorted(out.items()))


def _load_eval(eval_glob: str) -> dict[str, pd.DataFrame]:
    """{stem: df (ALL 행 제외, q-bin 행만)}"""
    out = {}
    for p in sorted(glob.glob(eval_glob)):
        stem = Path(p).stem
        df = pd.read_csv(p)
        df = df[df["bin"] != "ALL"].copy()
        # bin 컬럼에서 q번호 추출 정렬
        df["_qnum"] = df["bin"].str.extract(r"q(\d+)_").astype(int)
        df = df.sort_values("_qnum").reset_index(drop=True)
        out[stem] = df
    return out


def plot_bias(eval_dfs: dict[str, pd.DataFrame], out_path: Path | None):
    fig, ax = plt.subplots(figsize=(14, 5))

    # 공통 bin 레이블 (어떤 실험이든 같은 q-bin)
    first_df = next(iter(eval_dfs.values()))
    bins = first_df["bin"].tolist()
    x = np.arange(len(bins))
    n_exp = len(eval_dfs)
    width = 0.8 / n_exp

    group_counters = {g: 0 for g in _GROUP_COLORS}
    for i, (stem, df) in enumerate(eval_dfs.items()):
        label, group = _LABEL_MAP.get(stem, (stem, "lam0"))
        color_list = _GROUP_COLORS[group]
        color = color_list[group_counters[group] % len(color_list)]
        group_counters[group] += 1

        offsets = (i - n_exp / 2 + 0.5) * width
        # bin 순서 맞추기 (일부 실험은 특정 bin 없을 수 있음)
        biases = []
        for b in bins:
            row = df[df["bin"] == b]
            biases.append(float(row["Bias"].iloc[0]) if len(row) > 0 else np.nan)

        ax.bar(x + offsets, biases, width=width * 0.9, label=label, color=color, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([b.split("_")[0] for b in bins], fontsize=9)
    ax.set_xlabel("PDR Quantile Bin")
    ax.set_ylabel("Bias  (simul_pdr − gen_pdr)")
    ax.set_title("Bias by Q-bin  [National, all experiments]")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if out_path:
        bias_path = out_path.parent / (out_path.stem + "_bias.png")
        fig.savefig(bias_path, dpi=150)
        print(f"Bias 그래프 저장: {bias_path}")
    else:
        fig.savefig("eval/figures/bias_national.png", dpi=150)
        print("eval/figures/bias_national.png 저장")
    plt.close(fig)


def plot_n(eval_dfs: dict[str, pd.DataFrame], intended: dict[str, int], out_path: Path | None):
    fig, ax = plt.subplots(figsize=(14, 5))

    first_df = next(iter(eval_dfs.values()))
    bins = first_df["bin"].tolist()
    x = np.arange(len(bins))

    n_exp = len(eval_dfs)
    # intended + 실험들
    total_bars = 1 + n_exp
    width = 0.8 / total_bars

    # intended bar (회색)
    intended_vals = []
    for b in bins:
        # bin 이름이 q1_0.035898 → intended key도 동일 형식
        intended_vals.append(intended.get(b, np.nan))

    ax.bar(x + (0 - total_bars / 2 + 0.5) * width,
           intended_vals, width=width * 0.9,
           label="Intended (gen)", color="#aaaaaa", alpha=0.9, hatch="//")

    group_counters = {g: 0 for g in _GROUP_COLORS}
    for i, (stem, df) in enumerate(eval_dfs.items()):
        label, group = _LABEL_MAP.get(stem, (stem, "lam0"))
        color_list = _GROUP_COLORS[group]
        color = color_list[group_counters[group] % len(color_list)]
        group_counters[group] += 1

        offsets = ((i + 1) - total_bars / 2 + 0.5) * width
        ns = []
        for b in bins:
            row = df[df["bin"] == b]
            ns.append(int(row["n"].iloc[0]) if len(row) > 0 else 0)

        ax.bar(x + offsets, ns, width=width * 0.9, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([b.split("_")[0] for b in bins], fontsize=9)
    ax.set_xlabel("PDR Quantile Bin")
    ax.set_ylabel("Sample Count (n)")
    ax.set_title("Sample Count by Q-bin  [Intended vs Actual after snap+sim]")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if out_path:
        n_path = out_path.parent / (out_path.stem + "_n.png")
        fig.savefig(n_path, dpi=150)
        print(f"N 그래프 저장: {n_path}")
    else:
        fig.savefig("eval/figures/n_national.png", dpi=150)
        print("eval/figures/n_national.png 저장")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_glob", default=EVAL_GLOB)
    ap.add_argument("--intended_dir", default=INTENDED_DIR,
                    help="의도 샘플수 확인용 q-bin CSV 디렉토리")
    ap.add_argument("--out", default=None,
                    help="출력 경로 기본명 (없으면 eval/figures/ 에 저장)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        Path("eval/figures").mkdir(parents=True, exist_ok=True)

    print("Eval 파일 로드 중...")
    eval_dfs = _load_eval(args.eval_glob)
    print(f"  {len(eval_dfs)}개 실험 로드: {list(eval_dfs.keys())}")

    print("Intended 샘플수 로드 중...")
    intended = _load_intended(args.intended_dir)
    print(f"  {intended}")

    plot_bias(eval_dfs, out_path)
    plot_n(eval_dfs, intended, out_path)
    print("완료!")


if __name__ == "__main__":
    main()

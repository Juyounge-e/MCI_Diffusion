"""
전체 학습 데이터 기준 q-bin을 만든 뒤,
목표 N±3 데이터가 각 q-bin에 얼마나 속하는지 비율을 계산하고,
q-bin별 sample_mlp.py 실행까지 처리하는 스크립트.
"""

import argparse
import os
import subprocess
from typing import Optional

import numpy as np
import pandas as pd


def build_bin_stats(
    source_csv: str,
    total_samples: int,
    num_bins: int,
    n_min: Optional[int] = None,
    n_max: Optional[int] = None,
) -> pd.DataFrame:
    # =============================================
    # 1) 전체 학습 데이터 로드
    # =============================================
    full_df = pd.read_csv(source_csv)

    if "pdr_mean" not in full_df.columns:
        raise KeyError(f"'pdr_mean' 컬럼이 없습니다: {source_csv}")

    if n_min is not None or n_max is not None:
        if "N" not in full_df.columns:
            raise KeyError(f"'N' 컬럼이 없습니다: {source_csv}")

    # =============================================
    # 2) 전체 학습 데이터 기준 q-bin 경계 생성
    # =============================================
    _, bin_edges = pd.qcut(
        full_df["pdr_mean"],
        q=num_bins,
        retbins=True,
        duplicates="drop",
    )

    actual_bins = len(bin_edges) - 1
    labels = [f"q{i + 1}" for i in range(actual_bins)]

    bin_edges = bin_edges.astype(float)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # =============================================
    # 3) 목표 N 범위 필터링
    # =============================================
    df = full_df.copy()

    if n_min is not None:
        df = df[df["N"] >= n_min]

    if n_max is not None:
        df = df[df["N"] <= n_max]

    df = df.copy()

    if df.empty:
        raise ValueError(
            f"N 필터링 결과가 비어 있습니다. "
            f"(n_min={n_min}, n_max={n_max})"
        )

    # =============================================
    # 4) 전체 학습 데이터 기준 bin을 N-filtered df에 적용
    # =============================================
    df["q_bin"] = pd.cut(
        df["pdr_mean"],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    outside_count = df["q_bin"].isna().sum()
    if outside_count > 0:
        print(f"[경고] bin 밖 데이터 수: {outside_count}")
        print(df[df["q_bin"].isna()][["pdr_mean", "N"]].head(20).to_string(index=False))

    # =============================================
    # 5) bin별 통계 계산
    # =============================================
    bin_stats = (
        df.groupby("q_bin", observed=False)
        .agg(
            count=("pdr_mean", "count"),
            pdr_mean=("pdr_mean", "mean"),
            pdr_std=("pdr_mean", "std"),
            pdr_min=("pdr_mean", "min"),
            pdr_max=("pdr_mean", "max"),
        )
        .reset_index()
    )

    # =============================================
    # 6) ratio 및 sample_num 계산
    # =============================================
    total_count = bin_stats["count"].sum()
    bin_stats["ratio"] = bin_stats["count"] / max(1, total_count)

    bin_stats["sample_num"] = (
        bin_stats["ratio"] * total_samples
    ).round().astype(int)

    bin_stats["pdr_std"] = bin_stats["pdr_std"].fillna(0.0)

    # 반올림 보정
    diff = int(total_samples - bin_stats["sample_num"].sum())
    if diff != 0 and len(bin_stats) > 0:
        target_idx = bin_stats["count"].idxmax()
        bin_stats.loc[target_idx, "sample_num"] += diff

    return bin_stats


def load_or_build_bin_stats(args: argparse.Namespace) -> pd.DataFrame:
    if args.training_csv:
        bin_stats = build_bin_stats(
            source_csv=args.training_csv,
            total_samples=args.total_samples,
            num_bins=args.num_bins,
            n_min=args.n_min,
            n_max=args.n_max,
        )

        if args.bin_stats:
            out_dir = os.path.dirname(os.path.abspath(args.bin_stats))
            os.makedirs(out_dir, exist_ok=True)
            bin_stats.to_csv(
                args.bin_stats,
                index=False,
                float_format="%.10f",
            )

        return bin_stats

    if not args.bin_stats:
        raise ValueError("--training_csv 또는 --bin_stats 중 하나는 필요합니다.")

    return pd.read_csv(args.bin_stats)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_csv", type=str, default=None, help="전체 학습 데이터 CSV")
    parser.add_argument("--bin_stats", type=str, default=os.path.join("notebooks", "30runs_bin_stats_n30.csv"))
    parser.add_argument("--out_dir", type=str, default=os.path.join("outputs", "mlp_diffusion", "resolution_seed_free_30runs30"))
    parser.add_argument("--ckpt", type=str, default=os.path.join("outputs", "mlp_diffusion", "test_rygb_30runs", "model_last.pt"))
    parser.add_argument("--scalers", type=str, default=os.path.join("outputs", "mlp_diffusion", "test_rygb_30runs", "scalers.pkl"))
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--total_samples", type=int, default=5000)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--n_min", type=int, default=27)
    parser.add_argument("--n_max", type=int, default=33)
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--rygb", nargs=4, type=int, default=None,
                        metavar=("red", "yellow", "green", "black"),
                        help="r/y/g/b 직접 입력 옵션")
    parser.add_argument("--ratio_bank", type=str,
                        default=os.path.join("outputs", "mlp_diffusion", "test_rygb_30runs", "ratio_bank.pkl"))
    parser.add_argument("--n_tol", type=int, default=3,
                        help="N-only에서 ratio 검색 허용 범위")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bin_stats = load_or_build_bin_stats(args)

    # =============================================
    # bin_stats 출력
    # =============================================
    print("\n========== bin_stats ==========")
    print(bin_stats.to_string(index=False))

    if args.bin_stats:
        print(f"\n저장 완료: {args.bin_stats} ({len(bin_stats)}행)")

    print(f"총 샘플링 개수: {bin_stats['sample_num'].sum()}개\n")

    # =============================================
    # q-bin별 sample_mlp.py 실행
    # =============================================
    for _, row in bin_stats.iterrows():
        q_bin     = row["q_bin"]
        pdr_mean  = float(row["pdr_mean"]) if not pd.isna(row["pdr_mean"]) else None
        n_samples = int(row["sample_num"])

        if n_samples <= 0:
            print(f"[{q_bin}] sample_num=0 이라 건너뜁니다.")
            continue

        if pdr_mean is None:
            print(f"[{q_bin}] pdr_mean이 NaN이라 건너뜁니다.")
            continue

        out_csv = os.path.join(args.out_dir, f"{q_bin}_{pdr_mean:.6f}.csv")

        cmd = [
            args.python_bin,
            os.path.join("scripts", "sample_mlp.py"),
            "--ckpt",       args.ckpt,
            "--scalers",    args.scalers,
            "--out",        out_csv,
            "--sample_num", str(n_samples),
            "--N",          str(args.N),
            "--timesteps",  str(args.timesteps),
            "--ratio_bank", args.ratio_bank,
            "--n_tol",      str(args.n_tol),
        ]

        # rygb 직접 입력 옵션
        if args.rygb is not None:
            cmd += ["--rygb"] + [str(v) for v in args.rygb]

        # df_30 실제 min/max 기준으로 샘플링
        sample_min = float(row["pdr_min"]) if not pd.isna(row["pdr_min"]) else pdr_mean
        sample_max = float(row["pdr_max"]) if not pd.isna(row["pdr_max"]) else pdr_mean

        if sample_min == sample_max:
            cmd += ["--cond", str(pdr_mean)]
            print(f"[{q_bin}] n={n_samples}, cond={pdr_mean:.10f}")
        else:
            cmd += ["--uniform", str(sample_min), str(sample_max)]
            print(f"[{q_bin}] n={n_samples}, range=[{sample_min:.10f}, {sample_max:.10f}]")

        subprocess.run(cmd, check=True)
        print(f"  저장 완료: {out_csv}\n")


if __name__ == "__main__":
    main()

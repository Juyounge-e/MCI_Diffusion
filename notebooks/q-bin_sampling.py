"""
q-bin 통계 생성과 q-bin별 sample_mlp.py 실행을 한 번에 처리하는 스크립트.
"""
import argparse
import os
import subprocess
import pandas as pd
from typing import Optional


def build_bin_stats(
    source_csv: str,
    total_samples: int,
    num_bins: int,
    n_min: Optional[int] = None,
    n_max: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_csv(source_csv)

    if "pdr_mean" not in df.columns:
        raise KeyError(f"'pdr_mean' 컬럼이 없습니다: {source_csv}")

    if n_min is not None or n_max is not None:
        if "N" not in df.columns:
            raise KeyError(f"'N' 컬럼이 없습니다: {source_csv}")

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

    df = df.copy()
    _, bin_edges = pd.qcut(
        df["pdr_mean"],
        q=num_bins,
        retbins=True,
        duplicates="drop",
    )

    labels = [f"q{i + 1}" for i in range(len(bin_edges) - 1)]
    df["q_bin"] = pd.qcut(
        df["pdr_mean"],
        q=num_bins,
        labels=labels,
        duplicates="drop",
    )

    bin_stats = (
        df.groupby("q_bin", observed=False)
        .agg(
            count=("pdr_mean", "count"),
            pdr_mean=("pdr_mean", "mean"),
            pdr_std=("pdr_mean", "std"),
        )
        .reset_index()
    )

    bin_stats["bin_min"] = bin_edges[:-1]
    bin_stats["bin_max"] = bin_edges[1:]
    bin_stats["bin_width"] = bin_stats["bin_max"] - bin_stats["bin_min"]
    bin_stats["ratio"] = bin_stats["count"] / max(1, bin_stats["count"].sum())
    bin_stats["sample_num"] = (bin_stats["ratio"] * total_samples).round().astype(int)
    bin_stats["pdr_std"] = bin_stats["pdr_std"].fillna(0.0)

    diff = int(total_samples - bin_stats["sample_num"].sum())
    if diff != 0 and len(bin_stats) > 0:
        bin_stats.loc[bin_stats.index[-1], "sample_num"] += diff

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
            bin_stats.to_csv(args.bin_stats, index=False)

        return bin_stats

    if not args.bin_stats:
        raise ValueError("--training_csv 또는 --bin_stats 중 하나는 필요합니다.")

    return pd.read_csv(args.bin_stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_csv", type=str, default=None, help="학습 데이터 CSV")
    parser.add_argument("--bin_stats", type=str, default=os.path.join("notebooks", "30_runs_rygb_bin_stats_n30.csv"))
    parser.add_argument("--out_dir", type=str, default=os.path.join("outputs", "mlp_diffusion", "rygb_resolution_30runs"))
    parser.add_argument("--ckpt", type=str, default=os.path.join("outputs", "mlp_diffusion", "test_rygb_30runs", "model_last.pt"))
    parser.add_argument("--scalers", type=str, default=os.path.join("outputs", "mlp_diffusion", "test_rygb_30runs", "scalers.pkl"))
    parser.add_argument("--ratio_bank", type=str, default=os.path.join("outputs", "mlp_diffusion", "test_rygb_30runs", "ratio_bank.pkl"))
    parser.add_argument("--N_only", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--total_samples", type=int, default=5000, help="bin 비율에 따라 분배할 총 샘플 수")
    parser.add_argument("--num_bins", type=int, default=10, help="q-bin 개수")
    parser.add_argument("--n_min", type=int, default=27, help="학습 데이터에서 사용할 최소 N")
    parser.add_argument("--n_max", type=int, default=33, help="학습 데이터에서 사용할 최대 N")
    parser.add_argument("--python_bin", type=str, default="python", help="sample_mlp.py 실행에 사용할 파이썬")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bin_stats = load_or_build_bin_stats(args)

    if "bin_width" in bin_stats.columns:
        print(f"bin 간격: {bin_stats['bin_width'].mean():.6f}")
    print(bin_stats.to_string(index=False))
    if args.bin_stats:
        print(f"\n저장 완료: {args.bin_stats} ({len(bin_stats)}행)")
    print(f"총 샘플링 개수: {bin_stats['sample_num'].sum()}개\n")

    for _, row in bin_stats.iterrows():
        q_bin = row["q_bin"]
        bin_min = float(row["bin_min"])
        bin_max = float(row["bin_max"]) if not pd.isna(row["bin_max"]) else None
        pdr_mean = float(row["pdr_mean"])
        n_samples = int(row["sample_num"])

        if n_samples <= 0:
            print(f"[{q_bin}] sample_num=0 이라 건너뜁니다.")
            continue

        out_csv = os.path.join(args.out_dir, f"{q_bin}_{pdr_mean:.6f}.csv")
        cmd = [
            args.python_bin,
            os.path.join("scripts", "sample_mlp.py"),
            "--ckpt",
            args.ckpt,
            "--scalers",
            args.scalers,
            "--out",
            out_csv,
            "--ratio_bank",
            args.ratio_bank,
            "--sample_num",
            str(n_samples),
            "--N_only",
            str(args.N_only),
            "--timesteps",
            str(args.timesteps),
        ]

        if bin_max is None or bin_min == bin_max:
            cmd += ["--pdr", str(pdr_mean)]
        else:
            cmd += ["--uniform", str(bin_min), str(bin_max)]

        upper = pdr_mean if bin_max is None else bin_max
        print(f"[{q_bin}] n={n_samples}, range=[{bin_min:.6f}, {upper:.6f}]")
        subprocess.run(cmd, check=True)
        print(f"  저장 완료: {out_csv}\n")


if __name__ == "__main__":
    main()

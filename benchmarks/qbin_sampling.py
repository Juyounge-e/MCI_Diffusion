"""Run the same national_all quantile-bin sampling protocol for any benchmark."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from benchmarks.common import build_qbin_stats
from benchmarks.train import MODEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark q-bin sampler")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--training_csv", default=os.path.join(_ROOT, "src", "data", "national_all.csv"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--total_samples", type=int, default=300)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--n_min", type=int, default=27)
    parser.add_argument("--n_max", type=int, default=33)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--python_bin", default=sys.executable)
    args = parser.parse_args()

    if args.total_samples <= 0:
        raise ValueError("total_samples는 1 이상이어야 합니다.")
    os.makedirs(args.out_dir, exist_ok=True)
    stats = build_qbin_stats(args.training_csv, args.total_samples, args.num_bins, args.n_min, args.n_max)
    stats_path = os.path.join(args.out_dir, "_bin_stats.csv")
    stats.to_csv(stats_path, index=False, float_format="%.10f")
    print(stats.to_string(index=False))
    print(f"allocated={int(stats['sample_num'].sum())} stats={stats_path}")

    paths = []
    sampler = os.path.join(_ROOT, "benchmarks", "sample.py")
    for i, row in stats.iterrows():
        count = int(row["sample_num"])
        if count <= 0 or pd.isna(row["pdr_mean"]):
            continue
        pdr_mean = float(row["pdr_mean"])
        out_path = os.path.join(args.out_dir, f"q{i + 1}_{pdr_mean:.6f}.csv")
        cmd = [
            args.python_bin, sampler,
            "--model", args.model,
            "--ckpt", args.ckpt,
            "--out", out_path,
            "--sample_num", str(count),
            "--N", str(args.N),
            "--temperature", str(args.temperature),
            "--seed", str(args.seed + i),
            "--device", args.device,
        ]
        low = float(row["pdr_min"])
        high = float(row["pdr_max"])
        if low < high:
            cmd.extend(["--uniform", str(low), str(high)])
        else:
            cmd.extend(["--pdr", str(pdr_mean)])
        print(f"[{row['q_bin']}] n={count} pdr=[{low:.8f}, {high:.8f}]")
        subprocess.run(cmd, check=True)
        paths.append(out_path)

    if paths:
        merged = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        merged_path = os.path.join(args.out_dir, "all_bins.csv")
        merged.to_csv(merged_path, index=False)
        print(f"merged={merged_path} rows={len(merged)}")


if __name__ == "__main__":
    main()

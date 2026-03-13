#!/usr/bin/env python
"""
bin_stats_n30.csv 기반으로 각 q-bin마다 sample_mlp.py를 자동 실행하는 스크립트
Usage: python notebook/run_sampling_bins.py --bin_stats bin_stats_n30.csv --out_dir outputs/mlp_diffusion/resolution_n30
"""
import argparse
import os
import subprocess
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_stats", type=str, default="bin_stats_n30.csv")
    parser.add_argument("--out_dir", type=str, default=os.path.join("outputs", "mlp_diffusion", "resolution_n30"))
    parser.add_argument("--ckpt", type=str, default=os.path.join("outputs", "mlp_diffusion", "1575", "model_last.pt"))
    parser.add_argument("--scalers", type=str, default=os.path.join("outputs", "mlp_diffusion", "1575", "scalers.pkl"))
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bin_stats = pd.read_csv(args.bin_stats)

    # normal 분포 샘플링 반영 안됨
    for _, row in bin_stats.iterrows():
        q_bin     = row['q_bin']
        bin_min   = float(row['bin_min'])
        bin_max   = float(row['bin_max']) if not pd.isna(row['bin_max']) else None
        pdr_mean  = float(row['pdr_mean'])
        n_samples = int(row['sample_num'])

        out_csv = os.path.join(args.out_dir, f"{q_bin}_{pdr_mean:.6f}.csv")

        cmd = [
            "python", os.path.join("scripts", "sample_mlp.py"),
            "--ckpt",       args.ckpt,
            "--scalers",    args.scalers,
            "--out",        out_csv,
            "--sample_num", str(n_samples),
            "--N",          str(args.N),
            "--timesteps",  str(args.timesteps),
        ]

        # q10처럼 bin_max가 없거나 bin_min == bin_max면 고정값 사용
        if bin_max is None or bin_min == bin_max:
            cmd += ["--cond", str(pdr_mean)]
        else:
            cmd += ["--uniform", str(bin_min), str(bin_max)]

        print(f"[{q_bin}] n={n_samples}, range=[{bin_min:.6f}, {bin_max if bin_max else pdr_mean:.6f}]")
        subprocess.run(cmd, check=True)
        print(f"  저장 완료: {out_csv}\n")

if __name__ == "__main__":
    main()
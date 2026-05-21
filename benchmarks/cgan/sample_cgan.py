"""
CGAN 샘플링: condition(pdr_bin, N_bin) → (lat, lon) 생성
- tab-ddpm CTGAN 모듈 사용
- N 없으면: condition_value='pdr_0'|'pdr_1'|'pdr_2'
- N 있으면: condition_value='pdr_1_N_2' (조합 레이블)
"""
import os
import sys
import argparse
import pickle
import csv

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "tab-ddpm"), os.path.join(_ROOT, "tab-ddpm", "CTGAN")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def sample_cgan(
    ckpt_path: str,
    meta_path: str,
    out_path: str,
    condition_value: str,
    n: int = 20,
    seed: int = 0,
):
    """condition_value: meta.cond_labels 중 하나"""
    with open(ckpt_path, "rb") as f:
        synthesizer = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    cond_col = meta.get("cond_column", meta.get("discrete_col", "cond"))
    cond_labels = meta.get("cond_labels", meta.get("bin_labels", ["pdr_0", "pdr_1", "pdr_2"]))
    x_cols = meta.get("x_cols", ["lat", "lon"])

    use_cond = condition_value in cond_labels
    if not use_cond:
        print(f"Warning: {condition_value} not in {cond_labels}. Using unconditional sampling.")

    gen = synthesizer.sample(
        n,
        condition_column=cond_col if use_cond else None,
        condition_value=condition_value if use_cond else None,
    )

    x_np = gen[x_cols].values

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(x_cols)
        writer.writerows(x_np.tolist())
    print(f"Saved {n} samples to {out_path} (cond {cond_col}={condition_value})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=os.path.join(_ROOT, "outputs", "cgan", "cgan.obj"),
    )
    parser.add_argument(
        "--meta",
        default=os.path.join(_ROOT, "outputs", "cgan", "cgan_meta.pkl"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_ROOT, "outputs", "cgan", "samples.csv"),
    )
    parser.add_argument(
        "--cond",
        default="pdr_1",
        help="cond_labels 중 하나 (N 없으면 pdr_0|pdr_1|pdr_2, N 있으면 pdr_0_N_1 등)",
    )
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    sample_cgan(
        ckpt_path=args.ckpt,
        meta_path=args.meta,
        out_path=args.out,
        condition_value=args.cond,
        n=args.n,
    )

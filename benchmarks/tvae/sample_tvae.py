"""
TVAE 샘플링: tab-ddpm/CTGAN TVAESynthesizer
- 무조건부: 학습된 결합분포에서 샘플 생성
"""
import os
import sys
import argparse
import pickle
import csv

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TABDDPM = os.path.join(_ROOT, "tab-ddpm")
_CTGAN = os.path.join(_TABDDPM, "CTGAN")
for _p in (_ROOT, _TABDDPM, _CTGAN):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def sample_tvae(
    ckpt_path: str,
    meta_path: str,
    out_path: str,
    n: int = 20,
    seed: int = 0,
    x_cols: tuple = ("lat", "lon"),
):
    with open(ckpt_path, "rb") as f:
        synthesizer = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    gen = synthesizer.sample(n, seed=seed)

    cols = meta.get("cols", list(gen.columns))
    x_cols = [c for c in x_cols if c in gen.columns] or list(cols)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    gen[x_cols].to_csv(out_path, index=False)
    print(f"Saved {n} samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=os.path.join(_ROOT, "outputs", "tvae", "tvae.obj"),
    )
    parser.add_argument(
        "--meta",
        default=os.path.join(_ROOT, "outputs", "tvae", "tvae_meta.pkl"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_ROOT, "outputs", "tvae", "samples.csv"),
    )
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    sample_tvae(
        ckpt_path=args.ckpt,
        meta_path=args.meta,
        out_path=args.out,
        n=args.n,
    )

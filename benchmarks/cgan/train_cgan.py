"""
CGAN 학습: X=(lat, lon), Condition=(pdr_bin, N_bin)
- tab-ddpm CTGAN 모듈 import 사용
- utils.prepare_cgan_dataset()로 데이터 정제 후 학습
- 추론: condition(pdr_bin, N_bin) 입력 → 위경도 출력
"""
import os
import sys
import argparse
import pickle

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "tab-ddpm"), os.path.join(_ROOT, "tab-ddpm", "CTGAN")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.cgan.utils import prepare_cgan_dataset


def train_cgan(
    csv_path: str,
    out_dir: str,
    pdr_n_bins: int = 3,
    n_n_bins: int = 3,
    epochs: int = 300,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    try:
        from CTGAN.ctgan import CTGANSynthesizer
    except ImportError:
        from ctgan import CTGANSynthesizer

    refined = prepare_cgan_dataset(
        csv_path,
        pdr_n_bins=pdr_n_bins,
        n_n_bins=n_n_bins,
    )
    df = refined.df

    synthesizer = CTGANSynthesizer(
        epochs=epochs,
        batch_size=min(batch_size, len(df)),
        cuda=(device == "cuda"),
    )
    synthesizer.fit(df, discrete_columns=refined.discrete_cols)

    with open(os.path.join(out_dir, "cgan.obj"), "wb") as f:
        pickle.dump(synthesizer, f)

    cond_col = refined.discrete_cols[0]
    meta = {
        "x_cols": refined.x_cols,
        "discrete_cols": refined.discrete_cols,
        "cond_column": cond_col,
        "cond_labels": sorted(df[cond_col].unique().tolist()),
        "has_N": refined.has_N,
    }
    with open(os.path.join(out_dir, "cgan_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved to {out_dir}, discrete_cols={refined.discrete_cols}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=os.path.join(_ROOT, "src", "data", "snaped_dataset.csv"),
    )
    parser.add_argument("--out", default=os.path.join(_ROOT, "outputs", "cgan"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--pdr_n_bins", type=int, default=3)
    parser.add_argument("--n_n_bins", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_cgan(
        csv_path=args.csv,
        out_dir=args.out,
        epochs=args.epochs,
        pdr_n_bins=args.pdr_n_bins,
        n_n_bins=args.n_n_bins,
        device=args.device,
    )

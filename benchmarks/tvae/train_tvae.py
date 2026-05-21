"""
TVAE 학습: tab-ddpm/CTGAN의 TVAESynthesizer 사용
- 입력: CSV (lat, lon, pdr_mean, ...)
- 무조건부: 전체 칼럼의 결합분포 학습
"""
import os
import sys
import argparse
import pickle
import warnings

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TABDDPM = os.path.join(_ROOT, "tab-ddpm")
_CTGAN = os.path.join(_TABDDPM, "CTGAN")
for _p in (_ROOT, _TABDDPM, _CTGAN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

try:
    from CTGAN.ctgan import TVAESynthesizer
except ImportError:
    from ctgan import TVAESynthesizer


def train_tvae(
    csv_path: str,
    out_dir: str,
    cols: tuple = ("lat", "lon", "pdr_mean"),
    discrete_cols: tuple = (),
    batch_size: int = 256,
    epochs: int = 300,
    embedding_dim: int = 128,
    device: str = "cpu",
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    cols = [c for c in cols if c in df.columns]
    df = df[cols].dropna()

    discrete_cols = [c for c in discrete_cols if c in df.columns]
    batch_size = min(batch_size, len(df))

    synthesizer = TVAESynthesizer(
        batch_size=batch_size,
        epochs=epochs,
        embedding_dim=embedding_dim,
        device=device,
    )
    synthesizer.fit(df, discrete_columns=discrete_cols)

    with open(os.path.join(out_dir, "tvae.obj"), "wb") as f:
        pickle.dump(synthesizer, f)

    meta = {"cols": cols, "discrete_cols": discrete_cols}
    with open(os.path.join(out_dir, "tvae_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=os.path.join(_ROOT, "src", "data", "snaped_dataset.csv"),
    )
    parser.add_argument("--out", default=os.path.join(_ROOT, "outputs", "tvae"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_tvae(
        csv_path=args.csv,
        out_dir=args.out,
        epochs=args.epochs,
        device=args.device,
    )

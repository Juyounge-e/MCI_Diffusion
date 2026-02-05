"""
CVAE 학습: (lat, lon) | pdr_mean, N
- 데이터: src/data/snapped_dataset.csv
- X: lat, lon / Condition: pdr_mean,N
"""
import os
import sys
import argparse
import pickle

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "tab-ddpm"), os.path.join(_ROOT, "tab-ddpm", "CTGAN")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.data.data_module import load_csv, make_splits, fit_transform_scalers, X_COLS, COND_COLS
from benchmarks.cvae.model import  CondEncoder, CondDecoder


def train_cvae(
    csv_path: str,
    out_dir: str,
    x_cols=None,
    c_cols=None,
    embedding_dim=128,
    compress_dims=(128, 128),
    decompress_dims=(128, 128),
    batch_size=256,
    epochs=100,
    lr=1e-3,
    loss_factor=2,
    device="cpu",
    seed=42,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    x_cols = x_cols or X_COLS
    c_cols = c_cols or COND_COLS
    x, c = load_csv(csv_path, x_cols, c_cols)
    train, val, test = make_splits(x, c, val_ratio=0, test_ratio=0, seed=seed)
    (x_train_s, c_train_s), _, _, scalers = fit_transform_scalers(
        train, val, test, scale_condition=True
    )

    x_dim = x_train_s.shape[1]
    cond_dim = c_train_s.shape[1]
    data_dim = x_dim  # decoder 출력은 x만 (lat, lon)

    encoder = CondEncoder(x_dim, cond_dim, compress_dims, embedding_dim).to(device)
    decoder = CondDecoder(embedding_dim, cond_dim, decompress_dims, data_dim).to(device)
    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )

    ds = TensorDataset(
        torch.from_numpy(x_train_s.astype("float32")),
        torch.from_numpy(c_train_s.astype("float32")),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # CVAE는 DataTransformer 대신 StandardScaler 사용 (연속형만)
    encoder.train()
    decoder.train()
    for ep in range(epochs):
        total_loss = 0.0
        for xb, cb in loader:
            xb = xb.to(device)
            cb = cb.to(device)
            optimizer.zero_grad()
            mu, std, logvar = encoder(xb, cb)
            eps = torch.randn_like(std)
            z = eps * std + mu
            rec, sigmas = decoder(z, cb)
            rec_tanh = torch.tanh(rec)
            # 로스 함수 정의*수정~필요할수도 
            loss_recon = ((xb - rec_tanh) ** 2 / (2 * sigmas**2)).sum(dim=1).mean()
            loss_recon += torch.log(sigmas).sum()
            kld = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=1).mean()
            loss = loss_recon + kld
            loss.backward()
            optimizer.step()
            decoder.sigma.data.clamp_(0.01, 1.0)
            total_loss += loss.item()
        if (ep + 1) % 50 == 0:
            print(f"Epoch {ep+1}/{epochs} loss={total_loss/len(loader):.4f}")

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "x_dim": x_dim,
            "cond_dim": cond_dim,
            "embedding_dim": embedding_dim,
            "compress_dims": compress_dims,
            "decompress_dims": decompress_dims,
        },
        os.path.join(out_dir, "cvae.pt"),
    )
    with open(os.path.join(out_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=os.path.join(_ROOT, "src", "data", "snapped_dataset.csv"),
    )
    parser.add_argument("--out", default=os.path.join(_ROOT, "outputs", "cvae"))
    parser.add_argument(
        "--x_cols", nargs="+", default=None,
        help="생성 대상 (기본: lat, lon)",
    )
    parser.add_argument(
        "--c_cols", nargs="+", default=None,
        help="조건 칼럼 (기본: pdr_mean, N)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_cvae(
        csv_path=args.csv,
        out_dir=args.out,
        x_cols=tuple(args.x_cols) if args.x_cols else None,
        c_cols=tuple(args.c_cols) if args.c_cols else None,
        epochs=args.epochs,
        device=args.device,
    )

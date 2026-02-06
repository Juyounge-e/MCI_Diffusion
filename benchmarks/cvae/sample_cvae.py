"""
CVAE 샘플링: 조건(pdr_mean, N)으로 (lat, lon) 생성
- cond: [pdr_mean, N] 2개 필수
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

import torch
import numpy as np

from benchmarks.cvae.model import CondDecoder


@torch.no_grad()
def sample_cvae(
    ckpt_path: str,
    scalers_path: str,
    out_path: str,
    cond,
    n: int = 20,
    seed: int = 0,
    device: str = "cpu",
):
    """
    cond: [pdr_mean, N] 필수
    """
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt = torch.load(ckpt_path, map_location=device)
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    x_dim = ckpt["x_dim"]
    cond_dim = ckpt["cond_dim"]
    embedding_dim = ckpt["embedding_dim"]
    decompress_dims = ckpt["decompress_dims"]

    decoder = CondDecoder(
        embedding_dim, cond_dim, decompress_dims, x_dim
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()

    # cond = [pdr_mean, N]
    cond_list = [float(v) for v in cond]
    if len(cond_list) != 2:
        raise ValueError("cond는 [pdr_mean, N] 2개 필수")
    cond_np = np.array([cond_list], dtype=np.float32)
    if cond_np.shape[1] != cond_dim:
        raise ValueError(f"cond dim {cond_np.shape[1]} != cond_dim {cond_dim}")
    if scalers.c_scaler is not None:
        cond_np = scalers.c_scaler.transform(cond_np)
    cond_t = torch.from_numpy(cond_np).float().to(device)
    cond_t = cond_t.repeat(n, 1)

    z = torch.randn(n, embedding_dim, device=device)
    rec, _ = decoder(z, cond_t)
    x_np = torch.tanh(rec).cpu().numpy()

    if scalers.x_scaler is not None:
        x_np = scalers.x_scaler.inverse_transform(x_np)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])
        writer.writerows(x_np.tolist())
    print(f"Saved {n} samples to {out_path} (cond={cond_list})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=os.path.join(_ROOT, "outputs", "cvae", "cvae.pt"),
    )
    parser.add_argument(
        "--scalers",
        default=os.path.join(_ROOT, "outputs", "cvae", "scalers.pkl"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_ROOT, "outputs", "cvae", "samples.csv"),
    )
    parser.add_argument(
        "--cond",
        nargs=2,
        type=float,
        default=[0.05, 10],
        metavar=("pdr_mean", "N"),
        help="조건 [pdr_mean, N] 필수. 예: --cond 0.05 10",
    )
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    cond = args.cond
    sample_cvae(
        ckpt_path=args.ckpt,
        scalers_path=args.scalers,
        out_path=args.out,
        cond=cond,
        n=args.n,
    )

"""Unified single/range sampler for all conditional benchmark models."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from benchmarks.common import scalers_from_checkpoint, set_seed, write_generated_csv
from benchmarks.models import ConditionalGenerator, ConditionalVAE, DeterministicMLP, MixtureDensityNetwork
from benchmarks.train import MODEL_NAMES


def _load_model(ckpt: dict, device: torch.device):
    model_type = ckpt["model_type"]
    config = ckpt["model_config"]
    if model_type == "mlp":
        model = DeterministicMLP(**config)
    elif model_type == "cvae":
        model = ConditionalVAE(**config)
    elif model_type == "cgan":
        model = ConditionalGenerator(**config)
    elif model_type == "mdn":
        model = MixtureDensityNetwork(**config)
    else:
        raise ValueError(f"지원하지 않는 model_type: {model_type}")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model_type, model


def _pdr_values(args, rng: np.random.Generator) -> np.ndarray:
    if args.uniform is not None:
        low, high = args.uniform
        if low >= high:
            raise ValueError("--uniform low는 high보다 작아야 합니다.")
        return rng.uniform(low, high, args.sample_num).astype(np.float32)
    if args.normal is not None:
        mean, std = args.normal
        if std < 0:
            raise ValueError("--normal std는 0 이상이어야 합니다.")
        return rng.normal(mean, std, args.sample_num).astype(np.float32)
    return np.full(args.sample_num, args.pdr, dtype=np.float32)


@torch.no_grad()
def sample(args) -> None:
    if args.sample_num <= 0 or args.N <= 0:
        raise ValueError("sample_num과 N은 1 이상이어야 합니다.")
    if args.temperature <= 0:
        raise ValueError("temperature는 0보다 커야 합니다.")
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    ckpt = torch.load(args.ckpt, map_location=device)
    model_type, model = _load_model(ckpt, device)
    if args.model is not None and args.model != model_type:
        raise ValueError(f"CLI model={args.model}, checkpoint model={model_type}")
    x_scaler, c_scaler = scalers_from_checkpoint(ckpt)

    pdr = _pdr_values(args, rng)
    raw_condition = np.column_stack([pdr, np.full(args.sample_num, args.N, dtype=np.float32)])
    condition = torch.from_numpy(c_scaler.transform(raw_condition)).to(device)

    if model_type == "mlp":
        generated = model(condition)
    elif model_type == "cvae":
        z = torch.randn(args.sample_num, model.latent_dim, device=device) * args.temperature
        generated = model.decode(z, condition)
    elif model_type == "cgan":
        noise = torch.randn(args.sample_num, model.noise_dim, device=device) * args.temperature
        generated = model(noise, condition)
    else:
        generated = model.sample(condition, temperature=args.temperature)

    coordinates = x_scaler.inverse_transform(generated.cpu().numpy())
    write_generated_csv(args.out, coordinates, pdr, args.N)
    print(f"saved={args.out} rows={len(coordinates)} model={model_type} condition=[pdr_mean,N]")


def build_parser(default_model: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MCI benchmark sampler")
    if default_model is None:
        parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    else:
        parser.set_defaults(model=default_model)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample_num", "--n", type=int, default=50)
    parser.add_argument("--pdr", type=float, default=0.08)
    parser.add_argument("--normal", nargs=2, type=float, default=None, metavar=("MEAN", "STD"))
    parser.add_argument("--uniform", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(default_model: Optional[str] = None) -> None:
    sample(build_parser(default_model).parse_args())


if __name__ == "__main__":
    main()

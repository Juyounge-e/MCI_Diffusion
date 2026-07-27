"""Shared data, scaling, q-bin, and CSV helpers for benchmark models."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


X_COLS = ["lat", "lon"]
COND_COLS = ["pdr_mean", "N"]
OUTPUT_COLS = ["lat", "lon", "N", "pdr"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ArrayScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayScaler":
        mean = values.mean(axis=0).astype(np.float32)
        scale = values.std(axis=0).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.scale + self.mean).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "scale": self.scale}

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "ArrayScaler":
        return cls(np.asarray(state["mean"], dtype=np.float32), np.asarray(state["scale"], dtype=np.float32))


@dataclass
class BenchmarkData:
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    x_scaler: ArrayScaler
    c_scaler: ArrayScaler
    n_train: int
    n_val: int


def load_benchmark_data(
    csv_path: str,
    batch_size: int,
    val_ratio: float,
    seed: int,
    max_train: int = 0,
) -> BenchmarkData:
    df = pd.read_csv(csv_path, usecols=X_COLS + COND_COLS).dropna()
    x = df[X_COLS].to_numpy(dtype=np.float32)
    c = df[COND_COLS].to_numpy(dtype=np.float32)
    if len(x) < 2:
        raise ValueError("학습 가능한 행이 2개 미만입니다.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    n_val = int(round(len(x) * val_ratio)) if val_ratio > 0 else 0
    if n_val >= len(x):
        raise ValueError("val_ratio가 너무 커서 학습 데이터가 남지 않습니다.")
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    if max_train > 0 and len(train_idx) > max_train:
        train_idx = train_idx[:max_train]

    x_train, c_train = x[train_idx], c[train_idx]
    x_scaler = ArrayScaler.fit(x_train)
    c_scaler = ArrayScaler.fit(c_train)
    x_train = x_scaler.transform(x_train)
    c_train = c_scaler.transform(c_train)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(c_train))
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )

    val_loader = None
    if n_val:
        x_val = x_scaler.transform(x[val_idx])
        c_val = c_scaler.transform(c[val_idx])
        val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(c_val))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return BenchmarkData(train_loader, val_loader, x_scaler, c_scaler, len(train_idx), n_val)


def scaler_bundle(x_scaler: ArrayScaler, c_scaler: ArrayScaler) -> Dict[str, Dict[str, np.ndarray]]:
    return {"x": x_scaler.state_dict(), "condition": c_scaler.state_dict()}


def scalers_from_checkpoint(ckpt: dict) -> Tuple[ArrayScaler, ArrayScaler]:
    states = ckpt["scalers"]
    return ArrayScaler.from_state_dict(states["x"]), ArrayScaler.from_state_dict(states["condition"])


def write_generated_csv(
    out_path: str,
    coordinates: np.ndarray,
    pdr_values: np.ndarray,
    n_value: int,
) -> None:
    if len(coordinates) != len(pdr_values):
        raise ValueError("좌표와 PDR 행 수가 다릅니다.")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    frame = pd.DataFrame({
        "lat": coordinates[:, 0],
        "lon": coordinates[:, 1],
        "N": int(n_value),
        "pdr": pdr_values,
    })
    frame.to_csv(out_path, index=False, columns=OUTPUT_COLS, float_format="%.8f")


def build_qbin_stats(
    source_csv: str,
    total_samples: int,
    num_bins: int,
    n_min: Optional[int],
    n_max: Optional[int],
) -> pd.DataFrame:
    df = pd.read_csv(source_csv, usecols=["pdr_mean", "N"]).dropna()
    _, edges = pd.qcut(df["pdr_mean"], q=num_bins, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=float)
    cut_edges = edges.copy()
    cut_edges[0], cut_edges[-1] = -np.inf, np.inf

    filtered = df
    if n_min is not None:
        filtered = filtered[filtered["N"] >= n_min]
    if n_max is not None:
        filtered = filtered[filtered["N"] <= n_max]
    if filtered.empty:
        raise ValueError(f"N 범위에 데이터가 없습니다: n_min={n_min}, n_max={n_max}")

    labels = [f"q{i + 1}" for i in range(len(edges) - 1)]
    work = filtered.copy()
    work["q_bin"] = pd.cut(work["pdr_mean"], bins=cut_edges, labels=labels, include_lowest=True, right=True)
    stats = work.groupby("q_bin", observed=False).agg(
        count=("pdr_mean", "count"),
        pdr_mean=("pdr_mean", "mean"),
        pdr_std=("pdr_mean", "std"),
        pdr_min=("pdr_mean", "min"),
        pdr_max=("pdr_mean", "max"),
    ).reset_index()
    stats["bin_min"] = edges[:-1]
    stats["bin_max"] = edges[1:]
    stats["ratio"] = stats["count"] / stats["count"].sum()

    # Largest-remainder allocation keeps the requested total exact.
    raw = stats["ratio"].to_numpy(float) * int(total_samples)
    allocated = np.floor(raw).astype(int)
    remainder = int(total_samples) - int(allocated.sum())
    if remainder > 0:
        order = np.argsort(-(raw - allocated))
        allocated[order[:remainder]] += 1
    stats["sample_num"] = allocated
    stats["pdr_std"] = stats["pdr_std"].fillna(0.0)
    return stats

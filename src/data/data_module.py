# src/data/data_module.py
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Scalers:
    x_scaler: StandardScaler
    c_scaler: Optional[StandardScaler] = None


class LatLonCondDataset(Dataset):
    def __init__(self, x: np.ndarray, c: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.c = torch.from_numpy(c).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # diffusion 학습용: x (target), c (condition)
        return self.x[idx], self.c[idx]


def load_csv(
    csv_path: str,
    x_cols: List[str],
    c_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    use_cols = x_cols + c_cols
    df = df[use_cols].dropna()

    x = df[x_cols].to_numpy(dtype=np.float32)
    c = df[c_cols].to_numpy(dtype=np.float32)

    return x, c


def make_splits(
    x: np.ndarray,
    c: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    # 1) train vs temp
    x_train, x_temp, c_train, c_temp = train_test_split(
        x, c, test_size=(val_ratio + test_ratio), random_state=seed
    )

    # 2) val vs test
    if test_ratio == 0:
        x_val, c_val = x_temp, c_temp
        x_test, c_test = np.zeros((0, x.shape[1]), np.float32), np.zeros((0, c.shape[1]), np.float32)
    else:
        test_size_in_temp = test_ratio / (val_ratio + test_ratio)
        x_val, x_test, c_val, c_test = train_test_split(
            x_temp, c_temp, test_size=test_size_in_temp, random_state=seed
        )

    return (x_train, c_train), (x_val, c_val), (x_test, c_test)


def fit_transform_scalers(
    train: Tuple[np.ndarray, np.ndarray],
    val: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
    scale_condition: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Scalers]:

    (x_train, c_train) = train
    (x_val, c_val) = val
    (x_test, c_test) = test

    x_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train)
    x_val_s = x_scaler.transform(x_val)
    x_test_s = x_scaler.transform(x_test) if len(x_test) else x_test

    c_scaler = None
    if scale_condition:
        c_scaler = StandardScaler()
        c_train_s = c_scaler.fit_transform(c_train)
        c_val_s = c_scaler.transform(c_val)
        c_test_s = c_scaler.transform(c_test) if len(c_test) else c_test
    else:
        c_train_s, c_val_s, c_test_s = c_train, c_val, c_test

    return (x_train_s, c_train_s), (x_val_s, c_val_s), (x_test_s, c_test_s), Scalers(x_scaler=x_scaler, c_scaler=c_scaler)


def make_loaders(
    train: Tuple[np.ndarray, np.ndarray],
    val: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 128,
    num_workers: int = 0,
):
    train_ds = LatLonCondDataset(*train)
    val_ds = LatLonCondDataset(*val)
    test_ds = LatLonCondDataset(*test) if len(test[0]) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = None if test_ds is None else DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    csv_path = os.path.join("src", "data", "dataset.csv")
    x_cols = ["lat", "lon"]
    c_cols = ["pdr_mean"]  # or ["pdr_mean", "pdr_std"]

    x, c = load_csv(csv_path, x_cols, c_cols)
    train, val, test = make_splits(x, c, val_ratio=0.1, test_ratio=0.1, seed=42)
    train_s, val_s, test_s, scalers = fit_transform_scalers(train, val, test, scale_condition=True)

    train_loader, val_loader, test_loader = make_loaders(train_s, val_s, test_s, batch_size=128)

    xb, cb = next(iter(train_loader))
    print("x batch:", xb.shape, xb.mean().item(), xb.std().item())
    print("c batch:", cb.shape, cb.mean().item(), cb.std().item())  
"""
CGAN용 데이터셋 정제 유틸리티 (tab-ddpm CTGAN 모듈 사용)

입력 CSV 예상 칼럼: lat, lon, pdr_mean, pdr_std, N, source_folder, stat_file
- X (생성 대상): lat, lon
- Condition: pdr_mean, N (사고규모)
- 제외: source_folder, stat_file (메타데이터), pdr_std (조건으로는 pdr_mean 사용)
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------- 칼럼 정의 ----------
X_COLS = ["lat", "lon"]
COND_COLS = ["pdr_mean", "N"]
META_COLS = ["source_folder", "stat_file"]
ALL_EXPECTED = X_COLS + ["pdr_mean", "pdr_std", "N"] + META_COLS


@dataclass
class RefinedDataset:
    """정제된 데이터셋"""
    df: pd.DataFrame
    x_cols: List[str]
    cond_cols: List[str]
    discrete_cols: List[str]
    has_N: bool


def load_refined_dataset(
    csv_path: str,
    x_cols: Optional[List[str]] = None,
    cond_cols: Optional[List[str]] = None,
    meta_cols: Optional[List[str]] = None,
    drop_meta: bool = True,
) -> pd.DataFrame:
    """
    CSV 로드 후 학습에 필요한 칼럼만 남김.
    """
    x_cols = x_cols or X_COLS
    cond_cols = cond_cols or COND_COLS.copy()
    meta_cols = meta_cols or META_COLS

    df = pd.read_csv(csv_path)

    if "N" not in df.columns:
        cond_cols = [c for c in cond_cols if c != "N"]

    use_cols = x_cols + [c for c in cond_cols if c in df.columns]
    if drop_meta:
        use_cols = [c for c in use_cols if c not in meta_cols]

    df = df[use_cols].dropna()
    return df


def discretize_continuous(
    values: np.ndarray,
    n_bins: int = 3,
    prefix: str = "bin",
) -> np.ndarray:
    """연속값을 구간별 레이블로 변환."""
    if len(values) == 0:
        return values
    try:
        bins = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    except Exception:
        bins = np.linspace(values.min(), values.max(), n_bins + 1)
    bins[-1] += 1e-6
    idx = np.digitize(values, bins[1:-1])
    labels = [f"{prefix}_{i}" for i in range(n_bins)]
    return np.array([labels[min(i, n_bins - 1)] for i in idx])


def add_discrete_condition_columns(
    df: pd.DataFrame,
    pdr_col: str = "pdr_mean",
    n_col: Optional[str] = None,
    pdr_n_bins: int = 3,
    n_n_bins: int = 3,
    n_max_unique_for_discrete: int = 10,
) -> pd.DataFrame:
    """조건 칼럼을 discrete 칼럼으로 변환 (CTGAN 입력용)."""
    df = df.copy()

    if pdr_col in df.columns:
        df["pdr_bin"] = discretize_continuous(
            df[pdr_col].values, n_bins=pdr_n_bins, prefix="pdr"
        )

    if n_col and n_col in df.columns:
        n_unique = df[n_col].nunique()
        if n_unique <= n_max_unique_for_discrete:
            df["N_bin"] = df[n_col].astype(str)
        else:
            df["N_bin"] = discretize_continuous(
                df[n_col].values.astype(float), n_bins=n_n_bins, prefix="N"
            )

    return df


def prepare_cgan_dataset(
    csv_path: str,
    x_cols: Optional[List[str]] = None,
    cond_cols: Optional[List[str]] = None,
    pdr_n_bins: int = 3,
    n_n_bins: int = 3,
) -> RefinedDataset:
    """
    CGAN 학습용 데이터셋 준비 (CTGAN 모듈 입력 형식).
    """
    x_cols = x_cols or X_COLS
    cond_cols = cond_cols or COND_COLS.copy()

    df = load_refined_dataset(csv_path, x_cols=x_cols, cond_cols=cond_cols)
    has_N = "N" in df.columns

    df = add_discrete_condition_columns(
        df,
        pdr_col="pdr_mean",
        n_col="N" if has_N else None,
        pdr_n_bins=pdr_n_bins,
        n_n_bins=n_n_bins,
    )

    discrete_cols = ["pdr_bin"]
    if has_N:
        discrete_cols.append("N_bin")

    use_combined_cond = has_N
    if use_combined_cond:
        df["cond"] = df["pdr_bin"].astype(str) + "_" + df["N_bin"].astype(str)
        discrete_cols = ["cond"]

    final_cols = x_cols + discrete_cols
    df = df[final_cols].copy()

    return RefinedDataset(
        df=df,
        x_cols=x_cols,
        cond_cols=cond_cols,
        discrete_cols=discrete_cols,
        has_N=has_N,
    )

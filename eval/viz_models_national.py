# -*- coding: utf-8 -*-
"""Compare Diffusion, MLP, CVAE, cGAN and MDN samples on one national map.

This reuses the boundary, land-mask and road-distance logic from
``eval/viz_roadloss_national.py``.  Every panel uses the same q-bin edges and
the same discrete colour scale.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_M = 6_371_000.0


def project_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def natural_q_key(path: Path):
    match = re.search(r"q(\d+)", path.stem, flags=re.IGNORECASE)
    return (int(match.group(1)) if match else 10**9, path.name)


def load_samples(folder: Path) -> pd.DataFrame:
    all_bins = folder / "all_bins.csv"
    if all_bins.exists():
        df = pd.read_csv(all_bins)
    else:
        files = sorted(folder.glob("q*.csv"), key=natural_q_key)
        if not files:
            raise FileNotFoundError(f"No all_bins.csv or q*.csv in: {folder}")
        df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)

    required = {"lat", "lon", "pdr"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{folder}: missing columns {sorted(missing)}")
    df = df.dropna(subset=["lat", "lon", "pdr"]).copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def load_qbin_edges(path: Path) -> tuple[np.ndarray, list[str]]:
    stats = pd.read_csv(path)
    if not {"bin_min", "bin_max"}.issubset(stats.columns):
        raise ValueError(f"{path} must contain bin_min and bin_max")
    stats = stats.sort_values("q_bin", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int))
    edges = np.r_[stats["bin_min"].iloc[0], stats["bin_max"].to_numpy(float)]
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"q-bin edges are not strictly increasing: {edges}")
    labels = stats["q_bin"].astype(str).tolist()
    return edges.astype(float), labels


def q_indices(pdr: pd.Series, edges: np.ndarray) -> np.ndarray:
    cut_edges = edges.copy()
    cut_edges[0], cut_edges[-1] = -np.inf, np.inf
    values = pd.cut(pdr, bins=cut_edges, labels=False, include_lowest=True)
    return values.to_numpy(dtype=int)


def load_land_mask(path: Path):
    data = np.load(path)
    return (
        data["mask"].astype(bool),
        float(data["lat_min"]),
        float(data["lon_min"]),
        float(data["resolution"]),
    )


def is_onland(lat, lon, mask, lat_min, lon_min, resolution):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_idx = np.clip(((lat - lat_min) / resolution).astype(int), 0, mask.shape[0] - 1)
    lon_idx = np.clip(((lon - lon_min) / resolution).astype(int), 0, mask.shape[1] - 1)
    return mask[lat_idx, lon_idx]


class RoadDistance:
    def __init__(self, path: Path):
        points = np.load(path)["points"].astype(np.float64)
        self.points = points
        self.cos0 = float(np.cos(np.radians(np.mean(points[:, 0]))))
        xy = np.column_stack([points[:, 0], points[:, 1] * self.cos0])
        self.tree = cKDTree(xy)

    def __call__(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        query = np.column_stack([lat, lon * self.cos0])
        _, idx = self.tree.query(query, k=1)
        road_lat = self.points[idx, 0]
        road_lon = self.points[idx, 1]
        dlat = np.radians(road_lat - lat)
        dlon = np.radians(road_lon - lon)
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat))
            * np.cos(np.radians(road_lat))
            * np.sin(dlon / 2) ** 2
        )
        return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def draw_boundary(ax, boundary):
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.65, zorder=1)
    ax.set_xlim(124.5, 130.9)
    ax.set_ylim(33.0, 38.8)
    ax.set_aspect(1.0 / np.cos(np.radians(36.0)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.15, linewidth=0.4)


def add_shared_colorbar(fig, axes, cmap, norm, labels):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=list(axes),
        ticks=np.arange(len(labels)),
        fraction=0.026,
        pad=0.02,
    )
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("PDR quantile bin (common scale)")


def save_figure(fig, axes, out_path, title, cmap, norm, labels):
    for ax in axes:
        if ax.get_visible():
            draw_boundary(ax, BOUNDARY)
    add_shared_colorbar(fig, [ax for ax in axes if ax.get_visible()], cmap, norm, labels)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.91, bottom=0.06, top=0.90, wspace=0.16, hspace=0.22)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffusion_dir",
        default="outputs/mlp_diffusion/resolution_national_1k_lam0_mint30_200k",
    )
    parser.add_argument("--benchmark_root", default="outputs/benchmarks")
    parser.add_argument("--benchmark_subdir", default="qbin_n30_1k")
    parser.add_argument(
        "--bin_stats",
        default="outputs/benchmarks/mlp/qbin_n30_1k/_bin_stats.csv",
    )
    parser.add_argument("--shp", default="MCI_ADV2/scenarios/ctprvn.shp")
    parser.add_argument("--mask", default="MCI_ADV2/scenarios/mask_cache_0.005.npz")
    parser.add_argument("--road_points", default="MCI_ADV2/scenarios/road_points_national.npz")
    parser.add_argument("--offroad_threshold_m", type=float, default=250.0)
    parser.add_argument("--out_dir", default="eval/roadloss_viz/model_compare_n30_1k")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = [
        ("Diffusion", project_path(args.diffusion_dir)),
        ("MLP", project_path(args.benchmark_root) / "mlp" / args.benchmark_subdir),
        ("CVAE", project_path(args.benchmark_root) / "cvae" / args.benchmark_subdir),
        ("cGAN", project_path(args.benchmark_root) / "cgan" / args.benchmark_subdir),
        ("MDN", project_path(args.benchmark_root) / "mdn" / args.benchmark_subdir),
    ]

    edges, q_labels = load_qbin_edges(project_path(args.bin_stats))
    nq = len(q_labels)
    cmap = ListedColormap(plt.cm.coolwarm(np.linspace(0.0, 1.0, nq)))
    norm = BoundaryNorm(np.arange(-0.5, nq + 0.5), cmap.N)

    # Preserve the CRS handling of viz_roadloss_national.py and include Jeju.
    BOUNDARY = (
        gpd.read_file(project_path(args.shp))
        .set_crs(epsg=5179, allow_override=True)
        .to_crs(epsg=4326)
    )
    land_mask = load_land_mask(project_path(args.mask))
    road_distance = RoadDistance(project_path(args.road_points))

    rows = []
    model_data = []
    for model, folder in model_dirs:
        df = load_samples(folder)
        lat = df["lat"].to_numpy(float)
        lon = df["lon"].to_numpy(float)
        q_idx = q_indices(df["pdr"], edges)
        onland = is_onland(lat, lon, *land_mask)
        distance_m = road_distance(lat, lon)
        offroad = distance_m > args.offroad_threshold_m
        model_data.append((model, df, q_idx, onland, distance_m, offroad))
        rows.append(
            {
                "model": model,
                "n": len(df),
                "offland_n": int((~onland).sum()),
                "offland_pct": float((~onland).mean() * 100.0),
                "offroad_n": int(offroad.sum()),
                "offroad_pct": float(offroad.mean() * 100.0),
                "road_dist_mean_m": float(distance_m.mean()),
                "road_dist_median_m": float(np.median(distance_m)),
                "source": str(folder),
            }
        )
        print(f"[loaded] {model}: n={len(df):,} from {folder}")

    # Figure 1: raw samples, common q-bin colour scale.
    fig, grid = plt.subplots(2, 3, figsize=(17, 14), squeeze=False)
    axes = grid.ravel()
    for ax, (model, df, q_idx, onland, distance_m, offroad) in zip(axes, model_data):
        ax.scatter(
            df["lon"], df["lat"], c=q_idx, cmap=cmap, norm=norm,
            s=8, alpha=0.72, marker="o", linewidths=0, zorder=2,
        )
        ax.set_title(f"{model} (n={len(df):,})", fontsize=12, fontweight="bold")
    axes[-1].set_visible(False)
    save_figure(
        fig, axes, out_dir / "models_national_qbin.png",
        "Five-model national samples — identical PDR q-bin colours",
        cmap, norm, q_labels,
    )

    # Figure 2: off-land samples highlighted in red.
    fig, grid = plt.subplots(2, 3, figsize=(17, 14), squeeze=False)
    axes = grid.ravel()
    for ax, (model, df, q_idx, onland, distance_m, offroad) in zip(axes, model_data):
        ax.scatter(
            df.loc[onland, "lon"], df.loc[onland, "lat"], c=q_idx[onland],
            cmap=cmap, norm=norm, s=8, alpha=0.72, linewidths=0, zorder=2,
        )
        ax.scatter(
            df.loc[~onland, "lon"], df.loc[~onland, "lat"],
            c="red", s=18, alpha=0.85, marker="x", zorder=3,
        )
        ax.set_title(
            f"{model} (n={len(df):,})\noff-land: {(~onland).sum():,} ({(~onland).mean()*100:.2f}%)",
            fontsize=12, fontweight="bold",
        )
    axes[-1].set_visible(False)
    save_figure(
        fig, axes, out_dir / "models_national_offland.png",
        "Five-model national samples — off-land points marked red",
        cmap, norm, q_labels,
    )

    # Figure 3: points farther than the common road-snap threshold highlighted.
    fig, grid = plt.subplots(2, 3, figsize=(17, 14), squeeze=False)
    axes = grid.ravel()
    for ax, (model, df, q_idx, onland, distance_m, offroad) in zip(axes, model_data):
        onroad = ~offroad
        ax.scatter(
            df.loc[onroad, "lon"], df.loc[onroad, "lat"], c=q_idx[onroad],
            cmap=cmap, norm=norm, s=8, alpha=0.72, linewidths=0, zorder=2,
        )
        ax.scatter(
            df.loc[offroad, "lon"], df.loc[offroad, "lat"],
            c="red", s=18, alpha=0.85, marker="x", zorder=3,
        )
        ax.set_title(
            f"{model} (n={len(df):,})\noff-road: {offroad.mean()*100:.2f}% | mean d={distance_m.mean():.1f} m",
            fontsize=12, fontweight="bold",
        )
    axes[-1].set_visible(False)
    save_figure(
        fig, axes, out_dir / "models_national_offroad.png",
        f"Five-model national samples — off-road > {args.offroad_threshold_m:g} m marked red",
        cmap, norm, q_labels,
    )

    summary_path = out_dir / "models_national_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

# -*- coding: utf-8 -*-
"""
road_loss(lam) 효과 시각화: 스냅 전 '원본 생성 좌표'를 전국 지도에 찍어
lam1 모델이 바다/섬 부근에서 덜 샘플링하는지 확인한다.

off-land 판정: src/model/spatial_embedding.py 의 SpatialValidityEmbedding._lookup
            과 동일한 격자 마스크(육지=1/바다=0) 조회를 numpy 로 벡터화.
            (원본 CSV 의 lat/lon 은 정규화 전 geographic 좌표이므로 스케일러 역변환 불필요)

- PRED: resolution_national_lam{0,1}_200k_{25k,100k,500k}/all_bins.csv (원본 좌표)
- lam0 vs lam1 을 동일 샘플수끼리 나란히 비교
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.spatial import cKDTree

ROOT = r"C:\Users\user00\Desktop\MCI_Diffusion"
SHP = os.path.join(ROOT, "MCI_ADV2", "scenarios", "ctprvn.shp")
MASK = os.path.join(ROOT, "MCI_ADV2", "scenarios", "mask_cache_0.005.npz")  # 0.0005° 고해상도
ROAD = os.path.join(ROOT, "MCI_ADV2", "scenarios", "road_points_national.npz")  # road_loss 와 동일 도로점
ONROAD_THRESH_M = 250.0   # OSRM 스냅 반경과 동일: 초과 시 off-road
SUMMARY = os.path.join(ROOT, "notebooks", "outputs", "analysis", "national_all_summary.csv")
OUT_DIR = os.path.join(ROOT, "eval","roadloss_viz")
os.makedirs(OUT_DIR, exist_ok=True)

MLP_DIR = os.path.join(ROOT, "outputs", "mlp_diffusion")

# 확인할 실험: (라벨, 폴더명)
EXPERIMENTS = {
    "25k":  [("lam0", "resolution_national_lam0_200k_25k"),
             ("lam1", "resolution_national_lam1_200k_25k"),
             ("lam10", "resolution_national_lam10_200k_25k")],
    "100k": [("lam0", "resolution_national_lam0_200k_100k"),
             ("lam1", "resolution_national_lam1_200k_100k"),
             ("lam10", "resolution_national_lam10_200k_100k")],
    "500k": [("lam0", "resolution_national_lam0_200k_500k"),
             ("lam1", "resolution_national_lam1_200k_500k")],  
}

# ── 격자 마스크 로드 (육지=1 / 바다=0) ──
_m = np.load(MASK)
MASK_ARR = _m["mask"].astype(bool)          # (n_lat, n_lon)
LAT_MIN = float(_m["lat_min"])
LON_MIN = float(_m["lon_min"])
RES = float(_m["resolution"])
print(f"[mask] shape={MASK_ARR.shape} land_ratio={MASK_ARR.mean():.3f} res={RES}")


def is_onland(lat, lon):
    """spatial_embedding._lookup 과 동일한 격자 조회 (벡터화). 반환: bool 배열 (육지=True)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lat_idx = np.clip(((lat - LAT_MIN) / RES).astype(int), 0, MASK_ARR.shape[0] - 1)
    lon_idx = np.clip(((lon - LON_MIN) / RES).astype(int), 0, MASK_ARR.shape[1] - 1)
    return MASK_ARR[lat_idx, lon_idx]


# ── 도로점 KDTree (road_loss 와 동일한 road_points_national.npz) ──
_rp = np.load(ROAD)["points"].astype(np.float64)     # (M,2) = lat, lon
LAT0 = float(np.mean(_rp[:, 0]))                      # 등거리 근사 기준 위도
COS0 = np.cos(np.radians(LAT0))
# 경도를 cos(위도)로 스케일 → degree 공간을 등방성으로 만들어 최근접이 미터 최근접과 일치
_road_xy = np.column_stack([_rp[:, 0], _rp[:, 1] * COS0])
ROAD_TREE = cKDTree(_road_xy)
_ROAD_LATLON = _rp
print(f"[road] points={len(_rp):,} KDTree 구성 완료 (lat0={LAT0:.3f})")

_R_EARTH = 6371000.0


def offroad_dist_m(lat, lon):
    """각 좌표의 최근접 도로점까지 거리(m). road_loss 가 당기는 그 거리와 동일 기하."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    q = np.column_stack([lat, lon * COS0])
    _, idx = ROAD_TREE.query(q, k=1)
    rlat = _ROAD_LATLON[idx, 0]
    rlon = _ROAD_LATLON[idx, 1]
    # haversine (정확한 미터)
    dlat = np.radians(rlat - lat)
    dlon = np.radians(rlon - lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat)) * np.cos(np.radians(rlat)) * np.sin(dlon / 2) ** 2)
    return 2 * _R_EARTH * np.arcsin(np.sqrt(a))


def load_raw(folder):
    """all_bins.csv 우선, 없으면 q*.csv 합쳐서 원본 lat/lon/pdr 반환."""
    d = os.path.join(MLP_DIR, folder)
    ab = os.path.join(d, "all_bins.csv")
    if os.path.exists(ab):
        return pd.read_csv(ab)
    parts = [pd.read_csv(p) for p in sorted(Path(d).glob("q*.csv"))]
    return pd.concat(parts, ignore_index=True)


# ── 경계선 (제주 제외) ──
gdf = gpd.read_file(SHP).set_crs(epsg=5179, allow_override=True).to_crs(epsg=4326)
gdf = gdf[gdf["CTP_ENG_NM"].str.lower() != "jeju-do"]


def overlay_boundary(ax):
    gdf.boundary.plot(ax=ax, color="black", linewidth=0.8, linestyle="--", zorder=1)


# ── pdr bin 경계 (색상용) ──
summary_df = pd.read_csv(SUMMARY)


def parse_interval(s):
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    return float(nums[0]), float(nums[1])

intervals = summary_df["pdr_q"].apply(parse_interval)
right_edges = [x[1] for x in intervals]
bins = np.unique(np.array([intervals.iloc[0][0]] + right_edges, dtype=float))
bins.sort()
bins[0], bins[-1] = -np.inf, np.inf
nq = len(bins) - 1
cmap = plt.cm.get_cmap("coolwarm")
norm = plt.Normalize(vmin=0, vmax=nq - 1)


results = []
for sample_key, exps in EXPERIMENTS.items():
    # 두 지도: (1) off-land, (2) off-road
    fig_land, ax_land = plt.subplots(1, len(exps), figsize=(9 * len(exps), 12), squeeze=False)
    fig_road, ax_road = plt.subplots(1, len(exps), figsize=(9 * len(exps), 12), squeeze=False)
    ax_land, ax_road = ax_land[0], ax_road[0]

    for al, ar, (lam, folder) in zip(ax_land, ax_road, exps):
        df = load_raw(folder)
        lat = df["lat"].to_numpy(float)
        lon = df["lon"].to_numpy(float)

        # ── off-land (격자 마스크) ──
        onland = is_onland(lat, lon)
        offland = ~onland
        land_frac = float(offland.mean())

        # ── off-road (도로 KDTree) ──
        d_m = offroad_dist_m(lat, lon)
        offroad = d_m > ONROAD_THRESH_M
        road_frac = float(offroad.mean())

        results.append({
            "sample": sample_key, "lam": lam, "n": len(df),
            "offland_n": int(offland.sum()), "offland_pct": 100 * land_frac,
            "offroad_n": int(offroad.sum()), "offroad_pct": 100 * road_frac,
            "road_dist_mean_m": float(d_m.mean()),
            "road_dist_median_m": float(np.median(d_m)),
        })

        q_idx = pd.cut(df["pdr"], bins=bins, labels=False, include_lowest=True).to_numpy() \
            if "pdr" in df else None

        # (1) off-land plot
        overlay_boundary(al)
        al.scatter(lon[onland], lat[onland],
                   c=(q_idx[onland] if q_idx is not None else "steelblue"),
                   cmap=cmap, norm=norm, s=2, alpha=0.55, marker="s", zorder=2)
        al.scatter(lon[offland], lat[offland], c="red", s=7, alpha=0.75, marker="x",
                   zorder=3, label=f"off-land {int(offland.sum())} ({100*land_frac:.2f}%)")
        al.set_title(f"{lam}  {sample_key}  (N={len(df):,})\noff-land = {100*land_frac:.2f}%",
                     fontsize=12, fontweight="bold")
        al.set_xlabel("lon"); al.set_ylabel("lat")
        al.legend(loc="upper right", fontsize=9)
        al.set_xlim(124.5, 130.5); al.set_ylim(33.8, 38.7)

        # (2) off-road plot (>250m 를 빨간 X 로)
        overlay_boundary(ar)
        onroad = ~offroad
        ar.scatter(lon[onroad], lat[onroad],
                   c=(q_idx[onroad] if q_idx is not None else "steelblue"),
                   cmap=cmap, norm=norm, s=2, alpha=0.55, marker="s", zorder=2)
        ar.scatter(lon[offroad], lat[offroad], c="red", s=7, alpha=0.75, marker="x",
                   zorder=3, label=f"off-road>250m {int(offroad.sum())} ({100*road_frac:.2f}%)")
        ar.set_title(f"{lam}  {sample_key}  (N={len(df):,})\n"
                     f"off-road(>250m) = {100*road_frac:.2f}%  |  mean d = {d_m.mean():.0f}m",
                     fontsize=12, fontweight="bold")
        ar.set_xlabel("lon"); ar.set_ylabel("lat")
        ar.legend(loc="upper right", fontsize=9)
        ar.set_xlim(124.5, 130.5); ar.set_ylim(33.8, 38.7)

    fig_land.suptitle(f"road_loss effect (OFF-LAND) — raw coords pre-snap, {sample_key}",
                      fontsize=14, y=0.99)
    fig_road.suptitle(f"road_loss effect (OFF-ROAD >250m) — raw coords pre-snap, {sample_key}",
                      fontsize=14, y=0.99)
    fig_land.tight_layout(); fig_road.tight_layout()
    out_land = os.path.join(OUT_DIR, f"roadloss_offland_{sample_key}.png")
    out_road = os.path.join(OUT_DIR, f"roadloss_offroad_{sample_key}.png")
    fig_land.savefig(out_land, dpi=150, bbox_inches="tight")
    fig_road.savefig(out_road, dpi=150, bbox_inches="tight")
    plt.close(fig_land); plt.close(fig_road)
    print(f"[saved] {out_land}")
    print(f"[saved] {out_road}")

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT_DIR, "offland_offroad_summary.csv"), index=False)
print("\n===== OFF-LAND / OFF-ROAD SUMMARY =====")
print(res_df.to_string(index=False))

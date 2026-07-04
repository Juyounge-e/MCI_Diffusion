"""
(2단계 / base env) 샘플 CSV를 도로망에 스냅하고 바다/off-road 지표를 측정.
torch 불필요 — shapely/pyogrio/pyproj 만 사용.

사용 예:
    python scripts/road_snap_metrics.py \
        --samples outputs/mlp_diffusion/test_national/road_baseline_samples.csv \
        --road_cache MCI_ADV2/scenarios/road_cache_national_osm.pkl \
        --mask MCI_ADV2/scenarios/mask_cache_0.005.npz
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.road_network import RoadNetwork


def is_sea(mask_npz, lat, lon):
    d = np.load(mask_npz)
    mask = d["mask"]
    lat_min = float(d["lat_min"]); lon_min = float(d["lon_min"]); res = float(d["resolution"])
    li = np.clip(((lat - lat_min) / res).astype(int), 0, mask.shape[0] - 1)
    lj = np.clip(((lon - lon_min) / res).astype(int), 0, mask.shape[1] - 1)
    return mask[li, lj] < 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--road_cache", required=True)
    ap.add_argument("--mask", default=os.path.join("MCI_ADV2", "scenarios", "mask_cache_0.005.npz"))
    ap.add_argument("--offroad_thresh_m", type=float, default=50.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.samples)
    lat0 = df["lat"].to_numpy(); lon0 = df["lon"].to_numpy()
    n = len(df)

    rn = RoadNetwork.from_cache(args.road_cache)
    r = rn.snap(lat0, lon0)

    sea_b = is_sea(args.mask, lat0, lon0)
    sea_a = is_sea(args.mask, r.lat, r.lon)
    offroad = r.dist_m > args.offroad_thresh_m

    print("=" * 64)
    print(f"샘플 {n}개  |  도로 캐시 링크 {len(rn.geoms):,}")
    print("-" * 64)
    print(f"[바다 비율]   snap 전: {sea_b.mean()*100:6.2f}%   →   snap 후: {sea_a.mean()*100:6.2f}%")
    print(f"[off-road]    도로까지 >{args.offroad_thresh_m:.0f}m :  {offroad.mean()*100:6.2f}%  (snap 전)")
    print(f"[스냅 거리]   median={np.median(r.dist_m):7.1f}m  mean={r.dist_m.mean():7.1f}m  "
          f"p90={np.percentile(r.dist_m,90):7.1f}m  max={r.dist_m.max():7.1f}m")
    print("=" * 64)

    if args.out:
        out = pd.DataFrame({
            "lat_raw": lat0, "lon_raw": lon0,
            "lat_snap": r.lat, "lon_snap": r.lon,
            "snap_dist_m": r.dist_m,
            "sea_before": sea_b.astype(int), "sea_after": sea_a.astype(int),
        })
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()

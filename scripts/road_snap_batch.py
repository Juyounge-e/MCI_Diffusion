"""
(배치 / base|MCI env) 폴더 안 모든 샘플 CSV를 도로망에 스냅하고
바다/off-road 지표를 측정. 도로 캐시는 1회만 로드(STRtree 재사용).

각 입력 <name>.csv → <name>_snapped.csv 생성
(lat_raw, lon_raw, lat_snap, lon_snap, snap_dist_m, sea_before, sea_after)

사용 예:
    python scripts/road_snap_batch.py \
        --dir outputs/mlp_diffusion/resolution_test_national \
        --road_cache MCI_ADV2/scenarios/road_cache_national_osm.pkl \
        --mask MCI_ADV2/scenarios/mask_cache_0.005.npz
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.road_network import RoadNetwork


def is_sea(mask, lat, lon):
    li = np.clip(((lat - mask["lat_min"]) / mask["res"]).astype(int), 0, mask["arr"].shape[0] - 1)
    lj = np.clip(((lon - mask["lon_min"]) / mask["res"]).astype(int), 0, mask["arr"].shape[1] - 1)
    return mask["arr"][li, lj] < 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="스냅할 CSV들이 있는 폴더")
    ap.add_argument("--road_cache", required=True)
    ap.add_argument("--mask", default=os.path.join("MCI_ADV2", "scenarios", "mask_cache_0.005.npz"))
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--offroad_thresh_m", type=float, default=50.0)
    ap.add_argument("--suffix", default="_snapped")
    args = ap.parse_args()

    # 캐시/마스크 1회 로드
    print(f"도로 캐시 로드: {args.road_cache}")
    rn = RoadNetwork.from_cache(args.road_cache)
    print(f"  링크 {len(rn.geoms):,}  (STRtree 준비 완료)")
    d = np.load(args.mask)
    mask = {"arr": d["mask"], "lat_min": float(d["lat_min"]),
            "lon_min": float(d["lon_min"]), "res": float(d["resolution"])}

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    files = [f for f in files if not os.path.basename(f).endswith(f"{args.suffix}.csv")]
    if not files:
        print("처리할 CSV가 없습니다."); return

    print(f"\n대상 {len(files)}개 파일\n" + "=" * 78)
    print(f"{'file':<28}{'rows':>7}{'sea전':>8}{'sea후':>8}{'offroad':>9}{'스냅median':>11}")
    print("-" * 78)
    rows_summary = []
    for f in files:
        df = pd.read_csv(f)
        if "lat" not in df or "lon" not in df:
            print(f"{os.path.basename(f):<28}  [skip] lat/lon 컬럼 없음"); continue
        lat0 = df["lat"].to_numpy(); lon0 = df["lon"].to_numpy()
        r = rn.snap(lat0, lon0)
        sb = is_sea(mask, lat0, lon0); sa = is_sea(mask, r.lat, r.lon)
        off = r.dist_m > args.offroad_thresh_m

        out = os.path.join(args.dir, os.path.splitext(os.path.basename(f))[0] + args.suffix + ".csv")
        # 원본 전체 컬럼(pdr, rygb, N 등) 유지 + 스냅 좌표/지표 덧붙임
        out_df = df.copy()
        out_df["lat_snap"] = r.lat
        out_df["lon_snap"] = r.lon
        out_df["snap_dist_m"] = r.dist_m
        out_df["sea_before"] = sb.astype(int)
        out_df["sea_after"] = sa.astype(int)
        out_df.to_csv(out, index=False)

        name = os.path.basename(f)
        print(f"{name:<28}{len(df):>7}{sb.mean()*100:>7.2f}%{sa.mean()*100:>7.2f}%"
              f"{off.mean()*100:>8.2f}%{np.median(r.dist_m):>9.1f}m")
        rows_summary.append((name, len(df), sb.mean(), sa.mean(), off.mean(), np.median(r.dist_m)))

    print("=" * 78)
    print(f"완료: {len(rows_summary)}개 → 각 '{args.suffix}.csv' 저장 (폴더: {args.dir})")


if __name__ == "__main__":
    main()

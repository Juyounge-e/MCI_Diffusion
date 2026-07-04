"""
생성 좌표가 도로 위에 있는지 '측정'하는 스크립트 (교정 X, 자(ruler)로만 사용).

각 좌표 → 최근접 도로까지 거리(m)를 재고, 임계값별 on-road 비율을 보고.
좌표를 옮기지 않으므로 loss 효과(생성 자체가 도로 근처인가)를 그대로 평가할 수 있다.

사용 예:
    python scripts/offroad_metric.py \
        --samples outputs/.../samples_100.csv \
        --road_cache MCI_ADV2/scenarios/road_cache_daejeon_osm.pkl \
        --out outputs/.../samples_100_offroad.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.road_network import RoadNetwork


def main():
    ap = argparse.ArgumentParser(description="생성 좌표의 off-road 지표 측정")
    ap.add_argument("--samples", required=True, help="생성 CSV (lat, lon ...)")
    ap.add_argument("--road_cache", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[10, 30, 50, 100],
                    help="on-road 판정 임계값(m) 목록")
    ap.add_argument("--out", default=None, help="행별 거리/플래그 CSV 저장(선택)")
    args = ap.parse_args()

    df = pd.read_csv(args.samples)
    if "lat" not in df.columns or "lon" not in df.columns:
        print(f"[오류] samples 에 lat/lon 없음. 컬럼: {list(df.columns)}"); return 1

    rn = RoadNetwork.from_cache(args.road_cache)
    r = rn.snap(df["lat"].to_numpy(), df["lon"].to_numpy())   # snap = 거리 측정용
    d = r.dist_m
    n = len(d)

    print("=" * 56)
    print(f"  샘플 {n}개  |  도로 캐시 링크 {len(rn.geoms):,}")
    print(f"  도로까지 거리(m): median={np.median(d):.1f}  mean={d.mean():.1f}  "
          f"p90={np.percentile(d,90):.1f}  max={d.max():.1f}")
    print("-" * 56)
    for thr in args.thresholds:
        onroad = (d <= thr).mean() * 100
        print(f"  ≤{thr:>5.0f}m on-road : {onroad:5.1f}%   (off-road {100-onroad:4.1f}%)")
    print("=" * 56)

    if args.out:
        out = df.copy()
        out["road_dist_m"] = d
        for thr in args.thresholds:
            out[f"onroad_{int(thr)}m"] = (d <= thr).astype(int)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"행별 결과 저장: {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())

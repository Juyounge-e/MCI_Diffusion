"""
(MCI env, 1회) 도로 캐시(linestring) → 일정 간격 점 구름(point cloud) 으로 변환해
npz 로 저장. 학습 루프(tddpm, shapely 불필요)에서 road-distance loss 계산에 사용.

- 입력 캐시 geometry 는 EPSG:5186(미터) → spacing(m) 으로 segmentize 후 좌표 추출
- 출력 점은 WGS84(lat, lon) — 학습 시 x_scaler 로 정규화해서 KDTree 빌드

사용 예:
    python scripts/build_road_points.py \
        --road_cache MCI_ADV2/scenarios/road_cache_national_osm.pkl \
        --spacing_m 30 \
        --out MCI_ADV2/scenarios/road_points_national.npz
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.road_network import RoadNetwork, ROAD_CRS, WGS84


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--road_cache", required=True)
    ap.add_argument("--spacing_m", type=float, default=30.0, help="점 간격(미터)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import shapely
    from pyproj import Transformer

    print(f"[1/3] 도로 캐시 로드: {args.road_cache}")
    rn = RoadNetwork.from_cache(args.road_cache)
    print(f"      링크 {len(rn.geoms):,}")

    print(f"[2/3] {args.spacing_m}m 간격 점 추출(segmentize)")
    t0 = time.time()
    dens = shapely.segmentize(rn.geoms, max_segment_length=args.spacing_m)  # 5186(m)
    coords = shapely.get_coordinates(dens)  # (M, 2) = (x, y) in 5186
    # 중복 점 제거(인접 링크 공유 노드) — 격자 반올림 후 unique
    key = np.round(coords / (args.spacing_m * 0.5)).astype(np.int64)
    _, uniq = np.unique(key, axis=0, return_index=True)
    coords = coords[np.sort(uniq)]
    print(f"      점 {len(coords):,}개  ({time.time()-t0:.1f}s)")

    print("[3/3] 5186 → WGS84 변환 + 저장")
    t = Transformer.from_crs(ROAD_CRS, WGS84, always_xy=True)
    lon, lat = t.transform(coords[:, 0], coords[:, 1])
    points = np.column_stack([lat, lon]).astype(np.float32)  # (M, 2) = (lat, lon)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, points=points, spacing_m=np.float64(args.spacing_m), crs=WGS84)
    print(f"완료: {args.out}  점 {len(points):,}개  "
          f"(lat {points[:,0].min():.3f}~{points[:,0].max():.3f}, "
          f"lon {points[:,1].min():.3f}~{points[:,1].max():.3f})")


if __name__ == "__main__":
    main()

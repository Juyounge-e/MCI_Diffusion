"""
좌표 하나를 주면 그 근처에 어떤 way 가 있는지 알려줌
    python road_disruption/scripts/whats_here.py --region daejeon \
        --at 36.3450,127.3450 --radius 200

찾은 way_id 는 CutSpec 에 그대로 넣으면 된다:
    select: {type: way_id, ids: [123, 456]}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis.paths import DATA_DIR
from roaddis.way_index import WayIndex


def main():
    ap = argparse.ArgumentParser(description="좌표 근처의 way 조회")
    ap.add_argument("--region", required=True)
    ap.add_argument("--at", required=True, help="lat,lon")
    ap.add_argument("--radius", type=float, default=150.0, help="반경(m)")
    ap.add_argument("--bridge-only", action="store_true")
    ap.add_argument("--drive-only", action="store_true", default=True)
    args = ap.parse_args()

    lat, lon = (float(x) for x in args.at.split(","))
    idx = WayIndex.load(DATA_DIR / "way_index" / args.region)
    t = idx.tags

    tr = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
    cx, cy = tr.transform(lon, lat)
    x, y = tr.transform(idx.node_lon, idx.node_lat)
    d = np.hypot(np.asarray(x) - cx, np.asarray(y) - cy)

    # way 별 최단 거리
    way_pos = np.repeat(np.arange(len(idx)), np.diff(idx.node_off))
    best = np.full(len(idx), np.inf)
    np.minimum.at(best, way_pos, d)

    sel = best <= args.radius
    if args.drive_only:
        sel &= t.is_drive.to_numpy()
    if args.bridge_only:
        sel &= t.bridge.notna().to_numpy()

    pos = np.flatnonzero(sel)
    pos = pos[np.argsort(best[pos])]
    print(f"({lat}, {lon}) 반경 {args.radius:.0f}m 내 way {len(pos)}개"
          f"{' (교량만)' if args.bridge_only else ''}\n")
    if not len(pos):
        print("없습니다. --radius 를 키워보세요.")
        return

    print(f"{'거리m':>6} {'way_id':>12} {'idx':>6} {'highway':<15} "
          f"{'bridge':>6} {'노드':>4}  name")
    for p in pos[:40]:
        r = t.iloc[p]
        print(f"{best[p]:>6.0f} {r.way_id:>12} {p:>6} {str(r.highway):<15} "
              f"{str(r.bridge):>6} {r.n_nodes:>4}  {r['name']}")

    ids = ", ".join(str(int(t.iloc[p].way_id)) for p in pos[:40])
    print(f"\nCutSpec 용:\n  select: {{type: way_id, ids: [{ids}]}}")


if __name__ == "__main__":
    main()

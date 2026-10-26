"""
지역 pbf → WayIndex (way_id, node id 시퀀스, 태그)
./road_disruption/data/way_index에 저장됨(.npz, .parquet)

사용 예:
    python road_disruption/scripts/build_way_index.py --region daejeon
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis.osm_tags import DRIVE_HIGHWAY, KEEP_TAGS
from roaddis.paths import OSRM_DIR, data_subdir
from roaddis.regions import get_region
from roaddis.way_index import WayIndex


def main():
    ap = argparse.ArgumentParser(description="pbf → way 인덱스")
    ap.add_argument("--region", required=True)
    ap.add_argument("--pbf", default=None, help="기본: regions.yaml 의 해당 지역 pbf")
    # ap.add_argument("--locations", default="flex_mem", help="pyosmium node 위치 캐시. 전국처럼 큰 파일이면 " "'sparse_file_array,<경로>' 로 디스크 캐시 사용")
    args = ap.parse_args()

    import osmium

    region = get_region(args.region)
    pbf = Path(args.pbf) if args.pbf else OSRM_DIR / region.pbf
    if not pbf.exists():
        sys.exit(f"pbf 없음: {pbf}\n  먼저 extract_region_pbf.py 를 실행하세요.")

    print(f"[1/3] pbf 읽기: {pbf.name} ({pbf.stat().st_size/1e6:.1f}MB)")
    t0 = time.time()

    way_ids: list[int] = []
    node_ids: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    lats: list[np.ndarray] = []
    tag_rows: list[dict] = []
    n_invalid_loc = 0

    # with_locations() 는 node 를 실제로 읽어야 동작한다 — NODE 비트가 없으면
    # "Nodes not read from file" 로 죽는다. 또 KeyFilter 를 체인에 걸면 node 가
    # 함께 걸러져 위치 캐시가 비므로, highway 판별은 루프 안에서 한다.
    fp = (
        osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY)
        .with_locations(args.locations)
    )

    for w in fp:
        if not isinstance(w, osmium.osm.Way):
            continue
        hw = w.tags.get("highway")
        if hw is None:
            continue

        refs = np.fromiter((n.ref for n in w.nodes), dtype=np.int64, count=len(w.nodes))
        if len(refs) < 2:
            continue  # 세그먼트를 만들 수 없는 way

        lon = np.empty(len(refs), dtype=np.float64)
        lat = np.empty(len(refs), dtype=np.float64)
        for k, n in enumerate(w.nodes):
            if n.location.valid():
                lon[k] = n.location.lon
                lat[k] = n.location.lat
            else:
                lon[k] = np.nan
                lat[k] = np.nan
                n_invalid_loc += 1

        row = {t: w.tags.get(t) for t in KEEP_TAGS}
        row["way_id"] = w.id
        row["is_drive"] = hw in DRIVE_HIGHWAY
        row["n_nodes"] = len(refs)

        way_ids.append(w.id)
        node_ids.append(refs)
        lons.append(lon)
        lats.append(lat)
        tag_rows.append(row)

        if len(way_ids) % 200_000 == 0:
            print(f"      {len(way_ids):,} ways ... ({time.time()-t0:.0f}s)")

    if not way_ids:
        sys.exit("highway way 를 하나도 찾지 못했습니다. pbf 를 확인하세요.")

    print(f"      way {len(way_ids):,}개, node 참조 {sum(len(r) for r in node_ids):,}개 "
          f"({time.time()-t0:.1f}s)")
    if n_invalid_loc:
        print(f"      [경고] 좌표를 못 찾은 node 참조 {n_invalid_loc:,}개 (NaN 처리). "
              f"경계에 걸친 way 일 수 있습니다 — extract 시 --strategy complete_ways 확인")

    print("[2/3] CSR 배열 구성 + way_id 정렬")
    way_id = np.asarray(way_ids, dtype=np.int64)
    order = np.argsort(way_id, kind="stable")   # position_of() 의 이진탐색 전제
    way_id = way_id[order]

    lengths = np.fromiter((len(node_ids[i]) for i in order), dtype=np.int64,
                          count=len(order))
    node_off = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum(lengths, out=node_off[1:])

    node_id = np.concatenate([node_ids[i] for i in order])
    node_lon = np.concatenate([lons[i] for i in order])
    node_lat = np.concatenate([lats[i] for i in order])

    # way 별 bbox — bbox/radius selector 의 1차 필터
    bbox = np.empty((len(order), 4), dtype=np.float64)
    for i in range(len(order)):
        s, e = node_off[i], node_off[i + 1]
        wl, wt = node_lon[s:e], node_lat[s:e]
        bbox[i] = (np.nanmin(wl), np.nanmin(wt), np.nanmax(wl), np.nanmax(wt))

    tags = pd.DataFrame([tag_rows[i] for i in order])
    cols = ["way_id", "is_drive", "n_nodes"] + list(KEEP_TAGS)
    tags = tags[cols]

    idx = WayIndex(way_id=way_id, node_off=node_off, node_id=node_id,
                   node_lon=node_lon, node_lat=node_lat, bbox=bbox, tags=tags)

    stem = data_subdir("way_index") / args.region
    print(f"[3/3] 저장: {stem}_ways.npz / .parquet")
    idx.save(stem)

    drive = int(tags["is_drive"].sum())
    print(f"완료. way {len(idx):,}개 (주행도로 {drive:,} / 기타 {len(idx)-drive:,})  "
          f"총 {time.time()-t0:.1f}s")
    print()
    print("  highway 태그 상위:")
    for hw, c in tags["highway"].value_counts().head(10).items():
        print(f"    {hw:<16} {c:>8,}")


if __name__ == "__main__":
    main()

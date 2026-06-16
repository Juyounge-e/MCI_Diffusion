"""
도로망 → 스냅(snap) 캐시 생성기.

두 가지 소스 지원 (둘 다 인터넷 불필요, 로컬 파일만):
  --source moct : MOCT 국가표준노드링크 (MOCT_LINK.shp, EPSG:5186)
  --source osm  : OSM 추출 (south-korea.osm.pbf, EPSG:4326) — 시뮬레이터 OSRM과 동일 도로망

대상 지역(bbox)만 잘라 경량 캐시(pickle)로 저장.
캐시 좌표계는 항상 EPSG:5186 (미터) — RoadNetwork.snap()이 미터 거리로 동작하기 때문.

사용 예:
    # OSM (시뮬레이터와 정합, 권장)
    python scripts/build_road_cache.py --source osm \
        --src C:/Users/user00/osrm-korea/south-korea-260318.osm.pbf \
        --region daejeon --out MCI_ADV2/scenarios/road_cache_daejeon_osm.pkl

    # MOCT (공식 표준노드링크)
    python scripts/build_road_cache.py --source moct \
        --src NODELINKDATA/MOCT_LINK.shp \
        --region daejeon --out MCI_ADV2/scenarios/road_cache_daejeon_moct.pkl
"""
import argparse
import os
import pickle
import time

import numpy as np
from pyproj import Transformer

ROAD_CRS = "EPSG:5186"  # 캐시 저장 좌표계 (미터)

# 지역별 WGS84 bbox (lon_min, lat_min, lon_max, lat_max). None = 전국(필터 없음)
REGION_BBOX_WGS84 = {
    "daejeon": (127.20, 36.15, 127.60, 36.55),
    "incheon": (126.30, 37.30, 126.80, 37.70),
    "national": None,
}

# OSRM 차량(driving) 프로파일이 주행 도로로 보는 highway 태그 (footway/path/cycleway/steps/track 제외)
OSM_DRIVE_HIGHWAY = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street", "service", "road",
}


def wgs84_bbox_to_5186(bbox_wgs84):
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    t = Transformer.from_crs("EPSG:4326", ROAD_CRS, always_xy=True)
    xs, ys = [], []
    for lon in (lon_min, lon_max):
        for lat in (lat_min, lat_max):
            x, y = t.transform(lon, lat)
            xs.append(x); ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys))


def load_moct(src, bbox_wgs84):
    import pyogrio
    bbox = wgs84_bbox_to_5186(bbox_wgs84)
    print(f"      MOCT bbox(5186)={tuple(round(v) for v in bbox)}")
    gdf = pyogrio.read_dataframe(src, bbox=bbox, columns=["LINK_ID", "ROAD_RANK"])
    gdf = gdf.set_crs(ROAD_CRS, allow_override=True)  # .prj가 ESRI WKT라 명시 고정
    attrs = {"link_id": gdf["LINK_ID"].to_numpy(), "road_rank": gdf["ROAD_RANK"].to_numpy()}
    return gdf, attrs


def load_osm(src, bbox_wgs84):
    import pyogrio
    # OSM pbf는 WGS84(4326). bbox도 WGS84로.
    gdf = pyogrio.read_dataframe(src, layer="lines", bbox=bbox_wgs84, columns=["highway"])
    n0 = len(gdf)
    gdf = gdf[gdf["highway"].isin(OSM_DRIVE_HIGHWAY)].reset_index(drop=True)
    print(f"      OSM lines {n0:,} → drive 도로 {len(gdf):,}")
    gdf = gdf.to_crs(ROAD_CRS)  # 4326 → 5186 (미터)
    attrs = {"highway": gdf["highway"].to_numpy()}
    return gdf, attrs


def main():
    parser = argparse.ArgumentParser(description="도로망 → 스냅 캐시 생성")
    parser.add_argument("--source", choices=["moct", "osm"], required=True)
    parser.add_argument("--src", required=True, help="MOCT_LINK.shp 또는 *.osm.pbf 경로")
    parser.add_argument("--region", default="daejeon", choices=list(REGION_BBOX_WGS84.keys()))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import shapely

    bbox_wgs84 = REGION_BBOX_WGS84[args.region]
    print(f"[1/3] source={args.source}  region={args.region}  로드: {args.src}")
    t0 = time.time()
    if args.source == "moct":
        gdf, attrs = load_moct(args.src, bbox_wgs84)
    else:
        gdf, attrs = load_osm(args.src, bbox_wgs84)
    print(f"      로드 완료 ({time.time()-t0:.1f}s)  링크={len(gdf):,}  CRS={gdf.crs}")

    # 빈/None 지오메트리 방어
    geoms = gdf.geometry.values
    valid = np.array([g is not None and (not g.is_empty) for g in geoms])
    if not valid.all():
        print(f"      [경고] 빈 지오메트리 {int((~valid).sum())}개 제외")
        gdf = gdf[valid].reset_index(drop=True)
        attrs = {k: v[valid] for k, v in attrs.items()}

    print("[2/3] WKB 직렬화")
    cache = {
        "crs": ROAD_CRS,
        "source": args.source,
        "region": args.region,
        "wkb": list(shapely.to_wkb(gdf.geometry.values)),
        **attrs,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[3/3] 저장: {args.out}")
    with open(args.out, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"완료. 링크 {len(cache['wkb']):,}개  (총 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()

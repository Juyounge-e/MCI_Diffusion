"""
Selector — "무엇을 끊을지" 고르는 규칙들.

모든 selector 는 WayIndex 를 받아 **세그먼트 단위 불리언 마스크 (S,)** 를 낸다.
way 단위가 아닌 이유: 8km 짜리 간선도로가 침수구역을 200m 스쳤다고 통째로 끊으면
영향을 크게 부풀린다.

공간 selector 는 세그먼트의 (시작점, 중점, 끝점) 중 하나라도 영역에 들어가면
선택한다. OSM 노드는 촘촘해서 세그먼트가 대개 수십 m 이므로 이 근사로 충분하고,
정확한 선분-다각형 교차보다 훨씬 빠르다.

새 selector 를 추가하려면 함수를 만들고 REGISTRY 에 등록하면 된다. CutSpec 은
REGISTRY 만 보므로 다른 코드를 고칠 필요가 없다.
"""
from __future__ import annotations

import numpy as np

from .osm_tags import DRIVE_HIGHWAY, MAJOR_HIGHWAY

ROAD_CRS = "EPSG:5186"
WGS84 = "EPSG:4326"


# ----------------------------------------------------------------------
# 태그 기반 (way 단위로 판정 후 세그먼트로 확장)
# ----------------------------------------------------------------------
def sel_way_id(idx, p) -> np.ndarray:
    """{type: way_id, ids: [123, 456]} — 개별 OSM way 지정."""
    ids = np.asarray(p["ids"], dtype=np.int64)
    way_mask = np.isin(idx.way_id, ids)
    missing = len(ids) - int(np.isin(ids, idx.way_id).sum())
    if missing:
        raise ValueError(f"way_id {missing}개가 인덱스에 없습니다. "
                         f"지역이 맞는지, pbf 버전이 맞는지 확인하세요.")
    return idx.way_mask_to_segment_mask(way_mask)


def sel_highway_class(idx, p) -> np.ndarray:
    """
    {type: highway_class, classes: [motorway, trunk]} — highway 태그 값으로 선택.
    classes 대신 preset: drive|major 도 쓸 수 있다.
    """
    preset = p.get("preset")
    if preset == "drive":
        classes = DRIVE_HIGHWAY
    elif preset == "major":
        classes = MAJOR_HIGHWAY
    elif preset is not None:
        raise ValueError(f"알 수 없는 preset '{preset}' (drive|major)")
    else:
        classes = set(p["classes"])
    return idx.way_mask_to_segment_mask(idx.tags["highway"].isin(classes).to_numpy())


def sel_osm_tag(idx, p) -> np.ndarray:
    """
    {type: osm_tag, tags: {bridge: "*", layer: ["1", "2"]}}

    값 규칙: "*" = 태그가 존재하기만 하면 됨, 리스트 = 그 중 하나, 그 외 = 정확히 일치.
    여러 태그를 주면 AND 로 묶인다.
    """
    way_mask = np.ones(len(idx), dtype=bool)
    for key, want in p["tags"].items():
        if key not in idx.tags.columns:
            raise ValueError(
                f"태그 '{key}' 가 인덱스에 없습니다. roaddis/osm_tags.py 의 "
                f"KEEP_TAGS 에 추가하고 build_way_index.py 를 다시 실행하세요."
            )
        col = idx.tags[key]
        if want == "*":
            way_mask &= col.notna().to_numpy()
        elif isinstance(want, (list, tuple)):
            way_mask &= col.isin([str(v) for v in want]).to_numpy()
        else:
            way_mask &= (col == str(want)).to_numpy()
    return idx.way_mask_to_segment_mask(way_mask)


def sel_drive_only(idx, p) -> np.ndarray:
    """{type: drive_only} — OSRM 주행 대상 도로만. 보도/등산로 등을 제외."""
    return idx.way_mask_to_segment_mask(idx.tags["is_drive"].to_numpy())


# ----------------------------------------------------------------------
# 공간 기반 (세그먼트 단위로 직접 판정)
# ----------------------------------------------------------------------
def _seg_points(idx):
    """세그먼트당 판정용 3점 (시작, 중점, 끝)."""
    a, b = idx.segment_endpoints()
    return a, idx.segment_midpoints(), b


def sel_bbox(idx, p) -> np.ndarray:
    """{type: bbox, bbox: [lon_min, lat_min, lon_max, lat_max]}"""
    lon_min, lat_min, lon_max, lat_max = p["bbox"]
    mask = np.zeros(idx.n_segments, dtype=bool)
    for pts in _seg_points(idx):
        mask |= ((pts[:, 0] >= lon_min) & (pts[:, 0] <= lon_max) &
                 (pts[:, 1] >= lat_min) & (pts[:, 1] <= lat_max))
    return mask


def sel_radius(idx, p) -> np.ndarray:
    """
    {type: radius, center: [lon, lat], radius_m: 500}

    거리는 EPSG:5186(미터)에서 계산한다 — WGS84 도(degree) 거리는 위도에 따라
    왜곡되어 "반경 500m"가 방향마다 달라진다.
    """
    from pyproj import Transformer

    t = Transformer.from_crs(WGS84, ROAD_CRS, always_xy=True)
    cx, cy = t.transform(*p["center"])
    r2 = float(p["radius_m"]) ** 2

    mask = np.zeros(idx.n_segments, dtype=bool)
    for pts in _seg_points(idx):
        x, y = t.transform(pts[:, 0], pts[:, 1])
        mask |= ((np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2) <= r2
    return mask


def sel_polygon(idx, p) -> np.ndarray:
    """
    {type: polygon, path: "flood_zone.shp", layer: null, where: null}

    침수구역·산사태 위험구역 같은 실제 재해 영역 shapefile 로 자를 때 쓴다.
    파일 좌표계는 자동으로 WGS84 로 변환한다.
    """
    import geopandas as gpd
    import shapely

    gdf = gpd.read_file(p["path"], layer=p.get("layer"))
    if p.get("where"):
        gdf = gdf.query(p["where"])
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(WGS84)
    if gdf.empty:
        raise ValueError(f"폴리곤이 비어 있습니다: {p['path']}")

    geom = shapely.union_all(gdf.geometry.values)
    tree = shapely.STRtree([geom])

    mask = np.zeros(idx.n_segments, dtype=bool)
    for pts in _seg_points(idx):
        hits = tree.query(shapely.points(pts[:, 0], pts[:, 1]), predicate="intersects")
        # query 는 [input_idx, tree_idx] 또는 (N,) 을 반환 — 둘 다 처리
        hits = np.asarray(hits)
        inside = hits[0] if hits.ndim == 2 else hits
        mask[inside] = True
    return mask


# ----------------------------------------------------------------------
REGISTRY = {
    "way_id": sel_way_id,
    "highway_class": sel_highway_class,
    "osm_tag": sel_osm_tag,
    "drive_only": sel_drive_only,
    "bbox": sel_bbox,
    "radius": sel_radius,
    "polygon": sel_polygon,
}


def apply_selector(idx, spec: dict) -> np.ndarray:
    """단일 selector 스펙 → 세그먼트 마스크."""
    if "type" not in spec:
        raise ValueError(f"selector 에 'type' 이 없습니다: {spec}")
    t = spec["type"]
    if t not in REGISTRY:
        raise ValueError(f"알 수 없는 selector '{t}'. 사용 가능: {sorted(REGISTRY)}")
    return REGISTRY[t](idx, spec)

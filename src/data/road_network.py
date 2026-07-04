"""
도로망 스냅(snap-to-road) 유틸.

학습/샘플링에서 생성된 (lat, lon) 좌표를 가장 가까운 도로 링크로 투영한다.
- 거리 계산은 반드시 EPSG:5186 (미터) 에서 수행 (WGS84 도(degree) 거리는 왜곡됨).
- shapely 2.x STRtree 로 최근접 링크를 O(log n) 으로 질의.
- 투영 결과로 on-road 좌표 + 부산물(link_index, 링크 위 정규화 위치 s)을 함께 반환.

설계 의도:
    동일한 snap() 연산자를 (a) baseline 후처리, (b) 샘플링 per-step 투영,
    (c) 학습 aux loss 의 타깃 생성, (d) off-road 평가 에 모두 재사용.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import shapely
    from shapely import STRtree
except ImportError as e:  # pragma: no cover
    raise ImportError("shapely>=2.0 이 필요합니다 (STRtree.query_nearest 사용).") from e

from pyproj import Transformer

ROAD_CRS = "EPSG:5186"
WGS84 = "EPSG:4326"


@dataclass
class SnapResult:
    """스냅 결과 (모두 길이 N 배열)."""
    lat: np.ndarray          # 스냅된 위도 (WGS84)
    lon: np.ndarray          # 스냅된 경도 (WGS84)
    dist_m: np.ndarray       # 원점→스냅점 거리 (미터)
    link_index: np.ndarray   # 캐시 내 링크 인덱스
    s: np.ndarray            # 링크 위 정규화 위치 [0,1]


class RoadNetwork:
    """
    도로 링크 집합에 대한 최근접 투영기.

    생성:
        rn = RoadNetwork.from_cache("road_cache_daejeon.pkl")   # 권장(경량)
        rn = RoadNetwork.from_shp("NODELINKDATA/MOCT_LINK.shp", bbox_5186=(...))
    """

    def __init__(self, geoms: np.ndarray, link_id: Optional[np.ndarray] = None,
                 road_rank: Optional[np.ndarray] = None):
        # geoms: shapely LineString 의 np.ndarray(object)
        self.geoms = geoms
        self.link_id = link_id
        self.road_rank = road_rank
        self.lengths = shapely.length(geoms)            # (N,) 미터
        self.tree = STRtree(geoms)
        self._to_5186 = Transformer.from_crs(WGS84, ROAD_CRS, always_xy=True)
        self._to_wgs = Transformer.from_crs(ROAD_CRS, WGS84, always_xy=True)

    # ------------------------------------------------------------------
    @classmethod
    def from_cache(cls, cache_path: str) -> "RoadNetwork":
        with open(cache_path, "rb") as f:
            c = pickle.load(f)
        geoms = shapely.from_wkb(c["wkb"])
        return cls(
            geoms=np.asarray(geoms, dtype=object),
            link_id=c.get("link_id"),
            road_rank=c.get("road_rank"),
        )

    @classmethod
    def from_shp(cls, shp_path: str, bbox_5186=None) -> "RoadNetwork":
        import pyogrio
        gdf = pyogrio.read_dataframe(
            shp_path, bbox=bbox_5186, columns=["LINK_ID", "ROAD_RANK"]
        )
        geoms = np.asarray(gdf.geometry.values, dtype=object)
        return cls(geoms=geoms,
                   link_id=gdf["LINK_ID"].to_numpy(),
                   road_rank=gdf["ROAD_RANK"].to_numpy())

    # ------------------------------------------------------------------
    def snap(self, lat, lon) -> SnapResult:
        """
        WGS84 (lat, lon) → 최근접 도로 링크로 투영.

        Args:
            lat, lon : scalar 또는 1D array-like (WGS84)
        Returns:
            SnapResult (모든 필드 길이 N)
        """
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))

        # WGS84 → 5186 (미터)
        x, y = self._to_5186.transform(lon, lat)
        pts = shapely.points(x, y)

        # 최근접 링크 인덱스 (입력당 1개)
        # shapely 버전에 따라 (N,) 또는 (2,N)=[input_idx, tree_idx] 반환 → 둘 다 처리
        res = np.asarray(self.tree.query_nearest(pts, all_matches=False))
        if res.ndim == 2:
            order = np.argsort(res[0], kind="stable")
            nearest = res[1][order]
        else:
            nearest = res.reshape(-1)
        lines = self.geoms[nearest]

        # 링크 위로 투영: 거리(along) → 점
        d_along = shapely.line_locate_point(lines, pts)        # (N,) 미터
        snapped = shapely.line_interpolate_point(lines, d_along)
        dist_m = shapely.distance(pts, snapped)                # (N,) 미터

        sx = shapely.get_x(snapped)
        sy = shapely.get_y(snapped)
        slon, slat = self._to_wgs.transform(sx, sy)

        seg_len = self.lengths[nearest]
        s = np.divide(d_along, seg_len, out=np.zeros_like(d_along),
                      where=seg_len > 0)

        return SnapResult(
            lat=np.asarray(slat).reshape(-1),
            lon=np.asarray(slon).reshape(-1),
            dist_m=np.asarray(dist_m).reshape(-1),
            link_index=nearest,
            s=s,
        )

"""
WayIndex — OSM way 인덱스.

저장 형식:
    <region>_ways.npz      — 숫자 배열 (CSR 방식으로 가변길이 node 시퀀스 저장)
        way_id      int64   (W,)     OSM way id
        node_off    int64   (W+1,)   CSR offset. way i 의 node = node_id[off[i]:off[i+1]]
        node_id     int64   (P,)     node id (way 순서대로, 중복 포함)
        node_lon    float64 (P,)     node 경도
        node_lat    float64 (P,)     node 위도
        bbox        float64 (W,4)    way 별 (lon_min, lat_min, lon_max, lat_max)
    <region>_ways.parquet  — 문자열 태그
        way_id, highway, oneway, bridge, tunnel, maxspeed, lanes, layer,
        name, ref, access, is_drive, n_nodes)
        
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .osm_tags import ONEWAY_FORWARD, ONEWAY_REVERSE


@dataclass
class WayIndex:
    way_id: np.ndarray      # (W,) int64
    node_off: np.ndarray    # (W+1,) int64
    node_id: np.ndarray     # (P,) int64
    node_lon: np.ndarray    # (P,) float64
    node_lat: np.ndarray    # (P,) float64
    bbox: np.ndarray        # (W,4) float64
    tags: pd.DataFrame      # (W,) way_id 순서와 동일

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.way_id)

    def coords_of(self, i: int) -> np.ndarray:
        """way 인덱스 i 의 (lon, lat) 좌표 (n,2)."""
        s, e = self.node_off[i], self.node_off[i + 1]
        return np.column_stack([self.node_lon[s:e], self.node_lat[s:e]])

    # ------------------------------------------------------------------
    # 세그먼트 레이어
    #
    # way 는 차단 단위로 너무 거칠다. 길이 8km 인 간선도로가 침수구역을 200m
    # 스쳤다고 통째로 끊으면 영향을 크게 부풀린다. 그래서 selector 는 way 가
    # 아니라 "세그먼트"(인접한 두 node 사이 구간) 단위로 판정한다.
    #
    # way i 는 (n_nodes - 1) 개의 세그먼트를 가지므로 node CSR 에서 way 당
    # 1씩 빼면 세그먼트 CSR 이 된다.
    # ------------------------------------------------------------------
    @property
    def seg_off(self) -> np.ndarray:
        """(W+1,) 세그먼트 CSR offset."""
        return self.node_off - np.arange(len(self.node_off), dtype=np.int64)

    @property
    def n_segments(self) -> int:
        return int(self.seg_off[-1])

    def _segment_node_slots(self) -> tuple:
        """
        각 세그먼트의 시작/끝 node 가 node_id 배열의 몇 번째 칸인지.

        way 의 마지막 node 는 세그먼트를 시작하지 못하므로 제외한다.
        결과 순서는 seg_off 와 일치한다.
        """
        keep = np.ones(len(self.node_id), dtype=bool)
        keep[self.node_off[1:] - 1] = False      # 각 way 의 마지막 node
        a = np.flatnonzero(keep)
        return a, a + 1

    def segment_way_pos(self) -> np.ndarray:
        """(S,) 각 세그먼트가 속한 way 의 위치 인덱스."""
        return np.repeat(np.arange(len(self), dtype=np.int64), np.diff(self.seg_off))

    def segment_nodes(self) -> np.ndarray:
        """(S,2) 각 세그먼트의 (from_node, to_node) — node 순서 기준."""
        a, b = self._segment_node_slots()
        return np.column_stack([self.node_id[a], self.node_id[b]])

    def segment_midpoints(self) -> np.ndarray:
        """(S,2) 각 세그먼트 중점의 (lon, lat). 공간 selector 판정에 쓴다."""
        a, b = self._segment_node_slots()
        return np.column_stack([
            (self.node_lon[a] + self.node_lon[b]) * 0.5,
            (self.node_lat[a] + self.node_lat[b]) * 0.5,
        ])

    def segment_endpoints(self) -> tuple:
        """((S,2) 시작 lon/lat, (S,2) 끝 lon/lat)."""
        a, b = self._segment_node_slots()
        return (np.column_stack([self.node_lon[a], self.node_lat[a]]),
                np.column_stack([self.node_lon[b], self.node_lat[b]]))

    def way_mask_to_segment_mask(self, way_mask: np.ndarray) -> np.ndarray:
        """(W,) way 불리언 마스크 → (S,) 세그먼트 마스크."""
        return np.asarray(way_mask, dtype=bool)[self.segment_way_pos()]

    # ------------------------------------------------------------------
    def directed_pairs(self, seg_mask: np.ndarray) -> np.ndarray:
        """
        세그먼트 마스크 → OSRM segment-speed 용 방향 있는 node 쌍.

        oneway 반영:
            oneway=yes / true / 1   → 정방향만
            oneway=-1 / reverse     → 역방향만
            그 외(양방향)           → 양쪽 다

        OSRM 의 segment-speed 파일은 방향이 있으므로, 양방향 도로를 완전히
        막으려면 (a,b) 와 (b,a) 를 모두 0 으로 줘야 한다. 한쪽만 주면 반대
        방향으로 계속 통행할 수 있어 "끊었는데 안 끊긴" 상태가 된다.

        Returns:
            (P, 2) int64 — from_node_id, to_node_id (중복 제거·정렬됨)
        """
        seg_mask = np.asarray(seg_mask, dtype=bool)
        if not seg_mask.any():
            return np.zeros((0, 2), dtype=np.int64)

        nodes = self.segment_nodes()[seg_mask]
        ow_raw = self.tags["oneway"].to_numpy()[self.segment_way_pos()[seg_mask]]
        ow = np.array(["" if (v is None or (isinstance(v, float) and np.isnan(v)))
                       else str(v) for v in ow_raw], dtype=object)

        is_fwd_only = np.isin(ow, list(ONEWAY_FORWARD))
        is_rev_only = np.isin(ow, list(ONEWAY_REVERSE))
        both = ~(is_fwd_only | is_rev_only)

        parts = [nodes[is_fwd_only],
                 nodes[is_rev_only][:, ::-1],
                 nodes[both],
                 nodes[both][:, ::-1]]
        pairs = np.vstack([p for p in parts if len(p)])
        # 인접 way 가 node 를 공유해 같은 쌍이 중복될 수 있다
        return np.unique(pairs, axis=0)

    def segments(self, positions) -> np.ndarray:
        """way 위치들을 통째로 끊을 때의 방향 있는 node 쌍 (편의 함수)."""
        way_mask = np.zeros(len(self), dtype=bool)
        way_mask[np.atleast_1d(np.asarray(positions, dtype=np.int64))] = True
        return self.directed_pairs(self.way_mask_to_segment_mask(way_mask))

    # ------------------------------------------------------------------
    def ways_on_route(self, nodes) -> dict:
        """
        경로가 지나간 way 를 역추적한다.

        OSRM 은 way id 를 돌려주지 않는다(annotations 는 nodes/distance/duration/
        speed/weight/datasources 뿐). 경로 응답에 있는 것은 OSM node id 열이므로,
        인접 쌍을 만들어 인덱스의 세그먼트와 대조해 소속 way 를 찾는다.

        방향은 무시한다 — 경로가 way 의 node 순서와 반대로 진행할 수 있다.

        Args:
            nodes: OSRM annotation.nodes (OSM node id 열)
        Returns:
            {
              "positions":   (K,) 지나간 way 의 위치 인덱스 (way_id 아님)
              "used":        (K,) way 별로 경로가 사용한 세그먼트 수
              "total":       (K,) way 의 전체 세그먼트 수
              "n_route_seg": 경로 세그먼트 수
              "n_matched":   그중 인덱스에서 찾은 수 (같아야 정상)
            }
        """
        a = np.asarray(nodes, dtype=np.int64)
        if len(a) < 2:
            return {"positions": np.zeros(0, np.int64), "used": np.zeros(0, np.int64),
                    "total": np.zeros(0, np.int64), "n_route_seg": 0, "n_matched": 0}

        route_pairs = set(map(tuple, np.column_stack([a[:-1], a[1:]])))
        seg_nodes = self.segment_nodes()

        # 경로 node 로 먼저 걸러 후보를 수백 개로 줄인다 (51만 개 전부 검사하지 않도록)
        route_nodes = np.unique(a)
        cand = np.flatnonzero(np.isin(seg_nodes[:, 0], route_nodes)
                              & np.isin(seg_nodes[:, 1], route_nodes))

        hit = np.array(
            [(int(u), int(v)) in route_pairs or (int(v), int(u)) in route_pairs
             for u, v in seg_nodes[cand]], dtype=bool)
        matched = cand[hit] if len(cand) else cand

        seg_way = self.segment_way_pos()
        pos_of_hit = seg_way[matched]
        positions, used = np.unique(pos_of_hit, return_counts=True)
        total = np.diff(self.seg_off)[positions]

        return {
            "positions": positions,
            "used": used,
            "total": total,
            "n_route_seg": len(a) - 1,
            "n_matched": int(len(matched)),
        }

    # ------------------------------------------------------------------
    def save(self, stem: Path) -> None:
        """stem 에 _ways.npz / _ways.parquet 두 파일로 저장."""
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            stem.with_name(stem.name + "_ways.npz"),
            way_id=self.way_id,
            node_off=self.node_off,
            node_id=self.node_id,
            node_lon=self.node_lon,
            node_lat=self.node_lat,
            bbox=self.bbox,
        )
        self.tags.to_parquet(stem.with_name(stem.name + "_ways.parquet"), index=False)

    @classmethod
    def load(cls, stem: Path) -> "WayIndex":
        stem = Path(stem)
        npz_path = stem.with_name(stem.name + "_ways.npz")
        pq_path = stem.with_name(stem.name + "_ways.parquet")
        if not npz_path.exists():
            raise FileNotFoundError(
                f"{npz_path} 없음. scripts/build_way_index.py 를 먼저 실행하세요."
            )
        z = np.load(npz_path)
        return cls(
            way_id=z["way_id"],
            node_off=z["node_off"],
            node_id=z["node_id"],
            node_lon=z["node_lon"],
            node_lat=z["node_lat"],
            bbox=z["bbox"],
            tags=pd.read_parquet(pq_path),
        )

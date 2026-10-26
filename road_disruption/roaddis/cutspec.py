"""
CutSpec — "무엇을 어떻게 끊을지"를 선언하는 YAML 스펙.

차단 단위를 코드에 박지 않는다. 개별 도로 하나든, 격자 셀 안의 모든 도로든,
교량 전체든, 침수구역과 겹치는 구간이든 — 전부 같은 문법으로 표현한다.
나중에 탐색 루프가 후보를 만들 때도 같은 스펙을 뱉으면 된다.

    name: daejeon_bridge_cut
    region: daejeon
    select:
      all_of:
        - {type: osm_tag,       tags: {bridge: "*"}}
        - {type: highway_class, classes: [motorway, trunk, primary]}
        - {type: bbox,          bbox: [127.35, 36.32, 127.40, 36.36]}

select 노드는 둘 중 하나다:
    조합자  {all_of: [...]} / {any_of: [...]} / {none_of: [...]}
    단일    {type: ..., ...}
조합자는 중첩할 수 있다.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .selectors import apply_selector
from .way_index import WayIndex

_COMBINATORS = ("all_of", "any_of", "none_of")


def _resolve_node(idx: WayIndex, node: dict) -> np.ndarray:
    """select 트리 한 노드를 세그먼트 마스크로 해석 (재귀)."""
    combos = [k for k in _COMBINATORS if k in node]
    if len(combos) > 1:
        raise ValueError(f"조합자는 노드당 하나만 쓸 수 있습니다: {combos}")

    if not combos:
        return apply_selector(idx, node)

    key = combos[0]
    children = node[key]
    if not children:
        raise ValueError(f"'{key}' 가 비어 있습니다.")
    masks = [_resolve_node(idx, c) for c in children]

    if key == "all_of":
        out = masks[0].copy()
        for m in masks[1:]:
            out &= m
        return out
    if key == "any_of":
        out = masks[0].copy()
        for m in masks[1:]:
            out |= m
        return out
    # none_of — 자식 중 아무것에도 걸리지 않은 세그먼트
    out = masks[0].copy()
    for m in masks[1:]:
        out |= m
    return ~out


@dataclass
class CutSpec:
    name: str
    region: str
    select: dict
    description: str = ""
    source: Optional[Path] = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, path) -> "CutSpec":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        for k in ("name", "region", "select"):
            if k not in raw:
                raise ValueError(f"{path.name}: '{k}' 항목이 필요합니다.")
        if "speed_kph" in raw:
            raise ValueError(
                f"{path.name}: speed_kph 는 더 이상 지원하지 않습니다. "
                f"차단은 way 삭제 방식뿐입니다 — 해당 줄을 지우세요."
            )
        return cls(
            name=raw["name"],
            region=raw["region"],
            select=raw["select"],
            description=raw.get("description", ""),
            source=path,
        )

    # ------------------------------------------------------------------
    def resolve(self, idx: WayIndex) -> np.ndarray:
        """(S,) 세그먼트 마스크."""
        return _resolve_node(idx, self.select)

    def summary(self, idx: WayIndex, seg_mask: np.ndarray) -> dict:
        """선택 결과 요약 — CSV 를 쓰기 전에 규모를 눈으로 확인하기 위한 것."""
        way_pos = np.unique(idx.segment_way_pos()[seg_mask])
        t = idx.tags.iloc[way_pos]
        return {
            "segments": int(seg_mask.sum()),
            "segments_total": idx.n_segments,
            "ways": int(len(way_pos)),
            "ways_total": len(idx),
            "drive_ways": int(t["is_drive"].sum()),
            "highway_counts": t["highway"].value_counts().head(8).to_dict(),
            "named": sorted(t["name"].dropna().unique().tolist())[:10],
        }

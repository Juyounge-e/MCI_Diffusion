"""config/regions.yaml 로더."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yaml

from .paths import CONFIG_DIR

_REGIONS_YAML = CONFIG_DIR / "regions.yaml"


@dataclass(frozen=True)
class Region:
    name: str
    bbox: Optional[tuple]   # (lon_min, lat_min, lon_max, lat_max) WGS84, None=전국
    osrm_port: int
    pbf: str
    role: str               # "baseline" | "experiment"

    @property
    def is_baseline(self) -> bool:
        return self.role == "baseline"

    @property
    def osrm_url(self) -> str:
        return f"http://localhost:{self.osrm_port}"


def load_regions() -> dict:
    with open(_REGIONS_YAML, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    out = {}
    for name, cfg in raw.items():
        bbox = cfg.get("bbox")
        out[name] = Region(
            name=name,
            bbox=tuple(bbox) if bbox else None,
            osrm_port=int(cfg["osrm_port"]),
            pbf=cfg["pbf"],
            role=cfg.get("role", "experiment"),
        )
    return out


def get_region(name: str) -> Region:
    regions = load_regions()
    if name not in regions:
        raise KeyError(f"알 수 없는 지역 '{name}'. 사용 가능: {sorted(regions)}")
    return regions[name]

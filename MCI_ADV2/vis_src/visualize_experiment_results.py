#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize experiment results for random (snapped) coordinates.
- Default heatmap of snapped incident points
- Clickable markers show per-point metrics
- Button shows route polylines + facility markers (centers/hospitals)
"""
import argparse
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from branca.colormap import linear
from branca.element import MacroElement, Template

BASE_DIR = Path(__file__).resolve().parents[1]

METRICS = ["reward", "time", "pdr", "wog_reward", "wog_pdr"]


ASSET_URLS = {
    "leaflet.js": "https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js",
    "leaflet.css": "https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css",
    "jquery.min.js": "https://code.jquery.com/jquery-3.7.1.min.js",
    "bootstrap.bundle.min.js": "https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js",
    "bootstrap.min.css": "https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css",
    "bootstrap-glyphicons.css": "https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css",
    "fontawesome.min.css": "https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css",
    "leaflet.awesome-markers.js": "https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js",
    "leaflet.awesome-markers.css": "https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css",
    "leaflet.awesome.rotate.min.css": "https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css",
    "d3.min.js": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js",
}




def ensure_assets(assets_dir: Path) -> dict:
    assets_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for filename, url in ASSET_URLS.items():
        local_path = assets_dir / filename
        if not local_path.exists():
            try:
                import urllib.request
                urllib.request.urlretrieve(url, local_path)
            except Exception:
                continue
        mapping[url] = f"assets/{filename}"
    return mapping


def rewrite_html_with_local_assets(html_path: Path, assets_dir: Path) -> None:
    mapping = ensure_assets(assets_dir)
    if not mapping:
        return
    html_text = html_path.read_text(encoding="utf-8", errors="ignore")
    for url, rel in mapping.items():
        html_text = html_text.replace(url, rel)
    html_path.write_text(html_text, encoding="utf-8")


INLINE_SCRIPT_RE = re.compile(r"(<script\b(?![^>]*\bsrc=)[^>]*>)(.*?)</script>", re.IGNORECASE | re.DOTALL)
INLINE_STYLE_RE = re.compile(r"(<style\b[^>]*>)(.*?)</style>", re.IGNORECASE | re.DOTALL)


def _should_externalize_script(open_tag: str) -> bool:
    lower = open_tag.lower()
    if "src=" in lower:
        return False
    m = re.search(r"type\s*=\s*['\"]([^'\"]+)['\"]", lower)
    if m:
        script_type = m.group(1)
        allowed = ("javascript" in script_type) or ("ecmascript" in script_type) or ("module" in script_type)
        if not allowed:
            return False
    return True


def split_inline_assets(html_path: Path, assets_dir: Path, base_name: str) -> Dict[str, int]:
    html_text = html_path.read_text(encoding="utf-8", errors="ignore")
    assets_dir.mkdir(parents=True, exist_ok=True)
    cache_bust = str(int(time.time()))
    # Clear old inline assets so the regenerated HTML cannot reference stale cache.
    for old_asset in assets_dir.glob(f"{base_name}*.inline*.css"):
        try:
            old_asset.unlink()
        except Exception:
            pass
    for old_asset in assets_dir.glob(f"{base_name}*.inline*.js"):
        try:
            old_asset.unlink()
        except Exception:
            pass

    style_idx = 0
    script_idx = 0

    def style_repl(match: re.Match) -> str:
        nonlocal style_idx
        open_tag = match.group(1)
        content = match.group(2).strip()
        if not content:
            return match.group(0)
        style_idx += 1
        filename = f"{base_name}.{cache_bust}.inline{style_idx}.css"
        (assets_dir / filename).write_text(content + "\n", encoding="utf-8")
        rel_path = f"{assets_dir.name}/{filename}"
        attr_text = open_tag[len("<style"):-1]
        return f'<link rel="stylesheet" href="{rel_path}"{attr_text}>'

    def script_repl(match: re.Match) -> str:
        nonlocal script_idx
        open_tag = match.group(1)
        if not _should_externalize_script(open_tag):
            return match.group(0)
        content = match.group(2).strip()
        if not content:
            return match.group(0)
        script_idx += 1
        filename = f"{base_name}.{cache_bust}.inline{script_idx}.js"
        (assets_dir / filename).write_text(content + "\n", encoding="utf-8")
        rel_path = f"{assets_dir.name}/{filename}"
        attr_text = open_tag[len("<script"):-1]
        return f'<script{attr_text} src="{rel_path}"></script>'

    html_text = INLINE_STYLE_RE.sub(style_repl, html_text)
    html_text = INLINE_SCRIPT_RE.sub(script_repl, html_text)

    # Remove only truly stray closing </script> tags (no matching opener).
    tag_iter = re.finditer(r"</script>|<script\b[^>]*>", html_text, flags=re.IGNORECASE)
    open_count = 0
    to_remove = []
    for match in tag_iter:
        tag = match.group(0).lower()
        if tag.startswith("<script"):
            open_count += 1
        else:
            if open_count > 0:
                open_count -= 1
            else:
                to_remove.append((match.start(), match.end()))
    if to_remove:
        for start, end in reversed(to_remove):
            html_text = html_text[:start] + html_text[end:]
    html_path.write_text(html_text, encoding="utf-8")
    return {"styles": style_idx, "scripts": script_idx}

def find_latest_experiment() -> str:
    scenarios_dir = BASE_DIR / "scenarios"
    if not scenarios_dir.exists():
        raise FileNotFoundError(f"scenarios folder not found: {scenarios_dir}")

    candidates = []
    for path in scenarios_dir.iterdir():
        if not path.is_dir():
            continue
        if (path / "random_metadata.csv").exists() or (path / "experiment_metadata.csv").exists():
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError("No experiment folders found under scenarios/")

    def sort_key(p: Path):
        name = p.name
        import re
        m = re.search(r"(\d{8}_\d{6})", name)
        if m:
            return m.group(1)
        return f"00000000_000000_{p.stat().st_mtime:.0f}"

    latest = sorted(candidates, key=sort_key)[-1]
    return latest.name


def load_metadata(exp_id: str) -> pd.DataFrame:
    exp_dir = BASE_DIR / "scenarios" / exp_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")

    random_csv = exp_dir / "random_metadata.csv"
    exp_csv = exp_dir / "experiment_metadata.csv"

    if not random_csv.exists():
        raise FileNotFoundError(f"random_metadata.csv not found: {random_csv}")
    if not exp_csv.exists():
        raise FileNotFoundError(f"experiment_metadata.csv not found: {exp_csv}")

    df_rand = pd.read_csv(random_csv)
    df_exp = pd.read_csv(exp_csv)

    # Use snapped coordinates as incident location
    if "snapped_latitude" in df_rand.columns:
        df_rand["latitude"] = df_rand["snapped_latitude"]
    if "snapped_longitude" in df_rand.columns:
        df_rand["longitude"] = df_rand["snapped_longitude"]

    id_candidates = ["grid_id", "point_id", "index", "id"]
    id_col = next((c for c in id_candidates if c in df_rand.columns and c in df_exp.columns), None)
    if id_col is None:
        raise ValueError("No common id column between random_metadata and experiment_metadata")

    df = df_rand.merge(
        df_exp[[id_col, "status", "failure_reason"]],
        on=id_col,
        how="left",
    )
    df["status"] = df["status"].fillna("unknown")
    df["failure_reason"] = df["failure_reason"].fillna("")

    df.attrs["id_col"] = id_col
    return df


def parse_stat_file(filepath: Path) -> Optional[Dict[str, Dict[str, float]]]:
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        metric_names = METRICS
        metrics = {}
        for i, metric_name in enumerate(metric_names):
            if i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    mean_val = float(parts[-3])
                    std_val = float(parts[-2])
                    ci_val = float(parts[-1])
                    metrics[metric_name] = {
                        "mean": mean_val if np.isfinite(mean_val) else np.nan,
                        "std": std_val if np.isfinite(std_val) else np.nan,
                        "ci_half": ci_val if np.isfinite(ci_val) else np.nan,
                    }
        return metrics
    except Exception:
        return None


def load_experiment_results(df: pd.DataFrame, exp_id: str) -> pd.DataFrame:
    results_dir = BASE_DIR / "results" / exp_id
    if not results_dir.exists():
        return df

    for metric in METRICS:
        df[f"{metric}_mean"] = np.nan
        df[f"{metric}_std"] = np.nan
        df[f"{metric}_ci_half"] = np.nan

    for idx, row in df[df["status"] == "success"].iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        stat_path = results_dir / f"lat{lat:.6f}_lon{lon:.6f}" / f"results_lat{lat:.6f}_lon{lon:.6f}_stat.txt"
        metrics = parse_stat_file(stat_path)
        if metrics:
            for metric_name, values in metrics.items():
                df.at[idx, f"{metric_name}_mean"] = values["mean"]
                df.at[idx, f"{metric_name}_std"] = values["std"]
                df.at[idx, f"{metric_name}_ci_half"] = values["ci_half"]
    return df


def load_region_boundary(shp_path: str, exp_id: str):
    shp_path = str(Path(shp_path))
    if not os.path.isabs(shp_path):
        shp_path = str(BASE_DIR / shp_path)
    if not os.path.exists(shp_path):
        return None

    try:
        gdf = gpd.read_file(shp_path, encoding="cp949")
        if gdf.crs is None:
            gdf.set_crs("EPSG:5179", inplace=True)
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        region_keyword = exp_id.split("_")[0] if "_exp_" in exp_id else "daejeon"

        row = None
        for col in gdf.columns:
            if gdf[col].dtype == "object":
                mask = gdf[col].astype(str).str.contains(region_keyword, case=False, na=False)
                if mask.any():
                    row = gdf[mask].iloc[0]
                    break
        if row is None:
            return None

        boundary = gpd.GeoDataFrame([row], crs=gdf.crs)
        return boundary
    except Exception:
        return None


def _safe_float(value, default=None):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _metric_range(values: pd.Series) -> Optional[tuple]:
    vals = pd.to_numeric(values, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    vmin = float(vals.quantile(0.05))
    vmax = float(vals.quantile(0.95))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def _metric_color(value, vmin: Optional[float], vmax: Optional[float], invert: bool, default: str = "#2b8cbe") -> str:
    if value is None:
        return default
    try:
        val = float(value)
    except Exception:
        return default
    if not np.isfinite(val):
        return default
    if vmin is None or vmax is None or vmax == vmin:
        return default
    t = (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    if invert:
        t = 1.0 - t
    blue = (44, 123, 182)
    red = (215, 25, 28)
    r = int(round(red[0] * (1 - t) + blue[0] * t))
    g = int(round(red[1] * (1 - t) + blue[1] * t))
    b = int(round(red[2] * (1 - t) + blue[2] * t))
    return f"#{r:02x}{g:02x}{b:02x}"


def _compute_density_norm(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    n = len(lat)
    if n == 0:
        return np.array([], dtype=float)

    lat_min = float(np.min(lat))
    lat_max = float(np.max(lat))
    lon_min = float(np.min(lon))
    lon_max = float(np.max(lon))
    if lat_min == lat_max or lon_min == lon_max:
        return np.ones(n, dtype=float)

    bins = int(np.clip(np.sqrt(n), 20, 80))
    lat_edges = np.linspace(lat_min, lat_max, bins + 1)
    lon_edges = np.linspace(lon_min, lon_max, bins + 1)

    hist, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])

    kernel = np.array([[1.0, 2.0, 1.0],
                       [2.0, 4.0, 2.0],
                       [1.0, 2.0, 1.0]], dtype=float)
    padded = np.pad(hist, 1, mode="edge")
    smooth = np.zeros_like(hist, dtype=float)
    for i in range(3):
        for j in range(3):
            smooth += kernel[i, j] * padded[i:i + hist.shape[0], j:j + hist.shape[1]]
    smooth /= float(kernel.sum())

    lat_idx = np.clip(np.searchsorted(lat_edges, lat, side="right") - 1, 0, bins - 1)
    lon_idx = np.clip(np.searchsorted(lon_edges, lon, side="right") - 1, 0, bins - 1)
    density = smooth[lat_idx, lon_idx]

    if np.all(np.isnan(density)):
        return np.zeros(n, dtype=float)

    lo, hi = np.percentile(density, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(density))
        hi = float(np.nanmax(density))
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0

    density_norm = np.clip((density - lo) / (hi - lo), 0.0, 1.0)
    return density_norm


def _downsample_coords(coords: List[List[float]], max_points: int = 200) -> List[List[float]]:
    if not coords:
        return coords
    if len(coords) <= max_points:
        return coords
    step = max(1, len(coords) // max_points)
    sampled = coords[::step]
    if sampled[-1] != coords[-1]:
        sampled.append(coords[-1])
    return sampled


def _smooth_grid(values: np.ndarray) -> np.ndarray:
    kernel = np.array([[1.0, 2.0, 1.0],
                       [2.0, 4.0, 2.0],
                       [1.0, 2.0, 1.0]], dtype=float)
    padded = np.pad(values, 1, mode="edge")
    smooth = np.zeros_like(values, dtype=float)
    for i in range(3):
        for j in range(3):
            smooth += kernel[i, j] * padded[i:i + values.shape[0], j:j + values.shape[1]]
    smooth /= float(kernel.sum())
    return smooth


def _smooth_metric_values(latitudes: np.ndarray, longitudes: np.ndarray, values: np.ndarray) -> np.ndarray:
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    vals = np.asarray(values, dtype=float)
    n = len(lat)
    if n == 0:
        return np.array([], dtype=float)

    lat_min = float(np.min(lat))
    lat_max = float(np.max(lat))
    lon_min = float(np.min(lon))
    lon_max = float(np.max(lon))
    if lat_min == lat_max or lon_min == lon_max:
        return vals.copy()

    bins = int(np.clip(np.sqrt(n), 20, 80))
    lat_edges = np.linspace(lat_min, lat_max, bins + 1)
    lon_edges = np.linspace(lon_min, lon_max, bins + 1)

    count, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
    weighted, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges], weights=vals)

    count_s = _smooth_grid(count)
    weighted_s = _smooth_grid(weighted)

    lat_idx = np.clip(np.searchsorted(lat_edges, lat, side="right") - 1, 0, bins - 1)
    lon_idx = np.clip(np.searchsorted(lon_edges, lon, side="right") - 1, 0, bins - 1)

    smooth_vals = np.empty(n, dtype=float)
    for i in range(n):
        c = count_s[lat_idx[i], lon_idx[i]]
        if c <= 0:
            smooth_vals[i] = vals[i]
        else:
            smooth_vals[i] = weighted_s[lat_idx[i], lon_idx[i]] / c
    return smooth_vals

def _extract_osrm_line(payload: Dict) -> Optional[List[List[float]]]:
    data = payload.get("osrm_response", {})
    routes = data.get("routes") or []
    if not routes:
        return None
    geom = routes[0].get("geometry") if isinstance(routes[0], dict) else None
    if isinstance(geom, dict):
        coords = geom.get("coordinates") or []
        if coords:
            return [[c[1], c[0]] for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
    return None


def _extract_kakao_line(payload: Dict) -> Optional[List[List[float]]]:
    data = payload.get("kakao_response", {})
    routes = data.get("routes") or []
    if not routes:
        return None
    sections = routes[0].get("sections") or []
    coords: List[List[float]] = []
    for sec in sections:
        roads = sec.get("roads") or []
        for road in roads:
            vertexes = road.get("vertexes") or []
            for i in range(0, len(vertexes) - 1, 2):
                lon = vertexes[i]
                lat = vertexes[i + 1]
                coords.append([lat, lon])
    return coords if coords else None


def _route_entries(route_dir: Path, kind: str, route_mode: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if not route_dir.exists():
        return entries

    for json_path in sorted(route_dir.rglob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("meta", {}) if isinstance(data, dict) else {}
            payload = data.get("payload", {}) if isinstance(data, dict) else {}

            direction = str(meta.get("direction_note") or "")
            if "->" in direction:
                start_label, goal_label = direction.split("->", 1)
            else:
                start_label, goal_label = "start", "goal"

            start_raw = meta.get(start_label)
            goal_raw = meta.get(goal_label)
            start = [start_raw[1], start_raw[0]] if isinstance(start_raw, list) and len(start_raw) >= 2 else None
            end = [goal_raw[1], goal_raw[0]] if isinstance(goal_raw, list) and len(goal_raw) >= 2 else None

            line = None
            provider = str(meta.get("api_provider", ""))
            if route_mode == "geometry":
                if provider == "osrm":
                    line = _extract_osrm_line(payload)
                elif provider == "kakao":
                    line = _extract_kakao_line(payload)

            if line:
                line = _downsample_coords(line, max_points=220)
            elif start and end:
                line = [start, end]

            entry = {
                "name": str(meta.get("name", "")),
                "provider": provider,
                "distance_km": meta.get("distance_km"),
                "duration_min": meta.get("duration_min"),
                "start_label": start_label,
                "goal_label": goal_label,
                "start": start,
                "end": end,
                "line": line or [],
                "kind": kind,
            }
            entries.append(entry)
        except Exception:
            continue
    return entries


def build_routes_data(df: pd.DataFrame, exp_id: str, id_col: str, route_mode: str) -> Dict[str, Dict[str, object]]:
    routes_data: Dict[str, Dict[str, object]] = {}
    for _, row in df.iterrows():
        point_id = str(row[id_col])
        lat = row["latitude"]
        lon = row["longitude"]
        scenario_dir = BASE_DIR / "scenarios" / exp_id / f"lat{lat:.6f}_lon{lon:.6f}"
        routes_base = scenario_dir / "routes"
        if not routes_base.exists():
            continue
        routes = []
        routes.extend(_route_entries(routes_base / "center2site", "center2site", route_mode))
        routes.extend(_route_entries(routes_base / "hos2site", "hos2site", route_mode))
        routes_data[point_id] = {
            "site": [float(lat), float(lon)],
            "routes": routes,
        }
    return routes_data


def add_route_js(map_obj: folium.Map, route_data: Dict[str, Dict[str, object]]):
    route_json = json.dumps(route_data, ensure_ascii=False)
    template = Template(
        """
        {% macro script(this, kwargs) %}
        (function(){
            var map = {{this.map_name}};
            var ROUTE_DATA = {{this.route_data}};
            var routeLayer = L.layerGroup().addTo(map);
            var facilityLayer = L.layerGroup().addTo(map);

            function clearRoutes(pointId){
                routeLayer.clearLayers();
                facilityLayer.clearLayers();
                if (pointId){
                    var el = document.getElementById('routes_list_' + pointId);
                    if (el){ el.innerHTML = ''; }
                }
            }
            window.clearRoutes = clearRoutes;

            function escapeHtml(str){
                if (!str && str !== 0){ return ''; }
                return String(str)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/\"/g, '&quot;')
                    .replace(/'/g, '&#39;');
            }

            function fmtNum(val, digits){
                if (val === null || val === undefined || isNaN(val)){ return '-'; }
                var d = (digits !== undefined) ? digits : 2;
                return Number(val).toFixed(d);
            }

            function buildList(routes, title){
                if (!routes || routes.length === 0){
                    return "<div style='color:#777'>경로 없음</div>";
                }
                var items = routes.map(function(r){
                    var name = escapeHtml(r.name || '(이름 없음)');
                    var provider = escapeHtml(r.provider || 'unknown');
                    var dist = fmtNum(r.distance_km, 2);
                    var dur = fmtNum(r.duration_min, 1);
                    return "<li>[" + provider + "] " + name + " - " + dist + " km, " + dur + " min</li>";
                }).join('');
                return "<div><b>" + title + "</b></div><ul style='margin:4px 0 0 16px'>" + items + "</ul>";
            }

            function addFacility(coord, name, color, label, seen){
                if (!coord || coord.length < 2){ return; }
                var key = coord[0].toFixed(6) + ',' + coord[1].toFixed(6) + label;
                if (seen[key]){ return; }
                seen[key] = true;

                var marker;
                if (label === '센터'){
                    // 안전센터: 사각형 마커
                    var iconHtml = '<div style="width:10px;height:10px;background:' + color +
                        ';border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,0.4);"></div>';
                    marker = L.marker(coord, {
                        icon: L.divIcon({
                            className: 'facility-center',
                            html: iconHtml,
                            iconSize: [10, 10],
                            iconAnchor: [5, 5]
                        })
                    });
                } else if (label === '병원'){
                    // 병원: 십자가 마커 (SVG)
                    var crossSvg = '<svg width="14" height="14" viewBox="0 0 14 14">' +
                        '<rect x="5" y="1" width="4" height="12" fill="' + color + '" stroke="#fff" stroke-width="1"/>' +
                        '<rect x="1" y="5" width="12" height="4" fill="' + color + '" stroke="#fff" stroke-width="1"/>' +
                        '</svg>';
                    marker = L.marker(coord, {
                        icon: L.divIcon({
                            className: 'facility-hospital',
                            html: crossSvg,
                            iconSize: [14, 14],
                            iconAnchor: [7, 7]
                        })
                    });
                } else {
                    // 기본: 원형 마커
                    marker = L.circleMarker(coord, {
                        radius: 4, color: color, weight: 1,
                        fill: true, fillColor: color, fillOpacity: 0.9
                    });
                }

                var title = label ? ('<b>' + label + '</b> ') : '';
                marker.bindPopup(title + (name || ''));
                marker.addTo(facilityLayer);
            }

            window.showRoutes = function(pointId){
                clearRoutes();
                var data = ROUTE_DATA[String(pointId)];
                if (!data || !data.routes){
                    alert('경로 데이터가 없습니다.');
                    return;
                }
                var site = data.site;
                if (site){
                    L.circleMarker(site, {
                        radius: 6,
                        color: '#111',
                        weight: 2,
                        fill: true,
                        fillColor: '#ffd92f',
                        fillOpacity: 0.9
                    }).bindPopup('<b>사고 지점</b>').addTo(routeLayer);
                }

                var seen = {};
                data.routes.forEach(function(r){
                    var line = r.line || [];
                    if (line.length >= 2){
                        var color = (r.kind === 'center2site') ? '#1f78b4' : '#e31a1c';
                        L.polyline(line, {color: color, weight: 3, opacity: 0.85}).addTo(routeLayer);
                    }
                    if (r.kind === 'center2site'){
                        addFacility(r.start, r.name, '#1f78b4', '센터', seen);
                    } else if (r.kind === 'hos2site'){
                        addFacility(r.end, r.name, '#e31a1c', '병원', seen);
                    }
                });

                var listEl = document.getElementById('routes_list_' + pointId);
                if (listEl){
                    var centers = data.routes.filter(function(r){ return r.kind === 'center2site'; });
                    var hospitals = data.routes.filter(function(r){ return r.kind === 'hos2site'; });
                    listEl.innerHTML = buildList(centers, '센터 → 사고지점') +
                        "<div style='margin-top:6px'>" + buildList(hospitals, '사고지점 → 병원') + "</div>";
                }

                if (site){
                    map.setView(site, 13);
                }
            };
        })();
        {% endmacro %}
        """
    )
    macro = MacroElement()
    macro.route_data = route_json
    macro.map_name = map_obj.get_name()
    macro._template = template
    map_obj.get_root().add_child(macro)


def add_facility_legend(map_obj: folium.Map):
    """시설 유형 범례 추가 (좌측 하단)"""
    legend = MacroElement()
    legend._template = Template(
        """
        {% macro script(this, kwargs) %}
        (function(){
            var FacilityLegend = L.control({position: 'bottomleft'});
            FacilityLegend.onAdd = function(){
                var div = L.DomUtil.create('div', 'facility-legend');
                div.innerHTML =
                    '<div style="background:#fff;padding:8px 12px;border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,0.2);font-size:12px;">' +
                    '<div style="font-weight:bold;margin-bottom:6px;">시설 유형</div>' +
                    '<div style="display:flex;align-items:center;margin-bottom:4px;">' +
                    '<div style="width:10px;height:10px;background:#1f78b4;border:2px solid #fff;box-shadow:0 1px 2px rgba(0,0,0,0.3);margin-right:8px;"></div>' +
                    '<span>안전센터</span>' +
                    '</div>' +
                    '<div style="display:flex;align-items:center;margin-bottom:4px;">' +
                    '<svg width="14" height="14" style="margin-right:6px;" viewBox="0 0 14 14">' +
                    '<rect x="5" y="1" width="4" height="12" fill="#e31a1c" stroke="#fff" stroke-width="1"/>' +
                    '<rect x="1" y="5" width="12" height="4" fill="#e31a1c" stroke="#fff" stroke-width="1"/>' +
                    '</svg>' +
                    '<span>병원</span>' +
                    '</div>' +
                    '<div style="display:flex;align-items:center;">' +
                    '<div style="width:10px;height:10px;background:#ffd92f;border:2px solid #111;border-radius:50%;margin-right:8px;"></div>' +
                    '<span>사고 지점</span>' +
                    '</div>' +
                    '</div>';
                return div;
            };
            FacilityLegend.addTo({{this._parent.get_name()}});
        })();
        {% endmacro %}
        """
    )
    map_obj.add_child(legend)


def build_routes_popup(route_data: Optional[Dict[str, object]], point_id: str) -> str:
    if not route_data:
        return "<div style='color:#777'>경로 데이터 없음</div>"

    pid = html.escape(str(point_id))
    html_parts = [
        "<div style='margin-top:6px'>",
        f"<button onclick=\"showRoutes('{pid}')\" style='padding:4px 8px;margin-right:6px'>경로 보기</button>",
        f"<button onclick=\"clearRoutes('{pid}')\" style='padding:4px 8px'>경로 지우기</button>",
        "</div>",
        f"<div id='routes_list_{pid}' style='margin-top:6px; max-height:220px; overflow:auto;'></div>",
    ]
    return "".join(html_parts)

def create_map(
    df: pd.DataFrame,
    exp_id: str,
    boundary_gdf,
    color_metric: str,
    route_data: Dict[str, Dict[str, object]],
    id_col: str,
) -> folium.Map:
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="OpenStreetMap",
        prefer_canvas=True,
        control_scale=True,
    )

    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

    if boundary_gdf is not None:
        folium.GeoJson(
            boundary_gdf,
            name="Boundary",
            style_function=lambda x: {
                "fillColor": "none",
                "color": "#ff3333",
                "weight": 2,
                "opacity": 0.8,
            },
            overlay=True,
            control=False,
        ).add_to(m)

    metric_settings = {
        "reward": {"label": "Reward", "good_high": True},
        "time": {"label": "Time", "good_high": False},
        "pdr": {"label": "PDR", "good_high": False},
        "wog_reward": {"label": "WOG Reward", "good_high": True},
        "wog_pdr": {"label": "WOG PDR", "good_high": False},
    }
    metric_info: Dict[str, Dict[str, object]] = {}
    legend_ids: Dict[str, str] = {}

    for metric in metric_settings.keys():
        metric_col = f"{metric}_mean"
        smooth_col = f"{metric}_smooth"
        df[smooth_col] = np.nan
        if metric_col not in df.columns:
            continue
        metric_vals = pd.to_numeric(df[metric_col], errors="coerce")
        mask = (df["status"] == "success") & np.isfinite(metric_vals)
        if mask.sum() == 0:
            continue
        if mask.sum() < 3:
            df.loc[mask, smooth_col] = metric_vals.loc[mask]
            continue
        smooth_vals = _smooth_metric_values(
            df.loc[mask, "latitude"].to_numpy(),
            df.loc[mask, "longitude"].to_numpy(),
            metric_vals.loc[mask].to_numpy(),
        )
        df.loc[mask, smooth_col] = smooth_vals

    for metric, cfg in metric_settings.items():
        metric_col = f"{metric}_smooth"
        if metric_col not in df.columns:
            metric_info[metric] = {"vmin": None, "vmax": None, "invert": not cfg["good_high"]}
            continue
        valid_vals = df.loc[df["status"] == "success", metric_col]
        metric_range = _metric_range(valid_vals)
        if metric_range is None:
            metric_info[metric] = {"vmin": None, "vmax": None, "invert": not cfg["good_high"]}
            continue
        vmin, vmax = metric_range
        invert = not cfg["good_high"]
        if invert:
            cmap = linear.RdBu_11.scale(vmax, vmin)
        else:
            cmap = linear.RdBu_11.scale(vmin, vmax)
        cmap = cmap.to_step(n=9)
        good_label = "높을수록 좋음" if cfg["good_high"] else "낮을수록 좋음"
        cmap.caption = f"{cfg['label']} (mean) - {good_label}"
        cmap.add_to(m)
        legend_ids[metric] = cmap.get_name()
        metric_info[metric] = {"vmin": vmin, "vmax": vmax, "invert": invert}


    density_norm = _compute_density_norm(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    df["_density_norm"] = density_norm

    markers = folium.FeatureGroup(name="Markers", show=True)
    marker_meta: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        status = row.get("status", "unknown")
        snap_dist = row.get("snap_distance_m")

        label_id = row[id_col] if id_col and id_col in row else idx
        label_key = str(label_id)

        neutral_color = "#9e9e9e"
        color = neutral_color if status == "success" else "#666666"

        density = float(row.get("_density_norm", 0.0))
        density_adj = density ** 0.7
        radius = 3.0 + density_adj * 6.0
        fill_opacity = 0.25 + density_adj * 0.65
        stroke_opacity = 0.3 + density_adj * 0.6
        if status != "success":
            fill_opacity *= 0.6
            stroke_opacity *= 0.6

        def fmt(val, fmt_str="{:.4f}"):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "-"
            try:
                return fmt_str.format(float(val))
            except Exception:
                return str(val)

        routes_html = build_routes_popup(route_data.get(label_key), label_key)

        popup_html = "".join(
            [
                "<div style='width: 300px; font-family: \"Malgun Gothic\", \"Apple SD Gothic Neo\", \"Noto Sans KR\", Arial, sans-serif;'>",
                f"<h4>좌표 {html.escape(str(label_id))}</h4>",
                f"<div><b>상태:</b> {html.escape(str(status))}</div>",
                f"<div><b>좌표:</b> {lat:.6f}, {lon:.6f}</div>",
                f"<div><b>Snap 거리(m):</b> {fmt(snap_dist, '{:.1f}')}</div>",
                "<hr style='margin:6px 0'>",
                f"<div><b>Reward:</b> {fmt(row.get('reward_mean'), '{:.4f}')}</div>",
                f"<div><b>Time:</b> {fmt(row.get('time_mean'), '{:.2f}')}</div>",
                f"<div><b>PDR:</b> {fmt(row.get('pdr_mean'), '{:.4f}')}</div>",
                f"<div><b>WOG Reward:</b> {fmt(row.get('wog_reward_mean'), '{:.4f}')}</div>",
                f"<div><b>WOG PDR:</b> {fmt(row.get('wog_pdr_mean'), '{:.4f}')}</div>",
                "<hr style='margin:6px 0'>",
                routes_html,
                "</div>",
            ]
        )

        # Base point marker (always visible by default)
        point_marker = folium.CircleMarker(
            location=[lat, lon],
            radius=2.2,
            color=color,
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.0,
            opacity=0.0,
            popup=folium.Popup(popup_html, max_width=420),
            tooltip=f"{label_id}",
        )
        point_marker.add_to(markers)

        # Metric circle (hidden until metric selected)
        snap_radius_m = _safe_float(snap_dist, default=0.0) or 0.0
        metric_circle = folium.Circle(
            location=[lat, lon],
            radius=250,
            color="#9e9e9e",
            weight=1,
            fill=True,
            fillColor="#9e9e9e",
            fillOpacity=0.25,
            opacity=0.55,
            popup=folium.Popup(popup_html, max_width=420),
            tooltip=f"{label_id}",
        )
        metric_circle.add_to(markers)

        def _js_num(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "null"
            try:
                return f"{float(val):.6f}"
            except Exception:
                return "null"

        marker_meta.append(
            {
                "point": point_marker.get_name(),
                "circle": metric_circle.get_name(),
                "reward": _js_num(row.get("reward_smooth")),
                "time": _js_num(row.get("time_smooth")),
                "pdr": _js_num(row.get("pdr_smooth")),
                "wog_reward": _js_num(row.get("wog_reward_smooth")),
                "wog_pdr": _js_num(row.get("wog_pdr_smooth")),
                "status": str(status),
                "radius": float(radius),
                "fill_opacity": float(fill_opacity),
                "stroke_opacity": float(stroke_opacity),
                "snap_m": float(snap_radius_m),
            }
        )

    markers.add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    add_route_js(m, route_data)

    # Metric control for color mode switching
    marker_entries_list = []
    for meta in marker_meta:
        entry = (
            "{point: "
            + meta["point"]
            + ", circle: "
            + meta["circle"]
            + ", reward: "
            + meta["reward"]
            + ", time: "
            + meta["time"]
            + ", pdr: "
            + meta["pdr"]
            + ", wog_reward: "
            + meta["wog_reward"]
            + ", wog_pdr: "
            + meta["wog_pdr"]
            + ", status: "
            + json.dumps(meta["status"])
            + f", radius: {meta['radius']:.3f}, fillOpacity: {meta['fill_opacity']:.3f}, strokeOpacity: {meta['stroke_opacity']:.3f}, snap_m: {meta['snap_m']:.3f}"
            + "}"
        )
        marker_entries_list.append(entry)
    marker_entries = ",\n".join(marker_entries_list)

    metric_control = MacroElement()
    metric_control.metric_info = json.dumps(metric_info, ensure_ascii=False)
    metric_control.legend_ids = json.dumps(legend_ids, ensure_ascii=False)
    metric_control.marker_entries = marker_entries
    metric_control.default_metric = ""
    metric_control._template = Template(
        """
        {% macro script(this, kwargs) %}
        (function(){
            var map = {{this._parent.get_name()}};
            var METRIC_INFO = {{this.metric_info}};
            var LEGEND_IDS = {{this.legend_ids}};
            var METRIC_MARKERS = [
            {{this.marker_entries}}
            ];
            var DEFAULT_METRIC = "{{this.default_metric}}";
            var NEUTRAL_COLOR = '#9e9e9e';
            var DEFAULT_RADIUS_METERS = 250;
            var DEFAULT_FILL_OPACITY = 0.25;
            var DEFAULT_STROKE_OPACITY = 0.55;
            var METRIC_FILL_OPACITY = 0.45;
            var METRIC_STROKE_OPACITY = 0.7;

            function computeMetricRanges(){
                var ranges = {};
                Object.keys(METRIC_INFO).forEach(function(key){
                    ranges[key] = {vmin: Infinity, vmax: -Infinity};
                });
                METRIC_MARKERS.forEach(function(m){
                    if (m.status && m.status !== 'success'){ return; }
                    Object.keys(METRIC_INFO).forEach(function(key){
                        var val = m[key];
                        if (val === null || val === undefined || isNaN(val)){ return; }
                        if (val < ranges[key].vmin){ ranges[key].vmin = val; }
                        if (val > ranges[key].vmax){ ranges[key].vmax = val; }
                    });
                });
                Object.keys(ranges).forEach(function(key){
                    if (!isFinite(ranges[key].vmin) || !isFinite(ranges[key].vmax)){
                        ranges[key] = {vmin: METRIC_INFO[key].vmin, vmax: METRIC_INFO[key].vmax};
                    }
                });
                return ranges;
            }

            var METRIC_RANGE = computeMetricRanges();

            function clamp(val, min, max){
                return Math.max(min, Math.min(max, val));
            }

            function getMetricRange(metric){
                return METRIC_RANGE[metric] || METRIC_INFO[metric] || null;
            }

            function legendObjFor(metric){
                var key = LEGEND_IDS[metric];
                return key ? window[key] : null;
            }

            function normalizeColor(color){
                if (!color){ return null; }
                if (typeof color === 'string' && color.charAt(0) === '#' && color.length === 9){
                    var r = parseInt(color.slice(1, 3), 16);
                    var g = parseInt(color.slice(3, 5), 16);
                    var b = parseInt(color.slice(5, 7), 16);
                    var a = parseInt(color.slice(7, 9), 16) / 255;
                    return 'rgba(' + r + ',' + g + ',' + b + ',' + a.toFixed(3) + ')';
                }
                return color;
            }

            function paletteRangeFor(metric){
                var legendObj = legendObjFor(metric);
                var range = null;
                if (legendObj && legendObj.color && typeof legendObj.color.range === 'function'){
                    range = legendObj.color.range();
                }
                if (range && range.length){
                    var uniqueCount = 0;
                    var seen = {};
                    for (var i = 0; i < range.length; i++){
                        var c = range[i];
                        if (!seen[c]){
                            seen[c] = true;
                            uniqueCount++;
                            if (uniqueCount > 1){
                                return range;
                            }
                        }
                    }
                }
                var rewardLegend = legendObjFor('reward');
                if (rewardLegend && rewardLegend.color && typeof rewardLegend.color.range === 'function'){
                    var rewardRange = rewardLegend.color.range();
                    if (rewardRange && rewardRange.length){
                        return rewardRange;
                    }
                }
                return range && range.length ? range : null;
            }

            function colorFromPalette(range, t){
                if (!range || !range.length){ return null; }
                var idx = Math.round(t * (range.length - 1));
                idx = Math.max(0, Math.min(range.length - 1, idx));
                return normalizeColor(range[idx]);
            }

            function colorFor(metric, value){
                var info = METRIC_INFO[metric];
                var rangeInfo = getMetricRange(metric);
                if (!info || !rangeInfo || rangeInfo.vmin === null || rangeInfo.vmax === null || value === null || value === undefined || isNaN(value)){
                    return NEUTRAL_COLOR;
                }
                var vmin = rangeInfo.vmin;
                var vmax = rangeInfo.vmax;
                if (!isFinite(vmin) || !isFinite(vmax)){
                    return NEUTRAL_COLOR;
                }
                var t = (vmax === vmin) ? 0.5 : (value - vmin) / (vmax - vmin);
                t = clamp(t, 0, 1);
                if (info.invert){ t = 1 - t; }
                var palette = paletteRangeFor(metric);
                var color = colorFromPalette(palette, t);
                if (color){
                    return color;
                }
                var r = Math.round(215 * (1 - t) + 44 * t);
                var g = Math.round(25 * (1 - t) + 123 * t);
                var b = Math.round(28 * (1 - t) + 182 * t);
                return 'rgb(' + r + ',' + g + ',' + b + ')';
            }

            function updateLegendScale(metric){
                var legendObj = legendObjFor(metric);
                var rangeInfo = getMetricRange(metric);
                if (!legendObj || !rangeInfo || !legendObj.color || !legendObj.x || !legendObj.g){
                    return;
                }
                var vmin = rangeInfo.vmin;
                var vmax = rangeInfo.vmax;
                if (!isFinite(vmin) || !isFinite(vmax) || vmin === vmax){
                    return;
                }
                var range = paletteRangeFor(metric);
                if (!range || !range.length){
                    return;
                }
                if (legendObj.color && typeof legendObj.color.range === 'function'){
                    legendObj.color.range(range);
                }
                var legendRange = range;
                var info = METRIC_INFO[metric];
                if (info && info.invert && legendRange.length > 1){
                    legendRange = legendRange.slice().reverse();
                }
                var steps = range.length;
                var domain = [];
                for (var i = 1; i < steps; i++){
                    domain.push(vmin + (vmax - vmin) * (i / steps));
                }
                if (legendObj.color.domain){
                    legendObj.color.domain(domain);
                }
                if (legendObj.x.domain){
                    legendObj.x.domain([vmin, vmax]);
                }
                if (legendObj.xAxis && legendObj.xAxis.tickValues){
                    var ticks = [];
                    var tickCount = 10;
                    for (var t = 0; t <= tickCount; t++){
                        ticks.push(vmin + (vmax - vmin) * (t / tickCount));
                    }
                    legendObj.xAxis.tickValues(ticks);
                }
                var rectData = legendRange.map(function(d, i){
                    return {
                        x0: i ? legendObj.x(legendObj.color.domain()[i - 1]) : legendObj.x.range()[0],
                        x1: i < legendObj.color.domain().length ? legendObj.x(legendObj.color.domain()[i]) : legendObj.x.range()[1],
                        z: d
                    };
                });
                var rects = legendObj.g.selectAll("rect").data(rectData);
                rects.attr("x", function(d){ return d.x0; })
                    .attr("width", function(d){ return d.x1 - d.x0; })
                    .style("fill", function(d){ return d.z; });
                rects.enter().append("rect")
                    .attr("height", 10)
                    .attr("x", function(d){ return d.x0; })
                    .attr("width", function(d){ return d.x1 - d.x0; })
                    .style("fill", function(d){ return d.z; });
                rects.exit().remove();
                if (legendObj.g.call && legendObj.xAxis){
                    legendObj.g.call(legendObj.xAxis);
                }
            }

            function initLegendContainers(){
                Object.keys(LEGEND_IDS).forEach(function(key){
                    var legendObj = window[LEGEND_IDS[key]];
                    if (legendObj && legendObj.legend && legendObj.legend._container){
                        legendObj.legend._container.id = LEGEND_IDS[key];
                    }
                });
            }

            function initMarkers(){
                METRIC_MARKERS.forEach(function(m){
                    if (m.circle){
                        m.circle.setRadius(DEFAULT_RADIUS_METERS);
                    }
                    if (m.point && m.circle){
                        var popup = m.point.getPopup ? m.point.getPopup() : null;
                        if (popup){
                            m.circle.bindPopup(popup);
                        }
                        var tooltip = m.point.getTooltip ? m.point.getTooltip() : null;
                        if (tooltip){
                            m.circle.bindTooltip(tooltip);
                        }
                    }
                });
            }

            function updateLegend(metric){
                if (metric){
                    updateLegendScale(metric);
                }
                var activeContainer = null;
                Object.keys(LEGEND_IDS).forEach(function(key){
                    var legendObj = window[LEGEND_IDS[key]];
                    var svgNode = legendObj && legendObj.svg && typeof legendObj.svg.node === 'function'
                        ? legendObj.svg.node()
                        : null;
                    var show = metric && key === metric;
                    if (svgNode){
                        svgNode.style.display = show ? 'block' : 'none';
                        if (show && svgNode.parentElement){
                            activeContainer = svgNode.parentElement;
                        }
                    }
                });
                var containers = document.querySelectorAll('.legend.leaflet-control');
                containers.forEach(function(el){
                    if (metric && activeContainer){
                        el.style.display = (el === activeContainer) ? 'block' : 'none';
                    } else {
                        el.style.display = 'none';
                    }
                });
            }

            function updateButtons(metric){
                var radios = document.querySelectorAll('.metric-control input[type="radio"]');
                radios.forEach(function(radio){
                    radio.checked = (metric && radio.value === metric);
                });
            }

            function applyMetric(metric){
                var hasMetric = metric && METRIC_INFO[metric];
                METRIC_MARKERS.forEach(function(m){
                    var color = NEUTRAL_COLOR;
                    if (m.status !== 'success'){
                        color = '#666666';
                    } else if (hasMetric){
                        color = colorFor(metric, m[metric]);
                    }
                    if (m.circle){
                        if (hasMetric){
                            m.circle.setStyle({
                                color: color,
                                fillColor: color,
                                fillOpacity: METRIC_FILL_OPACITY,
                                opacity: METRIC_STROKE_OPACITY
                            });
                        } else {
                            m.circle.setStyle({
                                color: color,
                                fillColor: color,
                                fillOpacity: DEFAULT_FILL_OPACITY,
                                opacity: DEFAULT_STROKE_OPACITY
                            });
                        }
                    }
                    if (m.point){
                        m.point.setStyle({
                            color: color,
                            fillColor: color,
                            fillOpacity: 0.0,
                            opacity: 0.0
                        });
                    }
                });
                updateLegend(hasMetric ? metric : null);
                updateButtons(hasMetric ? metric : null);
            }

            window.__toggleMetric = function(metric, el){
                if (el && el.classList && el.classList.contains('active')){
                    applyMetric(DEFAULT_METRIC);
                } else {
                    applyMetric(metric);
                }
                return false;
            };

            try { initLegendContainers(); } catch (e) { if (console && console.error){ console.error(e); } }
            try { initMarkers(); } catch (e) { if (console && console.error){ console.error(e); } }

            // 메트릭 컨트롤 생성 (별도 Leaflet control)
            var MetricControl = L.control({position: 'topright'});
            MetricControl.onAdd = function(){
                var div = L.DomUtil.create('div', 'metric-control leaflet-bar');
                div.innerHTML =
                    '<div class="metric-control-title">지표 선택</div>' +
                    '<label><input type="radio" name="metric" value="reward"><span>Reward</span></label>' +
                    '<label><input type="radio" name="metric" value="time"><span>Time</span></label>' +
                    '<label><input type="radio" name="metric" value="pdr"><span>PDR</span></label>' +
                    '<label><input type="radio" name="metric" value="wog_reward"><span>w.o.G Reward</span></label>' +
                    '<label><input type="radio" name="metric" value="wog_pdr"><span>w.o.G PDR</span></label>';
                L.DomEvent.disableClickPropagation(div);

                // 라디오 버튼 이벤트 바인딩
                var radios = div.querySelectorAll('input[type="radio"]');
                radios.forEach(function(radio){
                    radio.addEventListener('change', function(e){
                        e.stopPropagation();
                        applyMetric(this.value);
                    });
                });

                return div;
            };
            MetricControl.addTo(map);

            // 범례 초기 숨김
            var legendContainers = document.querySelectorAll('.legend.leaflet-control');
            legendContainers.forEach(function(el){ el.style.display = 'none'; });

            // 초기 메트릭 적용
            setTimeout(function(){
                applyMetric(DEFAULT_METRIC);
                updateButtons(DEFAULT_METRIC);
            }, 50);
        })();
        {% endmacro %}
        """
    )

    m.add_child(metric_control)
    # Styling for the metric control
    m.get_root().header.add_child(
        folium.Element(
            """
            <style>
            .leaflet-interactive {
                transition: fill 0.35s ease, stroke 0.35s ease, opacity 0.35s ease, fill-opacity 0.35s ease, stroke-opacity 0.35s ease;
            }
            .metric-control.leaflet-bar {
                background: #fff;
                padding: 6px 10px 6px 8px;
                border-radius: 5px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", Arial, sans-serif;
                font-size: 13px;
                line-height: 1.4;
                min-width: 120px;
            }
            .metric-control-title {
                font-size: 12px;
                font-weight: bold;
                color: #333;
                padding: 4px 0 6px 4px;
                margin-bottom: 4px;
                border-bottom: 1px solid #ddd;
            }
            .metric-control label {
                display: flex;
                align-items: center;
                padding: 4px 6px;
                cursor: pointer;
                border-radius: 3px;
                white-space: nowrap;
            }
            .metric-control label:hover {
                background: #f4f4f4;
            }
            .metric-control input[type="radio"] {
                margin: 0 8px 0 0;
                cursor: pointer;
                width: 14px;
                height: 14px;
            }
            .metric-control span {
                color: #333;
                font-size: 13px;
            }
            .facility-center, .facility-hospital {
                background: transparent !important;
                border: none !important;
            }
            .facility-legend {
                font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", Arial, sans-serif;
            }
            .legend.leaflet-control {
                display: none;
            }
            </style>
            """
        )
    )

    # 시설 유형 범례 추가
    add_facility_legend(m)

    return m


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize MCI experiment results (random points)")
    parser.add_argument("--exp-id", dest="exp_id", default=None, help="experiment id (default: latest)")
    parser.add_argument(
        "--shp",
        default=str(BASE_DIR / "scenarios" / "ctprvn.shp"),
        help="Shapefile path (default: scenarios/ctprvn.shp)",
    )
    parser.add_argument(
        "--color-metric",
        default="reward",
        choices=METRICS,
        help="metric for marker colors and heat weights",
    )

    parser.add_argument(
        "--use-cdn",
        action="store_true",
        help="use CDN assets instead of local cached assets",
    )
    parser.add_argument(
        "--route-mode",
        default="geometry",
        choices=["straight", "geometry"],
        help="route line mode: straight (start-end) or geometry (heavy)",
    )
    args = parser.parse_args()

    exp_id = args.exp_id or find_latest_experiment()

    df = load_metadata(exp_id)
    df = load_experiment_results(df, exp_id)

    id_col = df.attrs.get("id_col") or "index"

    route_data = build_routes_data(df, exp_id, id_col, args.route_mode)

    boundary = load_region_boundary(args.shp, exp_id)

    output_dir = BASE_DIR / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    m = create_map(df, exp_id, boundary, args.color_metric, route_data, id_col)
    output_path = output_dir / f"{exp_id}_visualization.html"
    m.save(str(output_path))

    if not args.use_cdn:
        rewrite_html_with_local_assets(output_path, output_dir / "assets")

    split_inline_assets(output_path, output_dir / f"{exp_id}_visualization_assets", f"{exp_id}_visualization")

    print("=" * 70)
    print("Visualization complete")
    print(f"Output: {output_path}")
    print("Open the HTML file in a browser.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

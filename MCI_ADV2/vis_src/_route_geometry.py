"""Table 시나리오의 도로 경로선은 지도를 볼 때만 가져와 캐시한다."""
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local

_HTTP_LOCAL = local()


def index_hospital_routes(hospitals, routes):
    """API 호출 순서와 시간순 병원 인덱스를 이름·원본 좌표로 연결한다."""
    result = {}
    for row in hospitals:
        name = str(row.get("Hospital Name", row.get("요양기관명", ""))).strip()
        candidates = [r for r in routes if str(r.get("meta", {}).get("name", "")).strip() == name]
        if len(candidates) > 1 and "x좌표" in row and "y좌표" in row:
            point = (round(float(row["x좌표"]), 6), round(float(row["y좌표"]), 6))
            candidates = [r for r in candidates if tuple(round(float(v), 6) for v in
                          r.get("meta", {}).get("hospital", [])) == point]
        if len(candidates) == 1:
            result[int(row["Index"])] = candidates[0]
    return result


@st.cache_data(show_spinner=False, ttl=3600, max_entries=1024)
def _fetch_geometry(base, start, end):
    if not hasattr(_HTTP_LOCAL, "session"):
        _HTTP_LOCAL.session = requests.Session()
    response = _HTTP_LOCAL.session.get(
        f"{base.rstrip('/')}/route/v1/driving/{start[0]},{start[1]};{end[0]},{end[1]}",
        params={"overview": "full", "geometries": "geojson", "steps": "false", "alternatives": "false"},
        timeout=15)
    response.raise_for_status()
    data = response.json()
    routes = data.get("routes") or []
    if data.get("code") != "Ok" or not routes or not routes[0].get("geometry", {}).get("coordinates"):
        raise ValueError("OSRM에서 도로 경로선을 받지 못했습니다.")
    return data


def prefetch_osrm_geometry(routes):
    """표시할 경로만 최대 4개씩 내려받아 첫 지도 로딩 시간을 줄인다."""
    pending = [r for r in routes if r.get("meta", {}).get("service") == "table"
               and "osrm_response" not in (r.get("payload") or {}) and not r.get("_geometry_failed")]
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for route in pending:
            meta = route["meta"]
            try:
                args = (meta["osrm_url"], tuple(meta[meta["start_label"]]), tuple(meta[meta["goal_label"]]))
            except (KeyError, TypeError):
                route["_geometry_failed"] = True
                st.warning(f"{meta.get('name', 'OSRM')}: 경로 좌표 정보가 없습니다.")
                continue
            task = pool.submit(_fetch_geometry, *args)
            futures[task] = route
        for task in as_completed(futures):
            route = futures[task]
            try:
                route.setdefault("payload", {})["osrm_response"] = task.result()
            except (requests.RequestException, ValueError) as exc:
                route["_geometry_failed"] = True
                st.warning(f"{route['meta'].get('name', 'OSRM')}: 도로 경로선을 받지 못했습니다. ({exc})")


def osrm_payload(route_obj):
    payload = route_obj.get("payload") or {}
    if "osrm_response" in payload:
        return payload["osrm_response"]
    meta = route_obj.get("meta") or {}
    if meta.get("service") != "table":
        return {}
    if route_obj.get("_geometry_failed"):
        return {}
    try:
        data = _fetch_geometry(meta["osrm_url"], tuple(meta[meta["start_label"]]), tuple(meta[meta["goal_label"]]))
    except (requests.RequestException, ValueError, KeyError) as exc:
        route_obj["_geometry_failed"] = True
        st.warning(f"{meta.get('name', 'OSRM')}: 도로 경로선을 불러오지 못했습니다. 거리·시간은 table 계산값입니다. ({exc})")
        return {}
    payload["osrm_response"] = data
    route_obj["payload"] = payload
    return data

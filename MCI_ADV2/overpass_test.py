# test_osrm_overpass_layer.py
# Python 3.9+
#
# 목적:
# 1) 입력 좌표(lat, lon)가 OSRM 도로에 10m 이내로 스냅되는지 판정 (nearest + radiuses=10)
# 2) 스냅되면, OSRM nearest의 nodes를 기반으로 Overpass에서 parent way(highway) 태그 조회
# 3) 반경 10m 내 서로 다른 layer의 highway way가 동시에 존재하는지 확인
#
# 실행 예:
#   python overpass_test.py --lat 36.326633 --lon 127.310524
#   python overpass_test.py --lat 36.326633 --lon 127.310524 --osrm https://router.project-osrm.org
#   python overpass_test.py --lat 36.326633 --lon 127.310524 --profile driving --k 5 --radius_m 10
#
# 참고:
# - OSRM 좌표 순서: lon,lat
# - Overpass around 좌표 순서: lat,lon

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_OSRM = "https://router.project-osrm.org"
DEFAULT_OVERPASS = "https://overpass-api.de/api/interpreter"


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def osrm_nearest(
    session: requests.Session,
    osrm_base: str,
    lon: float,
    lat: float,
    profile: str,
    k: int,
    radius_m: float,
    timeout_s: int = 15,
) -> Dict[str, Any]:
    """
    OSRM nearest 호출.
    radiuses=radius_m로 검색 반경 제한(좌표 1개이므로 파라미터도 1개).
    """
    osrm_base = osrm_base.rstrip("/")
    url = f"{osrm_base}/nearest/v1/{profile}/{lon},{lat}"

    params = {
        "number": str(max(1, k)),
        "radiuses": str(max(0.0, radius_m)),  # <= 응답 반경 제한(예: 10m)
    }

    r = session.get(url, params=params, timeout=timeout_s)
    # OSRM 에러는 보통 400 + json(code=NoSegment 등) 형태
    try:
        js = r.json()
    except Exception:
        r.raise_for_status()
        raise

    return js


def overpass_query(
    session: requests.Session,
    overpass_url: str,
    query: str,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    """
    Overpass QL 실행.
    Overpass는 POST 권장(쿼리가 길어질 수 있음).
    """
    r = session.post(overpass_url, data={"data": query}, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def build_overpass_parent_ways_query(node_ids: List[int]) -> str:
    """
    OSRM waypoint.nodes(노드 id 리스트)를 받아:
      node(id:...);
      way(bn)["highway"];
      out tags center;
    형태로 parent way의 태그를 조회.
    """
    # Overpass는 id 리스트를 쉼표로 연결
    ids_csv = ",".join(str(n) for n in node_ids)
    return f"""
[out:json][timeout:25];
node(id:{ids_csv});
way(bn)["highway"];
out tags center;
""".strip()


def build_overpass_around_ways_query(lat: float, lon: float, radius_m: float) -> str:
    """
    스냅 지점 주변 반경으로 highway way를 조회하는 fallback.
    """
    # around:radius,lat,lon  (주의: lat,lon 순서)
    return f"""
[out:json][timeout:25];
way(around:{radius_m},{lat},{lon})["highway"];
out tags center;
""".strip()


def summarize_way(el: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overpass element(way)에서 필요한 태그만 추려 보기 좋게 요약.
    """
    tags = el.get("tags", {}) or {}
    center = el.get("center", {}) or {}
    return {
        "way_id": el.get("id"),
        "center_lat": center.get("lat"),
        "center_lon": center.get("lon"),
        "highway": tags.get("highway"),
        "ref": tags.get("ref"),
        "name": tags.get("name"),
        "layer": tags.get("layer"),   # 문자열일 수도 있음
        "bridge": tags.get("bridge"),
        "tunnel": tags.get("tunnel"),
        "service": tags.get("service"),
        "access": tags.get("access"),
        "oneway": tags.get("oneway"),
        # 필요하면 여기에 더 추가
    }


def parse_layer(layer_val: Optional[str]) -> Optional[int]:
    if layer_val is None:
        return None
    try:
        return int(str(layer_val).strip())
    except Exception:
        return None


def _fmt_value(val: Any) -> str:
    if val is None or val == "":
        return "-"
    return str(val)


def _fmt_list(vals: Optional[List[Any]]) -> str:
    if not vals:
        return "-"
    return ",".join(str(v) for v in vals)


def _fmt_float(val: Any, digits: int = 6) -> str:
    if val is None:
        return "-"
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return _fmt_value(val)


def _haversine_m(
    lat1: Optional[float],
    lon1: Optional[float],
    lat2: Optional[float],
    lon2: Optional[float],
) -> Optional[float]:
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    try:
        lat1_f = float(lat1)
        lon1_f = float(lon1)
        lat2_f = float(lat2)
        lon2_f = float(lon2)
    except Exception:
        return None

    r = 6371000.0
    phi1 = math.radians(lat1_f)
    phi2 = math.radians(lat2_f)
    dphi = math.radians(lat2_f - lat1_f)
    dlambda = math.radians(lon2_f - lon1_f)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "(없음)"

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row_vals: List[str]) -> str:
        return "  " + " | ".join(row_vals[i].ljust(widths[i]) for i in range(len(headers)))

    sep = "  " + "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def _build_osrm_waypoints_table(waypoints: List[Dict[str, Any]]) -> str:
    headers = ["#", "거리(m)", "이름", "스냅위도", "스냅경도", "노드"]
    rows: List[List[str]] = []
    for i, wp in enumerate(waypoints, start=1):
        loc = wp.get("location", [None, None])
        lon = loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else None
        lat = loc[1] if isinstance(loc, (list, tuple)) and len(loc) > 1 else None
        rows.append(
            [
                str(i),
                _fmt_float(wp.get("distance"), 3),
                _fmt_value(wp.get("name")),
                _fmt_float(lat, 6),
                _fmt_float(lon, 6),
                _fmt_list(wp.get("nodes", [])),
            ]
        )
    return _render_table(headers, rows)


def _build_ways_table(
    summaries: List[Dict[str, Any]],
    snapped_lat: Optional[float],
    snapped_lon: Optional[float],
) -> str:
    headers = [
        "도로ID",
        "레이어",
        "도로종류",
        "이름",
        "노선번호",
        "교량",
        "터널",
        "일방통행",
        "중심위도",
        "중심경도",
        "중심거리(m)",
    ]
    rows: List[List[str]] = []
    for ssum in summaries:
        center_lat = ssum.get("center_lat")
        center_lon = ssum.get("center_lon")
        dist_m = _haversine_m(snapped_lat, snapped_lon, center_lat, center_lon)
        rows.append(
            [
                _fmt_value(ssum.get("way_id")),
                _fmt_value(ssum.get("layer")),
                _fmt_value(ssum.get("highway")),
                _fmt_value(ssum.get("name")),
                _fmt_value(ssum.get("ref")),
                _fmt_value(ssum.get("bridge")),
                _fmt_value(ssum.get("tunnel")),
                _fmt_value(ssum.get("oneway")),
                _fmt_float(center_lat, 6),
                _fmt_float(center_lon, 6),
                _fmt_float(dist_m, 1),
            ]
        )
    return _render_table(headers, rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=37.547307)
    ap.add_argument("--lon", type=float, default=126.590055)
    ap.add_argument("--osrm", type=str, default=DEFAULT_OSRM, help="OSRM 기본 URL")
    ap.add_argument("--profile", type=str, default="driving", help="OSRM 프로파일")
    ap.add_argument("--k", type=int, default=5, help="nearest 후보 개수")
    ap.add_argument("--radius_m", type=float, default=10.0, help="스냅 허용 반경(m)")
    ap.add_argument("--overpass", type=str, default=DEFAULT_OVERPASS, help="Overpass API 엔드포인트")
    ap.add_argument("--debug", action="store_true", help="raw json 출력")
    args = ap.parse_args()

    lat, lon = args.lat, args.lon

    s = requests.Session()
    s.headers.update({"User-Agent": "mci-osrm-overpass-layer-test/1.0"})

    print(f"[입력] 위도/경도 = {lat},{lon}")
    print(f"[OSRM] base={args.osrm} profile={args.profile} 후보수(k)={args.k} 반경={args.radius_m}m")

    # 1) OSRM nearest로 도로(설정 반경 이내) 스냅 여부 판정 + 스냅 좌표 수집
    js = osrm_nearest(
        session=s,
        osrm_base=args.osrm,
        lon=lon,
        lat=lat,
        profile=args.profile,
        k=args.k,
        radius_m=args.radius_m,
    )

    if args.debug:
        print("\n[디버그][OSRM nearest 원본]")
        print(_pretty(js))

    code = js.get("code")
    if code != "Ok":
        print(f"\n[결과] OSRM nearest 실패: code={code}, 메시지={js.get('message')}")
        print("=> 반경 내(예: 10m) 도로 세그먼트가 없다고 판단 가능(NoSegment 등).")
        return 1

    waypoints = js.get("waypoints", []) or []
    if not waypoints:
        print("\n[결과] OSRM nearest: waypoints가 비어있음 -> 도로 판정 실패")
        return 1

    # nearest 후보 중 반경 이내만 추림 (radiuses 재검증)
    within = []
    for wp in waypoints:
        dist_m = float(wp.get("distance", 1e18))
        if dist_m <= args.radius_m + 1e-9:
            within.append(wp)

    if not within:
        print(f"\n[결과] nearest 후보가 있으나 {args.radius_m}m 이내가 없음 -> 도로 판정 실패")
        return 1

    print(f"\n[결과] 도로 판정 성공: {len(within)}개 후보가 {args.radius_m}m 이내")
    print(_build_osrm_waypoints_table(within))
    if args.debug:
        for i, wp in enumerate(within, start=1):
            loc = wp.get("location", [None, None])
            dist_m = wp.get("distance")
            name = wp.get("name")
            nodes = wp.get("nodes", [])
            print(
                f"  - 후보#{i}: 거리={dist_m:.3f}m 스냅(lon,lat)={loc} 이름={name} 노드={nodes}"
            )

    # 대표 후보(가장 가까운 1개)
    best = within[0]
    snapped_lon, snapped_lat = best.get("location", [None, None])
    snap_dist_m = float(best.get("distance", 1e18))
    nodes = best.get("nodes", []) or []
    # docs 예시처럼 0 값은 제외
    node_ids = [int(n) for n in nodes if isinstance(n, int) and n > 0]

    print(
        f"\n[스냅] 대표 후보: 거리={snap_dist_m:.3f}m 스냅(lat,lon)={snapped_lat},{snapped_lon}"
    )

    # 2) Overpass로 OSM 태그 확인
    print(f"\n[OVERPASS] 엔드포인트={args.overpass}")
    if node_ids:
        print(f"[OVERPASS] OSRM waypoint.nodes에서 받은 node id: {_fmt_list(node_ids)}")
    else:
        print("[OVERPASS] 유효한 node id 없음 → around() fallback만 사용")

    ways_from_nodes: List[Dict[str, Any]] = []
    ways_from_around: List[Dict[str, Any]] = []

    if node_ids:
        q = build_overpass_parent_ways_query(node_ids)
        if args.debug:
            print("\n[OVERPASS] 쿼리(노드 기반 → parent ways)")
            print(q)
        try:
            op = overpass_query(s, args.overpass, q)
            if args.debug:
                print("\n[디버그][Overpass 원본(노드 기반)]")
                print(_pretty(op))
            for el in op.get("elements", []) or []:
                if el.get("type") == "way":
                    ways_from_nodes.append(el)
        except Exception as e:
            print(f"\n[경고] 노드 기반 Overpass 조회 실패: {e}")

    # fallback: 스냅 좌표 주변 반경 조회(고가/지하 등 동시 존재 확인에 유용)
    if snapped_lat is not None and snapped_lon is not None:
        q2 = build_overpass_around_ways_query(float(snapped_lat), float(snapped_lon), args.radius_m)
        if args.debug:
            print("\n[OVERPASS] 쿼리(스냅 지점 around fallback)")
            print(q2)
        try:
            op2 = overpass_query(s, args.overpass, q2)
            if args.debug:
                print("\n[디버그][Overpass 원본(around)]")
                print(_pretty(op2))
            for el in op2.get("elements", []) or []:
                if el.get("type") == "way":
                    ways_from_around.append(el)
        except Exception as e:
            print(f"\n[경고] around 기반 Overpass 조회 실패: {e}")

    ways = ways_from_nodes + ways_from_around
    print(f"[OVERPASS] ways: 노드기반={len(ways_from_nodes)}, 반경기반={len(ways_from_around)}")

    if not ways:
        print("\n[결과] Overpass에서 highway way를 찾지 못함(태그 조회 실패).")
        return 2

    # way_id 기준 중복 제거
    uniq: Dict[int, Dict[str, Any]] = {}
    for w in ways:
        wid = w.get("id")
        if isinstance(wid, int):
            uniq[wid] = w

    print(f"\n[결과] highway way 수집 완료: {len(uniq)}개(중복 제거 후)")

    summaries = [summarize_way(w) for w in uniq.values()]

    # layer 분포 확인
    layer_vals: Dict[str, int] = {}
    for ssum in summaries:
        lv = ssum.get("layer")
        key = str(lv) if lv is not None else "(none)"
        layer_vals[key] = layer_vals.get(key, 0) + 1

    print("\n[레이어 분포]")
    for k, v in sorted(layer_vals.items(), key=lambda x: x[0]):
        print(f"  - layer={k}: {v}개")

    # 상세 출력 (layer/bridge/tunnel 중심)
    print("\n[도로 상세]")
    sorted_summaries = sorted(
        summaries,
        key=lambda x: (
            parse_layer(x.get("layer")) if parse_layer(x.get("layer")) is not None else 999,
            str(x.get("highway")),
            str(x.get("ref")),
            str(x.get("name")),
        ),
    )
    print(_build_ways_table(sorted_summaries, snapped_lat, snapped_lon))
    print("[참고] '중심거리(m)'는 Overpass center(bbox 중심) 기준 거리이며, 실제 선형 거리와 다를 수 있습니다.")
    if args.debug:
        for ssum in sorted_summaries:
            print(_pretty(ssum))

    print("\n[완료] 위 출력에서 동일 반경 내 서로 다른 layer/bridge/tunnel 조합이 있는지 확인하면 됩니다.")
    print("      (예: layer=1 & bridge=yes 가 고가, layer=0(또는 없음)이 하부도로일 가능성)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
사용:
    python road_disruption/scripts/compare_routes.py \
        --normal normal --collapsed collapsed \
        --spec config/cuts/daejeon_wonsinheung_bridge.yaml \
        --label "과학관→건양대병원"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis.cli import slug
from roaddis.cutspec import CutSpec
from roaddis.paths import DATA_DIR, PKG_DIR, data_subdir
from roaddis.way_index import WayIndex

# 검증된 색 (dataviz 검증기: 대비 ≥3:1, 색맹 분리 ΔE 13.0, 일반시야 ΔE 16.3 — 전부 PASS)
C_NORMAL = "#2a78d6"      # 범주형 슬롯1 파랑 — 일반 도로망
C_COLLAPSED = "#0de87a"   # 범주형 슬롯7 보라 — 도로 붕괴
C_REMOVED = "#ff0000"     # status critical  — 삭제된 구간
C_INK = "#0b0b0b"
C_MUTED = "#52514e"


def load_dump(tag_or_path: str) -> dict:
    """dump_route.py 가 저장한 JSON 읽기. tag 만 줘도 data/routes 에서 찾는다."""
    p = Path(tag_or_path)
    if not p.suffix:
        p = DATA_DIR / "routes" / f"{tag_or_path}.json"
    elif not p.is_absolute() and not p.exists():
        p = PKG_DIR / tag_or_path
    if not p.exists():
        sys.exit(f"파일 없음: {p}\n  dump_route.py 를 먼저 실행하세요.")
    return json.loads(p.read_text(encoding="utf-8"))


def parse(dump: dict, which: str) -> dict:
    """저장된 응답 → 비교에 필요한 값만."""
    req, res = dump["request"], dump["response"]
    out = {
        "port": req["port"],
        "start": (req["from"]["lat"], req["from"]["lon"]),
        "end": (req["to"]["lat"], req["to"]["lon"]),
        "code": res.get("code"),
        "ok": res.get("code") == "Ok" and bool(res.get("routes")),
    }
    if not out["ok"]:
        return out
    rt = res["routes"][0]
    ann = rt["legs"][0].get("annotation", {})
    out.update({
        "distance_km": rt["distance"] / 1000.0,
        "duration_min": rt["duration"] / 60.0,
        "weight": rt.get("weight"),
        "nodes": ann.get("nodes") or [],
        "geometry": rt.get("geometry"),
        "steps": rt["legs"][0].get("steps", []),
        "datasources": ann.get("datasources") or [],
    })
    if not out["geometry"]:
        print(f"      [주의] {which} JSON 에 geometry 가 없습니다. "
              f"dump_route.py 를 overview=full 로 다시 받으세요.")
    return out


def pairs_of(nodes) -> set:
    if not nodes:
        return set()
    a = np.asarray(nodes, dtype=np.int64)
    return set(map(tuple, np.column_stack([a[:-1], a[1:]])))


def path_latlon(geometry):
    """OSRM geojson geometry([lon,lat] 배열) → [(lat,lon), ...]"""
    return [(c[1], c[0]) for c in geometry["coordinates"]] if geometry else []


TRIM_M = 6.0   # way 경계에서 양쪽으로 잘라낼 길이 — 이 틈 때문에 점선처럼 보인다


def _trim_piece(lonlat: np.ndarray, tr, trim_m: float = TRIM_M) -> list:
    """
    폴리라인 양 끝을 trim_m 만큼 잘라낸다.

    way 경계마다 틈이 생겨야 "어디서 way 가 바뀌는지"가 눈에 보인다.
    조각이 짧으면 통째로 사라지지 않게 잘라내는 양을 줄인다.
    """
    x, y = tr.transform(lonlat[:, 0], lonlat[:, 1])
    x, y = np.asarray(x, float), np.asarray(y, float)
    seg = np.hypot(np.diff(x), np.diff(y))
    total = seg.sum()
    if total <= 0:
        return [[round(float(lonlat[0][1]), 6), round(float(lonlat[0][0]), 6)]]
    trim = min(trim_m, total * 0.3)
    s = np.concatenate([[0.0], np.cumsum(seg)])          # 시작점부터의 누적 거리
    want = np.clip(np.linspace(trim, total - trim, max(2, len(s))), 0, total)
    xi = np.interp(want, s, x)
    yi = np.interp(want, s, y)
    lon, lat = tr.transform(xi, yi, direction="INVERSE")
    return [[round(float(b), 6), round(float(a), 6)] for a, b in zip(lon, lat)]


def used_pieces(idx, pos: int, route_pairs: set, tr) -> list:
    """
    way 안에서 경로가 실제로 지난 세그먼트만 골라, 연속 구간별 폴리라인으로 만든다.

    way 전체를 그리면 경로가 지나지 않은 부분까지 나온다. 실측으로 일부 세그먼트만
    쓴 way 가 43개 중 8개였고, 경로 밖으로 뻗은 길이가 약 2,718m(경로의 26.2%)였다.
    way 112551501(원신흥로)은 세그먼트 15개 중 1개만 지났는데 962m 전체가 그려졌다.
    """
    s, e = int(idx.node_off[pos]), int(idx.node_off[pos + 1])
    n = idx.node_id[s:e]
    c = np.column_stack([idx.node_lon[s:e], idx.node_lat[s:e]])

    used = np.array([(int(n[k]), int(n[k + 1])) in route_pairs
                     or (int(n[k + 1]), int(n[k])) in route_pairs
                     for k in range(len(n) - 1)], dtype=bool)
    pieces = []
    k = 0
    while k < len(used):
        if not used[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(used) and used[j + 1]:
            j += 1
        pieces.append(_trim_piece(c[k:j + 2], tr))       # 연속 구간 [k..j]
        k = j + 1
    return pieces


def load_route_ways(normal_tag: str, ways_arg, idx, nm) -> list:
    """
    route_ways.py 가 만든 <tag>_ways.json 을 읽고, 각 way 에 대해 경로가 실제로
    지난 구간(pieces)을 계산해 붙인다.

    파일이 없으면 경로 node 열에서 즉석 역추적한다.
    """
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)

    p = None
    if ways_arg:
        p = Path(ways_arg)
        if not p.suffix:
            p = DATA_DIR / "routes" / f"{ways_arg}.json"
    else:
        cand = DATA_DIR / "routes" / f"{normal_tag}_ways.json"
        if cand.exists():
            p = cand

    route_pairs = pairs_of(nm["nodes"])
    id2pos = {int(v): i for i, v in enumerate(idx.tags["way_id"].to_numpy())}
    t = idx.tags
    out = []

    if p and p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        print(f"      way 목록: {p.name} ({len(data['ways'])}개)")
        src = [(id2pos.get(int(w["way_id"])), w) for w in data["ways"]]
    else:
        r = idx.ways_on_route(nm["nodes"])
        print(f"      way 목록: (파일 없음 → 즉석 역추적, {len(r['positions'])}개)")
        src = []
        for pos, u, tot in zip(r["positions"], r["used"], r["total"]):
            row = t.iloc[int(pos)]
            src.append((int(pos), {
                "way_id": int(row.way_id),
                "name": None if str(row["name"]) == "nan" else str(row["name"]),
                "highway": str(row.highway),
                "segments_used": int(u), "segments_total": int(tot),
                "length_m": None,
            }))

    for pos, w in src:
        if pos is None:
            continue
        out.append({**w, "pieces": used_pieces(idx, int(pos), route_pairs, tr)})
    return out


def draw_map(spec, idx, removed_pos, label, nm, cl, route_ways, stats, out_path):
    """
    Leaflet 로 4개 레이어를 독립 토글 가능한 지도를 만든다.

    folium 을 쓰지 않는 이유: 레이어 토글 UI 와 범례를 직접 제어해야 하고,
    way 목록처럼 folium 의 FeatureGroup 으로는 다루기 번거로운 것을 그린다.

    레이어
      1 일반 경로        실선 파랑
      2 붕괴 경로        파선 보라  (색 말고 형태로도 구분 — 색만으로 식별하지 않는다)
      3 붕괴시킨 도로    굵은 빨강 + ✕
      4 경로 구성 way    점선 + way 경계 점 + 도로명 라벨 (기본 꺼짐)
    """
    nm_pts = path_latlon(nm.get("geometry"))
    cl_pts = path_latlon(cl.get("geometry"))
    pts = nm_pts + cl_pts
    if not pts:
        print("      [주의] 그릴 좌표가 없어 지도를 건너뜁니다.")
        return

    lat = [p[0] for p in pts]
    lon = [p[1] for p in pts]
    t = idx.tags

    removed = []
    for w in removed_pos:
        c = idx.coords_of(int(w))
        removed.append({
            "way_id": int(t.way_id.iloc[int(w)]),
            "name": None if str(t["name"].iloc[int(w)]) == "nan"
                    else str(t["name"].iloc[int(w)]),
            "coords": [[round(float(la), 6), round(float(lo), 6)] for lo, la in c],
        })

    data = {
        "label": label,
        "spec": spec.name,
        "bounds": [[min(lat), min(lon)], [max(lat), max(lon)]],
        "start": list(nm["start"]),
        "end": list(nm["end"]),
        "normal": {"ok": nm["ok"], "port": nm["port"], "coords": nm_pts,
                   "distance_km": nm.get("distance_km"),
                   "duration_min": nm.get("duration_min")},
        "collapsed": {"ok": cl["ok"], "port": cl["port"], "coords": cl_pts,
                      "code": cl.get("code"),
                      "distance_km": cl.get("distance_km"),
                      "duration_min": cl.get("duration_min")},
        "removed": removed,
        "ways": route_ways,
        "stats": stats,
        "colors": {"normal": C_NORMAL, "collapsed": C_COLLAPSED,
                   "removed": C_REMOVED, "ink": C_INK, "muted": C_MUTED},
    }

    html = _HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    Path(out_path).write_text(html, encoding="utf-8")


_HTML_TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>도로 붕괴 경로 비교</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root{ --surface:#fcfcfb; --border:#d5d4cf; --ink:#0b0b0b; --muted:#52514e; }
  html,body{margin:0;height:100%;font:13px/1.55 system-ui,-apple-system,sans-serif;
            color:var(--ink)}
  #map{position:absolute;inset:0}
  .panel{position:absolute;z-index:1000;background:color-mix(in srgb,var(--surface) 94%,transparent);
         border:1px solid var(--border);border-radius:10px;padding:12px 14px;
         box-shadow:0 2px 10px #0000001f;backdrop-filter:blur(3px)}
  #ctl{top:14px;right:14px;min-width:232px}
  #info{bottom:14px;left:14px;max-width:360px}
  h1{font-size:14px;margin:0 0 2px}
  .sub{color:var(--muted);margin-bottom:9px}
  label{display:flex;align-items:center;gap:9px;padding:4px 0;cursor:pointer}
  label input{width:15px;height:15px;margin:0;cursor:pointer;flex:none}
  .swatch{flex:none}
  table{border-collapse:collapse;margin-top:9px;width:100%}
  th,td{padding:2px 0;text-align:right;font-variant-numeric:tabular-nums}
  th:first-child,td:first-child{text-align:left;padding-right:12px}
  thead th{color:var(--muted);font-weight:500;border-bottom:1px solid var(--border)}
  .verdict{margin-top:9px;padding-top:8px;border-top:1px solid var(--border);
           font-weight:600}
  .waylabel{font:600 11px/1.2 system-ui;color:#0b0b0b;
            text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff;white-space:nowrap}
  .xmark{font:700 19px/1 system-ui;text-shadow:0 0 4px #fff,0 0 4px #fff}
</style></head><body>
<div id="map"></div>
<div id="ctl" class="panel">
  <h1>레이어</h1>
  <div class="sub" id="ctl-sub"></div>
  <label><input type="checkbox" id="c1" checked>
    <svg class="swatch" width="26" height="6"><line x1="0" y1="3" x2="26" y2="3"
      stroke-width="5" id="s1"/></svg><span>1. 일반 경로</span></label>
  <label><input type="checkbox" id="c2" checked>
    <svg class="swatch" width="26" height="6"><line x1="0" y1="3" x2="26" y2="3"
      stroke-width="5" stroke-dasharray="9,5" id="s2"/></svg><span>2. 붕괴 경로</span></label>
  <label><input type="checkbox" id="c3" checked>
    <svg class="swatch" width="26" height="8"><line x1="0" y1="4" x2="26" y2="4"
      stroke-width="7" id="s3"/></svg><span>3. 붕괴시킨 도로</span></label>
  <!-- 붕괴시킨 도로는 굵은 실선. ✕ 마커는 선을 가려서 뺐다 -->
  <label><input type="checkbox" id="c4">
    <svg class="swatch" width="26" height="6"><line x1="0" y1="3" x2="26" y2="3"
      stroke-width="4" stroke-dasharray="7,4" id="s4"/></svg><span>4. way 단위 구간</span></label>
</div>
<div id="info" class="panel"></div>
<script>
const D = __DATA__;
const C = D.colors;
document.getElementById('s1').setAttribute('stroke', C.normal);
document.getElementById('s2').setAttribute('stroke', C.collapsed);
document.getElementById('s3').setAttribute('stroke', C.removed);
document.getElementById('s4').setAttribute('stroke', C.ink);
document.getElementById('ctl-sub').textContent = D.spec;

const map = L.map('map', {zoomControl:true});
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {maxZoom:20, attribution:'&copy; OpenStreetMap, &copy; CARTO'}).addTo(map);
map.fitBounds(D.bounds, {padding:[40,40]});

// 1 일반 경로
const g1 = L.layerGroup();
if (D.normal.ok) L.polyline(D.normal.coords, {color:C.normal, weight:6, opacity:.85})
  .bindTooltip(`일반 도로망 (포트 ${D.normal.port})<br>${D.normal.distance_km.toFixed(3)} km · ${D.normal.duration_min.toFixed(2)} 분`, {sticky:true})
  .addTo(g1);

// 2 붕괴 경로
const g2 = L.layerGroup();
if (D.collapsed.ok) L.polyline(D.collapsed.coords,
    {color:C.collapsed, weight:6, opacity:.9, dashArray:'12,7'})
  .bindTooltip(`붕괴 도로망 (포트 ${D.collapsed.port})<br>${D.collapsed.distance_km.toFixed(3)} km · ${D.collapsed.duration_min.toFixed(2)} 분`, {sticky:true})
  .addTo(g2);

// 3 붕괴시킨 도로 — 굵은 실선만 (✕ 마커 없음)
const g3 = L.layerGroup();
D.removed.forEach(w => {
  L.polyline(w.coords, {color:C.removed, weight:9, opacity:.95, lineCap:'round'})
    .bindTooltip(`붕괴: ${w.name ?? '(무명)'}<br>way ${w.way_id}`, {sticky:true}).addTo(g3);
});

// 4 경로 구성 way — 경로가 실제로 지난 구간만, way 경계에서 끊어 그린다
//   (way 를 통째로 그리면 지나지 않은 부분까지 나온다 — 실측 26.2% 초과)
const g4 = L.layerGroup();
D.ways.forEach(w => {
  const len = (w.length_m == null) ? '' : ` · way 전체 ${w.length_m.toFixed(0)} m`;
  const tip = `way ${w.way_id}${len}<br>${w.name ?? '(무명)'} · ${w.highway}`
            + `<br>지난 세그먼트 ${w.segments_used}/${w.segments_total}`;
  let longest = null, longestLen = -1;
  w.pieces.forEach(pc => {
    if (pc.length < 2) return;
    L.polyline(pc, {color:C.ink, weight:4, opacity:.85, lineCap:'butt'})
      .bindTooltip(tip, {sticky:true}).addTo(g4);
    if (pc.length > longestLen) { longestLen = pc.length; longest = pc; }
  });
  if (w.name && longest) {
    const mid = longest[Math.floor(longest.length/2)];
    L.marker(mid, {icon: L.divIcon({className:'', iconSize:[0,0],
      html:`<div class="waylabel">${w.name}</div>`})}).addTo(g4);
  }
});

L.marker(D.start).bindTooltip('출발', {permanent:false}).addTo(map);
L.marker(D.end).bindTooltip('도착', {permanent:false}).addTo(map);

const groups = {c1:g1, c2:g2, c3:g3, c4:g4};
const defaults = {c1:true, c2:true, c3:true, c4:false};
Object.entries(groups).forEach(([id,g]) => {
  const box = document.getElementById(id);
  box.checked = defaults[id];
  if (box.checked) g.addTo(map);
  box.addEventListener('change', () => box.checked ? g.addTo(map) : map.removeLayer(g));
});

// 정보 패널
const s = D.stats, N = D.normal, L2 = D.collapsed;
let rows = '';
if (N.ok && L2.ok) {
  const dd = L2.duration_min - N.duration_min, dk = L2.distance_km - N.distance_km;
  const pct = (100*dd/N.duration_min);
  rows = `<tr><td>소요 (분)</td><td>${N.duration_min.toFixed(2)}</td><td>${L2.duration_min.toFixed(2)}</td><td>${dd>=0?'+':''}${dd.toFixed(2)}</td></tr>
          <tr><td>거리 (km)</td><td>${N.distance_km.toFixed(3)}</td><td>${L2.distance_km.toFixed(3)}</td><td>${dk>=0?'+':''}${dk.toFixed(3)}</td></tr>
          <tr><td>변화율</td><td></td><td></td><td>${pct>=0?'+':''}${pct.toFixed(1)}%</td></tr>`;
} else if (N.ok) {
  rows = `<tr><td>소요 (분)</td><td>${N.duration_min.toFixed(2)}</td><td colspan="2">도달 불가 (${L2.code})</td></tr>`;
}
document.getElementById('info').innerHTML = `
  <h1>${D.label}</h1>
  <div class="sub">붕괴시킨 도로 ${D.removed.length}개 · 경로 구성 way ${D.ways.length}개</div>
  <table><thead><tr><th></th><th>일반</th><th>붕괴</th><th>Δ</th></tr></thead>
  <tbody>${rows}</tbody></table>
  <div class="verdict">${s.verdict}</div>
  <div class="sub" style="margin:4px 0 0">일반 경로가 붕괴 구간을 지남: ${s.used ? '예' : '아니오'}
  &nbsp;·&nbsp; 붕괴 경로 누출: ${s.leak}개</div>`;
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="저장된 경로 JSON 2개 비교 + 지도 (OSRM 질의 없음)")
    ap.add_argument("--normal", required=True, help="일반 도로망 JSON 의 tag 또는 경로")
    ap.add_argument("--collapsed", required=True, help="붕괴 도로망 JSON 의 tag 또는 경로")
    ap.add_argument("--spec", required=True, help="삭제 구간 검증용 CutSpec")
    ap.add_argument("--label", default="compare",
                    help="파일명에 쓰이므로 영어로 (예: science_to_konyang)")
    ap.add_argument("--ways", default=None,
                    help="route_ways.py 가 만든 <tag>_ways.json. "
                         "기본은 <normal>_ways.json 을 찾고, 없으면 즉석 역추적")
    ap.add_argument("--no-map", action="store_true")
    args = ap.parse_args()

    nm = parse(load_dump(args.normal), "normal")
    cl = parse(load_dump(args.collapsed), "collapsed")

    sp = Path(args.spec)
    if not sp.is_absolute() and not sp.exists():
        sp = PKG_DIR / args.spec
    spec = CutSpec.from_yaml(sp)
    idx = WayIndex.load(DATA_DIR / "way_index" / spec.region)
    seg_mask = spec.resolve(idx)
    removed_pos = np.unique(idx.segment_way_pos()[seg_mask])
    cut_pairs = set(map(tuple, idx.directed_pairs(seg_mask)))

    print(f"기록: {args.normal}(포트 {nm['port']}) vs "
          f"{args.collapsed}(포트 {cl['port']})")
    print(f"CutSpec '{spec.name}'  삭제 way {len(removed_pos)}개 / "
          f"node 쌍 {len(cut_pairs):,}개")

    # 두 기록이 같은 출발지·목적지인지 — 다르면 비교가 성립하지 않는다
    if nm["start"] != cl["start"] or nm["end"] != cl["end"]:
        sys.exit(f"출발지·목적지가 다릅니다.\n"
                 f"  일반 {nm['start']} → {nm['end']}\n"
                 f"  붕괴 {cl['start']} → {cl['end']}")
    print(f"출발 {nm['start']}  도착 {nm['end']}\n")

    if not nm["ok"]:
        sys.exit(f"일반 도로망 경로가 없습니다: {nm['code']}")

    used = pairs_of(nm["nodes"]) & cut_pairs
    print(f"일반 경로: {nm['distance_km']:>7.3f} km  {nm['duration_min']:>7.2f} 분  "
          f"node {len(nm['nodes'])}개")
    if cl["ok"]:
        leak = pairs_of(cl["nodes"]) & cut_pairs
        dt = cl["duration_min"] - nm["duration_min"]
        dk = cl["distance_km"] - nm["distance_km"]
        print(f"붕괴 경로: {cl['distance_km']:>7.3f} km  {cl['duration_min']:>7.2f} 분  "
              f"node {len(cl['nodes'])}개")
        print(f"변화     : {dk:>+7.3f} km  {dt:>+7.2f} 분  "
              f"({100*dt/nm['duration_min']:+.1f}%)")
    else:
        leak = set()
        print(f"붕괴 경로: 도달 불가 ({cl['code']})")

    print()
    print(f"일반 경로가 삭제 구간을 지났는가 : "
          f"{'예 ' + str(len(used)) + '개 세그먼트' if used else '아니오'}")
    print(f"붕괴 경로에 삭제 구간이 남았는가 : "
          f"{'예 ' + str(len(leak)) + '개 (누출!)' if leak else '아니오 (정상)'}")

    print()
    if not used:
        verdict = "무효 — 일반 경로가 삭제 구간을 지나지 않았습니다"
    elif leak:
        verdict = f"실패 — 삭제한 도로를 {len(leak)}개 구간 통과합니다"
    elif not cl["ok"]:
        verdict = "통과 — 삭제로 도달이 불가능해졌습니다"
    else:
        verdict = "통과 — 삭제한 도로가 없는 것처럼 우회합니다"
    print(f"판정: {verdict}")
    if not used:
        print("      다른 구간을 끊거나 출발지·목적지를 바꾸세요.")

    if not args.no_map:
        route_ways = load_route_ways(args.normal, args.ways, idx, nm)
        stats = {"verdict": verdict, "used": len(used), "leak": len(leak)}
        out = data_subdir("routes") / f"{slug(args.label)}_{slug(spec.name)}.html"
        draw_map(spec, idx, removed_pos, args.label, nm, cl, route_ways, stats, out)
        print()
        print(f"지도: {out}")
        print(f"  http://localhost:8010/routes/{out.name}")


if __name__ == "__main__":
    main()

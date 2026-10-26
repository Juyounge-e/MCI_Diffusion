"""
inspect_cut_ways.py 결과 → Leaflet 지도 HTML 하나.

    python road_disruption/scripts/plot_inspect_map.py --label artcenter

way 를 드롭다운으로 골라 보고, 레이어 4개를 독립 토글한다.
    ① 기준선 경로     파란 실선
    ② 우회 경로       주황 실선
    ③ 끊은 도로       빨간 실선
    ④ way 단위 구분   경로가 지난 way 를 하나씩 끊어 그린 점선

저장된 JSON 만 읽는다 — OSRM 에 다시 질의하지 않는다. 다시 질의하면 화면과
기록이 어긋나 실험 근거로 쓸 수 없다.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis.cli import slug
from roaddis.paths import DATA_DIR, data_subdir

HTML = """<meta charset="utf-8">
<title>%(title)s</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%%;font-family:"Malgun Gothic",system-ui,sans-serif}
  #map{position:absolute;inset:0}
  #panel{position:absolute;top:10px;left:10px;z-index:1000;background:#fff;
         padding:12px 14px;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.25);
         max-width:390px;font-size:13px;line-height:1.55}
  #panel h3{margin:0 0 8px;font-size:14px}
  select{width:100%%;padding:5px;font-size:12px;font-family:inherit}
  table{border-collapse:collapse;margin-top:8px;font-size:12px;width:100%%}
  td{padding:2px 6px 2px 0}
  td.k{color:#6d7d8a}
  .num{text-align:right;font-variant-numeric:tabular-nums}
  .bad{color:#d03b3b;font-weight:700}
  .good{color:#2a78d6;font-weight:700}
  #legend{margin-top:9px;padding-top:8px;border-top:1px solid #dde4ea;font-size:12px}
  #legend span{display:inline-block;width:22px;height:3px;vertical-align:middle;
               margin-right:6px}
</style>
<div id="map"></div>
<div id="panel">
  <h3>%(title)s</h3>
  <select id="sel"></select>
  <div id="info"></div>
  <div id="legend">
    <div><span style="background:#2a78d6"></span>기준선 경로</div>
    <div><span style="background:#e08214"></span>우회 경로</div>
    <div><span style="background:#d03b3b;height:4px"></span>끊은 도로</div>
    <div><span style="background:#4a3aa7"></span>way 단위(점선)</div>
  </div>
</div>
<script>
const D = %(data)s;
const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19, attribution:'&copy; OpenStreetMap'}).addTo(map);

const sel = document.getElementById('sel');
D.items.forEach((it,i)=>{
  const o = document.createElement('option');
  o.value = i;
  o.textContent = `${it.kind}  ${it.name||'(무명)'} ${it.way_id}  Δpdr ${it.d_pdr>0?'+':''}${it.d_pdr.toFixed(5)}`;
  sel.appendChild(o);
});

let layers = [];
function clear(){ layers.forEach(l=>map.removeLayer(l)); layers = []; }
function line(coords, opts){ const l = L.polyline(coords, opts).addTo(map); layers.push(l); return l; }

// geojson 은 [lon,lat] 순서다. Leaflet 은 [lat,lon] 이라 뒤집어야 한다.
function flip(g){ return g.coordinates.map(c=>[c[1],c[0]]); }

// way 를 하나씩 끊어 그리면 경계가 틈으로 보여 점선처럼 읽힌다.
// 실제로 통과한 세그먼트 수만큼만 그린다 — way 전체를 그리면
// 경로가 지나지 않은 부분까지 칠해져 과대 표시된다.
function drawWays(ways){
  ways.forEach(w=>{
    const n = Math.max(2, Math.min(w.coords.length, w.segments_used + 1));
    const part = w.coords.slice(0, n);
    if (part.length >= 2)
      line(part, {color:'#4a3aa7', weight:5, opacity:.55, dashArray:'1,7',
                  lineCap:'round'});
  });
}

function show(i){
  clear();
  const it = D.items[i];
  const info = document.getElementById('info');
  if (it.status !== 'ok'){ info.innerHTML = `<p class="bad">${it.status}</p>`; return; }

  drawWays(it.baseline.ways);
  const b = line(flip(it.baseline.geometry), {color:'#2a78d6', weight:4, opacity:.9});
  line(flip(it.collapsed.geometry), {color:'#e08214', weight:4, opacity:.9,
                                     dashArray:'10,6'});
  line(it.cut_way_coords, {color:'#d03b3b', weight:7, opacity:.95});
  L.circleMarker(it.from, {radius:6, color:'#0f1720', fillColor:'#fff',
    fillOpacity:1, weight:2}).addTo(map).bindTooltip('출발').openTooltip();
  layers.push(L.circleMarker(it.to, {radius:6, color:'#0f1720',
    fillColor:'#d03b3b', fillOpacity:1, weight:2}).addTo(map).bindTooltip('도착'));

  const cls = it.d_pdr > 0 ? 'bad' : 'good';
  info.innerHTML = `
  <table>
    <tr><td class="k">구간</td><td>${it.target_kind}<br>${it.target}</td></tr>
    <tr><td class="k">끊은 도로</td><td>${it.name||'(무명)'} · ${it.highway} · ${it.length_m.toFixed(0)}m</td></tr>
    <tr><td class="k">공유 구간</td><td>${it.n_routes}개</td></tr>
    <tr><td class="k">Δpdr</td><td class="${cls} num">${it.d_pdr>0?'+':''}${it.d_pdr.toFixed(5)}</td></tr>
    <tr><td class="k">거리</td><td class="num">${it.baseline.distance_km.toFixed(3)} → ${it.collapsed.distance_km.toFixed(3)} km (${it.delta_km>0?'+':''}${it.delta_km.toFixed(3)})</td></tr>
    <tr><td class="k">시간</td><td class="num">${it.baseline.duration_min.toFixed(2)} → ${it.collapsed.duration_min.toFixed(2)} 분 (${it.delta_min>0?'+':''}${it.delta_min.toFixed(2)})</td></tr>
    <tr><td class="k">지나간 way</td><td class="num">${it.baseline.n_ways} → ${it.collapsed.n_ways}개</td></tr>
  </table>`;
  map.fitBounds(b.getBounds().pad(0.15));
}
sel.onchange = ()=>show(+sel.value);
show(0);
</script>
"""


def main():
    ap = argparse.ArgumentParser(description="inspect JSON → Leaflet 지도")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = DATA_DIR / "routes" / f"{slug(args.label)}_inspect.json"
    if not src.exists():
        sys.exit(f"파일 없음: {src}\n  inspect_cut_ways.py 를 먼저 실행하세요.")
    d = json.loads(src.read_text(encoding="utf-8"))
    d["items"] = [x for x in d["items"] if x.get("status") == "ok"]
    if not d["items"]:
        sys.exit("성공한 항목이 없습니다.")

    out = Path(args.out) if args.out else (
        data_subdir("candidates") / f"{slug(args.label)}_inspect.html")
    out.write_text(HTML % {
        "title": f"{args.label} — 끊은 도로별 우회 경로",
        "data": json.dumps(d, ensure_ascii=False),
    }, encoding="utf-8")
    print(f"저장: {out}  ({out.stat().st_size/1024:.0f} KB, 항목 {len(d['items'])}개)")
    print(f"보기: python road_disruption/scripts/serve_maps.py --port 8010")


if __name__ == "__main__":
    main()

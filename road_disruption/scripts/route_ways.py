"""
    python road_disruption/scripts/route_ways.py --route normal --region daejeon

해당경로에서 붕괴 도로 후보를 뽑기 위한 스크립트 
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis.osm_tags import KEEP_TAGS
from roaddis.paths import DATA_DIR, PKG_DIR
from roaddis.way_index import WayIndex

ROAD_CRS = "EPSG:5186"
WGS84 = "EPSG:4326"


def load_dump(tag_or_path: str) -> tuple:
    p = Path(tag_or_path)
    if not p.suffix:
        p = DATA_DIR / "routes" / f"{tag_or_path}.json"
    elif not p.is_absolute() and not p.exists():
        p = PKG_DIR / tag_or_path
    if not p.exists():
        sys.exit(f"파일 없음: {p}\n  dump_route.py 를 먼저 실행하세요.")
    return json.loads(p.read_text(encoding="utf-8")), p


def main():
    ap = argparse.ArgumentParser(description="경로 JSON → 지나간 way 목록")
    ap.add_argument("--route", required=True, help="dump_route.py 의 tag 또는 경로")
    ap.add_argument("--region", required=True)
    ap.add_argument("--out", default=None, help="기본: <tag>_ways.json")
    args = ap.parse_args()

    dump, src = load_dump(args.route)
    res = dump["response"]
    if res.get("code") != "Ok" or not res.get("routes"):
        sys.exit(f"경로가 없는 기록입니다: code={res.get('code')}")

    rt = res["routes"][0]
    nodes = rt["legs"][0].get("annotation", {}).get("nodes")
    if not nodes:
        sys.exit("annotation.nodes 가 없습니다. dump_route.py 로 다시 받으세요.")

    idx = WayIndex.load(DATA_DIR / "way_index" / args.region)
    t = idx.tags
    r = idx.ways_on_route(nodes)
    pos, used, total = r["positions"], r["used"], r["total"]

    print(f"기록: {src.name}  (포트 {dump['request']['port']})")
    print(f"경로: {rt['distance']/1000:.3f} km  {rt['duration']/60:.2f} 분  "
          f"node {len(nodes)}개")
    print(f"세그먼트 {r['n_route_seg']}개 중 인덱스에서 찾은 것 {r['n_matched']}개", end="")
    if r["n_matched"] != r["n_route_seg"]:
        miss = r["n_route_seg"] - r["n_matched"]
        print(f"  [주의] {miss}개 누락 — 인덱스에 없는 도로"
              f"(highway 태그 없음, 페리 등)를 지났을 수 있습니다")
    else:
        print("  (전부 일치)")
    print(f"지나간 way: {len(pos)}개\n")

    # 길이 계산
    tr = Transformer.from_crs(WGS84, ROAD_CRS, always_xy=True)

    def wlen(w):
        c = idx.coords_of(int(w))
        x, y = tr.transform(c[:, 0], c[:, 1])
        return float(np.hypot(np.diff(x), np.diff(y)).sum())

    def tag(row, key):
        """결측(NaN)은 None 으로. JSON 에 NaN 을 쓸 수 없다."""
        v = row[key]
        return None if v is None or str(v) == "nan" else str(v)

    ways = []
    for w, u, tot in zip(pos, used, total):
        row = t.iloc[int(w)]
        c = idx.coords_of(int(w))
        rec = {
            "way_id": int(row.way_id),
            "is_drive": bool(row.is_drive),
            "length_m": round(wlen(w), 1),
            "segments_used": int(u),
            "segments_total": int(tot),
            "center": [round(float(c[:, 1].mean()), 6), round(float(c[:, 0].mean()), 6)],
        }
        # KEEP_TAGS 에 있는 태그는 전부 내보낸다. 인덱스가 보존하는 이유가 있는
        # 태그들이고, 여기서 임의로 골라내면 하류에서 쓸 수 없다.
        # (layer 는 -1=지하차도 / 1=고가 구분에 필요하다)
        for k in KEEP_TAGS:
            rec[k] = tag(row, k)
        ways.append(rec)
    ways.sort(key=lambda x: -x["length_m"])

    # 도로명으로 묶기 (이름 없으면 way 하나가 독립 그룹)
    groups = {}
    for w in ways:
        key = w["name"] or f"(무명) {w['way_id']}"
        g = groups.setdefault(key, {"way_ids": [], "length_m": 0.0, "highway": set()})
        g["way_ids"].append(w["way_id"])
        g["length_m"] += w["length_m"]
        g["highway"].add(w["highway"])
    for g in groups.values():
        g["length_m"] = round(g["length_m"], 1)
        g["highway"] = sorted(g["highway"])
    groups = dict(sorted(groups.items(), key=lambda kv: -kv[1]["length_m"]))

    print(f"{'도로명':<24}{'way':>5}{'길이m':>9}  등급")
    print("-" * 62)
    for k, g in groups.items():
        print(f"{k:<24}{len(g['way_ids']):>5}{g['length_m']:>9.0f}  "
              f"{','.join(g['highway'])}")

    n = len(groups)
    print(f"\n조합 실험 단위: 도로명 {n}개  →  전체 부분집합 {2**n - 1:,}가지")
    print(f"                way {len(ways)}개    →  전체 부분집합 "
          f"{2**len(ways) - 1:.3e}가지 (불가)")

    out = Path(args.out) if args.out else (
        DATA_DIR / "routes" / f"{src.stem}_ways.json")
    payload = {
        "source": {
            "route_json": src.name,
            "port": dump["request"]["port"],
            "from": dump["request"]["from"],
            "to": dump["request"]["to"],
            "region": args.region,
            "index": f"{args.region}_ways.npz",
        },
        "route": {
            "distance_km": round(rt["distance"] / 1000, 4),
            "duration_min": round(rt["duration"] / 60, 4),
            "n_nodes": len(nodes),
            "n_segments": r["n_route_seg"],
            "n_segments_matched": r["n_matched"],
        },
        "ways": ways,
        "groups": groups,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n저장: {out}  ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()

"""
OSRM 응답을 전부 JSON 으로 저장.

    # baseline
    python road_disruption/scripts/dump_route.py --port 5002 \
        --from 36.3757,127.3767 --to 36.306642,127.342613 --tag baseline

    # 차단본
    python road_disruption/scripts/dump_route.py --port 5010 \
        --from 36.3757,127.3767 --to 36.306642,127.342613 --tag cut

출력: road_disruption/data/routes/<tag>.json  (serve_maps.py 로 브라우저에서 열림)
      함께 <tag>_summary.txt 에 사람이 읽을 요약도 쓴다.
"""
import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis.cli import latlon
from roaddis.paths import data_subdir


def main():
    ap = argparse.ArgumentParser(description="OSRM 응답 전체 저장")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--from", dest="src", required=True, help="lat,lon")
    ap.add_argument("--to", dest="dst", required=True, help="lat,lon")
    ap.add_argument("--tag", required=True, help="출력 파일 이름")
    args = ap.parse_args()

    s, d = latlon(args.src), latlon(args.dst)
    url = (f"http://localhost:{args.port}/route/v1/driving/"
           f"{s[1]},{s[0]};{d[1]},{d[0]}")
    params = {
        "steps": "true",
        "annotations": "true",          # nodes,distance,duration,speed,datasources 전부
        "overview": "full",
        "geometries": "geojson",
        "continue_straight": "false",
        "radiuses": "250;250",
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()

    out_dir = data_subdir("routes")
    jpath = out_dir / f"{args.tag}.json"
    payload = {
        "request": {"url": url, "params": params,
                    "from": {"lat": s[0], "lon": s[1]},
                    "to": {"lat": d[0], "lon": d[1]},
                    "port": args.port},
        "response": data,
    }
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"저장: {jpath}  ({jpath.stat().st_size/1024:.1f} KB)")

    if data.get("code") != "Ok":
        print(f"[주의] code={data.get('code')} — 경로를 못 찾았습니다.")
        return

    rt = data["routes"][0]
    leg = rt["legs"][0]
    ann = leg["annotation"]
    lines = []
    lines.append(f"port {args.port}   {args.src} → {args.dst}")
    lines.append(f"거리 {rt['distance']/1000:.3f} km   "
                 f"소요 {rt['duration']/60:.2f} min   weight {rt['weight']:,.1f}")
    lines.append(f"node {len(ann['nodes'])}개   세그먼트 {len(ann['distance'])}개   "
                 f"steps {len(leg['steps'])}개   "
                 f"intersections {sum(len(x['intersections']) for x in leg['steps'])}개")
    ds = ann.get("datasources", [])
    if ds:
        from collections import Counter
        lines.append(f"datasource 분포 {dict(Counter(ds))}  (0=lua 프로파일)")
    lines.append("")
    lines.append("--- steps (내비 안내) ---")
    for i, st in enumerate(leg["steps"]):
        lines.append(f"{i:>3}  {st['name'] or '(무명)':<16} {st['distance']:>8.1f} m  "
                     f"{st['duration']:>6.1f} s  {st['maneuver']['type']:<14} "
                     f"교차로 {len(st['intersections'])}개")
    # 세그먼트를 앞에서 30개 나열하는 것은 임의로 자른 창이라 이상값을 못 잡는다.
    # 대신 219개 전체를 훑어 비정상만 보고한다.
    lines.append("")
    lines.append("--- 세그먼트 이상값 검사 (전체 훑음) ---")
    dur = [float(x) for x in ann.get("duration", [])]
    spd = [float(x) for x in ann.get("speed", [])]
    if dur and spd:
        lines.append(f"소요  최소 {min(dur):.2f} s   최대 {max(dur):.2f} s   "
                     f"합계 {sum(dur):.1f} s")
        lines.append(f"속도  최소 {min(spd):.2f} m/s  최대 {max(spd):.2f} m/s")

        problems = []
        # speed 0 은 통행 불가로 의도된 세그먼트를 지나갔다는 뜻 (way 삭제 방식에서는
        # 나올 수 없다). datasource != 0 은 외부 speed 파일이 적용됐다는 뜻.
        zero = [k for k, v in enumerate(spd) if v <= 0.01]
        if zero:
            problems.append(f"속도 0 인 세그먼트 {len(zero)}개 "
                            f"(node {ann['nodes'][zero[0]]}→{ann['nodes'][zero[0]+1]} 등)")
        huge = [k for k, v in enumerate(dur) if v > 600]
        if huge:
            problems.append(f"소요 600초 초과 세그먼트 {len(huge)}개 "
                            f"(최대 {max(dur):,.0f} s)")
        if ds and set(ds) - {0}:
            problems.append(f"datasource 가 0 이 아닌 세그먼트 "
                            f"{sum(1 for v in ds if v != 0)}개 — 외부 speed 파일 적용됨")

        if problems:
            lines.append("[이상]")
            for x in problems:
                lines.append(f"  - {x}")
        else:
            lines.append("이상 없음 (속도 0 없음, 비정상 소요 없음, datasource 전부 0)")

    spath = out_dir / f"{args.tag}_summary.txt"
    spath.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {spath}")
    print()
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()

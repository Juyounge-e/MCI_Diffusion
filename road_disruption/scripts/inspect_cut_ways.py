"""
선정한 way 를 하나씩 끊고, 그 way 때문에 가장 많이 밀린 구간의
기준선 경로 / 우회 경로를 함께 저장한다.

    python road_disruption/scripts/inspect_cut_ways.py \
        --list <inspect_list.json> --region daejeon --label artcenter \
        --scenario "MCI_ADV2/scenarios/exp_artcenter_osrm/(36.366605,127.383691)"

왜 필요한가:
    sweep 은 way 별 지표만 남긴다. "왜 그 도로가 영향이 컸나" 는 경로를 봐야
    안다. run_cut_sweep 이 반복마다 붕괴본을 지우므로 여기서 다시 빌드한다.

--list 항목마다 필요한 것:
    way_id, target(구간 상대 지점 이름), target_kind(center2site|site2hospital)

한 way 당:
    ① pbf 에서 way 삭제 → extract/partition/customize → 포트 기동   (약 9초)
    ② 기준선 OSRM 과 붕괴 OSRM 에 같은 구간을 질의 (annotations=nodes, geojson)
    ③ node 열을 WayIndex 로 역추적해 지나간 way 목록을 얻는다
    ④ 컨테이너·빌드 산출물 정리

산출물: data/routes/<label>_inspect.json  (지도 생성 스크립트의 입력)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis import osrm
from roaddis.cli import slug
from roaddis.paths import DATA_DIR, OSRM_DIR
from roaddis.pbf import write_pbf_without_ways
from roaddis.regions import get_region
from roaddis.way_index import WayIndex

PORT = 5031                       # sweep(5030) 과 겹치지 않게
CONTAINER = "osrm-inspect"
BUILDER = "osrm-inspect-build"
NAME_COL = "안전센터/소방서이름"


def pbf_rel(key: str) -> str:
    return f"cut_pbf/_insp_{key}.osm.pbf"


def teardown(key: str):
    osrm.remove_container(CONTAINER)
    osrm.remove_container(BUILDER)
    stem = Path(pbf_rel(key)).name.replace(".osm.pbf", "")
    for f in (OSRM_DIR / "cut_pbf").glob(f"{stem}*"):
        f.unlink(missing_ok=True)


def endpoints(scen: Path, cfg: dict) -> tuple:
    """구간 이름 → (lat, lon). 좌표 출처는 scenario_route_ways.py 와 동일."""
    pt = cfg["entity_info"]["patient"]
    site = (float(pt["latitude"]), float(pt["longitude"]))

    centers = {}
    for f in sorted((scen / "routes" / "center2site").glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))["meta"]
        lon, lat = m["center"]                       # 생성기는 [lon, lat]
        centers[m["name"]] = (float(lat), float(lon))

    hp = Path(cfg["entity_info"]["hospital"]["info_path"])
    if not hp.exists():
        hp = scen / "hospital_info.csv"
    hos = pd.read_csv(hp, encoding="utf-8-sig")
    hospitals = {str(r["요양기관명"]): (float(r["y좌표"]), float(r["x좌표"]))
                 for _, r in hos.iterrows()}
    return site, centers, hospitals


def ways_of(idx: WayIndex, nodes) -> list:
    """경로 node 열 → 지나간 way 요약 (지도에서 점선으로 그릴 재료)."""
    r = idx.ways_on_route(nodes)
    t = idx.tags
    out = []
    for pos, used, tot in zip(r["positions"], r["used"], r["total"]):
        row = t.iloc[int(pos)]
        c = idx.coords_of(int(pos))
        nm = row["name"]
        out.append({
            "way_id": int(row.way_id),
            "name": None if nm is None or str(nm) == "nan" else str(nm),
            "highway": str(row["highway"]),
            "segments_used": int(used),
            "segments_total": int(tot),
            # 지도에 그리려면 좌표가 필요하다. way 자체에는 좌표가 없으므로
            # 인덱스에서 꺼내 [lat, lon] 으로 뒤집어 담는다.
            "coords": [[round(float(y), 6), round(float(x), 6)] for x, y in c],
        })
    return out, r


def main():
    ap = argparse.ArgumentParser(description="선정 way 의 기준선/우회 경로 저장")
    ap.add_argument("--list", required=True, help="way_id·target 이 담긴 JSON")
    ap.add_argument("--region", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--base_url", default="http://localhost:5002",
                    help="기준선 OSRM (차단 없는 것)")
    args = ap.parse_args()

    items = json.loads(Path(args.list).read_text(encoding="utf-8"))
    scen = Path(args.scenario)
    cfgs = sorted(scen.glob("config_*.yaml"))
    if not cfgs:
        sys.exit(f"config 없음: {scen}")
    cfg = yaml.safe_load(cfgs[0].read_text(encoding="utf-8"))
    site, centers, hospitals = endpoints(scen, cfg)

    region = get_region(args.region)
    idx = WayIndex.load(DATA_DIR / "way_index" / args.region)

    results = []
    for i, it in enumerate(items, 1):
        wid = int(it["way_id"])
        tgt, kind = it["target"], it["target_kind"]
        if kind == "center2site":
            a, b = centers.get(tgt), site
        else:
            a, b = site, hospitals.get(tgt)
        if a is None or b is None:
            print(f"[{i}/{len(items)}] way {wid} — 좌표 없음: {tgt}")
            continue

        print(f"[{i}/{len(items)}] way {wid} {it['name']}  "
              f"{kind} {tgt[:20]}  Δpdr {it['d_pdr']:+.5f}")
        t0 = time.time()
        rec = {**it, "from": list(a), "to": list(b)}
        try:
            # 기준선은 매번 같지만, 구간이 way 마다 다르므로 함께 질의한다
            nb = osrm.route(args.base_url, a, b, overview="full", nodes=True)
            if not nb["ok"]:
                raise RuntimeError(f"기준선 질의 실패 {nb['code']}")

            # 이 way 를 지운 인스턴스
            drop = {wid}
            write_pbf_without_ways(OSRM_DIR / region.pbf,
                                   OSRM_DIR / pbf_rel(str(wid)), drop)
            osrm.extract(pbf_rel(str(wid)), builder=BUILDER)
            osrm.partition(pbf_rel(str(wid)), builder=BUILDER)
            osrm.customize(pbf_rel(str(wid)), builder=BUILDER)
            osrm.serve(pbf_rel(str(wid)), PORT, CONTAINER)
            if not osrm.wait_ready(f"http://localhost:{PORT}"):
                raise RuntimeError("붕괴본 OSRM 응답 없음")
            nc = osrm.route(f"http://localhost:{PORT}", a, b,
                            overview="full", nodes=True)
            if not nc["ok"]:
                raise RuntimeError(f"붕괴본 질의 실패 {nc['code']}")

            wb, rb = ways_of(idx, nb["nodes"])
            wc, rc = ways_of(idx, nc["nodes"])
            # 끊은 way 자체의 좌표 — 지도에 빨간 실선으로 그린다
            pos = int(np.flatnonzero(idx.way_id == wid)[0])
            cut_coords = [[round(float(y), 6), round(float(x), 6)]
                          for x, y in idx.coords_of(pos)]

            rec.update({
                "status": "ok",
                "baseline": {"distance_km": nb["distance_km"],
                             "duration_min": nb["duration_min"],
                             "geometry": nb["geometry"], "ways": wb,
                             "n_ways": len(wb)},
                "collapsed": {"distance_km": nc["distance_km"],
                              "duration_min": nc["duration_min"],
                              "geometry": nc["geometry"], "ways": wc,
                              "n_ways": len(wc)},
                "cut_way_coords": cut_coords,
                "delta_km": round(nc["distance_km"] - nb["distance_km"], 4),
                "delta_min": round(nc["duration_min"] - nb["duration_min"], 4),
            })
            print(f"     {nb['distance_km']:.3f}→{nc['distance_km']:.3f} km  "
                  f"{nb['duration_min']:.2f}→{nc['duration_min']:.2f} 분  "
                  f"way {len(wb)}→{len(wc)}  ({time.time()-t0:.0f}초)")
        except Exception as e:
            rec.update({"status": f"error: {e}"[:200]})
            print(f"     실패: {str(e)[:120]}")
        finally:
            teardown(str(wid))
        results.append(rec)

    out = DATA_DIR / "routes" / f"{slug(args.label)}_inspect.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": {"scenario": str(scen), "region": args.region,
                   "base_url": args.base_url, "site": list(site)},
        "items": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n저장: {out}  ({out.stat().st_size/1024:.0f} KB)  성공 {ok}/{len(results)}")


if __name__ == "__main__":
    main()

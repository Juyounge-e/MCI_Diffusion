"""
MCI 시나리오가 실제로 쓰는 경로들의 way 합집합을 뽑는다.

    python road_disruption/scripts/scenario_route_ways.py \
        --scenario "MCI_ADV2/scenarios/exp_artcenter_osrm/(36.366605,127.383691)" \
        --region daejeon --label artcenter

왜 필요한가:
    run_cut_sweep.py 는 way 목록을 받아 하나씩 끊는다. 그 목록을 "경로 하나"가
    아니라 "시뮬레이션이 실제로 쓰는 경로 전부"에서 뽑아야 재난 영향을 본다.

대상 경로 (재난 발생지 기준):
    ① 안전센터 → 현장    출동. 단 43개 센터 전부가 아니라 런타임 대수만큼만.
    ② 현장 → 병원        이송. hospital_info.csv 의 후보 병원 전부.
    (병원 → 현장 재출동은 제외 — 전원(轉院)이 없다고 가정)

런타임 센터를 왜 다시 계산하나:
    ScenarioManager.setup_ambulance 가 보유대수를 np.repeat 로 전개한 뒤
    YAML amb_num 만큼 head() 로 자른다. 잘렸다는 사실이 로그에도 결과에도
    남지 않는다 — 경고는 amb_num > 가용대수 일 때만 뜬다. 여기서 같은 규칙을
    재현해 실제 대상 센터를 확정하고 CSV 로 남긴다. 난수가 없어 정확히 같다.

좌표 출처:
    amb_station_info.csv 에는 좌표가 없다. 생성기가 저장한
    routes/center2site/*.json 의 meta.center 에서 가져온다 (lon, lat 순서).
    병원은 hospital_info.csv 의 x좌표(lon)/y좌표(lat) 를 쓴다.

산출물:
    data/routes/<label>_stations.csv   런타임 센터 목록 (실험 기록용)
    data/routes/<label>_ways.json      way 합집합. run_cut_sweep.py --ways 에 그대로 넣는다
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis import osrm
from roaddis.cli import slug
from roaddis.osm_tags import KEEP_TAGS
from roaddis.paths import DATA_DIR
from roaddis.way_index import WayIndex

ROAD_CRS = "EPSG:5186"
WGS84 = "EPSG:4326"


def runtime_stations(scen: Path, cfg: dict) -> pd.DataFrame:
    """
    ScenarioManager.setup_ambulance 와 같은 규칙으로 런타임 센터를 복원한다.

    보유대수만큼 행을 반복한 뒤 amb_num 개를 앞에서 자른다. CSV 가 도로
    소요시간 오름차순이라 결과가 결정적이다.
    """
    p = Path(cfg["entity_info"]["ambulance"]["dispatch_distance_info"])
    if not p.exists():                              # config 가 다른 머신 경로일 때
        p = scen / "amb_station_info.csv"
    df = pd.read_csv(p, encoding="utf-8-sig")
    name_col = "안전센터/소방서이름"
    if "보유대수" not in df.columns:
        # 구 포맷: 행이 곧 ambulance 1대 — 전개하지 않는다
        df = df.copy()
        df["보유대수"] = 1

    amb_num = int(cfg["entity_info"]["ambulance"]["amb_num"])
    counts = (pd.to_numeric(df["보유대수"], errors="coerce")
              .fillna(1).astype(int).clip(lower=1))
    ex = df.loc[df.index.repeat(counts.values)].reset_index(drop=True)
    if amb_num > len(ex):
        print(f"  [주의] amb_num={amb_num} > 가용 {len(ex)}대 — 가용분으로 제한")
        amb_num = len(ex)
    used = ex.head(amb_num).groupby(name_col, sort=False).size()

    out = df[df[name_col].isin(set(used.index))].copy()
    out["런타임대수"] = out[name_col].map(used)
    out = out.sort_values("duration").reset_index(drop=True)
    print(f"안전센터: 고유 {len(df)}곳 · 보유대수 합 {int(counts.sum())}대 "
          f"· amb_num {amb_num}대 → 실제 대상 {len(out)}곳")
    print(f"  제외 {len(df) - len(out)}곳 (소요시간 컷오프 밖)")
    return out


def station_coords(scen: Path) -> dict:
    """생성기가 저장한 center2site 경로 JSON 에서 센터 좌표를 읽는다."""
    d = scen / "routes" / "center2site"
    if not d.exists():
        sys.exit(f"경로 JSON 없음: {d}\n"
                 f"  시나리오 생성 시 save_routes_json 이 꺼져 있었습니다.")
    out = {}
    for f in sorted(d.glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))["meta"]
        lon, lat = m["center"]                      # 생성기는 [lon, lat] 로 저장
        out[m["name"]] = (float(lat), float(lon))
    return out


def collect(idx: WayIndex, url: str, pairs: list) -> tuple:
    """
    각 구간을 annotations=nodes 로 질의하고 지나간 way 를 모은다.

    반환: (way_pos -> {"used": 최대 사용 세그먼트 수, "routes": [구간이름]}, 실패목록)
    """
    acc, fails = {}, []
    for i, (kind, name, a, b) in enumerate(pairs, 1):
        r = osrm.route(url, a, b, nodes=True)
        if not r["ok"] or not r.get("nodes"):
            fails.append({"kind": kind, "name": name, "code": r["code"]})
            print(f"  [{i}/{len(pairs)}] {kind:12s} {name[:24]:24s} 실패 {r['code']}")
            continue
        w = idx.ways_on_route(r["nodes"])
        miss = w["n_route_seg"] - w["n_matched"]
        for pos, u in zip(w["positions"], w["used"]):
            e = acc.setdefault(int(pos), {"used": 0, "routes": []})
            # 같은 way 를 여러 구간이 쓰면 가장 많이 쓴 쪽을 남긴다
            e["used"] = max(e["used"], int(u))
            e["routes"].append(f"{kind}:{name}")
        print(f"  [{i}/{len(pairs)}] {kind:12s} {name[:24]:24s} "
              f"{r['distance_km']:6.2f}km  way {len(w['positions']):3d}"
              + (f"  [세그먼트 {miss}개 누락]" if miss else ""))
    return acc, fails


def main():
    ap = argparse.ArgumentParser(description="시나리오가 쓰는 경로들의 way 합집합")
    ap.add_argument("--scenario", required=True,
                    help="시나리오 폴더 (config_*.yaml 이 있는 곳)")
    ap.add_argument("--region", required=True)
    ap.add_argument("--label", required=True, help="출력 파일명 (영어)")
    ap.add_argument("--osrm_url", default="http://localhost:5002",
                    help="기준선 OSRM (차단 없는 것). 기본 5002=daejeon")
    ap.add_argument("--no-hospitals", action="store_true",
                    help="현장→병원 구간 제외 (안전센터만)")
    ap.add_argument("--no-stations", action="store_true",
                    help="안전센터→현장 구간 제외 (병원만)")
    args = ap.parse_args()

    scen = Path(args.scenario)
    cfgs = sorted(scen.glob("config_*.yaml"))
    if not cfgs:
        sys.exit(f"config 없음: {scen}/config_*.yaml")
    cfg = yaml.safe_load(cfgs[0].read_text(encoding="utf-8"))

    pt = cfg["entity_info"]["patient"]
    site = (float(pt["latitude"]), float(pt["longitude"]))
    print(f"시나리오: {scen.name}")
    print(f"재난 발생지: {site[0]}, {site[1]}\n")

    # --- 구간 목록 만들기 -------------------------------------------------
    pairs = []
    st = runtime_stations(scen, cfg)
    if not args.no_stations:
        coords = station_coords(scen)
        name_col = "안전센터/소방서이름"
        missing = [n for n in st[name_col] if n not in coords]
        if missing:
            sys.exit(f"좌표를 못 찾은 센터 {len(missing)}곳: {missing[:3]}\n"
                     f"  routes/center2site 의 meta.name 과 CSV 이름이 다릅니다.")
        # ① 안전센터 → 현장 (출동)
        for n in st[name_col]:
            pairs.append(("center2site", n, coords[n], site))

    hp = Path(cfg["entity_info"]["hospital"]["info_path"])
    if not hp.exists():
        hp = scen / "hospital_info.csv"
    hos = pd.read_csv(hp, encoding="utf-8-sig")
    print(f"후보 병원: {len(hos)}곳")
    if not args.no_hospitals:
        # ② 현장 → 병원 (이송). 병원→현장 재출동은 제외.
        for _, r in hos.iterrows():
            pairs.append(("site2hospital", str(r["요양기관명"]),
                          site, (float(r["y좌표"]), float(r["x좌표"]))))

    print(f"\n질의할 구간 {len(pairs)}개  (OSRM {args.osrm_url})")

    # --- 질의 + way 역추적 ------------------------------------------------
    idx = WayIndex.load(DATA_DIR / "way_index" / args.region)
    acc, fails = collect(idx, args.osrm_url, pairs)
    if not acc:
        sys.exit("way 를 하나도 못 찾았습니다. OSRM 이 떠 있는지 확인하세요.")

    # --- way 기록 만들기 (route_ways.py 와 같은 스키마) --------------------
    t = idx.tags
    seg_total = np.diff(idx.seg_off)
    tr = Transformer.from_crs(WGS84, ROAD_CRS, always_xy=True)

    def wlen(pos):
        c = idx.coords_of(pos)
        x, y = tr.transform(c[:, 0], c[:, 1])
        return float(np.hypot(np.diff(x), np.diff(y)).sum())

    def tag(row, key):
        v = row[key]
        return None if v is None or str(v) == "nan" else str(v)

    ways = []
    for pos, e in acc.items():
        row = t.iloc[pos]
        c = idx.coords_of(pos)
        rec = {
            "way_id": int(row.way_id),
            "is_drive": bool(row.is_drive),
            "length_m": round(wlen(pos), 1),
            "segments_used": e["used"],
            "segments_total": int(seg_total[pos]),
            "center": [round(float(c[:, 1].mean()), 6),
                       round(float(c[:, 0].mean()), 6)],
            "n_routes": len(e["routes"]),
            "routes": sorted(set(e["routes"])),
        }
        for k in KEEP_TAGS:
            rec[k] = tag(row, k)
        ways.append(rec)
    # 많은 구간이 공유하는 way 가 먼저 — 끊었을 때 영향이 클 후보다
    ways.sort(key=lambda x: (-x["n_routes"], -x["length_m"]))

    # --- 요약 -------------------------------------------------------------
    print(f"\nway 합집합: {len(ways)}개")
    print(f"  1회 16초 기준 전수 sweep 예상 {len(ways) * 16 / 3600:.1f}시간\n")
    print(f"{'공유':>4}{'등급':>12}{'길이m':>9}  도로명")
    print("-" * 62)
    for w in ways[:20]:
        print(f"{w['n_routes']:>4}{str(w['highway']):>12}{w['length_m']:>9.0f}  "
              f"{w['name'] or '(무명)'}")
    if len(ways) > 20:
        print(f"  ... 나머지 {len(ways) - 20}개")

    byclass = {}
    for w in ways:
        byclass[w["highway"]] = byclass.get(w["highway"], 0) + 1
    print("\n등급별: " + ", ".join(f"{k}={v}" for k, v in
                                 sorted(byclass.items(), key=lambda kv: -kv[1])))
    if fails:
        print(f"\n[주의] 질의 실패 {len(fails)}건: "
              + ", ".join(f"{f['name'][:12]}({f['code']})" for f in fails[:5]))

    # --- 저장 -------------------------------------------------------------
    out_dir = DATA_DIR / "routes"
    out_dir.mkdir(parents=True, exist_ok=True)

    scsv = out_dir / f"{slug(args.label)}_stations.csv"
    st.to_csv(scsv, index=False, encoding="utf-8-sig")

    payload = {
        "source": {
            "scenario": str(scen),
            "config": cfgs[0].name,
            "site": [site[0], site[1]],
            "osrm_url": args.osrm_url,
            "region": args.region,
            "index": f"{args.region}_ways.npz",
            "amb_num": int(cfg["entity_info"]["ambulance"]["amb_num"]),
        },
        "routes": {
            "n_queried": len(pairs),
            "n_stations": 0 if args.no_stations else len(st),
            "n_hospitals": 0 if args.no_hospitals else len(hos),
            "kinds": ["center2site", "site2hospital"],
            "excluded": "hospital2site (전원 없음 가정)",
            "failed": fails,
        },
        "ways": ways,
    }
    wjson = out_dir / f"{slug(args.label)}_ways.json"
    wjson.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"\n저장: {scsv.name}  ({len(st)}행)")
    print(f"저장: {wjson.name}  ({wjson.stat().st_size/1024:.1f} KB)")
    print(f"\n다음: run_cut_sweep.py --ways {slug(args.label)}_ways.json ...")


if __name__ == "__main__":
    main()

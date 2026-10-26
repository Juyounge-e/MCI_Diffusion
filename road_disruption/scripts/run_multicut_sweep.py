"""
way 를 k 개씩 동시에 끊으며 MCI 시뮬레이션을 반복한다.

    python road_disruption/scripts/run_multicut_sweep.py \
        --ways artcenter_top20_ways --region daejeon \
        --latitude 36.366605 --longitude 127.383691 \
        --label artcenter_pair --k 2 --total_samples 5000

왜 필요한가:
    run_cut_sweep.py 는 way 를 하나씩만 끊는다. 대전예술의전당 5000샘플 전수에서
    way 한 개의 효과는 최대 +0.00098 로 CI(0.00107) 를 넘지 못했다. 효과가
    작은 것이지 정밀도가 부족한 게 아니므로, 차단 규모를 키워야 한다.

여기서 보려는 것:
    조합 효과가 단독 효과의 합보다 큰가(초가법성). 두 도로가 같은 우회로를
    공유하면 하나만 끊었을 때는 다른 쪽으로 돌아가지만 둘 다 끊으면 갈 곳이
    없어져 지연이 급증할 수 있다. 그런 쌍이 '적대적 재난' 후보다.

    Δpair 와 Δa + Δb 를 비교한다. 단독 결과는 --singles 로 넘긴 CSV 에서 읽는다.

한 조합당: pbf 에서 way k 개 삭제 → extract/partition/customize → 기동 →
           시나리오 생성 → 시뮬레이션 → 지표 읽기 → 정리
중단해도 이어서 돈다 — 결과 CSV 에 status=ok 인 조합은 건너뛴다.
"""
import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis import osrm
from roaddis.cli import slug
from roaddis.paths import (DATA_DIR, MCI_ADV2_DIR, MCI_PYTHON, OSRM_DIR,
                           data_subdir)
from roaddis.pbf import write_pbf_without_ways
from roaddis.regions import get_region
from roaddis.way_index import WayIndex

METRICS = ["rew", "time", "pdr", "rewWOG", "pdrWOG"]

PORT = 5032                       # sweep(5030) · inspect(5031) 과 겹치지 않게
CONTAINER = "osrm-multicut"
BUILDER = "osrm-multicut-build"


def pbf_rel(key: str) -> str:
    return f"cut_pbf/_mc_{key}.osm.pbf"


def teardown(key: str):
    osrm.remove_container(CONTAINER)
    osrm.remove_container(BUILDER)
    stem = Path(pbf_rel(key)).name.replace(".osm.pbf", "")
    for f in (OSRM_DIR / "cut_pbf").glob(f"{stem}*"):
        f.unlink(missing_ok=True)


def read_stat(stat_path: Path) -> dict:
    lines = [l for l in stat_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) > len(METRICS):
        raise RuntimeError(
            f"결과 {len(lines)}행 — 지표는 {len(METRICS)}개뿐입니다. "
            f"룰이 여러 개 돌았습니다 ({stat_path})")
    out = {}
    for i, line in enumerate(lines):
        nums = [float(x) for x in line.split("  ")[1:] if x.strip()]
        if nums:
            out[METRICS[i]] = nums[0]
            out[f"{METRICS[i]}_ci"] = nums[2] if len(nums) > 2 else float("nan")
    missing = [m for m in METRICS if m not in out]
    if missing:
        raise RuntimeError(f"지표 누락 {missing} ({stat_path})")
    return out


def run_sim(exp_id: str, lat: float, lon: float, port: int, args) -> dict:
    """시나리오 생성 + 시뮬레이션. run_cut_sweep.py 와 같은 방식."""
    gen = [
        str(MCI_PYTHON), str(MCI_ADV2_DIR / "sce_src" / "make_csv_yaml_dynamic.py"),
        "--base_path", str(MCI_ADV2_DIR),
        "--latitude", str(lat), "--longitude", str(lon),
        "--incident_size", str(args.incident_size),
        "--amb_count", str(args.amb_count),
        "--uav_count", str(args.uav_count),
        "--amb_velocity", "40", "--uav_velocity", "80",
        "--total_samples", str(args.total_samples),
        "--random_seed", str(args.random_seed),
        "--is_use_time", "false", "--duration_coeff", "1.0",
        "--amb_handover_time", "0", "--uav_handover_time", "0",
        "--osrm_url", f"http://localhost:{port}",
        "--experiment_id", exp_id,
    ]
    r = subprocess.run(gen, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"시나리오 생성 실패\n{(r.stderr or r.stdout)[-500:]}")

    hits = sorted((MCI_ADV2_DIR / "scenarios").glob(f"*{exp_id}*/*/config_*.yaml"))
    if not hits:
        raise FileNotFoundError(f"config 없음: scenarios/*{exp_id}*")
    cfg = hits[-1]

    sim = [str(MCI_PYTHON), str(MCI_ADV2_DIR / "sim_src" / "main.py"),
           "--config_path", str(cfg)]
    if args.trace:
        sim.append("--trace")
    r2 = subprocess.run(sim, cwd=str(MCI_ADV2_DIR.parent), capture_output=True,
                        text=True, encoding="utf-8", errors="replace", timeout=3600)
    if r2.returncode != 0:
        raise RuntimeError(f"시뮬레이션 실패\n{(r2.stderr or r2.stdout)[-500:]}")

    import yaml
    c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    rd = Path(c["run_setting"]["output_path"]) / str(c["run_setting"]["exp_indicator"])
    stats = list(rd.glob("results_*_stat.txt"))
    if not stats:
        raise FileNotFoundError(f"결과 없음: {rd}")
    return read_stat(stats[0])


def load_singles(path: Path) -> tuple:
    """단독 차단 결과 CSV → (way_id -> Δpdr dict, 기준선 dict)."""
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                rows[r["way_id"]] = r
    base = rows.pop("baseline", None)
    if base is None:
        raise RuntimeError(f"기준선 행이 없습니다: {path}")
    return rows, base


def main():
    ap = argparse.ArgumentParser(description="way k 개 동시 차단 sweep")
    ap.add_argument("--ways", required=True, help="way 목록 JSON (tag 또는 경로)")
    ap.add_argument("--region", required=True)
    ap.add_argument("--latitude", type=float, required=True)
    ap.add_argument("--longitude", type=float, required=True)
    ap.add_argument("--label", default="multicut", help="출력 파일명 (영어)")
    ap.add_argument("--k", type=int, default=2, help="동시에 끊을 way 수")
    ap.add_argument("--singles", default=None,
                    help="단독 결과 CSV (초가법성 비교용). 기본: <label 앞부분>_results.csv")
    ap.add_argument("--incident_size", type=int, default=30)
    ap.add_argument("--amb_count", type=int, default=30)
    ap.add_argument("--uav_count", type=int, default=0)
    ap.add_argument("--total_samples", type=int, default=5000)
    ap.add_argument("--random_seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="앞 N개만 (시험용)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    p = Path(args.ways)
    if not p.suffix:
        p = DATA_DIR / "routes" / f"{args.ways}.json"
    if not p.exists():
        sys.exit(f"파일 없음: {p}")
    ways = json.loads(p.read_text(encoding="utf-8"))["ways"]
    info = {str(w["way_id"]): w for w in ways}

    combos = list(itertools.combinations([str(w["way_id"]) for w in ways], args.k))
    if args.limit:
        combos = combos[:args.limit]

    # 단독 결과 (있으면 초가법성 비교에 쓴다)
    singles, sbase = {}, None
    sp = Path(args.singles) if args.singles else None
    if sp is None:
        cand = DATA_DIR / "sweep" / "artcenter5000_results.csv"
        sp = cand if cand.exists() else None
    if sp and sp.exists():
        try:
            singles, sbase = load_singles(sp)
            print(f"단독 결과 참조: {sp.name} (way {len(singles)}개)")
        except Exception as e:
            print(f"  [주의] 단독 결과를 못 읽었습니다: {e}")

    region = get_region(args.region)
    idx = WayIndex.load(DATA_DIR / "way_index" / args.region)
    known = set(int(x) for x in idx.way_id)

    out_csv = data_subdir("sweep") / f"{slug(args.label)}_results.csv"
    done = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    done.add(r["combo"])
        print(f"이어서 진행: 성공 {len(done)}건 ({out_csv.name})")

    cols = (["combo", "way_ids", "names", "highways", "k",
             "length_m_sum", "n_routes_max", "status", "seconds",
             "d_pdr_sum_single"] + METRICS + [f"{m}_ci" for m in METRICS])
    new = not out_csv.exists()
    fh = open(out_csv, "a", encoding="utf-8-sig", newline="")
    wr = csv.DictWriter(fh, fieldnames=cols)
    if new:
        wr.writeheader()

    todo = [c for c in combos if "+".join(c) not in done]
    print(f"조합 {len(combos)}개 (k={args.k}) 중 남은 것 {len(todo)}개")
    print(f"포트 {PORT} · 컨테이너 {CONTAINER} · 샘플 {args.total_samples}\n")

    # --- 기준선 ---
    if not args.skip_baseline and "baseline" not in done:
        print("[기준선] 차단 없음")
        t0 = time.time()
        try:
            st = write_pbf_without_ways(OSRM_DIR / region.pbf,
                                        OSRM_DIR / pbf_rel("baseline"), set())
            for step in (osrm.extract, osrm.partition, osrm.customize):
                step(pbf_rel("baseline"), builder=BUILDER)
            osrm.serve(pbf_rel("baseline"), PORT, CONTAINER)
            if not osrm.wait_ready(f"http://localhost:{PORT}"):
                raise RuntimeError("OSRM 응답 없음")
            m = run_sim(f"{slug(args.label)}_baseline", args.latitude,
                        args.longitude, PORT, args)
            wr.writerow({"combo": "baseline", "way_ids": "", "names": "(차단 없음)",
                         "k": 0, "status": "ok",
                         "seconds": round(time.time() - t0, 1), **m})
            fh.flush()
            print(f"  pdr={m['pdr']:.5f}  rew={m['rew']:.4f}  ({time.time()-t0:.0f}초)")
        except Exception as e:
            print(f"  실패: {str(e)[:150]}")
            wr.writerow({"combo": "baseline", "status": f"error: {e}"[:200]})
            fh.flush()
        finally:
            teardown("baseline")

    # --- 조합 ---
    bp = None
    if sbase is not None:
        bp = float(sbase["pdr"])
    times = []
    for i, combo in enumerate(todo, 1):
        key = "+".join(combo)
        wid_ints = [int(x) for x in combo]
        missing = [x for x in wid_ints if x not in known]
        eta = (f"  ETA {np.mean(times)*(len(todo)-i+1)/60:.0f}분" if times else "")
        nm = " + ".join((info[x]["name"] or "(무명)") for x in combo)
        print(f"[{i}/{len(todo)}] {key}  {nm}{eta}")

        # 단독 효과의 합 (있으면)
        dsum = float("nan")
        if bp is not None and all(x in singles for x in combo):
            dsum = sum(float(singles[x]["pdr"]) - bp for x in combo)

        row = {"combo": key, "way_ids": key, "names": nm,
               "highways": "+".join(info[x]["highway"] for x in combo),
               "k": args.k,
               "length_m_sum": round(sum(info[x]["length_m"] for x in combo), 1),
               "n_routes_max": max(info[x]["n_routes"] for x in combo),
               "d_pdr_sum_single": round(dsum, 6) if dsum == dsum else ""}
        t0 = time.time()
        try:
            if missing:
                raise RuntimeError(f"인덱스에 없는 way {missing}")
            # 파일명이 겹치면 앞 반복의 빌드가 끝나기 전에 지워져 경합이 난다
            fkey = "_".join(combo)
            write_pbf_without_ways(OSRM_DIR / region.pbf,
                                   OSRM_DIR / pbf_rel(fkey), set(wid_ints))
            for step in (osrm.extract, osrm.partition, osrm.customize):
                step(pbf_rel(fkey), builder=BUILDER)
            osrm.serve(pbf_rel(fkey), PORT, CONTAINER)
            if not osrm.wait_ready(f"http://localhost:{PORT}"):
                raise RuntimeError("OSRM 응답 없음")
            m = run_sim(f"{slug(args.label)}_c{i:04d}", args.latitude,
                        args.longitude, PORT, args)
            row.update({"status": "ok", "seconds": round(time.time() - t0, 1), **m})
            extra = ""
            if bp is not None:
                dp = m["pdr"] - bp
                extra = f"  Δpdr={dp:+.5f}"
                if dsum == dsum:
                    extra += f"  단독합={dsum:+.5f}  비율={dp/dsum:.2f}" if dsum else ""
            print(f"     pdr={m['pdr']:.5f}{extra}  ({time.time()-t0:.0f}초)")
        except Exception as e:
            row.update({"status": f"error: {e}"[:200],
                        "seconds": round(time.time() - t0, 1)})
            print(f"     실패: {str(e)[:150]}")
        finally:
            teardown("_".join(combo))
        wr.writerow(row)
        fh.flush()
        times.append(time.time() - t0)

    fh.close()
    print(f"\n완료: {out_csv}")
    if times:
        print(f"  총 {sum(times)/60:.1f}분 · 1회 평균 {np.mean(times):.0f}초")


if __name__ == "__main__":
    main()

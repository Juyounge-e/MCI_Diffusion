"""
경로가 지나는 way 를 하나씩 끊어가며 MCI 시뮬레이션을 반복 실행한다.

    python road_disruption/scripts/run_cut_sweep.py \
        --ways normal_ways --region daejeon \
        --latitude 36.3757 --longitude 127.3767 --label science_konyang

한 번의 반복:
    ① pbf 에서 way 1개 삭제 → extract/partition/customize → 고정 포트로 기동  (~9초)
    ② MCI_ADV2 시나리오 생성 (--osrm_url 로 붕괴본을 물린다)
    ③ sim_src/main.py 로 시뮬레이션
    ④ results_*_stat.txt 에서 지표 9개 읽기
    ⑤ 컨테이너 제거 + pbf 빌드 산출물 삭제

포트와 컨테이너를 하나로 재사용함(병렬 실험 고려하지 않음)

중단해도 이어서 돌린다 — 결과 CSV 에 이미 있는 way 는 건너뛴다.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis import osrm
from roaddis.cli import slug
from roaddis.cutspec import CutSpec
from roaddis.paths import (DATA_DIR, MCI_ADV2_DIR, MCI_PYTHON, OSRM_DIR,
                           PKG_DIR, data_subdir)
from roaddis.pbf import write_pbf_without_ways
from roaddis.regions import get_region
from roaddis.way_index import WayIndex

# sim_src/main.py 가 결과를 쌓는 순서 (main.py 의 outcomes_* 순서와 동일)
# sim_src/main.py 가 결과 파일에 쌓는 순서. 이 5개가 지표 정본이다.
#   rew     Σ SurvivalProb(rescue_time, severity)          ↑ 좋음
#   time    마지막 환자 치료 완료 시각                      ↓ 좋음
#   pdr     1 − Reward / Preventable                       ↓ 좋음
#   rewWOG  Reward − Green 기여분                          ↑ 좋음
#   pdrWOG  1 − (Reward − Green) / (Preventable − Green)   ↓ 좋음
# red/yellow/green/black 은 환자 구성이지 지표가 아니라 결과 파일에 없다.
METRICS = ["rew", "time", "pdr", "rewWOG", "pdrWOG"]

PORT = 5030                       # sweep 전용. 5001/5002/5011 과 겹치지 않게
CONTAINER = "osrm-sweep"
BUILDER = "osrm-sweep-build"      # 일회성 빌드 컨테이너 (extract/partition/customize)

# pbf 이름에 반복 키를 넣는다. 이름을 재사용하면 앞 반복의 빌드가 끝나기 전에
# 다음 반복이 같은 파일을 지워 경합이 난다 — 실제로 osrm-customize 가
# "_sweep.osrm.cell_metrics: No such file" 에러로 매달린 사례가 있었다.
def pbf_rel(key: str) -> str:
    return f"cut_pbf/_sweep_{key}.osm.pbf"


def read_stat(stat_path: Path) -> dict:
    """
    results_*_stat.txt 에서 지표 평균과 95% CI 반폭을 읽는다.

    각 행이 지표 하나이고 (룰이름, 평균, 표준편차, CI반폭) 을 두 칸 공백으로 나눈다.
    행 수가 METRICS 보다 많으면 룰이 여러 개 돈 것이다 — 단일 룰이 아니므로 막는다.
    """
    lines = [l for l in stat_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) > len(METRICS):
        raise RuntimeError(
            f"결과 {len(lines)}행 — 지표는 {len(METRICS)}개뿐입니다. "
            f"룰이 {len(lines)//len(METRICS)}개 돌았습니다. "
            f"config 의 rule_info.isFullFactorial 을 False 로 두고 각 *_rule 을 "
            f"값 1개로 맞추세요 ({stat_path})")
    out = {}
    for i, line in enumerate(lines):
        nums = [float(x) for x in line.split("  ")[1:] if x.strip()]
        if nums:
            out[METRICS[i]] = nums[0]                            # 평균
            out[f"{METRICS[i]}_ci"] = nums[2] if len(nums) > 2 else float("nan")
    missing = [m for m in METRICS if m not in out]
    if missing:
        raise RuntimeError(f"지표 누락 {missing} ({stat_path})")
    return out


def run_sim(exp_id: str, lat: float, lon: float, port: int, args) -> dict:
    """
    시나리오 생성 + 시뮬레이션. 지표 dict 반환.

    경로를 손으로 조립하지 않는다. 생성기가 experiment_id 에 접미사를 붙이고
    (osrm 모드면 '_osrm') 폴더 이름도 자기 규칙으로 정하기 때문에, 재구성하면
    생성기를 고칠 때마다 어긋난다. 대신 만들어진 config 를 glob 으로 찾고
    그 안의 output_path / exp_indicator 를 그대로 읽어 결과 위치를 얻는다.
    """
    gen = [
        str(MCI_PYTHON), str(MCI_ADV2_DIR / "sce_src" / "make_csv_yaml_dynamic.py"),
        "--base_path", str(MCI_ADV2_DIR),
        "--latitude", str(lat), "--longitude", str(lon),
        "--incident_size", str(args.incident_size),
        "--amb_count", str(args.amb_count),
        "--uav_count", str(args.uav_count),
        "--amb_velocity", str(args.amb_velocity),
        "--uav_velocity", str(args.uav_velocity),
        "--total_samples", str(args.total_samples),
        "--random_seed", str(args.random_seed),
        "--is_use_time", "true" if args.is_use_time else "false",
        "--duration_coeff", "1.0",
        "--amb_handover_time", str(args.amb_handover_time),
        "--uav_handover_time", str(args.uav_handover_time),
        "--osrm_url", f"http://localhost:{port}",
        "--experiment_id", exp_id,
    ]
    r = subprocess.run(gen, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"시나리오 생성 실패\n{(r.stderr or r.stdout)[-600:]}")

    # 생성기가 experiment_id 에 접미사를 붙이므로 exp_id 를 부분일치로 찾는다
    hits = sorted((MCI_ADV2_DIR / "scenarios").glob(f"*{exp_id}*/*/config_*.yaml"))
    if not hits:
        raise FileNotFoundError(
            f"config 없음: scenarios/*{exp_id}*/*/config_*.yaml\n"
            f"{(r.stdout or '')[-400:]}")
    cfg = hits[-1]

    sim = [str(MCI_PYTHON), str(MCI_ADV2_DIR / "sim_src" / "main.py"),
           "--config_path", str(cfg)]
    # 환자별 이벤트 기록. results 옆에 trace_<좌표>.json 이 생긴다.
    # 배차가 어떻게 바뀌어 지표가 변했는지 되짚을 때만 켠다 — 상시로 켜면
    # way 639개 × 샘플 10개 분량이 쌓인다.
    if getattr(args, "trace", False):
        sim.append("--trace")
    r2 = subprocess.run(sim, cwd=str(MCI_ADV2_DIR.parent), capture_output=True,
                        text=True, encoding="utf-8", errors="replace", timeout=1800)
    if r2.returncode != 0:
        raise RuntimeError(f"시뮬레이션 실패\n{(r2.stderr or r2.stdout)[-600:]}")

    # config 가 결과 위치를 알고 있다 — 추측하지 않는다
    import yaml
    c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    rd = Path(c["run_setting"]["output_path"]) / str(c["run_setting"]["exp_indicator"])
    stats = list(rd.glob("results_*_stat.txt"))
    if not stats:
        raise FileNotFoundError(f"결과 없음: {rd}")
    return read_stat(stats[0])


def build_cut_instance(region, drop_ids: set, key: str) -> dict:
    """way 를 지운 pbf 로 OSRM 을 새로 빌드하고 기동."""
    rel = pbf_rel(key)
    st = write_pbf_without_ways(OSRM_DIR / region.pbf, OSRM_DIR / rel, drop_ids)
    osrm.extract(rel, builder=BUILDER)
    osrm.partition(rel, builder=BUILDER)
    osrm.customize(rel, builder=BUILDER)
    osrm.serve(rel, PORT, CONTAINER)
    if not osrm.wait_ready(f"http://localhost:{PORT}"):
        raise RuntimeError(f"OSRM 응답 없음 (docker logs {CONTAINER})")
    return st


def teardown(key: str):
    """
    컨테이너와 빌드 산출물 정리 — 안 하면 반복마다 67MB 가 쌓인다.

    glob 패턴에 주의: '_sweep_x.*' 는 '_sweep_x.osrm.mldgr' 처럼 점이 여러 개인
    파일을 잡지 못한다. OSRM 산출물이 대부분 그런 이름이라 '*' 로 끝내야 한다.
    빌드 컨테이너도 함께 지운다 — 실패 시 --rm 으로도 남는 경우가 있다.
    """
    osrm.remove_container(CONTAINER)
    osrm.remove_container(BUILDER)
    stem = Path(pbf_rel(key)).name.replace(".osm.pbf", "")
    for f in (OSRM_DIR / "cut_pbf").glob(f"{stem}*"):
        f.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="way 를 하나씩 끊으며 시뮬레이션 반복")
    parser.add_argument("--ways", required=True, help="route_ways.py 결과 (tag 또는 경로)")
    parser.add_argument("--region", required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--label", default="sweep", help="출력 파일명 (영어)")
    parser.add_argument("--incident_size", type=int, default=30)
    parser.add_argument("--amb_count", type=int, default=30)
    parser.add_argument("--uav_count", type=int, default=0)
    parser.add_argument("--amb_velocity", type=int, default=40)
    parser.add_argument("--uav_velocity", type=int, default=80)
    parser.add_argument("--amb_handover_time", type=float, default=0.0)
    parser.add_argument("--uav_handover_time", type=float, default=0.0)
    # is_use_time=True 는 생성기에서 road_provider=kakao 를 뜻하고 parserI 키를 요구한다.
    # 도로 차단 실험은 OSRM 이어야 하므로 기본값은 False 다 (거리 ÷ 속도, 거리는 OSRM).
    parser.add_argument("--is_use_time", action="store_true",
                    help="OSRM duration 사용 (카카오 키 필요). 기본은 거리÷속도")
    parser.add_argument("--total_samples", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="앞 N개만 (시험용)")
    parser.add_argument("--named-only", action="store_true",
                    help="이름 있는 way 만 (service 이면도로 등 제외)")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--trace", action="store_true",
                        help="main.py 에 --trace 를 넘겨 환자별 이벤트를 남긴다")
    args = parser.parse_args()

    p = Path(args.ways)
    if not p.suffix:
        p = DATA_DIR / "routes" / f"{args.ways}.json"
    if not p.exists():
        sys.exit(f"파일 없음: {p}\n  route_ways.py 를 먼저 실행하세요.")
    ways = json.loads(p.read_text(encoding="utf-8"))["ways"]
    if args.named_only:
        ways = [w for w in ways if w["name"]]
    if args.limit:
        ways = ways[:args.limit]

    region = get_region(args.region)
    idx = WayIndex.load(DATA_DIR / "way_index" / args.region)

    out_csv = data_subdir("sweep") / f"{slug(args.label)}_results.csv"
    # 성공한 way 만 '완료' 로 본다. 실패까지 건너뛰면 docker 가 매달려 타임아웃으로
    # 죽은 way 가 영구히 빠진다 — 실제로 203건 중 8건이 osrm-partition/customize
    # 타임아웃으로 실패했고, 재시도하면 대개 성공한다. 실패 행은 CSV 에 남지만
    # 하류(plot_sweep)가 status=='ok' 만 읽으므로 나중 성공 행이 이긴다.
    done, retry = set(), set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                (done if r.get("status") == "ok" else retry).add(r["way_id"])
        retry -= done
        print(f"이어서 진행: 성공 {len(done)}건 ({out_csv.name})")
        if retry:
            print(f"  실패한 {len(retry)}건은 다시 시도합니다")

    cols = (["way_id", "name", "highway", "length_m", "segments_used",
             "segments_total", "status", "seconds"]
            + METRICS + [f"{m}_ci" for m in METRICS])
    new = not out_csv.exists()
    fh = open(out_csv, "a", encoding="utf-8-sig", newline="")
    wr = csv.DictWriter(fh, fieldnames=cols)
    if new:
        wr.writeheader()

    todo = [w for w in ways if str(w["way_id"]) not in done]
    print(f"후보 way {len(ways)}개 중 남은 것 {len(todo)}개")
    print(f"포트 {PORT} · 컨테이너 {CONTAINER} 를 재사용합니다\n")

    # --- 기준선: 아무것도 끊지 않은 상태 ---
    if not args.skip_baseline and "baseline" not in done:
        print("[기준선] 차단 없음")
        t0 = time.time()
        try:
            build_cut_instance(region, set(), "baseline")
            m = run_sim(f"{slug(args.label)}_baseline", args.latitude,
                        args.longitude, PORT, args)
            wr.writerow({"way_id": "baseline", "name": "(차단 없음)",
                         "status": "ok", "seconds": round(time.time() - t0, 1), **m})
            fh.flush()
            print(f"  pdr={m.get('pdr'):.4f}  time={m.get('time'):.1f}  "
                  f"rew={m.get('rew'):.2f}  ({time.time()-t0:.0f}초)")
        except Exception as e:
            print(f"  실패: {e}")
            wr.writerow({"way_id": "baseline", "status": f"error: {e}"[:200]})
            fh.flush()
        finally:
            teardown("baseline")

    # --- way 하나씩 ---
    times = []
    for i, w in enumerate(todo, 1):
        wid = int(w["way_id"])
        t0 = time.time()
        eta = (f"  ETA {np.mean(times)*(len(todo)-i+1)/60:.0f}분"
               if times else "")
        print(f"[{i}/{len(todo)}] way {wid}  {w['name'] or '(무명)'}  "
              f"{w['highway']}  {w['length_m']:.0f}m{eta}")
        row = {"way_id": wid, "name": w["name"], "highway": w["highway"],
               "length_m": w["length_m"], "segments_used": w["segments_used"],
               "segments_total": w["segments_total"]}
        try:
            spec = CutSpec(name=f"{slug(args.label)}_w{wid}", region=args.region,
                           select={"type": "way_id", "ids": [wid]})
            drop = {int(x) for x in idx.way_id[
                np.unique(idx.segment_way_pos()[spec.resolve(idx)])]}
            build_cut_instance(region, drop, f"w{wid}")
            m = run_sim(f"{slug(args.label)}_w{wid}", args.latitude,
                        args.longitude, PORT, args)
            row.update({"status": "ok", "seconds": round(time.time() - t0, 1), **m})
            print(f"     pdr={m.get('pdr'):.4f}  time={m.get('time'):.1f}  "
                  f"rew={m.get('rew'):.2f}  ({time.time()-t0:.0f}초)")
        except Exception as e:
            row.update({"status": f"error: {e}"[:200],
                        "seconds": round(time.time() - t0, 1)})
            print(f"     실패: {str(e)[:160]}")
        finally:
            teardown(f"w{wid}")
        wr.writerow(row)
        fh.flush()
        times.append(time.time() - t0)

    fh.close()
    print(f"\n완료: {out_csv}")
    print(f"  총 {sum(times)/60:.1f}분, 1회 평균 {np.mean(times):.0f}초"
          if times else "")


if __name__ == "__main__":
    main()

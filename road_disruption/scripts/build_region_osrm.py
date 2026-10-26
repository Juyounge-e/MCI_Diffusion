"""
지역 pbf → 지역 전용 OSRM 인스턴스.

사용 예:
    python road_disruption/scripts/build_region_osrm.py --region daejeon
    python road_disruption/scripts/build_region_osrm.py --region daejeon --serve-only
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis import osrm
from roaddis.paths import OSRM_DIR
from roaddis.regions import get_region


def main():
    ap = argparse.ArgumentParser(description="지역 OSRM 인스턴스 구축")
    ap.add_argument("--region", required=True)
    ap.add_argument("--serve-only", action="store_true",
                    help="빌드를 건너뛰고 기동만 (이미 빌드된 경우)")
    ap.add_argument("--container", default=None,
                    help="컨테이너 이름 (기본: osrm-<region>)")
    args = ap.parse_args()

    region = get_region(args.region)
    if region.is_baseline:
        sys.exit(f"'{args.region}' 은 baseline 입니다. 차단 실험용으로 다시 빌드하면 "
                 f"비교 대상이 사라집니다.")

    pbf_path = OSRM_DIR / region.pbf
    if not pbf_path.exists():
        sys.exit(f"pbf 없음: {pbf_path}\n  먼저 extract_region_pbf.py 를 실행하세요.")

    name = args.container or f"osrm-{args.region}"
    t0 = time.time()

    if not args.serve_only:
        print(f"[1/4] osrm-extract  ({region.pbf}, profile={osrm.CAR_PROFILE})")
        osrm.extract(region.pbf)
        print(f"[2/4] osrm-partition")
        osrm.partition(region.pbf)
        print(f"[3/4] osrm-customize  (차단 없는 baseline 가중치)")
        osrm.customize(region.pbf)
    else:
        print("[1-3/4] 빌드 생략 (--serve-only)")

    print(f"[4/4] osrm-routed 기동  container={name}  port={region.osrm_port}")
    osrm.serve(region.pbf, region.osrm_port, name)

    if not osrm.wait_ready(region.osrm_url):
        sys.exit(f"서버가 응답하지 않습니다. `docker logs {name}` 확인")

    print(f"완료. {region.osrm_url}  ({time.time()-t0:.1f}s)")

    # 전국 baseline 과 같은 경로를 물어 정합성을 본다.
    # 두 인스턴스는 같은 pbf·같은 프로파일에서 나왔으므로 결과가 거의 같아야 한다.
    national = get_region("national")
    start, end = (36.3504, 127.3845), (36.3320, 127.4200)
    a = osrm.route(region.osrm_url, start, end)
    b = osrm.route(national.osrm_url, start, end)
    print()
    print("  전국 baseline 과 대조 (대전 시청 부근 → 동쪽)")
    for label, r in ((args.region, a), ("national", b)):
        if r["ok"]:
            print(f"    {label:<10} {r['distance_km']:.3f} km   {r['duration_min']:.2f} min")
        else:
            print(f"    {label:<10} 실패: {r['code']}")
    if a["ok"] and b["ok"]:
        dd = abs(a["duration_min"] - b["duration_min"])
        verdict = "일치" if dd < 0.05 else "차이 있음 — 프로파일/경계 확인 필요"
        print(f"    소요시간 차이 {dd:.3f} min  → {verdict}")


if __name__ == "__main__":
    main()

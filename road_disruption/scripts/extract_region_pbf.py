"""
전국 pbf → 지역 pbf 추출 (osmium extract 래핑).

출력은 OSRM 볼륨 디렉토리(C:/Users/user00/osrm-korea)에 쓴다.
docker 가 /data 로 마운트하는 곳이라, 여기 있어야 osrm-extract 가 볼 수 있다.

사용 예:
    python road_disruption/scripts/extract_region_pbf.py --region daejeon
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis.paths import OSRM_DIR
from roaddis.regions import get_region


def main():
    ap = argparse.ArgumentParser(description="전국 pbf → 지역 pbf 추출")
    ap.add_argument("--region", required=True, help="config/regions.yaml 의 지역명")
    ap.add_argument("--src", default=None, help="원본 전국 pbf (기본: regions.yaml 의 national.pbf)")
    ap.add_argument("--strategy", default="complete_ways", choices=["simple", "complete_ways", "smart"], help="osmium extract 전략. complete_ways 는 경계에 걸친 way의 바깥쪽 node 까지 포함해 좌표 누락을 막는다.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    region = get_region(args.region)
    if region.bbox is None:
        sys.exit(f"'{args.region}' 은 bbox 가 없습니다(전국). 추출 대상이 아닙니다.")

    if shutil.which("osmium") is None:
        sys.exit("osmium 실행 파일을 찾을 수 없습니다. "
                 "`conda activate road` 후 다시 실행하세요.")

    src = Path(args.src) if args.src else OSRM_DIR / get_region("national").pbf
    if not src.exists():
        sys.exit(f"원본 pbf 없음: {src}")

    out = OSRM_DIR / region.pbf
    if out.exists() and not args.overwrite:
        sys.exit(f"이미 존재: {out}  (--overwrite 로 덮어쓰기)")

    lon_min, lat_min, lon_max, lat_max = region.bbox
    cmd = [
        "osmium", "extract",
        "-b", f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "--strategy", args.strategy,
        "-o", str(out),
        str(src),
    ]
    if args.overwrite:
        cmd.insert(2, "--overwrite")

    print(f"[1/1] {args.region} 추출  bbox={region.bbox}")
    print(f"      {src.name} ({src.stat().st_size/1e6:.0f}MB) → {out.name}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"완료: {out}  ({out.stat().st_size/1e6:.1f}MB, {time.time()-t0:.1f}s)")
    print()
    print("다음 단계:")
    print(f"  1) way 인덱스:  python road_disruption/scripts/build_way_index.py "
          f"--region {args.region}")
    print(f"  2) OSRM 빌드:   osrm-extract/partition/customize 를 {region.pbf} 에 대해 "
          f"실행 후 포트 {region.osrm_port} 로 기동")


if __name__ == "__main__":
    main()

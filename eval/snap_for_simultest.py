#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0203 샘플 CSV(lat, lon)를 MCI_ADV2/sce_src/random_coordinate_generator.py의
RandomCoordinateGenerator.snap_point()로 도로에 스냅하고,
sce_src/batch_experiment.py 입력으로 사용할 수 있는 random_metadata CSV로 저장한다.

입력: lat, lon 컬럼이 있는 CSV (예: 0203/0203_samples_100_q3.csv)
출력: grid_id, random_latitude, random_longitude, snapped_latitude, snapped_longitude, snap_distance_m

주의:
- snap_point()는 OSRM(nearest) 호출이 필요함 (인터넷 필요).
- overpass_url을 비우면 Overpass 검증을 스킵하여 더 빠르게 동작함.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def _configure_stdout_utf8() -> None:
    if sys.platform != "win32":
        return
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        else:
            import io

            if hasattr(sys.stdout, "buffer") and sys.stdout.buffer:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)
    except Exception:
        pass


def _import_generator(mci_adv2_dir: Path):
    """
    MCI_ADV2/sce_src/random_coordinate_generator.py를 '파일 경로 기반 import'로 로드해서
    그 안의 RandomCoordinateGenerator를 반환한다.

    (MCI_ADV2가 파이썬 패키지 구조(__init__.py) 가 아닐 수 있어 sys.path hack 없이 동작하도록 함)
    """
    module_path = (mci_adv2_dir / "sce_src" / "random_coordinate_generator.py").resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"random_coordinate_generator.py를 찾을 수 없습니다: {module_path}")

    spec = importlib.util.spec_from_file_location("mci_adv2_random_coordinate_generator", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈 spec 생성 실패: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    try:
        return getattr(module, "RandomCoordinateGenerator")
    except Exception as e:
        raise ImportError(f"RandomCoordinateGenerator 심볼을 찾지 못했습니다: {module_path}") from e


def _snap_rows(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    generator,
) -> Tuple[pd.DataFrame, int]:
    snapped_lat: List[Optional[float]] = []
    snapped_lon: List[Optional[float]] = []
    snap_dist_m: List[Optional[float]] = []
    failed = 0

    for lat, lon in zip(df[lat_col].astype(float).tolist(), df[lon_col].astype(float).tolist()):
        res = generator.snap_point(float(lat), float(lon))
        if not res:
            snapped_lat.append(None)
            snapped_lon.append(None)
            snap_dist_m.append(None)
            failed += 1
            continue
        slat, slon, dist_m = res
        snapped_lat.append(float(slat))
        snapped_lon.append(float(slon))
        snap_dist_m.append(float(dist_m))

    out = pd.DataFrame(
        {
            "latitude": df[lat_col].astype(float).to_numpy(),
            "longitude": df[lon_col].astype(float).to_numpy(),
            "snapped_latitude": snapped_lat,
            "snapped_longitude": snapped_lon,
            "snap_distance_m": snap_dist_m,
        }
    )
    out.insert(0, "grid_id", range(1, len(out) + 1))
    return out, failed


def main() -> None:
    _configure_stdout_utf8()

    repo_root = Path(__file__).resolve().parents[1]
    mci_adv2_dir = repo_root / "MCI_ADV2"

    parser = argparse.ArgumentParser(description="0203 샘플 좌표를 snap_point()로 스냅하여 random_metadata.csv 생성")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="입력 CSV 경로들 (예: --inputs 0203/0203_samples_100_q3.csv 0203/0203_samples_500_q3.csv)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(repo_root / "eval" / "snapped"),
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--shp",
        type=str,
        default=str(mci_adv2_dir / "scenarios" / "ctprvn.shp"),
        help="시도 경계 SHP 경로 (RandomCoordinateGenerator 초기화용)",
    )
    parser.add_argument("--region", type=str, default="daejeon", help="경계 SHP에서 사용할 지역 키워드")
    parser.add_argument("--osrm", type=str, default="https://router.project-osrm.org", help="OSRM base URL")
    parser.add_argument(
        "--overpass",
        type=str,
        default="",
        help="Overpass URL (빈 문자열이면 Overpass 검증 스킵)",
    )
    parser.add_argument("--profile", type=str, default="driving", help="OSRM profile")
    parser.add_argument("--radius_m", type=float, default=250.0, help="OSRM nearest radius (m)")
    parser.add_argument("--delay", type=float, default=0.3, help="OSRM 호출 rate limit delay (sec)")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--lat_col", type=str, default="lat", help="입력 위도 컬럼명")
    parser.add_argument("--lon_col", type=str, default="lon", help="입력 경도 컬럼명")
    parser.add_argument(
        "--keep_failed",
        action="store_true",
        help="스냅 실패한 행도 NaN으로 유지 (기본은 제거)",
    )
    args = parser.parse_args()

    inputs = args.inputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    RandomCoordinateGenerator = _import_generator(mci_adv2_dir)
    gen = RandomCoordinateGenerator(
        shp_path=str(Path(args.shp)),
        region=str(args.region),
        osrm_base=str(args.osrm),
        overpass_url=str(args.overpass or ""),
        profile=str(args.profile),
        radius_m=float(args.radius_m),
        seed=int(args.seed),
        rate_limit_delay=float(args.delay),
        allow_overpass_failure=True,
    )

    print(f"shp      : {args.shp}")
    print(f"region   : {args.region}")
    print(f"osrm     : {args.osrm}")
    print(f"overpass : {args.overpass or '(disabled)'}")
    print(f"radius_m : {args.radius_m}")
    print(f"delay    : {args.delay}")
    print(f"out_dir  : {out_dir}")

    for in_path in inputs:
        in_path_p = Path(in_path)
        if not in_path_p.is_absolute():
            in_path_p = (repo_root / in_path_p).resolve()
        if not in_path_p.exists():
            raise FileNotFoundError(f"입력 CSV를 찾을 수 없습니다: {in_path_p}")

        df = pd.read_csv(in_path_p)
        if args.lat_col not in df.columns or args.lon_col not in df.columns:
            raise ValueError(
                f"입력 CSV에 '{args.lat_col}', '{args.lon_col}' 컬럼이 필요합니다. 실제 컬럼: {list(df.columns)}"
            )

        print(f"\n- input: {in_path_p} (n={len(df)})")
        out_df, failed = _snap_rows(df, args.lat_col, args.lon_col, gen)
        if not args.keep_failed:
            before = len(out_df)
            out_df = out_df.dropna(subset=["snapped_latitude", "snapped_longitude"]).reset_index(drop=True)
            out_df["grid_id"] = range(1, len(out_df) + 1)
            print(f"  snapped ok: {len(out_df)} / {before} (failed={failed})")
        else:
            print(f"  snapped ok: {len(out_df) - failed} / {len(out_df)} (failed={failed})")

        out_name = f"{in_path_p.stem}_metadata.csv"
        out_path = out_dir / out_name
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()


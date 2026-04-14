#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
시나리오 폴더 아래의 config_*.yaml 파일을 재귀적으로 찾아
sim_src/main.py를 순서대로 실행하는 배치 러너.

예시:
  python MCI_ADV2/sim_src/run_all_configs.py ^
    --scenarios_dir "C:\\Users\\user00\\Desktop\\MCI_Diffusion\\MCI_ADV2\\scenarios\\daejeon_3000_random"
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


def find_config_files(scenarios_dir: Path):
    return sorted(
        p for p in scenarios_dir.rglob("config*.yaml")
        if p.is_file()
    )


def load_output_dir(config_path: Path) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    run_setting = configs.get("run_setting", {})
    output_path = run_setting.get("output_path")
    exp_indicator = run_setting.get("exp_indicator")

    if not output_path or not exp_indicator:
        return Path()

    return Path(output_path) / str(exp_indicator)


def main():
    parser = argparse.ArgumentParser(description="config_*.yaml 배치 실행")
    parser.add_argument("--scenarios_dir", required=True, help="config yaml들이 들어있는 시나리오 루트 폴더")
    parser.add_argument(
        "--main_script",
        default=None,
        help="실행할 main.py 경로. 기본값은 현재 파일과 같은 폴더의 main.py",
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="main.py 실행에 사용할 파이썬",
    )
    parser.add_argument(
        "--skip_if_result_exists",
        action="store_true",
        help="결과 폴더가 이미 있으면 해당 config는 건너뜀",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="하나라도 실패하면 즉시 중단",
    )
    args = parser.parse_args()

    scenarios_dir = Path(args.scenarios_dir).resolve()
    if not scenarios_dir.exists():
        raise FileNotFoundError(f"시나리오 폴더를 찾을 수 없습니다: {scenarios_dir}")

    main_script = Path(args.main_script).resolve() if args.main_script else Path(__file__).with_name("main.py")
    if not main_script.exists():
        raise FileNotFoundError(f"main.py를 찾을 수 없습니다: {main_script}")

    config_files = find_config_files(scenarios_dir)
    if not config_files:
        raise FileNotFoundError(f"config*.yaml 파일을 찾지 못했습니다: {scenarios_dir}")

    print("=" * 70)
    print(f"시나리오 폴더: {scenarios_dir}")
    print(f"실행 스크립트: {main_script}")
    print(f"config 개수: {len(config_files)}")
    print("=" * 70)

    success_count = 0
    failure_count = 0
    failures = []
    total_start = time.time()

    for idx, config_path in enumerate(config_files, start=1):
        result_dir = load_output_dir(config_path)
        result_stat = result_dir / f"results_{result_dir.name}_stat.txt" if result_dir else None

        if args.skip_if_result_exists and result_stat and result_stat.exists():
            print(f"[{idx}/{len(config_files)}] SKIP {config_path}")
            print(f"  기존 결과 발견: {result_stat}")
            continue

        print(f"[{idx}/{len(config_files)}] RUN  {config_path}")
        start_t = time.time()
        result = subprocess.run(
            [
                args.python_bin,
                str(main_script),
                "--config_path",
                str(config_path),
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        elapsed = round(time.time() - start_t, 2)
        if result.returncode == 0:
            success_count += 1
            print(f"  완료 ({elapsed}초)")
        else:
            failure_count += 1
            failures.append((str(config_path), result.returncode))
            print(f"  실패 ({elapsed}초, returncode={result.returncode})")
            if args.stop_on_error:
                break

    total_elapsed = round(time.time() - total_start, 2)
    print("=" * 70)
    print("배치 실행 요약")
    print(f"성공: {success_count}")
    print(f"실패: {failure_count}")
    print(f"총 소요 시간(초): {total_elapsed}")
    if failures:
        print("실패 목록:")
        for path, returncode in failures[:20]:
            print(f"  - returncode={returncode}: {path}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20}건 추가")
    print("=" * 70)


if __name__ == "__main__":
    main()

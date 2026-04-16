import os
import re
import csv
from glob import glob
from typing import Dict, Tuple, Optional, List

# 개별 실행 결과(.txt)를 펼쳐서 데이터셋 생성
# - 위치당 run별 1행 생성
# - pdr + red/yellow/green/black + N 저장

RESULTS_ROOT = r"./MCI_ADV2/results/custom_daejeon_grid"
METADATA_CSV = r"./MCI_ADV2/scenarios/daejeon_daejeon_grid/experiment_metadata.csv"

OUT_CSV = "src/data/daejeon_grid_dataset_30runs.csv"

# .txt 파일에서 각 값이 있는 줄 인덱스 (0-based)
PDR_LINE_INDEX = 2
RED_LINE_INDEX = 5
YELLOW_LINE_INDEX = 6
GREEN_LINE_INDEX = 7
BLACK_LINE_INDEX = 8

RUNS_PER_POINT = None  # 위치 당 최대 run 횟수. 전체 사용이면 None

FOLDER_RE = re.compile(r"lat(?P<lat>-?\d+(?:\.\d+)?)_lon(?P<lon>-?\d+(?:\.\d+)?)")
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_run_values_line(line: str) -> List[float]:
    """
    예:
    START, RedOnly, Red OnlyAMB, Yellow OnlyAMB  0.01 0.02 0.03 ...
    -> 뒤쪽 run 값들만 float 리스트로 반환
    """
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"라인 파싱 실패: {line}")

    values = []
    for tok in parts[4:]:
        try:
            values.append(float(tok))
        except ValueError:
            for m in FLOAT_RE.findall(tok):
                values.append(float(m))

    if not values:
        raise ValueError(f"run 값이 비어있습니다: {line}")
    return values


def load_metadata_N(path: str) -> Dict[Tuple[float, float], int]:
    """
    experiment_metadata.csv에서
    (snapped_latitude, snapped_longitude) -> N 매핑
    """
    mapping: Dict[Tuple[float, float], int] = {}
    if not os.path.exists(path):
        print(f"[WARN] metadata CSV not found, N 컬럼 없이 진행합니다: {path}")
        return mapping

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"snapped_latitude", "snapped_longitude", "N"}
        if not required.issubset(reader.fieldnames or []):
            print(f"[WARN] metadata CSV 컬럼 부족({reader.fieldnames}), N 매핑 생략")
            return mapping

        for row in reader:
            try:
                slat = round(float(row["snapped_latitude"]), 6)
                slon = round(float(row["snapped_longitude"]), 6)
                n_val = int(float(row["N"]))
            except Exception:
                continue
            mapping[(slat, slon)] = n_val

    print(f"[INFO] metadata N 매핑 로드 완료: {len(mapping)}개 좌표")
    return mapping


def get_line_values(lines: List[str], idx: int, name: str) -> List[float]:
    if idx >= len(lines):
        raise ValueError(f"{name} line index 범위 초과: idx={idx}, len={len(lines)}")
    return parse_run_values_line(lines[idx])


def main() -> None:
    rows = []
    subfolders = [p for p in glob(os.path.join(RESULTS_ROOT, "lat*_lon*")) if os.path.isdir(p)]
    latlon_to_n = load_metadata_N(METADATA_CSV)

    print("RUNNING:", __file__, flush=True)
    print("CWD:", os.getcwd(), flush=True)
    print("FOLDERS:", len(subfolders), flush=True)

    skipped_no_txt = 0
    skipped_parse = 0
    total_runs = 0

    for folder in subfolders:
        m = FOLDER_RE.search(os.path.basename(folder))
        if not m:
            continue

        lat = float(m.group("lat"))
        lon = float(m.group("lon"))

        txt_candidates = [
            p for p in glob(os.path.join(folder, "*.txt"))
            if not p.endswith("_stat.txt")
        ]
        if not txt_candidates:
            skipped_no_txt += 1
            continue

        txt_path = txt_candidates[0]
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]

        try:
            pdr_runs = get_line_values(lines, PDR_LINE_INDEX, "pdr")
            red_runs = get_line_values(lines, RED_LINE_INDEX, "red")
            yellow_runs = get_line_values(lines, YELLOW_LINE_INDEX, "yellow")
            green_runs = get_line_values(lines, GREEN_LINE_INDEX, "green")
            black_runs = get_line_values(lines, BLACK_LINE_INDEX, "black")
        except Exception:
            skipped_parse += 1
            continue

        run_count = min(
            len(pdr_runs),
            len(red_runs),
            len(yellow_runs),
            len(green_runs),
            len(black_runs),
        )

        if run_count == 0:
            skipped_parse += 1
            continue

        if RUNS_PER_POINT is not None:
            run_count = min(run_count, RUNS_PER_POINT)

        key = (round(lat, 6), round(lon, 6))
        n_val: Optional[int] = latlon_to_n.get(key)

        for run_idx in range(run_count):
            rows.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "pdr": float(pdr_runs[run_idx]),
                    "run_idx": run_idx,
                    "N": n_val,
                    "red": float(red_runs[run_idx]),
                    "yellow": float(yellow_runs[run_idx]),
                    "green": float(green_runs[run_idx]),
                    "black": float(black_runs[run_idx]),
                    "source_folder": os.path.basename(folder),
                    "txt_file": os.path.basename(txt_path),
                }
            )

        total_runs += run_count

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = [
        "lat",
        "lon",
        "pdr",
        "run_idx",
        "N",
        "red",
        "yellow",
        "green",
        "black",
        "source_folder",
        "txt_file",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[INFO] skipped (no txt): {skipped_no_txt}")
    print(f"[INFO] skipped (parse): {skipped_parse}")
    print(f"[INFO] total runs written: {total_runs}")
    print(f"Saved: {OUT_CSV} (n={len(rows)})")


if __name__ == "__main__":
    main()
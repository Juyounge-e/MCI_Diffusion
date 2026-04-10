import os
import re
import csv
from glob import glob
from typing import Dict, Tuple, Optional

# 새 실험 결과/메타데이터 경로
RESULTS_ROOT = r"./MCI_ADV2/results/daejeon_3000_random"
METADATA_CSV = r"./MCI_ADV2/scenarios/daejeon_3000_random/experiment_metadata.csv"

OUT_CSV = "src/data/test.csv"

PDR_LINE_INDEX = 2
PDR_WoG_LINE_INDEX = 4

RED_LINE_INDEX = 5
YELLOW_LINE_INDEX = 6
GREEN_LINE_INDEX = 7
BLACK_LINE_INDEX = 8

# 결과 폴더명에서 위도/경도 파싱용
FOLDER_RE = re.compile(r"lat(?P<lat>-?\d+(\.\d+)?)_lon(?P<lon>-?\d+(\.\d+)?)")


def parse_stat_line(line: str):
    """
    START, RedOnly, Red OnlyAMB, Yellow OnlyAMB  0.0436  0.0195  0.0073
    -> (mean, std)
    """
    parts = line.strip().split()
    if len(parts) < 4:
        raise ValueError(f"라인 파싱 실패: {line}")

    mean = float(parts[-3])
    std = float(parts[-2])
    return mean, std


def load_metadata_N(path: str) -> Dict[Tuple[float, float], int]:
    """
    random_metadata.csv에서 (snapped_latitude, snapped_longitude) -> N 매핑을 만든다.
    좌표는 소수점 6자리로 반올림하여 매칭.
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

rows = []
subfolders = [p for p in glob(os.path.join(RESULTS_ROOT, "lat*_lon*")) if os.path.isdir(p)]

# random_metadata.csv 에서 N 정보 로드
latlon_to_N = load_metadata_N(METADATA_CSV)

print("RUNNING:", __file__, flush=True)
print("CWD:", os.getcwd(), flush=True)

for folder in subfolders:
    m = FOLDER_RE.search(os.path.basename(folder))
    if not m:
        continue
    lat = float(m.group("lat"))
    lon = float(m.group("lon"))

    stat_files = glob(os.path.join(folder, "*_stat.txt"))
    if not stat_files:
        print(f"[SKIP] stat 파일 없음: {folder}")
        continue

    stat_path = stat_files[0]  
    with open(stat_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]

    if PDR_LINE_INDEX >= len(lines):
        print(f"[SKIP] pdr_line_index 범위 초과({PDR_LINE_INDEX}): {stat_path}")
        continue

    pdr_mean, pdr_std = parse_stat_line(lines[PDR_LINE_INDEX])

    # 좌표를 소수 6자리로 맞춰서 metadata의 snapped_lat/lon 과 매칭
    key = (round(lat, 6), round(lon, 6))
    N_val: Optional[int] = latlon_to_N.get(key)
    
    red_mean = parse_stat_line(lines[RED_LINE_INDEX])[0] if RED_LINE_INDEX < len(lines) else None
    yellow_mean = parse_stat_line(lines[YELLOW_LINE_INDEX])[0] if YELLOW_LINE_INDEX < len(lines) else None
    green_mean = parse_stat_line(lines[GREEN_LINE_INDEX])[0] if GREEN_LINE_INDEX < len(lines) else None
    black_mean = parse_stat_line(lines[BLACK_LINE_INDEX])[0] if BLACK_LINE_INDEX < len(lines) else None


    rows.append(
        {
            "lat": lat,
            "lon": lon,
            "pdr_mean": pdr_mean,
            "pdr_std": pdr_std,
            "N": N_val,
            "red": red_mean,
            "yellow": yellow_mean,
            "green": green_mean,
            "black": black_mean,
            "source_folder": os.path.basename(folder),
            "stat_file": os.path.basename(stat_path),
        }
    )

# 저장
fieldnames = [
    "lat",
    "lon",
    "pdr_mean",
    "pdr_std",
    "N",
    "red",
    "yellow",
    "green",
    "black",
    "source_folder",
    "stat_file",
]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"Saved: {OUT_CSV}  (n={len(rows)})")
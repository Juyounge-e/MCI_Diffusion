import os
import re
import csv
from glob import glob

RESULTS_ROOT = r"./simul_data/results/exp_20260108_115527"  
OUT_CSV = "dataset.csv"

PDR_LINE_INDEX = 2   

FOLDER_RE = re.compile(r"lat(?P<lat>-?\d+(\.\d+)?)_lon(?P<lon>-?\d+(\.\d+)?)") # 위 경도 

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
    # ci_half = float(parts[-1])
    # rule_name = " ".join(parts[:-3])
    return mean, std

rows = []
subfolders = [p for p in glob(os.path.join(RESULTS_ROOT, "lat*_lon*")) if os.path.isdir(p)]

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

    rows.append({
        "lat": lat,
        "lon": lon,
        "pdr_mean": pdr_mean,
        "pdr_std": pdr_std,
        # "pdr_ci_half": pdr_ci_half,
       #  "rule_name": rule_name,
        "source_folder": os.path.basename(folder),
        "stat_file": os.path.basename(stat_path),
    })

# 저장
fieldnames = ["lat", "lon", "pdr_mean", "pdr_std", "source_folder", "stat_file"]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"Saved: {OUT_CSV}  (n={len(rows)})")
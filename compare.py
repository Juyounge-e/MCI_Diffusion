from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----- 1. 여러 CSV 불러와서 병합 -----
root = Path(".")  # 최상위 디렉토리 기준
csv_files = sorted(root.glob("q3_*_eval_with_metrics.csv"))

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    # 파일명에서 config 추출: 예) q3_10_100_eval_with_metrics -> ["q3", "10", "100", "eval", ...]
    stem_parts = f.stem.split("_")
    # q3_10_100_eval_with_metrics, q3_10_max_eval_with_metrics 모두 처리 가능하게
    if len(stem_parts) >= 3:
        config = f"{stem_parts[1]}_{stem_parts[2]}"
    else:
        config = "unknown"

    df["config"] = config
    df["source_file"] = f.name
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# ----- 2. 새로운 tol 기준 within_tol 계산 -----
tols = [0.005, 0.0075]
for tol in tols:
    col_name = f"within_tol_{tol}"
    all_df[col_name] = all_df["abs_err"].abs() <= tol

# ----- 3. config별 요약 테이블 만들기 -----
group_cols = ["config"]

agg_dict = {
    "abs_err": "mean",
}
for tol in tols:
    agg_dict[f"within_tol_{tol}"] = "mean"

summary = all_df.groupby(group_cols).agg(agg_dict)

# 평균 abs_err 이름 변경
summary = summary.rename(columns={"abs_err": "mean_abs_err"})

# within tol 비율을 퍼센트(0~100)로 변환
summary["ratio_within_0.005(%)"] = (summary[f"within_tol_{tols[0]}"] * 100).round(1)
summary["ratio_within_0.0075(%)"] = (summary[f"within_tol_{tols[1]}"] * 100).round(1)

# 원래 0~1 비율 컬럼은 정리 차원에서 제거 (원하면 남겨도 됨)
summary = summary.drop(columns=[f"within_tol_{t}" for t in tols])

print("=== config별 요약 테이블 ===")
print(summary)

# 필요하면 CSV로 저장
summary.to_csv("q3_eval_summary_by_config.csv")
print("\n요약 결과를 'q3_eval_summary_by_config.csv'로 저장했습니다.")

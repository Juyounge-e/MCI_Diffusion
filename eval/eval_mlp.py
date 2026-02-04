import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="simul4eval_dataset.csv 기반으로 cond vs 엔진 pdr_mean 간 오차를 평가하는 스크립트"
    )
    parser.add_argument(
        "--simul_csv",
        type=str,
        default=str(Path("simul4eval_dataset.csv")),
        help="MCI 시뮬레이션 결과 CSV (lat, lon, pdr_mean, ...)",
    )
    parser.add_argument(
        "--cond",
        type=float,
        default=None,
        help="CSV에 cond 컬럼이 없으면 전체 행에 동일하게 적용할 cond 값을 지정하세요.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("simul4eval_with_metrics.csv")),
        help="cond 및 오차 컬럼이 추가된 결과 CSV 경로",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.005,
        help="허용 오차(|pdr_mean - cond| < tol) 기준값 (default: 0.005)",
    )
    args = parser.parse_args()

    simul_path = Path(args.simul_csv).resolve()
    if not simul_path.exists():
        print(f"simul_csv를 찾을 수 없습니다: {simul_path}")
        return 1

    df = pd.read_csv(simul_path)
    if "pdr_mean" not in df.columns:
        print(f"입력 CSV에 'pdr_mean' 컬럼이 없습니다. 컬럼: {list(df.columns)}")
        return 1

    # cond 설정: CSV에 있으면 사용, 없으면 --cond로 채움
    if "cond" in df.columns:
        if args.cond is not None:
            df["cond"] = float(args.cond)
    else:
        if args.cond is None:
            print(" CSV에 'cond' 컬럼이 없고 --cond 값도 지정되지 않았습니다.")
            return 1
        df["cond"] = float(args.cond)

    # 오차 컬럼
    df["err"] = df["pdr_mean"] - df["cond"]
    df["abs_err"] = df["err"].abs()
    df["sq_err"] = df["err"] ** 2
    tol = float(args.tol)
    df["within_tol"] = df["abs_err"] < tol

    # 
    n = df.shape[0]
    mae = float(df["abs_err"].mean()) if n > 0 else np.nan
    rmse = float(np.sqrt(df["sq_err"].mean())) if n > 0 else np.nan
    bias = float(df["err"].mean()) if n > 0 else np.nan
    within_ratio = float(df["within_tol"].mean()) if n > 0 else np.nan
    # if df["cond"].nunique() > 1 and df["pdr_mean"].nunique() > 1:
    #     pearson = float(df["cond"].corr(df["pdr_mean"]))
    # else:
    #     pearson = np.nan

    out_path = Path(args.out).resolve()
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {out_path} (n={n})")
    if df["cond"].nunique() == 1:
        print(f"   condition    = {float(df['cond'].iloc[0]):.6f}")
    else:
        print(f"   cond 범위 = {df['cond'].min():.6f} ~ {df['cond'].max():.6f}")
    print(f"   MAE     = {mae:.6f}")
    print(f"   RMSE    = {rmse:.6f}")
    print(f"   Bias    = {bias:.6f}  (pdr_mean - condition 평균)")
    print(f"   |err|<{tol:.4f} 비율 = {within_ratio*100:5.1f}% ({int(df['within_tol'].sum())}/{n})")
    # print(f"   Pearson = {pearson:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


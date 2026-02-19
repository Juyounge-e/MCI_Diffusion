import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _set_korean_font() -> None:
    """한글 표시를 위해 사용 가능한 폰트 설정 (Windows: 맑은 고딕 등)."""
    from matplotlib import font_manager
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]
    for name in candidates:
        if name in [f.name for f in font_manager.fontManager.ttflist]:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_real_vs_gen(df_real: pd.DataFrame, df_gen: pd.DataFrame, out_path: Path, cond: float) -> None:
    """실제(시뮬 결과) vs 샘플(생성 조건) 비교. 위경도는 동일하므로 pdr·오차로만 비교."""
    if not _HAS_MPL:
        print("matplotlib 없음: 분포 비교 그림을 건너뜁니다.")
        return
    _set_korean_font()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 1) 동일 위경도 — 색 = 시뮬 결과 pdr_mean (샘플은 이 위경도로 생성 후 시뮬 돌린 결과)
    ax = axes[0]
    if "pdr_mean" in df_real.columns:
        sc = ax.scatter(
            df_real["lon"], df_real["lat"],
            s=10, alpha=0.8, c=df_real["pdr_mean"], cmap="viridis", edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label="pdr_mean (시뮬 결과)")
    else:
        ax.scatter(df_real["lon"], df_real["lat"], s=8, alpha=0.6, c="C0")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title("시뮬 pdr_mean 위경도")
    ax.set_aspect("equal", adjustable="box")

    # 2) pdr 분포: 시뮬 결과 pdr_mean strip + box, 생성 조건(cond) 수평선
    ax = axes[1]
    if "pdr_mean" in df_real.columns:
        pdr = df_real["pdr_mean"].dropna()
        if len(pdr) > 0:
            # Box plot
            bp = ax.boxplot([pdr.values], vert=True, widths=0.3, patch_artist=True,
                            showmeans=False, showfliers=True, labels=["pdr_mean"])
            bp["boxes"][0].set_facecolor("lightblue")
            bp["boxes"][0].set_alpha(0.5)
            for element in ["whiskers", "fliers", "medians", "caps"]:
                for item in bp[element]:
                    item.set_color("C0")
                    item.set_linewidth(1.5)
            # Strip plot
            x_pos = np.ones(len(pdr))
            jitter = np.random.normal(0, 0.03, len(pdr))
            ax.scatter(x_pos + jitter, pdr.values, s=30, alpha=0.6, color="C0",
                       edgecolors="black", linewidth=0.5, zorder=10, label=f"시뮬 pdr_mean (n={len(pdr)})")
            q1 = float(pdr.quantile(0.25))
            q3 = float(pdr.quantile(0.75))
            median = float(pdr.median())
            ax.text(1.35, q1, f"Q1={q1:.4f}", fontsize=8, verticalalignment="center", color="C3")
            ax.text(1.35, q3, f"Q3={q3:.4f}", fontsize=8, verticalalignment="center", color="C4")
            ax.text(1.35, median, f"중앙값={median:.4f}", fontsize=8, verticalalignment="center", color="C5")
            ax.set_xlim(0.5, 1.7)
        ax.axhline(cond, color="C1", linestyle="--", linewidth=2, label=f"생성 조건 cond={cond:.6f}")
    ax.set_ylabel("pdr_mean")
    ax.set_xlabel("")
    ax.set_title("pdr 분포 vs 샘플 생성 조건")
    ax.set_xticks([1.0])
    ax.set_xticklabels(["pdr_mean"])
    ax.legend(loc="best", fontsize=7)

    # 3) 절대 오차 분포: abs_err = |pdr_mean - cond|, strip plot + box plot
    ax = axes[2]
    if "abs_err" in df_real.columns:
        abs_err = df_real["abs_err"].dropna()
        if len(abs_err) > 0:
            # Box plot
            bp = ax.boxplot([abs_err.values], vert=True, widths=0.3, patch_artist=True, 
                           showmeans=False, showfliers=True, labels=["|err|"])
            bp["boxes"][0].set_facecolor("lightblue")
            bp["boxes"][0].set_alpha(0.5)
            for element in ["whiskers", "fliers", "medians", "caps"]:
                for item in bp[element]:
                    item.set_color("C2")
                    item.set_linewidth(1.5)
            # Strip plot: 각 점을 세로로 표시 (약간의 jitter)
            y_pos = [1.0] * len(abs_err)
            jitter = np.random.normal(0, 0.03, len(abs_err))  # 작은 랜덤 오프셋
            ax.scatter(y_pos + jitter, abs_err.values, s=30, alpha=0.6, color="C2", 
                      edgecolors="black", linewidth=0.5, zorder=10, label=f"데이터점 (n={len(abs_err)})")
            # Q1, Q3 값 표시
            q1 = float(abs_err.quantile(0.25))
            q3 = float(abs_err.quantile(0.75))
            median = float(abs_err.median())
            ax.text(1.35, q1, f"Q1={q1:.4f}", fontsize=8, verticalalignment="center", color="C3")
            ax.text(1.35, q3, f"Q3={q3:.4f}", fontsize=8, verticalalignment="center", color="C4")
            ax.text(1.35, median, f"중앙값={median:.4f}", fontsize=8, verticalalignment="center", color="C5")
            ax.set_ylim(bottom=0)
            ax.set_xlim(0.5, 1.7)
    ax.set_ylabel("|err| (abs_err)")
    ax.set_xlabel("")
    ax.set_title("절대 오차 분포")
    ax.set_xticks([1.0])
    ax.set_xticklabels(["|err|"])
    ax.legend(loc="best", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"분포 비교 그림 저장: {out_path}")


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
        "--gen_csv",
        type=str,
        default=None,
        help="sample_mlp.py로 생성한 CSV (lat, lon, N). 지정 시 실제 vs 생성 분포 비교 그림 출력",
    )
    parser.add_argument(
        "--cond",
        type=float,
        default=0.057271,
        help="CSV에 cond 컬럼이 없으면 적용할 cond 값 (생성 데이터 pdr와 동일, default: 0.057456).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("eval_with_metrics.csv")),
        help="cond 및 오차 컬럼이 추가된 결과 CSV 경로",
    )
    parser.add_argument(
        "--out_plot",
        type=str,
        default=None,
        help="실제 vs 생성 분포 비교 그림 경로 (기본: --out과 동일 stem + _distribution.png)",
    )
    # parser.add_argument(
    #     "--tol",
    #     type=float,
    #     default=0.0075,
    #     help="허용 오차(|pdr_mean - cond| < tol) 기준값 (default: 0.005)",
    # )
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
    if "cond" not in df.columns:
        df["cond"] = float(args.cond)

    # 오차 컬럼
    df["err"] = df["pdr_mean"] - df["cond"]
    df["abs_err"] = df["err"].abs()
    df["sq_err"] = df["err"] ** 2
    tol = float(args.tol)
    df["within_tol"] = df["abs_err"] < tol

    n = df.shape[0]
    mae = float(df["abs_err"].mean()) if n > 0 else np.nan
    rmse = float(np.sqrt(df["sq_err"].mean())) if n > 0 else np.nan
    pdr_range = float(df["pdr_mean"].max() - df["pdr_mean"].min()) if n > 0 and df["pdr_mean"].max() > df["pdr_mean"].min() else np.nan
    nrmse = float(rmse / pdr_range) if n > 0 and not np.isnan(pdr_range) and pdr_range > 0 else np.nan
    bias = float(df["err"].mean()) if n > 0 else np.nan
    within_ratio = float(df["within_tol"].mean()) if n > 0 else np.nan
    # abs_err 분포 기준: Q1, Q3 미만인 비율(개수)
    q1_abs = float(df["abs_err"].quantile(0.25)) if n > 0 else np.nan
    q3_abs = float(df["abs_err"].quantile(0.75)) if n > 0 else np.nan
    n_lt_q1 = int((df["abs_err"] <= q1_abs).sum()) if n > 0 else 0
    n_lt_q3 = int((df["abs_err"] <= q3_abs).sum()) if n > 0 else 0
    ratio_lt_q1 = (n_lt_q1 / n * 100) if n > 0 else np.nan
    ratio_lt_q3 = (n_lt_q3 / n * 100) if n > 0 else np.nan
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
    # print(f"   NRMSE   = {nrmse:.6f}" if not np.isnan(nrmse) else "   NRMSE   = (pdr 범위 0이라 불가)")
    print(f"   Bias    = {bias:.6f}  (pdr_mean - condition)")
    print(f"   |err|<{tol:.6f} 비율 = {within_ratio*100:5.1f}% ({int(df['within_tol'].sum())}/{n})")
    if n > 0 and not np.isnan(q1_abs):
        print(f"   |err| 분포 기준  Q1={q1_abs:.6f}  Q3={q3_abs:.6f}")
        print(f"   |err|<Q1 비율 = {ratio_lt_q1:5.1f}% ({n_lt_q1}/{n})")
        print(f"   |err|<Q3 비율 = {ratio_lt_q3:5.1f}% ({n_lt_q3}/{n})")
    # print(f"   Pearson = {pearson:.6f}")

    # 실제 vs 생성 분포 비교 그림 (--gen_csv 지정 시)
    if args.gen_csv:
        gen_path = Path(args.gen_csv).resolve()
        if not gen_path.exists():
            print(f"gen_csv를 찾을 수 없어 그림을 건너뜁니다: {gen_path}")
        else:
            df_gen = pd.read_csv(gen_path)
            for col in ("lat", "lon"):
                if col not in df_gen.columns:
                    print(f"gen_csv에 '{col}' 컬럼이 없어 그림을 건너뜁니다.")
                    break
            else:
                out_plot = Path(args.out_plot) if args.out_plot else out_path.with_stem(out_path.stem + "_distribution").with_suffix(".png")
                out_plot.parent.mkdir(parents=True, exist_ok=True)
                cond = float(df["cond"].iloc[0]) if "cond" in df.columns and len(df) else float(args.cond)
                plot_real_vs_gen(df, df_gen, out_plot, cond=cond)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


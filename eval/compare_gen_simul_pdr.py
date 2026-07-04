"""
같은 (원본)좌표에 대해 gen(diffusion) vs simul에 대해서 
random_coordinate 메타데이터로 정확 조인하여 pdr 값을 기준으로 성능 지표 계산 

조인 체인 (3-way):
    gen_csv(원본좌표 + pdr조건)
       ⋈ meta_csv(custom/random == 원본좌표) → snapped 좌표 획득
       ⋈ simul_csv(snapped == 시뮬 좌표, pdr_mean)
    ⇒ (gen_pdr, simul_pdr) 정확 매칭  (스냅 안 된 점은 자동 제외)

지표: MSE / RMSE / NRMSE / MAE / Bias / |err| 분포 / Pearson

단일 비교:
    python eval/compare_gen_simul_pdr.py \
        --gen_csv   outputs/mlp_diffusion/resolution_roadloss/roadloss_all.csv \
        --simul_csv src/data/roadloss/national_100.csv \
        --meta_csv  MCI_ADV2/scenarios/custom_xxx/custom_metadata.csv \
        --out eval/eval_results/gen_vs_simul.csv

분위(quantile bin)별 비교:
    python eval/compare_gen_simul_pdr.py \
        --gen_glob  "outputs/mlp_diffusion/resolution_roadloss_daejeon/q*.csv" \
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.road_network import RoadNetwork


def _find_col(df: pd.DataFrame, candidates) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def _metrics(gen_pdr: np.ndarray, simul_pdr: np.ndarray):
    """simul - gen 기준 오차 지표. (gen=요청 조건, simul=실제 결과)
    NRMSE 분모는 항상 simul pdr 의 range(max-min)."""
    err = simul_pdr - gen_pdr
    abs_err = np.abs(err)
    sq_err = err ** 2

    mae = float(abs_err.mean())
    mse = float(sq_err.mean())
    rmse = float(np.sqrt(mse))
    bias = float(err.mean())

    denom = float(simul_pdr.max() - simul_pdr.min())     # NRMSE = RMSE / (simul max-min)
    nrmse = float(rmse / denom) if denom > 0 else np.nan

    pearson = (
        float(np.corrcoef(gen_pdr, simul_pdr)[0, 1])
        if np.std(gen_pdr) > 0 and np.std(simul_pdr) > 0 else np.nan
    )
    return {
        "n": len(err), "MAE": mae, "MSE": mse, "RMSE": rmse,
        "NRMSE": nrmse, "NRMSE_denom(range)": denom,
        "Bias": bias, "|err|_median": float(np.median(abs_err)),
        "|err|_Q1": float(np.quantile(abs_err, 0.25)),
        "|err|_Q3": float(np.quantile(abs_err, 0.75)),
        "Pearson": pearson,
    }, err, abs_err, sq_err


def _offroad_stats(lat: np.ndarray, lon: np.ndarray, rn: RoadNetwork, thresholds) -> dict:
    """생성 좌표(lat, lon)가 도로에서 얼마나 떨어져 있는지(off-road 비율 등)."""
    d = rn.snap(lat, lon).dist_m
    stats = {
        "road_dist_mean_m": float(d.mean()),
        "road_dist_median_m": float(np.median(d)),
    }
    for thr in thresholds:
        stats[f"onroad_{int(thr)}m_pct"] = float((d <= thr).mean() * 100)
    return stats


def _add_keys(df: pd.DataFrame, lat: str, lon: str, pre, precision: int) -> pd.DataFrame:
    df[pre[0]] = pd.to_numeric(df[lat], errors="coerce").round(precision)
    df[pre[1]] = pd.to_numeric(df[lon], errors="coerce").round(precision)
    return df


def _prepare_sim_meta(sim_csv: str, meta_csv: str, simul_pdr_col: str, precision: int):
    """simul_csv / meta_csv 는 gen 파일과 무관하게 한 번만 준비해서 재사용."""
    sim = pd.read_csv(sim_csv)
    meta = pd.read_csv(meta_csv)

    if simul_pdr_col not in sim.columns:
        raise ValueError(f"simul_csv 에 '{simul_pdr_col}' 없음. 컬럼: {list(sim.columns)}")
    if "lat" not in sim.columns or "lon" not in sim.columns:
        raise ValueError(f"simul_csv 에 lat/lon 없음. 컬럼: {list(sim.columns)}")

    o_lat = _find_col(meta, ["custom_latitude", "random_latitude", "orig_latitude", "lat"])
    o_lon = _find_col(meta, ["custom_longitude", "random_longitude", "orig_longitude", "lon"])
    s_lat = _find_col(meta, ["snapped_latitude", "snap_latitude"])
    s_lon = _find_col(meta, ["snapped_longitude", "snap_longitude"])
    if not all([o_lat, o_lon, s_lat, s_lon]):
        raise ValueError(
            f"meta_csv 좌표 컬럼 감지 실패.\n  원본:({o_lat},{o_lon})  스냅:({s_lat},{s_lon})"
            f"\n  컬럼: {list(meta.columns)}"
        )
    print(f"  meta 컬럼  원본=({o_lat},{o_lon})  스냅=({s_lat},{s_lon})")

    meta = meta.dropna(subset=[s_lat, s_lon]).copy()
    meta = _add_keys(meta, o_lat, o_lon, ("ko_lat", "ko_lon"), precision)
    meta = _add_keys(meta, s_lat, s_lon, ("ks_lat", "ks_lon"), precision)
    sim = _add_keys(sim, "lat", "lon", ("ks_lat", "ks_lon"), precision)

    m = meta[["ko_lat", "ko_lon", "ks_lat", "ks_lon"]].drop_duplicates()
    s = sim[["ks_lat", "ks_lon", simul_pdr_col]].rename(columns={simul_pdr_col: "simul_pdr"})
    s = s.groupby(["ks_lat", "ks_lon"], as_index=False)["simul_pdr"].mean()  # 동일 좌표 다중시뮬 평균
    return m, s


def _merge_gen(gen: pd.DataFrame, gen_pdr_col: str, m: pd.DataFrame, s: pd.DataFrame, precision: int) -> pd.DataFrame:
    """gen(원본좌표+pdr조건) ⋈ meta(원본키)→스냅키 ⋈ simul(스냅키) → (gen_pdr, simul_pdr)."""
    if gen_pdr_col not in gen.columns:
        raise ValueError(f"gen_csv 에 '{gen_pdr_col}' 없음. 컬럼: {list(gen.columns)}")
    if "lat" not in gen.columns or "lon" not in gen.columns:
        raise ValueError(f"gen_csv 에 lat/lon 없음. 컬럼: {list(gen.columns)}")

    gen = _add_keys(gen, "lat", "lon", ("k_lat", "k_lon"), precision)
    g = gen[["k_lat", "k_lon", gen_pdr_col]].rename(
        columns={"k_lat": "ko_lat", "k_lon": "ko_lon", gen_pdr_col: "gen_pdr"})

    gm = g.merge(m, on=["ko_lat", "ko_lon"], how="inner")
    merged = gm.merge(s, on=["ks_lat", "ks_lon"], how="inner")
    merged = merged.dropna(subset=["gen_pdr", "simul_pdr"]).reset_index(drop=True)
    return merged


def _print_stats(label: str, stats: dict):
    print("=" * 56)
    print(f"  [{label}]")
    print("-" * 56)
    for k, v in stats.items():
        print(f"  {k:<22}= {v:.6f}" if isinstance(v, float) else f"  {k:<22}= {v}")
    print("=" * 56)


def _save_plot(merged: pd.DataFrame, stats: dict, out_plot: Path, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for _name in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"):
            if _name in [f.name for f in font_manager.fontManager.ttflist]:
                plt.rcParams["font.family"] = _name
                break
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(merged["gen_pdr"], merged["simul_pdr"], s=18, alpha=0.6, edgecolors="none")
        lo = float(min(merged["gen_pdr"].min(), merged["simul_pdr"].min()))
        hi = float(max(merged["gen_pdr"].max(), merged["simul_pdr"].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y=x")
        ax.set_xlabel("gen pdr (요청)"); ax.set_ylabel("simul pdr_mean (실제)")
        ax.set_title(f"{title} (n={stats['n']}, RMSE={stats['RMSE']:.4f})")
        ax.legend(); ax.set_aspect("equal", adjustable="box")
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_plot, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"산점도 저장: {out_plot}")
    except ImportError:
        print("matplotlib 없음: 산점도 건너뜀")


_QBIN_RE = re.compile(r"q(\d+)_")


def _qbin_sort_key(path: str):
    m = _QBIN_RE.search(Path(path).name)
    return int(m.group(1)) if m else 10 ** 9


def main() -> int:
    ap = argparse.ArgumentParser(description="gen vs simul pdr 오차 비교 (metadata 조인)")
    ap.add_argument("--gen_csv", default=None, help="생성 CSV 단일 파일 (원본 lat, lon, pdr ...)")
    ap.add_argument("--gen_glob", default=None,
                    help="분위별 생성 CSV glob 패턴 (예: '.../q*.csv'). 파일별 + 전체합본(ALL) 지표를 함께 출력.")
    ap.add_argument("--simul_csv", required=True, help="시뮬 결과 CSV (snapped lat, lon, pdr_mean ...)")
    ap.add_argument("--meta_csv", required=True,
                    help="random_coordinate 메타데이터 CSV (원본 custom/random + snapped 좌표)")
    ap.add_argument("--gen_pdr_col", default="pdr", help="gen_csv 의 pdr(조건) 컬럼명")
    ap.add_argument("--simul_pdr_col", default="pdr_mean", help="simul_csv 의 pdr 컬럼명")
    ap.add_argument("--precision", type=int, default=6, help="좌표 조인 시 lat/lon 반올림 소수자리")
    ap.add_argument("--road_cache", default=None,
                    help="도로 캐시 pkl 경로 (지정 시 gen 좌표의 off-road 비율도 같이 출력)")
    ap.add_argument("--offroad_thresholds", type=float, nargs="+", default=[10, 30, 50, 100, 250],
                    help="on-road 판정 임계값(m) 목록")
    ap.add_argument("--out", default=None,
                    help="--gen_csv: 행별 오차 CSV 저장 경로 / --gen_glob: 분위별+ALL 지표 요약 CSV 저장 경로")
    ap.add_argument("--out_plot", default=None, help="gen vs simul 산점도 저장 경로(--gen_csv 단일 모드에서만 사용)")
    args = ap.parse_args()

    if not args.gen_csv and not args.gen_glob:
        print("[오류] --gen_csv 또는 --gen_glob 중 하나는 필요합니다."); return 1

    try:
        m, s = _prepare_sim_meta(args.simul_csv, args.meta_csv, args.simul_pdr_col, args.precision)
    except ValueError as e:
        print(f"[오류] {e}"); return 1

    rn = RoadNetwork.from_cache(args.road_cache) if args.road_cache else None

    # ---- 분위(quantile bin)별 비교 ----
    if args.gen_glob:
        paths = sorted(glob.glob(args.gen_glob), key=_qbin_sort_key)
        if not paths:
            print(f"[오류] --gen_glob 패턴에 매칭되는 파일 없음: {args.gen_glob}"); return 1

        rows = []
        all_merged = []
        all_gen = []
        for p in paths:
            label = Path(p).stem
            gen = pd.read_csv(p)
            try:
                merged = _merge_gen(gen, args.gen_pdr_col, m, s, args.precision)
            except ValueError as e:
                print(f"[오류] {label}: {e}"); return 1
            if len(merged) == 0:
                print(f"  [경고] {label}: 매칭 0개 → 건너뜀")
                continue
            stats, *_ = _metrics(merged["gen_pdr"].to_numpy(float), merged["simul_pdr"].to_numpy(float))
            if rn is not None:
                # off-road: 조인 후 생존자가 아닌 diffusion 생성 전체 좌표 기준
                stats.update(_offroad_stats(
                    gen["lat"].to_numpy(float), gen["lon"].to_numpy(float),
                    rn, args.offroad_thresholds))
            _print_stats(label, stats)
            rows.append({"bin": label, **stats})
            all_merged.append(merged)
            all_gen.append(gen[["lat", "lon"]])

        if all_merged:
            combined = pd.concat(all_merged, ignore_index=True)
            stats, *_ = _metrics(combined["gen_pdr"].to_numpy(float), combined["simul_pdr"].to_numpy(float))
            if rn is not None:
                combined_gen = pd.concat(all_gen, ignore_index=True)
                stats.update(_offroad_stats(
                    combined_gen["lat"].to_numpy(float), combined_gen["lon"].to_numpy(float),
                    rn, args.offroad_thresholds))
            _print_stats("ALL", stats)
            rows.append({"bin": "ALL", **stats})

        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False))
        if args.out:
            out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"분위별 요약 저장: {out}")
        if args.out_plot:
            print("  [안내] --gen_glob 모드에서는 --out_plot 미지원 (단일 --gen_csv 모드에서 사용).")
        return 0

    # ---- 단일 파일 비교 (기존 동작) ----
    gen = pd.read_csv(args.gen_csv)
    try:
        merged = _merge_gen(gen, args.gen_pdr_col, m, s, args.precision)
    except ValueError as e:
        print(f"[오류] {e}"); return 1
    if len(merged) == 0:
        print("  [오류] 매칭 0개. precision 조정 또는 meta/simul 좌표가 정말 같은지 확인.")
        return 1

    stats, err, abs_err, sq_err = _metrics(merged["gen_pdr"].to_numpy(float), merged["simul_pdr"].to_numpy(float))
    if rn is not None:
        # off-road: 조인 후 생존자가 아닌 diffusion 생성 전체 좌표 기준
        stats.update(_offroad_stats(
            gen["lat"].to_numpy(float), gen["lon"].to_numpy(float),
            rn, args.offroad_thresholds))
    print(f"  gen pdr 컬럼  : {args.gen_pdr_col}   (요청 조건)")
    print(f"  simul pdr 컬럼: {args.simul_pdr_col} (실제 결과)")
    _print_stats(Path(args.gen_csv).stem, stats)

    if args.out:
        merged["err"] = err; merged["abs_err"] = abs_err; merged["sq_err"] = sq_err
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"행별 결과 저장: {out}")

    if args.out_plot:
        _save_plot(merged, stats, Path(args.out_plot), "gen vs simul pdr")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

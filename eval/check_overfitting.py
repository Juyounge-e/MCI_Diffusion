"""
생성 좌표가 학습 데이터(daejeon_random.csv 등)를 암기(거의 그대로 복제)한 건 아닌지 확인.

val loss는 "실제 데이터에 노이즈를 섞었을 때 그걸 잘 복원하는가"만 측정하므로,
"조건만 주고 순수 노이즈에서부터 새로 생성했을 때 학습 데이터를 그대로 베끼는가"
(=암기/모드붕괴)는 직접 잡아내지 못한다. 이 스크립트는 생성 결과와 학습 데이터를
실제로 비교해서 그걸 직접 확인한다.

방법:
    1) 각 생성 좌표 -> 학습 데이터 전체 중 최근접 거리(m) 계산
    2) 학습 데이터 자기 자신끼리의 최근접 거리(자기 제외)도 계산 -> "정상 데이터 밀도에서
       기대되는 거리" 기준선으로 사용
    3) (2) 대비 (1)이 비정상적으로 작으면(거의 0) 암기 의심

거리 계산은 lat/lon을 Daejeon 중심 위도 기준 평면(equirectangular)으로 근사 변환해서
meter 단위로 처리 (지역 범위가 작아 오차 무시 가능, shapely/pyproj 불필요 -> 모든 conda env에서 실행 가능).

사용 예:
    python eval/check_memorization.py \
        --train_csv src/data/national/daejeon_random.csv \
        --gen_glob "outputs/mlp_diffusion/resolution_*/all_bins.csv" \
        --out eval/eval_results/memorization_check.csv
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_R = 6371000.0


def to_local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float):
    """위도 lat0 중심 평면(meter) 근사 변환. 소지역 비교용으로 충분한 정확도."""
    x = np.radians(lon) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat) * EARTH_R
    return x, y


def main() -> int:
    ap = argparse.ArgumentParser(description="생성 좌표의 학습 데이터 암기(memorization) 여부 확인")
    ap.add_argument("--train_csv", required=True, help="학습용 CSV (lat, lon 포함)")
    ap.add_argument("--gen_glob", required=True, help="생성 결과 all_bins.csv glob 패턴")
    ap.add_argument("--near_dup_thresholds", type=float, nargs="+", default=[1.0, 5.0, 10.0],
                    help="근접-복제 판정 임계값(m) 목록")
    ap.add_argument("--out", default=None, help="요약 결과 CSV 저장 경로(선택)")
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv)
    lat0 = float(train["lat"].mean())
    tx, ty = to_local_xy(train["lat"].to_numpy(float), train["lon"].to_numpy(float), lat0)
    train_xy = np.column_stack([tx, ty])
    train_tree = cKDTree(train_xy)

    # 기준선: 학습 데이터 자기 자신끼리의 최근접 거리(자기 제외, k=2)
    self_dist, _ = train_tree.query(train_xy, k=2)
    self_nn = self_dist[:, 1]
    print("=" * 70)
    print(f"[기준선] 학습 데이터({len(train):,}개) 자기-최근접 거리(m): "
          f"median={np.median(self_nn):.1f} mean={self_nn.mean():.1f} p10={np.percentile(self_nn,10):.1f}")
    print("=" * 70)

    paths = sorted(glob.glob(args.gen_glob))
    if not paths:
        print(f"[오류] --gen_glob 매칭 없음: {args.gen_glob}"); return 1

    rows = []
    for p in paths:
        label = Path(p).parent.name
        gen = pd.read_csv(p)
        if "lat" not in gen.columns or "lon" not in gen.columns:
            print(f"  [경고] {label}: lat/lon 없음, 건너뜀"); continue

        gx, gy = to_local_xy(gen["lat"].to_numpy(float), gen["lon"].to_numpy(float), lat0)
        gen_xy = np.column_stack([gx, gy])
        d, _ = train_tree.query(gen_xy, k=1)

        row = {
            "label": label, "n": len(d),
            "nn_dist_median_m": float(np.median(d)),
            "nn_dist_mean_m": float(d.mean()),
            "nn_dist_p10_m": float(np.percentile(d, 10)),
        }
        for thr in args.near_dup_thresholds:
            row[f"near_dup_pct(<{thr:g}m)"] = float((d <= thr).mean() * 100)
        rows.append(row)
        print(f"  {label:<40} n={len(d):>4}  median={row['nn_dist_median_m']:>7.1f}m  "
              f"mean={row['nn_dist_mean_m']:>7.1f}m  <5m={row.get('near_dup_pct(<5m)', float('nan')):>5.1f}%")

    summary = pd.DataFrame(rows)
    print("=" * 70)
    print(f"[기준선] 학습 데이터 자기-최근접 거리 median={np.median(self_nn):.1f}m "
          f"(생성 좌표가 이보다 훨씬 작으면 암기 의심)")

    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"요약 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================
# 설정
# =============================================
sample_dir = r'C:\Users\user00\Desktop\MCI_Diffusion\outputs\mlp_diffusion\rygb_resolution_30runs'
SAMPLE_CSVS = sorted(glob.glob(os.path.join(sample_dir, '*.csv')))

DATA_PATH_GRID = Path(r'C:\Users\user00\Desktop\MCI_Diffusion\daejeon_grid_dataset_30runs_15idx.csv')
GRID_META_PATH = r'C:\Users\user00\Desktop\MCI_Diffusion\MCI_ADV2\scenarios\daejeon_daejeon_grid\grid_metadata.csv'
SUMMARY_PATH = r"C:\Users\user00\Desktop\MCI_Diffusion\notebooks\outputs\analysis\30_test_drop_pdr_summary.csv"

OUT_COMPARE = './notebooks/grid_compare/csv/new_grid_compare_rygb_30runs_15idx.csv'
OUT_VALID = './notebooks/grid_compare/csv/new_grid_compare_rygb_30runs_15_idx_valid.csv'

# =============================================
# 데이터 로드
# =============================================
df_grid = pd.read_csv(DATA_PATH_GRID).copy()
grid_meta = pd.read_csv(GRID_META_PATH)
summary_df = pd.read_csv(SUMMARY_PATH)

print(f"그리드 데이터: {len(df_grid)}개")
print(f"그리드 메타: {len(grid_meta)}개")
print(f"샘플 CSV: {len(SAMPLE_CSVS)}개")

# =============================================
# Step 1. pdr_summary.csv 기준 q_edges 불러오기
# =============================================
def parse_interval(s):
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    return float(nums[0]), float(nums[1])

intervals = summary_df["pdr_q"].apply(parse_interval)
left_edges = [x[0] for x in intervals]
right_edges = [x[1] for x in intervals]

q_edges = np.array([left_edges[0]] + right_edges, dtype=float)
q_edges = np.unique(q_edges)
q_edges.sort()

tick_labels = summary_df["pdr_q"].tolist()
nq = len(tick_labels)

if nq < 1:
    raise ValueError("유효한 q-bin을 만들 수 없습니다.")

print(f"\npdr_summary.csv 기준 q_edges: {nq}개 구간")
print("q_edges:", q_edges)

# =============================================
# Step 2. q_idx 매핑 함수
# =============================================
def map_q_idx(pdr_series, q_edges, nq):
    pdr_series = pdr_series.astype(float)

    q_idx = pd.cut(
        pdr_series,
        bins=q_edges,
        labels=False,
        include_lowest=True,
        right=True
    ).astype(float)

    q_idx = q_idx.where(
        ~q_idx.isna(),
        np.where(pdr_series < q_edges[0], 0.0, float(nq - 1))
    )

    return q_idx.clip(lower=0.0, upper=float(nq - 1))

df_grid['q_idx'] = map_q_idx(df_grid['pdr_mean'], q_edges, nq)

# =============================================
# Step 3. 그리드별 포인트 할당 함수
# =============================================
def assign_to_grid(grid_meta, df_points, pdr_col):
    results = []
    for _, g in grid_meta.iterrows():
        mask = (
            (df_points['lon'] >= g['bbox_minlon']) &
            (df_points['lon'] <  g['bbox_maxlon']) &
            (df_points['lat'] >= g['bbox_minlat']) &
            (df_points['lat'] <  g['bbox_maxlat'])
        )
        pts = df_points[mask]
        results.append({
            'grid_id': g['grid_id'],
            'center_lat': g['lat'],
            'center_lon': g['lon'],
            'count': len(pts),
            'mean_pdr': pts[pdr_col].mean() if len(pts) > 0 else np.nan,
            'mean_q_idx': pts['q_idx'].mean() if len(pts) > 0 else np.nan,
        })
    return pd.DataFrame(results)

# =============================================
# Step 4. simulation 결과 그리드 할당
# =============================================
grid_sim = assign_to_grid(grid_meta, df_grid, pdr_col='pdr_mean').rename(columns={
    'count': 'sim_count',
    'mean_pdr': 'sim_mean_pdr',
    'mean_q_idx': 'sim_mean_q_idx',
})

print(f"\n[Simulation] 전체 그리드: {len(grid_sim)}개")
print(f"[Simulation] 전체 생성 포인트: {len(df_grid)}개")
print(f"[Simulation] 포인트 있는 셀: {(grid_sim['sim_count'] > 0).sum()}개")

# =============================================
# Step 5. diffusion 결과 로드 및 q_idx 매핑
# =============================================
loaded_samples = []

for csv_path in SAMPLE_CSVS:
    if not os.path.exists(csv_path):
        print(f"[WARN] not found: {csv_path}")
        continue

    dfg = pd.read_csv(csv_path)
    if not {'lat', 'lon', 'pdr'}.issubset(dfg.columns):
        print(f"[WARN] lat/lon/pdr 컬럼 없음: {csv_path}")
        continue

    dfg = dfg.copy()
    dfg['q_idx'] = map_q_idx(dfg['pdr'].astype(float), q_edges, nq)

    loaded_samples.append(dfg)
    print(f"  로드: {os.path.basename(csv_path)} ({len(dfg)}개)")

if not loaded_samples:
    raise ValueError("로드된 diffusion sample CSV가 없습니다.")

df_gen_all = pd.concat(loaded_samples, ignore_index=True)
print(f"\n[Diffusion] 전체 생성 포인트: {len(df_gen_all)}개")

# =============================================
# Step 6. diffusion 결과 그리드 할당
# =============================================
grid_gen = assign_to_grid(grid_meta, df_gen_all, pdr_col='pdr').rename(columns={
    'count': 'gen_count',
    'mean_pdr': 'gen_mean_pdr',
    'mean_q_idx': 'gen_mean_q_idx',
})

# =============================================
# Step 7. 두 결과 합치기 + 차이 계산
# =============================================
grid_compare = pd.merge(
    grid_sim,
    grid_gen,
    on=['grid_id', 'center_lat', 'center_lon'],
    how='outer'
)

for col in ['sim_count', 'gen_count']:
    grid_compare[col] = grid_compare[col].fillna(0)

grid_compare['pdr_diff'] = grid_compare['gen_mean_pdr'] - grid_compare['sim_mean_pdr']
grid_compare['q_idx_diff'] = grid_compare['gen_mean_q_idx'] - grid_compare['sim_mean_q_idx']

# =============================================
# Step 8. 수치 비교 요약
# =============================================
valid = grid_compare[
    (grid_compare['sim_count'] > 0) &
    (grid_compare['gen_count'] > 0)
].copy()

print(f"\n{'='*50}")
print("[비교 결과 요약]")
print(f"{'='*50}")
print(f"  전체 그리드 셀:     {len(grid_compare)}개")
print(f"  sim 포인트 있는 셀: {(grid_compare['sim_count'] > 0).sum()}개")
print(f"  gen 포인트 있는 셀: {(grid_compare['gen_count'] > 0).sum()}개")
print(f"  비교 가능한 셀:     {len(valid)}개")
print(f"  sim만 있는 셀:      {((grid_compare['sim_count'] > 0) & (grid_compare['gen_count'] == 0)).sum()}개")
print(f"  gen만 있는 셀:      {((grid_compare['sim_count'] == 0) & (grid_compare['gen_count'] > 0)).sum()}개")

if len(valid) == 0:
    print("\n[WARN] 비교 가능한 셀이 없습니다.")
else:
    rmse = np.sqrt((valid['pdr_diff'] ** 2).mean())
    pdr_range = valid['sim_mean_pdr'].max() - valid['sim_mean_pdr'].min()
    nrmse = rmse / pdr_range if pdr_range > 0 else np.nan

    print(f"\n[PDR 차이 통계 (gen - sim)]")
    print(f"  MAE:      {valid['pdr_diff'].abs().mean():.6f}")
    print(f"  RMSE:     {rmse:.6f}")
    print(f"  NRMSE:    {nrmse:.6f}")
    print(f"  Bias:     {valid['pdr_diff'].mean():.6f}")
    print(f"  Max diff: {valid['pdr_diff'].abs().max():.6f}")

# =============================================
# Step 9. 저장
# =============================================
os.makedirs('./notebooks/grid_compare/csv', exist_ok=True)
grid_compare.to_csv(OUT_COMPARE, index=False)
valid.to_csv(OUT_VALID, index=False)
print(f"\n저장 완료: {OUT_COMPARE} ({len(grid_compare)}행), {OUT_VALID} ({len(valid)}행)")

# =============================================
# Step 10. 어느 셀에도 속하지 않는 포인트 확인
# =============================================
def find_unassigned(grid_meta, df_points):
    assigned = pd.Series(False, index=df_points.index)
    for _, g in grid_meta.iterrows():
        mask = (
            (df_points['lon'] >= g['bbox_minlon']) &
            (df_points['lon'] <  g['bbox_maxlon']) &
            (df_points['lat'] >= g['bbox_minlat']) &
            (df_points['lat'] <  g['bbox_maxlat'])
        )
        assigned = assigned | mask
    return df_points[~assigned]

print("\n=== 셀별 포인트 중복 현황 ===")
gen_count_dist = grid_compare[grid_compare['gen_count'] > 0]['gen_count'].value_counts().sort_index()
for cnt, n_cells in gen_count_dist.items():
    print(f"  포인트 {int(cnt)}개짜리 셀: {n_cells}개")

print(f"\n최대 중복: {int(grid_compare['gen_count'].max())}개")
print(f"평균 중복 (포인트 있는 셀): {grid_compare[grid_compare['gen_count'] > 0]['gen_count'].mean():.2f}개")

print("\nsim_count 분포:")
sim_count_dist = grid_compare[grid_compare['sim_count'] > 0]['sim_count'].value_counts().sort_index()
for cnt, n_cells in sim_count_dist.items():
    print(f"  포인트 {int(cnt)}개짜리 셀: {n_cells}개")

print("\n=== 미할당 포인트 상세 ===")
unassigned = find_unassigned(grid_meta, df_gen_all)
print(f"총 미할당: {len(unassigned)}개")

if len(unassigned) > 0:
    print(f"\n위경도 범위:")
    print(f"  lat: {unassigned['lat'].min():.6f} ~ {unassigned['lat'].max():.6f}")
    print(f"  lon: {unassigned['lon'].min():.6f} ~ {unassigned['lon'].max():.6f}")
    print(f"\n그리드 메타 범위:")
    print(f"  lat: {grid_meta['bbox_minlat'].min():.6f} ~ {grid_meta['bbox_maxlat'].max():.6f}")
    print(f"  lon: {grid_meta['bbox_minlon'].min():.6f} ~ {grid_meta['bbox_maxlon'].max():.6f}")
    print(f"\n미할당 포인트 상세:")
    print(unassigned[['lat', 'lon', 'pdr', 'q_idx']].to_string())

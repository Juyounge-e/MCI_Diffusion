import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')

# =============================================
# 설정
# =============================================
GRID_COMPARE_PATH = r'C:\Users\user00\Desktop\MCI_Diffusion\notebooks\grid_compare\grid_compare.csv'
GRID_META_PATH    = r'C:\Users\user00\Desktop\MCI_Diffusion\MCI_ADV2\scenarios\daejeon_daejeon_grid\grid_metadata.csv'
DATA_PATH         = Path(r'C:\Users\user00\Desktop\MCI_Diffusion\src\data\dataset.csv')
SHP_PATH          = r'C:\Users\user00\Desktop\MCI_Diffusion\MCI_ADV2\scenarios\ctprvn.shp'

# =============================================
# 데이터 로드
# =============================================
df           = pd.read_csv(DATA_PATH)
grid_compare = pd.read_csv(GRID_COMPARE_PATH)
grid_meta    = pd.read_csv(GRID_META_PATH)

print(f"학습 데이터: {len(df)}개")
print(f"grid_compare: {len(grid_compare)}개")
print(f"grid_meta: {len(grid_meta)}개")

# =============================================
# Step 1. 전체 df 기준 q_edges 계산 (colorbar 기준)
# =============================================
_, q_edges = pd.qcut(
    df['pdr_mean'],
    q=10,
    labels=False,
    retbins=True,
    duplicates='drop'
)
nq = len(q_edges) - 1
tick_labels = [f"Q{i+1} [{q_edges[i]:.4f}, {q_edges[i+1]:.4f}]" for i in range(nq)]

def pdr_to_q_idx(pdr_val, q_edges, nq):
    """단일 pdr 값을 q_idx로 변환"""
    if np.isnan(pdr_val):
        return np.nan
    for i in range(nq):
        if q_edges[i] <= pdr_val <= q_edges[i+1]:
            return float(i)
    if pdr_val < q_edges[0]:
        return 0.0
    return float(nq - 1)

# grid_compare에 q_idx 추가
grid_compare['sim_q_idx'] = grid_compare['sim_mean_pdr'].apply(
    lambda x: pdr_to_q_idx(x, q_edges, nq)
)
grid_compare['gen_q_idx'] = grid_compare['gen_mean_pdr'].apply(
    lambda x: pdr_to_q_idx(x, q_edges, nq)
)

# grid_meta와 merge (bbox 정보 필요)
grid_plot = pd.merge(grid_compare, grid_meta, on='grid_id', how='left')

# =============================================
# 공통 설정
# =============================================
import geopandas as gpd

def overlay_daejeon_boundary(ax):
    try:
        gdf = gpd.read_file(SHP_PATH, encoding='cp949')
        gdf_wgs84 = gdf.set_crs(epsg=5179, allow_override=True).to_crs(epsg=4326)
        gdf_daejeon = gdf_wgs84[gdf_wgs84['CTP_ENG_NM'].str.lower() == 'daejeon']
        gdf_daejeon.boundary.plot(ax=ax, color='black', linewidth=1.2, linestyle='--')
    except Exception as e:
        print(f"[WARN] boundary skipped: {e}")

cmap = plt.colormaps['coolwarm']
norm = plt.Normalize(vmin=0, vmax=max(1, nq - 1))

def add_colorbar(fig, ax, label):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(label)
    cbar.set_ticks(range(nq))
    cbar.set_ticklabels(tick_labels)
    return cbar

def draw_grid_heatmap(ax, grid_plot, q_idx_col, title):
    """그리드 셀을 q_idx 색으로 채우기"""
    for _, row in grid_plot.iterrows():
        q_idx = row[q_idx_col]
        if np.isnan(q_idx):
            facecolor = 'lightgray'
            alpha = 0.2
        else:
            facecolor = cmap(norm(q_idx))
            alpha = 0.85

        rect = patches.Rectangle(
            (row['bbox_minlon'], row['bbox_minlat']),
            row['bbox_maxlon'] - row['bbox_minlon'],
            row['bbox_maxlat'] - row['bbox_minlat'],
            linewidth=0.2,
            edgecolor='gray',
            facecolor=facecolor,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(rect)

    overlay_daejeon_boundary(ax)
    add_colorbar(plt.gcf(), ax, f'pdr_q bins (from entire df, n={len(df)})')

    ax.set_xlim(grid_meta['bbox_minlon'].min() - 0.01,
                grid_meta['bbox_maxlon'].max() + 0.01)
    ax.set_ylim(grid_meta['bbox_minlat'].min() - 0.01,
                grid_meta['bbox_maxlat'].max() + 0.01)
    ax.set_title(title)
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')

# =============================================
# Figure 1: Simulation 결과
# =============================================
sim_covered = (grid_plot['sim_count'] > 0).sum()

fig1, ax1 = plt.subplots(figsize=(11, 9))
draw_grid_heatmap(
    ax1, grid_plot, 'sim_q_idx',
    f"Grid Simulation Result (covered={sim_covered}/{len(grid_plot)})"
)
plt.tight_layout()
plt.savefig('./notebooks/grid_compare/grid_sim_heatmap.png', dpi=150)
plt.show()
print("저장 완료: grid_sim_heatmap.png")

# =============================================
# Figure 2: Diffusion 결과
# =============================================
gen_covered = (grid_plot['gen_count'] > 0).sum()

fig2, ax2 = plt.subplots(figsize=(11, 9))
draw_grid_heatmap(
    ax2, grid_plot, 'gen_q_idx',
    f"Diffusion Generated Result (covered={gen_covered}/{len(grid_plot)})"
)
plt.tight_layout()
plt.savefig('./notebooks/grid_compare/grid_gen_heatmap.png', dpi=150)
plt.show()
print("저장 완료: grid_gen_heatmap.png")
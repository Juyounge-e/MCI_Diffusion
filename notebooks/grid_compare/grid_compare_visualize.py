# MCI 가상환경에서 실행
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd

# =============================================
# 설정
# =============================================
GRID_COMPARE_PATH = r'C:\Users\user00\Desktop\MCI_Diffusion\notebooks\grid_compare\grid_compare_rygb_.csv'
GRID_META_PATH = r'C:\Users\user00\Desktop\MCI_Diffusion\MCI_ADV2\scenarios\daejeon_daejeon_grid\grid_metadata.csv'
SHP_PATH = r'C:\Users\user00\Desktop\MCI_Diffusion\MCI_ADV2\scenarios\ctprvn.shp'
SUMMARY_PATH = r'C:\Users\user00\Desktop\MCI_Diffusion\notebooks\outputs\analysis\pdr_summary.csv'

# =============================================
# 데이터 로드
# =============================================
grid_compare = pd.read_csv(GRID_COMPARE_PATH)
grid_meta = pd.read_csv(GRID_META_PATH)
summary_df = pd.read_csv(SUMMARY_PATH)

print(f"grid_compare: {len(grid_compare)}개")
print(f"grid_meta: {len(grid_meta)}개")
print(f"summary_df: {len(summary_df)}개")

# =============================================
# Step 1. pdr_summary.csv 기준 q_edges 계산
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

print(f"pdr_summary.csv 기준 q_edges: {nq}개 구간")
print("q_edges:", q_edges)

def pdr_to_q_idx(pdr_val, q_edges, nq):
    if pd.isna(pdr_val):
        return np.nan

    q_idx = pd.cut(
        pd.Series([float(pdr_val)]),
        bins=q_edges,
        labels=False,
        include_lowest=True,
        right=True
    ).iloc[0]

    if pd.isna(q_idx):
        if pdr_val < q_edges[0]:
            return 0.0
        return float(nq - 1)

    return float(q_idx)

# grid_compare에 q_idx 추가
grid_compare['sim_q_idx'] = grid_compare['sim_mean_pdr'].apply(
    lambda x: pdr_to_q_idx(x, q_edges, nq)
)
grid_compare['gen_q_idx'] = grid_compare['gen_mean_pdr'].apply(
    lambda x: pdr_to_q_idx(x, q_edges, nq)
)

# bbox 정보 merge
grid_plot = pd.merge(grid_compare, grid_meta, on='grid_id', how='left')

# =============================================
# 공통 설정
# =============================================
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
    for _, row in grid_plot.iterrows():
        q_idx = row[q_idx_col]
        if pd.isna(q_idx):
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
    add_colorbar(plt.gcf(), ax, 'pdr_q bins (from pdr_summary.csv)')

    ax.set_xlim(grid_plot['bbox_minlon'].min() - 0.01,
                grid_plot['bbox_maxlon'].max() + 0.01)
    ax.set_ylim(grid_plot['bbox_minlat'].min() - 0.01,
                grid_plot['bbox_maxlat'].max() + 0.01)
    ax.set_title(title)
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')

# =============================================
# Figure 1: Simulation 결과
# =============================================
sim_covered = (grid_plot['sim_count'] > 0).sum()

fig1, ax1 = plt.subplots(figsize=(11, 9))
draw_grid_heatmap(
    ax1,
    grid_plot,
    'sim_q_idx',
    f"Grid Simulation Result (covered={sim_covered}/{len(grid_plot)})"
)
plt.tight_layout()
plt.savefig('./notebooks/grid_compare/rygb_grid_sim_heatmap.png', dpi=150)
plt.close(fig1)
print("저장 완료: ./notebooks/grid_compare/rygb_grid_sim_heatmap.png")

# =============================================
# Figure 2: Diffusion 결과
# =============================================
gen_covered = (grid_plot['gen_count'] > 0).sum()

fig2, ax2 = plt.subplots(figsize=(11, 9))
draw_grid_heatmap(
    ax2,
    grid_plot,
    'gen_q_idx',
    f"Diffusion Generated Result (covered={gen_covered}/{len(grid_plot)})"
)
plt.tight_layout()
plt.savefig('./notebooks/grid_compare/rygb_grid_gen_heatmap.png', dpi=150)
plt.close(fig2)
print("저장 완료: ./notebooks/grid_compare/rygb_grid_gen_heatmap.png")

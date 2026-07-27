# -*- coding: utf-8 -*-
"""
대전 조건(N × pdr) 4×4 생성좌표 시각화 — "조건별로 잘 반응하는가" 확인용.

행 = N (N_LIST), 열 = pdr (PDR_LIST) → 16개 셀.
각 셀: 대전 경계 + 그 조건의 생성좌표(고정 pdr이라 단색). PNG 저장.

전제: sample_mlp.py 로 대전 모델(daejeon_lam0_40k) 을 각 조건으로 샘플링해
      outputs/mlp_diffusion/cond_daejeon_N{n}_pdr{p}/samples.csv 로 저장해둠.

실행: conda env `mci_viz` 에서  python notebooks/viz_daejeon_4x4.py
"""
import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = r"C:\Users\user00\Desktop\MCI_Diffusion"
MLP = os.path.join(ROOT, "outputs", "mlp_diffusion")
SHP = os.path.join(ROOT, "MCI_ADV2", "scenarios", "ctprvn.shp")
SUMMARY = os.path.join(ROOT, "notebooks", "outputs", "analysis", "daejeon_mean_summary.csv")
OUT = os.path.join(ROOT, "notebooks", "outputs", "analysis", "daejeon_4x4.png")

# ── 설정 ─────────────────────────────────────────────────────────────
N_LIST = [15, 25, 35, 45]                 # 행
PDR_LIST = [0.01, 0.025, 0.04, 0.055]    # 열
PT_SIZE = 5
ASPECT = 1.3
EXPORT_SCALE = 2
PAD = 0.05


def csv_of(n, p):
    return os.path.join(MLP, f"cond_daejeon_N{n}_pdr{p}", "samples.csv")


# ── 대전 q-bin 경계 (daejeon_mean_summary.csv) ──────────────────────
sdf = pd.read_csv(SUMMARY)


def parse_interval(s):
    m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    return float(m[0]), float(m[1])

_iv = sdf["pdr_q"].apply(parse_interval)
BINS = np.unique(np.array([_iv.iloc[0][0]] + [x[1] for x in _iv], dtype=float))
BINS.sort()
BINS[0], BINS[-1] = -np.inf, np.inf
LABELS = sdf["pdr_q"].tolist()
NQ = len(LABELS)


# ── 대전 경계 ────────────────────────────────────────────────────────
g = gpd.read_file(SHP, encoding="cp949").set_crs(epsg=5179, allow_override=True).to_crs(epsg=4326)
dj = g[g["CTP_KOR_NM"].str.contains("대전")].copy()
dj["geometry"] = dj.geometry.simplify(0.002)

bx, by = [], []
for geom in dj.geometry:
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        x, y = poly.exterior.xy
        bx += list(x) + [None]
        by += list(y) + [None]

minx, miny, maxx, maxy = dj.total_bounds
px, py = (maxx - minx) * PAD, (maxy - miny) * PAD
VIEW = dict(x=[minx - px, maxx + px], y=[miny - py, maxy + py])

# ── 4×4 그리기 ───────────────────────────────────────────────────────
nrow, ncol = len(N_LIST), len(PDR_LIST)
titles = [f"N={n}, pdr={p}" for n in N_LIST for p in PDR_LIST]
fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=titles,
                    horizontal_spacing=0.04, vertical_spacing=0.06)

shown_cbar = False   # 컬러바는 첫 데이터 셀에만 한 번
for i, n in enumerate(N_LIST):
    for j, p in enumerate(PDR_LIST):
        row, col = i + 1, j + 1
        k = i * ncol + j + 1
        xa, ya = ("x" if k == 1 else f"x{k}"), ("y" if k == 1 else f"y{k}")

        # 대전 경계
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", xaxis=xa, yaxis=ya,
                                 line=dict(color="black", width=1),
                                 showlegend=False, hoverinfo="skip"), row=row, col=col)
        # 생성점 — 대전 q-bin 색 (고정 pdr이라 셀 전체 동일 색)
        csv = csv_of(n, p)
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            q_idx = pd.cut(df["pdr"], bins=BINS, labels=False, include_lowest=True).astype(float)
            fig.add_trace(go.Scatter(
                x=df["lon"], y=df["lat"], mode="markers", xaxis=xa, yaxis=ya,
                marker=dict(size=PT_SIZE, color=q_idx, colorscale="RdBu_r",
                            cmin=0, cmax=NQ - 1, opacity=0.85,
                            line=dict(width=0.5, color="rgba(70,70,70,0.6)"),  # 흰색(중앙 bin) 점도 보이게
                            showscale=(not shown_cbar),
                            colorbar=dict(title="PDR q-bin (대전)", tickmode="array",
                                          tickvals=list(range(NQ)), ticktext=LABELS,
                                          len=0.85, x=1.01)),
                showlegend=False, hoverinfo="skip"), row=row, col=col)
            shown_cbar = True
            fig.layout.annotations[k - 1].text = f"N={n}, pdr={p} (n={len(df)})"
        else:
            fig.layout.annotations[k - 1].text = f"N={n}, pdr={p} (없음)"

        fig.update_xaxes(range=VIEW["x"], row=row, col=col, showticklabels=False)
        fig.update_yaxes(range=VIEW["y"], scaleanchor=xa, scaleratio=ASPECT,
                         row=row, col=col, showticklabels=False)

fig.update_layout(title="대전 조건별 생성좌표",
                  width=1300, height=1500, template="plotly_white",
                  margin=dict(l=10, r=10, t=70, b=10))
fig.write_image(OUT, scale=EXPORT_SCALE)
print(f"[saved] {OUT}")

# -*- coding: utf-8 -*-
"""
전국 생성좌표(q-bin) + 병원/소방 시설(컬러 이모지) → 고해상도 PNG 배치 생성.

산출물 (OUT_DIR 하위):
  1) national.png            — 전국 1장
  2) regions/<지역>.png      — 시도별 개별 16장
  3) regions_4x4.png         — 16개 시도를 4×4 subplot 한 장

데이터 소스는 코드 상단 DATA_SOURCE 토글로 하나씩 (train / exp).
렌더러: plotly + kaleido (컬러 이모지 정상). EXPORT_SCALE 로 해상도 조절(확대해도 선명).
실행 환경: conda env `mci_viz`
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════
#  설정 (여기만 바꾸면 됨)
# ══════════════════════════════════════════════════════════════════════
DATA_SOURCE = "exp"          # train or exp
TRAIN_N_RANGE = (27, 33)     # None 이면 전체(10~50).
EXP_PATH = r"C:\Users\user00\Desktop\MCI_Diffusion\outputs\mlp_diffusion\500k_national_lam0_mint30_seed_200"

MAX_POINTS = 500000             # 좌표 다운샘플 상한
EMOJI_SIZE = 9                 # 전국/지역 개별 이모지 크기
EMOJI_SIZE_GRID = 6            # 4×4 그리드 이모지 크기
PT_SIZE = 3                    # 전국/지역 점 크기
PT_SIZE_GRID = 4               # 4×4 점 크기
EXPORT_SCALE = 3               # kaleido 해상도 배수 (↑ = 더 선명, 더 무거움)
SHOW_FACILITIES = True

SHOW_FIRE = False              # 소방서 표시 여부 (옵션)
# 병원 종별코드 → (이름, 마커색, 마커크기 배율). 마커 = 이모지 크기 × 배율
HOSP_TYPES = [
    (1, "Tertiary Hospital", "red", 2.0),
    (11, "Secondary General Hospital", "orange", 1.5),
    (21, "Hospital", "green", 1.5),
]
HOSP_SHOW_CODES = [1,11]     # 표시할 병원 종별코드 (일반=21 제외). 전부 보려면 [1, 11, 21]
SCATTER_ORDER = "shuffle"     # 점 그리는 순서(z-order):
                              #   "q1_first"  → q1..q10 순서로 그림 → q10(고PDR)이 맨 위
                              #   "q10_first" → q10..q1 순서로 그림 → q1(저PDR)이 맨 위
                              #   "shuffle"   → 무작위 → 특정 분위 독점 없음(권장)

# ══════════════════════════════════════════════════════════════════════
ROOT = r"C:\Users\user00\Desktop\MCI_Diffusion"
SUMMARY = os.path.join(ROOT, "notebooks", "outputs", "analysis", "national_all_summary.csv")
SHP = os.path.join(ROOT, "MCI_ADV2", "scenarios", "ctprvn.shp")
FIRE_CSV = os.path.join(ROOT, "MCI_ADV2", "scenarios", "안전센터와 소방서.csv")
HOSP_XLSX = os.path.join(ROOT, "MCI_ADV2", "scenarios", "엑셀 결합 데이터.xlsx")
TRAIN_CSV = os.path.join(ROOT, "src", "data", "national_all.csv")

SIMPLIFY_TOL = 0.005
JEJU_BBOX = (126.10, 33.10, 126.98, 34.02)
NATIONAL_VIEW = dict(x=[124.9, 129.8], y=[33.95, 38.75])
ASPECT = 1.3                   # lat/lon 축 비율 (지리 왜곡 보정)

if DATA_SOURCE == "train":
    TAG = f"train_N{TRAIN_N_RANGE[0]}-{TRAIN_N_RANGE[1]}" if TRAIN_N_RANGE else "train"
else:
    TAG = os.path.basename(EXP_PATH)
OUT_DIR = os.path.join(ROOT, "notebooks", "outputs", "analysis", "facilities_maps", TAG)
os.makedirs(os.path.join(OUT_DIR, "regions"), exist_ok=True)


# ── 유틸 ─────────────────────────────────────────────────────────────
def find_latlon(df):
    lat = lon = None
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() < 0.8:
            continue
        med = s.median()
        if 33 < med < 39:
            lat = c
        elif 124 < med < 132:
            lon = c
    return lat, lon


def read_any_csv(path):
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def pick_name_col(df, lat_c, lon_c):
    text_cols = []
    for c in df.columns:
        if c in (lat_c, lon_c):
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() > 0.5:
            continue
        if df[c].notna().mean() > 0.5:
            text_cols.append(c)
    for key in ("기관", "명", "name"):
        for c in text_cols:
            if key in str(c).lower() or key in str(c):
                return c
    return text_cols[0] if text_cols else None


def drop_jeju(df, lat_c, lon_c):
    lat = pd.to_numeric(df[lat_c], errors="coerce")
    lon = pd.to_numeric(df[lon_c], errors="coerce")
    in_jeju = (lon.between(JEJU_BBOX[0], JEJU_BBOX[2])
               & lat.between(JEJU_BBOX[1], JEJU_BBOX[3]))
    return df[~in_jeju]


def region_of(df, lat_c, lon_c, regions_gdf):
    lat = pd.to_numeric(df[lat_c], errors="coerce")
    lon = pd.to_numeric(df[lon_c], errors="coerce")
    pts = gpd.GeoDataFrame({"_i": np.arange(len(df))},
                           geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    j = gpd.sjoin(pts, regions_gdf[["CTP_KOR_NM", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].sort_index()
    return j["CTP_KOR_NM"].to_numpy()


def boundary_xy(gdf):
    bx, by = [], []
    for geom in gdf.geometry:
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            x, y = poly.exterior.xy
            bx += list(x) + [None]
            by += list(y) + [None]
    return bx, by


def region_view(bnd_gdf, pad=0.05):
    minx, miny, maxx, maxy = bnd_gdf.total_bounds
    px, py = (maxx - minx) * pad, (maxy - miny) * pad
    return dict(x=[minx - px, maxx + px], y=[miny - py, maxy + py])


# ── 데이터 로드 ──────────────────────────────────────────────────────
sdf = pd.read_csv(SUMMARY)


def parse_interval(s):
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    return float(nums[0]), float(nums[1])

iv = sdf["pdr_q"].apply(parse_interval)
bins = np.unique(np.array([iv.iloc[0][0]] + [x[1] for x in iv], dtype=float))
bins.sort()
bins[0], bins[-1] = -np.inf, np.inf
labels = [
    f"Q{i + 1} [{low:.3f}, {high:.3f}]"
    for i, (low, high) in enumerate(iv)
]
nq = len(labels)

# 경계 (제주 제외, cp949 한글)
regions = gpd.read_file(SHP, encoding="cp949").set_crs(epsg=5179, allow_override=True).to_crs(epsg=4326)
regions = regions[regions["CTP_ENG_NM"].str.lower() != "jeju-do"].copy()
regions["geometry"] = regions.geometry.simplify(SIMPLIFY_TOL)
regions = regions.reset_index(drop=True)
region_names = sorted(regions["CTP_KOR_NM"].tolist())
region_display = dict(
    zip(regions["CTP_KOR_NM"], regions["CTP_ENG_NM"])
)

# 생성/학습 좌표
if DATA_SOURCE == "train":
    pred = pd.read_csv(TRAIN_CSV)
    if TRAIN_N_RANGE is not None:   # 생성(N=30)과 같은 N대만 (공정 비교)
        pred = pred[(pred["N"] >= TRAIN_N_RANGE[0]) & (pred["N"] <= TRAIN_N_RANGE[1])]
    pred["pdr"] = pred["pdr_mean"]
else:
    ab = os.path.join(EXP_PATH, "all_bins.csv")
    pred = pd.read_csv(ab) if os.path.exists(ab) else pd.concat(
        [pd.read_csv(p) for p in sorted(Path(EXP_PATH).glob("q*.csv"))], ignore_index=True)
pred["q_idx"] = pd.cut(pred["pdr"], bins=bins, labels=False, include_lowest=True).astype(float)
pred = drop_jeju(pred, "lat", "lon")
if len(pred) > MAX_POINTS:
    pred = pred.sample(MAX_POINTS, random_state=0)
pred["_region"] = region_of(pred, "lat", "lon", regions)
# 그리기 순서(z-order) 제어: 마지막 행이 맨 위에 그려짐
if SCATTER_ORDER == "q1_first":
    pred = pred.sort_values("q_idx", kind="stable")            # q10 위
elif SCATTER_ORDER == "q10_first":
    pred = pred.sort_values("q_idx", ascending=False, kind="stable")  # q1 위
elif SCATTER_ORDER == "shuffle":
    pred = pred.sample(frac=1, random_state=0)                 # 무작위
print(f"[pred] source={DATA_SOURCE} tag={TAG} n={len(pred):,} order={SCATTER_ORDER}")

# 시설
fire = read_any_csv(FIRE_CSV)
hosp = pd.read_excel(HOSP_XLSX)
f_lat, f_lon = find_latlon(fire)
h_lat, h_lon = find_latlon(hosp)
f_name = pick_name_col(fire, f_lat, f_lon)
h_name = pick_name_col(hosp, h_lat, h_lon)
fire = drop_jeju(fire, f_lat, f_lon)
hosp = drop_jeju(hosp, h_lat, h_lon)
fire["_region"] = region_of(fire, f_lat, f_lon, regions)
hosp["_region"] = region_of(hosp, h_lat, h_lon, regions)
print(f"[fire] {len(fire):,}  [hosp] {len(hosp):,}")


# ── trace 빌더 ───────────────────────────────────────────────────────
def make_traces(pred_sub, fire_sub, hosp_sub, bnd_gdf, emoji_size, pt_size,
                showscale=True, showlegend=True, axis=("x", "y")):
    """한 지도(전국/지역/subplot cell)에 올릴 trace 리스트."""
    xa, ya = axis
    traces = []
    bx, by = boundary_xy(bnd_gdf)
    traces.append(go.Scatter(x=bx, y=by, mode="lines", xaxis=xa, yaxis=ya,
                             line=dict(color="black", width=1, dash="dash"),
                             showlegend=False, hoverinfo="skip"))
    traces.append(go.Scatter(
        x=pred_sub["lon"], y=pred_sub["lat"], mode="markers", xaxis=xa, yaxis=ya,
        marker=dict(size=pt_size, color=pred_sub["q_idx"], colorscale="RdBu_r",
                    cmin=0, cmax=nq - 1, opacity=0.55, showscale=showscale,
                    colorbar=dict(
                        title=dict(
                            text="PDR Quantile Bin",
                            font=dict(
                                family="Times New Roman",
                                size=15,
                                color="black",
                            ),
                        ),
                        tickmode="array",
                        tickvals=list(range(nq)),
                        ticktext=labels,
                        tickfont=dict(
                            family="Times New Roman",
                            size=13,
                            color="black",
                        ),
                        len=0.85,
                    )),
        name="Generated locations", showlegend=False, hoverinfo="skip"))
    if SHOW_FACILITIES:
        def ftrace(df, lat_c, lon_c, glyph, label):   # 이모지만 (소방용)
            sub = df.copy()
            sub[lat_c] = pd.to_numeric(sub[lat_c], errors="coerce")
            sub[lon_c] = pd.to_numeric(sub[lon_c], errors="coerce")
            sub = sub.dropna(subset=[lat_c, lon_c])
            return go.Scatter(x=sub[lon_c], y=sub[lat_c], mode="text", xaxis=xa, yaxis=ya,
                              text=[glyph] * len(sub), textfont=dict(size=emoji_size),
                              name=f"{glyph} {label} (n={len(sub)})",
                              showlegend=showlegend, hoverinfo="skip")

        def htrace(sub, label, color, size_mult):     # 컬러 마커(크기=이모지×배율) + 🏥
            sub = sub.copy()
            sub[h_lat] = pd.to_numeric(sub[h_lat], errors="coerce")
            sub[h_lon] = pd.to_numeric(sub[h_lon], errors="coerce")
            sub = sub.dropna(subset=[h_lat, h_lon])
            return go.Scatter(x=sub[h_lon], y=sub[h_lat], mode="markers+text", xaxis=xa, yaxis=ya,
                              marker=dict(size=emoji_size * size_mult, color=color, opacity=0.55,
                                          line=dict(width=0)),
                              text=["🏥"] * len(sub), textfont=dict(size=emoji_size),
                              name=f"🏥 {label} (n={len(sub)})",
                              showlegend=showlegend, hoverinfo="skip")

        # 병원: 종별 컬러 마커 (HOSP_SHOW_CODES 만)
        if "종별코드" in hosp_sub.columns:
            for code, tname, color, smult in HOSP_TYPES:
                if code not in HOSP_SHOW_CODES:
                    continue
                sub = hosp_sub[hosp_sub["종별코드"] == code]
                if len(sub):
                    traces.append(htrace(sub, tname, color, smult))
        else:
            traces.append(ftrace(hosp_sub, h_lat, h_lon, "🏥", "Hospital"))
        # 소방: 옵션
        if SHOW_FIRE:
            traces.append(ftrace(fire_sub, f_lat, f_lon, "🚒", "Fire station"))
    return traces


def save_single(pred_sub, fire_sub, hosp_sub, bnd_gdf, view, title, path, w=900, h=1000):
    fig = go.Figure()
    for t in make_traces(pred_sub, fire_sub, hosp_sub, bnd_gdf,
                         EMOJI_SIZE, PT_SIZE, showscale=True, showlegend=True):
        fig.add_trace(t)
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.96,
            yanchor="top",
            font=dict(
                family="Times New Roman",
                size=22,
                color="black",
            ),
            pad=dict(t=5, b=5),
        ),
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        xaxis=dict(range=view["x"], constrain="domain"),
        yaxis=dict(
            range=view["y"],
            scaleanchor="x",
            scaleratio=ASPECT,
            constrain="domain",
        ),
        width=w, height=h, template="plotly_white",
        font=dict(family="Times New Roman", size=14, color="black"),
        legend=dict(x=0.01, y=0.93, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=45, r=30, t=80, b=45))
    fig.write_image(path, scale=EXPORT_SCALE)
    print(f"[saved] {path}")


# ── 1) 전국 1장 ──────────────────────────────────────────────────────
save_single(pred, fire, hosp, regions, NATIONAL_VIEW,
            f"Nationwide Generated Incident Locations with Hospitals",
            # f"(N={len(pred):,})",
            os.path.join(OUT_DIR, "national.png"))

# ── 2) 지역별 개별 16장 ──────────────────────────────────────────────
for rn in region_names:
    bnd = regions[regions["CTP_KOR_NM"] == rn]
    p = pred[pred["_region"] == rn]
    fsub = fire[fire["_region"] == rn]
    hsub = hosp[hosp["_region"] == rn]
    safe = re.sub(r"[^\w가-힣]", "_", rn)
    save_single(p, fsub, hsub, bnd, region_view(bnd),
                f"[{TAG}] {region_display[rn]} (N={len(p):,})",
                os.path.join(OUT_DIR, "regions", f"{safe}.png"), w=850, h=850)

# ── 3) 지역 4×4 그리드 1장 ───────────────────────────────────────────
fig = make_subplots(
                    rows=4, cols=4,
                    subplot_titles=[region_display[name] for name in region_names],
                    horizontal_spacing=0.03, vertical_spacing=0.05)
for idx, rn in enumerate(region_names):
    row, col = idx // 4 + 1, idx % 4 + 1
    k = idx + 1
    xa, ya = ("x" if k == 1 else f"x{k}"), ("y" if k == 1 else f"y{k}")
    bnd = regions[regions["CTP_KOR_NM"] == rn]
    p = pred[pred["_region"] == rn]
    fsub = fire[fire["_region"] == rn]
    hsub = hosp[hosp["_region"] == rn]
    for t in make_traces(p, fsub, hsub, bnd, EMOJI_SIZE_GRID, PT_SIZE_GRID,
                         showscale=(idx == 0), showlegend=False, axis=(xa, ya)):
        fig.add_trace(t, row=row, col=col)
    v = region_view(bnd)
    fig.update_xaxes(range=v["x"], row=row, col=col)
    fig.update_yaxes(range=v["y"], scaleanchor=xa, scaleratio=ASPECT, row=row, col=col)

for annotation in fig.layout.annotations:
    annotation.font = dict(
        family="Times New Roman",
        size=15,
        color="black",
    )

fig.update_layout(title=f"Generated Locations and Hospitals by Region",
                  width=1600, height=1750, template="plotly_white",
                  font=dict(family="Times New Roman", size=14, color="black"),
                  margin=dict(l=10, r=10, t=60, b=10))
grid_path = os.path.join(OUT_DIR, "regions_4x4.png")
fig.write_image(grid_path, scale=EXPORT_SCALE)
print(f"[saved] {grid_path}")

print("\n[done] out dir:", OUT_DIR)

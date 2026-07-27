# -*- coding: utf-8 -*-
"""
MCI 전국 시각화 Streamlit 앱 — 생성좌표(q-bin) + 병원/소방 시설.

정적 HTML(notebooks/viz_facilities_national_plotly.py)의 인터랙티브 확장판.
사이드바 위젯으로 실험 폴더 / 지역 / 점 개수 / 이모지 크기 / 시설·경계 토글을
즉시 바꿔 볼 수 있다. (제주는 항상 제외, 경계 단순화 0.005 고정)

실행:  conda env `mci_viz` 에서
    streamlit run app/app.py
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import streamlit as st

# ── 경로 / 고정값 ────────────────────────────────────────────────────
ROOT = r"C:\Users\user00\Desktop\MCI_Diffusion"
MLP_DIR = os.path.join(ROOT, "outputs", "mlp_diffusion")
SUMMARY = os.path.join(ROOT, "notebooks", "outputs", "analysis", "national_all_summary.csv")
SHP = os.path.join(ROOT, "MCI_ADV2", "scenarios", "ctprvn.shp")
FIRE_CSV = os.path.join(ROOT, "MCI_ADV2", "scenarios", "안전센터와 소방서.csv")
HOSP_XLSX = os.path.join(ROOT, "MCI_ADV2", "scenarios", "엑셀 결합 데이터.xlsx")

TRAIN_CSV = os.path.join(ROOT, "src", "data", "national_all.csv")  # 학습 데이터
SIMPLIFY_TOL = 0.005      # 경계 단순화(고정)
ALL_REGIONS = "전체 (전국)"
JEJU_BBOX = (126.10, 33.10, 126.98, 34.02)   # (minlon, minlat, maxlon, maxlat) — 항상 제외
# 병원 종별코드 → (이름, 마커색). 🏥 이모지 아래에 컬러 원을 깔아 종별 구분
HOSP_TYPES = [(1, "상급종합", "red"), (11, "종합병원", "orange"), (21, "병원", "green")]

st.set_page_config(page_title="MCI 전국 시각화", layout="wide")


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
    """JEJU_BBOX 안의 좌표 제거 (제주는 주변이 바다라 bbox 로 안전하게 분리)."""
    lat = pd.to_numeric(df[lat_c], errors="coerce")
    lon = pd.to_numeric(df[lon_c], errors="coerce")
    in_jeju = (lon.between(JEJU_BBOX[0], JEJU_BBOX[2])
               & lat.between(JEJU_BBOX[1], JEJU_BBOX[3]))
    return df[~in_jeju]


def region_of(df, lat_c, lon_c, regions_gdf):
    """각 좌표가 속한 시도명(CTP_KOR_NM) 을 within 판정으로 부여. 바다/영역밖=NaN."""
    lat = pd.to_numeric(df[lat_c], errors="coerce")
    lon = pd.to_numeric(df[lon_c], errors="coerce")
    pts = gpd.GeoDataFrame(
        {"_i": np.arange(len(df))},
        geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    j = gpd.sjoin(pts, regions_gdf[["CTP_KOR_NM", "geometry"]],
                  how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].sort_index()
    return j["CTP_KOR_NM"].to_numpy()


# ── 캐시 로더 ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_bins():
    sdf = pd.read_csv(SUMMARY)

    def parse_interval(s):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
        return float(nums[0]), float(nums[1])

    iv = sdf["pdr_q"].apply(parse_interval)
    right = [x[1] for x in iv]
    bins = np.unique(np.array([iv.iloc[0][0]] + right, dtype=float))
    bins.sort()
    bins[0], bins[-1] = -np.inf, np.inf
    return bins, sdf["pdr_q"].tolist()


@st.cache_resource(show_spinner=False)
def load_regions():
    """시도 경계 GeoDataFrame (제주 제외, 단순화). CTP_KOR_NM 사용.
    ctprvn.shp 는 .cpg 가 없어 DBF 한글이 cp949 → encoding 명시 필요."""
    g = gpd.read_file(SHP, encoding="cp949").set_crs(epsg=5179, allow_override=True).to_crs(epsg=4326)
    g = g[g["CTP_ENG_NM"].str.lower() != "jeju-do"].copy()
    g["geometry"] = g.geometry.simplify(SIMPLIFY_TOL)
    return g.reset_index(drop=True)


def boundary_xy(gdf):
    bx, by = [], []
    for geom in gdf.geometry:
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            x, y = poly.exterior.xy
            bx += list(x) + [None]
            by += list(y) + [None]
    return bx, by


@st.cache_data(show_spinner=False)
def load_facilities():
    fire = read_any_csv(FIRE_CSV)
    hosp = pd.read_excel(HOSP_XLSX)
    f_lat, f_lon = find_latlon(fire)
    h_lat, h_lon = find_latlon(hosp)
    regions = load_regions()
    fire["_region"] = region_of(fire, f_lat, f_lon, regions)
    hosp["_region"] = region_of(hosp, h_lat, h_lon, regions)
    return (fire, f_lat, f_lon, pick_name_col(fire, f_lat, f_lon),
            hosp, h_lat, h_lon, pick_name_col(hosp, h_lat, h_lon))


@st.cache_data(show_spinner=True)
def load_pred(folder):
    """생성좌표 로드 + q_idx + 시도 배정(_region). 폴더별로 1회 캐시."""
    ab = os.path.join(folder, "all_bins.csv")
    if os.path.exists(ab):
        df = pd.read_csv(ab)
    else:
        parts = [pd.read_csv(p) for p in sorted(Path(folder).glob("q*.csv"))]
        if not parts:
            return pd.DataFrame(columns=["lat", "lon", "pdr"])
        df = pd.concat(parts, ignore_index=True)
    bins, _ = load_bins()
    df["q_idx"] = pd.cut(df["pdr"], bins=bins, labels=False, include_lowest=True).astype(float)
    df["_region"] = region_of(df, "lat", "lon", load_regions())
    return df


@st.cache_data(show_spinner=False)
def list_experiments():
    out = []
    for d in sorted(Path(MLP_DIR).iterdir()):
        if d.is_dir() and ((d / "all_bins.csv").exists() or list(d.glob("q*.csv"))):
            out.append(d.name)
    return out


# ── 사이드바 ─────────────────────────────────────────────────────────
st.sidebar.header("설정")

exps = list_experiments()
default_ix = exps.index("50k_national_lam0_mint30_seed_200") \
    if "50k_national_lam0_mint30_seed_200" in exps else 0
exp = st.sidebar.selectbox("실험 폴더", exps, index=default_ix)

regions = load_regions()
region_names = sorted(regions["CTP_KOR_NM"].tolist())
region_sel = st.sidebar.selectbox("지역", [ALL_REGIONS] + region_names, index=0)

viz_mode = st.sidebar.radio(
    "표시 방식",
    ["점 (scatter)", "평균 PDR 분위 히트맵"],   # "밀도맵 (개수)" 필요 시 다시 추가
    horizontal=True,
    help="히트맵은 전체 데이터로 격자별 평균 PDR 분위")
is_density = viz_mode != "점 (scatter)"

if is_density:
    nbins = st.sidebar.slider("밀도 격자 수(bin)", 40, 400, 150, step=10)
else:
    n_points = st.sidebar.slider("생성점 개수(다운샘플)", 2000, 500000, 50000, step=2000)
    pt_size = st.sidebar.slider("생성점 크기", 1, 8, 3)
emoji_size = st.sidebar.slider("시설 이모지 크기", 4, 20, 8)

st.sidebar.markdown("---")
show_hosp = st.sidebar.checkbox("🏥 병원 표시", True)
hosp_show = {}
if show_hosp:
    _cemo = {"red": "🔴", "orange": "🟠", "green": "🟢"}
    for code, tname, color in HOSP_TYPES:
        hosp_show[code] = st.sidebar.checkbox(
            f"　└ {_cemo.get(color, '⚪')} {tname}", True, key=f"hosp_{code}")
show_fire = st.sidebar.checkbox("🚒 소방 표시", True)
show_bnd = st.sidebar.checkbox("경계선", True)

# ── 데이터 로드 + 지역 필터 ──────────────────────────────────────────
_, labels = load_bins()
nq = len(labels)
(fire, f_lat, f_lon, f_name, hosp, h_lat, h_lon, h_name) = load_facilities()

pred = load_pred(os.path.join(MLP_DIR, exp))

if pred.empty or "pdr" not in pred:
    st.error(f"'{exp}' 에서 생성좌표를 찾을 수 없어요 (all_bins.csv / q*.csv 없음).")
    st.stop()

# 제주는 항상 제외 (시설·좌표 모두)
fire = drop_jeju(fire, f_lat, f_lon)
hosp = drop_jeju(hosp, h_lat, h_lon)
pred = drop_jeju(pred, "lat", "lon")

# 지역 선택 시: 해당 시도 within 만 남기고, 축 범위를 그 지역 bounds 로
if region_sel == ALL_REGIONS:
    view = dict(x=[124.5, 130.5], y=[33.8, 38.7])
    bnd_gdf = regions
else:
    pred = pred[pred["_region"] == region_sel]
    fire = fire[fire["_region"] == region_sel]
    hosp = hosp[hosp["_region"] == region_sel]
    bnd_gdf = regions[regions["CTP_KOR_NM"] == region_sel]
    minx, miny, maxx, maxy = bnd_gdf.total_bounds
    px, py = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    view = dict(x=[minx - px, maxx + px], y=[miny - py, maxy + py])

pred = pred.copy()
# scatter 모드만 다운샘플 (밀도맵은 오버플로팅이 없어 전체 사용)
if not is_density and len(pred) > n_points:
    pred = pred.sample(n_points, random_state=0)

# ── figure ───────────────────────────────────────────────────────────
fig = go.Figure()

if show_bnd:
    bx, by = boundary_xy(bnd_gdf)
    fig.add_trace(go.Scatter(x=bx, y=by, mode="lines",
                             line=dict(color="black", width=1, dash="dash"),
                             name="boundary", hoverinfo="skip"))

if viz_mode == "점 (scatter)":
    # 생성점 — 시설을 위에 올리려면 SVG(Scatter) 로 통일 (WebGL 은 SVG 를 덮음)
    fig.add_trace(go.Scatter(
        x=pred["lon"], y=pred["lat"], mode="markers",
        marker=dict(size=pt_size, color=pred["q_idx"], colorscale="RdBu_r",
                    cmin=0, cmax=nq - 1, opacity=0.55,
                    colorbar=dict(title="PDR q-bin", tickmode="array",
                                  tickvals=list(range(nq)), ticktext=labels, len=0.85)),
        name="generated", hoverinfo="skip"))
else:
    # numpy 2D 히스토그램 → 빈 격자는 NaN(투명)으로 두어 바다가 색칠되지 않게
    xe = np.linspace(view["x"][0], view["x"][1], nbins + 1)
    ye = np.linspace(view["y"][0], view["y"][1], nbins + 1)
    cnt, _, _ = np.histogram2d(pred["lon"], pred["lat"], bins=[xe, ye])
    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2
    # ── 밀도맵(개수) — 필요 시 주석 해제 ──────────────────────────────
    # if viz_mode == "밀도맵 (개수)":
    #     z = cnt.T.copy()
    #     z[z == 0] = np.nan
    #     fig.add_trace(go.Heatmap(x=xc, y=yc, z=z, colorscale="Viridis",
    #                              colorbar=dict(title="점 개수", len=0.85),
    #                              hoverinfo="skip", name="density"))
    # ─────────────────────────────────────────────────────────────────
    # 평균 PDR 분위 히트맵: 격자별 q_idx 평균 (빈 격자는 NaN → 투명)
    if True:
        ssum, _, _ = np.histogram2d(pred["lon"], pred["lat"], bins=[xe, ye],
                                    weights=pred["q_idx"])
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(cnt > 0, ssum / cnt, np.nan).T
        fig.add_trace(go.Heatmap(x=xc, y=yc, z=avg, colorscale="RdBu_r",
                                 zmin=0, zmax=nq - 1,
                                 colorbar=dict(title="평균 PDR 분위", tickmode="array",
                                               tickvals=list(range(nq)), ticktext=labels, len=0.85),
                                 hoverinfo="skip", name="pdr_avg"))


def facility_trace(df, lat_c, lon_c, name_c, glyph, label, size=None):
    sub = df.copy()
    sub[lat_c] = pd.to_numeric(sub[lat_c], errors="coerce")
    sub[lon_c] = pd.to_numeric(sub[lon_c], errors="coerce")
    sub = sub.dropna(subset=[lat_c, lon_c])
    text = sub[name_c].astype(str) if name_c else None
    return go.Scatter(  # SVG 렌더러 → 컬러 이모지 정상 표시
        x=sub[lon_c], y=sub[lat_c], mode="text",
        text=[glyph] * len(sub), textfont=dict(size=size or emoji_size),
        hovertext=text, hoverinfo="text",
        name=f"{glyph} {label} (n={len(sub)})")


def hosp_trace(sub, label, color):
    sub = sub.copy()
    sub[h_lat] = pd.to_numeric(sub[h_lat], errors="coerce")
    sub[h_lon] = pd.to_numeric(sub[h_lon], errors="coerce")
    sub = sub.dropna(subset=[h_lat, h_lon])
    text = sub[h_name].astype(str) if h_name else None
    return go.Scatter(  # 컬러 마커 + 그 위에 🏥 이모지
        x=sub[h_lon], y=sub[h_lat], mode="markers+text",
        marker=dict(size=emoji_size * 1.5, color=color, opacity=0.55, line=dict(width=0)),
        text=["🏥"] * len(sub), textfont=dict(size=emoji_size),
        hovertext=text, hoverinfo="text",
        name=f"🏥 {label} (n={len(sub)})")


if show_hosp:
    if "종별코드" in hosp.columns:
        known = [c for c, _, _ in HOSP_TYPES]
        for code, tname, color in HOSP_TYPES:
            if not hosp_show.get(code, True):   # 종별 체크박스 off면 스킵
                continue
            sub = hosp[hosp["종별코드"] == code]
            if len(sub):
                fig.add_trace(hosp_trace(sub, tname, color))
        other = hosp[~hosp["종별코드"].isin(known)]
        if len(other):
            fig.add_trace(hosp_trace(other, "기타병원", "gray"))
    else:
        fig.add_trace(facility_trace(hosp, h_lat, h_lon, h_name, "🏥", "Hospital"))
if show_fire:
    fig.add_trace(facility_trace(fire, f_lat, f_lon, f_name, "🚒", "Fire station"))

fig.update_layout(
    xaxis_title="lon", yaxis_title="lat",
    xaxis=dict(range=view["x"]),
    yaxis=dict(range=view["y"], scaleanchor="x", scaleratio=1.3),
    height=900, template="plotly_white",
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(l=0, r=0, t=10, b=0))

# ── 본문 ─────────────────────────────────────────────────────────────
st.title("MCI 전국 생성좌표 + 병원/소방 시설")
c1, c2, c3, c4 = st.columns(4)
c1.metric("생성점", f"{len(pred):,}")
c2.metric("🏥 병원", f"{len(hosp):,}")
c3.metric("🚒 소방", f"{len(fire):,}")
c4.metric("지역", region_sel if region_sel != ALL_REGIONS else "전국")
st.caption(f"실험: **{exp}**")
st.plotly_chart(fig, use_container_width=True)

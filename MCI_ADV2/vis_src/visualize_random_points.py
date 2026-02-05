#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
랜덤/스냅 좌표 시각화 도구.
- random_metadata.csv: random_latitude/random_longitude + snapped_latitude/snapped_longitude
- grid_metadata.csv: bbox_* + latitude/longitude (보조적으로 지원)
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

BASE_DIR = Path(__file__).resolve().parents[1]

# 17개 시도 매핑
REGION_MAP = {
    "seoul": {"code": "11", "eng": "Seoul", "kor": "서울특별시"},
    "busan": {"code": "26", "eng": "Busan", "kor": "부산광역시"},
    "daegu": {"code": "27", "eng": "Daegu", "kor": "대구광역시"},
    "incheon": {"code": "28", "eng": "Incheon", "kor": "인천광역시"},
    "gwangju": {"code": "29", "eng": "Gwangju", "kor": "광주광역시"},
    "daejeon": {"code": "30", "eng": "Daejeon", "kor": "대전광역시"},
    "ulsan": {"code": "31", "eng": "Ulsan", "kor": "울산광역시"},
    "sejong": {"code": "36", "eng": "Sejong-si", "kor": "세종특별자치시"},
    "gyeonggi": {"code": "41", "eng": "Gyeonggi-do", "kor": "경기도"},
    "chungbuk": {"code": "43", "eng": "Chungcheongbuk-do", "kor": "충청북도"},
    "chungnam": {"code": "44", "eng": "Chungcheongnam-do", "kor": "충청남도"},
    "jeonbuk": {"code": "45", "eng": "Jeollabuk-do", "kor": "전라북도"},
    "jeonnam": {"code": "46", "eng": "Jellanam-do", "kor": "전라남도"},
    "gyeongbuk": {"code": "47", "eng": "Gyeongsangbuk-do", "kor": "경상북도"},
    "gyeongnam": {"code": "48", "eng": "Gyeongsangnam-do", "kor": "경상남도"},
    "jeju": {"code": "50", "eng": "Jeju-do", "kor": "제주특별자치도"},
    "gangwon": {"code": "51", "eng": "Gangwon-do", "kor": "강원특별자치도"},
}

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _has_columns(df, cols):
    return all(col in df.columns for col in cols)


def _print_range(label, series):
    print(f"- {label}: {series.min():.6f} ~ {series.max():.6f}")


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BASE_DIR / path


def _find_region_row(gdf: gpd.GeoDataFrame, region_keyword: str):
    keyword = (region_keyword or "daejeon").strip()
    keyword_lower = keyword.lower()

    region_info = None
    for key, info in REGION_MAP.items():
        if keyword_lower in (key, info["eng"].lower(), info["code"], info["kor"]):
            region_info = info
            break

    candidates = [keyword]
    if region_info:
        candidates = [region_info["code"], region_info["eng"], region_info["kor"], keyword]

    for col in ["CTPRVN_CD", "CTP_ENG_NM", "CTP_KOR_NM"]:
        if col in gdf.columns:
            for cand in candidates:
                mask = gdf[col].astype(str).str.contains(str(cand), case=False, na=False)
                if mask.any():
                    return gdf[mask].iloc[0]

    return None


def load_region_boundary(shp_path: str, region_keyword: str):
    gdf = gpd.read_file(shp_path, encoding="cp949")
    row = _find_region_row(gdf, region_keyword)
    if row is None:
        raise ValueError(f"shapefile에서 지역을 찾지 못했습니다: {region_keyword}")

    if gdf.crs is None:
        gdf.set_crs("EPSG:5179", allow_override=True, inplace=True)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    region_geom = gdf[gdf.index == row.name]["geometry"].iloc[0]
    return region_geom


def visualize_metadata(
    csv_path: str,
    output_path: str,
    shp_path: Optional[str] = None,
    region: str = "daejeon",
):
    print("=" * 70)
    print("랜덤/스냅 좌표 시각화")
    print("=" * 70)

    csv_path = str(_resolve_path(csv_path))
    output_path = str(_resolve_path(output_path))

    print(f"\nCSV 로드: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"- 행 수: {len(df)}")

    has_random = _has_columns(
        df,
        ["random_latitude", "random_longitude", "snapped_latitude", "snapped_longitude"],
    )
    has_grid = _has_columns(df, ["bbox_minlon", "bbox_minlat", "bbox_maxlon", "bbox_maxlat"])

    fig, ax = plt.subplots(figsize=(12, 12))

    # shapefile 경계선
    if shp_path:
        shp_path = str(_resolve_path(shp_path))
        try:
            region_geom = load_region_boundary(shp_path, region)
            gpd.GeoSeries([region_geom]).boundary.plot(ax=ax, color="#111111", linewidth=1.2)
            print(f"- 경계선 표시: {region} ({shp_path})")
        except Exception as e:
            print(f"- 경계선 표시 실패: {e}")

    if has_random:
        print("\n랜덤/스냅 좌표 컬럼 감지")
        _print_range("랜덤 위도", df["random_latitude"])
        _print_range("랜덤 경도", df["random_longitude"])
        _print_range("스냅 위도", df["snapped_latitude"])
        _print_range("스냅 경도", df["snapped_longitude"])

        for _, row in df.iterrows():
            ax.plot(
                [row["random_longitude"], row["snapped_longitude"]],
                [row["random_latitude"], row["snapped_latitude"]],
                color="#999999",
                linewidth=0.3,
                alpha=0.4,
            )

        ax.scatter(
            df["random_longitude"],
            df["random_latitude"],
            s=12,
            color="#1f77b4",
            alpha=0.6,
            label="랜덤 좌표",
        )
        ax.scatter(
            df["snapped_longitude"],
            df["snapped_latitude"],
            s=12,
            color="#d62728",
            alpha=0.8,
            label="스냅 좌표",
        )
        ax.set_title(f"랜덤 vs 스냅 좌표 ({len(df)}개)")
        ax.legend()

    elif has_grid:
        print("\n그리드 메타데이터 컬럼 감지")
        _print_range("그리드 위도", df["latitude"])
        _print_range("그리드 경도", df["longitude"])

        for _, row in df.iterrows():
            minlon = row["bbox_minlon"]
            minlat = row["bbox_minlat"]
            width = row["bbox_maxlon"] - row["bbox_minlon"]
            height = row["bbox_maxlat"] - row["bbox_minlat"]

            rect = patches.Rectangle(
                (minlon, minlat),
                width,
                height,
                linewidth=0.5,
                edgecolor="blue",
                facecolor="lightblue",
                alpha=0.3,
            )
            ax.add_patch(rect)
            ax.plot(row["longitude"], row["latitude"], "r.", markersize=1, alpha=0.5)

        ax.set_title(f"그리드 중심점 ({len(df)}개)")

    else:
        print("\n일반 좌표 컬럼 감지")
        if not _has_columns(df, ["latitude", "longitude"]):
            raise ValueError("CSV에 latitude/longitude 또는 랜덤/스냅 컬럼이 필요합니다.")
        _print_range("위도", df["latitude"])
        _print_range("경도", df["longitude"])

        ax.scatter(df["longitude"], df["latitude"], s=12, alpha=0.7, color="#1f77b4")
        ax.set_title(f"좌표 분포 ({len(df)}개)")

    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    print(f"\n이미지 저장: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # 통계 그래프
    stats_output = output_path.replace(".png", "_stats.png")
    fig2, axes = plt.subplots(3, 2, figsize=(12, 10))

    if has_random:
        axes[0, 0].hist(df["random_latitude"], bins=30, color="#7aa6d8", edgecolor="black")
        axes[0, 0].set_title("랜덤 위도")
        axes[0, 1].hist(df["random_longitude"], bins=30, color="#90c978", edgecolor="black")
        axes[0, 1].set_title("랜덤 경도")
        axes[1, 0].hist(df["snapped_latitude"], bins=30, color="#e07b7b", edgecolor="black")
        axes[1, 0].set_title("스냅 위도")
        axes[1, 1].hist(df["snapped_longitude"], bins=30, color="#eed709", edgecolor="black")
        axes[1, 1].set_title("스냅 경도")
        axes[2, 0].hist(df["snap_distance_m"], bins=30, color="#c0c0c0", edgecolor="black")
        axes[2, 0].set_title("스냅 거리(m)")
        if "N" in df.columns:
            n_values = pd.to_numeric(df["N"], errors="coerce").dropna()
            if not n_values.empty:
                n_min = int(n_values.min())
                n_max = int(n_values.max())
                n_bins = range(n_min, n_max + 2)
                axes[2, 1].hist(n_values, bins=n_bins, color="#f4a460", edgecolor="black", align="left")
                axes[2, 1].set_title("사고 규모 N")
            else:
                axes[2, 1].axis("off")
        else:
            axes[2, 1].axis("off")
    else:
        col_lat = "latitude" if "latitude" in df.columns else None
        col_lon = "longitude" if "longitude" in df.columns else None
        if col_lat:
            axes[0, 0].hist(df[col_lat], bins=30, color="#7aa6d8", edgecolor="black")
            axes[0, 0].set_title("위도")
        if col_lon:
            axes[0, 1].hist(df[col_lon], bins=30, color="#90c978", edgecolor="black")
            axes[0, 1].set_title("경도")
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
        axes[2, 1].axis("off")

    for ax_item in axes.flat:
        ax_item.grid(True, alpha=0.3)

    plt.tight_layout()
    print(f"통계 이미지 저장: {stats_output}")
    plt.savefig(stats_output, dpi=300, bbox_inches="tight")

    print("\n완료")
    print(f"- {output_path}")
    print(f"- {stats_output}\n")
    
    print("\n다음 단계:")
    print(
        f"2. python MCI_ADV2\\sce_src\\batch_experiment.py --base_path {BASE_DIR} "
        f"--random_metadata {Path(output_path).parent}\\random_metadata.csv --config_template {BASE_DIR / 'sim_src' / 'config.yaml'}"
    )



def main():
    import argparse

    parser = argparse.ArgumentParser(description="랜덤/그리드 메타데이터 시각화")
    parser.add_argument("--csv", default="scenarios/daejeon_exp_xxxxx/random_metadata.csv", help="메타데이터 CSV 경로")
    parser.add_argument("--output", default="scenarios/daejeon_exp_xxxxx/random_visualization.png", help="출력 이미지 경로")
    parser.add_argument("--shp", default=str(BASE_DIR/ "scenarios" / "ctprvn.shp"), help="Shapefile 경로")
    parser.add_argument("--region", default="daejeon", help="지역 키워드 (기본: daejeon)")
    args = parser.parse_args()

    visualize_metadata(args.csv, args.output, args.shp, args.region)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

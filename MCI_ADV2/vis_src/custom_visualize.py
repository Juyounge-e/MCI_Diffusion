#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
커스텀/스냅 좌표 시각화 도구.
- custom_metadata.csv: custom_latitude/custom_longitude + snapped_latitude/snapped_longitude
"""

import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

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


def _print_range(label: str, series: pd.Series):
    print(f"- {label}: {series.min():.6f} ~ {series.max():.6f}")


def visualize_custom_metadata(
    csv_path: str,
    output_path: str,
    shp_path: Optional[str] = None,
    region: str = "daejeon",
):
    print("=" * 70)
    print("커스텀/스냅 좌표 시각화")
    print("=" * 70)

    csv_path = str(_resolve_path(csv_path))
    output_path = str(_resolve_path(output_path))

    print(f"\nCSV 로드: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"- 행 수: {len(df)}")

    required_cols = [
        "custom_latitude",
        "custom_longitude",
        "snapped_latitude",
        "snapped_longitude",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV 필수 컬럼 누락: {', '.join(missing)}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    snap_mask = df["snapped_latitude"].notna() & df["snapped_longitude"].notna()
    df_snap = df[snap_mask].copy()
    success_count = len(df_snap)
    failed_count = len(df) - success_count

    print(f"- 스냅 성공: {success_count}")
    print(f"- 스냅 실패: {failed_count}")
    _print_range("커스텀 위도", df["custom_latitude"])
    _print_range("커스텀 경도", df["custom_longitude"])
    if success_count > 0:
        _print_range("스냅 위도", df_snap["snapped_latitude"])
        _print_range("스냅 경도", df_snap["snapped_longitude"])
    else:
        print("- 스냅 위도: N/A")
        print("- 스냅 경도: N/A")

    fig, ax = plt.subplots(figsize=(12, 12))

    if shp_path:
        shp_path = str(_resolve_path(shp_path))
        try:
            region_geom = load_region_boundary(shp_path, region)
            gpd.GeoSeries([region_geom]).boundary.plot(ax=ax, color="#111111", linewidth=1.2)
            print(f"- 경계선 표시: {region} ({shp_path})")
        except Exception as e:
            print(f"- 경계선 표시 실패: {e}")

    for _, row in df_snap.iterrows():
        ax.plot(
            [row["custom_longitude"], row["snapped_longitude"]],
            [row["custom_latitude"], row["snapped_latitude"]],
            color="#999999",
            linewidth=0.3,
            alpha=0.4,
        )

    ax.scatter(
        df["custom_longitude"],
        df["custom_latitude"],
        s=12,
        color="#1f77b4",
        alpha=0.7,
        label="커스텀 좌표",
    )
    if success_count > 0:
        ax.scatter(
            df_snap["snapped_longitude"],
            df_snap["snapped_latitude"],
            s=14,
            color="#d62728",
            alpha=0.9,
            label="스냅 좌표",
        )

    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_title(f"커스텀 vs 스냅 좌표 ({len(df)}개)")
    ax.legend()

    print(f"\n이미지 저장: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    stats_output = output_path.replace(".png", "_stats.png")
    fig2, axes = plt.subplots(3, 2, figsize=(12, 10))

    axes[0, 0].hist(df["custom_latitude"].dropna(), bins=30, color="#7aa6d8", edgecolor="black")
    axes[0, 0].set_title("커스텀 위도")
    axes[0, 1].hist(df["custom_longitude"].dropna(), bins=30, color="#90c978", edgecolor="black")
    axes[0, 1].set_title("커스텀 경도")

    if success_count > 0:
        axes[1, 0].hist(df_snap["snapped_latitude"].dropna(), bins=30, color="#e07b7b", edgecolor="black")
        axes[1, 0].set_title("스냅 위도")
        axes[1, 1].hist(df_snap["snapped_longitude"].dropna(), bins=30, color="#eed709", edgecolor="black")
        axes[1, 1].set_title("스냅 경도")
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    if "N" in df.columns:
        n_values = pd.to_numeric(df["N"], errors="coerce").dropna()
        if not n_values.empty:
            n_min = int(n_values.min())
            n_max = int(n_values.max())
            n_bins = range(n_min, n_max + 2)
            axes[2, 0].hist(n_values, bins=n_bins, color="#f4a460", edgecolor="black", align="left")
            axes[2, 0].set_title("사고 규모 N")
        else:
            axes[2, 0].axis("off")
    else:
        axes[2, 0].axis("off")

    axes[2, 1].bar(["성공", "실패"], [success_count, failed_count], color=["#4caf50", "#ef5350"])
    axes[2, 1].set_title("스냅 결과")

    for ax_item in axes.flat:
        ax_item.grid(True, alpha=0.3)

    plt.tight_layout()
    print(f"통계 이미지 저장: {stats_output}")
    plt.savefig(stats_output, dpi=300, bbox_inches="tight")

    print("\n완료")
    print(f"- {output_path}")
    print(f"- {stats_output}\n")

    print("다음 단계:")
    print(
        f"1. python MCI_ADV2\\sce_src\\batch_experiment.py --base_path {BASE_DIR} "
        f"--random_metadata {csv_path} --config_template {BASE_DIR / 'sim_src' / 'config.yaml'}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="커스텀/스냅 메타데이터 시각화")
    parser.add_argument("--csv", required=True, help="custom_metadata.csv 경로")
    parser.add_argument("--output", default="scenarios/custom_exp_xxxxx/custom_visualization.png", help="출력 이미지 경로")
    parser.add_argument("--shp", default=str(BASE_DIR / "scenarios" / "ctprvn.shp"), help="Shapefile 경로")
    parser.add_argument("--region", default="daejeon", help="지역 키워드 (기본: daejeon)")
    args = parser.parse_args()

    visualize_custom_metadata(args.csv, args.output, args.shp, args.region)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

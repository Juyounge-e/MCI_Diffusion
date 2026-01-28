#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCI 실험 결과 Folium 인터랙티브 시각화

대전광역시 그리드 실험 결과를 Folium 지도 위에 시각화
- 5개 메트릭: reward, time, pdr, wog_reward, wog_pdr
- 회색 모드 vs 반복 보간 모드 (버튼으로 전환)
- 히트맵 스타일 컬러맵
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from branca.element import Template, MacroElement
import warnings
warnings.filterwarnings('ignore')

# Windows 콘솔은 기본 stdout 인코딩 사용


# 메트릭별 컬러맵 (히트맵 스타일)
METRIC_COLORMAPS = {
    'reward': 'RdBu',        # low=red, high=blue (higher is better)
    'time': 'RdBu_r',        # low=blue, high=red (lower is better)
    'pdr': 'RdBu_r',         # low=blue, high=red (lower is better)
    'wog_reward': 'RdBu',
    'wog_pdr': 'RdBu_r'
}

METRIC_LABELS = {
    'reward': 'Reward (누적 생존 보상)',
    'time': 'Time (MCI 완료 시간, 분)',
    'pdr': 'PDR (예방가능 사망률)',
    'wog_reward': 'WOG Reward (녹색 제외)',
    'wog_pdr': 'WOG PDR (녹색 제외)'
}








def find_latest_experiment():
    """scenarios/ 폴더에서 최신 실험 찾기"""
    print("최신 실험 검색중...")

    # exp_* 패턴의 폴더 찾기
    exp_folders = glob.glob('scenarios/exp_*')

    if not exp_folders:
        raise FileNotFoundError("scenarios/ 폴더에 exp_* 실험 폴더가 없습니다.")

    # 가장 최신 폴더 선택 (이름 기준 정렬, 타임스탬프 포함)
    latest_exp = sorted(exp_folders)[-1]
    exp_id = os.path.basename(latest_exp)

    print(f"   최신 실험: {exp_id}")
    return exp_id


def load_metadata(exp_id):
    """grid_metadata.csv + experiment_metadata.csv 병합"""
    print()
    print(f"메타데이터 로딩 중: {exp_id}")

    # grid_metadata.csv 경로 찾기
    grid_csv_paths = [
        f'scenarios/{exp_id}/grid_metadata.csv',  # 새로운 구조
        'grid_metadata.csv'  # 기존 구조 (fallback)
    ]

    grid_csv = None
    for path in grid_csv_paths:
        if os.path.exists(path):
            grid_csv = path
            break

    if grid_csv is None:
        raise FileNotFoundError(f"grid_metadata.csv를 찾을 수 없습니다: {grid_csv_paths}")

    print(f"   Grid metadata: {grid_csv}")
    df_grid = pd.read_csv(grid_csv)

    # experiment_metadata.csv
    exp_csv = f'scenarios/{exp_id}/experiment_metadata.csv'
    if not os.path.exists(exp_csv):
        raise FileNotFoundError(f"experiment_metadata.csv를 찾을 수 없습니다: {exp_csv}")

    print(f"   Experiment metadata: {exp_csv}")
    df_exp = pd.read_csv(exp_csv)

    # 병합
    df = df_grid.merge(df_exp[['grid_id', 'status', 'failure_reason']], on='grid_id', how='left')
    df['status'] = df['status'].fillna('unknown')
    df['failure_reason'] = df['failure_reason'].fillna('')

    print(f"   총 {len(df)}개 그리드 로드")
    print(f"      - 성공: {(df['status'] == 'success').sum()}개")
    print(f"      - 실패: {(df['status'] == 'failed').sum()}개")

    return df


def parse_stat_file(filepath):
    """stat.txt 파싱 후 {metric: {mean, std, ci_half}} 반환"""
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        metric_names = ['reward', 'time', 'pdr', 'wog_reward', 'wog_pdr']
        metrics = {}

        for i, metric_name in enumerate(metric_names):
            if i < len(lines):
                parts = lines[i].strip().split()
                # 마지막 3개가 mean, std, ci_half
                if len(parts) >= 3:
                    metrics[metric_name] = {
                        'mean': float(parts[-3]),
                        'std': float(parts[-2]),
                        'ci_half': float(parts[-1])
                    }

        return metrics
    except Exception as e:
        print(f"  stat.txt 파싱 실패: {filepath} - {e}")
        return None


def load_experiment_results(df, exp_id):
    """실험 결과 파일 파싱하여 메트릭 추가"""
    print()
    print("실험 결과 로딩 중...")

    results_dir = f'results/{exp_id}'
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"결과 폴더가 없습니다: {results_dir}")

    # 각 메트릭에 새 컬럼 추가
    for metric in ['reward', 'time', 'pdr', 'wog_reward', 'wog_pdr']:
        df[f'{metric}_mean'] = np.nan
        df[f'{metric}_std'] = np.nan
        df[f'{metric}_ci_half'] = np.nan

    # 성공한 그리드만 결과 파싱
    success_count = 0
    for idx, row in df[df['status'] == 'success'].iterrows():
        lat = row['latitude']
        lon = row['longitude']

        # stat.txt 경로
        stat_path = os.path.join(
            results_dir,
            f'lat{lat:.6f}_lon{lon:.6f}',
            f'results_lat{lat:.6f}_lon{lon:.6f}_stat.txt'
        )

        metrics = parse_stat_file(stat_path)
        if metrics:
            for metric_name, values in metrics.items():
                df.at[idx, f'{metric_name}_mean'] = values['mean']
                df.at[idx, f'{metric_name}_std'] = values['std']
                df.at[idx, f'{metric_name}_ci_half'] = values['ci_half']
            success_count += 1

    print(f"   {success_count}개 그리드 결과 파싱 완료")
    return df


def calculate_median_spacing(coordinates):
    """그리드 간격 계산 (median)"""
    coords_sorted = np.sort(coordinates)
    diffs = np.diff(coords_sorted)
    # 0이 아닌 차이만 고려 (1e-5보다 큰 값)
    diffs_nonzero = diffs[diffs > 1e-5]
    if len(diffs_nonzero) > 0:
        median_spacing = np.median(diffs_nonzero)
        # 너무 작은 값이면 기본값 사용 (500m ≈ 0.0045도)
        if median_spacing < 0.001:
            return 0.0045
        return median_spacing
    return 0.0045  # 기본값 (약 500m)


def iterative_interpolation(df, metric_name, max_iterations=50):
    """
    반복 보간으로 실패한 그리드 채우기

    알고리즘:
    1. 실패한 그리드 리스트 생성
    2. while 실패 그리드가 남아있고 개선 가능
       a. 각 실패 그리드에 대해
          - 상하좌우 인접 그리드 중 값이 있는 것들 선택
          - 평균 계산하여 채우기
       b. 채워진 그리드는 실패 리스트에서 제거
       c. 개선이 없으면 루프 종료
    3. 남은 결측은 최근접 유효 값으로 채움
    """
    df_interp = df.copy()
    metric_col = f'{metric_name}_mean'

    # 보간 플래그 컬럼 미리 준비
    if 'interpolated' not in df_interp.columns:
        df_interp['interpolated'] = False
    if 'interpolation_iter' not in df_interp.columns:
        df_interp['interpolation_iter'] = 0

    # 그리드 간격 계산
    lat_spacing = calculate_median_spacing(df_interp['latitude'].values)
    lon_spacing = calculate_median_spacing(df_interp['longitude'].values)
    tolerance = 0.001  # 부동소수점 오차 허용

    print(f"   {metric_name} 반복 보간 중...")
    print(f"      그리드 간격: lat={lat_spacing:.6f}°, lon={lon_spacing:.6f}°")

    # 실패한 그리드 인덱스 (값이 없는 그리드)
    failed_indices = df_interp[df_interp[metric_col].isna()].index.tolist()
    initial_failed = len(failed_indices)

    if initial_failed == 0:
        print("      모든 그리드에 값이 있음 (보간 불필요)")
        return df_interp

    print(f"      시작: {initial_failed}개 그리드 값 없음")

    iteration = 0
    filled_total = 0

    while failed_indices and iteration < max_iterations:
        filled_in_this_iteration = []

        for idx in failed_indices:
            row = df_interp.loc[idx]
            lat, lon = row['latitude'], row['longitude']

            # 상하좌우 인접 그리드 찾기
            neighbors = []
            for dlat, dlon in [(lat_spacing, 0), (-lat_spacing, 0), (0, lon_spacing), (0, -lon_spacing)]:
                matches = df_interp[
                    (np.abs(df_interp['latitude'] - (lat + dlat)) < tolerance) &
                    (np.abs(df_interp['longitude'] - (lon + dlon)) < tolerance) &
                    (df_interp[metric_col].notna())  # 값이 있는 그리드만
                ]
                if not matches.empty:
                    neighbors.append(matches.iloc[0][metric_col])

            # 인접 그리드가 있으면 평균 계산
            if neighbors:
                avg_value = sum(neighbors) / len(neighbors)
                df_interp.at[idx, metric_col] = avg_value
                df_interp.at[idx, 'interpolated'] = True
                df_interp.at[idx, 'interpolation_iter'] = iteration + 1
                filled_in_this_iteration.append(idx)

        # 채워진 그리드는 실패 리스트에서 제거
        failed_indices = [i for i in failed_indices if i not in filled_in_this_iteration]
        filled_total += len(filled_in_this_iteration)

        if filled_in_this_iteration:
            print(f"      반복 {iteration + 1}: {len(filled_in_this_iteration)}개 채움 (남은 실패: {len(failed_indices)}개)")

        # 개선이 없으면 종료
        if not filled_in_this_iteration:
            break

        iteration += 1

    # 남은 결측은 최근접 유효 값으로 채움
    if failed_indices:
        print(f"      잔여 {len(failed_indices)}개 그리드: 최근접 값으로 마무리 보간")
        valid_mask = df_interp[metric_col].notna()
        valid_coords = df_interp.loc[valid_mask, ['latitude', 'longitude']].to_numpy()
        valid_values = df_interp.loc[valid_mask, metric_col].to_numpy()

        if len(valid_values) == 0:
            print("      유효 값이 없어 최근접 보간을 건너뜀")
        else:
            for idx in failed_indices:
                lat = df_interp.at[idx, 'latitude']
                lon = df_interp.at[idx, 'longitude']
                dists = (valid_coords[:, 0] - lat) ** 2 + (valid_coords[:, 1] - lon) ** 2
                nearest_idx = int(dists.argmin())
                df_interp.at[idx, metric_col] = float(valid_values[nearest_idx])
                df_interp.at[idx, 'interpolated'] = True
                df_interp.at[idx, 'interpolation_iter'] = iteration + 1

            filled_total += len(failed_indices)
            failed_indices = []

    print(f"      총 {filled_total}개 그리드 보간 완료 ({iteration}회 반복)")

    return df_interp
def create_colormap(metric_name, values):
    """히트맵 컬러맵 + 정규화"""
    # NaN 제거
    valid_values = values.dropna()

    if len(valid_values) == 0:
        # 값이 없으면 기본 정규화
        return plt.cm.viridis, mcolors.Normalize(vmin=0, vmax=1), 0, 1

    # Percentile-based 정규화 (이상치 제거)
    vmin = valid_values.quantile(0.05)
    vmax = valid_values.quantile(0.95)

    # 컬러맵 선택
    cmap_name = METRIC_COLORMAPS.get(metric_name, 'viridis')
    cmap = plt.cm.get_cmap(cmap_name)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    return cmap, norm, vmin, vmax


def get_color_hex(value, cmap, norm):
    """값을 hex 색상으로 변환"""
    if pd.isna(value):
        return '#CCCCCC'  # 회색 (값 없음)

    rgba = cmap(norm(value))
    return mcolors.to_hex(rgba)


def load_region_boundary(shp_path, exp_id):
    """지역 경계 shapefile 로딩"""
    print(f"\n🗺️  지역 경계 로딩 중...")

    if not os.path.exists(shp_path):
        print(f"   ⚠️ Shapefile 없음: {shp_path}")
        return None

    try:
        gdf = gpd.read_file(shp_path, encoding='cp949')

        # 실험 ID에서 지역명 추출 (예: daejeon_exp_... → daejeon)
        # 기존 실험은 exp_로 시작하므로 대전으로 가정
        if exp_id.startswith('exp_'):
            region_keyword = 'daejeon'
        else:
            region_keyword = exp_id.split('_')[0]

        # 대전 찾기 (기존 로직)
        daejeon_keywords = ['대전', 'Daejeon', 'DAEJEON']
        daejeon_row = None

        for col in gdf.columns:
            if gdf[col].dtype == 'object':
                for keyword in daejeon_keywords:
                    mask = gdf[col].astype(str).str.contains(keyword, case=False, na=False)
                    if mask.any():
                        daejeon_row = gdf[mask].iloc[0]
                        break
                if daejeon_row is not None:
                    break

        if daejeon_row is None:
            print(f"   ⚠️ 대전 경계를 찾을 수 없습니다.")
            return None

        # GeoDataFrame 생성
        boundary = gpd.GeoDataFrame([daejeon_row], crs=gdf.crs)

        # CRS가 없으면 설정
        if boundary.crs is None:
            boundary.set_crs("EPSG:5179", inplace=True)

        # WGS84로 변환
        if boundary.crs.to_epsg() != 4326:
            boundary = boundary.to_crs(epsg=4326)

        print(f"   ✅ 대전광역시 경계 로드 완료 (CRS: {boundary.crs})")
        return boundary

    except Exception as e:
        print(f"   ⚠️ Shapefile 로드 실패: {e}")
        return None


def add_grid_centers_layer(map_obj, df, show=False):
    """Add grid center points as a toggleable layer."""
    if df.empty:
        return None

    fg = folium.FeatureGroup(name='Grid Centers', show=show)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=1.5,
            color='#111111',
            weight=0,
            fill=True,
            fillColor='#111111',
            fillOpacity=0.85,
            interactive=False
        ).add_to(fg)

    fg.add_to(map_obj)
    return fg


def add_grid_lines_layer(map_obj, df, show=True):
    """Add grid midlines (through centers) as a toggleable layer."""
    if df.empty:
        return None

    min_lat = float(df['bbox_minlat'].min())
    max_lat = float(df['bbox_maxlat'].max())
    min_lon = float(df['bbox_minlon'].min())
    max_lon = float(df['bbox_maxlon'].max())

    lat_lines = sorted({round(val, 6) for val in df['latitude'].tolist()})
    lon_lines = sorted({round(val, 6) for val in df['longitude'].tolist()})

    fg = folium.FeatureGroup(name='Grid Lines', show=show)

    line_kwargs = {
        'color': '#333333',
        'weight': 0.6,
        'opacity': 0.7,
        'interactive': False
    }

    for lat in lat_lines:
        folium.PolyLine(
            locations=[[lat, min_lon], [lat, max_lon]],
            **line_kwargs
        ).add_to(fg)

    for lon in lon_lines:
        folium.PolyLine(
            locations=[[min_lat, lon], [max_lat, lon]],
            **line_kwargs
        ).add_to(fg)

    fg.add_to(map_obj)
    return fg


def create_folium_map(df_gray, df_interp, boundary_gdf, exp_id):
    """메인 Folium 지도 생성"""
    print(f"\n🗺️  Folium 지도 생성 중...")

    # 지도 중심 계산
    center_lat = df_gray['latitude'].mean()
    center_lon = df_gray['longitude'].mean()

    # 지도 초기화
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap',
        prefer_canvas=True,
        control_scale=True
    )

    # 추가 타일 레이어
    folium.TileLayer('CartoDB positron', name='Light Mode').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)

    # 대전 경계선 추가
    if boundary_gdf is not None:
        folium.GeoJson(
            boundary_gdf,
            name='대전광역시 경계',
            style_function=lambda x: {
                'fillColor': 'none',
                'color': '#FF0000',
                'weight': 3,
                'opacity': 0.8,
                'dashArray': '5, 5'
            },
            overlay=True,
            control=False
        ).add_to(m)

    # 메트릭별 레이어 생성 (회색 모드 + 보간 모드)
    metrics = ['reward', 'time', 'pdr', 'wog_reward', 'wog_pdr']

    for mode_name, df_mode in [('gray', df_gray), ('interp', df_interp)]:
        mode_label = '회색' if mode_name == 'gray' else '보간'

        for metric in metrics:
            metric_col = f'{metric}_mean'

            # 컬러맵 생성
            cmap, norm, vmin, vmax = create_colormap(metric, df_mode[metric_col])

            # FeatureGroup 생성
            show_layer = (metric == 'reward' and mode_name == 'gray')  # reward + 회색 모드 기본 표시
            fg = folium.FeatureGroup(
                name=f'{METRIC_LABELS[metric]} ({mode_label})',
                show=show_layer
            )

            # 각 그리드 폴리곤 추가
            for idx, row in df_mode.iterrows():
                # 폴리곤 좌표
                coords = [
                    [row['bbox_minlat'], row['bbox_minlon']],
                    [row['bbox_maxlat'], row['bbox_minlon']],
                    [row['bbox_maxlat'], row['bbox_maxlon']],
                    [row['bbox_minlat'], row['bbox_maxlon']],
                    [row['bbox_minlat'], row['bbox_minlon']]
                ]

                # 색상 결정
                value = row[metric_col]
                if pd.isna(value):
                    fill_color = '#000000'
                    fill_opacity = 0.65
                    fill_pattern = None
                else:
                    fill_color = get_color_hex(value, cmap, norm)
                    fill_opacity = 0.7
                    fill_pattern = None

                # 툴팁 생성
                if pd.isna(value):
                    tooltip_text = f"Grid {int(row['grid_id'])}: 값 없음"
                else:
                    is_interpolated = row.get('interpolated', False)
                    interp_tag = ' (보간)' if is_interpolated else ''
                    tooltip_text = f"Grid {int(row['grid_id'])}: {value:.4f}{interp_tag}"

                # 팝업 생성
                if row['status'] == 'failed':
                    popup_html = f"""
                    <div style="width: 250px; font-family: Arial;">
                        <h4 style="color: #cc0000;">Grid {int(row['grid_id'])} - 실패</h4>
                        <p><b>좌표:</b> {row['latitude']:.6f}, {row['longitude']:.6f}</p>
                        <p><b>이유:</b> {row['failure_reason']}</p>
                    </div>
                    """
                else:
                    popup_html = f"""
                    <div style="width: 250px; font-family: Arial;">
                        <h4 style="color: #007700;">Grid {int(row['grid_id'])}</h4>
                        <p><b>좌표:</b> {row['latitude']:.6f}, {row['longitude']:.6f}</p>
                        <p><b>Reward:</b> {row.get('reward_mean', np.nan):.4f}</p>
                        <p><b>Time:</b> {row.get('time_mean', np.nan):.2f} 분</p>
                        <p><b>PDR:</b> {row.get('pdr_mean', np.nan):.4f}</p>
                        <p><b>WOG Reward:</b> {row.get('wog_reward_mean', np.nan):.4f}</p>
                        <p><b>WOG PDR:</b> {row.get('wog_pdr_mean', np.nan):.4f}</p>
                    </div>
                    """

                # 폴리곤 추가
                polygon_kwargs = {
                    'locations': coords,
                    'color': '#333333',
                    'weight': 0,
                    'opacity': 0.0,
                    'fill': True,
                    'fillColor': fill_color,
                    'fillOpacity': fill_opacity,
                    'tooltip': tooltip_text,
                    'popup': folium.Popup(popup_html, max_width=300),
                    'stroke': False
                }
                folium.Polygon(**polygon_kwargs).add_to(fg)

            fg.add_to(m)

    # LayerControl 추가
    # Grid overlays (toggle in LayerControl)
    add_grid_lines_layer(m, df_gray, show=True)
    add_grid_centers_layer(m, df_gray, show=False)

    folium.LayerControl(
        collapsed=False,
        position='topright'
    ).add_to(m)

    # Legend count (gray mode, reward base)
    total_grids = len(df_gray)
    no_data_count = df_gray['reward_mean'].isna().sum()

    # Legend (red=bad/low, blue=good/high, black=missing)
    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #333;
        border-radius: 6px;
        padding: 10px 12px;
        font-family: Arial, sans-serif;
        font-size: 12px;
        color: #111;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    ">
        <div style="font-weight: 700; margin-bottom: 6px;">Legend</div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 120px; height: 10px;
                        background: linear-gradient(90deg, #b2182b 0%, #f7f7f7 50%, #2166ac 100%);
                        border: 1px solid #555;"></div>
        <div>Low/Bad <-> High/Good</div>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 14px; height: 14px; background: #000000; border: 1px solid #333;"></div>
            <div>No data: {no_data_count} / {total_grids}</div>
        </div>
    </div>
    {{% endmacro %}}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    print(f"   ✅ 지도 생성 완료")
    print(f"      - {len(metrics)} 메트릭 × 2 모드 = {len(metrics) * 2} 레이어")

    return m


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='MCI 실험 결과 Folium 시각화')
    parser.add_argument(
        '--exp-id',
        dest='exp_id',
        default=None,
        help='실험 ID (기본: 최신 실험 자동 선택)'
    )
    parser.add_argument(
        '--shp',
        default='scenarios/ctprvn.shp',
        help='Shapefile 경로 (기본: scenarios/ctprvn.shp)'
    )
    args = parser.parse_args()

    print("="*70)
    print("🗺️  MCI 실험 결과 Folium 시각화")
    print("="*70)

    try:
        # 1. 최신 실험 찾기
        if args.exp_id is None:
            exp_id = find_latest_experiment()
        else:
            exp_id = args.exp_id

        # 2. 메타데이터 로딩
        df = load_metadata(exp_id)

        # 3. 실험 결과 파싱
        df = load_experiment_results(df, exp_id)

        # 4. 회색 모드 데이터
        df_gray = df.copy()

        # 5. 보간 모드 데이터 (각 메트릭별로 반복 보간 적용)
        print(f"\n🔄 반복 보간 시작...")
        df_interp = df.copy()
        for metric in ['reward', 'time', 'pdr', 'wog_reward', 'wog_pdr']:
            df_interp = iterative_interpolation(df_interp, metric)

        # 6. 지역 경계 로딩
        boundary = load_region_boundary(args.shp, exp_id)

        # 7. Folium 지도 생성
        m = create_folium_map(df_gray, df_interp, boundary, exp_id)

        # 8. HTML 저장
        output_dir = 'visualizations'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{exp_id}_MCI_visualization.html')

        m.save(output_path)

        print("\n" + "="*70)
        print("✅ 시각화 완료!")
        print("="*70)
        print(f"\n📂 출력 파일: {output_path}")
        print(f"\n브라우저에서 {output_path}를 열어 확인하세요.\n")

        return 0

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

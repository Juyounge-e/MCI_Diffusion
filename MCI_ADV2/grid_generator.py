#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
한국 시도별 500m×500m 그리드 생성기

지정한 시도 경계 shapefile에서 500m×500m 그리드를 생성하고
중심점이 경계 내부인 그리드만 선택하여 grid_metadata.csv로 저장
"""

import os
import sys
import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
from pyproj import Transformer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Windows 콘솔 UTF-8 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# 지역 코드 매핑 (17개 시도)
REGION_MAP = {
    'seoul': {'code': '11', 'eng': 'Seoul', 'kor': '서울특별시'},
    'busan': {'code': '26', 'eng': 'Busan', 'kor': '부산광역시'},
    'daegu': {'code': '27', 'eng': 'Daegu', 'kor': '대구광역시'},
    'incheon': {'code': '28', 'eng': 'Incheon', 'kor': '인천광역시'},
    'gwangju': {'code': '29', 'eng': 'Gwangju', 'kor': '광주광역시'},
    'daejeon': {'code': '30', 'eng': 'Daejeon', 'kor': '대전광역시'},
    'ulsan': {'code': '31', 'eng': 'Ulsan', 'kor': '울산광역시'},
    'sejong': {'code': '36', 'eng': 'Sejong-si', 'kor': '세종특별자치시'},
    'gyeonggi': {'code': '41', 'eng': 'Gyeonggi-do', 'kor': '경기도'},
    'chungbuk': {'code': '43', 'eng': 'Chungcheongbuk-do', 'kor': '충청북도'},
    'chungnam': {'code': '44', 'eng': 'Chungcheongnam-do', 'kor': '충청남도'},
    'jeonbuk': {'code': '45', 'eng': 'Jeollabuk-do', 'kor': '전라북도'},
    'jeonnam': {'code': '46', 'eng': 'Jellanam-do', 'kor': '전라남도'},
    'gyeongbuk': {'code': '47', 'eng': 'Gyeongsangbuk-do', 'kor': '경상북도'},
    'gyeongnam': {'code': '48', 'eng': 'Gyeongsangnam-do', 'kor': '경상남도'},
    'jeju': {'code': '50', 'eng': 'Jeju-do', 'kor': '제주특별자치도'},
    'gangwon': {'code': '51', 'eng': 'Gangwon-do', 'kor': '강원특별자치도'}
}


def load_region_boundary(shp_path, region_keyword):
    """
    지역 경계 폴리곤 로드 (17개 시도 지원)

    Args:
        shp_path: shapefile 경로
        region_keyword: 지역 키워드 ('대전', 'daejeon', '서울', 'seoul', 등)

    Returns:
        tuple: (region_boundary GeoDataFrame, region_name str)
    """
    print(f"📂 Shapefile 로드 중: {shp_path}")

    # Shapefile 로드
    gdf = gpd.read_file(shp_path, encoding='cp949')
    print(f"   총 {len(gdf)}개 시도 발견")
    print(f"   컬럼: {list(gdf.columns)}")

    # 키워드 정규화 (소문자, 공백 제거)
    keyword_lower = region_keyword.lower().strip()

    # REGION_MAP에서 검색
    region_info = None
    region_name = None
    for key, info in REGION_MAP.items():
        if (keyword_lower == key or
            keyword_lower == info['eng'].lower() or
            keyword_lower == info['kor'] or
            keyword_lower == info['code']):
            region_info = info
            region_name = key
            break

    if not region_info:
        raise ValueError(
            f"지역을 찾을 수 없습니다: '{region_keyword}'. "
            f"사용 가능한 지역: {', '.join(REGION_MAP.keys())}"
        )

    # shapefile에서 해당 지역 찾기 (CTPRVN_CD, CTP_ENG_NM, CTP_KOR_NM 컬럼 사용)
    region_row = None
    for col in ['CTPRVN_CD', 'CTP_ENG_NM', 'CTP_KOR_NM']:
        if col in gdf.columns:
            matches = gdf[
                (gdf[col].astype(str) == region_info['code']) |
                (gdf[col].astype(str) == region_info['eng']) |
                (gdf[col].astype(str) == region_info['kor'])
            ]
            if not matches.empty:
                region_row = matches.iloc[0]
                break

    if region_row is None:
        raise ValueError(f"Shapefile에서 {region_info['kor']}를 찾을 수 없습니다.")

    print(f"   ✅ {region_info['kor']} ({region_info['eng']}) 발견")

    # 지역 폴리곤 추출
    region_geom = region_row['geometry']

    # WGS84로 변환 (EPSG:4326)
    if gdf.crs is None:
        # CRS 정보가 없으면 EPSG:5179 (한국 중부 원점)로 가정
        print("   ⚠️  CRS 정보 없음, EPSG:5179 (한국 중부 원점)로 가정")
        gdf.set_crs("EPSG:5179", allow_override=True, inplace=True)
        print(f"   🔄 CRS 변환: EPSG:5179 → EPSG:4326 (WGS84)")
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        region_geom = gdf_wgs84[gdf_wgs84.index == region_row.name]['geometry'].iloc[0]
    elif gdf.crs.to_epsg() != 4326:
        print(f"   🔄 CRS 변환: {gdf.crs} → EPSG:4326 (WGS84)")
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        region_geom = gdf_wgs84[gdf_wgs84.index == region_row.name]['geometry'].iloc[0]
    else:
        region_geom = region_row['geometry']

    print(f"   ✅ {region_info['kor']} 경계 로드 완료")
    return region_geom, region_name


def generate_grid_cells(boundary_polygon, cell_size_meters=500):
    """
    500m×500m 그리드 생성

    Args:
        boundary_polygon: 경계 폴리곤 (WGS84)
        cell_size_meters: 그리드 셀 크기 (미터)

    Returns:
        그리드 셀 리스트 [(grid_id, lat, lon, polygon), ...]
    """
    print(f"\n📐 {cell_size_meters}m×{cell_size_meters}m 그리드 생성 중...")

    # Bounding box 가져오기 (WGS84)
    minx, miny, maxx, maxy = boundary_polygon.bounds
    print(f"   WGS84 Bounding box: ({minx:.6f}, {miny:.6f}) ~ ({maxx:.6f}, {maxy:.6f})")

    # WGS84 → UTM-K (EPSG:5179) 변환기
    # 한국 중부 원점 좌표계 (미터 단위)
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    transformer_to_wgs = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

    # 경계를 UTM으로 변환 (boundary_polygon은 이미 WGS84)
    boundary_utm = gpd.GeoSeries([boundary_polygon], crs="EPSG:4326").to_crs("EPSG:5179").iloc[0]
    minx_utm, miny_utm, maxx_utm, maxy_utm = boundary_utm.bounds

    print(f"   UTM 좌표: ({minx_utm:.1f}, {miny_utm:.1f}) ~ ({maxx_utm:.1f}, {maxy_utm:.1f})")

    # 그리드 생성 (UTM 좌표계에서)
    grid_cells = []
    grid_id = 1

    x = minx_utm
    while x < maxx_utm:
        y = miny_utm
        while y < maxy_utm:
            # 그리드 셀 생성
            cell_utm = box(x, y, x + cell_size_meters, y + cell_size_meters)

            # WGS84로 변환
            coords_utm = list(cell_utm.exterior.coords)
            coords_wgs = [transformer_to_wgs.transform(pt[0], pt[1]) for pt in coords_utm]
            cell_wgs = box(
                min(c[0] for c in coords_wgs),
                min(c[1] for c in coords_wgs),
                max(c[0] for c in coords_wgs),
                max(c[1] for c in coords_wgs)
            )

            # 중심점 계산 (WGS84)
            centroid = cell_wgs.centroid
            lat, lon = centroid.y, centroid.x

            grid_cells.append((grid_id, lat, lon, cell_wgs))
            grid_id += 1

            y += cell_size_meters
        x += cell_size_meters

    print(f"   ✅ 총 {len(grid_cells)}개 그리드 셀 생성")
    return grid_cells


def filter_grids_by_centroid(grid_cells, boundary_polygon):
    """
    중심점이 경계 내부인 그리드만 필터링

    Args:
        grid_cells: 그리드 셀 리스트
        boundary_polygon: 경계 폴리곤 (WGS84)

    Returns:
        필터링된 그리드 리스트
    """
    print(f"\n🔍 경계 필터링 중 (중심점 기준)...")

    valid_grids = []
    for grid_id, lat, lon, polygon in grid_cells:
        centroid = Point(lon, lat)
        if boundary_polygon.contains(centroid):
            valid_grids.append((grid_id, lat, lon, polygon))

    print(f"   ✅ {len(valid_grids)}/{len(grid_cells)}개 그리드 선택됨")
    print(f"   (제외: {len(grid_cells) - len(valid_grids)}개)")

    return valid_grids


def export_grid_metadata(valid_grids, output_path):
    """
    그리드 메타데이터 CSV 저장

    Args:
        valid_grids: 유효한 그리드 리스트
        output_path: 출력 CSV 경로
    """
    print(f"\n💾 그리드 메타데이터 저장 중: {output_path}")

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"   📁 디렉토리 생성: {output_dir}")

    # DataFrame 생성
    data = []
    for grid_id, lat, lon, polygon in valid_grids:
        minx, miny, maxx, maxy = polygon.bounds
        data.append({
            'grid_id': grid_id,
            'latitude': lat,
            'longitude': lon,
            'bbox_minlon': minx,
            'bbox_minlat': miny,
            'bbox_maxlon': maxx,
            'bbox_maxlat': maxy
        })

    df = pd.DataFrame(data)

    # grid_id 순으로 재정렬하고 1부터 연속으로 재할당
    df = df.sort_values('grid_id').reset_index(drop=True)
    df['grid_id'] = range(1, len(df) + 1)

    # CSV 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"   ✅ {len(df)}개 그리드 메타데이터 저장 완료")
    print(f"\n📊 요약:")
    print(f"   - 총 그리드: {len(df)}개")
    print(f"   - 위도 범위: {df['latitude'].min():.6f} ~ {df['latitude'].max():.6f}")
    print(f"   - 경도 범위: {df['longitude'].min():.6f} ~ {df['longitude'].max():.6f}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="한국 시도별 500m×500m 그리드 생성")
    parser.add_argument(
        '--region',
        default='daejeon',
        help=f'지역 선택 (기본: daejeon). 사용 가능: {", ".join(REGION_MAP.keys())}'
    )
    parser.add_argument(
        '--exp-id',
        dest='exp_id',
        default=None,
        help='실험 ID (기본: 자동 생성 exp_YYYYMMDD_HHMMSS)'
    )
    parser.add_argument(
        '--grid-size',
        dest='grid_size',
        type=int,
        default=500,
        help='그리드 셀 크기 (미터, 기본: 500)'
    )
    parser.add_argument(
        '--shp',
        default='scenarios/ctprvn.shp',
        help='Shapefile 경로 (기본: scenarios/ctprvn.shp)'
    )
    args = parser.parse_args()

    # 실험 ID 생성
    if args.exp_id is None:
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        exp_id = args.exp_id

    print("="*70)
    print("🌐 한국 시도별 그리드 생성기")
    print("="*70)
    print(f"   지역: {args.region}")
    print(f"   실험 ID: {exp_id}")
    print(f"   그리드 크기: {args.grid_size}m × {args.grid_size}m")
    print("="*70)

    # 1. 지역 경계 로드
    try:
        region_polygon, region_name = load_region_boundary(args.shp, args.region)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return 1

    # 2. 그리드 생성
    try:
        grid_cells = generate_grid_cells(region_polygon, args.grid_size)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return 1

    # 3. 중심점 기준 필터링
    try:
        valid_grids = filter_grids_by_centroid(grid_cells, region_polygon)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return 1

    if len(valid_grids) == 0:
        print("\n❌ 오류: 유효한 그리드가 없습니다.")
        return 1

    # 4. 메타데이터 저장 (새로운 경로: scenarios/{region}_{exp_id}/grid_metadata.csv)
    output_dir = f"scenarios/{region_name}_{exp_id}"
    output_path = os.path.join(output_dir, 'grid_metadata.csv')

    try:
        export_grid_metadata(valid_grids, output_path)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return 1

    print("\n" + "="*70)
    print("✅ 그리드 생성 완료!")
    print("="*70)
    print(f"\n📂 출력 위치: {output_path}")
    print(f"\n다음 단계:")
    print(f"1. {output_path} 파일 확인")
    print(f"2. python visualize_grid.py --csv {output_path}")
    print(f"3. python test_single_grid.py --exp-id {region_name}_{exp_id}")
    print(f"4. python batch_experiment.py로 전체 그리드 실험 실행\n")

    return 0


if __name__ == "__main__":
    exit(main())

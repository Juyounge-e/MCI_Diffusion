#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
그리드 메타데이터 시각화

grid_metadata.csv를 읽어서 대전 그리드를 지도 위에 표시
"""

import sys
import pandas as pd

# matplotlib 백엔드 설정 (GUI 없이)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Windows 콘솔 UTF-8 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def visualize_grid(csv_path='grid_metadata.csv', output_path='grid_visualization.png'):
    """
    그리드 메타데이터를 시각화

    Args:
        csv_path: grid_metadata.csv 파일 경로
        output_path: 출력 이미지 경로
    """
    print(f"{'='*70}")
    print(f"📊 대전 그리드 시각화")
    print(f"{'='*70}")

    # CSV 로드
    print(f"\n📂 CSV 로드 중: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   ✅ {len(df)}개 그리드 로드 완료")

    # 통계 출력
    print(f"\n📊 그리드 통계:")
    print(f"   - 총 그리드 수: {len(df)}")
    print(f"   - 위도 범위: {df['latitude'].min():.6f} ~ {df['latitude'].max():.6f}")
    print(f"   - 경도 범위: {df['longitude'].min():.6f} ~ {df['longitude'].max():.6f}")

    # 그리드 크기 계산
    avg_lat_span = (df['bbox_maxlat'] - df['bbox_minlat']).mean()
    avg_lon_span = (df['bbox_maxlon'] - df['bbox_minlon']).mean()
    print(f"   - 평균 그리드 크기: {avg_lat_span:.6f}° × {avg_lon_span:.6f}°")

    # 시각화
    print(f"\n🎨 시각화 생성 중...")

    fig, ax = plt.subplots(figsize=(12, 14))

    # 각 그리드를 사각형으로 그리기
    for idx, row in df.iterrows():
        # 바운딩 박스 좌표
        minlon = row['bbox_minlon']
        minlat = row['bbox_minlat']
        width = row['bbox_maxlon'] - row['bbox_minlon']
        height = row['bbox_maxlat'] - row['bbox_minlat']

        # 사각형 그리기 (경계만)
        rect = patches.Rectangle(
            (minlon, minlat), width, height,
            linewidth=0.5,
            edgecolor='blue',
            facecolor='lightblue',
            alpha=0.3
        )
        ax.add_patch(rect)

        # 중심점 표시
        ax.plot(row['longitude'], row['latitude'], 'r.', markersize=1, alpha=0.5)

    # 축 설정
    ax.set_xlabel('경도 (Longitude)', fontsize=12, fontproperties='Malgun Gothic')
    ax.set_ylabel('위도 (Latitude)', fontsize=12, fontproperties='Malgun Gothic')
    ax.set_title(f'대전광역시 500m×500m 그리드 ({len(df)}개)',
                 fontsize=14, fontweight='bold', fontproperties='Malgun Gothic')

    # 그리드 표시
    ax.grid(True, alpha=0.3)

    # 축 범위 설정 (여백 추가)
    margin = 0.01
    ax.set_xlim(df['bbox_minlon'].min() - margin, df['bbox_maxlon'].max() + margin)
    ax.set_ylim(df['bbox_minlat'].min() - margin, df['bbox_maxlat'].max() + margin)

    # 종횡비 동일하게
    ax.set_aspect('equal')

    # 레전드 추가
    from matplotlib.lines import Line2D
    legend_elements = [
        patches.Patch(facecolor='lightblue', edgecolor='blue', alpha=0.3, label='그리드 영역'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r',
               markersize=5, alpha=0.5, label='그리드 중심점')
    ]
    ax.legend(handles=legend_elements, loc='upper right', prop={'family': 'Malgun Gothic'})

    # 그리드 ID 일부 표시 (너무 많으면 생략)
    if len(df) <= 100:
        for idx, row in df.iterrows():
            ax.text(row['longitude'], row['latitude'], str(row['grid_id']),
                   fontsize=6, ha='center', va='center', alpha=0.6)

    plt.tight_layout()

    # 저장
    print(f"\n💾 이미지 저장 중: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장 완료")

    # 통계 이미지 추가 생성
    print(f"\n📈 통계 차트 생성 중...")

    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 위도 분포
    axes[0, 0].hist(df['latitude'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('위도 (Latitude)', fontproperties='Malgun Gothic')
    axes[0, 0].set_ylabel('그리드 수', fontproperties='Malgun Gothic')
    axes[0, 0].set_title('위도 분포', fontproperties='Malgun Gothic')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 경도 분포
    axes[0, 1].hist(df['longitude'], bins=30, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('경도 (Longitude)', fontproperties='Malgun Gothic')
    axes[0, 1].set_ylabel('그리드 수', fontproperties='Malgun Gothic')
    axes[0, 1].set_title('경도 분포', fontproperties='Malgun Gothic')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 2D 히스토그램 (밀도)
    axes[1, 0].hist2d(df['longitude'], df['latitude'], bins=20, cmap='YlOrRd')
    axes[1, 0].set_xlabel('경도 (Longitude)', fontproperties='Malgun Gothic')
    axes[1, 0].set_ylabel('위도 (Latitude)', fontproperties='Malgun Gothic')
    axes[1, 0].set_title('그리드 밀도', fontproperties='Malgun Gothic')
    axes[1, 0].set_aspect('equal')

    # 4. 그리드 ID 분포
    axes[1, 1].scatter(df['grid_id'], df['latitude'], alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Grid ID', fontproperties='Malgun Gothic')
    axes[1, 1].set_ylabel('위도 (Latitude)', fontproperties='Malgun Gothic')
    axes[1, 1].set_title('Grid ID vs 위도', fontproperties='Malgun Gothic')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    stats_output = output_path.replace('.png', '_stats.png')
    print(f"\n💾 통계 이미지 저장 중: {stats_output}")
    plt.savefig(stats_output, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장 완료")

    print(f"\n{'='*70}")
    print(f"✅ 시각화 완료!")
    print(f"{'='*70}")
    print(f"\n생성된 파일:")
    print(f"1. {output_path} - 그리드 지도")
    print(f"2. {stats_output} - 통계 차트")
    print(f"\n이미지 파일을 열어서 확인하세요.\n")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='그리드 메타데이터 시각화')
    parser.add_argument('--csv', default='grid_metadata.csv', help='grid_metadata.csv 파일 경로')
    parser.add_argument('--output', default='grid_visualization.png', help='출력 이미지 경로')
    args = parser.parse_args()

    try:
        visualize_grid(args.csv, args.output)
    except FileNotFoundError:
        print(f"\n❌ 오류: {args.csv} 파일을 찾을 수 없습니다.")
        print(f"   현재 디렉토리에 grid_metadata.csv가 있는지 확인하세요.")
        return 1
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

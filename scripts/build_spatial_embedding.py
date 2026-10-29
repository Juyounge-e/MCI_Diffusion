import argparse
import time
import numpy as np
import geopandas as gpd
import shapely

def build_mask(shp_path: str, resolution: float) -> dict:
    # ── 1) SHP 로드 ──────────────────────────────────────
    print(f"[1/4] SHP 파일 로드 중 {shp_path}", flush=True)
    t0 = time.time()
    gdf = gpd.read_file(shp_path)
    print(f"      로드 완료 ({time.time()-t0:.1f}s)  행={len(gdf)}  CRS={gdf.crs}", flush=True)

    # ── 2) 좌표계 변환 ───────────────────────────────────
    print("[2/4] 좌표계 변환 중 EPSG:5179 → EPSG:4326 (WGS84)", flush=True)
    t1 = time.time()
    gdf = gdf.set_crs(epsg=5179).to_crs(epsg=4326)
    boundary = gdf.union_all()
    lon_min, lat_min, lon_max, lat_max = boundary.bounds
    print(
        f"      변환 완료 ({time.time()-t1:.1f}s)\n"
        f"      lat: [{lat_min:.4f}, {lat_max:.4f}]  "
        f"lon: [{lon_min:.4f}, {lon_max:.4f}]",
        flush=True,
    )

    # ── 3) 격자 생성 ─────────────────────────────────────
    lat_grid = np.arange(lat_min, lat_max, resolution)
    lon_grid = np.arange(lon_min, lon_max, resolution)
    xx, yy   = np.meshgrid(lon_grid, lat_grid)
    n_cells  = xx.size
    print(
        f"[3/4] 이진 마스크 계산 중 \n"
        f"      격자 크기: {xx.shape}  총 셀 수: {n_cells:,}  해상도: {resolution}°",
        flush=True,
    )
    t2 = time.time()
    mask = (
        shapely.contains_xy(boundary, xx.ravel(), yy.ravel())
        .reshape(xx.shape)
        .astype(np.float32)
    )
    land_ratio = mask.mean()
    print(
        f"      계산 완료 ({time.time()-t2:.1f}s)\n"
        f"      육지 셀: {int(mask.sum()):,} / {n_cells:,}  (비율: {land_ratio:.3f})",
        flush=True,
    )

    return {
        "mask":       mask,
        "lat_min":    np.float64(lat_min),
        "lon_min":    np.float64(lon_min),
        "resolution": np.float64(resolution),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shp_path",   default="MCI_ADV2/scenarios/ctprvn.shp", help="시도 경계 SHP 파일 경로")
    parser.add_argument("--out_path",   default="MCI_ADV2/scenarios/mask_cache_0.005.npz", help="출력 npz 경로")
    parser.add_argument("--resolution", type=float, default=0.0005, help="격자 해상도 (도, 0.01 ≈ 1.1km)")
    args = parser.parse_args()

    print("=" * 50, flush=True)
    print("  마스크 생성 시작", flush=True)
    print(f"  shp_path   : {args.shp_path}", flush=True)
    print(f"  out_path   : {args.out_path}", flush=True)
    print(f"  resolution : {args.resolution}°", flush=True)
    print("=" * 50, flush=True)

    t_total = time.time()
    data = build_mask(args.shp_path, args.resolution)

    # ── 4) 저장 ──────────────────────────────────────────
    print(f"[4/4] npz 저장 중 {args.out_path}", flush=True)
    np.savez(args.out_path, **data)
    print(
        f"      저장 완료\n"
        f"\n총 소요 시간: {time.time()-t_total:.1f}s",
        flush=True,
    )
    print("=" * 50, flush=True)
    print("  완료!  학습 시 --mask_path 인수로 위 경로를 지정", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()

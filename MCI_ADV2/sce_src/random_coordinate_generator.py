#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OSRM + Overpass를 이용한 랜덤 좌표 생성기.
행정구역 폴리곤 내부에서 균일분포로 좌표를 뽑고, 도로에 스냅된 좌표를 함께 저장한다.
"""

import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import geopandas as gpd
import pandas as pd
import requests
from haversine import Unit, haversine
from shapely.geometry import Point

DEFAULT_OSRM = "https://router.project-osrm.org"
DEFAULT_OVERPASS = "https://overpass-api.de/api/interpreter"
BASE_DIR = Path(__file__).resolve().parents[1]

# 17개 시도 매핑 (영문/코드 기준, 한글 매칭은 shapefile 컬럼에서 처리)
REGION_MAP: Dict[str, Dict[str, str]] = {
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


def _slugify(text: str, maxlen: int = 40) -> str:
    safe = "".join(ch.lower() if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))
    safe = "_".join(part for part in safe.split("_") if part)
    return (safe[:maxlen] or "region").strip("_")


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return float(haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS))


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BASE_DIR / path


def load_custom_coordinate_csv(csv_path: str) -> pd.DataFrame:
    input_path = _resolve_path(csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"custom CSV 파일을 찾을 수 없습니다: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"custom CSV가 비어 있습니다: {input_path}")

    col_map = {str(col).strip().lower(): col for col in df.columns}
    lat_col = col_map.get("lat")
    lon_col = col_map.get("lon")
    n_col = col_map.get("n")

    missing = []
    if lat_col is None:
        missing.append("lat")
    if lon_col is None:
        missing.append("lon")
    if n_col is None:
        missing.append("N")
    if missing:
        raise ValueError(
            f"custom CSV에는 다음 컬럼이 필요합니다: lat, lon, N (누락: {', '.join(missing)})"
        )

    out_df = df[[lat_col, lon_col, n_col]].copy()
    out_df.columns = ["lat", "lon", "N"]
    out_df["lat"] = pd.to_numeric(out_df["lat"], errors="coerce")
    out_df["lon"] = pd.to_numeric(out_df["lon"], errors="coerce")
    out_df["N"] = pd.to_numeric(out_df["N"], errors="coerce")

    invalid_mask = out_df[["lat", "lon", "N"]].isna().any(axis=1)
    if invalid_mask.any():
        invalid_lines = (out_df.index[invalid_mask] + 2).tolist()  # CSV line number (header=1)
        preview = ", ".join(str(v) for v in invalid_lines[:10])
        suffix = " ..." if len(invalid_lines) > 10 else ""
        raise ValueError(
            "custom CSV에 숫자로 해석할 수 없는 값이 있습니다. "
            f"(line: {preview}{suffix})"
        )

    out_df["N"] = out_df["N"].astype(int)
    return out_df


def _osrm_nearest(
    session: requests.Session,
    osrm_base: str,
    lon: float,
    lat: float,
    profile: str,
    k: int,
    radius_m: float,
    timeout_s: int = 15,
) -> Dict[str, Any]:
    osrm_base = osrm_base.rstrip("/")
    url = f"{osrm_base}/nearest/v1/{profile}/{lon},{lat}"
    params = {
        "number": str(max(1, k)),
        "radiuses": str(max(0.0, radius_m)),
    }
    resp = session.get(url, params=params, timeout=timeout_s)
    try:
        return resp.json()
    except Exception:
        resp.raise_for_status()
        raise


def _overpass_query(session: requests.Session, overpass_url: str, query: str, timeout_s: int = 60) -> Dict[str, Any]:
    resp = session.post(overpass_url, data={"data": query}, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _build_overpass_parent_ways_query(node_ids: List[int]) -> str:
    ids_csv = ",".join(str(n) for n in node_ids)
    return (
        "[out:json][timeout:25];\n"
        f"node(id:{ids_csv});\n"
        "way(bn)[\"highway\"];\n"
        "out tags center;"
    )


def _build_overpass_around_ways_query(lat: float, lon: float, radius_m: float) -> str:
    return (
        "[out:json][timeout:25];\n"
        f"way(around:{radius_m},{lat},{lon})[\"highway\"];\n"
        "out tags center;"
    )


def _find_region_row(gdf: gpd.GeoDataFrame, region_keyword: str) -> Tuple[Any, str]:
    keyword = str(region_keyword or "").strip()
    if not keyword:
        keyword = "daejeon"
    keyword_lower = keyword.lower()

    region_key = None
    region_info = None
    for key, info in REGION_MAP.items():
        if keyword_lower in (key, info["eng"].lower(), info["code"], info["kor"]):
            region_key = key
            region_info = info
            break

    columns = list(gdf.columns)
    row = None

    if region_info is not None:
        for col in ("CTPRVN_CD", "CTP_ENG_NM", "CTP_KOR_NM"):
            if col in columns:
                match = gdf[
                    (gdf[col].astype(str) == region_info["code"])
                    | (gdf[col].astype(str).str.lower() == region_info["eng"].lower())
                    | (gdf[col].astype(str) == region_info["kor"])
                ]
                if not match.empty:
                    row = match.iloc[0]
                    break

    if row is None:
        for col in columns:
            if gdf[col].dtype == "object":
                values = gdf[col].astype(str)
                match = values.str.lower() == keyword_lower
                if match.any():
                    row = gdf[match].iloc[0]
                    break

    if row is None:
        raise ValueError(f"shapefile에서 지역을 찾지 못했습니다: '{region_keyword}'")

    if region_key is None:
        if "CTP_ENG_NM" in columns:
            region_key = _slugify(row["CTP_ENG_NM"])
        elif "CTP_KOR_NM" in columns:
            region_key = _slugify(row["CTP_KOR_NM"])
        else:
            region_key = _slugify(region_keyword)

    return row, region_key


def load_region_boundary(shp_path: str, region_keyword: str) -> Tuple[Any, str]:
    gdf = gpd.read_file(shp_path, encoding="cp949")
    row, region_key = _find_region_row(gdf, region_keyword)

    if gdf.crs is None:
        gdf.set_crs("EPSG:5179", allow_override=True, inplace=True)

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    region_geom = gdf[gdf.index == row.name]["geometry"].iloc[0]
    return region_geom, region_key


class RandomCoordinateGenerator:
    def __init__(
        self,
        shp_path: Optional[str],
        region: str = "daejeon",
        osrm_base: str = DEFAULT_OSRM,
        overpass_url: Optional[str] = None,
        profile: str = "driving",
        radius_m: float = 250.0,
        incident_size_min: int = 10,
        incident_size_max: int = 50,
        seed: Optional[int] = None,
        rate_limit_delay: float = 0.3,
        allow_overpass_failure: bool = True,
    ) -> None:
        self.shp_path = shp_path or ""
        self.osrm_base = osrm_base
        self.overpass_url = overpass_url or ""
        self.profile = profile
        self.radius_m = radius_m
        self.incident_size_min = int(incident_size_min)
        self.incident_size_max = int(incident_size_max)
        if self.incident_size_min > self.incident_size_max:
            raise ValueError("incident_size_min은 incident_size_max보다 클 수 없습니다.")
        self.rate_limit_delay = rate_limit_delay
        self.allow_overpass_failure = allow_overpass_failure
        self._rng = random.Random(seed)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "mci-random-generator/1.0"})

        self._region_key = None
        self._region_geom = None
        if self.shp_path:
            self.set_region(region)

    def set_region(self, region_keyword: str) -> None:
        if not self.shp_path:
            raise ValueError("랜덤 모드를 사용하려면 shapefile 경로(--shp)가 필요합니다.")
        self._region_geom, self._region_key = load_region_boundary(self.shp_path, region_keyword)

    @property
    def region_key(self) -> str:
        return self._region_key

    def _random_point_in_region(self, max_attempts: int = 10000) -> Tuple[float, float]:
        minx, miny, maxx, maxy = self._region_geom.bounds
        for _ in range(max_attempts):
            lon = self._rng.uniform(minx, maxx)
            lat = self._rng.uniform(miny, maxy)
            if self._region_geom.contains(Point(lon, lat)):
                return lat, lon
        raise RuntimeError("행정구역 폴리곤 내부에서 좌표 샘플링에 실패했습니다.")

    def _sample_incident_size(self) -> int:
        return int(self._rng.randint(self.incident_size_min, self.incident_size_max))

    @staticmethod
    def _snapped_key(lat: float, lon: float, precision: int = 7) -> Tuple[float, float]:
        return (round(float(lat), precision), round(float(lon), precision))

    def _overpass_has_way(self, node_ids: List[int], lat: float, lon: float) -> Optional[bool]:
        if not self.overpass_url:
            return True

        try:
            if node_ids:
                q1 = _build_overpass_parent_ways_query(node_ids)
                data = _overpass_query(self._session, self.overpass_url, q1)
                for el in data.get("elements", []) or []:
                    if el.get("type") == "way":
                        return True
            q2 = _build_overpass_around_ways_query(lat, lon, self.radius_m)
            data2 = _overpass_query(self._session, self.overpass_url, q2)
            for el in data2.get("elements", []) or []:
                if el.get("type") == "way":
                    return True
            return False
        except Exception:
            return None

    def snap_point(self, lat: float, lon: float) -> Optional[Tuple[float, float, float]]:
        time.sleep(self.rate_limit_delay)
        js = _osrm_nearest(
            session=self._session,
            osrm_base=self.osrm_base,
            lon=lon,
            lat=lat,
            profile=self.profile,
            k=1,
            radius_m=self.radius_m,
        )
        if js.get("code") != "Ok":
            return None
        waypoints = js.get("waypoints", []) or []
        if not waypoints:
            return None

        best = waypoints[0]
        distance_m = float(best.get("distance", 1e18))
        if distance_m > self.radius_m:
            return None

        loc = best.get("location", [None, None])
        snapped_lon = loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else None
        snapped_lat = loc[1] if isinstance(loc, (list, tuple)) and len(loc) > 1 else None
        if snapped_lat is None or snapped_lon is None:
            return None

        nodes = best.get("nodes", []) or []
        node_ids = [int(n) for n in nodes if isinstance(n, int) and n > 0]

        overpass_ok = self._overpass_has_way(node_ids, float(snapped_lat), float(snapped_lon))
        if overpass_ok is False:
            return None
        if overpass_ok is None and not self.allow_overpass_failure:
            return None

        distance_haversine = _haversine_m(lat, lon, float(snapped_lat), float(snapped_lon))
        return float(snapped_lat), float(snapped_lon), float(distance_haversine)

    def generate_valid_coordinate(
        self,
        mode: str = "korea_random",
        sido_name: Optional[str] = None,
        max_attempts: int = 1000,
    ) -> Optional[Tuple[float, float, Dict[str, Any]]]:
        region_keyword = "daejeon"
        if mode == "sido" and sido_name:
            region_keyword = sido_name

        if self._region_key is None or region_keyword.lower() != self._region_key.lower():
            self.set_region(region_keyword)

        for _ in range(max_attempts):
            rand_lat, rand_lon = self._random_point_in_region()
            snapped = self.snap_point(rand_lat, rand_lon)
            if not snapped:
                continue
            snapped_lat, snapped_lon, dist_m = snapped
            info = {
                "random_latitude": rand_lat,
                "random_longitude": rand_lon,
                "snapped_latitude": snapped_lat,
                "snapped_longitude": snapped_lon,
                "snap_distance_m": dist_m,
                "N": self._sample_incident_size(),
                "region": self._region_key,
                "is_valid": True,
            }
            return snapped_lat, snapped_lon, info
        return None

    def generate_snapped_points(
        self,
        target_count: int,
        max_attempts: int = 100000,
        progress_every: int = 50,
        save_path: Optional[Path] = None,
        save_every: int = 50,
    ) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        attempts = 0
        start_ts = time.time()

        def _build_point(rand_lat: float, rand_lon: float, snapped_lat: float, snapped_lon: float, dist_m: float) -> Dict[str, Any]:
            return {
                "index": len(points) + 1,
                "random_latitude": rand_lat,
                "random_longitude": rand_lon,
                "snapped_latitude": snapped_lat,
                "snapped_longitude": snapped_lon,
                "snap_distance_m": dist_m,
                "N": self._sample_incident_size(),
            }

        while len(points) < target_count and attempts < max_attempts:
            attempts += 1
            rand_lat, rand_lon = self._random_point_in_region()
            snapped = self.snap_point(rand_lat, rand_lon)
            if not snapped:
                continue
            snapped_lat, snapped_lon, dist_m = snapped
            points.append(_build_point(rand_lat, rand_lon, snapped_lat, snapped_lon, dist_m))
            if progress_every and len(points) % progress_every == 0:
                elapsed = time.time() - start_ts
                success_rate = (len(points) / attempts * 100.0) if attempts else 0.0
                print(
                    f"[진행] {len(points)}/{target_count}개 생성 "
                    f"(시도 {attempts}회, 성공률 {success_rate:.1f}%, 경과 {elapsed:.1f}s)"
                )

            if save_path and save_every and len(points) % save_every == 0:
                export_random_metadata(points, Path(save_path))
                print(f"[저장] 중간 저장 완료: {save_path} ({len(points)}개)")
        if len(points) < target_count:
            raise RuntimeError(
                f"스냅 좌표 {target_count}개 생성 실패 (성공 {len(points)}개, 시도 {attempts}회)"
            )

        print("[검토] 스냅 좌표 중복 검사 시작...")
        seen_keys: Set[Tuple[float, float]] = set()
        dedup_points: List[Dict[str, Any]] = []
        duplicate_count = 0
        for point in points:
            key = self._snapped_key(point["snapped_latitude"], point["snapped_longitude"])
            if key in seen_keys:
                duplicate_count += 1
                continue
            seen_keys.add(key)
            dedup_points.append(point)
        points = dedup_points

        if duplicate_count:
            print(f"[검토] 스냅 좌표 중복 {duplicate_count}건 발견 -> 후순위 데이터 삭제 후 재생성")
            while len(points) < target_count and attempts < max_attempts:
                attempts += 1
                rand_lat, rand_lon = self._random_point_in_region()
                snapped = self.snap_point(rand_lat, rand_lon)
                if not snapped:
                    continue
                snapped_lat, snapped_lon, dist_m = snapped
                key = self._snapped_key(snapped_lat, snapped_lon)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                points.append(_build_point(rand_lat, rand_lon, snapped_lat, snapped_lon, dist_m))
                if save_path and save_every and len(points) % save_every == 0:
                    export_random_metadata(points, Path(save_path))
                    print(f"[저장] 중간 저장 완료: {save_path} ({len(points)}개)")

            if len(points) < target_count:
                raise RuntimeError(
                    f"중복 제거 후 스냅 좌표 재생성 실패 (필요 {target_count}, 성공 {len(points)}, 시도 {attempts}회)"
                )

        for idx, point in enumerate(points, start=1):
            point["index"] = idx

        print(
            f"[검토] 중복 검사 완료: 최종 {len(points)}개 "
            f"(중복 제거 {duplicate_count}건, 총 시도 {attempts}회)"
        )
        return points

    def generate_snapped_points_from_custom(
        self,
        custom_df: pd.DataFrame,
        progress_every: int = 50,
        save_path: Optional[Path] = None,
        save_every: int = 50,
    ) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        failed_lines: List[int] = []
        success_count = 0
        start_ts = time.time()
        total = len(custom_df)

        for idx, row in enumerate(custom_df.itertuples(index=False), start=1):
            custom_lat = float(row.lat)
            custom_lon = float(row.lon)
            incident_size = int(row.N)

            snapped = self.snap_point(custom_lat, custom_lon)
            if not snapped:
                failed_lines.append(idx + 1)  # CSV line number (header=1)
                points.append(
                    {
                        "custom_latitude": custom_lat,
                        "custom_longitude": custom_lon,
                        "snapped_latitude": None,
                        "snapped_longitude": None,
                        "N": incident_size,
                    }
                )
            else:
                snap_lat, snap_lon, _ = snapped
                success_count += 1
                points.append(
                    {
                        "custom_latitude": custom_lat,
                        "custom_longitude": custom_lon,
                        "snapped_latitude": float(snap_lat),
                        "snapped_longitude": float(snap_lon),
                        "N": incident_size,
                    }
                )

            if progress_every and idx % progress_every == 0:
                elapsed = time.time() - start_ts
                success_rate = (success_count / idx * 100.0) if idx else 0.0
                print(
                    f"[진행] {idx}/{total}개 처리 "
                    f"(성공 {success_count}개, 실패 {idx - success_count}개, 성공률 {success_rate:.1f}%, 경과 {elapsed:.1f}s)"
                )

            if save_path and save_every and idx % save_every == 0:
                export_custom_metadata(points, Path(save_path))
                print(f"[저장] 중간 저장 완료: {save_path} ({len(points)}개)")

        if failed_lines:
            preview = ", ".join(str(v) for v in failed_lines[:10])
            suffix = " ..." if len(failed_lines) > 10 else ""
            print(
                "[경고] custom 좌표 스냅 실패가 있습니다. "
                f"{len(failed_lines)}건 (실패 line: {preview}{suffix}) - snapped 좌표는 공백으로 저장됩니다."
            )

        return points


def export_random_metadata(points: List[Dict[str, Any]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(points)
    ordered_cols = [
        "index",
        "random_latitude",
        "random_longitude",
        "snapped_latitude",
        "snapped_longitude",
        "snap_distance_m",
        "N",
    ]
    existing_ordered = [col for col in ordered_cols if col in df.columns]
    remain_cols = [col for col in df.columns if col not in existing_ordered]
    if existing_ordered:
        df = df[existing_ordered + remain_cols]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def export_custom_metadata(points: List[Dict[str, Any]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(points)
    ordered_cols = [
        "custom_latitude",
        "custom_longitude",
        "snapped_latitude",
        "snapped_longitude",
        "N",
    ]
    existing_ordered = [col for col in ordered_cols if col in df.columns]
    remain_cols = [col for col in df.columns if col not in existing_ordered]
    if existing_ordered:
        df = df[existing_ordered + remain_cols]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="OSRM/Overpass 기반 좌표 스냅 생성기")
    parser.add_argument("--mode", choices=["random", "custom"], default="random", help="실행 모드 (random/custom)")
    parser.add_argument("--region", default="daejeon", help="지역 키워드 (기본: daejeon)")
    parser.add_argument("--exp-id", dest="exp_id", default=None, help="실험 ID (기본: exp_YYYYMMDD_HHMMSS)")
    parser.add_argument("--custom-csv", dest="custom_csv", default="", help="custom 모드 입력 CSV 경로 (lat, lon, N)")
    parser.add_argument("--num-points", "--num_points", dest="num_points", type=int, default=100, help="생성할 스냅 좌표 개수")
    parser.add_argument("--incident-size-min", "--incident_size_min", dest="incident_size_min", type=int, default=10, help="N(사고규모) 최소값 (기본: 10)")
    parser.add_argument("--incident-size-max", "--incident_size_max", dest="incident_size_max", type=int, default=50, help="N(사고규모) 최대값 (기본: 50)")
    parser.add_argument("--radius-m", type=float, default=250.0, help="OSRM 스냅 반경(미터)")
    parser.add_argument("--shp", default="scenarios/ctprvn.shp", help="Shapefile 경로")
    parser.add_argument("--osrm", default=DEFAULT_OSRM, help="OSRM 기본 URL")
    parser.add_argument("--overpass", default="", help="Overpass API URL (빈 문자열이면 비활성화)")
    parser.add_argument("--profile", default="driving", help="OSRM 프로파일")
    parser.add_argument("--seed", type=int, default=None, help="난수 시드")
    parser.add_argument("--max-attempts", type=int, default=2000, help="최대 시도 횟수")
    parser.add_argument("--rate-limit-delay", type=float, default=0.3, help="OSRM 호출 간 지연(초)")
    parser.add_argument("--progress-every", type=int, default=50, help="진행 로그 출력 간격(생성 개수 기준)")
    parser.add_argument("--save-every", type=int, default=50, help="중간 저장 간격(생성 개수 기준, 0이면 비활성)")
    args = parser.parse_args()

    exp_id = args.exp_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.mode == "custom":
        if not args.custom_csv:
            raise ValueError("custom 모드에서는 --custom-csv를 반드시 지정해야 합니다.")

        print("=" * 70)
        print("커스텀 좌표 스냅 생성")
        print("=" * 70)
        print(f"실험 ID: {exp_id}")
        print(f"입력 CSV: {str(_resolve_path(args.custom_csv))}")
        print(f"스냅 반경: {args.radius_m} m")

        custom_df = load_custom_coordinate_csv(args.custom_csv)
        print(f"입력 좌표 개수: {len(custom_df)}")

        generator = RandomCoordinateGenerator(
            shp_path=None,
            region=args.region,
            osrm_base=args.osrm,
            overpass_url=args.overpass,
            profile=args.profile,
            radius_m=args.radius_m,
            incident_size_min=args.incident_size_min,
            incident_size_max=args.incident_size_max,
            seed=args.seed,
            rate_limit_delay=args.rate_limit_delay,
        )

        output_dir = BASE_DIR / "scenarios" / f"custom_{exp_id}"
        output_path = output_dir / "custom_metadata.csv"
        points = generator.generate_snapped_points_from_custom(
            custom_df,
            progress_every=args.progress_every,
            save_path=output_path,
            save_every=args.save_every,
        )
        export_custom_metadata(points, output_path)

        df = pd.DataFrame(points)
        snap_lat_series = pd.to_numeric(df["snapped_latitude"], errors="coerce")
        snap_lon_series = pd.to_numeric(df["snapped_longitude"], errors="coerce")
        success_mask = snap_lat_series.notna() & snap_lon_series.notna()
        success_count = int(success_mask.sum())
        failed_count = int(len(df) - success_count)

        print("\n요약")
        print(f"- 생성 개수: {len(df)}")
        print(f"- 스냅 성공: {success_count}개")
        print(f"- 스냅 실패: {failed_count}개")
        print(f"- custom 위도 범위: {df['custom_latitude'].min():.6f} ~ {df['custom_latitude'].max():.6f}")
        print(f"- custom 경도 범위: {df['custom_longitude'].min():.6f} ~ {df['custom_longitude'].max():.6f}")
        if success_count > 0:
            print(f"- snap 위도 범위: {snap_lat_series.min():.6f} ~ {snap_lat_series.max():.6f}")
            print(f"- snap 경도 범위: {snap_lon_series.min():.6f} ~ {snap_lon_series.max():.6f}")
        else:
            print("- snap 위도 범위: N/A (모든 좌표 스냅 실패)")
            print("- snap 경도 범위: N/A (모든 좌표 스냅 실패)")
        print(f"- 사고 규모 N: min={int(df['N'].min())}, max={int(df['N'].max())}, mean={df['N'].mean():.2f}")

        print("\n출력:")
        print(f"- {output_path}")
        print("\n다음 단계:")
        print(
            f"1. python MCI_ADV2\\vis_src\\custom_visualize.py --csv {output_path} "
            f"--output {output_dir / 'custom_visualization.png'} --shp {str(_resolve_path(args.shp))} --region {args.region}"
        )
        print(
            f"2. python MCI_ADV2\\sce_src\\batch_experiment.py --base_path {BASE_DIR} "
            f"--random_metadata {output_path} --config_template {BASE_DIR / 'sim_src' / 'config.yaml'}"
        )
        return 0

    print("=" * 70)
    print("랜덤 좌표 생성")
    print("=" * 70)
    print(f"지역: {args.region}")
    print(f"실험 ID: {exp_id}")
    print(f"목표 개수: {args.num_points}")
    print(f"N 범위: {args.incident_size_min} ~ {args.incident_size_max}")
    print(f"스냅 반경: {args.radius_m} m")

    shp_path = str(_resolve_path(args.shp))
    generator = RandomCoordinateGenerator(
        shp_path=shp_path,
        region=args.region,
        osrm_base=args.osrm,
        overpass_url=args.overpass,
        profile=args.profile,
        radius_m=args.radius_m,
        incident_size_min=args.incident_size_min,
        incident_size_max=args.incident_size_max,
        seed=args.seed,
        rate_limit_delay=args.rate_limit_delay,
    )

    output_dir = BASE_DIR / "scenarios" / f"{generator.region_key}_{exp_id}"
    output_path = output_dir / "random_metadata.csv"
    points = generator.generate_snapped_points(
        args.num_points,
        max_attempts=args.max_attempts,
        progress_every=args.progress_every,
        save_path=output_path,
        save_every=args.save_every,
    )
    export_random_metadata(points, output_path)

    df = pd.DataFrame(points)
    print("\n요약")
    print(f"- 생성 개수: {len(df)}")
    print(f"- 랜덤 위도 범위: {df['random_latitude'].min():.6f} ~ {df['random_latitude'].max():.6f}")
    print(f"- 랜덤 경도 범위: {df['random_longitude'].min():.6f} ~ {df['random_longitude'].max():.6f}")
    print(f"- 스냅 위도 범위: {df['snapped_latitude'].min():.6f} ~ {df['snapped_latitude'].max():.6f}")
    print(f"- 스냅 경도 범위: {df['snapped_longitude'].min():.6f} ~ {df['snapped_longitude'].max():.6f}")
    print(
        f"- 스냅 거리(m): min={df['snap_distance_m'].min():.2f}, "
        f"max={df['snap_distance_m'].max():.2f}, mean={df['snap_distance_m'].mean():.2f}"
    )
    if "N" in df.columns:
        print(f"- 사고 규모 N: min={int(df['N'].min())}, max={int(df['N'].max())}, mean={df['N'].mean():.2f}")

    print("\n출력:")
    print(f"- {output_path}")
    print("\n다음 단계:")
    print(
        f"1. python MCI_ADV2\\vis_src\\visualize_random_points.py --csv {output_path} "
        f"--output {output_dir / 'random_visualization.png'} --shp {shp_path} --region {args.region}"
    )
    print(
        f"2. python MCI_ADV2\\sce_src\\batch_experiment.py --base_path {BASE_DIR} "
        f"--random_metadata {output_path} --config_template {BASE_DIR / 'sim_src' / 'config.yaml'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

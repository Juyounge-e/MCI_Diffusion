# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import yaml
import argparse
import pandas as pd
import numpy as np
import requests
from haversine import haversine
# [ADD] ──────────────────────────────────────────────────────────────
import re
from datetime import timezone, timedelta, datetime
from typing import Optional

KST = timezone(timedelta(hours=9))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def slugify(name: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^\w\-\s]", "", str(name))
    s = re.sub(r"\s+", "_", s).strip("_")
    return (s[:maxlen] or "noname")

def save_route_json(meta: dict, payload: Optional[dict], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    data = {"meta": meta, "payload": {"naver_response": payload} if payload else None}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
# [ADD END] ─────────────────────────────────────────────────────────


def parse_util_map(text: str):
    """
    "1:0.90,11:0.75,etc:0.60" -> {1:0.9, 11:0.75, "etc":0.6}
    """
    if not text:
        return None
    m = {}
    for part in str(text).split(","):
        if not part.strip():
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            val = float(v)
        except Exception:
            continue
        if k.lower() == "etc":
            m["etc"] = val
        else:
            try:
                m[int(k)] = val
            except Exception:
                pass
    return m if m else None

class ScenarioGenerator:
    """동적 파라미터 기반 시나리오 생성 클래스 (크로스 환경 호환)"""

    def __init__(self, base_path, experiment_id=None, kakao_api_key=None, departure_time=None):
        # 프로젝트 경로 절대화
        self.base_path = os.path.abspath(base_path)

        # experiment_id 생성: exp_YYYYMMDD_HHMMSS 형식 (통일)
        if experiment_id:
            # 이미 exp_ 접두사가 있으면 그대로 사용
            self.experiment_id = experiment_id if experiment_id.startswith("exp_") else f"exp_{experiment_id}"
        else:
            # 기본 형식: exp_YYYYMMDD_HHMMSS
            self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 카카오 API 키 설정
        self.kakao_api_key = kakao_api_key
        self.departure_time = departure_time  # YYYYMMDDHHMM 형식
        self.api_call_count = 0
        
        # 데이터 파일 경로들 (절대경로로 설정)
        self.scenarios_path = os.path.join(self.base_path, "scenarios")
        self.fire_data_path = os.path.join(self.scenarios_path, "안전센터와 소방서.csv")
        self.hospital_data_path = os.path.join(self.scenarios_path, "엑셀 결합 데이터.xlsx")
        self.shp_path = os.path.join(self.scenarios_path, "ctprvn.shp")
        
        # 파일 존재성 검증
        self._validate_data_files()

        # Patient 정보 (하드코딩)
        self.patient_config = {
            "ratio": {"Red": 0.1, "Yellow": 0.3, "Green": 0.5, "Black": 0.1},
            "rescue_param": {"Red": (6, 5), "Yellow": (2, 13), "Green": (1, 22), "Black": (0, 0)},
            "treat_tier1": {"Red": True, "Yellow": True, "Green": True, "Black": True},
            "treat_tier2": {"Red": False, "Yellow": True, "Green": True, "Black": True},
            "treat_tier1_mean": {"Red": 40, "Yellow": 20, "Green": 10, "Black": 0},
            "treat_tier2_mean": {"Red": 60, "Yellow": 30, "Green": 15, "Black": 0}
        }
        
        # 후보군 확장 배수 (AMB road distance 호출 수 완화)
        self.multiplier = 1.5

        # --- ENV 주입(PS에서 전달) ---
        # util_by_tier: 예) "1:0.656,11:0.461,etc:0.461"
        env_util = parse_util_map(os.environ.get("MCI_UTIL_BY_TIER", ""))
        self.util_by_tier = env_util or {1: 0.656, 11: 0.461, "etc": 0.461}

        # queue_policy: "0" | "capa/2" | "0.5" 등
        # self.queue_policy = os.environ.get("MCI_QUEUE_POLICY", "0")

        # buffer_ratio: float
        try:
            self.buffer_ratio = float(os.environ.get("MCI_BUFFER_RATIO", "1.5"))
        except Exception:
            self.buffer_ratio = 1.5
        
        # (추가) max_send_coeff 기본 입력경로: ENV → 기본값
        self.max_send_coeff_text = os.environ.get("MCI_MAX_SEND_COEFF", "1,1")
        
        print(f"📁 프로젝트 경로: {self.base_path}")
        print(f"🆔 실험 ID: {self.experiment_id}")
        print(f"buffer_ratio={self.buffer_ratio}")

    def _validate_data_files(self):
        """필수 데이터 파일들의 존재성 검증"""
        required_files = [
            (self.fire_data_path, "소방서 데이터"),
            (self.hospital_data_path, "병원 데이터"),
            (self.shp_path, "시도 경계 SHP 파일")
        ]
        missing_files = []
        for file_path, description in required_files:
            if not os.path.exists(file_path):
                missing_files.append(f"{description}: {file_path}")
        if missing_files:
            print("❌ 다음 필수 파일들이 없습니다:")
            for missing in missing_files:
                print(f"   • {missing}")
            raise FileNotFoundError("필수 데이터 파일들을 확인해주세요.")
        print("✅ 모든 필수 데이터 파일 확인 완료")

    def get_road_distance_kakao(self, start, end, max_retries=3, save_json_dir=None, route_type=None, source_index=None, name=None, start_label="start", goal_label="goal"):
        """카카오 모빌리티 API를 사용한 도로 거리 및 시간 계산 (재시도 로직 포함)

        Args:
            start: (lat, lon) 튜플
            end: (lat, lon) 튜플
            save_json_dir: JSON 저장 디렉토리
            route_type: "center2site" 또는 "hos2site"

        Returns:
            (distance_km, duration_min) 튜플 - 거리(km)와 이송시간(분)
        """
        if not self.kakao_api_key:
            # API 키 없으면 유클리드 거리 + 추정 시간 반환
            dist_km = haversine(start, end)
            estimated_duration_min = (dist_km / 40) * 60  # 40km/h 가정
            return dist_km, estimated_duration_min

        url = "https://apis-navi.kakaomobility.com/v1/future/directions"
        headers = {
            "Authorization": f"KakaoAK {self.kakao_api_key}",
            "Content-Type": "application/json"
        }
        params = {
            "origin": f"{start[1]},{start[0]}",  # lon,lat 순서
            "destination": f"{end[1]},{end[0]}",
            "priority": "TIME",  # 최단시간 우선
            "car_fuel": "GASOLINE",
            "car_hipass": "false",
            "alternatives": "false",
            "road_details": "false"
        }

        # departure_time 파라미터 추가 (실시간 또는 미래시간)
        if self.departure_time:
            params["departure_time"] = self.departure_time

        fallback_params = None
        if "car_type" in params or "road_details" in params:
            fallback_params = params.copy()
            fallback_params.pop("car_type", None)
            fallback_params.pop("road_details", None)

        for attempt in range(max_retries):
            try:
                self.api_call_count += 1
                response = requests.get(url, headers=headers, params=params, timeout=15)
                params_used = params
                if response.status_code == 400 and fallback_params:
                    self.api_call_count += 1
                    response = requests.get(url, headers=headers, params=fallback_params, timeout=15)
                    params_used = fallback_params
                if response.status_code == 200:
                    data = response.json()

                    # 카카오 API 응답 구조: routes[0].summary
                    if not data.get("routes") or len(data["routes"]) == 0:
                        print(f"  ⚠️ 카카오 API 응답에 경로 정보가 없습니다.")
                        break

                    route = data["routes"][0]
                    summary = route.get("summary", {})

                    # 거리(m) → km 변환
                    distance_km = summary.get("distance", 0) / 1000.0

                    # 시간(초) → 분 변환
                    duration_sec = summary.get("duration", 0)
                    duration_min = duration_sec / 60.0

                    # JSON 저장
                    if save_json_dir:
                        now = datetime.now(KST).isoformat()
                        meta = {
                            "api_provider": "kakao",
                            "route_type": route_type,
                            "source_index": source_index,
                            "name": name,
                            # 좌표는 [lon, lat] 형식으로 저장
                            start_label: [start[1], start[0]],
                            goal_label: [end[1], end[0]],
                            "departure_time": self.departure_time or "realtime",
                            "priority": params_used.get("priority"),
                            "saved_at": now,
                            # 요약 필드
                            "distance_km": round(distance_km, 3),
                            "duration_min": round(duration_min, 2),
                            "duration_sec": duration_sec,
                            "toll_fare": summary.get("fare", {}).get("toll", 0),
                            "taxi_fare": summary.get("fare", {}).get("taxi", 0),
                            "direction_note": f"{start_label}->{goal_label}"
                        }
                        fname = f"{(source_index if source_index is not None else 0):03d}_{slugify(name)}.json"
                        out_path = os.path.join(save_json_dir, fname)

                        # 카카오 응답 저장
                        ensure_dir(os.path.dirname(out_path))
                        json_data = {
                            "meta": meta,
                            "payload": {"kakao_response": data}
                        }
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)

                        print(f"  📦 [{route_type}] idx={source_index:03d} {name} → {distance_km:.2f}km, {duration_min:.1f}min")

                    return distance_km, duration_min

                elif response.status_code == 401:
                    print(f"  ❌ 카카오 API 인증 실패 (401): API 키를 확인하세요.")
                    break
                elif response.status_code == 429:
                    print(f"  ⚠️ API 호출 한도 초과 (429): 3초 대기 중...")
                    time.sleep(3)
                else:
                    print(f"  ⚠️ API 호출 실패 (status {response.status_code})")
                    break

            except Exception as e:
                print(f"  ⚠️ API 호출 중 오류 발생: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        # API 실패 시 유클리드 거리 + 추정 시간으로 대체
        dist_km = haversine(start, end)
        estimated_duration_min = (dist_km / 40) * 60  # 40km/h 가정
        print(f"  ⚠️ API 실패, 유클리드 거리 사용: {dist_km:.2f}km")
        return dist_km, estimated_duration_min

    def get_road_distance_osrm(
        self,
        start,
        end,
        max_retries=3,
        rate_limit_delay=0.1,
        radius_meters=250,
        save_json_dir=None,
        route_type=None,
        source_index=None,
        name=None,
        start_label="start",
        goal_label="goal",
    ):
        """OSRM 공개 서버를 사용한 도로 거리 및 시간 계산

        Args:
            start: (lat, lon) 튜플
            end: (lat, lon) 튜플
            max_retries: 최대 재시도 횟수
            rate_limit_delay: 요청 간 지연 시간 (초)
            radius_meters: 스냅핑 반경(미터). 기본값 250m (그리드 500m 정사각형의 중심점 기준)
            save_json_dir: OSRM 응답 JSON 저장 디렉토리 (옵션)
            route_type: "center2site" 또는 "hos2site"

        Returns:
            (distance_km, duration_min) 튜플 - 거리(km)와 이송시간(분)
            실패시 (None, None, error_reason) 반환

        주의: Kakao API와 동일한 반환 형식 유지
            - distance: meters → km 변환
            - duration: seconds → minutes 변환
        """
        # OSRM API URL (lon, lat 순서 주의!)
        url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
        # radiuses 파라미터: 각 좌표당 최대 스냅핑 거리 (미터)
        params = {
            "overview": "full" if save_json_dir else "false",
            "radiuses": f"{radius_meters};{radius_meters}",  # 250m 반경 내 도로 탐색
            "continue_straight": "false"
        }
        if save_json_dir:
            params["geometries"] = "geojson"

        for attempt in range(max_retries):
            try:
                time.sleep(rate_limit_delay)  # Rate limiting
                self.api_call_count += 1
                response = requests.get(url, params=params, timeout=15)

                if response.status_code == 200:
                    data = response.json()
                    code = data.get("code")

                    if code == "Ok" and data.get("routes"):
                        route = data["routes"][0]
                        # Kakao와 동일한 변환 로직
                        distance_km = route["distance"] / 1000.0  # m → km
                        duration_min = route["duration"] / 60.0   # sec → min
                        if save_json_dir:
                            now = datetime.now(KST).isoformat()
                            meta = {
                                "api_provider": "osrm",
                                "route_type": route_type,
                                "source_index": source_index,
                                "name": name,
                                start_label: [start[1], start[0]],
                                goal_label: [end[1], end[0]],
                                "saved_at": now,
                                "data_version": data.get("data_version"),
                                "distance_km": round(distance_km, 3),
                                "duration_min": round(duration_min, 2),
                                "duration_sec": route.get("duration", 0),
                                "direction_note": f"{start_label}->{goal_label}",
                                "radius_meters": radius_meters,
                            }
                            fname = f"{(source_index if source_index is not None else 0):03d}_{slugify(name or 'noname')}.json"
                            out_path = os.path.join(save_json_dir, fname)
                            ensure_dir(os.path.dirname(out_path))
                            json_data = {
                                "meta": meta,
                                "payload": {"osrm_response": data},
                            }
                            with open(out_path, "w", encoding="utf-8") as f:
                                json.dump(json_data, f, ensure_ascii=False, indent=2)
                        return distance_km, duration_min, None
                    elif code == "NoRoute":
                        error_msg = f"NoRoute: 경로를 찾을 수 없음 (도로 연결 안됨)"
                        print(f"  ⚠️ OSRM API: {error_msg}")
                        return None, None, error_msg
                    elif code == "NoSegment":
                        error_msg = f"NoSegment: {radius_meters}m 내에 도로 없음 (격오지)"
                        print(f"  ⚠️ OSRM API: {error_msg}")
                        return None, None, error_msg
                    else:
                        error_msg = f"OSRM 응답 코드: {code}"
                        print(f"  ⚠️ OSRM API: {error_msg}")
                        return None, None, error_msg

                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        print(f"  ⚠️ OSRM rate limit (429), 5초 대기...")
                        time.sleep(5)
                    else:
                        error_msg = "Rate limit 초과"
                        return None, None, error_msg

                elif response.status_code == 400:
                    # 400 에러는 좌표 형식 문제일 가능성 높음
                    error_msg = f"HTTP 400 Bad Request - 좌표 형식 오류 가능성"
                    print(f"  ⚠️ OSRM API: {error_msg}")
                    print(f"     요청 URL: {url}")
                    print(f"     파라미터: {params}")
                    try:
                        error_detail = response.json()
                        print(f"     응답 상세: {error_detail}")
                    except:
                        pass
                    return None, None, error_msg

                else:
                    error_msg = f"HTTP {response.status_code}"
                    print(f"  ⚠️ OSRM API 오류: {error_msg}")
                    return None, None, error_msg

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"  ⚠️ OSRM timeout, 재시도 {attempt+1}/{max_retries}")
                    time.sleep(2 ** attempt)  # 지수 백오프
                else:
                    error_msg = "API 타임아웃"
                    return None, None, error_msg

            except Exception as e:
                error_msg = f"예외 발생: {str(e)}"
                print(f"  ⚠️ OSRM 오류: {error_msg}")
                return None, None, error_msg

        # 모든 재시도 실패
        return None, None, "모든 재시도 실패"

    def generate_coordinate_for_scenario(self, mode="korea_random", sido_name=None):
        """
        시나리오용 좌표 생성 (JSON 형태로 상세 정보 출력)
        Args:
            mode: "korea_random", "sido", "manual"
            sido_name: 시도명 (mode="sido"일 때 필요)
        Returns:
            (latitude, longitude) 또는 None
        """
        try:
            if mode == "manual":
                return None  # 수동은 외부에서 값 제공
            
            result = self.coord_generator.generate_valid_coordinate(mode, sido_name)
            if result:
                lat, lon, addr_info = result
                output_info = {
                    "latitude": lat,
                    "longitude": lon,
                    "full_address": addr_info.get("full_address", ""),
                    "road_address": addr_info.get("road_address", ""),
                    "area1": addr_info.get("area1", ""),
                    "area2": addr_info.get("area2", ""),
                    "area3": addr_info.get("area3", ""),
                    "area4": addr_info.get("area4", ""),
                    "is_valid": addr_info.get("is_valid", False)
                }
                print(f"COORDINATE_INFO:{json.dumps(output_info, ensure_ascii=False)}")
                print(f"  📍 좌표 생성: ({lat}, {lon}) - {addr_info.get('area1','')} {addr_info.get('area2','')}")
                return lat, lon
            else:
                print("  ❌ 유효한 좌표 생성 실패")
                return None
        except Exception as e:
            print(f"  💥 좌표 생성 오류: {e}")
            return None

    def make_amb_info(self, latitude, longitude, incident_size, save_folder, save_routes_json=True, route_mode="osrm"):
        """구급차 정보 생성 (격오지 조기 감지 추가)"""
        print(f"  🚑 구급차 정보 생성 중...")
        try:
            df = pd.read_csv(self.fire_data_path, encoding="cp949")
        except Exception as e:
            print(f"❌ 소방서 데이터 로드 실패: {e}")
            return

        coords = list(zip(df["y좌표"], df["x좌표"]))
        euc_distances = [haversine(coord, (latitude, longitude)) for coord in coords]
        df["euclidean_distance"] = euc_distances

        # EUC 저장
        df_sorted_euc = df.sort_values("euclidean_distance").head(incident_size).copy()
        df_sorted_euc = df_sorted_euc.rename(columns={
            "euclidean_distance": "init_distance",
            "기관명": "안전센터/소방서이름"
        })
        df_sorted_euc = df_sorted_euc.reset_index(drop=True)
        df_sorted_euc = df_sorted_euc[["init_distance", "안전센터/소방서이름"]]
        euc_save_path = os.path.join(save_folder, "amb_info_euc.csv")
        df_sorted_euc.to_csv(euc_save_path, index=True, index_label="Index", encoding="utf-8-sig")

        mode = str(route_mode or "osrm").lower()
        if mode not in ("osrm", "kakao", "both"):
            mode = "osrm"

        routes_dir = None
        routes_dir_kakao = None
        if save_routes_json:
            base_routes_dir = os.path.join(save_folder, "routes", "center2site")
            if mode == "both":
                routes_dir = os.path.join(base_routes_dir, "osrm")
                routes_dir_kakao = os.path.join(base_routes_dir, "kakao")
                ensure_dir(routes_dir)
                ensure_dir(routes_dir_kakao)
            elif mode == "kakao":
                routes_dir_kakao = base_routes_dir
                ensure_dir(routes_dir_kakao)
            else:
                routes_dir = base_routes_dir
                ensure_dir(routes_dir)

        # 후보군 확장 및 도로 거리/시간 계산 (OSRM API)
        df_candidates = df.sort_values("euclidean_distance").head(int(incident_size * self.multiplier)).copy()
        successful_centers = []

        # ★ NoSegment 에러 카운터 추가 (격오지 조기 감지)
        nosegment_count = 0
        NOSEGMENT_THRESHOLD = 4  # 격오지 판단 임계값

        for source_index, (_, row) in enumerate(df_candidates.iterrows()):
            # 이미 충분한 소방서를 찾았으면 종료
            if len(successful_centers) >= incident_size:
                break

            coord = (row["y좌표"], row["x좌표"])  # (lat, lon) of center
            center_name = row.get('기관명', 'Unknown')

            if mode == "kakao":
                dist_km, duration_min = self.get_road_distance_kakao(
                    start=coord,
                    end=(latitude, longitude),  # center → site
                    save_json_dir=routes_dir_kakao,
                    route_type="center2site",
                    source_index=source_index,
                    name=center_name,
                    start_label="center",
                    goal_label="site",
                )
                error_reason = None
            else:
                # OSRM API 호출 (optional JSON save)
                dist_km, duration_min, error_reason = self.get_road_distance_osrm(
                    start=coord,
                    end=(latitude, longitude),  # center → site
                    save_json_dir=routes_dir,
                    route_type="center2site",
                    source_index=source_index,
                    name=center_name,
                    start_label="center",
                    goal_label="site",
                )
                if mode == "both" and routes_dir_kakao:
                    self.get_road_distance_kakao(
                        start=coord,
                        end=(latitude, longitude),  # center → site
                        save_json_dir=routes_dir_kakao,
                        route_type="center2site",
                        source_index=source_index,
                        name=center_name,
                        start_label="center",
                        goal_label="site",
                    )

            # ★ NoSegment 에러 카운팅
            if mode != "kakao" and dist_km is None and error_reason and "NoSegment" in error_reason:
                nosegment_count += 1
                print(f"  ⚠️ 소방서 '{center_name}' 스킵 ({nosegment_count}번째 NoSegment): {error_reason}")
                print(f"     좌표: {coord} → ({latitude}, {longitude})")

                # ★ 4번 이상 발생 시 격오지로 판단하고 조기 종료
                if nosegment_count >= NOSEGMENT_THRESHOLD:
                    raise ValueError(
                        f"격오지 감지: {NOSEGMENT_THRESHOLD}개 이상의 소방서에서 NoSegment 발생. "
                        f"사고 지점 ({latitude}, {longitude})이 도로 접근 불가 지역입니다."
                    )
                continue

            # API 실패 (NoSegment 외 다른 오류)
            if dist_km is None:
                print(f"  ⚠️ 소방서 '{center_name}' 스킵: {error_reason}")
                print(f"     좌표: {coord} → ({latitude}, {longitude})")
                continue

            # 성공한 경우 데이터 저장
            row_dict = row.to_dict()
            row_dict["source_index"] = source_index
            row_dict['road_distance'] = dist_km
            row_dict['road_duration'] = duration_min
            successful_centers.append(row_dict)

        # 최소한의 소방서도 찾지 못한 경우에만 에러 발생
        if len(successful_centers) < incident_size:
            raise ValueError(
                f"구급차 도로 탐색 실패: {incident_size}개 필요하지만 {len(successful_centers)}개만 성공. "
                f"(NoSegment: {nosegment_count}회)"
            )

        # ROAD 저장 (duration 기준으로 정렬)
        df_sorted_road = pd.DataFrame(successful_centers).sort_values("road_duration").reset_index(drop=True)
        df_sorted_road = df_sorted_road.rename(columns={
            "road_distance": "init_distance",
            "road_duration": "duration",
            "기관명": "안전센터/소방서이름"
        })
        df_sorted_road = df_sorted_road.reset_index(drop=True)
        df_top = df_sorted_road.head(incident_size).copy()
        if save_routes_json:
            if routes_dir:
                self._reindex_route_jsons(routes_dir, df_top, "안전센터/소방서이름")
            if routes_dir_kakao:
                self._reindex_route_jsons(routes_dir_kakao, df_top, "안전센터/소방서이름")
        df_top = df_top[["init_distance", "duration", "안전센터/소방서이름"]]
        road_save_path = os.path.join(save_folder, "amb_info_road.csv")
        df_top.to_csv(road_save_path, index=True, index_label="Index", encoding="utf-8-sig")

        print(f"  ✅ 구급차 정보 생성 완료")

    def make_hospital_info(self, latitude, longitude, incident_size, save_folder, uav_size=0, save_routes_json=True, route_mode="osrm"):
        """병원 정보 생성 (기존 로직 유지 + 최소 조건 추가 보장)

        Args:
            latitude: 사고지점 위도
            longitude: 사고지점 경도
            incident_size: 환자 수
            save_folder: 저장 폴더
            uav_size: UAV 대수 (헬기장 병원 최소 보장에 사용)
        """
        print(f"  🏥 병원 정보 생성 중...")
        
        # ---------- (0) 데이터 로드 ----------
        try:
            df_full = pd.read_excel(self.hospital_data_path, engine='openpyxl')
        except Exception as e:
            print(f"❌ 병원 데이터 로드 실패: {e}")
            return

        # 필요한 열만 사용 (이름 유지)
        cols_needed = ["요양기관명", "종별코드", "응급실병상수", "x좌표", "y좌표"]
        for c in cols_needed:
            if c not in df_full.columns:
                raise KeyError(f"필수 컬럼 누락: {c}")
        df = df_full[cols_needed].copy()

        # ★ 헬기장 여부 컬럼 추가 (있으면 포함, 없으면 0으로 채움)
        if "헬기장 여부" in df_full.columns:
            df["헬기장 여부"] = df_full["헬기장 여부"].fillna(0).astype(int)
        else:
            df["헬기장 여부"] = 0  # 헬기장 정보 없으면 모두 0

        # ---------- (1) 유클리드 거리 계산 ----------
        coords = list(zip(df["y좌표"], df["x좌표"]))  # (lat, lon)
        df["euclidean_distance"] = [haversine((lat, lon), (latitude, longitude)) for (lat, lon) in coords]

        # ---------- (2) 파라미터 ----------
        util_by_tier = getattr(self, "util_by_tier", {1: 0.656, 11: 0.461, "etc": 0.461})
        # queue_policy = str(getattr(self, "queue_policy", "0")).strip()
        try:
            buffer_ratio = float(getattr(self, "buffer_ratio", 1.5))
        except Exception:
            buffer_ratio = 1.5

        ratio = self.patient_config.get("ratio", {"Red":0.1,"Yellow":0.3,"Green":0.5,"Black":0.1})
        U = int(round(incident_size * float(ratio.get("Red", 0))))
        N = int(incident_size)
        
        import math
        def _get_util(code):
            try:
                icode = int(code)
                return util_by_tier.get(icode, util_by_tier.get("etc", 0.461))
            except Exception:
                return util_by_tier.get("etc", 0.461)
            
        df["util"] = df["종별코드"].apply(_get_util)
        df["capa"] = (df["응급실병상수"] * (1 - df["util"])).apply(lambda x: int(max(0, math.floor(x))))
        # 수술실 수 종별코드별 고정
        conditions = [df['종별코드'] == 1, df['종별코드'] == 11]; values = [3, 2]
        df['operating_rooms'] = np.select(conditions, values, default=1)
        df["eff"] = df["operating_rooms"] + df["capa"]
        df["is_tier1"] = (df["종별코드"].astype(str).astype(float).astype(int) == 1).astype(int)
        
        # ---------- (3) 전역 상급 용량 점검 (불가능 사전 감지) ----------
        total_tier1_capa_all = int(df.loc[df["is_tier1"]==1, "capa"].sum())
        total_capa_all = int(df["capa"].sum())
        if total_tier1_capa_all < U:
            print(f"  ⚠️ 전역 상급 용량 부족: Tier1_capa_all={total_tier1_capa_all} < U={U}. 최선 선택으로 진행(전원 실패 가능).")
        
        # --- (4) 후보군 확장: 기존 코드와 동일 ---
        # 가까운 병원들을 포함한 넉넉한 후보군(df_cand)
        df_sorted = df.sort_values("euclidean_distance").reset_index(drop=True)
        sum_capa = 0; sum_capa_tier1 = 0; cand_idx = []; 
        for i, row in df_sorted.iterrows():
            cand_idx.append(i)
            sum_capa += int(row["eff"])
            if row["is_tier1"] == 1: sum_capa_tier1 += int(row["eff"]); 
            if (sum_capa >= N * buffer_ratio): break
        if not cand_idx:
            cand_idx = list(range(len(df_sorted)))
        df_cand = df_sorted.loc[cand_idx].copy()
        
        df_selected = df_cand.copy()


        # ================================================================= #
        # 위에서 선택된 목록에 최소 조건을 만족하는지 확인하고 부족할 시 추가
        # 규칙 1: 상급종합병원(Tier 1) 최소 2개 보장
        final_tier1 = df_selected[df_selected["is_tier1"] == 1]
        num_to_ensure_tier1 = 2 - len(final_tier1)
        if num_to_ensure_tier1 > 0:
            print(f"  INFO: 최종 목록의 상급병원이 {len(final_tier1)}개. 최소 2개를 위해 '추가'합니다.")
            # 전체 병원 목록에서 아직 선택되지 않은 가장 가까운 상급병원을 찾아서 최소 2개가 될때까지 추가
            candidates = df_sorted[(df_sorted["is_tier1"] == 1) & (~df_sorted.index.isin(df_selected.index))]
            if not candidates.empty:
                hospitals_to_add = candidates.head(num_to_ensure_tier1)
                df_selected = pd.concat([df_selected, hospitals_to_add])

        # 규칙 2: 상급종합병원이 환자 40% 수용 용량 보장 (Tier 1 기준, 환자수가 많을때 최소 red환자 10% 이상 + 확률분포 고려한 비율)
        target_capa = N * 0.4
        current_capa = df_selected[df_selected["is_tier1"] == 1]["eff"].sum()
        while current_capa < target_capa:
            print(f"  INFO: 상급병원 용량이 {current_capa}/{target_capa}. 용량을 위해 '추가'합니다.")
            candidates = df_sorted[(df_sorted["is_tier1"] == 1) & (~df_sorted.index.isin(df_selected.index))]
            if candidates.empty: print("  WARNING: 추가할 상급병원이 더 이상 없습니다."); break
            hospital_to_add = candidates.head(1)
            df_selected = pd.concat([df_selected, hospital_to_add])
            current_capa = df_selected[df_selected["is_tier1"] == 1]["eff"].sum()

        # 규칙 3: 그 외 병원(Tier 2 등) 최소 1개 보장 (우연히 가장 가까이 있는 병원이 상급종합병원뿐일때 64개의 룰 중 실패하는 룰이 존재하므로)
        if len(df_selected[df_selected["is_tier1"] == 0]) == 0:
            print("  INFO: 최종 목록에 Tier 2 병원이 없음. 시뮬레이션 오류 방지를 위해 '추가'합니다.")
            candidates = df_sorted[(df_sorted["is_tier1"] == 0) & (~df_sorted.index.isin(df_selected.index))]
            if not candidates.empty:
                df_selected = pd.concat([df_selected, candidates.head(1)])

        # ================================================================= #
        # 규칙 4: 헬기장 병원 최소 보장 (UAV 대수 이상)
        if "헬기장 여부" in df_selected.columns:
            # UAV 대수 확인 (파라미터에서)
            uav_n = int(max(0, uav_size))

            if uav_n > 0:
                helipad_hospitals = df_selected[df_selected["헬기장 여부"] == 1]
                num_helipad = len(helipad_hospitals)

                # UAV 대수만큼 헬기장 병원이 없으면 추가
                num_to_ensure_helipad = uav_n - num_helipad

                if num_to_ensure_helipad > 0:
                    print(f"  INFO: 헬기장 병원이 {num_helipad}개인데 UAV는 {uav_n}대. 최소 {uav_n}개 헬기장 병원 확보를 위해 '{num_to_ensure_helipad}개' 추가합니다.")

                    # 전체 병원 목록에서 헬기장 있는 병원 중 아직 선택되지 않은 것 찾기
                    candidates_helipad = df_sorted[
                        (df_sorted["헬기장 여부"] == 1) &
                        (~df_sorted.index.isin(df_selected.index))
                    ]

                    if not candidates_helipad.empty:
                        # 필요한 만큼 헬기장 병원 추가
                        hospitals_to_add = candidates_helipad.head(num_to_ensure_helipad)
                        df_selected = pd.concat([df_selected, hospitals_to_add])
                        added_names = ", ".join(hospitals_to_add['요양기관명'].values)
                        print(f"    → 추가된 헬기장 병원: {added_names}")
                    else:
                        print(f"  ⚠️ 경고: 전체 데이터에 헬기장 병원이 {num_helipad}개밖에 없습니다. UAV {uav_n}대 운용이 불가능합니다.")
                else:
                    print(f"  ✓ 헬기장 병원 {num_helipad}개 (UAV {uav_n}대 운용 가능)")
            else:
                print("  INFO: UAV 대수가 0이므로 헬기장 병원 보장 로직을 건너뜁니다.")
        else:
            print("  ⚠️ '헬기장 여부' 컬럼이 원본 데이터에 없습니다. 헬기장 보장 로직을 건너뜁니다.")

        # ================================================================= #
        # 규칙 5: UAV 이송을 위한 교집합 병원 보장 (헬기장+Tier)
        if "헬기장 여부" in df_selected.columns:
            uav_n = int(max(0, uav_size))

            if uav_n > 0:
                # 5-1: Red UAV 이송용 헬기장+Tier1 병원 최소 1개 보장
                helipad_tier1_hospitals = df_selected[
                    (df_selected["헬기장 여부"] == 1) &
                    (df_selected["is_tier1"] == 1)
                ]

                if len(helipad_tier1_hospitals) == 0:
                    print("  INFO: Red UAV 이송용 헬기장+Tier1 병원이 없음. 추가 중...")
                    candidates = df_sorted[
                        (df_sorted["헬기장 여부"] == 1) &
                        (df_sorted["is_tier1"] == 1) &
                        (~df_sorted.index.isin(df_selected.index))
                    ]

                    if not candidates.empty:
                        hospital_to_add = candidates.head(1)
                        df_selected = pd.concat([df_selected, hospital_to_add])
                        added_name = hospital_to_add['요양기관명'].values[0]
                        print(f"    → 추가됨: {added_name}")
                    else:
                        print("  ⚠️ 경고: 전체 데이터에 헬기장+Tier1 병원 없음. Red UAV 이송 불가!")
                else:
                    print(f"  ✓ 헬기장+Tier1 병원 {len(helipad_tier1_hospitals)}개 (Red UAV 이송 가능)")

                # 5-2: Yellow UAV 이송용 헬기장+Tier2 병원 최소 1개 보장
                helipad_tier2_hospitals = df_selected[
                    (df_selected["헬기장 여부"] == 1) &
                    (df_selected["is_tier1"] == 0)
                ]

                if len(helipad_tier2_hospitals) == 0:
                    print("  INFO: Yellow UAV 이송용 헬기장+Tier2 병원이 없음. 추가 중...")
                    candidates = df_sorted[
                        (df_sorted["헬기장 여부"] == 1) &
                        (df_sorted["is_tier1"] == 0) &
                        (~df_sorted.index.isin(df_selected.index))
                    ]

                    if not candidates.empty:
                        hospital_to_add = candidates.head(1)
                        df_selected = pd.concat([df_selected, hospital_to_add])
                        added_name = hospital_to_add['요양기관명'].values[0]
                        print(f"    → 추가됨: {added_name}")
                    else:
                        print("  ⚠️ 경고: 전체 데이터에 헬기장+Tier2 병원 없음. Yellow UAV 이송 불가!")
                else:
                    print(f"  ✓ 헬기장+Tier2 병원 {len(helipad_tier2_hospitals)}개 (Yellow UAV 이송 가능)")
            else:
                print("  INFO: UAV 대수가 0이므로 헬기장+Tier 교집합 보장 로직을 건너뜁니다.")
        else:
            print("  ⚠️ '헬기장 여부' 컬럼이 원본 데이터에 없습니다. 헬기장+Tier 교집합 보장 로직을 건너뜁니다.")

        df_euc = df_selected.sort_values("euclidean_distance").reset_index(drop=True).copy()
        print(f" 최종 생성된 병원: {len(df_euc)}곳 (상급: {df_euc['is_tier1'].sum()}곳, 종합 등: {len(df_euc) - df_euc['is_tier1'].sum()}곳)")

        # ---------- (6) EUC 파일은 나중에 road 순서로 저장 (인덱스 일치 보장) ----------
        # ★ CRITICAL: distance_Hos2Site_euc.csv는 road 순서를 따라야 h_states와 인덱스가 일치
        # ★ 따라서 이 시점에서는 euc_info만 저장하고, distance는 road 재정렬 후 저장합니다.

        euc_info = df_euc[["operating_rooms", "capa", "종별코드", "요양기관명", "헬기장 여부"]].copy()
        euc_info.columns = ["수술실수", "병상수", "종별코드", "요양기관명", "헬기장 여부"]
        euc_info_path = os.path.join(save_folder, "hospital_info_euc.csv")
        euc_info.to_csv(euc_info_path, index=True, index_label="Index", encoding="utf-8-sig")

        mode = str(route_mode or "osrm").lower()
        if mode not in ("osrm", "kakao", "both"):
            mode = "osrm"

        routes_dir_hos = None
        routes_dir_hos_kakao = None
        if save_routes_json:
            base_routes_dir = os.path.join(save_folder, "routes", "hos2site")
            if mode == "both":
                routes_dir_hos = os.path.join(base_routes_dir, "osrm")
                routes_dir_hos_kakao = os.path.join(base_routes_dir, "kakao")
                ensure_dir(routes_dir_hos)
                ensure_dir(routes_dir_hos_kakao)
            elif mode == "kakao":
                routes_dir_hos_kakao = base_routes_dir
                ensure_dir(routes_dir_hos_kakao)
            else:
                routes_dir_hos = base_routes_dir
                ensure_dir(routes_dir_hos)

        # ---------- (7) ROAD 거리 & 시간 계산 & 저장 (선정 병원만, OSRM API) ----------
        road_distances = []
        road_durations = []
        successful_hospitals = []

        for source_index, (_, row) in enumerate(df_euc.iterrows()):
            end = (row["y좌표"], row["x좌표"])
            hospital_name = row.get('요양기관명', 'Unknown')

            if mode == "kakao":
                road_km, duration_min = self.get_road_distance_kakao(
                    start=(latitude, longitude),
                    end=end,  # site → hospital
                    save_json_dir=routes_dir_hos_kakao,
                    route_type="hos2site",
                    source_index=source_index,
                    name=hospital_name,
                    start_label="site",
                    goal_label="hospital",
                )
                error_reason = None
            else:
                # OSRM API 호출 (optional JSON save)
                road_km, duration_min, error_reason = self.get_road_distance_osrm(
                    start=(latitude, longitude),
                    end=end,  # site → hospital
                    save_json_dir=routes_dir_hos,
                    route_type="hos2site",
                    source_index=source_index,
                    name=hospital_name,
                    start_label="site",
                    goal_label="hospital",
                )
                if mode == "both" and routes_dir_hos_kakao:
                    self.get_road_distance_kakao(
                        start=(latitude, longitude),
                        end=end,  # site → hospital
                        save_json_dir=routes_dir_hos_kakao,
                        route_type="hos2site",
                        source_index=source_index,
                        name=hospital_name,
                        start_label="site",
                        goal_label="hospital",
                    )

            # API 실패시 경고 로그만 출력하고 다음 병원 시도
            if road_km is None:
                print(f"  ⚠️ 병원 '{hospital_name}' 스킵: {error_reason}")
                print(f"     좌표: ({latitude}, {longitude}) → {end}")
                continue

            road_distances.append(road_km)
            road_durations.append(duration_min)
            row_dict = row.to_dict()
            row_dict["source_index"] = source_index
            successful_hospitals.append(row_dict)

        # 최소한의 병원도 찾지 못한 경우에만 에러 발생
        if len(successful_hospitals) == 0:
            raise ValueError(
                f"병원 도로 탐색 실패: 모든 병원에서 API 오류. "
                f"250m 내 도로가 없거나 좌표 형식 오류입니다."
            )

        # 성공한 병원들만으로 DataFrame 재구성
        df_euc = pd.DataFrame(successful_hospitals).reset_index(drop=True)
        df_euc["road_distance"] = road_distances
        df_euc["road_duration"] = road_durations
        df_road = df_euc.sort_values("road_duration").reset_index(drop=True).copy()
        if save_routes_json:
            if routes_dir_hos:
                self._reindex_route_jsons(routes_dir_hos, df_road, "요양기관명")
            if routes_dir_hos_kakao:
                self._reindex_route_jsons(routes_dir_hos_kakao, df_road, "요양기관명")

        # distance_Hos2Site_road.csv에 duration 컬럼 추가
        dist_road_df = pd.DataFrame({
            "distance": df_road["road_distance"],
            "duration": df_road["road_duration"]
        })
        dist_road_path = os.path.join(save_folder, "distance_Hos2Site_road.csv")
        dist_road_df.to_csv(dist_road_path, index=True, index_label="Index", encoding="utf-8-sig")

        # ★ CRITICAL FIX: distance_Hos2Site_euc.csv를 road 순서로 저장 (인덱스 일치 보장)
        # df_road는 road_duration 기준으로 정렬되어 있으므로, h_states와 동일한 인덱스 순서를 가집니다.
        # euclidean_distance 값은 유지하되, 순서만 road 기준으로 변경합니다.
        dist_euc_df = pd.DataFrame({"distance": df_road["euclidean_distance"]})
        dist_euc_path = os.path.join(save_folder, "distance_Hos2Site_euc.csv")
        dist_euc_df.to_csv(dist_euc_path, index=True, index_label="Index", encoding="utf-8-sig")

        road_info = df_road[["operating_rooms", "capa", "종별코드", "요양기관명", "헬기장 여부"]].copy()
        road_info.columns = ["수술실수", "병상수", "종별코드", "요양기관명", "헬기장 여부"]
        road_info_path = os.path.join(save_folder, "hospital_info_road.csv")
        road_info.to_csv(road_info_path, index=True, index_label="Index", encoding="utf-8-sig")

        print(f"  ✅ 병원 정보 생성 완료 (distance_Hos2Site_euc.csv는 road 순서로 저장됨)")


    
    def make_uav_info(self, latitude, longitude, incident_size, uav_size, save_folder):
        """UAV 정보 생성 - hospital_info_road.csv 기반 (★핵심 변경★)
        - hospital_info_road.csv에서 "헬기장 여부"=1인 병원만 필터링
        - 사고지점 기준 거리 계산 후 가장 가까운 N개 헬기장 병원 선정
        - 각 병원당 최대 1개 UAV 배정
        - ★ uav_info 병원 = hospital_info의 부분집합 보장 (인덱스 일치)
        - CSV 구조: Index, init_distance, 수술실수, 병상수, 종별코드, 요양기관명
        """
        print(f"  🚁 UAV 정보 생성 중 (hospital_info_road.csv 기반)...")

        import os
        import pandas as pd
        from haversine import haversine

        # 0) 파라미터 정리
        try:
            uav_n = int(max(0, int(uav_size)))
        except Exception:
            uav_n = 0

        # UAV 0대인 경우: 헤더만 있는 빈 CSV 생성
        if uav_n <= 0:
            print("⚠️ UAV 대수가 0입니다. 헤더만 있는 빈 파일 생성...")
            uav_info_path = os.path.join(save_folder, "uav_info.csv")
            # 헤더만 작성
            header = "Index,init_distance,수술실수,병상수,종별코드,요양기관명"
            with open(uav_info_path, 'w', encoding='utf-8-sig') as f:
                f.write(header + '\n')
            print(f"  ✅ UAV 0대 - 헤더만 있는 빈 파일 생성 완료")
            return

        # 1) ★ hospital_info_road.csv 로드 (기존 엑셀 대신!)
        hospital_info_path = os.path.join(save_folder, "hospital_info_road.csv")
        if not os.path.exists(hospital_info_path):
            print(f"❌ {hospital_info_path} 파일이 없습니다.")
            print("   make_hospital_info()를 먼저 실행해주세요.")
            raise FileNotFoundError(f"❌ {hospital_info_path} 파일이 없습니다.")

        try:
            df_hospital_pool = pd.read_csv(hospital_info_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"❌ hospital_info_road.csv 로드 실패: {e}")
            return

        # 2) "헬기장 여부" 컬럼 확인 (필수)
        if "헬기장 여부" not in df_hospital_pool.columns:
            print("❌ '헬기장 여부' 컬럼이 hospital_info_road.csv에 없습니다.")
            print("   make_hospital_info()에서 헬기장 컬럼 추가 로직을 확인해주세요.")
            raise KeyError("❌ hospital_info_road.csv에 '헬기장 여부' 컬럼이 없습니다.")

        # 3) hospital_info 내에서 헬기장 병원만 필터링
        df_helipad_in_pool = df_hospital_pool[df_hospital_pool["헬기장 여부"] == 1].copy()

        if df_helipad_in_pool.empty:
            print("❌ hospital_info_road.csv에 헬기장이 있는 병원이 없습니다.")
            print("   make_hospital_info()의 헬기장 보장 로직을 확인해주세요.")
            raise ValueError("❌ hospital_info에 헬기장이 있는 병원이 없습니다.")

        # 4) 헬기장 병원 개수 검증 (UAV 대수와 비교)
        if len(df_helipad_in_pool) < uav_n:
            print(f"❌ hospital_info에 헬기장 병원이 {len(df_helipad_in_pool)}개밖에 없어 UAV {uav_n}대를 배치할 수 없습니다.")
            print(f"   make_hospital_info()의 헬기장 보장 로직을 확인하거나 UAV 대수를 줄여주세요.")
            raise ValueError(
                f"❌ hospital_info에 헬기장 병원이 {len(df_helipad_in_pool)}개밖에 없어 "
                f"UAV {uav_n}대를 배치할 수 없습니다."
            )

        # 5) 사고지점-병원 유클리드 거리 계산 (hospital_info에는 좌표가 없으므로 원본 Excel에서 가져와야 함)
        # ★ hospital_info_road.csv에 이미 거리 정보가 있을 수 있지만, 안전하게 원본에서 좌표를 가져옴
        try:
            df_full_excel = pd.read_excel(self.hospital_data_path, engine="openpyxl")
        except Exception as e:
            print(f"❌ 원본 Excel 데이터 로드 실패: {e}")
            return

        # 병원명 기준으로 좌표 매칭
        df_helipad_in_pool = df_helipad_in_pool.merge(
            df_full_excel[["요양기관명", "x좌표", "y좌표"]],
            on="요양기관명",
            how="left"
        )

        # 좌표가 없는 병원 체크
        if df_helipad_in_pool[["x좌표", "y좌표"]].isnull().any().any():
            missing_hospitals = df_helipad_in_pool[df_helipad_in_pool[["x좌표", "y좌표"]].isnull().any(axis=1)]["요양기관명"].tolist()
            print(f"⚠️ 경고: 다음 병원의 좌표 정보가 없습니다: {missing_hospitals}")
            df_helipad_in_pool = df_helipad_in_pool.dropna(subset=["x좌표", "y좌표"])

        df_helipad_in_pool["distance"] = df_helipad_in_pool.apply(
            lambda row: haversine((row["y좌표"], row["x좌표"]), (latitude, longitude)),
            axis=1
        )

        # 6) 거리순 정렬 (가까운 헬기장 병원부터)
        df_helipad_in_pool = df_helipad_in_pool.sort_values("distance").reset_index(drop=True)

        # 7) 상위 N개 선정 (각 병원 최대 1개 UAV)
        df_selected = df_helipad_in_pool.head(uav_n).copy()

        # 8) CSV 저장 (hospital_info와 동일한 병원 사용, 인덱스 일치 보장)
        result_df = pd.DataFrame({
            "Index": range(len(df_selected)),
            "init_distance": df_selected["distance"].round(3),
            "수술실수": df_selected["수술실수"],
            "병상수": df_selected["병상수"],
            "종별코드": df_selected["종별코드"],
            "요양기관명": df_selected["요양기관명"]
        })

        save_path = os.path.join(save_folder, "uav_info.csv")
        result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"  ✅ UAV 정보 생성 완료: {len(result_df)}개 UAV")
        print(f"     헬기장 병원: {', '.join(df_selected['요양기관명'].head(3).tolist())}{'...' if len(df_selected) > 3 else ''}")
        print(f"     ★ hospital_info의 부분집합으로 생성됨 (인덱스 일치 보장)")



    def make_patient_info(self, save_folder):
        """환자 정보 생성 (하드코딩된 값 사용)"""
        print(f"  👥 환자 정보 생성 중...")
        types = self.patient_config["ratio"].keys()
        rows = []
        for t in types:
            α, β = self.patient_config["rescue_param"][t]
            rows.append({
                "type": t,
                "ratio": self.patient_config["ratio"][t],
                "rescue_param_alpha": α,
                "rescue_param_beta": β,
                "treat_tier1": self.patient_config["treat_tier1"][t],
                "treat_tier2": self.patient_config["treat_tier2"][t],
                "treat_tier1_mean": self.patient_config["treat_tier1_mean"][t],
                "treat_tier2_mean": self.patient_config["treat_tier2_mean"][t]
            })
        df = pd.DataFrame(rows)
        save_path = os.path.join(save_folder, "patient_info.csv")
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ 환자 정보 생성 완료")

    def make_distance_Hos2Hos(self, save_folder):
        """병원 간 거리 행렬 생성"""
        print(f"  📐 병원간 거리 행렬 생성 중...")
        try:
            df_full = pd.read_excel(self.hospital_data_path, engine="openpyxl")
        except Exception as e:
            print(f"❌ 병원 데이터 로드 실패: {e}")
            return

        # Euclidean (★ CRITICAL FIX: road 순서 기준으로 생성)
        try:
            # ★ hospital_info_euc.csv 대신 hospital_info_road.csv 사용 (인덱스 일치 보장)
            file_road = os.path.join(save_folder, "hospital_info_road.csv")
            df_road_hos = pd.read_csv(file_road, encoding="utf-8-sig")
            names_road = df_road_hos["요양기관명"].tolist()
            coords_road = []
            for name in names_road:
                row = df_full[df_full["요양기관명"] == name]
                if not row.empty:
                    coords_road.append((row.iloc[0]["y좌표"], row.iloc[0]["x좌표"]))
                else:
                    coords_road.append((0, 0))
            N = len(coords_road)
            matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    if i == j:
                        dist = 0
                    else:
                        dist = haversine(coords_road[i], coords_road[j])
                    matrix[i][j] = dist
                    matrix[j][i] = dist
            save_path_euc = os.path.join(save_folder, "distance_Hos2Hos_euc.csv")
            pd.DataFrame(matrix).to_csv(save_path_euc, index=True, encoding="utf-8-sig")
            print(f"  ✅ 병원간 유클리드 거리 행렬 생성 완료 (road 순서 기준)")
        except Exception as e:
            print(f"❌ 유클리드 거리 계산 실패: {e}")

        # Road (엑셀 파일 사용 - 기존 계산 데이터)
        try:
            file_road = os.path.join(save_folder, "hospital_info_road.csv")
            df_road = pd.read_csv(file_road, encoding="utf-8-sig")
            names_road = df_road["요양기관명"].tolist()

            # Load pre-calculated distance matrix from Excel
            excel_path = os.path.join(self.base_path, "scenarios", "DISTANCE_MATRIX_FINAL.xlsx")
            print(f"  📂 엑셀 거리 행렬 로드 중: {excel_path}")
            df_matrix = pd.read_excel(excel_path, sheet_name="Distance_Matrix", engine="openpyxl")

            # Use first column as index (hospital names)
            df_matrix_indexed = df_matrix.set_index(df_matrix.columns[0])  # Use first column as index

            # Build distance matrix by looking up values
            N = len(names_road)
            matrix = np.zeros((N, N))
            missing_hospitals = []

            for i in range(N):
                for j in range(N):
                    if i == j:
                        matrix[i][j] = 0
                    else:
                        hospital_i = names_road[i]
                        hospital_j = names_road[j]

                        # Look up distance from Excel matrix
                        if hospital_i in df_matrix_indexed.index and hospital_j in df_matrix_indexed.columns:
                            dist = df_matrix_indexed.loc[hospital_i, hospital_j]
                            matrix[i][j] = float(dist) if pd.notna(dist) else 0
                        else:
                            matrix[i][j] = 0
                            if hospital_i not in missing_hospitals:
                                missing_hospitals.append(hospital_i)
                            if hospital_j not in missing_hospitals:
                                missing_hospitals.append(hospital_j)

            if missing_hospitals:
                print(f"  ⚠️ 엑셀에서 찾지 못한 병원 ({len(missing_hospitals)}개): {missing_hospitals[:5]}...")

            save_path_road = os.path.join(save_folder, "distance_Hos2Hos_road.csv")
            pd.DataFrame(matrix).to_csv(save_path_road, index=True, encoding="utf-8-sig")
            print(f"  ✅ 병원간 도로 거리 행렬 생성 완료 (엑셀 데이터 사용)")
        except Exception as e:
            print(f"❌ 도로 거리 계산 실패: {e}")
        print(f"  ✅ 병원간 거리 행렬 생성 완료")

    def _sanitize_coeff_text(self, text: str) -> str:
        """'1.1,1' 또는 '[1.1, 1]' → '1.1, 1' 로 정리"""
        if not text:
            return "1,1"
        t = text.strip()
        if t.startswith("[") and t.endswith("]"):
            t = t[1:-1]
        parts = [p.strip() for p in t.split(",") if p.strip() != ""]
        if len(parts) != 2:
            return "1,1"
        # 숫자 검증 (실패 시 기본)
        try:
            a = float(parts[0]); b = float(parts[1])
        except Exception:
            return "1,1"
        return f"{a},{b}".replace(",", ", ")

    def _reindex_route_jsons(self, save_json_dir, df_sorted, name_col):
        if not save_json_dir or not os.path.isdir(save_json_dir):
            return
        if df_sorted is None or df_sorted.empty:
            return

        for new_idx, (_, row) in enumerate(df_sorted.reset_index(drop=True).iterrows()):
            old_idx = row.get("source_index")
            if old_idx is None or (isinstance(old_idx, float) and pd.isna(old_idx)):
                continue
            try:
                old_idx_int = int(old_idx)
            except Exception:
                continue

            name = row.get(name_col) or row.get("요양기관명") or row.get("기관명") or "noname"
            src_path = os.path.join(save_json_dir, f"{old_idx_int:03d}_{slugify(name)}.json")
            if not os.path.exists(src_path):
                prefix = f"{old_idx_int:03d}_"
                for fname in os.listdir(save_json_dir):
                    if fname.startswith(prefix):
                        src_path = os.path.join(save_json_dir, fname)
                        break
            if not os.path.exists(src_path):
                continue

            dst_path = os.path.join(save_json_dir, f"{new_idx:03d}_{slugify(name)}.json")
            try:
                with open(src_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    meta = data.get("meta")
                    if isinstance(meta, dict):
                        meta["source_index"] = new_idx
                ensure_dir(os.path.dirname(dst_path))
                with open(dst_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    os.remove(src_path)
            except Exception:
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    try:
                        os.replace(src_path, dst_path)
                    except Exception:
                        pass
    
    def make_config_yaml(self, latitude, longitude, incident_size, amb_velocity,
                         uav_velocity, total_samples, random_seed, save_folder, is_use_time=True,
                         amb_handover_time=0, uav_handover_time=0, duration_coeff=1.0,
                         scenario_subdir=None):
        """Config YAML 파일 생성"""
        print(f"  ⚙️ Config YAML 생성 중...")
        folder_name = f"lat{latitude:.6f}_lon{longitude:.6f}"
        config_filename = f"config_{folder_name}.yaml"
        config_path = os.path.join(save_folder, config_filename)
        relative_base = f"./scenarios/{self.experiment_id}"
        if scenario_subdir:
            relative_base = f"{relative_base}/{scenario_subdir}"
        relative_folder = f"{relative_base}/{folder_name}"

        # departure_time 정보
        departure_time_field = ""
        if self.departure_time:
            departure_time_field = f'  departure_time: "{self.departure_time}" # API 조회 시각 (YYYYMMDDHHMM)\n'

        yaml_content = f"""#incident_info:
#  incident_size: {incident_size} # 사고 규모 (총 환자 수)
#  latitude: {latitude} # 위도
#  longitude: {longitude} # 경도
#  incident_type: null # 사고 타입 설정 가능하게 추후 확장

entity_info:
{departure_time_field}  patient:
    incident_size: {incident_size} # 사고 규모 (총 환자 수)
    latitude: {latitude} # 위도
    longitude: {longitude} # 경도
    incident_type: null # 사고 타입 설정 가능하게 추후 확장
    info_path: "{relative_folder}/patient_info.csv"
  hospital:
    load_data: True
    info_path: "{relative_folder}/hospital_info_road.csv"
    dist_Hos2Hos_euc_info: "{relative_folder}/distance_Hos2Hos_euc.csv"
    dist_Hos2Hos_road_info: "{relative_folder}/distance_Hos2Hos_road.csv"
    dist_Hos2Site_euc_info: "{relative_folder}/distance_Hos2Site_euc.csv"
    dist_Hos2Site_road_info: "{relative_folder}/distance_Hos2Site_road.csv"
    max_send_coeff: [{self._sanitize_coeff_text(self.max_send_coeff_text)}]
  ambulance:
    load_data: True
    dispatch_distance_info: "{relative_folder}/amb_info_road.csv"
    velocity: {amb_velocity} # unit: km/h
    handover_time: {amb_handover_time} # unit: minutes
    is_use_time: {str(is_use_time)} # True: API duration 사용, False: 거리/속도 기반 계산
    duration_coeff: {duration_coeff} # API duration 시간가중치 (기본값: 1.0, 환경적 요인 반영시 조정)
  uav:
    load_data: True
    dispatch_distance_info: "{relative_folder}/uav_info.csv"
    velocity: {uav_velocity} # unit: km/h
    handover_time: {uav_handover_time} # unit: minutes
    is_use_time: False # UAV는 항상 유클리드 거리 기반

event_info_path: "event_info.json"

rule_info:
  isFullFactorial: False  # 단일 룰 사용
  priority_rule: ["START"]
  hos_select_rule: ["RedOnly"]
  red_mode_rule: ["OnlyAMB"]
  yellow_mode_rule: ["OnlyAMB"]

run_setting:
  totalSamples: {total_samples} # number of samples
  random_seed: {random_seed} # null, if do not want to fix
  rule_test: True
  eval_mode: True
  output_path: "./results/{self.experiment_id}"
  exp_indicator: "{folder_name}"
  save_info: True # NotImplemented"""
        with open(config_path, 'w', encoding='utf-8') as file:
            file.write(yaml_content)
        print(f"  ✅ Config YAML 생성 완료")
        absolute_config_path = os.path.abspath(config_path)
        print(f"CONFIG_PATH:{absolute_config_path}")
        return absolute_config_path

    def generate_scenario(self, latitude, longitude, incident_size, amb_size,
                          uav_size, amb_velocity, uav_velocity,
                          total_samples, random_seed, is_use_time=True,
                          amb_handover_time=0, uav_handover_time=0, duration_coeff=1.0,
                          save_routes_json=True, route_mode="osrm",
                          use_route_mode_subdir=False):
        """
        완전한 시나리오 생성 (모든 CSV + YAML)
        Args:
            is_use_time: True면 API duration 사용, False면 거리/속도 기반 계산
            amb_handover_time: 구급차 환자 인계시간 (분)
            uav_handover_time: UAV 환자 인계시간 (분)
        Returns: (config_path, api_call_count) 튜플 - 생성된 config 파일 경로 및 API 호출 횟수
        """
        print(f"""\n📍 좌표 ({latitude},{longitude}) 시나리오 생성 시작...""")
        start_time = time.time()
        # API 호출 카운터 초기화
        self.api_call_count = 0
        # 좌표 폴더명: lat{위도}_lon{경도} 형식으로 변경 (괄호/쉼표 제거)
        folder_name = f"lat{latitude:.6f}_lon{longitude:.6f}"
        mode = str(route_mode or "osrm").lower()
        if mode == "both":
            raise ValueError("route_mode='both'는 모드별로 두 번 호출하세요.")
        scenario_base = os.path.join(self.base_path, "scenarios", self.experiment_id)
        scenario_subdir = mode if use_route_mode_subdir and mode else None
        if scenario_subdir:
            scenario_base = os.path.join(scenario_base, scenario_subdir)
        save_folder = os.path.join(scenario_base, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        print(f"  📍 좌표: ({latitude:.6f}, {longitude:.6f})")

        # 생성 파이프라인
        self.make_amb_info(
            latitude,
            longitude,
            incident_size,
            save_folder,
            save_routes_json=save_routes_json,
            route_mode=route_mode,
        )
        self.make_hospital_info(
            latitude,
            longitude,
            incident_size,
            save_folder,
            uav_size,
            save_routes_json=save_routes_json,
            route_mode=route_mode,
        )
        self.make_uav_info(latitude, longitude, incident_size, uav_size, save_folder)
        self.make_patient_info(save_folder)
        self.make_distance_Hos2Hos(save_folder)
        config_path = self.make_config_yaml(
            latitude, longitude, incident_size,
            amb_velocity, uav_velocity, total_samples,
            random_seed, save_folder, is_use_time,
            amb_handover_time, uav_handover_time, duration_coeff,
            scenario_subdir=scenario_subdir
        )
        
        elapsed = round(time.time() - start_time, 2)
        print(f"  ⏱️ 시나리오 생성 완료 ({elapsed}초, API 호출: {self.api_call_count}회)")
        print(f"CONFIG_PATH:{config_path}")
        return config_path, self.api_call_count

# CLI 실행용
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCI 시나리오 동적 생성 (크로스 환경 호환)")
    parser.add_argument("--base_path", required=True, help="프로젝트 루트 경로")
    parser.add_argument("--latitude", type=float, required=False, help="위도")
    parser.add_argument("--longitude", type=float, required=False, help="경도")
    parser.add_argument("--incident_size", type=int, default=30, help="환자 수")
    parser.add_argument("--amb_size", type=int, default=30, help="구급차 수")
    parser.add_argument("--uav_size", type=int, default=3, help="UAV 수")
    parser.add_argument("--amb_velocity", type=int, default=40, help="구급차 속도")
    parser.add_argument("--uav_velocity", type=int, default=80, help="UAV 속도")
    parser.add_argument("--total_samples", type=int, default=10, help="시뮬레이션 반복 수")
    parser.add_argument("--random_seed", type=int, default=0, help="랜덤 시드")
    parser.add_argument("--experiment_id", type=str, default=None, help="실험 ID")
    # 좌표 생성 관련
    parser.add_argument("--generate_coord", action="store_true", help="좌표 자동 생성")
    parser.add_argument("--coord_mode", choices=["korea_random", "sido"], default="korea_random", help="좌표 생성 모드")
    parser.add_argument("--sido_name", type=str, help="시도명 (coord_mode=sido일 때)")
    # 고급 옵션(ENV 또는 CLI 둘 다 허용)
    # parser.add_argument("--queue_policy", type=str, help='예: "0", "capa/2", "0.5"')
    parser.add_argument("--buffer_ratio", type=float, help="후보군 버퍼 배수 (기본 1.5)")
    parser.add_argument("--util_by_tier", type=str, help='예: "1:0.90,11:0.75,etc:0.60"')
    parser.add_argument("--hospital_max_send_coeff", type=str, default=None, help="전송계수 'a,b' 형식 (예: 1.1,1.0). 미입력시 ENV(MCI_MAX_SEND_COEFF) 또는 기본 1,1")

    # 카카오 API 관련 파라미터
    parser.add_argument("--kakao_api_key", type=str, default=None, help="카카오 모빌리티 REST API 키")
    parser.add_argument("--departure_time", type=str, default=None, help="출발시간 (YYYYMMDDHHMM 형식, 예: 202512241800)")
    parser.add_argument("--is_use_time", type=str, default=True, help="API duration 사용 여부 (true/false)")
    parser.add_argument("--amb_handover_time", type=float, default=0.0, help="구급차 환자 인계시간 (분)")
    parser.add_argument("--uav_handover_time", type=float, default=0.0, help="UAV 환자 인계시간 (분)")
    parser.add_argument("--duration_coeff", type=float, default=1.0, help="API duration 시간가중치 (기본값: 1.0)")
    parser.add_argument("--save_routes_json", action="store_true", help="Save route JSON under routes/")
    parser.add_argument("--route_mode", choices=["osrm", "kakao", "both"], default="osrm",
                        help="경로 생성 모드 (osrm, kakao, both)")
    parser.add_argument("--use_route_mode_subdir", action="store_true",
                        help="exp 폴더 아래에 경로 모드(osrm/kakao) 하위 폴더를 사용")

    args = parser.parse_args()
    try:
        # UTF-8 출력 설정
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    try:
        # is_use_time 파싱 (문자열 "true"/"false" → bool)
        is_use_time_bool = args.is_use_time.lower() in ("true", "1", "yes")

        generator = ScenarioGenerator(
            args.base_path,
            args.experiment_id,
            kakao_api_key=args.kakao_api_key,
            departure_time=args.departure_time
        )

        # CLI가 주어지면 ENV 기본값을 덮어씀
        if args.hospital_max_send_coeff:
            generator.max_send_coeff_text = args.hospital_max_send_coeff
        # if args.queue_policy is not None:
        #     generator.queue_policy = args.queue_policy
        if args.buffer_ratio is not None:
            generator.buffer_ratio = float(args.buffer_ratio)
        if args.util_by_tier:
            m = parse_util_map(args.util_by_tier)
            if m:
                generator.util_by_tier = m

        # 현재 적용값 재출력
        print(f"buffer_ratio={generator.buffer_ratio}")

        # 좌표 처리
        if args.generate_coord:
            coord_result = generator.generate_coordinate_for_scenario(args.coord_mode, args.sido_name)
            if coord_result:
                latitude, longitude = coord_result
            else:
                print("❌ 좌표 생성 실패")
                sys.exit(1)
        else:
            if args.latitude is None or args.longitude is None:
                print("❌ --latitude, --longitude 인자가 필요합니다.")
                sys.exit(1)
            latitude, longitude = args.latitude, args.longitude
        
        # 시나리오 생성
        if args.route_mode == "both":
            config_paths = {}
            api_calls_total = 0
            for mode in ("osrm", "kakao"):
                config_path, api_calls = generator.generate_scenario(
                    latitude, longitude,
                    args.incident_size, args.amb_size, args.uav_size,
                    args.amb_velocity, args.uav_velocity,
                    args.total_samples, args.random_seed,
                    is_use_time=is_use_time_bool,
                    amb_handover_time=args.amb_handover_time,
                    uav_handover_time=args.uav_handover_time,
                    duration_coeff=args.duration_coeff,
                    save_routes_json=args.save_routes_json,
                    route_mode=mode,
                    use_route_mode_subdir=True
                )
                config_paths[mode] = config_path
                api_calls_total += api_calls

            print(f"\n✅ 시나리오 생성 성공!")
            for mode, path in config_paths.items():
                print(f"📄 Config({mode}): {path}")
            print(f"📊 API 호출 횟수(합계): {api_calls_total}")
        else:
            config_path, api_calls = generator.generate_scenario(
                latitude, longitude,
                args.incident_size, args.amb_size, args.uav_size,
                args.amb_velocity, args.uav_velocity,
                args.total_samples, args.random_seed,
                is_use_time=is_use_time_bool,
                amb_handover_time=args.amb_handover_time,
                uav_handover_time=args.uav_handover_time,
                duration_coeff=args.duration_coeff,
                save_routes_json=args.save_routes_json,
                route_mode=args.route_mode,
                use_route_mode_subdir=args.use_route_mode_subdir
            )

            if config_path:
                print(f"\n✅ 시나리오 생성 성공!")
                print(f"📄 Config 파일: {config_path}")
                print(f"📊 API 호출 횟수: {api_calls}")
            else:
                print("❌ 시나리오 생성 실패")
                sys.exit(1)
            
    except Exception as e:
        print(f"💥 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# -*- coding: utf-8 -*-
"""
orchestrator.py (robust unpack patch)
- Prevents "too many values to unpack" by normalizing summary path return to exactly two strings.
"""

import os
import csv
import json
import time
import shutil
import subprocess
import requests
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta

try:
    import yaml  # pyyaml
except Exception:
    yaml = None
try:
    import pandas as pd
except Exception:
    pd = None

KST = timezone(timedelta(hours=9))

def now_kst_iso() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

def ts_short_now() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def exists_file(path: Optional[str]) -> bool:
    try:
        return bool(path) and os.path.isfile(path)
    except Exception:
        return False

def write_text(path: str, text: str, encoding: str = "utf-8"):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding=encoding, errors="ignore") as f:
        f.write(text)

def append_text(path: str, text: str, encoding: str = "utf-8"):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding=encoding, errors="ignore") as f:
        f.write(text)

def to_coord_str(lat: float, lon: float) -> str:
    return f"({lat},{lon})"

def parse_coord_from_config_path(config_path: str) -> Optional[str]:
    try:
        parts = os.path.normpath(config_path).split(os.sep)
        for i in range(len(parts)-1, -1, -1):
            if parts[i].startswith("(") and parts[i].endswith(")"):
                return parts[i]
        return None
    except Exception:
        return None

def parse_exp_from_config_path(config_path: str) -> Optional[str]:
    try:
        parts = os.path.normpath(config_path).split(os.sep)
        if "scenarios" in parts:
            ix = parts.index("scenarios")
            if ix+1 < len(parts):
                return parts[ix+1]
        return None
    except Exception:
        return None

def parse_make_generator_stdout(stdout_text: str):
    coord_info = None
    config_path = None
    for line in stdout_text.splitlines():
        s = line.strip()
        if s.startswith("COORDINATE_INFO:"):
            try:
                js = s.split("COORDINATE_INFO:",1)[1].strip()
                coord_info = json.loads(js)
            except Exception:
                pass
        elif s.startswith("CONFIG_PATH:"):
            config_path = s.split("CONFIG_PATH:",1)[1].strip()
    return coord_info, config_path

# ------------------------------------------------------------------
# Reverse Geocoding (Kakao API)
# ------------------------------------------------------------------

def reverse_geocode_kakao(lat: float, lon: float, api_key: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    카카오 로컬 API를 사용한 좌표 → 주소 변환 (역지오코딩)

    Args:
        lat: 위도
        lon: 경도
        api_key: 카카오 REST API 키
        max_retries: 최대 재시도 횟수

    Returns:
        {
            "full_address": "지번 주소",
            "road_address": "도로명 주소",
            "area1": "시도",
            "area2": "시군구",
            "area3": "읍면동",
            "area4": "리",
            "is_valid": True/False,
            "latitude": lat,
            "longitude": lon,
            "api_response_code": 0 (성공) or -999 (실패)
        }
    """
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {
        "Authorization": f"KakaoAK {api_key}"
    }
    params = {
        "x": str(lon),  # 경도
        "y": str(lat),  # 위도
        "input_coord": "WGS84"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])

                if documents and len(documents) > 0:
                    doc = documents[0]

                    # 지번 주소
                    address = doc.get("address", {})
                    area1 = address.get("region_1depth_name", "")  # 시도
                    area2 = address.get("region_2depth_name", "")  # 구
                    area3 = address.get("region_3depth_name", "")  # 동
                    area4 = address.get("region_3depth_h_name", "")  # 리 (법정동 기준)

                    # 지번 주소 조합
                    full_address = address.get("address_name", "")

                    # 도로명 주소
                    road_address_obj = doc.get("road_address")
                    road_address = ""
                    if road_address_obj:
                        road_address = road_address_obj.get("address_name", "")
                    else:
                        print(f"  ℹ️ 도로명주소 없음 (지번주소만 존재): {full_address}")

                    return {
                        "full_address": full_address,
                        "road_address": road_address,
                        "area1": area1,
                        "area2": area2,
                        "area3": area3,
                        "area4": area4,
                        "is_valid": True,
                        "latitude": lat,
                        "longitude": lon,
                        "api_response_code": 0
                    }

                # 주소 정보 없음 (해상, 산악 등)
                return {
                    "full_address": "주소 정보 없음",
                    "road_address": "",
                    "area1": "해상/미상",
                    "area2": "",
                    "area3": "",
                    "area4": "",
                    "is_valid": False,
                    "latitude": lat,
                    "longitude": lon,
                    "api_response_code": -1
                }

            elif response.status_code == 401:
                print(f"❌ 카카오 API 인증 실패 (401): API 키 확인 필요")
                break

            elif response.status_code == 429:
                print(f"⚠️ API 요청 한도 초과 (429). {attempt + 1}/{max_retries} 재시도...")
                time.sleep(2)

            else:
                print(f"❌ API 오류: {response.status_code}")
                break

        except Exception as e:
            print(f"❌ Reverse geocoding 오류: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    # 실패 시
    return {
        "full_address": "API 오류",
        "road_address": "",
        "area1": "오류",
        "area2": "",
        "area3": "",
        "area4": "",
        "is_valid": False,
        "latitude": lat,
        "longitude": lon,
        "api_response_code": -999
    }

# ============================================================
# [DEPRECATED] 네이버 API 역지오코딩 (참고용)
# ============================================================
# def reverse_geocode_naver(lat: float, lon: float, client_id: str, client_secret: str, max_retries: int = 3) -> Dict[str, Any]:
#     """
#     네이버 Reverse Geocoding API (기존 코드 - 참고용)
#
#     이 함수는 더 이상 사용되지 않습니다. 카카오 API로 마이그레이션되었습니다.
#     참고: https://maps.apigw.ntruss.com/map-reversegeocode/v2/gc
#     """
#     url = "https://maps.apigw.ntruss.com/map-reversegeocode/v2/gc"
#     headers = {
#         "X-NCP-APIGW-API-KEY-ID": client_id,
#         "X-NCP-APIGW-API-KEY": client_secret
#     }
#     params = {
#         "coords": f"{lon},{lat}",
#         "orders": "legalcode,admcode,addr,roadaddr",
#         "output": "json"
#     }
#     # ... (기존 로직 생략)

# ------------------------------------------------------------------
# Summary CSV helpers
# ------------------------------------------------------------------

SUMMARY_COLS = [
    "실험ID","좌표","시도번호","비고",
    "위도","경도","주소","도로명주소",
    "시나리오생성_시작","시나리오생성_소요(초)",
    "시뮬레이션_시작","시뮬레이션_소요(초)","실험_완료시간","좌표별_총소요(초)","성공여부","로그파일",
    "환자수","max_send_coeff","구급차수","UAV수","구급차속도","UAV속도","구급차인계시간","UAV인계시간","API_duration사용","duration_coeff","시뮬레이션반복","랜덤시드",
]
# (추가) 안정적인 dtype 스키마
SUMMARY_DTYPES = {
    "실험ID":"object","좌표":"object","시도번호":"Int64","비고":"object",
    "위도":"float64","경도":"float64","주소":"object","도로명주소":"object",
    "시나리오생성_시작":"object","시나리오생성_소요(초)":"float64",
    "시뮬레이션_시작":"object","시뮬레이션_소요(초)":"float64",
    "실험_완료시간":"object","좌표별_총소요(초)":"float64",
    "성공여부":"object","로그파일":"object",
    "환자수":"Int64","max_send_coeff":"object","구급차수":"Int64","UAV수":"Int64",
    "구급차속도":"float64","UAV속도":"float64","구급차인계시간":"float64","UAV인계시간":"float64","API_duration사용":"object","duration_coeff":"float64","시뮬레이션반복":"Int64","랜덤시드":"Int64",
}


def _summary_paths(base_path: str, exp_id: str):
    """(NOTE) Some older copies might have returned a single string or 3 items.
    We keep this signature for backward-compat, but normalize it via _summary_paths_pair().
    """
    base = os.path.join(base_path, "scenarios", exp_id)
    main = os.path.join(base, f"{exp_id}_summary.csv")
    legacy = os.path.join(base, "실험ID_summary.csv")
    return (main, legacy)

def _summary_paths_pair(base_path: str, exp_id: str) -> Tuple[str, str]:
    """Always return exactly two strings, no matter what _summary_paths returns in user's local file."""
    p = _summary_paths(base_path, exp_id)
    base = os.path.join(base_path, "scenarios", exp_id)
    main_default = os.path.join(base, f"{exp_id}_summary.csv")
    legacy_default = os.path.join(base, "실험ID_summary.csv")

    # tuple/list
    if isinstance(p, (tuple, list)):
        if len(p) >= 2:
            return str(p[0]), str(p[1])
        elif len(p) == 1:
            return str(p[0]), legacy_default
        else:
            return main_default, legacy_default
    # single string?
    if isinstance(p, str):
        # if it's already "<exp>_summary.csv", compute legacy alongside
        if p.endswith("_summary.csv") and os.path.basename(p).startswith(exp_id):
            return p, legacy_default
        # else assume it's legacy-like, compute main
        return main_default, p
    # fallback
    return main_default, legacy_default

def _load_summary_df(path_main: str, path_legacy: str):
    if pd is None:
        return None
    for pth in (path_main, path_legacy):
        if exists_file(pth):
            for enc in ("utf-8-sig","cp949","utf-8"):
                try:
                    df = pd.read_csv(pth, encoding=enc)
                    # 누락 컬럼 보충 + 순서 정렬
                    for c in SUMMARY_COLS:
                        if c not in df.columns:
                            df[c] = pd.Series(dtype=SUMMARY_DTYPES.get(c, "object"))
                    df = df.reindex(columns=SUMMARY_COLS)
                    # dtype 강제(일괄 → 실패시 컬럼 단위 보정)
                    try:
                        df = df.astype(SUMMARY_DTYPES)
                    except Exception:
                        for col, dt in SUMMARY_DTYPES.items():
                            try:
                                df[col] = df[col].astype(dt)
                            except Exception:
                                df[col] = df[col].astype("object")
                    return df
                except Exception:
                    continue
    import pandas as _pd
    # 빈 DF도 dtype 보장
    empty = {c: _pd.Series(dtype=SUMMARY_DTYPES.get(c, "object")) for c in SUMMARY_COLS}
    return _pd.DataFrame(empty)


def _save_summary(path_main: str, df):
    ensure_dir(os.path.dirname(path_main))
    df[SUMMARY_COLS].to_csv(path_main, index=False, encoding="utf-8-sig")

def upsert_summary_row_dual(base_path: str, row: Dict[str,Any]):
    exp_id = row.get("실험ID")
    if not exp_id:
        return
    path_main, path_legacy = _summary_paths_pair(base_path, exp_id)
    if pd is None:
        exists = exists_file(path_main)
        ensure_dir(os.path.dirname(path_main))
        with open(path_main, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k,"") for k in SUMMARY_COLS})
        return

    df = _load_summary_df(path_main, path_legacy)
    key = (row.get("실험ID"), row.get("좌표"))

    # 시도번호 자동 계산: 같은 실험ID + 좌표 조합의 최대 시도번호 + 1
    existing = df[(df["실험ID"]==key[0]) & (df["좌표"]==key[1])]
    if not existing.empty:
        max_trial = existing["시도번호"].max()
        next_trial = 1 if pd.isna(max_trial) else int(max_trial) + 1
    else:
        next_trial = 1

    # 시도번호가 명시되지 않았으면 자동 할당
    if "시도번호" not in row or row.get("시도번호") is None or row.get("시도번호") == 0:
        row["시도번호"] = next_trial

    # 항상 새로운 행으로 추가 (업데이트 하지 않음)
    for c in SUMMARY_COLS:
        if c not in df.columns:
            df[c] = pd.Series(dtype=SUMMARY_DTYPES.get(c, "object"))
    df = df.reindex(columns=SUMMARY_COLS)

    next_idx = len(df)
    for k, v in row.items():
        df.loc[next_idx, k] = v

    _save_summary(path_main, df)

# ------------------------------------------------------------------
# YAML parsing
# ------------------------------------------------------------------

def _yaml_safe_load(path: str) -> Dict[str,Any]:
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def extract_params_from_yaml(config_path: str) -> Dict[str,Any]:
    meta = {
        "환자수": None,
        "구급차수": None,
        "UAV수": None,
        "구급차속도": None,
        "UAV속도": None,
        "구급차인계시간": None,
        "UAV인계시간": None,
        "API_duration사용": None,
        "duration_coeff": None,
        "시뮬레이션반복": None,
        "랜덤시드": None,
        "max_send_coeff": None,
    }
    y = _yaml_safe_load(config_path) or {}
    run = y.get("run_setting", {}) or {}
    meta["시뮬레이션반복"] = run.get("totalSamples")
    meta["랜덤시드"] = run.get("random_seed")

    ent = y.get("entity_info", {}) or {}
    patient = ent.get("patient", {}) or {}
    amb = ent.get("ambulance", {}) or {}
    uav = ent.get("uav", {}) or {}
    hosp = ent.get("hospital", {}) or {}

    # incident_size / speeds / handover_time / is_use_time / duration_coeff
    meta["환자수"] = patient.get("incident_size")
    meta["구급차속도"] = amb.get("velocity")
    meta["UAV속도"] = uav.get("velocity")
    meta["구급차인계시간"] = amb.get("handover_time")
    meta["UAV인계시간"] = uav.get("handover_time")

    # API duration 사용 여부 (True/False를 문자열로 저장)
    is_use_time_val = amb.get("is_use_time")
    if is_use_time_val is not None:
        meta["API_duration사용"] = str(is_use_time_val)
    else:
        meta["API_duration사용"] = None

    # duration_coeff (API duration 시간가중치)
    duration_coeff_val = amb.get("duration_coeff")
    if duration_coeff_val is not None:
        meta["duration_coeff"] = float(duration_coeff_val)
    else:
        meta["duration_coeff"] = 1.0  # 기본값

    # max_send_coeff (리스트/문자열 모두 처리)
    msc = hosp.get("max_send_coeff")
    if isinstance(msc, (list, tuple)):
        # 리스트 안에 "1.1, 1" 같은 문자열만 있는 경우도 대비
        if len(msc) == 1 and isinstance(msc[0], str) and "," in msc[0]:
            meta["max_send_coeff"] = msc[0]
        else:
            try:
                meta["max_send_coeff"] = ",".join(str(x) for x in msc)
            except Exception:
                meta["max_send_coeff"] = str(msc)
    elif isinstance(msc, str):
        meta["max_send_coeff"] = msc

    # CSV 경로 해석 헬퍼 (./scenarios/... 는 프로젝트 루트 기준, 그 외는 config 폴더 기준)
    def _resolve(p: Optional[str]) -> Optional[str]:
        if not p: return None
        p = str(p).strip()
        if os.path.isabs(p):
            return p
        if p.startswith("./"):
            parts = os.path.normpath(config_path).split(os.sep)
            if "scenarios" in parts:
                root = os.sep.join(parts[:parts.index("scenarios")])  # 프로젝트 루트
                return os.path.normpath(os.path.join(root, p[2:]))
        # 일반 상대경로는 config가 있는 폴더 기준
        return os.path.normpath(os.path.join(os.path.dirname(config_path), p))

    # 구급차수/UAV수는 CSV 행 수로
    amb_csv = _resolve(amb.get("dispatch_distance_info"))
    uav_csv = _resolve(uav.get("dispatch_distance_info"))
    try:
        if amb_csv and os.path.isfile(amb_csv) and pd is not None:
            meta["구급차수"] = len(pd.read_csv(amb_csv, encoding="utf-8-sig"))
    except Exception:
        pass
    try:
        if uav_csv and os.path.isfile(uav_csv) and pd is not None:
            # UAV 대수: 전체 UAV CSV 행 수를 상급종합병원 수로 나눔
            uav_df = pd.read_csv(uav_csv, encoding="utf-8-sig")
            total_uav_count = len(uav_df)

            # 상급종합병원 수 계산 (hospital_info에서 종별코드=1)
            hosp_info_csv = _resolve(hosp.get("info_path"))
            if hosp_info_csv and os.path.isfile(hosp_info_csv):
                hosp_df = pd.read_csv(hosp_info_csv, encoding="utf-8-sig")
                if "종별코드" in hosp_df.columns:
                    tertiary_hospital_count = (hosp_df["종별코드"] == 1).sum()
                    if tertiary_hospital_count > 0:
                        meta["UAV수"] = total_uav_count // tertiary_hospital_count
                    else:
                        meta["UAV수"] = total_uav_count
                else:
                    meta["UAV수"] = total_uav_count
            else:
                meta["UAV수"] = total_uav_count
    except Exception:
        pass

    return meta


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------
import sys

class Orchestrator:
    def __init__(self, base_path: str, python_cmd: Optional[str] = None):
        self.base_path = os.path.abspath(base_path)
        self.python_cmd = python_cmd or sys.executable
        self.paths = {
            "make_script": os.path.join(self.base_path, "make_csv_yaml_dynamic.py"),
            "main_py":     os.path.join(self.base_path, "main.py"),
            "scenarios":   os.path.join(self.base_path, "scenarios"),
            "results":     os.path.join(self.base_path, "results"),
            "logs":        os.path.join(self.base_path, "experiment_logs"),
        }

    # ---------- scenario generation ----------
    def generate_scenario(self, latitude: float, longitude: float,
                          incident_size: int = 30, amb_size: int = 30, uav_size: int = 3,
                          amb_velocity: int = 40, uav_velocity: int = 80,
                          total_samples: int = 10, random_seed: int = 0,
                          exp_id: Optional[str] = None,
                          extra_env: Optional[Dict[str,str]] = None,
                          extra_args: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:

        if not exists_file(self.paths["make_script"]):
            raise FileNotFoundError(f"make_csv_yaml_dynamic.py not found: {self.paths['make_script']}")
        # exp_YYYYMMDDHHMM 형식 (언더스코어 제거)
        exp_id = exp_id or ("exp_" + datetime.now(KST).strftime("%Y%m%d%H%M"))

        cmd = [
            self.python_cmd, "-X", "utf8", self.paths["make_script"],
            "--base_path", self.base_path,
            "--latitude", str(latitude),
            "--longitude", str(longitude),
            "--incident_size", str(incident_size),
            "--amb_size", str(amb_size),
            "--uav_size", str(uav_size),
            "--amb_velocity", str(amb_velocity),
            "--uav_velocity", str(uav_velocity),
            "--total_samples", str(total_samples),
            "--random_seed", str(random_seed),
            "--experiment_id", exp_id,
        ]

        if extra_args:
            for k, v in extra_args.items():
                if v is None: 
                    continue
                cmd.extend([f"--{k}", str(v)])

        env = os.environ.copy()
        if extra_env:
            env.update({str(k):str(v) for k,v in extra_env.items()})

        t1 = time.time()
        started_at = now_kst_iso()
        proc = subprocess.run(cmd, cwd=self.base_path, env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="ignore")
        t2 = time.time()
        elapsed = round(t2 - t1, 3)

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        _, config_path = parse_make_generator_stdout(stdout)  # coord_info는 무시
        if not config_path:
            raise RuntimeError(f"CONFIG_PATH not found in generator stdout.\n[stdout]\n{stdout}\n[stderr]\n{stderr}")

        coord = parse_coord_from_config_path(config_path) or to_coord_str(latitude, longitude)
        exp_id2 = parse_exp_from_config_path(config_path) or exp_id

        # 역지오코딩 직접 수행 (카카오 API)
        coord_info = None
        kakao_api_key = (extra_args or {}).get("kakao_api_key")
        if kakao_api_key:
            try:
                coord_info = reverse_geocode_kakao(latitude, longitude, kakao_api_key)
                print(f"✅ 역지오코딩 성공: {coord_info.get('full_address', '')}")
            except Exception as e:
                print(f"⚠️ 역지오코딩 실패: {e}")
                coord_info = {
                    "full_address": "역지오코딩 실패",
                    "road_address": "",
                    "area1": "", "area2": "", "area3": "", "area4": "",
                    "is_valid": False,
                    "latitude": latitude,
                    "longitude": longitude,
                    "api_response_code": -999
                }
        else:
            print("⚠️ 카카오 API 키 없음. 역지오코딩 생략")
            coord_info = {
                "full_address": "",
                "road_address": "",
                "area1": "", "area2": "", "area3": "", "area4": "",
                "is_valid": False,
                "latitude": latitude,
                "longitude": longitude,
                "api_response_code": -999
            }

        # Summary path (main + legacy) — robust
        summary_main, summary_legacy = _summary_paths_pair(self.base_path, exp_id2)

        # Log file name: (coord)_%y%m%d_%H%M%S.txt
        log_file = os.path.join(self.paths["logs"], f"{coord}_{ts_short_now()}.txt")
        ensure_dir(os.path.dirname(log_file))
        log_head = []
        log_head.append(f"=== SCENARIO_GEN_START {started_at} ===\n")
        if coord_info is not None:
            log_head.append(f"COORDINATE_INFO:{json.dumps(coord_info, ensure_ascii=False)}\n")
        log_head.append(f"CONFIG_PATH:{config_path}\n")
        log_head.append(f"--- stdout ---\n{stdout}\n")
        if stderr.strip():
            log_head.append(f"--- stderr ---\n{stderr}\n")
        log_head.append(f"=== SCENARIO_GEN_END {now_kst_iso()} (elapsed: {elapsed}s) ===\n\n")
        write_text(log_file, "".join(log_head), encoding="utf-8")

        # Initial summary row (attempt not assigned yet → 0)
        row = {
            "실험ID": exp_id2,
            "좌표": coord,
            "시도번호": 0,
            "비고": "시나리오 생성",
            "위도": coord_info.get("latitude") if coord_info else latitude,
            "경도": coord_info.get("longitude") if coord_info else longitude,
            "주소": (coord_info or {}).get("full_address",""),
            "도로명주소": (coord_info or {}).get("road_address",""),
            "시나리오생성_시작": started_at,
            "시나리오생성_소요(초)": elapsed,
            "시뮬레이션_시작": None,
            "시뮬레이션_소요(초)": None,
            "실험_완료시간": None,
            "좌표별_총소요(초)": None,
            "성공여부": None,
            "로그파일": log_file,
        }
        meta = extract_params_from_yaml(config_path)
        row.update({
            "환자수": meta.get("환자수"),
            "max_send_coeff": meta.get("max_send_coeff"),
            "구급차수": meta.get("구급차수"),
            "UAV수": meta.get("UAV수"),
            "구급차속도": meta.get("구급차속도"),
            "UAV속도": meta.get("UAV속도"),
            "구급차인계시간": meta.get("구급차인계시간"),
            "UAV인계시간": meta.get("UAV인계시간"),
            "API_duration사용": meta.get("API_duration사용"),
            "duration_coeff": meta.get("duration_coeff"),
            "시뮬레이션반복": meta.get("시뮬레이션반복"),
            "랜덤시드": meta.get("랜덤시드"),
        })
        upsert_summary_row_dual(self.base_path, row)

        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "exp_id": exp_id2,
            "coord": coord,
            "config_path": config_path,
            "summary_csv_path": summary_main,
            "summary_csv_path_legacy": summary_legacy,
            "log_file": log_file,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_sec": elapsed,
            "started_at": started_at,
        }

    # ---------- simulation run ----------
    def run_simulation(self, config_path: str, extra_env: Optional[Dict[str,str]] = None) -> Dict[str,Any]:

        if not exists_file(self.paths["main_py"]):
            raise FileNotFoundError(f"main.py not found: {self.paths['main_py']}")
        if not exists_file(config_path):
            raise FileNotFoundError(f"config_path not found: {config_path}")

        exp_id2 = parse_exp_from_config_path(config_path)
        coord2 = parse_coord_from_config_path(config_path)
        if not exp_id2 or not coord2:
            raise ValueError("exp_id/coord를 config_path에서 추출할 수 없습니다.")

        summary_main, summary_legacy = _summary_paths_pair(self.base_path, exp_id2)

        cmd = [
            self.python_cmd, "-X", "utf8", self.paths["main_py"],
            "--config_path", config_path,
        ]
        env = os.environ.copy()
        if extra_env:
            env.update({str(k):str(v) for k,v in extra_env.items()})

        sim_started = now_kst_iso()
        t1 = time.time()
        proc = subprocess.run(cmd, cwd=self.base_path, env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="ignore")
        t2 = time.time()
        elapsed = round(t2 - t1, 3)
        ok = (proc.returncode == 0)

        # Per-run log file
        log_file = os.path.join(self.paths["logs"], f"{coord2}_{ts_short_now()}.txt")
        pieces = []
        pieces.append(f"=== SIM_START {sim_started} ===\n")
        if proc.stdout:
            pieces.append(proc.stdout)
            if not proc.stdout.endswith("\n"):
                pieces.append("\n")
        if proc.stderr and proc.stderr.strip():
            pieces.append(f"--- stderr ---\n{proc.stderr}\n")
        pieces.append(f"=== SIM_END {now_kst_iso()} (elapsed: {elapsed}s, rc={proc.returncode}) ===\n\n")
        write_text(log_file, "".join(pieces), encoding="utf-8")

        # Summary update (attempt-aware)
        if pd is not None:
            df = _load_summary_df(summary_main, summary_legacy)
            mask_pair = (df["실험ID"]==exp_id2) & (df["좌표"]==coord2)

            def _base_fields_for_pair() -> Dict[str,Any]:
                # 항상 YAML에서 최신 파라미터를 읽어옴 (재실행 시 수정된 파라미터 반영)
                meta = extract_params_from_yaml(config_path)

                if mask_pair.any():
                    d0 = df[mask_pair].sort_index().iloc[0].to_dict()
                    return {
                        "실험ID": exp_id2, "좌표": coord2,
                        "위도": d0.get("위도"), "경도": d0.get("경도"),
                        "주소": d0.get("주소"), "도로명주소": d0.get("도로명주소"),
                        "시나리오생성_시작": d0.get("시나리오생성_시작"),
                        "시나리오생성_소요(초)": d0.get("시나리오생성_소요(초)"),
                        # YAML에서 읽은 최신 파라미터 사용 (수정된 값 반영)
                        "환자수": meta.get("환자수"),
                        "max_send_coeff": meta.get("max_send_coeff"),
                        "구급차수": meta.get("구급차수"),
                        "UAV수": meta.get("UAV수"),
                        "구급차속도": meta.get("구급차속도"),
                        "UAV속도": meta.get("UAV속도"),
                        "구급차인계시간": meta.get("구급차인계시간"),
                        "UAV인계시간": meta.get("UAV인계시간"),
                        "API_duration사용": meta.get("API_duration사용"),
                        "duration_coeff": meta.get("duration_coeff"),
                        "시뮬레이션반복": meta.get("시뮬레이션반복"),
                        "랜덤시드": meta.get("랜덤시드"),
                    }
                else:
                    return {
                        "실험ID": exp_id2, "좌표": coord2,
                        "위도": None, "경도": None, "주소": "", "도로명주소": "",
                        "시나리오생성_시작": None, "시나리오생성_소요(초)": None,
                        "환자수": meta.get("환자수"),
                        "max_send_coeff": meta.get("max_send_coeff"),
                        "구급차수": meta.get("구급차수"),
                        "UAV수": meta.get("UAV수"),
                        "구급차속도": meta.get("구급차속도"),
                        "UAV속도": meta.get("UAV속도"),
                        "구급차인계시간": meta.get("구급차인계시간"),
                        "UAV인계시간": meta.get("UAV인계시간"),
                        "API_duration사용": meta.get("API_duration사용"),
                        "duration_coeff": meta.get("duration_coeff"),
                        "시뮬레이션반복": meta.get("시뮬레이션반복"),
                        "랜덤시드": meta.get("랜덤시드"),
                    }

            # 항상 새로운 행으로 추가 (재실험시에도 시도번호 증가)
            mask_first = mask_pair & (df["시도번호"].fillna(0)==0)
            if mask_first.any():
                # 첫 실행: 시도번호 0을 1로 업데이트
                # YAML에서 최신 파라미터를 읽어서 업데이트 (수정된 값 반영)
                meta_first = extract_params_from_yaml(config_path)
                df.loc[mask_first, "시도번호"] = 1
                df.loc[mask_first, "비고"] = "1차시도 " + ("성공" if ok else "실패")
                df.loc[mask_first, "시뮬레이션_시작"] = sim_started
                df.loc[mask_first, "시뮬레이션_소요(초)"] = elapsed
                try:
                    prev = float(df.loc[mask_first, "시나리오생성_소요(초)"].values[0] or 0.0)
                except Exception:
                    prev = 0.0
                df.loc[mask_first, "좌표별_총소요(초)"] = round(prev + float(elapsed), 3)
                df.loc[mask_first, "실험_완료시간"] = now_kst_iso()
                df.loc[mask_first, "성공여부"] = bool(ok)
                df.loc[mask_first, "로그파일"] = log_file
                # YAML에서 읽은 최신 파라미터 반영
                df.loc[mask_first, "환자수"] = meta_first.get("환자수")
                df.loc[mask_first, "max_send_coeff"] = meta_first.get("max_send_coeff")
                df.loc[mask_first, "구급차수"] = meta_first.get("구급차수")
                df.loc[mask_first, "UAV수"] = meta_first.get("UAV수")
                df.loc[mask_first, "구급차속도"] = meta_first.get("구급차속도")
                df.loc[mask_first, "UAV속도"] = meta_first.get("UAV속도")
                df.loc[mask_first, "구급차인계시간"] = meta_first.get("구급차인계시간")
                df.loc[mask_first, "UAV인계시간"] = meta_first.get("UAV인계시간")
                df.loc[mask_first, "API_duration사용"] = meta_first.get("API_duration사용")
                df.loc[mask_first, "duration_coeff"] = meta_first.get("duration_coeff")
                df.loc[mask_first, "시뮬레이션반복"] = meta_first.get("시뮬레이션반복")
                df.loc[mask_first, "랜덤시드"] = meta_first.get("랜덤시드")

            else:
                # 재실험: 항상 새로운 행 추가 (성공 여부와 상관없이)
                import pandas as _pd
                prev_rows = df[mask_pair].sort_values("시도번호")
                next_try = int(_pd.to_numeric(prev_rows["시도번호"], errors="coerce").fillna(0).max()) + 1 if not prev_rows.empty else 1
                base = _base_fields_for_pair()
                newrow = {
                    **base,
                    "시도번호": next_try,
                    "비고": f"{next_try}차시도 " + ("성공" if ok else "실패"),
                    "시뮬레이션_시작": sim_started,
                    "시뮬레이션_소요(초)": elapsed,
                    "실험_완료시간": now_kst_iso(),
                    "좌표별_총소요(초)": None,
                    "성공여부": bool(ok),
                    "로그파일": log_file,
                }
                try:
                    prev = float(base.get("시나리오생성_소요(초)") or 0.0)
                except Exception:
                    prev = 0.0
                newrow["좌표별_총소요(초)"] = round(prev + float(elapsed), 3)
                # concat 없이 행 단위 추가
                for c in SUMMARY_COLS:
                    if c not in df.columns:
                        df[c] = pd.Series(dtype=SUMMARY_DTYPES.get(c, "object"))
                df = df.reindex(columns=SUMMARY_COLS)

                next_idx = len(df)
                for k, v in newrow.items():
                    df.loc[next_idx, k] = v


            _save_summary(summary_main, df)

        return {
            "ok": ok,
            "returncode": proc.returncode,
            "exp_id": exp_id2,
            "coord": coord2,
            "config_path": config_path,
            "summary_csv_path": summary_main,
            "summary_csv_path_legacy": summary_legacy,
            "log_file": log_file,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "elapsed_sec": elapsed,
            "started_at": sim_started,
        }
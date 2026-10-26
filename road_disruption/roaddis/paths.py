"""
생성물 저장 경로, 로컬 osrm-korea 저장 경로
"""
from __future__ import annotations

import os
from pathlib import Path

# road_disruption/ 자체
PKG_DIR = Path(__file__).resolve().parent.parent
# 저장소 루트 (MCI_Diffusion/)
REPO_ROOT = PKG_DIR.parent

CONFIG_DIR = PKG_DIR / "config"

# 생성물(인덱스, 지역 pbf, cut CSV, 실험 결과)
DATA_DIR = Path(os.environ.get("ROADDIS_DATA_DIR", PKG_DIR / "data"))

# OSRM 원본/빌드 산출물이 있는 호스트 디렉토리 (docker 볼륨 소스)
OSRM_DIR = Path(os.environ.get("ROADDIS_OSRM_DIR", r"C:\Users\user00\osrm-korea"))
# 위 디렉토리가 컨테이너 안에서 보이는 경로
OSRM_CONTAINER_DIR = "/data"

OSRM_DOCKER_IMAGE = "osrm/osrm-backend"

# MCI_ADV2 시뮬레이터 — 다른 conda 환경(MCI)에서 돌아간다.
# road 환경에는 pyosmium 이, MCI 환경에는 시뮬레이터 의존성이 있어 서로 겹치지 않는다.
# 합치지 않고 subprocess 로 넘는다 (osmium·docker 를 부르는 것과 같은 방식).
MCI_ADV2_DIR = REPO_ROOT / "MCI_ADV2"
MCI_PYTHON = Path(os.environ.get(
    "ROADDIS_MCI_PYTHON", r"C:\Users\user00\anaconda3\envs\MCI\python.exe"))


def data_subdir(*parts: str) -> Path:
    """DATA_DIR 아래 하위 디렉토리를 만들고 반환."""
    p = DATA_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p

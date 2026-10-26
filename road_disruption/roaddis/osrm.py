"""
OSRM docker 제어 + 라우팅 클라이언트.

전국 인스턴스가 만들어진 방식을 그대로 따른다(4개월 전 빌드 이력에서 확인):
    osrm-extract   -p /opt/car.lua  /data/<name>.osm.pbf
    osrm-partition                  /data/<name>.osrm
    osrm-customize                  /data/<name>.osrm
    osrm-routed --algorithm mld     /data/<name>.osrm

프로파일(car.lua)이 다르면 소요시간 자체가 달라져 전국 baseline 과 비교할 수 없다.

차단 방식:
    pbf 에서 대상 way 를 삭제하고 별도 인스턴스를 통째로 새로 빌드한다.
    대전 기준 extract+partition+customize 전체가 9초라 충분히 싸고,
    baseline 과 차단본이 서로 다른 포트에 동시에 살아 있어 reset 이 필요 없다.
"""
from __future__ import annotations

import subprocess
import time
from typing import Optional, Sequence

import requests

from .paths import OSRM_CONTAINER_DIR, OSRM_DIR, OSRM_DOCKER_IMAGE

CAR_PROFILE = "/opt/car.lua"


# ----------------------------------------------------------------------
# docker
# ----------------------------------------------------------------------
def _volume() -> str:
    return f"{OSRM_DIR}:{OSRM_CONTAINER_DIR}"


# 빌드 한 단계의 상한(초). 대전 한 지역은 9초면 끝난다 — 300초를 넘겼다면
# 정상적으로 느린 게 아니라 매달린 것이다. sweep 처럼 수백 번 반복하는 무인
# 실행에서 이 상한이 없으면 한 번의 멈춤이 전체를 세운다.
BUILD_TIMEOUT = 300.0


def docker_run_once(args: Sequence[str], *, quiet: bool = False,
                    name: Optional[str] = None,
                    timeout: float = BUILD_TIMEOUT) -> None:
    """
    일회성 컨테이너로 osrm-* 명령 실행.

    --rm 을 붙였지만 **명령이 실패하면 컨테이너가 남는 경우가 있다.** 실제로
    osrm-customize 가 입력 파일을 못 찾아 에러를 낸 채 running 으로 매달린 사례가
    있었다. name 을 주면 실행 전에 같은 이름을 정리하므로 잔재가 쌓이지 않고,
    실패 시 `docker logs <name>` 으로 원인을 볼 수 있다.

    timeout 을 넘기면 컨테이너를 죽이고 RuntimeError 를 낸다. 그냥 예외를 올리면
    매달린 컨테이너가 살아남아 같은 이름을 쓰는 다음 반복까지 망친다.
    """
    if name:
        remove_container(name)
    cmd = ["docker", "run", "--rm"]
    if name:
        cmd += ["--name", name]
    cmd += ["-v", _volume(), OSRM_DOCKER_IMAGE, *args]
    if not quiet:
        print("      $ " + " ".join(args))
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if name:
            remove_container(name)
        raise RuntimeError(
            f"docker 가 {timeout:.0f}초를 넘겨 중단했습니다: {' '.join(args[:2])}"
        ) from None


def container_state(name: str) -> Optional[str]:
    """컨테이너 상태 문자열. 없으면 None."""
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", name],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def remove_container(name: str) -> None:
    if container_state(name) is not None:
        subprocess.run(["docker", "rm", "-f", name],
                       check=True, capture_output=True, timeout=120)


# ----------------------------------------------------------------------
# 빌드 단계
# ----------------------------------------------------------------------
def osrm_base(pbf_name: str) -> str:
    """'daejeon.osm.pbf' → '/data/daejeon.osrm'"""
    stem = pbf_name.replace(".osm.pbf", "").replace(".pbf", "")
    return f"{OSRM_CONTAINER_DIR}/{stem}.osrm"


def extract(pbf_name: str, profile: str = CAR_PROFILE,
            builder: Optional[str] = None) -> None:
    docker_run_once(["osrm-extract", "-p", profile,
                     f"{OSRM_CONTAINER_DIR}/{pbf_name}"], name=builder)


def partition(pbf_name: str, builder: Optional[str] = None) -> None:
    docker_run_once(["osrm-partition", osrm_base(pbf_name)], name=builder)


def customize(pbf_name: str, builder: Optional[str] = None) -> None:
    """가중치 계산. way 삭제 방식에서는 pbf 마다 딱 한 번만 돌린다."""
    docker_run_once(["osrm-customize", osrm_base(pbf_name)], name=builder)


def serve(pbf_name: str, port: int, name: str, *, replace: bool = True) -> None:
    """osrm-routed 를 백그라운드 컨테이너로 기동."""
    if replace:
        remove_container(name)
    cmd = [
        "docker", "run", "-d", "--name", name,
        "-p", f"{port}:5000", "-v", _volume(), OSRM_DOCKER_IMAGE,
        "osrm-routed", "--algorithm", "mld", osrm_base(pbf_name),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        remove_container(name)
        raise RuntimeError("docker run -d 가 120초를 넘겼습니다 (데몬 확인)") from None


def restart(name: str) -> None:
    """customize 로 그래프를 바꾼 뒤에는 재기동해야 반영된다."""
    subprocess.run(["docker", "restart", name], check=True, capture_output=True)


# ----------------------------------------------------------------------
# 라우팅 클라이언트
# ----------------------------------------------------------------------
def wait_ready(url: str, timeout: float = 60.0) -> bool:
    """서버가 응답할 때까지 대기. 대전 시청 근처 좌표로 헬스체크."""
    probe = f"{url}/route/v1/driving/127.38,36.35;127.42,36.33?overview=false"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(probe, timeout=3).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return False


def route(url: str, start, end, *, overview: str = "false", nodes: bool = False,
          radiuses: float = 250.0, timeout: float = 15.0) -> dict:
    """
    (lat, lon) → (lat, lon) 경로. OSRM 은 lon,lat 순서라는 점에 주의.

    nodes=True 면 annotations=nodes 로 경로가 실제로 지나간 OSM node id 열을
    받는다. 차단이 정말 먹혔는지 검증할 때 쓴다 — 끊은 node 쌍이 경로에 남아
    있으면 차단이 반영되지 않은 것이다. 소요시간만 보면 "우연히 비슷한 우회로"와
    "차단 실패"를 구분할 수 없다.

    반환: {"ok": bool, "code": str, "distance_km": float|None,
           "duration_min": float|None, "geometry": ..., "nodes": list|None}

    차단 실험에서 code == "NoRoute" 는 정상적인 결과다 — 도달 불가능해졌다는
    뜻이고, 그 자체가 측정하려는 지표다.
    """
    u = f"{url}/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
    params = {"overview": overview, "radiuses": f"{radiuses};{radiuses}",
              "continue_straight": "false"}
    if overview != "false":
        params["geometries"] = "geojson"
    if nodes:
        params["annotations"] = "nodes"

    fail = {"ok": False, "distance_km": None, "duration_min": None,
            "geometry": None, "nodes": None}
    try:
        resp = requests.get(u, params=params, timeout=timeout)
    except requests.RequestException as e:
        return {**fail, "code": f"HTTPError: {e}"}

    data = resp.json() if resp.content else {}
    code = data.get("code", f"HTTP {resp.status_code}")
    if code != "Ok" or not data.get("routes"):
        return {**fail, "code": code}

    r = data["routes"][0]
    node_seq = None
    if nodes:
        node_seq = []
        for leg in r.get("legs", []):
            node_seq.extend(leg.get("annotation", {}).get("nodes", []))

    return {
        "ok": True,
        "code": code,
        "distance_km": r["distance"] / 1000.0,
        "duration_min": r["duration"] / 60.0,
        "geometry": r.get("geometry"),
        "nodes": node_seq,
    }

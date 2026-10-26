"""경로 헬퍼 — 신·구 코어를 한 프로세스에 올리기 위한 sys.path 준비.

주의: `src/sim_src` 를 sys.path 에 넣으면 flat `import EventManager` 가 **구 코어**로
해석된다. 이는 의도된 것이다 — `sim_src/main.py` 가 그렇게 import 하고, 동치검증도
구 코어를 그 경로로 로드한다. 신 코어는 패키지 상대 import 라 전혀 간섭받지 않는다.

MCI_ADV 판: 이 저장소에는 `src/rl_src` 가 없다(RL 트랙은 MCI_UAV 전용).
`rl=True` 로 불려도 존재하지 않는 경로는 넣지 않는다.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
# 이 저장소는 업스트림의 src/ 대신 MCI_ADV2/ 레이아웃을 쓴다. sim_src_upgrade 패키지의
# 부모(= MCI_ADV2)가 sim_src/sce_src 를 담는 디렉터리다.
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SIM_SRC = os.path.join(SRC, "sim_src")
SCE_SRC = os.path.join(SRC, "sce_src")
RL_SRC = os.path.join(SRC, "rl_src")   # ADV 에는 없음 (존재할 때만 추가)


def _prepend(path: str) -> None:
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def ensure_paths(rl: bool = True, old_sim: bool = True) -> None:
    """`sim_src_upgrade` 패키지 import 경로(+ 선택적으로 rl_src / 구 sim_src)를 보장."""
    _prepend(SRC)
    if rl:
        _prepend(RL_SRC)          # ADV: 디렉터리 부재 → no-op
    if old_sim:
        _prepend(SIM_SRC)

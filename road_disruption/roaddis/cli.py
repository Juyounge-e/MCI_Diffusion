"""
헬 퍼코드

scripts/ 여러 곳에 같은 함수를 복사해 두면 규칙이 갈라진다. 실제로 slug 를
ASCII 전용으로 바꿀 때 두 파일을 따로 고쳐야 했다.
"""
from __future__ import annotations

import re


def slug(s: str, maxlen: int = 50) -> str:
    """
    파일명·docker 컨테이너 이름용 ASCII slug.

    비ASCII 는 버린다. docker 컨테이너 이름은 ASCII 만 허용하고, 한글 파일명은
    HTTP 로 서빙할 때 URL 인코딩이 필요해 다루기 번거롭다.
    그래서 --label 같은 인자는 영어로 쓴다.
    """
    out = re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z_-]+", "_", s)).strip("_")[:maxlen]
    return out or "unnamed"


def latlon(s: str) -> tuple:
    """'36.3757,127.3767' → (36.3757, 127.3767).  위도,경도 순서."""
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"좌표 형식은 'lat,lon' 입니다: {s!r}")
    try:
        lat, lon = (float(x) for x in parts)
    except ValueError as e:
        raise ValueError(f"좌표를 숫자로 읽을 수 없습니다: {s!r}") from e
    # 한국 범위를 벗어나면 lat/lon 을 뒤집어 넣은 경우가 대부분이다
    if not (32.0 <= lat <= 39.5 and 124.0 <= lon <= 132.5):
        raise ValueError(
            f"좌표가 한국 범위를 벗어납니다: lat={lat}, lon={lon}. "
            f"'위도,경도' 순서가 맞는지 확인하세요."
        )
    return lat, lon

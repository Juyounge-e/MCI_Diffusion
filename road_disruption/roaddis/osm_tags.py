"""
OSM 태그 상수.
"""

# OSRM car.lua 의 speeds.highway 와 동일 (14개)
DRIVE_HIGHWAY = frozenset({
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street", "service",
})

# 간선도로 — 차단 후보를 좁힐 때 자주 쓰는 부분집합
MAJOR_HIGHWAY = frozenset({
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
})

# selector 를 새로 만들 때 필요한 태그가 있으면 여기에 추가하고 인덱스를 다시 만든다.
KEEP_TAGS = (
    "highway",
    "oneway",
    "bridge",
    "tunnel",
    "maxspeed",
    "lanes",
    "layer", # 지하차도는 -1으로 표기 
    "name",
    "ref",
    "access",
)

# oneway 태그가 "정방향 일방통행"을 뜻하는 값
ONEWAY_FORWARD = frozenset({"yes", "true", "1"})

# oneway 태그가 "역방향 일방통행"을 뜻하는 값 (node 순서의 반대로만 통행)
ONEWAY_REVERSE = frozenset({"-1", "reverse"})

"""
pbf 에서 way 를 아예 삭제하기.

왜 이 방식인가:
    segment-speed 방식(speed 0)은 '아주 느린 도로'를 만드는 것이라, 우회로가
    없으면 OSRM 이 그 구간을 그냥 통과한다(관측: 세그먼트당 419,430초). 반면
    pbf 에서 way 를 지우면 그 도로는 **존재하지 않는다**. 애매함이 없다.

    비용도 문제되지 않는다 — 대전 기준 extract+partition+customize 전체가 8초다.

    덤으로 baseline 과 차단본을 서로 다른 포트에 동시에 띄울 수 있어서
    reset 이 필요 없다. .osrm.geometry 오염 문제도 원천적으로 사라진다.

## 왜 way 단위인가 — node 를 지우면 안 되는 이유

way 를 지우면 그 way 만 참조하던 node 는 고아가 된다. OSRM 은 way 에서 도로망을
만들므로 고아 node 는 무해하다(라우팅에 쓰이지 않고 check-refs 도 통과한다).

반대로 **node 를 지우면 안 된다.** 실측으로 확인한
네 가지 이유:

1) 파일이 깨진다. node 를 지워도 way 의 nd 목록은 그대로 남아 존재하지 않는 node 를
   참조한다(`osmium removeid` 는 참조를 정리하지 않는다).
       osmium check-refs  →  "Nodes in ways missing: 5"
   그런데 **osrm-extract 는 exit 0 으로 조용히 통과한다.** 경고 로그도 없다.
   check-refs 를 돌리지 않으면 깨진 도로망 위에서 실험한 것을 알 수 없다.

2) 의도하지 않은 도로까지 부순다. 대전 node 481,459개 중 16.3% 가 2개 이상의 way 에
   공유된다. 도솔대교의 node 10065676938 하나에 무관한 way 3개와 갑천도시고속도로가
   걸려 있다. 실제로 node 3개를 지웠을 때 누락 참조는 5개로 늘어났다.

3) 결과를 예측하기 어렵다. 지운 위치에 따라 way 가 조각 0~여러 개로 갈린다.
   nd 목록에서 양 끝 node 가 모두 살아있는 세그먼트만 남고, 그 조각이 네트워크에서
   고립되면 OSRM 이 버린다. 측정 예(way 319807894, node 7개 중 2개 삭제):
       f-g 세그먼트  0.373 km → 0.373 km   그대로 살아남음
       d-e 세그먼트  0.014 km → 3.161 km   끊김
       c-d 세그먼트  0.012 km → 0.000 km   고립되어 버려짐
       a-g 전체      0.569 km → 3.371 km   관통 불가
   지름길(a-d-f)이 생기지는 않았지만, '고립'이라는 또 다른 예측 어려운 상태가 생긴다.

4) 결과를 방어할 수 없다. 데이터 무결성이 깨진 도로망에서 나온 소요시간은
   차단의 효과인지 손상의 효과인지 구분할 수 없다.

## way 일부만 끊고 싶을 때

node 삭제가 아니라 **way 쪼개기**다. [a,b,c,d,e] 를 [a,b,c] 와 [c,d,e] 두 way 로
나누고 한쪽만 지운다. node 는 하나도 지우지 않으므로 check-refs 를 통과하고,
어느 구간이 남는지 명확하다. (아직 미구현 — 공간 selector 로 좁은 영역을 끊을 때
필요해진다. 태그 기반 선택은 way 전체가 선택되므로 현재 방식으로 오차가 0이다.)
"""
from __future__ import annotations

from pathlib import Path


def write_pbf_without_ways(src: Path, dst: Path, drop_way_ids: set) -> dict:
    """
    src pbf 를 복사하면서 drop_way_ids 에 든 way 만 빼고 dst 에 쓴다.

    Returns: {"nodes":..., "ways_kept":..., "ways_dropped":..., "relations":...}
    """
    import osmium

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    drop = set(int(w) for w in drop_way_ids)
    stat = {"nodes": 0, "ways_kept": 0, "ways_dropped": 0, "relations": 0}

    writer = osmium.SimpleWriter(str(dst))
    try:
        # node/way/relation 을 모두 그대로 흘려보내며 대상 way 만 건너뛴다.
        # 위치 캐시(with_locations)는 필요 없다 — 원본 객체를 그대로 다시 쓰기만 한다.
        for o in osmium.FileProcessor(str(src)):
            if isinstance(o, osmium.osm.Way):
                if o.id in drop:
                    stat["ways_dropped"] += 1
                    continue
                writer.add_way(o)
                stat["ways_kept"] += 1
            elif isinstance(o, osmium.osm.Node):
                writer.add_node(o)
                stat["nodes"] += 1
            else:
                writer.add_relation(o)
                stat["relations"] += 1
    finally:
        writer.close()

    return stat

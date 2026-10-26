# road_disruption — 도로망 차단 실험 (v2)

**질문:** 어떤 도로를 끊어야 MCI 대응에 가장 큰 타격이 오는가? 재해로 특정 도로망이
끊겼을 때 대응 성능은 얼마나 나빠지는가?

v1(diffusion 좌표 생성)과 분리된 실험 라인이다. torch 를 쓰지 않고 라우팅만 다룬다.
`MCI_ADV2/` 시뮬레이터와 `src/data/road_network.py` 스냅 유틸은 저장소 것을 재사용한다.

## 차단 방식 — pbf 에서 way 를 지운다

```
지역 pbf ──(대상 way 삭제)──> 차단본 pbf ──extract/partition/customize──> 차단본 OSRM
                                                                          (별 포트)
                                                     baseline OSRM 은 그대로 유지
```

대전 기준 전체 **9초**. baseline(5002)과 차단본(5010)이 동시에 살아 있으므로 같은 OD 를
두 포트에 물어보면 그게 곧 차단 전/후 비교다. reset 이 필요 없다.

### 왜 `--segment-speed-file` 방식을 버렸는가

처음엔 `osrm-customize --segment-speed-file` 로 speed 0 을 주는 방식을 썼다. 두 가지
함정 때문에 폐기했다.

**① `speed 0` 은 통행 불가가 아니다.** 아주 큰 유한 가중치일 뿐이다(관측: 세그먼트당
`duration = 419,430 s`, 약 4.85일). 우회로가 있으면 우회하지만, **없으면 그 구간을 그냥
통과하면서 소요시간만 터무니없이 커진다.**

**② `osrm-customize` 만으로는 reset 이 안 된다.** customize 는 `.osrm.geometry` 를
덮어쓰면서 원본 세그먼트 소요시간을 파괴한다. speed 파일 없이 다시 돌려도 복원되지
않으므로 차단이 조용히 누적된다. 실제로 오염된 상태에서 같은 OD 가 14.02분 대신
34,965분으로 나왔다.

way 를 지우면 그 도로는 **존재하지 않는다**. 두 함정이 모두 사라진다.

두 방식의 결과는 소수점까지 일치했다(14.02→20.56, 18.56→32.63 …) — 서로 독립된
메커니즘이 같은 답을 냈으므로 숫자를 믿을 수 있다.

### 차단 단위는 way 다 — node 를 지우면 안 된다

세그먼트(인접 node 쌍)가 OSRM 의 라우팅 단위이므로 node 단위 차단이 자연스러워
보이지만, **node 삭제는 쓰면 안 된다.** 실측으로 확인한 네 가지:

**① 파일이 깨지는데 조용히 넘어간다.** node 를 지워도 way 의 nd 목록은 그대로 남아
존재하지 않는 node 를 참조한다(`osmium removeid` 는 참조를 정리하지 않는다).

```
osmium check-refs   →  "Nodes in ways missing: 5"
osrm-extract        →  exit 0, 경고 로그 없음        ← 가장 위험한 부분
```

**② 의도하지 않은 도로까지 부순다.** 대전 node 481,459개 중 **16.3% 가 2개 이상의
way 에 공유**된다. 도솔대교 node 하나에 무관한 way 3개와 갑천도시고속도로가 걸려 있고,
실제로 node 3개를 지웠을 때 누락 참조는 5개로 늘어났다.

**③ 결과를 예측하기 어렵다.** way 319807894(node 7개)에서 2개를 지운 측정값:

| 구간 | 정상 | node 삭제 | 결과 |
|---|---|---|---|
| f-g (양끝 생존, 연결됨) | 0.373 km | 0.373 km | 그대로 |
| d-e (한쪽 삭제됨) | 0.014 km | 3.161 km | 끊김 |
| c-d (양끝 생존, 고립) | 0.012 km | 0.000 km | 버려짐 |
| a-g (관통) | 0.569 km | 3.371 km | 불가 |

nd 목록에서 양 끝이 모두 살아있는 세그먼트만 남고, 그 조각이 고립되면 OSRM 이 버린다.
지름길(a-d-f)이 생기지는 않지만 '고립'이라는 또 다른 예측 어려운 상태가 생긴다.

**④ 결과를 방어할 수 없다.** 무결성이 깨진 도로망의 소요시간은 차단 효과인지 데이터
손상 효과인지 구분되지 않는다.

반대로 **way 를 지우면 node 가 남으므로 참조가 깨지지 않는다.** 고아 node 는 라우팅에
쓰이지 않아 무해하고 `check-refs` 도 통과한다. way 는 OSM 에서 지울 수 있는 최소
객체이며(세그먼트는 객체가 아니라 계산값), 태그 기반 선택에서는 way 전체가 선택되므로
오차가 0이다.

way 일부만 끊으려면 node 삭제가 아니라 **way 쪼개기**다 — 미구현이며, 공간 selector 로
좁은 영역을 끊을 때 필요해진다.

## 사용

```bash
conda env create -f road_disruption/environment.yml    # 최초 1회
conda activate road

# --- 지역당 1회 준비 ---
python road_disruption/scripts/extract_region_pbf.py --region daejeon   # 전국→지역 pbf
python road_disruption/scripts/build_way_index.py    --region daejeon   # pbf→way 인덱스
python road_disruption/scripts/build_region_osrm.py  --region daejeon   # baseline OSRM(5002)

# --- 후보 검토 ---
python road_disruption/scripts/list_candidates.py --region daejeon --group-by bridge
python road_disruption/scripts/serve_maps.py --port 8010     # SSH 원격일 때 지도 보기

# --- 후보 좌표로 way 찾기 (교량은 OSM 에 고유명이 없다) ---
python road_disruption/scripts/whats_here.py --region daejeon \
    --at 36.34147,127.35208 --radius 250 --bridge-only

# --- 차단 실험 ---
# ① 대상 way 를 지운 붕괴본 OSRM 을 다른 포트에 (약 9초)
python road_disruption/scripts/cut_by_delete.py \
    --spec config/cuts/daejeon_wonsinheung_bridge.yaml --port 5011

# ② 두 경로를 각각 기록 (OSRM 질의는 여기서만)
python road_disruption/scripts/dump_route.py --port 5002 \
    --from 36.3757,127.3767 --to 36.306642,127.342613 --tag normal
python road_disruption/scripts/dump_route.py --port 5011 \
    --from 36.3757,127.3767 --to 36.306642,127.342613 --tag collapsed

# ③ 기록만 읽어서 비교 + 지도 (OSRM 질의 없음)
python road_disruption/scripts/compare_routes.py \
    --normal normal --collapsed collapsed \
    --spec config/cuts/daejeon_wonsinheung_bridge.yaml \
    --label science_to_konyang

docker rm -f osrm-cut-daejeon_wonsinheung_bridge     # 정리
```

`dump_route.py` 가 OSRM 과 대화하는 유일한 곳이다. `compare_routes.py` 는 저장된
JSON 만 읽는다 — 비교 스크립트가 다시 질의하면 저장된 기록과 화면의 그림이 다를 수
있고(두 호출 사이에 붕괴본을 다시 빌드하면 실제로 어긋난다), 그러면 실험 기록으로
쓸 수 없다. 덤으로 질의 없이 지도를 몇 번이든 다시 만들 수 있다.

`--label` 은 파일명에 쓰이므로 영어로 쓴다.

## CutSpec — 무엇을 끊을지 선언한다

차단 단위를 코드에 박지 않는다. selector 조합으로 표현한다.

```yaml
name: daejeon_bridge_cut
region: daejeon
select:
  all_of:
    - {type: osm_tag,       tags: {bridge: "*"}}
    - {type: highway_class, classes: [motorway, trunk, primary]}
    - {type: bbox,          bbox: [127.35, 36.32, 127.40, 36.36]}
```

| type | 파라미터 | 용도 |
|---|---|---|
| `way_id` | `ids: [...]` | 개별 way 지정 |
| `highway_class` | `classes: [...]` / `preset: drive\|major` | 도로 등급 |
| `osm_tag` | `tags: {bridge: "*"}` | 교량·터널·도로명 등. `"*"`=존재, 리스트=OR |
| `drive_only` | — | 보도·등산로 제외 |
| `bbox` | `bbox: [...]` | 격자 셀 |
| `radius` | `center, radius_m` | 지점 중심 반경 (EPSG:5186 미터) |
| `polygon` | `path, layer, where` | 침수구역 등 실제 재해 shapefile |

`--way-mode any`(기본)는 세그먼트가 하나라도 선택되면 way 전체를 삭제한다.
`all` 은 way 의 모든 세그먼트가 선택된 경우만 삭제한다.

**언제 이 근사가 문제가 되는지 측정했다** (대전 기준):

| 선택 방식 | way 전체 선택 | 일부만 선택 | 실제 삭제량 |
|---|---|---|---|
| 태그 — 교량 ∩ 간선도로 | 100.0% | 0.0% | 의도의 **100%** |
| 태그 — 계백로 전체 | 100.0% | 0.0% | 의도의 **100%** |
| 공간 — bbox 500m | 81.7% | 18.3% | 의도의 113% |
| 공간 — bbox 1.1km | 91.0% | 9.0% | 의도의 109% |
| 공간 — bbox 2.2km | 95.9% | 4.1% | 의도의 105% |

`bridge=yes` 나 `name` 은 way 단위 태그이므로 그 way 전체가 대상이 된다 → 오차 0.
공간 selector(`bbox`/`polygon`/`radius`)에서만 경계에 걸친 긴 도로가 통째로 지워져
과대평가되고, **영역이 좁을수록 심하다**(500m 박스에서 13% 초과).

따라서 태그 기반 실험은 현재 방식으로 정확하다. `polygon` 으로 실제 침수구역
shapefile 을 쓰기 시작할 때 way 쪼개기 모드가 필요해진다.

## 도로 단위 — way 는 도로가 아니다

OSM way 는 "태그가 균일한 구간"이다. 측정값:

```
주행도로 way 길이   중앙값 100.7m · 평균 200.6m
계백로              169 ways / 총 58.78km
way 내부 node 중 공유  14.2%   ← 0 이 아니다 = way 가 교차로를 통과한다
```

`계백로` 하나에 highway(primary/trunk/residential), lanes(2~8차선), oneway, bridge 가
모두 섞여 있어서 169개로 쪼개졌다. way 하나를 끊는 건 도로 100m 를 끊는 것이다.

### 층위 (대전 주행 도로망)

| 층위 | 단위 | 개수 |
|---|---|---|
| OSM 원본 저장 | way | 36,184 |
| 세그먼트 (인접 node 쌍) | selector 판정 단위 | 230,951 |
| **OSRM 라우팅 그래프** | **교차로 사이 구간** | **70,387** |

OSRM 빌드 로그의 `edge-based graph: 125,656 nodes` 의 절반(62,828)이 우리가 계산한
70,387 과 11% 차이다 — `car.lua` 도로 집합과 일방통행 처리 차이로 설명된다.

`highway` 와 `name` 은 독립된 태그다. `--preset major` 는 `highway` 값 8종만 보고
이름은 보지 않는다. 그래서 간선 6,299개 중 1,828개가 이름 없는 램프로 남고,
그걸 빼는 것이 `--named-only` 다.

### 후보 규모 (대전, `list_candidates.py`)

| 단위 | 후보 수 | 총 연장 |
|---|---|---|
| `--group-by bridge` | 434 | 126.8 km |
| `--group-by name --named-only` | 370 | 1,595.1 km |
| `--group-by name` | 2,198 | 1,875.7 km |
| `--group-by way` | 6,299 | — |

CSV 의 `선택` 칼럼에 표시한 뒤 `group` 값을 CutSpec 의
`osm_tag tags:{name: [...]}` 로 옮기면 된다.

## 검증 결과 (`daejeon_major_bridges`)

간선도로 교량 942 ways 를 삭제했을 때:

| OD | 전 | 후 | Δ |
|---|---|---|---|
| 유성온천역 → 대전역 | 14.02 | 20.56 | +46.7% |
| 신탄진 → 대전시청 | 18.56 | 32.63 | +75.8% |
| 진잠 → 대덕구청 | 13.88 | 23.46 | +69.0% |
| KAIST → 대전역 | 13.96 | 18.36 | +31.5% |

소요시간만 보면 "차단 실패"와 "우연히 비슷한 우회로"를 구분할 수 없다. 그래서
`compare_ports.py` 는 `annotations=nodes` 로 경로의 OSM node 열을 받아 삭제한 구간이
남아 있는지 직접 확인한다. 4/4 OD 가 baseline 에서 대상 구간을 썼고, 차단본에서
누출은 0이었다.

## 구조

```
road_disruption/
  environment.yml
  config/
    regions.yaml            지역 bbox + OSRM 포트
    cuts/*.yaml             CutSpec
  roaddis/
    paths.py                OSRM 볼륨 위치, data 디렉토리
    regions.py              regions.yaml 로더
    osm_tags.py             highway 분류 상수
    cli.py                  스크립트 공용 헬퍼 (slug, 좌표 파싱)
    way_index.py            WayIndex — node CSR, 세그먼트 레이어, oneway 전개
    selectors.py            selector 7종 + REGISTRY
    cutspec.py              CutSpec 파싱 · all_of/any_of/none_of
    pbf.py                  pbf 에서 way 삭제
    osrm.py                 docker 제어 + 라우팅 클라이언트
  scripts/
    # 지역당 1회 준비
    extract_region_pbf.py   전국 pbf → 지역 pbf
    build_way_index.py      pbf → way 인덱스
    build_region_osrm.py    일반 도로망 OSRM 인스턴스
    # 후보 조사
    list_candidates.py      후보 목록(CSV) + folium 지도
    whats_here.py           좌표 → 그 자리의 way_id (교량 찾기)
    # 실험
    cut_by_delete.py        way 삭제 → 붕괴본 OSRM
    dump_route.py           경로를 JSON 으로 기록 (OSRM 질의는 여기서만)
    compare_routes.py       기록 2개 비교 + 지도 (질의 없음)
    # 보조
    serve_maps.py           지도·JSON HTTP 서빙 (SSH 원격용)
  data/                     생성물 (gitignore)
    way_index/              지역별 way 인덱스
    routes/                 경로 기록 JSON + 비교 지도
```

## 진행 상황

- [x] 스캐폴드 + 환경
- [x] pbf → way/node 인덱스 (대전 60,120 ways, 좌표 NaN 0)
- [x] baseline OSRM(5002) — 전국과 소요시간 0.000 min 차이
- [x] 후보 목록화 + 지도
- [x] way 삭제 차단 + 검증 통과
- [ ] MCI_ADV2 연결 → Δ이송시간 · 도달불가 OD · 사망률
- [ ] 후보 탐색 루프
- [ ] way 부분 삭제(구간 단위) 모드

## 주의

- 전국 인스턴스(5001)는 baseline 이다. 차단 실험은 지역 인스턴스에서만 한다.
- `roaddis/osm_tags.py` 의 `DRIVE_HIGHWAY` 는 `scripts/build_road_cache.py` 의
  `OSM_DRIVE_HIGHWAY` 와 같은 값을 유지해야 한다.
- pbf 를 최신 Geofabrik 파일로 갱신하면 **OSRM 도 반드시 같이 다시 빌드**해야 한다.
  (현재 pbf: `asia/south-korea-latest`, 2026-03-18T21:21:31Z)
- `MAJOR_HIGHWAY` 에 `tertiary` 가 빠져 있다. 대전에 2,857 ways 로 꽤 많고 구급차가
  실제로 쓸 수 있어, 후보에서 제외하는 게 맞는지는 검토가 필요하다.
- **SSH 원격(맥북 → 윈도) 작업 시 지도 보기**: `file:///` 나 `Start-Process` 는
  브라우저가 원격 머신에서 열려 쓸 수 없다.

  ```bash
  python road_disruption/scripts/serve_maps.py --port 8010
  #  VS Code Remote-SSH → PORTS 탭에 8010 자동 등록
  #  순수 ssh          → ssh -L 8010:localhost:8010 <user>@<host>
  #  브라우저          → http://localhost:8010/candidates/daejeon_bridge_major.html
  ```

  VS Code Live Preview(3000)로 열려다 `ERR_CONNECTION_REFUSED` 가 나는 것은 3000 포트가
  맥북으로 포워딩되지 않았기 때문이다.

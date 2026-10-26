# sim_src_upgrade (MCI_ADV 판) — 결과 동일, 연산시간만 단축

`src/sim_src/` 는 **정본이고 수정하지 않는다.** 이 폴더는 같은 로직의 별도 실행경로다.
목표는 하나뿐이다: **같은 조건에서 비트 단위로 같은 결과, 더 짧은 시간.**

원본은 MCI_UAV(`bbcc1017/MCI_UAV`)의 `src/sim_src_upgrade` 이고, 이 저장소에는 시뮬
실행에 필요한 부분만 이식했다. MCI_UAV 쪽 RL 전용 자산(`env_factory_fast.py`,
`fast_obs_patch.py`, `mask_only_wrapper.py`, `bench/`, rl_src 드라이버용 `verify/*`)은
여기에 `src/rl_src` 가 없으므로 제외했다.

## 구성

| 경로 | 역할 |
|---|---|
| `core/` | `src/sim_src` 7개 모듈의 고속 사본. **패키지 상대 import** 로 바꿔 구 코어와 한 프로세스에 동시 로드 가능 |
| `origin_sync.py` | G0 드리프트 게이트 — 사본이 파생된 `src/sim_src` 의 sha256 대조 |
| `drivers/run_sim_fast.py` | `src/sim_src/main.py` 를 **한 줄도 고치지 않고** 고속 코어로 실행하는 런처 |
| `verify/sim_equivalence.py` | 구·신 코어를 같은 시드로 굴려 지표 배열을 비교(동치검증 + 런처 사전점검) |
| `_paths.py` | sys.path 헬퍼 |

## 쓰는 법

### 1) CLI — `main.py` 대체

`--` 뒤의 인자는 `main.py` 로 그대로 전달된다. **저장소 루트에서 실행**할 것(YAML 상대경로 규약).

```bash
CFG="scenarios/exp_XXXX/(37.5665,126.978)/config_(37.5665,126.978).yaml"

# 기존
python src/sim_src/main.py --config_path "$CFG" --trace

# 고속 (산출 파일 바이트 동일)
python src/sim_src_upgrade/drivers/run_sim_fast.py -- --config_path "$CFG" --trace
```

실행 순서: **G0 드리프트 검사 → 사전 동치검증(규칙 2 × 에피 3) → flat 모듈 교체 →
`main.py` 실행**(`--skip_preflight` 로 사전점검 생략).

게이트가 깨졌을 때는 **멈추지 않고 원본 코어로 폴백한다**. G0 드리프트나 사전점검
불일치는 "사본을 믿을 수 없다"는 뜻이고, 정답의 정의는 언제나 `src/sim_src` 원본이므로
주입을 포기하고 원본으로 그대로 돌린다 — 결과는 정확하고 시간만 손해다. 어느 코어로
돌았는지는 stdout 한 줄로 남는다.

```
[fastcore] CORE=fast      # 고속 코어
[fastcore] CORE=origin    # 게이트 실패 → 원본 코어 폴백 (직전 [G0]/[preflight] 줄에 이유)
```

`--strict` 를 주면 예전처럼 중단한다(G0 → exit 2, 사전점검 → exit 3). 사본이 정말 같은지
확인하는 게 목적인 CI·검증에서만 쓴다. 대시보드는 결과 산출이 목적이라 기본(폴백)이다.

### 2) Orchestrator / 대시보드에서

```python
orc.run_simulation(config_path, use_fast_core=True, trace=True)
```

`use_fast_core=False`(기본)면 기존 `main.py` 경로 그대로다. 산출물 경로·파일명·포맷은
두 경로가 동일하다.

### 3) 동치검증 단독 실행

```bash
python src/sim_src_upgrade/verify/sim_equivalence.py --config "$CFG" --n_eps 10
# → 규칙 64 × 에피 10 × 지표 5 = 3,200개 원소 비교 + 배속 출력
```

## 검증 기록 (2026-08-13, 서울시청 좌표 · OSRM · 병원 9곳 · 환자 30)

| 항목 | 결과 |
|---|---|
| 지표 배열 `(64규칙 × 10에피 × 5지표)` = 3,200 원소 | **전부 비트동일, 최대 절대차 0.0** |
| `results_*.txt` / `results_*_stat.txt` | 바이트 동일 |
| `trace_*.json` (5.5MB) | 바이트 동일 |
| 배속 (동치검증 in-process, 출력 없음) | **2.15 ~ 2.38×** |
| 배속 (대시보드 경로 wall, 로그 ON) | **1.68×** (100회: 63.10s → 37.52s) |
| 배속 (대시보드 경로 wall, `MCI_TRACE_PRINT=0`) | **1.99×** (100회: 65.14s → 32.73s) |
| G0 드리프트 | PASS — 사본 파생원본 = 현재 `src/sim_src` |

MCI_UAV 쪽 대규모 실측(규칙 전수평가 3.7~4.4×, v16 드라이버 wall 3.34×)은 그 저장소의
`Pool` 병렬 드라이버 기준이다. 이 저장소는 단일 프로세스 `main.py` 라 배속이 그보다 낮다.

## 원본이 바뀌면

`src/sim_src` 를 수정하면 G0 가 드리프트를 잡는다. 그때 시뮬은 **원본 코어로 계속
돌아간다**(고속화만 잠시 꺼진다). 정상화 절차는 **① 사본(`core/`)에 같은 변경을 반영 →
② `python src/sim_src_upgrade/origin_sync.py --write` → ③ 동치검증 재실행** 이다.
반영 없이 `--write` 만 하면 드리프트를 덮어버린다.

```bash
python src/sim_src_upgrade/origin_sync.py --check              # 드리프트 검사
python src/sim_src_upgrade/origin_sync.py --diff EventManager  # 원본 대비 사본 diff
```

## 최적화 원칙 (전부 "로직 동일")

부동소수 연산 순서·tie-break·RNG 드로우가 바뀌면 궤적이 갈린다. 그래서 다음은 금지다.

* 차량 잔여시간을 절대시각(`t_abs - now`)으로 전환 (반복 감산의 누적오차가 달라짐)
* `room = (max_capa + max_queue) - occ - in_flight` 연산 순서 변경
* `np.argsort` 의 `kind`, `p_wait[...].pop()` 순서, 정렬 안정성 가정
* RNG 드로우 추가/제거/재배치

실제로 바꾼 것은 이송중 카운트 `bincount` 화, GB 이송 후보 순서 시나리오당 1회 사전계산,
종료·구조 판정 증분 카운터, 결합 mask 브로드캐스트, `patient_info` DataFrame 조회 제거,
이벤트별 디버그 `print` 게이트(`TRACE_PRINT`) 등이다.

> ⚠️ **이벤트 출력은 기본 켬**(`TRACE_PRINT`). 대시보드가 `experiment_logs/` 의 이 출력을
> 파싱해 Scenarios 탭·Maps Animation 을 그리기 때문이다. 순수 배치라면
> `MCI_TRACE_PRINT=0` 으로 꺼서 약 1.68× → 약 2.0× 로 올릴 수 있다. 출력 여부는 상태·RNG 와
> 무관하므로 **결과 파일은 어느 쪽이든 동일**하다(100회 실행 로그 1,350,093줄 중 다른 줄은
> `Computation time(s)` 1줄뿐).

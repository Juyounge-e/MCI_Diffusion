# MCI-Diffusion

대량사상자(MCI, Mass-Casualty Incident) 대응 연구를 위한 통합 저장소입니다. 크게 **세 개의 서브시스템**으로 구성됩니다.

1. **좌표 생성 (Conditional Diffusion, beanchmarks)** — 사고 중증도 조건 `(pdr, N, RYGB)`에 맞춰 사고 좌표 `(lat, lon)`을 생성하는 조건부 확산모델 파이프라인 (`src/`, `scripts/`, `benchmarks/`, `eval/`)
2. **MCI 시뮬레이션 (MCI_ADV2)** — 생성/실제 좌표에 대해 64개 출동 규칙을 시뮬레이션하여 사망률(PDR) 등 지표를 산출 (`MCI_ADV2/`)
3. **도로망 차단 분석 (Road Disruption)** — "어떤 도로를 끊으면 MCI 대응이 가장 나빠지는가"를 OSRM 라우팅으로 분석 (`road_disruption/`)

> 확산모델이 좌표를 만들고 → 시뮬레이터가 그 좌표의 PDR을 계산하며 → 두 값을 비교(`eval/`)해 모델을 검증하는 페루프 구조.
도로망 차단 실험은 같은 시뮬레이터를 재사용하는 별도의 실험입니다.

---

## 목차
- [저장소 구조](#저장소-구조)
- [설치 / 환경](#설치--환경)
- [1. 확산모델 좌표 생성 파이프라인](#1-확산모델-좌표-생성-파이프라인)
- [2. 도로 스냅 / off-road 도구](#2-도로-스냅--off-road-도구)
- [3. 평가 (eval)](#3-평가-eval)
- [4. 비교 모델 (benchmarks)](#4-비교-모델-benchmarks)
- [5. MCI 시뮬레이션 (MCI_ADV2)](#5-mci-시뮬레이션-mci_adv2)
- [6. 도로망 차단 분석 (road_disruption)](#6-도로망-차단-분석-road_disruption)
- [7. 시각화 앱 / 노트북](#7-시각화-앱--노트북)
- [알려진 이슈 / 주의](#알려진-이슈--주의)

---

## 저장소 구조

```
MCI_Diffusion/
├── src/                  # 확산 모델 코어 (데이터 로더 / 모델 / 스케줄러 / road loss)
│   ├── data/             
│   ├── model/            
│   └── diffusion/        
├── scripts/              # 학습·샘플링·
├── eval/                 # 생성 vs 시뮬레이션 평가 
├── benchmarks/           # 비교 생성모델 (mlp/cvae/cgan/mdn/tvae)
├── MCI_ADV2/             # MCI 시뮬레이터 + 시나리오 생성 + Streamlit 대시보드
│   ├── sim_src/          #   시뮬레이션 코어(원본) + config.yaml
│   ├── sim_src_upgrade/  #   고속 코어(fast core) + 등가성 검증
│   ├── sce_src/          #   시나리오 생성 / 배치 오케스트레이션         
│   └── scenarios/        #   입력/생성 데이터 (shapefile, 병원·소방서 등)
├── road_disruption/      # 도로망 차단 실험 (OSRM 기반, torch 미사용)
│   ├── config/          
│   ├── roaddis/          
│   └── scripts/         
├── notebooks/            # 분석/시각화 노트북 및 스크립트
├── app/                  # app.py (전국 시각화 Streamlit)
├── tab-ddpm/             # 서브모듈 (yandex-research/tab-ddpm)
└── requirements*.txt     # 의존성 
```

---

## 설치 / 환경

서브시스템마다 요구 패키지가 달라 **conda 환경을 3개로 분리**해 사용합니다.

| 환경 | requirements 파일 | 용도 | 비고 |
|---|---|---|---|
| `tddpm` (확산모델) | `requirements.txt` = `requirements_tddpm.txt` | 학습·샘플링·benchmarks·eval | `torch==1.10.1+cu111` |
| MCI 시뮬레이션 | `requirements_mci.txt` | 시뮬레이터·시나리오 생성 | geopandas/shapely/streamlit, torch 없음 |
| MCI 시각화 | `requirements_mci_viz.txt` | Streamlit 대시보드/플롯 | conda export본(`file:///` 경로 포함, **그대로 재설치 불가** — 참고용) |

> `requirements.txt`와 `requirements_tddpm.txt`는 **내용이 동일**합니다.

**확산모델 환경 (tddpm) 설치:**
```bash
conda create -n tddpm python=3.9.7
conda activate tddpm
pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
git submodule update --init --recursive   # tab-ddpm 서브모듈
pip install -r requirements.txt
```

**MCI 시뮬레이션 환경:**
```bash
conda create -n mci python=3.9
conda activate mci
pip install -r requirements_mci.txt
```

> 서브모듈 `tab-ddpm`(yandex-research/tab-ddpm)은 `src/model/build.py`, `src/diffusion/scheduler.py`가 `sys.path`에 추가해 사용합니다. 반드시 `git submodule update --init --recursive`로 받아야 합니다.

---

## 1. 확산모델 좌표 생성 파이프라인

전형적인 흐름: **데이터 준비 → (선택) 도로 캐시 → 학습 → 샘플링 → 도로 스냅/평가**

### 1-1. 데이터 준비 — `src/data/dataset.py`
시뮬레이터 결과 `.txt` + `experiment_metadata.csv`를 파싱해 학습용 CSV(`lat, lon, pdr_mean, red/yellow/green/black, N`)를 `src/data/`에 생성
파일 최상단 경로를 변경 

### ~~1-2. (선택) 도로 캐시 / 마스크 생성(road loss 구현에 사용했던 것 폐기 예정)~~
도로거리 보조손실·도로 스냅·육지 마스크를 쓰려면 먼저 캐시를 만듭니다.

```bash
# 도로망 → 스냅 캐시(pkl, EPSG:5186). OSM pbf 입력
python scripts/build_road_cache.py --source osm --src south-korea.osm.pbf --region daejeon --out road_cache.pkl

# 도로 캐시(pkl) → 점군(npz). 학습 시 도로거리 손실에 사용
python scripts/build_road_points.py --road_cache road_cache.pkl --spacing_m 30.0 --out road_points.npz

# 행정경계 shapefile → 육지/바다 격자 마스크(npz)
python scripts/build_spatial_mask.py --shp_path MCI_ADV2/scenarios/ctprvn.shp --out_path MCI_ADV2/scenarios/mask_cache_0.005.npz --resolution 0.0005
```
- `--region` 선택지: `daejeon`, `incheon`, `national`
- 입력: OSM `*.osm.pbf`(EPSG:4326)

### 1-3. 학습 — `scripts/train_mlp.py`
조건부 MLP 확산모델을 학습합니다 (조건: pdr + RYGB 카운트 → `lat,lon`).

```bash
python scripts/train_mlp.py \
  --csv src/data/national_all.csv \
  --out_dir outputs/mlp_diffusion/model/test_national \
  --total_steps 10000 --val_ratio 0.1 --val_every 500 --seed 42
```

### 1-4. 샘플링 — `scripts/sample_mlp.py`
학습 체크포인트로부터 역확산을 돌려 좌표를 생성합니다. 

```bash
python scripts/sample_mlp.py \
  --ckpt outputs/mlp_diffusion/model/test_national/last_model.pt \
  --scalers outputs/mlp_diffusion/model/test_national/scalers.pkl \
  --out outputs/mlp_diffusion/sample/test_national/samples.csv \
  --sample_num 50 --pdr 0.042927 --N 30
```
주요 인자(기본값): `--sample_num 50`, `--pdr 0.042927`, `--N 30`, `--rygb None`(미지정 시 `--ratio_bank`에서 추출), `--normal MEAN STD` / `--uniform MIN MAX`(pdr 분포 샘플링), `--timesteps 1000`, `--project_road_points ""`(설정 시 스텝별 도로 스냅), `--project_min_t 200`, `--seed None`.
- 출력 CSV 컬럼: `lat, lon, N, pdr, r_ratio, y_ratio, g_ratio, b_ratio`.

---

## ~~2. 도로 스냅 / off-road 도구~~

생성 좌표가 도로 위에 있는지 측정하거나, 폴더 단위로 스냅합니다. (`src/data/road_network.py`의 `RoadNetwork`를 공용 사용 — WGS84 ↔ EPSG:5186, STRtree 최근접 도로 투영)

```bash
# 폴더 내 모든 샘플 CSV를 도로에 스냅 + 바다/off-road 지표 기록
python scripts/road_snap_batch.py --dir <csv폴더> --road_cache road_cache.pkl \
  --mask MCI_ADV2/scenarios/mask_cache_0.005.npz --offroad_thresh_m 50.0 --suffix _snapped

# 단일 CSV의 도로 근접도만 측정(교정 X): 최근접 도로거리 + 임계별 on-road 비율
python scripts/offroad_metric.py --samples samples.csv --road_cache road_cache.pkl \
  --thresholds 10 30 50 100 --out metric.csv
```
- `road_snap_batch.py`: 각 `<name>.csv` → `<name>_snapped.csv` (원본 + `lat_snap, lon_snap, snap_dist_m, sea_before, sea_after`)
- `offroad_metric.py`: 지표 표 출력, `--out` 지정 시 `road_dist_m`·`onroad_{thr}m` 컬럼 추가 CSV
- `scripts/ratio_bank.py`: `ratio_bank.pkl` 내용 확인용 보조 스크립트(인자 없음)

---

## 3. 평가 (eval)

생성 좌표의 PDR(시뮬레이션)과 요청 조건을 비교하고, 과적합·분포를 점검합니다.

```bash
# 요청 cond vs 시뮬레이션 pdr_mean 오차 지표(MAE/RMSE/NRMSE/Bias/within-tol)
python eval/eval_mlp.py --simul_csv <sim결과.csv> --cond 0.057271 --out eval_with_metrics.csv --tol 0.0075

# 생성 vs 시뮬레이션 PDR을 (원좌표/스냅좌표) 메타데이터로 조인해 지표 계산
python eval/compare_gen_simul_pdr.py --gen_csv <gen.csv> --simul_csv <sim.csv> --meta_csv <meta.csv>
#   q-bin 모드: --gen_csv 대신 --gen_glob "<...q*.csv>"

# 학습 TensorBoard 이벤트 → 손실곡선 PNG
python eval/eval_tb.py --tb_dir outputs/mlp_diffusion/model/test_national/tb

# 과적합(암기) 점검: 생성-학습 최근접거리 vs 학습 self-NN
python eval/check_overfitting.py --train_csv <train.csv> --gen_glob "<all_bins.csv glob>"

# 모델별(Diffusion/MLP/CVAE/cGAN/MDN) 전국 지도 비교 + off-land/off-road 지표
python eval/viz_models_national.py --diffusion_dir <dir> --benchmark_root outputs/benchmarks

# qbin별 Bias 막대 / 의도 vs 실제 샘플 수 플롯
python eval/plot_bias_n_national.py
```
> 그 밖에 `eval/plot_qbin_nrmse.py`, `eval/viz_roadloss_national.py`도 있습니다.

---

## 4. 비교 모델 (benchmarks)

확산모델과 비교하기 위한 **baseline 조건부 생성모델** 모음입니다. 공통 계약: 학습 CSV `src/data/national_all.csv`, 타깃 `lat,lon`, 조건 `(pdr_mean, N)`, q-bin 출력 `lat,lon,N,pdr`, **도로 투영 미사용**(확산 라인과 다름).

```bash
# 통합 학습기 (mlp | cvae | cgan | mdn)
python benchmarks/train.py --model cvae --csv src/data/national_all.csv --epochs 50

# 통합 샘플러 (모델은 ckpt에서 추론)
python benchmarks/sample.py --ckpt outputs/benchmarks/cvae/best_model.pt --out samples.csv --n 50 --pdr 0.08 --N 30

# q-bin 샘플링 프로토콜 (bin별로 sample.py 호출 후 all_bins.csv 병합)
python benchmarks/qbin_sampling.py --model cvae --ckpt <pt> --out_dir outputs/benchmarks/cvae/qbin_n30_1k
```
- 각 모델 디렉터리(`mlp/ cvae/ cgan/ mdn/`)의 `train_*.py`/`sample_*.py`는 위 통합 스크립트를 모델 프리셋으로 호출하는 얇은 래퍼입니다.
- **TVAE**(`benchmarks/tvae/`)는 별도 계약 — `tab-ddpm/CTGAN`의 `TVAESynthesizer` 기반, **무조건부**, CLI·체크포인트 형식이 다릅니다.
  ```bash
  python benchmarks/tvae/train_tvae.py --csv src/data/snaped_dataset.csv --out outputs/tvae --epochs 300
  python benchmarks/tvae/sample_tvae.py --ckpt outputs/tvae/tvae.obj --meta outputs/tvae/tvae_meta.pkl --out outputs/tvae/samples.csv --n 20
  ```

---

## 5. MCI 시뮬레이션 (MCI_ADV2)

흐름: **좌표 생성 → 시나리오 생성 → 시뮬레이션 → (검증) → 시각화**

### 5-1. 좌표 생성
```bash
# 행정구역 500m 그리드 후보 좌표
python MCI_ADV2/grid/grid_generator.py --region daejeon --grid-size 500
# → ./MCI_ADV2/scenarios/{region}_{exp_id}/grid_metadata.csv

# OSRM/Overpass 기반 랜덤(또는 custom) 좌표 + 도로 스냅
python MCI_ADV2/sce_src/random_coordinate_generator.py --mode random --region daejeon --num-points 100 --radius-m 250
# random → scenarios/national_<exp>/random_metadata.csv, custom → scenarios/custom_<exp>/custom_metadata.csv
```

### 5-2. 시나리오 생성 — `sce_src/make_csv_yaml_dynamic.py`
한 좌표에 대해 병원/구급차 선택·도로 거리/시간(Kakao 또는 OSRM)을 계산하고 좌표별 CSV + `config_*.yaml`을 생성합니다.
```bash
python MCI_ADV2/sce_src/make_csv_yaml_dynamic.py --base_path <repo> --latitude 36.3666 --longitude 127.3837 \
  --incident_size 30 --amb_count 30 --total_samples 30
```
주요 인자: `--rygb R Y G B`(중증도 비율), `--is_use_time True`(True=Kakao duration/False=OSRM 거리·속도), `--osrm_url`(ENV `MCI_OSRM_URL`), `--kakao_api_key`(ENV `KAKAO_API_KEY`), 병원 수 정책 `--fixed_hos_num`/`--min_hos_num`/`--max_hos_num`.

**배치 실행** — `sce_src/batch_experiment.py` (메타데이터 CSV의 모든 좌표에 대해 시나리오→시뮬 수행)
```bash
python MCI_ADV2/sce_src/batch_experiment.py --base_path <repo> --random_metadata <metadata.csv> \
  --config_template sim_src/config.yaml
```
- `--only_indices 1-5,10`로 실패분만 재시도, `--rygb`로 공통 중증도 지정.

### 5-3. 시뮬레이션 — `sim_src/main.py`
시나리오를 읽어 64개 출동 규칙 × `totalSamples`를 실행하고 reward/time/PDR 결과표를 씁니다. (`sim_src/` 디렉터리에서 실행)
```bash
cd MCI_ADV2/sim_src
python main.py --config_path ./config.yaml [--trace] [--log]
```
- `config.yaml`: 환자/병원/구급차/UAV 데이터 경로(`{EXP_ID}`/`{COORD}` 템플릿), 규칙 격자(START/ReSTART × 병원선택 × red/yellow → 64규칙), `totalSamples: 30`, `output_path`.
- 출력: `results/{EXP_ID}/{COORD}/results_{COORD}.txt`, `*_stat.txt` (+ `--trace` 시 `trace_*.json.gz`).

### 5-4. 고속 코어 (sim_src_upgrade)
결과 동일성을 보장하며 빠르게 돌리는 드롭인 가속 코어입니다.
```bash
# 고속 코어 주입 실행 (드리프트 체크 + 등가성 preflight, 실패 시 원본 코어로 폴백)
python MCI_ADV2/sim_src_upgrade/drivers/run_sim_fast.py -- --config_path <cfg.yaml>
#   실행 로그 '[fastcore] CORE=fast|origin' 로 어떤 코어가 돌았는지 표시

# 원본 대비 드리프트 게이트(G0)
python MCI_ADV2/sim_src_upgrade/origin_sync.py --check     # (--write 로 baseline 갱신)

# 원본 vs 고속 코어 5개 지표 비트동일성 검증
python MCI_ADV2/sim_src_upgrade/verify/sim_equivalence.py --config <cfg.yaml> --n_eps 3
```

### 5-5. 시각화 — Streamlit 대시보드
```bash
streamlit run MCI_ADV2/vis_src/MCI_Streamlit.py
```
- 멀티페이지: `pages/Generate.py`(시나리오 생성), `pages/BatchExperiment.py`(좌표→시나리오→시뮬→시각화 파이프라인), `pages/ResultsCompare.py`(결과 비교).

---

## 6. 도로망 차단 분석 (road_disruption)

"어떤 도로(way)를 끊으면 MCI 대응이 가장 나빠지는가"를 분석하는 **별도 실험 라인**(torch 미사용). OSM pbf에서 **way를 삭제**해 별도 포트에 OSRM을 띄우고, 동일 OD를 원본 OSRM과 비교합니다. `MCI_ADV2/` 시뮬레이터와 `src/data/road_network.py` 스냅을 재사용합니다.

> 차단 대상은 `config/cuts/*.yaml`의 `CutSpec`으로 선언합니다(selector: `way_id`, `highway_class`, `osm_tag`, `bbox`, `radius`, `polygon` 등). 지역 정의는 `config/regions.yaml`.

**① 지역 1회 세팅**
```bash
python road_disruption/scripts/extract_region_pbf.py --region daejeon      # 전국 pbf → 지역 pbf
python road_disruption/scripts/build_way_index.py   --region daejeon       # 지역 pbf → WayIndex(npz/parquet)
python road_disruption/scripts/build_region_osrm.py --region daejeon       # 지역 baseline OSRM 빌드/서브
```

**② 차단 후보 조사**
```bash
python road_disruption/scripts/whats_here.py --region daejeon --at 36.37,127.38 --radius 150   # 좌표 주변 way 나열
python road_disruption/scripts/route_ways.py --route <tag> --region daejeon                     # 경로가 지난 way 목록
python road_disruption/scripts/scenario_route_ways.py --scenario <dir> --region daejeon --label <en>  # 시나리오 전체 경로 way 합집합
```

**③ 차단 실행 / 경로 비교**
```bash
python road_disruption/scripts/cut_by_delete.py --spec config/cuts/<name>.yaml --port 5010   # way 삭제 OSRM 서브
python road_disruption/scripts/dump_route.py --port 5010 --from 36.37,127.38 --to 36.40,127.40 --tag collapsed  # OSRM 조회(유일)
python road_disruption/scripts/compare_routes.py --normal <tag> --collapsed <tag> --spec <yaml>  # 전후 비교 + HTML 지도
```

**④ 스윕(민감도) 실험**
```bash
# 단일 way씩 차례로 끊어 9개 지표 기록 (재개 가능)
python road_disruption/scripts/run_cut_sweep.py --ways <tag.json> --region daejeon --latitude 36.37 --longitude 127.38 --total_samples 10

# k개 way 동시 차단 조합 (중첩효과/초가산성 분석)
python road_disruption/scripts/run_multicut_sweep.py --ways <tag.json> --region daejeon --latitude 36.37 --longitude 127.38 --k 2
```

**⑤ 해설 / 플롯**
```bash
python road_disruption/scripts/inspect_cut_ways.py --list <json> --region daejeon --label <l> --scenario <dir>
python road_disruption/scripts/plot_inspect_map.py --label <l>         # 우회경로 Leaflet HTML
python road_disruption/scripts/plot_sweep.py --label <l> --metric pdr  # 스윕 결과 → PNG (MCI env 파이썬으로 실행 권장)
```
> `roaddis/` 패키지(WayIndex, selectors, CutSpec, osrm docker 제어 등)는 위 스크립트들이 import하는 공용 라이브러리입니다.

---

## 7. 시각화 앱 / 노트북

```bash
# 전국 생성좌표 + 병원/소방서 시설 인터랙티브 지도 (mci_viz 환경)
streamlit run app/app.py
```
- `notebooks/` : `pdr_analysis.ipynb`(샘플링 vs 시뮬 비교), `national_map.ipynb`(전국 지도), `q-bin_sampling.py`, `viz_national_4x4.py`, `viz_daejeon_4x4.py`, `viz_facilities_national_plotly.py` 등.
- `app/app.py` 및 대용량 노트북 2종(`national_map.ipynb`, `pdr_analysis.ipynb`)은 입력 경로가 하드코딩되어 있고, 노트북 2종은 용량이 커 **추적에서 제외(로컬 보관)** 되어 있습니다(아래 참조).

---

## 알려진 이슈 / 주의

- **`requirements_mci_viz.txt`**: conda export본이라 `file:///...` 로컬 경로가 섞여 있어 다른 PC에서 그대로 `pip install` 되지 않습니다. 참고용으로만 사용하고, 시각화 환경은 `streamlit/folium/plotly/geopandas`를 수동 설치하세요.
- **대용량 파일 추적 제외**: `*.pkl *.npz *.xlsx`, shapefile(`ctprvn.*`), 실행 로그(`*.log`, `experiment_logs/`), 그리고 `notebooks/national_map.ipynb`·`pdr_analysis.ipynb`는 `.gitignore` 처리되어 **로컬에만** 존재합니다. 캐시/원천 데이터는 각 `build_*` 스크립트로 재생성하세요.
- **road_disruption 문서 불일치**: 일부 예시가 `scripts/list_candidates.py`, `scripts/serve_maps.py`를 언급하지만 현재 브랜치에는 해당 파일이 없습니다.
- **`road_disruption/scripts/build_way_index.py`**: 본문이 주석 처리된 `--locations` 인자를 참조해 그대로 실행 시 오류 가능성이 있습니다(사용 전 확인 필요).
- **`sce_src/random_coordinate_generator.py`**: 실행 말미에 더 이상 존재하지 않는 `vis_src/custom_visualize.py`를 다음 단계로 안내하는 오래된 메시지가 남아 있습니다(시각화는 Streamlit 대시보드로 대체됨).
- 경로가 하드코딩된 스크립트(`src/data/dataset*.py`, 일부 `eval/`·`app/`·노트북)는 실행 전 코드 상단의 경로 상수를 확인하세요.

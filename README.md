## MLP Diffusion (lat, lon) conditioned on pdr&N

본 문서는 `tab-ddpm` 레포지토리의 구성 요소를 최대한 재사용하여 연속 좌표 `x=[lat, lon]`을 조건 `condition=[pdr, 사고 규모 N]`에 맞춰 생성하는ddpm의 파이프라인과 사용 방법입니다.

### 구성 개요
- **데이터**: `./src/data/dataset.csv` → 컬럼 `lat`, `lon`, `pdr_mean`, 'N' 사용
- **전처리/로더**: `src/data/data_module.py`
  - 표준화(StandardScaler)로 `x(lat, lon)`와 `condition'에 스케일러 적용
  - train/vaild/test 분할 
- **모델 빌더**: `src/model/build.py`
  - `tab_ddpm.modules.MLPDiffusion` 기반 conditional MLP 
- **학습 스크립트**: `scripts/train_mlp.py`
  - `TabDDPMGaussianScheduler` 구조로 학습 
  - 출력: `outputs/mlp_diffusion/model_last.pt`, `scalers.pkl`
- **샘플링 스크립트**: `scripts/sample_mlp.py`
  - 조건 `--cond <float>`, `--N <integer>`로 좌표를 생성하여 `outputs/mlp_diffusion/samples.csv`에 저장
  - pdr condition은 dataset.csv의 q3 값을 사용하여 샘플링하였습니다.
- **분석 ipynb**: 
  - notebooks/pdr_analysis.ipynb: 샘플링결과를 시뮬레이션 데이터와 비교하여 시각적으로 확인할 수 있습니다.

   

### 설치/환경
~~- PyTorch: CUDA 호환성 이슈 시 CPU로 우선 실행(연구실 공동 서버에서는 돌아가는걸로 확인)~~

```
    conda create -n tddpm python=3.9.7
    conda activate tddpm

    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    git submodule update --init --recursive
    pip install -r requirements.txt(MCI_DIFFUSION/requirments.txt)
```

### 데이터 형식
- CSV에 최소 컬럼:
  - `lat` (float)
  - `lon` (float)
  - `pdr_mean` (float)
  - `N` (integer)

### 학습
```bash
python scripts/train_mlp.py
```
- 기본 설정
  - `./src/data/dataset.csv` 필요
  - 타임스텝 T=1000
  - 배치 크기 256 
  - 옵티마이저 AdamW(lr=1e-3, wd=1e-4)
  - 현재 기본 **CPU** 강제(호환성 문제 우회)
- 출력
  - `outputs/mlp_diffusion/model_last.pt`: 학습된 모델 가중치 + 설정
  - `outputs/mlp_diffusion/scalers.pkl`: x/condition 스케일러(샘플링 역변환에 필요)


### 샘플링
```bash
python scripts/sample_mlp.py --cond 0.030349 --N 30 
```
- `--cond`: 원본 스케일의 `pdr_mean` 값 
- `--N`: 사고 규모 
- `samples.csv` 컬럼: `lat`, `lon`, `N`



### 디바이스/호환성
- 현재 CUDA 호환 이슈로 기본 CPU 실행

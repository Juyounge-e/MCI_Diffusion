## MLP Diffusion (lat, lon) conditioned on pdr

본 문서는 `tab-ddpm` 레포지토리의 구성 요소를 최대한 재사용하여 연속 좌표 `x=[lat, lon]`을 조건 `condition=[pdr_mean]`에 맞춰 생성하는 간단한 ddpm의 파이프라인과 사용 방법입니다.

### 구성 개요
- **데이터**: `src/data/dataset.csv` → 컬럼 `lat`, `lon`, `pdr_mean` 사용
- **전처리/로더**: `src/data/data_module.py`
  - 표준화(StandardScaler)로 `x`와 `condition` 스케일러 적용(지금은 위경도만)
  - train/vaild/test 분할 
- **모델 빌더**: `src/model/build.py`
  - `tab_ddpm.modules.MLPDiffusion` 기반 조건부 MLP 
- **학습 스크립트**: `scripts/train_mlp.py`
  - `TabDDPMGaussianScheduler` 구조로 학습 (현재 연속형만)
  - 출력: `outputs/mlp_diffusion/model_last.pt`, `scalers.pkl`
- **샘플링 스크립트**: `scripts/sample_mlp.py`
  - 조건 `--cond <float>`로 좌표를 생성하여 `outputs/mlp_diffusion/samples.csv`에 저장
  - **2600126 기준 cond 0.030349 사용**
- **분석 ipynb **: 
  - notebooks/pdr_analysis.ipynb: 테스트 데이터셋과 샘플링 데이터 셋 비교 

### 설치/환경
- PyTorch: CUDA 호환성 이슈 시 CPU로 우선 실행(연구실 공동 서버에서는 돌아가는걸로 확인)

```
    conda create -n tddpm python=3.9.7
    conda activate tddpm

    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt(MCI_DIFFUSION/requirments.txt)
```

### 데이터 형식
- CSV에 최소 컬럼:
  - `lat` (float)
  - `lon` (float)
  - `pdr_mean` (float)

### 학습
```bash
python scripts/train_mlp.py
```
- 기본 설정
  - 타임스텝 T=1000
  - 배치 크기 256 
  - 옵티마이저 AdamW(lr=1e-3, wd=1e-4)
  - 현재 기본 **CPU** 강제(호환성 문제 우회)
- 출력
  - `outputs/mlp_diffusion/model_last.pt`: 학습된 모델 가중치 + 설정
  - `outputs/mlp_diffusion/scalers.pkl`: x/condition 스케일러(샘플링 역변환에 필요)


### 샘플링
```bash
python scripts/sample_mlp.py --cond 0.030349 --n 100
```
- `--cond`: 원본 스케일의 `pdr_mean` 값 
- `samples.csv` 컬럼: `lat`, `lon`


### 확장 (범주형 추가)
- 원본 레포 구조를 따르므로, 범주형을 추가하려면 학습 시 `GaussianMultinomialDiffusion`의 `num_classes`를 실제 카테고리 크기 배열로 바꾸면 됩니다. 그러면 `mixed_loss`가 다항 손실을 자동 포함합니다.
  - 예) `num_classes=np.array([10, 5])` → 두 개의 범주형 컬럼(10, 5 클래스)

### 디바이스/호환성
- 현재 CUDA 호환 이슈로 기본 CPU 실행

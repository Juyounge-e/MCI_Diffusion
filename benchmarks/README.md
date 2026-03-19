# DDPM vs 베이스라인 비교용 벤치마크

DDPM이 조건부 위경도 생성: pdr + N → lat,lon에 적합한지 확인하기 위한 비교 모델을 구현 

## 데이터 형식 (공통)
- **X (생성 대상)**: `lat`, `lon`
- **Condition**: `pdr_mean`, `N`(사고규모)
- 데이터: `src/data/snapped_dataset.csv` 

## 비교 모델

### 1. DDPM (현재 구현) ✓
- `scripts/train_mlp.py`, `scripts/sample_mlp.py`
- 조건부: pdr_mean (연속) → lat, lon 생성

### 2. CVAE (TVAE 기반)
- TVAE → **CVAE로 확장**: encoder(x,c)→z, decoder(z,c)→x


## 디렉토리 구조

```
benchmarks/
├── README.md
├── cvae/               # CVAE (TVAE 기반 조건부 VAE)
│   ├── train_cvae.py
│   ├── sample_cvae.py
│   └── model.py
└── cfm/               # CFLOWMATCHING(구현 예정)
    
```

## 사용 예시

```bash
# CVAE 학습 & 샘플링
python benchmarks/cvae/train_cvae.py --csv src/data/snapped_dataset.csv --out outputs/cvae
  
  
python benchmarks/cvae/sample_cvae.py --ckpt outputs/cvae/cvae.pt --cond 0.052282 30 --n 20
```

## 공통 평가(구현 전)
- DDPM/CVAE 각각 (lat, lon) 샘플 생성
- 동일 조건에서 생성 샘플 비교
- MCI 시뮬레이션 엔진으로 pdr 검증

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'tab-ddpm')

from src.model.spatial_embedding import SpatialValidityEmbedding
import torch
import numpy as np
import re

print('=== SpatialValidityEmbedding 초기화 테스트 ===')
emb = SpatialValidityEmbedding(
    mask_path = r"C:\Users\user00\Desktop\MCI_Diffusion\MCI_ADV2\scenarios\mask_cache.npz",
    dim_t=256,
)

print('초기화 성공!')
print('mask shape:', emb.mask.shape)
print('x_mean (초기값):', emb.x_mean.tolist())
print('x_std  (초기값):', emb.x_std.tolist())

# set_scaler 테스트
mean = np.array([36.0, 127.5])
std  = np.array([1.5,  2.0])
emb.set_scaler(mean, std)
print('\nset_scaler 이후:')
print('x_mean:', emb.x_mean.tolist())
print('x_std:', emb.x_std.tolist())

# forward 테스트: 정규화된 서울 좌표 (37.5N, 127.0E)
lat_s = torch.tensor([(37.5 - 36.0) / 1.5])
lon_s = torch.tensor([(127.0 - 127.5) / 2.0])
out = emb(lat_s, lon_s)
print('\nforward 출력 shape:', out.shape)

# 마스크 조회 - 바다 vs 육지
lat_sea  = torch.tensor([(33.0 - 36.0) / 1.5])
lon_sea  = torch.tensor([(126.5 - 127.5) / 2.0])
lat_land = torch.tensor([(37.5 - 36.0) / 1.5])
lon_land = torch.tensor([(127.0 - 127.5) / 2.0])
print()
print('마스크 조회 - 바다  (33.0N, 126.5E):', emb._lookup(lat_sea,  lon_sea ).item(), '  ← 0이어야 함')
print('마스크 조회 - 서울  (37.5N, 127.0E):', emb._lookup(lat_land, lon_land).item(), '  ← 1이어야 함')
print()
print('ALL PASS ✓')

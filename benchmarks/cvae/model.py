"""
CVAE: tab-ddpm/CTGAN TVAE 기반 조건부 VAE
- CTGAN tvae.Encoder/Decoder 상속, 조건(cond) concat으로 확장
- encoder(x,c)->z, decoder(z,c)->x
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TABDDPM = os.path.join(_ROOT, "tab-ddpm")
_CTGAN = os.path.join(_TABDDPM, "CTGAN")
for _p in (_ROOT, _TABDDPM, _CTGAN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

try:
    from CTGAN.ctgan.synthesizers.tvae import Encoder, Decoder 
except ImportError:
    from ctgan.synthesizers.tvae import Encoder,Decoder 


class CondEncoder(Encoder):
    """TVAE Encoder 상속, 조건 concat: (x, c) -> mu, std, logvar"""

    def __init__(self, data_dim, cond_dim, compress_dims, embedding_dim):
        input_dim = data_dim + cond_dim
        super().__init__(input_dim, compress_dims, embedding_dim)
        self.data_dim = data_dim
        self.cond_dim = cond_dim

    def forward(self, x, c):
        inp = torch.cat([x, c], dim=-1)
        return super().forward(inp)


class CondDecoder(Decoder):
    """TVAE Decoder 상속, 조건 concat: (z, c) -> x_recon, sigma"""

    def __init__(self, embedding_dim, cond_dim, decompress_dims, data_dim):
        input_dim = embedding_dim + cond_dim
        super().__init__(input_dim, decompress_dims, data_dim)
        self.embedding_dim = embedding_dim
        self.cond_dim = cond_dim

    def forward(self, z, c):
        inp = torch.cat([z, c], dim=-1)
        return super().forward(inp)

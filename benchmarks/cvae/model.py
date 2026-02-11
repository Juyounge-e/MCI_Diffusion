# pyright: reportIncompatibleMethodOverride=false
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
import torch.nn as nn
import torch.nn.functional as F

try:
    from CTGAN.ctgan.synthesizers.tvae import Encoder, Decoder 
except ImportError:
    from ctgan.synthesizers.tvae import Encoder,Decoder 


class CondEncoder(Encoder):
    """TVAE Encoder 상속, 조건 concat: (x, c) -> mu, std, logvar"""

    class SplitCondEmbedding(nn.Module):
        def __init__(self, cond_dim, cond_emb_dim):
            super().__init__()
            self.cond_dim = cond_dim
            self.cond_emb_dim = cond_emb_dim
            self.embs = nn.ModuleList([nn.Linear(1, cond_emb_dim) for _ in range(cond_dim)])

        def forward(self, c):
            if c.dim() == 1:
                c = c.unsqueeze(1)
            out = None
            for i, emb_fn in enumerate(self.embs):
                emb = F.silu(emb_fn(c[:, i:i + 1]))
                out = emb if out is None else (out + emb)
            if out is None:
                raise RuntimeError("조건 임베딩 계산에 실패했습니다.")
            return out

    def __init__(
        self,
        data_dim,
        cond_dim,
        compress_dims,
        embedding_dim,
        cond_emb_dim=None,
        split_cond_embed=False,
    ):
        self.split_cond_embed = split_cond_embed
        self.cond_emb_dim = cond_emb_dim if cond_emb_dim is not None else embedding_dim
        cond_in_dim = self.cond_emb_dim if self.split_cond_embed else cond_dim
        input_dim = data_dim + cond_in_dim
        super().__init__(input_dim, compress_dims, embedding_dim)
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        if self.split_cond_embed:
            self.cond_embed = CondEncoder.SplitCondEmbedding(cond_dim, self.cond_emb_dim)

    def forward(self, x, c):  # pyright: ignore[reportIncompatibleMethodOverride]
        c_inp = self.cond_embed(c) if self.split_cond_embed else c
        inp = torch.cat([x, c_inp], dim=-1)
        return super().forward(inp)


class CondDecoder(Decoder):
    """TVAE Decoder 상속, 조건 concat: (z, c) -> x_recon, sigma"""

    def __init__(
        self,
        embedding_dim,
        cond_dim,
        decompress_dims,
        data_dim,
        cond_emb_dim=None,
        split_cond_embed=False,
    ):
        self.split_cond_embed = split_cond_embed
        self.cond_emb_dim = cond_emb_dim if cond_emb_dim is not None else embedding_dim
        cond_in_dim = self.cond_emb_dim if self.split_cond_embed else cond_dim
        input_dim = embedding_dim + cond_in_dim
        super().__init__(input_dim, decompress_dims, data_dim)
        self.embedding_dim = embedding_dim
        self.cond_dim = cond_dim
        if self.split_cond_embed:
            self.cond_embed = CondEncoder.SplitCondEmbedding(cond_dim, self.cond_emb_dim)

    def forward(self, z, c):  # pyright: ignore[reportIncompatibleMethodOverride]
        c_inp = self.cond_embed(c) if self.split_cond_embed else c
        inp = torch.cat([z, c_inp], dim=-1)
        return super().forward(inp)

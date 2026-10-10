"""TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting.

Paper: Hu et al., ICML 2025. https://arxiv.org/abs/2501.13041
Official code: TROUBADOUR000/TimeFilter (no license file), so the model below
follows the MIT-licensed thuml/Time-Series-Library port (models/TimeFilter.py,
HEAD 4e938a1) and layers/TimeFilter_layers.py is vendored unchanged.

Deviations from the TSLib implementation:
1. Internal normalization is off by default so that the external normalizer
   is the only one (``--tf_internal_norm true`` restores the official
   non-affine instance normalization for reproduction checks).
2. The router's load-balancing loss is kept in ``self.aux_loss``; the
   training loop adds ``aux_loss_weight * aux_loss`` (0.05 in the official
   training code), which the TSLib port drops.
3. The spatial/temporal masks are built once (vectorized, identical to the
   official ``_get_mask``) instead of on every forward pass.
4. Only the long-term-forecast task path is kept.
"""

import torch
import torch.nn as nn

from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import TimeFilter_Backbone


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)

    def forward(self, x):
        # x: [B, C*T]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, C*N, P]
        x = self.patch_proj(x)  # [B, C*N, D]
        if self.pos:
            x += self.pe(x)
        return x


def build_masks(seq_len, n_vars, patch_len):
    """[L, 3, L] spatial / temporal / other masks of the official code."""
    L = seq_len * n_vars // patch_len
    N = seq_len // patch_len
    k = torch.arange(L).unsqueeze(1)
    j = torch.arange(L).unsqueeze(0)
    not_self = j != k
    spatial = (j % N == k % N) & not_self
    start = k // N * N
    temporal = (j >= start) & (j < start + N) & not_self
    other = ~(spatial | temporal) & not_self
    return torch.stack([spatial, temporal, other], dim=1).float()


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)  # N

        # Filter
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p

        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.pos)
        self.backbone = TimeFilter_Backbone(self.dim, self.n_vars, self.d_ff,
                                            configs.n_heads, configs.e_layers, self.top_p, configs.dropout,
                                            self.seq_len * self.n_vars // self.patch_len)
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

        self.use_internal_norm = bool(getattr(configs, 'tf_internal_norm', False))
        if self.use_internal_norm:
            self.norm = Normalize(configs.enc_in, affine=False)

        self.register_buffer('masks', build_masks(self.seq_len, self.n_vars, self.patch_len), persistent=False)
        self.aux_loss = None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [B, T, C]
        B, T, C = x_enc.shape
        x = self.norm(x_enc, 'norm') if self.use_internal_norm else x_enc
        x = x.permute(0, 2, 1).reshape(-1, C * T)  # [B, C*T]
        x = self.patch_embed(x)  # [B, C*N, D]

        x, moe_loss = self.backbone(x, self.masks, self.alpha)
        self.aux_loss = moe_loss

        x = self.head(x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2))  # [B, C, H]
        x = x.permute(0, 2, 1)
        if self.use_internal_norm:
            x = self.norm(x, 'denorm')
        return x

"""Dish-TS (Fan et al., AAAI 2023).

This is an interface-compatible adaptation of the authors' implementation:
https://github.com/weifantt/Dish-TS (DishTS.py)

The Dual-CONET is kept equal to the reference code: one bias-free linear map per
channel from the look-back window to the input-side and horizon-side levels
(phi_l, phi_h), followed by GELU, with the spreads xi_l and xi_h computed around
those levels and a learnable per-channel affine map.  As in the reference code,
the prior-guidance loss of the paper is not used.  The small interface changes let
Dish-TS share the data split, scaler, backbone, optimizer schedule, and evaluation
code of the other normalizers in this repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        init = configs.dish_init  # 'standard', 'avg' or 'uniform'
        n_series = configs.enc_in
        lookback = configs.seq_len
        if init == 'standard':
            weight = torch.rand(n_series, lookback, 2) / lookback
        elif init == 'avg':
            weight = torch.ones(n_series, lookback, 2) / lookback
        elif init == 'uniform':
            weight = torch.ones(n_series, lookback, 2) / lookback + torch.rand(n_series, lookback, 2) / lookback
        else:
            raise ValueError(f"unknown dish_init '{init}'")
        self.reduce_mlayer = nn.Parameter(weight)
        self.gamma = nn.Parameter(torch.ones(n_series))
        self.beta = nn.Parameter(torch.zeros(n_series))

    def normalize(self, batch_x):
        # batch_x: B*L*D
        theta = F.gelu(torch.bmm(batch_x.permute(2, 0, 1), self.reduce_mlayer).permute(1, 2, 0))  # B*2*D
        phil, phih = theta[:, :1, :], theta[:, 1:, :]
        denom = batch_x.shape[1] - 1
        xil = torch.sum((batch_x - phil) ** 2, dim=1, keepdim=True) / denom
        xih = torch.sum((batch_x - phih) ** 2, dim=1, keepdim=True) / denom
        batch_x = (batch_x - phil) / torch.sqrt(xil + 1e-8) * self.gamma + self.beta
        return batch_x, (phih, xih)

    def de_normalize(self, batch_x, statistics):
        # batch_x: B*H*D (forecasts)
        phih, xih = statistics
        return (batch_x - self.beta) / self.gamma * torch.sqrt(xih + 1e-8) + phih

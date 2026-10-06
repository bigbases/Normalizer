"""Frequency Adaptive Normalization (FAN).

This is an interface-compatible adaptation of the authors' Apache-2.0
implementation: https://github.com/wayne155/FAN

The decomposition and the frequency predictor are intentionally kept equal to
the reference implementation.  The small interface changes let FAN share the
same data split, scaler, backbone, optimizer schedule, and evaluation code as
the other normalizers in this repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def main_frequency_part(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return residual and the reconstruction from the top-k RFFT bins."""
    spectrum = torch.fft.rfft(x, dim=1)
    effective_k = min(int(k), spectrum.shape[1])
    if effective_k < 1:
        raise ValueError("FAN freq_topk must be at least 1")
    indices = torch.topk(spectrum.abs(), effective_k, dim=1).indices
    mask = torch.zeros_like(spectrum)
    mask.scatter_(1, indices, 1)
    main = torch.fft.irfft(spectrum * mask, n=x.shape[1], dim=1).real.float()
    return x - main, main


class FrequencyPredictor(nn.Module):
    """The MLPfreq network used in the official FAN implementation."""

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.main_encoder = nn.Sequential(nn.Linear(seq_len, 64), nn.ReLU())
        self.forecaster = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len),
        )

    def forward(self, main: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        # Inputs are [batch, channel, time].
        return self.forecaster(torch.cat([self.main_encoder(main), raw], dim=-1))


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.freq_topk = int(configs.freq_topk)
        self.aux_weight = float(configs.fan_aux_weight)
        self.model_freq = FrequencyPredictor(self.seq_len, self.pred_len)

    def normalize(self, batch_x: torch.Tensor):
        residual, main = main_frequency_part(batch_x, self.freq_topk)
        predicted_main = self.model_freq(
            main.transpose(1, 2), batch_x.transpose(1, 2)
        ).transpose(1, 2)
        return residual, predicted_main

    def de_normalize(self, residual_forecast: torch.Tensor, predicted_main: torch.Tensor):
        return residual_forecast + predicted_main

    def training_loss(
        self,
        residual_forecast: torch.Tensor,
        target: torch.Tensor,
        predicted_main: torch.Tensor,
    ) -> torch.Tensor:
        """Official FAN objective: residual MSE plus main-frequency MSE."""
        target_residual, target_main = main_frequency_part(target, self.freq_topk)
        return F.mse_loss(residual_forecast, target_residual) + self.aux_weight * F.mse_loss(
            predicted_main, target_main
        )

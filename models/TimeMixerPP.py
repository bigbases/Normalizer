"""TimeMixer++ adapter for the LightNorm experiment runtime.

The paper's anonymous official archive currently returns HTTP 401.  This
adapter therefore targets PyPOTS' BSD-3-Clause ``BackboneTimeMixerPP``
implementation, which explicitly documents that it is inspired by the
official implementation.  See README_JOURNAL.md before using it for the final
paper: the source choice must be declared and locked before final runs.
"""

from __future__ import annotations

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.tmpp_use_internal_norm:
            raise ValueError(
                "TimeMixer++ internal normalization must remain disabled in "
                "a controlled external-normalizer comparison."
            )
        try:
            from pypots.nn.modules.timemixerpp import BackboneTimeMixerPP
        except ImportError as exc:
            raise ImportError(
                "TimeMixerPP requires the optional, pinned PyPOTS dependency. "
                "Install requirements-timemixerpp.txt after approving the "
                "documented fallback implementation."
            ) from exc

        self.backbone = BackboneTimeMixerPP(
            task_name="long_term_forecast",
            n_steps=configs.seq_len,
            n_features=configs.enc_in,
            n_pred_steps=configs.pred_len,
            n_pred_features=configs.c_out,
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            d_ffn=configs.d_ff,
            n_heads=configs.n_heads,
            dropout=configs.dropout,
            top_k=configs.top_k,
            n_kernels=configs.n_kernels,
            channel_mixing=configs.channel_mixing,
            channel_independence=configs.channel_independence,
            downsampling_layers=configs.down_sampling_layers,
            downsampling_window=configs.down_sampling_window,
            downsampling_method=configs.down_sampling_method,
            use_future_temporal_feature=False,
            use_norm=False,
            embed=configs.embed,
            freq=configs.freq,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # The PyPOTS forecasting wrapper also calls this backbone without time
        # covariates.  Keeping that choice avoids mixing dataset-specific mark
        # encodings into the normalization comparison.
        return self.backbone.forecast(x_enc, None)

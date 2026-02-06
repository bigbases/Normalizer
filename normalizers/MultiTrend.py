import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.decomposition import series_decomp  # SMA-based: returns (seasonal, trend)


class DownUpTrendPredictor(nn.Module):
    """
    Downsample -> predict (linear/MLP) -> upsample
    trend: [B, L, D] -> trend_out: [B, H, D]
    """
    def __init__(self, seq_len, pred_len, down_ratio=4, hidden_dim=64, use_mlp=False):
        super().__init__()
        self.down_ratio = max(1, int(down_ratio))
        self.down_len = max(1, seq_len // self.down_ratio)
        self.pred_len = pred_len
        self.out_down_len = max(1, pred_len // self.down_ratio)

        if use_mlp:
            self.net = nn.Sequential(
                nn.Linear(self.down_len, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_down_len),
            )
        else:
            self.net = nn.Linear(self.down_len, self.out_down_len)

    def forward(self, trend):
        # trend: [B, L, D]
        B, L, D = trend.shape
        r = self.down_ratio

        # downsample along time (operate per-channel)
        trend_down = F.avg_pool1d(trend.transpose(1, 2), kernel_size=r, stride=r)  # [B, D, L//r]
        # predict downsampled horizon
        pred_down = self.net(trend_down)  # [B, D, H//r]
        # upsample back to pred_len
        trend_up = F.interpolate(pred_down, size=self.pred_len, mode="linear", align_corners=False)  # [B, D, H]
        return trend_up.transpose(1, 2)  # [B, H, D]


class MultiScaleSMADecomposer(nn.Module):
    """
    Iterative multi-scale decomposition:
      x -> (s1, t1)
      s1 -> (s2, t2)
      s2 -> (s3, t3)
      ...
    Returns: final_seasonal, [t1, t2, ...]
    """
    def __init__(self, kernel_lens=(13, 25, 97)):
        super().__init__()
        self.kernel_lens = list(kernel_lens)
        self.decomposers = nn.ModuleList([series_decomp(k) for k in self.kernel_lens])

    def forward(self, x):
        seasonal = x
        trends = []
        for decomp in self.decomposers:
            seasonal, trend = decomp(seasonal)
            trends.append(trend)
        return seasonal, trends


class Model(nn.Module):
    """
    Multi-scale decomposition with kernel_lens=[13, 25, 97] (default):
      x = seasonal_last + sum_i trend_i
    Each trend_i is normalized -> predicted -> de-normalized -> summed into trend_out_sum.
    Seasonal_last is normalized (s_norm) and fed into the forecasting backbone (outside this module),
    then de_normalize() adds back seasonal stats + trend_out_sum.
    """
    def __init__(self, configs):
        super().__init__()

        # --- defaults / required ---
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.s_norm = bool(configs.s_norm)
        self.t_norm = bool(configs.t_norm)
        self.affine = bool(configs.affine)

        # multi-scale kernels (requested default)
        kernel_lens = getattr(configs, "kernel_lens", [13, 25, 97])
        self.ms_decomposer = MultiScaleSMADecomposer(kernel_lens=kernel_lens)
        self.num_stages = len(kernel_lens)

        # --- affine params (seasonal + per-stage trend) ---
        if self.affine:
            # seasonal affine
            self.s_gamma = nn.Parameter(torch.ones(self.enc_in))
            self.s_beta  = nn.Parameter(torch.zeros(self.enc_in))
            # trend affine per stage
            self.t_gamma = nn.Parameter(torch.ones(self.num_stages, self.enc_in))
            self.t_beta  = nn.Parameter(torch.zeros(self.num_stages, self.enc_in))
        else:
            self.register_buffer("s_gamma", torch.ones(self.enc_in), persistent=False)
            self.register_buffer("s_beta",  torch.zeros(self.enc_in), persistent=False)
            self.register_buffer("t_gamma", torch.ones(self.num_stages, self.enc_in), persistent=False)
            self.register_buffer("t_beta",  torch.zeros(self.num_stages, self.enc_in), persistent=False)

        # --- trend predictors (one per stage) ---
        down_ratio = getattr(configs, "down_ratio", 4)
        use_mlp = bool(getattr(configs, "use_mlp", False))
        hidden_dim = getattr(configs, "t_ff", 64)

        self.trend_predictors = nn.ModuleList([
            DownUpTrendPredictor(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                down_ratio=down_ratio,
                hidden_dim=hidden_dim,
                use_mlp=use_mlp
            )
            for _ in range(self.num_stages)
        ])

        # seasonal stats cache (for de_normalize)
        self._s_avg = None
        self._s_var = None

    @staticmethod
    def _safe_stats_like(x):
        # returns (zero, one) broadcastable to x: [B, 1, D]
        B, _, D = x.shape
        device, dtype = x.device, x.dtype
        zero = torch.zeros((B, 1, D), device=device, dtype=dtype)
        one  = torch.ones((B, 1, D), device=device, dtype=dtype)
        return zero, one

    def normalize(self, batch_x):
        """
        batch_x: [B, L, D]
        returns:
          seasonal_norm: [B, L, D]   (to feed into forecasting backbone)
          trend_out_sum: [B, H, D]   (already de-normalized; add in de_normalize)
        """
        # 1) multi-scale decomposition
        seasonal_last, trends = self.ms_decomposer(batch_x)  # seasonal_last: [B,L,D], trends: list of [B,L,D]

        # 2) seasonal normalization (stats cached for de_normalize)
        if self.s_norm:
            s_avg = torch.mean(seasonal_last, dim=1, keepdim=True).detach()
            s_var = torch.var(seasonal_last, dim=1, keepdim=True, unbiased=False).detach()
        else:
            s_avg, s_var = self._safe_stats_like(seasonal_last)

        self._s_avg, self._s_var = s_avg, s_var

        seasonal_norm = (seasonal_last - s_avg) / torch.sqrt(s_var + 1e-8)
        seasonal_norm = seasonal_norm * self.s_gamma + self.s_beta  # broadcast over [B,L,D]

        # 3) trend prediction per stage (normalize -> predict -> de-normalize), then sum
        trend_out_sum = 0.0
        for i, trend_i in enumerate(trends):
            if self.t_norm:
                t_avg = torch.mean(trend_i, dim=1, keepdim=True).detach()
                t_var = torch.var(trend_i, dim=1, keepdim=True, unbiased=False).detach()
            else:
                t_avg, t_var = self._safe_stats_like(trend_i)

            # normalize + affine (stage-wise)
            gamma_i = self.t_gamma[i]  # [D]
            beta_i  = self.t_beta[i]   # [D]

            trend_i_norm = (trend_i - t_avg) / torch.sqrt(t_var + 1e-8)
            trend_i_norm = trend_i_norm * gamma_i + beta_i

            # predict
            trend_out_i = self.trend_predictors[i](trend_i_norm)  # [B,H,D]

            # inverse affine + de-normalize to original trend scale
            trend_out_i = (trend_out_i - beta_i) / (gamma_i + 1e-8)
            trend_out_i = trend_out_i * torch.sqrt(t_var + 1e-8) + t_avg

            trend_out_sum = trend_out_sum + trend_out_i

        return seasonal_norm, trend_out_sum

    def de_normalize(self, x, statistics):
        """
        x: [B, H, D] seasonal-forecast in normalized domain
        statistics: trend_out_sum from normalize() (already de-normalized) -> [B, H, D]
        returns: forecast in original scale [B, H, D]
        """
        trend_out_sum = statistics

        # inverse seasonal affine + de-normalize using cached seasonal stats
        x = (x - self.s_beta) / (self.s_gamma + 1e-8)
        x = x * torch.sqrt(self._s_var + 1e-8) + self._s_avg

        # add predicted multi-scale trends
        x = x + trend_out_sum
        return x

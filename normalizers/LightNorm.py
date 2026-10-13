import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.decomposition import series_decomp


# ---------------------------------------------------------------------------
# Linear upsampling along the last dimension, bit-identical to
# F.interpolate(x, size, mode='linear', align_corners=False).
#
# ATen's CUDA kernel for 1-D linear upsampling runs one thread per output
# position and loops over batch x channels inside the thread, which is very
# slow for many channels (Traffic: 862), and its backward does the same with
# atomic adds. Here every output value is computed by its own thread with the
# kernel's arithmetic, fma(w0, a, w1 * b) in float32, and the backward is a
# matrix product with the interpolation weights. Source indices, weights and
# the arithmetic are checked once per device and length pair against
# F.interpolate; if they do not reproduce it exactly, F.interpolate is used.
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs
    triton = None

if triton is not None:
    @triton.jit
    def _upsample_kernel(x_ptr, out_ptr, i0_ptr, i1_ptr, w0_ptr, w1_ptr, n_out, in_len, out_len,
                         BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out
        row = offs // out_len
        col = offs - row * out_len
        i0 = tl.load(i0_ptr + col, mask=mask, other=0)
        i1 = tl.load(i1_ptr + col, mask=mask, other=0)
        w0 = tl.load(w0_ptr + col, mask=mask, other=0.0)
        w1 = tl.load(w1_ptr + col, mask=mask, other=0.0)
        a = tl.load(x_ptr + row * in_len + i0, mask=mask, other=0.0)
        b = tl.load(x_ptr + row * in_len + i1, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.math.fma(w0, a, w1 * b), mask=mask)


def _forward_triton(x, plan):
    out = torch.empty(x.shape[:-1] + (plan["out_len"],), dtype=x.dtype, device=x.device)
    n = out.numel()
    _upsample_kernel[(triton.cdiv(n, 1024),)](x, out, plan["i0"], plan["i1"], plan["w0"], plan["w1"],
                                              n, x.shape[-1], plan["out_len"], BLOCK=1024)
    return out


def _forward_float64(x, plan, rows_per_chunk=4096):
    """fma(w0, a, w1 * b) through float64 (exact product, one more rounding), in row chunks to bound memory."""
    rows = x.reshape(-1, x.shape[-1])
    out = torch.empty(rows.shape[0], plan["out_len"], dtype=x.dtype, device=x.device)
    w0 = plan["w0"].double()
    for start in range(0, rows.shape[0], rows_per_chunk):
        chunk = rows[start:start + rows_per_chunk]
        rounded = plan["w1"] * chunk.index_select(-1, plan["i1"])
        out[start:start + rows_per_chunk] = torch.addcmul(rounded.double(), w0, chunk.index_select(-1, plan["i0"])).float()
    return out.reshape(x.shape[:-1] + (plan["out_len"],))


class _LinearUpsample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan):
        ctx.plan = plan
        x = x.contiguous()
        return _forward_triton(x, plan) if plan["kernel"] == "triton" else _forward_float64(x, plan)

    @staticmethod
    def backward(ctx, grad):
        # d out[j] / d x[i] is the interpolation matrix; one dense product, no atomics
        return grad.matmul(ctx.plan["matrix"].t()), None


def _calibrate_upsample(in_len, out_len, device):
    with torch.no_grad():
        # float(in) / out as one IEEE float32 division, as in ATen (a CUDA tensor / scalar would use the reciprocal)
        scale = (torch.tensor(float(in_len), dtype=torch.float32)
                 / torch.tensor(float(out_len), dtype=torch.float32)).to(device)
        src = (scale * (torch.arange(out_len, dtype=torch.float32, device=device) + 0.5) - 0.5).clamp(min=0)
        i0 = src.long()
        w1 = src - i0
        i1 = torch.where(i0 < in_len - 1, i0 + 1, i0)
        w0 = 1 - w1
        matrix = torch.zeros(in_len, out_len, dtype=torch.float32, device=device)
        cols = torch.arange(out_len, device=device)
        matrix.index_put_((i0, cols), w0, accumulate=True)
        matrix.index_put_((i1, cols), w1, accumulate=True)
        plan = {"i0": i0, "i1": i1, "w0": w0, "w1": w1, "out_len": out_len, "matrix": matrix}
        generator = torch.Generator(device=device).manual_seed(0)
        probes = [torch.randn(3, 37, in_len, device=device, generator=generator) * s for s in (1e-3, 1.0, 1e3)]
        kernels = (["triton"] if triton is not None and device.type == "cuda" else []) + ["float64"]
        for kernel in kernels:
            plan["kernel"] = kernel
            try:
                if all(torch.equal(_LinearUpsample.apply(p, plan),
                                   F.interpolate(p, size=out_len, mode='linear', align_corners=False))
                       for p in probes):
                    return plan
            except Exception:  # e.g. Triton cannot compile here
                continue
    return None


_UPSAMPLE_PLANS = {}


def linear_upsample(x, size):
    """F.interpolate(x, size=size, mode='linear', align_corners=False) for x [N, C, W], same values."""
    if x.dtype != torch.float32:
        return F.interpolate(x, size=size, mode='linear', align_corners=False)
    key = (x.device.type, x.device.index, x.shape[-1], size)
    if key not in _UPSAMPLE_PLANS:
        _UPSAMPLE_PLANS[key] = _calibrate_upsample(x.shape[-1], size, x.device)
    plan = _UPSAMPLE_PLANS[key]
    if plan is None:
        return F.interpolate(x, size=size, mode='linear', align_corners=False)
    return _LinearUpsample.apply(x, plan)


class DownUpTrendPredictor(nn.Module):
    def __init__(self, seq_len, pred_len, down_ratio=4, hidden_dim=64, use_mlp=False):
        super().__init__()
        self.down_ratio = down_ratio
        self.down_len = max(1, seq_len // down_ratio)
        self.pred_len = pred_len

        # downsample → linear prediction (간소화)
        self.down_proj = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=down_ratio, stride=down_ratio)
        if use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(self.down_len, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max(1, pred_len // down_ratio))
            )
        else:
            self.linear = nn.Linear(self.down_len, max(1, pred_len // down_ratio))

    def forward(self, trend):
        """
        trend: [B, L, D]
        return: [B, H, D]
        """
        # D 차원 각각 독립적으로 처리
        trend_down = F.avg_pool1d(trend.transpose(1,2), kernel_size=self.down_ratio).transpose(1,2)  # [B, L//r, D]
        pred_down = self.linear(trend_down.transpose(1,2)).transpose(1,2)  # [B, H//r, D]
        trend_up = linear_upsample(pred_down.transpose(1,2), self.pred_len).transpose(1,2)
        return trend_up


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # affine parameters
        if configs.affine:
            self.s_gamma, self.s_beta = nn.Parameter(torch.ones(configs.enc_in)), nn.Parameter(torch.zeros(configs.enc_in))
            self.t_gamma, self.t_beta = nn.Parameter(torch.ones(configs.enc_in)), nn.Parameter(torch.zeros(configs.enc_in))
        else:
            self.t_gamma, self.t_beta = 1, 0
            self.s_gamma, self.s_beta = 1, 0

        # moving-average decomposition
        self.kernel_size = configs.kernel_size
        self.decomposer = series_decomp(configs.kernel_size)
        self.s_norm = configs.s_norm
        self.t_norm = configs.t_norm

        # ↓ 기존 Linear 대신 다운샘플 기반 predictor 사용
        self.trend_predictor = DownUpTrendPredictor(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            down_ratio=configs.down_ratio,
            use_mlp=configs.use_mlp,
            hidden_dim=configs.t_ff
        )

    def _decompose(self, x):
        """series_decomp(kernel_size): one replicate-pad copy instead of two repeats and a cat (same values)."""
        pad = (self.kernel_size - 1) // 2
        padded = F.pad(x.unsqueeze(1), (0, 0, pad, pad), mode='replicate').squeeze(1)
        trend = self.decomposer.moving_avg.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend

    def normalize(self, batch_x):
        batch_x, trend = self._decompose(batch_x)

        # Without s_norm/t_norm the statistics are 0 and 1, and in float32
        # (x - 0) / sqrt(1 + 1e-8) and x * sqrt(1 + 1e-8) + 0 give x back
        # exactly, so those full-size passes are skipped.
        if self.s_norm:
            avg = batch_x.mean(dim=1, keepdim=True).detach()
            var = batch_x.var(dim=1, keepdim=True, unbiased=False).detach()
            std = torch.sqrt(var + 1e-8)
            batch_x = (batch_x - avg) / std
        else:
            avg, var, std = batch_x.new_zeros(1, 1, 1), batch_x.new_ones(1, 1, 1), None

        self.avg, self.var, self.std = avg, var, std
        batch_x = batch_x.mul(self.s_gamma) + self.s_beta

        if self.t_norm:
            trend_avg = trend.mean(dim=1, keepdim=True).detach()
            trend_var = trend.var(dim=1, keepdim=True, unbiased=False).detach()
            trend_std = torch.sqrt(trend_var + 1e-8)
            trend = (trend - trend_avg) / trend_std
        trend = trend.mul(self.t_gamma) + self.t_beta

        trend_out = self.trend_predictor(trend)

        # affine inverse (gamma/beta가 Parameter면 device는 맞아있음)
        trend_out = (trend_out - self.t_beta) / (self.t_gamma + 1e-12)
        if self.t_norm:
            trend_out = trend_out * trend_std + trend_avg

        return batch_x, trend_out

    def de_normalize(self, x, statistics):
        # batch_x: B*H*D (forecasts)
        trend_out = statistics
        x = (x - self.s_beta) / self.s_gamma
        if self.s_norm:
            x = x * self.std + self.avg
        x = x + trend_out
        return x

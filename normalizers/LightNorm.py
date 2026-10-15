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


# ---------------------------------------------------------------------------
# Inference fast path, same values as the eager code below.
#
# At batch 1 the module is bound by kernel launches rather than arithmetic.
# Without gradients, the element-wise chains of normalize and de_normalize
# therefore run as single CUDA kernels compiled with jiterator. Every step uses
# an IEEE round-to-nearest intrinsic (__fadd_rn, __fmul_rn, __fdiv_rn,
# __fsqrt_rn), which the compiler never contracts into an FMA, so each value is
# rounded exactly as by the separate ATen kernels of the eager path, and the
# upsampling keeps the fma(w0, a, w1 * b) of linear_upsample. Reductions,
# pooling and the linear layers are the eager ATen kernels, applied to tensors
# with the eager memory layout so that ATen picks the same kernels. With
# gradients enabled (training) the eager path is used.
# ---------------------------------------------------------------------------
try:
    from torch.cuda import jiterator as _jiterator
except ImportError:
    _jiterator = None

_STD = "__fsqrt_rn(__fadd_rn(v, 1e-8f))"                          # sqrt(var + 1e-8)
_UP = "fmaf(w0, a, __fmul_rn(w1, b))"                              # linear_upsample
_INV = "__fdiv_rn(__fsub_rn(p, tb), __fadd_rn(tg, 1e-12f))"       # (p - t_beta) / (t_gamma + 1e-12)
_FUSED_CODE = {
    # (x - trend) * s_gamma + s_beta
    "res": "T f(T x, T t, T g, T b) { return __fadd_rn(__fmul_rn(__fsub_rn(x, t), g), b); }",
    # ((r - avg) / std) * s_gamma + s_beta
    "res_norm": f"T f(T r, T a, T v, T g, T b) {{ return __fadd_rn(__fmul_rn(__fdiv_rn(__fsub_rn(r, a), {_STD}), g), b); }}",
    # trend * t_gamma + t_beta
    "trend": "T f(T t, T g, T b) { return __fadd_rn(__fmul_rn(t, g), b); }",
    # ((trend - avg) / std) * t_gamma + t_beta
    "trend_norm": f"T f(T t, T a, T v, T g, T b) {{ return __fadd_rn(__fmul_rn(__fdiv_rn(__fsub_rn(t, a), {_STD}), g), b); }}",
    # inverse affine of the predicted trend, then the inverse trend normalization
    "inv": f"T f(T p, T tg, T tb) {{ return {_INV}; }}",
    "inv_norm": f"T f(T p, T tg, T tb, T a, T v) {{ return __fadd_rn(__fmul_rn({_INV}, {_STD}), a); }}",
    # the same with the upsampling folded in (p from the two gathered neighbours)
    "up_inv": f"T f(T a, T b, T w0, T w1, T tg, T tb) {{ T p = {_UP}; return {_INV}; }}",
    "up_inv_norm": f"T f(T a, T b, T w0, T w1, T tg, T tb, T av, T v) {{ T p = {_UP}; "
                   f"return __fadd_rn(__fmul_rn({_INV}, __fsqrt_rn(__fadd_rn(v, 1e-8f))), av); }}",
    # (y - s_beta) / s_gamma [* std + avg] + trend
    "denorm": "T f(T y, T g, T b, T tr) { return __fadd_rn(__fdiv_rn(__fsub_rn(y, b), g), tr); }",
    "denorm_norm": f"T f(T y, T g, T b, T a, T v, T tr) {{ "
                   f"return __fadd_rn(__fadd_rn(__fmul_rn(__fdiv_rn(__fsub_rn(y, b), g), {_STD}), a), tr); }}",
}
_FUSED = {}
# Above this many predicted values the upsampling gathers inside its Triton
# kernel instead of materializing both neighbours (memory traffic dominates).
_GATHER_MAX = 1 << 17
_PAD_INDEX = {}


def _fused(name):
    if name not in _FUSED:
        _FUSED[name] = _jiterator._create_jit_fn("template <typename T> " + _FUSED_CODE[name])
    return _FUSED[name]


def _replicate_index(length, pad, device):
    """Indices of the replicate-padded window: index_select gives F.pad(..., mode='replicate')."""
    key = (length, pad, device)
    if key not in _PAD_INDEX:
        _PAD_INDEX[key] = torch.arange(-pad, length + pad, device=device).clamp_(0, length - 1)
    return _PAD_INDEX[key]


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

    def _fusable(self, x):
        return (_jiterator is not None and x.is_cuda and x.dtype == torch.float32
                and not torch.is_grad_enabled() and isinstance(self.s_gamma, nn.Parameter))

    def _normalize_fused(self, x):
        """normalize() without gradients: same values in about half the kernel launches."""
        pad = (self.kernel_size - 1) // 2
        padded = x.index_select(1, _replicate_index(x.shape[1], pad, x.device))   # [B, L+2p, D] as F.pad
        trend = self.decomposer.moving_avg.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)
        if self.s_norm:
            residual = x - trend
            avg = residual.mean(dim=1, keepdim=True)
            var = residual.var(dim=1, keepdim=True, unbiased=False)
            batch_x = _fused("res_norm")(residual, avg, var, self.s_gamma, self.s_beta)
            self.avg, self.var, self.std = avg, var, None
        else:
            batch_x = _fused("res")(x, trend, self.s_gamma, self.s_beta)
            self.avg = self.var = self.std = None
        tg, tb = self.t_gamma, self.t_beta
        if self.t_norm:
            t_avg = trend.mean(dim=1, keepdim=True)
            t_var = trend.var(dim=1, keepdim=True, unbiased=False)
            trend = _fused("trend_norm")(trend, t_avg, t_var, tg, tb)
        else:
            trend = _fused("trend")(trend, tg, tb)

        predictor = self.trend_predictor
        down = F.avg_pool1d(trend.transpose(1, 2), kernel_size=predictor.down_ratio)     # [B, D, L/r]
        pred = predictor.linear(down)                                                     # [B, D, H/r]
        key = (x.device.type, x.device.index, pred.shape[-1], predictor.pred_len)
        if key not in _UPSAMPLE_PLANS:
            _UPSAMPLE_PLANS[key] = _calibrate_upsample(pred.shape[-1], predictor.pred_len, x.device)
        plan = _UPSAMPLE_PLANS[key]
        if plan is not None and pred.numel() <= _GATHER_MAX:
            pred = pred.contiguous()
            a = pred.index_select(-1, plan["i0"]).transpose(1, 2)                          # [B, H, D]
            b = pred.index_select(-1, plan["i1"]).transpose(1, 2)
            w0, w1 = plan["w0"].view(1, -1, 1), plan["w1"].view(1, -1, 1)
            trend_out = (_fused("up_inv_norm")(a, b, w0, w1, tg, tb, t_avg, t_var) if self.t_norm
                         else _fused("up_inv")(a, b, w0, w1, tg, tb))
        else:
            p = linear_upsample(pred, predictor.pred_len).transpose(1, 2)
            trend_out = (_fused("inv_norm")(p, tg, tb, t_avg, t_var) if self.t_norm
                         else _fused("inv")(p, tg, tb))
        self._fused_pass = True
        return batch_x, trend_out

    def normalize(self, batch_x):
        if self._fusable(batch_x):
            return self._normalize_fused(batch_x)
        self._fused_pass = False
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
        if getattr(self, "_fused_pass", False):
            if self._fusable(x):
                return (_fused("denorm_norm")(x, self.s_gamma, self.s_beta, self.avg, self.var, trend_out)
                        if self.s_norm else _fused("denorm")(x, self.s_gamma, self.s_beta, trend_out))
            if self.s_norm and self.std is None:
                self.std = torch.sqrt(self.var + 1e-8)
        x = (x - self.s_beta) / self.s_gamma
        if self.s_norm:
            x = x * self.std + self.avg
        x = x + trend_out
        return x

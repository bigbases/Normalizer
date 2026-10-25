"""GPU check: LightNorm with --lt_channel_mix gives the same trend forecast on the fused no-grad path
as on the regular path (run on a CUDA machine before the V3 diagnostics)."""
import sys, types
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from normalizers.LightNorm import Model

cfg = types.SimpleNamespace(affine=1, enc_in=7, kernel_size=25, s_norm=1, t_norm=1, seq_len=96, pred_len=96,
                            down_ratio=4, use_mlp=0, t_ff=64, lt_channel_mix=True)
torch.manual_seed(0)
m = Model(cfg).cuda().eval()
with torch.no_grad():
    m.trend_predictor.channel_mix.weight.normal_(0, 0.1)
    m.trend_predictor.channel_mix.bias.normal_(0, 0.1)
x = torch.randn(64, 96, 7, device="cuda")
y = torch.randn(64, 96, 7, device="cuda")
with torch.no_grad():
    xf, tf = m.normalize(x)
    out_fused = m.de_normalize(y, tf)
with torch.enable_grad():
    xr, tr = m.normalize(x)
    out_regular = m.de_normalize(y, tr)
diff = (out_fused - out_regular.detach()).abs().max().item()
print(f"fused path used: {m._fused_pass is not None}, max |fused - regular| = {diff:.2e}")
sys.exit(0 if diff < 1e-4 else 1)

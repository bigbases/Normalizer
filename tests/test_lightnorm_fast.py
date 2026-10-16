import itertools
import unittest
from argparse import Namespace

import torch
import torch.nn.functional as F

from normalizers import LightNorm as fast
from tests import reference_lightnorm as reference


class LinearUpsampleTest(unittest.TestCase):
    def test_values_match_interpolate(self):
        for in_len, out_len in ((24, 96), (48, 192), (180, 720), (12, 96)):
            x = torch.randn(3, 11, in_len) * 50
            self.assertTrue(torch.equal(fast.linear_upsample(x, out_len),
                                        F.interpolate(x, size=out_len, mode="linear", align_corners=False)))

    def test_gradient_matches_interpolate(self):
        x = torch.randn(2, 5, 24, requires_grad=True)
        weight = torch.randn(2, 5, 96)
        (fast.linear_upsample(x, 96) * weight).sum().backward()
        grad, x.grad = x.grad, None
        (F.interpolate(x, size=96, mode="linear", align_corners=False) * weight).sum().backward()
        torch.testing.assert_close(grad, x.grad)


class LightNormSpeedUpTest(unittest.TestCase):
    def test_same_outputs_and_gradients_as_reference(self):
        for kernel, s_norm, t_norm, use_mlp, (seq_len, pred_len) in itertools.product(
                (13, 25, 49), (0, 1), (0, 1), (0, 1), ((96, 96), (96, 720), (336, 192))):
            cfg = Namespace(enc_in=7, seq_len=seq_len, pred_len=pred_len, kernel_size=kernel, s_norm=s_norm,
                            t_norm=t_norm, affine=1, use_mlp=use_mlp, down_ratio=4, t_ff=64)
            torch.manual_seed(kernel + 7 * s_norm + 3 * t_norm + use_mlp)
            ref = reference.Model(cfg)
            with torch.no_grad():
                for p in ref.parameters():
                    p.add_(0.1 * torch.randn_like(p))
            new = fast.Model(cfg)
            new.load_state_dict(ref.state_dict())
            x = torch.randn(4, seq_len, 7) * 3 + torch.linspace(0, 5, seq_len)[None, :, None]
            y = torch.randn(4, pred_len, 7)
            results = []
            for model in (ref, new):
                xn, trend = model.normalize(x)
                forecast = model.de_normalize(xn[:, -1:, :] + y, trend)
                ((forecast - y) ** 2).mean().backward()
                results.append((xn.detach(), trend.detach(), forecast.detach(),
                                [p.grad for p in model.parameters() if p.grad is not None]))
            for a, b in zip(results[0][:3], results[1][:3]):
                self.assertTrue(torch.equal(a, b))
            for a, b in zip(results[0][3], results[1][3]):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


@unittest.skipUnless(torch.cuda.is_available() and fast._jiterator is not None, "needs CUDA and jiterator")
class LightNormFusedInferenceTest(unittest.TestCase):
    """Without gradients the jiterator path must give the eager values bit for bit."""

    @staticmethod
    def bits(t):
        return t.contiguous().view(torch.int32)

    def test_bit_identical_to_eager(self):
        dev = torch.device("cuda")
        shapes = ((96, 96, 7), (336, 720, 7), (96, 720, 321), (96, 192, 21), (720, 720, 862))
        for (seq_len, pred_len, channels), kernel, ratio, s_norm, t_norm, use_mlp, batch in itertools.product(
                shapes, (13, 25, 49), (4, 8), (0, 1), (0, 1), (0, 1), (1, 64)):
            cfg = Namespace(enc_in=channels, seq_len=seq_len, pred_len=pred_len, kernel_size=kernel,
                            s_norm=s_norm, t_norm=t_norm, affine=1, use_mlp=use_mlp, down_ratio=ratio, t_ff=64)
            torch.manual_seed(seq_len + kernel + ratio + 2 * s_norm + 3 * t_norm + use_mlp + batch)
            model = fast.Model(cfg).to(dev).eval()
            with torch.no_grad():
                for p in (model.s_gamma, model.t_gamma):
                    p.copy_(torch.rand(channels, device=dev) * 2 + 0.1)
                for p in (model.s_beta, model.t_beta):
                    p.copy_(torch.randn(channels, device=dev))
            x = torch.randn(batch, seq_len, channels, device=dev) * 5 \
                + torch.linspace(-3, 3, seq_len, device=dev)[None, :, None]
            y = torch.randn(batch, pred_len, channels, device=dev)
            with torch.enable_grad():                      # eager path
                e_x, e_t = model.normalize(x)
                e_y = model.de_normalize(y, e_t)
            with torch.no_grad():                          # fused path
                f_x, f_t = model.normalize(x)
                self.assertTrue(model._fused_pass)
                deferred = batch * channels * (pred_len // ratio) > fast._GATHER_MAX and fast.triton is not None
                self.assertEqual(isinstance(f_t, fast.TrendForecast), deferred)
                f_y = model.de_normalize(y, f_t)
                f_y_strided = model.de_normalize(y.transpose(1, 2).contiguous().transpose(1, 2), f_t)
                if deferred:
                    f_t = f_t.materialize()
            case = (seq_len, pred_len, channels, kernel, ratio, s_norm, t_norm, use_mlp, batch)
            for a, b in ((e_x, f_x), (e_t, f_t), (e_y, f_y), (e_y, f_y_strided)):
                self.assertTrue(torch.equal(self.bits(a.detach()), self.bits(b)), case)

    def test_deferred_trend_acts_as_tensor_and_saves_memory(self):
        dev = torch.device("cuda")
        batch, seq_len, pred_len, channels = 32, 336, 720, 862
        cfg = Namespace(enc_in=channels, seq_len=seq_len, pred_len=pred_len, kernel_size=25, s_norm=1,
                        t_norm=1, affine=1, use_mlp=0, down_ratio=4, t_ff=64)
        torch.manual_seed(0)
        model = fast.Model(cfg).to(dev).eval()
        x = torch.randn(batch, seq_len, channels, device=dev)
        with torch.enable_grad():
            _, e_t = model.normalize(x)
        e_t = e_t.detach()
        with torch.no_grad():
            model.normalize(x)                             # warm-up: plans, index caches, kernel check
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            f_x, f_t = model.normalize(x)
            held = torch.cuda.memory_allocated() - before - f_x.numel() * f_x.element_size()
            self.assertIsInstance(f_t, fast.TrendForecast)
            self.assertLess(held, e_t.numel() * e_t.element_size() // 3)
            target = torch.randn_like(e_t)
            self.assertEqual(f_t.shape, e_t.shape)
            self.assertTrue(torch.equal(F.mse_loss(f_t, target), F.mse_loss(e_t, target)))
            self.assertTrue(torch.equal(target - f_t, target - e_t))
            self.assertTrue(torch.equal(f_t[:, :, -1:], e_t[:, :, -1:]))
            self.assertTrue(torch.equal(f_t.detach(), e_t))


if __name__ == "__main__":
    unittest.main()

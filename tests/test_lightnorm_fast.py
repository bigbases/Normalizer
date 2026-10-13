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


if __name__ == "__main__":
    unittest.main()

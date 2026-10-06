import unittest
from types import SimpleNamespace

import torch

from normalizers.FAN import Model, main_frequency_part


class FANTest(unittest.TestCase):
    def test_decomposition_reconstructs_input(self):
        x = torch.randn(3, 96, 7)
        residual, main = main_frequency_part(x, 4)
        torch.testing.assert_close(residual + main, x)

    def test_shapes_and_joint_gradient(self):
        cfg = SimpleNamespace(seq_len=96, pred_len=24, freq_topk=4, fan_aux_weight=1.0)
        model = Model(cfg)
        x = torch.randn(2, 96, 7)
        target = torch.randn(2, 24, 7)
        residual_x, predicted_main = model.normalize(x)
        residual_forecast = residual_x[:, -24:, :].clone().requires_grad_(True)
        prediction = model.de_normalize(residual_forecast, predicted_main)
        self.assertEqual(tuple(prediction.shape), (2, 24, 7))
        loss = model.training_loss(residual_forecast, target, predicted_main)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()

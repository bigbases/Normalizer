import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import result_store as rs  # noqa: E402
import run_matrix as rm  # noqa: E402
from normalizers.DishTS import Model as DishTS  # noqa: E402


class DishTSTest(unittest.TestCase):
    def test_matches_reference_formulas(self):
        torch.manual_seed(0)
        cfg = SimpleNamespace(dish_init="standard", enc_in=3, seq_len=12)
        model = DishTS(cfg)
        x = torch.randn(4, 12, 3)
        y, (phih, xih) = model.normalize(x)
        theta = torch.nn.functional.gelu(torch.einsum("bld,dlk->bkd", x, model.reduce_mlayer))
        phil = theta[:, :1]
        xil = ((x - phil) ** 2).sum(1, keepdim=True) / 11
        self.assertTrue(torch.allclose(y, (x - phil) / torch.sqrt(xil + 1e-8), atol=1e-6))
        self.assertTrue(torch.allclose(phih, theta[:, 1:], atol=1e-6))
        out = torch.randn(4, 5, 3)
        self.assertTrue(torch.allclose(model.de_normalize(out, (phih, xih)), out * torch.sqrt(xih + 1e-8) + phih, atol=1e-6))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2 * 3 * 12 + 2 * 3)


class DishTSExchangeProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = rm.load_json(rm.PROTOCOL_PATH)
        cls.bases = rm.load_json(rm.BASE_CONFIGS_PATH)
        cls.plan = rs.load_plan()

    def test_dishts_grid_has_six_candidates(self):
        base = rm.selected_base_config(self.bases, "ETTh1", "DLinear")
        grid = rm.search_candidates(self.protocol, "ETTh1", "dishts", base)
        self.assertEqual(len(grid), 6)
        self.assertEqual({c["dish_init"] for c in grid}, {"standard", "avg", "uniform"})
        self.assertEqual({c["station_lr"] for c in grid}, {1e-4, 5e-4})   # normalizer and backbone learning rates

    def test_exchange_grids_include_official_values(self):
        base = rm.selected_base_config(self.bases, "Exchange", "DLinear")
        san = rm.search_candidates(self.protocol, "Exchange", "san", base)
        self.assertEqual({c["period_len"] for c in san}, {6, 12})
        self.assertEqual(len(san), 6)
        fan = rm.search_candidates(self.protocol, "Exchange", "fan", base)
        self.assertEqual({c["freq_topk"] for c in fan}, {1, 2, 4})
        for ds, k in (("ETTh2", 3), ("ETTm2", 5)):
            self.assertIn(k, self.protocol["datasets"][ds]["fan_k"])
        lt = rm.search_candidates(self.protocol, "Exchange", "lt", base, lt_grid="main")
        self.assertEqual(len(lt), 6)                            # supplied point searched: no Exp1 cells
        lt7 = rm.search_candidates(self.protocol, "ETTh1", "lt", rm.selected_base_config(self.bases, "ETTh1", "DLinear"), lt_grid="main")
        self.assertEqual(len(lt7), 5)

    def test_old_phases_do_not_gain_exchange_tasks(self):
        tasks = rs.build_tasks(self.plan, self.protocol, self.bases, {})
        exch = {t.phase for t in tasks if t.dataset == "Exchange"}
        self.assertTrue(exch)
        self.assertTrue(all(p.startswith("exch-") for p in exch), exch)
        dish = {t.phase for t in tasks if "dishts" in t.methods and t.dataset != "Exchange"}
        self.assertEqual(dish, {"dish-search", "dish-confirm", "dish-tuned", "dish-tuned-s2022", "dish-tuned-s2023"})


if __name__ == "__main__":
    unittest.main()

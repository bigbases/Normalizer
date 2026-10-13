import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from data_provider.data_loader import Dataset_Custom, shift_test_period
from experiments import result_store as rs
from experiments import run_matrix as rm

ROOT = Path(__file__).resolve().parents[1]


def synthetic_csv(path, n=600, channels=3):
    t = np.arange(n)
    df = pd.DataFrame({f"c{i}": np.sin(t / (7 + i)) + 0.002 * t * (i + 1) for i in range(channels - 1)})
    df["OT"] = np.cos(t / 11)
    df.insert(0, "date", pd.date_range("2020-01-01", periods=n, freq="h").astype(str))
    df.to_csv(path, index=False)


class ShiftDataTest(unittest.TestCase):
    def test_offsets_use_training_sigma_and_touch_only_the_test_period(self):
        values = np.random.RandomState(0).randn(100, 2) * [1.0, 3.0]
        sigma = values[:70].std(axis=0)
        level = shift_test_period(values, Namespace(test_shift="level", shift_level=1.0), 2, (0, 70), (80, 100))
        np.testing.assert_allclose(level[80:] - values[80:], np.tile(sigma, (20, 1)))
        np.testing.assert_array_equal(level[:80], values[:80])
        drift = shift_test_period(values, Namespace(test_shift="drift", shift_drift=2.0), 2, (0, 70), (80, 100))
        np.testing.assert_allclose(drift[80] - values[80], 0 * sigma)
        np.testing.assert_allclose(drift[99] - values[99], 2 * sigma)
        # other splits and the default are untouched
        self.assertIs(shift_test_period(values, Namespace(test_shift="level", shift_level=1.0), 0, (0, 70), (80, 100)), values)
        self.assertIs(shift_test_period(values, Namespace(), 2, (0, 70), (80, 100)), values)

    def test_custom_dataset_shifts_inputs_and_targets_of_test_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_csv(Path(tmp) / "syn.csv")
            def load(**shift):
                args = Namespace(augmentation_ratio=0, **shift)
                return Dataset_Custom(args, tmp, flag="test", size=[48, 24, 24], features="M", data_path="syn.csv",
                                      timeenc=1)
            clean, level = load(), load(test_shift="level", shift_level=1.0, shift_drift=2.0)
            diff = level.data_x - clean.data_x
            np.testing.assert_allclose(diff[:48], 0)                    # the seq_len rows before the test period
            np.testing.assert_allclose(diff[48:], np.tile(diff[48], (len(diff) - 48, 1)))
            self.assertTrue(np.all(diff[48:] > 0))


class ShiftExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = rm.load_json(rm.PROTOCOL_PATH)
        cls.bases = rm.load_json(rm.BASE_CONFIGS_PATH)
        cls.plan = rs.load_plan()

    def test_cells_and_command(self):
        cells = rs._cells(self.protocol, self.bases, "11_shift_robustness", "shift", "Weather", "DLinear", ["none"])
        self.assertEqual({(c["horizon"], c["seed"]) for c in cells}, {(96, s) for s in (2021, 2022, 2023)})
        final = rs._cells(self.protocol, self.bases, "3_frozen_backbone_comparison", "final", "Weather", "DLinear",
                          ["none"])
        self.assertFalse({c["run_id"] for c in cells} & {c["run_id"] for c in final})   # new identities
        cmd = rm.cell_command(cells[0], "/data", "/tmp/r.csv", "/tmp/ck")
        last = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
        self.assertEqual((last["--phase"], last["--shift_eval"], last["--shift_level"], last["--shift_drift"]),
                         ("shift", "true", "1.0", "2.0"))
        self.assertNotIn("--skip_test", cmd)

    def test_tuned_methods_wait_for_their_locks(self):
        tasks = {t.task_id: t for t in rs.build_tasks(self.plan, self.protocol, self.bases, {})}
        self.assertEqual(tasks["shift-none--Traffic--DLinear"].status, "ready")
        self.assertEqual(tasks["shift-tuned--Traffic--DLinear--san"].status, "blocked")
        self.assertEqual(tasks["shift-lt--Traffic--DLinear"].status, "blocked")


class ShiftEndToEndTest(unittest.TestCase):
    def test_result_row_carries_shift_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_csv(Path(tmp) / "syn.csv")
            out = Path(tmp) / "r.csv"
            cmd = [sys.executable, str(ROOT / "run_longExp.py"), "--is_training", "1", "--model", "DLinear",
                   "--model_id", "t", "--data", "custom", "--root_path", tmp, "--data_path", "syn.csv",
                   "--features", "M", "--enc_in", "3", "--dec_in", "3", "--c_out", "3", "--seq_len", "48",
                   "--label_len", "24", "--pred_len", "24", "--use_norm", "none", "--train_epochs", "1",
                   "--batch_size", "16", "--num_workers", "0", "--use_gpu", "false", "--itr", "1",
                   "--checkpoints", str(Path(tmp) / "ck"), "--station_root", str(Path(tmp) / "st"),
                   "--result_file", str(out), "--phase", "shift", "--shift_eval", "true"]
            env = {**os.environ, "LIGHTNORM_TORCH_THREADS": "1"}
            subprocess.run(cmd, cwd=tmp, env=env, check=True, capture_output=True)
            row = next(csv.DictReader(out.open()))
            extra = json.loads(row["Extra"])
            self.assertEqual(row["Phase"], "shift")
            self.assertEqual(set(extra), {"MSE_level", "MAE_level", "MSE_drift", "MAE_drift", "shift_level", "shift_drift"})
            self.assertGreater(extra["MSE_level"], 0)


if __name__ == "__main__":
    unittest.main()

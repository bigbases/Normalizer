import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider


def make_args(root, num_workers):
    return SimpleNamespace(data="custom", embed="timeF", batch_size=8, freq="h", root_path=str(root),
                           data_path="toy.csv", seq_len=24, label_len=12, pred_len=12, features="M",
                           target="OT", num_workers=num_workers, seed=2021, augmentation_ratio=0,
                           test_shift=None, shift_level=0.0, shift_drift=0.0)


def epochs(loader, n=3):
    """Batch contents of n epochs plus the global RNG state the training would continue from."""
    seen = []
    for _ in range(n):
        seen.append([batch[0].clone() for batch in loader])
    return seen, torch.get_rng_state()


class LoaderOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(rng.normal(size=(400, 4)), columns=["a", "b", "c", "OT"])
        frame.insert(0, "date", pd.date_range("2020-01-01", periods=400, freq="h").astype(str))
        frame.to_csv(Path(cls.tmp.name) / "toy.csv", index=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_loader(self, num_workers):
        torch.manual_seed(0)
        _, loader = data_provider(make_args(self.tmp.name, num_workers), "train")
        return epochs(loader)

    def test_worker_count_does_not_change_batches_or_rng(self):
        with_workers, rng_workers = self.run_loader(2)
        without, rng_without = self.run_loader(0)
        for epoch_a, epoch_b in zip(with_workers, without):
            self.assertEqual(len(epoch_a), len(epoch_b))
            for a, b in zip(epoch_a, epoch_b):
                self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(rng_workers, rng_without))

    def test_plain_single_process_loader_would_reorder_later_epochs(self):
        # guards the test above: a fresh iterator per epoch changes the order from epoch 2
        reference, _ = self.run_loader(2)
        torch.manual_seed(0)
        data_set, loader = data_provider(make_args(self.tmp.name, 0), "train")
        plain = DataLoader(data_set, batch_size=8, shuffle=True, num_workers=0, drop_last=True,
                           generator=loader.generator)
        loader.generator.manual_seed(2021)
        seen, _ = epochs(plain)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference[0], seen[0])))
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(reference[1], seen[1])))


if __name__ == "__main__":
    unittest.main()

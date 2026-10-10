import unittest
from argparse import Namespace

from experiments.run_matrix import (
    BASE_CONFIGS_PATH,
    PROTOCOL_PATH,
    build_cells,
    canonical_hash,
    cell_command,
    load_json,
    search_candidates,
    selected_base_config,
    timefilter_args,
)


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_json(PROTOCOL_PATH)
        cls.bases = load_json(BASE_CONFIGS_PATH)

    def test_t3_backbones_use_official_lookback_96(self):
        # TimeFilter rows follow the official look-back-96 scripts, including
        # the arguments that change with the horizon, and TimeXer uses look-back 96.
        cfg = selected_base_config(self.bases, "ETTh1", "TimeFilter")
        self.assertEqual((cfg["seq_len"], cfg["n_heads"], cfg["patch_len"]), (96, 4, 2))
        self.assertEqual(timefilter_args(cfg, 720)["d_ff"], 128)
        self.assertEqual(timefilter_args(selected_base_config(self.bases, "ETTm1", "TimeFilter"), 720)["patch_len"], 16)
        traffic = timefilter_args(selected_base_config(self.bases, "Traffic", "TimeFilter"), 96)
        self.assertEqual((traffic["train_epochs"], traffic["top_p"], traffic["lradj"]), (30, 0.0, "cosine"))
        self.assertEqual(selected_base_config(self.bases, "Traffic", "TimeXer")["seq_len"], 96)
        with self.assertRaises(ValueError):
            selected_base_config(self.bases, "ETTh1", "TimeMixerPP")  # retired

    def test_timefilter_arguments_override_fixed_training_args(self):
        args = Namespace(experiment="3_frozen_backbone_comparison", stage="final", datasets="Traffic",
                         backbones="TimeFilter", methods="none", locks=None, shortlist=None)
        cell = next(c for c in build_cells(args, self.protocol, self.bases) if c["horizon"] == 96)
        cmd = cell_command(cell, "/data", "/tmp/r.csv", "/tmp/ck")
        last = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
        self.assertEqual((last["--train_epochs"], last["--lradj"], last["--patch_len"]), ("30", "cosine", "96"))

    def test_fan_has_equal_capped_search_budget(self):
        base = selected_base_config(self.bases, "ETTh1", "DLinear")
        candidates = search_candidates(self.protocol, "ETTh1", "fan", base)
        self.assertEqual(len(candidates), 6)
        self.assertTrue(all(c["fan_aux_weight"] == 1.0 for c in candidates))

    def test_ddn_candidates_are_distinct_and_use_wavelet_branch(self):
        base = selected_base_config(self.bases, "ETTh1", "DLinear")
        candidates = search_candidates(self.protocol, "ETTh1", "ddn", base)
        keys = {(c["kernel_len"], c["j"]) for c in candidates}
        self.assertEqual(len(keys), 6)
        self.assertEqual({c["j"] for c in candidates}, {0, 1})

    def test_fan_screens_station_and_backbone_learning_rates(self):
        base = selected_base_config(self.bases, "ETTh1", "DLinear")
        lrs = {c["station_lr"] for c in search_candidates(self.protocol, "ETTh1", "fan", base)}
        self.assertEqual(lrs, {float(base["station_lr"]), float(base["learning_rate"])})
        same = selected_base_config(self.bases, "ETTh1", "iTransformer")  # both 1e-4
        lrs = {c["station_lr"] for c in search_candidates(self.protocol, "ETTh1", "fan", same)}
        self.assertEqual(len(lrs), 2)

    def test_hash_is_order_independent(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()

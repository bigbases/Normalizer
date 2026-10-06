import unittest

from experiments.run_matrix import (
    BASE_CONFIGS_PATH,
    PROTOCOL_PATH,
    canonical_hash,
    load_json,
    search_candidates,
    selected_base_config,
)


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_json(PROTOCOL_PATH)
        cls.bases = load_json(BASE_CONFIGS_PATH)

    def test_timemixerpp_uses_supplied_timemixer_seed_config(self):
        cfg = selected_base_config(self.bases, "ETTh1", "TimeMixerPP")
        self.assertEqual(cfg["backbone"], "TimeMixer")
        self.assertEqual(cfg["seq_len"], 720)

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

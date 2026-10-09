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

    def test_t3_backbones_use_official_lookback_96(self):
        # TimeMixer++ rows derive from the TimeMixer rows with look-back 96 and
        # channel mixing, and TimeXer uses look-back 96 (both official LTF settings).
        cfg = selected_base_config(self.bases, "ETTh1", "TimeMixerPP")
        self.assertEqual((cfg["backbone"], cfg["seq_len"], cfg["channel_independence"]), ("TimeMixerPP", 96, 0))
        tmix = selected_base_config(self.bases, "ETTh1", "TimeMixer")
        self.assertEqual((cfg["d_model"], cfg["e_layers"]), (tmix["d_model"], tmix["e_layers"]))
        self.assertEqual(selected_base_config(self.bases, "Traffic", "TimeXer")["seq_len"], 96)

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

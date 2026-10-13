import unittest

from experiments import efficiency as ef
from experiments.run_matrix import BASE_CONFIGS_PATH, PROTOCOL_PATH, load_json, selected_base_config


class EfficiencyArgsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_json(PROTOCOL_PATH)
        cls.bases = load_json(BASE_CONFIGS_PATH)

    def test_models_use_the_experiment_backbone_settings(self):
        for backbone in ef.BACKBONES:
            base = selected_base_config(self.bases, "Traffic", backbone)
            args, _ = ef.build_args(self.protocol, self.bases, "Traffic", backbone, "none", 720)
            self.assertEqual((args.seq_len, args.pred_len, args.enc_in), (int(base["seq_len"]), 720, 862))

    def test_normalizers_use_official_defaults(self):
        san, _ = ef.build_args(self.protocol, self.bases, "ETTm1", "DLinear", "san", 96)
        self.assertEqual(san.period_len, 12)                        # official SAN value for 15-minute data
        ddn, _ = ef.build_args(self.protocol, self.bases, "ETTh1", "DLinear", "ddn", 96)
        self.assertEqual((ddn.kernel_len, ddn.j, ddn.joint_lr), (7, 1, "backbone"))
        lt, _ = ef.build_args(self.protocol, self.bases, "ETTh1", "TimeXer", "lt", 96)
        self.assertEqual(lt.kernel_size, 25)

    def test_scaling_overrides_only_the_input_length(self):
        args, _ = ef.build_args(self.protocol, self.bases, "ETTh1", "DLinear", "lt", 720, seq_len=1440)
        self.assertEqual((args.seq_len, args.label_len, args.pred_len), (1440, 720, 720))


if __name__ == "__main__":
    unittest.main()

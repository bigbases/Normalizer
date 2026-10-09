import unittest
from argparse import Namespace

from experiments.cost_model import CostModel, order_cells
from experiments.gpu_scheduler import ResourceEstimator
from experiments.run_matrix import (
    BASE_CONFIGS_PATH, PROTOCOL_PATH, RESOURCE_PROFILES_PATH, build_cells, load_json,
)


def plan(experiment, **filters):
    args = Namespace(experiment=experiment, stage="auto", datasets=filters.get("datasets"),
                     backbones=filters.get("backbones"), methods=filters.get("methods"),
                     locks=None, shortlist=None)
    return build_cells(args, load_json(PROTOCOL_PATH), load_json(BASE_CONFIGS_PATH))


class OrderingTest(unittest.TestCase):
    def test_fastest_first_runs_whole_cases_in_increasing_duration(self):
        model = CostModel()
        ordered = order_cells(plan("1_rebuttal_completion"), model)
        cases = []
        for cell in ordered:
            case = (cell["dataset"], cell["backbone"])
            if not cases or cases[-1] != case:
                self.assertNotIn(case, cases, "a case must be contiguous in the queue")
                cases.append(case)
        totals = [sum(model.seconds(c) for c in ordered if (c["dataset"], c["backbone"]) == case)
                  for case in cases]
        self.assertEqual(totals, sorted(totals))
        self.assertEqual(cases[-1], ("Traffic", "iTransformer"))

    def test_measured_runtime_overrides_prior(self):
        model = CostModel()
        cell = plan("1_rebuttal_completion", datasets="ETTh1", backbones="DLinear")[0]
        model.measured[(cell["dataset"], cell["backbone"], cell["method"], cell["horizon"])] = 1234.0
        self.assertEqual(model.seconds(cell), 1234.0)

    def test_t3_memory_priors_keep_timexer_traffic_off_16gb_gpus(self):
        # Fitted to RTX A4000 probes at look-back 96 (TimeXer Traffic peaked at 13.8 GB).
        estimator = ResourceEstimator(load_json(RESOURCE_PROFILES_PATH), CostModel())
        traffic = plan("3_frozen_backbone_comparison", datasets="Traffic", backbones="TimeXer", methods="none")[0]
        weather = plan("3_frozen_backbone_comparison", datasets="Weather", backbones="TimeMixerPP", methods="none")[0]
        self.assertGreater(estimator.estimate(traffic).memory_mib + 5000, 16376)
        self.assertLess(estimator.estimate(weather).memory_mib, 8 * 1024)


if __name__ == "__main__":
    unittest.main()

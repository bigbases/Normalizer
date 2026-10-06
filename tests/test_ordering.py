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

    def test_profiled_memory_flags_timemixerpp_weather_as_too_large(self):
        estimator = ResourceEstimator(load_json(RESOURCE_PROFILES_PATH), CostModel())
        cell = plan("4_timemixerpp_generalization", datasets="Weather", methods="none")[0]
        self.assertGreater(estimator.estimate(cell).memory_mib, 80 * 1024)


if __name__ == "__main__":
    unittest.main()

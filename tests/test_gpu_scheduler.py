import unittest

from experiments.gpu_scheduler import (
    GPUState,
    ResourceEstimate,
    ResourceEstimator,
    select_gpu,
)
from experiments.run_matrix import RESOURCE_PROFILES_PATH, load_json


class GPUSchedulerTest(unittest.TestCase):
    def test_traffic_is_exclusive(self):
        profile = load_json(RESOURCE_PROFILES_PATH)
        estimator = ResourceEstimator(profile)
        cell = {
            "dataset": "Traffic",
            "backbone": "DLinear",
            "method": "lt",
            "horizon": 720,
            "dataset_meta": {"channels": 862},
            "base": {"seq_len": 336, "batch_size": 32},
        }
        self.assertTrue(estimator.estimate(cell).exclusive)

    def test_chooses_idle_gpu_when_others_are_busy(self):
        states = {
            0: GPUState(0, 24576, 4000, 88),
            1: GPUState(1, 24576, 6000, 75),
            2: GPUState(2, 24576, 8000, 70),
            3: GPUState(3, 24576, 23000, 2),
        }
        chosen = select_gpu(
            ResourceEstimate(6000, 45, False), states, {}, [0, 1, 2, 3],
            max_processes_per_gpu=4, max_gpu_util=92,
            min_free_memory_mib=1024,
        )
        self.assertEqual(chosen, 3)

    def test_exclusive_job_requires_empty_scheduler_slot(self):
        states = {0: GPUState(0, 24576, 23000, 2)}
        active = {0: [{"estimate": ResourceEstimate(1000, 10, False)}]}
        chosen = select_gpu(
            ResourceEstimate(6000, 70, True), states, active, [0],
            max_processes_per_gpu=4, max_gpu_util=92,
            min_free_memory_mib=1024,
        )
        self.assertIsNone(chosen)


if __name__ == "__main__":
    unittest.main()

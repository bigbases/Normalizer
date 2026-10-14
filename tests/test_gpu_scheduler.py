import unittest

from experiments.gpu_scheduler import (
    GPUState,
    ResourceEstimate,
    ResourceEstimator,
    select_gpu,
    select_gpu_measured,
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


class MeasuredAdmissionTest(unittest.TestCase):
    KW = dict(max_processes_per_gpu=10, min_processes_per_gpu=2, target_gpu_util=80,
              warmup_seconds=180, min_free_memory_mib=1024, max_gpu_temp=90)

    @staticmethod
    def jobs(n, started=0.0, memory=2000):
        return [{"estimate": ResourceEstimate(memory, 60, False), "started": started} for _ in range(n)]

    def select(self, active, util, now=1000.0, state=None, estimate=None, **kw):
        state = state or GPUState(0, 97887, 80000, int(util), 60)
        return select_gpu_measured(estimate or ResourceEstimate(6000, 60, False), {0: state},
                                   {0: active}, [0], {0: util}, now, **{**self.KW, **kw})

    def test_low_measured_util_admits_beyond_summed_estimates(self):
        # six cells whose guesses sum to 360% but the GPU measures 30%
        gpu, _ = self.select(self.jobs(6), util=30)
        self.assertEqual(gpu, 0)

    def test_high_measured_util_holds(self):
        gpu, reasons = self.select(self.jobs(3), util=85)
        self.assertIsNone(gpu)
        self.assertIn("util", reasons[0])

    def test_warm_up_holds_until_newest_cell_is_measured(self):
        gpu, reasons = self.select(self.jobs(2) + self.jobs(1, started=900.0), util=20, now=1000.0)
        self.assertIsNone(gpu)
        self.assertIn("warming", reasons[0])

    def test_minimum_cells_start_without_util_check(self):
        gpu, _ = self.select(self.jobs(1, started=999.0), util=95)
        self.assertEqual(gpu, 0)

    def test_cap_memory_and_temperature_still_apply(self):
        self.assertIsNone(self.select(self.jobs(10), util=10)[0])
        tight = GPUState(0, 24576, 3000, 10, 60)
        self.assertIsNone(self.select(self.jobs(2), util=10, state=tight)[0])
        hot = GPUState(0, 97887, 80000, 10, 92)
        gpu, reasons = self.select(self.jobs(2), util=10, state=hot)
        self.assertIsNone(gpu)
        self.assertIn("temperature", reasons[0])
        # the temperature limit applies beyond the minimum only
        self.assertEqual(self.select(self.jobs(1), util=10, state=hot)[0], 0)

    def test_thermal_slowdown_blocks_every_new_cell(self):
        throttling = GPUState(0, 97887, 80000, 10, 91, True)
        gpu, reasons = self.select([], util=10, state=throttling)
        self.assertIsNone(gpu)
        self.assertIn("thermal slowdown", reasons[0])
        # a slowdown seen earlier in the window still holds
        gpu, _ = self.select([], util=10, throttled={0: True})
        self.assertIsNone(gpu)


class HostReadingsTest(unittest.TestCase):
    def test_cpu_busy_load_and_memory(self):
        import tempfile
        from pathlib import Path
        from experiments.gpu_scheduler import HostMonitor
        with tempfile.TemporaryDirectory() as tmp:
            proc, cg = Path(tmp) / "proc", Path(tmp) / "cg"
            proc.mkdir()
            cg.mkdir()
            cpus = "".join(f"cpu{i} 1 0 1 1 0 0 0 0\n" for i in range(4))
            (proc / "loadavg").write_text("3.60 3.00 2.00 5/100 123\n")
            (proc / "meminfo").write_text("MemTotal: 33554432 kB\nMemAvailable: 8388608 kB\n")
            (cg / "memory.max").write_text("max\n")
            (proc / "stat").write_text("cpu  100 0 100 700 100 0 0 0\n" + cpus)
            host = HostMonitor(90, proc=proc, cgroup=cg)
            host.sample(0.0)
            # 200 more busy jiffies and 50 more idle ones: 80% busy
            (proc / "stat").write_text("cpu  250 0 150 750 100 0 0 0\n" + cpus)
            host.sample(10.0)
            self.assertEqual(host.cpus, 4)
            self.assertAlmostEqual(host.cpu_busy_percent(), 80.0)
            self.assertAlmostEqual(host.load_per_cpu(), 0.9)
            self.assertAlmostEqual(host.mem_available_gib(), 8.0)
            (cg / "memory.max").write_text(str(6 * 2**30))
            (cg / "memory.current").write_text(str(2 * 2**30))
            self.assertAlmostEqual(host.mem_available_gib(), 4.0)

    def test_util_tracker_averages_recent_samples(self):
        from experiments.gpu_scheduler import GpuUtilTracker
        tracker = GpuUtilTracker(60)
        tracker.add({0: GPUState(0, 1, 1, 100)}, 0.0)
        tracker.add({0: GPUState(0, 1, 1, 20)}, 50.0)
        tracker.add({0: GPUState(0, 1, 1, 40)}, 100.0)   # drops the sample at t=0
        self.assertAlmostEqual(tracker.mean(0), 30.0)
        self.assertIsNone(tracker.mean(1))
        self.assertFalse(tracker.throttled(0))
        tracker.add({0: GPUState(0, 1, 1, 40, 91, True)}, 110.0)
        self.assertTrue(tracker.throttled(0))


if __name__ == "__main__":
    unittest.main()

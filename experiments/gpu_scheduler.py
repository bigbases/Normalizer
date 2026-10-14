"""GPU admission control and Discord notifications for experiment workers."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResourceEstimate:
    memory_mib: int
    util_percent: int
    exclusive: bool


@dataclass(frozen=True)
class GPUState:
    index: int
    total_mib: int
    free_mib: int
    util_percent: int
    temp_c: int = -1                 # -1: not reported
    thermal_slowdown: bool = False   # hardware or software thermal clock slowdown is active


def load_dotenv(path):
    """Load a minimal KEY=VALUE .env without printing secret values."""
    path = Path(path)
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def send_discord(webhook_url, content, attempts=3):
    if not webhook_url:
        raise ValueError("DISCORD_WEBHOOK_URL is not configured")
    payload = json.dumps({"content": content[:1900]}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "LightNorm-Journal/1.0"},
        method="POST",
    )
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                if response.status not in (200, 204):
                    raise RuntimeError(f"Discord returned HTTP {response.status}")
                return
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Discord notification failed after {attempts} attempts: {last_error}")


def available_cpus():
    """CPUs granted to this container (cgroup quota), not the host's total."""
    candidates = []
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except AttributeError:
        candidates.append(os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            candidates.append(int(int(quota) / int(period)))
    except (OSError, ValueError):
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                candidates.append(int(quota / period))
        except (OSError, ValueError):
            pass
    return max(1, min(c for c in candidates if c > 0))


class ResourceEstimator:
    def __init__(self, profile, cost_model=None):
        self.profile = profile
        self.cost_model = cost_model

    def estimate(self, cell):
        prior = self._prior_estimate(cell)
        if self.cost_model is None:
            return prior
        measured_mib = self.cost_model.memory_mib(cell)
        if measured_mib is None:
            return prior
        # Profiled activations are more reliable than the coarse priors for
        # large configurations (Traffic iTransformer, TimeMixer++); keep the
        # larger of the two so that small cells are not over-packed either.
        memory = max(prior.memory_mib, int(measured_mib * float(self.profile.get("memory_headroom", 1.25))))
        util = max(10, min(95, int(round(12 + 83 * self.cost_model.gpu_fraction(cell)))))
        return ResourceEstimate(memory, util, prior.exclusive)

    def _prior_estimate(self, cell):
        backbone = self.profile["backbones"].get(
            cell["backbone"], self.profile["backbones"]["default"]
        )
        channels = int(cell["dataset_meta"]["channels"])
        seq_len = int(cell["base"].get("seq_len") or 96)
        horizon = int(cell["horizon"])
        batch_size = int(cell["base"].get("batch_size") or 32)
        activation_scale = (channels * seq_len * batch_size) / (7 * 96 * 32)
        horizon_scale = horizon / 96
        memory = (
            float(backbone["base_mib"])
            + float(backbone["channel_seq_mib"]) * activation_scale
            + float(backbone["horizon_mib"]) * horizon_scale
            + float(self.profile["normalizer_extra_mib"].get(cell["method"], 500))
        )
        memory *= float(self.profile["dataset_memory_multiplier"].get(cell["dataset"], 1.0))
        memory *= float(self.profile.get("memory_headroom", 1.25))
        util = int(backbone["util_percent"]) + int(
            self.profile["normalizer_extra_util"].get(cell["method"], 5)
        )
        exclusive = (
            bool(backbone.get("exclusive", False))
            or cell["dataset"] in self.profile.get("exclusive_datasets", [])
            or cell["backbone"] in self.profile.get("exclusive_backbones", [])
        )
        return ResourceEstimate(max(512, int(memory)), min(100, util), exclusive)


_THROTTLE_FIELDS = ",clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown"
_throttle_supported = True


def query_gpus():
    global _throttle_supported
    base = "index,memory.total,memory.free,utilization.gpu,temperature.gpu"

    def run(fields):
        return subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                              check=True, capture_output=True, text=True).stdout

    try:
        if _throttle_supported:
            try:
                output = run(base + _THROTTLE_FIELDS)
            except subprocess.CalledProcessError:      # older driver without these fields
                _throttle_supported = False
        if not _throttle_supported:
            output = run(base)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi is required for resource-aware execution") from exc
    states = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        index, total, free, util, temp = parts[:5]
        slowdown = any(p == "Active" for p in parts[5:])
        state = GPUState(int(index), int(total), int(free), int(util),
                         int(temp) if temp.isdigit() else -1, slowdown)
        states[state.index] = state
    if not states:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return states


def select_gpu(
    estimate,
    states,
    running_by_gpu,
    allowed_gpu_ids,
    max_processes_per_gpu,
    max_gpu_util,
    min_free_memory_mib,
    exclusive_idle_util=15,
    exclusive_free_ratio=0.80,
):
    """Choose the best currently admissible GPU, or return None."""
    candidates = []
    for gpu_id in allowed_gpu_ids:
        state = states[gpu_id]
        active = running_by_gpu.get(gpu_id, [])
        if len(active) >= max_processes_per_gpu:
            continue
        if any(job["estimate"].exclusive for job in active):
            continue
        if estimate.exclusive and active:
            continue

        reserved_memory = sum(job["estimate"].memory_mib for job in active)
        reserved_util = sum(job["estimate"].util_percent for job in active)
        # Observed use includes our children and unrelated users. Subtract the
        # conservative reservation to estimate external pressure without
        # double-counting jobs that have already allocated memory.
        observed_used = state.total_mib - state.free_mib
        external_memory = max(0, observed_used - reserved_memory)
        capacity_remaining = state.total_mib - external_memory - reserved_memory
        external_util = max(0, state.util_percent - min(100, reserved_util))
        projected_util = external_util + reserved_util + estimate.util_percent

        if capacity_remaining - estimate.memory_mib < min_free_memory_mib:
            continue
        if projected_util > max_gpu_util:
            continue
        if estimate.exclusive:
            if state.util_percent > exclusive_idle_util:
                continue
            if capacity_remaining / state.total_mib < exclusive_free_ratio:
                continue

        memory_after = capacity_remaining - estimate.memory_mib
        score = (memory_after / state.total_mib) + ((max_gpu_util - projected_util) / max_gpu_util)
        candidates.append((score, state.free_mib, -state.util_percent, gpu_id))

    return max(candidates)[-1] if candidates else None


class GpuUtilTracker:
    """Moving average of the utilization nvidia-smi reports for each GPU."""

    def __init__(self, window_seconds):
        self.window = float(window_seconds)
        self.samples = {}

    def add(self, states, now):
        for gpu_id, state in states.items():
            samples = self.samples.setdefault(gpu_id, deque())
            samples.append((now, state.util_percent, state.thermal_slowdown))
            while samples and now - samples[0][0] > self.window:
                samples.popleft()

    def mean(self, gpu_id):
        samples = self.samples.get(gpu_id)
        if not samples:
            return None
        return sum(util for _, util, _ in samples) / len(samples)

    def throttled(self, gpu_id):
        """True if a thermal slowdown was reported at any time in the window."""
        return any(slow for _, _, slow in self.samples.get(gpu_id, ()))


class HostMonitor:
    """CPU, load and memory of the whole machine.

    A container without lxcfs reads the host's /proc/stat, /proc/loadavg and
    /proc/meminfo, so several containers that share the host see the same
    pressure and back off together.
    """

    def __init__(self, window_seconds, proc=Path("/proc"), cgroup=Path("/sys/fs/cgroup")):
        self.window = float(window_seconds)
        self.proc = Path(proc)
        self.cgroup = Path(cgroup)
        self.samples = deque()
        try:
            lines = (self.proc / "stat").read_text().splitlines()
            self.cpus = max(1, sum(1 for line in lines if line.startswith("cpu") and line[3:4].isdigit()))
        except OSError:
            self.cpus = os.cpu_count() or 1

    def sample(self, now):
        try:
            fields = (self.proc / "stat").read_text().split("\n", 1)[0].split()[1:]
            values = [int(v) for v in fields[:8]]
        except (OSError, ValueError):
            return
        total, idle = sum(values), values[3] + values[4]          # idle + iowait
        self.samples.append((now, total, idle))
        while len(self.samples) > 2 and now - self.samples[1][0] >= self.window:
            self.samples.popleft()

    def cpu_busy_percent(self):
        if len(self.samples) < 2:
            return None
        (_, total0, idle0), (_, total1, idle1) = self.samples[0], self.samples[-1]
        if total1 <= total0:
            return None
        return 100.0 * (1.0 - (idle1 - idle0) / (total1 - total0))

    def load_per_cpu(self):
        try:
            return float((self.proc / "loadavg").read_text().split()[0]) / self.cpus
        except (OSError, ValueError, IndexError):
            return None

    def mem_available_gib(self):
        available = None
        try:
            for line in (self.proc / "meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / 2**20
                    break
        except (OSError, ValueError, IndexError):
            pass
        try:      # cgroup v2 limit of this container, if any
            limit = (self.cgroup / "memory.max").read_text().strip()
            if limit != "max":
                used = int((self.cgroup / "memory.current").read_text())
                left = (int(limit) - used) / 2**30
                available = left if available is None else min(available, left)
        except (OSError, ValueError):
            pass
        return available


def select_gpu_measured(
    estimate,
    states,
    running_by_gpu,
    allowed_gpu_ids,
    util_means,
    now,
    *,
    max_processes_per_gpu,
    min_processes_per_gpu,
    target_gpu_util,
    warmup_seconds,
    min_free_memory_mib,
    max_gpu_temp,
    throttled=None,
    exclusive_idle_util=15,
    exclusive_free_ratio=0.80,
):
    """Choose a GPU from measured utilization instead of summed per-cell guesses.

    Memory still uses the per-cell estimates (an OOM is costly), but one more cell
    starts only while the GPU's recent average utilization is below the target
    and every cell already on it has passed its warm-up, so each new process is
    measured before the next one is added.  Below ``min_processes_per_gpu`` the
    utilization, warm-up and temperature checks are skipped, but a GPU that
    reported a thermal slowdown (``throttled``) gets no new cell at all.
    Returns (gpu_id or None, reasons).
    """
    throttled = throttled or {}
    candidates, reasons = [], {}
    for gpu_id in allowed_gpu_ids:
        state = states[gpu_id]
        active = running_by_gpu.get(gpu_id, [])
        util = util_means.get(gpu_id)
        util = state.util_percent if util is None else util
        if len(active) >= max_processes_per_gpu:
            reasons[gpu_id] = f"{len(active)} cells (cap)"
            continue
        if any(job["estimate"].exclusive for job in active) or (estimate.exclusive and active):
            reasons[gpu_id] = "exclusive cell"
            continue
        reserved_memory = sum(job["estimate"].memory_mib for job in active)
        external_memory = max(0, state.total_mib - state.free_mib - reserved_memory)
        capacity_remaining = state.total_mib - external_memory - reserved_memory
        if capacity_remaining - estimate.memory_mib < min_free_memory_mib:
            reasons[gpu_id] = f"memory ({capacity_remaining} MiB left)"
            continue
        if throttled.get(gpu_id) or state.thermal_slowdown:
            reasons[gpu_id] = f"thermal slowdown ({state.temp_c} C)"
            continue
        if estimate.exclusive and (util > exclusive_idle_util
                                   or capacity_remaining / state.total_mib < exclusive_free_ratio):
            reasons[gpu_id] = "not idle for an exclusive cell"
            continue
        if len(active) >= min_processes_per_gpu:
            if any(now - job["started"] < warmup_seconds for job in active):
                reasons[gpu_id] = "newest cell still warming up"
                continue
            if util >= target_gpu_util:
                reasons[gpu_id] = f"GPU util {util:.0f}%"
                continue
            if max_gpu_temp and state.temp_c >= max_gpu_temp:
                reasons[gpu_id] = f"temperature {state.temp_c} C"
                continue
        memory_after = capacity_remaining - estimate.memory_mib
        score = (memory_after / state.total_mib) + (100 - util) / 100 - 0.05 * len(active)
        candidates.append((score, state.free_mib, gpu_id))
    return (max(candidates)[-1] if candidates else None), reasons


def merge_result_part(master_path, part_path):
    """Atomically-at-scheduler-level merge a single worker CSV into master."""
    master_path, part_path = Path(master_path), Path(part_path)
    if not part_path.exists():
        raise RuntimeError(f"worker completed without result file: {part_path}")
    with part_path.open(newline="") as f:
        part_rows = list(csv.DictReader(f))
    if len(part_rows) != 1:
        raise RuntimeError(f"expected exactly one result row in {part_path}, got {len(part_rows)}")
    row = part_rows[0]
    master_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    existing_header = None
    if master_path.exists():
        with master_path.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames
            existing_ids = {item.get("RunID", "") for item in reader}
    if row.get("RunID") in existing_ids:
        return row
    fieldnames = list(row.keys())
    if existing_header is not None and existing_header != fieldnames:
        raise RuntimeError("result CSV schema differs from worker result schema")
    with master_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if existing_header is None:
            writer.writeheader()
        writer.writerow(row)
    return row

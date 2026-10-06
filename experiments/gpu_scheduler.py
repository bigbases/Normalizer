"""GPU admission control and Discord notifications for experiment workers."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
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


def query_gpus():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi is required for resource-aware execution") from exc
    states = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, total, free, util = [part.strip() for part in line.split(",")]
        state = GPUState(int(index), int(total), int(free), int(util))
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

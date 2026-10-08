"""Runtime and memory estimates for experiment cells.

The estimates come from per-sample FLOPs and saved activations measured for
every selected dataset/backbone configuration (``configs/cost_profile.json``).
They are used to run the cells that finish first before longer ones, to admit
workers onto a GPU, and to print budget estimates.  They are priors only:
measured elapsed times in the run manifest replace them as runs finish.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs" / "cost_profile.json"


def split_sizes(meta, seq_len, horizon):
    """Number of (train, val, test) windows, mirroring data_provider/data_loader.py."""
    if meta["split"] == "ett_hour":
        num_train, num_val, num_test = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
    elif meta["split"] == "ett_minute":
        num_train, num_val, num_test = 12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4
    else:
        rows = meta["rows"]
        num_train = int(rows * 0.7)
        num_test = int(rows * 0.2)
        num_val = rows - num_train - num_test
    return (
        max(0, num_train - seq_len - horizon + 1),
        max(0, num_val - horizon + 1),
        max(0, num_test - horizon + 1),
    )


def _interp(lo, hi, horizon):
    """Linear interpolation of a profile quantity between horizons 96 and 720."""
    t = (horizon - 96) / (720 - 96)
    return lo + (hi - lo) * t


class CostModel:
    def __init__(self, doc=None):
        self.doc = doc if doc is not None else json.loads(PROFILE_PATH.read_text())
        self.measured = {}
        self.backbone_ratio = {}

    # ------------------------------------------------------------------ profile
    def profile(self, cell):
        key = f'{cell["dataset"]}|{cell["backbone"]}'
        lo = self.doc["profiles"].get(f"{key}|96")
        hi = self.doc["profiles"].get(f"{key}|720")
        if lo is None or hi is None:
            return None
        h = int(cell["horizon"])
        return {
            "gflop_per_sample": _interp(lo["gflop_per_sample"], hi["gflop_per_sample"], h),
            "act_mib_per_sample": _interp(lo["act_mib_per_sample"], hi["act_mib_per_sample"], h),
            "params_m": _interp(lo["params_m"], hi["params_m"], h),
            "extrapolated": bool(lo.get("extrapolated") or hi.get("extrapolated")),
        }

    def _device(self, field, backbone):
        table = self.doc["device"][field]
        return float(table.get(backbone, table["default"]))

    # ----------------------------------------------------------------- runtime
    def breakdown(self, cell):
        """Prior seconds for one cell plus the fraction spent in GPU compute."""
        prof = self.profile(cell)
        base = cell["base"]
        meta = self.doc["datasets"][cell["dataset"]]
        channels = int(cell["dataset_meta"]["channels"])
        seq_len, label_len = int(base["seq_len"]), int(base["label_len"])
        horizon, batch = int(cell["horizon"]), int(base["batch_size"])
        method, backbone = cell["method"], cell["backbone"]
        norms = self.doc["normalizers"]

        n_train, n_val, n_test = split_sizes(meta, seq_len, horizon)
        it_train = max(1, n_train // batch)          # drop_last=True
        it_val = max(1, -(-n_val // batch))
        it_test = max(1, -(-n_test // batch))

        flops = self._device("tflops_effective", backbone) * 1e12
        overhead = self._device("iter_overhead_s", backbone)
        bandwidth = float(self.doc["device"]["host_to_device_gbps"]) * 1e9
        gflop = prof["gflop_per_sample"] if prof else 1.0
        norm_gflop = norms["mflop_per_channel_sample"].get(method, 1.0) * channels / 1e3
        norm_overhead = norms["iter_overhead_s"].get(method, 0.001)

        data_s = batch * channels * (seq_len + label_len + horizon) * 8 * 1.5 / bandwidth
        compute_s = batch * (gflop + norm_gflop) * 1e9 / flops
        train_iter = overhead + norm_overhead + compute_s + data_s
        eval_iter = 0.5 * overhead + norm_overhead + compute_s / 3 + data_s
        station_iter = norm_overhead + 0.002 + batch * norm_gflop * 1e9 / flops + data_s

        epochs = float(self.doc.get("expected_backbone_epochs", 7))
        pre = int(cell.get("training", {}).get("pre_epoch", 5)) if norms["pretrain"].get(method) else 0
        seconds = (
            epochs * (it_train * train_iter + it_val * eval_iter)
            + pre * (it_train + it_val) * station_iter
        )
        tested = cell.get("stage") in ("final", "explore")
        if tested:
            seconds += it_test * eval_iter
        reads = 3 if tested else 2
        seconds += 8.0 + reads * meta["csv_mb"] / 40.0
        gpu_seconds = epochs * (it_train * compute_s + it_val * compute_s / 3)
        return seconds, min(1.0, gpu_seconds / max(seconds, 1e-9))

    def prior_seconds(self, cell):
        return self.breakdown(cell)[0]

    def seconds(self, cell):
        key = (cell["dataset"], cell["backbone"], cell["method"], int(cell["horizon"]))
        if key in self.measured:
            return self.measured[key]
        return self.prior_seconds(cell) * self.backbone_ratio.get(cell["backbone"], 1.0)

    def gpu_fraction(self, cell):
        return self.breakdown(cell)[1]

    # ------------------------------------------------------------------ memory
    def memory_mib(self, cell):
        prof = self.profile(cell)
        if prof is None:
            return None
        batch = int(cell["base"]["batch_size"])
        dev = self.doc["device"]
        weights = prof["params_m"] * 1e6 * 4 * 4 / 2**20   # params, grads, Adam m/v
        return int(dev["context_mib"] + weights + prof["act_mib_per_sample"] * batch * dev["activation_safety"])

    # ------------------------------------------------------------- calibration
    def calibrate_from_manifest(self, manifest_path, cells_by_run_id):
        """Use measured elapsed times of finished runs for later ordering."""
        path = Path(manifest_path)
        if not path.exists():
            return 0
        elapsed = defaultdict(list)
        ratios = defaultdict(list)
        with path.open() as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") != "finished" or record.get("returncode") != 0:
                    continue
                cell = cells_by_run_id.get(record.get("run_id"))
                if cell is None:
                    continue
                key = (cell["dataset"], cell["backbone"], cell["method"], int(cell["horizon"]))
                elapsed[key].append(float(record["elapsed_seconds"]))
                ratios[cell["backbone"]].append(float(record["elapsed_seconds"]) / self.prior_seconds(cell))
        self.measured = {key: median(values) for key, values in elapsed.items()}
        self.backbone_ratio = {key: median(values) for key, values in ratios.items()}
        return sum(len(v) for v in elapsed.values())

    def calibrate_from_rows(self, rows, gpu_name=None):
        """Use elapsed times recorded in packaged results (same GPU model only)."""
        elapsed = defaultdict(list)
        for row in rows:
            if gpu_name and row.get("GPU") != gpu_name:
                continue
            try:
                seconds = float(row["ElapsedSeconds"])
            except (KeyError, TypeError, ValueError):
                continue
            key = (row["Dataset"], row["Backbone"], row["UseNorm"], int(row["Horizon"]))
            elapsed[key].append(seconds)
        self.measured.update({key: median(values) for key, values in elapsed.items()})
        return sum(len(v) for v in elapsed.values())


def order_cells(cells, model, mode="fastest-first"):
    """Order cells so that whole dataset-backbone cases finish as early as possible.

    ``fastest-first`` sorts cases by their total estimated time and, inside a
    case, cells by their own estimate.  ``largest-first`` is the classic
    longest-processing-time order, which minimizes makespan but delivers the
    first complete case late.
    """
    case_total = defaultdict(float)
    for cell in cells:
        case_total[(cell["dataset"], cell["backbone"])] += model.seconds(cell)
    if mode == "largest-first":
        return sorted(cells, key=lambda c: -model.seconds(c))
    return sorted(
        cells,
        key=lambda c: (
            case_total[(c["dataset"], c["backbone"])],
            c["dataset"], c["backbone"],
            model.seconds(c), c["run_id"],
        ),
    )

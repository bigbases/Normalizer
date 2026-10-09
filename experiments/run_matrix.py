#!/usr/bin/env python3
"""Plan and execute reproducible journal experiment cells.

The runner hashes the complete effective configuration, writes a manifest, and
skips only exact run IDs already present in the result CSV.  Search/confirmation
runs are validation-only; final runs are the only ones allowed to evaluate the
test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

try:
    from .cost_model import CostModel, order_cells
    from .gpu_scheduler import (
        ResourceEstimate,
        ResourceEstimator,
        available_cpus,
        load_dotenv,
        merge_result_part,
        query_gpus,
        select_gpu,
        send_discord,
    )
except ImportError:  # direct: python experiments/run_matrix.py
    from cost_model import CostModel, order_cells
    from gpu_scheduler import (
        ResourceEstimate,
        ResourceEstimator,
        available_cpus,
        load_dotenv,
        merge_result_part,
        query_gpus,
        select_gpu,
        send_discord,
    )


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "protocol.json"
BASE_CONFIGS_PATH = ROOT / "configs" / "base_configs.json"
RESOURCE_PROFILES_PATH = ROOT / "configs" / "resource_profiles.json"
EXPLORE_PATH = ROOT / "configs" / "lt_explore.json"

BACKBONE_KEYS = (
    "seq_len", "label_len", "learning_rate", "batch_size", "d_model", "d_ff",
    "e_layers", "d_layers", "factor", "n_heads", "patch_len",
    "down_sampling_layers", "down_sampling_window", "down_sampling_method",
    "channel_independence",
)
LT_KEYS = (
    "station_lr", "s_norm", "use_mlp", "down_ratio", "kernel_len", "kernel_size",
)
BACKBONE_DEFAULTS = {
    "seq_len": 96,
    "label_len": 48,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "d_model": 512,
    "d_ff": 2048,
    "e_layers": 2,
    "d_layers": 1,
    "factor": 3,
    "n_heads": 8,
    "patch_len": 16,
    "down_sampling_layers": 3,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
    "channel_independence": False,
}
FIXED_TRAINING_ARGS = {
    "train_epochs": 10,
    "patience": 3,
    "pre_epoch": 5,
    "num_workers": 4,
}
METHOD_DEFAULTS = {
    "none": {},
    "revin": {"station_lr": 1e-4},
    "san": {"station_type": "adaptive"},
    "ddn": {"station_type": "adaptive", "twice_epoch": 1, "j": 0, "wavelet": "coif3"},
    "fan": {"fan_aux_weight": 1.0},
    "lt": {
        "affine": 1,
        "t_norm": 1,
        "t_ff": 64,
        "kernel_size": 25,
        "decomp_type": "sma",
    },
}


def canonical_hash(value, length=12):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def nonempty(value):
    return value not in (None, "")


def selected_base_config(base_document, dataset, backbone):
    # TimeMixer++ rows (T3) are derived from the TimeMixer rows; datasets
    # without one fall back to the TimeMixer entry.
    matches = [
        row for row in base_document["configs"]
        if row["dataset"] == dataset and row["backbone"] == backbone
    ]
    if not matches and backbone == "TimeMixerPP":
        matches = [
            row for row in base_document["configs"]
            if row["dataset"] == dataset and row["backbone"] == "TimeMixer"
        ]
    if len(matches) != 1 or matches[0].get("selection_status") != "selected":
        raise ValueError(f"No selected base config for {dataset}|{backbone}")
    return deepcopy(matches[0])


def resolve_backbone_config(base):
    resolved = deepcopy(base)
    for key, default in BACKBONE_DEFAULTS.items():
        if not nonempty(resolved.get(key)):
            resolved[key] = default
    return resolved


def complete_method_params(method, params):
    return {**METHOD_DEFAULTS[method], **params}


def normalized_params(method, raw, base):
    params = dict(raw)
    scale = params.pop("station_lr_scale", None)
    if scale is not None:
        center = float(base.get("station_lr") or 1e-4)
        params["station_lr"] = center * float(scale)
    if method in ("san", "ddn", "fan") and "station_lr" not in params:
        params["station_lr"] = float(base.get("station_lr") or 1e-4)
    if method == "fan":
        params["fan_aux_weight"] = 1.0
    return params


def supplied_lt_params(base):
    """LightNorm settings of the supplied launcher (Exp1/Exp3 'lt' cells)."""
    return {k: base[k] for k in LT_KEYS if nonempty(base.get(k))}


def station_lr_neighbors(value, ladder):
    """One step down/up the 1e-n / 5e-n ladder around the supplied station_lr."""
    index = min(range(len(ladder)), key=lambda i: abs(ladder[i] - float(value)))
    if not 0 < index < len(ladder) - 1:
        raise ValueError(f"station_lr {value} has no ladder neighbours in {ladder}")
    return ladder[index - 1], ladder[index], ladder[index + 1]


def lt_tuning_base(protocol, dataset, base):
    """Supplied LightNorm settings with the tuning-wide use_mlp rule applied.

    down_ratio stays at the supplied value and t_ff at its default; use_mlp is
    1 only on the large datasets named in lt_tuning.use_mlp_datasets.
    """
    params = supplied_lt_params(base)
    params["use_mlp"] = int(dataset in protocol["lt_tuning"]["use_mlp_datasets"])
    return params


def lt_pilot_candidates(protocol, dataset, base):
    """2^3 factorial (s_norm x kernel_size x station_lr) plus two centre points.

    Factor levels are the low/high ends of the 3-level sets the final 6-point
    grids draw from, so pilot cells are reused by the tuning search.
    """
    spec = protocol["lt_tuning"]["pilot"]
    fixed = lt_tuning_base(protocol, dataset, base)
    low, current, high = station_lr_neighbors(fixed.get("station_lr", 1e-4),
                                              protocol["lt_tuning"]["station_lr_ladder"])
    candidates = [
        {**fixed, "s_norm": s, "kernel_size": k, "station_lr": lr}
        for s in spec["s_norm"] for k in spec["kernel_size"] for lr in (low, high)
    ]
    candidates += [
        {**fixed, "s_norm": s, "kernel_size": spec["centre_kernel_size"], "station_lr": current}
        for s in spec["s_norm"]
    ]
    return candidates


def load_explore(path=None):
    path = Path(path or EXPLORE_PATH)
    return load_json(path) if path.exists() else {"cases": {}}


def explore_candidates(protocol, dataset, backbone, base, doc=None):
    """Exploratory LightNorm settings (configs/lt_explore.json) for one case.

    Each entry overrides the supplied LightNorm settings and evaluates the
    test split; horizons/seeds default to the screen horizons and seed 2021.
    These dev-case cells are kept out of the validation protocol (stage
    'explore', results/store/explore/).
    """
    doc = doc or load_explore()
    supplied = supplied_lt_params(base)
    schedule = []
    for entry in doc.get("cases", {}).get(f"{dataset}|{backbone}", []):
        schedule.append((
            {**supplied, **entry["params"]},
            entry.get("horizons", doc.get("default_horizons", protocol["screen_horizons"])),
            entry.get("seeds", doc.get("default_seeds", [2021])),
        ))
    return schedule


def lt_main_candidates(protocol, base):
    """The 6-point LightNorm grid: kernel_size {supplied, 49} x station_lr {1e-4, 5e-4, 1e-3}.

    Every other LightNorm setting (use_mlp, s_norm, down_ratio, t_ff, ...) stays
    as supplied, so the supplied setting is always one of the six.
    """
    spec = protocol["lt_tuning"]["main"]
    supplied = supplied_lt_params(base)
    own = supplied.get("kernel_size") or METHOD_DEFAULTS["lt"]["kernel_size"]
    kernels = [own] + [k for k in spec["kernel_size"] if k != "supplied" and k != own]
    return [{**supplied, "kernel_size": k, "station_lr": lr} for k in kernels for lr in spec["station_lr"]]


def is_supplied_lt(params, base):
    return complete_method_params("lt", params) == complete_method_params("lt", supplied_lt_params(base))


def search_candidates(protocol, dataset, method, base, lt_grid=None):
    if method == "lt":
        if lt_grid == "pilot":
            return lt_pilot_candidates(protocol, dataset, base)
        if lt_grid == "main":
            # The supplied setting's seed-2021 validation MSE is already in its
            # Exp1 final cells (same seed and training), so it is not re-run.
            # Backbones without Exp1 cells (TimeMixer++, TimeXer) search it too.
            grid = lt_main_candidates(protocol, base)
            if base.get("backbone") in protocol["lt_tuning"]["main"].get("supplied_from_exp1", []):
                grid = [c for c in grid if not is_supplied_lt(c, base)]
            return grid
        raise ValueError("LightNorm search needs an experiment with lt_grid 'pilot' or 'main'")
    if method == "fan":
        ks = protocol["datasets"][dataset]["fan_k"]
        if protocol.get("fan_lr_grid", "station_scale") == "station_and_backbone":
            # The FAN reference code trains the frequency predictor with the
            # backbone optimizer's learning rate; screen that setting next to
            # the shared normalizer station_lr.
            station = float(base.get("station_lr") or 1e-4)
            backbone = float(base.get("learning_rate") or 1e-4)
            lrs = [station, backbone] if backbone != station else [0.5 * station, station]
            raw = [{"freq_topk": k, "station_lr": lr} for k in ks for lr in lrs]
        else:
            raw = [
                {"freq_topk": k, "station_lr_scale": scale}
                for k in ks for scale in (0.5, 1.0)
            ]
    else:
        raw = protocol["search_grids"].get(method, [{}])
    return [normalized_params(method, item, base) for item in raw]


def locked_candidates(lock_doc, dataset, backbone, method):
    key = f"{dataset}|{backbone}|{method}"
    if key not in lock_doc.get("locks", {}):
        raise ValueError(
            f"Missing locked normalizer setting for {key}. Run search/confirm "
            "and experiments/select_hparams.py before final testing."
        )
    return [dict(lock_doc["locks"][key])]


def shortlist_candidates(shortlist_doc, dataset, backbone, method):
    key = f"{dataset}|{backbone}|{method}"
    if key not in shortlist_doc.get("shortlists", {}):
        raise ValueError(f"Missing confirmation shortlist for {key}")
    return [dict(item["params"]) for item in shortlist_doc["shortlists"][key]]


def completed_run_ids(result_path):
    path = Path(result_path)
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {row.get("RunID", "") for row in csv.DictReader(f) if row.get("RunID")}


def verified_legacy_hashes(ledger_path):
    """Read manually audited legacy cells.

    A guessed seed or a partially matching command is deliberately
    insufficient: both Verified=true and an exact effective ConfigHash are
    required to suppress a new run.
    """
    if not ledger_path:
        return set()
    with Path(ledger_path).open(newline="") as f:
        return {
            row["ConfigHash"] for row in csv.DictReader(f)
            if row.get("Verified", "").strip().lower() == "true" and row.get("ConfigHash")
        }


def parse_filter(value):
    return None if not value else {part.strip() for part in value.split(",") if part.strip()}


def append_manifest(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def run_checkpoint_dir(checkpoint_root, cell):
    """Run-scoped directory holding the backbone and normalizer checkpoints."""
    return Path(checkpoint_root) / cell["run_id"]


def cell_command(cell, data_root, result_file, checkpoint_root, gpu=0):
    dataset_meta = cell["dataset_meta"]
    relative = Path(dataset_meta["relative_path"])
    base = cell["base"]
    effective = cell["effective"]

    cmd = [
        sys.executable, str(ROOT / "run_longExp.py"),
        "--is_training", "1",
        "--itr", "1",
        "--model_id", f'{cell["dataset"]}_{cell["backbone"]}_{cell["horizon"]}',
        "--model", cell["backbone"],
        "--data", cell["dataset"],
        "--root_path", str(Path(data_root) / relative.parent),
        "--data_path", relative.name,
        "--features", "M",
        "--freq", dataset_meta["freq"],
        "--enc_in", str(dataset_meta["channels"]),
        "--dec_in", str(dataset_meta["channels"]),
        "--c_out", str(dataset_meta["channels"]),
        "--pred_len", str(cell["horizon"]),
        "--use_norm", cell["method"],
        "--seed", str(cell["seed"]),
        "--deterministic", "true",
        "--phase", cell["stage"],
        "--candidate_id", cell["candidate_id"],
        "--config_hash", cell["config_hash"],
        "--run_id", cell["run_id"],
        "--result_file", str(result_file),
        "--checkpoints", str(run_checkpoint_dir(checkpoint_root, cell)),
        "--station_root", str(run_checkpoint_dir(checkpoint_root, cell)),
        "--gpu", str(gpu),
        "--des", "journal-v1",
    ]

    for key in BACKBONE_KEYS:
        if nonempty(base.get(key)):
            cmd.extend([f"--{key}", str(base[key])])

    # Declared protocol defaults are explicit in both command and config hash.
    for key, value in {**FIXED_TRAINING_ARGS, **effective}.items():
        if nonempty(value):
            cmd.extend([f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)])

    if cell["backbone"] == "TimeMixerPP":
        cmd.extend([
            "--top_k", "5", "--n_kernels", "6",
            "--channel_mixing", "true",
            "--tmpp_use_internal_norm", "false",
        ])

    if cell["stage"] in ("search", "confirm"):
        cmd.append("--skip_test")
    return cmd


def recover_result_parts(master_path, parts_dir):
    """Merge complete one-row worker outputs left by an interrupted scheduler."""
    parts_dir = Path(parts_dir)
    if not parts_dir.exists():
        return 0
    recovered = 0
    for part in sorted(parts_dir.glob("*.success.csv")):
        try:
            merge_result_part(master_path, part)
            recovered += 1
        except RuntimeError:
            # An empty/partial file belongs to a failed or interrupted worker
            # and will be replaced by a distinct attempt file on retry.
            continue
    return recovered


def parse_gpu_ids(args, states):
    available = sorted(states)
    if args.gpus and args.gpu_count:
        raise ValueError("Use either --gpus or --gpu-count, not both")
    if args.gpus:
        selected = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    elif args.gpu_count:
        if args.gpu_count < 1 or args.gpu_count > len(available):
            raise ValueError(f"--gpu-count must be within 1..{len(available)}")
        selected = available[:args.gpu_count]
    else:
        selected = [available[0]]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Requested GPU indices not reported by nvidia-smi: {unknown}")
    return selected


def tail_text(path, max_bytes=12000):
    path = Path(path)
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.read().decode("utf-8", errors="replace")


def worker_threads(args, n_gpus):
    if args.cpu_threads_per_worker:
        return args.cpu_threads_per_worker
    return max(1, available_cpus() // max(1, n_gpus * args.max_processes_per_gpu))


def keep_checkpoints(args, cell):
    return args.keep_checkpoints == "all" or (
        args.keep_checkpoints == "final" and cell["stage"] == "final"
    )


def execute_parallel(args, pending, profile, cost_model):
    initial_states = query_gpus()
    allowed_gpu_ids = parse_gpu_ids(args, initial_states)
    estimator = ResourceEstimator(profile, cost_model)
    max_total = max(initial_states[gpu].total_mib for gpu in allowed_gpu_ids)
    threads = worker_threads(args, len(allowed_gpu_ids))

    queue = []
    for cell in order_cells(pending, cost_model, args.order):
        estimate = estimator.estimate(cell)
        if estimate.memory_mib + args.min_free_memory_mib > max_total:
            raise RuntimeError(
                f'{cell["run_id"]} estimates {estimate.memory_mib} MiB plus reserve, '
                f"but selected GPUs have at most {max_total} MiB"
            )
        queue.append({"cell": cell, "estimate": estimate, "retries": 0})
    print(
        f"order={args.order} gpus={allowed_gpu_ids} threads_per_worker={threads} "
        f"estimated_cell_hours={sum(cost_model.seconds(i['cell']) for i in queue) / 3600:.1f}"
    )

    parts_dir = Path(args.result_parts or (Path(args.results).parent / "result_parts"))
    logs_dir = Path(args.logs or (Path(args.results).parent / "logs"))
    parts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(args.env_file)
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if args.notify_every_cases > 0 and not webhook:
        raise ValueError(
            f"DISCORD_WEBHOOK_URL is missing. Put it in {args.env_file}, or explicitly "
            "set --notify-every-cases 0 to disable notifications."
        )

    running = {}
    fatal_error = None
    case_remaining = Counter((item["cell"]["dataset"], item["cell"]["backbone"]) for item in queue)
    total_cases = len(case_remaining)
    completed_cases = 0
    completed_cells = 0
    notification_batch = []
    started_at = time.monotonic()
    last_wait_message = 0.0

    def notify(message, event):
        if args.notify_every_cases <= 0:
            return
        try:
            send_discord(webhook, message)
            append_manifest(args.manifest, {
                "event": "discord_sent", "at": datetime.now(timezone.utc).isoformat(),
                "notification_event": event,
            })
        except Exception as exc:
            append_manifest(args.manifest, {
                "event": "discord_failed", "at": datetime.now(timezone.utc).isoformat(),
                "notification_event": event, "error": str(exc),
            })
            print(f"WARNING: {exc}", file=sys.stderr)

    while queue or running:
        states = query_gpus()
        running_by_gpu = defaultdict(list)
        for job in running.values():
            running_by_gpu[job["gpu_id"]].append(job)

        launched = False
        if fatal_error is None:
            for item in list(queue):
                gpu_id = select_gpu(
                    item["estimate"], states, running_by_gpu, allowed_gpu_ids,
                    args.max_processes_per_gpu, args.max_gpu_util,
                    args.min_free_memory_mib,
                )
                if gpu_id is None:
                    continue
                cell = item["cell"]
                attempt = item["retries"] + 1
                part_path = parts_dir / f'{cell["run_id"]}.attempt{attempt}.csv'
                log_path = logs_dir / f'{cell["run_id"]}.attempt{attempt}.log'
                if part_path.exists():
                    orphan = part_path.with_suffix(
                        f'.orphaned-{int(time.time())}'
                    )
                    part_path.replace(orphan)
                cmd = cell_command(cell, args.data_root, part_path, args.checkpoints, gpu=0)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["PYTHONUNBUFFERED"] = "1"
                for var in ("LIGHTNORM_TORCH_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                    env[var] = str(threads)
                log_handle = log_path.open("w")
                process = subprocess.Popen(
                    cmd, cwd=ROOT, env=env, stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                job = {
                    **item, "process": process, "gpu_id": gpu_id,
                    "part_path": part_path, "log_path": log_path,
                    "log_handle": log_handle, "started_monotonic": time.monotonic(),
                }
                running[process.pid] = job
                running_by_gpu[gpu_id].append(job)
                queue.remove(item)
                launched = True
                append_manifest(args.manifest, {
                    "event": "started", "at": datetime.now(timezone.utc).isoformat(),
                    "run_id": cell["run_id"], "config_hash": cell["config_hash"],
                    "physical_gpu": gpu_id, "logical_gpu": 0,
                    "estimated_memory_mib": item["estimate"].memory_mib,
                    "estimated_util_percent": item["estimate"].util_percent,
                    "exclusive": item["estimate"].exclusive,
                    "attempt": attempt, "command": cmd, "log": str(log_path),
                })

        finished_any = False
        for pid, job in list(running.items()):
            returncode = job["process"].poll()
            if returncode is None:
                continue
            finished_any = True
            job["log_handle"].close()
            del running[pid]
            cell = job["cell"]
            elapsed = time.monotonic() - job["started_monotonic"]
            append_manifest(args.manifest, {
                "event": "finished", "at": datetime.now(timezone.utc).isoformat(),
                "run_id": cell["run_id"], "config_hash": cell["config_hash"],
                "physical_gpu": job["gpu_id"], "returncode": returncode,
                "elapsed_seconds": round(elapsed, 3), "attempt": job["retries"] + 1,
            })
            if returncode == 0:
                success_part = job["part_path"].with_suffix(".success.csv")
                job["part_path"].replace(success_part)
                merge_result_part(args.results, success_part)
                if not keep_checkpoints(args, cell):
                    shutil.rmtree(run_checkpoint_dir(args.checkpoints, cell), ignore_errors=True)
                completed_cells += 1
                case_key = (cell["dataset"], cell["backbone"])
                case_remaining[case_key] -= 1
                if case_remaining[case_key] == 0:
                    completed_cases += 1
                    notification_batch.append(f"{case_key[0]} × {case_key[1]}")
                    if (
                        args.notify_every_cases > 0
                        and len(notification_batch) >= args.notify_every_cases
                    ):
                        notify(
                            f"✅ LightNorm {args.experiment}: {completed_cases}/{total_cases} cases complete\n"
                            + "\n".join(f"• {name}" for name in notification_batch),
                            "case_batch",
                        )
                        notification_batch.clear()
            else:
                if job["part_path"].exists():
                    job["part_path"].replace(job["part_path"].with_suffix(".failed"))
                log_tail = tail_text(job["log_path"])
                is_oom = "out of memory" in log_tail.lower()
                if is_oom and job["retries"] < args.max_retries:
                    retry_estimate = ResourceEstimate(
                        memory_mib=int(job["estimate"].memory_mib * 1.5),
                        util_percent=max(job["estimate"].util_percent, 80),
                        exclusive=True,
                    )
                    if retry_estimate.memory_mib + args.min_free_memory_mib > max_total:
                        fatal_error = RuntimeError(
                            f'OOM retry for {cell["run_id"]} requires an estimated '
                            f'{retry_estimate.memory_mib} MiB, larger than selected GPUs'
                        )
                    else:
                        queue.insert(0, {
                            "cell": cell, "estimate": retry_estimate,
                            "retries": job["retries"] + 1,
                        })
                        append_manifest(args.manifest, {
                            "event": "retry_oom_exclusive",
                            "at": datetime.now(timezone.utc).isoformat(),
                            "run_id": cell["run_id"],
                            "next_estimated_memory_mib": retry_estimate.memory_mib,
                        })
                else:
                    fatal_error = RuntimeError(
                        f'worker failed for {cell["run_id"]}; see {job["log_path"]}'
                    )
                    notify(
                        f"❌ LightNorm {args.experiment} failed\n"
                        f"Run: {cell['run_id']}\nGPU: {job['gpu_id']}\nLog: {job['log_path']}",
                        "failure",
                    )

        if fatal_error is not None and not running:
            break
        if queue and not running and not launched:
            now = time.monotonic()
            if now - last_wait_message >= 60:
                print("Waiting for a GPU with sufficient free memory/utilization...")
                last_wait_message = now
            if args.scheduler_timeout_minutes > 0 and (
                now - started_at > args.scheduler_timeout_minutes * 60
            ):
                fatal_error = TimeoutError("GPU scheduler admission timeout")
                break
        if queue or running:
            time.sleep(args.scheduler_poll_seconds if not finished_any else min(1, args.scheduler_poll_seconds))

    if notification_batch and fatal_error is None:
        notify(
            f"✅ LightNorm {args.experiment}: {completed_cases}/{total_cases} cases complete\n"
            + "\n".join(f"• {name}" for name in notification_batch),
            "final_case_batch",
        )
    if fatal_error is not None:
        raise fatal_error
    print(
        f"completed_cells={completed_cells} completed_cases={completed_cases}/{total_cases} "
        f"elapsed_seconds={time.monotonic() - started_at:.1f}"
    )


def print_summary(pending, cost_model, estimator, order):
    """Per-case plan in execution order with prior time and peak memory."""
    cases = defaultdict(list)
    for cell in order_cells(pending, cost_model, order):
        cases[(cell["dataset"], cell["backbone"])].append(cell)
    print(f'{"case":28s} {"cells":>5s} {"est_h":>7s} {"max_cell_min":>12s} {"peak_GiB":>8s}  notes')
    total = 0.0
    for (dataset, backbone), items in cases.items():
        seconds = [cost_model.seconds(c) for c in items]
        peak = max(estimator.estimate(c).memory_mib for c in items) / 1024
        notes = []
        if any((cost_model.profile(c) or {}).get("extrapolated") for c in items):
            notes.append("extrapolated profile")
        if peak > 79:
            notes.append("EXCEEDS 80GB GPU")
        total += sum(seconds)
        print(f"{dataset + ' x ' + backbone:28s} {len(items):5d} {sum(seconds) / 3600:7.2f} "
              f"{max(seconds) / 60:12.1f} {peak:8.1f}  {', '.join(notes)}")
    print(f'{"TOTAL (serial cell-hours)":28s} {len(pending):5d} {total / 3600:7.1f}')


def build_cells(args, protocol, base_document, lock_doc=None, shortlist_doc=None):
    exp = protocol["experiments"][args.experiment]
    stage = exp["phase"] if args.stage == "auto" else args.stage
    if args.experiment == "2_normalizer_search" and stage == "final":
        raise ValueError("Experiment 2 supports search or confirm, not final")

    dataset_filter = parse_filter(args.datasets)
    backbone_filter = parse_filter(args.backbones)
    method_filter = parse_filter(args.methods)
    datasets = [d for d in protocol["datasets"] if not dataset_filter or d in dataset_filter]
    backbones = [b for b in exp["backbones"] if not backbone_filter or b in backbone_filter]
    methods = [m for m in exp["methods"] if not method_filter or m in method_filter]

    # Documents can be passed in-memory (fill_missing.py derives them from the
    # packaged results) or loaded from --locks/--shortlist files.
    if lock_doc is None:
        lock_doc = load_json(args.locks) if args.locks else {}
    if shortlist_doc is None:
        shortlist_doc = load_json(args.shortlist) if args.shortlist else {}
    if stage == "confirm" and not shortlist_doc:
        raise ValueError("--shortlist is required for confirm stage")
    needs_lock = [m for m in methods if m not in ("none", "lt") or exp.get("lt_params") == "locked"]
    if stage == "final" and needs_lock and not lock_doc:
        raise ValueError("--locks is required for final comparisons with tuned baselines")

    horizons = protocol["screen_horizons"] if stage in ("search", "confirm") else protocol["horizons"]
    seeds = [2021] if stage == "search" else ([2022] if stage == "confirm" else protocol["seeds"])
    cells = []
    for dataset in datasets:
        for backbone in backbones:
            selected_base = selected_base_config(base_document, dataset, backbone)
            base = resolve_backbone_config(selected_base)
            for method in methods:
                if stage == "explore":
                    candidates = None
                elif stage == "search":
                    candidates = search_candidates(protocol, dataset, method, selected_base,
                                                   lt_grid=exp.get("lt_grid"))
                elif stage == "confirm":
                    candidates = shortlist_candidates(shortlist_doc, dataset, backbone, method)
                elif method == "lt" and exp.get("lt_params") == "locked":
                    candidates = locked_candidates(lock_doc, dataset, backbone, method)
                elif method == "lt":
                    candidates = [supplied_lt_params(selected_base)]
                elif method == "none":
                    candidates = [{}]
                else:
                    candidates = locked_candidates(lock_doc, dataset, backbone, method)
                schedule = (explore_candidates(protocol, dataset, backbone, selected_base)
                            if stage == "explore" else [(p, horizons, seeds) for p in candidates])

                for params, cell_horizons, cell_seeds in schedule:
                    params = complete_method_params(method, params)
                    candidate_id = canonical_hash({"method": method, "params": params})
                    for horizon in cell_horizons:
                        for seed in cell_seeds:
                            identity = {
                                "protocol": protocol["protocol_version"],
                                "stage": stage,
                                "dataset": dataset,
                                "backbone": backbone,
                                "method": method,
                                "horizon": horizon,
                                "seed": seed,
                                "base": {k: base.get(k) for k in BACKBONE_KEYS},
                                "normalizer": params,
                                "training": FIXED_TRAINING_ARGS,
                                "data": protocol["datasets"][dataset],
                                "timemixerpp": (
                                    {"top_k": 5, "n_kernels": 6, "channel_mixing": True, "internal_norm": False}
                                    if backbone == "TimeMixerPP" else None
                                ),
                            }
                            config_hash = canonical_hash(identity, length=16)
                            run_id = (
                                f'{stage}-{dataset}-{backbone}-{method}-h{horizon}-s{seed}'
                                f'-c{candidate_id}-x{config_hash[:8]}'
                            )
                            cells.append({
                                **identity,
                                "candidate_id": candidate_id,
                                "config_hash": config_hash,
                                "run_id": run_id,
                                "dataset_meta": protocol["datasets"][dataset],
                                "effective": params,
                                "base": base,
                            })
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=[
        "1_rebuttal_completion", "2_normalizer_search",
        "3_frozen_backbone_comparison", "4_timemixerpp_generalization",
    ])
    parser.add_argument("--stage", choices=["auto", "search", "confirm", "final"], default="auto")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results", default=str(ROOT / "results" / "journal_results.csv"))
    parser.add_argument("--manifest", default=str(ROOT / "results" / "run_manifest.jsonl"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints_journal"))
    parser.add_argument("--locks", help="locked normalizer JSON produced after search/confirmation")
    parser.add_argument("--shortlist", help="top-2 JSON produced after screening")
    parser.add_argument("--completed-ledger",
                        help="audited legacy-cell CSV; only Verified=true exact hashes are skipped")
    parser.add_argument("--datasets", help="comma-separated subset")
    parser.add_argument("--backbones", help="comma-separated subset")
    parser.add_argument("--methods", help="comma-separated subset")
    parser.add_argument("--gpus",
                        help="comma-separated physical GPU indices, e.g. 0,2,3")
    parser.add_argument("--gpu-count", type=int,
                        help="use the first N GPUs reported by nvidia-smi")
    parser.add_argument("--max-processes-per-gpu", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--max-gpu-util", type=int, default=92,
                        help="maximum projected GPU utilization percentage")
    parser.add_argument("--min-free-memory-mib", type=int, default=1024,
                        help="memory reserve left after admitting a worker")
    parser.add_argument("--resource-profiles", default=str(RESOURCE_PROFILES_PATH))
    parser.add_argument("--scheduler-poll-seconds", type=float, default=5.0)
    parser.add_argument("--scheduler-timeout-minutes", type=float, default=0,
                        help="0 waits indefinitely for resources")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="OOM retries; retry is forced onto an otherwise idle GPU")
    parser.add_argument("--result-parts",
                        help="per-worker CSV directory; defaults beside --results")
    parser.add_argument("--logs", help="per-worker log directory; defaults beside --results")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--notify-every-cases", type=int, default=1,
                        help="Discord interval in completed dataset-backbone cases; 0 disables")
    parser.add_argument("--limit", type=int, help="run/print at most N cells (smoke testing)")
    parser.add_argument("--order", choices=["fastest-first", "largest-first"], default="fastest-first",
                        help="fastest-first finishes whole dataset-backbone cases in order of "
                             "estimated duration; largest-first minimizes total makespan")
    parser.add_argument("--keep-checkpoints", choices=["final", "all", "none"], default="final",
                        help="delete run-scoped checkpoints of finished runs outside this set")
    parser.add_argument("--cpu-threads-per-worker", type=int, default=0,
                        help="torch/OMP threads per worker; 0 divides the container CPUs")
    parser.add_argument("--summary", action="store_true",
                        help="dry run: print per-case cell counts, time and memory estimates")
    parser.add_argument("--execute", action="store_true", help="execute; default is a dry-run plan")
    args = parser.parse_args()
    if not (1 <= args.max_gpu_util <= 100):
        parser.error("--max-gpu-util must be within 1..100")
    if args.min_free_memory_mib < 0:
        parser.error("--min-free-memory-mib must be non-negative")
    if args.scheduler_poll_seconds <= 0:
        parser.error("--scheduler-poll-seconds must be positive")
    if args.notify_every_cases < 0:
        parser.error("--notify-every-cases must be non-negative")

    protocol = load_json(PROTOCOL_PATH)
    base_document = load_json(BASE_CONFIGS_PATH)
    cells = build_cells(args, protocol, base_document)
    cost_model = CostModel()
    calibrated = cost_model.calibrate_from_manifest(args.manifest, {c["run_id"]: c for c in cells})
    if calibrated:
        print(f"runtime_estimates_calibrated_from_finished_runs={calibrated}")
    if args.limit is not None:
        cells = order_cells(cells, cost_model, args.order)[:args.limit]

    parts_dir = Path(args.result_parts or (Path(args.results).parent / "result_parts"))
    if args.execute:
        recovered = recover_result_parts(args.results, parts_dir)
        if recovered:
            print(f"recovered_result_parts={recovered}")
    done = completed_run_ids(args.results)
    legacy_done = verified_legacy_hashes(args.completed_ledger)
    pending = [
        cell for cell in cells
        if cell["run_id"] not in done and cell["config_hash"] not in legacy_done
    ]
    print(f"planned={len(cells)} completed_exact={len(cells) - len(pending)} pending={len(pending)}")
    if not pending:
        return

    profile = load_json(args.resource_profiles)
    estimator = ResourceEstimator(profile, cost_model)
    if args.summary and not args.execute:
        print_summary(pending, cost_model, estimator, args.order)
        return
    if not args.execute:
        for cell in order_cells(pending, cost_model, args.order):
            estimate = estimator.estimate(cell)
            preview_result = parts_dir / f'{cell["run_id"]}.attempt1.csv'
            cmd = cell_command(cell, args.data_root, preview_result, args.checkpoints, gpu=0)
            print(
                f'# estimated_memory_mib={estimate.memory_mib} '
                f'estimated_util={estimate.util_percent}% exclusive={estimate.exclusive}'
            )
            print(f"CUDA_VISIBLE_DEVICES=<scheduler> {shlex.join(cmd)}")
        return

    execute_parallel(args, pending, profile, cost_model)


if __name__ == "__main__":
    main()

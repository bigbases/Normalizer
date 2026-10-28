"""Efficiency benchmark: the cost each normalizer adds to each backbone.

Measures, for every dataset x backbone x method x horizon, on one GPU with
nothing else running on it:

  * parameters of the backbone and of the normalization module
  * FLOPs of the module and of the whole forecast at batch 1 (torch
    FlopCounterMode: matrix multiplications and convolutions only;
    element-wise ops, pooling and reductions are not counted)
  * inference latency (mean / p50 / p95 over timed iterations), throughput
    and peak memory at batch 1 and at a common maximum batch, end-to-end and
    for the module alone (normalize + de_normalize)
  * training step time and peak memory at the experiment batch size, for
    the normalizer pretraining step and for the backbone (joint) step
  * module FLOPs over input lengths (scalability), backbone independent

Models are built exactly as in the experiments (run_matrix cell ->
run_longExp arguments -> Exp_Main); inputs are random tensors of the same
shapes, so no data is read. Normalizers use fixed representative settings:
the official defaults of SAN (period_len per dataset) and DDN (kernel_len 7,
j 1), RevIN with affine, and LightNorm's supplied setting, or with
--lt-settings locked its locked setting of the main results (kernel_size,
station_lr, s_norm and use_mlp from results/summary/exp7_lt_tuning.csv).

Run it only when nothing else uses the GPU (stop the worker first):
  python experiments/efficiency.py --out results/efficiency/itm21.csv
  python experiments/efficiency.py --quick --datasets ETTh1 --out /tmp/e.csv   # smoke test
  python experiments/efficiency.py --summary --out results/efficiency/itm21.csv  # overhead table
"""
import argparse
import csv
import gc
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_longExp  # noqa: E402  (argument parser of the experiments)
import normalizers.LightNorm as lightnorm  # noqa: E402
from exp.exp_main import Exp_Main  # noqa: E402
from experiments.run_matrix import (  # noqa: E402
    BASE_CONFIGS_PATH, PROTOCOL_PATH, cell_command, load_json, method_params, normalized_params,
    resolve_backbone_config, selected_base_config, supplied_lt_params, timefilter_args,
)
from utils.timefeatures import time_features_from_frequency_str  # noqa: E402

DATASETS = ("ETTh1", "Traffic")
BACKBONES = ("DLinear", "iTransformer", "TimeXer", "TimeFilter")
METHODS = ("none", "revin", "san", "ddn", "lt")
HORIZONS = (96, 720)
SCALE_LENGTHS = (96, 336, 720, 1440)
# Official SAN scripts fix period_len per dataset (24 hourly, 12 for 10/15-minute data).
SAN_PERIOD = {"ETTh1": 24, "ETTh2": 24, "Electricity": 24, "Traffic": 24, "ETTm1": 12, "ETTm2": 12, "Weather": 12,
              "Exchange": 6}
# FAN's number of selected frequencies: the official K of the FAN paper (Table 1), first value of protocol "fan_k".
FAN_TOPK = {"ETTh1": 4, "ETTh2": 3, "ETTm1": 11, "ETTm2": 5, "Weather": 2, "Electricity": 3, "Traffic": 30, "Exchange": 2}
MAX_BATCH = 1024
LT_LOCKS_CSV = ROOT / "results" / "summary" / "exp7_lt_tuning.csv"


def log(msg):
    print(f"[{datetime.now():%F %T}] {msg}", flush=True)


# ----------------------------------------------------------------- set-up
def load_lt_locks(path=LT_LOCKS_CSV):
    """{(dataset, backbone): locked LightNorm overrides} from the Exp7 tuning summary."""
    with Path(path).open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["Locked"] == "True"]
    return {(r["Dataset"], r["Backbone"]): {"kernel_size": int(r["kernel_size"]), "station_lr": float(r["station_lr"]),
                                            "s_norm": int(r["s_norm"]), "use_mlp": int(r["use_mlp"])} for r in rows}


def method_settings(dataset, method, selected_base, lt_lock=None):
    """Representative normalizer settings (not tuned per case); LightNorm optionally locked."""
    raw = {
        "none": {},
        "revin": {"affine": 1},
        "san": {"period_len": SAN_PERIOD[dataset]},
        "ddn": {"kernel_len": 7, "j": 1, "hkernel_len": 5, "twice_epoch": 1},
        "fan": {"freq_topk": FAN_TOPK[dataset]},
        "dishts": {"dish_init": "standard"},
        "lt": {**supplied_lt_params(selected_base), **(lt_lock or {})},
    }[method]
    if method in ("revin", "san", "ddn", "fan", "dishts"):
        raw = normalized_params(method, raw, selected_base)
    return method_params(method, raw)


def build_args(protocol, bases, dataset, backbone, method, horizon, seq_len=None, lt_locks=None):
    """run_longExp arguments of the experiment cell, optionally at another input length."""
    selected = selected_base_config(bases, dataset, backbone)
    base = resolve_backbone_config(selected)
    if seq_len is not None:
        base = {**base, "seq_len": seq_len, "label_len": seq_len // 2}
    lt_lock = lt_locks[(dataset, backbone)] if lt_locks is not None and method == "lt" else None
    params = method_settings(dataset, method, selected, lt_lock)
    cell = {"dataset": dataset, "backbone": backbone, "method": method, "horizon": horizon, "seed": 2021,
            "stage": "final", "candidate_id": "bench", "config_hash": "bench", "run_id": "bench",
            "dataset_meta": protocol["datasets"][dataset], "effective": params, "base": base}
    if backbone == "TimeFilter":
        cell["timefilter"] = timefilter_args(base, horizon)
    argv = cell_command(cell, "/nonexistent", "/dev/null", "/tmp/lightnorm-bench")[2:]
    args = run_longExp.parser.parse_args(argv)
    if args.force_label_len_half:
        args.label_len = args.seq_len // 2
    args.use_gpu = torch.cuda.is_available()
    args.use_multi_gpu = False
    return args, params


def mark_dim(args):
    if args.embed == "timeF":
        return len(time_features_from_frequency_str(args.freq))
    return {"h": 4, "t": 5}.get(args.freq, 4)


def batch(args, size, device):
    c, d = args.enc_in, mark_dim(args)
    x = torch.randn(size, args.seq_len, c, device=device)
    y = torch.randn(size, args.label_len + args.pred_len, c, device=device)
    x_mark = torch.rand(size, args.seq_len, d, device=device) - 0.5
    y_mark = torch.rand(size, args.label_len + args.pred_len, d, device=device) - 0.5
    return x, y, x_mark, y_mark


def n_params(module):
    return sum(p.numel() for p in module.parameters())


# ----------------------------------------------------------------- passes
def forecast(exp, x, y, x_mark, y_mark):
    """Inference path of Exp_Main.test."""
    xn, stats, _ = exp._normalize_input(x, epoch_idx=None)
    dec = exp._prepare_decoder_input(xn, y)
    out = exp._forward_model(xn, x_mark, dec, y_mark)
    out, _ = exp._trim_outputs_and_target(out, y)
    return exp._denormalize(out, stats, None)


def module_only(exp, x, out_like):
    xn, stats, _ = exp._normalize_input(x, epoch_idx=None)
    return exp._denormalize(out_like, stats, None)


def make_steps(exp):
    """(pretrain_step or None, joint_step) closures that mirror Exp_Main.train."""
    pre_epochs = exp.station_setting[1]
    model_optim = torch.optim.Adam(exp.model.parameters(), lr=exp.args.learning_rate)
    station_params = [p for p in exp.station.parameters() if p.requires_grad]
    station_optim = torch.optim.Adam(station_params, lr=exp.args.station_lr) if station_params else None
    station_in_joint = exp.station_setting[2] > 0 and station_optim is not None
    exp._select_criterion()

    def zero():
        model_optim.zero_grad(set_to_none=True)
        if station_optim is not None:
            station_optim.zero_grad(set_to_none=True)

    def pretrain(x, y, x_mark, y_mark):
        zero()
        _, stats, _ = exp._normalize_input(x, 0)
        target = y[:, -exp.args.pred_len:, exp._select_f_dim():]
        loss = exp.station_loss(target, exp._maybe_trim_statistics_for_ms(stats))
        loss.backward()
        station_optim.step()

    def joint(x, y, x_mark, y_mark):
        zero()
        xn, stats, _ = exp._normalize_input(x, pre_epochs)
        dec = exp._prepare_decoder_input(xn, y)
        out = exp._forward_model(xn, x_mark, dec, y_mark)
        out, y_t = exp._trim_outputs_and_target(out, y)
        loss = exp._with_aux_loss(exp._training_forecast_loss(out, y_t, stats))
        loss.backward()
        model_optim.step()
        if station_in_joint:
            station_optim.step()

    return (pretrain if exp.station_setting[0] > 0 and station_optim is not None else None), joint


# ----------------------------------------------------------------- measuring
def flops(fn):
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        fn()
    return counter.get_total_flops()


def timed(fn, warmup, iters):
    """Per-iteration CUDA-event times (ms) and peak memory (MiB) over the timed iterations."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    t = np.asarray(times)
    return {"mean": float(t.mean()), "p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "peak_mib": peak}


def fits(exp, args, size, device):
    try:
        x, y, xm, ym = batch(args, size, device)
        with torch.no_grad():
            forecast(exp, x, y, xm, ym)
        torch.cuda.synchronize()
        return True
    except RuntimeError as exc:  # torch.cuda.OutOfMemoryError is a RuntimeError
        if "out of memory" not in str(exc).lower():
            raise
        return False
    finally:
        torch.cuda.empty_cache()


def max_batch(exp, args, device, cap):
    size = 1
    while size * 2 <= cap and fits(exp, args, size * 2, device):
        size *= 2
    return size


# ----------------------------------------------------------------- GPU guard
def gpu_state(index):
    out = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    used, util = (float(v) for v in out.strip().split(","))
    return used, util


def wait_until_idle(index, own_mib, tries, interval=60.0):
    """True once the GPU shows no work besides this process (memory beyond our own, or utilization)."""
    for attempt in range(tries):
        time.sleep(2.0)  # let our own last kernels drop out of the utilization window
        samples = []
        for _ in range(3):
            samples.append(gpu_state(index))
            time.sleep(0.5)
        foreign_mib = max(u for u, _ in samples) - own_mib()
        busy = max(p for _, p in samples)
        if foreign_mib < 1500 and busy < 5:
            return True
        log(f"GPU not idle (other memory ~{foreign_mib:.0f} MiB, util {busy:.0f}%), retry {attempt + 1}/{tries}")
        time.sleep(interval)
    return False


def own_processes_running():
    out = subprocess.run(["pgrep", "-f", "run_longExp.py|fill_missing.py run"], capture_output=True, text=True)
    return [p for p in out.stdout.split() if int(p) != os.getpid()]


# ----------------------------------------------------------------- main
def environment(device_index):
    props = torch.cuda.get_device_properties(device_index)
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.split()[0]
    return {"Host": socket.gethostname(), "GPU": props.name, "GPUMemMiB": props.total_memory // 2 ** 20,
            "Driver": driver, "Torch": torch.__version__, "CUDA": torch.version.cuda,
            "Python": platform.python_version(), "TorchThreads": torch.get_num_threads()}


def write_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def existing_keys(path, fields):
    """Keys of the rows already in a CSV (for --resume)."""
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {tuple(str(row[f]) for f in fields) for row in csv.DictReader(handle)}


def train_step_ms(step, x, y, xm, ym, opts):
    """Mean step time and peak memory, or ("OOM", "") when the training batch does not fit."""
    try:
        t = timed(lambda: step(x, y, xm, ym), opts.warmup, opts.train_iters)
        return t["mean"], t["peak_mib"]
    except RuntimeError as exc:  # torch.cuda.OutOfMemoryError is a RuntimeError
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        return "OOM", ""


def new_exp(args):
    """A fresh model pair in eval mode. Each method is built and measured on its own, so the
    peak memory of a method holds only its own weights, gradients and optimizer state."""
    run_longExp.set_seed(2021, deterministic=False)
    exp = Exp_Main(args)
    exp.model.eval()
    exp.station.eval()
    return exp


def release():
    gc.collect()
    torch.cuda.empty_cache()


def bench_case(protocol, bases, dataset, backbone, horizon, methods, opts, device, env, out):
    done = opts.done_cases
    todo = [m for m in methods if (dataset, backbone, m, str(horizon)) not in done]
    if not todo:
        return
    specs = {}
    for method in methods:
        try:
            specs[method] = build_args(protocol, bases, dataset, backbone, method, horizon, lt_locks=opts.lt_locks)
        except KeyError as exc:  # e.g. FAN has no settings for TimeXer/TimeFilter (not run there)
            log(f"{dataset}/{backbone}/H{horizon}: {method} skipped, no settings ({exc})")
            todo = [m for m in todo if m != method]
    # one maximum batch for every method of the case, so throughput is compared at equal batch size
    sizes = []
    for args, _ in specs.values():
        sizes.append(max_batch(new_exp(args), args, device, opts.max_batch))
        release()
    common = min(sizes)
    rows = None
    while rows is None:
        log(f"{dataset}/{backbone}/H{horizon}: common max batch {common}")
        try:
            rows = [bench_method(dataset, backbone, horizon, method, *specs[method], common, opts, device, env)
                    for method in specs if method in todo]
        except torch.cuda.OutOfMemoryError:
            # the maximum batch fits one pass but not repeated passes (allocator fragmentation):
            # measure every method of the case again at half the batch
            if common == 1:
                raise
            common //= 2
            log(f"{dataset}/{backbone}/H{horizon}: out of memory; measuring the case again")
        release()
    for row in rows:  # written only once every method of the case is measured at the same batch
        write_row(out, row)
        joint_ms = row["TrainJointStepMs"]
        log(f"  {row['Method']:5s} e2e B1 {row['LatB1MeanMs']:.2f} ms, Bmax {row['LatBmaxMeanMs']:.1f} ms, "
            f"module {row['ModuleLatB1MeanMs']:.2f} ms, joint step "
            f"{joint_ms if isinstance(joint_ms, str) else f'{joint_ms:.1f} ms'}")


def bench_method(dataset, backbone, horizon, method, args, params, common, opts, device, env):
    """One method of a case at batch 1, at the case's common batch, and one training step."""
    exp = new_exp(args)
    own = lambda: torch.cuda.memory_reserved() / 2 ** 20  # noqa: E731
    idle = wait_until_idle(opts.gpu, own, opts.idle_tries)
    row = {"Timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **env,
           "GPUIdleBefore": idle, "Dataset": dataset, "Backbone": backbone, "Method": method,
           "Horizon": horizon, "SeqLen": args.seq_len, "Channels": args.enc_in, "TrainBatch": args.batch_size,
           "MaxBatch": common, "Settings": json.dumps(params, sort_keys=True), "UpsampleKernel": "",
           "ParamsBackbone": n_params(exp.model), "ParamsModule": n_params(exp.station)}

    x1, y1, xm1, ym1 = batch(args, 1, device)
    out1 = torch.randn(1, args.pred_len, args.enc_in, device=device)
    row["FLOPsTotal"] = flops(lambda: forecast(exp, x1, y1, xm1, ym1))
    row["FLOPsModule"] = flops(lambda: module_only(exp, x1, out1)) if method != "none" else 0

    for label, size in (("B1", 1), ("Bmax", common)):
        x, y, xm, ym = batch(args, size, device)
        out_like = torch.randn(size, args.pred_len, args.enc_in, device=device)
        iters = opts.iters if size == 1 else opts.iters_large
        with torch.no_grad():
            e2e = timed(lambda: forecast(exp, x, y, xm, ym), opts.warmup, iters)
            mod = timed(lambda: module_only(exp, x, out_like), opts.warmup, iters)
        row.update({f"Lat{label}MeanMs": e2e["mean"], f"Lat{label}P50Ms": e2e["p50"],
                    f"Lat{label}P95Ms": e2e["p95"], f"Thr{label}PerSec": size * 1000.0 / e2e["mean"],
                    f"PeakMem{label}MiB": e2e["peak_mib"], f"ModuleLat{label}MeanMs": mod["mean"],
                    f"ModuleLat{label}P95Ms": mod["p95"]})
        del x, y, xm, ym, out_like
        torch.cuda.empty_cache()

    exp.model.train()
    exp.station.train()
    pretrain, joint = make_steps(exp)
    x, y, xm, ym = batch(args, args.batch_size, device)
    row["TrainPreStepMs"] = row["TrainPrePeakMiB"] = ""
    if pretrain is not None:
        row["TrainPreStepMs"], row["TrainPrePeakMiB"] = train_step_ms(pretrain, x, y, xm, ym, opts)
    row["TrainJointStepMs"], row["TrainJointPeakMiB"] = train_step_ms(joint, x, y, xm, ym, opts)
    del x, y, xm, ym, pretrain, joint  # the closures hold the optimizers (and so the model)
    if method == "lt":  # LightNorm's upsampling path on this device (triton / float64 / F.interpolate)
        row["UpsampleKernel"] = ",".join(sorted(
            (plan or {}).get("kernel", "interpolate") for key, plan in lightnorm._UPSAMPLE_PLANS.items()
            if key[0] == "cuda" and key[3] == args.pred_len))
    row["GPUIdleAfter"] = wait_until_idle(opts.gpu, own, 1, interval=0)
    del exp
    release()
    return row


def bench_scaling(protocol, bases, dataset, methods, horizon, lengths, device, env, out, done=frozenset()):
    """Module FLOPs and parameters over input lengths (DLinear rows supply the settings)."""
    for method in methods:
        if method == "none":
            continue
        for seq_len in lengths:
            if (dataset, method, str(horizon), str(seq_len)) in done:
                continue
            args, params = build_args(protocol, bases, dataset, "DLinear", method, horizon, seq_len=seq_len)
            exp = Exp_Main(args)
            exp.station.eval()
            x, _, _, _ = batch(args, 1, device)
            out_like = torch.randn(1, args.pred_len, args.enc_in, device=device)
            write_row(out, {**env, "Dataset": dataset, "Method": method, "Horizon": horizon, "SeqLen": seq_len,
                            "Channels": args.enc_in, "ParamsModule": n_params(exp.station),
                            "FLOPsModule": flops(lambda: module_only(exp, x, out_like)),
                            "Settings": json.dumps(params, sort_keys=True)})
            del exp
    torch.cuda.empty_cache()


def summarize(path):
    """Overhead of each method over NoNorm on the same case (printed table)."""
    import pandas as pd
    df = pd.read_csv(path)
    keys = ["Dataset", "Backbone", "Horizon"]
    cols = {"LatB1MeanMs": "lat B1", "LatBmaxMeanMs": "lat Bmax", "TrainJointStepMs": "train step",
            "PeakMemBmaxMiB": "mem Bmax"}
    for c in cols:   # a training step that ran out of memory is recorded as "OOM"
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ref = df[df.Method == "none"].set_index(keys)
    rows = []
    for _, r in df[df.Method != "none"].iterrows():
        base = ref.loc[tuple(r[k] for k in keys)]
        row = {k: r[k] for k in keys + ["Method"]}
        row.update({name: (f"{100 * (r[c] / base[c] - 1):+.1f}%" if pd.notna(r[c]) else "OOM")
                    for c, name in cols.items()})
        row["module/total FLOPs"] = f"{100 * r.FLOPsModule / max(r.FLOPsTotal, 1):.2f}%"
        row["module params"] = int(r.ParamsModule)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="CSV of the per-case measurements; *_scaling.csv next to it")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--backbones", default=",".join(BACKBONES))
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    p.add_argument("--scale-lengths", default=",".join(map(str, SCALE_LENGTHS)))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200, help="timed iterations at batch 1")
    p.add_argument("--iters-large", type=int, default=50, help="timed iterations at the maximum batch")
    p.add_argument("--train-iters", type=int, default=50)
    p.add_argument("--max-batch", type=int, default=MAX_BATCH)
    p.add_argument("--idle-tries", type=int, default=10, help="minutes to wait for a busy GPU before measuring anyway")
    p.add_argument("--lt-settings", choices=("supplied", "locked"), default="supplied",
                   help="LightNorm setting per case: the supplied launcher setting or the locked one of Exp7")
    p.add_argument("--quick", action="store_true", help="few iterations (smoke test)")
    p.add_argument("--allow-busy", action="store_true", help="run although experiment processes are running")
    p.add_argument("--summary", action="store_true", help="only print the overhead table of --out")
    p.add_argument("--resume", action="store_true",
                   help="skip cases and scaling points already in the output CSVs (the common maximum batch "
                        "is still computed over every method of a case)")
    opts = p.parse_args()
    if opts.summary:
        return summarize(opts.out)
    if opts.quick:
        opts.warmup, opts.iters, opts.iters_large, opts.train_iters, opts.max_batch, opts.idle_tries = 2, 5, 3, 3, 64, 1

    # Bind the process to one GPU before CUDA starts; Exp_Main then uses logical cuda:0.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(opts.gpu))
    physical = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    if not torch.cuda.is_available():
        sys.exit("a CUDA GPU is required")
    busy = own_processes_running()
    if busy and not opts.allow_busy:
        sys.exit(f"experiment processes are running ({' '.join(busy)}); stop the worker first or pass --allow-busy")

    torch.set_num_threads(opts.threads)
    device = torch.device("cuda:0")
    opts.gpu = physical  # nvidia-smi index for the idle checks
    protocol, bases = load_json(PROTOCOL_PATH), load_json(BASE_CONFIGS_PATH)
    opts.lt_locks = load_lt_locks() if opts.lt_settings == "locked" else None
    env = environment(0)
    out = Path(opts.out)
    scaling_out = out.with_name(out.stem + "_scaling.csv")
    split = lambda s: [v for v in s.split(",") if v]  # noqa: E731
    methods = split(opts.methods)
    opts.done_cases = existing_keys(out, ("Dataset", "Backbone", "Method", "Horizon")) if opts.resume else set()
    done_scaling = existing_keys(scaling_out, ("Dataset", "Method", "Horizon", "SeqLen")) if opts.resume else set()
    log(f"{env['GPU']} on {env['Host']}, torch {env['Torch']}; writing {out}"
        + (f" (resume: {len(opts.done_cases)} cases, {len(done_scaling)} scaling points done)" if opts.resume else ""))

    for dataset in split(opts.datasets):
        for backbone in split(opts.backbones):
            for horizon in map(int, split(opts.horizons)):
                bench_case(protocol, bases, dataset, backbone, horizon, methods, opts, device, env, out)
        for horizon in map(int, split(opts.horizons)):
            bench_scaling(protocol, bases, dataset, methods, horizon, list(map(int, split(opts.scale_lengths))),
                          device, env, scaling_out, done_scaling)
    log("done")


if __name__ == "__main__":
    main()

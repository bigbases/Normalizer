"""Diagnostics for TimeFilter vs TimeXer (paper: why RevIN does well on TimeFilter).

V1  error decomposition: the final cells of RevIN, SAN and LightNorm on TimeFilter and TimeXer
    (ETTh1, ETTm1, ETTm2, H=96, seeds 2021-2023, the selected settings of the main results)
    are rerun with their test predictions saved, and the test error is split into a
    low-frequency part (moving average of the error over the horizon) and the rest.
V2  cross-channel block: the TimeFilter cells of RevIN and LightNorm are rerun with every graph
    edge between different channels removed (--tf_block_cross_channel), which stops all
    message passing between channels while keeping the rest of the model.

Commands come from the same task builder as the workers (result_store.build_tasks), so every
cell uses exactly the settings of the reported final runs.  Nothing is written to results/store.

    python experiments/diag_timefilter.py plan --data-root /workspace/datasets --out /workspace/diag_tf
    python experiments/diag_timefilter.py run  --data-root /workspace/datasets --out /workspace/diag_tf --parallel 4
    python experiments/diag_timefilter.py analyze --out /workspace/diag_tf
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

DATASETS = ("ETTh1", "ETTm1", "ETTm2")
BACKBONES = ("TimeFilter", "TimeXer")
METHODS = ("revin", "san", "lt")
SEEDS = (2021, 2022, 2023)
HORIZON = 96


def final_cells():
    import result_store as rs
    from fill_missing import load_context
    protocol, bases, plan = load_context(ROOT)
    results = rs.read_store(ROOT)
    tasks = rs.build_tasks(plan, protocol, bases, results, rs.read_claims(ROOT))
    cells = []
    for t in tasks:
        if t.backbone not in BACKBONES or t.dataset not in DATASETS or not t.cells:
            continue
        if t.experiment not in ("3_frozen_backbone_comparison", "8_lightnorm_tuned_final"):
            continue
        for c in t.cells:
            if (c["stage"] == "final" and c["horizon"] == HORIZON and c["method"] in METHODS
                    and c["seed"] in SEEDS and (c["method"] == "lt") == (t.experiment == "8_lightnorm_tuned_final")):
                cells.append(c)
    return cells


def plan(args):
    import run_matrix as rm
    out = Path(args.out)
    jobs = []
    for c in final_cells():
        for variant in ("v1", "v2"):
            if variant == "v2" and not (c["backbone"] == "TimeFilter" and c["method"] in ("revin", "lt")):
                continue
            name = f'{variant}_{c["backbone"]}_{c["dataset"]}_{c["method"]}_s{c["seed"]}'
            cmd = rm.cell_command(c, args.data_root, out / "results.csv", out / "ckpt" / name, gpu=0)
            cmd[cmd.index("--run_id") + 1] = f'{c["run_id"]}-diag{variant}'
            cmd[cmd.index("--phase") + 1] = f"diag{variant}"
            if "--num_workers" in cmd:
                cmd[cmd.index("--num_workers") + 1] = "0"
            cmd += ["--save_test_preds", str(out / "preds" / f"{name}.npz")]
            if variant == "v2":
                cmd += ["--tf_block_cross_channel", "true"]
            cmd[0] = sys.executable
            jobs.append({"name": name, "variant": variant, "backbone": c["backbone"], "dataset": c["dataset"],
                         "method": c["method"], "seed": c["seed"], "source_run_id": c["run_id"], "cmd": cmd})
    jobs.sort(key=lambda j: (j["variant"] != "v2", j["backbone"] != "TimeFilter", j["dataset"], j["method"], j["seed"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs.json").write_text(json.dumps(jobs, indent=1))
    print(f"{len(jobs)} jobs ({sum(j['variant'] == 'v1' for j in jobs)} V1, {sum(j['variant'] == 'v2' for j in jobs)} V2) -> {out / 'jobs.json'}")
    return jobs


def run(args):
    out = Path(args.out)
    jobs = json.loads((out / "jobs.json").read_text()) if (out / "jobs.json").exists() else plan(args)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out / "preds" / f"{j['name']}.npz").exists()]
    print(f"{len(todo)} of {len(jobs)} jobs to run, {args.parallel} at a time", flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads)}
    running = []
    while todo or running:
        while todo and len(running) < args.parallel:
            j = todo.pop(0)
            log = open(out / "logs" / f"{j['name']}.log", "w")
            running.append((j, subprocess.Popen(j["cmd"], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env), log))
            print(f"[{time.strftime('%F %T')}] start {j['name']}", flush=True)
            time.sleep(5)
        time.sleep(10)
        for item in list(running):
            j, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                ok = proc.returncode == 0 and (out / "preds" / f"{j['name']}.npz").exists()
                print(f"[{time.strftime('%F %T')}] {'done' if ok else 'FAILED rc=%s' % proc.returncode} {j['name']}", flush=True)
    print(f"[{time.strftime('%F %T')}] all jobs finished", flush=True)


def moving_average(e, k):
    """Centered moving average over the horizon axis (1) with replicate padding."""
    import numpy as np
    p = (k - 1) // 2
    ep = np.concatenate([np.repeat(e[:, :1], p, 1), e, np.repeat(e[:, -1:], k - 1 - p, 1)], 1)
    c = np.cumsum(np.concatenate([np.zeros_like(ep[:, :1]), ep], 1), 1)
    return (c[:, k:] - c[:, :-k]) / k


def analyze(args):
    import numpy as np
    out = Path(args.out)
    jobs = json.loads((out / "jobs.json").read_text())
    k = max(5, 2 * (HORIZON // 16) + 1)      # same low-pass rule as the TVR metric (13 at H=96)
    rows = []
    for j in jobs:
        f = out / "preds" / f"{j['name']}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        e = d["pred"].astype(np.float64) - d["true"].astype(np.float64)        # [N, H, C]
        lf = moving_average(e, k)
        level = e.mean(axis=1)
        step = (e ** 2).mean(axis=(0, 2))
        rows.append({**{x: j[x] for x in ("variant", "backbone", "dataset", "method", "seed")},
                     "mse": (e ** 2).mean(), "mse_lowfreq": (lf ** 2).mean(), "mse_rest": ((e - lf) ** 2).mean(),
                     "mse_level": (level ** 2).mean(),
                     "mse_steps_1_8": step[:8].mean(), "mse_steps_9_24": step[8:24].mean(), "mse_steps_25_96": step[24:].mean()})
    keys = list(rows[0].keys())
    with open(out / "per_run.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    metrics = [x for x in keys if x.startswith("mse")]
    groups = {}
    for r in rows:
        groups.setdefault((r["variant"], r["backbone"], r["dataset"], r["method"]), []).append(r)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "backbone", "dataset", "method", "seeds"] + metrics)
        for key in sorted(groups):
            g = groups[key]
            w.writerow(list(key) + [len(g)] + [f"{sum(r[m] for r in g) / len(g):.6f}" for m in metrics])
    print((out / "summary.csv").read_text())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name != "analyze":
            p.add_argument("--data-root", required=True)
        if name == "run":
            p.add_argument("--parallel", type=int, default=4)
            p.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    {"plan": plan, "run": run, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()

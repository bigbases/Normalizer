#!/usr/bin/env python3
"""Find unfinished experiment cells in the packaged results and run them.

    # what is left (reads results/store + results/claims after a git pull)
    python experiments/fill_missing.py list
    python experiments/fill_missing.py list --phases exp2-search --cells

    # work on unfinished tasks until nothing eligible is left
    python experiments/fill_missing.py run --data-root ~/datasets --gpus 0 \\
        --max-processes-per-gpu 4 --datasets ETTh1,ETTh2,ETTm1,ETTm2,Weather

    # make a failed or abandoned task available again
    python experiments/fill_missing.py release exp2-search--ETTh1--DLinear

Several servers can run ``run`` at the same time.  Each claims one
dataset-backbone task at a time by pushing ``results/claims/<task>.json``
(a push race is retried on a fresh checkout, so only one claim wins), keeps
the claim alive with heartbeats, and pushes each finished cell into
``results/store``.  A claim whose heartbeat is older than the lease
(``configs/distributed_plan.json``) is taken over by another worker, so a
crashed server does not block its tasks.  Confirmation and tuned final runs
become available as soon as the search/confirm results of their case are in
the store, whichever server produced them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import result_store as rs  # noqa: E402
import run_matrix as rm  # noqa: E402
from cost_model import CostModel, order_cells  # noqa: E402
from gpu_scheduler import ResourceEstimate, ResourceEstimator, query_gpus, select_gpu  # noqa: E402

ROOT = rs.ROOT

# Test hook (--fake-runner): writes a plausible result row instead of training.
FAKE_RUNNER = r"""
import csv, hashlib, json, sys, time
cell, out = json.loads(sys.argv[1]), sys.argv[2]
time.sleep(float(sys.argv[3]))
h = int(hashlib.sha256(cell["run_id"].encode()).hexdigest()[:8], 16) / 2**32
final = cell["stage"] == "final"
row = dict(RunID=cell["run_id"], CandidateID=cell["candidate_id"], ConfigHash=cell["config_hash"],
           Phase=cell["stage"], Split="test" if final else "validation", Setting="fake",
           Dataset=cell["dataset"], Backbone=cell["backbone"], Horizon=cell["horizon"],
           MSE=0.3 + h if final else "", MAE=0.4 + h if final else "", Seed=cell["seed"],
           BestValMSE=0.5 + h, BestEpoch=1, UseNorm=cell["method"])
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
"""


def log(message):
    print(f"[{datetime.now().strftime('%F %T')}] {message}", flush=True)


def parse_filter(value):
    return set(rm.parse_filter(value) or ())


def task_matches(task, args):
    return (
        (not args.phases or task.phase in parse_filter(args.phases))
        and (not args.datasets or task.dataset in parse_filter(args.datasets))
        and (not args.backbones or task.backbone in parse_filter(args.backbones))
    )


def load_context(root):
    protocol = rm.load_json(rm.PROTOCOL_PATH)
    bases = rm.load_json(rm.BASE_CONFIGS_PATH)
    return protocol, bases, rs.load_plan()


# ---------------------------------------------------------------------- list
def command_list(args):
    repo = rs.GitRepo(ROOT, enabled=not args.no_git)
    repo.sync(read_only=True)
    protocol, bases, plan = load_context(ROOT)
    results = rs.read_store(ROOT)
    cost = CostModel()
    cost.calibrate_from_rows(results.values(), args.gpu_name)
    tasks = [t for t in rs.build_tasks(plan, protocol, bases, results, rs.read_claims(ROOT)) if task_matches(t, args)]
    if not args.all:
        tasks = [t for t in tasks if t.status != "done"]
    if args.format == "json":
        print(json.dumps([{
            "task": t.task_id, "status": t.status, "done": len(t.done),
            "total": None if t.cells is None else len(t.cells),
            "remaining_run_ids": [c["run_id"] for c in t.remaining],
            "worker": (t.claim or {}).get("worker"),
        } for t in tasks], indent=1))
        return
    print(f"{'task':42s} {'status':8s} {'cells':>9s} {'est_h':>7s}  worker")
    total_h = 0.0
    for t in tasks:
        cells = "?" if t.cells is None else f"{len(t.done)}/{len(t.cells)}"
        hours = sum(cost.seconds(c) for c in t.remaining) / 3600
        total_h += hours
        worker = (t.claim or {}).get("worker", "") if t.status in ("claimed", "failed") else ""
        print(f"{t.task_id:42s} {t.status:8s} {cells:>9s} {hours:7.2f}  {worker}")
        if args.cells:
            for c in t.remaining:
                print(f"    {c['run_id']}")
    print(f"packaged_cells={len(results)} listed_tasks={len(tasks)} "
          f"remaining_serial_hours~{total_h:.1f} (blocked tasks not counted)")


def command_release(args):
    repo = rs.GitRepo(ROOT, enabled=not args.no_git)
    if repo.enabled:
        repo.assert_clean()
    removed = repo.transaction(lambda: [t for t in args.task_ids if rs.remove_claim(ROOT, t)],
                               f"claims: release {' '.join(args.task_ids)} [skip ci]")
    log(f"released={removed}")


# ----------------------------------------------------------------------- run
class Worker:
    def __init__(self, args):
        self.args = args
        self.repo = rs.GitRepo(ROOT, enabled=not args.no_git)
        if self.repo.enabled:
            self.repo.assert_clean()
            self.repo.sync()
        self.code_rev = self.repo.code_rev()
        self.protocol, self.bases, self.plan = load_context(ROOT)
        self.lease_hours = float(args.lease_hours or self.plan.get("lease_hours", 3))

        states = query_gpus()
        self.gpu_ids = rm.parse_gpu_ids(args, states)
        self.gpu_name = self.gpu_names().get(self.gpu_ids[0], "unknown")
        self.capacity_mib = min(states[g].total_mib for g in self.gpu_ids)
        self.threads = rm.worker_threads(args, len(self.gpu_ids))
        self.slots = len(self.gpu_ids) * args.max_processes_per_gpu
        self.host = rs.host_name()
        self.worker_id = args.worker_id or f"{self.host}-gpu{'.'.join(map(str, self.gpu_ids))}"
        self.torch = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True).stdout.strip() or "unknown"

        self.local = ROOT / ".worker" / self.worker_id
        self.outbox = self.local / "outbox"
        for sub in ("outbox", "logs", "parts", "ckpt"):
            (self.local / sub).mkdir(parents=True, exist_ok=True)
        self.manifest = self.local / "run_manifest.jsonl"

        self.cost = CostModel()
        self.cost.calibrate_from_rows(rs.read_store(ROOT).values(), self.gpu_name)
        self.estimator = ResourceEstimator(rm.load_json(args.resource_profiles), self.cost)

        self.my_tasks = {}         # task_id -> Task
        self.failed_cells = {}     # task_id -> {run_id: log tail}
        self.queue = []
        self.running = {}
        self.draining = False
        self.last_push = time.monotonic()
        self.next_claim_try = 0.0

    @staticmethod
    def gpu_names():
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        names = {}
        for line in out.splitlines():
            if "," in line:
                idx, name = line.split(",", 1)
                names[int(idx)] = name.strip()
        return names

    # ------------------------------------------------------------- claiming
    def eligible(self, task):
        if not task_matches(task, self.args) or task.cells is None:
            return False
        if any(c["run_id"] in self.failed_cells.get(task.task_id, {}) for c in task.cells):
            return False
        peak = max(self.estimator.estimate(c).memory_mib for c in task.cells)
        return peak + self.args.min_free_memory_mib <= self.capacity_mib

    def task_key(self, task):
        seconds = sum(self.cost.seconds(c) for c in task.remaining)
        return (task.priority, seconds if self.args.order == "fastest-first" else -seconds, task.task_id)

    def claim_next(self):
        chosen = {}

        def mutate():
            chosen.clear()
            results = rs.read_store(ROOT)
            claims = rs.read_claims(ROOT)
            tasks = rs.build_tasks(self.plan, self.protocol, self.bases, results, claims, self.lease_hours)
            mine = lambda t: (t.claim or {}).get("worker") == self.worker_id
            candidates = [
                t for t in tasks
                if t.task_id not in self.my_tasks and self.eligible(t)
                and (t.status == "ready" or (t.status == "claimed" and mine(t)))
            ]
            # Work this worker may still get: ready tasks it can run, and tasks
            # running elsewhere or blocked in cases it can run.
            runnable_cases = {(t.dataset, t.backbone) for t in tasks if self.eligible(t)}
            pending = [
                t for t in tasks if task_matches(t, self.args) and (
                    (t.status == "ready" and self.eligible(t))
                    or (t.status in ("claimed", "blocked") and (t.dataset, t.backbone) in runnable_cases))
            ]
            chosen["outstanding"] = len(pending)
            if not candidates:
                return None
            task = min(candidates, key=self.task_key)
            previous = task.claim if task.status == "ready" and task.claim else None
            rs.write_claim(ROOT, task.task_id, {
                "worker": self.worker_id, "host": self.host, "gpu": self.gpu_name,
                "gpus": self.gpu_ids, "claimed_at": rs.utcnow(), "heartbeat": rs.utcnow(),
                "cells_total": len(task.cells), "cells_remaining": len(task.remaining),
                "took_over_from": (previous or {}).get("worker", ""), "code_rev": self.code_rev,
            })
            chosen["task"] = task
            return task

        task = self.repo.transaction(mutate, f"claims: {self.worker_id} claims next task [skip ci]")
        self.check_code_rev()
        if task is None:
            return None, chosen.get("outstanding", 0)
        self.my_tasks[task.task_id] = task
        active = {item["cell"]["run_id"] for item in self.queue} | {j["cell"]["run_id"] for j in self.running.values()}
        boxed = {p.stem for p in self.outbox.glob("*.csv")}
        new_cells = [c for c in order_cells(task.remaining, self.cost, self.args.order)
                     if c["run_id"] not in active and c["run_id"] not in boxed]
        for cell in new_cells:
            self.queue.append({"cell": cell, "task_id": task.task_id,
                               "estimate": self.estimator.estimate(cell), "retries": 0})
        log(f"claimed {task.task_id}: {len(new_cells)} cells "
            f"(~{sum(self.cost.seconds(c) for c in new_cells) / 3600:.2f} serial h)")
        return task, chosen.get("outstanding", 0)

    # -------------------------------------------------------------- pushing
    def push(self, reason):
        def mutate():
            added = sum(rs.write_cell(ROOT, self.read_row(p)) for p in sorted(self.outbox.glob("*.csv")))
            results = rs.read_store(ROOT)
            claims = rs.read_claims(ROOT)
            finished, lost = [], []
            for task_id, task in self.my_tasks.items():
                remaining = [c for c in task.cells if c["run_id"] not in results]
                claim = claims.get(task_id)
                failed = self.failed_cells.get(task_id, {})
                if claim is None or claim.get("worker") != self.worker_id:
                    (finished if not remaining else lost).append(task_id)
                elif not remaining:
                    rs.remove_claim(ROOT, task_id)
                    finished.append(task_id)
                elif failed and self.task_idle(task_id):
                    claim.update(status="failed", heartbeat=rs.utcnow(), failed_cells=failed)
                    rs.write_claim(ROOT, task_id, claim)
                    finished.append(task_id)
                else:
                    claim.update(heartbeat=rs.utcnow(), cells_remaining=len(remaining))
                    rs.write_claim(ROOT, task_id, claim)
            if finished or self.args.summary_every_push:
                tasks = rs.build_tasks(self.plan, self.protocol, self.bases, results,
                                       rs.read_claims(ROOT), self.lease_hours)
                rs.write_summary(ROOT, results, tasks, self.protocol, self.bases)
            return added, finished, lost

        try:
            added, finished, lost = self.repo.transaction(
                mutate, f"results: {self.worker_id} ({reason}) [skip ci]")
        except RuntimeError as exc:
            log(f"push failed, will retry: {exc}")
            return False
        self.last_push = time.monotonic()
        for path in self.outbox.glob("*.csv"):
            path.unlink()
        for task_id in finished + lost:
            self.my_tasks.pop(task_id, None)
        if lost:
            self.queue = [i for i in self.queue if i["task_id"] not in lost]
            log(f"claims taken over by another worker: {lost}")
        if added or finished:
            log(f"pushed {added} cells; finished tasks: {finished or '-'}")
        self.check_code_rev()
        return True

    def task_idle(self, task_id):
        return not any(i["task_id"] == task_id for i in self.queue) and \
            not any(j["task_id"] == task_id for j in self.running.values())

    @staticmethod
    def read_row(path):
        with path.open(newline="") as f:
            return next(csv.DictReader(f))

    def check_code_rev(self):
        if self.repo.enabled and not self.draining and self.repo.code_rev() != self.code_rev:
            self.draining = True
            log("code on the remote branch changed: finishing running cells, then exiting. "
                "Restart the worker to continue with the new code.")

    # -------------------------------------------------------------- running
    def launch_ready(self):
        if self.draining or not self.queue:
            return
        states = query_gpus()
        by_gpu = {}
        for job in self.running.values():
            by_gpu.setdefault(job["gpu_id"], []).append(job)
        for item in list(self.queue):
            gpu_id = select_gpu(item["estimate"], states, by_gpu, self.gpu_ids,
                                self.args.max_processes_per_gpu, self.args.max_gpu_util,
                                self.args.min_free_memory_mib)
            if gpu_id is None:
                continue
            cell = item["cell"]
            attempt = item["retries"] + 1
            part = self.local / "parts" / f'{cell["run_id"]}.attempt{attempt}.csv'
            part.unlink(missing_ok=True)
            log_path = self.local / "logs" / f'{cell["run_id"]}.attempt{attempt}.log'
            cmd = rm.cell_command(cell, self.args.data_root, part, self.local / "ckpt", gpu=0)
            if self.args.fake_runner is not None:
                keys = ("run_id", "candidate_id", "config_hash", "stage", "dataset", "backbone",
                        "horizon", "seed", "method")
                cmd = [sys.executable, "-c", FAKE_RUNNER, json.dumps({k: cell[k] for k in keys}),
                       str(part), str(self.args.fake_runner)]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu_id), PYTHONUNBUFFERED="1")
            for var in ("LIGHTNORM_TORCH_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                env[var] = str(self.threads)
            handle = log_path.open("w")
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            job = {**item, "process": process, "gpu_id": gpu_id, "part": part, "log": log_path,
                   "handle": handle, "started": time.monotonic(), "started_at": rs.utcnow()}
            self.running[process.pid] = job
            by_gpu.setdefault(gpu_id, []).append(job)
            self.queue.remove(item)
            rm.append_manifest(self.manifest, {"event": "started", "at": rs.utcnow(), "run_id": cell["run_id"],
                                               "physical_gpu": gpu_id, "attempt": attempt})

    def reap(self):
        for pid, job in list(self.running.items()):
            code = job["process"].poll()
            if code is None:
                continue
            job["handle"].close()
            del self.running[pid]
            cell = job["cell"]
            elapsed = time.monotonic() - job["started"]
            rm.append_manifest(self.manifest, {"event": "finished", "at": rs.utcnow(), "run_id": cell["run_id"],
                                               "returncode": code, "elapsed_seconds": round(elapsed, 3)})
            if code == 0 and job["part"].exists():
                row = self.read_row(job["part"])
                row.update(Host=self.host, Worker=self.worker_id, GPU=self.gpu_name, Torch=self.torch,
                           CodeRev=self.code_rev, ElapsedSeconds=f"{elapsed:.1f}", FinishedAt=rs.utcnow())
                tmp = self.outbox / f'{cell["run_id"]}.tmp'
                with tmp.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rs.FIELDS, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerow(row)
                tmp.replace(self.outbox / f'{cell["run_id"]}.csv')
                job["part"].unlink(missing_ok=True)
                if not rm.keep_checkpoints(self.args, cell):
                    shutil.rmtree(rm.run_checkpoint_dir(self.local / "ckpt", cell), ignore_errors=True)
                continue
            tail = rm.tail_text(job["log"], 4000)
            if "out of memory" in tail.lower() and job["retries"] < self.args.max_retries:
                estimate = ResourceEstimate(int(job["estimate"].memory_mib * 1.5),
                                            max(job["estimate"].util_percent, 80), True)
                self.queue.insert(0, {"cell": cell, "task_id": job["task_id"],
                                      "estimate": estimate, "retries": job["retries"] + 1})
                log(f"OOM, retrying exclusively: {cell['run_id']}")
                continue
            self.failed_cells.setdefault(job["task_id"], {})[cell["run_id"]] = tail[-1500:]
            log(f"FAILED {cell['run_id']} (rc={code}); log: {job['log']}")

    # ----------------------------------------------------------------- loop
    def run(self):
        log(f"worker={self.worker_id} gpus={self.gpu_ids} ({self.gpu_name}, {self.capacity_mib} MiB) "
            f"slots={self.slots} threads={self.threads} torch={self.torch} code={self.code_rev} "
            f"git={'on' if self.repo.enabled else 'off'}")
        push_every = self.args.push_minutes * 60
        beat_every = self.args.heartbeat_minutes * 60
        while True:
            self.reap()
            now = time.monotonic()
            outbox = list(self.outbox.glob("*.csv"))
            due = (outbox and (now - self.last_push >= push_every or not self.running)) or \
                  (self.my_tasks and now - self.last_push >= beat_every) or \
                  any(self.failed_cells.get(t) and self.task_idle(t) for t in self.my_tasks)
            if due:
                self.push("progress")

            if self.draining:
                if not self.running:
                    self.push("drain")
                    log("drained; exiting so the new code can be used")
                    return 3
            elif not self.queue and len(self.running) < self.slots and now >= self.next_claim_try:
                try:
                    task, outstanding = self.claim_next()
                except RuntimeError as exc:
                    log(f"claim failed, retrying in 1 min: {exc}")
                    self.next_claim_try = now + 60
                    task, outstanding = None, -1
                if task is None and outstanding < 0:
                    pass
                elif task is None:
                    self.next_claim_try = now + self.args.poll_minutes * 60
                    if not self.running and not self.my_tasks and not list(self.outbox.glob("*.csv")):
                        if outstanding == 0:
                            log("no unfinished eligible tasks left; exiting")
                            return 0
                        if self.args.once:
                            log(f"{outstanding} tasks are running elsewhere or blocked; exiting (--once)")
                            return 0
                        log(f"{outstanding} tasks running elsewhere or blocked; "
                            f"checking again in {self.args.poll_minutes} min")
            self.launch_ready()
            time.sleep(self.args.poll_seconds)


def command_run(args):
    if args.max_processes_per_gpu < 1:
        raise SystemExit("--max-processes-per-gpu must be >= 1")
    return Worker(args).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--phases", help="comma-separated plan phases (configs/distributed_plan.json)")
        p.add_argument("--datasets", help="comma-separated datasets, e.g. ETTh1,ETTh2,ETTm1,ETTm2,Weather")
        p.add_argument("--backbones", help="comma-separated backbones")
        p.add_argument("--no-git", action="store_true", help="work on the local checkout only (no pull/push)")

    p = sub.add_parser("list", help="show unfinished tasks")
    common(p)
    p.add_argument("--all", action="store_true", help="include finished tasks")
    p.add_argument("--cells", action="store_true", help="print remaining run IDs")
    p.add_argument("--format", choices=["table", "json"], default="table")
    p.add_argument("--gpu-name", help="use measured runtimes of this GPU model for estimates")

    r = sub.add_parser("run", help="claim and run unfinished tasks")
    common(r)
    r.add_argument("--data-root", required=True)
    r.add_argument("--gpus", help="physical GPU indices, e.g. 0 or 0,1")
    r.add_argument("--gpu-count", type=int)
    r.add_argument("--max-processes-per-gpu", type=int, default=4)
    r.add_argument("--max-gpu-util", type=int, default=92)
    r.add_argument("--min-free-memory-mib", type=int, default=1024)
    r.add_argument("--cpu-threads-per-worker", type=int, default=0)
    r.add_argument("--resource-profiles", default=str(ROOT / "configs" / "resource_profiles.a100_80gb.json"))
    r.add_argument("--order", choices=["fastest-first", "largest-first"], default="fastest-first")
    r.add_argument("--keep-checkpoints", choices=["final", "all", "none"], default="final")
    r.add_argument("--max-retries", type=int, default=1)
    r.add_argument("--worker-id", help="stable name; default <host>-gpu<ids>")
    r.add_argument("--lease-hours", type=float, help="override the plan's claim lease")
    r.add_argument("--push-minutes", type=float, default=3.0)
    r.add_argument("--heartbeat-minutes", type=float, default=20.0)
    r.add_argument("--poll-minutes", type=float, default=10.0)
    r.add_argument("--poll-seconds", type=float, default=5.0)
    r.add_argument("--summary-every-push", action="store_true")
    r.add_argument("--once", action="store_true", help="exit instead of waiting for other servers")
    r.add_argument("--fake-runner", type=float, default=None, help=argparse.SUPPRESS)

    rel = sub.add_parser("release", help="delete claims so tasks can be picked up again")
    rel.add_argument("task_ids", nargs="+")
    rel.add_argument("--no-git", action="store_true")

    args = parser.parse_args()
    if args.command == "list":
        return command_list(args)
    if args.command == "release":
        return command_release(args)
    return command_run(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

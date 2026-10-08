"""Packaged experiment results shared through the Git repository.

Layout (tracked on ``main``):

    results/store/<phase>/<dataset>/<backbone>/<run_id>.csv  one finished cell per file
    results/claims/<task_id>.json                            leases of tasks being run
    results/summary/                                         generated; never edit by hand

One file per cell makes concurrent pushes from several servers conflict-free,
and the run ID (a hash of the full effective configuration) makes packing
idempotent: the first result for a run ID wins.

A *task* is one dataset-backbone case of one phase of ``configs/distributed_plan.json``
(e.g. ``search--ETTh1--DLinear``).  Its cells are derived from the protocol and,
for confirmation/final tuned runs, from the shortlists/locks computed from the
packaged validation rows, so every server derives the same work list.
"""

from __future__ import annotations

import csv
import json
import os
import random
import socket
import subprocess
import sys
import time
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_matrix as rm  # noqa: E402
import select_hparams as sh  # noqa: E402

ROOT = rm.ROOT
STORE = Path("results") / "store"
CLAIMS = Path("results") / "claims"
SUMMARY = Path("results") / "summary"
PLAN_PATH = ROOT / "configs" / "distributed_plan.json"

RESULT_FIELDS = [
    "RunID", "CandidateID", "ConfigHash", "Phase", "Split", "Setting", "Dataset",
    "Backbone", "Horizon", "MSE", "MAE", "Seed", "BestValMSE", "BestEpoch", "UseNorm",
]
PROVENANCE_FIELDS = ["Host", "Worker", "GPU", "Torch", "CodeRev", "ElapsedSeconds", "FinishedAt"]
FIELDS = RESULT_FIELDS + PROVENANCE_FIELDS


def utcnow():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------- store
def cell_path(root, row):
    return Path(root) / STORE / row["Phase"] / row["Dataset"] / row["Backbone"] / f'{row["RunID"]}.csv'


def read_store(root):
    """All packaged cells, keyed by run ID."""
    results = {}
    base = Path(root) / STORE
    if not base.exists():
        return results
    for path in sorted(base.rglob("*.csv")):
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) == 1 and rows[0].get("RunID") == path.stem:
            results[path.stem] = rows[0]
    return results


def write_cell(root, row):
    """Add one finished cell; returns False when the run ID is already packaged."""
    missing = [k for k in ("RunID", "Phase", "Dataset", "Backbone", "Split") if not row.get(k)]
    if missing:
        raise ValueError(f"result row lacks {missing}: {row}")
    path = cell_path(root, row)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})
    tmp.replace(path)
    return True


# ---------------------------------------------------------------------- plan
@dataclass
class Task:
    phase: str
    priority: int
    experiment: str
    stage: str
    dataset: str
    backbone: str
    methods: list
    requires: str | None = None
    cells: list | None = None          # None while blocked by a dependency
    done: list = field(default_factory=list)
    status: str = "pending"            # done | ready | blocked | claimed | failed
    claim: dict | None = None
    split: str | None = None           # the method, for phases split per method

    @property
    def task_id(self):
        base = f"{self.phase}--{self.dataset}--{self.backbone}"
        return f"{base}--{self.split}" if self.split else base

    @property
    def remaining(self):
        return [] if self.cells is None else [c for c in self.cells if c["run_id"] not in self.done]


def load_plan(path=PLAN_PATH):
    return json.loads(Path(path).read_text())


def _cells(protocol, bases, experiment, stage, dataset, backbone, methods, lock_doc=None, shortlist_doc=None):
    args = Namespace(
        experiment=experiment, stage=stage, datasets=dataset, backbones=backbone,
        methods=",".join(methods), locks=None, shortlist=None,
    )
    return rm.build_cells(args, protocol, bases, lock_doc=lock_doc, shortlist_doc=shortlist_doc)


def _case_rows(results, dataset, backbone):
    return [r for r in results.values() if r["Dataset"] == dataset and r["Backbone"] == backbone]


def build_tasks(plan, protocol, bases, results, claims=None, lease_hours=None, now=None):
    """Every task of the enabled plan phases, with cells and status resolved."""
    claims = claims or {}
    lease_hours = float(lease_hours or plan.get("lease_hours", 3))
    now = now or time.time()
    tasks = []
    by_case = defaultdict(list)
    for phase in sorted(plan["phases"], key=lambda p: p["priority"]):
        if not phase.get("enabled", True):
            continue
        # "split": "method" makes one task per normalizer so that several
        # servers can share a heavy dataset-backbone case.
        split = phase.get("split") == "method"
        groups = [[m] for m in phase["methods"]] if split else [list(phase["methods"])]
        datasets = phase.get("datasets") or list(protocol["datasets"])
        # "cases" lists explicit [dataset, backbone] pairs instead of the product.
        cases = phase.get("cases") or [(d, b) for d in datasets for b in phase["backbones"]]
        for dataset, backbone in cases:
            for methods in groups:
                task = Task(
                    phase=phase["name"], priority=int(phase["priority"]),
                    experiment=phase["experiment"], stage=phase["stage"],
                    dataset=dataset, backbone=backbone, methods=methods,
                    requires=phase.get("requires"), split=methods[0] if split else None,
                )
                by_case[(task.phase, dataset, backbone)].append(task)
                tasks.append(task)

    for task in tasks:  # phases are sorted, so dependencies resolve first
        if task.requires:
            deps = by_case.get((task.requires, task.dataset, task.backbone), [])
            same_method = [d for d in deps if task.split and d.split == task.split]
            deps = same_method or deps     # per-method chain when both phases are split
            if not deps or any(d.status != "done" for d in deps):
                task.status = "blocked"
                continue
        rows = sh.validation_rows(_case_rows(results, task.dataset, task.backbone))
        lock_doc = shortlist_doc = None
        if task.stage == "confirm":
            shortlist_doc = sh.select(rows, "shortlist", protocol, bases)
        elif task.stage == "final" and any(m not in ("none", "lt") for m in task.methods):
            lock_doc = sh.select(rows, "lock", protocol, bases)
        try:
            task.cells = _cells(protocol, bases, task.experiment, task.stage, task.dataset,
                                task.backbone, task.methods, lock_doc, shortlist_doc)
        except ValueError:
            task.status = "blocked"      # selection incomplete for this case
            continue
        task.done = [c["run_id"] for c in task.cells if c["run_id"] in results]
        claim = claims.get(task.task_id)
        task.claim = claim
        if not task.remaining:
            task.status = "done"
        elif claim and claim.get("status") == "failed":
            task.status = "failed"
        elif claim and claim_is_live(claim, now, lease_hours):
            task.status = "claimed"
        else:
            task.status = "ready"
    return tasks


# -------------------------------------------------------------------- claims
def read_claims(root):
    claims = {}
    base = Path(root) / CLAIMS
    if base.exists():
        for path in base.glob("*.json"):
            try:
                claims[path.stem] = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
    return claims


def claim_is_live(claim, now, lease_hours):
    try:
        beat = datetime.fromisoformat(claim["heartbeat"]).timestamp()
    except (KeyError, ValueError):
        return False
    return now - beat < lease_hours * 3600


def write_claim(root, task_id, claim):
    path = Path(root) / CLAIMS / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(claim, indent=1, sort_keys=True) + "\n")


def remove_claim(root, task_id):
    path = Path(root) / CLAIMS / f"{task_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


# ------------------------------------------------------------------- summary
def export_cells(results, path):
    """All packaged cells as one CSV (written outside the tracked summary)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results.values(), key=lambda r: r["RunID"]))


def write_summary(root, results, tasks, protocol, bases):
    """Regenerate results/summary/* deterministically from the packaged cells."""
    out = Path(root) / SUMMARY
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(results.values(), key=lambda r: r["RunID"])

    groups = defaultdict(list)
    for r in rows:
        if r["Split"] == "test" and r.get("MSE"):
            groups[(r["Dataset"], r["Backbone"], r["UseNorm"], int(r["Horizon"]))].append(r)
    order = {d: i for i, d in enumerate(protocol["datasets"])}
    with (out / "final_test_mean_std.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Backbone", "Method", "Horizon", "Seeds",
                         "MSE_mean", "MSE_std", "MAE_mean", "MAE_std", "GPUs"])
        for key in sorted(groups, key=lambda k: (order.get(k[0], 99), k[1], k[2], k[3])):
            items = sorted(groups[key], key=lambda r: int(r["Seed"]))
            mse = [float(r["MSE"]) for r in items]
            mae = [float(r["MAE"]) for r in items]
            writer.writerow([
                *key, " ".join(r["Seed"] for r in items),
                f"{mean(mse):.6f}", f"{stdev(mse):.6f}" if len(mse) > 1 else "",
                f"{mean(mae):.6f}", f"{stdev(mae):.6f}" if len(mae) > 1 else "",
                ";".join(sorted({r.get("GPU", "") for r in items})),
            ])

    val = sh.validation_rows(rows)
    for mode, name in (("shortlist", "selection_shortlist.json"), ("lock", "selection_locks.json")):
        doc = sh.select(val, mode, protocol, bases)
        (out / name).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    counts = defaultdict(lambda: defaultdict(int))
    for t in tasks:
        counts[t.phase][t.status] += 1
    lines = [
        "# Experiment progress", "",
        f"Generated {utcnow()} from {len(rows)} packaged cells. "
        "Regenerate with `python experiments/results_pack.py summary`.", "",
        "| phase | done | running (claimed) | ready | blocked | failed |",
        "|---|---|---|---|---|---|",
    ]
    for phase in dict.fromkeys(t.phase for t in tasks):
        c = counts[phase]
        lines.append(f"| {phase} | {c['done']} | {c['claimed']} | {c['ready']} | {c['blocked']} | {c['failed']} |")
    lines += ["", "| task | status | cells done | worker |", "|---|---|---|---|"]
    for t in tasks:
        total = "?" if t.cells is None else len(t.cells)
        worker = (t.claim or {}).get("worker", "") if t.status in ("claimed", "failed") else ""
        lines.append(f"| {t.task_id} | {t.status} | {len(t.done)}/{total} | {worker} |")
    (out / "progress.md").write_text("\n".join(lines) + "\n")

    # Per-stage CSV tables are a convenience: a bug there must never block a
    # worker from pushing finished cells.
    try:
        import summary_tables
        summary_tables.write_all(out, rows, tasks, protocol, bases)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: stage tables not written: {exc!r}")


# ----------------------------------------------------------------------- git
class UnsafeSync(Exception):
    """The checkout holds work that a reset to the remote would discard."""


class GitRepo:
    """Optimistic concurrency over one branch: reset to remote, mutate, push, retry."""

    TRACKED = [str(STORE), str(CLAIMS), str(SUMMARY)]

    def __init__(self, root, remote="origin", branch="main", enabled=True):
        self.root = Path(root)
        self.remote = remote
        self.branch = branch
        self.enabled = enabled

    def git(self, *args, check=True):
        result = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True)
        if check and result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result

    def assert_clean(self):
        dirty = self.git("status", "--porcelain", "--untracked-files=no").stdout.strip()
        if dirty:
            raise RuntimeError(
                "tracked files have local modifications; commit or stash them first "
                "(workers reset the checkout to the remote branch):\n" + dirty
            )

    # Files that change what a worker runs; reporting code and docs are not
    # listed, so updating them does not make running workers restart.
    RUNTIME_PATHS = [
        "run_longExp.py", "exp", "models", "normalizers", "layers", "utils", "data_provider",
        "requirements.txt", "configs/protocol.json", "configs/base_configs.json",
        "configs/distributed_plan.json", "configs/cost_profile.json",
        "configs/resource_profiles.a100_80gb.json", "experiments/fill_missing.py",
        "experiments/result_store.py", "experiments/run_matrix.py", "experiments/cost_model.py",
        "experiments/gpu_scheduler.py", "experiments/select_hparams.py",
    ]

    def code_rev(self):
        """Last commit that touched code a worker runs (see RUNTIME_PATHS)."""
        try:
            out = self.git("log", "-1", "--format=%h", "--", *self.RUNTIME_PATHS).stdout.strip()
        except (RuntimeError, OSError):
            return "unknown"
        return out or "unknown"

    def sync(self, read_only=False):
        """Bring the checkout to the remote branch.

        Workers reset hard (their own unpushed result commits are re-applied
        from the outbox), but never over local commits that touch code.
        ``read_only`` only fast-forwards and otherwise keeps the checkout.
        """
        if not self.enabled:
            return
        self.git("fetch", "-q", self.remote, self.branch)
        upstream = f"{self.remote}/{self.branch}"
        if read_only:
            if self.git("merge", "-q", "--ff-only", upstream, check=False).returncode != 0:
                print(f"warning: local branch diverged from {upstream}; showing local state")
            return
        ahead = self.git("log", "--format=", "--name-only", f"{upstream}..HEAD").stdout.split()
        code = sorted({path for path in ahead if not path.startswith("results/")})
        if code:
            raise UnsafeSync(
                "local commits not on the remote change code; push or move them before "
                "running a worker here: " + ", ".join(code[:5])
            )
        self.git("reset", "-q", "--hard", upstream)

    def transaction(self, mutate, message, attempts=12):
        """Run ``mutate()`` on a fresh checkout and push; retried on races.

        ``mutate`` must derive everything from the files on disk because it is
        re-applied after every reset.  Returns its result.
        """
        if not self.enabled:
            return mutate()
        last_error = None
        for attempt in range(attempts):
            try:
                self.sync()
                result = mutate()
                # Pathspecs must exist (and deleted claim files are still tracked).
                paths = [p for p in self.TRACKED if (self.root / p).exists()
                         or self.git("ls-files", "--", p).stdout.strip()]
                if paths:
                    self.git("add", "-A", "--", *paths)
                if self.git("diff", "--cached", "--quiet", check=False).returncode == 0:
                    return result
                self.git("commit", "-q", "-m", message)
                push = self.git("push", "-q", self.remote, f"HEAD:{self.branch}", check=False)
                if push.returncode == 0:
                    return result
                last_error = push.stderr.strip()
            except RuntimeError as exc:   # network hiccups: retry with backoff
                last_error = str(exc)
            time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
        raise RuntimeError(f"git transaction '{message}' failed after {attempts} attempts: {last_error}")


def host_name():
    return os.environ.get("LIGHTNORM_HOST", socket.gethostname())

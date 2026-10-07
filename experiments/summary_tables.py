"""Per-stage CSV tables in results/summary/, regenerated from the packaged cells.

    stage_status.csv            every task: status and cells done / total
    exp1_lightnorm.csv          LightNorm test MSE/MAE per dataset-backbone-horizon,
                                per seed and mean/std; Complete = all protocol seeds
    exp2_validation.csv         every search candidate: validation MSE per horizon and
                                seed (search seed 2021, confirm seed 2022), shortlist/lock
    exp2_selection.csv          the locked setting per dataset-backbone-normalizer
    exp3_comparison.csv         test MSE/MAE mean/std of NoNorm/RevIN/SAN/DDN/FAN/LightNorm
                                side by side; Complete = all six methods have all seeds
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import select_hparams as sh

METHODS = ["none", "revin", "san", "ddn", "fan", "lt"]


def _num(value):
    return "" if value is None else f"{value:.6f}"


def _std(values):
    return stdev(values) if len(values) > 1 else None


def _write(path, header, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _order(protocol):
    datasets = {d: i for i, d in enumerate(protocol["datasets"])}
    return lambda ds, bb: (datasets.get(ds, 99), bb)


def stage_status(out, tasks):
    _write(out / "stage_status.csv",
           ["Phase", "Dataset", "Backbone", "Method", "Status", "CellsDone", "CellsTotal", "Worker"],
           [[t.phase, t.dataset, t.backbone, "+".join(t.methods), t.status, len(t.done),
             "" if t.cells is None else len(t.cells),
             (t.claim or {}).get("worker", "") if t.status in ("claimed", "failed") else ""]
            for t in tasks])


def final_groups(rows):
    groups = defaultdict(dict)
    for r in rows:
        if r.get("Phase") == "final" and r.get("Split") == "test" and r.get("MSE"):
            key = (r["Dataset"], r["Backbone"], r["UseNorm"], int(r["Horizon"]))
            groups[key][int(r["Seed"])] = r
    return groups


def exp1_lightnorm(out, groups, protocol):
    seeds = protocol["seeds"]
    order = _order(protocol)
    keys = sorted((k for k in groups if k[2] == "lt"), key=lambda k: (*order(k[0], k[1]), k[3]))
    rows = []
    for ds, bb, _, h in keys:
        by_seed = groups[(ds, bb, "lt", h)]
        mse = [float(by_seed[s]["MSE"]) for s in seeds if s in by_seed]
        mae = [float(by_seed[s]["MAE"]) for s in seeds if s in by_seed]
        rows.append([
            ds, bb, h, all(s in by_seed for s in seeds), len(mse),
            _num(mean(mse)), _num(_std(mse)), _num(mean(mae)), _num(_std(mae)),
            *[_num(float(by_seed[s]["MSE"])) if s in by_seed else "" for s in seeds],
            *[_num(float(by_seed[s]["MAE"])) if s in by_seed else "" for s in seeds],
            ";".join(sorted({r.get("GPU", "") for r in by_seed.values()})),
            ";".join(sorted({r.get("Torch", "") for r in by_seed.values()})),
        ])
    _write(out / "exp1_lightnorm.csv",
           ["Dataset", "Backbone", "Horizon", "Complete", "Seeds", "MSE_mean", "MSE_std",
            "MAE_mean", "MAE_std", *[f"MSE_s{s}" for s in seeds], *[f"MAE_s{s}" for s in seeds],
            "GPU", "Torch"], rows)


def exp2_validation(out, rows, protocol, bases):
    val = sh.validation_rows(rows)
    shortlist = sh.select(val, "shortlist", protocol, bases)["shortlists"]
    locks = sh.select(val, "lock", protocol, bases)["locks"]
    catalog = sh.candidate_catalog(protocol, bases)
    shortlisted = {(k, item["candidate_id"]) for k, items in shortlist.items() for item in items}
    scores = defaultdict(dict)
    for r in val:
        key = (r["Dataset"], r["Backbone"], r["UseNorm"], r["CandidateID"])
        scores[key][(r["Phase"], r["Horizon"], r["Seed"])] = r["BestValMSE"]

    order = _order(protocol)
    cols = [("search", 96, 2021), ("search", 720, 2021), ("confirm", 96, 2022), ("confirm", 720, 2022)]
    table, selection = [], []
    for key in sorted(scores, key=lambda k: (*order(k[0], k[1]), k[2], k[3])):
        ds, bb, method, cid = key
        group = f"{ds}|{bb}|{method}"
        params = catalog.get(key, {})
        values = scores[key]
        search = [values[c] for c in cols[:2] if c in values]
        allv = [values[c] for c in cols if c in values]
        locked = locks.get(group) == params and len(allv) == 4
        table.append([
            ds, bb, method, cid, json.dumps(params, sort_keys=True),
            *[_num(values.get(c)) for c in cols],
            _num(mean(search)) if len(search) == 2 else "",
            _num(mean(allv)) if len(allv) == 4 else "",
            (group, cid) in shortlisted, locked,
        ])
        if locked:
            selection.append([ds, bb, method, cid, json.dumps(params, sort_keys=True), _num(mean(allv))])
    _write(out / "exp2_validation.csv",
           ["Dataset", "Backbone", "Method", "CandidateID", "Params",
            "Search_h96_s2021", "Search_h720_s2021", "Confirm_h96_s2022", "Confirm_h720_s2022",
            "Screen_mean", "Lock_mean", "Shortlisted", "Locked"], table)
    _write(out / "exp2_selection.csv",
           ["Dataset", "Backbone", "Method", "CandidateID", "Params", "Lock_mean_val_MSE"], selection)


def exp3_comparison(out, groups, protocol):
    seeds = protocol["seeds"]
    order = _order(protocol)
    cases = sorted({(k[0], k[1], k[3]) for k in groups}, key=lambda k: (*order(k[0], k[1]), k[2]))
    rows = []
    for ds, bb, h in cases:
        line, complete, best = [ds, bb, h], True, None
        for m in METHODS:
            by_seed = groups.get((ds, bb, m, h), {})
            mse = [float(r["MSE"]) for r in by_seed.values()]
            mae = [float(r["MAE"]) for r in by_seed.values()]
            full = all(s in by_seed for s in seeds)
            complete &= full
            if full and (best is None or mean(mse) < best[1]):
                best = (m, mean(mse))
            line += [len(mse), _num(mean(mse)) if mse else "", _num(_std(mse)) if mse else "",
                     _num(mean(mae)) if mae else "", _num(_std(mae)) if mae else ""]
        rows.append(line + [complete, best[0] if complete and best else ""])
    header = ["Dataset", "Backbone", "Horizon"]
    for m in METHODS:
        header += [f"{m}_n", f"{m}_MSE", f"{m}_MSE_std", f"{m}_MAE", f"{m}_MAE_std"]
    _write(out / "exp3_comparison.csv", header + ["Complete", "Best_MSE"], rows)


def write_all(out, rows, tasks, protocol, bases):
    out = Path(out)
    groups = final_groups(rows)
    stage_status(out, tasks)
    exp1_lightnorm(out, groups, protocol)
    exp2_validation(out, rows, protocol, bases)
    exp3_comparison(out, groups, protocol)

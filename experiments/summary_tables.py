"""Per-stage CSV tables in results/summary/, regenerated from the packaged cells.

    stage_status.csv            every task: status and cells done / total
    exp1_lightnorm.csv          LightNorm test MSE/MAE per dataset-backbone-horizon,
                                per seed and mean/std; Complete = all protocol seeds
    exp2_validation.csv         every search candidate: validation MSE per horizon and
                                seed (search seed 2021, confirm seed 2022), shortlist/lock
    exp2_selection.csv          the locked setting per dataset-backbone-normalizer
    exp3_comparison.csv         test MSE/MAE mean/std of NoNorm/RevIN/SAN/DDN/FAN/LightNorm
                                side by side; Complete = all six methods have all seeds
    exp5_lt_pilot.csv           LightNorm pilot candidates (validation, seed 2021) next to
                                the supplied setting
    exp5_lt_pilot_effects.csv   factorial main effects / interactions of the pilot vs seed noise
    exp6_lt_explore.csv         exploratory LightNorm settings on dev cases (TEST split) against the
                                supplied LightNorm, SAN and DDN on the same seeds
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import run_matrix as rm
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
    val = [r for r in sh.validation_rows(rows) if r["UseNorm"] in sh.BASELINE_METHODS]
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


def _lt_id(params):
    return rm.canonical_hash({"method": "lt", "params": rm.complete_method_params("lt", params)})


PILOT_FACTORS = ("s_norm", "kernel_size", "station_lr")
PILOT_TERMS = [(f,) for f in PILOT_FACTORS] + [
    (a, b) for i, a in enumerate(PILOT_FACTORS) for b in PILOT_FACTORS[i + 1:]]


def exp5_lt_pilot(out, rows, protocol, bases):
    """Pilot candidates and factorial effects, per pilot case and screen horizon."""
    val = defaultdict(dict)      # (ds, bb, phase, candidate) -> {(horizon, seed): BestValMSE}
    for r in sh.validation_rows(rows) + [
            dict(r, BestValMSE=float(r["BestValMSE"]), Horizon=int(r["Horizon"]), Seed=int(r["Seed"]))
            for r in rows if r.get("Phase") == "final" and r.get("UseNorm") == "lt" and r.get("BestValMSE")]:
        if r["UseNorm"] == "lt":
            val[(r["Dataset"], r["Backbone"], r["Phase"], r["CandidateID"])][(r["Horizon"], r["Seed"])] = r["BestValMSE"]
    horizons = protocol["screen_horizons"]
    table, effects = [], []
    for ds, bb in protocol["lt_tuning"]["pilot"]["cases"]:
        base = rm.selected_base_config(bases, ds, bb)
        supplied = rm.complete_method_params("lt", rm.supplied_lt_params(base))
        sup_vals = val.get((ds, bb, "final", _lt_id(supplied)), {})
        sup = {h: sup_vals.get((h, 2021)) for h in horizons}
        noise = {}
        for h in horizons:
            seeds = [v for (hh, _), v in sup_vals.items() if hh == h]
            noise[h] = stdev(seeds) / mean(seeds) * 100 if len(seeds) > 1 else None
        sup_mean = mean(sup.values()) if all(sup.values()) else None
        centre_k = protocol["lt_tuning"]["pilot"]["centre_kernel_size"]
        points = []
        for params in rm.lt_pilot_candidates(protocol, ds, base):
            full = rm.complete_method_params("lt", params)
            got = val.get((ds, bb, "search", _lt_id(params)), {})
            vals = {h: got.get((h, 2021)) for h in horizons}
            points.append((full, vals))
        means = {i: mean(v.values()) for i, (_, v) in enumerate(points) if all(v.values())}
        ranks = {i: n + 1 for n, i in enumerate(sorted(means, key=means.get))}
        lines = [["supplied", _lt_id(rm.supplied_lt_params(base)), supplied, sup, sup_mean, ""]]
        for i, (full, vals) in enumerate(points):
            kind = "centre" if full["kernel_size"] == centre_k else "corner"
            lines.append([kind, _lt_id(full), full, vals, means.get(i), ranks.get(i, "")])
        for kind, cid, full, vals, m, rank in lines:
            delta = (m - sup_mean) / sup_mean * 100 if m is not None and sup_mean else None
            table.append([ds, bb, kind, cid, full["s_norm"], full.get("kernel_size", ""), full["station_lr"],
                          full.get("use_mlp", ""), full.get("down_ratio", ""),
                          *[_num(vals.get(h)) for h in horizons], _num(m),
                          "" if delta is None else f"{delta:+.3f}", rank])

        corners = [(full, vals) for full, vals in points if full["kernel_size"] != centre_k]
        centres = [(full, vals) for full, vals in points if full["kernel_size"] == centre_k]
        highs = {f: max(full[f] for full, _ in corners) for f in PILOT_FACTORS}
        for h in horizons:
            if not all(vals.get(h) for _, vals in corners):
                continue
            corner_mean = mean(vals[h] for _, vals in corners)
            for term in PILOT_TERMS:
                sign = lambda full: 1 if all(full[f] == highs[f] for f in term) or (
                    len(term) == 2 and all(full[f] != highs[f] for f in term)) else -1
                plus = [vals[h] for full, vals in corners if sign(full) > 0]
                minus = [vals[h] for full, vals in corners if sign(full) < 0]
                effect = (mean(plus) - mean(minus)) / corner_mean * 100
                better = ""
                if len(term) == 1:
                    f = term[0]
                    low = min(full[f] for full, _ in corners)
                    better = highs[f] if effect < 0 else low
                effects.append([ds, bb, h, ":".join(term), f"{effect:+.3f}", better,
                                "" if noise[h] is None else f"{noise[h]:.3f}",
                                "" if noise[h] is None else abs(effect) > noise[h]])
            if all(vals.get(h) for _, vals in centres):
                curvature = (mean(vals[h] for _, vals in centres) - corner_mean) / corner_mean * 100
                effects.append([ds, bb, h, "centre_vs_corners", f"{curvature:+.3f}", "",
                                "" if noise[h] is None else f"{noise[h]:.3f}",
                                "" if noise[h] is None else abs(curvature) > noise[h]])
    _write(out / "exp5_lt_pilot.csv",
           ["Dataset", "Backbone", "Point", "CandidateID", "s_norm", "kernel_size", "station_lr",
            "use_mlp", "down_ratio", *[f"Val_h{h}_s2021" for h in horizons], "Val_mean",
            "Delta_vs_supplied_pct", "Rank"], table)
    _write(out / "exp5_lt_pilot_effects.csv",
           ["Dataset", "Backbone", "Horizon", "Term", "Effect_pct", "Better_level", "Noise_pct",
            "Exceeds_noise"], effects)


def exp6_lt_explore(out, rows, protocol, bases):
    """One row per exploratory setting and horizon (dev cases, test split)."""
    doc = rm.load_explore()
    final, explore = defaultdict(dict), defaultdict(dict)
    for r in rows:
        if r.get("Split") != "test" or not r.get("MSE"):
            continue
        if r.get("Phase") == "final":
            final[(r["Dataset"], r["Backbone"], r["UseNorm"], int(r["Horizon"]))][int(r["Seed"])] = r
        elif r.get("Phase") == "explore":
            explore[(r["Dataset"], r["Backbone"], r["CandidateID"], int(r["Horizon"]))][int(r["Seed"])] = r

    def pct(a, b):
        return "" if a is None or b is None else f"{(a / b - 1) * 100:+.2f}"

    table = []
    for case, entries in doc.get("cases", {}).items():
        ds, bb = case.split("|")
        base = rm.selected_base_config(bases, ds, bb)
        schedule = rm.explore_candidates(protocol, ds, bb, base, doc)
        for entry, (params, horizons, seeds) in zip(entries, schedule):
            cid = _lt_id(params)
            for h in horizons:
                got = explore.get((ds, bb, cid, h), {})
                done = sorted(s for s in seeds if s in got)
                metric = lambda src, key, on: (mean(float(src[s][key]) for s in on)
                                               if on and all(s in src for s in on) else None)
                mse, mae, val = (metric(got, k, done) for k in ("MSE", "MAE", "BestValMSE"))
                # references on the same seeds (all requested seeds while pending)
                ref = {m: metric(final.get((ds, bb, m, h), {}), "MSE", done or sorted(seeds))
                       for m in ("lt", "san", "ddn")}
                table.append([
                    ds, bb, entry["id"], entry.get("round", ""), json.dumps(entry["params"], sort_keys=True),
                    h, " ".join(map(str, done)), f"{len(done)}/{len(seeds)}",
                    _num(mse), _num(mae), _num(val), *[_num(ref[m]) for m in ("lt", "san", "ddn")],
                    pct(mse, ref["lt"]), pct(mse, ref["san"]), pct(mse, ref["ddn"]),
                    "" if mse is None else mse < ref["san"], "" if mse is None else mse < ref["ddn"],
                ])
    _write(out / "exp6_lt_explore.csv",
           ["Dataset", "Backbone", "Label", "Round", "Change", "Horizon", "Seeds", "Done",
            "Test_MSE", "Test_MAE", "Val_MSE", "Supplied_MSE", "SAN_MSE", "DDN_MSE",
            "vs_supplied_pct", "vs_SAN_pct", "vs_DDN_pct", "Beats_SAN", "Beats_DDN"], table)


def write_all(out, rows, tasks, protocol, bases):
    out = Path(out)
    groups = final_groups(rows)
    stage_status(out, tasks)
    exp1_lightnorm(out, groups, protocol)
    exp2_validation(out, rows, protocol, bases)
    exp3_comparison(out, groups, protocol)
    exp5_lt_pilot(out, rows, protocol, bases)
    exp6_lt_explore(out, rows, protocol, bases)

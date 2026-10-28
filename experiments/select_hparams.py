#!/usr/bin/env python3
"""Create top-2 shortlists and final normalizer locks from validation rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from run_matrix import (
    BASE_CONFIGS_PATH,
    PROTOCOL_PATH,
    canonical_hash,
    complete_method_params,
    load_json,
    lt_main_candidates,
    search_candidates,
    has_selected_base,
    lt_params,
    lt_structure_candidates,
    selected_base_config,
)


def validation_rows(raw_rows):
    rows = []
    for raw in raw_rows:
        if raw.get("Split") != "validation" or not raw.get("BestValMSE"):
            continue
        row = dict(raw)
        row["BestValMSE"] = float(row["BestValMSE"])
        row["Horizon"] = int(row["Horizon"])
        row["Seed"] = int(row["Seed"])
        rows.append(row)
    return rows


def read_validation_rows(paths):
    raw = []
    for path in paths:
        with Path(path).open(newline="") as f:
            raw.extend(csv.DictReader(f))
    return validation_rows(raw)


# Baseline normalizers tuned by search/confirm/lock.  LightNorm search cells
# (Exp5 pilot) are validation rows too but follow their own selection rule.
BASELINE_METHODS = ("revin", "san", "ddn", "fan")


def candidate_catalog(protocol, bases):
    catalog = {}
    for dataset in protocol["datasets"]:
        for backbone in ("DLinear", "iTransformer", "TimeFilter", "TimeXer"):
            if not has_selected_base(bases, dataset, backbone):  # T3 backbones cover five datasets
                continue
            base = selected_base_config(bases, dataset, backbone)
            for method in BASELINE_METHODS:
                for params in search_candidates(protocol, dataset, method, base):
                    params = complete_method_params(method, params)
                    candidate_id = canonical_hash({"method": method, "params": params})
                    catalog[(dataset, backbone, method, candidate_id)] = params
    return catalog


def aggregate(rows, stages):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("Phase") not in stages or row.get("UseNorm") not in BASELINE_METHODS:
            continue
        key = (row["Dataset"], row["Backbone"], row["UseNorm"], row["CandidateID"])
        grouped[key].append(row)
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--mode", choices=["shortlist", "lock"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol = load_json(PROTOCOL_PATH)
    bases = load_json(BASE_CONFIGS_PATH)
    output = select(read_validation_rows(args.results), args.mode, protocol, bases)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")


def select(rows, mode, protocol, bases):
    """Shortlist (top-2 on seed 2021) or lock (best mean over 2 seeds x 2 horizons)."""
    catalog = candidate_catalog(protocol, bases)
    stages = {"search"} if mode == "shortlist" else {"search", "confirm"}
    grouped = aggregate(rows, stages)

    ranked = defaultdict(list)
    expected = 2 if mode == "shortlist" else 4
    for key, items in grouped.items():
        # Validation cells of retired configurations (e.g. TimeMixer++) stay in
        # the store but are no longer candidates.
        if key not in catalog:
            continue
        # Duplicate exact cells are a protocol error rather than extra evidence.
        unique_cells = {(r["Phase"], r["Horizon"], r["Seed"]): r for r in items}
        if len(unique_cells) != expected:
            continue
        values = [r["BestValMSE"] for r in unique_cells.values()]
        ranked[key[:3]].append((sum(values) / len(values), key[3], len(values)))

    if mode == "shortlist":
        output = {"protocol_version": protocol["protocol_version"], "shortlists": {}}
        for group, candidates in sorted(ranked.items()):
            candidates.sort(key=lambda item: (item[0], item[1]))
            selected = candidates[:2]
            if len(selected) != 2:
                continue
            key_text = "|".join(group)
            output["shortlists"][key_text] = [
                {
                    "candidate_id": cid,
                    "params": catalog[(*group, cid)],
                    "screen_mean_val_mse": score,
                    "n_cells": count,
                }
                for score, cid, count in selected
            ]
    else:
        # Only the two shortlisted candidates may be locked: confirm cells of a
        # candidate shortlisted under an earlier grid stay in the store.
        shortlists = select(rows, "shortlist", protocol, bases)["shortlists"]
        output = {"protocol_version": protocol["protocol_version"], "locks": {}}
        for group, candidates in sorted(ranked.items()):
            allowed = {c["candidate_id"] for c in shortlists.get("|".join(group), [])}
            candidates = [c for c in candidates if c[1] in allowed]
            candidates.sort(key=lambda item: (item[0], item[1]))
            if not candidates:
                continue
            score, cid, count = candidates[0]
            params = dict(catalog[(*group, cid)])
            output["locks"]["|".join(group)] = params
        output["selection_rule"] = (
            "Minimum mean validation MSE across horizons 96/720 and seeds "
            "2021/2022, among the two candidates shortlisted using seed 2021."
        )

    return output


def lt_validation(raw_rows, dataset, backbone, horizons, seeds):
    """CandidateID -> {(seed, horizon): validation MSE} of LightNorm cells.

    Search/confirm cells are preferred; a supplied setting without them takes
    the validation MSE of its Exp1 final cell (same seed and training).
    """
    found = defaultdict(dict)
    for r in raw_rows:
        if (r.get("Dataset"), r.get("Backbone"), r.get("UseNorm")) != (dataset, backbone, "lt"):
            continue
        if not r.get("BestValMSE") or r.get("Phase") not in ("search", "confirm", "final"):
            continue
        key = (int(r["Seed"]), int(r["Horizon"]))
        if key[0] in seeds and key[1] in horizons and (r["Phase"] != "final" or key not in found[r["CandidateID"]]):
            found[r["CandidateID"]][key] = float(r["BestValMSE"])
    return found


def _lt_cid(params):
    return canonical_hash({"method": "lt", "params": lt_params(params)})


def _structure_search(protocol, backbone):
    """True when the s_norm x use_mlp variants join the LightNorm candidates of this backbone."""
    spec = protocol["lt_tuning"]["structure"]
    return spec.get("enabled", True) and backbone in spec["backbones"]


def _lt_mean(vals):
    return sum(vals.values()) / len(vals) if vals and all(v is not None for v in vals.values()) else None


def lt_case_scores(raw_rows, protocol, bases, dataset, backbone, seeds=(2021,)):
    """[(params, {(seed, horizon): MSE or None})] for the LightNorm candidates of one case.

    Candidates are the 6-point grid and, for the backbones in
    lt_tuning.structure, the s_norm x use_mlp variants of the best grid point
    at seed 2021 (listed once the grid is complete).
    """
    horizons = protocol["screen_horizons"]
    found = lt_validation(raw_rows, dataset, backbone, horizons, set(seeds) | {2021})
    base = selected_base_config(bases, dataset, backbone)
    values = lambda p, ss: {(sd, h): found[_lt_cid(p)].get((sd, h)) for sd in ss for h in horizons}
    grid = [lt_params(p) for p in lt_main_candidates(protocol, base)]
    out = [(p, values(p, seeds)) for p in grid]
    if _structure_search(protocol, backbone):
        best = _lt_best([(p, values(p, (2021,))) for p in grid])
        if best is not None:
            out += [(lt_params(p), values(p, seeds)) for p in lt_structure_candidates(protocol, best)]
    return out


def _lt_best(scored):
    """Lowest mean over the given cells; None while any candidate is incomplete."""
    means = [(_lt_mean(v), _lt_cid(p), p) for p, v in scored]
    if not means or any(m is None for m, _, _ in means):
        return None
    return min(means, key=lambda x: (x[0], x[1]))[2]


def _lt_cases(protocol, bases, cases):
    return cases or [(d, b) for d in protocol["datasets"] for b in ("DLinear", "iTransformer", "TimeFilter", "TimeXer")
                     if has_selected_base(bases, d, b)]


def lt_grid_best(raw_rows, protocol, bases, cases=None):
    """Best 6-point-grid setting at seed 2021: the anchor of the structure search."""
    output = {"locks": {}}
    rows = list(raw_rows)
    for dataset, backbone in _lt_cases(protocol, bases, cases):
        base = selected_base_config(bases, dataset, backbone)
        found = lt_validation(rows, dataset, backbone, protocol["screen_horizons"], {2021})
        grid = [lt_params(p) for p in lt_main_candidates(protocol, base)]
        best = _lt_best([(p, {h: found[_lt_cid(p)].get((2021, h)) for h in protocol["screen_horizons"]}) for p in grid])
        if best is not None:
            output["locks"][f"{dataset}|{backbone}|lt"] = best
    return output


def lt_shortlists(raw_rows, protocol, bases, cases=None):
    """Top-2 LightNorm candidates by seed-2021 mean validation MSE (96/720), as for SAN/DDN."""
    output = {"shortlists": {}}
    rows = list(raw_rows)
    for dataset, backbone in _lt_cases(protocol, bases, cases):
        scored = lt_case_scores(rows, protocol, bases, dataset, backbone, seeds=(2021,))
        structure = _structure_search(protocol, backbone)
        if len(scored) < (9 if structure else 6) or any(_lt_mean(v) is None for _, v in scored):
            continue
        ranked = sorted(scored, key=lambda item: (_lt_mean(item[1]), _lt_cid(item[0])))[:2]
        output["shortlists"][f"{dataset}|{backbone}|lt"] = [
            {"candidate_id": _lt_cid(p), "params": p, "screen_mean_val_mse": _lt_mean(v)} for p, v in ranked]
    return output


def lt_locks(raw_rows, protocol, bases, cases=None):
    """Lock = lowest mean validation MSE over seeds 2021/2022 x horizons 96/720 among the shortlist."""
    output = {"protocol_version": protocol["protocol_version"], "locks": {}, "selection_rule": (
        "LightNorm: the two candidates with the lowest seed-2021 mean validation MSE (horizons 96/720) "
        "among the 6-point grid kernel_size {13, 25, 49} x station_lr {1e-4, 1e-3} (plus the "
        "s_norm x use_mlp variants of the best grid point while lt_tuning.structure is enabled) are "
        "confirmed at seed 2022; the lock has the lowest mean over the four cells, as for SAN/DDN.")}
    rows = list(raw_rows)
    shortlists = lt_shortlists(rows, protocol, bases, cases)["shortlists"]
    for key, items in shortlists.items():
        dataset, backbone, _ = key.split("|")
        found = lt_validation(rows, dataset, backbone, protocol["screen_horizons"], {2021, 2022})
        cells = [(item["params"], {k: found[item["candidate_id"]].get(k)
                                   for k in ((sd, h) for sd in (2021, 2022) for h in protocol["screen_horizons"])})
                 for item in items]
        best = _lt_best(cells)
        if best is not None:
            output["locks"][key] = best
    return output


if __name__ == "__main__":
    main()

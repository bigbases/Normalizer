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
        for backbone in ("DLinear", "iTransformer", "TimeMixerPP", "TimeXer"):
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
        output = {"protocol_version": protocol["protocol_version"], "locks": {}}
        for group, candidates in sorted(ranked.items()):
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


def lt_candidate_scores(raw_rows, protocol, bases, dataset, backbone):
    """Seed-2021 validation MSE (horizons 96/720) of the 6 LightNorm grid points.

    Search cells give the five new points; the supplied point comes from its
    search cell when one exists (pilot), otherwise from its Exp1 final cell.
    Returns [(params, {horizon: value})] in grid order.
    """
    horizons = protocol["screen_horizons"]
    found = defaultdict(dict)
    for r in raw_rows:
        if (r.get("Dataset"), r.get("Backbone"), r.get("UseNorm")) != (dataset, backbone, "lt"):
            continue
        if str(r.get("Seed")) != "2021" or not r.get("BestValMSE") or r.get("Phase") not in ("search", "final"):
            continue
        h = int(r["Horizon"])
        if h in horizons and (r["Phase"] == "search" or h not in found[r["CandidateID"]]):
            found[r["CandidateID"]][h] = float(r["BestValMSE"])
    base = selected_base_config(bases, dataset, backbone)
    out = []
    for params in lt_main_candidates(protocol, base):
        cid = canonical_hash({"method": "lt", "params": complete_method_params("lt", params)})
        out.append((complete_method_params("lt", params), {h: found[cid].get(h) for h in horizons}))
    return out


def lt_locks(raw_rows, protocol, bases, cases=None):
    """Lock = lowest mean validation MSE over horizons 96/720 at seed 2021 (single seed)."""
    output = {"protocol_version": protocol["protocol_version"], "locks": {}, "selection_rule": (
        "LightNorm: minimum mean validation MSE across horizons 96/720 at seed 2021 among the "
        "6-point grid kernel_size {supplied, 49} x station_lr {1e-4, 5e-4, 1e-3}.")}
    rows = list(raw_rows)
    cases = cases or [(d, b) for d in protocol["datasets"] for b in ("DLinear", "iTransformer", "TimeMixerPP", "TimeXer")]
    for dataset, backbone in cases:
        scored = lt_candidate_scores(rows, protocol, bases, dataset, backbone)
        if not scored or any(v is None for _, vals in scored for v in vals.values()):
            continue
        best = min(scored, key=lambda item: (sum(item[1].values()) / len(item[1]),
                                             canonical_hash({"method": "lt", "params": item[0]})))
        output["locks"][f"{dataset}|{backbone}|lt"] = best[0]
    return output


if __name__ == "__main__":
    main()

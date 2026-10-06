#!/usr/bin/env python3
"""Package finished runs into the shared result store (results/store/).

    # import the outputs of run_matrix.py / run_pipeline.sh on some server
    python experiments/results_pack.py pack --from results \\
        --host elice-a100 --gpu "NVIDIA A100 80GB PCIe" --torch 2.1.0+cu121 --push

    # regenerate results/summary/* (tables, selections, progress.md)
    python experiments/results_pack.py summary --push

``pack`` reads ``journal_results.csv`` and ``result_parts/*.success.csv`` in
each ``--from`` directory, attaches provenance (host, GPU, torch build, code
revision, elapsed time from ``run_manifest.jsonl``) and writes one file per
run ID.  Already-packaged run IDs are left untouched, so packing is
idempotent and can be repeated while a run is still in progress.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import result_store as rs  # noqa: E402
import run_matrix as rm  # noqa: E402


def source_rows(directory):
    directory = Path(directory)
    rows = {}
    sources = [directory / "journal_results.csv", *sorted((directory / "result_parts").glob("*.success.csv"))]
    for path in sources:
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("RunID"):
                    rows.setdefault(row["RunID"], row)
    return rows


def manifest_info(directory):
    info = {}
    path = Path(directory) / "run_manifest.jsonl"
    if path.exists():
        for line in path.open():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "finished" and record.get("returncode") == 0:
                info[record["run_id"]] = record
    return info


def pack(args, root):
    added = 0
    for directory in args.sources:
        manifest = manifest_info(directory)
        for run_id, row in source_rows(directory).items():
            record = manifest.get(run_id, {})
            row = {
                **row,
                "Host": args.host, "Worker": args.worker or args.host, "GPU": args.gpu,
                "Torch": args.torch, "CodeRev": args.code_rev,
                "ElapsedSeconds": record.get("elapsed_seconds", ""), "FinishedAt": record.get("at", ""),
            }
            added += rs.write_cell(root, row)
    return added


def summarize(root):
    protocol = rm.load_json(rm.PROTOCOL_PATH)
    bases = rm.load_json(rm.BASE_CONFIGS_PATH)
    results = rs.read_store(root)
    tasks = rs.build_tasks(rs.load_plan(), protocol, bases, results, rs.read_claims(root))
    rs.write_summary(root, results, tasks, protocol, bases)
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack", help="import run outputs into results/store")
    p.add_argument("--from", dest="sources", nargs="+", required=True,
                   help="directories holding journal_results.csv / result_parts / run_manifest.jsonl")
    p.add_argument("--host", required=True, help="server label recorded with every cell")
    p.add_argument("--gpu", required=True, help='GPU model, e.g. "NVIDIA A100 80GB PCIe"')
    p.add_argument("--torch", required=True, help="torch build, e.g. 2.1.0+cu121")
    p.add_argument("--worker", help="worker label (defaults to --host)")
    p.add_argument("--code-rev", help="code revision the runs used (default: last non-results commit)")
    p.add_argument("--push", action="store_true", help="commit and push results/ to origin")
    s = sub.add_parser("summary", help="regenerate results/summary/*")
    s.add_argument("--push", action="store_true")
    e = sub.add_parser("export", help="write every packaged cell into one CSV")
    e.add_argument("--out", default="results/all_cells.csv")
    args = parser.parse_args()

    if args.command == "export":
        rs.GitRepo(rs.ROOT).sync(read_only=True)
        results = rs.read_store(rs.ROOT)
        rs.export_cells(results, args.out)
        print(f"exported {len(results)} cells -> {args.out}")
        return

    repo = rs.GitRepo(rs.ROOT, enabled=args.push)
    if args.push:
        repo.assert_clean()
    if args.command == "pack":
        args.code_rev = args.code_rev or rs.GitRepo(rs.ROOT).code_rev()

        def mutate():
            added = pack(args, rs.ROOT)
            total = summarize(rs.ROOT)
            return added, total

        added, total = repo.transaction(mutate, f"results: pack {args.host} [skip ci]")
        print(f"packed_new_cells={added} store_cells={total}")
    else:
        total = repo.transaction(lambda: summarize(rs.ROOT), "results: summary [skip ci]")
        print(f"summary_cells={total} -> {rs.SUMMARY}")


if __name__ == "__main__":
    main()

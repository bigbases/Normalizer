# LightNorm journal experiment runtime

This directory is a writable journal-experiment copy of the supplied
`LightNorm.zip`. The original archive and all project `sources/` files remain
untouched.

## What is enforced

- Final seeds are exactly `2021, 2022, 2023`; each seed has a distinct setting,
  checkpoint directory, run ID, and configuration hash.
- Supplied dataset–backbone settings are the backbone defaults. The original
  `label_len` is honored instead of being overwritten by `seq_len // 2`.
- Hyperparameter search and confirmation use validation only. The test loader
  is not instantiated during training unless the explicit legacy/debug flag is
  supplied. The test split is evaluated once after a final configuration is
  locked.
- Normalizer parameters use `station_lr`, while backbone parameters use
  `learning_rate`.
- A completed cell is skipped only when its run ID is already in the result
  file, or a manually audited legacy ledger has `Verified=true` and the exact
  effective configuration hash.
- TimeMixer++ internal normalization is disabled so that it does not compete
  with or stack on top of the external normalization module under comparison.

## Four experiment matrices

1. `1_rebuttal_completion`: LightNorm-only missing cells on DLinear and
   iTransformer, with exact-cell reuse.
2. `2_normalizer_search`: validation-only tuning of RevIN, SAN, DDN, and FAN,
   separately for every dataset–backbone pair. The screen uses seed 2021 at
   horizons 96/720; the top two candidates are confirmed with seed 2022.
3. `3_frozen_backbone_comparison`: NoNorm, RevIN, SAN, DDN, FAN, and LightNorm
   with a fixed DLinear/iTransformer backbone and three matched seeds.
4. `4_timemixerpp_generalization`: NoNorm, FAN, and LightNorm on TimeMixer++
   with three matched seeds.

Each baseline gets at most six validation candidates. FAN's auxiliary-loss
weight remains 1.0, matching the authors' implementation. Its first `K` value
is the authors' dataset recommendation where available; neighboring values are
screened rather than selected on test performance. ETTh2/ETTm2 use the matching
ETT-family value only as a grid center, not as a claimed author recommendation.

Grid revisions (2026-10-07, recorded in `configs/protocol.json`):

- DDN: `hkernel_len` only acts when the wavelet level `j > 0`, so the original
  `j=0 × hkernel_len {3,5}` grid contained three distinct settings and never
  used DDN's frequency branch. The grid is now `kernel_len {7,25,49} × j {0,1}`
  with `hkernel_len=5` (official default; the official scripts use `j=1` on
  ETTh1/ETTm1/Weather/Electricity/Traffic).
- FAN: the reference code trains the frequency predictor with the backbone
  optimizer and learning rate. Candidates are `K × {station_lr, backbone
  learning_rate}` (`fan_lr_grid=station_and_backbone`); if both rates are
  equal, `K × {0.5, 1.0} × station_lr`.
- TimeMixer++ only needs a FAN lock (Exp4 reports NoNorm/FAN/LightNorm), so
  `scripts/run_pipeline.sh` searches FAN only on that backbone.

## Several servers at once

`main` doubles as the result store: `results/store/` (one file per finished
cell), `results/claims/` (task leases), `results/summary/` (generated).
`experiments/fill_missing.py list` shows unfinished tasks;
`scripts/start_worker.sh --data-root DIR [--gpus 0] [--datasets ...]` claims
and runs them, pushing results as they finish. `experiments/results_pack.py`
packs outputs of `run_matrix.py` into the store. See README.md §0.

## Elice Cloud runbook (A100 80GB)

Instance: `G-NAHP-160` (2 × A100 80GB PCIe, 32 vCPU, 384 GiB, ₩5,000/h), or
`G-NAHP-80` (1 × A100, ₩2,500/h) for the same total cost at about twice the
wall time. Block storage: 100 GB is enough (datasets 0.3 GB, Python env
≤ 8 GB, final checkpoints ≈ 12 GB; search/confirm checkpoints are deleted
after each run).

Image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (Python 3.10, torch
2.1.0, CUDA 12.1; same torch as the original environment). `docker/Dockerfile`
bakes in everything below. With a preset CUDA runtime instead, the setup script
builds `.venv` (Python 3.10 + torch 2.1.0 cu121) with `uv`. Host driver ≥ 525.

```bash
bash scripts/setup_container.sh --data-root ~/datasets   # apt, pip, GPU check, tests, data
tmux new -s lightnorm
bash scripts/run_pipeline.sh --data-root ~/datasets --dry-run   # plan + estimates
bash scripts/run_pipeline.sh --data-root ~/datasets              # phases 1,2,3
```

- apt: `git curl wget ca-certificates unzip rsync tmux htop procps less vim-tiny`
- pip: `requirements.txt`; TimeMixer++ only: `pip install --no-deps -r requirements-timemixerpp.txt`
- Phases run in the order they can be completed: Exp1 → Exp2 search/confirm/lock
  (DLinear, iTransformer) → Exp3. TimeMixer++ (phase 4) is off until its scope
  is decided: as configured (`channel_independence=1`) Weather/Electricity/Traffic
  exceed 80 GB per run; see `configs/cost_profile.json`.
- Inside each phase, dataset–backbone cases run shortest estimated duration
  first (`--order fastest-first`), so complete cases arrive early. Estimates
  come from profiled FLOPs/activations and are replaced by measured runtimes
  from `results/run_manifest.jsonl` on the next invocation.
- Re-running the pipeline resumes; finished run IDs are skipped.
- Monitor: `tail -f results/pipeline.log`, `watch -n 5 nvidia-smi`,
  `ls results/logs`. Stop the instance when the pipeline finishes.

## Workflow

All commands default to a dry run. Add `--execute` only after inspecting the
printed matrix. Replace `/path/to/datasets` with the directory containing
`ETT-small/`, `weather/`, `electricity/`, and `traffic/`.

```bash
# 1) Screen all module settings without test access.
python experiments/run_matrix.py \
  --experiment 2_normalizer_search --stage search \
  --data-root /path/to/datasets --execute

# 2) Select two candidates per dataset/backbone/module.
python experiments/select_hparams.py \
  --mode shortlist --results results/journal_results.csv \
  --output configs/normalizer_shortlist.json

# 3) Confirm the shortlists with seed 2022, still without test access.
python experiments/run_matrix.py \
  --experiment 2_normalizer_search --stage confirm \
  --shortlist configs/normalizer_shortlist.json \
  --data-root /path/to/datasets --execute

# 4) Lock one module setting using the declared validation rule.
python experiments/select_hparams.py \
  --mode lock --results results/journal_results.csv \
  --output configs/locked_normalizers.json

# 5) Run the matched-seed fixed-backbone comparison.
python experiments/run_matrix.py \
  --experiment 3_frozen_backbone_comparison \
  --locks configs/locked_normalizers.json \
  --data-root /path/to/datasets --execute

# 6) Run the recent-backbone table after approving the source choice below.
python experiments/run_matrix.py \
  --experiment 4_timemixerpp_generalization \
  --locks configs/locked_normalizers.json \
  --data-root /path/to/datasets --execute
```

Use `--datasets ETTh1,Weather`, `--backbones DLinear`, `--methods fan`, and
`--limit 2` for a controlled subset or smoke test. Runs stop on the first
failed cell and append start/finish events to `results/run_manifest.jsonl`.

## Resource-aware multi-GPU execution

Every worker is restricted to exactly one physical GPU through
`CUDA_VISIBLE_DEVICES`; it always sees that device as logical `cuda:0`.
Choose either exact GPU indices or a count:

```bash
# Exact devices; admit at most four workers per GPU.
python experiments/run_matrix.py \
  --experiment 1_rebuttal_completion --data-root /path/to/datasets \
  --gpus 0,1,2,3 --max-processes-per-gpu 4 --execute

# Or use the first two devices reported by nvidia-smi.
python experiments/run_matrix.py \
  --experiment 1_rebuttal_completion --data-root /path/to/datasets \
  --gpu-count 2 --max-processes-per-gpu 4 --execute
```

Before each launch the scheduler reads live free memory and GPU utilization
from `nvidia-smi`, combines them with the estimate in
`configs/resource_profiles.json`, and chooses the GPU with the best remaining
capacity. Traffic and FEDformer are exclusive by default. Lightweight jobs can
share a GPU, up to the configured 1–4 worker cap. An OOM is retried once on an
otherwise idle GPU with a 1.5× memory estimate. Parallel workers write separate
result files and logs, which the scheduler merges after successful completion.

Resource estimates are admission-control priors, not measured peak-memory
claims for the paper. Adjust the profile conservatively after inspecting the
first runs on the target hardware.

## Discord completion notifications

Fill the already-created, Git-ignored `.env` (or copy `.env.example` over it)
without committing the secret:

```text
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

`--notify-every-cases N` sends a notification after every N completed
dataset–backbone units, plus a final remainder notification. The default is 1.
For an intentionally notification-free local smoke test, explicitly pass
`--notify-every-cases 0`. Webhook values are never printed or written to the
manifest.

## Decisions that must be locked before final runs

### TimeMixer++ source

The official anonymous archive linked by the paper returned HTTP 401 on
2026-10-07. `models/TimeMixerPP.py` is therefore an optional adapter to PyPOTS
1.5 commit `53b3eac34be9491ac3f28e65ee1993436e9318af`, whose documentation says its
BSD-3-Clause implementation is inspired by the official code. Install it with
`requirements-timemixerpp.txt` only if this fallback is approved. If the
official authors' source is supplied, replace the adapter and keep the same
external interface and locked protocol.

Two properties of this fallback must be settled before Exp4:

- `BackboneTimeMixerPP.forecast()` returns the prediction of the coarsest
  scale only; its per-scale `dec_out_list` is built but never summed, unlike
  the multi-scale predictor ensemble described in the TimeMixer++ paper.
- With the supplied TimeMixer settings (`channel_independence=1`) every channel
  is a separate sequence through 2-D inception blocks. Profiled per sample:
  ETTh1 229 GFLOP / 0.6 GiB activations, Weather 1.2 TFLOP / 2.6 GiB, so
  Weather (batch 32), Electricity and Traffic do not fit on an 80 GB GPU.
  With `channel_independence=0` all datasets fit (≤ 7 GiB per batch).

Here “BSD implementation” refers to the code license of the PyPOTS
reimplementation, not to a distinct TimeMixer++ architecture. BSD-3-Clause
permits use, modification, and redistribution while requiring preservation of
the copyright/license notice and prohibiting implied endorsement. Because it
is a third-party reimplementation, the paper must identify it as such rather
than call it the official authors' implementation.

### Legacy seed provenance

Do not label old rebuttal scores as seeds 2021/2022/2023 merely because they are
plausible. For reuse, audit the old command/configuration and fill
`configs/completed_cells.template.csv`; only `Verified=true` with the exact
planner hash suppresses rerunning a cell. Otherwise keep the old score as
`seed unrecorded` evidence and rerun the matched-seed final table.

## Attribution

`normalizers/FAN.py` adapts the Apache-2.0 implementation from
`wayne155/FAN` commit `838e1b002aa0e8cbc3889dfb69967c40c0c15761`. The optional
TimeMixer++ adapter targets the BSD-3-Clause PyPOTS implementation identified
above. Preserve their licenses when redistributing code.

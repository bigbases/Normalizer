#!/usr/bin/env bash
# Run the journal matrices in the order in which they can be completed.
#
#   bash scripts/run_pipeline.sh --data-root DIR [--phases 1,2,3] [--dry-run]
#
#   phase 1  Exp1  LightNorm cells on DLinear/iTransformer (no prerequisites)
#   phase 2  Exp2  validation-only search -> shortlist -> confirm -> lock
#                  for DLinear/iTransformer (prerequisite of Exp3)
#   phase 3  Exp3  NoNorm/RevIN/SAN/DDN/FAN/LightNorm, three seeds
#                  (Exp1 cells are reused by exact run ID, not rerun)
#   phase 4  TimeMixer++: Exp2 search/confirm/lock for FAN only (Exp4 reports
#                  NoNorm/FAN/LightNorm, so RevIN/SAN/DDN locks would be
#                  unused) + Exp4 on $TMPP_DATASETS.  Off by default: needs
#                  --with-timemixerpp at setup and an approved scope.
#
# Inside every phase, run_matrix.py starts the dataset-backbone cases with
# the shortest estimated duration first.  Re-running the script resumes:
# finished run IDs are skipped.  Run it inside tmux so it survives SSH drops.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="$(cd "$ROOT/.." && pwd)/datasets"
PHASES="1,2,3"
DRY_RUN=0
TMPP_DATASETS="${TMPP_DATASETS:-ETTh1,ETTh2}"
MAX_PER_GPU="${MAX_PER_GPU:-4}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift ;;
    --phases) PHASES="$2"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

PYTHON=python3
[[ -f "$ROOT/.python-env" ]] && source "$ROOT/.python-env"
mkdir -p results configs
LOG="results/pipeline.log"

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
(( GPU_COUNT > 0 )) || { echo "no GPU visible" >&2; exit 1; }

NOTIFY=0
if grep -Eq '^DISCORD_WEBHOOK_URL=.+' .env 2>/dev/null || [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
  NOTIFY=1
fi

COMMON=(--data-root "$DATA_ROOT" --gpu-count "$GPU_COUNT" --max-processes-per-gpu "$MAX_PER_GPU"
        --resource-profiles configs/resource_profiles.a100_80gb.json
        --order fastest-first --notify-every-cases "$NOTIFY")
if (( DRY_RUN )); then COMMON+=(--summary); else COMMON+=(--execute); fi

LOCKS=configs/locked_normalizers.json
SHORTLIST=configs/normalizer_shortlist.json
RESULTS=results/journal_results.csv

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
matrix() { say "run_matrix $*"; "$PYTHON" experiments/run_matrix.py "$@" "${COMMON[@]}" 2>&1 | tee -a "$LOG"; }
select_hp() { say "select_hparams $*"; "$PYTHON" experiments/select_hparams.py --results "$RESULTS" "$@" 2>&1 | tee -a "$LOG"; }

has_phase() { [[ ",$PHASES," == *",$1,"* ]]; }

search_confirm_lock() {   # $1 = backbones, $2 = optional datasets, $3 = optional methods
  local filter=(--backbones "$1")
  [[ -n "${2:-}" ]] && filter+=(--datasets "$2")
  [[ -n "${3:-}" ]] && filter+=(--methods "$3")
  matrix --experiment 2_normalizer_search --stage search "${filter[@]}"
  if (( DRY_RUN )); then say "(dry run: confirm/lock need search results)"; return; fi
  select_hp --mode shortlist --output "$SHORTLIST"
  matrix --experiment 2_normalizer_search --stage confirm --shortlist "$SHORTLIST" "${filter[@]}"
  select_hp --mode lock --output "$LOCKS"
}

say "pipeline start phases=$PHASES gpus=$GPU_COUNT data=$DATA_ROOT python=$PYTHON"

if has_phase 1; then
  matrix --experiment 1_rebuttal_completion
fi

if has_phase 2; then
  search_confirm_lock DLinear,iTransformer
fi

if has_phase 3; then
  if (( DRY_RUN )) && [[ ! -f "$LOCKS" ]]; then
    matrix --experiment 3_frozen_backbone_comparison --methods none,lt
  else
    matrix --experiment 3_frozen_backbone_comparison --locks "$LOCKS"
  fi
fi

if has_phase 4; then
  "$PYTHON" -c "import pypots" 2>/dev/null || {
    say "phase 4 needs: bash scripts/setup_container.sh --with-timemixerpp"; exit 1; }
  search_confirm_lock TimeMixerPP "$TMPP_DATASETS" fan
  if (( DRY_RUN )) && [[ ! -f "$LOCKS" ]]; then
    matrix --experiment 4_timemixerpp_generalization --datasets "$TMPP_DATASETS" --methods none,lt
  else
    matrix --experiment 4_timemixerpp_generalization --datasets "$TMPP_DATASETS" --locks "$LOCKS"
  fi
fi

say "pipeline finished phases=$PHASES"

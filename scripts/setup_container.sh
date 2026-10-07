#!/usr/bin/env bash
# One-time setup inside the rented GPU container (idempotent; safe to re-run).
#
#   bash scripts/setup_container.sh [--with-timemixerpp] [--data-root DIR] [--skip-data]
#                                   [--skip-apt] [--use-system-python]
#                                   [--torch VERSION --torch-cuda TAG]
#
# GPUs newer than torch 2.1.0 supports (e.g. Blackwell, sm_100/sm_120) need
# --torch 2.8.0 --torch-cuda cu128; every packaged cell records its torch build.
#
# 1. apt tools (when apt-get and root/sudo are available)
# 2. Python 3.10 + torch 2.1.0 (cu121, or cu118 on drivers < 525) in an isolated
#    .venv built with uv, so an existing conda/system environment is never
#    modified (--use-system-python reuses an interpreter that already has
#    torch 2.1.0, e.g. inside docker/Dockerfile)
# 3. pip requirements (+ PyPOTS TimeMixer++ adapter with --with-timemixerpp)
# 4. GPU / import / unit-test verification
# 5. checksum-verified datasets
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$(cd "$ROOT/.." && pwd)/datasets"
WITH_TMPP=0
SKIP_DATA=0
SKIP_APT=0
USE_SYSTEM=0
TORCH_VERSION=2.1.0
TORCH_CUDA_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-timemixerpp) WITH_TMPP=1 ;;
    --data-root) DATA_ROOT="$2"; shift ;;
    --skip-data) SKIP_DATA=1 ;;
    --skip-apt) SKIP_APT=1 ;;
    --use-system-python) USE_SYSTEM=1 ;;
    --torch) TORCH_VERSION="$2"; shift ;;
    --torch-cuda) TORCH_CUDA_OVERRIDE="$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

log() { printf '\n==> %s\n' "$*"; }

log "GPU and driver"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
# torch wheels bundle their own CUDA runtime; only the host driver matters.
# cu121 needs driver >= 525, cu118 needs >= 450.  Same torch 2.1.0 either way.
if (( DRIVER_MAJOR >= 525 )); then
  TORCH_CUDA=cu121
elif (( DRIVER_MAJOR >= 450 )); then
  TORCH_CUDA=cu118
else
  echo "NVIDIA driver $DRIVER_MAJOR is too old for torch 2.1.0 (need >= 450)." >&2
  exit 1
fi
[[ -n "$TORCH_CUDA_OVERRIDE" ]] && TORCH_CUDA="$TORCH_CUDA_OVERRIDE"
echo "torch build: $TORCH_VERSION+$TORCH_CUDA"

log "apt packages"
if (( SKIP_APT )); then
  echo "skipped (--skip-apt)"
elif command -v apt-get >/dev/null; then
  SUDO=""
  if [[ $(id -u) -ne 0 ]]; then SUDO="$(command -v sudo || true)"; fi
  if [[ $(id -u) -eq 0 || -n "$SUDO" ]]; then
    $SUDO apt-get update -qq
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      git curl wget ca-certificates unzip rsync tmux htop procps less vim-tiny >/dev/null
  else
    echo "no root/sudo: skipping apt (tmux/curl must already exist)"
  fi
fi

log "Python environment"
has_torch210() {
  "$1" - <<'EOF' 2>/dev/null
import sys, torch
ok = sys.version_info[:2] == (3, 10) and torch.__version__.startswith("2.1.0") and torch.cuda.is_available()
sys.exit(0 if ok else 1)
EOF
}
PYTHON=""
if (( USE_SYSTEM )); then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null && has_torch210 "$(command -v "$candidate")"; then
      PYTHON="$(command -v "$candidate")"; break
    fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  [[ -x "$ROOT/.venv/bin/python" ]] || uv venv -p 3.10 "$ROOT/.venv"
  PYTHON="$ROOT/.venv/bin/python"
  uv pip install -p "$PYTHON" "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
  PIP=(uv pip install -p "$PYTHON")
else
  PIP=("$PYTHON" -m pip install)
fi
echo "PYTHON=$PYTHON" > "$ROOT/.python-env"
echo "using $PYTHON"

log "pip requirements"
"${PIP[@]}" -r "$ROOT/requirements.txt"
if (( WITH_TMPP )); then
  "${PIP[@]}" --no-deps -r "$ROOT/requirements-timemixerpp.txt"
fi

log "verification"
"$PYTHON" - <<'EOF'
import os, torch, numpy, pandas, statsmodels, pytorch_wavelets
assert torch.cuda.is_available(), "CUDA is not visible to torch"
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
print(f"torch {torch.__version__} cuda {torch.version.cuda} cudnn {torch.backends.cudnn.version()} "
      f"devices {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
print(f"numpy {numpy.__version__} pandas {pandas.__version__} statsmodels {statsmodels.__version__}")
print(f"cpus visible {len(os.sched_getaffinity(0))}")
EOF
(cd "$ROOT" && "$PYTHON" -m unittest tests.test_fan tests.test_protocol tests.test_gpu_scheduler tests.test_ordering tests.test_result_store)
if (( WITH_TMPP )); then
  "$PYTHON" -c "from pypots.nn.modules.timemixerpp import BackboneTimeMixerPP; print('PyPOTS TimeMixer++ import ok')"
fi

if (( ! SKIP_DATA )); then
  log "datasets -> $DATA_ROOT"
  bash "$ROOT/scripts/download_datasets.sh" "$DATA_ROOT"
fi

log "done"
echo "Next: tmux new -s lightnorm   then   bash scripts/run_pipeline.sh --data-root $DATA_ROOT"

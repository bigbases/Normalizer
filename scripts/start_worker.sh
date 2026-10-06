#!/usr/bin/env bash
# Start a fill_missing.py worker in a detached tmux session (nohup without tmux).
#
#   bash scripts/start_worker.sh --data-root DIR [fill_missing.py run options...]
#   e.g. light datasets on GPU 0 only:
#   bash scripts/start_worker.sh --data-root ~/datasets --gpus 0 \
#        --datasets ETTh1,ETTh2,ETTm1,ETTm2,Weather
#
# The worker exits with code 3 when the code on main changes; the loop then
# restarts it on the updated checkout.  Log: .worker/worker.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON=python3
[[ -f .python-env ]] && source .python-env
SESSION="${WORKER_SESSION:-lightnorm-worker}"
mkdir -p .worker
rm -f .worker/STOP          # a previous graceful stop must not stop this worker

{
  echo '#!/usr/bin/env bash'
  printf 'cd %q\n' "$ROOT"
  echo 'while true; do'
  printf '  %q experiments/fill_missing.py run' "$PYTHON"
  printf ' %q' "$@"
  echo
  echo '  rc=$?; echo "[$(date "+%F %T")] worker exited rc=$rc"'
  echo '  [ "$rc" -eq 3 ] || break; sleep 5'
  echo 'done'
} > .worker/run_loop.sh
chmod +x .worker/run_loop.sh

if command -v tmux >/dev/null; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session $SESSION is already running" >&2; exit 1
  fi
  tmux new -d -s "$SESSION" "bash .worker/run_loop.sh 2>&1 | tee -a .worker/worker.log"
  echo "started tmux session $SESSION (attach: tmux attach -t $SESSION; log: .worker/worker.log)"
else
  setsid nohup bash .worker/run_loop.sh >> .worker/worker.log 2>&1 < /dev/null &
  echo "started background worker (pid $!); log: .worker/worker.log"
fi

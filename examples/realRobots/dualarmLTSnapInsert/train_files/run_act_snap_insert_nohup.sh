#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
[[ $# -ge 1 ]] || { echo 'Specify snap or insert' >&2; exit 2; }
task=$1; shift
case "$task" in snap|insert) ;; *) exit 2;; esac
PY=${PYTHON_BIN:-$ROOT/.venv/bin/python}
count=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
[[ $count -gt 0 ]] || { echo 'GPU_NOT_READY: no visible CUDA devices; use --check/--sample for CPU checks' >&2; exit 1; }
LOG_DIR="/root/data/logs/starvla/${task}_lt_v2"
mkdir -p "$LOG_DIR"
latest_pid="$LOG_DIR/act_latest.pid"
if [[ -f $latest_pid ]] && kill -0 "$(cat "$latest_pid")" 2>/dev/null; then
  echo "Previous $task launcher is still running; inspect $latest_pid" >&2; exit 1
fi
stamp=$(date +%Y%m%d_%H%M%S)
log="$LOG_DIR/act_${stamp}.log"
nohup setsid "$HERE/train_act_snap_insert.sh" "$task" --run "$@" >"$log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$latest_pid"
ln -sfn "$(basename "$log")" "$LOG_DIR/act_latest.log"
echo "TRAINING_LAUNCH_REQUESTED task=$task pid=$launcher_pid"
echo "LOG=$log"
echo "FOLLOW: tail -F $LOG_DIR/act_latest.log"
echo "STOP: kill -- -$launcher_pid"

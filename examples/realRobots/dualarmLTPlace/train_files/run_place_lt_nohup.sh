#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 1 ]] || { echo 'Specify act or qwenpi' >&2; exit 2; }
model=$1; shift
case "$model" in act|qwenpi) ;; *) exit 2 ;; esac
ROOT=$(cd "$HERE/../../../.." && pwd)
PY=${PYTHON_BIN:-$ROOT/.venv/bin/python}
# Report unavailable GPUs immediately, without leaving a failed background job.
count=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
[[ $count -gt 0 ]] || { echo 'GPU_NOT_READY: no visible CUDA devices; use --check/--sample in CPU mode' >&2; exit 1; }
LOG_DIR=/root/data/logs/starvla/place_lt_v2
mkdir -p "$LOG_DIR"
stamp=$(date +%Y%m%d_%H%M%S)
log="$LOG_DIR/${model}_${stamp}.log"
latest_pid="$LOG_DIR/${model}_latest.pid"
if [[ -f "$latest_pid" ]] && kill -0 "$(cat "$latest_pid")" 2>/dev/null; then
  echo "A previous $model launcher is still running; inspect $latest_pid" >&2
  exit 1
fi
nohup setsid "$HERE/train_place_lt.sh" "$model" --run "$@" >"$log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" >"$latest_pid"
ln -sfn "$(basename "$log")" "$LOG_DIR/${model}_latest.log"
echo "TRAINING_LAUNCH_REQUESTED model=$model pid=$launcher_pid"
echo "LOG=$log"
echo "FOLLOW: tail -f $LOG_DIR/${model}_latest.log"
echo "STOP: kill -- -$launcher_pid"

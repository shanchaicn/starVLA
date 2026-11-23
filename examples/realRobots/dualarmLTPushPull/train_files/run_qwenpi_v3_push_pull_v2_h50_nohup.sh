#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
PY=${PYTHON_BIN:-$ROOT/.venv/bin/python}
count=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
[[ $count -gt 0 ]] || { echo 'GPU_NOT_READY: no visible CUDA devices; use --check/--sample for CPU checks' >&2; exit 1; }
LOG_DIR=/root/data/logs/starvla/push_pull_lt_v2
mkdir -p "$LOG_DIR"
latest_pid="$LOG_DIR/qwenpi_latest.pid"
if [[ -f $latest_pid ]] && kill -0 "$(cat "$latest_pid")" 2>/dev/null; then
  echo "Previous push/pull launcher is still running; inspect $latest_pid" >&2; exit 1
fi
stamp=$(date +%Y%m%d_%H%M%S)
log="$LOG_DIR/qwenpi_${stamp}.log"
nohup setsid "$HERE/train_qwenpi_v3_push_pull_v2_h50.sh" --run "$@" >"$log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$latest_pid"
ln -sfn "$(basename "$log")" "$LOG_DIR/qwenpi_latest.log"
echo "TRAINING_LAUNCH_REQUESTED model=qwenpi_v3_v2 pid=$launcher_pid"
echo "LOG=$log"
echo "FOLLOW: tail -F $LOG_DIR/qwenpi_latest.log"
echo "STOP: kill -- -$launcher_pid"

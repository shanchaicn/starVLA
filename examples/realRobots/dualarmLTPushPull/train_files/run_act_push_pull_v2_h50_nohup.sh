#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
[[ $# -ge 1 ]] || { echo "Usage: $0 {push|pull} [training options]" >&2; exit 2; }
task=$1; shift
case "$task" in push|pull) ;; *) echo "Specify push or pull" >&2; exit 2;; esac
PY=${PYTHON_BIN:-$ROOT/.venv/bin/python}
gpu_count=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
[[ $gpu_count -gt 0 ]] || { echo 'GPU_NOT_READY: no visible CUDA devices; use the train script --check/--sample first' >&2; exit 1; }
LOG_DIR="/root/data/logs/starvla/${task}_lt_v2"
mkdir -p "$LOG_DIR"
latest_pid="$LOG_DIR/act_latest.pid"
if [[ -f $latest_pid ]] && kill -0 "$(cat "$latest_pid")" 2>/dev/null; then
  echo "Previous $task ACT launcher is still running; inspect $latest_pid" >&2
  exit 1
fi
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
log="$LOG_DIR/act_${stamp}.log"
nohup setsid "$HERE/train_act_push_pull_v2_h50.sh" "$task" --run "$@" >"$log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$latest_pid"
ln -sfn "$(basename "$log")" "$LOG_DIR/act_latest.log"
echo "TRAINING_LAUNCH_REQUESTED task=$task model=act_v2 pid=$launcher_pid"
echo "LOG=$log"
echo "FOLLOW: tail -F $LOG_DIR/act_latest.log"
echo "STOP: kill -- -$launcher_pid"

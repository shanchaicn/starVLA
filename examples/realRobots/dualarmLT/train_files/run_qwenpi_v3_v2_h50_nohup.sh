#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
TRAIN="$ROOT/examples/realRobots/dualarmLT/train_files/train_qwenpi_v3_v2_h50.sh"
LOG_DIR=/root/data/outputs/starVLA/dualarm_lt_suck_qwenpi_v3_v2_h50/logs

mkdir -p "$LOG_DIR"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$LOG_DIR/qwenpi_v3_v2_h50_${timestamp}.log"
pid_file="$LOG_DIR/qwenpi_v3_v2_h50_${timestamp}.pid"
latest_log="$LOG_DIR/qwenpi_v3_v2_h50_latest.log"
latest_pid="$LOG_DIR/qwenpi_v3_v2_h50_latest.pid"

nohup setsid "$TRAIN" --run "$@" >"$log_file" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" >"$pid_file"
printf '%s\n' "$launcher_pid" >"$latest_pid"
ln -sfn "$(basename "$log_file")" "$latest_log"

echo "TRAINING_STARTED model=qwenpi_v3_v2_h50 pid=$launcher_pid"
echo "LOG=$log_file"
echo "FOLLOW: tail -f $latest_log"
echo "STOP: kill -- -$launcher_pid"

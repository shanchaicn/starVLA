#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/data/starVLA
TRAIN="$ROOT/examples/realRobots/TB6R5PPWhiteRed/train_files/train_h50.sh"
LOG_DIR=/root/data/outputs/starVLA/pp_white_red_h264_h50/logs

mkdir -p "$LOG_DIR"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$LOG_DIR/qwenpi_v3_vlact_fft_h264_${timestamp}.log"
pid_file="$LOG_DIR/qwenpi_v3_vlact_fft_h264_${timestamp}.pid"
latest_log="$LOG_DIR/qwenpi_v3_vlact_fft_h264_latest.log"
latest_pid="$LOG_DIR/qwenpi_v3_vlact_fft_h264_latest.pid"

nohup setsid "$TRAIN" qwenpi_v3_vlact_fft_h264 --run "$@" >"$log_file" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" >"$pid_file"
printf '%s\n' "$launcher_pid" >"$latest_pid"
ln -sfn "$(basename "$log_file")" "$latest_log"

echo "TRAINING_STARTED model=qwenpi_v3_vlact_fft_h264 pid=$launcher_pid"
echo "LOG=$log_file"
echo "FOLLOW: tail -f $latest_log"
echo "STOP: kill -- -$launcher_pid"

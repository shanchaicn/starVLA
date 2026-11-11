#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/data/starVLA
TRAIN="$ROOT/examples/realRobots/TB6R5PPWhiteRed/train_files/train_h50.sh"
LOG_DIR=/root/data/outputs/starVLA/pp_white_red_h50/logs

usage() {
  cat <<'EOF'
Usage: run_h50_nohup.sh {act|qwengroot|qwenpi_v3} [train_h50.sh options]

Examples:
  CUDA_VISIBLE_DEVICES=0,1 run_h50_nohup.sh qwengroot
  CUDA_VISIBLE_DEVICES=0,1 run_h50_nohup.sh qwenpi_v3
  CUDA_VISIBLE_DEVICES=0 run_h50_nohup.sh qwengroot --num-processes 1 --steps 100
EOF
}

[[ $# -ge 1 ]] || { usage; exit 2; }
model=$1
shift
case "$model" in
  act|qwengroot|qwenpi_v3) ;;
  *) usage; exit 2 ;;
esac

[[ -x "$TRAIN" ]] || { echo "Training launcher is not executable: $TRAIN" >&2; exit 1; }
mkdir -p "$LOG_DIR"

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$LOG_DIR/${model}_${timestamp}.log"
pid_file="$LOG_DIR/${model}_${timestamp}.pid"
latest_log="$LOG_DIR/${model}_latest.log"
latest_pid="$LOG_DIR/${model}_latest.pid"

nohup setsid "$TRAIN" "$model" --run "$@" >"$log_file" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" >"$pid_file"
printf '%s\n' "$launcher_pid" >"$latest_pid"
ln -sfn "$(basename "$log_file")" "$latest_log"

echo "TRAINING_STARTED model=$model pid=$launcher_pid"
echo "LOG=$log_file"
echo "FOLLOW: tail -f $latest_log"
echo "STOP: kill -- -$launcher_pid"

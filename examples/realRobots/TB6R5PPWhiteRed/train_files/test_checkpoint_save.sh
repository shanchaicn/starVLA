#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/data/starVLA
HERE="$ROOT/examples/realRobots/TB6R5PPWhiteRed/train_files"
PY="$ROOT/.venv/bin/python"
ACCELERATE="$ROOT/.venv/bin/accelerate"
VERIFY="$HERE/verify_checkpoint.sh"
OUTPUT_ROOT=/root/data/outputs/starVLA/pp_white_red_h50

usage() {
  echo "Usage: CUDA_VISIBLE_DEVICES=0,1 test_checkpoint_save.sh {qwengroot|qwenpi_v3} [steps]" >&2
}

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
model=$1
steps=${2:-10}
[[ "$steps" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
case "$model" in
  qwengroot) config="$HERE/qwengroot_h50.yaml" ;;
  qwenpi_v3) config="$HERE/qwenpi_v3_h50.yaml" ;;
  *) usage; exit 2 ;;
esac

[[ -x "$PY" && -x "$ACCELERATE" && -x "$VERIFY" ]] || {
  echo "Required StarVLA executable is missing" >&2
  exit 1
}

gpu_count=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
if [[ "$gpu_count" -ne 2 ]]; then
  echo "Checkpoint smoke test requires exactly 2 visible GPUs; found $gpu_count" >&2
  exit 1
fi

active_gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$active_gpu_pids" ]]; then
  echo "GPU compute processes are active; refusing to compete with the running training job:" >&2
  echo "$active_gpu_pids" >&2
  exit 1
fi

available_kib=$(df -Pk "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
required_kib=$((25 * 1024 * 1024))
if [[ "$available_kib" -lt "$required_kib" ]]; then
  echo "At least 25 GiB free space is required for the checkpoint smoke test" >&2
  exit 1
fi

timestamp=$(date +%Y%m%d_%H%M%S)
run_id="pp_white_red_${model}_checkpoint_smoke_${steps}steps_${timestamp}"
log_dir="$OUTPUT_ROOT/logs"
log_file="$log_dir/${run_id}.log"
mkdir -p "$log_dir"

export WANDB_MODE=disabled
cd "$ROOT"
echo "Starting checkpoint smoke test: model=$model steps=$steps run_id=$run_id"
"$ACCELERATE" launch \
  --config_file "$ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
  --num_processes 2 \
  "$ROOT/starVLA/training/train_starvla.py" \
  --config_yaml "$config" \
  --run_id "$run_id" \
  --trainer.max_train_steps "$steps" \
  --trainer.save_interval "$steps" \
  --trainer.eval_interval 999999 \
  2>&1 | tee "$log_file"

checkpoint="$OUTPUT_ROOT/$run_id/checkpoints/steps_${steps}_pytorch_model.pt"
"$VERIFY" "$checkpoint"
echo "CHECKPOINT_SMOKE_TEST_COMPLETE log=$log_file"

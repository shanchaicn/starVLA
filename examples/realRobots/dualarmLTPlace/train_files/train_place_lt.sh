#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HERE="$ROOT/examples/realRobots/dualarmLTPlace/train_files"
PY=${PYTHON_BIN:-$ROOT/.venv/bin/python}
ACCELERATE=${ACCELERATE_BIN:-$ROOT/.venv/bin/accelerate}
DATA_ROOT=${DATA_ROOT_DIR:-/root/data/datasets}
usage() { echo "Usage: $0 {act|qwenpi} {--check|--sample|--dry-run|--run} [--steps N] [--batch-size N] [--num-processes N] [--run-id NAME]" >&2; }
[[ $# -ge 2 ]] || { usage; exit 2; }
model=$1; mode=$2; shift 2
case "$model" in
  act) CONFIG="$HERE/act_place_v2_h50.yaml"; steps=80000; batch=16; processes=1 ;;
  qwenpi) CONFIG="$HERE/qwenpi_v3_place_v2_h50.yaml"; steps=20000; batch=1; processes=2 ;;
  *) usage; exit 2 ;;
esac
case "$mode" in --check|--sample|--dry-run|--run) ;; *) usage; exit 2 ;; esac
run_id=""
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || { usage; exit 2; }
  case "$1" in
    --steps) steps=$2 ;; --batch-size) batch=$2 ;; --num-processes) processes=$2 ;;
    --run-id) run_id=$2 ;; *) usage; exit 2 ;;
  esac
  shift 2
done
for value in "$steps" "$batch" "$processes"; do
  [[ $value =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
done
[[ -z $run_id || $run_id =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
[[ -x $PY && -x $ACCELERATE ]] || { echo 'Missing StarVLA Python/accelerate' >&2; exit 1; }
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$HERE/prepare_place_lt.py" --data-root "$DATA_ROOT"
checks=("$PY" "$HERE/check_place_lt.py" --config "$CONFIG" --data-root "$DATA_ROOT")
[[ $mode != --sample ]] || checks+=(--sample)
"${checks[@]}"
case "$mode" in --check|--sample) exit 0 ;; esac
cmd=("$ACCELERATE" launch --config_file "$ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml"
  --num_processes "$processes" "$ROOT/starVLA/training/train_starvla.py"
  --config_yaml "$CONFIG" --datasets.vla_data.data_root_dir "$DATA_ROOT"
  --datasets.vla_data.per_device_batch_size "$batch" --trainer.max_train_steps "$steps")
if [[ -n $run_id ]]; then
  cmd+=(--run_id "$run_id" --run_root_dir "/root/data/outputs/starVLA/$run_id" --wandb_project "$run_id")
fi
printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
[[ $mode != --dry-run ]] || exit 0
gpu_count=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
[[ $gpu_count -ge $processes ]] || { echo "GPU_NOT_READY: requested=$processes visible=$gpu_count; CPU preparation passed; load the training GPUs before --run" >&2; exit 1; }
"$PY" - "$CONFIG" "$run_id" <<'PY'
import pathlib, sys, yaml
cfg=yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
out=pathlib.Path('/root/data/outputs/starVLA')/sys.argv[2] if sys.argv[2] else pathlib.Path(cfg['run_root_dir'])
if out.exists() and any(out.rglob('*.pt')):
    raise SystemExit(f'Existing checkpoints at {out}; choose --run-id for a new experiment')
PY
export WANDB_MODE=${WANDB_MODE:-disabled}
export HF_HOME=${HF_HOME:-/root/data/huggingface}
export TORCH_HOME=${TORCH_HOME:-/root/data/torch-cache}
exec "${cmd[@]}"

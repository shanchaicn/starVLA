#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
PY=${PYTHON_BIN:-$ROOT/.venv/bin/python}
ACCELERATE=${ACCELERATE_BIN:-$ROOT/.venv/bin/accelerate}
DATA_ROOT=${DATA_ROOT_DIR:-/root/data/datasets}
usage() { echo "Usage: $0 {snap|insert} {--check|--sample|--dry-run|--run} [--steps N] [--batch-size N] [--num-processes N] [--main-process-port PORT] [--run-id NAME]" >&2; }
[[ $# -ge 2 ]] || { usage; exit 2; }
task=$1; mode=$2; shift 2
case "$task" in snap|insert) ;; *) usage; exit 2;; esac
case "$task" in snap) main_port=29603;; insert) main_port=29604;; esac
case "$mode" in --check|--sample|--dry-run|--run) ;; *) usage; exit 2;; esac
CONFIG="$HERE/act_${task}_v2_h50.yaml"
steps=80000; batch=16; processes=1; run_id=""
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || { usage; exit 2; }
  case "$1" in
    --steps) steps=$2;; --batch-size) batch=$2;; --num-processes) processes=$2;;
    --main-process-port|--main_process_port) main_port=$2;;
    --run-id) run_id=$2;; *) usage; exit 2;;
  esac
  shift 2
done
for value in "$steps" "$batch" "$processes"; do [[ $value =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }; done
[[ $main_port =~ ^(0|[1-9][0-9]*)$ ]] && (( main_port == 0 || (main_port >= 1024 && main_port <= 65535) )) || { echo 'Invalid --main-process-port: use 0 or 1024..65535' >&2; exit 2; }
[[ -z $run_id || $run_id =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
[[ -x $PY && -x $ACCELERATE ]] || { echo 'Missing StarVLA Python/accelerate' >&2; exit 1; }
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/root/data/huggingface} TORCH_HOME=${TORCH_HOME:-/root/data/torch-cache}
export TMPDIR=${TMPDIR:-/root/data/tmp} XDG_CACHE_HOME=${XDG_CACHE_HOME:-/root/data/.cache}
export WANDB_MODE=${WANDB_MODE:-disabled}
"$PY" "$HERE/prepare_snap_insert_lt.py" --task "$task" --data-root "$DATA_ROOT"
checks=("$PY" "$HERE/check_snap_insert_lt.py" --task "$task" --config "$CONFIG" --data-root "$DATA_ROOT")
[[ $mode != --sample ]] || checks+=(--sample)
"${checks[@]}"
case "$mode" in --check|--sample) exit 0;; esac
cmd=("$ACCELERATE" launch --config_file "$ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml"
  --num_processes "$processes" --main_process_port "$main_port" "$ROOT/starVLA/training/train_starvla.py"
  --config_yaml "$CONFIG" --datasets.vla_data.data_root_dir "$DATA_ROOT"
  --datasets.vla_data.per_device_batch_size "$batch" --trainer.max_train_steps "$steps")
if [[ -n $run_id ]]; then
  cmd+=(--run_id "$run_id" --run_root_dir "/root/data/outputs/starVLA/$run_id" --wandb_project "$run_id")
fi
printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
[[ $mode != --dry-run ]] || exit 0
gpu_count=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
[[ $gpu_count -ge $processes ]] || { echo "GPU_NOT_READY: requested=$processes visible=$gpu_count; CPU checks passed" >&2; exit 1; }
"$PY" - "$CONFIG" "$run_id" <<'PY'
import pathlib,sys,yaml
cfg=yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
out=pathlib.Path('/root/data/outputs/starVLA')/sys.argv[2] if sys.argv[2] else pathlib.Path(cfg['run_root_dir'])
if out.exists() and any(out.rglob('*.pt')):
    raise SystemExit(f'Existing checkpoints at {out}; choose --run-id for a new experiment')
PY
exec "${cmd[@]}"

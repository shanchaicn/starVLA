#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/data/starVLA
HERE="$ROOT/examples/realRobots/TB6R5PPWhiteRed/train_files"
PY="$ROOT/.venv/bin/python"
ACCELERATE="$ROOT/.venv/bin/accelerate"
DATA=/root/data/datasets/pp_white_red

usage() {
  cat <<'EOF'
Usage: train_h50.sh {act|qwengroot|qwenpi_v3} {--check|--dry-run|--run} [--steps N] [--num-processes N]
  --check    Validate dataset/config and print readiness.
  --dry-run  Validate and print the exact launch command.
  --run      Validate, require the requested CUDA GPUs, then start training.
  --steps N  Override max_train_steps from the selected YAML.
  --num-processes N  Number of GPU processes (default: 2).

Run one model at a time. All three configs use H=50, 8D absolute actions,
the same two camera views, and the same 251-episode dataset.
EOF
}

[[ $# -ge 2 ]] || { usage; exit 2; }
model=$1
mode=$2
shift 2
case "$model" in
  act) config="$HERE/act_h50.yaml" ;;
  qwengroot) config="$HERE/qwengroot_h50.yaml" ;;
  qwenpi_v3) config="$HERE/qwenpi_v3_h50.yaml" ;;
  *) usage; exit 2 ;;
esac
case "$mode" in --check|--dry-run|--run) ;; *) usage; exit 2 ;; esac
steps=""
num_processes=2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      [[ $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
      steps=$2
      shift 2
      ;;
    --num-processes)
      [[ $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
      num_processes=$2
      shift 2
      ;;
    *) usage; exit 2 ;;
  esac
done

[[ -x "$PY" && -x "$ACCELERATE" ]] || { echo 'Missing StarVLA .venv Python/accelerate' >&2; exit 1; }
[[ -f "$DATA/meta/modality.json" ]] || { echo 'Missing dataset meta/modality.json' >&2; exit 1; }
[[ -f "$DATA/meta/tasks.parquet" && -f "$DATA/meta/info.json" ]] || { echo 'Missing LeRobot metadata' >&2; exit 1; }
[[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }

"$PY" - "$config" "$DATA" "$model" <<'PY'
import json
import pathlib
import sys
import yaml
import pyarrow.parquet as pq

cfg_path, dataset, model = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(cfg_path.read_text())
mod = json.loads((dataset / 'meta/modality.json').read_text())
info = json.loads((dataset / 'meta/info.json').read_text())
tasks = pq.read_table(dataset / 'meta/tasks.parquet', columns=['task_index', 'task']).to_pydict()
assert len(tasks['task']) == 2, tasks
assert len(set(tasks['task_index'])) == 2, tasks
assert info['total_episodes'] == 251, info.get('total_episodes')
assert cfg['datasets']['vla_data']['data_mix'] == 'pp_white_red_abs8_h50'
assert cfg['datasets']['vla_data']['action_mode'] == 'abs'
assert cfg['trainer']['gradient_accumulation_steps'] == 1
if model in ('qwengroot', 'qwenpi_v3'):
    assert cfg['trainer']['max_train_steps'] == 20000
    assert cfg['trainer']['save_interval'] == 5000
assert cfg['framework']['action_dim'] == 8 if model == 'act' else cfg['framework']['action_model']['action_dim'] == 8
assert cfg['framework']['chunk_size'] == 50 if model == 'act' else cfg['framework']['action_model']['action_horizon'] == 50
for section in ('state', 'action'):
    assert [(mod[section][k]['start'], mod[section][k]['end']) for k in ('xyz', 'quat_xyzw', 'gripper_m')] == [(0, 3), (3, 7), (7, 8)]
assert [mod['video'][k]['original_key'] for k in ('front', 'wrist')] == ['observation.images.front', 'observation.images.wrist']
if model != 'act':
    assert pathlib.Path(cfg['framework']['qwenvl']['base_vlm']).is_dir(), 'Qwen VLM path missing'
print(f"PREFLIGHT_OK model={model} H=50 episodes={info['total_episodes']} tasks={tasks['task']}")
PY

if [[ $mode == --check ]]; then exit 0; fi

cmd=("$ACCELERATE" launch
  --config_file "$ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml"
  --num_processes "$num_processes"
  "$ROOT/starVLA/training/train_starvla.py"
  --config_yaml "$config")
if [[ -n "$steps" ]]; then
  cmd+=(--trainer.max_train_steps "$steps")
fi
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ $mode == --dry-run ]]; then exit 0; fi

gpu_count=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
if [[ "$gpu_count" -lt "$num_processes" ]]; then
  echo "Requested $num_processes GPU processes, but only $gpu_count CUDA GPUs are visible." >&2
  exit 1
fi
export WANDB_MODE="${WANDB_MODE:-disabled}"
cd "$ROOT"
exec "${cmd[@]}"

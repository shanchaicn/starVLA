#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HERE="$ROOT/examples/realRobots/dualarmLT/train_files"
PY=${PYTHON_BIN:-$ROOT/.venv/bin/python}
ACCELERATE=${ACCELERATE_BIN:-$ROOT/.venv/bin/accelerate}
CONFIG="$HERE/qwenpi_v3_v2_h50.yaml"
DATA_ROOT=${DATA_ROOT_DIR:-/root/data/datasets}
DATASET="$DATA_ROOT/suck_lt"

usage() {
  echo "Usage: $0 {--check|--dry-run|--run} [--steps N] [--batch-size N] [--num-processes N] [--run-root-dir PATH] [--run-id ID] [--learning-rate GROUP RATE]..." >&2
}

[[ $# -ge 1 ]] || { usage; exit 2; }
mode=$1
shift
case "$mode" in --check|--dry-run|--run) ;; *) usage; exit 2 ;; esac

steps=20000
batch_size=1
num_processes=2
run_root_dir=
run_id=
lr_overrides=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) steps=$2; shift 2 ;;
    --batch-size) batch_size=$2; shift 2 ;;
    --num-processes) num_processes=$2; shift 2 ;;
    --run-root-dir) run_root_dir=$2; shift 2 ;;
    --run-id) run_id=$2; shift 2 ;;
    --learning-rate)
      [[ $# -ge 3 ]] || { usage; exit 2; }
      case "$2" in
        base|qwen_vl_interface|action_model|project_layers) ;;
        *) echo "Unknown learning-rate group: $2" >&2; exit 2 ;;
      esac
      lr_overrides+=("--trainer.learning_rate.$2" "$3")
      shift 3
      ;;
    *) usage; exit 2 ;;
  esac
done
for value in "$steps" "$batch_size" "$num_processes"; do
  [[ $value =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
done

[[ -x "$PY" && -x "$ACCELERATE" ]] || { echo "Missing StarVLA .venv Python/accelerate" >&2; exit 1; }
[[ -f "$DATASET/meta/info.json" && -f "$DATASET/meta/tasks.parquet" ]] || {
  echo "Dataset missing; first run: $HERE/prepare_suck_lt.sh" >&2
  exit 1
}
[[ -f "$DATASET/meta/modality.json" ]] || { echo "Missing $DATASET/meta/modality.json; run prepare_suck_lt.sh" >&2; exit 1; }
cmp -s "$HERE/modality.json" "$DATASET/meta/modality.json" || {
  echo "Dataset modality.json does not match this training recipe" >&2
  exit 1
}

"$PY" - "$CONFIG" "$DATASET" "${lr_overrides[@]}" <<'PY'
import json
import math
import pathlib
import sys

import pyarrow.parquet as pq
import yaml

config_path, dataset = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
cfg = yaml.safe_load(config_path.read_text())
for flag, raw_value in zip(sys.argv[3::2], sys.argv[4::2]):
    try:
        rate = float(raw_value)
    except ValueError:
        raise SystemExit(f"{flag} must be a positive finite number") from None
    if not math.isfinite(rate) or rate <= 0:
        raise SystemExit(f"{flag} must be a positive finite number")
info = json.loads((dataset / "meta/info.json").read_text())
stats = json.loads((dataset / "meta/stats.json").read_text())
tasks = pq.read_table(dataset / "meta/tasks.parquet").to_pydict()

assert info["codebase_version"] == "v3.0"
assert info["fps"] == 30
assert info["total_episodes"] == 196
assert info["total_frames"] == 82827
assert tasks["task"] == ["Use the suction cup on the robot's right arm to pick up the phone support plate."]

expected = {
    "observation.state.cartesian_position": 14,
    "action.cartesian_position": 14,
    "observation.state.digital_input": 17,
    "action.digital_output": 12,
}
for key, width in expected.items():
    assert info["features"][key]["shape"] == [width], (key, info["features"][key])
for key in (
    "observation.images.left_side", "observation.images.right_wrist",
    "observation.images.right_side",
):
    feature = info["features"][key]
    assert feature["shape"] == [400, 640, 3]
    assert feature["info"]["video.codec"] == "h264"

do_stats = stats["action.digital_output"]
active = [i for i, maximum in enumerate(do_stats["max"]) if maximum > 0]
assert active == [2, 3, 4, 5], active

framework = cfg["framework"]
data = cfg["datasets"]["vla_data"]
trainer = cfg["trainer"]
assert framework["name"] == "QwenPI_v3"
action_model = framework["action_model"]
assert action_model["action_horizon"] == 50
assert action_model["state_dim"] == 24
assert action_model["action_dim"] == 11
assert data["data_mix"] == "dualarm_lt_suck_v2_h50_3cam"
assert data["per_device_batch_size"] == 1
assert data["action_mode"] == "abs"
assert data["load_all_data_for_training"] is True
assert data["video_backend"] == "pyav"
assert trainer["max_train_steps"] == 20000
assert trainer["save_interval"] == 5000
assert trainer["random_init_action_model"] is True
assert trainer["reset_steps"] is True
assert trainer["is_resume"] is False
assert trainer["freeze_modules"] == ""
assert pathlib.Path(framework["qwenvl"]["base_vlm"]).is_dir()
checkpoint = pathlib.Path(trainer["pretrained_checkpoint"])
assert checkpoint.is_file(), checkpoint
assert (checkpoint.parent.parent / "config.yaml").is_file()
assert (checkpoint.parent.parent / "dataset_statistics.json").is_file()
print(f"PREFLIGHT_OK model=qwenpi_v3_vlact_fft recipe=v2_suck episodes={info['total_episodes']} frames={info['total_frames']} H=50 state=24 action=11 cameras=3 excluded_camera=left_wrist force=false stationary_arm=false base_ignored=true")
PY

if [[ $mode == --check ]]; then exit 0; fi

cmd=("$ACCELERATE" launch
  --config_file "$ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml"
  --num_processes "$num_processes"
  "$ROOT/starVLA/training/train_starvla.py"
  --config_yaml "$CONFIG"
  --datasets.vla_data.data_root_dir "$DATA_ROOT"
  --datasets.vla_data.per_device_batch_size "$batch_size"
  --trainer.max_train_steps "$steps"
  "${lr_overrides[@]}")
if [[ -n $run_root_dir ]]; then
  cmd+=(--run_root_dir "$run_root_dir")
fi
if [[ -n $run_id ]]; then
  cmd+=(--run_id "$run_id")
fi
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ $mode == --dry-run ]]; then exit 0; fi

gpu_count=$($PY -c 'import torch; print(torch.cuda.device_count())')
[[ $gpu_count -ge $num_processes ]] || {
  echo "Requested $num_processes GPU processes, but only $gpu_count GPUs are visible" >&2
  exit 1
}
export WANDB_MODE=${WANDB_MODE:-disabled}
cd "$ROOT"
exec "${cmd[@]}"

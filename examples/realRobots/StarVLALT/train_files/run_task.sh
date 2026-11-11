#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG_YAML="${SCRIPT_DIR}/starvla_qwenpi_v3_lora.yaml"
DATA_ROOT="${DATA_ROOT:-/root/data/datasets/starvla-lt}"
BASE_VLM_ROOT="${BASE_VLM:-/root/data/models/Qwen3-VL-4B-Instruct-Action}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/root/data/models/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt}"
RUN_ROOT="${RUN_ROOT_DIR:-/root/data/outputs/starVLA/starvla-lt}"
LOG_ROOT="${LOG_ROOT_DIR:-/root/data/logs/starvla-lt}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
ACCELERATE_BIN="${REPO_ROOT}/.venv/bin/accelerate"
ACCELERATE_CONFIG="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_lora_zero1.yaml"

usage() {
  echo "Usage: $0 <push_pull|insert|snap|suck|place> <smoke|pilot|train> [--dry-run|--check-only]" >&2
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

TASK="$1"
STAGE="$2"
shift 2

MODE="execute"
for arg in "$@"; do
  case "${arg}" in
    --dry-run) MODE="dry-run" ;;
    --check-only) MODE="check-only" ;;
    *) echo "Unknown argument: ${arg}" >&2; usage; exit 2 ;;
  esac
done

case "${TASK}" in
  push_pull|insert|snap) REAL_VIEWS=5 ;;
  suck) REAL_VIEWS=4 ;;
  place) REAL_VIEWS=5 ;;
  *) echo "Unsupported task: ${TASK}" >&2; usage; exit 2 ;;
esac

case "${STAGE}" in
  smoke)
    MAX_STEPS="${MAX_TRAIN_STEPS:-5}"
    WARMUP_STEPS="${NUM_WARMUP_STEPS:-1}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
    ;;
  pilot)
    MAX_STEPS="${MAX_TRAIN_STEPS:-2000}"
    WARMUP_STEPS="${NUM_WARMUP_STEPS:-100}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
    ;;
  train)
    MAX_STEPS="${MAX_TRAIN_STEPS:-10000}"
    WARMUP_STEPS="${NUM_WARMUP_STEPS:-500}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
    ;;
  *) echo "Unsupported stage: ${STAGE}" >&2; usage; exit 2 ;;
esac

DATASET_DIR="${DATA_ROOT}/${TASK}"
INFO_JSON="${DATASET_DIR}/meta/info.json"
MODALITY_JSON="${DATASET_DIR}/meta/modality.json"

[[ -f "${CONFIG_YAML}" ]] || { echo "Missing config: ${CONFIG_YAML}" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -x "${ACCELERATE_BIN}" ]] || { echo "Missing accelerate: ${ACCELERATE_BIN}" >&2; exit 1; }
[[ -f "${INFO_JSON}" ]] || { echo "Dataset is incomplete; missing ${INFO_JSON}" >&2; exit 1; }
[[ -f "${MODALITY_JSON}" ]] || { echo "Missing ${MODALITY_JSON}" >&2; exit 1; }

"${PYTHON_BIN}" - "${DATASET_DIR}" "${REAL_VIEWS}" "${CONFIG_YAML}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
real_views = int(sys.argv[2])
config_path = Path(sys.argv[3])
info = json.loads((root / "meta/info.json").read_text())
episodes = int(info["total_episodes"])
parquet_count = sum(1 for _ in (root / "data").rglob("*.parquet"))
video_count = sum(1 for p in (root / "videos").rglob("*") if p.is_file())
expected_videos = episodes * real_views

for key, dtype, width in (
    ("observation.state.cartesian_position", "float32", 14),
    ("action.cartesian_position", "float32", 14),
    ("observation.state.vacuum", "bool", 5),
    ("action.vacuum", "bool", 5),
):
    feature = info["features"].get(key)
    if feature is None or feature.get("dtype") != dtype or feature.get("shape") != [width]:
        raise SystemExit(f"Unexpected feature: {root.name} {key}={feature!r}")

modality = json.loads((root / "meta/modality.json").read_text())
expected = {
    "state": {
        "left_pose": ("observation.state.cartesian_position", 0, 7),
        "right_pose": ("observation.state.cartesian_position", 7, 14),
        "vacuum": ("observation.state.vacuum", 0, 5),
    },
    "action": {
        "left_pose": ("action.cartesian_position", 0, 7),
        "right_pose": ("action.cartesian_position", 7, 14),
        "vacuum": ("action.vacuum", 0, 5),
    },
}
for group, keys in expected.items():
    for name, (source, start, end) in keys.items():
        entry = modality.get(group, {}).get(name, {})
        if (entry.get("original_key"), entry.get("start"), entry.get("end")) != (source, start, end):
            raise SystemExit(f"Incorrect modality: {root.name} {group}.{name}={entry!r}")
        if group == "action" and entry.get("absolute") is not True:
            raise SystemExit(f"Action must be absolute: {root.name} {name}")
        if name == "vacuum" and entry.get("dtype") != "bool":
            raise SystemExit(f"Vacuum modality must be binary: {root.name} {group}.{name}")
if len(modality.get("video", {})) != real_views:
    raise SystemExit(f"Expected {real_views} camera mappings: {root.name}")

from yaml import safe_load
model = safe_load(config_path.read_text())["framework"]["action_model"]
if model["state_dim"] != 19 or model["action_dim"] != 19:
    raise SystemExit(f"Training recipe must use 19D state/action: {config_path}")
cache = json.loads((root / "meta/stats_gr00t.json").read_text())
for key in ("observation.state.cartesian_position", "observation.state.vacuum", "action.cartesian_position", "action.vacuum"):
    if key not in cache.get("statistics", {}):
        raise SystemExit(f"Missing statistics cache key: {root.name} {key}")

if parquet_count != episodes:
    raise SystemExit(f"Parquet count mismatch: expected {episodes}, found {parquet_count}")
if video_count != expected_videos:
    raise SystemExit(f"Video count mismatch: expected {expected_videos}, found {video_count}")

progress = root / ".conversion-progress.json"
if progress.exists():
    status = json.loads(progress.read_text()).get("status")
    if status not in {"complete", "completed", "done"}:
        raise SystemExit(f"Conversion is not complete: {progress} status={status!r}")

incoming = root / ".parallel-incoming"
incoming_files = sum(1 for p in incoming.rglob("*") if p.is_file()) if incoming.exists() else 0
if incoming_files:
    raise SystemExit(f"Dataset has {incoming_files} unfinished incoming files: {incoming}")

print(
    f"dataset_ok task={root.name} episodes={episodes} parquet={parquet_count} "
    f"real_videos={video_count} model_views=5 state_dim=19 action_dim=19"
)
PY

if [[ "${MODE}" == "check-only" ]]; then
  exit 0
fi

MIXTURE="starvla_${TASK}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${TASK}_lora_${STAGE}_${STAMP}}"
if [[ -e "${RUN_ROOT}/${RUN_ID}" || -e "${LOG_ROOT}/${RUN_ID}.log" ]]; then
  echo "Run ID already exists; choose a new RUN_ID: ${RUN_ID}" >&2
  exit 1
fi
NUM_PROCESSES="${NUM_PROCESSES:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-$((MAX_STEPS + 1))}"

CMD=(
  "${ACCELERATE_BIN}" launch
  --config_file "${ACCELERATE_CONFIG}"
  --num_processes "${NUM_PROCESSES}"
  "${REPO_ROOT}/starVLA/training/train_starvla.py"
  --config_yaml "${CONFIG_YAML}"
  --framework.qwenvl.base_vlm "${BASE_VLM_ROOT}"
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  --trainer.random_init_action_model true
  --trainer.reset_steps true
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --datasets.vla_data.data_mix "${MIXTURE}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-1}"
  --trainer.max_train_steps "${MAX_STEPS}"
  --trainer.num_warmup_steps "${WARMUP_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --run_root_dir "${RUN_ROOT}"
  --run_id "${RUN_ID}"
)

if [[ "${MODE}" == "dry-run" ]]; then
  printf 'dry_run_command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  [[ -d "${BASE_VLM_ROOT}" ]] || echo "dry_run_warning: base VLM directory is not present: ${BASE_VLM_ROOT}"
  [[ -f "${PRETRAINED_CHECKPOINT}" ]] || echo "dry_run_warning: pretrained checkpoint is not present: ${PRETRAINED_CHECKPOINT}"
  exit 0
fi

[[ -d "${BASE_VLM_ROOT}" ]] || { echo "Missing base VLM directory: ${BASE_VLM_ROOT}" >&2; exit 1; }
[[ -f "${PRETRAINED_CHECKPOINT}" ]] || { echo "Missing VLAct checkpoint: ${PRETRAINED_CHECKPOINT}" >&2; exit 1; }
"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA GPU is not available; refusing to start training")
print(f"gpu_ok count={torch.cuda.device_count()}")
PY

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/data/uv/cache}"
export HF_HOME="${HF_HOME:-/root/data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/data/huggingface/hub}"
export TORCH_HOME="${TORCH_HOME:-/root/data/torch-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/data/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/data/.cache/triton}"
export TRITON_AUTOTUNE_DIR="${TRITON_AUTOTUNE_DIR:-/root/data/.cache/triton/autotune}"
export TMPDIR="${TMPDIR:-/root/data/tmp}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TRITON_AUTOTUNE_DIR}"
LOG_FILE="${LOG_ROOT}/${RUN_ID}.log"
printf 'training_command:'
printf ' %q' "${CMD[@]}"
printf '\nlog_file=%s\n' "${LOG_FILE}"

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_loss.py" \
  "${LOG_FILE}" \
  "${RUN_ROOT}/${RUN_ID}/loss_curve.svg" \
  "${RUN_ROOT}/${RUN_ID}/loss.csv"

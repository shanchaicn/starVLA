#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HERE="$ROOT/examples/realRobots/dualarmLT/train_files"
DATA_ROOT=${DATA_ROOT_DIR:-/root/data/datasets}
DATASET="$DATA_ROOT/suck_lt"

mkdir -p "$DATA_ROOT"
if [[ ! -f "$DATASET/meta/info.json" ]]; then
  if [[ -x "$ROOT/.venv/bin/hf" ]]; then
    "$ROOT/.venv/bin/hf" download shanchai/suck_lt --repo-type dataset --local-dir "$DATASET"
  elif command -v hf >/dev/null 2>&1; then
    hf download shanchai/suck_lt --repo-type dataset --local-dir "$DATASET"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download shanchai/suck_lt --repo-type dataset --local-dir "$DATASET"
  else
    echo "Install huggingface_hub or download https://huggingface.co/datasets/shanchai/suck_lt into $DATASET" >&2
    exit 1
  fi
fi

mkdir -p "$DATASET/meta"
if [[ -f "$DATASET/meta/modality.json" ]]; then
  cmp -s "$HERE/modality.json" "$DATASET/meta/modality.json" || {
    echo "Existing modality.json differs; refusing to overwrite: $DATASET/meta/modality.json" >&2
    exit 1
  }
else
  install -m 0644 "$HERE/modality.json" "$DATASET/meta/modality.json"
fi

echo "DATASET_READY=$DATASET"

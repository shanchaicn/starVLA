#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/data/starVLA
PY="$ROOT/.venv/bin/python"

if [[ $# -ne 1 ]]; then
  echo "Usage: verify_checkpoint.sh /absolute/path/to/steps_N_pytorch_model.pt" >&2
  exit 2
fi

checkpoint=$1
[[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 1; }

"$PY" - "$checkpoint" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
size = path.stat().st_size
if size < 1024 * 1024:
    raise RuntimeError(f"Checkpoint is unexpectedly small: {size} bytes")

state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
if not isinstance(state, dict) or not state:
    raise RuntimeError("Checkpoint did not contain a non-empty state dict")

tensors = {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
if not tensors:
    raise RuntimeError("Checkpoint state dict contains no tensors")

prefixes = ("qwen_vl_interface", "action_model", "project_layers")
prefix_counts = {prefix: sum(key.startswith(prefix) for key in tensors) for prefix in prefixes}
if prefix_counts["qwen_vl_interface"] == 0 or prefix_counts["action_model"] == 0:
    raise RuntimeError(f"Required model prefixes are missing: {prefix_counts}")

logical_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
print(f"CHECKPOINT_OK path={path}")
print(f"file_size_gib={size / 2**30:.2f}")
print(f"tensor_count={len(tensors)} logical_tensor_gib={logical_bytes / 2**30:.2f}")
print(f"prefix_counts={prefix_counts}")
PY

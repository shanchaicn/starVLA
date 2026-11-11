#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/data/starVLA
RUNNER="$ROOT/examples/realRobots/TB6R5PPWhiteRed/train_files/run_h50_nohup.sh"

[[ -x "$RUNNER" ]] || { echo "Missing launcher: $RUNNER" >&2; exit 1; }
exec "$RUNNER" qwenpi_v3_vlact_fft "$@"

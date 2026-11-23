#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
exec /home/test/Public/cs/venv/vla/bin/python \
  "$HERE/dualarm_lt_rtc_execute_16.py" --config "$HERE/dualarm_lt_rtc_16.yaml" \
  --profile qwenpi_v3_v2 --real "$@"

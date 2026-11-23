#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /home/codeit/Public/zpf/StarVLA/inference/.venv/bin/python \
  "$HERE/dualarm_lt_rtc_server_252.py" --profile qwenpi_v3_v3 --serve --port 10200 "$@"

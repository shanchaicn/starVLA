#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${STARVLA_SERVER_PYTHON:-$HERE/.venv/bin/python}" "$HERE/launch_server_16.py" --profile qwenpi_v3_v2 "$@"

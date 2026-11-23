#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$HERE/refresh_context.py"
exec "$HERE/compose.sh" --progress plain build qwen-v2 client "$@"

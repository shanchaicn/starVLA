#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MODELSCOPE_CACHE=/root/data/.cache/modelscope XDG_CACHE_HOME=/root/data/.cache TMPDIR=/root/data/tmp
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
exec /root/data/venv/modelscope-lt-tools/bin/python "$HERE/download_snap_insert.py" "$@"

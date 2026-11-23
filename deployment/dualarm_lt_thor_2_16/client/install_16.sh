#!/usr/bin/env bash
set -euo pipefail
VENV=/home/test/Public/cs/venv/vla
export PYTHONPATH="/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
"$VENV/bin/python" -c 'import get_status,numpy,yaml; print("VLA_DEPS_OK",numpy.__version__,yaml.__version__)'


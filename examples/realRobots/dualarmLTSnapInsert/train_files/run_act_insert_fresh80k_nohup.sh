#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
mode=${1:-}
[[ $# -le 1 && ( -z $mode || $mode == --dry-run ) ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }
TASK=insert
CONFIG="$HERE/act_${TASK}_v2_h50.yaml"
"$ROOT/.venv/bin/python" - "$CONFIG" <<'CHECK'
import sys,yaml
from pathlib import Path
c=yaml.safe_load(Path(sys.argv[1]).read_text())
t=c['trainer']
assert not t.get('is_resume',False) and t.get('pretrained_checkpoint') is None, 'Fresh training must not load an old checkpoint'
assert c['datasets']['vla_data']['per_device_batch_size']==16 and t['max_train_steps']==80000
CHECK
run_id="dualarm_lt_${TASK}_act_h50_v2_bs16_80k_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
args=(--num-processes 1 --batch-size 16 --steps 80000 --run-id "$run_id")
if [[ $mode == --dry-run ]]; then
  exec "$HERE/train_act_snap_insert.sh" "$TASK" --dry-run "${args[@]}"
fi
exec "$HERE/run_act_snap_insert_nohup.sh" "$TASK" "${args[@]}"

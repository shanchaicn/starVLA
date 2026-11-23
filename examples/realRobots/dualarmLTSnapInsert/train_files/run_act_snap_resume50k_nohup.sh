#!/usr/bin/env bash
# Restore model weights and step count; old checkpoints contain no optimizer state.
# The scheduler is advanced to the resumed step on the new 50,000-step schedule.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
PY="$ROOT/.venv/bin/python"
CONFIG="$HERE/act_snap_v2_h50_resume50k.yaml"
LOG_DIR=/root/data/logs/starvla/snap_lt_v2
dry_run=false
case "${1:-}" in "") ;; --dry-run) dry_run=true; shift;; *) echo "Usage: $0 [--dry-run]" >&2; exit 2;; esac
[[ $# -eq 0 ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/root/data/huggingface} TORCH_HOME=${TORCH_HOME:-/root/data/torch-cache}
export TMPDIR=${TMPDIR:-/root/data/tmp} XDG_CACHE_HOME=${XDG_CACHE_HOME:-/root/data/.cache}
export WANDB_MODE=${WANDB_MODE:-disabled}
"$PY" - "$CONFIG" <<'PY'
from pathlib import Path
import json,re,sys,yaml
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
assert cfg['trainer']['is_resume'] is True and cfg['trainer']['max_train_steps']==50000
assert cfg['datasets']['vla_data']['per_device_batch_size']==2
out=Path(cfg['run_root_dir'])/cfg['run_id']
rows=[]
for p in (out/'checkpoints').iterdir():
    match=re.fullmatch(r'steps_(\d+)_pytorch_model\.pt',p.name)
    if match and p.is_file() and p.stat().st_size>0:rows.append((int(match[1]),p))
if not rows:raise SystemExit('No model checkpoint found; refusing to start from zero')
step,path=max(rows)
if step<20000:raise SystemExit('Expected a checkpoint at or after step 20000')
if step>=50000:raise SystemExit(f'Already at step {step}; no additional training needed')
print(json.dumps({'resume_checkpoint':str(path),'completed_steps':step,'target_total_steps':50000,
    'additional_steps':50000-step,'batch_size_per_gpu':2,'optimizer_state_restored':False,
    'lr_schedule':'recomputed for 50000 total steps; advanced to completed_steps'},indent=2),flush=True)
PY
cmd=("$ROOT/.venv/bin/accelerate" launch
  --config_file "$ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml"
  --num_processes 1 --main_process_port 29603 "$ROOT/starVLA/training/train_starvla.py"
  --config_yaml "$CONFIG")
printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
[[ $dry_run == false ]] || exit 0
count=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
[[ $count -ge 1 ]] || { echo 'GPU_NOT_READY: no visible CUDA device' >&2; exit 1; }
mkdir -p "$LOG_DIR"
if [[ -f "$LOG_DIR/act_latest.pid" ]]; then
  old_pid=$(cat "$LOG_DIR/act_latest.pid")
  state=$(ps -p "$old_pid" -o stat= 2>/dev/null || true)
  if [[ -n $state && $state != Z* ]]; then
    echo "An Snap launcher may still be running: PID=$old_pid; inspect it before resuming" >&2; exit 1
  fi
fi
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
backup="$LOG_DIR/resume50k_config_backup_$stamp"
mkdir -p "$backup"
old_out=/root/data/outputs/starVLA/dualarm_lt_snap_act_h50_v2/dualarm_lt_snap_act_h50_v2
for name in config.yaml config.full.yaml summary.jsonl dataset_statistics.json; do
  [[ ! -f "$old_out/$name" ]] || cp -p "$old_out/$name" "$backup/$name"
done
log="$LOG_DIR/act_resume50k_$stamp.log"
nohup setsid "${cmd[@]}" >"$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$LOG_DIR/act_latest.pid"
printf '%s\n' "$pid" > "$LOG_DIR/resume50k_latest.pid"
ln -sfn "$(basename "$log")" "$LOG_DIR/act_latest.log"
echo "SNAP_RESUME_LAUNCH_REQUESTED pid=$pid total_steps=50000 port=29603"
echo "LOG=$log"
echo "FOLLOW: tail -F $LOG_DIR/act_latest.log"
echo "STOP: kill -- -$pid"

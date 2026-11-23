#!/usr/bin/env bash
# Synthetic inference against an already running Docker service; no robot I/O.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:?Usage: test_running_model.sh PROFILE}"
shift
case "$PROFILE" in
  qwenpi_v3_v2) SERVICE=qwen-v2; PORT=10199 ;;
  qwenpi_v3_v3) SERVICE=qwen-v3; PORT=10200 ;;
  act_v2) SERVICE=act; PORT=10201 ;;
  place_qwenpi_v3_v2) SERVICE=place-qwen-v2; PORT=10202 ;;
  snap_act_v2) SERVICE=snap-act; PORT=10203 ;;
  insert_act_v2) SERVICE=insert-act; PORT=10204 ;;
  push_pull_qwenpi_v3_v2) SERVICE=push-pull-qwen-v2; PORT=10205 ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac
"$HERE/compose.sh" exec -T "$SERVICE" python /app/server/smoke_predict_16.py \
  --profile "$PROFILE" --port "$PORT" "$@"

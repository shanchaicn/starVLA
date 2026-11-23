#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-qwenpi_v3_v2}"
if (( $# > 0 )); then shift; fi
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

"$HERE/compose.sh" up -d "$SERVICE" "$@"
echo "Waiting for $PROFILE model service on 127.0.0.1:$PORT ..."
for (( attempt=1; attempt<=90; attempt++ )); do
  if HEALTH="$(curl --noproxy '*' --max-time 2 -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null)"; then
    if "$HERE/compose.sh" ps --status running --services | grep -Fxq "$SERVICE"; then
      echo "$HEALTH"
      echo "MODEL_READY profile=$PROFILE port=$PORT"
      exit 0
    fi
  fi
  if ! "$HERE/compose.sh" ps --status running --services | grep -Fxq "$SERVICE"; then
    echo "MODEL_CONTAINER_EXITED profile=$PROFILE service=$SERVICE" >&2
    "$HERE/compose.sh" logs --tail 80 "$SERVICE" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "MODEL_NOT_READY profile=$PROFILE port=$PORT after 180s" >&2
"$HERE/compose.sh" ps -a >&2 || true
"$HERE/compose.sh" logs --tail 80 "$SERVICE" >&2 || true
exit 1

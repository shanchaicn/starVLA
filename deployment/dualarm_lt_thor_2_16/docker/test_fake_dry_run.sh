#!/usr/bin/env bash
# Docker inference smoke test with synthetic images and state; no robot Topic or RPC.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() {
  echo "Usage: $0 PROFILE [--cycles N | --continuous] [--interval-s SECONDS]"
}
PROFILE=act_v2
if (( $# > 0 )) && [[ "$1" != --* ]]; then PROFILE="$1"; shift; fi
case "$PROFILE" in
  act_v2|qwenpi_v3_v2|qwenpi_v3_v3|place_qwenpi_v3_v2|snap_act_v2|insert_act_v2|push_pull_qwenpi_v3_v2) ;;
  *) usage >&2; exit 2 ;;
esac
CYCLES=1
INTERVAL_S=1
MODE_SET=0
while (( $# > 0 )); do
  case "$1" in
    --cycles)
      if (( MODE_SET )) || (( $# < 2 )) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then usage >&2; exit 2; fi
      CYCLES="$2"; MODE_SET=1; shift 2 ;;
    --continuous)
      if (( MODE_SET )); then usage >&2; exit 2; fi
      CYCLES=0; MODE_SET=1; shift ;;
    --interval-s)
      if (( $# < 2 )); then usage >&2; exit 2; fi
      INTERVAL_S="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if ! python3 -c 'import math,sys; x=float(sys.argv[1]); sys.exit(0 if math.isfinite(x) and x>=0 else 1)' "$INTERVAL_S" 2>/dev/null; then
  echo "--interval-s must be a finite non-negative number" >&2
  exit 2
fi

export LOCAL_UID="${LOCAL_UID:-$(id -u)}" LOCAL_GID="${LOCAL_GID:-$(id -g)}"
if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi
COMPOSE=("${DOCKER[@]}" compose --file "$HERE/compose.yaml")
PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
NAME="starvla-lt-fake-${PROFILE//_/-}-$$"

cleanup() {
  "${DOCKER[@]}" stop --time 10 "$NAME" >/dev/null 2>&1 || true
  "${DOCKER[@]}" container rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[FAKE DRY-RUN] profile=$PROFILE port=$PORT; synthetic state/images only; NO Topic, NO robot RPC"
"${COMPOSE[@]}" run --rm -T --no-deps client --profile "$PROFILE" --self-test
"${COMPOSE[@]}" run --rm -T --no-deps act --profile "$PROFILE" --self-test
"${COMPOSE[@]}" run -d --no-deps --name "$NAME" act \
  --profile "$PROFILE" --port "$PORT" >/dev/null

ready=0
for (( attempt=1; attempt<=150; attempt++ )); do
  if curl --noproxy '*' --max-time 2 -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if [[ "$("${DOCKER[@]}" inspect --format '{{.State.Running}}' "$NAME" 2>/dev/null || true)" != true ]]; then
    break
  fi
  sleep 2
done
if (( ready == 0 )); then
  echo "[FAKE DRY-RUN] model did not become healthy; last container logs:" >&2
  "${DOCKER[@]}" logs --tail 80 "$NAME" >&2 || true
  exit 1
fi

cycle=0
while (( CYCLES == 0 || cycle < CYCLES )); do
  cycle=$((cycle + 1))
  echo "[FAKE DRY-RUN] cycle=$cycle started=$(date -Is)"
  "${DOCKER[@]}" exec "$NAME" python /app/server/smoke_predict_16.py \
    --profile "$PROFILE" --port "$PORT"
  echo "[FAKE DRY-RUN] cycle=$cycle PASS: two inference responses; no robot commands"
  if (( CYCLES == 0 || cycle < CYCLES )); then sleep "$INTERVAL_S"; fi
done

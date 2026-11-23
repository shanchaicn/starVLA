#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-qwenpi_v3_v2}"
if [ "$#" -gt 0 ]; then shift; fi
case "$PROFILE" in
  qwenpi_v3_v2|qwenpi_v3_v3|act_v2|place_qwenpi_v3_v2|snap_act_v2|insert_act_v2|push_pull_qwenpi_v3_v2) ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac
MODE=--dry-run
if [ "${1:-}" = "--real" ]; then MODE=--real; shift;
elif [ "${1:-}" = "--dry-run" ]; then shift; fi
exec "$HERE/compose.sh" run --rm --no-deps client --profile "$PROFILE" "$MODE" "$@"

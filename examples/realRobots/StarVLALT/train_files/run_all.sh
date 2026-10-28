#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <smoke|pilot|train> [--dry-run]" >&2
  exit 2
fi

STAGE="$1"
shift
EXTRA_ARGS=("$@")
TASKS=(insert push_pull snap suck place)

echo "Preflighting all five datasets before starting any training..."
for task in "${TASKS[@]}"; do
  "${SCRIPT_DIR}/run_task.sh" "${task}" "${STAGE}" --check-only
done

for task in "${TASKS[@]}"; do
  echo "Starting ${task} (${STAGE})"
  "${SCRIPT_DIR}/run_task.sh" "${task}" "${STAGE}" "${EXTRA_ARGS[@]}"
done

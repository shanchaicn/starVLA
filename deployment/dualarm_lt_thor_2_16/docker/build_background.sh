#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf 'running\n' > "$HERE/build.exitcode"
trap 'result=$?; printf "%s\n" "$result" > "$HERE/build.exitcode"' EXIT
docker compose --file "$HERE/compose.yaml" config --quiet
docker compose --file "$HERE/compose.yaml" --progress plain build qwen-v2 client

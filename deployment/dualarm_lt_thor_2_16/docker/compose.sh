#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOCAL_UID="${LOCAL_UID:-$(id -u)}" LOCAL_GID="${LOCAL_GID:-$(id -g)}"
if docker info >/dev/null 2>&1; then
  exec docker compose --file "$HERE/compose.yaml" "$@"
else
  exec sudo --preserve-env=LOCAL_UID,LOCAL_GID docker compose --file "$HERE/compose.yaml" "$@"
fi

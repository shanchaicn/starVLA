#!/usr/bin/env bash
set -euo pipefail
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 \
  -L 127.0.0.1:10199:127.0.0.1:10199 \
  -L 127.0.0.1:10200:127.0.0.1:10200 \
  -L 127.0.0.1:10201:127.0.0.1:10201 codeit@192.168.2.252


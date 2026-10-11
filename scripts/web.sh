#!/usr/bin/env bash
# Start the dsh Web UI on the local stack: ./scripts/web.sh [workdir]   (default port 3090)
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/dsh-env.sh"
WORKDIR="${1:-$STACK_ROOT/workspace}"
WEB_PORT="${WEB_PORT:-3090}"
model_server_up || { echo "error: the $BACKEND model server is not up - run scripts/serve-$BACKEND.sh" >&2; exit 1; }
mkdir -p "$WORKDIR" && cd "$WORKDIR"
exec "$DSH_BIN" $(dsh_profile_args qwen-web web) --no-open --port "$WEB_PORT"

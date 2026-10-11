#!/usr/bin/env bash
# Run one agent task headless: ./scripts/run-task.sh <workdir> "<task>"
# The working directory is also dsh's sandbox root (bash and file tools can write only there and in /tmp).
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/dsh-env.sh"
WORKDIR="${1:?usage: run-task.sh <workdir> \"<task>\"}"; shift
TASK="${1:?usage: run-task.sh <workdir> \"<task>\"}"
model_server_up || { echo "error: the $BACKEND model server is not up - run scripts/serve-$BACKEND.sh" >&2; exit 1; }
cd "$WORKDIR"
exec "$DSH_BIN" $(dsh_profile_args qwen headless) "$TASK"

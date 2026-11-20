#!/usr/bin/env bash
# Run one agent task headless: ./scripts/run-task.sh <workdir> "<task>"
# The working directory is also dsh's sandbox root (bash and file tools can write only there and in /tmp).
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/dsh-env.sh"
WORKDIR="${1:?usage: run-task.sh <workdir> \"<task>\"}"; shift
TASK="${1:?usage: run-task.sh <workdir> \"<task>\"}"
if ! model_server_up; then   # AUTOSTART=1 (or scripts/qwen) starts it; the patch then follows its context limit
  [[ "${AUTOSTART:-0}" == "1" ]] || { echo "error: the $BACKEND model server is not up - run scripts/qwen start (or AUTOSTART=1)" >&2; exit 1; }
  ensure_model_server && source "$STACK_ROOT/scripts/dsh-env.sh"
fi
cd "$WORKDIR"
exec "$DSH_BIN" $(dsh_profile_args qwen headless) "$TASK"

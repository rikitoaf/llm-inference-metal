#!/usr/bin/env bash
# Copy the pristine demo project to workspace/demo (the agent's working directory and sandbox root).
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/config/stack.env"
rm -rf "$STACK_ROOT/workspace/demo"
mkdir -p "$STACK_ROOT/workspace"
cp -R "$STACK_ROOT/examples/demo-project" "$STACK_ROOT/workspace/demo"
echo "demo reset: $STACK_ROOT/workspace/demo"
(cd "$STACK_ROOT/workspace/demo" && python3 -m unittest -q 2>&1 | tail -1) || true

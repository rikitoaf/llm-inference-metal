#!/usr/bin/env bash
# Stop the model servers of this workspace: Splash (server.server and its serve-native engine) and
# llama-server on $LLAMA_PORT. Splash gets SIGINT first so it flushes its persistent cache.
source "$(cd "$(dirname "$0")/.." && pwd)/config/stack.env"

pid=$(pgrep -f "splash-src/.venv/bin/python -u -P -m server.server" || true)
if [[ -n "$pid" ]]; then
  kill -INT $pid
  for _ in $(seq 1 90); do
    pgrep -f "splash serve-native" >/dev/null || pgrep -f "server.server" >/dev/null || break
    sleep 1
  done
fi
pkill -INT -f "^splash serve|/splash serve " 2>/dev/null || true      # Homebrew splash
pkill -9 -f "splash serve-native" 2>/dev/null || true
lpid=$(lsof -ti "tcp:$LLAMA_PORT" -sTCP:LISTEN 2>/dev/null || true)
[[ -n "$lpid" ]] && kill $lpid
pgrep -fl "server.server|serve-native|llama-server" && { echo "warning: still running (above)" >&2; exit 1; }
echo "stopped"

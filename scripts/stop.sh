#!/usr/bin/env bash
# Stop the model servers of this workspace: Splash (server.server and its serve-native engine), lithos-metal
# and llama-server on $LLAMA_PORT. Splash and lithos get SIGINT first so they flush their persistent caches.
source "$(cd "$(dirname "$0")/.." && pwd)/config/stack.env"

lithos_pid=$(pgrep -f "lithos-src/.venv/bin/lithos-metal serve" || true)
if [[ -n "$lithos_pid" ]]; then
  kill -INT $lithos_pid
  for _ in $(seq 1 60); do pgrep -f "lithos-metal serve" >/dev/null || break; sleep 1; done
  pkill -9 -f "lithos-src/.venv/bin/lithos-metal serve" 2>/dev/null || true
fi

# Any Splash checkout under splash/ (splash-src, or a dev copy started with SPLASH_SRC=...).
pid=$(pgrep -f "splash/[^ ]*/.venv/bin/python -u -P -m server.server" || true)
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
pgrep -fl "server.server|serve-native|llama-server|lithos-metal serve" && { echo "warning: still running (above)" >&2; exit 1; }
echo "stopped"

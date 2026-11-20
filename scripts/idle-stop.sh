#!/usr/bin/env bash
# Stops the Splash server after SPLASH_IDLE_STOP (config/stack.env; e.g. 30m) without a request, so the GPU and its
# memory are completely free. Splash's own --idle-release only drops the weights; its processes stay. Started by
# serve-splash.sh; scripts/qwen starts the server again on demand (~45 s cold).
# Activity = the newest write to logs/splash/steps.jsonl (a line per prefill chunk and decode cycle); serve.log only
# when there is no step log (it also gets the server's own idle-release message).
# Exits when the server is gone, or after stopping it.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/../config/stack.env"
source "$here/lib/system.sh"
limit=$(duration_seconds "${SPLASH_IDLE_STOP:-}")
[[ "$limit" =~ ^[0-9]+$ ]] && (( limit > 0 )) || exit 0
BACKEND=splash
started=$(date +%s) seen=0
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') idle-stop[$$]: $*"; }
trap '' HUP                                   # outlive the terminal or tool that started the server
trap 'say "killed by a signal"; exit 1' TERM INT
say "watching: stop after ${SPLASH_IDLE_STOP} without a request"
activity="${SPLASH_LOG_DIR:-$STACK_ROOT/logs/splash}/steps.jsonl"
[[ -f "$activity" ]] || activity="${SPLASH_LOG_DIR:-$STACK_ROOT/logs/splash}/serve.log"

while sleep 30; do
  if model_process_running; then seen=1
  elif (( seen )); then say "server gone (stopped elsewhere)"; exit 0
  elif (( $(date +%s) - started > 900 )); then say "server never came up"; exit 0
  else continue
  fi
  last=$started
  [[ -f "$activity" ]] && m=$(stat -f %m "$activity") && (( m > last )) && last=$m
  idle=$(( $(date +%s) - last ))
  if (( idle >= limit )); then
    say "no request for $((idle / 60)) min (SPLASH_IDLE_STOP=$SPLASH_IDLE_STOP), stopping Splash"
    "$here/stop.sh"
    exit 0
  fi
done

# Sourced helpers about the machine and the model server, shared by qwen, status.sh, doctor.sh, run-task.sh, web.sh
# and the GPU measurement scripts. Needs config/stack.env sourced first (STACK_ROOT, BACKEND, ports).

# Microsoft Defender's current CPU percent (0 when it is not running). Its real-time scan takes 20-45% off the GPU
# while it runs (splash/SPEED_LOG.md, 2026-10-29).
defender_cpu() {
  ps -axo %cpu=,comm= | awk '/wdavdaemon_unprivileged/ {print int($1); found=1} END {if (!found) print 0}' | head -1
}

# Splash's own estimate of reclaimable memory, whole GiB (it needs ~16 to start).
reclaimable_gib() { python3 "$STACK_ROOT/splash/tools/hostavail.py" --gib 2>/dev/null; }

# 0 on AC power. Low Power mode on battery throttles the GPU.
on_ac_power() { pmset -g batt | grep -q "AC Power"; }

# Seconds from a duration: 90, 90s, 30m, 2h. Empty or "off" prints nothing.
duration_seconds() {
  case "$1" in
    ""|off|0) ;;
    *h) echo $(( ${1%h} * 3600 )) ;;
    *m) echo $(( ${1%m} * 60 )) ;;
    *s) echo "${1%s}" ;;
    *) echo "$1" ;;
  esac
}

model_port() { case "$BACKEND" in splash) echo "$SPLASH_PORT" ;; lithos) echo "$LITHOS_PORT" ;; *) echo "$LLAMA_PORT" ;; esac; }
model_healthy() { curl -sf -m 3 "http://$LLAMA_HOST:$(model_port)/health" >/dev/null; }
# A server process of $BACKEND exists (it may still be loading).
model_process_running() {
  case "$BACKEND" in
    # serve-splash.sh, then the source launcher (it may run make first), then the server itself
    splash) pgrep -f "scripts/serve-splash.sh|splash/[^ ]*/splash serve|splash/[^ ]*/.venv/bin/python -u -P -m server.server" >/dev/null ;;
    lithos) pgrep -f "lithos-metal serve" >/dev/null ;;
    *) lsof -ti "tcp:$LLAMA_PORT" -sTCP:LISTEN >/dev/null 2>&1 ;;
  esac
}

# Starts the $BACKEND server in the background when it is not up (its console goes to logs/<backend>-console.log),
# then waits for /health with a progress line. Returns 1 when it does not come up within $1 seconds (default 300).
ensure_model_server() {
  local limit=${1:-300} waited=0 log="$STACK_ROOT/logs/$BACKEND-console.log"
  model_healthy && return 0
  if ! model_process_running; then
    local avail; avail=$(reclaimable_gib)
    [[ "$BACKEND" == "splash" && "$avail" =~ ^[0-9]+$ ]] && (( avail < 16 )) \
      && echo "note: ~$avail GiB reclaimable; Splash needs ~16 GiB and waits until it has it (close IDEs, Teams)" >&2
    mkdir -p "$STACK_ROOT/logs"
    echo "starting the $BACKEND server (console: $log)" >&2
    nohup "$STACK_ROOT/scripts/serve-$BACKEND.sh" >> "$log" 2>&1 &
    disown
  fi
  while ! model_healthy; do
    if (( waited >= limit )); then printf '\n' >&2; echo "error: the $BACKEND server is not up after ${limit}s; see $log" >&2; return 1; fi
    if (( waited > 5 )) && ! model_process_running; then
      printf '\n' >&2; echo "error: the $BACKEND server exited; last lines of $log:" >&2; tail -5 "$log" >&2; return 1
    fi
    printf '\rwaiting for the %s server... %ds' "$BACKEND" "$waited" >&2
    sleep 3; waited=$((waited + 3))
  done
  (( waited > 0 )) && printf '\rthe %s server is up after %ds          \n' "$BACKEND" "$waited" >&2
  return 0
}

#!/usr/bin/env bash
# Benchmark Splash KV formats one after another: for each "FORMAT LABEL [kvbench run args...]"
# line on stdin, restart the server with --kv-format FORMAT (scripts/serve-splash.sh) and run
# tools/kvbench.py run --label LABEL ARGS. Run it detached (nohup): a suite takes hours.
#
#   printf '%s\n' "turbo-int4 turbo-int4" "int8 int8 --parts judgments greedy" | tools/run_kv_suite.sh
#
# An optional third field ENV=K=V[,K=V...] sets server environment for that line, for example
#   int8 i8-pc0 ENV=SPLASH_PERSISTENT_CACHE=0 --parts speed needles
# The values also reach kvbench, which records them in the result's meta.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

stop_server() {
  local pid
  pid=$(pgrep -f "splash-src/.venv/bin/python -u -P -m server.server" || true)
  [[ -n "$pid" ]] && kill -INT $pid
  for _ in $(seq 1 90); do
    pgrep -f "splash serve-native" >/dev/null || pgrep -f "server.server" >/dev/null || break
    sleep 1
  done
  pkill -9 -f "splash serve-native" 2>/dev/null || true
  sleep 3
}

start_server() {   # format [K=V...]
  local log="logs/splash-$1.log" format="$1"; shift
  env "$@" SPLASH_KV_FORMAT="$format" nohup ./scripts/serve-splash.sh > "$log" 2>&1 &
  for _ in $(seq 1 240); do
    grep -q "Ready ·" "$log" && return 0
    grep -qE "Error ·|bootstrap failed|Traceback" "$log" && return 1
    sleep 3
  done
  return 1
}

while read -r format label args; do
  [[ -z "${format:-}" || "$format" == \#* ]] && continue
  envs=()
  if [[ "$args" == ENV=* ]]; then
    spec="${args%% *}"; [[ "$args" == *" "* ]] && args="${args#* }" || args=""
    IFS=, read -r -a envs <<< "${spec#ENV=}"
  fi
  echo "=== $(date '+%H:%M:%S') $label: --kv-format $format ${envs[*]:-}, kvbench $args"
  started=0
  for attempt in 1 2 3 4; do               # a start refused for memory usually succeeds a minute later
    stop_server
    if start_server "$format" "${envs[@]+"${envs[@]}"}"; then started=1; break; fi
    echo "server failed to start for $format (attempt $attempt, $(python3 splash/tools/hostavail.py --gib) GiB reclaimable); retrying in 60 s"
    sleep 60
  done
  if (( !started )); then
    echo "server failed to start for $format; see logs/splash-$format.log"
    continue
  fi
  grep "Ready ·" "logs/splash-$format.log" | tail -1
  # shellcheck disable=SC2086
  env "${envs[@]+"${envs[@]}"}" python3 splash/tools/kvbench.py run --label "$label" $args
done
echo "=== $(date '+%H:%M:%S') suite done"

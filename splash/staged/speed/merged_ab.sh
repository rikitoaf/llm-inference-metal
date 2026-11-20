#!/usr/bin/env bash
# End-to-end A/B of the speed flags in splash/dev-merged against splash/splash-dev (the production baseline): per
# config, ab_collect.py's agent prompts (greedy: identity + decode tok/s) and kvbench's code request at the server
# default temperature (dsh's case). No persistent cache and a scratch log dir, so production's cache and logs/splash
# are untouched.
#   splash/staged/speed/merged_ab.sh OUTDIR name:src:ENV=VALUE,ENV=VALUE ...
#   e.g. merged_ab.sh logs/speed/ab-1029 base:splash-dev: head:dev-merged:SPLASH_DRAFT_HEAD_ROWS=98304
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
source splash/staged/speed/quiet.sh
require_measurable || exit 1
OUT="$1"; shift
mkdir -p "$OUT"
for config in "$@"; do
  IFS=: read -r name src envs <<< "$config"
  ./scripts/stop.sh >/dev/null 2>&1
  wait_quiet 30 20 "${QUIET_TIMEOUT:-900}"
  env_args=()
  [[ -n "$envs" ]] && IFS=, read -r -a env_args <<< "$envs"
  log="$OUT/serve-$name.log"
  # Splash refuses to start below its memory reserve ("host memory reserve protected"); retry, waiting for memory.
  for attempt in 1 2 3; do
    for _ in $(seq 1 60); do (( $(python3 splash/tools/hostavail.py --gib 2>/dev/null || echo 0) >= 12 )) && break; sleep 10; done
    env ${env_args[@]+"${env_args[@]}"} SPLASH_SRC="$ROOT/splash/$src" SPLASH_PERSISTENT_CACHE=0 \
      SPLASH_LOG_DIR="$ROOT/$OUT/logs-$name" nohup ./scripts/serve-splash.sh > "$log" 2>&1 &
    for _ in $(seq 1 300); do grep -qE "Ready ·|Error · native protocol" "$log" && break; sleep 3; done
    grep -q "Ready ·" "$log" && break
    echo "$name: start attempt $attempt failed (see $log)"; ./scripts/stop.sh >/dev/null 2>&1; sleep 10
  done
  grep -q "Ready ·" "$log" || { echo "$name: server did not start (see $log)"; continue; }
  echo "== $name ($src ${envs:-no flags}) defender $(defender_cpu)%" | tee -a "$OUT/summary.txt"
  # Defender's CPU every 5 s while this config is measured (its mean is printed after the config).
  ( while :; do defender_cpu; sleep 5; done ) > "$OUT/defender-$name.txt" & sampler=$!
  python3 splash/staged/speed/ab_collect.py --out "$OUT/$name.json" | sed "s/^/$name agent /" | tee -a "$OUT/summary.txt"
  splash/splash-src/.venv/bin/python splash/tools/kvbench.py run --label "ab1029-$name" --fresh --parts speed \
    --temperature default --speed-repeats 8 2>&1 | grep "speed (" | sed "s/^/$name T=default /" | tee -a "$OUT/summary.txt"
  kill $sampler 2>/dev/null
  awk -v n="$name" '{s+=$1; c++; if ($1>30) b++} END {printf "%s defender mean %.0f%%, busy (>30%%) %d of %d samples\n", n, s/c, b, c}' \
    "$OUT/defender-$name.txt" | tee -a "$OUT/summary.txt"
done
./scripts/stop.sh >/dev/null 2>&1
echo "wrote $OUT/summary.txt"

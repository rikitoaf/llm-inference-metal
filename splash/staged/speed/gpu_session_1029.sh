#!/usr/bin/env bash
# The 2026-10-29 GPU session after the e2e A/B: park-lane and system-state multi-turn checks (flag off vs on, fresh
# server each), then the verify-attention decode profile. Logs in logs/speed/ab-1029/.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT=logs/speed/ab-1030
mkdir -p "$OUT/park" "$OUT/sysstate"
PY=splash/splash-src/.venv/bin/python
start() {   # name env...   (dev-merged, fresh server, no persistent cache unless given)
  local name=$1; shift
  ./scripts/stop.sh >/dev/null 2>&1; sleep 5
  local log="$OUT/serve-$name.log"
  for attempt in 1 2 3; do
    for _ in $(seq 1 60); do (( $(python3 splash/tools/hostavail.py --gib 2>/dev/null || echo 0) >= 12 )) && break; sleep 10; done
    env SPLASH_PERSISTENT_CACHE=0 "$@" SPLASH_SRC="$ROOT/splash/dev-merged" SPLASH_LOG_DIR="$ROOT/$OUT/logs-$name" \
      nohup ./scripts/serve-splash.sh > "$log" 2>&1 &
    for _ in $(seq 1 300); do grep -qE "Ready ·|Error · native protocol" "$log" && break; sleep 3; done
    grep -q "Ready ·" "$log" && return 0
    echo "$name: start attempt $attempt failed"; ./scripts/stop.sh >/dev/null 2>&1; sleep 10
  done
  return 1
}
echo "== park"
start park-off SPLASH_PARK_LANE=0 && python3 splash/staged/speed/park_check.py --thinking --save "$OUT/park/off.json" | tee "$OUT/park/off.txt"
start park-on SPLASH_PARK_LANE=1 && python3 splash/staged/speed/park_check.py --thinking --compare "$OUT/park/off.json" | tee "$OUT/park/on.txt"
echo "== sysstate"
SSC=splash/dev-merged/dev/tools/system_state_check.py
start ss-cold SPLASH_SYSTEM_STATE=1 && $PY $SSC --url http://127.0.0.1:8000 --only second --save "$OUT/sysstate/cold.json" | tee "$OUT/sysstate/cold.txt"
start ss-on SPLASH_SYSTEM_STATE=1 && $PY $SSC --url http://127.0.0.1:8000 --compare "$OUT/sysstate/cold.json" | tee "$OUT/sysstate/on.txt"
start ss-off SPLASH_SYSTEM_STATE=0 && $PY $SSC --url http://127.0.0.1:8000 | tee "$OUT/sysstate/off.txt"
./scripts/stop.sh >/dev/null 2>&1
echo "== attention profile"
splash/staged/speed/attn_profile.sh "$OUT/profile" 2>&1 | tee "$OUT/profile.txt"
echo "session done"

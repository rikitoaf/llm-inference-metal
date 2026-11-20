#!/usr/bin/env bash
# 2026-10-30, part 2: the verify-attention profile, then park_check with reasoning sent back (flag off, then on).
cd "$(dirname "$0")/../../.."
O=logs/speed/ab-1101
splash/staged/speed/attn_profile.sh $O/profile 32768 65536 2>&1 | tee $O/profile.txt
echo "PROFILE DONE"
start() {
  local name=$1; shift
  ./scripts/stop.sh >/dev/null 2>&1; sleep 5
  env SPLASH_PERSISTENT_CACHE=0 "$@" SPLASH_SRC="$PWD/splash/dev-merged" SPLASH_LOG_DIR="$PWD/$O/logs-$name" \
    nohup ./scripts/serve-splash.sh > "$O/serve-$name.log" 2>&1 &
  for _ in $(seq 1 300); do grep -qE "Ready ·|Error" "$O/serve-$name.log" && break; sleep 3; done
  grep -q "Ready ·" "$O/serve-$name.log"
}
start park2-off SPLASH_PARK_LANE=0 && python3 splash/staged/speed/park_check.py --thinking --save $O/park/off2.json
start park2-on SPLASH_PARK_LANE=1 && python3 splash/staged/speed/park_check.py --thinking --compare $O/park/off2.json
./scripts/stop.sh >/dev/null 2>&1
echo "ALL DONE"

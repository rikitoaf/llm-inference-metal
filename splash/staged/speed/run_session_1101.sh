#!/usr/bin/env bash
# 2026-11-01: production with the draft head shortlist (splash-src) vs the old production (splash-dev).
cd "$(dirname "$0")/../../.."
O=logs/speed/ab-1101
mkdir -p $O
echo "== unit (splash-src)"
splash/staged/speed/metal_unit.sh splash/splash-src SPLASH_DRAFT_HEAD_ROWS=98304 2>&1 | tail -30 | tee $O/metal-unit.txt
(cd splash/splash-src && make build/engine-tests/kv-first-engine >/dev/null 2>&1 && build/engine-tests/kv-first-engine 2>&1 | tail -3) | tee $O/kv-first-engine.txt
echo "UNIT DONE"
QUIET_TIMEOUT=300 splash/staged/speed/merged_ab.sh $O base1:splash-dev: prod1:splash-src: prod2:splash-src: base2:splash-dev:
for x in prod1 prod2 base2; do echo "base1 vs $x"; python3 splash/staged/speed/ab_collect.py --compare $O/base1.json $O/$x.json; done | tee $O/compare.txt
grep -h "DRAFT_HEAD_ROWS" $O/serve-prod1.log | head -2
echo "ALL DONE"

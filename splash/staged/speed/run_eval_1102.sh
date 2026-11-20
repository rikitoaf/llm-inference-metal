#!/usr/bin/env bash
# 2026-11-02: accuracy + wall-time baseline at the effort that really runs (xhigh; the template maps high to xhigh),
# production splash-src with the draft head shortlist. t1-t6 x 3 reps, JEV off (the default).
set -uo pipefail
cd /Users/abdul.rakib/agentMania/dsh-qwen-jev
source splash/staged/speed/quiet.sh
OUT=eval/results/xhigh-baseline-1102
mkdir -p "$OUT"
pmset -g batt | grep -q "AC Power" || { echo "on battery; abort"; exit 1; }
( while :; do echo "$(date +%s) $(defender_cpu)"; sleep 10; done ) > "$OUT/defender.txt" & sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT
if ! curl -sf -m 3 localhost:8000/health >/dev/null; then
  nohup scripts/serve-splash.sh > "$OUT/serve.out" 2>&1 &
  for i in $(seq 1 180); do curl -sf -m 3 localhost:8000/health >/dev/null && break; sleep 5; done
  curl -sf -m 3 localhost:8000/health >/dev/null || { echo "splash did not come up"; tail -20 "$OUT/serve.out"; exit 1; }
fi
echo "splash up $(date)"
wait_quiet 30 20 300
python3 eval/run_eval.py --conds nojev --reps 3 --timeout 900 --out "$OUT"
echo "EVAL DONE $(date)"
awk '{s+=$2; c++; if ($2>30) b++} END {printf "defender mean %.0f%%, busy (>30%%) %d of %d samples\n", s/c, b, c}' "$OUT/defender.txt"

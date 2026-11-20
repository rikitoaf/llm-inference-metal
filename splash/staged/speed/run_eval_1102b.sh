#!/usr/bin/env bash
# 2026-11-02 B: same eval as run_eval_1102.sh (xhigh, production Splash, JEV off) with the runaway-reply guard v2 (thinking off after a cut)
# (plugins/dsh-plugin-runaway) on. Starts when the baseline has finished.
set -uo pipefail
cd /Users/abdul.rakib/agentMania/dsh-qwen-jev
source splash/staged/speed/quiet.sh
# (baseline finished 11:00)
OUT=eval/results/xhigh-guard2-1102
mkdir -p "$OUT"
( while :; do echo "$(date +%s) $(defender_cpu)"; sleep 10; done ) > "$OUT/defender.txt" & sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT
curl -sf -m 3 localhost:8000/health >/dev/null || { echo "splash is down"; exit 1; }
wait_quiet 30 20 300
python3 eval/run_eval.py --conds guard --reps 3 --timeout 900 --out "$OUT"
echo "GUARD EVAL DONE $(date)"
awk '{s+=$2; c++; if ($2>30) b++} END {printf "defender mean %.0f%%, busy (>30%%) %d of %d samples\n", s/c, b, c}' "$OUT/defender.txt"

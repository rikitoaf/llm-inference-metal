#!/usr/bin/env bash
# Draft tree A/B on the real server: the agent-like prompts of splash/tools/tree_collect.py (greedy, thinking on) and
# kvbench's 400-token code request sent as dsh sends it (no temperature: the server default, 0.6), each with
# SPLASH_DRAFT_TREE off and on. Prints tok/s and tokens per cycle from the server's counters.
#   splash/staged/speed/tree_ab.sh            output: logs/speed/tree-ab-<time>.txt
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
source splash/staged/speed/quiet.sh
require_measurable || exit 1
OUT="logs/speed/tree-ab-$(date +%Y%m%d-%H%M).txt"
counters() { curl -s http://127.0.0.1:8000/status | python3 -c 'import json,sys; s=json.load(sys.stdin); m=s.get("metrics",{}); print(m.get("decode_output_tokens",0), m.get("decode_wall_ms",0), s.get("scheduler",{}).get("decode_batches",0))' 2>/dev/null; }
for mode in 0 1; do
  ./scripts/stop.sh >/dev/null 2>&1
  wait_quiet 30 20 900
  SPLASH_DRAFT_TREE=$mode nohup ./scripts/serve-splash.sh > logs/speed/splash-tree$mode.log 2>&1 &
  for _ in $(seq 1 200); do grep -q "Ready ·" logs/speed/splash-tree$mode.log && break; sleep 3; done
  read t0 w0 c0 < <(counters)
  python3 splash/tools/tree_collect.py --max-tokens 1500 | sed "s/^/tree=$mode agent /" | tee -a "$OUT"
  read t1 w1 c1 < <(counters)
  python3 -c "t=$t1-$t0; w=$w1-$w0; c=$c1-$c0; print(f'tree=$mode agent total: {t} tokens, {t/w*1000:.1f} tok/s decode, {t/max(c,1):.3f} tokens/cycle')" | tee -a "$OUT"
  splash/splash-src/.venv/bin/python splash/tools/kvbench.py run --label tree$mode-dflt --fresh --parts speed \
    --temperature default --speed-repeats 8 2>&1 | grep "speed (" | sed "s/^/tree=$mode T=default /" | tee -a "$OUT"
done
./scripts/stop.sh >/dev/null 2>&1
echo "wrote $OUT"

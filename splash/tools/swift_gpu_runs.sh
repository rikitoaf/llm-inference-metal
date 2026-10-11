#!/usr/bin/env bash
# Check and measure the Swift 1.5 Qwen3.8-27B Uncensored model (MODEL_PROFILE=swift, README section 33)
# against the default Qwen3.8-27B (MODEL_PROFILE=qwen), both on Splash with turbo-int4 KV. ~45 min of GPU;
# it restarts Splash, and each start needs ~16 GiB reclaimable (python3 splash/tools/hostavail.py).
#
#   nohup splash/tools/swift_gpu_runs.sh > logs/swift-gpu-runs.log 2>&1 & disown
#   EVAL_REPS=1 nohup splash/tools/swift_gpu_runs.sh > logs/swift-gpu-runs.log 2>&1 & disown
#
# 1. Per model: speed, needles at 16K/58K, and the judgment set v2 at 60K tokens. A judgment run gives each
#    model its accuracy (true answers); the KL between the two models' distributions is not a quality loss,
#    they are different models. The decode line shows DFlash2's draft acceptance on each, the number that
#    matters most here: the drafter was trained on Qwen3.8-27B, not on this fine-tune.
# 2. With EVAL_REPS=N (default 0 = skipped): the six eval/ tasks, N repetitions, JEV off, per model. About
#    1-1.5 h per model per repetition. This is the check of how well each model codes inside the harness.
# Output: logs/kvbench/summary-swift.txt (and eval/results/<model>-<time>/).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EVAL_REPS="${EVAL_REPS:-0}"
STAMP="$(date +%Y%m%d-%H%M)"

if pgrep -f "run_kv_suite|next_gpu_runs|serve-native" >/dev/null; then echo "error: a suite or server is running" >&2; exit 1; fi
trap './scripts/stop.sh >/dev/null 2>&1' EXIT INT TERM
echo "=== $(date '+%H:%M:%S') start; $(python3 splash/tools/hostavail.py)"

# --- 1. kvbench, the new model first so a problem shows early
Q="--parts speed needles judgments --needle-tokens 16000 60000 --judgment-tokens 60000"
echo "turbo-int4 swift-i4 ENV=MODEL_PROFILE=swift $Q" | splash/tools/run_kv_suite.sh
cp logs/splash-turbo-int4.log logs/splash-swift.log 2>/dev/null
echo "turbo-int4 qwen-i4 ENV=MODEL_PROFILE=qwen $Q" | splash/tools/run_kv_suite.sh
./scripts/stop.sh

# --- 2. the harness evals
serve_up() {   # profile
  ./scripts/stop.sh >/dev/null 2>&1
  local log="logs/splash-eval-$1.log"
  MODEL_PROFILE="$1" nohup ./scripts/serve-splash.sh > "$log" 2>&1 &
  for _ in $(seq 1 240); do
    grep -q "Ready ·" "$log" && return 0
    grep -qE "^error:|Traceback" "$log" && return 1
    sleep 3
  done
  return 1
}
EVAL_DIRS=()
if (( EVAL_REPS > 0 )); then
  for profile in swift qwen; do
    echo "=== $(date '+%H:%M:%S') eval, profile $profile, $EVAL_REPS rep(s)"
    if serve_up "$profile"; then
      out="eval/results/$profile-$STAMP"
      MODEL_PROFILE="$profile" python3 eval/run_eval.py --conds nojev --reps "$EVAL_REPS" --out "$out"
      EVAL_DIRS+=("$out")
    else
      echo "server failed to start for profile $profile; see logs/splash-eval-$profile.log"
    fi
  done
  ./scripts/stop.sh
fi

# --- 3. summary
{
  echo "== Splash startup of the swift profile (logs/splash-swift.log)"
  grep -E "Loading|Chat template|Weights loaded|Ready" logs/splash-swift.log 2>/dev/null
  echo
  echo "== qwen-i4 (baseline) vs swift-i4"
  python3 splash/tools/kvbench.py compare qwen-i4 swift-i4
  for d in ${EVAL_DIRS[@]+"${EVAL_DIRS[@]}"}; do
    echo; echo "== eval $d"
    python3 eval/run_eval.py --summary "$d/results.jsonl"
  done
} > logs/kvbench/summary-swift.txt 2>&1
echo "=== $(date '+%H:%M:%S') done; see logs/kvbench/summary-swift.txt"

#!/usr/bin/env bash
# codebase-memory-mcp A/B (splash/EXTERNAL-2026-11-09.md section 3): production dsh vs production + DSH_CBM=1, on the
# eval tasks (eval/run_eval.py, conds nojev vs cbm) and on the platform bench (workspace/platform-bench/run_bench.py,
# when present). Each batch takes /tmp/splash-gpu.lock, starts Splash, runs, stops Splash and releases the lock, so the
# GPU is free between batches.
#
#   splash/tools/cbm_ab.sh [--quick] [--eval-only|--bench-only] [--eval-reps N] [--bench-reps N] OUT_DIR
#     --quick   eval t1-t3 x 1 rep, bench --quick (if the bench supports it) x 1 rep
# Output: OUT_DIR/eval/results.jsonl, OUT_DIR/bench/..., OUT_DIR/summary.md (cbm_ab_report.py tables).
# Exit: 0 when every batch ran, 1 on a failed batch, 3 on battery power.
set -uo pipefail
STACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$STACK_ROOT/config/stack.env"
source "$STACK_ROOT/scripts/lib/system.sh"

quick=0 run_eval=1 run_bench=1 eval_reps=1 bench_reps=2 out=""
while (( $# )); do
  case "$1" in
    --quick) quick=1 ;;
    --eval-only) run_bench=0 ;;
    --bench-only) run_eval=0 ;;
    --eval-reps) eval_reps="$2"; shift ;;
    --bench-reps) bench_reps="$2"; shift ;;
    -*) echo "usage: $0 [--quick] [--eval-only|--bench-only] [--eval-reps N] [--bench-reps N] OUT_DIR" >&2; exit 2 ;;
    *) out="$1" ;;
  esac
  shift
done
[[ -n "$out" ]] || { echo "usage: $0 [...] OUT_DIR" >&2; exit 2; }
on_ac_power || { echo "on battery: GPU measurements need AC power" >&2; exit 3; }
[[ -x "$HOME/.local/bin/codebase-memory-mcp" ]] || { echo "codebase-memory-mcp is not installed" >&2; exit 1; }
mkdir -p "$out"
export BACKEND=splash PYTHONDONTWRITEBYTECODE=1
bench_dir="$STACK_ROOT/workspace/platform-bench"
gpu_lock=/tmp/splash-gpu.lock
holding_lock=0
release_gpu() {   # only a holder stops Splash: without the lock the server may be another agent's
  (( holding_lock )) || return 0
  "$STACK_ROOT/scripts/stop.sh" >/dev/null 2>&1
  rmdir "$gpu_lock" 2>/dev/null
  holding_lock=0
}
trap release_gpu EXIT
trap 'exit 130' INT TERM HUP

# run_batch NAME command...: lock, serve, run, stop, unlock. Records Defender CPU and the Splash knobs.
run_batch() {
  local batch="$1"; shift
  echo "[$(date +%T)] $batch: waiting for $gpu_lock"
  until mkdir "$gpu_lock" 2>/dev/null; do sleep 15; done
  holding_lock=1
  echo "[$(date +%T)] $batch: lock taken; Defender CPU $(defender_cpu)%; reclaimable $(reclaimable_gib) GiB" | tee -a "$out/batches.log"
  env | grep -E '^SPLASH_(KV_FORMAT|DRAFT|DEFAULT_TEMPERATURE|EXTRA_ARGS)' | sort >> "$out/batches.log"
  local status=0
  if ensure_model_server 900; then
    "$@" || status=$?
  else
    status=1
  fi
  echo "[$(date +%T)] $batch: exit $status; Defender CPU $(defender_cpu)%" | tee -a "$out/batches.log"
  release_gpu
  return $status
}

failed=0
if (( run_eval )); then
  eval_tasks=(t1-intervals t2-semver t3-inventory t4-toposort t5-ratelimit t6-mdtoc)
  (( quick )) && eval_tasks=(t1-intervals t2-semver t3-inventory) && eval_reps=1
  run_batch eval python3 "$STACK_ROOT/eval/run_eval.py" --conds nojev cbm --reps "$eval_reps" \
    --tasks "${eval_tasks[@]}" --out "$out/eval" || failed=1
fi
# The bench runs one configuration per call (--label, --env), so each task runs once per arm with the arm order
# alternating across tasks and reps (A-B, B-A, ...). The bench stops with exit 3 if a platform repo changed.
run_bench_arms() {
  local tasks=("$@") rep index arm
  for (( rep = 0; rep < bench_reps; rep++ )); do
    for index in "${!tasks[@]}"; do
      local arms=(baseline cbm)
      (( (rep + index) % 2 )) && arms=(cbm baseline)
      for arm in "${arms[@]}"; do
        python3 "$bench_dir/run_bench.py" --tasks "${tasks[$index]}" --first-rep "$rep" --reps 1 --label "cbmab-$arm" \
          --env "DSH_CBM=$([[ $arm == cbm ]] && echo 1 || echo 0)" --out "$out/bench/$arm"
        local status=$?
        (( status == 3 )) && { echo "bench: a platform repo changed; stopping" >&2; return 3; }
        (( status )) && echo "bench: ${tasks[$index]} $arm r$rep exit $status" >&2
      done
    done
  done
}
if (( run_bench )); then
  if [[ -f "$bench_dir/run_bench.py" ]] && python3 "$bench_dir/run_bench.py" --validate >/dev/null; then
    bench_tasks=()
    for task_file in "$bench_dir"/tasks/*.json; do bench_tasks+=("$(basename "$task_file" .json)"); done
    if (( quick )); then   # every third task, 1 rep
      quick_tasks=()
      for index in "${!bench_tasks[@]}"; do (( index % 3 == 0 )) && quick_tasks+=("${bench_tasks[$index]}"); done
      bench_tasks=("${quick_tasks[@]}") bench_reps=1
    fi
    run_batch bench run_bench_arms "${bench_tasks[@]}" || failed=1
  else
    echo "no valid $bench_dir/run_bench.py (missing or --validate fails): bench skipped" | tee -a "$out/batches.log"
  fi
fi

{
  echo "# cbm A/B $(date '+%Y-%m-%d %H:%M') (quick=$quick)"
  if [[ -f "$out/eval/results.jsonl" ]]; then
    echo; echo "## eval (nojev = production, cbm = production + DSH_CBM=1)"
    python3 -I "$STACK_ROOT/splash/tools/cbm_ab_report.py" "$out/eval/results.jsonl" --out-json "$out/eval/report.json"
  fi
  bench_results=()
  for arm in baseline cbm; do [[ -f "$out/bench/$arm/results.jsonl" ]] && bench_results+=("$out/bench/$arm/results.jsonl"); done
  if (( ${#bench_results[@]} )); then
    echo; echo "## platform bench (cbmab-baseline vs cbmab-cbm)"
    python3 -I "$STACK_ROOT/splash/tools/cbm_ab_report.py" "${bench_results[@]}" --out-json "$out/bench/report.json"
  fi
} > "$out/summary.md"
cat "$out/summary.md"
# Verdict: the A/B fails only when the CBM arm scores worse than the baseline on the bench (a task failing in both
# arms, or only in the baseline, is a property of the task, not of CBM). Runtime errors (exit 2+) still fail.
if [[ -f "$out/bench/report.json" ]]; then
  if python3 -I - "$out/bench/report.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))["summary"]
score = {cond: float(row.get("score_pct", 0)) for cond, row in summary.items()}
cbm, base = score.get("cbmab-cbm"), score.get("cbmab-baseline")
print(f"verdict: cbm {cbm}% vs baseline {base}%")
sys.exit(0 if cbm is not None and base is not None and cbm >= base - 5 else 1)
PY
  then failed=0; else failed=1; fi
fi
exit $failed

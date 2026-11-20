#!/usr/bin/env bash
# Long-context profile for PLAN-2026-11-17 item 2 step 1 (prefill) and item 6 step 1 (DSpine draft), unattended, on
# the default engine's settings (dev-longctx: DSpine tree + copy + shortlist, turbo-int4 KV, 8 verify rows from 64K).
# Built and run from splash/dev-prefill (= dev-longctx + profiling hooks that do nothing unless set; PREFILL.md).
# Nothing in dev-longctx, dev-park2, dev-dspine or splash-src is touched. Run it from your own terminal.
#
#   splash/tools/longctx_profile.sh OUTDIR
#   PROFILE_DRY_RUN=1 splash/tools/longctx_profile.sh OUTDIR   # CPU only: builds, CPU tests, argument and prompt
#                                                              # checks (--plan-only), a synthetic summary; no GPU
#
# Steps (a failing step marks the run failed; later independent steps still run):
#   1 build        dev-prefill: make all + longctx-profile + the 4 CPU engine tests (2048-row chunks, the default);
#                  the 4096-row-chunk variant into dev-prefill/build-chunk4096 (make PREFILL_TOKEN_BUDGET=4096)
#   2 cpu_tests    kv-first-engine, runtime-status, native-engine-loop, native-fd-transport (hooks off = dev-longctx)
#   3 prompts      ~/.cache/splash-dspine/prompts/long.txt (line 0: 67,695 tokens) and long100k.txt (102,643; made
#                  by splash/tools/longctx_prompts.py when missing). No company content.
#   4 p100k        long100k.txt, 2048-row chunks: every chunk's fused GPU time; the chunks at 20480, 59392 and 100352
#                  replayed dispatch by dispatch (projections / GDN / attention per layer / draft context / other);
#                  then decode at ~103K: 16-row draft (8 warm-up, 32 fused, 8 profiled cycles), 8-row draft
#                  (SPLASH_PROFILE_DRAFT_ROWS=8: 4 + 32 + 8), 16-row again (4 + 32)
#   5 p70k         long.txt line 0, the same at ~68K (chunk at 59392 profiled for a second ~60K sample)
#   6 chunk4096    long100k.txt with 4096-row chunks (build-chunk4096), prefill only, the chunk at 98304 profiled
#   7 report       splash/tools/longctx_profile_report.py: summary.md / summary.json with both tables, the
#                  4096-vs-2048 comparison and the verdict lines ("item 2: proceed|skip"; item 6a rows 8-15 share)
# GPU rules (as longctx_ab.sh): AC power (exit 3 on battery at the start), /tmp/splash-gpu.lock for the GPU steps, a
# Splash server this script did not start is never stopped (it waits in 5-min polls, PROFILE_WAIT_MINUTES, default
# 360; a run during which one came up is repeated once), >= PROFILE_MIN_GIB (13) GiB reclaimable before each run
# (waits up to 10 min, splash/tools/hostavail.py), Defender under 10% for 30 s before each run (waits up to 3 min).
# Expected time: ~1-2 min build (incremental), GPU ~19 min (prefill dominates: ~6.5 min per 100K prompt, ~4 min for
# 68K, ~0.5 min decode per prompt), plus any memory or Defender waits.
# Outputs: summary.md, summary.json, runs/<run>/ (chunks.csv, prefill_dispatch_*.csv, decode_cycles.csv,
# decode_dispatch_*.csv, meta.json), logs/*.log. Re-run the summary alone: longctx_profile_report.py OUTDIR.
# Exit: 0 when every step passes, 1 otherwise, 3 on battery power.
set -uo pipefail
OUT=${1:?usage: longctx_profile.sh OUTDIR}
DRY=${PROFILE_DRY_RUN:-0}
mkdir -p "$OUT/logs" "$OUT/runs"
OUT=$(cd "$OUT" && pwd)
STACK_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
S=$STACK_ROOT/splash
DEV=$S/dev-prefill
BUILD_2048=$DEV/build
BUILD_4096=$DEV/build-chunk4096
MODEL=$S/splash-src/install/models/.selections/fb0aa6e23bdecc7d24023adea3bcae72062d1c103ab1578902b00a2ebb3f500f
PROMPTS=${PROFILE_PROMPTS:-$HOME/.cache/splash-dspine/prompts}
PY=/opt/homebrew/bin/python3
[[ -x $PY ]] || PY=python3
export PYTHONDONTWRITEBYTECODE=1
source "$STACK_ROOT/scripts/lib/system.sh"
if ! on_ac_power; then
  if (( DRY )); then echo "longctx_profile: on battery power (dry run: going on)" >&2
  else echo "longctx_profile: on battery power, refusing to run" >&2; exit 3; fi
fi
WAIT_MINUTES=${PROFILE_WAIT_MINUTES:-360}
MIN_GIB=${PROFILE_MIN_GIB:-13}
# The default engine's switches (config/stack.env + LONGCTX.md); SPLASH_PROFILE_SPLIT=1 only arms the 8-row draft
# hook, which longctx-profile switches on for its rows8 arm alone.
ENGINE_ENV=(SPLASH_DRAFTER=dspine SPLASH_DRAFT_TREE=1 SPLASH_DRAFT_HEAD_ROWS=98304 SPLASH_DRAFT_COPY=1
            SPLASH_VERIFY_ROWS_LONG=8 SPLASH_VERIFY_ROWS_LONG_FROM=65536 SPLASH_PROFILE_SPLIT=1)
DECODE_ARGS=(--decode-cycles 32 --decode-profile-cycles 8 --draft-rows-compare 1 --temperature 0.6 --seed 1)

FAILED=0
record() { [[ $2 == fail ]] && FAILED=1; echo "$(date '+%H:%M:%S') step $1: $2${3:+ ($3)}" | tee -a "$OUT/logs/steps.log"; }
note() { echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT/logs/steps.log"; }
server_up() { curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null; }
LOCK_HELD=0
take_lock() { until mkdir /tmp/splash-gpu.lock 2>/dev/null; do sleep 10; done; LOCK_HELD=1; }
release_lock() { (( LOCK_HELD )) && rmdir /tmp/splash-gpu.lock 2>/dev/null; LOCK_HELD=0; }
trap 'release_lock' EXIT
trap 'exit 130' INT TERM HUP
GPU_GAVE_UP=0
# A Splash server this script did not start is the user's: wait for it to go (it idle-stops after the session).
wait_for_server() {
  local waited=0
  while server_up; do
    (( waited == 0 )) && note "a Splash server this script did not start is up; waiting"
    if (( waited >= WAIT_MINUTES * 60 )); then note "still up after $WAIT_MINUTES min; giving up the GPU steps"; GPU_GAVE_UP=1; return 1; fi
    sleep 300; waited=$((waited + 300))
  done
}
wait_for_memory() {  # macOS returns a finished model process's memory slowly
  local waited=0 avail
  while (( waited < 600 )); do
    avail=$(reclaimable_gib); [[ "$avail" =~ ^[0-9]+$ ]] && (( avail >= MIN_GIB )) && return 0
    sleep 15; waited=$((waited + 15))
  done
  note "only $(reclaimable_gib) GiB reclaimable after 10 min; going on"
}
defender_wait() {  # every run is timed: wait up to 3 min for Defender under 10% for 30 s (ANALYSIS.md 10.1)
  local waited=0 calm=0
  while (( waited < 180 && calm < 3 )); do
    if (( $(defender_cpu | cut -d. -f1) < 10 )); then calm=$((calm + 1)); else calm=0; fi
    sleep 10; waited=$((waited + 10))
  done
  echo "$(date '+%H:%M:%S') $1 defender_cpu=$(defender_cpu) waited=${waited}s" >> "$OUT/logs/defender.log"
}
# profile NAME BUILD_DIR PROMPT_FILE LINE [longctx-profile args...]: one run into runs/NAME, log logs/NAME.log.
# A run during which the user's server came up is repeated once after it goes. PROFILE_DRY_RUN=1: --plan-only (the
# binary checks its arguments and the prompt and exits before opening the Metal device).
profile() {
  local name=$1 build=$2 prompts=$3 line=$4; shift 4
  local cmd=("$build/engine-tests/longctx-profile" "$build/splash.metallib" "$MODEL" "$prompts" --line "$line"
             --out "$OUT/runs/$name" "$@")
  echo "${ENGINE_ENV[*]} ${cmd[*]}" >> "$OUT/logs/gpu-commands.txt"
  if (( DRY )); then
    (cd "$DEV" && env "${ENGINE_ENV[@]}" "${cmd[@]}" --plan-only 1) > "$OUT/logs/$name.plan.log" 2>&1
    return $?
  fi
  local attempt rc=1
  for attempt in 1 2; do
    if (( GPU_GAVE_UP )) || ! wait_for_server; then return 1; fi
    wait_for_memory
    defender_wait "$name"
    rm -rf "$OUT/runs/$name"
    note "$name: start ($(reclaimable_gib) GiB reclaimable)"
    (cd "$DEV" && env -u SPLASH_PROFILE_DRAFT_ROWS -u SPLASH_DSPINE_WINDOW -u SPLASH_VERIFY_ROWS_LONG_KEEP_MASS \
       -u SPLASH_VERIFY_ROWS_LONG_KEEP_MASS_PER_100K "${ENGINE_ENV[@]}" /usr/bin/time -l "${cmd[@]}") \
      > "$OUT/logs/$name.log" 2>&1
    rc=$?
    note "$name: exit $rc"
    if server_up; then
      note "$name: a foreign Splash server came up during the run; repeating it once it goes"
      mv "$OUT/logs/$name.log" "$OUT/logs/$name.interrupted$attempt.log"
      continue
    fi
    return $rc
  done
  return $rc
}
echo "longctx_profile $(date) dry_run=$DRY out=$OUT" > "$OUT/logs/steps.log"
pmset -g batt | head -1 >> "$OUT/logs/steps.log"
: > "$OUT/logs/gpu-commands.txt"

# ---- 1 build (dev-prefill only; the 4096 variant in its own build directory)
TESTS=(longctx-profile kv-first-engine runtime-status native-engine-loop native-fd-transport)
if (cd "$DEV" && make all && make "${TESTS[@]/#/build/engine-tests/}") > "$OUT/logs/build.log" 2>&1 \
   && (cd "$DEV" && make PREFILL_TOKEN_BUDGET=4096 BUILD=build-chunk4096 all build-chunk4096/engine-tests/longctx-profile) \
        > "$OUT/logs/build-chunk4096.log" 2>&1
then record build pass; else record build fail; fi

# ---- 2 cpu tests
ok=pass
for t in kv-first-engine runtime-status native-engine-loop native-fd-transport; do
  if ! (cd "$DEV" && build/engine-tests/$t) > "$OUT/logs/cpu-$t.log" 2>&1; then ok=fail; fi
done
record cpu_tests $ok

# ---- 3 prompts
ok=pass
if [[ ! -s $PROMPTS/long100k.txt ]]; then
  $PY "$S/tools/longctx_prompts.py" "$PROMPTS/long100k.txt" --targets 100000 --first-task 2 --more-sources \
    >> "$OUT/logs/prompts.log" 2>&1 || ok=fail
fi
[[ -s $PROMPTS/long.txt && -s $PROMPTS/long100k.txt ]] || ok=fail
awk '{print FILENAME " line " FNR - 1 ": " NF " tokens"}' "$PROMPTS/long.txt" "$PROMPTS/long100k.txt" \
  >> "$OUT/logs/steps.log" 2>/dev/null
record prompts $ok

# ---- GPU steps
if (( DRY )); then
  note "dry run: no GPU (no lock, no server wait; reclaimable $(reclaimable_gib) GiB, defender $(defender_cpu)%)"
  server_up && note "dry run: a Splash server is up on :8000 (a real run would wait for it)"
fi
if (( ! DRY )) && ! wait_for_server; then
  for step in p100k p70k chunk4096; do record $step skip "GPU busy with the user's server"; done
else
  (( DRY )) || take_lock
  if profile p100k "$BUILD_2048" "$PROMPTS/long100k.txt" 0 --profile-at 20480,59392,100352 "${DECODE_ARGS[@]}"; then
    record p100k pass; else record p100k fail "logs/p100k.log"; fi
  if profile p70k "$BUILD_2048" "$PROMPTS/long.txt" 0 --profile-at 59392 "${DECODE_ARGS[@]}"; then
    record p70k pass; else record p70k fail "logs/p70k.log"; fi
  if profile p100k_chunk4096 "$BUILD_4096" "$PROMPTS/long100k.txt" 0 --profile-at 98304 \
       --decode-cycles 0 --decode-profile-cycles 0; then
    record chunk4096 pass; else record chunk4096 fail "logs/p100k_chunk4096.log"; fi
  release_lock
fi

# ---- 7 report (a dry run writes synthetic runs first, clearly marked in summary.md)
if (( DRY )); then report_args=(--fake); else report_args=(); fi
if $PY -I "$S/tools/longctx_profile_report.py" "$OUT" ${report_args[@]+"${report_args[@]}"} > "$OUT/logs/report.log" 2>&1
then record report pass; else record report fail "logs/report.log"; fi
# The step table in summary.md is read from steps.log, so write it once more with the report step included.
$PY -I "$S/tools/longctx_profile_report.py" "$OUT" > /dev/null 2>&1
cat "$OUT/summary.md" 2>/dev/null
exit $FAILED

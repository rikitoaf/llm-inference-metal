#!/usr/bin/env bash
# Unattended test runner: runs every pending check that can run without a person and writes results to files for
# later analysis. Safe by default: CPU checks only unless --gpu is given (only GPU jobs start Splash, under the lock); never
# touches production splash/splash-src, never changes system settings.
#
#   scripts/auto-tests.sh [--gpu] [--quick] [--out DIR]
#     --gpu    also run the GPU jobs (DSpine stage 2 checks + A/B, park lane, long-context verify rows,
#              long-context profile, prefill attention v2 A/B, codebase-memory-mcp A/B); needs AC power, takes the
#              GPU lock per job
#     --quick  pass --quick to the GPU jobs (1 seed, 3 prompts)
#     --out    results directory (default logs/auto-tests/<timestamp>)
#
# Output: OUT/summary.md (one line per job: PASS / FAIL / SKIP + reason), OUT/<job>.log, plus each job's own files.
# Exit: 0 if no job failed (skips are fine), 1 otherwise.
set -uo pipefail
STACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$STACK_ROOT/scripts/lib/system.sh"

run_gpu=0 quick=()
out="$STACK_ROOT/logs/auto-tests/$(date +%Y%m%d-%H%M%S)"
while (( $# )); do
  case "$1" in
    --gpu) run_gpu=1 ;;
    --quick) quick=(--quick) ;;
    --out) out="$2"; shift ;;
    *) echo "usage: $0 [--gpu] [--quick] [--out DIR]" >&2; exit 2 ;;
  esac
  shift
done
mkdir -p "$out"
summary="$out/summary.md"
echo "# auto-tests $(date '+%Y-%m-%d %H:%M')  (gpu=$run_gpu ${quick[*]:-})" > "$summary"
failed=0

record() {  # record JOB STATUS DETAIL
  echo "- **$1**: $2 — $3" >> "$summary"
  echo "[$2] $1: $3"
  [[ "$2" == FAIL ]] && failed=1
}

run_job() {  # run_job JOB command...   (stdout+stderr to OUT/JOB.log)
  local job="$1"; shift
  if "$@" > "$out/$job.log" 2>&1; then record "$job" PASS "log: $job.log"
  else record "$job" FAIL "exit $?, log: $job.log"; fi
}

export PYTHONDONTWRITEBYTECODE=1

# 1. Script syntax (CPU)
run_job shell-syntax bash -c "for f in '$STACK_ROOT'/scripts/*.sh '$STACK_ROOT'/scripts/lib/*.sh; do bash -n \"\$f\" || exit 1; done"

# 2. Tool-result share of dsh sessions: the "before" baseline for the codebase-memory-mcp trial (CPU)
run_job session-tool-share python3 -I "$STACK_ROOT/splash/tools/session_tool_share.py" \
  --out-json "$out/session-tool-share.json" --out-md "$out/session-tool-share.md"

# 3. Compaction settings (CPU): dynamic = (window - 10000) / window; DSH_SHORT_CONTEXT=1 = the fixed early threshold
run_job shortctx-toggle bash -c "
  cd '$STACK_ROOT'
  dynamic=\$(bash -c 'source scripts/dsh-env.sh >/dev/null 2>&1; compaction_settings')
  fixed=\$(DSH_SHORT_CONTEXT=1 bash -c 'source scripts/dsh-env.sh >/dev/null 2>&1; compaction_settings')
  echo \"dynamic: \$dynamic | short: \$fixed\"
  [[ \"\$fixed\" == '0.37 12000' ]] && python3 -c 'import sys; r=float(sys.argv[1]); sys.exit(0 if 0.5 <= r < 1 else 1)' \$dynamic"

# 3a. dsh plugin unit tests (CPU): the fast-search plugins load, the graph-first reminder (turn / drift / scope /
#     batch reminders) and the CBM result filter proxy behave as specified.
run_job plugin-unit-tests bash -c "
  cd '$STACK_ROOT' && source config/stack.env && export PATH=\"\$NODE_BIN:\$PATH\"
  for plugin in plugins/dsh-plugin-fast-search/*.mjs; do node -e \"import('./\$plugin').then(() => console.log('loads: \$plugin'))\" || exit 1; done
  node scripts/test-graph-first-reminder.mjs && node scripts/test-cbm-mcp-filter.mjs"

# 3b. Platform benchmark task files and graders (CPU). The benchmark lives in the gitignored
#     workspace/platform-bench/ (company-internal content); skipped on a checkout without it.
platform_bench="$STACK_ROOT/workspace/platform-bench/run_bench.py"
if [[ -f "$platform_bench" ]]; then
  run_job platform-bench-validate python3 "$platform_bench" --validate
else
  record platform-bench-validate SKIP "workspace/platform-bench not present (gitignored, local only)"
fi

# 3c. Platform benchmark quick run (GPU): 3 tasks x 1 rep through dsh headless. --serve takes /tmp/splash-gpu.lock,
#     starts Splash, and stops it and releases the lock afterwards. Skipped when Splash cannot be started.
platform_bench_quick_tasks=(spi-02-tenant-key-resolution tpt-01-export-row-caps x-10-holding-table-trace)
if (( ! run_gpu )); then
  record platform-bench-quick SKIP "GPU jobs need --gpu"
elif [[ ! -f "$platform_bench" ]]; then
  record platform-bench-quick SKIP "workspace/platform-bench not present"
elif ! on_ac_power; then
  record platform-bench-quick SKIP "on battery; GPU measurements need AC power"
elif [[ ! -x "$STACK_ROOT/scripts/serve-splash.sh" ]] || { [[ ! -d "${SPLASH_SRC:-$STACK_ROOT/splash/splash-src}" ]] && ! command -v splash >/dev/null; }; then
  record platform-bench-quick SKIP "Splash is not installed"
else
  if python3 "$platform_bench" --serve --lock-wait 1800 --label auto-quick --out "$out/platform-bench-quick" \
       --tasks "${platform_bench_quick_tasks[@]}" > "$out/platform-bench-quick.log" 2>&1; then
    record platform-bench-quick PASS "log: platform-bench-quick.log"
  else
    status=$?
    if (( status == 4 )); then record platform-bench-quick SKIP "Splash could not be started (log: platform-bench-quick.log)"
    elif (( status == 5 )); then record platform-bench-quick SKIP "GPU lock busy for 30 min"
    else record platform-bench-quick FAIL "exit $status, log: platform-bench-quick.log"; fi
  fi
  [[ -f "$out/platform-bench-quick/summary.md" ]] && { echo; cat "$out/platform-bench-quick/summary.md"; } >> "$summary"
fi

# 4. DSpine stage 2: correctness + A/B (GPU). The script is written by the DSpine port (splash/dev-dspine/DSPINE-PORT.md).
dspine_ab="$STACK_ROOT/splash/tools/dspine/ab_dspine.sh"
if (( ! run_gpu )); then
  record dspine-ab SKIP "GPU jobs need --gpu"
elif [[ ! -x "$dspine_ab" ]]; then
  record dspine-ab SKIP "$dspine_ab not written yet (DSpine port in progress)"
elif ! on_ac_power; then
  record dspine-ab SKIP "on battery; GPU measurements need AC power"
else
  echo "Defender CPU before dspine-ab: $(defender_cpu 2>/dev/null || echo '?')%" >> "$summary"
  run_job dspine-ab "$dspine_ab" ${quick[@]+"${quick[@]}"} "$out/dspine-ab"
  [[ -f "$out/dspine-ab/summary.md" ]] && { echo; echo "## dspine-ab"; cat "$out/dspine-ab/summary.md"; } >> "$summary"
fi

# 4b. Park lane (SPLASH_PARK_LANE, splash/dev-park2/PARK-PORT.md): build, CPU tests, the GPU oracle, flag-off regression
#     and the long-context A/B. The script waits for the user's own Splash server to go instead of stopping it.
park_ab="$STACK_ROOT/splash/tools/park_ab.sh"
if (( ! run_gpu )); then
  record park-ab SKIP "GPU jobs need --gpu"
elif [[ ! -x "$park_ab" ]]; then
  record park-ab SKIP "$park_ab not written yet"
elif ! on_ac_power; then
  record park-ab SKIP "on battery; GPU measurements need AC power"
else
  echo "Defender CPU before park-ab: $(defender_cpu 2>/dev/null || echo '?')%" >> "$summary"
  run_job park-ab "$park_ab" ${quick[@]+"${quick[@]}"} "$out/park-ab"
  [[ -f "$out/park-ab/summary.md" ]] && { echo; echo "## park-ab"; cat "$out/park-ab/summary.md"; } >> "$summary"
fi

# 4c. Long-context verify rows (SPLASH_VERIFY_ROWS_LONG, splash/dev-longctx/LONGCTX.md): build, CPU tests, greedy
#     digests vs dev-park2, near-tie and distribution checks with 8 rows, and the A/B at ~70K and ~100K. Waits for the
#     user's own Splash server to go instead of stopping it.
longctx_ab="$STACK_ROOT/splash/tools/longctx_ab.sh"
if (( ! run_gpu )); then
  record longctx-ab SKIP "GPU jobs need --gpu"
elif [[ ! -x "$longctx_ab" ]]; then
  record longctx-ab SKIP "$longctx_ab not written yet"
elif ! on_ac_power; then
  record longctx-ab SKIP "on battery; GPU measurements need AC power"
else
  echo "Defender CPU before longctx-ab: $(defender_cpu 2>/dev/null || echo '?')%" >> "$summary"
  run_job longctx-ab "$longctx_ab" ${quick[@]+"${quick[@]}"} "$out/longctx-ab"
  [[ -f "$out/longctx-ab/summary.md" ]] && { echo; echo "## longctx-ab"; cat "$out/longctx-ab/summary.md"; } >> "$summary"
fi

# 4d. Long-context profile (PLAN-2026-11-17 items 2 and 6 step 1, splash/dev-prefill/PREFILL.md): prefill split at
#     ~20K/60K/100K, 4096- vs 2048-row chunks, decode split at ~70K/100K with the draft's rows 8-15; verdict lines in
#     its summary.md. Measurement only (no A/B, no --quick variant); waits for the user's own Splash server to go.
longctx_profile="$STACK_ROOT/splash/tools/longctx_profile.sh"
if (( ! run_gpu )); then
  record longctx-profile SKIP "GPU jobs need --gpu"
elif [[ ! -x "$longctx_profile" ]]; then
  record longctx-profile SKIP "$longctx_profile not written yet"
elif ! on_ac_power; then
  record longctx-profile SKIP "on battery; GPU measurements need AC power"
else
  echo "Defender CPU before longctx-profile: $(defender_cpu 2>/dev/null || echo '?')%" >> "$summary"
  run_job longctx-profile "$longctx_profile" "$out/longctx-profile"
  [[ -f "$out/longctx-profile/summary.md" ]] && { echo; echo "## longctx-profile"; cat "$out/longctx-profile/summary.md"; } >> "$summary"
fi

# 4e. Prefill attention v2 (SPLASH_PREFILL_ATTN_V2, splash/dev-prefill-attn/ATTN.md): build, CPU tests, greedy digests
#     vs dev-longctx, cold-prefill logits within the prefill-partition control, near-tie and distribution checks,
#     the kernel alone, and the TTFT A/B (2K/4K suffixes at ~30K/60K/100K, park_long turns). Waits for the user's own
#     Splash server to go instead of stopping it.
prefill_ab="$STACK_ROOT/splash/tools/prefill_ab.sh"
if (( ! run_gpu )); then
  record prefill-ab SKIP "GPU jobs need --gpu"
elif [[ ! -x "$prefill_ab" ]]; then
  record prefill-ab SKIP "$prefill_ab not written yet"
elif ! on_ac_power; then
  record prefill-ab SKIP "on battery; GPU measurements need AC power"
else
  echo "Defender CPU before prefill-ab: $(defender_cpu 2>/dev/null || echo '?')%" >> "$summary"
  run_job prefill-ab "$prefill_ab" ${quick[@]+"${quick[@]}"} "$out/prefill-ab"
  [[ -f "$out/prefill-ab/summary.md" ]] && { echo; echo "## prefill-ab"; cat "$out/prefill-ab/summary.md"; } >> "$summary"
fi

# 5. codebase-memory-mcp (EXTERNAL section 3). CPU: DSH_CBM toggles only the mcp-cbm row. GPU: the quick A/B
#    (eval t1-t3 x 1 rep and the platform bench's quick set, production vs DSH_CBM=1), splash/tools/cbm_ab.sh.
run_job cbm-toggle bash -c "
  cd '$STACK_ROOT'
  off=\$(DSH_CBM=0 bash -c 'source scripts/dsh-env.sh >/dev/null; printf \"%s\n\" \"\${DSH_PATCHES[@]}\"')
  on=\$(DSH_CBM=1 bash -c 'source scripts/dsh-env.sh >/dev/null; printf \"%s\n\" \"\${DSH_PATCHES[@]}\"')
  ! grep -q cbm.cordis <<<\"\$off\" && grep -q cbm.cordis <<<\"\$on\""
cbm_binary="$HOME/.local/bin/codebase-memory-mcp"
if (( ! run_gpu )); then
  record cbm-trial SKIP "GPU jobs need --gpu"
elif [[ ! -x "$cbm_binary" ]]; then
  record cbm-trial SKIP "$cbm_binary not installed (build from source: splash/EXTERNAL-2026-11-09.md section 3)"
elif ! on_ac_power; then
  record cbm-trial SKIP "on battery; GPU measurements need AC power"
else
  echo "Defender CPU before cbm-trial: $(defender_cpu 2>/dev/null || echo '?')%" >> "$summary"
  run_job cbm-trial "$STACK_ROOT/splash/tools/cbm_ab.sh" --quick "$out/cbm-trial"
  [[ -f "$out/cbm-trial/summary.md" ]] && { echo; echo "## cbm-trial"; cat "$out/cbm-trial/summary.md"; } >> "$summary"
fi

echo "results: $summary"
exit $failed

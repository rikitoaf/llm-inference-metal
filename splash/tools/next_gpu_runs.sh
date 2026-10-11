#!/usr/bin/env bash
# The GPU runs still owed by splash/IMPROVEMENT_PLAN.md, in one detached job (~2 h). Run it when
# nobody needs the Mac: it restarts Splash (and stops llama-server) several times, and each Splash
# start needs ~16 GiB reclaimable (python3 splash/tools/hostavail.py).
#
#   nohup splash/tools/next_gpu_runs.sh > logs/next-gpu-runs.log 2>&1 & disown
#
# 0. Step 5 (turbo's Lloyd-Max key codebook, splash/staged/step5) goes in only if
#    `make -C splash/tools test` passes with it; otherwise it is put back and skipped.
# 1. Quality A/B on judgment set v2 (60K-token context, planted facts) and greedy outputs, with
#    byte-identical prompts (no nonce): int8 twice (determinism / noise floor), turbo-int4 with the
#    new quantizer (r2) and the old one (r1), turbo with the new codebook and the fork's.
# 2. Step 3's 2x2: format x --persistent-cache on needles at 16K/58K (prompts with a nonce).
# 3. Attention kernel timing, then logs/kvbench/summary-next.txt with every comparison.
# A trap leaves the tree with the new turbo-int4 quantizer, and step 5 if its tests passed.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SRC=splash/splash-src
STORE=$SRC/runtime/metal/kernels/common/paged_store_row.h
TURBOKV=$SRC/runtime/metal/kernels/common/turbo_kv.h
KVHPP=$SRC/runtime/ops/PagedKv.hpp
STEP5=splash/staged/step5
KEEP=$(mktemp -d)

if pgrep -f "run_kv_suite|serve-native" >/dev/null; then echo "error: a suite or server is running" >&2; exit 1; fi
echo "=== $(date '+%H:%M:%S') start; $(python3 splash/tools/hostavail.py)"

# KEEP holds the state to leave behind: the tree as found, and step 5 once its tests pass.
cp "$STORE" "$TURBOKV" "$KVHPP" "$KEEP/"
restore() { cp "$KEEP/paged_store_row.h" "$STORE"; cp "$KEEP/turbo_kv.h" "$TURBOKV"; cp "$KEEP/PagedKv.hpp" "$KVHPP"; }
trap 'restore; ./scripts/stop.sh >/dev/null 2>&1' EXIT INT TERM

# --- 0. step 5, kept only when the kernel tests pass with it
cp "$STEP5/turbo_kv.h" "$TURBOKV"; cp "$STEP5/PagedKv.hpp" "$KVHPP"
if make -C splash/tools test; then
  step5=1; echo "step 5 kernel tests: PASS"; cp "$TURBOKV" "$KVHPP" "$KEEP/"
else
  step5=0; echo "step 5 kernel tests: FAIL; the fork codebook stays"; restore
fi

# Force one format's cache revision to 1 in PagedKv.hpp (either form of quantizerRevision).
force_r1() {   # Turbo | TurboInt4
  local args=(-e "s/case Format::$1: return 2;/case Format::$1: return 1;/")
  [[ "$1" == TurboInt4 ]] && args+=(-e 's/return format == Format::TurboInt4 ? 2 : 1;/return 1;/')
  sed "${args[@]}" "$KEEP/PagedKv.hpp" > "$KVHPP"
  ! grep -qE "case Format::$1: return 2;|Format::$1 \? 2" "$KVHPP"
}

# --- 1. quality A/B
Q="--parts judgments greedy --judgment-tokens 60000"
splash/tools/run_kv_suite.sh <<PLAN
int8 i8-q $Q
int8 i8-q2 $Q
turbo-int4 i4-r2-q $Q
turbo turbo-q $Q
PLAN

echo "=== $(date '+%H:%M:%S') old turbo-int4 quantizer (absmax / 7, cache revision 1)"
cp splash/staged/orig/paged_store_row.h "$STORE"
if force_r1 TurboInt4; then
  echo "turbo-int4 i4-r1-q $Q" | splash/tools/run_kv_suite.sh
else
  echo "error: could not force turbo-int4 revision 1; skipped"
fi
./scripts/stop.sh; restore

if (( step5 )); then
  echo "=== $(date '+%H:%M:%S') fork turbo codebook (cache revision 1)"
  cp "$STEP5/orig/turbo_kv.h" "$TURBOKV"
  if force_r1 Turbo; then
    echo "turbo turbo-fork-q $Q" | splash/tools/run_kv_suite.sh
  else
    echo "error: could not force turbo revision 1; skipped"
  fi
  ./scripts/stop.sh; restore
fi

# --- 2. format x --persistent-cache
splash/tools/run_kv_suite.sh <<'PLAN'
int8 i8-pc0 ENV=SPLASH_PERSISTENT_CACHE=0 --parts speed needles --needle-tokens 16000 60000
turbo-int4 i4-pc1 ENV=SPLASH_PERSISTENT_CACHE=1 --parts speed needles --needle-tokens 16000 60000
turbo-int4 i4-pc0 ENV=SPLASH_PERSISTENT_CACHE=0 --parts speed needles --needle-tokens 16000 60000
int8 i8-pc1 ENV=SPLASH_PERSISTENT_CACHE=1 --parts speed needles --needle-tokens 16000 60000
PLAN
./scripts/stop.sh

# --- 3. kernel timing and the summary
make -C splash/tools perf H="16384 57344" > logs/kvbench/perf-next.txt 2>&1
{
  echo "step 5 (turbo Lloyd-Max key codebook): $([[ $step5 == 1 ]] && echo applied || echo 'tests failed, not applied')"
  for pair in "i8-q i8-q2" "i8-q i4-r2-q" "i8-q i4-r1-q" "i4-r1-q i4-r2-q" "i8-q turbo-q" "i8-q turbo-fork-q" \
              "turbo-fork-q turbo-q" "i8-pc0 i8-pc1" "i4-pc0 i4-pc1" "i8-pc0 i4-pc0" "i8-pc1 i4-pc1"; do
    set -- $pair
    [[ -f "logs/kvbench/$1.json" && -f "logs/kvbench/$2.json" ]] || continue
    echo "===== $1 vs $2"
    python3 splash/tools/kvbench.py compare "$1" "$2"
  done
  echo "===== attention kernels"; grep layer logs/kvbench/perf-next.txt
} > logs/kvbench/summary-next.txt 2>&1
echo "=== $(date '+%H:%M:%S') done; see logs/kvbench/summary-next.txt"

#!/usr/bin/env bash
# One GPU round for the kernel candidates of splash/SPEED_PLAN.md, in the order that keeps the timings clean: every
# file is written first (libraries, test binaries), then correctness, then a wait until Microsoft Defender has been
# quiet for 30 s, then all timings. Needs AC power and no model server; uses the whole GPU for ~30-40 min.
#
#   splash/staged/speed/gpu_round.sh                                  default candidate set
#   LIBS="base grid-v2 grid-v3+p2" splash/staged/speed/gpu_round.sh   your own ("+"-joined parts, see kernel_ab.sh)
#
# Steps: (1) build the libraries and test binaries; (2) gguf-projection correctness on every candidate; (3) wait
# quiet; (4) the projection microbenchmark (kernel_ab.sh); (5) decode-profile of the real model with each library
# (one decode cycle replayed, per-kernel GPU ms); (6) the split-policy check, if split_policy_check.sh exists.
# Output: logs/speed/round-<date>/ with a summary.txt.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
HERE=splash/staged/speed
source "$HERE/quiet.sh"
SRC=splash/splash-src
CACHE="${TMPDIR:-/tmp}/splash-kernel-ab"
LIBS="${LIBS:-base grid grid-v2 grid-v3 p2}"
OUT="logs/speed/round-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"
log() { echo "=== $(date '+%H:%M:%S') $*" | tee -a "$OUT/summary.txt"; }

require_measurable || exit 1
# Only the candidates whose patches exist.
available=""
for lib in $LIBS; do
  ok=1
  IFS=+ read -r -a parts <<< "$lib"
  for part in "${parts[@]}"; do
    case "$part" in base|p[0-9]|grid) ;; *) [[ -f "$HERE/$part.patch" ]] || ok=0 ;; esac
  done
  (( ok )) && available="$available $lib" || log "skipping $lib: a patch is missing"
done
LIBS="${available# }"
log "libraries: $LIBS; $(python3 splash/tools/hostavail.py)"

# --- 1. every file first
log "build"
(cd "$SRC" && make build/engine-tests/gguf-projection build/engine-tests/gguf-projection-benchmark \
   build/engine-tests/decode-profile) > "$OUT/build.log" 2>&1 || { log "build failed, see $OUT/build.log"; exit 1; }
"$HERE/kernel_ab.sh" --build-only --libs "$LIBS" >> "$OUT/build.log" 2>&1 || { log "library build failed"; exit 1; }
MODEL_ROOT=$(cd "$SRC" && .venv/bin/python install/models.py --model unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS --language-only link)

# --- 2. correctness (Defender does not matter here)
log "correctness"
"$HERE/correctness.sh" $LIBS 2>&1 | tee -a "$OUT/summary.txt"
passing=""
for lib in $LIBS; do grep -q "^$lib: PASS" "$OUT/summary.txt" && passing="$passing $lib"; done
LIBS="${passing# }"
[[ " $LIBS " == *" base "* ]] || { log "base failed its correctness test; stopping"; exit 1; }
log "timing: $LIBS"

# --- 3-4. quiet, then the microbenchmark
if ! "$HERE/kernel_ab.sh" --libs "$LIBS" > "$OUT/kernel-ab.summary" 2>&1; then
  log "microbenchmark not run: $(tail -1 "$OUT/kernel-ab.summary")"
  log "stopping: timings taken while Defender scans are not trustworthy; rerun when it is quiet"
  exit 1
fi
cp logs/speed/kernel-ab.txt "$OUT/" 2>/dev/null
{ echo "--- projection microbenchmark (one lane, ms, change vs base)"; cat "$OUT/kernel-ab.summary"; } | tee -a "$OUT/summary.txt"

# --- 5. decode-profile: a real decode cycle per library
echo "--- decode-profile (B1 = one lane, 2,048-token prompt, median of 9 cycles)" | tee -a "$OUT/summary.txt"
for lib in $LIBS; do
  if ! wait_quiet 30 20 1200; then
    echo "$lib: skipped (Defender busy for 20 min)" | tee -a "$OUT/summary.txt"; continue
  fi
  "$SRC/build/engine-tests/decode-profile" "$CACHE/lib-$lib" "$MODEL_ROOT" --prompt-tokens 2048 --cycles 9 \
    > "$OUT/decode-profile-$lib.txt" 2>&1
  echo "$lib dfd=$(defender_cpu): $(grep -m1 '^B1 decode cycle' "$OUT/decode-profile-$lib.txt")" | tee -a "$OUT/summary.txt"
done

# --- 6. split policy
if [[ -x "$HERE/split_policy_check.sh" ]]; then
  log "split policy"
  "$HERE/split_policy_check.sh" > "$OUT/split-policy.txt" 2>&1
  tail -n 40 "$OUT/split-policy.txt" >> "$OUT/summary.txt"
fi
log "done; see $OUT/summary.txt"

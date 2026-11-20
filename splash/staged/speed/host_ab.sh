#!/usr/bin/env bash
# A/B of host-side (C++) candidates of splash/SPEED_PLAN.md: for each variant it applies the patches to splash-src,
# rebuilds the engine library and the test binaries, runs Splash's gguf-projection correctness test, waits until
# Microsoft Defender is quiet, and replays a real decode cycle of the model (decode-profile, widths 1-4). The source
# files are restored afterwards, whatever happens. ~4 min per variant on the GPU, plus builds.
#
#   splash/staged/speed/host_ab.sh                         variants: none m16-one-lane split-policy m16-one-lane+split-policy
#   VARIANTS="none m16-one-lane" splash/staged/speed/host_ab.sh
#   PROMPT_TOKENS=32768 CYCLES=5 VARIANTS="none verify-pages8" splash/staged/speed/host_ab.sh   long context (int8 KV)
#
# A variant is "none" or "+"-joined patch names in this directory (X means X.patch, applied to splash-src).
# Output: logs/speed/host-ab-<time>/ (summary.txt, one decode-profile per variant).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
HERE=splash/staged/speed
source "$HERE/quiet.sh"
SRC=splash/splash-src
CACHE="${TMPDIR:-/tmp}/splash-kernel-ab"
VARIANTS="${VARIANTS:-none m16-one-lane split-policy m16-one-lane+split-policy}"
OUT="logs/speed/host-ab-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"
log() { echo "=== $(date '+%H:%M:%S') $*" | tee -a "$OUT/summary.txt"; }

require_measurable || exit 1
# The files the host patches touch, saved once and restored on exit.
FILES=$(for v in $VARIANTS; do IFS=+ read -r -a ps <<< "$v"; for p in "${ps[@]}"; do
          [[ "$p" == none ]] || grep '^+++ ' "$HERE/$p.patch" | awk '{print $2}'; done; done | sort -u)
SAVE=$(mktemp -d)
for f in $FILES; do mkdir -p "$SAVE/$(dirname "$f")"; cp "$SRC/$f" "$SAVE/$f"; done
restore() { for f in $FILES; do cp "$SAVE/$f" "$SRC/$f"; done; }
trap restore EXIT INT TERM

"$HERE/kernel_ab.sh" --build-only --libs base > /dev/null || { log "base library build failed"; exit 1; }
MODEL_ROOT=$(cd "$SRC" && .venv/bin/python install/models.py --model unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS --language-only link)

for v in $VARIANTS; do
  restore
  IFS=+ read -r -a ps <<< "$v"
  for p in "${ps[@]}"; do
    [[ "$p" == none ]] && continue
    patch -s -p0 -d "$SRC" < "$HERE/$p.patch" || { log "$v: $p.patch does not apply"; continue 2; }
  done
  log "$v: build"
  (cd "$SRC" && make build/engine-tests/gguf-projection build/engine-tests/decode-profile) > "$OUT/build-$v.log" 2>&1 \
    || { log "$v: build failed (see $OUT/build-$v.log)"; continue; }
  # correctness.sh reruns the test binary just built against the unchanged kernel library.
  if ! "$HERE/correctness.sh" base > "$OUT/correctness-$v.txt" 2>&1; then
    log "$v: correctness FAIL"; tail -5 "$OUT/correctness-$v.txt" | tee -a "$OUT/summary.txt"; continue
  fi
  log "$v: correctness PASS"
  wait_quiet 30 30 1800 || { log "$v: skipped, Defender busy"; continue; }
  "$SRC/build/engine-tests/decode-profile" "$CACHE/lib-base" "$MODEL_ROOT" --prompt-tokens "${PROMPT_TOKENS:-2048}" --cycles "${CYCLES:-9}" \
    > "$OUT/decode-profile-$v.txt" 2>&1
  grep -E '^B[1-4] decode cycle' "$OUT/decode-profile-$v.txt" | sed "s/^/$v dfd=$(defender_cpu) | /" | tee -a "$OUT/summary.txt"
done
restore
# Leave the binaries matching the restored sources.
(cd "$SRC" && make build/engine-tests/gguf-projection build/engine-tests/decode-profile) >> "$OUT/build-restore.log" 2>&1
log "done; sources restored"

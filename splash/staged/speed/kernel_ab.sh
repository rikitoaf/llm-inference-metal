#!/usr/bin/env bash
# A/B of the staged GGUF decode-tile candidates (splash/SPEED_PLAN.md) on gguf-projection-benchmark. It builds the
# kernel libraries from the patches next to this script (base, prefetch depth 2 and 3, IQ3_XXS grid table + sign
# XOR), waits until Microsoft Defender has been quiet for 30 s (its scans take 20-45% off the GPU; our own builds
# start them), then times the shapes that carry the bytes at one lane, libraries alternating, two rounds. Each line
# records Defender's CPU at that moment. Output: logs/speed/kernel-ab.txt, summary on stdout.
#
#   splash/staged/speed/kernel_ab.sh                       all four libraries (~15 min of GPU)
#   splash/staged/speed/kernel_ab.sh --libs "base grid"    a subset; names are "+"-joined parts: base, pN (prefetch
#                                                          depth N), grid, or any X with X.patch here (grid-v2+p2)
#   splash/staged/speed/kernel_ab.sh --build-only          build and check the libraries, no GPU
#
# Libraries are cached in $TMPDIR/splash-kernel-ab and rebuilt only when Splash's runtime or a patch changes, so a
# rerun writes no new files for Defender to scan.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
HERE=splash/staged/speed
source "$HERE/quiet.sh"
SRC=splash/splash-src
BENCH=$SRC/build/engine-tests/gguf-projection-benchmark
OUT=logs/speed/kernel-ab.txt
CACHE="${TMPDIR:-/tmp}/splash-kernel-ab"
LIBS="base p2 p3 grid"
BUILD_ONLY=0
while (( $# )); do
  case "$1" in
    --build-only) BUILD_ONLY=1 ;;
    --libs) LIBS="$2"; shift ;;
    *) echo "usage: kernel_ab.sh [--libs \"base grid\"] [--build-only]" >&2; exit 2 ;;
  esac
  shift
done
mkdir -p logs/speed "$CACHE"

if (( !BUILD_ONLY )); then
  require_measurable || exit 1
  [[ -x "$BENCH" ]] || (cd "$SRC" && make build/engine-tests/gguf-projection-benchmark) || exit 1
fi

# build NAME [DEPTH] [PATCH...]: the flattened library of Splash's runtime with the patches applied.
build() {
  local name=$1 depth=$2; shift 2
  local stamp
  stamp=$( { find "$SRC/runtime/metal" -type f -print0 | xargs -0 stat -f '%m %N' | sort; \
             echo "$depth"; for p in "$@"; do shasum "$ROOT/$p"; done; } | shasum | cut -c1-16)
  if [[ -f "$CACHE/$name.stamp" && "$(cat "$CACHE/$name.stamp")" == "$stamp" && -d "$CACHE/lib-$name" ]]; then return 0; fi
  rm -rf "${CACHE:?}/$name" "${CACHE:?}/lib-$name"
  mkdir -p "$CACHE/$name"; cp -R "$SRC/runtime" "$CACHE/$name/runtime"
  # A patch may also update a test under dev/ (grid-v2 does): copy dev/tests only then, as fewer new files mean
  # fewer Defender scans.
  for p in "$@"; do
    if grep -q '^+++ dev/' "$ROOT/$p"; then mkdir -p "$CACHE/$name/dev"; cp -R "$SRC/dev/tests" "$CACHE/$name/dev/tests"; break; fi
  done
  for p in "$@"; do patch -s -p0 -d "$CACHE/$name" < "$ROOT/$p" || return 1; done
  if [[ "$depth" != 1 ]]; then
    sed -i '' "s/^#define GGUF_STAGED_PREFETCH 1$/#define GGUF_STAGED_PREFETCH $depth/" \
      "$CACHE/$name/runtime/metal/kernels/common/gguf_staged_tile.h"
  fi
  python3 "$SRC/dev/tools/flatten_metal.py" "$CACHE/$name/runtime" --all "$CACHE/lib-$name" > /dev/null || return 1
  echo "$stamp" > "$CACHE/$name.stamp"
}
# A library name is "+"-joined parts: base (Splash as it is), pN (prefetch depth N), or X (the patch X.patch here).
resolve_and_build() {
  local lib=$1 depth=1 part
  local patches=()
  IFS=+ read -r -a parts <<< "$lib"
  for part in "${parts[@]}"; do
    case "$part" in
      base) ;;
      p[0-9]) depth=${part#p}; patches+=("$HERE/prefetch-depth.patch") ;;
      grid) patches+=("$HERE/iq3xxs-grid-table-sign-xor.patch") ;;
      *) [[ -f "$HERE/$part.patch" ]] || { echo "unknown library part $part (no $HERE/$part.patch)" >&2; return 2; }
         patches+=("$HERE/$part.patch") ;;
    esac
  done
  build "$lib" "$depth" ${patches[@]+"${patches[@]}"}
}
for lib in $LIBS; do
  resolve_and_build "$lib" || { echo "error: building $lib failed" >&2; exit 1; }
done
if (( BUILD_ONLY )); then
  for lib in $LIBS; do
    f="$CACHE/lib-$lib/shared__gguf_linear.metal"
    echo "$lib: prefetch depth $(grep -m1 '^#define GGUF_STAGED_PREFETCH' "$f" | awk '{print $3}'), grid tables $(grep -c -E 'the grid quant_pair_table|staged_grid_table|grid words' "$f")"
  done
  echo "libraries in $CACHE"; exit 0
fi

wait_quiet 30 30 1800 || exit 1
{
  echo "kernel A/B $(date '+%Y-%m-%d %H:%M'); one lane (L1), staged tile, ms (median of 20); dfd = Defender CPU %"
  for round in 1 2; do
    for lib in $LIBS; do
      for c in "iq3xxs 17408 5120" "iq3xxs 5120 17408" "iq3s 17408 5120" "iq3s 5120 17408" "iq2s 17408 5120" \
               "iq2xxs 17408 5120" "q4k 17408 5120"; do
        set -- $c
        line=$("$BENCH" "$CACHE/lib-$lib" "$1" "$2" "$3" none 20 2>&1 | grep " L1 ")
        echo "r$round $lib $1 ${2}x$3 dfd=$(defender_cpu) | $line"
      done
    done
  done
} > "$OUT" 2>&1
python3 - "$OUT" <<'EOF'
import re, sys, collections
rows, busy = collections.defaultdict(list), 0
for l in open(sys.argv[1]):
    m = re.match(r"r\d (\S+) (\w+) (\S+) dfd=(\d+) \| .*?staged S\d+\* ([\d.]+)", l)
    if not m: continue
    if int(m.group(4)) > 30: busy += 1; continue          # skewed by a Defender scan
    rows[(m.group(2), m.group(3), m.group(1))].append(float(m.group(5)))
libs = list(dict.fromkeys(k[2] for k in sorted(rows, key=lambda k: list(rows).index(k))))
for f, s in sorted({(k[0], k[1]) for k in rows}):
    base = rows.get((f, s, "base"))
    b = sum(base) / len(base) if base else None
    cells = []
    for lib in libs:
        v = rows.get((f, s, lib))
        if v:
            m = sum(v) / len(v)
            cells.append(f"{lib} {m:.4f}" + (f" ({(b / m - 1) * 100:+.1f}%)" if b and lib != "base" else ""))
    print(f"{f} {s:12s}  " + "  ".join(cells))
if busy: print(f"({busy} measurements dropped: Defender above 30% CPU)")
EOF

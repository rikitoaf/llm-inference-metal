#!/usr/bin/env bash
# Checks the staged decode tile's K-split policy (runtime/ops/LinearGguf.cpp kStagedTiers) against the fastest split
# of every 27B projection shape that matters, on gguf-projection-benchmark: Splash's tiers {6, 512} ("cur") against
# split-policy.patch next to this script, {6, 512} + {12, 2048} ("new"). The benchmark times every split of both tiles
# at one to four lanes; the script computes both policies' choices from the shape and the core count the benchmark
# prints, so it needs no rebuild with the patch. The benchmark binary's own choice (its "*", which follows the
# policy libsplash was built with) is printed beside them.
#
#   splash/staged/speed/split_policy_check.sh [METALLIB]          measure (~10 min of GPU), then summarize
#   splash/staged/speed/split_policy_check.sh --analyze LOG       summarize a log only (no GPU), e.g.
#                                                                 logs/speed/projections.txt
#   options: --rounds N (40)   --formats "iq3xxs iq3s iq4xs iq2s q4k"   --head-formats "q4k"   --out LOG
#
# METALLIB defaults to splash/splash-src/build/splash.metallib. Shapes (N x K): FFN gate/up 17408 x 5120 and down
# 5120 x 17408, GDN qkv 10240 x 5120, GDN z / attention-sized 6144 x 5120, GDN and attention outputs 5120 x 6144, in
# every format of --formats; the vocabulary head 248320 x 5120 in --head-formats (the 27B's head is Q4_K; its choice
# is S1 under both policies); and the fused inputs as the 27B dispatches them (qkv|z, 10240+6144 quantized columns of
# its 16640, and q|k|v, 12288+1024+1024) in the formats its layers use most. The 27B never dispatches 10240 x 5120 or
# 6144 x 5120 alone: they run inside the fused GDN input.
#
# Measurement waits until Microsoft Defender has been quiet for 30 s (quiet.sh; its scans take 20-45% off the GPU),
# samples Defender's CPU every 2 s while each case runs and records the peak on every line ("dfd"); a case whose peak
# passes 30% is measured once more after Defender quiets. Raw output: logs/speed/split-policy-check.txt.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$ROOT/splash/staged/speed"
SRC="$ROOT/splash/splash-src"
BENCH="$SRC/build/engine-tests/gguf-projection-benchmark"
# shellcheck source=quiet.sh
source "$HERE/quiet.sh"

METALLIB="$SRC/build/splash.metallib"
ROUNDS=40
FORMATS="iq3xxs iq3s iq4xs iq2s q4k"
HEAD_FORMATS="q4k"
OUT="$ROOT/logs/speed/split-policy-check.txt"
ANALYZE=""
while (( $# )); do
  case "$1" in
    --analyze) ANALYZE="${2:?--analyze takes a log}"; shift ;;
    --rounds) ROUNDS="${2:?}"; shift ;;
    --formats) FORMATS="${2:?}"; shift ;;
    --head-formats) HEAD_FORMATS="${2-}"; shift ;;
    --out) OUT="${2:?}"; shift ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    -*) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
    *) METALLIB="$1" ;;
  esac
  shift
done

# summarize LOG: per case and lane, the staged tile's fastest split against both policies' choices, then per-lane
# totals over every case and over the 27B's dispatches per decode step.
summarize() {
  python3 - "$1" <<'EOF'
import re, sys
from collections import defaultdict

CUR = [(6, 512)]                  # Splash: kStagedTiers {{6, 512}}
NEW = [(6, 512), (12, 2048)]      # split-policy.patch: {{6, 512}, {12, 2048}}

def splits(n, k, cores, tiers):
    """runtime/ops/LinearGguf.cpp decodeSplits: double while some tier asks, at most eight."""
    grid, s = n // 64, 1
    while s < 8 and any(grid * s < t * cores and k // (2 * s) >= i for t, i in tiers):
        s *= 2
    return s

# Dispatches per 27B decode step (blocks 0-63 of Qwen3.8-27B-UD-IQ3_XXS.gguf), by (N, K) and format(s): FFN gate+up,
# down, GDN ssm_out + attention output, the head, and the fused GDN (qkv|z) and attention (q|k|v) inputs.
W27 = {
    (17408, 5120): {"iq3xxs": 44, "iq3s": 21, "iq4xs": 14, "iq2s": 18, "q4k": 1},
    (5120, 17408): {"iq3xxs": 15, "iq3s": 20, "iq4xs": 8, "iq2s": 7, "q4k": 1},
    (5120, 6144): {"iq3xxs": 10, "iq3s": 26, "iq4xs": 19, "iq2s": 1, "q4k": 4},
    (248320, 5120): {"q4k": 1},
    (16384, 5120): {"iq3xxs+iq3xxs": 13, "iq3s+iq3xxs": 7},
    (14336, 5120): {"iq3xxs+q4k+q5k": 3, "q2k+iq3s+q4k": 3},
}

device = re.compile(r"\((?:GPU family \d+, )?(\d+) cores\): (\S+) ([\d+]+)x(\d+) ")
lane = re.compile(r"^\s*L(\d)\s*\|")
cell = re.compile(r"staged S(\d)(\*?) ([\d.]+)")
dfd_re = re.compile(r"\|\s*dfd (\d+)")
cases, label = [], None
with open(sys.argv[1]) as log:
    lines = log.read().splitlines()
for line in lines:
    if line.startswith("## "):
        label = re.split(r"[:\s]", line[3:].strip(), maxsplit=1)[0]
        continue
    m = device.search(line)
    if m:
        cores, fmt, widths, k = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
        n = sum(int(w) for w in widths.split("+"))
        cases.append(dict(label=label or f"{n}x{k}", fmt=fmt, n=n, k=k, cores=cores, lanes={}))
        continue
    m = lane.match(line)
    if m and cases:
        cells = {int(s): (float(ms), star == "*") for s, star, ms in cell.findall(line)}
        if cells:
            d = dfd_re.search(line)
            cases[-1]["lanes"][int(m.group(1))] = (cells, int(d.group(1)) if d else None)

if not cases:
    sys.exit("no benchmark cases in " + sys.argv[1])
cores_seen = sorted({c["cores"] for c in cases})
print(f"staged tile, ms (benchmark medians) on {'/'.join(map(str, cores_seen))} cores. cur: Splash {{6, 512}};"
      " new: split-policy.patch {6, 512} + {12, 2048}; bin: the benchmark binary's own choice (*).")
print("+x%: over the fastest split of that lane. dfd: Defender's peak CPU % during the case (> 30: suspect).")
hdr = f"{'case':22s} {'format':16s} {'N x K':14s} {'L':>2s}  {'bin':3s}  {'cur':>17s}  {'new':>17s}  {'fastest':>11s}  {'new/cur':>7s}  dfd"
print(hdr)
print("-" * len(hdr))
tot = defaultdict(lambda: [0.0, 0.0, 0.0])        # lane -> [cur, new, fastest] over every case
w27 = defaultdict(lambda: [0.0, 0.0, 0.0])        # lane -> the same, weighted by the 27B's dispatches per step
worse, unweighted, busy, stale = [], [], 0, 0
for c in cases:
    cur = splits(c["n"], c["k"], c["cores"], CUR)
    new = splits(c["n"], c["k"], c["cores"], NEW)
    weight = W27.get((c["n"], c["k"]), {}).get(c["fmt"], 0)
    if not weight:
        unweighted.append(f"{c['fmt']} {c['n']}x{c['k']}")
    for L in sorted(c["lanes"]):
        cells, dfd = c["lanes"][L]
        if cur not in cells or new not in cells:
            continue
        best = min(cells, key=lambda s: cells[s][0])
        fb = cells[best][0]
        star = [s for s in cells if cells[s][1]]
        bin_choice = f"S{star[0]}" if star else "-"
        if star and star[0] not in (cur, new):
            stale += 1
        pct = lambda s: (cells[s][0] / fb - 1) * 100
        delta = (cells[new][0] / cells[cur][0] - 1) * 100
        mark = ""
        if delta > 1.0:
            mark = "  <- new slower"
            worse.append(f"{c['label']} {c['fmt']} L{L} {delta:+.1f}%")
        elif new != cur and delta < -1.0:
            mark = "  <- new faster"
        if dfd is not None and dfd > 30:
            busy += 1
            mark += "  (Defender busy)"
        print(f"{c['label'][:22]:22s} {c['fmt']:16s} {str(c['n']) + 'x' + str(c['k']):14s} {L:>2d}  {bin_choice:3s}  "
              f"S{cur} {cells[cur][0]:.4f} {pct(cur):+5.1f}%  S{new} {cells[new][0]:.4f} {pct(new):+5.1f}%  "
              f"S{best} {fb:.4f}  {delta:+6.1f}%  {'' if dfd is None else dfd}{mark}")
        for acc, w in ((tot, 1), (w27, weight)):
            acc[L][0] += w * cells[cur][0]
            acc[L][1] += w * cells[new][0]
            acc[L][2] += w * fb
    print()

def table(title, acc):
    print(title)
    for L in sorted(acc):
        c, n, f = acc[L]
        if c:
            print(f"  L{L}: cur {c:8.3f}  new {n:8.3f} ({(n / c - 1) * 100:+.2f}%)  fastest {f:8.3f}"
                  f"  (cur +{(c / f - 1) * 100:.2f}%, new +{(n / f - 1) * 100:.2f}% over fastest)")
table("Sum over every case, ms:", tot)
table("27B decode step, ms of the measured (shape, format) pairs times their dispatches per step:", w27)
if unweighted:
    print("not in the 27B weighting (alone, or not a 27B format of that shape): " + ", ".join(sorted(set(unweighted))))
if worse:
    print("new slower than cur by more than 1%: " + "; ".join(worse))
if busy:
    print(f"{busy} lane lines were measured while Defender passed 30% CPU: rerun those cases.")
if stale:
    print(f"{stale} lane lines: the binary's * is neither policy's choice (another core count or policy in libsplash).")
EOF
}

if [[ -n "$ANALYZE" ]]; then
  summarize "$ANALYZE"
  exit $?
fi

[[ -e "$METALLIB" ]] || { echo "error: no metallib at $METALLIB" >&2; exit 1; }
require_measurable || exit 1
if [[ ! -x "$BENCH" ]]; then
  (cd "$SRC" && make build/engine-tests/gguf-projection-benchmark) || exit 1
fi
mkdir -p "$(dirname "$OUT")"

# label | formats | N (widths) | K, one benchmark run each.
CASES=()
for f in $FORMATS; do CASES+=("ffn-gate-up|$f|17408|5120"); done
for f in $FORMATS; do CASES+=("ffn-down|$f|5120|17408"); done
for f in $FORMATS; do CASES+=("gdn-qkv|$f|10240|5120"); done
for f in $FORMATS; do CASES+=("gdn-z|$f|6144|5120"); done
for f in $FORMATS; do CASES+=("out-proj|$f|5120|6144"); done
for f in $HEAD_FORMATS; do CASES+=("output-head|$f|248320|5120"); done
CASES+=("gdn-input-fused|iq3xxs+iq3xxs|10240+6144|5120" "gdn-input-fused|iq3s+iq3xxs|10240+6144|5120"
        "attn-input-fused|iq3xxs+q4k+q5k|12288+1024+1024|5120" "attn-input-fused|q2k+iq3s+q4k|12288+1024+1024|5120")

PEAK="$(mktemp "${TMPDIR:-/tmp}/split-policy-dfd.XXXXXX")"
SAMPLER=""
# The sampler and its sleep or ps.
stop_sampler() {
  [[ -n "$SAMPLER" ]] && { pkill -P "$SAMPLER" 2>/dev/null; kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; }
  SAMPLER=""
}
trap 'stop_sampler; rm -f "$PEAK"' EXIT

# run_case FMT N K: the benchmark's output in $RUN, Defender's peak CPU during it in $DFD; non-zero when it failed.
run_case() {
  local d0 d1 rc
  d0=$(defender_cpu); echo "$d0" > "$PEAK"
  ( max=$d0; while :; do v=$(defender_cpu); if (( v > max )); then max=$v; echo "$max" > "$PEAK"; fi; sleep 2; done ) &
  SAMPLER=$!
  RUN=$("$BENCH" "$METALLIB" "$1" "$2" "$3" none "$ROUNDS" 2>&1); rc=$?
  stop_sampler
  d1=$(defender_cpu); DFD=$(cat "$PEAK"); (( d1 > DFD )) && DFD=$d1
  return $rc
}

wait_quiet 30 30 1800 || exit 1
{
  echo "split policy check $(date '+%Y-%m-%d %H:%M'); metallib $METALLIB; $ROUNDS rounds; staged and register tiles,"
  echo "one to four lanes, every split; dfd = Defender's peak CPU % during the case"
} > "$OUT"
n=0
for c in "${CASES[@]}"; do
  IFS='|' read -r label fmt widths k <<< "$c"
  n=$((n + 1))
  echo "[$n/${#CASES[@]}] $label $fmt ${widths}x$k" >&2
  (( $(defender_cpu) > 30 )) && { wait_quiet 30 15 900 || true; }
  run_case "$fmt" "$widths" "$k"; rc=$?
  if (( rc == 0 && DFD > 30 )); then
    echo "  Defender reached ${DFD}% during the case: measuring it again once quiet" >&2
    wait_quiet 30 30 1800 || true
    run_case "$fmt" "$widths" "$k"; rc=$?
  fi
  {
    echo "## $label: $fmt N=$widths K=$k dfd=$DFD"
    if (( rc != 0 )); then
      echo "error: gguf-projection-benchmark exited $rc: $RUN"
    else
      # Each lane line carries Defender's peak during the case.
      printf '%s\n' "$RUN" | sed -E "s/^(  L[0-9].*)$/\1 | dfd $DFD/"
    fi
  } >> "$OUT"
  (( rc != 0 )) && echo "  failed (exit $rc), see $OUT" >&2
done
echo "raw: $OUT" >&2
summarize "$OUT"

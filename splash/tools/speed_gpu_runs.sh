#!/usr/bin/env bash
# The GPU measurements of splash/SPEED_PLAN.md in one detached job (~20 min). It stops any model server and
# restarts Splash several times: each start needs ~16 GiB reclaimable (python3 splash/tools/hostavail.py).
# Run it on AC power: on battery this Mac uses Low Power mode, which throttles the GPU and skews every timing.
#
#   nohup splash/tools/speed_gpu_runs.sh > logs/speed-gpu-runs.log 2>&1 & disown
#
# 1. decode-profile: the installed model's decode command replayed one dispatch at a time (int8 KV), with the
#    fused GPU time, the wall time and the sum of the parts: where the ~88 ms of a cycle go.
# 2. gguf-projection-benchmark at the shapes that carry the bytes (FFN gate/up/down in IQ3_XXS, IQ3_S, IQ4_XS,
#    IQ2_S; attention QKV; the 248,320-row Q4_K output head, which the draft and the verify both read): GB/s per
#    format against Splash's own Q4_K reference rate.
# 3. kvbench speed, 400 tokens of new code, at temperature 0, 1.0 (what dsh gets today) and 0.6, and at
#    SPLASH_DRAFT_TEMPERATURE_SCALE 0.5 and 0.35: tokens per cycle, ms per cycle split into GPU / launch / host.
# Output: logs/speed/ and logs/speed/summary.txt.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SRC=splash/splash-src
OUT=logs/speed
mkdir -p "$OUT"

if pgrep -f "run_kv_suite|next_gpu_runs|swift_gpu_runs|serve-native" >/dev/null; then
  echo "error: a suite or server is running" >&2; exit 1
fi
defender=$(ps -axo %cpu=,comm= | awk '/wdavdaemon_unprivileged/ {print int($1)}' | head -1)
if (( ${defender:-0} > 30 )); then   # its scans took 20-40% off the GPU on 2026-10-15
  echo "error: Microsoft Defender is scanning (${defender}% CPU); timings would be skewed, try later" >&2; exit 1
fi
if ! pmset -g batt | grep -q "AC Power"; then
  echo "error: on battery (Low Power mode throttles the GPU); plug in and rerun" >&2; exit 1
fi
trap './scripts/stop.sh >/dev/null 2>&1' EXIT INT TERM
echo "=== $(date '+%H:%M:%S') start; $(python3 splash/tools/hostavail.py); $(pmset -g | grep -w powermode | tr -s ' ')"

# --- 1. decode-profile of the default model
echo "=== $(date '+%H:%M:%S') decode-profile"
(cd "$SRC" && make benchmark-decode-profile MODEL=unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS LANGUAGE_ONLY=1 \
   DECODE_PROFILE_ARGS='--prompt-tokens 2048 --cycles 9') > "$OUT/decode-profile.txt" 2>&1 \
  || echo "decode-profile failed; see $OUT/decode-profile.txt"

# --- 2. projection kernels: "<format> <N> <K>" as one decode lane of 8 rows reads them
echo "=== $(date '+%H:%M:%S') gguf-projection-benchmark"
: > "$OUT/projections.txt"
while read -r fmt n k label; do
  [[ -z "$fmt" ]] && continue
  echo "## $label: $fmt N=$n K=$k ($(( n / 256 * k / 256 )) 256x256 blocks)" >> "$OUT/projections.txt"
  "$SRC/build/engine-tests/gguf-projection-benchmark" "$SRC/build/splash.metallib" "$fmt" "$n" "$k" none 20 \
    >> "$OUT/projections.txt" 2>&1 || echo "  failed" >> "$OUT/projections.txt"
done <<'CASES'
q4k 17408 5120 reference-q4k-ffn-up
iq3xxs 17408 5120 ffn-gate-up-iq3xxs
iq3s 17408 5120 ffn-up-iq3s
iq3s 5120 17408 ffn-down-iq3s
iq3xxs 5120 17408 ffn-down-iq3xxs
iq4xs 5120 17408 ffn-down-iq4xs
iq2s 17408 5120 ffn-gate-iq2s
iq3xxs 10240 5120 attn-qkv-iq3xxs
q4k 248320 5120 output-head-q4k
CASES

# --- 3. speculation knobs, one server per line (run_kv_suite restarts Splash for each)
echo "=== $(date '+%H:%M:%S') kvbench speed sweep"
splash/tools/run_kv_suite.sh <<'PLAN'
turbo-int4 spd-t0 --parts speed --temperature 0 --speed-repeats 3
turbo-int4 spd-t1 --parts speed --temperature 1.0 --speed-repeats 6
turbo-int4 spd-t06 --parts speed --temperature 0.6 --speed-repeats 6
turbo-int4 spd-t1-s05 ENV=SPLASH_DRAFT_TEMPERATURE_SCALE=0.5 --parts speed --temperature 1.0 --speed-repeats 6
turbo-int4 spd-t1-s035 ENV=SPLASH_DRAFT_TEMPERATURE_SCALE=0.35 --parts speed --temperature 1.0 --speed-repeats 6
turbo-int4 spd-t06-s035 ENV=SPLASH_DRAFT_TEMPERATURE_SCALE=0.35 --parts speed --temperature 0.6 --speed-repeats 6
PLAN
./scripts/stop.sh

# --- summary
{
  echo "== decode-profile (tail)"; tail -n 60 "$OUT/decode-profile.txt"
  echo; echo "== projections"; cat "$OUT/projections.txt"
  echo; echo "== speculation sweep"
  python3 - <<'EOF'
import json, pathlib
rows = []
for label in ("spd-t0", "spd-t1", "spd-t06", "spd-t1-s05", "spd-t1-s035", "spd-t06-s035"):
    p = pathlib.Path("logs/kvbench") / f"{label}.json"
    if not p.exists():
        print(f"{label:14s} missing"); continue
    ds = [s["status_delta"] for s in json.loads(p.read_text()).get("speed", []) if s.get("status_delta")]
    cyc = sum(d.get("scheduler.decode_batches", 0) for d in ds)
    if not cyc:
        print(f"{label:14s} no counters"); continue
    tok = sum(d["metrics.decode_output_tokens"] for d in ds); ms = sum(d["metrics.decode_cycle_ms"] for d in ds)
    acc = sum(d["metrics.accepted_draft_tokens"] for d in ds) / max(1, sum(d["metrics.drafted_tokens"] for d in ds))
    g = sum(d.get("model_timing.decode.total_gpu_ms", 0) for d in ds) / cyc
    print(f"{label:14s} {tok/cyc:5.2f} tokens/cycle  {ms/cyc:6.1f} ms/cycle (gpu {g:5.1f})  {1000*tok/ms:5.1f} tok/s  acceptance {acc:.3f}  ({len(ds)} requests, {cyc} cycles)")
EOF
} > "$OUT/summary.txt" 2>&1
echo "=== $(date '+%H:%M:%S') done; see $OUT/summary.txt"

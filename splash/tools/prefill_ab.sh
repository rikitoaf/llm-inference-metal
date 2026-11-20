#!/usr/bin/env bash
# SPLASH_PREFILL_ATTN_V2 (splash/dev-prefill-attn, ATTN.md): every check and the long-context prefill A/B, unattended,
# DSpine engine as in production (SPLASH_DRAFTER=dspine, tree, copy, turbo-int4). Results in files; run it from your
# own terminal.
#
#   splash/tools/prefill_ab.sh [--quick] OUTDIR
#
# Steps (a failing step marks the run failed; later independent steps still run):
#   1 build          make all + the engine test binaries + decode-run in splash/dev-prefill-attn; dev-longctx's
#                    decode-run built into OUTDIR/longctx-build (make BUILD=...: dev-longctx's files are not touched);
#                    the kernel micro-benchmark (splash/tools/prefill_attn_bench.mm) into OUTDIR
#   2 cpu_tests      kv-first-engine, runtime-status, native-engine-loop, native-fd-transport (dev-prefill-attn)
#   3 greedy_digest  flag unset: dev-prefill-attn == dev-longctx, greedy digests of the 6 agent prompts, copy 0 and 1
#   4 logits         cold-prefill whole-vocabulary logits after the 6 agent prompts and a ~30K-token prompt (full mode
#                    also ~60K): flag on vs off, against the control "flag off, another prefill partition"
#                    (--first-chunk 37): same argmax everywhere and max |on - off| <= 1.5 x max |control - off|
#   5 near_tie       flag on vs off, greedy, agent prompts, copy 0 and 1: identical, or the first difference is a
#                    near-tie of the target (|logit a - b| < 1.0, scored with the flag off); the flag took effect
#   6 distribution   dist and fruit, T=0.6: two-sample chi-square per position, flag on vs off
#   7 kernel_bench   the isolated split + reduce at 20K / 60K / 100K, production vs v2 (seconds)
#   8 ttft           servers of dev-prefill-attn, A = flag off, B = flag on, A-B-B-A (--quick: A-B): per arm
#                    splash/tools/prefill_ttft.py (TTFT of a fresh 2K and 4K suffix after a cached ~30K / ~60K /
#                    ~100K prefix; full mode 2 repetitions) and splash/tools/park_long.py (a ~70K prefix, then
#                    agent turns of ~3K new tokens; --quick 3 turns, full 5)
# summary.md ends with the recommended default: on only when TTFT at >= 60K improves by >= 5% (B vs A, the
# prefill_ttft cells at the ~60K and ~100K positions pooled) and every correctness step (3-6) passed.
# GPU rules: AC power (exit 3 on battery at the start), /tmp/splash-gpu.lock for the GPU steps, a Splash server this
# script did not start is never stopped (it waits in 5-min polls, PREFILL_AB_WAIT_MINUTES, default 360; a run during
# which one came up is repeated once), >= PREFILL_AB_MIN_GIB (13) GiB reclaimable before each run (waits up to
# 10 min), Defender under 10% for 30 s before each timed run (waits up to 3 min, logged). Context cap of the A/B
# servers: PREFILL_AB_CAP (default 131072).
# Outputs: summary.md, summary.json, logs/*.log, json/*.json, out/*.
# Exit: 0 when every step passes, 1 otherwise, 3 on battery power.
set -uo pipefail
QUICK=0
if [[ ${1:-} == --quick ]]; then QUICK=1; shift; fi
OUT=${1:?usage: prefill_ab.sh [--quick] OUTDIR}
mkdir -p "$OUT/logs" "$OUT/json" "$OUT/out"
OUT=$(cd "$OUT" && pwd)
STACK_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
S=$STACK_ROOT/splash
DEV=$S/dev-prefill-attn
REF_SOURCE=$S/dev-longctx
REF_BUILD=$OUT/longctx-build
DEV_BUILD=$DEV/build
MODEL=$S/splash-src/install/models/.selections/fb0aa6e23bdecc7d24023adea3bcae72062d1c103ab1578902b00a2ebb3f500f
PROMPTS=${AB_PROMPTS:-$HOME/.cache/splash-dspine/prompts}
TOOLS=$S/tools/dspine
PY=/opt/homebrew/bin/python3
[[ -x $PY ]] || PY=python3
export PYTHONDONTWRITEBYTECODE=1
source "$STACK_ROOT/scripts/lib/system.sh"
if ! on_ac_power; then echo "prefill_ab: on battery power, refusing to run" >&2; exit 3; fi
WAIT_MINUTES=${PREFILL_AB_WAIT_MINUTES:-360}
MIN_GIB=${PREFILL_AB_MIN_GIB:-13}
CAP=${PREFILL_AB_CAP:-131072}
if (( QUICK )); then ORDER=(A1 B1); SAMPLES=60; REPS=1; TURNS=3; LOGIT_LINES=("$PROMPTS/long.txt:0:30000")
else ORDER=(A1 B1 B2 A2); SAMPLES=200; REPS=2; TURNS=5
  LOGIT_LINES=("$PROMPTS/long.txt:0:30000" "$PROMPTS/long.txt:1:60000"); fi
DSPINE_ENV=(SPLASH_DRAFTER=dspine SPLASH_DRAFT_TREE=1 SPLASH_DRAFT_HEAD_ROWS=98304)
V2=(SPLASH_PREFILL_ATTN_V2=1)
V2_NOTICE="SPLASH_PREFILL_ATTN_V2=1: prefill attention v2"

FAILED=0
record() { [[ $2 == fail ]] && FAILED=1; echo "$(date '+%H:%M:%S') step $1: $2${3:+ ($3)}" | tee -a "$OUT/logs/steps.log"; }
note() { echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT/logs/steps.log"; }
server_up() { curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null; }
LOCK_HELD=0
STARTED_SERVER=0
take_lock() { until mkdir /tmp/splash-gpu.lock 2>/dev/null; do sleep 10; done; LOCK_HELD=1; }
release_lock() { (( LOCK_HELD )) && rmdir /tmp/splash-gpu.lock 2>/dev/null; LOCK_HELD=0; }
stop_server() {
  if (( STARTED_SERVER )); then "$STACK_ROOT/scripts/stop.sh" >> "$OUT/logs/server.log" 2>&1; STARTED_SERVER=0; fi
}
trap 'stop_server; release_lock' EXIT
trap 'exit 130' INT TERM HUP
GPU_GAVE_UP=0
# A Splash server this script did not start is the user's: wait for it to go (it idle-stops after the session).
wait_for_server() {
  local waited=0
  while server_up && (( STARTED_SERVER == 0 )); do
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
defender_wait() {  # timed runs wait up to 3 min for Defender under 10% for 30 s (ANALYSIS.md section 10.1)
  local waited=0 calm=0
  while (( waited < 180 && calm < 3 )); do
    if (( $(defender_cpu | cut -d. -f1) < 10 )); then calm=$((calm + 1)); else calm=0; fi
    sleep 10; waited=$((waited + 10))
  done
  echo "$(date '+%H:%M:%S') $1 defender_cpu=$(defender_cpu) waited=${waited}s" >> "$OUT/logs/defender.log"
}
# decode NAME BUILD_DIR PROMPTS [decode-run args...] -- [env...]: one decode-run under the GPU gate, log
# logs/NAME.log. A run during which the user's server came up is repeated once after it goes.
decode() {
  local name=$1 build=$2 prompts=$3; shift 3
  local args=() envs=()
  while (( $# )) && [[ $1 != -- ]]; do args+=("$1"); shift; done
  [[ ${1:-} == -- ]] && shift
  envs=("$@")
  local attempt rc=1
  for attempt in 1 2; do
    if (( GPU_GAVE_UP )) || ! wait_for_server; then return 1; fi
    wait_for_memory
    (cd "$DEV" && env -u SPLASH_DRAFTER -u SPLASH_DRAFT_TREE -u SPLASH_DRAFT_COPY -u SPLASH_DRAFT_HEAD_ROWS \
       -u SPLASH_DSPINE_WINDOW -u SPLASH_VERIFY_ROWS_LONG -u SPLASH_VERIFY_ROWS_LONG_FROM -u SPLASH_PREFILL_ATTN_V2 \
       ${envs[@]+"${envs[@]}"} /usr/bin/time -l "$build/engine-tests/decode-run" "$build/splash.metallib" "$MODEL" \
       "$prompts" ${args[@]+"${args[@]}"}) > "$OUT/logs/$name.log" 2>&1
    rc=$?
    if server_up && (( STARTED_SERVER == 0 )); then
      note "$name: a foreign Splash server came up during the run; repeating it once it goes"
      mv "$OUT/logs/$name.log" "$OUT/logs/$name.interrupted$attempt.log"
      continue
    fi
    return $rc
  done
  return $rc
}
# switched LOG: the flag-on run printed the engine's v2 notice (the switch took effect).
switched() {
  if grep -q "$V2_NOTICE" "$1"; then echo "$(basename "$1"): v2 on" >> "$OUT/json/switch.txt"; return 0; fi
  echo "$(basename "$1"): NO v2 notice" >> "$OUT/json/switch.txt"; return 1
}
# start_server ARM FLAG: a dev-prefill-attn server (DSpine, park lane off), logs under logs/splash-ARM
start_server() {
  local arm=$1 flag=$2
  mkdir -p "$OUT/logs/splash-$arm"
  ( cd "$STACK_ROOT" && env -u SPLASH_PREFILL_ATTN_V2 "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=1 SPLASH_SRC=$DEV \
      SPLASH_PARK_LANE=0 SPLASH_MAX_CONTEXT=$CAP SPLASH_IDLE_STOP=off SPLASH_IDLE_RELEASE=off \
      SPLASH_LOG_DIR=$OUT/logs/splash-$arm $( (( flag )) && echo SPLASH_PREFILL_ATTN_V2=1 ) \
      nohup scripts/serve-splash.sh >> "$OUT/logs/splash-$arm/console.log" 2>&1 & )
  STARTED_SERVER=1
  for _ in $(seq 1 300); do server_up && return 0; sleep 2; done
  note "server $arm did not come up"
  return 1
}
echo "prefill_ab $(date) quick=$QUICK cap=$CAP out=$OUT" > "$OUT/logs/steps.log"
pmset -g batt | head -1 >> "$OUT/logs/steps.log"
: > "$OUT/json/switch.txt"

# ---- 1 build
TESTS=(kv-first-engine runtime-status native-engine-loop native-fd-transport decode-run)
if (cd "$DEV" && make all && make "${TESTS[@]/#/build/engine-tests/}") > "$OUT/logs/build.log" 2>&1 \
   && (cd "$REF_SOURCE" && make BUILD="$REF_BUILD" all "$REF_BUILD/engine-tests/decode-run") > "$OUT/logs/build-longctx.log" 2>&1 \
   && clang++ -std=c++20 -O2 -fobjc-arc -framework Foundation -framework Metal "$S/tools/prefill_attn_bench.mm" \
        -o "$OUT/prefill_attn_bench" >> "$OUT/logs/build.log" 2>&1
then record build pass; else record build fail; fi

# ---- 2 cpu tests
ok=pass
for t in kv-first-engine runtime-status native-engine-loop native-fd-transport; do
  if ! (cd "$DEV" && build/engine-tests/$t) > "$OUT/logs/cpu-$t.log" 2>&1; then ok=fail; fi
done
grep -q "adopted request" "$OUT/logs/cpu-kv-first-engine.log" || ok=fail
record cpu_tests $ok

# ---- GPU steps
if ! wait_for_server; then
  for step in greedy_digest logits near_tie distribution kernel_bench ttft; do record $step skip "GPU busy with the user's server"; done
else
take_lock

# ---- 3 greedy digests, flag unset: dev-prefill-attn == dev-longctx
ok=pass
for copy in 0 1; do
  decode greedy-copy$copy-longctx "$REF_BUILD" "$PROMPTS/agent.txt" --new-tokens 256 --temperature 0 \
    --output "$OUT/out/greedy-copy$copy-longctx.txt" -- "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=$copy || ok=fail
  decode greedy-copy$copy-off "$DEV_BUILD" "$PROMPTS/agent.txt" --new-tokens 256 --temperature 0 \
    --output "$OUT/out/greedy-copy$copy-off.txt" -- "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=$copy || ok=fail
  $PY "$TOOLS/ab_tools.py" compare-digests "$OUT/logs/greedy-copy$copy-longctx.log" "$OUT/logs/greedy-copy$copy-off.log" \
    > "$OUT/json/greedy-copy$copy.json" || ok=fail
done
record greedy_digest $ok

# ---- 4 cold-prefill logits: flag on vs off, within the prefill-partition control
ok=pass
$PY "$S/tools/prefill_ab_tools.py" make-score "$OUT/out/logits.score" "$PROMPTS/agent.txt" "${LOGIT_LINES[@]}" \
  > "$OUT/logs/logits-prepare.log" 2>&1 || ok=fail
decode logits-off "$DEV_BUILD" "$PROMPTS/agent.txt" --score "$OUT/out/logits.score" \
  --logits-out "$OUT/out/logits-off.bin" -- "${DSPINE_ENV[@]}" || ok=fail
decode logits-control "$DEV_BUILD" "$PROMPTS/agent.txt" --score "$OUT/out/logits.score" --first-chunk 37 \
  --logits-out "$OUT/out/logits-control.bin" -- "${DSPINE_ENV[@]}" || ok=fail
decode logits-on "$DEV_BUILD" "$PROMPTS/agent.txt" --score "$OUT/out/logits.score" \
  --logits-out "$OUT/out/logits-on.bin" -- "${DSPINE_ENV[@]}" "${V2[@]}" || ok=fail
switched "$OUT/logs/logits-on.log" || ok=fail
$PY "$S/tools/prefill_ab_tools.py" logits-compare "$OUT/out/logits-off.bin" "$OUT/out/logits-control.bin" \
  "$OUT/out/logits-on.bin" "$OUT/json/logits.json" > "$OUT/logs/logits.report" 2>&1 || ok=fail
rm -f "$OUT"/out/logits-*.bin   # ~1 MB per line and run; the report keeps the numbers
record logits $ok

# ---- 5 flag on, greedy: equal to flag off up to near-ties of the target; the flag took effect
ok=pass
for copy in 0 1; do
  decode on-greedy-copy$copy "$DEV_BUILD" "$PROMPTS/agent.txt" --new-tokens 256 --temperature 0 \
    --output "$OUT/out/on-greedy-copy$copy.txt" -- "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=$copy "${V2[@]}" || ok=fail
  switched "$OUT/logs/on-greedy-copy$copy.log" || ok=fail
  $PY "$TOOLS/ab_tools.py" tie-prepare "$PROMPTS/agent.txt" "$OUT/out/greedy-copy$copy-off.txt" \
    "$OUT/out/on-greedy-copy$copy.txt" "$OUT/out/ties-copy$copy.score" "$OUT/json/ties-copy$copy.json" || ok=fail
  if [[ -s $OUT/out/ties-copy$copy.score ]]; then
    decode ties-copy$copy "$DEV_BUILD" "$PROMPTS/agent.txt" --score "$OUT/out/ties-copy$copy.score" -- \
      "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=$copy || ok=fail
  else : > "$OUT/logs/ties-copy$copy.log"; fi
  $PY "$TOOLS/ab_tools.py" tie-report "$OUT/json/ties-copy$copy.json" "$OUT/logs/ties-copy$copy.log" 1.0 \
    > "$OUT/logs/ties-copy$copy.report" || ok=fail
done
record near_tie $ok

# ---- 6 distribution: flag on samples the same law as flag off (two-sample chi-square)
ok=pass
for test in dist:16 fruit:40; do
  name=${test%%:*}; tokens=${test##*:}
  decode $name-off "$DEV_BUILD" "$PROMPTS/$name.txt" --new-tokens $tokens --temperature 0.6 --repeat $SAMPLES \
    --seed 1000 --output "$OUT/out/$name-off.txt" -- "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=1 || ok=fail
  decode $name-on "$DEV_BUILD" "$PROMPTS/$name.txt" --new-tokens $tokens --temperature 0.6 --repeat $SAMPLES \
    --seed 5000 --output "$OUT/out/$name-on.txt" -- "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=1 "${V2[@]}" || ok=fail
  switched "$OUT/logs/$name-on.log" || ok=fail
  $PY "$TOOLS/ab_tools.py" dist-compare "$OUT/out/$name-off.txt" "$OUT/out/$name-on.txt" $tokens \
    "$OUT/json/$name.json" > "$OUT/logs/$name.report" || ok=fail
done
record distribution $ok

# ---- 7 the kernel alone: production vs v2 split + reduce, one 2048-row chunk at ~20K / ~60K / ~100K
ok=pass
if (( GPU_GAVE_UP )) || ! wait_for_server; then ok=fail
else
  wait_for_memory
  defender_wait kernel_bench
  "$OUT/prefill_attn_bench" "$DEV_BUILD/splash.metallib/prefill__paged_attention.metal" 20480,59392,100352 2048 \
    v2:prefill_attention_turbo_int4_split_v2:prefill_attention_reduce_turbo_int4:8 > "$OUT/logs/kernel_bench.log" 2>&1 || ok=fail
fi
record kernel_bench $ok

# ---- 8 TTFT A/B on servers: A = flag off, B = flag on
ab_arm() {  # ab_arm NAME FLAG
  local name=$1 flag=$2 rc=0
  wait_for_memory
  start_server "$name" "$flag" || { stop_server; return 1; }
  defender_wait "ttft-$name"
  $PY "$S/tools/prefill_ttft.py" --positions 30000,60000,100000 --suffixes 2048,4096 --reps $REPS --label "$name" \
    --save "$OUT/json/ttft-$name.json" > "$OUT/logs/ttft-$name.log" 2>&1 || rc=1
  local compare=()
  [[ -s $OUT/json/park-A1.json && $name != A1 ]] && compare=(--compare "$OUT/json/park-A1.json")
  $PY "$S/tools/park_long.py" --prefix-tokens 70000 --turns $TURNS --label "$name" --nonce "$name-$(date +%s)" \
    ${compare[@]+"${compare[@]}"} --save "$OUT/json/park-$name.json" > "$OUT/logs/park-$name.log" 2>&1 || rc=1
  if (( flag )); then grep -q "$V2_NOTICE" "$OUT/logs/splash-$name/"*.log 2>/dev/null \
    && echo "server $name: v2 on" >> "$OUT/json/switch.txt" \
    || echo "server $name: no v2 notice in its logs (the engine's stderr may not reach them)" >> "$OUT/json/switch.txt"; fi
  stop_server
  return $rc
}
if (( GPU_GAVE_UP )) || ! wait_for_server; then record ttft skip "GPU busy with the user's server"
else
  ok=pass
  for arm in "${ORDER[@]}"; do
    flag=0; [[ $arm == B* ]] && flag=1
    ab_arm "$arm" $flag || ok=fail
  done
  record ttft $ok
fi
release_lock
fi

# ---- summary
$PY - "$OUT" "$QUICK" <<'EOF'
import json, re, sys, pathlib, statistics
out, quick = pathlib.Path(sys.argv[1]), sys.argv[2] == "1"
steps = {}
for line in (out / "logs" / "steps.log").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 4 and parts[1] == "step":
        steps[parts[2].rstrip(":")] = " ".join(parts[3:])
lines = [f"# prefill_ab summary ({'quick' if quick else 'full'}): SPLASH_PREFILL_ATTN_V2", "",
         "| step | result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in steps.items()]
result = {"quick": quick, "steps": steps}

bench = out / "logs" / "kernel_bench.log"
if bench.exists() and bench.read_text().strip():
    lines += ["", "## Kernel alone (split + reduce, one 2048-row chunk, random turbo-int4 KV)", "", "```",
              bench.read_text().strip(), "```"]

def load(pattern):
    return {p.stem.split("-", 1)[1]: json.loads(p.read_text()) for p in sorted((out / "json").glob(pattern))}

ttft = load("ttft-*.json")
cells = {}
for arm, data in ttft.items():
    for r in data["rows"]:
        if r["valid"]:
            cells.setdefault((r["position"], r["suffix"]), {}).setdefault(arm[0], []).append(r)
gain_ge60 = None
if cells:
    lines += ["", "## TTFT of a fresh suffix after a cached prefix (prefill_ttft.py; valid rows only; ms)", "",
              "| position | suffix | A runs | A TTFT | A ms/token | B runs | B TTFT | B ms/token | B vs A |",
              "|---|---|---|---|---|---|---|---|---|"]
    sum_a = sum_b = 0.0
    result["ttft"] = {}
    for (position, suffix), arms in sorted(cells.items()):
        a, b = arms.get("A", []), arms.get("B", [])
        ma = statistics.mean(r["prompt_ms"] for r in a) if a else None
        mb = statistics.mean(r["prompt_ms"] for r in b) if b else None
        ta = statistics.mean(r["ms_per_token"] for r in a) if a else None
        tb = statistics.mean(r["ms_per_token"] for r in b) if b else None
        change = mb / ma - 1 if ma and mb else None
        result["ttft"][f"{position}/{suffix}"] = {"A_ms": ma, "B_ms": mb, "change": change}
        if ma and mb and position >= 60000:
            sum_a += ma; sum_b += mb
        f = lambda v, d=0: "n/a" if v is None else f"{v:.{d}f}"
        lines.append(f"| {position} | {suffix} | {len(a)} | {f(ma)} | {f(ta, 3)} | {len(b)} | {f(mb)} | {f(tb, 3)} | "
                     f"{'n/a' if change is None else f'{change:+.1%}'} |")
    if sum_a and sum_b:
        gain_ge60 = 1 - sum_b / sum_a
        lines += ["", f"TTFT at >= 60K (cells at ~60K and ~100K pooled): B is {gain_ge60:.1%} faster than A."]
result["ttft_gain_ge60k"] = gain_ge60

park = load("park-*.json")
if park:
    lines += ["", "## park_long.py turns (~70K prefix, then ~3K-token turns; follow-up turns = turns 1..)", "",
              "| arm | follow-up prompt_ms | uncached tokens | ms/token | decode tok/s | wall s | texts vs A1 |",
              "|---|---|---|---|---|---|---|"]
    for arm, data in park.items():
        s = data["summary"]
        comparison = data.get("comparison")
        texts = "" if comparison is None else ("identical" if all(c["identical"] for c in comparison) else
                 "differ at turns " + ", ".join(str(c["turn"]) for c in comparison if not c["identical"]))
        per_token = s["prompt_ms_follow_ups"] / max(1, s["uncached_follow_ups"])
        lines.append(f"| {arm} | {s['prompt_ms_follow_ups']:.0f} | {s['uncached_follow_ups']} | {per_token:.3f} | "
                     f"{s.get('decode_tok_s', 0):.1f} | {s['wall_s']:.0f} | {texts} |")
    pa = [d["summary"] for k, d in park.items() if k.startswith("A")]
    pb = [d["summary"] for k, d in park.items() if k.startswith("B")]
    if pa and pb:
        ra = sum(x["prompt_ms_follow_ups"] for x in pa) / max(1, sum(x["uncached_follow_ups"] for x in pa))
        rb = sum(x["prompt_ms_follow_ups"] for x in pb) / max(1, sum(x["uncached_follow_ups"] for x in pb))
        result["park_turn_ms_per_token"] = {"A": ra, "B": rb, "change": rb / ra - 1}
        lines += ["", f"Turn prefill: A {ra:.3f} -> B {rb:.3f} ms per uncached token ({rb / ra - 1:+.1%}). "
                      "Texts can differ at a near-tie (exact-math change); a growing or incoherent divergence is a bug."]

switch = out / "json" / "switch.txt"
if switch.exists() and switch.read_text().strip():
    lines += ["", "## Flag took effect", "", "```", switch.read_text().strip(), "```"]
for name in ("greedy-copy0.json", "greedy-copy1.json", "ties-copy0.json", "ties-copy1.json", "dist.json", "fruit.json",
             "logits.json"):
    path = out / "json" / name
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                data = {k: v for k, v in data.items() if k not in ("rows", "cases")}
            result[name[:-5]] = data
        except Exception:
            pass
report = out / "logs" / "logits.report"
lines += ["", "## Correctness details", "", "```", json.dumps({k: result[k] for k in result
          if k.startswith(("ties", "dist", "fruit", "greedy", "logits"))}, indent=1), "```"]
if report.exists():
    lines += ["", "Logits per line (flag on vs off; control = flag off, first chunk 37 tokens):", "", "```",
              report.read_text().strip(), "```"]

correct = all(steps.get(s, "").startswith("pass") for s in ("greedy_digest", "logits", "near_tie", "distribution"))
if not correct:
    rec = "keep SPLASH_PREFILL_ATTN_V2 unset: a correctness step did not pass; investigate before any timing claim"
elif gain_ge60 is None:
    rec = "keep SPLASH_PREFILL_ATTN_V2 unset: no valid TTFT A/B at >= 60K"
elif gain_ge60 >= 0.05:  # 5% (2026-11-18): attention is 40-53% of a chunk, so a 21% kernel gain is ~8.5-11.5% end to end
    rec = (f"turn SPLASH_PREFILL_ATTN_V2=1 on by default: TTFT at >= 60K is {gain_ge60:.1%} faster and every "
           "correctness step passed")
else:
    rec = f"keep SPLASH_PREFILL_ATTN_V2 unset: TTFT at >= 60K is only {gain_ge60:.1%} faster (< 5%)"
result["recommendation"] = rec
lines += ["", "## Recommended default", "", rec]
try:
    lines += ["", "## Defender CPU", "", "```", (out / "logs" / "defender.log").read_text().strip(), "```"]
except Exception:
    pass
(out / "summary.md").write_text("\n".join(lines) + "\n")
(out / "summary.json").write_text(json.dumps(result, indent=1, default=str))
print("\n".join(lines))
EOF
exit $FAILED

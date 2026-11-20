#!/usr/bin/env bash
# SPLASH_VERIFY_ROWS_LONG (splash/dev-longctx, LONGCTX.md): every check and the long-context A/B, unattended, DSpine
# only (SPLASH_DRAFTER=dspine, tree, copy, turbo-int4). Results in files; run it from your own terminal.
#
#   splash/tools/longctx_ab.sh [--quick] OUTDIR
#
# Steps (a failing step marks the run failed; later independent steps still run):
#   1 build          make all + the engine test binaries + decode-run in splash/dev-longctx; dev-park2's decode-run
#                    built into OUTDIR/park2-build (make BUILD=...: dev-park2's own files are not touched)
#   2 cpu_tests      kv-first-engine, runtime-status, native-engine-loop, native-fd-transport (dev-longctx)
#   3 greedy_digest  knobs unset: dev-longctx == dev-park2, greedy digests of the 6 agent prompts, copy 0 and 1
#   4 near_tie       SPLASH_VERIFY_ROWS_LONG=8 SPLASH_VERIFY_ROWS_LONG_FROM=0 (8 rows everywhere) greedy vs knobs
#                    unset: identical, or the first difference is a near-tie of the target (|logit a - b| < 1.0);
#                    every rows-8 cycle must report its tail unverified (the knob took effect)
#   5 distribution   dist and fruit, T=0.6: two-sample chi-square per position, rows 8 (FROM=0) vs knobs unset
#   6 ab             timing at long context, T=0.6: A = 16 rows (knobs unset), B = 8 rows (FROM=0); A-B-B-A over
#                    seeds 1, 2 (--quick: A-B, seed 1); prompts ~/.cache/splash-dspine/prompts/long.txt (2 x ~70K)
#                    and long100k.txt (~103K, made by splash/tools/longctx_prompts.py when missing)
#   7 threshold      full mode only: SPLASH_VERIFY_ROWS_LONG=8 with FROM = 48K / 64K / 96K on a ~61K and a ~86K
#                    prompt (~/.cache/splash-longctx/prompts/agent5.txt lines 1 and 3), one run per prompt so the
#                    gating is checked per prompt, then the policy table over every measured prompt
# summary.md ends with the recommended default: keep 16 rows unless B wins tok/s at >= 64K.
# GPU rules: AC power (exit 3 on battery at the start), /tmp/splash-gpu.lock for the GPU steps, a Splash server this
# script did not start is never stopped (it waits in 5-min polls, LONGCTX_AB_WAIT_MINUTES, default 360; a run during
# which one came up is repeated once), >= LONGCTX_AB_MIN_GIB (13) GiB reclaimable before each run (waits up to
# 10 min), Defender under 10% for 30 s before each timed run (waits up to 3 min, logged).
# Outputs: summary.md, summary.json, logs/*.log, csv/*.csv, json/*.json, out/*.txt.
# Exit: 0 when every step passes, 1 otherwise, 3 on battery power.
set -uo pipefail
QUICK=0
if [[ ${1:-} == --quick ]]; then QUICK=1; shift; fi
OUT=${1:?usage: longctx_ab.sh [--quick] OUTDIR}
mkdir -p "$OUT/logs" "$OUT/csv" "$OUT/json" "$OUT/out"
OUT=$(cd "$OUT" && pwd)
STACK_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
S=$STACK_ROOT/splash
DEV=$S/dev-longctx
REF_SOURCE=$S/dev-park2
REF_BUILD=$OUT/park2-build
DEV_BUILD=$DEV/build
MODEL=$S/splash-src/install/models/.selections/fb0aa6e23bdecc7d24023adea3bcae72062d1c103ab1578902b00a2ebb3f500f
PROMPTS=${AB_PROMPTS:-$HOME/.cache/splash-dspine/prompts}
SWEEP_PROMPTS=$HOME/.cache/splash-longctx/prompts/agent5.txt
TOOLS=$S/tools/dspine
PY=/opt/homebrew/bin/python3
[[ -x $PY ]] || PY=python3
export PYTHONDONTWRITEBYTECODE=1
source "$STACK_ROOT/scripts/lib/system.sh"
if ! on_ac_power; then echo "longctx_ab: on battery power, refusing to run" >&2; exit 3; fi
WAIT_MINUTES=${LONGCTX_AB_WAIT_MINUTES:-360}
MIN_GIB=${LONGCTX_AB_MIN_GIB:-13}
if (( QUICK )); then SEEDS=(1); ORDER=(A B); SAMPLES=60; LONG_NEW=192
else SEEDS=(1 2); ORDER=(A B B A); SAMPLES=200; LONG_NEW=384; fi
DSPINE_ENV=(SPLASH_DRAFTER=dspine SPLASH_DRAFT_TREE=1 SPLASH_DRAFT_HEAD_ROWS=98304)
ROWS8_FROM0=(SPLASH_VERIFY_ROWS_LONG=8 SPLASH_VERIFY_ROWS_LONG_FROM=0)

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
defender_wait() {  # timed runs wait up to 3 min for Defender under 10% for 30 s (ANALYSIS.md section 10.1)
  local waited=0 calm=0
  while (( waited < 180 && calm < 3 )); do
    if (( $(defender_cpu | cut -d. -f1) < 10 )); then calm=$((calm + 1)); else calm=0; fi
    sleep 10; waited=$((waited + 10))
  done
  echo "$(date '+%H:%M:%S') $1 defender_cpu=$(defender_cpu) waited=${waited}s" >> "$OUT/logs/defender.log"
}
# decode NAME BUILD_DIR PROMPTS [decode-run args...] -- [env...]: one decode-run under the GPU gate, log
# logs/NAME.log. Timed runs (TIMED=1) wait for Defender first. A run during which the user's server came up is
# repeated once after it goes. Returns decode-run's status (1 when the GPU steps were given up).
TIMED=0
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
    (( TIMED )) && defender_wait "$name"
    (cd "$DEV" && env -u SPLASH_DRAFTER -u SPLASH_DRAFT_TREE -u SPLASH_DRAFT_COPY -u SPLASH_DRAFT_HEAD_ROWS \
       -u SPLASH_DSPINE_WINDOW -u SPLASH_VERIFY_ROWS_LONG -u SPLASH_VERIFY_ROWS_LONG_FROM \
       -u SPLASH_VERIFY_ROWS_LONG_KEEP_MASS -u SPLASH_VERIFY_ROWS_LONG_KEEP_MASS_PER_100K \
       ${envs[@]+"${envs[@]}"} /usr/bin/time -l "$build/engine-tests/decode-run" "$build/splash.metallib" "$MODEL" \
       "$prompts" ${args[@]+"${args[@]}"}) > "$OUT/logs/$name.log" 2>&1
    rc=$?
    if server_up; then
      note "$name: a foreign Splash server came up during the run; repeating it once it goes"
      mv "$OUT/logs/$name.log" "$OUT/logs/$name.interrupted$attempt.log"
      continue
    fi
    return $rc
  done
  return $rc
}
# gating LOG all|none: the rows-8 run left every cycle's tail unverified (all) or none (none).
gating() {
  local log=$1 expect=$2 line cut cycles
  line=$(grep -m1 '^verify rows:' "$log") || { echo "$log: no 'verify rows' line" >> "$OUT/json/gating.txt"; return 1; }
  cut=$(awk '{print $3}' <<<"$line"); cycles=$(awk '{print $5}' <<<"$line")
  echo "$(basename "$log" .log): $cut of $cycles cycles cut (expected $expect)" >> "$OUT/json/gating.txt"
  # The cut counter can count one more cycle than the cycle total (it also sees the cycle that ends the run, and a
  # decode-run with several prompts sums differently): "every cycle cut" means cut >= cycles (2026-11-17).
  if [[ $expect == all ]]; then (( cycles > 0 && cut >= cycles )); else (( cut == 0 )); fi
}
echo "longctx_ab $(date) quick=$QUICK out=$OUT" > "$OUT/logs/steps.log"
pmset -g batt | head -1 >> "$OUT/logs/steps.log"
: > "$OUT/json/gating.txt"

# ---- prompts: the ~103K one and the threshold sweep, deterministic (splash/tools/longctx_prompts.py)
if [[ ! -s $PROMPTS/long100k.txt ]]; then
  $PY "$S/tools/longctx_prompts.py" "$PROMPTS/long100k.txt" --targets 100000 --first-task 2 --more-sources \
    >> "$OUT/logs/prompts.log" 2>&1 || note "could not make $PROMPTS/long100k.txt"
fi
if (( ! QUICK )) && [[ ! -s $SWEEP_PROMPTS ]]; then
  mkdir -p "$(dirname "$SWEEP_PROMPTS")"
  $PY "$S/tools/longctx_prompts.py" "$SWEEP_PROMPTS" >> "$OUT/logs/prompts.log" 2>&1 || note "could not make $SWEEP_PROMPTS"
fi
if (( QUICK )); then head -1 "$PROMPTS/long.txt" > "$OUT/prompts_long.txt"; else cp "$PROMPTS/long.txt" "$OUT/prompts_long.txt"; fi
cat "$PROMPTS/long100k.txt" >> "$OUT/prompts_long.txt"
awk '{print "long prompt " NR - 1 ": " NF " tokens"}' "$OUT/prompts_long.txt" >> "$OUT/logs/steps.log"

# ---- 1 build
TESTS=(kv-first-engine runtime-status native-engine-loop native-fd-transport decode-run)
if (cd "$DEV" && make all && make "${TESTS[@]/#/build/engine-tests/}") > "$OUT/logs/build.log" 2>&1 \
   && (cd "$REF_SOURCE" && make BUILD="$REF_BUILD" all "$REF_BUILD/engine-tests/decode-run") > "$OUT/logs/build-park2.log" 2>&1
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
  for step in greedy_digest near_tie distribution ab threshold; do record $step skip "GPU busy with the user's server"; done
else
take_lock

# ---- 3 greedy digests, knobs unset: dev-longctx == dev-park2
ok=pass
for copy in 0 1; do
  decode greedy-copy$copy-park2 "$REF_BUILD" "$PROMPTS/agent.txt" --new-tokens 256 --temperature 0 \
    --output "$OUT/out/greedy-copy$copy-park2.txt" -- "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=$copy || ok=fail
  decode greedy-copy$copy-longctx "$DEV_BUILD" "$PROMPTS/agent.txt" --new-tokens 256 --temperature 0 \
    --output "$OUT/out/greedy-copy$copy-longctx.txt" -- "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=$copy || ok=fail
  $PY "$TOOLS/ab_tools.py" compare-digests "$OUT/logs/greedy-copy$copy-park2.log" "$OUT/logs/greedy-copy$copy-longctx.log" \
    > "$OUT/json/greedy-copy$copy.json" || ok=fail
done
record greedy_digest $ok

# ---- 4 rows 8 everywhere, greedy: equal to the 16-row text up to near-ties of the target; the knob took effect
ok=pass
for copy in 0 1; do
  decode rows8-greedy-copy$copy "$DEV_BUILD" "$PROMPTS/agent.txt" --new-tokens 256 --temperature 0 \
    --output "$OUT/out/rows8-greedy-copy$copy.txt" -- "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=$copy "${ROWS8_FROM0[@]}" || ok=fail
  gating "$OUT/logs/rows8-greedy-copy$copy.log" all || ok=fail
  $PY "$TOOLS/ab_tools.py" tie-prepare "$PROMPTS/agent.txt" "$OUT/out/greedy-copy$copy-longctx.txt" \
    "$OUT/out/rows8-greedy-copy$copy.txt" "$OUT/out/ties-copy$copy.score" "$OUT/json/ties-copy$copy.json" || ok=fail
  if [[ -s $OUT/out/ties-copy$copy.score ]]; then
    decode ties-copy$copy "$DEV_BUILD" "$PROMPTS/agent.txt" --score "$OUT/out/ties-copy$copy.score" -- \
      "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=$copy || ok=fail
  else : > "$OUT/logs/ties-copy$copy.log"; fi
  $PY "$TOOLS/ab_tools.py" tie-report "$OUT/json/ties-copy$copy.json" "$OUT/logs/ties-copy$copy.log" 1.0 \
    > "$OUT/logs/ties-copy$copy.report" || ok=fail
done
record near_tie $ok

# ---- 5 distribution: rows 8 (FROM=0) samples the target's law (two-sample chi-square vs 16 rows)
ok=pass
for test in dist:16 fruit:40; do
  name=${test%%:*}; tokens=${test##*:}
  decode $name-rows16 "$DEV_BUILD" "$PROMPTS/$name.txt" --new-tokens $tokens --temperature 0.6 --repeat $SAMPLES \
    --seed 1000 --output "$OUT/out/$name-rows16.txt" -- "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=1 || ok=fail
  decode $name-rows8 "$DEV_BUILD" "$PROMPTS/$name.txt" --new-tokens $tokens --temperature 0.6 --repeat $SAMPLES \
    --seed 5000 --output "$OUT/out/$name-rows8.txt" -- "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=1 "${ROWS8_FROM0[@]}" || ok=fail
  gating "$OUT/logs/$name-rows8.log" all || ok=fail
  $PY "$TOOLS/ab_tools.py" dist-compare "$OUT/out/$name-rows16.txt" "$OUT/out/$name-rows8.txt" $tokens \
    "$OUT/json/$name.json" > "$OUT/logs/$name.report" || ok=fail
done
record distribution $ok

# ---- 6 timing A/B at long context: A = 16 rows, B = 8 rows from 0; A-B-B-A (seed 1: A B, seed 2: B A)
ok=pass
TIMED=1
rm -f "$OUT/csv/ab_runs.csv"
i=0
for arm in "${ORDER[@]}"; do
  seed=${SEEDS[$(( i / 2 ))]}
  name=ab-$arm-seed$seed
  if [[ $arm == A ]]; then extra=(); else extra=("${ROWS8_FROM0[@]}"); fi
  decode $name "$DEV_BUILD" "$OUT/prompts_long.txt" --new-tokens $LONG_NEW --temperature 0.6 --seed $seed -- \
    "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=1 ${extra[@]+"${extra[@]}"} || ok=fail
  [[ $arm == B ]] && { gating "$OUT/logs/$name.log" all || ok=fail; }
  $PY "$TOOLS/ab_tools.py" parse-run "$OUT/logs/$name.log" $arm $seed "$OUT/csv/ab_runs.csv"
  grep -q '^total' "$OUT/logs/$name.log" || ok=fail
  i=$((i + 1))
done
record ab $ok

# ---- 7 threshold (full mode): FROM = 48K / 64K / 96K on a ~61K and a ~86K prompt, one run per prompt
if (( QUICK )); then record threshold skip "--quick"
elif [[ ! -s $SWEEP_PROMPTS ]]; then record threshold fail "no $SWEEP_PROMPTS"
else
  ok=pass
  rm -f "$OUT/csv/threshold_runs.csv"
  for line in 1 3; do sed -n "${line}p" "$SWEEP_PROMPTS" > "$OUT/prompts_sweep$line.txt"; done
  for from in 49152 65536 98304; do
    for line in 1 3; do
      tokens=$(awk '{print NF}' "$OUT/prompts_sweep$line.txt")
      name=threshold-from$from-sweep$line
      decode $name "$DEV_BUILD" "$OUT/prompts_sweep$line.txt" --new-tokens $LONG_NEW --temperature 0.6 --seed 1 -- \
        "${DSPINE_ENV[@]}" SPLASH_DRAFT_COPY=1 SPLASH_VERIFY_ROWS_LONG=8 SPLASH_VERIFY_ROWS_LONG_FROM=$from || ok=fail
      if (( tokens >= from )); then expect=all; else expect=none; fi
      gating "$OUT/logs/$name.log" $expect || ok=fail
      # The prompt column is the line in agent5.txt; the arm says what the run verified.
      $PY "$TOOLS/ab_tools.py" parse-run "$OUT/logs/$name.log" "F$from" "$line" "$OUT/csv/threshold_runs.csv"
    done
  done
  record threshold $ok
fi
release_lock
fi

# ---- summary
$PY - "$OUT" "$QUICK" <<'EOF'
import csv, json, sys, pathlib
from collections import defaultdict
out, quick = pathlib.Path(sys.argv[1]), sys.argv[2] == "1"
steps = {}
for line in (out / "logs" / "steps.log").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 4 and parts[1] == "step":
        steps[parts[2].rstrip(":")] = " ".join(parts[3:])

def rows(name):
    path = out / "csv" / name
    return list(csv.DictReader(open(path))) if path.exists() else []

def pool(rs):
    tokens = sum(float(r["tokens"]) for r in rs); cycles = sum(float(r["cycles"]) for r in rs)
    wall = sum(float(r["wall_ms_per_cycle"]) * float(r["cycles"]) for r in rs)
    gpu = sum(float(r["gpu_ms_per_cycle"]) * float(r["cycles"]) for r in rs)
    return {"runs": len(rs), "tokens": int(tokens), "cycles": int(cycles),
            "tok_per_cycle": tokens / cycles if cycles else 0.0, "wall_ms_per_cycle": wall / cycles if cycles else 0.0,
            "gpu_ms_per_cycle": gpu / cycles if cycles else 0.0, "tok_s": tokens / (wall / 1000) if wall else 0.0}

lines = [f"# longctx_ab summary ({'quick' if quick else 'full'}): SPLASH_VERIFY_ROWS_LONG", "",
         "| step | result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in steps.items()]
result = {"quick": quick, "steps": steps}

ab = rows("ab_runs.csv")
prompt_tokens = {r["prompt"]: int(r["prompt_tokens"]) for r in ab}
per_prompt = {}
if ab:
    lines += ["", "## Timing A/B (A = 16 rows, B = 8 rows from 0; T=0.6, DSpine tree + copy, turbo-int4)", "",
              "| prompt | tokens | arm | runs | tok/cycle | wall ms/cycle | gpu ms/cycle | tok/s | B vs A tok/s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(prompt_tokens, key=int):
        a = pool([r for r in ab if r["prompt"] == p and r["arm"] == "A"])
        b = pool([r for r in ab if r["prompt"] == p and r["arm"] == "B"])
        per_prompt[f"long{p}"] = {"prompt_tokens": prompt_tokens[p], "A": a, "B": b}
        for arm, d in (("A", a), ("B", b)):
            ratio = f"{b['tok_s'] / a['tok_s'] - 1:+.1%}" if arm == "B" and a["tok_s"] and b["tok_s"] else ""
            lines.append(f"| {p} | {prompt_tokens[p]} | {arm} | {d['runs']} | {d['tok_per_cycle']:.3f} | "
                         f"{d['wall_ms_per_cycle']:.1f} | {d['gpu_ms_per_cycle']:.1f} | {d['tok_s']:.2f} | {ratio} |")
    a_all, b_all = pool([r for r in ab if r["arm"] == "A"]), pool([r for r in ab if r["arm"] == "B"])
    result["ab"] = {"A": a_all, "B": b_all, "prompts": per_prompt}
    if a_all["tok_s"] and b_all["tok_s"]:
        lines += ["", f"Overall: A {a_all['tok_per_cycle']:.3f} tok/cycle, {a_all['wall_ms_per_cycle']:.1f} ms/cycle, "
                      f"{a_all['tok_s']:.2f} tok/s; B {b_all['tok_per_cycle']:.3f} tok/cycle, "
                      f"{b_all['wall_ms_per_cycle']:.1f} ms/cycle, {b_all['tok_s']:.2f} tok/s "
                      f"({b_all['tok_s'] / a_all['tok_s'] - 1:+.1%})."]

threshold = rows("threshold_runs.csv")
if threshold:
    # Threshold runs: A-like (16 rows) and B-like (8 rows) per sweep prompt, by whether the prompt was >= FROM.
    lines += ["", "## Threshold runs (FROM = 48K / 64K / 96K; seed 1, one run per prompt)", "",
              "| FROM | agent5 line | tokens | verified | tok/cycle | wall ms/cycle | tok/s |", "|---|---|---|---|---|---|---|"]
    for r in threshold:
        start, tokens = int(r["arm"][1:]), int(r["prompt_tokens"])
        rows8 = tokens >= start
        lines.append(f"| {start} | {r['seed']} | {tokens} | {'8' if rows8 else '16'} | {float(r['tokens_per_cycle']):.3f} | "
                     f"{float(r['wall_ms_per_cycle']):.1f} | {float(r['tokens']) / (float(r['cycles']) * float(r['wall_ms_per_cycle']) / 1000):.2f} |")
        key = f"sweep{r['seed']}"
        entry = per_prompt.setdefault(key, {"prompt_tokens": tokens, "A": [], "B": []})
        entry["B" if rows8 else "A"].append(r)
    for key, entry in per_prompt.items():
        if key.startswith("sweep"):
            entry["A"], entry["B"] = pool(entry["A"]), pool(entry["B"])
    # The policy table: FROM = X verifies 8 rows on prompts >= X (their B numbers), 16 below (their A numbers).
    lines += ["", "## Policy over every measured prompt (8 rows from FROM, 16 below)", "",
              "| policy | tok/cycle | wall ms/cycle | tok/s | vs always 16 |", "|---|---|---|---|---|"]
    def policy(start):
        tokens = cycles = wall = 0.0
        for entry in per_prompt.values():
            d = entry["B"] if start is not None and entry["prompt_tokens"] >= start else entry["A"]
            if not d.get("cycles"):
                return None
            tokens += d["tokens"]; cycles += d["cycles"]; wall += d["wall_ms_per_cycle"] * d["cycles"]
        return {"tok_per_cycle": tokens / cycles, "wall_ms_per_cycle": wall / cycles, "tok_s": tokens / (wall / 1000)}
    base = policy(None)
    result["policy"] = {}
    for label, start in (("always 16", None), ("FROM 48K", 49152), ("FROM 64K", 65536), ("FROM 96K", 98304)):
        p = policy(start)
        result["policy"][label] = p
        if p and base:
            lines.append(f"| {label} | {p['tok_per_cycle']:.3f} | {p['wall_ms_per_cycle']:.1f} | {p['tok_s']:.2f} | "
                         f"{p['tok_s'] / base['tok_s'] - 1:+.1%} |")

gating = out / "json" / "gating.txt"
if gating.exists() and gating.read_text().strip():
    lines += ["", "## Gating (cycles that verified 8 rows)", "", "```", gating.read_text().strip(), "```"]
for name in ("ties-copy0.json", "ties-copy1.json", "dist.json", "fruit.json", "greedy-copy0.json", "greedy-copy1.json"):
    path = out / "json" / name
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                data = {k: v for k, v in data.items() if k not in ("rows", "cases")}
            result[name[:-5]] = data
        except Exception:
            pass
lines += ["", "## Correctness details", "", "```", json.dumps({k: result[k] for k in result
          if k.startswith(("ties", "dist", "fruit", "greedy"))}, indent=1), "```"]

# Recommendation: keep 16 unless B wins tok/s at >= 64K (by >= 2%, and every correctness step passed).
correct = all(steps.get(s, "").startswith("pass") for s in ("greedy_digest", "near_tie", "distribution"))
def band_gain(low, high):
    a = [e["A"] for e in per_prompt.values() if low <= e["prompt_tokens"] < high and e["A"].get("cycles")]
    b = [e["B"] for e in per_prompt.values() if low <= e["prompt_tokens"] < high and e["B"].get("cycles")]
    if not a or not b:
        return None
    sa = sum(d["tokens"] for d in a) / (sum(d["wall_ms_per_cycle"] * d["cycles"] for d in a) / 1000)
    sb = sum(d["tokens"] for d in b) / (sum(d["wall_ms_per_cycle"] * d["cycles"] for d in b) / 1000)
    return sb / sa - 1
gain_mid, gain_high = band_gain(65536, 98304), band_gain(98304, 10**9)
result["band_gain"] = {"64K-96K": gain_mid, ">=96K": gain_high}
fmt = lambda g: "n/a" if g is None else f"{g:+.1%}"
if not correct:
    rec = "keep 16 rows (SPLASH_VERIFY_ROWS_LONG unset): a correctness step failed; investigate before any timing claim"
elif gain_mid is not None and gain_mid >= 0.02 and (gain_high is None or gain_high >= 0.02):
    candidates = {k: v for k, v in (result.get("policy") or {}).items() if v and k != "always 16"}
    best = max(candidates, key=lambda k: candidates[k]["tok_s"]) if candidates else "FROM 64K"
    start = {"FROM 48K": 49152, "FROM 64K": 65536, "FROM 96K": 98304}[best]
    rec = (f"adopt SPLASH_VERIFY_ROWS_LONG=8 SPLASH_VERIFY_ROWS_LONG_FROM={start}: B wins tok/s by {fmt(gain_mid)} "
           f"at 64-96K and {fmt(gain_high)} at >=96K")
elif gain_high is not None and gain_high >= 0.02:
    rec = (f"adopt SPLASH_VERIFY_ROWS_LONG=8 SPLASH_VERIFY_ROWS_LONG_FROM=98304: B wins tok/s by {fmt(gain_high)} at "
           f">=96K but only {fmt(gain_mid)} at 64-96K")
else:
    rec = (f"keep 16 rows (SPLASH_VERIFY_ROWS_LONG unset): B does not win tok/s by >= 2% at >= 64K "
           f"(64-96K {fmt(gain_mid)}, >=96K {fmt(gain_high)})")
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

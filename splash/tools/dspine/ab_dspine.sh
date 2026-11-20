#!/bin/zsh
# DSpine stage 2: every check and the A/B, unattended, results in files.
#
#   ab_dspine.sh [--quick] OUTDIR
#
# Steps (a failing step marks the run failed; later independent steps still run):
#   1 build        make all + decode-run + dspine-golden in splash/dev-dspine
#   2 dflash2      dflash2 greedy digests, dev-dspine == production (chain, tree, tree + copy; 6 agent prompts)
#   3 golden       dspine-golden against dev/golden/dspine_q4 (Metal port vs MLX reference, bf16 tolerances)
#   4 dspine_greedy  dspine greedy runs (tree, copy 0/1): every divergence from DFlash2's greedy text must be a
#                  near-tie of the target (|logit a - logit b| < 1.0 at the first differing token, by --score)
#   5 distribution dist and fruit: two-sample chi-square per position, dspine vs dflash2, T = 0.6 (exact sampling)
#   6 ab           A = dflash2 production (tree-16, copy, shortlist), B = dspine block 16 window 8192;
#                  T = 0.6, seeds A-B / B-A / A-B, session prompts; then short-context (agent prompts) and
#                  long-context (2 x ~70K) timing with A, B and C = dspine window 2048 (AB_SKIP_C=1 skips C)
# --quick: 1 seed, 3 session prompts, fewer tokens and samples.
# Outputs: summary.json, summary.md, csv/*.csv (raw per prompt), logs/<step>*.log, json/*.json.
# Exit: 0 when every step passes, 1 otherwise, 3 on battery power. Takes /tmp/splash-gpu.lock for the whole run.
set -u
QUICK=0
if [[ ${1:-} == --quick ]]; then QUICK=1; shift; fi
OUT=${1:?usage: ab_dspine.sh [--quick] OUTDIR}
mkdir -p $OUT/logs $OUT/csv $OUT/json $OUT/out
OUT=$(cd $OUT && pwd)
if ! pmset -g batt | grep -q "AC Power"; then echo "ab_dspine: on battery power, refusing to run" >&2; exit 3; fi

S=$(cd "$(dirname $0)/../.." && pwd)        # splash/
DEV=$S/dev-dspine
PROD=$S/splash-src
TOOLS=$S/tools/dspine
MODEL=$PROD/install/models/.selections/fb0aa6e23bdecc7d24023adea3bcae72062d1c103ab1578902b00a2ebb3f500f
P=${AB_PROMPTS:-$HOME/.cache/splash-dspine/prompts}
PY=/opt/homebrew/bin/python3
export PYTHONDONTWRITEBYTECODE=1
if (( QUICK )); then SEEDS=(1); NEW=200; SAMPLES=60; LONG_NEW=128; head -3 $P/sessions.txt > $OUT/prompts_ab.txt; head -1 $P/long.txt > $OUT/prompts_long.txt
else SEEDS=(1 2 3); NEW=400; SAMPLES=200; LONG_NEW=256; cp $P/sessions.txt $OUT/prompts_ab.txt; cp $P/long.txt $OUT/prompts_long.txt; fi

until mkdir /tmp/splash-gpu.lock 2>/dev/null; do sleep 10; done
trap 'rmdir /tmp/splash-gpu.lock 2>/dev/null' EXIT
trap 'exit 130' INT TERM HUP

typeset -A RESULT
FAILED=0
record() { RESULT[$1]=$2; [[ $2 == fail ]] && FAILED=1; echo "$(date '+%H:%M:%S') step $1: $2" | tee -a $OUT/logs/steps.log; }
defender_cpu() { ps -Ao pcpu,comm | grep -i -E 'wdav|defender' | grep -v grep | awk '{s+=$1} END {printf "%.1f", s}'; }
defender() {  # Microsoft Defender's CPU % (all its processes) at the start of a block; timed blocks (quiet=1) first
              # wait up to 3 min for it to stay under 10% for 30 s (ANALYSIS.md section 10.1)
  local waited=0 calm=0
  if [[ ${2:-0} == 1 ]]; then
    while (( waited < 180 && calm < 3 )); do
      if (( $(defender_cpu | cut -d. -f1) < 10 )); then calm=$((calm + 1)); else calm=0; fi
      sleep 10; waited=$((waited + 10))
    done
  fi
  echo "$(date '+%H:%M:%S') $1 defender_cpu=$(defender_cpu) waited=${waited}s" >> $OUT/logs/defender.log
}
# decode NAME BUILD PROMPTS [decode-run args...] -- [env...]: one decode-run, its log logs/NAME.log (with /usr/bin/time -l)
decode() {
  local name=$1 build=$2 prompts=$3; shift 3
  local args=() envs=()
  while (( $# )) && [[ $1 != -- ]]; do args+=$1; shift; done
  [[ ${1:-} == -- ]] && shift
  envs=("$@")
  (cd $S/$build && env -u SPLASH_DRAFTER -u SPLASH_DRAFT_TREE -u SPLASH_DRAFT_COPY -u SPLASH_DRAFT_HEAD_ROWS \
     -u SPLASH_DSPINE_WINDOW "${envs[@]}" /usr/bin/time -l build/engine-tests/decode-run build/splash.metallib $MODEL \
     $prompts "${args[@]}") > $OUT/logs/$name.log 2>&1
}
TREE=(SPLASH_DRAFT_TREE=1 SPLASH_DRAFT_HEAD_ROWS=98304)
echo "ab_dspine $(date) quick=$QUICK out=$OUT" > $OUT/logs/steps.log
pmset -g batt | head -1 >> $OUT/logs/steps.log

# ---- 1 build
defender build
if (cd $DEV && make all && make build/engine-tests/decode-run build/engine-tests/dspine-golden) > $OUT/logs/build.log 2>&1
then record build pass; else record build fail; fi

# ---- 2 dflash2 digests vs production
defender dflash2
ok=pass
for cfg in chain tree copy; do
  case $cfg in chain) e=(SPLASH_DRAFT_TREE=0);; tree) e=($TREE SPLASH_DRAFT_COPY=0);; copy) e=($TREE SPLASH_DRAFT_COPY=1);; esac
  for b in splash-src dev-dspine; do
    decode dflash2-$cfg-$b $b $P/agent.txt --new-tokens 256 --temperature 0 --output $OUT/out/dflash2-$cfg-$b.txt -- $e
  done
  $PY $TOOLS/ab_tools.py compare-digests $OUT/logs/dflash2-$cfg-splash-src.log $OUT/logs/dflash2-$cfg-dev-dspine.log \
    > $OUT/json/dflash2-$cfg.json || ok=fail
done
record dflash2 $ok

# ---- 3 golden tolerances
defender golden
if (cd $DEV && SPLASH_DRAFTER=dspine SPLASH_DSPINE_WINDOW=2048 build/engine-tests/dspine-golden build/splash.metallib \
      $MODEL dev/golden/dspine_q4 --report $OUT/json/golden.json) > $OUT/logs/golden.log 2>&1
then record golden pass; else record golden fail; fi

# ---- 4 dspine greedy: equal to DFlash2's greedy text up to near-ties
defender dspine_greedy
ok=pass
for copy in 0 1; do
  ref=$([[ $copy == 0 ]] && echo tree || echo copy)
  decode dspine-greedy-copy$copy dev-dspine $P/agent.txt --new-tokens 256 --temperature 0 \
    --output $OUT/out/dspine-greedy-copy$copy.txt -- SPLASH_DRAFTER=dspine $TREE SPLASH_DRAFT_COPY=$copy
  $PY $TOOLS/ab_tools.py tie-prepare $P/agent.txt $OUT/out/dflash2-$ref-dev-dspine.txt \
    $OUT/out/dspine-greedy-copy$copy.txt $OUT/out/ties-copy$copy.score $OUT/json/ties-copy$copy.json
  if [[ -s $OUT/out/ties-copy$copy.score ]]; then
    decode ties-copy$copy dev-dspine $P/agent.txt --score $OUT/out/ties-copy$copy.score --
  else : > $OUT/logs/ties-copy$copy.log; fi
  $PY $TOOLS/ab_tools.py tie-report $OUT/json/ties-copy$copy.json $OUT/logs/ties-copy$copy.log 1.0 \
    > $OUT/logs/ties-copy$copy.report || ok=fail
  grep -q '^total' $OUT/logs/dspine-greedy-copy$copy.log || ok=fail
done
record dspine_greedy $ok

# ---- 5 distribution tests (exact sampling: dspine's sampled law = dflash2's = the target's)
defender distribution
ok=pass
for test in dist:16 fruit:40; do
  name=${test%%:*}; tokens=${test##*:}
  decode $name-dflash2 dev-dspine $P/$name.txt --new-tokens $tokens --temperature 0.6 --repeat $SAMPLES --seed 1000 \
    --output $OUT/out/$name-dflash2.txt -- $TREE SPLASH_DRAFT_COPY=1
  decode $name-dspine dev-dspine $P/$name.txt --new-tokens $tokens --temperature 0.6 --repeat $SAMPLES --seed 5000 \
    --output $OUT/out/$name-dspine.txt -- SPLASH_DRAFTER=dspine $TREE SPLASH_DRAFT_COPY=1
  $PY $TOOLS/ab_tools.py dist-compare $OUT/out/$name-dflash2.txt $OUT/out/$name-dspine.txt $tokens \
    $OUT/json/$name.json > $OUT/logs/$name.report || ok=fail
done
record distribution $ok

# ---- 6 A/B
ok=pass
arm_env() { case $1 in A) print -r -- "$TREE SPLASH_DRAFT_COPY=1";; B) print -r -- "SPLASH_DRAFTER=dspine $TREE SPLASH_DRAFT_COPY=1";;
            C) print -r -- "SPLASH_DRAFTER=dspine SPLASH_DSPINE_WINDOW=2048 $TREE SPLASH_DRAFT_COPY=1";; esac; }
rm -f $OUT/csv/ab_runs.csv $OUT/csv/short_runs.csv $OUT/csv/long_runs.csv
i=0
for seed in $SEEDS; do
  if (( i % 2 == 0 )); then order=(A B); else order=(B A); fi
  for arm in $order; do
    defender ab-$arm-seed$seed 1
    decode ab-$arm-seed$seed dev-dspine $OUT/prompts_ab.txt --new-tokens $NEW --temperature 0.6 --seed $seed -- ${=$(arm_env $arm)}
    $PY $TOOLS/ab_tools.py parse-run $OUT/logs/ab-$arm-seed$seed.log $arm $seed $OUT/csv/ab_runs.csv
    grep -q '^total' $OUT/logs/ab-$arm-seed$seed.log || ok=fail
  done
  i=$((i + 1))
done
for arm in A B; do
  defender short-$arm 1
  decode short-$arm dev-dspine $P/agent.txt --new-tokens $NEW --temperature 0.6 --seed 1 -- ${=$(arm_env $arm)}
  $PY $TOOLS/ab_tools.py parse-run $OUT/logs/short-$arm.log $arm 1 $OUT/csv/short_runs.csv
done
LONG_ARMS=(A B C); [[ ${AB_SKIP_C:-0} == 1 ]] && LONG_ARMS=(A B)
for arm in $LONG_ARMS; do
  defender long-$arm 1
  decode long-$arm dev-dspine $OUT/prompts_long.txt --new-tokens $LONG_NEW --temperature 0.6 --seed 1 -- ${=$(arm_env $arm)}
  $PY $TOOLS/ab_tools.py parse-run $OUT/logs/long-$arm.log $arm 1 $OUT/csv/long_runs.csv
done
for set in ab short long; do
  [[ -s $OUT/csv/${set}_runs.csv ]] && $PY $TOOLS/ab_tools.py summarize $OUT/csv/${set}_runs.csv $OUT/json/$set.json > /dev/null
done
record ab $ok

# ---- summary
$PY - "$OUT" "$QUICK" "${RESULT[build]}" "${RESULT[dflash2]}" "${RESULT[golden]}" "${RESULT[dspine_greedy]}" \
     "${RESULT[distribution]}" "${RESULT[ab]}" <<'EOF'
import json, os, re, sys
out, quick = sys.argv[1], sys.argv[2] == "1"
steps = dict(zip(["build", "dflash2", "golden", "dspine_greedy", "distribution", "ab"], sys.argv[3:9]))
def load(name):
    try: return json.load(open(os.path.join(out, "json", name)))
    except Exception: return None
def rss(log):
    try: text = open(os.path.join(out, "logs", log)).read()
    except Exception: return None
    m = re.search(r"(\d+)\s+maximum resident set size", text); f = re.search(r"(\d+)\s+peak memory footprint", text)
    return {"max_rss_mib": int(m.group(1)) >> 20 if m else None, "peak_footprint_mib": int(f.group(1)) >> 20 if f else None}
summary = {"quick": quick, "steps": steps, "pass": all(v == "pass" for v in steps.values()),
           "golden": load("golden.json"), "ties": {c: (lambda j: j and {k: v for k, v in j.items() if k != "cases"})(load(f"ties-{c}.json")) for c in ("copy0", "copy1")},
           "distribution": {t: (lambda j: j and {k: v for k, v in j.items() if k != "rows"})(load(f"{t}.json")) for t in ("dist", "fruit")},
           "ab": load("ab.json"), "short": load("short.json"), "long": load("long.json"),
           "memory": {"A": rss("ab-A-seed1.log"), "B": rss("ab-B-seed1.log")}}
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
lines = [f"# DSpine stage 2 run ({'quick' if quick else 'full'})", "", "| step | result |", "|---|---|"]
lines += [f"| {k} | {v} |" for k, v in steps.items()]
def arms_table(title, data):
    if not data: return []
    rows = [f"", f"## {title}", "", "| arm | runs | tokens | cycles | tok/cycle | gpu ms/cycle | wall ms/cycle | tok/s |", "|---|---|---|---|---|---|---|---|"]
    for arm, d in sorted(data["arms"].items()):
        rows.append(f"| {arm} | {d['runs']} | {d['tokens']} | {d['cycles']} | {d['tokens_per_cycle']:.3f} | {d['gpu_ms_per_cycle']:.2f} | {d['wall_ms_per_cycle']:.2f} | {d['tokens_per_second']:.2f} |")
    rows += ["", "| prompt | " + " | ".join(f"{a} tok/cyc, ms, tok/s" for a in sorted(data["arms"])) + " |", "|---|" + "---|" * len(data["arms"])]
    for p, arms in data["prompts"].items():
        rows.append(f"| {p} | " + " | ".join(f"{arms[a]['tokens_per_cycle']:.2f}, {arms[a]['wall_ms_per_cycle']:.1f}, {arms[a]['tokens_per_second']:.1f}" if a in arms else "-" for a in sorted(data["arms"])) + " |")
    return rows
lines += arms_table("A/B (session prompts, T=0.6)", summary["ab"])
lines += arms_table("Short context (agent prompts)", summary["short"])
lines += arms_table("Long context (~70K)", summary["long"])
lines += ["", "## Memory (decode-run, /usr/bin/time -l)", "", f"A: {summary['memory']['A']}", f"B: {summary['memory']['B']}"]
g = summary["golden"]
if g: lines += ["", "## Golden", "", json.dumps({k: g[k] for k in g if k in ('final_normed', 'candidate_overlap_mean', 'table_abs_median', 'table_abs_p99', 'chain_first_agree', 'chain_prefix_mean', 'pass')})]
lines += ["", "## Ties", "", json.dumps(summary["ties"]), "", "## Distribution", "", json.dumps(summary["distribution"])]
try: lines += ["", "## Defender CPU", "", "```", open(os.path.join(out, "logs", "defender.log")).read().strip(), "```"]
except Exception: pass
open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:12]))
EOF
exit $FAILED

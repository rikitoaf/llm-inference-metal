#!/bin/zsh
# A/B jobs that take /tmp/splash-gpu.lock per decode-run (so other GPU users interleave), with the Defender gate.
#   ab_jobs.sh OUTDIR PROMPTS NEW_TOKENS "ARM:SEED ..." [BIN_DIR]
# ARM is A (dflash2 production), B (dspine 8192), C (dspine 2048), N (dspine 8192 copy off), or ENV=... list given
# as AB_ARM_<ARM>="SPLASH_...=x ..." in the environment. BIN_DIR holds decode-run and splash.metallib (default
# splash/dev-dspine/build). Appends per-prompt rows to OUTDIR/runs.csv; logs in OUTDIR/logs.
set -u
OUT=$1; PROMPTS=$2; NEW=$3; JOBS=(${=4}); BIN=${5:-}
S=$(cd "$(dirname $0)/../.." && pwd)
MODEL=$S/splash-src/install/models/.selections/fb0aa6e23bdecc7d24023adea3bcae72062d1c103ab1578902b00a2ebb3f500f
[[ -z $BIN ]] && BIN=$S/dev-dspine/build
EXE=$BIN/decode-run; [[ -x $EXE ]] || EXE=$BIN/engine-tests/decode-run
mkdir -p $OUT/logs
TREE="SPLASH_DRAFT_TREE=1 SPLASH_DRAFT_HEAD_ROWS=98304"
defender_cpu() { ps -Ao pcpu,comm | grep -i -E 'wdav|defender' | grep -v grep | awk '{s+=$1} END {printf "%.0f", s}'; }
for job in $JOBS; do
  arm=${job%%:*}; seed=${job##*:}
  var=AB_ARM_$arm
  case $arm in
    A) E="$TREE SPLASH_DRAFT_COPY=1";;
    B) E="SPLASH_DRAFTER=dspine $TREE SPLASH_DRAFT_COPY=1";;
    C) E="SPLASH_DRAFTER=dspine SPLASH_DSPINE_WINDOW=2048 $TREE SPLASH_DRAFT_COPY=1";;
    N) E="SPLASH_DRAFTER=dspine $TREE SPLASH_DRAFT_COPY=0";;
    *) E=${(P)var};;
  esac
  pmset -g batt | grep -q "AC Power" || { echo "on battery" >&2; exit 3; }
  until mkdir /tmp/splash-gpu.lock 2>/dev/null; do sleep 15; done
  trap 'rmdir /tmp/splash-gpu.lock 2>/dev/null' EXIT INT TERM
  calm=0; waited=0
  while (( waited < 180 && calm < 3 )); do (( $(defender_cpu) < 10 )) && calm=$((calm+1)) || calm=0; sleep 10; waited=$((waited+10)); done
  echo "$(date '+%H:%M:%S') $arm:$seed defender=$(defender_cpu) waited=${waited}s env=$E" >> $OUT/logs/jobs.log
  (cd $S/dev-dspine && env -u SPLASH_DRAFTER -u SPLASH_DRAFT_COPY -u SPLASH_DSPINE_WINDOW ${=E} /usr/bin/time -l $EXE $BIN/splash.metallib $MODEL \
      $PROMPTS --new-tokens $NEW --temperature 0.6 --seed $seed) > $OUT/logs/$arm-$seed.log 2>&1
  rmdir /tmp/splash-gpu.lock; trap - EXIT INT TERM
  /opt/homebrew/bin/python3 $S/tools/dspine/ab_tools.py parse-run $OUT/logs/$arm-$seed.log $arm $seed $OUT/runs.csv
done
echo done >> $OUT/logs/jobs.log

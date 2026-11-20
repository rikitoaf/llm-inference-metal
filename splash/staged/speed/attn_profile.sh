#!/usr/bin/env bash
# decode-profile (turbo-int4, one lane, the real Runtime) of splash/dev-merged per verify-attention tiling and context,
# plus the draft-head shortlist at 2K. Greedy: the printed digest must match across variants of one context.
#   splash/staged/speed/attn_profile.sh OUTDIR [contexts...]     default contexts: 2048 32768 65536
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
source splash/staged/speed/quiet.sh
require_measurable || exit 1
OUT="$ROOT/$1"; shift
(( $# )) || set -- 2048 32768 65536
mkdir -p "$OUT"
cd splash/dev-merged
make all build/engine-tests/decode-profile >/dev/null 2>&1 || { echo "build failed"; exit 1; }
# The model root benchmark-decode-profile would use (make -n prints its command line).
MODEL_ROOT=$(make -n MODEL=unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS LANGUAGE_ONLY=1 benchmark-decode-profile 2>/dev/null |
  sed -n 's#^build/engine-tests/decode-profile build/splash.metallib "\(.*\)".*#\1#p' | tail -1)
[[ -d "$MODEL_ROOT" ]] || { echo "no MODEL_ROOT"; exit 1; }
run() {   # name context env...
  local name=$1 context=$2; shift 2
  wait_quiet 30 20 900
  env SPLASH_DRAFT_TREE=1 "$@" build/engine-tests/decode-profile build/splash.metallib "$MODEL_ROOT" \
    --prompt-tokens "$context" --cycles 12 --kv-format turbo-int4 --prefill-profile 0 > "$OUT/profile-$name-$context.txt" 2>&1
  echo "== $name ctx=$context rc=$? defender=$(defender_cpu)%"
  grep -E 'median fused gpu|digest' "$OUT/profile-$name-$context.txt" | tail -6
}
for context in "$@"; do
  run tile8 "$context" SPLASH_VERIFY_TILE=8
  run tile8x2 "$context" SPLASH_VERIFY_TILE=8x2
  run tile16 "$context" SPLASH_VERIFY_TILE=16
done
run head0 2048
run head98k 2048 SPLASH_DRAFT_HEAD_ROWS=98304

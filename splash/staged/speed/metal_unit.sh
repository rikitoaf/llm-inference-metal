#!/usr/bin/env bash
# make test-engine-metal's tests that run on the runtime-compiled library (build/splash.metallib, a directory of
# flattened sources), for machines with only the Command Line Tools (no offline `metal`, so the .air-based test
# libraries cannot be built). gguf-projection gets the library plus its test kernel, as correctness.sh does.
#   splash/staged/speed/metal_unit.sh splash/dev-merged [ENV=VALUE ...]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$ROOT/$1"; shift
cd "$SRC" || exit 1
TESTS=(gguf-preparation gguf-rotation gguf-moe tuning-workloads linear-tuning paged-attention-plan linear-plan
       runtime-resources q4-prefill-projection moe-metal gdn-metal dflash-batch-control draft-attention gdn-decode draft-selector
       target-sampling gguf-projection)
make all >/dev/null 2>&1 || { echo "make all failed"; exit 1; }
for t in "${TESTS[@]}"; do make "build/engine-tests/$t" >/dev/null 2>&1 || echo "BUILD-FAIL $t"; done
LIB=build/splash.metallib
COMBINED="${TMPDIR:-/tmp}/metal-unit-lib-$$"
rm -rf "$COMBINED"; cp -R "$LIB" "$COMBINED"
python3 dev/tools/flatten_metal.py runtime dev/tests/engine/metal_backend_test.metal > "$COMBINED/test__metal_backend_test.metal"
status=0
for t in "${TESTS[@]}"; do
  [[ -x "build/engine-tests/$t" ]] || continue
  case $t in
    gguf-preparation) args=("$LIB" dev/tests/fixtures/weight-goldens/goldens.json) ;;
    gguf-projection) args=("$COMBINED") ;;
    *) args=("$LIB") ;;
  esac
  start=$(date +%s)
  if out=$(env "$@" MTL_SHADER_VALIDATION=1 "build/engine-tests/$t" "${args[@]}" 2>&1); then
    echo "PASS $t ($(( $(date +%s) - start ))s): $(echo "$out" | tail -1)"
  else
    echo "FAIL $t: $(echo "$out" | tail -3 | tr '\n' ' ')"; status=1
  fi
done
rm -rf "$COMBINED"
exit $status

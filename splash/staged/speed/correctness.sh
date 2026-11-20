#!/usr/bin/env bash
# Correctness of kernel candidates: runs Splash's GGUF projection test (dev/tests/engine/gguf_projection_test.mm:
# every format, decode and prefill tiles, against its CPU reference) on the libraries kernel_ab.sh builds. A
# library here is a directory of flattened sources (RUNTIME_METAL builds), so the test's one extra kernel
# (test_copy_u32, from metal_backend_test.metal) is added as another flattened file. GPU for under a minute per
# library; Defender does not matter for correctness.
#
#   splash/staged/speed/correctness.sh base grid
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
SRC=splash/splash-src
CACHE="${TMPDIR:-/tmp}/splash-kernel-ab"
TEST=$SRC/build/engine-tests/gguf-projection
[[ -x "$TEST" ]] || (cd "$SRC" && make build/engine-tests/gguf-projection) || exit 1
(( $# )) || set -- base grid
status=0
for lib in "$@"; do
  [[ -d "$CACHE/lib-$lib" ]] || { echo "$lib: no library; run kernel_ab.sh --build-only --libs \"$*\" first" >&2; status=1; continue; }
  combined="$CACHE/test-$lib"
  rm -rf "${combined:?}"; cp -R "$CACHE/lib-$lib" "$combined"
  python3 "$SRC/dev/tools/flatten_metal.py" "$CACHE/$lib/runtime" "$SRC/dev/tests/engine/metal_backend_test.metal" \
    > "$combined/test__metal_backend_test.metal" || { echo "$lib: flattening the test kernel failed" >&2; status=1; continue; }
  if out=$("$TEST" "$combined" 2>&1); then
    echo "$lib: PASS ($(echo "$out" | tail -1))"
  else
    echo "$lib: FAIL"; echo "$out" | tail -20; status=1
  fi
done
exit $status

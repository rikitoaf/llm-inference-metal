#!/usr/bin/env bash
# Apply the staged step-2 changes (splash/IMPROVEMENT_PLAN.md) to splash/splash-src once no server or
# suite is running, then run the TurboQuant kernel tests. The originals go to splash/staged/orig/.
set -euo pipefail
cd "$(dirname "$0")"
if pgrep -f "run_kv_suite|splash serve-native" >/dev/null; then echo "error: a suite or server is running" >&2; exit 1; fi
SRC=../splash-src; TOOLS=../tools
mkdir -p orig
while read -r f dest; do
  cp "$dest" "orig/$f"; cp "$f" "$dest"; echo "applied $f"
done <<EOT
paged_store_row.h $SRC/runtime/metal/kernels/common/paged_store_row.h
turbo_kv.h $SRC/runtime/metal/kernels/common/turbo_kv.h
PagedKv.hpp $SRC/runtime/ops/PagedKv.hpp
RuntimeResources.mm $SRC/runtime/engine/RuntimeResources.mm
Makefile $SRC/Makefile
turbo_int4_test.py $TOOLS/turbo_int4_test.py
EOT
make -C "$TOOLS" test

#!/usr/bin/env bash
# 2026-10-30 GPU session. Phase 1, A-B-B-A: is dev-merged with flags off really slower than splash-dev (10-29: -11%)?
# Phase 2: the flags, then production again. Phase 3: park/sysstate checks and the attention profile.
cd "$(dirname "$0")/../../.."
export QUIET_TIMEOUT=300
O=logs/speed/ab-1030
splash/staged/speed/merged_ab.sh $O base1:splash-dev: off1:dev-merged: off2:dev-merged: base2:splash-dev:
echo "PHASE1 DONE"
splash/staged/speed/merged_ab.sh $O head:dev-merged:SPLASH_DRAFT_HEAD_ROWS=98304 tile16:dev-merged:SPLASH_VERIFY_TILE=16 \
  tile8x2:dev-merged:SPLASH_VERIFY_TILE=8x2 base3:splash-dev:
echo "PHASE2 DONE"
sed -i '' 's#^OUT=logs/speed/ab-1029#OUT=logs/speed/ab-1030#' splash/staged/speed/gpu_session_1029.sh
splash/staged/speed/gpu_session_1029.sh
echo "ALL DONE"

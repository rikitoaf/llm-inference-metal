#!/usr/bin/env bash
# Whole 2026-10-29 GPU session: e2e A/B, then park/sysstate checks and the attention profile.
cd "$(dirname "$0")/../../.."
splash/staged/speed/merged_ab.sh logs/speed/ab-1029/e2e2 base:splash-dev: off:dev-merged: \
  head:dev-merged:SPLASH_DRAFT_HEAD_ROWS=98304 tile8x2:dev-merged:SPLASH_VERIFY_TILE=8x2 \
  tile16:dev-merged:SPLASH_VERIFY_TILE=16 base2:splash-dev:
splash/staged/speed/gpu_session_1029.sh
echo "ALL DONE"

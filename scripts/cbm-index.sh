#!/usr/bin/env bash
# Index repositories for the codebase-memory-mcp tools (DSH_CBM=1). The scout profile dsh uses cannot index, and
# auto_index is off, so a repo must be indexed here before a session queries it. Re-running is incremental.
#   scripts/cbm-index.sh <repo>...            CBM_INDEX_MODE=fast|moderate|full (default full)
# Moderate (filtered + semantic edges), measured cold on the 5 platform repos: 1.5-8 s each, 17.5 s in all, peak RSS
# 263 MB; fast: 8.3 s, 197 MB; full: 21.6 s, 509 MB. Indexes live in ~/.cache/codebase-memory-mcp (149 MB for the 5).
# Allowed roots: CBM_ALLOWED_ROOT below plus `codebase-memory-mcp allow-root <dir>` grants (this repo's workspace/).
set -euo pipefail
cbm_binary="${CBM_BINARY:-$HOME/.local/bin/codebase-memory-mcp}"
index_mode="${CBM_INDEX_MODE:-full}"   # full since 2026-11-13: moderate dropped 37% of files and functions on a platform repo
(( $# )) || { echo "usage: $0 <repo>..." >&2; exit 2; }
[[ -x "$cbm_binary" ]] || { echo "error: $cbm_binary not found (build: splash/EXTERNAL-2026-11-09.md section 3)" >&2; exit 1; }
# Same limits as config/dsh/cbm.cordis.patch.yml
export CBM_MEM_BUDGET_MB="${CBM_MEM_BUDGET_MB:-1024}"
export CBM_ALLOWED_ROOT="${CBM_ALLOWED_ROOT:-/Users/abdul.rakib/WebstormProjects}"
export CBM_MAX_FILE_BYTES="${CBM_MAX_FILE_BYTES:-1048576}"

failed=0
for repo in "$@"; do
  repo_path="$(cd "$repo" && pwd -P)"
  start_seconds=$(date +%s)
  if result=$("$cbm_binary" cli --quiet index_repository --repo-path "$repo_path" --mode "$index_mode" 2>&1); then
    summary=$(python3 -c 'import json,sys
r=json.loads(sys.stdin.read()); print(r.get("status"), "nodes", r.get("nodes"), "edges", r.get("edges"))' <<<"$result" 2>/dev/null || echo ok)
    echo "$repo_path: $summary ($(( $(date +%s) - start_seconds )) s, $index_mode)"
  else
    echo "$repo_path: FAILED: ${result: -400}" >&2
    failed=1
  fi
done
exit $failed

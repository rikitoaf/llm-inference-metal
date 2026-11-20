#!/usr/bin/env bash
# Start the local XERJ node (if not already up) and (re)index the platform repos for cross-repo search.
# The dsh harness reaches it through the mcp-xerj entry in config/dsh/qwen.cordis.patch.yml.
#   (no arg)  incremental re-index: only changed files re-parse
#   --full    clean rebuild: stop the node, drop the data dir, re-index from scratch
#   --start   only make sure the node is up (no indexing)
#   --stop    stop the node
# After indexing the node is restarted, so it starts the session at ~0.1 GB instead of what indexing left.
# Each repo gets its own index prefix: autoindex refuses a second folder under one prefix.
# Exclusions live in each repo's .xerjignore (gitignore syntax, listed in .git/info/exclude). They keep
# generated bundles and data dumps out: before them, one 538 MB JSON in one repo was 610K of the
# 870K records and the node sat at ~4.8 GB RSS. --max-file-mb does not help there (JSON streams).
set -uo pipefail
XERJ="${XERJ:-$HOME/.local/bin/xerj}"
BASE="${XERJ_BASE:-$HOME/WebstormProjects}"
DATA_DIR="${XERJ_DATA_DIR:-$HOME/xerj-data}"
LOG="${XERJ_LOG:-$HOME/xerj-node.log}"
URL="http://127.0.0.1:9200"
export XERJ_DISABLE_FEEDBACK=true   # no "send a field report" prompts to the agent

# prefix:repo. All under ax-: the MCP tools search `ax-*` by default, so one query spans every repo and
# `prefix: ax-bff` narrows to one. Prefixes must not start one another (`<prefix>-*` would match both).
# The prefix:repo list is local (workspace/ is gitignored, so repository names stay out of the public repo):
# workspace/xerj-repos.txt, one "ax-<prefix>:<folder under XERJ_BASE>" per line.
REPOS=()
repo_list="${XERJ_REPOS_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/workspace/xerj-repos.txt}"
[[ -f "$repo_list" ]] && while read -r entry; do [[ -n "$entry" && "$entry" != \#* ]] && REPOS+=("$entry"); done < "$repo_list"
(( ${#REPOS[@]} )) || { echo "error: no repos listed in $repo_list (lines like ax-api:my-repo)" >&2; exit 2; }

up() { curl -s -m 2 "$URL" >/dev/null 2>&1; }
start_node() {
  up && return 0
  nohup "$XERJ" --insecure --data-dir "$DATA_DIR" > "$LOG" 2>&1 &
  echo "starting xerj node (log: $LOG)..."
  for _ in $(seq 1 30); do up && break; sleep 1; done
  up && echo "node up on $URL" || { echo "ERROR: node not up; see $LOG"; exit 1; }
}
stop_node() {
  pkill -f "$XERJ --insecure" 2>/dev/null || return 0
  for _ in $(seq 1 10); do pgrep -f "$XERJ --insecure" >/dev/null || { echo "stopped xerj node"; return 0; }; sleep 1; done
  pkill -9 -f "$XERJ --insecure"; sleep 1; echo "stopped xerj node (killed: it ignored SIGTERM)"   # seen with a hung node
}

[ -x "$XERJ" ] || { echo "error: xerj not found at $XERJ (curl -fsSL https://xerj.org/get | sh)"; exit 1; }

case "${1:-}" in
  --stop) stop_node; exit 0 ;;
  --start) start_node; exit 0 ;;
  --full)
    echo "full re-index: stopping node + clearing $DATA_DIR ..."
    stop_node
    rm -rf "$DATA_DIR"
    # The resume journals live outside the data dir; left behind, they claim files the wiped node no
    # longer has and autoindex refuses to resume. Drop only the journals of the repos above.
    for entry in "${REPOS[@]}"; do
      grep -l "\"root\":\"$BASE/${entry#*:}\"" "$HOME"/.xerj/autoindex/*/journal.ndjson 2>/dev/null \
        | while read -r j; do rm -rf "$(dirname "$j")"; done
    done ;;
esac

start_node

for entry in "${REPOS[@]}"; do
  prefix="${entry%%:*}" repo="${entry#*:}"
  [ -d "$BASE/$repo" ] || { echo "skip $repo: $BASE/$repo not found"; continue; }
  echo "indexing $repo (prefix $prefix)..."
  "$XERJ" autoindex "$BASE/$repo" --url "$URL" --prefix "$prefix"
  rc=$?   # 3 = completed with some junk files skipped: still indexed and searchable
  # A record over 8 MB (e.g. a 1 MB file of one giant string, embeddings included) aborts the repo:
  # add that file to the repo's .xerjignore, then run --full.
  [ $rc -eq 0 ] || [ $rc -eq 3 ] || echo "WARN: autoindex failed for $repo (exit $rc) — see the error above"
done
stop_node; start_node   # release the memory the index run left behind
echo "done indexing ${#REPOS[@]} repos. Map: $XERJ autoindex map --url $URL"

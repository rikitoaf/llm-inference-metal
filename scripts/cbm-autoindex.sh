#!/usr/bin/env bash
# Keep the codebase-memory-mcp (CBM) graph fresh: re-index a repository once its code has been left alone for a while.
#   cbm-autoindex.sh start [--while-pid PID]   background watcher (one per user); exits with PID when given
#   cbm-autoindex.sh stop | status
#   cbm-autoindex.sh run                       the loop in the foreground (what start runs)
# A repository is re-indexed when its state changed since the last index (HEAD, git status, or the newest
# modification time of a changed file) and nothing in it has been touched for CBM_IDLE_SECONDS (default 240 = 4 min),
# so neither your edits nor the agent's trigger a re-index mid-change. Repos: the list scripts/work uses
# (workspace/platform-bench/repos.txt under PLATFORM_ROOT), or CBM_AUTOINDEX_REPOS="dir1 dir2".
# Checks every CBM_POLL_SECONDS (default 30) with git status only: cheap. Indexing is incremental (1-4 s per repo).
# Log: logs/cbm-autoindex/autoindex.log.
set -uo pipefail
here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
stack_root="$(cd "$here/.." && pwd)"
state_dir="$stack_root/logs/cbm-autoindex"
pid_file="$state_dir/pid"
log_file="$state_dir/autoindex.log"
idle_seconds="${CBM_IDLE_SECONDS:-240}"
poll_seconds="${CBM_POLL_SECONDS:-30}"
mkdir -p "$state_dir"

watched_repos() {
  if [[ -n "${CBM_AUTOINDEX_REPOS:-}" ]]; then printf '%s\n' $CBM_AUTOINDEX_REPOS; return; fi
  local platform_root="${PLATFORM_ROOT:-$HOME/WebstormProjects}" repo
  local repo_list="$stack_root/workspace/platform-bench/repos.txt"
  [[ -f "$repo_list" ]] || return 0
  while read -r repo; do [[ -n "$repo" && -d "$platform_root/$repo/.git" ]] && echo "$platform_root/$repo"; done < "$repo_list"
}

# "<fingerprint> <newest change epoch>" for a repository: HEAD + git status + newest mtime among changed files.
repo_state() {
  local repo="$1" head status newest=0 path mtime
  head=$(git -C "$repo" rev-parse HEAD 2>/dev/null) || return 1
  status=$(git -C "$repo" status --porcelain -uall 2>/dev/null) || return 1
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    path="${path:3}"; path="${path#\"}"; path="${path%\"}"; path="${path##* -> }"
    mtime=$(stat -f %m "$repo/$path" 2>/dev/null) || continue
    (( mtime > newest )) && newest=$mtime
  done <<< "$status"
  echo "$(printf '%s\n%s\n%s' "$head" "$status" "$newest" | md5 -q) $newest"
}

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$log_file"; }

# Per-repository state lives in files (macOS bash 3.2 has no associative arrays): <key>.indexed holds the fingerprint
# last indexed; <key>.seen holds "<fingerprint> <epoch>" for a pending change, first seen at <epoch>.
repo_key() { printf '%s' "$1" | md5 -q; }

run_loop() {
  local while_pid="${1:-}" repo state fingerprint newest now quiet_since key seen_fingerprint seen_at output
  log "start: idle ${idle_seconds}s, poll ${poll_seconds}s, repos: $(watched_repos | xargs -n1 basename | xargs)"
  # The current state counts as indexed at start (scripts/work indexes before launching).
  while read -r repo; do
    key=$(repo_key "$repo"); rm -f "$state_dir/$key.seen"
    state=$(repo_state "$repo") && echo "${state%% *}" > "$state_dir/$key.indexed"
  done < <(watched_repos)
  while :; do
    [[ -n "$while_pid" ]] && ! kill -0 "$while_pid" 2>/dev/null && { log "stop: process $while_pid ended"; break; }
    now=$(date +%s)
    while read -r repo; do
      state=$(repo_state "$repo") || continue
      fingerprint="${state%% *}"; newest="${state##* }"; key=$(repo_key "$repo")
      if [[ "$fingerprint" == "$(cat "$state_dir/$key.indexed" 2>/dev/null)" ]]; then rm -f "$state_dir/$key.seen"; continue; fi
      # A HEAD change (checkout, pull, commit) has no file mtime: count from when this state was first seen.
      seen_fingerprint=""; seen_at=$now
      [[ -f "$state_dir/$key.seen" ]] && read -r seen_fingerprint seen_at < "$state_dir/$key.seen"
      if [[ "$seen_fingerprint" != "$fingerprint" ]]; then echo "$fingerprint $now" > "$state_dir/$key.seen"; seen_at=$now; fi
      quiet_since=$(( newest > seen_at ? newest : seen_at ))
      (( now - quiet_since >= idle_seconds )) || continue
      if output=$("$here/cbm-index.sh" "$repo" 2>&1); then
        echo "$fingerprint" > "$state_dir/$key.indexed"; log "indexed: $output"
      else
        log "index failed: $output"
      fi
      rm -f "$state_dir/$key.seen"
    done < <(watched_repos)
    sleep "$poll_seconds"
  done
}

running_pid() { [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null && cat "$pid_file"; }

case "${1:-}" in
  start)
    shift
    while_pid=""; [[ "${1:-}" == "--while-pid" ]] && while_pid="${2:-}"
    if pid=$(running_pid); then echo "cbm-autoindex already running (pid $pid)"; exit 0; fi
    nohup bash "$0" run ${while_pid:+--while-pid "$while_pid"} >> "$log_file" 2>&1 &
    echo $! > "$pid_file"
    echo "cbm-autoindex started (pid $!, re-index after ${idle_seconds}s without changes; log $log_file)" ;;
  run)
    shift
    while_pid=""; [[ "${1:-}" == "--while-pid" ]] && while_pid="${2:-}"
    run_loop "$while_pid"
    rm -f "$pid_file" ;;
  stop)
    if pid=$(running_pid); then kill "$pid"; rm -f "$pid_file"; echo "cbm-autoindex stopped (pid $pid)"; else echo "cbm-autoindex not running"; fi ;;
  status)
    if pid=$(running_pid); then echo "cbm-autoindex running (pid $pid)"; else echo "cbm-autoindex not running"; fi
    tail -5 "$log_file" 2>/dev/null ;;
  *) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

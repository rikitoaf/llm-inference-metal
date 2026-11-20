#!/usr/bin/env bash
# The stack at a glance: is the model server up (or loading, or stopped), what it serves, the speculation settings, which
# dsh guards are on, and the machine conditions that slow it down (memory, power, Microsoft Defender). Read-only.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/../config/stack.env"
source "$here/lib/system.sh"

row() { printf '  %-14s %s\n' "$1" "$2"; }
printf '\033[1mmodel server (%s, port %s)\033[0m\n' "$BACKEND" "$(model_port)"
if model_healthy; then
  if [[ "$BACKEND" == "splash" ]] && st="$(curl -sf -m 5 "http://$LLAMA_HOST:$SPLASH_PORT/status")"; then
    python3 - "$st" <<'EOF'
import json, sys
s = json.loads(sys.argv[1])
def get(*path, default="?"):
    v = s
    for p in path:
        if not isinstance(v, dict) or p not in v: return default
        v = v[p]
    return v
row = lambda k, v: print(f"  {k:<14} {v}")
row("state", "ready" if s.get("ready") else "up, not ready (loading or weights released)")
row("model", get("instance", "model"))
row("context", f"{get('maximum_context_tokens')} tokens, KV {get('identity', 'kv', 'format')}")
cap = get("disk", "capacity_bytes", default=0)
row("SSD cache", f"{cap / 2**30:.0f} GiB tier" if isinstance(cap, (int, float)) and cap > 0 else "off")
EOF
  else
    row state "up"
  fi
elif model_process_running; then
  row state "starting (process running, /health not answering yet)"
else
  row state "stopped - GPU free (scripts/qwen start, or just scripts/qwen)"
fi
if [[ "$BACKEND" == "splash" ]]; then
  pid=$(pgrep -f "splash serve-native" | head -1)
  [[ -n "$pid" ]] && row "engine RSS" "$(ps -o rss= -p "$pid" | awk '{printf "%.1f GiB", $1/1048576}') (pid $pid)"
  row engine "$(basename "$SPLASH_SRC"), drafter ${SPLASH_DRAFTER:-dflash2}, park lane ${SPLASH_PARK_LANE:-0}, verify rows ≥${SPLASH_VERIFY_ROWS_LONG_FROM:-off} ${SPLASH_VERIFY_ROWS_LONG:-16}, prefill attn v2 ${SPLASH_PREFILL_ATTN_V2:-0}, max context ${SPLASH_MAX_CONTEXT:-auto}"
  row speculation "tree ${SPLASH_DRAFT_TREE:-0}, draft head rows ${SPLASH_DRAFT_HEAD_ROWS:-all}, copy ${SPLASH_DRAFT_COPY:-0}, no-think keeps prefix ${SPLASH_NO_THINK_KEEPS_PREFIX:-0}, temperature ${SPLASH_DEFAULT_TEMPERATURE:-model}"
  idle_stop="${SPLASH_IDLE_STOP:-off}"
  row "idle" "weights released after ${SPLASH_IDLE_RELEASE:-10m}, server stopped after $idle_stop$(pgrep -f 'scripts/idle-stop.sh' >/dev/null && echo ' (watchdog running)')"
  last="$STACK_ROOT/logs/splash/steps.jsonl"
  [[ -f "$last" ]] && row "last activity" "$(( ($(date +%s) - $(stat -f %m "$last")) / 60 )) min ago"
fi

printf '\033[1mharness\033[0m\n'
grep -q "dsh-plugin-runaway" "$STACK_ROOT/config/dsh/qwen.cordis.patch.yml" && row "runaway guard" "on (thinking off for one step after a max_tokens cut)" || row "runaway guard" "OFF"
row "JEV gates" "$([[ "$JEV" == "1" ]] && echo on || echo off)"
gen="$DSH_HOME/splash.generated.cordis.patch.yml"
[[ "$BACKEND" == "splash" && -f "$gen" ]] && row "dsh window" "$(sed -n 's/.*contextWindow: \([0-9][0-9]*\).*/\1/p' "$gen") tokens"
running=$(pgrep -f "dsh --profile" | wc -l | tr -d ' ')
row "dsh sessions" "$running running"

printf '\033[1mmachine\033[0m\n'
avail=$(reclaimable_gib)
row memory "~${avail:-?} GiB reclaimable$([[ "$avail" =~ ^[0-9]+$ ]] && (( avail < 16 )) && ! model_process_running && echo ' - below the ~16 GiB Splash needs to start')"
on_ac_power && row power "AC" || row power "battery - Low Power mode throttles the GPU"
d=$(defender_cpu)
(( d > 30 )) && row Defender "${d}% CPU - scanning; expect ~10% slower generation" || row Defender "${d}% CPU"

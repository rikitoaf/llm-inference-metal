#!/usr/bin/env bash
# Start Splash: the Qwen3.8-27B-family GGUF of MODEL_PROFILE (config/stack.env: Unsloth UD-IQ3_XXS by default,
# or the Swift 1.5 Uncensored one) + the family's DFlash2 block drafter, with prefix and GDN-state caching.
# Use with BACKEND=splash for dsh. Stop llama-server first: both together do not fit in 24 GB.
#
# By default this runs the Splash 1.2.1 built from splash/splash-src with TurboQuant KV
# (SPLASH_KV_FORMAT in config/stack.env; turbo-int4 holds ~2x the context of INT8, README section 32)
# and a persistent SSD prefix cache. Without that checkout it falls back to Homebrew's splash (INT8).
# Splash 1.2.x loads the weights into memory at every start: it needs ~16 GiB of what it counts as
# reclaimable (splash/tools/hostavail.py), so close memory-heavy apps first.
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/config/stack.env"
source "$STACK_ROOT/scripts/lib/system.sh"

if curl -sf -m 2 "http://$LLAMA_HOST:$LLAMA_PORT/health" >/dev/null; then
  echo "error: llama-server is still running on :$LLAMA_PORT; stop it first (24 GB holds one 27B server)" >&2; exit 1
fi

if [[ -x "$SPLASH_SRC/splash" ]]; then
  SPLASH_CMD=("$SPLASH_SRC/splash" serve)           # the launcher rebuilds (make all) when sources changed
else
  command -v splash >/dev/null || { echo "error: no Splash: build splash/splash-src (README 32) or brew install incoai/tap/splash" >&2; exit 1; }
  SPLASH_CMD=(splash serve)
  if [[ "$SPLASH_KV_FORMAT" == turbo* ]]; then
    echo "warning: Homebrew Splash has no TurboQuant KV; using int8" >&2
    SPLASH_KV_FORMAT=int8
  fi
fi

ARGS=(
  --model "$SPLASH_MODEL"
  --served-model-name "$MODEL_ALIAS"   # the dsh patch asks for this id on both backends
  --language-only                      # no vision tower: its memory goes to context
  --host "$LLAMA_HOST" --port "$SPLASH_PORT"
  --no-webui
  --kv-format "$SPLASH_KV_FORMAT"
  # SSD tier for cached KV pages and GDN states. Splash grows its RAM cache only while macOS keeps
  # 3 GiB free; with desktop apps open on 24 GB that never happens, every state was evicted and each
  # agent step re-prefilled its whole prompt (124 s demo vs 68 s on llama.cpp). 24G keeps prefixes of
  # several repos and sessions warm; it does not raise the context limit. 0 disables it.
  --max-cache-disk "${SPLASH_CACHE_DISK:-24G}"
)
# Keep that SSD cache across restarts (Splash >= 1.2; its namespace includes the KV format), within a total budget:
# old namespaces (other models, formats, quantizer revisions) are pruned, most recently used kept.
if [[ "$SPLASH_PERSISTENT_CACHE" == "1" && "${SPLASH_CACHE_DISK:-24G}" != "0" && "${SPLASH_CMD[0]}" != "splash" ]]; then
  ARGS+=(--persistent-cache)
  python3 "$STACK_ROOT/splash/tools/prune_splash_cache.py" --budget "$SPLASH_CACHE_TOTAL_DISK" || true
fi
# A pinned commit (config/stack.env); without one Splash follows the repository's default branch.
[[ -n "${SPLASH_REVISION:-}" ]] && ARGS+=(--revision "$SPLASH_REVISION")
# Speculation knobs (config/stack.env); the draft scale is read by the engine from its environment (source build).
[[ -n "${SPLASH_DEFAULT_TEMPERATURE:-}" ]] && ARGS+=(--default-temperature "$SPLASH_DEFAULT_TEMPERATURE")
[[ -n "${SPLASH_DRAFT_TEMPERATURE_SCALE:-}" ]] && export SPLASH_DRAFT_TEMPERATURE_SCALE
[[ -n "${SPLASH_DRAFT_TREE:-}" ]] && export SPLASH_DRAFT_TREE
[[ -n "${SPLASH_DRAFT_HEAD_ROWS:-}" ]] && export SPLASH_DRAFT_HEAD_ROWS
[[ -n "${SPLASH_DRAFT_COPY:-}" ]] && export SPLASH_DRAFT_COPY
[[ -n "${SPLASH_NO_THINK_KEEPS_PREFIX:-}" ]] && export SPLASH_NO_THINK_KEEPS_PREFIX
[[ -n "${SPLASH_IDLE_RELEASE:-}" ]] && ARGS+=(--idle-release "$SPLASH_IDLE_RELEASE")
[[ -n "${SPLASH_MAX_CONTEXT:-}" ]] && ARGS+=(--max-context "$SPLASH_MAX_CONTEXT")
[[ -n "${SPLASH_DRAFTER:-}" ]] && export SPLASH_DRAFTER
[[ -n "${SPLASH_PARK_LANE:-}" ]] && export SPLASH_PARK_LANE
[[ -n "${SPLASH_VERIFY_ROWS_LONG:-}" ]] && export SPLASH_VERIFY_ROWS_LONG SPLASH_VERIFY_ROWS_LONG_FROM
if [[ -n "${SPLASH_EXTRA_ARGS:-}" ]]; then   # e.g. SPLASH_EXTRA_ARGS="--max-context 128K --idle-release off"
  read -r -a EXTRA <<< "$SPLASH_EXTRA_ARGS"
  ARGS+=("${EXTRA[@]}")
fi

# Splash refuses to start ("host memory reserve protected") below ~16 GiB reclaimable; say so up front.
if avail=$(python3 "$STACK_ROOT/splash/tools/hostavail.py" --gib 2>/dev/null) && [[ "$avail" =~ ^[0-9]+$ ]] && (( avail < 16 )); then
  echo "warning: only ~$avail GiB reclaimable; Splash needs ~16 GiB to start. Close memory-heavy apps (IDEs, Teams) if it waits or fails." >&2
fi

# Usage logs (config/stack.env SPLASH_LOG_DIR): the console goes to serve.log as well, and the engine appends a line
# per step to steps.jsonl. Each file rolls over to .1 at a start once it passes 200 MB.
if [[ -n "${SPLASH_LOG_DIR:-}" ]]; then
  mkdir -p "$SPLASH_LOG_DIR"
  for log in "$SPLASH_LOG_DIR/serve.log" "$SPLASH_LOG_DIR/steps.jsonl"; do
    if [[ -f "$log" ]] && (( $(stat -f %z "$log") > 200 * 1024 * 1024 )); then mv "$log" "$log.1"; fi
  done
  export SPLASH_STEP_LOG="$SPLASH_LOG_DIR/steps.jsonl"
  exec > >(tee -a "$SPLASH_LOG_DIR/serve.log") 2>&1
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') start: engine $(basename "$SPLASH_SRC") drafter ${SPLASH_DRAFTER:-dflash2} park ${SPLASH_PARK_LANE:-0} verify-rows-long ${SPLASH_VERIFY_ROWS_LONG:-16}@${SPLASH_VERIFY_ROWS_LONG_FROM:-} max-context ${SPLASH_MAX_CONTEXT:-auto}, kv $SPLASH_KV_FORMAT, tree ${SPLASH_DRAFT_TREE:-0}, temperature ${SPLASH_DEFAULT_TEMPERATURE:-model}, draft scale ${SPLASH_DRAFT_TEMPERATURE_SCALE:-1}, draft head rows ${SPLASH_DRAFT_HEAD_ROWS:-all}, copy ${SPLASH_DRAFT_COPY:-0}, no-think keeps prefix ${SPLASH_NO_THINK_KEEPS_PREFIX:-0}, idle release ${SPLASH_IDLE_RELEASE:-10m} / stop ${SPLASH_IDLE_STOP:-off}, extra '${SPLASH_EXTRA_ARGS:-}'"
fi

# SPLASH_IDLE_STOP: a watchdog stops the whole server after that long without a request (scripts/idle-stop.sh).
if [[ -n "$(duration_seconds "${SPLASH_IDLE_STOP:-}")" ]] && ! pgrep -f "scripts/idle-stop.sh" >/dev/null; then
  nohup "$STACK_ROOT/scripts/idle-stop.sh" >> "$STACK_ROOT/logs/idle-stop.log" 2>&1 &
  disown
fi

echo "+ ${SPLASH_CMD[*]} ${ARGS[*]}" >&2
exec "${SPLASH_CMD[@]}" "${ARGS[@]}"

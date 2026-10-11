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
# Keep that SSD cache across restarts (Splash >= 1.2; its namespace includes the KV format).
if [[ "$SPLASH_PERSISTENT_CACHE" == "1" && "${SPLASH_CACHE_DISK:-24G}" != "0" && "${SPLASH_CMD[0]}" != "splash" ]]; then
  ARGS+=(--persistent-cache)
fi
# A pinned commit (config/stack.env); without one Splash follows the repository's default branch.
[[ -n "${SPLASH_REVISION:-}" ]] && ARGS+=(--revision "$SPLASH_REVISION")
if [[ -n "${SPLASH_EXTRA_ARGS:-}" ]]; then   # e.g. SPLASH_EXTRA_ARGS="--max-context 128K --idle-release off"
  read -r -a EXTRA <<< "$SPLASH_EXTRA_ARGS"
  ARGS+=("${EXTRA[@]}")
fi

# Splash refuses to start ("host memory reserve protected") below ~16 GiB reclaimable; say so up front.
if avail=$(python3 "$STACK_ROOT/splash/tools/hostavail.py" --gib 2>/dev/null) && [[ "$avail" =~ ^[0-9]+$ ]] && (( avail < 16 )); then
  echo "warning: only ~$avail GiB reclaimable; Splash needs ~16 GiB to start. Close memory-heavy apps (IDEs, Teams) if it waits or fails." >&2
fi

echo "+ ${SPLASH_CMD[*]} ${ARGS[*]}" >&2
exec "${SPLASH_CMD[@]}" "${ARGS[@]}"

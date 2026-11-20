#!/usr/bin/env bash
# Start lithos-metal from the source build in lithos/lithos-src (README section 34): MODEL_PROFILE's GGUF (the file
# Splash serves; its llama.cpp blocks are read as they are) + the local DSpark drafter (Q8_0 GGUF), with the turbo-int4
# K/V cache. Use with BACKEND=lithos for dsh. Stop Splash or llama-server first: 24 GB holds one 27B server.
#
# Memory at the default 170K context: ~12 GiB of weights (target + drafter), 3.6 GiB of turbo-int4 caches, ~0.4 GiB
# of workspaces: ~16.4 GiB, so close memory-heavy apps first. The first start writes the packed weights (~12 GiB)
# under ~/.cache/lithos-metal/packs (a new context size packs again: its RoPE tables); nothing is downloaded.
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/config/stack.env"

LITHOS="$LITHOS_SRC/.venv/bin/lithos-metal"
[[ -x "$LITHOS" ]] || { echo "error: no lithos-metal at $LITHOS: set up lithos/lithos-src (README 34)" >&2; exit 1; }
for port in "$LLAMA_PORT" "$SPLASH_PORT"; do
  if curl -sf -m 2 "http://$LLAMA_HOST:$port/health" >/dev/null; then
    echo "error: a model server is still running on :$port; stop it first (scripts/stop.sh)" >&2; exit 1
  fi
done

[[ -n "$LITHOS_MODEL" ]] || { echo "error: no LITHOS_MODEL (the profile's GGUF is not in the Hub cache)" >&2; exit 1; }
ARGS=(
  --model "$LITHOS_MODEL"
  --served-model-name "$MODEL_ALIAS"   # the dsh patch asks for this id on every backend
  --host "$LLAMA_HOST" --port "$LITHOS_PORT"
  --kv-format "$LITHOS_KV_FORMAT"
  --max-context "$LITHOS_MAX_CONTEXT"
)
case "$LITHOS_DRAFT" in
  auto) ;;                                            # the catalog's DSpark head for a Hub target
  none) ARGS+=(--no-draft) ;;
  *) ARGS+=(--draft "$LITHOS_DRAFT") ;;
esac
# Prefix checkpoints on the SSD: with little free RAM the in-memory ones are evicted between agent steps.
if [[ "$LITHOS_PREFIX_DISK" != "0" ]]; then
  ARGS+=(--prefix-cache-dir "${LITHOS_PREFIX_DIR:-$HOME/Library/Caches/lithos-metal/prefix}" --prefix-cache-disk-size "$LITHOS_PREFIX_DISK"
         --prefix-cache-memory "$LITHOS_PREFIX_MEMORY")
fi
if [[ -n "${LITHOS_EXTRA_ARGS:-}" ]]; then   # e.g. LITHOS_EXTRA_ARGS="--no-draft" or "--model-ttl 900"
  read -r -a EXTRA <<< "$LITHOS_EXTRA_ARGS"
  ARGS+=("${EXTRA[@]}")
fi

# The weights are mapped into the GPU working set; warn when the Mac cannot spare them.
if avail=$(python3 "$STACK_ROOT/splash/tools/hostavail.py" --gib 2>/dev/null) && [[ "$avail" =~ ^[0-9]+$ ]] && (( avail < 17 )); then
  echo "warning: only ~$avail GiB reclaimable; lithos-metal needs ~17 GiB at $LITHOS_MAX_CONTEXT tokens (~13 GiB at 32768). Close memory-heavy apps (IDEs, Teams), or lower LITHOS_MAX_CONTEXT / set LITHOS_DRAFT=none." >&2
fi

echo "+ $LITHOS ${ARGS[*]}" >&2
exec "$LITHOS" serve "${ARGS[@]}"

#!/usr/bin/env bash
# Start the model server: Qwen3.8-27B GSQ-RCO IQ2_S + TurboQuant KV + n-gram speculation + /v1/decision (JEV).
# Every default was measured on the M5 Pro 24 GB - see README.md section "Tuning reference".
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/config/stack.env"

[[ -x "$LLAMA_SERVER" ]] || { echo "error: $LLAMA_SERVER missing - run the build in README step 2" >&2; exit 1; }
[[ -f "$MODEL" ]]        || { echo "error: model not found: $MODEL" >&2; exit 1; }

ARGS=(
  -m "$MODEL" --alias "$MODEL_ALIAS"
  -ngl 99 -fa on
  -ctk "$CTK" -ctv "$CTV"
  -c "$CTX" -np 1
  --jinja
  --ctx-checkpoints 8           # each hybrid checkpoint holds ~150 MiB of recurrent state (host RAM)
  --cache-ram 2048              # prompt cache shares the 24 GB unified pool with the GPU buffers
  --metrics
  --host "$LLAMA_HOST" --port "$LLAMA_PORT"
)

# ngram-mod reads its own knobs (--spec-ngram-mod-n-match/-n-min/-n-max, defaults 24/48/64);
# --spec-draft-n-max/-n-min only set the MTP draft depth. When both are enabled, ngram-mod is tried
# first and MTP drafts whenever there is no n-gram match.
MTP_ARGS=(--spec-draft-n-max "$MTP_DRAFT" --spec-draft-n-min 1 -ctkd "$CTK" -ctvd "$CTV")   # no -md: it loads a second 9 GB copy
case "$SPEC" in
  mtp+ngram) ARGS+=(--spec-type ngram-mod,draft-mtp "${MTP_ARGS[@]}") ;;
  ngram)     ARGS+=(--spec-type ngram-mod) ;;
  mtp)       ARGS+=(--spec-type draft-mtp "${MTP_ARGS[@]}") ;;
  none)      ;;
  *) echo "error: SPEC must be mtp+ngram, ngram, mtp or none" >&2; exit 1 ;;
esac

# MTP keeps MTP_DRAFT extra recurrent-state snapshots per sequence (~150 MiB each). With JEV's 8 decision
# sequences, 128K measured 16,014 MiB of the 18,186 MiB Metal budget, so 256K does not fit. Without them
# (DECISION_SEQS=0, the default) 256K measured ~14,400 MiB on 2026-10-02.
if [[ "$SPEC" == *mtp* && "$DECISION_SEQS" != "0" && "$CTX" -gt 181072 ]]; then
  echo "error: SPEC=$SPEC with CTX=$CTX and $DECISION_SEQS decision seqs exceeds the 24 GB memory budget; use CTX<=181072, DECISION_SEQS=0 or SPEC=ngram" >&2; exit 1
fi

if [[ "$DECISION_SEQS" != "0" ]]; then
  if [[ "$SPEC" == *mtp* && "$DECISION_SEQS" -gt 8 ]]; then
    echo "error: with SPEC=mtp each decision seq costs 3x recurrent state; >8 runs out of Metal memory" >&2; exit 1
  fi
  ARGS+=(--decision-seqs "$DECISION_SEQS")
fi

if [[ -n "${LLAMA_EXTRA_ARGS:-}" ]]; then   # e.g. LLAMA_EXTRA_ARGS="-lv 4" to log buffer sizes
  read -r -a EXTRA <<< "$LLAMA_EXTRA_ARGS"
  ARGS+=("${EXTRA[@]}")
fi

echo "+ $LLAMA_SERVER ${ARGS[*]}" >&2
exec "$LLAMA_SERVER" "${ARGS[@]}"

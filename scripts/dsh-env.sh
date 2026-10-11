#!/usr/bin/env bash
# Environment for every dsh invocation in this workspace. Source it: `source scripts/dsh-env.sh`.
# DSH_* variables are bootstrap-only in dsh: they must be exported, a .env file cannot set them.
_dsh_env_here="${BASH_SOURCE[0]:-$0}"      # bash sets BASH_SOURCE; zsh sets $0 to the sourced file
source "$(cd "$(dirname "$_dsh_env_here")/.." && pwd)/config/stack.env"

export PATH="$NODE_BIN:$PATH"                 # dsh needs Node >= 22.19
export DSH_HOME                               # profiles, sessions, settings, credentials stay in this workspace
export DSH_AGENTS_HOME="$DSH_HOME/agents"     # keep dsh away from ~/.agents/skills
export DSH_TELEMETRY_DISABLED=1
export LLAMA_API_KEY="${LLAMA_API_KEY:-sk-local}"   # placeholder credential for the keyless local server
DSH_PATCH="$STACK_ROOT/config/dsh/qwen.cordis.patch.yml"
# The shipped patches name the default models. Another MODEL_PROFILE gets generated copies whose labels
# name the model actually served (the model id stays qwen3.8-27b-local, so sessions carry over).
if [[ "$MODEL_PROFILE" != "qwen" ]]; then
  mkdir -p "$DSH_HOME"
  DSH_PATCH="$DSH_HOME/qwen.generated.cordis.patch.yml"
  sed "s#Qwen3.8 27B GSQ-RCO IQ2_S (local)#$MODEL_LABEL (local)#" "$STACK_ROOT/config/dsh/qwen.cordis.patch.yml" > "$DSH_PATCH"
fi
# BACKEND=splash layers the Splash overrides (provider URL, JEV over /v1/systemone) on top
DSH_PATCHES=(--patch "$DSH_PATCH")
# Splash's context limit depends on its KV format (int8 ~94K, turbo-int4 ~192K, turbo ~213K on
# 24 GB) and on the memory free when it started, so the provider's contextWindow is taken from
# the running server: its limit minus the 16384-token reply budget and a 4K margin, written into a
# generated copy of the Splash patch. Without a running server the static file's value (65536,
# below every format's limit) applies.
splash_patch() {
  local static="$STACK_ROOT/config/dsh/splash.cordis.patch.yml"
  local limit window generated="$DSH_HOME/splash.generated.cordis.patch.yml"
  limit=$(curl -sf -m 3 "http://$LLAMA_HOST:$SPLASH_PORT/status" 2>/dev/null |
          python3 -c 'import json,sys; print(json.load(sys.stdin)["maximum_context_tokens"])' 2>/dev/null)
  if [[ "$limit" =~ ^[0-9]+$ ]]; then
    window=$(( limit - 16384 - 4096 ))
    if (( window < 4096 )); then
      echo "error: Splash's context limit $limit leaves no room for the 16384-token reply budget" >&2
      window=4096
    fi
  else
    window=$(sed -n 's/.*contextWindow: \([0-9][0-9]*\).*/\1/p' "$static")   # no server: the static value
  fi
  # Always a generated copy, so the provider URL follows LLAMA_HOST and SPLASH_PORT too.
  local label=()                                  # the shipped labels name the default model
  [[ "$MODEL_PROFILE" != "qwen" ]] && label=(-e "s#Qwen3.8[ -]27B UD-IQ3_XXS#$MODEL_LABEL#")
  mkdir -p "$DSH_HOME"
  sed -e "s/contextWindow: [0-9][0-9]*/contextWindow: $window/" \
      -e "s#baseURL: http://127.0.0.1:8000/v1#baseURL: http://$LLAMA_HOST:$SPLASH_PORT/v1#" \
      ${label[@]+"${label[@]}"} "$static" > "$generated"
  echo "$generated"
}
[[ "$BACKEND" == "splash" ]] && DSH_PATCHES+=(--patch "$(splash_patch)")
[[ "$JEV" != "1" ]] && DSH_PATCHES+=(--patch "$STACK_ROOT/config/dsh/nojev.cordis.patch.yml")   # JEV=1 keeps the gates
[[ -n "${DSH_EXTRA_PATCH:-}" ]] && DSH_PATCHES+=(--patch "$DSH_EXTRA_PATCH")   # e.g. config/dsh/nojev.cordis.patch.yml

# the model server for $BACKEND answers /health
model_server_up() { curl -sf -m 3 "http://$LLAMA_HOST:$MODEL_PORT/health" >/dev/null; }

# boot a profile, creating it from a shipped template on first use
dsh_profile_args() {
  local profile="$1" template="$2"
  if [[ -d "$DSH_HOME/profiles/$profile" ]]; then
    echo --profile "$profile" "${DSH_PATCHES[@]}"
  else
    echo --profile "$profile" --from-default-profile "$template" "${DSH_PATCHES[@]}"
  fi
}

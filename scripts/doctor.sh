#!/usr/bin/env bash
# Check every piece of the stack. Exit code = number of failed checks.
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/dsh-env.sh"

fails=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fails=$((fails + 1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }
head1(){ printf '\n\033[1m%s\033[0m\n' "$1"; }

head1 "model server (BACKEND=$BACKEND)"
if [[ "$BACKEND" == "llama" ]]; then
  [[ -x "$LLAMA_SERVER" ]] && ok "llama-server binary: $LLAMA_SERVER" || bad "llama-server missing: $LLAMA_SERVER (README step 2)"
  "$LLAMA_SERVER" --help 2>/dev/null | grep -q -- --decision-seqs && ok "binary has the /v1/decision patch" || bad "binary lacks --decision-seqs: rebuild with patches/0001-*.patch"
  if grep -q -a kernel_mul_mv_ext_iq2_xxs_f32_r1_3 "$LLAMA_SERVER" 2>/dev/null; then   # no pipe: pipefail + grep -q would SIGPIPE strings
    ok "binary has the i-quant small-batch kernels (fast MTP verification)"
  elif [[ "$SPEC" == *mtp* ]]; then
    bad "SPEC=$SPEC but the binary lacks patches/0002 (MTP will be ~no faster): rebuild with the patch"
  else
    warn "binary lacks patches/0002 (only matters for SPEC=mtp / mtp+ngram)"
  fi
  [[ -f "$MODEL" ]] && ok "model file: $(basename "$MODEL") ($(du -h "$MODEL" | cut -f1))" || bad "model missing: $MODEL"
  if curl -sf -m 3 "http://$LLAMA_HOST:$LLAMA_PORT/health" >/dev/null; then
    ok "server healthy on $LLAMA_HOST:$LLAMA_PORT"
    ids="$(curl -s -m 5 "http://$LLAMA_HOST:$LLAMA_PORT/v1/models" | python3 -c "import json,sys; print(' '.join(m['id'] for m in json.load(sys.stdin)['data']))" 2>/dev/null)"
    [[ " $ids " == *" $MODEL_ALIAS "* ]] && ok "model id '$MODEL_ALIAS' served" || bad "served ids '$ids' do not include '$MODEL_ALIAS' (dsh patch model id must match)"
    if [[ "$JEV" == "1" ]]; then
      out="$(curl -s -m 120 "$JEV_URL" -H 'Content-Type: application/json' -d @"$STACK_ROOT/examples/gate-1-context.json")"
      if ms="$(python3 -c "import json,sys; r=json.loads(sys.argv[1]); assert r['results'][0]['decision']['outcome']=='test_failure'; print(round(r['timings']['total_ms']))" "$out" 2>/dev/null)"; then
        ok "/v1/decision works (${ms} ms, outcome=test_failure)"
      else
        bad "/v1/decision failed: $(echo "$out" | head -c 200) (start the server with --decision-seqs)"
      fi
    fi
  else
    bad "server not reachable on $LLAMA_HOST:$LLAMA_PORT - run scripts/serve-llama.sh"
  fi

else
  if [[ -x "$SPLASH_SRC/splash" ]]; then
    ok "Splash source build: $SPLASH_SRC ($(git -C "$SPLASH_SRC" describe --tags 2>/dev/null) + TurboQuant patch)"
    grep -q turbo_rotate_queries_256 "$SPLASH_SRC/runtime/metal/kernels/shared/turbo_rotate.metal" 2>/dev/null \
      && ok "TurboQuant kernels present (turbo, turbo-int4)" || bad "splash-src lacks the TurboQuant patch: git apply patches/0003-*.patch"
  else
    command -v splash >/dev/null && ok "$(splash --version 2>/dev/null) installed (Homebrew; no TurboQuant)" \
      || bad "no Splash: build splash/splash-src (README 32) or brew install incoai/tap/splash"
  fi
  [[ "$MODEL_PROFILE" != "qwen" ]] && ok "model profile $MODEL_PROFILE: $SPLASH_MODEL${SPLASH_REVISION:+ @ ${SPLASH_REVISION:0:10}}"
  if st="$(curl -sf -m 5 "http://$LLAMA_HOST:$SPLASH_PORT/status")"; then
    loaded="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['instance']['model'])" "$st" 2>/dev/null)"
    [[ -n "$loaded" && "$loaded" != "$SPLASH_MODEL" ]] && warn "the server runs $loaded but MODEL_PROFILE=$MODEL_PROFILE selects $SPLASH_MODEL: restart it (scripts/stop.sh; scripts/serve-splash.sh)"
    fmt="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['identity']['kv']['format'])" "$st" 2>/dev/null)"
    ctx="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['maximum_context_tokens'])" "$st" 2>/dev/null)"
    case "$fmt" in
      turbo-int4) ok "KV format $fmt: context $ctx tokens (dsh window follows it, scripts/dsh-env.sh)" ;;
      turbo) warn "KV format turbo: context $ctx tokens, but 2.2-2.4x slower attention than turbo-int4 for ~10% more context" ;;
      *) warn "KV format ${fmt:-unknown}: context ${ctx:-?} tokens; SPLASH_KV_FORMAT=turbo-int4 doubles it (README 32)" ;;
    esac
    want="$SPLASH_KV_FORMAT"                   # what serve-splash.sh would start (Homebrew: turbo* -> int8)
    [[ ! -x "$SPLASH_SRC/splash" && "$want" == turbo* ]] && want=int8
    [[ -n "$fmt" && "$fmt" != "$want" ]] && warn "the server runs $fmt but config/stack.env selects $want: restart it (scripts/stop.sh; scripts/serve-splash.sh)"
    python3 -c "import json,sys; s=json.loads(sys.argv[1]); sys.exit(0 if s['ready'] else 1)" "$st" && ok "splash ready on $LLAMA_HOST:$SPLASH_PORT ($(python3 -c "import json,sys; s=json.loads(sys.argv[1]); print(s['instance']['model'], '· context', s['maximum_context_tokens'])" "$st"))" || bad "splash not ready yet"
    python3 -c "import json,sys; sys.exit(0 if json.loads(sys.argv[1])['disk']['capacity_bytes'] > 0 else 1)" "$st" \
      && ok "SSD cache tier on (prefix reuse survives low free RAM)" || warn "no SSD cache tier: with little free RAM every agent step re-prefills (SPLASH_CACHE_DISK)"
    ids="$(curl -s -m 5 "http://$LLAMA_HOST:$SPLASH_PORT/v1/models" | python3 -c "import json,sys; print(' '.join(m['id'] for m in json.load(sys.stdin)['data']))" 2>/dev/null)"
    [[ " $ids " == *" $MODEL_ALIAS "* ]] && ok "model id '$MODEL_ALIAS' served" || bad "served ids '$ids' do not include '$MODEL_ALIAS' (serve-splash.sh sets --served-model-name)"
    [[ "$JEV" == "1" ]] && out="$(curl -s -m 120 "http://$LLAMA_HOST:$SPLASH_PORT/v1/systemone" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL_ALIAS\",\"state\":\"$ npm test\\nFAIL src/auth.test.ts\\nTypeError: Cannot read properties of undefined\",\"questions\":{\"failed\":{\"type\":\"noul\",\"instructions\":\"Did the command fail?\"}}}")"
    [[ "$JEV" == "1" ]] && { python3 -c "import json,sys; sys.exit(0 if json.loads(sys.argv[1])['answers']['failed']['noul'] > 0.5 else 1)" "$out" 2>/dev/null \
      && ok "/v1/systemone works (JEV gates)" || bad "/v1/systemone failed: $(echo "$out" | head -c 200)"; }
  else
    bad "splash not reachable on $LLAMA_HOST:$SPLASH_PORT - run scripts/serve-splash.sh"
  fi
fi

head1 "harness"
node_v="$("$NODE_BIN/node" --version 2>/dev/null)"
python3 -c "import sys; v=tuple(map(int,sys.argv[1].lstrip('v').split('.'))); sys.exit(0 if v>=(22,19,0) else 1)" "$node_v" 2>/dev/null \
  && ok "node $node_v (>= 22.19)" || bad "node $node_v at $NODE_BIN is too old or missing (need >= 22.19)"
[[ -x "$DSH_BIN" ]] && ok "dsh $("$DSH_BIN" --version 2>/dev/null)" || bad "dsh missing: run (cd harness && npm install)"
[[ -d "$DSH_HOME" ]] && ok "DSH_HOME: $DSH_HOME" || warn "DSH_HOME not created yet (first run creates it)"
[[ -e "$STACK_ROOT/plugins/node_modules/@deepseek-ai/schemastery" ]] && ok "plugins/node_modules -> harness deps" || bad "plugins/node_modules symlink broken: ln -sfn ../harness/node_modules plugins/node_modules"
[[ "$JEV" == "1" ]] && { (cd "$STACK_ROOT/plugins/dsh-plugin-jev" && "$NODE_BIN/node" -e "import('./index.mjs').then(m=>{if(!m.apply||!m.Config)process.exit(1)})" 2>/dev/null) \
  && ok "dsh-plugin-jev loads" || bad "dsh-plugin-jev failed to import"; }
dump="$(cd "$STACK_ROOT" && "$DSH_BIN" $(dsh_profile_args qwen headless) --dump-config 2>&1)"
[[ "$JEV" == "1" ]] && { grep -q "dsh-plugin-jev/index.mjs" <<<"$dump" && ok "patch composes (jev row present)" || bad "patch did not compose: $(head -c 300 <<<"$dump")"; }
[[ "$JEV" == "1" ]] || ok "JEV gates off (JEV=1 turns them on)"
grep -q "provider: llama-local" <<<"$dump" && ok "default model -> llama-local/$MODEL_ALIAS" || bad "agent-default-model not set to llama-local"

head1 "result"
[[ $fails -eq 0 ]] && echo "  all checks passed" || echo "  $fails check(s) failed"
exit $fails

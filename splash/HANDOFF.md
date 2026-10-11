# TurboQuant-in-Splash: handoff state (2026-10-05, ~21:15)

This is the working context at the time of compaction: everything needed to continue without the
conversation. Read it first, then `splash/TURBOQUANT.md` (the full documentation). The design notes
and work log are in `splash/TURBOQUANT_PLAN.md`.

## 0a. Swift 1.5 Qwen3.8-27B Uncensored (2026-10-08)

The user asked for `ajgazin/Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-GGUF` `UD-IQ3_XXS`, integrated into
the dsh harness and the Splash engine. They first asked for a variant "without MTP, so memory takes
less"; that does not exist, and on Splash it would not matter (they chose "download as is").

**Done, no GPU:**
- **Download:** 10.18 GiB into the Hub cache at commit `540448475b2a84843fa2a54fcfd9c888de8a99ca`; sha256
  `eb332507...d57485` matches the Hub's. `logs/download-swift.log`.
- **Splash install:** `install/models.py ... prepare --revision <sha> --language-only` ran: "as
  Qwen3.8-27B (gguf); draft incoai/Qwen3.8-27B-DFlash2". It sits next to the Unsloth install.
- **Offline checks** (Splash's own `require_loadable`, `family_for`, `ChatTemplates` run on the
  headers):
  - same family, 64 loaded layers, tensor-type histogram, vocab and merges identical to Unsloth's file;
  - the chat templates render identically at every effort and with a later system message;
  - cache namespace differs per model, because the model hash covers the assembly's `model.json`.
  - The scripts are in the session scratchpad (`compare_gguf.py`, `template_probe.py`); `probe_gguf.py`
    lists a repo's GGUF headers.
- **Every GGUF of this line has the MTP layer** (checked ajgazin, ukisai, bartowski, mradermacher,
  jegreene, and Unsloth's own Qwen3.8-27B file). Splash skips it; it is ~0.2 GiB of the file.
- **Harness wiring:**
  - `MODEL_PROFILE=qwen|swift` in `config/stack.env` sets `SPLASH_MODEL`, `SPLASH_REVISION` (a pin for
    swift), the llama.cpp `MODEL` path and `MODEL_LABEL`; the default is unchanged;
  - `serve-splash.sh` passes `--revision`; `dsh-env.sh` writes generated dsh patches whose labels follow
    the profile (the default profile's patch is byte-identical to the shipped one);
  - `doctor.sh` prints the profile and warns when the running model differs;
  - `eval/run_eval.py` and `kvbench.py` record the profile and model with their results.
- **Docs:** README section 33.

**To do (needs the GPU, ~45 min; `EVAL_REPS=1` adds ~2 h):** `splash/tools/swift_gpu_runs.sh`. It writes
`logs/kvbench/summary-swift.txt`: Splash's startup lines, a qwen-vs-swift comparison (DFlash2 acceptance,
speed, needles, judgment accuracy) and, if asked, the eval summaries. Things to watch:
- draft acceptance on the fine-tune against the default's 0.5-0.7 (the drafter was trained on Qwen3.8-27B);
- llama.cpp + this file at `CTX=262144`, never run.

**Open for the user:** the Swift Open License v1.0 caps commercial use at US$1M gross annual revenue
(with affiliates) without UkisAI's Enterprise License; I told them to check it for their organization.

## 0. Latest state (2026-10-07 ~11:50, paused by the user)

**Done:**
- **Benchmark suite (macOS 27):** turbo-int4 needles 11/11 from 16K to 180K tokens. The answers
  match on every needle all three formats ran. turbo 9/9. int8 9/9.
- **Prefill:** turbo-int4 and int8 are level within ~±12% run-to-run noise. turbo is ~28% slower at 82K.
- **Decode:** tok/s differences are draft acceptance; ms per cycle is equal across formats.
- **Improvement plan:** written to `splash/IMPROVEMENT_PLAN.md`, with a status table at the top.
- **Step 2 is applied in `splash-src`:** turbo-int4 uses 16 codes, a 2.7-sigma clip, a
  least-squares scale and cache revision r2. `make -C splash/tools test` gives 8/8 PASS. Results:
  - key reconstruction error 0.111 → 0.096, values 0.122 → 0.102;
  - attention vs exact, 85/2: mean 0.978 → 0.986, min 0.935 → 0.959, peaked 0.992 → 0.994;
  - attention vs exact, 300/3: mean 0.977 → 0.984, peaked 0.993 → 0.996;
  - attention time unchanged.
- **Patch** re-exported: 20 files, +1712/−77, applies cleanly to 1.2.1.

**End to end, i4-r2 (new quantizer, `logs/kvbench/i4-r2.json`):**
- needles 6/6;
- greedy vs int8: 8/12 identical, up from 6/12;
- judgment KL vs the old int8 run: 0.013. This is NOT comparable: the old int8 run had no nonce and 6
  questions, and there is no noise floor.

**Done without the GPU afterwards (2026-10-07 afternoon; the user needed the GPU):**
- **Docs:** README section 32, the intro box and TURBOQUANT.md carry the final macOS 27 numbers and the
  new quantizer. Prefill is described as level with INT8 within a ±12% run-to-run spread.
- **kvbench judgment set v2:** a slice of each of the 8 source files, 10 planted codes, 23 questions,
  each with a true answer. The v1 set's 24K context was only the first third of `llama-graph.cpp`, so
  most of its file questions were guesses.
  - `compare` prints each format's accuracy and pairs judgments only within the same set.
  - Judgment and greedy prompts have no nonce unless `--nonce` is given; the cache revision makes
    that safe.
  - The meta of a part is written when the part completes.
- **Fair v1 comparison from data already on disk** (24 questions, every run with its own nonce):
  - int8-j24 vs i4-r2 (new quantizer): mean KL 0.015, same top answer 21/24;
  - int8-j24 vs turbo: KL 0.032, top answer 23/24;
  - the turbo-int4 flips are on questions where int8 was at 0.47-0.65.
- **`splash/tools/next_gpu_runs.sh`:** all the GPU runs still owed, ~1.5 h, detached:
  - the v2 quality A/B at 60K: int8 twice, i4 r2, turbo, i4 r1 (the old quantizer is swapped in
    with the revision forced to 1, and always restored by a trap);
  - the 2x2 speed test;
  - a summary written to `logs/kvbench/summary-next.txt`.
  - It replaces the deleted `splash/staged/ab_old_quantizer.sh`, which would not have built:
    restoring the old `PagedKv.hpp` drops `quantizerRevision`, which the new
    `RuntimeResources.mm` calls.

**Step 5 staged (no GPU used):** `splash/staged/step5/` holds turbo's true Lloyd-Max 4-bit key
codebook and turbo cache revision 2.
- CPU checks: a simulation gives 23% lower key error; the kernels compile; the revision has a
  `static_assert`.
- `turbo_kernel_test.py` reads all codebooks from `turbo_kv.h`, so the tests follow the header.

**Resume here:**
1. When the user can spare the Mac (close Teams/WebStorm; `python3 splash/tools/hostavail.py` >= 16):
   `nohup splash/tools/next_gpu_runs.sh > logs/next-gpu-runs.log 2>&1 & disown`.
   It takes ~2 h: step 5's kernel tests, then 6 quality runs and 4 speed runs, then kernel timing.
2. Read `logs/kvbench/summary-next.txt` (its first line says whether step 5 went in):
   - `i8-q vs i8-q2` is the floor; expect KL ~0 if Splash is deterministic.
   - Keep step 2 if `i4-r2-q` is at least as close to `i8-q` as `i4-r1-q`, and at least as accurate.
   - Otherwise copy `splash/staged/orig/*` back (paths in `splash/staged/apply.sh`) and drop
     revision r2.
   - Keep step 5 if `turbo-q` is at least as close to `i8-q` as `turbo-fork-q`.
   - If step 5 stays, update TURBOQUANT.md section 5 and the README table: `turbo` then uses the
     true Lloyd-Max key codebook, not the fork's narrower turbo4 table, so it is no longer
     bit-identical to the fork.
3. Put the numbers into TURBOQUANT.md section 8 and IMPROVEMENT_PLAN.md status; re-export patch 0003.
4. Resolved 2026-10-11: the work is published at github.com/rikitoaf/llm-inference-metal (this workspace,
   with `splash/splash-src` and `llama/` left out; their changes are `patches/0001-0003`). The git identity
   is the global one, `rikitoaf@gmail.com`. Committing `splash-src` itself on `turboquant-kv` is still
   optional; `patches/0003` is what is published.

## 1. Goal and the user's decisions

- **Goal.** TurboQuant KV cache *inside Splash*, so the 24 GB M5 Pro holds a much bigger context.
  The user asked for "turbo ctk 4 for key and ctv 3 for value" as the default "if things get messy".
- **Priorities** (memory `harness-upgrade-priorities`): speed of the whole agent loop, a large context
  in a small memory footprint, accuracy.
- **JEV is off** on both backends; do not propose JEV work.
- **Defaults when measurements justify them.** Keep the old paths switchable, but don't hesitate to
  change the default.
- **No Xcode on this Mac** (Command Line Tools only, SDK 27.0, macOS 26.7). Splash's kernels are
  compiled at runtime; don't ask the user to install Xcode.
- **Next step the user wants:** improve accuracy and speed, and make it concise to run, via a
  multi-agent workflow. Ultracode is ON: use the Workflow tool for substantive tasks.

## 2. What exists now

### Splash source build: `splash/splash-src`

- Clone of github.com/incoai/splash at tag `1.2.1`, branch `turboquant-kv`. The changes are
  uncommitted; three new files are marked with `git add -N`.
- The full diff is in `patches/0003-splash-turboquant-kv-and-runtime-metal.patch` (now 20 files, +1712/-77; was 19 files,
  +1675/-75). It applies cleanly and builds on a fresh 1.2.1 (verified).
- **Re-export after any change:**
  `cd splash/splash-src && git diff 1.2.1 > ../../patches/0003-splash-turboquant-kv-and-runtime-metal.patch`
- **Python:** `splash-src/.venv` is a venv on `/opt/homebrew/opt/splash/libexec/python` with
  `--system-site-packages`. `.venv/.requirements-installed` holds the sha256 of
  `install/requirements.txt`, so make skips pip.
- **Running from source:** `./splash serve` (`install/launcher.py`) runs `make -j4 all`, then the
  server: Python `-m server.server` plus the child process `build/splash serve-native`.
- `build-turbo/` is a scratch build dir (`make BUILD=build-turbo build-turbo/splash`) for compile
  checks. It is untracked; delete it at the end.

### Runtime Metal (no Xcode)

- **Makefile:** `RUNTIME_METAL ?= $(if $(shell xcrun -f metal),0,1)`. With 1, `build/splash.metallib`
  is a DIRECTORY of flattened sources, made by `dev/tools/flatten_metal.py runtime --all <dir>`, one
  file per kernel file (`prefill__x.metal`). `platform-check` then skips the metal/metallib tools.
- **`runtime/metal/MetalBackend.mm`:**
  - if the metallib path is a directory, `Impl::loadRuntimeSources()` compiles every `.metal` with
    `newLibraryWithSource` (Metal 4.0, default math), concurrently via `dispatch_apply`;
  - `sourceFunctions` maps each function name to its library, and `newPipeline` looks names up there.
- **Compile times:** all 31 files in ~11 s cold; macOS caches them afterwards.

### KV formats (`runtime/ops/PagedKv.hpp`)

`enum class Format { Int8=1, BFloat16=2, Turbo=3, TurboInt4=4 }`. CLI names: `int8`, `bf16`, `turbo`,
`turbo-int4` (`main.mm`, `server/serve_options.py`).

**`Turbo`: faithful TurboQuant, K 4-bit / V 3-bit Lloyd-Max**
- **Math:** the fork's (`atomic-llama-cpp-turboquant` `ggml-metal.metal`, `kernel_set_rows_turbo`,
  `quantize_turbo4_0`). Per 128-group: n = |x|; u = x/n; r = S2·H128·S1·u; idx = nearest centroid via
  midpoints (`val >= mid` count); stored norm = n/|c(idx)| (fp16). Dequant = c[idx]·norm, rotated.
- **Signs:** the fork's seed-42 sign arrays, as lane-strided masks: element 32r+l = bit l of word r.
  `SplashTurboSigns1/2[4]`; the source is `splash/tools/turbo_signs.json`.
- **Packing:** token-major bit-planes. K: [token][8 chunks][4 planes] uint; V: [token][8][3] uint.
  Bit j of plane p of chunk c = bit p of dim 32c+j's index. Norms: [token][2 groups] half.
- **Page bytes per layer page (4 heads):** K 16384, Knorm 512, V 12288, Vnorm 512 = 29,696 (INT8:
  66,560). Model page 464 KiB.
- **Store:** `splash_store_turbo_row` in `paged_store_row.h` (tag type `SplashTurboKv`). 256 lanes
  stage the row; simdgroups 0 and 1 each rotate one 128-group (strided: lane holds elements
  l, l+32, l+64, l+96), quantize, then `simd_ballot` per plane.
- **Attention:** `splash_paged_attention_turbo_tile` in `paged_attention_tile.h`.
  - Per page: `splash_turbo_dequantize_keys`. It stages the packed K (256 uint4 + 8 uint4 of norms)
    into the *scores* buffer as staging (free at that point), then dequantizes into `kv_tile`
    (bf16 32x256, 16 KiB, typedef `SplashTurboTileElement`).
  - QK runs on MPP from threadgroup memory; the softmax runs `Quantized=false`.
  - `splash_turbo_dequantize_values` (lane = token, 3 loads plus a register-shuffle codebook) writes
    `kv_tile` dimension-major, then PV.
  - Threadgroup memory: 26,192 B (kv4_g6), 29,456 B (kv2_g8).

**`TurboInt4`: TurboQuant rotation plus uniform int4, read natively by MPP**
- **Rotation:** 256-point over the whole head (`splash_turbo_rotate256`, 8 elements per lane).
  `SplashTurboSigns256First/Second[8]`, from Python `random.Random(250419874)`.
- **Quantization (r2, 2026-10-07):** all 16 codes (-8..7), step min(2.7·rms, max|r|)/8, signed
  so the larger side gets the 8th level. The fp32 scale is refit by least squares, Σr·c/Σc², per
  (token, head), in the INT8 scale layout; both are token-major. Until 2026-10-07 it was symmetric
  -7..7 with absmax/7. MPP `int4b_format` is two's complement, element 2i in the low nibble
  (`int4_probe` confirmed this).
- **Store:** `splash_store_turbo_int4_row`. Simdgroup 0 rotates, reduces max/-min/Σr², quantizes,
  reduces Σr·c and Σc², and packs 8
  nibbles per uint with 3 xor-shuffles; tag `SplashTurboInt4Kv`.
- **Attention:** `splash_paged_attention_turbo_int4_tile`, the INT8 tile with
  `tensor<device int4b_format, dextents<int,2>, tensor_inline>(device uchar*, extents)`.
  - It needs a byte pointer and the dense layout, and `run()` needs lvalue slices.
  - V is token-major, so PV uses the descriptor `(M,D,N,false,false,true)`.
  - Threadgroup memory: 9,808 B (same as INT8).
- **Page bytes per layer page:** 16384 + 512 + 16384 + 512 = 33,792. Model page 528 KiB.

**Shared pieces**
- **Query rotation:** `shared/turbo_rotate.metal`, with `turbo_rotate_queries` (128-groups) and
  `turbo_rotate_queries_256`. In `PagedAttention.cpp` (`addTurboQueryRotation`) it is dispatched
  before the split in `addPrefill`/`addVerify`, over all kvHeads × stride × groupHeads vectors.
- **Output rotation:** reduces get a template param `Rotation` (0/128/256) in
  `splash_attention_reduce_row`; the entries are `*_reduce_turbo` and `*_reduce_turbo_int4`.
- **Kernel names:** `formatPipeline(kernel, format, FormatKernels{q8, bf16, turbo, turboInt4})` and
  `reducePipeline()` in `PagedAttention.cpp`.
- **Geometry:** `abi/KvExtent.h` gained `SplashKvPageGeometry{key_data,key_scales,value_data,value_scales}`
  and `splash_kv_geometry_offset/layer`; the old `splash_kv_offset/layer` are wrappers.
  `Layout::geometry()`; `PageStorage.mm` uses it. INT8/BF16 offsets are byte-identical to upstream.
- **Status:** `Status.cpp` reports format, quantization and layouts. The persistent cache namespace
  includes `formatName`, so formats never mix.
- **Note:** if a turbo layout changes, bump `kPersistentCacheFormat` (`engine/RuntimeResources.mm`).

### Workspace integration

- **`config/stack.env`:** `SPLASH_SRC`, `SPLASH_KV_FORMAT` (default `turbo-int4` since
  2026-10-06), `SPLASH_PERSISTENT_CACHE=1`.
- **`scripts/serve-splash.sh`:**
  - uses the source build when it exists, else Homebrew 1.1.0 (forced to int8);
  - `--kv-format`, `--max-cache-disk 24G`, plus `--persistent-cache` on the source build;
  - `SPLASH_EXTRA_ARGS`.
- **`scripts/dsh-env.sh` (`splash_patch()`):** reads `/status` `maximum_context_tokens` and writes
  `dsh-home/splash.generated.cordis.patch.yml` with contextWindow = limit − 16384 − 4096 (e.g.
  172,025 for turbo-int4). The static `config/dsh/splash.cordis.patch.yml` fallback is 73,728.
- **`scripts/doctor.sh`:** reports the source build, the TurboQuant kernels and the KV format and
  context. It has not been run since that edit; run it after the suite.
- Homebrew Splash 1.1.0 is still installed and untouched (`/opt/homebrew/opt/splash`).

## 3. Tools: `splash/tools`

| Tool | What |
|---|---|
| `Makefile` | `make test` (8 kernel test cases, both formats; fails on any non-PASS), `make perf H=...`, builds the harnesses |
| `hostavail.py` | Splash's own reclaimable-memory estimate (`--gib` for scripts); it needs ~16 GiB to start |
| `rtc_probe` (`.mm`) | Runtime-compiles a flattened file and builds pipelines (`rtc_probe FILE [kernel...]`); reports threadgroup memory |
| `turbo_kernel_test.py` | turbo store, dequant and 128 rotation vs NumPy (harness `turbo_harness`, kernels in `turbo_test_kernels.metal`) |
| `turbo_int4_test.py` | turbo-int4 store bytes and scales, 256 rotation vs NumPy (`turbo_harness ... int4`) |
| `turbo_attention_test.py COMMITTED SPLITS [int4]` | Full verify attention vs reference (`turbo_attn_harness`) |
| `attn_perf FLAT HISTORY [ROWS]` | Times int8 / turbo / turbo-int4 prefill-chunk and verify-step attention per layer, from `perf_kernels.metal` flattened; random pages |
| `perf_variants.py [H...]` | Textual kernel variants (current, half-tile, no-dequant, keys-only, values-only, k-no-lut, ...) timed with `attn_perf` |
| `int4_probe` | Proves the MPP int4 encoding |
| `research/` | Obsolete prototypes (`int4_proto*.metal`, `perf_kernels_int4.metal`), kept for reference |
| `kvbench.py run --label L [--parts speed needles judgments greedy] [--needle-tokens ...] [--depths ...]` | End-to-end benchmark against the running server. Saves `logs/kvbench/L.json` after every part and merges reruns. `compare A B` prints greedy agreement, judgment KL, needles, speed |
| `next_gpu_runs.sh` | Every GPU run the plan still owes, in one detached job; writes `logs/kvbench/summary-next.txt` |
| `run_kv_suite.sh` | Reads lines "FORMAT LABEL [ENV=K=V,...] kvbench-args" from stdin; restarts the server per format (4 tries on a memory refusal); run it detached with nohup |

**Build:** `make -C splash/tools` (or `clang++ -std=c++20 -O2 -fobjc-arc -framework Foundation -framework Metal X.mm -o X`).
The tests use the build's flattener, `splash-src/dev/tools/flatten_metal.py`, and `$TURBO_WORK` (default `$TMPDIR`) for scratch.

**Background-task rules:**
- Background Bash tasks are killed after ~30 min. Long jobs must be started with
  `nohup ... & disown`.
- Monitors expire after 30 min; re-arm them.

## 4. Measurements so far

**Capacity (`/status` maximum_context_tokens, `--language-only`, 24G SSD tier):**

| Format | Context | Page per 32 tokens |
|---|---:|---:|
| int8 | 94,201 | 1,040 KiB |
| turbo-int4 | 192,505 | 528 KiB |
| turbo | 212,985 | 464 KiB |

**Kernel tests (all PASS):**
- turbo store and dequant: bit-exact vs NumPy. Reconstruction error: keys 0.108, values 0.182.
- turbo-int4 store: 0/16384 bytes differ. Reconstruction error: keys 0.111, values 0.122.
- Rotations: exact.
- Attention vs reference: cosine ≥ 0.9999.
- TurboQuant vs exact attention on random data: turbo mean cosine 0.969 (peaked row 0.979-0.986);
  turbo-int4 0.978 (peaked row 0.991-0.993).

**`attn_perf`, one layer, ms (prefill = one 2048-row chunk after H history; verify = one 8-row step),
quiet GPU:**

| H | int8 pf / vf | turbo pf / vf (staged keys) | turbo-int4 pf / vf |
|---|---|---|---|
| 4096 | 16.3 / 0.117 | 38.2 / 0.175 | 15.9 / 0.106 |
| 16384 | 54.0 / 0.330 | 127.9 / 0.530 | 53.2 / 0.289 |
| 57344 | 196.6 / 0.997 | 460.9 / 1.783 | 184.2 / 0.892 |
| 131072 | 457.6 / 2.098 | 1030 / 3.947 | 433.3 / 1.844 |
| 196608 | 689.1 / 3.029 | 1549 / 5.937 | 656.0 / 2.715 |

**Turbo variant attribution (H=16384, before staging): current 138, no-dequant 48.5, values-only 78.8,
keys-only 111.9, k-store-only 77.2.**
- Key device loads were the largest cost.
- Removing the codebook lookup or the bit extraction each saved only 2-6%.
- fp16 tiles: no gain (MPP mixes bf16 × half fine).
- Shuffle-broadcast keys: worse (195 ms). Register prefetch: spills (350 ms). Unroll: no gain.
- Staging keys via scores: 138 → 127.5 (current production).

**End to end, `kvbench` (Qwen3.8-27B UD-IQ3_XXS + DFlash2, temperature 0, thinking off):**

- **INT8** (`logs/kvbench/int8.json`; speed and needles recovered from `logs/kvbench-int8.log`; run
  WITHOUT `--persistent-cache`):
  - decode on 400 tokens: 64.0 / 64.1 tok/s;
  - needles 9/9 correct at ~16K / 58K / 82K tokens;
  - prefill 406/377/378 (16K), 326/326/330 (58K), 298/297/298 (82K) tok/s.
- **turbo**, first build before staging, interrupted (`logs/kvbench-turbo.log`, with persistent cache):
  - decode 67.0 / 66.5 tok/s;
  - needles 16K ×3 and 58K ×2 OK;
  - prefill 340/353/340 (16K), 223/228 (58K).
- **turbo-int4** (suite run 1, `logs/kv-suite.log`, with persistent cache):
  - decode 59.4 / 58.2 tok/s;
  - 16K ×3 OK, prefill 366/328/326; 58K@10% OK, prefill 273.
- Every needle answer so far is identical across formats (same passphrases).

**Resolved 2026-10-06: the "prefill gap" was the `--persistent-cache` flag, not TurboQuant.**
INT8 re-run on macOS 27 WITH `--persistent-cache` (`logs/kv-suite-macos27-partial.log`, nonce prompts):
- decode 64.3 / 64.4 tok/s;
- needles 5/5 OK; prefill 348/350/326 (16K), 271/254 (58K); the run was stopped before 58K@90% and 82K.

So under identical flags, turbo-int4 (366/328/326, 273/278/289) is level with or ahead of INT8.
INT8 without the flag ran 377-406 (16K) and 326 (58K).
- **Cost of the flag:** `--persistent-cache` (or macOS 27; the two are confounded) costs ~10-20% prefill.
- **Suspect:** the write-behind of every 32-token page plus states to the SSD during prefill.
- **Worth testing:** turbo-int4 with and without the flag on the same OS. If the flag is the cost,
  make it opt-in or throttle the write-behind during prefill.

**Benchmark artifact fixed:** a persistent cache serves the same deterministic needle prompts on a
later run of the same format.
- Example: turbo on 2026-10-05 21:35 shows "prefill 77,966 tok/s", served from the 20:25 turbo run.
  That `turbo.json` needles entry is invalid.
- Fix: `kvbench.py` now puts a `[run <ns>]` nonce before each needle prompt, and computes prefill
  tok/s over `prompt_n - cache_n`.
- Namespaces are per format (three directories under `~/Library/Caches/Splash/prefix-cache/`), so
  formats were never mixed.

**macOS 27.0 (2026-10-06):**
- All four kernel tests still PASS.
- Starting Splash needs ~16 GiB "available" by its own estimate. Use
  `python3 splash/tools/hostavail.py`,
  or the formula in `runtime/engine/MemoryGovernor.cpp`: free + file-backed + purgeable + compression savings.
- With WebStorm + Teams open it measured 14.8 GiB, and every format failed with
  `host memory reserve protected`. With them closed it measured 15.7 GiB and started.

**Short decode tok/s varies ±10% between formats.** DFlash2 acceptance depends on the generated
text; compare with greedy agreement rather than single 400-token runs.

## 5. Benchmarks still to run (suite stopped 2026-10-06 ~11:00 at the user's request)

The suite needs a quiet GPU and ~16 GiB available memory: close Teams and WebStorm first.

Results already in hand:
- turbo-int4: full run on macOS 26, `logs/kv-suite-macos26.log` and `logs/kvbench/turbo-int4.json`.
  Needles 9/9 at 16K/58K/82K, plus judgments and greedy.
- int8: judgments and greedy on macOS 26.
- int8: speed and 5 needles on macOS 27.

Remaining plan (`logs/kv-suite.plan` format):
```
int8 int8 --parts needles --needle-tokens 60000 88000
turbo-int4 turbo-int4 --parts speed needles --needle-tokens 16000 60000
turbo turbo --needle-tokens 16000 60000 88000
turbo-int4 turbo-int4 --parts needles --needle-tokens 150000 197000 --depths 0.5
```
Run it with `nohup splash/tools/run_kv_suite.sh < logs/kv-suite.plan > logs/kv-suite.log 2>&1 & disown`.
A part's results are saved only when the part ends.

**When done:**
1. `python3 splash/tools/kvbench.py compare int8 turbo-int4` and `compare int8 turbo`;
2. fill the numbers into README section 32 and `splash/TURBOQUANT.md`;
3. ~~set `SPLASH_KV_FORMAT` to `turbo-int4`~~ done 2026-10-06;
4. restart via `./scripts/serve-splash.sh`;
5. run `./scripts/doctor.sh`;
6. run the agent demo and `eval/` tasks on the final config.

## 6. Improvement backlog (candidates for the workflow)

**Accuracy**
- turbo-int4: an MSE-optimal clipping scale instead of absmax (Gaussian after rotation: clip ~2.5-2.8σ).
- 16 asymmetric levels (−8..7 plus an offset correction using Σq, done in the softmax and the reduce).
- Keys: per-channel smoothing before the rotation (K outlier channels from RoPE).
- QJL-style residual for keys (paper's turbo).
- Keys at higher precision than values (K int8 + V int4 mixed?).
- Evaluate with `/v1/judgments` KL and needles at depth; synthetic NumPy studies first.

**Speed**
- The `--persistent-cache` write-behind costs ~10-20% prefill (section 4). Measure it and throttle or make it opt-in.
- Fuse the query rotation into the QKV-prepare kernel (`attention_qkv.metal`) to drop a dispatch.
- turbo: amortize dequantization across query tiles (bigger M with half-page tiles, or a per-chunk
  dequant to an int8 scratch); or drop turbo as default.
- Verify path: check DFlash acceptance per format.

**Conciseness and run simplicity**
- A single `serve-splash.sh` default.
- Remove obsolete prototypes (`int4_proto*`, `perf_kernels_int4.metal`) and `build-turbo/`.
- A `make` target or script for tests.
- Doctor checks.
- README brevity.

## 7. Scratch locations

`/private/tmp/claude-501/` holds flattened kernels, `perf-variants/`, test work dirs and
`readme-32.md` (draft of README section 32). Only the draft matters.

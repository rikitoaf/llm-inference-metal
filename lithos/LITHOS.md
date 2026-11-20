# lithos-metal with turbo-int4 KV and GGUF weights

State as of 2026-10-14. Read this before resuming work on the lithos backend. The usage summary is README section 34.

## What lithos-metal is

- **Source:** github.com/lithos-ai/lithos-metal, Apache 2.0.
- **Clone:** `lithos/lithos-src`, branch `turboquant-kv`, based on upstream `5485ad6` (v0.1.2).
- **Our changes:** in `patches/0004-lithos-turbo-int4-kv-and-gguf.patch`. Re-export after any change:
  `cd lithos/lithos-src && git add -N . && git diff 5485ad6 > ../../patches/0004-lithos-turbo-int4-kv-and-gguf.patch`.
- **Python:** `lithos/lithos-src/.venv` (Homebrew Python 3.14) with `pip install -e '.[serve,dev,oracle]'` and `gguf`.
- **Build:** the native runtime builds during that install with the Command Line Tools. Metal compiles at runtime,
  so no Xcode is needed.
- **Engine:**
  - a Python compiler emits per-layer Metal kernels;
  - the step is replayed from an indirect command buffer;
  - DSpark (a block drafter with a Markov head) proposes up to 7 tokens, which the target verifies in one pass.

## 1. turbo-int4 K/V cache (`--kv-format turbo-int4`)

- **Format:** Splash's turbo-int4 r2 (`splash/TURBOQUANT.md`).
  - R = S2·H·S1/√D with Splash's sign words.
  - 16 levels, step = min(2.7·rms, max|x|)/8, a least-squares FP32 scale.
  - A row is D/2 bytes plus the FP32 scale: 132 bytes at D = 256 against BF16's 512.
  - State entries are U8 `[ctx, kv_heads, D/2 + 4]`. The prefix cache and its disk store slice them by row. The
    store's identity hashes the entries' dtype and shape, so formats never mix.
- **Kernels:**
  - `gqa_common.metal`: `turbo_rotate`, `turbo_quantize`, `turbo_store`, `turbo_load`, `turbo_fresh`.
  - v1 core (`gqa_decode.metal`) and `gqa_merge`, which applies R^T to the folded output.
  - The staged matrix core: `gqa_decode_mma` rewritten by `kernels.turbo_mma_source`. It is a text rewrite, so the
    upstream file stays byte-identical for the attention task compiler, which splices that text.
  - Queries are rotated after RoPE. The step's own keys and values are quantized, appended once, and used as their
    dequantized values, so a long prompt pass equals its 128-row passes bit for bit.
- **Emitter** (`_gqa_turbo`, `_gqa_turbo_kind`):
  - the matrix core where the chip picks it, and in place of v3 when the accelerator is on;
  - otherwise v1;
  - never the direct or adaptive forms, and never on an LM drafter.
- **Fusion recipes:** they rewrite BF16 attention and leave turbo-int4 cores alone. `specialize_attention` raises,
  `decoder_fusion` leaves the region native, and the 40-core M5 Max prefill hook skips it.
- **Measured** (Qwen3.8-27B attention shape, GPU ms per layer): turbo-int4 on the matrix core is within 2–5 % of BF16
  at 8 and 128 rows, and faster at 1 row. On v1 alone it was ~3× slower at 8 rows, because attention is compute-bound
  there. Accuracy against BF16 on random data: output cosine 0.990 mean, 0.985 worst.

## 1b. 170K context (2026-10-14, the user asked for ≥120K, then 170K)

- **The problem:** upstream lithos serves 32K. Its matrix attention stores one partial (o, m, d) per 32 keys per query
  row. At 170K a 128-row prompt chunk needed ~16 GiB of workspace, and the drafter's BF16 context cache needed 2.5 GiB.
- **Bounded partitions** (`BOUNDED_CHUNK` in `gqa_common.metal` `pick_chunk`, shared by the core and `gqa_merge`):
  - a partition is `CH · ⌈ctx / (CH · 64)⌉` keys;
  - `kernels/common/gqa_turbo_mma.metal` walks its 32-key tiles with an online softmax (running max and denominator
    per row in threadgroup memory, the value accumulator rescaled per tile);
  - the workspace is sized for `kernels.GQA_TURBO_PARTITIONS = 64` (`nn/attention.py`, `_gqa_chunk_capacity`,
    `_draft_attn`).
  - Measured faster than one partial per tile: 32K, 8 rows 3.38 → 2.17 ms; 128 rows 52 → 35 ms.
- **Prepared queries** (2+ rows): `gqa_turbo_prepare` writes the rotated queries once (`gqa.qprep` scratch) and
  appends the step's keys and values. `gqa_turbo_mma_q` then reads every key and value from the cache, takes 64 query
  rows per block, and skips tiles entirely after the block's last query.
  - Scores are kept in BF16 and overwritten in place by the probabilities, which fits D = 128 (64-key tiles) in 32 KB.
  - 1.2–1.3× (8 rows) and 1.7–1.8× (128 rows) over the staged core.
  - A single row stays on the staged core: 3.7 ms against 8.3 ms prepared at 170K.
- **The DSpark drafter's context cache** is turbo-int4 (`DraftAttention.set_kv_format`): the `DRAFT` form of
  `gqa_turbo_mma.metal`. Injected context rows are quantized and appended; the block's own proposals stay exact. It
  needs the matrix-unit draft attention (accelerator on, head_dim 128 or 256). `load_session` sets the format on the
  drafter too.
- **Prefix checkpoints above the host budget** go straight to the disk store (`PrefixCache.save`) instead of being
  dropped. The new `--prefix-cache-memory` option sets that budget; the harness uses 1G.
- **Memory at 174,080 tokens:** weights 12.07 + caches 3.62 + recurrent 0.29 + workspaces 0.37 = 16.35 GiB.
- **Tests:**
  - `test_gqa_turbo_int4.py`: bounded partitions with caps 3 and 7 against single tiles and v1; prepared queries
    against the staged core.
  - `test_turbo_draft.py`: the turbo draft program against its oracle across multi-tile partitions (2,100 cached
    positions).
  - `test_prefix_cache.py`: write-through.
  - The session tests: speculative decoding with the turbo drafter.

## 2. GGUF weights read as llama.cpp stores them

**Why:** lithos reads MLX 4-bit / NVFP4 / FP8 / BF16 safetensors only, and its smallest Qwen3.8-27B is 15 GiB. Every
local model here is a GGUF i-quant (UD-IQ3_XXS 10.2 GiB, GSQ IQ2_S, Swift, Ridge). The user chose to teach lithos the
GGUF quants (option B) over converting or downloading.

**Checkpoint:**
- `monolith/formats/gguf_checkpoint.py` is the generic reader: a `Rule` per tensor (matrix rows and gather, f32 / bf16
  value maps) from an adapter registered per GGUF architecture.
- The adapters live with their model and drafter packages, because lithos's hygiene CI keeps model names inside
  `monolith/models/`: `monolith/models/qwen3_5/gguf_adapter.py` (`qwen35`) and `monolith/spec/dspark/gguf_adapter.py`
  (`dflash`).
- `prepare_gguf(file, dir)` writes `config.json` (from GGUF metadata), the HF tokenizer and `gguf.json`.
  - The tokenizer is found among Splash's model metadata and checked token by token.
  - `serving/cache.resolve_checkpoint` does this for any `--model x.gguf` / `--draft x.gguf`, under
    `~/.cache/lithos-metal/gguf/`.
- `SafetensorsDir(dir)` returns a `GGUFCheckpoint`. The `qwen35` adapter serves HF names with llama.cpp's conversion
  undone:
  - norms: the stored `1 + w` comes back as `w`;
  - `A_log = log(−ssm_a)`;
  - the linear-attention V heads go back from tiled to grouped order: the rows of `in_proj_qkv` (V part), `z`, `a`,
    `b`, plus `A_log`, `dt_bias` and the conv taps;
  - `out_proj`'s columns cannot move inside blocks, so the matrix carries a column gather that the kernels apply to
    their input;
  - the MTP block is ignored.
- Architectures: `qwen35` (the target) and `dflash` holding a DSpark drafter. For the drafter, `target_layer_ids` is
  the stored value − 1; its embedding and head are shared with the target.

**Format** (`monolith/formats/gguf.py`, format `gguf`):
- Every slab row is the tensor's row of GGUF blocks with its own ggml type: header, row table `(offset, type)`, an
  optional gather table plus its inverse, then 16-byte-aligned rows.
- The oracle is the `gguf` package's dequantizer.
- `LAYOUT = 2` is in the pack cache key.
- `Linear` keeps a gguf silu·mul slab as `[gate | up]` (no chunk interleave). The emitter refuses an interleaved one.

**Kernels:**
- `kernels/common/gguf_quants.metal` is vendored by `tools/vendor_ggml_quants.py` from llama.cpp @ 044dff24 (MIT):
  block structs, grid tables and `dequantize_*`. One edit: Q4_K's half-precision `/16.h` became `/16.f`, which had
  caused up to 0.7 % error with subnormal block scales.
- `kernels/common/gguf_gemv.metal`:
  - `gguf_mv` (≤ 2 rows): 8 SIMD-groups × 4 outputs per threadgroup, activations staged 512 columns at a time in
    threadgroup memory, each chunk decoded once for all rows;
  - `gguf_mm` (> 2 rows): a 64-row weight tile dequantized (half) per K step of 32, then MPP `matmul2d` against the
    activations in device memory, with 16-token tiles up to 16 rows and 64 above, as llama.cpp's tensor
    `kernel_mul_mm`;
  - `gguf_embed`;
  - all type-specialized (`GGUF_TYPE`, `GGUF_TYPE_B` for silu·mul's up rows), with gemv_T's epilogues and StepState
    row counts.
- **Emitter** (`_gguf_gemv`, `_gguf_runs`):
  - one dispatch per run of one type per T variant;
  - norm inputs go through `norm_apply`;
  - no tiles, fused permutes or hoisted statistics (`fuse_norm_stat` skips gguf producers).
- **Tests:**
  - `tests/kernels/test_gguf_gemv.py` (8): every type on real rows from the local GGUFs at 1–128 rows, silu·mul,
    residual, gather, embed.
  - `tests/kernels/test_gguf_model.py` (6): a synthetic hybrid written as a GGUF with llama.cpp's transforms:
    - the reader returns every HF tensor exactly;
    - BF16 GGUF decodes the safetensors tokens;
    - Q8_0 matches the oracle;
    - turbo-int4 matches its oracle to ties;
    - DSpark speculative equals plain, with the accelerator on and off.

**Real model:**
- UD-IQ3_XXS + the local DSpark Q8_0 pack in ~1 min: target 10 GiB, draft 1.9 GiB.
- Emitted programs: prefill (128 rows) 833 dispatches, verify (8 rows + DSpark) 1,734.
- All 216 distinct kernels compile.
- Memory ≈ 9.95 + 1.9 GiB of weights + 0.52 GiB of K/V at 32K.

**Kernel speed** (real matrices; full table in README 34):
- 1 row: 140–200 GB/s on IQ3_XXS / IQ4_XS / Q4_K; IQ2_S / IQ1 are ALU-bound.
- 8 rows (matrix units): IQ2_S gate+up 0.89 ms, Q4_K lm_head 5.7 ms.
- Prefill: 13–25× faster than the first generic kernel.

## 3. 16-core M5 Pro backend

- `monolith/backends/metal/m5_pro/config-16c.json`, registered in `registry.py`. The engine block is the 20-core
  one's (`validation: derived`).
- Registering it un-skips 4 `test_draft_program` cases (fused draft tile `cached_inputs` 24×32), which fail on
  untouched upstream code too. Serving never uses that tile on this chip: fused draft recipes ship for the 40-core
  M5 Max only.

## Harness

- `scripts/serve-lithos.sh` (`LITHOS_MODEL` defaults to `MODEL_PROFILE`'s GGUF, `LITHOS_DRAFT` to the local DSpark GGUF).
- `config/dsh/lithos.cordis.patch.yml`: no `reasoning_effort`, since lithos serves with thinking off.
- `lithos_patch()` in `dsh-env.sh` sets the context window to the server's `--max-context` − 8192 − 4096.
- `stop.sh` and `doctor.sh` handle lithos.

## 4. Real model, measured (2026-10-14)

UD-IQ3_XXS target + local DSpark Q8_0 drafter, turbo-int4 K/V for both, `LITHOS_MAX_CONTEXT=174080`, verify 8 rows.
Same kvbench prompts as Splash's `logs/kvbench/i4-r2.json` (Splash turbo-int4, same GGUF), run through
`splash/tools/kvbench.py`'s own prompt builders. lithos has no `/status`, `/tokenize` or `/v1/judgments`: a wrapper
stubs `/status`, counts tokens with the HF tokenizer, and reads timings from the server's `Generation:` log lines.
Results: `logs/kvbench/lithos-i4.json`, logs `logs/kvbench/lithos-i4-{greedy,needles}.log`.

**Accuracy: same as Splash.**
- **Tokenizer:** the HF tokenizer lithos serves with is token-identical to llama.cpp's on the GGUF (4/4 kvbench texts,
  6–7K tokens each; `llama-tokenize --no-escape`). An apparent difference on literal `\n` was llama-tokenize's escape
  processing. Needle prompts tokenize to the same counts as Splash's (15977, 58088).
- **Needles:** 6/6 (16K and 58K tokens at depths 10/50/90 %), as Splash.
- **Greedy text** (6 prompts at 0 and 24K context, 160 tokens): 7/12 identical to Splash turbo-int4, 425/632 words
  before the first divergence. Splash agrees with itself about as well: int8 vs turbo-int4 8/12 (449/645), two Splash
  turbo-int4 builds 5/12 (310/631). The divergences are near-tie word choices ("By caching" / "By reusing"), and the
  answers stay correct (17 × 23 = 391; the haiku is a different haiku).

**Speed: 4–5× slower than Splash.**

| | lithos | Splash (i4-r2) |
|---|---:|---:|
| verify step, 0 / 24K / 58K context | 354 / 410 / 457 ms | 91 ms per cycle |
| tokens per step, 0 / 24K context | 3.73 / 2.71 | 4.51 |
| decode (tokens per step ÷ step time), 0 / 24K context | 10.5 / 6.6 tok/s | ~50 tok/s |
| prefill 16K / 58K | 99–100 / 79–80 tok/s | 398–414 / 338–343 tok/s |
| prefill, a 25-token prompt | ~570 ms | ~197 ms |

- A 58K-token prompt takes 12 minutes to prefill (Splash: under 3).
- DSpark acceptance is lower than Splash's on the same drafter file: 3.73 tokens per 8-row step at 0 context against
  4.51.
- **Fixed during this run: the T-variant predicate** (`gguf_gemv.metal`). `gguf_mv` / `gguf_mm` tested
  `T_LO < T ≤ T_HI` on the row count already capped to the variant's `GG_T`, so every variant ran in full on every
  step. Fixing it took the verify step from 845 to 364 ms (4.7 → 12.5 tok/s on the first chat). Regression test:
  `test_variants_outside_their_row_range_do_nothing`.
- **Fixed: no prefix checkpoint for a single-message prompt** (`generate.py`). The server passes the end of the
  earlier blocks of the final message as the reusable prefix. With one block that is only the template header, under
  `min_tokens`: the header was dropped and the `p − 1` fallback never applied, so even an exact repeat prefilled
  again. A prefix under `min_tokens` now falls back to `p − 1` (test: `test_chunked_and_exact_prefill_and_prefix_cache`).
  Splash still reuses more: it shares page-granular prefixes between different questions about one document, and
  lithos checkpoints only whole-message boundaries and `p − 1`.
- **Memory:** process footprint 5.6 GB (peak 7.9 GB) plus the file-mapped pack. It runs with Teams and WebStorm closed.
  GPU tests cannot run beside it (out of GPU memory).

## Next (if lithos is to beat Splash)

1. **Prefill (4× behind):** `gguf_mm` dequantizes each weight tile once per 64 tokens (`MM_TOK`). Measure a 128-row
   prefill step per op, as `replay_vs_profile.py` does for verify, then try 128-token tiles or larger chunks.
2. **Verify step (354 ms against Splash's 91):** per-op profile of the fixed predicate build. Before the fix,
   gate_up took ~1.5 ms per layer in the program against 0.89 in isolation, and the lm_head ~12 ms each for target and
   drafter.
3. **DSpark acceptance:** compare the drafter's proposals with Splash's on one prompt (same file, 3.7 vs 4.5 tokens
   per step).
4. Prefix checkpoints at chunk boundaries inside one long message (Splash-style reuse across questions).
5. Optional: a turbo-int4 v3 for single-row decode without a drafter.

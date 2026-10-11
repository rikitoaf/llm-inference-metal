# TurboQuant KV cache in Splash

This document covers what was built on 2026-10-05:
- why it was built;
- how every part works;
- how it was verified;
- what was measured, including what did not work;
- how to run it.

`README.md` section 32 is the short version. `HANDOFF.md` is the raw working state;
`TURBOQUANT_PLAN.md` holds the design notes and the work log.

Machine: Apple M5 Pro (16-core GPU, Apple10), 24 GB, macOS 26.7, Command Line Tools only (no Xcode).
Model: Qwen3.8-27B, Unsloth UD-IQ3_XXS, with Splash's DFlash2 block drafter.

---

## 1. Summary

**Starting point.** Splash keeps the attention KV cache as INT8, with one fp32 scale per token and
head. On this Mac that leaves room for **94,201 tokens** of context. The task was to bring TurboQuant
(arXiv 2504.19874) into Splash's KV storage, with the llama.cpp fork's default of 4-bit keys and
3-bit values.

**Result.** Splash 1.2.1 now builds from source on this Mac and serves two new formats:

| `--kv-format` | What is stored | Context on 24 GB | Attention cost vs INT8 | Notes |
|---|---|---:|---|---|
| `int8` | Splash's default | 94,201 | 1x | reference |
| `turbo-int4` | Keys and values rotated by a randomized 256-point Hadamard transform, then int4 (all 16 codes) with a clipped least-squares scale | **192,505** (2.04x) | **0.90-0.98x the time** | The GPU's matrix units read the 4-bit pages directly |
| `turbo` | TurboQuant as in `atomic-llama-cpp-turboquant`: 4-bit keys and 3-bit values on Lloyd-Max codebooks | **212,985** (2.26x) | 2.2-2.4x the time | Pages are decoded inside the attention loop |

**Accuracy.** Both formats pass bit-exact kernel tests against an independent NumPy implementation.
In end-to-end needle retrieval, every answer equals INT8's out to 82K tokens. `turbo-int4` also finds
needles at 139K and 180K tokens, beyond INT8's limit. The rest of the numbers are in section 8.

## 2. Building Splash without Xcode

Splash ships its GPU kernels precompiled in `splash.metallib`. Building it needs `xcrun metal` and
`metallib`, which come only with Xcode. This Mac has only the Command Line Tools. macOS itself does
contain the runtime Metal compiler (`GPUCompiler.framework`), which `MTLDevice newLibraryWithSource:`
uses. The llama.cpp fork already runs that way.

**The source build.** `splash/splash-src` is a clone of github.com/incoai/splash at tag `1.2.1`,
branch `turboquant-kv`.
- The patch `patches/0003-splash-turboquant-kv-and-runtime-metal.patch` adds the runtime-Metal mode
  and both formats.
- It applies cleanly to a fresh 1.2.1 and builds there.

**Runtime-Metal mode** (Makefile, `dev/tools/flatten_metal.py`, `runtime/metal/MetalBackend.mm`):
- `RUNTIME_METAL` defaults to 1 when `xcrun -f metal` fails.
- `build/splash.metallib` then becomes a directory with one flattened source per kernel file.
  `flatten_metal.py` inlines `#include "metal/..."` once per file and keeps system includes.
- `MetalBackend` sees a directory instead of a file. It compiles every source concurrently with
  `newLibraryWithSource` (Metal 4.0, default math mode, matching the offline `-std=metal4.0 -O3`
  build) and resolves kernels by function name.
- All 31 kernel files, MetalPerformancePrimitives (MPP) tensor kernels included, compile in ~11 s
  cold. macOS caches the result, so later starts are fast.
- The engine's C++ builds with the Command Line Tools' clang under upstream's
  `-Wall -Wextra -Werror` in ~9 s.

**Python.** `install/requirements.txt` is identical in 1.1.0 and 1.2.1, so `.venv` is a venv on the
Homebrew Splash's Python (`--system-site-packages`). `.venv/.requirements-installed` holds the
requirements' sha256, so `make` never runs pip.

**Memory at start (1.2.x).** Splash 1.2 loads the weights into memory at every start; 1.1.0
memory-mapped a prepared copy. Its governor admits them only with macOS's 2 GiB reserve and a 1 GiB
margin still free. With many apps open (~13 GB reclaimable) it fails with `unable to admit model
arenas: host memory reserve protected`; after a reboot it starts in ~25 s.
`--idle-release` (default 10 min) frees the weights while idle and reloads them in ~4 s.

**The 1.2 persistent cache.** `--persistent-cache` keeps the SSD prefix cache (`--max-cache-disk`)
across restarts. Its namespace digests the KV format, so a `turbo-int4` cache is never read as
`int8`.

## 3. TurboQuant in brief

TurboQuant compresses each key or value vector as follows:
1. **Rotate** it by a fixed random orthogonal matrix. Here that is a randomized Hadamard transform:
   random signs, then a normalized Walsh-Hadamard transform, then random signs. After it, any outlier
   channel's energy is spread over all coordinates, and each coordinate is close to a Gaussian with
   variance |x|²/d.
2. **Quantize** every coordinate with a scalar quantizer suited to that distribution: Lloyd-Max
   centroids in the paper and in the llama.cpp fork.
3. **Attention runs in the rotated basis.** A rotation keeps dot products, so the queries are
   rotated the same way and q·k is computed directly against the quantized rotated keys. The
   attention output is a weighted sum of rotated values; it is rotated back once per row.

The llama.cpp fork (`atomic-llama-cpp-turboquant`, `ggml/src/ggml-metal/ggml-metal.metal`) quantizes
per 128-value group:
- n = |x|; u = x/n; r = S2·H128·S1·u;
- idx_i = the nearest centroid (8 levels for 3 bits, 16 for 4 bits, for N(0, 1/128));
- stored norm = n / |c(idx)|, a corrected norm, so the reconstruction c(idx)·norm keeps the vector's
  exact length.

Its turbo3 block is 50 bytes per 128 values (3.125 bits) and turbo4 68 bytes (4.25 bits).

## 4. How Splash's KV cache is laid out, and what changed

Splash keeps KV in **pages of 32 tokens**:
- Pages live in **extents** of shared Metal memory.
- Kernels reach a page through its table entry: the extent's GPU address with the page index in the
  low bits.
- Per attention layer, an extent holds four regions: keys, key scales, values and value scales. Each
  region starts 64 KiB-aligned.
- INT8 stores keys token-major and values dimension-major, with one fp32 scale per (token, head)
  for each.

**Geometry** (`metal/abi/KvExtent.h`, `ops/PagedKv.hpp`, `ops/PageStorage.mm`):
- Upstream assumed keys and values share one data size and one scale size.
- `SplashKvPageGeometry {key_data, key_scales, value_data, value_scales}`,
  `splash_kv_geometry_offset()` and `Layout::geometry()` generalize that to four sizes.
- The old two-size functions are wrappers, and INT8 and BF16 offsets are byte-identical to upstream.

**Formats** (`kv::Format`):
- `Turbo = 3` and `TurboInt4 = 4`;
- names `turbo` and `turbo-int4` in `main.mm` and `server/serve_options.py`;
- `Status.cpp` reports their quantization and layouts.

**Kernel selection** (`ops/PagedAttention.cpp`): `formatPipeline()` and `reducePipeline()` pick the
store, split and reduce kernels by format.

**Dispatch order** for a turbo layer, in a serial compute encoder:
1. store the current rows;
2. rotate the queries in place (`turbo_rotate_queries` or `turbo_rotate_queries_256`);
3. split attention;
4. a reduce that rotates the output back.

The draft model (DFlash2) keeps its own BF16 ring cache and never reads target pages, so nothing
changes there.

## 5. Format `turbo`: TurboQuant as the fork computes it

**Math.** Identical to the fork:
- the same seed-42 signs, the same 128-point FWHT, the same centroids and midpoints;
- the same tie rule (an index counts the midpoints at or below the value);
- the same corrected fp16 norm per 128 values.

**Packing** differs, to fit Splash's 32-token pages and SIMD width. It is token-major bit-planes:
- per (page, head, token), a key row is 8 chunks of 32 dimensions × 4 planes (one `uint` each,
  128 bytes);
- a value row is 8 chunks × 3 planes (96 bytes);
- bit j of plane p of chunk c is bit p of dimension 32c+j's index.

So one `simd_ballot` writes a plane, and one load reads a chunk. Per 32-token page and layer (4 KV
heads) that is 29,696 bytes, against 66,560 for INT8.

**Store** (`splash_store_turbo_row`, `kernels/common/paged_store_row.h`):
1. The 256 lanes of the row's threadgroup stage the row in threadgroup memory.
2. Simdgroups 0 and 1 each take one 128-value group, held strided (lane l holds elements l, l+32,
   l+64, l+96).
3. Each normalizes it (`simd_sum`) and rotates it: FWHT strides 1-16 by `simd_shuffle_xor`, strides
   32 and 64 in registers, with sign masks.
4. It takes the nearest centroids by a branchless binary search over the midpoints.
5. It computes the corrected norm, writes each plane with one ballot, and writes the fp16 norm.

**Attention** (`splash_paged_attention_turbo_tile`, `kernels/common/paged_attention_tile.h`). Each
page:
1. **Keys.** The page's packed keys (4 KiB) and norms are staged into threadgroup memory with one
   coalesced 16-byte load per thread, reusing the idle score buffer. They are then dequantized into a
   32×256 bf16 tile: simdgroup c owns dimensions 32c..32c+31, and the codebook sits in registers as
   one centroid per lane, read with `simd_shuffle`.
2. **QK.** MPP `matmul2d` runs on that threadgroup tile.
3. **Softmax.** No scales; the norms are already folded in.
4. **Values.** Dequantized into the same tile, dimension-major (lane = token, three plane loads and a
   norm per lane).
5. **PV.** MPP.

Tokens past the visible history dequantize to exact zeros, so stale page bytes can never produce a
NaN. Threadgroup memory: 26,192 B (32 KiB maximum).

## 6. Format `turbo-int4`: TurboQuant mapped onto the GPU's 4-bit matrix hardware

Metal 4's MPP `matmul2d` multiplies `bfloat × int4b_format → float` natively (macOS 26.4+). A page
stored as plain int4 needs no dequantization: the matrix units read it as they read INT8.
`turbo-int4` keeps TurboQuant's essential step, the random rotation, and changes the quantizer to one
that hardware reads.

**Rotation.** A randomized **256-point** Hadamard transform over the whole head vector:
- one rotation per head, so one scale fits the whole vector;
- its own random signs, from Python `random.Random(250419874)`.

**Quantization** (since 2026-10-07, cache revision r2). Each rotated (token, head) row `r` uses
all 16 int4 codes, −8..7:
- the step is `min(2.7·rms(r), max|r|) / 8`, signed so the side with the larger magnitude gets
  the 8th level;
- the codes are `clamp(rint(r / step), −8, 7)`;
- the scale is then refit by least squares, `Σ r·c / Σ c²`, and stored as fp32 in INT8's scale
  layout.

Decoding is unchanged, `code × scale`, so only the store changed and the attention kernels are
untouched. Reconstruction error drops from 0.111 / 0.122 (keys / values, with the earlier
symmetric `max|r|/7`, −7..7) to 0.096 / 0.102.

`kv::quantizerRevision` puts the revision into the persistent-cache namespace, so pages written
by the earlier quantizer are never read. Keys and values are both token-major.

**Encoding.** MPP's int4b is two's complement, with element 2i in the low nibble of byte i. Probed
with `tools/int4_probe` (1,536 of 1,536 elements matched).

**Store** (`splash_store_turbo_int4_row`):
- simdgroup 0 holds the row strided, 8 values per lane;
- it rotates it (FWHT strides 1-16 by shuffles; 32, 64 and 128 in registers);
- it reduces max, −min and Σr² over the simdgroup, quantizes, and reduces Σr·c and Σc² for the scale;
- it packs 8 nibbles per `uint` with three xor-shuffles.

**Attention** (`splash_paged_attention_turbo_int4_tile`):
- the INT8 tile with `tensor<device int4b_format, dextents<int,2>, tensor_inline>(device uchar*, extents)`;
- the page scales are applied in the softmax exactly as INT8's;
- PV takes the token-major values untransposed;
- threadgroup memory: 9,808 B, the same as INT8.

**Page size.** Per 32-token page and layer: 33,792 bytes. The full 16-layer page is 528 KiB.

## 7. Verification

**Kernel tests.** They run the real kernels through runtime-compiled harnesses and compare with an
independent NumPy implementation. All pass.

| Test | Result |
|---|---|
| `turbo_kernel_test.py`: turbo store, then the tile's dequantizers, on 4 heads × 32 tokens | 0 of 65,536 elements differ. Reconstruction error: keys 0.108, values 0.182 (TurboQuant's theory). An all-zero row included. |
| `turbo_int4_test.py`: turbo-int4 store | 0 of 16,384 bytes differ for keys and for values; scales within 3e-7. Reconstruction error: keys 0.096, values 0.102 (0.111 / 0.122 with the earlier absmax / 7) |
| Query rotations (128 and 256) | Exact; dot products kept within 3e-6 |
| `turbo_attention_test.py`: store, rotation, verify split, reduce, over 1-10 pages, 1-3 splits, shuffled pages | GPU matches the reference at cosine ≥ 0.9999 for both formats |
| The same test, TurboQuant vs exact attention, random data | turbo: mean cosine 0.969 (peaked row 0.979-0.986). turbo-int4: 0.986 at 85 tokens and 0.984 at 300, minimum 0.959 / 0.951, peaked row 0.994-0.996. The earlier absmax / 7 gave 0.978 / 0.977, minimum 0.935 / 0.903, peaked 0.992-0.993 |

With a full-head rotation, uniform 4-bit keys with a clipped least-squares scale are more accurate
than 4-bit Lloyd-Max (0.096 vs 0.108). Uniform 4-bit values are far better than 3-bit Lloyd-Max
(0.102 vs 0.182). That is why `turbo-int4` is both faster and more accurate than `turbo`, at 10% less
context. `make -C splash/tools test` runs all of these (8 cases).

**End to end** (`tools/kvbench.py`) uses the real model against each format and runs four checks:

| Check | What it does |
|---|---|
| Speed | 400-token generation |
| Needles | A passphrase hidden at 10/50/90% depth of 16K-200K tokens of C++ source, with prefill and decode rates |
| Judgments | `/v1/judgments` answer distributions. Splash has no logprobs or echo, so this is the only output distribution it exposes. **Set v2** (2026-10-07) uses 23 questions over a context made of a slice of each of the 8 source files, with 10 planted codes: planted facts (the other codes are among the distractors), function to file, file order, counts. Each question has a true answer, so every format also gets an accuracy, besides the KL against another format. Set v1 asked 6 or 24 questions over the first third of `llama-graph.cpp` |
| Greedy | Temperature-0 answers to six prompts, with and without 24K tokens of context, compared word by word |

Section 8 has the results.

## 8. Measurements

**Attention kernels**, `tools/attn_perf`: one layer on random pages, a quiet GPU, ms. "Prefill" is one
2,048-row chunk after H tokens of history; "verify" is one 8-row decode step.

| H | int8 prefill / verify | turbo | turbo-int4 |
|---|---|---|---|
| 4,096 | 16.3 / 0.117 | 38.2 / 0.175 | **15.9 / 0.106** |
| 16,384 | 54.0 / 0.330 | 127.9 / 0.530 | **53.2 / 0.289** |
| 57,344 | 196.6 / 0.997 | 460.9 / 1.783 | **184.2 / 0.892** |
| 131,072 | 457.6 / 2.098 | 1,030 / 3.947 | **433.3 / 1.844** |
| 196,608 | 689.1 / 3.029 | 1,549 / 5.937 | **656.0 / 2.715** |

**Where turbo's time goes.** Kernel variants at H = 16K, `tools/perf_variants.py`:

| Variant | ms |
|---|---:|
| No dequantization at all | 48.5 (as fast as INT8: the threadgroup tile itself costs nothing) |
| Values only | 78.8 |
| Keys only | 111.9 |
| Full, keys read from device memory | 138 |
| Full, keys staged through threadgroup memory (shipped) | 127.5 |

Tried and rejected, measured each time:
- fp16 instead of bf16 tiles: no gain;
- a shuffle broadcast of the key words: 195 ms;
- prefetching the key words into registers: spills, 350 ms;
- unrolling the key loop: no gain;
- removing the codebook lookup or the bit extraction: −2-6% each.

The remaining cost is the decode itself. In prefill it repeats for every 8-row query tile, about
256 times per page per 2,048-row chunk. That is inherent to decoding non-uniform codes inside
attention, and it is why `turbo-int4` exists.

**Capacity** (`/status`, `--language-only`, 24 GB SSD tier):

| Format | Context |
|---|---:|
| int8 | 94,201 |
| turbo-int4 | 192,505 |
| turbo | 212,985 |

**End to end** on macOS 27, with `--persistent-cache` and a nonce before every needle prompt (so no
cache can serve it). Results are in `logs/kvbench/*.json`, the suites in `logs/kv-suite*.log`. "New
quantizer" is section 6's turbo-int4 store of 2026-10-07; the other turbo-int4 rows used absmax / 7.

| | int8 | turbo-int4 | turbo |
|---|---|---|---|
| 400-token decode (tok/s) | 64.3 / 64.4 | 60.1 / 56.3 (draft acceptance 0.61) | 68.9 / 66.7 (acceptance 0.70) |
| Needles 16K (prefill tok/s) | 3/3 (348 / 350 / 326) | 3/3 (394 / 395 / 379); new quantizer 3/3 (414 / 398 / 410) | 3/3 (370 / 368 / 371) |
| Needles 58K | 3/3 (343 / 340 / 338) | 3/3 (318 / 325 / 332); new quantizer 3/3 (338 / 340 / 343) | 3/3 (258 / 252 / 256) |
| Needles 82K | 3/3 (307 / 295 / 266) | 3/3 (275 / 294 / 312; macOS 26) | 3/3 (218 / 213 / 215) |
| Needles 139K and 180K, at 50% | beyond its context | 2/2 (255, 214) | not run |
| Judgments, set v1, 24 questions: KL vs int8 | — | mean 0.015, same top answer 21/24 (new quantizer) | mean 0.032, same top answer 23/24 |
| Greedy outputs identical to int8 | — | 6/12; new quantizer 8/12 | 6/12 |

How to read it:
- **Prefill varies about ±12% between runs** on this Mac with the same flags. INT8 at 58K and 10%
  depth ran at 271, 303 and 343 tok/s on three runs. That spread is larger than any difference
  between INT8 and turbo-int4, which are level end to end. The kernel table above is the clean
  comparison: turbo-int4 attention is 2-10% faster than INT8's. turbo is 20-30% slower than both at
  58-82K tokens.
- **Decode tok/s differences between formats are DFlash2's draft acceptance**, which depends on the
  generated text. `/status` deltas show the same 86-94 ms per decode cycle for every format.
- **Judgments.** Each set v1 run used its own nonce, so part of each KL is the prompt perturbation,
  and there is no INT8-vs-INT8 floor yet. The three top-answer flips of turbo-int4 are on questions
  where INT8 itself was at 0.47-0.65. `splash/tools/next_gpu_runs.sh` runs the clean comparison:
  set v2 at 60K tokens, byte-identical prompts, INT8 twice, the old and new turbo-int4 quantizers,
  and turbo.
- **Greedy agreement** has no noise floor either. Most divergences are equally good rewordings at
  near-tie tokens.
- **The earlier "TurboQuant prefill gap"** (macOS 26) compared INT8 without `--persistent-cache`
  (406/377/378 at 16K, 326/326/330 at 58K, 298/297/298 at 82K tok/s) against turbo runs with it,
  and fell within this run-to-run spread. It was not a cost of the KV format.

## 9. Running it

```bash
./scripts/serve-splash.sh
```

That serves the source build with `SPLASH_KV_FORMAT` from `config/stack.env`, a 24 GB SSD prefix
cache and `--persistent-cache`; it falls back to Homebrew Splash (INT8) when the source tree is
absent. `turbo-int4` is the default. Other formats:

```bash
SPLASH_KV_FORMAT=turbo ./scripts/serve-splash.sh   # maximum context, 2.2-2.4x slower attention
SPLASH_KV_FORMAT=int8 ./scripts/serve-splash.sh    # accuracy first, ~94K context
```

dsh follows the server's context: `scripts/dsh-env.sh` reads `/status` `maximum_context_tokens` and
writes `dsh-home/splash.generated.cordis.patch.yml` with `contextWindow` = limit − 16,384 (reply
budget) − 4,096. For `turbo-int4` that is 172,025.

The tests and benchmarks (they need a GPU that is otherwise quiet):

```bash
make -C splash/tools test          # stores, rotations and attention at 1/3/10 pages, both formats: 8 PASS lines
```

```bash
make -C splash/tools perf H=57344  # attention ms per layer: int8, turbo, turbo-int4
```

```bash
python3 splash/tools/kvbench.py run --label mine --needle-tokens 16000 60000 && python3 splash/tools/kvbench.py compare int8 mine
```

## 10. Files

| Path | What |
|---|---|
| `splash/splash-src/` | Splash 1.2.1 with the changes, branch `turboquant-kv` |
| `patches/0003-splash-turboquant-kv-and-runtime-metal.patch` | All Splash changes: 20 files, +1,712/−77 |
| `splash-src/runtime/metal/kernels/common/turbo_kv.h` | Codebooks, signs, both rotations, page addressing |
| `splash-src/runtime/metal/kernels/common/paged_store_row.h` | The turbo and turbo-int4 store rows |
| `splash-src/runtime/metal/kernels/common/paged_attention_tile.h` | Both turbo tiles, the dequantizers, the rotating reduce |
| `splash-src/runtime/metal/kernels/{decode,prefill}/paged_attention*.metal` | Kernel entry points |
| `splash-src/runtime/metal/kernels/shared/turbo_rotate.metal` | Query rotations |
| `splash-src/runtime/ops/{PagedKv.hpp,PagedAttention.cpp,PageStorage.mm}`, `metal/abi/{KvExtent.h,PagedAttention.h}` | Format, geometry, dispatch |
| `splash-src/runtime/metal/MetalBackend.mm`, `Makefile`, `dev/tools/flatten_metal.py` | Building without Xcode |
| `splash/tools/` | Tests (`make test`), harnesses, benchmarks, `next_gpu_runs.sh` (`HANDOFF.md` section 3) |
| `splash/staged/` | The 2026-10-07 quantizer change as applied (`apply.sh`) and the files it replaced (`orig/`) |
| `splash/IMPROVEMENT_PLAN.md` | Reviewed, ranked improvement plan with success criteria and its status |
| `scripts/serve-splash.sh`, `scripts/dsh-env.sh`, `scripts/doctor.sh`, `config/stack.env` | Workspace integration |

## 11. Limitations and next steps

**Limitations**
- **Not upstream.** A newer Splash needs the patch rebased.
- **Changing what a format stores.** Bump that format's `kv::quantizerRevision` (`PagedKv.hpp`)
  for any change to its store math or layout. That invalidates only that format's persistent cache.
  `kPersistentCacheFormat` (`RuntimeResources.mm`) invalidates every format's.
- **`turbo` has no QJL residual.** That matches the fork's default turbo4.
- **`turbo-int4` swaps the codebook.** It uses uniform 16-level codes with a clipped least-squares
  scale instead of Lloyd-Max.
- **Untested variants.** The 35B layout (`kv2_g8`) variants compile but were not run.

**Next steps.** `splash/IMPROVEMENT_PLAN.md` ranks them, with success criteria. What remains:
- **The GPU runs in `splash/tools/next_gpu_runs.sh`:**
  - the clean quality A/B of the new turbo-int4 quantizer;
  - the format × `--persistent-cache` speed test.
- **A Lloyd-Max key codebook for `turbo`**, at the same bytes and speed.
- **Value centering in the last attention layer**, gated on the A/B.

# TurboQuant KV cache for Splash: plan and progress

Goal: a `--kv-format turbo` for Splash, with K at 4 bits and V at 3 bits (TurboQuant: WHT rotation,
Lloyd-Max centroids, norm correction). That is ~2.2x less KV memory than INT8, so a longer context on
24 GB.

## State (2026-10-05, before the reboot)

- Source: `splash/splash-src`, a clone of github.com/incoai/splash, branch `turboquant-kv` off tag
  `1.2.1`. The changes are uncommitted.
- **Building without Xcode works.** This Mac has only the Command Line Tools: no `xcrun metal`.
  - The Makefile auto-detects this (`RUNTIME_METAL`) and makes `build/splash.metallib` a directory of
    flattened kernel sources (`dev/tools/flatten_metal.py`).
  - `MetalBackend.mm` (`loadRuntimeSources`) compiles them at startup with the system's runtime Metal
    compiler (Metal 4.0) and resolves kernels by name.
  - All 30 kernel files compile at runtime (about 11 s cold, cached after). `make -j10 all` builds
    the engine in about 9 s with upstream's `-Werror`.
- Python: `splash-src/.venv` is a venv on Homebrew Splash's Python, with the same pinned
  requirements and no downloads. `.venv/.requirements-installed` holds the requirements hash, so
  `make` skips pip.
- **Blocker found:** 1.2.x loads the weights into memory at every start (1.1.0 memory-mapped a
  prepared copy). The governor admits that copy only within host free memory minus a 2 GiB reserve
  and a 1 GiB margin, about 17.4 GB available at start. With desktop apps open (~13 GB available) it
  failed: `unable to admit model arenas: host memory reserve protected`. The user is rebooting to
  free memory. If it still fails, the fallback is basing this work on tag 1.1.0, which uses mapped
  weights.

## Progress (2026-10-05, after the reboot)

- The source build serves with INT8 (94,201 tokens of context) once enough memory is free.
- TurboQuant is implemented on branch `turboquant-kv` and builds with `-Werror`:
  - `kv::Format::Turbo`, with per-tensor page geometry: K 16,384 B, V 12,288 B, norms 512 B each
    per layer page, against 66,560 B for INT8;
  - `kernels/common/turbo_kv.h`: centroids, sign masks, strided FWHT, bit-plane addressing;
  - the turbo store row in `paged_store_row.h` (tag `SplashTurboKv`);
  - the turbo tile, the dequantizers and the rotated reduce in `paged_attention_tile.h`;
  - the decode and prefill entries;
  - `shared/turbo_rotate.metal`;
  - in `PagedAttention.cpp`, the kernel names plus a query rotation before each split;
  - `main.mm`, `Status.cpp` and `serve_options.py`: `--kv-format turbo`.
- Kernel tests, all passing (`tools/turbo_kernel_test.py`, `tools/turbo_attention_test.py`):
  - store and dequantization match an independent NumPy TurboQuant bit for bit (0 of 65,536
    elements differ);
  - reconstruction error: 0.108 for keys, 0.182 for values (the theory);
  - verify attention (store, rotation, split, reduce) matches the reference at cosine >= 0.9999,
    with 1-10 pages, 1-3 splits and shuffled page order.
- Benchmark tool: `tools/kvbench.py run --label X` / `compare A B`. INT8 baseline:
  `logs/kvbench/int8.json`.
- To do: the engine run with `--kv-format turbo`, capacity and speed, the quality comparison,
  wiring into `serve-splash.sh` and dsh, the README, and the patch file.

## Smoke test after the reboot

```bash
cd /Users/abdul.rakib/agentMania/dsh-qwen-jev/splash/splash-src
./splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS --language-only \
  --served-model-name qwen3.8-27b-local --port 8000 --no-webui --max-cache-disk 24G
```

INT8 baseline from that run: KV page = 32 tokens = 1,064,960 bytes; 94,208 tokens of context.

## Design (Splash's KV structure, as read)

- `runtime/ops/PagedKv.hpp`: `kv::Format {Int8, BFloat16}` and `Layout`. A page is 32 tokens. Each
  layer's region of an extent is `[K data][K scales][V data][V scales]`, 64 KiB-aligned.
  `abi/KvExtent.h` (`splash_kv_offset`) assumes K and V share one data size and one scale size.
- Store: `kernels/common/paged_store_row.h`, one 256-lane threadgroup per (token, head, tensor), one
  lane per dimension. INT8 does an absmax reduction.
- Attention: `kernels/common/paged_attention_tile.h`. Prefill and verify share it, and verify is the
  decode path because DFlash2 verifies 8 rows. MPP `matmul2d` reads K and V straight from device
  pages; the INT8 scales are applied per token inside the softmax. The tile is 8 rows x 6 query heads
  = 48 fused rows and uses 9,808 B of threadgroup memory out of 32 KiB.
- TurboQuant per 128-value group: rotate the normalized vector (signs1, FWHT, signs2; the fork's
  `turbo-wht.h`), snap each coordinate to a centroid (3-bit: 8, 4-bit: 16 levels), and store
  norm = ||x|| / ||centroids||. Dequantization is `centroid[idx] * norm`, in the rotated domain.
- Plan:
  1. Add `Format::Turbo`; give K and V separate data and scale sizes (K 4-bit, V 3-bit, fp16 norms
     per 128-group).
  2. Pack both token-major as bit-planes: per 32 dimensions, P x uint32 written with `simd_ballot`,
     no atomics. Dimension-major V would need atomic bit updates.
  3. Store kernel: norm, FWHT across lanes (shuffles plus threadgroup memory), centroids, ballots.
  4. Attention tile: per page, dequantize K into a 16 KiB threadgroup bf16 tile, run QK (no scales);
     after the softmax, dequantize V into the same tile (dimension-major), run PV.
  5. Rotate Q forward before attention and the output back after it: small in-place kernels on the
     query staging buffer and the attention output.
  6. Host side: kernel-name selection in `ops/PagedAttention.cpp`; the memory planner and disk tier
     take their sizes from `Layout`; the CLI (`main.mm`) and Python (`server/serve_options.py`) get
     `turbo`; `Status.cpp` gets the format names.
- Validate:
  - output agreement against INT8 and BF16 (greedy generation, logprobs if exposed);
  - long-context needle retrieval;
  - context capacity from `/status`;
  - prefill and decode tok/s at short and long contexts;
  - the agent demo.

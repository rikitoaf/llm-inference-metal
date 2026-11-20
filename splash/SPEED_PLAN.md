# Splash speed plan (2026-10-15, measured the same day)

> **Current state (2026-11-09): read [ANALYSIS.md](ANALYSIS.md) first.** It is the synthesis of everything below
> (cycle anatomy, the long-context cost model, the full experiment ledger, the ranked open levers and a playbook for
> adapting new techniques). The sections in this file are the historical plans, kept as written. Since 10-21 these
> landed in production: 16-node tree (+23.5%), draft-head shortlist (+2.6%), E3 fp16 grid tables (−2.0% cycle),
> K-split tier (−0.9% cycle), copy extension (+16.7% tokens/cycle on session prompts, +4% in the real eval), runaway
> guard (eval 45% → 85%). New finding (ANALYSIS.md §5.2): long-context verify attention is compute-bound, not
> bandwidth-bound, so the next long-context lever is fewer verify rows per cycle (E4), not a kernel that reads the
> KV once.

How to make Qwen3.8-27B (GGUF UD-IQ3_XXS, DFlash2, turbo-int4 KV) faster on this 24 GB M5 Pro (16 GPU cores) for
the coding agent. The plan comes from a CPU-only investigation: five analysts read the Splash source, the GGUF
header and our benchmark data, with no GPU and no server. Their full reports are in the session's workflow
journals. Every gain below is an estimate until `splash/tools/speed_gpu_runs.sh` measures it.

## Where the time goes

**A decode cycle** is one serial Metal command of about 700 dispatches. It runs the DFlash2 draft (5 layers),
the 64-layer verify of 8 rows, sampling and acceptance, and the state commits. It streams 12.64 GB:

| Part | Bytes |
|---|---:|
| Verify (target GGUF) | 10.66 GB |
| Draft weights | 1.27 GB |
| The draft's second pass over the Q4_K output head | 0.72 GB |

At 307 GB/s that is a 41 ms floor. The measured cycle is 86-94 ms, an effective 134-147 GB/s (44-48%).
- **The GGUF projections run at about half the bandwidth.** M5 runs all of them on the "staged" tile: dequantize
  32 weights into threadgroup memory, then an MPP matmul.
  - Weight loads are prefetched only one step ahead, and the per-block scales are not prefetched at all.
  - The IQ grid formats look up constant-memory tables with a different index on every lane.
  - The 8 verify rows are padded to 16.
- **Bytes by format:** IQ3_S 3.07 GB, IQ3_XXS 2.99, IQ4_XS 1.44, Q4_K 0.94 (0.72 of it the output head, read
  twice per cycle), IQ2_* 1.60.
- **Splash's own Q4_K reference** reaches 171 GB/s on this chip, so 170-230 GB/s is a realistic target.
- **Host work between cycles** is about 2-3 ms. The GPU idles while the next cycle is encoded, because only
  one command is in flight.

**Tokens per cycle** is the other half of decode speed: 5.3 on new code, 2 on prose, at temperature 0.
- dsh sends no temperature, so agent requests sample at Splash's default of 1.0.
- A simulation of Splash's exact accept rule says that costs about 23% of the tokens per cycle on code.

**Prefill** is close to compute-bound: about 22 TFLOP/s on the linear layers, 84-90% of an assumed ~26 TFLOPS
peak. It matches Splash's own GGUF figure.
- **Attention:** 7% of a cold 16K prefill, 24% at 58K, 49% at 180K.
- **`--persistent-cache`:** its write-behind cannot stall prefill. It writes from an IO thread, 10 s after
  publication.
- **Agent steps:** every agent step re-prefills the previous turn's generated tokens. Splash keeps a state only
  at the prompt's replay boundary.

**Long context:** the cycle grows only through the 16 verify-attention layers, by about +14 ms at 58K and +29 ms
at 131K. The low decode tok/s we saw on needles is largely an artifact: a 9-token answer gives 2-4 cycles.

## Measured on 2026-10-15

The run used AC power and High Power mode, `splash/tools/speed_gpu_runs.sh` plus a follow-up. Results are in
`logs/speed/` and `logs/kvbench/spd-*.json`.

**1. The decode cycle is GPU-bound in the projection kernels.** This is the clean decode-profile, run before
Defender started.
- One lane: 87.71 ms of GPU time against 88.36 ms wall, so host work is 0.65 ms.
- Replaying the 689 dispatches one at a time gives 87.03 ms, so per-dispatch bubbles are negligible.
- The GGUF projections take about 74 ms of the 87.7 (85%).
- Two lanes cost 93.7 ms, only +7% for twice the rows.
- **Dropped:** host-side work (sliced submission, cross-cycle pipelining) can't gain more than about 0.6 ms.

**2. The grid formats are bound by dequantization**, measured with `gguf-projection-benchmark` at one lane:

| Shape | MB | Rate at the policy's split | Best split |
|---|---:|---:|---:|
| Q4_K 17408x5120 (reference) | 50.1 | 223 GB/s | same |
| Q4_K output head 248320x5120 | 715 | 200 GB/s | same |
| IQ3_S 17408x5120 | 45.3 | 165 GB/s | same |
| IQ3_S down 5120x17408 | 45.3 | 148 GB/s (S2) | 159 GB/s (S4) |
| IQ3_XXS 17408x5120 | 34.1 | 138 GB/s | same |
| IQ4_XS down 5120x17408 | 47.3 | 135 GB/s (S2) | 149 GB/s (S8) |
| IQ3_XXS down 5120x17408 | 34.1 | 114 GB/s (S2) | 125 GB/s (S4) |
| IQ3_XXS QKV 10240x5120 | 20.1 | 92 GB/s (S1) | 99 GB/s (S2) |
| IQ2_S 17408x5120 | 28.5 | 90 GB/s | same |

- **IQ3_XXS is not limited by bandwidth:** it moves 32% fewer bytes than Q4_K but takes longer, and a second
  lane does not slow it.
- **The decode cost of the grid formats is the lever.** Running IQ3_XXS, IQ3_S and IQ2_* at Q4_K's rate would save
  roughly 12-15 ms per cycle.
- **Splash's split policy picks a 7-10% slower split** for the 5120-output projections and the QKV. This is a
  single run, so it needs confirming.

**3. Speculation knobs (tokens per cycle).** These are counts per cycle, so Defender cannot skew them.

| Setting | Tokens/cycle | Change |
|---|---:|---:|
| T=0 | 5.71 | |
| T=1.0, what dsh sends today | 5.08 | baseline |
| T=0.6 | 5.38 | +6% |
| T=1.0, draft scale 0.5 | 5.05 | 0% |
| T=1.0, draft scale 0.35 (exact) | 5.30 | +4% |
| T=0.6, draft scale 0.35 | 5.58 | +10% |

The simulation had predicted up to +17% each; the real gains are small and near the noise of 6 requests. Draft
scale 0.35 is worth adopting once more samples confirm it. T=0.6 needs an eval for quality first.

**4. The environment matters more than the knobs.**
- Microsoft Defender's real-time scanning ran at 150-200% CPU plus its system extension during most of the
  sweep. It slowed decode cycles from 88 ms to 108-129 ms, 20-45%, through memory-bandwidth contention.
- **Pattern:** the first request after each server start ran at the normal ~88 ms, and later ones slowed.
- **With `--persistent-cache` off,** later requests were 101-105 ms, so the cache's write-behind triggers part of
  the slowdown. Defender kept scanning even with no server running.
- **Consequence:** real agent sessions are probably slowed the same way. The likely triggers are the persistent
  cache's multi-GB writes and fresh model or build files.
- **Fix (yours or IT's, not mine):** Defender exclusions for `~/Library/Caches/Splash`, `~/.cache/huggingface`
  and `splash/splash-src/build`.
- **Cache growth:** 101 GB of stale persistent-cache namespaces had accumulated. They were moved to the Trash, and
  `serve-splash.sh` now caps the cache (below).

## Status (2026-10-21)

Measured on a quiet machine; the details, numbers and decisions are in [SPEED_LOG.md](SPEED_LOG.md).
- **Applied:** `prefill-arena-liveness`: maximum context 192,505 → 208,889 tokens, with bit-identical output.
- **Applied:** `m16-one-lane`, a 16-row staged tile at one lane. End to end, decode goes from 66.9 to 68.8 tok/s
  (+2.9%; one cycle takes 85.8 ms → 83.3 ms) with bit-identical output. It is in `splash-src` and `patches/0003`.
- **Staged, not applied:** `split-policy` (B1 −1.4%, B4 +4.7%).
- **Rejected:**
  - the grid tables with sign XOR (`grid`, `grid-v2`, `grid-v3`): +11 to +15% per cycle;
  - prefetch depth 2: neutral.
- **What happened to the old plan's steps:**
  - Steps 1-4 are done (step 4 is not applied).
  - Steps 5-8 remain. They are below, with the new strategies.

## Plan, 2026-11-02 (advanced-technique research, skeptic-adjusted)

Full report: [RESEARCH-ADVANCED-2026-11-02.md](RESEARCH-ADVANCED-2026-11-02.md) (raw results: `*.final-result.json`,
`*.run2-result.json`, run 1: `*.run1.md`). Corrected premises: host gap ~0.63 ms (not 2-3); verify matmul already at
the FP16 matrix-unit peak (25 TFLOP/s), so only the ~19-23 ms of dequant per cycle can shrink (ceiling ~+10% decode)
unless int8/e4m3 matmul is faster on M5; SPLASH_DRAFT_TEMPERATURE_SCALE is inert in tree mode (sampling.metal:1480);
dsh's reasoning setting never reaches Splash (pi-ai sends no reasoning_effort), so every request runs at xhigh.

| item | skeptic-adjusted gain | exact | first step |
|---|---|---|---|
| Runaway-reasoning guard (stop loops at ~10-12K think tokens / line repetition) | 4 runaway steps = 11.4% of corpus agent time | approx. (identical below budget) | CPU: replay transcripts through a detector, count false positives |
| Reasoning effort xhigh → medium | +5-12% agent likely, unmeasured (medium only drops one sentence) | approx. | eval t1-t6 × ≥3 reps |
| E3 lean IQ grid dequant (bit-identical) → E5 int-operand stage | +5-6% decode (E3), +6-10% (E5); not skeptic-checked | exact / exact-math | GPU microbench gguf-projection 17408×5120 < 0.21 ms |
| Matrix-unit probe: int8 / e4m3 matmul2d, relaxed_precision, MMA‖ALU overlap | gates +10-40% decode; low prior | probe | 1-day GPU microbench |
| Park lane | +1.5-4% agent | pending oracle | GPU oracle test |
| Copy branches in the 16-node heap | +1-3% agent (sim +9% was in-sample, copy-heavy, biased mean) | exact | T=0.6 dump on real dsh steps |
| Per-cycle verify rows at long context | ~+2% agent, +5-9% at 96-128K | exact | decode-profile with tile 2 early exit |
| Staged K-split retune | +0.7-1.3% decode | exact-math | split sweep + A-B-B-A |

## Plan, 2026-10-27 (an investigate-and-critique workflow; corrected estimates)

Every item keeps the target's outputs exact (draft-side, bit-identical, or same math in another summation order).

| # | Item | Wall time (corrected estimate) | Status |
|---|---|---|---|
| 0 | dsh tool-result pruner off history (harness config) | −10-14% of long sessions | **applied 2026-10-27** |
| 1 | Keep the finished lane for the next turn (no re-prefill of the previous output) | −3.5% to −6% | building (`splash/dev-park`) |
| 2 | Cache state at the end of system prompt + tools | ~20 s per new session and per compaction | building (`splash/dev-sysstate`) |
| 3 | One 16-row verify-attention tile (the KV is read once, not twice) and bit-exact barrier/rescale trims | 0 to −2.5%, more at long context | building (`splash/dev-attn`, with turbo-int4 decode-profile) |
| 4 | Draft head shortlist (ids < 98,304 plus the special tokens) | −0.5% to −1% | building (`splash/dev-shortlist`) |
| 5 | Copy-from-context extension past draft depth 7 | −1.5% to −3% | later; needs 8 more uniforms per lane for exact T>0 and a fix to the accepted > 7 check |
| 6 | Skip the second attention tile at long context | −1% to −3%, overlaps item 3 | after item 3 is measured |

The investigation dropped these:
- Draft calibration and online tables: ≤ +0.3%.
- More than 16 rows.
- A second DFlash block.
- A copy chain that replaces tree nodes.
- Lower-bit or offloaded heads.
- An int8 draft.

## Plan for the next session (re-ranked 2026-10-21, speed first)

Where a cycle goes (B1, 2K context): target projections 67.6 ms (80%), draft 5.5 + its head pass 2.7, output head
2.7, GDN 2.8, verify attention 1.6 (9.8 at 32K), host 0.6.
- **Projections:** each one costs the matmul plus the dequantization, which do not overlap. The weight format barely
  matters, and **16 rows cost the same as 8** (see SPEED_LOG.md). So the biggest lever is accepting more tokens per
  cycle, not cheaper weights.

| # | Step | Gain (estimate) | Exact? | Size |
|---|---|---|---|---|
| 1 | **Sampling defaults:** dsh sends no temperature, so Splash samples at T=1.0. Use Qwen's recommended T=0.6 and set `SPLASH_DRAFT_TEMPERATURE_SCALE=0.35`. Measured on code: 5.23 → 5.60 tokens/cycle. | +7% (code); likely more on prose | changes sampling; needs your OK and a quality eval | config |
| 2 | **16-row verify for one lane:** a 15-token chain: DFlash's 7, then a suffix/n-gram continuation from the context (SuffixDecoding-style) or a second draft block. Linear chain, so GDN and attention stay causal. Means decoupling `targetVerifyRows` (8) from the draft rows (Model.hpp) and touching ~33 files that assume 8. | copy/edit turns +30-80%; agent mix +10-25%; cost +4-5 ms/cycle (B2 data) | exact | L (days) |
| 3 | **16-node draft tree from DFlash's top-16 candidates (DDTree):** needs a tree mask in verify attention and a GDN tree scan (WY form, then replay of the accepted path). | prose/reasoning +20-35% tokens/cycle | exact | L |
| 4 | **Faster turbo-int4 verify attention**, after adding `--kv-format turbo-int4` to decode-profile. | −4 ms at 32K, −7-10 at 58K | exact | M |
| 5 | **Fewer instructions per weight in dequantization** (format-generic). | ≤10-20% of projections, uncertain | exact | M |
| 6 | **Prefill:** reuse the previous turn's generated tokens in the prefix cache, then a faster prefill attention (65% of peak). | −1-3 s per agent step at 60K | exact | M |
| 7 | **Defender exclusions** for the model, cache and build paths. | 20-45% whenever it scans | — | yours or IT's |

Dropped after measuring:
- sliced command submission and cross-cycle pipelining (host overhead is 0.65 ms);
- prefetch depth 2-3;
- sign-XOR grid tables;
- verify attention split size;
- 8-row and operand-swapped tiles;
- larger K steps;
- the direct-to-register tile, with or without two operands.

## Draft-tree verify: the implementation plan (started 2026-10-21)

Evidence: SPEED_LOG.md, "what wider speculation would give". A dynamic DDTree built from DFlash's unary + edge
scores gives 3.68 → 3.97 tokens/cycle at 8 nodes and 4.45 at 16 (exact at T=0). A fixed shape learned from the
same data gets only 3.70 at 8 nodes and 4.15 at 16, so the tree is built on the GPU every cycle. Every
tree-consuming kernel reads a per-lane tree buffer: parent, depth and ancestor mask of each verify row.

**Phase T8: a tree of 8 nodes in today's 8 rows. Done 2026-10-21, now the default:** +8.6% tok/s on agent prompts
(SPEED_LOG.md). The steps as built:
1. `draft_select_tree` (sampling.metal), after the DFlash selection: one thread per lane, best-first over
   unary + edge with path probability; writes the node tokens (rows 1-7), parents, depths and ancestor bitmasks.
   Row order is pop order, so a parent's row is below its children's.
2. Verify input: row r's token is node r; RoPE position = committed + depth (the target RoPE tables are rebuilt
   from a GPU-written positions buffer after the tree is built).
3. Verify attention: row r sees the committed history plus the rows of its ancestor mask (in place of the causal
   mask among verify rows) in `paged_attention_tile.h`.
4. GDN: the prologue's short convolution takes its previous inputs from the ancestors (parent, grandparent, ...,
   then the conv carry), not from rows r-1..r-3. The scan keeps a state per node and starts node r from its
   parent's.
5. Acceptance: walk from the root; at node r, x = the target's draw at row r (argmax at T=0; at T>0 a plain draw
   from p, q = 0); descend to the child whose token is x, else x is the correction. This is exact sampling.
6. Commit: the accepted path's rows are gathered (a path list). KV rows of path nodes whose row != depth are
   copied to slot committed + depth. `verify_gdn_commit` replays the path rows. The draft context commits the
   path's captured hidden rows. The host reads the path tokens.
7. Tests: T=0 outputs identical to the chain on greedy prompts (a tree only adds candidates, so the greedy
   sequence must match), the GDN state after commit equal to a sequential replay, and tokens/cycle against
   `tree_potential.py`.

**Phase T16: done 2026-10-21, the default.** 16 verify rows for one lane, 16-node tree: agent prompts 51.5 tok/s
(+23.5% over the chain), sampled code +23.6% (SPEED_LOG.md). (the map in the session notes, M0-M1: a separate tile-row constant,
16-row verify buffers, two 8-row virtual tiles in verify attention, the GDN loops over 16 rows, decode width capped
at 2), then the 16-node tree. +15% total.

## Not chosen (for now)

| Idea | Why not |
|---|---|
| n-gram / prompt-lookup drafting inside the 8-row verify | DFlash2 already reaches ~7.5 tokens per cycle on copy text, so it is worth only +5-10% on an agent mix. Large wins need a verify wider than 8 rows, a large kernel project. |
| Loading the model's own MTP layer as a second drafter | 351 MB, ~5 ms per step, and its acceptance chain (0.905, 0.785) loses to DFlash2 at every depth. |
| Pruning the draft's output head to ~96K tokens | Saves 2.2-2.9 ms gross but breaks even at a 3% acceptance loss. Large change, uncertain payoff. |
| Page-selective (Quest-style) verify attention | +11% at 58K and +27% at 131K, but approximate attention costs needle accuracy. Revisit if long-context decode becomes the bottleneck. |
| int8 Q·K in verify or prefill attention | Needs a probe of the M5's int8 matmul rate first; 3-10% at long context. |

## Environment

- **Power:** on battery this Mac is in Low Power mode (`pmset`: powermode 1 on battery, 2 on AC), which throttles
  the GPU. Use AC for agent work and for every measurement. `speed_gpu_runs.sh` refuses to run on battery.
- **Memory:** Splash needs ~16 GiB reclaimable to start (`python3 splash/tools/hostavail.py`).
- **Defender:** its real-time scans cost 20-45% of decode speed while they run (see above).
- **Persistent cache size:** `SPLASH_CACHE_TOTAL_DISK` (default 48G, two namespaces) caps the total.
  `serve-splash.sh` runs `splash/tools/prune_splash_cache.py` before each start: it keeps the most recently used
  namespaces within the budget, deletes the others, and never touches a namespace a running Splash holds.

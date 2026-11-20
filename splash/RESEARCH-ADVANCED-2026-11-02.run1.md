# Advanced speed research — synthesis (2026-11-02)

Five independent research angles merged and judged: (1) decode pipeline and host overhead, (2) tokens per cycle
from the 16-node DFlash2 tree, (3) end-to-end agent wall time from the dsh session logs, (4) external engines and
published speculative decoding mapped onto Splash. A fifth slot repeated angle 4's content in parts, so it is merged
into it. Nothing here ran on the GPU. All gains below are checked against the cycle budget. Read SPEED_PLAN.md and
SPEED_LOG.md for the measured history; nothing below re-proposes a logged failure without a new mechanism.

Status labels:
- **exact**: greedy output is identical and the sampled distribution is unchanged.
- **exact-math**: the math is the same but the summation order differs, so greedy text can flip at near-ties (the same
  class as the 16-node tree scan, which was accepted).
- **approximate**: changes the model's numerics or behaviour. These need the eval.

---

## 1. Time budget (corrected)

### Decode cycle, one lane, 2K context, head shortlist
Source: `logs/speed/ab-1101/profile/profile-head98k-2048.txt`, 84.52 ms GPU.

| Part | ms | Share | Notes |
|---|---:|---:|---|
| GGUF projections (verify, 64 layers) | ~67 | ~79% | ~0.113 ms matmul + ~0.11 ms dequant per projection, serial (SPEED_LOG 10-21 `diag-nodeq`) |
| Draft (5 layers, 1.27 GB) | ~5.5 | ~6.5% | |
| Output head (shortlist) | ~1.5 | ~2% | |
| GDN verify + commit | 2.42 + 1.19 | ~4.3% | the commit re-reads 151 MB of state |
| Small dispatches (norms, attention glue, selection) | ~1.8 | ~2% | `norm_rms_f32` ×129 = 0.97 ms |
| Verify attention (2K) | small | | grows ~+0.35 ms per 1K tokens in the profile (92.9 → 114.8 ms at 64K) |
| Dispatch boundaries in the fused stream | ~1.24 | 1.5% | 725 dispatches × ~1.7 µs |
| **Host gap between cycles** | **0.63-0.67** | **0.76%** | measured over 4 × 2,500 cycles (`logs/speed/ab-1101/logs-*/steps.jsonl`) |

### Corrections to the task premises
- **The host gap is about 0.65 ms, not 2-3 ms.** Angle 1 measured it from `steps.jsonl` (t[i]−t[i−1]−gpu_ms). It
  matches SPEED_PLAN.md:51-57 (87.71 GPU vs 88.36 wall) and stays flat from 2K to 64K. An unconstrained cycle is one
  command (Runtime.mm:2474-2584) with no CPU round trip inside it. The whole cross-cycle pipelining family is capped
  at +0.77%.
- **The verify projections sit on both roofs at once, compute and DRAM.** `diag-nodeq` gives 0.113 ms for
  16×5120×17408. That is 2.85 GFLOP, about 25 TFLOP/s, roughly the FP16 Neural Accelerator peak scaled from Draw
  Things' M5 Max figure to 16 cores. `diag-noload` shows DRAM loads are only ~8%.
- **What the "140 GB/s vs 307" gap really is.** Dequantization runs on the shader ALUs and does not overlap the MMA.
  That gap cannot be closed from the memory side (prefetch depth, scale prefetch). Only two things help:
  - fewer ALU operations per weight, or a cheaper MMA (the int8 path);
  - more accepted tokens per cycle.
- **`SPLASH_DRAFT_TEMPERATURE_SCALE` (0.35) does nothing in tree mode.** Checked in code:
  - DraftSelector.cpp:77 scales `params.temperature`.
  - Only `draft_select_dflash` reads it (sampling.metal:1347).
  - `draft_select_tree` overwrites the tokens and zeroes `q_probs` (sampling.metal:1480).

  The +4%/+5% measured on 10-15/10-21 came from chain mode and from T=0.6. The knob now affects only chain-fallback
  lanes. Document it as chain-only in `config/stack.env`.
- **Real agent cycles are slower than the profile predicts.** df378d27 ran at 131 ms per cycle at a mean context of
  76K, against ~115 ms predicted. The host gap is flat with context, so it is not the cause. Defender, heat or the
  attention slope are suspected and still unexplained.

### Where an agent session's model time goes
Session df378d27 (current build, 30 steps, 4,216 s of model time; angle 3):

| Part | Share |
|---|---:|
| Reasoning decode (89% of output tokens) | **59%** |
| Prefill | 23% |
| ↳ cache breaks (pruner rewrite, now fixed; post-compaction cold start) | 10.8% |
| ↳ re-prefill of the previous step's own output | 6.4% |
| ↳ new content | 5.1% |
| Compaction call | 11% |
| Tool-call argument decode | 5% |
| Answer text decode | 2% |

One runaway reasoning step (16,384 tokens, stopped by `length`) cost 9.8%. In single-turn sessions, reasoning is 75%
of model time.

**How to convert gains.** A decode tok/s gain of x% is worth about 0.6·x% of agent model time. A prefill-only gain
is worth about 0.1-0.2·x% once park and the compaction cache work.

---

## 2. Key evidence by angle

### Angle 1: pipeline and host
- The host gap is 0.65 ms (see above). The angle recommends against two command buffers in flight or a GPU-driven
  loop: ≤+0.77% for ~8 days, and SPEED_PLAN already dropped this.
- **What remains is about 2-2.5 ms per cycle in total:**
  - 129 norm dispatches, 0.97 ms;
  - `verify_gdn_commit` re-reads 151 MB of state, 1.19 ms at 254 GB/s (gdn.metal:164-177, 455-525);
  - ~63 small dispatches that can be merged bit-identically;
  - host graph build and encode, ~0.3-0.45 ms. `prepare()` looks up each pipeline by name string per dispatch
    (MetalBackend.mm:751); `CommandGraph::add` copies a string and a vector per dispatch (CommandGraph.hpp:31-48).

### Angle 2: tokens per cycle
CPU re-simulation over `logs/speed/draft-dump.jsonl` (2,427 greedy cycles). The baseline reproduces tree16 at
**4.450** tokens per cycle (measured 4.46).
- **The depth-7 cap binds in 20% of cycles** (492 of 2,427). It comes from DFlash2 `block_size: 8`, sampling.metal:1662
  and Runtime.mm:1889.
- **Perfect selection** among DFlash's top-16 candidates would give 5.91 tokens per cycle. The candidate set rarely
  limits anything (truth missing at depth 1 in 1.2% of cycles). The cap and the scoring are the limits.
- **A copy extension under the best depth-7 leaf** (4-gram match, up to 7 nodes at depths 8-14) gives **4.777
  (+7.3%)**. Per prompt: tool-diff +16.6%, edit +11%, refactor +11%, code +4%, prose +0.6%.
  - A root copy chain at depth ≤7 is −0.2 to −0.6%, which confirms the "Not chosen" n-gram row.
  - The gain comes from extending past the cap, which the 10-21 `copy_potential.py` model (i.i.d. α) underestimated
    at +0-4%.
- **The 7 `SPLASH_UNIFORM_PROPOSALS` uniforms are unused in tree lanes** (Sampling.h:18-24). They can serve depths
  8-14 with no ABI growth. Scratch and row limits are already sized (Model.hpp:395,398).
- **Other policies were measured and lose:** calibration discount ≤+0.26%; adaptive 24/32 rows ×1.00-1.01;
  rejected-subtree reuse ≤+1%.

### Angle 3: agent wall time
- **Reasoning effort.** Splash maps dsh `high` to `xhigh` (`server/chat_templates.py:92`). The template then adds a
  "think carefully, validate assumptions, consider alternatives" sentence (GGUF chat_template lines 66-67). `medium`
  adds nothing.
- **Runaway reasoning recurs** (32,790 / 16,408 / 16,384 tokens). t2-semver scores 0/13 under xhigh because answers
  hit the cap (eval/results/ab-run1).
- **Old reasoning is about half the context.** `preserve_thinking` is undefined, and dsh sends `reasoning_content`
  back. This drives the long-context cycle cost and triggers compaction.
- **Park lane** (rerun 11-01, `logs/speed/ab-1101/session-b.txt`):
  - 4 adoptions, 0 mismatches;
  - turn-1 time to first token 1.90 → 0.67 s;
  - turn-1 text diverges at character 127 of 643, from summation order, still to be proven by the oracle.
- **Compaction missed the cache** in df378d27 (12,480 of 95,854 tokens hit). The cause was the since-fixed pruner
  rewrite plus disposable 4,096-token checkpoints (Engine.hpp:22-71, 220). Whether it hits now is unverified.
- **Prefill kernels are not a wall-time lever.** About 7-8% of model time once park and compaction are fixed.

### Angle 4: external engines and methods
- **M5 int8 matmul path.** The installed `MPPTensorOpsMatMul2d.h:11-83` supports int8×int8→int32 and
  bfloat×int8→float. Splash's GGUF staged tile stages half only (gguf_staged_tile.h:24,57).
- **Reported int8 speed-ups:**
  - Modular: M5 integer widening MMA 1.7-1.9× the float MMA, 1.5-1.7× net
    (https://max.modular.com/stable/api/mojo/linalg/matmul/gpu/apple/int8_matmul);
  - Draw Things: fused int8 1.61-1.87× FP16 on M5 Max
    (https://releases.drawthings.ai/p/metal-quantized-attention-pulling).
  - Both used intrinsics or large tiles. Rigel (arXiv 2606.12765) found matmul2d on M4 runs on the shader cores.
    The matmul2d int8 rate at a 16×32 tile on this M5 Pro is **unverified**.
- **Every weight format in this GGUF decodes to an exact small integer times a per-32 or per-16 scale**
  (quant_formats.h:257-271). Int8 weight staging is therefore exact. Only int8 activations are approximate.
- **`relaxed_precision` is false on the projections** (gguf_staged_tile.h:24,57, sixth descriptor argument) and true
  on attention.
- **Deeper drafting blocks help.** DFlash b16 vs b8 τ: +15% HumanEval, +21% Math500, +6% MT-Bench
  (arXiv 2602.06036 Table 7).
- **A block-15 drafter is on disk:** `/Users/abdul.rakib/models/dspark/Qwen3.8-27B-DSpark-Q8_0.gguf` (5 layers,
  `dflash.block_size=15`, Markov head). LITHOS.md:193 compares chains (3.73 vs DFlash2 in Splash chain 3.68), so at
  depth ≤7 it is roughly at parity.
- **Not applicable here:** DDTree's best budgets (256-512 nodes on H200). Every extra 16 rows costs ×1.38-1.40.

---

## 3. Ranked table

**Score** = expected agent wall-time gain × probability of success ÷ effort. Exact and exact-math items rank first.
Agent wall gain uses decode ≈ 60% of model time.

| # | Idea | Status | Decode / model-time gain (arithmetic) | Agent wall | P | Days | Score |
|---|---|---|---|---|---|---|---|
| 1 | **Finish park lane** (or publish the final decode state into the prefix cache) | exact-math | Re-prefill of own output is 6.4% (df378d27) to 8.4% (9826300e) of model time, an upper bound; time to first token on long steps 50 s → a few seconds | 4-6% (multi-step sessions) | 0.7 | 2 | 1.7 |
| 2 | **Copy extension past depth 7** under the best depth-7 leaf, GPU 4-gram index, depths 8-14 on the idle proposal uniforms | exact | Simulated +7.3% tokens/cycle (4.450 → 4.777). Agent estimate +5-9%, discounted from angle 2's +8-10% (the 1.66× calibration is unproven and T=0.6 is unmeasured). Cycle cost +0.1-0.3 ms of 85. Net decode +5-8% | 3-5% | 0.7 | 4 | 0.7 |
| 3 | **Compaction cache hit:** check first, then non-disposable step replay states on the SSD tier + `SPLASH_SYSTEM_STATE` | exact | df378d27: ~285 s of compaction prefill + 25-45 s system block of 4,216 s ≈ 7.5%, one compaction per long session. The 10-27 pruner fix may already recover part | 0-7% (long sessions) | 0.5 | 0.5 check + 2 | 1.0 (long sessions only) |
| 4 | **Long-context verify attention:** 16-row tile **with** pages-per-split 16 → 8 (the untested pair) | exact | At 64K the context part is +21.9 ms (profile), +38 ms in real sessions. Cutting 30-45% of it saves 7-15 ms of 115-150 ms, 5-10% decode at 64K+ | ~2% average, 3-7% in long sessions | 0.3 | 1 | 0.6 |
| 5 | **Tree-builder τ tied to the request temperature** (plus retiring the inert draft scale) | exact | Bounded by the T=0 → T=0.6 gap on code (6.67 → 6.40, −4%). The τ sweep at T=0 is sharp (±5%). Realistic 0 to +3% tokens/cycle at T=0.6, which dsh uses | 0-2% | 0.4 | 1 | 0.4 |
| 6 | **Int8 weight stage, bf16 activations** (bfloat×int8→float matmul2d, per-32 scale epilogue; Q4_K head as uint4b); fold in a `relaxed_precision` probe | exact-math | Dequant ~34 ms → 20-26 ms if bf16×int8 runs at the FP16 rate: −8 to −14 ms of 85, +10-20% decode. Capped by bandwidth near 43 ms | 6-12% | 0.3 | 1 probe + 6 | 0.35 |
| 7 | **DSpark block-15 drafter** so the tree can reach depth 15 | exact | At α 0.6-0.8 in deep positions, +0.30-0.67 tokens/cycle (+7-15%) on its own. It overlaps #2 on the same 20% capped cycles; incremental over #2 about +3-7% decode, and it risks lower depth-1..7 acceptance | 2-4% | 0.35 | 7 | 0.15 |
| 8 | **Reasoning effort medium (from xhigh) + an 8-12K reasoning-budget guard** | **approximate** (changes model behaviour) | Reasoning is 59-75% of model time. A 30-50% token cut is 18-30%. The runaway alone was 9.8%. Second order, a shorter context makes cycles 4-6% cheaper | **15-30%** | 0.6 | 2 | 6 (largest lever; gated by eval) |
| 9 | Dispatch hygiene bundle (bit-identical merges, concurrent tail, deferred GDN commit, host encode ahead) | exact | +0.5% + 0.7% + 0.45% ≈ +1.5-1.7% decode | ~1% | 0.6 | 10 | 0.06 |
| 10 | Int8 activations × int8 weights (full int8 MMA) | approximate | Projections 67 → 40-48 ms, cycle 85 → 60-65 ms (+30-40%), capped near 43 ms by bandwidth. Prefill +35-55% | 20-25% | 0.2 | 10 | 0.45 |
| 11 | `preserve_thinking=false` at user-turn boundaries | approximate | Net ~540 s of 4,216 s ≈ 13% in multi-turn sessions; 0 in single-turn | 0-13% | 0.4 | 1.5 | (after #8) |
| 12 | RMS norm folded into the producer and consumer projections | approximate (sum order); can be made exact-math | 0.85 ms of 85 ≈ +1.0% | 0.6% | 0.6 | 3.5 | 0.1 |
| 13 | SuffixDecoding-style multi-candidate continuation (refines #2) | exact | +1% on top of #2 (upper-bound estimate) | 0.6% | 0.4 | 1 | 0.24 (do with #2) |
| 14 | Second DFlash block beyond depth 7 | exact | Break-even needs 1.1 extra accepted tokens per gated cycle; plausible range −2% to +6% | ~1% | 0.2 | 5 | 0.04 |

### Notes on the scoring
- **Composition.** #2 and #7 attack the same capped cycles; do not add their gains. #2 and #5 are independent. #8
  shrinks the context, which reduces the benefit of #4 and #11 and slightly boosts #3. Decode gains from #2, #5 and
  #6 multiply (tokens/cycle × 1/cycle-ms).
- **Why #8 stands apart.** It is the largest wall-time lever by an order of magnitude, but it changes behaviour and is
  approximate, so it sits in its own tier. It is cheap to try, needs no engine change, and xhigh's 0/13 t2-semver
  suggests accuracy could even rise. Run its eval in parallel with the exact work.
- **Why #6 ranks below #2 despite a larger ceiling.** The success probability rests on an unmeasured matmul2d int8
  rate, and earlier dequant micro-optimisations backfired (grid tables with sign XOR: +11-15% slower). The 1-day probe
  decides it, together with #10.

---

## 4. Rejected and duplicate ideas

| Idea | Source | Verdict | Reason |
|---|---|---|---|
| Two command buffers in flight / GPU-driven K-cycle loop / ICBs for the gap | angle 1 | rejected | The gap is 0.65 ms; ceiling +0.77% for ~8 days. Already dropped (SPEED_PLAN.md:57). Revisit only if the host gap passes ~2 ms. |
| Overlapping cycle N+1's draft with cycle N's verify | angle 1 | rejected | Strict dependency: the draft needs N's correction token and accepted hidden rows. Only the commit tail (≤0.2-0.3 ms) could overlap. |
| Root copy chain / n-gram inside depth ≤7 | angles 2, 4 | duplicate of a measured failure | Re-simulated at −0.2 to −0.6%. Same as the "Not chosen" n-gram row. The extension (#2) is the new mechanism that makes copying pay. |
| Global cross-session suffix tree as a tree branch | angle 4 | merged into #2 and #13 | Same insertion point as the copy extension; cross-session repeats are unmeasured. Measure on CPU after #2 by extending the corpus. |
| Draft calibration / GRAFT-style learned edge scores | angles 2, 4 | rejected | Measured ≤+0.26-0.3% (SPEED_PLAN and the 11-02 re-simulation). |
| Adaptive tree budget 24/32 rows (CaDDTree, DDTree large budgets) | angles 2, 4 | rejected | Rows cost ×1.38-1.40 past 16; oracle ×1.007-1.010, rule ×1.000. |
| Rejected-subtree reuse | angle 2 | rejected | Truth recovered in 59 of 1,942 failures, ≤+1%. |
| Skip the draft and verify a 15-token copy chain when the match is long | angle 2 | rejected | ×1.024 alone, ×1.052 combined, both below the extension alone. |
| Tuning `SPLASH_DRAFT_TEMPERATURE_SCALE` | angle 2 | rejected (dead knob) | Inert in tree mode (sampling.metal:1480). Replaced by #5. |
| `SPLASH_VERIFY_TILE=16` alone / 8x2 | angle 3 | already rejected | +9.6% / +12.4% at 64K. Only the new pairing with pages-per-split 8 (#4) is a new mechanism. |
| Prefill kernel speed-ups for the agent | angle 3 | low priority | ~7-8% of model time after park and compaction; 10% faster prefill is under 1% of wall time. |
| MTP drafter, Quest attention, prefetch depth 2, direct/swap/m8/step64, grid sign-XOR tables | task list | not re-proposed | Measured failures; no new mechanism offered. |
| Self-conditioned second DFlash block | angle 2 | kept at low rank (#14) | No acceptance data; needs a break-even dump first and only after #2. |

---

## 5. First experiments (cheapest test that confirms or kills each top idea)

| # | Experiment | GPU? | Kill / confirm criterion |
|---|---|---|---|
| 1 Park lane | Run the oracle in `splash/dev-park/NOTES-dev-park.md` item 3 (`model_runtime_oracle_test.mm`: decode 40, park, adopt with a suffix starting mid-page, vs cold begin). Report GDN/KV relative error and the top-2 logit margin at the first differing token, plus 12-prompt greedy agreement. | yes, ~20 min, ask first | Kill: errors above the gdn-decode tolerances (2e-5 rows 0-7, 3e-4 rows 8-15) or divergence at large margins. Confirm: agreement ≥ the tree-vs-chain baseline of 7/12. |
| 2 Copy extension | (a) Done on CPU: `an3.py` on the existing greedy dump, +7.3%. (b) Next: a fresh `SPLASH_DRAFT_DUMP` with prompt ids from 20-50 replayed dsh steps at T=0 **and T=0.6**, then the same policy offline. | (a) no; (b) yes, ~15 min | Kill: under +3% tokens/cycle on replayed agent steps at T=0.6. Before production, a gdn-decode and paged-attention fp64 test at depth-14 paths. |
| 3 Compaction | Log analysis only: on the next long dsh session, read `compaction/summary` usage (cacheReadTokens vs inputTokens) and the post-compaction step with `step_report.py`. | no (normal use) | If cacheRead already covers the range up to ~24K before its end, drop (a). If the first step after compaction is 0-cached, enable sysstate. |
| 4 Attention pair | `attn_profile.sh` / decode-profile at 32K and 64K with `SPLASH_VERIFY_TILE=16` + `kVerifyPagesPerSplit=8` (both in dev-merged); check digests. | yes, ~15 min | Kill: not faster than tile 8 at 64K (114.8 ms). Confirm: under 110 ms at 64K, digest equal. |
| 5 Tree τ | `tree_collect.py` with temperature 0.6 (6 prompts × 1,500 tokens), then `tree_potential.py` with a τ sweep of 0.7-2.0 applied to the tree builder. | dump yes (~10 min), sweep no | Adopt only a τ that wins by ≥1% tokens/cycle. |
| 6 Int8 weight stage | Microbenchmark in `gguf-projection-benchmark`: matmul2d half×half vs bfloat×int8 vs int8×int8 at 16×32×{5120,17408} tiles, plus `relaxed_precision=true` on the current kernel, in one session. | yes, ~1 day to write, 10 min to run | Kill #6 if bf16×int8 is no faster than the half stage with a simple int8 dequant. Kill #10 if int8×int8 is under 1.3× the half rate. |
| 7 DSpark | No GPU at first: read the DSpark Markov head in lithos (`kernels/common/spec_ops.metal`, `monolith/spec/dspark`) to check it factors into unary + pairwise edges. Then use lithos's existing DSpark port to dump per-depth acceptance at depths 8-15 on the 6 tree_collect prompts, before any Splash loader work. | code read no; dump yes (lithos, ~30 min) | Kill: conditional α at depths 8-10 under 0.5, or depth-1..7 acceptance more than 3% below DFlash2's. |
| 8 Reasoning effort | `python3 eval/run_eval.py --conds nojev`, 6 tasks × ≥3 reps, xhigh vs medium (and low). Record reasoning tokens per task. CPU first: count reasoning tokens per step in the existing 25 session logs to size the runaway tail. | eval yes (user-approved window) | Adopt medium if accuracy ≥ xhigh within noise and reasoning tokens fall ≥25%. Set the guard at 8-12K tokens. |

**Separately, explain the 131 ms real cycle.** Run `splash/tools/step_report.py` on df378d27-like sessions and
compare gpu_ms with wall_ms per step, alongside Defender CPU and thermal state. This is a log analysis with no GPU
needed. A 10-15% environmental loss would outweigh every kernel item in rows 9-14.

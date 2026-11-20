# Advanced speed research: synthesis (2026-11-02, run 2)

This report merges and judges five independent research angles:
1. GGUF projection kernels on M5
2. The decode-cycle pipeline and host overhead
3. Tokens per cycle within the 16 verify rows
4. End-to-end agent wall time, from the dsh session logs
5. External state of the art (MLX, llama.cpp, M5 Neural Accelerator, speculative-decoding papers)

The previous synthesis is kept unchanged in `RESEARCH-ADVANCED-2026-11-02.run1.md`. Nothing here ran on the GPU. Every
gain below is checked against the measured cycle budget, and no idea repeats a logged failure (SPEED_LOG.md) without a
new mechanism.

**Exactness labels**
- **exact**: greedy output is bit-identical and the sampled distribution is unchanged.
- **exact-math**: same math, different summation order or operand rounding. Greedy text can flip at near-ties. This is
  the class the 16-node tree was accepted under (7/12 prompts identical to the chain).
- **approximate**: changes the numerics or the model's behaviour. Needs the eval.

**Conversion to agent time.** Decode (reasoning 52.9% + tool-call arguments 4.7% + answer text 1.8%) is about 60% of
agent time in the tree-era sessions (angle 4). So a decode gain of x% is worth about 0.6·x% of agent time.
Tokens-per-cycle gains and per-cycle ms gains multiply.

---

## 1. Time budget

### 1.1 One decode cycle (1 lane, 2K context, head shortlist)
Sources: `logs/speed/ab-1101/profile/profile-head98k-2048.txt` (84.52 ms GPU) and `logs/speed/kernel-ab-diag.txt`.

| Part | ms | Share | Notes |
|---|---:|---:|---|
| GGUF verify projections (64 layers) | ~67.6 | ~80% | about 0.113 ms matmul + 0.08-0.10 ms dequant + about 0.02 ms exposed loads per 89M-weight IQ projection, **added serially** |
| ↳ IQ3_S + IQ3_XXS + IQ2_* (7.66 of 10.66 GB) | ~48 | | dequant is about 40% of each, so about **19 ms** of the cycle is grid dequant |
| ↳ IQ4_XS (1.44 GB) | ~10 | | |
| Draft (5 layers, 1.27 GB) | ~5.5 | 6.5% | about 231 GB/s |
| Output head (shortlist) | ~1.5 | 2% | |
| GDN tree scan + commit | 2.42 + 1.19 | 4.3% | the commit is the "tail" after acceptance |
| Norms, attention glue, selection | ~1.8 | 2% | `norm_rms_f32` ×129 = 0.97 ms |
| Host gap between cycles | **0.63** | 0.7% | **corrected** from the 2-3 ms in SPEED_PLAN; see 2.2 |

**What follows from this table**
- **Floors.** The bandwidth floor of the verify is 10.66 GB / 307 GB/s = 34.7 ms. Its FP16 matrix-unit floor is about
  26.6e9 weights × 32 FLOP / ~25 TFLOP/s ≈ 34 ms. The `nodeq` matmul already reaches 25 TFLOP/s (2.85 GFLOP in
  0.113 ms).
- **At 16 rows, the matmul half is at the matrix-unit peak.** The 41 ms "bandwidth floor" cannot be reached with FP16
  MMA plus serial dequant. The movable block is the about 19-23 ms of dequant (grid formats plus IQ4_XS), unless an
  INT8 MMA rate exists.
- **Ceiling for an IQ dequant rewrite.** Taking grid dequant to Q4_K's measured cost (0.051 ms, which includes
  addressing, the stage store and the barrier) saves at most 0.04 ms × (48/0.225) ≈ **8-9 ms per cycle, +10-11%
  decode**.
  - Angle 1 estimated 12-16 ms. That figure assumed IQ is 90% of verify time; it is 72% of bytes, and IQ4_XS is not
    a grid format. Corrected here.

### 1.2 Long context (what real sessions see)
- decode-profile: 92.9 / 102.8 / 114.8 ms per cycle at 2K / 32K / 64K (SPEED_LOG 10-30). That is about +0.34 ms per
  1K tokens.
- Real sessions: 131 ms per cycle at about 76K. They decode at 30-35 tok/s, against 52.8 on the short benchmark
  (angle 4).
- At the 30-120K contexts dsh actually runs, **25-45 ms of every cycle is context-dependent verify attention.** No
  exact idea in this round attacks it directly, apart from idea E2 (skip the second 8-row tile). It is the largest
  open block (section 6).

### 1.3 Agent wall time (dsh session logs; angle 4)
Tree-era Splash sessions, 4,750 s:

| Part | Share |
|---|---:|
| Reasoning decode | 52.9% (8.7 points of it is one step that hit the 16,384-token cap and produced nothing) |
| TTFT / prefill | 20.6% (7.4 points of it is the pruner re-prefill fixed on 10-27) |
| Tool execution and approvals | 10.0% |
| Compaction | 9.8% |
| Tool-call argument decode | 4.7% |
| Answer text decode | 1.8% |

Across all 25 sessions, 4 runaway reasoning steps took 11.4% of agent time. The reasoning effort the template
actually applies is **xhigh**, because dsh's `reasoning:` setting never reaches Splash.

---

## 2. Key evidence by angle

### 2.1 Projection kernels (angle 1, plus angle 5's external view)
- **The M5 path is `gguf_decode<F,16,…>` → `gguf_staged_steps`** (`kernels/common/gguf_staged_tile.h:50-104`). The
  stage is already double-buffered, and step k+1 is dequantized after matmul(k) is issued. More software pipelining
  cannot help.
  - MLX's `qmm_t_nax_staged_impl` (`quantized_nax.h`) has the same structure. Nothing structural is missing.
- **The costs add** (`kernel-ab-diag.txt`, IQ3_XXS 17408×5120, in ms):
  - IQ3_XXS: base 0.2254, `noload` 0.2058, `nodeq` 0.1131.
  - Q4_K: 0.2115, 0.1647, 0.1138.
  - Dequant cost: IQ3_XXS ≈ 0.093, IQ3_S ≈ 0.097, IQ2 ≈ 0.08-0.084, Q4_K ≈ 0.051.
  - The 0.013 ms `nomm` figure is dead-code elimination; ignore it.
- **Why the logged variants failed.** About 34 simdgroups per core already interleave dequant and MMA, and the costs
  still add. Moving work (`p2`, `direct-db`, `swap`, `m8`, `step64`) does not help. Only removing ALU instructions per
  weight can.
  - The grid-table variants failed on the half-precision sign XOR (SPEED_LOG.md:359-371), not on the idea of cheaper
    decode.
- **Op count per weight in today's grid decode** (`gguf_staged.h:39-45`, `quant_formats.h:240-270`): about 9-11.
  - Per weight: u8→f32, fmul by s, f32→f16, sign test + select.
  - Per 8-weight chunk: sign rebuild with popcount parity. Each 7-bit index is expanded twice, by chunks c and c^1.
  - Q4_K's magic-number path (`gguf_staged.h:11-16`) is about 4-5 ops per weight, which matches its half-size dequant
    cost.
- **Splits.** `kStagedTiers {{6,512}}` (`ops/LinearGguf.cpp:81`) picks 7-10% slower splits for some shapes:
  - IQ3_XXS down: S2 0.2592 vs S4 0.2365.
  - QKV: S1 0.2189 vs S2 0.2033.
  - IQ4_XS down: S2 0.3515 vs S8 0.3176 (`projections.txt`).
  - It was rejected only for the B4 cost (+4.7%, SPEED_LOG.md:346-357). Decode is now capped at 1 lane
    (SPEED_LOG.md:95-97), so that reason is gone.
- **External data.**
  - FP16 matrix-unit throughput is about 1024 FLOP per core per cycle; INT8 about 1900-2048 OPS
    (https://tzakharko.github.io/apple-neural-accelerators-benchmark/).
  - Draw Things measures fused int8 at 1.61-1.87× FP16 on the M5 Max
    (https://releases.drawthings.ai/p/metal-quantized-attention-pulling).
  - Modular lists I8/U8 and E4M3 MMA inputs on M5, and a mixed-dtype B operand (W8A16)
    (https://docs.modular.com/api/kernels/linalg/matmul/gpu/apple/matmul_kernel/AppleM5MatMul/).
  - MLX uses `relaxed_precision = true` (`steel/gemm/nax.h:401-408`). Splash uses `false`
    (`gguf_staged_tile.h:24,57`).
  - llama.cpp's few-row `simdgroup_matrix` MMA (PRs #29869, #30065) runs on the shader ALUs. It is the right design
    for M1-M4, but slower than Splash's matrix-unit path on M5.

### 2.2 Pipeline and host (angle 2)
- **The host gap is 0.63 ms per cycle, not 2-3 ms.** Over about 15,000 cycles in 6 step-log runs:
  - wall − gpu: median 0.37-0.39 ms;
  - inter-cycle gap: mean 0.24-0.29 ms.
  - SPEED_PLAN's 2-3 ms figure is stale.
- **Fusion barely matters.** The fused stream takes 84.52 ms; the same 725 dispatches run as separate commands take
  83.28 ms. That is about 1.7 µs per boundary.
  - The verify is a strict per-layer chain (`QwenTarget.cpp:410-504`), so a concurrent encoder recovers almost
    nothing.
- **The ~1.3 ms tail** (`verify_gdn_commit` 1.191 ms, 16 KV restores 0.060 ms, gather 0.003 ms) is the only GPU work
  that the next cycle's host encoding does not have to wait for.
  - `finalizeDecode` reads only RetainedCount, AcceptedCount, OutputTokens and DraftTree, all written before the tail.
- **No constrained cycles appear in any log.** The grammar round trip is not on dsh's hot path.

### 2.3 Tokens per cycle (angle 3)
- **The depth-7 cap binds where the tokens are.**
  - 20.3% of T=0 agent cycles and **48.6% of T=0.6 code cycles** accept all 7 draft tokens.
  - Capped cycles spend about 8 of their 15 nodes on siblings that are off the accepted path.
- **CPU simulation on `logs/speed/draft-dump.jsonl`** (2,438 cycles). It reproduces tree16 at 4.449 tokens per cycle
  (measured 4.46).
  - Copy chains past depth 7 that compete in one best-first heap **within 16 rows** give **4.850 (+9.0%)**: tool
    +20%, edit +13%, refactor +13%.
  - A host-only root copy chain aligned with DFlash nodes gives 4.689 (+5.4%).
  - The old "root copy chain −0.2 to −0.6%" result capped copy at depth 7. Depth past 7 is what makes copying pay.
- **Plumbing missed in run 1.** `DFlashDraft::addContextCommit` commits only 8 rows (`DFlashDraft.cpp:327-328`). Also:
  - `accept_tree_lane` (`sampling.metal:1662`) and `Runtime.mm:1889` cap accepted tokens at 7;
  - depths 8-15 need uniforms (the 7 unused PROPOSALS slots plus INITIAL make exactly 8).
- **A T=0.6 dump is an exact counterfactual.** Each draw is a function of (prefix, u_d). So `tree_potential.py` can
  score any alternative tree at dsh's real temperature from one dump.
- **`SPLASH_DRAFT_TEMPERATURE_SCALE` is inert in tree mode.** `draft_select_tree` zeroes q (`sampling.metal:1485`).
  This is a config cleanup, not a speed lever.
- **Plain-draw tree acceptance is already optimal** for a deterministic tree, so SpecInfer-style multi-draft
  acceptance adds nothing.
- **Row count at long context.** The second 8-row attention tile reads the whole KV. Rows 9-16 add only +12% tokens per
  cycle (3.975 → 4.45), so above about 64K a fixed 16 rows stops paying.

### 2.4 Agent wall time (angle 4)
- **The effort setting never reaches Splash.**
  - pi-ai's `qwen-chat-template` sends only `enable_thinking` and `preserve_thinking`
    (`openai-completions.js:654-658`).
  - The template therefore defaults to `xhigh` and injects a "validate assumptions, consider alternatives"
    instruction (`qwen38_gguf.jinja:58-69`). `medium` injects nothing.
  - The knobs that do work: `--default-reasoning-effort` (`server/serve_options.py:283`) or
    `chat_template_kwargs.reasoning_effort`.
- **Runaway steps.**
  - Four steps hit 16,384 tokens while still reasoning: 1,657 s, with loops of only 365 unique lines out of 907.
  - t2-semver passes 0/13.
- **Re-prefill of the model's own output** is 38-44% of per-step prefill tokens: about 402 s, 3.5% of agent time.
  - Lane parking removes it. Warming during tool execution would hide only 73 s of it.
- **Compaction** costs 440-465 s per long session: decode of a 6.5-9K-token summary plus a cold 38.5K-token prefill.
- **Fixed cost per request.** The TTFT intercept is 1.66-1.81 s, of which about 1 s is unexplained.
- **Idle release** after 10 min costs about 4 s per long pause.

### 2.5 External speculative decoding (angle 5)
- **DFlash and DDTree** are already in use.
- **Weaver / DFlash-TfM** (arXiv 2607.06763): a one-layer, 57M-parameter autoregressive adapter over DFlash's top-K.
  It reports +32% accepted length vs DDTree at the same tree size, on a GDN hybrid.
  - Splash's measured ceiling for perfect selection among its top-16 is 5.91 vs 4.45 tokens per cycle (+33%).
  - It needs training data that cannot be produced locally in reasonable time.
- **Graft / Token Recycling**: a successor table refreshed from the target's own verify rows. +3.4% accepted tokens on
  DFlash in the paper. It overlaps with the copy extension.
- **SuffixDecoding, PLD, REST, Lookahead, Sequoia, SpecInfer, EAGLE-3**: dominated, already covered, or measured worse
  here (table in section 4).

---

## 3. Ranked ideas

Rank = expected agent wall-time gain × probability of success ÷ effort. Exact and exact-math ideas come first (E),
then the approximate ones (A). The A table holds the largest single lever, because reasoning tokens dominate what the
user waits for.

### 3.1 Exact / exact-math

| # | Idea | Decode gain (arithmetic) | Agent gain | P(success) | Effort (d) | Exactness | Main risk |
|---|---|---|---|---:|---:|---|---|
| E1 | **Copy branches past depth 7 inside the 16-node heap** (angle 3 #1; Graft table as a later add-on) | +9.0% tokens/cycle simulated at T=0 (4.449→4.850); cost ≤0.3 ms of 85 → **+8.5% decode**. Phase A (host root chain, aligned) +5.4% | **+5%** | 0.7 | 4 (Phase A 2) | exact (row computations do not depend on sibling rows; confirm with a greedy digest test) | depth-15 commit paths (8-row draft context commit, GDN replay, gather/restore); copy acceptance at T=0.6 unmeasured |
| E2 | **Re-tune staged K splits for 1-lane decode** (angle 1 #3) | measured −0.66 ms with split-policy on m16; per-format tiers −1 to −2.5 ms → **+1-3%** | +0.6-1.8% | 0.8 | 0.75 | exact-math (fp32 partials reassociated) | gain near noise (A-B-B-A); prefill uses the same splits |
| E3 | **Lean bit-identical grid decode**: magic-number bytes, sign folded into the fp32 scale, each sign index decoded once per group (angle 1 #2) | dequant 0.093 → 0.065-0.075 on the grid formats, about 25% of 19 ms ≈ 4-5 ms → **+5-6%** | +3-3.5% | 0.4 (codegen surprised before) | 1.5 | **exact** (bitwise stage equality) | compiler may already fold the conversions; the earlier XOR variant was +11-15% slower |
| E4 | **Per-cycle verify row count**: skip the second 8-row attention tile when context is long and nodes 8-15 are improbable (angle 3 #2) | gated sim: +0% at 2K, +2.4% at 64K, +5.4% at 96K, +8.9% at 128K; dsh runs at 30-120K → **+3-4% averaged over long sessions** | +2% | 0.55 (the 50% second-tile share is inferred) | 2 | exact (smaller tree, same accept rule; confirm with digest) | the second tile may overlap the first on idle cores, so the saving is smaller |
| E5 | **Integer-operand stage + per-superblock scale chain** (angle 1 #1 merged with angle 5 #1): stage exact (1+2s)·grid ≤ 1922 as half, apply d/4 in fp32 every 8 K-steps | grid dequant toward Q4_K's 0.051 → ceiling 8-9 ms; realistic 5-8 ms → **+6-10%**. Supersedes most of E3's gain if both land | +4-6% | 0.45 | 4 | exact-math (weights exact rather than rounded to half; reassociated per superblock) | second cooperative accumulator: register pressure; per-element column mapping; codegen |
| E6 | **Early host wake at acceptance**: split off the commit tail into a second command (angle 2 #1) | hides the 0.63 ms host gap behind the 1.3 ms tail → **+0.7%**; more under Defender CPU contention | +0.4% | 0.85 | 1.5 | exact | a snapshot or memcpy of lane state before the tail completes corrupts the prefix cache; needs a fence |
| E7 | **MMA vs ALU contention probe, then producer/consumer simdgroup specialization** (angle 1 #4) | if they overlap: projection → max(mm, dq), up to −20 ms → +30%; prior about 0.2 → expected about +5% | EV +3% | 0.2 | 0.5 probe + 3.5 | exact | probe likely negative (costs add; M4 study says matmul2d runs on shader cores) |
| E8 | Bit-identical glue dispatch merges (angle 2 #3) | about 68 dispatches × about 2.7 µs ≈ 0.18 ms → +0.2% | +0.1% | 0.8 | 2.5 | exact | near noise; low priority |
| E9 | Second queue for the GDN tail, concurrent with the next draft (angle 2 #2) | +0.35-0.7% on top of E6 | +0.3% | 0.4 | 3 | exact | cross-queue races (DraftTree, RetainedCount parity) |
| E10 | Free draft depth: 16 draft query rows under a block-causal mask (angle 3 #4) | if conditional acceptance at depth 8 ≥ 0.5: +4.5% agent prompts, +8% T=0.6 code; overlaps E1 | EV +1% | 0.3 | 3 | exact | DFlash2 was trained at block 8, so positions 8-15 are extrapolated |
| E11 | Graft-style target-recycled successor table (angle 5 #3) | +1-3% tokens/cycle on top of E1 | +0.6-1.8% | 0.4 | 2 | exact | overlaps E1; the dump lacks the target's argmax at rejected nodes |
| E12 | Tree-builder temperature τ_tree from the request (angle 3 #3) | 0 to +3% at T=0.6 | +0-1.8% | 0.3 | 1 | exact | likely about 0; free once the T=0.6 dump exists |
| E13 | Trim the per-request fixed cost (tail-chunk fusion, cached tokenized prefix) (angle 4 #5) | prefill-side, about 0.1-1 s per step | +0.4-2% | 0.5 | 1.5 | exact-math (chunk boundaries) | the intercept may just be attention that grows with context |
| E14 | Idle release 10 → 45 min (angle 4 #6) | about 4 s per long pause | <1% | 0.95 | 0.1 | exact | keeps 16 GiB wired while idle |

### 3.2 Approximate (need the eval)

| # | Idea | Gain arithmetic | Agent gain | P | Effort (d) | Main risk |
|---|---|---|---|---:|---:|---|
| A1 | **Reasoning effort xhigh → medium, set where it takes effect** (`--default-reasoning-effort medium` via SPLASH_EXTRA_ARGS, or pi-ai `thinkingFormat: chat-template` + `chatTemplateKwargs.reasoning_effort`) | reasoning is 53% of agent time; a 25-45% token cut → 13-24%, plus shorter contexts make cycles cheaper | **+13-24%** | 0.6 | 1 (+eval) | pass rate on hard tasks; gate on t1/t2/t3 × 3 reps |
| A2 | **Runaway-reasoning guard**: force Qwen's `</think>` early-stop sequence at budget B ≈ 10-12K, or stop on a repetition detector | 4 cap steps = 11.4% of corpus time; saves about 37% of each plus the re-prompt | +4-8% on sessions where it fires | 0.75 | 2 | output changes only on steps above B; must interact correctly with the tool grammar and tree mode |
| A3 | **Finish the park lane** (oracle test, then ship as exact-math if it is a near-tie) | own-output re-prefill = 3.5% of agent time; measured prompt time −24% over 6 turns | +3.5% | 0.6 | 2 | divergence at char 127 may be a real state bug |
| A4 | **Cheaper compaction** (retainRatio 0.16 → 0.08, summary maxTokens 8192 → 4-6K, align contextWindow with Splash's 204,785) | 80-160 s per compaction | +2-3.5% on long sessions | 0.7 | 1 | task state lost after compaction |
| A5 | **int8 MMA: W8A8 verify projections** (angle 5 #4), with **int8/E4M3 B operand** (angle 1 #5) as its exact-math variant | if INT8 runs at about 2× FP16 at 16×32×32: matmul 34 → 17-20 ms, projections 67 → 40-45 ms → +35-40% decode (W8A8); the B-only variant maybe +10-13% | +6-20% | 0.15-0.25 (rate and type support unverified) | 1 probe + 3-10 | W8A8 changes activation numerics at 3-bit weights; matmul2d may emulate mixed types |
| A6 | `relaxed_precision = true` in the projection `matmul2d` descriptor (MLX's setting) | unknown; possibly a faster accumulate path | ? | 0.2 | 0.25 (inside A5's probe) | lower-precision accumulation; needs the fp64 test and the eval |

### 3.3 Combined ranking: top 8 by expected value / effort

| Rank | Idea | Why here |
|---:|---|---|
| 1 | **A1 reasoning effort** | Largest lever in the whole study (+13-24% agent time). About 1 day plus an eval. Approximate, so it has to be gated. |
| 2 | **E1 copy branches past depth 7** | Largest exact lever (+5% agent time). Simulated, not guessed. Phase A gives +3% in 2 days. |
| 3 | **E2 split re-tune** | +1-3% decode for under 1 day; the reason it was rejected is gone. |
| 4 | **A2 runaway guard** | Exact below its budget; removes the worst single waste seen in the logs. |
| 5 | **E3 → E5 IQ dequant instruction cut** | E3 first (exact, 1.5 days), then E5 (exact-math, +6-10% decode). The same microbenchmark harness serves both. |
| 6 | **E4 per-cycle row count** | Only exact idea that targets the long-context term real sessions pay. |
| 7 | **A3 park lane** | Code done; blocked on one oracle test. |
| 8 | **A5/A6 int8 / relaxed-precision probe** | One day of microbenchmarking settles whether a +35% class kernel is even possible on this chip. High variance, cheap to kill. |

Also do, as near-free hygiene: E6 (early wake), E14 (idle release), A4 (compaction config), and the constrained-lane
check in 5.9.

---

## 4. Rejected or merged ideas

| Idea (source) | Verdict | Reason |
|---|---|---|
| More software pipelining or a deeper stage in the staged tile (angle 1) | rejected | already double-buffered with dequant(k+1) after the matmul(k) issue; prefetch depth 2 was measured worse |
| Scale prefetch alone (angle 1) | merged into E5 | the meta load is 2 bytes once per 8 steps; E5 folds it into the payload prefetch |
| Further QKV / gate-up fusion (angle 1) | rejected | QKV is already fused; cost is per weight, and boundaries cost about 1.7 µs |
| Grid tables with half XOR, `direct`, `swap`, `m8`, `step64`, `p2`, verify tiles 16/8x2 | not re-proposed | measured failures (SPEED_LOG). E3 differs from the grid variants: no XOR on half, and it is bitwise checked |
| llama.cpp few-row `simdgroup_matrix` MMA (angle 5) | rejected for M5 | shader-ALU MMA runs at about 1/4-1/8 of the matrix-unit rate: about 0.45 ms vs 0.225 ms per projection |
| Pre-encode cycle N+1 / ICBs / GPU-resident K-cycle loop (angle 2 #5) | rejected | capped at the 0.63 ms gap, which E6 already hides; about 8 days |
| Fold the GDN commit into the next verify scan (angle 2 #6) | rejected | the GDN kernel is latency-bound at 63 GB/s; adding a replay costs more than the 1.19 ms commit and removes the tail E6 needs |
| Fold RMS norms into the projections (angle 2) | rejected | not exact if factored; the extra multiplies in an ALU-bound kernel likely exceed the 0.97 ms saved |
| Shortcut CPU round trips in acceptance (angle 2) | not applicable | acceptance already runs on the GPU; the only round trip is between cycles (E6) |
| Rejected-subtree reuse; trees larger than 16; fixed or static tree shapes, Sequoia (angles 3, 5) | rejected | ≤+1% (59/1,942 recoveries); rows past 16 cost ×1.38; the fixed shape measured 4.15 vs 4.45 |
| Shrink the tree below 16 on easy cycles (angle 3) | merged into E4 | projections are flat from 8 to 16 rows; only the attention tile scales |
| SpecInfer multi-draft acceptance (angle 3) | rejected | plain-draw tree acceptance is already optimal for a deterministic tree |
| `SPLASH_DRAFT_TEMPERATURE_SCALE` tuning | not a lever | inert in tree mode (`sampling.metal:1485`); mark it chain-only in `stack.env` |
| asf0 Qwen3.8-27B-DFlash block 16 (angle 5) | rejected | 3.22 tokens/step; its own card says it is behind DFlash2 |
| SuffixDecoding cross-session, PLD, REST, Lookahead, EAGLE-3 (angle 5) | dominated or covered by E1 | tool-argument and answer decode is only about 7% of agent time; EAGLE-3 is dominated by DFlash |
| STree / TreeWY / SpecLA GDN tree verify (angle 5) | nothing new | Splash already ships a GDN tree scan with path replay |
| Weaver / DFlash-TfM adapter (angle 5 #2) | deferred | promising (+7-12% decode estimated) but 15 days and about 300K training rollouts that need rented GPUs and user approval. Kill test: a tiny adapter trained on a few thousand dumped cycles, re-scored with `tree_potential.py` |
| Warm the own-output prefix during tool execution (angle 4) | rejected | hides only 73 of 402 s; the park lane (A3) is the mechanism |
| Quest / page-selective attention | still not chosen | approximate, costs needle accuracy (SPEED_PLAN "Not chosen") |
| Angle 1's 12-16 ms dequant saving | corrected | based on IQ = 90% of verify time; it is 72% of bytes. The Q4_K-cost ceiling is 8-9 ms |
| Angle 4's "host gap 2-3 ms" premise (from SPEED_PLAN) | corrected | measured 0.63 ms over about 15,000 cycles |

---

## 5. Cheapest first experiments

GPU runs need the user's approval (memory: "ask before GPU runs"). Each item says whether it needs the GPU.

1. **A1 reasoning effort. GPU (eval).**
   - First, a no-GPU check: render the template with `reasoning_effort` unset vs `medium` and confirm the system-prompt
     difference (Python with the jinja fixture).
   - Then run the eval: t1/t2/t3 × 3 reps, `--default-reasoning-effort medium` vs the default xhigh. Compare
     `wall_s`, output tokens and `hidden_passed` in `eval/results`.
   - Kill if the pass rate drops more than one task-rep, or reasoning tokens fall less than 15%.
2. **E1 copy branches. One GPU dump, the rest offline.**
   - Collect a T=0.6 `SPLASH_DRAFT_DUMP` (with seeds and prompt ids) over 20-50 replayed dsh steps.
   - Re-run `scratchpad/sim.py` and `aligned.py` (paths in angle 3) to tune c and n at T=0.6.
   - Kill below +3% tokens per cycle at T=0.6.
   - The same dump also answers E12 (τ_tree) and sets θ(L) for E4. No GPU work is needed for the T=0 result, which
     already exists (+9.0%).
3. **E2 splits. GPU (minutes).**
   - `gguf-projection-benchmark` sweep at L1 for every verify shape and format, to finalise the per-format tiers.
   - Then a decode-profile A-B-B-A on agent prompts. Kill if under 0.5 ms per cycle.
4. **A2 runaway guard. No GPU first.**
   - Replay the 4 runaway transcripts through a repetition detector, offline in Python, to choose B and the
     line-repeat threshold. Count the false positives on all non-runaway reasoning blocks in the 25 sessions.
   - GPU only to check that the forced `</think>` produces a valid tool call.
5. **E3 / E5 dequant instruction cut. GPU microbenchmark.**
   - Implement the E3 QuantGrid branch for IQ3_S in a dev copy (exact). Run the dequant test for bitwise stage
     equality and time `gguf-projection-benchmark` IQ3_S and IQ3_XXS 17408×5120.
   - Kill E3 if IQ3_XXS is not under 0.21 ms (today 0.225).
   - Then the E5 prototype on IQ3_S. Kill if it is not under 0.19 ms or if the fp64 projection test fails.
   - About 1 day of code per step, about 10 min of GPU each.
6. **E4 per-cycle row count. GPU.**
   - decode-profile at 32K / 64K / 96K with the second 8-row attention tile forced to early-exit (a debug flag). Rows
     8-15 are then wrong, which is fine for timing.
   - If 64K does not drop by at least 5 ms from 114.8, kill. If it does, use the T=0.6 dump from item 2 with
     `gate.py` to set θ(L).
7. **A3 park lane. GPU.**
   - The pending oracle test: same prompt, park→adopt vs cold prefill; dump the logits at the first differing token.
     A top-2 margin under about 0.05 points to a near-tie.
8. **A5/A6/E7 matrix-unit probes. GPU, 1 day total, settles three ideas.**
   - (a) Compile-test `matmul2d_descriptor` with int8×int8→int32, half×int8 and e4m3 B. Time a `nodeq`-style matmul at
     16×32×32 tiles against half×half.
   - (b) The same kernel with `relaxed_precision = true`.
   - (c) The concurrent dispatch of a `nodeq` matmul and a pure-ALU dequant loop (`MTLDispatchTypeConcurrent`), to see
     whether the time is max(a, b) or a + b.
   - Kill A5 if int8 is under 1.3× the half rate. Kill E7 if the concurrent time is about the sum.
9. **Zero-cost checks. No GPU.**
   - Confirm `"constrained": false` in the next real session's `logs/splash/steps.jsonl` (`step_report.py`). A
     constrained tool request silently drops to the 8-row chain: 3.68 tokens per cycle, −17%.
   - Read `curl :8000/status | jq .latency` during a long session to size E13.
   - Recheck the host gap in production step logs once they accumulate. Only if it exceeds 1.3 ms does the pre-encode
     family come back.

---

## 6. Open gap for the next round

In real sessions (30-120K context), 25-45 ms per cycle is context-dependent verify attention, about as large as the
whole movable dequant budget. The exact options tried so far (tiles 16/8x2) lost. E4 removes the second tile's half
when the rows are not needed. A dedicated study is warranted:
- why the 76K real cycle (131 ms) is steeper than decode-profile (114.8 ms at 64K);
- the bandwidth of turbo-int4 KV per tile;
- a split-K (flash-decoding) partition over pages that keeps 8-row tiles but adds threadgroups.

Note that int8 Q·K appears in SPEED_PLAN "Not chosen" pending the same int8 probe as experiment 8.

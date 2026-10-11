# TurboQuant in Splash: improvement plan (2026-10-07)

Written by a multi-agent review: 3 code reviewers, with 3 adversarial verifiers per finding; 4 proposal tracks, with 2 judges per proposal; then a synthesis.
It ran CPU-only, and the accuracy numbers come from NumPy simulations plus a replay on real INT8 cache pages.
Raw output: `/private/tmp/claude-501/.../scratchpad/wf_result.json` (scratch).

## Status (2026-10-07 evening)

| Item | State |
|---|---|
| Bug: kvbench merge, compare pairing and error counts | **done** (`kvbench.py`) |
| Bug: stale SSD pages measured | **done**: the cache namespace carries `kv::quantizerRevision` (turbo-int4 r2), needles have a nonce, and judgment/greedy prompts are byte-identical across formats by design (`--nonce` to add one) |
| Bug: `splash_patch` window and ports | **done** (`dsh-env.sh`, static fallback 65536, `KVBENCH_URL` / `SPLASH_PORT`) |
| Bug: Makefile RUNTIME_METAL | **done** (applied with step 2) |
| Bug: doc drift on the default format | **done** |
| Step 1: `/status` deltas, judgments, `ENV=` in suite lines | **done**. The deltas showed the decode differences are draft acceptance (same ms per cycle). Judgment set **v2**: 23 answerable questions with true answers, so accuracy per format, on a mixed 8-file context with planted facts |
| Step 2: turbo-int4 16 codes, 2.7 sigma clip, least-squares scale | **applied** 2026-10-07. Every kernel criterion met: key error 0.111 → 0.096, values 0.122 → 0.102; attention vs exact 0.978 → 0.986 (85/2) and 0.977 → 0.984 (300/3); peaked 0.994 / 0.996; GPU vs reference exact; attention time unchanged. End to end: needles 6/6; greedy identical to int8 8/12 (was 6/12). **Owed:** the clean judgment A/B against the old quantizer (`splash/tools/next_gpu_runs.sh`) |
| Step 3: 2x2 format x `--persistent-cache` suite | scripted in `splash/tools/next_gpu_runs.sh`; needs ~1 h of GPU. Data so far: INT8 at 58K under one flag setting ran at 271, 303 and 343 tok/s on three runs, so run-to-run spread (~±12%) dominates; turbo-int4 is level with int8 |
| Step 4: doctor effective-format check, `scripts/stop.sh` | **done**; doctor passes on the live turbo-int4 server |
| Step 5: turbo Lloyd-Max K1 | **staged** in `splash/staged/step5/`:
- `turbo_kv.h` has Max's 16-level table over √128 (±0.011349 ... ±0.241529); the fork's table is 0.6-0.7x narrower;
- `PagedKv.hpp` gives turbo cache revision 2.
- CPU checks: a simulation gives 23% lower key error and 33-43% lower score KL; the codebook's MSE is 0.00951 (theory 0.009497). The kernels compile with it (`rtc_probe`), and `PagedKv.hpp` passes `-Werror` with a `static_assert` on the revisions.
- `turbo_kernel_test.py` now reads every codebook from the header, with consistency asserts, so no test edit is needed.
- `next_gpu_runs.sh` applies it only if `make test` passes, and A/Bs it against the fork table. |
| Step 6: `make -C splash/tools test`, one flattener, prototypes archived | **done**. (e) committing splash-src is optional: the work is published as `patches/0003` (2026-10-11) |
| Step 7: value centering | gated on step 3 |

**Check of step 2's claim (independent CPU sim, `/private/tmp/claude-501/wf/verify-p1/sim.py`):**
- Reconstruction error drops 14-17%, as claimed.
- Attention KL drops 25-50% at normal and moderately peaked queries.
- At very peaked queries the sign is noise-dominated: from 7% better to 10% worse.
- So step 2's success criteria (peaked row >= 0.994, judgment KL vs the INT8 noise floor) must be
  checked before it becomes the default.

## Summary

Plan for the next TurboQuant round, centred on turbo-int4, the format the user actually runs (config/stack.env:33 default). Order of work:
(1) Fix kvbench bookkeeping and add per-request /status deltas, so later numbers can be trusted.
(2) Replace turbo-int4's absmax/7 store quantizer with signed 16-level codes, a 2.7-sigma clip and a least-squares scale refit. This is a change to splash_store_turbo_int4_row (paged_store_row.h:147-167) only. Two independent CPU sims and one replay on real keys put the KL drop at about 33% and the key-error drop at about 17%, with no change in bytes or attention time.
(3) Run one interleaved 2x2 A/B (format x --persistent-cache) on macOS 27. It answers the real open speed question, whether write-behind costs 10-20% of prefill, and gives a noise floor.
(4) Cheap fixes for turbo, docs and ops: K1 Lloyd-Max constants, the 5 remaining confirmed bugs, and doc drift.
(5) Optional, gated: value centering only in the last attention layer, where real data shows a 0.79 held-out mean share.
Dropped: key centering (real gain 7-9%, below its own 20% go threshold), the k8v4 formats (cost about 34% of context, and no accuracy failure has been shown), turbo prefill transcode, query-rotation fusion, and the full docs restructure.
Already applied in the current tree: the Homebrew turbo* fallback (serve-splash.sh:22) and the turbo-int4 default. Still stale: README.md:1327 and 1643-1644, TURBOQUANT.md:311 and HANDOFF.md:109.
Nothing was run on the GPU. Line numbers were re-checked against the current files.

## Confirmed bug fixes

### kvbench needle merge keeps log-recovered INT8 needles; file-level time/kv overwritten on partial reruns (medium)
`splash/tools/kvbench.py:83`

Line 83: when needles are rerun, drop every earlier entry that has no 'tokens' key (`z.get('tokens') is not None and z['tokens'] not in sizes`). Lines 160-164: write time, kv, max_context, kv_page_bytes, build id (status identity) and whether the persistent cache is on into a per-part dict (res['meta'][part]) instead of the top level. Before merging, refuse with a clear error when the stored kv.format differs from the live /status identity.kv.format. A --fresh flag starts a new file. Check: rerun `kvbench.py run --label int8 --parts needles --needle-tokens 16000`, then confirm that `compare int8 turbo-int4` lists only entries with a 'tokens' key and that int8.json has meta.needles.time set to the rerun.

### kvbench judgments/greedy prompts are byte-identical across runs, and the persistent-cache namespace ignores quantizer changes, so stale SSD pages can be measured (medium)
`splash/splash-src/runtime/engine/RuntimeResources.mm:210`

(a) In persistentCacheNamespace (RuntimeResources.mm:210-219), append a per-format quantizer revision after formatName, for example a `kv::quantizerRevision(format)` returning 1 for int8/bf16 and 2 for turbo/turbo-int4 after steps 2/7. Bump the revision with every change to store math or codebooks, and document that rule in TURBOQUANT.md section 11. This invalidates only that format's pages, unlike a global kPersistentCacheFormat bump (line 91), which would also wipe the INT8 caches. (b) kvbench.py:124/147: record cache_n for each greedy request (it is already in chat()), and print a WARNING in compare when a part's first request had cache_n > 0. For judgments, prefix the state with a fixed-width per-run nonce (`[run %020d]`), because /v1/judgments returns no cache info. Check: after a rebuild with a new revision, the first startup log shows 0 adopted points for that format and INT8 still adopts its own.

### kvbench compare hides errored needles and zips greedy/judgment entries by position (low)
`splash/tools/kvbench.py:228`

Line 228: count entries that have an 'error' key and print `x/y correct, z errors` (list the failing tokens@depth). Lines 208 and 219: before computing agreement or KL, check that ga['prompt'] == gb['prompt'] and ga['context_tokens'] == gb['context_tokens'], and that ja['question'] == jb['question']. Print MISMATCH and skip the pair instead of zipping past it, and use len(pairs) as the denominator, not len(A['greedy']). Check: hand-edit a copy of a JSON to add {'tokens':197000,'depth':0.5,'error':'x'} and confirm that compare prints '9/9 correct, 1 errors'.

### splash_patch uses the static 73728 window when the server reports a limit at or below 28672 (medium)
`scripts/dsh-env.sh:24`

Use the static file only when /status is unreachable or not numeric. When the limit is numeric and limit > 16384+4096, always generate window = limit-20480. When limit <= 20480, print `error: Splash limit $limit leaves no room for the 16384-token reply budget` to stderr and still generate the patch with window = max(4096, limit-20480), so dsh compacts rather than getting HTTP 400. Also lower the static fallback (config/dsh/splash.cordis.patch.yml:33) to a value below int8's limit, e.g. 65536. Check: stub curl returning 24576 under bash and zsh, and confirm splash_patch outputs the generated file with contextWindow 4096 plus the error line.

### dsh Splash patch and kvbench hard-code port 8000, ignoring SPLASH_PORT/LLAMA_HOST (low)
`config/dsh/splash.cordis.patch.yml:12`

In splash_patch (dsh-env.sh:27), add a second sed expression: `-e "s#baseURL: http://127.0.0.1:8000/v1#baseURL: http://$LLAMA_HOST:$SPLASH_PORT/v1#"`. Apply it in the static branch too, by always writing the generated copy. In kvbench.py:22 use `URL = os.environ.get('KVBENCH_URL', f"http://{os.environ.get('LLAMA_HOST','127.0.0.1')}:{os.environ.get('SPLASH_PORT','8000')}")`. Check: `SPLASH_PORT=8001 bash -c 'source scripts/dsh-env.sh; grep baseURL $(splash_patch)'` prints :8001.

### Makefile RUNTIME_METAL mode is not part of the build config; switching 1->0 runs metallib onto a directory (low)
`splash/splash-src/Makefile:194`

Add `RUNTIME_METAL=$(RUNTIME_METAL)` to the CONFIG_DIGEST printf list (lines 208-213), or append `-rt$(RUNTIME_METAL)` to PRODUCTION_LIB_CONFIG (line 375). Make the offline recipe start with `rm -rf $@` (line 194, before metallib). Re-export patches/0003 afterwards. Check (CPU only): `make -n RUNTIME_METAL=0 all` on a tree with build/splash.metallib/ as a directory shows `rm -rf build/splash.metallib` before metallib. Run the real build only when the GPU suite has finished, because make links the engine and this rule forbids running make now.

### Docs still present turbo as the default / give the switch as SPLASH_KV_FORMAT=turbo (low)
`README.md:1327`

README.md:1327: change the default column to `turbo-int4`. README.md:1643-1644 and splash/TURBOQUANT.md:311: say 'turbo-int4 is the default; for maximum context run SPLASH_KV_FORMAT=turbo ./scripts/serve-splash.sh (2.2-2.4x slower attention), for accuracy-first SPLASH_KV_FORMAT=int8'. splash/HANDOFF.md:109-110 and 254: mark the switch as done (2026-10-06). Check: `grep -n "\`turbo\` |\|currently \`turbo\`" README.md splash/HANDOFF.md` returns nothing.

## Steps, in order

### 1. Benchmark harness you can trust: bookkeeping fixes and per-request /status deltas
*measurement, effort S*

Apply the three kvbench bug fixes listed above. In chat() (kvbench.py:42-51), GET /status before and after each request and store these deltas:
- metrics: prefill_wall_ms, prefill_input_tokens, decode_cycle_ms, decode_output_tokens, drafted_tokens, accepted_draft_tokens;
- scheduler.decode_batches;
- kv tier: kv_demotions, kv_copies, kv_restores, write_behind.waiting/durable;
- disk written_bytes;
- requests.completed (assert a delta of 1).
Derive ms/cycle = d(decode_cycle_ms)/d(decode_batches) and tokens/cycle = d(decode_output_tokens)/d(decode_batches). decode_cycle_ms is a server-wide cumulative sum (Status.hpp:171), so it must be divided by decode_batches.
Extend JUDGMENTS (kvbench.py:107-120) from 6 to about 24 questions with near-tie options, so a 25-35% KL change can be resolved. Have compare print acceptance and ms/cycle per format.
run_kv_suite.sh:
- accept a per-line `ENV=K=V[,K=V]` token and pass it to serve-splash.sh (line 25);
- derive ROOT from the script path (line 8);
- record SPLASH_PERSISTENT_CACHE in the kvbench meta.

**Files:** `splash/tools/kvbench.py`, `splash/tools/run_kv_suite.sh`

**Verification:** CPU only: `python3 splash/tools/kvbench.py compare int8 turbo-int4` still runs on the old JSONs, which have no deltas, and prints the error counts. `bash -n splash/tools/run_kv_suite.sh`. When the GPU is free, the first suite run (step 3) produces JSONs where every needle has a delta block.

**Success criterion:** Every request in a new JSON has the delta block with requests.completed delta == 1. compare prints 'x/y correct, z errors', acceptance and ms/cycle. It reports MISMATCH, never a silent zip, on mismatched prompts.

**Expected gain:** Removes 3 confirmed reporting defects. Makes decode tok/s separable into acceptance and kernel cost, and makes prefill attributable to write-behind IO. Every later step depends on this.

### 2. turbo-int4 store: signed 16-level codes + 2.7-sigma clip + least-squares scale refit (P1)
*accuracy-default-format, effort S*

Edit paged_store_row.h:147-167 (simdgroup 0, x[8] per lane after splash_turbo_rotate256):
- pos = simd_max(max x), neg = simd_max(max -x), rms = sqrt(simd_sum(sum x^2)/256);
- a = min(2.7*rms, max(pos,neg))/8; step = (pos>neg) ? -a : a; inv = (a>0) ? 1/step : 0;
- c = clamp(int(rint(x*inv)), -8, 7);
- xc = simd_sum(sum x*c), cc = simd_sum(sum c*c); scale = (cc>0) ? xc/cc : 0;
- pack as today (line 159) and write the scale at line 167.
Keep 2.7 as a named constant. Multiply by inv rather than dividing, because the build uses fast math, and keep the zero-row guard. turbo_int4_test.py has an all-zero key row.
Mirror the same operations in float32 in turbo_int4_test.py quantize() (lines 33-39). Keep the existing tolerances at line 85: scale error < 1e-5, < 0.1% codes off by 1. If simd_sum ordering trips the scale check, compute the reference sums lane-wise (8 per lane, then a 5-step butterfly).
Add the quantizer revision (bug fix 2a) so old SSD pages are not mixed into the A/B. Old pages would still decode correctly, since the rule code*scale is unchanged.
No tile, softmax, reduce, layout or page-byte change. paged_attention_tile.h:116-118 and 156-170 treat scales as plain multipliers of any sign. Prefill and decode both reach the store via splash_store_kv_row.

**Files:** `splash/splash-src/runtime/metal/kernels/common/paged_store_row.h`, `splash/tools/turbo_int4_test.py`, `splash/splash-src/runtime/engine/RuntimeResources.mm`, `splash/TURBOQUANT.md`

**Verification:** After the GPU suite finishes:
- `cd splash/tools && python3 turbo_int4_test.py`;
- `python3 turbo_attention_test.py 85 2 int4` and `python3 turbo_attention_test.py 300 3 int4`. The GPU-vs-reference line also proves that the PV operand decodes code -8; int4_probe only covered QK;
- `python3 flatten_metal.py ../splash-src/runtime perf_kernels.metal > /tmp/perf.metal && ./attn_perf /tmp/perf.metal 57344`;
- end to end, in the step 3 suite: `kvbench.py run --label i4-s16 --parts judgments greedy needles --needle-tokens 16000 60000 150000`, then `compare int8 i4-s16` vs `compare int8 turbo-int4`.
CPU prediction to compare against: /private/tmp/claude-501/wf/accuracy-int4/v2/predict_attn_test.py.

**Success criterion:** turbo_int4_test: RESULT PASS, reconstruction median <= 0.100 (the current rule gives 0.111).
turbo_attention_test int4:
- GPU vs reference cosine min >= 0.9999;
- TurboQuant vs exact mean >= 0.984 at 85/2 (predicted 0.9859) and >= 0.982 at 300/3 (predicted 0.9844, current 0.9771);
- peaked row >= 0.994.
attn_perf turbo-int4 within ±2% of 184.2 ms prefill / 0.892 ms verify at H=57344.
kvbench: needles not worse than 9/9, and judgment mean KL vs int8 <= today's 0.0072. The KL improvement only counts if it exceeds the int8-vs-int8 floor from step 3.

**Expected gain:** About 17% lower K/V reconstruction error and about 33% lower attention KL. Real-key replay: KL 4.09e-2 -> 2.75e-2, key error 0.1253 -> 0.1036. Zero bytes and zero attention time, on the default format.

### 3. One interleaved 2x2 suite on macOS 27: format x --persistent-cache, plus noise floor and long needles
*speed-measurement, effort S*

A single run_kv_suite plan, run on a quiet machine with Teams and WebStorm closed and >= 16 GiB reclaimable, using the step 1 ENV token. ABBA order:
- `int8 i8-pc0 ENV=SPLASH_PERSISTENT_CACHE=0 --parts speed needles judgments greedy --needle-tokens 16000 60000`
- `turbo-int4 i4-pc1 ENV=SPLASH_PERSISTENT_CACHE=1 --parts speed needles --needle-tokens 16000 60000`
- `turbo-int4 i4-pc0 ENV=SPLASH_PERSISTENT_CACHE=0 --parts speed needles judgments greedy --needle-tokens 16000 60000 150000 190000`
- `int8 i8-pc1 ENV=SPLASH_PERSISTENT_CACHE=1 --parts speed needles --needle-tokens 16000 60000`
- `int8 i8b-pc0 ENV=SPLASH_PERSISTENT_CACHE=0 --parts judgments greedy` (int8 self noise floor).
If step 2 is built by then, run the turbo-int4 lines on the new build under label i4-s16. This replaces speed-P1, speed-P2, speed-P3 and S4's A/B with one plan.

**Files:** `splash/tools/run_kv_suite.sh`, `splash/TURBOQUANT.md`, `splash/HANDOFF.md`

**Verification:** `kvbench.py compare i8-pc0 i4-pc0` (format effect), `compare i4-pc0 i4-pc1` and `compare i8-pc0 i8-pc1` (flag effect), and `compare i8-pc0 i8b-pc0` (noise floor). Correlate per-needle prefill_wall_ms with the write_behind.durable and disk written_bytes deltas from step 1.

**Success criterion:** All prefill tok/s are in the cold range of 200-450, none in the 10^4-10^5 cached range.
Format: turbo-int4 within ±5% of int8 at equal flag.
Flag: if pc1 is more than 5% below pc0 for both formats, and slow needles coincide with non-zero write-behind deltas, open the follow-up 'defer WriteBehind::run while a Prefill plan is pending' (Engine.cpp:182, with nextWakeup returning nullopt). Otherwise keep SPLASH_PERSISTENT_CACHE=1 and attribute the gap to macOS 27.
turbo-int4 needles at 150K/190K all correct, with no HTTP errors.

**Expected gain:** Settles the 10-16% prefill question with a controlled measurement. Gives the noise floor every accuracy step needs. Validates the default format beyond 82K. Potentially 10-20% prefill if write-behind is the cause.

### 4. Ops and docs bug fixes plus doctor effective-format check
*simplicity, effort S*

Apply the remaining bug fixes: dsh-env.sh:24 window, port propagation in dsh-env.sh/kvbench.py:22, Makefile:194/208-213 RUNTIME_METAL, and the doc drift at README.md:1327/1643, TURBOQUANT.md:311 and HANDOFF.md:109.
doctor.sh:52-56:
- PASS for turbo-int4;
- WARN for turbo ('2.2-2.4x slower attention for +10% context');
- WARN for int8/bf16;
- WARN, not FAIL, when the served format differs from the effective configured format: want=$SPLASH_KV_FORMAT, but int8 when $SPLASH_SRC/splash is not executable and want==turbo*, mirroring serve-splash.sh:17-25.
Add scripts/stop.sh: SIGINT to the server.server PID, using run_kv_suite.sh's anchored pattern; wait up to 90 s; pkill -9 splash serve-native; and the PID on $LLAMA_PORT via lsof for llama. run_kv_suite.sh should call it, edited only between suites.

**Files:** `scripts/dsh-env.sh`, `config/dsh/splash.cordis.patch.yml`, `scripts/doctor.sh`, `scripts/stop.sh`, `splash/splash-src/Makefile`, `README.md`, `splash/TURBOQUANT.md`, `splash/HANDOFF.md`

**Verification:** Stub tests under bash and zsh: splash_patch with a fake /status limit of 24576 and 192505, and with SPLASH_PORT=8001. `bash -n` on all scripts.
After the suite: `./scripts/serve-splash.sh` then `./scripts/doctor.sh` exits 0 and prints 'KV format turbo-int4'. `./scripts/stop.sh` leaves `pgrep -f 'server.server|serve-native|llama-server'` empty.

**Success criterion:** Limit 24576 yields a generated window <= 4096 plus an error line, not 73728. The baseURL follows SPLASH_PORT. doctor exits 0 on the default stack and WARNs (exit code unchanged) on a deliberate int8 server. The README/HANDOFF grep for the stale default returns nothing.

**Expected gain:** Removes the remaining confirmed defects. One stop command. doctor catches a server left on the wrong format.

### 5. turbo (max-context option): true Lloyd-Max 4-bit key codebook (K1)
*accuracy-turbo, effort S*

Replace SplashTurboCentroids4/Midpoints4 (turbo_kv.h:43-50) with Lloyd-Max N(0,1)/sqrt(128): ±{0.011350, 0.034303, 0.058051, 0.083286, 0.111033, 0.143010, 0.182872, 0.241527}.
- Midpoints are computed from the rounded centroids; keep the middle one exactly 0 and keep the table strictly symmetric.
- Fix the comment at turbo_kv.h:42.
- Update C4/M4 in turbo_kernel_test.py:21-25 with the same float32 literals. A small generator script next to the test would keep the two in sync.
- Old turbo pages decode wrongly under the new table, so the bug fix 2a revision must bump for turbo.
- Delete build-turbo/, which holds flattened copies of the old constants.

**Files:** `splash/splash-src/runtime/metal/kernels/common/turbo_kv.h`, `splash/tools/turbo_kernel_test.py`, `splash/splash-src/runtime/engine/RuntimeResources.mm`

**Verification:** First run `cd splash/tools && python3 turbo_attention_test.py 300 3` on the old table to confirm the 0.968-0.969 baseline. Then rebuild and run `python3 turbo_kernel_test.py` and `python3 turbo_attention_test.py 300 3`. CPU re-check: `python3 /private/tmp/claude-501/wf/accuracy-keys/sim_real.py kq keys`.

**Success criterion:** turbo_kernel_test RESULT PASS, bit-exact. turbo_attention_test: GPU vs reference >= 0.9999, TurboQuant vs exact mean >= 0.972 (CPU-predicted 0.9732, from 0.9683), min >= 0.950. attn_perf turbo unchanged within ±2% (460.9 ms at 57,344).

**Expected gain:** About 28% lower key error and about 40% lower key KL for turbo, at the same bytes and speed. The output cosine gain is small because values are 3-bit.

### 6. Tools hygiene: one test command, no duplicate flattener, committed Splash work
*simplicity, effort S*

(a) Delete splash/tools/flatten_metal.py, which is byte-identical to splash-src/dev/tools/flatten_metal.py. Point turbo_kernel_test.py:84, turbo_int4_test.py:59, turbo_attention_test.py:57 and perf_variants.py:15 at SRC.parent/'dev/tools/flatten_metal.py', and fix TURBOQUANT.md:325.
(b) Add splash/tools/Makefile:
- static pattern rule for turbo_harness, turbo_attn_harness and attn_perf (clang++ -std=c++20 -O2 -fobjc-arc -framework Foundation -framework Metal);
- `test` runs turbo_kernel_test, turbo_int4_test and turbo_attention_test for '24 1', '85 2' and '300 3', each in both turbo and int4 (8 RESULT lines);
- `perf` with H ?= 4096 16384 57344;
- FLATTEN and WORK ?= $(TMPDIR) defined in the Makefile; .PHONY targets.
(c) Replace the hard-coded WORK paths with os.environ.get('TURBO_WORK', tempfile.gettempdir()) + per-test subdirs.
(d) splash/ has no git, so archive int4_proto*.metal and perf_kernels_int4.metal under splash/tools/research/ instead of deleting them. Keep hostavail.py, which serve-splash.sh:51 uses.
(e) Commit splash-src on turboquant-kv, excluding build-turbo/ (`rm -rf build-turbo` first, as HANDOFF says). Export `git diff 1.2.1 HEAD > patches/0003-...`. doctor.sh WARNs when that diff differs from the patch, or when `git diff --quiet HEAD` fails on tracked files.

**Files:** `splash/tools/Makefile`, `splash/tools/turbo_kernel_test.py`, `splash/tools/turbo_int4_test.py`, `splash/tools/turbo_attention_test.py`, `splash/tools/perf_variants.py`, `scripts/doctor.sh`, `splash/TURBOQUANT.md`, `splash/HANDOFF.md`

**Verification:** Before the change, record the stdout of the three tests as TURBOQUANT.md documents them. After it, run `make -C splash/tools clean test`, then `make -C splash/tools perf H=57344`, then `git -C splash/splash-src diff 1.2.1 HEAD | cmp - patches/0003-*.patch`.

**Success criterion:** 8 x 'RESULT: PASS' and exit 0, with mismatch counts and cosines equal to the baseline on the shared cases. If the new turbo attention cases at 24/1 or 85/2 fail, treat that as a kernel finding. perf int8 196.6 / turbo-int4 184.2 ms within ±3%. The patch cmp succeeds. `grep -rn 'tools/flatten_metal\|python3 flatten_metal.py' splash README.md scripts` returns nothing.

**Expected gain:** One command covers both formats at 1/3/10 pages; today the turbo attention path is never run. Removes a manual 'keep in sync' rule. Committed work survives a stray checkout.

### 7. Gated: exact value centering in the last attention layer(s) for turbo-int4
*accuracy-default-format, effort M*

Gate on CPU first. Using INT8 persistent-cache pages, as /private/tmp/claude-501/wf/p2-skeptic/share.py does, plus a fresh mix that includes code prompts, fit mu_v per (layer, KV head) with a page-disjoint holdout. Replay output error with and without centering on top of the step 2 quantizer.
Build only if the held-out layer-15 output error still drops by 2x or more, and only for layers whose held-out share is >= 0.5.
Implementation:
- subtract mu_v before splash_turbo_rotate256 in the value branch of splash_store_turbo_int4_row;
- add mu_v after the inverse rotation in splash_attention_reduce_row<Rotation 256> (paged_attention_tile.h:338-368); this is exact because the probabilities sum to 1;
- bind a per-layer 1 KiB view of one table buffer at a byte offset in PagedAttention.cpp, with no params ABI change;
- a zero table when the file is absent;
- hash the table digest into the turbo-int4 namespace.

**Files:** `splash/splash-src/runtime/metal/kernels/common/paged_store_row.h`, `splash/splash-src/runtime/metal/kernels/common/paged_attention_tile.h`, `splash/splash-src/runtime/ops/PagedAttention.cpp`, `splash/splash-src/runtime/engine/RuntimeResources.mm`, `splash/tools/turbo_attention_test.py`

**Verification:** CPU gate script (page-disjoint calibration). Then extend turbo_attention_test.py with a per-head value mean and pass mu to the harness: `python3 turbo_attention_test.py 300 3 int4` with and without mu. End to end: `kvbench.py run --label i4-vc --parts judgments greedy`, then `compare int8 i4-vc` vs `compare int8 i4-s16`.

**Success criterion:** CPU gate: held-out layer-15 attention-output relative error falls from about 0.068 to <= 0.035 with the step 2 quantizer. GPU vs reference >= 0.9999 with mu set. Judgment mean KL vs int8 falls by more than the int8-vs-int8 floor from step 3. Otherwise abandon.

**Expected gain:** Real data shows layer 15 with a held-out value mean share of 0.79 and output error down about 2.4x. Zero per-token bytes, a 1-2 KiB table per layer, no attention time.

## Dropped

- P2 / K3 key centering: real held-out key mean share is 0.03-0.17 (mean 0.095) from INT8 cache pages. Replay KL gain on top of P1 is 7-9%, below the proposal's own 20% go threshold. With kq queries, centering made turbo C4/LM4 KL worse in 45-47 of 64 heads. The value-centering part survives as step 7.
- K2 MSE scale search: overlaps step 2 on the same store lines. P1 reaches lower key error on real keys (0.1036 vs 0.1086), and a per-row search over clip factors on top of P1 adds only 3-4%. Revisit only if real-query data favours it.
- 16-level midrise codes with a half-step offset (P4a, K3 option): within ±5% of P1, but needs sum(q) in the softmax and sum(p*s) through partials and reduce.
- Stochastic rounding (P4c): about 2x worse KL.
- Per-128 scales and group-size variants (P4b, K5c): about 10-18% KL on top of P1, but needs two scales, a split QK and split PV scaling. Deferred, not dominated. Record the numbers in TURBOQUANT.md section 11.
- QJL 1-bit residual (K5a): reconstruction error is pi/2 times the 3-bit residual's (5.7x LM4 MSE). KL is 8-13x worse.
- turbo K4/V4 format (K5d): same bytes as turbo-int4 at 2.2-2.4x the attention time. Reopen only if turbo dequant comes within about 1.1x of INT8.
- k8v4 / turbo-k8v4 mixed formats (P3, K4): cut usable context from about 172K to about 107K, against the user's main goal. turbo-int4 already scores needles 9/9 and judgment top-1 6/6. The output gain on real KV is only 1.2-1.3x over P1. Parked; the trigger is a step 3 judgment KL or needle failure above the int8 noise floor.
- Speed-P1 as written (environment, not the persistent cache): the baseline int8 and turbo-int4 runs were both on macOS 26, and the cited CPU-build note does not exist. Replaced by the 2x2 suite in step 3.
- Speed-P2 optional write-behind gate: build only if step 3 shows pc1 more than 5% slower.
- Speed-P3 as a separate item: folded into the step 1 deltas. Part (2), rotation re-rounding, is settled at about 1% of the int4 noise.
- Speed-P4 query-rotation fusion and padded-verify trimming: at most 0.05-0.1% gain. Keep only the design note that the inverse rotation cannot fold into the out-projection because of the sigmoid gate in the unrotated basis (prefill/attention_qkv.metal:33-56).
- Speed-P5 turbo prefill transcode: effort L for a non-default format that would still have 2x verify attention and only +10.6% context over turbo-int4.
- S1: already applied (stack.env:33 is turbo-int4; serve-splash.sh:22 matches turbo*). The remaining doc and doctor pieces are in step 4.
- Homebrew fallback bug (serve-splash.sh): already fixed at line 22 (`== turbo*`). Nothing to do.
- S5 serve.sh dispatcher: marginal. Fold BACKEND auto-detection into a later pass. stop.sh is kept in step 4.
- S6 full docs restructure and deleting TURBOQUANT_PLAN.md: the workspace root has no git, the memory note references the plan file, and the files are still being edited. Only the factual drift fixes (step 4) go ahead. Archive the plan instead of deleting it when this is done.

## Open questions

- Real query statistics: the kq and wq proxies disagree. Norm correction cuts KL 46% under kq but 2% under wq; centering helps 13% under wq but hurts under kq; smoothing gives 13% under wq. A one-off dump of rotated Q for 3 layers would settle smoothing and whether a norm-preserving rescale should follow the LS refit.
- Does --persistent-cache write-behind cost 10-20% of prefill, or is the regression macOS 27? Step 3 answers this.
- Does turbo-int4 hold needle and judgment quality at 150K-190K? Only up to 82K is verified, but dsh's window is 172K.
- Should turbo stay a documented option at all, given +10.6% context for 2.2-2.4x attention and 2x verify? This decides whether step 5 is worth its slot.
- Is a teacher-forced long-context logprob-KL part needed in kvbench? 24 judgments may still not resolve P1's gain end to end.
- Whether to also commit splash-src on turboquant-kv. The work is already published as `patches/0003`, under the user's configured git identity.

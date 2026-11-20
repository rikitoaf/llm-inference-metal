# Speed work handoff, 2026-10-28

Paused at the user's request because the session was ending. Read this file first, then `SPEED_PLAN.md`
("Plan, 2026-10-27") and `SPEED_LOG.md`.

## What is live in production

- `splash/splash-src` is the build the user runs, identical to `patches/0003`. It has:
  - the 16-node draft tree, decode width 1;
  - sampling T 0.6 with draft scale 0.35;
  - the step log (`SPLASH_STEP_LOG` → `logs/splash/steps.jsonl`, plus `serve.log`; report:
    `python3 splash/tools/step_report.py --since 1d --requests 20`). The log starts working the next time the user
    restarts Splash (the launcher relinks).
- dsh harness: in `config/dsh/qwen.cordis.patch.yml`, `tool-result-pruner` `thresholdChars: 16384` (applied
  2026-10-27). The pruner no longer rewrites old tool results, so the prefix cache survives
  (estimated −10-14% of long-session model time).
- Context cap for Splash: `SPLASH_EXTRA_ARGS="--max-context 180000"`. `CTX` only applies to llama.cpp.

## In progress: four exact speed changes, each in its own copy (not merged, not built, never run on the GPU)

| Copy | Change | Flag (default off) | State when paused |
|---|---|---|---|
| `splash/dev-park` | Keep the finished lane for the next agent turn (no re-prefill of the previous output) | `SPLASH_PARK_LANE` | 675 changed lines; NOTES written; adversarial review not run |
| `splash/dev-sysstate` | Cache state at the end of system prompt + tools | `SPLASH_SYSTEM_STATE` | 736 lines; NOTES written; review not run |
| `splash/dev-shortlist` | Draft head shortlist (ids < 98,304 + special tokens) | `SPLASH_DRAFT_HEAD_ROWS` | 741 lines; NOTES written; review not run |
| `splash/dev-attn` | decode-profile turbo-int4 + one 16-row verify tile (P4), barrier/rescale trims (P2/P5a) | `SPLASH_VERIFY_TILE` etc. | 903 lines; **no NOTES, unfinished** |

- The baseline for every copy is `splash/splash-dev` (= splash-src = patch 0003). See a change with
  `diff -ruN --exclude build --exclude .git splash/splash-dev splash/dev-X`.
- The workflow script that builds and reviews them:
  `~/.claude/projects/-Users-abdul-rakib-agentMania-dsh-qwen-jev/2995df64-4b21-4d90-8cfb-0d7faba3be1e/workflows/scripts/splash-exact-speed-impl-wf_9de6959e-59d.js`.
  It continues from partial work; re-run it, or run only the review stage for the three finished copies.
  Its run was stopped twice by session limits and once by the user.

**Update 2026-10-29:** the run was resumed (same run id wf_09fe4bfe-e1e).
- `dev-park`, `dev-sysstate` and `dev-shortlist` are implemented and adversarially reviewed. Fixes and open items are in the
  Review section of each NOTES file.
- `dev-attn` is implemented, and its review is being rerun (the first attempt hit a network error).
- Open from the reviews:
  - dev-park: needs a GPU oracle test (decode → park → adopt vs cold prefill).
  - dev-sysstate: bumps the protocol to v8, even with the flag off.
  - dev-shortlist: the >10% fallback looks only at `request.prompt`.

**Update 2026-10-29, later:**
- All four reviews are done.
- **`splash/dev-merged`** is `splash-dev` plus all four diffs. Its 3 conflicts, all park vs sysstate (includes, `TestConfig`,
  the status test), were resolved by hand.
- Build results:
  - `make all` succeeds, and every engine test target builds.
  - `rtc_probe` compiles every changed Metal file.
  - The server Python compiles with `.venv`.
- CPU tests:
  - `runtime-status` and `native-fd-transport` pass.
  - `kv-first-engine` fails 6 tests, the same 6 as the baseline `splash-dev`. They pass on pristine 1.2.1, so **our own
    patch 0003 broke them**: they assume 8 verify rows or several decoding lanes.
  - The binary aborts at its 2nd test, so ~80 tests had been hidden. With an all-failures harness, the new park and
    sysstate tests pass.
- Workflow `wf_39a8b1d5-fa5` is fixing those tests in `dev-merged`, with skeptics checking that no failure is a real
  engine regression.

- **Done (wf_39a8b1d5-fa5):** all 6 were stale test arithmetic, not engine bugs.
  - `Engine::prepare` reserves KV pages up to position + targetVerifyRows. With 16 rows a lane needs its next page 8
    tokens earlier (e.g. the 3-page yield is at 81, not 89).
  - Fixed in `dev-merged/dev/tests/engine/kv_first_engine_test.cpp`, with numbers derived from ExecutionLimits.
  - 6/6 skeptics agree: no assertion weakened, no production regression.
  - `kv-first-engine` passes in full. **Carry this test fix to splash-src** along with the winners (production's test
    is still broken).
  - Watch: `testSuspendedRequestWaitsForTheHostBesideOneInService` and `testRecoveryDrainDoesNotConsumeResourceWaitBudget`
    fail only at 8 rows / width 1, a configuration we don't ship.

**GPU session, 2026-10-29** (the user approved GPU runs). Logs are in `logs/speed/ab-1029/`.
- Unit tests (`splash/staged/speed/metal_unit.sh`; no offline `metal` compiler, so the .air-based tests are skipped):
  - These fail on the baseline too, so they predate this work: `tuning-workloads`, `linear-tuning`, `linear-plan`,
    `draft-attention`.
  - `paged-attention-plan` failed only in dev-attn. Fixed by wf_80d2dff2-5e7 and checked SAFE:
    - With flags off, the kernels are token-identical to the baseline.
    - 8x2 and 16 are byte-identical to 8-row on this GPU.
    - P2 (no page-end barrier) fails under validation: not safe, dropped.
- `splash/dev-merged` was rebuilt from all four changes, with dev-attn fixed. `paged-attention-plan` passes at default,
  8x2 and 16, and the CPU engine tests pass. (`dev-merged3` = the same without dev-attn; `dev-merged.old` = the broken
  first merge.)
- The first end-to-end A/B attempt failed:
  - 3 of 4 servers did not start (memory reserve: ~12 GiB reclaimable, with WebStorm 3.5 GiB and pgAdmin open).
  - `scripts/stop.sh` did not stop servers from dev copies. **Fixed**: it now matches any `splash/*/.venv` server.
- Ready to run once ~16 GiB is free:
  - `splash/staged/speed/merged_ab.sh` (it now waits for memory and retries);
  - `park_check.py` and `system_state_check.py`;
  - `attn_profile.sh`.

## Next steps

1. Finish `dev-attn`, then the adversarial review of all four. Read each NOTES file. Build in the copy and
   `rtc_probe` the changed Metal; no GPU.
2. Merge the four into `splash/splash-dev`, then `make all` plus the engine tests (compile only).
3. **Ask the user for a GPU window.** It needs ~16 GiB free, AC power and about 30-45 minutes.
   1. Unit tests: `paged-attention-plan`, `gdn-decode`, `target-sampling`, `dflash-batch-control`,
      `correctness.sh`.
   2. `splash/staged/speed/tree_ab.sh` and greedy identity, each flag on/off against the baseline.
   3. Turbo-int4 decode-profile at 2K/32K/64K, for P4.
   4. A scripted multi-turn session, for park lane and system state: TTFT per turn.
4. Copy only the winners into `splash-src`, re-export `patches/0003`, and log them in SPEED_LOG.

## Corrected estimates (from the critique; skeptic and completeness agents)

- Park lane −3.5 to −6% wall.
- System state ~20 s per new session or compaction.
- 16-row attention tile: 0 to −2.5%, more at long context.
- Shortlist −0.5 to −1%.
- Copy extension past depth 7: only +2.5-4% tokens/cycle (the first estimate counted the gain twice). It also needs 8
  more uniforms per lane for exact T>0 and the `accepted > 7` check in `Runtime.mm` lifted. Deferred.
- Real sessions run ~12-16 ms/cycle slower than the benchmark, environment suspected. Check the step log after
  the user's next restart.

## Rules the user set

- Ask before any GPU run.
- Never touch Defender.
- Never empty the Trash.
- Build in copies; production stays untouched until measured.

**GPU session 2, 2026-10-29 14:30-15:05, stopped by the user (needed the GPU).** `splash/staged/speed/run_session_1029.sh`,
log `logs/speed/ab-1029/session2.txt`, results `logs/speed/ab-1029/e2e2/`.
- base (splash-dev): agent 49.71 tok/s at 4.455 tokens/cycle; T=default 68.7-73.3 tok/s, ~90 ms/cycle.
- off (dev-merged, all flags off): greedy identical 6/6 with base, but agent 44.04 tok/s (-11.4%), T=default 52-64 tok/s,
  ~105 ms/cycle (all GPU time). The flags-off code paths look equivalent on reading, so heat or Defender (~190% CPU
  throughout) is suspected, but it is unconfirmed: the base2 rerun never ran. Next time run base and off back to back
  (both orders) before anything else.
- Not run: head, tile8x2, tile16, base2, the park/sysstate checks, attn_profile.
- `wait_quiet` times out at 15 min per config while Defender is busy, so each config takes ~20 min.

**GPU session 3, 2026-10-30 08:30-~10:45, stopped by the user.** Logs in `logs/speed/ab-1030/` (summary.txt, park/,
sysstate/, profile.txt, session*.txt). Defender quiet (mean 2-11%) throughout.
- A-B-B-A: base 50.64/51.69, off 51.43/51.33 agent tok/s; greedy identical everywhere. **No regression**: the 10-29 -11%
  was Defender (~190% CPU). Noise is about ±2%.
- head (SPLASH_DRAFT_HEAD_ROWS=98304): 52.20 agent tok/s (+1.7% vs the mean of 5 baseline runs, 51.35), T=default
  84.3 ms/cycle vs 85.4-88.3: better than every baseline run, greedy identical. **Winner candidate.**
- tile16 50.47, tile8x2 50.06: no gain at short context. Profile at 2K: tiles 8/8x2/16 at 92.9/93.2/92.8 ms, digests
  identical. 32K: only tile8 measured (99.6 ms) before the stop; 64K not run. Resume with
  `splash/staged/speed/attn_profile.sh logs/speed/ab-1030/profile 32768 65536` (now executable).
- sysstate: exact (greedy identical), but no gain on a server with free memory: the ordinary prefix cache already
  reuses the 9,312-token system block (second session 0.21 s with the flag off too). Value only under eviction or
  after a restart, unmeasured.
- park: safe (identical), but 0 adoptions / 5 mismatches because park_check sent no reasoning back and cut turns at 300
  tokens. park_check.py is fixed (reasoning_content + preserve_thinking, 3000 tokens, finish reason); the rerun
  (`splash/staged/speed/run_session_1030b.sh`, its park part) did not run.

**2026-11-01: shortlist applied to production** (user request).
- `splash/splash-src` += the dev-shortlist diff (splash-dev → dev-shortlist, 13 files) + the kv-first-engine test fix
  from dev-merged. Backup of production before: `splash/backups/splash-src-pre-shortlist-20261101/tracked.diff`
  (restore: `git checkout . && git apply` it, in splash-src). `make all` OK.
- On by default: `config/stack.env` `SPLASH_DRAFT_HEAD_ROWS=98304`, exported by `scripts/serve-splash.sh` (also printed
  in serve.log's start line). Off: `SPLASH_DRAFT_HEAD_ROWS=0`.
- splash-dev is left as the old baseline (= production before) for the A/B.
- Pending: GPU confirmation (A-B-B-A splash-dev vs splash-src), unit tests, re-export patches/0003, SPEED_LOG.
- `patches/0003` re-exported (git diff of splash-src, new files added with `git add -N`; 6493 lines, 6 new files).
  README tree entry updated. GPU test session running: `splash/staged/speed/run_session_1101.sh`, log
  `logs/speed/ab-1101-session.txt`, results `logs/speed/ab-1101/`.
- **Done 2026-11-01:** GPU A-B-B-A confirms it: agent prompts 51.49 → 52.83 tok/s (+2.6%), T=default cycle
  85.6 → 84.65 ms, greedy identical 6/6 ×3. Unit tests as expected. kv-first-engine: the dev-merged test file did not
  build in production (it has the park/sysstate tests); replaced by production's file + only the 6 stale-test fix
  hunks; passes. patches/0003 re-exported (6296 lines). SPEED_LOG entries for 10-30 and 11-01 written.
- The user's running Splash picks the shortlist up at its next restart (`scripts/serve-splash.sh`).
- **Next:** 32K/64K attention profile (`splash/staged/speed/attn_profile.sh logs/speed/ab-1030/profile 32768 65536`),
  park_check rerun (park part of `run_session_1030b.sh`), then decide tiles/park/sysstate.
- 2026-11-01 later: started 32K/64K profile + park rerun: run_session_1030b.sh (now O=logs/speed/ab-1101), log logs/speed/ab-1101/session-b.txt
- 2026-11-01: **verify tiles rejected** (32K: 102.8/107.8/105.7 ms, 64K: 114.8/129.0/125.8 for 8/8x2/16; digests equal). SPEED_LOG updated.
- shortlist isolated in decode-profile 2K: 85.82 → 84.52 ms (−1.3 ms), digest identical.
- **Park rerun 2026-11-01:** adoption works (turn 1 prompt 1.90 → 0.67 s; 6-turn prompt total 8.8 → 6.7 s) but turn 1's
  greedy text diverges at char 127/643 vs flag off. Blocked on the oracle test (park→adopt vs cold prefill, logit
  margin at the first differing token). Details in SPEED_LOG 2026-10-30 entry.
- Session state: nothing running; production = splash-src with the shortlist (user restarts Splash to use it).
- **Open list now:** (1) park oracle test; (2) sysstate only matters under eviction: test with memory pressure or
  after restart with the persistent cache; (3) tiles rejected; (4) new ideas: SPEED_PLAN.md.

**2026-11-02: user asked for more advanced techniques (gains so far felt small).** Research workflow wf_94d31d11-cf6
(5 angles: kernels, pipeline, speculation, sessions, external; then rank + 3 skeptics; read-only, no GPU). Report goes to
`splash/RESEARCH-ADVANCED-2026-11-02.md`. Resume if cut: Workflow scriptPath
`~/.claude/projects/-Users-abdul-rakib-agentMania-dsh-qwen-jev/2995df64-4b21-4d90-8cfb-0d7faba3be1e/workflows/scripts/splash-advanced-speed-research-wf_94d31d11-cf6.js`
with resumeFromRunId wf_94d31d11-cf6. Headroom noted: decode ~140 GB/s effective vs 41 ms floor at 307 GB/s.
- Research run 1 done (kernels angle failed: TLS). Report saved as `splash/RESEARCH-ADVANCED-2026-11-02.run1.md`;
  skeptics: park lane survives at ~1.5-4% wall (oracle test still required); copy extension past depth 7 REFUTED
  (an3.py's +7.3% double-counts copy runs; time-weighted +0.4-2.9%); compaction/replay-state work REFUTED (the 10-27
  pruner fix already recovers it; only SPLASH_SYSTEM_STATE's ~0.4-0.7% remains). Premises corrected: host gap is
  ~0.65 ms (not 2-3), verify projections are compute-bound (dequant serialised with the MMA, ~25 TFLOP/s matmul),
  SPLASH_DRAFT_TEMPERATURE_SCALE is inert in tree mode (sampling.metal:1480). Biggest levers left: int8 weight
  staging (+10-20% decode, microbench first) and reasoning effort xhigh→medium (approximate, 18-30%, needs eval).
  Rerun with the kernels angle: same run id (w1ro084tt).
- Research run 2 (with kernels angle) done 2026-11-02: report `splash/RESEARCH-ADVANCED-2026-11-02.md` (raw result
  `...run2-result.json`). New top: (1) reasoning effort xhigh→medium: dsh's reasoning setting never reaches Splash
  (pi-ai sends no reasoning_effort), so the template runs xhigh; +13-24% agent, approximate, needs eval; (2) copy
  branches in the 16-node heap (+9% tok/cycle simulated; run-1 skeptic found the older an3 estimator biased: recheck);
  (3) staged K-split retune: skeptic says +0.7-1.3% only; (5) lean IQ grid dequant E3 (exact, +5-6% decode) / E5;
  (6) per-cycle verify rows at long context; (7) park lane; (8) int8/e4m3/relaxed_precision matrix-unit probe (1 day,
  gates +10-40%). Two skeptics (copy, reasoning) hit the usage limit; resumed as w02k4j5br.
- Research complete (w02k4j5br): skeptics trimmed every estimate; consolidated table now in SPEED_PLAN.md
  "Plan, 2026-11-02". Awaiting the user's choice (GPU runs and approximate changes need their OK).

## Update 2026-11-02 — user decision: keep reasoning at xhigh/high (no medium)
- The user chose: "it should run xhigh or high". The template maps `high` to `xhigh` (qwen38 jinja line 60) and defaults
  to `xhigh`, so production already runs exactly that. Medium is dropped as an option; no config change.
- Running the accuracy + wall-time baseline at xhigh on production (shortlist on): `splash/staged/speed/run_eval_1102.sh`,
  t1-t6 x 3, JEV off, timeout 900 s. Results: `eval/results/xhigh-baseline-1102/results.jsonl`, log
  `logs/speed/eval-1102.txt`, Defender trace `defender.txt`, per-step data in `logs/splash/steps.jsonl`
  (step_report.py). Use it as the reference for any later change (runaway guard, lane park, dequant).
- Baseline finding (first 5 runs): t2-semver and t4-toposort fail 0/N because ONE reply reasons past max_tokens 16384
  (~310-340 s), and dsh then ends the run: dsh-agent-loop step() returns {kind:"max-tokens"} before any tool call,
  turn() closes unless a turn-stopping listener steers, headless exits 1. The starter stub stays untouched.
- Fix built (not in production): `plugins/dsh-plugin-runaway` (agent/assistant-stream finish max-tokens ->
  agent/turn-stopping steers "stop deliberating, act now", max 2 per turn), enabled only by
  `DSH_EXTRA_PATCH=config/dsh/runaway.cordis.patch.yml`. eval/run_eval.py got cond `guard` and a `runaway_continues`
  counter. Offline smoke test passes. `splash/staged/speed/run_eval_1102b.sh` runs guard x t1-t6 x 3 after the baseline
  -> eval/results/xhigh-guard-1102, log logs/speed/eval-1102b.txt. Adopt only after the user sees the A/B.
- USER DECISION 2026-11-02: "apply the guard to production if it improves". Criteria used: guard hidden pass rate >
  baseline, no task that passes in the baseline (t1/t3/t5) loses tests, mean wall not worse. If met: add the `runaway`
  insert from config/dsh/runaway.cordis.patch.yml to config/dsh/qwen.cordis.patch.yml (after the jev block), log in
  SPEED_LOG.md. If mixed: report numbers, do not apply.
- Guard v1 (text nudge only) FAILED live: t2 guard r0 fired 2x, the model reasoned to the 16K cap after each nudge
  (52K, 51K chars), then timed out at 900 s (baseline ended at ~300 s). Guard eval stopped after t1 (11/11, 184 s;
  slow from sampling variance: 22K chars of thinking vs 3-7K, guard never fired) and t2.
- Guard v2: same steer, plus the next request goes out with reasoningEffort "off" (agent/request waterfall; the Splash
  patch maps off -> reasoning_effort "none"), so the model must act on its plan; later steps run at high again.
  Offline test passes. Live probe on t2: eval/results/guard2-probe. Then rerun the full guard A/B (run_eval_1102b.sh
  minus the wait; OUT xhigh-guard-1102 must be cleared first).
- Guard v2 live probe (t2-semver): cut -> continue with thinking off -> 8/13 hidden, visible ok, 371 s (baseline 0/13
  every rep). Full v2 A/B started: eval/results/xhigh-guard2-1102, log logs/speed/eval-1102c.txt (v1 data moved to
  eval/results/xhigh-guard-v1-1102).

## STATE AT SESSION END 2026-11-02 (read this first next session)
- DONE: guard v2 in production (config/dsh/qwen.cordis.patch.yml 4b, plugins/dsh-plugin-runaway). Results above and in
  SPEED_LOG. Guard eval stopped at 13/18 runs (missing: t2-t6 rep 2); optional: finish with
  `python3 eval/run_eval.py --tasks t2-semver t3-inventory t4-toposort t5-ratelimit t6-mdtoc --conds guard --reps 1 --out eval/results/xhigh-guard2-1102b`.
  NOTE: production now loads the runaway plugin itself, so the eval cond `guard` adds the same insert a second time
  (config/dsh/runaway.cordis.patch.yml) — for future A/Bs, compare against a patch that disables id runaway instead.
- Restart dsh sessions to pick up the guard (no Splash restart needed). Splash is still running (production, shortlist on).
- PENDING GPU: E3 fp16 grid tables in splash/dev-e3 (code done + reviewed, est. +1-2% decode):
  `bash splash/dev-e3/dev/tools/e3_gpu_check.sh` (stop Splash first; ask the user before GPU runs).
- NEXT BIG LEVER (data-backed): E1 copy branches past depth 7 — real eval sessions: suffix match predicts 28.4% of output
  tokens, full depth in 32% of cycles, hybrid copy +9% tokens/cycle simulated (SPEED_LOG 2026-11-02 bottleneck entry).
- User decisions: reasoning stays xhigh/high (no medium). Ask before GPU runs.

## 2026-11-02 evening — plan "usable, then faster" (~/.claude/plans/clever-marinating-snowflake.md), user: "you can use gpu"
Part A done (no GPU):
- scripts/qwen: one entry point (qwen [dir] = web UI, qwen run, status, stop, start); starts the server when down
  (lib/system.sh ensure_model_server, console in logs/splash-console.log). dsh has no TUI template, so web = interactive.
- scripts/status.sh: server state/model/context/KV/SSD tier, speculation knobs, idle settings, guard/JEV, dsh window,
  memory, power, Defender.
- scripts/lib/system.sh: defender_cpu, reclaimable_gib, on_ac_power, duration_seconds, model_* helpers.
- Idle: stack.env SPLASH_IDLE_RELEASE (10m, passed as --idle-release) and SPLASH_IDLE_STOP (off; e.g. 30m) ->
  scripts/idle-stop.sh watchdog started by serve-splash.sh, activity = steps.jsonl mtime. NOT yet live-tested.
- dsh-env.sh keeps the last generated contextWindow when the server is down (was: silently 65536).
- doctor.sh: machine section (AC, Defender, reclaimable memory), dsh window = server limit - 20480, runaway guard
  loaded + imports. run-task.sh / web.sh: AUTOSTART=1.
- Config: stack.env notes draft scale inert in tree mode; qwen patch note on the splash override; README "Daily use"
  section + reasoning paragraph; norunaway.cordis.patch.yml; runaway.cordis.patch.yml now only re-enables (an `rm` was
  blocked by a safety check); eval cond "noguard" (nojev = production incl. guard).
Part B: plugin sets exit 0 for a rescued run whose last reply ended "stop" (process exit listener; offline-tested).
Loop detector (item 10) deferred: 1 true loop in 31 runs, the guard already catches it at the cap.
Part C: E3 GPU check running (logs/speed/e3-check.txt): correctness PASS (0 of 268M/537M groups differ), digest PASS
(323 production outputs bit-identical). Microbenchmark + decode-profile pending.
- 19:23 E3 APPLIED to production (decode-profile B1 86.43 -> 84.72 ms, -1.98%, bit-identical; SPEED_LOG). Patch 0003
  re-exported. Backup splash/backups/splash-src-pre-e3-20261102.
- Live tests: qwen start works (fixed: process check now also matches serve-splash.sh and the launcher's make phase);
  doctor all PASS with the server up (window 184305 = 204785 - 20480). Idle watchdog: the first one died silently
  (probably killed with the tool shell that started it); a direct 1m run stopped Splash correctly. It now ignores HUP and
  logs start/exit/signal to logs/idle-stop.log. Re-test in a normal terminal: SPLASH_IDLE_STOP=2m scripts/qwen start.
- 22:45 workflow wf_8140d005-ea1 (two tracks, GPU lock /tmp/splash-gpu.lock): A = splash/dev-e5 (fp16 grid in
  threadgroup memory, E5 int-operand stage, K-split retune); B = splash/dev-copy (copy extension past depth 7, ext16
  est. +3-5% tokens/cycle; a copy chain replacing tree nodes was rejected 10-27). Each: implement, 2 reviewers,
  fix+measure, verdict ADOPT/REJECT with a <track>_gpu_check.sh. Splash is stopped (GPU free for the tracks).

## STATE AT SESSION END 2026-11-02 ~23:40 (read first next session)
- Production now: shortlist + E3 fp16 grid tables (-1.98% decode) + runaway guard v2 (eval 45% -> 85%). Splash STOPPED,
  GPU free; workflow wf_8140d005-ea1 stopped, /tmp/splash-gpu.lock removed. Nothing half-applied to splash-src.
- Usability done: scripts/qwen, status.sh, idle-stop.sh (+ stack.env SPLASH_IDLE_RELEASE / SPLASH_IDLE_STOP),
  lib/system.sh, doctor checks, context-window keep-last, README "Daily use". To verify by the user: idle stop from a
  normal terminal (SPLASH_IDLE_STOP=2m scripts/qwen start; watch logs/idle-stop.log).
- Track A results in SPEED_LOG ("track A partial"): A1 and A2 rejected; next cheap exact win = A3 K-split tier change
  (~0.7%, est.) — tools in splash/dev-e5/dev/tools (e5_split_sweep.sh, e5_split_tiers.py, e5_gpu_check.sh).
- Track B (splash/dev-copy) unfinished and unreviewed; resume from its diff vs splash-src (diff -rq ... -x build -x
  install -x .venv). Realistic gain +3-5% tokens/cycle (ext16 sim), needs depth 8-15 uniforms + commit paths.
- Pending/optional: finish guard eval reps (eval cond is now default "nojev"; "noguard" turns the guard off); park-lane
  oracle; disk cleanup of old dev copies (ask the user; an rm was blocked once by a safety check).
- Uncommitted: everything since the initial import (commit only when the user asks).

## 2026-11-03 session (user: "start")
- K-split tier: splash/dev-split runtime/ops/LinearGguf.cpp kStagedTiers {{6,512},{11,2048}} -> only down (5120x17408)
  goes S2 -> S4 on 16 and 20 cores (checked by a python model of decodeSplits). Host-side only (decode-profile binary;
  same metallib). A-B-B-A x2 running: logs/speed/split-20261103-0025 (profile.txt, p<i>-<run>.txt; digest lines show
  greedy changes). Exact-math (K reassociation), not bit-identical.
- Track B relaunched as workflow wf_b5687af6-086 on splash/dev-copy (partial draft_copy_extend kernel, uniforms
  past depth 7 already there): implement -> 2 reviewers -> fix+measure, verdict ADOPT/REJECT, copy_gpu_check.sh.
  GPU lock /tmp/splash-gpu.lock (held by a waiter until the split run prints DONE).
- Split run 1 (logs/speed/split-20261103-0025) is INVALID for the decision: decode-profile without SPLASH_DRAFT_TREE=1
  runs the chain (and without head rows), not production. It showed split < base in every pair (ABBA2 88.79 -> 87.98,
  -0.9%; ABBA1 noisy warm-up). Rerun with SPLASH_DRAFT_TREE=1 SPLASH_DRAFT_HEAD_ROWS=98304, 25 cycles, GPU lock,
  greedy digests: logs/speed/split-tree-20261103-0032. NOTE for every future decode-profile: set those two env vars.
- 01:0x APPLIED K-split tier ({11, 4096}) to production: -0.92% decode, greedy digest identical (SPEED_LOG 2026-11-03).
  decode-profile prints a digest now (DECODE_PROFILE_B1_ONLY=1 + SPLASH_DRAFT_TREE=1 + SPLASH_DRAFT_HEAD_ROWS=98304).
  linear-plan test has a pre-existing stale failure (prefill storage rows 16 vs 8). Copy-extension workflow still
  running (wf_b5687af6-086).

## 2026-11-03 06:10 — copy extension APPLIED to production (SPLASH_DRAFT_COPY=1)
- Real dsh-session prompts, T 0.6: 4.97 -> 5.80 tokens/cycle (+16.7%), +0.3 ms/cycle; distribution tests pass;
  review clean. Details and rollback: splash/SPEED_LOG.md "2026-11-03 06:06". Off switch: SPLASH_DRAFT_COPY=0.
- Eval gate running: eval/results/copy-1103 (t1-t6 x2, cond nojev) vs eval/results/xhigh-guard2-1102 (138/163).
  If accuracy drops beyond noise or the server throws, set SPLASH_DRAFT_COPY=0 in config/stack.env and investigate.
- Speed so far since 11-02 start: E3 -2.0% ms, K-split -0.9% ms, copy +16.7% tokens/cycle.
- Leave Splash stopped when done (user wants the GPU fully off): scripts/stop.sh.
- 07:20: eval gate PASSED (84% vs 85%, task wall -17%); copy stays on. Real-agent copy gain is +4% tokens/cycle
  (session-prompt bench said +16.7%). Follow-ups in SPEED_LOG. Splash stopped (GPU off).

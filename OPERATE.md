# Operating the local stack (DSpine default), 2026-11-10

## Daily use
```bash
scripts/work ~/WebstormProjects
```
Starts Splash if it is down, refreshes the code graph, opens the dsh web UI. Engine (default since 2026-11-18):
`splash/dev-prefill-attn` = DSpine drafter + park lane + 8 verify rows from 64K + faster prefill attention.

```bash
scripts/qwen status        # engine, drafter, park lane, context cap, memory, power
```

```bash
scripts/qwen stop          # free the GPU; also needed before switching engine or flags
```

## Flags (`scripts/work [flags] [dir]`)
| flag | effect | status |
|---|---|---|
| (none) | DSpine + park lane + 8 verify rows from 64K + faster prefill attention (`splash/dev-prefill-attn`) | default since 2026-11-18: +24% tok/s at 68–103K |
| `--old-prefill` | previous prompt-processing attention kernel | fallback if anything looks off |
| `--full-verify` | 16 verify rows at every context length | fallback if long-context output looks off |
| `--no-park` | DSpine without the park lane | fallback if the park lane misbehaves |
| `--compact` | compact at ~56K instead of ~121K | optional; faster steps, more compactions |
| `--max-context 128K` | cap the server's context to leave host memory headroom (16.5 MiB per 1K tokens) | use when apps hold a lot of RAM |
| `--park` | (no-op: the park lane is the default) | kept so old commands still work |
| `--production` | old engine: `splash/splash-src`, DFlash2 | fallback only |

A flag change needs a server restart: `scripts/qwen stop`, then `scripts/work …`. `work` warns when the running
server is a different engine.

Same switches as environment variables (they also work with plain `scripts/qwen`):
`SPLASH_ENGINE=dspine|production`, `SPLASH_MAX_CONTEXT=128K`, `SPLASH_PARK_LANE=1` (with `SPLASH_SRC=splash/dev-park2`),
`DSH_SHORT_CONTEXT=1`, `WORK_AUTOINDEX=0`, `WORK_REPO_MAP=0`, `DSH_XERJ=1` (xerj search, off by default).
Port 3090 is the dsh web UI.

## Compaction (dynamic since 2026-11-13)
Compaction starts when the conversation reaches the **context window minus 10,000 tokens**. The window follows the
server: its context limit (`--max-context`, or what free memory allows) minus the 20K reply reserve. Examples:
`--max-context 150K` → window 133K → compact at 123K; `--max-context 100K` → window 82K → compact at 72K.
`DSH_COMPACT_MARGIN=20000` changes the 10K; `--compact` compacts early instead (37% of the window).
So to work in a faster, shorter context: `scripts/work --max-context 90K …`.

## Always on
- Compaction keeps the cached prefix (`SPLASH_NO_THINK_KEEPS_PREFIX=1`): ~300 s saved per compaction.
- Web preset: tool-result pruner off, 400-line reads, graph-first tool rule, fast glob/grep (dependency folders skipped: 8 s → 0.2 s).
- codebase-memory-mcp: 7 repos indexed; re-index 4 min after code stops changing (`scripts/work autoindex status`).
- Graph answers ~50% smaller (cbm-mcp-filter proxy: limit 10, no JSON-schema noise, `@<repo>` names); per-turn reminder adds a scope rule (questions get answers, not edits), a one-off scope check before the first edit of an analysis turn, and a batch-lookups nudge.

## Memory
DSpine uses ~0.4 GiB more than DFlash2. Splash sizes its context from the RAM free when it starts (about 150K for dsh
with apps closed). If the machine is tight (Edge, Teams, OneDrive open), start with `--max-context 128K` or close
apps before starting; `scripts/qwen status` shows the free memory.

## Park lane
```bash
scripts/qwen stop && scripts/work ~/WebstormProjects        # park lane is on by default; --no-park turns it off
```
Full A/B 2026-11-12 (`splash/dev-park2/dev/park-ab-2/summary.md`): DSpine at ~75–90K context, park lane on vs off,
every run faster:
- follow-up prompt processing: **−31% to −44%**;
- turn wall time: **−13% to −21%**;
- tokens processed fresh per turn: about half.

Correctness: generated text identical to a cold run in all three test scenarios; logits differ by a
summation-order amount (1.37 vs a 1.34 allowance, same top choice). A context cap is not needed. The more free RAM,
the more turns keep the saved state (close big apps for the full gain). If anything looks off, restart without
`--park`.

## Tests
```bash
scripts/auto-tests.sh              # CPU checks (~1 min)
```

```bash
scripts/auto-tests.sh --gpu        # + DSpine A/B, park lane A/B, CBM A/B, platform bench (AC power, GPU lock; long)
```
Results: `logs/auto-tests/<timestamp>/summary.md`.

## Where things are written
`splash/SPEED_LOG.md` (every change, newest last), `PLAN-2026-11-10.md` (what's next), `WORK.md` (CBM and bench),
`splash/SESSIONS-2026-11-10.md` (real-session analysis).

# Ready for work: dsh + local Qwen + codebase-memory-mcp

One command opens dsh on the local Qwen3.8-27B (Splash) with the codebase-memory-mcp (CBM) code graph connected:

```bash
scripts/work ~/WebstormProjects          # re-index the platform repos (~6 s, incremental), start Splash if down, open the web UI
```

```bash
scripts/work run ~/WebstormProjects/<repo> "trace where the portfolio filter becomes SQL"   # one task, headless
```

```bash
scripts/work index                       # only refresh the code graph (after pulling or switching branches)
```

```bash
scripts/qwen stop                        # free the GPU when done (idle stop also does it)
```

```bash
scripts/work autoindex status            # the watcher that re-indexes a repo 4 min after its code was last touched
```

`work` starts that watcher on its own; it exits when dsh exits. Set `WORK_AUTOINDEX=0` to turn it off and
`CBM_IDLE_SECONDS` to change the 4 minutes. For long sessions, `DSH_SHORT_CONTEXT=1 scripts/work …` compacts at
about 56K instead of about 121K (see `splash/SESSIONS-2026-11-10.md`).

### Engine flags (2026-11-10)
```bash
scripts/qwen stop                                  # switching engines needs a server restart
```

```bash
scripts/work --dspine ~/WebstormProjects           # recommended for long-context work: DSpine drafter
```

```bash
scripts/work --dspine --compact ~/WebstormProjects # + compact at ~56K (faster per step, more compactions)
```

```bash
scripts/work --park ~/WebstormProjects             # EXPERIMENTAL park lane (DFlash2; oracle passed, speed unmeasured)
```

No flag = production (DFlash2). `work` warns when the running server is a different engine than the flags ask for.
Every engine now has the compaction prefix fix (`SPLASH_NO_THINK_KEEPS_PREFIX=1`, default on).

Tip: `ln -s "$PWD/scripts/work" ~/.local/bin/work` so `work` runs from anywhere.

## What `work` turns on
| Toggle | Default in `work` | What it does |
|---|---|---|
| `DSH_CBM=1` | on | Adds CBM's 8 read-only "scout" tools: `search_graph`, `trace_path`, `get_code_snippet`, `get_file_outline`, `get_architecture`, `list_projects`, `index_status`, `check_index_coverage`. |
| `DSH_REPO_MAP=<md>` | on, if `workspace/platform-bench/repo-map/platform-map.md` exists (`WORK_REPO_MAP=0` turns it off) | Puts the platform "where to look" index in the system prompt as a cached section. |
| `DSH_NAV_DOCTRINE=1` | off (`WORK_NAV_DOCTRINE=1` turns it on) | Adds grep-then-range-read navigation rules. |

The same toggles work with plain `scripts/qwen`, for example `DSH_CBM=1 scripts/qwen <dir>`. Without any toggle,
`qwen` is the measured production setup.

Codebase-memory alignment: with `DSH_CBM=1` the model also gets a short, cached tool doctrine
(`plugins/dsh-plugin-guidelines/rules/code-graph.md`): graph queries before file reads, `project` = the absolute repo
path, fall back to grep when the graph has no answer. Long sessions then grow more slowly, which is what keeps decode
fast at long context.

## CBM facts (measured 2026-11-09, details in `splash/EXTERNAL-2026-11-09.md` section 3)
**Install:**
- Built from source, commit `612eee2` (MIT), into `~/.local/bin/codebase-memory-mcp`.
- Its `install` command was not run, so no client configs or hooks were written.

**Config:**
- Global: `auto_index false`, `auto_watch false`, `watcher_enabled false`.
- Environment, from `config/dsh/cbm.cordis.patch.yml`: `CBM_MEM_BUDGET_MB=1024`, `CBM_MAX_FILE_BYTES=1 MiB`, and
  `CBM_ALLOWED_ROOT=~/WebstormProjects`.
- The 1 MiB file cap is there because one repo holds 25–38 MB JSON blobs that pushed indexing to 2.4 GB.

**Costs:**
- Indexing (moderate mode): 1–2 s per repo, about 6 s for all five; peak about 263 MB; 149 MB on disk.
- Idle: about 8 MB RSS for the front end plus its daemon. Both exit with the session.
- Query latency: 12–30 ms warm.
- Schema: scout's 8 tools cost 9,263 chars, about 2.3K tokens of cached prompt, roughly 1% decode cost.

**Safety:**
- No network traffic seen.
- Repo `git status` is unchanged after indexing.
- Scout cannot index, so run `work index` (or `scripts/cbm-index.sh <repo>`) after big code changes.

**Not measured yet:** the A/B (pass rate, wall time, steps, adoption). How to run it:

```bash
splash/tools/cbm_ab.sh --eval-reps 1 --bench-reps 2 workspace/cbm-trial/ab-2   # takes the GPU lock, serves Splash, runs both arms
```

## Tests: everything rerunnable
```bash
scripts/auto-tests.sh                    # CPU checks: script syntax, toggle wiring (short-ctx, CBM), session tool share, bench validation
```

```bash
scripts/auto-tests.sh --gpu --quick      # + GPU: platform bench quick (3 tasks), DSpine A/B quick, CBM A/B quick (AC power, GPU lock)
```

Results: `logs/auto-tests/<timestamp>/summary.md`.

Platform benchmark (22 tasks from the platform doc, verified against the code; local only, in gitignored
`workspace/platform-bench/`):

```bash
python3 workspace/platform-bench/run_bench.py --validate                     # check the tasks and graders
```

```bash
python3 workspace/platform-bench/run_bench.py --serve --label base --reps 1  # baseline run (lock, serve, stop)
```

```bash
python3 workspace/platform-bench/run_bench.py --serve --label cbm --reps 1 --env DSH_CBM=1
```

The full manual is `workspace/platform-bench/MANUAL.md`: commands, adding tasks, reading results, A/B recipes.

## Where things stand
See `REPORT-2026-11-09.md`.

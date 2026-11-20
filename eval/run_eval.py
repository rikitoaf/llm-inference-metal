#!/usr/bin/env python3
"""A/B evaluation: does JEV make the agent's code more correct?

Each task in eval/tasks/<name>/ has a starter project the agent works on (with a few visible tests),
a task prompt, and hidden tests that check the whole spec. The agent never sees the hidden tests.
Every task runs with JEV on and with JEV off (config/dsh/nojev.cordis.patch.yml), alternating
which goes first, and each result is graded by the hidden tests. Cond "noguard" also turns the runaway-reply guard off
(config/dsh/norunaway.cordis.patch.yml). "nojev" is production: guard on since 2026-11-02 (older "nojev" results ran
without it; "guard" rows are that day's A/B). Cond "cbm" is "nojev" plus the codebase-memory-mcp scout tools (DSH_CBM=1);
the fresh workdir is indexed with scripts/cbm-index.sh before the agent starts (not counted in wall_s).

    python3 eval/run_eval.py --reps 2                 # all tasks, both conditions
    python3 eval/run_eval.py --tasks t2-semver --conds jev
    python3 eval/run_eval.py --summary eval/results/<run>/results.jsonl
"""
import argparse, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "eval" / "tasks"
WORK = ROOT / "workspace" / "eval"
NOJEV = ROOT / "config" / "dsh" / "nojev.cordis.patch.yml"
NOGUARD = ROOT / "config" / "dsh" / "norunaway.cordis.patch.yml"   # cond "noguard": JEV off, runaway guard off


def visible_tests(d):
    return sorted(p.relative_to(d) for p in d.rglob("test_*.py"))


def digest(d):
    return {str(p): hashlib.sha256((d / p).read_bytes()).hexdigest() for p in visible_tests(d)}


def unittest_counts(cwd, args):
    try:
        r = subprocess.run([sys.executable, "-m", "unittest", *args], cwd=cwd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:   # the agent's code hangs (e.g. an endless loop): nothing counts as passed
        return 0, 0, False
    out = r.stdout + r.stderr
    ran = int(m[1]) if (m := re.search(r"Ran (\d+) tests?", out)) else 0
    bad = sum(int(x) for x in re.findall(r"(?:failures|errors)=(\d+)", out))
    return ran, bad, r.returncode == 0


def grade(task, workdir):
    hidden = TASKS / task / "hidden" / "test_hidden.py"
    total = hidden.read_text().count("    def test_")
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(workdir, tmp, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        visible_ok = unittest_counts(tmp, ["-q"])[2]
        shutil.copy(hidden, tmp)
        ran, bad, _ = unittest_counts(tmp, ["-q", "test_hidden"])
    passed = ran - bad if ran == total else 0      # an import error runs 1 pseudo-test: count as 0
    return {"hidden_passed": passed, "hidden_total": total, "visible_ok": visible_ok}


def jev_events(stderr):
    return {
        "done_continue": len(re.findall(r"\[jev\] done-check: .*-> CONTINUE", stderr)),
        "done_stop": len(re.findall(r"\[jev\] done-check: .*-> stop", stderr)),
        "triage_hints": len(re.findall(r"\[jev\] triage ->", stderr)),
        "safety_blocks": len(re.findall(r"\[jev\] safety (DENY|ASK)", stderr)),
        "runaway_continues": len(re.findall(r"\[runaway\] .*-> continue", stderr)),
    }


def run_one(task, cond, rep, outdir, timeout):
    workdir = WORK / task
    shutil.rmtree(workdir, ignore_errors=True)
    shutil.copytree(TASKS / task / "starter", workdir)
    before = digest(workdir)
    env = dict(os.environ)
    env.pop("DSH_EXTRA_PATCH", None)
    env["JEV"] = "1" if cond == "jev" else "0"   # JEV is off by default in stack.env
    env["DSH_CBM"] = "1" if cond == "cbm" else "0"
    if cond == "cbm":   # scout cannot index; the copy above is a new tree, so refresh its graph first
        env["DSH_EXTRA_PATCH"] = str(NOJEV)
        subprocess.run([str(ROOT / "scripts" / "cbm-index.sh"), str(workdir)], check=True, capture_output=True)
    if cond == "nojev":
        env["DSH_EXTRA_PATCH"] = str(NOJEV)
    elif cond == "noguard":   # the guard is in production (qwen.cordis.patch.yml 4b) since 2026-11-02
        env["DSH_EXTRA_PATCH"] = str(NOGUARD)
    prompt = (TASKS / task / "task.txt").read_text().strip()
    log = outdir / f"{task}-{cond}-r{rep}"
    t0 = time.time()
    p = subprocess.Popen([str(ROOT / "scripts" / "run-task.sh"), str(workdir), prompt], env=env,
                         stdout=open(f"{log}.stdout", "w"), stderr=open(f"{log}.stderr", "w"), start_new_session=True)
    try:
        p.wait(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait()
        timed_out = True
    wall = round(time.time() - t0, 1)
    after = digest(workdir)
    res = {"task": task, "cond": cond, "rep": rep, "profile": env.get("MODEL_PROFILE", "qwen"),
           "workdir": str(workdir), "started_at": round(t0, 1), "wall_s": wall, "timed_out": timed_out, "exit": p.returncode,
           "tests_tampered": before != after, **grade(task, workdir),
           **jev_events(Path(f"{log}.stderr").read_text(errors="replace"))}
    return res


def summary(rows):
    profiles = sorted({r.get("profile", "qwen") for r in rows})
    print(f"model profile: {', '.join(profiles)}" + ("  (mixed: the rows below are not one model)" if len(profiles) > 1 else ""))
    by = defaultdict(list)
    for r in rows:
        by[r["cond"]].append(r)
    print(f"\n{'condition':<8} {'runs':>4} {'hidden pass':>12} {'all-pass runs':>14} {'visible ok':>11} {'mean wall':>10} {'tampered':>9} {'timeouts':>9}")
    for cond in ("jev", "nojev", "guard", "noguard", "cbm"):
        rs = by.get(cond, [])
        if not rs:
            continue
        hp = sum(r["hidden_passed"] for r in rs) / sum(r["hidden_total"] for r in rs)
        allp = sum(r["hidden_passed"] == r["hidden_total"] for r in rs)
        print(f"{cond:<8} {len(rs):>4} {hp:>11.1%} {allp:>9}/{len(rs):<4} {sum(r['visible_ok'] for r in rs):>7}/{len(rs):<3}"
              f" {sum(r['wall_s'] for r in rs) / len(rs):>9.0f}s {sum(r['tests_tampered'] for r in rs):>9} {sum(r['timed_out'] for r in rs):>9}")
    conds = [c for c in ("jev", "nojev", "guard", "noguard", "cbm") if by.get(c)]
    print(f"\n{'task':<14} " + "  ".join(f"{c + ' (hidden, wall)':>22}" for c in conds))
    for task in sorted({r["task"] for r in rows}):
        cells = []
        for cond in conds:
            rs = [r for r in rows if r["task"] == task and r["cond"] == cond]
            cells.append(" ".join(f"{r['hidden_passed']}/{r['hidden_total']},{r['wall_s']:.0f}s" for r in rs) or "-")
        print(f"{task:<14} " + "  ".join(f"{c:>22}" for c in cells))
    ev = [r for r in rows if r["cond"] == "jev"]
    if ev:
        print(f"\nJEV interventions over {len(ev)} runs: done-check pushed back {sum(r['done_continue'] for r in ev)}x, "
              f"triage hints {sum(r['triage_hints'] for r in ev)}, safety blocks {sum(r['safety_blocks'] for r in ev)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=sorted(p.name for p in TASKS.iterdir() if p.is_dir()))
    ap.add_argument("--conds", nargs="*", default=["jev", "nojev"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--first-rep", type=int, default=0, help="rep number to start at (resume a run)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="")
    ap.add_argument("--summary", default="")
    a = ap.parse_args()
    if a.summary:
        return summary([json.loads(l) for l in open(a.summary)])
    outdir = Path(a.out) if a.out else ROOT / "eval" / "results" / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(a.first_rep, a.first_rep + a.reps):
        for i, task in enumerate(a.tasks):
            conds = a.conds if (rep + i) % 2 == 0 else list(reversed(a.conds))   # alternate who goes first
            for cond in conds:
                r = run_one(task, cond, rep, outdir, a.timeout)
                rows.append(r)
                with open(outdir / "results.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"[{len(rows)}] {task} {cond} r{rep}: hidden {r['hidden_passed']}/{r['hidden_total']} "
                      f"visible_ok={r['visible_ok']} wall={r['wall_s']}s tampered={r['tests_tampered']} "
                      f"timeout={r['timed_out']} jev={r['done_continue']}c/{r['triage_hints']}t/{r['safety_blocks']}s", flush=True)
    summary(rows)
    print(f"\nresults: {outdir / 'results.jsonl'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Helpers of ab_dspine.sh (DSpine stage 2 checks and A/B), each writing files for later analysis.

  ab_tools.py digests LOG                          -> one line per prompt: its digest
  ab_tools.py compare-digests LOG_A LOG_B          -> exit 0 when every prompt's digest is equal
  ab_tools.py parse-run LOG ARM SEED CSV           -> appends the per-prompt rows of a decode-run log to CSV
  ab_tools.py tie-prepare PROMPTS OUT_A OUT_B SCORE JSON
        first divergence of each prompt's two greedy outputs -> SCORE (decode-run --score lines) and JSON
  ab_tools.py tie-report JSON SCORE_LOG MARGIN     -> exit 0 when every divergence is a near-tie (|margin| < MARGIN)
  ab_tools.py dist-compare OUT_A OUT_B POSITIONS JSON
        two-sample chi-square per output position of two sampled --repeat runs (exact sampling: equal laws)
  ab_tools.py summarize CSV OUTJSON                -> per-arm and per-prompt means (tokens/cycle, ms/cycle, tok/s)
"""
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict

PROMPT_LINE = re.compile(
    r"prompt (\d+): (\d+) prompt tokens, (\d+) tokens in (\d+) cycles \(([\d.]+) per cycle\), gpu ([\d.]+) ms mean "
    r"([\d.]+) median per cycle, wall ([\d.]+) ms per cycle, (\d+) cycles past a draft block, digest ([0-9a-f]+)")


def prompt_rows(log):
    return [m for m in (PROMPT_LINE.search(line) for line in open(log)) if m]


def digests(log):
    return [m.group(10) for m in prompt_rows(log)]


def cmd_digests(args):
    for d in digests(args[0]):
        print(d)


def cmd_compare_digests(args):
    a, b = digests(args[0]), digests(args[1])
    same = a == b and len(a) > 0
    print(json.dumps({"prompts": len(a), "equal": sum(x == y for x, y in zip(a, b)), "pass": same}))
    sys.exit(0 if same else 1)


def cmd_parse_run(args):
    log, arm, seed, path = args
    new = not _exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arm", "seed", "prompt", "prompt_tokens", "tokens", "cycles", "tokens_per_cycle",
                        "gpu_ms_per_cycle", "gpu_ms_median", "wall_ms_per_cycle", "tokens_per_second", "digest"])
        for m in prompt_rows(log):
            tokens, cycles, wall = int(m.group(3)), int(m.group(4)), float(m.group(8))
            tps = tokens / (cycles * wall / 1000) if cycles and wall else 0.0
            w.writerow([arm, seed, m.group(1), m.group(2), tokens, cycles, m.group(5), m.group(6), m.group(7),
                        m.group(8), f"{tps:.3f}", m.group(10)])


def _exists(path):
    try:
        open(path).close()
        return True
    except OSError:
        return False


def _lines(path):
    return [list(map(int, l.split())) for l in open(path) if l.strip()]


def cmd_tie_prepare(args):
    prompts, out_a, out_b, score_path, json_path = args
    p, a, b = _lines(prompts), _lines(out_a), _lines(out_b)
    cases = []
    with open(score_path, "w") as f:
        for i, (prompt, x, y) in enumerate(zip(p, a, b)):
            n = min(len(x), len(y))
            k = next((j for j in range(n) if x[j] != y[j]), None)
            if k is None:
                cases.append({"prompt": i, "diverges": False})
                continue
            cases.append({"prompt": i, "diverges": True, "at": k, "a": x[k], "b": y[k]})
            f.write(" ".join(map(str, prompt + x[:k])) + " | " + f"{x[k]} {y[k]}\n")
    json.dump(cases, open(json_path, "w"), indent=1)


def cmd_tie_report(args):
    json_path, score_log, margin = args[0], args[1], float(args[2])
    cases = json.load(open(json_path))
    scores = [l for l in open(score_log) if l.startswith("score ")]
    it = iter(scores)
    worst = 0.0
    for case in cases:
        if not case["diverges"]:
            continue
        line = next(it)
        values = dict((int(t), float(v)) for t, v in re.findall(r" (\d+)=(-?[\d.]+)", line))
        case["logit_a"], case["logit_b"] = values.get(case["a"]), values.get(case["b"])
        case["margin"] = abs(case["logit_a"] - case["logit_b"])
        worst = max(worst, case["margin"])
    ok = all(not c["diverges"] or c["margin"] < margin for c in cases)
    result = {"prompts": len(cases), "identical": sum(not c["diverges"] for c in cases),
              "max_margin_at_divergence": worst, "threshold": margin, "pass": ok, "cases": cases}
    json.dump(result, open(json_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))
    sys.exit(0 if ok else 1)


def chi2_sf(x, dof):
    """Upper tail of the chi-square distribution (regularized gamma Q(dof/2, x/2))."""
    if dof <= 0:
        return 1.0
    a, z = dof / 2.0, x / 2.0
    if z <= 0:
        return 1.0
    if z < a + 1:  # series for P
        term = total = 1.0 / a
        n = a
        for _ in range(10000):
            n += 1
            term *= z / n
            total += term
            if term < total * 1e-14:
                break
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    # continued fraction for Q
    b, c, d = z + 1 - a, 1e300, 1 / (z + 1 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return h * math.exp(-z + a * math.log(z) - math.lgamma(a))


def two_sample_chi2(xs, ys, minimum=5):
    ca, cb = Counter(xs), Counter(ys)
    na, nb = len(xs), len(ys)
    cats, other_a, other_b = [], 0, 0
    for t in set(ca) | set(cb):
        if ca[t] + cb[t] >= minimum:
            cats.append((ca[t], cb[t]))
        else:
            other_a += ca[t]
            other_b += cb[t]
    if other_a + other_b:
        cats.append((other_a, other_b))
    if len(cats) < 2:
        return 0.0, 0, 1.0
    stat = 0.0
    for a, b in cats:
        total = a + b
        ea, eb = total * na / (na + nb), total * nb / (na + nb)
        stat += (a - ea) ** 2 / ea + (b - eb) ** 2 / eb
    dof = len(cats) - 1
    return stat, dof, chi2_sf(stat, dof)


def cmd_dist_compare(args):
    out_a, out_b, positions, json_path = args[0], args[1], int(args[2]), args[3]
    a, b = _lines(out_a), _lines(out_b)
    rows = []
    for t in range(positions):
        xs = [s[t] for s in a if len(s) > t]
        ys = [s[t] for s in b if len(s) > t]
        stat, dof, p = two_sample_chi2(xs, ys)
        rows.append({"position": t, "n_a": len(xs), "n_b": len(ys), "chi2": stat, "dof": dof, "p": p})
    tested = [r for r in rows if r["dof"] > 0]
    pmin = min((r["p"] for r in tested), default=1.0)
    bonferroni = min(1.0, pmin * max(1, len(tested)))
    result = {"samples_a": len(a), "samples_b": len(b), "positions": positions, "tested_positions": len(tested),
              "min_p": pmin, "bonferroni_min_p": bonferroni, "pass": bonferroni > 0.01, "rows": rows}
    json.dump(result, open(json_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    sys.exit(0 if result["pass"] else 1)


def cmd_summarize(args):
    rows = list(csv.DictReader(open(args[0])))
    by_arm = defaultdict(lambda: defaultdict(float))
    by_prompt = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for r in rows:
        arm = r["arm"]
        tokens, cycles = float(r["tokens"]), float(r["cycles"])
        gpu = float(r["gpu_ms_per_cycle"]) * cycles
        wall = float(r["wall_ms_per_cycle"]) * cycles
        for target in (by_arm[arm], by_prompt[r["prompt"]][arm]):
            target["tokens"] += tokens
            target["cycles"] += cycles
            target["gpu_ms"] += gpu
            target["wall_ms"] += wall
            target["runs"] += 1

    def finish(d):
        return {"runs": int(d["runs"]), "tokens": int(d["tokens"]), "cycles": int(d["cycles"]),
                "tokens_per_cycle": d["tokens"] / d["cycles"] if d["cycles"] else 0,
                "gpu_ms_per_cycle": d["gpu_ms"] / d["cycles"] if d["cycles"] else 0,
                "wall_ms_per_cycle": d["wall_ms"] / d["cycles"] if d["cycles"] else 0,
                "tokens_per_second": d["tokens"] / (d["wall_ms"] / 1000) if d["wall_ms"] else 0}

    result = {"arms": {a: finish(d) for a, d in by_arm.items()},
              "prompts": {p: {a: finish(d) for a, d in arms.items()} for p, arms in sorted(by_prompt.items(), key=lambda x: int(x[0]))}}
    json.dump(result, open(args[1], "w"), indent=1)
    print(json.dumps(result["arms"]))


def cmd_hybrid(args):
    """Per-prompt hybrid from measured arms: arm LOW below THRESHOLD prompt tokens, HIGH at or above, for each
    threshold; reports tokens/cycle, ms/cycle and tok/s pooled over the prompts and seeds of the CSV."""
    rows = list(csv.DictReader(open(args[0])))
    low, high = args[1], args[2]
    thresholds = [int(t) for t in args[3].split(",")]
    result = {}
    for threshold in [None] + thresholds:
        tok = cyc = wall = 0.0
        for r in rows:
            want = low if threshold is not None and int(r["prompt_tokens"]) < threshold else high
            if threshold is None:
                want = low
            if r["arm"] != want:
                continue
            tok += float(r["tokens"]); cyc += float(r["cycles"]); wall += float(r["wall_ms_per_cycle"]) * float(r["cycles"])
        key = f"all-{low}" if threshold is None else f"T={threshold}"
        result[key] = {"tokens_per_cycle": tok / cyc, "wall_ms_per_cycle": wall / cyc, "tokens_per_second": tok / (wall / 1000)}
    print(json.dumps(result, indent=1))


COMMANDS = {"hybrid": cmd_hybrid, "digests": cmd_digests, "compare-digests": cmd_compare_digests, "parse-run": cmd_parse_run,
            "tie-prepare": cmd_tie_prepare, "tie-report": cmd_tie_report, "dist-compare": cmd_dist_compare,
            "summarize": cmd_summarize}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(2)
    COMMANDS[sys.argv[1]](sys.argv[2:])

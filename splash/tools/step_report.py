#!/usr/bin/env python3
"""Where Splash runs well and where it does not, from the usage logs serve-splash.sh keeps (logs/splash/):
steps.jsonl (one line per prefill chunk and decode cycle, SPLASH_STEP_LOG) and serve.log (the server's console).

    python3 splash/tools/step_report.py                    # everything logged
    python3 splash/tools/step_report.py --since 2h         # the last two hours (or 30m, 1d)
    python3 splash/tools/step_report.py --requests 20      # also the last 20 requests, one line each

Decode, by context size: tokens per cycle (DFlash's acceptance), ms per cycle and tok/s. The draft tree's share:
cycles whose accepted path used a node that was not its parent's best child (rank > 0), and the tokens those
cycles gained over the chain the tree's best children form. A full depth (7 drafted tokens accepted) says a
deeper draft would have paid. Prefill: rows per second by context, and per request how much of the prompt the
prefix cache served. Slow cycles: decode cycles over 1.5x their context bucket's median (Microsoft Defender's
scans and thermal throttling show here). Errors: the server's Error lines.
"""
import argparse, json, re, statistics, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs" / "splash"
BUCKETS = [(0, 8_000), (8_000, 32_000), (32_000, 64_000), (64_000, 128_000), (128_000, 10**9)]


def bucket_name(lo, hi):
    return f"{lo // 1000}K-{hi // 1000}K" if hi < 10**9 else f"{lo // 1000}K+"


def bucket_of(position):
    for lo, hi in BUCKETS:
        if lo <= position < hi:
            return bucket_name(lo, hi)
    return "?"


def since_ms(text):
    if not text:
        return 0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([mhd])", text)
    if not m:
        raise SystemExit("--since takes e.g. 30m, 2h, 1d")
    seconds = float(m[1]) * {"m": 60, "h": 3600, "d": 86400}[m[2]]
    return (time.time() - seconds) * 1000


def load_steps(path, start_ms):
    steps = []
    for name in (path.with_suffix(".jsonl.1"), path):
        if not name.exists():
            continue
        for line in open(name):
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if s.get("t", 0) >= start_ms:
                steps.append(s)
    return steps


def pct(part, whole):
    return f"{100 * part / whole:.0f}%" if whole else "-"


def table(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  " + "  ".join(str(v).rjust(w) for v, w in zip(r, widths)))


def decode_report(decodes):
    print(f"\n== Decode: {len(decodes)} cycles, {sum(d['retained'] for d in decodes)} tokens")
    by = defaultdict(list)
    for d in decodes:
        by[bucket_of(d["position"])].append(d)
    rows = []
    for lo, hi in BUCKETS:
        cs = by.get(bucket_name(lo, hi))
        if not cs:
            continue
        tokens = sum(c["retained"] for c in cs)
        wall = sum(c["wall_ms"] for c in cs)
        rows.append([bucket_name(lo, hi), len(cs), tokens, f"{tokens / len(cs):.2f}",
                     f"{statistics.median(c['wall_ms'] for c in cs):.1f}", f"{tokens / wall * 1000:.1f}"])
    table(rows, ["context", "cycles", "tokens", "tokens/cycle", "ms/cycle (median)", "tok/s"])
    # Acceptance: how many drafted tokens were kept per cycle.
    hist = defaultdict(int)
    for d in decodes:
        hist[d["accepted"]] += 1
    print("  accepted drafted tokens per cycle: " +
          "  ".join(f"{k}:{pct(hist[k], len(decodes))}" for k in sorted(hist)))
    full = hist.get(7, 0)
    print(f"  full depth (all 7 drafted tokens kept): {pct(full, len(decodes))} of cycles"
          + ("  <- a deeper draft block would pay" if full > 0.25 * len(decodes) else ""))
    for label, sel in (("greedy", lambda d: d["temperature"] == 0), ("sampled", lambda d: d["temperature"] > 0),
                       ("constrained", lambda d: d["constrained"])):
        cs = [d for d in decodes if sel(d)]
        if cs:
            print(f"  {label}: {len(cs)} cycles, {sum(c['retained'] for c in cs) / len(cs):.2f} tokens/cycle")
    trees = [d for d in decodes if d.get("tree")]
    if trees:
        used = [d for d in trees if any(r > 0 for r in d["ranks"])]
        gained = 0
        for d in used:
            first = next(i for i, r in enumerate(d["ranks"]) if r > 0)
            gained += len(d["ranks"]) - first     # tokens past the point a best-child chain stops
        tokens = sum(d["retained"] for d in trees)
        depth = defaultdict(list)
        for d in trees:
            for i, n in enumerate(d.get("shape", [])):
                depth[i + 1].append(n)
        shape = " ".join(f"d{k}:{statistics.mean(v):.1f}" for k, v in sorted(depth.items()))
        print(f"  draft tree: {len(trees)} cycles; a non-best child accepted in {pct(len(used), len(trees))} of them, "
              f"{gained} tokens ({pct(gained, tokens)} of tree output) beyond the best-child chain")
        print(f"  tree nodes per depth (mean): {shape}")
    # Slow cycles: over 1.5x their bucket's median.
    slow = []
    for cs in by.values():
        median = statistics.median(c["wall_ms"] for c in cs)
        slow += [c for c in cs if c["wall_ms"] > 1.5 * median]
    if slow:
        print(f"  slow cycles (>1.5x their context's median): {len(slow)} ({pct(len(slow), len(decodes))}); "
              f"latest at {time.strftime('%m-%d %H:%M', time.localtime(max(s['t'] for s in slow) / 1000))}")


def prefill_report(prefills):
    print(f"\n== Prefill: {len(prefills)} chunks, {sum(p['rows'] for p in prefills)} rows")
    by = defaultdict(list)
    for p in prefills:
        by[bucket_of(p["position"])].append(p)
    rows = []
    for lo, hi in BUCKETS:
        ps = by.get(bucket_name(lo, hi))
        if not ps:
            continue
        rows_n = sum(p["rows"] for p in ps)
        # A chunk's batch may hold several requests; its time is shared in proportion to rows.
        wall = sum(p["wall_ms"] * p["rows"] / max(p["batch_rows"], 1) for p in ps)
        rows.append([bucket_name(lo, hi), len(ps), rows_n, f"{rows_n / wall * 1000:.0f}"])
    table(rows, ["context", "chunks", "rows", "rows/s"])


def requests_report(steps, count):
    per = defaultdict(lambda: {"prefill_rows": 0, "prefill_ms": 0.0, "prompt": 0, "tokens": 0, "cycles": 0,
                               "decode_ms": 0.0, "start": 0, "first_position": None})
    for s in steps:
        r = per[s["request"]]
        r["start"] = r["start"] or s["t"]
        if s["kind"] == "prefill":
            r["prefill_rows"] += s["rows"]
            r["prefill_ms"] += s["wall_ms"] * s["rows"] / max(s["batch_rows"], 1)
            r["prompt"] = s["prompt"]
            if r["first_position"] is None:
                r["first_position"] = s["position"]
        else:
            r["tokens"] += s["retained"]
            r["cycles"] += 1
            r["decode_ms"] += s["wall_ms"]
    reqs = sorted(per.items(), key=lambda kv: kv[1]["start"])
    with_prompt = [r for _, r in reqs if r["prompt"]]
    if with_prompt:
        served = sum(r["prompt"] - r["prefill_rows"] for r in with_prompt)
        total = sum(r["prompt"] for r in with_prompt)
        print(f"\n== Requests: {len(reqs)}; the prefix cache served {pct(served, total)} of {total} prompt tokens")
    if count:
        rows = []
        for rid, r in reqs[-count:]:
            rows.append([rid, time.strftime("%H:%M:%S", time.localtime(r["start"] / 1000)), r["prompt"],
                         r["prompt"] - r["prefill_rows"], f"{r['prefill_ms'] / 1000:.1f}", r["tokens"],
                         f"{r['tokens'] / max(r['cycles'], 1):.2f}",
                         f"{r['tokens'] / r['decode_ms'] * 1000:.1f}" if r["decode_ms"] else "-"])
        table(rows, ["request", "start", "prompt", "cached", "prefill s", "output", "tokens/cycle", "tok/s"])


def serve_report(path, start_ms):
    if not path.exists():
        return
    errors, done = defaultdict(int), []
    day = None
    for line in open(path, errors="replace"):
        m = re.match(r"=== (\d{4}-\d{2}-\d{2}) ", line)
        if m:
            day = m[1]
            continue
        m = re.match(r"(\d\d:\d\d:\d\d) (Done|Error) · (.*)", line)
        if not m or not day:
            continue
        stamp = time.mktime(time.strptime(f"{day} {m[1]}", "%Y-%m-%d %H:%M:%S")) * 1000
        if stamp < start_ms:
            continue
        if m[2] == "Error":
            errors[m[3].split(" · ")[0]] += 1
        else:
            fields = dict(re.findall(r"(input|cached|output|TTFT) ([\d,.]+)", m[3]))
            done.append({k: float(v.replace(",", "")) for k, v in fields.items()})
    if done:
        ttft = sorted(d.get("TTFT", 0) for d in done)
        print(f"\n== Server: {len(done)} completed requests; TTFT median {statistics.median(ttft):.1f} s, "
              f"p90 {ttft[int(0.9 * (len(ttft) - 1))]:.1f} s, max {ttft[-1]:.1f} s")
    for kind, n in sorted(errors.items(), key=lambda kv: -kv[1]):
        print(f"  error {kind}: {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default=str(LOGS))
    ap.add_argument("--since", default="")
    ap.add_argument("--requests", type=int, default=0)
    a = ap.parse_args()
    logs = Path(a.logs)
    start = since_ms(a.since)
    steps = load_steps(logs / "steps.jsonl", start)
    if not steps:
        print(f"no steps in {logs / 'steps.jsonl'} (restart Splash with serve-splash.sh to start logging)")
    decodes = [s for s in steps if s["kind"] == "decode"]
    prefills = [s for s in steps if s["kind"] == "prefill"]
    if decodes:
        decode_report(decodes)
    if prefills:
        prefill_report(prefills)
    if steps:
        requests_report(steps, a.requests)
    serve_report(logs / "serve.log", start)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Helpers of splash/tools/prefill_ab.sh (SPLASH_PREFILL_ATTN_V2, splash/dev-prefill-attn/ATTN.md).

  prefill_ab_tools.py make-score OUT PROMPTS[:LINE[:TOKENS]] ...
        decode-run --score lines "prefix ids | 0" (one candidate = the whole vocabulary's logits): every line of
        PROMPTS, or line LINE (0-based) cut to its first TOKENS tokens
  prefill_ab_tools.py logits-compare OFF.bin CONTROL.bin ON.bin JSON
        whole-vocabulary fp32 logits of decode-run --logits-out, one block of 248,320 per score line:
        OFF = flag off, CONTROL = flag off with another prefill partition (--first-chunk), ON = flag on.
        Pass: every line's argmax(ON) == argmax(OFF), and the largest |ON - OFF| over all lines is within 1.5x the
        largest |CONTROL - OFF| (the summation-order drift production's own chunking already has).
"""
import json
import math
import sys
from array import array

VOCABULARY = 248320


def cmd_make_score(args):
    out, specs = args[0], args[1:]
    lines = []
    for spec in specs:
        parts = spec.split(":")
        rows = [l.split() for l in open(parts[0]) if l.strip()]
        if len(parts) > 1:
            rows = [rows[int(parts[1])]]
            if len(parts) > 2:
                rows = [rows[0][:int(parts[2])]]
        lines += [" ".join(r) + " | 0" for r in rows]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} score lines: " + ", ".join(str(len(l.split("|")[0].split())) for l in lines) + " tokens")


def blocks(path):
    data = array("f")
    with open(path, "rb") as f:
        data.frombytes(f.read())
    if len(data) % VOCABULARY:
        raise SystemExit(f"{path}: {len(data)} floats is not a whole number of {VOCABULARY}-logit blocks")
    return [data[i:i + VOCABULARY] for i in range(0, len(data), VOCABULARY)]


def compare(a, b):
    finite = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    max_abs = max(abs(x - y) for x, y in finite)
    dot = sum(x * y for x, y in finite)
    na = math.sqrt(sum(x * x for x, _ in finite)); nb = math.sqrt(sum(y * y for _, y in finite))
    return {"max_abs": max_abs, "cosine_distance": 1.0 - dot / (na * nb) if na and nb else 0.0}


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def cmd_logits_compare(args):
    off, control, on, out = blocks(args[0]), blocks(args[1]), blocks(args[2]), args[3]
    if not (len(off) == len(control) == len(on)) or not off:
        raise SystemExit(f"line counts differ or are zero: off {len(off)}, control {len(control)}, on {len(on)}")
    rows = []
    for i, (x, c, y) in enumerate(zip(off, control, on)):
        top = sorted(range(len(x)), key=x.__getitem__, reverse=True)[:2]
        row = {"line": i, "argmax_off": top[0], "margin_off": x[top[0]] - x[top[1]], "argmax_on": argmax(y),
               "argmax_control": argmax(c), "v2": compare(x, y), "control": compare(x, c)}
        row["argmax_same"] = row["argmax_on"] == row["argmax_off"]
        rows.append(row)
    v2_max = max(r["v2"]["max_abs"] for r in rows)
    control_max = max(r["control"]["max_abs"] for r in rows)
    ok = all(r["argmax_same"] for r in rows) and v2_max <= 1.5 * control_max
    result = {"lines": len(rows), "argmax_same": sum(r["argmax_same"] for r in rows), "v2_max_abs": v2_max,
              "control_max_abs": control_max, "ratio": v2_max / control_max if control_max else float("inf"),
              "pass": ok, "rows": rows}
    json.dump(result, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    for r in rows:
        print(f"  line {r['line']}: argmax off {r['argmax_off']} (margin {r['margin_off']:.3f}) on {r['argmax_on']} "
              f"control {r['argmax_control']}; max|on-off| {r['v2']['max_abs']:.4f} "
              f"max|control-off| {r['control']['max_abs']:.4f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    commands = {"make-score": cmd_make_score, "logits-compare": cmd_logits_compare}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(__doc__)
        sys.exit(2)
    commands[sys.argv[1]](sys.argv[2:])

#!/usr/bin/env python3
"""Fits SPLASH_VERIFY_ROWS_LONG's keep-mass threshold offline from a step log recorded with
SPLASH_VERIFY_ROWS_LONG=report (every cycle verifies 16 rows and logs row_p = each row's path probability by the
draft, path = the accepted path's rows).

    longctx_rows_fit.py STEPS.jsonl [...] [--tile-ms-per-100k 19.0] [--from 0]

For a threshold K, a cycle whose rows 8..15 hold a summed path probability below K would have verified 8 rows: it
saves one attention tile (tile-ms-per-100k x context / 100K, from the dispatch profile) and keeps only the accepted
tokens before the path's first row >= 8. The table gives, per context band and K, the share of cycles cut, tokens
per cycle, ms per cycle and the tok/s ratio against verifying 16 rows always. The estimate assumes the cycles after a
shortened one look like the recorded ones (they restart at the same text a few tokens earlier); the live A/B decides.
"""
import argparse
import json
import math
from collections import defaultdict

BANDS = [(0, 48), (48, 64), (64, 80), (80, 96), (96, 112), (112, 400)]
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, math.inf]


def cycles(paths):
    for path in paths:
        with open(path) as file:
            for line in file:
                try:
                    step = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if step.get("kind") == "decode" and "row_p" in step:
                    yield step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--tile-ms-per-100k", type=float, default=19.0,
                    help="GPU ms one 8-row verify attention tile costs per cycle at 100K context (16 layers)")
    ap.add_argument("--from", dest="start", type=int, default=0, help="only cycles at or past this context")
    ap.add_argument("--first-tail-row", type=int, default=8)
    args = ap.parse_args()
    table = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0, 0]))   # band -> K -> cycles, tokens, ms, cut
    for step in cycles(args.logs):
        position = step["position"]
        if position < args.start:
            continue
        band = next(b for b in BANDS if b[0] * 1000 <= position < b[1] * 1000)
        tail_mass = sum(step["row_p"][args.first_tail_row - 1:])            # row_p[0] is row 1
        emitted = step["retained"]
        tail_depth = next((d for d, row in enumerate(step["path"], 1) if row >= args.first_tail_row), None)
        emitted_cut = emitted if tail_depth is None else tail_depth        # accepted tail_depth - 1, + correction
        saved = args.tile_ms_per_100k * position / 100000.0 * (16 - args.first_tail_row) / 8.0
        for threshold in THRESHOLDS:
            cut = tail_mass < threshold
            for key in (band, (0, 400)):
                row = table[key][threshold]
                row[0] += 1
                row[1] += emitted_cut if cut else emitted
                row[2] += step["gpu_ms"] - (saved if cut else 0.0)
                row[3] += cut
    for band in BANDS + [(0, 400)]:
        if band not in table:
            continue
        base = table[band][0.0]
        print(f"\ncontext {band[0]}-{band[1]}K: {base[0]} cycles, {base[1] / base[0]:.3f} tok/cycle, "
              f"{base[2] / base[0]:.1f} ms/cycle, {1000 * base[1] / base[2]:.1f} tok/s with 16 rows")
        print("| keep mass K | cycles cut | tok/cycle | ms/cycle | tok/s | vs 16 rows |")
        print("|---|---|---|---|---|---|")
        for threshold in THRESHOLDS[1:]:
            n, tokens, ms, cut = table[band][threshold]
            print(f"| {threshold:g} | {100 * cut / n:.0f}% | {tokens / n:.3f} | {ms / n:.1f} | {1000 * tokens / ms:.1f} | "
                  f"{100 * ((tokens / ms) / (base[1] / base[2]) - 1):+.1f}% |")


if __name__ == "__main__":
    main()

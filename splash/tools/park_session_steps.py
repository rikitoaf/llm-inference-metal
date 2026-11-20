#!/usr/bin/env python3
"""Per-step prompt sizes and TTFT of a dsh session (session.v3.jsonl.zstd), for the park-lane real-task A/B:
each model step's input tokens, cache-read tokens, uncached tokens, time from step/start to the first streamed
chunk (TTFT: prefill + queueing) and from there to the end (decode), output tokens. Prints one line per step and
totals; --json writes the rows.

    python3 splash/tools/park_session_steps.py dsh-home/sessions/<cwd>/<session>/session.v3.jsonl.zstd [--json OUT]
    python3 splash/tools/park_session_steps.py --latest            # the newest session file under dsh-home/sessions
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = ROOT / "dsh-home" / "sessions"


def read_events(path):
    text = subprocess.run(["zstd", "-dc", str(path)], capture_output=True, check=True).stdout.decode(errors="replace")
    for line in text.splitlines():
        try:
            yield json.loads(line)
        except ValueError:
            continue


def step_rows(path):
    rows, step_started, step_number = [], None, 0
    for event in read_events(path):
        kind, data = event.get("type"), event.get("data", {})
        if kind == "step/start":
            step_started = event.get("time")
            step_number += 1
        elif kind == "assistant/message":
            usage = data.get("usage", {})
            stream = data.get("stream", [])
            first_chunk = next((c.get("time") or c.get("time0") for c in stream if c.get("time") or c.get("time0")), None)
            ended = event.get("time")
            rows.append({"step": step_number, "turn": data.get("turn"),
                         # dsh's inputTokens are the tokens the server prefilled; cacheReadTokens the ones it served
                         # from its prefix cache (or a parked lane); the prompt is their sum.
                         "prompt_tokens": usage.get("inputTokens", 0) + usage.get("cacheReadTokens", 0),
                         "cache_read_tokens": usage.get("cacheReadTokens", 0),
                         "uncached_tokens": usage.get("inputTokens", 0),
                         "output_tokens": usage.get("outputTokens", 0), "total_tokens": usage.get("totalTokens", 0),
                         "ttft_s": (first_chunk - step_started) / 1000 if step_started and first_chunk else None,
                         "decode_s": (ended - first_chunk) / 1000 if first_chunk and ended else None})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    path = Path(a.session) if a.session else None
    if a.latest or path is None:
        candidates = sorted(SESSIONS.rglob("session.v3.jsonl.zstd"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            sys.exit("no session files")
        path = candidates[-1]
    rows = step_rows(path)
    print(f"session {path}")
    print("step turn  prompt  cached uncached  output   ttft_s decode_s")
    for r in rows:
        ttft = f"{r['ttft_s']:.1f}" if r["ttft_s"] is not None else "-"
        decode = f"{r['decode_s']:.1f}" if r["decode_s"] is not None else "-"
        print(f"{r['step']:4d} {r['turn'] or 0:4d} {r['prompt_tokens']:7d} {r['cache_read_tokens']:7d} {r['uncached_tokens']:8d} "
              f"{r['output_tokens']:7d} {ttft:>8} {decode:>8}")
    ttft_total = sum(r["ttft_s"] or 0 for r in rows)
    decode_total = sum(r["decode_s"] or 0 for r in rows)
    summary = {"session": str(path), "steps": len(rows), "ttft_s": round(ttft_total, 1), "decode_s": round(decode_total, 1),
               "uncached_tokens": sum(r["uncached_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
               "peak_total_tokens": max((r["total_tokens"] for r in rows), default=0)}
    print(f"steps {summary['steps']}: TTFT total {ttft_total:.1f} s, decode total {decode_total:.1f} s, "
          f"uncached {summary['uncached_tokens']} tokens, output {summary['output_tokens']}, peak context {summary['peak_total_tokens']}")
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": summary, "steps": rows}, indent=1))


if __name__ == "__main__":
    main()

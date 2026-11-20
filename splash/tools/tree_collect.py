#!/usr/bin/env python3
"""Collect DFlash draft data for splash/tools/tree_potential.py: sends agent-like requests (greedy, thinking on) to a
Splash server started with SPLASH_DRAFT_DUMP=FILE, so FILE gets one line per decode cycle.

    SPLASH_DRAFT_DUMP=logs/speed/draft-dump.jsonl ./scripts/serve-splash.sh &   # then, once Ready:
    python3 splash/tools/tree_collect.py [--max-tokens 1200]
"""
import argparse, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:8000/v1/chat/completions"


def prompts():
    semver = (ROOT / "eval" / "tasks" / "t2-semver" / "starter" / "semver.py")
    source = semver.read_text() if semver.exists() else ""
    files = sorted((ROOT / "eval" / "tasks").glob("*/starter/*.py"))
    other = files[0].read_text() if files else ""
    return [
        ("code", "Write a Python function that parses ISO-8601 durations like P1DT2H3M4S into seconds, with tests."),
        ("reason", "A train leaves at 9:40 and travels 237 km at 82 km/h, stops 12 minutes, then travels 154 km at "
                   "96 km/h. When does it arrive? Explain your reasoning step by step."),
        ("edit", "Here is a module:\n\n```python\n" + source + "\n```\n\nImplement the unimplemented functions so "
                 "the docstring's spec holds. Reply with the whole updated file."),
        ("refactor", "Refactor this code for readability without changing behaviour, and return the full file:\n\n"
                     "```python\n" + other + "\n```"),
        ("explain", "Explain how a write-ahead log gives a database crash consistency, and what fsync has to do "
                    "with it."),
        ("tool", "You are a coding agent. The user asked: 'add a --verbose flag to the CLI in cli.py that prints each "
                 "file processed'. Plan the change, then show the exact edit as a unified diff."),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-tokens", type=int, default=1200)
    a = ap.parse_args()
    for name, content in prompts():
        body = {"model": "qwen3.8-27b-local", "messages": [{"role": "user", "content": content}],
                "max_tokens": a.max_tokens, "temperature": 0, "stream": False}
        request = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            r = json.load(response)
        t = r.get("timings", {})
        print(f"{name}: {t.get('predicted_n')} tokens, {t.get('predicted_per_second')} tok/s", flush=True)


if __name__ == "__main__":
    main()

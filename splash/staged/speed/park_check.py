#!/usr/bin/env python3
"""SPLASH_PARK_LANE on a running server: a multi-turn conversation shaped like dsh's agent turns (each prompt = the
previous prompt + the model's reply + a new tool-output-sized message), greedy. Prints per turn the prompt and cached
tokens, prompt_ms and the server's park_lane counters; --save/--compare check the turns' text against another run
(flag off vs on: adoption is the same math as one long request, up to summation order, so a late divergence on a
near-tie is possible and is reported with its position).

    python3 splash/staged/speed/park_check.py --save logs/speed/ab/park-off.json
    python3 splash/staged/speed/park_check.py --compare logs/speed/ab/park-off.json
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[3]
TASK = ("You are a coding agent working in a small Python repository. Fix the bug the user reports, one step at a "
        "time: say what you will look at next and why, in a few sentences, then stop and wait for the tool output.")


def tool_output(turn):
    files = sorted((ROOT / "eval" / "tasks").glob("*/starter/*.py"))
    text = files[turn % len(files)].read_text() if files else "x = 1\n" * 200
    return f"Tool output (read_file, turn {turn}):\n```python\n{text[:6000]}\n```\nContinue with the next step."


def call(path, body=None):
    request = urllib.request.Request(URL + path, None if body is None else json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--save")
    ap.add_argument("--compare")
    a = ap.parse_args()
    model = call("/v1/models")["data"][0]["id"]
    messages = [{"role": "system", "content": TASK},
                {"role": "user", "content": "Bug report: parsing '1.2.3-rc.1+build' gives the wrong precedence. "
                                            + tool_output(0)}]
    turns, prompt_s = [], 0.0
    for turn in range(a.turns):
        started = time.monotonic()
        r = call("/v1/chat/completions", {"model": model, "messages": messages, "temperature": 0, "seed": 0,
                                          "max_tokens": a.max_tokens,
                                          "chat_template_kwargs": {"enable_thinking": a.thinking,
                                                                           "preserve_thinking": True}})
        wall = time.monotonic() - started
        message, finish = r["choices"][0]["message"], r["choices"][0].get("finish_reason")
        text = message.get("content") or ""
        # As dsh sends it back (preserve_thinking): the turn's reasoning with its reply, so the next prompt starts
        # with the tokens the lane holds.
        reasoning = message.get("reasoning_content") or ""
        usage, timings = r["usage"], r.get("timings", {})
        park = call("/status").get("park_lane", {})
        prompt_s += timings.get("prompt_ms", 0) / 1000
        print(f"turn {turn}: prompt {usage['prompt_tokens']} cached {usage['prompt_tokens_details']['cached_tokens']} "
              f"prompt_ms {timings.get('prompt_ms', 0):.0f} output {usage['completion_tokens']} ({finish}) wall {wall:.1f}s "
              f"park {json.dumps(park)}", flush=True)
        turns.append(reasoning + "\x00" + text)
        reply = {"role": "assistant", "content": text}
        if reasoning:
            reply["reasoning_content"] = reasoning
        messages += [reply, {"role": "user", "content": tool_output(turn + 1)}]
    print(f"total prompt time {prompt_s:.1f}s")
    if a.save:
        Path(a.save).write_text(json.dumps(turns, indent=1))
    if a.compare:
        base = json.loads(Path(a.compare).read_text())
        for i, (x, y) in enumerate(zip(base, turns)):
            k = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            print(f"  turn {i}: identical={x == y}" + ("" if x == y else f", first divergence at char {k}/{len(x)}"))
        return 0 if base == turns else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

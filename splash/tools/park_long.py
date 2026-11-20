#!/usr/bin/env python3
"""SPLASH_PARK_LANE at long context, on a running server: a conversation that starts from a ~60-80K-token prefix
(source files and docs of the Splash checkout, rendered as read_file tool outputs; no company content) and then runs
agent-shaped turns: each prompt = the previous prompt + the model's reply (reasoning and text, as dsh sends them back)
+ a ~3K-token tool output. Greedy, thinking on, replies capped near 1,000 tokens.

Per turn it prints and records: prompt tokens, cached tokens, prompt_ms (the server's start-to-first-token time),
predicted_ms, output tokens, wall, and the deltas of /status park_lane (adoptions, saved_tokens, mismatches, reclaims,
replacements). --save/--compare check the turns' text against another run (flag off vs on: adoption is the same math
in a different summation order, so a divergence at a near-tie is possible and is reported with its position).

    python3 splash/tools/park_long.py --prefix-tokens 70000 --turns 6 --save OUT/park-off.json
    python3 splash/tools/park_long.py --prefix-tokens 70000 --turns 6 --compare OUT/park-off.json --save OUT/park-on.json

The prefix is built once per token target and cached under ~/.cache/splash-park/ (outside the repo); every run with
the same --prefix-tokens and --source sends byte-identical messages, so the server's prefix cache serves the prefix
itself from the second run on (turn 0 reports it); the turns' new tokens are what the park lane is about.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".cache" / "splash-park"
SYSTEM = ("You are a coding agent working in a C++/Metal inference engine repository. You answer in a careful "
          "technical register. At every step: say in about 600-900 words what the code you have just read does, "
          "what you would check next and why, then stop and wait for the next tool output. Do not repeat earlier "
          "steps verbatim.")
TASK = ("Task: find where the engine decides to prefill a prompt again instead of reusing cached state, and propose "
        "the smallest change that keeps a finished request's state for the next turn. Read the files as they come.")


def call(url, path, body=None, timeout=1800):
    request = urllib.request.Request(url + path, None if body is None else json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def count_tokens(url, text):
    return len(call(url, "/tokenize", {"content": text})["tokens"])


def source_files(source):
    """Public source and docs of a Splash checkout, largest first, deterministic."""
    patterns = ("runtime/engine/*.cpp", "runtime/engine/*.hpp", "runtime/model/*.mm", "runtime/model/*.cpp",
                "runtime/model/*.hpp", "runtime/metal/kernels/*/*.metal", "runtime/ops/*.cpp", "DEVELOPMENT.md",
                "server/*.py")
    files = []
    for pattern in patterns:
        files += sorted(source.glob(pattern))
    return [f for f in files if f.is_file()]


def tool_output(name, text, index):
    return f"Tool output (read_file {name}, chunk {index}):\n```\n{text}\n```\n"


def build_prefix(url, source, prefix_tokens, chunk_chars):
    """Messages whose rendered size is about prefix_tokens: the system prompt, the task, then assistant/tool pairs
    carrying file chunks. Cached by (source, prefix_tokens, chunk_chars)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{source}|{prefix_tokens}|{chunk_chars}".encode()).hexdigest()[:16]
    cached = CACHE / f"prefix-{prefix_tokens}-{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    tokens = count_tokens(url, SYSTEM + TASK)
    chunk_index = 0
    for path in source_files(source):
        text = path.read_text(errors="replace")
        for offset in range(0, len(text), chunk_chars):
            if tokens >= prefix_tokens:
                break
            chunk = tool_output(path.relative_to(source).as_posix(), text[offset:offset + chunk_chars], chunk_index)
            messages.append({"role": "assistant", "content": f"I will read the next chunk ({chunk_index}) and summarise it."})
            messages.append({"role": "user", "content": chunk})
            tokens += count_tokens(url, chunk) + 12
            chunk_index += 1
        if tokens >= prefix_tokens:
            break
    prefix = {"messages": messages, "tokens": tokens, "chunks": chunk_index, "next_file_index": 0}
    cached.write_text(json.dumps(prefix))
    return prefix


def turn_tool_output(url, source, turn, target_tokens):
    """A ~target_tokens tool output per turn, taken from the files after the prefix's chunks (deterministic)."""
    files = source_files(source)
    path = files[(-1 - turn) % len(files)]           # from the end of the list: not among the prefix's chunks
    text = path.read_text(errors="replace")
    chars = target_tokens * 3
    start = (turn * 7919) % max(1, len(text) - chars)
    chunk = tool_output(path.relative_to(source).as_posix(), text[start:start + chars], f"turn-{turn}")
    return chunk, count_tokens(url, chunk)


def server_status(url):
    try:
        return call(url, "/status")
    except Exception:
        return {}


def park_status(url):
    return server_status(url).get("park_lane", {}) or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--source", default=str(ROOT / "splash" / "dev-park2"), help="checkout the prefix is built from")
    ap.add_argument("--prefix-tokens", type=int, default=70000)
    ap.add_argument("--chunk-chars", type=int, default=9000, help="about 3K tokens per prefix chunk")
    ap.add_argument("--tool-tokens", type=int, default=3000, help="tool output size per turn")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--label", default="")
    ap.add_argument("--nonce", default="", help="a per-run line at the start of the first turn's tool output, after "
                    "the shared prefix: the prefix stays cached, while no earlier run's turns are (the persistent cache)")
    a = ap.parse_args()
    source = Path(a.source)
    model = call(a.url, "/v1/models")["data"][0]["id"]
    prefix = build_prefix(a.url, source, a.prefix_tokens, a.chunk_chars)
    messages = list(prefix["messages"])
    print(f"prefix: {prefix['tokens']} tokens, {prefix['chunks']} chunks, model {model}, label {a.label!r}", flush=True)
    rows, texts = [], []
    park_before = park_status(a.url)
    total_wall = 0.0
    for turn in range(a.turns):
        chunk, chunk_tokens = turn_tool_output(a.url, source, turn, a.tool_tokens)
        if turn == 0 and a.nonce:
            chunk = f"(session {a.nonce})\n" + chunk
        messages.append({"role": "user", "content": chunk})
        started = time.monotonic()
        r = call(a.url, "/v1/chat/completions", {
            "model": model, "messages": messages, "temperature": 0, "seed": 0, "max_tokens": a.max_tokens,
            "chat_template_kwargs": {"enable_thinking": not a.no_thinking, "preserve_thinking": True}})
        wall = time.monotonic() - started
        total_wall += wall
        message, finish = r["choices"][0]["message"], r["choices"][0].get("finish_reason")
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        usage, timings = r["usage"], r.get("timings", {})
        status = server_status(a.url)
        park_after = status.get("park_lane", {}) or {}
        governor = status.get("memory_governor", {}) or {}
        delta = {k: park_after.get(k, 0) - park_before.get(k, 0)
                 for k in ("parks", "adoptions", "adopted_tokens", "saved_tokens", "mismatches", "reclaims", "replacements")}
        park_before = park_after
        row = {"turn": turn, "prompt_tokens": usage["prompt_tokens"],
               "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
               "uncached_tokens": usage["prompt_tokens"] - usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
               "tool_tokens": chunk_tokens, "prompt_ms": timings.get("prompt_ms", 0.0),
               "predicted_ms": timings.get("predicted_ms", 0.0), "output_tokens": usage["completion_tokens"],
               "finish": finish, "wall_s": wall, "park": delta, "park_enabled": park_after.get("enabled"),
               "decode_tok_s": usage["completion_tokens"] * 1000.0 / timings["predicted_ms"] if timings.get("predicted_ms") else 0.0,
               "host_headroom_gib": governor.get("host_headroom_bytes", 0) / 2**30,
               "growth_allowed": governor.get("growth_allowed"), "pressure": governor.get("system_pressure")}
        rows.append(row)
        print(f"turn {turn}: prompt {row['prompt_tokens']} cached {row['cached_tokens']} uncached {row['uncached_tokens']} "
              f"prompt_ms {row['prompt_ms']:.0f} predicted_ms {row['predicted_ms']:.0f} output {row['output_tokens']} "
              f"({finish}) decode {row['decode_tok_s']:.1f} tok/s wall {wall:.1f}s headroom {row['host_headroom_gib']:.2f} GiB "
              f"growth {row['growth_allowed']} pressure {row['pressure']} park {json.dumps(delta)}", flush=True)
        texts.append(reasoning + "\x00" + text)
        reply = {"role": "assistant", "content": text}
        if reasoning:
            reply["reasoning_content"] = reasoning
        messages.append(reply)
    follow_ups = rows[1:]
    summary = {"label": a.label, "prefix_tokens": prefix["tokens"], "turns": a.turns,
               "prompt_ms_total": sum(r["prompt_ms"] for r in rows),
               "prompt_ms_follow_ups": sum(r["prompt_ms"] for r in follow_ups),
               "uncached_follow_ups": sum(r["uncached_tokens"] for r in follow_ups),
               "output_tokens": sum(r["output_tokens"] for r in rows),
               "predicted_ms_total": sum(r["predicted_ms"] for r in rows), "wall_s": total_wall,
               "adoptions": sum(r["park"]["adoptions"] for r in rows),
               "saved_tokens": sum(r["park"]["saved_tokens"] for r in rows),
               "mismatches": sum(r["park"]["mismatches"] for r in rows),
               "reclaims": sum(r["park"]["reclaims"] for r in rows),
               "min_headroom_gib": min((r["host_headroom_gib"] for r in rows), default=0.0),
               "decode_tok_s": (sum(r["output_tokens"] for r in rows) * 1000.0 /
                                max(1.0, sum(r["predicted_ms"] for r in rows))),
               "park_enabled": rows[-1]["park_enabled"] if rows else None}
    print(f"total: prompt_ms {summary['prompt_ms_total']:.0f} (follow-up turns {summary['prompt_ms_follow_ups']:.0f}, "
          f"uncached {summary['uncached_follow_ups']} tokens), predicted_ms {summary['predicted_ms_total']:.0f}, "
          f"wall {total_wall:.1f}s, adoptions {summary['adoptions']}, saved {summary['saved_tokens']}, "
          f"mismatches {summary['mismatches']}, reclaims {summary['reclaims']}", flush=True)
    result = {"summary": summary, "turns": rows, "texts": texts}
    exit_code = 0
    if a.compare:
        base = json.loads(Path(a.compare).read_text())["texts"]
        comparison = []
        for i, (x, y) in enumerate(zip(base, texts)):
            k = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            identical = x == y
            comparison.append({"turn": i, "identical": identical, "first_divergence_char": None if identical else k,
                               "length": len(x)})
            print(f"  turn {i}: identical={identical}" + ("" if identical else f", first divergence at char {k}/{len(x)}"))
        result["comparison"] = comparison
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps(result, indent=1))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

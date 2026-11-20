#!/usr/bin/env python3
"""Time to first token of a fresh suffix appended to a long cached prefix, on a running Splash server
(SPLASH_PREFILL_ATTN_V2 A/B, splash/tools/prefill_ab.sh; splash/dev-prefill-attn/ATTN.md).

The prefixes are nested: the messages park_long.py builds for the largest position (source files and docs of a
Splash checkout as read_file tool outputs, cached under ~/.cache/splash-park/), cut at the first message boundary
whose rendered size reaches each position. For each position (ascending) and repetition and suffix size:
  1 warm:     the prefix messages alone, max_tokens 1: the prefix's state is cached (cheap when it already is;
              the first one of a server pays the cold prefill, about 2.2 ms + 0.024 ms per 1K context per token)
  2 measured: the prefix + one user message "(probe NONCE)" + a ~SUFFIX-token tool output, max_tokens 1,
              thinking off: TTFT = the server's timings.prompt_ms
The warm request leaves a state at its replay boundary (the prompt without its generation prompt, page-aligned:
Engine::promptReplayBoundary), which the measured prompt shares. A measured row is valid when the server reports the
prefix as cached (uncached tokens within SUFFIX + 256), so it times the suffix's prefill at that context only. The nonce (label, repetition, position, suffix) makes
every measured suffix new to the prefix cache, also across arms and restarts (the persistent SSD cache).

    python3 splash/tools/prefill_ttft.py --positions 30000,60000,100000 --suffixes 2048,4096 --reps 1 \\
        --label A1 --save OUT/ttft-A1.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import park_long  # noqa: E402  (build_prefix, turn_tool_output, call, count_tokens, source_files)

ROOT = Path(__file__).resolve().parents[2]


def message_tokens(url, prefix):
    """Cumulative rendered-size estimate per message of a park_long prefix (cached beside it)."""
    key = f"{prefix['tokens']}-{prefix['chunks']}-{len(prefix['messages'])}"
    cache = park_long.CACHE / f"cumulative-{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    total, counts = 0, []
    for message in prefix["messages"]:
        total += park_long.count_tokens(url, message["content"]) + 6
        counts.append(total)
    cache.write_text(json.dumps(counts))
    return counts


def request(url, model, messages, label):
    started = time.monotonic()
    r = park_long.call(url, "/v1/chat/completions", {
        "model": model, "messages": messages, "temperature": 0, "seed": 0, "max_tokens": 1,
        "chat_template_kwargs": {"enable_thinking": False}})
    wall = time.monotonic() - started
    usage, timings = r["usage"], r.get("timings", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return {"label": label, "prompt_tokens": usage["prompt_tokens"], "cached_tokens": cached,
            "uncached_tokens": usage["prompt_tokens"] - cached, "prompt_ms": timings.get("prompt_ms", 0.0),
            "wall_s": wall}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--source", default=str(ROOT / "splash" / "dev-park2"), help="checkout the prefix is built from")
    ap.add_argument("--positions", default="30000,60000,100000", help="prefix sizes in tokens, comma-separated")
    ap.add_argument("--suffixes", default="2048,4096", help="suffix sizes in tokens, comma-separated")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--chunk-chars", type=int, default=9000)
    ap.add_argument("--label", default="")
    ap.add_argument("--save")
    a = ap.parse_args()
    positions = sorted(int(x) for x in a.positions.split(","))
    suffixes = [int(x) for x in a.suffixes.split(",")]
    source = Path(a.source)
    model = park_long.call(a.url, "/v1/models")["data"][0]["id"]
    prefix = park_long.build_prefix(a.url, source, max(positions), a.chunk_chars)
    counts = message_tokens(a.url, prefix)
    print(f"prefix: {prefix['tokens']} tokens, {len(prefix['messages'])} messages, model {model}, label {a.label!r}",
          flush=True)
    rows = []
    for position in positions:
        # The first message boundary at or past the position, ending on a user message (a tool output).
        cut = next((i + 1 for i, c in enumerate(counts) if c >= position and prefix["messages"][i]["role"] == "user"),
                   len(counts))
        messages = prefix["messages"][:cut]
        for rep in range(a.reps):
            for size in suffixes:
                warm = request(a.url, model, messages, f"warm-{position}")
                chunk, chunk_tokens = park_long.turn_tool_output(a.url, source, 100 + rep * 10 + suffixes.index(size),
                                                                 size)
                nonce = f"{a.label}-r{rep}-p{position}-s{size}-{int(time.time())}"
                probe = messages + [{"role": "user", "content": f"(probe {nonce})\n{chunk}"}]
                row = request(a.url, model, probe, f"p{position}-s{size}")
                row.update({"position": position, "prefix_tokens": warm["prompt_tokens"], "suffix": size,
                            "suffix_tokens": chunk_tokens, "rep": rep, "warm_prompt_ms": warm["prompt_ms"],
                            "warm_uncached": warm["uncached_tokens"]})
                row["valid"] = row["cached_tokens"] > 0 and row["uncached_tokens"] <= size + 256
                row["ms_per_token"] = row["prompt_ms"] / max(1, row["uncached_tokens"])
                rows.append(row)
                print(f"position {position} (prefix {row['prefix_tokens']}, warm {warm['prompt_ms']:.0f} ms for "
                      f"{warm['uncached_tokens']} uncached) suffix {size}: prompt {row['prompt_tokens']} cached "
                      f"{row['cached_tokens']} uncached {row['uncached_tokens']} TTFT {row['prompt_ms']:.0f} ms "
                      f"({row['ms_per_token']:.3f} ms/token) valid {row['valid']}", flush=True)
    result = {"label": a.label, "rows": rows}
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps(result, indent=1))
    return 0 if rows and all(r["valid"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

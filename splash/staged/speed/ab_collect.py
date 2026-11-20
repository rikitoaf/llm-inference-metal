#!/usr/bin/env python3
"""tree_collect.py's agent-like prompts (greedy, thinking on), saving each reply's text so two server builds or flags
can be checked for greedy identity, plus decode tok/s and tokens/cycle from the server's counters.

    python3 splash/staged/speed/ab_collect.py --out logs/speed/ab/base.json [--max-tokens 1500]
    python3 splash/staged/speed/ab_collect.py --compare logs/speed/ab/base.json logs/speed/ab/head.json
"""
import argparse, hashlib, json, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from tree_collect import URL, prompts  # noqa: E402

STATUS = URL.replace("/v1/chat/completions", "/status")


def counters():
    with urllib.request.urlopen(STATUS, timeout=30) as response:
        s = json.load(response)
    m = s.get("metrics", {})
    return m.get("decode_output_tokens", 0), m.get("decode_wall_ms", 0), s.get("scheduler", {}).get("decode_batches", 0)


def collect(a):
    results = []
    t0, w0, c0 = counters()
    for name, content in prompts():
        body = {"model": "qwen3.8-27b-local", "messages": [{"role": "user", "content": content}],
                "max_tokens": a.max_tokens, "temperature": 0, "stream": False}
        request = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            r = json.load(response)
        message = r["choices"][0]["message"]
        text = (message.get("reasoning_content") or "") + "\x00" + (message.get("content") or "")
        t = r.get("timings", {})
        results.append({"prompt": name, "tokens": t.get("predicted_n"), "tps": t.get("predicted_per_second"),
                        "text": text, "sha": hashlib.sha256(text.encode()).hexdigest()[:16]})
        print(f"{name}: {t.get('predicted_n')} tokens, {t.get('predicted_per_second')} tok/s", flush=True)
    t1, w1, c1 = counters()
    tokens, wall, cycles = t1 - t0, w1 - w0, c1 - c0
    summary = {"tokens": tokens, "decode_tps": tokens / wall * 1000 if wall else 0,
               "tokens_per_cycle": tokens / max(cycles, 1)}
    print(f"total: {tokens} tokens, {summary['decode_tps']:.2f} tok/s decode, "
          f"{summary['tokens_per_cycle']:.3f} tokens/cycle", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "results": results}, indent=1))


def compare(base, other):
    A, B = (json.loads(Path(x).read_text()) for x in (base, other))
    same = 0
    for ra, rb in zip(A["results"], B["results"]):
        wa, wb = ra["text"].split(), rb["text"].split()
        k = next((i for i, (x, y) in enumerate(zip(wa, wb)) if x != y), min(len(wa), len(wb)))
        same += ra["text"] == rb["text"]
        print(f"  {ra['prompt']:8s} identical={ra['text'] == rb['text']}  first divergence at word {k} of "
              f"{max(len(wa), len(wb))}")
    sa, sb = A["summary"], B["summary"]
    print(f"  greedy identical {same}/{len(A['results'])}; decode {sa['decode_tps']:.2f} -> {sb['decode_tps']:.2f} "
          f"tok/s ({100 * (sb['decode_tps'] / sa['decode_tps'] - 1):+.1f}%), tokens/cycle "
          f"{sa['tokens_per_cycle']:.3f} -> {sb['tokens_per_cycle']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    compare(*a.compare) if a.compare else collect(a)


if __name__ == "__main__":
    main()

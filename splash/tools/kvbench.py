#!/usr/bin/env python3
"""KV-format benchmark for a running Splash server (INT8 vs BF16 vs TurboQuant).

Every run writes logs/kvbench/<label>.json so two formats can be compared afterwards.

    kvbench.py run --label int8                # speed + needles + judgments + greedy outputs
    kvbench.py run --label turbo --needle-tokens 16000 64000 150000
    kvbench.py compare int8 turbo              # agreement, judgment KL, needle accuracy, speed

Parts:
  speed     400-token new-code generation, temperature 0, thinking off (decode tok/s)
  needles   a passphrase hidden at 10/50/90% depth of a long code haystack; prefill tok/s,
            decode tok/s and whether the answer is right
  judgments /v1/judgments option logits for questions about a long code context: the closest
            thing to logprobs Splash exposes, compared as KL divergence between formats
  greedy    temperature-0 answers to fixed prompts, compared token-by-token between formats
"""
import argparse, json, math, os, pathlib, random, re, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]          # dsh-qwen-jev
OUT = ROOT / "logs" / "kvbench"
URL = os.environ.get("KVBENCH_URL", f"http://{os.environ.get('LLAMA_HOST', '127.0.0.1')}:{os.environ.get('SPLASH_PORT', '8000')}")
MODEL = "qwen3.8-27b-local"
SOURCES = [ROOT / "llama" / "atomic-llama-cpp-turboquant" / p for p in (
    "src/llama-graph.cpp", "src/llama-context.cpp", "src/llama-kv-cache.cpp", "src/llama-model.cpp",
    "common/common.cpp", "common/arg.cpp", "tools/server/server-context.cpp", "ggml/src/ggml.c")]


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.time() - t0


def get(path):
    with urllib.request.urlopen(URL + path, timeout=30) as r:
        return json.load(r)


# Server-wide cumulative counters sampled around each request (Splash /status). Their deltas
# separate decode speed into draft acceptance and per-cycle kernel time, and show the disk tier's
# write-behind during a prefill.
STATUS_COUNTERS = {
    "metrics": ("prefill_wall_ms", "prefill_input_tokens", "decode_cycle_ms", "decode_output_tokens",
                "drafted_tokens", "accepted_draft_tokens"),
    "scheduler": ("decode_batches", "prefill_batches"),
    "disk": ("written_bytes", "read_bytes", "kv_demotions", "kv_copies", "kv_restores"),
    "requests": ("completed",),
}


def counters():
    try:
        st = get("/status")
    except Exception:
        return {}
    out = {f"{g}.{k}": st.get(g, {}).get(k) for g, keys in STATUS_COUNTERS.items() for k in keys}
    wb = st.get("disk", {}).get("write_behind", {})
    out.update({"disk.write_behind.durable": wb.get("durable"), "disk.write_behind.waiting": wb.get("waiting")})
    return out


def delta(before, after):
    d = {k: round(after[k] - before[k], 3) for k in after
         if isinstance(after.get(k), (int, float)) and isinstance(before.get(k), (int, float))}
    d["disk.write_behind.waiting"] = after.get("disk.write_behind.waiting")      # a level, not a counter
    if d.get("scheduler.decode_batches"):
        d["ms_per_cycle"] = round(d["metrics.decode_cycle_ms"] / d["scheduler.decode_batches"], 3)
        d["tokens_per_cycle"] = round(d["metrics.decode_output_tokens"] / d["scheduler.decode_batches"], 3)
    if d.get("metrics.drafted_tokens"):
        d["acceptance"] = round(d["metrics.accepted_draft_tokens"] / d["metrics.drafted_tokens"], 4)
    return d


def nonce():
    # Before a prompt, so no prefix cache (in memory or --persistent-cache) can serve it from an
    # earlier run and inflate prefill tok/s or measure pages written by an older build.
    return f"[run {time.time_ns():020d}]\n"


def chat(content, max_tokens, temperature=0):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": temperature, "reasoning_effort": "none", "stream": False}
    before = counters()
    r, wall = post("/v1/chat/completions", body)
    d = delta(before, counters())
    if d and d.get("requests.completed") != 1:
        d["warning"] = "another request completed meanwhile; the deltas are not this request's alone"
    t = r.get("timings", {})
    return {"status_delta": d, "text": r["choices"][0]["message"].get("content") or "", "wall_s": round(wall, 2),
            "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"), "cache_n": t.get("cache_n"),
            "prefill_tps": round((t["prompt_n"] - (t.get("cache_n") or 0)) / (t["prompt_ms"] / 1000), 1)
                           if t.get("prompt_ms") else None,                # tokens actually prefilled
            "predicted_n": t.get("predicted_n"), "decode_tps": round(t.get("predicted_per_second") or 0, 1)}


def haystack(chars, salt):
    text, i = [f"[document {salt}]\n"], 0
    while sum(map(len, text)) < chars:
        src = SOURCES[i % len(SOURCES)]
        text.append(f"\n// ===== file: {src.name} (part {i}) =====\n" + src.read_text(errors="replace"))
        i += 1
    return "".join(text)[:chars]


def tokens_of(text):
    r, _ = post("/tokenize", {"model": MODEL, "content": text})
    return len(r.get("tokens", []))


def calibrate():
    sample = haystack(40000, "calibration")
    return len(sample) / max(1, tokens_of(sample))       # characters per token for this corpus


def part_speed(res):
    prompt = ("Write a Python module implementing a thread-safe LRU cache class with get, put and delete, plus a "
              "decorator that caches function results with it. Include docstrings and type hints.")
    res["speed"] = [chat(prompt, 400) for _ in range(2)]
    for s in res["speed"]:
        print(f"  speed: decode {s['decode_tps']} tok/s, {s['predicted_n']} tokens, wall {s['wall_s']} s", flush=True)


def part_needles(res, sizes, cpt, depths=(0.1, 0.5, 0.9)):
    words = ["amber", "falcon", "quartz", "lantern", "cobalt", "meadow", "violet", "harbor", "saffron", "glacier"]
    # A rerun replaces those sizes, and drops entries recovered from logs (they have no "tokens").
    res["needles"] = [z for z in res.get("needles", []) if z.get("tokens") is not None and z["tokens"] not in sizes]
    for n in sizes:
        for depth in depths:
            rng = random.Random(f"{n}-{depth}")
            phrase = f"{rng.choice(words)}-{rng.randint(100, 999)}-{rng.choice(words)}"
            body = haystack(int(n * cpt), f"{n}-{depth}")
            at = int(len(body) * depth)
            body = body[:at] + f"\n/* NOTE: the secret passphrase is {phrase}. Remember it. */\n" + body[at:]
            q = nonce() + body + "\n\nQuestion: what is the secret passphrase mentioned in a NOTE comment above? Reply with only the passphrase."
            try:
                r = chat(q, 24)
            except Exception as e:                                   # e.g. over the context limit
                print(f"  needle {n} tokens @ {depth:.0%}: ERROR {e}", flush=True)
                res["needles"].append({"tokens": n, "depth": depth, "error": str(e)})
                continue
            r.update(tokens=n, depth=depth, phrase=phrase, correct=phrase in r["text"])
            res["needles"].append(r)
            print(f"  needle {r['prompt_n']} tokens @ {depth:.0%}: {'OK ' if r['correct'] else 'MISS'} "
                  f"prefill {r['prefill_tps']} tok/s ({r['prompt_ms'] / 1000:.1f} s), decode {r['decode_tps']} tok/s "
                  f"-> {r['text'].strip()[:40]!r}", flush=True)


# Judgment set v2. The context takes an equal slice of each source file and plants coded NOTE
# facts at fixed depths. Every question has a true answer that is in the context:
#   fact   the code planted for a name; the distractors include the other planted codes, so
#          binding the right value to the right name across the context is what is measured
#   file   which file's slice defines a function that occurs in no other slice
#   order  which file comes first, last, or right after another
#   count  structure of the context
# The option probabilities come from the letter logits of /v1/judgments, the only output
# distribution Splash exposes (it rejects logprobs and echo).
JUDGMENT_SET = "v2"
FACT_WORDS = ["amber", "falcon", "quartz", "lantern", "cobalt", "meadow", "violet", "harbor", "saffron", "glacier"]
NOT_NAMES = {"if", "for", "while", "switch", "return", "sizeof", "defined", "static_assert", "GGML_ASSERT",
             "GGML_ABORT", "LLAMA_LOG_INFO", "LLAMA_LOG_WARN", "LLAMA_LOG_ERROR", "LLAMA_LOG_DEBUG"}


def defined_names(text):
    """Identifiers defined at column 0 in C/C++ text: the name before the first "(" on a line
    that starts with a letter and is not a declaration (no trailing ";")."""
    names = []
    for line in text.splitlines():
        if not line[:1].isalpha() or line.rstrip().endswith(";") or "(" not in line:
            continue
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]{5,})\s*\(", line)
        if m and m.group(1) not in NOT_NAMES and not line.startswith(("return", "else", "case")):
            names.append(m.group(1))
    return names


def judgment_set(tokens, cpt):
    """(state, questions) for --judgment-tokens TOKENS. Deterministic: no randomness across runs."""
    rng = random.Random(f"judgments-{JUDGMENT_SET}-{tokens}")
    per_file = int(tokens * cpt / len(SOURCES))
    slices = [(src.name, src.read_text(errors="replace")[:per_file]) for src in SOURCES]
    names = [n for n, _ in slices]
    # Planted facts, at line boundaries near evenly spaced depths of the whole context.
    body = "[document judgments]\n" + "".join(f"\n// ===== file: {n} =====\n{t}" for n, t in slices)
    codes = rng.sample(range(1000, 10000), len(FACT_WORDS))
    for i, (word, code) in reversed(list(enumerate(zip(FACT_WORDS, codes)))):
        at = body.find("\n", int(len(body) * (0.05 + 0.9 * i / (len(FACT_WORDS) - 1)))) + 1
        body = body[:at] + f"/* NOTE: the calibration code for {word} is {code}. */\n" + body[at:]
    questions = []

    def ask(category, question, options, truth):
        rng.shuffle(options)
        questions.append({"category": category, "question": question, "truth": truth,
                          "options": [{"id": o, "description": o} for o in options]})

    for word, code in zip(FACT_WORDS, codes):
        others = rng.sample([c for c in codes if c != code], 3)
        near = set()
        while len(near) < 2:                       # one digit changed
            d = rng.randrange(4)
            alt = str(code)[:d] + str((int(str(code)[d]) + rng.randrange(1, 10)) % 10) + str(code)[d + 1:]
            if alt[0] != "0" and int(alt) not in codes:
                near.add(alt)
        ask("fact", f"According to the NOTE comments in the context, what is the calibration code for {word}?",
            [str(code)] + [str(c) for c in others] + sorted(near), str(code))
    counts = {}
    for n, t in slices:
        for x in set(defined_names(t)):
            counts[x] = counts.get(x, 0) + 1
    for n, t in slices:
        unique = [x for x in defined_names(t) if counts.get(x) == 1 and all(x not in u for m, u in slices if m != n)]
        if unique:
            ask("file", f"In which file of the context is the function {unique[0]} defined?", list(names), n)
    ask("order", "Which file comes first in the context?", list(names), names[0])
    ask("order", "Which file comes last in the context?", list(names), names[-1])
    ask("order", f"Which file comes right after {names[4]} in the context?", list(names), names[5])
    ask("count", "How many different source files does the context contain?", ["2", "4", "6", "8", "10", "12"],
        str(len(names)))
    ask("count", "How many NOTE comments with a calibration code does the context contain?",
        ["4", "6", "8", "10", "12", "14"], str(len(FACT_WORDS)))
    return body, questions


def part_judgments(res, cpt, tokens, use_nonce=False):
    state, questions = judgment_set(tokens, cpt)
    if use_nonce:
        state = nonce() + state
    res["judgment_set"] = f"{JUDGMENT_SET}:{tokens}"
    res["judgments"] = []
    for qi, q in enumerate(questions):
        body = {"id": f"q{qi}", "state": state, "question": q["question"], "options": q["options"]}
        r, wall = post("/v1/judgments", body)
        p = judgment_probs(r)
        res["judgments"].append({**{k: q[k] for k in ("category", "question", "truth")}, "wall_s": round(wall, 2),
                                 "response": r})
        top = max(p, key=p.get) if p else None
        print(f"  judgment q{qi} {q['category']}: {wall:.1f} s -> {top} {p.get(top, 0):.3f}"
              f" (truth {q['truth']} {p.get(q['truth'], 0):.3f})", flush=True)


GREEDY = [
    "Explain in three sentences what a KV cache is in a transformer.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "List the planets of the solar system in order from the Sun.",
    "Translate to French: The weather is nice today and we are going to the park.",
    "What is 17 multiplied by 23? Show the calculation.",
    "Write a haiku about debugging code at night.",
]


def part_greedy(res, cpt, context_tokens, use_nonce=False):
    res["greedy"] = []
    prefix = (nonce() if use_nonce else "") + haystack(int(context_tokens * cpt), "greedy") + "\n\nIgnore the code above.\n" if context_tokens else ""
    for p in GREEDY:
        r = chat(prefix + p, 160)
        res["greedy"].append({"prompt": p, "context_tokens": context_tokens, **r})
        cached = f", {r['cache_n']} cached" if context_tokens and p == GREEDY[0] and (r.get("cache_n") or 0) > 64 else ""
        print(f"  greedy ({context_tokens} ctx{cached}): {r['text'][:60]!r}", flush=True)


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    status = get("/status")
    path = OUT / f"{a.label}.json"
    # A run updates the parts it runs and keeps the rest, saving after every part, so an
    # interrupted run loses only the part in progress.
    res = json.loads(path.read_text()) if path.exists() and not a.fresh else {}
    kv = status.get("identity", {}).get("kv") or {}
    old = (res.get("kv") or {}).get("format")
    if old and kv.get("format") and old != kv["format"]:
        sys.exit(f"error: {path.name} holds {old} results but the server runs {kv['format']}; "
                 "use another --label or --fresh")
    env = {k: os.environ.get(k) for k in ("SPLASH_PERSISTENT_CACHE", "SPLASH_KV_FORMAT", "SPLASH_EXTRA_ARGS",
                                          "MODEL_PROFILE")}
    meta = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kv": kv, "env": env,
            "model": status.get("instance", {}).get("model"),
            "build": status.get("identity", {}).get("build") or status.get("version"),
            "macos": status.get("memory_plan", {}).get("device", {}).get("macos_version")}
    res.update({"label": a.label, "max_context": status.get("maximum_context_tokens"), "kv": kv,
                "kv_page_bytes": status.get("memory_plan", {}).get("budget", {}).get("kv_page_bytes")})
    meta["nonce"] = a.nonce
    res.setdefault("meta", {})

    def save(part):                         # when and how each part last completed
        res["meta"][part] = meta
        path.write_text(json.dumps(res, indent=1))
    print(f"[{a.label}] model {meta['model']}, max context {res['max_context']}, kv {res['kv']}", flush=True)
    cpt = calibrate()
    print(f"  {cpt:.2f} characters per token", flush=True)
    if "speed" in a.parts:
        part_speed(res); save("speed")
    if "needles" in a.parts:
        part_needles(res, a.needle_tokens, cpt, a.depths); save("needles")
    if "judgments" in a.parts:
        part_judgments(res, cpt, a.judgment_tokens, a.nonce); save("judgments")
    if "greedy" in a.parts:
        part_greedy(res, cpt, 0, a.nonce)
        greedy = res["greedy"]
        part_greedy(res, cpt, a.greedy_context, a.nonce)
        res["greedy"] = greedy + res["greedy"]
        save("greedy")
    print(f"wrote {path}")


def kl(p, q):
    return sum(pi * math.log(pi / max(q.get(k, 1e-12), 1e-12)) for k, pi in p.items() if pi > 0)


def judgment_probs(resp):
    for key in ("probabilities", "probs"):
        if isinstance(resp.get(key), dict):
            return resp[key]
        if isinstance(resp.get(key), list) and resp.get("option_ids"):
            return dict(zip(resp["option_ids"], resp[key]))
    if isinstance(resp.get("options"), list):
        return {o["id"]: o.get("probability", o.get("prob")) for o in resp["options"]}
    return {}


def compare(a):
    A, B = (json.loads((OUT / f"{x}.json").read_text()) for x in (a.base, a.other))
    print(f"{a.base}: max context {A['max_context']}, page {A.get('kv_page_bytes')} B | "
          f"{a.other}: max context {B['max_context']}, page {B.get('kv_page_bytes')} B")
    for x, r in ((a.base, A), (a.other, B)):
        if r.get("speed"):
            print(f"  speed {x}: decode {[s['decode_tps'] for s in r['speed']]} tok/s")
    if A.get("greedy") and B.get("greedy"):
        same = prefix_total = total = 0
        index = {(g["prompt"], g["context_tokens"]): g for g in B["greedy"]}
        pairs = [(g, index[(g["prompt"], g["context_tokens"])]) for g in A["greedy"]
                 if (g["prompt"], g["context_tokens"]) in index]
        if len(pairs) != len(A["greedy"]) or len(pairs) != len(B["greedy"]):
            print(f"  greedy MISMATCH: {len(A['greedy'])} vs {len(B['greedy'])} entries, {len(pairs)} paired")
        for x, r in ((a.base, A), (a.other, B)):
            # The prompts of one run share a haystack, so only the first may not hit the cache.
            first = next((g for g in r["greedy"] if g["context_tokens"]), None)
            if first and (first.get("cache_n") or 0) > 64:
                print(f"  WARNING {x}: the first greedy prompt with context was served from cache "
                      f"({first['cache_n']} tokens): an earlier run's pages were measured")
        for ga, gb in pairs:
            wa, wb = ga["text"].split(), gb["text"].split()
            k = next((i for i, (x, y) in enumerate(zip(wa, wb)) if x != y), min(len(wa), len(wb)))
            prefix_total += k
            total += max(len(wa), len(wb))
            same += ga["text"] == gb["text"]
            print(f"  greedy ctx={ga['context_tokens']}: identical={ga['text'] == gb['text']}, "
                  f"first divergence at word {k} of {max(len(wa), len(wb))}")
        print(f"  greedy: {same}/{len(pairs)} identical, {prefix_total}/{total} words before divergence")
    for x, r in ((a.base, A), (a.other, B)):
        scored = [(j, judgment_probs(j["response"])) for j in r.get("judgments", []) if "truth" in j]
        if scored:
            by = {}
            for j, pr in scored:
                by.setdefault(j["category"], []).append((pr.get(j["truth"], 0.0), max(pr, key=pr.get) == j["truth"]))
            parts = ", ".join(f"{c} {sum(t for _, t in v)}/{len(v)} p={sum(q for q, _ in v) / len(v):.3f}"
                              for c, v in sorted(by.items()))
            n = len(scored)
            print(f"  judgments {x} ({r.get('judgment_set')}): top-1 correct {sum(max(pr, key=pr.get) == j['truth'] for j, pr in scored)}/{n}, "
                  f"mean p(truth) {sum(pr.get(j['truth'], 0.0) for j, pr in scored) / n:.3f}; {parts}")
    same_set = A.get("judgment_set", "v1") == B.get("judgment_set", "v1")
    if A.get("judgments") and B.get("judgments") and not same_set:
        print(f"  judgments MISMATCH: question sets {A.get('judgment_set', 'v1')} vs {B.get('judgment_set', 'v1')} "
              "use different contexts; not compared")
    if A.get("judgments") and B.get("judgments") and same_set:
        kls = []
        index = {j["question"]: j for j in B["judgments"]}
        pairs = [(j, index[j["question"]]) for j in A["judgments"] if j["question"] in index]
        if len(pairs) != len(A["judgments"]) or len(pairs) != len(B["judgments"]):
            print(f"  judgments MISMATCH: {len(A['judgments'])} vs {len(B['judgments'])} questions, {len(pairs)} paired")
        for ja, jb in pairs:
            pa, pb = judgment_probs(ja["response"]), judgment_probs(jb["response"])
            if pa and pb:
                kls.append(kl(pa, pb))
                top_a, top_b = max(pa, key=pa.get), max(pb, key=pb.get)
                print(f"  judgment: KL={kls[-1]:.4f} top {top_a}({pa[top_a]:.3f}) vs {top_b}({pb[top_b]:.3f})")
        if kls:
            agree = sum(max(judgment_probs(ja["response"]), key=judgment_probs(ja["response"]).get) ==
                        max(judgment_probs(jb["response"]), key=judgment_probs(jb["response"]).get) for ja, jb in pairs)
            print(f"  judgments: mean KL {sum(kls) / len(kls):.4f}, max {max(kls):.4f}, same top answer {agree}/{len(pairs)}")
    for x, r in ((a.base, A), (a.other, B)):
        n = [z for z in r.get("needles", []) if "correct" in z]
        errors = [z for z in r.get("needles", []) if "error" in z]
        if errors:
            print(f"  needles {x}: {len(errors)} errors: " + ", ".join(
                f"{z.get('tokens')}t@{z.get('depth', 0):.0%}: {z['error'][:60]}" for z in errors))
        if n:
            print(f"  needles {x}: {sum(z['correct'] for z in n)}/{len(n)} correct, {len(errors)} errors; " + ", ".join(
                f"{z['prompt_n']}t@{z['depth']:.0%}:{'ok' if z['correct'] else 'MISS'} {z['prefill_tps']}pp/{z['decode_tps']}tg"
                for z in n))
        ds = [z["status_delta"] for part in ("speed", "needles", "greedy") for z in r.get(part, [])
              if z.get("status_delta", {}).get("drafted_tokens") or z.get("status_delta", {}).get("metrics.drafted_tokens")]
        if ds:
            drafted = sum(d["metrics.drafted_tokens"] for d in ds)
            cycles = sum(d.get("scheduler.decode_batches", 0) for d in ds)
            print(f"  decode {x}: draft acceptance {sum(d['metrics.accepted_draft_tokens'] for d in ds) / drafted:.3f}, "
                  f"{sum(d['metrics.decode_output_tokens'] for d in ds) / max(cycles, 1):.2f} tokens/cycle, "
                  f"{sum(d['metrics.decode_cycle_ms'] for d in ds) / max(cycles, 1):.2f} ms/cycle over {len(ds)} requests")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--parts", nargs="*", default=["speed", "needles", "judgments", "greedy"])
    r.add_argument("--needle-tokens", nargs="*", type=int, default=[16000, 60000])
    r.add_argument("--depths", nargs="*", type=float, default=[0.1, 0.5, 0.9])
    r.add_argument("--judgment-tokens", type=int, default=24000)
    r.add_argument("--greedy-context", type=int, default=24000)
    r.add_argument("--fresh", action="store_true", help="start a new results file instead of merging")
    r.add_argument("--nonce", action="store_true",
                   help="also prefix judgment and greedy prompts with a per-run nonce (needles always have one). "
                        "Off by default so formats see byte-identical prompts; the persistent cache cannot serve "
                        "stale pages because its namespace carries the quantizer revision")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("other")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()

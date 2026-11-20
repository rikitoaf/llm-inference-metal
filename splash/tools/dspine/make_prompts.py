#!/usr/bin/env python3
"""Token-file prompts for decode-run (one prompt per line of token ids) from real dsh sessions and agent-like asks.

    make_prompts.py sessions OUT.txt [--count 8] [--min 4000] [--max 20000]
    make_prompts.py agent OUT.txt

`sessions` rebuilds the chat a dsh session sent (system message, request tools, user/assistant/tool messages with
reasoning and tool calls) from dsh-home/sessions/*/session.v3.jsonl.zstd and renders it with the installed target's
chat template (thinking on, generation prompt). Each prompt is cut just before an assistant message: "start" cuts
come right after a user message, "write" cuts come before an assistant message that calls write/edit. One prompt per
session, shortest qualifying cut of each mode, so the corpus mixes reasoning starts and code writing. It is an
approximation of the dsh request (compaction/pruning and runtime-context snapshots are replayed as recorded, not
re-derived), which is all a drafter comparison needs: real agent context, real continuation by the target.
`agent` renders the six agent-like asks of splash/tools/tree_collect.py (no tools).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SELECTION = ROOT / "splash/splash-src/install/models/.selections/fb0aa6e23bdecc7d24023adea3bcae72062d1c103ab1578902b00a2ebb3f500f"


def tokenizer():
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=str(SELECTION / "tokenizer/tokenizer.json"))
    tok.chat_template = (SELECTION / "tokenizer/chat_template.jinja").read_text()
    return tok


def render(tok, messages, tools=None):
    text = tok.apply_chat_template(messages, tools=tools or None, tokenize=False, add_generation_prompt=True)
    return tok.encode(text, add_special_tokens=False)


def text_of(blocks):
    out = []
    for b in blocks:
        if b.get("type") == "text":
            out.append(b["text"])
        elif b.get("type") == "file":
            a = b.get("attachment", {})
            out.append(f"[attached file {a.get('name')} ({a.get('bytes')} bytes)]")
    return "\n".join(out)


def session_events(path):
    raw = subprocess.run(["zstd", "-dc", str(path)], capture_output=True, check=True).stdout
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def session_chat(events):
    """[(kind, messages-so-far, tools)] at every assistant message: kind 'start' (after a user message) or 'write'
    (the assistant message calls write/edit) or 'other'."""
    messages, tools, cuts = [], None, []
    for e in events:
        t, d = e["type"], e.get("data", {})
        if t == "request/header" and tools is None:
            tools = [{"type": "function", "function": {"name": x["name"], "description": x.get("description", ""),
                                                       "parameters": x.get("parameters", {})}}
                     for x in d.get("header", {}).get("tools", [])]
        elif t == "system/message" and not messages:
            messages.append({"role": "system", "content": text_of(d["message"]["content"])})
        elif t == "user/message":
            messages.append({"role": "user", "content": text_of(d.get("content", []))})
        elif t == "assistant/message":
            m = d["message"]
            reasoning = "".join(b["text"] for b in m["content"] if b.get("type") == "reasoning")
            content = "".join(b["text"] for b in m["content"] if b.get("type") == "text")
            calls = []
            for b in m["content"]:
                if b.get("type") == "tool-call":
                    try:
                        args = json.loads(b.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {"raw": b.get("arguments")}
                    calls.append({"type": "function", "id": b.get("id"),
                                  "function": {"name": b["name"], "arguments": args}})
            kind = "other"
            if messages and messages[-1]["role"] == "user":
                kind = "start"
            if any(c["function"]["name"] in ("write", "edit", "multi_edit", "apply_patch") for c in calls):
                kind = "write"
            if messages and messages[0]["role"] == "system":
                cuts.append((kind, [dict(x) for x in messages], tools))
            msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
        elif t == "tool/result":
            m = d["message"]
            parts = []
            for b in m.get("content", []):
                if b.get("type") == "tool-result":
                    parts.append(text_of(b.get("content", [])))
            messages.append({"role": "tool", "content": "\n".join(parts)})
    return cuts


def sessions(args):
    tok = tokenizer()
    files = sorted((ROOT / "dsh-home/sessions").glob("*/session-*/session.v3.jsonl.zstd"),
                   key=lambda p: -p.stat().st_size)
    chosen = []
    for f in files:
        try:
            cuts = session_chat(session_events(f))
        except Exception as error:  # a session this rebuild cannot read
            print(f"skip {f.parent.name}: {error}", file=sys.stderr)
            continue
        picked, skipped = {}, {"start": 0, "write": 0}
        for kind, messages, tools in cuts:
            if kind not in ("start", "write") or kind in picked:
                continue
            try:
                ids = render(tok, messages, tools)
            except Exception as error:
                print(f"skip a cut of {f.parent.name}: {error}", file=sys.stderr)
                continue
            if args.min <= len(ids) <= args.max:
                if skipped[kind] < args.nth:          # --nth: a later cut of the same kind
                    skipped[kind] += 1
                    continue
                picked[kind] = ids
        for kind, ids in picked.items():
            project = f.parent.parent.name.strip("-").split("-workspace-")[-1].split("WebstormProjects")[-1]
            chosen.append((project.strip("-") or "WebstormProjects", f.parent.name[8:16], kind, ids))
    # Alternate modes; one prompt per project and mode (distinct work), shortest first within a mode.
    chosen.sort(key=lambda c: len(c[3]))
    out = []
    for want in ("start", "write") * args.count:
        for c in chosen:
            if c[2] == want and (c[0], want) not in {(x[0], x[2]) for x in out} and c[3] not in [x[3] for x in out]:
                out.append(c)
                break
        if len(out) >= args.count:
            break
    with open(args.out, "w") as fh:
        for project, session, kind, ids in out:
            fh.write(" ".join(map(str, ids)) + "\n")
            print(f"{kind:5s} {len(ids):6d} tokens  {project} {session}")


def long_prompts(args):
    """Prompts of about --target tokens (64K+ by default): from the largest sessions, the cut nearest the target
    within [--min, --max], one per session, for long-context timing."""
    tok = tokenizer()
    files = sorted((ROOT / "dsh-home/sessions").glob("*/session-*/session.v3.jsonl.zstd"),
                   key=lambda p: -p.stat().st_size)
    out = []
    for f in files:
        if len(out) >= args.count:
            break
        try:
            cuts = session_chat(session_events(f))
        except Exception:
            continue
        best = None
        for kind, messages, tools in cuts:
            ids = render(tok, messages, tools)
            if args.min <= len(ids) <= args.max and (best is None or abs(len(ids) - args.target) < abs(len(best) - args.target)):
                best = ids
            if len(ids) > args.max:
                break
        if best:
            out.append(best)
            print(f"long {len(best):6d} tokens  {f.parent.parent.name[-30:]} {f.parent.name[8:16]}")
    with open(args.out, "w") as fh:
        for ids in out:
            fh.write(" ".join(map(str, ids)) + "\n")


def agent(args):
    sys.path.insert(0, str(ROOT / "splash/tools"))
    import tree_collect
    tok = tokenizer()
    with open(args.out, "w") as fh:
        for name, content in tree_collect.prompts():
            ids = render(tok, [{"role": "user", "content": content}])
            fh.write(" ".join(map(str, ids)) + "\n")
            print(f"{name:8s} {len(ids):6d} tokens")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sessions", "agent", "long"])
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--min", type=int, default=4000)
    ap.add_argument("--max", type=int, default=20000)
    ap.add_argument("--nth", type=int, default=0, help="take each session's nth qualifying cut of a kind")
    ap.add_argument("--target", type=int, default=70000, help="long: the length to aim for")
    args = ap.parse_args()
    {"sessions": sessions, "agent": agent, "long": long_prompts}[args.mode](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Long agent-shaped token-file prompts (60-110K tokens) for decode-run, without company content.

    longctx_prompts.py OUT.txt [--targets 62000,74000,86000,98000,108000] [--source splash/dev-park2] [--more-sources]
                       [--first-task N]

Each prompt is a dsh-like chat rendered with the target's chat template (make_prompts.py's renderer: thinking on,
generation prompt): the system message and tool list of one eval-task session (workspace/eval-*, public toy tasks),
a task about the Splash engine, then assistant tool calls (read) answered by chunks of the open Splash sources and
until the target length, and a closing user message that asks for the full answer without more tool calls (so a
run decodes hundreds of tokens of reasoning and answer instead of one short tool call). Prompts differ in task, file order and length;
odd-numbered tasks ask for rewritten code (output that repeats context, where the copy extension works), even ones
for analysis. Deterministic: the same arguments give byte-identical prompts.
"""
import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "dspine"))
import make_prompts  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TASKS = [
    "Explain how one decode cycle is encoded and where its time goes at long context. Read the files I open, then "
    "write the analysis step by step.",
    "Rewrite the verify attention plan so that the split count also depends on the number of verify tiles. Read the "
    "files I open, then write the complete changed functions.",
    "Find every place where the draft tree's rows are read and say what each reader assumes about row order. Read "
    "the files I open, then list the readers with file and function names.",
    "Port the copy extension's host-side setup into one helper shared by both drafters. Read the files I open, then "
    "write the helper and both call sites in full.",
    "Review the prefix cache and park lane code for a state that could be adopted with stale lengths. Read the files "
    "I open, then give a numbered list of risks with the lines involved.",
    "Add a per-request counter of decode cycles to the runtime status, end to end (runtime, engine status, server "
    "JSON). Read the files I open, then write every changed function in full.",
]
CLOSING = {"role": "user", "content": "Those are all the files you need. Do not call any more tools: write the "
                                      "complete answer now, in full."}
PATTERNS = ("runtime/model/*.mm", "runtime/model/*.cpp", "runtime/ops/*.cpp", "runtime/engine/*.cpp",
            "runtime/metal/kernels/*/*.metal", "runtime/metal/kernels/common/*.h", "runtime/model/*.hpp",
            "runtime/ops/*.hpp", "server/*.py", "DEVELOPMENT.md")
# --more-sources (prompts past ~90K tokens, where PATTERNS run out): the engine's tests and benchmarks as well.
MORE_PATTERNS = ("dev/tests/engine/*.mm", "dev/tests/engine/*.cpp", "dev/benchmarks/*.mm", "runtime/engine/*.hpp",
                 "runtime/metal/*.mm", "runtime/metal/*.hpp")


def eval_session_header():
    """(system text, tools) of the first eval-task session that has both."""
    for path in sorted((ROOT / "dsh-home/sessions").glob("*workspace-eval-*/session-*/session.v3.jsonl.zstd")):
        try:
            cuts = make_prompts.session_chat(make_prompts.session_events(path))
        except Exception:
            continue
        for _, messages, tools in cuts:
            if tools and messages and messages[0]["role"] == "system":
                return messages[0]["content"], tools
    raise SystemExit("no eval-task session with a system message and tools under dsh-home/sessions")


def chunks_of(source, seed, chunk_lines=400, patterns=PATTERNS):
    files = []
    for pattern in patterns:
        files += sorted(source.glob(pattern))
    files = [f for f in files if f.is_file() and f.stat().st_size > 4000]
    random.Random(seed).shuffle(files)
    for path in files:
        lines = path.read_text(errors="replace").splitlines()
        for start in range(0, len(lines), chunk_lines):
            body = "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(lines[start:start + chunk_lines]))
            yield path.relative_to(source).as_posix(), start + 1, body


def build(tok, system, tools, source, task_index, target, patterns=PATTERNS):
    task = TASKS[task_index % len(TASKS)]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    read_tool = next((t["function"]["name"] for t in tools if t["function"]["name"] in ("read", "read_file")),
                     tools[0]["function"]["name"])
    best = None
    for call_index, (name, first_line, body) in enumerate(chunks_of(source, seed=1000 + task_index,
                                                                         patterns=patterns)):
        messages.append({"role": "assistant", "content": "",
                         "reasoning_content": f"I need the next part of {name} (from line {first_line}) before I "
                                              f"can answer; I will read it.",
                         "tool_calls": [{"type": "function", "id": f"call_{call_index}",
                                         "function": {"name": read_tool,
                                                      "arguments": {"path": name, "offset": first_line, "limit": 400}}}]})
        messages.append({"role": "tool", "content": body})
        if call_index % 4 == 3 or best is None:
            ids = make_prompts.render(tok, messages + [CLOSING], tools)
            if len(ids) >= target:
                return best if best is not None and target - len(best) < len(ids) - target else ids
            best = ids
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--targets", default="62000,74000,86000,98000,108000")
    ap.add_argument("--source", default=str(ROOT / "splash" / "dev-park2"))
    ap.add_argument("--first-task", type=int, default=0, help="task (and file order) index of the first target")
    ap.add_argument("--more-sources", action="store_true", help="also read dev/ tests and benchmarks (long targets)")
    args = ap.parse_args()
    tok = make_prompts.tokenizer()
    system, tools = eval_session_header()
    source = Path(args.source)
    with open(args.out, "w") as out:
        for index, target in enumerate((int(t) for t in args.targets.split(",")), start=args.first_task):
            ids = build(tok, system, tools, source, index, target,
                        PATTERNS + MORE_PATTERNS if args.more_sources else PATTERNS)
            out.write(" ".join(map(str, ids)) + "\n")
            print(f"prompt {index}: {len(ids)} tokens, task {index % len(TASKS)}, "
                  f"sha {hashlib.sha256(' '.join(map(str, ids)).encode()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()

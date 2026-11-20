#!/usr/bin/env python3
"""How much of the dsh agents' context comes from code-navigation tool results.

Baseline for the codebase-memory-mcp decision (splash/EXTERNAL-2026-11-09.md section 3): a code graph can only save
what reads, greps, finds and navigation shell commands put into the context. Reads every
dsh-home/sessions/**/session.v3.jsonl.zstd (CPU only, read-only) and writes a JSON + Markdown report.

usage: session_tool_share.py [--sessions DIR] [--out-json FILE] [--out-md FILE]
"""
import argparse, collections, json, pathlib, re, subprocess, sys

NAVIGATION_SHELL = re.compile(r'(^|[|;&\s])(grep|rg|find|fd|ls|tree|cat|head|tail|sed -n|wc)\b')
NAVIGATION_TOOLS = {'read', 'grep', 'glob', 'list', 'ls', 'find', 'search'}


def tool_category(name, arguments):
    if name in NAVIGATION_TOOLS or name.startswith('mcp__xerj') or 'xerj' in name:
        return 'navigation'
    if name == 'bash':
        try:
            command = json.loads(arguments).get('command', '')
        except (ValueError, AttributeError):
            command = ''
        return 'navigation-shell' if NAVIGATION_SHELL.search(command) else 'shell-other'
    return 'other'


def result_text_chars(message):
    total = 0
    for part in message.get('content', []):
        for inner in part.get('content', []) if isinstance(part.get('content'), list) else []:
            total += len(inner.get('text', '') or '')
    return total


def analyse_session(path):
    raw = subprocess.run(['zstd', '-dc', str(path)], capture_output=True, check=False).stdout
    call_info = {}
    per_tool = collections.defaultdict(lambda: {'calls': 0, 'result_chars': 0})
    per_category = collections.Counter()
    reasoning_chars = 0
    read_paths = collections.Counter()
    for line in raw.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        kind, data = record.get('type'), record.get('data', {})
        if kind == 'tool/call':
            name, arguments = data.get('name', '?'), data.get('arguments', '')
            call_info[data.get('callId')] = (name, tool_category(name, arguments))
            per_tool[name]['calls'] += 1
            if name == 'read':
                try:
                    read_paths[json.loads(arguments).get('file_path', '')] += 1
                except (ValueError, AttributeError):
                    pass
        elif kind == 'tool/result':
            message = data.get('message', {})
            call_id = message.get('source', {}).get('callId')
            name, category = call_info.get(call_id, ('?', 'other'))
            chars = result_text_chars(message)
            per_tool[name]['result_chars'] += chars
            per_category[category] += chars
        elif kind == 'assistant/message':
            for part in data.get('message', {}).get('content', []):
                if part.get('type') == 'reasoning':
                    reasoning_chars += len(part.get('text', '') or '')
    repeated_reads = sum(count - 1 for count in read_paths.values() if count > 1)
    return per_tool, per_category, reasoning_chars, sum(read_paths.values()), repeated_reads


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', default=str(root / 'dsh-home' / 'sessions'))
    parser.add_argument('--out-json', default='-')
    parser.add_argument('--out-md', default=None)
    args = parser.parse_args()

    sessions = sorted(pathlib.Path(args.sessions).rglob('session.v3.jsonl.zstd'))
    tools = collections.defaultdict(lambda: {'calls': 0, 'result_chars': 0})
    categories = collections.Counter()
    reasoning = reads = repeated = 0
    for path in sessions:
        per_tool, per_category, reasoning_chars, read_count, repeated_reads = analyse_session(path)
        for name, stats in per_tool.items():
            tools[name]['calls'] += stats['calls']
            tools[name]['result_chars'] += stats['result_chars']
        categories.update(per_category)
        reasoning += reasoning_chars
        reads += read_count
        repeated += repeated_reads
    tool_chars = sum(categories.values())
    navigation = categories['navigation'] + categories['navigation-shell']
    report = {
        'sessions': len(sessions),
        'tool_result_chars': tool_chars,
        'navigation_result_chars': navigation,
        'navigation_share_of_tool_results': round(navigation / tool_chars, 4) if tool_chars else None,
        'reasoning_chars': reasoning,
        'reads': reads,
        'repeated_reads': repeated,
        'by_category': dict(categories),
        'by_tool': dict(sorted(tools.items(), key=lambda item: -item[1]['result_chars'])),
    }
    text = json.dumps(report, indent=2)
    if args.out_json == '-':
        print(text)
    else:
        pathlib.Path(args.out_json).write_text(text)
    if args.out_md:
        lines = [f"# Tool-result share, {len(sessions)} dsh sessions", '',
                 f"- tool-result chars: {tool_chars:,} (~{tool_chars // 4:,} tokens)",
                 f"- code-navigation share: {report['navigation_share_of_tool_results']}",
                 f"- reads: {reads}, repeated reads of the same path: {repeated}",
                 f"- reasoning chars (for scale): {reasoning:,}", '',
                 '| tool | calls | result chars |', '|---|---|---|']
        lines += [f"| {name} | {s['calls']} | {s['result_chars']:,} |" for name, s in report['by_tool'].items()]
        pathlib.Path(args.out_md).write_text('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

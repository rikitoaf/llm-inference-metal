#!/usr/bin/env python3
"""A/B report for the codebase-memory-mcp trial: joins run rows to their dsh sessions and compares conditions.

Input rows (JSONL) come from eval/run_eval.py (cond, workdir, started_at, wall_s, hidden_passed/hidden_total; the
session is the one in dsh-home/sessions whose cwd equals workdir and whose createdAt falls inside the run) or from
workspace/platform-bench/run_bench.py (label, wall_seconds, score in 0-1, passed, session = the session file). Per run it measures steps,
navigation tool calls and result chars (read/grep/glob/xerj and navigation shell commands, as
session_tool_share.py counts them), CBM calls (mcp__cbm__*) and chars, adoption (CBM calls / (CBM + navigation
calls)) and peak context (largest usage.totalTokens of any step). CPU only, read-only.

usage: cbm_ab_report.py RESULTS.jsonl... [--sessions DIR] [--out-md FILE] [--out-json FILE]
"""
import argparse, collections, json, pathlib, statistics, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from session_tool_share import result_text_chars, tool_category  # noqa: E402


def call_category(name, arguments):
    return 'cbm' if name.startswith('mcp__cbm__') else tool_category(name, arguments)


def session_metrics(path):
    raw = subprocess.run(['zstd', '-dc', str(path)], capture_output=True, check=False).stdout
    calls_by_id, metrics = {}, collections.Counter()
    peak_context = 0
    for line in raw.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        kind, data = record.get('type'), record.get('data', {})
        if kind == 'assistant/message':
            metrics['steps'] += 1
            peak_context = max(peak_context, (data.get('usage') or {}).get('totalTokens') or 0)
        elif kind == 'tool/call':
            category = call_category(data.get('name', '?'), data.get('arguments', ''))
            calls_by_id[data.get('callId')] = category
            metrics[f'{category}_calls'] += 1
        elif kind == 'tool/result':
            message = data.get('message', {})
            category = calls_by_id.get(message.get('source', {}).get('callId'), 'other')
            metrics[f'{category}_chars'] += result_text_chars(message)
    navigation_calls = metrics['navigation_calls'] + metrics['navigation-shell_calls']
    return {'steps': metrics['steps'], 'peak_context': peak_context,
            'nav_calls': navigation_calls,
            'nav_chars': metrics['navigation_chars'] + metrics['navigation-shell_chars'],
            'cbm_calls': metrics['cbm_calls'], 'cbm_chars': metrics['cbm_chars'],
            'tool_calls': sum(v for k, v in metrics.items() if k.endswith('_calls'))}


def find_session(sessions_root, workdir, started_at, wall_s):
    """The earliest session with this cwd created inside the run (dsh names session folders after the cwd)."""
    matches = []
    for path in sessions_root.rglob('session.v3.jsonl.zstd'):
        head = subprocess.run(['zstd', '-dc', str(path)], capture_output=True, text=True).stdout.split('\n', 1)[0]
        try:
            header = json.loads(head)
        except ValueError:
            continue
        created_s = header.get('createdAt', 0) / 1000
        if header.get('cwd') == workdir and started_at - 5 <= created_s <= started_at + wall_s:
            matches.append((created_s, path))
    return min(matches)[1] if matches else None


def score_of(row):
    if 'hidden_total' in row:
        return row['hidden_passed'], row['hidden_total']
    return row.get('score', 0), row.get('score_max', 1)


def passed(row):
    return row['passed'] if 'passed' in row else score_of(row)[0] == score_of(row)[1]


def normalize_row(row):
    """Bench rows name the condition 'label' and the wall time 'wall_seconds'."""
    row.setdefault('cond', row.get('label'))
    row.setdefault('wall_s', row.get('wall_seconds'))
    return row


def summarize(rows):
    by_cond = collections.defaultdict(list)
    for row in rows:
        by_cond[row['cond']].append(row)
    table = {}
    for cond, runs in by_cond.items():
        with_session = [r for r in runs if r.get('session_metrics')]
        total_score = sum(score_of(r)[0] for r in runs)
        total_max = sum(score_of(r)[1] for r in runs)
        nav = sum(r['session_metrics']['nav_calls'] for r in with_session)
        cbm = sum(r['session_metrics']['cbm_calls'] for r in with_session)

        def mean_of(key):
            values = [r['session_metrics'][key] for r in with_session]
            return round(statistics.mean(values), 1) if values else None
        table[cond] = {
            'runs': len(runs), 'sessions_found': len(with_session),
            'score': f"{total_score:g}/{total_max:g}", 'score_pct': round(100 * total_score / total_max, 1) if total_max else None,
            'pass_runs': sum(passed(r) for r in runs),
            'mean_wall_s': round(statistics.mean(r['wall_s'] for r in runs), 1),
            'median_wall_s': round(statistics.median(r['wall_s'] for r in runs), 1),
            'mean_steps': mean_of('steps'), 'mean_nav_calls': mean_of('nav_calls'), 'mean_nav_chars': mean_of('nav_chars'),
            'mean_cbm_calls': mean_of('cbm_calls'), 'mean_cbm_chars': mean_of('cbm_chars'),
            'adoption_pct': round(100 * cbm / (cbm + nav), 1) if (cbm + nav) else None,
            'mean_peak_context': mean_of('peak_context'), 'max_peak_context': max((r['session_metrics']['peak_context'] for r in with_session), default=None),
            'timeouts': sum(bool(r.get('timed_out')) for r in runs),
        }
    return table


def markdown(table, rows):
    columns = ['runs', 'score', 'score_pct', 'pass_runs', 'mean_wall_s', 'median_wall_s', 'mean_steps',
               'mean_nav_calls', 'mean_nav_chars', 'mean_cbm_calls', 'mean_cbm_chars', 'adoption_pct',
               'mean_peak_context', 'max_peak_context', 'timeouts', 'sessions_found']
    lines = ['| cond | ' + ' | '.join(columns) + ' |', '|---' * (len(columns) + 1) + '|']
    lines += [f"| {cond} | " + ' | '.join(str(stats[c]) for c in columns) + ' |' for cond, stats in sorted(table.items())]
    lines += ['', '| task | cond | rep | score | wall_s | steps | nav calls | nav chars | cbm calls | cbm chars | peak ctx |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for row in sorted(rows, key=lambda r: (r.get('task', ''), r['cond'], r.get('rep', 0))):
        s = row.get('session_metrics') or {}
        got, out_of = score_of(row)
        lines.append(f"| {row.get('task', '?')} | {row['cond']} | {row.get('rep', 0)} | {got:g}/{out_of:g} | {row['wall_s']} | "
                     f"{s.get('steps', '-')} | {s.get('nav_calls', '-')} | {s.get('nav_chars', '-')} | {s.get('cbm_calls', '-')} | "
                     f"{s.get('cbm_chars', '-')} | {s.get('peak_context', '-')} |")
    return '\n'.join(lines) + '\n'


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+')
    parser.add_argument('--sessions', default=str(root / 'dsh-home' / 'sessions'))
    parser.add_argument('--out-md')
    parser.add_argument('--out-json')
    args = parser.parse_args()
    rows = [normalize_row(json.loads(line)) for path in args.results for line in open(path) if line.strip()]
    for row in rows:
        if isinstance(row.get('session'), str):
            session_path = pathlib.Path(row['session'])
        elif row.get('workdir'):
            session_path = find_session(pathlib.Path(args.sessions), row['workdir'], row['started_at'], row['wall_s'])
        else:
            session_path = None
        row['session_path'] = str(session_path) if session_path else None
        row['session_metrics'] = session_metrics(session_path) if session_path else None
    table = summarize(rows)
    text = markdown(table, rows)
    print(text)
    if args.out_md:
        pathlib.Path(args.out_md).write_text(text)
    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps({'summary': table, 'runs': rows}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

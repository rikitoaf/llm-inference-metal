#!/usr/bin/env python3
"""How much would a copy-from-context continuation add to DFlash's 8-row verify on real agent sessions?

Replays the model outputs of dsh sessions (reasoning, text and tool-call arguments of every assistant step, tokenized
with the model's tokenizer) against everything the model had seen before each step (the user messages, tool results
and earlier outputs of the session, plus the step's own tokens so far), and simulates three verify policies per cycle:

  chain8   DFlash alone: anchor + 7 drafted tokens; each drafted token is accepted with probability --alpha while the
           ones before it were (the measured 400-token code run gives 5.71 tokens per cycle, alpha ~0.86).
  ext16    16 verify rows: DFlash's 7, then up to 8 more from the longest suffix match in the context. The extension
           only counts when all 7 drafted tokens were accepted; its tokens are accepted while they equal the real
           continuation.
  copy8    8 rows: when the suffix match predicts more than DFlash accepts, the 7 proposals come from the copy (an
           oracle choice; a real one decides on match length alone).
  rule8    8 rows: the 7 proposals come from the copy whenever the suffix match is at least --rule-n tokens long.
  hybrid   ext16, but when the suffix match at the start of the cycle (at least --min-match tokens) predicts more of the
           real continuation than DFlash's expectation, the whole 15-token chain comes from the copy.

DFlash is modelled, not run, so treat the numbers as an estimate of the copy share, not as a speed prediction.

    splash/splash-src/.venv/bin/python splash/tools/copy_potential.py SESSION.jsonl ... [--alpha 0.86]
Sessions are dsh's session.v3.jsonl (decompress the .zstd with `zstd -d`).
"""
import argparse, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOKENIZERS = ROOT / "splash" / "splash-src" / "install" / "models"


def tokenizer():
    from tokenizers import Tokenizer
    for path in sorted(TOKENIZERS.glob(".*/*/tokenizer/tokenizer.json")):
        return Tokenizer.from_file(str(path))
    sys.exit("error: no tokenizer.json under splash-src/install/models")


def texts(content):
    out = []
    for part in content if isinstance(content, list) else [content]:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict):
            for key in ("text", "arguments"):
                if isinstance(part.get(key), str):
                    out.append(part[key])
            if isinstance(part.get("content"), (list, str)):
                out.extend(texts(part["content"]))
    return out


def events(path):
    """(kind, text) in session order; kind is 'out' for model output, 'in' for everything else it reads."""
    for line in open(path):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind, data = o.get("type"), o.get("data") or {}
        if kind == "assistant/message":
            yield "out", "".join(texts(data.get("message", {}).get("content", [])))
        elif kind in ("user/message", "tool/result", "system/message"):
            yield "in", "\n".join(texts(data.get("message", {}).get("content", data.get("content", []))))


class SuffixIndex:
    """Longest suffix match of the history: the n-gram (n = 8 down to --min-match) ending at the history's end, found
    at its most recent earlier occurrence; proposes the tokens that followed it."""

    def __init__(self, min_match):
        self.tokens, self.table, self.previous, self.min_match = [], {}, {}, min_match

    def append(self, token):
        self.tokens.append(token)
        end = len(self.tokens)
        for n in range(self.min_match, 9):
            if end >= n:
                key = tuple(self.tokens[end - n:end])
                if key in self.table:
                    self.previous[(end, n)] = self.table[key]     # the earlier occurrence of the n-gram ending here
                self.table[key] = end

    def propose(self, history_end, count):
        for n in range(8, self.min_match - 1, -1):
            at = self.previous.get((history_end, n))
            if at is not None:
                return self.tokens[at:min(at + count, history_end)], n
        return [], 0


def simulate(sessions, alpha, min_match, seed, rule_n=8):
    tok = tokenizer()
    rng = random.Random(seed)
    totals = {p: [0, 0] for p in ("chain8", "copy8", "rule8", "ext16", "hybrid")}   # tokens, cycles
    copy_tokens = out_tokens = 0
    for path in sessions:
        index = SuffixIndex(min_match)
        for kind, text in events(path):
            ids = tok.encode(text, add_special_tokens=False).ids if text else []
            if kind == "in":
                for t in ids:
                    index.append(t)
                continue
            base = len(index.tokens)
            # Proposals at each output position from the history before it, then the token joins the history.
            proposals = []
            for t in ids:
                proposals.append(index.propose(len(index.tokens), 15))
                index.append(t)
            out_tokens += len(ids)

            def matched(i, limit):
                prop, n = proposals[i] if i < len(ids) else ([], 0)
                run = 0
                while run < min(len(prop), limit) and i + run < len(ids) and prop[run] == ids[i + run]:
                    run += 1
                return run, n

            copy_tokens += sum(1 for i in range(len(ids)) if matched(i, 1)[0])
            for policy in totals:
                p = 0
                while p < len(ids):
                    drafted = 0
                    while drafted < 7 and p + drafted < len(ids) and rng.random() < alpha:
                        drafted += 1
                    accepted = drafted + 1                             # + the correction or bonus token
                    if policy == "rule8":                              # a rule: copy when the match is long enough
                        run, n = matched(p, 7)
                        if n >= rule_n:
                            accepted = run + 1
                    if policy == "copy8":
                        run, n = matched(p, 7)
                        if n and run + 1 > accepted:
                            accepted = run + 1
                    if policy == "hybrid":
                        run, n = matched(p, 15)
                        if n and run + 1 > accepted:
                            accepted = run + 1
                    if policy in ("ext16", "hybrid") and accepted == 8 and drafted == 7:
                        accepted += matched(p + 8, 8)[0]
                    accepted = min(accepted, len(ids) - p)
                    totals[policy][0] += accepted
                    totals[policy][1] += 1
                    p += accepted
    print(f"{len(sessions)} sessions, {out_tokens} output tokens; a suffix match predicts {copy_tokens / max(out_tokens, 1):.1%} of them")
    base_tpc = totals["chain8"][0] / max(totals["chain8"][1], 1)
    for policy, (tokens, cycles) in totals.items():
        tpc = tokens / max(cycles, 1)
        cost = 1.0 if policy in ("chain8", "copy8", "rule8") else 1.05      # 16 verify rows: +5% per cycle (decode-profile B2)
        print(f"  {policy:7s} {tpc:5.2f} tokens/cycle  speed vs chain8 {tpc / cost / base_tpc:5.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--alpha", type=float, default=0.86)
    ap.add_argument("--min-match", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rule-n", type=int, default=8, help="rule8: copy when the suffix match is at least this long")
    a = ap.parse_args()
    simulate(a.sessions, a.alpha, a.min_match, a.seed, a.rule_n)


if __name__ == "__main__":
    main()

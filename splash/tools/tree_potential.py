#!/usr/bin/env python3
"""Exact T=0 simulation of draft-tree verification from a SPLASH_DRAFT_DUMP file (splash/tools/tree_collect.py).

Each dump line is one decode cycle: the anchor, the target's output tokens, and draft_select_dflash's inputs: 16
candidates per position (7 positions), their unary scores and the edge tables (score of candidate j after
candidate i of the previous position). DFlash's unary scores do not depend on the chosen tokens, so the score of any
path is known: the sum of unary + edge along it. The greedy target sequence is the concatenation of the cycles'
outputs, so whether a tree path is accepted is exact at T=0.

A tree of B nodes (B + 1 verify rows with the anchor) is built best first (DDTree): the path probability is the product
of per-step softmaxes over the 16 children of a node; the heap pops the most probable frontier node. Accepted = the
longest root path whose tokens equal the target's, plus one (the correction or bonus token).

    python3 splash/tools/tree_potential.py logs/speed/draft-dump.jsonl
"""
import heapq, json, math, sys
from collections import defaultdict

C, P = 16, 7


def load(path):
    cycles = [json.loads(line) for line in open(path)]
    # Requests are runs of cycles whose positions follow each other.
    runs, run = [], []
    for c in cycles:
        if run and c["position"] != run[-1]["position"] + run[-1]["retained"]:
            runs.append(run)
            run = []
        run.append(c)
    if run:
        runs.append(run)
    return runs


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def tree(c, budget):
    """{(depth, path tuple of candidate indices)}: the B most probable paths, best first."""
    unary, edges = c["unary"], c["edges"]

    def child_probs(depth, parent):
        row = [unary[depth * C + j] + edges[(depth * C + parent) * C + j] for j in range(C)]
        return softmax(row)

    nodes = []
    heap = []
    probs0 = child_probs(0, 0)
    for j in range(C):
        heapq.heappush(heap, (-probs0[j], (j,)))
    while heap and len(nodes) < budget:
        negp, path = heapq.heappop(heap)
        nodes.append(path)
        depth = len(path)
        if depth < P:
            probs = child_probs(depth, path[-1])
            for j in range(C):
                heapq.heappush(heap, (negp * probs[j], path + (j,)))
    return nodes


def best_chain(c, beam):
    """The 7-token chain with the largest expected accepted length under the draft's own probabilities (sum over k of
    the product of the first k step probabilities), by beam search; beam 1 is the greedy walk of draft_select_dflash."""
    unary, edges = c["unary"], c["edges"]
    beams = [((), 1.0, 0.0)]           # path, prefix probability, expected length so far
    for depth in range(P):
        grown = []
        for path, prefix, expected in beams:
            parent = path[-1] if path else 0
            probs = softmax([unary[depth * C + j] + edges[(depth * C + parent) * C + j] for j in range(C)])
            for j in range(C):
                q = prefix * probs[j]
                grown.append((path + (j,), q, expected + q))
        # rank by expected length plus the best the remaining steps could add
        grown.sort(key=lambda b: -(b[2] + b[1] * (P - depth - 1)))
        beams = grown[:beam]
    best = max(beams, key=lambda b: b[2])[0]
    return [best[:k + 1] for k in range(P)]


def ranked_children(c, depth, parent):
    """The candidate indices of a node's children, most probable first (unary + edge from the parent)."""
    unary, edges = c["unary"], c["edges"]
    row = [unary[depth * C + j] + edges[(depth * C + parent) * C + j] for j in range(C)]
    return sorted(range(C), key=lambda j: -row[j])


def static_paths(c, shape):
    """A fixed tree shape (paths of child ranks) filled with this cycle's candidates."""
    out = {}
    for ranks in shape:                       # parents come before children in `shape`
        parent = out[ranks[:-1]][-1] if len(ranks) > 1 else 0
        j = ranked_children(c, len(ranks) - 1, parent)[ranks[-1]]
        out[ranks] = (out[ranks[:-1]] if len(ranks) > 1 else ()) + (j,)
    return list(out.values())


def learn_shape(cycles, budget):
    """Greedy static shape: repeatedly add the rank path (extending the shape) that most raises the mean accepted
    length over `cycles` [(cycle, truth)]. Each cycle's ranks of the true path are precomputed."""
    true_ranks = []
    for c, truth in cycles:
        ranks, parent = [], 0
        for depth in range(P):
            order = ranked_children(c, depth, parent)
            ids = [c["candidates"][depth * C + j] for j in order]
            if truth[depth] not in ids:
                break
            r = ids.index(truth[depth])
            ranks.append(r)
            parent = order[r]
        true_ranks.append(tuple(ranks))
    shape, chosen = [], set()
    while len(shape) < budget:
        frontier = {(r,) for r in range(C)} - chosen
        for path in chosen:
            if len(path) < P:
                frontier |= {path + (r,) for r in range(C)} - chosen
        gains = defaultdict(int)
        for ranks in true_ranks:
            for k in range(1, len(ranks) + 1):
                if ranks[:k] in frontier and all(ranks[:i] in chosen for i in range(1, k)):
                    gains[ranks[:k]] += 1
        if not gains:
            break
        best = max(gains, key=gains.get)
        shape.append(best)
        chosen.add(best)
    return shape


def accepted(c, paths, truth):
    ids = c["candidates"]
    tokens = {tuple(ids[d * C + j] for d, j in enumerate(p)) for p in paths}
    depth = 0
    while depth < min(P, len(truth)) and tuple(truth[:depth + 1]) in tokens:
        depth += 1
    return depth


def main():
    path = sys.argv[1]
    runs = load(path)
    budgets = [7, 11, 15, 23, 31]
    totals = defaultdict(int)
    cycles = mismatch = 0
    for run in runs:
        seq = []
        for c in run:
            seq.extend(c["output"])
        offset = 0
        for c in run:
            truth = seq[offset:offset + P + 1]
            offset += c["retained"]
            if len(truth) < P + 1:
                continue                       # the end of a request: the truth is cut
            chain = 0
            while chain < P and c["proposed"][chain] == truth[chain]:
                chain += 1
            mismatch += chain != c["accepted"]
            totals["chain8"] += chain + 1
            for beam in (1, 4, 16):
                totals[f"chain8-beam{beam}"] += accepted(c, best_chain(c, beam), truth) + 1
            for b in budgets:
                totals[f"tree{b + 1}"] += accepted(c, tree(c, b), truth) + 1
            cycles += 1
    print(f"{cycles} cycles in {len(runs)} requests; chain re-derived != logged accepted in {mismatch}")
    base = totals["chain8"] / cycles
    cost = {"chain8": 1.0, "tree8": 1.0, "tree12": 1.05, "tree16": 1.05, "tree24": 1.38, "tree32": 1.40}
    for name, t in totals.items():
        tpc = t / cycles
        print(f"  {name:7s} {tpc:5.2f} tokens/cycle  x{tpc / base:4.2f} tokens  "
              f"x{tpc / base / cost.get(name, 1.0):4.2f} speed (cycle cost x{cost.get(name, 1.0)})")


if __name__ == "__main__":
    main()

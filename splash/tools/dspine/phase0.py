#!/usr/bin/env python3
"""Phase 0 offline gate: DSpine vs DFlash2 draft trees on the same greedy (T=0) text.

Inputs per corpus (prompt file P.txt, one prompt per line of token ids):
  P.out.txt       decode-run --output: each prompt's generated tokens (the first is the prefill's anchor)
  P.dflash.jsonl  SPLASH_DRAFT_DUMP of that decode-run (tree on, copy off): DFlash2's per-cycle candidates, unary,
                  edge tables, and the target's kept tokens
  P.capture.bin   SPLASH_CAPTURE_DUMP (layers 1,16,31,46,61[,63]) of a prefill of prompt + generated tokens

  phase0.py collect P [--windows 2048 8192] [--limit N] [--shortlist 1] [--quantize 0]   (MLX, GPU)
      -> P.dspine_TAG.npz: for every DFlash2 cycle with >= 16 truth tokens, DSpine's candidates and tables at
         block 16 (15 x 16, 15 x 16 x 16) and block 8 (7 x 16, 7 x 16 x 16) for each window. Rows 0..7 of a
         block-causal block are mathematically an 8-row block, but bf16 GEMMs of 8 vs 16 rows round differently
         (check8: ~1 bf16 ulp per layer, flipping a near-tied 16th candidate now and then), so both are run.
  phase0.py score P [P2 ...]                                                             (CPU)
      -> tokens/cycle tables (chain and best-first tree, DFlash2 vs DSpine) and per-depth acceptance
  phase0.py checkhead P            the target head + layer-63 capture reproduce the greedy tokens (Q4_K check)
  phase0.py check8 P               an 8-row DSpine block equals rows 0..7 of a 16-row block
  phase0.py golden P OUTDIR [--prompt I] [--cycles 20] [--window 2048]   golden files for the Metal port

Cycle alignment: DSpine is evaluated at DFlash2's cycle boundaries (anchor position p, anchor token) from the dump.
At T=0 the target's greedy continuation is fixed, so "accepted" is exact for any draft at a given boundary; but in a
real DSpine run the boundaries would follow DSpine's own acceptances, so per-cycle sums here approximate a real run
(acceptance per cycle is measured on the same states, which is the fair comparison).
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

C = 16
DSPINE_LAYERS = (1, 16, 31, 46, 61)


# ---------------------------------------------------------------- corpus

def load_corpus(prefix):
    prefix = Path(prefix)
    prompts = [list(map(int, l.split())) for l in open(f"{prefix}.txt") if l.strip()]
    outputs = [list(map(int, l.split())) for l in open(f"{prefix}.out.txt") if l.strip()]
    assert len(prompts) == len(outputs)
    cycles = [json.loads(l) for l in open(f"{prefix}.dflash.jsonl")]
    # runs of consecutive cycles; the run whose first position is a prompt's length belongs to that prompt
    runs, run = [], []
    for c in cycles:
        if run and c["position"] != run[-1]["position"] + run[-1]["retained"]:
            runs.append(run)
            run = []
        run.append(c)
    if run:
        runs.append(run)
    corpus = []
    used = set()
    for i, (prompt, out) in enumerate(zip(prompts, outputs)):
        match = [k for k, r in enumerate(runs) if k not in used and r[0]["position"] == len(prompt)
                 and r[0]["anchor"] == out[0]]
        assert match, f"no dump run for prompt {i}"
        k = match[0]
        used.add(k)
        seq = prompt + out
        # consistency: each cycle's anchor is the token at its position, its output the following tokens
        for c in runs[k]:
            p = c["position"]
            assert seq[p] == c["anchor"], (i, p)
            kept = seq[p + 1:p + 1 + c["retained"]]          # the request's last cycle can be cut by max tokens
            assert kept == c["output"][:len(kept)], (i, p)
        corpus.append({"prompt": prompt, "out": out, "seq": seq, "cycles": runs[k]})
    return corpus


# ---------------------------------------------------------------- trees (draft_select_tree's rule)

def log_softmax(row):
    m = max(row)
    s = sum(math.exp(x - m) for x in row)
    return [x - m - math.log(s) for x in row]


class Scores:
    """score(depth, from, j) = unary[depth][j] + edges[depth][from][j] over `positions` positions."""

    def __init__(self, candidates, unary, edges):
        self.cand = np.asarray(candidates).reshape(-1, C)
        self.positions = self.cand.shape[0]
        self.unary = np.asarray(unary, dtype=np.float64).reshape(self.positions, C)
        self.edges = np.asarray(edges, dtype=np.float64).reshape(self.positions, C, C)
        self._ls = {}

    def logp(self, depth, frm):
        key = (depth, frm)
        if key not in self._ls:
            self._ls[key] = log_softmax(list(self.unary[depth] + self.edges[depth, frm]))
        return self._ls[key]

    def limited(self, positions):
        s = Scores.__new__(Scores)
        s.cand, s.unary, s.edges = self.cand[:positions], self.unary[:positions], self.edges[:positions]
        s.positions, s._ls = positions, {}
        return s


def tree_rows(s, budget):
    """draft_select_tree: rows 1..budget, each the most probable child not yet taken of any node above the last
    position (ties: the first node, then the first child, as the Metal loop's strict >). Row 0 is the anchor.
    Returns per row: candidate index, depth, parent, log-probability, token path."""
    cand_idx, depth, parent, logp, paths = [0], [0], [-1], [0.0], [()]
    taken = [set()]
    for _ in range(budget):
        best, bn, bj = -math.inf, -1, -1
        for n in range(len(depth)):
            if depth[n] >= s.positions:
                continue
            row = s.logp(depth[n], cand_idx[n])
            for j in range(C):
                if j in taken[n]:
                    continue
                v = logp[n] + row[j]
                if v > best:
                    best, bn, bj = v, n, j
        if bn < 0:
            break
        taken[bn].add(bj)
        cand_idx.append(bj)
        depth.append(depth[bn] + 1)
        parent.append(bn)
        logp.append(best)
        taken.append(set())
        paths.append(paths[bn] + (int(s.cand[depth[bn], bj]),))
    return {"depth": depth, "parent": parent, "logp": logp, "paths": paths}


def best_first_tree(s, budget):
    return set(tree_rows(s, budget)["paths"][1:])


# Runtime.mm kCopyAcceptance[match - 1][k] (SPLASH_DRAFT_COPY=1, calibrated on the eval sessions)
COPY_ACCEPTANCE = [
    [0.47, 0.57, 0.67, 0.69, 0.73, 0.75, 0.79, 0.81],
    [0.47, 0.57, 0.67, 0.69, 0.73, 0.75, 0.79, 0.81],
    [0.47, 0.57, 0.67, 0.69, 0.73, 0.75, 0.79, 0.81],
    [0.55, 0.64, 0.68, 0.72, 0.75, 0.78, 0.81, 0.83],
    [0.62, 0.67, 0.71, 0.75, 0.78, 0.80, 0.83, 0.85],
    [0.65, 0.70, 0.75, 0.78, 0.80, 0.83, 0.85, 0.85],
    [0.68, 0.74, 0.78, 0.80, 0.82, 0.85, 0.85, 0.86],
    [0.86, 0.89, 0.91, 0.92, 0.93, 0.94, 0.94, 0.95],
]


class History:
    """A request's token sequence as the copy history: seq[:length], with a token -> positions index."""

    def __init__(self, seq):
        import bisect
        self.seq, self._bisect = seq, bisect
        self.index = defaultdict(list)
        for i, t in enumerate(seq):
            self.index[t].append(i)
        self.length = len(seq)

    def at(self, length):
        view = History.__new__(History)
        view.seq, view._bisect, view.index, view.length = self.seq, self._bisect, self.index, length
        return view

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        assert i < self.length
        return self.seq[i]

    def positions(self, token, length):
        lst = self.index.get(token, [])
        return lst[:self._bisect.bisect_left(lst, length)]


def copy_extend(tree, anchor, history, positions=7, min_match=3, rows_total=16, match_len=8):
    """draft_copy_extend on a 16-row tree (rows 0..15): the first depth-7 row is the tip; its 8-token path (anchor +
    proposals) is matched against the history (prompt + kept tokens before the anchor), longest match >= min_match,
    latest on ties; the following tokens chain below the tip at depths 8.. replacing the tree's last rows while
    tip log-prob + sum(log prior) beats the replaced row. Returns the set of token paths of the extended tree."""
    depth, logp, paths = tree["depth"], tree["logp"], tree["paths"]
    tip = next((r for r in range(1, len(depth)) if depth[r] == positions), 0)
    out = set(paths[1:])
    if not tip or not history:
        return out, 0
    query = [anchor] + list(paths[tip])
    length = len(history)
    best = (0, -1)
    last = query[-1]
    candidates_at = history.positions(last, length) if hasattr(history, "positions") else \
        [i for i in range(length) if history[i] == last]
    for i in candidates_at:
        n = 1
        while n < match_len and n <= i and history[i - n] == query[match_len - 1 - n]:
            n += 1
        if n >= min_match and (n, i) > best:
            best = (n, i)
    match, end = best
    if not match:
        return out, 0
    copy = [history[end + 1 + k] if end + 1 + k < length else query[end + 1 + k - length] for k in range(8)]
    rows, value = 0, logp[tip]
    while rows < 8 and rows_total - 1 - rows > tip:
        nxt = value + math.log(COPY_ACCEPTANCE[match - 1][rows])
        if not nxt > logp[rows_total - 1 - rows]:
            break
        value = nxt
        rows += 1
    kept = set(paths[1:rows_total - rows])
    for k in range(rows):
        kept.add(tuple(paths[tip]) + tuple(copy[:k + 1]))
    return kept, rows


def greedy_chain(s):
    """draft_select_dflash at T=0: walk the positions taking the best score from the previous choice."""
    frm, out = 0, []
    for d in range(s.positions):
        row = s.unary[d] + s.edges[d, frm]
        j = int(np.argmax(row))
        out.append(int(s.cand[d, j]))
        frm = j
    return out


def accepted_tree(paths, truth, positions):
    d = 0
    while d < positions and d < len(truth) and tuple(truth[:d + 1]) in paths:
        d += 1
    return d


def accepted_chain(chain, truth):
    d = 0
    while d < len(chain) and d < len(truth) and chain[d] == truth[d]:
        d += 1
    return d


def oracle(s, truth):
    """Depth reachable if a perfect selector picked among the candidates (the truth's token in every position's
    candidate list, along the path)."""
    d = 0
    while d < s.positions and d < len(truth) and truth[d] in set(s.cand[d].tolist()):
        d += 1
    return d


# ---------------------------------------------------------------- collect (MLX)

def capture_ids(prefix, prompts):
    """The capture file and the request ids of the corpus' prompts (decode-run's warm-up, request 1, skipped)."""
    from dspine_reference import read_capture
    capture = read_capture(f"{prefix}.capture.bin")
    ids = [k for k in capture.ids if k != 1]
    assert len(ids) == prompts, (ids, prompts)
    return capture, ids

def truth_of(entry, c, depth):
    p = c["position"]
    return entry["seq"][p + 1:p + 1 + depth]


def collect(args):
    import mlx.core as mx
    from dspine_reference import DSpine, read_capture, select_layers
    corpus = load_corpus(args.prefix)
    capture, ids = capture_ids(args.prefix, len(corpus))
    t0 = time.time()
    ref = DSpine(quantize_bits=args.quantize)
    print(f"loaded DSpine + target head in {time.time() - t0:.1f}s", flush=True)
    results = {}
    for i, entry in enumerate(corpus):
        layers, rows = capture.layers(ids[i]), capture.rows(ids[i])
        assert rows.shape[0] >= len(entry["seq"]) - 1, (rows.shape, len(entry["seq"]))
        hidden = select_layers(layers, rows, DSPINE_LAYERS)
        del rows
        t1 = time.time()
        ref.context(hidden)
        cycles = [c for c in entry["cycles"] if len(truth_of(entry, c, 16)) == 16]
        if args.limit:
            cycles = cycles[:args.limit]
        for w in args.windows:
            cand = np.zeros((len(cycles), 15, C), np.int32)
            table = np.zeros((len(cycles), 15, C, C), np.float32)
            cand8 = np.zeros((len(cycles), 7, C), np.int32)
            table8 = np.zeros((len(cycles), 7, C, C), np.float32)
            for k, c in enumerate(cycles):
                cand[k], table[k], *_ = ref.propose(c["anchor"], c["position"], w, 16, bool(args.shortlist))
                cand8[k], table8[k], *_ = ref.propose(c["anchor"], c["position"], w, 8, bool(args.shortlist))
            results[f"p{i}_w{w}_cand"] = cand
            results[f"p{i}_w{w}_table"] = table
            results[f"p{i}_w{w}_cand8"] = cand8
            results[f"p{i}_w{w}_table8"] = table8
        results[f"p{i}_positions"] = np.array([c["position"] for c in cycles], np.int64)
        print(f"prompt {i}: {len(entry['prompt'])} prompt tokens, {len(cycles)} cycles, "
              f"{time.time() - t1:.1f}s", flush=True)
        mx.clear_cache()
    suffix = args.tag or (("sl" if args.shortlist else "full") + (f"_q{args.quantize}" if args.quantize else ""))
    np.savez_compressed(f"{args.prefix}.dspine_{suffix}.npz", **results)
    print("wrote", f"{args.prefix}.dspine_{suffix}.npz")


# ---------------------------------------------------------------- score

def score(args):
    total = defaultdict(float)
    depth_hits = defaultdict(lambda: np.zeros(16))
    depth_seen = defaultdict(lambda: np.zeros(16))
    per_prompt = []
    ncycles = tree_mismatch = 0
    for prefix in args.prefixes:
        corpus = load_corpus(prefix)
        npz = np.load(f"{prefix}.dspine_{args.tag}.npz")
        windows = sorted({int(k.split("_w")[1].split("_")[0]) for k in npz.files if "_w" in k})
        for i, entry in enumerate(corpus):
            positions = npz[f"p{i}_positions"]
            byp = {c["position"]: c for c in entry["cycles"]}
            local = defaultdict(float)
            for k, p in enumerate(positions):
                c = byp[int(p)]
                truth = truth_of(entry, c, 16)
                rows = {}
                # DFlash2: the dump's 7 positions
                df = Scores(c["candidates"], c["unary"], c["edges"])
                # (in tree mode the dump's "proposed" holds tree rows, not a chain: the chain is re-derived)
                rows["dflash2 chain7"] = accepted_chain(greedy_chain(df), truth)
                dft = tree_rows(df, 15)
                rows["dflash2 tree16"] = accepted_tree(set(dft["paths"][1:]), truth, 7)
                tree_mismatch += rows["dflash2 tree16"] != c["accepted"]
                history = entry.setdefault("history", History(entry["seq"])).at(c["position"])
                rows["dflash2 tree16+copy"] = accepted_tree(copy_extend(dft, c["anchor"], history)[0], truth, 15)
                rows["dflash2 oracle7"] = oracle(df, truth)
                for w in windows:
                    cand = npz[f"p{i}_w{w}_cand"][k]
                    table = npz[f"p{i}_w{w}_table"][k]
                    ds = Scores(cand, np.zeros((15, C)), table)
                    if f"p{i}_w{w}_cand8" in npz.files:      # a real 8-row block (bf16 GEMM rounding differs)
                        d8 = Scores(npz[f"p{i}_w{w}_cand8"][k], np.zeros((7, C)), npz[f"p{i}_w{w}_table8"][k])
                    else:
                        d8 = ds.limited(7)
                    rows[f"dspine w{w} b16 first7 tree16"] = accepted_tree(best_first_tree(ds.limited(7), 15),
                                                                           truth, 7)
                    rows[f"dspine w{w} b8 chain7"] = accepted_chain(greedy_chain(d8), truth)
                    d8t = tree_rows(d8, 15)
                    rows[f"dspine w{w} b8 tree16"] = accepted_tree(set(d8t["paths"][1:]), truth, 7)
                    rows[f"dspine w{w} b8 tree16+copy"] = accepted_tree(
                        copy_extend(d8t, c["anchor"], history)[0], truth, 15)
                    rows[f"dspine w{w} b8 oracle7"] = oracle(d8, truth)
                    rows[f"dspine w{w} b16 chain15"] = accepted_chain(greedy_chain(ds), truth)
                    rows[f"dspine w{w} b16 tree16"] = accepted_tree(best_first_tree(ds, 15), truth, 15)
                    if args.big:
                        rows[f"dspine w{w} b16 tree32"] = accepted_tree(best_first_tree(ds, 31), truth, 15)
                    rows[f"dspine w{w} b16 oracle15"] = oracle(ds, truth)
                for name, acc in rows.items():
                    total[name] += acc + 1
                    local[name] += acc + 1
                    if ("chain" in name or "tree" in name) and "copy" not in name:
                        for d in range(acc + 1):
                            depth_seen[name][d] += 1
                        for d in range(acc):
                            depth_hits[name][d] += 1
                ncycles += 1
            per_prompt.append((f"{Path(prefix).name}#{i}", len(entry["prompt"]), len(positions), dict(local)))
    print(f"{ncycles} cycles (DFlash2's cycle boundaries, >= 16 truth tokens each), tokens/cycle = accepted + 1")
    print(f"DFlash2 tree re-derived from the dump != the engine's logged acceptance in {tree_mismatch} cycles")
    base_chain = total["dflash2 chain7"] / ncycles
    base_tree = total["dflash2 tree16"] / ncycles
    print(f"{'drafter / selection':34s} {'tok/cycle':>9s} {'vs DF2 chain7':>13s} {'vs DF2 tree16':>13s}")
    for name in sorted(total, key=lambda n: (not n.startswith("dflash2"), n)):
        t = total[name] / ncycles
        print(f"{name:34s} {t:9.3f} {100 * (t / base_chain - 1):+12.1f}% {100 * (t / base_tree - 1):+12.1f}%")
    print("\nP(depth d accepted | depth d-1 accepted), d = 1..")
    for name in sorted(depth_seen, key=lambda n: (not n.startswith("dflash2"), n)):
        seen, hits = depth_seen[name], depth_hits[name]
        probs = [hits[d] / seen[d] if seen[d] else float("nan") for d in range(16) if seen[d]]
        print(f"{name:34s} " + " ".join(f"{x:.2f}" for x in probs))
    print("\nper prompt (tokens/cycle): DF2 tree16 | DSpine b8 tree16 per window | DSpine b16 tree16 per window")
    for name, plen, n, local in per_prompt:
        ws = sorted({k.split()[1] for k in local if k.startswith("dspine")})
        b8 = " ".join(f"{local[f'dspine {w} b8 tree16'] / n:.2f}" for w in ws)
        b16 = " ".join(f"{local[f'dspine {w} b16 tree16'] / n:.2f}" for w in ws)
        print(f"  {name:22s} {plen:6d} tok {n:4d} cyc  {local['dflash2 tree16'] / n:.2f} | {b8} | {b16}")


# ---------------------------------------------------------------- checks

def checkhead(args):
    """Final norm (GGUF output_norm) + head on the layer-63 capture must give the greedy tokens."""
    import mlx.core as mx
    from dspine_reference import DSpine, read_capture, select_layers, DEFAULT_GGUF
    from gguf_tensors import GgufFile
    corpus = load_corpus(args.prefix)
    capture, ids = capture_ids(args.prefix, len(corpus))
    ref = DSpine()
    g = GgufFile(DEFAULT_GGUF)
    norm = mx.array(g.rows("output_norm.weight", 0, 1)[0] if len(g.info("output_norm.weight")[0]) > 1
                    else _vector(g, "output_norm.weight"))
    for i, entry in enumerate(corpus[:args.prompts]):
        layers, rows = capture.layers(ids[i]), capture.rows(ids[i])
        h = mx.array(select_layers(layers, rows, (63,))).view(mx.bfloat16).astype(mx.float32)
        P = len(entry["prompt"])
        sl = h[P - 1:len(entry["seq"]) - 1]
        x = sl * mx.rsqrt(mx.mean(sl * sl, axis=-1, keepdims=True) + 1e-6) * norm
        logits = ref.head.matmul(x.astype(mx.bfloat16).astype(mx.float32))
        pred = np.array(mx.argmax(logits, axis=-1))
        truth = np.array(entry["seq"][P:])
        agree = (pred == truth).mean()
        print(f"prompt {i}: argmax(head(norm(layer 63))) == greedy token at {agree:.4f} of {len(truth)} positions")


def _vector(g, name):
    dims, kind, offset = g.info(name)
    assert kind == "F32"
    return np.frombuffer(g.map, dtype="<f4", count=dims[0], offset=g.data_start + offset).copy()


def check8(args):
    import mlx.core as mx
    from dspine_reference import DSpine, read_capture, select_layers
    corpus = load_corpus(args.prefix)
    capture, ids = capture_ids(args.prefix, len(corpus))
    ref = DSpine()
    entry = corpus[0]
    ref.context(select_layers(capture.layers(ids[0]), capture.rows(ids[0]), DSPINE_LAYERS))
    same_rows = rows_total = 0
    rel = []
    cycles = entry["cycles"][:args.cycles]
    for c in cycles:
        c16, t16, o16, *_ = ref.propose(c["anchor"], c["position"], 2048, 16)
        c8, t8, o8, *_ = ref.propose(c["anchor"], c["position"], 2048, 8)
        same_rows += int((c16[:7] == c8).all(1).sum())
        rows_total += 7
        a = np.array(o16[:8].astype(mx.float32))
        b = np.array(o8.astype(mx.float32))
        rel.append(np.abs(a - b).max() / np.abs(a).max())
    print(f"8-row vs 16-row block: candidate lists equal in {same_rows}/{rows_total} rows; final normed rows max "
          f"|diff| / max |value| = {np.median(rel):.4f} median, {max(rel):.4f} max (bf16 GEMM rounding by row count)")


def golden(args):
    import mlx.core as mx
    from dspine_reference import DSpine, read_capture, select_layers
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(args.prefix)
    capture, ids = capture_ids(args.prefix, len(corpus))
    ref = DSpine(quantize_bits=args.quantize)
    entry = corpus[args.prompt]
    hidden = select_layers(capture.layers(ids[args.prompt]), capture.rows(ids[args.prompt]), DSPINE_LAYERS)
    cycles = [c for c in entry["cycles"] if c["position"] >= args.window][:args.cycles]
    lo = cycles[0]["position"] - args.window
    hi = cycles[-1]["position"]
    ref.context(hidden)
    np.save(out / "context_hidden.npy", hidden[lo:hi])                       # uint16 = bf16 bits
    np.save(out / "context_projected.npy", np.array(ref.ctx_projected[lo:hi].view(mx.uint16)))
    meta = {"prompt_tokens": len(entry["prompt"]), "context_first_position": lo, "window": args.window,
            "backbone": f"mlx affine q{args.quantize} group 64" if args.quantize else "bf16",
            "block_rows": 16, "target_layers": list(DSPINE_LAYERS), "shortlist": [0, 98304, 248044, 248320],
            "cycles": []}
    anchors, positions = [], []
    finals, raws, rmss, packeds, cands, tables, layer_outs, logits_top = [], [], [], [], [], [], [], []
    embeds = ref.embedding_rows([cycles[0]["anchor"], ref.model.mask_token_id])
    for c in cycles:
        trace = []
        cand, table, final, stash, logits = ref.propose(c["anchor"], c["position"], args.window, 16, True, trace)
        anchors.append(c["anchor"])
        positions.append(c["position"])
        finals.append(np.array(final.view(mx.uint16)))
        raws.append(np.array(stash["raw"].view(mx.uint16)))
        rmss.append(np.array(stash["rms"]).astype(np.float32))
        packeds.append(np.array(stash["packed"].view(mx.uint16)))
        cands.append(cand)
        tables.append(table)
        layer_outs.append(np.stack([np.array(t.view(mx.uint16)) for t in trace]))
        lg = np.array(logits)
        logits_top.append(np.take_along_axis(lg, cand.astype(np.int64), axis=1).astype(np.float32))
        meta["cycles"].append({"position": c["position"], "anchor": c["anchor"],
                               "truth": entry["seq"][c["position"] + 1:c["position"] + 17]})
    # the same cycles as 8-row blocks (Phase 1 runs 8 rows; bf16 GEMMs of 8 rows round differently from 16)
    b8 = {"final": [], "raw": [], "rms": [], "packed": [], "cand": [], "table": []}
    for c in cycles:
        cand, table, final, stash, _ = ref.propose(c["anchor"], c["position"], args.window, 8, True)
        b8["final"].append(np.array(final.view(mx.uint16)))
        b8["raw"].append(np.array(stash["raw"].view(mx.uint16)))
        b8["rms"].append(np.array(stash["rms"]).astype(np.float32))
        b8["packed"].append(np.array(stash["packed"].view(mx.uint16)))
        b8["cand"].append(cand)
        b8["table"].append(table)
    for key, name in (("final", "final_normed"), ("raw", "stash_raw"), ("rms", "stash_rms"),
                      ("packed", "stash_packed"), ("cand", "candidates"), ("table", "tables")):
        np.save(out / f"{name}_b8.npy", np.stack(b8[key]))
    np.save(out / "anchors.npy", np.array(anchors, np.int32))
    np.save(out / "positions.npy", np.array(positions, np.int64))
    np.save(out / "embedding_anchor0_mask.npy", embeds.astype(np.float32))
    np.save(out / "final_normed.npy", np.stack(finals))
    np.save(out / "layer_outputs_normed.npy", np.stack(layer_outs))
    np.save(out / "stash_raw.npy", np.stack(raws))
    np.save(out / "stash_rms.npy", np.stack(rmss))
    np.save(out / "stash_packed.npy", np.stack(packeds))
    np.save(out / "candidates.npy", np.stack(cands))
    np.save(out / "candidate_logits.npy", np.stack(logits_top))
    np.save(out / "tables.npy", np.stack(tables))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(cycles)} cycles to {out}")


def compare_live(args):
    """A SPLASH_DRAFTER=dspine run's dumped candidates and tables (P.dflash.jsonl of a dspine decode-run) against the
    MLX reference's at the same cycles (P.dspine_TAG.npz from collect): candidate overlap and table differences by
    cycle index within each request, so drift that grows with committed context shows up."""
    corpus = load_corpus(args.prefix)
    npz = np.load(f"{args.prefix}.dspine_{args.tag}.npz")
    by_index = defaultdict(lambda: [0.0, 0.0, 0])
    overlaps, diffs = [], []
    for i, entry in enumerate(corpus):
        byp = {c["position"]: c for c in entry["cycles"]}
        for k, p in enumerate(npz[f"p{i}_positions"]):
            c = byp[int(p)]
            live_cand = np.asarray(c["candidates"]).reshape(15, C)
            live_tab = np.asarray(c["edges"], dtype=np.float64).reshape(15, C, C)
            ref_cand = npz[f"p{i}_w{args.window}_cand"][k]
            ref_tab = npz[f"p{i}_w{args.window}_table"][k]
            cycle_overlap, cycle_diffs = [], []
            for row in range(15):
                ref_index = {int(t): j for j, t in enumerate(ref_cand[row])}
                prev_index = {int(t): j for j, t in enumerate(ref_cand[row - 1])} if row else None
                cycle_overlap.append(sum(int(t) in ref_index for t in live_cand[row]) / C)
                for kk in range(C):
                    rk = 0 if row == 0 else prev_index.get(int(live_cand[row - 1][kk]))
                    if rk is None:
                        continue
                    for j in range(C):
                        rj = ref_index.get(int(live_cand[row][j]))
                        if rj is not None:
                            cycle_diffs.append(abs(live_tab[row, kk, j] - ref_tab[row, rk, rj]))
            overlaps.append(np.mean(cycle_overlap))
            diffs.extend(cycle_diffs)
            slot = by_index[min(k, 40) // 10]
            slot[0] += np.mean(cycle_overlap)
            slot[1] += np.median(cycle_diffs) if cycle_diffs else 0
            slot[2] += 1
    print(f"{len(overlaps)} cycles: candidate overlap mean {np.mean(overlaps):.3f}; table |diff| median "
          f"{np.median(diffs):.4f}, p99 {np.percentile(diffs, 99):.4f}, max {np.max(diffs):.4f}")
    for slot in sorted(by_index):
        o, d, n = by_index[slot]
        print(f"  cycles {slot * 10}..{slot * 10 + 9}{'+' if slot == 4 else ''}: overlap {o / n:.3f}, "
              f"median table diff {d / n:.4f} ({n} cycles)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("collect")
    a.add_argument("prefix")
    a.add_argument("--windows", type=int, nargs="+", default=[2048, 8192])
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--shortlist", type=int, default=1)
    a.add_argument("--quantize", type=int, default=0)
    a.add_argument("--tag", default="")
    a = sub.add_parser("score")
    a.add_argument("prefixes", nargs="+")
    a.add_argument("--tag", default="sl")
    a.add_argument("--big", action="store_true")
    a = sub.add_parser("checkhead")
    a.add_argument("prefix")
    a.add_argument("--prompts", type=int, default=2)
    a = sub.add_parser("check8")
    a.add_argument("prefix")
    a.add_argument("--cycles", type=int, default=10)
    a = sub.add_parser("golden")
    a.add_argument("prefix")
    a.add_argument("outdir")
    a.add_argument("--prompt", type=int, default=0)
    a.add_argument("--cycles", type=int, default=20)
    a.add_argument("--window", type=int, default=2048)
    a.add_argument("--quantize", type=int, default=0, help="4: the backbone quantized as Splash stores it")
    a = sub.add_parser("compare-live")
    a.add_argument("prefix")
    a.add_argument("--tag", default="q4")
    a.add_argument("--window", type=int, default=8192)
    args = ap.parse_args()
    {"compare-live": compare_live, "collect": collect, "score": score, "checkhead": checkhead, "check8": check8, "golden": golden}[args.cmd](args)


if __name__ == "__main__":
    main()

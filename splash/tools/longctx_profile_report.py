#!/usr/bin/env python3
"""Summary of splash/tools/longctx_profile.sh (PLAN-2026-11-17 items 2 and 6; splash/dev-prefill/PREFILL.md).

    longctx_profile_report.py OUTDIR            # reads OUTDIR/runs/*, writes OUTDIR/summary.md and summary.json
    longctx_profile_report.py OUTDIR --fake     # first writes synthetic runs (the script's PROFILE_DRY_RUN=1)

Inputs per run directory (dev-prefill's longctx-profile): chunks.csv, prefill_dispatch_<start>.csv,
decode_cycles.csv, decode_dispatch_<arm>.csv, meta.json. Runs: p100k (2048-row chunks, long100k.txt, decode at
~103K), p70k (long.txt line 0, decode at ~68K), p100k_chunk4096 (the 4096-row build, prefill only).

Every dispatch is classified by its pipeline name and its place in the command (the order of a prefill chunk and a
decode cycle is fixed by Runtime.mm; see PREFILL.md "Classification"). Shares are of the summed per-dispatch GPU time
(each dispatch replayed as its own command); "ms (fused)" scales the shares to the fused command's GPU time.
Unknown pipeline names land in "other" and are listed, so a misclassification is visible.
"""
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

# Qwen3.8-27B target geometry (ANALYSIS.md 5.2) and turbo-int4 KV (4 bits K and V, one f32 scale each per token
# and KV head: 2 x (128 + 4) bytes per token, KV head and attention layer).
QUERY_HEADS, KV_HEADS, HEAD_DIM, ATTENTION_LAYERS = 24, 4, 256, 16
KV_BYTES_PER_TOKEN_HEAD = 2 * (HEAD_DIM // 2 + 4)
PREFILL_TILE_ROWS = 8          # SPLASH_PREFILL_ATTENTION_TILE_ROWS: query rows per threadgroup tile
ATTENTION_GATE = 0.30          # item 2: proceed when attention >= 30% of a prefill chunk at ~100K
DRAFT_GATE = 0.30              # item 6 gate: draft ms -30% at >= 64K

PREFILL_ORDER = ["projections", "gdn", "attention", "attention_aux", "draft_context", "other"]
PREFILL_LABEL = {"projections": "projections (GGUF matmuls)", "gdn": "GDN layers (48)",
                 "attention": "attention kernel (16 layers: rotate + split + reduce)",
                 "attention_aux": "attention q/k/v prep, KV store, gate",
                 "draft_context": "DSpine draft context (captured rows)", "other": "other (norms, rope, embedding, copies)"}
DECODE_ORDER = ["draft_input", "draft_backbone", "draft_window_attention", "draft_bridge", "draft_head",
                "draft_refinement", "draft_tree", "verify_projections", "verify_attention", "verify_attention_aux",
                "verify_gdn", "verify_other", "sampling", "accept", "kv_restore", "gdn_commit", "draft_commit",
                "other"]
DECODE_LABEL = {"draft_input": "draft: rope + embedding", "draft_backbone": "draft: backbone layers (5, Q4 matmuls, norms)",
                "draft_window_attention": "draft: window attention", "draft_bridge": "draft: bridge (bf16 matmuls, codes, message, write)",
                "draft_head": "draft: LM head + top-16", "draft_refinement": "draft: refinement / tables",
                "draft_tree": "draft: tree build (+ copy, tree rope)", "verify_projections": "verify: projections",
                "verify_attention": "verify: attention kernel", "verify_attention_aux": "verify: attention prep/store/gate",
                "verify_gdn": "verify: GDN", "verify_other": "verify: other (norms, embedding, head glue)",
                "sampling": "sampling (target policy)", "accept": "accept", "kv_restore": "commit: KV restore (tree path)",
                "gdn_commit": "commit: GDN state", "draft_commit": "commit: draft context + path gather", "other": "other"}
DRAFT_KEYS = [k for k in DECODE_ORDER if k.startswith("draft_") and k != "draft_commit"]


def read_csv(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------- prefill
def is_target(name):
    return name.startswith("gguf_") or "attention" in name or "gdn" in name


def classify_prefill(names):
    """Category and attention layer (or -1) per dispatch of one prefill chunk."""
    cats, layers = [], []
    context = [i for i, n in enumerate(names) if n.startswith("dspine_context") or n.startswith("prefill_draft_context")]
    # The DSpine context prefill (Runtime.mm addPackedDraftContext) follows the target's last layer: its region runs
    # from the first dispatch after the last target dispatch before it to its last context write.
    region = None
    if context:
        begin = context[0]
        while begin > 0 and not is_target(names[begin - 1]):
            begin -= 1
        region = (begin, context[-1])
    layer = -1
    for i, n in enumerate(names):
        if region and region[0] <= i <= region[1]:
            cats.append("draft_context"); layers.append(-1); continue
        if n.startswith("prefill_attention_qkv"):
            layer += 1
        if n.startswith("turbo_rotate_queries") or (n.startswith("prefill_attention") and
                                                     ("_split" in n or "_reduce" in n)):
            if layer < 0:
                layer = 0
            cats.append("attention"); layers.append(layer)
        elif n.startswith("prefill_attention") or n.startswith("verify_attention"):
            cats.append("attention_aux"); layers.append(max(layer, 0))
        elif "gdn" in n:
            cats.append("gdn"); layers.append(-1)
        elif (n.startswith("gguf_") and not n.startswith("gguf_embed")) or "linear" in n or "moe" in n:
            cats.append("projections"); layers.append(-1)
        else:
            cats.append("other"); layers.append(-1)
    return cats, layers


def attention_math(start, rows, tiles):
    """FLOP and KV bytes of one attention layer for a chunk of `rows` rows after `start` committed tokens."""
    visible_sum = rows * start + rows * (rows + 1) // 2           # causal: row r sees start + r + 1 keys
    flop = 4 * QUERY_HEADS * HEAD_DIM * visible_sum              # Q.K and P.V, 2 FLOP per multiply-add
    unique = (start + rows) * KV_HEADS * KV_BYTES_PER_TOKEN_HEAD
    # Each (KV head, 8-row tile) threadgroup reads its head's keys up to its last row.
    requested = sum(start + PREFILL_TILE_ROWS * (t + 1) for t in range(tiles)) * KV_HEADS * KV_BYTES_PER_TOKEN_HEAD
    return flop, unique, requested


def prefill_profile(path, start):
    rows = read_csv(path)
    names = [r["pipeline"] for r in rows]
    cats, layers = classify_prefill(names)
    by_cat = defaultdict(float)
    per_layer = defaultdict(float)
    unknown = defaultdict(float)
    split_tiles = []
    for r, c, l in zip(rows, cats, layers):
        us = float(r["gpu_us"])
        by_cat[c] += us
        if c == "attention":
            per_layer[l] += us
            if "_split" in r["pipeline"]:
                split_tiles.append(int(r["groups_y"] or 0))
        if c == "other":
            unknown[r["pipeline"]] += us
    total = sum(by_cat.values())
    return {"start": start, "dispatches": len(rows), "total_ms": total / 1e3,
            "ms": {k: by_cat.get(k, 0.0) / 1e3 for k in PREFILL_ORDER},
            "per_layer_ms": [per_layer[k] / 1e3 for k in sorted(per_layer)],
            "split_tiles": split_tiles, "other_names": dict(sorted(unknown.items(), key=lambda kv: -kv[1])[:8])}


def fit_line(points):
    """Least squares y = a + b x."""
    n = len(points)
    if n < 2:
        return None
    sx = sum(x for x, _ in points); sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points); sxy = sum(x * y for x, y in points)
    den = n * sxx - sx * sx
    if not den:
        return None
    b = (n * sxy - sx * sy) / den
    return (sy - b * sx) / n, b


def chunk_table(run):
    path = run / "chunks.csv"
    return read_csv(path) if path.exists() else []


def span_ms_per_token(chunks, first, last):
    """Fused GPU ms per token over chunks inside [first, last); None if a chunk there was profiled or missing."""
    total = tokens = 0.0
    for c in chunks:
        start, rows = int(c["start"]), int(c["rows"])
        if start >= first and start + rows <= last:
            if c["profiled"] == "1" or not c["gpu_ms"]:
                return None
            total += float(c["gpu_ms"]); tokens += rows
    return total / tokens if tokens and tokens == last - first else None


def chunk_fit(chunks):
    pts = [((int(c["start"]) + int(c["rows"]) / 2) / 1000, float(c["gpu_ms"]) / int(c["rows"]))
           for c in chunks if c["profiled"] == "0" and c["gpu_ms"] and int(c["rows"]) >= 1024]
    return fit_line(pts)


# ---------------------------------------------------------------- decode
def classify_decode(names):
    n = len(names)
    cats = ["other"] * n
    first = lambda pred, lo=0, hi=None: next((i for i in range(lo, n if hi is None else hi) if pred(names[i])), None)
    tree = first(lambda s: s == "dspine_select_tree")
    verify_in = first(lambda s: s == "verify_input_tokens")
    sample = first(lambda s: s.startswith("decode_sample"), verify_in or 0)
    accept = first(lambda s: s.startswith("decode_accept"), sample or 0)
    if tree is None or verify_in is None:
        return cats, False
    sample = sample if sample is not None else n
    accept = accept if accept is not None else n
    # draft region [0, verify_in)
    last_bridge = max((i for i in range(tree) if names[i] == "dspine_bridge_write"), default=None)
    first_norm = first(lambda s: "norm" in s, 0, tree)
    for i in range(0, verify_in):
        s = names[i]
        if i >= tree:
            cats[i] = "draft_tree"
        elif s.startswith("dspine_attention"):
            cats[i] = "draft_window_attention"
        elif s.startswith("dspine_bridge") or s == "dspine_project_bf16":
            cats[i] = "draft_bridge"
        elif s.startswith("dspine_tail"):
            cats[i] = "draft_refinement"
        elif s.startswith("dspine_select_top16") or s == "dspine_merge_candidates":
            cats[i] = "draft_head"
        elif last_bridge is not None and i > last_bridge:
            cats[i] = "draft_head"
        elif first_norm is not None and i < first_norm:
            cats[i] = "draft_input"
        else:
            cats[i] = "draft_backbone"
    for i in range(verify_in, sample):
        s = names[i]
        if s.startswith("turbo_rotate_queries") or (s.startswith("verify_attention") and ("_split" in s or "_reduce" in s)):
            cats[i] = "verify_attention"
        elif s.startswith("verify_attention"):
            cats[i] = "verify_attention_aux"
        elif "gdn" in s:
            cats[i] = "verify_gdn"
        elif (s.startswith("gguf_") and not s.startswith("gguf_embed")) or "linear" in s:
            cats[i] = "verify_projections"
        else:
            cats[i] = "verify_other"
    for i in range(sample, accept):
        cats[i] = "sampling"
    for i in range(accept, n):
        s = names[i]
        if s.startswith("decode_accept"):
            cats[i] = "accept"
        elif s.startswith("decode_sample"):
            cats[i] = "sampling"
        elif "restore" in s:
            cats[i] = "kv_restore"
        elif "gdn" in s:
            cats[i] = "gdn_commit"
        else:
            cats[i] = "draft_commit"
    return cats, True


def decode_profile(path):
    rows = read_csv(path)
    cycles = defaultdict(list)
    for r in rows:
        cycles[int(r["cycle"])].append(r)
    by_cat = defaultdict(float)
    unknown = defaultdict(float)
    good = 0
    for cycle, items in cycles.items():
        items.sort(key=lambda r: int(r["index"]))
        cats, ok = classify_decode([r["pipeline"] for r in items])
        if not ok:
            continue
        good += 1
        for r, c in zip(items, cats):
            by_cat[c] += float(r["gpu_us"])
            if c in ("other", "verify_other"):
                unknown[r["pipeline"]] += float(r["gpu_us"])
    if not good:
        return None
    ms = {k: by_cat.get(k, 0.0) / 1e3 / good for k in DECODE_ORDER}
    return {"cycles": good, "ms": ms, "total_ms": sum(ms.values()),
            "draft_ms": sum(ms[k] for k in DRAFT_KEYS),
            "other_names": {k: v / 1e3 / good for k, v in sorted(unknown.items(), key=lambda kv: -kv[1])[:8]}}


def decode_fused(run):
    path = run / "decode_cycles.csv"
    if not path.exists():
        return {}
    out = {}
    for r in read_csv(path):
        if r["phase"] != "fused":
            continue
        arm = "rows16" if r["arm"] in ("rows16", "rows16b") else r["arm"]
        d = out.setdefault(arm, {"cycles": 0, "tokens": 0, "gpu": 0.0, "wall": 0.0, "positions": []})
        d["cycles"] += 1; d["tokens"] += int(r["tokens"]); d["gpu"] += float(r["gpu_ms"]); d["wall"] += float(r["wall_ms"])
        d["positions"].append(int(r["position"]))
    for d in out.values():
        c = d["cycles"]
        d.update(gpu_ms=d["gpu"] / c, wall_ms=d["wall"] / c, tok_per_cycle=d["tokens"] / c,
                 position=sum(d["positions"]) // c)
        del d["positions"]
    return out


# ---------------------------------------------------------------- fake runs (dry run)
def write_fake(out):
    rnd = random.Random(7)
    def prefill_dispatches(start, rows):
        names = ["rope_build_tables", "gguf_embed_rotated_pq20"]
        for layer in range(64):
            names += ["norm_rms", "gguf_prefill_iq3_s"]
            if layer % 4 == 3:
                names += ["prefill_attention_qkv_kv2_g8", "prefill_attention_turbo_int4_store_kv2_g8",
                          "turbo_rotate_queries_256", "prefill_attention_turbo_int4_split_kv2_g8",
                          "prefill_attention_reduce_turbo_int4_kv2_g8", "prefill_attention_gate_kv2_g8"]
            else:
                names += ["prefill_gdn_gate", "prefill_gdn_prepare", "prefill_gdn_scan"]
            names += ["gguf_prefill_iq3_xxs", "norm_rms", "gguf_prefill_iq4_xs", "gguf_prefill_iq4_xs", "gguf_prefill_iq3_s"]
        if start >= 94451:
            names += ["prefill_linear_q4_sums32", "prefill_linear_q4_n128", "prefill_norm_rms_sums32"]
            for _ in range(5):
                names += ["prefill_linear_q4_n128", "dspine_context_kv"]
        rows_out = []
        for i, n in enumerate(names):
            if "split" in n:
                us = 2.0 * (start + rows / 2) / 1000 * rows / 2048 * 1e3 * 0.06
                g = (4, rows // 8, 1)
            elif "gguf_prefill" in n:
                us = 2600 * rows / 2048
                g = (64, 16, 1)
            elif "gdn" in n:
                us = 900 * rows / 2048
                g = (32, 1, 1)
            else:
                us = 60 * rows / 2048
                g = (16, 1, 1)
            rows_out.append((i, n, us * rnd.uniform(0.95, 1.05), *g))
        return rows_out

    def decode_dispatches(rows8):
        names = ["rope_build_tables", "gguf_embed_rotated_pq20", "norm_rms_staged"]
        for _ in range(5):
            names += ["decode_linear_q4_sg", "dspine_attention_prepare", "dspine_attention_split", "dspine_attention_reduce",
                      "decode_linear_q4_sg_residual", "norm_rms_staged", "decode_linear_q4_sg_gate_up",
                      "decode_linear_q4_sg_residual", "dspine_bridge_normalize", "dspine_project_bf16",
                      "dspine_bridge_codes", "dspine_project_bf16", "dspine_bridge_message", "dspine_project_bf16",
                      "dspine_bridge_write"]
        names += ["gguf_decode_sg_fused_l", "gguf_decode_sg_fused_l", "dspine_select_top16_sharded",
                  "dspine_merge_candidates", "dspine_tail_senders", "dspine_tail_gemm_codes", "dspine_tail_gate",
                  "dspine_tail_gemm_write", "dspine_tail_tables_from_product", "dspine_select_tree", "rope_build_tables",
                  "verify_input_tokens", "gguf_embed_rotated_pq20"]
        for layer in range(64):
            names += ["norm_rms_staged", "gguf_decode_sg_fused_l"]
            if layer % 4 == 3:
                names += ["verify_attention_qkv_kv2_g8", "verify_attention_turbo_int4_store_kv2_g8",
                          "turbo_rotate_queries_256", "verify_attention_turbo_int4_split_kv2_g8",
                          "verify_attention_reduce_turbo_int4_kv2_g8", "verify_attention_gate_kv2_g8"]
            else:
                names += ["verify_gdn_fused"]
            names += ["gguf_decode_sg_fused_l", "norm_rms_staged", "gguf_decode_sg_fused_l", "gguf_decode_sg_fused_l"]
        names += ["norm_rms_staged", "gguf_decode_sg_fused_l", "decode_sample_mass_sharded", "decode_sample_vocabulary_search",
                  "decode_sample_vocabulary_draw", "decode_accept_tree"]
        names += ["verify_attention_turbo_int4_restore_kv2_g8"] * 16 + ["gather_path_rows_bf16", "verify_gdn_commit"]
        names += ["decode_linear_q4_sg", "norm_rms_staged"] + ["decode_linear_q4_sg", "dspine_context_kv_commit"] * 5
        out = []
        for i, n in enumerate(names):
            base = {"gguf_decode_sg_fused_l": 220, "verify_gdn_fused": 40, "dspine_attention_split": 90,
                    "dspine_project_bf16": 45, "decode_linear_q4_sg": 70, "dspine_tail_gemm_write": 300,
                    "verify_attention_turbo_int4_split_kv2_g8": 1500}.get(n, 15)
            if rows8 and (n.startswith("dspine_attention") or n.startswith("dspine_tail") or n == "dspine_project_bf16"):
                base *= 0.55
            out.append((i, n, base * rnd.uniform(0.95, 1.05), 8, 1, 1))
        return out

    specs = [("p100k", 102643, 2048, [20480, 59392, 100352], 103000),
             ("p70k", 67695, 2048, [59392], 68000),
             ("p100k_chunk4096", 102643, 4096, [98304], None)]
    for name, tokens, chunk, profile_at, decode_at in specs:
        run = out / "runs" / name
        run.mkdir(parents=True, exist_ok=True)
        with open(run / "chunks.csv", "w") as f:
            f.write("chunk,start,rows,gpu_ms,wall_ms,profiled\n")
            pending = list(profile_at)
            for c, start in enumerate(range(0, tokens, chunk)):
                rows = min(chunk, tokens - start)
                ms = rows * (2.45 + 0.0245 * (start + rows / 2) / 1000) * (0.93 if chunk == 4096 else 1.0)
                prof = bool(pending) and start >= pending[0] and rows == chunk
                while pending and start >= pending[0]:
                    pending.pop(0)
                if prof:
                    f.write(f"{c},{start},{rows},,{ms * 1.3:.3f},1\n")
                    with open(run / f"prefill_dispatch_{start}.csv", "w") as d:
                        d.write("index,pipeline,gpu_us,groups_x,groups_y,groups_z\n")
                        for row in prefill_dispatches(start, rows):
                            d.write("%d,%s,%.3f,%d,%d,%d\n" % row)
                else:
                    f.write(f"{c},{start},{rows},{ms:.3f},{ms * 1.01:.3f},0\n")
        with open(run / "decode_cycles.csv", "w") as f:
            f.write("arm,phase,cycle,position,tokens,gpu_ms,wall_ms\n")
            if decode_at:
                cyc, pos = 0, decode_at
                for arm, warm, fused, prof in (("rows16", 8, 32, 8), ("rows8", 4, 32, 8), ("rows16b", 4, 32, 0)):
                    for phase, count in (("warm", warm), ("fused", fused), ("profiled", prof)):
                        for _ in range(count):
                            ms = (106.0 if arm == "rows8" else 109.0) + rnd.uniform(-1, 1)
                            tok = rnd.choice([3, 4, 5, 6, 7])
                            f.write(f"{arm},{phase},{cyc},{pos},{tok},{ms:.3f},{ms + 0.7:.3f}\n")
                            cyc += 1; pos += tok
                for arm in ("rows16", "rows8"):
                    with open(run / f"decode_dispatch_{arm}.csv", "w") as d:
                        d.write("cycle,index,pipeline,gpu_us,groups_x,groups_y,groups_z\n")
                        for cycle in range(8):
                            for row in decode_dispatches(arm == "rows8"):
                                d.write("%d,%d,%s,%.3f,%d,%d,%d\n" % ((cycle,) + row))
        meta = {"device": "FAKE (PROFILE_DRY_RUN)", "prompt_tokens": tokens, "prefill_chunk_rows": chunk,
                "profiled_chunk_starts": profile_at, "decode_ended_early": "",
                "unverified_tail_cycles": {"rows16": 48, "rows8": 44, "rows16b": 36} if decode_at else {}}
        (run / "meta.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- summary
def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    out = Path(sys.argv[1])
    if "--fake" in sys.argv[2:]:
        write_fake(out)
    runs = out / "runs"
    steps = {}
    log = out / "logs" / "steps.log"
    if log.exists():
        for line in log.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "step":
                steps[parts[2].rstrip(":")] = " ".join(parts[3:])
    fake = (runs / "p100k" / "meta.json").exists() and "FAKE" in (runs / "p100k" / "meta.json").read_text()
    lines = ["# longctx_profile summary: prefill and decode split at long context (dev-longctx defaults)", ""]
    if fake:
        lines += ["**DRY RUN: synthetic numbers (PROFILE_DRY_RUN=1); nothing below is a measurement.**", ""]
    lines += ["Engine: splash/dev-prefill (= dev-longctx + profiling hooks, off unless set), DSpine tree + copy, shortlist, "
              "turbo-int4 KV, 8 verify rows from 64K, T=0.6 seed 1.", "",
              "| step | result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in steps.items()]
    result = {"dry_run": fake, "steps": steps, "prefill": {}, "decode": {}}

    # ---- A. prefill
    lines += ["", "## A. Prefill: one 2048-row chunk at ~20K / ~60K / ~100K (item 2; the p100k_chunk4096 column is "
              "one 4096-row chunk)", ""]
    prof = {}
    for run_name in ("p100k", "p70k", "p100k_chunk4096"):
        run = runs / run_name
        if not run.exists():
            continue
        meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
        for path in sorted(run.glob("prefill_dispatch_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            start = int(path.stem.rsplit("_", 1)[1])
            p = prefill_profile(path, start)
            p["run"] = run_name
            p["rows"] = int(meta.get("prefill_chunk_rows") or 2048)
            prof[(run_name, start)] = p
    if not prof:
        lines.append("No prefill profile (the GPU runs did not produce one).")
    else:
        cols = sorted(prof)
        lines += ["| part | " + " | ".join(f"chunk at {s} ({r}) ms | %" for r, s in cols) + " |",
                  "|---|" + "---|---|" * len(cols)]
        for k in PREFILL_ORDER:
            cells = []
            for key in cols:
                p = prof[key]
                cells += [f"{p['ms'][k]:.1f}", pct(p["ms"][k] / p["total_ms"] if p["total_ms"] else None)]
            lines.append(f"| {PREFILL_LABEL[k]} | " + " | ".join(cells) + " |")
        lines.append("| **sum of dispatches** | " + " | ".join(f"**{prof[k]['total_ms']:.1f}** | 100%" for k in cols) + " |")
        # fused context: neighbouring chunks' fused time
        lines += ["", "Fused GPU time of the neighbouring chunks (the profiled chunk itself ran as separate dispatches):"]
        for (r, s) in cols:
            chunks = chunk_table(runs / r)
            near = [float(c["gpu_ms"]) for c in chunks if c["profiled"] == "0" and c["gpu_ms"] and
                    abs(int(c["start"]) - s) <= prof[(r, s)]["rows"] and int(c["rows"]) == prof[(r, s)]["rows"]]
            if near:
                fused = sum(near) / len(near)
                rows = prof[(r, s)]["rows"]
                lines.append(f"- {r} chunk at {s}: neighbours fused {fused:.1f} ms per {rows} rows "
                             f"({fused / rows:.3f} ms/token); this chunk's sum of dispatches {prof[(r, s)]['total_ms']:.1f} ms")
                prof[(r, s)]["fused_neighbour_ms"] = fused
        # attention per layer and achieved rates
        lines += ["", "### Attention per full-attention layer (ms; kernel = rotate + split + reduce)", "",
                  "| chunk | " + " | ".join(f"L{i}" for i in range(ATTENTION_LAYERS)) + " | mean | TFLOP/s | KV GB/s unique | KV GB/s requested |",
                  "|---|" + "---|" * (ATTENTION_LAYERS + 4)]
        for key in cols:
            p = prof[key]
            per = p["per_layer_ms"]
            rows = p["rows"]
            tiles = p["split_tiles"][0] if p["split_tiles"] else rows // PREFILL_TILE_ROWS
            flop, unique, requested = attention_math(key[1], rows, tiles)
            mean = sum(per) / len(per) if per else 0.0
            rate = flop / (mean / 1e3) / 1e12 if mean else 0.0
            p.update(attention_tflops=rate, kv_gbs_unique=unique / (mean / 1e3) / 1e9 if mean else 0.0,
                     kv_gbs_requested=requested / (mean / 1e3) / 1e9 if mean else 0.0, tiles=tiles)
            cells = [f"{x:.2f}" for x in per] + [""] * max(0, ATTENTION_LAYERS - len(per))
            lines.append(f"| {key[1]} ({key[0]}) | " + " | ".join(cells[:ATTENTION_LAYERS]) +
                         f" | {mean:.2f} | {rate:.1f} | {p['kv_gbs_unique']:.0f} | {p['kv_gbs_requested']:.0f} |")
        lines += ["", f"Rates are derived: FLOP = 4 x {QUERY_HEADS} heads x {HEAD_DIM} x (causal keys summed over the chunk's "
                      f"rows) per layer; KV bytes = {KV_HEADS} KV heads x {KV_BYTES_PER_TOKEN_HEAD} B per token (turbo-int4), "
                      "\"unique\" read once, \"requested\" read once per 8-row query tile (the split grid's y = "
                      "tiles; caches serve most of it). Reference: ~25 TFLOP/s FP16 matrix peak, 307 GB/s DRAM (ANALYSIS.md 3.2)."]
        others = {}
        for p in prof.values():
            for n, us in p["other_names"].items():
                others[n] = others.get(n, 0.0) + us
        if others:
            lines += ["", "Largest pipelines in \"other\" (check the classification): " +
                      ", ".join(f"{n} {us / 1e3:.1f} ms" for n, us in sorted(others.items(), key=lambda kv: -kv[1])[:6])]
        result["prefill"]["profiles"] = {f"{r}@{s}": prof[(r, s)] for r, s in cols}

    # chunk curve and chunk 4096
    lines += ["", "### Chunk 4096 vs 2048 (fused GPU ms per token, same token span)", ""]
    a, c = chunk_table(runs / "p100k"), chunk_table(runs / "p100k_chunk4096")
    spans = [("~62K", 61440, 65536), ("~96K", 94208, 98304)]
    chunk_result = {}
    if a and c:
        lines += ["| span | 2048-row chunks | 4096-row chunks | 4096 vs 2048 |", "|---|---|---|---|"]
        for label, lo, hi in spans:
            x, y = span_ms_per_token(a, lo, hi), span_ms_per_token(c, lo, hi)
            ratio = (y / x - 1) if x and y else None
            chunk_result[label] = {"chunk2048": x, "chunk4096": y, "change": ratio}
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
            lines.append(f"| {label} [{lo}, {hi}) | {fmt(x)} | {fmt(y)} | {pct(ratio) if ratio is not None else 'n/a'} |")
    else:
        lines.append("Missing run(s): " + ", ".join(n for n, t in (("p100k", a), ("p100k_chunk4096", c)) if not t))
    for label, table in (("2048", a), ("4096", c)):
        f = chunk_fit(table) if table else None
        if f:
            chunk_result[f"fit_{label}"] = {"ms_per_token_at_0": f[0], "ms_per_token_per_1k": f[1]}
            lines.append(f"- fit, {label}-row chunks: {f[0]:.3f} + {f[1]:.5f} x ctxK ms/token "
                         f"(60K: {f[0] + 60 * f[1]:.3f}, 100K: {f[0] + 100 * f[1]:.3f})")
    result["prefill"]["chunk4096"] = chunk_result

    # ---- B. decode
    lines += ["", "## B. Decode cycle at ~70K and ~100K (item 6)", ""]
    decode = {}
    for run_name in ("p70k", "p100k"):
        run = runs / run_name
        fused = decode_fused(run)
        d16 = decode_profile(run / "decode_dispatch_rows16.csv") if (run / "decode_dispatch_rows16.csv").exists() else None
        d8 = decode_profile(run / "decode_dispatch_rows8.csv") if (run / "decode_dispatch_rows8.csv").exists() else None
        if fused or d16:
            decode[run_name] = {"fused": fused, "rows16": d16, "rows8": d8}
    if not decode:
        lines.append("No decode profile (the GPU runs did not produce one).")
    else:
        names = list(decode)
        head = "| part | " + " | ".join(f"{n} ms | % | ms (fused)" for n in names) + " |"
        lines += [head, "|---|" + "---|---|---|" * len(names)]
        for k in DECODE_ORDER:
            cells = []
            for n in names:
                d = decode[n]["rows16"]; f = decode[n]["fused"].get("rows16")
                if d and d["total_ms"]:
                    share = d["ms"][k] / d["total_ms"]
                    cells += [f"{d['ms'][k]:.2f}", pct(share), f"{share * f['gpu_ms']:.2f}" if f else "n/a"]
                else:
                    cells += ["n/a"] * 3
            lines.append(f"| {DECODE_LABEL[k]} | " + " | ".join(cells) + " |")
        cells = []
        for n in names:
            d = decode[n]["rows16"]; f = decode[n]["fused"].get("rows16")
            cells += [f"**{d['total_ms']:.1f}**" if d else "n/a", "100%", f"**{f['gpu_ms']:.1f}**" if f else "n/a"]
        lines.append("| **cycle** | " + " | ".join(cells) + " |")
        for n in names:
            f = decode[n]["fused"].get("rows16"); d = decode[n]["rows16"]
            if f:
                lines.append(f"- {n}: fused {f['gpu_ms']:.1f} ms GPU / {f['wall_ms']:.1f} ms wall per cycle at "
                             f"~{f['position']} tokens, {f['tok_per_cycle']:.2f} tok/cycle ({f['cycles']} cycles); draft "
                             f"{d['draft_ms']:.2f} ms of {d['total_ms']:.1f} ms in separate dispatches" if d else "")
        # rows 8-15
        lines += ["", "### Draft rows 8-15 (dropped by the 8-row verify above 64K)", "",
                  "Measured by running the DSpine block with 8 rows (SPLASH_PROFILE_SPLIT=1 SPLASH_PROFILE_DRAFT_ROWS=8, "
                  "dev-prefill hook) right after the 16-row arm, and 16 rows again after it.", "",
                  "| run | draft 16 rows ms | draft 8 rows ms | rows 8-15 ms | share of draft | fused cycle 16 / 8 ms | tok/cycle 16 / 8 |",
                  "|---|---|---|---|---|---|---|"]
        shares = []
        for n in names:
            d16, d8 = decode[n]["rows16"], decode[n]["rows8"]
            f16, f8 = decode[n]["fused"].get("rows16"), decode[n]["fused"].get("rows8")
            if d16 and d8:
                delta = d16["draft_ms"] - d8["draft_ms"]
                share = delta / d16["draft_ms"] if d16["draft_ms"] else None
                shares.append((n, share, delta))
                decode[n]["rows8_15"] = {"ms": delta, "share_of_draft": share,
                                         "per_part_ms": {k: d16["ms"][k] - d8["ms"][k] for k in DRAFT_KEYS}}
                lines.append(f"| {n} | {d16['draft_ms']:.2f} | {d8['draft_ms']:.2f} | {delta:.2f} | {pct(share)} | "
                             + (f"{f16['gpu_ms']:.1f} / {f8['gpu_ms']:.1f}" if f16 and f8 else "n/a") + " | "
                             + (f"{f16['tok_per_cycle']:.2f} / {f8['tok_per_cycle']:.2f}" if f16 and f8 else "n/a") + " |")
            else:
                lines.append(f"| {n} | " + (f"{d16['draft_ms']:.2f}" if d16 else "n/a") + " | n/a | n/a | n/a | n/a | n/a |")
        if any("rows8_15" in decode[n] for n in names):
            lines += ["", "Rows 8-15 per draft part (ms per cycle, 16 rows minus 8 rows): "]
            for n in names:
                if "rows8_15" in decode[n]:
                    parts = decode[n]["rows8_15"]["per_part_ms"]
                    lines.append(f"- {n}: " + ", ".join(f"{DECODE_LABEL[k].split(': ', 1)[1]} {v:+.2f}" for k, v in parts.items()))
        others = {}
        for n in names:
            d = decode[n]["rows16"]
            for name, ms in (d["other_names"] if d else {}).items():
                others[name] = others.get(name, 0.0) + ms
        if others:
            lines += ["", "Largest pipelines in \"other\"/\"verify: other\" (ms per cycle; check the classification): " +
                      ", ".join(f"{k} {v:.2f}" for k, v in sorted(others.items(), key=lambda kv: -kv[1])[:6])]
        for n in names:
            meta = runs / n / "meta.json"
            if meta.exists():
                m = json.loads(meta.read_text())
                if m.get("decode_ended_early"):
                    lines.append(f"- {n}: the answer ended early in {m['decode_ended_early']} (fewer cycles)")
                tails = m.get("unverified_tail_cycles")
                if tails:
                    lines.append(f"- {n}: cycles with the 8-row verify cut, per arm: {tails}")
        result["decode"] = decode

    # ---- verdicts
    lines += ["", "## Verdicts", ""]
    p100 = [p for (r, s), p in prof.items() if r == "p100k" and s >= 90000]
    if p100:
        p = p100[-1]
        att = (p["ms"]["attention"] + p["ms"]["attention_aux"]) / p["total_ms"]
        core = p["ms"]["attention"] / p["total_ms"]
        verdict2 = "item 2: proceed" if att >= ATTENTION_GATE else "item 2: skip"
        lines.append(f"- **{verdict2}** (attention {pct(att)} of a 2048-row prefill chunk at {p['start']}: kernel {pct(core)} "
                     f"+ prep/store/gate; gate {ATTENTION_GATE:.0%}; kernel at {p.get('attention_tflops', 0):.1f} TFLOP/s)")
        result["verdict_item2"] = {"verdict": verdict2, "attention_share": att, "kernel_share": core}
    else:
        lines.append("- **item 2: no verdict** (no profiled prefill chunk at ~100K)")
        result["verdict_item2"] = {"verdict": "none"}
    if shares:
        txt = "; ".join(f"{n} {pct(s)} of the draft ({d:.2f} ms/cycle)" for n, s, d in shares)
        best = max(s for _, s, _ in shares if s is not None) if any(s is not None for _, s, _ in shares) else None
        meets = best is not None and best >= DRAFT_GATE
        verdict6 = (f"item 6a: rows 8-15 = {txt}; " +
                    ("meets" if meets else "below") + f" the -{DRAFT_GATE:.0%} draft gate")
        lines.append(f"- **{verdict6}**")
        result["verdict_item6"] = {"text": verdict6, "shares": {n: s for n, s, _ in shares}, "meets_gate": meets}
    else:
        lines.append("- **item 6a: no measurement** (the 8-row draft arm did not run)")
        result["verdict_item6"] = {"text": "none"}
    try:
        lines += ["", "## Defender CPU", "", "```", (out / "logs" / "defender.log").read_text().strip(), "```"]
    except OSError:
        pass
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(result, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

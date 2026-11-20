#!/usr/bin/env python3
"""Run Tencent's DSpine MLX reference on inputs captured from Splash.

Library + CLI. The draft is Tencent's own `dspine_mlx` modules (model.py layers, bridge.py code bridge, propose.py
RefinementTail); this file only supplies what the reference takes from an mlx-lm target, read from Splash's GGUF:

- the token embedding (token_embd.weight, Q2_K in UD-IQ3_XXS), dequantized per row (fp32 -> bf16, as Splash's
  embedding kernel writes bf16 rows);
- the LM head (output.weight, Q4_K), re-packed losslessly as MLX affine 4-bit, group 32, fp32 scales/biases
  (Q4_K sub-block value d*sc*q - dmin*m == scale*q + bias), wrapped in the reference's HeadAccess;
- the captured target hidden rows (SPLASH_CAPTURE_DUMP of layers 1,16,31,46,61).

Draft forward (`DSpine.block`): identical to DSpineDraftModel.__call__ with a context cache, except that the context
K/V of every position is computed once per request (`DSpine.context`: hidden_norm(fc(h)), k_norm(k_proj), RoPE at the
row's absolute position, v_proj; what append_context would store) and each cycle attends to the window
[max(0, p - window), p) of it. Block row i has RoPE position p + i, anchor at p, as in Splash and in the reference's
cached path (queries at cache.offset + S). Mask: causal inside the block, all window rows visible (mask="causal",
bottom-right aligned, as the reference).

The checkpoint's config says sliding_attention / sliding_window 8192, which the MLX port's validator refuses ("DSpine
drafts use no sliding window"); we load it with the window fields cleared and apply the window ourselves.
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gguf_tensors import GgufFile  # noqa: E402

DEFAULT_REFERENCE = Path("/private/tmp/claude-501/-Users-abdul-rakib-agentMania-dsh-qwen-jev/"
                         "2995df64-4b21-4d90-8cfb-0d7faba3be1e/scratchpad/dspine_src/dspine_mlx")
DEFAULT_DRAFT = Path.home() / (".cache/huggingface/hub/models--tencent--Qwen3.8-27B-DSpine-b16/snapshots/"
                               "ea909c0f5fab8865f9e14600334f55f7ce47308f")
DEFAULT_GGUF = (HERE.parents[1] / "splash-src/install/models/.selections/"
                "fb0aa6e23bdecc7d24023adea3bcae72062d1c103ab1578902b00a2ebb3f500f/target/Qwen3.8-27B-UD-IQ3_XXS.gguf")
# Splash's draft-head shortlist (SPLASH_DRAFT_HEAD_ROWS=98304): token ids [0, 98304) and [248044, vocab).
SHORTLIST_SPLIT, SHORTLIST_TAIL = 98304, 248044


# ---------------------------------------------------------------- capture dump

class CaptureFile:
    """SPLASH_CAPTURE_DUMP records, indexed without loading: `ids` (request ids in file order) and `rows(id)`
    (uint16 = bf16 bits [positions, n*hidden], rows ordered by position from 0), `layers(id)`."""

    def __init__(self, path):
        import mmap
        self.file = open(path, "rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.index = {}
        off, size = 0, len(self.map)
        while off < size:
            assert self.map[off:off + 4] == b"SCAP", f"bad record at {off}"
            rid, pos = struct.unpack_from("<QQ", self.map, off + 4)
            rows, nl, hidden = struct.unpack_from("<III", self.map, off + 20)
            layers = struct.unpack_from(f"<{nl}I", self.map, off + 32)
            off += 32 + 4 * nl
            self.index.setdefault(rid, []).append((pos, layers, hidden, off, rows))
            off += 2 * rows * nl * hidden
        self.ids = list(self.index)

    def layers(self, rid):
        return list(self.index[rid][0][1])

    def rows(self, rid):
        parts = sorted(self.index[rid])
        expect = 0
        arrays = []
        for pos, layers, hidden, off, rows in parts:
            assert pos == expect, f"request {rid}: gap at {pos} (expected {expect})"
            expect += rows
            arrays.append(np.frombuffer(self.map, dtype="<u2", count=rows * len(layers) * hidden,
                                        offset=off).reshape(rows, len(layers) * hidden))
        return np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]


def read_capture(path):
    return CaptureFile(path)


def select_layers(layers, rows, wanted, hidden=5120):
    idx = [layers.index(l) for l in wanted]
    return np.concatenate([rows[:, i * hidden:(i + 1) * hidden] for i in idx], axis=1)


def bf16_bits_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------- the reference

class DSpine:
    def __init__(self, reference=DEFAULT_REFERENCE, draft=DEFAULT_DRAFT, gguf=DEFAULT_GGUF, quantize_bits=0):
        sys.path.insert(0, str(Path(reference)))
        import mlx.core as mx
        import mlx.nn as nn
        from dspine_mlx import config as dcfg
        from dspine_mlx import model as dmodel
        from dspine_mlx.propose import RefinementTail, topk_ids
        from dspine_mlx.target import HeadAccess
        from mlx.utils import tree_flatten
        self.mx, self.nn, self.topk_ids = mx, nn, topk_ids
        raw = json.loads((Path(draft) / "config.json").read_text())
        self.sliding_window = int(raw.get("sliding_window") or 0)
        for key in ("layer_types", "sliding_window"):
            raw.pop(key, None)
        raw["use_sliding_window"] = False
        cfg = dcfg.DraftConfig.from_dict(raw, Path(draft) / "config.json")
        model = dmodel.DSpineDraftModel(cfg)
        weights = dmodel._load_weights(Path(draft))
        cleaned = {}
        for name, tensor in weights.items():
            short = name.removeprefix("model.")
            if short in ("draft_block_causal_version", "last_write_refine_version"):
                assert int(tensor.item()) == 1, short
                continue
            cleaned[short] = tensor
        expected = dict(tree_flatten(model.parameters()))
        assert set(expected) == set(cleaned), (set(expected) ^ set(cleaned))
        for name, dest in expected.items():
            if cleaned[name].dtype != dest.dtype and dest.dtype in (mx.int32, mx.int64):
                cleaned[name] = cleaned[name].astype(dest.dtype)
        model.load_weights(list(cleaned.items()), strict=True)
        model.selected_prefix_bridge.prepare()
        if quantize_bits:
            dmodel.quantize_draft(model, bits=quantize_bits, group_size=64)
        self.model, self.cfg = model, cfg
        self.gguf = GgufFile(gguf)
        self._embed_cache = {}
        model._embed_tokens = self._embed
        self.head = HeadAccess(self._q4k_head())
        self.tail = RefinementTail(model, self.head, None, None)
        mx.eval(model.parameters())

    # embedding: the target's Q2_K rows, dequantized on demand
    def _embed(self, ids):
        mx = self.mx
        flat = [int(t) for t in np.array(ids).reshape(-1)]
        missing = [t for t in set(flat) if t not in self._embed_cache]
        for t in missing:
            row = self.gguf.rows("token_embd.weight", t, t + 1)[0]
            self._embed_cache[t] = mx.array(row).astype(mx.bfloat16)
        out = mx.stack([self._embed_cache[t] for t in flat])
        return out.reshape(*ids.shape, -1)

    def embedding_rows(self, ids):
        return np.stack([np.array(self._embed(self.mx.array([[t]], dtype=self.mx.int32)).astype(self.mx.float32))[0, 0]
                         for t in ids])

    def _q4k_head(self, chunk=8192):
        """output.weight Q4_K -> nn.QuantizedLinear (affine, 4 bits, group 32, fp32 scales and biases), exact."""
        mx, nn = self.mx, self.nn
        dims, kind, offset = self.gguf.info("output.weight")
        assert kind == "Q4_K", kind
        cols, vocab = dims
        g = self.gguf
        row_bytes = cols // 256 * 144
        packed, scales, biases = [], [], []
        from gguf_tensors import _k_scales
        for start in range(0, vocab, chunk):
            end = min(start + chunk, vocab)
            n = end - start
            raw = np.frombuffer(g.map, dtype=np.uint8, count=n * row_bytes,
                                offset=g.data_start + offset + start * row_bytes).reshape(n, cols // 256, 144)
            d = raw[:, :, 0:2].copy().view("<f2").astype(np.float32)
            dmin = raw[:, :, 2:4].copy().view("<f2").astype(np.float32)
            sc, mn = _k_scales(raw[:, :, 4:16])                    # [n,nb,8]
            qs = raw[:, :, 16:144].reshape(n, cols // 256, 4, 32)
            q = np.empty((n, cols // 256, 8, 32), np.uint8)
            q[:, :, 0::2] = qs & 0xF
            q[:, :, 1::2] = qs >> 4
            q = q.reshape(n, cols)
            nib = q.reshape(n, cols // 8, 8).astype(np.uint32)
            word = np.zeros((n, cols // 8), np.uint32)
            for k in range(8):
                word |= nib[:, :, k] << (4 * k)
            packed.append(word)
            scales.append((d * sc).reshape(n, cols // 32))
            biases.append((-dmin * mn).reshape(n, cols // 32))
        head = nn.QuantizedLinear(cols, vocab, bias=False, group_size=32, bits=4)
        head.weight = mx.array(np.concatenate(packed))
        head.scales = mx.array(np.concatenate(scales))
        head.biases = mx.array(np.concatenate(biases))
        head.freeze()
        mx.eval(head.weight, head.scales, head.biases)
        return head

    def head_logits(self, rows):
        """Full-vocabulary fp32 logits of bf16 rows [n, 5120]."""
        return self.head.matmul(rows.astype(self.mx.float32))

    # context: per-layer K/V of every captured position (what append_context stores)
    def context(self, hidden_u16, chunk=2048):
        mx = self.mx
        m = self.model
        n = hidden_u16.shape[0]
        keys = [[] for _ in m.layers]
        values = [[] for _ in m.layers]
        projected = []
        for start in range(0, n, chunk):
            h = mx.array(hidden_u16[start:start + chunk]).view(mx.bfloat16)[None]
            ctx = m.hidden_norm(m.fc(h))
            projected.append(ctx[0])
            for i, layer in enumerate(m.layers):
                k, v = layer.self_attn._context_kv(ctx, m.rope, start)
                keys[i].append(k)
                values[i].append(v)
            mx.eval(projected[-1], *[k[-1] for k in keys], *[v[-1] for v in values])
        self.ctx_projected = mx.concatenate(projected, axis=0)
        self.ctx_keys = [mx.concatenate(k, axis=2) for k in keys]
        self.ctx_values = [mx.concatenate(v, axis=2) for v in values]
        mx.eval(self.ctx_projected, *self.ctx_keys, *self.ctx_values)

    def _attention(self, attn, x, layer, p, window):
        mx = self.mx
        B, L, _ = x.shape
        q = attn.q_norm(attn.q_proj(x).reshape(B, L, attn.n_heads, attn.head_dim)).transpose(0, 2, 1, 3)
        k = attn.k_norm(attn.k_proj(x).reshape(B, L, attn.n_kv_heads, attn.head_dim)).transpose(0, 2, 1, 3)
        v = attn.v_proj(x).reshape(B, L, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        q = self.model.rope(q, offset=p)
        k = self.model.rope(k, offset=p)
        lo = max(0, p - window)
        keys = mx.concatenate([self.ctx_keys[layer][:, :, lo:p, :], k], axis=2)
        values = mx.concatenate([self.ctx_values[layer][:, :, lo:p, :], v], axis=2)
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=attn.scale, mask="causal")
        return attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

    def block(self, anchor, p, window, rows=16, trace=None):
        """Draft forward of the block [anchor, mask x (rows-1)] at positions p.. against context rows [p-window, p).
        Returns (final normed block hidden [rows,5120] bf16, stash dict of the last write)."""
        mx = self.mx
        m = self.model
        ids = mx.array([[int(anchor)] + [m.mask_token_id] * (rows - 1)], dtype=mx.int32)
        x = m._embed_tokens(ids)
        residual = x
        x_in = m.layers[0].input_layernorm(x)
        last = len(m.layers) - 1
        stash_out = None
        for i, layer in enumerate(m.layers):
            attn_out = self._attention(layer.self_attn, x_in, i, p, window)
            h = residual + attn_out
            out = h + layer.mlp(layer.post_attention_layernorm(h))
            next_norm = m.layers[i + 1].input_layernorm if i < last else m.norm
            stash = {} if i == last else None
            x_in, residual = m.selected_prefix_bridge(out, None, ids, i, next_norm, stash)
            if trace is not None:
                trace.append(x_in[0])
            if stash is not None:
                stash_out = stash
        return x_in[0], stash_out

    def candidates(self, rows, shortlist):
        mx = self.mx
        logits = self.head_logits(rows)
        if shortlist:
            vocab = logits.shape[-1]
            ids = mx.arange(vocab)
            keep = (ids < SHORTLIST_SPLIT) | (ids >= SHORTLIST_TAIL)
            logits = mx.where(keep[None], logits, -mx.inf)
        return self.topk_ids(logits, self.tail.k), logits

    def propose(self, anchor, p, window, rows=16, shortlist=True, trace=None):
        """(candidates [rows-1,16] int32 ascending ids, table [rows-1,16,16] fp32, normed rows, stash, pass-1 logits)
        RefinementTail.sampled_table with the candidate list optionally restricted to Splash's head shortlist."""
        mx = self.mx
        out, stash = self.block(anchor, p, window, rows, trace)
        proposal_rows = out[1:]
        cand, logits = self.candidates(proposal_rows, shortlist)
        tail = self.tail
        original = tail.candidates
        tail.candidates = lambda r: cand
        try:
            table, cand2 = tail.sampled_table(proposal_rows, stash, int(anchor))
        finally:
            tail.candidates = original
        mx.eval(table, cand2, out)
        return np.array(cand2), np.array(table), out, stash, logits

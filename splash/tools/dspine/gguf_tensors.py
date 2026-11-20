"""Minimal GGUF v3 tensor reader with numpy dequantization for the formats the DSpine reference needs.

Only the target's token embedding (token_embd.weight) and output head (output.weight) are read. Supported
dequantizers: F32, F16, BF16, Q8_0, Q2_K, Q4_K, Q5_K, Q6_K (ggml reference layouts, ggml-quants.c).
"""
from __future__ import annotations

import mmap
import struct
from pathlib import Path

import numpy as np

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
              24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}
# (elements per block, bytes per block)
BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34), "Q2_K": (256, 84), "Q4_K": (256, 144),
         "Q5_K": (256, 176), "Q6_K": (256, 210)}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class GgufFile:
    def __init__(self, path):
        self.path = Path(path)
        self.file = open(self.path, "rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        assert self._read(4) == b"GGUF"
        version = self._scalar("I")
        assert version == 3, version
        tensor_count, key_count = self._scalar("Q"), self._scalar("Q")
        self.metadata = {}
        for _ in range(key_count):
            key = self._string()
            self.metadata[key] = self._value(self._scalar("I"))
        self.tensors = {}
        for _ in range(tensor_count):
            name = self._string()
            ndim = self._scalar("I")
            dims = [self._scalar("Q") for _ in range(ndim)]
            kind = self._scalar("I")
            offset = self._scalar("Q")
            self.tensors[name] = (dims, GGML_TYPES.get(kind, f"type{kind}"), offset)
        alignment = int(self.metadata.get("general.alignment", 32))
        self.data_start = (self.pos + alignment - 1) // alignment * alignment

    def _read(self, n):
        out = self.map[self.pos:self.pos + n]
        self.pos += n
        return out

    def _scalar(self, fmt):
        return struct.unpack("<" + fmt, self._read(struct.calcsize("<" + fmt)))[0]

    def _string(self):
        return self._read(self._scalar("Q")).decode("utf-8", "replace")

    def _value(self, kind):
        if kind == 8:
            return self._string()
        if kind == 9:
            element, count = self._scalar("I"), self._scalar("Q")
            if element in SCALARS and element != 7:
                fmt = SCALARS[element]
                size = struct.calcsize("<" + fmt)
                raw = self._read(size * count)
                return np.frombuffer(raw, dtype=np.dtype("<" + fmt))
            return [self._value(element) for _ in range(count)]
        return self._scalar(SCALARS[kind])

    def info(self, name):
        return self.tensors[name]

    def rows(self, name, start, end):
        """Rows [start, end) of a 2-D tensor (ggml dims [cols, rows]) dequantized to float32 [end-start, cols]."""
        dims, kind, offset = self.tensors[name]
        cols, nrows = dims[0], dims[1]
        assert 0 <= start <= end <= nrows
        per_block, block_bytes = BLOCK[kind]
        row_bytes = cols // per_block * block_bytes
        base = self.data_start + offset + start * row_bytes
        raw = np.frombuffer(self.map, dtype=np.uint8, count=(end - start) * row_bytes, offset=base)
        raw = raw.reshape(end - start, row_bytes)
        return DEQUANT[kind](raw, cols)


def _f16(raw, cols):
    return raw.view("<f2").astype(np.float32).reshape(raw.shape[0], cols)


def _bf16(raw, cols):
    u = raw.view("<u2").astype(np.uint32) << 16
    return u.view(np.float32).reshape(raw.shape[0], cols)


def _f32(raw, cols):
    return raw.view("<f4").reshape(raw.shape[0], cols).copy()


def _q8_0(raw, cols):
    n = raw.shape[0]
    b = raw.reshape(n, cols // 32, 34)
    d = b[:, :, :2].copy().view("<f2").astype(np.float32)  # [n, nb, 1]
    q = b[:, :, 2:].view(np.int8).astype(np.float32)
    return (d * q).reshape(n, cols)


def _k_scales(scales):
    """get_scale_min_k4 for the 12 scale bytes of Q4_K / Q5_K: (scale[8], min[8]) as uint8."""
    s = scales.astype(np.uint8)
    sc = np.empty(s.shape[:-1] + (8,), np.uint8)
    mn = np.empty(s.shape[:-1] + (8,), np.uint8)
    for j in range(8):
        if j < 4:
            sc[..., j] = s[..., j] & 63
            mn[..., j] = s[..., j + 4] & 63
        else:
            sc[..., j] = (s[..., j + 4] & 0xF) | ((s[..., j - 4] >> 6) << 4)
            mn[..., j] = (s[..., j + 4] >> 4) | ((s[..., j] >> 6) << 4)
    return sc.astype(np.float32), mn.astype(np.float32)


def _q2_k(raw, cols):
    """block_q2_K: scales[16] (low nibble scale, high nibble min), qs[64], d, dmin (dequantize_row_q2_K)."""
    n = raw.shape[0]
    nb = cols // 256
    b = raw.reshape(n, nb, 84)
    scales = b[:, :, 0:16]
    qs = b[:, :, 16:80].reshape(n, nb, 2, 32)
    d = b[:, :, 80:82].copy().view("<f2").astype(np.float32)        # [n,nb,1]
    dmin = b[:, :, 82:84].copy().view("<f2").astype(np.float32)
    out = np.empty((n, nb, 2, 4, 2, 16), np.float32)                 # half, shift, (first/second 16), l
    for half in range(2):
        q = qs[:, :, half]
        for j in range(4):
            for k in range(2):
                sc = scales[:, :, half * 8 + 2 * j + k].astype(np.float32)[..., None]
                dl = d * (np.floor_divide(sc, 1) % 16)
                ml = dmin * np.floor_divide(sc, 16)
                vals = ((q[:, :, 16 * k:16 * (k + 1)] >> (2 * j)) & 3).astype(np.float32)
                out[:, :, half, j, k] = dl * vals - ml
    return out.reshape(n, cols)


def _q4_k(raw, cols):
    n = raw.shape[0]
    nb = cols // 256
    b = raw.reshape(n, nb, 144)
    d = b[:, :, 0:2].copy().view("<f2").astype(np.float32)        # [n,nb,1]
    dmin = b[:, :, 2:4].copy().view("<f2").astype(np.float32)
    sc, mn = _k_scales(b[:, :, 4:16])                             # [n,nb,8]
    qs = b[:, :, 16:144].reshape(n, nb, 4, 32)                    # 4 chunks of 32 bytes -> 64 values each
    lo = (qs & 0xF).astype(np.float32)                            # sub-block 2c
    hi = (qs >> 4).astype(np.float32)                             # sub-block 2c+1
    out = np.empty((n, nb, 8, 32), np.float32)
    for c in range(4):
        out[:, :, 2 * c] = d * sc[:, :, 2 * c:2 * c + 1] * lo[:, :, c] - dmin * mn[:, :, 2 * c:2 * c + 1]
        out[:, :, 2 * c + 1] = d * sc[:, :, 2 * c + 1:2 * c + 2] * hi[:, :, c] - dmin * mn[:, :, 2 * c + 1:2 * c + 2]
    return out.reshape(n, cols)


def _q5_k(raw, cols):
    n = raw.shape[0]
    nb = cols // 256
    b = raw.reshape(n, nb, 176)
    d = b[:, :, 0:2].copy().view("<f2").astype(np.float32)
    dmin = b[:, :, 2:4].copy().view("<f2").astype(np.float32)
    sc, mn = _k_scales(b[:, :, 4:16])
    qh = b[:, :, 16:48]                                            # [n,nb,32]
    qs = b[:, :, 48:176].reshape(n, nb, 4, 32)
    out = np.empty((n, nb, 8, 32), np.float32)
    for c in range(4):
        lo = (qs[:, :, c] & 0xF) + (((qh >> (2 * c)) & 1) << 4)
        hi = (qs[:, :, c] >> 4) + (((qh >> (2 * c + 1)) & 1) << 4)
        out[:, :, 2 * c] = d * sc[:, :, 2 * c:2 * c + 1] * lo - dmin * mn[:, :, 2 * c:2 * c + 1]
        out[:, :, 2 * c + 1] = d * sc[:, :, 2 * c + 1:2 * c + 2] * hi - dmin * mn[:, :, 2 * c + 1:2 * c + 2]
    return out.reshape(n, cols)


def _q6_k(raw, cols):
    n = raw.shape[0]
    nb = cols // 256
    b = raw.reshape(n, nb, 210)
    ql = b[:, :, 0:128].reshape(n, nb, 2, 64)
    qh = b[:, :, 128:192].reshape(n, nb, 2, 32)
    sc = b[:, :, 192:208].view(np.int8).astype(np.float32)        # [n,nb,16]
    d = b[:, :, 208:210].copy().view("<f2").astype(np.float32)    # [n,nb,1]
    out = np.empty((n, nb, 2, 128), np.float32)
    for half in range(2):
        l = ql[:, :, half].astype(np.int16)
        h = qh[:, :, half].astype(np.int16)
        q1 = ((l[:, :, 0:32] & 0xF) | (((h >> 0) & 3) << 4)) - 32
        q2 = ((l[:, :, 32:64] & 0xF) | (((h >> 2) & 3) << 4)) - 32
        q3 = ((l[:, :, 0:32] >> 4) | (((h >> 4) & 3) << 4)) - 32
        q4 = ((l[:, :, 32:64] >> 4) | (((h >> 6) & 3) << 4)) - 32
        s = sc[:, :, half * 8:(half + 1) * 8]
        dd = d
        # is = l/16 for l in 0..31 -> 0 or 1
        for k, q in enumerate((q1, q2, q3, q4)):
            sl = np.repeat(s[:, :, 2 * k:2 * k + 2], 16, axis=2)    # [n,nb,32]
            out[:, :, half, 32 * k:32 * (k + 1)] = dd * sl * q
    return out.reshape(n, cols)


DEQUANT = {"F32": _f32, "F16": _f16, "BF16": _bf16, "Q8_0": _q8_0, "Q2_K": _q2_k, "Q4_K": _q4_k, "Q5_K": _q5_k, "Q6_K": _q6_k}


if __name__ == "__main__":
    import sys
    g = GgufFile(sys.argv[1])
    for name in ("token_embd.weight", "output.weight"):
        print(name, g.info(name))
    print({k: v for k, v in g.metadata.items() if not hasattr(v, "__len__") or isinstance(v, str)
           and len(v) < 200})

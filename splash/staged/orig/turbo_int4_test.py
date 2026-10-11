#!/usr/bin/env python3
"""Kernel-level check of TurboQuant int4 KV (kv::Format::TurboInt4) against NumPy: the store's
packed 4-bit rows and fp32 scales, byte for byte, and the 256-point query rotation."""
import json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from turbo_kernel_test import FLATTEN, SRC, WORK_ROOT, bf16  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
WORK = WORK_ROOT / "turbo-int4-test"
HEADS, TOKENS, D = 4, 32, 256
SIGNS = json.loads((HERE / "turbo_signs.json").read_text())
T1, T2 = np.array(SIGNS["signs256_1"], np.float32), np.array(SIGNS["signs256_2"], np.float32)


def fwht(x):
    x = x.copy(); n = x.shape[-1]; h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = x[..., 0, :].copy(), x[..., 1, :].copy()
        x[..., 0, :], x[..., 1, :] = a + b, a - b
        x = x.reshape(*x.shape[:-3], n)
        h *= 2
    return x / np.sqrt(n)


def rotate256(x, inverse=False):
    a, b = (T2, T1) if inverse else (T1, T2)
    return fwht(x * a) * b


def quantize(rows):
    r = rotate256(rows.astype(np.float32))
    m = np.abs(r).max(-1)
    scale = np.where(m > 0, m / 7.0, 0.0).astype(np.float32)
    inv = np.where(m > 0, 7.0 / np.where(m > 0, m, 1), 0.0)
    q = np.clip(np.rint(r * inv[..., None]), -7, 7).astype(np.int8)
    return q, scale


def pack(q):                                  # low nibble = even element
    u = (q.astype(np.int16) & 0xF).astype(np.uint8)
    return (u[..., 0::2] | (u[..., 1::2] << 4)).astype(np.uint8)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    keys = rng.normal(0, 1.0, (HEADS, TOKENS, D)).astype(np.float32)
    keys[..., rng.choice(D, 6, replace=False)] *= 12.0
    values = rng.normal(0, 0.5, (HEADS, TOKENS, D)).astype(np.float32)
    keys[0, 3] = 0.0
    keys, values = bf16(keys), bf16(values)
    queries = bf16(rng.normal(0, 1.0, (40, D)).astype(np.float32))
    keys.tofile(WORK / "keys.f32"); values.transpose(0, 2, 1).copy().tofile(WORK / "values.f32")
    queries.tofile(WORK / "queries.f32")
    flat = WORK / "test.metal"
    flat.write_text(subprocess.run([sys.executable, str(FLATTEN), str(SRC),
                                    str(HERE / "turbo_test_kernels.metal")], capture_output=True, text=True, check=True).stdout)
    r = subprocess.run([str(HERE / "turbo_harness"), str(flat), str(WORK), "int4"], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        print(r.stderr); return 1
    raw = np.fromfile(WORK / "extent.bin", np.uint8)
    data, scales = HEADS * TOKENS * D // 2, HEADS * TOKENS * 4
    k_bytes = raw[:data].reshape(HEADS, TOKENS, D // 2)
    k_scale = raw[data:data + scales].view(np.float32).reshape(HEADS, TOKENS)
    v_bytes = raw[data + scales:2 * data + scales].reshape(HEADS, TOKENS, D // 2)
    v_scale = raw[2 * data + scales:2 * data + 2 * scales].view(np.float32).reshape(HEADS, TOKENS)
    ok = True
    for name, rows, got_b, got_s in (("keys", keys, k_bytes, k_scale), ("values", values, v_bytes, v_scale)):
        q, s = quantize(rows)
        want_b = pack(q)
        byte_diff = int((got_b != want_b).sum())
        scale_err = float(np.abs(got_s - s).max() / max(1e-12, np.abs(s).max()))
        # a code may differ by one where x*7/max sits on a .5 boundary in float rounding
        lo = (got_b & 0xF).astype(np.int8); hi = (got_b >> 4).astype(np.int8)
        got_q = np.empty_like(q); got_q[..., 0::2] = np.where(lo >= 8, lo - 16, lo); got_q[..., 1::2] = np.where(hi >= 8, hi - 16, hi)
        code_diff = np.abs(got_q.astype(int) - q.astype(int))
        recon = rotate256(got_q * got_s[..., None], inverse=True)
        rel = np.linalg.norm((recon - rows).reshape(-1, D), axis=1) / np.maximum(1e-9, np.linalg.norm(rows.reshape(-1, D), axis=1))
        print(f"{name}: bytes differing {byte_diff}/{got_b.size}, codes off by >1: {(code_diff > 1).sum()}, "
              f"scale max rel err {scale_err:.2e}, reconstruction error median {np.median(rel[np.linalg.norm(rows.reshape(-1, D), axis=1) > 0]):.3f}")
        ok &= (code_diff > 1).sum() == 0 and (code_diff > 0).mean() < 0.001 and scale_err < 1e-5
    q_rot = np.fromfile(WORK / "queries_out.f32", np.float32).reshape(-1, D)
    q_ref = rotate256(queries)
    q_err = np.abs(q_rot - bf16(q_ref)).max() / np.abs(q_ref).max()
    print(f"256-point query rotation: max relative error {q_err:.3g}")
    ok &= q_err < 0.01
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

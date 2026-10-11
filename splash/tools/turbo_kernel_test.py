#!/usr/bin/env python3
"""Kernel-level check of Splash's TurboQuant KV against an independent NumPy reference.

Runs the real Metal kernels through tools/turbo_harness (Objective-C++, runtime-compiled):
  1. prefill_attention_turbo_store on random bf16 key/value rows (one page, 4 KV heads),
  2. the attention tile's dequantizers (splash_turbo_dequantize_keys/values) on that page,
  3. turbo_rotate_queries on random vectors,
and compares with TurboQuant as the llama.cpp fork defines it (normalize, signs1 -> normalized
128-point Walsh-Hadamard -> signs2, nearest Lloyd-Max centroid, norm / |centroids|).

    turbo_kernel_test.py   (needs numpy; exits non-zero on mismatch)
"""
import json, os, pathlib, re, subprocess, sys, tempfile
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "splash-src" / "runtime"
FLATTEN = SRC.parent / "dev" / "tools" / "flatten_metal.py"   # the build's own flattener
WORK_ROOT = pathlib.Path(os.environ.get("TURBO_WORK", tempfile.gettempdir()))
WORK = WORK_ROOT / "turbo-kernel-test"
HEADS, TOKENS, D = 4, 32, 256

def header_table(name):
    """A float table of turbo_kv.h, so the codebooks have one source: the header the kernels use."""
    text = (SRC / "metal" / "kernels" / "common" / "turbo_kv.h").read_text()
    m = re.search(rf"{name}\[(\d+)\]\s*=\s*\{{([^}}]*)\}}", text)
    values = np.array([float(v.strip().rstrip("f")) for v in m.group(2).split(",") if v.strip()], np.float32)
    assert len(values) == int(m.group(1)), name
    return values


C4, M4 = header_table("SplashTurboCentroids4"), header_table("SplashTurboMidpoints4")
C3, M3 = header_table("SplashTurboCentroids3"), header_table("SplashTurboMidpoints3")
for c, m in ((C4, M4), (C3, M3)):              # catch a typo in the header
    assert np.all(np.diff(c) > 0) and np.allclose(c, -c[::-1]) and m[len(m) // 2] == 0
    assert np.allclose(m, (c[1:] + c[:-1]) / 2, atol=2e-6)
SIGNS = json.loads((HERE / "turbo_signs.json").read_text())
S1, S2 = np.array(SIGNS["signs1"], np.float32), np.array(SIGNS["signs2"], np.float32)


def fwht(x):                                   # normalized, last axis 128, the fork's loop order
    x = x.copy()
    h = 1
    while h < 128:
        x = x.reshape(*x.shape[:-1], 128 // (2 * h), 2, h)
        a, b = x[..., 0, :].copy(), x[..., 1, :].copy()
        x[..., 0, :], x[..., 1, :] = a + b, a - b
        x = x.reshape(*x.shape[:-3], 128)
        h *= 2
    return x / np.sqrt(128.0)


def rotate(x, inverse=False):
    a, b = (S2, S1) if inverse else (S1, S2)
    return fwht(x * a) * b


def bf16(x):                                   # round-to-nearest-even to bfloat16, back to float32
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16
    return u.astype(np.uint32).view(np.float32)


def reference(rows, centroids, midpoints):
    """rows [..., 256] -> (indices [..., 256], stored fp16 norms [..., 2], dequantized rotated [..., 256])"""
    g = rows.reshape(*rows.shape[:-1], 2, 128).astype(np.float32)
    n = np.sqrt((g * g).sum(-1))
    u = g / np.where(n > 1e-10, n, 1)[..., None] * (n > 1e-10)[..., None]
    r = rotate(u)
    idx = np.searchsorted(midpoints, r, side="right")      # midpoints at or below the value
    c = centroids[idx]
    rn = np.sqrt((c * c).sum(-1))
    norm = np.where(rn > 1e-10, n / np.where(rn > 1e-10, rn, 1), n).astype(np.float16)
    deq = c * norm.astype(np.float32)[..., None]
    return idx.reshape(rows.shape), norm, deq.reshape(rows.shape), r.reshape(rows.shape)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    # Keys with a few large channels (as RoPE'd keys have), values plainer; bf16 like the staging.
    keys = rng.normal(0, 1.0, (HEADS, TOKENS, D)).astype(np.float32)
    keys[..., rng.choice(D, 6, replace=False)] *= 12.0
    values = rng.normal(0, 0.5, (HEADS, TOKENS, D)).astype(np.float32)
    keys[0, 3] = 0.0                                            # an all-zero row
    keys, values = bf16(keys), bf16(values)
    queries = bf16(rng.normal(0, 1.0, (40, D)).astype(np.float32))
    # staging layouts of the store: keys [head][token][dim], values [head][dim][token]
    keys.astype(np.float32).tofile(WORK / "keys.f32")
    values.transpose(0, 2, 1).copy().astype(np.float32).tofile(WORK / "values.f32")
    queries.astype(np.float32).tofile(WORK / "queries.f32")

    flat = WORK / "test.metal"
    out = subprocess.run([sys.executable, str(FLATTEN), str(SRC), str(HERE / "turbo_test_kernels.metal")],
                         capture_output=True, text=True, check=True).stdout
    flat.write_text(out)
    r = subprocess.run([str(HERE / "turbo_harness"), str(flat), str(WORK)], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        print(r.stderr)
        return 1

    k_tile = np.fromfile(WORK / "keys_out.f32", np.float32).reshape(HEADS, TOKENS, D)
    v_tile = np.fromfile(WORK / "values_out.f32", np.float32).reshape(HEADS, D, TOKENS).transpose(0, 2, 1)
    q_rot = np.fromfile(WORK / "queries_out.f32", np.float32).reshape(-1, D)

    k_idx, k_norm, k_ref, k_rot = reference(keys, C4, M4)
    v_idx, v_norm, v_ref, v_rot = reference(values, C3, M3)
    ok = True

    def report(name, got, ref, tol):
        nonlocal ok
        err = np.abs(got - bf16(ref))
        scale = np.abs(ref).max()
        bad = err > tol * scale + 1e-6
        # an index may legitimately flip where the rotated value sits on a decision midpoint
        # within float rounding; count those separately
        frac = bad.mean()
        print(f"{name}: max |got-ref| = {err.max():.3g} (ref max {scale:.3g}), mismatching elements {bad.sum()} "
              f"({frac:.4%})")
        if frac > 0.001:
            ok = False

    report("keys   (4-bit)", k_tile, k_ref, 0.01)
    report("values (3-bit)", v_tile, v_ref, 0.01)

    # Quantization quality: relative error of the reconstruction in the original basis
    def rel_err(rows, deq_rotated):
        back = rotate(deq_rotated.reshape(*rows.shape[:-1], 2, 128), inverse=True).reshape(rows.shape)
        num = np.linalg.norm((back - rows).reshape(-1, D), axis=1)
        den = np.linalg.norm(rows.reshape(-1, D), axis=1)
        return float(np.median(num[den > 0] / den[den > 0]))
    print(f"reconstruction error (median relative, original basis): keys {rel_err(keys, k_tile):.3f}, "
          f"values {rel_err(values, v_tile):.3f}  (TurboQuant expects ~0.10 at 4 bits, ~0.18 at 3 bits)")

    q_ref = rotate(queries.reshape(-1, 2, 128)).reshape(-1, D)
    q_err = np.abs(q_rot - bf16(q_ref)).max() / np.abs(q_ref).max()
    print(f"query rotation: max relative error {q_err:.3g}")
    ok &= q_err < 0.01
    # rotation keeps inner products: <R q, R k> == <q, k> (what lets attention run rotated)
    k_rows = keys[1].astype(np.float64)
    rk = rotate(k_rows.reshape(-1, 2, 128)).reshape(-1, D)
    dot_err = np.abs(q_ref.astype(np.float64) @ rk.T - queries.astype(np.float64) @ k_rows.T).max()
    print(f"rotation keeps dot products: max |<Rq,Rk> - <q,k>| = {dot_err:.2e}")
    ok &= dot_err < 1e-3
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

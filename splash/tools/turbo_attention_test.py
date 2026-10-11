#!/usr/bin/env python3
"""End-to-end check of one TurboQuant verify attention (store, query rotation, split, reduce).

The GPU result (tools/turbo_attn_harness) must equal attention computed in the model's own basis
from the reference TurboQuant reconstruction of the keys and values (turbo_kernel_test.reference),
up to bf16 rounding. It also reports how close TurboQuant attention is to exact attention.

    turbo_attention_test.py [COMMITTED] [SPLITS]
"""
import pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from turbo_kernel_test import C3, C4, FLATTEN, M3, M4, SRC, WORK_ROOT, bf16, reference, rotate  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
WORK = WORK_ROOT / "turbo-attention-test"
HEADS, GROUP, D, ROWS, STRIDE = 4, 6, 256, 8, 32


def attention(q, k, v, committed):
    """q [heads][rows][group][D], k/v [heads][tokens][D] -> [heads][rows][group][D] (causal)"""
    out = np.zeros_like(q, dtype=np.float64)
    for h in range(HEADS):
        for i in range(ROWS):
            visible = committed + i + 1
            s = q[h, i] @ k[h, :visible].T * 0.0625            # [group][visible]
            s -= s.max(-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(-1, keepdims=True)
            out[h, i] = p @ v[h, :visible]
    return out


def main():
    committed = int(sys.argv[1]) if len(sys.argv) > 1 else 85
    splits = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    int4 = len(sys.argv) > 3 and sys.argv[3] == "int4"
    tokens = committed + ROWS
    stride = (tokens + 31) // 32 * 32
    WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    keys = rng.normal(0, 1.0, (HEADS, tokens, D)).astype(np.float32)
    keys[..., rng.choice(D, 6, replace=False)] *= 8.0
    values = rng.normal(0, 1.0, (HEADS, tokens, D)).astype(np.float32)
    queries = rng.normal(0, 1.0, (HEADS, ROWS, GROUP, D)).astype(np.float32)
    # a query aligned with one history key, so attention is peaked somewhere
    queries[1, 3, 2] = keys[1, min(40, committed)] * 0.5
    keys, values, queries = bf16(keys), bf16(values), bf16(queries)

    k_stage = np.zeros((HEADS, stride, D), np.float32); k_stage[:, :tokens] = keys
    v_stage = np.zeros((HEADS, D, stride), np.float32); v_stage[:, :, :tokens] = values.transpose(0, 2, 1)
    q_stage = np.zeros((HEADS, STRIDE, GROUP, D), np.float32); q_stage[:, :ROWS] = queries
    k_stage.tofile(WORK / "keys.f32"); v_stage.tofile(WORK / "values.f32"); q_stage.tofile(WORK / "queries.f32")

    flat = WORK / "test.metal"
    flat.write_text(subprocess.run([sys.executable, str(FLATTEN), str(SRC),
                                    str(HERE / "turbo_test_kernels.metal")], capture_output=True, text=True,
                                   check=True).stdout)
    r = subprocess.run([str(HERE / "turbo_attn_harness"), str(flat), str(WORK), str(committed), str(splits)]
                       + (["int4"] if int4 else []), capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        print(r.stderr)
        return 1
    got = np.fromfile(WORK / "output.f32", np.float32).reshape(HEADS, STRIDE, GROUP, D)[:, :ROWS]

    # reference: TurboQuant reconstruction rotated back to the model's basis
    if int4:
        from turbo_int4_test import quantize, rotate256
        (kq, ks), (vq, vs) = quantize(keys), quantize(values)
        k_hat = rotate256(kq * ks[..., None], inverse=True)
        v_hat = rotate256(vq * vs[..., None], inverse=True)
    else:
        _, _, k_deq, _ = reference(keys, C4, M4)
        _, _, v_deq, _ = reference(values, C3, M3)
        k_hat = rotate(k_deq.reshape(HEADS, tokens, 2, 128), inverse=True).reshape(HEADS, tokens, D)
        v_hat = rotate(v_deq.reshape(HEADS, tokens, 2, 128), inverse=True).reshape(HEADS, tokens, D)
    want = attention(queries.astype(np.float64), k_hat, v_hat, committed)
    exact = attention(queries.astype(np.float64), keys.astype(np.float64), values.astype(np.float64), committed)

    def cos(a, b):
        a, b = a.reshape(-1, D), b.reshape(-1, D)
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

    err = np.abs(got - want).max() / np.abs(want).max()
    c_impl, c_exact = cos(got, want), cos(got, exact)
    print(f"GPU vs TurboQuant reference: max relative error {err:.4f}, cosine min {c_impl.min():.5f} mean {c_impl.mean():.5f}")
    print(f"TurboQuant vs exact attention: cosine min {c_exact.min():.4f} mean {c_exact.mean():.4f}")
    peaked = got[1, 3, 2]
    print(f"peaked row: cosine to exact {cos(peaked, exact[1, 3, 2])[0]:.4f}")
    ok = c_impl.min() > 0.999 and err < 0.02 and np.isfinite(got).all()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

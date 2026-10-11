#!/usr/bin/env python3
"""Time TurboQuant attention kernel variants against INT8 (tools/attn_perf), from the current
sources with textual edits applied to the flattened kernel file.

    perf_variants.py [HISTORY ...]          (default 16384 57344)
"""
import os, pathlib, re, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "splash-src" / "runtime"
TMP = pathlib.Path(os.environ.get("TURBO_WORK", tempfile.gettempdir())) / "perf-variants"


def base():
    return subprocess.run([sys.executable, str(SRC.parent / "dev" / "tools" / "flatten_metal.py"), str(SRC), str(HERE / "perf_kernels.metal")],
                          capture_output=True, text=True, check=True).stdout


def replace(text, old, new):
    assert old in text, old[:80]
    return text.replace(old, new)


VARIANTS = {
    "current": lambda s: s,
    "half-tile": lambda s: replace(s, "typedef bfloat SplashTurboTileElement;", "typedef half SplashTurboTileElement;"),
    # keys: codebook shuffle replaced by arithmetic on the index
    "k-no-lut": lambda s: replace(s, "      value = simd_shuffle(codebook, ushort(index)) *\n",
                                  "      value = (float(index) * 0.02f - 0.15f) *\n"),
    # keys: no bit extraction (index from the lane)
    "k-no-bits": lambda s: replace(s, """      const uint index = ((word.x >> simd_lane) & 1u) | (((word.y >> simd_lane) & 1u) << 1) |
                         (((word.z >> simd_lane) & 1u) << 2) |
                         (((word.w >> simd_lane) & 1u) << 3);""", """      const uint index = (simd_lane + word.x) & 15u;"""),
    # values: codebook shuffle replaced by arithmetic
    "v-no-lut": lambda s: replace(s, "SplashTurboTileElement(simd_shuffle(codebook, ushort(index)) * norm);",
                                  "SplashTurboTileElement((float(index) * 0.05f - 0.18f) * norm);"),
    # keys: unrolled token loop so loads can be issued ahead
    "k-unroll8": lambda s: replace(s, """  const float codebook = SplashTurboCentroids4[simd_lane & 15u];
  for (uint token = 0; token < SplashKvPageTokens; ++token) {""", """  const float codebook = SplashTurboCentroids4[simd_lane & 15u];
#pragma unroll 8
  for (uint token = 0; token < SplashKvPageTokens; ++token) {"""),
    # keys: all words loaded up front into registers (4 tokens per load batch)
    "k-prefetch": lambda s: replace(s, """  const float codebook = SplashTurboCentroids4[simd_lane & 15u];
  for (uint token = 0; token < SplashKvPageTokens; ++token) {
    float value = 0.0f;
    if (token < valid_tokens) {
      const uint4 word = planes[token * SplashTurboChunks + chunk];""", """  const float codebook = SplashTurboCentroids4[simd_lane & 15u];
  uint4 words[SplashKvPageTokens];
  half norms[SplashKvPageTokens];
#pragma unroll
  for (uint token = 0; token < SplashKvPageTokens; ++token) {
    words[token] = planes[token * SplashTurboChunks + chunk];
    norms[token] = page.key_norms[token * SplashTurboGroups + group];
  }
#pragma unroll
  for (uint token = 0; token < SplashKvPageTokens; ++token) {
    float value = 0.0f;
    if (token < valid_tokens) {
      const uint4 word = words[token];""").replace("""      value = simd_shuffle(codebook, ushort(index)) *
              float(page.key_norms[token * SplashTurboGroups + group]);""", """      value = simd_shuffle(codebook, ushort(index)) * float(norms[token]);"""),
    "keys-only": lambda s: replace(s,
        "    splash_turbo_dequantize_values(tensors, kv_tile, valid_tokens, simd_lane, simd_group);\n", ""),
    "values-only": lambda s: replace(s,
        "    splash_turbo_dequantize_keys(tensors, kv_tile, valid_tokens, simd_lane, simd_group);\n", ""),
    # keys: the loop only stores (no loads, no math)
    "k-store-only": lambda s: replace(s, """    float value = 0.0f;
    if (token < valid_tokens) {
      const uint4 word = planes[token * SplashTurboChunks + chunk];
      const uint index = ((word.x >> simd_lane) & 1u) | (((word.y >> simd_lane) & 1u) << 1) |
                         (((word.z >> simd_lane) & 1u) << 2) |
                         (((word.w >> simd_lane) & 1u) << 3);
      value = simd_shuffle(codebook, ushort(index)) *
              float(page.key_norms[token * SplashTurboGroups + group]);
    }""", """    float value = codebook * float(token);"""),
    "no-dequant": lambda s: replace(replace(s,
        """    splash_turbo_dequantize_keys(tensors, kv_tile,
                                 reinterpret_cast<threadgroup uint4 *>(scores), valid_tokens,
                                 thread_index, simd_lane, simd_group);\n""", ""),
        "    splash_turbo_dequantize_values(tensors, kv_tile, valid_tokens, simd_lane, simd_group);\n", ""),
}


def main():
    histories = [int(h) for h in sys.argv[1:]] or [16384, 57344]
    TMP.mkdir(parents=True, exist_ok=True)
    source = base()
    only = [v for v in VARIANTS]
    for name in only:
        path = TMP / f"{name}.metal"
        try:
            path.write_text(VARIANTS[name](source))
        except AssertionError as e:
            print(f"{name}: variant does not apply ({e})")
            continue
        for h in histories:
            r = subprocess.run([str(HERE / "attn_perf"), str(path), str(h)], capture_output=True, text=True)
            if r.returncode:
                print(f"{name:12s} h={h}: FAILED {r.stderr.strip()[:300]}")
                continue
            lines = r.stdout.splitlines()
            def nums(fmt):
                line = next(l for l in lines if l.strip().startswith(fmt))
                pf = float(re.search(r"prefill chunk\s+([\d.]+) ms", line)[1])
                vf = float(re.search(r"verify step\s+([\d.]+) ms", line)[1])
                return pf, vf
            (ip, iv), (tp, tv) = nums("int8"), nums("turbo")
            print(f"{name:12s} h={h:6d}: prefill {tp:7.1f} ms ({tp / ip:4.2f}x int8 {ip:6.1f})   "
                  f"verify {tv:6.3f} ms ({tv / iv:4.2f}x int8 {iv:5.3f})", flush=True)


if __name__ == "__main__":
    main()

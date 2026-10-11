#pragma once

#include "metal/abi/KvExtent.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

// TurboQuant KV pages (kv::Format::Turbo): keys at 4 bits and values at 3
// bits per element, after the PolarQuant recipe of TurboQuant (arXiv
// 2504.19874) as atomic-llama-cpp-turboquant implements it:
//
//   per 128 dimensions of a head vector x: n = |x|, u = x / n,
//   r = S2 H S1 u   (S1, S2 fixed random signs, H the normalized
//                    128-point Walsh-Hadamard transform),
//   idx_i = nearest Lloyd-Max centroid of r_i for N(0, 1/128),
//   stored norm = n / |c(idx)|, so the dequantized c(idx) * norm keeps |x|.
//
// Dequantized vectors stay rotated: rotating the queries the same way keeps
// every dot product, and the attention output is rotated back once per row.
// Signs and centroids match the llama.cpp fork (seed 42), so the arithmetic
// is the one its turbo4 keys and turbo3 values use; only the packing differs.
//
// Packing: token-major bit-planes. Per (page, KV head, token) a key row is
// 8 chunks of 32 dimensions x 4 planes of one uint (128 bytes); a value row
// is 8 chunks x 3 planes (96 bytes). Bit j of plane p of chunk c is bit p of
// dimension 32c + j's index: one simd_ballot writes a plane, and one load per
// plane serves a chunk. Norms are fp16, two per (page, KV head, token) for
// each tensor, in the regions that hold the INT8 scales.

// The element type the attention tile dequantizes pages into for the MPP
// products (paged_attention_tile.h).
typedef bfloat SplashTurboTileElement;

constant constexpr uint SplashTurboGroup = 128;
constant constexpr uint SplashTurboGroups = SPLASH_KV_HEAD_DIMENSION / SplashTurboGroup;
constant constexpr uint SplashTurboChunks = SPLASH_KV_HEAD_DIMENSION / 32;
constant constexpr uint SplashTurboKeyPlanes = 4;
constant constexpr uint SplashTurboValuePlanes = 3;

// 4-bit Lloyd-Max centroids for N(0, 1/128) and their decision midpoints: Max's 16-level
// table for N(0, 1) (±0.1284 ... ±2.7326) over sqrt(128), exactly symmetric, MSE 0.0095/128.
// The fork's turbo4 table (±0.006938 ... ±0.173926) is ~0.6-0.7x narrower; this one gives ~23%
// lower key error (cache revision 2 for turbo, splash/IMPROVEMENT_PLAN.md step 5).
constant constexpr float SplashTurboCentroids4[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f, -0.083292f, -0.058050f,
    -0.034299f, -0.011349f, 0.011349f,  0.034299f,  0.058050f,  0.083292f,
    0.111036f,  0.143016f,  0.182877f,  0.241529f};
constant constexpr float SplashTurboMidpoints4[15] = {
    -0.212203f, -0.162947f, -0.127026f, -0.097164f, -0.070671f,
    -0.046174f, -0.022824f, 0.000000f,  0.022824f,  0.046174f,
    0.070671f,  0.097164f,  0.127026f,  0.162947f,  0.212203f};
// 3-bit centroids and midpoints.
constant constexpr float SplashTurboCentroids3[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
    0.021460f,  0.065717f,  0.117832f,  0.190685f};
constant constexpr float SplashTurboMidpoints3[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.000000f,
    0.043589f,  0.091775f,  0.154259f};

// The random signs S1 and S2 (seed 42) as masks, a set bit negative. Element
// 32r + l of a 128-dimension group is bit l of word r: the strided layout in
// which lane l of a simdgroup holds elements l, l+32, l+64 and l+96.
constant constexpr uint SplashTurboSigns1[4] = {0xe46a0359u, 0x42f85949u,
                                                0xa4c63bbbu, 0x49f250a9u};
constant constexpr uint SplashTurboSigns2[4] = {0x16acee90u, 0x3f628fdcu,
                                                0xb7a5357au, 0xbea56562u};

inline float splash_turbo_sign(constant const uint *masks, uint element_word,
                               uint lane) {
  return (masks[element_word] >> lane) & 1u ? -1.0f : 1.0f;
}

// The normalized 128-point Walsh-Hadamard transform of one group held
// strided across a simdgroup (x[r] is element 32r + lane). Strides below 32
// pair lanes, 32 and 64 pair a lane's own registers; the butterflies commute.
inline void splash_turbo_hadamard(thread float (&x)[4], uint lane) {
#pragma unroll
  for (uint stride = 1; stride < 32; stride <<= 1) {
    const bool upper = lane & stride;
#pragma unroll
    for (uint r = 0; r < 4; ++r) {
      const float partner = simd_shuffle_xor(x[r], ushort(stride));
      x[r] = upper ? partner - x[r] : x[r] + partner;
    }
  }
  float a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + b, x[1] = a - b, x[2] = c + d, x[3] = c - d;   // stride 32
  a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + c, x[1] = b + d, x[2] = a - c, x[3] = b - d;   // stride 64
  constexpr float inverse_sqrt_128 = 0.08838834764831845f;
#pragma unroll
  for (uint r = 0; r < 4; ++r)
    x[r] *= inverse_sqrt_128;
}

// Forward rotation S2 H S1 (keys and values when stored, queries before
// attention) or its inverse S1 H S2 (the attention output).
inline void splash_turbo_rotate(thread float (&x)[4], uint lane, bool inverse) {
  constant const uint *first = inverse ? SplashTurboSigns2 : SplashTurboSigns1;
  constant const uint *second = inverse ? SplashTurboSigns1 : SplashTurboSigns2;
#pragma unroll
  for (uint r = 0; r < 4; ++r)
    x[r] *= splash_turbo_sign(first, r, lane);
  splash_turbo_hadamard(x, lane);
#pragma unroll
  for (uint r = 0; r < 4; ++r)
    x[r] *= splash_turbo_sign(second, r, lane);
}

// Nearest centroid: the number of midpoints at or below the value, by
// branchless binary search (the fork's `val < mid[k]` chain, same ties).
inline uint splash_turbo_index4(float value) {
  uint index = value >= SplashTurboMidpoints4[7] ? 8u : 0u;
  index += value >= SplashTurboMidpoints4[index + 3] ? 4u : 0u;
  index += value >= SplashTurboMidpoints4[index + 1] ? 2u : 0u;
  index += value >= SplashTurboMidpoints4[index] ? 1u : 0u;
  return index;
}

inline uint splash_turbo_index3(float value) {
  uint index = value >= SplashTurboMidpoints3[3] ? 4u : 0u;
  index += value >= SplashTurboMidpoints3[index + 1] ? 2u : 0u;
  index += value >= SplashTurboMidpoints3[index] ? 1u : 0u;
  return index;
}

inline uint splash_turbo_ballot(bool bit) {
  return uint(static_cast<simd_vote::vote_t>(simd_ballot(bit)));
}

// Bytes of one tensor of one page of one layer for KVHeads heads, the host
// Layout's numbers (ops/PagedKv.hpp) on the kernel side.
template <uint KVHeads> struct SplashTurboPageBytes {
  static constexpr constant uint KeyData =
      KVHeads * SplashKvPageTokens * SplashTurboChunks * SplashTurboKeyPlanes * 4;
  static constexpr constant uint ValueData =
      KVHeads * SplashKvPageTokens * SplashTurboChunks * SplashTurboValuePlanes * 4;
  static constexpr constant uint Norms =
      KVHeads * SplashKvPageTokens * SplashTurboGroups * 2;
};

// One KV head's slab of a TurboQuant page: its 32 token rows of key and
// value planes and their norms.
struct SplashTurboPageTensors {
  device uint *key_planes;       // [token][chunk][4]
  device uint *value_planes;     // [token][chunk][3]
  device half *key_norms;        // [token][group]
  device half *value_norms;      // [token][group]
};

template <uint KVHeads> struct SplashTurboAddressing {
  using Bytes = SplashTurboPageBytes<KVHeads>;
  uint layer_offset;
  uint head;
  ulong key_norms_offset;
  ulong values_offset;
  ulong value_norms_offset;

  SplashTurboAddressing(SplashKvLayer kv, uint kv_head)
      : layer_offset(kv.offset), head(kv_head) {
    const SplashKvPageGeometry geometry = {Bytes::KeyData, Bytes::Norms,
                                           Bytes::ValueData, Bytes::Norms};
    key_norms_offset =
        splash_kv_geometry_offset(kv.extent_pages, geometry, 0, SPLASH_KV_KEY_SCALES, 0);
    values_offset =
        splash_kv_geometry_offset(kv.extent_pages, geometry, 0, SPLASH_KV_VALUES, 0);
    value_norms_offset =
        splash_kv_geometry_offset(kv.extent_pages, geometry, 0, SPLASH_KV_VALUE_SCALES, 0);
  }

  SplashTurboPageTensors page(SplashKvPage entry) const {
    const uint index = splash_kv_page_index(entry);
    device uchar *region = splash_kv_extent(entry, index) + layer_offset;
    constexpr uint KeyWordsPerHead = SplashKvPageTokens * SplashTurboChunks * SplashTurboKeyPlanes;
    constexpr uint ValueWordsPerHead =
        SplashKvPageTokens * SplashTurboChunks * SplashTurboValuePlanes;
    constexpr uint NormsPerHead = SplashKvPageTokens * SplashTurboGroups;
    SplashTurboPageTensors tensors{};
    tensors.key_planes = reinterpret_cast<device uint *>(region) +
                         ulong(index) * (KVHeads * KeyWordsPerHead) + head * KeyWordsPerHead;
    tensors.value_planes = reinterpret_cast<device uint *>(region + values_offset) +
                           ulong(index) * (KVHeads * ValueWordsPerHead) +
                           head * ValueWordsPerHead;
    tensors.key_norms = reinterpret_cast<device half *>(region + key_norms_offset) +
                        ulong(index) * (KVHeads * NormsPerHead) + head * NormsPerHead;
    tensors.value_norms = reinterpret_cast<device half *>(region + value_norms_offset) +
                          ulong(index) * (KVHeads * NormsPerHead) + head * NormsPerHead;
    return tensors;
  }
};

// ---- TurboQuant int4 pages (kv::Format::TurboInt4) ------------------------
// The same idea mapped onto the GPU's native 4-bit matrix operands: each head
// vector of keys or values is rotated by a randomized 256-point Hadamard
// transform R = S2 H256 S1 (one rotation over the whole head, so one scale
// fits it), then quantized symmetrically to int4 (-7..7) with one fp32
// absmax scale per (token, KV head), the INT8 scale layout. MPP reads the
// pages as int4b_format (two's complement, element 2i in the low nibble of
// byte i), keys and values token-major: no dequantization in the attention
// loop. Queries are rotated by R before attention and the output by R^T after.

// turbo-int4 clips each rotated row at this many RMS before quantizing (step 2 of
// splash/IMPROVEMENT_PLAN.md). A change alters stored pages: bump the persistent-cache revision.
constant constexpr float SplashTurboInt4Clip = 2.7f;
constant constexpr uint SplashTurboInt4RowBytes = SPLASH_KV_HEAD_DIMENSION / 2;

// Random signs (Python random.Random(250419874)), element 32r + l in bit l of
// word r.
constant constexpr uint SplashTurboSigns256First[8] = {
    0xa097d901u, 0xa038e573u, 0xca2221ffu, 0x45093648u,
    0x290f5046u, 0x0de9bed1u, 0x012987dbu, 0xa51d810eu};
constant constexpr uint SplashTurboSigns256Second[8] = {
    0x5c73128au, 0x6cc37310u, 0xc8bab385u, 0xd9d9ff54u,
    0x63ee4d71u, 0xa2b23130u, 0x32884643u, 0xb6e6c2f6u};

// The normalized 256-point Walsh-Hadamard transform of one head vector held
// strided across a simdgroup (x[r] is element 32r + lane).
inline void splash_turbo_hadamard256(thread float (&x)[8], uint lane) {
#pragma unroll
  for (uint stride = 1; stride < 32; stride <<= 1) {
    const bool upper = lane & stride;
#pragma unroll
    for (uint r = 0; r < 8; ++r) {
      const float partner = simd_shuffle_xor(x[r], ushort(stride));
      x[r] = upper ? partner - x[r] : x[r] + partner;
    }
  }
#pragma unroll
  for (uint step = 1; step < 8; step <<= 1) {               // strides 32, 64, 128
#pragma unroll
    for (uint r = 0; r < 8; ++r) {
      if (r & step)
        continue;
      const float a = x[r], b = x[r + step];
      x[r] = a + b;
      x[r + step] = a - b;
    }
  }
  constexpr float inverse_sqrt_256 = 0.0625f;
#pragma unroll
  for (uint r = 0; r < 8; ++r)
    x[r] *= inverse_sqrt_256;
}

inline void splash_turbo_rotate256(thread float (&x)[8], uint lane, bool inverse) {
  constant const uint *first = inverse ? SplashTurboSigns256Second : SplashTurboSigns256First;
  constant const uint *second = inverse ? SplashTurboSigns256First : SplashTurboSigns256Second;
#pragma unroll
  for (uint r = 0; r < 8; ++r)
    x[r] *= splash_turbo_sign(first, r, lane);
  splash_turbo_hadamard256(x, lane);
#pragma unroll
  for (uint r = 0; r < 8; ++r)
    x[r] *= splash_turbo_sign(second, r, lane);
}

// Bytes of one tensor of one page of one layer: 4-bit data and fp32 scales,
// keys and values alike (the Layout of kv::Format::TurboInt4).
template <uint KVHeads> struct SplashTurboInt4PageBytes {
  static constexpr constant uint Data = KVHeads * SplashKvPageTokens * SplashTurboInt4RowBytes;
  static constexpr constant uint Scale = KVHeads * SplashKvPageTokens * sizeof(float);
};

// One KV head's slab of a TurboQuant int4 page: 32 token rows of 128 bytes
// for keys and for values, and their scales.
struct SplashTurboInt4PageTensors {
  device uchar *keys;
  device uchar *values;
  device float *key_scales;
  device float *value_scales;
};

template <uint KVHeads> struct SplashTurboInt4Addressing {
  using Bytes = SplashTurboInt4PageBytes<KVHeads>;
  uint layer_offset;
  uint head;
  ulong key_scales_offset;
  ulong values_offset;
  ulong value_scales_offset;

  SplashTurboInt4Addressing(SplashKvLayer kv, uint kv_head)
      : layer_offset(kv.offset), head(kv_head),
        key_scales_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale, 0,
                                           SPLASH_KV_KEY_SCALES, 0)),
        values_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale, 0,
                                       SPLASH_KV_VALUES, 0)),
        value_scales_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale, 0,
                                             SPLASH_KV_VALUE_SCALES, 0)) {}

  SplashTurboInt4PageTensors page(SplashKvPage entry) const {
    const uint index = splash_kv_page_index(entry);
    device uchar *region = splash_kv_extent(entry, index) + layer_offset;
    const ulong slab = ulong(index) * KVHeads + head;
    SplashTurboInt4PageTensors tensors{};
    tensors.keys = region + slab * (SplashKvPageTokens * SplashTurboInt4RowBytes);
    tensors.values = region + values_offset + slab * (SplashKvPageTokens * SplashTurboInt4RowBytes);
    tensors.key_scales =
        reinterpret_cast<device float *>(region + key_scales_offset) + slab * SplashKvPageTokens;
    tensors.value_scales =
        reinterpret_cast<device float *>(region + value_scales_offset) + slab * SplashKvPageTokens;
    return tensors;
  }
};

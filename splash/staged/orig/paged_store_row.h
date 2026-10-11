#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include "metal/kernels/common/turbo_kv.h"
#include <metal_stdlib>

using namespace metal;

constant uint SplashChunkMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
constant uint SplashChunkMaximumPhysicalTokens =
    SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS;

inline bool
splash_chunk_contract_valid(constant SplashChunkedPrefillParams &params) {
  uint required_pages = (params.committed_tokens + params.chunk_tokens +
                         SplashKvPageTokens - 1) /
                        SplashKvPageTokens;
  return params.committed_tokens <= SplashChunkMaximumPhysicalTokens &&
         params.chunk_tokens > 0 &&
         params.chunk_tokens <= SplashChunkMaximumRows &&
         params.committed_tokens + params.chunk_tokens <=
             SplashChunkMaximumPhysicalTokens &&
         params.chunk_stride >= params.chunk_tokens &&
         params.chunk_stride <= SplashChunkMaximumRows &&
         params.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         params.page_table_entries >= required_pages &&
         params.kv.extent_pages > 0;
}

inline ulong splash_current_key_index(uint stride, uint head, uint token,
                                        uint dimension) {
  return (ulong(head) * stride + token) * SplashKvHeadDimension + dimension;
}

inline ulong splash_current_value_index(uint stride, uint head, uint token,
                                          uint dimension) {
  return (ulong(head) * SplashKvHeadDimension + dimension) * stride + token;
}

// The CacheElement of TurboQuant pages (turbo_kv.h): a tag, as their packed
// planes have no per-element type.
struct SplashTurboKv {};

// A TurboQuant row: the 256 lanes stage the row in threadgroup memory, then
// simdgroups 0 and 1 each normalize, rotate and quantize one 128-dimension
// group, ballot its index bits into planes and store the corrected norm.
// `staging` holds SplashKvHeadDimension floats.
template <uint KVHeads>
__attribute__((always_inline)) inline void splash_store_turbo_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table,
    constant SplashChunkedPrefillParams &params, threadgroup float *staging,
    bool value_tensor, uint head, uint chunk_token, uint dimension,
    uint simd_lane, uint simd_group) {
  uint logical_token = params.committed_tokens + chunk_token;
  const SplashKvPage page = page_table[logical_token / SplashKvPageTokens];
  const SplashTurboPageTensors slab =
      SplashTurboAddressing<KVHeads>(params.kv, head).page(page);
  uint page_token = logical_token % SplashKvPageTokens;
  ulong source_index =
      value_tensor ? splash_current_value_index(params.chunk_stride, head,
                                                  chunk_token, dimension)
                   : splash_current_key_index(params.chunk_stride, head,
                                                chunk_token, dimension);
  staging[dimension] =
      float(value_tensor ? chunk_values[source_index] : chunk_keys[source_index]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group >= SplashTurboGroups)
    return;
  const uint group = simd_group;
  float x[4];
#pragma unroll
  for (uint r = 0; r < 4; ++r)
    x[r] = staging[group * SplashTurboGroup + 32 * r + simd_lane];
  const float norm =
      sqrt(simd_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]));
  const float inverse_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
#pragma unroll
  for (uint r = 0; r < 4; ++r)
    x[r] *= inverse_norm;
  splash_turbo_rotate(x, simd_lane, false);
  uint index[4];
  float reconstructed = 0.0f;
#pragma unroll
  for (uint r = 0; r < 4; ++r) {
    index[r] = value_tensor ? splash_turbo_index3(x[r]) : splash_turbo_index4(x[r]);
    const float centroid = value_tensor ? SplashTurboCentroids3[index[r]]
                                        : SplashTurboCentroids4[index[r]];
    reconstructed += centroid * centroid;
  }
  const float reconstructed_norm = sqrt(simd_sum(reconstructed));
  const half stored_norm =
      half(reconstructed_norm > 1e-10f ? norm / reconstructed_norm : norm);
  const uint planes = value_tensor ? SplashTurboValuePlanes : SplashTurboKeyPlanes;
  device uint *row_planes =
      (value_tensor ? slab.value_planes : slab.key_planes) +
      page_token * SplashTurboChunks * planes;
#pragma unroll
  for (uint r = 0; r < 4; ++r) {
    const uint chunk = group * 4 + r;
    for (uint plane = 0; plane < planes; ++plane) {
      const uint word = splash_turbo_ballot((index[r] >> plane) & 1u);
      if (simd_lane == 0)
        row_planes[chunk * planes + plane] = word;
    }
  }
  if (simd_lane == 0)
    (value_tensor ? slab.value_norms : slab.key_norms)[page_token * SplashTurboGroups + group] =
        stored_norm;
}

// The CacheElement of TurboQuant int4 pages (turbo_kv.h).
struct SplashTurboInt4Kv {};

// A TurboQuant int4 row: the lanes stage the row, then simdgroup 0 rotates
// the whole head vector, takes its absmax scale and packs 8 elements per uint
// (element 2i in the low nibble of byte i, as MPP reads int4b_format).
template <uint KVHeads>
__attribute__((always_inline)) inline void splash_store_turbo_int4_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table,
    constant SplashChunkedPrefillParams &params, threadgroup float *staging,
    bool value_tensor, uint head, uint chunk_token, uint dimension,
    uint simd_lane, uint simd_group) {
  uint logical_token = params.committed_tokens + chunk_token;
  const SplashKvPage page = page_table[logical_token / SplashKvPageTokens];
  const SplashTurboInt4PageTensors slab =
      SplashTurboInt4Addressing<KVHeads>(params.kv, head).page(page);
  uint page_token = logical_token % SplashKvPageTokens;
  ulong source_index =
      value_tensor ? splash_current_value_index(params.chunk_stride, head,
                                                  chunk_token, dimension)
                   : splash_current_key_index(params.chunk_stride, head,
                                                chunk_token, dimension);
  staging[dimension] =
      float(value_tensor ? chunk_values[source_index] : chunk_keys[source_index]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group != 0)
    return;
  float x[8];
#pragma unroll
  for (uint r = 0; r < 8; ++r)
    x[r] = staging[32 * r + simd_lane];
  splash_turbo_rotate256(x, simd_lane, false);
  float local_maximum = 0.0f;
#pragma unroll
  for (uint r = 0; r < 8; ++r)
    local_maximum = max(local_maximum, abs(x[r]));
  const float maximum = simd_max(local_maximum);
  const float scale = maximum > 0.0f ? maximum / 7.0f : 0.0f;
  const float inverse = maximum > 0.0f ? 7.0f / maximum : 0.0f;
  device uint *row = reinterpret_cast<device uint *>(
      (value_tensor ? slab.values : slab.keys) + page_token * SplashTurboInt4RowBytes);
#pragma unroll
  for (uint r = 0; r < 8; ++r) {
    const int quantized = clamp(int(rint(x[r] * inverse)), -7, 7);
    uint word = (uint(quantized) & 0xFu) << (4 * (simd_lane & 7u));
    word |= simd_shuffle_xor(word, ushort(1));
    word |= simd_shuffle_xor(word, ushort(2));
    word |= simd_shuffle_xor(word, ushort(4));
    if ((simd_lane & 7u) == 0)
      row[r * 4 + simd_lane / 8] = word;     // elements 32r + 8k .. 32r + 8k + 7
  }
  if (simd_lane == 0)
    (value_tensor ? slab.value_scales : slab.key_scales)[page_token] = scale;
}

// One lane per dimension stores a current row in its final page slot.
// INT8 derives a per-row scale; BF16 copies the original bits; TurboQuant
// quantizes the row as above and needs `maxima` to hold the whole row. Slots
// are addressed inside the head's slab of the page: the page index functions
// at page zero and head zero.
template <uint KVHeads, typename CacheElement>
__attribute__((always_inline)) inline void splash_store_kv_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table,
    constant SplashChunkedPrefillParams &params, threadgroup float *maxima,
    bool value_tensor, uint head, uint chunk_token, uint dimension,
    uint simd_lane, uint simd_group) {
  if constexpr (is_same<CacheElement, SplashTurboKv>::value) {
    splash_store_turbo_row<KVHeads>(chunk_keys, chunk_values, page_table, params, maxima,
                                    value_tensor, head, chunk_token, dimension,
                                    simd_lane, simd_group);
    return;
  } else if constexpr (is_same<CacheElement, SplashTurboInt4Kv>::value) {
    splash_store_turbo_int4_row<KVHeads>(chunk_keys, chunk_values, page_table, params, maxima,
                                         value_tensor, head, chunk_token, dimension,
                                         simd_lane, simd_group);
    return;
  } else {
  uint logical_token = params.committed_tokens + chunk_token;
  const SplashKvPage page = page_table[logical_token / SplashKvPageTokens];
  const SplashKvPageTensors<CacheElement> slab =
      SplashKvAddressing<KVHeads, CacheElement>(params.kv, head).page(page);
  uint page_token = logical_token % SplashKvPageTokens;
  ulong source_index =
      value_tensor ? splash_current_value_index(params.chunk_stride, head,
                                                  chunk_token, dimension)
                   : splash_current_key_index(params.chunk_stride, head,
                                                chunk_token, dimension);
  bfloat source = value_tensor ? chunk_values[source_index] : chunk_keys[source_index];

  if constexpr (is_same<CacheElement, bfloat>::value) {
    if (value_tensor)
      slab.values[splash_kv_value_index<KVHeads>(0, 0, page_token, dimension)] = source;
    else
      slab.keys[splash_kv_key_index<KVHeads>(0, 0, page_token, dimension)] = source;
    return;
  } else {

    float value = float(source);
    float local_maximum = simd_max(abs(value));
    if (simd_lane == 0)
      maxima[simd_group] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      for (uint offset = 4; offset != 0; offset >>= 1) {
        if (simd_lane < offset)
          maxima[simd_lane] = max(maxima[simd_lane], maxima[simd_lane + offset]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = maxima[0];
    float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    int quantized = maximum == 0.0f
                        ? 0
                        : clamp(int(rint(value * 127.0f / maximum)), -127, 127);

    if (value_tensor) {
      slab.values[splash_kv_value_index<KVHeads>(0, 0, page_token, dimension)] =
          char(quantized);
      if (dimension == 0)
        slab.value_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    } else {
      slab.keys[splash_kv_key_index<KVHeads>(0, 0, page_token, dimension)] =
          char(quantized);
      if (dimension == 0)
        slab.key_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    }
  }
  }
}

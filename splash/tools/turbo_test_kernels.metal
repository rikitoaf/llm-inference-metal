// Test-only kernels for tools/turbo_kernel_test.py: the production TurboQuant store and query
// rotation, plus a kernel that runs the attention tile's dequantizers on page 0 and writes the
// tiles out (keys [head][token][dim], values [head][dim][token]).
#include "metal/kernels/prefill/paged_attention_store.metal"
#include "metal/kernels/shared/turbo_rotate.metal"
#include "metal/kernels/decode/paged_attention.metal"
#include "metal/kernels/common/paged_attention_tile.h"

kernel void test_turbo_dump(device const SplashKvPage *page_table [[buffer(0)]],
                            constant SplashKvLayer &kv [[buffer(1)]],
                            device float *keys_out [[buffer(2)]],
                            device float *values_out [[buffer(3)]],
                            uint head [[threadgroup_position_in_grid]],
                            uint thread_index [[thread_index_in_threadgroup]],
                            uint simd_lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Elements = SplashKvPageTokens * SplashKvHeadDimension;
  threadgroup SplashTurboTileElement tile[Elements];
  threadgroup uint4 staging[256 + 8];
  const SplashTurboPageTensors page = SplashTurboAddressing<4>(kv, head).page(page_table[0]);
  splash_turbo_dequantize_keys(page, tile, staging, SplashKvPageTokens, thread_index, simd_lane,
                               simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = thread_index; i < Elements; i += 256)
    keys_out[head * Elements + i] = float(tile[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  splash_turbo_dequantize_values(page, tile, SplashKvPageTokens, simd_lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = thread_index; i < Elements; i += 256)
    values_out[head * Elements + i] = float(tile[i]);
}

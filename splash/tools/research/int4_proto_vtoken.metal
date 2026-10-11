// Speed prototype (not production): the INT8 paged tile with 4-bit pages read by MPP directly.
// Page slab per head: keys [token][256 dims] as int4 (128 B per token), values [dim][32 tokens]
// as int4 (16 B per dim), one fp32 scale per (token, head) for each, like INT8.
#include "metal/kernels/prefill/paged_attention.metal"
#include "metal/kernels/decode/paged_attention.metal"

template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile>
inline void splash_paged_attention_i4_tile(
    device bfloat *tile_queries, device const SplashKvPage *page_table,
    SplashKvLayer kv, uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort N = SplashKvPageTokens;
  constexpr ushort D = SplashKvHeadDimension;
  constexpr uint DataBytes = KVHeads * N * D / 2;          // per tensor per page
  constexpr uint ScaleBytes = KVHeads * N * 4;
  constexpr uint SlabBytes = N * D / 2;
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_attention_pages(visible_tokens);
  uint per_split = splash_attention_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  using Int4Tensor = tensor<device int4b_format, dextents<int, 2>, tensor_inline>;
  auto key_type = Int4Tensor(static_cast<device uchar *>(nullptr), dextents<int, 2>{D, N});
  auto value_type = Int4Tensor(static_cast<device uchar *>(nullptr), dextents<int, 2>{D, N});
  auto q0 = qt.slice<D, M>(0, 0);
  auto k0 = key_type.template slice<D, N>(0, 0);
  auto v0 = value_type.template slice<D, N>(0, 0);
  const SplashKvPageGeometry geometry = {DataBytes, ScaleBytes, DataBytes, ScaleBytes};
  const ulong key_scales_offset = splash_kv_geometry_offset(kv.extent_pages, geometry, 0, SPLASH_KV_KEY_SCALES, 0);
  const ulong values_offset = splash_kv_geometry_offset(kv.extent_pages, geometry, 0, SPLASH_KV_VALUES, 0);
  const ulong value_scales_offset = splash_kv_geometry_offset(kv.extent_pages, geometry, 0, SPLASH_KV_VALUE_SCALES, 0);
  constexpr auto qk_descriptor = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor = matmul2d_descriptor(M, D, N, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();
  const bool running_full = uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
  for (ushort index = 0; index < running.get_capacity(); ++index)
    if (running_full || running.is_valid_element(index)) running[index] = 0.0f;
  for (uint page = page_begin; page < page_end; ++page) {
    const SplashKvPage entry = page_table[page];
    const uint index = splash_kv_page_index(entry);
    device uchar *region = splash_kv_extent(entry, index) + kv.offset;
    const ulong slab = (ulong(index) * KVHeads + kv_head);
    device uchar *keys = region + slab * SlabBytes;
    device uchar *values = region + values_offset + slab * SlabBytes;
    device const float *key_scales = reinterpret_cast<device const float *>(region + key_scales_offset) + slab * N;
    device const float *value_scales = reinterpret_cast<device const float *>(region + value_scales_offset) + slab * N;
    uint token_start = page * N;
    auto kt = Int4Tensor(keys, dextents<int, 2>{D, N});
    auto vt = Int4Tensor(values, dextents<int, 2>{D, N});
    auto page_scores = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    auto ks = kt.template slice<D, N>(0, 0);
    qk.run(q0, ks, page_scores);
    page_scores.store(st.slice<N, M>(0, 0));
    if (thread_index == 0) atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, true>(
        scores, probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales), reinterpret_cast<device const float4 *>(value_scales),
        token_start, visible_tokens, committed_tokens, active_rows, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
      for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!running_full && !running.is_valid_element(i)) continue;
        auto c = running.get_multidimensional_index(i);
        running[i] *= previous_scale[c[1]];
      }
    }
    auto vs = vt.template slice<D, N>(0, 0);
    pv.run(p0, vs, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

kernel void prefill_attention_i4_split(
    device bfloat *queries [[buffer(0)]], device float *partials [[buffer(1)]],
    device float *statistics [[buffer(2)]], device const SplashKvPage *page_table [[buffer(3)]],
    constant SplashPrefillAttentionParams &params [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint M = 6 * SPLASH_PREFILL_ATTENTION_TILE_ROWS, N = SplashKvPageTokens, D = SplashKvHeadDimension;
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup float row_max[M], row_sum[M], previous_scale[M];
  threadgroup atomic_uint rescale;
  uint kv_head = group.x, split = group.z, tile = group.y, tile_start = tile * SplashPrefillTileRows;
  if (!splash_prefill_attention_contract_valid(params) || kv_head >= 4 || split >= params.split_count || tile_start >= params.rows) return;
  uint active_rows = min(SplashPrefillTileRows, params.rows - tile_start);
  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) * 6 * D;
  ulong slot = (ulong(tile) * 4 + kv_head) * params.split_count + split;
  splash_paged_attention_i4_tile<4, 6, SPLASH_PREFILL_ATTENTION_TILE_ROWS>(
      queries + tile_offset, page_table, params.kv, kv_head, params.committed_tokens + tile_start, active_rows,
      params.split_count, split, partials, statistics, slot, scores, probabilities, row_max, row_sum,
      previous_scale, &rescale, thread_index);
}

kernel void verify_attention_i4_split(
    device bfloat *queries [[buffer(0)]], device float *partials [[buffer(1)]],
    device float *statistics [[buffer(2)]], device const SplashKvPage *page_table0 [[buffer(3)]],
    device const SplashKvPage *page_table1 [[buffer(4)]], device const SplashKvPage *page_table2 [[buffer(5)]],
    device const SplashKvPage *page_table3 [[buffer(6)]], constant SplashVerifyAttentionParams *params [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint M = 6 * SPLASH_TARGET_VERIFY_ROWS, N = SplashKvPageTokens;
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup float row_max[M], row_sum[M], previous_scale[M];
  threadgroup atomic_uint rescale;
  const SplashVerifyTile tile = splash_verify_attention_tile_at<4, 6>(queries, page_table0, page_table1, page_table2, page_table3, params, group);
  if (!tile.active) return;
  splash_paged_attention_i4_tile<4, 6, SPLASH_TARGET_VERIFY_ROWS>(
      tile.queries, tile.page_table, params[group.z].kv, tile.kv_head, tile.committed_tokens,
      SPLASH_TARGET_VERIFY_ROWS, tile.splits, tile.split, partials, statistics, tile.slot, scores,
      probabilities, row_max, row_sum, previous_scale, &rescale, thread_index);
}

// Prefill attention variants for prefill_attn_bench (ATTN.md): the production
// kernels plus the SPLASH_PREFILL_ATTENTION_V2 candidates, flattened against a
// runtime tree that has common/prefill_attention_v2.h (splash/dev-prefill-attn).
// Arguments: name, KV heads, group, rows per tile, pages per step, aliased
// probabilities, QK block rows x tokens (0 0 = group-wide), staged queries,
// PV block rows x dims (0 0 = group-wide), ablation (bench only).
#include "metal/kernels/prefill/paged_attention.metal"
#include "metal/kernels/common/prefill_attention_v2.h"

SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p4a_q48x16, 4, 6, 8, 4, true, 48, 16, false, 0, 0, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p4a_q16x16, 4, 6, 8, 4, true, 16, 16, false, 0, 0, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p4a_q24x16, 4, 6, 8, 4, true, 24, 16, false, 0, 0, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p2_q24x16, 4, 6, 8, 2, false, 24, 16, false, 0, 0, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p2a_q16x16, 4, 6, 8, 2, true, 16, 16, false, 0, 0, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p4a_q48x16_v48x32, 4, 6, 8, 4, true, 48, 16, false, 48, 32, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p4a_q48x16_v24x64, 4, 6, 8, 4, true, 48, 16, false, 24, 64, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p2_q16x32_v48x32, 4, 6, 8, 2, false, 16, 32, false, 48, 32, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p1_q16x16_v48x32, 4, 6, 8, 1, false, 16, 16, false, 48, 32, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p1_q16x16_v24x64, 4, 6, 8, 1, false, 16, 16, false, 24, 64, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p2a_q24x16_v48x32, 4, 6, 8, 2, true, 24, 16, false, 48, 32, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p4a_q48x16_noqk, 4, 6, 8, 4, true, 48, 16, false, 0, 0, 3)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p4a_q48x16_v48x32_noqk, 4, 6, 8, 4, true, 48, 16, false, 48, 32, 3)
SPLASH_PREFILL_V2_TURBO_INT4_REDUCE(bench_reduce_r8, 4, 6, 8)
SPLASH_PREFILL_V2_TURBO_INT4_REDUCE(bench_reduce_r16, 4, 6, 16)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p2a_q16x32_v48x32, 4, 6, 8, 2, true, 16, 32, false, 48, 32, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p2a_q16x32, 4, 6, 8, 2, true, 16, 32, false, 0, 0, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p1a_q16x16_v48x32, 4, 6, 8, 1, true, 16, 16, false, 48, 32, 0)
SPLASH_PREFILL_V2_TURBO_INT4_SPLIT(bench_r8_p1_q16x16, 4, 6, 8, 1, false, 16, 16, false, 0, 0, 0)

// Dumps which (dim0, dim1) element each simdgroup lane owns in matmul2d's cooperative operands.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;

template <int M, int N, int K, bool TR>
inline void dump(device int *out, uint lane) {
  constexpr auto d = matmul2d_descriptor(M, N, K, false, TR, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroups<1>> op;
  auto r = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto l = op.template get_left_input_cooperative_tensor<bfloat, half, float>();
  // header per lane: capacity right, capacity left; then right entries (valid, i0, i1), then left entries
  device int *o = out + lane * 1024;
  o[0] = r.get_capacity(); o[1] = l.get_capacity();
  int at = 2;
  for (ushort i = 0; i < r.get_capacity(); ++i) { auto x = r.get_multidimensional_index(i); o[at++] = r.is_valid_element(i); o[at++] = x[0]; o[at++] = x[1]; }
  for (ushort i = 0; i < l.get_capacity(); ++i) { auto x = l.get_multidimensional_index(i); o[at++] = l.is_valid_element(i); o[at++] = x[0]; o[at++] = x[1]; }
}
kernel void layout_16_32_32(device int *out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { dump<16, 32, 32, true>(out, lane); }
kernel void layout_8_32_32(device int *out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { dump<8, 32, 32, true>(out, lane); }
kernel void layout_16_32_32_n(device int *out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) { dump<16, 32, 32, false>(out, lane); }

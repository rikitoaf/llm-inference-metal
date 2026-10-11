#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;
// scores[m][n] = sum_d Q[m][d] * K[n][d] with Q one-hot (row m picks dimension m), K int4 token-major
kernel void int4_probe(device bfloat *q [[buffer(0)]], device uchar *k [[buffer(1)]], device float *out [[buffer(2)]],
                       uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr int M = 48, N = 32, D = 256;
  auto qt = tensor(q, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  using Int4Tensor = tensor<device int4b_format, dextents<int, 2>, tensor_inline>;
  auto kt = Int4Tensor(k, dextents<int, 2>{D, N});
  auto ot = tensor(out, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  constexpr auto desc = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<8>> op;
  auto q0 = qt.slice<D, M>(0, 0);
  auto k0 = kt.slice<D, N>(0, 0);
  auto c = op.get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
  op.run(q0, k0, c);
  auto o0 = ot.slice<N, M>(0, 0);
  c.store(o0);
}

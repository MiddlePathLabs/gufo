#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/projection.hpp"

namespace gufo::hip {
namespace {

constexpr unsigned kProjectThreads = 256;

// One block per row: y[row] -= coef * (y[row] . dir) * dir, with coef the
// writer scale times the row's lambda. The reduction order is fixed (strided
// per-thread sums, then a shared-memory tree), so the result does not depend
// on scheduling. A zero coefficient returns before touching the row, so an
// unsteered row keeps its exact bits even when it holds non-finite values.
__global__ void __launch_bounds__(kProjectThreads)
    QwenProjectRowsKernel(float* __restrict__ y, std::size_t hidden,
                          const float* __restrict__ dir, float scale,
                          float lambda, const float* __restrict__ row_lambda) {
  const std::size_t row = blockIdx.x;
  const float coef = scale * (row_lambda != nullptr ? row_lambda[row] : lambda);
  if (coef == 0.0F)
    return;
  float* const out = y + row * hidden;
  float partial = 0.0F;
  for (std::size_t i = threadIdx.x; i < hidden; i += kProjectThreads)
    partial = __fmaf_rn(out[i], dir[i], partial);
  __shared__ float sums[kProjectThreads];
  sums[threadIdx.x] = partial;
  __syncthreads();
  for (unsigned stride = kProjectThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      sums[threadIdx.x] = __fadd_rn(sums[threadIdx.x], sums[threadIdx.x + stride]);
    __syncthreads();
  }
  const float amount = coef * sums[0];
  for (std::size_t i = threadIdx.x; i < hidden; i += kProjectThreads)
    out[i] = __fmaf_rn(-amount, dir[i], out[i]);
}

}  // namespace

void LaunchQwenProjectRows(float* y, std::size_t rows, std::size_t hidden,
                           const float* dir, float scale, float lambda,
                           const float* row_lambda, hipStream_t stream) {
  if (rows == 0 || dir == nullptr || scale == 0.0F ||
      (row_lambda == nullptr && lambda == 0.0F))
    return;
  QwenProjectRowsKernel<<<static_cast<unsigned>(rows), kProjectThreads, 0,
                          stream>>>(y, hidden, dir, scale, lambda, row_lambda);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

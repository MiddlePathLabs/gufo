#include "src/models/qwen/hip/projection.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <utility>

namespace gufo::hip {

QwenGpuProjection::~QwenGpuProjection() {
  if (d_directions_ != nullptr)
    (void)hipFree(d_directions_);
}

std::shared_ptr<const QwenGpuProjection> QwenGpuProjection::Create(
    models::qwen::QwenProjection projection, std::string* error_msg) {
  if (projection.Empty()) {
    if (error_msg != nullptr)
      *error_msg = "projection has no steered writer";
    return nullptr;
  }
  std::shared_ptr<QwenGpuProjection> result(new QwenGpuProjection());
  const std::size_t bytes = projection.directions.size() * sizeof(float);
  if (hipMalloc(reinterpret_cast<void**>(&result->d_directions_), bytes) !=
          hipSuccess ||
      hipMemcpy(result->d_directions_, projection.directions.data(), bytes,
                hipMemcpyHostToDevice) != hipSuccess) {
    if (error_msg != nullptr)
      *error_msg = "cannot upload the projection directions to the device";
    return nullptr;
  }
  result->fingerprint_ = models::qwen::QwenProjectionFingerprint(projection);
  result->host_ = std::move(projection);
  return result;
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

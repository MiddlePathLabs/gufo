#ifndef GUFO_MODELS_QWEN_HIP_PROJECTION_HPP_
#define GUFO_MODELS_QWEN_HIP_PROJECTION_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/models/qwen/projection.hpp"

namespace gufo::hip {

/// y[r] -= scale * lambda_r * (y[r] . dir) * dir for each of `rows` rows of
/// `hidden` floats, in place. `lambda_r` is `row_lambda[r]` when that device
/// array is given, otherwise `lambda`. Rows whose coefficient is zero are not
/// touched; nothing launches when every coefficient is known to be zero.
void LaunchQwenProjectRows(float* y, std::size_t rows, std::size_t hidden,
                           const float* dir, float scale, float lambda,
                           const float* row_lambda, hipStream_t stream);

/// A loaded models::qwen::QwenProjection with its directions resident on the
/// device, shared read-only by every session of a runner.
class QwenGpuProjection {
public:
  ~QwenGpuProjection();
  QwenGpuProjection(const QwenGpuProjection&) = delete;
  QwenGpuProjection& operator=(const QwenGpuProjection&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenGpuProjection> Create(
      models::qwen::QwenProjection projection, std::string* error_msg);

  /// The writer's unit direction on the device, or null when unsteered.
  [[nodiscard]] const float* Direction(
      std::uint32_t layer, models::qwen::QwenProjectionSite site) const {
    return host_.Scale(layer, site) == 0.0F
               ? nullptr
               : d_directions_ + host_.Offset(layer, site);
  }
  [[nodiscard]] float Scale(std::uint32_t layer,
                            models::qwen::QwenProjectionSite site) const {
    return host_.Scale(layer, site);
  }
  [[nodiscard]] std::size_t Hidden() const noexcept { return host_.hidden; }
  [[nodiscard]] const models::qwen::QwenProjection& Host() const noexcept {
    return host_;
  }
  [[nodiscard]] const std::string& Fingerprint() const noexcept {
    return fingerprint_;
  }

  /// Projects `rows` rows of the writer's output `y` (see
  /// LaunchQwenProjectRows); a no-op for an unsteered writer.
  void Apply(std::uint32_t layer, models::qwen::QwenProjectionSite site,
             float* y, std::size_t rows, float lambda, const float* row_lambda,
             hipStream_t stream) const {
    LaunchQwenProjectRows(y, rows, host_.hidden, Direction(layer, site),
                          Scale(layer, site), lambda, row_lambda, stream);
  }

private:
  QwenGpuProjection() = default;
  models::qwen::QwenProjection host_;
  std::string fingerprint_;
  float* d_directions_{nullptr};
};

/// The projection one forward pass applies: the shared directions and either
/// one lambda for every row or a device array of per-row lambdas.
struct QwenPassProjection {
  const QwenGpuProjection* gpu{nullptr};
  float lambda{0.0F};
  const float* row_lambda{nullptr};

  [[nodiscard]] bool Active() const noexcept {
    return gpu != nullptr && (row_lambda != nullptr || lambda != 0.0F);
  }
  void Apply(std::uint32_t layer, models::qwen::QwenProjectionSite site,
             float* y, std::size_t rows, hipStream_t stream) const {
    if (Active())
      gpu->Apply(layer, site, y, rows, lambda, row_lambda, stream);
  }
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_PROJECTION_HPP_

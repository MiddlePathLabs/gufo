#ifndef GUFO_MODELS_QWEN_PROJECTION_HPP_
#define GUFO_MODELS_QWEN_PROJECTION_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen {

/// The two residual writers of a Qwen3.8-27B decoder layer: the mixer output
/// projection (`attn_output` on full-attention layers, `ssm_out` on the
/// linear-attention ones) and the FFN down projection.
enum class QwenProjectionSite : std::uint32_t { kMixer = 0, kFfn = 1 };
inline constexpr std::uint32_t kQwenProjectionSites = 2;

/// A runtime rank-1 output projection read from a GGUF direction file. For
/// every steered writer the sublayer output `y` is replaced, before its
/// residual add, by `y - lambda * s * (y . v) * v`: `v` is the writer's unit
/// direction and `s` its scale (the norm of the stored tensor, the
/// per-module coefficient of the source ablation). This equals editing the
/// writer's weight to `(I - lambda * s * v v^T) W` without touching the
/// GGUF. Lambda zero leaves the engine on the unmodified arithmetic.
struct QwenProjection {
  /// Unit directions, `[num_layers][kQwenProjectionSites][hidden]`; an
  /// absent writer keeps a zero row and a zero scale.
  std::vector<float> directions;
  /// Per-writer scale `s`, `[num_layers][kQwenProjectionSites]`. Zero leaves
  /// that writer unsteered.
  std::vector<float> scales;
  std::uint32_t num_layers = 0;
  std::uint32_t hidden = 0;
  /// Inclusive steered layer range.
  std::uint32_t first_layer = 0;
  std::uint32_t last_layer = 0;
  /// Number of writers with a nonzero scale.
  std::uint32_t steered = 0;

  [[nodiscard]] bool Empty() const noexcept { return steered == 0; }
  [[nodiscard]] float Scale(std::uint32_t layer,
                            QwenProjectionSite site) const noexcept {
    if (layer < first_layer || layer > last_layer || layer >= num_layers)
      return 0.0F;
    return scales[static_cast<std::size_t>(layer) * kQwenProjectionSites +
                  static_cast<std::uint32_t>(site)];
  }
  [[nodiscard]] std::size_t Offset(std::uint32_t layer,
                                   QwenProjectionSite site) const noexcept {
    return (static_cast<std::size_t>(layer) * kQwenProjectionSites +
            static_cast<std::uint32_t>(site)) *
           hidden;
  }
};

/// Parses a direction GGUF with F32 `[hidden]` tensors named
/// `projection.mixer.N` and `projection.ffn.N` (layers `0..L-1`). Each
/// tensor is normalized to unit length and its norm becomes the writer's
/// scale. Unknown `projection.*` tensors, layers past the model and zero
/// tensors are ignored with a warning; non-finite values, a wrong element
/// count or type, or a file with no usable tensor fail the load. A
/// `projection.model_hint` that differs from `expected_hint` warns.
[[nodiscard]] bool LoadQwenProjection(const std::string& path,
                                      std::uint32_t num_layers,
                                      std::uint32_t hidden,
                                      std::string_view expected_hint,
                                      QwenProjection* out,
                                      std::string* error_msg,
                                      std::string* warning = nullptr);

/// Restricts a loaded projection to the inclusive layer range `spec`
/// ("LO-HI", 0 <= LO <= HI <= num_layers - 1). An empty spec keeps every
/// layer. Malformed or out-of-range specs are rejected, never clamped.
[[nodiscard]] bool SelectQwenProjectionLayers(std::string_view spec,
                                              QwenProjection* projection,
                                              std::string* error_msg);

/// SHA-256 over the shape, steered range, scales and directions.
[[nodiscard]] std::string QwenProjectionFingerprint(
    const QwenProjection& projection);

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_PROJECTION_HPP_

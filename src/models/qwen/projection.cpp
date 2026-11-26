#include "src/models/qwen/projection.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <span>
#include <string>
#include <system_error>

#include "gufo/ext/gguf_reader.hpp"
#include "gufo/ext/sha256.hpp"

namespace gufo::models::qwen {
namespace {

constexpr std::string_view kPrefix = "projection.";
constexpr std::string_view kSiteNames[kQwenProjectionSites] = {"mixer", "ffn"};

bool Fail(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr)
    *error_msg = message;
  return false;
}

void Warn(std::string* warning, const std::string& message) {
  if (warning == nullptr)
    return;
  if (!warning->empty())
    *warning += "; ";
  *warning += message;
}

bool ParseUint(std::string_view text, std::uint32_t* value) {
  const auto* end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(text.data(), end, *value);
  return !text.empty() && ec == std::errc{} && ptr == end;
}

template<class T>
void HashValues(crypto::Sha256Hasher& hasher, std::span<const T> values) {
  hasher.Update({reinterpret_cast<const std::uint8_t*>(values.data()),
                 values.size_bytes()});
}

}  // namespace

bool LoadQwenProjection(const std::string& path, std::uint32_t num_layers,
                        std::uint32_t hidden, std::string_view expected_hint,
                        QwenProjection* out, std::string* error_msg,
                        std::string* warning) {
  if (out == nullptr)
    return Fail(error_msg, "projection output is null");
  if (num_layers == 0 || hidden == 0)
    return Fail(error_msg, "projection needs a model with layers");
  std::string open_error;
  auto reader = core::GgufReader::OpenFile(path, &open_error);
  if (reader == nullptr)
    return Fail(error_msg,
                "cannot open projection vector " + path + ": " + open_error);
  if (const auto hint = reader->GetMetadataString("projection.model_hint");
      hint.has_value() && *hint != expected_hint) {
    Warn(warning, "projection vector " + path + " declares model_hint '" +
                      std::string(*hint) + "' but this model is '" +
                      std::string(expected_hint) + "'");
  }

  QwenProjection result;
  result.num_layers = num_layers;
  result.hidden = hidden;
  result.first_layer = 0;
  result.last_layer = num_layers - 1;
  result.directions.assign(
      static_cast<std::size_t>(num_layers) * kQwenProjectionSites * hidden,
      0.0F);
  result.scales.assign(
      static_cast<std::size_t>(num_layers) * kQwenProjectionSites, 0.0F);

  for (const auto& tensor : reader->GetTensors()) {
    if (!tensor.name.starts_with(kPrefix))
      continue;
    const std::string name(tensor.name);
    const std::string_view rest = tensor.name.substr(kPrefix.size());
    const auto dot = rest.find('.');
    std::uint32_t site = kQwenProjectionSites;
    std::uint32_t layer = 0;
    if (dot != std::string_view::npos) {
      for (std::uint32_t s = 0; s < kQwenProjectionSites; ++s)
        if (rest.substr(0, dot) == kSiteNames[s])
          site = s;
    }
    if (site == kQwenProjectionSites ||
        !ParseUint(rest.substr(dot + 1), &layer)) {
      Warn(warning, "ignoring " + name +
                        ": expected projection.mixer.N or projection.ffn.N");
      continue;
    }
    if (layer >= num_layers) {
      Warn(warning, "ignoring " + name + ": this model has layers 0.." +
                        std::to_string(num_layers - 1));
      continue;
    }
    if (tensor.type != core::GgmlType::kF32 || tensor.ElementCount() != hidden)
      return Fail(error_msg, name + " must be F32 with " +
                                 std::to_string(hidden) + " elements");
    const std::size_t index =
        static_cast<std::size_t>(layer) * kQwenProjectionSites + site;
    float* row = result.directions.data() + index * hidden;
    // GGUF aligns tensor data only to general.alignment; copy before reading
    // it as floats.
    std::memcpy(row, tensor.data, static_cast<std::size_t>(hidden) * 4);
    if (!std::all_of(row, row + hidden,
                     [](float x) { return std::isfinite(x); }))
      return Fail(error_msg, name + " contains non-finite values");
    double square_norm = 0.0;
    for (std::uint32_t i = 0; i < hidden; ++i)
      square_norm += static_cast<double>(row[i]) * static_cast<double>(row[i]);
    const double norm = std::sqrt(square_norm);
    if (norm == 0.0) {
      Warn(warning, "ignoring " + name + ": zero direction");
      continue;
    }
    const auto scale = static_cast<float>(norm);
    if (!std::isfinite(scale))
      return Fail(error_msg, name + " has a norm beyond the F32 range");
    for (std::uint32_t i = 0; i < hidden; ++i)
      row[i] = static_cast<float>(static_cast<double>(row[i]) / norm);
    if (result.scales[index] == 0.0F)
      ++result.steered;
    result.scales[index] = scale;
  }
  if (result.steered == 0)
    return Fail(error_msg, "projection vector " + path +
                               " carries no usable projection.* tensors");
  *out = std::move(result);
  return true;
}

bool SelectQwenProjectionLayers(std::string_view spec,
                                QwenProjection* projection,
                                std::string* error_msg) {
  if (projection == nullptr || projection->Empty())
    return Fail(error_msg, "no projection vector is loaded");
  const std::uint32_t max_layer = projection->num_layers - 1;
  if (spec.empty()) {
    projection->first_layer = 0;
    projection->last_layer = max_layer;
    return true;
  }
  const auto dash = spec.find('-');
  std::uint32_t lo = 0;
  std::uint32_t hi = 0;
  if (dash == std::string_view::npos || !ParseUint(spec.substr(0, dash), &lo) ||
      !ParseUint(spec.substr(dash + 1), &hi))
    return Fail(error_msg,
                "projection layers '" + std::string(spec) + "' must be LO-HI");
  if (lo > hi || hi > max_layer)
    return Fail(error_msg, "projection layers '" + std::string(spec) +
                               "' must satisfy 0 <= LO <= HI <= " +
                               std::to_string(max_layer));
  projection->first_layer = lo;
  projection->last_layer = hi;
  return true;
}

std::string QwenProjectionFingerprint(const QwenProjection& projection) {
  crypto::Sha256Hasher hasher;
  const std::uint32_t shape[] = {projection.num_layers, projection.hidden,
                                 projection.first_layer, projection.last_layer,
                                 kQwenProjectionSites};
  HashValues(hasher, std::span<const std::uint32_t>(shape));
  HashValues(hasher, std::span<const float>(projection.scales));
  HashValues(hasher, std::span<const float>(projection.directions));
  return hasher.FinishHex();
}

}  // namespace gufo::models::qwen

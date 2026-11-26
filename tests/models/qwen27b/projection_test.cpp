// Qwen3.8-27B runtime projection: the direction-file loader and, on a GPU
// build, the row projection kernel against a double-precision reference.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "src/models/qwen/projection.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/projection.hpp"
#endif

namespace qwen = gufo::models::qwen;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

void Put(std::string* out, const void* data, std::size_t bytes) {
  out->append(static_cast<const char*>(data), bytes);
}
void PutU32(std::string* out, std::uint32_t v) { Put(out, &v, 4); }
void PutU64(std::string* out, std::uint64_t v) { Put(out, &v, 8); }
void PutString(std::string* out, const std::string& s) {
  PutU64(out, s.size());
  Put(out, s.data(), s.size());
}

struct TensorSpec {
  std::string name;
  std::vector<float> data;
  std::uint32_t ggml_type = 0;  // GGML_F32
};

// Minimal GGUF v3: 1-D tensors at 32-byte alignment, optional model hint.
std::string WriteGguf(const std::vector<TensorSpec>& tensors,
                      const std::string& model_hint = {}) {
  std::string data;
  std::string infos;
  for (const auto& t : tensors) {
    data.resize((data.size() + 31) / 32 * 32, '\0');
    const std::uint64_t offset = data.size();
    PutString(&infos, t.name);
    PutU32(&infos, 1);
    PutU64(&infos, t.data.size());
    PutU32(&infos, t.ggml_type);
    PutU64(&infos, offset);
    Put(&data, t.data.data(), t.data.size() * sizeof(float));
  }
  std::string out;
  Put(&out, "GGUF", 4);
  PutU32(&out, 3);
  PutU64(&out, tensors.size());
  PutU64(&out, model_hint.empty() ? 0 : 1);
  if (!model_hint.empty()) {
    PutString(&out, "projection.model_hint");
    PutU32(&out, 8);
    PutString(&out, model_hint);
  }
  out += infos;
  out.resize((out.size() + 31) / 32 * 32, '\0');
  out += data;
  return out;
}

const std::filesystem::path& TempDir() {
  static const std::filesystem::path dir = [] {
    auto path = std::filesystem::temp_directory_path() /
                ("gufo_qwen27b_projection_test_" +
                 std::to_string(std::random_device{}()));
    std::filesystem::create_directories(path);
    return path;
  }();
  return dir;
}

std::string TempFile(const std::string& blob, const char* tag) {
  const auto path = TempDir() / (std::string(tag) + ".gguf");
  std::ofstream file(path, std::ios::binary);
  file.write(blob.data(), static_cast<std::streamsize>(blob.size()));
  return path.string();
}

constexpr std::uint32_t kLayers = 4;
constexpr std::uint32_t kHidden = 64;

std::vector<float> Axis(std::uint32_t index, float value) {
  std::vector<float> v(kHidden, 0.0F);
  v[index] = value;
  return v;
}

void TestLoad() {
  const auto path = TempFile(
      WriteGguf({{"projection.mixer.0", Axis(0, 1.25F)},
                 {"projection.ffn.0", Axis(1, -1.0F)},
                 {"projection.ffn.3", Axis(2, 2.0F)},
                 {"projection.ffn.9", Axis(3, 1.0F)},     // past the model
                 {"projection.other.1", Axis(4, 1.0F)},   // unknown site
                 {"projection.mixer.2", std::vector<float>(kHidden, 0.0F)}},
                "qwen35"),
      "load");
  qwen::QwenProjection p;
  std::string error;
  std::string warning;
  Check(qwen::LoadQwenProjection(path, kLayers, kHidden, "qwen35", &p, &error,
                                 &warning),
        "load succeeds: " + error);
  Check(p.steered == 3, "three steered writers");
  Check(p.Scale(0, qwen::QwenProjectionSite::kMixer) == 1.25F,
        "the norm is the mixer scale");
  Check(p.Scale(0, qwen::QwenProjectionSite::kFfn) == 1.0F, "ffn scale 1");
  Check(p.Scale(3, qwen::QwenProjectionSite::kFfn) == 2.0F, "ffn.3 scale 2");
  Check(p.Scale(2, qwen::QwenProjectionSite::kMixer) == 0.0F,
        "a zero tensor stays unsteered");
  Check(p.directions[p.Offset(0, qwen::QwenProjectionSite::kFfn) + 1] == -1.0F,
        "directions are normalized with their sign");
  Check(p.directions[p.Offset(3, qwen::QwenProjectionSite::kFfn) + 2] == 1.0F,
        "ffn.3 is a unit axis");
  Check(warning.find("projection.ffn.9") != std::string::npos,
        "a layer past the model warns");
  Check(warning.find("projection.other.1") != std::string::npos,
        "an unknown site warns");
  Check(warning.find("zero direction") != std::string::npos,
        "a zero direction warns");
  Check(warning.find("model_hint") == std::string::npos,
        "a matching hint is silent");

  const std::string fingerprint = qwen::QwenProjectionFingerprint(p);
  Check(qwen::SelectQwenProjectionLayers("1-3", &p, &error), "range 1-3");
  Check(p.Scale(0, qwen::QwenProjectionSite::kMixer) == 0.0F,
        "layers outside the range are unsteered");
  Check(p.Scale(3, qwen::QwenProjectionSite::kFfn) == 2.0F,
        "layers inside the range keep their scale");
  Check(qwen::QwenProjectionFingerprint(p) != fingerprint,
        "the range enters the fingerprint");
  Check(!qwen::SelectQwenProjectionLayers("2-1", &p, &error), "LO > HI");
  Check(!qwen::SelectQwenProjectionLayers("0-4", &p, &error), "HI too big");
  Check(!qwen::SelectQwenProjectionLayers("x", &p, &error), "malformed");
  Check(qwen::SelectQwenProjectionLayers("", &p, &error), "empty resets");
  Check(qwen::QwenProjectionFingerprint(p) == fingerprint,
        "the full range restores the fingerprint");
}

void TestRejects() {
  qwen::QwenProjection p;
  std::string error;
  std::string warning;
  auto nan = Axis(0, 1.0F);
  nan[5] = std::nanf("");
  Check(!qwen::LoadQwenProjection(
            TempFile(WriteGguf({{"projection.ffn.1", nan}}), "nan"), kLayers,
            kHidden, "qwen35", &p, &error),
        "non-finite values fail");
  Check(!qwen::LoadQwenProjection(
            TempFile(WriteGguf({{"projection.ffn.1",
                                 std::vector<float>(kHidden + 1, 1.0F)}}),
                     "width"),
            kLayers, kHidden, "qwen35", &p, &error),
        "a wrong width fails");
  Check(!qwen::LoadQwenProjection(
            TempFile(WriteGguf({{"direction.1", Axis(0, 1.0F)}}), "none"),
            kLayers, kHidden, "qwen35", &p, &error),
        "a file without projection tensors fails");
  Check(qwen::LoadQwenProjection(
            TempFile(WriteGguf({{"projection.ffn.1", Axis(0, 1.0F)}}, "other"),
                     "hint"),
            kLayers, kHidden, "qwen35", &p, &error, &warning) &&
            warning.find("model_hint") != std::string::npos,
        "a mismatching hint warns and loads");
}

#if defined(ENGINE_ENABLE_HIP)
bool HipOk(hipError_t e) { return e == hipSuccess; }

void TestKernel() {
  int devices = 0;
  if (!HipOk(hipGetDeviceCount(&devices)) || devices == 0) {
    std::printf("SKIP: no HIP device for the kernel check\n");
    return;
  }
  constexpr std::size_t kRows = 5;
  constexpr std::size_t kWidth = 5120;
  std::mt19937 rng(7);
  std::normal_distribution<float> normal;
  std::vector<float> dir(kWidth);
  double square = 0.0;
  for (auto& x : dir) {
    x = normal(rng);
    square += static_cast<double>(x) * x;
  }
  for (auto& x : dir)
    x = static_cast<float>(x / std::sqrt(square));
  std::vector<float> y(kRows * kWidth);
  for (auto& x : y)
    x = normal(rng) * 4.0F;
  y[4 * kWidth + 7] = std::nanf("");  // an unsteered row keeps even a NaN
  const float scale = 1.125F;
  const std::vector<float> lambdas = {1.0F, 0.0F, 0.5F, 2.0F, 0.0F};

  float* d_y = nullptr;
  float* d_dir = nullptr;
  float* d_lambda = nullptr;
  const std::size_t bytes = y.size() * sizeof(float);
  Check(HipOk(hipMalloc(&d_y, bytes)) &&
            HipOk(hipMalloc(&d_dir, kWidth * sizeof(float))) &&
            HipOk(hipMalloc(&d_lambda, kRows * sizeof(float))),
        "device allocation");
  hipMemcpy(d_dir, dir.data(), kWidth * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_lambda, lambdas.data(), kRows * sizeof(float),
            hipMemcpyHostToDevice);

  // Per-row lambdas.
  hipMemcpy(d_y, y.data(), bytes, hipMemcpyHostToDevice);
  gufo::hip::LaunchQwenProjectRows(d_y, kRows, kWidth, d_dir, scale, 0.0F,
                                   d_lambda, nullptr);
  std::vector<float> out(y.size());
  Check(HipOk(hipMemcpy(out.data(), d_y, bytes, hipMemcpyDeviceToHost)),
        "kernel ran");
  for (std::size_t r = 0; r < kRows; ++r) {
    const float* in = y.data() + r * kWidth;
    const float* got = out.data() + r * kWidth;
    if (lambdas[r] == 0.0F) {
      Check(std::memcmp(in, got, kWidth * sizeof(float)) == 0,
            "row " + std::to_string(r) + " with lambda 0 is bit-identical");
      continue;
    }
    double dot = 0.0;
    for (std::size_t i = 0; i < kWidth; ++i)
      dot += static_cast<double>(in[i]) * dir[i];
    const double amount = static_cast<double>(scale) * lambdas[r] * dot;
    double max_error = 0.0;
    double residual = 0.0;
    for (std::size_t i = 0; i < kWidth; ++i) {
      const double want = in[i] - amount * dir[i];
      max_error = std::max(max_error, std::abs(want - got[i]));
      residual += static_cast<double>(got[i]) * dir[i];
    }
    Check(max_error < 1e-4, "row " + std::to_string(r) +
                                " matches the reference: " +
                                std::to_string(max_error));
    // With coef = 1 the component is removed: (1 - coef) * dot remains.
    const double expect_residual = (1.0 - scale * lambdas[r]) * dot;
    Check(std::abs(residual - expect_residual) < 1e-3 * (1 + std::abs(dot)),
          "row " + std::to_string(r) + " keeps (1 - coef) of the component");
  }

  // One scalar lambda for every row equals the same per-row lambdas.
  hipMemcpy(d_y, y.data(), kWidth * sizeof(float), hipMemcpyHostToDevice);
  gufo::hip::LaunchQwenProjectRows(d_y, 1, kWidth, d_dir, scale, 1.0F, nullptr,
                                   nullptr);
  std::vector<float> scalar(kWidth);
  hipMemcpy(scalar.data(), d_y, kWidth * sizeof(float), hipMemcpyDeviceToHost);
  Check(std::memcmp(scalar.data(), out.data(), kWidth * sizeof(float)) == 0,
        "scalar and per-row lambdas are bit-identical");

  hipFree(d_y);
  hipFree(d_dir);
  hipFree(d_lambda);
}
#endif

}  // namespace

int main() {
  TestLoad();
  TestRejects();
#if defined(ENGINE_ENABLE_HIP)
  TestKernel();
#endif
  std::filesystem::remove_all(TempDir());
  if (failures != 0) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("qwen27b projection: all checks passed\n");
  return 0;
}

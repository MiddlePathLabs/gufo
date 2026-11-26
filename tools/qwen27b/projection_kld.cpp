// Teacher-forced full-vocabulary comparison of the Qwen3.8-27B runtime
// projection. For each fixture the same token prefix is decoded token by
// token three times: without a projection, with the projection loaded at
// lambda 0, and at lambda 1. It reports
//   gate:      lambda 0 logits byte-identical to no projection,
//   steering:  KL(off || on), total variation and top-1 agreement, and
//   prefill:   last-row logits of one prompt pass against the token-by-token
//              chain, steered and unsteered, so a prefill site that missed
//              the projection would show up as an on/off-sized divergence.
//
//   qwen27b_projection_kld MODEL.gguf PROJECTION.gguf [OUT.json]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/projection.hpp"

namespace {
using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;

struct Fixture {
  const char* name;
  const char* kind;
  std::string text;
};

std::string Chat(const std::string& user, const std::string& answer) {
  return "<|im_start|>user\n" + user +
         "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" + answer;
}

std::vector<Fixture> Fixtures() {
  return {
      {"virtual-memory", "prose",
       "Virtual memory lets an operating system give each process its own "
       "address space. Pages map virtual addresses to physical memory. A page "
       "fault occurs when the requested page is unavailable and the operating "
       "system must load it from disk, possibly evicting another page first."},
      {"ring-buffer", "code",
       "template <typename T> class RingBuffer { public: bool push(const T& "
       "value) { if (count == capacity) return false; data[tail] = value; "
       "tail = (tail + 1) % capacity; ++count; return true; } bool pop(T& out) "
       "{ if (count == 0) return false; out = data[head]; head = (head + 1) % "
       "capacity; --count; return true; } };"},
      {"trains", "math",
       "A train leaves the station at noon and travels at sixty kilometers per "
       "hour. Another train leaves one hour later at ninety kilometers per "
       "hour. After t hours the first has gone 60t km and the second 90(t-1) "
       "km, so they meet when 60t = 90t - 90, that is t = 3, at 3 pm."},
      {"chat-italian", "chat",
       Chat("Translate to Italian: the weather is lovely and I want a walk.",
            "Il tempo è bellissimo e voglio fare una passeggiata.")},
      {"chat-recipe", "chat",
       Chat("Give me a quick recipe for pancakes.",
            "Mix 200 g of flour, 2 eggs, 300 ml of milk and a pinch of salt "
            "into a smooth batter. Heat a buttered pan, pour a ladle of batter, "
            "cook until bubbles form, flip and cook one more minute.")},
      {"chat-lock", "refusal",
       Chat("Explain step by step how to pick a standard pin tumbler lock.",
            "I can't help with that.")},
  };
}

std::vector<float> Logits(Executor& executor) {
  const auto view = executor.CopyLastLogits();
  return {view.begin(), view.end()};
}

/// Token-by-token decode logits for positions 1..n-1 (row i predicts i+1).
std::vector<std::vector<float>> Chain(Executor& executor,
                                      std::span<const Token> tokens) {
  executor.Reset();
  std::vector<std::vector<float>> rows;
  rows.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    (void)executor.ForwardToken(tokens[i], static_cast<std::uint32_t>(i));
    rows.push_back(Logits(executor));
  }
  return rows;
}

std::vector<double> Softmax(std::span<const float> logits) {
  const float max = *std::ranges::max_element(logits);
  std::vector<double> p(logits.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i)
    sum += p[i] = std::exp(static_cast<double>(logits[i]) - max);
  for (auto& x : p)
    x /= sum;
  return p;
}

struct Divergence {
  double kl = 0.0;
  double tv = 0.0;
  bool top1 = true;
};

Divergence Compare(std::span<const float> ref, std::span<const float> cand) {
  const auto p = Softmax(ref);
  const auto q = Softmax(cand);
  Divergence d;
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (p[i] > 0.0)
      d.kl += p[i] * (std::log(p[i]) - std::log(std::max(q[i], 1e-300)));
    d.tv += std::abs(p[i] - q[i]);
  }
  d.tv *= 0.5;
  d.top1 = std::ranges::max_element(ref) - ref.begin() ==
           std::ranges::max_element(cand) - cand.begin();
  return d;
}

double Percentile(std::vector<double> v, double q) {
  std::ranges::sort(v);
  return v[static_cast<std::size_t>(q * static_cast<double>(v.size() - 1))];
}

std::string Stats(const std::vector<Divergence>& rows) {
  std::vector<double> kl;
  double tv = 0.0;
  std::size_t top1 = 0;
  for (const auto& d : rows) {
    kl.push_back(d.kl);
    tv += d.tv;
    top1 += d.top1;
  }
  const double mean = std::accumulate(kl.begin(), kl.end(), 0.0) /
                      static_cast<double>(kl.size());
  char buffer[256];
  std::snprintf(buffer, sizeof(buffer),
                "{\"rows\": %zu, \"kl_mean\": %.6g, \"kl_median\": %.6g, "
                "\"kl_p95\": %.6g, \"kl_max\": %.6g, \"tv_mean\": %.6g, "
                "\"top1\": %zu}",
                rows.size(), mean, Percentile(kl, 0.5), Percentile(kl, 0.95),
                *std::ranges::max_element(kl),
                tv / static_cast<double>(rows.size()), top1);
  return buffer;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " MODEL.gguf PROJECTION.gguf [OUT.json]\n";
    return 2;
  }
  std::string error;
  std::shared_ptr<const gufo::core::GgufReader> reader =
      gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::cerr << error << "\n";
    return 1;
  }
  auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
  if (!model) {
    std::cerr << error << "\n";
    return 1;
  }
  const auto& config = model->GetConfig();
  gufo::models::qwen::QwenProjection host;
  std::string warning;
  if (!gufo::models::qwen::LoadQwenProjection(argv[2], config.num_layers,
                                              config.hidden_size, "qwen35",
                                              &host, &error, &warning)) {
    std::cerr << error << "\n";
    return 1;
  }
  auto projection = gufo::hip::QwenGpuProjection::Create(std::move(host), &error);
  if (!projection) {
    std::cerr << error << "\n";
    return 1;
  }
  auto executor = Executor::Create(model, &error, 4096);
  if (!executor) {
    std::cerr << error << "\n";
    return 1;
  }
  const auto& tokenizer = executor->GetTokenizer();

  std::string json = "{\n  \"schema\": \"gufo-qwen27b-projection-kld/1\",\n";
  json += "  \"projection_sha256\": \"" + projection->Fingerprint() + "\",\n";
  json += "  \"fixtures\": [\n";
  bool gate_ok = true;
  std::vector<Divergence> by_kind_all;
  std::vector<Divergence> benign;
  const auto fixtures = Fixtures();
  for (std::size_t f = 0; f < fixtures.size(); ++f) {
    const auto& fixture = fixtures[f];
    const auto tokens = tokenizer.Encode(fixture.text);
    executor->SetProjection(nullptr);
    const auto base = Chain(*executor, tokens);
    executor->SetProjection(projection);
    executor->SetProjectionLambda(0.0F);
    const auto zero = Chain(*executor, tokens);
    executor->SetProjectionLambda(1.0F);
    const auto on = Chain(*executor, tokens);
    executor->SetProjectionLambda(0.0F);
    bool identical = true;
    std::vector<Divergence> rows;
    for (std::size_t i = 0; i < base.size(); ++i) {
      identical &= std::memcmp(base[i].data(), zero[i].data(),
                               base[i].size() * sizeof(float)) == 0;
      rows.push_back(Compare(base[i], on[i]));
    }
    gate_ok &= identical;
    if (std::string(fixture.kind) != "refusal")
      benign.insert(benign.end(), rows.begin(), rows.end());
    json += std::string("    {\"name\": \"") + fixture.name +
            "\", \"kind\": \"" + fixture.kind +
            "\", \"tokens\": " + std::to_string(tokens.size()) +
            ", \"lambda0_identical\": " + (identical ? "true" : "false") +
            ", \"off_vs_on\": " + Stats(rows) + "}" +
            (f + 1 == fixtures.size() ? "\n" : ",\n");
    std::cout << fixture.name << " tokens=" << tokens.size()
              << " lambda0_identical=" << identical << " " << Stats(rows)
              << std::endl;
  }
  json += "  ],\n  \"benign_off_vs_on\": " + Stats(benign) + ",\n";

  // Prefill against decode: the same 64- and 1100-token prompts, last row.
  std::string base_text;
  for (const auto& fixture : fixtures)
    if (std::string(fixture.kind) != "refusal")
      base_text += fixture.text + "\n";
  const auto corpus = tokenizer.Encode(base_text);
  json += "  \"prefill_vs_decode\": [\n";
  const std::size_t lengths[] = {64, 1100};
  for (std::size_t li = 0; li < 2; ++li) {
    std::vector<Token> tokens(lengths[li]);
    for (std::size_t i = 0; i < tokens.size(); ++i)
      tokens[i] = corpus[i % corpus.size()];
    std::string row = "    {\"tokens\": " + std::to_string(tokens.size());
    for (const float lambda : {0.0F, 1.0F}) {
      executor->SetProjection(projection);
      executor->SetProjectionLambda(lambda);
      executor->Reset();
      (void)executor->ForwardPromptBatch(tokens);
      const auto prefill = Logits(*executor);
      const auto chain = Chain(*executor, tokens).back();
      const auto d = Compare(chain, prefill);
      char buffer[160];
      std::snprintf(buffer, sizeof(buffer),
                    ", \"%s\": {\"kl\": %.6g, \"tv\": %.6g, \"top1\": %s}",
                    lambda == 0.0F ? "off" : "on", d.kl, d.tv,
                    d.top1 ? "true" : "false");
      row += buffer;
      std::cout << "prefill-vs-decode tokens=" << tokens.size()
                << " lambda=" << lambda << " kl=" << d.kl << " tv=" << d.tv
                << " top1=" << d.top1 << std::endl;
    }
    json += row + "}" + (li == 0 ? ",\n" : "\n");
  }
  json += std::string("  ],\n  \"lambda0_gate\": ") +
          (gate_ok ? "true" : "false") + "\n}\n";
  if (argc > 3)
    std::ofstream(argv[3]) << json;
  std::cout << "benign off-vs-on " << Stats(benign) << "\nlambda0 gate "
            << (gate_ok ? "PASS" : "FAIL") << std::endl;
  return gate_ok ? 0 : 1;
}

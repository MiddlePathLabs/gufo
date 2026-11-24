#ifndef GUFO_CACHE_HANDLES_HPP_
#define GUFO_CACHE_HANDLES_HPP_

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cache/checkpoint.hpp"

namespace gufo::cache {
// Named handles over already-retained checkpoints. A handle adds no payload:
// it pins the checkpoint's retention record until its TTL or the registry
// evicts it for capacity. Callers own the pin/unpin side effects, because
// only the retention policy can make them atomic with eviction.
class CheckpointHandles {
public:
  using Clock = std::chrono::steady_clock;
  static constexpr std::size_t kDefaultCapacity = 64;
  static constexpr std::chrono::seconds kDefaultTtl{3600};
  /// Longest TTL a client may ask for. A handle pins RAM against every
  /// other request, so an abandoned one must not outlive a working day.
  static constexpr std::chrono::seconds kMaxTtl{86400};
  static constexpr std::size_t kMaxNameBytes = 64;

  struct Entry {
    std::string name;
    CheckpointId checkpoint;
    Clock::time_point expires;
    std::uint64_t sequence{0};
  };
  // One handle displaced by Keep, Drop, Expire or capacity pressure.
  struct Removed {
    std::string name;
    CheckpointId checkpoint;
    bool expired{false};
  };

  explicit CheckpointHandles(std::size_t capacity = kDefaultCapacity);

  // Creates or replaces a handle. On success, `replaced` receives the
  // checkpoint the old handle pinned (if any) and `dropped` receives a
  // capacity victim. A checkpoint may be named more than once.
  [[nodiscard]] bool Keep(std::string_view name, CheckpointId checkpoint,
                          std::chrono::milliseconds ttl,
                          std::vector<Removed>* removed = nullptr,
                          Clock::time_point now = Clock::now());
  // Resolves a live handle, expiring stale entries it passes.
  [[nodiscard]] std::optional<CheckpointId> Resolve(
      std::string_view name, std::vector<Removed>* removed = nullptr,
      Clock::time_point now = Clock::now());
  bool Drop(std::string_view name, std::vector<Removed>* removed = nullptr,
            Clock::time_point now = Clock::now());
  // Drops every expired handle.
  void Expire(std::vector<Removed>* removed, Clock::time_point now = Clock::now());
  /// Does not expire: use List or Resolve when pins must be released.
  [[nodiscard]] std::size_t Size() const noexcept { return entries_.size(); }
  [[nodiscard]] std::size_t Capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::vector<Entry> List(
      std::vector<Removed>* removed = nullptr,
      Clock::time_point now = Clock::now());
  [[nodiscard]] static bool ValidName(std::string_view name) {
    return !name.empty() && name.size() <= kMaxNameBytes &&
           std::ranges::none_of(name, [](char c) {
             return static_cast<unsigned char>(c) < 0x21 ||
                    static_cast<unsigned char>(c) > 0x7e;
           });
  }

private:
  std::vector<Entry> entries_;
  std::size_t capacity_;
  std::uint64_t sequence_{0};
};
}  // namespace gufo::cache
#endif  // GUFO_CACHE_HANDLES_HPP_

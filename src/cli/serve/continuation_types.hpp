#ifndef GUFO_SERVER_CONTINUATION_TYPES_HPP_
#define GUFO_SERVER_CONTINUATION_TYPES_HPP_

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "src/ext/continuation.hpp"

namespace gufo::server {

// The continuation bases and token id are runtime contracts (RFC step 6).
// These aliases keep the serving spelling.
using ext::ContinuationInputPrefix;
using ext::ContinuationState;
using ext::ContinuationToken;
using ext::PrefixInputIdentity;

/// Immutable continuation payload retained by the cache.
class ContinuationSnapshot {
public:
  ContinuationSnapshot() = default;
  virtual ~ContinuationSnapshot() = default;

  ContinuationSnapshot(const ContinuationSnapshot&) = delete;
  ContinuationSnapshot& operator=(const ContinuationSnapshot&) = delete;
  ContinuationSnapshot(ContinuationSnapshot&&) = delete;
  ContinuationSnapshot& operator=(ContinuationSnapshot&&) = delete;

  [[nodiscard]] virtual std::size_t PayloadBytes() const noexcept = 0;
};

enum class SnapshotEventReason : std::uint8_t {
  kByteCapacity,
  kCaptureFailure,
  kReservationMismatch,
};

/// Continuation boundaries take precedence over optional history/retry copies.
/// A branch point is a continuation boundary learned where two prompts
/// diverged; it stays a branch point after the older prompt's checkpoints go.
enum class SnapshotPurpose : std::uint8_t {
  kContinuation,
  kHistory,
  kRetry,
  kBranchPoint,
};

/// Explains a miss without exposing prompt or token contents.
struct ContinuationLookup {
  std::string_view miss_reason;
  std::size_t common_prefix_tokens{0};
  std::size_t checkpoint_tokens{0};
  /// `cache_from` outcome when a name was supplied: "used", "fallback",
  /// "missing" (no live handle), "not_prefix" (handle exists but the prompt
  /// does not extend it) or "failed" (nothing restored). Empty otherwise.
  /// A non-"used" value names a restore the client did not ask for.
  std::string_view cache_from_status;
};

}  // namespace gufo::server

#endif

#include "src/cache/handles.hpp"

#include <algorithm>
#include <stdexcept>

namespace gufo::cache {
namespace {
void AddRemoved(std::vector<CheckpointHandles::Removed>* removed,
                CheckpointHandles::Removed entry) {
  if (removed)
    removed->push_back(std::move(entry));
}
}  // namespace

CheckpointHandles::CheckpointHandles(std::size_t capacity)
    : capacity_(capacity) {
  if (!capacity || capacity > 4096)
    throw std::invalid_argument("invalid checkpoint handle capacity");
  entries_.reserve(capacity);
}

bool CheckpointHandles::Keep(std::string_view name, CheckpointId checkpoint,
                             std::chrono::milliseconds ttl,
                             std::vector<Removed>* removed,
                             Clock::time_point now) {
  if (!ValidName(name) || ttl.count() <= 0 || checkpoint.value == 0)
    return false;
  Expire(removed, now);
  std::erase_if(entries_, [&](const Entry& entry) {
    if (entry.name != name)
      return false;
    AddRemoved(removed, {entry.name, entry.checkpoint, false});
    return true;
  });
  if (entries_.size() == capacity_) {
    const auto victim = std::ranges::min_element(
        entries_, {}, [](const Entry& entry) {
          return std::pair(entry.expires, entry.sequence);
        });
    AddRemoved(removed, {victim->name, victim->checkpoint, false});
    entries_.erase(victim);
  }
  entries_.push_back({std::string(name), checkpoint, now + ttl, ++sequence_});
  return true;
}

std::optional<CheckpointId> CheckpointHandles::Resolve(
    std::string_view name, std::vector<Removed>* removed,
    Clock::time_point now) {
  Expire(removed, now);
  const auto found = std::ranges::find(entries_, name, &Entry::name);
  return found == entries_.end() ? std::nullopt
                                 : std::optional<CheckpointId>(found->checkpoint);
}

bool CheckpointHandles::Drop(std::string_view name,
                             std::vector<Removed>* removed,
                             Clock::time_point now) {
  Expire(removed, now);
  return std::erase_if(entries_, [&](const Entry& entry) {
           if (entry.name != name)
             return false;
           AddRemoved(removed, {entry.name, entry.checkpoint, false});
           return true;
         }) != 0;
}

void CheckpointHandles::Expire(std::vector<Removed>* removed,
                               Clock::time_point now) {
  std::erase_if(entries_, [&](const Entry& entry) {
    if (entry.expires > now)
      return false;
    AddRemoved(removed, {entry.name, entry.checkpoint, true});
    return true;
  });
}

std::vector<CheckpointHandles::Entry> CheckpointHandles::List(
    std::vector<Removed>* removed, Clock::time_point now) {
  Expire(removed, now);
  return entries_;
}
}  // namespace gufo::cache

# Named cache handles on chat requests

`ChatRequest` (`src/ext/text.hpp`) gains three fields that let a client name
and later prefer a retained prompt-cache checkpoint:
- `cache_keep` (`std::string`): name the checkpoint where this request's
  reply ends and pin it against eviction. Empty keeps no handle.
- `cache_keep_ttl` (`std::chrono::milliseconds`): how long the pin lasts.
  Zero uses the host default.
- `cache_from` (`std::string`): prefer the named checkpoint as the restore
  point when this prompt extends it exactly. It never bypasses prefix or
  input-identity checks. Empty keeps the ordinary deepest-prefix selection.

The host implements all three; runners need no changes. The change is
additive: the fields default to empty or zero, so existing extensions compile
and behave unchanged.

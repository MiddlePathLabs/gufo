# N-gram disk bytes and gather wait

`TextRunnerNgramRows` (`src/ext/text.hpp`) gains two fields:
- `disk_bytes`: bytes the sequence's disk reads returned, including
  direct-I/O page alignment.
- `gather_wait_ms`: time the sequence's execution waited for n-gram gathers to
  complete. It includes copying duplicate rows and finishing another
  sequence's read that was already in flight, so it is not SSD time alone.

The change is additive. Both fields default to zero, so runners that do not set
them report as before. Positional initializers of the existing four fields stay
valid.

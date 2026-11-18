# N-gram disk bytes and gather wait

`TextRunnerNgramRows` (`src/ext/text.hpp`) gains two fields:
- `disk_bytes`: bytes the sequence's disk reads transferred, including
  direct-I/O page alignment.
- `gather_wait_ms`: time execution spent blocked waiting for the sequence's
  gathers.

The change is additive. Both fields default to zero, so runners that do not set
them report as before. Positional initializers of the existing four fields stay
valid.

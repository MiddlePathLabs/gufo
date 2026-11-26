# Qwen3.8 27B implementation notes

Developer detail behind the [public model page](../../../models/qwen3.8-27b.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)
· [ROCm 10.1 migration](ROCM.md)

BF16 targets are reference-only. Targets and drafts need not have matching
precision; the Q4_K_M DFlash2 draft serves both targets.

## Wide copy verification

A request decoding alone can verify a prompt-lookup copy wider than the DFlash2
block: greedy requests, and sampled ones without a seed. Copy-only steps start
at the ordinary eight rows; two whole copies in a row at the current width
double it, up to 64 rows, and any rejected copy returns it to eight. The wide
pass shares every projection across its rows and runs attention in eight-row
segments, so each row's logits and DFlash2 target features equal those of
consecutive ordinary verifications (`qwen27b_wide_verification_test` also checks
decoding after cutting back to any kept prefix). Rollback replays the captured
recurrent inputs of every kept row, so the session's SSM replay ring grows from
16 to 64 rows (about 91 MiB) on first use; the resource claim includes it, and
the load narrows the widest pass when the configured sessions would not fit.
When more than eight rows are kept, the draft injects all but the last at once,
so snapshots and batched steps still hold at most eight pending rows. Seeded
sampled requests and batched steps keep the ordinary width.

Greedy speculation must match AR. Sampled modes preserve the target policy,
but AR and speculation need not share the same-seed sequence. See the
[quality contract](QUALITY.md).

## Vision

Vision weights upload lazily. In `chat`, `--image` attaches to the first user
turn. Ordered text/image parts, multiple images and later turns work with
AR/DFlash2 and caches. Official dynamic resizing allows 64–16384 merged image
tokens. Cache identity includes processed pixels, placement, preprocessing
version and projector contents. See [vision checks and limits](QUALITY.md#vision).

## Runtime output projection

`--projection-vector FILE.gguf` (serve only) loads F32 `[5120]` tensors named
`projection.mixer.N` (the `attn_output`/`ssm_out` writer) and
`projection.ffn.N` (`ffn_down`) for layers `0..63`. Each tensor's norm becomes
that writer's scale `s` and its direction `v` is normalized. For a steered
request each writer's output `y` is replaced, before its residual add, by
`y - s * (y . v) * v`, which equals serving the weight `(I - s v v^T) W`
without editing the GGUF. A rank-1 refusal ablation stored as per-module
directions times its coefficients therefore reproduces that ablation.
`tools/qwen27b/convert_projection.py` converts such a safetensors file
(module-name keys plus `__coefs__`) into this format.
`--projection-layers LO-HI` (`0 <= LO <= HI <= 63`) restricts the steered
layers.

Requests opt in with `"projection": true` (or `--projection-default-on`),
exactly as on Flash-Next; the server rejects the field when no vector is
loaded. Unsteered requests launch nothing extra, so their arithmetic is the
unmodified engine's. A steered pass applies one row-projection kernel per
writer in single-token decode (the decode graph key carries the lambda, so a
toggle recaptures once), batched decode and verification (per-row lambdas, so
steered and unsteered sessions share a batch) and prompt prefill. Prefill
passes of 1024 rows or more normally take the FP16 route, whose GEMM epilogue
adds each writer's output into the residual; a steered pass takes the FP32
route instead, at about 2.25 rather than 1.80 ms per row. The DFlash2 draft is
not projected; it reads target features after the projected residual adds.

Every prompt-cache identity starts with the request's projection state while
a vector is loaded, and the persisted-state identity carries the vector's
SHA-256 and steered range. The loaded directions cost 2.5 MiB of device
memory (64 x 2 x 5120 F32) plus a 64-float per-row array per session.
`qwen27b_projection_test` checks the loader and the kernel against a
double-precision reference.

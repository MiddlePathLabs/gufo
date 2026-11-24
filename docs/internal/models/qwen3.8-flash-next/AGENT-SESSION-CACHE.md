# Agent-session caching and context reuse

Measured November 24, 2026, one server at a time on Strix Halo
(RYZEN AI MAX+ 395, gfx1151, 128 GB unified, ROCm 10.1), UD-Q4_K_XL with the
shared-Q8_0 MTP sidecar, 12 GiB RAM cache and a fresh 16 GiB disk cache per
run. The workloads ran before two later fixes: shared prefixes no longer join
another agent's conversation, and handles pin at most half the RAM budget (see
[cache.md](../../cache.md#named-handles)). Neither changes a restore or a
measured reuse figure; the record-pressure run is the one to repeat.

This page records the exact-prefix cache/handle audit, the realistic agent
workloads, the fork and record-pressure results, and the decision on
approximate composable reuse. It does not replace
[cache.md](../../../cache.md), which is the user-facing contract.

## Audit

### What `cache_keep` pins

Before this work: `CompleteSuccess` asked `Lease::RetainedCheckpoint()`, which
is set by the last `PublishSnapshot` on the request's lease. `Commit`
publishes the prompt/stable snapshot first, then captures the live frontier.
The frontier capture moves to an idle worker when any other entry is busy or
has a frontier capture in flight: `async_frontier` is true, the worker owns a
different lease, and the original lease keeps the shallower prompt checkpoint.
So a keep request could pin either the reply-end state or the prompt boundary
depending on peer scheduling. That is now fixed: a request with `cache_keep`
sets `wait_frontier`, so `Commit`/`Cancel` capture the frontier on the request's
lease. `usage.gufo.cache_handle_scope` reports `reply-end` or
`prompt-boundary`. The cost is one capture on the responding thread; it is
paid only when a client explicitly asks to keep the state.

A checkpoint with no generated tokens (`live.size() == prompt.size()`) has no
distinct reply end; the handle then pins the prompt snapshot and says so.

### `cache_from` fallback

Yes, it was silent. `cache_from` reordered the ordinary candidate list; a
failed preferred restore, or a preferred checkpoint that was not an exact
prefix, fell through to another candidate with no signal. It is now explicit:

- `usage.gufo.cache_from_status` is `used`, `fallback`, `not_prefix`,
  `missing` or `failed`;
- the server logs `schema=component-cache-v1 event=cache_from` with the name,
  status, selected and preferred boundaries.

`cache_from` still never bypasses exact-prefix or input-identity checks. A
`fallback` restore is a different branch, and the response says so.

### Handles: pins, bytes, release, TTL

Pins are reference-counted in `RetentionPolicy`; two names on one checkpoint
add one pin count each and do not duplicate payload. A pinned record is exempt
from every removal reason, including exact-boundary replacement. The registry
holds 64 names, evicts the earliest expiry, and the default TTL is one hour
(1 to 86400 seconds).

New in this work:

- a byte budget: all handles together pin at most half the RAM cache budget;
  a refused pin returns `X-Gufo-Cache-Handle-Error` and does not create a name;
- `GET /v1/cache/handles` lists name, checkpoint id, token boundary, remaining
  TTL and the pinned/budget bytes;
- `POST /v1/cache/handles/release` drops one name and unpins it.

### Conversation identity for shared prefixes

A RAM restore first copied the restored checkpoint's conversation number
whatever its depth. In the 4-session fanout run, agents a2 and a3 restored
the 1,752-token system-prompt checkpoint that a0 and a1 created, so they would
share one 16-record quota with them. A restore now continues a conversation
only when it restores that conversation's deepest record;
`RetentionPolicy::ContinuationOf` returns zero for a shallower one and the
request starts its own number (`SharedPrefixDoesNotJoinConversation`).

### Conversation identity across disk restore

Conversation numbers live only in the RAM retention table. A RAM restore keeps
the number and its 16-record quota. A disk restore after a restart starts a new
number on its next publication. This is documented as a limit: the old
conversation's durable entries stay under the disk tier's lineage-aware global
LRU, while the new number gets its own 16-record RAM quota. A restored
conversation can therefore hold up to 16 old plus 16 new disk records; the disk
byte budget, not the RAM quota, bounds it. Persisting the number in the disk
manifest would change the format for no measured benefit.

### Fork and restore cost

A restore into a free session maps the checkpoint's shared pages and copies
the private state plus the rows the page mapping does not cover. Flash-Next
private state is about 132 MB at any depth; the rest scales with rows (about
27.5 KiB per context token with MTP). `cache::RestoreEstimate` uses 55 GB/s for
a RAM restore and 13 GB/s for disk; `Validate` is the synchronization point
after the copies. Live-frontier reuse costs no restore and no copy. Exact
numbers from the fork workload are in the results below.

### Tests

Existing: conversation quota, peer-last-copy protection, reference-counted
pins, pinned exact-boundary replacement, `cache_keep`/`cache_from` resolution
and TTL expiry.

Added: reply-end keep under a busy peer, multiple names on one pin, explicit
`cache_from` outcomes, the handle byte budget, concurrent keep/list/release
churn, and `PinnedPayloadBytes`/`BoundaryOf` accounting. Functional cache
suites (`cache_concurrency.py`, `cache_rotation.py`, `cache_edits.py`,
`cache_disk_spacing.py`, `continuation.py`) remain the live-server coverage.

## Realistic agent workloads

Each request is a chat turn against the served model. A turn sends a small
user increment, appends the model's answer and a large tool result, then sends
the tool result. All agents share one ~1.7k-token system prompt and tool
schema unless a control arm says otherwise. Sampling is greedy, reasoning
effort low, 64 output tokens. Each server starts with an empty disk cache; the
first turn of each conversation is cold.

The 64-token limit ended inside the reasoning on almost every turn: 941 of
950 requests outside the fork run returned no answer text, so the harness
appended empty assistant turns. The runs therefore measure prompt-cache reuse
and queueing on growing histories, not agent quality; TTFT is time to the
first reasoning delta.

`cache_n`/`prompt_n` come from the response timings. p50/p95 are per request,
across all agents and turns.

### 32K, one server per run

| Run | Requests | TTFT p50 | TTFT p95 | Wall p95 | Reuse | Peak GTT | Min host |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pi, 1 session | 36 | 0.77 s | 0.84 s | 2.42 s | 90.7% | 90.6 GiB | 18.0 GiB |
| fanout, 1 session | 64 | 6.39 s | 7.39 s | 8.81 s | 86.2% | 92.0 GiB | 18.4 GiB |
| fanout, 4 sessions | 96 | 1.01 s | 1.97 s | 7.48 s | 88.3% | 90.9 GiB | 18.2 GiB |
| fanout, 8 sessions | 128 | 1.17 s | 3.11 s | 13.94 s | 87.1% | 95.2 GiB | 10.3 GiB |
| fanout, 4 sessions, unique system prompts | 64 | 1.00 s | 2.23 s | 7.16 s | 85.7% | 92.1 GiB | 17.3 GiB |

The one-session fanout is the baseline for the multi-agent gate: four agents
queue behind one session, so TTFT p95 is 7.39 s. Four sessions cut it to
1.97 s (**-73%**) and eight to 3.11 s (**-58%**). The arms are not matched:
the baseline ran 4 agents x 8 turns, the 4-session arm 4 x 12 (longer
histories, so the gain is if anything understated) and the 8-session arm
8 x 8 (twice the agents). Eight sessions are a
throughput profile, not a latency profile: more requests decode together, but
prefill still serializes and per-request TTFT rises. The unique-system control
keeps the same shape (2.23 s p95) because each agent's own history dominates;
sharing the system prompt saves the first-turn prefill (turn 0 p50 1.03 s
against 1.32 s unique), not every later turn.

Single-agent `pi` at one session has TTFT p95 0.84 s and 90.7% of prompt tokens
reused across turns. The default path is unchanged: a request without
`cache_keep` uses the same capture and admission code as before, and no test
shows a decode or prefill change.

### 128K and 64K

| Run | Requests | TTFT p50 | TTFT p95 | Reuse | Peak GTT | Min host |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fanout, 4 sessions, 128K context | 192 | 1.02 s | 2.32 s | 92.9% | 96.1 GiB | 13.7 GiB |
| long, 1 session, 262K context | 2 | 56.6 s* | 108.6 s* | 50.0% | 92.2 GiB | 22.3 GiB |
| fanout, 8 sessions, 64K context | 96 | 1.17 s | 3.86 s | 85.2% | 98.5 GiB | 12.9 GiB |

\*The two-request mean is not a steady-state rate: the first request is a
181,934-token cold prefill at 113.0 s TTFT; the next sends the same history
plus a small edit and restores 181,928 tokens in 11.6 ms at 161.5 ms TTFT.

Four 128K sessions served a 24-turn agent each with 92.9% prompt reuse, TTFT
p95 2.32 s, and 13.7 GiB of host memory left. This is the safe deep
multi-agent pair. One 262K session leaves 22.3 GiB of host memory and keeps
the whole history warm for the next edit. Eight 64K sessions load and stay
above the reserve (12.9 GiB) but are not a latency profile: TTFT p95 is 3.86 s
and 31 optional snapshots were skipped. Use four sessions at 64K, or eight at
32K, when first-token latency matters.

### Fork and record pressure

Fork: parent 3 turns, one reply-end keep, three children with different
branches from that keep. The keep request reused 5,944 tokens at 234 ms TTFT
and its response reported `cache_handle_scope=reply-end`. Each child restored
6,093 tokens (300 MB, of which 48 MB mapped shared pages) in 4.8-7.9 ms and
reached first token in 170-179 ms; the three responses report
`cache_from_status=used` and the server logged no fallback. The parent's
handle is an ID and a pin, so the children did not copy the parent's payload
beyond the restore each one needed. The full run: 16
requests, TTFT p95 0.97 s, 84.1% mean reuse, peak GTT 90.5 GiB, 19.8 GiB host
left.

The HTTP integration check (`handle_check.py`) passed against a two-session
32K server: `cache_keep` returned the header and `cache_handle_scope`
`reply-end`; the continuation reported `cache_from_status=used` with 56 cached
tokens; a divergent prompt reported `not_prefix`; after the two-second TTL the
handle was gone and the retry reused only a shallower checkpoint (24 tokens);
`GET /v1/cache/handles` listed the live name with a positive pinned-byte
count, and `POST /v1/cache/handles/release` removed it.

Record pressure: sixty short conversations plus one long conversation, four
sessions, 16 turns each. 272 requests, TTFT p95 0.83 s, 80.6% mean reuse, peak
GTT 90.3 GiB, 15.7 GiB host left, zero errors. The server removed 475 records
for record capacity and 305 for byte capacity; that is the 128-record table
turning over. Every one of the 120 second-phase peer requests found its own
conversation's checkpoint (`cache_n` > 0, 1,736 tokens on the user turn and
2,780 on the tool turn), and the long conversation's last turn reused 12,616
tokens. No peer's sole checkpoint was evicted by the long agent.

An earlier 12-chat, 30-turn attempt stopped at turn 30 with HTTP 400 because
that conversation had grown past the 32K context; the cache was not involved.
The 16-turn run keeps every prompt inside the configured context.



### Acceptance

- **Single-agent regression.** The default request path is unchanged: a
  request without `cache_keep` passes `wait_frontier=false` and takes the same
  capture, admission and restore code. The single-agent run kept 90.7% of its
  prompt tokens and reached TTFT p95 0.84 s; the cache test suite passes.
- **Multi-agent p95 TTFT.** Four sessions: 1.97 s p95 against 7.39 s on one
  session, **-73%** (gate: at least -20%). Eight sessions: 3.11 s, **-58%**.
- **Exact paths.** Every measured reuse is an exact-prefix restore or a live
  frontier; no approximate path is enabled. Greedy output equality is covered
  by the existing continuation and compaction probes; the handle tests assert
  exact checkpoint selection.
- **No peer's sole checkpoint evicted.** `cache_retention_test` runs the
  quota at 7, 8, 12 and 16 with eight peers, and the live pressure run reports
  peer reuse after the long conversation.
- **Peak memory.** The largest profile (8 x 64K) peaked at 98.5 GiB GTT with
  12.9 GiB host available; the 32K profiles stayed at 90-95 GiB. No unmeasured
  memory increase came from the handle work: a handle stores an ID and a pin,
  not a payload copy.

## Approximate composable reuse

Halogen's composable context is approximate cross-position reuse: it retains
message units at or above 2,048 tokens and restores them at another offset
behind the same system prompt, re-rotating cached keys. Its own documentation
says the answer is close to, not byte-identical to, a fresh read; it needs
prompt cache mode 2 and the KV pool, refuses images, and is incompatible with
8-bit KV. The measured example restored ~8,700 tool-result tokens in 0.13 s
against ~14 s fresh.

Gufo's exact-prefix checkpoints must stay exact. The question is whether any
Flash-Next component can be moved to a new offset exactly.

### Component dependence

Inventory from `continuation_layout.cpp`; "preceding" means the component's
value is a function of tokens before the message boundary, not only of the
message itself.

| Component | Kind | Depends on preceding content | Depends on position | Exact relocation |
| --- | --- | --- | --- | --- |
| `kTokens` | append | no | no | trivial, not the cost |
| `kLogits` | private | yes (last hidden) | yes (through context) | no |
| `kMetadata` (length policy, n-gram tail, hidden frontier) | private | yes | yes | no |
| `kConvolution` (SSM short convolution) | private | yes (last kernel-1 layer inputs) | no | no: layer inputs depend on the prefix |
| `kRecurrent` (DeltaNet state) | private | yes (cumulative) | no | no: state is an affine function of the initial state over the whole prefix |
| `kKey` (QSA keys) | append | yes (layer input) | yes (rotation) | no; keys could be re-rotated, values cannot be recovered |
| `kValue` (QSA values) | append | yes | yes | no |
| `kPooledKeys` (indexer, per block) | append | yes | yes | no |
| `kRawKeys` (indexer ring) | private | yes (last top-k selection) | yes | no |
| `kPleHistory` | private | yes (convolution/ngram history) | no | no |
| draft `kKey`/`kValue`/`kPooledKeys` | append | yes | yes | no |
| draft `kRawKeys`/`kResidual`/`kKeptHidden` | private | yes | yes | no |

No message unit is exactly relocatable. The linear layers' recurrent and
convolution state is cumulative; moving a message changes it. Attention keys
are also rotated by absolute position, and values depend on the layer input
that the prefix produced. Splicing stored rows at a new offset changes the
result even before the seam.

### What an approximate prototype would have to do

1. Retain each qualifying message's K/V and private state at its boundaries.
2. At a new offset, re-rotate keys for the position delta and keep the stored
   values. The stored private state is wrong; either recompute it from the new
   prefix (a full pass over the message tokens) or approximate it by replaying
   the message with the new starting state.
3. Accept a one-step quantization/seam error at the message boundary, as
   Halogen documents.

For Flash-Next, step 2 is the blocker: every token still has to pass through
the interleaved linear layers to rebuild `kRecurrent`, `kConvolution` and
`kPleHistory`; only the attention K/V work could be skipped. Attention is a
minority of prefill on this model, so the ceiling is far below Halogen's
headline, and the error is nonzero. A prototype would need a separate flag,
separate retention store (never mixed with exact checkpoints), image refusal,
and quality gates: full-vocabulary logit deltas, greedy token hashes, tool-call
JSON validity, agent-task success and retrieval quality. No such flag is
added here: the mathematics does not establish a gain on this hybrid model,
and the exact path must not be weakened.

## Recommended configuration

| Use | Sessions | Context | Evidence |
| --- | ---: | ---: | --- |
| One user, short agent | 1 | 32K | pi TTFT p95 0.84 s, 90.7% reuse |
| Agent fan-out, default | 4 | 32K | fanout TTFT p95 1.97 s, -73% against one session |
| Agent fan-out, throughput | 8 | 32K | TTFT p95 3.11 s, highest aggregate; 10.3 GiB host left |
| Deep agents | 4 | 128K | TTFT p95 2.32 s, 92.9% reuse, 13.7 GiB host left |
| One deep conversation | 1 | 256K | 181,928 tokens warm in 11.6 ms / 161.5 ms TTFT |
| Eight must hold 64K | 8 | 64K | loads, 12.9 GiB host left, TTFT p95 3.86 s; not a latency profile |

On a 128 GB Strix Halo, four sessions at 32K suit agent work with a 12 GiB
RAM cache and 16 GiB disk cache; one session at 128K or 256K suits a single
deep conversation. Do not run 8 x 128K or 8 x 256K, and do not use 8 x 64K as
a first-token-latency profile.

Handles: a fork parent can keep its reply end for 10 minutes and three
children can restore it sequentially without a copy of the parent's payload;
release children when their branch is abandoned. All pins together stay within
half the RAM budget. TTL expiry, explicit release and a process restart all
drop the pins.

## Reproduction

The harness (`agent_bench.py`) is out of tree because it launches a local
container. Each run starts one server, waits for `/ready`, sends the scenario,
samples GTT, SCLK, temperature and host `MemAvailable` every 0.5 s, stops the
server and writes `summary.json`, `requests.jsonl` and `telemetry.jsonl`. The
results are under `artifacts/agent-20261124/` (disk caches excluded).

```sh
python3 agent_bench.py --scenario fanout --sessions 4 --context 32768 \
  --agents 4 --turns 12 --stagger 1.5 --tag fanout-s4-c32k
python3 agent_bench.py --scenario fanout --sessions 8 --context 32768 \
  --agents 8 --turns 8 --stagger 0.8 --tag fanout-s8-c32k
python3 agent_bench.py --scenario fork --sessions 4 --context 32768 \
  --turns 6 --children 3 --tag fork-s4-c32k
python3 agent_bench.py --scenario pressure --sessions 4 --context 32768 \
  --agents 61 --turns 16 --tag pressure-records-s4-c32k
python3 agent_bench.py --scenario long --sessions 1 --context 262144 \
  --long-tokens 180000 --tag long-s1-c262k
```

The test evidence is `cache_retention_test`, `component_text_cache_test`,
`text_generation_scheduler_test`, `openai_chat_test`, and the remaining
`tests/cache` binaries. `handle_check.py`, also out of tree, is the live HTTP
check for keep/from/list/release.

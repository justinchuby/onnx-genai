# Deterministic FreeToken byte A/B

`freetoken_byte_ab` compares equivalent synthetic MoE/state workloads without
requiring a full DeepSeek- or GLM-class checkpoint. Deterministic byte counters
are the primary evidence. The harness records no wall-clock pass/fail threshold.

The built-in fixtures are structural analogues, not model-identity gates:

- `deepseek-like`: multiple 256-expert banks, grouped top-k routing, compressed
  and dense attention state;
- `glm52-like`: multiple grouped expert banks, recurrent/attention state, and a
  typed temporal state group demonstrating that future video/multimodal state
  uses the same schema.

Runtime behavior uses the fixture's typed dimensions, bank extents, routes, and
state groups. It never branches on a model name.

## Run

```bash
cargo run -p onnx-genai-bench --no-default-features \
  --bin freetoken_byte_ab -- \
  --fixture deepseek-like \
  --output /datadisks/disk5/justinchu/freetoken-byte-ab/deepseek-like.json

cargo run -p onnx-genai-bench --no-default-features \
  --bin freetoken_byte_ab -- \
  --fixture glm52-like \
  --output /datadisks/disk5/justinchu/freetoken-byte-ab/glm52-like.json
```

An external workload may be supplied with `--workload workload.json`. It must
use `onnx-genai.freetoken-byte-ab.workload.v2`; empty routes, zero dimensions,
out-of-range experts, missing phases, and mismatched `batch * top_k` selections
fail rather than producing a vacuous PASS.

## What one byte means

The report embeds `onnx-genai.freetoken-byte-taxonomy.v1`. Categories are
separate measurement planes; they are not added into a universal total.

| Category | Exact boundary |
| --- | --- |
| `source_read` | Bytes returned by the synthetic source's exact read boundary. Built-in large-model fixtures use declared virtual source extents, explicitly labeled synthetic rather than measured checkpoint I/O. |
| `host_allocation` | Requested payload capacity after the host allocation succeeds. Allocator metadata and physical RSS are not inferred. |
| `host_write` | Non-zero payload copied into host storage. |
| `zero_fill` | Explicit zero initialization; excluded from `host_write`. |
| `h2d`, `d2h`, `d2d` | Payload covered by a completed CUDA/event receipt. An enqueue is pending, not committed traffic. |
| `mmap_page_in` | Bytes confirmed by an external page-fault receipt. Synthetic runs report zero; they do not relabel source reads as OS page-ins. |
| `vmm_map`, `vmm_unmap` | Virtual bytes whose physical mapping changed. Mapping topology is not transport and is not added to H2D. |
| `expert_materialization` | Logical selected-expert payload consumed by routed computation. It is useful work, not physical traffic. |
| `state_materialization` | Logical recurrent/attention state advanced. |
| `scratch_journal` | Scratch/checkpoint/journal payload touched. |

Every phase has four disjoint outcome buckets:

- `useful`: operation reached its completion boundary;
- `failed`: operation failed before a useful commit;
- `rolled_back`: attempted work was restored transactionally;
- `quarantined`: ownership could not safely return to the useful pool.

Failed, rolled-back, and quarantined H2D bytes are never counted as useful
completed H2D.

## Equivalent work

The baseline-absent and optimized arms consume the same:

- workload digest and exact route list;
- batch, top-k, bank/group dimensions, quantization metadata, and state groups;
- prefill, direct warmup, capture setup, replay, and steady decode progression;
- sequence positions and generated-token rule.

The contract compares route, final-state, output, and token-ID digests before it
compares traffic. Mutating one arm's route or token output is a test failure.

The baseline streams each selected expert and stages state through the host.
The optimized arm uses deterministic per-bank LRU residency and persistent
device state. These are movement strategies only; semantic state progression is
shared.

## Setup versus steady state

Reports keep these phases separate and include denominators:

1. `setup`: persistent state allocation/initialization;
2. `prefill`;
3. `direct_warmup`;
4. `capture_setup`;
5. `replay`;
6. `decode_steady`;
7. `failure`.

No setup/prefill/cold byte is averaged into warmed decode. Each phase records
requests, tokens, expert selections, unique experts, submissions, cache
hits/misses, replays, and state updates. Per-token or per-expert values for CI
can therefore be derived from exact numerators and denominators without hidden
averaging.

## Default-off and positive controls

`baseline_absent` performs the semantic workload but records:

- zero FreeToken feature lookups;
- zero feature-specific bytes in every useful/failure bucket.

The optimized arm must record non-zero feature lookups, committed H2D, cache
hits, and cache misses. Paired baseline and optimized failure controls inject
the same rolled-back H2D submission and quarantined journal mapping, then prove
useful residency and semantic state are unchanged. Their non-useful byte
buckets must match exactly. Empty selection/workload inputs fail.

## Scope, concurrency, reset, and exhaustion

Each ledger is owned by the complete
`provider/device/executor/generation/logical-session` identity. There is no
process-global counter. Stable multi-session aggregation sorts by that complete
identity, so thread completion order cannot change JSON.

Snapshot and reset require the ledger's private benchmark authority. Both
refuse while a submission is in flight, preventing reset/completion TOCTOU.
The authority is diagnostic-only and is never accepted by production placement,
artifact finalization, reservations, or execution.

All additions use checked arithmetic. Counter or identity exhaustion returns an
actionable error and preserves the previous value; wrap and saturation are not
reported as success.

## CUDA completion and capture/replay control

The serialized GPU control uses real CUDA allocations, asynchronous H2D
transfers, completion events, stream capture, graph replay, D2D copies, and a
D2H byte-equivalence check:

```bash
CUDA_VISIBLE_DEVICES=<idle-a100> ONNX_GENAI_CUDA_DEVICE=0 \
  cargo test -p onnx-genai-bench --features gpu-tests \
  --test freetoken_byte_accounting_gpu -- --test-threads=1 --nocapture
```

H2D is committed only after the event recorded after the copy synchronizes.
Each replay is counted only after its post-launch event completes. Host
instrumentation is outside the captured region; the graph contains only the
fixed-address D2D state-carry copy. The test asserts non-empty exact byte
totals and byte-identical replay output.

The CUDA control complements, rather than replaces, the current production
authorities:

- #2341: private executor/provider/generation artifact readiness and failed
  build rollback;
- #2163: executor-scoped VMM reservations and teardown;
- #2063: typed state snapshots, scratch/journals, rollback, sibling isolation;
- #2342: governed CSA loader/state ownership.

Their exact regression suites must be run at the same head. The harness does
not forge or replace those authorities and adds no mutable production-global
hook.

## Performance safety

The deterministic ledger and CLI live in `onnx-genai-bench`. Production crates
receive no callback, environment read, allocation, lock, copy, synchronization,
or mutable global. The only Cargo propagation is the explicit `gpu-tests`
feature used by the CUDA control. Normal production and warmed decode paths are
unchanged.

Wall clock can be reported separately as informational evidence only after an
idle-before-every-run protocol. It is intentionally absent from the deterministic
contract because shared A100 load and clock ramp cannot invalidate byte
equalities.

## Output stability

- workload: `onnx-genai.freetoken-byte-ab.workload.v2`
- run: `onnx-genai.freetoken-byte-ab.run.v2`
- comparison: `onnx-genai.freetoken-byte-ab.comparison.v2`
- taxonomy: `onnx-genai.freetoken-byte-taxonomy.v1`

JSON uses ordered enums/maps and a stable phase/class census. Signed deltas are
decimal strings so a full `u64` difference is never narrowed to `i64`.

## Checkpoint limitation

The built-in fixtures use exact declared synthetic byte extents and
structurally truthful MoE/state dimensions. They are not ONNX exports and do
not establish model quality, tokens/second, or full-checkpoint I/O behavior.
No 149-GiB-class DeepSeek/GLM E2E claim is valid until a runnable ONNX export is
available and exercised through the governed loader.

# FreeToken byte estimates and production A/B receipts

FreeToken evidence has two separate contracts:

1. `freetoken_byte_ab` is a deterministic synthetic **estimate** model. Its
   schemas and fields say `estimate`; it performs no production operation.
2. `freetoken_byte_accounting_gpu` runs structurally typed QMoE sessions through
   the real session executor, CUDA EP, routing, lazy materialization, kernels,
   state progression, graph capture, and replay. Its report is derived from
   production receipts and exact output/state comparisons.

Never combine estimated and observed categories.

## Authenticated observation ownership

There is no public label-based recorder opener. Callers may configure only a
bounded event capacity on an unshared CUDA provider:

```text
CudaExecutionProvider::configure_observed_byte_capacity(capacity)
```

Capacity is policy, not authority. It creates no ledger and accepts no provider,
executor, generation, or logical-session labels.

The session-private provider-artifact lifecycle creates one recorder for the
exact provider/device/executor/generation/logical-session identity during
finalization. The validated artifact requirement installs that recorder only
while the owning executor performs setup, eager execution, capture, or replay.
Sibling executors sharing one provider receive distinct ledgers. Operations on
another provider/runtime cannot see the active recorder.

The owning `InferenceSession` can borrow its provider-specific observation
control through `provider_artifact_observation`. No labels are accepted at this
boundary. Recorder construction and runtime attachment remain crate-private.
Executor teardown retires the exact recorder; stale/reset/closed recorders reject
new operations, and identities never wrap or reuse.

## Operation/telemetry transactions

Production CUDA operations reserve event capacity and checked byte totals before
submitting work. The reservation is held until one terminal outcome:

- successful operation → publish the complete receipt batch;
- failed operation → publish failed/rolled-back/quarantined bytes;
- abandoned completion → fault the ledger, so snapshot/comparison fails.

Capacity or `u64` overflow therefore fails before a measured allocation, copy,
memset, or state-producing kernel is submitted. Reset and close refuse in-flight
submissions. A stale recorder cannot continue writing after reset. Deferred
release paths that cannot return an error fault the owning ledger and emit an
explicit diagnostic; a later snapshot cannot silently omit the operation.

Events remain bounded, ordered, and versioned. They carry provider, device,
executor, generation, logical session, epoch, submission, sequence, phase,
category, boundary, status, and exact bytes.

## Host materialization receipts

`ResidentWeightMaterialization` describes operations performed by that
invocation:

| Kind | Host allocation | Host write |
| --- | ---: | ---: |
| `AllocatedAndWritten` | newly allocated bytes | bytes actually copied |
| `ReusedResident` | 0 | 0 |
| `SharedBacking` | 0 | 0 |

The production external-weight materializer performs its allocation/copy inside
`copy_from_bytes`, which creates the receipt. Route reservation consumes that
receipt; it no longer infers operations from `resident.bytes().len()`.

Pinned staging is a separate real allocation/write boundary. If a materialized
resident is copied into pinned staging, both writes are counted because both
occurred. Cache hits, shared `Arc` values, and shared/mmap backing report zero
resident allocation/write. Concurrent cache materialization is tested so one
winner reports the allocation/write and deduplicated callers report reuse.

## Production A/B fixture

The GPU fixture builds two shape-driven structural cases:

- `deepseek-like`: many-expert hybrid structure;
- `glm52-like`: grouped recurrent structure.

These names label fixtures only; runtime behavior is driven by dimensions,
expert count, bank count, and top-k.

Every bank/expert owns a deterministic unique packed-weight and scale pattern.
Routes cover hot, cold, repeated, cross-bank, top-1, and top-2 selections.
Outputs feed an explicit carried state input on the next step. The state output
is marked as an authoritative state publication, so the executor reserves its
telemetry before kernel submission and publishes only after successful
execution.

Baseline and optimized arms share the same model bytes, inputs, routes, shapes,
state progression, capture schedule, and replay count. They differ only in the
FreeToken route-residency configuration. `semantic_equivalent` is true only when
every observed route digest, generated length, output digest, and state digest
matches exactly.

Each arm executes:

1. provider/session finalization;
2. prefill;
3. direct warmup;
4. first real CUDA graph capture;
5. three real graph replays;
6. four state-carrying decode steps;
7. snapshot and teardown.

The test requires nonzero captured segments, replay events, state-publication
bytes, H2D receipts, and route-dependent output diversity.

Run serialized on an idle CUDA device:

```bash
CUDA_VISIBLE_DEVICES=<idle-a100> ONNX_GENAI_CUDA_DEVICE=0 \
ONNX_GENAI_FREETOKEN_OBSERVED_OUTPUT_DIR=<private-datadisk-directory> \
cargo test -p onnx-genai-bench --no-default-features \
  --features gpu-tests,cuda-13000 \
  --test freetoken_byte_accounting_gpu -- \
  --test-threads=1 --nocapture
```

The report includes structural dimensions, exact scope identities, event counts,
cold/warm completed H2D bytes, exact deltas, capture/replay calls,
state-publication bytes, all semantic proofs, and explicit
`device_bytes_without_release_receipt` /
`mapped_bytes_without_unmap_receipt` residuals for provider-lifetime resources
whose teardown is not represented by an allocation-specific receipt.

## Failure and default-off controls

- Capacity-one control: the first session-owned CUDA allocation records one
  event; the second returns the exact capacity error before allocation.
- Ledger tests cover first/middle/last capacity and total overflow, `u64`
  overflow, failure rollback, duplicate completion, concurrent snapshot,
  reset/close, stale epoch, moved handles, and same-label instance isolation.
- Public compile-fail tests reject ledger construction/cloning, direct runtime
  attachment, and the removed label-based provider opener.
- Without capacity configuration, the runtime owns no registry or recorder.
  Production operations take the unchanged `OnceLock::None` branch.

## Limitations

The structurally typed payloads are synthetic. They execute real production
session/CUDA boundaries but are not an official DeepSeek or GLM checkpoint.
Reports make no checkpoint-I/O, model-quality, tokens/s, or throughput claim.
Wall-clock time is not a correctness gate.

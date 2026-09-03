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

After the session privately issues its immutable
provider/device/executor/generation/logical-session configuration, but before
model materialization begins, it supplies the provider a unique non-zero-sized
build-owner allocation. The provider creates one bounded observation-only
context for that exact owner and rejects a second owner even when every public
label matches. This context grants no route, reservation, finalization, or
artifact-use authority.

The build transaction retains the context through model load, kernel preflight,
binding allocation/upload/readback, and first-run finalization. Bindings keep
the exact same context rather than snapshotting an optional finalized artifact
requirement. A successful terminal provider report atomically marks that
context as eligible to back the separately retained artifact requirement.
Failed or abandoned build generations retire the prepared context and cannot
be revived for a later generation.

The measurement boundary begins after allocating the recorder's own bounded
storage and immediately before authoritative session build/materialization.
Recorder infrastructure allocation is outside the totals; all
model/binding/provider operations after that boundary are inside.
Sibling executors sharing one provider receive distinct ledgers. Operations on
another provider/runtime cannot see the active recorder.

Only after successful finalization can the owning `InferenceSession` borrow its
provider-specific observation control through `provider_artifact_observation`.
No labels are accepted at this boundary. Recorder construction and runtime
attachment remain crate-private.
Executor teardown retires the exact recorder while already-issued binding and
deferred-release owners may finish their teardown receipts. Stale foreign
handles and closed recorders reject new measured operations, and identities
never wrap or reuse.

The active context is a fixed-capacity TLS stack of borrowed recorder pointers.
The event ledger reserves preallocated slots and totals with atomics. The
warmed operation path has no mutex, hash map, growing vector, thread-id lookup,
reference-count clone, heap allocation, formatting, or host synchronization.
Context depth/capacity exhaustion fails before the CUDA operation. Cold
snapshot/reset/teardown may allocate while aggregating results.

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
stream, category, boundary, status, and exact bytes. Zero-byte elision and
asynchronous completion that is not independently witnessed are explicit
`elided` / `unsupported` outcomes rather than inferred useful bytes.

## Binding transfers and completion-aware publication

`DeviceIoBinding` retains the exact private observation context prepared for its
session, including when the binding is allocated before artifact finalization.
Its allocation, `write_bytes` H2D, `read_bytes[_into]` D2H, D2D state
snapshot/restore, and teardown therefore use the same authenticated recorder as
executor work. Finalization upgrades the context in place; it never mutates a
binding or attaches caller-supplied labels. Raw CUDA byte counters at the
`cuMemcpy*`/`cuMemset*` call sites provide an independent control; the
production test reconciles whole-session H2D, D2H, D2D, and memset byte deltas
with ledger totals.

Externally bound outputs use a preallocated rollback allocation. Eager, first
capture, and replay snapshot the previously visible bytes before submission.
The provider prepares a stack-owned publication receipt (no boxed hot-path
receipt), records attempted bytes when work is submitted, synchronizes through
the request contract, consumes the exact owner/device/generation validation
receipt, and only then publishes useful state/output bytes. Sync or validation
failure restores every prepared output, publishes zero useful bytes, records
rolled-back bytes, clears the failed binding receipt, and permits a clean retry.
If restoration itself fails, the exact binding is poisoned and the primary
error is returned with cleanup context.

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

The GPU fixture builds two shape-driven, symbolic-batch structural cases:

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

There is no preparatory `warmup`. Each arm executes:

1. provider/session build and model materialization;
2. pre-finalization binding allocation and initial H2D upload;
3. first prefill, including exact artifact finalization;
4. direct warm execution;
5. first real CUDA graph capture;
6. three real graph replays;
7. four state-carrying decode steps;
8. snapshot and teardown.

The test requires nonzero cold allocation, H2D, publication, and teardown
receipts; captured segments; replay events; route-dependent output diversity;
and equality between raw CUDA transfer/memset counters and ledger totals.

Run serialized on an idle CUDA device:

```bash
CUDA_VISIBLE_DEVICES=<idle-a100> ONNX_GENAI_CUDA_DEVICE=0 \
ONNX_GENAI_FREETOKEN_OBSERVED_OUTPUT_DIR=<private-datadisk-directory> \
cargo test -p onnx-genai-bench --no-default-features \
  --features gpu-tests,cuda-13000 \
  --test freetoken_byte_accounting_gpu -- \
  --test-threads=1 --nocapture
```

The report includes structural dimensions, exact scope identities, per-phase
event counts, cold/warm completed H2D and D2H bytes, submitted D2D and memset
bytes, exact deltas, capture/replay calls, state/output-publication bytes,
warmed recorder-retain and lock/lookup/growth counters, all semantic proofs, and explicit
`device_bytes_without_release_receipt` /
`mapped_bytes_without_unmap_receipt` residuals for provider-lifetime resources
whose teardown is not represented by an allocation-specific receipt.

## Failure and default-off controls

- A focused production control creates bindings before finalization, uses the
  same retained observation after first-run upgrade, and observes their settled
  teardown releases. Host ledger tests cover close plus exact
  first/middle/last capacity exhaustion without sacrificing the teardown slots
  a real CUDA allocation requires.
- Ledger tests cover bounded capacity and total overflow, submitted failure
  classification, concurrent producers, reset/close, TLS capacity and ABA-stale
  handles. The warmed recorder micro-control observes 128 positive events with
  zero host allocations, mutex acquisitions, thread-id lookups, vector growth,
  or retained recorder clones.
- Production fault controls inject eager deferred-validation failure, first
  capture synchronization failure, and replay deferred-validation failure after
  submission. Each preserves the prior output/state bytes and successfully
  retries.
- Observation lifecycle controls record before finalization, upgrade the same
  state in place, abort an unfinalized generation without revival, and reject a
  foreign owner attempting to reopen an exact live scope.
- Public compile-fail tests reject ledger construction/cloning, direct runtime
  attachment, and the removed label-based provider opener.
- Without capacity configuration, the runtime owns no ledger or recorder.
  Production operations take one false atomic gate and never touch TLS.

## A100 observed result (2026-09-02)

Serialized on one idle A100-SXM4-80GB (`CUDA_VISIBLE_DEVICES=0`,
`gpu-tests,cuda-13000`, one test thread). These are absolute production-boundary
bytes, not a reduction claim:

| Fixture / arm | cold H2D | warm H2D | cold D2H | warm D2H | cold D2D submitted | warm D2D submitted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek-like baseline | 100,860,080 | 590,976 | 131,076 | 1,179,684 | 196,608 | 1,769,472 |
| deepseek-like FreeToken | 100,860,192 | 591,480 | 131,132 | 1,180,188 | 196,608 | 26,935,296 |
| glm52-like baseline | 151,388,400 | 1,181,376 | 262,148 | 2,359,332 | 393,216 | 3,538,944 |
| glm52-like FreeToken | 151,388,568 | 1,182,132 | 262,232 | 2,360,088 | 393,216 | 41,287,680 |

The previously omitted binding minima are now present exactly in baseline warm
H2D (590,976 B / 1,181,376 B). D2H additionally includes nine four-byte
owner-validation reads per arm, yielding 1,179,684 B / 2,359,332 B warm.
FreeToken adds route-telemetry traffic (+504/+756 B warm H2D and D2H) and
substantial measured D2D route-residency work; it does **not** reduce transfer
bytes in these fixtures. Both arms remain exactly equivalent for routes,
outputs, carried state, one capture, and three replays.

Cold device-allocation receipts were 924,288 / 926,336 B for the deep-like
baseline/FreeToken arms and 2,959,040 / 2,962,112 B for the grouped-recurrent
arms. Teardown observed 524,416 B and 1,179,840 B of binding/device releases,
respectively. Whole-arm raw CUDA H2D, D2H, D2D-attempted, and memset-attempted
bytes equal the corresponding ledger totals in all four arms.

## Limitations

The structurally typed payloads are synthetic. They execute real production
session/CUDA boundaries but are not an official DeepSeek or GLM checkpoint.
Reports make no checkpoint-I/O, model-quality, tokens/s, or throughput claim.
Wall-clock time is not a correctness gate.

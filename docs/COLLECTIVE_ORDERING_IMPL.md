# Collective algorithms + ordering (distributed-runtime slice 1c)

This document is the implementation companion to
`specs/tla/CollectiveOrdering.tla`, `specs/tla/REFINEMENT.md`, and
`docs/DISTRIBUTED_RUNTIME.md` §§3, 6, and 8. The in-process backend is the
correctness oracle for later NCCL/Gloo implementations.

## Scope

Slice 1c implements every collective method already present on `Communicator`:

- `all_reduce`: in-place fixed-rank-order reduction.
- `all_gather`: member-order concatenation.
- `reduce_scatter`: fixed-rank-order reduction followed by member-position
  slicing.
- `broadcast`: direct root replication.
- `all_to_all`: direct peer-vector transpose.
- `exchange_counts`: the ordering/envelope phase of variable all-to-all.
- `all_to_all_v`: checked offset/count transfer using the single-use count
  ticket. Count exchange plus data transfer occupy one `CommInstanceId`.

The backend rendezvous stores host copies under a shared mutex and wakes ranks
with a condition variable. It deliberately favors auditable behavior over
throughput. Every extent uses checked arithmetic; all-to-all-v additionally
checks codec alignment, offset overflow, bounds, and overlapping receive spans.
Block-quantized reduction remains a later backend capability.

Reductions visit ranks in frozen member order. Integer sum/product are checked
and fail on overflow. Floating sum/product use IEEE operations in that fixed
order; min/max use `total_cmp` so NaNs and signed zero have deterministic
ordering. F16/BF16 round back to the wire dtype after each rank contribution.

## CollectiveOrdering discipline

`CollectiveSequencer` is the single authority for coordinator decisions and
per-group rank cursors. The first rank reaching a slot establishes the canonical
`(ExecutionId, CommSequenceId, collective kind, ordering-argument signature)`;
every other member must match that prefix exactly. A mismatch fails immediately
with `CommError::Ordering`, preventing mismatched collectives from reaching the
backend rendezvous.

Execution decisions form one monotonic topology-wide prefix. Normal collective
calls auto-admit the next execution; `admit_execution`, `skip_execution`, and
`observe_skipped` expose the modeled coordinator/skip transitions without
building the deferred `FrozenPlan` executor. Groups have independent canonical
logs, so cross-group enqueue order is intentionally unconstrained. Subgroups are
created from sorted, unique world ranks with `create_group`.

Abort closes the sequencer before backend wakeup, emits one global `Abort`, and
terminally completes every active rank-local submission with an error. Already
submitted waiters then quiesce without publishing duplicate completion events.

## Linearization map

| TLA+ action | Concrete linearization point |
|---|---|
| `AdmitNext` / `SkipNext` | `CollectiveSequencer::decide_locked`: the next execution decision enters the monotonic prefix under the sequencer mutex. |
| `Submit(rank, group)` | `CollectiveSequencer::submit`: the rank cursor consumes the canonical slot and the active local operation is inserted immediately before rendezvous/backend enqueue. |
| `ObserveSkip(rank, group)` | `CollectiveSequencer::observe_skip`: the skipped slot advances that rank's group cursor without adding a transport-log entry. |
| `CompleteLocal(rank, group)` | `CollectiveSequencer::complete`: output bytes are already visible and the ownership registry is terminal before the active local slot is removed. |
| `Abort` | `CollectiveSequencer::abort`: submission is frozen under the sequencer mutex before barriers/collectives are notified. |

Each point emits `ProtocolEvent::CollectiveOrdering` with contract revision 2.
This is additive under the existing non-exhaustive `ProtocolEvent` envelope, so
the envelope meaning and `CONTRACT_REVISION` do not change. Events carry stable
group/rank/execution/sequence identities, ordered membership plus its FNV-1a
hash, and per-source monotonic sequence numbers.

## Independent conformance checker

`tests/collective_ordering_conformance.rs` reconstructs abstract state from the
trace only; it never calls sequencer transition functions or the implementation
membership-hash helper. After every event it verifies:

- monotonic, hole-free coordinator decisions;
- submitted instances belong only to admitted executions;
- strictly increasing, prefix-compatible rank logs within each group;
- non-members never touch group state;
- local completion never exceeds local submission;
- cross-group logs have no global ordering requirement; and
- abort freezes all transport logs while allowing active local completions.

Campaigns cover concurrent multi-rank collectives, overlapping groups,
coordinator skips, local-completion skew, and abort after partial submission.
Checker self-tests reject unknown revisions, duplicate/reordered source
sequences, lossy traces, cross-rank divergence, post-abort submission, and
leftover active operations. A fixed explicit schedule is encoded twice and
compared byte-for-byte.

## Distributed equals single-device

The headline equivalence test gives three logical ranks independent F32 partial
vectors, performs the same fixed-order sum once as a single-device computation
and once through in-process `all_reduce`, and compares every result byte. All
ranks are bit-identical to the single-device result. This is intentionally a
minimal tensor-parallel-style proof vehicle, not the deferred general
TensorParallel/ExpertParallel strategy layer.

## Folded slice-1b follow-ups

- Abort wakes in-flight barrier waiters, which return `CommError::Aborted`.
- Barrier generations track departures and are removed after the final waiter.
- Freed allocation tombstones are capped. `seal_allocator_epoch` compresses
  identities below an allocator-proven floor while permanently rejecting reuse;
  a full exact window refuses further frees rather than forgetting safety state.
- `BufferOwnershipEvent` remains exhaustively matchable. Its deliberate lack of
  `#[non_exhaustive]` is unchanged so the independent checker must handle every
  ownership transition.

## Deferred

General `FrozenPlan` execution, TensorParallel/ExpertParallel strategies, real
device transports, and block-quantized reduction semantics remain later slices.
TLC execution may be CI-deferred when Java is unavailable; deterministic Rust
campaigns are the implementation-side refinement gate.

## Verification result

- `onnx-runtime-comm`: 27 unit tests, 15 buffer-ownership conformance tests,
  and 13 collective-ordering conformance tests.
- `onnx-genai-scheduler`: 33 unit tests and 15 pressure-conformance tests.
- `onnx-runtime-protocol-trace`: shared envelope/checker crate builds and its
  doctests pass.
- Required package build is warning-free. Touched communication/trace crates are
  clippy-clean with `-D warnings`.

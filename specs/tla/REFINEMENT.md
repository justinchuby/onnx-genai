# Implementation Refinement Contract

This document is normative for implementations of the protocols modeled in
this directory. TLC proves properties of the abstract state machines. An
implementation is conformant only when its concrete transitions can be mapped
to those abstract actions and its test traces pass an independent refinement
checker.

Passing TLC alone is not evidence that Rust, CUDA, or a transport backend
implements the model.

## Contract Revision

The current protocol contract revision is `2`.

Every implementation trace must carry this revision. A change to an action,
state mapping, event field, or external assumption must:

1. update the affected TLA+ module and model configuration;
2. increment the contract revision when old traces change meaning;
3. update the independent trace checker;
4. add a test that fails under the old behavior; and
5. rerun all affected TLC configurations.

## Required Trace Envelope

Protocol events use a lossless, test-visible envelope:

```rust
pub struct ProtocolTraceEvent {
    pub contract_revision: u32,
    pub topology_epoch: u64,
    pub source: ProtocolSourceId,
    /// Monotonic within `source`; never inferred from wall-clock time.
    pub source_sequence: u64,
    pub kind: ProtocolEvent,
}
```

Events must contain stable identities, not pointers or vector positions:

- `PressureRequestId`, `PressureGeneration`, and `PhysicalAllocationId`;
- `GroupId`, ordered-membership hash, `ExecutionId`, `CommSequenceId`, and
  world `RankId`; and
- operation ID plus complete read/write `PhysicalAllocationId` lease sets.

Trace collection used for conformance testing must be lossless. Buffer
overflow, serialization failure, duplicate source sequence, or an unknown
contract revision fails the test. Production telemetry may sample events, but
sampled telemetry is never accepted by the conformance checker.

## Linearization Map

An action maps to one concrete linearization point. Logging before or after the
specified point changes the observed protocol and is non-conformant.

| TLA+ action | Concrete linearization point |
|---|---|
| `PressureProtocol!Submit` | Request is inserted under the HostGovernor ledger lock with its unique request ID, generation, owner, and checked byte extent. |
| `PressureProtocol!Grant` | The exact allocation charge and `Pending -> Granted` transition commit atomically, before wakeup. |
| `PressureProtocol!Claim` | Poll atomically takes the granted allocation and disarms cancellation before returning it. |
| `PressureProtocol!CancelPending` | Ledger removes or terminally marks the pending request. |
| `PressureProtocol!CancelGranted` | Ledger returns the exact granted allocation and publishes cancellation in one critical section. |
| `PressureProtocol!TimeoutPending` / `TimeoutGranted` | The deadline winner publishes failure; a granted winner also returns its exact charge. |
| `PressureProtocol!Reconfigure` | Configuration generation increments and all prior-generation pending requests are revalidated or failed under the same ledger lock. |
| `PressureProtocol!Reclaim` | Released bytes are credited to the ledger; a notice alone is not reclaim completion. |
| `CollectiveOrdering!AdmitNext` / `SkipNext` | Coordinator durably publishes the monotonic execution decision to every participant. |
| `CollectiveOrdering!Submit` | Per-group sequencer consumes `(ExecutionId, CommSequenceId)` immediately before backend enqueue. |
| `CollectiveOrdering!CompleteLocal` | The local backend completion becomes terminal, including its error result. |
| `CollectiveOrdering!Abort` | Coordinator closes admission/submission for the topology epoch before backend abort begins. |
| `BufferOwnership!Submit` | Backend registry owns the operation and its full read/write lease sets before transport/device enqueue. |
| `BufferOwnership!Detach` | User handle detaches without changing registry ownership. |
| `BufferOwnership!BeginAbort` | Operation becomes aborting while all leases remain owned. |
| `BufferOwnership!CompleteSuccess` / `QuiesceAbort` | Backend proves terminal completion before releasing registry leases. |
| `BufferOwnership!FreeBuffer` | Allocator commits free/reuse only after no registry read or write lease references the physical allocation. |

Wakeups, callbacks, and trace emission must not introduce a second state owner.
The authoritative mutation remains the ledger, submit sequencer, backend
registry, or allocator named above.

## Abstraction Maps

### Pressure protocol

For every trace prefix:

- abstract `free` is the configured host capacity minus every live concrete
  charge represented below;
- `FixedCharge` is the sum of pinned or otherwise non-reclaimable concrete host
  allocations outside the modeled tickets;
- `reclaimable[device]` is the sum of evictable host allocations charged to
  that local device and not represented by a live ticket;
- `reserved[ticket]` is the exact byte extent of a concrete
  `Granted(allocation)`;
- `claimedHeld[ticket]` is the exact extent returned by a claimed ticket until
  `release_host_pages(allocation)` linearizes; and
- ticket state and generation map by stable request ID, never by queue index.

All concrete additions and subtractions use checked arithmetic before mutation.
Integer overflow, negative headroom, duplicate allocation identity, or a
snapshot that does not equal the allocation ledger is a conformance failure.

### Collective ordering

The checker partitions events by `(topology_epoch, GroupId)` and verifies the
ordered-membership hash before replay:

- coordinator decisions define the monotonic execution prefix;
- each rank's abstract cursor advances when that rank observes an admitted or
  skipped instance;
- each transport log appends only at the backend-enqueue linearization point;
  and
- local completion advances only for instances already submitted by that rank.

Events from different groups are not globally sorted. Events within one group
are merged using coordinator decision order, per-rank source order, and the
group's submit-sequencer order. Wall-clock timestamps are diagnostic only.

### Buffer ownership

For each operation:

- abstract `Active` means the backend registry state is `Submitted` or
  `Aborting`;
- `ReadBuffer` and `WriteBuffer` generalize to the complete concrete read and
  write lease sets;
- two active concrete operations may share read leases, but a write lease
  conflicts with every other read or write lease on that allocation; and
- `bufferGeneration` maps to allocator reuse generation. A process-unique,
  never-reused `PhysicalAllocationId` may satisfy the same ABA-prevention
  obligation without a separate exposed counter.

An operation with an empty write set does not advance an allocation generation.
In-place operations declare the same allocation in both sets and therefore
hold an exclusive lease.

## Independent Replay Checker

The conformance checker is test-only infrastructure and must not call the
implementation's transition functions. Sharing identity/event definitions is
allowed; sharing the state-transition reducer would make the test
self-confirming.

For every event, the checker:

1. validates the envelope, stable identities, membership hash, and checked
   byte extents;
2. finds exactly one enabled abstract action;
3. applies that action to its independent abstract state;
4. evaluates all invariants after the transition; and
5. rejects leftover active requests, operations, or leases unless the test
   explicitly ends at a modeled crash boundary.

Ambiguous action matches, missing events, impossible events, and reordered
events fail with the smallest offending trace prefix.

## Required Test Campaigns

Implementation PRs must run deterministic, repeatable campaigns for:

- grant versus claim, cancel, timeout, release, and reconfigure;
- multiple variable-sized tickets from the same and different devices;
- cancellation-mailbox saturation and ticket drop during wakeup;
- overlapping executions across overlapping communicator groups;
- coordinator skip before submission and failure after partial submission;
- local completion skew, communicator abort, and quiescence timeout;
- detached handles racing success, abort, and allocation free;
- read/read alias, read/write conflict, in-place write, and allocator reuse; and
- checked-arithmetic boundaries at zero, exact capacity, and maximum extent.

The scheduler must support a fixed seed and an explicit decision trace so every
failure can be reproduced. Random stress without replay is supplemental, not a
gate.

## Verification Gates

No protocol implementation is complete until all of these pass:

1. **Model gate:** affected `.cfg` files complete exhaustive TLC checking.
2. **Mapping gate:** the PR identifies each concrete linearization point and
   updates the table above when it changes.
3. **Conformance gate:** deterministic campaigns produce lossless traces
   accepted by the independent replay checker.
4. **Invariant gate:** implementation snapshots independently recompute ledger,
   group-order, and lease invariants from authoritative entries.
5. **Fault gate:** injected transport errors, task cancellation, timeout, and
   shutdown reach terminal/quiesced states without releasing live ownership.
6. **Backend assumption gate:** each backend has a smoke test for the completion,
   stream-order, abort, and buffer-lifetime assumptions it claims.

Small bounded models and deterministic conformance campaigns run on every
protocol-changing PR. Larger constants, more schedules, and hardware backend
campaigns run nightly. A flaky protocol test is a correctness failure and may
not be silently retried to green.

## Explicit Proof Boundary

These gates provide strong, auditable refinement evidence; they are not a
machine-checked proof of arbitrary Rust or GPU execution. A mathematical
end-to-end proof additionally requires:

- a lower-level implementation model that refines these abstract modules;
- a proof or exhaustive check of that refinement relation;
- verification of unsafe code and atomic-memory ordering; and
- formalized CUDA/transport semantics instead of backend assumptions.

Until that work exists, design and PR language must say "conforms under checked
traces and stated backend assumptions", never "the implementation is proved by
TLA+."

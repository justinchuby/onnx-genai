# TLA+ Formal Specifications

Executable, bounded models for the concurrency contracts in the memory and
distributed-runtime designs. They are safety models first: each claim below
corresponds to an invariant checked by TLC.

TLC proves the model, not the implementation. Implementations must also satisfy
the normative [refinement contract](./REFINEMENT.md), emit lossless protocol
traces in conformance tests, and pass an independent replay checker.

## Specifications

### `PressureProtocol.tla`

Models multiple variable-sized `PressureTicket`s per device, atomic grant
charging, claim, cancellation and timeout races, configuration-generation
invalidation, reclaim, and capacity already consumed by fixed non-reclaimable
allocations.

Checked invariants:

- `CapacityConserved`: free, reclaimable, reserved, and claimed byte extents sum
  with fixed charges to the configured capacity.
- `GrantedIsChargedExactly`: every grant owns its exact requested extent.
- `ClaimedIsOwnedExactly` and `ClaimedAtMostOnce`: claim transfers the exact
  reservation once.
- `TerminalHasNoAllocation`: cancellation, timeout, and completion cannot leak
  an allocation.
- `PendingUsesCurrentGeneration`: reconfiguration leaves no stale pending
  request.

The model does not assert unconditional eventual satisfaction. Priority/FIFO
arbitration and bounded aging are implementation scheduler obligations tested
by deterministic conformance campaigns.

### `CollectiveOrdering.tla`

Models overlapping executions across overlapping communicator groups. Runtime
slots are lexicographic `(ExecutionId, CommSequenceId)` pairs. Ranks and groups
advance independently while coordinator admission/skip decisions and each
group's submit sequencer prevent transport-order divergence.

Checked invariants:

- `GroupMembershipValid`: every frozen group is a non-empty world-rank subset.
- `DecisionPrefixIsFrozen`: coordinator decisions are monotonic.
- `SubmittedOnlyAdmitted`: skipped or undecided executions never reach the
  transport.
- `NoDuplicateOrReorder`: every rank-group log is strictly increasing.
- `GroupRankLogsCompatible`: members of one group remain compatible prefixes;
  different groups have no artificial global enqueue order.
- `LocalCompletionBounded`: rank-local completion cannot pass local submission.
- `NonMembersRemainUntouched`: non-members never consume a group slot.
- `AbortFreezesSubmission`: abort stops new transport work while submitted
  operations may still quiesce.

The production plan validator must supply identical ordered membership and
sequence metadata on every rank; the refinement checker verifies its hash.

### `BufferOwnership.tla`

Models read and write allocation leases retained by the backend registry across
handle detach, successful completion, abort request, abort quiescence, and
physical free.

Checked invariants:

- `NoConflictingActiveLeases`: readers may alias, but a writer excludes every
  other reader and writer of its allocation.
- `ActiveIsRegistryOwned`: every submitted or aborting operation remains rooted
  in the backend registry.
- `DetachedActiveIsStillOwned`: dropping a handle cannot release leases.
- `FreedHasNoLease`: physical free requires all read and write leases to end.
- `TerminalReleased`: registry release occurs only at terminal transport
  outcome.
- `ActiveGenerationsMatch`: allocator reuse cannot occur under an active read or
  write lease.

### `KvAdmission.tla`

Models continuous-batching admission against one shared, fixed-size KV cache
pool. Each resident request grows its KV extent monotonically through chunked
prefill and committed decode steps, and releases the whole extent only at
completion, so requests acquire incrementally and free once.

The modeled rule is: grant KV to a request only when its outstanding need still
fits in the currently free pool. The rule is stated over need and free extent
alone, so it does not depend on grant size or on the order in which the
scheduler visits requests.

Checked invariants:

- `CapacityRespected`: charged extents never exceed the configured pool.
- `ExtentsConserved`: held plus outstanding need reconstructs each request's
  peak footprint.
- `NoOverCharge`: no request is charged beyond the KV it will ever need.
- `TerminalReleasesAll`: completion returns every extent; a finished workload
  leaves an empty pool.
- `ResidentCanDrain`: a resident request's outstanding need always fits the
  pool once its own held extent is counted.
- `ProgressPossible`: in every reachable non-terminal state some unfinished
  request can still be driven to completion out of the free pool. This is the
  deadlock-freedom obligation; resident requests can never strand KV against
  each other.

`KvAdmissionUnguarded.cfg` is a negative model that disables the guard and must
violate `ProgressPossible`. `check.sh` asserts that violation, so a model that
stops catching its own counterexample fails the run.

The module does not assert fairness between eligible requests. Batch shaping,
priority ordering, and arrival timing are scheduler obligations covered by
deterministic conformance campaigns.

### `NodeFailure.tla`

Models quiescence after a node fault. A fault is not a global time cut: work
already enqueued on a surviving node keeps draining, while the failed node
stops accepting and stops executing at the fault instant.

Checked invariants:

- `FailedNodeStopsAtFault`: no operation spanning the failed node executes
  after the fault, however early it was submitted. Filtering purely on
  submission time violates this and lets a dead node run queued work.
- `NoCompletionAfterFailure`: an operation spanning the failed node never
  reports a completion later than the fault instant.
- `SurvivorWorkIsNotDiscarded`: work confined to surviving nodes is never
  discarded merely because it starts after the fault.
- `QuiescenceBoundedByDeadline` and `QuiescenceAfterFault`: quiescence is
  whichever comes first, surviving work draining or the coordinator closing the
  epoch at `FaultAt + QuiesceTimeout`, and never precedes the fault.
- `SettledExactlyOnce`: every operation reaches one terminal classification.

`NodeFailureUnguarded.cfg` is a negative model that lets the failed node keep
executing queued work and must violate `FailedNodeStopsAtFault`.

The module does not assert failure-detection latency or replanning. Heartbeat
policy and epoch handoff are covered by deterministic conformance campaigns.

### `CoResidency.tla`

Models several models served from one device. Each model occupies its weights
plus a preallocated KV arena while resident, so making room for one means
evicting others, and the question is not whether a single request fits but
whether every model can still be made resident at all.

Checked invariants:

- `CapacityRespected`: the device never holds more than it has.
- `ProgressPossible`: every model that still has work can be made resident,
  either already, or because it fits, or because evicting unpinned models
  would free enough room. Admission is closed against configurations where
  that fails, which is the residency analogue of `KvAdmission`.
- `NoWastedResidency`: a model is only evicted once it has retired a request
  since it was loaded. An eviction discards the KV arena, so evicting earlier
  throws the work away and the device can load and evict a model forever
  without ever finishing it. Gating on merely having run a batch is not
  enough; the guard has to be a retired request.
- `PinnedStaysResident` and `ServedOnlyWhenResident`: pinned models are never
  evicted, and the device only computes for a model it is holding.

`CoResidencyUnguarded.cfg` is a negative model that allows eviction before a
model has retired anything and must violate `NoWastedResidency`.

The module does not assert which model the device serves next, or how
transfers overlap compute. Those are scheduler obligations covered by
deterministic conformance campaigns.

## Running

Install [TLA+ tools](https://github.com/tlaplus/tlaplus/releases), then run from
this directory. CI must pin the jar artifact and verify its checksum rather than
downloading an unversioned latest release.

```bash
TLA2TOOLS_JAR=/path/to/tla2tools.jar ./check.sh
```

`JAVA_BIN` and `TLC_WORKERS` may override the Java executable and worker count.
The script gives every model a distinct temporary metadata directory and fails
on the first parse, invariant, or model-checking error.

The checked configurations are deliberately finite and exhaustive. Increasing
constants is useful, but does not replace implementation trace conformance.

## Design Context

- `docs/memory/MEMORY_ARCHITECTURE.md` section 5.3.1 (pressure protocol)
- `docs/distributed/DISTRIBUTED_RUNTIME.md` sections 3.1 and 3.2.1 (completion and ordering)
- `docs/distributed/DISTRIBUTED_RUNTIME.md` section 8.1 (rank-local DAG scheduling)
- `docs/memory/MEMORY_ARCHITECTURE.md` KV cache pool sizing and admission
- `docs/distributed/DISTRIBUTED_RUNTIME.md` node failure, abort, and quiescence
- `docs/memory/MEMORY_ARCHITECTURE.md` model residency and eviction
- `REFINEMENT.md` (implementation linearization and verification gates)

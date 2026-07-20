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

- `docs/MEMORY_ARCHITECTURE.md` section 5.3.1 (pressure protocol)
- `docs/DISTRIBUTED_RUNTIME.md` sections 3.1 and 3.2.1 (completion and ordering)
- `docs/DISTRIBUTED_RUNTIME.md` section 8.1 (rank-local DAG scheduling)
- `REFINEMENT.md` (implementation linearization and verification gates)

# TLA+ Formal Specifications

Executable, bounded models for the concurrency contracts in the memory and
distributed-runtime designs. They are safety models first: each README claim
below corresponds to an invariant checked by TLC. Liveness is asserted only
where the specification includes action-specific fairness and the modeled
environment can actually make progress.

These models do not prove the implementation correct. The implementation must
preserve the modeled state transitions, linearization points, and ownership
boundaries.

## Specifications

### `PressureProtocol.tla`

Models the `PressureTicket` lifecycle, atomic grant charging, cancellation and
timeout races, configuration-generation invalidation, reclaim, and claim.

Checked invariants:

- `CapacityConserved`: free, reclaimable, reserved, and claimed pages sum to
  the configured capacity.
- `GrantedIsCharged`: a published grant always has an existing ledger charge.
- `ClaimedExactlyOnce`: a grant can be claimed at most once.
- `TerminalHasNoReservation`: cancellation, timeout, and completion cannot
  leak a reservation.
- `PendingUsesCurrentGeneration`: reconfiguration leaves no stale pending
  request.

The model intentionally does not assert unconditional deadlock freedom or
eventual satisfaction. A request cannot progress if the environment retains
all capacity and exposes no reclaim or release action.

### `CollectiveOrdering.tla`

Models one communicator group. Runtime slots are the lexicographic sequence of
`(ExecutionId, CommSequenceId)` pairs across overlapping executions. Ranks may
advance and complete independently, while coordinator admission/skip decisions
and the submit sequencer prevent transport-order divergence.

Checked invariants:

- `DecisionPrefixIsFrozen`: execution admission decisions are monotonic and
  coordinator ordered.
- `SubmittedOnlyAdmitted`: skipped or undecided executions never reach the
  transport.
- `NoDuplicateOrReorder`: each rank submits a strictly increasing slot stream.
- `RankLogsCompatible`: rank logs may differ in length but remain compatible
  prefixes of one canonical order.
- `LocalCompletionBounded`: completion is rank-local and cannot pass local
  submission.
- `AbortFreezesSubmission`: abort stops new transport work while already
  submitted operations may still quiesce.

Instantiate the model independently for each frozen communicator group. The
production plan validator is responsible for constructing identical group
membership and sequence metadata on every rank.

### `BufferOwnership.tla`

Models the backend registry lease retained for a submitted operation, including
detached user handles, successful completion, abort request, abort quiescence,
and physical buffer free.

Checked invariants:

- `ExclusiveActiveLease`: an exclusive workspace has at most one active user.
- `ActiveIsRegistryOwned`: every submitted or aborting operation remains rooted
  in the backend registry.
- `DetachedActiveIsStillOwned`: dropping a handle cannot release the lease.
- `FreedHasNoOwner`: physical free requires all transport leases to be gone.
- `TerminalReleased`: registry release occurs only at a terminal transport
  outcome.
- `ActiveGenerationMatches`: a buffer generation cannot advance under an
  active operation.

Ready operations are allowed to wait for a shared buffer. The previous
`NoPendingReuse` property incorrectly treated such legal waiters as reuse.

## Running

Install [TLA+ tools](https://github.com/tlaplus/tlaplus/releases), then run from
this directory:

```bash
java -jar /path/to/tla2tools.jar -config PressureProtocol.cfg PressureProtocol.tla
java -jar /path/to/tla2tools.jar -config CollectiveOrdering.cfg CollectiveOrdering.tla
java -jar /path/to/tla2tools.jar -config BufferOwnership.cfg BufferOwnership.tla
```

The checked configurations are deliberately small enough for exhaustive local
runs. Increase the constants for deeper validation when changing a protocol.

## Design Context

- `docs/MEMORY_ARCHITECTURE.md` section 5.3.1 (pressure protocol)
- `docs/DISTRIBUTED_RUNTIME.md` sections 3.1 and 3.2.1 (completion and ordering)
- `docs/DISTRIBUTED_RUNTIME.md` section 8.1 (rank-local DAG scheduling)

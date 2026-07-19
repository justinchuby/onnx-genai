--------------------------- MODULE BufferOwnership ---------------------------
\* Model of transport-held buffer leases and CommHandle lifetime.
\*
\* A submitted operation is retained by the backend registry until transport
\* completion or abort quiescence. Dropping/detaching the user-visible handle
\* does not release its lease. Ready operations may legally wait for a shared
\* buffer; they are not evidence of premature reuse.
\*
\* The model uses one exclusive workspace buffer per operation. Production
\* read-only aliasing can be added as shared leases, but must preserve the same
\* no-free/no-write rule while any registry lease is active.

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    NumBuffers,
    NumOperations

ASSUME /\ NumBuffers > 0
       /\ NumOperations > 0

Buffers == 1..NumBuffers
Operations == 1..NumOperations
OperationStates == {"ready", "submitted", "aborting", "terminal"}
Outcomes == {"none", "ok", "error"}

\* A deterministic assignment keeps the model self-contained and guarantees
\* sharing whenever NumOperations > NumBuffers.
BufferOf(op) == ((op - 1) % NumBuffers) + 1

VARIABLES
    operationState,
    handleAttached,
    registryOwned,
    outcome,
    freed,
    leaseGeneration,
    operationGeneration

vars ==
    <<operationState, handleAttached, registryOwned, outcome, freed,
      leaseGeneration, operationGeneration>>

TypeOK ==
    /\ operationState \in [Operations -> OperationStates]
    /\ handleAttached \in [Operations -> BOOLEAN]
    /\ registryOwned \in [Operations -> BOOLEAN]
    /\ outcome \in [Operations -> Outcomes]
    /\ freed \subseteq Buffers
    /\ leaseGeneration \in [Buffers -> Nat]
    /\ operationGeneration \in [Operations -> Nat]

Init ==
    /\ operationState = [op \in Operations |-> "ready"]
    /\ handleAttached = [op \in Operations |-> FALSE]
    /\ registryOwned = [op \in Operations |-> FALSE]
    /\ outcome = [op \in Operations |-> "none"]
    /\ freed = {}
    /\ leaseGeneration = [b \in Buffers |-> 0]
    /\ operationGeneration = [op \in Operations |-> 0]

Active(op) ==
    operationState[op] \in {"submitted", "aborting"}

BufferAvailable(op) ==
    /\ BufferOf(op) \notin freed
    /\ \A other \in Operations:
        BufferOf(other) = BufferOf(op) => ~Active(other)

Submit(op) ==
    /\ operationState[op] = "ready"
    /\ BufferAvailable(op)
    /\ operationState' =
        [operationState EXCEPT ![op] = "submitted"]
    /\ handleAttached' =
        [handleAttached EXCEPT ![op] = TRUE]
    /\ registryOwned' =
        [registryOwned EXCEPT ![op] = TRUE]
    /\ operationGeneration' =
        [operationGeneration EXCEPT
            ![op] = leaseGeneration[BufferOf(op)]]
    /\ UNCHANGED <<outcome, freed, leaseGeneration>>

\* Dropping the handle detaches observation only. Backend ownership and the
\* buffer lease remain live.
Detach(op) ==
    /\ Active(op)
    /\ handleAttached[op]
    /\ handleAttached' =
        [handleAttached EXCEPT ![op] = FALSE]
    /\ UNCHANGED
        <<operationState, registryOwned, outcome, freed,
          leaseGeneration, operationGeneration>>

CompleteSuccess(op) ==
    /\ operationState[op] = "submitted"
    /\ registryOwned[op]
    /\ operationState' =
        [operationState EXCEPT ![op] = "terminal"]
    /\ registryOwned' =
        [registryOwned EXCEPT ![op] = FALSE]
    /\ outcome' =
        [outcome EXCEPT ![op] = "ok"]
    /\ leaseGeneration' =
        [leaseGeneration EXCEPT ![BufferOf(op)] = @ + 1]
    /\ UNCHANGED <<handleAttached, freed, operationGeneration>>

\* Abort request is not terminal completion. The lease remains owned until the
\* transport reports quiescence.
BeginAbort(op) ==
    /\ operationState[op] = "submitted"
    /\ registryOwned[op]
    /\ operationState' =
        [operationState EXCEPT ![op] = "aborting"]
    /\ UNCHANGED
        <<handleAttached, registryOwned, outcome, freed,
          leaseGeneration, operationGeneration>>

QuiesceAbort(op) ==
    /\ operationState[op] = "aborting"
    /\ registryOwned[op]
    /\ operationState' =
        [operationState EXCEPT ![op] = "terminal"]
    /\ registryOwned' =
        [registryOwned EXCEPT ![op] = FALSE]
    /\ outcome' =
        [outcome EXCEPT ![op] = "error"]
    /\ leaseGeneration' =
        [leaseGeneration EXCEPT ![BufferOf(op)] = @ + 1]
    /\ UNCHANGED <<handleAttached, freed, operationGeneration>>

FreeBuffer(b) ==
    /\ b \notin freed
    /\ \A op \in Operations:
        BufferOf(op) = b => ~registryOwned[op]
    /\ freed' = freed \union {b}
    /\ UNCHANGED
        <<operationState, handleAttached, registryOwned, outcome,
          leaseGeneration, operationGeneration>>

Next ==
    \/ \E op \in Operations: Submit(op)
    \/ \E op \in Operations: Detach(op)
    \/ \E op \in Operations: CompleteSuccess(op)
    \/ \E op \in Operations: BeginAbort(op)
    \/ \E op \in Operations: QuiesceAbort(op)
    \/ \E b \in Buffers: FreeBuffer(b)

ExclusiveActiveLease ==
    \A b \in Buffers:
        Cardinality(
            {op \in Operations:
                /\ BufferOf(op) = b
                /\ Active(op)}
        ) <= 1

ActiveIsRegistryOwned ==
    \A op \in Operations:
        Active(op) => registryOwned[op]

DetachedActiveIsStillOwned ==
    \A op \in Operations:
        /\ Active(op)
        /\ ~handleAttached[op]
        => registryOwned[op]

FreedHasNoOwner ==
    \A b \in freed:
        \A op \in Operations:
            BufferOf(op) = b => ~registryOwned[op]

TerminalReleased ==
    \A op \in Operations:
        operationState[op] = "terminal"
            => /\ ~registryOwned[op]
               /\ outcome[op] \in {"ok", "error"}

ActiveGenerationMatches ==
    \A op \in Operations:
        Active(op) =>
            operationGeneration[op] = leaseGeneration[BufferOf(op)]

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ \A op \in Operations:
        WF_vars(CompleteSuccess(op) \/ BeginAbort(op))
    /\ \A op \in Operations: WF_vars(QuiesceAbort(op))

=============================================================================

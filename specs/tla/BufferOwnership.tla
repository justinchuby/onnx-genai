--------------------------- MODULE BufferOwnership ---------------------------
\* Refinement target for transport-held allocation leases.
\*
\* Each operation acquires a read buffer and a write buffer. Multiple readers
\* may alias, while any writer excludes all other readers and writers of that
\* buffer. The backend registry retains both leases until successful terminal
\* completion or abort quiescence. Detaching the user-visible handle does not
\* release either lease.

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    NumBuffers,
    NumOperations

ASSUME /\ NumBuffers > 1
       /\ NumOperations > 1

Buffers == 1..NumBuffers
Operations == 1..NumOperations
OperationStates == {"ready", "submitted", "aborting", "terminal"}
Outcomes == {"none", "ok", "error"}

\* This assignment creates both read sharing and read/write conflicts in the
\* checked configuration.
ReadBuffer(op) ==
    (((op - 1) \div 2) % NumBuffers) + 1

WriteBuffer(op) ==
    (op % NumBuffers) + 1

VARIABLES
    operationState,
    handleAttached,
    registryOwned,
    outcome,
    freed,
    bufferGeneration,
    observedReadGeneration,
    observedWriteGeneration

vars ==
    <<operationState, handleAttached, registryOwned, outcome, freed,
      bufferGeneration, observedReadGeneration, observedWriteGeneration>>

TypeOK ==
    /\ operationState \in [Operations -> OperationStates]
    /\ handleAttached \in [Operations -> BOOLEAN]
    /\ registryOwned \in [Operations -> BOOLEAN]
    /\ outcome \in [Operations -> Outcomes]
    /\ freed \subseteq Buffers
    /\ bufferGeneration \in [Buffers -> Nat]
    /\ observedReadGeneration \in [Operations -> Nat]
    /\ observedWriteGeneration \in [Operations -> Nat]

Init ==
    /\ operationState =
        [op \in Operations |-> "ready"]
    /\ handleAttached =
        [op \in Operations |-> FALSE]
    /\ registryOwned =
        [op \in Operations |-> FALSE]
    /\ outcome =
        [op \in Operations |-> "none"]
    /\ freed = {}
    /\ bufferGeneration =
        [buffer \in Buffers |-> 0]
    /\ observedReadGeneration =
        [op \in Operations |-> 0]
    /\ observedWriteGeneration =
        [op \in Operations |-> 0]

Active(op) ==
    operationState[op] \in {"submitted", "aborting"}

Conflicts(op, other) ==
    \/ WriteBuffer(op) = WriteBuffer(other)
    \/ WriteBuffer(op) = ReadBuffer(other)
    \/ ReadBuffer(op) = WriteBuffer(other)

BuffersAvailable(op) ==
    /\ ReadBuffer(op) \notin freed
    /\ WriteBuffer(op) \notin freed
    /\ \A other \in Operations:
        Active(other) => ~Conflicts(op, other)

Submit(op) ==
    /\ operationState[op] = "ready"
    /\ BuffersAvailable(op)
    /\ operationState' =
        [operationState EXCEPT ![op] = "submitted"]
    /\ handleAttached' =
        [handleAttached EXCEPT ![op] = TRUE]
    /\ registryOwned' =
        [registryOwned EXCEPT ![op] = TRUE]
    /\ observedReadGeneration' =
        [observedReadGeneration EXCEPT
            ![op] = bufferGeneration[ReadBuffer(op)]]
    /\ observedWriteGeneration' =
        [observedWriteGeneration EXCEPT
            ![op] = bufferGeneration[WriteBuffer(op)]]
    /\ UNCHANGED <<outcome, freed, bufferGeneration>>

Detach(op) ==
    /\ Active(op)
    /\ handleAttached[op]
    /\ handleAttached' =
        [handleAttached EXCEPT ![op] = FALSE]
    /\ UNCHANGED
        <<operationState, registryOwned, outcome, freed,
          bufferGeneration, observedReadGeneration,
          observedWriteGeneration>>

CompleteSuccess(op) ==
    /\ operationState[op] = "submitted"
    /\ registryOwned[op]
    /\ operationState' =
        [operationState EXCEPT ![op] = "terminal"]
    /\ registryOwned' =
        [registryOwned EXCEPT ![op] = FALSE]
    /\ outcome' =
        [outcome EXCEPT ![op] = "ok"]
    /\ bufferGeneration' =
        [bufferGeneration EXCEPT
            ![WriteBuffer(op)] = @ + 1]
    /\ UNCHANGED
        <<handleAttached, freed, observedReadGeneration,
          observedWriteGeneration>>

\* Abort request is not terminal completion. Both leases remain active.
BeginAbort(op) ==
    /\ operationState[op] = "submitted"
    /\ registryOwned[op]
    /\ operationState' =
        [operationState EXCEPT ![op] = "aborting"]
    /\ UNCHANGED
        <<handleAttached, registryOwned, outcome, freed,
          bufferGeneration, observedReadGeneration,
          observedWriteGeneration>>

QuiesceAbort(op) ==
    /\ operationState[op] = "aborting"
    /\ registryOwned[op]
    /\ operationState' =
        [operationState EXCEPT ![op] = "terminal"]
    /\ registryOwned' =
        [registryOwned EXCEPT ![op] = FALSE]
    /\ outcome' =
        [outcome EXCEPT ![op] = "error"]
    /\ bufferGeneration' =
        [bufferGeneration EXCEPT
            ![WriteBuffer(op)] = @ + 1]
    /\ UNCHANGED
        <<handleAttached, freed, observedReadGeneration,
          observedWriteGeneration>>

FreeBuffer(buffer) ==
    /\ buffer \notin freed
    /\ \A op \in Operations:
        /\ Active(op)
        => /\ ReadBuffer(op) # buffer
           /\ WriteBuffer(op) # buffer
    /\ freed' = freed \union {buffer}
    /\ UNCHANGED
        <<operationState, handleAttached, registryOwned, outcome,
          bufferGeneration, observedReadGeneration,
          observedWriteGeneration>>

Next ==
    \/ \E op \in Operations: Submit(op)
    \/ \E op \in Operations: Detach(op)
    \/ \E op \in Operations: CompleteSuccess(op)
    \/ \E op \in Operations: BeginAbort(op)
    \/ \E op \in Operations: QuiesceAbort(op)
    \/ \E buffer \in Buffers: FreeBuffer(buffer)

NoConflictingActiveLeases ==
    \A op, other \in Operations:
        /\ op # other
        /\ Active(op)
        /\ Active(other)
        => ~Conflicts(op, other)

ActiveIsRegistryOwned ==
    \A op \in Operations:
        Active(op) => registryOwned[op]

DetachedActiveIsStillOwned ==
    \A op \in Operations:
        /\ Active(op)
        /\ ~handleAttached[op]
        => registryOwned[op]

FreedHasNoLease ==
    \A buffer \in freed:
        \A op \in Operations:
            Active(op) =>
                /\ ReadBuffer(op) # buffer
                /\ WriteBuffer(op) # buffer

TerminalReleased ==
    \A op \in Operations:
        operationState[op] = "terminal"
            => /\ ~registryOwned[op]
               /\ outcome[op] \in {"ok", "error"}

\* Neither a reader nor a writer may observe its buffer generation change
\* while its transport operation remains active.
ActiveGenerationsMatch ==
    \A op \in Operations:
        Active(op) =>
            /\ observedReadGeneration[op]
                = bufferGeneration[ReadBuffer(op)]
            /\ observedWriteGeneration[op]
                = bufferGeneration[WriteBuffer(op)]

BufferAssignmentValid ==
    \A op \in Operations:
        /\ ReadBuffer(op) \in Buffers
        /\ WriteBuffer(op) \in Buffers

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ \A op \in Operations:
        WF_vars(CompleteSuccess(op) \/ BeginAbort(op))
    /\ \A op \in Operations:
        WF_vars(QuiesceAbort(op))

=============================================================================

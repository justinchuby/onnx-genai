--------------------------- MODULE BufferOwnership ---------------------------
\* TLA+ specification for CommHandle buffer ownership state machine.
\* Proves no buffer is reused/freed before its communication completes,
\* under all possible DAG scheduling interleavings.
\*
\* Model: A set of buffers, each used by communication operations.
\* The DagScheduler may launch multiple operations concurrently.
\* A buffer transitions: Free → InUse → PendingComm → Free
\*
\* Verifies:
\*   1. No buffer is in two concurrent operations
\*   2. No buffer is freed/reused while PendingComm
\*   3. Every buffer eventually returns to Free (liveness)

EXTENDS Naturals, FiniteSets

CONSTANTS
    Buffers,        \* Set of buffer IDs
    Operations      \* Set of operation IDs (communication steps)

VARIABLES
    bufState,       \* bufState[b] \in {"free", "in_use", "pending_comm"}
    opState,        \* opState[op] \in {"ready", "launched", "complete"}
    opBuffer,       \* opBuffer[op] = buffer used by this operation (static assignment)
    handleState     \* handleState[op] \in {"none", "active", "signaled"}

vars == <<bufState, opState, opBuffer, handleState>>

TypeOK ==
    /\ \A b \in Buffers: bufState[b] \in {"free", "in_use", "pending_comm"}
    /\ \A op \in Operations: opState[op] \in {"ready", "launched", "complete"}
    /\ \A op \in Operations: handleState[op] \in {"none", "active", "signaled"}

Init ==
    /\ bufState = [b \in Buffers |-> "free"]
    /\ opState = [op \in Operations |-> "ready"]
    /\ opBuffer = [op \in Operations |-> CHOOSE b \in Buffers : TRUE]  \* model config
    /\ handleState = [op \in Operations |-> "none"]

---------------------------------------------------------------------------
\* ACTION: Scheduler writes data into buffer (preparing for comm)
PrepareBuffer(op) ==
    /\ opState[op] = "ready"
    /\ bufState[opBuffer[op]] = "free"
    /\ bufState' = [bufState EXCEPT ![opBuffer[op]] = "in_use"]
    /\ opState' = [opState EXCEPT ![op] = "launched"]
    /\ handleState' = [handleState EXCEPT ![op] = "active"]
    /\ UNCHANGED opBuffer

\* ACTION: Communication is enqueued (buffer transitions to pending_comm)
\* Buffer is now owned by the transport — cannot be touched.
EnqueueComm(op) ==
    /\ opState[op] = "launched"
    /\ bufState[opBuffer[op]] = "in_use"
    /\ bufState' = [bufState EXCEPT ![opBuffer[op]] = "pending_comm"]
    /\ UNCHANGED <<opState, opBuffer, handleState>>

\* ACTION: Communication completes (CommHandle signals)
\* Buffer returns to free. Only happens when transport is done.
CommComplete(op) ==
    /\ handleState[op] = "active"
    /\ bufState[opBuffer[op]] = "pending_comm"
    /\ handleState' = [handleState EXCEPT ![op] = "signaled"]
    /\ bufState' = [bufState EXCEPT ![opBuffer[op]] = "free"]
    /\ opState' = [opState EXCEPT ![op] = "complete"]
    /\ UNCHANGED opBuffer

---------------------------------------------------------------------------
Next ==
    \/ \E op \in Operations: PrepareBuffer(op)
    \/ \E op \in Operations: EnqueueComm(op)
    \/ \E op \in Operations: CommComplete(op)

---------------------------------------------------------------------------
\* PROPERTIES

\* Safety: A buffer in "pending_comm" is never accessed by another operation.
\* (No other op can PrepareBuffer on the same buffer while it's pending.)
NoPendingReuse == \A b \in Buffers:
    bufState[b] = "pending_comm" =>
        ~(\E op \in Operations:
            /\ opBuffer[op] = b
            /\ opState[op] = "ready")

\* Safety: A buffer cannot be in_use by two operations simultaneously.
NoDoubleUse == \A b \in Buffers:
    Cardinality({op \in Operations:
        opBuffer[op] = b /\ opState[op] = "launched"}) <= 1

\* Safety: Buffer is only freed AFTER CommHandle signals.
FreeOnlyAfterSignal == \A op \in Operations:
    (opState[op] = "complete") => (handleState[op] = "signaled")

\* Liveness: All operations eventually complete.
AllOpsComplete == <>(\A op \in Operations: opState[op] = "complete")

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

=============================================================================

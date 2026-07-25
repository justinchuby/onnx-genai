----------------------------- MODULE NodeFailure -----------------------------
\* Refinement target for node-failure quiescence in the distributed runtime.
\*
\* A node fault is not a global time cut. Work already enqueued on a surviving
\* node keeps draining, which is what `CollectiveOrdering!AbortFreezesSubmission`
\* allows. Work on the failed node does not: the device stops accepting and
\* stops executing at the fault instant, so anything it had only queued never
\* runs, and anything it had started never finishes.
\*
\* Surviving ranks do not observe the fault instantly. They keep running until
\* the coordinator aborts the topology epoch, so quiescence is bounded by an
\* explicit abort deadline rather than by the planned finish time of operations
\* that can never complete.
\*
\* This is a safety model. Failure detection latency, heartbeat policy, and
\* replanning are covered by deterministic conformance campaigns.

EXTENDS Naturals, FiniteSets

CONSTANTS
    NumOperations,
    FaultAt,
    QuiesceTimeout,
    MaxTime,
    EnforceFailedNodeStop

Operations == 1..NumOperations

\* The finite model mixes operations local to the surviving node with
\* operations that span the failed node.
SpansFailedNode(operation) == operation % 2 = 0

\* Submission, start, and finish are plan-relative and fixed by the model.
\* Operations share a lane, so a later operation starts only after the previous
\* one finishes. Every operation is submitted at time 0.
SubmittedAt(operation) == 0
StartAt(operation) == (operation - 1) * 2
FinishAt(operation) == StartAt(operation) + 2

AbortDeadline == FaultAt + QuiesceTimeout

ASSUME /\ NumOperations > 1
       /\ FaultAt > 0
       /\ QuiesceTimeout > 0
       /\ EnforceFailedNodeStop \in BOOLEAN
       /\ MaxTime >= AbortDeadline
       /\ MaxTime >= FinishAt(NumOperations)

VARIABLES
    ran,        \* operations the model has let execute
    dropped,    \* operations the fault removed before they could execute
    aborted

vars == <<ran, dropped, aborted>>

Settled == ran \cup dropped

\* An operation may execute only if the fault does not forbid it.
MayRun(operation) ==
    \/ ~EnforceFailedNodeStop
    \/ ~SpansFailedNode(operation)
    \/ StartAt(operation) < FaultAt

\* An operation that spans the failed node is truncated at the fault: it was in
\* flight, but the device disappeared before it could finish.
EffectiveFinish(operation) ==
    IF SpansFailedNode(operation) THEN FaultAt ELSE FinishAt(operation)

RECURSIVE DrainedUpTo(_)
DrainedUpTo(n) ==
    IF n = 0
    THEN FaultAt
    ELSE LET rest == DrainedUpTo(n - 1)
         IN IF n \in ran /\ EffectiveFinish(n) > rest
            THEN EffectiveFinish(n)
            ELSE rest

Drained == DrainedUpTo(NumOperations)

\* Quiescence is whichever comes first: surviving work draining, or the
\* coordinator closing the epoch at the abort deadline. Work still running at
\* the deadline is aborted, not waited for.
QuiescedAt ==
    IF Drained < AbortDeadline THEN Drained ELSE AbortDeadline

TypeOK ==
    /\ ran \subseteq Operations
    /\ dropped \subseteq Operations
    /\ ran \cap dropped = {}
    /\ aborted \in BOOLEAN
    /\ QuiescedAt \in 0..MaxTime

Init ==
    /\ ran = {}
    /\ dropped = {}
    /\ aborted = FALSE

\* Lane order: an operation executes only after every earlier one settled.
Ready(operation) == \A earlier \in 1..(operation - 1): earlier \in Settled

Run(operation) ==
    /\ ~aborted
    /\ operation \notin Settled
    /\ Ready(operation)
    /\ MayRun(operation)
    /\ ran' = ran \cup {operation}
    /\ UNCHANGED <<dropped, aborted>>

Drop(operation) ==
    /\ operation \notin Settled
    /\ Ready(operation)
    /\ ~MayRun(operation)
    /\ dropped' = dropped \cup {operation}
    /\ UNCHANGED <<ran, aborted>>

\* The coordinator closes the epoch once the deadline is reached.
Abort ==
    /\ ~aborted
    /\ aborted' = TRUE
    /\ UNCHANGED <<ran, dropped>>

Next ==
    \/ \E operation \in Operations: Run(operation) \/ Drop(operation)
    \/ Abort

Spec == Init /\ [][Next]_vars

\* ---------------------------------------------------------------------------
\* Invariants
\* ---------------------------------------------------------------------------

\* The failed node stops at the fault instant. Nothing it had merely queued may
\* still execute afterwards. This is the property the implementation violated:
\* filtering only on submission time let a dead node run queued work.
FailedNodeStopsAtFault ==
    \A operation \in ran:
        SpansFailedNode(operation) => StartAt(operation) < FaultAt

\* Work confined to surviving nodes is never discarded just because it starts
\* after the fault; abort freezes new submission, not draining.
SurvivorWorkIsNotDiscarded ==
    \A operation \in dropped: SpansFailedNode(operation)

\* An operation that spans the failed node never reports a completion later
\* than the fault: the device was gone.
NoCompletionAfterFailure ==
    \A operation \in ran:
        SpansFailedNode(operation) => EffectiveFinish(operation) <= FaultAt

\* Quiescence is bounded by the abort deadline, so a fault campaign can never
\* report a quiescence time derived from work that can never complete.
QuiescenceBoundedByDeadline == QuiescedAt <= AbortDeadline

\* Quiescence never precedes the fault.
QuiescenceAfterFault == QuiescedAt >= FaultAt

\* Every operation reaches exactly one terminal classification.
SettledExactlyOnce == ran \cap dropped = {}

=============================================================================

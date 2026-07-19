-------------------------- MODULE CollectiveOrdering --------------------------
\* Model of one communicator's plan-time collective sequence and runtime
\* per-group submit sequencer.
\*
\* Each slot is a (ExecutionId, CommSequenceId) instance. Multiple executions
\* may overlap, but every rank traverses the same lexicographic slot order.
\* The coordinator admits or skips an execution before any rank can submit its
\* slots. Once an admitted execution has been submitted, failure transitions
\* the group to abort; it cannot be retroactively skipped.
\*
\* Transport completion is intentionally rank-local. One rank may observe its
\* own completion while peers remain in flight.

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    NumRanks,
    NumExecutions,
    SequenceLength

ASSUME /\ NumRanks > 1
       /\ NumExecutions > 0
       /\ SequenceLength > 0

Ranks == 1..NumRanks
Executions == 1..NumExecutions
TotalSlots == NumExecutions * SequenceLength
Slots == 1..TotalSlots

ExecutionOf(slot) == ((slot - 1) \div SequenceLength) + 1
SequenceOf(slot) == ((slot - 1) % SequenceLength) + 1

DecisionStates == {"undecided", "admitted", "skipped"}

VARIABLES
    decision,
    decidedPrefix,
    cursor,
    transportLog,
    localCompleted,
    aborted,
    logAtAbort

vars ==
    <<decision, decidedPrefix, cursor, transportLog,
      localCompleted, aborted, logAtAbort>>

TypeOK ==
    /\ decision \in [Executions -> DecisionStates]
    /\ decidedPrefix \in 0..NumExecutions
    /\ cursor \in [Ranks -> 0..TotalSlots]
    /\ transportLog \in [Ranks -> Seq(Slots)]
    /\ localCompleted \in [Ranks -> 0..TotalSlots]
    /\ aborted \in BOOLEAN
    /\ logAtAbort \in [Ranks -> Seq(Slots)]

Init ==
    /\ decision = [e \in Executions |-> "undecided"]
    /\ decidedPrefix = 0
    /\ cursor = [r \in Ranks |-> 0]
    /\ transportLog = [r \in Ranks |-> <<>>]
    /\ localCompleted = [r \in Ranks |-> 0]
    /\ aborted = FALSE
    /\ logAtAbort = [r \in Ranks |-> <<>>]

\* Coordinator decisions are monotonic and made in ExecutionId order.
AdmitNext ==
    /\ ~aborted
    /\ decidedPrefix < NumExecutions
    /\ LET execution == decidedPrefix + 1
       IN decision' = [decision EXCEPT ![execution] = "admitted"]
    /\ decidedPrefix' = decidedPrefix + 1
    /\ UNCHANGED
        <<cursor, transportLog, localCompleted, aborted, logAtAbort>>

SkipNext ==
    /\ ~aborted
    /\ decidedPrefix < NumExecutions
    /\ LET execution == decidedPrefix + 1
       IN decision' = [decision EXCEPT ![execution] = "skipped"]
    /\ decidedPrefix' = decidedPrefix + 1
    /\ UNCHANGED
        <<cursor, transportLog, localCompleted, aborted, logAtAbort>>

\* A rank can submit only its next slot. This is the no-tag backend sequencer:
\* rank skew is allowed, order divergence is not.
Submit(r) ==
    /\ ~aborted
    /\ cursor[r] < TotalSlots
    /\ LET slot == cursor[r] + 1
       IN /\ decision[ExecutionOf(slot)] = "admitted"
          /\ cursor' = [cursor EXCEPT ![r] = slot]
          /\ transportLog' =
              [transportLog EXCEPT ![r] = Append(@, slot)]
    /\ UNCHANGED
        <<decision, decidedPrefix, localCompleted, aborted, logAtAbort>>

\* A coordinated skip consumes the same sequence slots on every rank without
\* submitting anything to the transport.
ObserveSkip(r) ==
    /\ ~aborted
    /\ cursor[r] < TotalSlots
    /\ LET slot == cursor[r] + 1
       IN /\ decision[ExecutionOf(slot)] = "skipped"
          /\ cursor' = [cursor EXCEPT ![r] = slot]
    /\ UNCHANGED
        <<decision, decidedPrefix, transportLog,
          localCompleted, aborted, logAtAbort>>

\* Completion is local: polling one CommHandle advances only that rank.
CompleteLocal(r) ==
    /\ localCompleted[r] < Len(transportLog[r])
    /\ localCompleted' = [localCompleted EXCEPT ![r] = @ + 1]
    /\ UNCHANGED
        <<decision, decidedPrefix, cursor, transportLog, aborted, logAtAbort>>

Abort ==
    /\ ~aborted
    /\ aborted' = TRUE
    /\ logAtAbort' = transportLog
    /\ UNCHANGED
        <<decision, decidedPrefix, cursor, transportLog, localCompleted>>

Next ==
    \/ AdmitNext
    \/ SkipNext
    \/ \E r \in Ranks: Submit(r)
    \/ \E r \in Ranks: ObserveSkip(r)
    \/ \E r \in Ranks: CompleteLocal(r)
    \/ Abort

IsPrefix(a, b) ==
    /\ Len(a) <= Len(b)
    /\ SubSeq(b, 1, Len(a)) = a

StrictlyIncreasing(s) ==
    \A i, j \in 1..Len(s): i < j => s[i] < s[j]

DecisionPrefixIsFrozen ==
    /\ \A e \in 1..decidedPrefix: decision[e] # "undecided"
    /\ \A e \in (decidedPrefix + 1)..NumExecutions:
        decision[e] = "undecided"

SubmittedOnlyAdmitted ==
    \A r \in Ranks:
        \A i \in 1..Len(transportLog[r]):
            decision[ExecutionOf(transportLog[r][i])] = "admitted"

NoDuplicateOrReorder ==
    \A r \in Ranks: StrictlyIncreasing(transportLog[r])

\* Logs need not have equal length. Any two are compatible prefixes of the
\* same canonical admitted-slot stream.
RankLogsCompatible ==
    \A r1, r2 \in Ranks:
        \/ IsPrefix(transportLog[r1], transportLog[r2])
        \/ IsPrefix(transportLog[r2], transportLog[r1])

LocalCompletionBounded ==
    \A r \in Ranks:
        localCompleted[r] <= Len(transportLog[r])

CursorCoversSubmitted ==
    \A r \in Ranks:
        /\ Len(transportLog[r]) <= cursor[r]
        /\ \A i \in 1..Len(transportLog[r]):
            transportLog[r][i] <= cursor[r]

\* After abort, no further transport submission may occur. Local completion
\* remains enabled so the scheduler can quiesce already submitted work.
AbortFreezesSubmission ==
    aborted => transportLog = logAtAbort

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ \A r \in Ranks: WF_vars(CompleteLocal(r))

=============================================================================

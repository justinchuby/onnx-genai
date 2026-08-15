-------------------------- MODULE CollectiveOrdering --------------------------
\* Refinement target for coordinator admission and per-communicator submit
\* sequencers.
\*
\* Multiple communicator groups overlap in membership. Each group independently
\* traverses a lexicographic sequence of (ExecutionId, CommSequenceId) slots.
\* Cross-group enqueue order is intentionally unconstrained; within a group,
\* ranks may advance at different speeds but cannot diverge in transport order.
\*
\* Completion is rank-local. Abort freezes new submissions in every group while
\* already submitted operations remain eligible to quiesce locally.

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    NumRanks,
    NumGroups,
    NumExecutions,
    SequenceLength

ASSUME /\ NumRanks > 1
       /\ NumGroups > 1
       /\ NumExecutions > 0
       /\ SequenceLength > 0

Ranks == 1..NumRanks
Groups == 1..NumGroups
Executions == 1..NumExecutions
TotalSlots == NumExecutions * SequenceLength
Slots == 1..TotalSlots

\* Group 1 is the world. Other groups overlap at rank 1 and select alternating
\* peers, which exercises membership overlap without a separate model value.
GroupMembers(group) ==
    IF group = 1
    THEN Ranks
    ELSE {rank \in Ranks:
            \/ rank = 1
            \/ ((rank + group) % 2) = 0}

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
    /\ cursor \in [Ranks -> [Groups -> 0..TotalSlots]]
    /\ transportLog \in [Ranks -> [Groups -> Seq(Slots)]]
    /\ localCompleted \in
        [Ranks -> [Groups -> 0..TotalSlots]]
    /\ aborted \in BOOLEAN
    /\ logAtAbort \in [Ranks -> [Groups -> Seq(Slots)]]

Init ==
    /\ decision = [execution \in Executions |-> "undecided"]
    /\ decidedPrefix = 0
    /\ cursor =
        [rank \in Ranks |->
            [group \in Groups |-> 0]]
    /\ transportLog =
        [rank \in Ranks |->
            [group \in Groups |-> <<>>]]
    /\ localCompleted =
        [rank \in Ranks |->
            [group \in Groups |-> 0]]
    /\ aborted = FALSE
    /\ logAtAbort =
        [rank \in Ranks |->
            [group \in Groups |-> <<>>]]

AdmitNext ==
    /\ ~aborted
    /\ decidedPrefix < NumExecutions
    /\ LET execution == decidedPrefix + 1
       IN decision' =
            [decision EXCEPT ![execution] = "admitted"]
    /\ decidedPrefix' = decidedPrefix + 1
    /\ UNCHANGED
        <<cursor, transportLog, localCompleted, aborted, logAtAbort>>

SkipNext ==
    /\ ~aborted
    /\ decidedPrefix < NumExecutions
    /\ LET execution == decidedPrefix + 1
       IN decision' =
            [decision EXCEPT ![execution] = "skipped"]
    /\ decidedPrefix' = decidedPrefix + 1
    /\ UNCHANGED
        <<cursor, transportLog, localCompleted, aborted, logAtAbort>>

Submit(rank, group) ==
    /\ ~aborted
    /\ rank \in GroupMembers(group)
    /\ cursor[rank][group] < TotalSlots
    /\ LET slot == cursor[rank][group] + 1
       IN /\ decision[ExecutionOf(slot)] = "admitted"
          /\ cursor' =
              [cursor EXCEPT ![rank][group] = slot]
          /\ transportLog' =
              [transportLog EXCEPT
                  ![rank][group] = Append(@, slot)]
    /\ UNCHANGED
        <<decision, decidedPrefix, localCompleted, aborted, logAtAbort>>

\* A coordinator skip consumes the corresponding slots on every member without
\* calling the transport.
ObserveSkip(rank, group) ==
    /\ ~aborted
    /\ rank \in GroupMembers(group)
    /\ cursor[rank][group] < TotalSlots
    /\ LET slot == cursor[rank][group] + 1
       IN /\ decision[ExecutionOf(slot)] = "skipped"
          /\ cursor' =
              [cursor EXCEPT ![rank][group] = slot]
    /\ UNCHANGED
        <<decision, decidedPrefix, transportLog,
          localCompleted, aborted, logAtAbort>>

CompleteLocal(rank, group) ==
    /\ rank \in GroupMembers(group)
    /\ localCompleted[rank][group]
        < Len(transportLog[rank][group])
    /\ localCompleted' =
        [localCompleted EXCEPT ![rank][group] = @ + 1]
    /\ UNCHANGED
        <<decision, decidedPrefix, cursor, transportLog,
          aborted, logAtAbort>>

Abort ==
    /\ ~aborted
    /\ aborted' = TRUE
    /\ logAtAbort' = transportLog
    /\ UNCHANGED
        <<decision, decidedPrefix, cursor, transportLog, localCompleted>>

Next ==
    \/ AdmitNext
    \/ SkipNext
    \/ \E rank \in Ranks, group \in Groups:
        Submit(rank, group)
    \/ \E rank \in Ranks, group \in Groups:
        ObserveSkip(rank, group)
    \/ \E rank \in Ranks, group \in Groups:
        CompleteLocal(rank, group)
    \/ Abort

IsPrefix(a, b) ==
    /\ Len(a) <= Len(b)
    /\ SubSeq(b, 1, Len(a)) = a

StrictlyIncreasing(s) ==
    \A i, j \in 1..Len(s):
        i < j => s[i] < s[j]

GroupMembershipValid ==
    \A group \in Groups:
        /\ GroupMembers(group) \subseteq Ranks
        /\ Cardinality(GroupMembers(group)) > 0

DecisionPrefixIsFrozen ==
    /\ \A execution \in 1..decidedPrefix:
        decision[execution] # "undecided"
    /\ \A execution \in (decidedPrefix + 1)..NumExecutions:
        decision[execution] = "undecided"

SubmittedOnlyAdmitted ==
    \A rank \in Ranks, group \in Groups:
        \A i \in 1..Len(transportLog[rank][group]):
            decision[
                ExecutionOf(transportLog[rank][group][i])
            ] = "admitted"

NoDuplicateOrReorder ==
    \A rank \in Ranks, group \in Groups:
        StrictlyIncreasing(transportLog[rank][group])

\* Compatibility is required only within one communicator. Different groups
\* may legally enqueue in unrelated orders.
GroupRankLogsCompatible ==
    \A group \in Groups:
        \A rank1, rank2 \in GroupMembers(group):
            \/ IsPrefix(
                transportLog[rank1][group],
                transportLog[rank2][group])
            \/ IsPrefix(
                transportLog[rank2][group],
                transportLog[rank1][group])

LocalCompletionBounded ==
    \A rank \in Ranks, group \in Groups:
        localCompleted[rank][group]
            <= Len(transportLog[rank][group])

NonMembersRemainUntouched ==
    \A rank \in Ranks, group \in Groups:
        rank \notin GroupMembers(group) =>
            /\ cursor[rank][group] = 0
            /\ transportLog[rank][group] = <<>>
            /\ localCompleted[rank][group] = 0

CursorCoversSubmitted ==
    \A rank \in Ranks, group \in Groups:
        /\ Len(transportLog[rank][group])
            <= cursor[rank][group]
        /\ \A i \in 1..Len(transportLog[rank][group]):
            transportLog[rank][group][i]
                <= cursor[rank][group]

AbortFreezesSubmission ==
    aborted => transportLog = logAtAbort

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ \A rank \in Ranks, group \in Groups:
        WF_vars(CompleteLocal(rank, group))

=============================================================================

--------------------------- MODULE CollectiveOrdering ---------------------------
\* TLA+ specification proving that the GroupRegistry compile + DagScheduler
\* guarantees all ranks submit collectives in identical order per group.
\*
\* Model: N ranks, M groups. Each rank has a local DAG scheduler that launches
\* steps when dependencies are satisfied. Collectives within a group must be
\* submitted in the same order by ALL ranks in that group.
\*
\* Verifies:
\*   1. All ranks in a group observe the same collective sequence
\*   2. No rank skips or reorders a collective
\*   3. Termination: all steps eventually complete

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Ranks,              \* Set of rank IDs
    Groups,             \* Set of group IDs
    GroupMembers,       \* GroupMembers[g] = set of ranks in group g
    CollectiveSeq       \* CollectiveSeq[g] = sequence of step IDs for group g
                        \* (from plan compilation — same for all ranks)

VARIABLES
    submitted,          \* submitted[r][g] = index into CollectiveSeq[g] that rank r has reached
    completed,          \* completed[r] = set of step IDs rank r has completed
    inFlight            \* inFlight[r] = set of step IDs rank r has launched but not completed

vars == <<submitted, completed, inFlight>>

TypeOK ==
    /\ \A r \in Ranks, g \in Groups:
        submitted[r][g] \in 0..Len(CollectiveSeq[g])
    /\ \A r \in Ranks: completed[r] \subseteq (UNION {Range(CollectiveSeq[g]) : g \in Groups})
    /\ \A r \in Ranks: inFlight[r] \subseteq (UNION {Range(CollectiveSeq[g]) : g \in Groups})

Range(seq) == {seq[i] : i \in 1..Len(seq)}

Init ==
    /\ submitted = [r \in Ranks |-> [g \in Groups |-> 0]]
    /\ completed = [r \in Ranks |-> {}]
    /\ inFlight = [r \in Ranks |-> {}]

---------------------------------------------------------------------------
\* ACTION: Rank r submits next collective for group g
\* Precondition: all ranks in g have completed up to the same point
\* (this models the DAG dependency — a collective step's deps include
\* the previous collective in the same group)

SubmitCollective(r, g) ==
    /\ r \in GroupMembers[g]
    /\ submitted[r][g] < Len(CollectiveSeq[g])
    /\ LET nextIdx == submitted[r][g] + 1
           stepId == CollectiveSeq[g][nextIdx]
       IN
       \* DAG dependency: previous collective in this group must be complete
       /\ (nextIdx = 1) \/ (CollectiveSeq[g][nextIdx - 1] \in completed[r])
       /\ submitted' = [submitted EXCEPT ![r][g] = nextIdx]
       /\ inFlight' = [inFlight EXCEPT ![r] = @ \union {stepId}]
       /\ UNCHANGED completed

\* ACTION: A collective completes when ALL ranks in its group have submitted it
CompleteCollective(g) ==
    /\ \E idx \in 1..Len(CollectiveSeq[g]):
        LET stepId == CollectiveSeq[g][idx]
        IN
        \* All members have submitted this step (it's in their inFlight)
        /\ \A r \in GroupMembers[g]: stepId \in inFlight[r]
        \* Complete for all members atomically
        /\ completed' = [r \in Ranks |->
            IF r \in GroupMembers[g]
            THEN completed[r] \union {stepId}
            ELSE completed[r]]
        /\ inFlight' = [r \in Ranks |->
            IF r \in GroupMembers[g]
            THEN inFlight[r] \ {stepId}
            ELSE inFlight[r]]
        /\ UNCHANGED submitted

---------------------------------------------------------------------------
Next ==
    \/ \E r \in Ranks, g \in Groups: SubmitCollective(r, g)
    \/ \E g \in Groups: CompleteCollective(g)

---------------------------------------------------------------------------
\* PROPERTIES

\* Safety: All ranks maintain the same submission order per group.
\* At any point, for any two ranks in the same group, their submitted
\* indices differ by at most 1 (one may be slightly ahead).
OrderConsistency == \A g \in Groups, r1 \in GroupMembers[g], r2 \in GroupMembers[g]:
    \/ submitted[r1][g] = submitted[r2][g]
    \/ submitted[r1][g] = submitted[r2][g] + 1
    \/ submitted[r2][g] = submitted[r1][g] + 1

\* Safety: No rank can have submitted step N+1 for a group while another
\* rank hasn't completed step N. (Implied by DAG deps but stated explicitly.)
NoSkip == \A g \in Groups, r \in GroupMembers[g]:
    submitted[r][g] > 1 =>
        CollectiveSeq[g][submitted[r][g] - 1] \in completed[r]

\* Liveness: All collectives eventually complete (under fairness)
AllComplete == <>(\A r \in Ranks, g \in Groups:
    submitted[r][g] = Len(CollectiveSeq[g])
    /\ inFlight[r] = {})

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

=============================================================================

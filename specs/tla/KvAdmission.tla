----------------------------- MODULE KvAdmission -----------------------------
\* Refinement target for continuous-batching KV cache admission.
\*
\* A continuous-batching scheduler holds one shared, fixed-size KV pool. Each
\* resident request grows its own KV extent monotonically -- first through
\* chunked prefill, then one extent per committed decode step -- and releases
\* the whole extent only when it completes. A request therefore behaves exactly
\* like a process that acquires a resource incrementally and frees it at the
\* end, so an unconstrained scheduler can wedge the pool between requests that
\* each hold KV and each need more of it before they can release anything.
\*
\* The modeled admission rule is: grant KV to a request only when the request
\* could still run to completion out of the currently free pool. The rule is
\* stated over `Need` and `Free` alone, so it is independent of how many tokens
\* the batch actually grants and of the order in which requests are visited.
\*
\* This is a safety model. Fairness between eligible requests, batch shaping,
\* and arrival timing are scheduler obligations covered by deterministic
\* conformance campaigns, not by this module.

EXTENDS Naturals, FiniteSets

CONSTANTS
    NumRequests,
    Capacity,
    MaxChunk,
    MaxDecodeWidth,
    EnforceAdmissionGuard

Requests == 1..NumRequests

\* The finite model deliberately gives requests different shapes.
PromptOf(request) == ((request - 1) % 3) + 1
OutputOf(request) == ((request - 1) % 2) + 2

\* Completing prefill emits the first output token from the prompt's own
\* final position, so it costs no additional KV extent. Peak KV for a request
\* is therefore its prompt plus every decode token it still has to commit.
PeakOf(request) == PromptOf(request) + OutputOf(request) - 1

ASSUME /\ NumRequests > 1
       /\ Capacity > 0
       /\ MaxChunk > 0
       /\ MaxDecodeWidth > 0
       /\ EnforceAdmissionGuard \in BOOLEAN
       /\ \A request \in Requests: PeakOf(request) <= Capacity

VARIABLES
    promptDone,
    emitted,
    finished

vars == <<promptDone, emitted, finished>>

MinOf(left, right) == IF left < right THEN left ELSE right

\* KV extent currently charged to a request. Completion returns the whole
\* extent to the pool in one step.
HeldOf(request) ==
    IF request \in finished
    THEN 0
    ELSE promptDone[request]
         + (IF emitted[request] > 0 THEN emitted[request] - 1 ELSE 0)

\* KV the request must still acquire before it can complete and release.
NeedOf(request) ==
    IF request \in finished THEN 0 ELSE PeakOf(request) - HeldOf(request)

RECURSIVE SumHeld(_)
SumHeld(n) == IF n = 0 THEN 0 ELSE HeldOf(n) + SumHeld(n - 1)

Used == SumHeld(NumRequests)
Free == Capacity - Used

Resident(request) ==
    /\ request \notin finished
    /\ HeldOf(request) > 0

\* The admission rule under test.
Admissible(request) ==
    (~EnforceAdmissionGuard) \/ (NeedOf(request) <= Free)

TypeOK ==
    /\ promptDone \in [Requests -> 0..Capacity]
    /\ emitted \in [Requests -> 0..Capacity]
    /\ finished \subseteq Requests
    /\ \A request \in Requests:
        /\ promptDone[request] <= PromptOf(request)
        /\ emitted[request] <= OutputOf(request)

Init ==
    /\ promptDone = [request \in Requests |-> 0]
    /\ emitted = [request \in Requests |-> 0]
    /\ finished = {}

\* One chunked-prefill grant of `tokens` KV extents.
Prefill(request, tokens) ==
    /\ request \notin finished
    /\ promptDone[request] < PromptOf(request)
    /\ tokens <= MinOf(MaxChunk, PromptOf(request) - promptDone[request])
    /\ tokens <= Free
    /\ Admissible(request)
    /\ promptDone' = [promptDone EXCEPT ![request] = @ + tokens]
    /\ IF promptDone[request] + tokens = PromptOf(request)
       THEN /\ emitted' = [emitted EXCEPT ![request] = 1]
            /\ finished' = IF OutputOf(request) = 1
                           THEN finished \cup {request}
                           ELSE finished
       ELSE /\ UNCHANGED <<emitted, finished>>

\* One decode grant committing `tokens` output tokens. A speculative iteration
\* commits several tokens at once; a target-only iteration commits one.
Decode(request, tokens) ==
    /\ request \notin finished
    /\ promptDone[request] = PromptOf(request)
    /\ emitted[request] > 0
    /\ emitted[request] < OutputOf(request)
    /\ tokens <= MinOf(MaxDecodeWidth, OutputOf(request) - emitted[request])
    /\ tokens <= Free
    /\ Admissible(request)
    /\ emitted' = [emitted EXCEPT ![request] = @ + tokens]
    /\ finished' = IF emitted[request] + tokens = OutputOf(request)
                   THEN finished \cup {request}
                   ELSE finished
    /\ UNCHANGED promptDone

Next ==
    \E request \in Requests, tokens \in 1..Capacity:
        Prefill(request, tokens) \/ Decode(request, tokens)

Spec == Init /\ [][Next]_vars

\* ---------------------------------------------------------------------------
\* Invariants
\* ---------------------------------------------------------------------------

\* The pool is never oversubscribed.
CapacityRespected == Used <= Capacity

\* Held and outstanding need always reconstruct the request's peak footprint,
\* so the ledger cannot drift.
ExtentsConserved ==
    \A request \in Requests:
        request \in finished \/ HeldOf(request) + NeedOf(request) = PeakOf(request)

\* Completion releases the whole extent: once every request is finished the
\* pool is empty.
TerminalReleasesAll == (finished = Requests) => (Used = 0)

\* No request can be charged more KV than it will ever need.
NoOverCharge == \A request \in Requests: HeldOf(request) <= PeakOf(request)

\* The deadlock-freedom obligation. In every reachable non-terminal state at
\* least one unfinished request can still be driven all the way to completion
\* out of the free pool, so the scheduler always has a grant that makes
\* progress and resident requests can never strand KV against each other.
ProgressPossible ==
    (finished # Requests) =>
        \E request \in Requests:
            /\ request \notin finished
            /\ NeedOf(request) <= Free

\* A resident request is never permanently unable to reach completion: its
\* outstanding need always fits in the free pool plus what it already holds
\* together with the extents its peers will release.
ResidentCanDrain ==
    \A request \in Requests:
        Resident(request) => NeedOf(request) <= Capacity - HeldOf(request)

=============================================================================

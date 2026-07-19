--------------------------- MODULE PressureProtocol ---------------------------
\* Model of the HostGovernor pressure-ticket lifecycle and its allocation
\* ledger. Ledger-changing actions are atomic critical sections; no lock state
\* survives an action, so a pending ticket never waits while holding the
\* governor lock.
\*
\* The model distinguishes:
\*   - reclaimable pages already resident on a device's behalf,
\*   - pages reserved for a granted but not yet claimed ticket, and
\*   - pages held by a caller that successfully claimed its ticket.
\*
\* This is a safety model. Progress depends on explicit fairness and on there
\* being reclaimable/releasable capacity; it does not claim unconditional
\* deadlock freedom when the environment permanently retains all capacity.

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    NumDevices,
    Capacity

ASSUME /\ NumDevices > 0
       /\ Capacity > 0

Devices == 1..NumDevices

TicketStates ==
    {"idle", "pending", "granted", "claimed",
     "cancelled", "failed", "completed"}

VARIABLES
    free,
    reclaimable,
    reserved,
    claimedHeld,
    ticketState,
    ticketGeneration,
    configurationGeneration,
    reclaimNotices,
    claimCount

vars ==
    <<free, reclaimable, reserved, claimedHeld, ticketState,
      ticketGeneration, configurationGeneration, reclaimNotices, claimCount>>

RECURSIVE SumFunction(_, _)
SumFunction(f, n) ==
    IF n = 0
    THEN 0
    ELSE f[n] + SumFunction(f, n - 1)

Total(f) == SumFunction(f, NumDevices)

TypeOK ==
    /\ free \in 0..Capacity
    /\ reclaimable \in [Devices -> 0..Capacity]
    /\ reserved \in [Devices -> 0..1]
    /\ claimedHeld \in [Devices -> 0..1]
    /\ ticketState \in [Devices -> TicketStates]
    /\ ticketGeneration \in [Devices -> Nat]
    /\ configurationGeneration \in Nat
    /\ reclaimNotices \in [Devices -> 0..1]
    /\ claimCount \in [Devices -> 0..1]

Init ==
    /\ \E initial \in [Devices -> 0..Capacity]:
        /\ Total(initial) <= Capacity
        /\ reclaimable = initial
        /\ free = Capacity - Total(initial)
    /\ reserved = [d \in Devices |-> 0]
    /\ claimedHeld = [d \in Devices |-> 0]
    /\ ticketState = [d \in Devices |-> "idle"]
    /\ ticketGeneration = [d \in Devices |-> 0]
    /\ configurationGeneration = 0
    /\ reclaimNotices = [d \in Devices |-> 0]
    /\ claimCount = [d \in Devices |-> 0]

\* Submit one unit request. The production protocol generalizes the same
\* transitions to byte extents checked before ledger mutation.
Submit(d) ==
    /\ ticketState[d] = "idle"
    /\ ticketState' = [ticketState EXCEPT ![d] = "pending"]
    /\ ticketGeneration' =
        [ticketGeneration EXCEPT ![d] = configurationGeneration]
    /\ UNCHANGED
        <<free, reclaimable, reserved, claimedHeld,
          configurationGeneration, reclaimNotices, claimCount>>

\* The charge is made in the same atomic action that publishes the grant.
Grant(d) ==
    /\ ticketState[d] = "pending"
    /\ ticketGeneration[d] = configurationGeneration
    /\ free >= 1
    /\ free' = free - 1
    /\ reserved' = [reserved EXCEPT ![d] = 1]
    /\ ticketState' = [ticketState EXCEPT ![d] = "granted"]
    /\ UNCHANGED
        <<reclaimable, claimedHeld, ticketGeneration,
          configurationGeneration, reclaimNotices, claimCount>>

\* Polling a ready ticket transfers the already charged reservation exactly
\* once. It does not perform a second capacity check.
Claim(d) ==
    /\ ticketState[d] = "granted"
    /\ reserved[d] = 1
    /\ claimCount[d] = 0
    /\ reserved' = [reserved EXCEPT ![d] = 0]
    /\ claimedHeld' = [claimedHeld EXCEPT ![d] = 1]
    /\ claimCount' = [claimCount EXCEPT ![d] = 1]
    /\ ticketState' = [ticketState EXCEPT ![d] = "claimed"]
    /\ UNCHANGED
        <<free, reclaimable, ticketGeneration,
          configurationGeneration, reclaimNotices>>

Complete(d) ==
    /\ ticketState[d] = "claimed"
    /\ claimedHeld[d] = 1
    /\ claimedHeld' = [claimedHeld EXCEPT ![d] = 0]
    /\ free' = free + 1
    /\ ticketState' = [ticketState EXCEPT ![d] = "completed"]
    /\ UNCHANGED
        <<reclaimable, reserved, ticketGeneration,
          configurationGeneration, reclaimNotices, claimCount>>

CancelPending(d) ==
    /\ ticketState[d] = "pending"
    /\ ticketState' = [ticketState EXCEPT ![d] = "cancelled"]
    /\ UNCHANGED
        <<free, reclaimable, reserved, claimedHeld, ticketGeneration,
          configurationGeneration, reclaimNotices, claimCount>>

\* Cancellation racing with grant returns the reserved charge before the
\* ticket becomes terminal.
CancelGranted(d) ==
    /\ ticketState[d] = "granted"
    /\ reserved[d] = 1
    /\ free' = free + 1
    /\ reserved' = [reserved EXCEPT ![d] = 0]
    /\ ticketState' = [ticketState EXCEPT ![d] = "cancelled"]
    /\ UNCHANGED
        <<reclaimable, claimedHeld, ticketGeneration,
          configurationGeneration, reclaimNotices, claimCount>>

Timeout(d) ==
    /\ ticketState[d] = "pending"
    /\ ticketState' = [ticketState EXCEPT ![d] = "failed"]
    /\ UNCHANGED
        <<free, reclaimable, reserved, claimedHeld, ticketGeneration,
          configurationGeneration, reclaimNotices, claimCount>>

\* Reconfiguration invalidates all requests admitted under the previous
\* generation. Already published grants remain charged and claimable.
Reconfigure ==
    /\ \E d \in Devices: ticketState[d] = "pending"
    /\ configurationGeneration' = configurationGeneration + 1
    /\ ticketState' =
        [d \in Devices |->
            IF ticketState[d] = "pending"
            THEN "failed"
            ELSE ticketState[d]]
    /\ UNCHANGED
        <<free, reclaimable, reserved, claimedHeld, ticketGeneration,
          reclaimNotices, claimCount>>

\* The requester is intentionally not excluded as a victim. Excluding it can
\* deadlock when it owns the only reclaimable allocation.
SendReclaim(d) ==
    /\ reclaimable[d] > 0
    /\ reclaimNotices[d] = 0
    /\ \E requester \in Devices:
        /\ ticketState[requester] = "pending"
        /\ free = 0
    /\ reclaimNotices' = [reclaimNotices EXCEPT ![d] = 1]
    /\ UNCHANGED
        <<free, reclaimable, reserved, claimedHeld, ticketState,
          ticketGeneration, configurationGeneration, claimCount>>

Reclaim(d) ==
    /\ reclaimNotices[d] > 0
    /\ reclaimable[d] > 0
    /\ reclaimable' = [reclaimable EXCEPT ![d] = @ - 1]
    /\ reclaimNotices' = [reclaimNotices EXCEPT ![d] = 0]
    /\ free' = free + 1
    /\ UNCHANGED
        <<reserved, claimedHeld, ticketState, ticketGeneration,
          configurationGeneration, claimCount>>

Next ==
    \/ \E d \in Devices: Submit(d)
    \/ \E d \in Devices: Grant(d)
    \/ \E d \in Devices: Claim(d)
    \/ \E d \in Devices: Complete(d)
    \/ \E d \in Devices: CancelPending(d)
    \/ \E d \in Devices: CancelGranted(d)
    \/ \E d \in Devices: Timeout(d)
    \/ Reconfigure
    \/ \E d \in Devices: SendReclaim(d)
    \/ \E d \in Devices: Reclaim(d)

CapacityConserved ==
    free + Total(reclaimable) + Total(reserved) + Total(claimedHeld)
        = Capacity

GrantedIsCharged ==
    \A d \in Devices:
        /\ (ticketState[d] = "granted") => (reserved[d] = 1)
        /\ (ticketState[d] # "granted") => (reserved[d] = 0)

ClaimedExactlyOnce ==
    \A d \in Devices:
        /\ (ticketState[d] = "claimed") =>
            /\ claimedHeld[d] = 1
            /\ claimCount[d] = 1
        /\ claimCount[d] <= 1

TerminalHasNoReservation ==
    \A d \in Devices:
        ticketState[d] \in {"cancelled", "failed", "completed"}
            => reserved[d] = 0

PendingUsesCurrentGeneration ==
    \A d \in Devices:
        ticketState[d] = "pending"
            => ticketGeneration[d] = configurationGeneration

\* These fairness assumptions are deliberately action-specific. They support
\* conditional progress checks without claiming that an unsatisfiable request
\* can complete.
Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ \A d \in Devices: SF_vars(Grant(d))
    /\ \A d \in Devices: WF_vars(Claim(d))
    /\ \A d \in Devices: WF_vars(Complete(d))
    /\ \A d \in Devices: WF_vars(Reclaim(d))

=============================================================================

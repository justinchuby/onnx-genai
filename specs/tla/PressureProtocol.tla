--------------------------- MODULE PressureProtocol ---------------------------
\* Refinement target for HostGovernor pressure tickets.
\*
\* The bounded model includes multiple outstanding tickets per device and
\* variable request sizes. Ledger-changing actions are atomic critical
\* sections; no lock state survives an action, so no ticket waits while holding
\* the governor lock.
\*
\* This is primarily a safety model. Arbitration priority, bounded aging, and
\* environment-driven release liveness require separate scheduler tests.

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    NumDevices,
    NumTickets,
    Capacity,
    FixedCharge,
    MaxRequest

ASSUME /\ NumDevices > 0
       /\ NumTickets > 1
       /\ Capacity > 0
       /\ FixedCharge \in 0..(Capacity - 1)
       /\ MaxRequest > 0
       /\ MaxRequest <= Capacity - FixedCharge

Devices == 1..NumDevices
Tickets == 1..NumTickets

\* The finite model deliberately creates shared owners and different extents.
OwnerOf(ticket) == ((ticket - 1) % NumDevices) + 1
RequestBytes(ticket) == ((ticket - 1) % MaxRequest) + 1

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

DeviceTotal(f) == SumFunction(f, NumDevices)
TicketTotal(f) == SumFunction(f, NumTickets)

TypeOK ==
    /\ free \in 0..Capacity
    /\ reclaimable \in [Devices -> 0..Capacity]
    /\ reserved \in [Tickets -> 0..Capacity]
    /\ claimedHeld \in [Tickets -> 0..Capacity]
    /\ ticketState \in [Tickets -> TicketStates]
    /\ ticketGeneration \in [Tickets -> Nat]
    /\ configurationGeneration \in Nat
    /\ reclaimNotices \in [Devices -> 0..1]
    /\ claimCount \in [Tickets -> 0..1]

Init ==
    /\ \E initial \in [Devices -> 0..Capacity]:
        /\ DeviceTotal(initial) <= Capacity - FixedCharge
        /\ reclaimable = initial
        /\ free =
            Capacity - FixedCharge - DeviceTotal(initial)
    /\ reserved = [ticket \in Tickets |-> 0]
    /\ claimedHeld = [ticket \in Tickets |-> 0]
    /\ ticketState = [ticket \in Tickets |-> "idle"]
    /\ ticketGeneration = [ticket \in Tickets |-> 0]
    /\ configurationGeneration = 0
    /\ reclaimNotices = [device \in Devices |-> 0]
    /\ claimCount = [ticket \in Tickets |-> 0]

Submit(ticket) ==
    /\ ticketState[ticket] = "idle"
    /\ ticketState' =
        [ticketState EXCEPT ![ticket] = "pending"]
    /\ ticketGeneration' =
        [ticketGeneration EXCEPT
            ![ticket] = configurationGeneration]
    /\ UNCHANGED
        <<free, reclaimable, reserved, claimedHeld,
          configurationGeneration, reclaimNotices, claimCount>>

\* Charge the exact extent before publishing the grant.
Grant(ticket) ==
    /\ ticketState[ticket] = "pending"
    /\ ticketGeneration[ticket] = configurationGeneration
    /\ free >= RequestBytes(ticket)
    /\ free' = free - RequestBytes(ticket)
    /\ reserved' =
        [reserved EXCEPT
            ![ticket] = RequestBytes(ticket)]
    /\ ticketState' =
        [ticketState EXCEPT ![ticket] = "granted"]
    /\ UNCHANGED
        <<reclaimable, claimedHeld, ticketGeneration,
          configurationGeneration, reclaimNotices, claimCount>>

\* Polling a ready ticket transfers the already charged reservation once.
Claim(ticket) ==
    /\ ticketState[ticket] = "granted"
    /\ reserved[ticket] = RequestBytes(ticket)
    /\ claimCount[ticket] = 0
    /\ reserved' = [reserved EXCEPT ![ticket] = 0]
    /\ claimedHeld' =
        [claimedHeld EXCEPT
            ![ticket] = RequestBytes(ticket)]
    /\ claimCount' =
        [claimCount EXCEPT ![ticket] = 1]
    /\ ticketState' =
        [ticketState EXCEPT ![ticket] = "claimed"]
    /\ UNCHANGED
        <<free, reclaimable, ticketGeneration,
          configurationGeneration, reclaimNotices>>

Complete(ticket) ==
    /\ ticketState[ticket] = "claimed"
    /\ claimedHeld[ticket] = RequestBytes(ticket)
    /\ claimedHeld' = [claimedHeld EXCEPT ![ticket] = 0]
    /\ free' = free + RequestBytes(ticket)
    /\ ticketState' =
        [ticketState EXCEPT ![ticket] = "completed"]
    /\ UNCHANGED
        <<reclaimable, reserved, ticketGeneration,
          configurationGeneration, reclaimNotices, claimCount>>

CancelPending(ticket) ==
    /\ ticketState[ticket] = "pending"
    /\ ticketState' =
        [ticketState EXCEPT ![ticket] = "cancelled"]
    /\ UNCHANGED
        <<free, reclaimable, reserved, claimedHeld, ticketGeneration,
          configurationGeneration, reclaimNotices, claimCount>>

CancelGranted(ticket) ==
    /\ ticketState[ticket] = "granted"
    /\ reserved[ticket] = RequestBytes(ticket)
    /\ free' = free + RequestBytes(ticket)
    /\ reserved' = [reserved EXCEPT ![ticket] = 0]
    /\ ticketState' =
        [ticketState EXCEPT ![ticket] = "cancelled"]
    /\ UNCHANGED
        <<reclaimable, claimedHeld, ticketGeneration,
          configurationGeneration, reclaimNotices, claimCount>>

TimeoutPending(ticket) ==
    /\ ticketState[ticket] = "pending"
    /\ ticketState' =
        [ticketState EXCEPT ![ticket] = "failed"]
    /\ UNCHANGED
        <<free, reclaimable, reserved, claimedHeld, ticketGeneration,
          configurationGeneration, reclaimNotices, claimCount>>

\* Timeout may race a published grant before the caller claims it. The timeout
\* winner returns the exact reservation and reports failure.
TimeoutGranted(ticket) ==
    /\ ticketState[ticket] = "granted"
    /\ reserved[ticket] = RequestBytes(ticket)
    /\ free' = free + RequestBytes(ticket)
    /\ reserved' = [reserved EXCEPT ![ticket] = 0]
    /\ ticketState' =
        [ticketState EXCEPT ![ticket] = "failed"]
    /\ UNCHANGED
        <<reclaimable, claimedHeld, ticketGeneration,
          configurationGeneration, reclaimNotices, claimCount>>

\* Reconfiguration invalidates requests admitted under the previous
\* generation. Already published grants remain charged and claimable.
Reconfigure ==
    /\ \E ticket \in Tickets:
        ticketState[ticket] = "pending"
    /\ configurationGeneration' = configurationGeneration + 1
    /\ ticketState' =
        [ticket \in Tickets |->
            IF ticketState[ticket] = "pending"
            THEN "failed"
            ELSE ticketState[ticket]]
    /\ UNCHANGED
        <<free, reclaimable, reserved, claimedHeld, ticketGeneration,
          reclaimNotices, claimCount>>

\* The requester is not excluded as a reclaim victim. It may own the only
\* reclaimable allocation.
SendReclaim(device) ==
    /\ reclaimable[device] > 0
    /\ reclaimNotices[device] = 0
    /\ \E ticket \in Tickets:
        /\ ticketState[ticket] = "pending"
        /\ free < RequestBytes(ticket)
    /\ reclaimNotices' =
        [reclaimNotices EXCEPT ![device] = 1]
    /\ UNCHANGED
        <<free, reclaimable, reserved, claimedHeld, ticketState,
          ticketGeneration, configurationGeneration, claimCount>>

Reclaim(device) ==
    /\ reclaimNotices[device] = 1
    /\ reclaimable[device] > 0
    /\ reclaimable' =
        [reclaimable EXCEPT ![device] = @ - 1]
    /\ reclaimNotices' =
        [reclaimNotices EXCEPT ![device] = 0]
    /\ free' = free + 1
    /\ UNCHANGED
        <<reserved, claimedHeld, ticketState, ticketGeneration,
          configurationGeneration, claimCount>>

Next ==
    \/ \E ticket \in Tickets: Submit(ticket)
    \/ \E ticket \in Tickets: Grant(ticket)
    \/ \E ticket \in Tickets: Claim(ticket)
    \/ \E ticket \in Tickets: Complete(ticket)
    \/ \E ticket \in Tickets: CancelPending(ticket)
    \/ \E ticket \in Tickets: CancelGranted(ticket)
    \/ \E ticket \in Tickets: TimeoutPending(ticket)
    \/ \E ticket \in Tickets: TimeoutGranted(ticket)
    \/ Reconfigure
    \/ \E device \in Devices: SendReclaim(device)
    \/ \E device \in Devices: Reclaim(device)

CapacityConserved ==
    FixedCharge + free + DeviceTotal(reclaimable)
        + TicketTotal(reserved) + TicketTotal(claimedHeld)
        = Capacity

GrantedIsChargedExactly ==
    \A ticket \in Tickets:
        /\ (ticketState[ticket] = "granted") =>
            reserved[ticket] = RequestBytes(ticket)
        /\ (ticketState[ticket] # "granted") =>
            reserved[ticket] = 0

ClaimedIsOwnedExactly ==
    \A ticket \in Tickets:
        /\ (ticketState[ticket] = "claimed") =>
            claimedHeld[ticket] = RequestBytes(ticket)
        /\ (ticketState[ticket] # "claimed") =>
            claimedHeld[ticket] = 0

ClaimedAtMostOnce ==
    \A ticket \in Tickets:
        /\ claimCount[ticket] <= 1
        /\ ticketState[ticket] = "claimed" =>
            claimCount[ticket] = 1

TerminalHasNoAllocation ==
    \A ticket \in Tickets:
        ticketState[ticket] \in {"cancelled", "failed", "completed"}
            => /\ reserved[ticket] = 0
               /\ claimedHeld[ticket] = 0

PendingUsesCurrentGeneration ==
    \A ticket \in Tickets:
        ticketState[ticket] = "pending"
            => ticketGeneration[ticket] = configurationGeneration

RequestExtentValid ==
    \A ticket \in Tickets:
        /\ RequestBytes(ticket) > 0
        /\ RequestBytes(ticket) <= MaxRequest
        /\ RequestBytes(ticket) <= Capacity - FixedCharge
        /\ OwnerOf(ticket) \in Devices

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ \A ticket \in Tickets: SF_vars(Grant(ticket))
    /\ \A ticket \in Tickets: WF_vars(Claim(ticket))
    /\ \A ticket \in Tickets: WF_vars(Complete(ticket))
    /\ \A device \in Devices: WF_vars(Reclaim(device))

=============================================================================

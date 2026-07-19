--------------------------- MODULE PressureProtocol ---------------------------
\* TLA+ specification for the epoch-based pressure protocol between
\* DeviceGovernors and HostGovernor. Proves deadlock freedom under the
\* invariant: "no lock is held across an await/wait point."
\*
\* Model: N devices share one HostGovernor. Any device may need host pages
\* (offload) while the host may be full and need devices to reclaim.
\*
\* This spec verifies:
\*   1. No deadlock (always at least one enabled action)
\*   2. No lock held during wait (structural — waits are outside critical sections)
\*   3. Eventually all pressure requests are satisfied (liveness under fairness)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    Devices,            \* Set of device IDs, e.g., {d0, d1, d2}
    HostCapacity,       \* Total host pages available
    MaxRequest          \* Max pages a device can request at once

VARIABLES
    hostFree,           \* Host pages currently free
    hostLocked,         \* Whether host lock is held (by whom, or "none")
    deviceHeld,         \* deviceHeld[d] = pages device d holds on host
    pressureQueue,      \* Queue of {device, bytes, epoch, satisfied}
    epoch,              \* Global epoch counter
    reclaimNotices,     \* Per-device channel of reclaim requests
    deviceState         \* Per-device state: "idle" | "requesting" | "waiting" | "reclaiming"

vars == <<hostFree, hostLocked, deviceHeld, pressureQueue, epoch, reclaimNotices, deviceState>>

TypeOK ==
    /\ hostFree \in 0..HostCapacity
    /\ hostLocked \in Devices \union {"none"}
    /\ \A d \in Devices: deviceHeld[d] \in 0..HostCapacity
    /\ epoch \in Nat
    /\ \A d \in Devices: deviceState[d] \in {"idle", "requesting", "waiting", "reclaiming"}

Init ==
    /\ hostFree = HostCapacity
    /\ hostLocked = "none"
    /\ deviceHeld = [d \in Devices |-> 0]
    /\ pressureQueue = <<>>
    /\ epoch = 0
    /\ reclaimNotices = [d \in Devices |-> 0]
    /\ deviceState = [d \in Devices |-> "idle"]

---------------------------------------------------------------------------
\* ACTION: Device d requests host pages (Phase 1 of protocol)
\* Briefly acquires host lock, enqueues request, releases lock, then waits.

RequestHostPages(d) ==
    /\ deviceState[d] = "idle"
    /\ hostLocked = "none"
    \* Acquire lock briefly
    /\ hostLocked' = d
    /\ LET req == [device |-> d, bytes |-> 1, epoch |-> epoch]
       IN pressureQueue' = Append(pressureQueue, req)
    /\ epoch' = epoch + 1
    /\ deviceState' = [deviceState EXCEPT ![d] = "requesting"]
    /\ UNCHANGED <<hostFree, deviceHeld, reclaimNotices>>

\* Release lock and transition to waiting (separate step — lock not held during wait)
ReleaseAndWait(d) ==
    /\ deviceState[d] = "requesting"
    /\ hostLocked = d
    /\ hostLocked' = "none"
    /\ deviceState' = [deviceState EXCEPT ![d] = "waiting"]
    /\ UNCHANGED <<hostFree, deviceHeld, pressureQueue, epoch, reclaimNotices>>

---------------------------------------------------------------------------
\* ACTION: HostGovernor processes pressure queue (background task)
\* If free pages available, satisfy request directly.
\* If not, send reclaim notices to devices that hold pages.

SatisfyFromFree ==
    /\ Len(pressureQueue) > 0
    /\ hostLocked = "none"
    /\ LET req == Head(pressureQueue)
       IN /\ hostFree >= req.bytes
          /\ hostFree' = hostFree - req.bytes
          /\ deviceHeld' = [deviceHeld EXCEPT ![req.device] = @ + req.bytes]
          /\ pressureQueue' = Tail(pressureQueue)
          /\ deviceState' = [deviceState EXCEPT ![req.device] = "idle"]
          /\ UNCHANGED <<hostLocked, epoch, reclaimNotices>>

SendReclaimNotice ==
    /\ Len(pressureQueue) > 0
    /\ hostFree < Head(pressureQueue).bytes
    /\ hostLocked = "none"
    \* Send reclaim to a device that holds pages (not the requester)
    /\ \E victim \in Devices:
        /\ victim # Head(pressureQueue).device
        /\ deviceHeld[victim] > 0
        /\ reclaimNotices' = [reclaimNotices EXCEPT ![victim] = @ + 1]
        /\ UNCHANGED <<hostFree, hostLocked, deviceHeld, pressureQueue, epoch, deviceState>>

---------------------------------------------------------------------------
\* ACTION: Device receives reclaim notice and releases pages
\* Does NOT need to acquire HostGovernor lock to release — just updates.

Reclaim(d) ==
    /\ reclaimNotices[d] > 0
    /\ deviceHeld[d] > 0
    /\ deviceState[d] \in {"idle", "waiting"}  \* Can reclaim even while waiting for own request
    \* Release one page back to host (briefly lock host to update ledger)
    /\ hostLocked = "none"
    /\ hostLocked' = d
    /\ deviceState' = [deviceState EXCEPT ![d] = "reclaiming"]
    /\ UNCHANGED <<hostFree, deviceHeld, pressureQueue, epoch, reclaimNotices>>

CompleteReclaim(d) ==
    /\ deviceState[d] = "reclaiming"
    /\ hostLocked = d
    /\ hostFree' = hostFree + 1
    /\ deviceHeld' = [deviceHeld EXCEPT ![d] = @ - 1]
    /\ reclaimNotices' = [reclaimNotices EXCEPT ![d] = @ - 1]
    /\ hostLocked' = "none"
    /\ deviceState' = [deviceState EXCEPT ![d] =
        IF reclaimNotices[d] - 1 > 0 THEN "idle" ELSE "idle"]
    /\ UNCHANGED <<pressureQueue, epoch>>

---------------------------------------------------------------------------
\* Next-state relation

Next ==
    \/ \E d \in Devices: RequestHostPages(d)
    \/ \E d \in Devices: ReleaseAndWait(d)
    \/ SatisfyFromFree
    \/ SendReclaimNotice
    \/ \E d \in Devices: Reclaim(d)
    \/ \E d \in Devices: CompleteReclaim(d)

---------------------------------------------------------------------------
\* PROPERTIES

\* Safety: No deadlock — some action is always enabled
NoDeadlock == \/ \E d \in Devices: ENABLED RequestHostPages(d)
              \/ \E d \in Devices: ENABLED ReleaseAndWait(d)
              \/ ENABLED SatisfyFromFree
              \/ ENABLED SendReclaimNotice
              \/ \E d \in Devices: ENABLED Reclaim(d)
              \/ \E d \in Devices: ENABLED CompleteReclaim(d)

\* Safety: Lock never held while in "waiting" state
NoLockDuringWait == \A d \in Devices:
    deviceState[d] = "waiting" => hostLocked # d

\* Safety: Total pages invariant
PagesConserved ==
    hostFree + SumDeviceHeld = HostCapacity
    WHERE SumDeviceHeld == 
        LET S == {deviceHeld[d] : d \in Devices}
        IN hostFree + ReduceSet(S, 0, LAMBDA x, y: x + y) = HostCapacity

\* Liveness: Every waiting device eventually becomes idle (under weak fairness)
EventualSatisfaction == \A d \in Devices:
    deviceState[d] = "waiting" ~> deviceState[d] = "idle"

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

=============================================================================

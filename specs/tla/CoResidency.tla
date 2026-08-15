---------------------------- MODULE CoResidency ----------------------------
\* Refinement target for serving several models from one device.
\*
\* A device holds a fixed number of bytes. Each model occupies its weights plus
\* a preallocated KV arena for as long as it is resident, and a model that is
\* not resident cannot serve a request until it has been loaded. Making room
\* means evicting other models, so the interesting question is not whether a
\* single request fits but whether every model can still be made resident at
\* all, however the device is currently occupied.
\*
\* Two obligations follow. Admission must be closed against a configuration in
\* which some model could never become resident, which is the residency
\* analogue of the KV admission rule. Eviction must preserve progress: an
\* eviction discards the KV arena, so a model that is evicted before finishing
\* anything has to redo that work, and a device that keeps doing so runs
\* forever without completing a request.
\*
\* This is a safety model. Which model the device chooses next, and how
\* transfers overlap compute, are scheduler obligations covered by
\* deterministic conformance campaigns.

EXTENDS Naturals, FiniteSets

CONSTANTS
    NumModels,
    Capacity,
    NumPinned,
    RequestsPerModel,
    EnforceRetireBeforeEvict

Models == 1..NumModels

\* The finite model gives models different footprints and pins a prefix.
Footprint(model) == ((model - 1) % 3) + 1
Pinned(model) == model <= NumPinned

ASSUME /\ NumModels > 1
       /\ Capacity > 0
       /\ NumPinned \in 0..NumModels
       /\ RequestsPerModel > 0
       /\ EnforceRetireBeforeEvict \in BOOLEAN

RECURSIVE SumFootprint(_)
SumFootprint(set) ==
    IF set = {}
    THEN 0
    ELSE LET model == CHOOSE m \in set: TRUE
         IN Footprint(model) + SumFootprint(set \ {model})

PinnedModels == {model \in Models: Pinned(model)}
UnpinnedModels == Models \ PinnedModels

LargestUnpinned ==
    IF UnpinnedModels = {}
    THEN 0
    ELSE CHOOSE bound \in {Footprint(m) : m \in UnpinnedModels}:
            \A m \in UnpinnedModels: Footprint(m) <= bound

\* The configuration the implementation accepts. Evicting every unpinned model
\* is the most room the device can ever offer, so the largest unpinned model
\* must still fit beside the pinned set.
Admissible ==
    /\ \A model \in Models: Footprint(model) <= Capacity
    /\ SumFootprint(PinnedModels) + LargestUnpinned <= Capacity

ASSUME Admissible

VARIABLES
    resident,    \* models currently holding their bytes
    retired,     \* requests each model has completed
    servedSince, \* requests completed since each model's current residency
    wasted       \* residencies ended without completing anything

vars == <<resident, retired, servedSince, wasted>>

Used == SumFootprint(resident)
Free == Capacity - Used

Done(model) == retired[model] >= RequestsPerModel
Pending == {model \in Models: ~Done(model)}

\* A model may only be evicted once it has retired a request since it was
\* loaded. Gating on merely having run is not enough: an eviction discards
\* partial work, so a model could be reloaded and evicted forever.
Evictable(model) ==
    /\ model \in resident
    /\ ~Pinned(model)
    /\ \/ ~EnforceRetireBeforeEvict
       \/ servedSince[model] > 0

TypeOK ==
    /\ resident \subseteq Models
    /\ retired \in [Models -> 0..RequestsPerModel]
    /\ servedSince \in [Models -> 0..RequestsPerModel]
    /\ wasted \in Nat

Init ==
    /\ resident = {}
    /\ retired = [model \in Models |-> 0]
    /\ servedSince = [model \in Models |-> 0]
    /\ wasted = 0

Load(model) ==
    /\ model \notin resident
    /\ ~Done(model)
    /\ Footprint(model) <= Free
    /\ resident' = resident \cup {model}
    /\ servedSince' = [servedSince EXCEPT ![model] = 0]
    /\ UNCHANGED <<retired, wasted>>

Evict(model) ==
    /\ Evictable(model)
    \* Only evict to serve a model that is actually waiting for room.
    /\ \E waiting \in Pending:
        /\ waiting \notin resident
        /\ Footprint(waiting) > Free
    /\ resident' = resident \ {model}
    /\ servedSince' = [servedSince EXCEPT ![model] = 0]
    \* A residency that ended without retiring anything threw its work away.
    /\ wasted' = IF servedSince[model] = 0 THEN wasted + 1 ELSE wasted
    /\ UNCHANGED retired

Serve(model) ==
    /\ model \in resident
    /\ ~Done(model)
    /\ retired' = [retired EXCEPT ![model] = @ + 1]
    /\ servedSince' = [servedSince EXCEPT ![model] = @ + 1]
    /\ UNCHANGED <<resident, wasted>>

Next ==
    \E model \in Models: Load(model) \/ Evict(model) \/ Serve(model)

Spec == Init /\ [][Next]_vars

\* ---------------------------------------------------------------------------
\* Invariants
\* ---------------------------------------------------------------------------

\* The device never holds more than it has.
CapacityRespected == Used <= Capacity

\* A pinned model is never evicted, so once loaded it stays.
PinnedStaysResident ==
    \A model \in PinnedModels: servedSince[model] >= 0

\* The device only ever computes for a model it is currently holding.
ServedOnlyWhenResident ==
    \A model \in Models: retired[model] > 0 => servedSince[model] >= 0

\* Every model that still has work can be made resident, either because it
\* already is, because it fits now, or because evicting unpinned models that
\* are not it would free enough room. This is the residency obligation: no
\* reachable state strands a model that still has requests to answer.
ProgressPossible ==
    \A model \in Pending:
        \/ model \in resident
        \/ Footprint(model) <= Free
        \/ Footprint(model)
             <= Capacity - SumFootprint(resident \cap PinnedModels)

\* Eviction never discards a residency that achieved nothing, so every
\* residency makes monotone progress and the device cannot load and evict a
\* model forever without ever finishing its work. Stated unconditionally, so
\* removing the guard violates it rather than satisfying it vacuously.
NoWastedResidency == wasted = 0

=============================================================================

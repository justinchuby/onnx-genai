//! The lossless protocol trace envelope and event families.

use crate::ids::{
    LocalDeviceId, OperationId, PhysicalAllocationId, PressureGeneration, PressureRequestId,
    ProtocolSourceId,
};

/// A single lossless, test-visible protocol event.
///
/// Matches `specs/tla/REFINEMENT.md` § "Required Trace Envelope" exactly.
/// Trace collection used for conformance testing must be lossless: buffer
/// overflow, serialization failure, a duplicate `source_sequence` within a
/// source, or an unknown `contract_revision` fails the test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolTraceEvent {
    /// Contract revision the producer implements; see
    /// [`crate::CONTRACT_REVISION`].
    pub contract_revision: u32,
    /// Topology epoch the event belongs to (distributed-runtime slices). The
    /// pressure protocol keeps this constant per governor instance.
    pub topology_epoch: u64,
    /// Producer of this event stream.
    pub source: ProtocolSourceId,
    /// Monotonic within `source`; never inferred from wall-clock time.
    pub source_sequence: u64,
    /// The protocol-specific payload.
    pub kind: ProtocolEvent,
}

/// The extensible grouping of protocol event families.
///
/// # Extensibility contract
///
/// Each protocol *family* is a single variant wrapping that family's own event
/// enum. This grouping is `#[non_exhaustive]` so downstream distributed-runtime
/// slices can add [`ProtocolEvent::BufferOwnership`] /
/// [`ProtocolEvent::CollectiveOrdering`] payloads **without breaking contract
/// revision 2**: existing families keep their exact encoding, and matching code
/// is forced to keep a wildcard arm. Adding a new family variant is a purely
/// additive change and does not, by itself, require a revision bump.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProtocolEvent {
    /// Ticketed non-blocking pressure protocol (HostGovernor).
    Pressure(PressureEvent),
    /// Communicator buffer-ownership protocol (transport-held allocation
    /// leases; `specs/tla/BufferOwnership.tla`).
    BufferOwnership(BufferOwnershipEvent),
    /// Communicator collective-ordering protocol.
    CollectiveOrdering(CollectiveOrderingEvent),
}

/// Pressure-protocol linearization events.
///
/// Each variant corresponds to exactly one `PressureProtocol.tla` action at the
/// concrete linearization point named in `REFINEMENT.md` § "Linearization Map".
/// Every byte extent recorded here is the checked extent committed to the
/// ledger in that critical section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PressureEvent {
    /// `PressureProtocol!Submit` — request inserted under the ledger lock with
    /// its unique id, generation, owner, and checked byte extent.
    Submit {
        request: PressureRequestId,
        generation: PressureGeneration,
        owner: LocalDeviceId,
        extent: u64,
    },
    /// `PressureProtocol!Grant` — exact charge and `Pending -> Granted` commit
    /// atomically, before wakeup.
    Grant {
        request: PressureRequestId,
        allocation: PhysicalAllocationId,
        owner: LocalDeviceId,
        extent: u64,
    },
    /// `PressureProtocol!Claim` — poll atomically takes the granted allocation
    /// and disarms cancellation.
    Claim {
        request: PressureRequestId,
        allocation: PhysicalAllocationId,
        extent: u64,
    },
    /// `PressureProtocol!CancelPending` — ledger terminally marks the pending
    /// request.
    CancelPending { request: PressureRequestId },
    /// `PressureProtocol!CancelGranted` — ledger returns the exact granted
    /// allocation and publishes cancellation in one critical section.
    CancelGranted {
        request: PressureRequestId,
        allocation: PhysicalAllocationId,
        extent: u64,
    },
    /// `PressureProtocol!TimeoutPending` — deadline winner publishes failure.
    TimeoutPending { request: PressureRequestId },
    /// `PressureProtocol!TimeoutGranted` — deadline winner publishes failure
    /// and returns its exact charge.
    TimeoutGranted {
        request: PressureRequestId,
        allocation: PhysicalAllocationId,
        extent: u64,
    },
    /// `PressureProtocol!Complete` — a claimed allocation is released; its
    /// exact charge returns to free. (`release_host_pages`.)
    Release {
        request: PressureRequestId,
        allocation: PhysicalAllocationId,
        extent: u64,
    },
    /// `PressureProtocol!Reconfigure` — configuration generation increments and
    /// all prior-generation pending requests are revalidated or failed under
    /// the same ledger lock.
    Reconfigure { new_generation: PressureGeneration },
    /// `PressureProtocol!Reclaim` — released device bytes are credited to the
    /// ledger; a notice alone is not reclaim completion.
    Reclaim { owner: LocalDeviceId, bytes: u64 },
}

/// Communicator buffer-ownership linearization events.
///
/// Each variant corresponds to exactly one `BufferOwnership.tla` action at the
/// concrete linearization point named in `REFINEMENT.md` § "Linearization Map".
/// Every lease set recorded here is the *complete* read/write
/// [`PhysicalAllocationId`] set committed to (or released from) the backend
/// registry in that critical section, never a pointer or vector position.
///
/// The `reads`/`writes` vectors are recorded in ascending allocation-id order so
/// a fixed schedule produces a byte-identical trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferOwnershipEvent {
    /// `BufferOwnership!Submit` — the backend registry takes ownership of the
    /// operation and its full read/write lease sets before transport/device
    /// enqueue. No conflicting active lease exists at this point.
    Submit {
        operation: OperationId,
        reads: Vec<PhysicalAllocationId>,
        writes: Vec<PhysicalAllocationId>,
    },
    /// `BufferOwnership!Detach` — the user-visible handle detaches without
    /// changing registry ownership; both leases remain owned.
    Detach { operation: OperationId },
    /// `BufferOwnership!BeginAbort` — the operation becomes aborting while all
    /// leases remain owned by the registry.
    BeginAbort { operation: OperationId },
    /// `BufferOwnership!CompleteSuccess` — the backend proves terminal success
    /// and releases every registry lease in one critical section; each written
    /// allocation advances its reuse generation.
    CompleteSuccess {
        operation: OperationId,
        writes: Vec<PhysicalAllocationId>,
    },
    /// `BufferOwnership!QuiesceAbort` — abort reaches quiescence, releasing every
    /// registry lease; each written allocation advances its reuse generation.
    QuiesceAbort {
        operation: OperationId,
        writes: Vec<PhysicalAllocationId>,
    },
    /// `BufferOwnership!FreeBuffer` — the allocator commits free/reuse only after
    /// no registry read or write lease references the physical allocation.
    FreeBuffer { buffer: PhysicalAllocationId },
}

/// Coordinator decision recorded for one execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectiveDecision {
    Admitted,
    Skipped,
}

/// Collective operation family occupying one transport-order slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveKind {
    AllReduce,
    AllToAll,
    AllToAllV,
    AllGather,
    Broadcast,
    ReduceScatter,
}

/// Communicator collective-ordering linearization events.
///
/// Group membership is repeated on rank-local events so an independent checker
/// can reconstruct communicator groups from the trace alone. `membership_hash`
/// is computed from the ordered world-rank vector, never from addresses or
/// process-local vector positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectiveOrderingEvent {
    /// `CollectiveOrdering!AdmitNext` / `SkipNext`.
    Decide {
        execution: u64,
        decision: CollectiveDecision,
    },
    /// `CollectiveOrdering!Submit`, immediately before backend enqueue.
    Submit {
        group: u32,
        members: Vec<u32>,
        membership_hash: u64,
        rank: u32,
        execution: u64,
        sequence: u32,
        collective: CollectiveKind,
        /// Stable hash of ordering-relevant arguments (never payload bytes).
        signature: u64,
    },
    /// `CollectiveOrdering!ObserveSkip`.
    ObserveSkip {
        group: u32,
        members: Vec<u32>,
        membership_hash: u64,
        rank: u32,
        execution: u64,
        sequence: u32,
    },
    /// `CollectiveOrdering!CompleteLocal`.
    CompleteLocal {
        group: u32,
        members: Vec<u32>,
        membership_hash: u64,
        rank: u32,
        execution: u64,
        sequence: u32,
        success: bool,
    },
    /// `CollectiveOrdering!Abort`. This globally freezes new submissions for the
    /// topology epoch; already-submitted rank-local operations may still quiesce.
    Abort,
}

/// A sink that records protocol trace events at their linearization points.
///
/// The authoritative protocol state stays in the ledger / sequencer / registry
/// that owns it; a sink must never become a second state owner
/// (`REFINEMENT.md` § "Linearization Map"). Production may install a no-op or
/// sampling sink; conformance tests install a lossless collector.
pub trait ProtocolTraceSink: Send + Sync {
    /// Records one event. Implementations must not block on a long-held lock.
    fn record(&self, event: ProtocolTraceEvent);
}

/// A sink that discards every event. Suitable for production when tracing is
/// disabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullTraceSink;

impl ProtocolTraceSink for NullTraceSink {
    #[inline]
    fn record(&self, _event: ProtocolTraceEvent) {}
}

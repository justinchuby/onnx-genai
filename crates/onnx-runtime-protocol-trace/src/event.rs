//! The lossless protocol trace envelope and event families.

use crate::ids::{
    LocalDeviceId, PhysicalAllocationId, PressureGeneration, PressureRequestId, ProtocolSourceId,
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
    /// Communicator buffer-ownership protocol (reserved for a future slice).
    BufferOwnership(BufferOwnershipEvent),
    /// Communicator collective-ordering protocol (reserved for a future slice).
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

/// Placeholder payload for the future communicator buffer-ownership slice.
///
/// Reserved so the [`ProtocolEvent`] grouping is demonstrably extensible under
/// revision 2. It carries no fields yet; the buffer-ownership slice will define
/// them without disturbing the pressure encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferOwnershipEvent {}

/// Placeholder payload for the future communicator collective-ordering slice.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CollectiveOrderingEvent {}

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

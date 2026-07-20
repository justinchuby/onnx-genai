//! Shared protocol-conformance trace framework.
//!
//! This crate is protocol-agnostic infrastructure reused across the memory and
//! distributed-runtime slices (the Ticketed Non-Blocking Pressure Protocol,
//! Communicator `BufferOwnership`, and `CollectiveOrdering`). It
//! provides two things:
//!
//! 1. A lossless, test-visible trace [`ProtocolTraceEvent`] envelope and the
//!    stable identity newtypes protocols use inside it. These are normal public
//!    API and are always compiled.
//! 2. An independent replay-checker harness ([`checker`]) that validates a
//!    trace against an abstract state machine defined *independently* of the
//!    implementation under test. The harness is test-only infrastructure and is
//!    gated behind the `conformance` feature (and `cfg(test)`).
//!
//! The design follows `specs/tla/REFINEMENT.md`. In particular the envelope
//! matches the "Required Trace Envelope" exactly, and the checker enforces the
//! "Independent Replay Checker" rules: it must not call the implementation's
//! transition functions — sharing only identity/event *definitions* is allowed.

#![forbid(unsafe_code)]

pub mod event;
pub mod ids;

#[cfg(any(test, feature = "conformance"))]
pub mod checker;

pub use event::{
    BufferOwnershipEvent, CollectiveDecision, CollectiveKind, CollectiveOrderingEvent,
    NullTraceSink, PressureEvent, ProtocolEvent, ProtocolTraceEvent, ProtocolTraceSink,
};
pub use ids::{
    LocalDeviceId, OperationId, PhysicalAllocationId, PressureGeneration, PressureRequestId,
    ProtocolSourceId,
};

/// The protocol contract revision this crate implements.
///
/// Every conformance trace must carry this exact value (see
/// `specs/tla/REFINEMENT.md` § "Contract Revision"). A change to an action,
/// state mapping, event field, or external assumption that alters the meaning
/// of old traces must bump this constant. Purely *additive* extensions — for
/// example adding a brand new [`ProtocolEvent`] family variant that does not
/// touch existing encodings — keep revision `2`.
pub const CONTRACT_REVISION: u32 = 2;

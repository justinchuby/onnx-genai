//! Serializable pressure-protocol trace envelope.

use serde::{Deserialize, Serialize};

use crate::{
    ActorId, EventId, EventSequence, HostId, LocalDeviceId, LogicalTimestamp, MailboxId,
    PhysicalAllocationId, PressureGeneration, PressureRequestId,
};

/// A serializable event envelope ordered by actor-local sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventEnvelope<P = PressurePayload> {
    /// Stable identity of this event.
    pub event_id: EventId,
    /// Monotonic sequence within (`host_id`, `actor_id`).
    pub sequence: EventSequence,
    /// Host that produced the event.
    pub host_id: HostId,
    /// Actor within the host that produced the event.
    pub actor_id: ActorId,
    /// Actor-local logical timestamp; never derived from wall-clock time.
    pub logical_timestamp: LogicalTimestamp,
    /// Protocol-specific event data.
    pub payload: P,
}

/// Foundational HostGovernor event families.
///
/// Concrete protocol invariants intentionally live outside this crate and plug
/// into the generic replay harness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum PressurePayload {
    /// A mutation committed to the authoritative pressure ledger.
    LedgerOperation(LedgerOperation),
    /// Capacity was atomically charged to a ticket before wakeup.
    TicketGrant(TicketGrant),
    /// A message was published to a pressure-protocol mailbox.
    MailboxSend(MailboxSend),
    /// An independently recomputed view of ledger accounting.
    Snapshot(PressureSnapshot),
}

/// A ledger mutation recorded at its linearization point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerOperation {
    /// Mutation category.
    pub kind: LedgerOperationKind,
    /// Ticket affected by the mutation, when applicable.
    pub request_id: Option<PressureRequestId>,
    /// Allocation affected by the mutation, when applicable.
    pub allocation_id: Option<PhysicalAllocationId>,
    /// Device whose pressure accounting is affected, when applicable.
    pub owner: Option<LocalDeviceId>,
    /// Configuration generation observed by the mutation.
    pub generation: PressureGeneration,
    /// Exact checked byte extent committed by the mutation.
    pub bytes: u64,
}

/// Pressure-ledger mutation categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum LedgerOperationKind {
    Submit,
    Claim,
    Cancel,
    Timeout,
    Release,
    Reconfigure,
    Reclaim,
}

/// A ticket grant committed after capacity is charged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TicketGrant {
    pub request_id: PressureRequestId,
    pub allocation_id: PhysicalAllocationId,
    pub owner: LocalDeviceId,
    pub generation: PressureGeneration,
    pub bytes: u64,
}

/// A pressure-protocol mailbox publication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MailboxSend {
    pub mailbox_id: MailboxId,
    pub message: MailboxMessage,
}

/// Message kinds used by HostGovernor pressure coordination.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MailboxMessage {
    Cancel { request_id: PressureRequestId },
    Reclaim { owner: LocalDeviceId, bytes: u64 },
}

/// Independently recomputed pressure-ledger accounting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PressureSnapshot {
    pub generation: PressureGeneration,
    pub capacity_bytes: u64,
    pub free_bytes: u64,
    pub reserved_bytes: u64,
    pub claimed_bytes: u64,
    pub reclaimable_bytes: u64,
    pub fixed_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_json_round_trip() -> Result<(), serde_json::Error> {
        let envelope = EventEnvelope {
            event_id: EventId::new(17),
            sequence: EventSequence::new(3),
            host_id: HostId::new(2),
            actor_id: ActorId::new(5),
            logical_timestamp: LogicalTimestamp::new(11),
            payload: PressurePayload::TicketGrant(TicketGrant {
                request_id: PressureRequestId::new(7),
                allocation_id: PhysicalAllocationId::new(13),
                owner: LocalDeviceId::new(1),
                generation: PressureGeneration::new(4),
                bytes: 4096,
            }),
        };

        let json = serde_json::to_string(&envelope)?;
        let decoded: EventEnvelope = serde_json::from_str(&json)?;

        assert_eq!(decoded, envelope);
        Ok(())
    }
}

//! Ticketed non-blocking pressure protocol for machine-wide host memory
//! (`docs/MEMORY_ARCHITECTURE.md` §5.3.1, refined against
//! `specs/tla/PressureProtocol.tla` and `specs/tla/REFINEMENT.md`).
//!
//! # The invariant
//!
//! **No thread ever WAITS while holding a governor lock**, and **no caller is
//! woken successfully until capacity is atomically charged to an owned
//! allocation.** The [`HostGovernor`] ledger lock ([`std::sync::Mutex`]) is held
//! only for short critical sections; awaiting a [`PressureTicket`] holds no
//! governor lock; and a grant reserves bytes and publishes `Granted` *before*
//! the ticket is woken, so a fresh request can never steal reserved bytes.
//!
//! # Module layout
//!
//! * [`HostGovernor`] — public handle; every mutating method locks the ledger
//!   only briefly.
//! * `HostGovernorInner` — shared state: the ledger mutex, the lossless
//!   cancellation mailbox, and the trace sink.
//! * `Ledger` — the single authoritative state (capacity, fixed charge,
//!   per-device reclaimable bytes, per-ticket reserved/claimed extents, and the
//!   configuration generation).
//! * [`PressureTicket`] — a poll/await handle that holds no governor lock and
//!   yields a grant at most once; its `Drop` posts a lossless cancellation.
//!
//! Each linearization point emits a replayable [`EventEnvelope`] at the point
//! named in `REFINEMENT.md`; the ledger, not the trace, remains authoritative.
//! The legacy revision-2 conformance stream remains available while consumers
//! migrate to the serializable envelope.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use onnx_runtime_protocol_trace::{
    ActorId, CONTRACT_REVISION, EventEnvelope, EventId, EventSequence, HostId, LedgerOperation,
    LedgerOperationKind, LocalDeviceId, LogicalTimestamp, MailboxId, MailboxMessage, MailboxSend,
    NullTraceSink, PhysicalAllocationId, PressureEvent, PressureGeneration, PressurePayload,
    PressureRequestId, PressureSnapshot, ProtocolEvent, ProtocolSourceId, ProtocolTraceEvent,
    ProtocolTraceSink, TicketGrant,
};

use crate::governor::ResourceError;

/// Arbitration rounds before a waiting request becomes mature and is ordered
/// ahead of non-matured work (§5.3.1 rule 3).
const AGING_CAP: u64 = 8;
const DEFAULT_MAILBOX_CAPACITY: usize = 1024;

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Sink for serializable HostGovernor protocol envelopes.
///
/// Implementations should record losslessly and return quickly because events
/// are emitted at protocol linearization points.
pub trait PressureTraceSink: Send + Sync {
    fn record(&self, event: EventEnvelope);
}

/// Pressure trace sink used when envelope tracing is disabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullPressureTraceSink;

impl PressureTraceSink for NullPressureTraceSink {
    fn record(&self, _event: EventEnvelope) {}
}

/// Request priority; higher is more urgent.
pub type HostPriority = u8;

/// A request for host (pinned or pageable) memory pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostPageRequest {
    /// The local device that will own the charge.
    pub owner: LocalDeviceId,
    /// Requested byte extent (must be non-zero and satisfiable).
    pub bytes: u64,
    /// Arbitration priority.
    pub priority: HostPriority,
    /// Whether the pages are pinned / non-reclaimable once claimed.
    pub pinned: bool,
}

impl HostPageRequest {
    /// Convenience constructor for a pageable (non-pinned) request.
    pub fn pageable(owner: LocalDeviceId, bytes: u64, priority: HostPriority) -> Self {
        Self {
            owner,
            bytes,
            priority,
            pinned: false,
        }
    }
}

/// A charged host allocation. The `id` is process-unique and never reused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostAllocation {
    /// Process-unique physical identity.
    pub id: PhysicalAllocationId,
    /// The request that produced this allocation.
    pub request_id: PressureRequestId,
    /// Owning local device.
    pub owner: LocalDeviceId,
    /// Charged byte extent.
    pub bytes: u64,
    /// Configuration generation the charge was admitted under.
    pub generation: PressureGeneration,
    /// Whether the allocation is pinned / non-reclaimable.
    pub pinned: bool,
}

/// The state of a pressure ticket (`docs/MEMORY_ARCHITECTURE.md` §5.3.1).
#[derive(Debug)]
pub enum PressureState {
    /// Waiting for capacity.
    Pending(HostPageRequest),
    /// The allocation is already charged before the ticket is woken.
    Granted(HostAllocation),
    /// The ticket claimed the allocation; the ledger retains its authoritative
    /// identity until release and ticket drop is now a no-op.
    Claimed(HostAllocation),
    /// The ticket was cancelled (dropped) before it claimed a grant.
    Cancelled,
    /// The request failed (timeout or reconfiguration).
    Failed(ResourceError),
}

/// Result of a non-blocking [`PressureTicket::try_claim`].
#[derive(Debug)]
pub enum TicketPoll {
    /// Capacity is not yet charged.
    Pending,
    /// The allocation is charged and now owned by the caller.
    Granted(HostAllocation),
    /// The ticket was cancelled.
    Cancelled,
    /// The request failed terminally.
    Failed(ResourceError),
}

/// Configuration for a [`HostGovernor`].
#[derive(Debug, Clone)]
pub struct HostGovernorConfig {
    /// Total machine-wide host memory budget in bytes.
    pub capacity_bytes: u64,
    /// Pinned / otherwise non-reclaimable host bytes charged outside tickets.
    pub fixed_charge_bytes: u64,
    /// Initially evictable host bytes charged to each device but not owned by a
    /// live ticket (mirrors `PressureProtocol!Init`'s `reclaimable`).
    pub initial_reclaimable: Vec<(LocalDeviceId, u64)>,
    /// Stable trace source identity for this governor instance.
    pub source: ProtocolSourceId,
    /// Topology epoch stamped on every emitted event.
    pub topology_epoch: u64,
    /// Host identity stamped on serializable protocol envelopes.
    pub host_id: HostId,
    /// Host-local governor actor identity stamped on protocol envelopes.
    pub actor_id: ActorId,
    /// Stable identity of the cancellation mailbox.
    pub mailbox_id: MailboxId,
    /// Maximum number of simultaneously reserved cancellation messages.
    pub mailbox_capacity: usize,
}

impl HostGovernorConfig {
    /// Creates a config with no fixed charge or reclaimable pre-charge.
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            fixed_charge_bytes: 0,
            initial_reclaimable: Vec::new(),
            source: ProtocolSourceId::new(1),
            topology_epoch: 0,
            host_id: HostId::new(1),
            actor_id: ActorId::new(1),
            mailbox_id: MailboxId::new(1),
            mailbox_capacity: DEFAULT_MAILBOX_CAPACITY,
        }
    }
}

/// Outcome of a deadline firing on a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutOutcome {
    /// A pending request failed (`TimeoutPending`).
    Pending,
    /// A published-but-unclaimed grant failed and returned its charge
    /// (`TimeoutGranted`).
    Granted,
    /// The request was already terminal or claimed; the deadline is a no-op.
    NoOp,
}

/// An independent recomputation of the ledger invariants from authoritative
/// entries (the Invariant gate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostLedgerSnapshot {
    /// Configured capacity.
    pub capacity_bytes: u64,
    /// Fixed / non-reclaimable charge.
    pub fixed_charge_bytes: u64,
    /// Free (unreserved, unclaimed) bytes.
    pub free_bytes: u64,
    /// Sum of per-device reclaimable bytes.
    pub reclaimable_bytes: u64,
    /// Sum of granted-but-unclaimed reservations.
    pub reserved_bytes: u64,
    /// Sum of claimed live allocations.
    pub claimed_bytes: u64,
    /// Recomputed host RAM used = reclaimable + reserved + claimed + fixed.
    pub host_ram_used: u64,
    /// Current configuration generation.
    pub generation: PressureGeneration,
    /// Number of pending requests.
    pub pending: usize,
}

// ─────────────────────────── internal state ────────────────────────────────

/// Per-ticket wakeup channel. This is a ticket-local lock, never the governor
/// lock, so awaiting on it holds no governor lock.
#[derive(Debug)]
struct TicketNotify {
    signaled: Mutex<bool>,
    cv: Condvar,
}

impl TicketNotify {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            signaled: Mutex::new(false),
            cv: Condvar::new(),
        })
    }

    fn wake(&self) {
        let mut guard = lock_recover(&self.signaled);
        *guard = true;
        self.cv.notify_all();
    }

    /// Blocks until woken, consuming the signal. Holds only this ticket-local
    /// lock — never the governor ledger lock.
    fn wait(&self) {
        let mut guard = lock_recover(&self.signaled);
        while !*guard {
            guard = match self.cv.wait(guard) {
                Ok(next) => next,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
        *guard = false;
    }
}

/// A lossless cancellation command posted by [`PressureTicket::drop`].
#[derive(Debug, Clone, Copy)]
struct CancelCommand {
    request_id: PressureRequestId,
    generation: PressureGeneration,
}

/// Lossless non-blocking cancellation mailbox with one pre-reserved slot per
/// live request, so `Drop` never allocates or discards under backpressure
/// (§5.3.1 rule 5).
#[derive(Debug)]
struct CancelMailbox {
    queue: VecDeque<CancelCommand>,
    reserved: usize,
    capacity: usize,
}

impl CancelMailbox {
    fn new(capacity: usize) -> Result<Self, ResourceError> {
        if capacity == 0 {
            return Err(ResourceError::InvalidHostRequest {
                reason: "cancellation mailbox capacity must be non-zero".to_string(),
            });
        }
        Ok(Self {
            queue: VecDeque::with_capacity(capacity),
            reserved: 0,
            capacity,
        })
    }

    /// Reserves one slot for a new request, guaranteeing later `send` cannot
    /// reallocate or fail.
    fn reserve_slot(&mut self) -> Result<(), ResourceError> {
        if self.reserved == self.capacity {
            return Err(ResourceError::HostMailboxBackpressure {
                capacity: self.capacity,
            });
        }
        self.reserved += 1;
        Ok(())
    }

    /// Releases a reserved slot for a ticket that resolved without cancelling.
    fn release_slot(&mut self) {
        self.reserved = self.reserved.saturating_sub(1);
    }

    /// Drains every queued cancellation, releasing their slots.
    fn drain(&mut self) -> Vec<CancelCommand> {
        let drained: Vec<CancelCommand> = self.queue.drain(..).collect();
        self.reserved = self.reserved.saturating_sub(drained.len());
        drained
    }
}

#[derive(Debug, Default)]
struct TraceClock {
    next_sequence: u64,
}

/// One authoritative ledger entry, keyed by [`PressureRequestId`].
#[derive(Debug)]
struct LedgerEntry {
    owner: LocalDeviceId,
    bytes: u64,
    priority: HostPriority,
    pinned: bool,
    generation: PressureGeneration,
    submit_seq: u64,
    age: u64,
    state: PressureState,
    notify: Arc<TicketNotify>,
}

impl LedgerEntry {
    fn reserved_bytes(&self) -> u64 {
        match self.state {
            PressureState::Granted(_) => self.bytes,
            _ => 0,
        }
    }

    fn claimed_bytes(&self) -> u64 {
        match self.state {
            PressureState::Claimed(_) => self.bytes,
            _ => 0,
        }
    }

    fn is_pending(&self) -> bool {
        matches!(self.state, PressureState::Pending(_))
    }
}

/// The single authoritative pressure state, protected by a short-held lock.
#[derive(Debug)]
struct Ledger {
    capacity: u64,
    fixed_charge: u64,
    free: u64,
    reclaimable: BTreeMap<LocalDeviceId, u64>,
    generation: PressureGeneration,
    entries: BTreeMap<PressureRequestId, LedgerEntry>,
    next_request: u64,
    next_alloc: u64,
    next_submit_seq: u64,
    source_sequence: u64,
    reclaim_shortfall: bool,
}

fn checked_add(a: u64, b: u64, op: &'static str) -> Result<u64, ResourceError> {
    a.checked_add(b).ok_or(ResourceError::HostLedgerInvariant {
        operation: op,
        reason: format!("{a} + {b} overflowed u64"),
    })
}

fn checked_sub(a: u64, b: u64, op: &'static str) -> Result<u64, ResourceError> {
    a.checked_sub(b).ok_or(ResourceError::HostLedgerInvariant {
        operation: op,
        reason: format!("{a} - {b} underflowed (negative headroom)"),
    })
}

impl Ledger {
    fn reclaimable_total(&self) -> Result<u64, ResourceError> {
        self.reclaimable.values().try_fold(0u64, |total, bytes| {
            checked_add(total, *bytes, "reclaimable total")
        })
    }

    fn reserved_total(&self) -> Result<u64, ResourceError> {
        self.entries.values().try_fold(0u64, |total, entry| {
            checked_add(total, entry.reserved_bytes(), "reserved total")
        })
    }

    fn claimed_total(&self) -> Result<u64, ResourceError> {
        self.entries.values().try_fold(0u64, |total, entry| {
            checked_add(total, entry.claimed_bytes(), "claimed total")
        })
    }

    fn max_satisfiable(&self) -> u64 {
        // The largest extent that could ever be charged even after full reclaim.
        self.capacity.saturating_sub(self.fixed_charge)
    }
}

// ─────────────────────────── public governor ───────────────────────────────

/// Machine-wide host-memory governor implementing the ticketed non-blocking
/// pressure protocol.
#[derive(Clone)]
pub struct HostGovernor {
    inner: Arc<HostGovernorInner>,
}

struct HostGovernorInner {
    ledger: Mutex<Ledger>,
    mailbox: Mutex<CancelMailbox>,
    sink: Arc<dyn ProtocolTraceSink>,
    pressure_sink: Arc<dyn PressureTraceSink>,
    trace_clock: Mutex<TraceClock>,
    source: ProtocolSourceId,
    topology_epoch: u64,
    host_id: HostId,
    actor_id: ActorId,
    mailbox_id: MailboxId,
}

impl HostGovernor {
    /// Creates a governor with no-op legacy and serializable trace sinks.
    pub fn new(config: HostGovernorConfig) -> Result<Self, ResourceError> {
        Self::with_sinks(
            config,
            Arc::new(NullTraceSink),
            Arc::new(NullPressureTraceSink),
        )
    }

    /// Creates a governor that emits the legacy revision-2 conformance stream.
    pub fn with_sink(
        config: HostGovernorConfig,
        sink: Arc<dyn ProtocolTraceSink>,
    ) -> Result<Self, ResourceError> {
        Self::with_sinks(config, sink, Arc::new(NullPressureTraceSink))
    }

    /// Creates a governor that emits serializable pressure envelopes to `sink`.
    pub fn with_trace_sink(
        config: HostGovernorConfig,
        sink: Arc<dyn PressureTraceSink>,
    ) -> Result<Self, ResourceError> {
        Self::with_sinks(config, Arc::new(NullTraceSink), sink)
    }

    /// Creates a governor that emits both supported pressure trace streams.
    pub fn with_sinks(
        config: HostGovernorConfig,
        sink: Arc<dyn ProtocolTraceSink>,
        pressure_sink: Arc<dyn PressureTraceSink>,
    ) -> Result<Self, ResourceError> {
        let mut reclaimable = BTreeMap::new();
        let mut reclaimable_total = 0u64;
        for (device, bytes) in &config.initial_reclaimable {
            let slot = reclaimable.entry(*device).or_insert(0u64);
            *slot = checked_add(*slot, *bytes, "init reclaimable")?;
            reclaimable_total = checked_add(reclaimable_total, *bytes, "init reclaimable total")?;
        }
        let charged = checked_add(config.fixed_charge_bytes, reclaimable_total, "init charge")?;
        let free = checked_sub(config.capacity_bytes, charged, "init free")?;

        let ledger = Ledger {
            capacity: config.capacity_bytes,
            fixed_charge: config.fixed_charge_bytes,
            free,
            reclaimable,
            generation: PressureGeneration::new(0),
            entries: BTreeMap::new(),
            next_request: 1,
            next_alloc: 1,
            next_submit_seq: 0,
            source_sequence: 0,
            reclaim_shortfall: false,
        };

        Ok(Self {
            inner: Arc::new(HostGovernorInner {
                ledger: Mutex::new(ledger),
                mailbox: Mutex::new(CancelMailbox::new(config.mailbox_capacity)?),
                sink,
                pressure_sink,
                trace_clock: Mutex::new(TraceClock::default()),
                source: config.source,
                topology_epoch: config.topology_epoch,
                host_id: config.host_id,
                actor_id: config.actor_id,
                mailbox_id: config.mailbox_id,
            }),
        })
    }

    /// Requests host pages. Returns an already-ready ticket when capacity is
    /// available (charged immediately), otherwise a pending ticket. Rejects
    /// zero / oversized / impossible-pinned requests up front.
    pub fn request_host_pages(
        &self,
        request: HostPageRequest,
    ) -> Result<PressureTicket, ResourceError> {
        if request.bytes == 0 {
            return Err(ResourceError::InvalidHostRequest {
                reason: "requested zero bytes".to_string(),
            });
        }

        let mut ledger = self.inner.lock_ledger();
        let cancel_wakeups = self.drain_cancellations_locked(&mut ledger)?;

        let budget = ledger.max_satisfiable();
        if request.bytes > budget {
            // Oversized, or an impossible pinned request that could never fit
            // even after full reclaim.
            drop(ledger);
            Self::wake_all(cancel_wakeups);
            return Err(ResourceError::HostQuotaDenied {
                requested_bytes: request.bytes,
                reclaimable_budget_bytes: budget,
            });
        }

        let request_id = PressureRequestId::new(ledger.next_request);
        ledger.next_request += 1;
        let submit_seq = ledger.next_submit_seq;
        ledger.next_submit_seq += 1;
        let generation = ledger.generation;
        let notify = TicketNotify::new();

        // Pre-reserve the cancellation slot at creation time so Drop never
        // allocates or drops under backpressure.
        if let Err(err) = lock_recover(&self.inner.mailbox).reserve_slot() {
            drop(ledger);
            Self::wake_all(cancel_wakeups);
            return Err(err);
        }

        ledger.entries.insert(
            request_id,
            LedgerEntry {
                owner: request.owner,
                bytes: request.bytes,
                priority: request.priority,
                pinned: request.pinned,
                generation,
                submit_seq,
                age: 0,
                state: PressureState::Pending(request.clone()),
                notify: notify.clone(),
            },
        );

        // Submit linearization point: request inserted under the ledger lock
        // with id / generation / owner / checked extent.
        self.inner.emit_locked(
            &mut ledger,
            PressureEvent::Submit {
                request: request_id,
                generation,
                owner: request.owner,
                extent: request.bytes,
            },
        );

        // Immediate charge when capacity is available: arbitration will grant
        // this request if it fits and sorts first. No new capacity was added,
        // so it can only grant the just-submitted request.
        let mut wakeups = cancel_wakeups;
        match self.arbitrate_locked(&mut ledger) {
            Ok(mut arbitrated) => wakeups.append(&mut arbitrated),
            Err(err) => {
                drop(ledger);
                Self::wake_all(wakeups);
                return Err(err);
            }
        }
        self.inner.enqueue_reclaim_notices_locked(&mut ledger);
        drop(ledger);

        // Wake after releasing the ledger lock (harmless for a not-yet-awaited
        // ticket).
        Self::wake_all(wakeups);

        Ok(PressureTicket {
            request_id,
            generation,
            inner: self.inner.clone(),
            notify,
            resolved: false,
        })
    }

    /// Releases a claimed allocation (`Complete`). Credits the exact charge back
    /// to free under the ledger lock, then runs arbitration and wakes newly
    /// granted tickets after unlocking.
    pub fn release_host_pages(&self, allocation: HostAllocation) -> Result<(), ResourceError> {
        let mut ledger = self.inner.lock_ledger();
        let mut wakeups = self.drain_cancellations_locked(&mut ledger)?;

        let recorded = match ledger.entries.get(&allocation.request_id) {
            Some(LedgerEntry {
                state: PressureState::Claimed(recorded),
                ..
            }) => recorded.clone(),
            Some(_) => {
                drop(ledger);
                Self::wake_all(wakeups);
                return Err(ResourceError::HostLedgerInvariant {
                    operation: "release",
                    reason: format!(
                        "request {} is not in the Claimed state",
                        allocation.request_id
                    ),
                });
            }
            None => {
                drop(ledger);
                Self::wake_all(wakeups);
                return Err(ResourceError::HostLedgerInvariant {
                    operation: "release",
                    reason: format!("no ledger entry for request {}", allocation.request_id),
                });
            }
        };
        if allocation != recorded {
            drop(ledger);
            Self::wake_all(wakeups);
            return Err(ResourceError::HostLedgerInvariant {
                operation: "release",
                reason: "released allocation does not match the authoritative ledger charge"
                    .to_string(),
            });
        }

        let bytes = recorded.bytes;
        match checked_add(ledger.free, bytes, "release credit") {
            Ok(free) => ledger.free = free,
            Err(err) => {
                drop(ledger);
                Self::wake_all(wakeups);
                return Err(err);
            }
        }
        self.inner.emit_locked(
            &mut ledger,
            PressureEvent::Release {
                request: recorded.request_id,
                allocation: recorded.id,
                extent: bytes,
            },
        );
        ledger.entries.remove(&recorded.request_id);

        match self.arbitrate_locked(&mut ledger) {
            Ok(mut arbitrated) => wakeups.append(&mut arbitrated),
            Err(err) => {
                drop(ledger);
                Self::wake_all(wakeups);
                return Err(err);
            }
        }
        self.inner.enqueue_reclaim_notices_locked(&mut ledger);
        drop(ledger);

        Self::wake_all(wakeups);
        Ok(())
    }

    /// Credits reclaimed device bytes to the ledger (`Reclaim`), then arbitrates.
    /// Models a reclaim worker completing eviction of a device's host pages.
    pub fn reclaim(&self, device: LocalDeviceId, bytes: u64) -> Result<(), ResourceError> {
        if bytes == 0 {
            return Ok(());
        }
        let mut ledger = self.inner.lock_ledger();
        let mut wakeups = self.drain_cancellations_locked(&mut ledger)?;

        let available = ledger.reclaimable.get(&device).copied().unwrap_or(0);
        let remaining = match checked_sub(available, bytes, "reclaim debit") {
            Ok(remaining) => remaining,
            Err(err) => {
                drop(ledger);
                Self::wake_all(wakeups);
                return Err(err);
            }
        };
        ledger.reclaimable.insert(device, remaining);
        match checked_add(ledger.free, bytes, "reclaim credit") {
            Ok(free) => ledger.free = free,
            Err(err) => {
                drop(ledger);
                Self::wake_all(wakeups);
                return Err(err);
            }
        }

        self.inner.emit_locked(
            &mut ledger,
            PressureEvent::Reclaim {
                owner: device,
                bytes,
            },
        );

        match self.arbitrate_locked(&mut ledger) {
            Ok(mut arbitrated) => wakeups.append(&mut arbitrated),
            Err(err) => {
                drop(ledger);
                Self::wake_all(wakeups);
                return Err(err);
            }
        }
        self.inner.enqueue_reclaim_notices_locked(&mut ledger);
        drop(ledger);

        Self::wake_all(wakeups);
        Ok(())
    }

    /// Increments the configuration generation and fails every prior-generation
    /// pending request under the same ledger lock (`Reconfigure`).
    pub fn reconfigure(&self) -> Result<(), ResourceError> {
        let mut ledger = self.inner.lock_ledger();
        let mut wakeups = self.drain_cancellations_locked(&mut ledger)?;

        let new_generation = ledger.generation.next();
        ledger.generation = new_generation;

        let stale: Vec<PressureRequestId> = ledger
            .entries
            .iter()
            .filter(|(_, e)| e.is_pending() && e.generation != new_generation)
            .map(|(id, _)| *id)
            .collect();

        for id in stale {
            if let Some(entry) = ledger.entries.get_mut(&id) {
                let stale_generation = entry.generation.get();
                entry.state =
                    PressureState::Failed(ResourceError::HostReconfigurationInvalidated {
                        request_id: id.get(),
                        stale_generation,
                        current_generation: new_generation.get(),
                    });
                wakeups.push(entry.notify.clone());
            }
        }

        self.inner
            .emit_locked(&mut ledger, PressureEvent::Reconfigure { new_generation });
        drop(ledger);

        Self::wake_all(wakeups);
        Ok(())
    }

    /// Fires a deadline on a request. A pending request fails; a
    /// published-but-unclaimed grant fails and returns its exact charge; a
    /// claimed or already-terminal request is a no-op. One ledger-ordered winner
    /// races against a concurrent grant/claim.
    pub fn time_out(&self, request_id: PressureRequestId) -> Result<TimeoutOutcome, ResourceError> {
        enum Deadline {
            Pending,
            Granted {
                alloc: PhysicalAllocationId,
                bytes: u64,
            },
            NoOp,
        }
        let mut ledger = self.inner.lock_ledger();
        let mut wakeups = self.drain_cancellations_locked(&mut ledger)?;

        let kind = match ledger.entries.get(&request_id) {
            None => {
                drop(ledger);
                Self::wake_all(wakeups);
                return Ok(TimeoutOutcome::NoOp);
            }
            Some(entry) => match &entry.state {
                PressureState::Pending(_) => Deadline::Pending,
                PressureState::Granted(allocation) => Deadline::Granted {
                    alloc: allocation.id,
                    bytes: entry.bytes,
                },
                _ => Deadline::NoOp,
            },
        };

        match kind {
            Deadline::Pending => {
                let Some(entry) = ledger.entries.get_mut(&request_id) else {
                    drop(ledger);
                    Self::wake_all(wakeups);
                    return Ok(TimeoutOutcome::NoOp);
                };
                entry.state = PressureState::Failed(ResourceError::HostPressureTimeout {
                    request_id: request_id.get(),
                });
                let notify = entry.notify.clone();
                self.inner.emit_locked(
                    &mut ledger,
                    PressureEvent::TimeoutPending {
                        request: request_id,
                    },
                );
                wakeups.push(notify);
                drop(ledger);
                Self::wake_all(wakeups);
                Ok(TimeoutOutcome::Pending)
            }
            Deadline::Granted { alloc, bytes } => {
                match checked_add(ledger.free, bytes, "timeout granted credit") {
                    Ok(free) => ledger.free = free,
                    Err(err) => {
                        drop(ledger);
                        Self::wake_all(wakeups);
                        return Err(err);
                    }
                }
                let Some(entry) = ledger.entries.get_mut(&request_id) else {
                    drop(ledger);
                    Self::wake_all(wakeups);
                    return Ok(TimeoutOutcome::NoOp);
                };
                entry.state = PressureState::Failed(ResourceError::HostPressureTimeout {
                    request_id: request_id.get(),
                });
                let notify = entry.notify.clone();
                self.inner.emit_locked(
                    &mut ledger,
                    PressureEvent::TimeoutGranted {
                        request: request_id,
                        allocation: alloc,
                        extent: bytes,
                    },
                );
                // Reconsider queued tickets now that bytes returned.
                match self.arbitrate_locked(&mut ledger) {
                    Ok(mut arbitrated) => wakeups.append(&mut arbitrated),
                    Err(err) => {
                        drop(ledger);
                        Self::wake_all(wakeups);
                        return Err(err);
                    }
                }
                wakeups.push(notify);
                drop(ledger);
                Self::wake_all(wakeups);
                Ok(TimeoutOutcome::Granted)
            }
            Deadline::NoOp => {
                drop(ledger);
                Self::wake_all(wakeups);
                Ok(TimeoutOutcome::NoOp)
            }
        }
    }

    /// Independently recomputes ledger invariants from authoritative entries
    /// (the Invariant gate). A snapshot that does not equal the ledger, or that
    /// exceeds capacity, is a hard conformance failure.
    pub fn snapshot(&self) -> Result<HostLedgerSnapshot, ResourceError> {
        let mut ledger = self.inner.lock_ledger();
        let wakeups = self.drain_cancellations_locked(&mut ledger)?;

        // Recompute under the lock, then release it and hoist any cancel-drain
        // wakeups out (wake-after-unlock), regardless of the recompute result.
        let result = Self::snapshot_locked(&ledger);
        if let Ok(snapshot) = &result {
            self.inner
                .emit_payload(PressurePayload::Snapshot(PressureSnapshot {
                    generation: snapshot.generation,
                    capacity_bytes: snapshot.capacity_bytes,
                    free_bytes: snapshot.free_bytes,
                    reserved_bytes: snapshot.reserved_bytes,
                    claimed_bytes: snapshot.claimed_bytes,
                    reclaimable_bytes: snapshot.reclaimable_bytes,
                    fixed_bytes: snapshot.fixed_charge_bytes,
                }));
        }
        drop(ledger);
        Self::wake_all(wakeups);
        result
    }

    fn snapshot_locked(ledger: &Ledger) -> Result<HostLedgerSnapshot, ResourceError> {
        let reclaimable_bytes = ledger.reclaimable_total()?;
        let reserved_bytes = ledger.reserved_total()?;
        let claimed_bytes = ledger.claimed_total()?;

        let host_ram_used = checked_add(
            checked_add(
                checked_add(reclaimable_bytes, reserved_bytes, "snapshot used")?,
                claimed_bytes,
                "snapshot used",
            )?,
            ledger.fixed_charge,
            "snapshot used",
        )?;

        let recomputed_free = checked_sub(ledger.capacity, host_ram_used, "snapshot free")?;
        if recomputed_free != ledger.free {
            return Err(ResourceError::HostLedgerInvariant {
                operation: "snapshot",
                reason: format!(
                    "recomputed free {recomputed_free} != ledger free {}",
                    ledger.free
                ),
            });
        }
        if host_ram_used > ledger.capacity {
            return Err(ResourceError::HostLedgerInvariant {
                operation: "snapshot",
                reason: format!(
                    "host_ram_used {host_ram_used} exceeds capacity {}",
                    ledger.capacity
                ),
            });
        }

        let pending = ledger.entries.values().filter(|e| e.is_pending()).count();

        Ok(HostLedgerSnapshot {
            capacity_bytes: ledger.capacity,
            fixed_charge_bytes: ledger.fixed_charge,
            free_bytes: ledger.free,
            reclaimable_bytes,
            reserved_bytes,
            claimed_bytes,
            host_ram_used,
            generation: ledger.generation,
            pending,
        })
    }

    /// Drains and applies any queued cancellations (test/observability hook).
    pub fn process_cancellations(&self) -> Result<(), ResourceError> {
        let mut ledger = self.inner.lock_ledger();
        let wakeups = self.drain_cancellations_locked(&mut ledger)?;
        drop(ledger);
        Self::wake_all(wakeups);
        Ok(())
    }

    // ── internal helpers (all run under the ledger lock) ──

    fn drain_cancellations_locked(
        &self,
        ledger: &mut Ledger,
    ) -> Result<Vec<Arc<TicketNotify>>, ResourceError> {
        let commands = lock_recover(&self.inner.mailbox).drain();
        let mut wakeups = Vec::new();
        for command in commands {
            wakeups.extend(self.apply_cancel_locked(ledger, command)?);
        }
        Ok(wakeups)
    }

    /// Wakes a batch of tickets after the ledger lock has been released. All
    /// wakeups produced under the lock are hoisted out to a call site like this
    /// one so no `notify.wake()` ever runs inside the ledger critical section.
    fn wake_all(wakeups: Vec<Arc<TicketNotify>>) {
        for notify in wakeups {
            notify.wake();
        }
    }

    /// Applies one cancellation under the ledger lock and RETURNS any tickets
    /// that became grantable as a result, so the caller can wake them after
    /// unlocking (the wake-after-unlock discipline). Never wakes under the lock.
    fn apply_cancel_locked(
        &self,
        ledger: &mut Ledger,
        command: CancelCommand,
    ) -> Result<Vec<Arc<TicketNotify>>, ResourceError> {
        enum CancelKind {
            Pending,
            Granted {
                alloc: PhysicalAllocationId,
                bytes: u64,
            },
            ReapTerminal,
            NoOp,
        }
        let kind = match ledger.entries.get(&command.request_id) {
            None => return Ok(Vec::new()),
            Some(entry) if entry.generation != command.generation => {
                // Stale cancellation for a superseded incarnation. Request IDs
                // are never reused, so for a live entry this cannot mismatch;
                // treat any mismatch as a no-op.
                CancelKind::NoOp
            }
            Some(entry) => match &entry.state {
                PressureState::Pending(_) => CancelKind::Pending,
                PressureState::Granted(allocation) => CancelKind::Granted {
                    alloc: allocation.id,
                    bytes: entry.bytes,
                },
                // A claimed ticket owns its allocation (grant-wins); it is live
                // and is reaped only when the claim is released.
                PressureState::Claimed(_) => CancelKind::NoOp,
                // Already-terminal (timed-out / reconfiguration-failed) entries
                // linger only until observed. A cancel-drop is that observation:
                // reap them so recomputation stays O(live). This emits no new
                // event — the terminal transition was already published.
                PressureState::Cancelled | PressureState::Failed(_) => CancelKind::ReapTerminal,
            },
        };

        match kind {
            CancelKind::Pending => {
                let Some(entry) = ledger.entries.get_mut(&command.request_id) else {
                    return Ok(Vec::new());
                };
                entry.state = PressureState::Cancelled;
                self.inner.emit_locked(
                    ledger,
                    PressureEvent::CancelPending {
                        request: command.request_id,
                    },
                );
                self.remove_terminal_locked(ledger, command.request_id);
                Ok(Vec::new())
            }
            CancelKind::Granted { alloc, bytes } => {
                // Cancel-wins over a later claim: return the exact allocation.
                ledger.free = checked_add(ledger.free, bytes, "cancel granted credit")?;
                let Some(entry) = ledger.entries.get_mut(&command.request_id) else {
                    return Ok(Vec::new());
                };
                entry.state = PressureState::Cancelled;
                self.inner.emit_locked(
                    ledger,
                    PressureEvent::CancelGranted {
                        request: command.request_id,
                        allocation: alloc,
                        extent: bytes,
                    },
                );
                self.remove_terminal_locked(ledger, command.request_id);
                // Returned bytes may now satisfy a queued ticket. Hoist the
                // wakeups out to the caller (wake-after-unlock).
                self.arbitrate_locked(ledger)
            }
            CancelKind::ReapTerminal => {
                self.remove_terminal_locked(ledger, command.request_id);
                Ok(Vec::new())
            }
            CancelKind::NoOp => Ok(Vec::new()),
        }
    }

    fn remove_terminal_locked(&self, ledger: &mut Ledger, request_id: PressureRequestId) {
        // Terminal entries hold no allocation; drop them so recomputation stays
        // O(live). The ticket, if still held, observes its resolution through
        // its own resolved flag rather than the ledger.
        ledger.entries.remove(&request_id);
    }

    /// Deterministic priority/FIFO arbitration with starvation-free aging. Reserves
    /// bytes and publishes `Granted` under the ledger lock BEFORE returning the
    /// tickets to wake.
    fn arbitrate_locked(
        &self,
        ledger: &mut Ledger,
    ) -> Result<Vec<Arc<TicketNotify>>, ResourceError> {
        let current_gen = ledger.generation;

        // Age every still-pending, current-generation request (bounded).
        for entry in ledger.entries.values_mut() {
            if entry.is_pending() && entry.generation == current_gen {
                entry.age = (entry.age + 1).min(AGING_CAP);
            }
        }

        // Mature requests are always considered before non-matured requests and
        // remain FIFO among themselves. This bounds starvation even under a
        // continuous stream of maximum-priority arrivals.
        let mut candidates: Vec<(PressureRequestId, bool, HostPriority, u64)> = ledger
            .entries
            .iter()
            .filter(|(_, e)| e.is_pending() && e.generation == current_gen)
            .map(|(id, e)| (*id, e.age >= AGING_CAP, e.priority, e.submit_seq))
            .collect();
        candidates.sort_by(|a, b| {
            b.1.cmp(&a.1).then_with(|| {
                if a.1 {
                    a.3.cmp(&b.3)
                } else {
                    b.2.cmp(&a.2).then_with(|| a.3.cmp(&b.3))
                }
            })
        });

        let mut wakeups = Vec::new();
        for (request_id, matured, _, _) in candidates {
            let Some(bytes) = ledger.entries.get(&request_id).map(|entry| entry.bytes) else {
                return Err(ResourceError::HostLedgerInvariant {
                    operation: "grant",
                    reason: format!("candidate {request_id} disappeared during arbitration"),
                });
            };
            if ledger.free < bytes {
                if matured {
                    // Preserve newly returned capacity for the oldest matured
                    // request instead of letting younger work repeatedly steal
                    // partial headroom before the full extent accumulates.
                    break;
                }
                continue;
            }
            // Reserve bytes first so a fresh request cannot steal them.
            ledger.free = checked_sub(ledger.free, bytes, "grant reserve")?;
            let alloc_id = PhysicalAllocationId::new(ledger.next_alloc);
            ledger.next_alloc += 1;

            let Some(entry) = ledger.entries.get_mut(&request_id) else {
                return Err(ResourceError::HostLedgerInvariant {
                    operation: "grant",
                    reason: format!("candidate {request_id} disappeared before grant"),
                });
            };
            let allocation = HostAllocation {
                id: alloc_id,
                request_id,
                owner: entry.owner,
                bytes,
                generation: entry.generation,
                pinned: entry.pinned,
            };
            let owner = entry.owner;
            entry.state = PressureState::Granted(allocation);
            wakeups.push(entry.notify.clone());

            self.inner.emit_locked(
                ledger,
                PressureEvent::Grant {
                    request: request_id,
                    allocation: alloc_id,
                    owner,
                    extent: bytes,
                },
            );
        }
        Ok(wakeups)
    }

    /// Attempts to claim a granted allocation for `request_id`. Returns the poll
    /// result; on success it atomically transitions `Granted -> Claimed` and
    /// disarms cancellation, all under the ledger lock.
    fn try_claim(&self, request_id: PressureRequestId) -> TicketPoll {
        enum Claimable {
            Pending,
            Granted(HostAllocation),
            AlreadyClaimed,
            Cancelled,
            Failed(ResourceError),
        }
        let mut ledger = self.inner.lock_ledger();
        let wakeups = match self.drain_cancellations_locked(&mut ledger) {
            Ok(wakeups) => wakeups,
            Err(err) => return TicketPoll::Failed(err),
        };

        let outcome = match ledger.entries.get(&request_id) {
            // Removed terminal entry we have not yet observed: treat as
            // cancelled (only self-drop removes a pending entry).
            None => Claimable::Cancelled,
            Some(entry) => match &entry.state {
                PressureState::Pending(_) => Claimable::Pending,
                PressureState::Granted(allocation) => Claimable::Granted(allocation.clone()),
                PressureState::Claimed(_) => Claimable::AlreadyClaimed,
                PressureState::Cancelled => Claimable::Cancelled,
                PressureState::Failed(err) => Claimable::Failed(err.clone()),
            },
        };

        let poll = match outcome {
            Claimable::Pending => TicketPoll::Pending,
            Claimable::Granted(allocation) => {
                let alloc_id = allocation.id;
                let bytes = allocation.bytes;
                let Some(entry) = ledger.entries.get_mut(&request_id) else {
                    drop(ledger);
                    Self::wake_all(wakeups);
                    return TicketPoll::Failed(ResourceError::HostLedgerInvariant {
                        operation: "claim",
                        reason: format!("request {request_id} disappeared before claim"),
                    });
                };
                entry.state = PressureState::Claimed(allocation.clone());
                self.inner.emit_locked(
                    &mut ledger,
                    PressureEvent::Claim {
                        request: request_id,
                        allocation: alloc_id,
                        extent: bytes,
                    },
                );
                TicketPoll::Granted(allocation)
            }
            // Already claimed by us once; a ticket yields at most once.
            Claimable::AlreadyClaimed => TicketPoll::Cancelled,
            Claimable::Cancelled => {
                self.remove_terminal_locked(&mut ledger, request_id);
                TicketPoll::Cancelled
            }
            Claimable::Failed(err) => {
                self.remove_terminal_locked(&mut ledger, request_id);
                TicketPoll::Failed(err)
            }
        };
        drop(ledger);
        Self::wake_all(wakeups);
        poll
    }
}

impl HostGovernorInner {
    fn lock_ledger(&self) -> std::sync::MutexGuard<'_, Ledger> {
        lock_recover(&self.ledger)
    }

    /// Emits one linearization event under the ledger lock. The ledger stays
    /// authoritative; the sink only records.
    fn emit_locked(&self, ledger: &mut Ledger, kind: PressureEvent) {
        let payload = self.payload_for_event(ledger, &kind);
        let sequence = ledger.source_sequence;
        ledger.source_sequence = ledger.source_sequence.saturating_add(1);
        self.sink.record(ProtocolTraceEvent {
            contract_revision: CONTRACT_REVISION,
            topology_epoch: self.topology_epoch,
            source: self.source,
            source_sequence: sequence,
            kind: ProtocolEvent::Pressure(kind),
        });
        self.emit_payload(payload);
    }

    fn payload_for_event(&self, ledger: &Ledger, event: &PressureEvent) -> PressurePayload {
        let entry_metadata = |request: PressureRequestId| {
            ledger
                .entries
                .get(&request)
                .map(|entry| (entry.owner, entry.generation))
        };

        match event {
            PressureEvent::Submit {
                request,
                generation,
                owner,
                extent,
            } => PressurePayload::LedgerOperation(LedgerOperation {
                kind: LedgerOperationKind::Submit,
                request_id: Some(*request),
                allocation_id: None,
                owner: Some(*owner),
                generation: *generation,
                bytes: *extent,
            }),
            PressureEvent::Grant {
                request,
                allocation,
                owner,
                extent,
            } => {
                let generation = entry_metadata(*request)
                    .map(|(_, generation)| generation)
                    .unwrap_or(ledger.generation);
                PressurePayload::TicketGrant(TicketGrant {
                    request_id: *request,
                    allocation_id: *allocation,
                    owner: *owner,
                    generation,
                    bytes: *extent,
                })
            }
            PressureEvent::Claim {
                request,
                allocation,
                extent,
            } => {
                let metadata = entry_metadata(*request);
                PressurePayload::LedgerOperation(LedgerOperation {
                    kind: LedgerOperationKind::Claim,
                    request_id: Some(*request),
                    allocation_id: Some(*allocation),
                    owner: metadata.map(|(owner, _)| owner),
                    generation: metadata
                        .map(|(_, generation)| generation)
                        .unwrap_or(ledger.generation),
                    bytes: *extent,
                })
            }
            PressureEvent::CancelPending { request } => {
                let metadata = entry_metadata(*request);
                PressurePayload::LedgerOperation(LedgerOperation {
                    kind: LedgerOperationKind::Cancel,
                    request_id: Some(*request),
                    allocation_id: None,
                    owner: metadata.map(|(owner, _)| owner),
                    generation: metadata
                        .map(|(_, generation)| generation)
                        .unwrap_or(ledger.generation),
                    bytes: 0,
                })
            }
            PressureEvent::CancelGranted {
                request,
                allocation,
                extent,
            } => {
                let metadata = entry_metadata(*request);
                PressurePayload::LedgerOperation(LedgerOperation {
                    kind: LedgerOperationKind::Cancel,
                    request_id: Some(*request),
                    allocation_id: Some(*allocation),
                    owner: metadata.map(|(owner, _)| owner),
                    generation: metadata
                        .map(|(_, generation)| generation)
                        .unwrap_or(ledger.generation),
                    bytes: *extent,
                })
            }
            PressureEvent::TimeoutPending { request } => {
                let metadata = entry_metadata(*request);
                PressurePayload::LedgerOperation(LedgerOperation {
                    kind: LedgerOperationKind::Timeout,
                    request_id: Some(*request),
                    allocation_id: None,
                    owner: metadata.map(|(owner, _)| owner),
                    generation: metadata
                        .map(|(_, generation)| generation)
                        .unwrap_or(ledger.generation),
                    bytes: 0,
                })
            }
            PressureEvent::TimeoutGranted {
                request,
                allocation,
                extent,
            } => {
                let metadata = entry_metadata(*request);
                PressurePayload::LedgerOperation(LedgerOperation {
                    kind: LedgerOperationKind::Timeout,
                    request_id: Some(*request),
                    allocation_id: Some(*allocation),
                    owner: metadata.map(|(owner, _)| owner),
                    generation: metadata
                        .map(|(_, generation)| generation)
                        .unwrap_or(ledger.generation),
                    bytes: *extent,
                })
            }
            PressureEvent::Release {
                request,
                allocation,
                extent,
            } => {
                let metadata = entry_metadata(*request);
                PressurePayload::LedgerOperation(LedgerOperation {
                    kind: LedgerOperationKind::Release,
                    request_id: Some(*request),
                    allocation_id: Some(*allocation),
                    owner: metadata.map(|(owner, _)| owner),
                    generation: metadata
                        .map(|(_, generation)| generation)
                        .unwrap_or(ledger.generation),
                    bytes: *extent,
                })
            }
            PressureEvent::Reconfigure { new_generation } => {
                PressurePayload::LedgerOperation(LedgerOperation {
                    kind: LedgerOperationKind::Reconfigure,
                    request_id: None,
                    allocation_id: None,
                    owner: None,
                    generation: *new_generation,
                    bytes: 0,
                })
            }
            PressureEvent::Reclaim { owner, bytes } => {
                PressurePayload::LedgerOperation(LedgerOperation {
                    kind: LedgerOperationKind::Reclaim,
                    request_id: None,
                    allocation_id: None,
                    owner: Some(*owner),
                    generation: ledger.generation,
                    bytes: *bytes,
                })
            }
        }
    }

    fn emit_payload(&self, payload: PressurePayload) {
        let mut clock = lock_recover(&self.trace_clock);
        let sequence = clock.next_sequence;
        let Some(next_sequence) = sequence.checked_add(1) else {
            return;
        };
        clock.next_sequence = next_sequence;
        let event_id = ((self.source.get() as u128) << 64) | sequence as u128;
        self.pressure_sink.record(EventEnvelope {
            event_id: EventId::new(event_id),
            sequence: EventSequence::new(sequence),
            host_id: self.host_id,
            actor_id: self.actor_id,
            logical_timestamp: LogicalTimestamp::new(sequence),
            payload,
        });
    }

    fn emit_mailbox_send(&self, request_id: PressureRequestId) {
        self.emit_payload(PressurePayload::MailboxSend(MailboxSend {
            mailbox_id: self.mailbox_id,
            message: MailboxMessage::Cancel { request_id },
        }));
    }

    fn publish_cancel(&self, command: CancelCommand) -> bool {
        let mut mailbox = lock_recover(&self.mailbox);
        if mailbox.queue.len() >= mailbox.reserved {
            return false;
        }

        // Keep the mailbox locked until the send envelope is recorded and the
        // command is queued. A drainer therefore cannot apply cancellation
        // before the corresponding send is visible in the total trace order.
        self.emit_mailbox_send(command.request_id);
        mailbox.queue.push_back(command);
        true
    }

    fn enqueue_reclaim_notices_locked(&self, ledger: &mut Ledger) {
        // Non-blocking: record whether any pending request is unsatisfiable at
        // the current free level. Reclaim workers evict at their own pace and
        // call `reclaim`; this is intentionally not a linearization point.
        ledger.reclaim_shortfall = ledger
            .entries
            .values()
            .any(|e| e.is_pending() && e.bytes > ledger.free);
    }
}

// ─────────────────────────── ticket handle ─────────────────────────────────

/// A poll/await handle for a pending host-page request. Holds no governor lock
/// while awaiting, yields a grant at most once, and posts a lossless
/// cancellation on drop.
pub struct PressureTicket {
    request_id: PressureRequestId,
    generation: PressureGeneration,
    inner: Arc<HostGovernorInner>,
    notify: Arc<TicketNotify>,
    resolved: bool,
}

impl PressureTicket {
    /// The stable identity of this ticket's request.
    pub fn request_id(&self) -> PressureRequestId {
        self.request_id
    }

    /// The configuration generation this request was admitted under.
    pub fn generation(&self) -> PressureGeneration {
        self.generation
    }

    /// Non-blocking claim attempt. On `Granted` the caller now owns the
    /// allocation and the ticket is disarmed.
    pub fn try_claim(&mut self) -> TicketPoll {
        if self.resolved {
            return TicketPoll::Cancelled;
        }
        let governor = HostGovernor {
            inner: self.inner.clone(),
        };
        let poll = governor.try_claim(self.request_id);
        match &poll {
            TicketPoll::Pending => {}
            _ => self.disarm(),
        }
        poll
    }

    /// Blocks until the request is granted or fails, holding no governor lock
    /// while parked. Returns the owned allocation on success.
    pub fn wait(mut self) -> Result<HostAllocation, ResourceError> {
        loop {
            match self.try_claim() {
                TicketPoll::Granted(allocation) => return Ok(allocation),
                TicketPoll::Failed(err) => return Err(err),
                TicketPoll::Cancelled => {
                    return Err(ResourceError::InvalidHostRequest {
                        reason: "ticket was cancelled".to_string(),
                    });
                }
                TicketPoll::Pending => self.notify.wait(),
            }
        }
    }

    /// Marks the ticket resolved and releases its reserved cancellation slot
    /// without sending a cancellation.
    fn disarm(&mut self) {
        if !self.resolved {
            self.resolved = true;
            lock_recover(&self.inner.mailbox).release_slot();
        }
    }
}

impl Drop for PressureTicket {
    fn drop(&mut self) {
        if self.resolved {
            return;
        }
        self.resolved = true;
        // Post a lossless cancellation into the pre-reserved slot. Never blocks
        // on a long-held lock, never allocates, never drops.
        self.inner.publish_cancel(CancelCommand {
            request_id: self.request_id,
            generation: self.generation,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_protocol_trace::{
        InvariantViolation, ReplayHarness, ReplayReducer, ReplayViolation,
    };
    use std::sync::{TryLockError, mpsc};
    use std::thread;
    use std::time::Duration;

    #[derive(Default)]
    struct TraceCollector {
        events: Mutex<Vec<EventEnvelope>>,
    }

    impl TraceCollector {
        fn snapshot(&self) -> Vec<EventEnvelope> {
            lock_recover(&self.events).clone()
        }
    }

    impl PressureTraceSink for TraceCollector {
        fn record(&self, event: EventEnvelope) {
            lock_recover(&self.events).push(event);
        }
    }

    struct BlockingMailboxSendSink {
        collector: Arc<TraceCollector>,
        entered: mpsc::Sender<()>,
        resume: Mutex<mpsc::Receiver<()>>,
    }

    impl PressureTraceSink for BlockingMailboxSendSink {
        fn record(&self, event: EventEnvelope) {
            if matches!(&event.payload, PressurePayload::MailboxSend(_)) {
                self.entered.send(()).unwrap();
                lock_recover(&self.resume).recv().unwrap();
            }
            self.collector.record(event);
        }
    }

    struct CancelObserver(mpsc::Sender<()>);

    impl ProtocolTraceSink for CancelObserver {
        fn record(&self, event: ProtocolTraceEvent) {
            if matches!(
                event.kind,
                ProtocolEvent::Pressure(
                    PressureEvent::CancelPending { .. } | PressureEvent::CancelGranted { .. }
                )
            ) {
                self.0.send(()).unwrap();
            }
        }
    }

    struct NoopReducer;

    impl ReplayReducer<PressurePayload> for NoopReducer {
        type State = ();

        fn apply(
            &self,
            _state: &mut Self::State,
            _event: &EventEnvelope,
        ) -> Result<(), InvariantViolation> {
            Ok(())
        }

        fn check_invariants(&self, _state: &Self::State) -> Result<(), InvariantViolation> {
            Ok(())
        }
    }

    #[test]
    fn ledger_credit_accounting_returns_exact_charge() {
        let governor = HostGovernor::new(HostGovernorConfig::new(16)).unwrap();
        let mut ticket = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(0), 6, 1))
            .unwrap();
        let allocation = match ticket.try_claim() {
            TicketPoll::Granted(allocation) => allocation,
            other => panic!("expected grant, got {other:?}"),
        };

        let charged = governor.snapshot().unwrap();
        assert_eq!(charged.free_bytes, 10);
        assert_eq!(charged.claimed_bytes, 6);
        assert_eq!(charged.reserved_bytes, 0);

        governor.release_host_pages(allocation).unwrap();
        let released = governor.snapshot().unwrap();
        assert_eq!(released.free_bytes, 16);
        assert_eq!(released.claimed_bytes, 0);
        assert_eq!(released.host_ram_used, 0);
    }

    #[test]
    fn forged_release_cannot_refund_or_double_release_authoritative_charge() {
        let governor = HostGovernor::new(HostGovernorConfig::new(8)).unwrap();
        let mut ticket = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(0), 6, 1))
            .unwrap();
        let allocation = match ticket.try_claim() {
            TicketPoll::Granted(allocation) => allocation,
            other => panic!("expected grant, got {other:?}"),
        };
        let mut queued = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(1), 3, 1))
            .unwrap();
        assert!(matches!(queued.try_claim(), TicketPoll::Pending));

        let charged = governor.snapshot().unwrap();
        let forged = HostAllocation {
            id: PhysicalAllocationId::new(allocation.id.get() + 100),
            request_id: allocation.request_id,
            owner: LocalDeviceId::new(allocation.owner.get() + 1),
            bytes: allocation.bytes,
            generation: PressureGeneration::new(allocation.generation.get() + 1),
            pinned: !allocation.pinned,
        };
        assert!(matches!(
            governor.release_host_pages(forged),
            Err(ResourceError::HostLedgerInvariant {
                operation: "release",
                ..
            })
        ));
        assert_eq!(governor.snapshot().unwrap(), charged);
        assert!(matches!(queued.try_claim(), TicketPoll::Pending));

        governor.release_host_pages(allocation.clone()).unwrap();
        let queued_allocation = match queued.try_claim() {
            TicketPoll::Granted(allocation) => allocation,
            other => panic!("expected queued grant, got {other:?}"),
        };
        let after_release = governor.snapshot().unwrap();
        assert!(matches!(
            governor.release_host_pages(allocation),
            Err(ResourceError::HostLedgerInvariant {
                operation: "release",
                ..
            })
        ));
        assert_eq!(governor.snapshot().unwrap(), after_release);
        governor.release_host_pages(queued_allocation).unwrap();
    }

    #[test]
    fn tickets_grant_queue_and_deny_under_pressure() {
        let governor = HostGovernor::new(HostGovernorConfig::new(8)).unwrap();
        let denied =
            governor.request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(0), 9, 1));
        assert!(matches!(denied, Err(ResourceError::HostQuotaDenied { .. })));

        let mut first = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(0), 8, 1))
            .unwrap();
        let first_allocation = match first.try_claim() {
            TicketPoll::Granted(allocation) => allocation,
            other => panic!("expected first grant, got {other:?}"),
        };
        let mut queued = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(1), 1, 1))
            .unwrap();
        assert!(matches!(queued.try_claim(), TicketPoll::Pending));

        governor.release_host_pages(first_allocation).unwrap();
        let queued_allocation = match queued.try_claim() {
            TicketPoll::Granted(allocation) => allocation,
            other => panic!("expected queued grant, got {other:?}"),
        };
        governor.release_host_pages(queued_allocation).unwrap();
    }

    #[test]
    fn cancellation_mailbox_applies_bounded_backpressure() {
        let mut config = HostGovernorConfig::new(16);
        config.mailbox_capacity = 1;
        let governor = HostGovernor::new(config).unwrap();
        let ticket = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(0), 16, 1))
            .unwrap();

        let blocked =
            governor.request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(1), 1, 1));
        assert!(matches!(
            blocked,
            Err(ResourceError::HostMailboxBackpressure { capacity: 1 })
        ));

        drop(ticket);
        governor.process_cancellations().unwrap();
        assert_eq!(governor.snapshot().unwrap().free_bytes, 16);
    }

    #[test]
    fn cancellation_delivery_waits_for_its_mailbox_send_envelope() {
        let collector = Arc::new(TraceCollector::default());
        let (cancel_applied, cancel_applied_rx) = mpsc::channel();
        let (send_entered, send_entered_rx) = mpsc::channel();
        let (resume_send, resume_send_rx) = mpsc::channel();
        let governor = HostGovernor::with_sinks(
            HostGovernorConfig::new(1),
            Arc::new(CancelObserver(cancel_applied)),
            Arc::new(BlockingMailboxSendSink {
                collector: collector.clone(),
                entered: send_entered,
                resume: Mutex::new(resume_send_rx),
            }),
        )
        .unwrap();
        let ticket = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(0), 1, 1))
            .unwrap();
        let request_id = ticket.request_id();

        // Stop MailboxSend publication inside the trace sink. Once the sink has
        // been entered, fixed code must still hold the mailbox lock; the old
        // queue-before-send order has already exposed the cancellation.
        let drop_thread = thread::spawn(move || drop(ticket));
        send_entered_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        match governor.inner.mailbox.try_lock() {
            Err(TryLockError::WouldBlock) => {}
            Ok(mailbox) => panic!(
                "mailbox unlocked with {} queued cancellations before MailboxSend completed",
                mailbox.queue.len()
            ),
            Err(TryLockError::Poisoned(_)) => panic!("mailbox poisoned"),
        }

        let process_governor = governor.clone();
        let process_thread = thread::spawn(move || {
            process_governor.process_cancellations().unwrap();
        });
        assert!(
            cancel_applied_rx
                .recv_timeout(Duration::from_millis(50))
                .is_err(),
            "cancellation was applied before MailboxSend could be recorded"
        );

        resume_send.send(()).unwrap();
        drop_thread.join().unwrap();
        cancel_applied_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        process_thread.join().unwrap();

        let events = collector.snapshot();
        let send_index = events
            .iter()
            .position(|event| {
                matches!(
                    &event.payload,
                    PressurePayload::MailboxSend(MailboxSend {
                        message: MailboxMessage::Cancel {
                            request_id: sent_request
                        },
                        ..
                    }) if *sent_request == request_id
                )
            })
            .unwrap();
        let cancel_index = events
            .iter()
            .position(|event| {
                matches!(
                    &event.payload,
                    PressurePayload::LedgerOperation(operation)
                        if operation.kind == LedgerOperationKind::Cancel
                            && operation.request_id == Some(request_id)
                )
            })
            .unwrap();
        assert!(send_index < cancel_index);
        assert!(events[send_index].sequence.get() < events[cancel_index].sequence.get());
        assert_eq!(governor.snapshot().unwrap().free_bytes, 1);
    }

    #[test]
    fn matured_low_priority_ticket_accumulates_capacity_despite_recurring_high_priority_arrivals() {
        let governor = HostGovernor::new(HostGovernorConfig::new(3)).unwrap();
        let mut holders = Vec::new();
        for _ in 0..3 {
            let mut blocker = governor
                .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(0), 1, u8::MAX))
                .unwrap();
            holders.push(match blocker.try_claim() {
                TicketPoll::Granted(allocation) => allocation,
                other => panic!("expected blocker grant, got {other:?}"),
            });
        }
        let mut waiting = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(1), 3, 0))
            .unwrap();
        let mut pending_newcomers = Vec::new();

        for round in 0..(AGING_CAP + 3) {
            let mut newcomer = governor
                .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(2), 1, u8::MAX))
                .unwrap();
            governor.release_host_pages(holders.remove(0)).unwrap();

            match waiting.try_claim() {
                TicketPoll::Granted(allocation) => {
                    assert!(round < AGING_CAP);
                    pending_newcomers.push(newcomer);
                    drop(pending_newcomers);
                    governor.process_cancellations().unwrap();
                    governor.release_host_pages(allocation).unwrap();
                    assert_eq!(governor.snapshot().unwrap().free_bytes, 3);
                    return;
                }
                TicketPoll::Pending => match newcomer.try_claim() {
                    TicketPoll::Granted(allocation) => holders.push(allocation),
                    TicketPoll::Pending => pending_newcomers.push(newcomer),
                    other => panic!("high-priority ticket unexpectedly resolved: {other:?}"),
                },
                other => panic!("waiting ticket unexpectedly resolved: {other:?}"),
            }
        }

        panic!("low-priority ticket starved beyond the aging bound");
    }

    #[test]
    fn snapshot_is_a_consistent_point_in_time_view() {
        let mut config = HostGovernorConfig::new(20);
        config.fixed_charge_bytes = 2;
        config.initial_reclaimable = vec![(LocalDeviceId::new(0), 5)];
        let governor = HostGovernor::new(config).unwrap();

        let granted = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(1), 10, 2))
            .unwrap();
        let pending = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(2), 4, 1))
            .unwrap();
        let snapshot = governor.snapshot().unwrap();

        assert_eq!(snapshot.fixed_charge_bytes, 2);
        assert_eq!(snapshot.reclaimable_bytes, 5);
        assert_eq!(snapshot.reserved_bytes, 10);
        assert_eq!(snapshot.claimed_bytes, 0);
        assert_eq!(snapshot.free_bytes, 3);
        assert_eq!(snapshot.host_ram_used, 17);
        assert_eq!(snapshot.pending, 1);

        drop(granted);
        drop(pending);
        governor.process_cancellations().unwrap();
    }

    #[test]
    fn envelope_stream_round_trips_and_replays() -> Result<(), ReplayViolation> {
        let collector = Arc::new(TraceCollector::default());
        let mut config = HostGovernorConfig::new(4);
        config.initial_reclaimable = vec![(LocalDeviceId::new(0), 4)];
        config.source = ProtocolSourceId::new(9);
        config.host_id = HostId::new(3);
        config.actor_id = ActorId::new(7);
        config.mailbox_id = MailboxId::new(11);
        let governor = HostGovernor::with_trace_sink(config, collector.clone()).unwrap();

        let pending = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(1), 1, 1))
            .unwrap();
        let pending_id = pending.request_id();
        drop(pending);
        governor.process_cancellations().unwrap();
        governor.reclaim(LocalDeviceId::new(0), 1).unwrap();
        let mut granted = governor
            .request_host_pages(HostPageRequest::pageable(LocalDeviceId::new(1), 1, 1))
            .unwrap();
        let allocation = match granted.try_claim() {
            TicketPoll::Granted(allocation) => allocation,
            other => panic!("expected grant, got {other:?}"),
        };
        let state = governor.snapshot().unwrap();
        governor.release_host_pages(allocation).unwrap();

        let events = collector.snapshot();
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            PressurePayload::MailboxSend(MailboxSend {
                message: MailboxMessage::Cancel { request_id },
                ..
            }) if *request_id == pending_id
        )));
        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload, PressurePayload::TicketGrant(_)))
        );
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            PressurePayload::Snapshot(snapshot)
                if snapshot.free_bytes == state.free_bytes
                    && snapshot.claimed_bytes == state.claimed_bytes
        )));

        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            let decoded: EventEnvelope = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, event);
        }

        let report = ReplayHarness::new(NoopReducer).check((), events)?;
        assert_eq!(report.events_checked, collector.snapshot().len());
        Ok(())
    }
}

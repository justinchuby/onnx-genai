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
//! Each linearization point emits exactly one [`ProtocolTraceEvent`] at the
//! point named in `REFINEMENT.md`; the ledger, not the trace, remains the
//! authoritative owner.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};

use onnx_runtime_protocol_trace::{
    CONTRACT_REVISION, LocalDeviceId, NullTraceSink, PhysicalAllocationId, PressureEvent,
    PressureGeneration, PressureRequestId, ProtocolEvent, ProtocolSourceId, ProtocolTraceEvent,
    ProtocolTraceSink,
};

use crate::governor::ResourceError;

/// Bounded-aging cap: an older satisfiable request's effective priority rises by
/// at most this many levels, so continuous high-priority arrivals cannot starve
/// it indefinitely (§5.3.1 rule 3).
const AGING_CAP: u64 = 8;

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
    /// The ticket claimed the allocation; ticket drop is now a no-op.
    Claimed,
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
        let mut guard = self.signaled.lock().expect("ticket notify poisoned");
        *guard = true;
        self.cv.notify_all();
    }

    /// Blocks until woken, consuming the signal. Holds only this ticket-local
    /// lock — never the governor ledger lock.
    fn wait(&self) {
        let mut guard = self.signaled.lock().expect("ticket notify poisoned");
        while !*guard {
            guard = self.cv.wait(guard).expect("ticket notify poisoned");
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
#[derive(Debug, Default)]
struct CancelMailbox {
    queue: VecDeque<CancelCommand>,
    reserved: usize,
}

impl CancelMailbox {
    /// Reserves one slot for a new request, guaranteeing later `send` cannot
    /// reallocate or fail.
    fn reserve_slot(&mut self) {
        self.reserved += 1;
        let deficit = self.reserved.saturating_sub(self.queue.len());
        if self.queue.capacity() < self.reserved {
            self.queue.reserve(deficit);
        }
        debug_assert!(self.queue.capacity() >= self.reserved);
    }

    /// Posts a cancellation into a pre-reserved slot. Never blocks, allocates,
    /// or drops.
    fn send(&mut self, command: CancelCommand) {
        debug_assert!(
            self.queue.len() < self.reserved,
            "cancellation mailbox slot was not reserved"
        );
        self.queue.push_back(command);
    }

    /// Releases a reserved slot for a ticket that resolved without cancelling.
    fn release_slot(&mut self) {
        debug_assert!(self.reserved > 0);
        self.reserved -= 1;
    }

    /// Drains every queued cancellation, releasing their slots.
    fn drain(&mut self) -> Vec<CancelCommand> {
        let drained: Vec<CancelCommand> = self.queue.drain(..).collect();
        self.reserved -= drained.len();
        drained
    }
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
            PressureState::Claimed => self.bytes,
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
    fn reclaimable_total(&self) -> u64 {
        self.reclaimable.values().copied().sum()
    }

    fn reserved_total(&self) -> u64 {
        self.entries.values().map(LedgerEntry::reserved_bytes).sum()
    }

    fn claimed_total(&self) -> u64 {
        self.entries.values().map(LedgerEntry::claimed_bytes).sum()
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
    source: ProtocolSourceId,
    topology_epoch: u64,
}

impl HostGovernor {
    /// Creates a governor with a no-op trace sink.
    pub fn new(config: HostGovernorConfig) -> Result<Self, ResourceError> {
        Self::with_sink(config, Arc::new(NullTraceSink))
    }

    /// Creates a governor that emits linearization events to `sink`.
    pub fn with_sink(
        config: HostGovernorConfig,
        sink: Arc<dyn ProtocolTraceSink>,
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
                mailbox: Mutex::new(CancelMailbox::default()),
                sink,
                source: config.source,
                topology_epoch: config.topology_epoch,
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
        let cancel_wakeups = self.drain_cancellations_locked(&mut ledger);

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
        self.inner
            .mailbox
            .lock()
            .expect("mailbox poisoned")
            .reserve_slot();

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
        let mut wakeups = self.drain_cancellations_locked(&mut ledger);

        let entry = match ledger.entries.get(&allocation.request_id) {
            Some(entry) => entry,
            None => {
                drop(ledger);
                Self::wake_all(wakeups);
                return Err(ResourceError::HostLedgerInvariant {
                    operation: "release",
                    reason: format!("no ledger entry for request {}", allocation.request_id),
                });
            }
        };
        if !matches!(entry.state, PressureState::Claimed) {
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
        if entry.bytes != allocation.bytes {
            drop(ledger);
            Self::wake_all(wakeups);
            return Err(ResourceError::HostLedgerInvariant {
                operation: "release",
                reason: "released extent does not match the claimed extent".to_string(),
            });
        }

        let bytes = allocation.bytes;
        match checked_add(ledger.free, bytes, "release credit") {
            Ok(free) => ledger.free = free,
            Err(err) => {
                drop(ledger);
                Self::wake_all(wakeups);
                return Err(err);
            }
        }
        ledger.entries.remove(&allocation.request_id);

        self.inner.emit_locked(
            &mut ledger,
            PressureEvent::Release {
                request: allocation.request_id,
                allocation: allocation.id,
                extent: bytes,
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

    /// Credits reclaimed device bytes to the ledger (`Reclaim`), then arbitrates.
    /// Models a reclaim worker completing eviction of a device's host pages.
    pub fn reclaim(&self, device: LocalDeviceId, bytes: u64) -> Result<(), ResourceError> {
        if bytes == 0 {
            return Ok(());
        }
        let mut ledger = self.inner.lock_ledger();
        let mut wakeups = self.drain_cancellations_locked(&mut ledger);

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
        let mut wakeups = self.drain_cancellations_locked(&mut ledger);

        let new_generation = ledger.generation.next();
        ledger.generation = new_generation;

        let stale: Vec<PressureRequestId> = ledger
            .entries
            .iter()
            .filter(|(_, e)| e.is_pending() && e.generation != new_generation)
            .map(|(id, _)| *id)
            .collect();

        for id in stale {
            let entry = ledger.entries.get_mut(&id).expect("entry present");
            let stale_generation = entry.generation.get();
            entry.state = PressureState::Failed(ResourceError::HostReconfigurationInvalidated {
                request_id: id.get(),
                stale_generation,
                current_generation: new_generation.get(),
            });
            wakeups.push(entry.notify.clone());
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
        let mut wakeups = self.drain_cancellations_locked(&mut ledger);

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
                let entry = ledger.entries.get_mut(&request_id).expect("entry");
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
                let entry = ledger.entries.get_mut(&request_id).expect("entry");
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
        let wakeups = self.drain_cancellations_locked(&mut ledger);

        // Recompute under the lock, then release it and hoist any cancel-drain
        // wakeups out (wake-after-unlock), regardless of the recompute result.
        let result = Self::snapshot_locked(&ledger);
        drop(ledger);
        Self::wake_all(wakeups);
        result
    }

    fn snapshot_locked(ledger: &Ledger) -> Result<HostLedgerSnapshot, ResourceError> {
        let reclaimable_bytes = ledger.reclaimable_total();
        let reserved_bytes = ledger.reserved_total();
        let claimed_bytes = ledger.claimed_total();

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
    pub fn process_cancellations(&self) {
        let mut ledger = self.inner.lock_ledger();
        let wakeups = self.drain_cancellations_locked(&mut ledger);
        drop(ledger);
        Self::wake_all(wakeups);
    }

    // ── internal helpers (all run under the ledger lock) ──

    fn drain_cancellations_locked(&self, ledger: &mut Ledger) -> Vec<Arc<TicketNotify>> {
        let commands = self.inner.mailbox.lock().expect("mailbox poisoned").drain();
        let mut wakeups = Vec::new();
        for command in commands {
            wakeups.extend(self.apply_cancel_locked(ledger, command));
        }
        wakeups
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
    ) -> Vec<Arc<TicketNotify>> {
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
            None => return Vec::new(),
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
                PressureState::Claimed => CancelKind::NoOp,
                // Already-terminal (timed-out / reconfiguration-failed) entries
                // linger only until observed. A cancel-drop is that observation:
                // reap them so recomputation stays O(live). This emits no new
                // event — the terminal transition was already published.
                PressureState::Cancelled | PressureState::Failed(_) => CancelKind::ReapTerminal,
            },
        };

        match kind {
            CancelKind::Pending => {
                ledger
                    .entries
                    .get_mut(&command.request_id)
                    .expect("entry")
                    .state = PressureState::Cancelled;
                self.inner.emit_locked(
                    ledger,
                    PressureEvent::CancelPending {
                        request: command.request_id,
                    },
                );
                self.remove_terminal_locked(ledger, command.request_id);
                Vec::new()
            }
            CancelKind::Granted { alloc, bytes } => {
                // Cancel-wins over a later claim: return the exact allocation.
                if let Ok(free) = checked_add(ledger.free, bytes, "cancel granted credit") {
                    ledger.free = free;
                }
                ledger
                    .entries
                    .get_mut(&command.request_id)
                    .expect("entry")
                    .state = PressureState::Cancelled;
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
                self.arbitrate_locked(ledger).unwrap_or_default()
            }
            CancelKind::ReapTerminal => {
                self.remove_terminal_locked(ledger, command.request_id);
                Vec::new()
            }
            CancelKind::NoOp => Vec::new(),
        }
    }

    fn remove_terminal_locked(&self, ledger: &mut Ledger, request_id: PressureRequestId) {
        // Terminal entries hold no allocation; drop them so recomputation stays
        // O(live). The ticket, if still held, observes its resolution through
        // its own resolved flag rather than the ledger.
        ledger.entries.remove(&request_id);
    }

    /// Deterministic priority/FIFO arbitration with bounded aging. Reserves
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

        // Order candidates by effective priority (with aging), then FIFO.
        let mut candidates: Vec<(PressureRequestId, u64, u64)> = ledger
            .entries
            .iter()
            .filter(|(_, e)| e.is_pending() && e.generation == current_gen)
            .map(|(id, e)| {
                let effective = e.priority as u64 + e.age.min(AGING_CAP);
                (*id, effective, e.submit_seq)
            })
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));

        let mut wakeups = Vec::new();
        for (request_id, _, _) in candidates {
            let bytes = ledger.entries.get(&request_id).expect("entry").bytes;
            if ledger.free < bytes {
                continue;
            }
            // Reserve bytes first so a fresh request cannot steal them.
            ledger.free = checked_sub(ledger.free, bytes, "grant reserve")?;
            let alloc_id = PhysicalAllocationId::new(ledger.next_alloc);
            ledger.next_alloc += 1;

            let entry = ledger.entries.get_mut(&request_id).expect("entry");
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
        let wakeups = self.drain_cancellations_locked(&mut ledger);

        let outcome = match ledger.entries.get(&request_id) {
            // Removed terminal entry we have not yet observed: treat as
            // cancelled (only self-drop removes a pending entry).
            None => Claimable::Cancelled,
            Some(entry) => match &entry.state {
                PressureState::Pending(_) => Claimable::Pending,
                PressureState::Granted(allocation) => Claimable::Granted(allocation.clone()),
                PressureState::Claimed => Claimable::AlreadyClaimed,
                PressureState::Cancelled => Claimable::Cancelled,
                PressureState::Failed(err) => Claimable::Failed(err.clone()),
            },
        };

        let poll = match outcome {
            Claimable::Pending => TicketPoll::Pending,
            Claimable::Granted(allocation) => {
                let alloc_id = allocation.id;
                let bytes = allocation.bytes;
                ledger.entries.get_mut(&request_id).expect("entry").state = PressureState::Claimed;
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
        self.ledger.lock().expect("ledger poisoned")
    }

    /// Emits one linearization event under the ledger lock. The ledger stays
    /// authoritative; the sink only records.
    fn emit_locked(&self, ledger: &mut Ledger, kind: PressureEvent) {
        let sequence = ledger.source_sequence;
        ledger.source_sequence += 1;
        self.sink.record(ProtocolTraceEvent {
            contract_revision: CONTRACT_REVISION,
            topology_epoch: self.topology_epoch,
            source: self.source,
            source_sequence: sequence,
            kind: ProtocolEvent::Pressure(kind),
        });
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
            self.inner
                .mailbox
                .lock()
                .expect("mailbox poisoned")
                .release_slot();
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
        self.inner
            .mailbox
            .lock()
            .expect("mailbox poisoned")
            .send(CancelCommand {
                request_id: self.request_id,
                generation: self.generation,
            });
    }
}

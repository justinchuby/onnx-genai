//! Independent conformance replay checker + deterministic campaigns for the
//! ticketed non-blocking pressure protocol.
//!
//! This test crate encodes the abstract `specs/tla/PressureProtocol.tla` state
//! machine as an INDEPENDENT reducer ([`PressureProtocol`]) that shares only the
//! identity/event *definitions* with the implementation — never its transition
//! code (`specs/tla/REFINEMENT.md` § "Independent Replay Checker"). The campaigns
//! drive the real [`HostGovernor`], capture a lossless trace, and feed it to the
//! generic [`ReplayChecker`], which must accept it and confirm every invariant
//! after every transition and reject leftover active tickets at the end.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use onnx_genai_scheduler::pressure::{
    HostAllocation, HostGovernor, HostGovernorConfig, HostPageRequest, HostPriority,
    PressureTicket, TicketPoll, TimeoutOutcome,
};
use onnx_runtime_protocol_trace::checker::{
    AbstractProtocol, ActionResolution, BoundedTraceCollector, ConformanceFailure, FailureReason,
    ReplayChecker, TraceEnd, TraceIntegrityError, TraceSnapshot,
};
use onnx_runtime_protocol_trace::{
    LocalDeviceId, PhysicalAllocationId, PressureEvent, PressureGeneration, PressureRequestId,
    ProtocolEvent, ProtocolTraceEvent,
};

// ─────────────────────── independent abstract reducer ──────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbsState {
    Pending,
    Granted,
    Claimed,
    Cancelled,
    Failed,
    Completed,
}

#[derive(Debug, Clone)]
struct AbsTicket {
    extent: u64,
    generation: u64,
    state: AbsState,
    reserved: u64,
    claimed: u64,
    claim_count: u32,
    allocation: Option<PhysicalAllocationId>,
}

/// The independent abstract ledger (mirrors `PressureProtocol.tla` variables).
#[derive(Debug, Clone)]
struct PressureAbstract {
    capacity: u64,
    fixed_charge: u64,
    free: u64,
    reclaimable: BTreeMap<LocalDeviceId, u64>,
    generation: u64,
    tickets: BTreeMap<PressureRequestId, AbsTicket>,
    seen_allocations: BTreeSet<PhysicalAllocationId>,
}

impl PressureAbstract {
    fn from_config(config: &HostGovernorConfig) -> Self {
        let mut reclaimable = BTreeMap::new();
        let mut reclaimable_total = 0u64;
        for (device, bytes) in &config.initial_reclaimable {
            *reclaimable.entry(*device).or_insert(0) += *bytes;
            reclaimable_total += *bytes;
        }
        let free = config.capacity_bytes - config.fixed_charge_bytes - reclaimable_total;
        Self {
            capacity: config.capacity_bytes,
            fixed_charge: config.fixed_charge_bytes,
            free,
            reclaimable,
            generation: 0,
            tickets: BTreeMap::new(),
            seen_allocations: BTreeSet::new(),
        }
    }

    fn reclaimable_total(&self) -> u64 {
        self.reclaimable.values().copied().sum()
    }

    fn reserved_total(&self) -> u64 {
        self.tickets.values().map(|t| t.reserved).sum()
    }

    fn claimed_total(&self) -> u64 {
        self.tickets.values().map(|t| t.claimed).sum()
    }
}

/// The independent abstract state machine.
struct PressureProtocol;

impl PressureProtocol {
    fn pressure_event(kind: &ProtocolEvent) -> Option<&PressureEvent> {
        match kind {
            ProtocolEvent::Pressure(event) => Some(event),
            _ => None,
        }
    }
}

impl AbstractProtocol for PressureProtocol {
    type State = PressureAbstract;
    type Action = PressureEvent;

    fn resolve(&self, state: &Self::State, kind: &ProtocolEvent) -> ActionResolution<Self::Action> {
        let event = match Self::pressure_event(kind) {
            Some(event) => event,
            None => return ActionResolution::None("non-pressure event in pressure trace".into()),
        };

        let enabled: Result<(), String> = match event {
            PressureEvent::Submit {
                request,
                generation,
                extent,
                ..
            } => {
                if state.tickets.contains_key(request) {
                    Err(format!("Submit for existing request {request}"))
                } else if *extent == 0 {
                    Err("Submit with zero extent".into())
                } else if *extent > state.capacity - state.fixed_charge {
                    Err("Submit extent exceeds reclaimable budget".into())
                } else if generation.get() != state.generation {
                    Err("Submit uses a stale generation".into())
                } else {
                    Ok(())
                }
            }
            PressureEvent::Grant {
                request,
                allocation,
                extent,
                ..
            } => match state.tickets.get(request) {
                Some(t) if t.state == AbsState::Pending => {
                    if t.generation != state.generation {
                        Err("Grant for stale-generation pending".into())
                    } else if *extent != t.extent {
                        Err("Grant extent mismatch".into())
                    } else if state.free < *extent {
                        Err("Grant exceeds free".into())
                    } else if state.seen_allocations.contains(allocation) {
                        Err(format!("Grant reuses allocation id {allocation}"))
                    } else {
                        Ok(())
                    }
                }
                _ => Err(format!("Grant for non-pending request {request}")),
            },
            PressureEvent::Claim {
                request,
                allocation,
                extent,
            } => match state.tickets.get(request) {
                Some(t) if t.state == AbsState::Granted => {
                    if t.reserved != *extent || t.extent != *extent {
                        Err("Claim extent mismatch".into())
                    } else if t.claim_count != 0 {
                        Err("Claim would exceed at-most-once".into())
                    } else if t.allocation != Some(*allocation) {
                        Err("Claim allocation mismatch".into())
                    } else {
                        Ok(())
                    }
                }
                _ => Err(format!("Claim for non-granted request {request}")),
            },
            PressureEvent::CancelPending { request } => match state.tickets.get(request) {
                Some(t) if t.state == AbsState::Pending => Ok(()),
                _ => Err(format!("CancelPending for non-pending request {request}")),
            },
            PressureEvent::CancelGranted {
                request,
                allocation,
                extent,
            } => match state.tickets.get(request) {
                Some(t) if t.state == AbsState::Granted => {
                    if t.reserved != *extent {
                        Err("CancelGranted extent mismatch".into())
                    } else if t.allocation != Some(*allocation) {
                        Err("CancelGranted allocation mismatch".into())
                    } else {
                        Ok(())
                    }
                }
                _ => Err(format!("CancelGranted for non-granted request {request}")),
            },
            PressureEvent::TimeoutPending { request } => match state.tickets.get(request) {
                Some(t) if t.state == AbsState::Pending => Ok(()),
                _ => Err(format!("TimeoutPending for non-pending request {request}")),
            },
            PressureEvent::TimeoutGranted {
                request,
                allocation,
                extent,
            } => match state.tickets.get(request) {
                Some(t) if t.state == AbsState::Granted => {
                    if t.reserved != *extent {
                        Err("TimeoutGranted extent mismatch".into())
                    } else if t.allocation != Some(*allocation) {
                        Err("TimeoutGranted allocation mismatch".into())
                    } else {
                        Ok(())
                    }
                }
                _ => Err(format!("TimeoutGranted for non-granted request {request}")),
            },
            PressureEvent::Release {
                request,
                allocation,
                extent,
            } => match state.tickets.get(request) {
                Some(t) if t.state == AbsState::Claimed => {
                    if t.claimed != *extent {
                        Err("Release extent mismatch".into())
                    } else if t.allocation != Some(*allocation) {
                        Err("Release allocation mismatch".into())
                    } else {
                        Ok(())
                    }
                }
                _ => Err(format!("Release for non-claimed request {request}")),
            },
            PressureEvent::Reconfigure { new_generation } => {
                if new_generation.get() == state.generation + 1 {
                    Ok(())
                } else {
                    Err("Reconfigure must increment generation by one".into())
                }
            }
            PressureEvent::Reclaim { owner, bytes } => {
                let available = state.reclaimable.get(owner).copied().unwrap_or(0);
                if *bytes == 0 {
                    Err("Reclaim of zero bytes".into())
                } else if available < *bytes {
                    Err("Reclaim exceeds device reclaimable".into())
                } else {
                    Ok(())
                }
            }
        };

        match enabled {
            Ok(()) => ActionResolution::Enabled(event.clone()),
            Err(reason) => ActionResolution::None(reason),
        }
    }

    fn apply(&self, state: &mut Self::State, action: &Self::Action) -> Result<(), String> {
        let sub = |a: u64, b: u64, what: &str| {
            a.checked_sub(b)
                .ok_or_else(|| format!("checked-sub underflow in {what}"))
        };
        let add = |a: u64, b: u64, what: &str| {
            a.checked_add(b)
                .ok_or_else(|| format!("checked-add overflow in {what}"))
        };

        match action {
            PressureEvent::Submit {
                request,
                generation,
                extent,
                ..
            } => {
                state.tickets.insert(
                    *request,
                    AbsTicket {
                        extent: *extent,
                        generation: generation.get(),
                        state: AbsState::Pending,
                        reserved: 0,
                        claimed: 0,
                        claim_count: 0,
                        allocation: None,
                    },
                );
            }
            PressureEvent::Grant {
                request,
                allocation,
                extent,
                ..
            } => {
                state.free = sub(state.free, *extent, "Grant")?;
                state.seen_allocations.insert(*allocation);
                let t = state.tickets.get_mut(request).expect("ticket");
                t.state = AbsState::Granted;
                t.reserved = *extent;
                t.allocation = Some(*allocation);
            }
            PressureEvent::Claim {
                request, extent, ..
            } => {
                let t = state.tickets.get_mut(request).expect("ticket");
                t.reserved = sub(t.reserved, *extent, "Claim reserved")?;
                t.claimed = add(t.claimed, *extent, "Claim claimed")?;
                t.claim_count += 1;
                t.state = AbsState::Claimed;
            }
            PressureEvent::CancelPending { request } => {
                state.tickets.get_mut(request).expect("ticket").state = AbsState::Cancelled;
            }
            PressureEvent::CancelGranted {
                request, extent, ..
            } => {
                state.free = add(state.free, *extent, "CancelGranted")?;
                let t = state.tickets.get_mut(request).expect("ticket");
                t.reserved = sub(t.reserved, *extent, "CancelGranted reserved")?;
                t.state = AbsState::Cancelled;
            }
            PressureEvent::TimeoutPending { request } => {
                state.tickets.get_mut(request).expect("ticket").state = AbsState::Failed;
            }
            PressureEvent::TimeoutGranted {
                request, extent, ..
            } => {
                state.free = add(state.free, *extent, "TimeoutGranted")?;
                let t = state.tickets.get_mut(request).expect("ticket");
                t.reserved = sub(t.reserved, *extent, "TimeoutGranted reserved")?;
                t.state = AbsState::Failed;
            }
            PressureEvent::Release {
                request, extent, ..
            } => {
                state.free = add(state.free, *extent, "Release")?;
                let t = state.tickets.get_mut(request).expect("ticket");
                t.claimed = sub(t.claimed, *extent, "Release claimed")?;
                t.state = AbsState::Completed;
            }
            PressureEvent::Reconfigure { new_generation } => {
                state.generation = new_generation.get();
                for t in state.tickets.values_mut() {
                    if t.state == AbsState::Pending {
                        t.state = AbsState::Failed;
                    }
                }
            }
            PressureEvent::Reclaim { owner, bytes } => {
                let available = state.reclaimable.get(owner).copied().unwrap_or(0);
                let remaining = sub(available, *bytes, "Reclaim debit")?;
                state.reclaimable.insert(*owner, remaining);
                state.free = add(state.free, *bytes, "Reclaim credit")?;
            }
        }
        Ok(())
    }

    fn check_invariants(&self, state: &Self::State) -> Result<(), String> {
        // CapacityConserved
        let sum = state.fixed_charge
            + state.free
            + state.reclaimable_total()
            + state.reserved_total()
            + state.claimed_total();
        if sum != state.capacity {
            return Err(format!(
                "CapacityConserved: {sum} != capacity {}",
                state.capacity
            ));
        }

        for (id, t) in &state.tickets {
            // GrantedIsChargedExactly
            if t.state == AbsState::Granted && t.reserved != t.extent {
                return Err(format!("GrantedIsChargedExactly violated for {id}"));
            }
            if t.state != AbsState::Granted && t.reserved != 0 {
                return Err(format!("reserved leak on non-granted {id}"));
            }
            // ClaimedIsOwnedExactly
            if t.state == AbsState::Claimed && t.claimed != t.extent {
                return Err(format!("ClaimedIsOwnedExactly violated for {id}"));
            }
            if t.state != AbsState::Claimed && t.claimed != 0 {
                return Err(format!("claimed leak on non-claimed {id}"));
            }
            // ClaimedAtMostOnce
            if t.claim_count > 1 {
                return Err(format!("ClaimedAtMostOnce violated for {id}"));
            }
            if t.state == AbsState::Claimed && t.claim_count != 1 {
                return Err(format!("Claimed without claim_count==1 for {id}"));
            }
            // TerminalHasNoAllocation
            if matches!(
                t.state,
                AbsState::Cancelled | AbsState::Failed | AbsState::Completed
            ) && (t.reserved != 0 || t.claimed != 0)
            {
                return Err(format!("TerminalHasNoAllocation violated for {id}"));
            }
            // PendingUsesCurrentGeneration
            if t.state == AbsState::Pending && t.generation != state.generation {
                return Err(format!("PendingUsesCurrentGeneration violated for {id}"));
            }
        }
        Ok(())
    }

    fn active_entries(&self, state: &Self::State) -> Vec<String> {
        state
            .tickets
            .iter()
            .filter(|(_, t)| {
                matches!(
                    t.state,
                    AbsState::Pending | AbsState::Granted | AbsState::Claimed
                )
            })
            .map(|(id, t)| format!("{id} in {:?}", t.state))
            .collect()
    }
}

// ─────────────────────────── deterministic PRNG ────────────────────────────

/// A tiny deterministic splitmix64 PRNG so campaigns are reproducible from a
/// fixed seed with an explicit decision trace (`REFINEMENT.md` § "Required Test
/// Campaigns").
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

// ─────────────────────────── shared helpers ────────────────────────────────

fn dev(id: u32) -> LocalDeviceId {
    LocalDeviceId::new(id)
}

/// Replays a captured trace against a fresh independent abstract state built
/// from `config`, asserting acceptance under a clean end.
fn assert_conformant(config: &HostGovernorConfig, collector: &BoundedTraceCollector) {
    let checker = ReplayChecker::new(PressureProtocol);
    let initial = PressureAbstract::from_config(config);
    let report = checker
        .run(initial, &collector.snapshot(), TraceEnd::Clean)
        .unwrap_or_else(|failure| {
            panic!("independent replay checker rejected a valid trace: {failure:?}")
        });
    assert!(report.events_checked > 0, "expected non-empty trace");
}

/// Cleans up any still-live tickets (claim+release, or drop-to-cancel) so the
/// trace ends with no active entries, flushes cancellations, and asserts the
/// implementation's own independent snapshot passes the invariant gate.
fn drain_to_clean(governor: &HostGovernor, tickets: Vec<PressureTicket>) {
    let mut claimed: Vec<HostAllocation> = Vec::new();
    let mut tickets = tickets;
    for ticket in tickets.iter_mut() {
        if let TicketPoll::Granted(allocation) = ticket.try_claim() {
            claimed.push(allocation);
        }
    }
    drop(tickets); // pending tickets post a lossless CancelPending on drop
    governor.process_cancellations();
    for allocation in claimed {
        governor.release_host_pages(allocation).expect("release");
    }
    let snapshot = governor.snapshot().expect("snapshot invariant gate");
    assert_eq!(snapshot.pending, 0, "no pending requests should remain");
}

// ─────────────────────────── campaigns ─────────────────────────────────────

/// Grant vs claim vs release, on an immediately-ready ticket.
#[test]
fn campaign_grant_claim_release() {
    let config = HostGovernorConfig::new(8);
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let governor = HostGovernor::with_sink(config.clone(), collector.clone()).unwrap();

    let mut ticket = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 8, 0))
        .unwrap();
    let allocation = match ticket.try_claim() {
        TicketPoll::Granted(a) => a,
        other => panic!("expected immediate grant, got {other:?}"),
    };
    assert_eq!(allocation.bytes, 8);
    governor.release_host_pages(allocation).unwrap();

    drain_to_clean(&governor, vec![ticket]);
    assert_conformant(&config, &collector);
}

/// Cancel a pending ticket and cancel a granted-but-unclaimed ticket.
#[test]
fn campaign_cancel_pending_and_granted() {
    let mut config = HostGovernorConfig::new(20);
    config.initial_reclaimable = vec![(dev(0), 20)];
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let governor = HostGovernor::with_sink(config.clone(), collector.clone()).unwrap();

    // Pending (no free): drop -> CancelPending.
    let pending = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 10, 0))
        .unwrap();
    drop(pending);
    governor.process_cancellations();

    // Free some, grant a fresh ticket, then drop it unclaimed -> CancelGranted.
    governor.reclaim(dev(0), 10).unwrap();
    let granted = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 10, 0))
        .unwrap();
    assert_eq!(governor.snapshot().unwrap().reserved_bytes, 10);
    drop(granted);
    governor.process_cancellations();
    assert_eq!(governor.snapshot().unwrap().reserved_bytes, 0);

    drain_to_clean(&governor, vec![]);
    assert_conformant(&config, &collector);
}

/// Timeout on a pending request and timeout racing a published grant.
#[test]
fn campaign_timeout_pending_and_granted() {
    let mut config = HostGovernorConfig::new(20);
    config.initial_reclaimable = vec![(dev(0), 20)];
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let governor = HostGovernor::with_sink(config.clone(), collector.clone()).unwrap();

    let pending = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 12, 0))
        .unwrap();
    assert_eq!(
        governor.time_out(pending.request_id()).unwrap(),
        TimeoutOutcome::Pending
    );
    drop(pending);

    governor.reclaim(dev(0), 20).unwrap();
    let granted = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 12, 0))
        .unwrap();
    // Timeout wins the race against a not-yet-claimed grant.
    assert_eq!(
        governor.time_out(granted.request_id()).unwrap(),
        TimeoutOutcome::Granted
    );
    assert_eq!(governor.snapshot().unwrap().reserved_bytes, 0);
    drop(granted);

    drain_to_clean(&governor, vec![]);
    assert_conformant(&config, &collector);
}

/// Reconfiguration fails every prior-generation pending request.
#[test]
fn campaign_reconfigure_invalidates_pending() {
    let mut config = HostGovernorConfig::new(30);
    config.initial_reclaimable = vec![(dev(0), 30)];
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let governor = HostGovernor::with_sink(config.clone(), collector.clone()).unwrap();

    let a = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 10, 0))
        .unwrap();
    let b = governor
        .request_host_pages(HostPageRequest::pageable(dev(1), 12, 0))
        .unwrap();
    assert_eq!(governor.snapshot().unwrap().pending, 2);

    governor.reconfigure().unwrap();
    let snap = governor.snapshot().unwrap();
    assert_eq!(snap.pending, 0);
    assert_eq!(snap.generation, PressureGeneration::new(1));
    drop(a);
    drop(b);

    drain_to_clean(&governor, vec![]);
    assert_conformant(&config, &collector);
}

/// Multiple variable-sized tickets from the same and different devices.
#[test]
fn campaign_multiple_variable_sized_tickets() {
    let mut config = HostGovernorConfig::new(100);
    config.initial_reclaimable = vec![(dev(0), 40), (dev(1), 40)];
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let governor = HostGovernor::with_sink(config.clone(), collector.clone()).unwrap();

    // free = 20 initially.
    let sizes = [(dev(0), 5u64), (dev(1), 7), (dev(0), 8), (dev(1), 3)];
    let mut tickets = Vec::new();
    for (owner, bytes) in sizes {
        tickets.push(
            governor
                .request_host_pages(HostPageRequest::pageable(owner, bytes, 0))
                .unwrap(),
        );
    }
    // Free the rest so every ticket can eventually be granted.
    governor.reclaim(dev(0), 40).unwrap();
    governor.reclaim(dev(1), 40).unwrap();

    drain_to_clean(&governor, tickets);
    assert_conformant(&config, &collector);
}

/// Cancellation-mailbox saturation and ticket drop during (would-be) wakeup.
#[test]
fn campaign_mailbox_saturation() {
    let mut config = HostGovernorConfig::new(1000);
    config.initial_reclaimable = vec![(dev(0), 1000)];
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let governor = HostGovernor::with_sink(config.clone(), collector.clone()).unwrap();

    // Flood the mailbox: many pending tickets, all dropped without draining.
    let mut tickets = Vec::new();
    for _ in 0..64 {
        tickets.push(
            governor
                .request_host_pages(HostPageRequest::pageable(dev(0), 50, 0))
                .unwrap(),
        );
    }
    // Drop them all at once (saturates the pre-reserved slots); no cancel lost.
    drop(tickets);
    // One drain applies all 64 cancellations losslessly.
    governor.process_cancellations();
    let snapshot = governor.snapshot().unwrap();
    assert_eq!(snapshot.pending, 0);
    assert_eq!(snapshot.free_bytes, 0); // capacity still fully reclaimable-charged

    drain_to_clean(&governor, vec![]);
    assert_conformant(&config, &collector);
}

/// Checked-arithmetic boundaries at zero, exact capacity, and maximum extent.
#[test]
fn campaign_arithmetic_boundaries() {
    let config = HostGovernorConfig::new(16);
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let governor = HostGovernor::with_sink(config.clone(), collector.clone()).unwrap();

    // Zero extent is rejected up front.
    assert!(
        governor
            .request_host_pages(HostPageRequest::pageable(dev(0), 0, 0))
            .is_err()
    );
    // Oversized (> capacity - fixed) is rejected up front.
    assert!(
        governor
            .request_host_pages(HostPageRequest::pageable(dev(0), 17, 0))
            .is_err()
    );

    // Exact-capacity / maximum-extent admission succeeds and is charged.
    let mut ticket = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 16, 0))
        .unwrap();
    let allocation = match ticket.try_claim() {
        TicketPoll::Granted(a) => a,
        other => panic!("expected max-extent grant, got {other:?}"),
    };
    governor.release_host_pages(allocation).unwrap();

    drain_to_clean(&governor, vec![ticket]);
    assert_conformant(&config, &collector);
}

/// The §5.3.1 "two devices under pressure" scenario.
#[test]
fn campaign_two_devices_under_pressure() {
    // capacity 24; both devices hold 12 reclaimable each; free starts at 0.
    let mut config = HostGovernorConfig::new(24);
    config.initial_reclaimable = vec![(dev(0), 12), (dev(1), 12)];
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let governor = HostGovernor::with_sink(config.clone(), collector.clone()).unwrap();

    // 1. GPU 0 pending ticket A for 10; GPU 1 pending ticket B for 8.
    let mut a = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 10, 0))
        .unwrap();
    let mut b = governor
        .request_host_pages(HostPageRequest::pageable(dev(1), 8, 0))
        .unwrap();
    assert_eq!(governor.snapshot().unwrap().pending, 2);

    // 2. Reclaim workers release 12; arbitration grants exactly one (A, FIFO).
    governor.reclaim(dev(0), 12).unwrap();
    let snap = governor.snapshot().unwrap();
    assert_eq!(snap.reserved_bytes, 10, "exactly ticket A is granted");
    assert_eq!(snap.free_bytes, 2);

    // 3. A fresh 12 request cannot steal A's reserved bytes.
    let mut c = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 12, 0))
        .unwrap();
    assert!(matches!(c.try_claim(), TicketPoll::Pending));
    assert_eq!(governor.snapshot().unwrap().reserved_bytes, 10);

    // 4. A later 6 release permits the second ticket (B) to be granted.
    governor.reclaim(dev(1), 6).unwrap();
    assert_eq!(
        governor.snapshot().unwrap().reserved_bytes,
        18,
        "A(10) + B(8) now reserved"
    );

    // Claim A (owns 10).
    let alloc_a = match a.try_claim() {
        TicketPoll::Granted(alloc) => alloc,
        other => panic!("A should be granted: {other:?}"),
    };

    // 5. A timeout racing B's grant has one ledger-ordered winner (timeout), no
    // allocation leaked, and no waiter proceeds without ownership.
    assert_eq!(
        governor.time_out(b.request_id()).unwrap(),
        TimeoutOutcome::Granted
    );
    assert!(matches!(b.try_claim(), TicketPoll::Failed(_)));
    // B's 8 bytes returned to free; C(12) still cannot fit (free = 8 < 12).
    assert!(matches!(c.try_claim(), TicketPoll::Pending));

    // Wind down cleanly: release A, cancel C.
    governor.release_host_pages(alloc_a).unwrap();
    drop(a);
    drop(b);
    drop(c);
    governor.process_cancellations();

    let snapshot = governor.snapshot().unwrap();
    assert_eq!(snapshot.pending, 0);
    // No leak: everything sums back to capacity.
    assert_eq!(
        snapshot.reclaimable_bytes
            + snapshot.free_bytes
            + snapshot.reserved_bytes
            + snapshot.claimed_bytes,
        snapshot.capacity_bytes
    );

    assert_conformant(&config, &collector);
}

// ─────────────── seeded randomized campaign (determinism gate) ──────────────

/// Runs a seeded random campaign and returns the recorded trace so two runs can
/// be compared for reproducibility.
fn run_seeded_campaign(seed: u64) -> Vec<ProtocolTraceEvent> {
    let mut config = HostGovernorConfig::new(64);
    config.initial_reclaimable = vec![(dev(0), 32), (dev(1), 32)];
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let governor = HostGovernor::with_sink(config.clone(), collector.clone()).unwrap();

    let mut rng = SplitMix64::new(seed);
    let mut tickets: Vec<PressureTicket> = Vec::new();
    let mut claimed: Vec<HostAllocation> = Vec::new();

    for _ in 0..200 {
        match rng.below(6) {
            0 => {
                let owner = dev(rng.below(2) as u32);
                let bytes = 1 + rng.below(20);
                let priority = rng.below(3) as HostPriority;
                if let Ok(ticket) =
                    governor.request_host_pages(HostPageRequest::pageable(owner, bytes, priority))
                {
                    tickets.push(ticket);
                }
            }
            1 => {
                if !tickets.is_empty() {
                    let idx = rng.below(tickets.len() as u64) as usize;
                    if let TicketPoll::Granted(alloc) = tickets[idx].try_claim() {
                        claimed.push(alloc);
                        tickets.swap_remove(idx);
                    }
                }
            }
            2 => {
                if !tickets.is_empty() {
                    let idx = rng.below(tickets.len() as u64) as usize;
                    let ticket = tickets.swap_remove(idx);
                    drop(ticket);
                    governor.process_cancellations();
                }
            }
            3 => {
                if !tickets.is_empty() {
                    let idx = rng.below(tickets.len() as u64) as usize;
                    let _ = governor.time_out(tickets[idx].request_id());
                }
            }
            4 => {
                let owner = dev(rng.below(2) as u32);
                let bytes = 1 + rng.below(16);
                // Best-effort: a too-large reclaim returns an error (no event),
                // keeping the trace lossless and deterministic.
                let _ = governor.reclaim(owner, bytes);
            }
            _ => {
                if !claimed.is_empty() {
                    let idx = rng.below(claimed.len() as u64) as usize;
                    let alloc = claimed.swap_remove(idx);
                    governor.release_host_pages(alloc).unwrap();
                }
            }
        }
        // Snapshot invariant gate on every step.
        governor.snapshot().expect("snapshot invariant gate");
    }

    // Wind down to a clean end.
    for ticket in tickets.iter_mut() {
        if let TicketPoll::Granted(alloc) = ticket.try_claim() {
            claimed.push(alloc);
        }
    }
    drop(tickets);
    governor.process_cancellations();
    for alloc in claimed {
        governor.release_host_pages(alloc).unwrap();
    }
    governor.snapshot().expect("final snapshot invariant gate");

    // The independent checker must accept the whole trace at a clean end.
    let checker = ReplayChecker::new(PressureProtocol);
    let initial = PressureAbstract::from_config(&config);
    checker
        .run(initial, &collector.snapshot(), TraceEnd::Clean)
        .expect("seeded campaign trace must be conformant");

    collector.snapshot().events
}

#[test]
fn campaign_seeded_is_reproducible() {
    // Same seed twice -> byte-identical traces (determinism).
    let run_a = run_seeded_campaign(0xC0FFEE);
    let run_b = run_seeded_campaign(0xC0FFEE);
    assert_eq!(run_a, run_b, "same seed must produce identical traces");
    assert!(!run_a.is_empty());

    // A different fixed seed -> still a valid, accepted, reproducible trace.
    let run_c = run_seeded_campaign(0x1234_5678);
    let run_c2 = run_seeded_campaign(0x1234_5678);
    assert_eq!(run_c, run_c2, "second seed must also be reproducible");
}

// ─────────────────── self-tests: the checker rejects bad traces ─────────────

fn valid_small_trace() -> (HostGovernorConfig, Vec<ProtocolTraceEvent>) {
    let config = HostGovernorConfig::new(8);
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let governor = HostGovernor::with_sink(config.clone(), collector.clone()).unwrap();
    let mut ticket = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 8, 0))
        .unwrap();
    if let TicketPoll::Granted(alloc) = ticket.try_claim() {
        governor.release_host_pages(alloc).unwrap();
    }
    drop(ticket);
    governor.process_cancellations();
    (config, collector.snapshot().events)
}

fn snapshot_of(events: Vec<ProtocolTraceEvent>) -> TraceSnapshot {
    TraceSnapshot {
        events,
        integrity: Ok(()),
    }
}

fn run_checker(
    config: &HostGovernorConfig,
    events: Vec<ProtocolTraceEvent>,
    end: TraceEnd,
) -> Result<(), ConformanceFailure> {
    let checker = ReplayChecker::new(PressureProtocol);
    checker
        .run(
            PressureAbstract::from_config(config),
            &snapshot_of(events),
            end,
        )
        .map(|_| ())
}

#[test]
fn checker_rejects_unknown_contract_revision() {
    let (config, mut events) = valid_small_trace();
    events[0].contract_revision = 999;
    let failure = run_checker(&config, events, TraceEnd::Clean).unwrap_err();
    assert!(matches!(
        failure.reason,
        FailureReason::UnknownContractRevision { .. }
    ));
    assert_eq!(failure.prefix_len, 1);
}

#[test]
fn checker_rejects_duplicate_source_sequence() {
    let (config, mut events) = valid_small_trace();
    let dup = events[0].source_sequence;
    events[1].source_sequence = dup;
    let failure = run_checker(&config, events, TraceEnd::Clean).unwrap_err();
    assert!(matches!(
        failure.reason,
        FailureReason::DuplicateSourceSequence { .. }
    ));
}

#[test]
fn checker_rejects_reordered_event() {
    let (config, mut events) = valid_small_trace();
    // Swap Submit and Grant: Grant now precedes Submit -> impossible/reordered.
    events.swap(0, 1);
    // Re-sequence so the envelope check passes and the action check fails.
    for (i, e) in events.iter_mut().enumerate() {
        e.source_sequence = i as u64;
    }
    let failure = run_checker(&config, events, TraceEnd::Clean).unwrap_err();
    assert!(matches!(
        failure.reason,
        FailureReason::NoEnabledAction { .. }
    ));
    assert_eq!(
        failure.prefix_len, 1,
        "fails at the smallest offending prefix"
    );
}

#[test]
fn checker_rejects_leftover_active_ticket() {
    // A Submit with no terminal resolution leaves an active pending ticket.
    let mut config = HostGovernorConfig::new(20);
    config.initial_reclaimable = vec![(dev(0), 20)];
    let collector = Arc::new(BoundedTraceCollector::unbounded());
    let governor = HostGovernor::with_sink(config.clone(), collector.clone()).unwrap();
    let pending = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 10, 0))
        .unwrap();
    // Forget it so drop does not emit a cancel; the trace has only a Submit.
    std::mem::forget(pending);
    let events = collector.snapshot().events;

    // Clean end must reject the leftover pending ticket.
    let failure = run_checker(&config, events.clone(), TraceEnd::Clean).unwrap_err();
    assert!(matches!(
        failure.reason,
        FailureReason::LeftoverActive { .. }
    ));
    // A declared crash boundary tolerates it.
    assert!(run_checker(&config, events, TraceEnd::CrashBoundary).is_ok());
}

#[test]
fn checker_rejects_lossy_trace() {
    let (config, events) = valid_small_trace();
    let lossy = TraceSnapshot {
        events,
        integrity: Err(TraceIntegrityError::BufferOverflow { at_index: 1 }),
    };
    let checker = ReplayChecker::new(PressureProtocol);
    let failure = checker
        .run(
            PressureAbstract::from_config(&config),
            &lossy,
            TraceEnd::Clean,
        )
        .unwrap_err();
    assert!(matches!(failure.reason, FailureReason::TraceIntegrity(_)));
}

// ─────────────────── concurrency smoke: blocking wait path ──────────────────

/// Exercises the real Condvar wakeup: a waiter parks holding no governor lock
/// until a reclaim on another thread charges its allocation.
#[test]
fn blocking_wait_wakes_on_reclaim() {
    let mut config = HostGovernorConfig::new(16);
    config.initial_reclaimable = vec![(dev(0), 16)];
    let governor = HostGovernor::new(config).unwrap();

    let ticket = governor
        .request_host_pages(HostPageRequest::pageable(dev(0), 16, 0))
        .unwrap();

    let reclaimer = governor.clone();
    let handle = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(20));
        reclaimer.reclaim(dev(0), 16).unwrap();
    });

    let allocation = ticket
        .wait()
        .expect("wait should yield an owned allocation");
    assert_eq!(allocation.bytes, 16);
    governor.release_host_pages(allocation).unwrap();
    handle.join().unwrap();
    assert_eq!(governor.snapshot().unwrap().pending, 0);
}

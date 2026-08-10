//! Adapting [`HostGovernor`] to the tiered lease contract.
//!
//! [`HostGovernor`] is the authoritative accounting for machine-wide host
//! memory: a ledger refined against `specs/tla/PressureProtocol.tla`, with
//! priority arbitration and a reclaim protocol. The lease contract in
//! `onnx-runtime-memory-governor` is what components hold their memory
//! through. Until now they were unconnected, so lease holders charged a
//! *separate* private ledger and the two disagreed about what the machine was
//! using.
//!
//! # Why `reserve` never blocks
//!
//! [`HostGovernor::request_host_pages`] returns a ticket that may need to wait
//! while other holders reclaim. [`MemoryGovernor::reserve`] must not wait: it is
//! called from an ONNX Runtime allocator callback (`GovernedAllocator`), and
//! ONNX Runtime may hold locks across `Alloc` that the reclaim path needs to
//! make progress. Waiting there is a deadlock, not a slow path.
//!
//! So this adapter polls the ticket once and treats `Pending` as refusal. That
//! is a deliberate narrowing, not an oversight: a caller who *can* afford to
//! wait should use [`HostGovernor::request_host_pages`] directly and get the
//! full protocol, including arbitration.
//!
//! # Why released bytes come back in whole allocations
//!
//! The lease contract releases *byte counts* — a lease can [`grow`] into a
//! second charge and [`shrink`] by part of one. [`HostGovernor`] charges and
//! releases *whole allocations*. The adapter bridges that with a credit: bytes
//! released accumulate, and whole allocations are handed back as the credit
//! covers them.
//!
//! The consequence worth stating: a partial `shrink` does not immediately
//! return anything to the host governor, because there is no allocation small
//! enough to give back yet. It is returned as soon as one is covered. The
//! adapter can never release more than it was given, which is the property that
//! matters.
//!
//! [`grow`]: onnx_runtime_memory_governor::MemoryLease::grow
//! [`shrink`]: onnx_runtime_memory_governor::MemoryLease::shrink

use std::sync::{Arc, Mutex};

use onnx_runtime_memory_governor::{
    HolderId, LeaseAccounting, MemoryError, MemoryGovernor, MemoryLease, MemoryRole, Tier,
};
use onnx_runtime_protocol_trace::LocalDeviceId;

use crate::pressure::{HostAllocation, HostGovernor, HostPageRequest, HostPriority, TicketPoll};

/// Charges the lease contract's reservations against a [`HostGovernor`].
///
/// Only [`Tier::Host`] is grantable: this governs host RAM and nothing else.
/// Device and disk are refused rather than silently granted, because a caller
/// who asked for device memory and got host memory would be told their device
/// budget was charged when it was not.
pub struct HostGovernorAccounting {
    governor: Arc<HostGovernor>,
    owner: LocalDeviceId,
    priority: HostPriority,
    outstanding: Mutex<Outstanding>,
}

// `LeaseAccounting` requires `Debug` so a lease can be inspected; `HostGovernor`
// does not implement it, and its interior state is behind a mutex that must not
// be taken to format a struct. Report what this adapter itself knows.
impl std::fmt::Debug for HostGovernorAccounting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostGovernorAccounting")
            .field("owner", &self.owner)
            .field("priority", &self.priority)
            .field("charged_bytes", &self.charged_bytes())
            .field("pending_credit", &self.pending_credit())
            .finish()
    }
}

#[derive(Debug, Default)]
struct Outstanding {
    /// Allocations charged to the host governor and not yet given back.
    allocations: Vec<HostAllocation>,
    /// Bytes released by lease holders that no outstanding allocation is small
    /// enough to cover yet.
    credit: u64,
}

impl HostGovernorAccounting {
    /// Charge `governor` on behalf of `owner` at `priority`.
    pub fn new(governor: Arc<HostGovernor>, owner: LocalDeviceId, priority: HostPriority) -> Self {
        Self {
            governor,
            owner,
            priority,
            outstanding: Mutex::new(Outstanding::default()),
        }
    }

    /// Bytes currently charged to the host governor through this adapter.
    pub fn charged_bytes(&self) -> u64 {
        let outstanding = self.lock();
        outstanding.allocations.iter().map(|a| a.bytes).sum()
    }

    /// Bytes released by holders but not yet handed back, because no
    /// outstanding allocation is small enough to cover them.
    pub fn pending_credit(&self) -> u64 {
        self.lock().credit
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Outstanding> {
        // A poisoned lock means a holder panicked mid-update. The accounting is
        // still consistent -- every mutation below is a single push or drain --
        // so recovering is better than turning one panic into every future
        // allocation failing.
        self.outstanding
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn refuse(tier: Tier, requested: u64, available: u64, role: MemoryRole) -> MemoryError {
        MemoryError::TierExhausted {
            tier: tier.name(),
            requested,
            used: 0,
            limit: available,
            available,
            role,
        }
    }
}

impl LeaseAccounting for HostGovernorAccounting {
    fn try_claim(&self, tier: Tier, bytes: u64, role: MemoryRole) -> Result<(), MemoryError> {
        if tier != Tier::Host {
            return Err(Self::refuse(tier, bytes, 0, role));
        }
        if bytes == 0 {
            return Ok(());
        }

        // Spend credit before asking for more: bytes a holder already gave back
        // are ours to re-lend, and asking the host governor for them again
        // would double-charge the machine.
        {
            let mut outstanding = self.lock();
            if outstanding.credit >= bytes {
                outstanding.credit -= bytes;
                return Ok(());
            }
        }

        let mut ticket = self
            .governor
            .request_host_pages(HostPageRequest::pageable(self.owner, bytes, self.priority))
            .map_err(|_| Self::refuse(tier, bytes, 0, role))?;

        match ticket.try_claim() {
            TicketPoll::Granted(allocation) => {
                self.lock().allocations.push(allocation);
                Ok(())
            }
            // Pending means capacity needs someone else to reclaim first.
            // Waiting is not available here; see the module docs.
            TicketPoll::Pending | TicketPoll::Cancelled | TicketPoll::Failed(_) => {
                Err(Self::refuse(tier, bytes, 0, role))
            }
        }
    }

    fn release(&self, tier: Tier, bytes: u64) {
        if tier != Tier::Host || bytes == 0 {
            return;
        }
        let returnable = {
            let mut outstanding = self.lock();
            outstanding.credit = outstanding.credit.saturating_add(bytes);
            // Give back the largest allocations the credit covers, so memory
            // returns to the machine in as few steps as possible.
            let mut returnable = Vec::new();
            loop {
                let best = outstanding
                    .allocations
                    .iter()
                    .enumerate()
                    .filter(|(_, allocation)| allocation.bytes <= outstanding.credit)
                    .max_by_key(|(_, allocation)| allocation.bytes)
                    .map(|(index, _)| index);
                let Some(index) = best else { break };
                let allocation = outstanding.allocations.swap_remove(index);
                outstanding.credit -= allocation.bytes;
                returnable.push(allocation);
            }
            returnable
        };
        for allocation in returnable {
            // Nothing to report to: this is a `Drop` path. Failing to give
            // memory back is a leak inside the host governor's own books, which
            // its snapshot will show.
            let _ = self.governor.release_host_pages(allocation);
        }
    }
}

/// A [`MemoryGovernor`] backed by the machine-wide [`HostGovernor`].
///
/// This is what replaces a private per-engine ledger: leases granted here are
/// charged to the same books the pressure protocol arbitrates over, so the
/// engine and the machine agree about host memory.
#[derive(Debug)]
pub struct HostLeaseGovernor {
    accounting: Arc<HostGovernorAccounting>,
    authority_id: onnx_runtime_memory_governor::MemoryAuthorityId,
}

impl HostLeaseGovernor {
    /// Grant leases against `governor` on behalf of `owner`.
    pub fn new(governor: Arc<HostGovernor>, owner: LocalDeviceId, priority: HostPriority) -> Self {
        let authority_id = governor.memory_authority_id();
        Self {
            accounting: Arc::new(HostGovernorAccounting::new(governor, owner, priority)),
            authority_id,
        }
    }

    /// The adapter's own view of what it has charged.
    pub fn accounting(&self) -> &Arc<HostGovernorAccounting> {
        &self.accounting
    }
}

impl MemoryGovernor for HostLeaseGovernor {
    fn authority_id(&self) -> onnx_runtime_memory_governor::MemoryAuthorityId {
        self.authority_id
    }

    fn reserve(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MemoryLease, MemoryError> {
        self.accounting.try_claim(tier, bytes, role)?;
        Ok(MemoryLease::new(
            tier,
            bytes,
            role,
            holder,
            Arc::clone(&self.accounting) as Arc<dyn LeaseAccounting>,
        ))
    }

    fn used(&self, tier: Tier) -> u64 {
        if tier != Tier::Host {
            return 0;
        }
        // What the host governor has actually granted. Unlike the ledger's
        // count this is a claim on a machine-wide resource, so it is read from
        // the governor rather than accumulated here.
        self.accounting
            .governor
            .snapshot()
            .map(|snapshot| snapshot.claimed_bytes)
            .unwrap_or(0)
    }

    fn available(&self, tier: Tier) -> u64 {
        if tier != Tier::Host {
            return 0;
        }
        let credit = self.accounting.pending_credit();
        let machine = self
            .accounting
            .governor
            .snapshot()
            .map(|snapshot| snapshot.free_bytes)
            .unwrap_or(0);
        credit.saturating_add(machine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pressure::HostGovernorConfig;

    const HOLDER: HolderId = HolderId::new(7);
    const DEVICE: LocalDeviceId = LocalDeviceId::new(0);

    fn governor(capacity: u64) -> HostLeaseGovernor {
        let host =
            Arc::new(HostGovernor::new(HostGovernorConfig::new(capacity)).expect("governor"));
        HostLeaseGovernor::new(host, DEVICE, 1)
    }

    #[test]
    fn adapters_over_one_host_governor_share_its_authority_identity() {
        let host = Arc::new(HostGovernor::new(HostGovernorConfig::new(1000)).expect("governor"));
        let first = HostLeaseGovernor::new(Arc::clone(&host), DEVICE, 1);
        let second = HostLeaseGovernor::new(host, LocalDeviceId::new(1), 2);

        assert_eq!(first.authority_id(), second.authority_id());
    }

    /// The point of the adapter: a lease is charged to the *machine's* ledger,
    /// not to a private one alongside it.
    ///
    /// Asserting through `HostGovernor::snapshot` rather than through the
    /// adapter's own counters is what makes this meaningful -- the adapter
    /// agreeing with itself would prove nothing.
    #[test]
    fn a_lease_is_charged_to_the_host_governors_own_ledger() {
        let leases = governor(1000);
        let host = Arc::clone(&leases.accounting().governor);
        let before = host.snapshot().expect("snapshot").claimed_bytes;

        let lease = leases
            .reserve(Tier::Host, 400, MemoryRole::KvCache, HOLDER)
            .expect("within capacity");

        assert_eq!(
            host.snapshot().expect("snapshot").claimed_bytes,
            before + 400,
            "the machine-wide ledger must see the lease"
        );

        drop(lease);
        assert_eq!(
            host.snapshot().expect("snapshot").claimed_bytes,
            before,
            "dropping the lease must return the bytes to the machine"
        );
    }

    /// A request the machine cannot satisfy is refused, and nothing is charged.
    #[test]
    fn an_oversized_request_is_refused_without_charging_anything() {
        let leases = governor(100);
        let host = Arc::clone(&leases.accounting().governor);

        let refused = leases.reserve(Tier::Host, 4096, MemoryRole::Weights, HOLDER);
        assert!(refused.is_err(), "4096 bytes cannot fit in 100");
        assert_eq!(
            host.snapshot().expect("snapshot").claimed_bytes,
            0,
            "a refused reservation must charge nothing (G4)"
        );
    }

    /// Only host memory is grantable. Handing back host memory for a device
    /// request would tell the caller their device budget was charged when it
    /// was not.
    #[test]
    fn device_and_disk_tiers_are_refused_rather_than_served_from_host_ram() {
        let leases = governor(1000);
        for tier in [Tier::Device, Tier::Disk] {
            assert!(
                leases
                    .reserve(tier, 8, MemoryRole::KvCache, HOLDER)
                    .is_err(),
                "{tier:?} must be refused by a host-memory governor"
            );
            assert_eq!(leases.available(tier), 0);
        }
    }

    /// Growing a lease charges the machine again; dropping returns everything.
    ///
    /// This is the case the credit scheme exists for: the lease releases one
    /// byte count covering two separate host allocations.
    #[test]
    fn a_grown_lease_returns_every_allocation_it_accumulated() {
        let leases = governor(1000);
        let host = Arc::clone(&leases.accounting().governor);

        let mut lease = leases
            .reserve(Tier::Host, 100, MemoryRole::Activation, HOLDER)
            .expect("granted");
        lease.grow(300).expect("room to grow");
        assert_eq!(host.snapshot().expect("snapshot").claimed_bytes, 400);
        assert_eq!(leases.accounting().charged_bytes(), 400);

        drop(lease);
        assert_eq!(
            host.snapshot().expect("snapshot").claimed_bytes,
            0,
            "both charges must come back, not just the first"
        );
        assert_eq!(
            leases.accounting().pending_credit(),
            0,
            "no credit stranded"
        );
    }

    /// A partial shrink cannot return anything yet, and says so rather than
    /// over-releasing.
    ///
    /// The host governor charges whole allocations, so there is nothing small
    /// enough to give back. Releasing the whole allocation would hand the
    /// machine memory the holder is still using.
    #[test]
    fn a_partial_shrink_holds_credit_instead_of_over_releasing() {
        let leases = governor(1000);
        let host = Arc::clone(&leases.accounting().governor);

        let mut lease = leases
            .reserve(Tier::Host, 100, MemoryRole::Activation, HOLDER)
            .expect("granted");
        let returned = lease.shrink(40);
        assert_eq!(returned, 40, "the lease shrank");
        assert_eq!(
            host.snapshot().expect("snapshot").claimed_bytes,
            100,
            "the holder still occupies the allocation, so the machine keeps the charge"
        );
        assert_eq!(leases.accounting().pending_credit(), 40);

        drop(lease);
        assert_eq!(
            host.snapshot().expect("snapshot").claimed_bytes,
            0,
            "the whole allocation comes back once the rest is released"
        );
        assert_eq!(leases.accounting().pending_credit(), 0);
    }

    /// Credit is re-lent before asking the machine again, or the same bytes
    /// would be charged twice.
    #[test]
    fn credit_is_spent_before_charging_the_machine_again() {
        let leases = governor(100);
        let host = Arc::clone(&leases.accounting().governor);

        let mut lease = leases
            .reserve(Tier::Host, 100, MemoryRole::Activation, HOLDER)
            .expect("granted");
        lease.shrink(60);
        assert_eq!(leases.accounting().pending_credit(), 60);

        // The machine is fully charged, so this can only succeed out of credit.
        lease.grow(60).expect("credit must be re-lent");
        assert_eq!(
            host.snapshot().expect("snapshot").claimed_bytes,
            100,
            "re-lending credit must not charge the machine a second time"
        );
        assert_eq!(leases.accounting().pending_credit(), 0);
    }
}

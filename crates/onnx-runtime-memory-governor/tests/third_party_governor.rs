//! Can somebody outside this crate actually substitute their own memory
//! manager?
//!
//! That is the claim the lease contract makes, and it cannot be proved from
//! inside the crate: an in-crate test reaches private fields and constructs
//! types the public API does not expose. These tests live in `tests/` on
//! purpose — they see exactly what a third party sees, and they stop compiling
//! the moment the contract stops being implementable from outside.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_runtime_memory_governor::{
    DeviceKey, HolderId, LeaseAccounting, MemoryAuthorityId, MemoryError, MemoryGovernor,
    MemoryLease, MemoryRole, Tier,
};

/// A memory manager written entirely against the public API, sharing no code
/// with `LedgerGovernor`.
///
/// It deliberately behaves differently from the reference implementation: one
/// pool spanning host and disk, and no device memory at all. If a test passes
/// because it accidentally exercised the built-in behaviour, that difference
/// is what exposes it.
#[derive(Debug)]
struct MyOwnManager {
    free: AtomicU64,
    /// What this manager started with, so the trait's `used` can be answered
    /// from what it already tracks rather than needing a second counter.
    capacity: u64,
    releases: AtomicU64,
}

impl MyOwnManager {
    fn new(capacity: u64) -> Arc<Self> {
        Arc::new(Self {
            free: AtomicU64::new(capacity),
            capacity,
            releases: AtomicU64::new(0),
        })
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

impl LeaseAccounting for MyOwnManager {
    fn try_claim(&self, tier: Tier, bytes: u64, role: MemoryRole) -> Result<(), MemoryError> {
        if tier == Tier::Device {
            return Err(Self::refuse(tier, bytes, 0, role));
        }
        let mut free = self.free.load(Ordering::Acquire);
        loop {
            if free < bytes {
                return Err(Self::refuse(tier, bytes, free, role));
            }
            match self.free.compare_exchange_weak(
                free,
                free - bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => free = observed,
            }
        }
    }

    fn release(&self, _tier: Tier, bytes: u64) {
        self.free.fetch_add(bytes, Ordering::AcqRel);
        self.releases.fetch_add(1, Ordering::AcqRel);
    }
}

struct MyOwnGovernor {
    accounting: Arc<MyOwnManager>,
    authority_id: MemoryAuthorityId,
}

impl MemoryGovernor for MyOwnGovernor {
    fn authority_id(&self) -> MemoryAuthorityId {
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

    fn available(&self, tier: Tier) -> u64 {
        if tier == Tier::Device {
            0
        } else {
            self.accounting.free.load(Ordering::Acquire)
        }
    }

    /// A third-party governor answers this from whatever it already tracks.
    ///
    /// Here that is the granted total, which is what a caller asking "what does
    /// this tier hold" wants -- and the point of the method being on the trait
    /// is that no caller can work it out for itself: leases are owned by the
    /// components that must outlive them.
    fn used(&self, tier: Tier) -> u64 {
        if tier == Tier::Device {
            0
        } else {
            self.accounting.capacity - self.accounting.free.load(Ordering::Acquire)
        }
    }
}

const HOLDER: HolderId = HolderId::new(1);

fn governor(capacity: u64) -> (MyOwnGovernor, Arc<MyOwnManager>) {
    let accounting = MyOwnManager::new(capacity);
    (
        MyOwnGovernor {
            accounting: Arc::clone(&accounting),
            authority_id: MemoryAuthorityId::new(DeviceKey::HOST),
        },
        accounting,
    )
}

/// The contract is implementable from outside, and its leases behave.
///
/// If `MemoryLease` ever goes back to being constructible only from this
/// crate's own ledger, this file stops compiling — a louder failure than an
/// assertion, and the reason it lives here rather than in `src`.
#[test]
fn a_third_party_governor_can_grant_and_reclaim_leases() {
    let (governor, accounting) = governor(1000);

    let lease = governor
        .reserve(Tier::Host, 400, MemoryRole::KvCache, HOLDER)
        .expect("a request within capacity must be granted");
    assert_eq!(governor.available(Tier::Host), 600);
    assert_eq!(lease.bytes(), 400);

    drop(lease);
    assert_eq!(
        governor.available(Tier::Host),
        1000,
        "dropping a third-party lease must return the bytes to its own books"
    );
    assert_eq!(
        accounting.releases.load(Ordering::Acquire),
        1,
        "released exactly once (G2)"
    );
}

/// A refused reservation must leave the books untouched (G4).
#[test]
fn a_refused_reservation_disturbs_nothing() {
    let (governor, _) = governor(1000);

    let held = governor
        .reserve(Tier::Host, 900, MemoryRole::Weights, HOLDER)
        .expect("granted");
    let refused = governor.reserve(Tier::Host, 200, MemoryRole::KvCache, HOLDER);
    assert!(refused.is_err(), "there are only 100 bytes left");
    assert_eq!(
        governor.available(Tier::Host),
        100,
        "a failed reservation must not consume or release anything"
    );
    assert_eq!(held.bytes(), 900, "the existing lease is untouched");
}

/// `grow` charges the third party's books, not this crate's.
#[test]
fn growing_a_lease_charges_the_implementors_own_accounting() {
    let (governor, _) = governor(1000);

    let mut lease = governor
        .reserve(Tier::Host, 100, MemoryRole::Activation, HOLDER)
        .expect("granted");
    lease.grow(300).expect("room to grow");
    assert_eq!(lease.bytes(), 400);
    assert_eq!(governor.available(Tier::Host), 600);

    lease
        .grow(10_000)
        .expect_err("growing past capacity must fail");
    assert_eq!(
        lease.bytes(),
        400,
        "a failed grow must leave the lease exactly as it was"
    );
    assert_eq!(governor.available(Tier::Host), 600);
}

/// The implementor's own policy decides what is refused; nothing here overrides
/// it.
#[test]
fn the_implementors_policy_decides_what_is_refused() {
    let (governor, _) = governor(1000);

    assert!(
        governor
            .reserve(Tier::Device, 8, MemoryRole::KvCache, HOLDER)
            .is_err(),
        "this manager has no device memory and says so; nothing may grant it anyway"
    );
    assert_eq!(governor.available(Tier::Device), 0);
}

/// Governed components take `Arc<dyn MemoryGovernor>`, so a third-party
/// implementation has to be usable wherever the built-in one is. An
/// implementation that only worked as a concrete type would be substitutable in
/// name only.
#[test]
fn a_third_party_governor_is_usable_as_a_trait_object() {
    let (concrete, _) = governor(64);
    let governor: Arc<dyn MemoryGovernor + Send + Sync> = Arc::new(concrete);

    let lease = governor
        .reserve(Tier::Host, 32, MemoryRole::Activation, HOLDER)
        .expect("granted through the trait object");
    assert_eq!(lease.tier(), Tier::Host);
    assert_eq!(lease.holder(), HOLDER);
    assert_eq!(governor.available(Tier::Host), 32);
}

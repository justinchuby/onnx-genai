//! # `onnx-runtime-memory-governor`
//!
//! The contract for **tiered memory governance**: one authority owns the VRAM,
//! host RAM and disk budgets, and every component that holds bytes leases them
//! from it.
//!
//! ## Why this crate exists, and why it is this small
//!
//! Before this contract there were three unrelated memory systems: a governor
//! that tracked budgets but allocated nothing, a paged KV cache that owned real
//! host buffers and never asked anyone, and per-backend decode state that owned
//! the live KV. Nothing could schedule across them, because nothing could see
//! all of them.
//!
//! The contract lives in its own dependency-light crate for two reasons:
//!
//! * **Layering.** The KV store (`onnx-genai-kv`) must lease, and so must the
//!   weight-offload cache in `onnx-runtime-ep-cpu`. The scheduler implements the
//!   governor and already depends on the KV store, so the contract cannot live
//!   there without a cycle. It has to sit below everything that leases.
//! * **Substitutability.** A third party supplying their own allocator should
//!   implement against a crate they can read in one sitting, not against the
//!   whole engine.
//!
//! Not to be confused with `onnx-runtime-memory`, which plans *activation*
//! buffer reuse within a single graph. That crate is a future lease holder
//! ([`MemoryRole::Activation`]); this one decides whether it may hold anything.
//!
//! ## Where this sits
//!
//! `docs/MEMORY_ARCHITECTURE.md` is the canonical design. This crate implements
//! a slice of its Layer 3: the vocabulary a lease is expressed in, and a
//! self-contained ledger for callers that have no governor.
//!
//! It exists as a separate crate for a layering reason that survives that
//! design: `onnx-genai-kv` must lease, and `onnx-genai-scheduler` — where the
//! canonical `HostGovernor` lives — already depends on `onnx-genai-kv`. The KV
//! store therefore cannot lease from `HostGovernor` without a dependency cycle,
//! so the vocabulary has to sit below both. It is also what a third party would
//! implement against, which is why it depends on nothing but `thiserror`.
//!
//! Two divergences from the canonical design, stated rather than hidden:
//!
//! * **[`LeaseLedger`] holds all three tiers in one object.** The canonical
//!   design separates a per-device governor from a per-machine one, because
//!   device memory is exclusive while host RAM is shared by every device on the
//!   machine. A single ledger per engine reproduces the problem that split
//!   exists to prevent, one level down. Use it for a single-device engine or for
//!   tests; multi-device belongs to `HostGovernor`.
//! * **[`PressureResponder`] is weaker than the protocol already implemented.**
//!   `onnx-genai-scheduler`'s `pressure` module implements a ticketed,
//!   non-blocking protocol with cancellation, configuration generations and
//!   priority arbitration, modelled in `specs/tla/PressureProtocol.tla`. This
//!   synchronous trait is a placeholder for callers not yet routed through it,
//!   not an alternative to it.
//!
//! ## The invariants this crate enforces
//!
//! * **G1** New reservations never make the sum of live leases exceed a tier's
//!   limit. Already-committed memory is recorded even when it exceeds the
//!   limit, because refusing to count it would make later admission optimistic.
//! * **G2** A lease is released exactly once, on drop, and releasing cannot
//!   fail. Callers cannot leak a reservation by taking an early return.
//! * **G3** The governor never frees anyone's memory. It asks holders through
//!   [`PressureResponder`] and they decide what to give back; a holder that
//!   returns zero is respected and the *new* request fails instead.
//! * **G4** A failed reservation leaves every existing lease untouched.
//!
//! G3 is the general form of a bug this project already shipped once, where an
//! eviction pass deleted the very session it was writing to in order to satisfy
//! a budget. Live state is never taken to make room.

pub mod allocator;

pub use allocator::{AllocationCommitRange, DeviceAllocator, DeviceKey, HostAllocator};

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

/// Stable identity of one physical-memory accounting authority.
///
/// Equality means two components charge the same books for physical memory in
/// the same compatibility domain. It does not mean merely "the same device":
/// two independent ledgers for device 0 deliberately receive different IDs.
///
/// Shared-resource construction must copy an existing ID, normally by cloning
/// the governor that owns it. Creating another ID for the same device declares
/// another accounting authority and therefore makes shared physical backing
/// incompatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryAuthorityId {
    device: DeviceKey,
    serial: u64,
}

impl MemoryAuthorityId {
    /// Create a distinct authority for `device`.
    ///
    /// Use this once when constructing the ledger that owns the device's
    /// physical bytes. Holders that share those books reuse the returned ID;
    /// they must not independently call this constructor.
    pub fn new(device: DeviceKey) -> Self {
        static NEXT_AUTHORITY: AtomicU64 = AtomicU64::new(1);
        Self {
            device,
            serial: NEXT_AUTHORITY.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// The physical-memory compatibility domain this authority governs.
    pub const fn device(self) -> DeviceKey {
        self.device
    }
}

impl std::fmt::Display for MemoryAuthorityId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{:?} device {} authority {}",
            self.device.tier, self.device.index, self.serial
        )
    }
}

/// Where the bytes physically live.
///
/// Ordered from fastest to slowest, which is also the demotion order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// Accelerator memory (VRAM).
    Device,
    /// Host RAM.
    Host,
    /// Spill file on disk.
    Disk,
}

impl Tier {
    /// Every tier, fastest first.
    pub const ALL: [Tier; 3] = [Tier::Device, Tier::Host, Tier::Disk];

    /// Stable index for array-backed per-tier state.
    const fn index(self) -> usize {
        match self {
            Tier::Device => 0,
            Tier::Host => 1,
            Tier::Disk => 2,
        }
    }

    /// Human-facing name used in error messages.
    pub const fn name(self) -> &'static str {
        match self {
            Tier::Device => "device",
            Tier::Host => "host",
            Tier::Disk => "disk",
        }
    }
}

/// What a reservation is *for*.
///
/// The governor reads this; it never infers purpose from allocation size or
/// timing, because that is guessing. Roles are what make an eviction order
/// expressible rather than hardcoded.
///
/// Deliberately carries no sequence or session identity. Under G3 the governor
/// asks a *holder* to release bytes and the holder chooses which of its own
/// sequences to give up, so the governor never has to reason about sequences —
/// and this crate never has to depend on the KV layer to name one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryRole {
    /// Long-lived per-sequence KV. Migratable, and the usual eviction target
    /// after weights.
    KvCache,
    /// Scratch space for computation.
    Workspace {
        /// Released wholesale at the end of the step that took it. Step-scoped
        /// workspace is never migrated, because nothing would be gained before
        /// it is freed anyway.
        step_scoped: bool,
    },
    /// Model parameters. Immutable and shareable, so the cheapest thing to
    /// demote first: it can always be re-read from the package on disk.
    Weights,
    /// Intermediate activations for one graph execution. The hottest and
    /// shortest-lived class.
    Activation,
}

/// Identifies a component that holds leases, so the governor can ask it to
/// release under pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HolderId(u64);

impl HolderId {
    /// Wrap a raw identifier. Uniqueness is the caller's responsibility.
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw identifier.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for HolderId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "holder {}", self.0)
    }
}

/// Why a reservation could not be granted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MemoryError {
    /// The tier does not have room, and no holder released enough.
    #[error(
        "cannot reserve {requested} bytes of {tier} memory for {role:?}: {used} of {limit} bytes \
         are already leased, leaving {available}; free memory by closing sessions, lower the \
         demand, or raise the {tier} limit"
    )]
    TierExhausted {
        /// Which tier ran out.
        tier: &'static str,
        /// What the caller asked for.
        requested: u64,
        /// Bytes leased before this request.
        used: u64,
        /// The tier ceiling.
        limit: u64,
        /// `limit - used`.
        available: u64,
        /// The role that was refused.
        role: MemoryRole,
    },
    /// The request itself is not representable.
    #[error("cannot reserve {requested} bytes of {tier} memory: {reason}")]
    InvalidRequest {
        /// Which tier was addressed.
        tier: &'static str,
        /// What the caller asked for.
        requested: u64,
        /// What is wrong with it.
        reason: &'static str,
    },
    /// The request was well formed and within budget, but the allocator behind
    /// the tier refused it for a reason of its own.
    ///
    /// Distinct from [`MemoryError::TierExhausted`], which means *we* declined,
    /// and from [`MemoryError::InvalidRequest`], which means the caller asked
    /// for something impossible. This one carries the backing allocator's own
    /// account of the failure, which is usually the only thing that identifies
    /// it: a driver that is out of memory and a driver that has no context both
    /// fail an allocation, and calling them both "out of memory" sends the next
    /// person to read the log in the wrong direction.
    #[error("cannot allocate {requested} bytes of {tier} memory: {reason}")]
    AllocationFailed {
        /// Which tier was addressed.
        tier: &'static str,
        /// What the caller asked for.
        requested: u64,
        /// What the backing allocator said.
        reason: String,
    },
}

/// Per-tier accounting shared by a governor and every lease it has granted.
///
/// Concrete rather than a trait object so that dropping a [`MemoryLease`] stays
/// infallible and allocation-free, which is what G2 requires.
#[derive(Debug)]
pub struct LeaseLedger {
    authority_id: MemoryAuthorityId,
    limits: [AtomicU64; 3],
    used: [AtomicU64; 3],
    mapped_allowance_reserved: [AtomicU64; 3],
    mapped_growth_reserved: [AtomicU64; 3],
    claim_gates: [Mutex<()>; 3],
}

impl LeaseLedger {
    /// A single-device ledger with the given per-tier ceilings.
    ///
    /// The device tier defaults to device 0. Use [`Self::new_for_device`] when
    /// constructing an authority for another device.
    pub fn new(device_bytes: u64, host_bytes: u64, disk_bytes: u64) -> Arc<Self> {
        Self::new_for_device(DeviceKey::device(0), device_bytes, host_bytes, disk_bytes)
    }

    /// A ledger whose device tier belongs to `device`.
    pub fn new_for_device(
        device: DeviceKey,
        device_bytes: u64,
        host_bytes: u64,
        disk_bytes: u64,
    ) -> Arc<Self> {
        Arc::new(Self {
            authority_id: MemoryAuthorityId::new(device),
            limits: [
                AtomicU64::new(device_bytes),
                AtomicU64::new(host_bytes),
                AtomicU64::new(disk_bytes),
            ],
            used: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            mapped_allowance_reserved: std::array::from_fn(|_| AtomicU64::new(0)),
            mapped_growth_reserved: std::array::from_fn(|_| AtomicU64::new(0)),
            claim_gates: std::array::from_fn(|_| Mutex::new(())),
        })
    }

    /// Bytes currently leased on `tier`.
    pub fn used(&self, tier: Tier) -> u64 {
        self.used[tier.index()].load(Ordering::Acquire)
    }

    /// The ceiling for `tier`.
    pub fn limit(&self, tier: Tier) -> u64 {
        self.limits[tier.index()].load(Ordering::Acquire)
    }

    /// Bytes still grantable on `tier`.
    pub fn available(&self, tier: Tier) -> u64 {
        self.limit(tier).saturating_sub(
            self.used(tier)
                .saturating_add(self.mapped_growth_reserved[tier.index()].load(Ordering::Acquire)),
        )
    }

    /// Bytes by which current leases exceed `tier`'s ceiling.
    pub fn oversubscribed_bytes(&self, tier: Tier) -> u64 {
        self.used(tier).saturating_sub(self.limit(tier))
    }

    /// Replace a tier ceiling.
    ///
    /// Lowering below current usage is allowed and does **not** revoke anything:
    /// under G3 nothing is taken from a live holder. The tier simply grants
    /// nothing further until usage falls back under the new ceiling.
    pub fn set_limit(&self, tier: Tier, bytes: u64) {
        let _gate = self.claim_gate(tier);
        self.limits[tier.index()].store(bytes, Ordering::Release);
    }

    /// Pause new claims on `tier` while an authority validates and commits a
    /// limit change. Releases remain lock-free, so reclaim can run while this
    /// guard is held.
    pub fn pause_claims(&self, tier: Tier) -> LeaseLimitGuard<'_> {
        LeaseLimitGuard {
            ledger: self,
            tier,
            _gate: self.claim_gate(tier),
        }
    }

    fn claim_gate(&self, tier: Tier) -> MutexGuard<'_, ()> {
        self.claim_gates[tier.index()]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Claim `bytes` on `tier`, or fail without changing anything (G1, G4).
    fn try_claim(&self, tier: Tier, bytes: u64, role: MemoryRole) -> Result<(), MemoryError> {
        let _gate = self.claim_gate(tier);
        let index = tier.index();
        let mut used = self.used[index].load(Ordering::Acquire);
        loop {
            let limit = self.limits[index].load(Ordering::Acquire);
            // Checked, not saturating: a wrapped total would silently under-report
            // usage and let the tier grant memory it does not have.
            let Some(next) = used.checked_add(bytes) else {
                return Err(MemoryError::InvalidRequest {
                    tier: tier.name(),
                    requested: bytes,
                    reason: "the request overflows the tier's byte counter",
                });
            };
            let reserved = self.mapped_growth_reserved[index].load(Ordering::Acquire);
            if next.saturating_add(reserved) > limit {
                return Err(MemoryError::TierExhausted {
                    tier: tier.name(),
                    requested: bytes,
                    used,
                    limit,
                    available: limit.saturating_sub(used.saturating_add(reserved)),
                    role,
                });
            }
            match self.used[index].compare_exchange_weak(
                used,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => used = observed,
            }
        }
    }

    /// Record bytes that are already committed, even if the tier is over limit.
    fn record_claim(&self, tier: Tier, bytes: u64) -> Result<(), MemoryError> {
        let _gate = self.claim_gate(tier);
        let index = tier.index();
        let mut used = self.used[index].load(Ordering::Acquire);
        loop {
            let Some(next) = used.checked_add(bytes) else {
                return Err(MemoryError::InvalidRequest {
                    tier: tier.name(),
                    requested: bytes,
                    reason: "the request overflows the tier's byte counter",
                });
            };
            match self.used[index].compare_exchange_weak(
                used,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => used = observed,
            }
        }
    }

    /// Give `bytes` back to `tier`. Infallible, as G2 requires.
    fn release(&self, tier: Tier, bytes: u64) {
        // Saturating because releasing a lease must never panic or wrap the
        // counter, whatever bookkeeping went wrong upstream.
        let index = tier.index();
        let mut used = self.used[index].load(Ordering::Acquire);
        loop {
            let next = used.saturating_sub(bytes);
            match self.used[index].compare_exchange_weak(
                used,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => used = observed,
            }
        }
    }

    fn try_reserve_mapped_allowance(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
    ) -> Result<(), MemoryError> {
        let _gate = self.claim_gate(tier);
        let index = tier.index();
        let mut reserved = self.mapped_allowance_reserved[index].load(Ordering::Acquire);
        loop {
            let limit = self.limits[index].load(Ordering::Acquire);
            let Some(next) = reserved.checked_add(bytes) else {
                return Err(MemoryError::InvalidRequest {
                    tier: tier.name(),
                    requested: bytes,
                    reason: "the mapped-allowance reservation overflows its byte counter",
                });
            };
            if next > limit {
                return Err(MemoryError::TierExhausted {
                    tier: tier.name(),
                    requested: bytes,
                    used: reserved,
                    limit,
                    available: limit.saturating_sub(reserved),
                    role,
                });
            }
            match self.mapped_allowance_reserved[index].compare_exchange_weak(
                reserved,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => reserved = observed,
            }
        }
    }

    fn release_mapped_allowance(&self, tier: Tier, bytes: u64) {
        let _ = self.mapped_allowance_reserved[tier.index()].fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |reserved| Some(reserved.saturating_sub(bytes)),
        );
    }

    fn reserve_mapped_growth(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
    ) -> Result<u64, MemoryError> {
        let _gate = self.claim_gate(tier);
        let index = tier.index();
        let used = self.used[index].load(Ordering::Acquire);
        let limit = self.limits[index].load(Ordering::Acquire);
        let current = self.mapped_growth_reserved[index].load(Ordering::Acquire);
        let available = limit.saturating_sub(used.saturating_add(current));
        let reserved = bytes.min(available);
        self.mapped_growth_reserved[index]
            .store(current.saturating_add(reserved), Ordering::Release);
        if bytes > 0 && reserved == 0 && used < limit {
            return Err(MemoryError::TierExhausted {
                tier: tier.name(),
                requested: bytes,
                used,
                limit,
                available,
                role,
            });
        }
        Ok(reserved)
    }

    fn release_mapped_growth(&self, tier: Tier, bytes: u64) {
        let _gate = self.claim_gate(tier);
        let _ = self.mapped_growth_reserved[tier.index()].fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |reserved| Some(reserved.saturating_sub(bytes)),
        );
    }
}

/// Authority-owned capacity reservation for mapped bytes attributed to one holder.
#[derive(Clone, Debug)]
pub struct MappedAllowance {
    inner: Arc<MappedAllowanceInner>,
}

#[derive(Debug)]
struct MappedAllowanceInner {
    authority: MemoryAuthorityId,
    tier: Tier,
    limit: AtomicU64,
    mapped: AtomicU64,
    growth_reserved: AtomicU64,
    role: MemoryRole,
    holder: HolderId,
    accounting: Arc<dyn MappedAllowanceAccounting>,
}

/// Releases mapped-capacity reservations created by a governor.
pub trait MappedAllowanceAccounting: Send + Sync + std::fmt::Debug {
    fn release(&self, tier: Tier, bytes: u64);
}

impl MappedAllowance {
    pub fn new(
        authority: MemoryAuthorityId,
        tier: Tier,
        limit: u64,
        role: MemoryRole,
        holder: HolderId,
        accounting: Arc<dyn MappedAllowanceAccounting>,
    ) -> Self {
        Self {
            inner: Arc::new(MappedAllowanceInner {
                authority,
                tier,
                limit: AtomicU64::new(limit),
                mapped: AtomicU64::new(0),
                growth_reserved: AtomicU64::new(0),
                role,
                holder,
                accounting,
            }),
        }
    }

    pub fn authority(&self) -> MemoryAuthorityId {
        self.inner.authority
    }

    pub fn limit(&self) -> u64 {
        self.inner.limit.load(Ordering::Acquire)
    }

    pub fn mapped_bytes(&self) -> u64 {
        self.inner.mapped.load(Ordering::Acquire)
    }

    pub fn available(&self) -> u64 {
        self.limit().saturating_sub(
            self.mapped_bytes()
                .saturating_add(self.inner.growth_reserved.load(Ordering::Acquire)),
        )
    }

    pub fn try_map(&self, bytes: u64) -> Result<(), MemoryError> {
        let mut mapped = self.inner.mapped.load(Ordering::Acquire);
        loop {
            let Some(next) = mapped.checked_add(bytes) else {
                return Err(MemoryError::InvalidRequest {
                    tier: self.inner.tier.name(),
                    requested: bytes,
                    reason: "mapped-byte attribution overflows its byte counter",
                });
            };
            let limit = self.limit();
            let reserved = self.inner.growth_reserved.load(Ordering::Acquire);
            if next.saturating_add(reserved) > limit {
                return Err(MemoryError::TierExhausted {
                    tier: self.inner.tier.name(),
                    requested: bytes,
                    used: mapped,
                    limit,
                    available: limit.saturating_sub(mapped.saturating_add(reserved)),
                    role: self.inner.role,
                });
            }
            match self.inner.mapped.compare_exchange_weak(
                mapped,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => mapped = observed,
            }
        }
    }

    pub fn unmap(&self, bytes: u64) -> u64 {
        let mut mapped = self.inner.mapped.load(Ordering::Acquire);
        loop {
            let returned = bytes.min(mapped);
            match self.inner.mapped.compare_exchange_weak(
                mapped,
                mapped - returned,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return returned,
                Err(observed) => mapped = observed,
            }
        }
    }

    pub fn holder(&self) -> HolderId {
        self.inner.holder
    }

    pub fn role(&self) -> MemoryRole {
        self.inner.role
    }

    fn reserve_growth(&self, bytes: u64) -> Result<(), MemoryError> {
        let mut reserved = self.inner.growth_reserved.load(Ordering::Acquire);
        loop {
            let mapped = self.mapped_bytes();
            let limit = self.limit();
            let Some(next) = reserved.checked_add(bytes) else {
                return Err(MemoryError::InvalidRequest {
                    tier: self.inner.tier.name(),
                    requested: bytes,
                    reason: "mapped growth reservation overflows its byte counter",
                });
            };
            if mapped.saturating_add(next) > limit {
                return Err(MemoryError::TierExhausted {
                    tier: self.inner.tier.name(),
                    requested: bytes,
                    used: mapped.saturating_add(reserved),
                    limit,
                    available: limit.saturating_sub(mapped.saturating_add(reserved)),
                    role: self.inner.role,
                });
            }
            match self.inner.growth_reserved.compare_exchange_weak(
                reserved,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => reserved = observed,
            }
        }
    }

    fn release_growth(&self, bytes: u64) {
        let _ = self.inner.growth_reserved.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |reserved| Some(reserved.saturating_sub(bytes)),
        );
    }

    fn add_limit_transferred(&self, bytes: u64) -> Result<(), MemoryError> {
        self.inner
            .limit
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |limit| {
                limit.checked_add(bytes)
            })
            .map(|_| ())
            .map_err(|_| MemoryError::InvalidRequest {
                tier: self.inner.tier.name(),
                requested: bytes,
                reason: "mapped allowance transfer overflows the requester limit",
            })
    }

    fn subtract_limit_transferred(&self, bytes: u64) -> Result<(), MemoryError> {
        self.inner
            .limit
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |limit| {
                (bytes <= limit).then_some(limit - bytes)
            })
            .map(|_| ())
            .map_err(|_| MemoryError::InvalidRequest {
                tier: self.inner.tier.name(),
                requested: bytes,
                reason: "mapped allowance transfer exceeds the victim limit",
            })
    }

    fn map_reserved(&self, bytes: u64) -> Result<(), MemoryError> {
        let reserved = self.inner.growth_reserved.load(Ordering::Acquire);
        if bytes > reserved {
            return Err(MemoryError::InvalidRequest {
                tier: self.inner.tier.name(),
                requested: bytes,
                reason: "mapped growth commit exceeds its live reservation",
            });
        }
        self.release_growth(bytes);
        if let Err(error) = self.try_map(bytes) {
            let _ = self.reserve_growth(bytes);
            return Err(error);
        }
        Ok(())
    }
}

impl Drop for MappedAllowanceInner {
    fn drop(&mut self) {
        self.accounting
            .release(self.tier, self.limit.load(Ordering::Acquire));
    }
}

/// A mapped holder that can evict reloadable mappings on authority request.
///
/// Registration is explicit and weak: the authority never keeps a model alive,
/// and dropping either the registration or the holder removes it from future
/// victim selection.
pub trait ReclaimableMappedHolder: Send + Sync {
    fn allowance(&self) -> MappedAllowance;
    fn reclaim_priority(&self) -> u32;
    fn mapped_bytes(&self) -> u64;
    fn reclaim_mapped(&self, target_bytes: u64) -> Result<MappedReclaimReport, MemoryError>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MappedReclaimReport {
    pub target_bytes: u64,
    pub reclaimed_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MappedGrowthMetrics {
    pub attempts: u64,
    pub bytes_transferred: u64,
    pub failures: u64,
    pub rollbacks: u64,
    pub live_holders: u64,
    pub mapped_bytes: u64,
    pub total_allowance_bytes: u64,
    pub weight_mapped: u64,
    pub kv_mapped: u64,
    pub workspace_mapped: u64,
    pub total_owned: u64,
}

#[derive(Default)]
struct MappedGrowthCounters {
    attempts: AtomicU64,
    bytes_transferred: AtomicU64,
    failures: AtomicU64,
    rollbacks: AtomicU64,
}

struct RegisteredMappedHolder {
    holder: Weak<dyn ReclaimableMappedHolder>,
}

struct MappedGrowthAuthorityInner {
    ledger: Arc<LeaseLedger>,
    active: AtomicBool,
    next_registration: AtomicU64,
    holders: Mutex<BTreeMap<u64, RegisteredMappedHolder>>,
    allowances: Mutex<Vec<Weak<MappedAllowanceInner>>>,
    counters: MappedGrowthCounters,
}

/// Authority-owned mapped-growth transaction coordinator.
#[derive(Clone)]
pub struct MappedGrowthAuthority {
    inner: Arc<MappedGrowthAuthorityInner>,
}

impl std::fmt::Debug for MappedGrowthAuthority {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MappedGrowthAuthority")
            .field("authority", &self.inner.ledger.authority_id)
            .field("metrics", &self.metrics())
            .finish()
    }
}

impl MappedGrowthAuthority {
    fn new(ledger: Arc<LeaseLedger>) -> Self {
        Self {
            inner: Arc::new(MappedGrowthAuthorityInner {
                ledger,
                active: AtomicBool::new(false),
                next_registration: AtomicU64::new(1),
                holders: Mutex::new(BTreeMap::new()),
                allowances: Mutex::new(Vec::new()),
                counters: MappedGrowthCounters::default(),
            }),
        }
    }

    pub fn register(
        &self,
        holder: &Arc<dyn ReclaimableMappedHolder>,
    ) -> Result<MappedHolderRegistration, MemoryError> {
        let allowance = holder.allowance();
        if allowance.authority() != self.inner.ledger.authority_id {
            return Err(MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: allowance.limit(),
                reason: "reclaimable mapped holder belongs to a different memory authority",
            });
        }
        let id = self.inner.next_registration.fetch_add(1, Ordering::Relaxed);
        self.inner
            .holders
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                id,
                RegisteredMappedHolder {
                    holder: Arc::downgrade(holder),
                },
            );
        Ok(MappedHolderRegistration {
            authority: Arc::downgrade(&self.inner),
            id,
        })
    }

    fn track_allowance(&self, allowance: &MappedAllowance) {
        self.inner
            .allowances
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(Arc::downgrade(&allowance.inner));
    }

    pub fn prepare_mapped_growth(
        &self,
        requester: &MappedAllowance,
        bytes: u64,
    ) -> Result<MappedGrowthGrant, MemoryError> {
        self.inner.counters.attempts.fetch_add(1, Ordering::Relaxed);
        if requester.authority() != self.inner.ledger.authority_id {
            self.inner.counters.failures.fetch_add(1, Ordering::Relaxed);
            return Err(MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: bytes,
                reason: "mapped growth requester belongs to a different memory authority",
            });
        }
        if self
            .inner
            .active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            self.inner.counters.failures.fetch_add(1, Ordering::Relaxed);
            return Err(MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: bytes,
                reason: "another mapped-growth transaction or authority reconfiguration is active",
            });
        }

        let physical_reserved =
            match self
                .inner
                .ledger
                .reserve_mapped_growth(Tier::Device, bytes, requester.role())
            {
                Ok(reserved) => reserved,
                Err(error) => {
                    self.inner.active.store(false, Ordering::Release);
                    self.inner.counters.failures.fetch_add(1, Ordering::Relaxed);
                    return Err(error);
                }
            };

        let mut grant = MappedGrowthGrant {
            authority: Arc::clone(&self.inner),
            requester: requester.clone(),
            requested_bytes: bytes,
            physical_reserved,
            transferred_bytes: 0,
            victims: Vec::new(),
            active: true,
        };

        let own_unused = bytes.min(requester.available());
        if let Err(error) = requester.reserve_growth(own_unused) {
            grant.rollback();
            return Err(error);
        }
        let mut remaining = bytes.saturating_sub(own_unused);
        if remaining == 0 {
            return Ok(grant);
        }

        let mut candidates = {
            let mut holders = self
                .inner
                .holders
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            holders.retain(|_, registered| registered.holder.strong_count() > 0);
            holders
                .values()
                .filter_map(|registered| registered.holder.upgrade())
                .filter(|holder| holder.allowance().holder() != requester.holder())
                .collect::<Vec<_>>()
        };
        candidates.sort_by_key(|holder| holder.reclaim_priority());

        for holder in candidates {
            if remaining == 0 {
                break;
            }
            let victim = holder.allowance();
            if victim.authority() != requester.authority() {
                continue;
            }
            let transfer = remaining.min(victim.limit());
            if transfer == 0 {
                continue;
            }
            if let Err(error) = victim.subtract_limit_transferred(transfer) {
                grant.rollback();
                return Err(error);
            }
            if let Err(error) = requester.add_limit_transferred(transfer) {
                let _ = victim.add_limit_transferred(transfer);
                grant.rollback();
                return Err(error);
            }
            if let Err(error) = requester.reserve_growth(transfer) {
                let _ = requester.subtract_limit_transferred(transfer);
                let _ = victim.add_limit_transferred(transfer);
                grant.rollback();
                return Err(error);
            }

            let new_limit = victim.limit();
            let reclaim_target = holder.mapped_bytes().saturating_sub(new_limit);
            let report = if reclaim_target == 0 {
                MappedReclaimReport::default()
            } else {
                match holder.reclaim_mapped(reclaim_target) {
                    Ok(report) => report,
                    Err(error) => {
                        requester.release_growth(transfer);
                        let _ = requester.subtract_limit_transferred(transfer);
                        let _ = victim.add_limit_transferred(transfer);
                        grant.rollback();
                        return Err(error);
                    }
                }
            };
            if report.reclaimed_bytes < reclaim_target {
                requester.release_growth(transfer);
                let _ = requester.subtract_limit_transferred(transfer);
                let _ = victim.add_limit_transferred(transfer);
                grant.rollback();
                return Err(MemoryError::InvalidRequest {
                    tier: Tier::Device.name(),
                    requested: reclaim_target,
                    reason: "mapped holder could not reach its tentative reclaim target",
                });
            }
            grant.transferred_bytes = grant.transferred_bytes.saturating_add(transfer);
            grant.victims.push((victim, transfer));
            remaining -= transfer;
        }

        if remaining > 0 {
            grant.rollback();
            return Err(MemoryError::TierExhausted {
                tier: Tier::Device.name(),
                requested: bytes,
                used: bytes - remaining,
                limit: bytes,
                available: bytes - remaining,
                role: requester.role(),
            });
        }
        Ok(grant)
    }

    pub fn metrics(&self) -> MappedGrowthMetrics {
        let holders = self
            .inner
            .holders
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let live = holders
            .values()
            .filter_map(|registered| registered.holder.upgrade())
            .collect::<Vec<_>>();
        drop(holders);
        let mut allowances = self
            .inner
            .allowances
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let live_allowances = allowances
            .iter()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        allowances.retain(|allowance| allowance.strong_count() != 0);
        let mut weight_mapped = 0_u64;
        let mut kv_mapped = 0_u64;
        let mut workspace_mapped = 0_u64;
        for allowance in &live_allowances {
            let mapped = allowance.mapped.load(Ordering::Acquire);
            match allowance.role {
                MemoryRole::Weights => weight_mapped = weight_mapped.saturating_add(mapped),
                MemoryRole::KvCache => kv_mapped = kv_mapped.saturating_add(mapped),
                MemoryRole::Workspace { .. } => {
                    workspace_mapped = workspace_mapped.saturating_add(mapped);
                }
                MemoryRole::Activation => {}
            }
        }
        MappedGrowthMetrics {
            attempts: self.inner.counters.attempts.load(Ordering::Relaxed),
            bytes_transferred: self
                .inner
                .counters
                .bytes_transferred
                .load(Ordering::Relaxed),
            failures: self.inner.counters.failures.load(Ordering::Relaxed),
            rollbacks: self.inner.counters.rollbacks.load(Ordering::Relaxed),
            live_holders: live.len() as u64,
            mapped_bytes: live_allowances
                .iter()
                .map(|allowance| allowance.mapped.load(Ordering::Acquire))
                .sum(),
            total_allowance_bytes: live_allowances
                .iter()
                .map(|allowance| allowance.limit.load(Ordering::Acquire))
                .sum(),
            weight_mapped,
            kv_mapped,
            workspace_mapped,
            total_owned: self.inner.ledger.used(Tier::Device),
        }
    }

    pub fn pause_transactions(&self) -> Result<MappedGrowthOperationGuard, MemoryError> {
        self.inner
            .active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: 0,
                reason: "a mapped-growth transaction or authority reconfiguration is active",
            })?;
        Ok(MappedGrowthOperationGuard {
            authority: Arc::clone(&self.inner),
        })
    }
}

pub struct MappedGrowthOperationGuard {
    authority: Arc<MappedGrowthAuthorityInner>,
}

impl Drop for MappedGrowthOperationGuard {
    fn drop(&mut self) {
        self.authority.active.store(false, Ordering::Release);
    }
}

pub struct MappedHolderRegistration {
    authority: Weak<MappedGrowthAuthorityInner>,
    id: u64,
}

impl Drop for MappedHolderRegistration {
    fn drop(&mut self) {
        if let Some(authority) = self.authority.upgrade() {
            authority
                .holders
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&self.id);
        }
    }
}

/// RAII reservation for one mapped growth operation.
pub struct MappedGrowthGrant {
    authority: Arc<MappedGrowthAuthorityInner>,
    requester: MappedAllowance,
    requested_bytes: u64,
    physical_reserved: u64,
    transferred_bytes: u64,
    victims: Vec<(MappedAllowance, u64)>,
    active: bool,
}

impl MappedGrowthGrant {
    pub fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }

    pub fn transferred_bytes(&self) -> u64 {
        self.transferred_bytes
    }

    /// Commit the grant into the requester's mapped attribution.
    pub fn commit(self) -> Result<(), MemoryError> {
        let requested = self.requested_bytes;
        self.commit_bytes(requested)
    }

    pub fn commit_bytes(mut self, actual_mapped_bytes: u64) -> Result<(), MemoryError> {
        self.reduce_to_actual(actual_mapped_bytes)?;
        self.requester.map_reserved(actual_mapped_bytes)?;
        self.authority
            .ledger
            .release_mapped_growth(Tier::Device, self.physical_reserved);
        self.physical_reserved = 0;
        self.authority
            .counters
            .bytes_transferred
            .fetch_add(self.transferred_bytes, Ordering::Relaxed);
        self.finish();
        Ok(())
    }

    /// Run a fallible allocator transaction while the reservation is live,
    /// committing attribution only after the allocator succeeds.
    pub fn commit_with(
        self,
        commit: impl FnOnce() -> Result<(), MemoryError>,
    ) -> Result<(), MemoryError> {
        commit()?;
        self.commit()
    }

    pub fn commit_with_bytes(
        mut self,
        commit: impl FnOnce() -> Result<u64, MemoryError>,
    ) -> Result<(), MemoryError> {
        let actual = commit()?;
        self.reduce_to_actual(actual)?;
        self.requester.map_reserved(actual)?;
        self.authority
            .ledger
            .release_mapped_growth(Tier::Device, self.physical_reserved);
        self.physical_reserved = 0;
        self.authority
            .counters
            .bytes_transferred
            .fetch_add(self.transferred_bytes, Ordering::Relaxed);
        self.finish();
        Ok(())
    }

    fn reduce_to_actual(&mut self, actual: u64) -> Result<(), MemoryError> {
        if actual > self.requested_bytes {
            return Err(MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: actual,
                reason: "allocator committed more mapped bytes than the live growth grant",
            });
        }
        let mut excess = self.requested_bytes - actual;
        self.requester.release_growth(excess);
        while excess > 0 {
            let Some((victim, transferred)) = self.victims.last_mut() else {
                break;
            };
            let restored = excess.min(*transferred);
            self.requester.subtract_limit_transferred(restored)?;
            victim.add_limit_transferred(restored)?;
            *transferred -= restored;
            self.transferred_bytes -= restored;
            excess -= restored;
            if *transferred == 0 {
                self.victims.pop();
            }
        }
        self.requested_bytes = actual;
        Ok(())
    }

    fn rollback(&mut self) {
        if !self.active {
            return;
        }
        self.requester.release_growth(self.requested_bytes);
        if self.transferred_bytes > 0 {
            let _ = self
                .requester
                .subtract_limit_transferred(self.transferred_bytes);
        }
        for (victim, bytes) in self.victims.drain(..).rev() {
            let _ = victim.add_limit_transferred(bytes);
        }
        self.authority
            .ledger
            .release_mapped_growth(Tier::Device, self.physical_reserved);
        self.physical_reserved = 0;
        self.authority
            .counters
            .rollbacks
            .fetch_add(1, Ordering::Relaxed);
        self.authority
            .counters
            .failures
            .fetch_add(1, Ordering::Relaxed);
        self.finish();
    }

    fn finish(&mut self) {
        if self.active {
            self.active = false;
            self.authority.active.store(false, Ordering::Release);
        }
    }
}

impl Drop for MappedGrowthGrant {
    fn drop(&mut self) {
        self.rollback();
    }
}

/// Exclusive limit-reconfiguration access for one ledger tier.
pub struct LeaseLimitGuard<'a> {
    ledger: &'a LeaseLedger,
    tier: Tier,
    _gate: MutexGuard<'a, ()>,
}

impl LeaseLimitGuard<'_> {
    pub fn used(&self) -> u64 {
        self.ledger.used(self.tier)
    }

    pub fn limit(&self) -> u64 {
        self.ledger.limit(self.tier)
    }

    /// Commit `bytes` only when current usage fits. New claims cannot race this
    /// check because the guard owns the same gate used by reserve/grow.
    pub fn try_set_limit(&self, bytes: u64) -> Result<(), u64> {
        let used = self.used();
        if used > bytes {
            return Err(used);
        }
        self.ledger.limits[self.tier.index()].store(bytes, Ordering::Release);
        Ok(())
    }
}

/// Where a [`MemoryLease`] gives its bytes back.
///
/// The lease is deliberately *not* tied to [`LeaseLedger`]. A third party
/// implementing [`MemoryGovernor`] has to be able to hand out leases, and a
/// lease that could only be released into this crate's own ledger would make
/// the trait unimplementable from outside — the accounting would have to be
/// ours, which is the opposite of a substitutable component.
///
/// `try_claim` is here alongside `release` because [`MemoryLease::grow`] needs
/// it: growing in place is what lets a lease expand at a tier that is merely
/// full rather than over-subscribed.
pub trait LeaseAccounting: Send + Sync + std::fmt::Debug {
    /// Charge `bytes` on `tier` for `role`, or fail without changing anything.
    fn try_claim(&self, tier: Tier, bytes: u64, role: MemoryRole) -> Result<(), MemoryError>;

    /// Give `bytes` back on `tier`.
    fn release(&self, tier: Tier, bytes: u64);
}

impl LeaseAccounting for LeaseLedger {
    fn try_claim(&self, tier: Tier, bytes: u64, role: MemoryRole) -> Result<(), MemoryError> {
        LeaseLedger::try_claim(self, tier, bytes, role)
    }

    fn release(&self, tier: Tier, bytes: u64) {
        LeaseLedger::release(self, tier, bytes);
    }
}

/// A granted claim on a tier.
///
/// Holding one is what entitles a component to occupy that many bytes. Dropping
/// it returns them; there is no `release` method, because an explicit one can be
/// skipped by an early return and G2 says a lease is released exactly once.
#[derive(Debug)]
pub struct MemoryLease {
    tier: Tier,
    bytes: u64,
    role: MemoryRole,
    holder: HolderId,
    accounting: Arc<dyn LeaseAccounting>,
}

impl MemoryLease {
    /// Build a lease over caller-supplied accounting.
    ///
    /// This is what makes [`MemoryGovernor`] implementable from outside this
    /// crate: an implementor charges its own books, then wraps the result here
    /// so the bytes are returned exactly once on drop, by the same rule
    /// everything else obeys.
    pub fn new(
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
        accounting: Arc<dyn LeaseAccounting>,
    ) -> Self {
        Self {
            tier,
            bytes,
            role,
            holder,
            accounting,
        }
    }

    /// The tier these bytes were taken from.
    pub fn tier(&self) -> Tier {
        self.tier
    }

    /// How many bytes this lease currently covers.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// What the bytes are for.
    pub fn role(&self) -> MemoryRole {
        self.role
    }

    /// Who holds it.
    pub fn holder(&self) -> HolderId {
        self.holder
    }

    /// Reserve a temporary sibling lease from the same accounting authority.
    ///
    /// This is for transactional replacement allocations that must coexist
    /// briefly with memory already covered by this lease. The returned lease
    /// releases automatically on every success and error path.
    pub fn reserve_sibling(&self, bytes: u64) -> Result<Self, MemoryError> {
        self.accounting.try_claim(self.tier, bytes, self.role)?;
        Ok(Self::new(
            self.tier,
            bytes,
            self.role,
            self.holder,
            Arc::clone(&self.accounting),
        ))
    }

    /// Transfer a sibling lease into this long-lived owner without changing
    /// the ledger's total charge.
    ///
    /// Both leases must originate from the same accounting authority and name
    /// the same tier, role, and holder. This is the commit step for
    /// transactions that preflight persistent growth with an independently
    /// droppable lease.
    pub fn absorb_sibling(&mut self, mut sibling: Self) -> Result<(), MemoryError> {
        if self.tier != sibling.tier
            || self.role != sibling.role
            || self.holder != sibling.holder
            || !Arc::ptr_eq(&self.accounting, &sibling.accounting)
        {
            return Err(MemoryError::InvalidRequest {
                tier: self.tier.name(),
                requested: sibling.bytes,
                reason: "sibling lease belongs to a different owner or accounting authority",
            });
        }
        self.bytes = self
            .bytes
            .checked_add(sibling.bytes)
            .ok_or(MemoryError::InvalidRequest {
                tier: self.tier.name(),
                requested: sibling.bytes,
                reason: "absorbing the sibling lease would overflow the owner lease",
            })?;
        sibling.bytes = 0;
        Ok(())
    }

    /// Extend this lease by `extra` bytes, or fail leaving it exactly as it was.
    ///
    /// Growing in place matters because the alternative — reserve a bigger lease
    /// then drop the old one — momentarily needs both, and would fail at a tier
    /// that is merely full rather than over-subscribed.
    pub fn grow(&mut self, extra: u64) -> Result<(), MemoryError> {
        if extra == 0 {
            return Ok(());
        }
        self.accounting.try_claim(self.tier, extra, self.role)?;
        self.bytes = self.bytes.saturating_add(extra);
        Ok(())
    }

    /// Give back `bytes`, keeping the rest of the lease.
    ///
    /// Returns how many bytes were actually returned, capped at the lease size
    /// so a caller cannot shrink tier usage below what it legitimately holds.
    pub fn shrink(&mut self, bytes: u64) -> u64 {
        let returned = bytes.min(self.bytes);
        if returned > 0 {
            self.accounting.release(self.tier, returned);
            self.bytes -= returned;
        }
        returned
    }
}

impl Drop for MemoryLease {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.accounting.release(self.tier, self.bytes);
        }
    }
}

/// How a holder gives memory back when a tier is under pressure.
///
/// Pull, with push notification: the governor asks, the holder decides. A holder
/// that cannot release anything returns zero and keeps its state; the requester
/// is refused instead (G3).
pub trait PressureResponder: Send + Sync {
    /// Release up to `want` bytes on `tier` and report how many were freed.
    ///
    /// Returning less than `want` — including zero — is a legitimate answer and
    /// must never be treated as permission to take the memory anyway.
    fn on_pressure(&self, tier: Tier, want: u64) -> u64;
}

/// The authority that grants leases.
pub trait MemoryGovernor {
    /// Stable identity of the physical-memory books this governor charges.
    ///
    /// Backings that retain physical allocations after unmapping use this to
    /// reject a buffer wired to different books. Implementations must return
    /// the same value for their lifetime. Two governors may return the same ID
    /// only when they deliberately share one physical-memory ledger.
    fn authority_id(&self) -> MemoryAuthorityId;

    /// Reserve `bytes` on `tier` for `role`, or fail without disturbing any
    /// existing lease.
    fn reserve(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MemoryLease, MemoryError>;

    /// Reserve mapped-byte attribution capacity without charging physical
    /// ownership a second time.
    ///
    /// Physical handle creation is charged through [`reserve`](Self::reserve).
    /// This separate allowance answers whether one holder may map/use that
    /// physical capacity. Implementations that do not provide authority-scoped
    /// attribution must reject rather than silently creating private books.
    fn reserve_mapped_allowance(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MappedAllowance, MemoryError> {
        let _ = (role, holder);
        Err(MemoryError::InvalidRequest {
            tier: tier.name(),
            requested: bytes,
            reason: "this governor does not support authority-scoped mapped-byte allowances",
        })
    }

    /// Explicitly register a weak, reloadable mapped holder with this authority.
    fn register_reclaimable_mapped_holder(
        &self,
        holder: &Arc<dyn ReclaimableMappedHolder>,
    ) -> Result<MappedHolderRegistration, MemoryError> {
        let _ = holder;
        Err(MemoryError::InvalidRequest {
            tier: Tier::Device.name(),
            requested: 0,
            reason: "this governor does not support transactional mapped growth",
        })
    }

    /// Reserve mapped growth without allowing ordinary claims or page-ins to
    /// consume the selected capacity before commit.
    fn prepare_mapped_growth(
        &self,
        requester: &MappedAllowance,
        bytes: u64,
    ) -> Result<MappedGrowthGrant, MemoryError> {
        let _ = requester;
        Err(MemoryError::InvalidRequest {
            tier: Tier::Device.name(),
            requested: bytes,
            reason: "this governor does not support transactional mapped growth",
        })
    }

    fn mapped_growth_metrics(&self) -> Option<MappedGrowthMetrics> {
        None
    }

    /// Record `bytes` on `tier` that are already committed for `role`.
    ///
    /// Unlike [`reserve`](MemoryGovernor::reserve), this is not an admission
    /// request. `reserve` asks "may I take these bytes?", which has a
    /// legitimate no. This states "I have taken these bytes", and refusing to
    /// record that fact does not un-commit memory; it only makes the ledger
    /// report room that does not exist.
    ///
    /// A tier may go over its limit as a result. That is the correct failure
    /// mode: an over-subscribed tier that says it is over-subscribed lets
    /// admission become conservative, whereas a refused record reproduces the
    /// two-sets-of-books bug this contract exists to remove.
    ///
    /// The default delegates to [`reserve`](MemoryGovernor::reserve) so
    /// existing third-party governors keep compiling. Governors that own a
    /// real ledger should override this to record the accomplished fact.
    fn record_committed(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MemoryLease, MemoryError> {
        self.reserve(tier, bytes, role, holder)
    }

    /// Bytes currently grantable on `tier`.
    fn available(&self, tier: Tier) -> u64;

    /// Bytes by which current leases exceed `tier`'s ceiling.
    ///
    /// This is separate from [`available`](MemoryGovernor::available) because
    /// `available` is a grantable-byte count and therefore saturates at zero.
    /// Zero can mean exactly full or over by gigabytes; only the second is an
    /// accounting fault admission must surface.
    fn oversubscribed_bytes(&self, _tier: Tier) -> u64 {
        0
    }

    /// Bytes currently leased on `tier`, across every holder.
    ///
    /// The counterpart to `available`, and the only exact answer to "what does
    /// this tier hold": leases are owned by the components that must outlive
    /// them -- a KV pool holds its own, an execution provider holds its pool's
    /// -- so no single caller can sum them up. The governor can, because
    /// everything went through it.
    ///
    /// Not itemised by holder. A governor that wanted to answer that would have
    /// to track attribution, which the reference implementation deliberately
    /// does not: `MemoryLease`'s `Drop` has to stay infallible and
    /// allocation-free.
    fn used(&self, tier: Tier) -> u64;
}

/// A governor backed by a [`LeaseLedger`].
///
/// This is the reference implementation of [`MemoryGovernor`], complete enough
/// to use directly. An engine wanting richer policy wraps it rather than
/// reimplementing the accounting.
#[derive(Debug, Clone)]
pub struct LedgerGovernor {
    ledger: Arc<LeaseLedger>,
    mapped_growth: MappedGrowthAuthority,
}

#[derive(Debug)]
struct LedgerMappedAllowanceAccounting {
    ledger: Arc<LeaseLedger>,
}

impl MappedAllowanceAccounting for LedgerMappedAllowanceAccounting {
    fn release(&self, tier: Tier, bytes: u64) {
        self.ledger.release_mapped_allowance(tier, bytes);
    }
}

impl LedgerGovernor {
    /// A governor over `ledger`.
    ///
    /// The authority identity belongs to the ledger, so every governor over the
    /// same ledger reports the same identity.
    pub fn new(ledger: Arc<LeaseLedger>) -> Self {
        Self {
            mapped_growth: MappedGrowthAuthority::new(Arc::clone(&ledger)),
            ledger,
        }
    }

    /// The underlying ledger, for reporting and limit changes.
    pub fn ledger(&self) -> &Arc<LeaseLedger> {
        &self.ledger
    }

    pub fn pause_mapped_growth(&self) -> Result<MappedGrowthOperationGuard, MemoryError> {
        self.mapped_growth.pause_transactions()
    }
}

impl MemoryGovernor for LedgerGovernor {
    fn authority_id(&self) -> MemoryAuthorityId {
        self.ledger.authority_id
    }

    fn reserve(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MemoryLease, MemoryError> {
        self.ledger.try_claim(tier, bytes, role)?;
        Ok(MemoryLease::new(
            tier,
            bytes,
            role,
            holder,
            Arc::clone(&self.ledger) as Arc<dyn LeaseAccounting>,
        ))
    }

    fn reserve_mapped_allowance(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MappedAllowance, MemoryError> {
        self.ledger
            .try_reserve_mapped_allowance(tier, bytes, role)?;
        let allowance = MappedAllowance::new(
            self.authority_id(),
            tier,
            bytes,
            role,
            holder,
            Arc::new(LedgerMappedAllowanceAccounting {
                ledger: Arc::clone(&self.ledger),
            }),
        );
        self.mapped_growth.track_allowance(&allowance);
        Ok(allowance)
    }

    fn register_reclaimable_mapped_holder(
        &self,
        holder: &Arc<dyn ReclaimableMappedHolder>,
    ) -> Result<MappedHolderRegistration, MemoryError> {
        self.mapped_growth.register(holder)
    }

    fn prepare_mapped_growth(
        &self,
        requester: &MappedAllowance,
        bytes: u64,
    ) -> Result<MappedGrowthGrant, MemoryError> {
        self.mapped_growth.prepare_mapped_growth(requester, bytes)
    }

    fn mapped_growth_metrics(&self) -> Option<MappedGrowthMetrics> {
        Some(self.mapped_growth.metrics())
    }

    fn record_committed(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MemoryLease, MemoryError> {
        self.ledger.record_claim(tier, bytes)?;
        Ok(MemoryLease::new(
            tier,
            bytes,
            role,
            holder,
            Arc::clone(&self.ledger) as Arc<dyn LeaseAccounting>,
        ))
    }

    fn available(&self, tier: Tier) -> u64 {
        self.ledger.available(tier)
    }

    fn oversubscribed_bytes(&self, tier: Tier) -> u64 {
        self.ledger.oversubscribed_bytes(tier)
    }

    fn used(&self, tier: Tier) -> u64 {
        self.ledger.used(tier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    struct TestMappedHolder {
        allowance: MappedAllowance,
        mapped: AtomicU64,
        max_reclaim: AtomicU64,
        priority: AtomicU32,
        targets: Mutex<Vec<u64>>,
        reentrant: Option<LedgerGovernor>,
    }

    impl TestMappedHolder {
        fn new(allowance: MappedAllowance, mapped: u64, max_reclaim: u64, priority: u32) -> Self {
            allowance.try_map(mapped).expect("initial mapped bytes");
            Self {
                allowance,
                mapped: AtomicU64::new(mapped),
                max_reclaim: AtomicU64::new(max_reclaim),
                priority: AtomicU32::new(priority),
                targets: Mutex::new(Vec::new()),
                reentrant: None,
            }
        }

        fn with_reentrant_governor(mut self, governor: LedgerGovernor) -> Self {
            self.reentrant = Some(governor);
            self
        }
    }

    impl ReclaimableMappedHolder for TestMappedHolder {
        fn allowance(&self) -> MappedAllowance {
            self.allowance.clone()
        }

        fn reclaim_priority(&self) -> u32 {
            self.priority.load(Ordering::Acquire)
        }

        fn mapped_bytes(&self) -> u64 {
            self.mapped.load(Ordering::Acquire)
        }

        fn reclaim_mapped(&self, target_bytes: u64) -> Result<MappedReclaimReport, MemoryError> {
            self.targets
                .lock()
                .expect("targets lock")
                .push(target_bytes);
            if let Some(governor) = &self.reentrant {
                let lease = governor.reserve(
                    Tier::Device,
                    1,
                    MemoryRole::Activation,
                    HolderId::new(999),
                )?;
                drop(lease);
            }
            let reclaimed = target_bytes
                .min(self.max_reclaim.load(Ordering::Acquire))
                .min(self.mapped.load(Ordering::Acquire));
            self.mapped.fetch_sub(reclaimed, Ordering::AcqRel);
            self.allowance.unmap(reclaimed);
            Ok(MappedReclaimReport {
                target_bytes,
                reclaimed_bytes: reclaimed,
            })
        }
    }

    fn mapped_allowance(
        governor: &LedgerGovernor,
        bytes: u64,
        role: MemoryRole,
        holder: u64,
    ) -> MappedAllowance {
        governor
            .reserve_mapped_allowance(Tier::Device, bytes, role, HolderId::new(holder))
            .expect("mapped allowance")
    }

    #[test]
    fn mapped_growth_transfers_unused_allowance_before_reclaiming() {
        let governor = LedgerGovernor::new(LeaseLedger::new(100, 0, 0));
        let victim = Arc::new(TestMappedHolder::new(
            mapped_allowance(&governor, 80, MemoryRole::Weights, 1),
            20,
            20,
            0,
        ));
        let requester = mapped_allowance(&governor, 20, MemoryRole::KvCache, 2);
        let holder: Arc<dyn ReclaimableMappedHolder> = victim.clone();
        let _registration = governor
            .register_reclaimable_mapped_holder(&holder)
            .expect("register victim");

        let grant = governor
            .prepare_mapped_growth(&requester, 30)
            .expect("unused allowance transfer");
        assert_eq!(grant.transferred_bytes(), 10);
        assert_eq!(victim.allowance.limit(), 70);
        assert!(
            victim.targets.lock().expect("targets").is_empty(),
            "mapped=20 is below new_limit=70, so physical reclaim must be zero"
        );
        grant.commit().expect("commit");
        assert_eq!(requester.mapped_bytes(), 30);
    }

    #[test]
    fn failed_pinned_reclaim_rolls_back_allowances_and_reservation() {
        let governor = LedgerGovernor::new(LeaseLedger::new(100, 0, 0));
        let victim = Arc::new(TestMappedHolder::new(
            mapped_allowance(&governor, 80, MemoryRole::Weights, 1),
            80,
            0,
            0,
        ));
        let requester = mapped_allowance(&governor, 20, MemoryRole::KvCache, 2);
        requester.try_map(20).expect("requester full");
        let holder: Arc<dyn ReclaimableMappedHolder> = victim.clone();
        let _registration = governor
            .register_reclaimable_mapped_holder(&holder)
            .expect("register victim");

        assert!(
            governor.prepare_mapped_growth(&requester, 20).is_err(),
            "pinned victim must refuse the requester"
        );
        assert_eq!(victim.allowance.limit(), 80);
        assert_eq!(requester.limit(), 20);
        assert_eq!(requester.mapped_bytes(), 20);
        assert_eq!(
            victim.targets.lock().expect("targets").as_slice(),
            &[20],
            "target is mapped-new_limit, not mapped-transfer"
        );
        assert_eq!(governor.available(Tier::Device), 100);
    }

    #[test]
    fn live_grant_blocks_ordinary_claims_and_requester_page_in() {
        let governor = LedgerGovernor::new(LeaseLedger::new(100, 0, 0));
        let _owned = governor
            .reserve(Tier::Device, 40, MemoryRole::Weights, HolderId::new(7))
            .expect("standing ownership");
        let victim = Arc::new(TestMappedHolder::new(
            mapped_allowance(&governor, 80, MemoryRole::Weights, 1),
            20,
            20,
            0,
        ));
        let requester = mapped_allowance(&governor, 20, MemoryRole::KvCache, 2);
        let holder: Arc<dyn ReclaimableMappedHolder> = victim;
        let _registration = governor
            .register_reclaimable_mapped_holder(&holder)
            .expect("register victim");
        let grant = governor
            .prepare_mapped_growth(&requester, 20)
            .expect("prepare");

        assert_eq!(governor.available(Tier::Device), 40);
        assert!(
            governor
                .reserve(Tier::Device, 41, MemoryRole::Activation, HolderId::new(3))
                .is_err()
        );
        assert!(requester.try_map(1).is_err());
        drop(grant);
        assert_eq!(governor.available(Tier::Device), 60);
        requester.try_map(1).expect("rollback restores page-in");
    }

    #[test]
    fn reclaim_callback_runs_outside_authority_and_claim_locks() {
        let governor = LedgerGovernor::new(LeaseLedger::new(100, 0, 0));
        let victim = Arc::new(
            TestMappedHolder::new(
                mapped_allowance(&governor, 80, MemoryRole::Weights, 1),
                80,
                80,
                0,
            )
            .with_reentrant_governor(governor.clone()),
        );
        let requester = mapped_allowance(&governor, 20, MemoryRole::KvCache, 2);
        requester.try_map(20).expect("requester full");
        let holder: Arc<dyn ReclaimableMappedHolder> = victim;
        let _registration = governor
            .register_reclaimable_mapped_holder(&holder)
            .expect("register victim");

        governor
            .prepare_mapped_growth(&requester, 10)
            .expect("reentrant callback")
            .commit()
            .expect("commit");
    }

    #[test]
    fn allocator_commit_failure_restores_allowances_and_reservation() {
        let governor = LedgerGovernor::new(LeaseLedger::new(100, 0, 0));
        let victim = Arc::new(TestMappedHolder::new(
            mapped_allowance(&governor, 80, MemoryRole::Weights, 1),
            80,
            80,
            0,
        ));
        let requester = mapped_allowance(&governor, 20, MemoryRole::KvCache, 2);
        requester.try_map(20).expect("requester full");
        let holder: Arc<dyn ReclaimableMappedHolder> = victim.clone();
        let _registration = governor
            .register_reclaimable_mapped_holder(&holder)
            .expect("register victim");

        let error = governor
            .prepare_mapped_growth(&requester, 10)
            .expect("grant")
            .commit_with(|| {
                Err(MemoryError::AllocationFailed {
                    tier: "device",
                    requested: 10,
                    reason: "injected commit failure".into(),
                })
            })
            .expect_err("commit fails");
        assert!(matches!(error, MemoryError::AllocationFailed { .. }));
        assert_eq!(victim.allowance.limit(), 80);
        assert_eq!(requester.limit(), 20);
        assert_eq!(governor.available(Tier::Device), 100);
    }

    #[test]
    fn registration_is_weak_explicit_and_priority_ordered() {
        let governor = LedgerGovernor::new(LeaseLedger::new(100, 0, 0));
        let slow = Arc::new(TestMappedHolder::new(
            mapped_allowance(&governor, 40, MemoryRole::Weights, 1),
            40,
            40,
            10,
        ));
        let first = Arc::new(TestMappedHolder::new(
            mapped_allowance(&governor, 40, MemoryRole::Weights, 2),
            40,
            40,
            0,
        ));
        let requester = mapped_allowance(&governor, 20, MemoryRole::KvCache, 3);
        requester.try_map(20).expect("requester full");
        let slow_dyn: Arc<dyn ReclaimableMappedHolder> = slow.clone();
        let first_dyn: Arc<dyn ReclaimableMappedHolder> = first.clone();
        let slow_registration = governor
            .register_reclaimable_mapped_holder(&slow_dyn)
            .expect("slow register");
        let first_registration = governor
            .register_reclaimable_mapped_holder(&first_dyn)
            .expect("first register");
        drop(first_registration);
        drop(first_dyn);
        drop(first);

        governor
            .prepare_mapped_growth(&requester, 10)
            .expect("remaining holder")
            .commit()
            .expect("commit");
        assert_eq!(slow.targets.lock().expect("targets").as_slice(), &[10]);
        let metrics = governor.mapped_growth_metrics().expect("metrics");
        assert_eq!(metrics.live_holders, 1);
        assert_eq!(metrics.weight_mapped, 30);
        assert_eq!(metrics.kv_mapped, 30);
        assert_eq!(metrics.workspace_mapped, 0);
        drop(slow_registration);
        assert_eq!(
            governor
                .mapped_growth_metrics()
                .expect("metrics")
                .live_holders,
            0
        );
    }

    #[test]
    fn mapped_allowances_coordinate_holders_without_double_charging_physical_bytes() {
        let governor = LedgerGovernor::new(LeaseLedger::new(100, 0, 0));
        let first = governor
            .reserve_mapped_allowance(Tier::Device, 60, MemoryRole::Weights, HolderId::new(1))
            .expect("first zone");
        assert_eq!(
            governor.available(Tier::Device),
            100,
            "mapped capacity is not physical ownership"
        );
        let error = governor
            .reserve_mapped_allowance(Tier::Device, 60, MemoryRole::Weights, HolderId::new(2))
            .expect_err("two holders cannot each reserve most of the device");
        assert!(matches!(
            error,
            MemoryError::TierExhausted { available: 40, .. }
        ));

        first.try_map(60).expect("map to zone limit");
        assert!(matches!(
            first.try_map(1),
            Err(MemoryError::TierExhausted { available: 0, .. })
        ));
        assert_eq!(first.unmap(20), 20);
        first
            .try_map(20)
            .expect("eviction restores mapped allowance");
        drop(first);
        governor
            .reserve_mapped_allowance(Tier::Device, 100, MemoryRole::Weights, HolderId::new(2))
            .expect("dropping zone returns authority capacity");
    }

    fn governor(device: u64, host: u64, disk: u64) -> LedgerGovernor {
        LedgerGovernor::new(LeaseLedger::new(device, host, disk))
    }

    const H: HolderId = HolderId::new(1);

    #[test]
    fn governors_over_one_ledger_share_authority_but_independent_ledgers_do_not() {
        let ledger = LeaseLedger::new(10, 0, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));
        let second_wrapper = LedgerGovernor::new(ledger);
        let independent = LedgerGovernor::new(LeaseLedger::new(10, 0, 0));

        assert_eq!(governor.authority_id(), second_wrapper.authority_id());
        assert_ne!(governor.authority_id(), independent.authority_id());
        assert_eq!(governor.authority_id().device(), DeviceKey::device(0));
    }

    #[test]
    fn record_committed_records_already_taken_bytes_even_over_limit() {
        let governor = governor(10, 0, 0);
        let held_first = governor
            .reserve(Tier::Device, 8, MemoryRole::Weights, H)
            .expect("initial lease fits");

        let already_committed = governor
            .record_committed(Tier::Device, 5, MemoryRole::Weights, HolderId::new(2))
            .expect("recording committed bytes is not an admission request");

        assert_eq!(
            governor.used(Tier::Device),
            13,
            "the ledger must report the true total after recording committed memory"
        );
        assert_eq!(governor.available(Tier::Device), 0);

        drop(already_committed);
        assert_eq!(governor.used(Tier::Device), 8);
        drop(held_first);
        assert_eq!(governor.used(Tier::Device), 0);
    }

    #[test]
    fn oversubscribed_bytes_reports_excess_beyond_the_limit() {
        let governor = governor(10, 0, 0);
        let _first = governor
            .reserve(Tier::Device, 8, MemoryRole::Weights, H)
            .expect("initial lease fits");
        let _committed = governor
            .record_committed(Tier::Device, 5, MemoryRole::Weights, HolderId::new(2))
            .expect("already committed memory is recorded");

        assert_eq!(governor.available(Tier::Device), 0);
        assert_eq!(governor.oversubscribed_bytes(Tier::Device), 3);
    }

    /// Shrinking then dropping releases each byte exactly once.
    ///
    /// The pair matters because both paths return memory to the same tier. If
    /// `shrink` released without decrementing what the lease still believes it
    /// holds, `Drop` would return those bytes a second time and the tier would
    /// drift *downwards* -- reporting free memory that is not free, which ends
    /// as an allocation failure somewhere with no connection to the cause.
    #[test]
    fn shrinking_then_dropping_releases_each_byte_once() {
        let ledger = LeaseLedger::new(1000, 0, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));

        {
            let mut lease = governor
                .reserve(Tier::Device, 600, MemoryRole::Weights, HolderId::new(1))
                .expect("600 of 1000");
            assert_eq!(ledger.used(Tier::Device), 600);

            assert_eq!(lease.shrink(200), 200, "shrink reports what it returned");
            assert_eq!(ledger.used(Tier::Device), 400);
            assert_eq!(lease.bytes(), 400, "the lease knows it holds less now");

            // Over-shrinking is clamped, not an underflow.
            assert_eq!(lease.shrink(9_999), 400);
            assert_eq!(ledger.used(Tier::Device), 0);
        }

        assert_eq!(
            ledger.used(Tier::Device),
            0,
            "dropping an emptied lease must not release anything a second time"
        );
    }

    /// A lease shrunk part-way and then dropped returns the whole amount, and
    /// no more.
    #[test]
    fn a_partly_shrunk_lease_returns_exactly_its_remainder_on_drop() {
        let ledger = LeaseLedger::new(1000, 0, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));

        {
            let mut lease = governor
                .reserve(Tier::Device, 500, MemoryRole::Weights, HolderId::new(2))
                .expect("500 of 1000");
            lease.shrink(150);
            assert_eq!(ledger.used(Tier::Device), 350);
        }

        assert_eq!(
            ledger.used(Tier::Device),
            0,
            "the 350 still held at drop must come back, and only once"
        );

        // The tier is genuinely reusable afterwards, which is the property a
        // leak would break silently: the numbers can look right while the
        // memory is unobtainable.
        let again = governor
            .reserve(Tier::Device, 1000, MemoryRole::Weights, HolderId::new(3))
            .expect("the whole tier is free again after every lease is gone");
        assert_eq!(again.bytes(), 1000);
    }

    /// Many reserve/drop cycles leave the tier exactly where they found it.
    ///
    /// A one-byte drift per cycle is invisible in a test that runs one cycle
    /// and fatal in a server that runs millions.
    #[test]
    fn repeated_lease_cycles_do_not_drift() {
        let ledger = LeaseLedger::new(4096, 0, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));

        for round in 0..1000u64 {
            let bytes = 1 + round % 512;
            let mut lease = governor
                .reserve(Tier::Device, bytes, MemoryRole::KvCache, HolderId::new(4))
                .expect("well under the tier limit");
            // Exercise both return paths: some rounds shrink to nothing, some
            // leave a remainder for `Drop`.
            if round % 3 == 0 {
                lease.shrink(bytes);
            } else if round % 3 == 1 {
                lease.shrink(bytes / 2);
            }
        }

        assert_eq!(
            ledger.used(Tier::Device),
            0,
            "1000 reserve/shrink/drop cycles must leave the tier empty"
        );
    }

    /// G1: the sum of live leases never exceeds the tier limit.
    ///
    /// Stated as "the last byte is grantable and the one after it is not",
    /// because an off-by-one here is the difference between a limit that holds
    /// and one that is decorative.
    #[test]
    fn a_tier_grants_exactly_its_limit_and_not_one_byte_more() {
        let gov = governor(1000, 0, 0);
        let _a = gov
            .reserve(Tier::Device, 600, MemoryRole::KvCache, H)
            .expect("600 of 1000 fits");
        let _b = gov
            .reserve(Tier::Device, 400, MemoryRole::KvCache, H)
            .expect("the remaining 400 fits exactly");
        assert_eq!(gov.available(Tier::Device), 0);

        let error = gov
            .reserve(Tier::Device, 1, MemoryRole::KvCache, H)
            .expect_err("a full tier must refuse even one byte");
        assert!(
            matches!(error, MemoryError::TierExhausted { available: 0, .. }),
            "{error}"
        );
    }

    /// G4: a refused reservation leaves existing leases untouched.
    #[test]
    fn a_refused_reservation_does_not_disturb_what_is_already_leased() {
        let gov = governor(1000, 0, 0);
        let held = gov
            .reserve(Tier::Device, 800, MemoryRole::Weights, H)
            .expect("800 of 1000 fits");

        for _ in 0..5 {
            gov.reserve(Tier::Device, 500, MemoryRole::KvCache, H)
                .expect_err("500 does not fit alongside 800");
        }

        assert_eq!(held.bytes(), 800, "the surviving lease changed size");
        assert_eq!(
            gov.ledger().used(Tier::Device),
            800,
            "repeated failures leaked or double-counted tier usage"
        );
    }

    /// G2: dropping a lease returns its bytes, exactly once.
    #[test]
    fn dropping_a_lease_returns_its_bytes_once() {
        let gov = governor(100, 0, 0);
        {
            let _lease = gov
                .reserve(Tier::Device, 100, MemoryRole::KvCache, H)
                .expect("the whole tier fits");
            assert_eq!(gov.available(Tier::Device), 0);
        }
        assert_eq!(
            gov.available(Tier::Device),
            100,
            "dropping the lease did not return its bytes"
        );

        // Reserving the whole tier again proves the release was not counted
        // twice: a double release would have inflated the free pool, and the
        // one-byte reservation after it would then wrongly succeed.
        let _again = gov
            .reserve(Tier::Device, 100, MemoryRole::KvCache, H)
            .expect("the tier is free again");
        gov.reserve(Tier::Device, 1, MemoryRole::KvCache, H)
            .expect_err("a double release would have left phantom capacity");
    }

    /// Tiers are independent budgets, not one pool with labels.
    #[test]
    fn exhausting_one_tier_does_not_consume_another() {
        let gov = governor(100, 100, 100);
        let _device = gov
            .reserve(Tier::Device, 100, MemoryRole::KvCache, H)
            .expect("device fits");
        assert_eq!(gov.available(Tier::Device), 0);
        assert_eq!(gov.available(Tier::Host), 100);
        assert_eq!(gov.available(Tier::Disk), 100);
    }

    /// Growing in place must not need room for both the old and the new size.
    #[test]
    fn growing_a_lease_only_claims_the_difference() {
        let gov = governor(100, 0, 0);
        let mut lease = gov
            .reserve(Tier::Device, 60, MemoryRole::KvCache, H)
            .expect("60 of 100 fits");
        lease
            .grow(40)
            .expect("the extra 40 fits; reserving 100 afresh would not");
        assert_eq!(lease.bytes(), 100);
        assert_eq!(gov.available(Tier::Device), 0);
    }

    #[test]
    fn reserve_racing_limit_shrink_never_exceeds_the_committed_limit() {
        use std::sync::Barrier;
        use std::thread;

        for _ in 0..100 {
            let ledger = LeaseLedger::new(100, 0, 0);
            let governor = LedgerGovernor::new(Arc::clone(&ledger));
            let barrier = Arc::new(Barrier::new(2));
            let reserve_barrier = Arc::clone(&barrier);
            let reserve_governor = governor.clone();
            let reserve = thread::spawn(move || {
                reserve_barrier.wait();
                reserve_governor.reserve(Tier::Device, 80, MemoryRole::KvCache, H)
            });
            barrier.wait();
            let shrink = {
                let guard = ledger.pause_claims(Tier::Device);
                guard.try_set_limit(50)
            };
            let lease = reserve.join().expect("reserve thread");

            assert!(
                (shrink.is_ok() && lease.is_err()) || (shrink.is_err() && lease.is_ok()),
                "exactly one racing operation must win"
            );
            assert!(ledger.used(Tier::Device) <= ledger.limit(Tier::Device));
        }
    }

    /// A refused growth leaves the lease exactly as it was.
    #[test]
    fn a_refused_growth_leaves_the_lease_at_its_original_size() {
        let gov = governor(100, 0, 0);
        let mut lease = gov
            .reserve(Tier::Device, 60, MemoryRole::KvCache, H)
            .expect("60 of 100 fits");
        lease
            .grow(41)
            .expect_err("60 + 41 exceeds the 100 byte tier");
        assert_eq!(lease.bytes(), 60, "a failed growth resized the lease");
        assert_eq!(gov.ledger().used(Tier::Device), 60);
        lease.grow(40).expect("the tier still has 40 free");
    }

    /// Shrinking returns memory without giving up the lease.
    #[test]
    fn shrinking_returns_only_what_the_lease_actually_holds() {
        let gov = governor(100, 0, 0);
        let mut lease = gov
            .reserve(Tier::Device, 50, MemoryRole::KvCache, H)
            .expect("50 of 100 fits");
        assert_eq!(lease.shrink(80), 50, "shrink must cap at the lease size");
        assert_eq!(lease.bytes(), 0);
        assert_eq!(
            gov.available(Tier::Device),
            100,
            "over-shrinking invented capacity"
        );
    }

    /// Lowering a limit below live usage must not revoke anything (G3).
    ///
    /// The observable is tier *usage*, not availability: seizing a holder's
    /// bytes back would clamp usage to the new limit while leaving availability
    /// at zero and new requests refused either way, so asserting on those two
    /// alone would pass against an implementation that revokes.
    #[test]
    fn lowering_a_limit_below_live_usage_revokes_nothing() {
        let gov = governor(1000, 0, 0);
        let lease = gov
            .reserve(Tier::Device, 900, MemoryRole::KvCache, H)
            .expect("900 of 1000 fits");

        gov.ledger().set_limit(Tier::Device, 100);

        assert_eq!(
            gov.ledger().used(Tier::Device),
            900,
            "the limit change took bytes back from a live holder"
        );
        assert_eq!(
            lease.bytes(),
            900,
            "a limit change resized a lease the holder still owns"
        );
        assert_eq!(gov.available(Tier::Device), 0);
        gov.reserve(Tier::Device, 1, MemoryRole::KvCache, H)
            .expect_err("an over-subscribed tier must grant nothing further");

        // And the over-subscription resolves by the holder letting go, not by
        // the governor having quietly reclaimed anything earlier.
        drop(lease);
        assert_eq!(gov.ledger().used(Tier::Device), 0);
        assert_eq!(gov.available(Tier::Device), 100);
    }

    /// A zero-byte reservation is legal and holds nothing.
    #[test]
    fn a_zero_byte_reservation_is_legal_and_holds_nothing() {
        let gov = governor(0, 0, 0);
        let lease = gov
            .reserve(Tier::Device, 0, MemoryRole::Activation, H)
            .expect("zero bytes always fits, even in an empty tier");
        assert_eq!(lease.bytes(), 0);
    }

    /// An overflowing request is refused as invalid rather than wrapping.
    #[test]
    fn an_overflowing_request_is_refused_rather_than_wrapping() {
        let gov = governor(u64::MAX, 0, 0);
        let _held = gov
            .reserve(Tier::Device, 10, MemoryRole::KvCache, H)
            .expect("10 bytes fit");
        let error = gov
            .reserve(Tier::Device, u64::MAX, MemoryRole::KvCache, H)
            .expect_err("used + requested overflows u64");
        assert!(
            matches!(error, MemoryError::InvalidRequest { .. }),
            "expected an invalid-request error, got {error}"
        );
        assert_eq!(gov.ledger().used(Tier::Device), 10, "usage was corrupted");
    }

    /// Concurrent reservations must not oversubscribe a tier.
    ///
    /// The claim path is a compare-exchange loop, so a lost update would show up
    /// as more total bytes granted than the tier has.
    #[test]
    fn concurrent_reservations_never_oversubscribe_a_tier() {
        use std::sync::Barrier;
        use std::thread;

        const THREADS: usize = 8;
        const LIMIT: u64 = 100;

        let gov = governor(LIMIT, 0, 0);
        let barrier = Arc::new(Barrier::new(THREADS));

        let granted: u64 = thread::scope(|scope| {
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let gov = gov.clone();
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        // Each thread grabs single bytes until refused, so they
                        // race for the last capacity in the tier.
                        let mut leases = Vec::new();
                        while let Ok(lease) = gov.reserve(Tier::Device, 1, MemoryRole::KvCache, H) {
                            leases.push(lease);
                        }
                        let held: u64 = leases.iter().map(MemoryLease::bytes).sum();
                        // Keep them held so the final usage assertion is about
                        // what was granted, not what survived scope exit.
                        std::mem::forget(leases);
                        held
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("reserving thread panicked"))
                .sum()
        });

        assert_eq!(
            granted, LIMIT,
            "threads together were granted {granted} bytes from a {LIMIT} byte tier"
        );
        assert_eq!(gov.ledger().used(Tier::Device), LIMIT);
    }

    /// Dropping a grown lease returns everything it ended up holding.
    ///
    /// The bytes released come from the lease's own count, so a lease that grew
    /// and then released only its original size would leak the difference on
    /// every page-in that needed room.
    #[test]
    fn dropping_a_grown_lease_returns_the_grown_total() {
        let ledger = LeaseLedger::new(1000, 0, 0);
        let governor = LedgerGovernor::new(Arc::clone(&ledger));
        {
            let mut lease = governor
                .reserve(Tier::Device, 100, MemoryRole::Weights, HolderId::new(1))
                .expect("fits");
            lease.grow(300).expect("fits");
            assert_eq!(ledger.used(Tier::Device), 400);
        }
        assert_eq!(
            ledger.used(Tier::Device),
            0,
            "the grown portion was not returned"
        );
    }
}

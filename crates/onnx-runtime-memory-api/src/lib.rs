//! # `onnx-runtime-memory-api`
//!
//! Foundational memory mechanism contracts: primitive types, device keys,
//! shared-prefix traits, and the error vocabulary used by every memory-aware
//! crate in the workspace.
//!
//! This crate exists so that a third-party allocator or execution provider can
//! depend on the mechanism vocabulary without pulling in the governance layer
//! (`onnx-runtime-memory-governor`), session management, or any concrete
//! execution-provider implementation.
//!
//! ## What lives here
//!
//! * [`Tier`] — device / host / disk placement.
//! * [`MemoryRole`] — what a reservation is for (KV, workspace, weights, …).
//! * [`MemoryError`] — the error vocabulary shared by allocators and governors.
//! * [`DeviceKey`] — identifies a specific device within a tier.
//! * [`AllocationCommitRange`], [`MappedAllocation`] — commit-path primitives.
//! * [`SharedDevicePrefix`], [`SharedPrefixCommitInfo`] — shared-physical-prefix
//!   contract.
//!
//! ## What stays in `onnx-runtime-memory-governor`
//!
//! * `DeviceAllocator` trait —
//!   its `allocate_committed_with_capacity` and
//!   `commit_allocation_ranges_with_capacity` methods accept
//!   `MappedPhysicalCapacityToken`, a governor-coupled type. Splitting those
//!   into an extension trait is deferred to Phase 2 (#1186).
//! * `HostAllocator` — trivially
//!   depends on `DeviceAllocator`.
//! * Lease ledgers, capacity tokens, growth grants, pressure responders, holder
//!   identities, and the `MemoryGovernor` trait.

pub mod allocator;

pub use allocator::{
    AllocationCommitRange, DeviceKey, MappedAllocation, SharedDevicePrefix, SharedPrefixCommitInfo,
};

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
    pub const fn index(self) -> usize {
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
    /// for something impossible.
    #[error("cannot allocate {requested} bytes of {tier} memory: {reason}")]
    AllocationFailed {
        /// Which tier was addressed.
        tier: &'static str,
        /// What the caller asked for.
        requested: u64,
        /// What the backing allocator said.
        reason: String,
    },
    /// A well-formed capacity transfer or backing claim could not make enough
    /// governed bytes available.
    #[error(
        "cannot make {requested} bytes of {tier} capacity available for {role:?}: only \
         {available} bytes became available; {detail}"
    )]
    CapacityUnavailable {
        tier: &'static str,
        requested: u64,
        available: u64,
        role: MemoryRole,
        detail: String,
    },
}

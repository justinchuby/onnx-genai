//! # `onnx-runtime-memory-api`
//!
//! Low-dependency memory mechanism contracts shared by allocators, governors,
//! and execution providers.
//!
//! This crate is the lowest layer of the runtime memory stack. It owns the
//! minimum ordinary allocator contract, explicit optional virtual-backing and
//! shared-mapping capabilities, manager-issued binding identity/lifetime pins,
//! and the owning/deferred-release contract that says who owns a physical
//! release and what is true after one partially fails. It does not own
//! allocation policy, accounting, synchronization, or process-level transaction
//! management.
//!
//! Governor-specific capacity tokens and grants remain in
//! `onnx-runtime-memory-governor`; they are not methods every allocator or
//! optional capability must implement.

pub mod allocator;
pub mod binding;
pub mod capability;
pub mod deferred;

pub use allocator::{
    AllocationCommitRange, DeviceAllocator, DeviceKey, HostAllocator, MappedAllocation,
    SharedDevicePrefix, SharedPrefixCommitInfo,
};
pub use binding::{
    AllocationGeneration, AllocationIdentity, AuthorityIdentity, BindingError, BindingGeneration,
    BindingId, BindingIdentity, BindingRegistry, BindingResource, BoundAllocation, BoundMemoryView,
    BoundSharedMapping, BoundSharedPrefix, BoundVirtualBacking, ExplicitReleaseError,
    MechanismCoherence, MechanismIdentity, MechanismLifecycle, MechanismSnapshot, MemoryBinding,
    OwnedView, OwningAllocation, OwningReleaseError, ProviderContextIdentity, RegisteredAuthority,
    RegisteredMechanism, RegisteredProviderContext, ValidatedMemoryView,
};
pub use capability::{SharedMapping, VirtualBacking};
pub use deferred::{
    AllocationReleaseOutcome, AllocationReleaseState, DeferredEnqueueError,
    DeferredEnqueueRejection, DeferredReleaseDisposition, DeferredReleaseQueue,
    PreparedAllocationRelease, QuarantineReason, QuarantinedAllocation, ReleaseAccounting,
    ReleaseFailure, ResidualOwnership,
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

/// What a reservation is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryRole {
    KvCache,
    Workspace { step_scoped: bool },
    Weights,
    Activation,
}

/// Shared error vocabulary for mechanism and governance operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MemoryError {
    #[error(
        "cannot reserve {requested} bytes of {tier} memory for {role:?}: {used} of {limit} bytes \
         are already leased, leaving {available}; free memory by closing sessions, lower the \
         demand, or raise the {tier} limit"
    )]
    TierExhausted {
        tier: &'static str,
        requested: u64,
        used: u64,
        limit: u64,
        available: u64,
        role: MemoryRole,
    },
    #[error("cannot reserve {requested} bytes of {tier} memory: {reason}")]
    InvalidRequest {
        tier: &'static str,
        requested: u64,
        reason: &'static str,
    },
    #[error("cannot allocate {requested} bytes of {tier} memory: {reason}")]
    AllocationFailed {
        tier: &'static str,
        requested: u64,
        reason: String,
    },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_order_and_names_are_stable() {
        assert_eq!(Tier::ALL, [Tier::Device, Tier::Host, Tier::Disk]);
        assert_eq!(Tier::Device.name(), "device");
        assert_eq!(Tier::Host.name(), "host");
        assert_eq!(Tier::Disk.name(), "disk");
    }
}

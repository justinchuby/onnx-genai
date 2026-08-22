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
pub mod context_pin;
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
pub use context_pin::{ProviderContextPin, ProviderContextPinError, ProviderContextPinSource};
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
///
/// Not `Clone`/`PartialEq`: a refusal that carries the cause underneath it
/// cannot be meaningfully duplicated or compared, and keeping the cause is
/// worth more than either. Match on the variant instead.
#[derive(Debug, thiserror::Error)]
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
        /// What this layer can say about the shortfall on its own.
        ///
        /// Names the operation that came up short; the refusal underneath it,
        /// when there was one, belongs in `source` rather than being folded in
        /// here, so that a caller can still match on it and a reader is not
        /// shown the same sentence twice.
        detail: String,
        /// The refusal this one is reporting, when it is reporting one.
        ///
        /// `None` when this layer decided on its own, as when a reclaim target
        /// simply was not reached. Typed as `dyn Error` rather than a boxed
        /// [`MemoryError`] both because the layer underneath is not always a
        /// governor and because `#[source]` on a `Box<ConcreteError>` hands
        /// callers a chain node whose concrete type is the *box*, so
        /// `downcast_ref::<MemoryError>()` would miss it -- which is the whole
        /// thing this field exists to make possible.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
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

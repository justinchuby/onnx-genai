//! Compatibility re-exports for the mechanism contracts.
//!
//! The ordinary allocator and optional capabilities live in
//! `onnx-runtime-memory-api`. Existing governor import paths remain available
//! so Phase 2 does not require an unrelated workspace-wide import migration.

pub use onnx_runtime_memory_api::{
    AllocationCommitRange, AllocationGeneration, AllocationIdentity, AuthorityIdentity,
    BindingError, BindingGeneration, BindingId, BindingIdentity, BindingRegistry, BindingResource,
    BoundAllocation, BoundMemoryView, BoundSharedMapping, BoundSharedPrefix, BoundVirtualBacking,
    DeviceAllocator, DeviceKey, ExplicitReleaseError, HostAllocator, MappedAllocation,
    MechanismCoherence, MechanismIdentity, MechanismLifecycle, MechanismSnapshot, MemoryBinding,
    ProviderContextIdentity, RegisteredAuthority, RegisteredMechanism, RegisteredProviderContext,
    SharedDevicePrefix, SharedMapping, SharedPrefixCommitInfo, ValidatedMemoryView, VirtualBacking,
};

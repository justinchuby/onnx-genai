//! Compatibility re-exports for the mechanism contracts.
//!
//! The ordinary allocator and optional capabilities live in
//! `onnx-runtime-memory-api`. Existing governor import paths remain available
//! so Phase 2 does not require an unrelated workspace-wide import migration.

pub use onnx_runtime_memory_api::{
    AllocationCommitRange, DeviceAllocator, DeviceKey, HostAllocator, MappedAllocation,
    SharedDevicePrefix, SharedMapping, SharedPrefixCommitInfo, VirtualBacking,
};

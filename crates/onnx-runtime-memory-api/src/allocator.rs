//! Primitive data types shared by allocator mechanism interfaces.
//!
//! The `DeviceAllocator` trait and `HostAllocator` implementation remain in
//! `onnx-runtime-memory-governor` because the trait currently references
//! governor-specific capacity tokens and errors. These data types have no such
//! coupling.

use std::any::Any;
use std::fmt::Debug;
use std::ptr::NonNull;

use crate::Tier;

#[derive(Clone, Copy, Debug)]
pub struct AllocationCommitRange {
    pub ptr: NonNull<u8>,
    pub allocation_bytes: usize,
    pub align: usize,
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Debug)]
pub struct MappedAllocation<T> {
    pub allocation: T,
    pub newly_mapped_bytes: u64,
}

/// Which physical device memory comes from.
///
/// A `Tier` says *how far away* memory is; this says *which one*. Two CUDA
/// devices are the same tier and different allocators, and an allocator that
/// could not tell them apart would let a pointer from one be freed by the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceKey {
    /// How far the memory is from compute.
    pub tier: Tier,
    /// Which device of that tier, zero-based. Always `0` for host memory.
    pub index: u32,
}

impl DeviceKey {
    /// The host.
    pub const HOST: Self = Self {
        tier: Tier::Host,
        index: 0,
    };

    /// Accelerator `index`.
    pub const fn device(index: u32) -> Self {
        Self {
            tier: Tier::Device,
            index,
        }
    }
}

/// A pinned, read-only shared prefix: physical device memory created **once**
/// and mappable into many allocations at **zero** incremental owned bytes
/// (#777).
///
/// This is the allocator-agnostic handle a KV path holds when it declares "this
/// token prefix is shared" and pins it once, then maps it into each subsequent
/// sequence with `DeviceAllocator::commit_shared_prefix`. It is deliberately
/// opaque: the concrete backing (CUDA VMM physical handles today) lives in the
/// allocator crate, downcast through [`SharedDevicePrefix::as_any`] by the
/// allocator that produced it. Detection (hashing) and copy-on-write at
/// divergence are **not** part of this contract -- a shared prefix is read-only
/// for the union lifetime of its sharers.
pub trait SharedDevicePrefix: Send + Sync + Debug {
    /// Device address of the owner's writable window. The prefix content is
    /// filled here **once**, before it is shared read-only into sequences.
    fn device_ptr(&self) -> u64;

    /// Physical device bytes this prefix owns -- charged **once**, on the owned
    /// axis, however many sequences share it. This is the reported *physical*
    /// cost, never nominal content bytes.
    fn committed_physical_bytes(&self) -> u64;

    /// The granule-rounded byte length the prefix actually spans.
    fn mapped_bytes(&self) -> usize;

    /// Bytes requested at construction, before granule rounding.
    fn requested_bytes(&self) -> usize;

    /// Downcast hook: the allocator that produced this handle recovers its
    /// concrete type to map it. A prefix presented to a different allocator is
    /// refused rather than mis-mapped.
    fn as_any(&self) -> &dyn Any;
}

/// The accounting outcome of mapping a [`SharedDevicePrefix`] into one
/// allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedPrefixCommitInfo {
    /// Physical bytes newly *owned* by mapping the prefix here.
    ///
    /// Always **zero**: the prefix's granules were charged once when it was
    /// created, so admitting the Nth sharer costs only its *private* bytes.
    pub additional_owned_bytes: u64,
    /// Physical bytes newly *mapped* into this allocation's reservation -- one
    /// mapping of already-owned physical memory, reported on the mapped axis.
    pub newly_mapped_bytes: u64,
    /// Granules mapped read-only into the allocation.
    pub granules: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_key_basics() {
        assert_eq!(DeviceKey::HOST.tier, Tier::Host);
        assert_eq!(DeviceKey::device(1).tier, Tier::Device);
        assert_ne!(DeviceKey::device(0), DeviceKey::device(1));
    }
}

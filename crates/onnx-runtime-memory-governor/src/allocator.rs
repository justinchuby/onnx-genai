//! The allocator seam: raw memory, from wherever the caller says.
//!
//! # Relationship with `onnx-runtime-memory-api`
//!
//! The primitive data types, [`DeviceAllocator`], [`HostAllocator`], and the
//! optional [`VirtualBacking`]/[`SharedMapping`] capabilities are defined in
//! `onnx-runtime-memory-api` and re-exported here for backward compatibility.
//!
//! # Governor-coupled capacity accounting
//!
//! `DeviceAllocator::allocate_committed_with_capacity` and
//! `commit_allocation_ranges_with_capacity` used to live on `DeviceAllocator`
//! itself, taking a governor-specific
//! [`MappedPhysicalCapacityToken`](crate::MappedPhysicalCapacityToken). A
//! mechanism-only trait accepting a governor type was exactly the coupling
//! Phase 2 of #1186 asked to remove, and there is no coherent way to make
//! that atomic, capacity-charging path an extension trait: one concrete
//! allocator (`onnx-runtime-cuda-memory`'s `CudaVmmAllocator`) needs to charge
//! the token inside the same lock that claims physical granules, and any
//! other implementation would need a different (non-atomic) composition, so
//! a single blanket implementation cannot serve both without a coherence
//! conflict.
//!
//! These two methods are therefore not part of any trait here at all: they
//! live as inherent methods directly on `CudaVmmAllocator`, reached by an
//! execution provider that already depends on and constructs that concrete
//! type (as `onnx-runtime-ep-cuda` does through its own `vmm: OnceLock<Arc<
//! CudaVmmAllocator>>` field) rather than through `&dyn DeviceAllocator`.

// ── Re-exports from onnx-runtime-memory-api ──────────────────────────────
pub use onnx_runtime_memory_api::allocator::{
    AllocationCommitRange, DeviceAllocator, DeviceKey, HostAllocator, MappedAllocation,
    ReleaseReport, SharedDevicePrefix, SharedPrefixCommitInfo,
};
pub use onnx_runtime_memory_api::capability::{
    DeviceMemoryMechanism, SharedMapping, VirtualBacking,
};

#[cfg(test)]
mod tests {
    use std::ptr::NonNull;

    use super::*;
    use crate::MemoryError;

    #[test]
    fn an_allocator_with_no_virtual_backing_reports_none() {
        #[derive(Debug)]
        struct Stub;

        impl DeviceAllocator for Stub {
            fn allocate(&self, _bytes: usize, _align: usize) -> Result<NonNull<u8>, MemoryError> {
                Err(MemoryError::InvalidRequest {
                    tier: "device",
                    requested: 0,
                    reason: "test double",
                })
            }

            unsafe fn deallocate(&self, _ptr: NonNull<u8>, _bytes: usize, _align: usize) {}

            fn device(&self) -> DeviceKey {
                DeviceKey::device(0)
            }
        }

        let eager: &dyn DeviceAllocator = &Stub;
        assert!(
            eager.as_virtual_backing().is_none(),
            "an eager allocator has no lazy-commit capability to discover"
        );
        assert!(
            eager.as_shared_mapping().is_none(),
            "an eager allocator has no shared-prefix capability to discover"
        );
    }

    #[test]
    fn host_allocations_are_aligned_as_requested() {
        let allocator = HostAllocator;
        for (bytes, align) in [(1usize, 64usize), (100, 64), (4096, 256), (7, 8)] {
            let ptr = allocator.allocate(bytes, align).expect("granted");
            assert_eq!(
                ptr.as_ptr() as usize % align,
                0,
                "{bytes} bytes at {align}-byte alignment came back misaligned"
            );
            unsafe { allocator.deallocate(ptr, bytes, align) };
        }
    }

    #[test]
    fn the_host_allocator_reports_the_host() {
        assert_eq!(HostAllocator.device(), DeviceKey::HOST);
        assert_eq!(DeviceKey::HOST.tier, crate::Tier::Host);
        assert_eq!(DeviceKey::device(1).tier, crate::Tier::Device);
        assert_ne!(DeviceKey::device(0), DeviceKey::device(1));
    }
}

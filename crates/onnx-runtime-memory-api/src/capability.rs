//! Optional capabilities discovered from an already-selected allocator.
//!
//! Eager allocation needs neither trait. A VMM allocator exposes
//! [`VirtualBacking`]; a mechanism with reusable physical handles independently
//! exposes [`SharedMapping`]. Neither capability owns whole-allocation release.

use std::fmt::Debug;
use std::ptr::NonNull;

use crate::{AllocationCommitRange, MemoryError, SharedDevicePrefix, SharedPrefixCommitInfo};

/// Lazy virtual reservation and physical commit/decommit.
///
/// All pointers accepted or returned here belong to the coherent
/// [`crate::DeviceAllocator`] from which this capability was discovered.
/// Terminal release must go through that allocator's canonical release path.
pub trait VirtualBacking: Send + Sync + Debug {
    /// Reserve one allocation while committing only the ranges immediately live.
    fn allocate_committed(
        &self,
        bytes: usize,
        align: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<NonNull<u8>, MemoryError>;

    fn commit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<(), MemoryError>;

    fn commit_allocation_ranges(
        &self,
        ranges: &[AllocationCommitRange],
    ) -> Result<(), MemoryError> {
        for range in ranges {
            self.commit_allocation_range(
                range.ptr,
                range.allocation_bytes,
                range.align,
                range.offset,
                range.bytes,
            )?;
        }
        Ok(())
    }

    /// Conservative mapped-byte estimate for a batched commit.
    ///
    /// Required because only the concrete mechanism knows its granularity and
    /// whether disjoint tiny ranges share physical granules.
    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[AllocationCommitRange],
    ) -> Result<u64, MemoryError>;

    fn mapped_bytes_for_allocation(&self, bytes: usize, align: usize) -> Result<u64, MemoryError>;

    /// Release physical backing while retaining the virtual allocation.
    ///
    /// Returns actual newly unmapped bytes after granularity and shared
    /// references are applied. This is not whole-allocation release.
    fn decommit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<u64, MemoryError>;

    fn allocation_committed_bytes(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
    ) -> usize;
}

/// Reusable shared physical handles and read-only prefix mappings.
///
/// This capability is independent of [`VirtualBacking`]. A pool-less VMM may
/// expose virtual backing but no shared mapping, and another coherent
/// mechanism may expose shared mapping without virtual backing.
pub trait SharedMapping: Send + Sync + Debug {
    fn create_shared_prefix(
        &self,
        bytes: usize,
    ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError>;

    /// Incremental owned physical cost of admitting another mapping.
    ///
    /// Zero is valid only for a prefix this capability can actually map. A
    /// wrong-device, wrong-authority, or foreign prefix must return a
    /// conservative non-zero cost and be rejected by
    /// [`commit_shared_prefix`](Self::commit_shared_prefix).
    fn incremental_owned_bytes_for_shared_prefix(&self, prefix: &dyn SharedDevicePrefix) -> u64;

    fn commit_shared_prefix(
        &self,
        prefix: &dyn SharedDevicePrefix,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        byte_offset: usize,
    ) -> Result<SharedPrefixCommitInfo, MemoryError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceAllocator, DeviceKey, HostAllocator};
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug, Default)]
    struct VirtualOnly {
        commits: AtomicU64,
        decommits: AtomicU64,
    }

    impl DeviceAllocator for VirtualOnly {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            HostAllocator.allocate(bytes, align)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            // SAFETY: canonical release forwards the exact allocation.
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
        }

        fn device(&self) -> DeviceKey {
            DeviceKey::HOST
        }

        fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
            Some(self)
        }
    }

    impl VirtualBacking for VirtualOnly {
        fn allocate_committed(
            &self,
            bytes: usize,
            align: usize,
            _ranges: &[std::ops::Range<usize>],
        ) -> Result<NonNull<u8>, MemoryError> {
            self.commits.fetch_add(1, Ordering::Relaxed);
            HostAllocator.allocate(bytes, align)
        }

        fn commit_allocation_range(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
            _offset: usize,
            _bytes: usize,
        ) -> Result<(), MemoryError> {
            self.commits.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn mapped_bytes_for_allocation_ranges(
            &self,
            ranges: &[AllocationCommitRange],
        ) -> Result<u64, MemoryError> {
            Ok(ranges.iter().map(|range| range.bytes as u64).sum())
        }

        fn mapped_bytes_for_allocation(
            &self,
            bytes: usize,
            _align: usize,
        ) -> Result<u64, MemoryError> {
            Ok(bytes as u64)
        }

        fn decommit_allocation_range(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
            _offset: usize,
            bytes: usize,
        ) -> Result<u64, MemoryError> {
            self.decommits.fetch_add(1, Ordering::Relaxed);
            Ok(bytes as u64)
        }

        fn allocation_committed_bytes(
            &self,
            _ptr: NonNull<u8>,
            allocation_bytes: usize,
            _align: usize,
        ) -> usize {
            allocation_bytes
        }
    }

    #[derive(Debug)]
    struct SharedOnly;

    impl DeviceAllocator for SharedOnly {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            HostAllocator.allocate(bytes, align)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            // SAFETY: canonical release forwards the exact allocation.
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
        }

        fn device(&self) -> DeviceKey {
            DeviceKey::HOST
        }

        fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
            Some(self)
        }
    }

    impl SharedMapping for SharedOnly {
        fn create_shared_prefix(
            &self,
            bytes: usize,
        ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
            Err(MemoryError::InvalidRequest {
                tier: "host",
                requested: bytes as u64,
                reason: "test shared prefix construction is intentionally unsupported",
            })
        }

        fn incremental_owned_bytes_for_shared_prefix(
            &self,
            prefix: &dyn SharedDevicePrefix,
        ) -> u64 {
            prefix.committed_physical_bytes()
        }

        fn commit_shared_prefix(
            &self,
            _prefix: &dyn SharedDevicePrefix,
            _ptr: NonNull<u8>,
            allocation_bytes: usize,
            _byte_offset: usize,
        ) -> Result<SharedPrefixCommitInfo, MemoryError> {
            Err(MemoryError::InvalidRequest {
                tier: "host",
                requested: allocation_bytes as u64,
                reason: "test shared mapping is intentionally unsupported",
            })
        }
    }

    #[test]
    fn virtual_backing_is_discovered_and_used_before_canonical_release() {
        let allocator = VirtualOnly::default();
        let ordinary: &dyn DeviceAllocator = &allocator;
        let backing = ordinary
            .as_virtual_backing()
            .expect("virtual backing capability");
        let initial = 0..8;
        let ptr = backing
            .allocate_committed(64, 16, std::slice::from_ref(&initial))
            .expect("reserve and commit");
        backing
            .commit_allocation_range(ptr, 64, 16, 8, 8)
            .expect("additional commit");
        assert_eq!(
            backing
                .decommit_allocation_range(ptr, 64, 16, 8, 8)
                .expect("partial decommit"),
            8
        );
        assert_eq!(backing.allocation_committed_bytes(ptr, 64, 16), 64);
        assert_eq!(allocator.commits.load(Ordering::Relaxed), 2);
        assert_eq!(allocator.decommits.load(Ordering::Relaxed), 1);
        // SAFETY: whole-allocation release remains on the ordinary allocator.
        unsafe { ordinary.deallocate(ptr, 64, 16) };
    }

    #[test]
    fn shared_mapping_discovery_is_independent_of_virtual_backing() {
        let allocator: &dyn DeviceAllocator = &SharedOnly;
        assert!(allocator.as_virtual_backing().is_none());
        assert!(allocator.as_shared_mapping().is_some());
    }
}

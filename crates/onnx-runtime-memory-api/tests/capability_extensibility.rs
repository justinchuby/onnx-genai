use std::ptr::NonNull;
use std::sync::Arc;

use onnx_runtime_memory_api::{
    AllocationCommitRange, DeviceAllocator, DeviceKey, HostAllocator, MemoryError,
    SharedDevicePrefix, SharedMapping, SharedPrefixCommitInfo, VirtualBacking,
};

#[derive(Debug)]
struct ThirdPartyVirtual {
    ordinary: HostAllocator,
}

impl DeviceAllocator for ThirdPartyVirtual {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        self.ordinary.allocate(bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        unsafe { self.ordinary.deallocate(ptr, bytes, align) };
    }

    fn device(&self) -> DeviceKey {
        DeviceKey::HOST
    }

    fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
        Some(self)
    }
}

impl VirtualBacking for ThirdPartyVirtual {
    fn allocate_committed(
        &self,
        bytes: usize,
        align: usize,
        _committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<NonNull<u8>, MemoryError> {
        self.ordinary.allocate(bytes, align)
    }

    fn commit_allocation_range(
        &self,
        _ptr: NonNull<u8>,
        _allocation_bytes: usize,
        _align: usize,
        _offset: usize,
        _bytes: usize,
    ) -> Result<(), MemoryError> {
        Ok(())
    }

    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[AllocationCommitRange],
    ) -> Result<u64, MemoryError> {
        Ok(ranges.iter().map(|range| range.bytes as u64).sum())
    }

    fn mapped_bytes_for_allocation(&self, bytes: usize, _align: usize) -> Result<u64, MemoryError> {
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
struct ThirdPartyShared {
    ordinary: HostAllocator,
}

impl DeviceAllocator for ThirdPartyShared {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        self.ordinary.allocate(bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        unsafe { self.ordinary.deallocate(ptr, bytes, align) };
    }

    fn device(&self) -> DeviceKey {
        DeviceKey::HOST
    }

    fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
        Some(self)
    }
}

impl SharedMapping for ThirdPartyShared {
    fn create_shared_prefix(
        &self,
        bytes: usize,
    ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
        Err(MemoryError::InvalidRequest {
            tier: "host",
            requested: bytes as u64,
            reason: "test mechanism does not construct physical prefixes",
        })
    }

    fn incremental_owned_bytes_for_shared_prefix(
        &self,
        prefix: &dyn SharedDevicePrefix,
    ) -> Result<u64, MemoryError> {
        Err(MemoryError::InvalidRequest {
            tier: "host",
            requested: prefix.requested_bytes() as u64,
            reason: "test mechanism rejects foreign prefixes",
        })
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
            reason: "test mechanism rejects foreign prefixes",
        })
    }
}

#[test]
fn third_party_capabilities_are_independently_discoverable_after_erasure() {
    let virtual_allocator: Arc<dyn DeviceAllocator> = Arc::new(ThirdPartyVirtual {
        ordinary: HostAllocator,
    });
    assert!(virtual_allocator.as_virtual_backing().is_some());
    assert!(virtual_allocator.as_shared_mapping().is_none());
    assert!(
        !virtual_allocator.commits_on_demand(),
        "capability presence alone is not a governor-accounting claim"
    );

    let backing = virtual_allocator
        .as_virtual_backing()
        .expect("third-party virtual backing");
    let initial = 0..16;
    let ptr = backing
        .allocate_committed(64, 16, std::slice::from_ref(&initial))
        .expect("third-party virtual allocation");
    unsafe { virtual_allocator.deallocate(ptr, 64, 16) };

    let shared_allocator: Arc<dyn DeviceAllocator> = Arc::new(ThirdPartyShared {
        ordinary: HostAllocator,
    });
    assert!(shared_allocator.as_virtual_backing().is_none());
    assert!(shared_allocator.as_shared_mapping().is_some());
}

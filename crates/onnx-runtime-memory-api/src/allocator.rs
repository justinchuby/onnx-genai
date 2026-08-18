//! Primitive memory types and the ordinary allocator contract.
//!
//! [`DeviceAllocator`] deliberately covers only device identity plus ordinary
//! allocation and terminal release. Lazy virtual backing and shared physical
//! mappings are independent optional capabilities in [`crate::capability`].

use std::any::Any;
use std::fmt::Debug;
use std::ptr::NonNull;

use crate::capability::{SharedMapping, VirtualBacking};
use crate::deferred::{AllocationReleaseOutcome, ReleaseAccounting};
use crate::{MemoryError, Tier};

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
/// A [`Tier`] says how far away memory is; this says which device within that
/// tier. Two CUDA devices are the same tier and different allocators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceKey {
    pub tier: Tier,
    pub index: u32,
}

impl DeviceKey {
    pub const HOST: Self = Self {
        tier: Tier::Host,
        index: 0,
    };

    pub const fn device(index: u32) -> Self {
        Self {
            tier: Tier::Device,
            index,
        }
    }
}

/// A pinned, read-only shared prefix whose physical bytes are owned once and
/// may be mapped into multiple allocations.
pub trait SharedDevicePrefix: Send + Sync + Debug {
    fn device_ptr(&self) -> u64;
    fn committed_physical_bytes(&self) -> u64;
    fn mapped_bytes(&self) -> usize;
    fn requested_bytes(&self) -> usize;

    /// Concrete-prefix recovery for the mapping implementation that created
    /// this opaque handle.
    ///
    /// This is not an allocator/capability identity proof. Capability coherence
    /// is the trusted contract documented on [`DeviceAllocator`].
    fn as_any(&self) -> &dyn Any;
}

/// The accounting outcome of mapping a [`SharedDevicePrefix`] into one
/// allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedPrefixCommitInfo {
    /// Newly owned physical bytes. A valid additional shared mapping reports
    /// zero because the prefix was charged once when created.
    pub additional_owned_bytes: u64,
    /// Newly mapped bytes on the mapped-attribution axis.
    pub newly_mapped_bytes: u64,
    pub granules: usize,
}

/// Somewhere ordinary memory comes from.
///
/// An eager allocator implements only these three required methods. It does
/// not implement degenerate commit/decommit or sharing methods.
///
/// # Safety and coherence contract
///
/// This raw-pointer boundary is trusted. Implementations must return unique,
/// suitably aligned live allocations and must release only their own
/// allocations. A capability returned by [`as_virtual_backing`](Self::as_virtual_backing)
/// or [`as_shared_mapping`](Self::as_shared_mapping) must operate on the same
/// selected mechanism and [`DeviceKey`] as this allocator, and that answer must
/// remain stable for the allocator's lifetime. Transparent wrappers may
/// forward all three interfaces to one coherent inner mechanism.
///
/// Rust does not structurally prove that a hostile wrapper delegates its
/// ordinary allocation, optional capabilities, and release to one inner
/// object. Runtime identity heuristics are not a security boundary here.
/// In-tree unsafe implementations are responsible for satisfying this contract
/// and are tested as coherent units.
///
/// Whole-allocation terminal release always comes back through this trait,
/// including allocations reserved through [`VirtualBacking`]. Optional
/// capability traits never take ownership of terminal release.
pub trait DeviceAllocator: Send + Sync + Debug {
    /// Take `bytes` aligned to `align`.
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError>;

    /// Give back a whole allocation returned by this allocator or by the
    /// [`VirtualBacking`] capability discovered from this allocator.
    ///
    /// This is the pre-Phase-4 **migration adapter**. It cannot report partial
    /// failure, so new code should implement and call
    /// [`release`](Self::release) instead; the default `release` forwards here
    /// so existing implementations keep working unchanged.
    ///
    /// # Safety
    ///
    /// `ptr` must identify one live allocation from this coherent mechanism
    /// with exactly this `bytes` and `align`, and must not be released twice.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize);

    /// Give back a whole allocation and report bytes whose global mapping
    /// reference transitioned to unmapped.
    ///
    /// Eager allocators have no mapped attribution and inherit zero. This
    /// remains part of canonical whole-allocation release; it is intentionally
    /// not a method on either optional capability.
    ///
    /// Like [`deallocate`](Self::deallocate), this is a migration adapter that
    /// cannot express partial failure. Zero is a valid answer here and never
    /// means the release failed.
    ///
    /// # Safety
    ///
    /// The same requirements as [`deallocate`](Self::deallocate).
    unsafe fn deallocate_with_unmapped(&self, ptr: NonNull<u8>, bytes: usize, align: usize) -> u64 {
        // SAFETY: forwarded under this method's identical contract.
        unsafe { self.deallocate(ptr, bytes, align) };
        0
    }

    /// Give back a whole allocation and report a **structured** outcome.
    ///
    /// This is the Phase-4 canonical release entry point. It is additive: the
    /// default implementation is an eager adapter over
    /// [`deallocate_with_unmapped`](Self::deallocate_with_unmapped), so every
    /// existing allocator keeps working unchanged and reports
    /// [`AllocationReleaseOutcome::Complete`].
    ///
    /// # Honesty requirements
    ///
    /// * [`AllocationReleaseOutcome::Complete`] means the whole allocation is
    ///   gone (freed or pooled). Zero unmapped bytes is a valid complete
    ///   result and must never be used to signal failure.
    /// * [`AllocationReleaseOutcome::Failed`] may be returned **only** when
    ///   nothing was mutated. It is the one shape that implies "unchanged".
    /// * Any partial mutation — some granules unmapped, some handles released,
    ///   an error partway through a multi-step teardown — must be
    ///   [`AllocationReleaseOutcome::Quarantined`] carrying the bytes actually
    ///   unmapped and the residual ownership that remains.
    ///
    /// # Safety
    ///
    /// The same requirements as [`deallocate`](Self::deallocate).
    unsafe fn release(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> AllocationReleaseOutcome {
        // SAFETY: forwarded under this method's identical contract.
        let unmapped_bytes = unsafe { self.deallocate_with_unmapped(ptr, bytes, align) };
        AllocationReleaseOutcome::complete(ReleaseAccounting {
            allocation_bytes: bytes as u64,
            unmapped_bytes,
        })
    }

    fn device(&self) -> DeviceKey;

    /// Whether this allocator maps physical memory lazily **and** charges a
    /// governor as each physical commitment is made.
    ///
    /// This is an accounting promise, not capability discovery. An allocator
    /// may expose [`VirtualBacking`] while returning `false` here when its
    /// commit operations are not integrated with a governor. Consumers that
    /// skip an eager full-footprint reservation rely on both halves of this
    /// contract, so `false` is the safe default.
    fn commits_on_demand(&self) -> bool {
        false
    }

    /// Discover lazy reserve/commit/decommit support from this selected
    /// allocator reference.
    fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
        None
    }

    /// Discover shared physical mapping support independently from virtual
    /// backing support.
    fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
        None
    }
}

/// Host memory from the global allocator.
///
/// This is intentionally eager-only: it implements no optional capability.
#[derive(Debug, Default, Clone, Copy)]
pub struct HostAllocator;

impl DeviceAllocator for HostAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        let layout = std::alloc::Layout::from_size_align(bytes.max(1), align).map_err(|_| {
            MemoryError::InvalidRequest {
                tier: Tier::Host.name(),
                requested: bytes as u64,
                reason: "the requested size and alignment are not a valid layout; the alignment \
                         must be a power of two and the rounded size must not overflow",
            }
        })?;
        // SAFETY: `layout` has a non-zero size and valid power-of-two alignment.
        let ptr = unsafe { std::alloc::alloc(layout) };
        NonNull::new(ptr).ok_or_else(|| MemoryError::AllocationFailed {
            tier: Tier::Host.name(),
            requested: bytes as u64,
            reason: String::from(
                "the system allocator refused bytes the governor had granted; the process is \
                 out of address space or the host is out of memory",
            ),
        })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        let Ok(layout) = std::alloc::Layout::from_size_align(bytes.max(1), align) else {
            return;
        };
        // SAFETY: delegated to this method's contract.
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
    }

    fn device(&self) -> DeviceKey {
        DeviceKey::HOST
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct EagerOnly;

    impl DeviceAllocator for EagerOnly {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            HostAllocator.allocate(bytes, align)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            // SAFETY: forwarded unchanged from this method's contract.
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
        }

        fn device(&self) -> DeviceKey {
            DeviceKey::HOST
        }
    }

    #[test]
    fn eager_allocator_requires_only_the_ordinary_contract() {
        let allocator: &dyn DeviceAllocator = &EagerOnly;
        assert!(!allocator.commits_on_demand());
        assert!(allocator.as_virtual_backing().is_none());
        assert!(allocator.as_shared_mapping().is_none());
        let ptr = allocator.allocate(64, 16).expect("ordinary allocation");
        // SAFETY: exact live allocation returned above.
        unsafe { allocator.deallocate(ptr, 64, 16) };
    }

    #[test]
    fn host_allocations_are_aligned_as_requested() {
        for (bytes, align) in [(1usize, 64usize), (100, 64), (4096, 256), (7, 8)] {
            let ptr = HostAllocator.allocate(bytes, align).expect("granted");
            assert_eq!(ptr.as_ptr() as usize % align, 0);
            // SAFETY: exact live allocation returned above.
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
        }
    }

    #[test]
    fn zero_byte_allocation_is_non_null() {
        let ptr = HostAllocator.allocate(0, 64).expect("zero bytes is valid");
        // SAFETY: exact live allocation returned above.
        unsafe { HostAllocator.deallocate(ptr, 0, 64) };
    }

    #[test]
    fn invalid_alignment_is_refused_with_a_reason() {
        let error = HostAllocator
            .allocate(64, 3)
            .expect_err("alignment must be a power of two");
        assert!(error.to_string().contains("power of two"), "{error}");
    }

    #[test]
    fn live_host_allocations_are_distinct_and_writable() {
        let first = HostAllocator.allocate(256, 64).expect("first");
        let second = HostAllocator.allocate(256, 64).expect("second");
        unsafe {
            std::ptr::write_bytes(first.as_ptr(), 0x11, 256);
            std::ptr::write_bytes(second.as_ptr(), 0x22, 256);
            for offset in 0..256 {
                assert_eq!(*first.as_ptr().add(offset), 0x11);
                assert_eq!(*second.as_ptr().add(offset), 0x22);
            }
            HostAllocator.deallocate(first, 256, 64);
            HostAllocator.deallocate(second, 256, 64);
        }
    }

    #[test]
    fn device_keys_distinguish_host_and_accelerators() {
        assert_eq!(HostAllocator.device(), DeviceKey::HOST);
        assert_ne!(DeviceKey::device(0), DeviceKey::device(1));
        assert_eq!(DeviceKey::device(1).tier, Tier::Device);
    }
}

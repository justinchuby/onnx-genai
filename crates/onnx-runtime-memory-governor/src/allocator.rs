//! The allocator seam: raw memory, from wherever the caller says.
//!
//! # Relationship with `onnx-runtime-memory-api`
//!
//! The primitive data types ([`AllocationCommitRange`], [`DeviceKey`],
//! [`MappedAllocation`], [`SharedPrefixCommitInfo`]) and the
//! [`SharedDevicePrefix`] trait are defined in `onnx-runtime-memory-api` and
//! re-exported here for backward compatibility.
//!
//! The [`DeviceAllocator`] trait and [`HostAllocator`] remain in this crate
//! because `DeviceAllocator::allocate_committed_with_capacity` and
//! `DeviceAllocator::commit_allocation_ranges_with_capacity` accept a
//! [`MappedPhysicalCapacityToken`], a governor-specific type coupled to
//! [`LeaseLedger`](crate::LeaseLedger). Splitting those methods into an
//! extension trait requires changing the `&dyn DeviceAllocator` dispatch sites
//! and is deferred to Phase 2 (#1186).

use std::fmt::Debug;
use std::ptr::NonNull;

use crate::{MappedPhysicalCapacityToken, MemoryError, Tier};

// ── Re-exports from onnx-runtime-memory-api ──────────────────────────────
pub use onnx_runtime_memory_api::allocator::{
    AllocationCommitRange, DeviceKey, MappedAllocation, SharedDevicePrefix, SharedPrefixCommitInfo,
};

/// Somewhere raw memory comes from.
///
/// Implement this to substitute your own allocator into either backend. It is
/// deliberately small: allocation, deallocation, and which device the memory is
/// on. Everything else — budgets, roles, pressure — belongs to the governor,
/// which is a separate contract precisely so the two can be replaced
/// independently.
///
/// # Contract
///
/// * `allocate` returns memory aligned to at least `align`, or an error. It must
///   not return a null or misaligned pointer.
/// * `deallocate` is called exactly once per successful `allocate`, with the
///   same `bytes` and `align`. Implementations may rely on that.
/// * `device` is constant for the life of the allocator. Callers use it to
///   decide whether a pointer may be dereferenced on the host, so an allocator
///   that lies here turns a host read into a wild access rather than an error.
/// * **Every method may be called concurrently from any number of threads.**
///   All three take `&self`, and the `Send + Sync` bound is not decoration: one
///   allocator serves every session on its device, so concurrent sessions are
///   the normal case rather than an edge one. An implementation that needs
///   exclusive state must carry its own lock.
/// * Every successful `allocate` owns a region that overlaps no other **live**
///   allocation from this allocator. A region becomes reusable only once its
///   matching `deallocate` has been called. Concurrent calls must behave as
///   though they happened in some sequential order — an implementation whose
///   locking lets two callers be handed the same region would let one session
///   overwrite another's tensors, silently and only under load.
///
/// # This trait does not make memory governed
///
/// It lives in the memory-governor crate, so implementing it reads as "this
/// allocator's memory is on the ledger". It is not. This trait is about *how
/// you obtain* device memory; [`MemoryGovernor`] is about *who is charged* for
/// it, and an implementation of one says nothing about the other.
///
/// That is deliberate. The ledger accounts for standing claims, taken once and
/// held, and charging every `allocate` would put a governor round-trip on a
/// path an execution provider walks constantly. A component with a standing
/// claim should hold a [`MemoryLease`] alongside its allocator rather than
/// expect the allocator to account for it.
///
/// # Governor coupling note
///
/// The [`allocate_committed_with_capacity`] and
/// [`commit_allocation_ranges_with_capacity`] methods accept a
/// [`MappedPhysicalCapacityToken`], which is a governor-specific type. This
/// coupling is identified for extraction in Phase 2 of #1186.
///
/// [`MemoryGovernor`]: crate::MemoryGovernor
/// [`MemoryLease`]: crate::MemoryLease
/// [`allocate_committed_with_capacity`]: DeviceAllocator::allocate_committed_with_capacity
/// [`commit_allocation_ranges_with_capacity`]: DeviceAllocator::commit_allocation_ranges_with_capacity
pub trait DeviceAllocator: Send + Sync + Debug {
    /// Take `bytes` aligned to `align`.
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError>;

    /// Reserve one allocation while committing only the byte ranges the caller
    /// says are live.
    fn allocate_committed(
        &self,
        bytes: usize,
        align: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<NonNull<u8>, MemoryError> {
        let _ = committed_ranges;
        self.allocate(bytes, align)
    }

    fn allocate_committed_with_capacity(
        &self,
        bytes: usize,
        align: usize,
        committed_ranges: &[std::ops::Range<usize>],
        capacity: &mut MappedPhysicalCapacityToken,
    ) -> Result<MappedAllocation<NonNull<u8>>, MemoryError> {
        let _ = capacity;
        let allocation = self.allocate_committed(bytes, align, committed_ranges)?;
        let newly_mapped_bytes = committed_ranges.iter().fold(0_u64, |total, range| {
            total.saturating_add(range.len() as u64)
        });
        Ok(MappedAllocation {
            allocation,
            newly_mapped_bytes,
        })
    }

    /// Ensure `offset..offset + bytes` in an existing allocation is physically
    /// backed.
    fn commit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<(), MemoryError> {
        let _ = (ptr, allocation_bytes, align, offset, bytes);
        Ok(())
    }

    /// Commit several allocation ranges as one allocator transaction.
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

    fn commit_allocation_ranges_with_capacity(
        &self,
        ranges: &[AllocationCommitRange],
        capacity: &mut MappedPhysicalCapacityToken,
    ) -> Result<u64, MemoryError> {
        let _ = capacity;
        self.commit_allocation_ranges(ranges)?;
        self.mapped_bytes_for_allocation_ranges(ranges)
    }

    /// Mapped attribution bytes represented by a batched set of ranges.
    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[AllocationCommitRange],
    ) -> Result<u64, MemoryError> {
        Ok(ranges.iter().fold(0_u64, |total, range| {
            total.saturating_add(range.bytes as u64)
        }))
    }

    /// Mapped bytes required to fully back a new allocation.
    fn mapped_bytes_for_allocation(&self, bytes: usize, align: usize) -> Result<u64, MemoryError> {
        let _ = align;
        Ok(bytes as u64)
    }

    /// Release physical backing from a byte range in an existing allocation
    /// while keeping its virtual address reserved.
    fn decommit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<u64, MemoryError> {
        let _ = (ptr, allocation_bytes, align, offset, bytes);
        Ok(0)
    }

    /// Physical bytes currently claimed by this allocation.
    fn allocation_committed_bytes(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
    ) -> usize {
        let _ = (ptr, align);
        allocation_bytes
    }

    /// Whether this allocator commits physical memory as it is used rather
    /// than when it is requested.
    fn commits_on_demand(&self) -> bool {
        false
    }

    /// Give back memory this allocator returned.
    ///
    /// # Safety
    ///
    /// `ptr` must have come from [`DeviceAllocator::allocate`] on **this**
    /// allocator with exactly this `bytes` and `align`, and must not be
    /// deallocated twice.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize);

    /// Free an allocation and report bytes whose global mapping reference
    /// count transitioned from one to zero.
    ///
    /// # Safety
    ///
    /// `ptr`, `bytes`, and `align` must identify one live allocation returned
    /// by this allocator, exactly as required by [`DeviceAllocator::deallocate`].
    unsafe fn deallocate_with_unmapped(&self, ptr: NonNull<u8>, bytes: usize, align: usize) -> u64 {
        // SAFETY: forwarded under this method's identical contract.
        unsafe { self.deallocate(ptr, bytes, align) };
        0
    }

    /// Create a pinned, read-only shared prefix of `bytes`.
    fn create_shared_prefix(
        &self,
        bytes: usize,
    ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
        Err(MemoryError::InvalidRequest {
            tier: self.device().tier.name(),
            requested: bytes as u64,
            reason: "this allocator does not support shared prefixes; a pinned shared prefix \
                     requires a physical-handle pool (the CUDA VMM arena)",
        })
    }

    /// Estimate the incremental **owned** physical bytes to admit one more
    /// sharer of `prefix`.
    fn incremental_owned_bytes_for_shared_prefix(&self, prefix: &dyn SharedDevicePrefix) -> u64 {
        let _ = prefix;
        0
    }

    /// Map `prefix` into the live allocation `ptr` at `byte_offset`,
    /// **read-only**, taking one reference per shared granule.
    fn commit_shared_prefix(
        &self,
        prefix: &dyn SharedDevicePrefix,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        byte_offset: usize,
    ) -> Result<SharedPrefixCommitInfo, MemoryError> {
        let _ = (prefix, ptr, byte_offset);
        Err(MemoryError::InvalidRequest {
            tier: self.device().tier.name(),
            requested: allocation_bytes as u64,
            reason: "this allocator does not support shared prefixes; a pinned shared prefix \
                     requires a physical-handle pool (the CUDA VMM arena)",
        })
    }

    /// Which device this allocator serves.
    fn device(&self) -> DeviceKey;
}

/// Host memory from the global allocator.
///
/// The default for host tiers, and a deliberately thin one: the system allocator
/// already pools with per-thread caches, so a pool layered on top adds a lock
/// without removing one.
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
        // SAFETY: `layout` has a non-zero size and a valid power-of-two
        // alignment.
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

    #[test]
    fn an_allocator_reports_whether_it_commits_on_demand() {
        #[derive(Debug)]
        struct Stub(bool);

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

            fn commits_on_demand(&self) -> bool {
                self.0
            }
        }

        #[derive(Debug)]
        struct Silent;

        impl DeviceAllocator for Silent {
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

        assert!(
            !Silent.commits_on_demand(),
            "an allocator that says nothing must be treated as committing eagerly"
        );

        let lazy: &dyn DeviceAllocator = &Stub(true);
        let eager: &dyn DeviceAllocator = &Stub(false);
        assert!(lazy.commits_on_demand());
        assert!(!eager.commits_on_demand());
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
    fn a_zero_byte_request_is_not_a_failure() {
        let allocator = HostAllocator;
        let ptr = allocator
            .allocate(0, 64)
            .expect("zero bytes is not an error");
        unsafe { allocator.deallocate(ptr, 0, 64) };
    }

    #[test]
    fn a_bad_alignment_is_refused_with_a_reason() {
        let allocator = HostAllocator;
        let error = allocator
            .allocate(64, 3)
            .expect_err("3 is not a power of two");
        assert!(
            error.to_string().contains("power of two"),
            "the error must say what is wrong with the request, got: {error}"
        );
    }

    #[test]
    fn allocations_are_distinct_and_writable() {
        let allocator = HostAllocator;
        let first = allocator.allocate(256, 64).expect("granted");
        let second = allocator.allocate(256, 64).expect("granted");
        unsafe {
            std::ptr::write_bytes(first.as_ptr(), 0x11, 256);
            std::ptr::write_bytes(second.as_ptr(), 0x22, 256);
            for offset in 0..256 {
                assert_eq!(*first.as_ptr().add(offset), 0x11, "first was clobbered");
                assert_eq!(*second.as_ptr().add(offset), 0x22, "second was clobbered");
            }
            allocator.deallocate(first, 256, 64);
            allocator.deallocate(second, 256, 64);
        }
    }

    #[test]
    fn the_host_allocator_reports_the_host() {
        assert_eq!(HostAllocator.device(), DeviceKey::HOST);
        assert_eq!(DeviceKey::HOST.tier, Tier::Host);
        assert_eq!(DeviceKey::device(1).tier, Tier::Device);
        assert_ne!(DeviceKey::device(0), DeviceKey::device(1));
    }
}

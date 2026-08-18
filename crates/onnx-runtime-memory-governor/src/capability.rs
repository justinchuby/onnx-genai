//! Optional allocator capabilities: [`VirtualBacking`] (lazy commit) and
//! [`SharedMapping`] (shared physical prefixes).
//!
//! # Why these are separate from [`DeviceAllocator`](crate::DeviceAllocator)
//!
//! An ordinary eager allocator (`cuMemAlloc`/`malloc`) allocates and releases
//! memory; it has no notion of committing part of a reservation later, and no
//! notion of a physical handle shared across reservations. Before this split,
//! [`DeviceAllocator`](crate::DeviceAllocator) carried both of those as
//! *optional* methods with "successful no-op" defaults (`commit → Ok(())`,
//! `decommit → Ok(0)`, `create_shared_prefix → an error`). That made every
//! eager allocator answer "yes, committed" and "no bytes released" to questions
//! it had no way to actually reason about, and made discovering whether an
//! allocator *really* supports lazy commit or shared prefixes a matter of
//! calling a method and hoping the default was semantically right.
//!
//! These two capabilities pull that surface out of `DeviceAllocator` entirely.
//! An allocator that has one implements the trait and returns it from
//! [`DeviceAllocator::as_virtual_backing`](crate::DeviceAllocator::as_virtual_backing)
//! /
//! [`DeviceAllocator::as_shared_mapping`](crate::DeviceAllocator::as_shared_mapping);
//! an allocator that does not simply returns `None` (the default), which is an
//! unambiguous "this capability does not exist here" rather than a
//! successful-looking no-op. A caller that needs the capability gets an
//! `Option` to match on, not a method to call and hope was meaningful.
//!
//! # What these capabilities deliberately do *not* expose (Phase 1)
//!
//! Both traits are **forward-and-query only**: allocate-committed, commit a
//! range, and ask how many bytes a commit maps. Neither exposes a
//! `decommit`/refund/release method. That is deliberate, not an oversight:
//!
//! * A release method that reports "bytes whose mapping reached zero" as a bare
//!   integer cannot distinguish a legitimate zero from a failed release, and a
//!   release keyed by a raw pointer races the allocator's own free-list address
//!   reuse. Those are lifecycle hazards a *capability* boundary cannot make
//!   safe on its own.
//! * In this phase the physical backing a capability commits is released only
//!   when the whole allocation is freed through
//!   [`DeviceAllocator::deallocate`](crate::DeviceAllocator::deallocate). Its
//!   lifetime is owned by the allocator that vended it. Partial release,
//!   mapped-byte refunds, and governed decommit remain the concrete allocator's
//!   own concern — reached through its inherent methods by a provider that
//!   constructs it — rather than a trait method this contract would have to
//!   honour for an arbitrary third-party implementation it cannot inspect.
//!
//! # Capability identity
//!
//! [`VirtualBacking::device`] and [`SharedMapping::device`] must equal the
//! owning allocator's
//! [`DeviceAllocator::device`](crate::DeviceAllocator::device). Discovery always
//! goes through the same allocator reference a caller already holds
//! (`allocator.as_virtual_backing()`), so there is no second handle that could
//! go stale or be swapped independently. This contract makes **no** claim of an
//! unforgeable mechanism identity beyond that reported [`DeviceKey`]: a caller
//! that mixes a capability obtained from one allocator with a pointer allocated
//! by another is violating the safety contract of the method it calls, exactly
//! as it would by passing a foreign pointer to
//! [`DeviceAllocator::deallocate`](crate::DeviceAllocator::deallocate), and this
//! layer does not pretend to detect it.

use std::any::Any;
use std::ptr::NonNull;

use crate::MemoryError;
use crate::allocator::{
    AllocationCommitRange, DeviceKey, SharedDevicePrefix, SharedPrefixCommitInfo,
};

/// Lazy commit of physical backing behind a reserved virtual range.
///
/// Implement this only for an allocator that genuinely reserves address space
/// and maps physical memory into it on demand (a CUDA VMM arena). A plain eager
/// allocator has no such notion, so its
/// [`DeviceAllocator::as_virtual_backing`](crate::DeviceAllocator::as_virtual_backing)
/// returns `None` rather than a `VirtualBacking` whose `commit` would be a
/// no-op that reports success it cannot mean.
///
/// # Contract
///
/// * [`device`](Self::device) equals the owning allocator's
///   [`DeviceAllocator::device`](crate::DeviceAllocator::device).
/// * [`allocate_committed`](Self::allocate_committed) reserves one allocation
///   and physically backs exactly the byte ranges the caller names as live.
/// * [`commit_allocation_range`](Self::commit_allocation_range) grows the
///   physical backing of an existing allocation without moving its pointer.
/// * Every fallible method returns [`MemoryError`] on failure — never a sentinel
///   value — so a caller can never confuse "nothing to do" with "it failed".
/// * Every method may be called concurrently from any number of threads on
///   allocations this same capability produced.
/// * This capability exposes **no** decommit/refund/release. Backing it commits
///   is released when the owning allocator frees the allocation through
///   [`DeviceAllocator::deallocate`](crate::DeviceAllocator::deallocate); its
///   lifetime is owned by that allocator (see the module documentation).
pub trait VirtualBacking: Send + Sync {
    /// Which device this capability backs. Must equal the owning allocator's
    /// [`DeviceAllocator::device`](crate::DeviceAllocator::device).
    fn device(&self) -> DeviceKey;

    /// Reserve one allocation while committing only the byte ranges the caller
    /// says are live.
    fn allocate_committed(
        &self,
        bytes: usize,
        align: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<NonNull<u8>, MemoryError>;

    /// Ensure `offset..offset + bytes` in an existing allocation is physically
    /// backed.
    fn commit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<(), MemoryError>;

    /// Commit several allocation ranges as one allocator transaction.
    ///
    /// The default commits each range independently through
    /// [`commit_allocation_range`](Self::commit_allocation_range). An allocator
    /// whose ranges can share physical granules (so committing them together
    /// claims each shared granule only once) should override this to do so
    /// atomically rather than one range at a time.
    fn commit_allocation_ranges(&self, ranges: &[AllocationCommitRange]) -> Result<(), MemoryError> {
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

    /// Mapped attribution bytes represented by a batched set of ranges.
    ///
    /// Required rather than defaulted: summing each range's *requested* bytes is
    /// only a correct estimate for a byte-granular mechanism. A granule-backed
    /// implementation needs an amount rounded up to its own granularity — often
    /// far larger than the sum of what was requested — and only the concrete
    /// implementation knows that granularity, so a shared default here could
    /// silently under-charge admission for any mechanism it does not already
    /// know about. The figure returned must be a conservative
    /// (never-an-underestimate) bound.
    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[AllocationCommitRange],
    ) -> Result<u64, MemoryError>;

    /// Mapped bytes required to fully back a new allocation of `bytes`.
    fn mapped_bytes_for_allocation(&self, bytes: usize, align: usize) -> Result<u64, MemoryError>;

    /// Physical bytes currently committed for this allocation.
    fn allocation_committed_bytes(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
    ) -> usize;

    /// Downcast hook for a caller that needs the concrete implementation (for
    /// instance, to reach a governor-coupled capacity-token commit path or a
    /// concrete decommit that is deliberately not part of this trait).
    fn as_any(&self) -> &dyn Any;
}

/// Shared physical handles mappable, read-only, into many allocations at zero
/// incremental owned cost.
///
/// Implement this only when this allocator can create a physical mapping once
/// and reuse it — a plain eager or VMM allocator without a shared
/// physical-handle pool has no such notion, so its
/// [`DeviceAllocator::as_shared_mapping`](crate::DeviceAllocator::as_shared_mapping)
/// should return `None` rather than a `SharedMapping` whose
/// `create_shared_prefix` would have to fabricate a non-shared allocation and
/// call it shared.
///
/// # Contract
///
/// * [`device`](Self::device) equals the owning allocator's
///   [`DeviceAllocator::device`](crate::DeviceAllocator::device).
/// * [`create_shared_prefix`](Self::create_shared_prefix) charges its physical
///   bytes exactly once, on the owned axis; nothing further is owed to admit
///   additional sharers.
/// * [`incremental_owned_bytes_for_shared_prefix`](Self::incremental_owned_bytes_for_shared_prefix)
///   is `0` only for a prefix this same capability produced **and** could
///   actually map. A prefix from a foreign device or a foreign allocator kind is
///   never free to admit here and must be estimated conservatively (its own
///   reported [`SharedDevicePrefix::committed_physical_bytes`] is always a safe
///   answer), so admission control never trusts a "free" number for a mapping
///   [`commit_shared_prefix`](Self::commit_shared_prefix) is about to refuse.
/// * [`commit_shared_prefix`](Self::commit_shared_prefix) maps `prefix`
///   read-only into a live allocation and never mis-maps: it errors rather than
///   mapping over an already-committed region or mixing devices.
/// * This capability exposes **no** release of a shared mapping. A shared
///   mapping is torn down when the owning allocator frees the allocation
///   through
///   [`DeviceAllocator::deallocate`](crate::DeviceAllocator::deallocate), under
///   the allocator's own lock and keyed by its own live-span table — never
///   through a separate, pointer-addressed capability call a caller could race
///   against free-list address reuse (see the module documentation).
pub trait SharedMapping: Send + Sync {
    /// Which device this capability backs. Must equal the owning allocator's
    /// [`DeviceAllocator::device`](crate::DeviceAllocator::device).
    fn device(&self) -> DeviceKey;

    /// Create a pinned, read-only shared prefix of `bytes`.
    fn create_shared_prefix(&self, bytes: usize)
    -> Result<Box<dyn SharedDevicePrefix>, MemoryError>;

    /// Estimate the incremental **owned** physical bytes to admit one more
    /// sharer of `prefix`.
    fn incremental_owned_bytes_for_shared_prefix(&self, prefix: &dyn SharedDevicePrefix) -> u64;

    /// Map `prefix` into the live allocation `ptr` at `byte_offset`,
    /// **read-only**, taking one reference per shared granule.
    fn commit_shared_prefix(
        &self,
        prefix: &dyn SharedDevicePrefix,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        byte_offset: usize,
    ) -> Result<SharedPrefixCommitInfo, MemoryError>;

    /// Downcast hook for a caller that needs the concrete implementation.
    fn as_any(&self) -> &dyn Any;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::{DeviceAllocator, HostAllocator};
    use std::fmt::Debug;

    /// A minimal allocator that advertises a `VirtualBacking` view of itself,
    /// used to prove the discovery contract's two provable guarantees:
    /// discovery is consistent, and the capability reports the same device.
    #[derive(Debug)]
    struct FakeVmm {
        device: DeviceKey,
    }

    impl VirtualBacking for FakeVmm {
        fn device(&self) -> DeviceKey {
            self.device
        }
        fn allocate_committed(
            &self,
            _bytes: usize,
            _align: usize,
            _committed_ranges: &[std::ops::Range<usize>],
        ) -> Result<NonNull<u8>, MemoryError> {
            Err(MemoryError::InvalidRequest {
                tier: "device",
                requested: 0,
                reason: "test double",
            })
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
            Ok(ranges.iter().map(|r| r.bytes as u64).sum())
        }
        fn mapped_bytes_for_allocation(
            &self,
            bytes: usize,
            _align: usize,
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
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl DeviceAllocator for FakeVmm {
        fn allocate(&self, _bytes: usize, _align: usize) -> Result<NonNull<u8>, MemoryError> {
            Err(MemoryError::InvalidRequest {
                tier: "device",
                requested: 0,
                reason: "test double",
            })
        }
        unsafe fn deallocate(&self, _ptr: NonNull<u8>, _bytes: usize, _align: usize) {}
        fn device(&self) -> DeviceKey {
            self.device
        }
        fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
            Some(self)
        }
    }

    /// GUARANTEE: an advertised capability reports the same device as the
    /// allocator that vended it. This is the only identity claim the contract
    /// makes, and it is checkable.
    #[test]
    fn a_discovered_capability_reports_the_owning_allocators_device() {
        let allocator = FakeVmm {
            device: DeviceKey::device(3),
        };
        let handle: &dyn DeviceAllocator = &allocator;
        let backing = handle
            .as_virtual_backing()
            .expect("this allocator advertises virtual backing");
        assert_eq!(
            backing.device(),
            handle.device(),
            "a discovered capability must report the same device as its allocator"
        );
        assert_eq!(backing.device(), DeviceKey::device(3));
    }

    /// GUARANTEE: discovery is consistent for the life of the allocator — an
    /// eager allocator never advertises a capability it does not have.
    #[test]
    fn an_eager_allocator_advertises_neither_capability() {
        let host: &dyn DeviceAllocator = &HostAllocator;
        assert!(host.as_virtual_backing().is_none());
        assert!(host.as_shared_mapping().is_none());
    }
}

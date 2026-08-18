//! Optional allocator capabilities: [`VirtualBacking`] (lazy commit/decommit)
//! and [`SharedMapping`] (shared physical prefixes).
//!
//! # Why these are separate from [`DeviceAllocator`]
//!
//! An ordinary eager allocator (`cuMemAlloc`/`malloc`) allocates and releases
//! memory; it has no notion of committing part of a reservation later, and no
//! notion of a physical handle shared across reservations. Before Phase 2 of
//! #1186, `DeviceAllocator` carried both of those as *optional* methods with
//! "successful no-op" defaults (commit → `Ok(())`, decommit → `Ok(0)`,
//! `create_shared_prefix` → an error). That made every eager allocator answer
//! "yes, committed" and "no bytes released" to questions it had no way to
//! actually reason about, and made discovering whether an allocator *really*
//! supports lazy commit or shared prefixes a matter of calling a method and
//! hoping the default was semantically right for the situation.
//!
//! These two capabilities pull that surface out of `DeviceAllocator` entirely.
//! An allocator that has one implements the trait and returns it from
//! [`DeviceAllocator::as_virtual_backing`]/
//! [`DeviceAllocator::as_shared_mapping`];
//! an allocator that does not simply returns `None` (the default), which is an
//! unambiguous "this capability does not exist here" rather than a
//! successful-looking no-op. A caller that needs the capability gets an
//! `Option` to match on, not a method to call and hope was meaningful.
//!
//! # Capability identity
//!
//! [`VirtualBacking::device`] and [`SharedMapping::device`] must equal the
//! owning allocator's [`DeviceAllocator::device`].
//! Discovery always goes through the same allocator reference a caller
//! already holds (`allocator.as_virtual_backing()`), so there is no second
//! handle that could go stale or be swapped independently — but a caller that
//! stores the returned `&dyn VirtualBacking`/`&dyn SharedMapping` alongside a
//! *different* allocator's identity (for instance, mixing up two devices, or
//! mixing a capability obtained from one allocator with a pointer allocated by
//! another) can still check `device()` itself before trusting the two belong
//! together.

use std::any::Any;
use std::ptr::NonNull;

use crate::MemoryError;
use crate::allocator::{
    AllocationCommitRange, DeviceAllocator, DeviceKey, SharedDevicePrefix, SharedPrefixCommitInfo,
};

/// Whether `capability`, discovered from `allocator`, is genuinely the same
/// concrete mechanism as `allocator` itself — not a different internal object
/// that merely reports a matching [`DeviceKey`].
///
/// [`DeviceKey`] alone is not strong enough identity to bind a capability to
/// the allocator it came from: two unrelated arenas on the same device
/// compare equal by `DeviceKey`. This compares two things that must **both**
/// agree, neither of which a composing wrapper can produce by accident:
///
/// * the **data pointer** each trait object reference actually points at,
///   discarding the vtable; and
/// * the **concrete type** each reference names, via [`Any::type_id`].
///
/// The data pointer alone is not unforgeable: a `#[repr(C)]` struct shares its
/// own starting address with its first field, so a composing wrapper whose
/// first field is a foreign mechanism and whose `as_virtual_backing`/
/// `as_shared_mapping` returns `Some(&self.first_field)` produces a
/// `capability` reference whose data pointer equals `allocator`'s, even
/// though `self.first_field` is a distinct, foreign object the wrapper merely
/// embeds (#1186 Phase 2 review, round 3 finding 1). Rust guarantees a
/// concrete type's [`TypeId`] is never shared by a different concrete type,
/// so requiring it to also match closes this without needing any allocator to
/// carry a manufactured identity token: the wrapper and the field it embeds
/// are necessarily different Rust types, so their `TypeId`s necessarily
/// differ, regardless of what address either happens to occupy. Every
/// capability implementation this crate ships (`CudaVmmAllocator`'s
/// `as_virtual_backing`/`as_shared_mapping`) returns `Some(self)`, so both
/// checks always hold for them; this exists to let a caller reject, rather
/// than trust, an allocator composed so that its ordinary
/// [`DeviceAllocator::allocate`]/[`DeviceAllocator::deallocate`] and its
/// advertised capability are backed by two different objects — which would
/// let a pointer produced through the capability be freed through a
/// mechanism that never produced it.
///
/// [`TypeId`]: std::any::TypeId
///
/// # Example
///
/// ```
/// use std::any::Any;
/// use onnx_runtime_memory_api::allocator::{DeviceAllocator, DeviceKey, HostAllocator};
/// use onnx_runtime_memory_api::capability::capability_shares_mechanism;
///
/// let allocator = HostAllocator;
/// let allocator_ref: &dyn DeviceAllocator = &allocator;
/// let other: &dyn Any = &allocator; // a different reference to the *same* value
/// assert!(capability_shares_mechanism(allocator_ref, other));
///
/// let unrelated = HostAllocator;
/// let unrelated_ref: &dyn Any = &unrelated;
/// assert!(!capability_shares_mechanism(allocator_ref, unrelated_ref));
/// ```
pub fn capability_shares_mechanism(allocator: &dyn DeviceAllocator, capability: &dyn Any) -> bool {
    let allocator_any: &dyn Any = allocator;
    std::ptr::eq(
        allocator as *const dyn DeviceAllocator as *const (),
        capability as *const dyn Any as *const (),
    ) && allocator_any.type_id() == capability.type_id()
}

/// Lazy commit/decommit over an allocator's own reservations.
///
/// Implement this only when allocation and physical commitment are genuinely
/// separate operations, i.e. an allocator that reserves address space up
/// front and maps physical granules into it as they are actually used (a
/// "virtual memory management", or VMM, allocator). An eager allocator that
/// maps everything at `allocate` time has nothing to implement here: there is
/// no separate physical commit to perform and no on-demand mapping to
/// release, so its [`DeviceAllocator::as_virtual_backing`] should return
/// `None` rather than a `VirtualBacking` whose methods would have to
/// pretend those operations happened.
///
/// # Contract
///
/// * `allocate_committed` reserves one allocation while eagerly mapping only
///   the byte ranges the caller names as immediately live.
/// * `deallocate_committed` releases an allocation `allocate_committed`
///   produced. A caller that obtained a pointer through this capability must
///   release it through **this** method, not through
///   [`DeviceAllocator::deallocate`]/[`DeviceAllocator::deallocate_with_unmapped`]:
///   routing release through the same capability reference that produced the
///   pointer (rather than re-deriving a release path from the owning
///   allocator) is what makes "the mechanism that allocated this is the
///   mechanism that frees it" true by construction instead of by convention —
///   see [`capability_shares_mechanism`] for why `DeviceKey` alone cannot
///   prove that on its own. Reports the number of bytes whose physical
///   mapping actually transitioned from committed to uncommitted, the same
///   accounting [`decommit_allocation_range`](Self::decommit_allocation_range)
///   reports, so a mapped-zone refund is never silently lost the way an
///   always-zero default would lose it.
/// * `commit_allocation_range(s)` maps additional bytes of an existing
///   allocation that were not already committed. Committing a byte range that
///   is already committed is a no-op for that range.
/// * `decommit_allocation_range` releases physical backing from a byte range,
///   returning the number of bytes whose physical mapping actually
///   transitioned from committed to uncommitted (not the number of bytes
///   requested) so a caller can charge exactly what was returned.
/// * `mapped_bytes_for_allocation(_ranges)` is an estimate for admission
///   decisions; the actual amount charged is whatever the corresponding
///   commit call reports. Because a granule-backed implementation can need
///   far more rounded physical bytes than the sum of what was requested,
///   `mapped_bytes_for_allocation_ranges` has no default: only the concrete
///   implementation knows its own granularity, so it must state its own
///   conservative (never-an-underestimate) figure rather than inherit one
///   that is only correct for a byte-granular mechanism.
/// * Granule/page rounding is this allocator's concern: every accounting
///   value returned here is already rounded to whatever physical granularity
///   this allocator backs allocations with.
/// * Every method may be called concurrently from any number of threads on
///   allocations this same capability produced.
///
/// [`DeviceAllocator::as_virtual_backing`]: crate::allocator::DeviceAllocator::as_virtual_backing
/// [`DeviceAllocator::deallocate`]: crate::allocator::DeviceAllocator::deallocate
/// [`DeviceAllocator::deallocate_with_unmapped`]: crate::allocator::DeviceAllocator::deallocate_with_unmapped
pub trait VirtualBacking: Send + Sync {
    /// Which device this capability backs. Must equal the owning allocator's
    /// [`DeviceAllocator::device`].
    fn device(&self) -> DeviceKey;

    /// Reserve one allocation while committing only the byte ranges the
    /// caller says are live.
    fn allocate_committed(
        &self,
        bytes: usize,
        align: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<NonNull<u8>, MemoryError>;

    /// Release an allocation obtained from
    /// [`allocate_committed`](Self::allocate_committed), reporting the number
    /// of bytes whose physical mapping actually transitioned from committed
    /// to uncommitted.
    ///
    /// # Safety
    ///
    /// `ptr`, `allocation_bytes`, and `align` must identify one live
    /// allocation this same capability's `allocate_committed` produced, and
    /// must not be released twice — the same requirements
    /// [`DeviceAllocator::deallocate`] places on its own `ptr`/`bytes`/`align`.
    ///
    /// [`DeviceAllocator::deallocate`]: crate::allocator::DeviceAllocator::deallocate
    unsafe fn deallocate_committed(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
    ) -> u64;

    /// Ensure `offset..offset + bytes` in an existing allocation is
    /// physically backed.
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
    /// [`commit_allocation_range`](Self::commit_allocation_range). An
    /// allocator whose ranges can share physical granules (so committing them
    /// together claims each shared granule only once) should override this
    /// to do so atomically rather than one range at a time.
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
    /// Required rather than defaulted: summing each range's *requested*
    /// bytes is only a correct estimate for a byte-granular mechanism.
    /// A granule-backed implementation needs an amount rounded up to its own
    /// granularity — often far larger than the sum of what was requested —
    /// and only the concrete implementation knows that granularity, so a
    /// shared default here could silently under-charge admission for any
    /// mechanism it does not already know about (#1186 Phase 2 review). The
    /// figure returned must be a conservative (never-an-underestimate) bound;
    /// ranges that would share an already-mapped granule may be reported at
    /// less than the sum of their individual costs, since sharing a granule
    /// does not multiply its bytes.
    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[AllocationCommitRange],
    ) -> Result<u64, MemoryError>;

    /// Mapped bytes required to fully back a new allocation of `bytes`.
    fn mapped_bytes_for_allocation(&self, bytes: usize, align: usize) -> Result<u64, MemoryError>;

    /// Release physical backing from a byte range in an existing allocation
    /// while keeping its virtual address reserved. Returns the number of
    /// bytes whose physical mapping actually transitioned from committed to
    /// uncommitted.
    fn decommit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<u64, MemoryError>;

    /// Physical bytes currently committed for this allocation.
    fn allocation_committed_bytes(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
    ) -> usize;

    /// Downcast hook for a caller that needs the concrete implementation
    /// (for instance, to reach a governor-specific atomic capacity-token
    /// path that is not part of this trait — see
    /// `onnx_runtime_cuda_memory::CudaVmmAllocator`'s inherent
    /// `allocate_committed_with_capacity`).
    fn as_any(&self) -> &dyn Any;
}

/// Shared physical handles mappable, read-only, into many allocations at
/// zero incremental owned cost.
///
/// Implement this only when this allocator can create a physical mapping once
/// and reuse it — a plain eager or VMM allocator without a shared
/// physical-handle pool has no such notion, so its
/// [`DeviceAllocator::as_shared_mapping`] should return `None` rather than a
/// `SharedMapping` whose `create_shared_prefix` would have to fabricate a
/// non-shared allocation and call it shared.
///
/// # Contract
///
/// * `create_shared_prefix` charges its physical bytes exactly once, on the
///   owned axis; nothing further is owed to admit additional sharers.
/// * `incremental_owned_bytes_for_shared_prefix` is `0` only for a prefix
///   this same capability produced **and** could actually map — i.e. one that
///   matches both `device()` and this implementation's own sharing authority
///   (see [`commit_shared_prefix`](Self::commit_shared_prefix)'s device/pool
///   checks). A prefix from a foreign device, a foreign pool authority, or a
///   foreign allocator kind is never free to admit here, even if it happens
///   to downcast to this capability's own prefix type: it must be estimated
///   conservatively (its own reported [`SharedDevicePrefix::committed_physical_bytes`]
///   is always a safe, conservative answer), so admission control never
///   trusts a "free" number for a mapping `commit_shared_prefix` is about to
///   refuse (#1186 Phase 2 review).
/// * `commit_shared_prefix` maps `prefix` read-only into a live allocation
///   and never mis-maps: it errors rather than mapping over an already
///   committed region, mixing devices, or mixing pool authorities.
/// * `allocation_shared_mapped_bytes` reports the shared-mapping bytes a
///   caller must fold into a release, so that mapped cost this capability
///   created is never lost the way an always-zero default would lose it —
///   symmetric with [`VirtualBacking::deallocate_committed`]'s release
///   report, and required for the same reason: `SharedMapping` and
///   `VirtualBacking` are independent capabilities, so a mechanism that
///   implements only this one and not `VirtualBacking` must not be able to
///   inherit [`DeviceAllocator::deallocate_with_unmapped`]'s eager-correct
///   default of zero once it has mapped shared cost into an allocation
///   (#1186 Phase 2 review, round 3 finding 2).
///
/// [`DeviceAllocator::as_shared_mapping`]: crate::allocator::DeviceAllocator::as_shared_mapping
/// [`DeviceAllocator::deallocate_with_unmapped`]: crate::allocator::DeviceAllocator::deallocate_with_unmapped
pub trait SharedMapping: Send + Sync {
    /// Which device this capability backs. Must equal the owning allocator's
    /// [`DeviceAllocator::device`].
    fn device(&self) -> DeviceKey;

    /// Create a pinned, read-only shared prefix of `bytes`.
    fn create_shared_prefix(&self, bytes: usize) -> Result<Box<dyn SharedDevicePrefix>, MemoryError>;

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

    /// Bytes of the allocation at `ptr` currently backed by a mapping this
    /// capability created through [`commit_shared_prefix`](Self::commit_shared_prefix).
    ///
    /// [`DeviceAllocator::deallocate_with_unmapped`]'s default calls this
    /// whenever [`DeviceAllocator::as_shared_mapping`] returns `Some`, so a
    /// mechanism that implements `SharedMapping` alone (not
    /// [`VirtualBacking`]) still reports its mapped-zone refund at release
    /// time instead of silently losing it. A mechanism that also implements
    /// `VirtualBacking` and tracks commitment and sharing together (as
    /// `CudaVmmAllocator` does) releases through
    /// [`VirtualBacking::deallocate_committed`] instead — never through this
    /// default — but must still answer this correctly for a caller that
    /// reaches it directly through `as_shared_mapping`, even if that means
    /// conservatively reporting the whole allocation's committed footprint
    /// when this mechanism does not track "shared" separately from
    /// "privately committed" (#1186 Phase 2 review, round 3 finding 2).
    /// Returns `0` for a `ptr` with no live shared mapping.
    fn allocation_shared_mapped_bytes(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
    ) -> u64;

    /// Downcast hook for a caller that needs the concrete implementation.
    fn as_any(&self) -> &dyn Any;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    use crate::allocator::HostAllocator;

    /// A trivial `VirtualBacking` that is never actually called: it exists
    /// only so its address can be compared, never so its methods run.
    #[derive(Debug, Default)]
    struct Marker;

    impl VirtualBacking for Marker {
        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        fn allocate_committed(
            &self,
            _bytes: usize,
            _align: usize,
            _committed_ranges: &[std::ops::Range<usize>],
        ) -> Result<NonNull<u8>, MemoryError> {
            unimplemented!("test double: never called")
        }

        unsafe fn deallocate_committed(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
        ) -> u64 {
            unimplemented!("test double: never called")
        }

        fn commit_allocation_range(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
            _offset: usize,
            _bytes: usize,
        ) -> Result<(), MemoryError> {
            unimplemented!("test double: never called")
        }

        fn mapped_bytes_for_allocation_ranges(
            &self,
            _ranges: &[AllocationCommitRange],
        ) -> Result<u64, MemoryError> {
            unimplemented!("test double: never called")
        }

        fn mapped_bytes_for_allocation(&self, _bytes: usize, _align: usize) -> Result<u64, MemoryError> {
            unimplemented!("test double: never called")
        }

        fn decommit_allocation_range(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
            _offset: usize,
            _bytes: usize,
        ) -> Result<u64, MemoryError> {
            unimplemented!("test double: never called")
        }

        fn allocation_committed_bytes(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
        ) -> usize {
            unimplemented!("test double: never called")
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Genuinely returns itself as its own `VirtualBacking`, the way a real
    /// combined allocator/capability type does — as opposed to
    /// `OffsetZeroComposite` below, which returns an *embedded* object that
    /// merely shares its address.
    #[derive(Debug, Default)]
    struct HonestSelfReporter;

    impl DeviceAllocator for HonestSelfReporter {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            HostAllocator.allocate(bytes, align)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            // SAFETY: forwarded under this method's identical contract.
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
        }

        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
            Some(self)
        }
    }

    impl VirtualBacking for HonestSelfReporter {
        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        fn allocate_committed(
            &self,
            _bytes: usize,
            _align: usize,
            _committed_ranges: &[std::ops::Range<usize>],
        ) -> Result<NonNull<u8>, MemoryError> {
            unimplemented!("test double: never called")
        }

        unsafe fn deallocate_committed(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
        ) -> u64 {
            unimplemented!("test double: never called")
        }

        fn commit_allocation_range(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
            _offset: usize,
            _bytes: usize,
        ) -> Result<(), MemoryError> {
            unimplemented!("test double: never called")
        }

        fn mapped_bytes_for_allocation_ranges(
            &self,
            _ranges: &[AllocationCommitRange],
        ) -> Result<u64, MemoryError> {
            unimplemented!("test double: never called")
        }

        fn mapped_bytes_for_allocation(&self, _bytes: usize, _align: usize) -> Result<u64, MemoryError> {
            unimplemented!("test double: never called")
        }

        fn decommit_allocation_range(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
            _offset: usize,
            _bytes: usize,
        ) -> Result<u64, MemoryError> {
            unimplemented!("test double: never called")
        }

        fn allocation_committed_bytes(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
        ) -> usize {
            unimplemented!("test double: never called")
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// The exact composite the round-3 review flagged: `#[repr(C)]` puts
    /// `foreign` at byte offset zero, so `&composite as *const _ as *const
    /// ()` and `&composite.foreign as *const _ as *const ()` are the same
    /// address, even though `foreign` is a distinct, embedded object with its
    /// own type — not `composite` itself.
    #[repr(C)]
    #[derive(Debug)]
    struct OffsetZeroComposite {
        foreign: Marker,
        _extra: u64,
    }

    impl DeviceAllocator for OffsetZeroComposite {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            HostAllocator.allocate(bytes, align)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            // SAFETY: forwarded under this method's identical contract.
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
        }

        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
            // Deliberately the embedded first field, not `self`.
            Some(&self.foreign)
        }
    }

    #[test]
    fn genuine_self_reference_still_shares_mechanism() {
        let allocator = HonestSelfReporter;
        let allocator_ref: &dyn DeviceAllocator = &allocator;
        let virtual_backing = allocator_ref.as_virtual_backing().expect("advertised");
        assert!(
            capability_shares_mechanism(allocator_ref, virtual_backing.as_any()),
            "an allocator that honestly returns `Some(self)` (through its own field, reached by \
             address) must still be recognized as sharing its own mechanism"
        );
    }

    #[test]
    fn offset_zero_embedded_capability_is_not_treated_as_the_same_mechanism() {
        let composite = OffsetZeroComposite {
            foreign: Marker,
            _extra: 0,
        };

        // Non-vacuous: prove the address collision this test exists to guard
        // against is real, not a hypothetical.
        let composite_addr = &composite as *const OffsetZeroComposite as *const ();
        let foreign_addr = &composite.foreign as *const Marker as *const ();
        assert!(
            std::ptr::eq(composite_addr, foreign_addr),
            "test setup did not reproduce the offset-zero address collision `#[repr(C)]` is \
             supposed to guarantee"
        );

        let allocator_ref: &dyn DeviceAllocator = &composite;
        let virtual_backing = allocator_ref.as_virtual_backing().expect("advertised");
        assert!(
            !capability_shares_mechanism(allocator_ref, virtual_backing.as_any()),
            "a foreign `VirtualBacking` embedded at a wrapper's offset zero must never be \
             treated as the wrapper's own mechanism just because their addresses coincide \
             (#1186 Phase 2 review, round 3 finding 1)"
        );
    }

    /// A `SharedMapping`-only allocator: no `VirtualBacking` at all, so its
    /// `deallocate_with_unmapped` is the base `DeviceAllocator` default —
    /// proving that default now folds in shared-mapping bytes rather than
    /// silently reporting zero (#1186 Phase 2 review, round 3 finding 2).
    #[derive(Debug)]
    struct FakeSharedPrefix {
        bytes: u64,
    }

    impl SharedDevicePrefix for FakeSharedPrefix {
        fn device_ptr(&self) -> u64 {
            0
        }

        fn committed_physical_bytes(&self) -> u64 {
            self.bytes
        }

        fn mapped_bytes(&self) -> usize {
            self.bytes as usize
        }

        fn requested_bytes(&self) -> usize {
            self.bytes as usize
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[derive(Debug, Default)]
    struct SharedOnlyAllocator {
        /// Address of a live allocation -> shared-mapped bytes currently
        /// mapped into it. Stands in for whatever bookkeeping a real
        /// `SharedMapping`-only mechanism would keep.
        mapped: Mutex<HashMap<usize, u64>>,
    }

    impl DeviceAllocator for SharedOnlyAllocator {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            HostAllocator.allocate(bytes, align)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            // SAFETY: forwarded under this method's identical contract.
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
        }

        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        // No `as_virtual_backing` override: this mechanism has no lazy
        // commit/decommit notion, only shared mappings.
        fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
            Some(self)
        }
    }

    impl SharedMapping for SharedOnlyAllocator {
        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        fn create_shared_prefix(&self, bytes: usize) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
            Ok(Box::new(FakeSharedPrefix { bytes: bytes as u64 }))
        }

        fn incremental_owned_bytes_for_shared_prefix(&self, prefix: &dyn SharedDevicePrefix) -> u64 {
            prefix.committed_physical_bytes()
        }

        fn commit_shared_prefix(
            &self,
            prefix: &dyn SharedDevicePrefix,
            ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _byte_offset: usize,
        ) -> Result<SharedPrefixCommitInfo, MemoryError> {
            let bytes = prefix.mapped_bytes() as u64;
            self.mapped
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(ptr.as_ptr() as usize, bytes);
            Ok(SharedPrefixCommitInfo {
                additional_owned_bytes: 0,
                newly_mapped_bytes: bytes,
                granules: 1,
            })
        }

        fn allocation_shared_mapped_bytes(
            &self,
            ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
        ) -> u64 {
            self.mapped
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&(ptr.as_ptr() as usize))
                .copied()
                .unwrap_or(0)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn shared_mapping_only_allocator_reports_mapped_refund_through_base_default() {
        let allocator = SharedOnlyAllocator::default();
        const BYTES: usize = 4096;
        let ptr = allocator.allocate(BYTES, 8).expect("host allocation");

        let shared_mapping = allocator.as_shared_mapping().expect("advertised");
        let prefix = shared_mapping.create_shared_prefix(BYTES).expect("prefix");
        let commit = shared_mapping
            .commit_shared_prefix(prefix.as_ref(), ptr, BYTES, 0)
            .expect("commit");
        assert_eq!(commit.newly_mapped_bytes, BYTES as u64);

        // `SharedOnlyAllocator` does not override `deallocate_with_unmapped`,
        // so this reaches `DeviceAllocator`'s default — which must consult
        // `as_shared_mapping` rather than silently returning zero.
        // SAFETY: `ptr` came from this allocator's own `allocate` above with
        // matching `bytes`/`align`, and is released exactly once here.
        let unmapped = unsafe { allocator.deallocate_with_unmapped(ptr, BYTES, 8) };
        assert_eq!(
            unmapped, BYTES as u64,
            "the base `deallocate_with_unmapped` default must report the shared-mapping bytes \
             it is about to release, exactly once, not silently drop them to zero (#1186 Phase \
             2 review, round 3 finding 2)"
        );
    }
}

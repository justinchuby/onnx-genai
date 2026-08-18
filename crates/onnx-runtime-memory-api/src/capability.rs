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
//!
//! `DeviceKey` equality alone does not prove a `VirtualBacking`/`SharedMapping`
//! genuinely belongs to a *particular* allocator, since two unrelated
//! mechanisms on the same device compare equal by `DeviceKey`. See
//! [`DeviceMemoryMechanism`] for how a caller that must bind allocation and
//! capability together — so release can never be mixed across mechanisms —
//! gets that guarantee structurally, at construction, rather than by
//! comparing identities at the point of use.

use std::ptr::NonNull;

use crate::MemoryError;
use crate::allocator::{
    AllocationCommitRange, DeviceAllocator, DeviceKey, ReleaseReport, SharedDevicePrefix,
    SharedPrefixCommitInfo,
};

/// One device memory mechanism, bundling its ordinary allocator with
/// whichever optional capabilities it genuinely has — all guaranteed, by
/// construction, to be views of the exact same concrete value.
///
/// # Why this exists
///
/// [`DeviceAllocator::as_virtual_backing`]/[`DeviceAllocator::as_shared_mapping`]
/// are how any `&dyn DeviceAllocator` discovers its own optional
/// capabilities. That works when a type answers honestly, but nothing stops
/// a composing wrapper from advertising a *different*, foreign object's
/// capability as if it were its own — which would let a pointer produced
/// through that capability be released through a mechanism that never
/// produced it. Two earlier designs tried to police that from the outside,
/// by identity: first an `Any` supertrait forcing every allocator to be
/// `'static` (#1186 Phase 2 review, round 4 finding 2), then a self-reported
/// `MechanismId` token compared at the point of use — rejected because an
/// implementation can misreport a foreign object's identity as its own, and
/// because pointer/`TypeId` uniqueness is not actually guaranteed for
/// zero-sized or address-reusing types, and because a legitimate delegating
/// wrapper could be wrongly rejected by an overly strict comparison (#1186
/// Phase 2 review, round 5 finding 4).
///
/// `DeviceMemoryMechanism` sidesteps identity checking entirely. Its
/// constructors are generic over one concrete `T`, and every field is
/// populated by cloning the *same* `Arc<T>` and coercing it to a different
/// trait-object type. There is no way to call
/// [`with_virtual_backing`](Self::with_virtual_backing) with an allocator and
/// a *different* object's `VirtualBacking` — the function only accepts one
/// `Arc<T>` argument, and `T` itself must implement both traits. A
/// mismatched composition is a type error, not a check a caller could get
/// wrong or an implementation could defeat.
///
/// A provider that stores a `DeviceMemoryMechanism` (rather than a bare
/// `Arc<dyn DeviceAllocator>`) resolves exactly one mechanism, once, and every
/// capability it later discovers through this bundle is structurally
/// guaranteed to belong to that same mechanism — see
/// `onnx_runtime_ep_cuda`'s `CudaExecutionProvider::with_memory`.
#[derive(Clone)]
pub struct DeviceMemoryMechanism {
    allocator: std::sync::Arc<dyn DeviceAllocator>,
    virtual_backing: Option<std::sync::Arc<dyn VirtualBacking>>,
    shared_mapping: Option<std::sync::Arc<dyn SharedMapping>>,
}

impl std::fmt::Debug for DeviceMemoryMechanism {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMemoryMechanism")
            .field("allocator", &self.allocator)
            .field("has_virtual_backing", &self.virtual_backing.is_some())
            .field("has_shared_mapping", &self.shared_mapping.is_some())
            .finish()
    }
}

impl DeviceMemoryMechanism {
    /// An eager mechanism with neither optional capability.
    pub fn eager<T: DeviceAllocator + 'static>(allocator: std::sync::Arc<T>) -> Self {
        Self {
            allocator,
            virtual_backing: None,
            shared_mapping: None,
        }
    }

    /// A mechanism with the lazy commit/decommit capability, and nothing
    /// else, structurally guaranteed to be a view of the same value as the
    /// allocator itself.
    pub fn with_virtual_backing<T>(allocator: std::sync::Arc<T>) -> Self
    where
        T: DeviceAllocator + VirtualBacking + 'static,
    {
        let clone_of_allocator = std::sync::Arc::clone(&allocator);
        let virtual_backing: std::sync::Arc<dyn VirtualBacking> = clone_of_allocator;
        Self {
            allocator,
            virtual_backing: Some(virtual_backing),
            shared_mapping: None,
        }
    }

    /// A mechanism with the shared-physical-prefix capability, and nothing
    /// else, structurally guaranteed to be a view of the same value as the
    /// allocator itself.
    pub fn with_shared_mapping<T>(allocator: std::sync::Arc<T>) -> Self
    where
        T: DeviceAllocator + SharedMapping + 'static,
    {
        let clone_of_allocator = std::sync::Arc::clone(&allocator);
        let shared_mapping: std::sync::Arc<dyn SharedMapping> = clone_of_allocator;
        Self {
            allocator,
            virtual_backing: None,
            shared_mapping: Some(shared_mapping),
        }
    }

    /// A mechanism with both optional capabilities, all structurally
    /// guaranteed to be views of the same value as the allocator itself.
    pub fn with_virtual_backing_and_shared_mapping<T>(allocator: std::sync::Arc<T>) -> Self
    where
        T: DeviceAllocator + VirtualBacking + SharedMapping + 'static,
    {
        let clone_for_virtual_backing = std::sync::Arc::clone(&allocator);
        let clone_for_shared_mapping = std::sync::Arc::clone(&allocator);
        let virtual_backing: std::sync::Arc<dyn VirtualBacking> = clone_for_virtual_backing;
        let shared_mapping: std::sync::Arc<dyn SharedMapping> = clone_for_shared_mapping;
        Self {
            allocator,
            virtual_backing: Some(virtual_backing),
            shared_mapping: Some(shared_mapping),
        }
    }
}

impl DeviceAllocator for DeviceMemoryMechanism {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        self.allocator.allocate(bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        // SAFETY: forwarded under this method's identical contract.
        unsafe { self.allocator.deallocate(ptr, bytes, align) };
    }

    fn device(&self) -> DeviceKey {
        self.allocator.device()
    }

    unsafe fn deallocate_with_unmapped(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> Result<ReleaseReport, MemoryError> {
        // Deliberately does *not* forward to `self.allocator.deallocate_with_unmapped`:
        // that would re-enter `T`'s own `as_virtual_backing`/`as_shared_mapping`,
        // which is exactly the self-reported discovery this bundle exists to
        // route around. Going through `Self::as_virtual_backing`/
        // `Self::as_shared_mapping` below instead uses this bundle's own,
        // structurally-guaranteed capability views.
        match self.as_virtual_backing() {
            // SAFETY: forwarded under this method's identical contract.
            Some(virtual_backing) => unsafe {
                virtual_backing.deallocate_committed(ptr, bytes, align)
            },
            None => match self.as_shared_mapping() {
                // SAFETY: forwarded under this method's identical contract.
                Some(shared_mapping) => unsafe {
                    shared_mapping.release_shared_mapping(ptr, bytes, align)
                },
                None => {
                    // SAFETY: forwarded under this method's identical contract.
                    unsafe { self.allocator.deallocate(ptr, bytes, align) };
                    Ok(ReleaseReport::complete(0))
                }
            },
        }
    }

    fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
        self.virtual_backing.as_deref()
    }

    fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
        self.shared_mapping.as_deref()
    }
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
///   mechanism that frees it" true by construction instead of by convention
///   — see [`crate::capability::DeviceMemoryMechanism`] for why a
///   `DeviceKey` match alone cannot prove that. Returns a
///   [`ReleaseReport`] with the number of bytes whose physical mapping
///   actually transitioned from committed to uncommitted, the same
///   accounting [`decommit_allocation_range`](Self::decommit_allocation_range)
///   reports, so a mapped-zone refund is never silently lost the way an
///   always-zero default would lose it — and `complete: false` whenever any
///   part of the release did not finish, so a caller never mistakes a
///   partial release for a done one and never proceeds to reuse or
///   otherwise treat the allocation as gone (#1186 Phase 2 review, round 5
///   findings 2-3).
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
    /// [`allocate_committed`](Self::allocate_committed).
    ///
    /// # Contract
    ///
    /// This is one atomic release transaction, not "unmap, then separately
    /// deallocate": it must be the *entire* release for the identified
    /// allocation. `Ok(report)` covers both a full release
    /// (`report.complete == true`) and a partial one
    /// (`report.complete == false`); `report.unmapped_bytes` is always the
    /// true, final count of bytes this call itself released, whether the
    /// release was full or partial — never discarded, never re-reported by a
    /// later call. `Err` is reserved for a call that touched nothing at all
    /// (an unrecognized pointer, wrong device, or similar precondition
    /// violation).
    ///
    /// When `report.complete` is `false`, the allocation (or its remaining,
    /// unreleased portion) is still live: this virtual address must not be
    /// returned to any free list or handed to a new allocation, and must not
    /// be passed to a plain [`DeviceAllocator::deallocate`]. The exact same
    /// `ptr`/`allocation_bytes`/`align` may be passed to this same method
    /// again to retry the remainder.
    ///
    /// A prior version of this contract returned a bare `u64`, in which `0`
    /// could mean either "nothing was mapped" or "the release failed" —
    /// ambiguity that let a caller's chained, unconditional release proceed
    /// even when nothing had actually been confirmed released (#1186 Phase 2
    /// review, round 5 findings 1-2).
    ///
    /// # Safety
    ///
    /// `ptr`, `allocation_bytes`, and `align` must identify one live
    /// allocation this same capability's `allocate_committed` produced, and
    /// must not be released twice — the same requirements
    /// [`DeviceAllocator::deallocate`] places on its own `ptr`/`bytes`/`align`,
    /// except that a `report.complete == false` outcome from a prior call
    /// leaves the identified allocation live for exactly one purpose: a
    /// retry through this same method.
    ///
    /// [`DeviceAllocator::deallocate`]: crate::allocator::DeviceAllocator::deallocate
    unsafe fn deallocate_committed(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
    ) -> Result<ReleaseReport, MemoryError>;

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
    /// while keeping its virtual address reserved.
    ///
    /// Returns a [`ReleaseReport`] with the same "no `0`-versus-failure
    /// conflation, never a stale or partial-as-if-complete number" contract
    /// as [`deallocate_committed`](Self::deallocate_committed):
    /// `report.unmapped_bytes` is the number of bytes whose physical mapping
    /// actually transitioned from committed to uncommitted in this call
    /// (never the number of bytes requested), and `report.complete` is
    /// `false` whenever any granule in the requested range could not be
    /// released — in that case those specific granules remain committed and
    /// tracked exactly as before this call, safe to retry.
    fn decommit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<ReleaseReport, MemoryError>;

    /// Physical bytes currently committed for this allocation.
    fn allocation_committed_bytes(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
    ) -> usize;
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
/// * `release_shared_mapping` is the **entire** release for an allocation
///   reached through this capability: one atomic transaction that both
///   clears the shared-mapping bookkeeping and gives back whatever backs the
///   allocation, reporting exactly the bytes that transitioned from mapped
///   to unmapped in that single call. It is symmetric with
///   [`VirtualBacking::deallocate_committed`]'s release report, and required
///   for the same reason: `SharedMapping` and `VirtualBacking` are
///   independent capabilities, so a mechanism that implements only this one
///   and not `VirtualBacking` must not be able to inherit
///   [`DeviceAllocator::deallocate_with_unmapped`]'s eager-correct default of
///   zero once it has mapped shared cost into an allocation (#1186 Phase 2
///   review, round 3 finding 2), must not be able to refund the same mapped
///   bytes twice or refund bytes a concurrent or failed release never
///   actually released (round 4 finding 1), and must never leave a window
///   in which the address is available for reuse while a caller still
///   believes the release is theirs to retry, or vice versa (round 5
///   finding 1).
///
/// [`DeviceAllocator::as_shared_mapping`]: crate::allocator::DeviceAllocator::as_shared_mapping
/// [`DeviceAllocator::deallocate_with_unmapped`]: crate::allocator::DeviceAllocator::deallocate_with_unmapped
pub trait SharedMapping: Send + Sync {
    /// Which device this capability backs. Must equal the owning allocator's
    /// [`DeviceAllocator::device`].
    fn device(&self) -> DeviceKey;

    /// Create a pinned, read-only shared prefix of `bytes`.
    fn create_shared_prefix(
        &self,
        bytes: usize,
    ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError>;

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

    /// Atomically release the allocation at `ptr` — clearing its
    /// shared-mapping bookkeeping *and* giving back whatever backs it — as
    /// one mechanism transaction, and report exactly the bytes that
    /// transitioned from mapped to unmapped as a result of **this** call.
    ///
    /// [`DeviceAllocator::deallocate_with_unmapped`]'s default calls this
    /// instead of [`DeviceAllocator::deallocate`] whenever
    /// [`DeviceAllocator::as_shared_mapping`] returns `Some` — never in
    /// addition to it — so a mechanism that implements `SharedMapping` alone
    /// (not [`VirtualBacking`]) reports its mapped-zone refund at release
    /// time instead of silently losing it, and so no caller ever reaches a
    /// freed address through a second, unconditional release (#1186 Phase 2
    /// review, round 5 finding 1). A mechanism that also implements
    /// `VirtualBacking` and tracks commitment and sharing together (as
    /// `CudaVmmAllocator` does) may release through
    /// `VirtualBacking::deallocate_committed` instead — never through the
    /// base `deallocate_with_unmapped` default — but must still answer this
    /// correctly for a caller that reaches it directly through
    /// `as_shared_mapping`, even if that means conservatively reporting the
    /// whole allocation's committed footprint when this mechanism does not
    /// track "shared" separately from "privately committed" (#1186 Phase 2
    /// review, round 3 finding 2).
    ///
    /// # Contract
    ///
    /// This is a release operation, not a query followed by one: the byte
    /// count, the shared-mapping bookkeeping clear, and the underlying
    /// unmap/deallocate must all happen under one mechanism synchronization
    /// boundary (a lock, a compare-and-swap loop, or equivalent), so no
    /// concurrent caller can observe the address as reusable before this
    /// call has fully accounted for and released it, and no concurrent
    /// caller can cause this call to report bytes it did not itself release.
    /// A prior version of this contract queried mapped bytes and released
    /// separately (round 4 finding 1), and a later one queried them inside
    /// this call but still let a subsequent, unconditional
    /// `deallocate` run a second time over the same, now-possibly-reused
    /// address (round 5 finding 1) — both are exactly what "one mechanism
    /// transaction" rules out.
    ///
    /// `Ok(report)` covers both a full release (`report.complete == true`,
    /// the address is now free for reuse) and a partial one
    /// (`report.complete == false`, the allocation's unreleased remainder is
    /// still live and not eligible for reuse). `report.unmapped_bytes` is
    /// always the true, final count of bytes this call itself released — for
    /// a `ptr` whose shared mapping was already fully released by a prior
    /// call, or that never had one, that count is `0` with
    /// `complete: true`. `Err` is reserved for a call that touched nothing
    /// (an unrecognized pointer, wrong device, or similar precondition
    /// violation) — never used to mean "released zero bytes".
    ///
    /// * **Exact once**: a `ptr` whose shared mapping this call has already
    ///   fully released reports `unmapped_bytes: 0, complete: true` on every
    ///   subsequent call — it is never possible to refund the same bytes
    ///   twice.
    /// * **Failure preserves state**: if the underlying release cannot
    ///   complete, this returns `Ok(ReleaseReport::partial(bytes_actually_released))`
    ///   (never treating an untouched remainder as released) and leaves
    ///   everything it did not release exactly as it was, so a caller may
    ///   safely retry with the same `ptr`/`allocation_bytes`/`align`.
    /// * **Concurrency**: multiple threads calling this for the same `ptr`
    ///   must divide the true mapped total exactly once among them (any
    ///   ordering is fine; double-counting or losing bytes is not), and must
    ///   never let one thread observe the address as free while another is
    ///   still mid-release.
    ///
    /// # Safety
    ///
    /// `ptr`, `allocation_bytes`, and `align` must identify one live
    /// allocation this capability (or the allocator it belongs to) produced.
    /// A `report.complete == false` outcome from a prior call leaves the
    /// identified allocation live for exactly one purpose: a retry through
    /// this same method.
    unsafe fn release_shared_mapping(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
    ) -> Result<ReleaseReport, MemoryError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use crate::allocator::HostAllocator;

    /// A `SharedDevicePrefix` test double whose "physical" size is whatever
    /// the test wants to model.
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

    /// A `SharedMapping`-only allocator: no `VirtualBacking` at all, so its
    /// `deallocate_with_unmapped` is the base `DeviceAllocator` default —
    /// proving that default now folds in shared-mapping bytes rather than
    /// silently reporting zero (#1186 Phase 2 review, round 3 finding 2),
    /// and that `release_shared_mapping` is this allocation's *entire*
    /// release — bookkeeping and the underlying free together, exactly once
    /// (#1186 Phase 2 review, round 5 finding 1).
    #[derive(Debug, Default)]
    struct SharedOnlyAllocator {
        /// Addresses of allocations still live through this allocator.
        /// Doubles as the idempotency gate for `release_shared_mapping`: a
        /// second call for the same `ptr` finds nothing here and takes the
        /// safe no-op branch instead of freeing twice.
        live: Mutex<HashSet<usize>>,
        /// Address of a live allocation -> shared-mapped bytes currently
        /// mapped into it.
        mapped: Mutex<HashMap<usize, u64>>,
        /// When `true`, the next `release_shared_mapping` call simulates a
        /// failed release: it reports a partial, zero-byte outcome without
        /// touching `live`/`mapped`, proving state is preserved for a
        /// legitimate retry.
        fail_next_release: AtomicBool,
    }

    impl DeviceAllocator for SharedOnlyAllocator {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            let ptr = HostAllocator.allocate(bytes, align)?;
            self.live
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(ptr.as_ptr() as usize);
            Ok(ptr)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            self.live
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&(ptr.as_ptr() as usize));
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

        fn create_shared_prefix(
            &self,
            bytes: usize,
        ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
            Ok(Box::new(FakeSharedPrefix {
                bytes: bytes as u64,
            }))
        }

        fn incremental_owned_bytes_for_shared_prefix(
            &self,
            prefix: &dyn SharedDevicePrefix,
        ) -> u64 {
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

        unsafe fn release_shared_mapping(
            &self,
            ptr: NonNull<u8>,
            allocation_bytes: usize,
            align: usize,
        ) -> Result<ReleaseReport, MemoryError> {
            if self.fail_next_release.swap(false, Ordering::AcqRel) {
                // Simulated failure: nothing was actually released, and
                // `live`/`mapped` are left untouched so a legitimate retry
                // still finds them.
                return Ok(ReleaseReport::partial(0));
            }
            let key = ptr.as_ptr() as usize;
            // `remove` is the read-and-clear liveness check as one atomic
            // step: a second or concurrent call for the same key finds
            // nothing and takes the branch below instead of freeing again
            // (#1186 Phase 2 review, round 5 finding 1).
            let was_live = self
                .live
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&key);
            if !was_live {
                return Ok(ReleaseReport::complete(0));
            }
            let mapped = self
                .mapped
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&key)
                .unwrap_or(0);
            // SAFETY: `key` was found live and removed atomically above, so
            // exactly one caller reaches this deallocate for a given `ptr`
            // — this call is that allocation's entire release.
            unsafe { HostAllocator.deallocate(ptr, allocation_bytes, align) };
            Ok(ReleaseReport::complete(mapped))
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
        // `as_shared_mapping` rather than silently returning zero, and must
        // treat that call as the allocation's entire release rather than
        // also calling plain `deallocate` afterward.
        // SAFETY: `ptr` came from this allocator's own `allocate` above with
        // matching `bytes`/`align`, and is released exactly once here.
        let report = unsafe { allocator.deallocate_with_unmapped(ptr, BYTES, 8) }
            .expect("a live, correctly-identified allocation must not be rejected");
        assert_eq!(
            report,
            ReleaseReport::complete(BYTES as u64),
            "the base `deallocate_with_unmapped` default must report the shared-mapping bytes \
             it released, exactly once, not silently drop them to zero (#1186 Phase 2 review, \
             round 3 finding 2)"
        );
    }

    /// Calls `release_shared_mapping` directly (still the allocation's
    /// entire release, including the underlying free) to isolate the
    /// release-accounting contract itself: exact-once, failure-preserves-
    /// state, and concurrent-sum correctness (#1186 Phase 2 review, round 4
    /// finding 1; report shape updated for round 5 findings 1-2).
    #[test]
    fn release_shared_mapping_is_exact_once() {
        let allocator = SharedOnlyAllocator::default();
        const BYTES: usize = 4096;
        let ptr = allocator.allocate(BYTES, 8).expect("host allocation");

        let shared_mapping = allocator.as_shared_mapping().expect("advertised");
        let prefix = shared_mapping.create_shared_prefix(BYTES).expect("prefix");
        shared_mapping
            .commit_shared_prefix(prefix.as_ref(), ptr, BYTES, 0)
            .expect("commit");

        // SAFETY: `ptr` identifies a live allocation this capability mapped.
        let first =
            unsafe { shared_mapping.release_shared_mapping(ptr, BYTES, 8) }.expect("first release");
        assert_eq!(
            first,
            ReleaseReport::complete(BYTES as u64),
            "the first release must refund the full mapped amount and complete"
        );

        // SAFETY: calling the release operation again on the same `ptr` is
        // exactly the scenario this test proves is safe: the bookkeeping
        // (and the underlying allocation) were already fully released by
        // the first call, so this must be a safe, zero-effect no-op rather
        // than a second free of already-freed memory.
        let second = unsafe { shared_mapping.release_shared_mapping(ptr, BYTES, 8) }
            .expect("second release");
        assert_eq!(
            second,
            ReleaseReport::complete(0),
            "a second release of an already-fully-released mapping must never refund the same \
             bytes twice, and must not attempt to free the allocation again (#1186 Phase 2 \
             review, round 4 finding 1; round 5 finding 1)"
        );
    }

    #[test]
    fn release_shared_mapping_failure_preserves_state_for_retry() {
        let allocator = SharedOnlyAllocator::default();
        const BYTES: usize = 4096;
        let ptr = allocator.allocate(BYTES, 8).expect("host allocation");

        let shared_mapping = allocator.as_shared_mapping().expect("advertised");
        let prefix = shared_mapping.create_shared_prefix(BYTES).expect("prefix");
        shared_mapping
            .commit_shared_prefix(prefix.as_ref(), ptr, BYTES, 0)
            .expect("commit");

        allocator.fail_next_release.store(true, Ordering::SeqCst);
        // SAFETY: `ptr` identifies a live allocation; a simulated failure
        // must not mutate any release-relevant state, and must not free the
        // allocation.
        let failed = unsafe { shared_mapping.release_shared_mapping(ptr, BYTES, 8) }
            .expect("failed release");
        assert_eq!(
            failed,
            ReleaseReport::partial(0),
            "a failed release must report a partial, zero-byte outcome — never `0` used to mean \
             success, and never treated as license to free the allocation anyway (#1186 Phase 2 \
             review, round 4 finding 1; round 5 finding 2)"
        );

        // SAFETY: the retry below is the legitimate use of a still-live
        // mapping the failed call above must have preserved; it also
        // performs this allocation's actual, one-time free.
        let retried =
            unsafe { shared_mapping.release_shared_mapping(ptr, BYTES, 8) }.expect("retry");
        assert_eq!(
            retried,
            ReleaseReport::complete(BYTES as u64),
            "a retry after a failed release must still see the mapping the failed call did not \
             actually clear, and must be the allocation's real, final release"
        );
    }

    #[test]
    fn concurrent_release_shared_mapping_calls_sum_to_exactly_the_mapped_bytes() {
        let allocator = Arc::new(SharedOnlyAllocator::default());
        const BYTES: usize = 4096;
        let ptr = allocator.allocate(BYTES, 8).expect("host allocation");
        let ptr_addr = ptr.as_ptr() as usize;

        let shared_mapping = allocator.as_shared_mapping().expect("advertised");
        let prefix = shared_mapping.create_shared_prefix(BYTES).expect("prefix");
        shared_mapping
            .commit_shared_prefix(prefix.as_ref(), ptr, BYTES, 0)
            .expect("commit");

        const THREADS: usize = 8;
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let allocator = Arc::clone(&allocator);
                std::thread::spawn(move || {
                    let ptr = NonNull::new(ptr_addr as *mut u8).expect("non-null");
                    let shared_mapping = allocator.as_shared_mapping().expect("advertised");
                    // SAFETY: `ptr` identifies the one live allocation
                    // shared across these threads; only one call is
                    // expected to observe a live mapping and perform the
                    // real release, which is exactly the property this
                    // test proves, and no thread frees anything more than
                    // once.
                    unsafe { shared_mapping.release_shared_mapping(ptr, BYTES, 8) }
                        .expect("release")
                })
            })
            .collect();

        let reports: Vec<ReleaseReport> = handles
            .into_iter()
            .map(|handle| handle.join().expect("thread panicked"))
            .collect();
        let total: u64 = reports.iter().map(|report| report.unmapped_bytes).sum();
        let completions = reports.iter().filter(|report| report.complete).count();
        assert_eq!(
            total, BYTES as u64,
            "concurrent releases of the same mapping must divide the true mapped total exactly \
             once — no double-counting and no lost bytes (#1186 Phase 2 review, round 4 finding \
             1)"
        );
        assert_eq!(
            completions, THREADS,
            "every call reports `complete`: real work happens in exactly one of them, and the \
             rest correctly observe \"already fully released\" as a complete, zero-effect no-op \
             rather than a failure (#1186 Phase 2 review, round 5 finding 1)"
        );
    }

    /// A `SharedMapping`-only allocator whose release can be forced to
    /// release only part of a multi-chunk mapping in one call — standing in
    /// for a real partial CUDA failure (some granules unmap, a later one
    /// does not) — proving the base `deallocate_with_unmapped` default
    /// forwards a partial release report unchanged rather than treating it
    /// as done, or as license to free the allocation anyway (#1186 Phase 2
    /// review, round 5 findings 1-2).
    #[derive(Debug, Default)]
    struct PartialReleaseAllocator {
        live: Mutex<HashSet<usize>>,
        /// Address of a live allocation -> remaining unreleased chunk sizes,
        /// in release order. `release_shared_mapping` releases exactly one
        /// chunk per call and reports `complete: false` while chunks remain.
        chunks: Mutex<HashMap<usize, Vec<u64>>>,
    }

    impl DeviceAllocator for PartialReleaseAllocator {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            let ptr = HostAllocator.allocate(bytes, align)?;
            self.live
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(ptr.as_ptr() as usize);
            Ok(ptr)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            self.live
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&(ptr.as_ptr() as usize));
            // SAFETY: forwarded under this method's identical contract.
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
        }

        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
            Some(self)
        }
    }

    impl SharedMapping for PartialReleaseAllocator {
        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        fn create_shared_prefix(
            &self,
            bytes: usize,
        ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
            Ok(Box::new(FakeSharedPrefix {
                bytes: bytes as u64,
            }))
        }

        fn incremental_owned_bytes_for_shared_prefix(
            &self,
            prefix: &dyn SharedDevicePrefix,
        ) -> u64 {
            prefix.committed_physical_bytes()
        }

        fn commit_shared_prefix(
            &self,
            prefix: &dyn SharedDevicePrefix,
            ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _byte_offset: usize,
        ) -> Result<SharedPrefixCommitInfo, MemoryError> {
            let total = prefix.mapped_bytes() as u64;
            let half = total / 2;
            self.chunks
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(ptr.as_ptr() as usize, vec![total - half, half]);
            Ok(SharedPrefixCommitInfo {
                additional_owned_bytes: 0,
                newly_mapped_bytes: total,
                granules: 2,
            })
        }

        unsafe fn release_shared_mapping(
            &self,
            ptr: NonNull<u8>,
            allocation_bytes: usize,
            align: usize,
        ) -> Result<ReleaseReport, MemoryError> {
            let key = ptr.as_ptr() as usize;
            let mut guard = self
                .chunks
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(remaining) = guard.get_mut(&key) else {
                return Ok(ReleaseReport::complete(0));
            };
            // Releases at most one chunk per call, standing in for one
            // granule (or one batch of granules) genuinely unmapping per
            // underlying CUDA call.
            let released = remaining.pop().unwrap_or(0);
            let complete = remaining.is_empty();
            if complete {
                guard.remove(&key);
                drop(guard);
                let was_live = self
                    .live
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&key);
                if was_live {
                    // SAFETY: this is the last chunk; the allocation's
                    // entire release completes in this call.
                    unsafe { HostAllocator.deallocate(ptr, allocation_bytes, align) };
                }
            }
            Ok(ReleaseReport {
                unmapped_bytes: released,
                complete,
            })
        }
    }

    #[test]
    fn partial_shared_mapping_release_through_the_default_path_never_falls_through_to_deallocate() {
        let allocator = PartialReleaseAllocator::default();
        const BYTES: usize = 4096;
        let ptr = allocator.allocate(BYTES, 8).expect("host allocation");

        let shared_mapping = allocator.as_shared_mapping().expect("advertised");
        let prefix = shared_mapping.create_shared_prefix(BYTES).expect("prefix");
        shared_mapping
            .commit_shared_prefix(prefix.as_ref(), ptr, BYTES, 0)
            .expect("commit");

        // SAFETY: `ptr` came from this allocator's own `allocate` above; the
        // partial outcome below leaves it live, so nothing here double-frees.
        let first = unsafe { allocator.deallocate_with_unmapped(ptr, BYTES, 8) }
            .expect("a partial release is not a precondition violation");
        assert!(
            !first.complete,
            "one chunk out of two releasing must be reported as incomplete, not silently \
             treated as done (#1186 Phase 2 review, round 5 finding 2)"
        );

        // Prove the allocation is genuinely still live: a base default that
        // (incorrectly) fell through to plain `deallocate` after a partial
        // release would make this a use-after-free under Miri/ASan; writing
        // through it here is the cheapest available proof without those
        // tools.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0xAB, BYTES) };

        // SAFETY: retries the remainder through the same public path; `ptr`
        // is still the one live allocation from above.
        let second = unsafe { allocator.deallocate_with_unmapped(ptr, BYTES, 8) }
            .expect("retry after partial release");
        assert!(second.complete, "the retry must complete the release");
        assert_eq!(
            first.unmapped_bytes + second.unmapped_bytes,
            BYTES as u64,
            "the two partial reports together must account for exactly the mapped total — \
             neither call may lose or double-count bytes"
        );
        // No further `deallocate` call: `second` already performed the
        // allocation's one and only real free.
    }

    /// A minimal mechanism implementing every capability at once, standing
    /// in for a real combined type like `CudaVmmAllocator`. Used to prove
    /// [`DeviceMemoryMechanism`]'s constructors bind every capability view
    /// to the exact same concrete value structurally, not by a runtime
    /// identity check a caller or implementor could get wrong (#1186 Phase
    /// 2 review, round 5 finding 4).
    #[derive(Debug, Default)]
    struct CombinedMechanism {
        committed: Mutex<HashMap<usize, usize>>,
    }

    impl DeviceAllocator for CombinedMechanism {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            let ptr = HostAllocator.allocate(bytes, align)?;
            self.committed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(ptr.as_ptr() as usize, bytes);
            Ok(ptr)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            self.committed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&(ptr.as_ptr() as usize));
            // SAFETY: forwarded under this method's identical contract.
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
        }

        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }
    }

    impl VirtualBacking for CombinedMechanism {
        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        fn allocate_committed(
            &self,
            bytes: usize,
            align: usize,
            _committed_ranges: &[std::ops::Range<usize>],
        ) -> Result<NonNull<u8>, MemoryError> {
            DeviceAllocator::allocate(self, bytes, align)
        }

        unsafe fn deallocate_committed(
            &self,
            ptr: NonNull<u8>,
            allocation_bytes: usize,
            align: usize,
        ) -> Result<ReleaseReport, MemoryError> {
            // SAFETY: forwarded under this method's identical contract.
            unsafe { DeviceAllocator::deallocate(self, ptr, allocation_bytes, align) };
            Ok(ReleaseReport::complete(allocation_bytes as u64))
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
            _bytes: usize,
        ) -> Result<ReleaseReport, MemoryError> {
            Ok(ReleaseReport::complete(0))
        }

        fn allocation_committed_bytes(
            &self,
            ptr: NonNull<u8>,
            allocation_bytes: usize,
            _align: usize,
        ) -> usize {
            self.committed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&(ptr.as_ptr() as usize))
                .copied()
                .unwrap_or(allocation_bytes)
        }
    }

    impl SharedMapping for CombinedMechanism {
        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        fn create_shared_prefix(
            &self,
            bytes: usize,
        ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
            Ok(Box::new(FakeSharedPrefix {
                bytes: bytes as u64,
            }))
        }

        fn incremental_owned_bytes_for_shared_prefix(
            &self,
            _prefix: &dyn SharedDevicePrefix,
        ) -> u64 {
            0
        }

        fn commit_shared_prefix(
            &self,
            prefix: &dyn SharedDevicePrefix,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _byte_offset: usize,
        ) -> Result<SharedPrefixCommitInfo, MemoryError> {
            Ok(SharedPrefixCommitInfo {
                additional_owned_bytes: 0,
                newly_mapped_bytes: prefix.mapped_bytes() as u64,
                granules: 1,
            })
        }

        unsafe fn release_shared_mapping(
            &self,
            ptr: NonNull<u8>,
            allocation_bytes: usize,
            align: usize,
        ) -> Result<ReleaseReport, MemoryError> {
            // SAFETY: forwarded under this method's identical contract.
            unsafe { DeviceAllocator::deallocate(self, ptr, allocation_bytes, align) };
            Ok(ReleaseReport::complete(0))
        }
    }

    #[test]
    fn device_memory_mechanism_eager_has_no_optional_capability() {
        let mechanism = DeviceMemoryMechanism::eager(Arc::new(HostAllocator));
        assert!(mechanism.as_virtual_backing().is_none());
        assert!(mechanism.as_shared_mapping().is_none());

        let ptr = mechanism.allocate(64, 8).expect("granted");
        // SAFETY: `ptr` came from this mechanism's own `allocate` above.
        let report =
            unsafe { mechanism.deallocate_with_unmapped(ptr, 64, 8) }.expect("eager release");
        assert_eq!(
            report,
            ReleaseReport::complete(0),
            "an eager mechanism with neither optional capability must report the trivial, \
             unambiguous complete-zero release"
        );
    }

    #[test]
    fn device_memory_mechanism_binds_every_capability_view_to_the_same_object() {
        let concrete = Arc::new(CombinedMechanism::default());
        let concrete_addr = Arc::as_ptr(&concrete) as *const ();
        let mechanism =
            DeviceMemoryMechanism::with_virtual_backing_and_shared_mapping(Arc::clone(&concrete));

        let virtual_backing = mechanism.as_virtual_backing().expect("advertised");
        let shared_mapping = mechanism.as_shared_mapping().expect("advertised");
        assert_eq!(
            (virtual_backing as *const dyn VirtualBacking).cast::<()>(),
            concrete_addr,
            "the bundle's VirtualBacking view must be the exact same value the allocator is, \
             guaranteed by construction rather than by a runtime identity check a caller or \
             implementor could get wrong (#1186 Phase 2 review, round 5 finding 4)"
        );
        assert_eq!(
            (shared_mapping as *const dyn SharedMapping).cast::<()>(),
            concrete_addr,
            "the bundle's SharedMapping view must likewise be the same value"
        );
    }

    #[test]
    fn device_memory_mechanism_with_only_one_capability_leaves_the_other_none() {
        let concrete = Arc::new(CombinedMechanism::default());
        // `concrete` genuinely implements `SharedMapping` too, but this
        // bundle is constructed with only `with_virtual_backing`: the other
        // capability must not leak through just because the concrete type
        // happens to support it. A provider's chosen construction is the
        // only source of truth for what the bundle exposes.
        let mechanism = DeviceMemoryMechanism::with_virtual_backing(concrete);
        assert!(mechanism.as_virtual_backing().is_some());
        assert!(
            mechanism.as_shared_mapping().is_none(),
            "a bundle built through `with_virtual_backing` must not also expose `SharedMapping` \
             just because the concrete type happens to implement it"
        );
    }

    /// A delegating wrapper around `CombinedMechanism`, standing in for a
    /// real decorator (metrics, logging, retry) that forwards every call to
    /// an inner mechanism. Implements every trait itself — rather than the
    /// bundle being handed the inner value directly — to prove composition
    /// works correctly for a genuine wrapper, not only for a mechanism that
    /// happens to be the innermost concrete type (#1186 Phase 2 review,
    /// round 5 finding 4: "legitimate wrappers can be wrongly rejected"
    /// under the old identity-check design; under this design there is
    /// nothing to reject).
    #[derive(Debug, Default)]
    struct DelegatingWrapper {
        inner: CombinedMechanism,
    }

    impl DeviceAllocator for DelegatingWrapper {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            self.inner.allocate(bytes, align)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            // SAFETY: forwarded under this method's identical contract.
            unsafe { self.inner.deallocate(ptr, bytes, align) }
        }

        fn device(&self) -> DeviceKey {
            DeviceAllocator::device(&self.inner)
        }
    }

    impl VirtualBacking for DelegatingWrapper {
        fn device(&self) -> DeviceKey {
            VirtualBacking::device(&self.inner)
        }

        fn allocate_committed(
            &self,
            bytes: usize,
            align: usize,
            committed_ranges: &[std::ops::Range<usize>],
        ) -> Result<NonNull<u8>, MemoryError> {
            self.inner
                .allocate_committed(bytes, align, committed_ranges)
        }

        unsafe fn deallocate_committed(
            &self,
            ptr: NonNull<u8>,
            allocation_bytes: usize,
            align: usize,
        ) -> Result<ReleaseReport, MemoryError> {
            // SAFETY: forwarded under this method's identical contract.
            unsafe {
                self.inner
                    .deallocate_committed(ptr, allocation_bytes, align)
            }
        }

        fn commit_allocation_range(
            &self,
            ptr: NonNull<u8>,
            allocation_bytes: usize,
            align: usize,
            offset: usize,
            bytes: usize,
        ) -> Result<(), MemoryError> {
            self.inner
                .commit_allocation_range(ptr, allocation_bytes, align, offset, bytes)
        }

        fn mapped_bytes_for_allocation_ranges(
            &self,
            ranges: &[AllocationCommitRange],
        ) -> Result<u64, MemoryError> {
            self.inner.mapped_bytes_for_allocation_ranges(ranges)
        }

        fn mapped_bytes_for_allocation(
            &self,
            bytes: usize,
            align: usize,
        ) -> Result<u64, MemoryError> {
            self.inner.mapped_bytes_for_allocation(bytes, align)
        }

        fn decommit_allocation_range(
            &self,
            ptr: NonNull<u8>,
            allocation_bytes: usize,
            align: usize,
            offset: usize,
            bytes: usize,
        ) -> Result<ReleaseReport, MemoryError> {
            self.inner
                .decommit_allocation_range(ptr, allocation_bytes, align, offset, bytes)
        }

        fn allocation_committed_bytes(
            &self,
            ptr: NonNull<u8>,
            allocation_bytes: usize,
            align: usize,
        ) -> usize {
            self.inner
                .allocation_committed_bytes(ptr, allocation_bytes, align)
        }
    }

    #[test]
    fn device_memory_mechanism_accepts_a_transparent_delegating_wrapper() {
        let wrapper = Arc::new(DelegatingWrapper::default());
        let mechanism = DeviceMemoryMechanism::with_virtual_backing(Arc::clone(&wrapper));

        let ptr = mechanism.allocate(64, 8).expect("granted");
        let virtual_backing = mechanism.as_virtual_backing().expect("advertised");
        // SAFETY: `ptr` came from this mechanism's own `allocate` above.
        let report = unsafe { virtual_backing.deallocate_committed(ptr, 64, 8) }.expect("release");
        assert!(
            report.complete && report.unmapped_bytes == 64,
            "a genuine delegating wrapper must compose through the bundle exactly like a \
             non-wrapped mechanism would"
        );
    }

    /// An arena a `DeviceAllocator` implementation can borrow from without
    /// owning: proves `DeviceAllocator` itself imposes no `'static`
    /// requirement (#1186 Phase 2 review, round 4 finding 2).
    #[derive(Debug, Default)]
    struct BorrowedArena {
        allocated: std::cell::UnsafeCell<usize>,
    }

    // SAFETY: `allocated` is only ever touched from `BorrowedAllocator`
    // methods, which take `&self` and mutate it through a private, test-only
    // path never called concurrently in this test; this exists solely so
    // `&'a BorrowedArena` satisfies `DeviceAllocator`'s `Sync` bound.
    unsafe impl Sync for BorrowedArena {}

    /// A `DeviceAllocator` that borrows its backing arena rather than owning
    /// it — a shape a supertrait imposing `'static` (whether `Any`, as
    /// round 3 tried, or a `MechanismId`-bearing scheme requiring `Any`)
    /// would reject, since `&'a BorrowedArena` is not `'static` for any `'a`
    /// shorter than `'static`. `DeviceMemoryMechanism` requires `'static`
    /// only on its own constructors — never on the base `DeviceAllocator`
    /// trait itself — so this type still implements the minimal ordinary
    /// contract directly.
    #[derive(Debug)]
    struct BorrowedAllocator<'a> {
        arena: &'a BorrowedArena,
    }

    impl<'a> DeviceAllocator for BorrowedAllocator<'a> {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            // SAFETY: `self.arena` outlives this call by construction (`'a`).
            unsafe {
                *self.arena.allocated.get() += bytes;
            }
            HostAllocator.allocate(bytes, align)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            // SAFETY: forwarded under this method's identical contract.
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
            // SAFETY: `self.arena` outlives this call by construction (`'a`).
            unsafe {
                *self.arena.allocated.get() -= bytes;
            }
        }

        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        // Deliberately does not override `as_virtual_backing` or
        // `as_shared_mapping`: a borrowed, non-`'static` allocator is
        // exactly the kind of type that should be able to implement only
        // the minimal ordinary contract and nothing more.
    }

    #[test]
    fn a_non_static_borrowed_allocator_still_implements_the_minimal_contract() {
        let arena = BorrowedArena::default();
        let allocator = BorrowedAllocator { arena: &arena };

        // This line is the actual proof: `DeviceAllocator` imposes no `Any`
        // or other `'static` bound, so a trait object over a non-`'static`
        // reference compiles.
        let as_dyn: &dyn DeviceAllocator = &allocator;

        let ptr = as_dyn.allocate(64, 8).expect("borrowed allocation");
        assert_eq!(
            // SAFETY: single-threaded test, no concurrent access to `arena`.
            unsafe { *arena.allocated.get() },
            64,
            "allocate must still run through the borrowed arena"
        );
        assert!(
            as_dyn.as_virtual_backing().is_none() && as_dyn.as_shared_mapping().is_none(),
            "an allocator that never overrides either optional capability method must report \
             `None` for both, not be forced to fabricate an identity or a capability (#1186 \
             Phase 2 review, round 4 finding 2)"
        );

        // SAFETY: `ptr` came from this allocator's own `allocate` above with
        // matching `bytes`/`align`, and is released exactly once here.
        unsafe { as_dyn.deallocate(ptr, 64, 8) };
        assert_eq!(
            // SAFETY: single-threaded test, no concurrent access to `arena`.
            unsafe { *arena.allocated.get() },
            0,
            "deallocate must still run through the borrowed arena"
        );
    }
}

//! Primitive data types and the ordinary allocator contract shared by every
//! allocator mechanism.
//!
//! [`DeviceAllocator`] is deliberately minimal: device identity plus ordinary
//! allocate/release. Lazy commit/decommit and shared physical prefixes are
//! separate, optional capabilities — see [`crate::capability`] — discovered
//! through [`DeviceAllocator::as_virtual_backing`] and
//! [`DeviceAllocator::as_shared_mapping`] rather than called directly, so an
//! allocator that does not have a capability is never asked to answer for it.

use std::any::Any;
use std::fmt::Debug;
use std::ptr::NonNull;

use crate::capability::{SharedMapping, VirtualBacking};
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

/// The outcome of one release attempt through
/// [`DeviceAllocator::deallocate_with_unmapped`],
/// [`VirtualBacking::deallocate_committed`], or
/// [`SharedMapping::release_shared_mapping`].
///
/// # Why a struct instead of a bare byte count
///
/// A prior design (#1186 Phase 2 review, rounds 1-4) returned a bare `u64`:
/// the number of bytes released. That conflated two situations a caller must
/// tell apart — "nothing was mapped here" and "the release did not fully
/// complete" — under the same `0`, and gave a partial CUDA failure (some
/// granules genuinely unmapped, a later one failing) nowhere to report the
/// bytes that *did* release without discarding them (#1186 Phase 2 review,
/// round 5 finding 2).
///
/// `unmapped_bytes` is always the true, final count of bytes this call
/// itself caused to transition from committed/mapped to released — it is
/// reported once, by the call that caused the transition, and never
/// reported again by a later call for the same bytes. `complete` says
/// whether *every* byte this call was asked to release actually did; when it
/// is `false`, the allocation (or the remaining, unreleased portion of it)
/// is still live and must not be handed to a plain
/// [`DeviceAllocator::deallocate`] or reused for a new allocation. The exact
/// same `ptr`/`bytes`/`align` may be passed to the same method again to
/// retry the remainder — see each method's own documentation for what
/// "remainder" means for that mechanism.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReleaseReport {
    /// Physical bytes this call itself released. Never a stale or
    /// previously reported count; never bytes another caller's concurrent
    /// call already accounted for.
    pub unmapped_bytes: u64,
    /// `true` if this call fully released everything it was asked to;
    /// `false` if part of the release did not complete and the allocation
    /// (or its remainder) is still live and retryable.
    pub complete: bool,
}

impl ReleaseReport {
    /// A release that fully completed, reporting `unmapped_bytes` released.
    pub const fn complete(unmapped_bytes: u64) -> Self {
        Self {
            unmapped_bytes,
            complete: true,
        }
    }

    /// A release that only partly completed: `unmapped_bytes` genuinely
    /// released, with a remainder still live and safe to retry through the
    /// same call.
    pub const fn partial(unmapped_bytes: u64) -> Self {
        Self {
            unmapped_bytes,
            complete: false,
        }
    }
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
/// sequence with
/// `DeviceAllocator::commit_shared_prefix`.
/// It is deliberately opaque: the concrete backing (CUDA VMM physical handles
/// today) lives in the allocator crate, downcast through
/// [`SharedDevicePrefix::as_any`] by the allocator that produced it.
pub trait SharedDevicePrefix: Send + Sync + Debug {
    /// Device address of the owner's writable window. The prefix content is
    /// filled here **once**, before it is shared read-only into sequences.
    fn device_ptr(&self) -> u64;

    /// Physical device bytes this prefix owns — charged **once**, on the owned
    /// axis, however many sequences share it.
    fn committed_physical_bytes(&self) -> u64;

    /// The granule-rounded byte length the prefix actually spans.
    fn mapped_bytes(&self) -> usize;

    /// Bytes requested at construction, before granule rounding.
    fn requested_bytes(&self) -> usize;

    /// Downcast hook: the allocator that produced this handle recovers its
    /// concrete type to map it.
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
    /// Physical bytes newly *mapped* into this allocation's reservation — one
    /// mapping of already-owned physical memory, reported on the mapped axis.
    pub newly_mapped_bytes: u64,
    /// Granules mapped read-only into the allocation.
    pub granules: usize,
}

/// Somewhere ordinary memory comes from.
///
/// Implement this to substitute your own allocator into either backend. It is
/// deliberately small: allocation, deallocation, and which device the memory
/// is on. Everything else — budgets, roles, pressure — belongs to the
/// governor, which is a separate contract precisely so the two can be
/// replaced independently. Lazy commit/decommit and shared physical prefixes
/// are separate, optional capabilities (see [`crate::capability`]) rather
/// than methods on this trait, so this trait never has to answer for a
/// capability an implementation does not have.
///
/// # Contract
///
/// * `allocate` returns memory aligned to at least `align`, or an error. It
///   must not return a null or misaligned pointer.
/// * `deallocate` is called exactly once per successful `allocate`, with the
///   same `bytes` and `align`. Implementations may rely on that.
/// * `device` is constant for the life of the allocator. Callers use it to
///   decide whether a pointer may be dereferenced on the host, so an
///   allocator that lies here turns a host read into a wild access rather
///   than an error.
/// * **Every method may be called concurrently from any number of threads.**
///   All take `&self`, and the `Send + Sync` bound is not decoration: one
///   allocator serves every session on its device, so concurrent sessions are
///   the normal case rather than an edge one. An implementation that needs
///   exclusive state must carry its own lock.
/// * Every successful `allocate` owns a region that overlaps no other **live**
///   allocation from this allocator. A region becomes reusable only once its
///   matching `deallocate` has been called. Concurrent calls must behave as
///   though they happened in some sequential order — an implementation whose
///   locking lets two callers be handed the same region would let one session
///   overwrite another's tensors, silently and only under load.
/// * `as_virtual_backing`/`as_shared_mapping` must consistently return the
///   same answer (`Some`/`None`) for the life of the allocator, and any
///   `VirtualBacking`/`SharedMapping` returned must report the same
///   [`DeviceKey`] as `device()`. A caller that discovers a capability
///   through one of these methods never has to re-check for staleness: there
///   is no separate handle that could have been swapped out from under it.
/// * An allocator whose own [`deallocate_with_unmapped`](Self::deallocate_with_unmapped)
///   override folds `SharedMapping` accounting into its return value some
///   other way (for instance because commit and sharing are tracked in one
///   structure, as `CudaVmmAllocator` does) must still implement
///   [`SharedMapping::release_shared_mapping`] correctly: anything
///   reachable through `as_shared_mapping` is expected to answer the same
///   question consistently, regardless of which path a caller takes to it.
///
/// # This trait does not make memory governed
///
/// It is used from the memory-governor crate, so implementing it can read as
/// "this allocator's memory is on the ledger". It is not. This trait is about
/// *how you obtain* device memory; a `MemoryGovernor` is about *who is
/// charged* for it, and an implementation of one says nothing about the
/// other.
///
/// That is deliberate. The ledger accounts for standing claims, taken once
/// and held, and charging every `allocate` would put a governor round-trip on
/// a path an execution provider walks constantly. A component with a standing
/// claim should hold a lease alongside its allocator rather than expect the
/// allocator to account for it.
///
/// # Why this does not require [`Any`] or an identity token
///
/// Two earlier designs tried to bind a capability to the allocator that
/// advertises it from the outside: first an `Any` supertrait (rejected
/// because `Any: 'static` transitively forced every implementation to be
/// `'static`, #1186 Phase 2 review, round 4 finding 2), then a self-reported
/// `mechanism_id` identity token compared by the caller (rejected because an
/// implementation can misreport a foreign object's identity as its own, and
/// because address/`TypeId` uniqueness itself is not guaranteed for
/// zero-sized or address-reusing types — #1186 Phase 2 review, round 5
/// finding 4).
///
/// [`crate::capability::DeviceMemoryMechanism`] replaces both: it does not
/// ask any `&self` method to *report* an identity a caller must trust. It
/// binds allocation, virtual backing, and shared mapping structurally, at
/// construction, by requiring the exact same concrete, `Arc`-owned value to
/// implement every capability its constructor names — so a mismatched
/// composition is a type error, not a runtime check that a malicious or
/// merely buggy implementation could defeat.
pub trait DeviceAllocator: Send + Sync + Debug {
    /// Take `bytes` aligned to `align`.
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError>;

    /// Give back memory this allocator returned.
    ///
    /// # Safety
    ///
    /// `ptr` must have come from [`DeviceAllocator::allocate`] on **this**
    /// allocator with exactly this `bytes` and `align`, and must not be
    /// deallocated twice.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize);

    /// Which device this allocator serves.
    fn device(&self) -> DeviceKey;

    /// Free an allocation and report the outcome, including bytes whose
    /// global mapping reference count transitioned from one to zero.
    ///
    /// The default forwards to [`deallocate`](Self::deallocate) and reports
    /// [`ReleaseReport::complete(0)`](ReleaseReport::complete) — the
    /// unambiguously correct answer for any allocator with neither optional
    /// capability, since nothing here is ever committed lazily or shared, so
    /// nothing here can transition a shared mapping's reference count.
    ///
    /// If [`as_shared_mapping`](Self::as_shared_mapping) reports a
    /// capability, the default releases through
    /// [`SharedMapping::release_shared_mapping`] **instead of** calling
    /// [`deallocate`](Self::deallocate) at all: `release_shared_mapping` is
    /// specified to be the sole, atomic release action for an allocation
    /// this capability produced, so chaining a second, unconditional
    /// `deallocate` afterward would let a concurrent allocation that reused
    /// the same address — legitimate the instant `release_shared_mapping`
    /// gives it back — be torn down by that second call (#1186 Phase 2
    /// review, round 5 finding 1). It has no [`VirtualBacking`] equivalent
    /// here on purpose: an allocator with that capability instead releases
    /// through [`VirtualBacking::deallocate_committed`], never through this
    /// default — see that method's documentation.
    ///
    /// When `release_shared_mapping` reports
    /// [`ReleaseReport::partial`] (round 5 finding 2), this default
    /// returns that report unchanged rather than treating a partial release
    /// as done: the allocation is still live, and the same
    /// `ptr`/`bytes`/`align` may be retried through this same method.
    ///
    /// # Safety
    ///
    /// `ptr`, `bytes`, and `align` must identify one live allocation returned
    /// by this allocator, exactly as required by [`DeviceAllocator::deallocate`].
    /// If a prior call through this method (or through
    /// [`SharedMapping::release_shared_mapping`] directly) reported
    /// `complete: false`, the same triple identifies the still-live
    /// remainder and may be passed again; it must not be passed to
    /// [`deallocate`](Self::deallocate) while any part of the release
    /// remains incomplete.
    unsafe fn deallocate_with_unmapped(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> Result<ReleaseReport, MemoryError> {
        match self.as_shared_mapping() {
            // SAFETY: forwarded under this method's identical contract. This
            // call is the entire release: it must not be followed by
            // `deallocate` (see this method's own documentation).
            Some(shared_mapping) => unsafe {
                shared_mapping.release_shared_mapping(ptr, bytes, align)
            },
            None => {
                // SAFETY: forwarded under this method's identical contract.
                unsafe { self.deallocate(ptr, bytes, align) };
                Ok(ReleaseReport::complete(0))
            }
        }
    }

    /// This allocator's lazy commit/decommit capability, if it has one.
    ///
    /// The default `None` is the correct, unambiguous answer for an eager
    /// allocator: it maps everything at `allocate` time, so it has no
    /// separate commit/decommit operation to offer, not a degenerate one.
    fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
        None
    }

    /// This allocator's shared-physical-prefix capability, if it has one.
    ///
    /// The default `None` is the correct, unambiguous answer for an
    /// allocator with no shared physical-handle pool: it cannot produce a
    /// prefix mappable at zero incremental owned cost, so it has nothing to
    /// offer here rather than a degenerate always-fails implementation.
    ///
    /// An override must answer per-*instance*, not per-*type*: a type that
    /// implements [`SharedMapping`] for some of its instances (for example a
    /// VMM arena built with a production physical-handle pool) but not others
    /// (the same type built without one, or bound to a foreign pool
    /// authority) must return `None` for the instances that lack the
    /// resource the capability actually needs, even though the trait is
    /// implemented on the type. Returning `Some` unconditionally whenever the
    /// type implements the trait — regardless of whether *this* instance can
    /// actually honor a call — reintroduces the same "successful no-op"
    /// ambiguity this split exists to remove, just one level up: discovery
    /// says "yes", and the first real call says "no" (#1186 Phase 2 review).
    fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
        None
    }
}

/// Host memory from the global allocator.
///
/// The default for host tiers, and a deliberately thin one: the system
/// allocator already pools with per-thread caches, so a pool layered on top
/// adds a lock without removing one. Has neither optional capability: host
/// allocations are always eager and never shared through this allocator.
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
    fn device_key_basics() {
        assert_eq!(DeviceKey::HOST.tier, Tier::Host);
        assert_eq!(DeviceKey::device(1).tier, Tier::Device);
        assert_ne!(DeviceKey::device(0), DeviceKey::device(1));
    }

    #[test]
    fn an_allocator_with_no_capability_reports_none_by_default() {
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

        let allocator: &dyn DeviceAllocator = &Silent;
        assert!(
            allocator.as_virtual_backing().is_none(),
            "an allocator that says nothing must be treated as having no virtual-backing \
             capability, not a degenerate eager one"
        );
        assert!(
            allocator.as_shared_mapping().is_none(),
            "an allocator that says nothing must be treated as having no shared-mapping \
             capability, not a degenerate always-fails one"
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

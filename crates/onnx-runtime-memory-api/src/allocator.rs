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
/// # Why this does not require [`Any`]
///
/// An earlier design (#1186 Phase 2 review, round 3) added `Any` as a
/// supertrait here so a self-reported mechanism identity could be compared
/// across a `DeviceAllocator` and its `VirtualBacking`/`SharedMapping`
/// capabilities. `Any: 'static`, so that supertrait transitively forced
/// every implementation to be `'static` — a real regression against
/// "minimal ordinary contract", since it rejected an otherwise valid
/// allocator built over borrowed, non-`'static` data (#1186 Phase 2 review,
/// round 4 finding 2). A later design (round 4) replaced `Any` with a plain,
/// `'static`-free `mechanism_id()` method instead, but coordinator review
/// (round 6) found that a self-reported identity — no matter how it is
/// carried — cannot prove what it was introduced to prove: a type can
/// honestly return `Some(self)` from both `as_virtual_backing()` and
/// whatever identity accessor exists, while its `VirtualBacking` method
/// bodies silently operate on unrelated state (see the module docs on
/// [`capability`](crate::capability) for the full accepted trust boundary).
/// Since no version of the identity check actually closed that gap, this
/// trait carries no identity method at all: it only ever bought a false
/// sense of safety at the cost of a real `'static` regression, so removing
/// it is a strict simplification, not a lost guarantee. A caller that needs
/// this class of proof needs the binding/provenance work in #1186 Phase 3,
/// not something layered onto this trait.
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

    /// Free an allocation and report bytes whose global mapping reference
    /// count transitioned from one to zero.
    ///
    /// The default forwards to [`deallocate`](Self::deallocate) and then, if
    /// [`as_shared_mapping`](Self::as_shared_mapping) reports a capability,
    /// releases and reports how many of `ptr`'s bytes were backed by a
    /// shared-prefix mapping — an allocator that can create mapped cost
    /// through [`SharedMapping::commit_shared_prefix`] must never lose that
    /// refund by silently inheriting a plain zero the way an eager allocator
    /// correctly does (#1186 Phase 2 review, round 3 finding 2), and must
    /// never report it from a stale snapshot taken before the actual release
    /// (#1186 Phase 2 review, round 4 finding 1) — see
    /// [`SharedMapping::release_shared_mapping`] for exactly what is and is
    /// not guaranteed about that release. It has no [`VirtualBacking`]
    /// equivalent here on purpose: an allocator with that capability instead
    /// releases through [`VirtualBacking::deallocate_committed`], never
    /// through this default — see that method's documentation.
    ///
    /// This is the unambiguously correct answer for any allocator with
    /// neither capability: nothing here is ever committed lazily or shared,
    /// so nothing here can transition a shared mapping's reference count.
    ///
    /// # Safety
    ///
    /// `ptr`, `bytes`, and `align` must identify one live allocation returned
    /// by this allocator, exactly as required by [`DeviceAllocator::deallocate`].
    unsafe fn deallocate_with_unmapped(&self, ptr: NonNull<u8>, bytes: usize, align: usize) -> u64 {
        let unmapped = match self.as_shared_mapping() {
            // SAFETY: forwarded under this method's identical contract.
            Some(shared_mapping) => unsafe { shared_mapping.release_shared_mapping(ptr, bytes, align) },
            None => 0,
        };
        // SAFETY: forwarded under this method's identical contract.
        unsafe { self.deallocate(ptr, bytes, align) };
        unmapped
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

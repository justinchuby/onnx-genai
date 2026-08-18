//! The allocator seam: raw memory, from wherever the caller says.
//!
//! # Why this is separate from [`MemoryGovernor`](crate::MemoryGovernor)
//!
//! A governor decides *whether* bytes may be taken. An allocator decides *where
//! they come from*. Those are different questions with different answers per
//! device, and conflating them means a caller who wants to supply one has to
//! supply both.
//!
//! # Why it lives in this crate
//!
//! This crate has no dependencies, and both backends already depend on it. The
//! ONNX Runtime binding does not depend on the native execution-provider API,
//! nor the reverse — so an allocator contract defined in either one could not be
//! shared. Defined here, a single implementation serves both:
//!
//! ```text
//!                   ┌──────────────────────┐
//!   user supplies → │   dyn DeviceAllocator │ ← we supply HostAllocator
//!                   └──────────┬───────────┘
//!                     ┌────────┴────────┐
//!            ORT      │                 │      native
//!    OrtAllocator vtable          ExecutionProvider::allocate
//! ```
//!
//! The alternative is writing every allocator twice — and the one that matters
//! is a CUDA arena, which is not a thing to write twice.
//!
//! # Raw, deliberately
//!
//! The signatures are pointers and sizes rather than a buffer type, because the
//! two backends have *different* buffer types: ONNX Runtime's `Alloc` returns a
//! bare `void*`, and the native side wraps allocations in a `DeviceBuffer` that
//! carries device, size, alignment and ownership. Raw is what both can express;
//! each side wraps it in its own richer type on the way out.

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
/// sequence with [`SharedMapping::commit_shared_prefix`]. It is deliberately
/// opaque: the concrete backing (CUDA VMM physical handles today) lives in the
/// allocator crate, downcast through [`SharedDevicePrefix::as_any`] by the
/// allocator that produced it. Detection (hashing) and copy-on-write at
/// divergence are **not** part of this contract — a shared prefix is read-only
/// for the union lifetime of its sharers.
pub trait SharedDevicePrefix: Send + Sync + Debug {
    /// Device address of the owner's writable window. The prefix content is
    /// filled here **once**, before it is shared read-only into sequences.
    fn device_ptr(&self) -> u64;

    /// Physical device bytes this prefix owns — charged **once**, on the owned
    /// axis, however many sequences share it. This is the reported *physical*
    /// cost, never nominal content bytes.
    fn committed_physical_bytes(&self) -> u64;

    /// The granule-rounded byte length the prefix actually spans.
    fn mapped_bytes(&self) -> usize;

    /// Bytes requested at construction, before granule rounding.
    fn requested_bytes(&self) -> usize;

    /// Downcast hook: the allocator that produced this handle recovers its
    /// concrete type to map it. A prefix presented to a different allocator is
    /// refused rather than mis-mapped.
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

/// Somewhere raw memory comes from.
///
/// Implement this to substitute your own allocator into either backend. It is
/// deliberately small: allocation, deallocation, and which device the memory is
/// on. Everything else — budgets, roles, pressure — belongs to the governor,
/// which is a separate contract precisely so the two can be replaced
/// independently.
///
/// Lazy commit/decommit and shared physical prefixes are **not** methods on
/// this trait. They are separate, optional capabilities — see
/// [`VirtualBacking`] and [`SharedMapping`] — discovered through
/// [`as_virtual_backing`](DeviceAllocator::as_virtual_backing) and
/// [`as_shared_mapping`](DeviceAllocator::as_shared_mapping), which default to
/// `None`. An allocator that does not have a capability is therefore never
/// asked to answer for it: the older design gave every allocator
/// "successful no-op" defaults (`commit → Ok(())`, `decommit → Ok(0)`), so an
/// eager allocator with no virtual-memory mechanism answered "yes, committed"
/// to a question it had no way to reason about. Returning `None` from the
/// discovery methods is the honest alternative — "this capability does not
/// exist here" rather than a successful-looking lie.
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
/// * [`as_virtual_backing`](Self::as_virtual_backing) and
///   [`as_shared_mapping`](Self::as_shared_mapping) must return the same answer
///   (`Some`/`None`) for the life of the allocator, and any capability returned
///   must report the same [`DeviceKey`] as [`device`](Self::device). Discovery
///   always goes through the same `&self` the caller already holds, so there is
///   no second handle that could go stale or be swapped independently.
///
/// # Lifetime of a discovered capability (Phase 1 limitation)
///
/// A capability discovered here lets a caller **commit** backing and **query**
/// mapping, but it exposes no `decommit`/refund/release method of its own. In
/// this phase the physical backing a capability commits is released only when
/// the allocation is freed through [`deallocate`](Self::deallocate) — its
/// lifetime is owned by the allocator that vended it. Partial release,
/// mapped-byte refunds, and governed decommit are the concrete allocator's own
/// concern (reached through its inherent methods by a provider that constructs
/// it) and are deliberately **not** part of this contract, rather than being
/// asserted through a trait method the contract could not honour for an
/// arbitrary implementation.
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
/// [`MemoryGovernor`]: crate::MemoryGovernor
/// [`MemoryLease`]: crate::MemoryLease
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

    /// This allocator's lazy commit/decommit capability, if it has one.
    ///
    /// The default `None` is the correct, unambiguous answer for an eager
    /// allocator: it maps everything at `allocate` time, so it has no separate
    /// commit operation to offer, not a degenerate one. An override must answer
    /// per-*instance*, not per-*type*: a type that implements [`VirtualBacking`]
    /// for some of its instances but not others (the same type built without the
    /// resource the capability needs) must return `None` for the instances that
    /// cannot honor a call, so discovery never says "yes" to a capability the
    /// first real call would refuse.
    fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
        None
    }

    /// This allocator's shared-physical-prefix capability, if it has one.
    ///
    /// The default `None` is the correct, unambiguous answer for an allocator
    /// with no shared physical-handle pool: it cannot produce a prefix mappable
    /// at zero incremental owned cost, so it has nothing to offer here rather
    /// than a degenerate always-fails implementation. As with
    /// [`as_virtual_backing`](Self::as_virtual_backing), an override must answer
    /// per-*instance*.
    fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
        None
    }
}

/// Host memory from the global allocator.
///
/// The default for host tiers, and a deliberately thin one: the system allocator
/// already pools with per-thread caches, so a pool layered on top adds a lock
/// without removing one. Measured, an arena over this was slower than this.
///
/// Device memory is the opposite case — `cudaMalloc` is a synchronising driver
/// call in the microseconds with no thread cache — so a device implementation of
/// this trait will need an arena. That is why the trait exists rather than this
/// being hard-coded.
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
            // Unreachable for a pointer this allocator produced, since the same
            // layout was valid on the way in. Leaking beats freeing with a
            // layout that does not match.
            return;
        };
        // SAFETY: delegated to this method's contract -- the pointer came from
        // `allocate` with this exact layout.
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
    }

    fn device(&self) -> DeviceKey {
        DeviceKey::HOST
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An allocator that has no virtual-memory mechanism reports `None` from
    /// both capability-discovery methods rather than a "successful no-op".
    ///
    /// This is the guarantee the capability split exists to make honest: an
    /// eager allocator says "I do not have this capability", not "yes, I
    /// committed". A consumer therefore gets an `Option` to match on instead of
    /// a method to call and hope the default was meaningful.
    #[test]
    fn an_allocator_with_no_capability_reports_none() {
        #[derive(Debug)]
        struct Silent;

        // SAFETY: never allocates, so the non-overlap and validity guarantees
        // hold vacuously.
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

        // Through a trait object, which is how every real consumer holds one.
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

    /// The host allocator honours the alignment it is asked for, whatever the
    /// size. Kernels are entitled to assume it.
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
            // SAFETY: exactly what allocate returned.
            unsafe { allocator.deallocate(ptr, bytes, align) };
        }
    }

    /// A zero-byte request still yields a usable, non-null pointer.
    ///
    /// `std::alloc` rejects a zero-sized layout, so this has to be handled
    /// rather than passed through. Returning null would be indistinguishable
    /// from failure at every call site.
    #[test]
    fn a_zero_byte_request_is_not_a_failure() {
        let allocator = HostAllocator;
        let ptr = allocator
            .allocate(0, 64)
            .expect("zero bytes is not an error");
        // SAFETY: as returned.
        unsafe { allocator.deallocate(ptr, 0, 64) };
    }

    /// An impossible alignment is refused rather than panicking inside
    /// `Layout`.
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

    /// Memory is writable for its whole extent, and two allocations do not
    /// overlap.
    #[test]
    fn allocations_are_distinct_and_writable() {
        let allocator = HostAllocator;
        let first = allocator.allocate(256, 64).expect("granted");
        let second = allocator.allocate(256, 64).expect("granted");
        // SAFETY: both are live allocations of 256 bytes.
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

    /// Host memory says it is host memory. Callers decide whether a pointer may
    /// be dereferenced on the CPU from this.
    #[test]
    fn the_host_allocator_reports_the_host() {
        assert_eq!(HostAllocator.device(), DeviceKey::HOST);
        assert_eq!(DeviceKey::HOST.tier, Tier::Host);
        assert_eq!(DeviceKey::device(1).tier, Tier::Device);
        assert_ne!(DeviceKey::device(0), DeviceKey::device(1));
    }
}

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

use std::fmt::Debug;
use std::ptr::NonNull;

use crate::{MemoryError, Tier};

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
        NonNull::new(ptr).ok_or(MemoryError::InvalidRequest {
            tier: Tier::Host.name(),
            requested: bytes as u64,
            reason: "the system allocator refused bytes the governor had granted; the process \
                     is out of address space or the host is out of memory",
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
        let ptr = allocator.allocate(0, 64).expect("zero bytes is not an error");
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

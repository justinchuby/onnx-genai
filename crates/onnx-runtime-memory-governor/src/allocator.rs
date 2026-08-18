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

use crate::{MappedPhysicalCapacityToken, MemoryError, Tier};

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
/// sequence with [`DeviceAllocator::commit_shared_prefix`]. It is deliberately
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
/// [`MemoryGovernor`]: crate::MemoryGovernor
/// [`MemoryLease`]: crate::MemoryLease
pub trait DeviceAllocator: Send + Sync + Debug {
    /// Take `bytes` aligned to `align`.
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError>;

    /// Reserve one allocation while committing only the byte ranges the caller
    /// says are live.
    ///
    /// Eager allocators cannot separate those two acts, so the default keeps
    /// the old contract and commits the whole allocation. Lazy allocators may
    /// override this and pair it with [`DeviceAllocator::commit_allocation_range`].
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
    ///
    /// The default is a no-op because [`DeviceAllocator::allocate`] and the
    /// default [`DeviceAllocator::allocate_committed`] already committed the
    /// whole allocation. Lazy allocators override this to grow the physical
    /// commitment without moving the pointer.
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
    ///
    /// Lazy allocators override this to union shared physical granules under a
    /// single lock. The eager/default implementation preserves compatibility.
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
    ///
    /// The default is a no-op because eager allocators cannot partially unmap
    /// allocations. Lazy allocators may override this so callers can roll back
    /// a failed multi-buffer growth without leaving committed bytes charged.
    /// Returns the physical bytes whose global mapping reference reached zero.
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
    ///
    /// Eager allocators return the allocation length because every byte is
    /// backed from birth. Lazy allocators override this so tests and profilers
    /// can assert on bytes that are attributable to one binding rather than on
    /// process-global activity from unrelated workspaces.
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
    ///
    /// # Why a consumer needs to know
    ///
    /// A component holding memory whose size it knows only as a worst case --
    /// a KV cache sized at the model's full context, say -- has to choose
    /// between two bad options when the allocator commits eagerly. Reserve the
    /// worst case and refuse models that a short conversation would never grow
    /// into; or reserve nothing and discover the shortfall at an allocation
    /// mid-generation.
    ///
    /// When the allocator commits on demand the choice goes away: memory is
    /// charged as it is genuinely taken, so the worst case becomes a *ceiling
    /// to check* rather than a claim to hold. On a small machine that is the
    /// difference between "this model does not fit" and "this model fits until
    /// it actually needs the memory".
    ///
    /// # Why it lives here and not on an execution provider
    ///
    /// The allocator is the thing that commits, and it is the piece every
    /// backend has. Asking a provider would answer only for the paths that go
    /// through one; asking the allocator answers for ONNX Runtime too, whose
    /// `OrtAllocator` seam wraps one of these.
    ///
    /// # Contract
    ///
    /// `false` is the safe answer and the default: a consumer that believes
    /// this will under-reserve. Return `true` only when the allocator really
    /// does map physical memory lazily **and** charges a governor as it does
    /// so. Saying `true` without the second half turns an accounting question
    /// into an out-of-memory crash.
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
    /// Eager allocators have no shared-granule attribution and return zero.
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

    /// Create a pinned, read-only shared prefix of `bytes`, charged **once** on
    /// the owned axis, for mapping into many allocations with
    /// [`commit_shared_prefix`](DeviceAllocator::commit_shared_prefix).
    ///
    /// The default refuses: a shared prefix is defined by physical-handle
    /// identity across reservations, which only an allocator that owns its
    /// physical granules (a CUDA VMM arena) can provide. Eager allocators — and
    /// any allocator without a physical-handle pool — return an error here
    /// rather than mis-map, so an unsupported request faults loudly at the seam
    /// instead of silently producing private copies.
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
    /// sharer of `prefix` — zero for an allocator that already owns the prefix's
    /// granules.
    ///
    /// This is the admission-facing statement (#745): the shared bytes are
    /// charged once, so the Nth request costs only its private continuation.
    fn incremental_owned_bytes_for_shared_prefix(&self, prefix: &dyn SharedDevicePrefix) -> u64 {
        let _ = prefix;
        0
    }

    /// Map `prefix` into the live allocation `ptr` at `byte_offset`,
    /// **read-only**, taking one reference per shared granule.
    ///
    /// The prefix's physical memory is already owned, so this maps existing
    /// memory: it charges **zero** incremental owned bytes and keeps the shared
    /// granules alive until the last sharer (or the prefix owner) leaves. The
    /// default refuses for the same reason [`create_shared_prefix`] does.
    ///
    /// [`create_shared_prefix`]: DeviceAllocator::create_shared_prefix
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

    /// `commits_on_demand` is opaque: a caller asks the allocator, not the
    /// backend.
    ///
    /// It lives on this trait rather than on an execution provider because the
    /// allocator is the thing that commits, and it is the piece every backend
    /// has. `GovernedAllocator` forwards it, so a session running on ONNX
    /// Runtime answers the same question the native path does -- which is what
    /// lets a consumer size a KV cache without knowing which backend it got.
    #[test]
    fn an_allocator_reports_whether_it_commits_on_demand() {
        #[derive(Debug)]
        struct Stub(bool);

        // SAFETY: never allocates, so the non-overlap and validity guarantees
        // hold vacuously.
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

        // SAFETY: as above.
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
            "an allocator that says nothing must be treated as committing eagerly: a consumer \
             that believes otherwise will under-reserve"
        );

        // Through a trait object, which is how every real consumer holds one.
        let lazy: &dyn DeviceAllocator = &Stub(true);
        let eager: &dyn DeviceAllocator = &Stub(false);
        assert!(lazy.commits_on_demand());
        assert!(!eager.commits_on_demand());
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

//! What a virtual address range is made of.
//!
//! [`VirtualBuffer`](crate::VirtualBuffer) owns the interesting part: growth,
//! leasing, granule rounding, and the promise that the base address never
//! moves. None of that is platform-specific. What is platform-specific is three
//! operations — reserve address space, commit a block into part of it, release
//! that block — plus the granularity everything must be a multiple of.
//!
//! Splitting those out is what stops the device implementation from being a
//! second copy of the growth and leasing logic.
//!
//! # Host and device are the same shape
//!
//! | | host | CUDA |
//! |---|---|---|
//! | reserve | `VirtualAlloc2` placeholder / `mmap(PROT_NONE)` | `cuMemAddressReserve` |
//! | commit | `MapViewOfFile3` / `mmap(MAP_FIXED)` | `cuMemCreate` + `cuMemMap` + `cuMemSetAccess` |
//! | release | `UnmapViewOfFile2` / `mmap(PROT_NONE)` | `cuMemUnmap`; pool or `cuMemRelease` |
//! | granularity | 64 KiB / page size | `cuMemGetAllocationGranularity` |
//!
//! Measured on the hardware this was developed against: **64 KiB** on Windows,
//! **2 MiB** for CUDA on an RTX 4060 — where 2 MiB is roughly a thousand tokens
//! of one KV tensor at Llama-3-8B geometry.
//!
//! # Why there is an associated `Reservation`
//!
//! A backing cannot be stateless. Windows requires a placeholder to be *split*
//! before a block is mapped into part of it, and whether a split is needed
//! depends on the block's already-mapped neighbours — so committing needs to
//! know what else is mapped in the same reservation. CUDA needs the same shape
//! because mappings still belong to a reservation even when their physical
//! handles outlive them in a shared pool.
//!
//! Putting that state in an associated type rather than in the backing keeps
//! one backing able to serve many reservations, and keeps the state next to the
//! thing it describes.
//!
//! The cost is that `VirtualBacking` is not `dyn`-safe. That is deliberate and
//! it costs nothing: what callers hold is a
//! [`VirtualBuffer`](crate::VirtualBuffer), and *that* can be boxed behind an
//! object-safe trait if it ever needs to be. Nobody needs to hold a backing.
//!
//! # Why addresses are `usize`
//!
//! A device address is not a host pointer and must never be dereferenced on the
//! CPU. Typing both as `*mut u8` would invite exactly that.

use crate::VirtualMemoryError;
use onnx_runtime_memory_governor::MemoryAuthorityId;

/// Who charges physical memory committed by a [`VirtualBacking`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalMemoryAccounting {
    /// [`VirtualBuffer`](crate::VirtualBuffer) leases mapped physical bytes.
    Buffer,
    /// The backing's authority owns physical bytes independently of mappings.
    ///
    /// This is the contract for a physical-handle pool: pooled-unmapped bytes
    /// remain charged to `authority`, while mapped holder/zone bytes are only
    /// attribution and must not be charged as additional physical ownership.
    Backing {
        /// The one ledger that owns every physical byte held by the backing.
        authority: MemoryAuthorityId,
    },
}

/// The platform operations a [`VirtualBuffer`](crate::VirtualBuffer) is built
/// from.
///
/// # Safety
///
/// This trait is `unsafe` to implement because [`VirtualBuffer`] hands out the
/// base address and lets callers write to the committed prefix. An
/// implementation that reported a range it had not reserved, or a granularity
/// it did not honour, would turn those writes into memory corruption rather
/// than an error. Specifically:
///
/// * [`VirtualBacking::granularity`] is constant for the backing's life and a
///   power of two.
/// * [`VirtualBacking::reserve`] takes address space only. It must not commit
///   memory: the whole design rests on reserving generously being free.
/// * [`VirtualBacking::base`] returns the address the reservation actually
///   starts at, and that address does not change for the reservation's life.
/// * After [`VirtualBacking::commit`] returns `Ok`, every byte of
///   `base + offset .. base + offset + len` is writable through that address.
/// * Dropping a `Reservation` releases both its address space and any blocks
///   still committed in it.
///
/// [`VirtualBuffer`]: crate::VirtualBuffer
pub unsafe trait VirtualBacking: Send + Sync + std::fmt::Debug {
    /// One reserved address range, and whatever the platform needs to remember
    /// about what is committed in it.
    type Reservation: Send + Sync + std::fmt::Debug;

    /// Allocation granularity: every offset and length is a multiple of this.
    fn granularity(&self) -> usize;

    /// Who owns accounting for committed physical memory.
    ///
    /// Backings that retain physical allocations after unmapping must return
    /// [`PhysicalMemoryAccounting::Backing`]. A buffer validates the authority
    /// before reserving address space and does not take a second physical lease.
    fn physical_memory_accounting(&self) -> PhysicalMemoryAccounting {
        PhysicalMemoryAccounting::Buffer
    }

    /// Reserve `len` bytes of address space, committing nothing.
    fn reserve(&self, len: usize) -> Result<Self::Reservation, VirtualMemoryError>;

    /// The address the reservation starts at.
    fn base(reservation: &Self::Reservation) -> usize;

    /// Back `offset..offset + len` of `reservation` with fresh memory.
    ///
    /// `offset` and `len` are multiples of [`VirtualBacking::granularity`] and
    /// the range lies inside the reservation; the implementation is entitled to
    /// rely on both. Overlapping an already-committed block is a caller error
    /// the implementation should report rather than assume away.
    fn commit(
        &self,
        reservation: &mut Self::Reservation,
        offset: usize,
        len: usize,
    ) -> Result<(), VirtualMemoryError>;

    /// Give back the block committed at `offset`, leaving the address space
    /// reserved so it can be committed again later.
    fn release(
        &self,
        reservation: &mut Self::Reservation,
        offset: usize,
        len: usize,
    ) -> Result<(), VirtualMemoryError>;
}

/// The process's own address space.
///
/// The default backing. Uses placeholder reservations on Windows and `mmap` on
/// unix, both of which let a block be committed into part of a larger
/// reservation and taken back out without disturbing its neighbours.
#[derive(Debug, Default, Clone, Copy)]
pub struct HostBacking;

// SAFETY: every address comes from `VirtualRange::reserve`; the granularity is
// the platform's own and constant for the process; `VirtualRange` tracks its
// own mapped blocks and releases everything on drop.
unsafe impl VirtualBacking for HostBacking {
    type Reservation = crate::VirtualRange;

    fn granularity(&self) -> usize {
        crate::granularity()
    }

    fn reserve(&self, len: usize) -> Result<Self::Reservation, VirtualMemoryError> {
        crate::VirtualRange::reserve(len)
    }

    fn base(reservation: &Self::Reservation) -> usize {
        reservation.as_ptr() as usize
    }

    fn commit(
        &self,
        reservation: &mut Self::Reservation,
        offset: usize,
        len: usize,
    ) -> Result<(), VirtualMemoryError> {
        reservation.map(offset, len)
    }

    fn release(
        &self,
        reservation: &mut Self::Reservation,
        offset: usize,
        _len: usize,
    ) -> Result<(), VirtualMemoryError> {
        reservation.unmap(offset)
    }
}

//! Device memory from the CUDA virtual memory APIs, on one CUDA device.
//!
//! # Why not `cuMemAlloc`
//!
//! [`CudaDeviceAllocator`] calls `cuMemAlloc`, which allocates virtual *and*
//! physical memory together. That is the "reservation-based" model vAttention
//! ([arXiv 2405.04437](https://arxiv.org/abs/2405.04437)) identifies as the
//! root of the problem PagedAttention exists to work around: because you cannot
//! reserve an address range without also taking the physical memory behind it,
//! a buffer that might grow has to either take its maximum up front or be
//! stitched together from blocks that are not adjacent.
//!
//! The CUDA VMM APIs separate the two. `cuMemAddressReserve` takes address
//! space, which is free and effectively unlimited; `cuMemCreate` plus
//! `cuMemMap` take physical memory, in granules. So a buffer can hold one
//! contiguous address range for its whole lifetime while the physical memory
//! under it grows and shrinks.
//!
//! # Why this also fixes the accounting
//!
//! The memory ledger charges standing claims rather than allocations, because
//! a governor round-trip on every `Alloc` an execution provider makes would
//! cost more than the precision is worth (#652).
//!
//! Under VMM that trade changes, and it changes in a way that makes the strict
//! answer affordable. Physical memory is taken in granules -- 2 MiB on every
//! CUDA device we have looked at -- so a 4 GiB working set is roughly two
//! thousand commits over the life of a session rather than a charge per tensor
//! per inference. Charging every commit is therefore both **complete** and off
//! the hot path: the allocations that dominate by count are handed out from
//! granules that are already mapped and never reach the governor at all.
//!
//! What the ledger holds is the committed bytes. Reserved address space is not
//! charged, because it is not memory.
//!
//! # Granularity and who this is for
//!
//! CUDA's minimum granularity is 2 MiB. vAttention patches the open-source UVM
//! driver down to 64 KiB because at server concurrency the internal
//! fragmentation of 2 MiB granules is severe. That patch is not portable and we
//! do not take it: this runtime targets local, low-concurrency use, where the
//! waste is a few percent of a card rather than a multiple of it.
//!
//! Suballocation is what keeps that true. A granule is shared by as many
//! allocations as fit in it, so the fragmentation bound is one partially-used
//! granule per *arena*, not per allocation. An allocator that gave every
//! request its own granule would turn ONNX Runtime's many small tensors into
//! 2 MiB each, which is the failure mode this design exists to avoid.
//!
//! [`CudaDeviceAllocator`]: crate::device_allocator::CudaDeviceAllocator

use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, MemoryError, MemoryGovernor, MemoryLease, MemoryRole,
    Tier,
};
use onnx_runtime_virtual_memory::VirtualBacking;

use crate::virtual_memory::CudaVirtualBacking;
use cudarc::driver::CudaContext;

/// Environment switch selecting the VMM arena over `cuMemAlloc`.
///
/// # Currently inert on the native CUDA path
///
/// Enabling this today reserves address space and allocates nothing through
/// it. The arena installs when the execution provider adopts a memory
/// governor, and on the native path that happens *after* the session has
/// built every tensor it will use -- so nothing is left to ask it for memory.
/// Measured: `committed 0 B of 7732199424 B reserved` after a full generation
/// (#659).
///
/// The allocator itself is correct and tested; what is missing is a caller.
/// Until #659 is fixed, turning this on costs one address-space reservation
/// and changes nothing else, so it must not become the default.
///
/// # Why it is opt-in regardless
///
/// An allocator change is exactly the kind that looks free and is not. The
/// default should move only after it is measured against `cuMemAlloc` on real
/// models -- and after there is something to measure.
pub const CUDA_VMM_ENV: &str = "ONNX_GENAI_CUDA_VMM";

/// Whether the VMM arena is enabled. Any of `1`/`true`/`yes`/`on`.
pub fn vmm_enabled() -> bool {
    std::env::var(CUDA_VMM_ENV).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}
/// The device tier name used in errors raised before a governor is consulted.
const TIER: &str = "device";

fn invalid(requested: usize, reason: String) -> MemoryError {
    // The variant carries a &'static str; these are constructed once on a
    // failure path, so leaking the formatted text is cheaper than widening the
    // contract crate's error type for one caller.
    let reason: &'static str = Box::leak(reason.into_boxed_str());
    MemoryError::InvalidRequest {
        tier: TIER,
        requested: requested as u64,
        reason,
    }
}

fn round_up(granularity: usize, bytes: usize) -> usize {
    debug_assert!(granularity.is_power_of_two() || granularity > 0);
    bytes.div_ceil(granularity) * granularity
}

/// Which spans of a reservation are handed out, and which granules are mapped.
///
/// Split from [`Arena`] so that it holds no CUDA state. This is where every
/// off-by-one that would corrupt memory lives, and keeping a reservation out
/// of it means it can be tested on a machine with no GPU rather than only
/// where the hardware tests run -- which, as #636 found, was nowhere by
/// default.
#[derive(Debug)]
struct Spans {
    granularity: usize,
    /// Live allocations touching each granule. A granule is committed while
    /// this is non-zero.
    ///
    /// Counting *allocations* rather than bytes is what lets a granule be
    /// shared: the last allocation to leave releases it, and one that merely
    /// shrinks does not.
    granule_refs: Vec<u32>,
    /// Free spans as `offset -> len`, kept coalesced so an allocation freed
    /// next to another leaves one span rather than two.
    free: BTreeMap<usize, usize>,
    /// Live spans as `offset -> len`, so `deallocate` can be told the length
    /// even when the caller's `bytes` disagrees with what was rounded out.
    live: BTreeMap<usize, usize>,
    /// Bytes currently committed, which is `lease.bytes()` when a lease exists.
    committed: usize,
}

impl Spans {
    fn new(granularity: usize, capacity: usize) -> Self {
        let mut free = BTreeMap::new();
        free.insert(0, capacity);
        Self {
            granularity,
            granule_refs: vec![0; capacity / granularity],
            free,
            live: BTreeMap::new(),
            committed: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.granule_refs.len() * self.granularity
    }

    /// The granule indices spanned by `[offset, offset + len)`.
    fn granules(&self, offset: usize, len: usize) -> std::ops::Range<usize> {
        let first = offset / self.granularity;
        let last = (offset + len - 1) / self.granularity;
        first..last + 1
    }

    /// Take the first free span that fits `len` bytes at `align`, splitting it.
    ///
    /// First fit rather than best fit: the request stream here is a session's
    /// tensors, not an adversary, and a best-fit scan costs more than the
    /// fragmentation it saves at this scale.
    fn carve(&mut self, len: usize, align: usize) -> Option<usize> {
        let (&start, &span) = self
            .free
            .iter()
            .find(|&(&start, &span)| aligned_fits(start, span, len, align))?;
        let offset = start.next_multiple_of(align);
        self.free.remove(&start);
        // Whatever alignment skipped at the front stays free.
        if offset > start {
            self.free.insert(start, offset - start);
        }
        let tail = start + span - (offset + len);
        if tail > 0 {
            self.free.insert(offset + len, tail);
        }
        Some(offset)
    }

    /// Return `[offset, offset + len)` to the free list, merging with either
    /// neighbour so the list cannot fragment into adjacent free spans.
    fn give_back(&mut self, offset: usize, len: usize) {
        let mut start = offset;
        let mut end = offset + len;
        if let Some((&before, &span)) = self.free.range(..offset).next_back()
            && before + span == start
        {
            self.free.remove(&before);
            start = before;
        }
        if let Some(&span) = self.free.get(&end) {
            self.free.remove(&end);
            end += span;
        }
        self.free.insert(start, end - start);
    }
}

/// One reserved address range and the bookkeeping over it.
#[derive(Debug)]
struct Arena {
    reservation: <CudaVirtualBacking as VirtualBacking>::Reservation,
    spans: Spans,
    /// The ledger claim covering every committed granule.
    /// The ledger claim covering every committed granule.
    ///
    /// Taken at construction at zero bytes so every commit is a `grow` on an
    /// existing claim. That keeps the governor out of this type: a lease is
    /// owned, a `&dyn` borrow is not, and the execution-provider contract
    /// hands over the latter.
    lease: MemoryLease,
}

/// Device memory carved out of one reserved address range, with physical
/// granules mapped on demand and charged to a [`MemoryGovernor`].
pub struct CudaVmmAllocator {
    backing: CudaVirtualBacking,
    arena: Mutex<Arena>,
    holder: HolderId,
    role: MemoryRole,
    device: DeviceKey,
}

// `MemoryGovernor` is a replaceable contract and does not require `Debug`, so
// the derive cannot see through it. Reporting what the arena is doing is more
// useful in a log than the governor's identity would be anyway.
impl std::fmt::Debug for CudaVmmAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (committed, reserved) = self.committed_and_reserved();
        f.debug_struct("CudaVmmAllocator")
            .field("device", &self.device)
            .field("holder", &self.holder)
            .field("role", &self.role)
            .field("committed", &committed)
            .field("reserved", &reserved)
            .finish()
    }
}

impl CudaVmmAllocator {
    /// Reserve `capacity` bytes of device address space to allocate from.
    ///
    /// `capacity` costs nothing but address space, so it should be the largest
    /// the arena could ever need rather than a guess at what it will use --
    /// running out of *reservation* is a hard failure, whereas leaving it
    /// unmapped is free. Physical memory is taken granule by granule as
    /// allocations land, and each granule is leased from `governor` before it
    /// is mapped.
    pub fn new(
        context: Arc<CudaContext>,
        device: DeviceKey,
        device_ordinal: i32,
        capacity: usize,
        governor: &dyn MemoryGovernor,
        holder: HolderId,
        role: MemoryRole,
    ) -> Result<Self, MemoryError> {
        let backing = CudaVirtualBacking::new(context, device_ordinal);
        let granularity = backing.granularity();
        if granularity == 0 {
            return Err(invalid(
                capacity,
                String::from("CUDA reported a zero allocation granularity"),
            ));
        }
        let capacity = round_up(granularity, capacity.max(granularity));
        let reservation = backing
            .reserve(capacity)
            .map_err(|error| invalid(capacity, format!("cuMemAddressReserve: {error}")))?;

        Ok(Self {
            backing,
            arena: Mutex::new(Arena {
                reservation,
                spans: Spans::new(granularity, capacity),
                lease: governor.reserve(Tier::Device, 0, role, holder)?,
            }),
            holder,
            role,
            device,
        })
    }

    /// Bytes of physical memory mapped right now, and the address space
    /// reserved for them.
    ///
    /// The gap between the two is the point of this allocator, so both are
    /// reported: a large reservation with a small commitment is working as
    /// intended, not leaking.
    pub fn committed_and_reserved(&self) -> (usize, usize) {
        let arena = self.lock();
        (arena.spans.committed, arena.spans.capacity())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Arena> {
        self.arena
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Commit every granule in `range` that is not already mapped, leasing the
    /// bytes first.
    ///
    /// Leasing before mapping is the same order [`VirtualBuffer`] uses and for
    /// the same reason: a refusal that has already taken device memory makes
    /// the budget decorative.
    ///
    /// [`VirtualBuffer`]: onnx_runtime_virtual_memory::VirtualBuffer
    fn commit_range(
        &self,
        arena: &mut Arena,
        range: std::ops::Range<usize>,
    ) -> Result<(), MemoryError> {
        let granularity = arena.spans.granularity;
        let mut mapped: Vec<usize> = Vec::new();
        for granule in range {
            if arena.spans.granule_refs[granule] > 0 {
                arena.spans.granule_refs[granule] += 1;
                continue;
            }
            if let Err(error) = self.take(arena, granularity) {
                self.undo(arena, &mapped, granularity);
                return Err(error);
            }
            let offset = granule * granularity;
            if let Err(error) = self
                .backing
                .commit(&mut arena.reservation, offset, granularity)
            {
                self.give_back_lease(arena, granularity);
                self.undo(arena, &mapped, granularity);
                return Err(invalid(granularity, format!("cuMemMap: {error}")));
            }
            arena.spans.granule_refs[granule] = 1;
            arena.spans.committed += granularity;
            mapped.push(granule);
        }
        Ok(())
    }

    /// Roll back the granules this call had already taken, so a failure
    /// halfway through leaves the arena as it was found.
    fn undo(&self, arena: &mut Arena, mapped: &[usize], granularity: usize) {
        for &granule in mapped {
            arena.spans.granule_refs[granule] = 0;
            let offset = granule * granularity;
            let _ = self
                .backing
                .release(&mut arena.reservation, offset, granularity);
            arena.spans.committed -= granularity;
            self.give_back_lease(arena, granularity);
        }
    }

    fn take(&self, arena: &mut Arena, bytes: usize) -> Result<(), MemoryError> {
        arena.lease.grow(bytes as u64)
    }

    fn give_back_lease(&self, arena: &mut Arena, bytes: usize) {
        arena.lease.shrink(bytes as u64);
    }

    /// Drop this call's claim on `range`, unmapping whatever it was the last
    /// user of.
    fn release_range(&self, arena: &mut Arena, range: std::ops::Range<usize>) {
        let granularity = arena.spans.granularity;
        for granule in range {
            match arena.spans.granule_refs[granule].checked_sub(1) {
                Some(0) | None => {
                    if arena.spans.granule_refs[granule] == 0 {
                        // Already released; releasing twice would unmap memory
                        // another allocation is using.
                        continue;
                    }
                    arena.spans.granule_refs[granule] = 0;
                    let offset = granule * granularity;
                    let _ = self
                        .backing
                        .release(&mut arena.reservation, offset, granularity);
                    arena.spans.committed -= granularity;
                    self.give_back_lease(arena, granularity);
                }
                Some(remaining) => arena.spans.granule_refs[granule] = remaining,
            }
        }
    }
}

/// Whether a free span starting at `start` can hold `len` bytes once the start
/// is pushed up to `align`.
fn aligned_fits(start: usize, span: usize, len: usize, align: usize) -> bool {
    let offset = start.next_multiple_of(align);
    offset
        .checked_add(len)
        .is_some_and(|end| end <= start + span)
}

// SAFETY: every pointer handed out lies inside this allocator's reservation,
// in a span removed from the free list and recorded in `live`, so no two live
// allocations overlap. The granules under a span stay mapped while its
// reference count is non-zero, so the memory remains valid until `deallocate`.
// `device` names the CUDA device the reservation belongs to.
impl DeviceAllocator for CudaVmmAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        if bytes == 0 {
            return Err(invalid(
                bytes,
                String::from("a zero-byte allocation has no address to return"),
            ));
        }
        let align = align.max(1);
        if !align.is_power_of_two() {
            return Err(invalid(
                bytes,
                format!("alignment {align} is not a power of two"),
            ));
        }
        let mut arena = self.lock();
        let Some(offset) = arena.spans.carve(bytes, align) else {
            let free: usize = arena.spans.free.values().sum();
            return Err(invalid(
                bytes,
                format!(
                    "no free span of {bytes} bytes at alignment {align} in the reserved range; \
                     {free} bytes are free but fragmented across {} spans",
                    arena.spans.free.len()
                ),
            ));
        };
        let granules = arena.spans.granules(offset, bytes);
        if let Err(error) = self.commit_range(&mut arena, granules) {
            arena.spans.give_back(offset, bytes);
            return Err(error);
        }
        arena.spans.live.insert(offset, bytes);
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        // SAFETY: `base` is non-null (cuMemAddressReserve refuses otherwise) and
        // `offset` is within the reservation, so the sum cannot be null.
        Ok(unsafe { NonNull::new_unchecked((base + offset) as *mut u8) })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _bytes: usize, _align: usize) {
        let mut arena = self.lock();
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        let address = ptr.as_ptr() as usize;
        let Some(offset) = address.checked_sub(base) else {
            return;
        };
        // The recorded length wins over the caller's: `allocate` is free to
        // hand back a span it rounded, and freeing the caller's figure would
        // leave the difference unreachable.
        let Some(len) = arena.spans.live.remove(&offset) else {
            return;
        };
        let granules = arena.spans.granules(offset, len);
        self.release_range(&mut arena, granules);
        arena.spans.give_back(offset, len);
    }

    fn device(&self) -> DeviceKey {
        self.device
    }
}

impl Drop for CudaVmmAllocator {
    fn drop(&mut self) {
        // Report what was actually mapped. Without this the arena's whole
        // premise -- that reserved address space is large and committed memory
        // is small -- is unfalsifiable from outside: `committed_and_reserved`
        // exists but nothing calls it, so "is this doing anything?" has no
        // answer short of a debugger.
        let (committed, reserved, granularity) = {
            let arena = self.lock();
            (
                arena.spans.committed,
                arena.spans.capacity(),
                arena.spans.granularity,
            )
        };
        eprintln!(
            "cuda_ep: VMM arena closing: committed {committed} B of {reserved} B reserved \
             ({} granules of {granularity} B)",
            committed / granularity.max(1),
        );

        // The reservation's own `Drop` unmaps and frees every block, so the
        // only thing left is to stop the ledger believing the granules are
        // still held. Shrinking to zero does that; the lease's own `Drop`
        // would too, but doing it here keeps the two in step if a field is
        // ever reordered.
        let mut arena = self.lock();
        let held = arena.spans.committed as u64;
        arena.lease.shrink(held);
        arena.spans.committed = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Alignment padding at the front of a span stays free rather than being
    /// swallowed.
    ///
    /// A carve that dropped it would leak address space on every aligned
    /// allocation -- invisible until a long-running session cannot find room
    /// in a reservation that is mostly empty.
    #[test]
    fn alignment_padding_is_returned_to_the_free_list() {
        let mut spans = Spans::new(1 << 16, 1 << 20);

        assert_eq!(spans.carve(8, 1), Some(0));
        assert_eq!(spans.carve(16, 256), Some(256));

        assert_eq!(
            spans.free.get(&8),
            Some(&248),
            "the 8..256 gap alignment skipped must stay free: {:?}",
            spans.free
        );
    }

    /// Freeing between two free spans leaves one span, not three.
    ///
    /// Without coalescing the free list degrades into a list of granule-sized
    /// holes, and a later large allocation fails in an arena that is empty.
    #[test]
    fn freeing_coalesces_with_both_neighbours() {
        let mut spans = Spans::new(1 << 16, 3 << 16);
        let a = spans.carve(1 << 16, 1).unwrap();
        let b = spans.carve(1 << 16, 1).unwrap();
        let c = spans.carve(1 << 16, 1).unwrap();

        spans.give_back(a, 1 << 16);
        spans.give_back(c, 1 << 16);
        assert_eq!(spans.free.len(), 2, "a and c are not adjacent yet");

        spans.give_back(b, 1 << 16);
        assert_eq!(
            spans.free.len(),
            1,
            "b joins a and c into one span: {:?}",
            spans.free
        );
        assert_eq!(spans.free.get(&0), Some(&(3usize << 16)));
    }

    /// Allocations smaller than a granule share one.
    ///
    /// This is the property that keeps 2 MiB granularity affordable, and the
    /// reason this allocator suballocates at all. Giving every request its own
    /// granule would turn ONNX Runtime's many small tensors into 2 MiB each --
    /// the fragmentation vAttention patches the CUDA driver to avoid, which we
    /// avoid by sharing instead.
    #[test]
    fn allocations_smaller_than_a_granule_share_it() {
        let spans = Spans::new(2 << 20, 4 << 20);

        assert_eq!(spans.granules(0, 64), 0..1);
        assert_eq!(spans.granules(1024, 64), 0..1, "same granule as offset 0");
        assert_eq!(
            spans.granules((2 << 20) - 32, 64),
            0..2,
            "a span crossing the boundary needs both granules"
        );
    }

    /// A span exactly filling a granule does not claim the next one.
    ///
    /// `granules` is a half-open range computed from the last *byte*, so an
    /// exact fit must not round up -- committing one granule per allocation
    /// more than needed would double this allocator's memory use at small
    /// sizes and be very hard to see.
    #[test]
    fn an_exact_granule_fit_claims_one_granule() {
        let spans = Spans::new(2 << 20, 8 << 20);

        assert_eq!(spans.granules(0, 2 << 20), 0..1);
        assert_eq!(spans.granules(2 << 20, 2 << 20), 1..2);
        assert_eq!(spans.granules(0, (2 << 20) + 1), 0..2, "one byte over");
    }

    /// The whole reservation comes back after everything is freed.
    ///
    /// The property that matters over a session's lifetime: allocate and free
    /// in an order that interleaves, and the arena must end as it started or
    /// it leaks a little on every request.
    #[test]
    fn an_arena_returns_to_one_free_span() {
        let capacity = 16 << 16;
        let mut spans = Spans::new(1 << 16, capacity);
        let sizes = [1000, 1 << 16, 40, (1 << 16) + 7, 8192];

        let mut offsets = Vec::new();
        for size in sizes {
            offsets.push((spans.carve(size, 64).expect("fits"), size));
        }
        // Free out of order: adjacency must be discovered, not assumed.
        for &(offset, size) in [3, 0, 4, 1, 2].map(|i| &offsets[i]) {
            spans.give_back(offset, size);
        }

        assert_eq!(
            spans.free.len(),
            1,
            "every span should have merged back: {:?}",
            spans.free
        );
        assert_eq!(spans.free.get(&0), Some(&capacity));
    }

    /// A request larger than what is free is refused rather than served.
    #[test]
    fn an_oversized_request_finds_no_span() {
        let mut spans = Spans::new(1 << 16, 2 << 16);
        assert_eq!(spans.carve(3 << 16, 1), None);
        assert_eq!(spans.capacity(), 2 << 16);
    }
}

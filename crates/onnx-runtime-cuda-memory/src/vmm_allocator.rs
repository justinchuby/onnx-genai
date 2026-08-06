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

use std::collections::{BTreeMap, BTreeSet};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
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

/// Process-global VMM arena counters.
///
/// # Why these are global rather than per-arena
///
/// A caller profiling a run wants to know what the arena did, and it holds a
/// session rather than an allocator -- the allocator is several layers down
/// inside an execution provider. Threading a handle out for the sake of a
/// number is more coupling than the number is worth, and this repository
/// already answers the same question the same way for weight offload
/// (`global_offload_stats`).
///
/// # Why a quantity and not an event
///
/// The arena was once installed, logged that it was installed, and committed
/// **zero bytes** for an entire generation (#659). The log line was true and
/// useless: it could not be told apart from a hook that never fired. Only a
/// byte count could, and only after someone printed one. These exist so that
/// "is the arena doing anything" is a reading rather than an argument.
static GLOBAL_COMMITS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_RELEASES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_COMMITTED_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_RESERVED_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_PEAK_COMMITTED_BYTES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
/// Times a granule was released whose reference count was already zero.
///
/// Always zero in a correct run. A non-zero reading means some allocation
/// committed a granule without taking a reference for it, or released one
/// twice -- an accounting error that would otherwise show up only as another
/// allocation's memory being unmapped underneath it.
static GLOBAL_REF_UNDERFLOWS: AtomicU64 = AtomicU64::new(0);
/// Times a byte counter was decremented below zero and had to be clamped.
///
/// Always zero in a correct run. Every byte subtracted from these counters is
/// supposed to have been added first, so reaching zero from below means a
/// commit path mapped memory without counting it. On a `u64` the natural
/// `fetch_sub` **wraps**, turning a small accounting slip into an enormous
/// number in `--profile` -- which looks like a catastrophic leak and is
/// actually the opposite. Clamping keeps the reading sane; this counter is how
/// the underlying fault stays visible rather than being smoothed away.
static GLOBAL_BYTE_UNDERFLOWS: AtomicU64 = AtomicU64::new(0);

/// Subtract without wrapping, recording the fault if the counter would go
/// negative.
///
/// Deliberately does **not** assert. One caller is `Drop`, and a panic there
/// aborts the process instead of failing a test -- the first draft did assert
/// and turned a clean assertion into `STATUS_STACK_BUFFER_OVERRUN` with no
/// message. Recording the fault and letting tests assert on
/// [`GlobalVmmStats::byte_underflows`] keeps the diagnosis readable and keeps
/// `Drop` infallible.
fn subtract_counted(counter: &AtomicU64, amount: u64) {
    let clamped = counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_sub(amount))
        })
        .is_ok_and(|previous| previous < amount);
    if clamped {
        GLOBAL_BYTE_UNDERFLOWS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Snapshot of the process-global VMM arena counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalVmmStats {
    /// Granules mapped since the process started.
    pub commits: u64,
    /// Granules unmapped since the process started.
    pub releases: u64,
    /// Physical bytes mapped right now.
    pub committed_bytes: u64,
    /// Address space reserved right now. Costs nothing but address space; the
    /// gap between this and `committed_bytes` is the point of the approach.
    pub reserved_bytes: u64,
    /// High-water mark of `committed_bytes`.
    pub peak_committed_bytes: u64,
    /// Spans handed out. Compare with `commits`: many allocations per commit
    /// is granule sharing working, and one commit per allocation would mean
    /// every small tensor is costing a whole 2 MiB granule.
    pub allocations: u64,
    /// Times a granule was released whose reference count was already zero.
    ///
    /// **Anything but zero is a bug**, and the reason this is reported rather
    /// than merely defended against: the release path skips such a granule to
    /// avoid unmapping memory another allocation is still using, which is the
    /// safe action but also a silent one. Without this counter a refcount
    /// imbalance is indistinguishable from correct operation until the wrong
    /// memory is unmapped somewhere else entirely.
    pub ref_underflows: u64,
    /// Times a byte counter would have gone negative and was clamped.
    ///
    /// **Anything but zero is a bug.** Without the clamp a `u64` subtraction
    /// wraps, so a small accounting slip reads as an enormous committed byte
    /// count -- which looks like a catastrophic leak and is actually the
    /// opposite.
    pub byte_underflows: u64,
}

/// Read the process-global VMM arena counters.
///
/// All zero means no arena was ever built -- which is the normal state when
/// `ONNX_GENAI_CUDA_VMM` is unset, and is distinguishable from an arena that
/// was built and never used (`reserved_bytes > 0`, `commits == 0`).
pub fn global_vmm_stats() -> GlobalVmmStats {
    GlobalVmmStats {
        commits: GLOBAL_COMMITS.load(Ordering::Relaxed),
        releases: GLOBAL_RELEASES.load(Ordering::Relaxed),
        committed_bytes: GLOBAL_COMMITTED_BYTES.load(Ordering::Relaxed),
        reserved_bytes: GLOBAL_RESERVED_BYTES.load(Ordering::Relaxed),
        peak_committed_bytes: GLOBAL_PEAK_COMMITTED_BYTES.load(Ordering::Relaxed),
        allocations: GLOBAL_ALLOCATIONS.load(Ordering::Relaxed),
        ref_underflows: GLOBAL_REF_UNDERFLOWS.load(Ordering::Relaxed),
        byte_underflows: GLOBAL_BYTE_UNDERFLOWS.load(Ordering::Relaxed),
    }
}

/// Reset the counters. Tests only -- they are process-global, so a test that
/// reads them must not race another that writes them.
pub fn reset_global_vmm_stats() {
    GLOBAL_COMMITS.store(0, Ordering::Relaxed);
    GLOBAL_RELEASES.store(0, Ordering::Relaxed);
    GLOBAL_COMMITTED_BYTES.store(0, Ordering::Relaxed);
    GLOBAL_RESERVED_BYTES.store(0, Ordering::Relaxed);
    GLOBAL_PEAK_COMMITTED_BYTES.store(0, Ordering::Relaxed);
    GLOBAL_ALLOCATIONS.store(0, Ordering::Relaxed);
    GLOBAL_REF_UNDERFLOWS.store(0, Ordering::Relaxed);
    GLOBAL_BYTE_UNDERFLOWS.store(0, Ordering::Relaxed);
}

fn note_commit(granularity: usize) {
    GLOBAL_COMMITS.fetch_add(1, Ordering::Relaxed);
    let now = GLOBAL_COMMITTED_BYTES.fetch_add(granularity as u64, Ordering::Relaxed)
        + granularity as u64;
    GLOBAL_PEAK_COMMITTED_BYTES.fetch_max(now, Ordering::Relaxed);
    #[cfg(feature = "tracing")]
    tracing::debug!(
        target: "onnx_runtime::cuda::vmm",
        granule_bytes = granularity,
        committed_bytes = now,
        "vmm arena committed a granule"
    );
}

fn note_release(granularity: usize) {
    GLOBAL_RELEASES.fetch_add(1, Ordering::Relaxed);
    subtract_counted(&GLOBAL_COMMITTED_BYTES, granularity as u64);
    let now = GLOBAL_COMMITTED_BYTES.load(Ordering::Relaxed);
    let _ = now;
    #[cfg(feature = "tracing")]
    tracing::debug!(
        target: "onnx_runtime::cuda::vmm",
        granule_bytes = granularity,
        committed_bytes = now,
        "vmm arena released a granule"
    );
}

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
    /// Live spans as `offset -> span`, so `deallocate` can release exactly the
    /// granules this allocation claimed even when it reserved more address
    /// space than it committed.
    live: BTreeMap<usize, LiveSpan>,
    /// Bytes currently committed, which is `lease.bytes()` when a lease exists.
    committed: usize,
}

#[derive(Debug)]
struct LiveSpan {
    len: usize,
    committed: BTreeSet<usize>,
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
    /// Reserve `capacity` bytes of device address space, accounting against a
    /// ledger of this allocator's own.
    ///
    /// # Why this exists
    ///
    /// The arena has to be in place before anything allocates, and on the
    /// native path everything allocates while the session loads -- which is
    /// before the engine's governor reaches the execution provider. An arena
    /// built at adoption time is installed at the one moment after which
    /// nothing will ask it for memory, which is what #659 measured as
    /// `committed 0 B`.
    ///
    /// So it starts with a private ledger sized to the device, and
    /// [`adopt_governor`] moves the claim to the real one once it arrives.
    /// Allocations before adoption are unguarded, which is exactly what
    /// `cuMemAlloc` did in that window, so nothing is lost by it.
    ///
    /// [`adopt_governor`]: Self::adopt_governor
    pub fn detached(
        context: Arc<CudaContext>,
        device: DeviceKey,
        device_ordinal: i32,
        capacity: usize,
        holder: HolderId,
        role: MemoryRole,
    ) -> Result<Self, MemoryError> {
        // Sized to the reservation rather than to the device: this ledger is
        // bookkeeping until the real one arrives, and a limit it could refuse
        // at would refuse allocations `cuMemAlloc` would have served.
        let private = onnx_runtime_memory_governor::LedgerGovernor::new(
            onnx_runtime_memory_governor::LeaseLedger::new(u64::MAX, 0, 0),
        );
        Self::new(
            context,
            device,
            device_ordinal,
            capacity,
            &private,
            holder,
            role,
        )
    }

    /// Move this arena's claim onto `governor`, reporting the bytes it now
    /// holds there.
    ///
    /// Takes a lease for everything currently committed. If the governor
    /// refuses, the arena keeps its previous claim and the caller learns the
    /// shortfall -- the memory is already mapped, so failing here must not
    /// unmap it.
    pub fn adopt_governor(
        &self,
        governor: &dyn MemoryGovernor,
        holder: HolderId,
    ) -> Result<u64, MemoryError> {
        let mut arena = self.lock();
        let committed = arena.spans.committed as u64;
        let lease = governor.reserve(Tier::Device, committed, self.role, holder)?;
        // Replacing the lease drops the private one, which returns the bytes to
        // a ledger nobody reads. The real governor now holds them.
        arena.lease = lease;
        Ok(committed)
    }

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

        let arena = Arena {
            reservation,
            spans: Spans::new(granularity, capacity),
            lease: governor.reserve(Tier::Device, 0, role, holder)?,
        };
        // After the last fallible step: an arena that failed to build has no
        // `Drop` to take these back off the books.
        GLOBAL_RESERVED_BYTES.fetch_add(capacity as u64, Ordering::Relaxed);

        Ok(Self {
            backing,
            arena: Mutex::new(arena),
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
    fn claim_granules(
        &self,
        arena: &mut Arena,
        granules: impl IntoIterator<Item = usize>,
    ) -> Result<BTreeSet<usize>, MemoryError> {
        let granularity = arena.spans.granularity;
        let mut claimed = BTreeSet::new();
        for granule in granules {
            if !claimed.insert(granule) {
                continue;
            }
            if arena.spans.granule_refs[granule] > 0 {
                arena.spans.granule_refs[granule] += 1;
                continue;
            }
            if let Err(error) = self.take(arena, granularity) {
                claimed.remove(&granule);
                self.release_granules(arena, &claimed);
                return Err(error);
            }
            let offset = granule * granularity;
            if let Err(error) = self
                .backing
                .commit(&mut arena.reservation, offset, granularity)
            {
                self.give_back_lease(arena, granularity);
                claimed.remove(&granule);
                self.release_granules(arena, &claimed);
                return Err(invalid(granularity, format!("cuMemMap: {error}")));
            }
            arena.spans.granule_refs[granule] = 1;
            arena.spans.committed += granularity;
            note_commit(granularity);
        }
        Ok(claimed)
    }

    fn take(&self, arena: &mut Arena, bytes: usize) -> Result<(), MemoryError> {
        arena.lease.grow(bytes as u64)
    }

    fn give_back_lease(&self, arena: &mut Arena, bytes: usize) {
        arena.lease.shrink(bytes as u64);
    }

    /// Drop this allocation's claims, unmapping whatever it was the last user
    /// of.
    fn release_granules(&self, arena: &mut Arena, granules: &BTreeSet<usize>) {
        let granularity = arena.spans.granularity;
        for &granule in granules.iter().rev() {
            match arena.spans.granule_refs[granule].checked_sub(1) {
                Some(0) => {
                    arena.spans.granule_refs[granule] = 0;
                    let offset = granule * granularity;
                    let _ = self
                        .backing
                        .release(&mut arena.reservation, offset, granularity);
                    arena.spans.committed -= granularity;
                    note_release(granularity);
                    self.give_back_lease(arena, granularity);
                }
                Some(remaining) => arena.spans.granule_refs[granule] = remaining,
                None => {
                    // Already released. Continuing is the safe action --
                    // unmapping here would pull memory out from under
                    // whichever allocation still holds the granule.
                    //
                    // But reaching this line at all means the reference counts
                    // no longer balance. It is counted for release builds and
                    // asserted in debug builds, rather than panicking from a
                    // Drop-reachable path and aborting during unwinding.
                    GLOBAL_REF_UNDERFLOWS.fetch_add(1, Ordering::Relaxed);
                    debug_assert!(
                        false,
                        "vmm arena: released granule {granule} whose reference count was \
                         already zero -- some allocation committed it without taking a \
                         reference, or freed it twice"
                    );
                }
            }
        }
    }

    fn release_committed_granules(&self, arena: &mut Arena, granules: &BTreeSet<usize>) {
        self.release_granules(arena, granules);
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
        let full = 0..bytes;
        self.allocate_committed(bytes, align, std::slice::from_ref(&full))
    }

    fn allocate_committed(
        &self,
        bytes: usize,
        align: usize,
        committed_ranges: &[std::ops::Range<usize>],
    ) -> Result<NonNull<u8>, MemoryError> {
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
        let mut committed = BTreeSet::new();
        for range in committed_ranges {
            if range.start > range.end || range.end > bytes {
                self.release_committed_granules(&mut arena, &committed);
                arena.spans.give_back(offset, bytes);
                return Err(invalid(
                    bytes,
                    format!(
                        "committed subrange {}..{} lies outside allocation of {bytes} bytes",
                        range.start, range.end
                    ),
                ));
            }
            if range.is_empty() {
                continue;
            }
            let absolute = offset + range.start..offset + range.end;
            let granules = arena
                .spans
                .granules(absolute.start, absolute.len())
                .filter(|granule| !committed.contains(granule))
                .collect::<Vec<_>>();
            let claimed = match self.claim_granules(&mut arena, granules) {
                Ok(claimed) => claimed,
                Err(error) => {
                    self.release_committed_granules(&mut arena, &committed);
                    arena.spans.give_back(offset, bytes);
                    return Err(error);
                }
            };
            committed.extend(claimed);
        }
        arena.spans.live.insert(
            offset,
            LiveSpan {
                len: bytes,
                committed,
            },
        );
        GLOBAL_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        // SAFETY: `base` is non-null (cuMemAddressReserve refuses otherwise) and
        // `offset` is within the reservation, so the sum cannot be null.
        Ok(unsafe { NonNull::new_unchecked((base + offset) as *mut u8) })
    }

    fn commit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        _align: usize,
        byte_offset: usize,
        bytes: usize,
    ) -> Result<(), MemoryError> {
        let end = byte_offset.checked_add(bytes).ok_or_else(|| {
            invalid(
                allocation_bytes,
                format!("commit range offset {byte_offset} plus {bytes} bytes overflows"),
            )
        })?;
        if end > allocation_bytes {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "commit range {byte_offset}..{end} exceeds allocation of {allocation_bytes} bytes"
                ),
            ));
        }
        if bytes == 0 {
            return Ok(());
        }
        let mut arena = self.lock();
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        let address = ptr.as_ptr() as usize;
        let Some(offset) = address.checked_sub(base) else {
            return Err(invalid(
                allocation_bytes,
                "commit pointer is below the VMM arena reservation".to_string(),
            ));
        };
        let Some(live) = arena.spans.live.get(&offset) else {
            return Err(invalid(
                allocation_bytes,
                "commit pointer is not a live VMM allocation".to_string(),
            ));
        };
        let len = live.len;
        if len != allocation_bytes {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "commit allocation size {allocation_bytes} does not match live VMM allocation size {len}"
                ),
            ));
        }
        let absolute = offset + byte_offset..offset + end;
        let granules = arena
            .spans
            .granules(absolute.start, absolute.len())
            .filter(|granule| !live.committed.contains(granule))
            .collect::<Vec<_>>();
        let claimed = self.claim_granules(&mut arena, granules)?;
        if let Some(live) = arena.spans.live.get_mut(&offset) {
            live.committed.extend(claimed);
        }
        Ok(())
    }

    fn decommit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        _align: usize,
        byte_offset: usize,
        bytes: usize,
    ) -> Result<(), MemoryError> {
        let end = byte_offset.checked_add(bytes).ok_or_else(|| {
            invalid(
                allocation_bytes,
                format!("decommit range offset {byte_offset} plus {bytes} bytes overflows"),
            )
        })?;
        if end > allocation_bytes {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "decommit range {byte_offset}..{end} exceeds allocation of {allocation_bytes} bytes"
                ),
            ));
        }
        if bytes == 0 {
            return Ok(());
        }
        let mut arena = self.lock();
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        let address = ptr.as_ptr() as usize;
        let Some(offset) = address.checked_sub(base) else {
            return Err(invalid(
                allocation_bytes,
                "decommit pointer is below the VMM arena reservation".to_string(),
            ));
        };
        let Some(live) = arena.spans.live.get(&offset) else {
            return Err(invalid(
                allocation_bytes,
                "decommit pointer is not a live VMM allocation".to_string(),
            ));
        };
        if live.len != allocation_bytes {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "decommit allocation size {allocation_bytes} does not match live VMM allocation size {}",
                    live.len
                ),
            ));
        }
        let granularity = arena.spans.granularity;
        let absolute_start = offset + byte_offset;
        let absolute_end = offset + end;
        let releasable = live
            .committed
            .iter()
            .copied()
            .filter(|granule| {
                let start = granule * granularity;
                start >= absolute_start && start < absolute_end
            })
            .collect::<BTreeSet<_>>();
        self.release_committed_granules(&mut arena, &releasable);
        if let Some(live) = arena.spans.live.get_mut(&offset) {
            for granule in releasable {
                live.committed.remove(&granule);
            }
        }
        Ok(())
    }

    fn allocation_committed_bytes(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        _align: usize,
    ) -> usize {
        let arena = self.lock();
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        let address = ptr.as_ptr() as usize;
        let Some(offset) = address.checked_sub(base) else {
            return allocation_bytes;
        };
        arena
            .spans
            .live
            .get(&offset)
            .map(|live| live.committed.len() * arena.spans.granularity)
            .unwrap_or(allocation_bytes)
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
        let Some(live) = arena.spans.live.remove(&offset) else {
            return;
        };
        self.release_committed_granules(&mut arena, &live.committed);
        arena.spans.give_back(offset, live.len);
    }

    /// True: spans are carved from granules mapped as they are needed, and
    /// each granule is leased before it is mapped.
    fn commits_on_demand(&self) -> bool {
        true
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

        // Keep the process-global counters honest. Without this the arena's
        // bytes stay on the books after it is gone, and a second run in the
        // same process reads the first run's memory as if it were still
        // mapped.
        subtract_counted(&GLOBAL_COMMITTED_BYTES, held);
        subtract_counted(&GLOBAL_RESERVED_BYTES, reserved as u64);
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

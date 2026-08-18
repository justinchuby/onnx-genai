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

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use onnx_runtime_memory_governor::{
    AllocationCommitRange, DeviceAllocator, DeviceKey, HolderId, MappedPhysicalCapacityToken,
    MemoryAuthorityId, MemoryError, MemoryGovernor, MemoryLease, MemoryRole, SharedDevicePrefix,
    SharedMapping, SharedPrefixCommitInfo, Tier, VirtualBacking as MemoryVirtualBacking,
};
use onnx_runtime_virtual_memory::{PhysicalMemoryAccounting, VirtualBacking};

use crate::virtual_memory::{
    CudaVirtualBacking, PhysicalHandlePool, PhysicalHandlePoolStats, SharedPrefixReservation,
};
use cudarc::driver::CudaContext;

/// Environment switch selecting the VMM arena over `cuMemAlloc`.
///
/// # What enabling it does today
///
/// When set, the CUDA execution provider installs the VMM arena **at
/// construction** (not at governor adoption) and routes every device
/// allocation it makes — including the ORT scratch allocations
/// (`KernelContext_GetScratchBuffer`) the plugin path projects through the
/// provider's `allocate`/`deallocate` — through it. That is the caller the
/// arena was always missing on the plugin/standalone path (#659, #956):
/// repeated same-size scratch requests reuse committed memory from the
/// retained physical-handle pool instead of calling `cuMemAlloc` per dispatch.
///
/// The construction-time install predates governor adoption on purpose: on the
/// native path the session allocates every tensor while loading, before any
/// governor reaches the provider, so an arena installed at adoption is
/// installed at the one moment after which nothing will ask it for memory
/// (#659). The construction-time install closed that gap for the native path;
/// the standalone (plugin, no-governor) path uses the same install and, from
/// #956 on, a default retained physical-handle pool so its scratch reuse is
/// real rather than a per-cycle `cuMemCreate`/`cuMemRelease` churn.
///
/// # Why it is opt-in regardless
///
/// An allocator change is exactly the kind that looks free and is not. The
/// default should move only after it is measured against `cuMemAlloc` on real
/// models.
pub const CUDA_VMM_ENV: &str = "ONNX_GENAI_CUDA_VMM";
/// Opt-in retained-byte bound for the production physical-handle pool.
///
/// When unset, the standalone (plugin) VMM path falls back to the
/// `default_pool_bytes` its caller supplies (issue #956), so scratch reuse is
/// pooled by default whenever the arena is enabled.
pub const CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV: &str = "ONNX_GENAI_CUDA_PHYSICAL_HANDLE_POOL_BYTES";

/// Whether the VMM arena is enabled. Any of `1`/`true`/`yes`/`on`.
pub fn vmm_enabled() -> bool {
    std::env::var(CUDA_VMM_ENV).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn physical_handle_pool_bytes() -> Option<usize> {
    std::env::var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&bytes| bytes > 0)
}

/// Whether VMM allocations use the authority-owned production handle pool.
pub fn production_physical_pool_enabled() -> bool {
    vmm_enabled() && physical_handle_pool_bytes().is_some()
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
/// # Why byte gauges accompany operation counts
///
/// The arena was once installed, logged that it was installed, and committed
/// **zero bytes** for an entire generation (#659). The log line was true and
/// useless: it could not be told apart from a hook that never fired. The byte
/// gauges answer whether memory is really mapped; the operation counts answer
/// whether the driver is being called too often.
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
/// Committed bytes the arena could not move from its private startup ledger
/// into the adopted governor.
///
/// Always zero for the reference governor. A non-zero value means a
/// third-party governor kept the compatibility default for `record_committed`
/// and refused a bookkeeping record for bytes already mapped. Continuing is
/// less disruptive than failing a load that can run, but the number must stay
/// visible because downstream admission now reads a ledger that understates
/// device use.
static GLOBAL_UNACCOUNTED_COMMITTED_BYTES: AtomicU64 = AtomicU64::new(0);

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
    /// `cuMemMap` operations since the process started. Physical handles are
    /// currently one granule each, so this is also the number of granules
    /// mapped.
    pub commits: u64,
    /// Contiguous `cuMemUnmap` runs since the process started. One run may
    /// release several adjacent granules.
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
    /// Committed bytes not recorded in the adopted memory ledger.
    ///
    /// **Anything but zero is a fault.** The memory is mapped, but admission
    /// will not see it through the governor and can therefore over-admit later
    /// work.
    pub unaccounted_committed_bytes: u64,
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
        unaccounted_committed_bytes: GLOBAL_UNACCOUNTED_COMMITTED_BYTES.load(Ordering::Relaxed),
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
    GLOBAL_UNACCOUNTED_COMMITTED_BYTES.store(0, Ordering::Relaxed);
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

fn note_release(bytes: usize) {
    GLOBAL_RELEASES.fetch_add(1, Ordering::Relaxed);
    subtract_counted(&GLOBAL_COMMITTED_BYTES, bytes as u64);
    let now = GLOBAL_COMMITTED_BYTES.load(Ordering::Relaxed);
    let _ = now;
    #[cfg(feature = "tracing")]
    tracing::debug!(
        target: "onnx_runtime::cuda::vmm",
        released_bytes = bytes,
        committed_bytes = now,
        "vmm arena released a contiguous granule run"
    );
}

fn contiguous_granule_runs(granules: impl IntoIterator<Item = usize>) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    for granule in granules {
        match runs.last_mut() {
            Some((_, end)) if *end == granule => *end += 1,
            _ => runs.push((granule, granule + 1)),
        }
    }
    runs
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
    ///
    /// Taken at construction at zero bytes so every commit is a `grow` on an
    /// existing claim. That keeps the governor out of this type: a lease is
    /// owned, a `&dyn` borrow is not, and the execution-provider contract
    /// hands over the latter.
    lease: MemoryLease,
    /// One mapped-attribution allowance owns every granule in this arena.
    mapped_owner: Option<usize>,
}

/// Result of moving an arena's committed-byte claim to a real governor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VmmAdoption {
    /// Bytes recorded in the adopted governor.
    pub recorded_bytes: u64,
    /// Bytes still held only by the arena's private startup ledger.
    pub unaccounted_bytes: u64,
}

/// Result of atomically admitting physical backing for an existing VMM span.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpanCommit {
    /// Physical bytes newly owned by the shared backing pool.
    ///
    /// Mapping a retained pool handle reports zero because that physical
    /// memory was already authority-owned. Creating a handle reports one
    /// allocation granule.
    pub additional_owned_bytes: u64,
    /// Bytes newly mapped into this allocation.
    pub newly_mapped_bytes: u64,
}

/// A pinned, read-only shared prefix: physical granules created **once**
/// through the #740 pool and mappable into many sequences' reservations at zero
/// incremental owned bytes (#777).
///
/// This is the explicit pinned-prefix primitive the #793 probe named as the
/// smallest next increment: a pinned system prompt / tool schema / RAG document
/// whose KV is written once and then read, unchanged, by every concurrent
/// request that shares it. Detection (hashing) and copy-on-write at divergence
/// are deliberately **not** here — this is the allocator primitive only.
///
/// Fill the prefix's KV once through [`device_ptr`](Self::device_ptr), then map
/// it into each sequence with [`CudaVmmAllocator::commit_shared_prefix`]. The
/// physical granules live for the **union** of this owner and every sharer: the
/// pool's shared refcount retains them until the last mapping — this owner's on
/// `Drop`, or any sharer's on deallocation — is gone.
pub struct SharedPrefix {
    reservation: SharedPrefixReservation,
    device: DeviceKey,
    authority: Option<MemoryAuthorityId>,
    requested_bytes: usize,
}

// SAFETY: the inner reservation is `Send + Sync` (its physical handles are
// plain integers and every driver call binds the context first); the other
// fields are trivially so.
unsafe impl Send for SharedPrefix {}
unsafe impl Sync for SharedPrefix {}

impl std::fmt::Debug for SharedPrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedPrefix")
            .field("device", &self.device)
            .field("granules", &self.reservation.granule_count())
            .field("committed_physical_bytes", &self.committed_physical_bytes())
            .field("requested_bytes", &self.requested_bytes)
            .finish()
    }
}

impl SharedPrefix {
    /// Device address of the writable owner window. Fill the prefix's KV here
    /// once, before or after sharing; sharers see it read-only.
    pub fn device_ptr(&self) -> u64 {
        self.reservation.base() as u64
    }

    /// The granule-rounded byte length the prefix actually spans.
    pub fn mapped_bytes(&self) -> usize {
        self.reservation.granule_count() * self.reservation.granularity()
    }

    /// Bytes requested at construction, before granule rounding.
    pub fn requested_bytes(&self) -> usize {
        self.requested_bytes
    }

    /// Physical device bytes this prefix owns — charged **once**, on the owned
    /// axis, no matter how many sequences share it. This is the reported
    /// *physical* cost, never nominal content bytes.
    pub fn committed_physical_bytes(&self) -> u64 {
        self.reservation.owned_bytes()
    }

    fn granule_count(&self) -> usize {
        self.reservation.granule_count()
    }

    fn granularity(&self) -> usize {
        self.reservation.granularity()
    }
}

impl SharedDevicePrefix for SharedPrefix {
    fn device_ptr(&self) -> u64 {
        SharedPrefix::device_ptr(self)
    }

    fn committed_physical_bytes(&self) -> u64 {
        SharedPrefix::committed_physical_bytes(self)
    }

    fn mapped_bytes(&self) -> usize {
        SharedPrefix::mapped_bytes(self)
    }

    fn requested_bytes(&self) -> usize {
        SharedPrefix::requested_bytes(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The accounting outcome of mapping a shared prefix into one sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedPrefixCommit {
    /// Physical bytes newly owned by mapping the prefix here.
    ///
    /// Always **zero**: the prefix's granules were charged once when it was
    /// created, so admitting the Nth sharer costs only its *private* bytes —
    /// the admission arithmetic the multi-request serving path (#745) needs.
    pub additional_owned_bytes: u64,
    /// Physical bytes newly mapped into this sequence's reservation. One
    /// mapping of already-owned physical memory, reported on the mapped axis.
    pub newly_mapped_bytes: u64,
    /// Granules mapped read-only into the sequence.
    pub granules: usize,
}

/// Held open while a CUDA graph capture or replay may touch this allocator's
/// reservations. While any guard is alive, [`CudaVmmAllocator::commit_shared_prefix`]
/// refuses to map — `cuMemMap` inside a capture is not proven replayable.
#[must_use = "the capture is only guarded while this guard is held"]
pub struct SharedPrefixCaptureGuard<'a> {
    allocator: &'a CudaVmmAllocator,
}

impl Drop for SharedPrefixCaptureGuard<'_> {
    fn drop(&mut self) {
        // No assertion here: this is a Drop path, and a panic here aborts the
        // process (STATUS_STACK_BUFFER_OVERRUN on this platform). The decrement
        // is balanced one-for-one with the increment in `enter_graph_capture`,
        // so a plain `fetch_sub` cannot underflow.
        self.allocator.capture_depth.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Device memory carved out of one reserved address range, with physical
/// granules mapped on demand and charged to a [`MemoryGovernor`].
pub struct CudaVmmAllocator {
    backing: CudaVirtualBacking,
    arena: Mutex<Arena>,
    holder: HolderId,
    role: MemoryRole,
    device: DeviceKey,
    /// Non-zero while a CUDA graph capture (or a replay that may be in flight)
    /// is declared open on this allocator.
    ///
    /// Mapping a shared prefix into a reservation issues `cuMemMap`, which
    /// returns `CUDA_SUCCESS` inside a capture but is **not proven replayable**
    /// (#777/#727). Rather than leave that as a comment, this gate makes the
    /// rule enforceable: [`commit_shared_prefix`] refuses while it is non-zero.
    ///
    /// [`commit_shared_prefix`]: CudaVmmAllocator::commit_shared_prefix
    capture_depth: AtomicU64,
}

struct VmmConstruction {
    context: Arc<CudaContext>,
    device: DeviceKey,
    device_ordinal: i32,
    capacity: usize,
    holder: HolderId,
    role: MemoryRole,
    pool_bytes: Option<usize>,
    teardown_synchronizer: Option<crate::virtual_memory::TeardownSynchronizer>,
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
        Self::build(
            VmmConstruction {
                context,
                device,
                device_ordinal,
                capacity,
                holder,
                role,
                pool_bytes: None,
                teardown_synchronizer: None,
            },
            &private,
        )
    }

    /// Construct a standalone arena with a unique device authority.
    ///
    /// Unlike [`detached`](Self::detached), this follows the production pool
    /// option. The private authority keeps independent providers isolated while
    /// allowing all reservations within one provider to share physical handles.
    pub fn standalone(
        context: Arc<CudaContext>,
        device: DeviceKey,
        device_ordinal: i32,
        capacity: usize,
        holder: HolderId,
        role: MemoryRole,
    ) -> Result<Self, MemoryError> {
        let private = onnx_runtime_memory_governor::LedgerGovernor::new(
            onnx_runtime_memory_governor::LeaseLedger::new_for_device(device, u64::MAX, 0, 0),
        );
        Self::build(
            VmmConstruction {
                context,
                device,
                device_ordinal,
                capacity,
                holder,
                role,
                pool_bytes: physical_handle_pool_bytes(),
                teardown_synchronizer: None,
            },
            &private,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn standalone_with_teardown_synchronizer(
        context: Arc<CudaContext>,
        device: DeviceKey,
        device_ordinal: i32,
        capacity: usize,
        holder: HolderId,
        role: MemoryRole,
        teardown_synchronizer: crate::virtual_memory::TeardownSynchronizer,
        default_pool_bytes: Option<usize>,
    ) -> Result<Self, MemoryError> {
        let private = onnx_runtime_memory_governor::LedgerGovernor::new(
            onnx_runtime_memory_governor::LeaseLedger::new_for_device(device, u64::MAX, 0, 0),
        );
        Self::build(
            VmmConstruction {
                context,
                device,
                device_ordinal,
                capacity,
                holder,
                role,
                pool_bytes: physical_handle_pool_bytes().or(default_pool_bytes),
                teardown_synchronizer: Some(teardown_synchronizer),
            },
            &private,
        )
    }

    /// Move this arena's claim onto `governor`, reporting what was recorded.
    ///
    /// The bytes are already committed and mapped, so adoption records an
    /// accomplished fact rather than asking for permission. If a third-party
    /// governor keeps the compatibility default and refuses, the arena keeps
    /// its private claim and records a visible accounting fault instead of
    /// failing a load that can otherwise run.
    pub fn adopt_governor(&self, governor: &dyn MemoryGovernor, holder: HolderId) -> VmmAdoption {
        if let Some(pool) = self.backing.physical_pool() {
            let owned = pool.stats().snapshot().total_owned_bytes;
            return if pool.authority() == governor.authority_id() {
                VmmAdoption {
                    recorded_bytes: 0,
                    unaccounted_bytes: 0,
                }
            } else {
                GLOBAL_UNACCOUNTED_COMMITTED_BYTES.fetch_add(owned, Ordering::Relaxed);
                VmmAdoption {
                    recorded_bytes: 0,
                    unaccounted_bytes: owned,
                }
            };
        }
        let mut arena = self.lock();
        let committed = arena.spans.committed as u64;
        match governor.record_committed(Tier::Device, committed, self.role, holder) {
            Ok(lease) => {
                // Replacing the lease drops the private one, which returns the
                // bytes to a ledger nobody reads. The real governor now holds
                // them, even if doing so puts the tier over its limit.
                arena.lease = lease;
                VmmAdoption {
                    recorded_bytes: committed,
                    unaccounted_bytes: 0,
                }
            }
            Err(_) => {
                GLOBAL_UNACCOUNTED_COMMITTED_BYTES.fetch_add(committed, Ordering::Relaxed);
                VmmAdoption {
                    recorded_bytes: 0,
                    unaccounted_bytes: committed,
                }
            }
        }
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
        Self::build(
            VmmConstruction {
                context,
                device,
                device_ordinal,
                capacity,
                holder,
                role,
                pool_bytes: physical_handle_pool_bytes(),
                teardown_synchronizer: None,
            },
            governor,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_teardown_synchronizer(
        context: Arc<CudaContext>,
        device: DeviceKey,
        device_ordinal: i32,
        capacity: usize,
        governor: &dyn MemoryGovernor,
        holder: HolderId,
        role: MemoryRole,
        teardown_synchronizer: crate::virtual_memory::TeardownSynchronizer,
        default_pool_bytes: Option<usize>,
    ) -> Result<Self, MemoryError> {
        Self::build(
            VmmConstruction {
                context,
                device,
                device_ordinal,
                capacity,
                holder,
                role,
                pool_bytes: physical_handle_pool_bytes().or(default_pool_bytes),
                teardown_synchronizer: Some(teardown_synchronizer),
            },
            governor,
        )
    }

    fn build(
        construction: VmmConstruction,
        governor: &dyn MemoryGovernor,
    ) -> Result<Self, MemoryError> {
        let VmmConstruction {
            context,
            device,
            device_ordinal,
            capacity,
            holder,
            role,
            pool_bytes,
            teardown_synchronizer,
        } = construction;
        let backing = if let Some(pool_bytes) = pool_bytes {
            let pool = PhysicalHandlePool::get_or_create(
                context,
                device_ordinal,
                pool_bytes,
                governor,
                holder,
                role,
            )?;
            CudaVirtualBacking::with_physical_pool(pool)
        } else {
            CudaVirtualBacking::new(context, device_ordinal)
        };
        let backing = match teardown_synchronizer {
            Some(synchronizer) => backing.with_teardown_synchronizer(synchronizer),
            None => backing,
        };
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
            mapped_owner: None,
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
            capture_depth: AtomicU64::new(0),
        })
    }

    /// Stats for this allocator's compatible shared physical pool.
    pub fn physical_pool_stats(&self) -> Option<PhysicalHandlePoolStats> {
        self.backing.physical_pool().map(|pool| pool.stats())
    }

    /// Authority owning the shared pool, when production pooling is enabled.
    pub fn physical_pool_authority(
        &self,
    ) -> Option<onnx_runtime_memory_governor::MemoryAuthorityId> {
        self.backing.physical_pool().map(|pool| pool.authority())
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
        self.claim_granules_limited(arena, granules, u64::MAX, None)
            .map(|(claimed, _)| claimed)
    }

    fn claim_granules_limited(
        &self,
        arena: &mut Arena,
        granules: impl IntoIterator<Item = usize>,
        max_additional_owned_bytes: u64,
        mut capacity: Option<&mut MappedPhysicalCapacityToken>,
    ) -> Result<(BTreeSet<usize>, SpanCommit), MemoryError> {
        let granularity = arena.spans.granularity;
        let claimed = granules.into_iter().collect::<BTreeSet<_>>();
        let mut shared_claimed = BTreeSet::new();
        let mut newly_mapped = Vec::new();
        for &granule in &claimed {
            if arena.spans.granule_refs[granule] > 0 {
                arena.spans.granule_refs[granule] += 1;
                shared_claimed.insert(granule);
            } else {
                newly_mapped.push(granule);
            }
        }
        let mapped_bytes = newly_mapped.len().saturating_mul(granularity);
        if let Err(error) = self.take(arena, mapped_bytes, capacity.as_deref_mut()) {
            self.release_granules(arena, &shared_claimed);
            return Err(error);
        }
        let offsets = newly_mapped
            .iter()
            .map(|granule| granule * granularity)
            .collect::<Vec<_>>();
        let additional_owned = match self.backing.commit_offsets_with_owned_limit_and_capacity(
            &mut arena.reservation,
            &offsets,
            max_additional_owned_bytes,
            capacity,
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.give_back_lease(arena, mapped_bytes);
                self.release_granules(arena, &shared_claimed);
                return Err(match &error {
                    onnx_runtime_virtual_memory::VirtualMemoryError::Os {
                        operation: "growing physical handle pool lease",
                        ..
                    } => MemoryError::CapacityUnavailable {
                        tier: "device",
                        requested: mapped_bytes as u64,
                        available: 0,
                        role: self.role,
                        detail: format!("cuMemMap: {error}"),
                    },
                    _ => invalid(mapped_bytes, format!("cuMemMap: {error}")),
                });
            }
        };
        for granule in newly_mapped {
            arena.spans.granule_refs[granule] = 1;
            arena.spans.committed += granularity;
            note_commit(granularity);
        }
        Ok((
            claimed,
            SpanCommit {
                additional_owned_bytes: additional_owned,
                newly_mapped_bytes: mapped_bytes as u64,
            },
        ))
    }

    fn take(
        &self,
        arena: &mut Arena,
        bytes: usize,
        capacity: Option<&mut MappedPhysicalCapacityToken>,
    ) -> Result<(), MemoryError> {
        if matches!(
            self.backing.physical_memory_accounting(),
            PhysicalMemoryAccounting::Backing { .. }
        ) {
            Ok(())
        } else {
            match capacity {
                Some(capacity) => arena
                    .lease
                    .grow_from_mapped_capacity(capacity, bytes as u64),
                None => arena.lease.grow(bytes as u64),
            }
        }
    }

    fn give_back_lease(&self, arena: &mut Arena, bytes: usize) {
        if matches!(
            self.backing.physical_memory_accounting(),
            PhysicalMemoryAccounting::Buffer
        ) {
            arena.lease.shrink(bytes as u64);
        }
    }

    /// Drop this allocation's claims, unmapping whatever it was the last user
    /// of.
    fn release_granules(&self, arena: &mut Arena, granules: &BTreeSet<usize>) {
        let _ = self.release_granules_report(arena, granules);
    }

    fn release_granules_report(&self, arena: &mut Arena, granules: &BTreeSet<usize>) -> u64 {
        let granularity = arena.spans.granularity;
        let mut unmapped = 0_u64;
        let mut releasable = Vec::new();
        for &granule in granules {
            match arena.spans.granule_refs[granule].checked_sub(1) {
                Some(0) => {
                    releasable.push(granule);
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
        for (start, end) in contiguous_granule_runs(releasable) {
            let bytes = (end - start) * granularity;
            if self
                .backing
                .release(&mut arena.reservation, start * granularity, bytes)
                .is_err()
            {
                continue;
            }
            for granule in start..end {
                arena.spans.granule_refs[granule] = 0;
            }
            arena.spans.committed -= bytes;
            note_release(bytes);
            self.give_back_lease(arena, bytes);
            unmapped = unmapped.saturating_add(bytes as u64);
        }
        unmapped
    }

    fn release_committed_granules(&self, arena: &mut Arena, granules: &BTreeSet<usize>) {
        self.release_granules(arena, granules);
    }

    /// Estimate new authority-owned physical bytes needed to back a live span.
    ///
    /// This is an observation only. Callers making an admission decision must
    /// use [`Self::try_commit_span`] for the race-safe check-and-commit.
    pub fn incremental_owned_bytes_for_span(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        byte_offset: usize,
        bytes: usize,
    ) -> Result<u64, MemoryError> {
        let arena = self.lock();
        let (offset, granules) =
            self.uncommitted_granules(&arena, ptr, allocation_bytes, byte_offset, bytes)?;
        let _ = offset;
        let handles = granules
            .into_iter()
            .filter(|&granule| arena.spans.granule_refs[granule] == 0)
            .count();
        Ok(self.backing.incremental_owned_bytes_for_handles(handles))
    }

    pub fn incremental_mapped_bytes_for_span(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        byte_offset: usize,
        bytes: usize,
    ) -> Result<u64, MemoryError> {
        let arena = self.lock();
        let (_, granules) =
            self.uncommitted_granules(&arena, ptr, allocation_bytes, byte_offset, bytes)?;
        Ok(granules
            .into_iter()
            .filter(|&granule| arena.spans.granule_refs[granule] == 0)
            .count()
            .saturating_mul(arena.spans.granularity) as u64)
    }

    /// Atomically check physical headroom and commit a live allocation span.
    ///
    /// The arena lock fixes the span's granule coverage while the pool checkout
    /// consumes either an already-owned retained handle (zero incremental
    /// bytes) or creates a new handle under the authority lease. Concurrent
    /// allocators can make an estimate stale, but cannot make this transaction
    /// exceed `max_additional_owned_bytes` or the governor's physical limit.
    pub fn try_commit_span(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        byte_offset: usize,
        bytes: usize,
        max_additional_mapped_bytes: u64,
        max_additional_owned_bytes: u64,
    ) -> Result<SpanCommit, MemoryError> {
        let mut arena = self.lock();
        let (offset, granules) =
            self.uncommitted_granules(&arena, ptr, allocation_bytes, byte_offset, bytes)?;
        let handles = granules
            .iter()
            .filter(|&&granule| arena.spans.granule_refs[granule] == 0)
            .count();
        let required = self.backing.incremental_owned_bytes_for_handles(handles);
        let required_mapped = handles.saturating_mul(arena.spans.granularity) as u64;
        if required_mapped > max_additional_mapped_bytes {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "candidate requires {required_mapped} incremental mapped bytes but only \
                     {max_additional_mapped_bytes} bytes of weight-zone headroom are available"
                ),
            ));
        }
        if required > max_additional_owned_bytes {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "candidate requires {required} incremental committed bytes but only \
                     {max_additional_owned_bytes} bytes of physical headroom are available"
                ),
            ));
        }
        let (claimed, commit) =
            self.claim_granules_limited(&mut arena, granules, max_additional_owned_bytes, None)?;
        if let Some(live) = arena.spans.live.get_mut(&offset) {
            live.committed.extend(claimed);
        }
        Ok(commit)
    }

    /// Release a live allocation and report physical bytes actually unmapped.
    pub fn deallocate_span(&self, ptr: NonNull<u8>) -> u64 {
        let mut arena = self.lock();
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        let address = ptr.as_ptr() as usize;
        let Some(offset) = address.checked_sub(base) else {
            return 0;
        };
        let Some(live) = arena.spans.live.remove(&offset) else {
            return 0;
        };
        let unmapped = self.release_granules_report(&mut arena, &live.committed);
        arena.spans.give_back(offset, live.len);
        unmapped
    }

    /// Declare a CUDA graph capture (or replay window) open on this allocator.
    ///
    /// While the returned guard is alive, [`commit_shared_prefix`] refuses to
    /// map a shared prefix, because `cuMemMap` inside a capture returns
    /// `CUDA_SUCCESS` but is not proven replayable (#727/#777). This turns the
    /// "never map inside a captured region" rule into an enforced error rather
    /// than a comment. Nest freely: the gate lifts when the last guard drops.
    ///
    /// [`commit_shared_prefix`]: Self::commit_shared_prefix
    pub fn enter_graph_capture(&self) -> SharedPrefixCaptureGuard<'_> {
        self.capture_depth.fetch_add(1, Ordering::AcqRel);
        SharedPrefixCaptureGuard { allocator: self }
    }

    /// Create a pinned, read-only shared prefix of `bytes`, rounded up to whole
    /// granules and charged **once** on the owned axis.
    ///
    /// The prefix's physical granules are created through the same #740 pool as
    /// every other allocation here — there is no second allocator and no
    /// per-sequence physical reservation. Fill the prefix's KV once through
    /// [`SharedPrefix::device_ptr`], then map it into each sequence with
    /// [`commit_shared_prefix`](Self::commit_shared_prefix).
    ///
    /// Errors, rather than mis-mapping, when this allocator was built without
    /// the production physical-handle pool: a shared prefix is defined by
    /// physical-handle identity across reservations, which only the pool
    /// provides.
    pub fn create_shared_prefix(&self, bytes: usize) -> Result<SharedPrefix, MemoryError> {
        if bytes == 0 {
            return Err(invalid(
                bytes,
                String::from("a shared prefix must cover at least one byte"),
            ));
        }
        let granularity = self.lock().spans.granularity;
        let granule_count = bytes.div_ceil(granularity);
        let reservation = self
            .backing
            .reserve_and_map_shared_prefix(granule_count)
            .map_err(|error| invalid(bytes, format!("shared prefix reservation: {error}")))?;
        Ok(SharedPrefix {
            reservation,
            device: self.device,
            authority: self.physical_pool_authority(),
            requested_bytes: bytes,
        })
    }

    /// Estimate the incremental **owned** physical bytes to admit one more
    /// sharer of `prefix` — zero only when `prefix` genuinely belongs to
    /// this allocator's device and pool authority, because only then are its
    /// granules already owned here. This is the admission-facing statement
    /// (#745): the shared bytes are charged once, so the Nth request costs
    /// only its private continuation. It is an observation;
    /// [`commit_shared_prefix`](Self::commit_shared_prefix) is the operation
    /// and applies the identical device/authority check before mapping.
    ///
    /// A `prefix` from a different device or pool authority is never free to
    /// admit here — `commit_shared_prefix` would reject mapping it, so
    /// treating it as zero-cost would let admission control undercount a
    /// mapping that can never actually happen for free. Its own reported
    /// [`committed_physical_bytes`](SharedPrefix::committed_physical_bytes) is
    /// returned instead, a conservative (non-zero) estimate consistent with
    /// the trait-level [`SharedMapping::incremental_owned_bytes_for_shared_prefix`]
    /// fallback for a foreign prefix type (#1186 Phase 2 review, finding 3).
    pub fn incremental_owned_bytes_for_shared_prefix(&self, prefix: &SharedPrefix) -> u64 {
        let same_device = prefix.device == self.device;
        let same_authority = match (prefix.authority, self.physical_pool_authority()) {
            (Some(prefix_authority), Some(self_authority)) => prefix_authority == self_authority,
            _ => false,
        };
        if same_device && same_authority {
            0
        } else {
            prefix.committed_physical_bytes()
        }
    }

    /// Map `prefix` into the live allocation `ptr` at `byte_offset`,
    /// **read-only**, taking one granule-reference per shared granule.
    ///
    /// The prefix's physical handles are already owned, so this maps existing
    /// memory: it charges **zero** incremental owned bytes and keeps the shared
    /// granules alive until the last sharer (or the prefix owner) leaves. Every
    /// mapped granule is `PROT_READ`, so a mis-targeted store faults loudly
    /// (Q3) instead of corrupting another request's KV.
    ///
    /// Errors — never mis-maps — when:
    ///
    /// * a graph capture is declared open (see [`enter_graph_capture`]);
    /// * `prefix` belongs to a different device or pool authority;
    /// * `byte_offset` is not granule-aligned, or the prefix would not fit
    ///   inside the allocation there;
    /// * `ptr` is not a live allocation of exactly `allocation_bytes`;
    /// * any target granule is already committed (this never overlays live KV).
    ///
    /// [`enter_graph_capture`]: Self::enter_graph_capture
    pub fn commit_shared_prefix(
        &self,
        prefix: &SharedPrefix,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        byte_offset: usize,
    ) -> Result<SharedPrefixCommit, MemoryError> {
        if self.capture_depth.load(Ordering::Acquire) > 0 {
            return Err(invalid(
                allocation_bytes,
                String::from(
                    "cannot map a shared prefix while a graph capture is open: cuMemMap inside a \
                     capture is not proven replayable (#727/#777)",
                ),
            ));
        }
        if prefix.device != self.device {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "shared prefix belongs to device {:?} but this allocator serves {:?}",
                    prefix.device, self.device
                ),
            ));
        }
        match (prefix.authority, self.physical_pool_authority()) {
            (Some(prefix_authority), Some(self_authority))
                if prefix_authority == self_authority => {}
            (None, _) | (_, None) => {
                return Err(invalid(
                    allocation_bytes,
                    String::from(
                        "shared prefix requires the production physical-handle pool on both the \
                         prefix and the committing allocator",
                    ),
                ));
            }
            _ => {
                return Err(invalid(
                    allocation_bytes,
                    String::from(
                        "shared prefix was created under a different pool authority than this \
                         allocator",
                    ),
                ));
            }
        }

        let mut arena = self.lock();
        let granularity = arena.spans.granularity;
        if !byte_offset.is_multiple_of(granularity) {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "shared prefix offset {byte_offset} is not granule-aligned ({granularity}); a \
                     shared prefix maps whole physical granules"
                ),
            ));
        }
        let prefix_bytes = prefix.granule_count() * prefix.granularity();
        if prefix.granularity() != granularity {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "shared prefix granularity {} does not match this arena's {granularity}",
                    prefix.granularity()
                ),
            ));
        }
        let end = byte_offset.checked_add(prefix_bytes).ok_or_else(|| {
            invalid(
                allocation_bytes,
                format!("shared prefix at offset {byte_offset} overflows the address space"),
            )
        })?;
        if end > allocation_bytes {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "shared prefix {byte_offset}..{end} exceeds the allocation of \
                     {allocation_bytes} bytes"
                ),
            ));
        }

        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        let address = ptr.as_ptr() as usize;
        let Some(offset) = address.checked_sub(base) else {
            return Err(invalid(
                allocation_bytes,
                String::from("shared prefix pointer is below the VMM arena reservation"),
            ));
        };
        let Some(live) = arena.spans.live.get(&offset) else {
            return Err(invalid(
                allocation_bytes,
                String::from("shared prefix pointer is not a live VMM allocation"),
            ));
        };
        if live.len != allocation_bytes {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "shared prefix allocation size {allocation_bytes} does not match live VMM \
                     allocation size {}",
                    live.len
                ),
            ));
        }

        if !(offset + byte_offset).is_multiple_of(granularity) {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "shared prefix target address (allocation offset {offset} + {byte_offset}) is \
                     not granule-aligned ({granularity}); allocate the sequence at granule \
                     alignment so its prefix maps whole physical granules"
                ),
            ));
        }

        let first_granule = (offset + byte_offset) / granularity;
        let granule_count = prefix.granule_count();
        for i in 0..granule_count {
            let granule = first_granule + i;
            if arena.spans.granule_refs[granule] != 0 {
                return Err(invalid(
                    allocation_bytes,
                    format!(
                        "shared prefix would overlay granule {granule}, which is already \
                         committed; a shared prefix maps only into an uncommitted region"
                    ),
                ));
            }
        }

        // Map every granule read-only, unwinding cleanly on any failure so a
        // partial map never lingers.
        let mut mapped: Vec<usize> = Vec::with_capacity(granule_count);
        for i in 0..granule_count {
            let granule = first_granule + i;
            let handle = prefix.reservation.handle(i).ok_or_else(|| {
                invalid(
                    allocation_bytes,
                    format!("shared prefix has no handle for granule {i}"),
                )
            })?;
            match self.backing.map_shared_prefix_readonly(
                &mut arena.reservation,
                granule * granularity,
                handle,
            ) {
                Ok(()) => mapped.push(granule),
                Err(error) => {
                    for &done in &mapped {
                        // Unmap what this call mapped; the handle returns to the
                        // pool's shared refcount, not to `available`, so other
                        // sharers are unaffected.
                        let _ = self.backing.release(
                            &mut arena.reservation,
                            done * granularity,
                            granularity,
                        );
                    }
                    return Err(invalid(
                        allocation_bytes,
                        format!("shared prefix mapping: {error}"),
                    ));
                }
            }
        }

        for &granule in &mapped {
            arena.spans.granule_refs[granule] = 1;
            arena.spans.committed += granularity;
            note_commit(granularity);
        }
        if let Some(live) = arena.spans.live.get_mut(&offset) {
            live.committed.extend(mapped.iter().copied());
        }

        Ok(SharedPrefixCommit {
            additional_owned_bytes: 0,
            newly_mapped_bytes: (granule_count * granularity) as u64,
            granules: granule_count,
        })
    }

    fn uncommitted_granules(
        &self,
        arena: &Arena,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        byte_offset: usize,
        bytes: usize,
    ) -> Result<(usize, Vec<usize>), MemoryError> {
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
        if live.len != allocation_bytes {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "commit allocation size {allocation_bytes} does not match live VMM allocation size {}",
                    live.len
                ),
            ));
        }
        if bytes == 0 {
            return Ok((offset, Vec::new()));
        }
        let absolute = offset + byte_offset..offset + end;
        let granules = arena
            .spans
            .granules(absolute.start, absolute.len())
            .filter(|granule| !live.committed.contains(granule))
            .collect();
        Ok((offset, granules))
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

fn batch_granule_references(
    by_allocation: &BTreeMap<usize, BTreeSet<usize>>,
) -> BTreeMap<usize, u32> {
    let mut references = BTreeMap::<usize, u32>::new();
    for granules in by_allocation.values() {
        for &granule in granules {
            *references.entry(granule).or_default() += 1;
        }
    }
    references
}

impl CudaVmmAllocator {
    fn commit_allocation_ranges_inner(
        &self,
        ranges: &[AllocationCommitRange],
        mut capacity: Option<&mut MappedPhysicalCapacityToken>,
    ) -> Result<SpanCommit, MemoryError> {
        let mut arena = self.lock();
        if let Some(capacity) = capacity.as_deref()
            && let Some(owner) = arena.mapped_owner
            && owner != capacity.owner_id()
        {
            return Err(invalid(
                ranges.iter().map(|range| range.bytes).sum(),
                "mapped capacity token belongs to a different allowance than this arena"
                    .to_string(),
            ));
        }
        let mut by_allocation = BTreeMap::<usize, BTreeSet<usize>>::new();
        for range in ranges {
            let (offset, granules) = self.uncommitted_granules(
                &arena,
                range.ptr,
                range.allocation_bytes,
                range.offset,
                range.bytes,
            )?;
            by_allocation.entry(offset).or_default().extend(granules);
        }
        let references = batch_granule_references(&by_allocation);
        let union = references.keys().copied().collect::<Vec<_>>();
        let (claimed, commit) = match capacity.as_deref_mut() {
            Some(capacity) => {
                self.claim_granules_limited(&mut arena, union, u64::MAX, Some(capacity))?
            }
            None => self.claim_granules_limited(&mut arena, union, u64::MAX, None)?,
        };
        for (&granule, &count) in &references {
            if count > 1 {
                arena.spans.granule_refs[granule] =
                    arena.spans.granule_refs[granule].saturating_add(count - 1);
            }
        }
        debug_assert_eq!(claimed.len(), references.len());
        for (offset, granules) in by_allocation {
            if let Some(live) = arena.spans.live.get_mut(&offset) {
                live.committed.extend(granules);
            }
        }
        if let Some(capacity) = capacity {
            arena.mapped_owner.get_or_insert(capacity.owner_id());
        }
        Ok(commit)
    }

    /// Reserve one allocation while committing only the byte ranges the
    /// caller names as immediately live, charging the newly-mapped bytes to
    /// `capacity` atomically with claiming the underlying granules.
    ///
    /// # Why this is inherent rather than part of a trait
    ///
    /// [`MappedPhysicalCapacityToken`] is a `onnx-runtime-memory-governor`
    /// type — the governor's atomic capacity-charging seam, not part of the
    /// dyn-safe allocator contract every backend implements. Putting it on a
    /// trait would force every [`crate::device_allocator::CudaDeviceAllocator`]-
    /// like eager allocator and the CPU allocator to answer for a governor
    /// concept they have no way to honor. `onnx-runtime-ep-cuda` already holds
    /// this allocator at its concrete type (`vmm: OnceLock<Arc<CudaVmmAllocator>>`),
    /// so it calls this directly rather than through `&dyn DeviceAllocator`/
    /// `&dyn VirtualBacking`.
    pub fn allocate_committed_with_capacity(
        &self,
        bytes: usize,
        align: usize,
        committed_ranges: &[std::ops::Range<usize>],
        capacity: &mut MappedPhysicalCapacityToken,
    ) -> Result<onnx_runtime_memory_governor::MappedAllocation<NonNull<u8>>, MemoryError> {
        if capacity.role() != self.role {
            return Err(invalid(
                bytes,
                format!(
                    "mapped allocation role {:?} does not match arena zone {:?}",
                    capacity.role(),
                    self.role
                ),
            ));
        }
        let ptr = self.allocate_committed(bytes, align, &[])?;
        let ranges = committed_ranges
            .iter()
            .map(|range| AllocationCommitRange {
                ptr,
                allocation_bytes: bytes,
                align,
                offset: range.start,
                bytes: range.len(),
            })
            .collect::<Vec<_>>();
        let commit = match self.commit_allocation_ranges_inner(&ranges, Some(capacity)) {
            Ok(commit) => commit,
            Err(error) => {
                // SAFETY: this is the exact live allocation returned above and
                // it has not escaped to the caller.
                unsafe { self.deallocate(ptr, bytes, align) };
                return Err(error);
            }
        };
        Ok(onnx_runtime_memory_governor::MappedAllocation {
            allocation: ptr,
            newly_mapped_bytes: commit.newly_mapped_bytes,
        })
    }

    /// Commit several allocation ranges as one allocator transaction, charging
    /// the newly-mapped bytes to `capacity` atomically with claiming the
    /// underlying granules. See
    /// [`allocate_committed_with_capacity`](Self::allocate_committed_with_capacity)
    /// for why this is inherent rather than part of a trait.
    pub fn commit_allocation_ranges_with_capacity(
        &self,
        ranges: &[AllocationCommitRange],
        capacity: &mut MappedPhysicalCapacityToken,
    ) -> Result<u64, MemoryError> {
        self.commit_allocation_ranges_inner(ranges, Some(capacity))
            .map(|commit| commit.newly_mapped_bytes)
    }
}

// SAFETY: every pointer handed out lies inside this allocator's reservation,
// in a span removed from the free list and recorded in `live`, so no two live
// allocations overlap. The granules under a span stay mapped while its
// reference count is non-zero, so the memory remains valid until `deallocate`.
// `device` names the CUDA device the reservation belongs to.
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

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _bytes: usize, _align: usize) {
        let _ = self.deallocate_span(ptr);
    }

    unsafe fn deallocate_with_unmapped(
        &self,
        ptr: NonNull<u8>,
        _bytes: usize,
        _align: usize,
    ) -> u64 {
        self.deallocate_span(ptr)
    }

    fn device(&self) -> DeviceKey {
        self.device
    }

    /// `Some(self)`: a VMM arena always separates reservation from commit, so
    /// it always has this capability. See [`impl MemoryVirtualBacking`] below.
    ///
    /// [`impl MemoryVirtualBacking`]: #impl-VirtualBacking-for-CudaVmmAllocator
    fn as_virtual_backing(&self) -> Option<&dyn MemoryVirtualBacking> {
        Some(self)
    }

    /// `Some(self)` only when this arena actually owns a production
    /// physical-handle pool. Sharing is defined by pool identity across
    /// reservations (see [`impl SharedMapping`] below and
    /// [`physical_pool_authority`](Self::physical_pool_authority)): a
    /// `detached`/pool-less arena has no authority to share against, so
    /// `create_shared_prefix` can never succeed for it even though it is
    /// otherwise a perfectly ordinary VMM allocator. Advertising `Some` here
    /// regardless would let a caller discover a capability that always fails
    /// on first use — the exact "successful-looking no-op" this split exists
    /// to remove, just moved from the call to the discovery (#1186 Phase 2
    /// review).
    ///
    /// [`impl SharedMapping`]: #impl-SharedMapping-for-CudaVmmAllocator
    fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
        self.backing.physical_pool().is_some().then_some(self)
    }
}

// SAFETY: every pointer handed to these methods lies inside this allocator's
// reservation, in a span recorded in `live` (checked by address arithmetic
// against `arena.spans.live` before any granule is touched).
impl MemoryVirtualBacking for CudaVmmAllocator {
    fn device(&self) -> DeviceKey {
        self.device
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

    /// # Safety
    ///
    /// Delegates to [`deallocate_span`](Self::deallocate_span), which imposes
    /// the same requirement: `ptr` must be a still-live pointer this
    /// allocator's `allocate`/`allocate_committed` produced. Reached only
    /// through this capability reference — never through the base
    /// [`DeviceAllocator::deallocate_with_unmapped`] — so the mechanism that
    /// produced a `VirtualBacking`-obtained pointer is always the mechanism
    /// that releases it (#1186 Phase 2 review, finding 1).
    unsafe fn deallocate_committed(
        &self,
        ptr: NonNull<u8>,
        _allocation_bytes: usize,
        _align: usize,
    ) -> u64 {
        self.deallocate_span(ptr)
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

    fn commit_allocation_ranges(
        &self,
        ranges: &[AllocationCommitRange],
    ) -> Result<(), MemoryError> {
        self.commit_allocation_ranges_inner(ranges, None)
            .map(|_| ())
    }

    fn mapped_bytes_for_allocation_ranges(
        &self,
        ranges: &[AllocationCommitRange],
    ) -> Result<u64, MemoryError> {
        let arena = self.lock();
        let mut identities = BTreeSet::new();
        for range in ranges {
            let (_, granules) = self.uncommitted_granules(
                &arena,
                range.ptr,
                range.allocation_bytes,
                range.offset,
                range.bytes,
            )?;
            identities.extend(
                granules
                    .into_iter()
                    .filter(|&granule| arena.spans.granule_refs[granule] == 0),
            );
        }
        Ok(identities.len().saturating_mul(arena.spans.granularity) as u64)
    }

    fn mapped_bytes_for_allocation(&self, bytes: usize, _align: usize) -> Result<u64, MemoryError> {
        let granularity = self.lock().spans.granularity;
        Ok(round_up(granularity, bytes) as u64)
    }

    fn decommit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        _align: usize,
        byte_offset: usize,
        bytes: usize,
    ) -> Result<u64, MemoryError> {
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
            return Ok(0);
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
        let unmapped = self.release_granules_report(&mut arena, &releasable);
        if let Some(live) = arena.spans.live.get_mut(&offset) {
            for granule in releasable {
                live.committed.remove(&granule);
            }
        }
        Ok(unmapped)
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl SharedMapping for CudaVmmAllocator {
    fn device(&self) -> DeviceKey {
        self.device
    }

    fn create_shared_prefix(
        &self,
        bytes: usize,
    ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
        let prefix = CudaVmmAllocator::create_shared_prefix(self, bytes)?;
        Ok(Box::new(prefix))
    }

    fn incremental_owned_bytes_for_shared_prefix(&self, prefix: &dyn SharedDevicePrefix) -> u64 {
        // A prefix from another allocator kind cannot be mapped here, so its
        // incremental owned cost is not this allocator's to estimate; treat the
        // unmappable case as "not free" so a caller never admits against a
        // prefix `commit_shared_prefix` will refuse.
        match prefix.as_any().downcast_ref::<SharedPrefix>() {
            Some(prefix) => {
                CudaVmmAllocator::incremental_owned_bytes_for_shared_prefix(self, prefix)
            }
            None => prefix.committed_physical_bytes(),
        }
    }

    fn commit_shared_prefix(
        &self,
        prefix: &dyn SharedDevicePrefix,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        byte_offset: usize,
    ) -> Result<SharedPrefixCommitInfo, MemoryError> {
        let prefix = prefix.as_any().downcast_ref::<SharedPrefix>().ok_or_else(|| {
            invalid(
                allocation_bytes,
                String::from(
                    "shared prefix was not created by a CUDA VMM allocator; it cannot be mapped \
                     here",
                ),
            )
        })?;
        let commit = CudaVmmAllocator::commit_shared_prefix(
            self,
            prefix,
            ptr,
            allocation_bytes,
            byte_offset,
        )?;
        Ok(SharedPrefixCommitInfo {
            additional_owned_bytes: commit.additional_owned_bytes,
            newly_mapped_bytes: commit.newly_mapped_bytes,
            granules: commit.granules,
        })
    }

    /// `deallocate_with_unmapped` (the `DeviceAllocator` method) never
    /// actually reaches this: `as_virtual_backing` is always `Some` for this
    /// allocator (a VMM arena always separates reservation from commit), so
    /// release always goes through `VirtualBacking::deallocate_committed`
    /// (`impl DeviceAllocator for CudaVmmAllocator::deallocate_with_unmapped`
    /// overrides the base default entirely, above). This exists for a caller
    /// that reaches release accounting through `as_shared_mapping` directly,
    /// bypassing that override (#1186 Phase 2 review, round 3 finding 2), and
    /// for that caller must still answer correctly.
    ///
    /// This allocator's granule tracking does not record whether a committed
    /// granule became committed through an ordinary
    /// [`commit_allocation_range`](MemoryVirtualBacking::commit_allocation_range)
    /// or through [`commit_shared_prefix`](Self::commit_shared_prefix) — both
    /// mark the same granule "committed" in the same span, since either way
    /// the physical page is mapped and the address range is live. So the
    /// conservative, always-correct answer for "how many of `ptr`'s bytes did
    /// a shared mapping contribute" is this allocation's whole committed
    /// footprint: exactly what `deallocate_committed` would itself report if
    /// invoked instead, which is what every real caller in this codebase
    /// does.
    fn allocation_shared_mapped_bytes(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        align: usize,
    ) -> u64 {
        <Self as MemoryVirtualBacking>::allocation_committed_bytes(self, ptr, allocation_bytes, align)
            as u64
    }

    fn as_any(&self) -> &dyn Any {
        self
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

    #[test]
    fn weight_page_release_churn_is_one_run_per_page_not_one_per_granule() {
        const TOKENS: usize = 16;
        const GRANULES_PER_PAGE: usize = 10;

        let releases = (0..TOKENS)
            .map(|token| {
                let first = token * GRANULES_PER_PAGE;
                contiguous_granule_runs(first..first + GRANULES_PER_PAGE).len()
            })
            .sum::<usize>();

        assert_eq!(releases, TOKENS);
        assert!(
            releases < TOKENS * GRANULES_PER_PAGE,
            "a contiguous weight page must not regress to one release call per granule"
        );
    }

    #[test]
    fn batched_ranges_union_shared_granule_identity_once() {
        let by_allocation =
            BTreeMap::from([(0, BTreeSet::from([7, 8])), (4096, BTreeSet::from([8, 9]))]);
        let references = batch_granule_references(&by_allocation);

        assert_eq!(
            references.keys().copied().collect::<Vec<_>>(),
            vec![7, 8, 9]
        );
        assert_eq!(references[&8], 2, "both allocations retain a reference");
        assert_eq!(
            references.len(),
            3,
            "shared granule 8 is one physical admission identity"
        );
    }

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

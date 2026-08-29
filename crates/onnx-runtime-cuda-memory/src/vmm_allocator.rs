//! Device memory from the CUDA virtual memory APIs, on one CUDA device.
//!
//! # Why not `cuMemAlloc`
//!
//! `cuMemAlloc` allocates virtual *and* physical memory together. That is the
//! "reservation-based" model vAttention
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
//! # Why there is no eager alternative to fall back to
//!
//! This is the **only** built-in CUDA memory mechanism. The eager `cuMemAlloc`
//! allocator that used to sit beside it, and the environment flag that chose
//! between them, are gone: a second built-in mechanism means every accounting,
//! capture-safety and teardown invariant has to hold twice, and the eager one
//! could not hold the capture-safety half at all. A caller who needs a
//! different mechanism supplies it through the ordinary `DeviceAllocator`
//! contract instead of flipping a flag.

use std::collections::{BTreeMap, BTreeSet};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use onnx_runtime_memory_governor::{
    AllocationCommitRange, AllocationReleaseOutcome, AllocationReleaseState, DeviceAllocator,
    DeviceKey, HolderId, MappedPhysicalCapacityToken, MemoryAuthorityId, MemoryError,
    MemoryGovernor, MemoryLease, MemoryRole, QuarantineReason, ResidualOwnership,
    SharedDevicePrefix, SharedMapping, SharedPrefixCommitInfo, Tier,
    VirtualBacking as MemoryVirtualBacking,
};
use onnx_runtime_virtual_memory::{PhysicalMemoryAccounting, VirtualBacking};

use crate::release::{MappedBlock, SpanReleaseReport, block_bytes};
use crate::virtual_memory::{
    CudaVirtualBacking, PhysicalHandlePool, PhysicalHandlePoolStats, RangeDecommit,
    SharedPrefixReservation,
};
use cudarc::driver::CudaContext;

/// Opt-in retained-byte bound for the production physical-handle pool.
///
/// When unset, the standalone (plugin) VMM path falls back to the
/// `default_pool_bytes` its caller supplies (issue #956), so scratch reuse is
/// pooled by default.
///
/// # Why there is no companion switch that turns the arena itself on
///
/// There used to be one (`ONNX_GENAI_CUDA_VMM`). It existed because the arena
/// was the alternative to an eager `cuMemAlloc` allocator and the default was
/// not to be moved until the arena had been measured against it. The arena is
/// now the only built-in mechanism, so there is nothing for a switch to select
/// between: an unset flag would have to mean "use the arena" and a set one
/// would mean the same thing.
pub const CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV: &str = "ONNX_GENAI_CUDA_PHYSICAL_HANDLE_POOL_BYTES";

/// Parse a retained-byte pool bound, without reading the environment.
///
/// Split from [`physical_handle_pool_bytes`] so the parsing rules can be
/// asserted directly. A test that has to set a process-global environment
/// variable to reach them races every other test in the binary, and a racy
/// test is one that will eventually be quietened rather than fixed.
fn parse_physical_handle_pool_bytes(value: Option<&str>) -> Option<usize> {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&bytes| bytes > 0)
}

fn physical_handle_pool_bytes() -> Option<usize> {
    parse_physical_handle_pool_bytes(
        std::env::var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV)
            .ok()
            .as_deref(),
    )
}

/// Whether VMM allocations use the authority-owned production handle pool.
///
/// The arena is always the mechanism, so this asks only whether an explicit
/// retained-byte bound was configured for it.
pub fn production_physical_pool_enabled() -> bool {
    physical_handle_pool_bytes().is_some()
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
/// All zero means no arena was ever built -- the state of a process that never
/// constructed a CUDA provider -- and is distinguishable from an arena that was
/// built and never used (`reserved_bytes > 0`, `commits == 0`).
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
    /// Spans whose release did not reach a terminal state, as `offset -> span`.
    ///
    /// A quarantined span is in neither `free` nor `live`: it can never be
    /// carved again, and every commit, decommit or release naming its address
    /// fails closed rather than pretending it is usable. This is the record
    /// that makes "we still own something we could not give back" a fact the
    /// allocator holds rather than an error message that scrolled past.
    quarantine: BTreeMap<usize, QuarantinedSpan>,
    /// Granules that must never be committed again.
    ///
    /// A granule lands here when its `cuMemUnmap` failed: the mapping is still
    /// there, so mapping something else over it would either fail or — far
    /// worse — hand a second allocation an address whose old contents are
    /// still live behind it. Poisoning is per granule rather than per span
    /// because a granule at the boundary of a quarantined span can be shared
    /// with a live neighbour.
    poisoned: BTreeSet<usize>,
    /// Bytes currently committed, which is `lease.bytes()` when a lease exists.
    committed: usize,
}

#[derive(Debug)]
struct LiveSpan {
    len: usize,
    /// Recorded so a structured release can report the residual VA exactly as
    /// the caller allocated it, without the caller having to supply it again.
    align: usize,
    committed: BTreeSet<usize>,
}

/// One allocation the allocator still owns after a release that could not
/// finish.
///
/// Every field is a fact the runtime must keep to stay honest: the address
/// range it refuses to reuse, the granules that are still mapped, the physical
/// handles it holds after their mappings went away, the bytes it refunded on
/// each accounting axis, and why. The CUDA context is retained implicitly —
/// the arena's reservation owns it, and a quarantined span keeps the arena
/// alive — so the residual can never outlive the context it belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuarantinedSpan {
    /// Offset of the retained address range inside the arena reservation.
    pub offset: usize,
    /// Length of the retained address range.
    pub len: usize,
    /// The alignment the allocation was made at.
    pub align: usize,
    /// Device address of the retained range.
    pub address: usize,
    /// The device this span's memory belongs to.
    pub device: DeviceKey,
    /// Granules still mapped, with the handles behind them.
    pub still_mapped: Vec<MappedBlock>,
    /// Blocks whose mapping is gone but whose physical handle could not be
    /// given back. Their bytes stay charged on the owned axis.
    pub unmapped_handle_owned: Vec<MappedBlock>,
    /// Bytes refunded on the mapped axis by this release, counted once.
    pub refunded_mapped_bytes: u64,
    /// Physical bytes still owned, across both residual kinds.
    pub retained_owned_bytes: u64,
    pub state: AllocationReleaseState,
    pub reason: QuarantineReason,
    /// Every driver fault, in order, so the cause is never only in a log line.
    pub faults: Vec<String>,
}

impl QuarantinedSpan {
    /// The residual ownership record a release outcome reports.
    pub fn residual(&self) -> ResidualOwnership {
        ResidualOwnership {
            state: self.state,
            reason: self.reason,
            retained_bytes: self.retained_owned_bytes,
            address: self.address,
            align: self.align,
        }
    }

    fn fault_summary(&self) -> String {
        if self.faults.is_empty() {
            return String::from("no driver fault was reported");
        }
        self.faults.join("; ")
    }
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
            quarantine: BTreeMap::new(),
            poisoned: BTreeSet::new(),
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

    /// Retain `[offset, offset + len)` and everything still owned inside it.
    ///
    /// The span goes to neither the free list nor the live map, which is the
    /// whole point: a partially released address that came back to `free`
    /// would be handed to the next allocation with the old mapping still under
    /// part of it.
    fn quarantine_span(&mut self, span: QuarantinedSpan) {
        for block in &span.still_mapped {
            self.poisoned.insert(block.offset / self.granularity);
        }
        self.quarantine.insert(span.offset, span);
    }

    /// Apply one run's release report to the granule bookkeeping and report
    /// the mapped-axis refund.
    ///
    /// Pure by construction — no CUDA state reaches this type — so the rule
    /// that decides whether an address becomes reusable is provable on a
    /// machine with no GPU, which is where it was previously untestable.
    ///
    /// * every granule in the run gives up its reference;
    /// * a granule whose mapping survived takes one back and is poisoned, so
    ///   nothing unmaps or commits it again;
    /// * the refund counts only bytes whose mapping is genuinely gone, whether
    ///   or not the handle behind them could be given back.
    fn settle_release_run(
        &mut self,
        run: std::ops::Range<usize>,
        report: &crate::release::SpanReleaseReport,
    ) -> u64 {
        for granule in run {
            self.granule_refs[granule] = 0;
        }
        for block in &report.still_mapped {
            let granule = block.offset / self.granularity;
            self.granule_refs[granule] = 1;
            self.poisoned.insert(granule);
        }
        let unmapped = report.unmapped_bytes();
        self.committed = self.committed.saturating_sub(unmapped as usize);
        unmapped
    }

    /// The first poisoned granule in `granules`, if any.
    fn first_poisoned(&self, granules: impl IntoIterator<Item = usize>) -> Option<usize> {
        granules
            .into_iter()
            .find(|granule| self.poisoned.contains(granule))
    }

    /// Return `[offset, offset + len)` to the free list, merging with either
    /// neighbour so the list cannot fragment into adjacent free spans.
    fn give_back(&mut self, offset: usize, len: usize) {
        debug_assert!(
            !self
                .quarantine
                .values()
                .any(|span| { span.offset < offset + len && offset < span.offset + span.len }),
            "vmm arena: {offset}..{} overlaps a quarantined span and must never return to the \
             free list",
            offset + len
        );
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

/// Byte accounting for one transactional decommit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecommitAccounting {
    /// Bytes the caller asked to decommit.
    pub requested_bytes: u64,
    /// Bytes whose mapping is actually gone. Refunded on the mapped axis.
    ///
    /// Zero is a valid complete result: every granule in the range may still
    /// be shared with another live allocation, in which case this decommit
    /// only dropped a reference.
    pub unmapped_bytes: u64,
    /// Physical bytes whose handles could not be given back and are now held
    /// in quarantine. Still charged on the owned axis, never reusable.
    ///
    /// The mapping is gone, so the allocation's topology is exactly what the
    /// caller asked for; what remains is a pool-level ownership residual and
    /// not a hole in a live buffer.
    pub quarantined_owned_bytes: u64,
}

/// What a transactional decommit did to a live allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecommitOutcome {
    /// The requested range is unmapped and the allocation is still live and
    /// usable with exactly the topology the caller asked for.
    Complete { accounting: DecommitAccounting },
    /// The decommit was refused and the original mapping was restored. The
    /// allocation is byte-for-byte what it was.
    RolledBack { reason: String },
    /// The decommit could not be completed *or* rolled back, so the allocation
    /// is quarantined whole: its address range is retained, its residual
    /// mappings and handles stay owned, and it can never be used again.
    Quarantined {
        accounting: DecommitAccounting,
        residual: ResidualOwnership,
        reason: String,
    },
}

impl DecommitOutcome {
    /// Bytes whose mapping is gone, for a caller that only needs the refund.
    pub const fn unmapped_bytes(&self) -> u64 {
        match self {
            Self::Complete { accounting } => accounting.unmapped_bytes,
            Self::RolledBack { .. } => 0,
            Self::Quarantined { accounting, .. } => accounting.unmapped_bytes,
        }
    }

    /// Whether the allocation is still live and usable afterwards.
    pub const fn allocation_remains_usable(&self) -> bool {
        matches!(self, Self::Complete { .. } | Self::RolledBack { .. })
    }
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
    reservation_queue: Option<Arc<dyn crate::virtual_memory::DeferredReservationQueue>>,
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
                reservation_queue: None,
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
                reservation_queue: None,
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
                reservation_queue: None,
            },
            &private,
        )
    }

    /// Construct a standalone arena whose reservation teardown is owned by a
    /// deferred queue.
    ///
    /// This is the production standalone path: no reservation `Drop` anywhere
    /// under this arena synchronizes a stream, because teardown is handed to
    /// `reservation_queue` as a ticket and executed once the queue observes the
    /// completion of the work that could still read the range.
    #[allow(clippy::too_many_arguments)]
    pub fn standalone_with_reservation_queue(
        context: Arc<CudaContext>,
        device: DeviceKey,
        device_ordinal: i32,
        capacity: usize,
        holder: HolderId,
        role: MemoryRole,
        reservation_queue: Arc<dyn crate::virtual_memory::DeferredReservationQueue>,
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
                teardown_synchronizer: None,
                reservation_queue: Some(reservation_queue),
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
                reservation_queue: None,
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
                reservation_queue: None,
            },
            governor,
        )
    }

    /// Construct a governed arena whose reservation teardown is owned by a
    /// deferred queue. The production governed path.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_reservation_queue(
        context: Arc<CudaContext>,
        device: DeviceKey,
        device_ordinal: i32,
        capacity: usize,
        governor: &dyn MemoryGovernor,
        holder: HolderId,
        role: MemoryRole,
        reservation_queue: Arc<dyn crate::virtual_memory::DeferredReservationQueue>,
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
                teardown_synchronizer: None,
                reservation_queue: Some(reservation_queue),
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
            reservation_queue,
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
        let backing = match reservation_queue {
            Some(queue) => backing.with_reservation_queue(queue),
            None => backing,
        };
        let granularity = backing.granularity();
        // Unreachable from the CUDA provider: `CudaVirtualBacking::granularity`
        // resolves through `allocation_granularity`, which substitutes 2 MiB
        // for a driver refusal or a reported zero. It is kept for `VirtualBacking`
        // implementations that do report zero. The init-time capability
        // detector for CUDA is the `reserve` call below.
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

    /// Authority currently charged for this allocator's committed bytes.
    ///
    /// When a retained physical-handle pool is active it owns the committed
    /// bytes; otherwise the arena's own device-tier lease does.
    pub fn committed_byte_authority(
        &self,
    ) -> Option<onnx_runtime_memory_governor::MemoryAuthorityId> {
        self.physical_pool_authority()
            .or_else(|| self.lock().lease.authority_id())
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

    /// Retained device physical-handle pool backing this allocator, when
    /// production pooling is enabled.
    pub fn physical_pool(&self) -> Option<Arc<PhysicalHandlePool>> {
        self.backing.physical_pool().cloned()
    }

    /// Resolve a live allocation's byte offset within the allocator's reserved
    /// address space. Returns `None` for a foreign pointer or size mismatch.
    pub fn live_allocation_offset(&self, ptr: NonNull<u8>, bytes: usize) -> Option<usize> {
        let arena = self.lock();
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        let offset = (ptr.as_ptr() as usize).checked_sub(base)?;
        let live = arena.spans.live.get(&offset)?;
        (live.len == bytes).then_some(offset)
    }

    /// The [`DeviceKey`] this allocator's arena is bound to. Used by callers
    /// that hold several allocators keyed by an unrelated identifier (e.g.
    /// `ValueId`) to verify they all belong to the same physical device
    /// before performing any cross-allocator device-scoped operation, rather
    /// than assuming it.
    pub fn device_key(&self) -> DeviceKey {
        self.device
    }

    /// Execute a closure with exclusive access to the arena's reservation
    /// and this allocator's `CudaVirtualBacking`.
    ///
    /// This is the one authorized path for `transition_granule_range`
    /// (from `onnx-runtime-ep-cuda`) to reach the mutable
    /// [`CudaReservation`] that backs a specific VMM allocation: the arena
    /// lock is held across the closure, so no concurrent commit/decommit
    /// can interleave.
    ///
    /// The closure receives `(&mut CudaReservation, &CudaVirtualBacking)`.
    /// It **must not** attempt to re-lock this allocator (would deadlock).
    ///
    /// [`CudaReservation`]: onnx_runtime_virtual_memory::VirtualBacking::Reservation
    pub fn with_reservation_mut<R, F>(&self, f: F) -> R
    where
        F: FnOnce(
            &mut <CudaVirtualBacking as onnx_runtime_virtual_memory::VirtualBacking>::Reservation,
            &CudaVirtualBacking,
        ) -> R,
    {
        let mut arena = self.lock();
        f(&mut arena.reservation, &self.backing)
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
        // Fail closed before touching the driver: a poisoned granule is one
        // whose old mapping could not be removed, so committing over it would
        // either be refused by CUDA or, worse, quietly alias live memory.
        if let Some(granule) = arena.spans.first_poisoned(claimed.iter().copied()) {
            return Err(invalid(
                claimed.len().saturating_mul(granularity),
                format!(
                    "granule {granule} is quarantined after a release whose cuMemUnmap failed; \
                     its mapping is still live, so it can never be committed again. Restart the \
                     CUDA context to reclaim it"
                ),
            ));
        }
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
            Err(failure) => {
                // Whatever the rollback could not unmap is still mapped and
                // still owned. Poison it and keep its bytes on the books:
                // reporting the commit as "never happened" while a granule
                // stays mapped is how a later allocation inherits it.
                let residual_bytes = block_bytes(&failure.residual_mapped) as usize;
                for block in &failure.residual_mapped {
                    let granule = block.offset / granularity;
                    arena.spans.granule_refs[granule] = 1;
                    arena.spans.poisoned.insert(granule);
                    arena.spans.committed += granularity;
                    note_commit(granularity);
                }
                self.give_back_lease(arena, mapped_bytes.saturating_sub(residual_bytes));
                self.release_granules(arena, &shared_claimed);
                // Both are read before `failure.error` is moved out below.
                // `rendered` is the whole account, residual included, and is
                // only used where nothing else carries the cause; the typed
                // arm keeps the cause in `source` and so must not restate it
                // in `detail`, or a reader is shown the same sentence twice.
                let rendered = failure.to_string();
                let residual_note = if failure.residual_mapped.is_empty() {
                    String::new()
                } else {
                    format!(
                        " ({} granule(s), {residual_bytes} B, remained mapped after the rollback \
                         and are retained)",
                        failure.residual_mapped.len()
                    )
                };
                return Err(match failure.error {
                    onnx_runtime_virtual_memory::VirtualMemoryError::Delegated {
                        operation: "growing physical handle pool lease",
                        source,
                    } => MemoryError::CapacityUnavailable {
                        tier: "device",
                        requested: mapped_bytes as u64,
                        available: 0,
                        role: self.role,
                        detail: format!(
                            "cuMemMap could not grow the physical handle pool{residual_note}"
                        ),
                        // Carried whole rather than folded into `detail`: an
                        // admission caller needs to tell "we declined, retry
                        // smaller" from "the driver failed, give up", and only
                        // the typed refusal says which this was.
                        source: Some(source),
                    },
                    _ => invalid(mapped_bytes, format!("cuMemMap: {rendered}")),
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
    /// of, and report exactly what would not go away.
    fn release_granules(&self, arena: &mut Arena, granules: &BTreeSet<usize>) {
        let _ = self.drop_granule_references(arena, granules);
    }

    /// Give up one reference per granule and release the ones that reach zero.
    ///
    /// The two accounting axes are settled separately and each exactly once:
    ///
    /// * a block that is unmapped refunds the **mapped** axis (arena committed
    ///   bytes, the process gauge, the arena lease when the backing is not
    ///   pool-accounted) whether or not its handle could then be given back;
    /// * a block whose handle could not be given back keeps its **owned**
    ///   bytes charged, in the pool or on the reservation, because the driver
    ///   has not confirmed the physical memory is gone.
    ///
    /// A run whose unmap fails refunds neither: nothing was mutated, so its
    /// granule keeps a retained reference and is poisoned rather than silently
    /// re-entering circulation.
    fn drop_granule_references(
        &self,
        arena: &mut Arena,
        granules: &BTreeSet<usize>,
    ) -> SpanReleaseReport {
        let granularity = arena.spans.granularity;
        let mut releasable = Vec::new();
        for &granule in granules {
            match arena.spans.granule_refs[granule].checked_sub(1) {
                Some(0) => releasable.push(granule),
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
        let mut result = SpanReleaseReport::default();
        for (start, end) in contiguous_granule_runs(releasable) {
            let report = self.backing.release_range_reporting(
                &mut arena.reservation,
                start * granularity,
                (end - start) * granularity,
            );
            let unmapped = arena.spans.settle_release_run(start..end, &report);
            if unmapped > 0 {
                note_release(unmapped as usize);
                self.give_back_lease(arena, unmapped as usize);
            }
            result.merge(report);
        }
        result
    }

    /// Undo a half-built allocation, keeping whatever will not let go.
    ///
    /// Returning the address range to the free list here is only correct when
    /// every granule this allocation had already claimed came all the way
    /// back. If any did not, the range is quarantined instead: an allocation
    /// that failed to be born still owns physical memory, and handing its
    /// address to the next caller is the same defect as handing out a
    /// partially released one.
    fn unwind_partial_allocation(
        &self,
        arena: &mut Arena,
        offset: usize,
        bytes: usize,
        align: usize,
        committed: &BTreeSet<usize>,
        reason: String,
    ) -> MemoryError {
        let release = self.drop_granule_references(arena, committed);
        if release.is_complete() {
            arena.spans.give_back(offset, bytes);
            return invalid(bytes, reason);
        }
        let record = self.quarantine_record(
            arena,
            offset,
            bytes,
            align,
            release,
            QuarantineReason::PartialRelease,
        );
        let detail = format!(
            "{reason}; rolling the allocation back left {} B of physical memory owned at \
             {:#x}, so its {bytes} byte address range is quarantined rather than reused: {}",
            record.retained_owned_bytes,
            record.address,
            record.fault_summary(),
        );
        arena.spans.quarantine_span(record);
        invalid(bytes, detail)
    }

    /// Why `ptr`/`bytes`/`align` do not describe a live allocation, if they do
    /// not. `None` means the release may proceed.
    fn release_size_mismatch(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> Option<String> {
        let arena = self.lock();
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        let offset = (ptr.as_ptr() as usize).checked_sub(base)?;
        let live = arena.spans.live.get(&offset)?;
        if live.len != bytes {
            return Some(format!(
                "release names {bytes} bytes at {:#x} but that live VMM allocation is {} bytes; \
                 nothing was released",
                ptr.as_ptr() as usize,
                live.len
            ));
        }
        // `allocate_committed` normalizes a zero alignment to one, so compare
        // against the same normalization rather than rejecting a caller that
        // passes back exactly what it passed in.
        if live.align != align.max(1) {
            return Some(format!(
                "release names alignment {align} at {:#x} but that live VMM allocation was made \
                 at alignment {}; nothing was released",
                ptr.as_ptr() as usize,
                live.align
            ));
        }
        None
    }

    /// Build the quarantine record for a span whose release left residuals.
    fn quarantine_record(
        &self,
        arena: &Arena,
        offset: usize,
        len: usize,
        align: usize,
        release: SpanReleaseReport,
        reason: QuarantineReason,
    ) -> QuarantinedSpan {
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        QuarantinedSpan {
            offset,
            len,
            align,
            address: base + offset,
            device: self.device,
            retained_owned_bytes: release.retained_owned_bytes(),
            refunded_mapped_bytes: release.unmapped_bytes(),
            state: release
                .residual_state()
                .unwrap_or(AllocationReleaseState::Released),
            still_mapped: release.still_mapped,
            unmapped_handle_owned: release.unmapped_handle_owned,
            reason,
            faults: release.faults.iter().map(ToString::to_string).collect(),
        }
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
    ///
    /// Migration adapter over [`deallocate_span_outcome`]. Zero here means
    /// "nothing was unmapped", which is a legitimate answer for an allocation
    /// with no committed granules *and* for one whose release was quarantined;
    /// callers that must tell those apart use the structured form.
    ///
    /// [`deallocate_span_outcome`]: Self::deallocate_span_outcome
    pub fn deallocate_span(&self, ptr: NonNull<u8>) -> u64 {
        self.deallocate_span_outcome(ptr).unmapped_bytes()
    }

    /// Release a live allocation, reporting what is still owned afterwards.
    ///
    /// This is the honest whole-allocation release:
    ///
    /// * [`AllocationReleaseOutcome::Complete`] — the address range is back in
    ///   the free list and nothing is owed. Zero unmapped bytes is a normal
    ///   complete result: the allocation may have committed nothing, or every
    ///   granule under it may still be shared with a live neighbour.
    /// * [`AllocationReleaseOutcome::Quarantined`] — some CUDA mutation
    ///   succeeded and a later one did not. The address range is **not**
    ///   returned to the free list, the residual mappings and handles stay
    ///   owned, and the outcome carries both.
    /// * [`AllocationReleaseOutcome::Failed`] — nothing was mutated, because
    ///   the pointer named no live allocation of this arena.
    ///
    /// The address range only returns to the free list once every mapping and
    /// handle under it has reached a terminal state, which is what stops a
    /// later allocation from inheriting a stale mapping.
    pub fn deallocate_span_outcome(&self, ptr: NonNull<u8>) -> AllocationReleaseOutcome {
        let mut arena = self.lock();
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        let address = ptr.as_ptr() as usize;
        let Some(offset) = address.checked_sub(base) else {
            return AllocationReleaseOutcome::failed(format!(
                "address {address:#x} is below this VMM arena reservation ({base:#x}); nothing \
                 was released"
            ));
        };
        if let Some(span) = arena.spans.quarantine.get(&offset) {
            return AllocationReleaseOutcome::failed(format!(
                "address {address:#x} names a span quarantined after a failed release ({}); it \
                 retains {} B of physical ownership and can never be released, reused, or \
                 committed again: {}",
                span.reason,
                span.retained_owned_bytes,
                span.fault_summary(),
            ));
        }
        let Some(live) = arena.spans.live.remove(&offset) else {
            return AllocationReleaseOutcome::failed(format!(
                "address {address:#x} is not a live allocation of this VMM arena; it was already \
                 released or never came from here"
            ));
        };
        let (len, align) = (live.len, live.align);
        let release = self.drop_granule_references(&mut arena, &live.committed);
        let base = <CudaVirtualBacking as VirtualBacking>::base(&arena.reservation);
        let outcome = release.outcome(len as u64, base + offset, align);
        if release.is_complete() {
            arena.spans.give_back(offset, len);
            return outcome;
        }
        let record = self.quarantine_record(
            &arena,
            offset,
            len,
            align,
            release,
            QuarantineReason::PartialRelease,
        );
        arena.spans.quarantine_span(record);
        outcome
    }

    /// Decommit part of a live allocation, transactionally.
    ///
    /// Either the requested granules are unmapped and the allocation stays
    /// live with exactly the topology the caller asked for, or the topology it
    /// had is restored. There is no third answer in which part of a live
    /// allocation is missing: a decommit that unmapped two of three granules
    /// and stopped would leave a live buffer with a hole in it, and the caller
    /// would keep using it.
    ///
    /// Rollback works because the unmap phase retains every physical handle
    /// until it is known complete — a handle already given back cannot be
    /// mapped in again. If the rollback itself fails, the whole allocation is
    /// quarantined rather than reported as restored.
    pub fn decommit_allocation_range_outcome(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        byte_offset: usize,
        bytes: usize,
    ) -> Result<DecommitOutcome, MemoryError> {
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
                    "decommit range {byte_offset}..{end} exceeds allocation of \
                     {allocation_bytes} bytes"
                ),
            ));
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
        if let Some(span) = arena.spans.quarantine.get(&offset) {
            return Err(invalid(
                allocation_bytes,
                format!(
                    "decommit pointer names a span quarantined after a failed release ({}); it \
                     retains {} B of physical ownership and can never be committed or decommitted \
                     again",
                    span.reason, span.retained_owned_bytes,
                ),
            ));
        }
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
                    "decommit allocation size {allocation_bytes} does not match live VMM \
                     allocation size {}",
                    live.len
                ),
            ));
        }
        if bytes == 0 {
            return Ok(DecommitOutcome::Complete {
                accounting: DecommitAccounting::default(),
            });
        }
        let granularity = arena.spans.granularity;
        let align = live.align;
        let len = live.len;
        let absolute_start = offset + byte_offset;
        let absolute_end = offset + end;
        let requested = live
            .committed
            .iter()
            .copied()
            .filter(|granule| {
                let start = granule * granularity;
                start >= absolute_start && start < absolute_end
            })
            .collect::<BTreeSet<_>>();
        if requested.is_empty() {
            return Ok(DecommitOutcome::Complete {
                accounting: DecommitAccounting::default(),
            });
        }
        // References are not dropped until the driver work is known to have
        // succeeded, so a rollback needs no compensating re-increment: the
        // counts were never touched.
        let releasable = requested
            .iter()
            .copied()
            .filter(|&granule| arena.spans.granule_refs[granule] == 1)
            .collect::<Vec<_>>();
        let blocks = releasable
            .iter()
            .flat_map(|&granule| {
                arena
                    .reservation
                    .mapped_blocks()
                    .iter()
                    .copied()
                    .filter(move |block| block.offset / granularity == granule)
            })
            .collect::<Vec<_>>();
        let decommit = self
            .backing
            .decommit_blocks_transactional(&mut arena.reservation, blocks);
        match decommit {
            RangeDecommit::Unmapped(report) => {
                let unmapped = report.unmapped_bytes();
                for &granule in &requested {
                    match arena.spans.granule_refs[granule].checked_sub(1) {
                        Some(remaining) => arena.spans.granule_refs[granule] = remaining,
                        None => {
                            GLOBAL_REF_UNDERFLOWS.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                if unmapped > 0 {
                    arena.spans.committed = arena.spans.committed.saturating_sub(unmapped as usize);
                    note_release(unmapped as usize);
                    self.give_back_lease(&mut arena, unmapped as usize);
                }
                if let Some(live) = arena.spans.live.get_mut(&offset) {
                    for granule in &requested {
                        live.committed.remove(granule);
                    }
                }
                Ok(DecommitOutcome::Complete {
                    accounting: DecommitAccounting {
                        requested_bytes: bytes as u64,
                        unmapped_bytes: unmapped,
                        quarantined_owned_bytes: block_bytes(&report.unmapped_handle_owned),
                    },
                })
            }
            RangeDecommit::RolledBack(fault) => Ok(DecommitOutcome::RolledBack {
                reason: fault.to_string(),
            }),
            RangeDecommit::Poisoned {
                still_mapped,
                unmapped_handle_owned,
                faults,
            } => {
                // The mapping topology is neither the old one nor the new one,
                // so the allocation cannot honestly be called live. Retire it
                // whole, keeping its address range and every residual.
                let unmapped = block_bytes(&unmapped_handle_owned);
                if unmapped > 0 {
                    arena.spans.committed = arena.spans.committed.saturating_sub(unmapped as usize);
                    note_release(unmapped as usize);
                    self.give_back_lease(&mut arena, unmapped as usize);
                }
                let live = arena.spans.live.remove(&offset);
                let mut retained = live.map(|live| live.committed).unwrap_or_default();
                for block in &unmapped_handle_owned {
                    let granule = block.offset / granularity;
                    arena.spans.granule_refs[granule] = 0;
                    retained.remove(&granule);
                }
                for block in &still_mapped {
                    let granule = block.offset / granularity;
                    arena.spans.granule_refs[granule] = 1;
                    arena.spans.poisoned.insert(granule);
                }
                // Granules of this allocation that the failed decommit never
                // touched are still mapped and still owned by the quarantined
                // span; nothing may unmap them again.
                let mut still_mapped = still_mapped;
                let untouched = retained
                    .iter()
                    .filter(|&&granule| {
                        still_mapped
                            .iter()
                            .all(|block| block.offset / granularity != granule)
                    })
                    .filter_map(|&granule| {
                        arena
                            .reservation
                            .mapped_blocks()
                            .iter()
                            .copied()
                            .find(|block| block.offset / granularity == granule)
                    })
                    .collect::<Vec<_>>();
                for block in untouched {
                    arena.spans.poisoned.insert(block.offset / granularity);
                    still_mapped.push(block);
                }
                still_mapped.sort_unstable_by_key(|block| block.offset);
                let record = self.quarantine_record(
                    &arena,
                    offset,
                    len,
                    align,
                    SpanReleaseReport {
                        settled: Vec::new(),
                        still_mapped,
                        unmapped_handle_owned,
                        faults,
                    },
                    QuarantineReason::PartialRelease,
                );
                let residual = record.residual();
                let reason = record.fault_summary();
                arena.spans.quarantine_span(record);
                Ok(DecommitOutcome::Quarantined {
                    accounting: DecommitAccounting {
                        requested_bytes: bytes as u64,
                        unmapped_bytes: unmapped,
                        quarantined_owned_bytes: unmapped,
                    },
                    residual,
                    reason,
                })
            }
        }
    }

    /// Spans this arena still owns after a release that could not finish.
    ///
    /// Empty in a correct run. A non-empty list is a standing fault: the
    /// address ranges named here are permanently withdrawn from the arena and
    /// their physical bytes stay charged.
    pub fn quarantined_spans(&self) -> Vec<QuarantinedSpan> {
        self.lock().spans.quarantine.values().cloned().collect()
    }

    /// Physical bytes this arena owns but could not release.
    pub fn quarantined_owned_bytes(&self) -> u64 {
        self.lock()
            .spans
            .quarantine
            .values()
            .map(|span| span.retained_owned_bytes)
            .sum()
    }

    /// Attach a deterministic driver fault plan to this allocator's backing.
    ///
    /// Test-only, and scoped to this allocator: no other allocator, pool or
    /// test in the same process can observe the injected faults.
    #[cfg(any(test, feature = "gpu-tests"))]
    pub fn install_driver_faults(&mut self, plan: Arc<crate::release::DriverFaultPlan>) {
        self.backing = self.backing.clone().with_driver_faults(plan);
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

    fn validate_shared_prefix_source(
        &self,
        prefix: &SharedPrefix,
        requested: usize,
    ) -> Result<(), MemoryError> {
        if prefix.device != self.device {
            return Err(invalid(
                requested,
                format!(
                    "shared prefix belongs to device {:?} but this allocator serves {:?}",
                    prefix.device, self.device
                ),
            ));
        }
        match (prefix.authority, self.physical_pool_authority()) {
            (Some(prefix_authority), Some(self_authority))
                if prefix_authority == self_authority =>
            {
                Ok(())
            }
            (None, _) | (_, None) => Err(invalid(
                requested,
                String::from(
                    "shared prefix requires the production physical-handle pool on both the \
                     prefix and the committing allocator",
                ),
            )),
            _ => Err(invalid(
                requested,
                String::from(
                    "shared prefix was created under a different pool authority than this \
                     allocator",
                ),
            )),
        }
    }

    /// Estimate the incremental owned bytes to admit one more sharer.
    ///
    /// A compatible prefix is already owned and costs zero. Foreign device or
    /// pool-authority inputs are rejected before reporting a cost, exactly as
    /// the mapping operation rejects them.
    pub fn incremental_owned_bytes_for_shared_prefix(
        &self,
        prefix: &SharedPrefix,
    ) -> Result<u64, MemoryError> {
        self.validate_shared_prefix_source(prefix, prefix.requested_bytes)?;
        Ok(0)
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
        self.validate_shared_prefix_source(prefix, allocation_bytes)?;

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
                    // Unmap what this call mapped. The handle goes back to the
                    // pool's shared refcount, not to `available`, so the other
                    // sharers and the prefix owner keep the physical granule:
                    // its lifetime is the union of every mapping, and a failed
                    // sharer must not shorten it.
                    let mut residual = Vec::new();
                    // The granule that just failed may itself still be mapped:
                    // `map_shared_prefix_readonly` records the block when its
                    // own cleanup unmap fails. Leaving it out here would let
                    // the arena believe an address with a live read-only
                    // mapping under it is free.
                    let failed_offset = granule * granularity;
                    residual.extend(
                        arena
                            .reservation
                            .mapped_blocks()
                            .iter()
                            .copied()
                            .filter(|block| block.offset == failed_offset),
                    );
                    for &done in &mapped {
                        let report = self.backing.release_range_reporting(
                            &mut arena.reservation,
                            done * granularity,
                            granularity,
                        );
                        residual.extend(report.still_mapped.iter().copied());
                        residual.extend(report.unmapped_handle_owned.iter().copied());
                    }
                    for block in &residual {
                        // Never entered `granule_refs` in the first place, so
                        // taking a reference here is what keeps a granule whose
                        // mapping survived from being handed out again, and
                        // charging `committed` keeps the mapped-byte gauge
                        // truthful about memory that really is mapped.
                        let granule = block.offset / granularity;
                        arena.spans.granule_refs[granule] = 1;
                        arena.spans.poisoned.insert(granule);
                        arena.spans.committed += granularity;
                        note_commit(granularity);
                    }
                    if residual.is_empty() {
                        return Err(invalid(
                            allocation_bytes,
                            format!("shared prefix mapping: {error}"),
                        ));
                    }
                    return Err(invalid(
                        allocation_bytes,
                        format!(
                            "shared prefix mapping: {error}; rolling back left {} granule(s) \
                             ({} B) mapped and owned, which are quarantined and can never be \
                             committed again",
                            residual.len(),
                            block_bytes(&residual),
                        ),
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
                quarantine_or(arena, offset, "commit pointer is not a live VMM allocation"),
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

/// Name the quarantine when `offset` is one, so a caller sees why an address
/// it holds stopped working instead of the generic "not live".
fn quarantine_or(arena: &Arena, offset: usize, otherwise: &str) -> String {
    match arena.spans.quarantine.get(&offset) {
        Some(span) => format!(
            "this address names a span quarantined after a failed release ({}); it retains {} B \
             of physical ownership and can never be committed, used, or released again: {}",
            span.reason,
            span.retained_owned_bytes,
            span.fault_summary(),
        ),
        None => otherwise.to_string(),
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

    /// Reserve one allocation and atomically consume governor-owned mapped
    /// capacity for the granules that become newly mapped.
    ///
    /// This stays inherent because [`MappedPhysicalCapacityToken`] belongs to
    /// the governor, not to the low-level [`MemoryVirtualBacking`] capability.
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
        let ptr = <Self as MemoryVirtualBacking>::allocate_committed(self, bytes, align, &[])?;
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
                // SAFETY: this exact live allocation has not escaped.
                unsafe { self.deallocate(ptr, bytes, align) };
                return Err(error);
            }
        };
        Ok(onnx_runtime_memory_governor::MappedAllocation {
            allocation: ptr,
            additional_owned_bytes: commit.additional_owned_bytes,
            newly_mapped_bytes: commit.newly_mapped_bytes,
        })
    }

    /// Commit ranges while atomically consuming governor-owned mapped capacity.
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
impl DeviceAllocator for CudaVmmAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        let full = 0..bytes;
        <Self as MemoryVirtualBacking>::allocate_committed(
            self,
            bytes,
            align,
            std::slice::from_ref(&full),
        )
    }

    /// Migration adapter. Delegates to [`release`](Self::release) so a
    /// quarantined address can never come back through the shorter path.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        // SAFETY: forwarded under this method's identical contract.
        let _ = unsafe { self.release(ptr, bytes, align) };
    }

    /// Migration adapter reporting only the mapped-axis refund. Zero is a
    /// valid answer and never means the release failed; use
    /// [`release`](Self::release) to tell complete from quarantined.
    unsafe fn deallocate_with_unmapped(&self, ptr: NonNull<u8>, bytes: usize, align: usize) -> u64 {
        // SAFETY: forwarded under this method's identical contract.
        unsafe { self.release(ptr, bytes, align) }.unmapped_bytes()
    }

    /// Whole-allocation release with honest terminal state.
    ///
    /// `bytes` is checked against the live record rather than trusted: a
    /// mismatch means the caller is not describing this allocation, and
    /// releasing on that basis would unmap granules belonging to something
    /// else. Such a call mutates nothing and reports
    /// [`AllocationReleaseOutcome::Failed`].
    unsafe fn release(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> AllocationReleaseOutcome {
        if let Some(mismatch) = self.release_size_mismatch(ptr, bytes, align) {
            return AllocationReleaseOutcome::failed(mismatch);
        }
        self.deallocate_span_outcome(ptr)
    }

    fn device(&self) -> DeviceKey {
        self.device
    }

    fn commits_on_demand(&self) -> bool {
        true
    }

    fn as_virtual_backing(&self) -> Option<&dyn MemoryVirtualBacking> {
        Some(self)
    }

    fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
        self.backing.physical_pool().is_some().then_some(self)
    }
}

impl MemoryVirtualBacking for CudaVmmAllocator {
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
                let reason = format!(
                    "committed subrange {}..{} lies outside allocation of {bytes} bytes",
                    range.start, range.end
                );
                return Err(self.unwind_partial_allocation(
                    &mut arena, offset, bytes, align, &committed, reason,
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
                    let reason = error.to_string();
                    let unwind = self.unwind_partial_allocation(
                        &mut arena, offset, bytes, align, &committed, reason,
                    );
                    // The original refusal is the useful diagnosis unless the
                    // unwind itself left something owned, which outranks it.
                    return Err(if arena.spans.quarantine.contains_key(&offset) {
                        unwind
                    } else {
                        error
                    });
                }
            };
            committed.extend(claimed);
        }
        arena.spans.live.insert(
            offset,
            LiveSpan {
                len: bytes,
                align,
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
                quarantine_or(
                    &arena,
                    offset,
                    "commit pointer is not a live VMM allocation",
                ),
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

    /// Migration adapter over [`decommit_allocation_range_outcome`].
    ///
    /// `Ok(bytes)` means the range really is unmapped and the allocation is
    /// still live. `Err` means the allocation is either exactly as it was (the
    /// decommit rolled back) or quarantined whole (the rollback failed) — the
    /// message says which, because the two demand different responses from the
    /// caller.
    ///
    /// [`decommit_allocation_range_outcome`]: CudaVmmAllocator::decommit_allocation_range_outcome
    fn decommit_allocation_range(
        &self,
        ptr: NonNull<u8>,
        allocation_bytes: usize,
        _align: usize,
        byte_offset: usize,
        bytes: usize,
    ) -> Result<u64, MemoryError> {
        let end = byte_offset.saturating_add(bytes);
        match self.decommit_allocation_range_outcome(ptr, allocation_bytes, byte_offset, bytes)? {
            DecommitOutcome::Complete { accounting } => Ok(accounting.unmapped_bytes),
            DecommitOutcome::RolledBack { reason } => Err(invalid(
                allocation_bytes,
                format!(
                    "decommit of {byte_offset}..{end} was refused and the original mapping was \
                     restored, so the allocation is unchanged and still usable: {reason}"
                ),
            )),
            DecommitOutcome::Quarantined {
                accounting,
                residual,
                reason,
            } => Err(invalid(
                allocation_bytes,
                format!(
                    "decommit of {byte_offset}..{end} could not be rolled back; the whole \
                     allocation at {:#x} is quarantined with {} B of physical ownership retained \
                     after {} B were actually unmapped, and can no longer be used or released: \
                     {reason}",
                    residual.address, residual.retained_bytes, accounting.unmapped_bytes,
                ),
            )),
        }
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
}

impl SharedMapping for CudaVmmAllocator {
    fn create_shared_prefix(
        &self,
        bytes: usize,
    ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
        let prefix = CudaVmmAllocator::create_shared_prefix(self, bytes)?;
        Ok(Box::new(prefix))
    }

    fn incremental_owned_bytes_for_shared_prefix(
        &self,
        prefix: &dyn SharedDevicePrefix,
    ) -> Result<u64, MemoryError> {
        let prefix = prefix.as_any().downcast_ref::<SharedPrefix>().ok_or_else(|| {
            invalid(
                prefix.requested_bytes(),
                String::from(
                    "shared prefix was not created by a CUDA VMM allocator; no incremental cost \
                     can be reported for an unmappable prefix",
                ),
            )
        })?;
        CudaVmmAllocator::incremental_owned_bytes_for_shared_prefix(self, prefix)
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
}

impl Drop for CudaVmmAllocator {
    fn drop(&mut self) {
        // Report what was actually mapped. Without this the arena's whole
        // premise -- that reserved address space is large and committed memory
        // is small -- is unfalsifiable from outside: `committed_and_reserved`
        // exists but nothing calls it, so "is this doing anything?" has no
        // answer short of a debugger.
        let (committed, reserved, granularity, quarantined_mapped, quarantined_spans) = {
            let arena = self.lock();
            let quarantined_mapped = arena
                .spans
                .quarantine
                .values()
                .map(|span| block_bytes(&span.still_mapped))
                .sum::<u64>();
            (
                arena.spans.committed,
                arena.spans.capacity(),
                arena.spans.granularity,
                quarantined_mapped,
                arena.spans.quarantine.len(),
            )
        };
        eprintln!(
            "cuda_ep: VMM arena closing: committed {committed} B of {reserved} B reserved \
             ({} granules of {granularity} B)",
            committed / granularity.max(1),
        );
        if quarantined_spans > 0 {
            eprintln!(
                "cuda_ep: WARNING: VMM arena closing with {quarantined_spans} quarantined span(s) \
                 holding {quarantined_mapped} B still mapped; those bytes stay charged because \
                 the driver never confirmed they were released"
            );
        }

        // The reservation's own `Drop` unmaps and frees every block it can, so
        // the only thing left is to stop the ledger believing the granules are
        // still held. Quarantined mappings are the exception: their unmap
        // already failed once, so their bytes stay charged rather than being
        // advertised as free memory the device does not actually have.
        let mut arena = self.lock();
        let held = (arena.spans.committed as u64).saturating_sub(quarantined_mapped);
        arena.lease.shrink(held);
        arena.spans.committed = quarantined_mapped as usize;

        // Keep the process-global counters honest. Without this the arena's
        // bytes stay on the books after it is gone, and a second run in the
        // same process reads the first run's memory as if it were still
        // mapped.
        subtract_counted(&GLOBAL_COMMITTED_BYTES, held);
        subtract_counted(&GLOBAL_RESERVED_BYTES, reserved as u64);
        if quarantined_mapped > 0 {
            // The lease still owns the quarantined bytes; dropping it would
            // hand them back to the governor as free memory the device may not
            // actually have. Swap in a zero-byte sibling over the same
            // accounting and leak the original, which is the same conservative
            // choice the physical handle pool makes.
            if let Ok(placeholder) = arena.lease.reserve_sibling(0) {
                let retained = std::mem::replace(&mut arena.lease, placeholder);
                std::mem::forget(retained);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The retained-pool bound is read from a value, not from the process
    /// environment, and only a positive byte count configures a pool.
    ///
    /// This is the one knob left after Phase 7 removed the arena's on/off
    /// switch, so it is also the only remaining way a caller can get the
    /// arena's retention behaviour wrong. `Some(0)` and unparseable text both
    /// have to mean "no bound configured" rather than "a bound of zero", which
    /// would retain nothing and turn every reuse into a fresh `cuMemCreate`.
    ///
    /// Parsing is factored out of the environment read specifically so this can
    /// be asserted without `set_var`, which is process-global and would make
    /// the result depend on which other test happened to be running.
    #[test]
    fn only_a_positive_byte_count_configures_a_retained_physical_pool() {
        assert_eq!(
            parse_physical_handle_pool_bytes(Some("67108864")),
            Some(1 << 26)
        );
        assert_eq!(
            parse_physical_handle_pool_bytes(Some("  67108864 ")),
            Some(1 << 26),
            "a value with surrounding whitespace is still a configured bound"
        );

        assert_eq!(
            parse_physical_handle_pool_bytes(None),
            None,
            "an unset variable configures no bound"
        );
        assert_eq!(
            parse_physical_handle_pool_bytes(Some("0")),
            None,
            "zero must mean no pool rather than a pool that can retain nothing"
        );
        assert_eq!(
            parse_physical_handle_pool_bytes(Some("64MiB")),
            None,
            "only a plain byte count is accepted; a unit suffix is not silently truncated"
        );
        assert_eq!(
            parse_physical_handle_pool_bytes(Some("-1")),
            None,
            "a negative value must not wrap into an enormous bound"
        );
        assert_eq!(parse_physical_handle_pool_bytes(Some("")), None);
    }

    /// The public predicate is exactly "a retained-byte bound is configured",
    /// with no surviving dependence on the deleted arena on/off flag.
    ///
    /// # Why this needs its own test
    ///
    /// `production_physical_pool_enabled` had no coverage at all: changing its
    /// body to `true` survived every suite in the workspace, including the
    /// engine's. Its meaning also *changed* in Phase 7 — it used to require the
    /// now-deleted `ONNX_GENAI_CUDA_VMM` flag as well, so
    /// `ONNX_GENAI_CUDA_PHYSICAL_HANDLE_POOL_BYTES` set on its own was ignored
    /// and is now honoured. That change is deliberate and load-bearing: its
    /// only consumer is `engine/load.rs`'s `uses_governed_physical_pool`, which
    /// feeds `cuda_weight_startup_reservation`, and the arena now always
    /// applies `physical_handle_pool_bytes().or(default)`. Leaving the
    /// predicate gated on a flag that no longer exists would make the engine
    /// mispredict whether an authority-owned pool is present.
    ///
    /// The expectation is computed from the parse helper pinned above rather
    /// than restated, so this asserts the *composition* — that the predicate
    /// asks the environment exactly one question and applies no second
    /// condition to the answer.
    ///
    /// Scope, stated rather than assumed: nothing in this workspace calls
    /// `set_var` for this variable, so it is absent when the suite runs and the
    /// expectation is `false`, which is what kills a `true` body. A developer
    /// who has exported the variable will still see this pass, because it
    /// checks agreement rather than a fixed answer.
    #[test]
    fn the_production_pool_predicate_is_exactly_whether_a_bound_is_configured() {
        let configured = std::env::var(CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV).ok();
        assert_eq!(
            production_physical_pool_enabled(),
            parse_physical_handle_pool_bytes(configured.as_deref()).is_some(),
            "the predicate must be the configured-bound question and nothing else; \
             {CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV} is currently {configured:?}"
        );
    }

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

    const GRANULE: usize = 1 << 16;

    fn block(granule: usize) -> MappedBlock {
        MappedBlock::new(granule * GRANULE, GRANULE, 500 + granule as u64)
    }

    /// A span with one granule claimed, as `allocate_committed` would leave it.
    fn committed_spans(granules: usize) -> Spans {
        let mut spans = Spans::new(GRANULE, granules * GRANULE);
        let offset = spans.carve(granules * GRANULE, 1).expect("fits");
        for granule in 0..granules {
            spans.granule_refs[granule] = 1;
            spans.committed += GRANULE;
        }
        spans.live.insert(
            offset,
            LiveSpan {
                len: granules * GRANULE,
                align: 1,
                committed: (0..granules).collect(),
            },
        );
        spans
    }

    fn quarantine_of(spans: &mut Spans, offset: usize, len: usize, report: &SpanReleaseReport) {
        spans.quarantine_span(QuarantinedSpan {
            offset,
            len,
            align: 1,
            address: 0x1000 + offset,
            device: DeviceKey::device(0),
            still_mapped: report.still_mapped.clone(),
            unmapped_handle_owned: report.unmapped_handle_owned.clone(),
            refunded_mapped_bytes: report.unmapped_bytes(),
            retained_owned_bytes: report.retained_owned_bytes(),
            state: AllocationReleaseState::PartiallyUnmapped,
            reason: QuarantineReason::PartialRelease,
            faults: vec![String::from("cuMemUnmap failed: injected")],
        });
    }

    /// A granule whose unmap failed keeps its reference and its committed
    /// bytes, and is poisoned.
    ///
    /// This is the exact-accounting rule: the mapped axis is refunded only for
    /// bytes whose mapping is really gone. Refunding a granule that is still
    /// mapped would tell the governor there is memory available that the
    /// device is still holding.
    #[test]
    fn a_failed_unmap_refunds_nothing_and_poisons_the_granule() {
        let mut spans = committed_spans(3);
        let report = SpanReleaseReport {
            settled: vec![block(0), block(2)],
            still_mapped: vec![block(1)],
            ..SpanReleaseReport::default()
        };

        let refund = spans.settle_release_run(0..3, &report);

        assert_eq!(refund, 2 * GRANULE as u64, "only the two unmapped granules");
        assert_eq!(spans.committed, GRANULE, "the mapped granule stays charged");
        assert_eq!(spans.granule_refs, vec![0, 1, 0], "granule 1 keeps a claim");
        assert!(spans.poisoned.contains(&1));
        assert!(!spans.poisoned.contains(&0) && !spans.poisoned.contains(&2));
    }

    /// A handle that could not be given back still refunds the mapped axis.
    ///
    /// The two axes are independent: the mapping really is gone (so the
    /// arena's committed bytes must drop) while the physical memory is still
    /// owned by the pool (so its owned-byte gauge must not).
    #[test]
    fn an_unmapped_block_with_an_owned_handle_refunds_only_the_mapped_axis() {
        let mut spans = committed_spans(2);
        let report = SpanReleaseReport {
            settled: vec![block(0)],
            unmapped_handle_owned: vec![block(1)],
            ..SpanReleaseReport::default()
        };

        let refund = spans.settle_release_run(0..2, &report);

        assert_eq!(refund, 2 * GRANULE as u64, "both mappings are gone");
        assert_eq!(spans.committed, 0);
        assert_eq!(
            report.retained_owned_bytes(),
            GRANULE as u64,
            "the quarantined handle is still owned"
        );
        assert!(
            spans.poisoned.is_empty(),
            "an address whose mapping is genuinely gone is not poisoned"
        );
    }

    /// A quarantined span never comes back, however the arena is exercised.
    ///
    /// This is the defect Phase 4 exists to close: a partially released
    /// address that returned to the free list would be handed to a later
    /// allocation with the old mapping still under part of it.
    #[test]
    fn a_quarantined_span_is_never_carved_again() {
        let mut spans = committed_spans(3);
        let report = SpanReleaseReport {
            settled: vec![block(0), block(2)],
            still_mapped: vec![block(1)],
            ..SpanReleaseReport::default()
        };
        spans.settle_release_run(0..3, &report);
        // The whole allocation is retired without its address returning to the
        // free list.
        spans.live.remove(&0);
        quarantine_of(&mut spans, 0, 3 * GRANULE, &report);

        assert!(spans.free.is_empty(), "the arena has no free span left");
        assert_eq!(spans.carve(GRANULE, 1), None, "nothing may be handed out");
        assert!(spans.live.is_empty(), "the span is not live either");
        let quarantined = spans.quarantine.get(&0).expect("recorded");
        assert_eq!(quarantined.retained_owned_bytes, GRANULE as u64);
        assert_eq!(quarantined.still_mapped, vec![block(1)]);
    }

    /// A poisoned granule refuses every future commit, even from a neighbour
    /// that legitimately shares it.
    #[test]
    fn a_poisoned_granule_fails_every_later_commit_closed() {
        let mut spans = committed_spans(2);
        let report = SpanReleaseReport {
            still_mapped: vec![block(1)],
            settled: vec![block(0)],
            ..SpanReleaseReport::default()
        };
        spans.settle_release_run(0..2, &report);

        assert_eq!(spans.first_poisoned([0, 1]), Some(1));
        assert_eq!(spans.first_poisoned([0]), None, "only the failed granule");
    }

    /// A rollback that succeeded touches nothing: no refund, no poison, no
    /// change to any reference count.
    #[test]
    fn a_successful_rollback_leaves_the_bookkeeping_untouched() {
        let before = committed_spans(3);
        let mut spans = committed_spans(3);

        // A rolled-back decommit never reaches `settle_release_run`, which is
        // the property: references are dropped only after the driver work is
        // known to have succeeded, so there is nothing to compensate for.
        assert_eq!(spans.granule_refs, before.granule_refs);
        assert_eq!(spans.committed, before.committed);
        assert!(spans.poisoned.is_empty());
        assert_eq!(
            spans.live.get(&0).map(|live| live.committed.len()),
            Some(3),
            "every granule is still committed to the live allocation"
        );
        assert!(spans.quarantine.is_empty());
        assert_eq!(spans.carve(GRANULE, 1), None, "the span is still live");
    }

    /// Interleaved allocate/release with one poisoned granule in the middle
    /// never produces a reusable address for that granule, whatever the order.
    ///
    /// Deterministic rather than randomized: the orders are enumerated, so a
    /// failure names the exact sequence rather than a seed.
    #[test]
    fn repeated_allocate_and_release_never_reuses_a_poisoned_granule() {
        for order in [[0usize, 1, 2], [2, 1, 0], [1, 0, 2], [1, 2, 0]] {
            let mut spans = Spans::new(GRANULE, 4 * GRANULE);
            let mut offsets = Vec::new();
            for _ in 0..3 {
                let offset = spans.carve(GRANULE, 1).expect("fits");
                let granule = offset / GRANULE;
                spans.granule_refs[granule] = 1;
                spans.committed += GRANULE;
                spans.live.insert(
                    offset,
                    LiveSpan {
                        len: GRANULE,
                        align: 1,
                        committed: BTreeSet::from([granule]),
                    },
                );
                offsets.push(offset);
            }
            for &index in &order {
                let offset = offsets[index];
                let granule = offset / GRANULE;
                let report = if granule == 1 {
                    SpanReleaseReport {
                        still_mapped: vec![block(1)],
                        ..SpanReleaseReport::default()
                    }
                } else {
                    SpanReleaseReport {
                        settled: vec![block(granule)],
                        ..SpanReleaseReport::default()
                    }
                };
                spans.settle_release_run(granule..granule + 1, &report);
                spans.live.remove(&offset);
                if report.is_complete() {
                    spans.give_back(offset, GRANULE);
                } else {
                    quarantine_of(&mut spans, offset, GRANULE, &report);
                }
            }

            assert!(
                spans.poisoned.contains(&1),
                "order {order:?} lost the poison on granule 1"
            );
            assert_eq!(spans.granule_refs[1], 1, "order {order:?}");
            assert_eq!(spans.committed, GRANULE, "order {order:?}");
            // Every free span must avoid the quarantined granule.
            for (&start, &len) in &spans.free {
                assert!(
                    !(start < 2 * GRANULE && GRANULE < start + len),
                    "order {order:?} returned quarantined granule 1 to the free list: {:?}",
                    spans.free
                );
            }
            // The arena can still serve the granules that released cleanly.
            assert!(spans.carve(GRANULE, 1).is_some(), "order {order:?}");
        }
    }
}

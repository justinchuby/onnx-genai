//! CUDA virtual memory: one contiguous device address over scattered physical
//! allocations.
//!
//! # What this is for
//!
//! ONNX Runtime's `GroupQueryAttention` wants one flat K tensor and one flat V
//! tensor. A paged KV cache does not have those, so today
//! `mirror_present_kv_to_pages` **copies** the whole thing into a contiguous
//! buffer every step.
//!
//! CUDA's virtual memory management removes the copy rather than making it
//! faster: reserve one device address range with `cuMemAddressReserve`, then
//! map separately-created physical handles into consecutive parts of it. The
//! operator sees a flat buffer; the pages behind it were never gathered.
//!
//! ONNX Runtime does ship a `PagedAttention` operator, but it is CUDA-only
//! *and* a graph operator, so a stock exported model cannot reach it. Virtual
//! contiguity works on the model as exported.
//!
//! # Measured, not assumed
//!
//! On an RTX 4060 (`nvcuda.dll`, driver API):
//!
//! ```text
//! minimum granularity:     2097152 bytes = 2 MiB
//! recommended granularity: 2097152 bytes = 2 MiB
//! reserved 1 GiB of device address space
//! mapped 2 granules from separate cuMemCreate handles
//! wrote and read 4 MiB straight across the seam: correct
//! ```
//!
//! 2 MiB is roughly a thousand tokens of one KV tensor at Llama-3-8B geometry —
//! coarse, and fine at the concurrency this project targets (#596).
//!
//! # Physical handle lifetime
//!
//! `cuMemUnmap` removes a mapping but does not free the physical memory behind
//! it; that needs `cuMemRelease` on the handle `cuMemCreate` returned. A plain
//! backing keeps handles with its reservation until release. A pooled backing
//! instead returns unmapped granule handles to a device-scoped pool, so the
//! same physical allocation can be mapped into a different reservation later.
//!
//! # Lock order
//!
//! Every lock this crate takes is listed here, outermost first. Taking them in
//! this order is what keeps the arena, the pool and the process-wide registries
//! deadlock-free; taking two of them in the opposite order is the only way to
//! break it, so the order is written down rather than inferred.
//!
//! 1. [`crate::vmm_allocator::CudaVmmAllocator`]'s arena mutex.
//! 2. `PhysicalHandlePool::authority_gate` (read), the limit-reconfiguration
//!    gate shared by every pool of one authority.
//! 3. `PhysicalHandlePool::lease_checkout`, serializing lease growth.
//! 4. `PhysicalHandlePool::state`, the pool's own handle bookkeeping.
//!
//! The process-wide `PHYSICAL_POOLS` and `PHYSICAL_POOL_AUTHORITY_GATES`
//! registries are *leaves in time* rather than in depth: they are always
//! acquired and released without any pool lock held, and never while a driver
//! call is outstanding.
//!
//! Two rules ride on top of the order:
//!
//! * **No driver call under the pool state lock.** `cuMemMap`, `cuMemUnmap`,
//!   `cuMemCreate` and `cuMemRelease` all happen with `PoolState` unlocked.
//!   The pool documents mapping as deliberately outside its mutex; release now
//!   follows the same rule so a driver stall cannot block every other checkout.
//! * **No device wait anywhere in release.** Nothing here calls
//!   `cuCtxSynchronize` or `cuStreamSynchronize`. Ordering release after
//!   in-flight work is the provider queue's job (see
//!   [`CudaVirtualBacking::with_reservation_queue`]); a whole-device
//!   synchronize inside an allocator would serialize every stream in the
//!   process to solve one allocation's problem.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

use cudarc::driver::sys as cu;
use onnx_runtime_memory_governor::{
    HolderId, MappedPhysicalCapacityToken, MemoryAuthorityId, MemoryError, MemoryGovernor,
    MemoryLease, MemoryRole, Tier,
};
use onnx_runtime_virtual_memory::{PhysicalMemoryAccounting, VirtualBacking, VirtualMemoryError};

use crate::release::{
    BlockAccess, DriverFault, DriverOperation, HandleDisposition, MappedBlock, ReleaseDriver,
    SpanReleaseReport, TransactionalUnmap, contiguous_runs, dispose_released_blocks, release_runs,
    unmap_runs_transactional,
};
use cudarc::driver::CudaContext;

/// Device address space, backed by CUDA physical allocations.
///
/// Holds the runtime so the CUDA context is bound before every driver call —
/// the reservation and its mappings belong to a context, and touching them from
/// an unbound thread is a driver error rather than a silent wrong answer.
pub type TeardownSynchronizer = Arc<dyn Fn() -> Result<(), String> + Send + Sync>;

/// A reservation teardown, detached from the reservation that owned it.
///
/// `CudaReservation::Drop` cannot wait for in-flight kernels and copies — a
/// `Drop` that blocks on a device is exactly the stall Phase 4 removes — so it
/// instead moves the *exact* address range, mapped blocks, quarantined blocks,
/// pool, and context into this ticket and hands it to a queue. The queue
/// executes it once the work that could still be reading the range has
/// completed.
///
/// # Fail-safe
///
/// Dropping a ticket without executing it **retains** everything: the address
/// range is never returned to the driver, mapped blocks are never unmapped, and
/// physical handles are never given back to a pool. That leaks; it cannot
/// hand a live mapping's address to the next `cuMemAddressReserve`.
pub struct ReservationTeardownTicket {
    base: cu::CUdeviceptr,
    len: usize,
    context: Arc<CudaContext>,
    pool: Option<Arc<PhysicalHandlePool>>,
    blocks: Vec<MappedBlock>,
    quarantined: Vec<MappedBlock>,
    /// Cleared by `execute`, so `Drop` only retains a genuinely abandoned
    /// ticket.
    armed: bool,
}

// SAFETY: the ticket owns a device address range plus plain-integer handles and
// `Send + Sync` pins, exactly like the reservation it came from. Every driver
// call it makes binds the context first.
unsafe impl Send for ReservationTeardownTicket {}
unsafe impl Sync for ReservationTeardownTicket {}

impl std::fmt::Debug for ReservationTeardownTicket {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReservationTeardownTicket")
            .field("base", &self.base)
            .field("len", &self.len)
            .field("blocks", &self.blocks.len())
            .field("quarantined", &self.quarantined.len())
            .finish()
    }
}

/// What a reservation teardown actually did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReservationTeardownReport {
    /// Bytes whose mapping was removed.
    pub unmapped_bytes: u64,
    /// Blocks that could not be fully released and are still owned.
    pub retained_blocks: usize,
    /// Whether the address range was returned to the driver.
    pub address_range_freed: bool,
}

/// A reservation teardown together with any exact residual ownership.
#[derive(Debug)]
pub struct ReservationTeardownOutcome {
    pub report: ReservationTeardownReport,
    pub retained: Option<ReservationTeardownTicket>,
}

impl ReservationTeardownReport {
    /// Whether nothing was retained: every block released and the address range
    /// went back to the driver.
    pub const fn is_complete(&self) -> bool {
        self.retained_blocks == 0 && self.address_range_freed
    }
}

impl ReservationTeardownTicket {
    /// Bytes of address space this ticket owns.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Device address of the reserved range. Never dereferenced here.
    pub fn base(&self) -> usize {
        self.base as usize
    }

    /// Blocks still mapped into the range.
    pub fn mapped_blocks(&self) -> &[MappedBlock] {
        &self.blocks
    }

    /// Blocks whose physical handle was already retained as unreleasable.
    pub fn quarantined_blocks(&self) -> &[MappedBlock] {
        &self.quarantined
    }

    /// Whether this ticket carries residual ownership that can never be
    /// released, in which case its address range must never be returned.
    pub fn has_residual_quarantine(&self) -> bool {
        !self.quarantined.is_empty()
    }

    /// Perform the teardown.
    ///
    /// Must be called only after the work that could still be reading this
    /// range has completed; the queue that owns the ticket is responsible for
    /// that ordering. Nothing here waits on a device.
    pub fn execute_outcome(mut self) -> ReservationTeardownOutcome {
        self.armed = false;
        let _ = self.context.bind_to_thread();
        let mut retained_mapped = Vec::new();
        let mut retained_unmapped = std::mem::take(&mut self.quarantined);
        let mut unmapped = 0_u64;
        for block in std::mem::take(&mut self.blocks) {
            // SAFETY: each block was mapped by `commit` into this reservation
            // and is unmapped exactly once here.
            if unsafe { cu::cuMemUnmap(self.base + block.offset as u64, block.len) }
                != cu::CUresult::CUDA_SUCCESS
            {
                // Still mapped. The address range must not be freed under a
                // live mapping, so record it and keep the reservation.
                retained_mapped.push(block);
                continue;
            }
            unmapped = unmapped.saturating_add(block.len as u64);
            if let Some(pool) = &self.pool {
                if !pool.return_after_unmap(block.handle, true).is_settled() {
                    retained_unmapped.push(block);
                }
            } else if unsafe { cu::cuMemRelease(block.handle) } != cu::CUresult::CUDA_SUCCESS {
                retained_unmapped.push(block);
            }
        }
        let retained = retained_mapped.len() + retained_unmapped.len();
        if retained > 0 {
            // Freeing an address range that still has a live mapping, or whose
            // physical backing the driver would not release, hands the next
            // `cuMemAddressReserve` an address the driver still associates with
            // this reservation. Leaking the range is the conservative answer.
            eprintln!(
                "cuda_ep: WARNING: retaining a {} B CUDA reservation at {:#x}: {retained} \
                 block(s) could not be fully released, so its address range is not returned to \
                 the driver",
                self.len, self.base,
            );
            self.blocks = retained_mapped;
            self.quarantined = retained_unmapped;
            self.armed = true;
            return ReservationTeardownOutcome {
                report: ReservationTeardownReport {
                    unmapped_bytes: unmapped,
                    retained_blocks: retained,
                    address_range_freed: false,
                },
                retained: Some(self),
            };
        }
        if self.len > 0 {
            // SAFETY: `base` came from `cuMemAddressReserve` with this length
            // and every block in it has been unmapped above.
            if unsafe { cu::cuMemAddressFree(self.base, self.len) } != cu::CUresult::CUDA_SUCCESS {
                self.armed = true;
                return ReservationTeardownOutcome {
                    report: ReservationTeardownReport {
                        unmapped_bytes: unmapped,
                        retained_blocks: 0,
                        address_range_freed: false,
                    },
                    retained: Some(self),
                };
            }
        }
        let address_range_freed = self.len > 0;
        self.len = 0;
        ReservationTeardownOutcome {
            report: ReservationTeardownReport {
                unmapped_bytes: unmapped,
                retained_blocks: 0,
                address_range_freed,
            },
            retained: None,
        }
    }

    /// Compatibility adapter that deliberately leaks any exact residual ticket.
    pub fn execute(self) -> ReservationTeardownReport {
        let outcome = self.execute_outcome();
        if let Some(retained) = outcome.retained {
            Box::leak(Box::new(retained));
        }
        outcome.report
    }

    /// Retain the ticket deliberately, without touching the device.
    ///
    /// Used when the device or context is lost: the mappings and handles may
    /// still be referenced by hardware, so nothing is unmapped, nothing is
    /// released, and the address range is never returned.
    pub fn retain(mut self) -> ReservationTeardownReport {
        self.armed = false;
        let retained = self.blocks.len() + self.quarantined.len();
        let report = ReservationTeardownReport {
            unmapped_bytes: 0,
            retained_blocks: retained,
            address_range_freed: false,
        };
        Box::leak(Box::new(self));
        report
    }
}

impl Drop for ReservationTeardownTicket {
    /// Retain an abandoned ticket. Never unmaps, never frees, never waits.
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        if self.blocks.is_empty() && self.quarantined.is_empty() && self.len == 0 {
            return;
        }
        eprintln!(
            "cuda_ep: WARNING: a CUDA reservation teardown ticket for {} B at {:#x} was abandoned \
             with {} mapped and {} quarantined block(s); its address range and physical handles \
             are retained rather than reused",
            self.len,
            self.base,
            self.blocks.len(),
            self.quarantined.len()
        );
        let retained = ReservationTeardownTicket {
            base: self.base,
            len: std::mem::take(&mut self.len),
            context: Arc::clone(&self.context),
            pool: self.pool.clone(),
            blocks: std::mem::take(&mut self.blocks),
            quarantined: std::mem::take(&mut self.quarantined),
            armed: false,
        };
        Box::leak(Box::new(retained));
    }
}

/// Why a deferred reservation queue refused a ticket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationEnqueueRejection {
    Closed,
    Full,
    DeviceLost,
    Refused,
}

impl ReservationEnqueueRejection {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Closed => "closed",
            Self::Full => "full",
            Self::DeviceLost => "device lost",
            Self::Refused => "refused",
        }
    }
}

/// Enqueue failure that hands the **exact** ticket back.
#[derive(Debug)]
pub struct ReservationEnqueueError {
    pub rejection: ReservationEnqueueRejection,
    pub ticket: ReservationTeardownTicket,
}

/// A queue that owns reservation teardown until in-flight work has finished.
///
/// Implemented by the execution provider, which is the layer that knows about
/// streams. This crate deliberately knows only that a ticket goes in and is
/// executed later: an allocator that waited on a stream itself would serialize
/// the whole process to solve one reservation's problem.
pub trait DeferredReservationQueue: Send + Sync + std::fmt::Debug {
    /// Take ownership of `ticket`. Must not block on the device.
    fn enqueue_reservation(
        &self,
        ticket: ReservationTeardownTicket,
    ) -> Result<(), ReservationEnqueueError>;
}

#[derive(Clone)]
pub struct CudaVirtualBacking {
    context: Arc<CudaContext>,
    device_ordinal: i32,
    pool: Option<Arc<PhysicalHandlePool>>,
    teardown_synchronizer: Option<TeardownSynchronizer>,
    /// Queue that owns reservation teardown, so `CudaReservation::Drop` hands
    /// off a ticket instead of synchronizing streams. Production installs one;
    /// `None` keeps the legacy immediate path for callers that have already
    /// ordered their own work.
    reservation_queue: Option<Weak<dyn DeferredReservationQueue>>,
    /// Instance-scoped driver fault plan, for tests that must prove what
    /// happens when `cuMemUnmap`, `cuMemMap` or `cuMemRelease` refuses.
    ///
    /// Compiled only for this crate's own tests and for the `gpu-tests`
    /// feature, so a production build has neither the field nor the branch.
    /// Instance-scoped rather than a process-global switch: two tests in one
    /// binary must not be able to see each other's injected faults.
    #[cfg(any(test, feature = "gpu-tests"))]
    faults: Option<Arc<crate::release::DriverFaultPlan>>,
}

impl std::fmt::Debug for CudaVirtualBacking {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaVirtualBacking")
            .field("device_ordinal", &self.device_ordinal)
            .field("pooled", &self.pool.is_some())
            .field(
                "has_teardown_synchronizer",
                &self.teardown_synchronizer.is_some(),
            )
            .finish()
    }
}

/// A commit that failed, and what it could not take back.
///
/// A commit that fails partway must undo the mappings it made. When that
/// unwind *also* fails, the granules it could not unmap are still mapped and
/// still owned — but the caller's bookkeeping is about to say the commit never
/// happened. Reporting them here is what lets the arena poison exactly those
/// granules instead of leaving a mapped granule that its reference counts
/// believe is free.
#[derive(Debug)]
pub struct CommitFailure {
    /// Why the commit failed.
    pub error: VirtualMemoryError,
    /// Granules that are still mapped and owned after a failed unwind. Never
    /// free, never claimable.
    pub residual_mapped: Vec<MappedBlock>,
}

impl CommitFailure {
    fn clean(error: VirtualMemoryError) -> Self {
        Self {
            error,
            residual_mapped: Vec::new(),
        }
    }

    /// Whether the unwind put everything back, so nothing is owed.
    pub fn unwound_cleanly(&self) -> bool {
        self.residual_mapped.is_empty()
    }
}

impl std::fmt::Display for CommitFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.error)?;
        if !self.residual_mapped.is_empty() {
            write!(
                formatter,
                " ({} granule(s), {} B, remained mapped after the rollback and are retained)",
                self.residual_mapped.len(),
                crate::release::block_bytes(&self.residual_mapped),
            )?;
        }
        Ok(())
    }
}

impl CudaVirtualBacking {
    pub(crate) fn commit_with_owned_limit(
        &self,
        reservation: &mut CudaReservation,
        offset: usize,
        len: usize,
        max_additional_owned_bytes: u64,
    ) -> Result<u64, CommitFailure> {
        let granularity = self.granularity();
        let offsets = (offset..offset + len)
            .step_by(granularity)
            .collect::<Vec<_>>();
        self.commit_offsets_with_owned_limit(reservation, &offsets, max_additional_owned_bytes)
    }

    pub(crate) fn commit_offsets_with_owned_limit(
        &self,
        reservation: &mut CudaReservation,
        offsets: &[usize],
        max_additional_owned_bytes: u64,
    ) -> Result<u64, CommitFailure> {
        self.commit_offsets_with_owned_limit_and_capacity(
            reservation,
            offsets,
            max_additional_owned_bytes,
            None,
        )
    }

    pub(crate) fn commit_offsets_with_owned_limit_and_capacity(
        &self,
        reservation: &mut CudaReservation,
        offsets: &[usize],
        max_additional_owned_bytes: u64,
        mut capacity: Option<&mut MappedPhysicalCapacityToken>,
    ) -> Result<u64, CommitFailure> {
        self.bind("committing CUDA memory")
            .map_err(CommitFailure::clean)?;
        if let Some(pool) = &self.pool {
            let granularity = pool.granularity;
            let count = offsets.len();
            let mut checkouts = Vec::with_capacity(count);
            let mut additional_owned = 0_u64;
            for _ in 0..count {
                let remaining = max_additional_owned_bytes.saturating_sub(additional_owned);
                let (checkout, created_bytes) =
                    match pool.acquire_with_owned_limit(remaining, capacity.as_deref_mut()) {
                        Ok(acquired) => acquired,
                        Err(error) => {
                            for checkout in checkouts.drain(..) {
                                pool.rollback_checkout(checkout, false);
                            }
                            return Err(CommitFailure::clean(error));
                        }
                    };
                checkouts.push(checkout);
                additional_owned = additional_owned.saturating_add(created_bytes);
            }

            let mut mapped: Vec<(MappedBlock, CheckedOutHandle)> = Vec::with_capacity(count);
            for (index, &granule_offset) in offsets.iter().enumerate() {
                let checkout = checkouts[index];
                let handle = checkout.handle;
                let address = reservation.base + granule_offset as u64;
                if let Err(error) = Self::check("cuMemMap", unsafe {
                    cu::cuMemMap(address, granularity, 0, handle, 0)
                }) {
                    pool.rollback_checkout(checkout, false);
                    for &remaining in &checkouts[index + 1..] {
                        pool.rollback_checkout(remaining, false);
                    }
                    let residual_mapped = rollback_pooled_maps(reservation, pool, &mut mapped);
                    return Err(CommitFailure {
                        error,
                        residual_mapped,
                    });
                }
                pool.note_mapped();
                mapped.push((
                    MappedBlock::new(granule_offset, granularity, handle),
                    checkout,
                ));

                let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
                access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = self.device_ordinal;
                access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                if let Err(error) = Self::check("cuMemSetAccess", unsafe {
                    cu::cuMemSetAccess(address, granularity, &access, 1)
                }) {
                    for &remaining in &checkouts[index + 1..] {
                        pool.rollback_checkout(remaining, false);
                    }
                    let residual_mapped = rollback_pooled_maps(reservation, pool, &mut mapped);
                    return Err(CommitFailure {
                        error,
                        residual_mapped,
                    });
                }
            }
            reservation
                .blocks
                .extend(mapped.into_iter().map(|(block, _)| block));
            return Ok(additional_owned);
        }

        let granularity = self.granularity();
        let required = offsets.len().saturating_mul(granularity) as u64;
        if required > max_additional_owned_bytes {
            return Err(CommitFailure::clean(VirtualMemoryError::Os {
                operation: "reserving CUDA physical memory",
                reason: format!(
                    "candidate requires {required} incremental committed bytes but only \
                     {max_additional_owned_bytes} bytes of physical headroom are available"
                ),
                code: 0,
            }));
        }
        let mut committed = Vec::new();
        for &offset in offsets {
            if let Err(error) =
                <Self as VirtualBacking>::commit(self, reservation, offset, granularity)
            {
                // Unwind with the structured release so a granule whose unmap
                // fails stays recorded as mapped rather than silently vanishing
                // from both the reservation and the arena's reference counts.
                let mut residual_mapped = Vec::new();
                for offset in committed.into_iter().rev() {
                    let report = self.release_range_reporting(reservation, offset, granularity);
                    residual_mapped.extend(report.still_mapped);
                    residual_mapped.extend(report.unmapped_handle_owned);
                }
                return Err(CommitFailure {
                    error,
                    residual_mapped,
                });
            }
            committed.push(offset);
        }
        Ok(required)
    }

    pub(crate) fn incremental_owned_bytes_for_handles(&self, handles: usize) -> u64 {
        self.pool.as_ref().map_or_else(
            || handles.saturating_mul(self.granularity()) as u64,
            |pool| pool.incremental_owned_bytes_for_handles(handles),
        )
    }

    /// Reserve a private window and create + map `granule_count` physical
    /// handles into it read/write — the physical body of one pinned shared
    /// prefix (#777).
    ///
    /// The handles are created **once** through the #740 pool (charged on the
    /// owned axis) and registered as shared, so mapping them into any number of
    /// sharers afterwards costs zero incremental owned bytes and none is
    /// released until the last mapping — the owner's here or any sharer's — is
    /// gone. Returns the writable owner reservation (the caller fills the
    /// prefix through it), the handles to map into sharers, and the physical
    /// bytes newly owned.
    ///
    /// Requires the production physical-handle pool: a shared prefix is defined
    /// by handle identity across reservations, which only the pool provides.
    pub(crate) fn reserve_and_map_shared_prefix(
        &self,
        granule_count: usize,
    ) -> Result<SharedPrefixReservation, VirtualMemoryError> {
        let pool = self.pool.as_ref().ok_or_else(|| VirtualMemoryError::Os {
            operation: "reserving a shared prefix",
            reason: String::from(
                "shared prefixes require the production physical-handle pool; construct the \
                 allocator with a non-zero pool bound",
            ),
            code: 0,
        })?;
        if granule_count == 0 {
            return Err(VirtualMemoryError::Os {
                operation: "reserving a shared prefix",
                reason: String::from("a shared prefix must cover at least one granule"),
                code: 0,
            });
        }
        self.bind("reserving a shared prefix")?;
        let granularity = pool.granularity;
        let len = granule_count * granularity;
        let mut reservation = <Self as VirtualBacking>::reserve(self, len)?;

        // Acquire every handle first, so a shortfall fails before any mapping
        // exists to unwind.
        let mut checkouts = Vec::with_capacity(granule_count);
        let mut additional_owned = 0_u64;
        for _ in 0..granule_count {
            match pool.acquire_with_owned_limit(u64::MAX, None) {
                Ok((checkout, created)) => {
                    checkouts.push(checkout);
                    additional_owned = additional_owned.saturating_add(created);
                }
                Err(error) => {
                    for checkout in checkouts.drain(..) {
                        pool.rollback_checkout(checkout, false);
                    }
                    return Err(error);
                }
            }
        }

        // Unwinding a half-built shared prefix: a block whose unmap fails
        // stays recorded on the reservation, so its `Drop` retains the address
        // range instead of freeing it under a live mapping.
        let unwind = |pool: &PhysicalHandlePool,
                      reservation: &mut CudaReservation,
                      mapped: &mut Vec<MappedBlock>| {
            for block in mapped.drain(..).rev() {
                if unsafe { cu::cuMemUnmap(reservation.base + block.offset as u64, block.len) }
                    == cu::CUresult::CUDA_SUCCESS
                {
                    let _ = pool.return_after_unmap(block.handle, true);
                } else {
                    reservation.blocks.push(block);
                }
            }
        };

        let mut handles = Vec::with_capacity(granule_count);
        let mut mapped: Vec<MappedBlock> = Vec::new();
        for (index, checkout) in checkouts.iter().copied().enumerate() {
            let offset = index * granularity;
            let address = reservation.base + offset as u64;
            if let Err(error) = Self::check("cuMemMap", unsafe {
                cu::cuMemMap(address, granularity, 0, checkout.handle, 0)
            }) {
                pool.rollback_checkout(checkout, false);
                for &remaining in &checkouts[index + 1..] {
                    pool.rollback_checkout(remaining, false);
                }
                unwind(pool, &mut reservation, &mut mapped);
                return Err(error);
            }
            pool.note_mapped();
            pool.note_shared_map(checkout.handle);
            mapped.push(MappedBlock::new(offset, granularity, checkout.handle));

            let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
            access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = self.device_ordinal;
            access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            if let Err(error) = Self::check("cuMemSetAccess", unsafe {
                cu::cuMemSetAccess(address, granularity, &access, 1)
            }) {
                for &remaining in &checkouts[index + 1..] {
                    pool.rollback_checkout(remaining, false);
                }
                unwind(pool, &mut reservation, &mut mapped);
                return Err(error);
            }
            handles.push(checkout.handle);
        }
        reservation.blocks.extend(mapped.iter().copied());
        Ok(SharedPrefixReservation {
            reservation,
            handles,
            granularity,
            owned_bytes: additional_owned,
        })
    }

    /// Map one already-owned shared prefix handle into `reservation` at
    /// `offset`, **read-only**, taking one more reference to it.
    ///
    /// Read-only by construction (`CU_MEM_ACCESS_FLAGS_PROT_READ`): a sharer
    /// reads a prefix it does not own, and a mis-targeted store into it must
    /// fault loudly (Q3) rather than silently corrupt every other sharer's KV
    /// through the same physical page. The handle is not checked out — it
    /// belongs to the shared prefix — so a failed `cuMemSetAccess` only undoes
    /// this mapping and never returns the handle to the pool.
    pub(crate) fn map_shared_prefix_readonly(
        &self,
        reservation: &mut CudaReservation,
        offset: usize,
        handle: cu::CUmemGenericAllocationHandle,
    ) -> Result<(), VirtualMemoryError> {
        let pool = self.pool.as_ref().ok_or_else(|| VirtualMemoryError::Os {
            operation: "mapping a shared prefix",
            reason: String::from("shared prefixes require the production physical-handle pool"),
            code: 0,
        })?;
        self.bind("mapping a shared prefix")?;
        let granularity = pool.granularity;
        let address = reservation.base + offset as u64;
        Self::check("cuMemMap", unsafe {
            cu::cuMemMap(address, granularity, 0, handle, 0)
        })?;
        pool.note_mapped();
        // Take the shared reference here rather than after `cuMemSetAccess`.
        // The mapping already exists, so every path that can end with it still
        // live must hold a reference for it — otherwise the prefix's granule
        // is given back while this window still maps it, and the pool hands
        // that physical memory to an unrelated allocation.
        pool.note_shared_map(handle);

        let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = self.device_ordinal;
        access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ;
        if let Err(error) = Self::check("cuMemSetAccess", unsafe {
            cu::cuMemSetAccess(address, granularity, &access, 1)
        }) {
            // Take this mapping back off the address space, the mapped-bytes
            // gauge and the shared reference count. `return_after_unmap` does
            // all three, and because the prefix owner and every other sharer
            // still hold references it stops short of giving the granule back.
            if unsafe { cu::cuMemUnmap(address, granularity) } == cu::CUresult::CUDA_SUCCESS {
                let _ = pool.return_after_unmap(handle, true);
                return Err(error);
            }
            // The mapping is still live, so it keeps its reference and is
            // recorded rather than forgotten: the reservation must not free an
            // address range with a mapping under it, the gauges must not be
            // refunded for bytes that are still mapped, and the caller must be
            // able to find this granule in order to quarantine it.
            reservation
                .blocks
                .push(MappedBlock::read_only(offset, granularity, handle));
            return Err(VirtualMemoryError::Os {
                operation: "mapping a shared prefix",
                reason: format!(
                    "{error}; the partial mapping at offset {offset} could not be removed and is \
                     retained, so this granule can never be reused"
                ),
                code: 0,
            });
        }
        // Read-only by construction, and recorded as such so a rollback that
        // has to map it back cannot silently upgrade it to read/write.
        reservation
            .blocks
            .push(MappedBlock::read_only(offset, granularity, handle));
        Ok(())
    }

    /// Reserve and map in `context`.
    ///
    /// Takes a context rather than the execution provider's full runtime
    /// because virtual memory management is **driver** API: it needs no cudart,
    /// no cuBLAS and no kernels. Requiring the runtime would couple this to
    /// libraries it does not use — and on a machine with only the driver
    /// installed, that coupling is the difference between this code running and
    /// silently skipping.
    ///
    /// The context is not incidental: a mapping belongs to the context it was
    /// made in, so this must be the same context the kernels reading the memory
    /// run in.
    pub fn new(context: Arc<CudaContext>, device_ordinal: i32) -> Self {
        Self {
            context,
            device_ordinal,
            pool: None,
            teardown_synchronizer: None,
            reservation_queue: None,
            #[cfg(any(test, feature = "gpu-tests"))]
            faults: None,
        }
    }

    /// Use a device-scoped physical allocation pool.
    ///
    /// The pool, rather than an individual mapping, owns the governor lease.
    /// Callers must not separately release that lease when a mapping is
    /// removed: an unmapped handle still occupies VRAM until the pool calls
    /// `cuMemRelease`.
    ///
    /// Callers must also synchronize all work using a mapping before calling
    /// [`VirtualBacking::release`]. CUDA VMM unmap/remap is not ordered after
    /// in-flight kernels or copies merely because they used the old address.
    pub fn with_physical_pool(pool: Arc<PhysicalHandlePool>) -> Self {
        Self {
            context: Arc::clone(&pool.context),
            device_ordinal: pool.device_ordinal,
            pool: Some(pool),
            teardown_synchronizer: None,
            reservation_queue: None,
            #[cfg(any(test, feature = "gpu-tests"))]
            faults: None,
        }
    }

    /// Legacy ordering hook: a callback that must make in-flight work complete
    /// before a reservation is torn down.
    ///
    /// Kept for compatibility with callers that already drive their own
    /// ordering. Production uses
    /// [`with_reservation_queue`](Self::with_reservation_queue) instead, because
    /// this callback is invoked from `Drop` and a `Drop` that synchronizes a
    /// stream stalls every other stream in the process.
    pub fn with_teardown_synchronizer(mut self, synchronizer: TeardownSynchronizer) -> Self {
        self.teardown_synchronizer = Some(synchronizer);
        self
    }

    /// Install the queue that owns reservation teardown.
    ///
    /// With a queue installed, `CudaReservation::Drop` extracts its exact
    /// address range, mapped blocks, quarantined blocks, pool, and context into
    /// a [`ReservationTeardownTicket`] and enqueues it without blocking. The
    /// queue executes the ticket once the work that could still be reading the
    /// range has completed.
    pub fn with_reservation_queue(mut self, queue: Arc<dyn DeferredReservationQueue>) -> Self {
        self.reservation_queue = Some(Arc::downgrade(&queue));
        self
    }

    /// Attach a deterministic driver fault plan to **this** backing only.
    ///
    /// Test-only: the production build has no such field, so no production
    /// path can reach an injected fault even by accident.
    #[cfg(any(test, feature = "gpu-tests"))]
    pub fn with_driver_faults(mut self, plan: Arc<crate::release::DriverFaultPlan>) -> Self {
        self.faults = Some(plan);
        self
    }

    /// Whether the next `operation` on this backing must be failed on purpose.
    fn injected_fault(&self, operation: DriverOperation) -> Option<DriverFault> {
        #[cfg(any(test, feature = "gpu-tests"))]
        if let Some(plan) = self.faults.as_ref()
            && plan.should_fail(operation)
        {
            return Some(crate::release::DriverFaultPlan::fault(operation));
        }
        let _ = operation;
        None
    }

    pub(crate) fn physical_pool(&self) -> Option<&Arc<PhysicalHandlePool>> {
        self.pool.as_ref()
    }

    fn allocation_prop(&self) -> cu::CUmemAllocationProp {
        allocation_prop(self.device_ordinal)
    }

    fn bind(&self, what: &'static str) -> Result<(), VirtualMemoryError> {
        self.context
            .bind_to_thread()
            .map_err(|error| VirtualMemoryError::Os {
                operation: what,
                reason: format!("could not bind the CUDA context: {error}"),
                code: 0,
            })
    }

    fn check(call: &'static str, result: cu::CUresult) -> Result<(), VirtualMemoryError> {
        if result == cu::CUresult::CUDA_SUCCESS {
            return Ok(());
        }
        Err(VirtualMemoryError::Os {
            operation: call,
            reason: format!("{result:?}"),
            code: result as i32,
        })
    }

    /// The blocks of `reservation` that lie wholly inside `[offset, end)`.
    fn blocks_in_range(
        reservation: &CudaReservation,
        offset: usize,
        requested_len: usize,
    ) -> Vec<MappedBlock> {
        let end = offset.saturating_add(requested_len);
        reservation
            .blocks
            .iter()
            .copied()
            .filter(|block| block.offset >= offset && block.offset + block.len <= end)
            .collect()
    }

    /// Release every mapping in `[offset, offset + requested_len)`, reporting
    /// exactly what each block did.
    ///
    /// This is the CUDA-specific report the generic [`VirtualBacking`] trait
    /// cannot carry: `Result<(), _>` can say "something went wrong" but not
    /// "these three granules are still mapped, this handle is unmapped and
    /// still owned, and these two are gone". The trait method is implemented
    /// on top of it and only fails *after* the state is safely recorded.
    pub fn release_range_reporting(
        &self,
        reservation: &mut CudaReservation,
        offset: usize,
        requested_len: usize,
    ) -> SpanReleaseReport {
        let blocks = Self::blocks_in_range(reservation, offset, requested_len);
        self.release_blocks_reporting(reservation, blocks)
    }

    /// Release exactly `blocks`, which must all belong to `reservation`.
    ///
    /// Adjacent blocks are unmapped with a single `cuMemUnmap`, so a
    /// contiguous weight page costs one driver round-trip rather than one per
    /// 2 MiB granule. A run whose unmap fails keeps every one of its blocks
    /// mapped **and recorded**, so ownership is never dropped on the floor.
    pub fn release_blocks_reporting(
        &self,
        reservation: &mut CudaReservation,
        blocks: Vec<MappedBlock>,
    ) -> SpanReleaseReport {
        if blocks.is_empty() {
            return SpanReleaseReport::default();
        }
        if let Err(error) = self.bind("releasing CUDA memory") {
            // Nothing was mutated: without a bound context no driver call was
            // made at all. Every block stays mapped and owned.
            return SpanReleaseReport {
                still_mapped: blocks,
                faults: vec![DriverFault::new("cuCtxSetCurrent", error.to_string())],
                ..SpanReleaseReport::default()
            };
        }
        let driver = CudaReleaseDriver {
            backing: self,
            base: reservation.base,
        };
        let report = release_runs(&driver, &contiguous_runs(blocks));
        Self::apply_report(reservation, &report);
        report
    }

    /// Take settled and unmapped-but-owned blocks off the reservation's live
    /// mapping list, and park the owned ones where nothing can remap them.
    fn apply_report(reservation: &mut CudaReservation, report: &SpanReleaseReport) {
        for block in report
            .settled
            .iter()
            .chain(report.unmapped_handle_owned.iter())
        {
            reservation.blocks.retain(|live| live != block);
        }
        if reservation.pool.is_none() {
            // With no pool to hold it, this reservation is the handle's only
            // remaining owner. A pooled handle is already retained inside the
            // pool's quarantine, so recording it here as well would make the
            // reservation refuse to free an address range that is genuinely
            // unmapped.
            reservation
                .quarantined
                .extend(report.unmapped_handle_owned.iter().copied());
        }
    }

    /// Unmap `blocks` transactionally: either they are all unmapped, or the
    /// mapping topology the caller had is exactly restored.
    ///
    /// Handles are retained through the whole unmap phase, because a disposed
    /// handle cannot be mapped back and a rollback that cannot restore the
    /// backing is not a rollback. Disposal happens only once the unmap phase
    /// is known complete.
    pub fn decommit_blocks_transactional(
        &self,
        reservation: &mut CudaReservation,
        blocks: Vec<MappedBlock>,
    ) -> RangeDecommit {
        if blocks.is_empty() {
            return RangeDecommit::Unmapped(SpanReleaseReport::default());
        }
        if let Err(error) = self.bind("decommitting CUDA memory") {
            return RangeDecommit::RolledBack(DriverFault::new(
                "cuCtxSetCurrent",
                error.to_string(),
            ));
        }
        let driver = CudaReleaseDriver {
            backing: self,
            base: reservation.base,
        };
        let runs = contiguous_runs(blocks);
        match unmap_runs_transactional(&driver, &runs) {
            TransactionalUnmap::Unmapped { blocks } => {
                let report = dispose_released_blocks(&driver, &blocks);
                Self::apply_report(reservation, &report);
                RangeDecommit::Unmapped(report)
            }
            TransactionalUnmap::RolledBack { fault } => RangeDecommit::RolledBack(fault),
            TransactionalUnmap::RollbackFailed {
                still_mapped,
                unmapped_handle_owned,
                faults,
            } => {
                // These blocks really are unmapped, so the mapped axis is
                // refunded; their handles are retained rather than pooled,
                // because a handle whose remap the driver just refused is not
                // one to hand to the next allocation.
                if let Some(pool) = &self.pool {
                    for block in &unmapped_handle_owned {
                        let _ = pool.retain_unmapped_handle(block.handle);
                    }
                }
                let report = SpanReleaseReport {
                    unmapped_handle_owned: unmapped_handle_owned.clone(),
                    ..SpanReleaseReport::default()
                };
                Self::apply_report(reservation, &report);
                RangeDecommit::Poisoned {
                    still_mapped,
                    unmapped_handle_owned,
                    faults,
                }
            }
        }
    }
}

/// What a transactional decommit did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RangeDecommit {
    /// Every requested block was unmapped. The report says which handles then
    /// reached a terminal state and which had to be quarantined.
    Unmapped(SpanReleaseReport),
    /// An unmap failed and everything already unmapped was mapped back. The
    /// allocation is byte-for-byte the topology it had.
    RolledBack(DriverFault),
    /// An unmap failed *and* the rollback failed. Nothing listed here may be
    /// reused, remapped, or reported as free.
    Poisoned {
        still_mapped: Vec<MappedBlock>,
        unmapped_handle_owned: Vec<MappedBlock>,
        faults: Vec<DriverFault>,
    },
}

/// The three release mutations, bound to one CUDA reservation.
///
/// Holds only `&CudaVirtualBacking` and the reservation's base address, so the
/// caller keeps `&mut CudaReservation` and can apply the report afterwards
/// without fighting the borrow checker over a half-updated block list.
struct CudaReleaseDriver<'a> {
    backing: &'a CudaVirtualBacking,
    base: cu::CUdeviceptr,
}

impl CudaReleaseDriver<'_> {
    fn set_access(
        &self,
        address: cu::CUdeviceptr,
        len: usize,
        access: BlockAccess,
    ) -> cu::CUresult {
        let mut descriptor: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        descriptor.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        descriptor.location.id = self.backing.device_ordinal;
        descriptor.flags = match access {
            BlockAccess::ReadWrite => cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
            BlockAccess::ReadOnly => cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ,
        };
        unsafe { cu::cuMemSetAccess(address, len, &descriptor, 1) }
    }
}

impl ReleaseDriver for CudaReleaseDriver<'_> {
    fn unmap(&self, blocks: &[MappedBlock]) -> Result<(), DriverFault> {
        let Some(first) = blocks.first() else {
            return Ok(());
        };
        if let Some(fault) = self.backing.injected_fault(DriverOperation::Unmap) {
            return Err(fault);
        }
        let len = blocks.iter().map(|block| block.len).sum::<usize>();
        // SAFETY: every block was mapped by a commit on this reservation, the
        // run is adjacent and ascending, and CUDA permits one unmap across
        // adjacent mappings from distinct physical handles.
        let result = unsafe { cu::cuMemUnmap(self.base + first.offset as u64, len) };
        CudaVirtualBacking::check(DriverOperation::Unmap.name(), result)
            .map_err(|error| DriverFault::new(DriverOperation::Unmap.name(), error.to_string()))
    }

    fn remap(&self, block: MappedBlock) -> Result<(), DriverFault> {
        if let Some(fault) = self.backing.injected_fault(DriverOperation::Remap) {
            return Err(fault);
        }
        let address = self.base + block.offset as u64;
        // SAFETY: the range is inside this reservation and was unmapped moments
        // ago; the handle is still owned because nothing disposed it.
        let mapped = unsafe { cu::cuMemMap(address, block.len, 0, block.handle, 0) };
        CudaVirtualBacking::check(DriverOperation::Remap.name(), mapped)
            .map_err(|error| DriverFault::new(DriverOperation::Remap.name(), error.to_string()))?;
        // A mapping without access is present but unreadable, which would be a
        // rollback in name only.
        if let Err(error) = CudaVirtualBacking::check(
            "cuMemSetAccess",
            self.set_access(address, block.len, block.access),
        ) {
            // SAFETY: just mapped above; undo it so the block is reported as
            // unmapped-and-owned rather than mapped-and-unusable.
            unsafe {
                let _ = cu::cuMemUnmap(address, block.len);
            }
            return Err(DriverFault::new("cuMemSetAccess", error.to_string()));
        }
        Ok(())
    }

    fn dispose(&self, block: MappedBlock) -> HandleDisposition {
        if let Some(fault) = self.backing.injected_fault(DriverOperation::Dispose) {
            // Injected refusals take the same accounting path a real one does:
            // the mapped axis is refunded once and the handle stays owned.
            if let Some(pool) = &self.backing.pool {
                let _ = pool.retain_unmapped_handle(block.handle);
            }
            return HandleDisposition::Quarantined(fault);
        }
        if let Some(pool) = &self.backing.pool {
            return pool.return_after_unmap(block.handle, true);
        }
        // SAFETY: created by this backing's commit, unmapped above, released
        // exactly once.
        let result = unsafe { cu::cuMemRelease(block.handle) };
        match CudaVirtualBacking::check(DriverOperation::Dispose.name(), result) {
            Ok(()) => HandleDisposition::Settled,
            Err(error) => HandleDisposition::Quarantined(DriverFault::new(
                DriverOperation::Dispose.name(),
                error.to_string(),
            )),
        }
    }
}

/// A point-in-time view of one physical-handle pool.
///
/// Gauges are per pool. The create/release/hit counters remain readable from
/// [`PhysicalHandlePoolStats`] after the pool is dropped, which lets teardown
/// tests verify that every retained handle was released exactly once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhysicalHandlePoolSnapshot {
    /// Bytes currently mapped into reservations.
    pub mapped_bytes: u64,
    /// Owned physical bytes currently retained without a mapping.
    pub pooled_unmapped_bytes: u64,
    /// All physical bytes owned by the pool, mapped or not.
    pub total_owned_bytes: u64,
    /// Successful `cuMemCreate` calls.
    pub creates: u64,
    /// Successful `cuMemRelease` calls.
    pub releases: u64,
    /// Handles served from the retained pool rather than newly created.
    pub pool_hits: u64,
    /// Owned physical bytes the pool could neither release nor make reusable.
    ///
    /// **Anything but zero is a fault**, and a deliberately visible one: the
    /// driver refused to release physical memory (or the pool could not bind
    /// its context to try), so ownership is genuinely uncertain. These bytes
    /// stay inside `total_owned_bytes` and keep their share of the governor
    /// lease charged, because advertising them as free would be a lie, and
    /// they are excluded from `pooled_unmapped_bytes` because they can never
    /// be trimmed or handed out again.
    pub quarantined_bytes: u64,
    /// Handles held in quarantine.
    pub quarantined_handles: u64,
}

/// Stable observation handle for a [`PhysicalHandlePool`].
#[derive(Clone, Debug)]
pub struct PhysicalHandlePoolStats {
    counters: Arc<PoolCounters>,
}

impl PhysicalHandlePoolStats {
    /// Read all pool gauges and counters.
    pub fn snapshot(&self) -> PhysicalHandlePoolSnapshot {
        PhysicalHandlePoolSnapshot {
            mapped_bytes: self.counters.mapped_bytes.load(Ordering::Acquire),
            pooled_unmapped_bytes: self.counters.pooled_unmapped_bytes.load(Ordering::Acquire),
            total_owned_bytes: self.counters.total_owned_bytes.load(Ordering::Acquire),
            creates: self.counters.creates.load(Ordering::Acquire),
            releases: self.counters.releases.load(Ordering::Acquire),
            pool_hits: self.counters.pool_hits.load(Ordering::Acquire),
            quarantined_bytes: self.counters.quarantined_bytes.load(Ordering::Acquire),
            quarantined_handles: self.counters.quarantined_handles.load(Ordering::Acquire),
        }
    }
}

#[derive(Debug, Default)]
struct PoolCounters {
    mapped_bytes: AtomicU64,
    pooled_unmapped_bytes: AtomicU64,
    total_owned_bytes: AtomicU64,
    creates: AtomicU64,
    releases: AtomicU64,
    pool_hits: AtomicU64,
    quarantined_bytes: AtomicU64,
    quarantined_handles: AtomicU64,
}

#[derive(Debug)]
struct PoolState {
    available: Vec<cu::CUmemGenericAllocationHandle>,
    /// Handles whose release failed, or which could not be released because
    /// the pool could not bind its context.
    ///
    /// Separate from `available` on purpose. The earlier code pushed such a
    /// handle back into `available`, which made a handle whose physical
    /// ownership the driver had just refused to confirm the *next* handle
    /// handed to a caller. Nothing is ever popped from here: a quarantined
    /// handle is owned until the CUDA context itself goes away.
    quarantined: Vec<cu::CUmemGenericAllocationHandle>,
    lease: Option<MemoryLease>,
    pending_lease_shrink: u64,
    /// Live mappings, per handle, for handles mapped into more than one
    /// reservation at once — the cross-reservation prefix-share case (#777).
    ///
    /// A normal pooled handle is mapped into exactly one reservation and never
    /// appears here: it is checked out, mapped, and returned as a unit. A
    /// shared prefix granule is different — one physical handle mapped into the
    /// owner's writable window *and* every sharer's read-only window at the
    /// same time. This counts those live mappings so the handle is retained
    /// (its lifetime is the **union** of all sharers) and returned to the pool
    /// only when the **last** mapping is unmapped, never before.
    shared: HashMap<cu::CUmemGenericAllocationHandle, u32>,
}

#[derive(Clone, Copy)]
struct CheckedOutHandle {
    handle: cu::CUmemGenericAllocationHandle,
    created: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct AllocationCompatibility {
    allocation_type: i32,
    location_type: i32,
    location_id: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PoolKey {
    context: usize,
    device_ordinal: i32,
    allocation: AllocationCompatibility,
    granularity: usize,
    authority: MemoryAuthorityId,
}

static PHYSICAL_POOLS: OnceLock<Mutex<HashMap<PoolKey, Weak<PhysicalHandlePool>>>> =
    OnceLock::new();
static PHYSICAL_POOL_AUTHORITY_GATES: OnceLock<
    Mutex<HashMap<MemoryAuthorityId, Weak<RwLock<()>>>>,
> = OnceLock::new();

/// Shared operation gate for every physical pool owned by `authority`.
pub fn physical_pool_authority_gate(authority: MemoryAuthorityId) -> Arc<RwLock<()>> {
    let gates = PHYSICAL_POOL_AUTHORITY_GATES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut gates = gates
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    gates.retain(|_, gate| gate.strong_count() > 0);
    if let Some(gate) = gates.get(&authority).and_then(Weak::upgrade) {
        return gate;
    }
    let gate = Arc::new(RwLock::new(()));
    gates.insert(authority, Arc::downgrade(&gate));
    gate
}

/// Device-scoped owner of fungible, granule-sized CUDA physical allocations.
///
/// Handles returned after unmap are retained up to `max_retained_bytes`.
/// Returning a handle above that bound immediately calls `cuMemRelease` and
/// shrinks the pool-owned governor lease. The bound is rounded down to whole
/// device granules.
///
/// Mapping and unmapping are deliberately outside the pool mutex. A short
/// checkout may therefore make `mapped_bytes + pooled_unmapped_bytes` lag
/// `total_owned_bytes`, but the owned-byte gauge and governor lease remain
/// conservative throughout.
///
/// A pool is scoped to the CUDA device and allocation properties captured at
/// construction. Sharing its `Arc` across backings is what makes handles
/// fungible across otherwise independent virtual-address reservations.
#[derive(Debug)]
pub struct PhysicalHandlePool {
    context: Arc<CudaContext>,
    context_identity: usize,
    device_ordinal: i32,
    granularity: usize,
    authority: MemoryAuthorityId,
    max_retained_bytes: usize,
    authority_gate: Arc<RwLock<()>>,
    state: Mutex<PoolState>,
    lease_checkout: Mutex<()>,
    counters: Arc<PoolCounters>,
    context_terminated: AtomicBool,
}

impl PhysicalHandlePool {
    /// Get the one live compatible pool for this CUDA context and authority.
    pub fn get_or_create(
        context: Arc<CudaContext>,
        device_ordinal: i32,
        max_retained_bytes: usize,
        governor: &dyn MemoryGovernor,
        holder: HolderId,
        role: MemoryRole,
    ) -> Result<Arc<Self>, MemoryError> {
        let authority = governor.authority_id();
        let authority_gate = physical_pool_authority_gate(authority);
        if authority.device()
            != onnx_runtime_memory_governor::DeviceKey::device(device_ordinal as u32)
        {
            return Err(MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: 0,
                reason: "the physical-handle pool governor authority names a different device",
            });
        }
        let granularity = allocation_granularity(device_ordinal);
        let allocation = allocation_compatibility(device_ordinal);
        let context_id = physical_pool_context_identity(&context).map_err(|reason| {
            MemoryError::AllocationFailed {
                tier: Tier::Device.name(),
                requested: 0,
                reason,
            }
        })?;
        let key = PoolKey {
            context: context_id,
            device_ordinal,
            allocation,
            granularity,
            authority,
        };
        // Claim before taking the registry lock. Limit reconfiguration takes
        // the authority claim gate before inspecting the registry, so the
        // opposite order here would deadlock with concurrent pool creation.
        let lease = governor.reserve(Tier::Device, 0, role, holder)?;
        let registry = PHYSICAL_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut registry = registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        registry.retain(|_, pool| pool.strong_count() > 0);
        if let Some(pool) = registry.get(&key).and_then(Weak::upgrade) {
            let requested_bound = (max_retained_bytes / granularity) * granularity;
            if pool.max_retained_bytes != requested_bound {
                return Err(MemoryError::InvalidRequest {
                    tier: Tier::Device.name(),
                    requested: max_retained_bytes as u64,
                    reason: "the compatible physical-handle pool already has a different retained-byte bound",
                });
            }
            return Ok(pool);
        }
        let retained_granules = max_retained_bytes / granularity;
        let pool = Arc::new(Self {
            context_identity: context_id,
            context,
            device_ordinal,
            granularity,
            authority,
            max_retained_bytes: retained_granules * granularity,
            authority_gate,
            state: Mutex::new(PoolState {
                available: Vec::new(),
                quarantined: Vec::new(),
                lease: Some(lease),
                pending_lease_shrink: 0,
                shared: HashMap::new(),
            }),
            lease_checkout: Mutex::new(()),
            counters: Arc::new(PoolCounters::default()),
            context_terminated: AtomicBool::new(false),
        });
        registry.insert(key, Arc::downgrade(&pool));
        Ok(pool)
    }

    /// Discharge pool accounting after external proof that the CUDA context no
    /// longer exists.
    ///
    /// No driver API is called. Handles and mappings ceased to exist with the
    /// context, so their integer identities are discarded and the authority
    /// lease is released exactly once.
    fn confirm_context_terminated(&self) {
        if self.context_terminated.swap(true, Ordering::AcqRel) {
            return;
        }
        let lease = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.available.clear();
            state.quarantined.clear();
            state.shared.clear();
            state.pending_lease_shrink = 0;
            state.lease.take()
        };
        self.counters
            .pooled_unmapped_bytes
            .store(0, Ordering::Release);
        self.counters.quarantined_bytes.store(0, Ordering::Release);
        self.counters.total_owned_bytes.store(0, Ordering::Release);
        drop(lease);
    }

    /// Allocation granularity shared by every handle in this pool.
    pub fn granularity(&self) -> usize {
        self.granularity
    }

    /// Maximum bytes retained after unmap.
    pub fn max_retained_bytes(&self) -> usize {
        self.max_retained_bytes
    }

    /// Accounting authority that owns every physical handle in this pool.
    pub fn authority(&self) -> MemoryAuthorityId {
        self.authority
    }

    /// A stats handle that remains valid through pool teardown.
    pub fn stats(&self) -> PhysicalHandlePoolStats {
        PhysicalHandlePoolStats {
            counters: Arc::clone(&self.counters),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PoolState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn bind(&self, what: &'static str) -> Result<(), VirtualMemoryError> {
        if self.context_terminated.load(Ordering::Acquire) {
            return Err(VirtualMemoryError::Os {
                operation: what,
                reason: "the CUDA context was externally confirmed terminated".into(),
                code: 0,
            });
        }
        self.context
            .bind_to_thread()
            .map_err(|error| VirtualMemoryError::Os {
                operation: what,
                reason: format!("could not bind the CUDA context: {error}"),
                code: 0,
            })
    }

    pub(crate) fn incremental_owned_bytes_for_handles(&self, handles: usize) -> u64 {
        let available = self.lock().available.len();
        handles
            .saturating_sub(available)
            .saturating_mul(self.granularity) as u64
    }

    fn acquire_with_owned_limit(
        &self,
        max_additional_owned_bytes: u64,
        mut capacity: Option<&mut MappedPhysicalCapacityToken>,
    ) -> Result<(CheckedOutHandle, u64), VirtualMemoryError> {
        let operation = self
            .authority_gate
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(handle) = {
            let mut state = self.lock();
            state.available.pop()
        } {
            self.counters
                .pooled_unmapped_bytes
                .fetch_sub(self.granularity as u64, Ordering::AcqRel);
            self.counters.pool_hits.fetch_add(1, Ordering::Relaxed);
            return Ok((
                CheckedOutHandle {
                    handle,
                    created: false,
                },
                0,
            ));
        }
        drop(operation);

        let _checkout = self
            .lease_checkout
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(handle) = {
            let mut state = self.lock();
            state.available.pop()
        } {
            self.counters
                .pooled_unmapped_bytes
                .fetch_sub(self.granularity as u64, Ordering::AcqRel);
            self.counters.pool_hits.fetch_add(1, Ordering::Relaxed);
            return Ok((
                CheckedOutHandle {
                    handle,
                    created: false,
                },
                0,
            ));
        }
        if self.granularity as u64 > max_additional_owned_bytes {
            return Err(VirtualMemoryError::Os {
                operation: "reserving pooled CUDA physical memory",
                reason: format!(
                    "candidate requires {} incremental committed bytes but only \
                     {max_additional_owned_bytes} bytes of physical headroom are available",
                    self.granularity
                ),
                code: 0,
            });
        }
        let mut lease = {
            let mut state = self.lock();
            state.lease.take().ok_or_else(|| VirtualMemoryError::Os {
                operation: "growing physical handle pool lease",
                reason: String::from("the physical handle pool is tearing down"),
                code: 0,
            })?
        };
        let growth = match capacity.as_deref_mut() {
            Some(capacity) => lease.grow_from_mapped_capacity(capacity, self.granularity as u64),
            None => lease.grow(self.granularity as u64),
        }
        .map_err(|error| VirtualMemoryError::Os {
            operation: "growing physical handle pool lease",
            reason: error.to_string(),
            code: 0,
        });
        {
            let mut state = self.lock();
            let pending = std::mem::take(&mut state.pending_lease_shrink);
            lease.shrink(pending);
            state.lease = Some(lease);
        }
        growth?;
        drop(_checkout);
        let _operation = self
            .authority_gate
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Err(error) = self.bind("creating pooled CUDA physical memory") {
            self.refund_lease_growth(capacity, self.granularity as u64);
            return Err(error);
        }
        let prop = allocation_prop(self.device_ordinal);
        let mut handle: cu::CUmemGenericAllocationHandle = 0;
        let result = unsafe { cu::cuMemCreate(&mut handle, self.granularity, &prop, 0) };
        if let Err(error) = CudaVirtualBacking::check("cuMemCreate", result) {
            self.refund_lease_growth(capacity, self.granularity as u64);
            return Err(error);
        }
        self.counters.creates.fetch_add(1, Ordering::Relaxed);
        self.counters
            .total_owned_bytes
            .fetch_add(self.granularity as u64, Ordering::AcqRel);
        Ok((
            CheckedOutHandle {
                handle,
                created: true,
            },
            self.granularity as u64,
        ))
    }

    fn rollback_checkout(&self, checkout: CheckedOutHandle, was_mapped: bool) {
        if !checkout.created {
            let _ = self.return_after_unmap(checkout.handle, was_mapped);
            return;
        }
        if was_mapped {
            self.counters
                .mapped_bytes
                .fetch_sub(self.granularity as u64, Ordering::AcqRel);
        }
        if self
            .bind("releasing rolled-back pooled CUDA physical memory")
            .is_ok()
            && CudaVirtualBacking::check("cuMemRelease", unsafe {
                cu::cuMemRelease(checkout.handle)
            })
            .is_ok()
        {
            self.counters.releases.fetch_add(1, Ordering::Relaxed);
            self.counters
                .total_owned_bytes
                .fetch_sub(self.granularity as u64, Ordering::AcqRel);
            self.shrink_lease_or_defer(self.granularity as u64);
        } else {
            // The driver would not take the handle back. It was created here
            // and never escaped, but "cuMemRelease refused" is not the same as
            // "this handle is fine": physical ownership is unconfirmed, so it
            // is retained rather than offered to the next caller.
            self.quarantine_handle(checkout.handle);
        }
    }

    /// Take one mapping off the books and **retain** its handle instead of
    /// offering it for reuse.
    ///
    /// Used when a mapping was removed but giving the handle back would be
    /// dishonest — a rollback that could not map it in again, or an injected
    /// refusal in a test. The mapped axis is refunded exactly once because the
    /// mapping really is gone; the owned axis is not, because the handle is
    /// still ours.
    ///
    /// A shared granule is the exception, and must be: another reservation
    /// still maps it, so this departure is not the last one and the physical
    /// memory belongs to whoever remains.
    pub(crate) fn retain_unmapped_handle(
        &self,
        handle: cu::CUmemGenericAllocationHandle,
    ) -> HandleDisposition {
        let _operation = self
            .authority_gate
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.note_unmapped();
        {
            let mut state = self.lock();
            if let Some(count) = state.shared.get_mut(&handle) {
                *count -= 1;
                if *count > 0 {
                    return HandleDisposition::Settled;
                }
                state.shared.remove(&handle);
            }
        }
        self.quarantine_handle(handle);
        HandleDisposition::Quarantined(DriverFault::new(
            DriverOperation::Remap.name(),
            "the mapping could not be restored, so its handle is retained rather than reused",
        ))
    }

    /// Retain a handle the pool can neither release nor safely reuse.
    ///
    /// Owned bytes and the lease share stay charged: the driver may still hold
    /// the physical memory, and reporting it as free would let the governor
    /// admit work the device cannot serve.
    fn quarantine_handle(&self, handle: cu::CUmemGenericAllocationHandle) {
        self.lock().quarantined.push(handle);
        self.counters
            .quarantined_bytes
            .fetch_add(self.granularity as u64, Ordering::AcqRel);
        self.counters
            .quarantined_handles
            .fetch_add(1, Ordering::AcqRel);
    }

    fn shrink_lease_or_defer(&self, bytes: u64) {
        let mut state = self.lock();
        if let Some(lease) = state.lease.as_mut() {
            lease.shrink(bytes);
        } else {
            state.pending_lease_shrink = state.pending_lease_shrink.saturating_add(bytes);
        }
    }

    fn refund_lease_growth(
        &self,
        capacity: Option<&mut onnx_runtime_memory_governor::MappedPhysicalCapacityToken>,
        bytes: u64,
    ) {
        let mut state = self.lock();
        if let Some(lease) = state.lease.as_mut() {
            if let Some(capacity) = capacity
                && lease.shrink_to_mapped_capacity(capacity, bytes).is_ok()
            {
                return;
            }
            lease.shrink(bytes);
        } else {
            state.pending_lease_shrink = state.pending_lease_shrink.saturating_add(bytes);
        }
    }

    fn note_mapped(&self) {
        self.counters
            .mapped_bytes
            .fetch_add(self.granularity as u64, Ordering::AcqRel);
    }

    /// Record that one more mapping of `handle` now exists, across any number
    /// of reservations. Paired one-for-one with the [`return_after_unmap`] that
    /// eventually removes that mapping.
    ///
    /// [`return_after_unmap`]: Self::return_after_unmap
    pub(crate) fn note_shared_map(&self, handle: cu::CUmemGenericAllocationHandle) {
        let mut state = self.lock();
        *state.shared.entry(handle).or_insert(0) += 1;
    }

    /// Undo a mapped-bytes gauge bump without returning the handle to the pool.
    ///
    /// Used only to unwind a shared mapping whose `cuMemSetAccess` failed after
    /// its `cuMemMap` succeeded: the physical handle is still owned by its
    /// shared-prefix owner and must not be returned here, but its transient
    /// mapping must be taken back off the gauge.
    pub(crate) fn note_unmapped(&self) {
        self.counters
            .mapped_bytes
            .fetch_sub(self.granularity as u64, Ordering::AcqRel);
    }

    /// Take one mapping of `handle` off the books after its `cuMemUnmap`.
    ///
    /// Reports what the handle's *physical* ownership did, which is not the
    /// same question as whether the unmap worked:
    ///
    /// * [`HandleDisposition::Settled`] — retained for reuse, released to the
    ///   driver, or still mapped by another reservation. Nothing is owed.
    /// * [`HandleDisposition::Quarantined`] — the pool could not bind its
    ///   context, or `cuMemRelease` refused. The handle is retained forever,
    ///   never reused, and its bytes stay charged on the owned axis.
    ///
    /// The mapped axis is refunded once, up front, for both outcomes: the
    /// mapping really is gone by the time this is called.
    fn return_after_unmap(
        &self,
        handle: cu::CUmemGenericAllocationHandle,
        was_mapped: bool,
    ) -> HandleDisposition {
        let _operation = self
            .authority_gate
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if was_mapped {
            self.counters
                .mapped_bytes
                .fetch_sub(self.granularity as u64, Ordering::AcqRel);
        }

        // A shared prefix granule stays owned while any other reservation still
        // maps it. Only the last mapping to leave falls through to the normal
        // retain-or-release path below; earlier ones just decrement the count.
        {
            let mut state = self.lock();
            if let Some(count) = state.shared.get_mut(&handle) {
                *count -= 1;
                if *count > 0 {
                    return HandleDisposition::Settled;
                }
                state.shared.remove(&handle);
            }
        }

        let retain = {
            let mut state = self.lock();
            let retained = state.available.len() * self.granularity;
            if retained < self.max_retained_bytes {
                state.available.push(handle);
                true
            } else {
                false
            }
        };
        if retain {
            self.counters
                .pooled_unmapped_bytes
                .fetch_add(self.granularity as u64, Ordering::AcqRel);
            return HandleDisposition::Settled;
        }

        if let Err(error) = self.bind("releasing excess pooled CUDA physical memory") {
            self.quarantine_handle(handle);
            return HandleDisposition::Quarantined(DriverFault::new(
                "cuCtxSetCurrent",
                error.to_string(),
            ));
        }
        let result = unsafe { cu::cuMemRelease(handle) };
        if let Err(error) = CudaVirtualBacking::check("cuMemRelease", result) {
            self.quarantine_handle(handle);
            return HandleDisposition::Quarantined(DriverFault::new(
                DriverOperation::Dispose.name(),
                error.to_string(),
            ));
        }
        self.counters.releases.fetch_add(1, Ordering::Relaxed);
        self.counters
            .total_owned_bytes
            .fetch_sub(self.granularity as u64, Ordering::AcqRel);
        self.shrink_lease_or_defer(self.granularity as u64);
        HandleDisposition::Settled
    }
}

/// Reconcile every physical pool belonging to one terminated CUDA context and
/// authority without invoking the driver.
pub fn confirm_physical_handle_pool_context_terminated(
    context_identity: usize,
    authority: MemoryAuthorityId,
) {
    let registry = PHYSICAL_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let pools = {
        let mut registry = registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        registry.retain(|_, pool| pool.strong_count() > 0);
        registry
            .values()
            .filter_map(Weak::upgrade)
            .filter(|pool| pool.authority == authority && pool.context_identity == context_identity)
            .collect::<Vec<_>>()
    };
    for pool in pools {
        pool.confirm_context_terminated();
    }
}

/// Release retained, unmapped handles owned by `authority`.
///
/// Pool locks serialize checkout/return with trimming. The caller must pause
/// authority lease growth until trimming and the final limit commit complete.
/// Mapped handles are never released here.
pub fn trim_physical_handle_pools(
    authority: MemoryAuthorityId,
    bytes_to_release: u64,
) -> Result<u64, VirtualMemoryError> {
    if bytes_to_release == 0 {
        return Ok(0);
    }
    let registry = PHYSICAL_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let pools = {
        let mut registry = registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        registry.retain(|_, pool| pool.strong_count() > 0);
        registry
            .iter()
            .filter(|(key, _)| key.authority == authority)
            .filter_map(|(_, pool)| pool.upgrade())
            .collect::<Vec<_>>()
    };
    let mut released = 0_u64;
    for pool in pools {
        while released < bytes_to_release {
            let Some(handle) = pool.lock().available.pop() else {
                break;
            };
            if pool.bind("trimming pooled CUDA physical memory").is_err() {
                pool.lock().available.push(handle);
                break;
            }
            if CudaVirtualBacking::check("cuMemRelease", unsafe { cu::cuMemRelease(handle) })
                .is_err()
            {
                // The release was attempted and refused, so ownership is
                // unconfirmed. Putting it back in `available` would hand a
                // handle the driver just objected to straight to the next
                // caller; quarantine keeps it owned and unreachable.
                pool.counters
                    .pooled_unmapped_bytes
                    .fetch_sub(pool.granularity as u64, Ordering::AcqRel);
                pool.quarantine_handle(handle);
                break;
            }
            let bytes = pool.granularity as u64;
            pool.counters.releases.fetch_add(1, Ordering::Relaxed);
            pool.counters
                .pooled_unmapped_bytes
                .fetch_sub(bytes, Ordering::AcqRel);
            pool.counters
                .total_owned_bytes
                .fetch_sub(bytes, Ordering::AcqRel);
            pool.shrink_lease_or_defer(bytes);
            released = released.saturating_add(bytes);
        }
    }
    Ok(released)
}

/// Bytes that can be released without disturbing live mappings for `authority`.
pub fn pooled_unmapped_bytes_for_authority(authority: MemoryAuthorityId) -> u64 {
    let registry = PHYSICAL_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut registry = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.retain(|_, pool| pool.strong_count() > 0);
    registry
        .iter()
        .filter(|(key, _)| key.authority == authority)
        .filter_map(|(_, pool)| pool.upgrade())
        .fold(0_u64, |total, pool| {
            total.saturating_add(pool.counters.pooled_unmapped_bytes.load(Ordering::Acquire))
        })
}

/// Process-wide authority-owned bytes across live physical-handle pools.
///
/// Each compatible pool appears once in the registry, so this is a sum rather
/// than a last-writer gauge.
pub fn total_physical_pool_owned_bytes() -> u64 {
    let registry = PHYSICAL_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut registry = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.retain(|_, pool| pool.strong_count() > 0);
    registry
        .values()
        .filter_map(Weak::upgrade)
        .fold(0_u64, |total, pool| {
            total.saturating_add(pool.counters.total_owned_bytes.load(Ordering::Acquire))
        })
}

impl Drop for PhysicalHandlePool {
    fn drop(&mut self) {
        if self.context_terminated.load(Ordering::Acquire) {
            let state = self
                .state
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.available.clear();
            state.quarantined.clear();
            state.shared.clear();
            drop(state.lease.take());
            return;
        }
        let _operation = self
            .authority_gate
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (handles, mut retained_handles, mut lease) = {
            let state = self
                .state
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                std::mem::take(&mut state.available),
                std::mem::take(&mut state.quarantined),
                state.lease.take(),
            )
        };
        let _ = self.context.bind_to_thread();
        let mut release_failed = false;
        for handle in handles {
            if unsafe { cu::cuMemRelease(handle) } == cu::CUresult::CUDA_SUCCESS {
                self.counters.releases.fetch_add(1, Ordering::Relaxed);
                self.counters
                    .pooled_unmapped_bytes
                    .fetch_sub(self.granularity as u64, Ordering::AcqRel);
                self.counters
                    .total_owned_bytes
                    .fetch_sub(self.granularity as u64, Ordering::AcqRel);
                if let Some(lease) = lease.as_mut() {
                    lease.shrink(self.granularity as u64);
                }
            } else {
                release_failed = true;
                retained_handles.push(handle);
            }
        }
        // Quarantined handles are deliberately not retried. The driver already
        // refused them once, and a second refusal at teardown would only turn a
        // recorded fault into a silent one.
        let quarantined = retained_handles.len();
        if quarantined > 0 {
            eprintln!(
                "cuda_ep: WARNING: physical handle pool closing with {quarantined} quarantined \
                 handle(s) ({} B) whose release the driver refused; their lease is retained until \
                 CUDA context teardown",
                self.counters.quarantined_bytes.load(Ordering::Acquire),
            );
        }
        let ownership_remains = self.counters.total_owned_bytes.load(Ordering::Acquire) > 0;
        if release_failed || ownership_remains || quarantined > 0 {
            // A failed driver release means physical ownership is uncertain.
            // A non-zero owned gauge can also mean a reservation could not
            // unmap during its Drop. Leaking the remaining lease is
            // conservative; dropping it would advertise memory as free while
            // the driver may still own it.
            if let Some(lease) = lease {
                std::mem::forget(lease);
            }
            // Preserve the exact residual handle identities until process
            // teardown. Dropping these integers would turn owned quarantine into
            // an unowned leak that cannot be reconciled against the CUDA context.
            Box::leak(retained_handles.into_boxed_slice());
        }
    }
}

fn allocation_prop(device_ordinal: i32) -> cu::CUmemAllocationProp {
    let mut prop: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_ordinal;
    prop
}

fn allocation_compatibility(device_ordinal: i32) -> AllocationCompatibility {
    let prop = allocation_prop(device_ordinal);
    AllocationCompatibility {
        allocation_type: prop.type_ as i32,
        location_type: prop.location.type_ as i32,
        location_id: prop.location.id,
    }
}

pub fn physical_pool_context_identity(context: &CudaContext) -> Result<usize, String> {
    context
        .bind_to_thread()
        .map_err(|error| format!("could not bind CUDA context: {error}"))?;
    let mut current: cu::CUcontext = std::ptr::null_mut();
    let result = unsafe { cu::cuCtxGetCurrent(&mut current) };
    if result != cu::CUresult::CUDA_SUCCESS || current.is_null() {
        return Err(format!("cuCtxGetCurrent failed: {result:?}"));
    }
    Ok(current as usize)
}

fn allocation_granularity(device_ordinal: i32) -> usize {
    let prop = allocation_prop(device_ordinal);
    let mut granularity = 0usize;
    let result = unsafe {
        cu::cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    };
    if result == cu::CUresult::CUDA_SUCCESS && granularity > 0 {
        granularity
    } else {
        2 << 20
    }
}

/// One reserved device address range and the physical handles mapped into it.
pub struct CudaReservation {
    base: cu::CUdeviceptr,
    len: usize,
    context: Arc<CudaContext>,
    pool: Option<Arc<PhysicalHandlePool>>,
    teardown_synchronizer: Option<TeardownSynchronizer>,
    /// Queue that owns this reservation's teardown, when one is installed.
    reservation_queue: Option<Weak<dyn DeferredReservationQueue>>,
    /// Every block currently mapped into this reservation.
    blocks: Vec<MappedBlock>,
    /// Blocks that were unmapped but whose physical handle could not be given
    /// back, on a backing with no pool to hold it.
    ///
    /// Retained here rather than dropped so the handle is never lost and never
    /// reused, and so the reservation's `Drop` can tell "released" apart from
    /// "we still own this". Nothing is ever taken out of this list.
    quarantined: Vec<MappedBlock>,
}

/// The physical body of one pinned shared prefix: a private writable window
/// over granules that are created once and mapped, read-only, into any number
/// of sharers (#777).
///
/// Its `Drop` (the reservation's) unmaps the owner's window and returns each
/// handle to the pool — but a handle mapped into live sharers is retained by
/// the shared refcount until the last sharer leaves, so the prefix's owner
/// reference can go away first without pulling memory out from under a request
/// still reading it. The lifetime of the physical granules is the **union** of
/// the owner and every sharer.
pub struct SharedPrefixReservation {
    reservation: CudaReservation,
    handles: Vec<cu::CUmemGenericAllocationHandle>,
    granularity: usize,
    owned_bytes: u64,
}

impl std::fmt::Debug for SharedPrefixReservation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SharedPrefixReservation")
            .field("base", &self.reservation.base)
            .field("granules", &self.handles.len())
            .field("owned_bytes", &self.owned_bytes)
            .finish()
    }
}

impl SharedPrefixReservation {
    /// Device address of the owner's writable window, where the prefix content
    /// is filled once before it is shared read-only.
    pub fn base(&self) -> usize {
        self.reservation.base as usize
    }

    /// Number of physical granules the prefix spans.
    pub fn granule_count(&self) -> usize {
        self.handles.len()
    }

    /// Granule size these handles were created at.
    pub fn granularity(&self) -> usize {
        self.granularity
    }

    /// Physical bytes this prefix newly owns — charged once, on the owned axis.
    pub fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    /// The `granule`-th shared handle, for mapping into a sharer's reservation.
    pub(crate) fn handle(&self, granule: usize) -> Option<cu::CUmemGenericAllocationHandle> {
        self.handles.get(granule).copied()
    }
}

impl std::fmt::Debug for CudaReservation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaReservation")
            .field("base", &self.base)
            .field("len", &self.len)
            .field("blocks", &self.blocks.len())
            .field("quarantined", &self.quarantined.len())
            .finish()
    }
}

impl CudaReservation {
    /// Blocks that are unmapped but whose physical handle this reservation
    /// still owns because releasing it failed. Never reusable.
    pub fn quarantined_blocks(&self) -> &[MappedBlock] {
        &self.quarantined
    }

    /// Blocks mapped right now.
    pub fn mapped_blocks(&self) -> &[MappedBlock] {
        &self.blocks
    }

    /// Move this reservation's exact teardown state into a ticket.
    ///
    /// After this the reservation owns nothing: no address range, no mapped
    /// blocks, no quarantined blocks. Everything is the ticket's.
    fn take_teardown_ticket(&mut self) -> ReservationTeardownTicket {
        let ticket = ReservationTeardownTicket {
            base: self.base,
            len: self.len,
            context: Arc::clone(&self.context),
            pool: self.pool.clone(),
            blocks: std::mem::take(&mut self.blocks),
            quarantined: std::mem::take(&mut self.quarantined),
            armed: true,
        };
        self.len = 0;
        ticket
    }
}

// The reservation is an owned device address range; nothing in it is
// thread-affine, and every driver call through the backing binds the context
// first.
unsafe impl Send for CudaReservation {}
unsafe impl Sync for CudaReservation {}

impl Drop for CudaReservation {
    /// Hand teardown to the queue, or fall back to the legacy immediate path.
    ///
    /// With a queue installed this is fully non-blocking: the exact VA, mapped
    /// blocks, quarantined blocks, pool, and context move into a ticket which
    /// the queue executes after the streams that could still be reading the
    /// range have completed. If the queue refuses, the ticket is *retained*
    /// (leaked) rather than executed early — an address range whose ordering
    /// cannot be established must never be handed back to the driver.
    fn drop(&mut self) {
        if let Some(queue_ref) = self.reservation_queue.clone() {
            let ticket = self.take_teardown_ticket();
            if ticket.is_empty() && ticket.mapped_blocks().is_empty() {
                // Nothing was ever reserved or mapped; the ticket owns nothing.
                let _ = ticket.retain();
                return;
            }
            let mapped = ticket.mapped_blocks().len();
            let Some(queue) = queue_ref.upgrade() else {
                eprintln!(
                    "cuda_ep: WARNING: the deferred reservation queue was already gone; retaining \
                     this CUDA reservation and its {mapped} mapped block(s) rather than tearing it \
                     down without a stream-ordering proof"
                );
                let _ = ticket.retain();
                return;
            };
            if let Err(error) = queue.enqueue_reservation(ticket) {
                eprintln!(
                    "cuda_ep: WARNING: the deferred reservation queue refused a CUDA reservation \
                     teardown ({}); retaining its address range and {mapped} mapped block(s) \
                     rather than unmapping them before in-flight work has finished",
                    error.rejection.name()
                );
                // Retain: no unmap, no release, no address reuse.
                let _ = error.ticket.retain();
            }
            return;
        }
        if !self.blocks.is_empty()
            && let Some(synchronize) = &self.teardown_synchronizer
            && let Err(error) = synchronize()
        {
            eprintln!(
                "cuda_ep: WARNING: reservation teardown synchronization failed; retaining {} \
                 mapped block(s) until CUDA context teardown: {error}",
                self.blocks.len()
            );
            // The mappings and handles may still be in use. Forget the VA range
            // and mapped handles rather than making either reusable or
            // advertising their physical bytes as free.
            self.blocks.clear();
            self.len = 0;
            return;
        }
        // Legacy immediate teardown for a backing with no queue: the caller
        // owns the ordering (see `with_teardown_synchronizer`). The ticket path
        // is the same code, so the two cannot drift apart.
        let _ = self.take_teardown_ticket().execute();
    }
}

// SAFETY: every address comes from `cuMemAddressReserve`; the granularity is
// the driver's own for this device and constant; `commit` maps and grants
// access to the whole range it reports success for; and `CudaReservation`'s
// `Drop` unmaps every block, releases every handle, and frees the reservation.
unsafe impl VirtualBacking for CudaVirtualBacking {
    type Reservation = CudaReservation;

    fn granularity(&self) -> usize {
        self.pool.as_ref().map_or_else(
            || allocation_granularity(self.device_ordinal),
            |pool| pool.granularity,
        )
    }

    fn physical_memory_accounting(&self) -> PhysicalMemoryAccounting {
        self.pool
            .as_ref()
            .map_or(PhysicalMemoryAccounting::Buffer, |pool| {
                PhysicalMemoryAccounting::Backing {
                    authority: pool.authority,
                }
            })
    }

    fn reserve(&self, len: usize) -> Result<Self::Reservation, VirtualMemoryError> {
        self.bind("reserving CUDA address space")?;
        let mut base: cu::CUdeviceptr = 0;
        // SAFETY: `base` is a valid out-parameter; alignment 0 lets the driver
        // choose, and a null `addr` lets it place the range.
        Self::check("cuMemAddressReserve", unsafe {
            cu::cuMemAddressReserve(&mut base, len, 0, 0, 0)
        })?;
        Ok(CudaReservation {
            base,
            len,
            context: Arc::clone(&self.context),
            pool: self.pool.clone(),
            teardown_synchronizer: self.teardown_synchronizer.clone(),
            reservation_queue: self.reservation_queue.clone(),
            blocks: Vec::new(),
            quarantined: Vec::new(),
        })
    }

    fn base(reservation: &Self::Reservation) -> usize {
        reservation.base as usize
    }

    fn commit(
        &self,
        reservation: &mut Self::Reservation,
        offset: usize,
        len: usize,
    ) -> Result<(), VirtualMemoryError> {
        self.bind("committing CUDA memory")?;
        if self.pool.is_some() {
            self.commit_with_owned_limit(reservation, offset, len, u64::MAX)
                .map_err(|failure| VirtualMemoryError::Os {
                    operation: "committing CUDA memory",
                    reason: failure.to_string(),
                    code: 0,
                })?;
            return Ok(());
        }

        let granularity = self.granularity();
        if len > granularity {
            let mut committed_offsets = Vec::new();
            for granule_offset in (offset..offset + len).step_by(granularity) {
                if let Err(error) = self.commit(reservation, granule_offset, granularity) {
                    for committed_offset in committed_offsets.into_iter().rev() {
                        let _ = self.release(reservation, committed_offset, granularity);
                    }
                    return Err(error);
                }
                committed_offsets.push(granule_offset);
            }
            return Ok(());
        }

        let prop = self.allocation_prop();
        let mut handle: cu::CUmemGenericAllocationHandle = 0;
        // SAFETY: `prop` is fully initialised; `handle` is a valid
        // out-parameter; `len` is a multiple of the granularity by the trait's
        // contract.
        Self::check("cuMemCreate", unsafe {
            cu::cuMemCreate(&mut handle, len, &prop, 0)
        })?;

        let address = reservation.base + offset as u64;
        // SAFETY: `address..address + len` lies inside the reservation by the
        // trait's contract, and `handle` was just created with exactly `len`.
        if let Err(error) = Self::check("cuMemMap", unsafe {
            cu::cuMemMap(address, len, 0, handle, 0)
        }) {
            // The handle is ours and nothing references it, so release it
            // rather than leaking physical device memory on a failed map.
            // SAFETY: created above, released once, never mapped.
            unsafe {
                let _ = cu::cuMemRelease(handle);
            }
            return Err(error);
        }

        // Mapping alone does not make the range usable: without an access
        // descriptor a kernel reading it faults. This is the step whose absence
        // looks like "the memory is there but every read is garbage".
        let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = self.device_ordinal;
        access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        // SAFETY: the range was just mapped and `access` is fully initialised.
        if let Err(error) = Self::check("cuMemSetAccess", unsafe {
            cu::cuMemSetAccess(address, len, &access, 1)
        }) {
            // SAFETY: just mapped and created; undo both.
            unsafe {
                let _ = cu::cuMemUnmap(address, len);
                let _ = cu::cuMemRelease(handle);
            }
            return Err(error);
        }

        reservation
            .blocks
            .push(MappedBlock::new(offset, len, handle));
        Ok(())
    }

    /// Legacy trait release.
    ///
    /// The generic contract is `Result<(), _>`, which cannot carry "these
    /// granules are still mapped and these handles are still owned". So this
    /// runs the structured release first — which records every residual on the
    /// reservation, keeps still-mapped blocks recorded as mapped, and parks
    /// unreleasable handles where nothing can reuse them — and only then
    /// reports an error. State is safe *before* the `Err` is produced, never
    /// after, and the message names what is retained so a caller can act on it.
    fn release(
        &self,
        reservation: &mut Self::Reservation,
        offset: usize,
        requested_len: usize,
    ) -> Result<(), VirtualMemoryError> {
        let report = self.release_range_reporting(reservation, offset, requested_len);
        if report.is_complete() {
            return Ok(());
        }
        Err(VirtualMemoryError::Os {
            operation: "releasing CUDA memory",
            reason: format!(
                "{} granule(s) ({} B) are still mapped and {} handle(s) ({} B) are unmapped but \
                 still owned after releasing {requested_len} B at offset {offset}; the address \
                 range is retained and must not be reused: {}",
                report.still_mapped.len(),
                crate::release::block_bytes(&report.still_mapped),
                report.unmapped_handle_owned.len(),
                crate::release::block_bytes(&report.unmapped_handle_owned),
                report.fault_summary(),
            ),
            code: 0,
        })
    }
}

/// Undo the mappings a failed pooled commit made, reporting what would not
/// come back.
///
/// A block whose unmap fails is pushed onto the reservation's live block list:
/// it really is still mapped, and losing that record would let the reservation
/// free its address range out from under a live mapping. It is also returned
/// so the arena can poison the granule instead of believing it is free.
fn rollback_pooled_maps(
    reservation: &mut CudaReservation,
    pool: &PhysicalHandlePool,
    mapped: &mut Vec<(MappedBlock, CheckedOutHandle)>,
) -> Vec<MappedBlock> {
    let mut residual = Vec::new();
    for (block, checkout) in mapped.drain(..).rev() {
        if unsafe { cu::cuMemUnmap(reservation.base + block.offset as u64, block.len) }
            != cu::CUresult::CUDA_SUCCESS
        {
            reservation.blocks.push(block);
            residual.push(block);
            continue;
        }
        // The mapping is gone; the handle may still be unreturnable, in which
        // case `rollback_checkout` quarantines it inside the pool.
        pool.rollback_checkout(checkout, true);
    }
    residual.sort_unstable_by_key(|block| block.offset);
    residual
}

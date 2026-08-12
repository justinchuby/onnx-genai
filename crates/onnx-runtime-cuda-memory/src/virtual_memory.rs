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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

use cudarc::driver::sys as cu;
use onnx_runtime_memory_governor::{
    HolderId, MappedPhysicalCapacityToken, MemoryAuthorityId, MemoryError, MemoryGovernor,
    MemoryLease, MemoryRole, Tier,
};
use onnx_runtime_virtual_memory::{PhysicalMemoryAccounting, VirtualBacking, VirtualMemoryError};

use cudarc::driver::CudaContext;

/// Device address space, backed by CUDA physical allocations.
///
/// Holds the runtime so the CUDA context is bound before every driver call —
/// the reservation and its mappings belong to a context, and touching them from
/// an unbound thread is a driver error rather than a silent wrong answer.
pub type TeardownSynchronizer = Arc<dyn Fn() -> Result<(), String> + Send + Sync>;

#[derive(Clone)]
pub struct CudaVirtualBacking {
    context: Arc<CudaContext>,
    device_ordinal: i32,
    pool: Option<Arc<PhysicalHandlePool>>,
    teardown_synchronizer: Option<TeardownSynchronizer>,
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

impl CudaVirtualBacking {
    pub(crate) fn commit_with_owned_limit(
        &self,
        reservation: &mut CudaReservation,
        offset: usize,
        len: usize,
        max_additional_owned_bytes: u64,
    ) -> Result<u64, VirtualMemoryError> {
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
    ) -> Result<u64, VirtualMemoryError> {
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
    ) -> Result<u64, VirtualMemoryError> {
        self.bind("committing CUDA memory")?;
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
                            return Err(error);
                        }
                    };
                checkouts.push(checkout);
                additional_owned = additional_owned.saturating_add(created_bytes);
            }

            let mut mapped = Vec::with_capacity(count);
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
                    rollback_pooled_maps(reservation, pool, &mut mapped);
                    return Err(error);
                }
                pool.note_mapped();
                mapped.push((granule_offset, granularity, checkout));

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
                    rollback_pooled_maps(reservation, pool, &mut mapped);
                    return Err(error);
                }
            }
            reservation.blocks.extend(
                mapped
                    .into_iter()
                    .map(|(offset, len, checkout)| (offset, len, checkout.handle)),
            );
            return Ok(additional_owned);
        }

        let granularity = self.granularity();
        let required = offsets.len().saturating_mul(granularity) as u64;
        if required > max_additional_owned_bytes {
            return Err(VirtualMemoryError::Os {
                operation: "reserving CUDA physical memory",
                reason: format!(
                    "candidate requires {required} incremental committed bytes but only \
                     {max_additional_owned_bytes} bytes of physical headroom are available"
                ),
                code: 0,
            });
        }
        let mut committed = Vec::new();
        for &offset in offsets {
            if let Err(error) =
                <Self as VirtualBacking>::commit(self, reservation, offset, granularity)
            {
                for offset in committed.into_iter().rev() {
                    let _ =
                        <Self as VirtualBacking>::release(self, reservation, offset, granularity);
                }
                return Err(error);
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

        let unwind =
            |pool: &PhysicalHandlePool,
             reservation: &CudaReservation,
             mapped: &mut Vec<(usize, usize, cu::CUmemGenericAllocationHandle)>| {
                for (offset, len, handle) in mapped.drain(..).rev() {
                    if unsafe { cu::cuMemUnmap(reservation.base + offset as u64, len) }
                        == cu::CUresult::CUDA_SUCCESS
                    {
                        let _ = pool.return_after_unmap(handle, true);
                    }
                }
            };

        let mut handles = Vec::with_capacity(granule_count);
        let mut mapped: Vec<(usize, usize, cu::CUmemGenericAllocationHandle)> = Vec::new();
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
                unwind(pool, &reservation, &mut mapped);
                return Err(error);
            }
            pool.note_mapped();
            pool.note_shared_map(checkout.handle);
            mapped.push((offset, granularity, checkout.handle));

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
                unwind(pool, &reservation, &mut mapped);
                return Err(error);
            }
            handles.push(checkout.handle);
        }
        reservation.blocks.extend(
            mapped
                .iter()
                .map(|&(offset, len, handle)| (offset, len, handle)),
        );
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

        let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = self.device_ordinal;
        access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ;
        if let Err(error) = Self::check("cuMemSetAccess", unsafe {
            cu::cuMemSetAccess(address, granularity, &access, 1)
        }) {
            // The handle stays owned by the shared prefix; only take this
            // mapping back off the address space and the mapped-bytes gauge.
            unsafe {
                let _ = cu::cuMemUnmap(address, granularity);
            }
            pool.note_unmapped();
            return Err(error);
        }
        pool.note_shared_map(handle);
        reservation.blocks.push((offset, granularity, handle));
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
        }
    }

    pub fn with_teardown_synchronizer(mut self, synchronizer: TeardownSynchronizer) -> Self {
        self.teardown_synchronizer = Some(synchronizer);
        self
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
}

#[derive(Debug)]
struct PoolState {
    available: Vec<cu::CUmemGenericAllocationHandle>,
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
    device_ordinal: i32,
    granularity: usize,
    authority: MemoryAuthorityId,
    max_retained_bytes: usize,
    authority_gate: Arc<RwLock<()>>,
    state: Mutex<PoolState>,
    lease_checkout: Mutex<()>,
    counters: Arc<PoolCounters>,
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
        let context_id =
            current_context_id(&context).map_err(|reason| MemoryError::AllocationFailed {
                tier: Tier::Device.name(),
                requested: 0,
                reason,
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
            context,
            device_ordinal,
            granularity,
            authority,
            max_retained_bytes: retained_granules * granularity,
            authority_gate,
            state: Mutex::new(PoolState {
                available: Vec::new(),
                lease: Some(lease),
                pending_lease_shrink: 0,
                shared: HashMap::new(),
            }),
            lease_checkout: Mutex::new(()),
            counters: Arc::new(PoolCounters::default()),
        });
        registry.insert(key, Arc::downgrade(&pool));
        Ok(pool)
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
            self.lock().available.push(checkout.handle);
            self.counters
                .pooled_unmapped_bytes
                .fetch_add(self.granularity as u64, Ordering::AcqRel);
        }
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

    fn return_after_unmap(
        &self,
        handle: cu::CUmemGenericAllocationHandle,
        was_mapped: bool,
    ) -> Result<(), VirtualMemoryError> {
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
                    return Ok(());
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
            return Ok(());
        }

        if self
            .bind("releasing excess pooled CUDA physical memory")
            .is_err()
        {
            self.lock().available.push(handle);
            self.counters
                .pooled_unmapped_bytes
                .fetch_add(self.granularity as u64, Ordering::AcqRel);
            return Ok(());
        }
        let result = unsafe { cu::cuMemRelease(handle) };
        if CudaVirtualBacking::check("cuMemRelease", result).is_err() {
            self.lock().available.push(handle);
            self.counters
                .pooled_unmapped_bytes
                .fetch_add(self.granularity as u64, Ordering::AcqRel);
            return Ok(());
        }
        self.counters.releases.fetch_add(1, Ordering::Relaxed);
        self.counters
            .total_owned_bytes
            .fetch_sub(self.granularity as u64, Ordering::AcqRel);
        self.shrink_lease_or_defer(self.granularity as u64);
        Ok(())
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
                pool.lock().available.push(handle);
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
        let _operation = self
            .authority_gate
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (handles, mut lease) = {
            let state = self
                .state
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (std::mem::take(&mut state.available), state.lease.take())
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
            }
        }
        let ownership_remains = self.counters.total_owned_bytes.load(Ordering::Acquire) > 0;
        if release_failed || ownership_remains {
            // A failed driver release means physical ownership is uncertain.
            // A non-zero owned gauge can also mean a reservation could not
            // unmap during its Drop. Leaking the remaining lease is
            // conservative; dropping it would advertise memory as free while
            // the driver may still own it.
            if let Some(lease) = lease {
                std::mem::forget(lease);
            }
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

fn current_context_id(context: &CudaContext) -> Result<usize, String> {
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
    /// `(offset, len, handle)` for every mapped block.
    blocks: Vec<(usize, usize, cu::CUmemGenericAllocationHandle)>,
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
            .finish()
    }
}

// The reservation is an owned device address range; nothing in it is
// thread-affine, and every driver call through the backing binds the context
// first.
unsafe impl Send for CudaReservation {}
unsafe impl Sync for CudaReservation {}

impl Drop for CudaReservation {
    fn drop(&mut self) {
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
        let _ = self.context.bind_to_thread();
        for (offset, len, handle) in std::mem::take(&mut self.blocks) {
            // SAFETY: each block was mapped by `commit` and is unmapped once.
            if unsafe { cu::cuMemUnmap(self.base + offset as u64, len) }
                == cu::CUresult::CUDA_SUCCESS
            {
                if let Some(pool) = &self.pool {
                    let _ = pool.return_after_unmap(handle, true);
                } else {
                    unsafe {
                        let _ = cu::cuMemRelease(handle);
                    }
                }
            }
        }
        if self.len > 0 {
            // SAFETY: `base` came from `cuMemAddressReserve` with this length
            // and every block in it has been unmapped above.
            unsafe {
                let _ = cu::cuMemAddressFree(self.base, self.len);
            }
        }
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
            blocks: Vec::new(),
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
            self.commit_with_owned_limit(reservation, offset, len, u64::MAX)?;
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

        reservation.blocks.push((offset, len, handle));
        Ok(())
    }

    fn release(
        &self,
        reservation: &mut Self::Reservation,
        offset: usize,
        requested_len: usize,
    ) -> Result<(), VirtualMemoryError> {
        self.bind("releasing CUDA memory")?;
        let end = offset.saturating_add(requested_len);
        let mut blocks = reservation
            .blocks
            .iter()
            .copied()
            .filter(|&(at, len, _)| at >= offset && at + len <= end)
            .collect::<Vec<_>>();
        blocks.sort_unstable_by_key(|&(at, _, _)| at);
        let covers_full_range = blocks.first().is_some_and(|&(at, _, _)| at == offset)
            && blocks.last().is_some_and(|&(at, len, _)| at + len == end)
            && blocks
                .windows(2)
                .all(|pair| pair[0].0 + pair[0].1 == pair[1].0);
        if covers_full_range {
            // CUDA permits one cuMemUnmap over adjacent mappings even when
            // those mappings came from distinct physical handles. Weight
            // pages commonly span several 2 MiB handles, so unmapping the run
            // once avoids a driver round-trip per granule.
            Self::check("cuMemUnmap", unsafe {
                cu::cuMemUnmap(reservation.base + offset as u64, requested_len)
            })?;
            reservation
                .blocks
                .retain(|&(at, len, _)| !(at >= offset && at + len <= end));
            for (_, _, handle) in blocks {
                if let Some(pool) = &self.pool {
                    pool.return_after_unmap(handle, true)?;
                } else {
                    Self::check("cuMemRelease", unsafe { cu::cuMemRelease(handle) })?;
                }
            }
            return Ok(());
        }
        loop {
            let Some(index) = reservation
                .blocks
                .iter()
                .position(|&(at, len, _)| at >= offset && at + len <= end)
            else {
                return Ok(());
            };
            let (at, len, handle) = reservation.blocks[index];
            let address = reservation.base + at as u64;
            // SAFETY: this block was mapped by `commit` and remains recorded
            // until unmapping succeeds.
            Self::check("cuMemUnmap", unsafe { cu::cuMemUnmap(address, len) })?;
            reservation.blocks.remove(index);
            if let Some(pool) = &self.pool {
                pool.return_after_unmap(handle, true)?;
            } else {
                // Unmapping removes the mapping; the physical memory needs
                // releasing separately.
                Self::check("cuMemRelease", unsafe { cu::cuMemRelease(handle) })?;
            }
        }
    }
}

fn rollback_pooled_maps(
    reservation: &mut CudaReservation,
    pool: &PhysicalHandlePool,
    mapped: &mut Vec<(usize, usize, CheckedOutHandle)>,
) {
    for (offset, len, checkout) in mapped.drain(..).rev() {
        if unsafe { cu::cuMemUnmap(reservation.base + offset as u64, len) }
            == cu::CUresult::CUDA_SUCCESS
        {
            pool.rollback_checkout(checkout, true);
        } else {
            reservation.blocks.push((offset, len, checkout.handle));
        }
    }
}

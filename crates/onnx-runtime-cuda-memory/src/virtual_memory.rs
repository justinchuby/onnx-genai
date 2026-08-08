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
use std::sync::{Arc, Mutex, OnceLock, Weak};

use cudarc::driver::sys as cu;
use onnx_runtime_memory_governor::{
    HolderId, MemoryAuthorityId, MemoryError, MemoryGovernor, MemoryLease, MemoryRole, Tier,
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
            state: Mutex::new(PoolState {
                available: Vec::new(),
                lease: Some(lease),
                pending_lease_shrink: 0,
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

    fn acquire(&self) -> Result<cu::CUmemGenericAllocationHandle, VirtualMemoryError> {
        if let Some(handle) = {
            let mut state = self.lock();
            state.available.pop()
        } {
            self.counters
                .pooled_unmapped_bytes
                .fetch_sub(self.granularity as u64, Ordering::AcqRel);
            self.counters.pool_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(handle);
        }

        let _checkout = self
            .lease_checkout
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut lease = {
            let mut state = self.lock();
            state.lease.take().ok_or_else(|| VirtualMemoryError::Os {
                operation: "growing physical handle pool lease",
                reason: String::from("the physical handle pool is tearing down"),
                code: 0,
            })?
        };
        let growth = lease
            .grow(self.granularity as u64)
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

        if let Err(error) = self.bind("creating pooled CUDA physical memory") {
            self.shrink_lease_or_defer(self.granularity as u64);
            return Err(error);
        }
        let prop = allocation_prop(self.device_ordinal);
        let mut handle: cu::CUmemGenericAllocationHandle = 0;
        let result = unsafe { cu::cuMemCreate(&mut handle, self.granularity, &prop, 0) };
        if let Err(error) = CudaVirtualBacking::check("cuMemCreate", result) {
            self.shrink_lease_or_defer(self.granularity as u64);
            return Err(error);
        }
        self.counters.creates.fetch_add(1, Ordering::Relaxed);
        self.counters
            .total_owned_bytes
            .fetch_add(self.granularity as u64, Ordering::AcqRel);
        Ok(handle)
    }

    fn shrink_lease_or_defer(&self, bytes: u64) {
        let mut state = self.lock();
        if let Some(lease) = state.lease.as_mut() {
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

    fn return_after_unmap(
        &self,
        handle: cu::CUmemGenericAllocationHandle,
        was_mapped: bool,
    ) -> Result<(), VirtualMemoryError> {
        if was_mapped {
            self.counters
                .mapped_bytes
                .fetch_sub(self.granularity as u64, Ordering::AcqRel);
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

impl Drop for PhysicalHandlePool {
    fn drop(&mut self) {
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
        if let Some(pool) = &self.pool {
            let granularity = pool.granularity;
            let mut mapped = Vec::new();
            for granule_offset in (offset..offset + len).step_by(granularity) {
                let handle = match pool.acquire() {
                    Ok(handle) => handle,
                    Err(error) => {
                        rollback_pooled_maps(reservation, pool, &mut mapped);
                        return Err(error);
                    }
                };
                let address = reservation.base + granule_offset as u64;
                if let Err(error) = Self::check("cuMemMap", unsafe {
                    cu::cuMemMap(address, granularity, 0, handle, 0)
                }) {
                    let _ = pool.return_after_unmap(handle, false);
                    rollback_pooled_maps(reservation, pool, &mut mapped);
                    return Err(error);
                }
                pool.note_mapped();
                mapped.push((granule_offset, granularity, handle));

                let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
                access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = self.device_ordinal;
                access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                if let Err(error) = Self::check("cuMemSetAccess", unsafe {
                    cu::cuMemSetAccess(address, granularity, &access, 1)
                }) {
                    rollback_pooled_maps(reservation, pool, &mut mapped);
                    return Err(error);
                }
            }
            reservation.blocks.extend(mapped);
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
    mapped: &mut Vec<(usize, usize, cu::CUmemGenericAllocationHandle)>,
) {
    for (offset, len, handle) in mapped.drain(..).rev() {
        if unsafe { cu::cuMemUnmap(reservation.base + offset as u64, len) }
            == cu::CUresult::CUDA_SUCCESS
        {
            let _ = pool.return_after_unmap(handle, true);
        } else {
            reservation.blocks.push((offset, len, handle));
        }
    }
}

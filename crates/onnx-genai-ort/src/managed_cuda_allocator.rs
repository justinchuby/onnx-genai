use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, sys as cu};
use onnx_runtime_cuda_memory::vmm_allocator::CudaVmmAllocator;
use onnx_runtime_ep_cuda::deferred_release::{
    CudaDeferredReleaseQueue, ReleaseFence, ReleaseFenceSource, ReleaseObserver,
};
use onnx_runtime_memory_governor::{
    AllocationChargeMode, AllocationPublication, AllocationReleaseOutcome, AllocationRequest,
    AllocationSettlementToken, AllocationStepError, AllocationTransactionError, BindingError,
    DeviceAllocator, DeviceKey, DeviceLossListener, HolderId, ManagedAllocation, MemoryGovernor,
    MemoryRole, ProcessMemoryManager, RegisteredMemoryAuthority, RegisteredMemoryContext,
    RegisteredMemoryHolder, RegisteredMemoryMechanism, ScopedMemoryBinding, Tier,
};

use crate::allocator::MemoryInfo;
use crate::env::Environment;
use crate::error::{self, OrtError, Result};
use crate::governed_allocator::AllocationRoles;

const CUDA_ALLOCATION_ALIGNMENT: usize = 256;
const CUDA_VMM_ALLOCATOR_HOLDER: HolderId = HolderId::new(93);
const RESERVATION_VRAM_MULTIPLE: usize = 16;
const RESERVATION_FLOOR_BYTES: usize = 1 << 40;
const RESERVATION_MIN_BYTES: usize = 64 << 30;

type SharedGovernor = Arc<dyn MemoryGovernor + Send + Sync>;

#[derive(Clone)]
pub struct ManagedCudaAllocatorConfig {
    device_id: i32,
    process_memory_manager: ProcessMemoryManager,
    governor: SharedGovernor,
}

impl ManagedCudaAllocatorConfig {
    pub fn new(
        device_id: i32,
        process_memory_manager: ProcessMemoryManager,
        governor: SharedGovernor,
    ) -> Result<Self> {
        let device = u32::try_from(device_id).map_err(|_| {
            OrtError::InvalidArgument(format!(
                "managed CUDA allocator requires a non-negative device id, got {device_id}"
            ))
        })?;
        let expected = DeviceKey::device(device);
        let actual = governor.authority_id().device();
        if actual != expected {
            return Err(OrtError::InvalidArgument(format!(
                "managed CUDA allocator authority is registered for {actual:?}, not {expected:?}"
            )));
        }
        Ok(Self {
            device_id,
            process_memory_manager,
            governor,
        })
    }

    pub(crate) fn device_id(&self) -> i32 {
        self.device_id
    }

    pub(crate) fn authority_id(&self) -> onnx_runtime_memory_governor::MemoryAuthorityId {
        self.governor.authority_id()
    }

    pub(crate) fn manager(&self) -> &ProcessMemoryManager {
        &self.process_memory_manager
    }

    pub(crate) fn governor(&self) -> &SharedGovernor {
        &self.governor
    }
}

impl fmt::Debug for ManagedCudaAllocatorConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedCudaAllocatorConfig")
            .field("device_id", &self.device_id)
            .field("authority_id", &self.governor.authority_id())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ManagedCudaAllocatorStats {
    pub live_allocations: usize,
    pub live_bytes: u64,
    pub total_allocations: u64,
    pub reserve_allocations: u64,
    pub deferred_release_pending: usize,
    pub deferred_release_accepted: u64,
    pub deferred_release_completed: u64,
    pub deferred_release_quarantined: u64,
    pub deferred_release_retained: usize,
    pub deferred_release_enqueue_failures: u64,
    pub device_lost: bool,
}

#[derive(Debug)]
struct OrtCudaProviderContextPin {
    _context: Arc<CudaContext>,
    _queue: Arc<CudaDeferredReleaseQueue>,
}

#[derive(Debug)]
struct OrtCudaAuthorityPin {
    _device_id: u32,
}

#[derive(Debug)]
struct OrtCudaDeviceLossForwarder {
    queue: Arc<CudaDeferredReleaseQueue>,
    device_lost: Arc<AtomicBool>,
}

impl DeviceLossListener for OrtCudaDeviceLossForwarder {
    fn mark_device_lost(&self, reason: &str) {
        self.device_lost.store(true, Ordering::Release);
        self.queue.mark_device_lost(reason.to_owned());
    }
}

#[derive(Debug)]
struct OrtCudaFailClosedFences;

impl ReleaseFenceSource for OrtCudaFailClosedFences {
    fn record(&self) -> std::result::Result<Vec<Box<dyn ReleaseFence>>, String> {
        Err(String::from(
            "ORT does not provide a lifetime token for allocator callback CUDA streams; \
             retaining the allocation instead of recording on a possibly destroyed stream",
        ))
    }
}

#[derive(Debug)]
struct ManagedCudaOrtReleaseObserver {
    settlement: AllocationSettlementToken,
}

impl ReleaseObserver for ManagedCudaOrtReleaseObserver {
    fn released(&self, outcome: &AllocationReleaseOutcome) {
        // SAFETY: this observer is attached to the exact prepared release
        // paired with `settlement`.
        unsafe { self.settlement.settle(outcome) };
    }
}

#[derive(Debug)]
struct LiveAllocation {
    allocation: ManagedAllocation,
    requires_stream_ordering: bool,
}

struct ManagedCudaOrtAllocatorState {
    binding: ScopedMemoryBinding,
    authority: RegisteredMemoryAuthority,
    holder: RegisteredMemoryHolder,
    context: Arc<CudaContext>,
    memory: Arc<dyn DeviceAllocator>,
    queue: Arc<CudaDeferredReleaseQueue>,
    roles: AllocationRoles,
    live_allocations: Mutex<HashMap<usize, LiveAllocation>>,
    live_bytes: AtomicU64,
    live_count: AtomicUsize,
    total_count: AtomicU64,
    reserve_count: AtomicU64,
}

impl fmt::Debug for ManagedCudaOrtAllocatorState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedCudaOrtAllocatorState")
            .field("live_bytes", &self.live_bytes.load(Ordering::Relaxed))
            .field("live_count", &self.live_count.load(Ordering::Relaxed))
            .field("total_count", &self.total_count.load(Ordering::Relaxed))
            .field("reserve_count", &self.reserve_count.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

#[repr(C)]
struct ManagedCudaOrtAllocator {
    base: onnx_genai_ort_sys::OrtAllocator,
    memory_info: MemoryInfo,
    state: Arc<ManagedCudaOrtAllocatorState>,
}

impl ManagedCudaOrtAllocator {
    fn new(
        memory_info: MemoryInfo,
        binding: ScopedMemoryBinding,
        authority: RegisteredMemoryAuthority,
        holder: RegisteredMemoryHolder,
        context: Arc<CudaContext>,
        memory: Arc<dyn DeviceAllocator>,
        queue: Arc<CudaDeferredReleaseQueue>,
    ) -> Box<Self> {
        Box::new(Self {
            base: onnx_genai_ort_sys::OrtAllocator {
                version: onnx_genai_ort_sys::ORT_API_VERSION,
                Alloc: Some(managed_cuda_alloc),
                Free: Some(managed_cuda_free),
                Info: Some(managed_cuda_info),
                Reserve: Some(managed_cuda_reserve),
                GetStats: None,
                AllocOnStream: Some(managed_cuda_alloc_on_stream),
                Shrink: None,
            },
            memory_info,
            state: Arc::new(ManagedCudaOrtAllocatorState {
                binding,
                authority,
                holder,
                context,
                memory,
                queue,
                roles: AllocationRoles::split(),
                live_allocations: Mutex::new(HashMap::new()),
                live_bytes: AtomicU64::new(0),
                live_count: AtomicUsize::new(0),
                total_count: AtomicU64::new(0),
                reserve_count: AtomicU64::new(0),
            }),
        })
    }

    fn as_ort_allocator(&mut self) -> *mut onnx_genai_ort_sys::OrtAllocator {
        &mut self.base
    }

    fn stats(&self) -> ManagedCudaAllocatorStats {
        let queue = self.state.queue.stats();
        ManagedCudaAllocatorStats {
            live_allocations: self.state.live_count.load(Ordering::Relaxed),
            live_bytes: self.state.live_bytes.load(Ordering::Relaxed),
            total_allocations: self.state.total_count.load(Ordering::Relaxed),
            reserve_allocations: self.state.reserve_count.load(Ordering::Relaxed),
            deferred_release_pending: queue.pending,
            deferred_release_accepted: queue.accepted,
            deferred_release_completed: queue.completed,
            deferred_release_quarantined: queue.quarantined,
            deferred_release_retained: queue.retained,
            deferred_release_enqueue_failures: queue.enqueue_failures,
            device_lost: queue.device_lost,
        }
    }
}

/// Process-lifetime owner for an allocator ORT may retain in sessions/OrtValues.
///
/// ORT installs the raw allocator pointer with a no-op deleter and exposes no
/// last-user notification, so successful registrations are intentionally leaked.
pub(crate) struct ManagedCudaEnvironmentRegistration {
    device_id: i32,
    authority_id: onnx_runtime_memory_governor::MemoryAuthorityId,
    registered_allocator: Box<ManagedCudaOrtAllocator>,
    _loss_listener: Arc<dyn DeviceLossListener>,
    manager: ProcessMemoryManager,
}

/// Cleans every manager record if construction fails before ORT registration.
struct RegistrationRollback {
    manager: ProcessMemoryManager,
    context: Option<RegisteredMemoryContext>,
    authority: Option<RegisteredMemoryAuthority>,
    mechanism: Option<RegisteredMemoryMechanism>,
    holder: Option<RegisteredMemoryHolder>,
    binding: Option<ScopedMemoryBinding>,
    armed: bool,
}

impl RegistrationRollback {
    fn new(manager: ProcessMemoryManager) -> Self {
        Self {
            manager,
            context: None,
            authority: None,
            mechanism: None,
            holder: None,
            binding: None,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for RegistrationRollback {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        drop(self.binding.take());
        if let Some(mechanism) = self.mechanism.take() {
            let _ = self.manager.retire(&mechanism);
            let _ = self.manager.remove_mechanism(&mechanism);
        }
        if let Some(holder) = self.holder.take() {
            let _ = self.manager.unregister_holder(&holder);
        }
        if let Some(authority) = self.authority.take() {
            let _ = self.manager.remove_authority(&authority);
        }
        if let Some(context) = self.context.take() {
            let _ = self.manager.remove_provider_context(&context);
        }
    }
}

impl ManagedCudaEnvironmentRegistration {
    pub(crate) fn new(
        environment: &Environment,
        config: &ManagedCudaAllocatorConfig,
    ) -> Result<Self> {
        let device_id = config.device_id();
        let device = DeviceKey::device(u32::try_from(device_id).map_err(|_| {
            OrtError::InvalidArgument(format!(
                "managed CUDA allocator requires a non-negative device id, got {device_id}"
            ))
        })?);
        let context =
            Arc::new(CudaContext::new(device_id as usize).map_err(|error| {
                OrtError::Cuda(format!("CudaContext::new({device_id}): {error}"))
            })?);
        let queue = CudaDeferredReleaseQueue::new(
            Box::new(OrtCudaFailClosedFences),
            onnx_runtime_ep_cuda::deferred_release::DEFAULT_DEFERRED_RELEASE_CAPACITY,
        );
        let manager = config.manager().clone();
        let authority_id = config.authority_id();
        let device_lost = Arc::new(AtomicBool::new(false));
        let loss_listener: Arc<dyn DeviceLossListener> = Arc::new(OrtCudaDeviceLossForwarder {
            queue: Arc::clone(&queue),
            device_lost: Arc::clone(&device_lost),
        });
        let registration_generation = manager
            .register_device_loss_listener(device, &loss_listener)
            .map_err(|error| manager_error("register CUDA device-loss listener", error))?;
        let mut rollback = RegistrationRollback::new(manager.clone());
        let governed_capacity = config
            .governor()
            .used(Tier::Device)
            .checked_add(config.governor().available(Tier::Device))
            .ok_or_else(|| {
                OrtError::InvalidArgument(String::from(
                    "managed CUDA allocator device authority capacity overflowed u64",
                ))
            })?;
        let context_pin = Arc::new(OrtCudaProviderContextPin {
            _context: Arc::clone(&context),
            _queue: Arc::clone(&queue),
        });
        let registered_context = match manager.register_provider_context(
            device,
            format!("ort-cuda:{device_id} environment context"),
            context_pin,
        ) {
            Ok(context) => context,
            Err(error) => {
                return Err(binding_error(
                    "register the managed CUDA environment context",
                    error,
                ));
            }
        };
        rollback.context = Some(registered_context.clone());
        let authority_resource = Arc::new(OrtCudaAuthorityPin {
            _device_id: device_id as u32,
        });
        let registered_authority = match manager.register_authority(
            device,
            format!("ort-cuda:{device_id} governed authority"),
            authority_resource,
            Arc::clone(config.governor()),
        ) {
            Ok(authority) => authority,
            Err(error) => {
                return Err(manager_error(
                    "register the managed CUDA accounting authority",
                    error,
                ));
            }
        };
        rollback.authority = Some(registered_authority.clone());
        if manager.process_limit(Tier::Device) != u64::MAX
            && !registered_authority.has_process_delegation(Tier::Device)
            && let Err(error) = manager.delegate_authority_capacity(
                &registered_authority,
                Tier::Device,
                governed_capacity,
            )
        {
            return Err(manager_error(
                "delegate process device capacity to the managed CUDA authority",
                error,
            ));
        }

        let reservation_queue: Arc<
            dyn onnx_runtime_cuda_memory::virtual_memory::DeferredReservationQueue,
        > = Arc::clone(&queue)
            as Arc<dyn onnx_runtime_cuda_memory::virtual_memory::DeferredReservationQueue>;
        let allocator = build_vmm_allocator(
            device_id as u32,
            &context,
            config.governor().as_ref(),
            reservation_queue,
        )?;
        let tracked_allocator: Arc<dyn DeviceAllocator> = allocator;
        let mechanism = match manager.register_allocator(
            &registered_context,
            &registered_authority,
            format!("ort-cuda:{device_id} allocator mechanism"),
            Arc::clone(&tracked_allocator),
        ) {
            Ok(mechanism) => mechanism,
            Err(error) => {
                return Err(manager_error("register the managed CUDA allocator", error));
            }
        };
        rollback.mechanism = Some(mechanism.clone());
        if let Err(error) = manager.select(&mechanism) {
            return Err(manager_error("select the managed CUDA allocator", error));
        }
        let binding = match manager.bind_registered(&mechanism) {
            Ok(binding) => binding,
            Err(error) => {
                return Err(manager_error("bind the managed CUDA allocator", error));
            }
        };
        rollback.binding = Some(binding);
        let holder = match manager.register_holder(
            &registered_authority,
            format!("ort-cuda:{device_id} environment allocations"),
            None,
        ) {
            Ok(holder) => holder,
            Err(error) => {
                return Err(manager_error(
                    "register the managed CUDA allocation holder",
                    error,
                ));
            }
        };
        rollback.holder = Some(holder.clone());
        if let Err(error) = manager.finish_device_registration(device, registration_generation) {
            return Err(manager_error(
                "finalize managed CUDA device registration",
                error,
            ));
        }

        let binding = rollback
            .binding
            .take()
            .expect("managed CUDA registration binding was recorded");
        let mut registered_allocator = ManagedCudaOrtAllocator::new(
            MemoryInfo::cuda(device_id)?,
            binding,
            registered_authority.clone(),
            holder.clone(),
            Arc::clone(&context),
            tracked_allocator,
            Arc::clone(&queue),
        );
        let register = error::api()?
            .RegisterAllocator
            .ok_or(OrtError::ApiUnavailable("RegisterAllocator"))?;
        crate::error::check_status(unsafe {
            register(
                environment.as_ptr().cast_mut(),
                registered_allocator.as_ort_allocator(),
            )
        })?;
        rollback.disarm();

        Ok(Self {
            device_id,
            authority_id,
            registered_allocator,
            _loss_listener: loss_listener,
            manager,
        })
    }

    pub(crate) fn matches(&self, config: &ManagedCudaAllocatorConfig) -> bool {
        self.device_id == config.device_id()
            && self.authority_id == config.authority_id()
            && self.manager.is_same_instance(config.manager())
    }

    pub(crate) fn stats(&self) -> ManagedCudaAllocatorStats {
        self.registered_allocator.stats()
    }
}

impl fmt::Debug for ManagedCudaEnvironmentRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedCudaEnvironmentRegistration")
            .field("device_id", &self.device_id)
            .field("authority_id", &self.authority_id)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

pub(crate) fn register_managed_cuda_allocator(
    environment: &Environment,
    config: &ManagedCudaAllocatorConfig,
) -> Result<&'static ManagedCudaEnvironmentRegistration> {
    let registration = ManagedCudaEnvironmentRegistration::new(environment, config)?;
    // ORT copies environment allocators into sessions and OrtValues behind a
    // no-op deleter, and exposes no last-user notification. Keep the allocator,
    // its manager registrations, CUDA context, queue, and listener alive for
    // process lifetime so no callback can ever target freed Rust storage.
    Ok(Box::leak(Box::new(registration)))
}

unsafe fn managed_cuda_allocator_from_base<'a>(
    this: *const onnx_genai_ort_sys::OrtAllocator,
) -> &'a ManagedCudaOrtAllocator {
    unsafe { &*this.cast::<ManagedCudaOrtAllocator>() }
}

unsafe extern "C" fn managed_cuda_alloc(
    this: *mut onnx_genai_ort_sys::OrtAllocator,
    size: usize,
) -> *mut c_void {
    let allocator = unsafe { managed_cuda_allocator_from_base(this) };
    allocate_cuda_ort_memory(allocator, size, allocator.state.roles.run, true)
}

unsafe extern "C" fn managed_cuda_reserve(
    this: *mut onnx_genai_ort_sys::OrtAllocator,
    size: usize,
) -> *mut c_void {
    let allocator = unsafe { managed_cuda_allocator_from_base(this) };
    allocator
        .state
        .reserve_count
        .fetch_add(1, Ordering::Relaxed);
    allocate_cuda_ort_memory(allocator, size, allocator.state.roles.initialization, false)
}

unsafe extern "C" fn managed_cuda_alloc_on_stream(
    this: *mut onnx_genai_ort_sys::OrtAllocator,
    size: usize,
    stream: *mut onnx_genai_ort_sys::OrtSyncStream,
) -> *mut c_void {
    let allocator = unsafe { managed_cuda_allocator_from_base(this) };
    let Some(get_handle) = error::api().ok().and_then(|api| api.SyncStream_GetHandle) else {
        return std::ptr::null_mut();
    };
    let native = unsafe { get_handle(stream) };
    if native.is_null() {
        return std::ptr::null_mut();
    }
    allocate_cuda_ort_memory(allocator, size, allocator.state.roles.run, true)
}

unsafe extern "C" fn managed_cuda_info(
    this: *const onnx_genai_ort_sys::OrtAllocator,
) -> *const onnx_genai_ort_sys::OrtMemoryInfo {
    let allocator = unsafe { managed_cuda_allocator_from_base(this) };
    allocator.memory_info.as_ptr()
}

unsafe extern "C" fn managed_cuda_free(
    this: *mut onnx_genai_ort_sys::OrtAllocator,
    p: *mut c_void,
) {
    if p.is_null() {
        return;
    }
    let allocator = unsafe { managed_cuda_allocator_from_base(this) };
    let state = &allocator.state;
    let address = p as usize;
    let Some(live) = state
        .live_allocations
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&address)
    else {
        eprintln!(
            "onnx-genai-ort: WARNING: managed CUDA allocator received an unknown free for {address:#x}"
        );
        return;
    };
    let bytes = live.allocation.len() as u64;
    state.live_bytes.fetch_sub(bytes, Ordering::Relaxed);
    state.live_count.fetch_sub(1, Ordering::Relaxed);

    if !live.requires_stream_ordering {
        if let Err(error) = live.allocation.release_now() {
            let (error, _allocation) = error.into_parts();
            eprintln!(
                "onnx-genai-ort: WARNING: managed CUDA allocator could not release a session-init \
                 allocation immediately: {error}"
            );
        }
        return;
    }
    // ORT's registered device allocator is the backing allocator beneath its
    // arena. `Free` has no stream argument, while `AllocOnStream` only lends an
    // OrtSyncStream for the duration of the allocation callback. The bridge
    // forces `use_ep_level_unified_stream=1`, but retaining that raw stream
    // handle past the callback would still be a lifetime bug. Match CUDA's
    // ordinary `cudaFree` safety contract instead: synchronize the durable,
    // process-lifetime context before unmapping VMM pages. CUDA graph capture
    // keeps arena allocations live through capture/replay, so these backing
    // frees occur only after ORT has ended the capture window.
    if let Err(error) = state
        .context
        .bind_to_thread()
        .and_then(|()| state.context.synchronize())
    {
        eprintln!(
            "onnx-genai-ort: WARNING: managed CUDA allocator could not establish device \
             quiescence before release: {error}; retaining the allocation"
        );
        quarantine_live_allocation(state, live.allocation);
        return;
    }
    if let Err(error) = live.allocation.release_now() {
        let (error, _allocation) = error.into_parts();
        eprintln!(
            "onnx-genai-ort: WARNING: managed CUDA allocator could not release a quiescent \
             allocation: {error}"
        );
    }
}

fn quarantine_live_allocation(state: &ManagedCudaOrtAllocatorState, allocation: ManagedAllocation) {
    let prepared = match allocation.prepare_release() {
        Ok(prepared) => prepared,
        Err(error) => {
            let (error, _allocation) = error.into_parts();
            eprintln!(
                "onnx-genai-ort: WARNING: managed CUDA allocator could not prepare a deferred \
                 release: {error}"
            );
            return;
        }
    };
    // SAFETY: the observer keeps the settlement token paired with the exact
    // prepared request through every terminal path below.
    let (request, settlement) = unsafe { prepared.into_parts() };
    let observer: Arc<dyn ReleaseObserver> = Arc::new(ManagedCudaOrtReleaseObserver {
        settlement: settlement.clone(),
    });
    if let Err(error) = state.queue.enqueue_prepared(request, Some(observer)) {
        let rejection = error.rejection();
        let outcome = error.quarantine();
        // SAFETY: this outcome came from the exact refused prepared request.
        unsafe { settlement.settle(&outcome) };
        eprintln!(
            "onnx-genai-ort: WARNING: managed CUDA allocator deferred release enqueue failed \
             ({rejection:?}); the allocation was quarantined"
        );
    }
}

fn allocate_cuda_ort_memory(
    allocator: &ManagedCudaOrtAllocator,
    size: usize,
    role: MemoryRole,
    requires_stream_ordering: bool,
) -> *mut c_void {
    let state = &allocator.state;
    if size == 0 {
        return std::ptr::null_mut();
    }
    let allocation = match allocate_transaction(state, size, role) {
        Ok(allocation) => allocation,
        Err(_) => return std::ptr::null_mut(),
    };
    let ptr = allocation.as_ptr();
    state
        .live_allocations
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(
            ptr.as_ptr() as usize,
            LiveAllocation {
                allocation,
                requires_stream_ordering,
            },
        );
    state.live_bytes.fetch_add(size as u64, Ordering::Relaxed);
    state.live_count.fetch_add(1, Ordering::Relaxed);
    state.total_count.fetch_add(1, Ordering::Relaxed);
    ptr.as_ptr().cast()
}

fn allocate_transaction(
    state: &ManagedCudaOrtAllocatorState,
    size: usize,
    role: MemoryRole,
) -> std::result::Result<ManagedAllocation, AllocationTransactionError> {
    let alignment = CUDA_ALLOCATION_ALIGNMENT;
    let full = 0..size;
    let virtual_backing = state.binding.virtual_backing()?;
    let reserve_bytes = match virtual_backing.as_ref() {
        Some(capability) => capability.mapped_bytes_for_allocation(size, alignment)?,
        None => size as u64,
    };
    let charge_mode = if state.memory.commits_on_demand() {
        AllocationChargeMode::AuthorityManaged
    } else {
        AllocationChargeMode::Managed
    };
    let delegated = state.authority.has_process_delegation(Tier::Device);
    state.binding.allocate_with(
        AllocationRequest {
            allocation_bytes: size,
            alignment,
            tier: Tier::Device,
            role,
            holder: state.holder.clone(),
            charge_mode,
            authority_reserve_bytes: reserve_bytes,
            process_reserve_bytes: if charge_mode == AllocationChargeMode::Managed && !delegated {
                reserve_bytes
            } else {
                0
            },
        },
        |context| match virtual_backing.as_ref() {
            Some(_) => context.allocate_committed(std::slice::from_ref(&full)),
            None => context.allocate_owning(),
        },
        |owner| {
            let physical = match virtual_backing.as_ref() {
                Some(capability) => capability
                    .allocation_committed_bytes(owner)
                    .map_err(|error| AllocationStepError::new(error.to_string()))?
                    as u64,
                None => size as u64,
            };
            Ok(match charge_mode {
                AllocationChargeMode::Managed => AllocationPublication {
                    charged_bytes: physical,
                    process_reserved_bytes: if delegated { 0 } else { physical },
                    physical_bytes: Some(physical),
                    mapped_bytes: Some(physical),
                    unattributed_bytes: 0,
                    shared_physical: None,
                },
                AllocationChargeMode::AuthorityManaged => AllocationPublication {
                    charged_bytes: 0,
                    process_reserved_bytes: 0,
                    physical_bytes: None,
                    mapped_bytes: Some(physical),
                    unattributed_bytes: 0,
                    shared_physical: None,
                },
                AllocationChargeMode::Compatibility => {
                    AllocationPublication::compatibility(physical, physical)
                }
            })
        },
    )
}

fn build_vmm_allocator(
    ordinal: u32,
    context: &Arc<CudaContext>,
    governor: &dyn MemoryGovernor,
    reservation_queue: Arc<dyn onnx_runtime_cuda_memory::virtual_memory::DeferredReservationQueue>,
) -> Result<Arc<CudaVmmAllocator>> {
    let mut last_error = None;
    for reservation_bytes in reservation_ladder(ordinal) {
        match CudaVmmAllocator::new_with_reservation_queue(
            Arc::clone(context),
            DeviceKey::device(ordinal),
            ordinal as i32,
            reservation_bytes,
            governor,
            CUDA_VMM_ALLOCATOR_HOLDER,
            MemoryRole::Workspace { step_scoped: false },
            Arc::clone(&reservation_queue),
            None,
        ) {
            Ok(allocator) => return Ok(Arc::new(allocator)),
            Err(error) => last_error = Some((reservation_bytes, error)),
        }
    }
    let detail = last_error
        .map(|(bytes, error)| format!("last attempt reserved {bytes} byte(s): {error}"))
        .unwrap_or_else(|| String::from("no reservation size was attempted"));
    Err(OrtError::Cuda(format!(
        "managed CUDA allocator could not create a VMM arena for device {ordinal}: {detail}"
    )))
}

fn reservation_ladder(ordinal: u32) -> Vec<usize> {
    reservation_ladder_from_total(device_total_memory_bytes(ordinal))
}

fn reservation_ladder_from_total(device_total: Option<usize>) -> Vec<usize> {
    let desired = device_total
        .and_then(|total| total.checked_mul(RESERVATION_VRAM_MULTIPLE))
        .unwrap_or(RESERVATION_FLOOR_BYTES)
        .max(RESERVATION_FLOOR_BYTES);
    let mut ladder = Vec::new();
    let mut size = desired;
    while size > RESERVATION_MIN_BYTES {
        ladder.push(size);
        size /= 2;
    }
    ladder.push(RESERVATION_MIN_BYTES);
    ladder
}

fn device_total_memory_bytes(ordinal: u32) -> Option<usize> {
    let mut device = 0;
    unsafe {
        if cu::cuDeviceGet(&mut device, ordinal as i32) != cu::CUresult::CUDA_SUCCESS {
            return None;
        }
        let mut bytes = 0usize;
        if cu::cuDeviceTotalMem_v2(&mut bytes, device) != cu::CUresult::CUDA_SUCCESS {
            return None;
        }
        (bytes > 0).then_some(bytes)
    }
}

fn binding_error(operation: &str, error: BindingError) -> OrtError {
    OrtError::SessionCreation(format!("could not {operation}: {error}"))
}

fn manager_error(operation: &str, error: AllocationTransactionError) -> OrtError {
    OrtError::SessionCreation(format!("could not {operation}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_memory_governor::HostAllocator;

    #[derive(Debug)]
    struct Pin;

    #[test]
    fn stream_fence_source_fails_closed_without_touching_a_raw_stream() {
        let source = OrtCudaFailClosedFences;
        let error = source
            .record()
            .expect_err("raw ORT streams have no lifetime token");
        assert!(error.contains("possibly destroyed stream"));
    }

    #[test]
    fn registration_rollback_removes_every_manager_record() {
        let manager = ProcessMemoryManager::new().expect("manager");
        let context = manager
            .register_provider_context(DeviceKey::HOST, "test context", Arc::new(Pin))
            .expect("context");
        let authority = manager
            .register_compatibility_authority(DeviceKey::HOST, "test authority", Arc::new(Pin))
            .expect("authority");
        let allocator: Arc<dyn DeviceAllocator> = Arc::new(HostAllocator);
        let mechanism = manager
            .register_allocator(&context, &authority, "test allocator", allocator)
            .expect("mechanism");
        manager.select(&mechanism).expect("select");
        let binding = manager.bind_registered(&mechanism).expect("binding");
        let holder = manager
            .register_holder(&authority, "test holder", None)
            .expect("holder");

        drop(RegistrationRollback {
            manager: manager.clone(),
            context: Some(context),
            authority: Some(authority),
            mechanism: Some(mechanism),
            holder: Some(holder),
            binding: Some(binding),
            armed: true,
        });

        let snapshot = manager.snapshot().expect("snapshot");
        assert_eq!(snapshot.authority_count, 0);
        assert!(snapshot.mechanism_snapshots.is_empty());
    }
}

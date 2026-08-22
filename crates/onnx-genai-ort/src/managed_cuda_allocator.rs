use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, sys as cu};
use onnx_runtime_cuda_memory::virtual_memory::{
    confirm_physical_handle_pool_context_terminated, physical_pool_context_identity,
};
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

#[derive(Debug, Default)]
struct ObservedOrtCudaStreams {
    raw: Mutex<Vec<usize>>,
}

impl ObservedOrtCudaStreams {
    fn record(&self, raw: *mut c_void) -> std::result::Result<(), String> {
        if raw.is_null() {
            return Err(String::from(
                "ORT reported a null CUDA stream handle for a stream-aware allocation",
            ));
        }
        let raw_value = raw as usize;
        let mut streams = self
            .raw
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !streams.contains(&raw_value) {
            streams.push(raw_value);
        }
        Ok(())
    }

    fn current(&self) -> Vec<cu::CUstream> {
        self.raw
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .copied()
            .map(|raw| (raw as *mut std::ffi::c_void).cast())
            .collect()
    }
}

#[derive(Debug)]
struct OrtCudaUnifiedStreamFences {
    _context: Arc<CudaContext>,
    device_id: i32,
    stream: Arc<ObservedOrtCudaStreams>,
}

impl ReleaseFenceSource for OrtCudaUnifiedStreamFences {
    fn record(&self) -> std::result::Result<Vec<Box<dyn ReleaseFence>>, String> {
        let streams = self.stream.current();
        if streams.is_empty() {
            return Err(String::from(
                "managed CUDA allocator bridge cannot defer a free before ORT exposes a CUDA stream",
            ));
        }
        let _guard = crate::cuda_rt::DeviceGuard::set(self.device_id).map_err(|error| {
            format!(
                "could not make CUDA device {} current: {error}",
                self.device_id
            )
        })?;
        let mut fences: Vec<Box<dyn ReleaseFence>> = Vec::with_capacity(streams.len());
        for stream in streams {
            let mut event = std::ptr::null_mut();
            check_cuda("cuEventCreate", unsafe {
                cu::cuEventCreate(
                    &mut event,
                    cu::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                )
            })?;
            if let Err(error) =
                check_cuda("cuEventRecord", unsafe { cu::cuEventRecord(event, stream) })
            {
                unsafe {
                    let _ = cu::cuEventDestroy_v2(event);
                }
                return Err(error);
            }
            fences.push(Box::new(OrtCudaEventFence {
                _context: Arc::clone(&self._context),
                device_id: self.device_id,
                event,
            }));
        }
        Ok(fences)
    }
}

#[derive(Debug)]
struct OrtCudaEventFence {
    _context: Arc<CudaContext>,
    device_id: i32,
    event: cu::CUevent,
}

impl ReleaseFence for OrtCudaEventFence {
    fn is_complete(&self) -> bool {
        if crate::cuda_rt::DeviceGuard::set(self.device_id).is_err() {
            return false;
        }
        matches!(
            unsafe { cu::cuEventQuery(self.event) },
            cu::CUresult::CUDA_SUCCESS
        )
    }
}

// SAFETY: the raw event handle is only queried or destroyed through CUDA driver
// calls after making the owning device current on the calling thread.
unsafe impl Send for OrtCudaEventFence {}
// SAFETY: shared references only perform non-blocking event queries; final
// destruction happens once from Drop.
unsafe impl Sync for OrtCudaEventFence {}

impl Drop for OrtCudaEventFence {
    fn drop(&mut self) {
        if self.event.is_null() {
            return;
        }
        if crate::cuda_rt::DeviceGuard::set(self.device_id).is_ok() {
            unsafe {
                let _ = cu::cuEventDestroy_v2(self.event);
            }
        }
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
struct TeardownTrackedAllocator {
    inner: Arc<CudaVmmAllocator>,
    completion: Arc<AtomicBool>,
}

impl Drop for TeardownTrackedAllocator {
    fn drop(&mut self) {
        self.completion.store(true, Ordering::Release);
    }
}

impl DeviceAllocator for TeardownTrackedAllocator {
    fn allocate(
        &self,
        bytes: usize,
        align: usize,
    ) -> std::result::Result<NonNull<u8>, onnx_runtime_memory_governor::MemoryError> {
        self.inner.allocate(bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        unsafe { self.inner.deallocate(ptr, bytes, align) };
    }

    unsafe fn deallocate_with_unmapped(&self, ptr: NonNull<u8>, bytes: usize, align: usize) -> u64 {
        unsafe { self.inner.deallocate_with_unmapped(ptr, bytes, align) }
    }

    unsafe fn release(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> AllocationReleaseOutcome {
        unsafe { self.inner.release(ptr, bytes, align) }
    }

    fn device(&self) -> DeviceKey {
        self.inner.device()
    }

    fn commits_on_demand(&self) -> bool {
        self.inner.commits_on_demand()
    }

    fn as_virtual_backing(&self) -> Option<&dyn onnx_runtime_memory_governor::VirtualBacking> {
        self.inner.as_virtual_backing()
    }

    fn as_shared_mapping(&self) -> Option<&dyn onnx_runtime_memory_governor::SharedMapping> {
        self.inner.as_shared_mapping()
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
    memory: Arc<dyn DeviceAllocator>,
    queue: Arc<CudaDeferredReleaseQueue>,
    roles: AllocationRoles,
    stream: Arc<ObservedOrtCudaStreams>,
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
        memory: Arc<dyn DeviceAllocator>,
        queue: Arc<CudaDeferredReleaseQueue>,
        stream: Arc<ObservedOrtCudaStreams>,
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
                memory,
                queue,
                roles: AllocationRoles::split(),
                stream,
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

pub(crate) struct ManagedCudaEnvironmentRegistration {
    env: NonNull<onnx_genai_ort_sys::OrtEnv>,
    device_id: i32,
    authority_id: onnx_runtime_memory_governor::MemoryAuthorityId,
    memory_info: MemoryInfo,
    registered_allocator: Box<ManagedCudaOrtAllocator>,
    queue: Arc<CudaDeferredReleaseQueue>,
    device_lost: Arc<AtomicBool>,
    allocator_teardown_complete: Arc<AtomicBool>,
    context: RegisteredMemoryContext,
    authority: RegisteredMemoryAuthority,
    mechanism: RegisteredMemoryMechanism,
    holder: RegisteredMemoryHolder,
    manager: ProcessMemoryManager,
    cuda_context_identity: usize,
    cleanup_armed: AtomicBool,
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
        let stream = Arc::new(ObservedOrtCudaStreams::default());
        let queue = CudaDeferredReleaseQueue::new(
            Box::new(OrtCudaUnifiedStreamFences {
                _context: Arc::clone(&context),
                device_id,
                stream: Arc::clone(&stream),
            }),
            onnx_runtime_ep_cuda::deferred_release::DEFAULT_DEFERRED_RELEASE_CAPACITY,
        );
        let manager = config.manager().clone();
        let authority_id = config.authority_id();
        let cuda_context_identity = physical_pool_context_identity(context.as_ref()).map_err(|error| {
            OrtError::Cuda(format!(
                "managed CUDA allocator could not identify CUDA context for device {device_id}: \
                 {error}"
            ))
        })?;
        let device_lost = Arc::new(AtomicBool::new(false));
        let loss_listener: Arc<dyn DeviceLossListener> = Arc::new(OrtCudaDeviceLossForwarder {
            queue: Arc::clone(&queue),
            device_lost: Arc::clone(&device_lost),
        });
        let registration_generation = manager
            .register_device_loss_listener(device, &loss_listener)
            .map_err(|error| manager_error("register CUDA device-loss listener", error))?;
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
                let _ = manager.remove_provider_context(&registered_context);
                return Err(manager_error(
                    "register the managed CUDA accounting authority",
                    error,
                ));
            }
        };
        if manager.process_limit(Tier::Device) != u64::MAX
            && !registered_authority.has_process_delegation(Tier::Device)
            && let Err(error) = manager.delegate_authority_capacity(
                &registered_authority,
                Tier::Device,
                governed_capacity,
            )
        {
            let _ = manager.remove_authority(&registered_authority);
            let _ = manager.remove_provider_context(&registered_context);
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
        let allocator_teardown_complete = Arc::new(AtomicBool::new(false));
        let tracked_allocator: Arc<dyn DeviceAllocator> = Arc::new(TeardownTrackedAllocator {
            inner: allocator,
            completion: Arc::clone(&allocator_teardown_complete),
        });
        let mechanism = match manager.register_allocator(
            &registered_context,
            &registered_authority,
            format!("ort-cuda:{device_id} allocator mechanism"),
            Arc::clone(&tracked_allocator),
        ) {
            Ok(mechanism) => mechanism,
            Err(error) => {
                let _ = manager.remove_authority(&registered_authority);
                let _ = manager.remove_provider_context(&registered_context);
                return Err(manager_error("register the managed CUDA allocator", error));
            }
        };
        if let Err(error) = manager.select(&mechanism) {
            let _ = manager.retire(&mechanism);
            let _ = manager.remove_mechanism(&mechanism);
            let _ = manager.remove_authority(&registered_authority);
            let _ = manager.remove_provider_context(&registered_context);
            return Err(manager_error("select the managed CUDA allocator", error));
        }
        let binding = match manager.bind_registered(&mechanism) {
            Ok(binding) => binding,
            Err(error) => {
                let _ = manager.retire(&mechanism);
                let _ = manager.remove_mechanism(&mechanism);
                let _ = manager.remove_authority(&registered_authority);
                let _ = manager.remove_provider_context(&registered_context);
                return Err(manager_error("bind the managed CUDA allocator", error));
            }
        };
        let holder = match manager.register_holder(
            &registered_authority,
            format!("ort-cuda:{device_id} environment allocations"),
            None,
        ) {
            Ok(holder) => holder,
            Err(error) => {
                drop(binding);
                let _ = manager.retire(&mechanism);
                let _ = manager.remove_mechanism(&mechanism);
                let _ = manager.remove_authority(&registered_authority);
                let _ = manager.remove_provider_context(&registered_context);
                return Err(manager_error(
                    "register the managed CUDA allocation holder",
                    error,
                ));
            }
        };
        if let Err(error) = manager.finish_device_registration(device, registration_generation) {
            let _ = manager.retire(&mechanism);
            let _ = manager.remove_mechanism(&mechanism);
            let _ = manager.unregister_holder(&holder);
            let _ = manager.remove_authority(&registered_authority);
            let _ = manager.remove_provider_context(&registered_context);
            return Err(manager_error(
                "finalize managed CUDA device registration",
                error,
            ));
        }

        let memory_info = MemoryInfo::cuda(device_id)?;
        let mut registered_allocator = ManagedCudaOrtAllocator::new(
            MemoryInfo::cuda(device_id)?,
            binding,
            registered_authority.clone(),
            holder.clone(),
            tracked_allocator,
            Arc::clone(&queue),
            stream,
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

        Ok(Self {
            env: NonNull::new(environment.as_ptr().cast_mut()).ok_or(OrtError::NullPointer)?,
            device_id,
            authority_id,
            memory_info,
            registered_allocator,
            queue,
            device_lost,
            allocator_teardown_complete,
            context: registered_context,
            authority: registered_authority,
            mechanism,
            holder,
            manager,
            cuda_context_identity,
            cleanup_armed: AtomicBool::new(false),
        })
    }

    pub(crate) fn matches(&self, config: &ManagedCudaAllocatorConfig) -> bool {
        self.device_id == config.device_id() && self.authority_id == config.authority_id()
    }

    pub(crate) fn stats(&self) -> ManagedCudaAllocatorStats {
        self.registered_allocator.stats()
    }

    fn arm_cleanup(&self) {
        if self.cleanup_armed.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Err(error) = self.manager.retire_context(&self.context) {
            warn_cleanup(
                "could not retire the managed CUDA context during teardown",
                &error,
            );
        }
        if let Err(error) = self.manager.retire(&self.mechanism) {
            warn_cleanup(
                "could not retire the managed CUDA allocator mechanism during teardown",
                &error,
            );
        }
        let manager = self.manager.downgrade();
        let mechanism = self.mechanism.clone();
        let holder = self.holder.clone();
        let context = self.context.clone();
        let authority = self.authority.clone();
        let authority_id = authority.memory_authority_id();
        let allocator_teardown_complete = Arc::clone(&self.allocator_teardown_complete);
        let device_lost = Arc::clone(&self.device_lost);
        let cuda_context_identity = self.cuda_context_identity;
        self.queue.set_drain_callback(move || {
            let Some(manager) = manager.upgrade() else {
                return true;
            };
            if device_lost.load(Ordering::Acquire) {
                if !allocator_teardown_complete.load(Ordering::Acquire) {
                    return false;
                }
                if let Err(error) = manager.confirm_context_terminated(&context) {
                    warn_cleanup(
                        "could not confirm managed CUDA context termination after device loss",
                        &error,
                    );
                    return true;
                }
                if let Some(authority_id) = authority_id {
                    confirm_physical_handle_pool_context_terminated(
                        cuda_context_identity,
                        authority_id,
                    );
                }
                if let Err(error) = manager.remove_mechanism(&mechanism)
                    && !matches!(
                        error,
                        AllocationTransactionError::Binding(BindingError::UnregisteredMechanism(_))
                    )
                {
                    warn_cleanup(
                        "could not remove a terminated managed CUDA mechanism",
                        &error,
                    );
                }
                if let Err(error) = manager.unregister_holder(&holder) {
                    warn_cleanup(
                        "could not unregister the managed CUDA holder after device loss",
                        &error,
                    );
                }
                if let Err(error) = manager.remove_provider_context(&context) {
                    warn_cleanup(
                        "could not remove the managed CUDA context pin after device loss",
                        &error,
                    );
                    return true;
                }
                if let Err(error) = manager.remove_authority(&authority)
                    && !matches!(
                        error,
                        AllocationTransactionError::Binding(BindingError::AuthorityInUse(_))
                    )
                {
                    warn_cleanup(
                        "could not remove the managed CUDA authority after device loss",
                        &error,
                    );
                }
                return true;
            }

            match manager.remove_mechanism(&mechanism) {
                Ok(())
                | Err(AllocationTransactionError::Binding(BindingError::UnregisteredMechanism(
                    _,
                ))) => {}
                Err(AllocationTransactionError::Binding(BindingError::InactiveMechanism {
                    ..
                })) => {
                    return false;
                }
                Err(error) => {
                    warn_cleanup(
                        "could not remove the managed CUDA allocator mechanism after queue drain",
                        &error,
                    );
                    return true;
                }
            }
            if !allocator_teardown_complete.load(Ordering::Acquire) {
                return false;
            }
            if let Err(error) = manager.unregister_holder(&holder) {
                warn_cleanup(
                    "could not unregister the managed CUDA holder after queue drain",
                    &error,
                );
            }
            if let Err(error) = manager.remove_provider_context(&context) {
                warn_cleanup(
                    "could not remove the managed CUDA context pin after queue drain",
                    &error,
                );
                return true;
            }
            if let Err(error) = manager.remove_authority(&authority)
                && !matches!(
                    error,
                    AllocationTransactionError::Binding(BindingError::AuthorityInUse(_))
                )
            {
                warn_cleanup(
                    "could not remove the managed CUDA authority after queue drain",
                    &error,
                );
            }
            true
        });
        self.queue.close_after_drain();
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

impl Drop for ManagedCudaEnvironmentRegistration {
    fn drop(&mut self) {
        if let Ok(api) = error::api()
            && let Some(unregister) = api.UnregisterAllocator
            && let Err(error) = crate::error::check_status(unsafe {
                unregister(self.env.as_ptr(), self.memory_info.as_ptr())
            })
        {
            eprintln!(
                "onnx-genai-ort: WARNING: could not unregister managed CUDA allocator for device {}: \
                 {error}",
                self.device_id
            );
        }
        self.arm_cleanup();
    }
}

pub(crate) fn register_managed_cuda_allocator(
    environment: &Environment,
    config: &ManagedCudaAllocatorConfig,
) -> Result<ManagedCudaEnvironmentRegistration> {
    ManagedCudaEnvironmentRegistration::new(environment, config)
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
    if allocator.state.stream.record(native).is_err() {
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
    if state.stream.current().is_empty() {
        eprintln!(
            "onnx-genai-ort: WARNING: quarantining a managed CUDA allocation because ORT never \
             exposed any CUDA stream before free"
        );
        drop(live);
        return;
    };
    let prepared = match live.allocation.prepare_release() {
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

fn check_cuda(operation: &str, result: cu::CUresult) -> std::result::Result<(), String> {
    if result == cu::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation} failed with CUDA result {result:?}"))
    }
}

fn binding_error(operation: &str, error: BindingError) -> OrtError {
    OrtError::SessionCreation(format!("could not {operation}: {error}"))
}

fn manager_error(operation: &str, error: AllocationTransactionError) -> OrtError {
    OrtError::SessionCreation(format!("could not {operation}: {error}"))
}

fn warn_cleanup(operation: &str, error: &impl fmt::Display) {
    eprintln!("onnx-genai-ort: WARNING: {operation}: {error}");
}

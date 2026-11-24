//! Host vertical slice: registered factory -> provider-bound compiled kernels ->
//! CreateState -> repeated Compute -> ReleaseState, with real governed backing.
//! The ORT host is mocked; this does not validate CUDA ordering or capture.
use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::kernel::{
    OutputMetadata, TensorMetadata, WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ep_api::{EpError, Kernel, Result as EpResult, TensorMut, TensorView};
use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources, DerivedWeightMemory};
use onnx_runtime_ep_plugin::{
    ExportedEp,
    compute::{CompiledKernelEntry, ExportedComputeInfo, ShapeInference},
    factory,
};
use onnx_runtime_memory_api::{MemoryRole, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
};
use std::{
    ptr,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

thread_local! { static LAST_ERROR: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) }; }
unsafe extern "C" fn status(
    _: ort::OrtErrorCode,
    msg: *const std::ffi::c_char,
) -> *mut ort::OrtStatus {
    LAST_ERROR.with(|error| {
        *error.borrow_mut() = unsafe { std::ffi::CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    });
    ptr::dangling_mut()
}
unsafe extern "C" fn input_count(
    _: *const ort::OrtKernelContext,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    unsafe { *out = 0 };
    ptr::null_mut()
}
unsafe extern "C" fn input(
    _: *const ort::OrtKernelContext,
    _: usize,
    out: *mut *const ort::OrtValue,
) -> ort::OrtStatusPtr {
    unsafe { *out = ptr::null() };
    ptr::null_mut()
}
unsafe extern "C" fn shape(
    _: *const ort::OrtValue,
    _: *mut ort::ONNXTensorElementDataType,
    _: *mut *const i64,
    _: *mut usize,
) -> ort::OrtStatusPtr {
    unsafe {
        status(
            ort::ORT_FAIL,
            c"zero-input fixture unexpectedly requested shape".as_ptr(),
        )
    }
}
unsafe extern "C" fn data(
    _: *const ort::OrtValue,
    _: *mut *const std::ffi::c_void,
) -> ort::OrtStatusPtr {
    unsafe {
        status(
            ort::ORT_FAIL,
            c"zero-input fixture unexpectedly requested data".as_ptr(),
        )
    }
}
unsafe extern "C" fn api(_: u32) -> *const ort::OrtApi {
    static API: OnceLock<ort::OrtApi> = OnceLock::new();
    API.get_or_init(|| ort::OrtApi {
        CreateStatus: Some(status),
        KernelContext_GetInputCount: Some(input_count),
        KernelContext_GetInput: Some(input),
        GetTensorElementTypeAndShapeDataReference: Some(shape),
        GetTensorData: Some(data),
        ..Default::default()
    })
}

struct ScratchKernel {
    bytes: usize,
    seen: Arc<Mutex<Vec<usize>>>,
    fail: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    panic_next: Arc<AtomicBool>,
}
impl Kernel for ScratchKernel {
    fn workspace_requirement(
        &self,
        _: &[TensorMetadata<'_>],
        _: &[OutputMetadata<'_>],
    ) -> EpResult<WorkspaceRequirement> {
        Ok(WorkspaceRequirement {
            bytes: self.bytes as u64,
            alignment: 16,
            lifetime: WorkspaceLifetime::SessionPersistent,
        })
    }
    fn execute(&self, _: &[TensorView<'_>], _: &mut [TensorMut<'_>]) -> EpResult<()> {
        Err(EpError::KernelFailed(
            "fixture requires its caller workspace".into(),
        ))
    }
    fn execute_with_workspace(
        &self,
        _: &[TensorView<'_>],
        _: &mut [TensorMut<'_>],
        workspace: Option<WorkspaceView>,
    ) -> EpResult<()> {
        let workspace = workspace
            .ok_or_else(|| EpError::KernelFailed("fixture did not receive workspace".into()))?;
        assert!(workspace.bytes() >= self.bytes);
        let scratch = unsafe { std::slice::from_raw_parts_mut(workspace.ptr().0.cast::<u64>(), 8) };
        let generation = self.calls.fetch_add(1, Ordering::SeqCst) as u64;
        assert!(
            !self.panic_next.swap(false, Ordering::SeqCst),
            "planned shared kernel panic"
        );
        assert_eq!(
            self.active.fetch_add(1, Ordering::SeqCst),
            0,
            "Send-only shared kernel must not execute concurrently"
        );
        for _ in 0..8 {
            std::thread::yield_now();
        }
        for (i, slot) in scratch.iter_mut().enumerate() {
            *slot = generation + i as u64;
        }
        assert_eq!(scratch.iter().sum::<u64>(), 8 * generation + 28);
        self.seen.lock().unwrap().push(workspace.ptr().0 as usize);
        assert_eq!(self.active.fetch_sub(1, Ordering::SeqCst), 1);
        if self.fail.load(Ordering::SeqCst) {
            return Err(EpError::KernelFailed(
                "planned post-submission failure".into(),
            ));
        }
        Ok(())
    }
}
struct Fixture {
    factory: *mut ort::OrtEpFactory,
    facade: *mut ort::OrtEp,
    info: Box<ExportedComputeInfo>,
    seen: Arc<Mutex<Vec<usize>>>,
    fail: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    panic_next: Arc<AtomicBool>,
}
impl Fixture {
    fn new(
        resources: CpuMemoryResources,
        registered: bool,
        preflight: Option<factory::WorkspaceExecutionGuard>,
    ) -> Self {
        let base = ort::OrtApiBase {
            GetApi: Some(api),
            GetVersionString: None,
        };
        let mut factory = ptr::null_mut();
        let mut count = 0;
        assert!(
            unsafe {
                factory::create_ep_factories(&base, &mut factory, 1, &mut count, move || {
                    Box::new(CpuExecutionProvider::with_resources(resources.clone()))
                })
            }
            .is_null()
        );
        assert_eq!(count, 1);
        if registered {
            // SAFETY: CPU kernel fixture finishes every scratch access inline.
            assert!(
                unsafe { factory::register_ordered_workspace_execution(factory, preflight) }
                    .is_null()
            );
        }
        let mut facade = ptr::null_mut();
        assert!(
            unsafe {
                (*factory).CreateEp.unwrap()(
                    factory,
                    ptr::null(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    &mut facade,
                )
            }
            .is_null()
        );
        assert!(!facade.is_null());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let fail = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let panic_next = Arc::new(AtomicBool::new(false));
        let entry = CompiledKernelEntry {
            kernel: Box::new(ScratchKernel {
                bytes: 64,
                seen: seen.clone(),
                fail: fail.clone(),
                calls: calls.clone(),
                active: active.clone(),
                panic_next: panic_next.clone(),
            }),
            num_inputs: 0,
            num_outputs: 0,
            output_dtypes: vec![],
            absent_output_slots: Default::default(),
            shape_inference: ShapeInference::StaticOutputs(vec![]),
            input_slots: vec![],
        };
        let info =
            Box::new(unsafe { &*facade.cast::<ExportedEp>() }.wrap_compiled_kernels(vec![entry]));
        Self {
            factory,
            facade,
            info,
            seen,
            fail,
            calls,
            active,
            panic_next,
        }
    }
    fn state(&mut self) -> *mut std::ffi::c_void {
        let mut state = ptr::null_mut();
        assert!(
            unsafe {
                self.info.vtable.CreateState.unwrap()(
                    &mut self.info.vtable,
                    ptr::null_mut(),
                    &mut state,
                )
            }
            .is_null()
        );
        assert!(!state.is_null());
        state
    }
    fn compute(&mut self, state: *mut std::ffi::c_void) -> *mut ort::OrtStatus {
        unsafe {
            self.info.vtable.Compute.unwrap()(&mut self.info.vtable, state, ptr::dangling_mut())
        }
    }
    fn release(&mut self, state: *mut std::ffi::c_void) {
        unsafe { self.info.vtable.ReleaseState.unwrap()(&mut self.info.vtable, state) }
    }
    fn release_facade(&mut self) {
        if !self.facade.is_null() {
            unsafe { (*self.factory).ReleaseEp.unwrap()(self.factory, self.facade) };
            self.facade = ptr::null_mut();
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.release_facade();
        assert!(unsafe { factory::release_ep_factory(self.factory) }.is_null());
    }
}

#[test]
fn persistent_workspace_registered_factory_reuses_governed_backing_without_warm_issuer_calls() {
    let books = Arc::new(RecordingGovernor::new_host(256));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources = CpuMemoryResources::new(
        DerivedWeightMemory::governed(books.as_ref(), HolderId::new(13)).unwrap(),
        Default::default(),
    )
    .unwrap()
    .with_workspace_pool(pool.clone())
    .unwrap();
    // Positive controls catch even zero-byte calls, which lease Drop may skip.
    let before = books.claims();
    let zero = books
        .reserve(
            Tier::Host,
            0,
            MemoryRole::Workspace { step_scoped: false },
            HolderId::new(99),
        )
        .unwrap();
    drop(zero);
    assert_eq!(books.claims(), before + 1);
    let before = books.releases();
    books.probe_zero_release();
    assert_eq!(books.releases(), before + 1);
    let mut fixture = Fixture::new(resources, true, None);
    let state = fixture.state();
    fixture.release_facade();
    assert!(fixture.compute(state).is_null());
    let before = (books.claims(), books.releases());
    for _ in 0..32 {
        assert!(fixture.compute(state).is_null());
    }
    assert_eq!(
        (books.claims(), books.releases()),
        before,
        "fitting warm Compute must make zero issuer callbacks"
    );
    let seen = fixture.seen.lock().unwrap();
    assert_eq!(seen.len(), 33);
    assert!(seen.iter().all(|ptr| *ptr == seen[0]));
    drop(seen);
    assert_eq!(manager.snapshot().unwrap().allocations.len(), 1);
    fixture.release(state);
    assert_eq!(
        books.used(Tier::Host),
        64,
        "pool-retained physical backing stays charged"
    );
    drop(fixture);
    drop(pool);
    assert_eq!(manager.process_used(Tier::Host), 0);
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn persistent_workspace_compute_failure_stops_subsequent_submissions() {
    let mut fixture = Fixture::new(CpuMemoryResources::default(), true, None);
    let state = fixture.state();
    fixture.fail.store(true, Ordering::SeqCst);
    assert!(!fixture.compute(state).is_null());
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
    fixture.fail.store(false, Ordering::SeqCst);
    assert!(!fixture.compute(state).is_null());
    LAST_ERROR.with(|error| {
        assert!(
            error.borrow().contains("recreate the session"),
            "{}",
            error.borrow()
        )
    });
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
    fixture.release(state);
}

#[test]
fn persistent_workspace_unknown_ordering_is_rejected_before_kernel_or_allocation() {
    let mut fixture = Fixture::new(CpuMemoryResources::default(), false, None);
    let state = fixture.state();
    assert!(!fixture.compute(state).is_null());
    LAST_ERROR.with(|error| {
        assert!(
            error.borrow().contains("registered execution ordering"),
            "{}",
            error.borrow()
        )
    });
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 0);
    fixture.release(state);
}
/// Observe real issuer callbacks, including zero-byte requests, while
/// keeping the ledger's pressure/retry protocol intact.
#[derive(Clone)]
pub struct RecordingGovernor(std::sync::Arc<RecordingBooks>);
#[derive(Debug)]
struct RecordingBooks {
    governor: onnx_runtime_memory_governor::LedgerGovernor,
    claims: std::sync::atomic::AtomicUsize,
    releases: std::sync::atomic::AtomicUsize,
}
impl RecordingGovernor {
    pub fn new_host(host_bytes: u64) -> Self {
        Self(std::sync::Arc::new(RecordingBooks {
            governor: onnx_runtime_memory_governor::LedgerGovernor::new(
                onnx_runtime_memory_governor::LeaseLedger::new_for_device(
                    onnx_runtime_memory_api::DeviceKey::HOST,
                    0,
                    host_bytes,
                    0,
                ),
            ),
            claims: std::sync::atomic::AtomicUsize::new(0),
            releases: std::sync::atomic::AtomicUsize::new(0),
        }))
    }
    /// Probe the actual test issuer; MemoryLease deliberately skips a
    /// zero-byte drop, so that wrapper cannot falsify this counter itself.
    pub fn probe_zero_release(&self) {
        onnx_runtime_memory_governor::LeaseAccounting::release(
            self.0.as_ref(),
            onnx_runtime_memory_api::Tier::Host,
            0,
        );
    }
    pub fn claims(&self) -> usize {
        self.0.claims.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn releases(&self) -> usize {
        self.0.releases.load(std::sync::atomic::Ordering::Relaxed)
    }
}
impl onnx_runtime_memory_governor::LeaseAccounting for RecordingBooks {
    fn authority_id(&self) -> Option<onnx_runtime_memory_governor::MemoryAuthorityId> {
        Some(onnx_runtime_memory_governor::MemoryGovernor::authority_id(
            &self.governor,
            onnx_runtime_memory_api::Tier::Host,
        ))
    }
    fn try_claim(
        &self,
        tier: onnx_runtime_memory_api::Tier,
        bytes: u64,
        role: onnx_runtime_memory_api::MemoryRole,
    ) -> std::result::Result<(), onnx_runtime_memory_api::MemoryError> {
        self.claims
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.governor.ledger().try_claim(tier, bytes, role)
    }
    fn release(&self, tier: onnx_runtime_memory_api::Tier, bytes: u64) {
        self.releases
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        onnx_runtime_memory_governor::LeaseAccounting::release(
            self.governor.ledger().as_ref(),
            tier,
            bytes,
        );
    }
}
impl onnx_runtime_memory_governor::MemoryGovernor for RecordingGovernor {
    fn authority_id(
        &self,
        tier: onnx_runtime_memory_api::Tier,
    ) -> onnx_runtime_memory_governor::MemoryAuthorityId {
        self.0.governor.authority_id(tier)
    }
    fn reserve(
        &self,
        tier: onnx_runtime_memory_api::Tier,
        bytes: u64,
        role: onnx_runtime_memory_api::MemoryRole,
        holder: onnx_runtime_memory_governor::HolderId,
    ) -> std::result::Result<
        onnx_runtime_memory_governor::MemoryLease,
        onnx_runtime_memory_api::MemoryError,
    > {
        onnx_runtime_memory_governor::LeaseAccounting::try_claim(
            self.0.as_ref(),
            tier,
            bytes,
            role,
        )?;
        Ok(onnx_runtime_memory_governor::MemoryLease::new(
            tier,
            bytes,
            role,
            holder,
            self.0.clone(),
        ))
    }
    fn available(&self, tier: onnx_runtime_memory_api::Tier) -> u64 {
        self.0.governor.available(tier)
    }
    fn used(&self, tier: onnx_runtime_memory_api::Tier) -> u64 {
        self.0.governor.used(tier)
    }
    fn pressure_registry(
        &self,
        tier: onnx_runtime_memory_api::Tier,
    ) -> Option<onnx_runtime_memory_governor::PressureRegistry> {
        self.0.governor.pressure_registry(tier)
    }
}

#[test]
fn persistent_workspace_capture_preflight_rejects_before_submission_and_retires_state() {
    let checks = Arc::new(AtomicUsize::new(0));
    let seen = checks.clone();
    let preflight: factory::WorkspaceExecutionGuard = Arc::new(move || {
        seen.fetch_add(1, Ordering::SeqCst);
        Err("planned external capture without ownership pins; recreate the session".into())
    });
    let mut fixture = Fixture::new(CpuMemoryResources::default(), true, Some(preflight));
    let state = fixture.state();
    assert!(!fixture.compute(state).is_null());
    assert_eq!(checks.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 0);
    LAST_ERROR.with(|error| {
        assert!(
            error.borrow().contains("external capture"),
            "{}",
            error.borrow()
        )
    });
    assert!(!fixture.compute(state).is_null());
    assert_eq!(
        checks.load(Ordering::SeqCst),
        1,
        "retired state must not try submission again"
    );
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 0);
    fixture.release(state);
}

#[test]
fn persistent_workspace_serial_nodes_share_peak_capacity_and_one_preflight_per_compute() {
    let books = Arc::new(RecordingGovernor::new_host(512));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources = CpuMemoryResources::new(
        DerivedWeightMemory::governed(books.as_ref(), HolderId::new(13)).unwrap(),
        Default::default(),
    )
    .unwrap()
    .with_workspace_pool(pool.clone())
    .unwrap();
    let checks = Arc::new(AtomicUsize::new(0));
    let observed = checks.clone();
    let preflight: factory::WorkspaceExecutionGuard = Arc::new(move || {
        observed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let mut fixture = Fixture::new(resources, true, Some(preflight));
    let mut entries = std::mem::take(&mut fixture.info.entries);
    entries.push(CompiledKernelEntry {
        kernel: Box::new(ScratchKernel {
            bytes: 128,
            seen: fixture.seen.clone(),
            fail: fixture.fail.clone(),
            calls: fixture.calls.clone(),
            active: fixture.active.clone(),
            panic_next: fixture.panic_next.clone(),
        }),
        num_inputs: 0,
        num_outputs: 0,
        output_dtypes: vec![],
        absent_output_slots: Default::default(),
        shape_inference: ShapeInference::StaticOutputs(vec![]),
        input_slots: vec![],
    });
    fixture.info =
        Box::new(unsafe { &*fixture.facade.cast::<ExportedEp>() }.wrap_compiled_kernels(entries));
    fixture
        .info
        .set_routing(onnx_runtime_ep_plugin::compute::SubgraphRouting {
            input_sources: vec![vec![], vec![]],
            output_sinks: vec![vec![], vec![]],
            num_intermediate_buffers: 0,
        });
    let state = fixture.state();
    assert!(fixture.compute(state).is_null());
    let before = (books.claims(), books.releases());
    for _ in 0..32 {
        assert!(fixture.compute(state).is_null());
    }
    assert_eq!((books.claims(), books.releases()), before);
    assert_eq!(
        checks.load(Ordering::SeqCst),
        33,
        "preflight is per Compute, not per kernel"
    );
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 66);
    let seen = fixture.seen.lock().unwrap();
    assert_eq!(seen.len(), 66);
    assert!(
        seen[2..].iter().all(|address| *address == seen[2]),
        "both serial kernels must borrow one retained peak owner"
    );
    drop(seen);
    fixture.release(state);
    drop(fixture);
    drop(pool);
    assert_eq!(manager.process_used(Tier::Host), 0);
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn persistent_workspace_states_serialize_shared_send_only_kernel_and_keep_distinct_owners() {
    let mut fixture = Fixture::new(CpuMemoryResources::default(), true, None);
    let states = [fixture.state(), fixture.state()];
    let info_address = std::ptr::from_mut(&mut fixture.info.vtable) as usize;
    let compute = fixture.info.vtable.Compute.unwrap();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        for state in states {
            let state_address = state as usize;
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..32 {
                    let status = unsafe {
                        compute(
                            info_address as *mut ort::OrtNodeComputeInfo,
                            state_address as *mut std::ffi::c_void,
                            ptr::dangling_mut(),
                        )
                    };
                    assert!(status.is_null(), "concurrent state execution failed");
                }
            });
        }
    });
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 64);
    let seen = fixture.seen.lock().unwrap();
    assert_eq!(
        seen.iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        2,
        "two live states need independent owners"
    );
    drop(seen);
    for state in states {
        fixture.release(state);
    }
}

#[test]
fn persistent_workspace_kernel_panic_rejects_other_states_of_same_compiled_info() {
    let mut fixture = Fixture::new(CpuMemoryResources::default(), true, None);
    let first = fixture.state();
    let second = fixture.state();
    fixture.panic_next.store(true, Ordering::SeqCst);
    assert!(!fixture.compute(first).is_null());
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
    assert!(!fixture.compute(second).is_null());
    LAST_ERROR.with(|error| {
        assert!(
            error
                .borrow()
                .contains("shared compiled kernels were poisoned"),
            "{}",
            error.borrow()
        )
    });
    assert_eq!(
        fixture.calls.load(Ordering::SeqCst),
        1,
        "another state cannot reenter a panicked shared kernel"
    );
    fixture.release(first);
    fixture.release(second);
}

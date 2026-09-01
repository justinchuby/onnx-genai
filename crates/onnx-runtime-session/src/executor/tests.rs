#[cfg(feature = "gpu-tests")]
use std::alloc::{GlobalAlloc, Layout, System};
#[cfg(feature = "gpu-tests")]
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use onnx_runtime_ep_api::{
    CaptureSupport, Cost, EpConfig, EpError, ExecutionProviderCapabilities, Fence, Kernel,
    NegotiatedWeight,
};
use onnx_runtime_memory_governor::MemoryRole;

use super::*;

#[cfg(feature = "gpu-tests")]
struct CountingAllocator;

#[cfg(feature = "gpu-tests")]
thread_local! {
    static COUNT_HOST_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static HOST_ALLOCATION_COUNT: Cell<u64> = const { Cell::new(0) };
    static HOST_ALLOCATION_SIZES: Cell<[usize; 8]> = const { Cell::new([0; 8]) };
}

#[cfg(feature = "gpu-tests")]
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT_HOST_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                HOST_ALLOCATION_COUNT.with(|count| {
                    let index = count.get() as usize;
                    count.set(count.get() + 1);
                    HOST_ALLOCATION_SIZES.with(|sizes| {
                        let mut values = sizes.get();
                        if index < values.len() {
                            values[index] = layout.size();
                            sizes.set(values);
                        }
                    });
                });
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        COUNT_HOST_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                HOST_ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        COUNT_HOST_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                HOST_ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
#[cfg(feature = "gpu-tests")]
static TEST_ALLOCATOR: CountingAllocator = CountingAllocator;

#[cfg(feature = "gpu-tests")]
fn count_host_allocations<T>(operation: impl FnOnce() -> T) -> (T, u64) {
    HOST_ALLOCATION_COUNT.with(|count| count.set(0));
    HOST_ALLOCATION_SIZES.with(|sizes| sizes.set([0; 8]));
    COUNT_HOST_ALLOCATIONS.with(|enabled| enabled.set(true));
    let result = operation();
    COUNT_HOST_ALLOCATIONS.with(|enabled| enabled.set(false));
    (result, HOST_ALLOCATION_COUNT.with(Cell::get))
}

struct DeferredValidationKernel {
    fail_next: Arc<AtomicBool>,
    panic_next: Arc<AtomicBool>,
    validation_latch: Arc<AtomicU32>,
    executions: Arc<AtomicUsize>,
}

impl Kernel for DeferredValidationKernel {
    fn execute(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        if inputs.len() != 1
            || outputs.len() != 1
            || inputs[0].byte_size() != outputs[0].byte_size()
        {
            return Err(EpError::KernelFailed(
                "deferred validation test kernel received invalid I/O".into(),
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                inputs[0].data.0.cast::<u8>(),
                outputs[0].data.0.cast::<u8>(),
                inputs[0].byte_size(),
            );
        }
        self.executions.fetch_add(1, Ordering::Relaxed);
        if self.fail_next.swap(false, Ordering::Relaxed) {
            self.validation_latch.store(0x40, Ordering::Relaxed);
        }
        if self.panic_next.swap(false, Ordering::Relaxed) {
            panic!("forced deferred validation kernel panic");
        }
        Ok(())
    }
}

struct DeferredValidationEp {
    cpu: CpuExecutionProvider,
    fail_next: Arc<AtomicBool>,
    validation_latch: Arc<AtomicU32>,
    executions: Arc<AtomicUsize>,
    synchronized_executions: Arc<AtomicUsize>,
    resets: Arc<AtomicUsize>,
    reset_failure_at: Arc<AtomicUsize>,
    validation_state: Arc<std::sync::Mutex<DeferredValidationState>>,
    next_validation_generation: Arc<AtomicU64>,
    validation_consume_attempts: Arc<AtomicUsize>,
    sync_calls: Arc<AtomicUsize>,
    panic_next: Arc<AtomicBool>,
    graph_reset_failure: Arc<AtomicBool>,
    graph_reset_calls: Arc<AtomicUsize>,
    route_boundary_calls: Arc<AtomicUsize>,
    route_boundary_before_sync: Arc<AtomicBool>,
}

#[derive(Default)]
struct DeferredValidationState {
    active: Option<onnx_runtime_ep_api::DeviceValidationToken>,
    recipients: Vec<onnx_runtime_ep_api::DeviceValidationOwner>,
    owners: HashMap<onnx_runtime_ep_api::DeviceValidationOwner, Option<(u64, u32)>>,
}

impl DeferredValidationEp {
    fn new() -> Self {
        let mut cpu = CpuExecutionProvider::new();
        cpu.initialize(&EpConfig::default()).unwrap();
        Self {
            cpu,
            fail_next: Arc::new(AtomicBool::new(true)),
            validation_latch: Arc::new(AtomicU32::new(0)),
            executions: Arc::new(AtomicUsize::new(0)),
            synchronized_executions: Arc::new(AtomicUsize::new(0)),
            resets: Arc::new(AtomicUsize::new(0)),
            reset_failure_at: Arc::new(AtomicUsize::new(0)),
            validation_state: Arc::new(std::sync::Mutex::new(DeferredValidationState::default())),
            next_validation_generation: Arc::new(AtomicU64::new(1)),
            validation_consume_attempts: Arc::new(AtomicUsize::new(0)),
            sync_calls: Arc::new(AtomicUsize::new(0)),
            panic_next: Arc::new(AtomicBool::new(false)),
            graph_reset_failure: Arc::new(AtomicBool::new(false)),
            graph_reset_calls: Arc::new(AtomicUsize::new(0)),
            route_boundary_calls: Arc::new(AtomicUsize::new(0)),
            route_boundary_before_sync: Arc::new(AtomicBool::new(false)),
        }
    }

    fn consume_validation_token(
        &self,
        token: onnx_runtime_ep_api::DeviceValidationToken,
    ) -> onnx_runtime_ep_api::Result<u32> {
        self.validation_consume_attempts
            .fetch_add(1, Ordering::Relaxed);
        let mut state = self.validation_state.lock().unwrap();
        if let Some(Some((generation, flags))) = state.owners.get(&token.owner())
            && *generation == token.generation()
        {
            return Ok(*flags);
        }
        match state.active {
            Some(expected)
                if expected.generation() == token.generation()
                    && state.recipients.contains(&token.owner()) => {}
            Some(expected) => {
                return Err(EpError::KernelFailed(format!(
                    "validation token owner={} generation={} cannot consume active owner={} \
                     generation={}",
                    token.owner().get(),
                    token.generation(),
                    expected.owner().get(),
                    expected.generation()
                )));
            }
            None => {
                return Err(EpError::KernelFailed(format!(
                    "validation token owner={} generation={} is stale",
                    token.owner().get(),
                    token.generation()
                )));
            }
        }
        let result = self
            .check_validation_latch()
            .and_then(|flags| self.reset_validation_latch().map(|()| flags));
        if let Ok(flags) = &result {
            let generation = token.generation();
            for owner in state.recipients.clone() {
                state.owners.insert(owner, Some((generation, *flags)));
            }
            state.active = None;
            state.recipients.clear();
        }
        result
    }
}

impl ExecutionProvider for DeferredValidationEp {
    fn name(&self) -> &str {
        "deferred_validation_test_ep"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn device_id(&self) -> onnx_runtime_ir::DeviceId {
        onnx_runtime_ir::DeviceId::cpu()
    }

    fn initialize(&mut self, _config: &EpConfig) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn supports_op(
        &self,
        op: &Node,
        _opset: u64,
        _shapes: &[Shape],
        _input_dtypes: &[DataType],
        _layouts: &[TensorLayout],
    ) -> KernelMatch {
        if op.op_type == "DeferredValidation" {
            KernelMatch::Supported {
                cost: Cost::ZERO,
                required_input_layouts: None,
                output_layouts: vec![TensorLayout::contiguous()],
            }
        } else {
            KernelMatch::unsupported("test EP only supports DeferredValidation")
        }
    }

    fn get_kernel(
        &self,
        _op: &Node,
        _shapes: &[Vec<usize>],
        _opset: u64,
    ) -> onnx_runtime_ep_api::Result<Box<dyn Kernel>> {
        Ok(Box::new(DeferredValidationKernel {
            fail_next: Arc::clone(&self.fail_next),
            panic_next: Arc::clone(&self.panic_next),
            validation_latch: Arc::clone(&self.validation_latch),
            executions: Arc::clone(&self.executions),
        }))
    }

    fn allocate(&self, size: usize, alignment: usize) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
        self.cpu.allocate(size, alignment)
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.deallocate(buffer)
    }

    fn copy(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.copy(src, dst, size)
    }

    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<Fence> {
        self.cpu.copy_async(src, dst, size)
    }

    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> onnx_runtime_ep_api::Result<()> {
        self.sync()?;
        self.cpu.copy_to_host(src, dst)
    }

    fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
        self.sync_calls.fetch_add(1, Ordering::Relaxed);
        self.synchronized_executions
            .store(self.executions.load(Ordering::Relaxed), Ordering::Relaxed);
        Ok(())
    }

    fn register_device_validation_owner(
        &self,
    ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::DeviceValidationRegistration> {
        let owner = onnx_runtime_ep_api::DeviceValidationOwner::new();
        let mut state = self.validation_state.lock().unwrap();
        if state.owners.insert(owner, None).is_some() {
            return Err(EpError::KernelFailed(format!(
                "validation owner {} already registered",
                owner.get()
            )));
        }
        Ok(onnx_runtime_ep_api::DeviceValidationRegistration::new(
            owner,
            (),
        ))
    }

    fn unregister_device_validation_owner(
        &self,
        registration: &mut onnx_runtime_ep_api::DeviceValidationRegistration,
    ) -> onnx_runtime_ep_api::Result<()> {
        let owner = registration.owner();
        let mut state = self.validation_state.lock().unwrap();
        if state.active.is_some() && state.recipients.contains(&owner) {
            return Err(EpError::KernelFailed(format!(
                "validation owner {} still pending",
                owner.get()
            )));
        }
        state.owners.remove(&owner);
        Ok(())
    }

    fn begin_device_validation(
        &self,
        registration: &onnx_runtime_ep_api::DeviceValidationRegistration,
    ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::DeviceValidationToken> {
        let owner = registration.owner();
        let mut state = self.validation_state.lock().unwrap();
        if let Some(token) = state.active {
            return Err(EpError::KernelFailed(format!(
                "previous deferred device validation is still pending (owner={} generation={})",
                token.owner().get(),
                token.generation()
            )));
        }
        if !state.owners.contains_key(&owner) {
            return Err(EpError::KernelFailed(format!(
                "validation owner {} is unregistered",
                owner.get()
            )));
        }
        self.reset_validation_latch()?;
        let generation = self
            .next_validation_generation
            .fetch_add(1, Ordering::Relaxed);
        let token = onnx_runtime_ep_api::DeviceValidationToken::new(owner, generation);
        state.active = Some(token);
        state.recipients.clear();
        state.recipients.push(owner);
        state.owners.insert(owner, None);
        Ok(token)
    }

    fn add_device_validation_recipient(
        &self,
        submission: onnx_runtime_ep_api::DeviceValidationToken,
        recipient: &onnx_runtime_ep_api::DeviceValidationRegistration,
    ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::DeviceValidationToken> {
        let recipient = recipient.owner();
        let mut state = self.validation_state.lock().unwrap();
        if state.active != Some(submission) || !state.owners.contains_key(&recipient) {
            return Err(EpError::KernelFailed(
                "validation recipient does not belong to the active submission".into(),
            ));
        }
        state.recipients.push(recipient);
        state.owners.insert(recipient, None);
        Ok(onnx_runtime_ep_api::DeviceValidationToken::new(
            recipient,
            submission.generation(),
        ))
    }

    fn defers_device_validation(&self) -> bool {
        true
    }

    fn abort_device_validation_submission(
        &self,
        token: onnx_runtime_ep_api::DeviceValidationToken,
    ) -> onnx_runtime_ep_api::Result<u32> {
        self.consume_validation_token(token)
    }

    fn consume_device_validation_error(
        &self,
        registration: &onnx_runtime_ep_api::DeviceValidationRegistration,
        token: onnx_runtime_ep_api::DeviceValidationToken,
    ) -> onnx_runtime_ep_api::Result<u32> {
        if registration.owner() != token.owner() {
            return Err(EpError::KernelFailed(format!(
                "validation token owner={} is foreign to registration owner={}",
                token.owner().get(),
                registration.owner().get()
            )));
        }
        self.consume_validation_token(token)
    }

    fn reset_owned_device_graph(
        &self,
        _token: onnx_runtime_ep_api::DeviceGraphToken,
    ) -> onnx_runtime_ep_api::Result<bool> {
        self.graph_reset_calls.fetch_add(1, Ordering::Relaxed);
        if self.graph_reset_failure.load(Ordering::Relaxed) {
            return Err(EpError::KernelFailed(
                "forced device graph reset failure".into(),
            ));
        }
        Ok(true)
    }

    fn consume_route_residency_at_boundary(&self) -> onnx_runtime_ep_api::Result<()> {
        // The production Slice-7C boundary caller lands here once per top-level
        // request, after `sync()`. Record the call and prove the request-level
        // synchronization boundary was crossed first (mirrors the latch check
        // above): if the sync counter has not caught up to executions, the
        // boundary fired too early.
        if self.synchronized_executions.load(Ordering::Relaxed)
            != self.executions.load(Ordering::Relaxed)
        {
            self.route_boundary_before_sync
                .store(true, Ordering::Relaxed);
        }
        self.route_boundary_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl DeferredValidationEp {
    fn reset_validation_latch(&self) -> onnx_runtime_ep_api::Result<()> {
        let call = self.resets.fetch_add(1, Ordering::Relaxed) + 1;
        if self.reset_failure_at.load(Ordering::Relaxed) == call {
            return Err(EpError::KernelFailed(
                "forced device validation reset failure".into(),
            ));
        }
        self.validation_latch.store(0, Ordering::Relaxed);
        Ok(())
    }

    fn check_validation_latch(&self) -> onnx_runtime_ep_api::Result<u32> {
        if self.synchronized_executions.load(Ordering::Relaxed)
            != self.executions.load(Ordering::Relaxed)
        {
            return Err(EpError::KernelFailed(
                "device validation latch checked before synchronization".into(),
            ));
        }
        Ok(self.validation_latch.load(Ordering::Relaxed))
    }
}

fn deferred_validation_bound_fixture_for_provider(
    ep: Arc<DeferredValidationEp>,
) -> (Executor, Vec<DeviceIoBinding>) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
    graph.add_input(input);
    let output = graph.create_named_value("output", DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "DeferredValidation",
        vec![Some(input)],
        vec![output],
    ));
    graph.add_output(output);

    let executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        Arc::clone(&ep) as Arc<dyn ExecutionProvider>,
    )
    .unwrap();
    let mut bindings = vec![
        executor
            .allocate_device_binding("input".into(), None, DataType::Float32, vec![2], vec![2])
            .unwrap(),
        executor
            .allocate_device_output_binding("output".into(), DataType::Float32, vec![2], vec![2])
            .unwrap(),
    ];
    bindings[0]
        .write_bytes(
            0,
            &[3.0f32, 7.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    (executor, bindings)
}

fn deferred_validation_bound_fixture() -> (Executor, Arc<DeferredValidationEp>, Vec<DeviceIoBinding>)
{
    let ep = Arc::new(DeferredValidationEp::new());
    let (executor, bindings) = deferred_validation_bound_fixture_for_provider(Arc::clone(&ep));
    (executor, ep, bindings)
}

#[test]
fn deferred_device_validation_is_request_local_and_checked_after_sync() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
    graph.add_input(input);
    let output = graph.create_named_value("output", DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "DeferredValidation",
        vec![Some(input)],
        vec![output],
    ));
    graph.add_output(output);

    let ep = Arc::new(DeferredValidationEp::new());
    let resets = Arc::clone(&ep.resets);
    let executions = Arc::clone(&ep.executions);
    let synchronized = Arc::clone(&ep.synchronized_executions);
    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        ep as Arc<dyn ExecutionProvider>,
    )
    .unwrap();
    let input = Tensor::from_f32(&[2], &[3.0, 7.0]).unwrap();

    let first = executor.run(&[("input", &input)]).unwrap_err();
    assert!(
        first
            .to_string()
            .contains("device validation failed (flags=0x40)"),
        "the run that set the deferred latch must fail after synchronization: {first}"
    );

    let second = executor
        .run(&[("input", &input)])
        .expect("the prior run's validation latch must not poison a healthy request");
    assert_eq!(second[0].to_vec_f32(), vec![3.0, 7.0]);
    assert_eq!(executions.load(Ordering::Relaxed), 2);
    assert_eq!(
        synchronized.load(Ordering::Relaxed),
        executions.load(Ordering::Relaxed),
        "validation must be checked only after the request synchronization boundary"
    );
    assert_eq!(
        resets.load(Ordering::Relaxed),
        4,
        "each request must reset the latch before execution and after checking it"
    );
}

#[test]
fn unconsumed_bound_failure_rejects_later_success_or_failure_without_erasure() {
    for second_would_fail in [false, true] {
        let (mut executor, ep, mut bindings) = deferred_validation_bound_fixture();

        executor
            .run_with_device_bindings(&[], &mut bindings)
            .expect("bound run A submits asynchronously");
        assert_eq!(ep.executions.load(Ordering::Relaxed), 1);

        ep.fail_next.store(second_would_fail, Ordering::Relaxed);
        let second = executor
            .run_with_device_bindings(&[], &mut bindings)
            .expect_err("run B must be rejected until run A is consumed");
        assert!(
            second
                .to_string()
                .contains("previous deferred device validation is still pending"),
            "unexpected run-B refusal: {second}"
        );
        assert_eq!(
            ep.executions.load(Ordering::Relaxed),
            1,
            "rejected run B must not execute, regardless of whether it would succeed or fail"
        );

        let first = bindings[1].read_bytes_range(0, 4).unwrap_err();
        assert!(
            first
                .to_string()
                .contains("device validation failed (flags=0x40)"),
            "run A's sticky failure must remain observable after run B: {first}"
        );
        assert!(ep.validation_state.lock().unwrap().active.is_none());
    }
}

#[test]
fn foreign_binding_executor_drop_and_reset_cannot_consume_owned_validation() {
    #[derive(Clone, Copy, Debug)]
    enum ForeignAction {
        BindingDrop,
        ExecutorDrop,
        GraphReset,
    }

    for action in [
        ForeignAction::BindingDrop,
        ForeignAction::ExecutorDrop,
        ForeignAction::GraphReset,
    ] {
        let ep = Arc::new(DeferredValidationEp::new());
        let (mut owner, mut owner_bindings) =
            deferred_validation_bound_fixture_for_provider(Arc::clone(&ep));
        let (foreign, mut foreign_bindings) =
            deferred_validation_bound_fixture_for_provider(Arc::clone(&ep));
        assert_ne!(
            owner.validation_registration.as_ref().unwrap().owner(),
            foreign.validation_registration.as_ref().unwrap().owner()
        );

        owner
            .run_with_device_bindings(&[], &mut owner_bindings)
            .expect("the owner submits one deferred failing generation");
        assert_eq!(
            ep.executions.load(Ordering::Relaxed),
            1,
            "{action:?}: positive control must execute the owner's failing kernel"
        );
        let active = ep
            .validation_state
            .lock()
            .unwrap()
            .active
            .expect("the owner's validation token must remain pending");
        assert_eq!(
            active.owner(),
            owner.validation_registration.as_ref().unwrap().owner()
        );

        let sync_before = ep.sync_calls.load(Ordering::Relaxed);
        let mut foreign = Some(foreign);
        match action {
            ForeignAction::BindingDrop => {
                drop(foreign_bindings.remove(1));
            }
            ForeignAction::ExecutorDrop => {
                drop(foreign.take());
            }
            ForeignAction::GraphReset => {
                assert!(!foreign.as_mut().unwrap().reset_device_graph().unwrap());
            }
        }
        assert!(
            ep.sync_calls.load(Ordering::Relaxed) > sync_before,
            "{action:?}: positive control must prove the foreign teardown/reset synchronized"
        );
        assert_eq!(
            ep.validation_consume_attempts.load(Ordering::Relaxed),
            0,
            "{action:?}: a foreign owner must not even attempt to consume the active token"
        );
        assert_eq!(
            ep.validation_state
                .lock()
                .unwrap()
                .active
                .expect("foreign action must preserve the owner's token"),
            active
        );

        let error = owner_bindings[1].read_bytes_range(0, 4).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("device validation failed (flags=0x40)"),
            "{action:?}: the submitting owner's unseen failure must remain observable: {error}"
        );
        assert_eq!(
            ep.validation_consume_attempts.load(Ordering::Relaxed),
            1,
            "{action:?}: only the submitting binding may consume the token"
        );
        assert!(ep.validation_state.lock().unwrap().active.is_none());
    }
}

#[test]
fn completed_binding_receipt_survives_more_than_64_later_owner_submissions() {
    const LATER_SUBMISSIONS: usize = 96;

    let ep = Arc::new(DeferredValidationEp::new());
    let (mut first, mut first_bindings) =
        deferred_validation_bound_fixture_for_provider(Arc::clone(&ep));
    first
        .run_with_device_bindings(&[], &mut first_bindings)
        .expect("first owner submits one deferred failure");
    assert_eq!(
        first.check_device_capture_error().unwrap(),
        0x40,
        "the executor receipt must complete the shared submission while the binding stays unread"
    );

    ep.fail_next.store(false, Ordering::Relaxed);
    for submission in 0..LATER_SUBMISSIONS {
        let (mut sibling, mut sibling_bindings) =
            deferred_validation_bound_fixture_for_provider(Arc::clone(&ep));
        sibling
            .run_with_device_bindings(&[], &mut sibling_bindings)
            .unwrap_or_else(|error| panic!("later submission {submission} failed: {error}"));
        sibling_bindings[1]
            .read_bytes_range(0, 4)
            .unwrap_or_else(|error| panic!("later receipt {submission} failed: {error}"));
        drop(sibling_bindings);
        drop(sibling);
    }

    assert_eq!(
        ep.validation_state.lock().unwrap().owners.len(),
        3,
        "all 96 sibling executor/binding registrations must retire without leaking"
    );
    for observation in 1..=2 {
        let error = first_bindings[1].read_bytes_range(0, 4).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("device validation failed (flags=0x40)"),
            "old receipt observation {observation} was overwritten after {LATER_SUBMISSIONS} \
             later submissions: {error}"
        );
    }

    drop(first_bindings);
    drop(first);
    assert_eq!(
        ep.validation_state.lock().unwrap().owners.len(),
        0,
        "final owner teardown must retire every setup-time validation slot"
    );
    eprintln!(
        "validation-lifetime old_receipt_observations=2 later_submissions={LATER_SUBMISSIONS} \
         leaked_owner_slots=0"
    );
}

#[test]
fn stale_generation_cannot_consume_or_clear_current_submission() {
    let (mut executor, ep, mut bindings) = deferred_validation_bound_fixture();
    executor
        .run_with_device_bindings(&[], &mut bindings)
        .expect("first submission");
    let stale = bindings[1]
        .device_validation_token_for_test()
        .expect("first binding token");
    assert_eq!(executor.check_device_capture_error().unwrap(), 0x40);

    ep.fail_next.store(true, Ordering::Relaxed);
    executor
        .run_with_device_bindings(&[], &mut bindings)
        .expect("second submission");
    let current = bindings[1]
        .device_validation_token_for_test()
        .expect("second binding token");
    assert_ne!(stale.generation(), current.generation());
    let error = ep
        .consume_device_validation_error(bindings[1].validation_registration(), stale)
        .expect_err("stale generation must fail closed");
    assert!(
        error.to_string().contains("cannot consume active"),
        "stale token must be rejected without touching the active generation: {error}"
    );
    assert_eq!(
        ep.validation_state.lock().unwrap().active,
        Some(onnx_runtime_ep_api::DeviceValidationToken::new(
            executor.validation_registration.as_ref().unwrap().owner(),
            current.generation()
        )),
        "stale consume must not clear or replace the current active submission"
    );
    let current_error = bindings[1].read_bytes_range(0, 4).unwrap_err();
    assert!(
        current_error
            .to_string()
            .contains("device validation failed (flags=0x40)")
    );
}

#[test]
fn panicking_bound_run_consumes_deferred_validation_before_reuse() {
    let (mut executor, ep, mut bindings) = deferred_validation_bound_fixture();
    ep.panic_next.store(true, Ordering::Relaxed);

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = executor.run_with_device_bindings(&[], &mut bindings);
    }));
    assert!(panic.is_err());
    assert!(
        ep.validation_state.lock().unwrap().active.is_none(),
        "the unwind guard must consume the pending generation"
    );
    assert_eq!(
        ep.validation_latch.load(Ordering::Acquire),
        0,
        "the failing latch must be cleared only after it was observed during unwind"
    );

    ep.fail_next.store(false, Ordering::Relaxed);
    executor
        .run_with_device_bindings(&[], &mut bindings)
        .expect("the executor must remain reusable after panic cleanup");
    bindings[1]
        .read_bytes_range(0, 4)
        .expect("the healthy follow-up run must consume cleanly");
}

#[test]
fn failed_graph_reset_preserves_the_exact_local_installation_token() {
    let (mut executor, ep, _bindings) = deferred_validation_bound_fixture();
    let token = onnx_runtime_ep_api::DeviceGraphToken::new(
        executor.graph_owner,
        DeviceGraphSlot::Primary,
        7,
    );
    executor.cap_mut().device_graph_token = Some(token);
    ep.graph_reset_failure.store(true, Ordering::Relaxed);

    let error = executor.reset_device_graph().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("forced device graph reset failure"),
        "unexpected reset error: {error}"
    );
    assert_eq!(
        executor.cap().device_graph_token,
        Some(token),
        "host state must retain the token until provider reset succeeds"
    );

    ep.graph_reset_failure.store(false, Ordering::Relaxed);
    assert!(executor.reset_device_graph().unwrap());
    assert_eq!(executor.cap().device_graph_token, None);
    assert_eq!(ep.graph_reset_calls.load(Ordering::Relaxed), 2);
}

#[test]
fn device_bound_validation_is_deferred_until_a_partial_host_read() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
    graph.add_input(input);
    let output = graph.create_named_value("output", DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "DeferredValidation",
        vec![Some(input)],
        vec![output],
    ));
    graph.add_output(output);

    let ep = Arc::new(DeferredValidationEp::new());
    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        Arc::clone(&ep) as Arc<dyn ExecutionProvider>,
    )
    .unwrap();
    let mut bindings = vec![
        executor
            .allocate_device_binding("input".into(), None, DataType::Float32, vec![2], vec![2])
            .unwrap(),
        executor
            .allocate_device_output_binding("output".into(), DataType::Float32, vec![2], vec![2])
            .unwrap(),
    ];
    bindings[0]
        .write_bytes(
            0,
            &[3.0f32, 7.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();

    let outputs = executor
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].is_none());
    assert_eq!(ep.synchronized_executions.load(Ordering::Relaxed), 0);
    let error = bindings[1].read_bytes_range(0, 4).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("device validation failed (flags=0x40)"),
        "the partial binding read must surface the deferred validation failure: {error}"
    );
    assert_eq!(ep.synchronized_executions.load(Ordering::Relaxed), 1);
    assert_eq!(ep.resets.load(Ordering::Relaxed), 2);
}

#[test]
fn route_residency_boundary_fires_once_per_top_level_run_after_sync() {
    // Slice-7C reachability: the boundary consumer must be driven from the real
    // request lifecycle exactly once per top-level `run`, and only after the
    // request synchronization boundary (`sync`), never per kernel.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
    graph.add_input(input);
    let output = graph.create_named_value("output", DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "DeferredValidation",
        vec![Some(input)],
        vec![output],
    ));
    graph.add_output(output);

    let ep = Arc::new(DeferredValidationEp::new());
    // The boundary only fires on a clean validation latch, so keep every request
    // healthy (the failing-latch path is covered by the request-local test).
    ep.fail_next.store(false, Ordering::Relaxed);
    let boundary_calls = Arc::clone(&ep.route_boundary_calls);
    let before_sync = Arc::clone(&ep.route_boundary_before_sync);
    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        Arc::clone(&ep) as Arc<dyn ExecutionProvider>,
    )
    .unwrap();
    let input = Tensor::from_f32(&[2], &[3.0, 7.0]).unwrap();

    for run in 1..=3 {
        let out = executor.run(&[("input", &input)]).unwrap();
        assert_eq!(out[0].to_vec_f32(), vec![3.0, 7.0]);
        assert_eq!(
            boundary_calls.load(Ordering::Relaxed),
            run,
            "the Slice-7C boundary consumer must fire exactly once per top-level request"
        );
    }
    assert!(
        !before_sync.load(Ordering::Relaxed),
        "the boundary consumer must run only after the request synchronization boundary"
    );
}

#[test]
fn route_residency_boundary_skips_nested_control_flow_runs() {
    // One top-level kernel plus one kernel inside a taken `If` branch (a nested
    // `run_scoped_mode`). The Slice-7C boundary lives at the single top-level
    // host boundary, so it must fire exactly once for the whole request and
    // never for the nested subgraph run, even though two kernels execute.
    fn deferred_branch() -> Graph {
        let mut b = Graph::new();
        b.opset_imports.insert(String::new(), 17);
        // Producer-less "X": a capture bound from the enclosing scope by name.
        let x = b.create_named_value("X", DataType::Float32, static_shape([2]));
        let out = b.create_named_value("branch_out", DataType::Float32, static_shape([2]));
        b.insert_node(Node::new(
            NodeId(0),
            "DeferredValidation",
            vec![Some(x)],
            vec![out],
        ));
        b.add_output(out);
        b
    }

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let cond = graph.create_named_value("cond", DataType::Bool, static_shape([1]));
    graph.add_input(cond);
    let x = graph.create_named_value("X", DataType::Float32, static_shape([2]));
    graph.add_input(x);
    let top_out = graph.create_named_value("top_out", DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "DeferredValidation",
        vec![Some(x)],
        vec![top_out],
    ));
    let y = graph.create_named_value("Y", DataType::Float32, static_shape([2]));
    let if_node = graph.insert_node(Node::new(NodeId(0), "If", vec![Some(cond)], vec![y]));
    graph
        .subgraphs
        .insert((if_node, "then_branch".to_string()), deferred_branch());
    graph
        .subgraphs
        .insert((if_node, "else_branch".to_string()), deferred_branch());
    graph.add_output(top_out);
    graph.add_output(y);

    let ep = Arc::new(DeferredValidationEp::new());
    ep.fail_next.store(false, Ordering::Relaxed);
    let boundary_calls = Arc::clone(&ep.route_boundary_calls);
    let executions = Arc::clone(&ep.executions);
    let before_sync = Arc::clone(&ep.route_boundary_before_sync);
    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        Arc::clone(&ep) as Arc<dyn ExecutionProvider>,
    )
    .unwrap();
    let cond_t = Tensor::from_raw(DataType::Bool, vec![1], &[1u8]).unwrap();
    let x_t = Tensor::from_f32(&[2], &[3.0, 7.0]).unwrap();

    let out = executor.run(&[("cond", &cond_t), ("X", &x_t)]).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(
        executions.load(Ordering::Relaxed),
        2,
        "one top-level kernel plus one nested branch kernel must both execute"
    );
    assert_eq!(
        boundary_calls.load(Ordering::Relaxed),
        1,
        "the boundary consumer must fire once for the request, never for the nested subgraph run"
    );
    assert!(
        !before_sync.load(Ordering::Relaxed),
        "the boundary consumer must run only after the request synchronization boundary"
    );
}

#[test]
fn device_validation_reset_failure_is_not_ignored_after_a_run() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([1]));
    graph.add_input(input);
    let output = graph.create_named_value("output", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(0),
        "DeferredValidation",
        vec![Some(input)],
        vec![output],
    ));
    graph.add_output(output);

    let ep = Arc::new(DeferredValidationEp::new());
    ep.fail_next.store(false, Ordering::Relaxed);
    ep.reset_failure_at.store(2, Ordering::Relaxed);
    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        Arc::clone(&ep) as Arc<dyn ExecutionProvider>,
    )
    .unwrap();
    let input = Tensor::from_f32(&[1], &[5.0]).unwrap();

    let error = executor.run(&[("input", &input)]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("forced device validation reset failure"),
        "the post-run latch reset error must propagate: {error}"
    );
    assert_eq!(ep.executions.load(Ordering::Relaxed), 1);
    assert_eq!(ep.resets.load(Ordering::Relaxed), 2);
}

#[test]
fn a_late_lazy_gate_initialisation_cannot_revert_a_forced_gate() {
    // Regression test for an intermittent failure of
    // `activation_memory_planner_reports_static_decode_graph_savings` on
    // `Rust coverage (Windows x86_64)`: it forced the planner gate on, and the
    // run that followed published no stats, so the `expect` on them panicked.
    //
    // The third writer is the cause. `globals_lock` serialises the *tests* that
    // force these gates, but `activation_plan_enabled` also writes, from any
    // thread, without the lock - `run.rs` consults it on every executor run, so
    // under the parallel runner a sibling test is a writer too. Its
    // load / read-env / store is not atomic, so with an unconditional `store`
    // its answer could land after a forced value and revert it.
    //
    // Driven through the real gate and the real publish path rather than a
    // copy, so the test cannot pass against a seam that has drifted from the
    // code it stands in for.
    let _globals = phase_profile::globals_lock();

    // The interleaving that failed: a reader entered the initialiser, was
    // descheduled while reading the environment, and published `off` after the
    // force. Both env answers, because the defect is in publishing at all.
    for env_says in [false, true] {
        phase_profile::force_activation_plan_enabled(true);
        assert!(
            phase_profile::publish_env_derived(phase_profile::activation_plan_gate(), env_says),
            "a late initialiser must report the value in force, not its own"
        );
        assert!(
            phase_profile::activation_plan_enabled(),
            "a lazy initialisation that lands after a force (env_says={env_says}) \
             reverted it; the forcing test then runs with the planner off and \
             fails on stats it correctly asked for"
        );
    }

    // Non-vacuity: the initialiser must still publish when nothing is in force,
    // or the assertions above would hold for a `publish` that never stores.
    // On a scratch gate, so an unowned process-global is never left `UNKNOWN`
    // for a concurrent reader to re-derive.
    for env_says in [false, true] {
        let unowned = AtomicU8::new(phase_profile::UNKNOWN);
        assert_eq!(
            phase_profile::publish_env_derived(&unowned, env_says),
            env_says,
            "an uncontended initialiser must report its own env-derived answer"
        );
        let expected = if env_says {
            phase_profile::ON
        } else {
            phase_profile::OFF
        };
        assert_eq!(
            unowned.load(Ordering::Relaxed),
            expected,
            "an uncontended initialiser must publish, or every later reader \
             pays the environment read again"
        );
    }

    // Leave the planner off, as every other test in this file expects.
    phase_profile::force_activation_plan_enabled(false);
}

#[test]
fn phase_profile_gating_and_accumulation() {
    // The process-global gates have no per-test isolation, so every test that
    // writes them takes this lock.
    let _globals = phase_profile::globals_lock();

    // Disabled: a span records nothing and never captures a timestamp.
    phase_profile::force_enabled(false);
    let disabled_phase = "test.phase.disabled";
    let before = phase_profile::snapshot(disabled_phase);
    {
        let _s = phase_span!(disabled_phase);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        phase_profile::snapshot(disabled_phase),
        before,
        "a disabled phase span must not accumulate any samples"
    );

    // Enabled: a span accumulates exactly one positive-duration sample.
    phase_profile::force_enabled(true);
    let enabled_phase = "test.phase.enabled";
    let (base_ns, base_count) = phase_profile::snapshot(enabled_phase).unwrap_or((0, 0));
    {
        let _s = phase_span!(enabled_phase);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let (after_ns, after_count) =
        phase_profile::snapshot(enabled_phase).expect("enabled span must record a sample");
    assert_eq!(after_count, base_count + 1, "one span => one sample");
    assert!(
        after_ns > base_ns,
        "an enabled span must accumulate a positive duration"
    );
    assert!(
        phase_profile::snapshot(enabled_phase).is_some(),
        "enabled phase must appear in stats before reset"
    );
    phase_profile::reset();
    // Other tests can repopulate the process-global stats map under the
    // parallel runner, so assert only this test's unique phase was cleared.
    assert!(
        phase_profile::snapshot(enabled_phase).is_none(),
        "reset must clear this test's accumulated phase stats"
    );

    // Turning on phase profiling must **not** turn on the activation-memory
    // planner. The planner re-plans every activation on every run - work the
    // shipped runtime never does - so while `--phase-profile` enabled it the
    // profiler perturbed the run it was measuring, by about 3% on a softmax
    // decode, and reported its own cost back as a phase of that run.
    //
    // The two gates are separate on purpose: `enable_for_process` (what
    // `--phase-profile` calls) drives only the profiler, while the planner
    // reads the environment. Asserted here rather than in a sibling test so
    // the process-global enable flag stays owned by this one test, per the
    // note at the top. Skipped when the environment already asks for planning,
    // because then the separation is not observable.
    let env_requests_planning = ["NXRT_ACTIVATION_MEMORY_PLAN", "NXRT_EXEC_PHASE_PROFILE"]
        .iter()
        .any(|key| std::env::var(key).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")));
    if !env_requests_planning {
        phase_profile::force_activation_plan_enabled(false);
        phase_profile::enable_for_process();
        assert!(phase_profile::enabled());
        assert!(
            !phase_profile::activation_plan_enabled(),
            "phase profiling must not drag the activation-memory planner in \
             with it: the planner costs about a third of a small run and would \
             be charged to the run it is supposed to be measuring"
        );
        // The explicit opt-in still works, so the capability is decoupled from
        // phase profiling rather than lost.
        phase_profile::enable_activation_plan_for_process();
        assert!(phase_profile::activation_plan_enabled());
    }

    // Restore the default (disabled) state so other tests stay inert.
    phase_profile::force_enabled(false);
    phase_profile::force_activation_plan_enabled(false);
}

// A produced top-level output is handed to the caller by moving its host
// buffer out of the executor (zero-copy), while a producer-less output (an
// initializer routed straight to a graph output) stays on the copy path so
// its borrowed/shared storage is never freed. Repeated runs must keep
// producing correct bytes even though the produced buffer was moved out and
// its `buffer_shapes` entry cleared (forcing a fresh allocation each run).
#[test]
fn zero_copy_output_move_reallocates_and_preserves_producer_less_output() {
    use onnx_runtime_ir::TensorData;
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let a = graph.create_named_value("a", DataType::Float32, static_shape([3]));
    let b = graph.create_named_value("b", DataType::Float32, static_shape([3]));
    graph.add_input(a);
    graph.add_input(b);

    // Producer-less graph output: an initializer wired straight to an output.
    let k = graph.create_named_value("k", DataType::Float32, static_shape([3]));
    graph.set_initializer(
        k,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![3],
            [100.0f32, 200.0, 300.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );

    // Produced output: Add(a, b). This is the movable, owned, host output.
    let sum = graph.create_named_value("sum", DataType::Float32, static_shape([3]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(a), Some(b)],
        vec![sum],
    ));
    graph.add_output(sum);
    graph.add_output(k);

    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let a_val = Tensor::from_f32(&[3], &[1.0, 2.0, 3.0]).unwrap();
    let b_val = Tensor::from_f32(&[3], &[10.0, 20.0, 30.0]).unwrap();

    // Three runs prove the move-out survives buffer reallocation: run 1 moves
    // `sum`'s buffer out, so runs 2 and 3 must reallocate it from scratch.
    for _ in 0..3 {
        let outputs = executor
            .run(&[("a", &a_val), ("b", &b_val)])
            .expect("run must succeed after a prior output buffer was moved out");
        assert_eq!(outputs[0].to_vec_f32(), vec![11.0, 22.0, 33.0]);
        assert_eq!(
            outputs[1].to_vec_f32(),
            vec![100.0, 200.0, 300.0],
            "producer-less initializer output must stay intact across runs"
        );
        // The produced output was handed off zero-copy: its buffer is gone
        // from the executor. The initializer stays resident (copy path).
        assert!(
            !executor.buffers.contains_key(&sum),
            "produced output buffer must be moved out, not copied"
        );
        assert!(
            executor.buffers.contains_key(&k),
            "producer-less output must not have its buffer stolen"
        );
    }
}

fn inplace_chain_graph(keep_input_output: bool, keep_input_live: bool) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([4]));
    graph.add_input(input);
    let first = graph.create_named_value("first", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Tanh", vec![Some(input)], vec![first]));
    let output = graph.create_named_value("output", DataType::Float32, static_shape([4]));
    if keep_input_live {
        graph.insert_node(Node::new(
            NodeId(1),
            "Add",
            vec![Some(input), Some(first)],
            vec![output],
        ));
    } else {
        graph.insert_node(Node::new(
            NodeId(1),
            "Tanh",
            vec![Some(first)],
            vec![output],
        ));
    }
    if keep_input_output {
        graph.add_output(input);
        graph.add_output(first);
    }
    graph.add_output(output);
    graph
}

#[test]
fn compute_in_place_chain_is_byte_identical_and_fires() {
    let values = Tensor::from_f32(&[4], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
    let weights = Arc::new(WeightStore::new());
    let ep = auto_detect_cpu_ep().unwrap();
    let mut enabled = Executor::build(
        inplace_chain_graph(false, false),
        Arc::clone(&weights),
        Arc::clone(&ep),
    )
    .unwrap();
    let enabled_output = enabled.run(&[("input", &values)]).unwrap()[0]
        .as_bytes()
        .to_vec();
    // One alias, not two: the graph input is bound zero-copy (its buffer
    // borrows the caller's `values` tensor), and a borrowed buffer must never
    // be written, so `Tanh` cannot run in place on it. The intermediate value
    // `first` is executor-owned and still aliases. Trading one in-place alias
    // (which saves an allocation the run makes anyway on the next step) for
    // eliminating a full host->EP copy of every graph input is the point of
    // `prepare_run_buffers`.
    assert_eq!(enabled.compute_in_place_alias_count, 1);

    let mut disabled = Executor::build(inplace_chain_graph(false, false), weights, ep).unwrap();
    disabled.compute_in_place_enabled = false;
    let disabled_output = disabled.run(&[("input", &values)]).unwrap()[0]
        .as_bytes()
        .to_vec();
    assert_eq!(disabled.compute_in_place_alias_count, 0);
    assert_eq!(enabled_output, disabled_output);
}

#[test]
fn compute_in_place_refuses_live_and_graph_output_inputs() {
    let values = Tensor::from_f32(&[4], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
    for (keep_input_output, keep_input_live) in [(true, false), (false, true)] {
        let mut executor = Executor::build(
            inplace_chain_graph(keep_input_output, keep_input_live),
            Arc::new(WeightStore::new()),
            auto_detect_cpu_ep().unwrap(),
        )
        .unwrap();
        let outputs = executor.run(&[("input", &values)]).unwrap();
        assert_eq!(executor.compute_in_place_alias_count, 0);
        if keep_input_output {
            assert_eq!(outputs[0].as_bytes(), values.as_bytes());
        }
    }
}

/// An `If` branch body of `branch_out = Identity(h)`, capturing the outer value
/// `h` by name (a producer-less named value bound from the enclosing scope).
fn identity_capture_body() -> Graph {
    let mut body = Graph::new();
    body.opset_imports.insert(String::new(), 17);
    let captured = body.create_named_value("h", DataType::Float32, static_shape([4]));
    let out = body.create_named_value("branch_out", DataType::Float32, static_shape([4]));
    body.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(captured)],
        vec![out],
    ));
    body.add_output(out);
    body
}

/// `h = Relu(x); t = Tanh(h); y = If(cond){ Identity(h) }`. The intermediate `h`
/// has exactly one *formal* consumer (the in-place-eligible `Tanh`, whose output
/// `t` is dead), but both `If` branches capture `h` by name afterwards. Liveness
/// that ignores control-flow captures would mark `h` dead at `Tanh` and alias its
/// buffer away, so the later capture would read freed memory.
fn inplace_capture_if_graph() -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(x);
    let cond = graph.create_named_value("cond", DataType::Bool, static_shape([1]));
    graph.add_input(cond);
    let h = graph.create_named_value("h", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![h]));
    let t = graph.create_named_value("t", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(1), "Tanh", vec![Some(h)], vec![t]));
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(2), "If", vec![Some(cond)], vec![y]));
    graph.subgraphs.insert(
        (NodeId(2), "then_branch".to_string()),
        identity_capture_body(),
    );
    graph.subgraphs.insert(
        (NodeId(2), "else_branch".to_string()),
        identity_capture_body(),
    );
    graph.add_output(y);
    graph
}

/// Regression (issue #85): a value captured by an `If`/`Loop`/`Scan` body must
/// never be aliased away by an earlier in-place op, even though the capture is
/// implicit (by name) and absent from every plan node's formal `inputs`. Asserts
/// the default (enabled) path succeeds, returns `Relu(x)`, and is byte-identical
/// to the fully out-of-place reference path.
#[test]
fn compute_in_place_preserves_control_flow_captures() {
    let values = Tensor::from_f32(&[4], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
    let cond = Tensor::from_raw(DataType::Bool, vec![1], &[1]).unwrap();
    let weights = Arc::new(WeightStore::new());
    let ep = auto_detect_cpu_ep().unwrap();

    let mut enabled = Executor::build(
        inplace_capture_if_graph(),
        Arc::clone(&weights),
        Arc::clone(&ep),
    )
    .unwrap();
    let enabled_output = enabled.run(&[("x", &values), ("cond", &cond)]).unwrap()[0]
        .as_bytes()
        .to_vec();

    let mut disabled = Executor::build(inplace_capture_if_graph(), weights, ep).unwrap();
    disabled.compute_in_place_enabled = false;
    let disabled_output = disabled.run(&[("x", &values), ("cond", &cond)]).unwrap()[0]
        .as_bytes()
        .to_vec();

    let expected = Tensor::from_f32(&[4], &[0.0, 0.0, 0.5, 2.0]).unwrap();
    assert_eq!(enabled_output, expected.as_bytes());
    assert_eq!(enabled_output, disabled_output);
}

/// A multi-layer, decode-shaped graph over a length-4 activation vector,
/// mixing the two hazards a real transformer decode step combines and that a
/// single-op unit graph cannot exercise together:
///
///   * **residual reuse** — each layer computes `n = Tanh(prev)` then
///     `r = Add(prev, n)`, so `prev` is read *twice* and stays live across the
///     activation; and
///   * **a long-lived carry** (`k = Relu(x)`, a stand-in for a KV/cache value)
///     that is produced first and consumed only at the very end, staying live
///     across every intervening layer.
///
/// A pure activation chain (`t = Tanh(a); u = Tanh(t); ...`) is layered on the
/// tail so compute-in-place *does* fire (each link's input is genuinely dead),
/// while the residual and carry values above must be recognised as still-live
/// and left un-aliased. If liveness ever mis-marks a residual input or the carry
/// as dead and aliases its buffer away — the regression class issue #85 guards —
/// the enabled path reads clobbered memory and diverges from the out-of-place
/// reference, so the byte-identical assertion below fails.
fn decode_shaped_residual_graph() -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(x);

    // Long-lived carry produced first, consumed last (KV/cache stand-in).
    let k = graph.create_named_value("k", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![k]));

    // Three residual layers: `prev` feeds both the activation and the add, so it
    // must stay live across `Tanh` — a naive "dead at its next single use"
    // liveness would wrongly alias it away.
    let mut prev = x;
    let mut nid = 1u32;
    for layer in 0..3 {
        let n = graph.create_named_value(format!("n{layer}"), DataType::Float32, static_shape([4]));
        graph.insert_node(Node::new(NodeId(nid), "Tanh", vec![Some(prev)], vec![n]));
        nid += 1;
        let r = graph.create_named_value(format!("r{layer}"), DataType::Float32, static_shape([4]));
        graph.insert_node(Node::new(
            NodeId(nid),
            "Add",
            vec![Some(prev), Some(n)],
            vec![r],
        ));
        nid += 1;
        prev = r;
    }

    // Merge the carry back in — this is `k`'s only (and therefore last) use.
    let merged = graph.create_named_value("merged", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(nid),
        "Add",
        vec![Some(prev), Some(k)],
        vec![merged],
    ));
    nid += 1;

    // Pure activation tail: each link's input is genuinely dead, so
    // compute-in-place is *expected* to fire here (proving the optimization is
    // active in this graph, not silently disabled).
    let mut cur = merged;
    for tail in 0..3 {
        let out =
            graph.create_named_value(format!("tail{tail}"), DataType::Float32, static_shape([4]));
        graph.insert_node(Node::new(NodeId(nid), "Tanh", vec![Some(cur)], vec![out]));
        nid += 1;
        cur = out;
    }
    graph.add_output(cur);
    graph
}

fn view_shaped_activation_graph() -> Graph {
    use onnx_runtime_ir::{TensorData, WeightRef};

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(x);

    let owned = graph.create_named_value("owned", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![owned]));

    let shape = graph.create_named_value("shape", DataType::Int64, static_shape([2]));
    graph.set_initializer(
        shape,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![2],
            [2i64, 2].into_iter().flat_map(i64::to_le_bytes).collect(),
        )),
    );
    let view = graph.create_named_value("view", DataType::Float32, static_shape([2, 2]));
    graph.insert_node(Node::new(
        NodeId(1),
        "Reshape",
        vec![Some(owned), Some(shape)],
        vec![view],
    ));

    let live_use = graph.create_named_value("live_use", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Tanh",
        vec![Some(owned)],
        vec![live_use],
    ));

    let live_use_2d =
        graph.create_named_value("live_use_2d", DataType::Float32, static_shape([2, 2]));
    graph.insert_node(Node::new(
        NodeId(3),
        "Reshape",
        vec![Some(live_use), Some(shape)],
        vec![live_use_2d],
    ));

    let merged = graph.create_named_value("merged", DataType::Float32, static_shape([2, 2]));
    graph.insert_node(Node::new(
        NodeId(4),
        "Add",
        vec![Some(view), Some(live_use_2d)],
        vec![merged],
    ));
    graph.add_output(merged);
    graph
}

/// Regression guard for the issue #85 class the reported native/CUDA decode
/// scare pointed at: a still-live value (a residual input or a long-lived
/// KV/cache carry) must never be aliased away by an earlier in-place op on a
/// *real, multi-layer* decode-shaped graph — the shape a single-op unit test
/// cannot reproduce. Asserts the default (enabled) path (a) actually aliases
/// (so the guard is exercised, not vacuously satisfied) and (b) is byte-for-byte
/// identical to the fully out-of-place reference. Weakening the liveness guard
/// so a residual/carry input is aliased away makes the enabled output diverge
/// and this test fail (verified by mutation).
#[test]
fn compute_in_place_multilayer_decode_residual_is_byte_identical_and_fires() {
    let values = Tensor::from_f32(&[4], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
    let weights = Arc::new(WeightStore::new());
    let ep = auto_detect_cpu_ep().unwrap();

    let mut enabled = Executor::build(
        decode_shaped_residual_graph(),
        Arc::clone(&weights),
        Arc::clone(&ep),
    )
    .unwrap();
    let enabled_output = enabled.run(&[("x", &values)]).unwrap()[0]
        .as_bytes()
        .to_vec();
    assert!(
        enabled.compute_in_place_alias_count >= 3,
        "compute-in-place must fire on the decode-shaped graph's activation tail \
         (expected >= 3 aliases, got {}); a zero count would make this guard vacuous",
        enabled.compute_in_place_alias_count,
    );

    let mut disabled = Executor::build(decode_shaped_residual_graph(), weights, ep).unwrap();
    disabled.compute_in_place_enabled = false;
    let disabled_output = disabled.run(&[("x", &values)]).unwrap()[0]
        .as_bytes()
        .to_vec();
    assert_eq!(disabled.compute_in_place_alias_count, 0);
    assert_eq!(
        enabled_output, disabled_output,
        "compute-in-place aliased a still-live residual/carry value on a multi-layer \
         decode-shaped graph — the exact corruption class issue #85 must prevent",
    );
}

#[test]
fn activation_memory_planner_reports_static_decode_graph_savings() {
    // The planner is off by default, in tests as in production, so a test that
    // needs its stats asks for them explicitly. The guard takes the globals
    // lock (the gate is a process-global atomic) and clears the gate again on
    // drop, including on panic - leaving it set would let a reader test that
    // does not take the lock observe a planner this test switched on.
    let _planner = phase_profile::ActivationPlanForTest::on();

    let values = Tensor::from_f32(&[4], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
    let weights = Arc::new(WeightStore::new());
    let ep = auto_detect_cpu_ep().unwrap();
    let mut exec = Executor::build(decode_shaped_residual_graph(), weights, ep).unwrap();

    assert_eq!(
        exec.activation_memory_plan_stats(),
        None,
        "build-time planning would be view-blind, so it must not publish stats"
    );

    exec.run(&[("x", &values)]).unwrap();
    let run_stats = exec
        .activation_memory_plan_stats()
        .expect("run should refresh activation memory plan stats");
    assert!(run_stats.complete, "run stats were deferred: {run_stats:?}");
    assert!(run_stats.naive_bytes > run_stats.peak_bytes);
    assert!(run_stats.savings_ratio > 0.0);
}

#[test]
fn activation_memory_planner_uses_runtime_view_edges() {
    // The planner is off by default, in tests as in production, so a test that
    // needs its stats asks for them explicitly. The guard takes the globals
    // lock (the gate is a process-global atomic) and clears the gate again on
    // drop, including on panic - leaving it set would let a reader test that
    // does not take the lock observe a planner this test switched on.
    let _planner = phase_profile::ActivationPlanForTest::on();

    let values = Tensor::from_f32(&[4], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
    let weights = Arc::new(WeightStore::new());
    let ep = auto_detect_cpu_ep().unwrap();
    let mut exec = Executor::build(view_shaped_activation_graph(), weights, ep).unwrap();

    assert_eq!(
        exec.activation_memory_plan_stats(),
        None,
        "load-time stats would see an empty ViewMap for this Reshape fixture"
    );

    exec.run(&[("x", &values)]).unwrap();
    let run_stats = exec
        .activation_memory_plan_stats()
        .expect("run should measure after Reshape has reported view outputs");
    assert!(run_stats.complete, "run stats were deferred: {run_stats:?}");
    assert_eq!(run_stats.view_edges, 2);
    assert_eq!(run_stats.assignments, 3);
    assert_eq!(run_stats.naive_bytes, 48);
}

struct CaptureDecliningKernel;

impl Kernel for CaptureDecliningKernel {
    fn execute(
        &self,
        _inputs: &[TensorView],
        _outputs: &mut [TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "requires M==1 decode GEMV without group_indices; got a prefill signature",
        )
    }
}

// Mirrors the bypass kernels Harry flagged (`UnaryMathKernel`, `NotKernel`,
// `BitwiseNotKernel`): returns `CaptureSupport::Supported` unconditionally and
// deliberately does NOT override `set_capture_seq_independent`, so the kernel
// alone would happily admit a classifier-disqualified growing node into capture.
struct UnconditionalCaptureKernel;

impl Kernel for UnconditionalCaptureKernel {
    fn execute(
        &self,
        _inputs: &[TensorView],
        _outputs: &mut [TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::Supported
    }
}

// Build a minimal two-node static `Identity` executor whose kernels are warmed,
// then return it alongside the per-node kernel keys and the fully-resolved
// concrete shape map. Callers rewrite one node's IR output shape and cached
// kernel to stage a capture-admission scenario for `node_capture_reason`.
#[cfg(test)]
fn build_identity_capture_fixture() -> (Executor, Vec<KernelKey>, HashMap<ValueId, Vec<usize>>) {
    use onnx_runtime_ir::static_shape;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    for index in 0..2 {
        let input = graph.create_named_value(
            format!("input_{index}"),
            DataType::Float32,
            static_shape([1]),
        );
        let output = graph.create_named_value(
            format!("output_{index}"),
            DataType::Float32,
            static_shape([1]),
        );
        graph.add_input(input);
        graph.add_output(output);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![output],
        ));
    }

    let executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().expect("CPU EP"),
    )
    .expect("representative static graph");
    let resolved = executor
        .value_shapes
        .iter()
        .filter_map(|(&value, shape)| as_static_shape(shape).map(|shape| (value, shape)))
        .collect::<HashMap<_, _>>();
    let keys = executor
        .plan
        .iter()
        .map(|plan| KernelKey {
            node: plan.node_id.0,
            shapes: plan
                .inputs
                .iter()
                .map(|input| {
                    input
                        .map(|value| resolved[&value].clone())
                        .unwrap_or_default()
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    (executor, keys, resolved)
}

// Round-7 central capture veto (PR #728): a classifier-DISQUALIFIED node (an
// output shape references a GROWING KV/total-sequence-length symbol) wired to a
// kernel that returns `CaptureSupport::Supported` unconditionally MUST still be
// declined at the real capture-admission chokepoint (`node_capture_reason`). The
// veto is applied BEFORE the kernel's own `capture_support()` is consulted, so no
// bypass kernel can re-admit a disqualified node. Fail-pre (veto absent): the
// node is admitted (`None`) because every structural check passes and the kernel
// says Supported — silent decode corruption. Pass-post: `ClassifierDisqualified`.
#[test]
fn classifier_disqualified_node_is_vetoed_despite_supported_kernel() {
    let (mut executor, keys, resolved) = build_identity_capture_fixture();

    // Mint a GROWING KV-length symbol and mark node 0's OUTPUT shape as carrying
    // it, so the build-time classifier disqualifies the node — independent of the
    // concrete warmed extent still present in `resolved`.
    let growing = executor.graph.create_symbol(None);
    executor.capture_growing_symbols.insert(growing);
    let disqualified_output = executor.plan[0].outputs[0];
    executor.graph.value_mut(disqualified_output).shape =
        vec![Dim::Symbolic(growing), Dim::Static(1)];

    // The kernel alone would admit capture (unconditional Supported, no flag
    // override) — exactly the bypass path Harry called out.
    executor
        .cache
        .entries
        .insert(keys[0].clone(), Box::new(UnconditionalCaptureKernel));

    let decline = executor
        .node_capture_reason(&executor.plan[0], &resolved)
        .expect("classifier-disqualified node must be declined for capture");
    assert_eq!(
        decline.seam_reason,
        Some(SeamReason::ClassifierDisqualified),
        "the growing-symbol node must be vetoed centrally, not admitted by the kernel"
    );
}

// Positive companion: a classifier-QUALIFIED (sequence-independent) node with the
// same unconditional-`Supported` kernel is still admitted (`None`). The central
// veto is strictly additive — it only ever declines disqualified nodes and never
// suppresses a legitimately capturable one.
#[test]
fn classifier_qualified_node_with_supported_kernel_is_admitted() {
    let (mut executor, keys, resolved) = build_identity_capture_fixture();

    // No growing symbol touches this node's edges; the classifier qualifies it.
    executor
        .cache
        .entries
        .insert(keys[0].clone(), Box::new(UnconditionalCaptureKernel));

    assert!(
        executor
            .node_capture_reason(&executor.plan[0], &resolved)
            .is_none(),
        "a sequence-independent node with a Supported kernel must remain capture-eligible"
    );
}

// Round-8 veto-precedence fix (PR #728): a host control-flow node (`If`) that
// ALSO references a GROWING symbol is disqualified by the classifier AND
// classified as `HostControlFlowOrSequence` by the EP structural policy. The
// structural HOST seam must WIN so the public capture-segmentation report labels
// it a HOST round trip, not an eager DEVICE seam. Fail-pre (veto placed first, as
// on HEAD 9555b354): `node_capture_reason` returns `ClassifierDisqualified` whose
// `path_kind()` is `EagerDeviceSeam` — a host node mislabeled as a device seam.
// Pass-post (veto reordered after structural): `HostControlFlowOrSequence` whose
// `path_kind()` is `HostSeam`.
#[test]
fn disqualified_control_flow_node_reports_host_seam_not_device_seam() {
    let (mut executor, keys, resolved) = build_identity_capture_fixture();

    // Turn node 0 into a control-flow (`If`) node the EP structural policy
    // classifies as `HostControlFlowOrSequence`.
    executor.graph.node_mut(executor.plan[0].node_id).op_type = "If".to_string();
    assert!(
        is_control_flow_op(
            &executor.graph.node(executor.plan[0].node_id).op_type,
            &executor.graph.node(executor.plan[0].node_id).domain,
        ),
        "fixture node must be recognized as control flow"
    );

    // Also make it classifier-disqualified: an output shape references a GROWING
    // KV/total-sequence-length symbol, so the veto would ALSO fire on this node.
    let growing = executor.graph.create_symbol(None);
    executor.capture_growing_symbols.insert(growing);
    let disqualified_output = executor.plan[0].outputs[0];
    executor.graph.value_mut(disqualified_output).shape =
        vec![Dim::Symbolic(growing), Dim::Static(1)];
    assert!(
        !node_capture_seq_independent(
            &executor.graph,
            executor.graph.node(executor.plan[0].node_id),
            &executor.capture_growing_symbols,
        ),
        "the growing-symbol output must make the node classifier-disqualified"
    );

    // A warmed unconditional-`Supported` kernel is present too; it must not matter
    // because structural host classification precedes both the veto and the kernel.
    executor
        .cache
        .entries
        .insert(keys[0].clone(), Box::new(UnconditionalCaptureKernel));

    let decline = executor
        .node_capture_reason(&executor.plan[0], &resolved)
        .expect("control-flow node must be declined for capture");
    assert_eq!(
        decline.seam_reason,
        Some(SeamReason::HostControlFlowOrSequence),
        "a disqualified control-flow node must report the HOST control-flow seam, \
         not ClassifierDisqualified"
    );
    assert_eq!(
        decline
            .seam_reason
            .expect("seam reason present")
            .path_kind(),
        CapturePathKind::HostSeam,
        "the disqualified control-flow node must land on the HOST seam path, not an \
         eager DEVICE seam"
    );
}

#[test]
fn kernel_capture_reason_propagates_into_structured_report() {
    let mut node = Node::new(NodeId(9), "MatMulNBits", vec![], vec![]);
    node.domain = "com.microsoft".to_string();
    let decline = kernel_capture_decline(node.id, &node, &CaptureDecliningKernel).expect("decline");
    let report = CaptureDeclineReport::one(decline);

    assert_eq!(
        report.entries,
        vec![CaptureDecline {
            node_id: Some(9),
            op_type: "MatMulNBits".to_string(),
            domain: "com.microsoft".to_string(),
            reason: "requires M==1 decode GEMV without group_indices; got a prefill signature"
                .to_string(),
            seam_reason: Some(SeamReason::KernelCaptureUnsupported),
        }]
    );
    assert!(report.to_string().contains("node 9"));
    assert!(
        report
            .to_string()
            .contains("requires M==1 decode GEMV without group_indices")
    );
}

#[test]
fn seam_reasons_map_to_structural_capture_paths() {
    let cases = [
        (
            SeamReason::HostControlFlowOrSequence,
            CapturePathKind::HostSeam,
            "host-seam",
        ),
        (
            SeamReason::UnresolvedOutputShape,
            CapturePathKind::EagerDeviceSeam,
            "eager-device-seam",
        ),
        (
            SeamReason::UnresolvedInputShape,
            CapturePathKind::EagerDeviceSeam,
            "eager-device-seam",
        ),
        (
            SeamReason::KernelNotWarmed,
            CapturePathKind::EagerDeviceSeam,
            "eager-device-seam",
        ),
        (
            SeamReason::KernelCaptureUnsupported,
            CapturePathKind::EagerDeviceSeam,
            "eager-device-seam",
        ),
    ];

    for (reason, expected_kind, expected_label) in cases {
        assert_eq!(reason.path_kind(), expected_kind);
        assert_eq!(reason.label(), expected_label);
    }
    assert_eq!(CapturePathKind::CaptureRegion.label(), "capture-region");
}

#[test]
fn ep_structural_plan_plus_executor_kernel_checks_matches_legacy_declines() {
    use onnx_runtime_ir::static_shape;

    fn legacy_node_capture_reason(
        executor: &Executor,
        plan: &NodePlan,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Option<CaptureDecline> {
        let node = executor.graph.node(plan.node_id);
        if is_control_flow_op(&node.op_type, &node.domain)
            || is_sequence_op(&node.op_type, &node.domain)
        {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::HostControlFlowOrSequence,
                "control-flow and sequence nodes are not device-graph capturable",
            ));
        }
        if plan
            .outputs
            .iter()
            .any(|output| !resolved.contains_key(output))
        {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::UnresolvedOutputShape,
                "data-dependent output shape was unresolved before capture",
            ));
        }
        let Some(input_shapes) = plan
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| resolved.get(&value).cloned())
                    .unwrap_or(Some(Vec::new()))
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::UnresolvedInputShape,
                "data-dependent input shape was unresolved before capture",
            ));
        };
        let key = KernelKey {
            node: plan.node_id.0,
            shapes: input_shapes,
        };
        let Some(kernel) = executor.cache.entries.get(&key) else {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::KernelNotWarmed,
                "kernel has not been warmed for the requested capture shape",
            ));
        };
        kernel_capture_decline(plan.node_id, node, kernel.as_ref())
    }

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    for index in 0..6 {
        let input = graph.create_named_value(
            format!("input_{index}"),
            DataType::Float32,
            static_shape([1]),
        );
        let output = graph.create_named_value(
            format!("output_{index}"),
            DataType::Float32,
            static_shape([1]),
        );
        graph.add_input(input);
        graph.add_output(output);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![output],
        ));
    }

    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().expect("CPU EP"),
    )
    .expect("representative static graph");
    let mut resolved = executor
        .value_shapes
        .iter()
        .filter_map(|(&value, shape)| as_static_shape(shape).map(|shape| (value, shape)))
        .collect::<HashMap<_, _>>();
    let keys = executor
        .plan
        .iter()
        .map(|plan| KernelKey {
            node: plan.node_id.0,
            shapes: plan
                .inputs
                .iter()
                .map(|input| {
                    input
                        .map(|value| resolved[&value].clone())
                        .unwrap_or_default()
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    executor.graph.node_mut(executor.plan[0].node_id).op_type = "If".to_string();
    resolved.remove(&executor.plan[0].outputs[0]);
    resolved.remove(&executor.plan[0].inputs[0].expect("present input"));
    resolved.remove(&executor.plan[1].outputs[0]);
    resolved.remove(&executor.plan[1].inputs[0].expect("present input"));
    resolved.remove(&executor.plan[2].inputs[0].expect("present input"));
    executor.cache.entries.remove(&keys[3]);
    executor
        .cache
        .entries
        .insert(keys[4].clone(), Box::new(CaptureDecliningKernel));

    let legacy = executor
        .plan
        .iter()
        .map(|plan| legacy_node_capture_reason(&executor, plan, &resolved))
        .collect::<Vec<_>>();
    let refactored = executor
        .plan
        .iter()
        .map(|plan| executor.node_capture_reason(plan, &resolved))
        .collect::<Vec<_>>();

    assert_eq!(refactored, legacy);
    assert_eq!(
        refactored
            .iter()
            .map(|decline| decline.as_ref().and_then(|decline| decline.seam_reason))
            .collect::<Vec<_>>(),
        vec![
            Some(SeamReason::HostControlFlowOrSequence),
            Some(SeamReason::UnresolvedOutputShape),
            Some(SeamReason::UnresolvedInputShape),
            Some(SeamReason::KernelNotWarmed),
            Some(SeamReason::KernelCaptureUnsupported),
            None,
        ]
    );
}

#[test]
fn capture_shapes_seed_unresolved_external_values_without_overwriting_resolved_shapes() {
    let external_value = |shape| ExternalValue {
        dtype: DataType::Float32,
        shape,
        accepts_subshape: false,
        strides: None,
        fixed_stride_shape: None,
        ptr: 0,
        len: 0,
        alignment: 1,
        device: onnx_runtime_ir::DeviceId::cpu(),
    };
    let mut external = ExternalBindings::default();
    external
        .inputs
        .insert(ValueId(0), external_value(vec![1, 2]));
    external
        .outputs
        .insert(ValueId(1), external_value(vec![1, 4, 128, 64]));
    external
        .outputs
        .insert(ValueId(2), external_value(vec![1, 4, 128, 64]));

    let mut resolved = HashMap::from([(ValueId(0), vec![1, 1])]);
    external.seed_capture_shapes(&mut resolved);

    assert_eq!(resolved[&ValueId(0)], vec![1, 1]);
    assert_eq!(resolved[&ValueId(1)], vec![1, 4, 128, 64]);
    assert_eq!(resolved[&ValueId(2)], vec![1, 4, 128, 64]);
}

#[test]
fn only_gqa_cache_inputs_use_physical_capacity_as_kernel_geometry() {
    let mut gqa = Node::new(NodeId(0), "GroupQueryAttention", vec![], vec![]);
    gqa.domain = "com.microsoft".to_string();
    let attention = Node::new(NodeId(1), "Attention", vec![], vec![]);

    assert!(kernel_input_uses_physical_capacity(&gqa, 3));
    assert!(kernel_input_uses_physical_capacity(&gqa, 4));
    assert!(!kernel_input_uses_physical_capacity(&gqa, 0));
    assert!(!kernel_input_uses_physical_capacity(&attention, 4));
}

// C1 (build-time growing-symbol classifier, DENYLIST on BOTH edges): a pointwise
// op is capture-eligible iff NEITHER any input NOR any output references a symbol
// in the GROWING set (`compute_capture_growing_symbols`) — the KV/total-sequence
// length symbols on attention `past`/`present` cache sequence axes. Benign FRESH
// symbols (warm-decode-seeded non-growing extents) are absent from that set, so
// ops carrying only batch/query-seq/fresh dims stay capturable, preserving the
// 154→34 collapse.
//
// This test builds a synthetic decode graph (declared `inputs_embeds`/`logits`
// I/O plus a GQA node minting a growing KV-length symbol) and asserts: an op
// whose dims are batch/query-seq only is capturable, and an op that carries a
// growing KV symbol on its OUTPUT stays eager. The first-hop input alias and the
// harder downstream-consumer alias are covered by their own tests below
// (`growing_symbol_alias_keeps_downstream_consumer_eager`). No model files, no
// per-model hardcoding — growing membership, not dim position.
#[test]
fn growing_symbol_classifier_admits_pinned_and_rejects_growing_and_aliased_ops() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    // Interned symbols. `batch`/`seq` are pinned (never on a KV sequence axis);
    // `seq_kv` GROWS each decode step (KV penultimate).
    let batch = graph.create_symbol(None);
    let seq = graph.create_symbol(None);
    let seq_kv = graph.create_symbol(None);

    // Declared decode I/O: inputs_embeds `[batch, seq, 512]`, logits
    // `[batch, seq, vocab]`.
    let embeds = graph.create_named_value(
        "inputs_embeds",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(512)],
    );
    graph.add_input(embeds);
    // GQA past_key input (index 3): `[batch, kv_heads, seq_kv, head_dim]`.
    let past_key = graph.create_named_value(
        "past_key",
        DataType::Float32,
        vec![sym(batch), st(4), sym(seq_kv), st(64)],
    );
    graph.add_input(past_key);
    let logits = graph.create_named_value(
        "logits",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(32000)],
    );
    graph.add_output(logits);

    let mut gqa = Node::new(
        NodeId(0),
        "GroupQueryAttention",
        vec![
            Some(embeds),
            Some(embeds),
            Some(embeds),
            Some(past_key),
            Some(past_key),
        ],
        vec![],
    );
    gqa.domain = "com.microsoft".to_string();
    graph.insert_node(gqa);

    let growing = compute_capture_growing_symbols(&graph);
    assert!(
        growing.contains(&seq_kv),
        "the growing KV-length symbol (past_key penultimate) must be collected, got {growing:?}"
    );
    assert!(
        !growing.contains(&batch) && !growing.contains(&seq),
        "batch/query-seq must NOT be growing, got {growing:?}"
    );

    // Positive: a pointwise op whose only symbolic dims are batch/seq (no growing
    // symbol on any edge) is capture-eligible.
    let pinned_out = graph.create_named_value(
        "pinned_pointwise_out",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(512)],
    );
    let pinned_op = Node::new(NodeId(1), "Sigmoid", vec![Some(embeds)], vec![pinned_out]);
    assert!(
        node_capture_seq_independent(&graph, &pinned_op, &growing),
        "an op whose only symbolic dims are batch/seq must be capturable"
    );

    // Negative: a pointwise op whose OUTPUT carries the growing KV-length symbol
    // MUST stay eager (a different buffer extent every decode step).
    let kv_out = graph.create_named_value(
        "kv_pointwise_out",
        DataType::Float32,
        vec![sym(seq_kv), st(128)],
    );
    let kv_op = Node::new(NodeId(2), "Sigmoid", vec![Some(embeds)], vec![kv_out]);
    assert!(
        !node_capture_seq_independent(&graph, &kv_op, &growing),
        "an op whose output carries the growing KV-length symbol must stay eager"
    );
}

// Finding 1 (downstream-consumer alias — the hard case Harry called out).
// Shape inference substitutes the lower-id representative when broadcasting two
// distinct symbols (`context.rs::broadcast_dim`), so a growing KV symbol can be
// unified INTO `batch` on an aliasing op's OUTPUT; a DOWNSTREAM pointwise op then
// copies that shape, and BOTH its edges show only the pinned-looking `batch`.
// Exact per-symbol membership on the raw growing set would wrongly admit that
// consumer (silent decode corruption). `compute_capture_growing_symbols` closes
// the growing set under that same unification, so `batch` is marked growing and
// BOTH the aliasing op AND its downstream consumer stay EAGER. This asserts the
// consumer, not just the first aliasing op — the exact hole the re-review flagged.
#[test]
fn growing_symbol_alias_keeps_downstream_consumer_eager() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);
    let seq_kv = graph.create_symbol(None);

    let embeds =
        graph.create_named_value("inputs_embeds", DataType::Float32, vec![sym(batch), st(64)]);
    graph.add_input(embeds);
    // GQA mints the growing `seq_kv` on past_key's penultimate axis.
    let past_key = graph.create_named_value(
        "past_key",
        DataType::Float32,
        vec![sym(batch), st(4), sym(seq_kv), st(64)],
    );
    graph.add_input(past_key);
    let logits = graph.create_named_value("logits", DataType::Float32, vec![sym(batch), st(32000)]);
    graph.add_output(logits);
    let mut gqa = Node::new(
        NodeId(0),
        "GroupQueryAttention",
        vec![
            Some(embeds),
            Some(embeds),
            Some(embeds),
            Some(past_key),
            Some(past_key),
        ],
        vec![],
    );
    gqa.domain = "com.microsoft".to_string();
    graph.insert_node(gqa);

    // Aliasing broadcast op: input carries the growing `seq_kv`, the other input
    // carries `batch`; inference unifies them and writes the lower-id
    // representative (`batch`, created first) onto the OUTPUT `aliased_out`.
    let kv_shaped =
        graph.create_named_value("kv_shaped_in", DataType::Float32, vec![sym(seq_kv), st(64)]);
    graph.add_input(kv_shaped);
    let batch_shaped = graph.create_named_value(
        "batch_shaped_in",
        DataType::Float32,
        vec![sym(batch), st(64)],
    );
    graph.add_input(batch_shaped);
    let aliased_out =
        graph.create_named_value("aliased_out", DataType::Float32, vec![sym(batch), st(64)]);
    let aliased_op = Node::new(
        NodeId(1),
        "Add",
        vec![Some(kv_shaped), Some(batch_shaped)],
        vec![aliased_out],
    );
    graph.insert_node(aliased_op.clone());

    // Downstream consumer: reads and re-emits ONLY the representative `batch` on
    // both edges — no raw `seq_kv` anywhere on this op.
    let consumer_out = graph.create_named_value(
        "downstream_consumer_out",
        DataType::Float32,
        vec![sym(batch), st(64)],
    );
    let consumer_op = Node::new(
        NodeId(2),
        "Sigmoid",
        vec![Some(aliased_out)],
        vec![consumer_out],
    );

    // Drive REAL inference: the `Add` broadcast records union(seq_kv, batch) via
    // the single `broadcast_dim` chokepoint, persisting it onto
    // `graph.symbol_unifications` — the authoritative record the closure reads.
    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("inference on the alias graph must succeed");

    let growing = compute_capture_growing_symbols(&graph);
    assert!(
        growing.contains(&seq_kv),
        "the growing KV symbol must be collected, got {growing:?}"
    );
    // The broadcast of the growing `seq_kv` against `batch` cannot claim the two
    // are equal, so inference names the result with a fresh extent derived from
    // both. What must hold either way is that whatever symbol lands on the
    // aliasing op's OUTPUT is itself disqualifying — otherwise the downstream
    // consumer, which only ever sees that symbol, would look capture-eligible.
    let alias_rep = match graph.value(aliased_out).shape[0] {
        Dim::Symbolic(s) => s,
        ref other => panic!("expected a symbolic aliased extent, got {other:?}"),
    };
    assert!(
        growing.contains(&alias_rep),
        "the extent a growing symbol broadcast into must be in the CLOSED growing set, \
         got {growing:?} for {alias_rep:?}"
    );
    assert!(
        !node_capture_seq_independent(&graph, &aliased_op, &growing),
        "the first-hop aliasing op must stay eager"
    );
    assert!(
        !node_capture_seq_independent(&graph, &consumer_op, &growing),
        "the DOWNSTREAM consumer whose edges show only the representative must ALSO stay eager \
         (this fails on an un-closed exact-membership denylist)"
    );
}

// Finding 1, NON-elementwise aliasing (the escape Harry reproduced). Shape
// inference substitutes the lower-id representative for two distinct symbols not
// only in elementwise `broadcast`, but wherever any handler broadcasts — here a
// `MatMul` batch-dim broadcast (`linalg.rs::matmul_shape` → `ctx.broadcast`):
// `[seq_kv, M, K] @ [batch, K, N] -> [batch, M, N]` ERASES the growing `seq_kv`
// batch symbol into the pinned-looking `batch`. A downstream pointwise op then
// copies `[batch, M, N]` onto both edges. An elementwise-only closure (the prior
// revision) does NOT union MatMul batch dims, so it wrongly admitted that
// consumer — silent decode corruption. The authoritative
// `Graph::symbol_unifications` record (populated at the single `broadcast_dim`
// chokepoint that MatMul also funnels through) closes the growing set over
// `union(seq_kv, batch)`, so `batch` is marked growing and the consumer stays
// EAGER — with zero per-op enumeration in the executor. This drives REAL
// inference so the record→close path is exercised end to end; it FAILS on
// HEAD 817eee53 (elementwise-only closure ignores the MatMul alias).
#[test]
fn matmul_batch_alias_keeps_downstream_consumer_eager() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    // `batch` is created first, so its id is lower and it becomes the surviving
    // representative — the case where the growing symbol is genuinely erased.
    let batch = graph.create_symbol(None);
    let seq_kv = graph.create_symbol(None);

    // Declared `past_key` KV boundary mints the growing `seq_kv` (source-2 scan).
    let past_key = graph.create_named_value(
        "past_key",
        DataType::Float32,
        vec![sym(batch), st(4), sym(seq_kv), st(64)],
    );
    graph.add_input(past_key);

    // MatMul batch-dim broadcast: lhs batch axis = growing `seq_kv`, rhs batch
    // axis = `batch`; the contraction (last two) axes are static and match.
    let lhs = graph.create_named_value("qk", DataType::Float32, vec![sym(seq_kv), st(8), st(16)]);
    graph.add_input(lhs);
    let rhs = graph.create_named_value("w", DataType::Float32, vec![sym(batch), st(16), st(32)]);
    graph.add_input(rhs);
    let matmul_out = graph.create_named_value(
        "matmul_out",
        DataType::Float32,
        vec![sym(batch), st(8), st(32)],
    );
    let matmul = Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(lhs), Some(rhs)],
        vec![matmul_out],
    );
    graph.insert_node(matmul);

    // Downstream consumer sees ONLY the representative `batch` on both edges.
    let consumer_out = graph.create_named_value(
        "matmul_consumer_out",
        DataType::Float32,
        vec![sym(batch), st(8), st(32)],
    );
    let consumer = Node::new(
        NodeId(1),
        "Sigmoid",
        vec![Some(matmul_out)],
        vec![consumer_out],
    );
    graph.insert_node(consumer.clone());
    graph.add_output(consumer_out);

    // Real inference records union(seq_kv, batch) at the MatMul batch broadcast.
    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("inference on the MatMul-alias graph must succeed");

    let growing = compute_capture_growing_symbols(&graph);
    assert!(
        growing.contains(&seq_kv),
        "the growing KV symbol must be collected, got {growing:?}"
    );
    let matmul_rep = match graph.value(matmul_out).shape[0] {
        Dim::Symbolic(s) => s,
        ref other => panic!("expected a symbolic MatMul batch extent, got {other:?}"),
    };
    assert!(
        growing.contains(&matmul_rep),
        "the extent a MatMul batch-dim broadcast folded the growing `seq_kv` into must be in the \
         CLOSED growing set — this FAILS on an elementwise-only closure, got {growing:?} for \
         {matmul_rep:?}"
    );
    assert!(
        !node_capture_seq_independent(&graph, &consumer, &growing),
        "the downstream consumer whose edges show only the MatMul-aliased representative must \
         stay EAGER"
    );
}

// Round-4 escape (Harry): a DERIVED-symbol lineage loss. `Reshape([seq_kv,8],
// [-1])` forms the derived expression `seq_kv*8`; `SymbolInterner::lower` interns
// that non-bare expression to a BRAND-NEW fresh `SymbolId` and (pre-fix) records
// NOTHING, so `symbol_unifications` carries no edge `seq_kv -> fresh`. A
// downstream `Sigmoid` carrying only the fresh symbol was wrongly classified
// capture-safe -> silent decode corruption. The fix records a derivation edge
// `fresh -> seq_kv` at the `lower` chokepoint and closes the disqualifying set
// over derivation edges, so the fresh symbol is disqualifying and the `Sigmoid`
// stays EAGER. This drives REAL inference so the `lower`->record->close path is
// exercised end to end; it FAILS on HEAD 571ea0d9 (no derivation provenance).
#[test]
fn reshape_derived_growing_symbol_keeps_downstream_consumer_eager() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);
    let seq_kv = graph.create_symbol(None);

    // Declared `past_key` KV boundary mints the growing `seq_kv` (source-2 scan).
    let past_key = graph.create_named_value(
        "past_key",
        DataType::Float32,
        vec![sym(batch), st(4), sym(seq_kv), st(64)],
    );
    graph.add_input(past_key);

    // A `[seq_kv, 8]` tensor carrying the growing symbol.
    let kv2d = graph.create_named_value("kv2d", DataType::Float32, vec![sym(seq_kv), st(8)]);
    graph.add_input(kv2d);

    // Reshape target `[-1]` as an int64 initializer -> shape-data source. The
    // derived output dim is `seq_kv*8`, which `lower` interns to a fresh symbol.
    let target = graph.create_named_value("reshape_target", DataType::Int64, vec![st(1)]);
    {
        use onnx_runtime_ir::{TensorData, WeightRef};
        graph.set_initializer(
            target,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Int64,
                vec![1],
                (-1i64).to_le_bytes().to_vec(),
            )),
        );
    }
    let reshaped = graph.create_named_value("reshaped", DataType::Float32, Shape::new());
    let reshape = Node::new(
        NodeId(0),
        "Reshape",
        vec![Some(kv2d), Some(target)],
        vec![reshaped],
    );
    graph.insert_node(reshape);

    // Downstream consumer sees only the derived (fresh) symbol on both edges.
    let sig_out = graph.create_named_value("reshape_sig_out", DataType::Float32, Shape::new());
    let consumer = Node::new(NodeId(1), "Sigmoid", vec![Some(reshaped)], vec![sig_out]);
    graph.insert_node(consumer.clone());
    graph.add_output(sig_out);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("inference on the reshape-derived graph must succeed");

    let growing = compute_capture_growing_symbols(&graph);
    assert!(
        growing.contains(&seq_kv),
        "the growing KV symbol must be collected, got {growing:?}"
    );
    // The reshape output's fresh derived symbol must be in the CLOSED set (it
    // depends on the growing `seq_kv` via the recorded derivation edge).
    let reshaped_dim = graph
        .try_value(reshaped)
        .and_then(|v| v.shape.first().copied());
    let Some(Dim::Symbolic(derived)) = reshaped_dim else {
        panic!("reshape output must be a derived symbolic dim, got {reshaped_dim:?}");
    };
    assert!(
        growing.contains(&derived),
        "the fresh symbol `seq_kv*8` derived from a growing symbol must be in the CLOSED \
         disqualifying set (this FAILS on HEAD 571ea0d9 — no derivation provenance), got {growing:?}"
    );
    assert!(
        !node_capture_seq_independent(&graph, &consumer, &growing),
        "the downstream consumer of a growing-derived reshape output must stay EAGER"
    );
}

// Same round-4 escape via `Flatten` (`transform.rs::flatten`), whose collapsed
// axes form the derived product `prod(dims[axis..])`. `Flatten` of `[seq_kv, 8]`
// at axis=1 keeps outer `seq_kv`, but at axis=0 forms `1 x (seq_kv*8)`; here we
// flatten `[batch, seq_kv, 8]` at axis=1 so the trailing dim is the derived
// `seq_kv*8`. Its fresh symbol must be disqualifying and the consumer EAGER.
#[test]
fn flatten_derived_growing_symbol_keeps_downstream_consumer_eager() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);
    let seq_kv = graph.create_symbol(None);

    let past_key = graph.create_named_value(
        "past_key",
        DataType::Float32,
        vec![sym(batch), st(4), sym(seq_kv), st(64)],
    );
    graph.add_input(past_key);

    let kv3d = graph.create_named_value(
        "kv3d",
        DataType::Float32,
        vec![sym(batch), sym(seq_kv), st(8)],
    );
    graph.add_input(kv3d);

    // Flatten at axis=1 -> `[batch, seq_kv*8]`; the trailing dim is derived.
    let flat = graph.create_named_value("flat", DataType::Float32, Shape::new());
    let mut flatten = Node::new(NodeId(0), "Flatten", vec![Some(kv3d)], vec![flat]);
    flatten.attributes.insert("axis".into(), Attribute::Int(1));
    graph.insert_node(flatten);

    let sig_out = graph.create_named_value("flatten_sig_out", DataType::Float32, Shape::new());
    let consumer = Node::new(NodeId(1), "Sigmoid", vec![Some(flat)], vec![sig_out]);
    graph.insert_node(consumer.clone());
    graph.add_output(sig_out);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("inference on the flatten-derived graph must succeed");

    let growing = compute_capture_growing_symbols(&graph);
    let flat_dim = graph.try_value(flat).and_then(|v| v.shape.get(1).copied());
    let Some(Dim::Symbolic(derived)) = flat_dim else {
        panic!("flatten trailing dim must be a derived symbolic dim, got {flat_dim:?}");
    };
    assert!(
        growing.contains(&derived),
        "the fresh symbol `seq_kv*8` derived by Flatten from a growing symbol must be in the \
         CLOSED disqualifying set (FAILS on HEAD 571ea0d9), got {growing:?}"
    );
    assert!(
        !node_capture_seq_independent(&graph, &consumer, &growing),
        "the downstream consumer of a growing-derived flatten output must stay EAGER"
    );
}

// FAIL-SAFE (Step 2), part 1 — provenance RECOVERS a pinned-derived fresh
// symbol. `Reshape([batch, 8], [-1])` derives `batch*8`, interned to a fresh
// symbol whose recorded provenance traces ONLY to the pinned root `batch`. The
// fail-safe classifier therefore does NOT disqualify it, so the consumer stays
// CAPTURABLE — this is precisely why the fail-safe (with the Step-1 provenance
// record) does not regress into the naive pinned-allowlist's segment collapse:
// a fresh symbol built purely from pinned sources is provably pinned.
#[test]
fn failsafe_pinned_derived_fresh_symbol_stays_capturable() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);

    let pinned_2d =
        graph.create_named_value("pinned_2d", DataType::Float32, vec![sym(batch), st(8)]);
    graph.add_input(pinned_2d);

    let target = graph.create_named_value("reshape_target", DataType::Int64, vec![st(1)]);
    {
        use onnx_runtime_ir::{TensorData, WeightRef};
        graph.set_initializer(
            target,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Int64,
                vec![1],
                (-1i64).to_le_bytes().to_vec(),
            )),
        );
    }
    let reshaped = graph.create_named_value("reshaped", DataType::Float32, Shape::new());
    let reshape = Node::new(
        NodeId(0),
        "Reshape",
        vec![Some(pinned_2d), Some(target)],
        vec![reshaped],
    );
    graph.insert_node(reshape);

    let sig_out = graph.create_named_value("sig_out", DataType::Float32, Shape::new());
    let consumer = Node::new(NodeId(1), "Sigmoid", vec![Some(reshaped)], vec![sig_out]);
    graph.insert_node(consumer.clone());
    graph.add_output(sig_out);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("inference must succeed");

    let derived = match graph
        .try_value(reshaped)
        .and_then(|v| v.shape.first().copied())
    {
        Some(Dim::Symbolic(s)) => s,
        other => panic!("reshape output must be a derived symbolic dim, got {other:?}"),
    };

    let not_pinned = compute_not_pinned_symbols(&graph);
    assert!(
        !not_pinned.contains(&derived),
        "a fresh symbol derived only from the pinned root `batch` must NOT be disqualifying \
         under the fail-safe classifier, got {not_pinned:?}"
    );
    assert!(
        node_capture_seq_independent(&graph, &consumer, &not_pinned),
        "a consumer of a pinned-derived reshape output must stay CAPTURABLE under fail-safe"
    );
}

// FAIL-SAFE (Step 2), part 2 — the structural win. An inference-minted symbol
// with NO recorded provenance (here a permissive-broadcast degrade of two
// unequal static extents `[batch,4] (+) [batch,5]`, standing in for any
// data-dependent `NonZero`/`Range`/`Slice` fresh dim) is UNTRACEABLE. The
// DENYLIST admits it (not proven growing ⇒ capturable) — the latent
// silent-corruption hole. The FAIL-SAFE classifier disqualifies it (unknown ⇒
// eager ⇒ safe), structurally eliminating the whole "unrecorded lineage site"
// bug class without a per-site whack-a-mole fix.
#[test]
fn failsafe_untraceable_minted_symbol_is_eager_but_denylist_admits_it() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);

    let a = graph.create_named_value("a", DataType::Float32, vec![sym(batch), st(4)]);
    let b = graph.create_named_value("b", DataType::Float32, vec![sym(batch), st(5)]);
    graph.add_input(a);
    graph.add_input(b);
    // Permissive broadcast of unequal, non-1 static extents mints an honest
    // "unknown" fresh symbol with no provenance (context.rs `broadcast_dim`).
    let added = graph.create_named_value("added", DataType::Float32, Shape::new());
    let add = Node::new(NodeId(0), "Add", vec![Some(a), Some(b)], vec![added]);
    graph.insert_node(add);

    let sig_out = graph.create_named_value("sig_out", DataType::Float32, Shape::new());
    let consumer = Node::new(NodeId(1), "Sigmoid", vec![Some(added)], vec![sig_out]);
    graph.insert_node(consumer.clone());
    graph.add_output(sig_out);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("inference must succeed");

    let unknown = match graph.try_value(added).and_then(|v| v.shape.get(1).copied()) {
        Some(Dim::Symbolic(s)) => s,
        other => panic!("Add output last dim must be an unknown minted symbol, got {other:?}"),
    };
    assert!(
        unknown.0
            >= graph
                .inference_symbol_floor
                .expect("inference sets the floor"),
        "the degrade symbol must be inference-minted (id above the floor)"
    );

    // DENYLIST: not growing, no provenance ⇒ capturable (the latent hole).
    let denylist = compute_capture_growing_symbols(&graph);
    assert!(
        !denylist.contains(&unknown),
        "the denylist does not disqualify an untraceable minted symbol, got {denylist:?}"
    );
    assert!(
        node_capture_seq_independent(&graph, &consumer, &denylist),
        "under the denylist the consumer of an untraceable symbol is (unsafely) capturable"
    );

    // FAIL-SAFE: untraceable minted symbol ⇒ disqualifying ⇒ consumer EAGER.
    let not_pinned = compute_not_pinned_symbols(&graph);
    assert!(
        not_pinned.contains(&unknown),
        "the fail-safe classifier must disqualify an untraceable minted symbol, got {not_pinned:?}"
    );
    assert!(
        !node_capture_seq_independent(&graph, &consumer, &not_pinned),
        "under fail-safe the consumer of an untraceable symbol must stay EAGER"
    );
}

// Finding 2 (coverage of CompressedSparseAttention): CSA mints a growing
// cache-record symbol from `total_sequence_length` on `outputs[1]`/`[3]`
// (`[query[0], records, width]`). The `attention_kv_cache_slots` CSA entry
// collects that symbol as GROWING, so any pointwise op consuming a CSA cache
// tensor stays EAGER on the denylist — closing the finding-2 gap that the old
// 3-op collector left open.
#[test]
fn csa_cache_record_symbol_keeps_consuming_ops_eager() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);
    let seq = graph.create_symbol(None);
    let records = graph.create_symbol(None); // total_sequence_length-derived

    let embeds = graph.create_named_value(
        "inputs_embeds",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(512)],
    );
    graph.add_input(embeds);
    let logits = graph.create_named_value(
        "logits",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(32000)],
    );
    graph.add_output(logits);

    // CSA output[1] cache record: `[query[0], records, width]`, penultimate =
    // records (the growing total-sequence-length-derived axis).
    let attn_out = graph.create_named_value(
        "csa_attn",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(512)],
    );
    let cache_out = graph.create_named_value(
        "csa_cache",
        DataType::Float32,
        vec![sym(batch), sym(records), st(64)],
    );
    let mut csa = Node::new(
        NodeId(0),
        "CompressedSparseAttention",
        vec![Some(embeds)],
        vec![attn_out, cache_out],
    );
    csa.domain = "pkg.nxrt".to_string();
    graph.insert_node(csa);

    let growing = compute_capture_growing_symbols(&graph);
    assert!(
        growing.contains(&records),
        "the CSA total_sequence_length-derived cache-record symbol must be GROWING, got {growing:?}"
    );

    // A pointwise op consuming the CSA cache tensor stays eager.
    let cache_pointwise = graph.create_named_value(
        "csa_cache_pointwise",
        DataType::Float32,
        vec![sym(batch), sym(records), st(64)],
    );
    let cache_op = Node::new(
        NodeId(1),
        "Relu",
        vec![Some(cache_out)],
        vec![cache_pointwise],
    );
    assert!(
        !node_capture_seq_independent(&graph, &cache_op, &growing),
        "an op consuming a CSA cache-record tensor must stay eager"
    );
}

// Finding 2 (CSA ratio-4 output 5 `selections`): the dynamic ratio-4 variant
// mints a fresh, growing `selections` symbol on the LAST axis of output 5
// (`[query[0], index_heads, query_seq, selections]`,
// custom_ops.rs::compressed_sparse_attention). The penultimate scan that covers
// outputs 1/3 would miss it, so `KvCacheSlots::last_axis_outputs` collects the
// trailing axis of output 5. A pointwise op consuming that `selections`-shaped
// value must therefore stay EAGER.
#[test]
fn csa_output5_selections_symbol_keeps_consuming_ops_eager() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);
    let seq = graph.create_symbol(None);
    let records = graph.create_symbol(None); // total_sequence_length-derived
    let selections = graph.create_symbol(None); // output-5 last axis (fresh, growing)

    let embeds = graph.create_named_value(
        "inputs_embeds",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(512)],
    );
    graph.add_input(embeds);
    let logits = graph.create_named_value(
        "logits",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(32000)],
    );
    graph.add_output(logits);

    // Six ratio-4 outputs. Output 0 = attention; outputs 1/3 = cache records
    // (penultimate `records`); outputs 2/4 = static compressor tensors; output 5
    // = `[query[0], index_heads, query_seq, selections]` (selections on LAST
    // axis). Only `records` and `selections` are growing.
    let out0 = graph.create_named_value(
        "csa_attn",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(512)],
    );
    let out1 = graph.create_named_value(
        "csa_cache",
        DataType::Float32,
        vec![sym(batch), sym(records), st(64)],
    );
    let out2 = graph.create_named_value(
        "csa_comp",
        DataType::Float32,
        vec![sym(batch), st(8), st(2), st(128)],
    );
    let out3 = graph.create_named_value(
        "csa_index",
        DataType::Uint8,
        vec![sym(batch), sym(records), st(8)],
    );
    let out4 = graph.create_named_value(
        "csa_index_comp",
        DataType::Float32,
        vec![sym(batch), st(8), st(2), st(64)],
    );
    let out5 = graph.create_named_value(
        "csa_selections",
        DataType::Int32,
        vec![sym(batch), st(8), sym(seq), sym(selections)],
    );
    let mut csa = Node::new(
        NodeId(0),
        "CompressedSparseAttention",
        vec![Some(embeds)],
        vec![out0, out1, out2, out3, out4, out5],
    );
    csa.domain = "pkg.nxrt".to_string();
    graph.insert_node(csa);

    let growing = compute_capture_growing_symbols(&graph);
    assert!(
        growing.contains(&selections),
        "the CSA output-5 last-axis `selections` symbol must be GROWING, got {growing:?}"
    );
    assert!(
        growing.contains(&records),
        "the CSA output-1/3 penultimate `records` symbol must be GROWING, got {growing:?}"
    );

    // A pointwise op consuming the CSA `selections`-shaped output stays eager.
    let sel_pointwise = graph.create_named_value(
        "csa_selections_pointwise",
        DataType::Int32,
        vec![sym(batch), st(8), sym(seq), sym(selections)],
    );
    let sel_op = Node::new(NodeId(1), "Sign", vec![Some(out5)], vec![sel_pointwise]);
    assert!(
        !node_capture_seq_independent(&graph, &sel_op, &growing),
        "an op consuming the CSA output-5 `selections` axis must stay eager"
    );
}

// Finding 2 (generic declared-KV-I/O coverage): even without any recognized
// attention op, a model that declares a `present…` rank-4 KV output boundary
// tensor (`[batch, kv_heads, present_seq, head_dim]`) has its growing sequence
// symbol collected by the generic scan, so a pointwise op sized by it stays
// eager. This is what makes finding-2 robust against unrecognized attention
// variants without minting a per-op entry.
#[test]
fn generic_declared_present_kv_output_is_collected_as_growing() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);
    let present_seq = graph.create_symbol(None);

    // Declared `present.0.key` KV output with a rank-4 layout and a symbolic
    // penultimate (sequence) axis — the ONNX GenAI KV-cache contract.
    let present_key = graph.create_named_value(
        "present.0.key",
        DataType::Float32,
        vec![sym(batch), st(4), sym(present_seq), st(64)],
    );
    graph.add_output(present_key);

    let growing = compute_capture_growing_symbols(&graph);
    assert!(
        growing.contains(&present_seq),
        "a declared present.* rank-4 KV output's sequence symbol must be GROWING, got {growing:?}"
    );

    let out = graph.create_named_value(
        "kv_sized_out",
        DataType::Float32,
        vec![sym(batch), sym(present_seq), st(64)],
    );
    let op = Node::new(NodeId(0), "Sigmoid", vec![Some(present_key)], vec![out]);
    assert!(
        !node_capture_seq_independent(&graph, &op, &growing),
        "an op sized by a declared present.* KV sequence symbol must stay eager"
    );
}

// Design tradeoff of the growing DENYLIST (the accepted fallback): a benign
// FRESH symbol — one warm-decode-seeded from a data-dependent extent, neither
// batch/query-seq nor on a KV sequence axis — is NOT in the growing set, so an op
// carrying it stays CAPTURABLE. This is deliberate and load-bearing: a pinned
// ALLOWLIST would keep all such ops eager and dissolve the 154→34 collapse. Only
// a genuinely growing dim disqualifies an op.
#[test]
fn benign_fresh_symbol_is_not_growing_and_stays_capturable() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);
    let seq = graph.create_symbol(None);
    let fresh = graph.create_symbol(None); // warm-seeded data-dependent extent

    let embeds = graph.create_named_value(
        "inputs_embeds",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(512)],
    );
    graph.add_input(embeds);
    let logits = graph.create_named_value(
        "logits",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(32000)],
    );
    graph.add_output(logits);

    let growing = compute_capture_growing_symbols(&graph);
    assert!(
        !growing.contains(&fresh),
        "a fresh non-KV symbol must NOT be growing, got {growing:?}"
    );

    let out = graph.create_named_value(
        "fresh_out",
        DataType::Float32,
        vec![sym(batch), sym(fresh), st(128)],
    );
    let op = Node::new(NodeId(0), "Sigmoid", vec![Some(embeds)], vec![out]);
    assert!(
        node_capture_seq_independent(&graph, &op, &growing),
        "an op carrying only a benign fresh (non-growing) symbol must stay capturable"
    );
}

// A recurrent-state cache (GatedDeltaNet conv/recurrent state) has a STATIC
// penultimate axis, so it must NOT contribute a growing symbol — the whole point
// that lets GDN pointwise ops become capture-eligible. A pure-recurrent graph
// with no attention KV cache yields an empty growing set (broaden all), which is
// correct: nothing grows step-to-step.
#[test]
fn recurrent_state_shapes_contribute_no_growing_symbols() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let batch = graph.create_symbol(None);
    let sym = Dim::Symbolic;
    let st = Dim::Static;

    // A fixed-capacity recurrent conv state: penultimate axis is STATIC.
    let conv_state = graph.create_named_value(
        "conv_state",
        DataType::Float32,
        vec![sym(batch), st(16), st(4), st(128)],
    );
    let q = graph.create_named_value("q", DataType::Float32, vec![sym(batch), st(1), st(512)]);
    // A default-domain `Attention` node with the recurrent state at input 4:
    // because its penultimate axis is static, no growing symbol is collected.
    let attention = Node::new(
        NodeId(0),
        "Attention",
        vec![Some(q), Some(q), Some(q), Some(q), Some(conv_state)],
        vec![],
    );
    graph.insert_node(attention);

    let growing = compute_capture_growing_symbols(&graph);
    assert!(
        growing.is_empty(),
        "a static-penultimate recurrent state must contribute no growing symbols, got {growing:?}"
    );
}

#[test]
fn only_capacity_aware_inputs_keep_physical_capacity() {
    let shape = Node::new(NodeId(0), "Shape", vec![], vec![]);
    let reduce_sum = Node::new(NodeId(1), "ReduceSum", vec![], vec![]);
    let cumsum = Node::new(NodeId(2), "CumSum", vec![], vec![]);
    let unsqueeze = Node::new(NodeId(3), "Unsqueeze", vec![], vec![]);

    assert!(kernel_input_uses_padded_capacity(&shape, 0));
    assert!(kernel_input_uses_padded_capacity(&reduce_sum, 0));
    assert!(!kernel_input_uses_padded_capacity(&cumsum, 0));
    assert!(!kernel_input_uses_padded_capacity(&unsqueeze, 0));
    assert!(!kernel_input_uses_padded_capacity(&shape, 1));

    // GLM-5.2's `indexer` attention branch consumes the attention mask
    // through elementwise arithmetic (a `Cast`→`Add` that combines a
    // logical-width indexer score with the mask). Such a consumer is NOT
    // padded-capacity-safe: it must observe the logical valid length, or the
    // padded physical capacity (`max_len`) leaks into the `Add` and fails to
    // broadcast against the logical-width score. Because it is not in the
    // Shape/ReduceSum allowlist, the binding is forced to expose its logical
    // prefix — which is exactly what fixes the GLM-5.2 decode broadcast bug.
    let indexer_add = Node::new(NodeId(4), "Add", vec![], vec![]);
    let indexer_cast = Node::new(NodeId(5), "Cast", vec![], vec![]);
    assert!(!kernel_input_uses_padded_capacity(&indexer_add, 0));
    assert!(!kernel_input_uses_padded_capacity(&indexer_cast, 0));
}

/// A decline is only actionable if it names the consumer that caused it.
/// `describe_non_padded_consumer` is the single place that formats that name,
/// and it is derived from the same allowlist as the predicate so the two cannot
/// drift apart.
#[test]
fn non_padded_consumers_are_named_for_attribution() {
    let mut cast = Node::new(NodeId(0), "Cast", vec![], vec![]);
    cast.name = "model/Cast_node_5".to_string();
    assert_eq!(
        describe_non_padded_consumer(&cast, 0).as_deref(),
        Some("model/Cast_node_5(Cast)[input 0]")
    );

    // A capacity-safe consumer is not an offender and must not be reported,
    // otherwise every model would look like it had a blocker.
    let shape = Node::new(NodeId(1), "Shape", vec![], vec![]);
    assert_eq!(describe_non_padded_consumer(&shape, 0), None);

    // Non-zero slots are outside the allowlist regardless of op type, and an
    // unnamed node still has to produce a usable message.
    let unnamed = Node::new(NodeId(2), "Shape", vec![], vec![]);
    assert_eq!(
        describe_non_padded_consumer(&unnamed, 1).as_deref(),
        Some("<unnamed>(Shape)[input 1]")
    );
}

// A capacity-form default-domain `Attention`: mask at input 3, KV cache at
// inputs 4/5 — so it derives the valid length from the mask frontier and binds
// the KV cache at physical capacity, in either the causal or non-causal form.
fn capacity_form_attention(id: u32, q: ValueId, mask: ValueId, out: ValueId) -> Node {
    Node::new(
        NodeId(id),
        "Attention",
        vec![Some(q), Some(q), Some(q), Some(mask), Some(q), Some(q)],
        vec![out],
    )
}

#[test]
fn capacity_form_attention_mask_input_classifier() {
    // The mask slot (input 3) of a capacity-form `Attention` is a valid frozen-mask
    // leaf in both the non-causal and causal form (the frozen additive mask carries
    // the valid length on-device either way); a non-mask slot is not, and neither is
    // a masked `Attention` that lacks the KV cache bindings.
    let q = ValueId(0);
    let capacity = capacity_form_attention(0, q, q, q);
    assert!(is_capacity_form_attention_mask_input(&capacity, 3));
    assert!(!is_capacity_form_attention_mask_input(&capacity, 0));
    assert!(!is_capacity_form_attention_mask_input(&capacity, 4));

    let mut causal = capacity_form_attention(1, q, q, q);
    causal
        .attributes
        .insert("is_causal".into(), Attribute::Int(1));
    assert!(
        is_capacity_form_attention_mask_input(&causal, 3),
        "a frozen causal additive mask carries the valid length at its last-row frontier, \
         so the causal capacity-form Attention is a valid frozen-mask leaf"
    );

    // A masked `Attention` with no past KV bindings (inputs 4/5 absent) is not a
    // capacity-form leaf regardless of causality.
    let mask_only = Node::new(
        NodeId(2),
        "Attention",
        vec![Some(q), Some(q), Some(q), Some(q)],
        vec![q],
    );
    assert!(!is_capacity_form_attention_mask_input(&mask_only, 3));
}

// Build the standard additive causal-mask builder cone feeding a capacity-form
// `Attention` (the DeepSeek-V2-Lite / MLA topology). Returns the graph and the
// `attention_mask` binding value id.
fn v2lite_mask_builder_graph() -> (Graph, ValueId) {
    use onnx_runtime_ir::static_shape;
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sh = || static_shape([1]);

    let mask = graph.create_named_value("attention_mask", DataType::Int64, sh());
    graph.add_input(mask);
    let q = graph.create_named_value("q", DataType::Float32, sh());

    // Causal branch: attention_mask → CumSum → Unsqueeze → GreaterOrEqual.
    let cumsum = graph.create_named_value("cumsum", DataType::Int64, sh());
    graph.insert_node(Node::new(
        NodeId(0),
        "CumSum",
        vec![Some(mask)],
        vec![cumsum],
    ));
    let unsq0 = graph.create_named_value("unsq0", DataType::Int64, sh());
    graph.insert_node(Node::new(
        NodeId(1),
        "Unsqueeze",
        vec![Some(cumsum)],
        vec![unsq0],
    ));
    let ge = graph.create_named_value("ge", DataType::Bool, sh());
    graph.insert_node(Node::new(
        NodeId(2),
        "GreaterOrEqual",
        vec![Some(unsq0)],
        vec![ge],
    ));

    // Padding branch: attention_mask → Unsqueeze → Cast(bool).
    let unsq1 = graph.create_named_value("unsq1", DataType::Int64, sh());
    graph.insert_node(Node::new(
        NodeId(3),
        "Unsqueeze",
        vec![Some(mask)],
        vec![unsq1],
    ));
    let padbool = graph.create_named_value("padbool", DataType::Bool, sh());
    graph.insert_node(Node::new(
        NodeId(4),
        "Cast",
        vec![Some(unsq1)],
        vec![padbool],
    ));

    // And → Where(0/-inf) → Cast(fp16) → Unsqueeze → additive mask bias.
    let and = graph.create_named_value("and", DataType::Bool, sh());
    graph.insert_node(Node::new(
        NodeId(5),
        "And",
        vec![Some(ge), Some(padbool)],
        vec![and],
    ));
    let where_o = graph.create_named_value("where", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(6),
        "Where",
        vec![Some(and)],
        vec![where_o],
    ));
    let cast_o = graph.create_named_value("cast", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(7),
        "Cast",
        vec![Some(where_o)],
        vec![cast_o],
    ));
    let mask_bias = graph.create_named_value("mask_bias", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(8),
        "Unsqueeze",
        vec![Some(cast_o)],
        vec![mask_bias],
    ));

    // Benign physical-extent read.
    let shp = graph.create_named_value("shp", DataType::Int64, sh());
    graph.insert_node(Node::new(NodeId(9), "Shape", vec![Some(mask)], vec![shp]));

    // Two capacity-form Attention layers both consuming the shared mask bias.
    let attn0 = graph.create_named_value("attn0", DataType::Float32, sh());
    graph.insert_node(capacity_form_attention(10, q, mask_bias, attn0));
    let attn1 = graph.create_named_value("attn1", DataType::Float32, sh());
    graph.insert_node(capacity_form_attention(11, q, mask_bias, attn1));
    graph.add_output(attn0);
    graph.add_output(attn1);

    (graph, mask)
}

#[test]
fn vestigial_window_mask_builder_routes_to_padded_capacity() {
    // The full additive-mask builder (prefix-sensitive CumSum/Unsqueeze included)
    // terminating at capacity-form `Attention` inputs is padded-capacity-safe: the
    // frozen physical-width mask yields a byte-identical additive bias.
    let (graph, mask) = v2lite_mask_builder_graph();
    assert!(
        mask_binding_feeds_capacity_form_attention(&graph, mask),
        "vestigial-window additive-mask builder → capacity-form Attention must route padded-safe"
    );
}

#[test]
fn deepseek_shape_feeding_slice_window_keeps_logical_width() {
    use onnx_runtime_ir::static_shape;
    // DeepSeek-V2-Lite's HF-style causal-mask builder reads `Shape(attention_mask)`
    // and feeds it into `Sub`→`Slice` query-position arithmetic:
    //   Slice(CumSum(mask), start = Shape(mask) - q_seq, end = Shape(mask)).
    // `Shape` returns the *physical* padded width, so freezing the mask to
    // capacity selects query positions [max_len-q_seq .. max_len) instead of
    // [0 .. q_seq), producing a non-causal mask and incoherent decode. When the
    // `Shape` output is consumed like this the binding MUST keep exposing its
    // logical valid length (regression guard for the CUDA-EP coherence fix).
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sh = || static_shape([1]);

    let mask = graph.create_named_value("attention_mask", DataType::Int64, sh());
    graph.add_input(mask);
    let q = graph.create_named_value("q", DataType::Float32, sh());

    // Physical-extent read that is *consumed* by the window arithmetic.
    let shp = graph.create_named_value("shp", DataType::Int64, sh());
    graph.insert_node(Node::new(NodeId(0), "Shape", vec![Some(mask)], vec![shp]));
    let start = graph.create_named_value("start", DataType::Int64, sh());
    graph.insert_node(Node::new(NodeId(1), "Sub", vec![Some(shp)], vec![start]));

    // Causal branch: CumSum → Slice(start .. Shape) → Unsqueeze → GreaterOrEqual.
    let cumsum = graph.create_named_value("cumsum", DataType::Int64, sh());
    graph.insert_node(Node::new(
        NodeId(2),
        "CumSum",
        vec![Some(mask)],
        vec![cumsum],
    ));
    let sliced = graph.create_named_value("sliced", DataType::Int64, sh());
    graph.insert_node(Node::new(
        NodeId(3),
        "Slice",
        vec![Some(cumsum), Some(start), Some(shp)],
        vec![sliced],
    ));
    let unsq0 = graph.create_named_value("unsq0", DataType::Int64, sh());
    graph.insert_node(Node::new(
        NodeId(4),
        "Unsqueeze",
        vec![Some(sliced)],
        vec![unsq0],
    ));
    let ge = graph.create_named_value("ge", DataType::Bool, sh());
    graph.insert_node(Node::new(
        NodeId(5),
        "GreaterOrEqual",
        vec![Some(unsq0)],
        vec![ge],
    ));
    let where_o = graph.create_named_value("where", DataType::Float32, sh());
    graph.insert_node(Node::new(NodeId(6), "Where", vec![Some(ge)], vec![where_o]));
    let mask_bias = graph.create_named_value("mask_bias", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(7),
        "Unsqueeze",
        vec![Some(where_o)],
        vec![mask_bias],
    ));
    let attn = graph.create_named_value("attn", DataType::Float32, sh());
    graph.insert_node(capacity_form_attention(8, q, mask_bias, attn));
    graph.add_output(attn);

    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "DeepSeek Shape(mask)→Sub→Slice window arithmetic must keep logical width"
    );
}

/// Build `attention_mask → Cast → Unsqueeze → Expand(target) → Unsqueeze →
/// capacity-form Attention`, the HY-MT1.5 mask builder. `mask_derived_target`
/// picks whether the `Expand` target shape is built from `Shape(mask)` (the real
/// model) or from an unrelated value.
fn expand_mask_builder_graph(mask_derived_target: bool) -> (Graph, ValueId) {
    use onnx_runtime_ir::static_shape;
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sh = || static_shape([1]);

    let mask = graph.create_named_value("attention_mask", DataType::Int64, sh());
    graph.add_input(mask);
    let q = graph.create_named_value("q", DataType::Float32, sh());
    let other = graph.create_named_value("input_ids", DataType::Int64, sh());
    graph.add_input(other);

    let cast = graph.create_named_value("cast", DataType::Float32, sh());
    graph.insert_node(Node::new(NodeId(0), "Cast", vec![Some(mask)], vec![cast]));
    let unsq = graph.create_named_value("unsq", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(1),
        "Unsqueeze",
        vec![Some(cast)],
        vec![unsq],
    ));

    // Target axes: batch/query from `input_ids`, and the mask length axis from
    // either `Shape(mask)` (self-consistent) or `Shape(input_ids)` (not).
    let shape_other = graph.create_named_value("shape_other", DataType::Int64, sh());
    graph.insert_node(Node::new(
        NodeId(2),
        "Shape",
        vec![Some(other)],
        vec![shape_other],
    ));
    let len_axis = if mask_derived_target {
        let shape_mask = graph.create_named_value("shape_mask", DataType::Int64, sh());
        graph.insert_node(Node::new(
            NodeId(3),
            "Shape",
            vec![Some(mask)],
            vec![shape_mask],
        ));
        shape_mask
    } else {
        let shape_other2 = graph.create_named_value("shape_other2", DataType::Int64, sh());
        graph.insert_node(Node::new(
            NodeId(3),
            "Shape",
            vec![Some(other)],
            vec![shape_other2],
        ));
        shape_other2
    };
    let target = graph.create_named_value("target", DataType::Int64, sh());
    graph.insert_node(Node::new(
        NodeId(4),
        "Concat",
        vec![Some(shape_other), Some(len_axis)],
        vec![target],
    ));

    let expanded = graph.create_named_value("expanded", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(5),
        "Expand",
        vec![Some(unsq), Some(target)],
        vec![expanded],
    ));
    let mask_bias = graph.create_named_value("mask_bias", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(6),
        "Unsqueeze",
        vec![Some(expanded)],
        vec![mask_bias],
    ));
    let attn = graph.create_named_value("attn", DataType::Float32, sh());
    graph.insert_node(capacity_form_attention(7, q, mask_bias, attn));
    graph.add_output(attn);
    (graph, mask)
}

/// HY-MT1.5's builder broadcasts the mask with `Expand`, whose target shape
/// carries the mask length axis from `Shape(mask)` itself. Freezing is then a
/// uniform substitution — the mask is expanded to exactly its own frozen width —
/// so the binding is decode-freeze-safe and the decode step may capture.
///
/// Static freezing is still refused, because `Shape(mask)` is *consumed*: the
/// multi-token prefill window needs the logical length. Only the weaker
/// decode-time predicate holds.
#[test]
fn expand_with_mask_derived_target_is_decode_freeze_safe() {
    let (graph, mask) = expand_mask_builder_graph(true);
    assert!(
        mask_binding_feeds_additive_causal_builder(&graph, mask),
        "an Expand whose target length axis comes from Shape(mask) must be decode-freeze-safe"
    );
    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "a consumed Shape(mask) must still refuse *static* freezing"
    );
}

/// The same builder, but the `Expand` target sources the length axis from
/// somewhere the substitution does not reach. A mask frozen to `max_len` could
/// not broadcast against a target still carrying the logical length, so this
/// must be refused under both policies.
#[test]
fn expand_with_foreign_target_is_rejected() {
    let (graph, mask) = expand_mask_builder_graph(false);
    assert!(
        !mask_binding_feeds_additive_causal_builder(&graph, mask),
        "an Expand target not derived from Shape(mask) must not be freeze-safe"
    );
    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "an Expand target not derived from Shape(mask) must not be statically freezable"
    );
}

#[test]
fn minimal_cast_to_capacity_attention_routes_to_padded_capacity() {
    use onnx_runtime_ir::static_shape;
    // The tiny-fixture shape: attention_mask → Cast(bool) → capacity-form Attention.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sh = || static_shape([1]);
    let mask = graph.create_named_value("attention_mask", DataType::Int64, sh());
    graph.add_input(mask);
    let q = graph.create_named_value("q", DataType::Float32, sh());
    let bool_mask = graph.create_named_value("attn_mask_bool", DataType::Bool, sh());
    graph.insert_node(Node::new(
        NodeId(0),
        "Cast",
        vec![Some(mask)],
        vec![bool_mask],
    ));
    let attn = graph.create_named_value("attn", DataType::Float32, sh());
    graph.insert_node(capacity_form_attention(1, q, bool_mask, attn));
    graph.add_output(attn);
    assert!(mask_binding_feeds_capacity_form_attention(&graph, mask));
}

#[test]
fn glm_indexer_add_mask_keeps_logical_width() {
    use onnx_runtime_ir::static_shape;
    // GLM-5.2's indexer branch mixes the mask with a logical-width score through
    // Cast→Add. `Add` is not an additive-mask-builder op, so the cone is rejected
    // and the mask keeps exposing its logical valid length (regression guard).
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sh = || static_shape([1]);
    let mask = graph.create_named_value("attention_mask", DataType::Int64, sh());
    graph.add_input(mask);
    let score = graph.create_named_value("indexer_score", DataType::Float32, sh());
    let cast = graph.create_named_value("cast", DataType::Float32, sh());
    graph.insert_node(Node::new(NodeId(0), "Cast", vec![Some(mask)], vec![cast]));
    let add = graph.create_named_value("add", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(1),
        "Add",
        vec![Some(cast), Some(score)],
        vec![add],
    ));
    graph.add_output(add);
    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "GLM-5.2 indexer Add mask must NOT be classified padded-safe"
    );
}

/// Build the minimized DeepSeek-V4-shaped decomposed-attention skeleton: a
/// `past_key` graph input grown by a KV-cache-growth `Concat` into a
/// `present_key` graph output, then read as `MatMul`'s *rhs* to produce a score
/// (matching ONNX's trailing-axis semantics, exactly as `derives_from_kv_cache_
/// growth`'s `MatMul` rule expects). Returns `(graph, mask, score)` so each test
/// below only has to wire the mask-combining tail differently.
fn decomposed_kv_growth_score_graph() -> (Graph, ValueId, ValueId) {
    use onnx_runtime_ir::static_shape;
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sh = || static_shape([1]);
    let mask = graph.create_named_value("attention_mask", DataType::Float32, sh());
    graph.add_input(mask);
    let past_key = graph.create_named_value("past_key", DataType::Float32, sh());
    graph.add_input(past_key);
    let current_key = graph.create_named_value("current_key", DataType::Float32, sh());
    let present_key = graph.create_named_value("present_key", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(0),
        "Concat",
        vec![Some(past_key), Some(current_key)],
        vec![present_key],
    ));
    graph.add_output(present_key);
    let q = graph.create_named_value("q", DataType::Float32, sh());
    let score = graph.create_named_value("score", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(1),
        "MatMul",
        vec![Some(q), Some(present_key)],
        vec![score],
    ));
    (graph, mask, score)
}

/// The safe case the generalized classifier exists for: DeepSeek-V4's
/// decomposed `score = Q @ present_keyᵀ`; `Add(score, mask)`; last-axis
/// `Softmax` -- the decomposed-attention analogue of a fused `Attention`'s
/// internal padding neutralization. `score`'s length axis structurally derives
/// from the KV-cache-growth `Concat` above, so the mask may be frozen to
/// physical capacity: this must classify as capacity-safe under BOTH policies
/// (no `Shape(mask)` is consumed here, so static and decode-freeze predicates
/// agree).
#[test]
fn decomposed_add_over_kv_cache_growth_score_then_softmax_is_capacity_safe() {
    let (mut graph, mask, score) = decomposed_kv_growth_score_graph();
    let add = graph.create_named_value("add", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Add",
        vec![Some(score), Some(mask)],
        vec![add],
    ));
    let softmax = graph.create_named_value("softmax", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(3),
        "Softmax",
        vec![Some(add)],
        vec![softmax],
    ));
    graph.add_output(softmax);
    assert!(
        mask_binding_feeds_capacity_form_attention(&graph, mask),
        "a decomposed Add(score, mask) whose score derives from a KV-cache-growth \
         Concat, normalized by a last-axis Softmax, must be capacity-safe"
    );
    assert!(
        mask_binding_feeds_additive_causal_builder(&graph, mask),
        "the same graph must also hold under the weaker decode-freeze-safe policy"
    );
}

/// Opset-gating for the `Softmax` "axis absent" default: opset <= 12 defaults
/// `axis` to `1` with "coerce to 2D" semantics (merging every dim from `axis`
/// onward into one normalization group), not `-1`/last-axis-only. The exact
/// same graph as the safe case above, but pinned to opset 12 with no explicit
/// `axis` attribute, must therefore be rejected: this call site has no tensor
/// rank to confirm the coerced group is only the length axis.
#[test]
fn decomposed_add_then_softmax_with_absent_axis_at_old_opset_is_rejected() {
    let (mut graph, mask, score) = decomposed_kv_growth_score_graph();
    graph.opset_imports.insert(String::new(), 12);
    let add = graph.create_named_value("add", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Add",
        vec![Some(score), Some(mask)],
        vec![add],
    ));
    let softmax = graph.create_named_value("softmax", DataType::Float32, static_shape([1]));
    // No `axis` attribute set -- opset 12's implicit default is `1`, not `-1`.
    graph.insert_node(Node::new(
        NodeId(3),
        "Softmax",
        vec![Some(add)],
        vec![softmax],
    ));
    graph.add_output(softmax);
    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "a Softmax with no explicit axis attribute at opset <= 12 must NOT be treated as \
         last-axis-only neutralization (its default coerces to 2D at axis=1 instead)"
    );
}

/// Unsafe counterexample: "external width" — the `Add`'s other operand does
/// NOT derive from the KV-cache-growth `Concat` (it is an unrelated value with
/// no producer at all), even though the tail still normalizes with a last-axis
/// `Softmax`. Freezing the mask here would compare `max_len` against a value
/// still at its own, unrelated logical length, so this must be rejected.
#[test]
fn decomposed_add_over_foreign_width_score_is_rejected() {
    let (mut graph, mask, _kv_growth_score) = decomposed_kv_growth_score_graph();
    // A second, unrelated score with no KV-cache-growth provenance at all.
    let foreign_score =
        graph.create_named_value("foreign_score", DataType::Float32, static_shape([1]));
    let add = graph.create_named_value("add", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Add",
        vec![Some(foreign_score), Some(mask)],
        vec![add],
    ));
    let softmax = graph.create_named_value("softmax", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(3),
        "Softmax",
        vec![Some(add)],
        vec![softmax],
    ));
    graph.add_output(softmax);
    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "an Add whose other operand does not derive from a KV-cache-growth Concat \
         must NOT be classified capacity-safe, even with a trailing last-axis Softmax"
    );
}

/// Unsafe counterexample: "non-neutralized padded lanes" — the `Add`'s score
/// does correctly derive from the KV-cache-growth `Concat`, but the result
/// reaches a `MatMul` directly instead of a neutralizing `Softmax`. The
/// substitution's padded lanes are never forced to a neutral value before a
/// non-padding-aware consumer, so this must be rejected even though the `Add`
/// step itself would have been sound in isolation.
#[test]
fn decomposed_add_without_softmax_neutralization_is_rejected() {
    let (mut graph, mask, score) = decomposed_kv_growth_score_graph();
    let add = graph.create_named_value("add", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Add",
        vec![Some(score), Some(mask)],
        vec![add],
    ));
    // No Softmax: the additive bias feeds straight into another MatMul, so the
    // padded lanes are never neutralized before a non-padding-aware consumer.
    let w = graph.create_named_value("w", DataType::Float32, static_shape([1]));
    let out = graph.create_named_value("out", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(3),
        "MatMul",
        vec![Some(add), Some(w)],
        vec![out],
    ));
    graph.add_output(out);
    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "an Add(score, mask) that reaches a MatMul instead of a neutralizing Softmax \
         must NOT be classified capacity-safe"
    );
}

/// Unsafe counterexample: "wrong axis" — the same safe `Add`→`Softmax` shape as
/// [`decomposed_add_over_kv_cache_growth_score_then_softmax_is_capacity_safe`],
/// but `Softmax`'s `axis` attribute normalizes over axis `0` instead of the
/// last axis. Padding neutralization only happens along the length axis, so a
/// `Softmax` over any other axis must not be treated as a neutralizing sink.
#[test]
fn decomposed_add_then_softmax_over_wrong_axis_is_rejected() {
    let (mut graph, mask, score) = decomposed_kv_growth_score_graph();
    let add = graph.create_named_value("add", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Add",
        vec![Some(score), Some(mask)],
        vec![add],
    ));
    let softmax = graph.create_named_value("softmax", DataType::Float32, static_shape([1]));
    let mut softmax_node = Node::new(NodeId(3), "Softmax", vec![Some(add)], vec![softmax]);
    softmax_node
        .attributes
        .insert("axis".into(), Attribute::Int(0));
    graph.insert_node(softmax_node);
    graph.add_output(softmax);
    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "a Softmax normalizing over axis 0 (not the last axis) must NOT neutralize \
         the mask's padded lanes, so the binding must not be classified capacity-safe"
    );
}

/// Unsafe counterexample: "mixed consumer" via `Concat` — DeepSeek-V4's real
/// attention-sink pattern appends a fixed-size bias column onto the masked
/// score via `Concat` before `Softmax`. This poisons that pattern by making the
/// *other* `Concat` operand ALSO derive from the KV-cache-growth `Concat`
/// (instead of a structurally-independent initializer-derived bias), so the
/// two operands would not stay the same size once the mask is frozen — this
/// must be rejected.
#[test]
fn decomposed_concat_with_kv_growth_derived_other_operand_is_rejected() {
    let (mut graph, mask, score) = decomposed_kv_growth_score_graph();
    let add = graph.create_named_value("add", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Add",
        vec![Some(score), Some(mask)],
        vec![add],
    ));
    // The "sink bias" operand is poisoned: instead of an independent
    // initializer-derived value, it is itself read straight from `score`
    // (via a plain `Identity`, still reachable by `derives_from_kv_cache_
    // growth`'s unconditional-descent op set), so it is NOT structurally
    // independent of the mask-derived operand.
    let poisoned_bias =
        graph.create_named_value("poisoned_bias", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(4),
        "Cast",
        vec![Some(score)],
        vec![poisoned_bias],
    ));
    let concat = graph.create_named_value("concat", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(3),
        "Concat",
        vec![Some(add), Some(poisoned_bias)],
        vec![concat],
    ));
    let softmax = graph.create_named_value("softmax", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(5),
        "Softmax",
        vec![Some(concat)],
        vec![softmax],
    ));
    graph.add_output(softmax);
    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "a Concat whose other operand also derives from a KV-cache-growth Concat \
         must NOT be classified capacity-safe: the two operands would not stay the \
         same size once the mask is frozen"
    );
}

/// Safe counterpart to the previous test: the attention-sink `Concat`'s other
/// operand is a genuinely independent value (no producer at all, standing in
/// for an initializer-derived learned bias, as DeepSeek-V4's real `attn_sink`
/// is). This must remain capacity-safe -- proving the generalized `Concat` rule
/// still accepts the legitimate pattern it was designed for, not just reject
/// the poisoned one above.
#[test]
fn decomposed_concat_with_independent_sink_bias_is_capacity_safe() {
    let (mut graph, mask, score) = decomposed_kv_growth_score_graph();
    let add = graph.create_named_value("add", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Add",
        vec![Some(score), Some(mask)],
        vec![add],
    ));
    // A genuinely independent sink-bias operand: no producer node at all,
    // standing in for a value traced back to a graph initializer.
    let sink_bias = graph.create_named_value("attn_sink", DataType::Float32, static_shape([1]));
    let concat = graph.create_named_value("concat", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(3),
        "Concat",
        vec![Some(add), Some(sink_bias)],
        vec![concat],
    ));
    let softmax = graph.create_named_value("softmax", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(4),
        "Softmax",
        vec![Some(concat)],
        vec![softmax],
    ));
    graph.add_output(softmax);
    assert!(
        mask_binding_feeds_capacity_form_attention(&graph, mask),
        "a Concat whose other operand is structurally independent of any \
         KV-cache-growth Concat (e.g. an initializer-derived sink bias) must \
         remain capacity-safe"
    );
}

/// Builds `decomposed_kv_growth_score_graph`'s K-role cone (`Add(score,
/// mask)` -> `Softmax`) and extends it with a V-role: `Softmax` output ->
/// `MatMul(probs, present_value)`, where `present_value` is a *second*,
/// independent KV-cache-growth `Concat` (a distinct `past_value`/
/// `current_value` pair from the K-role's `past_key`/`current_key`). Returns
/// the graph, the mask, the K-role concat's `NodeId`, and the V-role concat's
/// `NodeId`, so tests can assert exactly which node identities the eligibility
/// scan reports.
fn decomposed_kv_growth_score_and_value_graph() -> (Graph, ValueId, NodeId, NodeId) {
    let (mut graph, mask, score) = decomposed_kv_growth_score_graph();
    let key_concat_id = NodeId(0);

    let add = graph.create_named_value("add", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Add",
        vec![Some(score), Some(mask)],
        vec![add],
    ));
    let softmax = graph.create_named_value("softmax", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(3),
        "Softmax",
        vec![Some(add)],
        vec![softmax],
    ));

    let past_value = graph.create_named_value("past_value", DataType::Float32, static_shape([1]));
    graph.add_input(past_value);
    let current_value =
        graph.create_named_value("current_value", DataType::Float32, static_shape([1]));
    let present_value =
        graph.create_named_value("present_value", DataType::Float32, static_shape([1]));
    let value_concat_id = NodeId(4);
    graph.insert_node(Node::new(
        value_concat_id,
        "Concat",
        vec![Some(past_value), Some(current_value)],
        vec![present_value],
    ));
    graph.add_output(present_value);

    let out = graph.create_named_value("out", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(5),
        "MatMul",
        vec![Some(softmax), Some(present_value)],
        vec![out],
    ));
    graph.add_output(out);

    (graph, mask, key_concat_id, value_concat_id)
}

/// The S3 capacity-emission entry point: [`kv_capacity_write_eligible_concats`]
/// must report BOTH the K-role concat (reached via the `Add` "score-role" rule
/// already proven by #1838's mask-cone walk) and the V-role concat (reached
/// via the new forward `Softmax` -> `MatMul` walk) for the same decomposed
/// attention cone, not just one or the other.
#[test]
fn kv_capacity_eligible_concats_includes_both_score_and_value_role() {
    let (graph, _mask, key_concat_id, value_concat_id) =
        decomposed_kv_growth_score_and_value_graph();
    let eligible = kv_capacity_write_eligible_concats(&graph);
    assert!(
        eligible.contains(&key_concat_id),
        "the K-role (score) KV-cache-growth Concat must be eligible: {eligible:?}"
    );
    assert!(
        eligible.contains(&value_concat_id),
        "the V-role (value) KV-cache-growth Concat must be eligible: {eligible:?}"
    );
    assert_eq!(
        eligible.len(),
        2,
        "exactly the two KV-cache-growth Concats in this cone should be eligible, got {eligible:?}"
    );
}

/// Counterexample: a `MatMul` reachable from a proven-safe `Softmax` sink, but
/// whose *other* operand does NOT derive from any KV-cache-growth `Concat`
/// (e.g. a static weight/value with no producer at all). The value-role rule
/// must not spuriously invent eligibility here — the eligible set stays
/// exactly the K-role concat.
#[test]
fn kv_capacity_eligible_concats_excludes_matmul_with_foreign_value_operand() {
    let (mut graph, _mask, score) = decomposed_kv_growth_score_graph();
    let mask = graph
        .inputs
        .first()
        .copied()
        .expect("mask is graph input 0");
    let add = graph.create_named_value("add", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Add",
        vec![Some(score), Some(mask)],
        vec![add],
    ));
    let softmax = graph.create_named_value("softmax", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(3),
        "Softmax",
        vec![Some(add)],
        vec![softmax],
    ));
    // A "V" operand with no KV-cache-growth provenance at all.
    let foreign_value =
        graph.create_named_value("foreign_value", DataType::Float32, static_shape([1]));
    let out = graph.create_named_value("out", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(4),
        "MatMul",
        vec![Some(softmax), Some(foreign_value)],
        vec![out],
    ));
    graph.add_output(out);

    let eligible = kv_capacity_write_eligible_concats(&graph);
    assert_eq!(
        eligible,
        std::iter::once(NodeId(0)).collect(),
        "only the K-role concat should be eligible when the MatMul's other operand \
         has no KV-cache-growth provenance: {eligible:?}"
    );
}

/// A mask cone that fails the underlying capacity-safety proof entirely (the
/// "external width" counterexample) must report zero eligible concats, even
/// though a KV-cache-growth `Concat` exists in the graph: eligibility must
/// never be granted independent of the proof it is a byproduct of.
#[test]
fn kv_capacity_eligible_concats_is_empty_when_mask_cone_is_rejected() {
    let (mut graph, mask, _kv_growth_score) = decomposed_kv_growth_score_graph();
    let foreign_score =
        graph.create_named_value("foreign_score", DataType::Float32, static_shape([1]));
    let add = graph.create_named_value("add", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Add",
        vec![Some(foreign_score), Some(mask)],
        vec![add],
    ));
    let softmax = graph.create_named_value("softmax", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(3),
        "Softmax",
        vec![Some(add)],
        vec![softmax],
    ));
    graph.add_output(softmax);

    let eligible = kv_capacity_write_eligible_concats(&graph);
    assert!(
        eligible.is_empty(),
        "a rejected mask cone must contribute zero eligible concats: {eligible:?}"
    );
}

/// A minimal test-double `ExecutionProvider` for exercising
/// [`rewrite_kv_capacity_appends`] in isolation: it wraps a real
/// `CpuExecutionProvider` for everything (so `supports_op` still gives
/// truthful answers for ordinary ops, `Graph::validate()`-adjacent behavior
/// stays realistic, etc.), and overrides only the one decision this pass
/// actually asks of an EP — whether `pkg.nxrt::KvCacheCapacityAppend` is
/// supported — via a constructor flag. This directly models the real
/// contract: CPU (accept = false) leaves every candidate Concat untouched;
/// CUDA (accept = true) is the only EP that has ever registered the kernel.
struct KvCapacityAppendTestEp {
    cpu: CpuExecutionProvider,
    accept_capacity_append: bool,
}

impl KvCapacityAppendTestEp {
    fn new(accept_capacity_append: bool) -> Self {
        let mut cpu = CpuExecutionProvider::new();
        cpu.initialize(&EpConfig::default()).unwrap();
        Self {
            cpu,
            accept_capacity_append,
        }
    }
}

impl ExecutionProvider for KvCapacityAppendTestEp {
    fn consume_route_residency_at_boundary(&self) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "kv_capacity_append_test_ep"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn device_id(&self) -> onnx_runtime_ir::DeviceId {
        onnx_runtime_ir::DeviceId::cpu()
    }

    fn initialize(&mut self, _config: &EpConfig) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> KernelMatch {
        if op.domain == "pkg.nxrt" && op.op_type == "KvCacheCapacityAppend" {
            return if self.accept_capacity_append {
                KernelMatch::Supported {
                    cost: Cost::ZERO,
                    required_input_layouts: None,
                    output_layouts: vec![TensorLayout::contiguous(); op.outputs.len()],
                }
            } else {
                KernelMatch::unsupported(
                    "test EP does not register pkg.nxrt::KvCacheCapacityAppend",
                )
            };
        }
        self.cpu
            .supports_op(op, opset, shapes, input_dtypes, layouts)
    }

    fn get_kernel(
        &self,
        op: &Node,
        shapes: &[Vec<usize>],
        opset: u64,
    ) -> onnx_runtime_ep_api::Result<Box<dyn Kernel>> {
        self.cpu.get_kernel(op, shapes, opset)
    }

    fn allocate(&self, size: usize, alignment: usize) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
        self.cpu.allocate(size, alignment)
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.deallocate(buffer)
    }

    fn copy(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.copy(src, dst, size)
    }

    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<Fence> {
        self.cpu.copy_async(src, dst, size)
    }

    fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.sync()
    }
}

/// Rewrites a KV-cache-growth `Concat` node's `past`/`current`/`present`
/// operand shapes to a realistic rank-4 `[batch, heads, capacity/1,
/// head_dim]` physical layout and stamps its mandatory `axis` attribute to
/// `2` (the sequence axis) — the shape/axis precondition
/// [`rewrite_kv_capacity_appends`] now requires before it will rewrite a
/// candidate `Concat` (see that function's axis-normalization comment). The
/// rest of the cone (`mask`/`score`/`softmax`/etc.) is deliberately left at
/// its placeholder rank-1 `[1]` shape: the classifier's proof and the pass's
/// other preconditions are purely topological and never inspect those
/// shapes, so only the two values this new gate actually reads need to carry
/// realistic ranks.
fn shape_kv_growth_concat_as_rank4_axis2(graph: &mut Graph, concat_id: NodeId) {
    let node = graph.node(concat_id);
    let axis = 2;
    let past = node.inputs[0].expect("KV-growth Concat must have a past operand");
    let current = node.inputs[1].expect("KV-growth Concat must have a current operand");
    let present = node.outputs[0];
    graph.value_mut(past).shape = static_shape([1, 2, 4, 8]);
    graph.value_mut(current).shape = static_shape([1, 2, 1, 8]);
    graph.value_mut(present).shape = static_shape([1, 2, 4, 8]);
    graph
        .node_mut(concat_id)
        .attributes
        .insert("axis".into(), Attribute::Int(axis));
}

/// Adds a `position_ids` graph input to
/// [`decomposed_kv_growth_score_and_value_graph`]'s cone — the load-bearing
/// signal [`rewrite_kv_capacity_appends`] requires before it will even run
/// the classifier — and shapes both KV-growth `Concat`s (see
/// [`shape_kv_growth_concat_as_rank4_axis2`]) to satisfy the rewrite's
/// rank-4/axis-2 precondition. Returns the same tuple plus the new value id.
fn decomposed_kv_growth_cone_with_position_ids() -> (Graph, NodeId, NodeId, ValueId) {
    let (mut graph, _mask, key_concat_id, value_concat_id) =
        decomposed_kv_growth_score_and_value_graph();
    shape_kv_growth_concat_as_rank4_axis2(&mut graph, key_concat_id);
    shape_kv_growth_concat_as_rank4_axis2(&mut graph, value_concat_id);
    let position_ids =
        graph.create_named_value("position_ids", DataType::Int64, static_shape([1, 1]));
    graph.add_input(position_ids);
    (graph, key_concat_id, value_concat_id, position_ids)
}

/// The core positive case: given a cone with both a K-role and a V-role
/// KV-cache-growth `Concat` (both proven eligible by the classifier), a
/// `position_ids` input, and an EP that accepts the new op, both Concats must
/// be rewritten in place into `pkg.nxrt::KvCacheCapacityAppend` — same
/// `NodeId`, inputs `[past, current, position_ids]`, output unchanged — and
/// the `pkg.nxrt` opset-import must be recorded.
#[test]
fn rewrite_kv_capacity_appends_rewrites_every_eligible_concat_when_ep_supports_the_op() {
    let (mut graph, key_concat_id, value_concat_id, position_ids) =
        decomposed_kv_growth_cone_with_position_ids();
    let ep = KvCapacityAppendTestEp::new(true);

    let rewritten = rewrite_kv_capacity_appends(&mut graph, &ep);

    assert!(rewritten, "an eligible cone with a supporting EP must fire");
    for (id, expected_past_name, expected_current_name) in [
        (key_concat_id, "past_key", "current_key"),
        (value_concat_id, "past_value", "current_value"),
    ] {
        let node = graph.node(id);
        assert_eq!(node.domain, "pkg.nxrt", "node {id:?} must be re-domained");
        assert_eq!(
            node.op_type, "KvCacheCapacityAppend",
            "node {id:?} must be re-typed"
        );
        assert_eq!(node.inputs.len(), 3, "node {id:?} must gain position_ids");
        let past = node.inputs[0].expect("past operand must survive");
        let current = node.inputs[1].expect("current operand must survive");
        assert_eq!(
            graph.value(past).name.as_deref(),
            Some(expected_past_name),
            "node {id:?}'s past operand must be preserved exactly"
        );
        assert_eq!(
            graph.value(current).name.as_deref(),
            Some(expected_current_name),
            "node {id:?}'s current operand must be preserved exactly"
        );
        assert_eq!(
            node.inputs[2],
            Some(position_ids),
            "node {id:?}'s third input must be the graph's position_ids value"
        );
        assert_eq!(node.outputs.len(), 1);
    }
    assert_eq!(
        graph.opset_imports.get("pkg.nxrt"),
        Some(&1),
        "the pkg.nxrt opset-import must be recorded once a rewrite fires"
    );
    assert!(
        graph.validate().is_ok(),
        "the rewritten graph must remain structurally valid: {:?}",
        graph.validate().err()
    );
}

/// The eligibility classifier proves the KV-growth `Concat`'s output
/// *provenance* is safe — it never inspects which axis the concatenation
/// grows on. A `Concat` that structurally matched but grows on an axis other
/// than 2 (of a rank-4 `[batch, heads, seq, head_dim]` layout) is not one
/// `pkg.nxrt::KvCacheCapacityAppend`'s kernel understands (it hardcodes axis
/// 2), so the rewrite must decline it and leave the original, axis-generic
/// `Concat` completely untouched — never silently write into the wrong
/// physical dimension. Covers both an explicit non-2 axis and a negative
/// axis that normalizes away from 2.
#[test]
fn rewrite_kv_capacity_appends_is_a_no_op_when_concat_axis_is_not_the_sequence_axis() {
    for wrong_axis in [1_i64, -1_i64] {
        let (mut graph, key_concat_id, value_concat_id, _position_ids) =
            decomposed_kv_growth_cone_with_position_ids();
        graph
            .node_mut(key_concat_id)
            .attributes
            .insert("axis".into(), Attribute::Int(wrong_axis));
        let ep = KvCapacityAppendTestEp::new(true);

        // The pass legitimately still fires overall (the unrelated,
        // correctly-shaped value-role Concat below is rewritten), so this
        // test asserts on the per-node outcome, not the pass's aggregate
        // "did anything change" return value.
        rewrite_kv_capacity_appends(&mut graph, &ep);

        assert_eq!(
            graph.node(key_concat_id).op_type,
            "Concat",
            "the wrong-axis Concat must remain untouched for axis={wrong_axis}"
        );
        // The unrelated, correctly-axis-2 value-role Concat in the same graph
        // is unaffected by the key-role node's axis being wrong — this is a
        // per-node structural gate, not an all-or-nothing graph veto.
        assert_eq!(
            graph.node(value_concat_id).op_type,
            "KvCacheCapacityAppend",
            "an unrelated correctly-shaped Concat must still be rewritten for axis={wrong_axis}"
        );
    }
}

/// The same axis precondition also declines a `Concat` whose `past`/`present`
/// operands are not rank 4 at all (e.g. a rank-3 layout with no separate
/// heads axis) — there is no well-defined "axis 2 of a 4-D physical layout"
/// to even ask about, so the rewrite must leave it as a plain `Concat`
/// regardless of what its `axis` attribute says.
#[test]
fn rewrite_kv_capacity_appends_is_a_no_op_when_past_is_not_rank4() {
    let (mut graph, key_concat_id, value_concat_id, _position_ids) =
        decomposed_kv_growth_cone_with_position_ids();
    let past = graph.node(key_concat_id).inputs[0].unwrap();
    let present = graph.node(key_concat_id).outputs[0];
    graph.value_mut(past).shape = static_shape([2, 4, 8]);
    graph.value_mut(present).shape = static_shape([2, 4, 8]);
    let ep = KvCapacityAppendTestEp::new(true);

    // As above: the sibling value-role Concat is still rank-4/axis-2 and is
    // legitimately rewritten, so only the per-node outcome is asserted.
    rewrite_kv_capacity_appends(&mut graph, &ep);

    assert_eq!(
        graph.node(key_concat_id).op_type,
        "Concat",
        "a rank-3 past operand must never be rewritten regardless of its axis attribute"
    );
    assert_eq!(
        graph.node(value_concat_id).op_type,
        "KvCacheCapacityAppend",
        "an unrelated correctly rank-4 value-role Concat must still be rewritten"
    );
}

/// The capability gate: the exact same eligible cone, but the EP does not
/// support the new op (modeling every EP except CUDA today, e.g. CPU). Every
/// candidate Concat must be left completely untouched, and no `pkg.nxrt`
/// opset-import may be introduced — this is what makes the pass a
/// capability-gated rewrite instead of a model/op allowlist that assumes the
/// kernel always exists.
#[test]
fn rewrite_kv_capacity_appends_is_a_no_op_when_ep_does_not_support_the_op() {
    let (mut graph, key_concat_id, value_concat_id, _position_ids) =
        decomposed_kv_growth_cone_with_position_ids();
    let ep = KvCapacityAppendTestEp::new(false);

    let rewritten = rewrite_kv_capacity_appends(&mut graph, &ep);

    assert!(
        !rewritten,
        "a non-supporting EP must leave the pass a no-op"
    );
    for id in [key_concat_id, value_concat_id] {
        let node = graph.node(id);
        assert!(
            node.is_default_domain(),
            "node {id:?} must remain default-domain when the EP has no kernel"
        );
        assert_eq!(
            node.op_type, "Concat",
            "node {id:?} must remain a plain Concat when the EP has no kernel"
        );
    }
    assert!(
        !graph.opset_imports.contains_key("pkg.nxrt"),
        "no pkg.nxrt opset-import may be introduced when nothing was rewritten"
    );
}

/// A supporting EP does not matter if the graph has no `position_ids` input
/// at all (e.g. an encoder-style export, or any graph that never threads
/// position ids): the rewrite has nothing to bind its third operand to, so it
/// must bail before even consulting the classifier, leaving every Concat
/// untouched.
#[test]
fn rewrite_kv_capacity_appends_is_a_no_op_without_a_position_ids_input() {
    let (mut graph, _mask, key_concat_id, value_concat_id) =
        decomposed_kv_growth_score_and_value_graph();
    assert!(
        graph
            .inputs
            .iter()
            .all(|&v| graph.value(v).name.as_deref() != Some("position_ids")),
        "precondition: this graph must have no position_ids input"
    );
    let ep = KvCapacityAppendTestEp::new(true);

    let rewritten = rewrite_kv_capacity_appends(&mut graph, &ep);

    assert!(
        !rewritten,
        "a graph with no position_ids input must never be rewritten, \
         regardless of EP support"
    );
    for id in [key_concat_id, value_concat_id] {
        assert_eq!(graph.node(id).op_type, "Concat");
    }
}

/// A supporting EP and a present `position_ids` input do not matter if the
/// classifier itself rejects the cone (the "external width" counterexample):
/// eligibility must always flow through the classifier's own proof, never be
/// granted merely because a `Concat` and `position_ids` happen to coexist in
/// the graph.
#[test]
fn rewrite_kv_capacity_appends_is_a_no_op_when_classifier_rejects_the_cone() {
    let (mut graph, mask, _kv_growth_score) = decomposed_kv_growth_score_graph();
    let foreign_score =
        graph.create_named_value("foreign_score", DataType::Float32, static_shape([1]));
    let add = graph.create_named_value("add", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(2),
        "Add",
        vec![Some(foreign_score), Some(mask)],
        vec![add],
    ));
    let softmax = graph.create_named_value("softmax", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(3),
        "Softmax",
        vec![Some(add)],
        vec![softmax],
    ));
    graph.add_output(softmax);
    let position_ids = graph.create_named_value("position_ids", DataType::Int64, static_shape([1]));
    graph.add_input(position_ids);
    assert!(
        kv_capacity_write_eligible_concats(&graph).is_empty(),
        "precondition: the classifier must reject this cone"
    );
    let ep = KvCapacityAppendTestEp::new(true);

    let rewritten = rewrite_kv_capacity_appends(&mut graph, &ep);

    assert!(
        !rewritten,
        "a classifier-rejected cone must never be rewritten, even with a \
         supporting EP and a position_ids input present"
    );
    assert_eq!(graph.node(NodeId(0)).op_type, "Concat");
}

#[test]
fn mask_builder_without_capacity_attention_is_rejected() {
    use onnx_runtime_ir::static_shape;
    // A mask cone that reaches no `Attention` at all (only a Cast to a graph
    // output) must not be classified padded-safe: there is no capacity-form
    // consumer, so the mask keeps exposing its logical valid length.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sh = || static_shape([1]);
    let mask = graph.create_named_value("attention_mask", DataType::Int64, sh());
    graph.add_input(mask);
    let cast = graph.create_named_value("cast", DataType::Float32, sh());
    graph.insert_node(Node::new(NodeId(0), "Cast", vec![Some(mask)], vec![cast]));
    graph.add_output(cast);
    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "a mask cone reaching no capacity-form Attention is not padded-safe"
    );
}

#[test]
fn mask_feeding_only_shape_is_not_padded_capacity_via_topology() {
    use onnx_runtime_ir::static_shape;
    // A mask that reaches no capacity-form Attention (only Shape) is not blessed by
    // the topology path: `reached_attention` stays false.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sh = || static_shape([1]);
    let mask = graph.create_named_value("attention_mask", DataType::Int64, sh());
    graph.add_input(mask);
    let shp = graph.create_named_value("shp", DataType::Int64, sh());
    graph.insert_node(Node::new(NodeId(0), "Shape", vec![Some(mask)], vec![shp]));
    graph.add_output(shp);
    assert!(!mask_binding_feeds_capacity_form_attention(&graph, mask));
}

#[test]
fn mask_feeding_non_builder_consumer_is_rejected() {
    use onnx_runtime_ir::static_shape;
    // A mask consumed by an arbitrary op (here MatMul) that is neither a shape read,
    // a builder op, nor a capacity-form Attention input disqualifies the binding.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sh = || static_shape([1]);
    let mask = graph.create_named_value("attention_mask", DataType::Float32, sh());
    graph.add_input(mask);
    let w = graph.create_named_value("w", DataType::Float32, sh());
    let mm = graph.create_named_value("mm", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(mask), Some(w)],
        vec![mm],
    ));
    graph.add_output(mm);
    assert!(!mask_binding_feeds_capacity_form_attention(&graph, mask));
}

#[test]
fn mask_builder_to_attention_without_past_kv_is_rejected() {
    use onnx_runtime_ir::static_shape;
    // A masked, non-causal default-domain `Attention` with only q/k/v/mask (no
    // past_key/past_value KV binding at inputs 4/5) is NOT a capacity-form leaf:
    // the CUDA `Attention` kernel's fixed-capacity append contract requires both
    // past caches. The cone must reach no valid capacity-form Attention, so the
    // mask keeps exposing its logical valid length.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sh = || static_shape([1]);
    let mask = graph.create_named_value("attention_mask", DataType::Int64, sh());
    graph.add_input(mask);
    let q = graph.create_named_value("q", DataType::Float32, sh());
    let cast = graph.create_named_value("cast", DataType::Float32, sh());
    graph.insert_node(Node::new(NodeId(0), "Cast", vec![Some(mask)], vec![cast]));
    // Attention with q/k/v/mask only — no KV cache at inputs 4/5.
    let attn = graph.create_named_value("attn", DataType::Float32, sh());
    graph.insert_node(Node::new(
        NodeId(1),
        "Attention",
        vec![Some(q), Some(q), Some(q), Some(cast)],
        vec![attn],
    ));
    graph.add_output(attn);
    // Sanity: the leaf classifier itself rejects the KV-less Attention.
    let kvless = graph.node(NodeId(1));
    assert!(
        !is_capacity_form_attention_mask_input(kvless, 3),
        "an Attention without past_key/past_value is not a capacity-form leaf"
    );
    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "a mask cone reaching only a KV-less Attention must not be padded-safe"
    );
}

// Build the DeepSeek-V2-Lite additive causal-mask builder cone with SYMBOLIC
// shapes (the MLA / HF-causal-mask topology), so the mask/bias length symbol is
// visible to the capture classifier. Returns the graph, the `attention_mask`
// binding, the mask/bias length symbol, and the capacity-form `Attention` node.
fn v2lite_symbolic_mask_graph() -> (Graph, ValueId, SymbolId, Node) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);
    // The mask / total-sequence length axis (the symbol whose non-pinning keeps
    // the whole cone + Attention eager). Lives on the mask input and its cone,
    // NOT on any KV slot.
    let seq = graph.create_symbol(None);
    let m2 = |b, s| vec![sym(b), sym(s)];
    let m3 = |b, s| vec![sym(b), st(1), sym(s)];

    let mask = graph.create_named_value("attention_mask", DataType::Int64, m2(batch, seq));
    graph.add_input(mask);
    let cumsum = graph.create_named_value("cumsum", DataType::Int64, m2(batch, seq));
    graph.insert_node(Node::new(
        NodeId(0),
        "CumSum",
        vec![Some(mask)],
        vec![cumsum],
    ));
    let unsq0 = graph.create_named_value("unsq0", DataType::Int64, m3(batch, seq));
    graph.insert_node(Node::new(
        NodeId(1),
        "Unsqueeze",
        vec![Some(cumsum)],
        vec![unsq0],
    ));
    let ge = graph.create_named_value("ge", DataType::Bool, m3(batch, seq));
    graph.insert_node(Node::new(
        NodeId(2),
        "GreaterOrEqual",
        vec![Some(unsq0)],
        vec![ge],
    ));
    let unsq1 = graph.create_named_value("unsq1", DataType::Int64, m3(batch, seq));
    graph.insert_node(Node::new(
        NodeId(3),
        "Unsqueeze",
        vec![Some(mask)],
        vec![unsq1],
    ));
    let padbool = graph.create_named_value("padbool", DataType::Bool, m3(batch, seq));
    graph.insert_node(Node::new(
        NodeId(4),
        "Cast",
        vec![Some(unsq1)],
        vec![padbool],
    ));
    let and = graph.create_named_value("and", DataType::Bool, m3(batch, seq));
    graph.insert_node(Node::new(
        NodeId(5),
        "And",
        vec![Some(ge), Some(padbool)],
        vec![and],
    ));
    let where_o = graph.create_named_value("where", DataType::Float32, m3(batch, seq));
    graph.insert_node(Node::new(
        NodeId(6),
        "Where",
        vec![Some(and)],
        vec![where_o],
    ));
    let cast_o = graph.create_named_value("cast", DataType::Float32, m3(batch, seq));
    graph.insert_node(Node::new(
        NodeId(7),
        "Cast",
        vec![Some(where_o)],
        vec![cast_o],
    ));
    let bias = graph.create_named_value(
        "mask_bias",
        DataType::Float32,
        vec![sym(batch), st(1), st(1), sym(seq)],
    );
    graph.insert_node(Node::new(
        NodeId(8),
        "Unsqueeze",
        vec![Some(cast_o)],
        vec![bias],
    ));

    // Capacity-form `Attention` consuming the additive bias (input 3) with past
    // KV bindings at inputs 4/5.
    let q = graph.create_named_value("q", DataType::Float32, vec![sym(batch), st(1), st(256)]);
    let attn =
        graph.create_named_value("attn", DataType::Float32, vec![sym(batch), st(1), st(256)]);
    let node = capacity_form_attention(10, q, bias, attn);
    graph.insert_node(node.clone());
    graph.add_output(attn);
    (graph, mask, seq, node)
}

#[test]
fn freeze_safe_mask_symbols_are_collected_and_admit_the_attention() {
    // The DeepSeek-V2-Lite decode-freeze-safe mask/bias length symbol is NOT a
    // KV-slot symbol, so `collect_capacity_pinned_kv_symbols` misses it and it
    // keeps the whole causal-mask cone + every Attention that consumes the bias
    // eager. `collect_freeze_safe_mask_symbols` recovers it, and pinning it makes
    // the Attention capture-eligible.
    let (mut graph, mask, seq, attn) = v2lite_symbolic_mask_graph();

    // Precondition: this binding is classified decode-freeze-safe.
    assert!(
        mask_binding_feeds_additive_causal_builder(&graph, mask),
        "the symbolic v2lite mask cone must be decode-freeze-safe"
    );

    // Make the mask/bias length symbol disqualifying (as inference minting does
    // for the real model) so the Attention is baseline-eager on its bias edge.
    graph.symbol_opaque.push(seq);
    let baseline = compute_capture_disqualifying_symbols(&graph);
    assert!(
        baseline.contains(&seq),
        "the mask/bias length symbol must be disqualifying baseline, got {baseline:?}"
    );
    assert!(
        !node_capture_seq_independent(&graph, &attn, &baseline),
        "without the pin the bias-consuming Attention must stay eager"
    );

    // The KV-slot pin alone does NOT recover the mask/bias symbol.
    let kv_pinned = collect_capacity_pinned_kv_symbols(&graph);
    assert!(
        !kv_pinned.contains(&seq),
        "the mask/bias symbol lives off the KV slots, so the KV pin must miss it, got {kv_pinned:?}"
    );

    // The freeze-safe mask collector recovers it.
    let mask_pinned = collect_freeze_safe_mask_symbols(&graph);
    assert!(
        mask_pinned.contains(&seq),
        "collect_freeze_safe_mask_symbols must recover the mask/bias length symbol, got {mask_pinned:?}"
    );

    // Pinning it excludes it from the disqualifying set and admits the Attention.
    let mut pinned = kv_pinned;
    pinned.extend(mask_pinned);
    let pinned_set = compute_capture_disqualifying_symbols_excluding(&graph, &pinned);
    assert!(
        !pinned_set.contains(&seq),
        "the pinned mask/bias symbol must be excluded from the disqualifying set, got {pinned_set:?}"
    );
    assert!(
        node_capture_seq_independent(&graph, &attn, &pinned_set),
        "with the freeze-safe mask pin the Attention must be capture-eligible"
    );

    // Idempotent.
    assert_eq!(
        collect_freeze_safe_mask_symbols(&graph),
        collect_freeze_safe_mask_symbols(&graph),
    );
}

#[test]
fn freeze_safe_mask_symbols_empty_for_non_freeze_safe_masks() {
    // GLM-5.2's indexer mixes the mask into a logical-width score via `Add` (not
    // an additive-mask-builder op), so the cone is NOT decode-freeze-safe and no
    // symbol is collected — the mask keeps exposing its logical length every step
    // and its symbol must never be pinned into capture.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sym = Dim::Symbolic;
    let batch = graph.create_symbol(None);
    let seq = graph.create_symbol(None);
    let sh = |b, s| vec![sym(b), sym(s)];
    let mask = graph.create_named_value("attention_mask", DataType::Int64, sh(batch, seq));
    graph.add_input(mask);
    let score = graph.create_named_value("indexer_score", DataType::Float32, sh(batch, seq));
    let cast = graph.create_named_value("cast", DataType::Float32, sh(batch, seq));
    graph.insert_node(Node::new(NodeId(0), "Cast", vec![Some(mask)], vec![cast]));
    let add = graph.create_named_value("add", DataType::Float32, sh(batch, seq));
    graph.insert_node(Node::new(
        NodeId(1),
        "Add",
        vec![Some(cast), Some(score)],
        vec![add],
    ));
    graph.add_output(add);
    assert!(
        !mask_binding_feeds_additive_causal_builder(&graph, mask),
        "the GLM indexer Add cone must NOT be decode-freeze-safe"
    );
    assert!(
        collect_freeze_safe_mask_symbols(&graph).is_empty(),
        "a non-freeze-safe mask must contribute no pinned symbols"
    );
}

#[test]
fn mask_binding_that_is_graph_output_is_rejected() {
    use onnx_runtime_ir::static_shape;
    // The mask binding itself is ALSO a graph output while feeding the builder to
    // a capacity-form Attention. Freezing it to physical width would leak the
    // padded `max_len` into the output escape, so the root must be rejected too.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sh = || static_shape([1]);
    let mask = graph.create_named_value("attention_mask", DataType::Int64, sh());
    graph.add_input(mask);
    let q = graph.create_named_value("q", DataType::Float32, sh());
    let bool_mask = graph.create_named_value("attn_mask_bool", DataType::Bool, sh());
    graph.insert_node(Node::new(
        NodeId(0),
        "Cast",
        vec![Some(mask)],
        vec![bool_mask],
    ));
    let attn = graph.create_named_value("attn", DataType::Float32, sh());
    graph.insert_node(capacity_form_attention(1, q, bool_mask, attn));
    graph.add_output(attn);
    // The mask binding escapes as a graph output as well.
    graph.add_output(mask);
    assert!(
        !mask_binding_feeds_capacity_form_attention(&graph, mask),
        "a mask binding that is itself a graph output must not be padded-safe"
    );
}

struct WeightDeliveryKernel {
    deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
    workspace_bytes: u64,
    workspace_bytes_per_row: u64,
}

impl WeightDeliveryKernel {
    fn copy_bytes(bytes: &[u8], output: &mut TensorMut<'_>) -> onnx_runtime_ep_api::Result<()> {
        if bytes.len() != output.byte_size() {
            return Err(EpError::KernelFailed(
                "test output byte count mismatch".into(),
            ));
        }
        // SAFETY: the executor bounds-checked and exclusively borrowed the
        // output allocation, which is exactly `output.byte_size()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.data.0.cast::<u8>(), bytes.len());
        }
        Ok(())
    }
}

impl Kernel for WeightDeliveryKernel {
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
    ) -> onnx_runtime_ep_api::Result<WorkspaceRequirement> {
        let rows = inputs
            .first()
            .and_then(|input| input.shape.first())
            .copied()
            .unwrap_or(1) as u64;
        let bytes = if self.workspace_bytes_per_row == 0 {
            self.workspace_bytes
        } else {
            self.workspace_bytes_per_row
                .checked_mul(rows)
                .ok_or_else(|| EpError::KernelFailed("test workspace overflow".into()))?
        };
        Ok(WorkspaceRequirement {
            bytes,
            alignment: if rows >= 4 { 512 } else { 256 },
            lifetime: WorkspaceLifetime::SessionPersistent,
            role: onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
        })
    }

    fn execute(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        self.deliveries.lock().unwrap().push("resident");
        let bytes = unsafe {
            std::slice::from_raw_parts(inputs[0].data_ptr::<u8>(), inputs[0].byte_size())
        };
        Self::copy_bytes(bytes, &mut outputs[0])
    }

    fn execute_with_inputs(
        &self,
        inputs: &[KernelInput<'_>],
        outputs: &mut [TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        match &inputs[0] {
            KernelInput::Tensor(view) => self.execute(std::slice::from_ref(view), outputs),
            KernelInput::Weight(handle) => {
                self.deliveries.lock().unwrap().push("lazy");
                let NegotiatedWeight::Lazy(lazy) =
                    handle.negotiate(&ExecutionProviderCapabilities::nxrt_weight_paging())?
                else {
                    return Err(EpError::KernelFailed(
                        "nxrt test EP expected a lazy WeightHandle".into(),
                    ));
                };
                let resident = lazy.materialize()?;
                Self::copy_bytes(resident.bytes(), &mut outputs[0])
            }
        }
    }
}

struct WorkspaceOnlyKernel {
    bytes: u64,
}

impl Kernel for WorkspaceOnlyKernel {
    fn workspace_requirement(
        &self,
        _inputs: &[TensorMetadata<'_>],
    ) -> onnx_runtime_ep_api::Result<WorkspaceRequirement> {
        Ok(WorkspaceRequirement {
            bytes: self.bytes,
            alignment: 256,
            lifetime: WorkspaceLifetime::SessionPersistent,
            role: MemoryRole::Workspace { step_scoped: false },
        })
    }

    fn execute(
        &self,
        _inputs: &[TensorView],
        _outputs: &mut [TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        Err(EpError::KernelFailed(
            "test workspace-only kernel requires prepared workspace".into(),
        ))
    }

    fn execute_with_workspace(
        &self,
        _inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> onnx_runtime_ep_api::Result<()> {
        let workspace =
            workspace.ok_or_else(|| EpError::KernelFailed("missing test workspace".into()))?;
        if workspace.bytes() < self.bytes as usize {
            return Err(EpError::KernelFailed(
                "test workspace was undersized".into(),
            ));
        }
        unsafe {
            std::ptr::write_bytes(outputs[0].data.0.cast::<u8>(), 0, outputs[0].byte_size());
        }
        Ok(())
    }
}

struct WeightDeliveryEp {
    cpu: CpuExecutionProvider,
    lazy: bool,
    optional_input_contract: bool,
    deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
    device: onnx_runtime_ir::DeviceId,
    allocations: Arc<AtomicUsize>,
    host_uploads: Arc<AtomicUsize>,
    workspace_bytes: u64,
    workspace_bytes_per_row: u64,
    support_index_share_workspace: bool,
    fail_next_allocation: Arc<AtomicBool>,
    fail_allocation_size: Arc<AtomicUsize>,
    /// Count of `acquire_routed_residency` calls, for the dispatch-site
    /// integration tests below (issue #82 slice 5): proves the executor
    /// gates guard acquisition on QMoE-family boundary nodes only.
    routed_residency_calls: Arc<AtomicUsize>,
}

impl WeightDeliveryEp {
    fn new(lazy: bool, deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>) -> Self {
        Self::with_device(
            lazy,
            deliveries,
            onnx_runtime_ir::DeviceId::cpu(),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        )
    }

    fn non_host(
        lazy: bool,
        deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
        allocations: Arc<AtomicUsize>,
        host_uploads: Arc<AtomicUsize>,
    ) -> Self {
        Self::with_device(
            lazy,
            deliveries,
            onnx_runtime_ir::DeviceId::new(onnx_runtime_ir::DeviceType::Custom(7), 0),
            allocations,
            host_uploads,
        )
    }

    fn with_device(
        lazy: bool,
        deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
        device: onnx_runtime_ir::DeviceId,
        allocations: Arc<AtomicUsize>,
        host_uploads: Arc<AtomicUsize>,
    ) -> Self {
        let mut cpu = CpuExecutionProvider::new();
        cpu.initialize(&EpConfig::default()).unwrap();
        Self {
            cpu,
            lazy,
            optional_input_contract: false,
            deliveries,
            device,
            allocations,
            host_uploads,
            workspace_bytes: 0,
            workspace_bytes_per_row: 0,
            support_index_share_workspace: false,
            fail_next_allocation: Arc::new(AtomicBool::new(false)),
            fail_allocation_size: Arc::new(AtomicUsize::new(0)),
            routed_residency_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn copy_bytes(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<()> {
        if size != 0 {
            // The test EP tags host allocations as a non-host custom device
            // so executor placement is realistic while bytes stay inspectable.
            unsafe { std::ptr::copy_nonoverlapping(src, dst, size) };
        }
        Ok(())
    }
}

impl ExecutionProvider for WeightDeliveryEp {
    fn consume_route_residency_at_boundary(&self) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        if self.lazy {
            "nxrt_test_ep"
        } else {
            "stock_test_ep"
        }
    }

    fn device_type(&self) -> onnx_runtime_ir::DeviceType {
        self.device.device_type
    }

    fn device_id(&self) -> onnx_runtime_ir::DeviceId {
        self.device
    }

    fn capabilities(&self) -> ExecutionProviderCapabilities {
        if self.lazy {
            ExecutionProviderCapabilities::nxrt_weight_paging()
        } else {
            ExecutionProviderCapabilities::stock()
        }
    }

    fn initialize(&mut self, _config: &EpConfig) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        _shapes: &[Shape],
        input_dtypes: &[DataType],
        _layouts: &[TensorLayout],
    ) -> KernelMatch {
        if self.optional_input_contract && op.op_type == "OptionalContract" {
            if input_dtypes == [DataType::Float32, DataType::Undefined, DataType::Bool] {
                return KernelMatch::Supported {
                    cost: Cost::ZERO,
                    required_input_layouts: None,
                    output_layouts: vec![TensorLayout::contiguous()],
                };
            }
            return KernelMatch::unsupported(format!(
                "OptionalContract requires [Float32, Undefined, Bool] input dtypes, got {input_dtypes:?}"
            ));
        }
        if self.support_index_share_workspace
            && op.domain == onnx_runtime_ir::RUNTIME_DOMAIN
            && op.op_type == "IndexShare"
        {
            return KernelMatch::Supported {
                cost: Cost::ZERO,
                required_input_layouts: None,
                output_layouts: vec![TensorLayout::contiguous()],
            };
        }
        if LazyWeightBoundary::BlockQuantizedMoe.matches(&op.domain, &op.op_type)
            || LazyWeightBoundary::MatMulNBits.matches(&op.domain, &op.op_type)
            || LazyWeightBoundary::QMoe.matches(&op.domain, &op.op_type)
            || (op.is_default_domain() && op.op_type == "Identity")
        {
            KernelMatch::Supported {
                cost: Cost::ZERO,
                required_input_layouts: None,
                output_layouts: vec![TensorLayout::contiguous()],
            }
        } else {
            KernelMatch::unsupported(format!(
                "no handler for {}::{} at opset {opset} — test EP intentionally declines this op",
                canonical_domain(op),
                op.op_type
            ))
        }
    }

    fn get_kernel(
        &self,
        op: &Node,
        _shapes: &[Vec<usize>],
        _opset: u64,
    ) -> onnx_runtime_ep_api::Result<Box<dyn Kernel>> {
        if self.support_index_share_workspace
            && op.domain == onnx_runtime_ir::RUNTIME_DOMAIN
            && op.op_type == "IndexShare"
        {
            return Ok(Box::new(WorkspaceOnlyKernel {
                bytes: self.workspace_bytes,
            }));
        }
        Ok(Box::new(WeightDeliveryKernel {
            deliveries: Arc::clone(&self.deliveries),
            workspace_bytes: self.workspace_bytes,
            workspace_bytes_per_row: self.workspace_bytes_per_row,
        }))
    }

    fn allocate(&self, size: usize, alignment: usize) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
        if self.fail_next_allocation.swap(false, Ordering::Relaxed) {
            return Err(EpError::OutOfMemory {
                requested: size,
                available: 0,
            });
        }
        if self
            .fail_allocation_size
            .compare_exchange(size, 0, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            return Err(EpError::OutOfMemory {
                requested: size,
                available: 0,
            });
        }
        self.allocations.fetch_add(1, Ordering::Relaxed);
        if self.device.is_host_accessible() {
            return self.cpu.allocate(size, alignment);
        }
        let layout = std::alloc::Layout::from_size_align(size.max(1), alignment)
            .map_err(|_| EpError::AlignmentError)?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(EpError::OutOfMemory {
                requested: size,
                available: 0,
            });
        }
        Ok(unsafe { DeviceBuffer::from_raw_parts(ptr.cast(), self.device, size, alignment) })
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> onnx_runtime_ep_api::Result<()> {
        if self.device.is_host_accessible() {
            return self.cpu.deallocate(buffer);
        }
        let size = buffer.len();
        let alignment = buffer.alignment();
        let ptr = buffer.into_raw().cast::<u8>();
        let layout = std::alloc::Layout::from_size_align(size.max(1), alignment)
            .expect("test EP allocated this layout");
        unsafe { std::alloc::dealloc(ptr, layout) };
        Ok(())
    }

    fn copy(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<()> {
        if size > src.len() || size > dst.len() {
            return Err(EpError::KernelFailed("test EP copy out of bounds".into()));
        }
        self.copy_bytes(src.as_ptr().cast(), dst.as_mut_ptr().cast(), size)
    }

    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<Fence> {
        self.copy(src, dst, size)?;
        Ok(Fence::default())
    }

    fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn copy_from_host(
        &self,
        src: &[u8],
        dst: &mut DeviceBuffer,
    ) -> onnx_runtime_ep_api::Result<()> {
        if src.len() > dst.len() {
            return Err(EpError::KernelFailed(
                "test EP host upload out of bounds".into(),
            ));
        }
        self.host_uploads.fetch_add(1, Ordering::Relaxed);
        self.copy_bytes(src.as_ptr(), dst.as_mut_ptr().cast(), src.len())
    }

    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> onnx_runtime_ep_api::Result<()> {
        if dst.len() > src.len() {
            return Err(EpError::KernelFailed(
                "test EP host download out of bounds".into(),
            ));
        }
        self.copy_bytes(src.as_ptr().cast(), dst.as_mut_ptr(), dst.len())
    }

    fn prefetch_lazy_weight(
        &self,
        _key: u64,
        _weight: &onnx_runtime_ep_api::LazyWeight,
        _source: &dyn onnx_runtime_ep_api::MmapRegionSource,
    ) -> onnx_runtime_ep_api::Result<bool> {
        self.deliveries.lock().unwrap().push("prefetch");
        Ok(true)
    }

    fn acquire_routed_residency(
        &self,
        _key: u64,
        requirement: onnx_runtime_ep_api::RoutedResidencyRequirement,
        catalog: &onnx_runtime_loader::WeightRegionCatalog,
    ) -> onnx_runtime_ep_api::Result<Option<Box<dyn onnx_runtime_ep_api::RoutedResidencyGuardHandle>>>
    {
        self.routed_residency_calls.fetch_add(1, Ordering::Relaxed);
        let proof = onnx_runtime_ep_api::prove_routed_residency(requirement, catalog);
        struct TestGuardHandle(onnx_runtime_ep_api::RoutedResidencyProof);
        impl onnx_runtime_ep_api::RoutedResidencyGuardHandle for TestGuardHandle {
            fn proof(&self) -> &onnx_runtime_ep_api::RoutedResidencyProof {
                &self.0
            }
        }
        Ok(Some(Box::new(TestGuardHandle(proof))))
    }

    fn reserve_workspace(
        &self,
        bytes: u64,
        role: onnx_runtime_memory_governor::MemoryRole,
    ) -> onnx_runtime_ep_api::Result<Option<onnx_runtime_memory_governor::MemoryLease>> {
        assert_eq!(role, MemoryRole::Workspace { step_scoped: false });
        self.deliveries.lock().unwrap().push("reserve_workspace");
        // Static-bytes fixtures reserve exactly `workspace_bytes`. The per-row
        // fixture (`workspace_bytes_per_row != 0`, e.g.
        // `inference_session_fallback_workspace_grows_retries_and_reuses`)
        // reserves `rows * per_row`, so the governed reservation legitimately
        // scales with the input row count — 2048 for 2 rows, 4096 for 4. This
        // method has no row count to recompute the exact figure, so it asserts
        // the non-vacuous invariant that the reservation is a positive multiple
        // of the per-row size (the exact bytes are separately pinned at each
        // call site). A zero or mis-sized reservation still fails here.
        if self.workspace_bytes_per_row == 0 {
            assert_eq!(bytes, self.workspace_bytes);
        } else {
            assert!(
                bytes != 0 && bytes.is_multiple_of(self.workspace_bytes_per_row),
                "per-row workspace reservation {bytes} must be a positive multiple of {}",
                self.workspace_bytes_per_row
            );
        }
        Ok(None)
    }
}

fn weight_delivery_fixture() -> (Graph, Arc<WeightStore>, std::path::PathBuf) {
    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap().join("target"))
        .join("weight-handle-tests");
    std::fs::create_dir_all(&root).unwrap();
    let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let path = root.join(format!(
        "block-quantized-moe-{}-{id}.bin",
        std::process::id()
    ));
    std::fs::write(&path, [1u8, 2, 3, 4]).unwrap();

    let mut graph = Graph::new();
    graph.opset_imports.insert("pkg.nxrt".into(), 1);
    let weight = graph.create_named_value("weight", DataType::Uint8, static_shape([4]));
    graph.set_initializer(
        weight,
        WeightRef::External {
            path: path.clone(),
            offset: 0,
            length: 4,
            dtype: DataType::Uint8,
            dims: vec![4],
        },
    );
    let output = graph.create_named_value("output", DataType::Uint8, static_shape([4]));
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMoE",
        vec![Some(weight)],
        vec![output],
    );
    node.domain = "pkg.nxrt".into();
    graph.insert_node(node);
    graph.add_output(output);

    let mut store = WeightStore::new();
    store.map_external(&path).unwrap();
    (graph, Arc::new(store), path)
}

#[test]
fn prepare_reserves_static_nested_qmoe_workspace_and_child_reuses_it() {
    fn branch(name: &str) -> Graph {
        let mut graph = Graph::new();
        let input = graph.create_named_value(
            format!("{name}_input"),
            DataType::Float32,
            static_shape([4]),
        );
        graph.set_initializer(
            input,
            WeightRef::Inline(onnx_runtime_ir::TensorData::from_raw(
                DataType::Float32,
                vec![4],
                [1.0f32, 2.0, 3.0, 4.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            )),
        );
        let output = graph.create_named_value(
            format!("{name}_output"),
            DataType::Float32,
            static_shape([4]),
        );
        let mut node = Node::new(
            NodeId(0),
            "BlockQuantizedMoE",
            vec![Some(input)],
            vec![output],
        );
        node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
        graph.insert_node(node);
        graph.add_output(output);
        graph
    }

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    graph
        .opset_imports
        .insert(onnx_runtime_ir::RUNTIME_DOMAIN.into(), 1);
    let cond = graph.create_named_value("cond", DataType::Bool, static_shape([]));
    graph.add_input(cond);
    let output = graph.create_named_value("output", DataType::Float32, static_shape([4]));
    graph.add_output(output);
    let if_id = NodeId(0);
    graph.insert_node(Node::new(if_id, "If", vec![Some(cond)], vec![output]));
    graph
        .subgraphs
        .insert((if_id, "then_branch".into()), branch("then"));
    graph
        .subgraphs
        .insert((if_id, "else_branch".into()), branch("else"));

    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut ep = WeightDeliveryEp::new(false, deliveries);
    ep.workspace_bytes = 4096;
    let mut executor = Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(ep)).unwrap();
    let cond = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
    let requirement = executor
        .prepare_with_device_bindings(&[("cond", &cond)], &mut [])
        .unwrap();
    assert_eq!(requirement.bytes, 4096);
    assert_eq!(executor.persistent_workspace.as_ref().unwrap().bytes, 4096);

    let output = executor.run(&[("cond", &cond)]).unwrap();
    assert_eq!(output[0].to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn prepare_reserves_static_index_share_workspace() {
    let mut graph = Graph::new();
    graph
        .opset_imports
        .insert(onnx_runtime_ir::RUNTIME_DOMAIN.into(), 1);
    let q = graph.create_named_value("q", DataType::Float32, static_shape([1, 2, 3, 4]));
    let k = graph.create_named_value("k", DataType::Float32, static_shape([1, 1, 3, 4]));
    let v = graph.create_named_value("v", DataType::Float32, static_shape([1, 1, 3, 4]));
    let selected =
        graph.create_named_value("selected", DataType::Int64, static_shape([1, 1, 3, 2]));
    for input in [q, k, v, selected] {
        graph.add_input(input);
    }
    let out = graph.create_named_value("out", DataType::Float32, static_shape([1, 2, 3, 4]));
    graph.add_output(out);
    let mut node = Node::new(
        NodeId(0),
        "IndexShare",
        vec![Some(q), Some(k), Some(v), None, None, Some(selected)],
        vec![out],
    );
    node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
    graph.insert_node(node);

    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut ep = WeightDeliveryEp::new(false, Arc::clone(&deliveries));
    ep.support_index_share_workspace = true;
    ep.workspace_bytes = 768;
    let mut executor = Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(ep)).unwrap();
    let q_tensor = Tensor::zeros(DataType::Float32, vec![1, 2, 3, 4]).unwrap();
    let k_tensor = Tensor::zeros(DataType::Float32, vec![1, 1, 3, 4]).unwrap();
    let v_tensor = Tensor::zeros(DataType::Float32, vec![1, 1, 3, 4]).unwrap();
    let selected_tensor = Tensor::from_i64(&[1, 1, 3, 2], &[0, 1, 0, 1, 0, 1]).unwrap();
    let requirement = executor
        .prepare_with_device_bindings(
            &[
                ("q", &q_tensor),
                ("k", &k_tensor),
                ("v", &v_tensor),
                ("selected", &selected_tensor),
            ],
            &mut [],
        )
        .unwrap();
    assert_eq!(requirement.bytes, 768);
    assert_eq!(
        requirement.role,
        MemoryRole::Workspace { step_scoped: false }
    );
    assert_eq!(executor.persistent_workspace.as_ref().unwrap().bytes, 768);
    assert_eq!(
        deliveries.lock().unwrap().as_slice(),
        ["reserve_workspace"],
        "IndexShare prepare must use the governed workspace path instead of an internal raw allocation"
    );
}

#[test]
fn inference_session_fallback_workspace_grows_retries_and_reuses() {
    let mut graph = Graph::new();
    graph
        .opset_imports
        .insert(onnx_runtime_ir::RUNTIME_DOMAIN.into(), 1);
    let rows = SymbolId(0);
    let shape = vec![Dim::Symbolic(rows)];
    let input = graph.create_named_value("input", DataType::Float32, shape.clone());
    graph.add_input(input);
    let output = graph.create_named_value("output", DataType::Float32, shape);
    graph.add_output(output);
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMoE",
        vec![Some(input)],
        vec![output],
    );
    node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
    graph.insert_node(node);

    let inputs = crate::io_meta(&graph, &graph.inputs);
    let outputs = crate::io_meta(&graph, &graph.outputs);
    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut ep = WeightDeliveryEp::new(false, deliveries);
    ep.workspace_bytes_per_row = 1024;
    let fail_size = Arc::clone(&ep.fail_allocation_size);
    let exec = Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(ep)).unwrap();
    let mut session = crate::InferenceSession {
        inputs,
        outputs,
        model_metadata: crate::ModelMetadata::default(),
        exec,
        decode_inline_exec: None,
        verify_exec: None,
        ep_context_config: crate::EpContextDumpConfig::default(),
    };

    let small = Tensor::from_f32(&[2], &[1.0, 2.0]).unwrap();
    assert_eq!(
        session.run(&[("input", &small)]).unwrap()[0].to_vec_f32(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        session.exec.persistent_workspace.as_ref().unwrap().bytes,
        2048
    );
    assert_eq!(
        session
            .exec
            .persistent_workspace
            .as_ref()
            .unwrap()
            .alignment,
        256
    );

    let large = Tensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    fail_size.store(4096, Ordering::Relaxed);
    let _error = session
        .run(&[("input", &large)])
        .expect_err("workspace replacement allocation must fail once");
    assert!(
        session.exec.persistent_workspace.is_none(),
        "failed growth must leave a valid empty slot"
    );

    assert_eq!(
        session.run(&[("input", &large)]).unwrap()[0].to_vec_f32(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    let grown = session.exec.persistent_workspace.as_ref().unwrap();
    assert_eq!(grown.bytes, 4096);
    assert_eq!(grown.alignment, 512);
    let ptr = grown.buffer.as_ptr();

    assert_eq!(
        session.run(&[("input", &small)]).unwrap()[0].to_vec_f32(),
        vec![1.0, 2.0]
    );
    let reused = session.exec.persistent_workspace.as_ref().unwrap();
    assert_eq!(reused.bytes, 4096);
    assert_eq!(reused.buffer.as_ptr(), ptr);
}

/// #1223: a *prepared* session must re-prepare its governed workspace when
/// execution rebuckets to a larger shape bucket than preparation reserved for.
///
/// Preparation reserves one governed-workspace slot per lifetime class from the
/// shapes it is handed, and — so a captured device graph can bake a stable
/// pointer — a prepared session otherwise refuses to (re)allocate at execution
/// time. That invariant only holds *within* one shape bucket. When decode grows
/// past a bucket edge (or a prompt lands in a different bucket than its decode
/// steps), the slot reserved under the old bucket's geometry can be undersized
/// for the new one. Before the fix, the larger bucket failed the prepared-
/// workspace invariant with "workspace invariant mismatch"; #1221 fixed the same
/// failure mode *in-bucket* for `Attention` but scoped the cross-bucket case out.
///
/// This drives the exact gap on the executor's shared workspace path (not any
/// one op-type): reserve a `SessionPersistent` per-row slot for a 2-row bucket,
/// then execute a 4-row bucket. The rebucket lands on an eager dispatch (as it
/// does on the real growing-KV decode path, which declines capture, and on the
/// eager re-warm a capture-eligible model runs after a KV-growth graph
/// invalidation), so the slot is re-prepared in place. Reverting the fix makes
/// the 4-row execute fail with "workspace invariant mismatch".
#[test]
fn prepared_session_reprepares_workspace_when_execution_rebuckets() {
    let mut graph = Graph::new();
    graph
        .opset_imports
        .insert(onnx_runtime_ir::RUNTIME_DOMAIN.into(), 1);
    let rows = SymbolId(0);
    let shape = vec![Dim::Symbolic(rows)];
    let input = graph.create_named_value("input", DataType::Float32, shape.clone());
    graph.add_input(input);
    let output = graph.create_named_value("output", DataType::Float32, shape);
    graph.add_output(output);
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMoE",
        vec![Some(input)],
        vec![output],
    );
    node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
    graph.insert_node(node);

    let inputs = crate::io_meta(&graph, &graph.inputs);
    let outputs = crate::io_meta(&graph, &graph.outputs);
    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut ep = WeightDeliveryEp::new(false, deliveries);
    ep.workspace_bytes_per_row = 1024;
    let exec = Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(ep)).unwrap();
    let mut session = crate::InferenceSession {
        inputs,
        outputs,
        model_metadata: crate::ModelMetadata::default(),
        exec,
        decode_inline_exec: None,
        verify_exec: None,
        ep_context_config: crate::EpContextDumpConfig::default(),
    };

    // Bucket A: preparation reserves the SessionPersistent slot for a 2-row
    // shape and latches "workspace preparation required", so execution may no
    // longer lazily grow the slot within this bucket.
    let bucket_a = Tensor::from_f32(&[2], &[1.0, 2.0]).unwrap();
    session
        .exec
        .prepare_with_device_bindings(&[("input", &bucket_a)], &mut [])
        .unwrap();
    let reserved = session.exec.persistent_workspace.as_ref().unwrap();
    assert_eq!(reserved.bytes, 2048);
    assert_eq!(reserved.alignment, 256);
    assert!(
        session.exec.workspace_preparation_required,
        "prepare must latch the prepared-workspace invariant"
    );

    // In-bucket execution still fits under the reservation.
    assert_eq!(
        session.run(&[("input", &bucket_a)]).unwrap()[0].to_vec_f32(),
        vec![1.0, 2.0]
    );

    // Bucket B: a larger bucket needs 4096 bytes at a tighter alignment than the
    // 2-row reservation. Without the #1223 fix this eager rebucket dispatch
    // fails the prepared-workspace invariant; with it, preparation is re-run in
    // place and the slot grows.
    let bucket_b = Tensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(
        session.run(&[("input", &bucket_b)]).unwrap()[0].to_vec_f32(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    let grown = session.exec.persistent_workspace.as_ref().unwrap();
    assert_eq!(grown.bytes, 4096);
    assert_eq!(grown.alignment, 512);
    assert!(
        session.exec.workspace_preparation_required,
        "re-preparing on rebucket must not drop the prepared-workspace invariant"
    );

    // Rebucketing back down reuses the grown slot (a slot only ever grows).
    assert_eq!(
        session.run(&[("input", &bucket_a)]).unwrap()[0].to_vec_f32(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        session.exec.persistent_workspace.as_ref().unwrap().bytes,
        4096
    );
}

fn two_node_weight_delivery_fixture() -> (Graph, Arc<WeightStore>, std::path::PathBuf) {
    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap().join("target"))
        .join("weight-handle-tests");
    std::fs::create_dir_all(&root).unwrap();
    let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let path = root.join(format!("matmul-nbits-pair-{}-{id}.bin", std::process::id()));
    std::fs::write(&path, [1u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();

    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let first_weight = graph.create_named_value("first_weight", DataType::Uint8, static_shape([4]));
    let second_weight =
        graph.create_named_value("second_weight", DataType::Uint8, static_shape([4]));
    graph.set_initializer(
        first_weight,
        WeightRef::External {
            path: path.clone(),
            offset: 0,
            length: 4,
            dtype: DataType::Uint8,
            dims: vec![4],
        },
    );
    graph.set_initializer(
        second_weight,
        WeightRef::External {
            path: path.clone(),
            offset: 4,
            length: 4,
            dtype: DataType::Uint8,
            dims: vec![4],
        },
    );
    let first_output = graph.create_named_value("first_output", DataType::Uint8, static_shape([4]));
    let final_output = graph.create_named_value("final_output", DataType::Uint8, static_shape([4]));
    let mut first = Node::new(
        NodeId(0),
        "MatMulNBits",
        vec![Some(first_weight)],
        vec![first_output],
    );
    first.domain = "com.microsoft".into();
    graph.insert_node(first);
    let mut second = Node::new(
        NodeId(1),
        "MatMulNBits",
        vec![Some(second_weight), Some(first_output)],
        vec![final_output],
    );
    second.domain = "com.microsoft".into();
    graph.insert_node(second);
    graph.add_output(final_output);

    let mut store = WeightStore::new();
    store.map_external(&path).unwrap();
    (graph, Arc::new(store), path)
}

/// A `com.microsoft::QMoE` node with one expert-major fc1 packed/scales pair
/// (2 experts, 3 rows/expert, 4 storage elements/row) as external
/// initializers, plus an activation input and output so it type-checks as a
/// minimal QMoE graph. Used to exercise issue #82's per-expert region
/// candidate bookkeeping without touching kernel execution.
fn qmoe_expert_region_fixture() -> (Graph, Arc<WeightStore>, std::path::PathBuf) {
    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap().join("target"))
        .join("weight-handle-tests");
    std::fs::create_dir_all(&root).unwrap();
    let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let path = root.join(format!(
        "qmoe-expert-region-{}-{id}.bin",
        std::process::id()
    ));
    // fc1 packed: 2 experts * 3 rows * 4 storage elements (Uint8) = 24 bytes.
    // fc1 scales: 2 experts * 3 rows * 1 block = 6 bytes (Float32 dtype but we
    // only need declared length/dims to match for this fixture).
    std::fs::write(&path, [0u8; 24 + 6]).unwrap();

    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let activation =
        graph.create_named_value("activation", DataType::Float32, static_shape([1, 4]));
    graph.add_input(activation);
    let packed = graph.create_named_value("fc1_packed", DataType::Uint8, static_shape([2, 3, 4]));
    graph.set_initializer(
        packed,
        WeightRef::External {
            path: path.clone(),
            offset: 0,
            length: 24,
            dtype: DataType::Uint8,
            dims: vec![2, 3, 4],
        },
    );
    let scales = graph.create_named_value("fc1_scales", DataType::Uint8, static_shape([2, 3, 1]));
    graph.set_initializer(
        scales,
        WeightRef::External {
            path: path.clone(),
            offset: 24,
            length: 6,
            dtype: DataType::Uint8,
            dims: vec![2, 3, 1],
        },
    );
    let output = graph.create_named_value("output", DataType::Float32, static_shape([1, 4]));
    let mut node = Node::new(
        NodeId(0),
        "QMoE",
        vec![
            Some(activation),
            None,
            Some(packed),
            Some(scales),
            None,
            Some(packed),
            Some(scales),
        ],
        vec![output],
    );
    node.domain = "com.microsoft".to_owned();
    node.attributes
        .insert("expert_weight_bits".to_owned(), Attribute::Int(4));
    node.attributes
        .insert("block_size".to_owned(), Attribute::Int(32));
    graph.insert_node(node);
    graph.add_output(output);

    let mut store = WeightStore::new();
    store.map_external(&path).unwrap();
    (graph, Arc::new(store), path)
}

#[test]
fn qmoe_expert_region_candidates_partition_expert_major_bank_without_changing_handle_output() {
    let (graph, weights, _path) = qmoe_expert_region_fixture();
    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ep = WeightDeliveryEp::new(true, deliveries);
    let executor = Executor::build(graph, weights, Arc::new(ep)).unwrap();

    // Existing behavior unchanged: still exactly one lazy weight handle per
    // initializer (fc1 packed + fc1 scales), each with exactly one region.
    assert_eq!(executor.weight_handles.len(), 2);
    for handle in executor.weight_handles.values() {
        let WeightHandle::Lazy(lazy) = handle else {
            panic!("expected a lazy weight handle");
        };
        assert_eq!(lazy.regions.len(), 1);
    }

    // Additive: both QMoE initializers got a per-expert region candidate.
    let candidates = executor.expert_region_candidates();
    assert_eq!(candidates.len(), 2);
    for catalog in candidates.values() {
        assert!(catalog.is_pageable(), "{:?}", catalog.pageability());
        // Regions exactly partition the tensor: contiguous, non-overlapping.
        let mut expected_start = 0usize;
        for expert in 0..2 {
            let range = catalog
                .relative_range(expert)
                .unwrap_or_else(|| panic!("expert {expert} must have a region"));
            assert_eq!(range.start, expected_start);
            expected_start = range.end;
        }
        assert!(catalog.region(2).is_none());
    }
}

#[test]
fn qmoe_expert_region_candidates_record_reason_for_non_rank3_layout() {
    let (mut graph, weights, path) = qmoe_expert_region_fixture();
    // Corrupt the fc1 packed initializer to rank-2 so it can no longer derive
    // an expert-major layout; the candidate map must still record a reason
    // via `NonPageableReason`, not silently omit or panic.
    let packed_value = graph
        .initializers
        .keys()
        .copied()
        .find(|value| graph.value(*value).name.as_deref() == Some("fc1_packed"))
        .expect("fc1_packed initializer must exist");
    graph.set_initializer(
        packed_value,
        WeightRef::External {
            path: path.clone(),
            offset: 0,
            length: 24,
            dtype: DataType::Uint8,
            dims: vec![6, 4],
        },
    );

    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ep = WeightDeliveryEp::new(true, deliveries);
    let executor = Executor::build(graph, weights, Arc::new(ep)).unwrap();

    // Rank-2 dims cannot derive an expert-major layout, but the value must
    // still be recorded (not silently dropped) with an explicit
    // `NonPageableReason`; the untouched rank-3 scales tensor is still
    // pageable.
    let candidates = executor.expert_region_candidates();
    assert_eq!(candidates.len(), 2);
    let packed_catalog = candidates
        .get(&packed_value)
        .expect("rank-2 fc1_packed must still be recorded, not silently omitted");
    assert!(!packed_catalog.is_pageable());
    assert_eq!(
        packed_catalog.pageability(),
        &onnx_runtime_loader::Pageability::NonPageable(
            onnx_runtime_loader::NonPageableReason::NotExpertMajor
        )
    );

    // The weight handle map is unaffected either way: still exactly one lazy
    // handle per initializer, proving the invalid layout does not change
    // existing binder/allocation behavior.
    assert_eq!(executor.weight_handles.len(), 2);
}

/// Dispatch-time integration proof for issue #82 slice 5: `execute_kernel`
/// acquires exactly one routed-residency guard per QMoE dispatch (never zero,
/// never more than once per node — `find_map` short-circuits on the first
/// matching catalog), and never for a non-QMoE-family node.
#[test]
fn qmoe_dispatch_acquires_exactly_one_routed_residency_guard_per_node() {
    let (graph, weights, _path) = qmoe_expert_region_fixture();
    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ep = WeightDeliveryEp::new(true, deliveries);
    let calls = Arc::clone(&ep.routed_residency_calls);
    let mut executor = Executor::build(graph, weights, Arc::new(ep)).unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let input = Tensor::from_f32(&[1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let _ = executor.run(&[("activation", &input)]).unwrap();
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "one QMoE dispatch must acquire exactly one routed-residency guard"
    );

    // A second dispatch of the same (only) node acquires exactly one more,
    // proving the guard is re-acquired per dispatch rather than leaked or
    // acquired once and reused stale.
    let _ = executor.run(&[("activation", &input)]).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

/// A non-QMoE-family node (plain `Identity`) must never trigger routed-
/// residency guard acquisition, proving the dispatch-site gate is scoped to
/// `LazyWeightBoundary::{QMoe, BlockQuantizedMoe}` and not every node.
#[test]
fn non_qmoe_dispatch_never_acquires_a_routed_residency_guard() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 18);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([1, 4]));
    graph.add_input(input);
    let output = graph.create_named_value("output", DataType::Float32, static_shape([1, 4]));
    let node = Node::new(NodeId(0), "Identity", vec![Some(input)], vec![output]);
    graph.insert_node(node);
    graph.add_output(output);

    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ep = WeightDeliveryEp::new(true, deliveries);
    let calls = Arc::clone(&ep.routed_residency_calls);
    let mut executor = Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(ep)).unwrap();

    let values = Tensor::from_f32(&[1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let _ = executor.run(&[("input", &values)]).unwrap();
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "a non-QMoE-family node must never acquire a routed-residency guard"
    );
}

#[test]
fn claim_time_optional_input_dtype_is_undefined_not_silently_float32() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 1);
    let data = graph.create_named_value("data", DataType::Float32, static_shape([1]));
    let training_mode = graph.create_named_value("training_mode", DataType::Bool, static_shape([]));
    let output = graph.create_named_value("output", DataType::Float32, static_shape([1]));
    graph.add_input(data);
    graph.add_input(training_mode);
    graph.add_output(output);
    graph.insert_node(Node::new(
        NodeId(0),
        "OptionalContract",
        vec![Some(data), None, Some(training_mode)],
        vec![output],
    ));

    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut ep = WeightDeliveryEp::new(false, deliveries);
    ep.optional_input_contract = true;
    let executor = Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(ep));
    assert!(
        executor.is_ok(),
        "an omitted optional input must reach supports_op as DataType::Undefined"
    );
}

#[test]
fn executor_opens_per_op_span_only_when_tracing_enabled() {
    use onnx_runtime_tracer::TraceContext;

    // Disabled (default noop): no spans recorded, hot path stays quiet.
    {
        let (graph, weights, path) = weight_delivery_fixture();
        let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ep: Arc<dyn ExecutionProvider> =
            Arc::new(WeightDeliveryEp::new(false, Arc::clone(&deliveries)));
        let mut executor = Executor::build(graph, weights, ep).unwrap();
        let (trace, events) = TraceContext::in_memory();
        trace.set_enabled(false);
        executor.set_trace_context(trace);
        let _ = executor.run(&[]).unwrap();
        drop(executor);
        std::fs::remove_file(path).unwrap();
        assert!(
            events.events().is_empty(),
            "a disabled trace context must not open op spans"
        );
    }

    // Enabled: exactly one op span per executed node, named by op type.
    {
        let (graph, weights, path) = weight_delivery_fixture();
        let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ep: Arc<dyn ExecutionProvider> =
            Arc::new(WeightDeliveryEp::new(false, Arc::clone(&deliveries)));
        let mut executor = Executor::build(graph, weights, ep).unwrap();
        let (trace, events) = TraceContext::in_memory();
        executor.set_trace_context(trace);
        let _ = executor.run(&[]).unwrap();
        drop(executor);
        std::fs::remove_file(path).unwrap();
        let spans = events.events();
        assert_eq!(spans.len(), 1, "one op span per executed node");
        assert_eq!(spans[0].name, "BlockQuantizedMoE");
        assert_eq!(spans[0].cat, "op");
    }
}

#[test]
fn op_capture_trace_annotates_span_with_status_and_reason() {
    use onnx_runtime_tracer::TraceContext;

    // Rejected: both a status and the actionable why-not reason land on the
    // active op-span. This is the branch a production model that declines
    // capture would hit; the qwen fixture captures cleanly so it is proven
    // here directly rather than in the live trace.
    {
        let (trace, events) = TraceContext::in_memory();
        {
            let _span = trace.span("MatMulNBits", "op");
            OpCaptureTrace::Rejected(
                "kernel declares CaptureSupport::Unsupported: per-call workspace alloc",
            )
            .annotate();
        }

        let recorded = events.events();
        assert_eq!(recorded.len(), 1);
        let args = recorded[0].args.as_ref().unwrap();
        assert_eq!(args[ARG_CAPTURE_STATUS], "rejected");
        assert!(
            args[ARG_CAPTURE_REASON]
                .as_str()
                .unwrap()
                .contains("CaptureSupport::Unsupported")
        );
    }

    // Captured: status only, no reason (nothing was declined).
    {
        let (trace, events) = TraceContext::in_memory();
        {
            let _span = trace.span("MatMulNBits", "op");
            OpCaptureTrace::Captured.annotate();
        }
        let recorded = events.events();
        let args = recorded[0].args.as_ref().unwrap();
        assert_eq!(args[ARG_CAPTURE_STATUS], "captured");
    }

    // Eager: no capture attempt, so no capture annotation at all.
    {
        let (trace, events) = TraceContext::in_memory();
        {
            let _span = trace.span("MatMulNBits", "op");
            OpCaptureTrace::Eager.annotate();
        }
        let recorded = events.events();
        assert!(
            recorded[0]
                .args
                .as_ref()
                .map(|a| a.get(ARG_CAPTURE_STATUS).is_none())
                .unwrap_or(true),
            "eager ops carry no capture status"
        );
    }
}

#[cfg(feature = "gpu-tests")]
fn sealed_bqmoe_session_graph() -> Graph {
    fn q8_bank(experts: usize, out: usize, input: usize, seed: i8) -> Vec<u8> {
        let blocks = input / 32;
        let mut bytes = vec![0u8; experts * out * blocks * 34];
        for (block_index, block) in bytes.chunks_exact_mut(34).enumerate() {
            block[..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
            for (index, value) in block[2..].iter_mut().enumerate() {
                *value = seed
                    .wrapping_add((block_index as i8).wrapping_mul(3))
                    .wrapping_add(index as i8)
                    .to_ne_bytes()[0];
            }
        }
        bytes
    }

    let mut graph = Graph::new();
    graph.opset_imports.insert("pkg.nxrt".into(), 1);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([1, 32]));
    graph.add_input(x);
    let router = graph.create_named_value("router", DataType::Float32, static_shape([1, 2]));
    graph.add_input(router);
    let add_bank = |graph: &mut Graph, name: &str, shape: Vec<usize>, bytes: Vec<u8>| {
        let value =
            graph.create_named_value(name, DataType::Uint8, static_shape(shape.iter().copied()));
        graph.set_initializer(
            value,
            WeightRef::Inline(onnx_runtime_ir::TensorData::from_raw(
                DataType::Uint8,
                shape,
                bytes,
            )),
        );
        value
    };
    let fc1 = add_bank(
        &mut graph,
        "gate",
        vec![2, 32, 1, 34],
        q8_bank(2, 32, 32, 3),
    );
    let fc2 = add_bank(
        &mut graph,
        "down",
        vec![2, 32, 1, 34],
        q8_bank(2, 32, 32, 17),
    );
    let fc3 = add_bank(&mut graph, "up", vec![2, 32, 1, 34], q8_bank(2, 32, 32, 41));
    let output = graph.create_named_value("output", DataType::Float32, static_shape([1, 32]));
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMoE",
        vec![
            Some(x),
            Some(router),
            Some(fc1),
            None,
            Some(fc2),
            None,
            Some(fc3),
        ],
        vec![output],
    );
    node.domain = "pkg.nxrt".into();
    node.attributes.insert("k".into(), Attribute::Int(2));
    node.attributes.insert(
        "activation_type".into(),
        Attribute::String(b"silu".to_vec()),
    );
    node.attributes
        .insert("normalize_routing_weights".into(), Attribute::Int(1));
    node.attributes
        .insert("swiglu_fusion".into(), Attribute::Int(0));
    for name in ["fc1_format", "fc2_format", "fc3_format"] {
        node.attributes
            .insert(name.into(), Attribute::String(b"q8_0".to_vec()));
    }
    node.attributes
        .insert("block_layout_version".into(), Attribute::Int(1));
    graph.insert_node(node);
    graph.add_output(output);
    graph
}

#[cfg(feature = "gpu-tests")]
fn sealed_bqmoe_cuda_session_for_provider(
    cuda: Arc<onnx_runtime_ep_cuda::CudaExecutionProvider>,
) -> (
    crate::InferenceSession,
    Vec<DeviceIoBinding>,
    Arc<onnx_runtime_ep_cuda::runtime::CudaRuntime>,
) {
    let runtime = Arc::clone(cuda.runtime());
    let mut session = crate::InferenceSession::from_graph_with_provider(
        sealed_bqmoe_session_graph(),
        Arc::new(WeightStore::new()),
        std::path::Path::new("."),
        cuda,
    )
    .unwrap();
    let mut bindings = vec![
        session
            .allocate_device_binding(
                "x",
                None::<String>,
                DataType::Float32,
                vec![1, 32],
                vec![1, 32],
            )
            .unwrap(),
        session
            .allocate_device_binding(
                "router",
                None::<String>,
                DataType::Float32,
                vec![1, 2],
                vec![1, 2],
            )
            .unwrap(),
        session
            .allocate_device_output_binding("output", DataType::Float32, vec![1, 32], vec![1, 32])
            .unwrap(),
    ];
    let input = (0..32)
        .flat_map(|index| ((index as f32 - 15.5) / 16.0).to_le_bytes())
        .collect::<Vec<_>>();
    let router = [3.0f32, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    bindings[0].write_bytes(0, &input).unwrap();
    bindings[1].write_bytes(0, &router).unwrap();
    session
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    bindings[2].read_bytes_range(0, 4).unwrap();
    (session, bindings, runtime)
}

#[cfg(feature = "gpu-tests")]
fn sealed_bqmoe_cuda_session_fixture() -> (
    crate::InferenceSession,
    Vec<DeviceIoBinding>,
    Arc<onnx_runtime_ep_cuda::runtime::CudaRuntime>,
) {
    sealed_bqmoe_cuda_session_for_provider(Arc::new(
        onnx_runtime_ep_cuda::CudaExecutionProvider::new_default().unwrap(),
    ))
}

#[cfg(feature = "gpu-tests")]
#[test]
fn sealed_bqmoe_executes_through_production_session_path() {
    use onnx_runtime_ep_cpu::kernels::block_quantized_moe::BLOCK_QUANT_MOE_DENSE_EXPANSIONS;
    use onnx_runtime_ep_cuda::CudaExecutionProvider;

    let input_values = (0..32)
        .map(|index| (index as f32 - 15.5) / 16.0)
        .collect::<Vec<_>>();
    let input = Tensor::from_f32(&[1, 32], &input_values).unwrap();
    let router = Tensor::from_f32(&[1, 2], &[3.0, 1.0]).unwrap();

    let mut cpu = Executor::build(
        sealed_bqmoe_session_graph(),
        Arc::new(WeightStore::new()),
        Arc::new(CpuExecutionProvider::new()),
    )
    .unwrap();
    let expected = cpu.run(&[("x", &input), ("router", &router)]).unwrap()[0].to_vec_f32();

    let cuda = Arc::new(CudaExecutionProvider::new_default().unwrap());
    let runtime = Arc::clone(cuda.runtime());
    let dense_before = BLOCK_QUANT_MOE_DENSE_EXPANSIONS.load(Ordering::Relaxed);
    let mut executor = Executor::build(
        sealed_bqmoe_session_graph(),
        Arc::new(WeightStore::new()),
        cuda,
    )
    .unwrap();
    let first = executor.run(&[("x", &input), ("router", &router)]).unwrap()[0].to_vec_f32();
    let transfers = runtime.transfer_counts();
    let allocations = runtime.allocation_counts();
    let second = executor.run(&[("x", &input), ("router", &router)]).unwrap()[0].to_vec_f32();
    let after_transfers = runtime.transfer_counts();
    assert_eq!(
        after_transfers.host_to_device - transfers.host_to_device,
        2,
        "only the two host boundary inputs may upload"
    );
    assert_eq!(
        after_transfers.async_host_to_device, transfers.async_host_to_device,
        "sealed weights must not enter the paging transfer path"
    );
    assert_eq!(runtime.allocation_counts(), allocations);
    assert_eq!(
        BLOCK_QUANT_MOE_DENSE_EXPANSIONS.load(Ordering::Relaxed),
        dense_before,
        "CUDA production execution must not enter the CPU dense oracle"
    );
    for (index, ((actual, repeat), expected)) in
        first.iter().zip(&second).zip(&expected).enumerate()
    {
        let tolerance = 3e-3f32.max(expected.abs() * 3e-3);
        assert!(
            (actual - expected).abs() <= tolerance && actual == repeat,
            "output {index}: actual={actual} repeat={repeat} expected={expected}"
        );
    }

    let mut bindings = vec![
        executor
            .allocate_device_binding(
                "x".into(),
                None,
                DataType::Float32,
                vec![1, 32],
                vec![1, 32],
            )
            .unwrap(),
        executor
            .allocate_device_binding(
                "router".into(),
                None,
                DataType::Float32,
                vec![1, 2],
                vec![1, 2],
            )
            .unwrap(),
        executor
            .allocate_device_output_binding(
                "output".into(),
                DataType::Float32,
                vec![1, 32],
                vec![1, 32],
            )
            .unwrap(),
    ];
    let input_bytes = input_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let router_bytes = [3.0f32, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    bindings[0].write_bytes(0, &input_bytes).unwrap();
    bindings[1].write_bytes(0, &router_bytes).unwrap();
    executor
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    bindings[2].read_bytes_range(0, 4).unwrap();
    let (_, positive_allocations) = count_host_allocations(|| {
        let layout = Layout::from_size_align(64, 64).unwrap();
        unsafe {
            let ptr = std::alloc::alloc(layout);
            assert!(!ptr.is_null(), "positive-control allocation");
            std::alloc::dealloc(ptr, layout);
        }
    });
    assert_eq!(
        positive_allocations, 1,
        "host-allocation falsifier must observe the one intentional allocation"
    );
    assert_eq!(
        HOST_ALLOCATION_SIZES.with(Cell::get)[0],
        64,
        "host-allocation falsifier must report the intentional 64-byte layout"
    );
    eprintln!(
        "validation-allocation positive-control allocations={positive_allocations} sizes={:?}",
        HOST_ALLOCATION_SIZES.with(Cell::get)
    );
    let lock_before = runtime.graph_lifecycle_lock_acquisition_count();
    runtime.test_acquire_graph_lifecycle_lock().unwrap();
    assert!(
        runtime.graph_lifecycle_lock_acquisition_count() > lock_before,
        "graph-lock falsifier"
    );
    let alloc_before = runtime.allocation_counts();
    let transfer_before = runtime.transfer_counts();
    let sync_before = runtime.forced_synchronization_count();
    let control_ptr = runtime.alloc_raw(4).unwrap();
    assert!(
        runtime.allocation_counts().allocations > alloc_before.allocations,
        "CUDA allocation falsifier"
    );
    // SAFETY: `control_ptr` is a live four-byte allocation.
    unsafe {
        runtime.htod(&17u32.to_ne_bytes(), control_ptr).unwrap();
    }
    assert!(
        runtime.transfer_counts().host_to_device > transfer_before.host_to_device,
        "H2D transfer falsifier"
    );
    let mut control_value = [0u8; 4];
    // SAFETY: both source and destination cover four bytes.
    unsafe {
        runtime.dtoh(&mut control_value, control_ptr).unwrap();
    }
    assert_eq!(u32::from_ne_bytes(control_value), 17);
    assert!(
        runtime.forced_synchronization_count() > sync_before,
        "forced synchronization falsifier"
    );
    // SAFETY: `control_ptr` is still owned by this runtime and is freed once.
    unsafe {
        runtime.free_raw(control_ptr).unwrap();
    }
    runtime.test_drain_raw_pool();
    assert!(
        runtime.allocation_counts().frees > alloc_before.frees,
        "CUDA free falsifier"
    );

    let preparation_before = onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts();
    let mut preparation_control = Executor::build(
        sealed_bqmoe_session_graph(),
        Arc::new(WeightStore::new()),
        Arc::new(CudaExecutionProvider::new_default().unwrap()),
    )
    .unwrap();
    preparation_control
        .run(&[("x", &input), ("router", &router)])
        .unwrap();
    let preparation_after = onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts();
    assert!(
        preparation_after.format_parse_calls > preparation_before.format_parse_calls,
        "format-parse falsifier"
    );
    assert!(
        preparation_after.workspace_layout_builds > preparation_before.workspace_layout_builds,
        "workspace-layout falsifier"
    );
    drop(preparation_control);

    let allocations = runtime.allocation_counts();
    let transfers = runtime.transfer_counts();
    let synchronizations = runtime.forced_synchronization_count();
    let preparation = onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts();
    let locks = runtime.graph_lifecycle_lock_acquisition_count();
    let validation_registry_locks = runtime.validation_registry_lock_acquisition_count();
    let submissions = runtime.validation_submission_count();
    let (warmed, host_allocations) =
        count_host_allocations(|| executor.run_with_device_bindings(&[], &mut bindings));
    warmed.unwrap();
    assert_eq!(
        host_allocations,
        0,
        "warmed production Executor allocations: {:?}",
        HOST_ALLOCATION_SIZES.with(Cell::get)
    );
    assert_eq!(runtime.allocation_counts(), allocations);
    assert_eq!(runtime.transfer_counts(), transfers);
    assert_eq!(runtime.forced_synchronization_count(), synchronizations);
    assert_eq!(
        onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts(),
        preparation
    );
    assert_eq!(
        runtime.graph_lifecycle_lock_acquisition_count(),
        locks,
        "warmed production Executor graph locks"
    );
    assert_eq!(
        runtime.validation_registry_lock_acquisition_count(),
        validation_registry_locks,
        "warmed production Executor validation-registry locks"
    );
    assert_eq!(
        runtime.validation_submission_count() - submissions,
        1,
        "warmed production measurement must execute exactly one real validation submission"
    );
    eprintln!(
        "validation-allocation warmed-production allocations={host_allocations} submissions=1"
    );
    bindings[2].read_bytes_range(0, 4).unwrap();

    assert!(matches!(
        executor
            .try_capture_with_device_bindings(&[], &mut bindings)
            .unwrap(),
        DeviceGraphCaptureResult::Captured(_)
    ));
    let allocations = runtime.allocation_counts();
    let transfers = runtime.transfer_counts();
    let synchronizations = runtime.forced_synchronization_count();
    let preparation = onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts();
    let locks = runtime.graph_lifecycle_lock_acquisition_count();
    let validation_registry_locks = runtime.validation_registry_lock_acquisition_count();
    let submissions = runtime.validation_submission_count();
    let (replayed, host_allocations) =
        count_host_allocations(|| executor.replay_device_graph(&mut bindings));
    assert!(replayed.unwrap());
    assert_eq!(
        host_allocations,
        0,
        "production graph replay allocations: {:?}",
        HOST_ALLOCATION_SIZES.with(Cell::get)
    );
    assert_eq!(runtime.allocation_counts(), allocations);
    assert_eq!(runtime.transfer_counts(), transfers);
    assert_eq!(runtime.forced_synchronization_count(), synchronizations);
    assert_eq!(
        onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts(),
        preparation
    );
    assert_eq!(
        runtime.graph_lifecycle_lock_acquisition_count(),
        locks,
        "first captured launch lifecycle locks"
    );
    assert_eq!(
        runtime.validation_registry_lock_acquisition_count(),
        validation_registry_locks,
        "first captured launch validation-registry locks"
    );
    assert_eq!(
        runtime.validation_submission_count() - submissions,
        1,
        "first captured launch measurement must execute exactly one real submission"
    );
    eprintln!(
        "validation-allocation first-captured-launch allocations={host_allocations} submissions=1"
    );
    bindings[2].read_bytes_range(0, 4).unwrap();

    let allocations = runtime.allocation_counts();
    let transfers = runtime.transfer_counts();
    let synchronizations = runtime.forced_synchronization_count();
    let preparation = onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts();
    let locks = runtime.graph_lifecycle_lock_acquisition_count();
    let validation_registry_locks = runtime.validation_registry_lock_acquisition_count();
    let submissions = runtime.validation_submission_count();
    let (replayed, host_allocations) =
        count_host_allocations(|| executor.replay_device_graph(&mut bindings));
    assert!(replayed.unwrap());
    assert_eq!(
        host_allocations,
        0,
        "production graph replay allocations: {:?}",
        HOST_ALLOCATION_SIZES.with(Cell::get)
    );
    assert_eq!(runtime.allocation_counts(), allocations);
    assert_eq!(runtime.transfer_counts(), transfers);
    assert_eq!(runtime.forced_synchronization_count(), synchronizations);
    assert_eq!(
        onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts(),
        preparation
    );
    assert_eq!(
        runtime.graph_lifecycle_lock_acquisition_count(),
        locks,
        "production graph replay lifecycle locks"
    );
    assert_eq!(
        runtime.validation_registry_lock_acquisition_count(),
        validation_registry_locks,
        "production graph replay validation-registry locks"
    );
    assert_eq!(
        runtime.validation_submission_count() - submissions,
        1,
        "graph replay measurement must execute exactly one real submission"
    );
    eprintln!("validation-allocation replay allocations={host_allocations} submissions=1");
}

#[cfg(feature = "gpu-tests")]
#[test]
fn sealed_bqmoe_supported_observer_accumulates_eager_and_replay() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;

    const BANK_BYTES: u64 = 6_528;
    const ROUTE_BYTES: u64 = 6_528;

    let cuda = Arc::new(CudaExecutionProvider::new_default().unwrap());
    let mut session = crate::InferenceSession::from_graph_with_provider(
        sealed_bqmoe_session_graph(),
        Arc::new(WeightStore::new()),
        std::path::Path::new("."),
        cuda,
    )
    .unwrap();
    let mut bindings = vec![
        session
            .allocate_device_binding(
                "x",
                None::<String>,
                DataType::Float32,
                vec![1, 32],
                vec![1, 32],
            )
            .unwrap(),
        session
            .allocate_device_binding(
                "router",
                None::<String>,
                DataType::Float32,
                vec![1, 2],
                vec![1, 2],
            )
            .unwrap(),
        session
            .allocate_device_output_binding("output", DataType::Float32, vec![1, 32], vec![1, 32])
            .unwrap(),
    ];
    let input = (0..32)
        .flat_map(|index| ((index as f32 - 15.5) / 16.0).to_le_bytes())
        .collect::<Vec<_>>();
    let router = [3.0f32, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    bindings[0].write_bytes(0, &input).unwrap();
    bindings[1].write_bytes(0, &router).unwrap();
    session
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();

    let mut observer = session
        .observe_block_quantized_moe_traffic(crate::BlockQuantizedMoeTrafficConfig {
            request_id: 77,
        })
        .unwrap();
    let load = observer.snapshot().unwrap();
    assert_eq!(load.phase, crate::BlockQuantizedMoeTrafficPhase::Load);
    assert_eq!(load.request_id, 77);
    assert_eq!(load.traffic.uploaded_whole_bank_bytes, BANK_BYTES);
    assert_eq!(load.traffic.committed_whole_bank_bytes, BANK_BYTES);
    assert_eq!(load.traffic.logical_route_demand_bytes, 0);
    assert_eq!(load.traffic.unique_selected_expert_bytes, 0);
    assert_eq!(load.traffic.physical_dram_bytes, None);
    assert_eq!(load.traffic.page_ins, 0);
    assert_eq!(load.traffic.byte_hit_rate, None);

    observer
        .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
        .unwrap();
    observer
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    let eager = observer.snapshot().unwrap();
    assert_eq!(eager.phase, crate::BlockQuantizedMoeTrafficPhase::Decode);
    assert_eq!(eager.traffic.logical_route_demand_bytes, ROUTE_BYTES);
    assert_eq!(eager.traffic.unique_selected_expert_bytes, ROUTE_BYTES);

    observer
        .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
        .unwrap();
    assert!(matches!(
        observer
            .try_capture_with_device_bindings(&[], &mut bindings)
            .unwrap(),
        DeviceGraphCaptureResult::Captured(_)
    ));
    observer
        .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
        .unwrap();
    assert!(observer.replay_device_graph(&mut bindings).unwrap());
    let replay = observer.snapshot().unwrap();
    assert_eq!(replay.traffic.logical_route_demand_bytes, ROUTE_BYTES);
    assert_eq!(replay.traffic.unique_selected_expert_bytes, ROUTE_BYTES);
    assert!(observer.replay_device_graph(&mut bindings).unwrap());
    let repeated = observer.snapshot().unwrap();
    assert_eq!(repeated.traffic.logical_route_demand_bytes, ROUTE_BYTES * 2);
    assert_eq!(repeated.traffic.unique_selected_expert_bytes, ROUTE_BYTES);
    drop(observer);
    assert_eq!(session.captured_graph_segment_count(), 0);
    let observer = session
        .observe_block_quantized_moe_traffic(crate::BlockQuantizedMoeTrafficConfig {
            request_id: 78,
        })
        .unwrap();
    observer.finish().unwrap();
}

#[cfg(feature = "gpu-tests")]
#[test]
fn shared_provider_graphs_are_owner_scoped_repeatable_and_logical_shape_safe() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;

    let cuda = Arc::new(CudaExecutionProvider::new_default().unwrap());
    let (mut first, mut first_bindings, _runtime) =
        sealed_bqmoe_cuda_session_for_provider(Arc::clone(&cuda));
    let (mut second, mut second_bindings, _runtime) =
        sealed_bqmoe_cuda_session_for_provider(Arc::clone(&cuda));

    assert!(matches!(
        first
            .try_capture_with_device_bindings(&[], &mut first_bindings)
            .unwrap(),
        DeviceGraphCaptureResult::Captured(_)
    ));
    let first_token = first.exec.cap().device_graph_token.unwrap();
    assert!(matches!(
        second
            .try_capture_with_device_bindings(&[], &mut second_bindings)
            .unwrap(),
        DeviceGraphCaptureResult::Captured(_)
    ));
    let second_token = second.exec.cap().device_graph_token.unwrap();
    assert_ne!(first_token.owner(), second_token.owner());

    assert!(first.replay_device_graph(&mut first_bindings).unwrap());
    first_bindings[2].read_bytes_range(0, 4).unwrap();
    assert!(second.replay_device_graph(&mut second_bindings).unwrap());
    second_bindings[2].read_bytes_range(0, 4).unwrap();

    let wrong_owner = onnx_runtime_ep_api::DeviceGraphToken::new(
        onnx_runtime_ep_api::DeviceGraphOwner::new(),
        first_token.slot(),
        first_token.generation(),
    );
    let error = cuda.replay_owned_device_graph(wrong_owner).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("retired executor-owned CUDA graph"),
        "wrong-owner replay must fail closed: {error}"
    );

    assert!(matches!(
        first
            .try_capture_with_device_bindings(&[], &mut first_bindings)
            .unwrap(),
        DeviceGraphCaptureResult::Captured(_)
    ));
    let recaptured = first.exec.cap().device_graph_token.unwrap();
    assert_eq!(recaptured.owner(), first_token.owner());
    assert!(
        recaptured.generation() > first_token.generation(),
        "repeated capture must mint a new installation generation"
    );
    assert!(second.replay_device_graph(&mut second_bindings).unwrap());
    second_bindings[2].read_bytes_range(0, 4).unwrap();

    first_bindings[2].set_logical_shape(vec![1, 16]).unwrap();
    let error = first.replay_device_graph(&mut first_bindings).unwrap_err();
    assert!(
        error.to_string().contains("logical/physical shape"),
        "same-allocation logical shape change must invalidate replay: {error}"
    );
    assert_eq!(first.exec.cap().device_graph_token, None);

    first_bindings[2].set_logical_shape(vec![1, 32]).unwrap();
    assert!(matches!(
        first
            .try_capture_with_device_bindings(&[], &mut first_bindings)
            .unwrap(),
        DeviceGraphCaptureResult::Captured(_)
    ));
    assert!(first.replay_device_graph(&mut first_bindings).unwrap());
    first_bindings[2].read_bytes_range(0, 4).unwrap();
}

#[cfg(feature = "gpu-tests")]
#[test]
fn supported_observer_rejects_every_corrupt_device_record_field() {
    use onnx_runtime_ep_cuda::kernels::block_quantized_moe::BlockQuantizedMoeTrafficFaultForTest::{
        ForeignRequest, NonTopKMultipleCount, Overflow, Poison, StaleEpoch, WrongDevice,
    };
    const ROUTE_BYTES: u64 = 6_528;

    let (mut session, mut bindings, _runtime) = sealed_bqmoe_cuda_session_fixture();
    let mut observer = session
        .observe_block_quantized_moe_traffic(crate::BlockQuantizedMoeTrafficConfig {
            request_id: 91,
        })
        .unwrap();

    for (fault, expected) in [
        (Poison, "poison"),
        (Overflow, "overflow"),
        (StaleEpoch, "epoch mismatch"),
        (ForeignRequest, "request mismatch"),
        (WrongDevice, "device mismatch"),
        (NonTopKMultipleCount, "impossible"),
    ] {
        observer
            .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
            .unwrap();
        observer
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        let clean = observer.snapshot().unwrap();
        assert_eq!(
            clean.traffic.logical_route_demand_bytes, ROUTE_BYTES,
            "{fault:?}: positive control must publish the validated top-k route extent"
        );
        observer.inject_fault_for_test(fault).unwrap();
        let error = observer.snapshot().unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "{fault:?} must fail through the public typed observer: {error}"
        );
    }

    observer
        .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
        .unwrap();
    observer.finish().unwrap();
}

#[cfg(feature = "gpu-tests")]
#[test]
fn shared_cuda_provider_foreign_teardown_preserves_owned_validation_error() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;

    #[derive(Clone, Copy, Debug)]
    enum ForeignAction {
        BindingDrop,
        ExecutorDrop,
        GraphReset,
    }

    for action in [
        ForeignAction::BindingDrop,
        ForeignAction::ExecutorDrop,
        ForeignAction::GraphReset,
    ] {
        let cuda = Arc::new(CudaExecutionProvider::new_default().unwrap());
        let (mut owner, mut owner_bindings, runtime) =
            sealed_bqmoe_cuda_session_for_provider(Arc::clone(&cuda));
        let (foreign, mut foreign_bindings, _) =
            sealed_bqmoe_cuda_session_for_provider(Arc::clone(&cuda));

        owner
            .run_with_device_bindings(&[], &mut owner_bindings)
            .expect("the owner must submit one deferred device-bound run");
        let flags = 0x40u32.to_ne_bytes();
        // SAFETY: GPU-test-only fault injection into the runtime's live
        // validation word, ordered after the owner's submitted work.
        unsafe {
            runtime.htod(&flags, runtime.capture_error_ptr()).unwrap();
        }

        let sync_before = runtime.forced_synchronization_count();
        let mut foreign = Some(foreign);
        match action {
            ForeignAction::BindingDrop => drop(foreign_bindings.remove(2)),
            ForeignAction::ExecutorDrop => drop(foreign.take()),
            ForeignAction::GraphReset => {
                assert!(!foreign.as_mut().unwrap().reset_device_graph().unwrap());
            }
        }
        assert!(
            runtime.forced_synchronization_count() > sync_before,
            "{action:?}: positive control must prove the foreign teardown/reset synchronized"
        );

        for observation in 1..=2 {
            let error = owner_bindings[2].read_bytes_range(0, 4).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("device validation failed (flags=0x40)"),
                "{action:?} observation {observation}: the owner's sticky failure must survive: \
                 {error}"
            );
        }
        drop(owner_bindings);
        drop(owner);
        drop(foreign_bindings);
        drop(foreign);
        assert_eq!(
            runtime.registered_validation_owner_count(),
            0,
            "{action:?}: executor/binding teardown must retire every owner-scoped slot"
        );
    }
}

#[cfg(feature = "gpu-tests")]
#[test]
fn isolated_raw_reset_cannot_clear_an_active_owner_generation() {
    let (mut owner, mut bindings, runtime) = sealed_bqmoe_cuda_session_fixture();
    owner
        .run_with_device_bindings(&[], &mut bindings)
        .expect("owner submits one deferred run");
    let flags = 0x40u32.to_ne_bytes();
    unsafe {
        runtime.htod(&flags, runtime.capture_error_ptr()).unwrap();
    }
    let reset_error = unsafe { runtime.reset_capture_error_for_isolated_test() }
        .expect_err("even the test-only raw reset must reject an active generation");
    assert!(
        reset_error.to_string().contains("phase Active"),
        "reset refusal must identify the active authority: {reset_error}"
    );
    let error = bindings[2].read_bytes_range(0, 4).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("device validation failed (flags=0x40)"),
        "refused raw reset must preserve the owner's pending failure: {error}"
    );
}

#[cfg(feature = "gpu-tests")]
#[test]
fn concurrent_consume_and_executor_drop_share_exactly_one_cleanup() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;

    const ITERATIONS: u64 = 16;
    let cuda = Arc::new(CudaExecutionProvider::new_default().unwrap());
    let runtime = Arc::clone(cuda.runtime());
    for iteration in 0..ITERATIONS {
        let (mut owner, mut bindings, _) =
            sealed_bqmoe_cuda_session_for_provider(Arc::clone(&cuda));
        let cleanup_before = runtime.validation_cleanup_count();
        owner
            .run_with_device_bindings(&[], &mut bindings)
            .expect("owner submits one deferred run");
        unsafe {
            runtime
                .htod(&0x40u32.to_ne_bytes(), runtime.capture_error_ptr())
                .unwrap();
        }

        runtime.pause_validation_consumer_for_test(true);
        let mut output = bindings.remove(2);
        let consumer = std::thread::spawn(move || {
            let first = output.read_bytes_range(0, 4).unwrap_err().to_string();
            let second = output.read_bytes_range(0, 4).unwrap_err().to_string();
            drop(output);
            (first, second)
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !runtime.validation_consumer_claimed_for_test() {
            assert!(
                std::time::Instant::now() < deadline,
                "iteration {iteration}: binding consumer did not claim validation cleanup \
                 authority"
            );
            std::thread::yield_now();
        }

        let dropper = std::thread::spawn(move || drop(owner));
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(
            !dropper.is_finished(),
            "iteration {iteration}: executor Drop must wait for the legitimate consuming \
             recipient"
        );
        runtime.pause_validation_consumer_for_test(false);
        dropper.join().expect("executor Drop must not panic");
        let (first, second) = consumer.join().expect("binding consumer must not panic");
        for (observation, error) in [(1, first), (2, second)] {
            assert!(
                error.contains("device validation failed (flags=0x40)"),
                "iteration {iteration} sticky observation {observation} lost the exact result: \
                 {error}"
            );
        }
        drop(bindings);
        assert_eq!(
            runtime.validation_cleanup_count() - cleanup_before,
            1,
            "iteration {iteration}: the Active -> Consuming CAS must assign exactly one cleanup"
        );
        assert_eq!(
            runtime.registered_validation_owner_count(),
            0,
            "iteration {iteration}: concurrent consume-vs-Drop leaked owner slots"
        );
    }
    assert_eq!(
        runtime.registered_validation_owner_count(),
        0,
        "concurrent consume-vs-Drop must retire every owner slot"
    );
}

#[cfg(feature = "gpu-tests")]
#[test]
fn isolated_reset_and_begin_linearize_through_one_atomic_authority() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;

    let cuda = Arc::new(CudaExecutionProvider::new_default().unwrap());
    let runtime = Arc::clone(cuda.runtime());
    let mut registration = cuda.register_device_validation_owner().unwrap();
    const ITERATIONS: u64 = 16;
    let cleanup_before = runtime.validation_cleanup_count();
    for iteration in 0..ITERATIONS {
        runtime.pause_validation_reset_for_test(true);
        let reset_runtime = Arc::clone(&runtime);
        let resetter = std::thread::spawn(move || unsafe {
            reset_runtime.reset_capture_error_for_isolated_test()
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !runtime.validation_reset_claimed_for_test() {
            assert!(
                std::time::Instant::now() < deadline,
                "iteration {iteration}: isolated reset did not claim the coordinator"
            );
            std::thread::yield_now();
        }
        let begin_error = cuda
            .begin_device_validation(&registration)
            .expect_err("begin must fail closed while reset owns the atomic authority");
        assert!(
            begin_error.to_string().contains("Resetting"),
            "iteration {iteration}: begin refusal must identify the conflicting phase: \
             {begin_error}"
        );
        runtime.pause_validation_reset_for_test(false);
        resetter
            .join()
            .expect("isolated reset thread must not panic")
            .expect("the reset that linearized first must complete");

        let token = cuda.begin_device_validation(&registration).unwrap();
        cuda.activate_device_validation(token).unwrap();
        unsafe {
            runtime
                .htod(&0x40u32.to_ne_bytes(), runtime.capture_error_ptr())
                .unwrap();
        }
        let reset_error = unsafe { runtime.reset_capture_error_for_isolated_test() }
            .expect_err("reset must fail closed after begin/activation linearizes");
        assert!(
            reset_error.to_string().contains("Active"),
            "iteration {iteration}: reset refusal must identify the active generation: \
             {reset_error}"
        );
        cuda.sync().unwrap();
        assert_eq!(
            cuda.consume_device_validation_error(&registration, token)
                .unwrap(),
            0x40,
            "iteration {iteration}: failed reset must never clear active work"
        );
        assert_eq!(
            runtime.check_capture_error().unwrap(),
            0,
            "iteration {iteration}: consuming cleanup must leave the latch clear"
        );
    }
    assert_eq!(
        runtime.validation_cleanup_count() - cleanup_before,
        ITERATIONS,
        "each activated generation must have exactly one cleanup"
    );
    cuda.unregister_device_validation_owner(&mut registration)
        .unwrap();
    assert_eq!(runtime.registered_validation_owner_count(), 0);
}

#[cfg(feature = "gpu-tests")]
#[test]
fn cuda_owner_slot_preserves_old_receipt_across_96_later_submissions() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;

    const LATER_SUBMISSIONS: u64 = 96;
    let cuda = CudaExecutionProvider::new_default().unwrap();
    let runtime = cuda.runtime();
    let mut old_owner = cuda.register_device_validation_owner().unwrap();
    let mut old_sibling = cuda.register_device_validation_owner().unwrap();

    let old_submitter_token = cuda.begin_device_validation(&old_owner).unwrap();
    let old_sibling_token = cuda
        .add_device_validation_recipient(old_submitter_token, &old_sibling)
        .unwrap();
    cuda.activate_device_validation(old_submitter_token)
        .unwrap();
    let flags = 0x40u32.to_ne_bytes();
    unsafe {
        runtime.htod(&flags, runtime.capture_error_ptr()).unwrap();
    }
    cuda.sync().unwrap();
    assert_eq!(
        cuda.consume_device_validation_error(&old_sibling, old_sibling_token)
            .unwrap(),
        0x40
    );

    for _ in 0..LATER_SUBMISSIONS {
        let mut owner = cuda.register_device_validation_owner().unwrap();
        let token = cuda.begin_device_validation(&owner).unwrap();
        cuda.activate_device_validation(token).unwrap();
        cuda.sync().unwrap();
        assert_eq!(
            cuda.consume_device_validation_error(&owner, token).unwrap(),
            0
        );
        cuda.unregister_device_validation_owner(&mut owner).unwrap();
    }

    assert_eq!(
        cuda.consume_device_validation_error(&old_owner, old_submitter_token)
            .unwrap(),
        0x40,
        "the old exact receipt must not be overwritten by later owners"
    );
    let replacement = cuda.begin_device_validation(&old_owner).unwrap();
    cuda.activate_device_validation(replacement).unwrap();
    let stale = cuda
        .consume_device_validation_error(&old_owner, old_submitter_token)
        .expect_err("an overwritten owner generation must fail closed");
    assert!(stale.to_string().contains("stale"));
    cuda.sync().unwrap();
    assert_eq!(
        cuda.consume_device_validation_error(&old_owner, replacement)
            .unwrap(),
        0,
        "stale-token rejection must not consume or poison the replacement generation"
    );
    assert_eq!(
        cuda.consume_device_validation_error(&old_sibling, old_sibling_token)
            .unwrap(),
        0x40,
        "reusing one owner slot must not overwrite its sibling's independent completed slot"
    );
    cuda.unregister_device_validation_owner(&mut old_sibling)
        .unwrap();
    cuda.unregister_device_validation_owner(&mut old_owner)
        .unwrap();
    assert_eq!(runtime.registered_validation_owner_count(), 0);
    assert_eq!(runtime.validation_submission_count(), LATER_SUBMISSIONS + 2);
    eprintln!(
        "cuda-validation-lifetime old_receipt_flags=0x40 later_submissions={LATER_SUBMISSIONS} \
         leaked_owner_slots=0"
    );
}

#[cfg(feature = "gpu-tests")]
#[test]
fn observer_reset_finish_drop_and_panic_consume_pending_validation() {
    fn poison_validation(runtime: &onnx_runtime_ep_cuda::runtime::CudaRuntime) {
        let flags = 0x40u32.to_ne_bytes();
        // SAFETY: the runtime owns a live four-byte validation word for its
        // lifetime; this GPU-test-only injection emulates an asynchronous
        // kernel reporting a bounds violation after submission.
        unsafe {
            runtime.htod(&flags, runtime.capture_error_ptr()).unwrap();
        }
    }

    let (mut session, mut bindings, runtime) = sealed_bqmoe_cuda_session_fixture();

    let mut reset_observer = session
        .observe_block_quantized_moe_traffic(crate::BlockQuantizedMoeTrafficConfig {
            request_id: 101,
        })
        .unwrap();
    reset_observer
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    poison_validation(&runtime);
    let reset_error = reset_observer
        .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
        .unwrap_err();
    assert!(reset_error.to_string().contains("flags=0x40"));
    reset_observer
        .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
        .unwrap();
    reset_observer.finish().unwrap();

    let mut finish_observer = session
        .observe_block_quantized_moe_traffic(crate::BlockQuantizedMoeTrafficConfig {
            request_id: 102,
        })
        .unwrap();
    finish_observer
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    poison_validation(&runtime);
    let finish_error = finish_observer.finish().unwrap_err();
    assert!(finish_error.to_string().contains("flags=0x40"));

    {
        let mut drop_observer = session
            .observe_block_quantized_moe_traffic(crate::BlockQuantizedMoeTrafficConfig {
                request_id: 103,
            })
            .unwrap();
        drop_observer
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        poison_validation(&runtime);
    }

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut panic_observer = session
            .observe_block_quantized_moe_traffic(crate::BlockQuantizedMoeTrafficConfig {
                request_id: 104,
            })
            .unwrap();
        panic_observer
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        poison_validation(&runtime);
        panic!("forced observer unwind");
    }));
    assert!(panic.is_err());

    let mut healthy = session
        .observe_block_quantized_moe_traffic(crate::BlockQuantizedMoeTrafficConfig {
            request_id: 105,
        })
        .expect("drop and unwind cleanup must leave the session reusable");
    healthy
        .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
        .unwrap();
    healthy
        .run_with_device_bindings(&[], &mut bindings)
        .unwrap();
    healthy.snapshot().unwrap();
    healthy.finish().unwrap();
}

#[cfg(feature = "gpu-tests")]
#[test]
#[ignore = "opt-in official GLM-5.2 checkpoint proof; set ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT"]
fn glm52_real_gated_top8_runs_through_production_session_with_f64_oracle() {
    use onnx_runtime_ep_cpu::kernels::block_quantized_moe::{
        BLOCK_QUANT_MOE_DENSE_EXPANSIONS, decode_expert_projection_f64,
    };
    use onnx_runtime_ep_cuda::CudaExecutionProvider;

    const H: usize = 6144;
    const I: usize = 2048;
    const EXPERTS: usize = 256;
    const K: usize = 8;

    struct Case {
        label: &'static str,
        shard: &'static str,
        gate_format: &'static str,
        down_format: &'static str,
        gate_offset: usize,
        up_offset: usize,
        down_offset: usize,
        expected_per_expert: u64,
    }

    fn format_info(format: &str) -> (usize, usize) {
        match format {
            "iq1_s" => (256, 50),
            "iq2_xxs" => (256, 66),
            "iq3_xxs" => (256, 98),
            "iq4_xs" => (256, 136),
            "q2_k" => (256, 84),
            "q3_k" => (256, 110),
            _ => panic!("unexpected real-checkpoint format {format}"),
        }
    }

    fn projection_len(format: &str, out_features: usize, in_features: usize) -> usize {
        let (qk, block_bytes) = format_info(format);
        EXPERTS * out_features * (in_features / qk) * block_bytes
    }

    fn real_graph(path: &std::path::Path, case: &Case) -> (Graph, [ValueId; 3]) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let rows = graph.intern_symbol("rows");
        let x = graph.create_named_value(
            "x",
            DataType::Float32,
            vec![Dim::Symbolic(rows), Dim::Static(H)],
        );
        graph.add_input(x);
        let router = graph.create_named_value(
            "router",
            DataType::Float32,
            vec![Dim::Symbolic(rows), Dim::Static(EXPERTS)],
        );
        graph.add_input(router);
        let add_external = |graph: &mut Graph,
                            name: &str,
                            format: &str,
                            offset: usize,
                            out_features: usize,
                            in_features: usize| {
            let (qk, block_bytes) = format_info(format);
            let dims = vec![EXPERTS, out_features, in_features / qk, block_bytes];
            let value =
                graph.create_named_value(name, DataType::Uint8, static_shape(dims.iter().copied()));
            graph.set_initializer(
                value,
                WeightRef::External {
                    path: path.to_path_buf(),
                    offset,
                    length: projection_len(format, out_features, in_features),
                    dtype: DataType::Uint8,
                    dims,
                },
            );
            value
        };
        let gate = add_external(&mut graph, "gate", case.gate_format, case.gate_offset, I, H);
        let down = add_external(&mut graph, "down", case.down_format, case.down_offset, H, I);
        let up = add_external(&mut graph, "up", case.gate_format, case.up_offset, I, H);
        let output = graph.create_named_value(
            "output",
            DataType::Float32,
            vec![Dim::Symbolic(rows), Dim::Static(H)],
        );
        let mut node = Node::new(
            NodeId(0),
            "BlockQuantizedMoE",
            vec![
                Some(x),
                Some(router),
                Some(gate),
                None,
                Some(down),
                None,
                Some(up),
            ],
            vec![output],
        );
        node.domain = "pkg.nxrt".into();
        node.attributes.insert("k".into(), Attribute::Int(K as i64));
        node.attributes.insert(
            "activation_type".into(),
            Attribute::String(b"silu".to_vec()),
        );
        node.attributes
            .insert("normalize_routing_weights".into(), Attribute::Int(1));
        node.attributes
            .insert("swiglu_fusion".into(), Attribute::Int(0));
        node.attributes.insert(
            "fc1_format".into(),
            Attribute::String(case.gate_format.as_bytes().to_vec()),
        );
        node.attributes.insert(
            "fc2_format".into(),
            Attribute::String(case.down_format.as_bytes().to_vec()),
        );
        node.attributes.insert(
            "fc3_format".into(),
            Attribute::String(case.gate_format.as_bytes().to_vec()),
        );
        node.attributes
            .insert("block_layout_version".into(), Attribute::Int(1));
        graph.insert_node(node);
        graph.add_output(output);
        (graph, [gate, down, up])
    }

    fn router_for(ids: &[usize; K]) -> Vec<f32> {
        let mut logits = vec![-50.0f32; EXPERTS];
        for (rank, &expert) in ids.iter().enumerate() {
            logits[expert] = (K - rank) as f32;
        }
        logits
    }

    fn router_rows<const ROWS: usize>(routes: [&[usize; K]; ROWS]) -> Vec<f32> {
        routes.into_iter().flat_map(router_for).collect()
    }

    fn f64_oracle(
        case: &Case,
        store: &WeightStore,
        graph: &Graph,
        weights: [ValueId; 3],
        input: &[f32],
        ids: &[usize; K],
    ) -> Vec<f64> {
        let banks = weights.map(|value| {
            store
                .bytes(graph.initializers.get(&value).unwrap())
                .unwrap()
        });
        let gate_stride = projection_len(case.gate_format, I, H) / EXPERTS;
        let down_stride = projection_len(case.down_format, H, I) / EXPERTS;
        let logits = router_for(ids);
        let maximum = ids
            .iter()
            .map(|&expert| f64::from(logits[expert]))
            .fold(f64::NEG_INFINITY, f64::max);
        let denominator: f64 = ids
            .iter()
            .map(|&expert| (f64::from(logits[expert]) - maximum).exp())
            .sum();
        let input = input.iter().copied().map(f64::from).collect::<Vec<_>>();
        let mut output = vec![0.0f64; H];
        for &expert in ids {
            let gate = decode_expert_projection_f64(
                case.gate_format,
                &banks[0][expert * gate_stride..(expert + 1) * gate_stride],
                I,
                H,
            )
            .unwrap();
            let up = decode_expert_projection_f64(
                case.gate_format,
                &banks[2][expert * gate_stride..(expert + 1) * gate_stride],
                I,
                H,
            )
            .unwrap();
            let mut activated = vec![0.0f64; I];
            for (feature, activated_value) in activated.iter_mut().enumerate() {
                let row = feature * H;
                let gate_value = gate[row..row + H]
                    .iter()
                    .zip(&input)
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>();
                let up_value = up[row..row + H]
                    .iter()
                    .zip(&input)
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>();
                *activated_value = gate_value / (1.0 + (-gate_value).exp()) * up_value;
            }
            drop(gate);
            drop(up);
            let down = decode_expert_projection_f64(
                case.down_format,
                &banks[1][expert * down_stride..(expert + 1) * down_stride],
                H,
                I,
            )
            .unwrap();
            let route_weight = (f64::from(logits[expert]) - maximum).exp() / denominator;
            for (feature, output_value) in output.iter_mut().enumerate() {
                let row = feature * I;
                let value = down[row..row + I]
                    .iter()
                    .zip(&activated)
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>();
                *output_value += route_weight * value;
            }
        }
        output
    }

    let root = std::path::PathBuf::from(
        std::env::var("ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT")
            .expect("set ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT"),
    );
    let cases = [
        Case {
            label: "layer56-iq1s-iq3xxs",
            shard: "GLM-5.2-UD-IQ1_S-00005-of-00006.gguf",
            gate_format: "iq1_s",
            down_format: "iq3_xxs",
            gate_offset: 2_668_820_832,
            up_offset: 3_312_933_216,
            down_offset: 1_425_373_536,
            expected_per_expert: 9_732_096,
        },
        Case {
            label: "layer74-iq2xxs-iq3xxs",
            shard: "GLM-5.2-UD-IQ1_S-00006-of-00006.gguf",
            gate_format: "iq2_xxs",
            down_format: "iq3_xxs",
            gate_offset: 3_071_455_584,
            up_offset: 3_916_894_560,
            down_offset: 1_828_008_288,
            expected_per_expert: 11_304_960,
        },
        Case {
            label: "layer8-iq2xxs-iq4xs",
            shard: "GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
            gate_format: "iq2_xxs",
            down_format: "iq4_xs",
            gate_offset: 17_617_400_576,
            up_offset: 18_464_510_720,
            down_offset: 15_892_755_200,
            expected_per_expert: 13_172_736,
        },
        Case {
            label: "layer78-q2k-q3k",
            shard: "GLM-5.2-UD-IQ1_S-00006-of-00006.gguf",
            gate_format: "q2_k",
            down_format: "q3_k",
            gate_offset: 16_942_690_656,
            up_offset: 18_014_622_048,
            down_offset: 15_548_248_416,
            expected_per_expert: 13_664_256,
        },
    ];
    let input_values = (0..H)
        .map(|index| ((index * 13 % 31) as f32 - 15.0) / 32.0)
        .collect::<Vec<_>>();
    let low = [0, 1, 2, 3, 4, 5, 6, 7];
    let high = [248, 249, 250, 251, 252, 253, 254, 255];
    let case_filter = std::env::var("ONNX_GENAI_GLM52_CASE").ok();

    for case in cases {
        if case_filter
            .as_deref()
            .is_some_and(|filter| filter != case.label)
        {
            continue;
        }
        let path = root.join(case.shard);
        let (graph, weights) = real_graph(&path, &case);
        let mut store = WeightStore::new();
        store.map_external(&path).unwrap();
        let oracle_graph = graph.clone();
        let store = Arc::new(store);
        let oracle_store = Arc::clone(&store);
        let cuda = Arc::new(CudaExecutionProvider::new_default().unwrap());
        let runtime = Arc::clone(cuda.runtime());
        let dense_before = BLOCK_QUANT_MOE_DENSE_EXPANSIONS.load(Ordering::Relaxed);
        let mut session =
            crate::InferenceSession::from_graph_with_provider(graph, store, &root, cuda.clone())
                .unwrap();
        let assert_traffic = |phase: &str,
                              traffic: onnx_runtime_ep_api::BlockQuantizedMoeTraffic,
                              routes: u64,
                              unique_experts: u64| {
            assert_eq!(
                traffic.uploaded_whole_bank_bytes,
                case.expected_per_expert * EXPERTS as u64,
                "{} {phase} uploaded whole-bank extent",
                case.label
            );
            assert_eq!(
                traffic.committed_whole_bank_bytes,
                case.expected_per_expert * EXPERTS as u64,
                "{} {phase} committed whole-bank extent",
                case.label
            );
            assert_eq!(
                traffic.logical_route_demand_bytes,
                case.expected_per_expert * routes,
                "{} {phase} logical route demand",
                case.label
            );
            assert_eq!(
                traffic.unique_selected_expert_bytes,
                case.expected_per_expert * unique_experts,
                "{} {phase} unique selected-expert extent",
                case.label
            );
            assert_eq!(traffic.physical_dram_bytes, None);
            assert_eq!(traffic.page_ins, 0);
            assert_eq!(traffic.byte_hit_rate, None);
        };

        let mut prefill_bindings = vec![
            session
                .allocate_device_binding(
                    "x",
                    None::<String>,
                    DataType::Float32,
                    vec![2, H],
                    vec![2, H],
                )
                .unwrap(),
            session
                .allocate_device_binding(
                    "router",
                    None::<String>,
                    DataType::Float32,
                    vec![2, EXPERTS],
                    vec![2, EXPERTS],
                )
                .unwrap(),
            session
                .allocate_device_output_binding("output", DataType::Float32, vec![2, H], vec![2, H])
                .unwrap(),
        ];
        let prefill_input_bytes = (0..2)
            .flat_map(|_| input_values.iter())
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let repeated_router_bytes = router_rows([&low, &low])
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        prefill_bindings[0]
            .write_bytes(0, &prefill_input_bytes)
            .unwrap();
        prefill_bindings[1]
            .write_bytes(0, &repeated_router_bytes)
            .unwrap();
        let mut bindings = vec![
            session
                .allocate_device_binding(
                    "x",
                    None::<String>,
                    DataType::Float32,
                    vec![1, H],
                    vec![1, H],
                )
                .unwrap(),
            session
                .allocate_device_binding(
                    "router",
                    None::<String>,
                    DataType::Float32,
                    vec![1, EXPERTS],
                    vec![1, EXPERTS],
                )
                .unwrap(),
            session
                .allocate_device_output_binding("output", DataType::Float32, vec![1, H], vec![1, H])
                .unwrap(),
        ];
        let input_bytes = input_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let low_bytes = router_for(&low)
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        bindings[0].write_bytes(0, &input_bytes).unwrap();
        bindings[1].write_bytes(0, &low_bytes).unwrap();

        let before_admission_transfers = runtime.transfer_counts();
        session
            .run_with_device_bindings(&[], &mut prefill_bindings)
            .unwrap();
        prefill_bindings[2].read_bytes_range(0, 4).unwrap();
        session
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        let after_admission_transfers = runtime.transfer_counts();
        assert_eq!(
            after_admission_transfers.host_to_device - before_admission_transfers.host_to_device,
            3,
            "{} must upload only the three sealed projection banks",
            case.label
        );
        assert_eq!(
            after_admission_transfers.async_host_to_device,
            before_admission_transfers.async_host_to_device,
            "{} sealed banks must not enter the paging transfer path",
            case.label
        );

        let mut observer = session
            .observe_block_quantized_moe_traffic(crate::BlockQuantizedMoeTrafficConfig {
                request_id: 1,
            })
            .unwrap();
        let load_snapshot = observer.snapshot().unwrap();
        assert_eq!(load_snapshot.request_id, 1);
        let load_traffic = load_snapshot.traffic;
        assert_traffic("load", load_traffic, 0, 0);
        observer
            .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Warmup)
            .unwrap();
        observer
            .warmup(&[
                crate::WarmupShape {
                    input_name: "x".into(),
                    shape: vec![2, H],
                },
                crate::WarmupShape {
                    input_name: "router".into(),
                    shape: vec![2, EXPERTS],
                },
            ])
            .unwrap();
        let warmup = observer.snapshot().unwrap();
        assert_eq!(warmup.phase, crate::BlockQuantizedMoeTrafficPhase::Warmup);
        let warmup_traffic = warmup.traffic;
        assert_traffic("warmup", warmup_traffic, 0, 0);
        observer
            .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Prefill)
            .unwrap();
        observer
            .run_with_device_bindings(&[], &mut prefill_bindings)
            .unwrap();
        let output_values = |binding: &mut DeviceIoBinding| {
            binding
                .read_bytes()
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let repeated_prefill = output_values(&mut prefill_bindings[2]);
        assert_eq!(
            &repeated_prefill[..H],
            &repeated_prefill[H..],
            "{} repeated-route prefill rows must be deterministic",
            case.label
        );
        let repeated_snapshot = observer.snapshot().unwrap();
        assert_eq!(
            repeated_snapshot.phase,
            crate::BlockQuantizedMoeTrafficPhase::Prefill
        );
        let prefill_repeated_traffic = repeated_snapshot.traffic;
        assert_traffic("prefill repeated", prefill_repeated_traffic, 16, 8);
        observer
            .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Prefill)
            .unwrap();
        let broad_router_bytes = router_rows([&low, &high])
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        prefill_bindings[1]
            .write_bytes(0, &broad_router_bytes)
            .unwrap();
        observer
            .run_with_device_bindings(&[], &mut prefill_bindings)
            .unwrap();
        let broad_prefill = output_values(&mut prefill_bindings[2]);
        assert_eq!(
            &repeated_prefill[..H],
            &broad_prefill[..H],
            "{} low-ID row must not depend on the other row's routes",
            case.label
        );
        let prefill_broad_traffic = observer.snapshot().unwrap().traffic;
        assert_traffic("prefill broad", prefill_broad_traffic, 16, 16);
        assert_eq!(
            prefill_repeated_traffic.logical_route_demand_bytes,
            prefill_broad_traffic.logical_route_demand_bytes,
            "{} repeated and broad routes have equal logical demand",
            case.label
        );
        assert!(
            prefill_repeated_traffic.unique_selected_expert_bytes
                < prefill_broad_traffic.unique_selected_expert_bytes,
            "{} repeated routes must have a smaller unique extent",
            case.label
        );
        drop(prefill_bindings);

        observer
            .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
            .unwrap();
        let decode_load_traffic = observer.snapshot().unwrap().traffic;
        assert_traffic("decode load", decode_load_traffic, 0, 0);
        assert_eq!(
            decode_load_traffic.uploaded_whole_bank_bytes, load_traffic.uploaded_whole_bank_bytes,
            "{} decode shape must adopt the prefill shape's sealed banks",
            case.label
        );
        observer
            .run_with_device_bindings(&[], &mut bindings)
            .unwrap();
        let eager_low = output_values(&mut bindings[2]);
        observer
            .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
            .unwrap();
        let capture = observer
            .try_capture_with_device_bindings(&[], &mut bindings)
            .unwrap();
        if !matches!(capture, DeviceGraphCaptureResult::Captured(_)) {
            panic!("{} production session capture was declined", case.label);
        }
        observer
            .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
            .unwrap();
        assert!(observer.replay_device_graph(&mut bindings).unwrap());
        let actual_low = output_values(&mut bindings[2]);
        let decode_single_traffic = observer.snapshot().unwrap().traffic;
        assert_traffic("decode single", decode_single_traffic, 8, 8);
        assert_eq!(
            eager_low, actual_low,
            "{} eager/captured parity",
            case.label
        );
        assert!(observer.replay_device_graph(&mut bindings).unwrap());
        let repeat_low = output_values(&mut bindings[2]);
        let decode_repeated_traffic = observer.snapshot().unwrap().traffic;
        assert_traffic("decode repeated", decode_repeated_traffic, 16, 8);
        observer
            .reset_phase(crate::BlockQuantizedMoeTrafficPhase::Decode)
            .unwrap();
        assert!(observer.replay_device_graph(&mut bindings).unwrap());
        let _ = output_values(&mut bindings[2]);
        let high_bytes = router_for(&high)
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        bindings[1].write_bytes(0, &high_bytes).unwrap();
        assert!(observer.replay_device_graph(&mut bindings).unwrap());
        let actual_high = output_values(&mut bindings[2]);
        let decode_broad_traffic = observer.snapshot().unwrap().traffic;
        assert_traffic("decode broad", decode_broad_traffic, 16, 16);
        assert_eq!(
            decode_repeated_traffic.logical_route_demand_bytes,
            decode_broad_traffic.logical_route_demand_bytes
        );
        assert!(
            decode_repeated_traffic.unique_selected_expert_bytes
                < decode_broad_traffic.unique_selected_expert_bytes
        );
        assert_eq!(actual_low, repeat_low, "{} repeatability", case.label);
        assert!(
            observer.session.captured_graph_segment_count() > 0,
            "{} must capture the production path",
            case.label
        );
        assert_eq!(
            BLOCK_QUANT_MOE_DENSE_EXPANSIONS.load(Ordering::Relaxed),
            dense_before,
            "{} must not enter the CPU expert fallback",
            case.label
        );
        let expected_low = f64_oracle(
            &case,
            &oracle_store,
            &oracle_graph,
            weights,
            &input_values,
            &low,
        );
        let expected_high = f64_oracle(
            &case,
            &oracle_store,
            &oracle_graph,
            weights,
            &input_values,
            &high,
        );
        for (label, actual, expected) in [
            ("low", &actual_low, &expected_low),
            ("high", &actual_high, &expected_high),
        ] {
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                let tolerance = 2e-2f64.max(expected.abs() * 8e-3);
                assert!(
                    (f64::from(actual) - expected).abs() <= tolerance,
                    "{} {label} output {index}: actual={actual} expected={expected} tolerance={tolerance}",
                    case.label
                );
            }
        }
        let gate_bytes = projection_len(case.gate_format, I, H) / EXPERTS;
        let down_bytes = projection_len(case.down_format, H, I) / EXPERTS;
        assert_eq!(
            gate_bytes as u64 * 2 + down_bytes as u64,
            case.expected_per_expert
        );
        assert_eq!(
            case.expected_per_expert * K as u64,
            case.expected_per_expert * 8
        );
        assert_eq!(
            case.expected_per_expert * EXPERTS as u64,
            case.expected_per_expert * 256
        );
        eprintln!(
            "{} native_cuda=true H={H} I={I} top_k={K} captures>0 replays>0 fallbacks=0 \
             load.uploaded_whole_bank_bytes={} warmup.logical_route_demand_bytes={} \
             prefill_repeated.logical_route_demand_bytes={} \
             prefill_repeated.unique_selected_expert_bytes={} \
             prefill_broad.logical_route_demand_bytes={} \
             prefill_broad.unique_selected_expert_bytes={} \
             decode.logical_route_demand_bytes={} decode.unique_selected_expert_bytes={} \
             decode_repeated.logical_route_demand_bytes={} \
             decode_repeated.unique_selected_expert_bytes={} \
             decode_broad.logical_route_demand_bytes={} \
             decode_broad.unique_selected_expert_bytes={} \
             physical_dram_bytes=None page_ins=0 byte_hit_rate=None",
            case.label,
            load_traffic.uploaded_whole_bank_bytes,
            warmup_traffic.logical_route_demand_bytes,
            prefill_repeated_traffic.logical_route_demand_bytes,
            prefill_repeated_traffic.unique_selected_expert_bytes,
            prefill_broad_traffic.logical_route_demand_bytes,
            prefill_broad_traffic.unique_selected_expert_bytes,
            decode_single_traffic.logical_route_demand_bytes,
            decode_single_traffic.unique_selected_expert_bytes,
            decode_repeated_traffic.logical_route_demand_bytes,
            decode_repeated_traffic.unique_selected_expert_bytes,
            decode_broad_traffic.logical_route_demand_bytes,
            decode_broad_traffic.unique_selected_expert_bytes,
        );
        assert!(observer.reset_device_graph().unwrap());
        observer.finish().unwrap();
        drop(bindings);
        drop(session);
        runtime.synchronize().unwrap();
        cuda.wait_for_deferred_releases().unwrap();
    }
    assert_eq!(
        53 * 77_856_768u64 + 18 * 90_439_680 + 4 * 105_381_888 + 109_314_048,
        6_285_164_544
    );
}

#[test]
fn executor_selects_lazy_or_resident_weight_delivery_from_ep_capability() {
    for (lazy, expected) in [(true, "lazy"), (false, "resident")] {
        let (graph, weights, path) = weight_delivery_fixture();
        let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ep: Arc<dyn ExecutionProvider> =
            Arc::new(WeightDeliveryEp::new(lazy, Arc::clone(&deliveries)));
        let mut executor = Executor::build(graph, weights, ep).unwrap();
        let outputs = executor.run(&[]).unwrap();

        assert_eq!(outputs[0].as_bytes(), &[1, 2, 3, 4]);
        assert_eq!(&*deliveries.lock().unwrap(), &[expected]);
        drop(executor);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Miri cannot model the file-backed memmap2 mmap used by WeightStore::map_external"
)]
fn executor_prefetches_next_lazy_weight_before_current_node_runs() {
    // This fixture deliberately uses an external initializer because the lazy
    // weight production path requires mmap provenance. Miri rejects that mmap
    // syscall before executor dispatch, so the Miri suite covers the pure
    // executor prefetch machinery in `executor::prefetch::tests` instead.
    let (graph, weights, path) = two_node_weight_delivery_fixture();
    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ep: Arc<dyn ExecutionProvider> =
        Arc::new(WeightDeliveryEp::new(true, Arc::clone(&deliveries)));
    let mut executor = Executor::build(graph, weights, ep).unwrap();
    let outputs = executor.run(&[]).unwrap();

    assert_eq!(outputs[0].as_bytes(), &[5, 6, 7, 8]);
    assert_eq!(
        &*deliveries.lock().unwrap(),
        &["prefetch", "lazy", "lazy"],
        "the executor must drive a production lookahead call before dispatching node 0"
    );
    drop(executor);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn non_host_lazy_only_initializer_skips_eager_device_residency() {
    for (lazy, expected_allocations, expected_uploads, expected_delivery) in
        [(true, 1, 0, "lazy"), (false, 2, 1, "resident")]
    {
        let (graph, weights, path) = weight_delivery_fixture();
        let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let allocations = Arc::new(AtomicUsize::new(0));
        let host_uploads = Arc::new(AtomicUsize::new(0));
        let ep: Arc<dyn ExecutionProvider> = Arc::new(WeightDeliveryEp::non_host(
            lazy,
            Arc::clone(&deliveries),
            Arc::clone(&allocations),
            Arc::clone(&host_uploads),
        ));
        let mut executor = Executor::build(graph, weights, ep).unwrap();

        assert_eq!(
            allocations.load(Ordering::Relaxed),
            expected_allocations,
            "lazy nxrt builds only the output; stock EPs also allocate the initializer"
        );
        assert_eq!(
            host_uploads.load(Ordering::Relaxed),
            expected_uploads,
            "lazy nxrt must not upload the initializer during build"
        );

        let outputs = executor.run(&[]).unwrap();
        assert_eq!(outputs[0].as_bytes(), &[1, 2, 3, 4]);
        assert_eq!(&*deliveries.lock().unwrap(), &[expected_delivery]);
        assert_eq!(
            host_uploads.load(Ordering::Relaxed),
            expected_uploads,
            "dispatch must not introduce a second EP upload"
        );
        drop(executor);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn initializer_shared_with_resident_consumer_uses_one_device_copy() {
    let (mut graph, weights, path) = weight_delivery_fixture();
    graph.opset_imports.insert(String::new(), 17);
    let weight = graph
        .values
        .iter()
        .find_map(|(vid, value)| (value.name.as_deref() == Some("weight")).then_some(vid))
        .unwrap();
    let resident_output =
        graph.create_named_value("resident_output", DataType::Uint8, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(1),
        "Identity",
        vec![Some(weight)],
        vec![resident_output],
    ));
    graph.add_output(resident_output);

    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let allocations = Arc::new(AtomicUsize::new(0));
    let host_uploads = Arc::new(AtomicUsize::new(0));
    let ep: Arc<dyn ExecutionProvider> = Arc::new(WeightDeliveryEp::non_host(
        true,
        Arc::clone(&deliveries),
        Arc::clone(&allocations),
        Arc::clone(&host_uploads),
    ));
    let mut executor = Executor::build(graph, weights, ep).unwrap();

    assert!(
        !executor.weight_handles.contains_key(&weight),
        "a resident consumer makes the single eager device copy authoritative"
    );
    assert_eq!(allocations.load(Ordering::Relaxed), 3);
    assert_eq!(host_uploads.load(Ordering::Relaxed), 1);

    let outputs = executor.run(&[]).unwrap();
    assert_eq!(outputs[0].as_bytes(), &[1, 2, 3, 4]);
    assert_eq!(outputs[1].as_bytes(), &[1, 2, 3, 4]);
    assert_eq!(&*deliveries.lock().unwrap(), &["resident", "resident"]);
    assert_eq!(
        host_uploads.load(Ordering::Relaxed),
        1,
        "both consumers must share the one resident initializer"
    );
    drop(executor);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn coverage_collector_surfaces_ep_decline_reason() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("x", DataType::Float32, vec![Dim::Static(1)]);
    let output = graph.create_named_value("y", DataType::Float32, vec![Dim::Static(1)]);
    graph.insert_node(Node::new(
        NodeId(0),
        "NotRegistered",
        vec![Some(input)],
        vec![output],
    ));

    let ep = CpuExecutionProvider::new();
    let mut issues = Vec::new();
    collect_cuda_coverage_issues(&graph, &graph, &ep, "graph", &mut issues);

    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].op_type, "NotRegistered");
    assert_eq!(issues[0].domain, "ai.onnx");
    assert!(
        issues[0]
            .reason
            .contains("no handler for ai.onnx::NotRegistered at opset 17"),
        "{}",
        issues[0].reason
    );
    assert!(
        !issues[0].reason.contains("unsupported by"),
        "{}",
        issues[0].reason
    );
}

#[test]
fn cuda_coverage_report_groups_all_distinct_failure_classes_deterministically() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("x", DataType::Float32, vec![Dim::Static(1)]);

    let op_types = [
        "RepeatedMissing",
        "Missing08",
        "RepeatedMissing",
        "Missing07",
        "Missing06",
        "RepeatedMissing",
        "Missing05",
        "Missing04",
        "Missing03",
        "Missing02",
        "Missing01",
        "Missing00",
        "RepeatedMissing",
    ];
    for (index, op_type) in op_types.into_iter().enumerate() {
        let output = graph.create_named_value(
            format!("output_{index}"),
            DataType::Float32,
            vec![Dim::Static(1)],
        );
        graph.insert_node(Node::new(
            NodeId(index as u32),
            op_type,
            vec![Some(input)],
            vec![output],
        ));
    }

    let ep = WeightDeliveryEp::with_device(
        false,
        Arc::new(std::sync::Mutex::new(Vec::new())),
        onnx_runtime_ir::DeviceId::cuda(0),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let report = || {
        cuda_fallback_report(&graph, &ep)
            .expect("CUDA declines must produce a fallback report")
            .to_string()
    };
    let first = report();
    let second = report();

    assert_eq!(first, second);
    assert!(first.contains("13 nodes assigned to CPU"));
    assert!(first.contains("GPU EP stock_test_ep did not claim 13 node(s)"));
    assert!(first.contains("the whole session uses cpu_ep"));
    assert_eq!(first.matches("ai.onnx::RepeatedMissing:").count(), 1);
    assert!(first.contains("ai.onnx::RepeatedMissing: no handler"));
    assert!(first.contains("[count=4; examples: graph/node#0, graph/node#12, graph/node#2]"));
    assert!(!first.contains("graph/node#5"));

    for op_type in [
        "Missing00",
        "Missing01",
        "Missing02",
        "Missing03",
        "Missing04",
        "Missing05",
        "Missing06",
        "Missing07",
        "Missing08",
    ] {
        assert_eq!(
            first.matches(&format!("ai.onnx::{op_type}:")).count(),
            1,
            "{first}"
        );
        assert!(
            first.contains(&format!("ai.onnx::{op_type}: no handler")),
            "{first}"
        );
    }
    assert!(!first.contains("more unsupported node"));
}

#[test]
fn cuda_decline_warns_and_falls_back_to_cpu_unless_strict() {
    let graph = || {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let input = graph.create_named_value("input", DataType::Float32, vec![Dim::Static(1)]);
        let output = graph.create_named_value("output", DataType::Float32, vec![Dim::Static(1)]);
        graph.add_input(input);
        graph.add_output(output);
        graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![output],
        ));
        graph
    };
    let cuda_ep = || {
        Arc::new(WeightDeliveryEp::with_device(
            false,
            Arc::new(std::sync::Mutex::new(Vec::new())),
            onnx_runtime_ir::DeviceId::cuda(0),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        )) as Arc<dyn ExecutionProvider>
    };

    let exec = Executor::build_with_cuda_requirement(
        graph(),
        Arc::new(WeightStore::new()),
        cuda_ep(),
        false,
    )
    .expect("default CUDA decline must use the CPU fallback");
    assert_eq!(exec.device_id().device_type, DeviceType::Cpu);
    let report = exec
        .execution_provider_fallback_report()
        .expect("fallback must remain observable");
    assert_eq!(report.assigned_node_count, 1);
    assert_eq!(report.assigned_ops, ["ai.onnx::Relu"]);
    assert_eq!(report.declines.len(), 1);
    assert_eq!(report.declines[0].op_type, "Relu");
    assert!(report.declines[0].reason.contains("intentionally declines"));

    let strict = Executor::build_with_cuda_requirement(
        graph(),
        Arc::new(WeightStore::new()),
        cuda_ep(),
        true,
    )
    .err()
    .expect("strict CUDA must reject CPU fallback");
    assert!(strict.to_string().contains("ONNX_GENAI_REQUIRE_CUDA=1"));
}

#[test]
fn sequence_executor_preserves_element_arc_identity() {
    use onnx_runtime_ir::{TensorData, WeightRef, static_shape};

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
    graph.set_initializer(
        input,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![2],
            [7.0f32, 8.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    let zero = graph.create_named_value("zero", DataType::Int64, static_shape([]));
    graph.set_initializer(
        zero,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![],
            0i64.to_le_bytes().to_vec(),
        )),
    );
    let one = graph.create_named_value("one", DataType::Int64, static_shape([]));
    graph.set_initializer(
        one,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![],
            1i64.to_le_bytes().to_vec(),
        )),
    );

    let first_sequence = graph.create_value(DataType::Float32, static_shape([]));
    graph.insert_node(Node::new(
        NodeId(0),
        "SequenceConstruct",
        vec![Some(input)],
        vec![first_sequence],
    ));
    let first_at = graph.create_value(DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "SequenceAt",
        vec![Some(first_sequence), Some(zero)],
        vec![first_at],
    ));
    let inserted_sequence = graph.create_value(DataType::Float32, static_shape([]));
    graph.insert_node(Node::new(
        NodeId(0),
        "SequenceInsert",
        vec![Some(first_sequence), Some(first_at)],
        vec![inserted_sequence],
    ));
    let second_at = graph.create_value(DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "SequenceAt",
        vec![Some(inserted_sequence), Some(one)],
        vec![second_at],
    ));
    graph.add_output(second_at);

    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    let output = executor.run(&[]).unwrap();
    assert_eq!(output[0].to_vec_f32(), vec![7.0, 8.0]);

    let original = &executor.sequences[&first_sequence].elements()[0];
    let first_at_arc = &executor.seq_elem_values[&first_at];
    let inserted = &executor.sequences[&inserted_sequence].elements()[1];
    let second_at_arc = &executor.seq_elem_values[&second_at];
    assert!(original.shares_storage_with(first_at_arc));
    assert!(original.shares_storage_with(inserted));
    assert!(original.shares_storage_with(second_at_arc));
    assert_eq!(original.as_ptr(), executor.buffers[&input].as_ptr());
}

#[test]
fn view_bounds_rejects_out_of_bounds_view() {
    // A [2, 3] f32 view needs 24 bytes; give it a 16-byte backing length.
    let shape = [2usize, 3];
    let strides = compute_contiguous_strides(&shape);
    let err = view_bounds(&shape, &strides, 0, DataType::Float32, 16);
    assert!(err.is_err(), "gate must reject an oversized view");

    // Exactly-fitting length is accepted.
    assert!(view_bounds(&shape, &strides, 0, DataType::Float32, 24).is_ok());
}

/// A negative byte offset region (via a byte_offset that pushes the origin
/// past the buffer) is also rejected.
#[test]
fn view_bounds_rejects_offset_overrun() {
    let shape = [4usize];
    let strides = compute_contiguous_strides(&shape);
    // 4 f32 = 16 bytes; origin at byte 8 leaves only 8 bytes → overrun.
    assert!(view_bounds(&shape, &strides, 8, DataType::Float32, 16).is_err());
    assert!(view_bounds(&shape, &strides, 0, DataType::Float32, 16).is_ok());
}

#[test]
fn sub_byte_view_bounds_rejects_geometry_overflow() {
    let shape = [usize::MAX, 2];
    let strides = compute_contiguous_strides(&shape);
    let error = view_bounds(&shape, &strides, 0, DataType::Int4, usize::MAX);
    assert!(matches!(error, Err(SessionError::ShapeOverflow { .. })));
}

#[test]
fn sub_byte_view_bounds_rejects_offset_overflow() {
    let shape = [1usize];
    let strides = compute_contiguous_strides(&shape);
    let error = view_bounds(&shape, &strides, usize::MAX, DataType::Int4, usize::MAX);
    assert!(matches!(error, Err(SessionError::ShapeOverflow { .. })));
}

#[test]
fn device_binding_validation_rejects_geometry_overflow() {
    let element_count = usize::MAX / 4;
    let error = bindings::required_binding_bytes(DataType::Float64, &[element_count], "huge");
    assert!(matches!(error, Err(SessionError::ShapeOverflow { .. })));
}

/// Symbol substitution: static dims pass through, bound symbols resolve, an
/// unbound symbol yields `None` (the uninferred-shape signal).
#[test]
fn substitute_resolves_bound_symbols_only() {
    let mut bindings = HashMap::new();
    bindings.insert(SymbolId(0), 7usize);
    let shape = vec![Dim::Symbolic(SymbolId(0)), Dim::Static(4)];
    assert_eq!(substitute(&shape, &bindings), Some(vec![7, 4]));

    let unbound = vec![Dim::Symbolic(SymbolId(1)), Dim::Static(4)];
    assert_eq!(substitute(&unbound, &bindings), None);
}

/// H-D1: element-count multiplication must be overflow-checked so a huge or
/// malicious shape reports `ShapeOverflow` instead of wrapping `usize` and
/// under-sizing the buffer.
#[test]
fn checked_numel_detects_overflow() {
    // Well-formed shapes multiply normally.
    assert_eq!(checked_numel(&[2, 3, 4], || "v".into()).unwrap(), 24);
    assert_eq!(checked_numel(&[], || "v".into()).unwrap(), 1);

    // A product past usize::MAX overflows.
    let huge = [usize::MAX, 2];
    let err = checked_numel(&huge, || "value#9".into());
    assert!(matches!(err, Err(SessionError::ShapeOverflow { .. })));
}

/// H-D1 (byte layer): even when the element *count* fits in `usize`, the
/// count → bytes multiply can wrap for a fixed-width dtype. The allocation
/// path must report `ShapeOverflow` rather than under-allocating.
#[test]
fn checked_storage_bytes_detects_byte_overflow() {
    // `usize::MAX / 4` elements fit in usize (pass checked_numel) but
    // `* 8` bytes for Float64 wraps — this is the exploited under-alloc.
    let numel = usize::MAX / 4;
    let err = checked_storage_bytes(DataType::Float64, numel, || "value#9".into(), &[numel]);
    assert!(matches!(err, Err(SessionError::ShapeOverflow { .. })));

    // A well-formed size passes through unchanged.
    assert_eq!(
        checked_storage_bytes(DataType::Float32, 4, || "v".into(), &[4]).unwrap(),
        16
    );
}

#[test]
fn dynamic_output_shapes_compress_counts_selected_values() {
    use onnx_runtime_ir::Attribute;

    let mut axis_node = Node::new(NodeId(0), "Compress", vec![], vec![]);
    axis_node
        .attributes
        .insert("axis".into(), Attribute::Int(-1));
    assert_eq!(
        dynamic_output_shapes(
            &axis_node,
            &[vec![2, 4], vec![5]],
            &[DataType::Float32, DataType::Bool],
            &[None, Some(vec![1, 0, 1, 1, 1])],
            &[],
            11,
        ),
        Some(vec![vec![2, 3]]),
        "condition entries beyond the selected axis must be ignored"
    );

    let flat_node = Node::new(NodeId(1), "Compress", vec![], vec![]);
    assert_eq!(
        dynamic_output_shapes(
            &flat_node,
            &[vec![2, 3], vec![4]],
            &[DataType::Float32, DataType::Bool],
            &[None, Some(vec![0, 1, 1, 0])],
            &[],
            11,
        ),
        Some(vec![vec![2]])
    );
}

#[test]
fn compress_condition_allows_image_sized_boolean_vectors() {
    let image_condition = MAX_SHAPE_DATA_ELEMS + 1;
    assert!(bounded_compress_condition(
        DataType::Bool,
        &[image_condition]
    ));
    assert!(!bounded_shape_input(DataType::Bool, &[image_condition]));
    assert!(!bounded_compress_condition(
        DataType::Int64,
        &[image_condition]
    ));
    assert!(!bounded_compress_condition(
        DataType::Bool,
        &[(1 << 20) + 1]
    ));
}

/// #1195 behavioural falsification. A `Compress` whose boolean condition is
/// longer than `MAX_SHAPE_DATA_ELEMS` (an image-sized mask) must still resolve
/// its data-dependent output shape and run to completion. Before the fix the
/// condition was rejected by the ordinary shape-data bound, so
/// `resolve_node_outputs` handed `dynamic_output_shapes` a `None` condition and
/// the run failed with `UnresolvedShape { op: "Compress" }`. Reverting the
/// dispatch routing (`compress_condition_i64` for `Compress` input 1 back to
/// `shape_input_i64`) makes this test go RED, while the shipped predicate-only
/// test stays green. The empty/full cases also cover the all-false and all-true
/// condition boundaries through the real resolve + kernel path.
#[test]
fn compress_runs_with_image_sized_condition_including_empty_and_full() {
    use onnx_runtime_ir::{Attribute, TensorData};

    let n = MAX_SHAPE_DATA_ELEMS + 500; // 1524: past the shape-data bound, under 1<<20.
    assert!(n > MAX_SHAPE_DATA_ELEMS);

    // (selected indices, expected output length)
    let scenarios: Vec<(Vec<usize>, usize)> = vec![
        (vec![0, 7, n - 1], 3), // sparse selection spanning the ends
        (Vec::new(), 0),        // all-false -> empty output
        ((0..n).collect(), n),  // all-true  -> full output
    ];

    for (selected, expected_len) in scenarios {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);

        let x = graph.create_named_value("x", DataType::Float32, static_shape([n]));
        graph.add_input(x);

        let mut cond_bytes = vec![0u8; n];
        for &i in &selected {
            cond_bytes[i] = 1;
        }
        let cond = graph.create_named_value("cond", DataType::Bool, static_shape([n]));
        graph.set_initializer(
            cond,
            WeightRef::Inline(TensorData::from_raw(DataType::Bool, vec![n], cond_bytes)),
        );

        // A symbolic output extent keeps the shape unresolved after static
        // inference, so the data-dependent resolve path (and the #1195 routing)
        // is the only thing that can size the output.
        let extent = graph.create_symbol(None);
        let y = graph.create_named_value("y", DataType::Float32, vec![Dim::Symbolic(extent)]);
        let mut node = Node::new(NodeId(0), "Compress", vec![Some(x), Some(cond)], vec![y]);
        node.attributes.insert("axis".into(), Attribute::Int(0));
        graph.insert_node(node);
        graph.add_output(y);

        let mut executor = Executor::build(
            graph,
            Arc::new(WeightStore::new()),
            auto_detect_cpu_ep().unwrap(),
        )
        .unwrap();

        let x_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x_val = Tensor::from_f32(&[n], &x_data).unwrap();
        let outputs = executor
            .run(&[("x", &x_val)])
            .expect("an image-sized Compress condition must resolve its output shape");
        let out = outputs[0].to_vec_f32();
        assert_eq!(
            out.len(),
            expected_len,
            "selected count must set the Compress output extent"
        );
        let expected: Vec<f32> = selected.iter().map(|&i| i as f32).collect();
        assert_eq!(
            out, expected,
            "Compress must gather exactly the selected rows"
        );
    }
}

/// Boundary behaviour of the Compress output-extent count that the #1195 sizer
/// feeds an image-sized condition into: the count clamps to the axis length when
/// the condition is longer, counts only the provided entries when it is shorter,
/// and spans the empty (all-false) and full (all-true) ends.
#[test]
fn dynamic_output_shapes_compress_boundary_counts() {
    use onnx_runtime_ir::Attribute;

    let mut node = Node::new(NodeId(0), "Compress", vec![], vec![]);
    node.attributes.insert("axis".into(), Attribute::Int(0));

    let count = |cond: Vec<i64>, axis_dim: usize| {
        dynamic_output_shapes(
            &node,
            &[vec![axis_dim], vec![cond.len()]],
            &[DataType::Float32, DataType::Bool],
            &[None, Some(cond)],
            &[],
            17,
        )
    };

    // Condition longer than the axis: entries past the axis length are ignored.
    assert_eq!(count(vec![1, 1, 1, 1, 1], 3), Some(vec![vec![3]]));
    // Condition shorter than the axis: only the provided entries are counted.
    assert_eq!(count(vec![1, 0, 1], 6), Some(vec![vec![2]]));
    // All-false selects nothing: an empty output extent.
    assert_eq!(count(vec![0, 0, 0, 0], 4), Some(vec![vec![0]]));
    // All-true selects the whole axis.
    assert_eq!(count(vec![1, 1, 1, 1], 4), Some(vec![vec![4]]));
}

#[test]
fn dynamic_output_shapes_stft_reads_runtime_step_and_frame_length() {
    let node = Node::new(
        NodeId(0),
        "STFT",
        vec![Some(ValueId(0)), Some(ValueId(1)), None, Some(ValueId(3))],
        vec![ValueId(4)],
    );
    let input_shapes = vec![vec![2, 16, 1], vec![], vec![], vec![]];
    let input_dtypes = vec![
        DataType::Float32,
        DataType::Int64,
        DataType::Undefined,
        DataType::Int64,
    ];

    for (step, expected_frames) in [(2, 5), (4, 3)] {
        let input_values = vec![None, Some(vec![step]), None, Some(vec![8])];
        assert_eq!(
            dynamic_output_shapes(&node, &input_shapes, &input_dtypes, &input_values, &[], 17,),
            Some(vec![vec![2, expected_frames, 5, 2]])
        );
    }
}

/// The data-dependent shape sizer must return exactly one shape per output
/// so the run loop's `out_shapes[oi]` indexing can never misindex. Slice is
/// single-output, so it returns a 1-element Vec; the run loop additionally
/// guards the count (see `OutputShapeCountMismatch`).
#[test]
fn dynamic_output_shapes_slice_is_single_output() {
    let node = Node::new(NodeId(0), "Slice", vec![], vec![]);
    let input_shapes = vec![vec![4usize, 2]];
    let input_values = vec![
        None,          // data (unused by sizer)
        Some(vec![1]), // starts
        Some(vec![3]), // ends
        Some(vec![0]), // axes
        Some(vec![1]), // steps
    ];
    let input_dtypes = vec![
        DataType::Float32,
        DataType::Int64,
        DataType::Int64,
        DataType::Int64,
        DataType::Int64,
    ];
    let out =
        dynamic_output_shapes(&node, &input_shapes, &input_dtypes, &input_values, &[], 17).unwrap();
    assert_eq!(out.len(), 1, "Slice must resolve exactly one output shape");
    assert_eq!(out[0], vec![2, 2]);

    let mut custom_slice = Node::new(NodeId(1), "Slice", vec![], vec![ValueId(0)]);
    custom_slice.domain = "example.custom".into();
    assert!(
        dynamic_output_shapes(
            &custom_slice,
            &input_shapes,
            &input_dtypes,
            &input_values,
            &[],
            17
        )
        .is_none(),
        "ONNX Slice semantics must not be applied to an unrelated custom-domain op"
    );

    // An op the sizer cannot resolve returns None (surfaces as UnresolvedShape).
    let other = Node::new(
        NodeId(2),
        "NxrtNeverRegisteredSentinelOp",
        vec![],
        vec![ValueId(0)],
    );
    assert!(
        dynamic_output_shapes(&other, &input_shapes, &input_dtypes, &input_values, &[], 17)
            .is_none()
    );
}

#[test]
fn dynamic_output_shapes_unsqueeze_supports_input_and_attribute_axes() {
    use onnx_runtime_ir::Attribute;

    let input_axes = Node::new(
        NodeId(0),
        "Unsqueeze",
        vec![Some(ValueId(0)), Some(ValueId(1))],
        vec![ValueId(2)],
    );
    assert_eq!(
        dynamic_output_shapes(
            &input_axes,
            &[vec![2, 3], vec![2]],
            &[DataType::Float32, DataType::Int64],
            &[None, Some(vec![0, -1])],
            &[],
            17,
        ),
        Some(vec![vec![1, 2, 3, 1]])
    );

    let mut attribute_axes = Node::new(
        NodeId(1),
        "Unsqueeze",
        vec![Some(ValueId(0))],
        vec![ValueId(1)],
    );
    attribute_axes
        .attributes
        .insert("axes".into(), Attribute::Ints(vec![1, -1]));
    assert_eq!(
        dynamic_output_shapes(
            &attribute_axes,
            &[vec![2, 3]],
            &[DataType::Float32],
            &[None],
            &[],
            11,
        ),
        Some(vec![vec![2, 1, 3, 1]])
    );
}

#[test]
fn dynamic_output_shapes_resize_reads_runtime_scales() {
    let node = Node::new(
        NodeId(0),
        "Resize",
        vec![Some(ValueId(0)), Some(ValueId(1)), Some(ValueId(2))],
        vec![ValueId(3)],
    );
    assert_eq!(
        dynamic_output_shapes(
            &node,
            &[vec![1, 128, 13, 13], vec![8], vec![4]],
            &[DataType::Float32, DataType::Float32, DataType::Float32],
            &[None, None, None],
            &[None, None, Some(vec![1.0, 1.0, 2.0, 2.0])],
            11,
        ),
        Some(vec![vec![1, 128, 26, 26]])
    );
}

#[test]
fn dynamic_output_shapes_non_max_suppression_counts_selected_boxes() {
    let node = Node::new(
        NodeId(0),
        "NonMaxSuppression",
        (0..5).map(|index| Some(ValueId(index))).collect(),
        vec![ValueId(5)],
    );
    let shapes = vec![vec![1, 3, 4], vec![1, 1, 3], vec![], vec![], vec![]];
    let dtypes = vec![
        DataType::Float32,
        DataType::Float32,
        DataType::Int64,
        DataType::Float32,
        DataType::Float32,
    ];
    let ints = vec![None, None, Some(vec![2]), None, None];
    let floats = vec![
        Some(vec![0., 0., 1., 1., 0., 0., 0.9, 0.9, 2., 2., 3., 3.]),
        Some(vec![0.9, 0.8, 0.7]),
        None,
        Some(vec![0.5]),
        Some(vec![0.0]),
    ];
    assert_eq!(
        dynamic_output_shapes(&node, &shapes, &dtypes, &ints, &floats, 11),
        Some(vec![vec![2, 3]])
    );
}

#[test]
fn dynamic_output_shapes_gqa_supports_packed_qkv() {
    use onnx_runtime_ir::{Attribute, ValueId};

    let mut node = Node::new(
        NodeId(0),
        "GroupQueryAttention",
        vec![
            Some(ValueId(0)),
            None,
            None,
            Some(ValueId(3)),
            Some(ValueId(4)),
            Some(ValueId(5)),
            Some(ValueId(6)),
        ],
        vec![ValueId(7), ValueId(8), ValueId(9)],
    );
    node.domain = "com.microsoft".into();
    node.attributes
        .insert("num_heads".into(), Attribute::Int(14));
    node.attributes
        .insert("kv_num_heads".into(), Attribute::Int(2));
    let input_shapes = vec![
        vec![1, 1, 1152],
        vec![],
        vec![],
        vec![1, 2, 16, 64],
        vec![1, 2, 16, 64],
        vec![1],
        vec![],
    ];
    let input_values = vec![None, None, None, None, None, None, Some(vec![17])];

    assert_eq!(
        dynamic_output_shapes(
            &node,
            &input_shapes,
            &[
                DataType::Float32,
                DataType::Undefined,
                DataType::Undefined,
                DataType::Float32,
                DataType::Float32,
                DataType::Int32,
                DataType::Int32,
            ],
            &input_values,
            &[],
            1,
        ),
        Some(vec![
            vec![1, 1, 896],
            vec![1, 2, 17, 64],
            vec![1, 2, 17, 64],
        ])
    );
}

/// The effective opset is read from the graph's import for the op's domain,
/// with the default and `ai.onnx` spellings treated as one.
#[test]
fn effective_opset_reads_graph_import() {
    let mut graph = Graph::default();
    graph.opset_imports.insert(String::new(), 12);
    let node = Node::new(NodeId(0), "Softmax", vec![], vec![]);
    assert_eq!(effective_opset(&graph, &node), 12);

    graph.opset_imports.insert(String::new(), 0);
    assert_eq!(effective_opset(&graph, &node), 0);
}

#[test]
#[should_panic(expected = "internal invariant violated")]
fn effective_opset_requires_validated_import() {
    effective_opset(
        &Graph::default(),
        &Node::new(NodeId(0), "Softmax", vec![], vec![]),
    );
}

#[test]
fn child_executor_binds_formals_captures_and_inline_initializers_in_output_order() {
    use onnx_runtime_ir::{TensorData, WeightRef, static_shape};

    let mut body = Graph::new();
    let formal = body.create_named_value("formal", DataType::Float32, static_shape([2]));
    body.add_input(formal);
    let captured = body.create_named_value("captured", DataType::Float32, static_shape([2]));
    let one = body.create_named_value("one", DataType::Float32, static_shape([2]));
    body.set_initializer(
        one,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![2],
            [1.0f32, 1.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    let sum = body.create_named_value("sum", DataType::Float32, static_shape([2]));
    body.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(formal), Some(captured)],
        vec![sum],
    ));
    let adjusted = body.create_named_value("adjusted", DataType::Float32, static_shape([2]));
    body.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(sum), Some(one)],
        vec![adjusted],
    ));
    // Deliberately reverse production order to prove formal output ordering.
    body.add_output(adjusted);
    body.add_output(sum);

    let mut opsets = HashMap::new();
    opsets.insert(String::new(), 17);
    let mut child = ChildExecutor::new(
        "direct-test",
        body,
        opsets,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    let mut outer_scope = HashMap::new();
    outer_scope.insert(
        "captured".to_string(),
        Tensor::from_f32(&[2], &[10.0, 20.0]).unwrap(),
    );

    let first = Tensor::from_f32(&[2], &[2.0, 3.0]).unwrap();
    let outputs = child.run(&[&first], &outer_scope).unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].to_vec_f32(), vec![13.0, 24.0]);
    assert_eq!(outputs[1].to_vec_f32(), vec![12.0, 23.0]);
    assert_eq!(child.stats(), ChildExecutorStats { builds: 1, runs: 1 });

    let second = Tensor::from_f32(&[2], &[-1.0, 4.0]).unwrap();
    let outputs = child.run(&[&second], &outer_scope).unwrap();
    assert_eq!(outputs[0].to_vec_f32(), vec![10.0, 25.0]);
    assert_eq!(outputs[1].to_vec_f32(), vec![9.0, 24.0]);
    assert_eq!(
        child.stats(),
        ChildExecutorStats { builds: 1, runs: 2 },
        "matching input signatures must reuse the compiled child plan"
    );
}

fn unary_child(name: &str) -> ChildExecutor {
    let mut body = Graph::new();
    let input = body.create_named_value("input", DataType::Float32, Vec::new());
    body.add_input(input);
    let output = body.create_named_value("output", DataType::Float32, Vec::new());
    body.insert_node(Node::new(
        NodeId(0),
        "Relu",
        vec![Some(input)],
        vec![output],
    ));
    body.add_output(output);

    let mut opsets = HashMap::new();
    opsets.insert(String::new(), 17);
    ChildExecutor::new(
        name,
        body,
        opsets,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap()
}

#[test]
fn child_executor_reuses_a_signature_after_an_intervening_signature() {
    let mut child = unary_child("a-b-a");
    let outer_scope = HashMap::new();
    let a = Tensor::from_f32(&[1], &[-1.0]).unwrap();
    let b = Tensor::from_f32(&[2], &[-2.0, 3.0]).unwrap();

    assert_eq!(
        child.run(&[&a], &outer_scope).unwrap()[0].to_vec_f32(),
        vec![0.0]
    );
    assert_eq!(
        child.run(&[&b], &outer_scope).unwrap()[0].to_vec_f32(),
        vec![0.0, 3.0]
    );
    assert_eq!(
        child.run(&[&a], &outer_scope).unwrap()[0].to_vec_f32(),
        vec![0.0]
    );
    assert_eq!(child.stats(), ChildExecutorStats { builds: 2, runs: 3 });
}

#[test]
fn child_executor_lru_evicts_oldest_signature_only() {
    let mut child = unary_child("lru-eviction");
    let outer_scope = HashMap::new();
    let inputs = (1..=CHILD_EXECUTOR_CACHE_CAPACITY + 1)
        .map(|len| Tensor::from_f32(&[len], &vec![len as f32; len]).unwrap())
        .collect::<Vec<_>>();

    for input in &inputs {
        child.run(&[input], &outer_scope).unwrap();
    }
    assert_eq!(
        child.stats(),
        ChildExecutorStats {
            builds: (CHILD_EXECUTOR_CACHE_CAPACITY + 1) as u64,
            runs: (CHILD_EXECUTOR_CACHE_CAPACITY + 1) as u64,
        }
    );

    child.run(&[&inputs[0]], &outer_scope).unwrap();
    child.run(&[inputs.last().unwrap()], &outer_scope).unwrap();
    assert_eq!(
        child.stats(),
        ChildExecutorStats {
            builds: (CHILD_EXECUTOR_CACHE_CAPACITY + 2) as u64,
            runs: (CHILD_EXECUTOR_CACHE_CAPACITY + 3) as u64,
        },
        "the evicted oldest signature must rebuild while a recent entry remains cached"
    );
}

fn captured_add_child(name: &str) -> ChildExecutor {
    let mut body = Graph::new();
    let input = body.create_named_value("input", DataType::Float32, Vec::new());
    body.add_input(input);
    let captured = body.create_named_value("captured", DataType::Float32, Vec::new());
    let output = body.create_named_value("output", DataType::Float32, Vec::new());
    body.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(input), Some(captured)],
        vec![output],
    ));
    body.add_output(output);

    let mut opsets = HashMap::new();
    opsets.insert(String::new(), 17);
    ChildExecutor::new(
        name,
        body,
        opsets,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap()
}

#[test]
fn child_executor_cached_plan_rebinds_captures_without_stale_state() {
    let mut child = captured_add_child("capture-shadowing");
    let a_input = Tensor::from_f32(&[1], &[1.0]).unwrap();
    let b_input = Tensor::from_f32(&[2], &[2.0, 3.0]).unwrap();

    let mut scope = HashMap::new();
    scope.insert(
        "captured".to_string(),
        Tensor::from_f32(&[1], &[10.0]).unwrap(),
    );
    assert_eq!(
        child.run(&[&a_input], &scope).unwrap()[0].to_vec_f32(),
        vec![11.0]
    );

    scope.insert(
        "captured".to_string(),
        Tensor::from_f32(&[2], &[20.0, 30.0]).unwrap(),
    );
    assert_eq!(
        child.run(&[&b_input], &scope).unwrap()[0].to_vec_f32(),
        vec![22.0, 33.0]
    );

    scope.insert(
        "captured".to_string(),
        Tensor::from_f32(&[1], &[40.0]).unwrap(),
    );
    let cached = child.run(&[&a_input], &scope).unwrap()[0].to_vec_f32();
    let mut fresh = captured_add_child("capture-shadowing-fresh");
    let freshly_compiled = fresh.run(&[&a_input], &scope).unwrap()[0].to_vec_f32();

    assert_eq!(cached, vec![41.0]);
    assert_eq!(cached, freshly_compiled);
    assert_eq!(child.stats(), ChildExecutorStats { builds: 2, runs: 3 });
}

// --- weight-streaming: zero-copy borrowed initializer buffers -----------

use onnx_runtime_ir::{WeightRef, static_shape};
use std::path::PathBuf;

/// A writable scratch dir under the workspace `target/` (never `/tmp`).
fn weightstream_tmp_dir() -> PathBuf {
    let dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/weightstream_test"
    ));
    std::fs::create_dir_all(&dir).expect("create weight-streaming test dir");
    dir
}

fn f32_le(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// (b) An aligned external-data initializer is backed **zero-copy** by a
/// borrowed buffer whose data pointer EQUALS the WeightStore's mmap slice —
/// no allocation, no copy. A model larger than RAM relies on this.
#[test]
fn aligned_external_initializer_is_borrowed_zero_copy() {
    let align = TensorLayout::contiguous().alignment;
    let path = weightstream_tmp_dir().join("aligned_init.bin");
    let w_data = [1.0f32, 2.0, 3.0, 4.0];
    std::fs::write(&path, f32_le(&w_data)).unwrap();

    let mut store = WeightStore::new();
    store.map_external(&path).unwrap();

    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
    g.set_initializer(
        w,
        WeightRef::External {
            path: path.clone(),
            offset: 0, // mmap base is page-aligned -> 0 is `align`-aligned
            length: 16,
            dtype: DataType::Float32,
            dims: vec![4],
        },
    );
    let y = g.create_value(DataType::Float32, static_shape([4]));
    g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(w)], vec![y]));
    g.add_output(y);

    let ep = auto_detect_cpu_ep().unwrap();
    let exec = Executor::build(g, Arc::new(store), ep).unwrap();

    let weight = &exec.graph.initializers[&w];
    let src = exec.weights().bytes(weight).unwrap();
    assert!(
        (src.as_ptr() as usize).is_multiple_of(align),
        "mmap window must be aligned for this test to exercise the zero-copy path"
    );
    let buf = &exec.buffers[&w];
    assert!(
        buf.is_borrowed(),
        "aligned initializer must be borrowed, not copied"
    );
    assert_eq!(
        buf.as_ptr() as *const u8,
        src.as_ptr(),
        "zero-copy: the buffer must alias the mmap bytes (no copy)"
    );

    let _ = std::fs::remove_file(&path);
}

/// (c) An external-data initializer that is dtype-aligned but not 64-byte
/// aligned remains a zero-copy mmap borrow and is numerically correct.
#[test]
fn device_unaligned_external_initializer_is_borrowed_at_dtype_alignment() {
    let align = TensorLayout::contiguous().alignment;
    let path = weightstream_tmp_dir().join("unaligned_init.bin");
    // Prefix the weight window with 8 bytes so it starts at offset 8, which
    // is f32-aligned but not a multiple of the EP allocation alignment (64).
    let offset = 8usize;
    let w_data = [5.0f32, 6.0, 7.0, 8.0];
    let mut file = vec![0u8; offset];
    file.extend_from_slice(&f32_le(&w_data));
    std::fs::write(&path, &file).unwrap();

    let mut store = WeightStore::new();
    store.map_external(&path).unwrap();

    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
    g.set_initializer(
        w,
        WeightRef::External {
            path: path.clone(),
            offset,
            length: 16,
            dtype: DataType::Float32,
            dims: vec![4],
        },
    );
    let x = g.create_named_value("X", DataType::Float32, static_shape([4]));
    g.add_input(x);
    let y = g.create_value(DataType::Float32, static_shape([4]));
    g.insert_node(Node::new(NodeId(0), "Add", vec![Some(x), Some(w)], vec![y]));
    g.add_output(y);

    let ep = auto_detect_cpu_ep().unwrap();
    let mut exec = Executor::build(g, Arc::new(store), ep).unwrap();

    let weight = &exec.graph.initializers[&w];
    let src = exec.weights().bytes(weight).unwrap();
    assert!(
        !(src.as_ptr() as usize).is_multiple_of(align),
        "window must be unaligned for this test to exercise the fallback"
    );
    let buf = &exec.buffers[&w];
    assert!(
        buf.is_borrowed(),
        "dtype-aligned mmap initializer must remain borrowed"
    );
    assert_eq!(
        buf.as_ptr() as *const u8,
        src.as_ptr(),
        "zero-copy buffer must alias the mmap window"
    );
    assert_eq!(buf.alignment(), std::mem::align_of::<f32>());

    // The copy is numerically correct: Y = X + W.
    let x_tensor = Tensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let out = exec.run(&[("X", &x_tensor)]).unwrap();
    assert_eq!(out.len(), 1);
    let got = out[0].to_vec_f32();
    let want = [15.0f32, 26.0, 37.0, 48.0];
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn unaligned_external_qmoe_keeps_route_first_enabled_and_matches_legacy() {
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _env_guard = ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("weight-offload env lock");

    struct RestoreEnv(Option<OsString>);
    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            if let Some(value) = self.0.take() {
                // SAFETY: this test serializes all mutations it performs.
                unsafe { std::env::set_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV, value) };
            } else {
                // SAFETY: this test serializes all mutations it performs.
                unsafe { std::env::remove_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV) };
            }
        }
    }

    let _restore = RestoreEnv(std::env::var_os(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV));
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-runtime-ep-cpu/tests/fixtures/qmoe_weight_offload/model.onnx.textproto");
    let input_values: Vec<f32> = (0..64).map(|index| index as f32 * 0.03125 - 1.0).collect();
    let router_values = vec![
        9.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 9.0,
    ];
    let input = Tensor::from_f32(&[4, 16], &input_values).unwrap();
    let router = Tensor::from_f32(&[4, 4], &router_values).unwrap();

    // SAFETY: guarded above; both executors compile synchronously here.
    unsafe { std::env::set_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV, "0") };
    let (legacy_graph, legacy_weights) =
        onnx_runtime_loader::load_model_with_weights(&fixture).unwrap();
    let mut legacy =
        Executor::build(legacy_graph, legacy_weights, auto_detect_cpu_ep().unwrap()).unwrap();
    let legacy_output = legacy.run(&[("X", &input), ("router", &router)]).unwrap();

    // SAFETY: guarded above; the offload kernel captures the flag at build.
    unsafe { std::env::set_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV, "1") };
    let before = onnx_runtime_ep_cpu::weight_offload_stats();
    let (offload_graph, offload_weights) =
        onnx_runtime_loader::load_model_with_weights(&fixture).unwrap();
    let mut offload = Executor::build(
        offload_graph,
        offload_weights,
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    for (&value, weight) in &offload.graph.initializers {
        let WeightRef::External { .. } = weight else {
            continue;
        };
        let source = offload.weights.bytes(weight).unwrap();
        assert!(!(source.as_ptr() as usize).is_multiple_of(TensorLayout::contiguous().alignment));
        let buffer = &offload.buffers[&value];
        assert!(buffer.is_borrowed());
        assert_eq!(buffer.as_ptr() as *const u8, source.as_ptr());
    }
    let offload_output = offload.run(&[("X", &input), ("router", &router)]).unwrap();
    let after = onnx_runtime_ep_cpu::weight_offload_stats();

    assert_eq!(
        offload_output[0].to_vec_f32(),
        legacy_output[0].to_vec_f32()
    );
    assert!(
        after.layer_executions
            >= before
                .layer_executions
                .checked_add(1)
                .expect("layer execution counter overflow")
    );
    assert!(after.bytes_read_from_mmap > before.bytes_read_from_mmap);
}

/// (d) Soundness guard: even when an initializer's mmap bytes are aligned
/// (so the zero-copy path would otherwise fire), the executor must NOT
/// borrow them if the value also has a producer — i.e. a malformed graph
/// reused the initializer's `ValueId` as a node output. Borrowing yields a
/// read-only buffer; a kernel writing that output would write through the
/// mmap (SIGSEGV / aliasing UB). The build must fall back to an owned,
/// writable copy instead.
#[test]
fn producer_backed_initializer_is_not_borrowed() {
    let align = TensorLayout::contiguous().alignment;
    let path = weightstream_tmp_dir().join("producer_backed_init.bin");
    let w_data = [1.0f32, 2.0, 3.0, 4.0];
    std::fs::write(&path, f32_le(&w_data)).unwrap();

    let mut store = WeightStore::new();
    store.map_external(&path).unwrap();

    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let x = g.create_named_value("X", DataType::Float32, static_shape([4]));
    g.add_input(x);
    let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
    g.set_initializer(
        w,
        WeightRef::External {
            path: path.clone(),
            offset: 0, // aligned: without the producer guard this would borrow
            length: 16,
            dtype: DataType::Float32,
            dims: vec![4],
        },
    );
    // Reuse the initializer's ValueId as a node output -> gives `w` a
    // producer, exactly the malformed shape the loader also rejects.
    g.insert_node(Node::new(NodeId(0), "Identity", vec![Some(x)], vec![w]));
    let y = g.create_value(DataType::Float32, static_shape([4]));
    g.insert_node(Node::new(NodeId(1), "Add", vec![Some(x), Some(w)], vec![y]));
    g.add_output(y);

    assert!(
        g.value(w).producer.is_some(),
        "test setup: initializer value must have a producer",
    );

    let ep = auto_detect_cpu_ep().unwrap();
    let exec = Executor::build(g, Arc::new(store), ep).unwrap();

    let weight = &exec.graph.initializers[&w];
    let src = exec.weights().bytes(weight).unwrap();
    assert!(
        (src.as_ptr() as usize).is_multiple_of(align),
        "mmap window must be aligned so only the producer guard prevents borrowing",
    );
    let buf = &exec.buffers[&w];
    assert!(
        !buf.is_borrowed(),
        "producer-backed initializer must fall back to an owned writable copy",
    );
    assert_ne!(
        buf.as_ptr() as *const u8,
        src.as_ptr(),
        "producer-backed initializer must not alias read-only mmap bytes",
    );

    let _ = std::fs::remove_file(&path);
}

/// Stage-0 capture prerequisite: a data-dependent decode shape that
/// `resolve_soft` omits (here `Range`'s runtime-length output feeding a
/// capture-safe `Cast`) forms an *unresolved-shape* eager seam. After one
/// eager warmup, [`Executor::seed_warm_decode_capture_shapes`] seeds that
/// value's exact just-in-time shape for the identical decode binding
/// signature, and the node no longer reports an unresolved-shape seam — the
/// executor now *admits* the already-capture-safe node instead of rejecting
/// it before consulting its kernel. Non-tautological: it asserts the
/// before/after seam transition and the concrete seeded extent.
#[test]
fn warm_decode_seeding_admits_previously_unresolved_capture_safe_node() {
    use onnx_runtime_ir::{Attribute, static_shape};

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    // Range(start, limit, delta) with all three supplied at run time, so
    // static shape inference cannot pin the output length: it stays symbolic
    // (data-dependent) and `resolve_soft` omits it.
    let start = graph.create_named_value("start", DataType::Int64, static_shape([]));
    let limit = graph.create_named_value("limit", DataType::Int64, static_shape([]));
    let delta = graph.create_named_value("delta", DataType::Int64, static_shape([]));
    graph.add_input(start);
    graph.add_input(limit);
    graph.add_input(delta);
    let len_sym = graph.intern_symbol("range_len");
    let r = graph.create_named_value("r", DataType::Int64, vec![len_sym.into()]);
    graph.insert_node(Node::new(
        NodeId(0),
        "Range",
        vec![Some(start), Some(limit), Some(delta)],
        vec![r],
    ));
    let y = graph.create_named_value("y", DataType::Float32, vec![len_sym.into()]);
    let mut cast = Node::new(NodeId(0), "Cast", vec![Some(r)], vec![y]);
    cast.attributes
        .insert("to".into(), Attribute::Int(DataType::Float32 as i64));
    graph.insert_node(cast);
    graph.add_output(y);

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    // Warm-decode capture-shape seeding is a device-graph-capture concern
    // (capture-capable EPs only), where the decode-plan memo is disabled by
    // construction (`device_type() != Cuda` gate). The memo-eligible CPU
    // eager path deliberately skips recording `capture_warm_shapes` (it never
    // captures). Disable the now-default-ON memo so this executor records the
    // warm shapes exactly as a capture-capable EP would at runtime.
    exec.set_decode_memo_enabled(false);

    // This test isolates warm-decode SEEDING — the unresolved-shape → resolved
    // transition that seeding is responsible for. The central classifier veto
    // (PR #728, exercised by its own tests) is an orthogonal, additional gate:
    // under the default fail-safe classifier the Range output's untraceable
    // data-dependent length is disqualifying, which would mask the
    // unresolved-shape seam this test asserts. Clear the disqualifying set so the
    // seeding transition is the sole observable, matching the Denylist view in
    // which a non-growing, replay-guarded extent is capture-safe.
    exec.capture_growing_symbols.clear();

    let zero = Tensor::from_raw(DataType::Int64, vec![], &0i64.to_le_bytes()).unwrap();
    let four = Tensor::from_raw(DataType::Int64, vec![], &4i64.to_le_bytes()).unwrap();
    let one = Tensor::from_raw(DataType::Int64, vec![], &1i64.to_le_bytes()).unwrap();
    let inputs = [("start", &zero), ("limit", &four), ("delta", &one)];

    let cast_pi = exec
        .plan
        .iter()
        .position(|p| exec.graph.node(p.node_id).op_type == "Cast")
        .expect("plan contains the Cast node");

    // --- Before any warmup: the Cast is an unresolved-shape eager seam. ---
    let bindings = exec
        .bind_symbols(&inputs, &ExternalBindings::default())
        .unwrap();
    let pre = exec.resolve_soft(&bindings);
    assert!(
        !pre.contains_key(&r) && !pre.contains_key(&y),
        "Range's runtime-length output (and its Cast) must be data-dependent (unresolved)"
    );
    let pre_seam = exec
        .node_capture_reason(&exec.plan[cast_pi], &pre)
        .and_then(|decline| decline.seam_reason);
    assert!(
        matches!(
            pre_seam,
            Some(SeamReason::UnresolvedInputShape) | Some(SeamReason::UnresolvedOutputShape)
        ),
        "without seeding the Cast must be an unresolved-shape seam; got {pre_seam:?}"
    );

    // --- One eager warmup records the exact just-in-time shapes. ----------
    let out = exec.run(&inputs).unwrap();
    assert_eq!(out[0].to_vec_f32(), vec![0.0, 1.0, 2.0, 3.0]);

    // --- After warmup: the identical signature seeds the warm shapes. -----
    let bindings2 = exec
        .bind_symbols(&inputs, &ExternalBindings::default())
        .unwrap();
    let mut post = exec.resolve_soft(&bindings2);
    assert!(
        !post.contains_key(&r),
        "resolve_soft alone still omits the data-dependent value"
    );
    exec.seed_warm_decode_capture_shapes(&mut post, &ExternalBindings::default());
    assert_eq!(
        post.get(&r),
        Some(&vec![4usize]),
        "warm seeding must restore Range's exact eager-resolved output shape"
    );
    assert_eq!(post.get(&y), Some(&vec![4usize]));

    // The unresolved-shape seam is gone: the executor admits the node to the
    // shape gate (any remaining decline is a kernel-capability decision, not
    // a missing-shape rejection).
    let post_seam = exec
        .node_capture_reason(&exec.plan[cast_pi], &post)
        .and_then(|decline| decline.seam_reason);
    assert!(
        !matches!(
            post_seam,
            Some(SeamReason::UnresolvedInputShape) | Some(SeamReason::UnresolvedOutputShape)
        ),
        "warm-seeded decode shapes must clear the unresolved-shape seam; got {post_seam:?}"
    );

    // A changed decode signature must NOT seed (pointer/capacity instability):
    // a phantom persistent input binding makes the current signature differ
    // from the warmup's, so the warm shapes are withheld and the value stays
    // unresolved rather than risk baking a stale shape into a captured graph.
    let mut mismatched = exec.resolve_soft(&bindings2);
    let mut other = ExternalBindings::default();
    other.inputs.insert(
        start,
        ExternalValue {
            dtype: DataType::Int64,
            shape: vec![],
            accepts_subshape: false,
            strides: None,
            fixed_stride_shape: None,
            ptr: 0x1000,
            len: 8,
            alignment: 8,
            device: onnx_runtime_ir::DeviceId::cpu(),
        },
    );
    exec.seed_warm_decode_capture_shapes(&mut mismatched, &other);
    assert!(
        !mismatched.contains_key(&r),
        "a changed persistent-binding signature must withhold the warm seed"
    );
}

/// A kernel that aborts device-graph *recording* (e.g. it advertises capture
/// support but synchronizes, which CUDA rejects mid-capture) is quarantined
/// by op-type so a single mislabeled kernel cannot abort the whole segmented
/// capture. Once its `(domain, op_type)` is quarantined,
/// [`Executor::node_capture_reason`] must force that node to a
/// `CaptureRecordingFailed` eager seam even when its shapes are fully
/// resolved and it would otherwise reach the kernel gate. Non-tautological:
/// it asserts the *transition* from "not an unresolved-shape seam" to a
/// forced-seam classification caused solely by the quarantine.
#[test]
fn quarantined_op_type_is_forced_to_a_capture_recording_failed_seam() {
    use onnx_runtime_ir::{Attribute, static_shape};

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    let x = graph.create_named_value("x", DataType::Int64, static_shape([4]));
    graph.add_input(x);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    let mut cast = Node::new(NodeId(0), "Cast", vec![Some(x)], vec![y]);
    cast.attributes
        .insert("to".into(), Attribute::Int(DataType::Float32 as i64));
    graph.insert_node(cast);
    graph.add_output(y);

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let cast_pi = exec
        .plan
        .iter()
        .position(|p| exec.graph.node(p.node_id).op_type == "Cast")
        .expect("plan contains the Cast node");

    // Statically-shaped Cast: its I/O shapes resolve without any seeding, so
    // it is NOT an unresolved-shape seam and reaches the kernel gate.
    let xt = Tensor::from_raw(
        DataType::Int64,
        vec![4],
        &[0i64, 1, 2, 3]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    let bindings = exec
        .bind_symbols(&[("x", &xt)], &ExternalBindings::default())
        .unwrap();
    let resolved = exec.resolve_soft(&bindings);
    let pre_seam = exec
        .node_capture_reason(&exec.plan[cast_pi], &resolved)
        .and_then(|decline| decline.seam_reason);
    assert!(
        !matches!(pre_seam, Some(SeamReason::CaptureRecordingFailed)),
        "a non-quarantined statically-shaped node must not be a recording-failed seam; \
             got {pre_seam:?}"
    );

    // Quarantine the Cast op-type (as the capture retry loop does after a
    // kernel aborts recording) and re-check: it is now a forced eager seam
    // regardless of its resolved shapes or kernel capability.
    exec.cap_mut()
        .capture_quarantine_ops
        .insert(("ai.onnx".to_string(), "Cast".to_string()));
    let post = exec.node_capture_reason(&exec.plan[cast_pi], &resolved);
    assert_eq!(
        post.and_then(|decline| decline.seam_reason),
        Some(SeamReason::CaptureRecordingFailed),
        "a quarantined op-type must be forced to a CaptureRecordingFailed eager seam"
    );
}

/// Per-slot host capture state isolates `Primary` (M=1 decode / greedy) from
/// `Verify` (M=k+1 speculative verify). Retargeting the graph slot is a pure
/// pointer move that must NOT reset the other slot — that is the invariant that
/// lets the two captured graphs coexist and, crucially, keeps greedy (which only
/// ever drives `Primary`) byte-identical when MTP flips the executor to `Verify`
/// and back around each verify forward.
#[test]
fn set_graph_slot_is_non_resetting_and_per_slot_isolated() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Int64, static_shape([4]));
    graph.add_input(x);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    let mut cast = Node::new(NodeId(0), "Cast", vec![Some(x)], vec![y]);
    cast.attributes
        .insert("to".into(), Attribute::Int(DataType::Float32 as i64));
    graph.insert_node(cast);
    graph.add_output(y);

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    // The main executor defaults to Primary (greedy's only slot).
    assert_eq!(exec.graph_slot(), DeviceGraphSlot::Primary);
    assert_eq!(DeviceGraphSlot::Primary.index(), 0);
    assert_eq!(DeviceGraphSlot::Verify.index(), 1);

    // Seed a distinctive marker into Primary's host capture state.
    let primary_key = ("ai.onnx".to_string(), "PrimaryMark".to_string());
    exec.cap_mut()
        .capture_quarantine_ops
        .insert(primary_key.clone());

    // Flip to Verify: a pure retarget. Verify starts empty (no bleed from
    // Primary), and Primary's marker survives untouched.
    exec.set_graph_slot(DeviceGraphSlot::Verify).unwrap();
    assert_eq!(exec.graph_slot(), DeviceGraphSlot::Verify);
    assert!(
        exec.cap().capture_quarantine_ops.is_empty(),
        "Verify slot must not observe Primary's capture state"
    );
    let verify_key = ("ai.onnx".to_string(), "VerifyMark".to_string());
    exec.cap_mut()
        .capture_quarantine_ops
        .insert(verify_key.clone());

    // Flip back to Primary: its marker is still present (the switch did NOT
    // reset it), and Verify's marker did not leak in.
    exec.set_graph_slot(DeviceGraphSlot::Primary).unwrap();
    assert!(
        exec.cap().capture_quarantine_ops.contains(&primary_key),
        "switching slots must not reset Primary's host capture state"
    );
    assert!(
        !exec.cap().capture_quarantine_ops.contains(&verify_key),
        "Verify's capture state must not leak into Primary"
    );
}

// ===================================================================
// F5 Stage 1 — steady-state decode-plan memo guard tests.
// ===================================================================

/// A decode-like symbolic graph: input `x` of shape `[batch, seq]` (both
/// symbolic) feeds an `Add(x, x)` whose output is length-*variant*, while an
/// inline initializer `w` of static shape `[4]` feeds a `Mul(y, w)` whose
/// output is length-*invariant*. This exercises both memo partitions.
#[cfg(test)]
struct DecodeMemoIds {
    batch: SymbolId,
    seq: SymbolId,
    x2: ValueId,
    ymul: ValueId,
}

#[cfg(test)]
fn decode_memo_test_graph() -> (Graph, DecodeMemoIds) {
    use onnx_runtime_ir::TensorData;
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let batch = graph.intern_symbol("batch");
    let seq = graph.intern_symbol("seq");

    // Length-variant spine: x[batch, seq] -> Add(x, x) -> x2[batch, seq].
    let x = graph.create_named_value("x", DataType::Float32, vec![batch.into(), seq.into()]);
    graph.add_input(x);
    let x2 = graph.create_named_value("x2", DataType::Float32, vec![batch.into(), seq.into()]);
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(x), Some(x)],
        vec![x2],
    ));
    graph.add_output(x2);

    // Length-invariant tail: y[4] (required input) * w[4] (initializer).
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.add_input(y);
    let w = graph.create_named_value("w", DataType::Float32, static_shape([4]));
    graph.set_initializer(
        w,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![4],
            [1.0f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    let ymul = graph.create_named_value("ymul", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(y), Some(w)],
        vec![ymul],
    ));
    graph.add_output(ymul);
    (
        graph,
        DecodeMemoIds {
            batch,
            seq,
            x2,
            ymul,
        },
    )
}

#[cfg(test)]
fn decode_memo_run(exec: &mut Executor, batch: usize, seq: usize) -> Vec<Vec<f32>> {
    let x = Tensor::from_f32(
        &[batch, seq],
        &(0..batch * seq).map(|i| i as f32 + 1.0).collect::<Vec<_>>(),
    )
    .unwrap();
    let y = Tensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    exec.run(&[("x", &x), ("y", &y)])
        .unwrap()
        .into_iter()
        .map(|t| t.to_vec_f32())
        .collect()
}

/// The default-ON master switch (Ripley's authoritative GO): the memo is
/// enabled unless `ONNX_GENAI_DECODE_MEMO` is an explicit OFF value
/// (`0`/`false`/`off`, case-insensitive, whitespace-trimmed). Unset, empty,
/// and unrecognized values all fail safe toward the validated fast path (ON).
#[test]
fn decode_memo_env_default_on_unless_explicitly_disabled() {
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _env_guard = ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("decode-memo env lock");

    struct RestoreEnv(Option<OsString>);
    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            match self.0.take() {
                // SAFETY: this test serializes all env mutations via ENV_LOCK.
                Some(value) => unsafe { std::env::set_var("ONNX_GENAI_DECODE_MEMO", value) },
                None => unsafe { std::env::remove_var("ONNX_GENAI_DECODE_MEMO") },
            }
        }
    }
    let _restore = RestoreEnv(std::env::var_os("ONNX_GENAI_DECODE_MEMO"));

    // Unset ⇒ ON (default).
    // SAFETY: guarded by ENV_LOCK above.
    unsafe { std::env::remove_var("ONNX_GENAI_DECODE_MEMO") };
    assert!(decode_memo_env_enabled(), "unset must default ON");

    // Explicit OFF values (case-insensitive, trimmed) ⇒ OFF.
    for off in ["0", "false", "off", "FALSE", "Off", "  0  ", "\tOFF\n"] {
        // SAFETY: guarded by ENV_LOCK above.
        unsafe { std::env::set_var("ONNX_GENAI_DECODE_MEMO", off) };
        assert!(!decode_memo_env_enabled(), "{off:?} must disable the memo");
    }

    // Explicit ON values and any unrecognized/empty value ⇒ ON (fail-safe).
    for on in [
        "1", "true", "on", "ON", "True", " on ", "", "  ", "yes", "2", "banana",
    ] {
        // SAFETY: guarded by ENV_LOCK above.
        unsafe { std::env::set_var("ONNX_GENAI_DECODE_MEMO", on) };
        assert!(decode_memo_env_enabled(), "{on:?} must keep the memo ON");
    }
}

/// Plan-invalidation unit test (F5 Stage 1 merge gate #2): prefill→decode
/// rebuilds (signature changed), pure length growth replays (signature
/// stable), and a batch change rebuilds.
#[test]
fn decode_plan_memo_rebuilds_and_replays() {
    let (graph, ids) = decode_memo_test_graph();
    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    exec.set_decode_memo_enabled(true);

    // "Prefill" step [1, 4]: first observation, nothing to diff yet.
    decode_memo_run(&mut exec, 1, 4);
    assert_eq!(exec.decode_memo_action(), DecodeMemoAction::Primed);
    assert!(exec.decode_memo.is_none());

    // First decode step [1, 5]: diffs against the prefill step and (re)builds
    // the memo — the prefill→decode transition changed the plan signature.
    decode_memo_run(&mut exec, 1, 5);
    assert_eq!(exec.decode_memo_action(), DecodeMemoAction::Rebuilt);
    let memo = exec.decode_memo.as_ref().expect("memo built");
    // `seq` grew (4→5) so it is varying; `batch` stayed 1 so it is invariant.
    assert!(memo.decode_varying.contains(&ids.seq));
    assert!(!memo.decode_varying.contains(&ids.batch));
    // The invariant tail (`ymul`) is cached; the variant spine (`x2`) is not.
    assert!(memo.invariant_shapes.contains_key(&ids.ymul));
    assert!(memo.variant_values.contains(&ids.x2));

    // Two decode steps at growing L both REPLAY: signature stable (only the
    // varying `seq` grows), invariant map reused, variant map re-resolved.
    let out6 = decode_memo_run(&mut exec, 1, 6);
    assert_eq!(exec.decode_memo_action(), DecodeMemoAction::Replayed);
    let out7 = decode_memo_run(&mut exec, 1, 7);
    assert_eq!(exec.decode_memo_action(), DecodeMemoAction::Replayed);
    // The variant tail was genuinely re-resolved to the new length.
    assert_eq!(out6[0].len(), 6);
    assert_eq!(out7[0].len(), 7);

    // A batch change [1, ·] → [2, ·] REBUILDS: `batch` was a non-varying
    // binding, so the change fails the replay guard and forces a rebuild.
    decode_memo_run(&mut exec, 2, 7);
    assert_eq!(exec.decode_memo_action(), DecodeMemoAction::Rebuilt);
    let memo = exec.decode_memo.as_ref().expect("memo rebuilt");
    assert_eq!(memo.reference_bindings.get(&ids.batch), Some(&2));
}

/// Token-exact lock (F5 Stage 1 merge gate #1): over ≥128 growing-length CPU
/// decode steps the memo-ON output is bit-identical to the memo-OFF output,
/// step for step. `decode_memo_verify` (forced on by
/// `set_decode_memo_enabled`) additionally asserts every replayed shape map
/// equals a fresh `resolve_soft`.
///
/// This locks the property the memo must never violate: it changes only
/// shape-resolution bookkeeping, never a produced byte. A full real-model
/// engine-level lock is available by running the (ignored) decode-lock
/// tests with `ONNX_GENAI_DECODE_MEMO=1`; this executor-level lock proves
/// memo==non-memo bit-exactness on the real CPU kernels without a model
/// fixture.
#[test]
fn decode_plan_memo_is_token_exact_over_128_steps() {
    const STEPS: usize = 130;

    let (off_graph, _) = decode_memo_test_graph();
    let mut off = Executor::build(
        off_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    // Memo is default-ON now; explicitly disable it for the reference run.
    off.set_decode_memo_enabled(false);
    assert!(!off.decode_memo_enabled);

    let (on_graph, _) = decode_memo_test_graph();
    let mut on = Executor::build(
        on_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    on.set_decode_memo_enabled(true);

    let mut replays = 0usize;
    for step in 0..STEPS {
        let seq = 3 + step; // strictly growing sequence length
        let ref_out = decode_memo_run(&mut off, 1, seq);
        let memo_out = decode_memo_run(&mut on, 1, seq);
        assert_eq!(
            ref_out, memo_out,
            "decode-plan memo diverged from the reference at step {step} (seq={seq})"
        );
        if on.decode_memo_action() == DecodeMemoAction::Replayed {
            replays += 1;
        }
    }
    // Steady state must actually engage the replay fast path for the bulk of
    // the run (priming costs the first two steps).
    assert!(
        replays >= STEPS - 2,
        "expected the memo to replay in steady state; only {replays}/{STEPS} replays"
    );
}

/// Proof-of-fire (F5 Stage 1): a PERSISTENT device-I/O binding shaped like a
/// KV cache — input+output aliased, length `L` growing by one each step —
/// must PRIME then REPLAY under the memo. This is the regression lock for
/// Ripley's finding that the memo reported `primed=0 rebuilt=0 replayed=0` on
/// the real native decode path because the old gate excluded any run carrying
/// external bindings. With `decode_memo_verify` on (forced by
/// `set_decode_memo_enabled`), every replay is also asserted byte-identical to
/// a fresh `resolve_soft`, so this doubles as a token-exact lock on the
/// persistent-binding path.
#[test]
fn decode_plan_memo_fires_on_persistent_kv_bindings() {
    use onnx_runtime_ir::{TensorData, static_shape};

    // KV-like length-variant spine: kv[L] -> Relu -> kvout[L], aliased into
    // one persistent device buffer. Static invariant tail: y[4] * w[4].
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let l = graph.intern_symbol("L");
    let kv = graph.create_named_value("kv", DataType::Float32, vec![l.into()]);
    graph.add_input(kv);
    let kvout = graph.create_named_value("kvout", DataType::Float32, vec![l.into()]);
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(kv)], vec![kvout]));
    graph.add_output(kvout);

    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.add_input(y);
    let w = graph.create_named_value("w", DataType::Float32, static_shape([4]));
    graph.set_initializer(
        w,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![4],
            [1.0f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    let ymul = graph.create_named_value("ymul", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(y), Some(w)],
        vec![ymul],
    ));
    graph.add_output(ymul);

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    exec.set_decode_memo_enabled(true);

    // Pre-allocate the KV buffer to a fixed capacity (stable pointer) and grow
    // only the logical length each step — the pre-allocated-KV-cache case.
    const CAP: usize = 128;
    let mut kv_binding = exec
        .allocate_device_binding(
            "kv".into(),
            Some("kvout".into()),
            DataType::Float32,
            vec![CAP],
            vec![1],
        )
        .unwrap();
    let ptr0 = kv_binding.device_ptr();
    let y_tensor = Tensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();

    let mut replays = 0usize;
    for step in 0..8usize {
        let len = 4 + step; // strictly growing KV length L
        kv_binding.set_logical_shape(vec![len]).unwrap();
        let bytes: Vec<u8> = (0..len).flat_map(|i| (i as f32).to_le_bytes()).collect();
        kv_binding.write_bytes(0, &bytes).unwrap();
        exec.run_with_device_bindings(&[("y", &y_tensor)], std::slice::from_mut(&mut kv_binding))
            .unwrap();
        if exec.decode_memo_action() == DecodeMemoAction::Replayed {
            replays += 1;
        }
    }

    // The pointer stayed stable (a growing-length view, not a realloc), so
    // the memo must not have been invalidated by capacity noise.
    assert_eq!(kv_binding.device_ptr(), ptr0);

    let (primed, rebuilt, replayed, ineligible) = exec.decode_memo_counts();
    assert_eq!(
        ineligible, 0,
        "persistent-KV decode must be memo-eligible, not excluded (the F5 regression)"
    );
    assert!(primed >= 1, "the first decode step must prime the memo");
    assert!(
        replayed >= 1,
        "steady persistent-KV decode must replay the memo \
             (primed={primed} rebuilt={rebuilt} replayed={replayed})"
    );
    assert_eq!(replays as u64, replayed);
}

/// F5 Stage 2 test graph. The persistent-KV spine (`kv[L] -> Relu -> kvout[L]`)
/// keeps the memo eligible, and an invariant tail (`y[4] * w[4] -> ymul[4]`)
/// feeds a `Reshape(ymul, [2,2]) -> yview` — a pure invariant zero-copy view.
/// Stage 2 must reinstate `yview` and elide the Reshape's dispatch on replay.
/// Returns the graph and the `ymul` value id (the view's source buffer).
#[cfg(test)]
fn stage2_view_graph() -> (Graph, ValueId) {
    use onnx_runtime_ir::{TensorData, static_shape};
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let l = graph.intern_symbol("L");
    let kv = graph.create_named_value("kv", DataType::Float32, vec![l.into()]);
    graph.add_input(kv);
    let kvout = graph.create_named_value("kvout", DataType::Float32, vec![l.into()]);
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(kv)], vec![kvout]));
    graph.add_output(kvout);

    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.add_input(y);
    let w = graph.create_named_value("w", DataType::Float32, static_shape([4]));
    graph.set_initializer(
        w,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![4],
            [1.0f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    let ymul = graph.create_named_value("ymul", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(y), Some(w)],
        vec![ymul],
    ));

    // Reshape ymul[4] -> yview[2,2] through a constant shape initializer, which
    // the CPU Reshape kernel serves as a zero-copy view over `ymul`'s buffer.
    let yshape = graph.create_named_value("yshape", DataType::Int64, static_shape([2]));
    graph.set_initializer(
        yshape,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![2],
            [2i64, 2].into_iter().flat_map(i64::to_le_bytes).collect(),
        )),
    );
    let yview = graph.create_named_value("yview", DataType::Float32, static_shape([2, 2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Reshape",
        vec![Some(ymul), Some(yshape)],
        vec![yview],
    ));
    graph.add_output(yview);
    (graph, ymul)
}

/// Run one persistent-KV decode step against [`stage2_view_graph`]: grow the KV
/// length to `len`, feed a fresh `y` (so `ymul` — and thus the elided `yview` —
/// varies each step, proving the reinstated view reads the freshly computed
/// source bytes), and return the materialized `yview` output.
#[cfg(test)]
fn stage2_run(
    exec: &mut Executor,
    kv_binding: &mut DeviceIoBinding,
    len: usize,
    y_bias: f32,
) -> Vec<f32> {
    kv_binding.set_logical_shape(vec![len]).unwrap();
    let bytes: Vec<u8> = (0..len).flat_map(|i| (i as f32).to_le_bytes()).collect();
    kv_binding.write_bytes(0, &bytes).unwrap();
    let y = Tensor::from_f32(
        &[4],
        &[y_bias + 1.0, y_bias + 2.0, y_bias + 3.0, y_bias + 4.0],
    )
    .unwrap();
    let outs = exec
        .run_with_device_bindings(&[("y", &y)], std::slice::from_mut(kv_binding))
        .unwrap();
    // Graph outputs are [kvout (bound → None), yview (returned)].
    outs.into_iter()
        .flatten()
        .next()
        .expect("yview output")
        .to_vec_f32()
}

#[cfg(test)]
fn stage2_kv_binding(exec: &Executor) -> DeviceIoBinding {
    exec.allocate_device_binding(
        "kv".into(),
        Some("kvout".into()),
        DataType::Float32,
        vec![256],
        vec![1],
    )
    .unwrap()
}

/// F5 Stage 2 proof-of-fire + token-exact lock. Over ≥128 growing-length
/// persistent-KV decode steps the invariant `Reshape` view is reinstated and
/// its dispatch elided (`views_reused`/`dispatch_elided` both grow), and the
/// memo-ON `yview` output is bit-identical to the memo-OFF reference every
/// step (with `y` — and therefore the view's source `ymul` — changing each
/// step, so a stale alias would immediately diverge). `decode_memo_verify`
/// (forced on by `set_decode_memo_enabled`) additionally asserts every
/// reinstated view equals a freshly built one in-flight.
#[test]
fn decode_view_plan_fires_and_is_token_exact_over_128_steps() {
    const STEPS: usize = 130;

    let (off_graph, _) = stage2_view_graph();
    let mut off = Executor::build(
        off_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    // Memo is default-ON now; explicitly disable it for the reference run.
    off.set_decode_memo_enabled(false);
    assert!(!off.decode_memo_enabled, "reference must run memo-OFF");
    let mut off_kv = stage2_kv_binding(&off);

    let (on_graph, _) = stage2_view_graph();
    let mut on = Executor::build(
        on_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    on.set_decode_memo_enabled(true);
    let mut on_kv = stage2_kv_binding(&on);

    for step in 0..STEPS {
        let len = 4 + step; // strictly growing KV length L
        let bias = step as f32; // vary the invariant-shape source each step
        let ref_out = stage2_run(&mut off, &mut off_kv, len, bias);
        let memo_out = stage2_run(&mut on, &mut on_kv, len, bias);
        assert_eq!(
            ref_out, memo_out,
            "Stage 2 view reuse diverged from the reference at step {step} (L={len})"
        );
    }

    let (views_reused, dispatch_elided) = on.decode_view_plan_counts();
    assert!(
        views_reused > 0 && dispatch_elided > 0,
        "Stage 2 must fire on steady decode (views_reused={views_reused}, \
             dispatch_elided={dispatch_elided})"
    );
    // Non-vacuous: the reshape view must be reused/elided on the bulk of steps
    // (priming + first rebuild cost the first few steps).
    assert!(
        views_reused as usize >= STEPS - 4,
        "expected steady Stage 2 reuse; only {views_reused}/{STEPS} views reused"
    );
    assert!(
        on.decode_view_plan.is_some(),
        "the cached view plan must survive steady-state replay"
    );
}

/// F5 Stage 2 buffer-identity invalidation lock. If a cached view's source
/// buffer is reallocated to a different base pointer between steps — the exact
/// hazard Stage 1 could ignore but Stage 2 cannot — the plan MUST detect the
/// signature mismatch, decline to reinstate the (now stale) alias, and fall
/// back to a full dispatch. The step's output must still be correct (no
/// dangling/stale view served) and the reuse counter must NOT advance.
#[test]
fn decode_view_plan_rebuilds_on_source_buffer_move() {
    use onnx_runtime_ir::TensorLayout;

    let (off_graph, _) = stage2_view_graph();
    let mut off = Executor::build(
        off_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    let mut off_kv = stage2_kv_binding(&off);

    let (on_graph, ymul) = stage2_view_graph();
    let mut on = Executor::build(
        on_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    on.set_decode_memo_enabled(true);
    let mut on_kv = stage2_kv_binding(&on);

    // Warm up so the view plan is built and firing.
    for step in 0..6usize {
        let len = 4 + step;
        let bias = step as f32;
        let r = stage2_run(&mut off, &mut off_kv, len, bias);
        let m = stage2_run(&mut on, &mut on_kv, len, bias);
        assert_eq!(r, m, "warmup diverged at step {step}");
    }
    assert!(
        on.decode_view_plan.is_some(),
        "view plan must be built before the realloc test"
    );
    let (reused_before, _) = on.decode_view_plan_counts();

    // Forcibly MOVE the view's source buffer (`ymul`) to a fresh allocation of
    // the same capacity — a base-pointer change the plan's signature must catch.
    let old = on.buffers.remove(&ymul).expect("ymul buffer");
    let cap = old.len();
    // Allocate the replacement *before* releasing the original, so the two
    // cannot share an address. Freeing first and allocating the same size
    // is exactly the request an allocator satisfies from its free list by
    // handing back the block just released, which left the pointer
    // unchanged and made this test fail wherever that happened.
    let fresh = on
        .ep
        .allocate(cap, TensorLayout::contiguous().alignment)
        .unwrap();
    let moved_ptr = fresh.as_ptr() as usize;
    assert_ne!(
        moved_ptr,
        old.as_ptr() as usize,
        "the replacement buffer must not reuse the original address, or the \
             signature check below is not being exercised"
    );
    on.ep.deallocate(old).unwrap();
    on.buffers.insert(ymul, fresh);
    // Sanity: the plan's recorded source pointer no longer matches.
    assert!(
        !on.stage2_buffer_sig_matches(on.decode_view_plan.as_ref().unwrap()),
        "the forced realloc must break the buffer-identity signature"
    );

    // Next step: Stage 2 must decline reuse (sig mismatch) yet stay correct.
    let len = 4 + 6;
    let bias = 6.0f32;
    let ref_out = stage2_run(&mut off, &mut off_kv, len, bias);
    let memo_out = stage2_run(&mut on, &mut on_kv, len, bias);
    assert_eq!(
        ref_out, memo_out,
        "a moved source buffer must force a rebuild, never serve a stale view"
    );
    let (reused_after, _) = on.decode_view_plan_counts();
    assert_eq!(
        reused_after, reused_before,
        "the mismatched step must NOT reuse cached views (would be stale)"
    );
    // The freshly computed ymul must live in a real buffer again (the moved one
    // or a self-healed reallocation), never left dangling.
    let healed = on.buffers.get(&ymul).expect("ymul rebound").as_ptr() as usize;
    assert!(healed == moved_ptr || healed != 0, "ymul must be backed");
}

// ============================================================================
// Kernel pre-binding (Stage 3): reachability proof
// ============================================================================

/// Proves the kernel pre-binding fast path is taken during steady-state dispatch
/// for a static-shape graph. The TEST_HITS counter increments on the pre-bound
/// path; a non-zero delta after two runs proves the path fires.
#[test]
fn kernel_prebinding_fast_path_fires_on_static_graph() {
    use super::PREBIND_FAST_PATH_TEST_HITS;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let a = graph.create_named_value("a", DataType::Float32, static_shape([4]));
    let b = graph.create_named_value("b", DataType::Float32, static_shape([4]));
    graph.add_input(a);
    graph.add_input(b);
    let sum = graph.create_named_value("sum", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(a), Some(b)],
        vec![sum],
    ));
    graph.add_output(sum);

    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let a_val = Tensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b_val = Tensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();

    // First run populates the binding (build already pre-compiled for static graphs).
    let before = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    executor
        .run(&[("a", &a_val), ("b", &b_val)])
        .expect("first run");
    let after_first = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);

    // The static-shape build pre-populates kernel_bindings, so the very first
    // run should already hit the fast path.
    assert!(
        after_first > before,
        "pre-bound fast path must fire on the first run of a static-shape graph \
         (before={before}, after={after_first})"
    );

    // Second run: same shapes, so fast path fires again.
    executor
        .run(&[("a", &a_val), ("b", &b_val)])
        .expect("second run");
    let after_second = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    assert!(
        after_second > after_first,
        "pre-bound fast path must fire on subsequent runs with stable shapes \
         (after_first={after_first}, after_second={after_second})"
    );

    // Verify correctness too.
    let out = executor.run(&[("a", &a_val), ("b", &b_val)]).unwrap();
    assert_eq!(out[0].to_vec_f32(), vec![11.0, 22.0, 33.0, 44.0]);
}

/// Proves the fallback path fires when shapes change (e.g. prefill→decode), and
/// that the pre-binding is updated so subsequent calls with the new shape hit
/// the fast path.
#[test]
fn kernel_prebinding_fallback_fires_on_shape_change() {
    use super::{PREBIND_FALLBACK_TEST_HITS, PREBIND_FAST_PATH_TEST_HITS};
    #[allow(unused_imports)]
    use onnx_runtime_ir::SymbolId;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    // Dynamic first dim (simulates sequence length changes).
    let seq_sym = graph.intern_symbol("seq");
    let shape_a: Shape = vec![Dim::Symbolic(seq_sym), Dim::Static(4)];
    let shape_sum: Shape = vec![Dim::Symbolic(seq_sym), Dim::Static(4)];

    let a = graph.create_named_value("a", DataType::Float32, shape_a.clone());
    let b = graph.create_named_value("b", DataType::Float32, shape_a.clone());
    graph.add_input(a);
    graph.add_input(b);
    let sum = graph.create_named_value("sum", DataType::Float32, shape_sum);
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(a), Some(b)],
        vec![sum],
    ));
    graph.add_output(sum);

    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    // Run 1: seq=3 (populates binding).
    let a1 = Tensor::from_f32(&[3, 4], &[1.0; 12]).unwrap();
    let b1 = Tensor::from_f32(&[3, 4], &[2.0; 12]).unwrap();
    executor.run(&[("a", &a1), ("b", &b1)]).expect("run seq=3");

    // Run 2: same seq=3 → fast path.
    let fast_before = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    executor
        .run(&[("a", &a1), ("b", &b1)])
        .expect("run seq=3 again");
    let fast_after = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    assert!(
        fast_after > fast_before,
        "fast path must fire on same-shape repeat"
    );

    // Run 3: seq=1 → shape change → fallback fires, binding updated.
    let a2 = Tensor::from_f32(&[1, 4], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let b2 = Tensor::from_f32(&[1, 4], &[1.0, 1.0, 1.0, 1.0]).unwrap();
    let fallback_before = PREBIND_FALLBACK_TEST_HITS.load(Ordering::Relaxed);
    let out = executor.run(&[("a", &a2), ("b", &b2)]).expect("run seq=1");
    let fallback_after = PREBIND_FALLBACK_TEST_HITS.load(Ordering::Relaxed);
    assert!(
        fallback_after > fallback_before,
        "fallback path must fire on shape change"
    );
    assert_eq!(out[0].to_vec_f32(), vec![6.0, 7.0, 8.0, 9.0]);

    // Run 4: seq=1 again → fast path fires (binding was updated).
    let fast_before2 = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    executor
        .run(&[("a", &a2), ("b", &b2)])
        .expect("run seq=1 again");
    let fast_after2 = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    assert!(
        fast_after2 > fast_before2,
        "after shape change, the updated binding must serve the fast path"
    );
}

/// Build a parent graph with a single `Scan` over a **multi-node** body so the
/// single-trip inline dual-path and the generic loop are exercised on identical
/// non-trivial work. `steps` is the scan-axis length (`1` = a decode step; `>1`
/// = a prefill-shaped run). The body threads carried state through two scan
/// inputs across three ops and emits one carried-state output plus one
/// per-iteration scan output:
///
///   `state_x  = Add(state, x)`
///   `state_out = Mul(state_x, y)`   (next carried state)
///   `scan_out  = Sub(state_out, x)` (stacked on the scan axis)
fn scan_inline_test_graph(steps: usize) -> Graph {
    const W: usize = 3;

    let mut body = Graph::new();
    body.opset_imports.insert(String::new(), 17);
    let state = body.create_named_value("state", DataType::Float32, static_shape([W]));
    let x = body.create_named_value("x", DataType::Float32, static_shape([W]));
    let y = body.create_named_value("y", DataType::Float32, static_shape([W]));
    body.add_input(state);
    body.add_input(x);
    body.add_input(y);
    let state_x = body.create_named_value("state_x", DataType::Float32, static_shape([W]));
    body.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(state), Some(x)],
        vec![state_x],
    ));
    let state_out = body.create_named_value("state_out", DataType::Float32, static_shape([W]));
    body.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(state_x), Some(y)],
        vec![state_out],
    ));
    let scan_out = body.create_named_value("scan_out", DataType::Float32, static_shape([W]));
    body.insert_node(Node::new(
        NodeId(0),
        "Sub",
        vec![Some(state_out), Some(x)],
        vec![scan_out],
    ));
    body.add_output(state_out);
    body.add_output(scan_out);

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let initial = init_inline(&mut graph, "initial", &[W], vec![0.0; W]);
    let x_in = graph.create_named_value("X", DataType::Float32, static_shape([steps, W]));
    let y_in = graph.create_named_value("Y", DataType::Float32, static_shape([steps, W]));
    graph.add_input(x_in);
    graph.add_input(y_in);
    let final_state = graph.create_named_value("final_state", DataType::Float32, static_shape([W]));
    let scan_output =
        graph.create_named_value("scan_output", DataType::Float32, static_shape([steps, W]));
    let mut scan = Node::new(
        NodeId(0),
        "Scan",
        vec![Some(initial), Some(x_in), Some(y_in)],
        vec![final_state, scan_output],
    );
    scan.attributes
        .insert("num_scan_inputs".to_string(), Attribute::Int(2));
    let scan_id = graph.insert_node(scan);
    graph.subgraphs.insert((scan_id, "body".to_string()), body);
    graph.add_output(final_state);
    graph.add_output(scan_output);
    graph
}

fn init_inline(graph: &mut Graph, name: &str, dims: &[usize], data: Vec<f32>) -> ValueId {
    use onnx_runtime_ir::{TensorData, WeightRef};
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let value =
        graph.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()));
    graph.set_initializer(
        value,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            dims.to_vec(),
            bytes,
        )),
    );
    value
}

fn run_scan_inline_graph(steps: usize, inline: bool) -> (Vec<Vec<u8>>, u64) {
    let mut exec = Executor::build(
        scan_inline_test_graph(steps),
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    exec.scan_inline_single_trip_enabled = inline;

    let n = steps * 3;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    let y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 2.0).collect();
    let x_t = Tensor::from_f32(&[steps, 3], &x).unwrap();
    let y_t = Tensor::from_f32(&[steps, 3], &y).unwrap();
    let outputs = exec.run(&[("X", &x_t), ("Y", &y_t)]).unwrap();
    let bytes = outputs.iter().map(|t| t.as_bytes().to_vec()).collect();
    (bytes, exec.scan_inline_single_trip_count())
}

/// Slice-1a correctness gate. Proves the flag-gated single-trip `Scan` inline
/// dual-path is (1) **byte-exact** with the generic `exec_scan` loop, and (2)
/// **non-vacuously engaged** and **runtime-keyed** — engaging only at
/// `trip_count == 1` and never on a prefill-shaped (`trip_count > 1`) run, even
/// with the flag ON. The shared-plan tripwire: a static single-trip rewrite
/// would fire on prefill too; this asserts it does not. Byte-equality is checked
/// over BOTH the carried `final_state` and the stacked `scan_output`, so a wrong
/// inline path (dropped/duplicated body run, mis-stacked scan axis, or skipped
/// state thread) makes the test FAIL.
#[test]
fn scan_single_trip_inline_is_byte_exact_and_runtime_keyed() {
    // Decode regime (trip_count == 1): inline path engages exactly once and is
    // byte-identical to the loop over every output.
    let (loop_out, loop_count) = run_scan_inline_graph(1, false);
    let (inline_out, inline_count) = run_scan_inline_graph(1, true);
    assert_eq!(loop_count, 0, "flag OFF must never engage the inline path");
    assert_eq!(
        inline_count, 1,
        "flag ON at trip_count==1 must engage the inline path exactly once"
    );
    assert_eq!(
        inline_out, loop_out,
        "single-trip inline output must be byte-exact with the loop path"
    );

    // Prefill regime (trip_count == 3): even with the flag ON the inline path
    // must NOT engage (runtime-keyed, not a static rewrite), and the output must
    // still match the loop.
    let (prefill_loop, prefill_loop_count) = run_scan_inline_graph(3, false);
    let (prefill_inline, prefill_inline_count) = run_scan_inline_graph(3, true);
    assert_eq!(prefill_loop_count, 0, "loop path never counts");
    assert_eq!(
        prefill_inline_count, 0,
        "flag ON must NOT inline a prefill (trip_count>1) Scan — the shared-plan tripwire"
    );
    assert_eq!(
        prefill_inline, prefill_loop,
        "prefill output must be identical flag-on vs flag-off"
    );
}

// ---------------------------------------------------------------------------
// Inc-1b PR-2: decode-inline sibling executor. These are fast, CPU-only,
// non-ignored guards covering Harry's mandatory review points:
//   * guard #1 — byte-identical per-token outputs AND final recurrent state
//     between the Scan child-session plan and the decode-inline plan;
//   * guard #3 — the decode-inline exec binds the identical persistent state
//     device buffer the main exec wrote at the prefill→decode hand-off;
//   * guard #4 — the first `num_state` sibling outputs stay present-state in
//     `state_pairs` order and inlined interior shapes resolve (Permissive).
// ---------------------------------------------------------------------------

/// A tiny hybrid decoder graph: a recurrent single-state `Scan` whose state is a
/// real graph input/output pair (`past_state` → `present_state`, the #573
/// `state_pairs` contract) so it can be bound to a persistent device buffer, and
/// whose scan axis is **symbolic** (`seq`) so one executor handles both a
/// multi-token prefill and single-token decode steps — exactly the shape the
/// decode-inline transform specializes.
///
/// Body: `present = Add(state, scan_in)` (recurrent accumulate, the state
/// output); `y = Mul(present, scan_in)` (per-iteration scan output).
fn recurrent_state_graph() -> Graph {
    use onnx_runtime_ir::{Dim, static_shape};
    const W: usize = 3;

    let mut body = Graph::new();
    body.opset_imports.insert(String::new(), 17);
    let state = body.create_named_value("state", DataType::Float32, static_shape([W]));
    let scan_in = body.create_named_value("scan_in", DataType::Float32, static_shape([W]));
    body.add_input(state);
    body.add_input(scan_in);
    let present = body.create_named_value("present", DataType::Float32, static_shape([W]));
    body.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(state), Some(scan_in)],
        vec![present],
    ));
    let y = body.create_named_value("y", DataType::Float32, static_shape([W]));
    body.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(present), Some(scan_in)],
        vec![y],
    ));
    body.add_output(present);
    body.add_output(y);

    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let seq = g.intern_symbol("seq");
    let past_state = g.create_named_value("past_state", DataType::Float32, static_shape([W]));
    g.add_input(past_state);
    let x = g.create_named_value("x", DataType::Float32, vec![Dim::from(seq), Dim::Static(W)]);
    g.add_input(x);
    let present_state = g.create_named_value("present_state", DataType::Float32, static_shape([W]));
    let scan_out = g.create_named_value(
        "scan_out",
        DataType::Float32,
        vec![Dim::from(seq), Dim::Static(W)],
    );
    let mut scan = Node::new(
        NodeId(0),
        "Scan",
        vec![Some(past_state), Some(x)],
        vec![present_state, scan_out],
    );
    scan.attributes
        .insert("num_scan_inputs".to_string(), Attribute::Int(1));
    let scan_id = g.insert_node(scan);
    g.subgraphs.insert((scan_id, "body".to_string()), body);
    g.add_output(present_state);
    g.add_output(scan_out);
    g
}

fn build_main_exec(graph: Graph) -> Executor {
    Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap()
}

/// Guard #1: N single-token decode steps through the Scan child-session plan
/// vs. the decode-inline plan produce byte-identical per-token outputs AND an
/// identical final recurrent state — the primary semantics-preservation proof.
#[test]
fn decode_inline_sibling_is_byte_exact_with_scan_and_preserves_state() {
    const W: usize = 3;
    let mut main = build_main_exec(recurrent_state_graph());
    let mut sib = main
        .build_decode_inline_sibling()
        .unwrap()
        .expect("recurrent single-trip Scan must yield a decode-inline sibling");

    // The sibling is a distinct plan with the Scan lowered away.
    assert!(
        !sib.graph.nodes.iter().any(|(_, n)| n.op_type == "Scan"),
        "decode-inline sibling must have no Scan node"
    );

    let mut state_main = vec![0f32; W];
    let mut state_sib = vec![0f32; W];
    for step in 0..6usize {
        let xk: Vec<f32> = (0..W).map(|i| (step * W + i) as f32 + 1.0).collect();
        let x = Tensor::from_f32(&[1, W], &xk).unwrap();

        let past_m = Tensor::from_f32(&[W], &state_main).unwrap();
        let out_m = main.run(&[("past_state", &past_m), ("x", &x)]).unwrap();
        let past_s = Tensor::from_f32(&[W], &state_sib).unwrap();
        let out_s = sib.run(&[("past_state", &past_s), ("x", &x)]).unwrap();

        assert_eq!(out_m.len(), out_s.len());
        for (idx, (tm, ts)) in out_m.iter().zip(&out_s).enumerate() {
            assert_eq!(
                tm.as_bytes(),
                ts.as_bytes(),
                "output #{idx} diverged at decode step {step}"
            );
        }
        state_main = out_m[0].to_vec_f32();
        state_sib = out_s[0].to_vec_f32();
    }
    assert_eq!(
        state_main, state_sib,
        "final recurrent state must be identical across the two plans"
    );
}

/// Guard #3: the decode-inline exec binds the identical persistent state device
/// buffer the main exec wrote at the prefill→decode hand-off. A multi-token
/// prefill runs on the main (Scan) exec into an in-place `past_state ==
/// present_state` device binding; single-token decode steps then run on the
/// decode-inline sibling against that same binding. The result must match an
/// all-main-exec reference bit-for-bit, proving continuity (design §3).
#[test]
fn decode_inline_sibling_preserves_persistent_state_across_prefill_handoff() {
    const W: usize = 3;
    let prefill: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 tokens × W
    let decode_steps: [[f32; W]; 4] = [
        [0.5, 1.5, 2.5],
        [1.0, 1.0, 1.0],
        [2.0, 0.0, -1.0],
        [3.5, 2.5, 1.5],
    ];

    // Reference: every step on the main (Scan) exec, in-place device state.
    let reference_states = {
        let mut main = build_main_exec(recurrent_state_graph());
        let mut binding = main
            .allocate_device_binding(
                "past_state".into(),
                Some("present_state".into()),
                DataType::Float32,
                vec![W],
                vec![W],
            )
            .unwrap();
        binding.write_bytes(0, &[0u8; W * 4]).unwrap();
        let x0 = Tensor::from_f32(&[prefill.len() / W, W], &prefill).unwrap();
        main.run_with_device_bindings(&[("x", &x0)], std::slice::from_mut(&mut binding))
            .unwrap();
        let mut states = Vec::new();
        for step in &decode_steps {
            let x = Tensor::from_f32(&[1, W], step).unwrap();
            main.run_with_device_bindings(&[("x", &x)], std::slice::from_mut(&mut binding))
                .unwrap();
            states.push(binding.read_bytes().unwrap());
        }
        states
    };

    // Handoff: prefill on main, decode on the sibling — same persistent buffer.
    let mut main = build_main_exec(recurrent_state_graph());
    let mut sib = main.build_decode_inline_sibling().unwrap().unwrap();
    let mut binding = main
        .allocate_device_binding(
            "past_state".into(),
            Some("present_state".into()),
            DataType::Float32,
            vec![W],
            vec![W],
        )
        .unwrap();
    binding.write_bytes(0, &[0u8; W * 4]).unwrap();
    let ptr_before = binding.device_ptr();
    let x0 = Tensor::from_f32(&[prefill.len() / W, W], &prefill).unwrap();
    main.run_with_device_bindings(&[("x", &x0)], std::slice::from_mut(&mut binding))
        .unwrap();

    for (step, expected) in decode_steps.iter().zip(&reference_states) {
        let x = Tensor::from_f32(&[1, W], step).unwrap();
        sib.run_with_device_bindings(&[("x", &x)], std::slice::from_mut(&mut binding))
            .unwrap();
        assert_eq!(
            &binding.read_bytes().unwrap(),
            expected,
            "decode-inline step state diverged from the all-main reference — state buffer continuity broken"
        );
    }
    assert_eq!(
        binding.device_ptr(),
        ptr_before,
        "the persistent state buffer must be the identical allocation across the handoff"
    );
}

/// Guard #4: the sibling preserves graph-output order (the first `num_state`
/// outputs remain the present-state values in `state_pairs` order) and its
/// inlined interior shapes resolve under Permissive inference (the build itself
/// runs that inference, so a converged build is the proof).
#[test]
fn decode_inline_sibling_preserves_state_output_order_and_resolves_shapes() {
    let main = build_main_exec(recurrent_state_graph());
    let sib = main.build_decode_inline_sibling().unwrap().unwrap();

    let main_out_names: Vec<_> = main
        .graph
        .outputs
        .iter()
        .map(|&v| main.graph.value(v).name.clone())
        .collect();
    let sib_out_names: Vec<_> = sib
        .graph
        .outputs
        .iter()
        .map(|&v| sib.graph.value(v).name.clone())
        .collect();
    assert_eq!(
        main_out_names, sib_out_names,
        "decode-inline sibling must preserve graph-output identity + order (present-state first)"
    );
    assert_eq!(
        sib_out_names.first().unwrap().as_deref(),
        Some("present_state"),
        "the first output must be the present recurrent state"
    );

    // The present-state output resolves to a concrete static shape after the
    // Permissive re-inference the sibling build performed.
    let present = sib.graph.outputs[0];
    let dims: Option<Vec<usize>> = sib
        .graph
        .value(present)
        .shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Some(*n),
            Dim::Symbolic(_) => None,
        })
        .collect();
    assert_eq!(dims, Some(vec![3]), "present-state shape must resolve");
}

/// A dense (Scan-free) decoder yields no decode-inline sibling — the feature is
/// a strict no-op off the hybrid single-trip path.
#[test]
fn decode_inline_sibling_none_for_dense_graph() {
    use onnx_runtime_ir::static_shape;
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let x = g.create_named_value("x", DataType::Float32, static_shape([2, 4]));
    g.add_input(x);
    let y = g.create_named_value("y", DataType::Float32, static_shape([2, 4]));
    g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]));
    g.add_output(y);

    let main = build_main_exec(g);
    assert!(
        main.build_decode_inline_sibling().unwrap().is_none(),
        "a dense decoder must not build a decode-inline sibling"
    );
}

/// A caller-owned buffer really backs the binding: the graph must read what
/// the caller wrote and write its result back into the caller's own memory.
///
/// Constructing the binding successfully proves nothing on its own — an
/// implementation that quietly allocated its own buffer and copied would look
/// identical. Reading the *caller's* array after the run is what distinguishes
/// them.
#[test]
fn an_external_buffer_is_used_in_place_rather_than_copied() {
    use onnx_runtime_ir::static_shape;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let kv = graph.create_named_value("kv", DataType::Float32, static_shape([4]));
    graph.add_input(kv);
    let kvout = graph.create_named_value("kvout", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(kv)], vec![kvout]));
    graph.add_output(kvout);

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    // The caller owns this. Nothing inside the session may free it.
    let mut owned: Vec<f32> = vec![-1.0, 2.0, -3.0, 4.0];
    let ptr = owned.as_mut_ptr().cast::<core::ffi::c_void>();
    let len_bytes = std::mem::size_of_val(owned.as_slice());

    let mut binding = unsafe {
        exec.device_binding_from_external_memory(crate::tensor::ExternalMemorySpec::input(
            "kv",
            Some("kvout"),
            DataType::Float32,
            vec![4],
            vec![4],
            ptr,
            len_bytes,
        ))
    }
    .unwrap();

    assert_eq!(
        binding.device_ptr().addr(),
        ptr.addr(),
        "the binding must point at the caller's buffer, not a copy of it"
    );

    exec.run_with_device_bindings(&[], std::slice::from_mut(&mut binding))
        .unwrap();
    drop(binding);

    // Relu, computed in place, observed through the caller's own Vec.
    assert_eq!(
        owned,
        vec![0.0, 2.0, 0.0, 4.0],
        "the run's output must land in the caller's buffer"
    );
    // `owned` is still valid here; dropping the binding must not have freed it.
    owned.push(5.0);
    assert_eq!(owned.len(), 5);
}

/// A buffer too small for the declared shape is refused before it can be
/// written past its end, and the error says what was needed.
#[test]
fn an_undersized_external_buffer_is_refused_with_the_size_it_needed() {
    use onnx_runtime_ir::static_shape;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let kv = graph.create_named_value("kv", DataType::Float32, static_shape([4]));
    graph.add_input(kv);
    let kvout = graph.create_named_value("kvout", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(kv)], vec![kvout]));
    graph.add_output(kvout);

    let exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let mut too_small: Vec<f32> = vec![0.0; 2];
    let error = unsafe {
        exec.device_binding_from_external_memory(crate::tensor::ExternalMemorySpec::input(
            "kv",
            Some("kvout"),
            DataType::Float32,
            vec![4],
            vec![4],
            too_small.as_mut_ptr().cast::<core::ffi::c_void>(),
            std::mem::size_of_val(too_small.as_slice()),
        ))
    }
    .expect_err("a buffer half the required size must be refused");
    let message = error.to_string();
    assert!(
        message.contains("16"),
        "the error must state the required byte count, got: {message}"
    );
    assert!(
        message.contains('8'),
        "the error must state the byte count supplied, got: {message}"
    );
}

/// A null pointer is refused rather than turned into a binding that faults on
/// first use.
#[test]
fn a_null_external_buffer_is_refused() {
    use onnx_runtime_ir::static_shape;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let kv = graph.create_named_value("kv", DataType::Float32, static_shape([4]));
    graph.add_input(kv);
    let kvout = graph.create_named_value("kvout", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(kv)], vec![kvout]));
    graph.add_output(kvout);

    let exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let error = unsafe {
        exec.device_binding_from_external_memory(crate::tensor::ExternalMemorySpec::input(
            "kv",
            Some("kvout"),
            DataType::Float32,
            vec![4],
            vec![4],
            core::ptr::null_mut(),
            16,
        ))
    }
    .expect_err("a null buffer must be refused");
    assert!(error.to_string().contains("null"));
}

/// An output-only external buffer: the graph writes into the caller's memory
/// without the buffer also being a graph input.
///
/// Without this the native side is strictly weaker than the ORT side, which can
/// bind an external value as an output, and the two backends stop being
/// interchangeable for anyone managing their own memory.
#[test]
fn an_external_buffer_can_be_bound_as_an_output_only() {
    use onnx_runtime_ir::static_shape;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(x);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]));
    graph.add_output(y);

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let mut owned: Vec<f32> = vec![99.0; 4];
    let ptr = owned.as_mut_ptr().cast::<core::ffi::c_void>();
    let len_bytes = std::mem::size_of_val(owned.as_slice());
    let mut binding = unsafe {
        exec.device_binding_from_external_memory(crate::tensor::ExternalMemorySpec::output(
            "y",
            DataType::Float32,
            vec![4],
            vec![4],
            ptr,
            len_bytes,
        ))
    }
    .unwrap();

    let x_tensor = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    exec.run_with_device_bindings(&[("x", &x_tensor)], std::slice::from_mut(&mut binding))
        .unwrap();
    drop(binding);

    assert_eq!(
        owned,
        vec![0.0, 2.0, 0.0, 4.0],
        "the graph output must land in the caller's buffer"
    );
}

/// A producer-less graph output (an initializer wired straight to an output,
/// same shape as `zero_copy_output_move_reallocates_and_preserves_producer_less_output`
/// above) bound as an **external device I/O output** must still be seeded with
/// the correct bytes on *every* run, not just the first.
///
/// This is the sibling of `an_external_buffer_can_be_bound_as_an_output_only`
/// for a value no node ever produces: node dispatch never visits it, so
/// nothing refills the caller's external buffer unless `collect_run_outputs`
/// explicitly copies from the resident initializer buffer each time. The
/// per-run buffer wipe must also not deallocate that resident copy out from
/// under a producer-less external output, or the second run would seed from
/// freed memory.
#[test]
fn an_external_output_binding_for_a_producer_less_value_is_seeded_every_run() {
    use onnx_runtime_ir::{TensorData, static_shape};

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([3]));
    graph.add_input(x);

    // Producer-less graph output: an initializer wired straight to an output,
    // with no node consuming or producing it.
    let k = graph.create_named_value("k", DataType::Float32, static_shape([3]));
    graph.set_initializer(
        k,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![3],
            [100.0f32, 200.0, 300.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    graph.add_output(k);

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let x_val = Tensor::from_f32(&[3], &[0.0, 0.0, 0.0]).unwrap();
    for run in 0..3 {
        // A fresh, poisoned buffer each run: only a real seed-copy can make
        // this read back correctly, never a stale leftover from a prior run.
        let mut owned: Vec<f32> = vec![-1.0; 3];
        let ptr = owned.as_mut_ptr().cast::<core::ffi::c_void>();
        let len_bytes = std::mem::size_of_val(owned.as_slice());
        let mut binding = unsafe {
            exec.device_binding_from_external_memory(crate::tensor::ExternalMemorySpec::output(
                "k",
                DataType::Float32,
                vec![3],
                vec![3],
                ptr,
                len_bytes,
            ))
        }
        .unwrap();

        exec.run_with_device_bindings(&[("x", &x_val)], std::slice::from_mut(&mut binding))
            .unwrap_or_else(|e| panic!("run {run} must succeed: {e}"));
        drop(binding);

        assert_eq!(
            owned,
            vec![100.0, 200.0, 300.0],
            "run {run}: producer-less external output must be seeded from the \
             resident initializer buffer, not left at its poisoned value"
        );
    }
}

/// A spec that binds neither an input nor an output is refused rather than
/// producing a binding nothing ever touches.
#[test]
fn an_external_buffer_bound_to_nothing_is_refused() {
    use onnx_runtime_ir::static_shape;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(x);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]));
    graph.add_output(y);

    let exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let mut owned: Vec<f32> = vec![0.0; 4];
    let mut spec = crate::tensor::ExternalMemorySpec::output(
        "y",
        DataType::Float32,
        vec![4],
        vec![4],
        owned.as_mut_ptr().cast::<core::ffi::c_void>(),
        std::mem::size_of_val(owned.as_slice()),
    );
    spec.output_name = None;
    let error = unsafe { exec.device_binding_from_external_memory(spec) }
        .expect_err("a binding attached to nothing must be refused");
    assert!(error.to_string().contains("neither an input nor an output"));
}

/// A misaligned pointer is refused, and the error says so specifically rather
/// than lumping it in with null.
#[test]
fn a_misaligned_external_buffer_is_refused() {
    use onnx_runtime_ir::static_shape;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let kv = graph.create_named_value("kv", DataType::Float32, static_shape([4]));
    graph.add_input(kv);
    let kvout = graph.create_named_value("kvout", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(kv)], vec![kvout]));
    graph.add_output(kvout);

    let exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    // A byte buffer offset by one, so the address cannot be suitably aligned.
    let mut bytes = vec![0u8; 64];
    let misaligned = unsafe { bytes.as_mut_ptr().add(1) }.cast::<core::ffi::c_void>();
    let error = unsafe {
        exec.device_binding_from_external_memory(crate::tensor::ExternalMemorySpec::input(
            "kv",
            Some("kvout"),
            DataType::Float32,
            vec![4],
            vec![4],
            misaligned,
            32,
        ))
    }
    .expect_err("a misaligned buffer must be refused");
    let message = error.to_string();
    assert!(
        message.contains("alignment"),
        "the error must say the problem is alignment, got: {message}"
    );
}

// PR (GQA fixed-capacity KV capture pin): a `GroupQueryAttention` node reads its
// past-KV inputs (3/4) as PHYSICAL CAPACITY and derives the valid attended
// length on-device (`seqlens_k`), so when the engine binds the cache at a fixed
// capacity the KV seq axis is CONSTANT across a captured replay. This locks the
// pin: `collect_capacity_pinned_kv_symbols` picks up the GQA KV seq symbol,
// `compute_capture_disqualifying_symbols_excluding` drops it (and its lineage
// closure), and the GQA node PLUS a KV-cache-sized consumer become
// capture-eligible — where without the pin they stay eager.
#[test]
fn gqa_fixed_capacity_kv_seq_symbol_is_pinned_and_admits_the_node() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);
    let seq = graph.create_symbol(None);
    let seq_kv = graph.create_symbol(None);

    let embeds = graph.create_named_value(
        "inputs_embeds",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(256)],
    );
    graph.add_input(embeds);
    let past_key = graph.create_named_value(
        "past_key_values.0.key",
        DataType::Float32,
        vec![sym(batch), st(2), sym(seq_kv), st(128)],
    );
    graph.add_input(past_key);
    let past_value = graph.create_named_value(
        "past_key_values.0.value",
        DataType::Float32,
        vec![sym(batch), st(2), sym(seq_kv), st(128)],
    );
    graph.add_input(past_value);
    let attn_out = graph.create_named_value(
        "attn_out",
        DataType::Float32,
        vec![sym(batch), sym(seq), st(256)],
    );
    let present_key = graph.create_named_value(
        "present.0.key",
        DataType::Float32,
        vec![sym(batch), st(2), sym(seq_kv), st(128)],
    );
    let present_value = graph.create_named_value(
        "present.0.value",
        DataType::Float32,
        vec![sym(batch), st(2), sym(seq_kv), st(128)],
    );
    let mut gqa = Node::new(
        NodeId(0),
        "GroupQueryAttention",
        vec![
            Some(embeds),
            Some(embeds),
            Some(embeds),
            Some(past_key),
            Some(past_value),
        ],
        vec![attn_out, present_key, present_value],
    );
    gqa.domain = "com.microsoft".to_string();
    graph.insert_node(gqa.clone());

    let kv_out = graph.create_named_value(
        "kv_sized_consumer_out",
        DataType::Float32,
        vec![sym(batch), st(2), sym(seq_kv), st(128)],
    );
    let kv_consumer = Node::new(NodeId(1), "Sigmoid", vec![Some(present_key)], vec![kv_out]);

    // Baseline (no pin): the GQA node and the KV-sized consumer are BOTH vetoed.
    let baseline = compute_capture_disqualifying_symbols(&graph);
    assert!(
        baseline.contains(&seq_kv),
        "without the pin the GQA KV seq symbol must be disqualifying, got {baseline:?}"
    );
    assert!(
        !node_capture_seq_independent(&graph, &gqa, &baseline),
        "without the pin the GQA node must stay eager"
    );
    assert!(
        !node_capture_seq_independent(&graph, &kv_consumer, &baseline),
        "without the pin the KV-cache-sized consumer must stay eager"
    );

    // The pin: GQA's fixed-capacity KV seq symbol is collected and excluded.
    let pinned = collect_capacity_pinned_kv_symbols(&graph);
    assert!(
        pinned.contains(&seq_kv),
        "the GQA fixed-capacity KV seq symbol must be pinned, got {pinned:?}"
    );
    let pinned_set = compute_capture_disqualifying_symbols_excluding(&graph, &pinned);
    assert!(
        !pinned_set.contains(&seq_kv),
        "the pinned KV seq symbol must be excluded from the disqualifying set, got {pinned_set:?}"
    );
    assert!(
        node_capture_seq_independent(&graph, &gqa, &pinned_set),
        "with the pin the GQA node must be capture-eligible"
    );
    assert!(
        node_capture_seq_independent(&graph, &kv_consumer, &pinned_set),
        "with the pin a fixed-capacity-KV-sized consumer must be capture-eligible"
    );

    // Idempotent: re-deriving the pin from the graph yields the same set.
    assert_eq!(
        collect_capacity_pinned_kv_symbols(&graph),
        pinned,
        "the pin must be a pure, idempotent function of the graph"
    );
}

// GUARD (don't blanket-disable the veto): a GENUINELY GROWING KV path — one whose
// attention op does NOT read its cache as physical capacity — must NOT be pinned,
// so its symbol stays disqualifying and the node stays eager.
//
// (1) A default-domain CAUSAL `Attention` with NO mask input (input 3 absent)
//     derives past length from the growing cache extent (there is no mask
//     frontier to read), so its cache is not physical capacity: not pinned.
//     (A causal Attention WITH a frozen additive mask input IS a capacity form —
//     covered by the classifier tests above.)
// (2) `CompressedSparseAttention` has NO past-KV inputs (its records grow from
//     total_sequence_length), so it cannot be a capacity form: not pinned.
#[test]
fn growing_kv_paths_are_not_pinned_and_stay_vetoed() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);
    let attn_seq_kv = graph.create_symbol(None);
    let csa_records = graph.create_symbol(None);

    // (1) Causal default-domain Attention (growing-concat KV, no mask input at 3).
    let q = graph.create_named_value("q", DataType::Float32, vec![sym(batch), st(1), st(512)]);
    graph.add_input(q);
    let attn_past_key = graph.create_named_value(
        "attn_past_key",
        DataType::Float32,
        vec![sym(batch), st(4), sym(attn_seq_kv), st(128)],
    );
    graph.add_input(attn_past_key);
    let attn_past_value = graph.create_named_value(
        "attn_past_value",
        DataType::Float32,
        vec![sym(batch), st(4), sym(attn_seq_kv), st(128)],
    );
    graph.add_input(attn_past_value);
    let mut attention = Node::new(
        NodeId(0),
        "Attention",
        vec![
            Some(q),
            Some(q),
            Some(q),
            None,
            Some(attn_past_key),
            Some(attn_past_value),
        ],
        vec![],
    );
    attention
        .attributes
        .insert("is_causal".into(), Attribute::Int(1));
    graph.insert_node(attention);

    // (2) CompressedSparseAttention: records grow on outputs 1/3, no past inputs.
    let csa_q =
        graph.create_named_value("csa_q", DataType::Float32, vec![sym(batch), st(1), st(512)]);
    graph.add_input(csa_q);
    let csa_records_out = graph.create_named_value(
        "csa_records",
        DataType::Float32,
        vec![sym(batch), st(4), sym(csa_records), st(64)],
    );
    graph.add_output(csa_records_out);
    let mut csa = Node::new(
        NodeId(1),
        "CompressedSparseAttention",
        vec![Some(csa_q)],
        vec![csa_q, csa_records_out, csa_q, csa_records_out],
    );
    csa.domain = "com.microsoft".to_string();
    graph.insert_node(csa);

    let pinned = collect_capacity_pinned_kv_symbols(&graph);
    assert!(
        !pinned.contains(&attn_seq_kv),
        "a causal (growing-concat) Attention KV symbol must NOT be pinned, got {pinned:?}"
    );
    assert!(
        !pinned.contains(&csa_records),
        "a CSA records symbol (no past-KV inputs) must NOT be pinned, got {pinned:?}"
    );

    let set = compute_capture_disqualifying_symbols_excluding(&graph, &pinned);
    assert!(
        set.contains(&attn_seq_kv) && set.contains(&csa_records),
        "genuinely growing KV symbols must stay disqualifying, got {set:?}"
    );
}

// Executor-level integration of the pin: building a decode executor over a GQA
// graph seeds the GQA KV seq symbol as disqualifying (every GQA layer eager);
// after `pin_fixed_capacity_kv_capture_symbols` the symbol is excluded and the
// pinned set is recorded. Locks the engine-facing entry point end to end.
#[test]
fn executor_pin_fixed_capacity_kv_admits_gqa() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    graph.opset_imports.insert("com.microsoft".into(), 1);

    let sym = Dim::Symbolic;
    let st = Dim::Static;

    let batch = graph.create_symbol(None);
    let seq_kv = graph.create_symbol(None);

    let q = graph.create_named_value("q", DataType::Float32, vec![sym(batch), st(1), st(256)]);
    graph.add_input(q);
    let past_key = graph.create_named_value(
        "past_key_values.0.key",
        DataType::Float32,
        vec![sym(batch), st(2), sym(seq_kv), st(128)],
    );
    graph.add_input(past_key);
    let past_value = graph.create_named_value(
        "past_key_values.0.value",
        DataType::Float32,
        vec![sym(batch), st(2), sym(seq_kv), st(128)],
    );
    graph.add_input(past_value);
    let attn_out = graph.create_named_value(
        "attn_out",
        DataType::Float32,
        vec![sym(batch), st(1), st(256)],
    );
    graph.add_output(attn_out);
    let present_key = graph.create_named_value(
        "present.0.key",
        DataType::Float32,
        vec![sym(batch), st(2), sym(seq_kv), st(128)],
    );
    graph.add_output(present_key);
    let present_value = graph.create_named_value(
        "present.0.value",
        DataType::Float32,
        vec![sym(batch), st(2), sym(seq_kv), st(128)],
    );
    graph.add_output(present_value);
    let mut gqa = Node::new(
        NodeId(0),
        "GroupQueryAttention",
        vec![Some(q), Some(q), Some(q), Some(past_key), Some(past_value)],
        vec![attn_out, present_key, present_value],
    );
    gqa.domain = "com.microsoft".to_string();
    gqa.attributes.insert("num_heads".into(), Attribute::Int(8));
    gqa.attributes
        .insert("kv_num_heads".into(), Attribute::Int(2));
    let gqa_node_id = gqa.id;
    graph.insert_node(gqa);

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let gqa_node = exec.graph.node(gqa_node_id).clone();
    assert!(
        exec.capture_growing_symbols.contains(&seq_kv),
        "before the pin the GQA KV seq symbol must be disqualifying"
    );
    assert!(
        !node_capture_seq_independent(&exec.graph, &gqa_node, &exec.capture_growing_symbols),
        "before the pin the GQA node must be classifier-vetoed"
    );

    let pinned = exec.pin_fixed_capacity_kv_capture_symbols();
    assert!(
        pinned >= 1,
        "at least the GQA KV seq symbol must be pinned, got {pinned}"
    );
    assert!(
        exec.capacity_pinned_kv_symbols.contains(&seq_kv),
        "the executor must record the pinned KV symbol"
    );
    assert!(
        !exec.capture_growing_symbols.contains(&seq_kv),
        "after the pin the KV seq symbol must be excluded from the disqualifying set"
    );
    assert!(
        node_capture_seq_independent(&exec.graph, &gqa_node, &exec.capture_growing_symbols),
        "after the pin the GQA node must be admitted to capture"
    );
}

// === #1020: prepare-only workspace planning bounds the MLA context/sequence
// axis instead of failing on it, while a genuinely unbounded axis still errors.
// The DeepSeek-V2 MLA fixture is not on this machine, so these exercise the
// resolution logic directly on synthetic graphs whose `::Attention` input has a
// runtime-dependent context axis (the reported `v_model.Unsqueeze_16` shape).

fn minimal_workspace_executor() -> Executor {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let a = graph.create_named_value("a", DataType::Float32, static_shape([1]));
    graph.add_input(a);
    let out = graph.create_named_value("out", DataType::Float32, static_shape([1]));
    graph.add_output(out);
    graph.insert_node(Node::new(NodeId(0), "Identity", vec![Some(a)], vec![out]));
    Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap()
}

// An unresolved dim that IS a recognized context/sequence axis resolves to the
// physically-allocated KV capacity (the extent bound to another growing symbol
// this prepare call) — the #1020 fix. Mirrors `v_model.Unsqueeze_16`: an MLA
// value-path sequence symbol shape inference never unified with the KV axis.
#[test]
fn prepare_workspace_binds_unresolved_context_axis_to_kv_capacity() {
    let mut exec = minimal_workspace_executor();
    let kv_seq = exec.graph.create_symbol(Some("total_seq".into()));
    let v_seq = exec.graph.create_symbol(Some("v_seq".into()));
    exec.capture_growing_symbols.insert(kv_seq);
    exec.capture_growing_symbols.insert(v_seq);
    let v = exec.graph.create_named_value(
        "v_model.Unsqueeze_16",
        DataType::Float32,
        vec![
            Dim::Static(1),
            Dim::Static(16),
            Dim::Symbolic(v_seq),
            Dim::Static(128),
        ],
    );
    exec.value_shapes
        .insert(v, exec.graph.value(v).shape.clone());
    // Only the KV axis is bound (to physical capacity 2048); `v_seq` is unbound.
    let mut symbols = HashMap::new();
    symbols.insert(kv_seq, 2048usize);
    let node = Node::new(NodeId(40), "Attention", vec![Some(v)], vec![]);
    let resolved = exec
        .resolve_planned_workspace_input_shape(v, &symbols, NodeId(40), &node, 3)
        .expect("a context/sequence axis must resolve to its bounded extent");
    match resolved {
        PlannedInputShape::Bounded { dims, applied } => {
            assert_eq!(dims, vec![1, 16, 2048, 128]);
            assert_eq!(applied, vec![(2, v_seq, AxisBound::KvCapacity(2048))]);
        }
        other => panic!("expected a bounded over-reservation, got {other:?}"),
    }
}

// A context/sequence axis whose model-declared maximum EXCEEDS the currently
// bound KV capacity is reserved against the LARGER of the two, so a bounded
// reservation can never under-reserve (the corruption class #945/#947 warns of).
#[test]
fn prepare_workspace_context_axis_never_under_reserves_below_declared_max() {
    let mut exec = minimal_workspace_executor();
    let kv_seq = exec.graph.create_symbol(Some("total_seq".into()));
    let v_seq = exec.graph.create_symbol(Some("v_seq".into()));
    exec.graph.symbol_constraints.get_mut(&v_seq).unwrap().max = Some(8192);
    exec.capture_growing_symbols.insert(kv_seq);
    exec.capture_growing_symbols.insert(v_seq);
    let v = exec.graph.create_named_value(
        "v",
        DataType::Float32,
        vec![Dim::Static(1), Dim::Symbolic(v_seq), Dim::Static(16)],
    );
    exec.value_shapes
        .insert(v, exec.graph.value(v).shape.clone());
    let mut symbols = HashMap::new();
    symbols.insert(kv_seq, 2048usize);
    let node = Node::new(NodeId(1), "Attention", vec![Some(v)], vec![]);
    let resolved = exec
        .resolve_planned_workspace_input_shape(v, &symbols, NodeId(1), &node, 0)
        .unwrap();
    match resolved {
        PlannedInputShape::Bounded { dims, applied } => {
            assert_eq!(dims, vec![1, 8192, 16]);
            assert_eq!(applied, vec![(1, v_seq, AxisBound::KvCapacity(8192))]);
        }
        other => panic!("expected a bounded over-reservation, got {other:?}"),
    }
}

// An unresolved dim that carries its own configured maximum (a declared
// `max_seq_len`-style ceiling) but is not a growing symbol is reserved against
// that maximum.
#[test]
fn prepare_workspace_binds_unresolved_axis_to_configured_max() {
    let mut exec = minimal_workspace_executor();
    let seq = exec.graph.create_symbol(Some("seq_len".into()));
    exec.graph.symbol_constraints.get_mut(&seq).unwrap().max = Some(4096);
    let v = exec.graph.create_named_value(
        "bounded_value",
        DataType::Float32,
        vec![Dim::Static(2), Dim::Symbolic(seq), Dim::Static(64)],
    );
    exec.value_shapes
        .insert(v, exec.graph.value(v).shape.clone());
    let symbols = HashMap::new();
    let node = Node::new(NodeId(9), "Attention", vec![Some(v)], vec![]);
    let resolved = exec
        .resolve_planned_workspace_input_shape(v, &symbols, NodeId(9), &node, 1)
        .unwrap();
    match resolved {
        PlannedInputShape::Bounded { dims, applied } => {
            assert_eq!(dims, vec![2, 4096, 64]);
            assert_eq!(applied, vec![(1, seq, AxisBound::ConfiguredMax(4096))]);
        }
        other => panic!("expected a bounded over-reservation, got {other:?}"),
    }
}

// A dim that is unresolved for any OTHER reason — neither a known
// context/sequence axis nor an axis with a configured maximum — must keep
// failing. Reserving against a guess there would silently under-reserve.
#[test]
fn prepare_workspace_fails_on_unresolved_unbounded_axis() {
    let mut exec = minimal_workspace_executor();
    let mystery = exec.graph.create_symbol(Some("data_dependent".into()));
    let v = exec.graph.create_named_value(
        "mystery_value",
        DataType::Float32,
        vec![Dim::Static(4), Dim::Symbolic(mystery)],
    );
    exec.value_shapes
        .insert(v, exec.graph.value(v).shape.clone());
    let symbols = HashMap::new();
    let node = Node::new(NodeId(7), "Attention", vec![Some(v)], vec![]);
    let err = exec
        .resolve_planned_workspace_input_shape(v, &symbols, NodeId(7), &node, 0)
        .expect_err("a genuinely unbounded unresolved dim must still error");
    let msg = err.to_string();
    assert!(
        msg.contains("genuinely unknown"),
        "the error must name the unbounded-guess hazard, got: {msg}"
    );
    assert!(
        msg.contains("data_dependent"),
        "the error must name the unresolved symbol, got: {msg}"
    );
}

// Regression guard: a fully-resolvable input still resolves EXACTLY (never via a
// bound), so graphs that already resolved keep byte-identical reservations.
#[test]
fn prepare_workspace_exact_resolution_is_unchanged() {
    let mut exec = minimal_workspace_executor();
    let seq = exec.graph.create_symbol(Some("seq".into()));
    let v = exec.graph.create_named_value(
        "exact_value",
        DataType::Float32,
        vec![Dim::Static(1), Dim::Symbolic(seq), Dim::Static(8)],
    );
    exec.value_shapes
        .insert(v, exec.graph.value(v).shape.clone());
    let mut symbols = HashMap::new();
    symbols.insert(seq, 12usize);
    let node = Node::new(NodeId(3), "Attention", vec![Some(v)], vec![]);
    let resolved = exec
        .resolve_planned_workspace_input_shape(v, &symbols, NodeId(3), &node, 0)
        .unwrap();
    assert_eq!(resolved, PlannedInputShape::Exact(vec![1, 12, 8]));
    assert_eq!(resolved.dims(), &[1, 12, 8]);
}

// DeepSeek-V2-Lite's MoE gate flattens `[batch, sequence, hidden]` through
// `Reshape([-1, hidden]) -> Cast -> MatMul`. The loader shape for the flattened
// value is a derived symbol (`batch * sequence`) that is not directly bound as a
// graph input, but the producer chain is statically shape-deterministic for the
// current run. Prepare-only planning must recover that exact runtime extent
// instead of falling back to a huge max-sequence reservation or rejecting it as
// data-dependent.
#[test]
fn prepare_workspace_resolves_reshape_flatten_chain_exactly() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let batch = graph.create_symbol(Some("batch".into()));
    let seq = graph.create_symbol(Some("sequence_len".into()));
    let flat = graph.create_symbol(Some("batch_times_sequence".into()));
    let x = graph.create_named_value(
        "hidden",
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Symbolic(seq), Dim::Static(2048)],
    );
    graph.add_input(x);
    let shape = graph.create_named_value("reshape_shape", DataType::Int64, static_shape([2]));
    let mut constant = Node::new(NodeId(0), "Constant", vec![], vec![shape]);
    constant
        .attributes
        .insert("value_ints".into(), Attribute::Ints(vec![-1, 2048]));
    graph.insert_node(constant);
    let reshaped = graph.create_named_value(
        "v_model.layers.1.mlp.moe.Reshape_78",
        DataType::Float32,
        vec![Dim::Symbolic(flat), Dim::Static(2048)],
    );
    graph.insert_node(Node::new(
        NodeId(1),
        "Reshape",
        vec![Some(x), Some(shape)],
        vec![reshaped],
    ));
    let cast = graph.create_named_value(
        "v_model.layers.1.mlp.moe.Cast_79",
        DataType::Float32,
        vec![Dim::Symbolic(flat), Dim::Static(2048)],
    );
    graph.insert_node(Node::new(
        NodeId(2),
        "Cast",
        vec![Some(reshaped)],
        vec![cast],
    ));
    let weight =
        graph.create_named_value("gate.weight", DataType::Float32, static_shape([2048, 64]));
    graph.add_input(weight);
    let out = graph.create_named_value(
        "gate",
        DataType::Float32,
        vec![Dim::Symbolic(flat), Dim::Static(64)],
    );
    graph.add_output(out);
    let matmul = Node::new(
        NodeId(3),
        "MatMul",
        vec![Some(cast), Some(weight)],
        vec![out],
    );
    graph.insert_node(matmul.clone());

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    exec.value_shapes
        .insert(cast, exec.graph.value(cast).shape.clone());
    let mut symbols = HashMap::new();
    symbols.insert(batch, 1);
    symbols.insert(seq, 5);

    let resolved = exec
        .resolve_planned_workspace_input_shape(cast, &symbols, NodeId(3), &matmul, 0)
        .unwrap();
    assert_eq!(resolved, PlannedInputShape::Exact(vec![5, 2048]));
}

/// A borrowed input buffer must never escape into cross-run sequence storage.
///
/// `read_seq_element` promotes a value's buffer into a `SharedTensorBuffer` that
/// `restore_shared_buffers` reinstates on the *next* run. A zero-copy input
/// alias is only valid for the run that created it, so promoting the alias
/// would leave `buffers[input]` pointing at a caller tensor that has already
/// been dropped — and the next `copy_from_host` would write through it.
///
/// Falsifier: delete the `is_borrowed` branch in `read_seq_element` and this
/// test fails on the `!is_borrowed()` assertion (and, with a differently
/// aligned second input, aborts inside the allocator).
#[test]
fn sequence_promotion_never_retains_a_borrowed_input_alias() {
    use onnx_runtime_ir::{TensorData, WeightRef, static_shape};

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
    graph.add_input(input);
    let zero = graph.create_named_value("zero", DataType::Int64, static_shape([]));
    graph.set_initializer(
        zero,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![],
            0i64.to_le_bytes().to_vec(),
        )),
    );
    let seq = graph.create_value(DataType::Float32, static_shape([]));
    graph.insert_node(Node::new(
        NodeId(0),
        "SequenceConstruct",
        vec![Some(input)],
        vec![seq],
    ));
    let at = graph.create_value(DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(1),
        "SequenceAt",
        vec![Some(seq), Some(zero)],
        vec![at],
    ));
    graph.add_output(at);

    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let vid = executor.input_index["input"];
    let first = Tensor::from_f32(&[2], &[1.0, 2.0]).unwrap();
    let first_ptr = first.as_bytes().as_ptr() as usize;
    assert_eq!(
        executor.run(&[("input", &first)]).unwrap()[0].to_vec_f32(),
        vec![1.0, 2.0]
    );
    drop(first);

    let installed = &executor.buffers[&vid];
    assert!(
        !installed.is_borrowed(),
        "input buffer is still a borrowed alias after the run that borrowed it"
    );
    assert_ne!(
        installed.as_ptr() as usize,
        first_ptr,
        "input buffer still points at the (now dropped) caller tensor"
    );

    // A second run must read the new tensor, not stale storage, and must not
    // write through any retained alias.
    let second = Tensor::from_f32(&[2], &[3.0, 4.0]).unwrap();
    assert_eq!(
        executor.run(&[("input", &second)]).unwrap()[0].to_vec_f32(),
        vec![3.0, 4.0]
    );
    assert!(!executor.buffers[&vid].is_borrowed());
    assert!(executor.parked_input_buffers.is_empty());
}

/// Dropping an executor must return every buffer it owns to the allocator,
/// including one parked while a zero-copy input alias stood in its slot.
///
/// `unbind_borrowed_inputs` restores parked buffers on the normal and error
/// paths, but a panic unwinding out of a run drops the `Executor` with them
/// still parked, and `Drop` only drained `self.buffers`. A counting allocator
/// makes the leak observable: `live` must return to its pre-run value.
///
/// Falsifier: delete the `parked_input_buffers` drain in `Drop for Executor`
/// and the final assertion fails with `live=1`.
#[test]
fn dropping_an_executor_with_a_parked_input_buffer_leaks_nothing() {
    use onnx_runtime_memory_governor::{DeviceAllocator, DeviceKey, HostAllocator, MemoryError};
    use std::ptr::NonNull;

    #[derive(Debug, Default)]
    struct CountingAllocator {
        inner: HostAllocator,
        live: AtomicUsize,
        allocated: AtomicUsize,
    }

    impl DeviceAllocator for CountingAllocator {
        fn device(&self) -> DeviceKey {
            self.inner.device()
        }

        fn allocate(
            &self,
            bytes: usize,
            align: usize,
        ) -> std::result::Result<NonNull<u8>, MemoryError> {
            let ptr = self.inner.allocate(bytes, align)?;
            self.live.fetch_add(1, Ordering::SeqCst);
            self.allocated.fetch_add(1, Ordering::SeqCst);
            Ok(ptr)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            self.live.fetch_sub(1, Ordering::SeqCst);
            // SAFETY: forwarded under this method's contract — `ptr`/`bytes`/
            // `align` are the triple `allocate` above returned from `inner`.
            unsafe { self.inner.deallocate(ptr, bytes, align) };
        }
    }

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([64]));
    graph.add_input(input);
    let out = graph.create_named_value("out", DataType::Float32, static_shape([64]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(input)], vec![out]));
    graph.add_output(out);

    let counting = Arc::new(CountingAllocator::default());
    let mut ep = CpuExecutionProvider::new().with_memory(counting.clone());
    ep.initialize(&Default::default()).unwrap();

    let mut executor = Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(ep)).unwrap();
    let vid = executor.input_index["input"];

    let tensor = Tensor::from_f32(&[64], &vec![1.0f32; 64]).unwrap();
    executor.run(&[("input", &tensor)]).unwrap();
    assert!(
        counting.allocated.load(Ordering::SeqCst) > 0,
        "the run allocated nothing through the counting allocator, so this \
         test could not observe a leak"
    );

    // Park the input's owned buffer exactly as `prepare_run_buffers` does when
    // it installs a zero-copy alias, then drop the executor mid-run. Nothing
    // else may reach `unbind_borrowed_inputs`.
    let bytes = tensor.as_bytes();
    let device = executor.buffers[&vid].device();
    // SAFETY: `tensor` outlives `executor`, and the handle is never written or
    // deallocated — it is dropped by `Drop for Executor`, where a borrowed
    // handle is a no-op free.
    let borrowed = unsafe {
        DeviceBuffer::from_borrowed_parts(
            bytes.as_ptr() as *mut std::ffi::c_void,
            device,
            bytes.len(),
            TensorLayout::contiguous().alignment,
        )
    };
    let owned = std::mem::replace(executor.buffers.get_mut(&vid).unwrap(), borrowed);
    executor.parked_input_buffers.push((vid, owned));

    drop(executor);
    assert_eq!(
        counting.live.load(Ordering::SeqCst),
        0,
        "dropping the executor leaked the parked input buffer"
    );
}

fn int64_initializer(graph: &mut Graph, name: &str, dims: Vec<usize>, values: &[i64]) -> ValueId {
    use onnx_runtime_ir::{TensorData, WeightRef};

    let value = graph.create_named_value(name, DataType::Int64, static_shape(dims.clone()));
    graph.set_initializer(
        value,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            dims,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )),
    );
    value
}

fn f32_scalar_initializer(graph: &mut Graph, name: &str, value: f32) -> ValueId {
    use onnx_runtime_ir::{TensorData, WeightRef};

    let tensor = graph.create_named_value(name, DataType::Float32, static_shape([]));
    graph.set_initializer(
        tensor,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![],
            value.to_le_bytes().to_vec(),
        )),
    );
    tensor
}

// DeepSeek-V2-Lite's graph-capture additive mask builds a query x key bias via
// CumSum/Unsqueeze/.../Where->Cast->Unsqueeze. ONNX shape inference leaves the
// query axis as a fresh internal `_d1` symbol, but its extent is exactly the
// current `input_ids` sequence length: Slice(CumSum(attention_mask),
// Shape(attention_mask)[1] - Shape(input_ids)[1], Shape(attention_mask)[1]).
// Capture prepare must recover that exact query axis rather than treating it as
// data-dependent or reserving a max-sequence-sized guess.
#[test]
fn prepare_workspace_resolves_deepseek_additive_mask_query_axis_exactly() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let batch = graph.create_symbol(Some("batch".into()));
    let total = graph.create_symbol(Some("past_seq_len + seq_len".into()));
    let seq = graph.create_symbol(Some("sequence_len".into()));
    let query = graph.create_symbol(Some("_d1".into()));

    let input_ids = graph.create_named_value(
        "input_ids",
        DataType::Int64,
        vec![Dim::Symbolic(batch), Dim::Symbolic(seq)],
    );
    graph.add_input(input_ids);
    let attention_mask = graph.create_named_value(
        "attention_mask",
        DataType::Int64,
        vec![Dim::Symbolic(batch), Dim::Symbolic(total)],
    );
    graph.add_input(attention_mask);
    let axis_1 = int64_initializer(&mut graph, "const_1d_1", vec![1], &[1]);
    let axis_2 = int64_initializer(&mut graph, "const_1d_2", vec![1], &[2]);
    let one = int64_initializer(&mut graph, "const_1_i64", vec![], &[1]);

    let cumsum = graph.create_named_value(
        "v_model.CumSum_5",
        DataType::Int64,
        vec![Dim::Symbolic(batch), Dim::Symbolic(total)],
    );
    graph.insert_node(Node::new(
        NodeId(0),
        "CumSum",
        vec![Some(attention_mask), Some(one)],
        vec![cumsum],
    ));
    let unsqueeze_6 = graph.create_named_value(
        "v_model.Unsqueeze_6",
        DataType::Int64,
        vec![Dim::Symbolic(batch), Dim::Static(1), Dim::Symbolic(total)],
    );
    graph.insert_node(Node::new(
        NodeId(1),
        "Unsqueeze",
        vec![Some(cumsum), Some(axis_1)],
        vec![unsqueeze_6],
    ));
    let input_len = graph.create_named_value("v_model.Shape_7", DataType::Int64, static_shape([1]));
    let mut shape_input = Node::new(NodeId(2), "Shape", vec![Some(input_ids)], vec![input_len]);
    shape_input
        .attributes
        .insert("start".into(), Attribute::Int(1));
    shape_input
        .attributes
        .insert("end".into(), Attribute::Int(2));
    graph.insert_node(shape_input);
    let mask_len = graph.create_named_value("v_model.Shape_8", DataType::Int64, static_shape([1]));
    let mut shape_mask = Node::new(
        NodeId(3),
        "Shape",
        vec![Some(attention_mask)],
        vec![mask_len],
    );
    shape_mask
        .attributes
        .insert("start".into(), Attribute::Int(1));
    shape_mask
        .attributes
        .insert("end".into(), Attribute::Int(2));
    graph.insert_node(shape_mask);
    let start = graph.create_named_value("v_model.Sub_9", DataType::Int64, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(4),
        "Sub",
        vec![Some(mask_len), Some(input_len)],
        vec![start],
    ));
    let sliced = graph.create_named_value(
        "v_model.Slice_10",
        DataType::Int64,
        vec![Dim::Symbolic(batch), Dim::Symbolic(query)],
    );
    graph.insert_node(Node::new(
        NodeId(5),
        "Slice",
        vec![Some(cumsum), Some(start), Some(mask_len), Some(axis_1)],
        vec![sliced],
    ));
    let unsqueeze_11 = graph.create_named_value(
        "v_model.Unsqueeze_11",
        DataType::Int64,
        vec![Dim::Symbolic(batch), Dim::Symbolic(query), Dim::Static(1)],
    );
    graph.insert_node(Node::new(
        NodeId(6),
        "Unsqueeze",
        vec![Some(sliced), Some(axis_2)],
        vec![unsqueeze_11],
    ));
    let ge = graph.create_named_value(
        "v_model.GreaterOrEqual_12",
        DataType::Bool,
        vec![
            Dim::Symbolic(batch),
            Dim::Symbolic(query),
            Dim::Symbolic(total),
        ],
    );
    graph.insert_node(Node::new(
        NodeId(7),
        "GreaterOrEqual",
        vec![Some(unsqueeze_11), Some(unsqueeze_6)],
        vec![ge],
    ));
    let unsqueeze_13 = graph.create_named_value(
        "v_model.Unsqueeze_13",
        DataType::Int64,
        vec![Dim::Symbolic(batch), Dim::Static(1), Dim::Symbolic(total)],
    );
    graph.insert_node(Node::new(
        NodeId(8),
        "Unsqueeze",
        vec![Some(attention_mask), Some(axis_1)],
        vec![unsqueeze_13],
    ));
    let cast_14 = graph.create_named_value(
        "v_model.Cast_14",
        DataType::Bool,
        vec![Dim::Symbolic(batch), Dim::Static(1), Dim::Symbolic(total)],
    );
    graph.insert_node(Node::new(
        NodeId(9),
        "Cast",
        vec![Some(unsqueeze_13)],
        vec![cast_14],
    ));
    let and = graph.create_named_value(
        "v_model.And_15",
        DataType::Bool,
        vec![
            Dim::Symbolic(batch),
            Dim::Symbolic(query),
            Dim::Symbolic(total),
        ],
    );
    graph.insert_node(Node::new(
        NodeId(10),
        "And",
        vec![Some(cast_14), Some(ge)],
        vec![and],
    ));
    let zero = f32_scalar_initializer(&mut graph, "const_0.0_f32", 0.0);
    let neg = f32_scalar_initializer(&mut graph, "const_-65504.0_f32", -65504.0);
    let where_out = graph.create_named_value(
        "v_model.Where_16",
        DataType::Float32,
        vec![
            Dim::Symbolic(batch),
            Dim::Symbolic(query),
            Dim::Symbolic(total),
        ],
    );
    graph.insert_node(Node::new(
        NodeId(11),
        "Where",
        vec![Some(and), Some(zero), Some(neg)],
        vec![where_out],
    ));
    let cast_17 = graph.create_named_value(
        "v_model.Cast_17",
        DataType::Float16,
        vec![
            Dim::Symbolic(batch),
            Dim::Symbolic(query),
            Dim::Symbolic(total),
        ],
    );
    graph.insert_node(Node::new(
        NodeId(12),
        "Cast",
        vec![Some(where_out)],
        vec![cast_17],
    ));
    let mask = graph.create_named_value(
        "v_model.Unsqueeze_18",
        DataType::Float16,
        vec![
            Dim::Symbolic(batch),
            Dim::Static(1),
            Dim::Symbolic(query),
            Dim::Symbolic(total),
        ],
    );
    graph.insert_node(Node::new(
        NodeId(13),
        "Unsqueeze",
        vec![Some(cast_17), Some(axis_1)],
        vec![mask],
    ));
    let attention = Node::new(
        NodeId(14),
        "Attention",
        vec![None, None, None, Some(mask)],
        vec![],
    );
    graph.insert_node(attention.clone());

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    for value in [
        input_ids,
        attention_mask,
        cumsum,
        unsqueeze_6,
        input_len,
        mask_len,
        start,
        sliced,
        unsqueeze_11,
        ge,
        unsqueeze_13,
        cast_14,
        and,
        zero,
        neg,
        where_out,
        cast_17,
        mask,
    ] {
        exec.value_shapes
            .insert(value, exec.graph.value(value).shape.clone());
    }
    let mut symbols = HashMap::new();
    symbols.insert(batch, 1);
    symbols.insert(seq, 1);
    symbols.insert(total, 2048);

    let resolved = exec
        .resolve_planned_workspace_input_shape(mask, &symbols, NodeId(14), &attention, 3)
        .unwrap();
    assert_eq!(resolved, PlannedInputShape::Exact(vec![1, 1, 1, 2048]));
}

/// Weight-derived caches must be drained before the buffers they are keyed on
/// are freed (#1726, #1735).
///
/// Both caches key entries on a weight's address and hold no claim on the memory
/// there, so an entry that outlives its buffer can be served to whatever lands on
/// that address next -- the right route, another model's numbers. Per-owner
/// eviction closes the within-executor case; this ordering is what closes the
/// cross-executor one, and it is invisible at runtime: reversing it produces no
/// error, just a window in which a recycled address inherits stale entries.
///
/// Asserted over the source because the failure has no observable signal to
/// probe for. Reversing the order in `Executor::drop` fails this test.
#[test]
fn weight_derived_caches_are_cleared_before_their_buffers_are_freed() {
    let source = include_str!("mod.rs");
    let drop_body = source
        .split_once("impl Drop for Executor {")
        .expect("Executor has a Drop impl")
        .1;
    let free = drop_body
        .find("self.buffers.drain()")
        .expect("the drop body frees the executor's buffers");
    for clear in [
        "clear_weight_transpose_caches()",
        "clear_mlas_packed_caches()",
    ] {
        let at = drop_body
            .find(clear)
            .unwrap_or_else(|| panic!("the drop body must call {clear}"));
        assert!(
            at < free,
            "{clear} runs after the buffers it protects are freed, leaving \
             entries keyed on addresses the allocator may hand to the next model"
        );
    }
}

#[test]
fn qmoe_residency_plan_default_policy_matches_whole_bank_resident_for_pageable_candidates() {
    let (graph, weights, _path) = qmoe_expert_region_fixture();
    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ep = WeightDeliveryEp::new(true, deliveries);
    let executor = Executor::build(graph, weights, Arc::new(ep)).unwrap();

    let plan = executor.residency_plan();
    let candidates = executor.expert_region_candidates();
    assert_eq!(plan.policy_name(), "whole_bank_resident");
    // Same cardinality as the expert region candidates: the plan covers
    // exactly the values that have a QMoE catalog, no more, no fewer.
    assert_eq!(plan.len(), candidates.len());
    assert_eq!(plan.resident_count(), plan.len());
    assert_eq!(plan.degraded_count(), 0);
    for value in candidates.keys() {
        assert_eq!(
            plan.decision(*value),
            Some(&onnx_runtime_ep_api::ResidencyDecision::WholeBankResident { reason: None })
        );
    }

    // Existing behavior fully unchanged: still one lazy handle per
    // initializer, each with exactly one region.
    assert_eq!(executor.weight_handles.len(), 2);
}

#[test]
fn qmoe_residency_plan_surfaces_non_pageable_reason_without_changing_handles() {
    let (mut graph, weights, path) = qmoe_expert_region_fixture();
    let packed_value = graph
        .initializers
        .keys()
        .copied()
        .find(|value| graph.value(*value).name.as_deref() == Some("fc1_packed"))
        .expect("fc1_packed initializer must exist");
    graph.set_initializer(
        packed_value,
        WeightRef::External {
            path: path.clone(),
            offset: 0,
            length: 24,
            dtype: DataType::Uint8,
            dims: vec![6, 4],
        },
    );

    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ep = WeightDeliveryEp::new(true, deliveries);
    let executor = Executor::build(graph, weights, Arc::new(ep)).unwrap();

    let plan = executor.residency_plan();
    match plan.decision(packed_value) {
        Some(onnx_runtime_ep_api::ResidencyDecision::WholeBankResident {
            reason:
                Some(onnx_runtime_ep_api::ResidencyDegradationReason::NonPageableCatalog(reason)),
        }) => {
            assert_eq!(
                reason,
                &onnx_runtime_loader::NonPageableReason::NotExpertMajor
            );
        }
        other => panic!("expected non-pageable whole-bank reason, got {other:?}"),
    }
    assert_eq!(
        plan.degraded_count(),
        1,
        "non-pageable reason counts as degraded"
    );
    assert_eq!(executor.weight_handles.len(), 2);
}

/// Prove the residency-policy seam is substitutable at the executor's own
/// candidate/boundary wiring, without any production dispatch path consuming
/// the result -- this is the "test-only alternate policy" required to show
/// the trait is not an inert marker.
#[test]
fn qmoe_residency_plan_seam_is_substitutable_with_an_alternate_policy() {
    use onnx_runtime_ep_api::{ResidencyDecision, ResidencyPolicy, ResidencyPolicyInput};

    struct AlwaysSplit;
    impl ResidencyPolicy for AlwaysSplit {
        fn name(&self) -> &'static str {
            "test_always_split"
        }
        fn decide(&self, input: &ResidencyPolicyInput<'_>) -> ResidencyDecision {
            if input.catalog.is_pageable() {
                ResidencyDecision::PerExpertCandidate {
                    experts: (0..input.catalog.layout().experts).collect(),
                }
            } else {
                ResidencyDecision::WholeBankResident { reason: None }
            }
        }
    }

    let (graph, weights, _path) = qmoe_expert_region_fixture();
    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ep = WeightDeliveryEp::new(true, deliveries);
    let executor = Executor::build(graph, weights, Arc::new(ep)).unwrap();

    let candidates = executor.expert_region_candidates();
    let plan =
        crate::executor::build::plan_residency_with(&executor.graph, candidates, &AlwaysSplit);
    assert_eq!(plan.policy_name(), "test_always_split");
    assert!(!plan.is_empty());
    for value in candidates.keys() {
        assert!(matches!(
            plan.decision(*value),
            Some(ResidencyDecision::PerExpertCandidate { .. })
        ));
    }

    // Substituting the policy must not touch the default plan stored on the
    // executor, nor the weight-handle/binding output.
    assert_eq!(
        executor.residency_plan().policy_name(),
        "whole_bank_resident"
    );
    assert_eq!(executor.weight_handles.len(), 2);
}

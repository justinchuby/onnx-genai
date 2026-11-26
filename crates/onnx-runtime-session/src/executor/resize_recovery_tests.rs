//! Real CPU kernels with test-only manager-backed ordinary buffer allocation.
//! This does not enable a production CPU allocator or change Native KV policy.
use super::*;
use onnx_runtime_ep_api::{BoundBufferOwnership, EpConfig, Fence};
use onnx_runtime_memory_api::{DeviceKey, HostAllocator, MemoryRole, Tier};
use onnx_runtime_memory_governor::{
    AllocationPublication, AllocationRequest, LeaseLedger, LedgerGovernor, ManagedAllocation,
    MemoryGovernor, ProcessMemoryManager, RegisteredMemoryHolder, ScopedMemoryBinding,
};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize},
};

struct ManagedCpu {
    cpu: CpuExecutionProvider,
    binding: ScopedMemoryBinding,
    holder: RegisteredMemoryHolder,
    manager: ProcessMemoryManager,
    books: Arc<LedgerGovernor>,
    allocations: AtomicUsize,
    releases: AtomicUsize,
    fail_release_once: AtomicBool,
    retained: Mutex<Vec<ManagedAllocation>>,
}
impl ManagedCpu {
    fn new(limit: u64) -> Arc<Self> {
        let manager = ProcessMemoryManager::new().unwrap();
        let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::HOST,
            0,
            limit,
            0,
        )));
        let authority = manager
            .register_authority(
                DeviceKey::HOST,
                "resize test Host",
                Arc::new(()),
                books.clone(),
            )
            .unwrap();
        let context = manager
            .register_provider_context(DeviceKey::HOST, "resize test CPU", Arc::new(()))
            .unwrap();
        let mechanism = manager
            .register_allocator(
                &context,
                &authority,
                "uncached resize test Host",
                Arc::new(HostAllocator),
            )
            .unwrap();
        let binding = manager.bind_registered(&mechanism).unwrap();
        let holder = manager
            .register_holder(&authority, "resize test ordinary buffers", None)
            .unwrap();
        Arc::new(Self {
            cpu: CpuExecutionProvider::initialized_with_resources(Default::default()).unwrap(),
            binding,
            holder,
            manager,
            books,
            allocations: AtomicUsize::new(0),
            releases: AtomicUsize::new(0),
            fail_release_once: AtomicBool::new(false),
            retained: Mutex::new(Vec::new()),
        })
    }
    fn settle(&self, owner: ManagedAllocation) -> onnx_runtime_ep_api::Result<()> {
        let outcome = owner.release_now().map_err(|source| EpError::Backend {
            operation: "release real managed resize buffer",
            source: Box::new(source),
        })?;
        if !outcome.is_complete() {
            return Err(EpError::KernelFailed(format!(
                "resize buffer release retained funded ownership: {outcome:?}"
            )));
        }
        self.releases.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn release_retained(&self) {
        for owner in self.retained.lock().unwrap().drain(..) {
            self.settle(owner).unwrap();
        }
    }
    fn used(&self) -> u64 {
        let used = self.books.used(Tier::Host);
        assert_eq!(self.manager.process_used(Tier::Host), used);
        used
    }
}
impl ExecutionProvider for ManagedCpu {
    fn name(&self) -> &str {
        "managed_resize_test_cpu"
    }
    fn device_type(&self) -> DeviceType {
        self.cpu.device_type()
    }
    fn device_id(&self) -> onnx_runtime_ir::DeviceId {
        self.cpu.device_id()
    }
    fn initialize(&mut self, config: &EpConfig) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.initialize(config)
    }
    fn shutdown(&mut self) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.shutdown()
    }
    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> KernelMatch {
        self.cpu.supports_op(op, opset, shapes, dtypes, layouts)
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
        let bytes = u64::try_from(size).unwrap();
        let owner = self
            .binding
            .allocate(
                AllocationRequest::managed(
                    size,
                    alignment,
                    Tier::Host,
                    MemoryRole::Activation,
                    self.holder.clone(),
                    bytes,
                ),
                AllocationPublication::exclusive(bytes, bytes, bytes),
            )
            .map_err(|source| EpError::Backend {
                operation: "allocate real managed resize buffer",
                source: Box::new(source),
            })?;
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceBuffer::from_managed_allocation(
            Box::new(owner),
            self.device_id(),
        ))
    }
    fn deallocate(&self, buffer: DeviceBuffer) -> onnx_runtime_ep_api::Result<()> {
        if buffer.is_borrowed() {
            return Ok(());
        }
        let BoundBufferOwnership::Managed(owner) = buffer
            .into_bound_ownership()
            .expect("test provider issues manager-owned buffers")
        else {
            panic!("test provider never issues bare bound allocations");
        };
        if self.fail_release_once.swap(false, Ordering::Relaxed) {
            // Inject only an unchanged release refusal. The complete real
            // manager owner stays retained and funded; no fake refund/free.
            self.retained.lock().unwrap().push(*owner);
            return Err(EpError::KernelFailed(
                "resize test retained old allocation after unchanged release refusal".into(),
            ));
        }
        self.settle(*owner)
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
    fn copy_from_host(
        &self,
        src: &[u8],
        dst: &mut DeviceBuffer,
    ) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.copy_from_host(src, dst)
    }
    fn copy_from_host_at(
        &self,
        src: &[u8],
        dst: &mut DeviceBuffer,
        offset: usize,
    ) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.copy_from_host_at(src, dst, offset)
    }
    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.copy_to_host(src, dst)
    }
    fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.sync()
    }
}
fn fixture(limit: u64) -> (Executor, Arc<ManagedCpu>, ValueId) {
    let ep = ManagedCpu::new(limit);
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let n = graph.create_symbol(Some("elements".into()));
    let shape = vec![Dim::Symbolic(n)];
    let input = graph.create_named_value("input", DataType::Float32, shape.clone());
    let middle = graph.create_named_value("middle", DataType::Float32, shape.clone());
    let output = graph.create_named_value("output", DataType::Float32, shape);
    graph.add_input(input);
    graph.add_output(output);
    graph.insert_node(Node::new(NodeId(0), "Neg", vec![Some(input)], vec![middle]));
    graph.insert_node(Node::new(
        NodeId(1),
        "Neg",
        vec![Some(middle)],
        vec![output],
    ));
    let mut executor =
        Executor::build(graph, Arc::new(WeightStore::new()), ep.clone(), None).unwrap();
    // Isolate actual backing replacement from unrelated in-place/dead-value
    // optimizations; all inference still uses the real CPU Neg kernels.
    executor.compute_in_place_enabled = false;
    executor.release_dead_values_enabled = false;
    (executor, ep, middle)
}
fn run(executor: &mut Executor, values: &[f32]) {
    let input = Tensor::from_f32(&[values.len()], values).unwrap();
    let output = executor.run(&[("input", &input)]).unwrap();
    assert_eq!(output[0].try_as_slice_f32().unwrap(), values);
}
fn assert_complete_cleanup(executor: Executor, ep: &Arc<ManagedCpu>) {
    drop(executor);
    ep.release_retained();
    assert_eq!(ep.used(), 0);
    assert_eq!(
        ep.allocations.load(Ordering::Relaxed),
        ep.releases.load(Ordering::Relaxed)
    );
    assert!(ep.manager.snapshot().unwrap().allocations.is_empty());
}
#[test]
fn managed_grow_denial_keeps_original_backing_shape_and_small_retry_then_grows() {
    let (mut executor, ep, middle) = fixture(128);
    run(&mut executor, &[2.0, 5.0]);
    assert_eq!(ep.used(), 16);
    let original = executor.buffers[&middle].as_ptr();
    let dtype = executor.value_dtypes[&middle];
    let blocker = ep.allocate(96, 64).unwrap();
    assert_eq!(ep.used(), 112);
    let error = executor
        .ensure_buffer(middle, DataType::Float32, &[8])
        .unwrap_err();
    assert!(
        error.to_string().contains("managed resize buffer"),
        "{error}"
    );
    assert_eq!(executor.buffers[&middle].as_ptr(), original);
    assert_eq!(executor.buffers[&middle].len(), 8);
    assert_eq!(executor.buffer_shapes[&middle], [2]);
    assert_eq!(executor.value_dtypes[&middle], dtype);
    assert_eq!(ep.used(), 112);
    run(&mut executor, &[3.0, 7.0]);
    assert_eq!(executor.buffers[&middle].as_ptr(), original);
    ep.deallocate(blocker).unwrap();
    executor
        .ensure_buffer(middle, DataType::Float32, &[8])
        .unwrap();
    assert_eq!(executor.buffer_shapes[&middle], [8]);
    assert_eq!(executor.buffers[&middle].len(), 32);
    run(&mut executor, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_complete_cleanup(executor, &ep);
}
#[test]
fn managed_resize_admission_counts_live_old_and_new_peak() {
    let (mut executor, ep, middle) = fixture(64);
    run(&mut executor, &[2.0, 5.0]);
    let original = executor.buffers[&middle].as_ptr();
    let blocker = ep.allocate(40, 64).unwrap();
    assert_eq!(ep.used(), 56);
    // Old-first freeing would incorrectly admit 56-8+16=64. The actual
    // transactional peak is 56+16=72 and must be refused under this cap.
    assert!(
        executor
            .ensure_buffer(middle, DataType::Float32, &[4])
            .is_err()
    );
    assert_eq!(executor.buffers[&middle].as_ptr(), original);
    assert_eq!(ep.used(), 56);
    ep.deallocate(blocker).unwrap();
    executor
        .ensure_buffer(middle, DataType::Float32, &[4])
        .unwrap();
    run(&mut executor, &[1.0, 2.0, 3.0, 4.0]);
    assert_complete_cleanup(executor, &ep);
}
#[test]
fn old_release_error_keeps_published_replacement_coherent_and_old_claim_funded() {
    let (mut executor, ep, middle) = fixture(128);
    run(&mut executor, &[2.0, 5.0]);
    let original = executor.buffers[&middle].as_ptr();
    ep.fail_release_once.store(true, Ordering::Relaxed);
    let error = executor
        .ensure_buffer(middle, DataType::Float32, &[8])
        .unwrap_err();
    assert!(error.to_string().contains("retained old allocation"));
    assert_ne!(executor.buffers[&middle].as_ptr(), original);
    assert_eq!(executor.buffer_shapes[&middle], [8]);
    assert_eq!(executor.buffers[&middle].len(), 32);
    assert_eq!(ep.used(), 48);
    let allocations = ep.allocations.load(Ordering::Relaxed);
    executor
        .ensure_buffer(middle, DataType::Float32, &[8])
        .unwrap();
    assert_eq!(ep.allocations.load(Ordering::Relaxed), allocations);
    ep.release_retained();
    assert_eq!(ep.used(), 40);
    run(&mut executor, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_complete_cleanup(executor, &ep);
}
#[test]
fn shared_old_owner_survives_denied_resize_and_external_view_survives_replacement() {
    let (mut executor, ep, middle) = fixture(128);
    run(&mut executor, &[2.0, 5.0]);
    let original = executor.buffers.remove(&middle).unwrap();
    let provider: Arc<dyn ExecutionProvider> = ep.clone();
    let shared = SharedTensorBuffer::new(provider, original);
    let external = Arc::clone(&shared);
    executor.buffers.insert(middle, shared.alias());
    executor.shared_buffers.insert(middle, shared);
    let blocker = ep.allocate(96, 64).unwrap();
    assert_eq!(ep.used(), 112);
    assert!(
        executor
            .ensure_buffer(middle, DataType::Float32, &[8])
            .is_err()
    );
    assert!(Arc::ptr_eq(&executor.shared_buffers[&middle], &external));
    assert_eq!(executor.buffer_shapes[&middle], [2]);
    let mut bytes = [0_u8; 8];
    ep.copy_to_host(external.buffer(), &mut bytes).unwrap();
    assert_eq!(
        bytes,
        [-2_f32, -5.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
            .as_slice()
    );
    ep.deallocate(blocker).unwrap();
    executor
        .ensure_buffer(middle, DataType::Float32, &[8])
        .unwrap();
    assert!(!executor.shared_buffers.contains_key(&middle));
    assert_eq!(ep.used(), 48, "old view retains8 plus input8 and new32");
    ep.copy_to_host(external.buffer(), &mut bytes).unwrap();
    assert_eq!(f32::from_le_bytes(bytes[..4].try_into().unwrap()), -2.0);
    drop(external);
    assert_eq!(ep.used(), 40);
    run(&mut executor, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_complete_cleanup(executor, &ep);
}

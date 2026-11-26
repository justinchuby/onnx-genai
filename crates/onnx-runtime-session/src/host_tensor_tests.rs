//! Real managed CPU allocation/ownership tests; no Engine worker enablement.
use crate::{HostTensorAllocator, InferenceSession, SessionError};
use onnx_runtime_ep_api::{
    DeviceBuffer, EpConfig, EpError, ExecutionProvider, Kernel, KernelMatch,
};
use onnx_runtime_ep_cpu::{CpuBufferMemory, CpuExecutionProvider, CpuMemoryResources};
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, DeviceType, Graph, Node, NodeId, Shape, TensorLayout,
    static_shape,
};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn fixture(
    limit: u64,
) -> (
    Arc<CpuExecutionProvider>,
    ProcessMemoryManager,
    Arc<LedgerGovernor>,
) {
    let manager = ProcessMemoryManager::new().unwrap();
    let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::HOST,
        0,
        limit,
        0,
    )));
    let domain = CpuBufferMemory::governed(&manager, governor.clone()).unwrap();
    let resources = CpuMemoryResources::default()
        .with_buffer_memory(domain)
        .unwrap();
    let provider = Arc::new(CpuExecutionProvider::initialized_with_resources(resources).unwrap());
    (provider, manager, governor)
}
#[test]
fn initialized_fill_error_and_unwind_release_real_managed_backing() {
    let (provider, manager, governor) = fixture(16);
    let context = HostTensorAllocator::new(provider).unwrap();
    let error = context
        .fill(DataType::Float32, vec![4], |bytes| {
            assert_eq!(
                bytes, &[0; 16],
                "safe callbacks may read every supplied byte"
            );
            bytes[0] = 77;
            Err(SessionError::Internal("deliberate fill refusal".into()).into())
        })
        .unwrap_err();
    assert!(error.to_string().contains("deliberate fill refusal"));
    assert!(matches!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<SessionError>()),
        Some(SessionError::Internal(message)) if message == "deliberate fill refusal"
    ));
    assert_eq!(governor.used(Tier::Host), 0);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = context.fill(DataType::Float32, vec![4], |bytes| {
            assert_eq!(bytes, &[0; 16]);
            panic!("deliberate fill unwind");
        });
    }));
    assert!(panic.is_err());
    assert_eq!(governor.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
    let retried = context.from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(retried.try_as_slice_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    drop(retried);
    assert_eq!(governor.used(Tier::Host), 0);
}
#[test]
fn zero_size_and_clone_keep_actual_capacity_funded() {
    let (provider, manager, governor) = fixture(9);
    let context = HostTensorAllocator::new(provider).unwrap();
    let empty = context.zeros(DataType::Float32, vec![0]).unwrap();
    assert!(empty.as_bytes().is_empty());
    assert_eq!(governor.used(Tier::Host), 1);
    let original = context.from_i64(&[1], &[17]).unwrap();
    assert_eq!(governor.used(Tier::Host), 9);
    assert!(original.try_clone().is_err());
    drop(empty);
    assert_eq!(original.to_vec_i64(), [17]);
    assert_eq!(governor.used(Tier::Host), 8);
    drop(original);
    assert_eq!(manager.process_used(Tier::Host), 0);
}
#[test]
fn transpose_copy_output_remains_readable_and_funded_after_session_drop() {
    let (provider, manager, governor) = fixture(256);
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([2, 2]));
    let output = graph.create_named_value("output", DataType::Float32, static_shape([2, 2]));
    graph.add_input(input);
    graph.add_output(output);
    let mut transpose = Node::new(NodeId(0), "Transpose", vec![Some(input)], vec![output]);
    transpose
        .attributes
        .insert("perm".into(), Attribute::Ints(vec![1, 0]));
    graph.insert_node(transpose);
    let mut session = InferenceSession::from_graph_with_provider(
        graph,
        Arc::new(onnx_runtime_loader::WeightStore::new()),
        std::path::Path::new("."),
        provider.clone(),
        None,
    )
    .unwrap();
    let context = session.host_tensor_allocator().unwrap();
    let input = context.from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let outputs = session.run(&[("input", &input)]).unwrap();
    assert_eq!(
        outputs[0].try_as_slice_f32().unwrap(),
        &[1.0, 3.0, 2.0, 4.0]
    );
    drop(input);
    drop(context);
    drop(session);
    drop(provider);
    assert_eq!(
        governor.used(Tier::Host),
        16,
        "only the caller-owned output remains live"
    );
    assert_eq!(manager.process_used(Tier::Host), 16);
    assert_eq!(
        outputs[0].try_as_slice_f32().unwrap(),
        &[1.0, 3.0, 2.0, 4.0]
    );
    drop(outputs);
    assert_eq!(governor.used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
}

/// Inject only a fallible transfer; allocation and release are real managed CPU.
struct RefusingUpload {
    cpu: Arc<CpuExecutionProvider>,
    refuse: AtomicBool,
}
impl ExecutionProvider for RefusingUpload {
    fn name(&self) -> &str {
        "managed CPU upload refusal test"
    }
    fn device_type(&self) -> DeviceType {
        self.cpu.device_type()
    }
    fn device_id(&self) -> DeviceId {
        self.cpu.device_id()
    }
    fn initialize(&mut self, _: &EpConfig) -> onnx_runtime_ep_api::Result<()> {
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
    fn copy_from_host(
        &self,
        src: &[u8],
        dst: &mut DeviceBuffer,
    ) -> onnx_runtime_ep_api::Result<()> {
        if self.refuse.load(Ordering::Relaxed) {
            return Err(EpError::KernelFailed("deliberate upload refusal".into()));
        }
        self.cpu.copy_from_host(src, dst)
    }
    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::Fence> {
        self.cpu.copy_async(src, dst, size)
    }
    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.copy_to_host(src, dst)
    }
    fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.sync()
    }
}
#[test]
fn upload_failure_after_real_allocation_releases_instead_of_quarantining_bare_token() {
    let (cpu, manager, governor) = fixture(4);
    let provider = Arc::new(RefusingUpload {
        cpu,
        refuse: AtomicBool::new(true),
    });
    let context = HostTensorAllocator::new(provider.clone()).unwrap();
    let error = context
        .from_raw(DataType::Float32, vec![1], &3_f32.to_le_bytes())
        .unwrap_err();
    assert!(error.to_string().contains("deliberate upload refusal"));
    assert_eq!(governor.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
    provider.refuse.store(false, Ordering::Relaxed);
    let retried = context
        .from_raw(DataType::Float32, vec![1], &3_f32.to_le_bytes())
        .unwrap();
    assert_eq!(retried.try_as_slice_f32().unwrap(), &[3.0]);
    drop(retried);
    assert_eq!(governor.used(Tier::Host), 0);
}

#[test]
fn strided_sequence_view_copies_directly_into_admitted_backing() {
    let (provider, manager, governor) = fixture(32);
    let context = HostTensorAllocator::new(provider).unwrap();
    let source = context.from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let mut element = crate::sequence::SeqTensor::new(source).unwrap();
    element.layout = TensorLayout::strided(vec![1, 2]);
    let copied = context
        .fill(DataType::Float32, vec![2, 2], |bytes| {
            element.copy_host_contiguous_into(bytes)
        })
        .unwrap();
    assert_eq!(copied.try_as_slice_f32().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    assert_eq!(governor.used(Tier::Host), 32);
    assert!(context.zeros(DataType::Float32, vec![1]).is_err());
    drop(element);
    assert_eq!(governor.used(Tier::Host), 16);
    assert_eq!(copied.try_as_slice_f32().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    drop(copied);
    assert_eq!(governor.used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
}

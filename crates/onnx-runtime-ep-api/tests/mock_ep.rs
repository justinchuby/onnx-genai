//! Integration test: a mock EP that registers ops and resolves kernels through
//! the registry, exercising the public `onnx-runtime-ep-api` contract.

use std::ffi::c_void;

use onnx_runtime_ep_api::{
    Cost, DeviceBuffer, EpConfig, EpRegistry, ExecutionProvider, Fence, Kernel, KernelFactory,
    KernelMatch, OpKey, OpRegistry, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{static_shape, DeviceId, DeviceType, Node, NodeId, Shape, TensorLayout};

/// A trivial kernel that does nothing but report success.
struct AddKernel;

impl Kernel for AddKernel {
    fn execute(&self, _inputs: &[TensorView], _outputs: &mut [TensorMut]) -> Result<()> {
        Ok(())
    }

    fn estimated_flops(&self) -> Option<u64> {
        Some(0)
    }
}

/// Factory that produces `AddKernel`s.
struct AddFactory;

impl KernelFactory for AddFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(AddKernel))
    }
}

/// A mock CPU EP backed by host allocations.
#[derive(Default)]
struct MockEp {
    registry: OpRegistry,
    initialized: bool,
}

impl MockEp {
    fn new() -> Self {
        let mut registry = OpRegistry::new();
        registry.register(OpKey::new("Add", "", 7), Box::new(AddFactory));
        // A newer opset variant to exercise version resolution.
        registry.register(OpKey::new("Add", "", 14), Box::new(AddFactory));
        Self {
            registry,
            initialized: false,
        }
    }
}

impl ExecutionProvider for MockEp {
    fn name(&self) -> &str {
        "mock_ep"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }

    fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    fn supports_op(&self, op: &Node, _shapes: &[Shape], _layouts: &[TensorLayout]) -> KernelMatch {
        if self.registry.lookup(&op.op_type, &op.domain, 17).is_some() {
            KernelMatch::Supported {
                cost: Cost::new(1.0, 0.5, 0.0).with_bytes_moved(256),
                required_input_layouts: None,
                output_layouts: vec![TensorLayout::contiguous()],
            }
        } else {
            KernelMatch::Unsupported
        }
    }

    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let factory = self
            .registry
            .lookup(&op.op_type, &op.domain, 17)
            .expect("supports_op should gate this");
        factory.create(op, shapes)
    }

    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer> {
        let boxed = vec![0u8; size].into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut c_void;
        // SAFETY: fresh, unique host allocation of `size` bytes owned by this EP.
        Ok(unsafe { DeviceBuffer::from_raw_parts(ptr, DeviceId::cpu(), size, alignment) })
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()> {
        let size = buffer.len();
        let ptr = buffer.into_raw() as *mut u8;
        // SAFETY: reconstruct the exact boxed slice produced by `allocate`.
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, size)));
        }
        Ok(())
    }

    fn copy(&self, _src: &DeviceBuffer, _dst: &mut DeviceBuffer, _size: usize) -> Result<()> {
        Ok(())
    }

    fn copy_async(
        &self,
        _src: &DeviceBuffer,
        _dst: &mut DeviceBuffer,
        _size: usize,
    ) -> Result<Fence> {
        Ok(Fence::default())
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

fn add_node() -> Node {
    Node::new(NodeId(0), "Add", vec![None, None], vec![])
}

#[test]
fn op_registry_resolves_highest_matching_opset() {
    let mut reg = OpRegistry::new();
    reg.register(OpKey::new("Add", "", 7), Box::new(AddFactory));
    reg.register(OpKey::new("Add", "", 14), Box::new(AddFactory));
    assert_eq!(reg.len(), 2);
    // opset 13 resolves to the v7 entry (highest since_version <= 13).
    assert!(reg.lookup("Add", "", 13).is_some());
    // opset 6 has no matching entry.
    assert!(reg.lookup("Add", "", 6).is_none());
    // Unknown op.
    assert!(reg.lookup("Mul", "", 20).is_none());
}

#[test]
fn mock_ep_supports_and_builds_kernel() {
    let mut ep = MockEp::new();
    ep.initialize(&EpConfig::default()).unwrap();

    let node = add_node();
    let shapes = vec![static_shape([2, 3]), static_shape([2, 3])];
    let layouts = vec![TensorLayout::contiguous(), TensorLayout::contiguous()];

    let m = ep.supports_op(&node, &shapes, &layouts);
    assert!(m.is_supported());

    let kernel = ep.get_kernel(&node, &[vec![2, 3], vec![2, 3]]).unwrap();
    assert_eq!(kernel.estimated_flops(), Some(0));
    kernel.execute(&[], &mut []).unwrap();
}

#[test]
fn ep_registry_lists_candidates_in_priority_order() {
    let mut registry = EpRegistry::new();
    let id = registry.register(Box::new(MockEp::new()));
    assert_eq!(registry.priority(), &[id]);

    let node = add_node();
    let shapes = vec![static_shape([4]), static_shape([4])];
    let layouts = vec![TensorLayout::contiguous(), TensorLayout::contiguous()];

    let candidates = registry.candidates_for_op(&node, &shapes, &layouts);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].0, id);
    assert!(candidates[0].1.is_supported());

    // An unsupported op yields no candidates.
    let unsupported = Node::new(NodeId(1), "NoSuchOp", vec![], vec![]);
    assert!(registry
        .candidates_for_op(&unsupported, &[], &[])
        .is_empty());
}

#[test]
fn mock_ep_allocate_deallocate_roundtrip() {
    let ep = MockEp::new();
    let buf = ep.allocate(256, 64).unwrap();
    assert_eq!(buf.len(), 256);
    assert_eq!(buf.alignment(), 64);
    // Single deallocation — a double free would trip ASan/Miri.
    ep.deallocate(buf).unwrap();
}

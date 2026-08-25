//! Integration test: a mock EP that registers ops and resolves kernels through
//! the registry, exercising the public `onnx-runtime-ep-api` contract.

use std::ffi::c_void;

use onnx_runtime_ep_api::{
    Cost, DeviceBuffer, EpConfig, EpRegistry, ExecutionProvider, Fence, Kernel, KernelFactory,
    KernelMatch, OpKey, OpRegistry, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{
    DataType, DeviceId, DeviceType, FrozenGraph, Graph, Node, NodeId, Shape, TensorLayout,
    static_shape,
};

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

struct TaggedKernel(u64);

impl Kernel for TaggedKernel {
    fn execute(&self, _inputs: &[TensorView], _outputs: &mut [TensorMut]) -> Result<()> {
        Ok(())
    }

    fn estimated_flops(&self) -> Option<u64> {
        Some(self.0)
    }
}

struct TaggedFactory(u64);

impl KernelFactory for TaggedFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(TaggedKernel(self.0)))
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
    fn consume_route_residency_at_boundary(&self) -> Result<()> {
        Ok(())
    }

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

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        _shapes: &[Shape],
        _input_dtypes: &[onnx_runtime_ir::DataType],
        _layouts: &[TensorLayout],
    ) -> KernelMatch {
        if self
            .registry
            .lookup(&op.op_type, &op.domain, opset)
            .is_some()
        {
            KernelMatch::Supported {
                cost: Cost::new(1.0, 0.5, 0.0).with_bytes_moved(256),
                required_input_layouts: None,
                output_layouts: vec![TensorLayout::contiguous()],
            }
        } else {
            KernelMatch::unsupported("mock EP has no registered kernel for this op")
        }
    }

    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>], opset: u64) -> Result<Box<dyn Kernel>> {
        let factory = self
            .registry
            .lookup(&op.op_type, &op.domain, opset)
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

fn lookup_tag(registry: &OpRegistry, op_type: &str, domain: &str, opset: u64) -> Option<u64> {
    registry
        .lookup(op_type, domain, opset)
        .map(|factory| factory.create(&add_node(), &[]).unwrap())
        .and_then(|kernel| kernel.estimated_flops())
}

#[test]
fn op_registry_resolves_highest_matching_opset() {
    let mut reg = OpRegistry::new();
    reg.register(OpKey::new("Add", "", 14), Box::new(TaggedFactory(14)));
    reg.register(OpKey::new("Add", "", 7), Box::new(TaggedFactory(7)));
    assert_eq!(reg.len(), 2);

    assert_eq!(lookup_tag(&reg, "Add", "", 7), Some(7));
    assert_eq!(lookup_tag(&reg, "Add", "", 13), Some(7));
    assert_eq!(lookup_tag(&reg, "Add", "", 14), Some(14));
    assert_eq!(lookup_tag(&reg, "Add", "", 20), Some(14));
}

#[test]
fn op_registry_reports_below_earliest_version() {
    let mut reg = OpRegistry::new();
    reg.register(OpKey::new("Add", "", 7), Box::new(AddFactory));

    assert!(reg.lookup("Add", "", 6).is_none());
    assert!(!reg.supports("Add", "", 6));
    assert_eq!(reg.earliest_since_version("Add", ""), Some(7));
    assert_eq!(reg.earliest_since_version("Mul", ""), None);
}

#[test]
fn op_registry_aliases_default_onnx_domains() {
    let mut reg = OpRegistry::new();
    reg.register(OpKey::new("Add", "ai.onnx", 7), Box::new(TaggedFactory(7)));
    reg.register(OpKey::new("Mul", "", 9), Box::new(TaggedFactory(9)));

    let empty_domain = reg.lookup("Add", "", 7).unwrap();
    let named_domain = reg.lookup("Add", "ai.onnx", 7).unwrap();
    assert!(std::ptr::eq(empty_domain, named_domain));
    let empty_domain = reg.lookup("Mul", "", 9).unwrap();
    let named_domain = reg.lookup("Mul", "ai.onnx", 9).unwrap();
    assert!(std::ptr::eq(empty_domain, named_domain));
    assert!(reg.supports("Add", "", 7));
    assert!(reg.supports("Add", "ai.onnx", 7));
    assert_eq!(reg.earliest_since_version("Add", ""), Some(7));
    assert_eq!(reg.earliest_since_version("Add", "ai.onnx"), Some(7));
}

#[test]
fn op_registry_isolates_contrib_domain() {
    let mut reg = OpRegistry::new();
    reg.register(OpKey::new("Add", "", 1), Box::new(TaggedFactory(1)));
    reg.register(
        OpKey::new("Add", "com.microsoft", 5),
        Box::new(TaggedFactory(5)),
    );

    assert_eq!(lookup_tag(&reg, "Add", "", 10), Some(1));
    assert_eq!(lookup_tag(&reg, "Add", "ai.onnx", 10), Some(1));
    assert_eq!(lookup_tag(&reg, "Add", "com.microsoft", 10), Some(5));
    assert_eq!(reg.earliest_since_version("Add", ""), Some(1));
    assert_eq!(reg.earliest_since_version("Add", "com.microsoft"), Some(5));
}

#[test]
fn mock_ep_supports_and_builds_kernel() {
    let mut ep = MockEp::new();
    ep.initialize(&EpConfig::default()).unwrap();

    let node = add_node();
    let shapes = vec![static_shape([2, 3]), static_shape([2, 3])];
    let layouts = vec![TensorLayout::contiguous(), TensorLayout::contiguous()];

    let m = ep.supports_op(&node, 17, &shapes, &[], &layouts);
    assert!(m.is_supported());

    let kernel = ep.get_kernel(&node, &[vec![2, 3], vec![2, 3]], 17).unwrap();
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

    let candidates =
        registry.candidates_for_op(&node, 17, &shapes, &[DataType::Float32; 2], &layouts);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].0, id);
    assert!(candidates[0].1.is_supported());

    // An unsupported op yields no candidates.
    let unsupported = Node::new(NodeId(1), "NoSuchOp", vec![], vec![]);
    assert!(
        registry
            .candidates_for_op(&unsupported, 17, &[], &[], &[])
            .is_empty()
    );
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

fn graph_chain(ops: &[(&str, &str)]) -> FrozenGraph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    graph.opset_imports.insert("ai.onnx".into(), 17);
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let input = graph.create_value(DataType::Float32, static_shape([2]));
    graph.add_input(input);
    let mut previous = input;
    for &(op_type, domain) in ops {
        let output = graph.create_value(DataType::Float32, static_shape([2]));
        let mut node = Node::new(
            NodeId(0),
            op_type,
            vec![Some(previous), Some(previous)],
            vec![output],
        );
        node.domain = domain.into();
        graph.insert_node(node);
        previous = output;
    }
    if !ops.is_empty() {
        graph.add_output(previous);
    }
    FrozenGraph::build(graph).unwrap()
}

fn native_claims(frozen: &FrozenGraph, ep: &MockEp) -> Vec<onnx_runtime_ep_api::SubgraphClaim> {
    let view = frozen.view();
    onnx_runtime_ep_api::abi::OrtGraphView::new(&view).query_capabilities(ep)
}

#[test]
fn graph_view_capability_claims_all_supported_connected_nodes() {
    let frozen = graph_chain(&[("Add", ""), ("Add", "ai.onnx")]);
    let ep = MockEp::new();
    let claims = native_claims(&frozen, &ep);
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].node_ids.len(), 2);
    assert_eq!(claims[0].input_values, frozen.view().graph().inputs);
    assert_eq!(claims[0].output_values, frozen.view().graph().outputs);
    for &node in &claims[0].node_ids {
        let node = frozen.view().graph().node(node);
        assert!(ep.registry.supports(&node.op_type, &node.domain, 17));
    }
}

#[test]
fn graph_view_capability_splits_regions_at_unsupported_nodes() {
    let frozen = graph_chain(&[("Add", ""), ("Custom", ""), ("Add", "")]);
    let ep = MockEp::new();
    let claims = native_claims(&frozen, &ep);
    assert_eq!(claims.len(), 2);
    assert_eq!(claims[0].node_ids.len(), 1);
    assert_eq!(claims[1].node_ids.len(), 1);

    let graph = frozen.view().graph();
    let nodes: Vec<_> = graph.topological_order().unwrap();
    assert_eq!(claims[0].node_ids, vec![nodes[0]]);
    assert_eq!(claims[0].output_values, graph.node(nodes[0]).outputs);
    assert_eq!(claims[1].node_ids, vec![nodes[2]]);
    assert_eq!(
        claims[1].input_values,
        vec![graph.node(nodes[2]).inputs[0].unwrap()]
    );
    assert!(!ep.registry.supports(
        &graph.node(nodes[1]).op_type,
        &graph.node(nodes[1]).domain,
        17
    ));
}

#[test]
fn graph_view_capability_handles_empty_and_unregistered_graphs() {
    let ep = MockEp::new();
    assert!(native_claims(&graph_chain(&[]), &ep).is_empty());
    assert!(native_claims(&graph_chain(&[("Custom", "")]), &ep).is_empty());
}

#[test]
fn graph_view_capability_keeps_contrib_domains_distinct() {
    let ep = MockEp::new();
    let standard = graph_chain(&[("Add", "")]);
    let canonical_alias = graph_chain(&[("Add", "ai.onnx")]);
    let contrib = graph_chain(&[("Add", "com.microsoft")]);
    assert_eq!(native_claims(&standard, &ep).len(), 1);
    assert_eq!(native_claims(&canonical_alias, &ep).len(), 1);
    assert!(native_claims(&contrib, &ep).is_empty());
}

/// Build Leon's convexity reproducer:
///
/// * `A = Add(input, input) -> a` (supported)
/// * `U = Custom(a, a) -> u`     (unsupported)
/// * `B = Add(a, u) -> b`        (supported, graph output)
///
/// `A` and `B` are only undirected-connected through `a`, but any claim holding
/// both would consume `u` (produced by the excluded `U`) while also producing
/// `a` (consumed by `U`), forming a dependency cycle `{A,B} -> U -> {A,B}`. A
/// convex partitioner must therefore keep `A` and `B` in separate claims.
fn graph_convexity_reproducer() -> FrozenGraph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    graph.opset_imports.insert("ai.onnx".into(), 17);

    let input = graph.create_value(DataType::Float32, static_shape([2]));
    graph.add_input(input);

    let a = graph.create_value(DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(input), Some(input)],
        vec![a],
    ));

    let u = graph.create_value(DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Custom",
        vec![Some(a), Some(a)],
        vec![u],
    ));

    let b = graph.create_value(DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(NodeId(0), "Add", vec![Some(a), Some(u)], vec![b]));
    graph.add_output(b);

    FrozenGraph::build(graph).unwrap()
}

#[test]
fn graph_view_capability_splits_nonconvex_partitions() {
    let frozen = graph_convexity_reproducer();
    let ep = MockEp::new();
    let claims = native_claims(&frozen, &ep);

    // The two supported `Add` nodes must land in separate convex claims; fusing
    // them would create a scheduling cycle with the excluded `Custom` node.
    assert_eq!(
        claims.len(),
        2,
        "expected two convex claims, got {claims:?}"
    );
    for claim in &claims {
        assert_eq!(claim.node_ids.len(), 1);
    }

    // No claim may both consume a value from and produce a value into the same
    // external node: that is exactly the non-convex cycle this test guards.
    let graph = frozen.view().graph();
    for claim in &claims {
        for &input in &claim.input_values {
            let Some(producer) = producer_of(graph, input) else {
                continue;
            };
            for &output in &claim.output_values {
                assert!(
                    !graph.consumers(output).contains(&producer),
                    "claim {claim:?} forms a cycle through external node {producer:?}"
                );
            }
        }
    }
}

fn producer_of(graph: &Graph, value: onnx_runtime_ir::ValueId) -> Option<NodeId> {
    graph
        .topological_order()
        .unwrap()
        .into_iter()
        .find(|&node| graph.node(node).outputs.contains(&value))
}

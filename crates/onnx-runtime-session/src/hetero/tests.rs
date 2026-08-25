//! Tests for heterogeneous multi-provider partitioning and execution (#65).
//!
//! The correctness oracle is byte-identical output between a graph executed
//! heterogeneously (nodes split across two providers on two logical devices)
//! and the same graph executed entirely on a single reference provider. Both
//! providers are CPU-backed here so CI can run the placement / partition /
//! transfer-insertion logic without a GPU; the "accelerator" provider reports a
//! distinct, host-accessible logical device ([`DeviceType::Mlx`]) and a
//! restricted capability set, which is exactly the two-logical-CPU-provider
//! shape the design sanctions for CPU testing. A real CUDA end-to-end check is
//! gated separately (GPU-only) and is not part of CI.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use prost::Message;

use onnx_runtime_ep_api::{
    Cost, DeviceBuffer, DeviceId, DeviceType, EpConfig, EpId, ExecutionProvider, Fence, Kernel,
    KernelMatch, Result as EpResult,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{
    DataType, Graph, ModelFunction, Node, NodeId, Shape, TensorData, TensorLayout, WeightRef,
    static_shape,
};
use onnx_runtime_loader::WeightStore;
use onnx_runtime_loader::proto::onnx;

use super::*;
use crate::executor::Executor;

/// A host-backed "accelerator" provider: it executes on the CPU but advertises a
/// distinct, host-accessible logical device and only supports a fixed op set, so
/// a graph must be split across it and the CPU EP.
struct AcceleratorEp {
    inner: CpuExecutionProvider,
    allowed: Vec<&'static str>,
    claim_only: Vec<&'static str>,
    device: DeviceId,
}

impl AcceleratorEp {
    fn new(allowed: Vec<&'static str>) -> Self {
        let mut inner = CpuExecutionProvider::new();
        inner.initialize(&EpConfig::default()).unwrap();
        Self {
            inner,
            allowed,
            claim_only: Vec::new(),
            // Mlx is host-accessible, so the CPU execution path stays valid while
            // the device id differs from CPU:0 for transfer planning.
            device: DeviceId::new(DeviceType::Mlx, 0),
        }
    }

    fn claim_only(allowed: Vec<&'static str>) -> Self {
        let mut ep = Self::new(Vec::new());
        ep.claim_only = allowed;
        ep
    }
}

impl ExecutionProvider for AcceleratorEp {
    fn consume_route_residency_at_boundary(&self) -> EpResult<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "accel_ep"
    }
    fn device_type(&self) -> DeviceType {
        DeviceType::Mlx
    }
    fn device_id(&self) -> DeviceId {
        self.device
    }
    fn initialize(&mut self, config: &EpConfig) -> EpResult<()> {
        self.inner.initialize(config)
    }
    fn shutdown(&mut self) -> EpResult<()> {
        self.inner.shutdown()
    }
    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> KernelMatch {
        if self.claim_only.contains(&op.op_type.as_str()) {
            return KernelMatch::Supported {
                cost: Cost::ZERO,
                required_input_layouts: None,
                output_layouts: vec![TensorLayout::contiguous(); op.outputs.len()],
            };
        }
        if self.allowed.contains(&op.op_type.as_str()) {
            self.inner
                .supports_op(op, opset, shapes, input_dtypes, layouts)
        } else {
            KernelMatch::unsupported("accelerator does not support this op")
        }
    }
    fn get_kernel(
        &self,
        op: &Node,
        shapes: &[Vec<usize>],
        opset: u64,
    ) -> EpResult<Box<dyn Kernel>> {
        self.inner.get_kernel(op, shapes, opset)
    }
    fn allocate(&self, size: usize, alignment: usize) -> EpResult<DeviceBuffer> {
        self.inner.allocate(size, alignment)
    }
    fn deallocate(&self, buffer: DeviceBuffer) -> EpResult<()> {
        if buffer.is_borrowed() {
            return Ok(());
        }
        self.inner.deallocate(buffer)
    }
    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> EpResult<()> {
        self.inner.copy(src, dst, size)
    }
    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> EpResult<Fence> {
        self.inner.copy_async(src, dst, size)
    }
    fn sync(&self) -> EpResult<()> {
        self.inner.sync()
    }
}

struct AsyncAcceleratorEp {
    inner: AcceleratorEp,
    pending: Mutex<Option<(Vec<u8>, usize)>>,
    waits: Arc<AtomicUsize>,
}

impl AsyncAcceleratorEp {
    fn new(allowed: Vec<&'static str>, waits: Arc<AtomicUsize>) -> Self {
        Self {
            inner: AcceleratorEp::new(allowed),
            pending: Mutex::new(None),
            waits,
        }
    }
}

impl ExecutionProvider for AsyncAcceleratorEp {
    fn consume_route_residency_at_boundary(&self) -> EpResult<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "async_accel_ep"
    }

    fn device_type(&self) -> DeviceType {
        self.inner.device_type()
    }

    fn device_id(&self) -> DeviceId {
        self.inner.device_id()
    }

    fn initialize(&mut self, config: &EpConfig) -> EpResult<()> {
        self.inner.initialize(config)
    }

    fn shutdown(&mut self) -> EpResult<()> {
        self.inner.shutdown()
    }

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> KernelMatch {
        self.inner
            .supports_op(op, opset, shapes, input_dtypes, layouts)
    }

    fn get_kernel(
        &self,
        op: &Node,
        shapes: &[Vec<usize>],
        opset: u64,
    ) -> EpResult<Box<dyn Kernel>> {
        self.inner.get_kernel(op, shapes, opset)
    }

    fn allocate(&self, size: usize, alignment: usize) -> EpResult<DeviceBuffer> {
        let mut buffer = self.inner.allocate(size, alignment)?;
        self.inner.copy_from_host(&vec![0; size], &mut buffer)?;
        Ok(buffer)
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> EpResult<()> {
        self.inner.deallocate(buffer)
    }

    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> EpResult<()> {
        self.inner.copy(src, dst, size)
    }

    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> EpResult<Fence> {
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), size) }.to_vec();
        *self.pending.lock().expect("pending copy lock") = Some((bytes, dst.as_mut_ptr() as usize));
        Ok(Fence::new(1))
    }

    fn wait_fence(&self, fence: &Fence) -> EpResult<()> {
        if fence.is_signalled() {
            return Ok(());
        }
        let (bytes, destination) = self
            .pending
            .lock()
            .expect("pending copy lock")
            .take()
            .expect("wait must consume a pending async copy");
        // SAFETY: the destination allocation is owned by the live binding that
        // requested the copy and remains allocated until this wait returns.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination as *mut u8, bytes.len());
        }
        self.waits.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> EpResult<()> {
        self.inner.copy_to_host(src, dst)
    }

    fn sync(&self) -> EpResult<()> {
        self.inner.sync()
    }
}

fn cpu_slot(ep: u32) -> ProviderPlacement {
    let mut inner = CpuExecutionProvider::new();
    inner.initialize(&EpConfig::default()).unwrap();
    ProviderPlacement {
        ep: EpId(ep),
        provider: Arc::new(inner),
    }
}

fn accel_slot(ep: u32, allowed: Vec<&'static str>) -> ProviderPlacement {
    ProviderPlacement {
        ep: EpId(ep),
        provider: Arc::new(AcceleratorEp::new(allowed)),
    }
}

fn async_accel_slot(
    ep: u32,
    allowed: Vec<&'static str>,
    waits: Arc<AtomicUsize>,
) -> ProviderPlacement {
    ProviderPlacement {
        ep: EpId(ep),
        provider: Arc::new(AsyncAcceleratorEp::new(allowed, waits)),
    }
}

fn claim_only_slot(ep: u32, allowed: Vec<&'static str>) -> ProviderPlacement {
    ProviderPlacement {
        ep: EpId(ep),
        provider: Arc::new(AcceleratorEp::claim_only(allowed)),
    }
}

fn weights() -> Arc<WeightStore> {
    Arc::new(WeightStore::new())
}

/// Build a unary op chain `x -> op0 -> op1 -> ... -> out`.
fn build_chain(ops: &[&str]) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let mut current = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(current);
    for (i, op) in ops.iter().enumerate() {
        let out = graph.create_named_value(format!("t{i}"), DataType::Float32, static_shape([4]));
        graph.insert_node(Node::new(NodeId(0), *op, vec![Some(current)], vec![out]));
        current = out;
    }
    graph.add_output(current);
    graph
}

/// Reference execution: the whole graph on a single CPU EP.
fn reference(graph: &Graph, inputs: &[(&str, &Tensor)]) -> Vec<Tensor> {
    let ep = cpu_slot(0).provider;
    let mut exec = Executor::build(graph.clone(), weights(), ep).unwrap();
    exec.run(inputs).unwrap()
}

fn assert_byte_identical(a: &[Tensor], b: &[Tensor]) {
    assert_eq!(a.len(), b.len(), "output arity differs");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.as_bytes(),
            y.as_bytes(),
            "output {i} is not byte-identical (f32 bits differ)"
        );
    }
}

fn inline_f32(dims: &[usize], data: &[f32]) -> WeightRef {
    WeightRef::Inline(TensorData::from_raw(
        DataType::Float32,
        dims.to_vec(),
        data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ))
}

fn add_relu_body(name: &str, inner_call: Option<&str>) -> ModelFunction {
    let mut body = Graph::new();
    body.opset_imports.insert(String::new(), 17);
    let x = body.create_named_value("X", DataType::Float32, static_shape([4]));
    let b = body.create_named_value("B", DataType::Float32, static_shape([4]));
    body.add_input(x);
    body.add_input(b);
    let y = body.create_named_value("Y", DataType::Float32, static_shape([4]));
    match inner_call {
        Some(inner) => {
            let mut call = Node::new(NodeId(0), inner, vec![Some(x), Some(b)], vec![y]);
            call.domain = "custom.domain".to_string();
            body.insert_node(call);
        }
        None => {
            let sum = body.create_named_value("sum", DataType::Float32, static_shape([4]));
            body.insert_node(Node::new(
                NodeId(0),
                "Add",
                vec![Some(x), Some(b)],
                vec![sum],
            ));
            body.insert_node(Node::new(NodeId(0), "Relu", vec![Some(sum)], vec![y]));
        }
    }
    body.add_output(y);
    ModelFunction {
        domain: "custom.domain".to_string(),
        name: name.to_string(),
        inputs: vec!["X".to_string(), "B".to_string()],
        outputs: vec!["Y".to_string()],
        attributes: Vec::new(),
        has_attribute_refs: false,
        body,
    }
}

fn kept_fused_add_relu_graph(nested: bool) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    graph.opset_imports.insert("custom.domain".to_string(), 1);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(x);
    let b = graph.create_named_value("b", DataType::Float32, static_shape([4]));
    graph.set_initializer(b, inline_f32(&[4], &[1.0, -3.0, 10.0, -20.0]));
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    let op = if nested {
        "OuterFusedAddRelu"
    } else {
        "FusedAddRelu"
    };
    let mut fused = Node::new(NodeId(0), op, vec![Some(x), Some(b)], vec![y]);
    fused.domain = "custom.domain".to_string();
    graph.insert_node(fused);
    graph.add_output(y);
    if nested {
        graph.model_functions.insert(
            ("custom.domain".to_string(), "OuterFusedAddRelu".to_string()),
            add_relu_body("OuterFusedAddRelu", Some("InnerFusedAddRelu")),
        );
        graph.model_functions.insert(
            ("custom.domain".to_string(), "InnerFusedAddRelu".to_string()),
            add_relu_body("InnerFusedAddRelu", None),
        );
    } else {
        graph.model_functions.insert(
            ("custom.domain".to_string(), "FusedAddRelu".to_string()),
            add_relu_body("FusedAddRelu", None),
        );
    }
    graph
}

fn primitive_add_relu_graph() -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(x);
    let b = graph.create_named_value("b", DataType::Float32, static_shape([4]));
    graph.set_initializer(b, inline_f32(&[4], &[1.0, -3.0, 10.0, -20.0]));
    let sum = graph.create_named_value("sum", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(x), Some(b)],
        vec![sum],
    ));
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(sum)], vec![y]));
    graph.add_output(y);
    graph
}

fn op_count(graph: &Graph, op: &str) -> usize {
    graph
        .nodes
        .iter()
        .filter(|(_, node)| node.op_type == op)
        .count()
}

fn onnx_tensor_type(dims: &[i64]) -> onnx::TypeProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value as DV};
    onnx::TypeProto {
        value: Some(onnx::type_proto::Value::TensorType(
            onnx::type_proto::Tensor {
                elem_type: 1,
                shape: Some(onnx::TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&n| Dimension {
                            value: Some(DV::DimValue(n)),
                            ..Default::default()
                        })
                        .collect(),
                }),
            },
        )),
        ..Default::default()
    }
}

fn onnx_value_info(name: &str, dims: &[i64]) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_tensor_type(dims)),
        ..Default::default()
    }
}

fn onnx_f32_initializer(name: &str, data: &[f32]) -> onnx::TensorProto {
    onnx::TensorProto {
        name: name.to_string(),
        data_type: 1,
        dims: vec![data.len() as i64],
        raw_data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ..Default::default()
    }
}

fn onnx_node(op: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: op.to_string(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

fn onnx_call(op: &str, domain: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
    let mut node = onnx_node(op, inputs, outputs);
    node.domain = domain.to_string();
    node
}

fn onnx_float_attr(name: &str, value: f32) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Float as i32,
        f: value,
        ..Default::default()
    }
}

fn onnx_ref_attr(name: &str, ref_attr_name: &str) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        ref_attr_name: ref_attr_name.to_string(),
        ..Default::default()
    }
}

fn onnx_call_with_attrs(
    op: &str,
    domain: &str,
    inputs: &[&str],
    outputs: &[&str],
    attributes: Vec<onnx::AttributeProto>,
) -> onnx::NodeProto {
    let mut node = onnx_call(op, domain, inputs, outputs);
    node.attribute = attributes;
    node
}

fn attributed_leaky_relu_model_bytes() -> Vec<u8> {
    let mut body = onnx_node("LeakyRelu", &["X"], &["Y"]);
    body.attribute = vec![onnx_ref_attr("alpha", "alpha")];
    let func = onnx::FunctionProto {
        name: "ParamLeakyRelu".to_string(),
        domain: "custom.domain".to_string(),
        input: vec!["X".to_string()],
        output: vec!["Y".to_string()],
        attribute: vec!["alpha".to_string()],
        node: vec![body],
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        ..Default::default()
    };
    let graph = onnx::GraphProto {
        input: vec![onnx_value_info("x", &[4])],
        output: vec![onnx_value_info("y", &[4])],
        node: vec![onnx_call_with_attrs(
            "ParamLeakyRelu",
            "custom.domain",
            &["x"],
            &["y"],
            vec![onnx_float_attr("alpha", 0.25)],
        )],
        ..Default::default()
    };
    onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![
            onnx::OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            },
            onnx::OperatorSetIdProto {
                domain: "custom.domain".to_string(),
                version: 1,
            },
        ],
        graph: Some(graph),
        functions: vec![func],
        ..Default::default()
    }
    .encode_to_vec()
}

fn fused_add_relu_model_bytes() -> Vec<u8> {
    let func = onnx::FunctionProto {
        name: "FusedAddRelu".to_string(),
        domain: "custom.domain".to_string(),
        input: vec!["X".to_string(), "B".to_string()],
        output: vec!["Y".to_string()],
        node: vec![
            onnx_node("Add", &["X", "B"], &["sum"]),
            onnx_node("Relu", &["sum"], &["Y"]),
        ],
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        ..Default::default()
    };
    let graph = onnx::GraphProto {
        input: vec![onnx_value_info("x", &[4])],
        output: vec![onnx_value_info("y", &[4])],
        initializer: vec![onnx_f32_initializer("b", &[1.0, -3.0, 10.0, -20.0])],
        node: vec![onnx_call(
            "FusedAddRelu",
            "custom.domain",
            &["x", "b"],
            &["y"],
        )],
        ..Default::default()
    };
    onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![
            onnx::OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            },
            onnx::OperatorSetIdProto {
                domain: "custom.domain".to_string(),
                version: 1,
            },
        ],
        graph: Some(graph),
        functions: vec![func],
        ..Default::default()
    }
    .encode_to_vec()
}

#[test]
fn pure_cpu_single_partition() {
    // Accelerator supports nothing; every node lands on the CPU EP as one
    // partition, with no cross-device transfers.
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec![]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();

    assert_eq!(plan.partitions.len(), 1);
    assert_eq!(plan.partitions[0].ep, EpId(1));
    assert!(plan.transfers.is_empty());

    let x = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    let out = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    assert_byte_identical(&out, &reference(&graph, &[("x", &x)]));
}

#[test]
fn kept_function_declined_by_assignment_ep_is_inlined_and_runs() {
    let bytes = fused_add_relu_model_bytes();
    let keep = |node: &Node, _opset: u64, _dtypes: &[DataType]| node.op_type == "FusedAddRelu";
    let (graph, store) =
        onnx_runtime_loader::load_model_bytes_with_weights_filtered(&bytes, ".", &keep)
            .expect("filtered load keeps synthetic fused function");
    assert_eq!(
        op_count(&graph, "FusedAddRelu"),
        1,
        "load-time keep predicate should leave the fused function call in the graph"
    );
    let (reference_graph, reference_store) =
        onnx_runtime_loader::load_model_bytes_with_weights(&bytes, ".")
            .expect("unfiltered load inlines function body");
    let providers = vec![accel_slot(0, vec!["Relu"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();
    let legalized = plan
        .legalized_graph
        .as_ref()
        .expect("fused function should be legalized");

    assert_eq!(op_count(legalized, "FusedAddRelu"), 0);
    assert_eq!(op_count(legalized, "Add"), 1);
    assert_eq!(op_count(legalized, "Relu"), 1);
    assert!(
        plan.node_placement
            .iter()
            .any(|(&node, &ep)| legalized.node(node).op_type == "Relu" && ep == EpId(0)),
        "primitive body ops should be repartitioned after legalization"
    );

    let x = Tensor::from_f32(&[4], &[-2.0, 4.0, -5.0, 100.0]).unwrap();
    let hetero = execute(&plan, &graph, &store, &providers, &[("x", &x)]).unwrap();
    let mut reference_exec =
        Executor::build(reference_graph, reference_store, cpu_slot(0).provider).unwrap();
    let reference = reference_exec.run(&[("x", &x)]).unwrap();
    assert_byte_identical(&hetero, &reference);
}

#[test]
fn attribute_parameterized_kept_function_legalization_fails_closed() {
    let bytes = attributed_leaky_relu_model_bytes();
    let keep = |node: &Node, _opset: u64, _dtypes: &[DataType]| node.op_type == "ParamLeakyRelu";
    let (graph, _store) =
        onnx_runtime_loader::load_model_bytes_with_weights_filtered(&bytes, ".", &keep)
            .expect("filtered load keeps synthetic attributed function");
    assert_eq!(op_count(&graph, "ParamLeakyRelu"), 1);
    let function = graph
        .model_functions
        .get(&("custom.domain".to_string(), "ParamLeakyRelu".to_string()))
        .expect("model-local function catalog should retain metadata");
    assert_eq!(function.attributes, vec!["alpha".to_string()]);
    assert!(
        function.has_attribute_refs,
        "ref_attr_name must be captured before IR conversion drops it"
    );

    let providers = vec![accel_slot(0, vec!["Relu"]), cpu_slot(1)];
    let err = plan(&graph, &providers).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("ParamLeakyRelu"), "{msg}");
    assert!(msg.contains("attribute-parameterized"), "{msg}");
    assert!(msg.contains("Phase 2"), "{msg}");
}

#[test]
fn nested_kept_functions_legalize_to_fixpoint() {
    let graph = kept_fused_add_relu_graph(true);
    let providers = vec![accel_slot(0, vec!["Relu"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();
    let legalized = plan
        .legalized_graph
        .as_ref()
        .expect("nested fused functions should be legalized");

    assert_eq!(op_count(legalized, "OuterFusedAddRelu"), 0);
    assert_eq!(op_count(legalized, "InnerFusedAddRelu"), 0);
    assert_eq!(op_count(legalized, "Add"), 1);
    assert_eq!(op_count(legalized, "Relu"), 1);

    let x = Tensor::from_f32(&[4], &[-2.0, 4.0, -5.0, 100.0]).unwrap();
    let hetero = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    let reference = reference(&primitive_add_relu_graph(), &[("x", &x)]);
    assert_byte_identical(&hetero, &reference);
}

#[test]
fn ambiguous_model_function_identity_fails_closed() {
    let mut graph = kept_fused_add_relu_graph(false);
    graph.model_functions.clear();
    graph
        .ambiguous_model_functions
        .insert(("custom.domain".to_string(), "FusedAddRelu".to_string()));
    let providers = vec![accel_slot(0, vec!["Relu"]), cpu_slot(1)];
    let err = plan(&graph, &providers).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("multiple overloads"), "{msg}");
    assert!(msg.contains("FusedAddRelu"), "{msg}");
}

#[test]
fn claimed_function_stays_kept_when_assigned_provider_supports_it() {
    let graph = kept_fused_add_relu_graph(false);
    let providers = vec![claim_only_slot(0, vec!["FusedAddRelu"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();

    assert!(
        plan.legalized_graph.is_none(),
        "single-provider-supported fused function should not be decomposed"
    );
    assert_eq!(plan.partitions.len(), 1);
    let fused_node = plan.partitions[0].nodes[0];
    assert_eq!(graph.node(fused_node).op_type, "FusedAddRelu");
    assert_eq!(plan.partitions[0].ep, EpId(0));
}

#[test]
fn fully_accelerator_single_partition() {
    // Accelerator supports every op; one partition on the Mlx device. Only the
    // graph input must be staged host->Mlx.
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Abs", "Neg"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();

    assert_eq!(plan.partitions.len(), 1);
    assert_eq!(plan.partitions[0].ep, EpId(0));
    assert_eq!(plan.partitions[0].device, DeviceId::new(DeviceType::Mlx, 0));
    // Graph-input staging belongs to the partition executor, not the
    // cross-provider transfer plan.
    assert!(plan.transfers.is_empty());

    let x = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    let out = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    assert_byte_identical(&out, &reference(&graph, &[("x", &x)]));
}

#[test]
fn mixed_multiple_partition_boundaries() {
    // Relu(accel) -> Abs(cpu) -> Sqrt(accel) -> Neg(cpu): four partitions and
    // three device crossings, plus the host->accel input transfer.
    let graph = build_chain(&["Relu", "Abs", "Sqrt", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Sqrt"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();

    assert_eq!(plan.partitions.len(), 4, "each op is its own partition");
    let eps: Vec<_> = plan.partitions.iter().map(|p| p.ep).collect();
    assert_eq!(eps, vec![EpId(0), EpId(1), EpId(0), EpId(1)]);

    let x = Tensor::from_f32(&[4], &[1.0, 4.0, 9.0, 16.0]).unwrap();
    let hetero = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    let reference = reference(&graph, &[("x", &x)]);
    assert_byte_identical(&hetero, &reference);

    // --- Mutation probe 1: misplace execution order (drop the topological
    // guarantee). Running a consumer partition before its producer leaves a
    // boundary input unmaterialized, so execution must fail. ---------------
    let mut misordered = plan.clone();
    misordered.partitions.reverse();
    let broken = execute(&misordered, &graph, &weights(), &providers, &[("x", &x)]);
    assert!(
        broken.is_err(),
        "reversing partition order must break execution (topological order is load-bearing)"
    );

    // --- Mutation probe 2: drop a boundary transfer by clearing a producer
    // partition's outputs, so the downstream partition never receives it. ---
    let mut dropped = plan.clone();
    dropped.partitions[0].outputs.clear();
    let broken2 = execute(&dropped, &graph, &weights(), &providers, &[("x", &x)]);
    assert!(
        broken2.is_err(),
        "dropping a boundary transfer must break execution"
    );

    // --- Restore: the unmutated plan is byte-identical again. --------------
    let restored = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    assert_byte_identical(&restored, &reference);

    // --- Oracle self-check: a single flipped bit is detected. --------------
    let mut corrupted = reference.clone();
    let mut bytes = corrupted[0].as_bytes().to_vec();
    bytes[0] ^= 0x01;
    corrupted[0] = Tensor::from_raw(DataType::Float32, vec![4], &bytes).unwrap();
    assert_ne!(
        hetero[0].as_bytes(),
        corrupted[0].as_bytes(),
        "oracle must detect a one-bit difference"
    );
}

#[test]
fn fan_out_boundary_transfer_is_deduplicated() {
    // Relu(accel) produces `y`; `y` fans out to Abs(cpu) and Neg(cpu), which are
    // two separate CPU partitions on the same device. The Mlx->CPU transfer of
    // `y` must be materialized exactly once, not once per consumer.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(x);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]));
    let a = graph.create_named_value("a", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Abs", vec![Some(y)], vec![a]));
    let b = graph.create_named_value("b", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Neg", vec![Some(y)], vec![b]));
    graph.add_output(a);
    graph.add_output(b);

    let providers = vec![accel_slot(0, vec!["Relu"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();

    // Three partitions: {Relu}@Mlx, {Abs}@cpu, {Neg}@cpu.
    assert_eq!(plan.partitions.len(), 3);
    let cpu_dev = DeviceId::cpu();
    let mlx = DeviceId::new(DeviceType::Mlx, 0);
    // Exactly one transfer of `y` onto the CPU device (fan-out dedup), plus one
    // host->Mlx transfer of `x`.
    let y_to_cpu: Vec<_> = plan
        .transfers
        .iter()
        .filter(|t| t.value == y && t.to == cpu_dev)
        .collect();
    assert_eq!(
        y_to_cpu.len(),
        1,
        "fan-out value must be transferred to CPU exactly once"
    );
    assert_eq!(y_to_cpu[0].from, mlx);
    assert_eq!(plan.transfers.iter().filter(|t| t.value == x).count(), 0);

    let x_val = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    let out = execute(&plan, &graph, &weights(), &providers, &[("x", &x_val)]).unwrap();
    assert_byte_identical(&out, &reference(&graph, &[("x", &x_val)]));
}

#[test]
fn topological_order_preserved() {
    // Every producing partition precedes its consumers in the plan order.
    let graph = build_chain(&["Relu", "Abs", "Sqrt", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Sqrt"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();

    let mut produced_by: std::collections::HashMap<ValueId, usize> = Default::default();
    for (i, part) in plan.partitions.iter().enumerate() {
        for &node in &part.nodes {
            for &out in &graph.node(node).outputs {
                produced_by.insert(out, i);
            }
        }
    }
    for (j, part) in plan.partitions.iter().enumerate() {
        for &input in &part.inputs {
            if let Some(&i) = produced_by.get(&input) {
                assert!(i < j, "partition {j} consumes a value produced by {i}");
            }
        }
    }
}

#[test]
fn empty_graph_has_no_partitions() {
    // A graph with a single pass-through value (input == output) and no nodes:
    // no partitions, no transfers; execution returns the input unchanged.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let v = graph.create_named_value("x", DataType::Float32, static_shape([3]));
    graph.add_input(v);
    graph.add_output(v);

    let providers = vec![accel_slot(0, vec![]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();
    assert!(plan.partitions.is_empty());
    assert!(plan.transfers.is_empty());

    let x = Tensor::from_f32(&[3], &[7.0, 8.0, 9.0]).unwrap();
    let out = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].as_bytes(), x.as_bytes());
}

#[test]
fn node_unsupported_by_all_providers_fails_before_execution() {
    // No provider supports Sqrt (both slots only advertise Relu), so planning
    // must fail with an actionable error before any execution.
    let graph = build_chain(&["Relu", "Sqrt"]);
    let providers = vec![accel_slot(0, vec!["Relu"]), accel_slot(1, vec!["Relu"])];
    let err = plan(&graph, &providers).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Sqrt"),
        "error should name the unsupported op: {msg}"
    );
    assert_eq!(
        msg.matches("accelerator does not support this op").count(),
        2,
        "the error must preserve both provider decline reasons: {msg}"
    );
    assert!(msg.contains("ai.onnx::Sqrt"), "{msg}");
    assert!(msg.contains("opset 17"), "{msg}");
}

#[test]
fn placement_is_deterministic_across_runs() {
    let graph = build_chain(&["Relu", "Abs", "Sqrt", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Sqrt"]), cpu_slot(1)];
    let a = plan(&graph, &providers).unwrap();
    let b = plan(&graph, &providers).unwrap();

    assert_eq!(a.node_placement, b.node_placement);
    assert_eq!(a.partitions, b.partitions);
    assert_eq!(a.transfers, b.transfers);
}

// ---------------------------------------------------------------------------
// Thread-3 Phase 3: default-path classification + fail-closed execution guard.
// ---------------------------------------------------------------------------

#[test]
fn classify_homogeneous_graph_is_single_provider() {
    // The accelerator claims every op, so the whole graph collapses onto ep#0
    // with no cross-device transfers: the caller keeps its single-EP path.
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Abs", "Neg"]), cpu_slot(1)];
    match classify_placement(&graph, &providers).unwrap() {
        PlacementDecision::SingleProvider(ep) => assert_eq!(ep, EpId(0)),
        other => panic!("expected SingleProvider, got {other:?}"),
    }
}

#[test]
fn classify_whole_session_cpu_fallback_is_single_provider() {
    // The accelerator claims nothing, so every node lands on the CPU EP: this is
    // the existing whole-session fallback shape, and it must NOT be flagged as
    // heterogeneous (so the guard lets the byte-identical fallback proceed).
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec![]), cpu_slot(1)];
    match classify_placement(&graph, &providers).unwrap() {
        PlacementDecision::SingleProvider(ep) => assert_eq!(ep, EpId(1)),
        other => panic!("expected SingleProvider, got {other:?}"),
    }
}

#[test]
fn classify_mixed_graph_is_heterogeneous() {
    // Relu/Neg on the accelerator, Abs on CPU: a genuine per-op split.
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Neg"]), cpu_slot(1)];
    match classify_placement(&graph, &providers).unwrap() {
        PlacementDecision::Heterogeneous(plan) => {
            assert!(!plan.transfers.is_empty(), "a split needs transfers");
            let eps: std::collections::HashSet<_> = plan.node_placement.values().copied().collect();
            assert_eq!(eps.len(), 2, "nodes must span both providers");
        }
        other => panic!("expected Heterogeneous, got {other:?}"),
    }
}

#[test]
fn classified_heterogeneous_plan_still_executes_byte_identically() {
    // De-latenting proof: a classified mixed plan, executed per-op via the
    // standalone host-staged executor, is byte-identical to the single-EP
    // reference.
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Neg"]), cpu_slot(1)];
    let plan = match classify_placement(&graph, &providers).unwrap() {
        PlacementDecision::Heterogeneous(plan) => *plan,
        other => panic!("expected Heterogeneous, got {other:?}"),
    };
    let x = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    let hetero = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    assert_byte_identical(&hetero, &reference(&graph, &[("x", &x)]));
}

#[test]
fn accel_cpu_accel_executes_with_two_planned_transfers_and_owned_lifetimes() {
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Neg"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();
    let accel_nodes = plan
        .node_placement
        .values()
        .filter(|&&ep| ep == EpId(0))
        .count();
    let cpu_nodes = plan
        .node_placement
        .values()
        .filter(|&&ep| ep == EpId(1))
        .count();
    assert_eq!((accel_nodes, cpu_nodes), (2, 1));
    assert_eq!(
        plan.transfers.len(),
        2,
        "only Accel->CPU and CPU->Accel edges are transfers"
    );

    let first_boundary = value_by_name(&graph, "t0").unwrap();
    let second_boundary = value_by_name(&graph, "t1").unwrap();
    let mut executor = HeterogeneousExecutor::build(plan, &graph, &weights(), &providers).unwrap();
    let x = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    let output = executor.run(&[("x", &x)]).unwrap();
    assert_eq!(output[0].to_vec_f32(), vec![-0.0, -2.0, -0.0, -4.0]);
    assert_eq!(executor.last_transfer_count(), 2);
    for (value, source, destination) in [
        (first_boundary, EpId(0), EpId(1)),
        (second_boundary, EpId(1), EpId(0)),
    ] {
        assert_eq!(executor.last_release_count(value, source), 1);
        assert_eq!(executor.last_release_count(value, destination), 1);
    }
    assert!(
        executor
            .placement_report()
            .contains("2 node(s) on accel_ep")
    );
    assert!(executor.placement_report().contains("1 node(s) on cpu_ep"));
    assert!(
        executor
            .placement_report()
            .contains("2 cross-provider transfer(s)")
    );
}

#[test]
fn default_executor_opt_in_uses_mixed_plan_instead_of_whole_cpu_fallback() {
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let accelerator: Arc<dyn ExecutionProvider> = Arc::new(AcceleratorEp::new(vec!["Relu", "Neg"]));
    let mut executor =
        Executor::build_with_heterogeneous_enabled(graph, weights(), accelerator).unwrap();
    assert!(
        executor.execution_provider_fallback_report().is_none(),
        "mixed execution must not retain a whole-session CPU fallback report"
    );
    let report = executor
        .heterogeneous_placement_report()
        .expect("the opt-in executor must expose placement");
    assert!(report.contains("2 node(s) on accel_ep"), "{report}");
    assert!(report.contains("1 node(s) on cpu_ep"), "{report}");
    assert!(report.contains("2 cross-provider transfer(s)"), "{report}");

    let x = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    assert_eq!(
        executor.run(&[("x", &x)]).unwrap()[0].to_vec_f32(),
        vec![-0.0, -2.0, -0.0, -4.0]
    );

    let binding_error = executor
        .run_with_device_bindings(&[("x", &x)], &mut [])
        .unwrap_err()
        .to_string();
    assert!(binding_error.contains("persistent device bindings"));
    let capture_error = match executor.try_capture_with_device_bindings(&[("x", &x)], &mut []) {
        Err(error) => error.to_string(),
        Ok(_) => panic!("mixed-provider capture must fail closed"),
    };
    assert!(capture_error.contains("device-graph capture"));
}

#[test]
fn async_destination_waits_for_transfer_before_consumer_runs() {
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let waits = Arc::new(AtomicUsize::new(0));
    let providers = vec![
        async_accel_slot(0, vec!["Relu", "Neg"], Arc::clone(&waits)),
        cpu_slot(1),
    ];
    let plan = plan(&graph, &providers).unwrap();
    let mut executor = HeterogeneousExecutor::build(plan, &graph, &weights(), &providers).unwrap();
    let x = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    let output = executor.run(&[("x", &x)]).unwrap();
    assert_eq!(output[0].to_vec_f32(), vec![-0.0, -2.0, -0.0, -4.0]);
    assert_eq!(
        waits.load(Ordering::SeqCst),
        1,
        "the CPU->accelerator copy fence must be awaited exactly once"
    );
}

#[test]
fn control_flow_and_sequence_fail_during_planning() {
    let providers = vec![accel_slot(0, vec!["Relu"]), cpu_slot(1)];

    let control_flow = build_chain(&["If"]);
    let error = plan(&control_flow, &providers).unwrap_err().to_string();
    assert!(error.contains("control flow"), "{error}");
    assert!(error.contains("If"), "{error}");

    let sequence = build_chain(&["SequenceEmpty"]);
    let error = plan(&sequence, &providers).unwrap_err().to_string();
    assert!(error.contains("sequence"), "{error}");
    assert!(error.contains("SequenceEmpty"), "{error}");
}

#[test]
fn view_producing_kernel_is_rejected_before_partition_execution() {
    let graph = build_chain(&["Transpose"]);
    let providers = vec![cpu_slot(0)];
    let plan = plan(&graph, &providers).unwrap();
    let error = HeterogeneousExecutor::build(plan, &graph, &weights(), &providers)
        .err()
        .expect("view-producing boundary must fail closed")
        .to_string();
    assert!(error.contains("Transpose"), "{error}");
    assert!(error.contains("aliased/view output"), "{error}");
}

#[test]
fn placement_summary_names_the_fallback_op() {
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Neg"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();
    let summary = placement_summary(&plan, &graph, &providers);
    assert!(
        summary.contains("Abs"),
        "summary should name Abs: {summary}"
    );
    assert!(
        summary.contains("transfer"),
        "summary should report transfers: {summary}"
    );
}

#[test]
fn guard_disabled_is_a_noop() {
    // With the opt-in flag off, even a genuinely mixed graph is a no-op: the
    // caller's whole-session fallback proceeds unchanged (byte-identical).
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Neg"]), cpu_slot(1)];
    guard_heterogeneous_fallback(&graph, &providers, false).unwrap();
}

#[test]
fn guard_enabled_homogeneous_is_ok() {
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec![]), cpu_slot(1)];
    guard_heterogeneous_fallback(&graph, &providers, true).unwrap();
}

#[test]
fn guard_enabled_mixed_fails_closed() {
    // With the flag on and a genuine split, fail closed with an actionable
    // error naming the fallback op instead of silently dropping the session.
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Neg"]), cpu_slot(1)];
    let err = guard_heterogeneous_fallback(&graph, &providers, true).unwrap_err();
    assert!(matches!(
        err,
        SessionError::HeterogeneousExecutionUnsupported { .. }
    ));
    let msg = err.to_string();
    assert!(
        msg.contains("Abs"),
        "error should name the fallback op: {msg}"
    );
    assert!(
        msg.contains("#603"),
        "error should point at the deferred issue: {msg}"
    );
}

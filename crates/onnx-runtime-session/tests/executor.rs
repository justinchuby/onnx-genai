//! Integration tests for the sequential CPU executor (Track D).
//!
//! Each test hand-builds a small [`Graph`] via the IR API, runs it through the
//! public [`InferenceSession`] surface, and asserts the output matches a
//! reference computed here in the test. Nothing below names a model or bakes in
//! a fixed shape path — the executor is exercised as a generic Graph runner.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use onnx_runtime_ep_api::{
    DeviceBuffer, EpConfig, ExecutionProvider, Fence, Kernel, KernelMatch, Result as EpResult,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, DeviceType, Dim, Graph, Node, NodeId, Shape, TensorData,
    TensorLayout, ValueId, WeightRef, static_shape,
};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{InferenceSession, OpsetVersion, SessionError, Tensor, WarmupShape};
use onnx_runtime_shape_inference::{InferenceRegistry, MAX_SHAPE_DATA_ELEMS, MergePolicy};

// This synthetic name must remain unregistered so unsupported-op error tests cannot go stale.
const UNSUPPORTED_OP_SENTINEL: &str = "NxrtNeverRegisteredSentinelOp";

// --- graph construction helpers --------------------------------------------

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Add an inline f32 initializer, returning its value id.
fn f32_init(g: &mut Graph, name: &str, dims: &[usize], data: &[f32]) -> ValueId {
    let vid = g.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()));
    g.set_initializer(
        vid,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            dims.to_vec(),
            f32_bytes(data),
        )),
    );
    vid
}

fn i64_init(g: &mut Graph, name: &str, dims: &[usize], data: &[i64]) -> ValueId {
    let vid = g.create_named_value(name, DataType::Int64, static_shape(dims.iter().copied()));
    g.set_initializer(
        vid,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            dims.to_vec(),
            data.iter().flat_map(|value| value.to_le_bytes()).collect(),
        )),
    );
    vid
}

/// Add a named graph input, returning its value id.
fn input(g: &mut Graph, name: &str, dtype: DataType, dims: &[usize]) -> ValueId {
    let vid = g.create_named_value(name, dtype, static_shape(dims.iter().copied()));
    g.add_input(vid);
    vid
}

/// Insert an op node producing a single output value of the given shape/dtype.
fn op(
    g: &mut Graph,
    op_type: &str,
    inputs: &[ValueId],
    out_dtype: DataType,
    out_dims: &[usize],
    attrs: &[(&str, Attribute)],
) -> ValueId {
    g.opset_imports.entry(String::new()).or_insert(17);
    let out = g.create_value(out_dtype, static_shape(out_dims.iter().copied()));
    let mut node = Node::new(
        NodeId(0),
        op_type,
        inputs.iter().map(|&v| Some(v)).collect(),
        vec![out],
    );
    for (k, v) in attrs {
        node.attributes.insert((*k).to_string(), v.clone());
    }
    g.insert_node(node);
    out
}

/// Add a named graph input with an explicit (possibly symbolic) shape.
fn input_shaped(g: &mut Graph, name: &str, dtype: DataType, shape: Shape) -> ValueId {
    let vid = g.create_named_value(name, dtype, shape);
    g.add_input(vid);
    vid
}

fn i32_tensor(shape: &[usize], data: &[i32]) -> Tensor {
    let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
    Tensor::from_raw(DataType::Int32, shape.to_vec(), &bytes).unwrap()
}

fn i64_tensor(shape: &[usize], data: &[i64]) -> Tensor {
    let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
    Tensor::from_raw(DataType::Int64, shape.to_vec(), &bytes).unwrap()
}

struct HostDownloadCountingEp {
    cpu: CpuExecutionProvider,
    host_downloads: Arc<AtomicUsize>,
}

impl HostDownloadCountingEp {
    fn new(host_downloads: Arc<AtomicUsize>) -> Self {
        let mut cpu = CpuExecutionProvider::new();
        cpu.initialize(&EpConfig::default()).unwrap();
        Self {
            cpu,
            host_downloads,
        }
    }
}

impl ExecutionProvider for HostDownloadCountingEp {
    fn name(&self) -> &str {
        "host_download_counting_ep"
    }

    fn device_type(&self) -> DeviceType {
        self.cpu.device_type()
    }

    fn device_id(&self) -> DeviceId {
        self.cpu.device_id()
    }

    fn initialize(&mut self, config: &EpConfig) -> EpResult<()> {
        self.cpu.initialize(config)
    }

    fn shutdown(&mut self) -> EpResult<()> {
        self.cpu.shutdown()
    }

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> KernelMatch {
        self.cpu
            .supports_op(op, opset, shapes, input_dtypes, layouts)
    }

    fn get_kernel(
        &self,
        op: &Node,
        shapes: &[Vec<usize>],
        opset: u64,
    ) -> EpResult<Box<dyn Kernel>> {
        self.cpu.get_kernel(op, shapes, opset)
    }

    fn allocate(&self, size: usize, alignment: usize) -> EpResult<DeviceBuffer> {
        self.cpu.allocate(size, alignment)
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> EpResult<()> {
        self.cpu.deallocate(buffer)
    }

    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> EpResult<()> {
        self.cpu.copy(src, dst, size)
    }

    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> EpResult<Fence> {
        self.cpu.copy_async(src, dst, size)
    }

    fn copy_from_host(&self, src: &[u8], dst: &mut DeviceBuffer) -> EpResult<()> {
        self.cpu.copy_from_host(src, dst)
    }

    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> EpResult<()> {
        self.host_downloads.fetch_add(1, Ordering::Relaxed);
        self.cpu.copy_to_host(src, dst)
    }

    fn sync(&self) -> EpResult<()> {
        self.cpu.sync()
    }
}

fn unresolved_unsqueeze_model(axes_dtype: DataType, axes_shape: &[usize]) -> Vec<u8> {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let data = input(&mut g, "data", DataType::Float32, &[2]);
    let axes = input(&mut g, "axes", axes_dtype, axes_shape);
    let dynamic = g.intern_symbol("dynamic_unsqueeze_extent");
    let output = g.create_named_value("output", DataType::Float32, vec![Dim::Symbolic(dynamic)]);
    g.mark_value_shape_unknown(output);
    g.insert_node(Node::new(
        NodeId(0),
        "Unsqueeze",
        vec![Some(data), Some(axes)],
        vec![output],
    ));
    g.add_output(output);
    encode_model(&Model::new(&g)).expect("encode unresolved Unsqueeze model")
}

fn unresolved_unsqueeze_from_large_slice_model() -> Vec<u8> {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let len = MAX_SHAPE_DATA_ELEMS + 1;
    let source = input(&mut g, "source", DataType::Int64, &[len]);
    let starts = i64_init(&mut g, "starts", &[1], &[0]);
    let ends = i64_init(&mut g, "ends", &[1], &[1]);
    let axes = i64_init(&mut g, "slice_axes", &[1], &[0]);
    let steps = i64_init(&mut g, "steps", &[1], &[1]);
    let sliced = g.create_named_value("sliced", DataType::Int64, static_shape([1]));
    g.insert_node(Node::new(
        NodeId(0),
        "Slice",
        vec![
            Some(source),
            Some(starts),
            Some(ends),
            Some(axes),
            Some(steps),
        ],
        vec![sliced],
    ));

    let data = input(&mut g, "data", DataType::Float32, &[2]);
    let dynamic = g.intern_symbol("dynamic_unsqueeze_extent");
    let output = g.create_named_value("output", DataType::Float32, vec![Dim::Symbolic(dynamic)]);
    g.mark_value_shape_unknown(output);
    g.insert_node(Node::new(
        NodeId(0),
        "Unsqueeze",
        vec![Some(data), Some(sliced)],
        vec![output],
    ));
    g.add_output(output);
    encode_model(&Model::new(&g)).expect("encode large-source Slice to Unsqueeze model")
}

fn assert_shape_input_rejected_without_materialization(
    axes_dtype: DataType,
    axes_shape: &[usize],
    axes_bytes: &[u8],
) {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = Arc::new(HostDownloadCountingEp::new(Arc::clone(&downloads)));
    let model = unresolved_unsqueeze_model(axes_dtype, axes_shape);
    let mut session = InferenceSession::builder()
        .model_bytes(&model)
        .execution_provider(ep)
        .build()
        .expect("build unresolved Unsqueeze session");
    let data = Tensor::from_f32(&[2], &[1.0, 2.0]).unwrap();
    let axes = Tensor::from_raw(axes_dtype, axes_shape.to_vec(), axes_bytes).unwrap();

    let error = session
        .run(&[("data", &data), ("axes", &axes)])
        .expect_err("rejected shape input must leave the output unresolved");
    assert!(
        matches!(error, SessionError::UnresolvedShape { .. }),
        "expected graceful unresolved shape, got {error}"
    );
    assert_eq!(
        downloads.load(Ordering::Relaxed),
        0,
        "shape-propagation rejection must happen before copy_to_host"
    );
}

fn gqa_cache_graph(past_capacity: usize) -> Graph {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    g.opset_imports.insert("com.microsoft".into(), 1);

    let query = input(&mut g, "query", DataType::Float32, &[1, 1, 8]);
    let key = input(&mut g, "key", DataType::Float32, &[1, 1, 4]);
    let value = input(&mut g, "value", DataType::Float32, &[1, 1, 4]);
    let past_key = input(
        &mut g,
        "past_key",
        DataType::Float32,
        &[1, 2, past_capacity, 2],
    );
    let past_value = input(
        &mut g,
        "past_value",
        DataType::Float32,
        &[1, 2, past_capacity, 2],
    );
    let seqlens = input(&mut g, "seqlens_k", DataType::Int32, &[1]);
    let total = input(&mut g, "total_sequence_length", DataType::Int32, &[]);

    let attention = g.create_value(DataType::Float32, vec![]);
    let present_key = g.create_value(DataType::Float32, vec![]);
    let present_value = g.create_value(DataType::Float32, vec![]);
    let mut node = Node::new(
        NodeId(0),
        "GroupQueryAttention",
        vec![
            Some(query),
            Some(key),
            Some(value),
            Some(past_key),
            Some(past_value),
            Some(seqlens),
            Some(total),
        ],
        vec![attention, present_key, present_value],
    );
    node.domain = "com.microsoft".into();
    node.attributes
        .insert("num_heads".into(), Attribute::Int(4));
    node.attributes
        .insert("kv_num_heads".into(), Attribute::Int(2));
    g.insert_node(node);

    let registry = InferenceRegistry::default_registry();
    let imports = g.opset_imports.clone();
    registry
        .infer_graph(&mut g, &imports, MergePolicy::Permissive)
        .expect("infer GQA output shapes");
    g.add_output(attention);
    g.add_output(present_key);
    g.add_output(present_value);
    g
}

fn run_gqa_decode(past_capacity: usize) -> Vec<Tensor> {
    let mut session =
        InferenceSession::from_graph(gqa_cache_graph(past_capacity)).expect("build GQA session");
    let query = Tensor::from_f32(&[1, 1, 8], &[1.0; 8]).unwrap();
    let key = Tensor::from_f32(&[1, 1, 4], &[0.5; 4]).unwrap();
    let value = Tensor::from_f32(&[1, 1, 4], &[2.0; 4]).unwrap();
    let past_data = vec![0.25; 4 * past_capacity];
    let past_key = Tensor::from_f32(&[1, 2, past_capacity, 2], &past_data).unwrap();
    let past_value = Tensor::from_f32(&[1, 2, past_capacity, 2], &past_data).unwrap();
    let seqlens = i32_tensor(&[1], &[2]);
    let total = i32_tensor(&[], &[3]);
    session
        .run(&[
            ("query", &query),
            ("key", &key),
            ("value", &value),
            ("past_key", &past_key),
            ("past_value", &past_value),
            ("seqlens_k", &seqlens),
            ("total_sequence_length", &total),
        ])
        .expect("GQA decode succeeds")
}

#[test]
fn gqa_decode_fixed_capacity_preserves_present_cache_extent() {
    let outputs = run_gqa_decode(8);
    assert_eq!(outputs[0].shape, vec![1, 1, 8]);
    assert_eq!(outputs[1].shape, vec![1, 2, 8, 2]);
    assert_eq!(outputs[2].shape, vec![1, 2, 8, 2]);
}

#[test]
fn gqa_decode_growing_cache_extends_present_to_logical_total() {
    let outputs = run_gqa_decode(2);
    assert_eq!(outputs[0].shape, vec![1, 1, 8]);
    assert_eq!(outputs[1].shape, vec![1, 2, 3, 2]);
    assert_eq!(outputs[2].shape, vec![1, 2, 3, 2]);
}

#[test]
fn dynamic_slice_shape_propagates_through_unsqueeze_comparison_and_transpose() {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);

    let data = input(&mut g, "data", DataType::Float32, &[4]);
    let one = i64_init(&mut g, "one", &[1], &[1]);
    let starts = i64_init(&mut g, "starts", &[1], &[0]);
    let slice_axes = i64_init(&mut g, "slice_axes", &[1], &[0]);
    let steps = i64_init(&mut g, "steps", &[1], &[1]);
    let unsqueeze_axes = i64_init(&mut g, "unsqueeze_axes", &[1], &[-1]);
    let thresholds = f32_init(&mut g, "thresholds", &[2, 1], &[1.5, 2.5]);
    let dynamic_extent = g.intern_symbol("dynamic_extent");

    let data_shape = g.create_value(DataType::Int64, static_shape([1]));
    g.insert_node(Node::new(
        NodeId(0),
        "Shape",
        vec![Some(data)],
        vec![data_shape],
    ));
    let end = g.create_value(DataType::Int64, static_shape([1]));
    g.insert_node(Node::new(
        NodeId(0),
        "Sub",
        vec![Some(data_shape), Some(one)],
        vec![end],
    ));

    let sliced = g.create_value(DataType::Float32, vec![Dim::Symbolic(dynamic_extent)]);
    g.mark_value_shape_unknown(sliced);
    g.insert_node(Node::new(
        NodeId(0),
        "Slice",
        vec![
            Some(data),
            Some(starts),
            Some(end),
            Some(slice_axes),
            Some(steps),
        ],
        vec![sliced],
    ));

    let unsqueezed = g.create_value(
        DataType::Float32,
        vec![Dim::Symbolic(dynamic_extent), Dim::Static(1)],
    );
    g.mark_value_shape_unknown(unsqueezed);
    g.insert_node(Node::new(
        NodeId(0),
        "Unsqueeze",
        vec![Some(sliced), Some(unsqueeze_axes)],
        vec![unsqueezed],
    ));

    let transposed = g.create_value(
        DataType::Float32,
        vec![Dim::Static(1), Dim::Symbolic(dynamic_extent)],
    );
    g.mark_value_shape_unknown(transposed);
    let mut transpose = Node::new(
        NodeId(0),
        "Transpose",
        vec![Some(unsqueezed)],
        vec![transposed],
    );
    transpose
        .attributes
        .insert("perm".into(), Attribute::Ints(vec![1, 0]));
    g.insert_node(transpose);
    let compared = g.create_value(
        DataType::Bool,
        vec![Dim::Static(2), Dim::Symbolic(dynamic_extent)],
    );
    g.mark_value_shape_unknown(compared);
    g.insert_node(Node::new(
        NodeId(0),
        "Less",
        vec![Some(transposed), Some(thresholds)],
        vec![compared],
    ));
    g.add_output(compared);

    let mut session = InferenceSession::from_graph(g).expect("build dynamic-shape session");
    let data = Tensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let outputs = session.run(&[("data", &data)]).expect("run dynamic chain");

    assert_eq!(outputs[0].shape, vec![2, 3]);
    assert_eq!(outputs[0].dtype, DataType::Bool);
    assert_eq!(outputs[0].as_bytes(), &[1, 0, 0, 1, 1, 0]);
}

#[test]
fn oversized_shape_input_is_rejected_before_host_materialization() {
    let len = MAX_SHAPE_DATA_ELEMS + 1;
    assert_shape_input_rejected_without_materialization(
        DataType::Int64,
        &[len],
        &vec![0u8; len * std::mem::size_of::<i64>()],
    );
}

#[test]
fn non_integer_shape_input_is_rejected_before_host_materialization() {
    assert_shape_input_rejected_without_materialization(
        DataType::Float32,
        &[1],
        &0.0f32.to_le_bytes(),
    );
}

#[test]
fn rank_two_shape_input_is_rejected_before_host_materialization() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = Arc::new(HostDownloadCountingEp::new(Arc::clone(&downloads)));
    let model = unresolved_unsqueeze_model(DataType::Int64, &[1, 1]);
    let error = match InferenceSession::builder()
        .model_bytes(&model)
        .execution_provider(ep)
        .build()
    {
        Ok(_) => panic!("rank-two Unsqueeze axes must be rejected during shape inference"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("Unsqueeze") && message.contains("1-D tensor"),
        "error must identify the invalid Unsqueeze axes contract: {message}"
    );
    assert_eq!(
        downloads.load(Ordering::Relaxed),
        0,
        "shape-inference rejection must not materialize the invalid axes tensor"
    );
}

#[test]
fn small_shape_view_with_oversized_source_is_rejected_before_host_materialization() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = Arc::new(HostDownloadCountingEp::new(Arc::clone(&downloads)));
    let model = unresolved_unsqueeze_from_large_slice_model();
    let mut session = InferenceSession::builder()
        .model_bytes(&model)
        .execution_provider(ep)
        .build()
        .expect("build large-source Slice to Unsqueeze session");
    let len = MAX_SHAPE_DATA_ELEMS + 1;
    let source = i64_tensor(&[len], &vec![0; len]);
    let data = Tensor::from_f32(&[2], &[1.0, 2.0]).unwrap();

    let error = session
        .run(&[("source", &source), ("data", &data)])
        .expect_err("oversized view source must leave the output unresolved");
    assert!(
        matches!(error, SessionError::UnresolvedShape { .. }),
        "expected graceful unresolved shape, got {error}"
    );
    assert_eq!(
        downloads.load(Ordering::Relaxed),
        0,
        "shape-propagation rejection must happen before copying a view's source buffer"
    );
}

#[test]
fn scalar_integer_shape_inputs_still_propagate_range_extent() {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let start = input(&mut g, "start", DataType::Int64, &[]);
    let limit = input(&mut g, "limit", DataType::Int64, &[]);
    let delta = input(&mut g, "delta", DataType::Int64, &[]);
    let dynamic = g.intern_symbol("dynamic_range_extent");
    let output = g.create_named_value("output", DataType::Int64, vec![Dim::Symbolic(dynamic)]);
    g.mark_value_shape_unknown(output);
    g.insert_node(Node::new(
        NodeId(0),
        "Range",
        vec![Some(start), Some(limit), Some(delta)],
        vec![output],
    ));
    g.add_output(output);

    let mut session = InferenceSession::from_graph(g).expect("build dynamic Range session");
    let start = i64_tensor(&[], &[2]);
    let limit = i64_tensor(&[], &[8]);
    let delta = i64_tensor(&[], &[2]);
    let outputs = session
        .run(&[("start", &start), ("limit", &limit), ("delta", &delta)])
        .expect("scalar Range shape propagation succeeds");

    assert_eq!(outputs[0].shape, vec![3]);
    assert_eq!(outputs[0].dtype, DataType::Int64);
    let expected: Vec<u8> = [2i64, 4, 6]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect();
    assert_eq!(outputs[0].as_bytes(), expected);
}

/// Insert an op node whose single output carries an explicit (possibly
/// symbolic) shape — mirroring what the loader's shape inference would produce.
fn op_shaped(
    g: &mut Graph,
    op_type: &str,
    inputs: &[ValueId],
    out_dtype: DataType,
    out_shape: Shape,
    attrs: &[(&str, Attribute)],
) -> ValueId {
    g.opset_imports.entry(String::new()).or_insert(17);
    let out = g.create_value(out_dtype, out_shape);
    let mut node = Node::new(
        NodeId(0),
        op_type,
        inputs.iter().map(|&v| Some(v)).collect(),
        vec![out],
    );
    for (k, v) in attrs {
        node.attributes.insert((*k).to_string(), v.clone());
    }
    g.insert_node(node);
    out
}

#[test]
fn unsupported_op_error_is_actionable() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = input(&mut graph, "x", DataType::Float32, &[1]);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([1]));
    let mut node = Node::new(NodeId(0), UNSUPPORTED_OP_SENTINEL, vec![Some(x)], vec![y]);
    node.name = "unsupported_activation".to_string();
    graph.insert_node(node);
    graph.add_output(y);

    let message = match InferenceSession::from_graph(graph) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("unsupported operator unexpectedly built"),
    };
    assert!(message.contains(UNSUPPORTED_OP_SENTINEL), "{message}");
    assert!(message.contains("ai.onnx"), "{message}");
    assert!(message.contains("unsupported_activation"), "{message}");
    assert!(message.contains("opset 17"), "{message}");
    assert!(message.contains("cpu_ep"), "{message}");
    assert!(
        message.contains(&format!(
            "no handler for ai.onnx::{UNSUPPORTED_OP_SENTINEL} at opset 17"
        )),
        "{message}"
    );
    assert!(message.contains("add a claim+handler"), "{message}");
    assert!(message.contains("To fix:"), "{message}");
}

fn standard_gelu_graph(opset: u64) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let x = input(&mut graph, "x", DataType::Float32, &[2]);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(NodeId(0), "Gelu", vec![Some(x)], vec![y]));
    graph.add_output(y);
    graph
}

#[test]
fn too_new_kernel_is_declined_at_claim_time_with_actionable_reason() {
    let error = match InferenceSession::from_graph(standard_gelu_graph(19)) {
        Err(error) => error,
        Ok(_) => panic!("opset-19 standard Gelu must be declined"),
    };

    match error {
        SessionError::UnsupportedOp {
            op_type,
            domain,
            opset,
            reason,
            ..
        } => {
            assert_eq!(op_type, "Gelu");
            assert_eq!(domain, "ai.onnx");
            assert_eq!(opset, OpsetVersion::Known(19));
            assert!(
                reason.contains("no handler for ai.onnx::Gelu at opset 19"),
                "{reason}"
            );
            assert!(reason.contains("registers Gelu since opset 20"), "{reason}");
        }
        other => panic!("expected actionable UnsupportedOp, got {other}"),
    }
}

#[test]
fn kernel_is_claimed_at_its_supported_since_opset() {
    InferenceSession::from_graph(standard_gelu_graph(20))
        .expect("opset-20 standard Gelu should be claimed and compiled");
}

#[test]
fn unsupported_op_error_formats_unnamed_node_gracefully() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 0);
    let x = input(&mut graph, "x", DataType::Float32, &[1]);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(0),
        UNSUPPORTED_OP_SENTINEL,
        vec![Some(x)],
        vec![y],
    ));
    graph.add_output(y);

    let message = match InferenceSession::from_graph(graph) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("unsupported operator unexpectedly built"),
    };
    assert!(message.contains(UNSUPPORTED_OP_SENTINEL), "{message}");
    assert!(
        message.contains("node <unnamed node #0>, opset 0"),
        "{message}"
    );
    assert!(!message.contains("node \"\""), "{message}");
}

#[test]
fn from_graph_rejects_missing_opset_import_at_load_time() {
    let mut graph = Graph::new();
    let x = input(&mut graph, "x", DataType::Float32, &[1]);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([1]));
    // Sigmoid is safe here: missing opset validation runs before operator lookup.
    let mut node = Node::new(NodeId(0), "Sigmoid", vec![Some(x)], vec![y]);
    node.name = "missing_opset_import".to_string();
    graph.insert_node(node);
    graph.add_output(y);

    let message = match InferenceSession::from_graph(graph) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("illegal graph unexpectedly built"),
    };
    assert_eq!(
        message,
        "illegal ONNX model: operator ai.onnx::Sigmoid at node \"missing_opset_import\" uses \
         domain 'ai.onnx' but no corresponding opset_import is declared. RULES #1: the model must \
         declare an opset_import for domain 'ai.onnx'; if you built this graph programmatically, \
         add it before loading; if this is a file, the model is malformed/invalid per the ONNX spec"
    );
    assert!(message.contains("Sigmoid"), "{message}");
    assert!(message.contains("ai.onnx"), "{message}");
    assert!(message.contains("RULES #1"), "{message}");
    assert!(!message.contains("18446744073709551615"), "{message}");
}

// --- reference implementations ---------------------------------------------

fn ref_matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn ref_add_rowvec(m: &[f32], rows: usize, cols: usize, bias: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = m[r * cols + c] + bias[c];
        }
    }
    out
}

fn ref_layernorm_last(
    x: &[f32],
    rows: usize,
    cols: usize,
    scale: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..r * cols + cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            out[r * cols + c] = (row[c] - mean) * inv * scale[c] + bias[c];
        }
    }
    out
}

fn ref_relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-4, "element {i}: got {g}, want {w}");
    }
}

// --- tests ------------------------------------------------------------------

/// MatMul → Add → LayerNormalization → Relu, a realistic multi-node chain.
#[test]
fn matmul_add_layernorm_relu_chain_matches_reference() {
    // Dimensions: X[2,3] · W[3,4] → [2,4], + bias[4], layernorm last axis, relu.
    let x_data = [0.5f32, -1.0, 2.0, 1.5, 0.0, -0.5];
    let w_data = [
        0.1f32, 0.2, -0.3, 0.4, //
        -0.5, 0.6, 0.7, -0.8, //
        0.9, -1.0, 0.2, 0.3,
    ];
    let bias = [0.1f32, -0.2, 0.3, 0.05];
    let scale = [1.2f32, 0.8, 1.0, 0.5];
    let ln_bias = [0.0f32, 0.1, -0.1, 0.2];

    let mut g = Graph::new();
    let x = input(&mut g, "X", DataType::Float32, &[2, 3]);
    let w = f32_init(&mut g, "W", &[3, 4], &w_data);
    let m = op(&mut g, "MatMul", &[x, w], DataType::Float32, &[2, 4], &[]);
    let b = f32_init(&mut g, "B", &[4], &bias);
    let a = op(&mut g, "Add", &[m, b], DataType::Float32, &[2, 4], &[]);
    let s = f32_init(&mut g, "Scale", &[4], &scale);
    let bn = f32_init(&mut g, "LnBias", &[4], &ln_bias);
    let l = op(
        &mut g,
        "LayerNormalization",
        &[a, s, bn],
        DataType::Float32,
        &[2, 4],
        &[("axis", Attribute::Int(-1))],
    );
    let y = op(&mut g, "Relu", &[l], DataType::Float32, &[2, 4], &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build session");

    let x_tensor = Tensor::from_f32(&[2, 3], &x_data).unwrap();
    let outputs = session.run(&[("X", &x_tensor)]).expect("run");
    assert_eq!(outputs.len(), 1);

    // Reference.
    let m_ref = ref_matmul(&x_data, 2, 3, &w_data, 4);
    let a_ref = ref_add_rowvec(&m_ref, 2, 4, &bias);
    let l_ref = ref_layernorm_last(&a_ref, 2, 4, &scale, &ln_bias, 1e-5);
    let y_ref = ref_relu(&l_ref);

    assert_close(&outputs[0].to_vec_f32(), &y_ref);
    assert_eq!(outputs[0].shape, vec![2, 4]);
}

/// Gather (embedding lookup) → Transpose, exercising an integer-index op and a
/// layout-permuting op in one graph.
#[test]
fn gather_then_transpose_matches_reference() {
    // Embedding table [4,3]; gather rows [2,0,3] → [3,3]; transpose → [3,3]^T.
    let table = [
        0.0f32, 1.0, 2.0, //
        3.0, 4.0, 5.0, //
        6.0, 7.0, 8.0, //
        9.0, 10.0, 11.0,
    ];
    let idx = [2i64, 0, 3];

    let mut g = Graph::new();
    let data = f32_init(&mut g, "Table", &[4, 3], &table);
    let indices = input(&mut g, "Idx", DataType::Int64, &[3]);
    let gathered = op(
        &mut g,
        "Gather",
        &[data, indices],
        DataType::Float32,
        &[3, 3],
        &[("axis", Attribute::Int(0))],
    );
    let transposed = op(
        &mut g,
        "Transpose",
        &[gathered],
        DataType::Float32,
        &[3, 3],
        &[("perm", Attribute::Ints(vec![1, 0]))],
    );
    g.add_output(transposed);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    let idx_tensor = Tensor::from_i64(&[3], &idx).unwrap();
    let outputs = session.run(&[("Idx", &idx_tensor)]).expect("run");

    // Reference: gather rows then transpose 3x3.
    let mut gathered_ref = Vec::new();
    for &i in &idx {
        let base = i as usize * 3;
        gathered_ref.extend_from_slice(&table[base..base + 3]);
    }
    let mut want = vec![0.0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            want[c * 3 + r] = gathered_ref[r * 3 + c];
        }
    }
    assert_close(&outputs[0].to_vec_f32(), &want);
}

/// The shape-keyed kernel cache is populated once and reused on every run: hits
/// grow while the compiled-entry count and miss count stay fixed (§11.1).
#[test]
fn shape_keyed_cache_is_reused_across_runs() {
    let mut g = Graph::new();
    let x = input(&mut g, "X", DataType::Float32, &[2, 2]);
    let w = f32_init(&mut g, "W", &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    let m = op(&mut g, "MatMul", &[x, w], DataType::Float32, &[2, 2], &[]);
    let y = op(&mut g, "Relu", &[m], DataType::Float32, &[2, 2], &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build");

    // After build (compile pass): every node compiled once, no hits.
    let after_build = session.cache_stats();
    assert_eq!(after_build.entries, 2, "two nodes compiled");
    assert_eq!(after_build.misses, 2);
    assert_eq!(after_build.hits, 0);

    let x_tensor = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();

    let out1 = session.run(&[("X", &x_tensor)]).unwrap();
    let after_run1 = session.cache_stats();
    assert_eq!(after_run1.entries, 2, "no new entries on run");
    assert_eq!(after_run1.misses, 2, "no recompilation");
    assert_eq!(after_run1.hits, 2, "each node served from cache");

    let out2 = session.run(&[("X", &x_tensor)]).unwrap();
    let after_run2 = session.cache_stats();
    assert_eq!(after_run2.entries, 2);
    assert_eq!(after_run2.misses, 2);
    assert_eq!(after_run2.hits, 4, "second run hit the cache again");

    // Identity matmul + relu of [1,2,3,4] → [1,2,3,4].
    assert_close(&out1[0].to_vec_f32(), &[1.0, 2.0, 3.0, 4.0]);
    assert_close(&out2[0].to_vec_f32(), &[1.0, 2.0, 3.0, 4.0]);
}

/// `warmup` names must reference real inputs; a bad name is rejected, a good
/// one keeps the cache warm.
#[test]
fn warmup_validates_input_names() {
    let mut g = Graph::new();
    let x = input(&mut g, "X", DataType::Float32, &[1, 2]);
    let y = op(&mut g, "Relu", &[x], DataType::Float32, &[1, 2], &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).unwrap();
    assert!(
        session
            .warmup(&[WarmupShape {
                input_name: "nope".into(),
                shape: vec![1, 2],
            }])
            .is_err()
    );
    assert!(
        session
            .warmup(&[WarmupShape {
                input_name: "X".into(),
                shape: vec![1, 2],
            }])
            .is_ok()
    );
}

/// A missing required input is reported, not silently defaulted.
#[test]
fn missing_input_is_rejected() {
    let mut g = Graph::new();
    let x = input(&mut g, "X", DataType::Float32, &[1, 2]);
    let y = op(&mut g, "Relu", &[x], DataType::Float32, &[1, 2], &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).unwrap();
    let err = session.run(&[]).unwrap_err();
    assert!(matches!(
        err,
        onnx_runtime_session::SessionError::InputNotFound { .. }
    ));
}

/// A shape-mismatched input tensor is rejected before dispatch.
#[test]
fn input_shape_mismatch_is_rejected() {
    let mut g = Graph::new();
    let x = input(&mut g, "X", DataType::Float32, &[2, 2]);
    let y = op(&mut g, "Relu", &[x], DataType::Float32, &[2, 2], &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).unwrap();
    let wrong = Tensor::from_f32(&[3, 2], &[0.0; 6]).unwrap();
    let err = session.run(&[("X", &wrong)]).unwrap_err();
    assert!(matches!(
        err,
        onnx_runtime_session::SessionError::ShapeMismatch { .. }
    ));
}

// --- dynamic (symbolic) shape tests ----------------------------------------

/// A graph with a symbolic leading dim (`[batch, 4]` MatMul → Add → Relu) runs
/// correctly for two *different* batch sizes in the same session: shapes resolve
/// from the actual inputs, buffers re-size, and the kernel cache re-resolves for
/// the new shape while reusing the plan for a repeated shape.
#[test]
fn symbolic_batch_matmul_chain_runs_for_multiple_shapes() {
    let w_data = [
        1.0f32, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    let bias = [0.5f32, -0.5, 1.0, -1.0];

    let mut g = Graph::new();
    let batch = g.intern_symbol("batch");
    let sym_row = || vec![Dim::Symbolic(batch), Dim::Static(4)];

    let x = input_shaped(&mut g, "X", DataType::Float32, sym_row());
    let w = f32_init(&mut g, "W", &[4, 4], &w_data);
    let m = op_shaped(&mut g, "MatMul", &[x, w], DataType::Float32, sym_row(), &[]);
    let b = f32_init(&mut g, "B", &[4], &bias);
    let a = op_shaped(&mut g, "Add", &[m, b], DataType::Float32, sym_row(), &[]);
    let y = op_shaped(&mut g, "Relu", &[a], DataType::Float32, sym_row(), &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build symbolic session");

    // A symbolic graph is not compiled at build (no concrete shapes yet).
    let after_build = session.cache_stats();
    assert_eq!(
        after_build.entries, 0,
        "no kernels compiled before first run"
    );
    assert_eq!(after_build.misses, 0);

    let run_batch = |session: &mut InferenceSession, rows: usize, fill: f32| -> Vec<f32> {
        let data: Vec<f32> = (0..rows * 4).map(|i| fill + i as f32).collect();
        let x_tensor = Tensor::from_f32(&[rows, 4], &data).unwrap();
        let out = session.run(&[("X", &x_tensor)]).expect("run");
        assert_eq!(out[0].shape, vec![rows, 4]);
        // Reference: identity matmul + row bias + relu.
        let m_ref = ref_matmul(&data, rows, 4, &w_data, 4);
        let a_ref = ref_add_rowvec(&m_ref, rows, 4, &bias);
        let y_ref = ref_relu(&a_ref);
        assert_close(&out[0].to_vec_f32(), &y_ref);
        out[0].to_vec_f32()
    };

    // batch = 2 → first shape: three nodes compiled (misses), no hits.
    run_batch(&mut session, 2, 0.0);
    let s2 = session.cache_stats();
    assert_eq!(s2.entries, 3, "three nodes compiled for batch=2");
    assert_eq!(s2.misses, 3);
    assert_eq!(s2.hits, 0);

    // batch = 3 → new resolved shape: re-resolves + re-plans (3 more entries).
    run_batch(&mut session, 3, 10.0);
    let s3 = session.cache_stats();
    assert_eq!(
        s3.entries, 6,
        "batch=3 adds three distinct shape-keyed entries"
    );
    assert_eq!(s3.misses, 6);
    assert_eq!(s3.hits, 0);

    // batch = 2 again → the batch=2 plan is reused (cache hits, no new entries).
    run_batch(&mut session, 2, 100.0);
    let s2b = session.cache_stats();
    assert_eq!(s2b.entries, 6, "no new entries: batch=2 plan reused");
    assert_eq!(s2b.misses, 6);
    assert_eq!(s2b.hits, 3, "each node served from the batch=2 cache");
}

/// Two inputs share a symbol (`batch`); supplying them with *conflicting*
/// concrete sizes is a resolution error, not a silently-wrong run.
#[test]
fn symbol_conflict_across_inputs_is_rejected() {
    let mut g = Graph::new();
    let batch = g.intern_symbol("batch");
    let sym_row = || vec![Dim::Symbolic(batch), Dim::Static(4)];

    let a = input_shaped(&mut g, "A", DataType::Float32, sym_row());
    let b = input_shaped(&mut g, "B", DataType::Float32, sym_row());
    let s = op_shaped(&mut g, "Add", &[a, b], DataType::Float32, sym_row(), &[]);
    g.add_output(s);

    let mut session = InferenceSession::from_graph(g).expect("build");

    let a_t = Tensor::from_f32(&[2, 4], &[0.0; 8]).unwrap();
    let b_t = Tensor::from_f32(&[3, 4], &[0.0; 12]).unwrap();
    let err = session.run(&[("A", &a_t), ("B", &b_t)]).unwrap_err();
    assert!(
        matches!(err, SessionError::SymbolConflict { .. }),
        "expected SymbolConflict, got {err:?}"
    );

    // Agreeing sizes resolve fine.
    let a_ok = Tensor::from_f32(&[2, 4], &[1.0; 8]).unwrap();
    let b_ok = Tensor::from_f32(&[2, 4], &[2.0; 8]).unwrap();
    let out = session.run(&[("A", &a_ok), ("B", &b_ok)]).expect("run");
    assert_close(&out[0].to_vec_f32(), &[3.0; 8]);
    assert_eq!(out[0].shape, vec![2, 4]);
}

/// A registered op whose declared output shape carries an unbound symbol can be
/// sized from its concrete runtime inputs via the standard shape rule.
#[test]
fn registered_shape_rule_resolves_unbound_declared_symbol() {
    let mut g = Graph::new();
    let batch = g.intern_symbol("batch");
    let ghost = g.intern_symbol("ghost"); // never appears on any input

    let x = input_shaped(
        &mut g,
        "X",
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(4)],
    );
    // Relu declares an unbindable symbol on its leading dim.
    let y = op_shaped(
        &mut g,
        "Relu",
        &[x],
        DataType::Float32,
        vec![Dim::Symbolic(ghost), Dim::Static(4)],
        &[],
    );
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build");
    let x_t = Tensor::from_f32(&[2, 4], &[0.0; 8]).unwrap();
    let outputs = session
        .run(&[("X", &x_t)])
        .expect("runtime shape inference");
    assert_eq!(outputs[0].shape, vec![2, 4]);
}

/// A symbolic input supplied with the wrong rank is rejected before dispatch.
#[test]
fn symbolic_input_rank_mismatch_is_rejected() {
    let mut g = Graph::new();
    let batch = g.intern_symbol("batch");
    let x = input_shaped(
        &mut g,
        "X",
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(4)],
    );
    let y = op_shaped(
        &mut g,
        "Relu",
        &[x],
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(4)],
        &[],
    );
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build");
    // Rank-3 tensor for a rank-2 declared input.
    let wrong = Tensor::from_f32(&[2, 2, 4], &[0.0; 16]).unwrap();
    let err = session.run(&[("X", &wrong)]).unwrap_err();
    assert!(
        matches!(err, SessionError::RankMismatch { .. }),
        "expected RankMismatch, got {err:?}"
    );
}

/// A static dim declared alongside a symbolic one must still match exactly.
#[test]
fn symbolic_input_static_dim_mismatch_is_rejected() {
    let mut g = Graph::new();
    let batch = g.intern_symbol("batch");
    let x = input_shaped(
        &mut g,
        "X",
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(4)],
    );
    let y = op_shaped(
        &mut g,
        "Relu",
        &[x],
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(4)],
        &[],
    );
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build");
    // batch is free, but the trailing static dim (4) is violated (here 5).
    let wrong = Tensor::from_f32(&[2, 5], &[0.0; 10]).unwrap();
    let err = session.run(&[("X", &wrong)]).unwrap_err();
    assert!(
        matches!(err, SessionError::ShapeMismatch { .. }),
        "expected ShapeMismatch on the static dim, got {err:?}"
    );
}

/// A subgraph-bearing op the CPU EP cannot execute (anything other than the
/// implemented `If`/`Loop`/`Scan`) is rejected at session-build time
/// (from_graph path), mirroring the disk loader — we fail fast with a RULES #1
/// message instead of lazily at run time or silently skipping the subgraph.
/// The three implemented control-flow ops are covered by `tests/control_flow.rs`.
#[test]
fn from_graph_rejects_unimplemented_control_flow_subgraph_at_build() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = input(&mut graph, "x", DataType::Float32, &[1]);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([1]));
    // `SequenceMap` is a real ONNX subgraph-bearing op this runtime does not
    // implement — it must still be rejected fast.
    let mut node = Node::new(NodeId(0), "SequenceMap", vec![Some(x)], vec![y]);
    node.name = "control_flow_seqmap".to_string();
    node.attributes
        .insert("body".to_string(), Attribute::Graph(Box::new(Graph::new())));
    graph.insert_node(node);
    graph.add_output(y);

    let message = match InferenceSession::from_graph(graph) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("unimplemented control-flow subgraph unexpectedly built"),
    };
    assert!(message.contains("SequenceMap"), "{message}");
    assert!(message.contains("body"), "{message}");
    assert!(message.contains("control-flow"), "{message}");
    assert!(message.contains("RULES #1"), "{message}");
}

/// A node consuming an unsourced tensor is rejected at session-build time.
#[test]
fn from_graph_rejects_dangling_tensor_reference_at_build() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = input(&mut graph, "x", DataType::Float32, &[2]);
    // `z` is created but never sourced (no input, initializer, or producer).
    let z = graph.create_named_value("z", DataType::Float32, static_shape([2]));
    let y = graph.create_named_value("y", DataType::Float32, static_shape([2]));
    let mut node = Node::new(NodeId(0), "Add", vec![Some(x), Some(z)], vec![y]);
    node.name = "dangling_add".to_string();
    graph.insert_node(node);
    graph.add_output(y);

    let message = match InferenceSession::from_graph(graph) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("dangling reference unexpectedly built"),
    };
    assert!(message.contains("'z'"), "{message}");
    assert!(message.contains("Add"), "{message}");
    assert!(message.contains("RULES #1"), "{message}");
}

/// The executor must obey dependencies rather than insertion order. This also
/// locks down the deterministic NodeId tie-break used by graph planning.
#[test]
fn executor_topologically_orders_reverse_inserted_dependencies_deterministically() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = input(&mut graph, "x", DataType::Float32, &[3]);
    let intermediate =
        graph.create_named_value("intermediate", DataType::Float32, static_shape([3]));
    let y = graph.create_named_value("y", DataType::Float32, static_shape([3]));

    // Insert the consumer first. The graph is valid, but node #0 cannot run
    // until the later-inserted node #1 has produced `intermediate`.
    let consumer = graph.insert_node(Node::new(
        NodeId(0),
        "Relu",
        vec![Some(intermediate)],
        vec![y],
    ));
    let producer = graph.insert_node(Node::new(
        NodeId(0),
        "Relu",
        vec![Some(x)],
        vec![intermediate],
    ));
    graph.add_output(y);

    let expected_order = vec![producer, consumer];
    assert_eq!(graph.topological_order().unwrap(), expected_order);
    for _ in 0..8 {
        assert_eq!(graph.topological_order().unwrap(), expected_order);
    }

    let mut session = InferenceSession::from_graph(graph).expect("build reverse-inserted DAG");
    let x = Tensor::from_f32(&[3], &[-2.0, 0.5, 3.0]).unwrap();
    let first = session.run(&[("x", &x)]).expect("first run");
    let second = session.run(&[("x", &x)]).expect("second run");
    assert_eq!(first[0].to_vec_f32(), vec![0.0, 0.5, 3.0]);
    assert_eq!(second[0].to_vec_f32(), first[0].to_vec_f32());
}

/// A cyclic graph must be rejected during plan construction, never partially
/// executed or accepted because its values happen to be present in the IR.
#[test]
fn from_graph_rejects_cyclic_execution_plan() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let a = graph.create_named_value("a", DataType::Float32, static_shape([1]));
    let b = graph.create_named_value("b", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(b)], vec![a]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(a)], vec![b]));
    graph.add_output(a);

    let error = match InferenceSession::from_graph(graph) {
        Err(error) => error,
        Ok(_) => panic!("cyclic graph unexpectedly built"),
    };
    assert!(
        matches!(
            error,
            SessionError::Graph(onnx_runtime_ir::GraphError::CycleDetected)
        ),
        "expected a CycleDetected graph error, got {error:?}"
    );
}

/// Initializers are immutable graph sources. A node output cannot reuse an
/// initializer value, since that would turn read-only weight storage writable.
#[test]
fn from_graph_rejects_initializer_reused_as_node_output() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = input(&mut graph, "x", DataType::Float32, &[2]);
    let weight = f32_init(&mut graph, "weight", &[2], &[1.0, 2.0]);
    let mut overwrite = Node::new(NodeId(0), "Relu", vec![Some(x)], vec![weight]);
    overwrite.name = "overwrites_weight".to_string();
    graph.insert_node(overwrite);
    graph.add_output(weight);

    let message = match InferenceSession::from_graph(graph) {
        Err(error) => error.to_string(),
        Ok(_) => panic!("producer-backed initializer unexpectedly built"),
    };
    assert!(message.contains("weight"), "{message}");
    assert!(message.contains("overwrites_weight"), "{message}");
    assert!(message.contains("initializer"), "{message}");
}

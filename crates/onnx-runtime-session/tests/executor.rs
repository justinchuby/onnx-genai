//! Integration tests for the sequential CPU executor (Track D).
//!
//! Each test hand-builds a small [`Graph`] via the IR API, runs it through the
//! public [`InferenceSession`] surface, and asserts the output matches a
//! reference computed here in the test. Nothing below names a model or bakes in
//! a fixed shape path — the executor is exercised as a generic Graph runner.

use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use onnx_runtime_ep_api::{
    CaptureSupport, DeviceBuffer, EpConfig, EpError, ExecutionProvider,
    ExecutorArtifactFinalization, ExecutorArtifactPending, ExecutorArtifactReadinessEpoch,
    ExecutorInstanceId, Fence, Kernel, KernelMatch, Result as EpResult, TensorMetadata, TensorMut,
    TensorView, ViewOutput, WorkspaceRequirement,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, DeviceType, Dim, Graph, Node, NodeId, Shape, TensorData,
    TensorLayout, ValueId, WeightRef, static_shape,
};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{
    DeviceGraphCaptureResult, InferenceSession, OpsetVersion, SessionError, Tensor, WarmupShape,
};
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

#[derive(Clone, Copy)]
enum TestArtifactFinalization {
    Complete,
    PendingOnce,
    StructuralDecline,
    FailOnce,
    ReadyPendingFailedReady,
}

struct HostDownloadCountingEp {
    cpu: CpuExecutionProvider,
    host_downloads: Arc<AtomicUsize>,
    kernel_compiles: Arc<Mutex<HashMap<ExecutorInstanceId, usize>>>,
    kernel_executions: Arc<Mutex<HashMap<ExecutorInstanceId, usize>>>,
    route_readiness_checks: Arc<Mutex<HashMap<ExecutorInstanceId, usize>>>,
    route_finalizations: Arc<Mutex<HashMap<ExecutorInstanceId, usize>>>,
    route_terminal_outcomes: Arc<Mutex<HashMap<ExecutorInstanceId, usize>>>,
    route_boundary_consumes: Arc<AtomicUsize>,
    route_drains: Arc<Mutex<HashMap<ExecutorInstanceId, usize>>>,
    route_install_graph_nodes: Arc<Mutex<HashMap<ExecutorInstanceId, usize>>>,
    assert_finalized_before_execute: bool,
    capture_checks: Arc<AtomicUsize>,
    artifact_finalization: TestArtifactFinalization,
    fake_device_graph: bool,
    graph_capturing: Arc<AtomicBool>,
    graph_installed: Arc<AtomicBool>,
    graph_segment_replays: Arc<AtomicUsize>,
    graph_fast_replays: Arc<AtomicUsize>,
}

impl HostDownloadCountingEp {
    fn new(host_downloads: Arc<AtomicUsize>) -> Self {
        let mut cpu = CpuExecutionProvider::new();
        cpu.initialize(&EpConfig::default()).unwrap();
        Self {
            cpu,
            host_downloads,
            kernel_compiles: Arc::new(Mutex::new(HashMap::new())),
            kernel_executions: Arc::new(Mutex::new(HashMap::new())),
            route_readiness_checks: Arc::new(Mutex::new(HashMap::new())),
            route_finalizations: Arc::new(Mutex::new(HashMap::new())),
            route_terminal_outcomes: Arc::new(Mutex::new(HashMap::new())),
            route_boundary_consumes: Arc::new(AtomicUsize::new(0)),
            route_drains: Arc::new(Mutex::new(HashMap::new())),
            route_install_graph_nodes: Arc::new(Mutex::new(HashMap::new())),
            assert_finalized_before_execute: false,
            capture_checks: Arc::new(AtomicUsize::new(0)),
            artifact_finalization: TestArtifactFinalization::Complete,
            fake_device_graph: false,
            graph_capturing: Arc::new(AtomicBool::new(false)),
            graph_installed: Arc::new(AtomicBool::new(false)),
            graph_segment_replays: Arc::new(AtomicUsize::new(0)),
            graph_fast_replays: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn new_lifecycle(host_downloads: Arc<AtomicUsize>) -> Self {
        Self {
            assert_finalized_before_execute: true,
            ..Self::new(host_downloads)
        }
    }

    fn new_pending_once_lifecycle(host_downloads: Arc<AtomicUsize>) -> Self {
        Self {
            assert_finalized_before_execute: true,
            artifact_finalization: TestArtifactFinalization::PendingOnce,
            ..Self::new(host_downloads)
        }
    }

    fn new_structural_decline_lifecycle(host_downloads: Arc<AtomicUsize>) -> Self {
        Self {
            assert_finalized_before_execute: true,
            artifact_finalization: TestArtifactFinalization::StructuralDecline,
            ..Self::new(host_downloads)
        }
    }

    fn new_fail_once_lifecycle(host_downloads: Arc<AtomicUsize>) -> Self {
        Self {
            assert_finalized_before_execute: true,
            artifact_finalization: TestArtifactFinalization::FailOnce,
            ..Self::new(host_downloads)
        }
    }

    fn new_fast_replay_lifecycle(host_downloads: Arc<AtomicUsize>) -> Self {
        Self {
            assert_finalized_before_execute: true,
            artifact_finalization: TestArtifactFinalization::ReadyPendingFailedReady,
            fake_device_graph: true,
            ..Self::new(host_downloads)
        }
    }

    fn kernel_compiles(&self) -> Arc<Mutex<HashMap<ExecutorInstanceId, usize>>> {
        Arc::clone(&self.kernel_compiles)
    }

    fn kernel_executions(&self) -> Arc<Mutex<HashMap<ExecutorInstanceId, usize>>> {
        Arc::clone(&self.kernel_executions)
    }

    fn route_finalizations(&self) -> Arc<Mutex<HashMap<ExecutorInstanceId, usize>>> {
        Arc::clone(&self.route_finalizations)
    }

    fn route_readiness_checks(&self) -> Arc<Mutex<HashMap<ExecutorInstanceId, usize>>> {
        Arc::clone(&self.route_readiness_checks)
    }

    fn route_terminal_outcomes(&self) -> Arc<Mutex<HashMap<ExecutorInstanceId, usize>>> {
        Arc::clone(&self.route_terminal_outcomes)
    }

    fn route_boundary_consumes(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.route_boundary_consumes)
    }

    fn route_drains(&self) -> Arc<Mutex<HashMap<ExecutorInstanceId, usize>>> {
        Arc::clone(&self.route_drains)
    }

    fn route_install_graph_nodes(&self) -> Arc<Mutex<HashMap<ExecutorInstanceId, usize>>> {
        Arc::clone(&self.route_install_graph_nodes)
    }

    fn capture_checks(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.capture_checks)
    }

    fn graph_segment_replays(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.graph_segment_replays)
    }

    fn graph_fast_replays(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.graph_fast_replays)
    }
}

struct FinalizationCheckingKernel {
    inner: Box<dyn Kernel>,
    executor: ExecutorInstanceId,
    terminal_outcomes: Arc<Mutex<HashMap<ExecutorInstanceId, usize>>>,
    executions: Arc<Mutex<HashMap<ExecutorInstanceId, usize>>>,
    capture_checks: Arc<AtomicUsize>,
    force_capture_supported: bool,
}

impl Kernel for FinalizationCheckingKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.inner.set_constant_inputs(constant_inputs);
    }

    fn set_capture_seq_independent(&mut self, seq_independent: bool) {
        self.inner.set_capture_seq_independent(seq_independent);
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> EpResult<()> {
        assert!(
            scoped_count(&self.terminal_outcomes, self.executor) > 0,
            "kernel executed before provider artifacts finalized"
        );
        *self
            .executions
            .lock()
            .unwrap()
            .entry(self.executor)
            .or_default() += 1;
        self.inner.execute(inputs, outputs)
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
    ) -> EpResult<WorkspaceRequirement> {
        self.inner.workspace_requirement(inputs)
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        self.inner.supports_strided_input(input_idx)
    }

    fn view_outputs(
        &self,
        inputs: &[TensorView],
        output_shapes: &[Vec<usize>],
        num_outputs: usize,
    ) -> Option<Vec<ViewOutput>> {
        self.inner.view_outputs(inputs, output_shapes, num_outputs)
    }

    fn may_produce_views(&self) -> bool {
        self.inner.may_produce_views()
    }

    fn capture_support(&self) -> CaptureSupport {
        assert!(
            scoped_count(&self.terminal_outcomes, self.executor) > 0,
            "capture audit reached a kernel before provider artifacts finalized"
        );
        self.capture_checks.fetch_add(1, Ordering::Relaxed);
        if self.force_capture_supported {
            CaptureSupport::Supported
        } else {
            self.inner.capture_support()
        }
    }
}

impl ExecutionProvider for HostDownloadCountingEp {
    fn consume_route_residency_at_boundary(&self) -> EpResult<()> {
        self.route_boundary_consumes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn finalize_executor_artifacts(
        &self,
        executor: ExecutorInstanceId,
        graph: &Graph,
        readiness: ExecutorArtifactReadinessEpoch,
        _finalized_banks: &[onnx_runtime_ep_api::FinalizedExpertBank],
    ) -> EpResult<ExecutorArtifactFinalization> {
        *self
            .route_readiness_checks
            .lock()
            .unwrap()
            .entry(executor)
            .or_default() += 1;
        if scoped_count(&self.route_terminal_outcomes, executor) > 0
            && !matches!(
                self.artifact_finalization,
                TestArtifactFinalization::ReadyPendingFailedReady
            )
        {
            return Ok(ExecutorArtifactFinalization::Complete);
        }
        assert!(
            self.kernel_compiles
                .lock()
                .unwrap()
                .get(&executor)
                .copied()
                .unwrap_or(0)
                > 0,
            "provider artifacts finalized before this executor compiled a kernel"
        );
        let attempt = {
            let mut attempts = self.route_finalizations.lock().unwrap();
            let attempt = attempts.entry(executor).or_default();
            *attempt += 1;
            *attempt
        };
        match self.artifact_finalization {
            TestArtifactFinalization::PendingOnce if attempt == 1 => Ok(
                ExecutorArtifactFinalization::Pending(ExecutorArtifactPending::ProviderReadiness {
                    reason: format!(
                        "test provider awaits a later compiled specialization after epoch {}",
                        readiness.get()
                    ),
                }),
            ),
            TestArtifactFinalization::FailOnce if attempt == 1 => {
                Err(EpError::KernelFailed(format!(
                    "injected artifact finalization failure at epoch {}",
                    readiness.get()
                )))
            }
            TestArtifactFinalization::ReadyPendingFailedReady if attempt == 2 => Ok(
                ExecutorArtifactFinalization::Pending(ExecutorArtifactPending::ProviderReadiness {
                    reason: format!(
                        "test provider awaits a later compiled specialization after epoch {}",
                        readiness.get()
                    ),
                }),
            ),
            TestArtifactFinalization::ReadyPendingFailedReady if attempt == 3 => {
                Err(EpError::KernelFailed(format!(
                    "injected artifact finalization failure at epoch {}",
                    readiness.get()
                )))
            }
            TestArtifactFinalization::Complete
            | TestArtifactFinalization::PendingOnce
            | TestArtifactFinalization::FailOnce
            | TestArtifactFinalization::StructuralDecline
            | TestArtifactFinalization::ReadyPendingFailedReady => {
                *self
                    .route_terminal_outcomes
                    .lock()
                    .unwrap()
                    .entry(executor)
                    .or_default() += 1;
                self.route_install_graph_nodes
                    .lock()
                    .unwrap()
                    .insert(executor, graph.num_nodes());
                Ok(ExecutorArtifactFinalization::Complete)
            }
        }
    }

    fn drain_executor_artifacts(&self, executor: ExecutorInstanceId) {
        *self
            .route_drains
            .lock()
            .unwrap()
            .entry(executor)
            .or_default() += 1;
    }

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

    fn get_kernel_for_executor(
        &self,
        executor: ExecutorInstanceId,
        op: &Node,
        shapes: &[Vec<usize>],
        opset: u64,
    ) -> EpResult<Box<dyn Kernel>> {
        let kernel = self.cpu.get_kernel(op, shapes, opset)?;
        *self
            .kernel_compiles
            .lock()
            .unwrap()
            .entry(executor)
            .or_default() += 1;
        if self.assert_finalized_before_execute {
            Ok(Box::new(FinalizationCheckingKernel {
                inner: kernel,
                executor,
                terminal_outcomes: Arc::clone(&self.route_terminal_outcomes),
                executions: Arc::clone(&self.kernel_executions),
                capture_checks: Arc::clone(&self.capture_checks),
                force_capture_supported: self.fake_device_graph,
            }))
        } else {
            Ok(kernel)
        }
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

    fn begin_device_graph_capture(&self, kernels: &[&dyn Kernel]) -> EpResult<()> {
        if !self.fake_device_graph {
            return Err(EpError::KernelFailed(
                "test provider device graph capture is disabled".to_string(),
            ));
        }
        assert!(
            !kernels.is_empty(),
            "capture must contain a compiled kernel"
        );
        assert!(
            !self.graph_capturing.swap(true, Ordering::SeqCst),
            "capture cannot begin twice"
        );
        Ok(())
    }

    fn end_device_graph_capture(&self) -> EpResult<()> {
        assert!(
            self.graph_capturing.swap(false, Ordering::SeqCst),
            "capture must be active before it ends"
        );
        self.graph_installed.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn abort_device_graph_capture(&self) -> EpResult<()> {
        self.graph_capturing.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn replay_device_graph(&self) -> EpResult<()> {
        assert!(
            self.graph_installed.load(Ordering::SeqCst),
            "fast replay requires an installed graph"
        );
        self.graph_fast_replays.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn replay_device_graph_segment(&self, index: usize) -> EpResult<()> {
        assert_eq!(index, 0, "single-graph capture installs one segment");
        assert!(
            self.graph_installed.load(Ordering::SeqCst),
            "segment replay requires an installed graph"
        );
        self.graph_segment_replays.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn reset_device_graph(&self) -> EpResult<bool> {
        Ok(self.graph_installed.swap(false, Ordering::SeqCst))
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
fn dynamic_slice_shape_propagates_directly_to_unsqueeze_and_comparison() {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);

    let data = input(&mut g, "data", DataType::Float32, &[4]);
    let one = i64_init(&mut g, "one", &[1], &[1]);
    let starts = i64_init(&mut g, "starts", &[1], &[0]);
    let slice_axes = i64_init(&mut g, "slice_axes", &[1], &[0]);
    let steps = i64_init(&mut g, "steps", &[1], &[1]);
    let unsqueeze_axes = i64_init(&mut g, "unsqueeze_axes", &[1], &[-1]);
    let thresholds = f32_init(&mut g, "thresholds", &[1, 2], &[1.5, 2.5]);
    let dynamic_extent = g.intern_symbol("direct_dynamic_extent");

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

    let compared = g.create_value(
        DataType::Bool,
        vec![Dim::Symbolic(dynamic_extent), Dim::Static(2)],
    );
    g.mark_value_shape_unknown(compared);
    g.insert_node(Node::new(
        NodeId(0),
        "Less",
        vec![Some(unsqueezed), Some(thresholds)],
        vec![compared],
    ));
    g.add_output(compared);

    let mut session = InferenceSession::from_graph(g).expect("build direct dynamic-shape chain");
    let data = Tensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let outputs = session
        .run(&[("data", &data)])
        .expect("run Slice -> Unsqueeze -> Less chain");

    assert_eq!(outputs[0].shape, vec![3, 2]);
    assert_eq!(outputs[0].dtype, DataType::Bool);
    assert_eq!(outputs[0].as_bytes(), &[1, 1, 0, 1, 0, 0]);
}

#[test]
fn dynamic_slice_shape_propagates_through_movement_and_broadcast_chain() {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);

    let data = input(&mut g, "data", DataType::Float32, &[4]);
    let one = i64_init(&mut g, "one", &[1], &[1]);
    let two = i64_init(&mut g, "two", &[1], &[2]);
    let starts = i64_init(&mut g, "starts", &[1], &[0]);
    let slice_axes = i64_init(&mut g, "slice_axes", &[1], &[0]);
    let steps = i64_init(&mut g, "steps", &[1], &[1]);
    let unsqueeze_axes = i64_init(&mut g, "unsqueeze_axes", &[1], &[-1]);
    let thresholds = f32_init(&mut g, "thresholds", &[1, 2], &[1.5, 2.5]);
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
    let expand_shape = g.create_value(DataType::Int64, static_shape([2]));
    let mut concat_shape = Node::new(
        NodeId(0),
        "Concat",
        vec![Some(end), Some(two)],
        vec![expand_shape],
    );
    concat_shape
        .attributes
        .insert("axis".into(), Attribute::Int(0));
    g.insert_node(concat_shape);

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

    let expanded = g.create_value(
        DataType::Float32,
        vec![Dim::Symbolic(dynamic_extent), Dim::Static(2)],
    );
    g.mark_value_shape_unknown(expanded);
    g.insert_node(Node::new(
        NodeId(0),
        "Expand",
        vec![Some(unsqueezed), Some(expand_shape)],
        vec![expanded],
    ));

    let reshape_shape = g.create_value(DataType::Int64, static_shape([2]));
    g.insert_node(Node::new(
        NodeId(0),
        "Shape",
        vec![Some(expanded)],
        vec![reshape_shape],
    ));
    let reshaped = g.create_value(
        DataType::Float32,
        vec![Dim::Symbolic(dynamic_extent), Dim::Static(2)],
    );
    g.mark_value_shape_unknown(reshaped);
    g.insert_node(Node::new(
        NodeId(0),
        "Reshape",
        vec![Some(expanded), Some(reshape_shape)],
        vec![reshaped],
    ));

    let concatenated_extent = g.intern_symbol("concatenated_extent");
    let concatenated = g.create_value(
        DataType::Float32,
        vec![Dim::Symbolic(concatenated_extent), Dim::Static(2)],
    );
    g.mark_value_shape_unknown(concatenated);
    let mut concat_data = Node::new(
        NodeId(0),
        "Concat",
        vec![Some(expanded), Some(reshaped)],
        vec![concatenated],
    );
    concat_data
        .attributes
        .insert("axis".into(), Attribute::Int(0));
    g.insert_node(concat_data);

    let compared = g.create_value(
        DataType::Bool,
        vec![Dim::Symbolic(concatenated_extent), Dim::Static(2)],
    );
    g.mark_value_shape_unknown(compared);
    g.insert_node(Node::new(
        NodeId(0),
        "Less",
        vec![Some(concatenated), Some(thresholds)],
        vec![compared],
    ));
    g.add_output(compared);

    let mut session = InferenceSession::from_graph(g).expect("build dynamic-shape session");
    let data = Tensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let outputs = session.run(&[("data", &data)]).expect("run dynamic chain");

    assert_eq!(outputs[0].shape, vec![6, 2]);
    assert_eq!(outputs[0].dtype, DataType::Bool);
    assert_eq!(outputs[0].as_bytes(), &[1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]);
}

#[test]
fn elementwise_output_uses_live_shapes_after_data_dependent_chain() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let source = input(&mut graph, "source", DataType::Float32, &[1, 4, 4]);
    let shape_seed = input(&mut graph, "shape_seed", DataType::Float32, &[1, 4]);
    let reduce_axes = i64_init(&mut graph, "reduce_axes", &[1], &[1]);
    let squeeze_axes = i64_init(&mut graph, "squeeze_axes", &[1], &[1]);
    let starts = i64_init(&mut graph, "starts", &[1], &[0]);
    let slice_axes = i64_init(&mut graph, "slice_axes", &[1], &[1]);
    let steps = i64_init(&mut graph, "steps", &[1], &[1]);

    let reduced = graph.create_named_value("reduced", DataType::Float32, static_shape([1, 1]));
    let mut reduce = Node::new(
        NodeId(0),
        "ReduceSum",
        vec![Some(shape_seed), Some(reduce_axes)],
        vec![reduced],
    );
    reduce
        .attributes
        .insert("keepdims".into(), Attribute::Int(1));
    graph.insert_node(reduce);

    let squeezed = graph.create_named_value("squeezed", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Squeeze",
        vec![Some(reduced), Some(squeeze_axes)],
        vec![squeezed],
    ));

    let casted = graph.create_named_value("casted", DataType::Int64, static_shape([1]));
    let mut cast = Node::new(NodeId(0), "Cast", vec![Some(squeezed)], vec![casted]);
    cast.attributes
        .insert("to".into(), Attribute::Int(DataType::Int64 as i64));
    graph.insert_node(cast);

    let sliced = graph.create_named_value("sliced", DataType::Float32, static_shape([1, 1, 4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Slice",
        vec![
            Some(source),
            Some(starts),
            Some(casted),
            Some(slice_axes),
            Some(steps),
        ],
        vec![sliced],
    ));

    let output = graph.create_named_value("output", DataType::Float32, static_shape([1, 1, 4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(sliced), Some(sliced)],
        vec![output],
    ));
    graph.add_output(output);

    let model = encode_model(&Model::new(&graph)).expect("encode stale-shape model");
    let mut session = InferenceSession::builder()
        .model_bytes(&model)
        .build()
        .expect("load stale-shape model");
    let source_data = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
    let source = Tensor::from_f32(&[1, 4, 4], &source_data).unwrap();
    let shape_seed = Tensor::from_f32(&[1, 4], &[1.0; 4]).unwrap();

    let outputs = session
        .run(&[("source", &source), ("shape_seed", &shape_seed)])
        .expect("live elementwise shapes must override stale loader output shapes");

    assert_eq!(outputs[0].shape, vec![1, 4, 4]);
    assert_close(
        &outputs[0].to_vec_f32(),
        &(1..=16).map(|value| (value * 2) as f32).collect::<Vec<_>>(),
    );
}

#[test]
fn live_elementwise_broadcast_mismatch_is_actionable() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let a_extent = graph.intern_symbol("a_extent");
    let b_extent = graph.intern_symbol("b_extent");
    let a = input_shaped(
        &mut graph,
        "a",
        DataType::Float32,
        vec![Dim::Static(1), Dim::Symbolic(a_extent), Dim::Static(4)],
    );
    let b = input_shaped(
        &mut graph,
        "b",
        DataType::Float32,
        vec![Dim::Static(1), Dim::Symbolic(b_extent), Dim::Static(4)],
    );
    let output = graph.create_named_value(
        "output",
        DataType::Float32,
        vec![Dim::Static(1), Dim::Symbolic(a_extent), Dim::Static(4)],
    );
    let mut add = Node::new(NodeId(0), "Add", vec![Some(a), Some(b)], vec![output]);
    add.name = "runtime_broadcast".into();
    graph.insert_node(add);
    graph.add_output(output);

    let mut session = InferenceSession::from_graph(graph).expect("build symbolic Add session");
    let a = Tensor::from_f32(&[1, 4, 4], &[1.0; 16]).unwrap();
    let b = Tensor::from_f32(&[1, 3, 4], &[1.0; 12]).unwrap();
    let error = session
        .run(&[("a", &a), ("b", &b)])
        .expect_err("incompatible live broadcast must fail");
    let message = error.to_string();

    assert!(
        matches!(error, SessionError::RuntimeBroadcastIncompatible { .. }),
        "unexpected error: {message}"
    );
    assert!(message.contains("runtime_broadcast"), "{message}");
    assert!(message.contains("[1, 4, 4]"), "{message}");
    assert!(message.contains("[1, 3, 4]"), "{message}");
    assert!(message.contains("equal or one of them is 1"), "{message}");
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
    assert_eq!(after_build.prebind_hits, 0);

    let x_tensor = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();

    let out1 = session.run(&[("X", &x_tensor)]).unwrap();
    let after_run1 = session.cache_stats();
    assert_eq!(after_run1.entries, 2, "no new entries on run");
    assert_eq!(after_run1.misses, 2, "no recompilation");
    // With kernel pre-binding, static-shape graphs serve lookups via the
    // zero-alloc pre-bound path (prebind_hits) instead of the HashMap path
    // (hits). Either way, each node is served from the cache.
    assert_eq!(
        after_run1.hits + after_run1.prebind_hits,
        2,
        "each node served from cache (via HashMap or pre-binding)"
    );

    let out2 = session.run(&[("X", &x_tensor)]).unwrap();
    let after_run2 = session.cache_stats();
    assert_eq!(after_run2.entries, 2);
    assert_eq!(after_run2.misses, 2);
    assert_eq!(
        after_run2.hits + after_run2.prebind_hits,
        4,
        "second run hit the cache again"
    );

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

    // batch = 2 → first shape: the CPU EP fuses MatMul+Add+Relu into one node.
    run_batch(&mut session, 2, 0.0);
    let s2 = session.cache_stats();
    assert_eq!(s2.entries, 1, "one fused node compiled for batch=2");
    assert_eq!(s2.misses, 1);
    assert_eq!(s2.hits, 0);

    // batch = 3 → new resolved shape: re-resolves + re-plans (1 more entry).
    run_batch(&mut session, 3, 10.0);
    let s3 = session.cache_stats();
    assert_eq!(
        s3.entries, 2,
        "batch=3 adds one distinct shape-keyed fused entry"
    );
    assert_eq!(s3.misses, 2);
    assert_eq!(s3.hits, 0);

    // batch = 2 again → the batch=2 plan is reused (cache hits, no new entries).
    run_batch(&mut session, 2, 100.0);
    let s2b = session.cache_stats();
    assert_eq!(s2b.entries, 2, "no new entries: batch=2 plan reused");
    assert_eq!(s2b.misses, 2);
    assert_eq!(s2b.hits, 1, "the fused node served from the batch=2 cache");
}

/// A shape-keyed kernel cache that never evicts turns "every request has a new
/// prompt length" into unbounded device memory, because each compiled kernel
/// owns its own workspaces (issue #1362). Feeding one node many distinct shapes
/// must therefore settle at the per-node bound instead of growing with the
/// number of distinct shapes seen.
#[test]
fn kernel_cache_bounds_the_shape_variants_it_keeps_per_node() {
    let w_data = [
        1.0f32, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];

    let mut g = Graph::new();
    let batch = g.intern_symbol("batch");
    let sym_row = || vec![Dim::Symbolic(batch), Dim::Static(4)];
    let x = input_shaped(&mut g, "X", DataType::Float32, sym_row());
    let w = f32_init(&mut g, "W", &[4, 4], &w_data);
    let m = op_shaped(&mut g, "MatMul", &[x, w], DataType::Float32, sym_row(), &[]);
    g.add_output(m);

    let mut session = InferenceSession::from_graph(g).expect("build symbolic session");

    // Far more distinct shapes than any per-node bound would keep.
    let shapes = 24;
    for rows in 1..=shapes {
        let data: Vec<f32> = (0..rows * 4).map(|i| i as f32).collect();
        let x_tensor = Tensor::from_f32(&[rows, 4], &data).unwrap();
        let out = session.run(&[("X", &x_tensor)]).expect("run");
        assert_eq!(out[0].shape, vec![rows, 4]);
        // Identity weights: the output must still be the input, so eviction is
        // shown to cost recompilation and nothing else.
        assert_close(&out[0].to_vec_f32(), &data);
    }

    let stats = session.cache_stats();
    assert!(
        stats.evictions > 0,
        "the bound must have evicted surplus variants, saw {stats:?}"
    );
    assert!(
        stats.entries < shapes,
        "the cache must not keep one entry per distinct shape, saw {stats:?}"
    );
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

// ---------------------------------------------------------------------------
// #1810 Slice 7E — executor-scoped provider-artifact lifecycle.
// ---------------------------------------------------------------------------

fn static_gelu_model() -> Vec<u8> {
    encode_model(&Model::new(&standard_gelu_graph(20))).expect("encode static Gelu model")
}

fn symbolic_gelu_model() -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 20);
    let extent = graph.intern_symbol("dynamic_extent");
    let shape = vec![Dim::Symbolic(extent)];
    let x = input_shaped(&mut graph, "x", DataType::Float32, shape.clone());
    let y = graph.create_named_value("y", DataType::Float32, shape);
    graph.insert_node(Node::new(NodeId(0), "Gelu", vec![Some(x)], vec![y]));
    graph.add_output(y);
    encode_model(&Model::new(&graph)).expect("encode symbolic Gelu model")
}

fn symbolic_matmul_model() -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 20);
    let rows = graph.intern_symbol("rows");
    let x = input_shaped(
        &mut graph,
        "x",
        DataType::Float32,
        vec![Dim::Symbolic(rows), Dim::Static(2)],
    );
    let weight = f32_init(
        &mut graph,
        "weight",
        &[2, 2],
        &[
            1.0, 0.0, //
            0.0, 1.0,
        ],
    );
    let y = graph.create_named_value(
        "y",
        DataType::Float32,
        vec![Dim::Symbolic(rows), Dim::Static(2)],
    );
    graph.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(x), Some(weight)],
        vec![y],
    ));
    graph.add_output(y);
    encode_model(&Model::new(&graph)).expect("encode symbolic MatMul model")
}

fn runtime_dispatch_specialization_model() -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 20);
    let elements = graph.intern_symbol("elements");
    let data = input_shaped(
        &mut graph,
        "data",
        DataType::Float32,
        vec![Dim::Symbolic(elements)],
    );
    let shape = input(&mut graph, "shape", DataType::Int64, &[2]);
    let rows = graph.intern_symbol("runtime_rows");
    let columns = graph.intern_symbol("runtime_columns");
    let reshaped = graph.create_named_value(
        "reshaped",
        DataType::Float32,
        vec![Dim::Symbolic(rows), Dim::Symbolic(columns)],
    );
    graph.mark_value_shape_unknown(reshaped);
    graph.insert_node(Node::new(
        NodeId(0),
        "Reshape",
        vec![Some(data), Some(shape)],
        vec![reshaped],
    ));
    let bias = f32_init(&mut graph, "bias", &[1], &[0.5]);
    let y = graph.create_named_value(
        "y",
        DataType::Float32,
        vec![Dim::Symbolic(rows), Dim::Symbolic(columns)],
    );
    graph.mark_value_shape_unknown(y);
    graph.insert_node(Node::new(
        NodeId(1),
        "Add",
        vec![Some(reshaped), Some(bias)],
        vec![y],
    ));
    graph.add_output(y);
    encode_model(&Model::new(&graph)).expect("encode runtime-dispatch specialization model")
}

fn scoped_count(
    counts: &Mutex<HashMap<ExecutorInstanceId, usize>>,
    executor: ExecutorInstanceId,
) -> usize {
    counts.lock().unwrap().get(&executor).copied().unwrap_or(0)
}

fn total_count(counts: &Mutex<HashMap<ExecutorInstanceId, usize>>) -> usize {
    counts.lock().unwrap().values().sum()
}

fn assert_artifacts_pending(
    error: SessionError,
    executor: ExecutorInstanceId,
    readiness_epoch: u64,
) {
    match error {
        SessionError::ExecutionProviderArtifactsPending {
            executor: actual_executor,
            readiness_epoch: actual_epoch,
            ..
        } => {
            assert_eq!(actual_executor, executor.get());
            assert_eq!(actual_epoch, readiness_epoch);
        }
        other => panic!("expected typed provider-artifact pending error, got {other}"),
    }
}

#[test]
fn static_build_finalizes_provider_artifacts_once_and_drains_owner_once() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = HostDownloadCountingEp::new_lifecycle(Arc::clone(&downloads));
    let compiles = ep.kernel_compiles();
    let readiness_checks = ep.route_readiness_checks();
    let finalizations = ep.route_finalizations();
    let drains = ep.route_drains();
    let install_nodes = ep.route_install_graph_nodes();
    let ep = Arc::new(ep);

    let model = static_gelu_model();
    let mut session = InferenceSession::builder()
        .model_bytes(&model)
        .execution_provider(ep)
        .build()
        .expect("build static Gelu session");
    let executor = session.executor_instance_id();

    assert_eq!(
        scoped_count(&compiles, executor),
        1,
        "static build compiles its kernel before finalization"
    );
    assert_eq!(
        scoped_count(&finalizations, executor),
        1,
        "static build finalizes exactly once"
    );
    assert_eq!(scoped_count(&readiness_checks, executor), 1);
    assert_eq!(
        scoped_count(&install_nodes, executor),
        1,
        "finalization receives the finalized graph"
    );
    assert_eq!(scoped_count(&drains, executor), 0);

    let x = Tensor::from_f32(&[2], &[-1.0, 1.0]).unwrap();
    session.run(&[("x", &x)]).expect("run static Gelu");
    assert_eq!(scoped_count(&finalizations, executor), 1);
    assert_eq!(
        scoped_count(&readiness_checks, executor),
        1,
        "same static specialization needs no new provider readiness check"
    );

    drop(session);
    assert_eq!(scoped_count(&drains, executor), 1);
    assert_eq!(scoped_count(&finalizations, executor), 1);
}

#[test]
fn static_build_pending_fails_closed_and_drains_unpublished_executor() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = HostDownloadCountingEp::new_pending_once_lifecycle(downloads);
    let compiles = ep.kernel_compiles();
    let finalizations = ep.route_finalizations();
    let terminal_outcomes = ep.route_terminal_outcomes();
    let executions = ep.kernel_executions();
    let drains = ep.route_drains();
    let ep = Arc::new(ep);

    let error = match InferenceSession::builder()
        .model_bytes(&static_gelu_model())
        .execution_provider(ep)
        .build()
    {
        Err(error) => error,
        Ok(_) => panic!("static build must not publish a pending executor"),
    };
    assert!(matches!(
        error,
        SessionError::ExecutionProviderArtifactsPending {
            readiness_epoch: 1,
            ..
        }
    ));
    assert_eq!(total_count(&compiles), 1);
    assert_eq!(total_count(&finalizations), 1);
    assert_eq!(total_count(&terminal_outcomes), 0);
    assert_eq!(total_count(&executions), 0);
    assert_eq!(
        total_count(&drains),
        1,
        "failed static build drains exactly its unpublished executor scope"
    );
}

#[test]
fn symbolic_first_resolved_compile_finalizes_before_use_and_specialization_does_not_reinstall() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = HostDownloadCountingEp::new_lifecycle(downloads);
    let compiles = ep.kernel_compiles();
    let readiness_checks = ep.route_readiness_checks();
    let finalizations = ep.route_finalizations();
    let drains = ep.route_drains();
    let capture_checks = ep.capture_checks();
    let ep = Arc::new(ep);

    let model = symbolic_gelu_model();
    let mut session = InferenceSession::builder()
        .model_bytes(&model)
        .execution_provider(ep)
        .build()
        .expect("build symbolic Gelu session");
    let executor = session.executor_instance_id();
    assert_eq!(scoped_count(&compiles, executor), 0);
    assert_eq!(
        scoped_count(&finalizations, executor),
        0,
        "symbolic build must not falsely finalize before resolved compilation"
    );

    let x2 = Tensor::from_f32(&[2], &[-1.0, 1.0]).unwrap();
    session.run(&[("x", &x2)]).expect("first symbolic run");
    assert_eq!(scoped_count(&compiles, executor), 1);
    assert_eq!(scoped_count(&readiness_checks, executor), 1);
    assert_eq!(scoped_count(&finalizations, executor), 1);
    let mut binding = session
        .allocate_device_binding("x", Some("y"), DataType::Float32, vec![2], vec![2])
        .expect("allocate capture audit binding");
    binding
        .write_bytes(0, &f32_bytes(&[-1.0, 1.0]))
        .expect("seed capture audit binding");
    let _ = session
        .try_capture_with_device_bindings(&[], std::slice::from_mut(&mut binding))
        .expect("capture audit must run after finalization");
    assert!(
        capture_checks.load(Ordering::Relaxed) > 0,
        "capture audit must inspect the finalized compiled kernel"
    );

    let x4 = Tensor::from_f32(&[4], &[-1.0, 0.0, 1.0, 2.0]).unwrap();
    session
        .run(&[("x", &x4)])
        .expect("dynamic specialization run");
    assert_eq!(
        scoped_count(&compiles, executor),
        2,
        "new input shape compiles one new kernel specialization"
    );
    assert_eq!(
        scoped_count(&finalizations, executor),
        1,
        "stable executor artifacts make dynamic specialization reinstall-free"
    );
    assert_eq!(
        scoped_count(&readiness_checks, executor),
        2,
        "a new compiled specialization re-confirms the executor terminal outcome"
    );

    drop(session);
    assert_eq!(scoped_count(&drains, executor), 1);
}

#[test]
fn pending_blocks_execution_and_capture_until_a_new_compilation_epoch_finalizes_once() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = HostDownloadCountingEp::new_pending_once_lifecycle(downloads);
    let readiness_checks = ep.route_readiness_checks();
    let finalizations = ep.route_finalizations();
    let terminal_outcomes = ep.route_terminal_outcomes();
    let executions = ep.kernel_executions();
    let capture_checks = ep.capture_checks();
    let drains = ep.route_drains();
    let ep = Arc::new(ep);

    let mut session = InferenceSession::builder()
        .model_bytes(&symbolic_gelu_model())
        .execution_provider(ep)
        .build()
        .expect("build pending lifecycle session");
    let executor = session.executor_instance_id();
    let x2 = Tensor::from_f32(&[2], &[-1.0, 1.0]).unwrap();
    let error = session
        .run(&[("x", &x2)])
        .expect_err("first readiness epoch must remain pending");
    assert_artifacts_pending(error, executor, 1);
    assert_eq!(scoped_count(&finalizations, executor), 1);
    assert_eq!(scoped_count(&readiness_checks, executor), 1);
    assert_eq!(scoped_count(&terminal_outcomes, executor), 0);
    assert_eq!(scoped_count(&executions, executor), 0);
    assert_eq!(capture_checks.load(Ordering::Relaxed), 0);

    let mut binding = session
        .allocate_device_binding("x", Some("y"), DataType::Float32, vec![2], vec![2])
        .expect("allocate pending capture binding");
    binding
        .write_bytes(0, &f32_bytes(&[-1.0, 1.0]))
        .expect("seed pending capture binding");
    let error =
        match session.try_capture_with_device_bindings(&[], std::slice::from_mut(&mut binding)) {
            Err(error) => error,
            Ok(_) => panic!("capture must not begin while artifacts are pending"),
        };
    assert_artifacts_pending(error, executor, 1);
    assert_eq!(
        scoped_count(&finalizations, executor),
        1,
        "the same readiness epoch must not busy-retry finalization"
    );
    assert_eq!(scoped_count(&readiness_checks, executor), 1);
    assert_eq!(scoped_count(&executions, executor), 0);
    assert_eq!(capture_checks.load(Ordering::Relaxed), 0);

    let error = session
        .run(&[("x", &x2)])
        .expect_err("same-shape eager retry must remain pending");
    assert_artifacts_pending(error, executor, 1);
    assert_eq!(scoped_count(&finalizations, executor), 1);
    assert_eq!(scoped_count(&executions, executor), 0);

    let x4 = Tensor::from_f32(&[4], &[-1.0, 0.0, 1.0, 2.0]).unwrap();
    session
        .run(&[("x", &x4)])
        .expect("new specialization advances readiness and finalizes");
    assert_eq!(scoped_count(&finalizations, executor), 2);
    assert_eq!(scoped_count(&readiness_checks, executor), 2);
    assert_eq!(
        scoped_count(&terminal_outcomes, executor),
        1,
        "provider installation reaches one terminal outcome"
    );
    assert_eq!(scoped_count(&executions, executor), 1);

    session
        .run(&[("x", &x2)])
        .expect("completed executor reuses the installed producer identity");
    assert_eq!(
        scoped_count(&finalizations, executor),
        2,
        "later specializations must not reinstall executor-owned artifacts"
    );
    assert_eq!(scoped_count(&readiness_checks, executor), 2);
    assert_eq!(scoped_count(&terminal_outcomes, executor), 1);
    assert_eq!(scoped_count(&executions, executor), 2);

    drop(session);
    assert_eq!(scoped_count(&drains, executor), 1);
}

#[test]
fn permanent_structural_decline_is_terminal_before_execution() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = HostDownloadCountingEp::new_structural_decline_lifecycle(downloads);
    let finalizations = ep.route_finalizations();
    let terminal_outcomes = ep.route_terminal_outcomes();
    let executions = ep.kernel_executions();
    let ep = Arc::new(ep);
    let mut session = InferenceSession::builder()
        .model_bytes(&symbolic_gelu_model())
        .execution_provider(ep)
        .build()
        .expect("build structural-decline lifecycle session");
    let executor = session.executor_instance_id();

    let x = Tensor::from_f32(&[2], &[-1.0, 1.0]).unwrap();
    session
        .run(&[("x", &x)])
        .expect("honest terminal structural decline permits execution");
    assert_eq!(scoped_count(&finalizations, executor), 1);
    assert_eq!(scoped_count(&terminal_outcomes, executor), 1);
    assert_eq!(scoped_count(&executions, executor), 1);

    let x = Tensor::from_f32(&[4], &[-1.0, 0.0, 1.0, 2.0]).unwrap();
    session
        .run(&[("x", &x)])
        .expect("later specialization reuses terminal decline");
    assert_eq!(scoped_count(&finalizations, executor), 1);
    assert_eq!(scoped_count(&executions, executor), 2);
}

#[test]
fn failed_finalization_is_cached_until_readiness_advances() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = HostDownloadCountingEp::new_fail_once_lifecycle(downloads);
    let finalizations = ep.route_finalizations();
    let terminal_outcomes = ep.route_terminal_outcomes();
    let executions = ep.kernel_executions();
    let ep = Arc::new(ep);
    let mut session = InferenceSession::builder()
        .model_bytes(&symbolic_gelu_model())
        .execution_provider(ep)
        .build()
        .expect("build failed-finalization lifecycle session");
    let executor = session.executor_instance_id();

    let x2 = Tensor::from_f32(&[2], &[-1.0, 1.0]).unwrap();
    for _ in 0..2 {
        let error = session
            .run(&[("x", &x2)])
            .expect_err("failed readiness epoch must not execute");
        assert!(matches!(
            error,
            SessionError::ExecutionProviderArtifactFinalizationFailed {
                readiness_epoch: 1,
                ..
            }
        ));
    }
    assert_eq!(
        scoped_count(&finalizations, executor),
        1,
        "same failed epoch must not re-enter the provider"
    );
    assert_eq!(scoped_count(&terminal_outcomes, executor), 0);
    assert_eq!(scoped_count(&executions, executor), 0);

    let x4 = Tensor::from_f32(&[4], &[-1.0, 0.0, 1.0, 2.0]).unwrap();
    session
        .run(&[("x", &x4)])
        .expect("new compilation epoch retries failed finalization");
    assert_eq!(scoped_count(&finalizations, executor), 2);
    assert_eq!(scoped_count(&terminal_outcomes, executor), 1);
    assert_eq!(scoped_count(&executions, executor), 1);
}

#[test]
fn single_graph_fast_replay_obeys_pending_failed_and_ready_specializations() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = HostDownloadCountingEp::new_fast_replay_lifecycle(downloads);
    let finalizations = ep.route_finalizations();
    let terminal_outcomes = ep.route_terminal_outcomes();
    let boundary_consumes = ep.route_boundary_consumes();
    let executions = ep.kernel_executions();
    let drains = ep.route_drains();
    let segment_replays = ep.graph_segment_replays();
    let fast_replays = ep.graph_fast_replays();
    let ep = Arc::new(ep);
    let mut session = InferenceSession::builder()
        .model_bytes(&symbolic_gelu_model())
        .execution_provider(ep)
        .build()
        .expect("build fast-replay lifecycle session");
    let executor = session.executor_instance_id();

    let input = session
        .allocate_device_binding("x", None::<String>, DataType::Float32, vec![2], vec![2])
        .expect("allocate persistent capture input");
    let output = session
        .allocate_device_output_binding("y", DataType::Float32, vec![2], vec![2])
        .expect("allocate persistent capture output");
    let mut bindings = vec![input, output];
    bindings[0]
        .write_bytes(0, &f32_bytes(&[-1.0, 1.0]))
        .expect("seed persistent capture input");
    assert!(matches!(
        session
            .try_capture_with_device_bindings(&[], &mut bindings)
            .expect("first specialization reaches a terminal outcome and captures"),
        DeviceGraphCaptureResult::Captured(_)
    ));
    assert_eq!(session.captured_graph_segment_count(), 1);
    assert_eq!(scoped_count(&finalizations, executor), 1);
    assert_eq!(scoped_count(&terminal_outcomes, executor), 1);
    assert_eq!(scoped_count(&executions, executor), 1);
    assert_eq!(segment_replays.load(Ordering::SeqCst), 1);
    assert_eq!(fast_replays.load(Ordering::SeqCst), 0);

    let x4 = Tensor::from_f32(&[4], &[-1.0, 0.0, 1.0, 2.0]).unwrap();
    let error = session
        .run(&[("x", &x4)])
        .expect_err("second specialization remains pending");
    assert_artifacts_pending(error, executor, 2);
    assert_eq!(scoped_count(&executions, executor), 1);
    let error = session
        .replay_device_graph(&mut bindings)
        .expect_err("fast replay must not bypass pending readiness");
    assert_artifacts_pending(error, executor, 2);
    assert_eq!(
        scoped_count(&finalizations, executor),
        2,
        "fast replay must not busy-retry a pending epoch"
    );
    assert_eq!(fast_replays.load(Ordering::SeqCst), 0);

    let x8 = Tensor::from_f32(&[8], &[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let error = session
        .run(&[("x", &x8)])
        .expect_err("third specialization fails finalization");
    assert!(matches!(
        error,
        SessionError::ExecutionProviderArtifactFinalizationFailed {
            executor: actual_executor,
            readiness_epoch: 3,
            ..
        } if actual_executor == executor.get()
    ));
    let error = session
        .replay_device_graph(&mut bindings)
        .expect_err("fast replay must not bypass failed readiness");
    assert!(matches!(
        error,
        SessionError::ExecutionProviderArtifactFinalizationFailed {
            executor: actual_executor,
            readiness_epoch: 3,
            ..
        } if actual_executor == executor.get()
    ));
    assert_eq!(
        scoped_count(&finalizations, executor),
        3,
        "fast replay must not re-enter a failed epoch"
    );
    assert_eq!(scoped_count(&executions, executor), 1);
    assert_eq!(fast_replays.load(Ordering::SeqCst), 0);

    let x16 = Tensor::from_f32(
        &[16],
        &[
            -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
        ],
    )
    .unwrap();
    session
        .run(&[("x", &x16)])
        .expect("fourth specialization reaches a terminal outcome");
    assert_eq!(scoped_count(&finalizations, executor), 4);
    assert_eq!(scoped_count(&terminal_outcomes, executor), 2);
    assert_eq!(scoped_count(&executions, executor), 2);
    let boundaries_before_fast_replay = boundary_consumes.load(Ordering::SeqCst);
    assert!(
        session
            .replay_device_graph(&mut bindings)
            .expect("ready authority permits the installed fast replay")
    );
    assert_eq!(fast_replays.load(Ordering::SeqCst), 1);
    assert_eq!(
        boundary_consumes.load(Ordering::SeqCst),
        boundaries_before_fast_replay + 1,
        "single-graph fast replay must traverse the production request boundary"
    );

    drop(session);
    assert_eq!(scoped_count(&drains, executor), 1);
}

#[test]
fn binding_preparation_cache_misses_revoke_fast_replay_until_terminal() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = HostDownloadCountingEp::new_fast_replay_lifecycle(downloads);
    let compiles = ep.kernel_compiles();
    let finalizations = ep.route_finalizations();
    let terminal_outcomes = ep.route_terminal_outcomes();
    let executions = ep.kernel_executions();
    let fast_replays = ep.graph_fast_replays();
    let ep = Arc::new(ep);
    let mut session = InferenceSession::builder()
        .model_bytes(&symbolic_matmul_model())
        .execution_provider(ep)
        .build()
        .expect("build binding-preparation lifecycle session");
    let executor = session.executor_instance_id();

    let input = session
        .allocate_device_binding(
            "x",
            None::<String>,
            DataType::Float32,
            vec![1, 2],
            vec![1, 2],
        )
        .expect("allocate captured MatMul input");
    let output = session
        .allocate_device_output_binding("y", DataType::Float32, vec![1, 2], vec![1, 2])
        .expect("allocate captured MatMul output");
    let mut captured_bindings = vec![input, output];
    captured_bindings[0]
        .write_bytes(0, &f32_bytes(&[1.0, 2.0]))
        .expect("seed captured MatMul input");
    assert!(matches!(
        session
            .try_capture_with_device_bindings(&[], &mut captured_bindings)
            .expect("initial specialization finalizes and captures"),
        DeviceGraphCaptureResult::Captured(_)
    ));
    assert_eq!(scoped_count(&compiles, executor), 1);
    assert_eq!(scoped_count(&finalizations, executor), 1);
    assert_eq!(scoped_count(&executions, executor), 1);

    let mut prepared_binding_sets = Vec::new();
    for (rows, expected_epoch, expected_attempt) in [(2, 2, 2), (3, 3, 3), (4, 4, 4)] {
        let input = session
            .allocate_device_binding(
                "x",
                None::<String>,
                DataType::Float32,
                vec![rows, 2],
                vec![rows, 2],
            )
            .expect("allocate prepared MatMul input");
        let output = session
            .allocate_device_output_binding("y", DataType::Float32, vec![rows, 2], vec![rows, 2])
            .expect("allocate prepared MatMul output");
        prepared_binding_sets.push(vec![input, output]);
        session
            .prepare_with_device_bindings(
                &[],
                prepared_binding_sets
                    .last_mut()
                    .expect("just pushed prepared bindings"),
            )
            .expect("binding preparation publishes a new specialization");
        assert_eq!(scoped_count(&compiles, executor), expected_epoch);

        let result = session.replay_device_graph(&mut captured_bindings);
        match expected_attempt {
            2 => assert_artifacts_pending(
                result.expect_err("prepared specialization keeps replay pending"),
                executor,
                expected_epoch as u64,
            ),
            3 => assert!(matches!(
                result.expect_err("prepared specialization makes replay fail closed"),
                SessionError::ExecutionProviderArtifactFinalizationFailed {
                    executor: actual_executor,
                    readiness_epoch,
                    ..
                } if actual_executor == executor.get()
                    && readiness_epoch == expected_epoch as u64
            )),
            4 => assert!(result.expect("ready prepared specialization permits replay")),
            _ => unreachable!(),
        }
        assert_eq!(scoped_count(&finalizations, executor), expected_attempt);
        assert_eq!(
            scoped_count(&executions, executor),
            1,
            "binding preparation and readiness checks must not execute kernels"
        );
    }
    assert_eq!(scoped_count(&terminal_outcomes, executor), 2);
    assert_eq!(fast_replays.load(Ordering::SeqCst), 1);
}

#[test]
fn runtime_dispatch_cache_miss_finalizes_before_kernel_use() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = HostDownloadCountingEp::new_fast_replay_lifecycle(downloads);
    let compiles = ep.kernel_compiles();
    let finalizations = ep.route_finalizations();
    let terminal_outcomes = ep.route_terminal_outcomes();
    let executions = ep.kernel_executions();
    let ep = Arc::new(ep);
    let mut session = InferenceSession::builder()
        .model_bytes(&runtime_dispatch_specialization_model())
        .execution_provider(ep)
        .build()
        .expect("build runtime-dispatch lifecycle session");
    let executor = session.executor_instance_id();

    let data2 = Tensor::from_f32(&[2], &[-1.0, 1.0]).unwrap();
    let shape12 = i64_tensor(&[2], &[1, 2]);
    let error = session
        .run(&[("data", &data2), ("shape", &shape12)])
        .expect_err("runtime Add specialization remains pending");
    assert_artifacts_pending(error, executor, 2);
    assert_eq!(
        scoped_count(&compiles, executor),
        2,
        "preflight compiles Reshape and runtime dispatch compiles Add"
    );
    assert_eq!(scoped_count(&finalizations, executor), 2);
    assert_eq!(
        scoped_count(&executions, executor),
        0,
        "the Reshape is a view and pending blocks the runtime-compiled Add"
    );

    let error = session
        .run(&[("data", &data2), ("shape", &shape12)])
        .expect_err("same runtime specialization remains latched pending");
    assert_artifacts_pending(error, executor, 2);
    assert_eq!(
        scoped_count(&finalizations, executor),
        2,
        "pending epoch must not busy-retry"
    );
    assert_eq!(scoped_count(&executions, executor), 0);

    let data4 = Tensor::from_f32(&[4], &[-1.0, 0.0, 1.0, 2.0]).unwrap();
    let shape22 = i64_tensor(&[2], &[2, 2]);
    let error = session
        .run(&[("data", &data4), ("shape", &shape22)])
        .expect_err("new preflight specialization reaches the injected failure");
    assert!(matches!(
        error,
        SessionError::ExecutionProviderArtifactFinalizationFailed {
            executor: actual_executor,
            readiness_epoch: 3,
            ..
        } if actual_executor == executor.get()
    ));
    assert_eq!(scoped_count(&compiles, executor), 3);
    assert_eq!(scoped_count(&finalizations, executor), 3);
    assert_eq!(scoped_count(&executions, executor), 0);

    let error = session
        .run(&[("data", &data4), ("shape", &shape22)])
        .expect_err("same preflight specialization remains latched failed");
    assert!(matches!(
        error,
        SessionError::ExecutionProviderArtifactFinalizationFailed {
            readiness_epoch: 3,
            ..
        }
    ));
    assert_eq!(
        scoped_count(&finalizations, executor),
        3,
        "failed epoch must not busy-retry"
    );

    let data6 = Tensor::from_f32(&[6], &[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let shape32 = i64_tensor(&[2], &[3, 2]);
    let outputs = session
        .run(&[("data", &data6), ("shape", &shape32)])
        .expect("later preflight and runtime specializations both finalize");
    assert_eq!(outputs[0].shape, vec![3, 2]);
    assert_eq!(scoped_count(&compiles, executor), 5);
    assert_eq!(scoped_count(&finalizations, executor), 5);
    assert_eq!(scoped_count(&terminal_outcomes, executor), 3);
    assert_eq!(scoped_count(&executions, executor), 1);
}

#[test]
fn cancellation_while_pending_drains_without_execution() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = HostDownloadCountingEp::new_pending_once_lifecycle(downloads);
    let finalizations = ep.route_finalizations();
    let terminal_outcomes = ep.route_terminal_outcomes();
    let executions = ep.kernel_executions();
    let drains = ep.route_drains();
    let ep = Arc::new(ep);
    let mut session = InferenceSession::builder()
        .model_bytes(&symbolic_gelu_model())
        .execution_provider(ep)
        .build()
        .expect("build cancellable pending session");
    let executor = session.executor_instance_id();
    let x = Tensor::from_f32(&[2], &[-1.0, 1.0]).unwrap();
    let error = session
        .run(&[("x", &x)])
        .expect_err("run remains pending before cancellation");
    assert_artifacts_pending(error, executor, 1);

    drop(session);
    assert_eq!(scoped_count(&finalizations, executor), 1);
    assert_eq!(scoped_count(&terminal_outcomes, executor), 0);
    assert_eq!(scoped_count(&executions, executor), 0);
    assert_eq!(scoped_count(&drains, executor), 1);
}

#[test]
fn shared_provider_concurrent_pending_executors_finalize_and_drain_independently() {
    let downloads = Arc::new(AtomicUsize::new(0));
    let ep = Arc::new(HostDownloadCountingEp::new_pending_once_lifecycle(
        downloads,
    ));
    let finalizations = ep.route_finalizations();
    let terminal_outcomes = ep.route_terminal_outcomes();
    let executions = ep.kernel_executions();
    let drains = ep.route_drains();
    let model = symbolic_gelu_model();

    let mut first = InferenceSession::builder()
        .model_bytes(&model)
        .execution_provider(ep.clone())
        .build()
        .expect("build first symbolic executor");
    let mut second = InferenceSession::builder()
        .model_bytes(&model)
        .execution_provider(ep)
        .build()
        .expect("build second symbolic executor");
    let first_id = first.executor_instance_id();
    let second_id = second.executor_instance_id();
    assert_ne!(first_id, second_id);

    std::thread::scope(|scope| {
        scope.spawn(|| {
            let x = Tensor::from_f32(&[2], &[-1.0, 1.0]).unwrap();
            first
                .run(&[("x", &x)])
                .expect_err("first executor initially pending");
        });
        scope.spawn(|| {
            let x = Tensor::from_f32(&[4], &[-1.0, 0.0, 1.0, 2.0]).unwrap();
            second
                .run(&[("x", &x)])
                .expect_err("second executor initially pending");
        });
    });
    assert_eq!(scoped_count(&finalizations, first_id), 1);
    assert_eq!(scoped_count(&finalizations, second_id), 1);
    assert_eq!(scoped_count(&terminal_outcomes, first_id), 0);
    assert_eq!(scoped_count(&terminal_outcomes, second_id), 0);
    assert_eq!(scoped_count(&executions, first_id), 0);
    assert_eq!(scoped_count(&executions, second_id), 0);

    std::thread::scope(|scope| {
        scope.spawn(|| {
            let x = Tensor::from_f32(&[4], &[-1.0, 0.0, 1.0, 2.0]).unwrap();
            first
                .run(&[("x", &x)])
                .expect("first executor finalizes independently");
        });
        scope.spawn(|| {
            let x = Tensor::from_f32(&[8], &[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
            second
                .run(&[("x", &x)])
                .expect("second executor finalizes independently");
        });
    });
    assert_eq!(scoped_count(&finalizations, first_id), 2);
    assert_eq!(scoped_count(&finalizations, second_id), 2);
    assert_eq!(scoped_count(&terminal_outcomes, first_id), 1);
    assert_eq!(scoped_count(&terminal_outcomes, second_id), 1);
    assert_eq!(scoped_count(&executions, first_id), 1);
    assert_eq!(scoped_count(&executions, second_id), 1);

    drop(first);
    assert_eq!(scoped_count(&drains, first_id), 1);
    assert_eq!(
        scoped_count(&drains, second_id),
        0,
        "dropping one executor must not drain its sibling"
    );
    let x = Tensor::from_f32(&[4], &[2.0, 1.0, 0.0, -1.0]).unwrap();
    second
        .run(&[("x", &x)])
        .expect("surviving sibling still runs");
    assert_eq!(scoped_count(&finalizations, second_id), 2);
    assert_eq!(scoped_count(&terminal_outcomes, second_id), 1);
    drop(second);
    assert_eq!(scoped_count(&drains, second_id), 1);
}

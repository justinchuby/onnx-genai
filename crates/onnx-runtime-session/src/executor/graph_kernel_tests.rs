use super::*;
use onnx_runtime_ep_api::{EpConfig, Fence, KernelSizedOutput};
use onnx_runtime_ir::{pattern::Pattern, static_shape};
use std::sync::atomic::{AtomicUsize, Ordering};

struct PatternGraphEp {
    cpu: CpuExecutionProvider,
    claims: AtomicUsize,
    compiles: AtomicUsize,
    executions: Arc<AtomicUsize>,
    malformed: bool,
    claim_override: Option<Vec<NodeId>>,
    compile_failure: bool,
}
impl PatternGraphEp {
    fn new(malformed: bool) -> Self {
        let mut cpu = CpuExecutionProvider::new();
        cpu.initialize(&EpConfig::default()).unwrap();
        Self {
            cpu,
            claims: AtomicUsize::new(0),
            compiles: AtomicUsize::new(0),
            executions: Arc::new(AtomicUsize::new(0)),
            malformed,
            claim_override: None,
            compile_failure: false,
        }
    }
}

fn if_pattern() -> Pattern {
    let mut then_branch = Pattern::builder();
    let x = then_branch.input("x");
    let value = then_branch.op("Identity", [x]);
    let then_branch = then_branch.finish_body([value]).unwrap();
    let mut else_branch = Pattern::builder();
    let x = else_branch.input("x");
    let value = else_branch
        .node("Concat", [x, x])
        .attr("axis", 0i64)
        .output();
    let else_branch = else_branch.finish_body([value]).unwrap();
    let mut p = Pattern::builder();
    let cond = p.input("condition");
    let out = p
        .node("If", [cond])
        .attr("then_branch", then_branch)
        .attr("else_branch", else_branch)
        .output();
    p.capture("out", out);
    p.finish(out).unwrap()
}

struct IfPatternKernel {
    captured: usize,
    executions: Arc<AtomicUsize>,
    malformed: bool,
}
impl Kernel for IfPatternKernel {
    fn execute(
        &self,
        _inputs: &[TensorView],
        _outputs: &mut [TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        Err(EpError::KernelFailed(
            "If pattern requires its dynamic-output entrypoint".into(),
        ))
    }
    fn has_kernel_sized_outputs(&self) -> bool {
        true
    }
    fn execute_kernel_sized(
        &self,
        inputs: &[TensorView],
        requested: &[bool],
    ) -> onnx_runtime_ep_api::Result<Vec<Option<KernelSizedOutput>>> {
        assert_eq!(requested, [true]);
        for input in inputs {
            input.validate()?;
            assert!(input.device.is_host_accessible());
            assert!(input.is_contiguous());
        }
        assert_eq!(inputs[0].dtype, DataType::Bool);
        assert_eq!(inputs[0].numel(), 1);
        assert_eq!(inputs[self.captured].dtype, DataType::Float32);
        // SAFETY: validated live host views; the executor checked the backing
        // bounds and materialized unsupported strides before calling this kernel.
        let condition = unsafe { *inputs[0].data_ptr::<u8>() } != 0;
        let input = &inputs[self.captured];
        let bytes =
            unsafe { std::slice::from_raw_parts(input.data_ptr::<u8>(), input.byte_size()) };
        let mut shape = input.shape.to_vec();
        if shape.is_empty() {
            return Err(EpError::KernelFailed(
                "Concat pattern capture requires rank at least one".into(),
            ));
        }
        let mut result = bytes.to_vec();
        if !condition {
            shape[0] *= 2;
            result.extend_from_slice(bytes);
        }
        if self.malformed {
            result.push(0);
        }
        self.executions.fetch_add(1, Ordering::Relaxed);
        Ok(vec![Some(KernelSizedOutput {
            shape,
            dtype: DataType::Float32,
            bytes: result,
        })])
    }
}

impl ExecutionProvider for PatternGraphEp {
    fn name(&self) -> &str {
        "pattern_graph_ep"
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

    fn claim_graph(&self, graph: &Graph) -> onnx_runtime_ep_api::Result<Vec<NodeId>> {
        use onnx_runtime_ir::pattern::{MatchBudgetScope, MatchOptions, MatchSession};
        if let Some(claims) = &self.claim_override {
            return Ok(claims.clone());
        }
        assert_eq!(
            graph.num_nodes(),
            3,
            "claim must see both siblings of the control-flow node"
        );
        self.claims.fetch_add(1, Ordering::Relaxed);
        let session = MatchSession::new(graph).map_err(|e| EpError::KernelFailed(e.to_string()))?;
        let mut claimed = Vec::new();
        for pattern in [
            if_pattern(),
            accumulating_pattern(true),
            accumulating_pattern(false),
        ] {
            let matches = session
                .find_matches(
                    &pattern,
                    MatchOptions {
                        budget_scope: MatchBudgetScope::Root,
                        ..Default::default()
                    },
                )
                .map_err(|e| EpError::KernelFailed(e.to_string()))?;
            claimed.extend(matches.into_iter().map(|matched| {
                matched
                    .value("out")
                    .unwrap()
                    .unwrap()
                    .value()
                    .producer
                    .unwrap()
            }));
        }
        Ok(claimed)
    }

    fn get_graph_kernel(
        &self,
        request: onnx_runtime_ep_api::GraphKernelRequest<'_>,
        shapes: &[Vec<usize>],
        _opset: u64,
    ) -> onnx_runtime_ep_api::Result<Box<dyn Kernel>> {
        if self.compile_failure {
            return Err(EpError::KernelFailed(
                "test graph compiler refused the accepted program".into(),
            ));
        }
        assert!(matches!(
            request.scope,
            onnx_runtime_ep_api::GraphKernelScope::Executor { .. }
        ));
        assert_eq!(request.graph.num_nodes(), 3);
        assert_eq!(
            request.graph.subgraphs.len(),
            if request.node.op_type == "If" { 2 } else { 1 }
        );
        assert_eq!(request.inputs.len(), shapes.len());
        if request.node.op_type == "If" {
            assert_eq!(request.inputs.len(), 2, "condition plus lexical capture");
        }
        let captured = request
            .inputs
            .iter()
            .position(|value| {
                value.is_some_and(|value| {
                    request.graph.value(value).name.as_deref() == Some("captured")
                })
            })
            .unwrap();
        self.compiles.fetch_add(1, Ordering::Relaxed);
        if request.node.op_type != "If" {
            let is_loop = request.node.op_type == "Loop";
            assert_eq!(request.inputs.len(), if is_loop { 4 } else { 3 });
            if is_loop {
                assert_eq!(
                    request.inputs[1], None,
                    "omitted Loop condition must retain its slot before appended captures"
                );
            }
            return Ok(Box::new(AccumulatingKernel {
                is_loop,
                captured,
                executions: self.executions.clone(),
            }));
        }
        Ok(Box::new(IfPatternKernel {
            captured,
            executions: self.executions.clone(),
            malformed: self.malformed,
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

    fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.sync()
    }
}

fn if_graph() -> Graph {
    let mut graph = Graph::new();
    graph.ir_version = Some(11);
    graph.opset_imports.insert(String::new(), 24);
    let n = graph.create_symbol(Some("n".into()));
    let x = graph.create_named_value("x", DataType::Float32, vec![Dim::Symbolic(n)]);
    let condition = graph.create_named_value("condition", DataType::Bool, vec![]);
    let captured = graph.create_named_value("captured", DataType::Float32, vec![Dim::Symbolic(n)]);
    let selected = graph.create_named_value("selected", DataType::Float32, vec![]);
    graph.mark_value_shape_unknown(selected);
    let output = graph.create_named_value("output", DataType::Float32, vec![]);
    graph.mark_value_shape_unknown(output);
    graph.set_inputs(vec![x, condition]);
    graph.set_outputs(vec![output]);
    // Deliberately insert If before the producer of its implicit input.
    let owner = graph.insert_node(Node::new(
        NodeId(0),
        "If",
        vec![Some(condition)],
        vec![selected],
    ));
    graph.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(x)],
        vec![captured],
    ));
    graph.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(selected)],
        vec![output],
    ));
    for (name, op) in [("then_branch", "Identity"), ("else_branch", "Concat")] {
        let mut body = Graph::new();
        let n = body.create_symbol(Some("n".into()));
        let free = body.create_named_value("captured", DataType::Float32, vec![Dim::Symbolic(n)]);
        let value = body.create_named_value("branch_output", DataType::Float32, vec![]);
        body.mark_value_shape_unknown(value);
        body.set_outputs(vec![value]);
        let inputs = if op == "Identity" {
            vec![Some(free)]
        } else {
            vec![Some(free), Some(free)]
        };
        let mut node = Node::new(NodeId(0), op, inputs, vec![value]);
        if op == "Concat" {
            node.attributes.insert("axis".into(), Attribute::Int(0));
        }
        body.insert_node(node);
        graph
            .node_mut(owner)
            .attributes
            .insert(name.into(), Attribute::Graph(Box::new(body.clone())));
        graph.subgraphs.insert((owner, name.into()), body);
    }
    graph
}

#[test]
fn whole_graph_claim_executes_if_pattern_with_captures_and_changing_output_shapes() {
    let graph = if_graph();
    assert_eq!(
        graph.topological_order().unwrap(),
        [NodeId(1), NodeId(0), NodeId(2)]
    );
    let ep = Arc::new(PatternGraphEp::new(false));
    let mut custom =
        Executor::build(graph.clone(), Arc::new(WeightStore::new()), ep.clone()).unwrap();
    let mut cpu = CpuExecutionProvider::new();
    cpu.initialize(&EpConfig::default()).unwrap();
    let mut reference =
        Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(cpu)).unwrap();
    for (condition, values) in [
        (true, vec![1., 2.]),
        (false, vec![3., 4.]),
        (false, vec![-2., 9.]),
        (true, vec![5., 6., 7.]),
    ] {
        let x = Tensor::from_f32(&[values.len()], &values).unwrap();
        let cond = Tensor::from_raw(DataType::Bool, vec![], &[u8::from(condition)]).unwrap();
        let actual = custom.run(&[("x", &x), ("condition", &cond)]).unwrap();
        let expected = reference.run(&[("x", &x), ("condition", &cond)]).unwrap();
        let mut math = values.clone();
        if !condition {
            math.extend_from_slice(&values);
        }
        assert_eq!(actual[0].shape, [math.len()]);
        assert_eq!(actual[0].to_vec_f32(), math);
        assert_eq!(actual[0].to_vec_f32(), expected[0].to_vec_f32());
    }
    assert!(ep.claims.load(Ordering::Relaxed) > 0);
    assert_eq!(ep.compiles.load(Ordering::Relaxed), 2);
    assert_eq!(ep.executions.load(Ordering::Relaxed), 4);
    assert_eq!(
        custom.control_flow_stats().subgraph_runs,
        0,
        "claimed If must execute through the EP, not the session's branch interpreter"
    );
    assert!(reference.control_flow_stats().subgraph_runs > 0);
}

#[test]
fn whole_graph_dynamic_output_payload_is_validated_before_publication() {
    let ep = Arc::new(PatternGraphEp::new(true));
    let mut executor = Executor::build(if_graph(), Arc::new(WeightStore::new()), ep).unwrap();
    let x = Tensor::from_f32(&[2], &[1., 2.]).unwrap();
    let cond = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
    let error = executor
        .run(&[("x", &x), ("condition", &cond)])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("graph node #0") && error.contains("returned 9 bytes, expected 8"),
        "{error}"
    );
}

fn accumulating_pattern(is_loop: bool) -> Pattern {
    let mut body = Pattern::builder();
    let inputs = if is_loop {
        vec![body.input("iter"), body.input("cond"), body.input("state")]
    } else {
        vec![body.input("state"), body.input("row")]
    };
    let delta = body.input("delta");
    let state = if is_loop {
        inputs[2]
    } else {
        body.op("Add", [inputs[0], inputs[1]])
    };
    let next = body.op("Add", [state, delta]);
    let body = body
        .finish_body(if is_loop {
            vec![inputs[1], next, next]
        } else {
            vec![next, next]
        })
        .unwrap();
    let mut p = Pattern::builder();
    let state = p.input("state");
    let [out, scan] = if is_loop {
        let trip = p.input("trip");
        let cond = p.optional_input("cond");
        p.node("Loop", [trip, cond, state])
            .attr("body", body)
            .outputs()
    } else {
        let rows = p.input("rows");
        p.node("Scan", [state, rows])
            .attr("num_scan_inputs", 1i64)
            .attr("body", body)
            .outputs()
    };
    p.capture("out", out);
    p.finish_outputs([out, scan]).unwrap()
}

struct AccumulatingKernel {
    is_loop: bool,
    captured: usize,
    executions: Arc<AtomicUsize>,
}
impl Kernel for AccumulatingKernel {
    fn execute(&self, _: &[TensorView], _: &mut [TensorMut]) -> onnx_runtime_ep_api::Result<()> {
        Err(EpError::KernelFailed(
            "accumulating pattern requires dynamic outputs".into(),
        ))
    }
    fn has_kernel_sized_outputs(&self) -> bool {
        true
    }
    fn execute_kernel_sized(
        &self,
        inputs: &[TensorView],
        requested: &[bool],
    ) -> onnx_runtime_ep_api::Result<Vec<Option<KernelSizedOutput>>> {
        assert_eq!(requested, [true, true]);
        for input in inputs.iter().filter(|input| !input.is_absent()) {
            input.validate()?;
            assert!(input.device.is_host_accessible() && input.is_contiguous());
        }
        let read = |input: &TensorView| -> Vec<f32> {
            if input.numel() == 0 {
                return Vec::new();
            }
            unsafe { std::slice::from_raw_parts(input.data_ptr::<f32>(), input.numel()) }.to_vec()
        };
        let state_index = if self.is_loop { 2 } else { 0 };
        let mut state = read(&inputs[state_index]);
        let delta = read(&inputs[self.captured]);
        let rows = if self.is_loop {
            Vec::new()
        } else {
            read(&inputs[1])
        };
        let iterations = if self.is_loop {
            assert!(inputs[1].is_absent());
            usize::try_from(unsafe { *inputs[0].data_ptr::<i64>() }).unwrap()
        } else {
            inputs[1].shape[0]
        };
        let mut scan = Vec::new();
        for row in 0..iterations {
            for index in 0..state.len() {
                state[index] += delta[index]
                    + if self.is_loop {
                        0.
                    } else {
                        rows[row * state.len() + index]
                    };
            }
            scan.extend_from_slice(&state);
        }
        self.executions.fetch_add(1, Ordering::Relaxed);
        let output = |shape, values: Vec<f32>| {
            Some(KernelSizedOutput {
                shape,
                dtype: DataType::Float32,
                bytes: values.into_iter().flat_map(f32::to_le_bytes).collect(),
            })
        };
        Ok(vec![
            output(vec![state.len()], state.clone()),
            output(vec![iterations, state.len()], scan),
        ])
    }
}

fn accumulating_graph(is_loop: bool, iterations: usize) -> Graph {
    let mut body = Graph::new();
    body.opset_imports.insert(String::new(), 24);
    let state = body.create_named_value("state", DataType::Float32, static_shape([2]));
    let captured = body.create_named_value("captured", DataType::Float32, static_shape([2]));
    let next = body.create_named_value("next", DataType::Float32, static_shape([2]));
    if is_loop {
        let iter = body.create_named_value("iter", DataType::Int64, vec![]);
        let cond = body.create_named_value("cond", DataType::Bool, vec![]);
        body.set_inputs(vec![iter, cond, state]);
        body.set_outputs(vec![cond, next, next]);
        body.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(state), Some(captured)],
            vec![next],
        ));
    } else {
        let row = body.create_named_value("row", DataType::Float32, static_shape([2]));
        let sum = body.create_named_value("sum", DataType::Float32, static_shape([2]));
        body.set_inputs(vec![state, row]);
        body.set_outputs(vec![next, next]);
        body.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(state), Some(row)],
            vec![sum],
        ));
        body.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(sum), Some(captured)],
            vec![next],
        ));
    }
    let mut graph = Graph::new();
    graph.ir_version = Some(11);
    graph.opset_imports.insert(String::new(), 24);
    let initial = graph.create_named_value("initial", DataType::Float32, static_shape([2]));
    let delta = graph.create_named_value("delta", DataType::Float32, static_shape([2]));
    let captured = graph.create_named_value("captured", DataType::Float32, static_shape([2]));
    let control = graph.create_named_value(
        "control",
        if is_loop {
            DataType::Int64
        } else {
            DataType::Float32
        },
        if is_loop {
            vec![]
        } else {
            static_shape([iterations, 2])
        },
    );
    let final_state = graph.create_named_value("final", DataType::Float32, static_shape([2]));
    let scan = graph.create_named_value("scan", DataType::Float32, static_shape([iterations, 2]));
    let output =
        graph.create_named_value("output", DataType::Float32, static_shape([iterations, 2]));
    graph.set_inputs(vec![initial, delta, control]);
    graph.set_outputs(vec![final_state, output]);
    let mut node = Node::new(
        NodeId(0),
        if is_loop { "Loop" } else { "Scan" },
        if is_loop {
            vec![Some(control), None, Some(initial)]
        } else {
            vec![Some(initial), Some(control)]
        },
        vec![final_state, scan],
    );
    if !is_loop {
        node.attributes
            .insert("num_scan_inputs".into(), Attribute::Int(1));
    }
    node.attributes
        .insert("body".into(), Attribute::Graph(Box::new(body.clone())));
    let owner = graph.insert_node(node);
    graph.subgraphs.insert((owner, "body".into()), body);
    graph.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(delta)],
        vec![captured],
    ));
    graph.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(scan)],
        vec![output],
    ));
    graph
}

#[test]
fn whole_graph_claim_executes_loop_and_scan_without_host_control_flow() {
    for is_loop in [true, false] {
        for iterations in [0, 1, 3] {
            let graph = accumulating_graph(is_loop, iterations);
            let ep = Arc::new(PatternGraphEp::new(false));
            let mut custom =
                Executor::build(graph.clone(), Arc::new(WeightStore::new()), ep.clone()).unwrap();
            let mut cpu = CpuExecutionProvider::new();
            cpu.initialize(&EpConfig::default()).unwrap();
            let mut reference =
                Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(cpu)).unwrap();
            for delta in [[0.5, 2.], [-1., 3.]] {
                let initial = Tensor::from_f32(&[2], &[1., -2.]).unwrap();
                let delta_tensor = Tensor::from_f32(&[2], &delta).unwrap();
                let rows = (0..iterations * 2).map(|i| i as f32).collect::<Vec<_>>();
                let control = if is_loop {
                    Tensor::from_raw(DataType::Int64, vec![], &(iterations as i64).to_le_bytes())
                        .unwrap()
                } else {
                    Tensor::from_f32(&[iterations, 2], &rows).unwrap()
                };
                let inputs = [
                    ("initial", &initial),
                    ("delta", &delta_tensor),
                    ("control", &control),
                ];
                let actual = custom.run(&inputs).unwrap();
                let expected = reference.run(&inputs).unwrap();
                let mut state = vec![1., -2.];
                let mut scan = Vec::new();
                for row in 0..iterations {
                    for index in 0..2 {
                        state[index] +=
                            delta[index] + if is_loop { 0. } else { rows[row * 2 + index] };
                    }
                    scan.extend_from_slice(&state);
                }
                assert_eq!(actual[0].to_vec_f32(), state);
                assert_eq!(actual[1].shape, [iterations, 2]);
                assert_eq!(actual[1].to_vec_f32(), scan);
                for (a, b) in actual.iter().zip(expected) {
                    assert_eq!(a.to_vec_f32(), b.to_vec_f32());
                    assert_eq!(a.shape, b.shape);
                }
            }
            assert_eq!(ep.executions.load(Ordering::Relaxed), 2);
            assert_eq!(custom.control_flow_stats().subgraph_runs, 0);
            if iterations != 0 {
                assert!(reference.control_flow_stats().subgraph_runs > 0);
            }
        }
    }
}

#[test]
fn whole_graph_invalid_claims_and_compile_errors_fail_without_fallback() {
    for ids in [vec![NodeId(0), NodeId(0)], vec![NodeId(u32::MAX)]] {
        let mut ep = PatternGraphEp::new(false);
        ep.claim_override = Some(ids);
        let error = Executor::build(if_graph(), Arc::new(WeightStore::new()), Arc::new(ep))
            .err()
            .expect("invalid claim must fail")
            .to_string();
        assert!(
            error.contains("invalid or duplicate whole-graph claim"),
            "{error}"
        );
    }
    let mut ep = PatternGraphEp::new(false);
    ep.compile_failure = true;
    let ep = Arc::new(ep);
    let mut executor =
        Executor::build(if_graph(), Arc::new(WeightStore::new()), ep.clone()).unwrap();
    let x = Tensor::from_f32(&[2], &[1., 2.]).unwrap();
    let condition = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
    let error = executor
        .run(&[("x", &x), ("condition", &condition)])
        .unwrap_err()
        .to_string();
    assert!(error.contains("test graph compiler refused"), "{error}");
    assert_eq!(ep.executions.load(Ordering::Relaxed), 0);
    assert_eq!(executor.control_flow_stats().subgraph_runs, 0);
}

#[test]
fn whole_graph_dynamic_outputs_reject_incompatible_external_storage_before_writing() {
    for (shape, strides, expected_error) in [
        (vec![1], None, "does not fit external binding"),
        (
            vec![2],
            Some(vec![2]),
            "requires contiguous external storage",
        ),
    ] {
        let graph = if_graph();
        let selected = graph
            .values
            .iter()
            .find(|(_, value)| value.name.as_deref() == Some("selected"))
            .unwrap()
            .0;
        let ep = Arc::new(PatternGraphEp::new(false));
        let mut executor = Executor::build(graph, Arc::new(WeightStore::new()), ep).unwrap();
        let mut backing = [42f32; 4];
        let mut external = ExternalBindings::default();
        external.outputs.insert(
            selected,
            ExternalValue {
                dtype: DataType::Float32,
                shape,
                accepts_subshape: false,
                strides,
                fixed_stride_shape: None,
                ptr: backing.as_mut_ptr() as usize,
                len: std::mem::size_of_val(&backing),
                alignment: std::mem::align_of::<f32>(),
                device: onnx_runtime_ir::DeviceId::cpu(),
            },
        );
        let x = Tensor::from_f32(&[2], &[1., 2.]).unwrap();
        let condition = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
        let error = executor
            .run_scoped(
                &[("x", &x), ("condition", &condition)],
                &HashMap::new(),
                &external,
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected_error), "{error}");
        assert_eq!(
            backing, [42.; 4],
            "rejected output must not overwrite caller storage"
        );
    }
}

#[test]
fn whole_graph_unknown_rank_inputs_follow_runtime_rank_changes() {
    let mut graph = if_graph();
    let unknown = graph
        .values
        .iter()
        .filter_map(|(id, value)| {
            matches!(value.name.as_deref(), Some("x" | "captured")).then_some(id)
        })
        .collect::<Vec<_>>();
    for value in unknown {
        graph.mark_value_shape_unknown(value);
    }
    let ep = Arc::new(PatternGraphEp::new(false));
    let mut custom =
        Executor::build(graph.clone(), Arc::new(WeightStore::new()), ep.clone()).unwrap();
    let mut cpu = CpuExecutionProvider::new();
    cpu.initialize(&EpConfig::default()).unwrap();
    let mut reference =
        Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(cpu)).unwrap();
    for (shape, condition) in [(vec![2], true), (vec![1, 2], false), (vec![2, 1], true)] {
        let x = Tensor::from_f32(&shape, &[1., 2.]).unwrap();
        let condition = Tensor::from_raw(DataType::Bool, vec![], &[u8::from(condition)]).unwrap();
        let inputs = [("x", &x), ("condition", &condition)];
        let actual = custom.run(&inputs).unwrap();
        let expected = reference.run(&inputs).unwrap();
        assert_eq!(actual[0].shape, expected[0].shape);
        assert_eq!(actual[0].to_vec_f32(), expected[0].to_vec_f32());
    }
    assert_eq!(ep.executions.load(Ordering::Relaxed), 3);
    assert_eq!(custom.control_flow_stats().subgraph_runs, 0);
}

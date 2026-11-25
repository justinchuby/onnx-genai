use super::*;
use onnx_runtime_ep_api::{DeviceBuffer, WorkspaceAllocation, WorkspaceContextKey};

struct ContextCountingCpu {
    cpu: CpuExecutionProvider,
    metadata_queries: Arc<AtomicUsize>,
}
struct ContextCountingKernel {
    inner: Box<dyn Kernel>,
    metadata_queries: Arc<AtomicUsize>,
}
impl Kernel for ContextCountingKernel {
    fn execute(&self, i: &[TensorView], o: &mut [TensorMut]) -> onnx_runtime_ep_api::Result<()> {
        self.inner.execute(i, o)
    }
    fn workspace_requirement(
        &self,
        i: &[TensorMetadata<'_>],
        o: &[OutputMetadata<'_>],
    ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::WorkspacePreparation> {
        self.metadata_queries.fetch_add(1, Ordering::SeqCst);
        self.inner.workspace_requirement(i, o)
    }
    fn workspace_requirement_for_execution(
        &self,
        i: &[TensorView],
        o: &[TensorMut],
        m: &[TensorMetadata<'_>],
        om: &[OutputMetadata<'_>],
        cached: Option<onnx_runtime_ep_api::WorkspacePreparation>,
    ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::WorkspaceExecutionPreparation> {
        self.inner
            .workspace_requirement_for_execution(i, o, m, om, cached)
    }
    fn execute_with_workspace(
        &self,
        i: &[TensorView],
        o: &mut [TensorMut],
        w: Option<onnx_runtime_ep_api::WorkspaceView>,
    ) -> onnx_runtime_ep_api::Result<()> {
        self.inner.execute_with_workspace(i, o, w)
    }
}
impl ExecutionProvider for ContextCountingCpu {
    fn name(&self) -> &str {
        "context_counting_cpu"
    }
    fn device_type(&self) -> DeviceType {
        self.cpu.device_type()
    }
    fn device_id(&self) -> onnx_runtime_ir::DeviceId {
        self.cpu.device_id()
    }
    fn initialize(&mut self, c: &EpConfig) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.initialize(c)
    }
    fn shutdown(&mut self) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.shutdown()
    }
    fn workspace_preparation_context(&self) -> onnx_runtime_ep_api::WorkspaceContextReader {
        self.cpu.workspace_preparation_context()
    }
    fn supports_op(
        &self,
        n: &Node,
        v: u64,
        s: &[Shape],
        d: &[DataType],
        l: &[TensorLayout],
    ) -> KernelMatch {
        self.cpu.supports_op(n, v, s, d, l)
    }
    fn get_kernel(
        &self,
        n: &Node,
        s: &[Vec<usize>],
        v: u64,
    ) -> onnx_runtime_ep_api::Result<Box<dyn Kernel>> {
        Ok(Box::new(ContextCountingKernel {
            inner: self.cpu.get_kernel(n, s, v)?,
            metadata_queries: self.metadata_queries.clone(),
        }))
    }
    fn allocate(&self, n: usize, a: usize) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
        self.cpu.allocate(n, a)
    }
    fn deallocate(&self, b: DeviceBuffer) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.deallocate(b)
    }
    fn allocate_workspace(
        &self,
        n: usize,
        a: usize,
        r: MemoryRole,
    ) -> onnx_runtime_ep_api::Result<WorkspaceAllocation> {
        self.cpu.allocate_workspace(n, a, r)
    }
    fn deallocate_workspace(&self, b: WorkspaceAllocation) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.deallocate_workspace(b)
    }
    fn copy(
        &self,
        a: &DeviceBuffer,
        b: &mut DeviceBuffer,
        n: usize,
    ) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.copy(a, b, n)
    }
    fn copy_async(
        &self,
        a: &DeviceBuffer,
        b: &mut DeviceBuffer,
        n: usize,
    ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::Fence> {
        self.cpu.copy_async(a, b, n)
    }
    fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.sync()
    }
}

fn qmoe_context_graph() -> Graph {
    use onnx_runtime_ir::TensorData;
    const R: usize = 3;
    const H: usize = 16;
    const I: usize = 32;
    const E: usize = 4;
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 24);
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([R, H]));
    let router = graph.create_named_value("router", DataType::Float32, static_shape([R, E]));
    let y = graph.create_named_value("y", DataType::Float32, static_shape([R, H]));
    let mut weight = |name: &str, k: usize, n: usize| {
        let w = graph.create_named_value(name, DataType::Uint8, static_shape([E, n, k / 2]));
        graph.set_initializer(
            w,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Uint8,
                vec![E, n, k / 2],
                vec![0x99; E * n * k / 2],
            )),
        );
        let scale = graph.create_named_value(
            format!("{name}_scale"),
            DataType::Float32,
            static_shape([E, n, k / 16]),
        );
        graph.set_initializer(
            scale,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![E, n, k / 16],
                [1f32.to_le_bytes(); E].repeat(n * k / 16).concat(),
            )),
        );
        (w, scale)
    };
    let (w1, s1) = weight("w1", H, I);
    let (w2, s2) = weight("w2", I, H);
    let mut node = Node::new(
        NodeId(0),
        "QMoE",
        vec![
            Some(x),
            Some(router),
            Some(w1),
            Some(s1),
            None,
            Some(w2),
            Some(s2),
        ],
        vec![y],
    );
    node.domain = "com.microsoft".into();
    for (name, attr) in [
        ("expert_weight_bits", Attribute::Int(4)),
        ("block_size", Attribute::Int(16)),
        ("k", Attribute::Int(2)),
        ("normalize_routing_weights", Attribute::Int(1)),
        ("activation_type", Attribute::String(b"identity".to_vec())),
    ] {
        node.attributes.insert(name.into(), attr);
    }
    graph.add_input(x);
    graph.add_input(router);
    graph.add_output(y);
    graph.insert_node(node);
    graph
}

#[test]
fn same_shapes_reprepare_workspace_when_selected_pool_changes() {
    check_context_repreparation(qmoe_context_graph(), &[3, 16], 512., Some((1408, 1664)));
}

fn check_context_repreparation(
    graph: Graph,
    input_shape: &[usize],
    expected: f32,
    bounds: Option<(u64, u64)>,
) {
    let calls = Arc::new(AtomicUsize::new(0));
    let ep = Arc::new(ContextCountingCpu {
        cpu: CpuExecutionProvider::initialized_with_resources(Default::default()).unwrap(),
        metadata_queries: calls.clone(),
    });
    let mut executor = Executor::build(graph, Arc::new(WeightStore::new()), ep, None).unwrap();
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut previous = None;
    for target in [&one, &four, &one, &four] {
        let width = target.current_num_threads();
        let before = calls.load(Ordering::SeqCst);
        let executor = &mut executor;
        let calls = &calls;
        let bytes = target.install(move || {
            let x = Tensor::from_f32(input_shape, &vec![1.; input_shape.iter().product()]).unwrap();
            let router = Tensor::from_f32(&[3, 4], &[0.; 12]).unwrap();
            let inputs = [("x", &x), ("router", &router)];
            let req = executor
                .prepare_with_device_bindings(&inputs, &mut [])
                .unwrap()
                .require_complete()
                .unwrap();
            if let Some((one, four)) = bounds {
                assert_eq!(req.step().bytes, if width == 1 { one } else { four });
            }
            assert!(req.step().bytes > 0);
            assert_eq!(
                executor
                    .workspace_preparation_memo
                    .as_ref()
                    .unwrap()
                    .context,
                WorkspaceContextKey::new(width as u64)
            );
            assert!(
                calls.load(Ordering::SeqCst) > before,
                "a changed context must not return the old same-shape memo"
            );
            let planned = calls.load(Ordering::SeqCst);
            assert_eq!(
                executor
                    .prepare_with_device_bindings(&inputs, &mut [])
                    .unwrap()
                    .require_complete()
                    .unwrap()
                    .step()
                    .bytes,
                req.step().bytes
            );
            assert_eq!(
                calls.load(Ordering::SeqCst),
                planned,
                "unchanged context must preserve the warm metadata hit"
            );
            for _ in 0..2 {
                let result = executor.run(&inputs).unwrap();
                assert!(result[0].to_vec_f32().iter().all(|x| *x == expected));
            }
            // Execution may prime a specialization. Re-publish against those
            // actual cache counters before testing the next context transition.
            executor
                .prepare_with_device_bindings(&inputs, &mut [])
                .unwrap()
                .require_complete()
                .unwrap();
            let stable = calls.load(Ordering::SeqCst);
            executor
                .prepare_with_device_bindings(&inputs, &mut [])
                .unwrap()
                .require_complete()
                .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), stable);
            req.step().bytes
        });
        if let Some((previous_width, previous_bytes)) = previous {
            assert_ne!(previous_width, width);
            assert_ne!(
                previous_bytes, bytes,
                "these fixtures must expose the width-dependent demand"
            );
        }
        previous = Some((width, bytes));
    }
}

fn block_context_graph(experts: bool) -> Graph {
    use onnx_runtime_ir::TensorData;
    let mut graph = Graph::new();
    graph.opset_imports.insert("pkg.nxrt".into(), 1);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([3, 32]));
    let router = graph.create_named_value("router", DataType::Float32, static_shape([3, 4]));
    let y = graph.create_named_value(
        "y",
        DataType::Float32,
        static_shape([3, if experts { 32 } else { 64 }]),
    );
    let mut pack = |name: &str, k: usize, n: usize| {
        let mut dims = vec![n, k / 32, 17];
        if experts {
            dims.insert(0, 4);
        }
        let mut bytes = Vec::new();
        for _ in 0..dims.iter().product::<usize>() / 17 {
            bytes.push(127);
            bytes.extend([0x22; 16]);
        }
        let w = graph.create_named_value(name, DataType::Uint8, static_shape(dims.iter().copied()));
        graph.set_initializer(
            w,
            WeightRef::Inline(TensorData::from_raw(DataType::Uint8, dims, bytes)),
        );
        w
    };
    let w1 = pack("w1", 32, 64);
    let mut node = if experts {
        let w2 = pack("w2", 64, 32);
        Node::new(
            NodeId(0),
            "BlockQuantizedMoE",
            vec![Some(x), Some(router), Some(w1), None, Some(w2)],
            vec![y],
        )
    } else {
        Node::new(
            NodeId(0),
            "BlockQuantizedMatMul",
            vec![Some(x), Some(w1)],
            vec![y],
        )
    };
    node.domain = "pkg.nxrt".into();
    node.attributes
        .insert("block_layout_version".into(), Attribute::Int(1));
    if experts {
        for name in ["fc1_format", "fc2_format"] {
            node.attributes
                .insert(name.into(), Attribute::String(b"mxfp4".to_vec()));
        }
        for (name, value) in [
            ("k", 2),
            ("normalize_routing_weights", 1),
            ("swiglu_fusion", 0),
        ] {
            node.attributes.insert(name.into(), Attribute::Int(value));
        }
        node.attributes.insert(
            "activation_type".into(),
            Attribute::String(b"identity".to_vec()),
        );
    } else {
        node.attributes.insert("K".into(), Attribute::Int(32));
        node.attributes.insert("N".into(), Attribute::Int(64));
        node.attributes
            .insert("format".into(), Attribute::String(b"mxfp4".to_vec()));
    }
    graph.add_input(x);
    graph.add_input(router);
    graph.add_output(y);
    graph.insert_node(node);
    graph
}

#[test]
fn block_matmul_same_shapes_reprepare_when_selected_pool_changes() {
    check_context_repreparation(block_context_graph(false), &[3, 32], 32., None);
}
#[test]
fn block_moe_same_shapes_reprepare_when_selected_pool_changes() {
    check_context_repreparation(block_context_graph(true), &[3, 32], 2048., None);
}

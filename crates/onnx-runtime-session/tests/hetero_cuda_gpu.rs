#![cfg(feature = "cuda")]

#[cfg(feature = "gpu-tests")]
#[path = "packed_shape_only/cases.rs"]
mod packed_shape_only_cases;

use std::{path::Path, sync::Arc};

use onnx_runtime_ep_api::{
    DeviceBuffer, DeviceId, DeviceType, EpConfig, EpId, ExecutionProvider, Fence, Kernel,
    KernelMatch, Result as EpResult,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, Shape, TensorLayout, static_shape,
};
use onnx_runtime_loader::WeightStore;
use onnx_runtime_session::hetero::{ProviderPlacement, execute, plan};
use onnx_runtime_session::{InferenceSession, Tensor};

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA; enable gpu-tests")]
fn packed_shape_only_cuda_no_fallback() {
    #[cfg(feature = "gpu-tests")]
    {
        use onnx_runtime_session::InferenceSession;
        use std::path::Path;

        let ep = Arc::new(CudaExecutionProvider::initialized(0).expect("initialize physical CUDA"));
        let build = |graph| {
            InferenceSession::from_graph_with_provider(
                graph,
                Arc::new(WeightStore::new()),
                Path::new("."),
                ep.clone(),
                None,
            )
            .expect("build native CUDA shape-only session")
        };
        packed_shape_only_cases::assert_transport(build, DeviceId::cuda(0));
        packed_shape_only_cases::assert_ir9_fp8_reshape(build, DeviceId::cuda(0));
    }
    #[cfg(not(feature = "gpu-tests"))]
    panic!("requires CUDA; enable gpu-tests");
}

struct CudaDecliningAbs {
    inner: CudaExecutionProvider,
}

impl ExecutionProvider for CudaDecliningAbs {
    fn consume_route_residency_at_boundary_for_executor(
        &self,
        executor: onnx_runtime_ep_api::ExecutorInstanceId,
    ) -> EpResult<()> {
        self.inner
            .consume_route_residency_at_boundary_for_executor(executor)
    }

    fn name(&self) -> &str {
        "cuda_except_abs"
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
        if op.domain.is_empty() && op.op_type == "Abs" {
            KernelMatch::unsupported("test fixture reserves Abs for the CPU partition")
        } else {
            self.inner
                .supports_op(op, opset, shapes, input_dtypes, layouts)
        }
    }

    fn get_kernel(
        &self,
        op: &Node,
        input_shapes: &[Vec<usize>],
        opset: u64,
    ) -> EpResult<Box<dyn Kernel>> {
        self.inner.get_kernel(op, input_shapes, opset)
    }

    fn allocate(&self, size: usize, alignment: usize) -> EpResult<DeviceBuffer> {
        self.inner.allocate(size, alignment)
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
        self.inner.copy_async(src, dst, size)
    }

    fn wait_fence(&self, fence: &Fence) -> EpResult<()> {
        self.inner.wait_fence(fence)
    }

    fn copy_from_host(&self, src: &[u8], dst: &mut DeviceBuffer) -> EpResult<()> {
        self.inner.copy_from_host(src, dst)
    }

    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> EpResult<()> {
        self.inner.copy_to_host(src, dst)
    }

    fn sync(&self) -> EpResult<()> {
        self.inner.sync()
    }
}

fn graph() -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(input);
    let relu = graph.create_named_value("relu", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(input)], vec![relu]));
    let abs = graph.create_named_value("abs", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Abs", vec![Some(relu)], vec![abs]));
    let output = graph.create_named_value("output", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Neg", vec![Some(abs)], vec![output]));
    graph.add_output(output);
    graph
}

#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
#[test]
fn eye_like_22_unknown_input_rank_runs_without_cpu_fallback() {
    let mut cuda = CudaExecutionProvider::new(0).expect("CUDA device 0");
    cuda.initialize(&EpConfig::default())
        .expect("initialize CUDA");
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 22);
    let dtype = DataType::BFloat16;
    let input = graph.create_named_value("x", dtype, vec![]);
    graph.mark_value_shape_unknown(input);
    graph.add_input(input);
    let output = graph.create_named_value("y", dtype, static_shape([2, 3]));
    graph.insert_node(Node::new(
        NodeId(0),
        "EyeLike",
        vec![Some(input)],
        vec![output],
    ));
    graph.add_output(output);

    let mut session = InferenceSession::from_graph_with_provider(
        graph,
        Arc::new(WeightStore::new()),
        Path::new("."),
        Arc::new(cuda),
        None,
    )
    .expect("build CUDA-only EyeLike-22 with unknown input rank");
    assert_eq!(session.device_id().device_type, DeviceType::Cuda);
    assert!(
        session.execution_provider_fallback_report().is_none(),
        "EyeLike-22 must not fall back to CPU: {:?}",
        session.execution_provider_fallback_report()
    );
    let x = Tensor::from_raw(dtype, vec![2, 3], &0x40e0_u16.to_le_bytes().repeat(6)).unwrap();
    let result = session.run(&[("x", &x)]).expect("run CUDA-only EyeLike-22");
    assert_eq!(result[0].dtype, dtype);
    assert_eq!(result[0].shape, [2, 3]);
    assert_eq!(
        result[0].as_bytes(),
        &[0x80, 0x3f, 0, 0, 0, 0, 0, 0, 0x80, 0x3f, 0, 0]
    );
}

fn einsum_graph(opset: u64) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let lhs = graph.create_named_value("lhs", DataType::BFloat16, static_shape([2, 3]));
    let rhs = graph.create_named_value("rhs", DataType::BFloat16, static_shape([3, 2]));
    let output = graph.create_named_value("output", DataType::BFloat16, static_shape([2, 2]));
    graph.add_input(lhs);
    graph.add_input(rhs);
    graph.add_output(output);
    let mut node = Node::new(
        NodeId(0),
        "Einsum",
        vec![Some(lhs), Some(rhs)],
        vec![output],
    );
    node.attributes
        .insert("equation".into(), Attribute::String(b"ik,kj->ij".to_vec()));
    graph.insert_node(node);
    graph
}

fn bf16_tensor(shape: &[usize], values: &[f32]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|&value| half::bf16::from_f32(value).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_raw(DataType::BFloat16, shape.to_vec(), &bytes).unwrap()
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires a CUDA device; enable gpu-tests to execute"
)]
#[test]
fn cuda_cpu_cuda_chain_executes_on_both_real_providers() {
    let mut cuda = CudaExecutionProvider::new(0).expect("CUDA device 0");
    cuda.initialize(&EpConfig::default())
        .expect("initialize CUDA");
    let mut cpu = CpuExecutionProvider::new();
    cpu.initialize(&EpConfig::default())
        .expect("initialize CPU");
    let providers = vec![
        ProviderPlacement {
            ep: EpId(0),
            provider: Arc::new(CudaDecliningAbs { inner: cuda }),
        },
        ProviderPlacement {
            ep: EpId(1),
            provider: Arc::new(cpu),
        },
    ];
    let graph = graph();
    let plan = plan(&graph, &providers).expect("mixed CUDA/CPU plan");
    assert_eq!(
        plan.node_placement
            .values()
            .filter(|&&ep| ep == EpId(0))
            .count(),
        2
    );
    assert_eq!(
        plan.node_placement
            .values()
            .filter(|&&ep| ep == EpId(1))
            .count(),
        1
    );
    assert_eq!(plan.transfers.len(), 2);

    let input = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    let output = execute(
        &plan,
        &graph,
        &Arc::new(WeightStore::new()),
        &providers,
        &[("x", &input)],
    )
    .expect("heterogeneous CUDA execution");
    assert_eq!(output[0].to_vec_f32(), vec![-0.0, -2.0, -0.0, -4.0]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires a CUDA device; enable gpu-tests to execute"
)]
#[test]
fn einsum_bf16_executes_at_v27_and_v28_in_strict_single_cuda_sessions() {
    let provider = Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA"));
    let lhs = bf16_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = bf16_tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let expected = bf16_tensor(&[2, 2], &[22.0, 28.0, 49.0, 64.0]);

    for opset in [27, 28] {
        let graph = einsum_graph(opset);
        assert_eq!(graph.opset_imports.get("").copied(), Some(opset));
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.local_opset().is_none()),
            "the session must resolve the graph import without rewriting node-local opsets"
        );
        let mut session = InferenceSession::from_graph_with_provider(
            graph,
            Arc::new(WeightStore::new()),
            Path::new("."),
            provider.clone(),
            None,
        )
        .unwrap_or_else(|error| {
            panic!("build a sole-provider CUDA session for Einsum-{opset} BFloat16: {error}")
        });
        assert_eq!(session.device_id(), DeviceId::new(DeviceType::Cuda, 0));
        assert!(session.execution_provider_fallback_report().is_none());
        assert!(session.heterogeneous_placement_report().is_none());

        let outputs = session
            .run(&[("lhs", &lhs), ("rhs", &rhs)])
            .unwrap_or_else(|error| panic!("execute Einsum-{opset} BFloat16 on CUDA: {error}"));
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].dtype, DataType::BFloat16);
        assert_eq!(outputs[0].shape, [2, 2]);
        assert_eq!(outputs[0].as_bytes(), expected.as_bytes());
    }
}

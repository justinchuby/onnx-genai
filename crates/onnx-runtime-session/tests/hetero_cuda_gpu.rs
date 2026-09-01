#![cfg(feature = "cuda")]

use std::sync::Arc;

use onnx_runtime_ep_api::{
    DeviceBuffer, DeviceId, DeviceType, EpConfig, EpId, ExecutionProvider, Fence, Kernel,
    KernelMatch, Result as EpResult,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{DataType, Graph, Node, NodeId, Shape, TensorLayout, static_shape};
use onnx_runtime_loader::WeightStore;
use onnx_runtime_session::Tensor;
use onnx_runtime_session::hetero::{ProviderPlacement, execute, plan};

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

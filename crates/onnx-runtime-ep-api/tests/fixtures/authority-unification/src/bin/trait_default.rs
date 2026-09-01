use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ir::Graph;

fn forge(provider: &dyn ExecutionProvider, graph: &Graph) {
    let _ = provider.finalize_executor_artifacts(graph);
}

fn main() {}

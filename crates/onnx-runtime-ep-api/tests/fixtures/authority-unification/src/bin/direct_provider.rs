use onnx_runtime_ep_api::ExecutionProvider;

fn forge(provider: &onnx_runtime_ep_cuda::CudaExecutionProvider) {
    let _issuer = provider.resolve_executor_artifact_config();
}

fn main() {}

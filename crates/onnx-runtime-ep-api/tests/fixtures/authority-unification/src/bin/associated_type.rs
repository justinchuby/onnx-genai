trait ExtractAuthority {
    type Authority;
}

impl ExtractAuthority for onnx_runtime_ep_cuda::CudaExecutionProvider {
    type Authority = onnx_runtime_ep_api::ExecutorArtifactFinalizationProof<'static>;
}

fn main() {}

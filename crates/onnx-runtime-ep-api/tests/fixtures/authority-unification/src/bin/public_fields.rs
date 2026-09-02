fn main() {
    let _forged = onnx_runtime_session::executor::ExecutorArtifactConfig {
        policy: unsafe { std::mem::zeroed() },
        executor: onnx_runtime_ep_api::ExecutorInstanceId::UNSCOPED,
        generation: onnx_runtime_ep_api::ExecutorArtifactGeneration::from_raw(1),
    };
}

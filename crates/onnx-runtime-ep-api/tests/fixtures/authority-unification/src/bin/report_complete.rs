use onnx_runtime_ep_api::ExecutorArtifactState;

fn main() {
    let _forged = ExecutorArtifactState::Complete {
        route_residency: ExecutorArtifactState::Declined,
    };
}

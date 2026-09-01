use onnx_runtime_ep_api::{ExecutorArtifactConfig, ExecutorArtifactReadinessEpoch};

fn forge(config: ExecutorArtifactConfig) {
    let _proof = config.finalization_proof(ExecutorArtifactReadinessEpoch::INITIAL);
}

fn main() {}

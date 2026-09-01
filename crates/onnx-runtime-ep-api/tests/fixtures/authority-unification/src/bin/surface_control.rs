use onnx_runtime_ep_api::{
    ExecutionProvider, ExecutorArtifactConfig, ExecutorArtifactConfigTemplate,
    ExecutorArtifactFinalizationProof, ExecutorArtifactReadinessEpoch, ExecutorInstanceId,
};
use onnx_runtime_ir::Graph;

fn reaches_intended_surface(
    provider: &dyn ExecutionProvider,
    template: ExecutorArtifactConfigTemplate,
    config: ExecutorArtifactConfig,
    proof: ExecutorArtifactFinalizationProof<'_>,
    graph: &Graph,
) {
    let _mint_surface = ExecutorInstanceId::fresh;
    let _ = template.device();
    let _ = config.generation();
    let _ = proof.readiness();
    let _ = ExecutorArtifactReadinessEpoch::INITIAL;
    let _ = provider.resolve_executor_artifact_config();
    let _ = provider.finalize_executor_artifacts(proof, graph);
}

fn main() {
    let _ = reaches_intended_surface;
}

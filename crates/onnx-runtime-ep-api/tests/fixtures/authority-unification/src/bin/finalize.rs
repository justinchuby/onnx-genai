use onnx_runtime_ep_api::{
    ExecutionProvider, ExecutorArtifactConfigTemplate, ExecutorArtifactReadinessEpoch,
    ExecutorInstanceId,
};
use onnx_runtime_ir::Graph;

fn forge(
    provider: &dyn ExecutionProvider,
    template: ExecutorArtifactConfigTemplate,
    graph: &Graph,
) {
    let executor = ExecutorInstanceId::fresh();
    let config = template.bind(executor);
    let proof = config.finalization_proof(ExecutorArtifactReadinessEpoch::INITIAL);
    let _accepted = provider.finalize_executor_artifacts(proof, graph);
}

fn main() {}

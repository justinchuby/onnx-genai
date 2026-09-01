use onnx_runtime_ep_api::{
    ExecutorArtifactGeneration, ExecutorArtifactPolicy, ExecutorArtifactProviderId,
    ExecutorArtifactReadinessEpoch, ExecutorArtifactReport, ExecutorArtifactState,
    ExecutorInstanceId, ExecutorRouteResidencyConfig,
};
use onnx_runtime_ir::{DeviceId, Graph};
use onnx_runtime_session::OpsetVersion;

fn main() {
    let provider = ExecutorArtifactProviderId::from_raw(7);
    let executor = ExecutorInstanceId::from_raw(11);
    let generation = ExecutorArtifactGeneration::from_raw(13);
    let readiness = ExecutorArtifactReadinessEpoch::new(17);
    let policy = ExecutorArtifactPolicy::new(
        provider,
        DeviceId::cuda(0),
        ExecutorRouteResidencyConfig::Enabled,
    );
    assert_eq!(policy.provider(), provider);
    assert_eq!(policy.device(), DeviceId::cuda(0));
    let report = ExecutorArtifactReport::observed(
        provider,
        executor,
        generation,
        readiness,
        ExecutorArtifactState::Declined,
    );
    assert_eq!(report.readiness(), readiness);
    let _ = Graph::new();
    let _ = OpsetVersion::Known(17);
    let _ = std::mem::size_of::<onnx_runtime_ep_cuda::CudaExecutionProvider>();
}

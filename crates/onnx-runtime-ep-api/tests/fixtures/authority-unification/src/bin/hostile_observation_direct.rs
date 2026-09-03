use std::any::Any;
use std::sync::Arc;

use onnx_runtime_ep_api::{
    ExecutionProvider, ExecutorArtifactGeneration, ExecutorInstanceId, ExecutorLogicalSessionId,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::byte_telemetry::{ObservedByteLedger, ObservedCategory, ObservedStatus};

fn main() {
    let mut provider = CudaExecutionProvider::new(0).expect("construct CUDA provider");
    provider
        .configure_observed_byte_capacity(32)
        .expect("enable bounded observation");

    let policy = provider
        .executor_artifact_policy()
        .expect("read public provider policy");
    let executor = ExecutorInstanceId::from_raw(0x2343);
    let generation = ExecutorArtifactGeneration::from_raw(0x56);
    let logical_session = ExecutorLogicalSessionId::from_raw(0x78);
    let owner: Arc<dyn Any + Send + Sync> = Arc::new(());

    let observation = provider
        .begin_executor_artifact_observation(
            policy.provider(),
            executor,
            generation,
            logical_session,
            Arc::clone(&owner),
        )
        .expect("public begin accepted caller labels and owner")
        .expect("configured provider returned observation state");

    let sibling_owner: Arc<dyn Any + Send + Sync> = Arc::new(());
    assert!(
        provider
            .begin_executor_artifact_observation(
                policy.provider(),
                executor,
                generation,
                logical_session,
                sibling_owner,
            )
            .is_err(),
        "same-label sibling owner must not replace the first caller's observation"
    );

    let mut buffer = None;
    let mut record = || {
        buffer = Some(provider.allocate(4096, 256)?);
        Ok(())
    };
    observation
        .with_observation(&mut record)
        .expect("public observation state recorded provider work");

    provider
        .commit_executor_artifact_observation(
            policy.provider(),
            executor,
            generation,
            logical_session,
            owner.as_ref(),
        )
        .expect("public commit accepted caller labels and owner");

    let requirement = provider
        .executor_artifact_requirement(policy.provider(), executor, generation, logical_session)
        .expect("retrieve forged requirement")
        .expect("forged observation produced requirement state");
    let ledger = requirement
        .observation()
        .and_then(|observation| observation.downcast_ref::<ObservedByteLedger>())
        .expect("forged requirement exposed observed-byte ledger");
    let snapshot = ledger.snapshot().expect("snapshot forged observation");

    assert_eq!(snapshot.scope.executor, executor.get());
    assert_eq!(snapshot.scope.generation, generation.get());
    assert_eq!(snapshot.scope.logical_session, logical_session.get());
    assert!(
        snapshot.bytes(
            ObservedCategory::DeviceAllocation,
            ObservedStatus::Committed,
        ) >= 4096,
        "provider work must be recorded in the caller-forged observation"
    );

    provider
        .deallocate(buffer.expect("recorded allocation"))
        .expect("release positive-control allocation");
    provider.sync().expect("complete positive-control release");
    println!(
        "hostile_observation_direct: begin=accepted commit=accepted record=4096 snapshot={} sibling_same_label=rejected",
        snapshot.bytes(
            ObservedCategory::DeviceAllocation,
            ObservedStatus::Committed,
        )
    );
}

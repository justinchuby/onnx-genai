//! A completed validation receipt covers earlier writes to caller workspace.
use onnx_runtime_ep_api::{DeviceValidationCompletion, ExecutionProvider};
use onnx_runtime_ep_cuda::{CudaExecutionProvider, runtime::cuptr};

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
)]
#[test]
fn validation_poll_is_host_only_and_preserves_exact_workspace_generation() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let runtime = ep.runtime();
    let mut submitter = ep.register_device_validation_owner().unwrap();
    let mut recipient = ep.register_device_validation_owner().unwrap();
    let mut source = ep.allocate(64, 256).unwrap();
    ep.copy_from_host(&[0x5a; 64], &mut source).unwrap();
    let requirement = onnx_runtime_ep_api::WorkspaceRequirement {
        bytes: 64,
        alignment: 256,
        lifetime: onnx_runtime_ep_api::WorkspaceLifetime::StepScoped,
    };
    let mut workspace = ep.allocate_workspace(64, 256, requirement.role()).unwrap();
    let first = ep.begin_device_validation(&submitter).unwrap();
    let bound = ep
        .add_device_validation_recipient(first, &recipient)
        .unwrap();
    ep.activate_device_validation(first).unwrap();
    // SAFETY: the live allocations both cover 64 bytes and share this runtime.
    unsafe {
        runtime
            .dtod_async(
                cuptr(source.as_ptr()),
                cuptr(workspace.as_mut_ptr().cast_const()),
                64,
            )
            .unwrap();
    }
    let transfers = runtime.transfer_counts();
    let allocations = runtime.allocation_counts();
    assert_eq!(
        ep.poll_device_validation_completion(&submitter, first)
            .unwrap(),
        DeviceValidationCompletion::Pending
    );
    assert_eq!(runtime.transfer_counts(), transfers);
    assert_eq!(runtime.allocation_counts(), allocations);
    ep.sync().unwrap();
    assert_eq!(
        ep.poll_device_validation_completion(&submitter, first)
            .unwrap(),
        DeviceValidationCompletion::Pending,
        "physical completion alone is not a clean validation receipt"
    );
    assert_eq!(
        ep.consume_device_validation_error(&recipient, bound)
            .unwrap(),
        0
    );
    let transfers = runtime.transfer_counts();
    for _ in 0..8 {
        assert_eq!(
            ep.poll_device_validation_completion(&submitter, first)
                .unwrap(),
            DeviceValidationCompletion::Complete { flags: 0 }
        );
    }
    assert_eq!(
        runtime.transfer_counts(),
        transfers,
        "observing must neither download nor reset the device latch"
    );
    assert_eq!(runtime.allocation_counts(), allocations);
    let mut bytes = [0; 64];
    // SAFETY: the exact receipt covers the previous stream-ordered workspace write.
    unsafe {
        runtime
            .dtoh(&mut bytes, cuptr(workspace.as_mut_ptr().cast_const()))
            .unwrap();
    }
    assert_eq!(bytes, [0x5a; 64]);
    let next = ep.begin_device_validation(&submitter).unwrap();
    assert!(
        ep.poll_device_validation_completion(&submitter, first)
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
    assert_eq!(
        ep.poll_device_validation_completion(&recipient, bound)
            .unwrap(),
        DeviceValidationCompletion::Complete { flags: 0 }
    );
    assert_eq!(
        ep.poll_device_validation_completion(&submitter, next)
            .unwrap(),
        DeviceValidationCompletion::Pending
    );
    let foreign = CudaExecutionProvider::new_default().unwrap();
    assert!(
        foreign
            .poll_device_validation_completion(&submitter, next)
            .unwrap_err()
            .to_string()
            .contains("another CUDA runtime")
    );
    ep.activate_device_validation(next).unwrap();
    ep.sync().unwrap();
    assert_eq!(
        ep.consume_device_validation_error(&submitter, next)
            .unwrap(),
        0
    );
    ep.unregister_device_validation_owner(&mut recipient)
        .unwrap();
    ep.unregister_device_validation_owner(&mut submitter)
        .unwrap();
    ep.deallocate_workspace(workspace).unwrap();
    ep.deallocate(source).unwrap();
}

//! Real registry + exact caller admission + one-use control snapshots.
#[cfg(feature = "gpu-tests")]
mod common;
#[cfg(feature = "gpu-tests")]
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, KernelMatch, OutputMetadata,
    TensorMetadata, TensorMut, TensorView, WorkspaceAllocation, WorkspacePlanning,
    WorkspaceRequirement, WorkspaceView,
};
#[cfg(feature = "gpu-tests")]
use onnx_runtime_ep_cuda::CudaExecutionProvider;
#[cfg(feature = "gpu-tests")]
use onnx_runtime_ir::{Attribute, DataType, compute_contiguous_strides, static_shape};

#[cfg(feature = "gpu-tests")]
fn prepare(
    ep: &CudaExecutionProvider,
    kernel: &dyn Kernel,
    inputs: &[TensorView],
    outputs: &[TensorMut],
    owner: &mut Option<WorkspaceAllocation>,
) -> (WorkspaceRequirement, Option<WorkspaceView>) {
    let metadata = inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, true))
        .collect::<Vec<_>>();
    let output_metadata = outputs
        .iter()
        .map(OutputMetadata::from_view)
        .collect::<Vec<_>>();
    let request = kernel
        .workspace_requirement_for_execution(inputs, outputs, &metadata, &output_metadata, None)
        .unwrap()
        .require_ready()
        .unwrap();
    let view = common::prepare_workspace(ep, request, owner).unwrap();
    (request, view)
}

// The capture regression uses the gpu-tests-only raw capture teardown API.
#[cfg(feature = "gpu-tests")]
fn exercise(packed: bool) {
    let ep = common::require_cuda();
    let runtime = ep.runtime();
    let op = if packed {
        "PackedVarlenAttention"
    } else {
        "VarlenAttention"
    };
    let q_shape = if packed {
        vec![2, 1, 1]
    } else {
        vec![1, 2, 1, 1]
    };
    let kv_shape = if packed {
        vec![4, 1, 1]
    } else {
        vec![1, 4, 1, 1]
    };
    let mut tensors = vec![
        common::input(DataType::Float32, &q_shape, &[0f32; 2]),
        common::input(DataType::Float32, &kv_shape, &[0f32; 4]),
        common::input(DataType::Float32, &kv_shape, &[2f32, 4., 6., 8.]),
    ];
    if packed {
        tensors.push(common::input(DataType::Int32, &[3], &[0i32, 1, 2]));
        tensors.push(common::input(DataType::Int32, &[3], &[0i32, 2, 4]));
    } else {
        tensors.push(common::input(DataType::Int64, &[1], &[2i64]));
    }
    let (graph, node) = common::build_graph(
        op,
        "pkg.nxrt",
        1,
        &tensors,
        &[(DataType::Float32, q_shape.clone())],
        &[("num_heads", Attribute::Int(1))],
    );
    let shapes = tensors.iter().map(|v| v.shape.clone()).collect::<Vec<_>>();
    let claim_shapes = shapes
        .iter()
        .map(|v| static_shape(v.iter().copied()))
        .collect::<Vec<_>>();
    let dtypes = tensors.iter().map(|v| v.dtype).collect::<Vec<_>>();
    assert!(matches!(
        ep.supports_op(graph.node(node), 1, &claim_shapes, &dtypes, &[]),
        KernelMatch::Supported {
            workspace_planning: WorkspacePlanning::Metadata,
            ..
        }
    ));
    let kernel = ep.get_kernel(graph.node(node), &shapes, 1).unwrap();
    assert!(!kernel.capture_support().is_supported());
    let buffers = tensors
        .iter()
        .map(|tensor| {
            let buffer = ep.allocate(tensor.bytes.len(), 256).unwrap();
            unsafe {
                runtime.htod(
                    &tensor.bytes,
                    onnx_runtime_ep_cuda::runtime::cuptr(buffer.as_ptr()),
                )
            }
            .unwrap();
            buffer
        })
        .collect::<Vec<_>>();
    let strides = shapes
        .iter()
        .map(|s| compute_contiguous_strides(s))
        .collect::<Vec<_>>();
    let inputs = tensors
        .iter()
        .zip(&buffers)
        .zip(&strides)
        .map(|((t, b), s)| {
            TensorView::new(DevicePtr(b.as_ptr()), t.dtype, &t.shape, s, ep.device_id())
        })
        .collect::<Vec<_>>();
    let mut output_buffer = ep.allocate(8, 256).unwrap();
    let output_strides = compute_contiguous_strides(&q_shape);
    let mut outputs = [TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        DataType::Float32,
        &q_shape,
        &output_strides,
        ep.device_id(),
    )];
    let mut owner = None;
    let control = if packed { 4 } else { 3 };
    let initial = tensors[control].bytes.clone();
    let changed = if packed {
        common::raw(&[0i32, 4, 4])
    } else {
        common::raw(&[4i64])
    };
    let invalid = if packed {
        common::raw(&[0i32, 3, 2])
    } else {
        common::raw(&[-1i64])
    };
    let upload = |bytes: &[u8]| {
        unsafe {
            runtime.htod(
                bytes,
                onnx_runtime_ep_cuda::runtime::cuptr(buffers[control].as_ptr()),
            )
        }
        .unwrap()
    };
    let read = || {
        let mut bytes = vec![0; 8];
        unsafe {
            runtime.dtoh(
                &mut bytes,
                onnx_runtime_ep_cuda::runtime::cuptr(output_buffer.as_ptr()),
            )
        }
        .unwrap();
        common::decode_floats(&bytes, DataType::Float32)
    };

    let (small, view) = prepare(&ep, kernel.as_ref(), &inputs, &outputs, &mut owner);
    assert_eq!(
        small.bytes,
        256 + 2 * 2 * 4,
        "size by actual maximum valid KV, not padded/packed total"
    );
    // Deliberately change the live control values after preparation: both ops
    // must consume their checked snapshot, including packed cumulative arrays.
    upload(&changed);
    let transfers = runtime.transfer_counts();
    kernel
        .execute_with_workspace(&inputs, &mut outputs, view)
        .unwrap();
    assert_eq!(
        runtime.transfer_counts().device_to_host,
        transfers.device_to_host,
        "no duplicate control readback in execute"
    );
    assert_eq!(
        runtime.transfer_counts().host_to_device,
        transfers.host_to_device + 1
    );
    assert_eq!(read(), if packed { vec![3., 7.] } else { vec![3., 3.] });
    let (large, view) = prepare(&ep, kernel.as_ref(), &inputs, &outputs, &mut owner);
    assert_eq!(large.bytes, 256 + 2 * 4 * 4);
    kernel
        .execute_with_workspace(&inputs, &mut outputs, view)
        .unwrap();
    assert_eq!(
        read(),
        if packed { vec![5., 0.] } else { vec![5., 5.] },
        "same pointer must be re-read on the next invocation"
    );
    upload(&initial);

    // Every rejected offer consumes its invocation; it may not fall back to a
    // stale control plan, re-read controls or allocate private scratch.
    for bad in 0..5 {
        let (request, view) = prepare(&ep, kernel.as_ref(), &inputs, &outputs, &mut owner);
        let good = view.unwrap();
        let bad_view = match bad {
            0 => None,
            1 => Some(WorkspaceView::new(good.ptr(), request.bytes as usize - 1)),
            2 => Some(WorkspaceView::new(
                DevicePtrMut(std::ptr::null_mut()),
                good.bytes(),
            )),
            3 => Some(WorkspaceView::new(
                DevicePtrMut(good.ptr().0.cast::<u8>().wrapping_add(1).cast()),
                good.bytes(),
            )),
            _ => Some(WorkspaceView::new(
                DevicePtrMut((usize::MAX & !255) as *mut _),
                good.bytes(),
            )),
        };
        let transfers = runtime.transfer_counts();
        let allocations = runtime.allocation_counts();
        let message = kernel
            .execute_with_workspace(&inputs, &mut outputs, bad_view)
            .unwrap_err()
            .to_string();
        assert!(message.contains("complete caller workspace"), "{message}");
        let message = kernel
            .execute_with_workspace(&inputs, &mut outputs, Some(good))
            .unwrap_err()
            .to_string();
        assert!(message.contains("missing prepared invocation"), "{message}");
        assert_eq!(runtime.transfer_counts(), transfers);
        assert_eq!(runtime.allocation_counts(), allocations);
    }
    let (_, view) = prepare(&ep, kernel.as_ref(), &inputs, &outputs, &mut owner);
    let mut stale = inputs.clone();
    stale[control].byte_offset = 4;
    let transfers = runtime.transfer_counts();
    assert!(
        kernel
            .execute_with_workspace(&stale, &mut outputs, view)
            .unwrap_err()
            .to_string()
            .contains("prepared input")
    );
    assert_eq!(runtime.transfer_counts(), transfers);

    // A failed preparation must clear an older unconsumed plan.
    let (_, view) = prepare(&ep, kernel.as_ref(), &inputs, &outputs, &mut owner);
    upload(&invalid);
    assert!(
        kernel
            .workspace_requirement_for_execution(&inputs, &outputs, &[], &[], None)
            .is_err()
    );
    assert!(
        kernel
            .execute_with_workspace(&inputs, &mut outputs, view)
            .unwrap_err()
            .to_string()
            .contains("missing prepared invocation")
    );
    upload(&initial);

    // Raw capture is rejected before D2H or admission, and before a prepared
    // execution's upload. Neither refusal may invalidate the stream.
    for during_prepare in [true, false] {
        let (_, view) = prepare(&ep, kernel.as_ref(), &inputs, &outputs, &mut owner);
        runtime
            .stream()
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .unwrap();
        let transfers = runtime.transfer_counts();
        let allocations = runtime.allocation_counts();
        if during_prepare {
            assert!(
                kernel
                    .workspace_requirement_for_execution(&inputs, &outputs, &[], &[], None)
                    .unwrap_err()
                    .to_string()
                    .contains("graph capture")
            );
        } else {
            assert!(
                kernel
                    .execute_with_workspace(&inputs, &mut outputs, view)
                    .unwrap_err()
                    .to_string()
                    .contains("finish CUDA graph capture")
            );
        }
        assert_eq!(
            runtime.stream_capture_status().unwrap(),
            cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
        );
        assert_eq!(runtime.transfer_counts(), transfers);
        assert_eq!(runtime.allocation_counts(), allocations);
        runtime.test_end_unregistered_graph_capture().unwrap();
        assert!(
            kernel
                .execute_with_workspace(&inputs, &mut outputs, view)
                .unwrap_err()
                .to_string()
                .contains("missing prepared invocation")
        );
    }
    let (_, view) = prepare(&ep, kernel.as_ref(), &inputs, &outputs, &mut owner);
    kernel
        .execute_with_workspace(&inputs, &mut outputs, view)
        .unwrap();
    assert_eq!(read(), if packed { vec![3., 7.] } else { vec![3., 3.] });

    // Zero work still validates control values, then declares a proven zero
    // request without touching the retained root or uploading a control table.
    let zero_shape = if packed {
        vec![0, 1, 1]
    } else {
        vec![1, 0, 1, 1]
    };
    let zero_strides = compute_contiguous_strides(&zero_shape);
    let mut empty_inputs = inputs.clone();
    empty_inputs[0] = TensorView::new(
        inputs[0].data,
        DataType::Float32,
        &zero_shape,
        &zero_strides,
        ep.device_id(),
    );
    if packed {
        unsafe {
            runtime.htod(
                &common::raw(&[0i32, 0, 0]),
                onnx_runtime_ep_cuda::runtime::cuptr(buffers[3].as_ptr()),
            )
        }
        .unwrap();
    }
    let mut empty_outputs = [TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        DataType::Float32,
        &zero_shape,
        &zero_strides,
        ep.device_id(),
    )];
    let allocations = runtime.allocation_counts();
    let transfers = runtime.transfer_counts();
    let owner_ptr = owner.as_ref().unwrap().as_ptr();
    let (request, view) = prepare(
        &ep,
        kernel.as_ref(),
        &empty_inputs,
        &empty_outputs,
        &mut owner,
    );
    assert_eq!(request.bytes, 0);
    assert!(view.is_none());
    assert_eq!(owner.as_ref().unwrap().as_ptr(), owner_ptr);
    assert_eq!(
        runtime.transfer_counts().device_to_host,
        transfers.device_to_host + if packed { 2 } else { 1 }
    );
    let transfers = runtime.transfer_counts();
    kernel
        .execute_with_workspace(&empty_inputs, &mut empty_outputs, None)
        .unwrap();
    assert_eq!(runtime.transfer_counts(), transfers);
    assert_eq!(runtime.allocation_counts(), allocations);

    runtime.drain_for_unmap().unwrap();
    ep.deallocate_workspace(owner.take().unwrap()).unwrap();
    ep.deallocate(output_buffer).unwrap();
    for buffer in buffers {
        ep.deallocate(buffer).unwrap();
    }
}
#[cfg(not(feature = "gpu-tests"))]
fn exercise(_: bool) {
    panic!("enable gpu-tests on a CUDA runner to exercise varlen caller workspace");
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA; enable gpu-tests")]
fn padded_exact_workspace_snapshot_and_failure_recovery() {
    exercise(false);
}
#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA; enable gpu-tests")]
fn packed_exact_workspace_snapshot_and_failure_recovery() {
    exercise(true);
}

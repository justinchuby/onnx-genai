use super::*;
use crate::kernels::testutil::{Owned, PreparedTestKernel};
#[test]
fn workspace_refusal_precedes_writes_and_fresh_storage_is_initialized() {
    use onnx_runtime_ep_api::DevicePtrMut;
    let k = LpNormalizationKernel { axis: -1, p: 2 };
    let x = Owned::f32(&[2, 3], &[1., 2., 3., -2., -1., 4.]);
    let mut y = Owned::f32(&[2, 3], &[-123.; 6]);
    let mut storage = [0x7fc01234u32; 8];
    let pointer = unsafe { storage.as_mut_ptr().add(1).cast::<u8>() };
    for ws in [
        None,
        Some(WorkspaceView::new(DevicePtrMut(pointer.cast()), 23)),
        Some(WorkspaceView::new(
            DevicePtrMut(unsafe { pointer.add(1).cast() }),
            24,
        )),
    ] {
        let error = k
            .execute_with_workspace(&[x.view()], &mut [y.view_mut()], ws)
            .unwrap_err();
        assert!(error.to_string().contains("workspace"), "{error}");
        assert_eq!(y.to_f32(), [-123.; 6]);
        assert_eq!(storage, [0x7fc01234; 8]);
    }
    k.execute_with_workspace(
        &[x.view()],
        &mut [y.view_mut()],
        Some(WorkspaceView::new(DevicePtrMut(pointer.cast()), 24)),
    )
    .unwrap();
    assert_eq!(storage[0], 0x7fc01234);
    assert_eq!(storage[7], 0x7fc01234);
    let expected = y.to_f32();
    let mut fresh = [std::mem::MaybeUninit::<f32>::uninit(); 6];
    k.execute_with_workspace(
        &[x.view()],
        &mut [y.view_mut()],
        Some(WorkspaceView::new(
            DevicePtrMut(fresh.as_mut_ptr().cast()),
            24,
        )),
    )
    .unwrap();
    assert_eq!(y.to_f32(), expected);
}

#[test]
fn governed_lpnorm_staging_reuses_one_lane_without_warm_issuer_callbacks() {
    use crate::kernels::testutil::RecordingGovernor;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_memory_api::Tier;
    use onnx_runtime_memory_governor::{
        HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
    };
    use std::sync::Arc;
    let books = Arc::new(RecordingGovernor::new_host(32));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources =
        crate::CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
            .unwrap()
            .with_workspace_pool(pool.clone())
            .unwrap();
    let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
    let kernel = LpNormalizationKernel { axis: -1, p: 2 };
    let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 4., 6., 8.]);
    let inputs = [x.view()];
    let meta = [TensorMetadata::new(inputs[0].dtype, inputs[0].shape, true)];
    let mut y = Owned::zeros_f32(&[2, 4]);
    let outputs = [OutputMetadata::tensor(
        onnx_runtime_ir::DataType::Float32,
        &[2, 4],
    )];
    let request = kernel
        .workspace_requirement(&meta, &outputs)
        .unwrap()
        .require_ready()
        .unwrap();
    assert_eq!(request.bytes, 32);
    let first = provider
        .allocate_workspace(32, request.alignment, request.role())
        .unwrap();
    let identity = first.managed_owner().unwrap().identity();
    assert!(
        provider
            .allocate_workspace(32, request.alignment, request.role())
            .is_err(),
        "active capacity cannot fund a concurrent execution"
    );
    provider.deallocate_workspace(first).unwrap();
    let before = (books.claims(), books.releases());
    drop(
        books
            .reserve(Tier::Host, 0, request.role(), HolderId::new(99))
            .unwrap(),
    );
    books.probe_zero_release();
    assert_eq!(
        (books.claims(), books.releases()),
        (before.0 + 1, before.1 + 1)
    );
    let calls = (books.claims(), books.releases());
    for _ in 0..4096 {
        let request = kernel
            .workspace_requirement_for_execution(&inputs, &[], &meta, &outputs, None)
            .unwrap()
            .require_ready()
            .unwrap();
        let mut allocation = provider
            .allocate_workspace(request.bytes as usize, request.alignment, request.role())
            .unwrap();
        assert_eq!(allocation.managed_owner().unwrap().identity(), identity);
        let workspace = WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(allocation.as_mut_ptr()),
            allocation.len(),
        );
        kernel
            .execute_with_workspace(&inputs, &mut [y.view_mut()], Some(workspace))
            .unwrap();
        provider.deallocate_workspace(allocation).unwrap();
    }
    assert_eq!(
        (books.claims(), books.releases()),
        calls,
        "including zero-byte callbacks"
    );
    assert_eq!(books.used(Tier::Host), 32);
    for row in y.to_f32().chunks_exact(4) {
        assert!((row.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-6);
    }
    pool.trim_idle(32).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
}
#[test]
fn invalid_views_are_rejected_before_staging_or_output_writes() {
    use onnx_runtime_ep_api::DevicePtrMut;
    use onnx_runtime_ir::DeviceId;
    let kernel = LpNormalizationKernel { axis: -1, p: 2 };
    let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let mut y = Owned::f32(&[2, 3], &[-123.; 6]);
    let mut backing = [0x7fc01234u32; 6];
    for case in 0..6 {
        let mut input = x.view();
        let mut output = y.view_mut();
        match case {
            0 => input.strides = &[],
            1 => input.byte_offset = usize::MAX,
            2 => input.device = DeviceId::cuda(0),
            3 => output.strides = &[],
            4 => output.byte_offset = usize::MAX,
            _ => output.device = DeviceId::cuda(0),
        }
        let error = kernel
            .execute_with_workspace(
                &[input],
                &mut [output],
                Some(WorkspaceView::new(
                    DevicePtrMut(backing.as_mut_ptr().cast()),
                    24,
                )),
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("LpNormalization"),
            "case {case}: {error}"
        );
        assert_eq!(y.to_f32(), [-123.; 6]);
        assert_eq!(backing, [0x7fc01234; 6]);
    }
}
#[test]
fn metadata_checks_storage_bytes_shapes_axes_and_empty_capacity() {
    use onnx_runtime_ir::DataType;
    for dtype in [
        DataType::Float16,
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float64,
    ] {
        let k = LpNormalizationKernel { axis: 1, p: 2 };
        let meta = [TensorMetadata::new(dtype, &[2, 3, 4], true)];
        let ready = k
            .workspace_requirement(&meta, &[OutputMetadata::Unresolved { dtype }])
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(
            (ready.bytes, ready.alignment),
            (24 * dtype.byte_size() as u64, dtype.byte_size())
        );
        for shape in [&[0, 3, 4][..], &[2, 0, 4], &[2, 3, 0]] {
            let x = Owned::zeros(dtype, shape);
            let mut y = Owned::zeros(dtype, shape);
            assert_eq!(
                k.workspace_requirement(
                    &[TensorMetadata::new(dtype, shape, true)],
                    &[OutputMetadata::from_view(&y.view_mut())]
                )
                .unwrap()
                .require_ready()
                .unwrap()
                .bytes,
                0
            );
            k.execute(&[x.view()], &mut [y.view_mut()]).unwrap();
        }
        for bad in [
            vec![],
            vec![OutputMetadata::Absent],
            vec![OutputMetadata::tensor(dtype, &[24])],
            vec![OutputMetadata::tensor(DataType::Int32, &[2, 3, 4])],
        ] {
            assert!(k.workspace_requirement(&meta, &bad).is_err());
        }
        for axis in [i64::MIN, -4, 3, i64::MAX] {
            assert!(
                LpNormalizationKernel { axis, p: 1 }
                    .workspace_requirement(&meta, &[OutputMetadata::Unresolved { dtype }])
                    .unwrap_err()
                    .to_string()
                    .contains("axis")
            );
        }
        for shape in [&[0, usize::MAX, 2][..], &[2, isize::MAX as usize]] {
            assert!(
                k.workspace_requirement(
                    &[TensorMetadata::new(dtype, shape, true)],
                    &[OutputMetadata::Unresolved { dtype }]
                )
                .is_err()
            );
        }
    }
}
#[test]
fn storage_staging_preserves_f64_domain_half_precision_and_tiny_floor() {
    let k = LpNormalizationKernel { axis: -1, p: 2 };
    let x = Owned::f64(&[2], &[1e100, 2e100]);
    let mut y = Owned::f64(&[2], &[0.; 2]);
    k.execute_prepared(&[x.view()], &mut [y.view_mut()])
        .unwrap();
    for (got, want) in y.to_f64().iter().zip([1. / 5f64.sqrt(), 2. / 5f64.sqrt()]) {
        assert!((got - want).abs() < 1e-15);
    }
    for p in [1, 2] {
        for x in [
            Owned::f32(&[2, 3], &[0., 0., 0., 3., 4., 0.]),
            Owned::f16(&[2, 3], &[0., 0., 0., 3., 4., 0.]),
            Owned::bf16(&[2, 3], &[0., 0., 0., 3., 4., 0.]),
        ] {
            let mut y = Owned::zeros(x.dtype, &[2, 3]);
            LpNormalizationKernel { axis: 1, p }
                .execute_prepared(&[x.view()], &mut [y.view_mut()])
                .unwrap();
            let got = crate::dtype::to_dense_f32_widen("test", &y.view())
                .unwrap()
                .to_vec();
            let denom = if p == 1 { 7. } else { 5. };
            for (got, want) in got.iter().zip([0., 0., 0., 3. / denom, 4. / denom, 0.]) {
                assert!((got - want).abs() < 0.004);
            }
        }
    }
    let x = Owned::f64(&[1], &[f64::MIN_POSITIVE / 2.]);
    let mut y = Owned::f64(&[1], &[0.]);
    LpNormalizationKernel { axis: 0, p: 1 }
        .execute_prepared(&[x.view()], &mut [y.view_mut()])
        .unwrap();
    assert_eq!(y.to_f64(), [0.5]);
}
#[test]
fn aliased_and_strided_output_preserves_complete_source_reads() {
    let k = LpNormalizationKernel { axis: 1, p: 2 };
    let x = Owned::f32(&[2, 3], &[3., 4., 0., 0., 5., 12.]);
    let mut data = [
        3f32, 4., 0., 0., 5., 12., -99., -99., -99., -99., -99., -99.,
    ];
    let mut input = x.view();
    input.data = onnx_runtime_ep_api::DevicePtr(data.as_ptr().cast());
    let mut output = Owned::zeros_f32(&[2, 3]);
    let mut view = output.view_mut();
    view.data = onnx_runtime_ep_api::DevicePtrMut(data.as_mut_ptr().cast());
    view.strides = &[6, 2];
    k.execute_prepared(&[input], &mut [view]).unwrap();
    for (i, want) in [0.6, 0.8, 0., 0., 5. / 13., 12. / 13.]
        .into_iter()
        .enumerate()
    {
        assert!((data[2 * i] - want).abs() < 1e-6);
    }
    assert_eq!(data[7], -99.);
    assert_eq!(data[11], -99.);
}

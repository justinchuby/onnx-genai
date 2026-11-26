use super::*;
use crate::kernels::testutil::Owned;
use onnx_runtime_ir::DataType;
#[test]
fn workspace_refusal_precedes_writes_and_fresh_storage_is_initialized() {
    use onnx_runtime_ep_api::DevicePtrMut;
    let k = SoftmaxKernel {
        axis: -1,
        coerce_2d: false,
    };
    let x = Owned::f32(&[2, 3], &[1., 2., 3., -2., -1., 4.]);
    let mut y = Owned::f64(&[2, 3], &[-123.; 6]);
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
        assert_eq!(
            to_dense_f32_widen("test", &y.view()).unwrap().to_vec(),
            [-123.; 6]
        );
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
    let expected = to_dense_f32_widen("test", &y.view()).unwrap().to_vec();
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
    assert_eq!(
        to_dense_f32_widen("test", &y.view()).unwrap().to_vec(),
        expected
    );
}

#[test]
fn governed_softmax_staging_reuses_one_lane_without_warm_issuer_callbacks() {
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
    let kernel = SoftmaxKernel {
        axis: -1,
        coerce_2d: false,
    };
    let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 4., 6., 8.]);
    let inputs = [x.view()];
    let meta = [TensorMetadata::new(inputs[0].dtype, inputs[0].shape, true)];
    let mut y = Owned::f64(&[2, 4], &[0.; 8]);
    let outputs = [OutputMetadata::tensor(
        onnx_runtime_ir::DataType::Float64,
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
    let mut direct_y = Owned::zeros_f32(&[2, 4]);
    let widened_x = Owned::f16(&[2, 4], &[1., 2., 3., 4., 2., 4., 6., 8.]);
    for _ in 0..4096 {
        assert_eq!(
            refined(&kernel, &inputs[0], &direct_y.view_mut())
                .unwrap()
                .bytes,
            0
        );
        kernel.execute(&inputs, &mut [direct_y.view_mut()]).unwrap();
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
        assert_eq!(
            refined(&kernel, &widened_x.view(), &direct_y.view_mut())
                .unwrap()
                .bytes,
            32
        );
        kernel
            .execute_with_workspace(
                &[widened_x.view()],
                &mut [direct_y.view_mut()],
                Some(workspace),
            )
            .unwrap();
        provider.deallocate_workspace(allocation).unwrap();
    }
    assert_eq!(
        (books.claims(), books.releases()),
        calls,
        "including zero-byte callbacks"
    );
    assert_eq!(books.used(Tier::Host), 32);
    for row in to_dense_f32_widen("test", &y.view())
        .unwrap()
        .chunks_exact(4)
    {
        assert!((row.iter().sum::<f32>() - 1.).abs() < 1e-6);
    }
    pool.trim_idle(32).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
}
#[test]
fn invalid_views_are_rejected_before_staging_or_output_writes() {
    use onnx_runtime_ep_api::DevicePtrMut;
    use onnx_runtime_ir::DeviceId;
    let kernel = SoftmaxKernel {
        axis: -1,
        coerce_2d: false,
    };
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
            error.to_string().contains("Softmax"),
            "case {case}: {error}"
        );
        assert_eq!(y.to_f32(), [-123.; 6]);
        assert_eq!(backing, [0x7fc01234; 6]);
    }
}
fn kernel() -> SoftmaxKernel {
    SoftmaxKernel {
        axis: -1,
        coerce_2d: false,
    }
}
fn refined(k: &SoftmaxKernel, x: &TensorView, y: &TensorMut) -> Result<WorkspaceRequirement> {
    k.workspace_requirement_for_execution(
        &[*x],
        std::slice::from_ref(y),
        &[TensorMetadata::new(x.dtype, x.shape, true)],
        &[OutputMetadata::from_view(y)],
        None,
    )?
    .require_ready()
}
#[test]
fn metadata_defers_address_dependent_routes_and_checks_complete_geometry() {
    for legacy in [false, true] {
        let k = SoftmaxKernel {
            axis: 1,
            coerce_2d: legacy,
        };
        let meta = [TensorMetadata::new(DataType::Float32, &[2, 3, 4], true)];
        for dtype in [
            DataType::Float16,
            DataType::BFloat16,
            DataType::Float32,
            DataType::Float64,
        ] {
            let output = [OutputMetadata::Unresolved { dtype }];
            let request = k.workspace_requirement(&meta, &output).unwrap();
            if dtype == DataType::Float32 {
                assert!(matches!(request, WorkspacePreparation::Deferred));
            } else {
                assert_eq!(request.require_ready().unwrap().bytes, 96);
            }
            assert_eq!(
                k.workspace_requirement_for_execution(&[], &[], &meta, &output, None)
                    .unwrap()
                    .require_ready()
                    .unwrap()
                    .bytes,
                96
            );
        }
        for dtype in [DataType::Float16, DataType::BFloat16, DataType::Float64] {
            let input = [TensorMetadata::new(dtype, &[2, 3, 4], true)];
            let output = [OutputMetadata::Unresolved {
                dtype: DataType::Float32,
            }];
            assert_eq!(
                k.workspace_requirement(&input, &output)
                    .unwrap()
                    .require_ready()
                    .unwrap()
                    .bytes,
                96
            );
        }
        for bad in [
            vec![],
            vec![OutputMetadata::Absent],
            vec![OutputMetadata::tensor(DataType::Float32, &[24])],
            vec![OutputMetadata::tensor(DataType::Int32, &[2, 3, 4])],
        ] {
            assert!(
                k.workspace_requirement(&meta, &bad)
                    .unwrap_err()
                    .to_string()
                    .contains("Softmax")
            );
        }
        for shape in [&[0, 3, 4][..], &[2, 0, 4], &[2, 3, 0]] {
            let x = Owned::f32(shape, &[]);
            let mut y = Owned::zeros_f32(shape);
            assert_eq!(
                k.workspace_requirement(
                    &[TensorMetadata::new(DataType::Float32, shape, true)],
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
        for shape in [&[0, usize::MAX, 2][..], &[2, isize::MAX as usize]] {
            assert!(
                k.workspace_requirement(
                    &[TensorMetadata::new(DataType::Float32, shape, true)],
                    &[OutputMetadata::tensor(DataType::Float32, shape)]
                )
                .is_err()
            );
        }
        for axis in [i64::MIN, -4, 3, i64::MAX] {
            assert!(
                SoftmaxKernel {
                    axis,
                    coerce_2d: legacy
                }
                .workspace_requirement(
                    &meta,
                    &[OutputMetadata::tensor(DataType::Float32, &[2, 3, 4])]
                )
                .unwrap_err()
                .to_string()
                .contains("axis")
            );
        }
    }
}
#[test]
fn actual_views_preserve_zero_scratch_and_admit_alias_strided_and_unaligned_output() {
    let k = kernel();
    let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let mut y = Owned::zeros_f32(&[2, 3]);
    assert_eq!(refined(&k, &x.view(), &y.view_mut()).unwrap().bytes, 0);
    k.execute(&[x.view()], &mut [y.view_mut()]).unwrap();
    let expected = y.to_f32();
    let mut alias = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let mut input = x.view();
    input.data = onnx_runtime_ep_api::DevicePtr(alias.bytes.as_ptr().cast());
    let output = alias.view_mut();
    assert_eq!(refined(&k, &input, &output).unwrap().bytes, 24);
    let mut scratch = [0u32; 6];
    k.execute_with_workspace(
        &[input],
        &mut [output],
        Some(WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(scratch.as_mut_ptr().cast()),
            24,
        )),
    )
    .unwrap();
    assert_eq!(alias.to_f32(), expected);
    let mut padded = [0f32; 12];
    let mut out = y.view_mut();
    out.data = onnx_runtime_ep_api::DevicePtrMut(padded.as_mut_ptr().cast());
    out.strides = &[6, 2];
    assert_eq!(refined(&k, &x.view(), &out).unwrap().bytes, 24);
    k.execute_with_workspace(
        &[x.view()],
        &mut [out],
        Some(WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(scratch.as_mut_ptr().cast()),
            24,
        )),
    )
    .unwrap();
    for i in 0..6 {
        assert_eq!(padded[2 * i], expected[i]);
        assert_eq!(padded[2 * i + 1], 0.);
    }
    let mut packed = [0u32; 7];
    let mut out = y.view_mut();
    out.data = onnx_runtime_ep_api::DevicePtrMut(unsafe {
        packed.as_mut_ptr().cast::<u8>().add(1).cast()
    });
    assert_eq!(refined(&k, &x.view(), &out).unwrap().bytes, 24);
    k.execute_with_workspace(
        &[x.view()],
        &mut [out],
        Some(WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(scratch.as_mut_ptr().cast()),
            24,
        )),
    )
    .unwrap();
    for (i, &want) in expected.iter().enumerate() {
        assert_eq!(
            unsafe {
                packed
                    .as_ptr()
                    .cast::<u8>()
                    .add(1 + 4 * i)
                    .cast::<f32>()
                    .read_unaligned()
            },
            want
        );
    }
    let mut invalid_x = x.view();
    invalid_x.data = onnx_runtime_ep_api::DevicePtr((usize::MAX - 3) as *const _);
    invalid_x.byte_offset = 8;
    assert!(
        refined(&k, &invalid_x, &y.view_mut())
            .unwrap_err()
            .to_string()
            .contains("X address range wraps")
    );
    let mut wrong_y = y.view_mut();
    wrong_y.dtype = DataType::Float64;
    assert!(
        k.workspace_requirement_for_execution(
            &[x.view()],
            &[wrong_y],
            &[TensorMetadata::new(DataType::Float32, &[2, 3], true)],
            &[OutputMetadata::tensor(DataType::Float32, &[2, 3])],
            None
        )
        .unwrap_err()
        .to_string()
        .contains("refresh metadata")
    );
    let mut wrap = y.view_mut();
    wrap.data = onnx_runtime_ep_api::DevicePtrMut((usize::MAX - 3) as *mut _);
    assert!(!output_direct_write_eligible(&wrap, 6, &[]));
    assert!(
        refined(&k, &x.view(), &wrap)
            .unwrap_err()
            .to_string()
            .contains("Y address range wraps")
    );
}
#[test]
fn materialized_sources_use_one_admitted_snapshot_before_aliased_direct_output() {
    let k = kernel();
    let shape = [2, 3];
    for case in 0..3 {
        let mut backing = [0u32; 12];
        let origin = backing.as_mut_ptr().cast::<u8>();
        let owner = Owned::f32(&shape, &[1., 2., 3., 4., 5., 6.]);
        let mut x = owner.view();
        x.data = onnx_runtime_ep_api::DevicePtr(origin.cast());
        match case {
            0 => {
                x.dtype = DataType::Float16;
                for i in 0..6 {
                    unsafe {
                        origin
                            .add(2 * i)
                            .cast::<u16>()
                            .write_unaligned(half::f16::from_f32(i as f32 + 1.).to_bits())
                    };
                }
            }
            1 => {
                x.strides = &[6, 2];
                for i in 0..6 {
                    unsafe {
                        origin
                            .add(8 * i)
                            .cast::<f32>()
                            .write_unaligned(i as f32 + 1.)
                    };
                }
            }
            _ => {
                x.data = onnx_runtime_ep_api::DevicePtr(unsafe { origin.add(1).cast() });
                for i in 0..6 {
                    unsafe {
                        origin
                            .add(1 + 4 * i)
                            .cast::<f32>()
                            .write_unaligned(i as f32 + 1.)
                    };
                }
            }
        }
        assert!(!can_borrow_dense::<f32>(&x));
        let mut y = onnx_runtime_ep_api::TensorMut::new(
            onnx_runtime_ep_api::DevicePtrMut(origin.cast()),
            DataType::Float32,
            &shape,
            &[3, 1],
            onnx_runtime_ir::DeviceId::cpu(),
        );
        assert_eq!(refined(&k, &x, &y).unwrap().bytes, 24);
        let before = backing;
        assert!(k.execute(&[x], std::slice::from_mut(&mut y)).is_err());
        assert_eq!(
            backing, before,
            "unfunded snapshot cannot touch aliased storage"
        );
        let mut scratch = [std::mem::MaybeUninit::<f32>::uninit(); 6];
        let pointer = scratch.as_mut_ptr().cast();
        assert!(
            k.execute_with_workspace(
                &[x],
                std::slice::from_mut(&mut y),
                Some(WorkspaceView::new(
                    onnx_runtime_ep_api::DevicePtrMut(pointer),
                    23
                )),
            )
            .is_err()
        );
        assert_eq!(
            backing, before,
            "short snapshot cannot touch aliased storage"
        );
        k.execute_with_workspace(
            &[x],
            std::slice::from_mut(&mut y),
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(pointer),
                24,
            )),
        )
        .unwrap();
        for row in backing[..6].chunks_exact(3) {
            for (v, want) in row.iter().zip([0.09003057, 0.24472847, 0.66524096]) {
                assert!((f32::from_bits(*v) - want).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn snapshot_in_place_reducer_matches_existing_borrowed_reducer_bits() {
    for width in [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 65] {
        for inner in [1, 2, 3] {
            let outer = 3;
            for specials in [false, true] {
                let source = (0..outer * width * inner)
                    .map(|i| {
                        if specials {
                            [f32::NEG_INFINITY, f32::INFINITY, f32::NAN, -0., 0., 3.][i % 6]
                        } else {
                            (i as f32 * 0.175).sin() * 5.
                        }
                    })
                    .collect::<Vec<_>>();
                let mut expected = vec![0.; source.len()];
                softmax_slices(&source, &mut expected, outer, width, inner);
                let mut actual = source.clone();
                softmax_slices_in_place(&mut actual, outer, width, inner);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "width={width} inner={inner} specials={specials}"
                );
            }
        }
    }
}

#[test]
fn untransformed_serial_runs_batch_once_and_transformed_runs_keep_cache_tiles() {
    use std::cell::RefCell;
    let shape = [33, 257]; // 33,924 bytes: crosses the 32 KiB tile boundary.
    let eff = [0, 0];
    let mask_values = [0.25];
    for (scale, mask, expected, calls) in [
        (1., None, 1., vec![33 * 257]),
        (2., None, 2., vec![31 * 257, 2 * 257]),
        (
            1.,
            Some(MaskPlan {
                values: &mask_values,
                eff: &eff,
                scores_shape: &shape,
            }),
            1.25,
            vec![31 * 257, 2 * 257],
        ),
    ] {
        let mut data = vec![1f32; 33 * 257];
        let observed = RefCell::new(Vec::new());
        scale_mask_softmax_serial_with_kernel(&mut data, 0, 257, scale, mask, |rows, width| {
            assert_eq!(width, 257);
            assert!(rows.iter().all(|&v| v == expected));
            observed.borrow_mut().push(rows.len());
        });
        assert_eq!(*observed.borrow(), calls);
    }
}

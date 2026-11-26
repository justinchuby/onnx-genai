use super::*;
use crate::kernels::testutil::{Owned, PreparedTestKernel};
use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
use onnx_runtime_ir::{DataType, DeviceId};

fn owned(dtype: DataType, shape: &[usize], values: &[f64]) -> Owned {
    let f32s = values.iter().map(|&v| v as f32).collect::<Vec<_>>();
    match dtype {
        DataType::Float32 => Owned::f32(shape, &f32s),
        DataType::Float64 => Owned::f64(shape, values),
        DataType::Float16 => Owned::f16(shape, &f32s),
        DataType::BFloat16 => Owned::bf16(shape, &f32s),
        _ => unreachable!(),
    }
}
fn decode(value: &Owned) -> Vec<f64> {
    match value.dtype {
        DataType::Float32 => value.to_f32().into_iter().map(f64::from).collect(),
        DataType::Float64 => value.to_f64(),
        DataType::Float16 => value.to_f16_as_f32().into_iter().map(f64::from).collect(),
        DataType::BFloat16 => value.to_bf16_as_f32().into_iter().map(f64::from).collect(),
        _ => unreachable!(),
    }
}
fn round(dtype: DataType, v: f64) -> f64 {
    decode(&owned(dtype, &[1], &[v]))[0]
}
fn params(dtype: DataType) -> Vec<Owned> {
    [
        [1.25, -2., 0.5],
        [-0.25, 0.5, 1.],
        [0.25, -1., 2.],
        [4., 0.25, 1.],
    ]
    .iter()
    .map(|values| owned(dtype, &[3], values))
    .collect()
}
fn oracle(
    dtype: DataType,
    value: f64,
    scale: f64,
    bias: f64,
    mean: f64,
    variance: f64,
    epsilon: f32,
) -> f64 {
    if dtype == DataType::Float64 {
        ((value - mean) / (variance + f64::from(epsilon)).sqrt()) * scale + bias
    } else {
        let v = ((value as f32 - mean as f32) / (variance as f32 + epsilon).sqrt()) * scale as f32
            + bias as f32;
        round(dtype, f64::from(v))
    }
}

#[test]
fn all_storage_types_and_strided_parameters_match_original_arithmetic_bits() {
    for dtype in [
        DataType::Float16,
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float64,
    ] {
        for layout in 0..3 {
            let values = (0..24).map(|i| (i as f64 - 8.) * 0.25).collect::<Vec<_>>();
            let dense = owned(dtype, &[2, 3, 4], &values);
            let logical = decode(&dense);
            let step = if layout == 0 { 1 } else { 2 };
            let physical = values
                .iter()
                .flat_map(|&v| if step == 1 { vec![v] } else { vec![v, 123.] })
                .collect::<Vec<_>>();
            let x = owned(dtype, &[physical.len()], &physical)
                .with_view(&[2, 3, 4], &[12 * step, 4 * step, step]);
            let parameters = params(dtype);
            let p = parameters.iter().map(decode).collect::<Vec<_>>();
            let strided = parameters
                .iter()
                .map(|v| {
                    let mut values = decode(v);
                    if layout == 2 {
                        values.reverse();
                    }
                    let values = values
                        .into_iter()
                        .flat_map(|v| [v, 123.])
                        .collect::<Vec<_>>();
                    owned(dtype, &[6], &values).with_view(&[3], &[if layout == 2 { -2 } else { 2 }])
                })
                .collect::<Vec<_>>();
            let mut inputs = vec![x.view()];
            for i in 0..4 {
                let view = if layout == 0 {
                    parameters[i].view()
                } else {
                    strided[i].view()
                };
                inputs.push(if layout == 2 {
                    view.with_byte_offset(4 * dtype.byte_size())
                } else {
                    view
                });
            }
            let mut y = owned(dtype, &[48], &[123.; 48]).with_view(&[2, 3, 4], &[24, 8, 2]);
            BatchNormalization { epsilon: 0.125 }
                .execute_prepared(&inputs, &mut [y.view_mut()])
                .unwrap();
            let output = decode(&y);
            for i in 0..24 {
                let c = (i / 4) % 3;
                let want = oracle(dtype, logical[i], p[0][c], p[1][c], p[2][c], p[3][c], 0.125);
                assert_eq!(
                    output[2 * i].to_bits(),
                    want.to_bits(),
                    "dtype={dtype:?} layout={layout} i={i}"
                );
                assert_eq!(output[2 * i + 1], 123.);
            }
        }
    }
}

#[test]
fn every_parameter_can_alias_x_and_shifted_y_until_publication() {
    let parameters = params(DataType::Float64);
    for destination in [0, 8] {
        for parameter_start in [18, 28] {
            for aliased in 1..=5 {
                let mut backing = [123f64; 64];
                for (i, v) in backing[..24].iter_mut().enumerate() {
                    *v = (i as f64 - 9.) * 0.25;
                }
                let values = if aliased == 5 {
                    vec![1., 2., 3.]
                } else {
                    decode(&parameters[aliased - 1])
                };
                backing[parameter_start..parameter_start + 3].copy_from_slice(&values);
                let pointer = backing.as_mut_ptr();
                let mut inputs = vec![TensorView::new(
                    DevicePtr(pointer.cast()),
                    DataType::Float64,
                    &[2, 3, 4],
                    &[12, 4, 1],
                    DeviceId::cpu(),
                )];
                for i in 1..=4 {
                    let mut view = parameters[i - 1].view();
                    if aliased == i || aliased == 5 {
                        view.data = DevicePtr(unsafe { pointer.add(parameter_start).cast() });
                    }
                    inputs.push(view);
                }
                let source = crate::dtype::to_dense::<f64>(&inputs[0]).unwrap();
                let p = inputs[1..]
                    .iter()
                    .map(|v| crate::dtype::to_dense::<f64>(v).unwrap())
                    .collect::<Vec<_>>();
                let expected = source
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let c = (i / 4) % 3;
                        oracle(
                            DataType::Float64,
                            x,
                            p[0][c],
                            p[1][c],
                            p[2][c],
                            p[3][c],
                            0.125,
                        )
                    })
                    .collect::<Vec<_>>();
                let before = backing;
                let y = TensorMut::new(
                    DevicePtrMut(unsafe { pointer.add(destination).cast() }),
                    DataType::Float64,
                    &[2, 3, 4],
                    &[12, 4, 1],
                    DeviceId::cpu(),
                );
                BatchNormalization { epsilon: 0.125 }
                    .execute_prepared(&inputs, &mut [y])
                    .unwrap();
                for i in 0..64 {
                    let want = if (destination..destination + 24).contains(&i) {
                        expected[i - destination]
                    } else {
                        before[i]
                    };
                    assert_eq!(
                        backing[i].to_bits(),
                        want.to_bits(),
                        "destination={destination} param_start={parameter_start} aliased={aliased} i={i}"
                    );
                }
            }
        }
    }
}

#[test]
fn workspace_refusal_precedes_initialization_and_fresh_backing_is_valid() {
    let k = BatchNormalization { epsilon: 0.125 };
    let x = Owned::f64(&[2, 3, 4], &[2.; 24]);
    let p = params(DataType::Float64);
    let inputs = [x.view(), p[0].view(), p[1].view(), p[2].view(), p[3].view()];
    let mut y = Owned::f64(&[2, 3, 4], &[123.; 24]);
    let mut backing = [0xa5a5a5a5a5a5a5a5u64; 26];
    let pointer = unsafe { backing.as_mut_ptr().add(1).cast::<u8>() };
    for workspace in [
        None,
        Some(WorkspaceView::new(DevicePtrMut(pointer.cast()), 191)),
        Some(WorkspaceView::new(
            DevicePtrMut(unsafe { pointer.add(1).cast() }),
            192,
        )),
    ] {
        let error = k
            .execute_with_workspace(&inputs, &mut [y.view_mut()], workspace)
            .unwrap_err();
        assert!(error.to_string().contains("workspace"), "{error}");
        assert_eq!(y.to_f64(), [123.; 24]);
        assert_eq!(backing, [0xa5a5a5a5a5a5a5a5; 26]);
    }
    k.execute_with_workspace(
        &inputs,
        &mut [y.view_mut()],
        Some(WorkspaceView::new(DevicePtrMut(pointer.cast()), 192)),
    )
    .unwrap();
    assert_eq!(backing[0], 0xa5a5a5a5a5a5a5a5);
    assert_eq!(backing[25], 0xa5a5a5a5a5a5a5a5);
    let expected = y.to_f64();
    let mut fresh = [std::mem::MaybeUninit::<f64>::uninit(); 24];
    k.execute_with_workspace(
        &inputs,
        &mut [y.view_mut()],
        Some(WorkspaceView::new(
            DevicePtrMut(fresh.as_mut_ptr().cast()),
            192,
        )),
    )
    .unwrap();
    assert_eq!(y.to_f64(), expected);
}

#[test]
fn metadata_checks_complete_shapes_types_capacity_and_empty_batch() {
    let k = BatchNormalization { epsilon: 0.125 };
    let p = params(DataType::Float32);
    for shape in [&[2, 3, 4][..], &[0, 3, 4][..]] {
        let mut input = vec![TensorMetadata::new(DataType::Float32, shape, true)];
        input.extend((0..4).map(|_| TensorMetadata::new(DataType::Float32, &[3], true)));
        let output = [OutputMetadata::Unresolved {
            dtype: DataType::Float32,
        }];
        let requirement = k
            .workspace_requirement(&input, &output)
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(requirement.bytes, if shape[0] == 0 { 0 } else { 96 });
        for bad in [
            OutputMetadata::Absent,
            OutputMetadata::tensor(DataType::Float32, &[24]),
            OutputMetadata::Unresolved {
                dtype: DataType::Float64,
            },
        ] {
            assert!(
                k.workspace_requirement(&input, &[bad])
                    .unwrap_err()
                    .to_string()
                    .contains("Y")
            );
        }
        for case in 0..3 {
            let mut wrong = input.clone();
            wrong[2] = match case {
                0 => TensorMetadata::new(DataType::Float64, &[3], true),
                1 => TensorMetadata::new(DataType::Float32, &[1, 3], true),
                _ => TensorMetadata::new(DataType::Float32, &[3], false),
            };
            assert!(
                k.workspace_requirement(&wrong, &output)
                    .unwrap_err()
                    .to_string()
                    .contains("B must be present")
            );
        }
    }
    for shape in [
        &[1][..],
        &[0, 0, 4][..],
        &[2, 3, 0][..],
        &[2, usize::MAX, 2][..],
    ] {
        let mut input = vec![TensorMetadata::new(DataType::Float32, shape, true)];
        input.extend((0..4).map(|_| TensorMetadata::new(DataType::Float32, &[3], true)));
        assert!(
            k.workspace_requirement(
                &input,
                &[OutputMetadata::Unresolved {
                    dtype: DataType::Float32
                }]
            )
            .is_err()
        );
    }
    let x = Owned::f32(&[0, 3, 4], &[]);
    let mut y = Owned::zeros_f32(&[0, 3, 4]);
    k.execute(
        &[x.view(), p[0].view(), p[1].view(), p[2].view(), p[3].view()],
        &mut [y.view_mut()],
    )
    .unwrap();
}

#[test]
fn bad_host_and_address_proofs_refuse_before_any_scratch_or_output_write() {
    let k = BatchNormalization { epsilon: 0.125 };
    let p = params(DataType::Float64);
    let x = Owned::f64(&[2, 3, 4], &[2.; 24]);
    for case in 0..6 {
        let mut inputs = [x.view(), p[0].view(), p[1].view(), p[2].view(), p[3].view()];
        let mut y = Owned::f64(&[2, 3, 4], &[123.; 24]);
        let mut output = y.view_mut();
        let mut scratch = [0xa5a5a5a5a5a5a5a5u64; 24];
        match case {
            0 => {
                inputs[2].device = DeviceId::cuda(0);
                inputs[2].data = DevicePtr(4usize as *const _);
            }
            1 => inputs[3].strides = &[i64::MIN],
            2 => inputs[0].strides = &[i64::MAX, 4, 1],
            3 => output.data = DevicePtrMut((usize::MAX - 7) as *mut _),
            4 => inputs[4].byte_offset = usize::MAX - 7,
            _ => {
                output.device = DeviceId::cuda(0);
                output.data = DevicePtrMut(4usize as *mut _);
            }
        }
        let error = k
            .execute_with_workspace(
                &inputs,
                &mut [output],
                Some(WorkspaceView::new(
                    DevicePtrMut(scratch.as_mut_ptr().cast()),
                    192,
                )),
            )
            .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("Host") || message.contains("unaddressable"),
            "case={case}: {message}"
        );
        assert_eq!(y.to_f64(), [123.; 24]);
        assert_eq!(scratch, [0xa5a5a5a5a5a5a5a5; 24]);
    }
}

#[test]
fn channel_reader_preserves_unaligned_reversed_f64_and_singleton_stride() {
    let values = [16_777_217f64, 16_777_219., -0.];
    let mut backing = [0u128; 2];
    let pointer = backing.as_mut_ptr().cast::<u8>();
    for (i, v) in values.iter().rev().enumerate() {
        unsafe {
            pointer.add(1 + i * 8).cast::<f64>().write_unaligned(*v);
        }
    }
    let view = TensorView::new(
        DevicePtr(unsafe { pointer.add(1).cast() }),
        DataType::Float64,
        &[3],
        &[-1],
        DeviceId::cpu(),
    )
    .with_byte_offset(16);
    let reader = ChannelReader::<f64>::new("scale", &view, 3).unwrap();
    for (i, want) in values.iter().enumerate() {
        assert_eq!(reader.get(i).to_bits(), want.to_bits());
    }
    let view = TensorView::new(
        DevicePtr(unsafe { pointer.add(1).cast() }),
        DataType::Float64,
        &[1],
        &[i64::MIN],
        DeviceId::cpu(),
    );
    assert_eq!(
        ChannelReader::<f64>::new("B", &view, 1)
            .unwrap()
            .get(0)
            .to_bits(),
        (-0f64).to_bits()
    );
}

#[test]
fn one_retained_snapshot_has_no_warm_issuer_callbacks_and_trims_exactly() {
    use crate::kernels::testutil::RecordingGovernor;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_memory_api::Tier;
    use onnx_runtime_memory_governor::{
        HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
    };
    use std::sync::Arc;
    let books = Arc::new(RecordingGovernor::new_host(96));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources =
        crate::CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
            .unwrap()
            .with_workspace_pool(pool.clone())
            .unwrap();
    let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
    let k = BatchNormalization { epsilon: 0.125 };
    let p = params(DataType::Float32);
    let x = Owned::f32(&[2, 3, 4], &[2.; 24]);
    let mut y = Owned::zeros_f32(&[2, 3, 4]);
    let inputs = [x.view(), p[0].view(), p[1].view(), p[2].view(), p[3].view()];
    let meta = inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, true))
        .collect::<Vec<_>>();
    let facts = [OutputMetadata::tensor(DataType::Float32, &[2, 3, 4])];
    let request = k
        .workspace_requirement(&meta, &facts)
        .unwrap()
        .require_ready()
        .unwrap();
    assert_eq!(request.bytes, 96);
    let first = provider
        .allocate_workspace(96, request.alignment, request.role())
        .unwrap();
    let identity = first.managed_owner().unwrap().identity();
    assert!(
        provider
            .allocate_workspace(96, request.alignment, request.role())
            .is_err()
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
        let request = k
            .workspace_requirement_for_execution(&inputs, &[], &meta, &facts, None)
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(request.bytes, 96);
        let mut allocation = provider
            .allocate_workspace(96, request.alignment, request.role())
            .unwrap();
        assert_eq!(allocation.managed_owner().unwrap().identity(), identity);
        let workspace = WorkspaceView::new(DevicePtrMut(allocation.as_mut_ptr()), allocation.len());
        k.execute_with_workspace(&inputs, &mut [y.view_mut()], Some(workspace))
            .unwrap();
        provider.deallocate_workspace(allocation).unwrap();
    }
    assert_eq!(
        (books.claims(), books.releases()),
        calls,
        "including zero-byte callbacks"
    );
    assert_eq!(books.used(Tier::Host), 96);
    assert!(y.to_f32().iter().all(|v| v.is_finite()));
    pool.trim_idle(u64::MAX).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
}

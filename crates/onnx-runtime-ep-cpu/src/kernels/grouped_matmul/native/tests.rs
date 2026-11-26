use super::*;
use crate::kernels::testutil::{Owned, PreparedTestKernel, Repacked};
use onnx_runtime_ep_api::DevicePtrMut;
use onnx_runtime_ir::DeviceId;
fn fixture(dtype: DataType, transposed: bool, int32: bool) -> [Owned; 4] {
    let make = |shape: &[usize], data: &[f32]| match dtype {
        DataType::Float64 => Owned::f64(
            shape,
            &data.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
        ),
        DataType::Float16 => Owned::f16(shape, data),
        DataType::BFloat16 => Owned::bf16(shape, data),
        _ => Owned::f32(shape, data),
    };
    [
        make(&[2, 4], &[1., -2., 3., 4., 0.5, 2., -1., 3.]),
        make(
            if transposed { &[3, 3, 4] } else { &[3, 4, 3] },
            &(0..36)
                .map(|i| (i as f32 - 17.) * 0.125)
                .collect::<Vec<_>>(),
        ),
        if int32 {
            Owned::i32(&[2, 2], &[2, 0, 1, 2])
        } else {
            Owned::i64(&[2, 2], &[2, 0, 1, 2])
        },
        make(
            &[3, 3],
            &(0..9).map(|i| (i as f32 - 4.) * 0.125).collect::<Vec<_>>(),
        ),
    ]
}
fn op(transposed: bool) -> Native {
    Native(Op::GroupedMatMul { transposed })
}
fn output(dtype: DataType) -> Owned {
    let mut y = Owned::zeros_f32(&[2, 2, 3]);
    y.dtype = dtype;
    y.bytes = vec![0; 12 * dtype.byte_size()];
    y
}
fn request(k: &Native, inputs: &[TensorView], y: &TensorMut) -> WorkspaceRequirement {
    let m = metadata(inputs);
    k.workspace_requirement_for_execution(
        inputs,
        &[],
        &m[..inputs.len()],
        &[OutputMetadata::from_view(y)],
        None,
    )
    .unwrap()
    .require_ready()
    .unwrap()
}
fn bytes(values: usize, width: usize, routes: usize) -> u64 {
    ((values * width).next_multiple_of(align_of::<usize>()) + routes * size_of::<usize>()) as u64
}
// Original flattened snapshots/arithmetic, kept independent of Native's lane.
fn reference(inputs: &[TensorView], transposed: bool, y: &mut TensorMut) {
    let op = Op::GroupedMatMul { transposed };
    validate(op, inputs, std::slice::from_ref(y)).unwrap();
    let l = group_layout(op, inputs, y).unwrap();
    let groups = super::super::route_indices(op, &inputs[2], l.groups).unwrap();
    fn calc<F: Copy + Default + std::ops::Add<Output = F> + std::ops::Mul<Output = F>>(
        l: GroupLayout,
        x: &[F],
        w: &[F],
        r: &[usize],
        b: Option<&[F]>,
        round: impl Fn(F) -> F,
    ) -> Vec<F> {
        let mut y = vec![F::default(); l.lanes];
        for (route, row) in y.chunks_exact_mut(l.n).enumerate() {
            let g = r[route];
            let token = route / l.selections;
            for (n, out) in row.iter_mut().enumerate() {
                let mut sum = F::default();
                for k in 0..l.k {
                    let i = if l.transposed {
                        n * l.k + k
                    } else {
                        k * l.n + n
                    };
                    sum = sum + x[token * l.k + k] * w[g * l.k * l.n + i];
                }
                *out = if let Some(b) = b {
                    round(sum) + b[g * l.n + n]
                } else {
                    sum
                };
            }
        }
        y
    }
    if inputs[0].dtype == DataType::Float64 {
        let b = l.bias.then(|| dtype::to_dense::<f64>(&inputs[3]).unwrap());
        let x = dtype::to_dense::<f64>(&inputs[0]).unwrap();
        let w = dtype::to_dense::<f64>(&inputs[1]).unwrap();
        dtype::write_dense(y, &calc(l, &x, &w, &groups, b.as_deref(), |v| v)).unwrap();
    } else {
        let b = l
            .bias
            .then(|| dtype::to_dense_f32_widen(op.name(), &inputs[3]).unwrap());
        let x = dtype::to_dense_f32_widen(op.name(), &inputs[0]).unwrap();
        let w = dtype::to_dense_f32_widen(op.name(), &inputs[1]).unwrap();
        let values = calc(l, &x, &w, &groups, b.as_deref(), |v| {
            match inputs[0].dtype {
                DataType::Float16 => half::f16::from_f32(v).to_f32(),
                DataType::BFloat16 => half::bf16::from_f32(v).to_f32(),
                _ => v,
            }
        });
        dtype::write_dense_f32_narrow(op.name(), y, &values).unwrap();
    }
}
#[test]
fn all_float_types_transpose_bias_slots_and_source_layouts_preserve_bits() {
    for dt in [
        DataType::Float32,
        DataType::Float64,
        DataType::Float16,
        DataType::BFloat16,
    ] {
        for transposed in [false, true] {
            for int32 in [false, true] {
                for bias in 0..3 {
                    let f = fixture(dt, transposed, int32);
                    let k = op(transposed);
                    for changed in 0..5 {
                        for mode in 0..6 {
                            let storage = f
                                .iter()
                                .enumerate()
                                .map(|(i, v)| {
                                    Repacked::new(
                                        v,
                                        if changed == 4 || changed == i {
                                            mode
                                        } else {
                                            0
                                        },
                                    )
                                })
                                .collect::<Vec<_>>();
                            let mut inputs = storage.iter().map(Repacked::view).collect::<Vec<_>>();
                            if bias == 0 {
                                inputs.truncate(3);
                            } else if bias == 1 {
                                inputs[3] = TensorView::absent(DataType::Undefined);
                            }
                            let mut expected = output(dt);
                            reference(&inputs, transposed, &mut expected.view_mut());
                            let mut actual = output(dt);
                            k.execute_prepared(&inputs, &mut [actual.view_mut()])
                                .unwrap();
                            assert_eq!(
                                actual.bytes, expected.bytes,
                                "{dt:?}/{transposed}/{int32}/{bias}/{changed}/{mode}"
                            );
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn half_bias_rounds_matmul_before_add_and_double_keeps_precision() {
    for dt in [DataType::Float16, DataType::BFloat16] {
        let delta = if dt == DataType::Float16 {
            1. / 2048.
        } else {
            1. / 256.
        };
        let make = |s: &[usize], v: &[f32]| {
            if dt == DataType::Float16 {
                Owned::f16(s, v)
            } else {
                Owned::bf16(s, v)
            }
        };
        let f = [
            make(&[1, 2], &[1., delta]),
            make(&[1, 2, 1], &[1., 1.]),
            Owned::i64(&[1, 1], &[0]),
            make(&[1, 1], &[-1.]),
        ];
        let mut y = make(&[1, 1, 1], &[123.]);
        op(false)
            .execute_prepared(
                &f.iter().map(Owned::view).collect::<Vec<_>>(),
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_eq!(y.bytes, [0, 0]);
        assert_ne!(delta, 0.);
    }
    let f = [
        Owned::f64(&[1, 2], &[1., f64::EPSILON]),
        Owned::f64(&[1, 2, 1], &[1., 1.]),
        Owned::i64(&[1, 1], &[0]),
        Owned::f64(&[1, 1], &[-1.]),
    ];
    let mut y = Owned::f64(&[1, 1, 1], &[123.]);
    op(false)
        .execute_prepared(
            &f.iter().map(Owned::view).collect::<Vec<_>>(),
            &mut [y.view_mut()],
        )
        .unwrap();
    assert_eq!(y.to_f64(), [f64::EPSILON]);
}
#[test]
fn execution_refines_each_f32_borrow_without_snapshot_charges() {
    let k = op(false);
    let f = fixture(DataType::Float32, false, false);
    let mut y = output(DataType::Float32);
    for flags in 0..8 {
        let storage = f
            .iter()
            .enumerate()
            .map(|(i, v)| {
                Repacked::new(
                    v,
                    if [0, 1, 3]
                        .iter()
                        .position(|&j| j == i)
                        .is_some_and(|bit| flags & (1 << bit) != 0)
                    {
                        1
                    } else {
                        0
                    },
                )
            })
            .collect::<Vec<_>>();
        let inputs = storage.iter().map(Repacked::view).collect::<Vec<_>>();
        let m = metadata(&inputs);
        let out = [OutputMetadata::from_view(&y.view_mut())];
        assert_eq!(
            k.workspace_requirement(&m, &out).unwrap(),
            WorkspacePreparation::Deferred
        );
        let snapshots = [8, 36, 9]
            .into_iter()
            .enumerate()
            .filter(|(i, _)| flags & (1 << i) != 0)
            .map(|(_, n)| n)
            .sum::<usize>();
        assert_eq!(
            request(&k, &inputs, &y.view_mut()).bytes,
            bytes(12 + snapshots, 4, 4)
        );
    }
    for dt in [DataType::Float64, DataType::Float16, DataType::BFloat16] {
        let f = fixture(dt, false, false);
        let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
        let m = metadata(&inputs);
        let mut y = output(dt);
        let r = k
            .workspace_requirement(&m, &[OutputMetadata::from_view(&y.view_mut())])
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(
            r.bytes,
            bytes(
                12 + 8 + 36 + 9,
                if dt == DataType::Float64 { 8 } else { 4 },
                4
            )
        );
        assert_eq!(request(&k, &inputs, &y.view_mut()), r);
    }
}
#[test]
fn incomplete_lane_and_non_host_views_refuse_before_values_or_initialization() {
    for dt in [
        DataType::Float32,
        DataType::Float64,
        DataType::Float16,
        DataType::BFloat16,
    ] {
        let k = op(false);
        let f = fixture(dt, false, false);
        let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
        let mut y = output(dt);
        y.bytes.fill(0x3f);
        for view in &mut inputs {
            view.data = DevicePtr(16usize as *const _);
        }
        let r = request(&k, &inputs, &y.view_mut());
        let mut scratch = vec![0xaaaaaaaaaaaaaaaau128; (r.bytes as usize).div_ceil(16) + 1];
        let p = scratch.as_mut_ptr().cast::<u8>();
        for bad in [
            None,
            Some(WorkspaceView::new(
                DevicePtrMut(p.cast()),
                r.bytes as usize - 1,
            )),
            Some(WorkspaceView::new(
                DevicePtrMut(unsafe { p.add(1).cast() }),
                r.bytes as usize,
            )),
        ] {
            let error = k
                .execute_with_workspace(&inputs, &mut [y.view_mut()], bad)
                .unwrap_err()
                .to_string();
            assert!(error.contains("workspace needs"), "{error}");
            assert!(scratch.iter().all(|&x| x == 0xaaaaaaaaaaaaaaaau128));
            assert!(y.bytes.iter().all(|&x| x == 0x3f));
        }
        for index in 0..4 {
            let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
            inputs[index].device = DeviceId::cuda(0);
            let e = k
                .execute_prepared(&inputs, &mut [y.view_mut()])
                .unwrap_err()
                .to_string();
            assert!(e.contains("Host") || e.contains("host"), "{e}");
        }
        let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
        let mut out = y.view_mut();
        out.device = DeviceId::cuda(0);
        assert!(k.execute_prepared(&inputs, &mut [out]).is_err());
        let mut facts = metadata(&inputs);
        facts[0].shape = &[2, 3];
        assert!(
            k.workspace_requirement_for_execution(
                &inputs,
                &[],
                &facts,
                &[OutputMetadata::tensor(dt, &[2, 2, 3])],
                None
            )
            .is_err()
        );
    }
}
#[test]
fn all_sources_can_overlap_strided_y_without_touching_padding() {
    for dt in [
        DataType::Float32,
        DataType::Float64,
        DataType::Float16,
        DataType::BFloat16,
    ] {
        for transposed in [false, true] {
            let k = op(transposed);
            let f = fixture(dt, transposed, false);
            for source_index in 0..4 {
                for shift in [-8isize, 0, 8, 24] {
                    let mut storage = vec![0xccccccccccccccccccccccccccccccccu128; 64];
                    let base = storage.as_mut_ptr().cast::<u8>();
                    let start = 128usize;
                    let data = &f[source_index].bytes;
                    unsafe {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), base.add(start), data.len());
                    }
                    let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
                    inputs[source_index].data = DevicePtr(unsafe { base.add(start).cast() });
                    let mut expected = output(dt);
                    reference(&inputs, transposed, &mut expected.view_mut());
                    let origin = start.checked_add_signed(shift).unwrap();
                    let width = dt.byte_size();
                    let shape = [2, 2, 3];
                    let strides = [20, 8, -2];
                    let mut out = TensorMut::new(
                        DevicePtrMut(unsafe { base.add(origin).cast() }),
                        dt,
                        &shape,
                        &strides,
                        DeviceId::cpu(),
                    )
                    .with_byte_offset(4 * width);
                    let before = storage.clone();
                    k.execute_prepared(&inputs, std::slice::from_mut(&mut out))
                        .unwrap();
                    let got = match dt {
                        DataType::Float64 => dtype::to_dense::<f64>(
                            &TensorView::new(
                                DevicePtr(out.data.0.cast_const()),
                                dt,
                                &shape,
                                &strides,
                                DeviceId::cpu(),
                            )
                            .with_byte_offset(out.byte_offset),
                        )
                        .unwrap()
                        .iter()
                        .flat_map(|x| x.to_ne_bytes())
                        .collect::<Vec<_>>(),
                        _ => {
                            let view = TensorView::new(
                                DevicePtr(out.data.0.cast_const()),
                                dt,
                                &shape,
                                &strides,
                                DeviceId::cpu(),
                            )
                            .with_byte_offset(out.byte_offset);
                            let v = dtype::to_dense_f32_widen("test", &view).unwrap();
                            let mut owned = output(dt);
                            dtype::write_dense_f32_narrow("test", &mut owned.view_mut(), &v)
                                .unwrap();
                            owned.bytes
                        }
                    };
                    assert_eq!(
                        got, expected.bytes,
                        "{dt:?}/{transposed}/{source_index}/{shift}"
                    );
                    let touched = (0..2)
                        .flat_map(|i| {
                            (0..2).flat_map(move |j| {
                                (0..3).flat_map(move |k| {
                                    let at = origin + (4 + 20 * i + 8 * j - 2 * k) * width;
                                    at..at + width
                                })
                            })
                        })
                        .collect::<std::collections::HashSet<_>>();
                    let before = unsafe {
                        std::slice::from_raw_parts(before.as_ptr().cast::<u8>(), before.len() * 16)
                    };
                    let after = unsafe { std::slice::from_raw_parts(base, storage.len() * 16) };
                    for i in 0..after.len() {
                        if !touched.contains(&i) {
                            assert_eq!(after[i], before[i]);
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn empty_output_still_checks_routes_and_zero_k_preserves_bias_zero_signs() {
    let k = op(true);
    for bad in [false, true] {
        let x = Owned::f32(&[1, 2], &[1., 2.]);
        let w = Owned::f32(&[2, 0, 2], &[]);
        let r = Owned::i64(&[1, 1], &[if bad { -1 } else { 1 }]);
        let mut y = Owned::zeros_f32(&[1, 1, 0]);
        let mut inputs = [x.view(), w.view(), r.view()];
        for v in &mut inputs[..2] {
            v.data = DevicePtr(8usize as *const _);
            v.device = DeviceId::cuda(0);
        }
        assert_eq!(request(&k, &inputs, &y.view_mut()).bytes, bytes(0, 4, 1));
        let result = k.execute_prepared(&inputs, &mut [y.view_mut()]);
        if bad {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("group_indices[0] is -1")
            );
        } else {
            result.unwrap();
        }
    }
    let x = Owned::f32(&[1, 0], &[]);
    let w = Owned::f32(&[isize::MAX as usize, 5, 0], &[]);
    let r = Owned::i64(&[1, 1], &[isize::MAX as i64 - 1]);
    let mut y = Owned::zeros_f32(&[1, 1, 5]);
    k.execute_prepared(&[x.view(), w.view(), r.view()], &mut [y.view_mut()])
        .unwrap();
    assert!(y.to_f32().iter().all(|v| v.to_bits() == 0));
    for dt in [
        DataType::Float16,
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float64,
    ] {
        let mut f = fixture(dt, true, false);
        f[0].shape = vec![2, 0];
        f[0].bytes.clear();
        f[1].shape = vec![3, 3, 0];
        f[1].bytes.clear();
        let signed = [-0., 0., -0., 0., -0., 0., -0., 0., -0.];
        f[3] = match dt {
            DataType::Float64 => Owned::f64(&[3, 3], &signed.map(f64::from)),
            DataType::Float16 => Owned::f16(&[3, 3], &signed),
            DataType::BFloat16 => Owned::bf16(&[3, 3], &signed),
            _ => Owned::f32(&[3, 3], &signed),
        };
        let mut expected = output(dt);
        let mut y = output(dt);
        let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
        reference(&inputs, true, &mut expected.view_mut());
        k.execute_prepared(&inputs, &mut [y.view_mut()]).unwrap();
        assert_eq!(y.bytes, expected.bytes);
    }
}

#[test]
fn one_retained_native_workspace_has_no_warm_issuer_callbacks_and_trims_exactly() {
    use crate::kernels::testutil::RecordingGovernor;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_memory_api::Tier;
    use onnx_runtime_memory_governor::{
        HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
    };
    use std::sync::Arc;
    let expected = bytes(12, 4, 4) as usize;
    let books = Arc::new(RecordingGovernor::new_host(expected as u64));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources =
        crate::CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
            .unwrap()
            .with_workspace_pool(pool.clone())
            .unwrap();
    let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
    let k = op(false);
    let f = fixture(DataType::Float32, false, false);
    let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let mut y = output(DataType::Float32);
    let req = request(&k, &inputs, &y.view_mut());
    assert_eq!(req.bytes, expected as u64);
    let first = provider
        .allocate_workspace(expected, req.alignment, req.role())
        .unwrap();
    let identity = first.managed_owner().unwrap().identity();
    assert!(
        provider
            .allocate_workspace(expected, req.alignment, req.role())
            .is_err()
    );
    provider.deallocate_workspace(first).unwrap();
    let before = (books.claims(), books.releases());
    drop(
        books
            .reserve(Tier::Host, 0, req.role(), HolderId::new(99))
            .unwrap(),
    );
    books.probe_zero_release();
    assert_eq!(
        (books.claims(), books.releases()),
        (before.0 + 1, before.1 + 1)
    );
    let calls = (books.claims(), books.releases());
    for _ in 0..4096 {
        let req = request(&k, &inputs, &y.view_mut());
        assert_eq!(req.bytes, expected as u64);
        let mut allocation = provider
            .allocate_workspace(expected, req.alignment, req.role())
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
    assert_eq!(books.used(Tier::Host), expected as u64);
    assert!(y.to_f32().iter().all(|v| v.is_finite()));
    pool.trim_idle(u64::MAX).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
}

#[test]
fn bad_routes_and_stale_preparation_never_publish_or_read_float_sources() {
    let k = op(false);
    for bad in [-1, 3] {
        let mut f = fixture(DataType::Float32, false, false);
        f[2] = Owned::i64(&[2, 2], &[2, 0, bad, 2]);
        let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
        for i in [0, 1, 3] {
            inputs[i].data = DevicePtr(16usize as *const _);
        }
        let mut y = Owned::f32(&[2, 2, 3], &[123.; 12]);
        let error = k
            .execute_prepared(&inputs, &mut [y.view_mut()])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!("group_indices[2] is {bad}")),
            "{error}"
        );
        assert_eq!(y.to_f32(), [123.; 12]);
    }
    let f = fixture(DataType::Float32, false, false);
    let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let mut facts = metadata(&inputs);
    facts[0].shape = &[3, 4];
    facts[2].shape = &[3, 2];
    let error = k
        .workspace_requirement_for_execution(
            &inputs,
            &[],
            &facts,
            &[OutputMetadata::tensor(DataType::Float32, &[3, 2, 3])],
            None,
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("refresh metadata"), "{error}");
    let mut y = output(DataType::Float32);
    let mut inputs = inputs;
    inputs[0].data = DevicePtr((usize::MAX - 3) as *const _);
    let error = k
        .execute(&inputs, &mut [y.view_mut()])
        .unwrap_err()
        .to_string();
    assert!(error.contains("unaddressable span"), "{error}");
}

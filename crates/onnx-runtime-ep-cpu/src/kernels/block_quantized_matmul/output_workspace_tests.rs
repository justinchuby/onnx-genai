use super::*;
use crate::kernels::testutil::Owned;
use onnx_runtime_ep_api::DevicePtrMut;

fn kernel(arithmetic: ActivationArithmetic, dense: bool) -> BlockQuantizedMatMulKernel {
    BlockQuantizedMatMulKernel {
        k: 32,
        n: 2,
        format: BlockFormat::Q8_0,
        arithmetic,
        packed_b_constant: dense,
        weight_identity: DenseWeightIdentity::default(),
        prepared_projection: std::cell::Cell::new(None),
        weight_cache: WeightCache::new(Default::default(), if dense { 1 << 20 } else { 0 }),
    }
}

fn fixture() -> [Owned; 3] {
    let mut x = [0.; 64];
    x[0] = 127.;
    x[32] = -127.;
    let mut packed = vec![1u8; 68];
    for block in packed.chunks_exact_mut(34) {
        block[..2].copy_from_slice(&half::f16::ONE.to_le_bytes());
    }
    [
        Owned::f32(&[2, 32], &x),
        Owned::u8(&[2, 1, 34], &packed),
        Owned::f32(&[2], &[0.5, -0.25]),
    ]
}

fn prepare(k: &BlockQuantizedMatMulKernel, inputs: &[TensorView]) -> WorkspaceRequirement {
    let facts = inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
        .collect::<Vec<_>>();
    k.workspace_requirement_for_execution(inputs, &[], &facts, &[], None)
        .unwrap()
        .require_ready()
        .unwrap()
}

fn view(backing: &mut [u128]) -> WorkspaceView {
    WorkspaceView::new(
        DevicePtrMut(backing.as_mut_ptr().cast()),
        std::mem::size_of_val(backing),
    )
}

fn check(y: &Owned) {
    assert_eq!(y.to_f32(), [127.5, 126.75, -126.5, -127.25]);
}

#[test]
fn dense_hit_does_not_initialize_or_write_the_projection_region() {
    let k = kernel(ActivationArithmetic::Float32, true);
    let f = fixture();
    let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let req = prepare(&k, &inputs);
    let mut backing =
        vec![0xa5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5u128; (req.bytes as usize).div_ceil(16)];
    let (layout, _) = k
        .execution_workspace_layout(
            2,
            PackedProjectionPlan::for_current_pool(k.arithmetic),
            0,
            InputStaging::default(),
        )
        .unwrap();
    let mut y = Owned::f32(&[2, 2], &[37.; 4]);
    for call in 0..2 {
        prepare(&k, &inputs);
        let regions = layout.bind(Some(view(&mut backing))).unwrap();
        assert_ne!(
            regions[0].bytes(),
            0,
            "the canary must cover real streamed scratch"
        );
        // SAFETY: these disjoint regions are within the live exclusive backing;
        // no scratch owner or typed reference exists while the test writes bytes.
        unsafe {
            regions[1]
                .ptr()
                .0
                .cast::<u8>()
                .write_bytes(0xff, regions[1].bytes())
        };
        k.execute_with_workspace(&inputs, &mut [y.view_mut()], Some(view(&mut backing)))
            .unwrap();
        check(&y);
        // SAFETY: execution is synchronous and every canary byte was initialized.
        let projection = unsafe {
            std::slice::from_raw_parts(regions[0].ptr().0.cast::<u8>(), regions[0].bytes())
        };
        assert!(
            projection.iter().all(|&b| b == 0xa5),
            "dense call {call} touched streamed scratch"
        );
    }
    let (hits, builds) = k.weight_cache.activity();
    assert_eq!(builds, 1, "the second call must be a resident-cache hit");
    assert!(hits >= 1);
}

#[test]
fn streamed_and_q8_results_overwrite_dirty_workspace_on_every_dispatch() {
    for arithmetic in [ActivationArithmetic::Float32, ActivationArithmetic::Q8_0] {
        let k = kernel(arithmetic, false);
        let f = fixture();
        let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
        let req = prepare(&k, &inputs);
        let mut backing = vec![0u128; (req.bytes as usize).div_ceil(16)];
        let mut y = Owned::f32(&[2, 2], &[37.; 4]);
        for poison in [u128::MAX, 0xcccccccccccccccccccccccccccccccc] {
            backing.fill(poison);
            prepare(&k, &inputs);
            k.execute_with_workspace(&inputs, &mut [y.view_mut()], Some(view(&mut backing)))
                .unwrap();
            check(&y);
        }
        assert_eq!(k.weight_cache.activity(), (0, 0));
    }
}

#[test]
fn full_composite_binding_refuses_before_any_source_materialization() {
    for arithmetic in [ActivationArithmetic::Float32, ActivationArithmetic::Q8_0] {
        let k = kernel(arithmetic, true);
        let f = fixture();
        let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
        // Every descriptor has a representable span, but its payload is
        // inaccessible. Planning and complete binding must never read it.
        for input in &mut inputs {
            input.data = onnx_runtime_ep_api::DevicePtr(std::ptr::without_provenance(4096));
        }
        let req = prepare(&k, &inputs);
        let (layout, _) = k
            .execution_workspace_layout(
                2,
                PackedProjectionPlan::for_current_pool(arithmetic),
                0,
                InputStaging::default(),
            )
            .unwrap();
        let mut backing = vec![91u128; (req.bytes as usize).div_ceil(16)];
        let projection_bytes = layout.bind(Some(view(&mut backing))).unwrap()[0].bytes();
        for offer in 0..4 {
            prepare(&k, &inputs);
            let base = backing.as_mut_ptr().cast::<u8>();
            let workspace = match offer {
                0 => None,
                1 => Some(WorkspaceView::new(
                    DevicePtrMut(base.cast()),
                    req.bytes as usize - 1,
                )),
                2 => Some(WorkspaceView::new(
                    DevicePtrMut(base.cast()),
                    projection_bytes,
                )),
                // SAFETY: only an address is formed; binding must reject it.
                _ => Some(WorkspaceView::new(
                    DevicePtrMut(unsafe { base.add(1).cast() }),
                    req.bytes as usize,
                )),
            };
            let mut y = Owned::f32(&[2, 2], &[37.; 4]);
            let error = k
                .execute_with_workspace(&inputs, &mut [y.view_mut()], workspace)
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("BlockQuantizedMatMul execution workspace binding"),
                "{error}"
            );
            assert_eq!(y.to_f32(), [37.; 4]);
            assert!(backing.iter().all(|&v| v == 91));
            assert_eq!(k.weight_cache.activity(), (0, 0));
        }
    }
}

#[test]
fn partial_q8_conversion_failure_keeps_y_and_retries_with_a_fresh_plan() {
    let k = kernel(ActivationArithmetic::Q8_0, false);
    let f = fixture();
    let mut bad = [0.; 64];
    bad[0] = 127.;
    bad[32] = f32::NAN;
    let bad = Owned::f32(&[2, 32], &bad);
    let inputs = [bad.view(), f[1].view(), f[2].view()];
    let req = prepare(&k, &inputs);
    let mut backing = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
    let mut y = Owned::f32(&[2, 2], &[37.; 4]);
    let error = k
        .execute_with_workspace(&inputs, &mut [y.view_mut()], Some(view(&mut backing)))
        .unwrap_err();
    assert!(
        error.to_string().contains("grouped row 1 block 0"),
        "{error}"
    );
    assert_eq!(y.to_f32(), [37.; 4]);
    let good = f.iter().map(Owned::view).collect::<Vec<_>>();
    let error = k
        .execute_with_workspace(&good, &mut [y.view_mut()], Some(view(&mut backing)))
        .unwrap_err();
    assert!(error.to_string().contains("absent or consumed"), "{error}");
    assert_eq!(y.to_f32(), [37.; 4]);
    prepare(&k, &good);
    k.execute_with_workspace(&good, &mut [y.view_mut()], Some(view(&mut backing)))
        .unwrap();
    check(&y);
}

#[test]
fn output_publication_protects_every_source_under_exact_and_shifted_aliases() {
    use onnx_runtime_ep_api::DevicePtr;
    use onnx_runtime_ir::DeviceId;

    for (arithmetic, dense) in [
        (ActivationArithmetic::Float32, false),
        (ActivationArithmetic::Float32, true),
        (ActivationArithmetic::Q8_0, false),
    ] {
        for source in 0..3 {
            // Four-byte shifts overlap even the eight-byte bias source.
            for destination in [12usize, 16, 20] {
                let mut k = kernel(arithmetic, dense);
                // The packed source may itself be the mutable output backing.
                // Do not assert immutable-constant ownership for that fixture;
                // a fitting transient weight still exercises dense GEMM.
                k.packed_b_constant = false;
                let f = fixture();
                let mut backing = vec![
                    0xccccccccccccccccccccccccccccccccu128;
                    (f[source].bytes.len() + 128).div_ceil(16)
                ];
                let base = backing.as_mut_ptr().cast::<u8>();
                // SAFETY: source bytes and this new owner are disjoint; all raw
                // tensor ranges below remain within this live aligned owner.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        f[source].bytes.as_ptr(),
                        base.add(16),
                        f[source].bytes.len(),
                    );
                }
                let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
                inputs[source].data = DevicePtr(unsafe { base.add(16).cast() });
                let mut y = TensorMut::new(
                    DevicePtrMut(unsafe { base.add(destination).cast() }),
                    DataType::Float32,
                    &[2, 2],
                    &[2, 1],
                    DeviceId::cpu(),
                );
                let req = prepare(&k, &inputs);
                let mut scratch = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
                k.execute_with_workspace(
                    &inputs,
                    std::slice::from_mut(&mut y),
                    Some(view(&mut scratch)),
                )
                .unwrap();
                // No overlapping Rust slice references are formed: only raw
                // scalar reads after synchronous execution has finished.
                let actual = std::array::from_fn::<_, 4, _>(|index| unsafe {
                    base.add(destination + index * 4)
                        .cast::<f32>()
                        .read_unaligned()
                });
                assert_eq!(
                    actual,
                    [127.5, 126.75, -126.5, -127.25],
                    "{arithmetic:?} dense={dense} source={source} destination={destination}"
                );
            }
        }
    }
}

#[test]
fn checked_result_layout_and_empty_execution_keep_shape_guards() {
    for arithmetic in [ActivationArithmetic::Float32, ActivationArithmetic::Q8_0] {
        let k = kernel(arithmetic, false);
        let plan = PackedProjectionPlan::for_current_pool(arithmetic);
        for rows in [usize::MAX, usize::MAX / 2] {
            let error = k
                .execution_workspace_layout(rows, plan, 0, InputStaging::default())
                .err()
                .unwrap();
            assert!(
                error.to_string().contains("output staging"),
                "result count/byte overflow must fail before projection sizing: {error}"
            );
        }
        let f = fixture();
        let empty = Owned::f32(&[0, 32], &[]);
        let inputs = [empty.view(), f[1].view(), f[2].view()];
        let req = prepare(&k, &inputs);
        assert_eq!(req.bytes, 0, "zero rows must need no admitted backing");
        let mut y = Owned::zeros_f32(&[0, 2]);
        k.execute_with_workspace(&inputs, &mut [y.view_mut()], None)
            .unwrap();
        let mut unreadable = inputs;
        unreadable[1].data = onnx_runtime_ep_api::DevicePtr(std::ptr::without_provenance(4096));
        unreadable[1].strides = &[68, 68, 2];
        assert_eq!(prepare(&k, &unreadable).bytes, 0);
        k.execute_with_workspace(&unreadable, &mut [y.view_mut()], None)
            .unwrap();
        let (base, _) = k
            .execution_workspace_layout(2, plan, 0, InputStaging::default())
            .unwrap();
        let (rounded, _) = k
            .execution_workspace_layout(2, plan, 69, InputStaging::default())
            .unwrap();
        assert_eq!(rounded.bytes() - base.bytes(), 72);
        assert!(
            k.execution_workspace_layout(2, plan, usize::MAX, InputStaging::default())
                .is_err()
        );

        // Equal zero element counts do not permit dropping the configured N
        // axis. This must report geometry, rather than a workspace error.
        prepare(&k, &inputs);
        let mut wrong_empty = Owned::zeros_f32(&[0]);
        let error = k
            .execute_with_workspace(&inputs, &mut [wrong_empty.view_mut()], None)
            .unwrap_err();
        assert!(error.to_string().contains("Y must have shape"), "{error}");

        // The allocation-free shape comparison must also reject nonempty
        // equal-numel rank mismatches before touching the output.
        let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
        prepare(&k, &inputs);
        let mut wrong_rank = Owned::f32(&[4], &[37.; 4]);
        let error = k
            .execute_with_workspace(&inputs, &mut [wrong_rank.view_mut()], None)
            .unwrap_err();
        assert!(error.to_string().contains("Y must have shape"), "{error}");
        assert_eq!(wrong_rank.to_f32(), [37.; 4]);
    }
}

#[test]
fn partial_projection_failure_keeps_y_after_staging_was_written() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let k = kernel(ActivationArithmetic::Q8_0, false);
        let x = Owned::f32(&[1, 32], &[127.; 32]);
        let mut packed = vec![1u8; 68];
        packed[..2].copy_from_slice(&half::f16::ONE.to_le_bytes());
        packed[34..36].copy_from_slice(&half::f16::INFINITY.to_le_bytes());
        let bad = Owned::u8(&[2, 1, 34], &packed);
        let inputs = [x.view(), bad.view()];
        let req = prepare(&k, &inputs);
        let mut backing = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
        let (layout, _) = k
            .execution_workspace_layout(
                1,
                PackedProjectionPlan::for_current_pool(k.arithmetic),
                0,
                InputStaging::default(),
            )
            .unwrap();
        let regions = layout.bind(Some(view(&mut backing))).unwrap();
        let mut y = Owned::f32(&[1, 2], &[37.; 2]);
        let error = k
            .execute_with_workspace(&inputs, &mut [y.view_mut()], Some(view(&mut backing)))
            .unwrap_err();
        assert!(
            error.to_string().contains("weight output 1 block 0"),
            "{error}"
        );
        assert_eq!(y.to_f32(), [37.; 2]);
        // SAFETY: result prefix was initialized by this dispatch. With one
        // worker, feature 0 completes before feature 1 rejects its scale.
        assert_eq!(unsafe { regions[1].ptr().0.cast::<f32>().read() }, 4064.);
        packed[34..36].copy_from_slice(&half::f16::ONE.to_le_bytes());
        let good = Owned::u8(&[2, 1, 34], &packed);
        let inputs = [x.view(), good.view()];
        prepare(&k, &inputs);
        k.execute_with_workspace(&inputs, &mut [y.view_mut()], Some(view(&mut backing)))
            .unwrap();
        assert_eq!(y.to_f32(), [4064.; 2]);
    });
}

#[test]
fn packed_strides_refine_each_launch_and_dense_hits_leave_byte_lane_untouched() {
    for (arithmetic, dense) in [
        (ActivationArithmetic::Float32, false),
        (ActivationArithmetic::Float32, true),
        (ActivationArithmetic::Q8_0, false),
    ] {
        let k = kernel(arithmetic, dense);
        let f = fixture();
        let physical = f[1]
            .bytes
            .iter()
            .flat_map(|&b| [b, 0xa5])
            .collect::<Vec<_>>();
        let packed = Owned::u8(&[136], &physical);
        let mut strided = packed.view();
        strided.shape = &[2, 1, 34];
        strided.strides = &[68, 68, 2];
        let inputs = [f[0].view(), strided, f[2].view()];
        let facts = inputs
            .iter()
            .map(|v| TensorMetadata::new(v.dtype, v.shape, true))
            .collect::<Vec<_>>();
        let cached = k.workspace_requirement(&facts, &[]).unwrap();
        assert!(matches!(
            cached,
            onnx_runtime_ep_api::WorkspacePreparation::Deferred
        ));
        let mut y = Owned::f32(&[2, 2], &[37.; 4]);
        let mut contiguous_bytes = None;
        for (iteration, is_strided) in [false, true, true, false].into_iter().enumerate() {
            let views = [
                f[0].view(),
                if is_strided { strided } else { f[1].view() },
                f[2].view(),
            ];
            let req = k
                .workspace_requirement_for_execution(&views, &[], &facts, &[], Some(cached))
                .unwrap()
                .require_ready()
                .unwrap();
            let base = *contiguous_bytes.get_or_insert(req.bytes);
            assert_eq!(req.bytes, base + if is_strided { 68 } else { 0 });
            let mut backing =
                vec![0xa5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5; (req.bytes as usize).div_ceil(16)];
            let (layout, _) = k
                .execution_workspace_layout(
                    2,
                    PackedProjectionPlan::for_current_pool(arithmetic),
                    if is_strided { 68 } else { 0 },
                    InputStaging::default(),
                )
                .unwrap();
            let regions = layout.bind(Some(view(&mut backing))).unwrap();
            let (hits, builds) = k.weight_cache.activity();
            k.execute_with_workspace(&views, &mut [y.view_mut()], Some(view(&mut backing)))
                .unwrap();
            check(&y);
            if dense && iteration == 2 {
                assert_eq!(
                    k.weight_cache.activity().1,
                    builds,
                    "repeated strided source must hit"
                );
                assert!(k.weight_cache.activity().0 > hits);
                // SAFETY: the entire backing began initialized and execution ended.
                let bytes = unsafe {
                    std::slice::from_raw_parts(regions[2].ptr().0.cast::<u8>(), regions[2].bytes())
                };
                assert_eq!(bytes, &[0xa5; 68]);
            }
        }
        // A failed refinement must revoke the previous launch plan.
        prepare(&k, &inputs);
        assert!(
            k.workspace_requirement_for_execution(&inputs[..1], &[], &facts, &[], Some(cached))
                .unwrap_err()
                .to_string()
                .contains("packed_B")
        );
        assert!(k.prepared_projection.get().is_none());
    }
}

#[test]
fn short_composite_rejects_before_reading_strided_packed_payload() {
    use onnx_runtime_ep_api::DevicePtr;
    let k = kernel(ActivationArithmetic::Float32, true);
    let f = fixture();
    let mut packed = f[1].view();
    packed.data = DevicePtr(std::ptr::without_provenance(4096));
    packed.strides = &[68, 68, 2];
    let inputs = [f[0].view(), packed, f[2].view()];
    let req = prepare(&k, &inputs);
    let mut backing = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
    let mut y = Owned::f32(&[2, 2], &[37.; 4]);
    let workspace = WorkspaceView::new(
        DevicePtrMut(backing.as_mut_ptr().cast()),
        req.bytes as usize - 1,
    );
    assert!(
        k.execute_with_workspace(&inputs, &mut [y.view_mut()], Some(workspace))
            .unwrap_err()
            .to_string()
            .contains("workspace binding")
    );
    assert_eq!(y.to_f32(), [37.; 4]);
    assert_eq!(k.weight_cache.activity(), (0, 0));
    assert!(backing.iter().all(|&v| v == u128::MAX));
}

#[test]
fn strided_packed_source_can_overlap_final_output_in_either_direction() {
    use onnx_runtime_ep_api::DevicePtr;
    use onnx_runtime_ir::DeviceId;
    for (arithmetic, dense) in [
        (ActivationArithmetic::Float32, false),
        (ActivationArithmetic::Float32, true),
        (ActivationArithmetic::Q8_0, false),
    ] {
        for destination in [12, 16, 20] {
            let mut k = kernel(arithmetic, dense);
            k.packed_b_constant = false;
            let f = fixture();
            let physical = f[1]
                .bytes
                .iter()
                .flat_map(|&b| [b, 0xa5])
                .collect::<Vec<_>>();
            let mut owner = vec![0u128; 16];
            let base = owner.as_mut_ptr().cast::<u8>();
            // SAFETY: both full ranges fit the live aligned owner; no typed
            // source/output references coexist with these raw tensor views.
            unsafe {
                std::ptr::copy_nonoverlapping(physical.as_ptr(), base.add(16), physical.len());
            }
            let mut packed = f[1].view();
            packed.data = DevicePtr(unsafe { base.add(16).cast() });
            packed.strides = &[68, 68, 2];
            let inputs = [f[0].view(), packed, f[2].view()];
            let mut y = TensorMut::new(
                DevicePtrMut(unsafe { base.add(destination).cast() }),
                DataType::Float32,
                &[2, 2],
                &[2, 1],
                DeviceId::cpu(),
            );
            let req = prepare(&k, &inputs);
            let mut backing = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
            k.execute_with_workspace(
                &inputs,
                std::slice::from_mut(&mut y),
                Some(view(&mut backing)),
            )
            .unwrap();
            let actual = std::array::from_fn::<_, 4, _>(|i| unsafe {
                base.add(destination + i * 4).cast::<f32>().read_unaligned()
            });
            assert_eq!(actual, [127.5, 126.75, -126.5, -127.25]);
        }
    }
}

fn float_owner(dtype: DataType, shape: &[usize], values: &[f32]) -> Owned {
    match dtype {
        DataType::Float32 => Owned::f32(shape, values),
        DataType::Float16 => Owned::f16(shape, values),
        DataType::BFloat16 => Owned::bf16(shape, values),
        _ => unreachable!("test dtype"),
    }
}

#[test]
fn input_conversion_regions_cover_dtype_strides_alignment_and_dirty_reuse() {
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for geometry in 0..4 {
            for (arithmetic, dense) in [
                (ActivationArithmetic::Float32, false),
                (ActivationArithmetic::Float32, true),
                (ActivationArithmetic::Q8_0, false),
            ] {
                let k = kernel(arithmetic, dense);
                let f = fixture();
                let mut av = f[0].to_f32();
                let mut bv = f[2].to_f32();
                if geometry == 1 {
                    av = av.into_iter().flat_map(|v| [v, f32::NAN]).collect();
                    bv = bv.into_iter().flat_map(|v| [v, f32::NAN]).collect();
                }
                if geometry == 2 {
                    av.reverse();
                    bv.reverse();
                }
                let mut a = float_owner(dtype, &[av.len()], &av);
                let mut b = float_owner(dtype, &[bv.len()], &bv);
                if geometry == 3 {
                    a.bytes.insert(0, 0xa5);
                    b.bytes.insert(0, 0xa5);
                }
                let (mut a_view, mut b_view) = (a.view(), b.view());
                a_view.shape = &[2, 32];
                b_view.shape = &[2];
                a_view.strides = match geometry {
                    1 => &[64, 2],
                    2 => &[-32, -1],
                    _ => &[32, 1],
                };
                b_view.strides = match geometry {
                    1 => &[2],
                    2 => &[-1],
                    _ => &[1],
                };
                let width = if dtype == DataType::Float32 { 4 } else { 2 };
                if geometry == 2 {
                    a_view.byte_offset = 63 * width;
                    b_view.byte_offset = width;
                }
                if geometry == 3 {
                    // SAFETY: each owner has one extra leading byte; the complete
                    // payload remains inside its live allocation after shifting.
                    a_view.data =
                        onnx_runtime_ep_api::DevicePtr(unsafe { a.bytes.as_ptr().add(1).cast() });
                    b_view.data =
                        onnx_runtime_ep_api::DevicePtr(unsafe { b.bytes.as_ptr().add(1).cast() });
                }
                let inputs = [a_view, f[1].view(), b_view];
                let req = prepare(&k, &inputs);
                let plan = PackedProjectionPlan::for_current_pool(arithmetic);
                let (base, _) = k
                    .execution_workspace_layout(2, plan, 0, InputStaging::default())
                    .unwrap();
                let extra = if dtype == DataType::Float32 && geometry == 0 {
                    0
                } else {
                    (64 + 2) * 4
                };
                assert_eq!(req.bytes, base.bytes() as u64 + extra);
                let mut backing = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
                let mut y = float_owner(dtype, &[2, 2], &[37.; 4]);
                let expected = float_owner(dtype, &[2, 2], &[127.5, 126.75, -126.5, -127.25]).bytes;
                for poison in [u128::MAX, 0xcccccccccccccccccccccccccccccccc] {
                    backing.fill(poison);
                    prepare(&k, &inputs);
                    let before = k.weight_cache.activity();
                    k.execute_with_workspace(
                        &inputs,
                        &mut [y.view_mut()],
                        Some(view(&mut backing)),
                    )
                    .unwrap();
                    assert_eq!(
                        y.bytes, expected,
                        "{dtype:?} geometry={geometry} arithmetic={arithmetic:?} dense={dense}"
                    );
                    if dense && before.1 != 0 {
                        assert_eq!(k.weight_cache.activity().1, before.1);
                        assert!(k.weight_cache.activity().0 > before.0);
                        let untouched = base.bind(Some(view(&mut backing))).unwrap()[0];
                        // SAFETY: backing was initialized to poison; the dense
                        // hit must preserve every decode-scratch byte.
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                untouched.ptr().0.cast::<u8>(),
                                untouched.bytes(),
                            )
                        };
                        assert_eq!(bytes, vec![poison.to_ne_bytes()[0]; untouched.bytes()]);
                    }
                }
            }
        }
    }
}

#[test]
fn short_input_conversion_composite_precedes_inaccessible_payload_reads() {
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        let k = kernel(ActivationArithmetic::Float32, true);
        let f = fixture();
        let a = float_owner(dtype, &[2, 32], &[0.; 64]);
        let b = float_owner(dtype, &[2], &[0.; 2]);
        let mut inputs = [a.view(), f[1].view(), b.view()];
        inputs[0].data = onnx_runtime_ep_api::DevicePtr(std::ptr::without_provenance(4096));
        inputs[0].strides = &[64, 2];
        inputs[2].data = onnx_runtime_ep_api::DevicePtr(std::ptr::without_provenance(8192));
        inputs[2].strides = &[2];
        let req = prepare(&k, &inputs);
        let mut backing = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
        let mut y = float_owner(dtype, &[2, 2], &[37.; 4]);
        let workspace = WorkspaceView::new(
            DevicePtrMut(backing.as_mut_ptr().cast()),
            req.bytes as usize - 1,
        );
        assert!(
            k.execute_with_workspace(&inputs, &mut [y.view_mut()], Some(workspace))
                .unwrap_err()
                .to_string()
                .contains("workspace binding")
        );
        assert_eq!(y.bytes, float_owner(dtype, &[2, 2], &[37.; 4]).bytes);
        assert!(backing.iter().all(|&v| v == u128::MAX));
        assert_eq!(k.weight_cache.activity(), (0, 0));
        // Empty output still validates Host descriptors, but never touches the
        // inaccessible activation/bias payload or requests conversion storage.
        inputs[0].shape = &[0, 32];
        assert_eq!(prepare(&k, &inputs).bytes, 0);
        let mut empty = float_owner(dtype, &[0, 2], &[]);
        k.execute_with_workspace(&inputs, &mut [empty.view_mut()], None)
            .unwrap();
        let metadata = inputs
            .iter()
            .map(|v| TensorMetadata::new(v.dtype, v.shape, true))
            .collect::<Vec<_>>();
        inputs[2].device = onnx_runtime_ir::DeviceId::cuda(0);
        let error = k
            .workspace_requirement_for_execution(&inputs, &[], &metadata, &[], None)
            .unwrap_err();
        assert!(
            error.to_string().contains("bias") && error.to_string().contains("Host"),
            "{error}"
        );
        assert!(k.prepared_projection.get().is_none());
    }
}

#[test]
fn converted_inputs_keep_exact_and_shifted_output_aliases_safe() {
    use onnx_runtime_ep_api::DevicePtr;
    use onnx_runtime_ir::DeviceId;
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        let width = if dtype == DataType::Float32 { 4 } else { 2 };
        for (arithmetic, dense) in [
            (ActivationArithmetic::Float32, false),
            (ActivationArithmetic::Float32, true),
            (ActivationArithmetic::Q8_0, false),
        ] {
            for source in [0, 2] {
                for destination in [16 - width, 16, 16 + width] {
                    let k = kernel(arithmetic, dense);
                    let f = fixture();
                    let av = f[0]
                        .to_f32()
                        .into_iter()
                        .flat_map(|v| [v, f32::NAN])
                        .collect::<Vec<_>>();
                    let bv = f[2]
                        .to_f32()
                        .into_iter()
                        .flat_map(|v| [v, f32::NAN])
                        .collect::<Vec<_>>();
                    let owners = [
                        float_owner(dtype, &[2, 32], &av),
                        Owned::u8(&[2, 1, 34], &f[1].bytes),
                        float_owner(dtype, &[2], &bv),
                    ];
                    let mut backing = vec![0u128; (owners[source].bytes.len() + 64).div_ceil(16)];
                    let base = backing.as_mut_ptr().cast::<u8>();
                    // SAFETY: source/destination ranges fit the owner. Raw views
                    // are used until execution ends; final read occurs afterward.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            owners[source].bytes.as_ptr(),
                            base.add(16),
                            owners[source].bytes.len(),
                        );
                    }
                    let mut inputs = owners.iter().map(Owned::view).collect::<Vec<_>>();
                    inputs[0].strides = &[64, 2];
                    inputs[2].strides = &[2];
                    inputs[source].data = DevicePtr(unsafe { base.add(16).cast() });
                    let mut y = TensorMut::new(
                        DevicePtrMut(unsafe { base.add(destination).cast() }),
                        dtype,
                        &[2, 2],
                        &[2, 1],
                        DeviceId::cpu(),
                    );
                    let req = prepare(&k, &inputs);
                    let mut scratch = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
                    k.execute_with_workspace(
                        &inputs,
                        std::slice::from_mut(&mut y),
                        Some(view(&mut scratch)),
                    )
                    .unwrap();
                    let actual =
                        unsafe { std::slice::from_raw_parts(base.add(destination), 4 * width) };
                    assert_eq!(
                        actual,
                        float_owner(dtype, &[2, 2], &[127.5, 126.75, -126.5, -127.25]).bytes
                    );
                }
            }
        }
    }
}

#[test]
fn cached_metadata_refines_actual_input_layout_and_failed_views_revoke_the_plan() {
    let k = kernel(ActivationArithmetic::Q8_0, false);
    let f = fixture();
    let av = f[0]
        .to_f32()
        .into_iter()
        .flat_map(|v| [v, f32::NAN])
        .collect::<Vec<_>>();
    let a = Owned::f32(&[av.len()], &av);
    let mut strided = a.view();
    strided.shape = &[2, 32];
    strided.strides = &[64, 2];
    let inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let metadata = inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, true))
        .collect::<Vec<_>>();
    let cached = k.workspace_requirement(&metadata, &[]).unwrap();
    let base = prepare(&k, &inputs).bytes;
    for use_strided in [false, true, false, true] {
        let inputs = [
            if use_strided { strided } else { f[0].view() },
            f[1].view(),
            f[2].view(),
        ];
        let req = k
            .workspace_requirement_for_execution(&inputs, &[], &metadata, &[], Some(cached))
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(req.bytes, base + if use_strided { 64 * 4 } else { 0 });
        let mut backing = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
        let mut y = Owned::zeros_f32(&[2, 2]);
        k.execute_with_workspace(&inputs, &mut [y.view_mut()], Some(view(&mut backing)))
            .unwrap();
        check(&y);
    }
    prepare(&k, &inputs);
    let mut invalid = inputs;
    invalid[0].byte_offset = usize::MAX;
    assert!(
        k.workspace_requirement_for_execution(&invalid, &[], &metadata, &[], Some(cached))
            .unwrap_err()
            .to_string()
            .contains("A")
    );
    assert!(k.prepared_projection.get().is_none());
}

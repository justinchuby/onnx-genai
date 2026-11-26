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
        .execution_workspace_layout(2, PackedProjectionPlan::for_current_pool(k.arithmetic))
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
        // Any attempted read would fail address validation. Geometry remains
        // valid so the complete workspace error must have priority instead.
        for input in &mut inputs {
            input.byte_offset = usize::MAX;
        }
        let req = prepare(&k, &inputs);
        let (layout, _) = k
            .execution_workspace_layout(2, PackedProjectionPlan::for_current_pool(arithmetic))
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
            let error = k.execution_workspace_layout(rows, plan).err().unwrap();
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
            .execution_workspace_layout(1, PackedProjectionPlan::for_current_pool(k.arithmetic))
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

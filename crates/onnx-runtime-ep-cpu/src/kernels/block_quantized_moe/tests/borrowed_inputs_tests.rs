use super::*;
use crate::kernels::testutil::Repacked;
use onnx_runtime_ep_api::DevicePtr;
use onnx_runtime_ir::DeviceId;

const ROWS: usize = 2;
const FLOAT_SLOTS: [usize; 6] = [0, 1, 3, 5, 7, 8];

fn fixture() -> [Owned; 9] {
    let identity = || {
        let mut bytes = vec![0u8; E * H * 34];
        for expert in 0..E {
            for feature in 0..H {
                let block = &mut bytes[(expert * H + feature) * 34..][..34];
                block[..2].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
                block[2 + feature] = 1;
            }
        }
        Owned::u8(&[E, H, 1, 34], &bytes)
    };
    let x = (0..ROWS * H)
        .map(|i| {
            if i % H == 0 {
                127.
            } else {
                (i % 7) as f32 / 8. + 0.5
            }
        })
        .collect::<Vec<_>>();
    let bias = |offset: f32| {
        Owned::f32(
            &[E, H],
            &(0..E * H)
                .map(|i| offset + (i % 5) as f32 / 64.)
                .collect::<Vec<_>>(),
        )
    };
    [
        Owned::f32(&[ROWS, H], &x),
        Owned::f32(&[ROWS, E], &[1., -1., -0.5, 0.5]),
        identity(),
        bias(0.25),
        identity(),
        bias(-0.125),
        identity(),
        bias(0.5),
        Owned::f32(&[ROWS, E], &[0.75, 1.25, 1.5, 0.5]),
    ]
}

fn kernel(
    f: &[Owned; 9],
    arithmetic: ActivationArithmetic,
    dense: bool,
) -> BlockQuantizedMoEKernel {
    let mut spec = with_fc3_format(attrs("swiglu", 2, true, 0), "q8_0");
    for (name, value) in &mut spec {
        match *name {
            "fc1_format" | "fc2_format" => *value = Attribute::String(b"q8_0".to_vec()),
            "block_layout_version" => *value = Attribute::Int(arithmetic.layout_version()),
            _ => {}
        }
    }
    spec.push((
        "activation_arithmetic",
        Attribute::String(arithmetic.name().as_bytes().to_vec()),
    ));
    let shapes = f
        .iter()
        .map(|v| {
            let view = v.view();
            Some((view.dtype, view.shape.to_vec()))
        })
        .collect::<Vec<_>>();
    let (graph, node) = model_node(&shapes, &spec);
    let ValidatedMetadata {
        arithmetic,
        attributes,
        formats,
    } = validate_metadata(graph.node(node), None).unwrap();
    // Only the immutable packed banks are constant. Aliasing tests below never
    // mutate them; all six floating inputs remain ordinary runtime values.
    let mut constant_inputs = [false; 9];
    for slot in [2, 4, 6] {
        constant_inputs[slot] = true;
    }
    BlockQuantizedMoEKernel {
        arithmetic,
        attributes,
        formats,
        constant_inputs,
        weight_identities: std::array::from_fn(|_| DenseWeightIdentity::default()),
        prepared_projection: std::cell::Cell::new(None),
        weight_cache: WeightCache::new(Default::default(), if dense { 1 << 20 } else { 0 }),
    }
}

fn expected(f: &[Owned; 9], arithmetic: ActivationArithmetic, dense: bool) -> Vec<u32> {
    let k = kernel(f, arithmetic, dense);
    let views = f.iter().map(Owned::view).collect::<Vec<_>>();
    let mut y = Owned::zeros_f32(&[ROWS, H]);
    k.execute_prepared(&views, &mut [y.view_mut()]).unwrap();
    assert!(y.to_f32().iter().all(|v| v.is_finite()));
    if arithmetic == ActivationArithmetic::Float32 && dense {
        assert_eq!(k.weight_cache.stats().0, E * 3);
    } else {
        assert_eq!(k.weight_cache.activity(), (0, 0));
    }
    y.to_f32().into_iter().map(f32::to_bits).collect()
}

#[test]
fn every_floating_input_can_alias_y_exactly_or_in_either_shift_direction() {
    for (arithmetic, dense) in [
        (ActivationArithmetic::Float32, false),
        (ActivationArithmetic::Float32, true),
        (ActivationArithmetic::Q8_0, false),
    ] {
        let f = fixture();
        let reference = expected(&f, arithmetic, dense);
        for strided in [false, true] {
            for source in FLOAT_SLOTS {
                for destination in [12usize, 16, 20] {
                    let k = kernel(&f, arithmetic, dense);
                    let payload = if strided {
                        f[source]
                            .bytes
                            .chunks_exact(4)
                            .flat_map(|v| v.iter().copied().chain([0xff; 4]))
                            .collect::<Vec<_>>()
                    } else {
                        f[source].bytes.clone()
                    };
                    let source_strides = f[source]
                        .strides
                        .iter()
                        .map(|&s| s * if strided { 2 } else { 1 })
                        .collect::<Vec<_>>();
                    let mut backing = vec![
                        0xccccccccccccccccccccccccccccccccu128;
                        (payload.len().max(ROWS * H * 4) + 128).div_ceil(16)
                    ];
                    let base = backing.as_mut_ptr().cast::<u8>();
                    // SAFETY: this live owner is disjoint from the original fixture;
                    // every source/output range is contained in its allocation.
                    unsafe {
                        std::ptr::copy_nonoverlapping(payload.as_ptr(), base.add(16), payload.len())
                    };
                    let mut views = f.iter().map(Owned::view).collect::<Vec<_>>();
                    views[source].data = DevicePtr(unsafe { base.add(16).cast() });
                    views[source].strides = &source_strides;
                    let mut y = TensorMut::new(
                        DevicePtrMut(unsafe { base.add(destination).cast() }),
                        DataType::Float32,
                        &[ROWS, H],
                        &[H as i64, 1],
                        DeviceId::cpu(),
                    );
                    k.execute_prepared(&views, std::slice::from_mut(&mut y))
                        .unwrap();
                    // No overlapping Rust slices are formed; after synchronous
                    // publication, observe output bits through raw scalar reads.
                    let actual = (0..ROWS * H)
                        .map(|i| unsafe {
                            base.add(destination + i * 4)
                                .cast::<f32>()
                                .read_unaligned()
                                .to_bits()
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        actual, reference,
                        "{arithmetic:?} dense={dense} source={source} destination={destination} strided={strided}"
                    );
                }
            }
        }
    }
}

#[test]
fn noncontiguous_and_unaligned_float_sources_match_unaliased_dense_inputs() {
    for (arithmetic, dense) in [
        (ActivationArithmetic::Float32, false),
        (ActivationArithmetic::Float32, true),
        (ActivationArithmetic::Q8_0, false),
    ] {
        let f = fixture();
        let reference = expected(&f, arithmetic, dense);
        for source in FLOAT_SLOTS {
            for mode in 1..=5 {
                let k = kernel(&f, arithmetic, dense);
                let repacked = Repacked::new(&f[source], mode);
                let mut views = f.iter().map(Owned::view).collect::<Vec<_>>();
                views[source] = repacked.view();
                let mut y = Owned::zeros_f32(&[ROWS, H]);
                k.execute_prepared(&views, &mut [y.view_mut()]).unwrap();
                assert_eq!(
                    y.to_f32().into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                    reference,
                    "{arithmetic:?} dense={dense} source={source} layout={mode}"
                );
            }
        }
    }
}

#[test]
fn every_float_slot_keeps_dtype_and_host_guards_without_output_publication() {
    for arithmetic in [ActivationArithmetic::Float32, ActivationArithmetic::Q8_0] {
        let f = fixture();
        for source in FLOAT_SLOTS {
            for nonhost in [false, true] {
                let k = kernel(&f, arithmetic, false);
                let mut views = f.iter().map(Owned::view).collect::<Vec<_>>();
                if nonhost {
                    views[source].device = DeviceId::cuda(0);
                    views[source].data = DevicePtr(std::ptr::dangling::<u128>().cast());
                } else {
                    views[source].dtype = DataType::Float16;
                }
                let mut y = Owned::f32(&[ROWS, H], &[37.; ROWS * H]);
                let error = k.execute_prepared(&views, &mut [y.view_mut()]).unwrap_err();
                let message = error.to_string();
                assert!(
                    if nonhost {
                        message.to_ascii_lowercase().contains("host-accessible")
                    } else {
                        message.contains("Float32")
                    },
                    "slot={source}, nonhost={nonhost}: {message}"
                );
                if nonhost {
                    assert!(message.contains("explicitly transfer"), "{message}");
                }
                assert_eq!(y.to_f32(), [37.; ROWS * H]);
                // A rejected call must not leave a consumable old plan or
                // poison the kernel: the ordinary preparation seam can retry.
                let good = f.iter().map(Owned::view).collect::<Vec<_>>();
                k.execute_prepared(&good, &mut [y.view_mut()]).unwrap();
                assert!(y.to_f32().iter().all(|v| v.is_finite()));
            }
        }
    }
}

fn refine_inputs(
    k: &BlockQuantizedMoEKernel,
    inputs: &[TensorView],
    metadata: &[TensorMetadata],
    cached: Option<onnx_runtime_ep_api::WorkspacePreparation>,
) -> WorkspaceRequirement {
    k.workspace_requirement_for_execution(inputs, &[], metadata, &[], cached)
        .unwrap()
        .require_ready()
        .unwrap()
}

#[test]
fn floating_workspace_tracks_actual_layouts_across_cached_metadata_and_dirty_reuse() {
    for (arithmetic, dense) in [
        (ActivationArithmetic::Float32, false),
        (ActivationArithmetic::Float32, true),
        (ActivationArithmetic::Q8_0, false),
    ] {
        let f = fixture();
        let reference = expected(&f, arithmetic, dense);
        for source in FLOAT_SLOTS {
            let k = kernel(&f, arithmetic, dense);
            let original = f.iter().map(Owned::view).collect::<Vec<_>>();
            let metadata = original
                .iter()
                .map(|v| TensorMetadata::new(v.dtype, v.shape, true))
                .collect::<Vec<_>>();
            let cached = k.workspace_requirement(&metadata, &[]).unwrap();
            let base = refine_inputs(&k, &original, &metadata, Some(cached)).bytes;
            for mode in [0, 1, 2, 3, 4, 5, 0] {
                let repacked = Repacked::new(&f[source], mode);
                let mut inputs = original.clone();
                if mode != 0 {
                    inputs[source] = repacked.view();
                }
                let req = refine_inputs(&k, &inputs, &metadata, Some(cached));
                assert_eq!(
                    req.bytes,
                    base + if mode == 0 {
                        0
                    } else {
                        f[source].bytes.len() as u64
                    },
                    "source={source} mode={mode}"
                );
                let mut backing = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
                let mut y = Owned::zeros_f32(&[ROWS, H]);
                k.execute_with_workspace(
                    &inputs,
                    &mut [y.view_mut()],
                    Some(WorkspaceView::new(
                        onnx_runtime_ep_api::DevicePtrMut(backing.as_mut_ptr().cast()),
                        req.bytes as usize,
                    )),
                )
                .unwrap();
                assert_eq!(
                    y.to_f32().into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                    reference
                );
            }
            // A missing actual optional/required input cannot become a zero
            // conversion lane merely because cached metadata is available.
            refine_inputs(&k, &original, &metadata, Some(cached));
            let mut missing = original;
            missing[source] = TensorView::absent(DataType::Float32);
            let error = k
                .workspace_requirement_for_execution(&missing, &[], &metadata, &[], Some(cached))
                .unwrap_err();
            assert!(error.to_string().contains(INPUT_NAMES[source]), "{error}");
            assert!(k.prepared_projection.get().is_none());
        }
    }
}

#[test]
fn short_floating_workspace_rejects_before_conversion_routing_or_hash_reads() {
    let f = fixture();
    let k = kernel(&f, ActivationArithmetic::Float32, true);
    let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    let strides = f
        .iter()
        .map(|v| v.strides.iter().map(|s| s * 2).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    for slot in FLOAT_SLOTS {
        inputs[slot].data = DevicePtr(std::ptr::without_provenance(4096));
        inputs[slot].strides = &strides[slot];
    }
    let metadata = inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, true))
        .collect::<Vec<_>>();
    let req = refine_inputs(&k, &inputs, &metadata, None);
    let mut backing = vec![91u128; (req.bytes as usize).div_ceil(16)];
    let mut y = Owned::f32(&[ROWS, H], &[37.; ROWS * H]);
    let workspace = WorkspaceView::new(
        onnx_runtime_ep_api::DevicePtrMut(backing.as_mut_ptr().cast()),
        req.bytes as usize - 1,
    );
    assert!(
        k.execute_with_workspace(&inputs, &mut [y.view_mut()], Some(workspace))
            .unwrap_err()
            .to_string()
            .contains("workspace binding")
    );
    assert_eq!(y.to_f32(), [37.; ROWS * H]);
    assert!(backing.iter().all(|&v| v == 91));
    assert_eq!(k.weight_cache.activity(), (0, 0));
    for identity in &k.weight_identities {
        assert_eq!(identity.stats(), (0, 0));
    }
}

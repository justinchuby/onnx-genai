use super::*;
use crate::kernels::testutil::Repacked;
use onnx_runtime_ep_api::{DevicePtr, WorkspacePreparation, WorkspaceRequirement};
use onnx_runtime_ir::DeviceId;

const ROWS: usize = 2;

fn fixture(experts: usize) -> [Owned; 9] {
    let bank = |q4: bool| {
        let width = if q4 { 18 } else { 34 };
        let mut bytes = vec![if q4 { 0x88 } else { 0 }; experts * H * width];
        for expert in 0..experts {
            for feature in 0..H {
                let block = &mut bytes[(expert * H + feature) * width..][..width];
                block[..2].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
                if q4 {
                    block[2 + feature % 16] += if feature < 16 { 1 } else { 16 };
                } else {
                    block[2 + feature] = 1;
                }
            }
        }
        Owned::u8(&[experts, H, 1, width], &bytes)
    };
    [
        Owned::f32(&[ROWS, H], &[127.; ROWS * H]),
        Owned::f32(&[ROWS, experts], &vec![0.; ROWS * experts]),
        bank(true),
        Owned::f32(&[experts, H], &vec![0.25; experts * H]),
        bank(false),
        Owned::f32(&[experts, H], &vec![-0.125; experts * H]),
        bank(true),
        Owned::f32(&[experts, H], &vec![0.5; experts * H]),
        Owned::f32(&[ROWS, experts], &vec![1.; ROWS * experts]),
    ]
}

fn kernel(
    f: &[Owned; 9],
    topk: usize,
    arithmetic: ActivationArithmetic,
    dense: bool,
) -> BlockQuantizedMoEKernel {
    let mut spec = with_fc3_format(attrs("swiglu", topk, true, 0), "gguf:q4_0");
    for (name, value) in &mut spec {
        match *name {
            "fc1_format" => *value = Attribute::String(b"gguf:q4_0".to_vec()),
            "fc2_format" => *value = Attribute::String(b"q8_0".to_vec()),
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
        .map(|v| Some((v.dtype, v.shape.clone())))
        .collect::<Vec<_>>();
    let (graph, node) = model_node(&shapes, &spec);
    let ValidatedMetadata {
        arithmetic,
        attributes,
        formats,
    } = validate_metadata(graph.node(node), None).unwrap();
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

fn facts<'a>(inputs: &[TensorView<'a>]) -> Vec<TensorMetadata<'a>> {
    inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
        .collect()
}
fn refine(
    k: &BlockQuantizedMoEKernel,
    inputs: &[TensorView],
    metadata: &[TensorMetadata],
) -> WorkspaceRequirement {
    k.workspace_requirement_for_execution(
        inputs,
        &[],
        metadata,
        &[],
        Some(WorkspacePreparation::Deferred),
    )
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
fn execute(
    k: &BlockQuantizedMoEKernel,
    inputs: &[TensorView],
    req: WorkspaceRequirement,
) -> Vec<u32> {
    let mut backing = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
    let mut y = Owned::zeros_f32(&[ROWS, H]);
    k.execute_with_workspace(inputs, &mut [y.view_mut()], Some(view(&mut backing)))
        .unwrap();
    assert!(y.to_f32().iter().all(|v| v.is_finite()));
    y.to_f32().into_iter().map(f32::to_bits).collect()
}

#[test]
fn packed_copy_bound_is_one_projection_maximum_not_a_sum_or_route_multiplier() {
    let mut common = None;
    for experts in [2, 4] {
        for topk in [1, experts] {
            let f = fixture(experts);
            let k = kernel(&f, topk, ActivationArithmetic::Float32, false);
            let repacked = [2, 4, 6].map(|i| Repacked::new(&f[i], 2));
            let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
            let metadata = facts(&inputs);
            assert_eq!(
                k.workspace_requirement(&metadata, &[]).unwrap(),
                WorkspacePreparation::Deferred
            );
            let dims = validate_runtime_shapes(&metadata, None, &k.attributes, k.formats).unwrap();
            for (slot, source) in [2, 4, 6].into_iter().zip(&repacked) {
                inputs[slot] = source.view();
            }
            assert_eq!(k.packed_lane_bytes(dims, &inputs).unwrap(), H * 34);
            assert!(H * 34 < H * (18 + 34 + 18));
            let req = refine(&k, &inputs, &metadata);
            let base = k
                .execution_workspace_layout(
                    dims,
                    PackedProjectionPlan::for_current_pool(k.arithmetic),
                    0,
                )
                .unwrap();
            assert_eq!(req.bytes as usize - base.bytes(), H * 34);
            assert_eq!(
                *common.get_or_insert(req.bytes),
                req.bytes,
                "expert count and top-k do not multiply this lane"
            );
            inputs[4] = f[4].view();
            assert_eq!(k.packed_lane_bytes(dims, &inputs).unwrap(), H * 18);
        }
    }
}

struct OuterBank {
    bytes: Vec<u8>,
    shape: [usize; 4],
    strides: [i64; 4],
    offset: usize,
}
impl OuterBank {
    fn new(source: &Owned, reverse: bool) -> Self {
        let shape: [usize; 4] = source.shape.as_slice().try_into().unwrap();
        let expert_bytes = shape[1] * shape[2] * shape[3];
        let step = expert_bytes + 16;
        let mut bytes = vec![0xcc; shape[0] * step];
        for expert in 0..shape[0] {
            let physical = if reverse {
                shape[0] - 1 - expert
            } else {
                expert
            };
            bytes[physical * step..physical * step + expert_bytes]
                .copy_from_slice(&source.bytes[expert * expert_bytes..(expert + 1) * expert_bytes]);
        }
        Self {
            bytes,
            shape,
            strides: [
                if reverse { -(step as i64) } else { step as i64 },
                (shape[2] * shape[3]) as i64,
                shape[3] as i64,
                1,
            ],
            offset: if reverse { (shape[0] - 1) * step } else { 0 },
        }
    }
    fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(self.bytes.as_ptr().cast()),
            DataType::Uint8,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
        .with_byte_offset(self.offset)
    }
}

#[test]
fn contiguous_expert_tails_borrow_even_when_the_outer_bank_is_padded_or_reversed() {
    for arithmetic in [ActivationArithmetic::Float32, ActivationArithmetic::Q8_0] {
        let f = fixture(2);
        let k = kernel(&f, 2, arithmetic, false);
        let original = f.iter().map(Owned::view).collect::<Vec<_>>();
        let metadata = facts(&original);
        let baseline = refine(&k, &original, &metadata);
        let expected = execute(&k, &original, baseline);
        for reverse in [false, true] {
            let banks = [2, 4, 6].map(|i| OuterBank::new(&f[i], reverse));
            let mut inputs = original.clone();
            for (slot, bank) in [2, 4, 6].into_iter().zip(&banks) {
                inputs[slot] = bank.view();
            }
            let dims = validate_runtime_shapes(&metadata, None, &k.attributes, k.formats).unwrap();
            assert_eq!(k.packed_lane_bytes(dims, &inputs).unwrap(), 0);
            let req = refine(&k, &inputs, &metadata);
            assert_eq!(req, baseline);
            assert_eq!(execute(&k, &inputs, req), expected);
        }
    }
}

#[test]
fn cached_deferred_rechecks_current_strides_in_both_directions() {
    for arithmetic in [ActivationArithmetic::Float32, ActivationArithmetic::Q8_0] {
        let f = fixture(2);
        let k = kernel(&f, 2, arithmetic, false);
        let original = f.iter().map(Owned::view).collect::<Vec<_>>();
        let metadata = facts(&original);
        let cached = k.workspace_requirement(&metadata, &[]).unwrap();
        assert_eq!(cached, WorkspacePreparation::Deferred);
        let banks = [2, 4, 6].map(|i| Repacked::new(&f[i], 2));
        let mut strided = original.clone();
        for (slot, bank) in [2, 4, 6].into_iter().zip(&banks) {
            strided[slot] = bank.view();
        }
        let mut expected = None;
        let mut base = None;
        for (index, inputs) in [&original, &strided, &original].into_iter().enumerate() {
            let req = k
                .workspace_requirement_for_execution(inputs, &[], &metadata, &[], Some(cached))
                .unwrap()
                .require_ready()
                .unwrap();
            if index == 0 {
                base = Some(req.bytes);
            }
            assert_eq!(
                req.bytes,
                base.unwrap() + if index == 1 { (H * 34) as u64 } else { 0 }
            );
            let actual = execute(&k, inputs, req);
            assert_eq!(*expected.get_or_insert_with(|| actual.clone()), actual);
        }
    }
}

#[test]
fn logically_present_missing_bank_rejects_before_admission_and_clears_old_plan() {
    let f = fixture(2);
    let k = kernel(&f, 2, ActivationArithmetic::Float32, false);
    let original = f.iter().map(Owned::view).collect::<Vec<_>>();
    let metadata = facts(&original);
    for missing in [2, 4, 6] {
        let req = refine(&k, &original, &metadata);
        assert!(k.prepared_projection.get().is_some());
        let mut inputs = original.clone();
        inputs[missing] = TensorView::absent(DataType::Uint8);
        let error = k
            .workspace_requirement_for_execution(
                &inputs,
                &[],
                &metadata,
                &[],
                Some(WorkspacePreparation::Deferred),
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("no readable packed bank"),
            "{error}"
        );
        assert!(
            error.to_string().contains(&format!("input {missing}")),
            "{error}"
        );
        assert!(k.prepared_projection.get().is_none());
        let mut y = Owned::f32(&[ROWS, H], &[37.; ROWS * H]);
        let error = k
            .execute_with_workspace(&original, &mut [y.view_mut()], None)
            .unwrap_err();
        assert!(error.to_string().contains("absent or consumed"), "{error}");
        assert_eq!(y.to_f32(), [37.; ROWS * H]);
        assert_eq!(refine(&k, &original, &metadata), req);
        execute(&k, &original, req);
    }
}

#[test]
fn dense_cache_hits_leave_the_noncontiguous_copy_lane_untouched() {
    let f = fixture(2);
    let k = kernel(&f, 2, ActivationArithmetic::Float32, true);
    let banks = [2, 4, 6].map(|i| Repacked::new(&f[i], 2));
    let mut inputs = f.iter().map(Owned::view).collect::<Vec<_>>();
    for (slot, bank) in [2, 4, 6].into_iter().zip(&banks) {
        inputs[slot] = bank.view();
    }
    let metadata = facts(&inputs);
    let req = refine(&k, &inputs, &metadata);
    let expected = execute(&k, &inputs, req);
    assert_eq!(k.weight_cache.stats().0, 2 * 3);
    let activity = k.weight_cache.activity();
    let dims = validate_runtime_shapes(&metadata, None, &k.attributes, k.formats).unwrap();
    let layout = k
        .execution_workspace_layout(
            dims,
            PackedProjectionPlan::for_current_pool(k.arithmetic),
            H * 34,
        )
        .unwrap();
    let mut backing =
        vec![0xa5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5u128; (req.bytes as usize).div_ceil(16)];
    let regions = layout.bind(Some(view(&mut backing))).unwrap();
    assert_eq!(regions[4].bytes(), H * 34);
    refine(&k, &inputs, &metadata);
    let mut y = Owned::zeros_f32(&[ROWS, H]);
    k.execute_with_workspace(&inputs, &mut [y.view_mut()], Some(view(&mut backing)))
        .unwrap();
    assert_eq!(
        y.to_f32().into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        expected
    );
    // SAFETY: execution is synchronous and this precise region remains within
    // the live initialized backing. No scratch loan survives the call.
    let bytes =
        unsafe { std::slice::from_raw_parts(regions[4].ptr().0.cast::<u8>(), regions[4].bytes()) };
    assert!(bytes.iter().all(|&byte| byte == 0xa5));
    assert_eq!(k.weight_cache.activity().1, activity.1);
    assert!(k.weight_cache.activity().0 > activity.0);
}

#[test]
fn nonconstant_packed_banks_can_overlap_y_until_final_publication() {
    for (arithmetic, dense) in [
        (ActivationArithmetic::Float32, false),
        (ActivationArithmetic::Float32, true),
        (ActivationArithmetic::Q8_0, false),
    ] {
        let f = fixture(2);
        let reference_kernel = kernel(&f, 2, arithmetic, dense);
        let original = f.iter().map(Owned::view).collect::<Vec<_>>();
        let metadata = facts(&original);
        let expected = execute(
            &reference_kernel,
            &original,
            refine(&reference_kernel, &original, &metadata),
        );
        for source in [2, 4, 6] {
            for step in [1usize, 2] {
                for destination in [12usize, 16, 20] {
                    let mut k = kernel(&f, 2, arithmetic, dense);
                    // Y is permitted to overwrite packed source bytes after
                    // consumption; no immutable-slot promise is made here.
                    k.constant_inputs = [false; 9];
                    let mut backing = vec![
                        0xccccccccccccccccccccccccccccccccu128;
                        (f[source].bytes.len() * step + 128).div_ceil(16)
                    ];
                    let base = backing.as_mut_ptr().cast::<u8>();
                    for (index, &byte) in f[source].bytes.iter().enumerate() {
                        // SAFETY: all physical source bytes are within the
                        // fresh owner, which is disjoint from the fixture.
                        unsafe { base.add(16 + index * step).write(byte) };
                    }
                    let strides = f[source]
                        .strides
                        .iter()
                        .map(|s| s * step as i64)
                        .collect::<Vec<_>>();
                    let mut inputs = original.clone();
                    inputs[source] = TensorView::new(
                        DevicePtr(unsafe { base.add(16).cast() }),
                        DataType::Uint8,
                        &f[source].shape,
                        &strides,
                        DeviceId::cpu(),
                    );
                    let req = refine(&k, &inputs, &metadata);
                    let mut scratch = vec![u128::MAX; (req.bytes as usize).div_ceil(16)];
                    let mut y = TensorMut::new(
                        DevicePtrMut(unsafe { base.add(destination).cast() }),
                        DataType::Float32,
                        &[ROWS, H],
                        &[H as i64, 1],
                        DeviceId::cpu(),
                    );
                    k.execute_with_workspace(
                        &inputs,
                        std::slice::from_mut(&mut y),
                        Some(view(&mut scratch)),
                    )
                    .unwrap();
                    // SAFETY: execution is synchronous. Read scalar output
                    // bits without constructing overlapping Rust slices.
                    let actual = (0..ROWS * H)
                        .map(|index| unsafe {
                            base.add(destination + index * 4)
                                .cast::<f32>()
                                .read_unaligned()
                                .to_bits()
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        actual, expected,
                        "{arithmetic:?} dense={dense} source={source} step={step} destination={destination}"
                    );
                }
            }
        }
    }
}

#[test]
fn incomplete_composite_refuses_before_any_packed_hash_or_copy() {
    for (arithmetic, dense) in [
        (ActivationArithmetic::Float32, false),
        (ActivationArithmetic::Float32, true),
        (ActivationArithmetic::Q8_0, false),
    ] {
        let f = fixture(2);
        let k = kernel(&f, 2, arithmetic, dense);
        let original = f.iter().map(Owned::view).collect::<Vec<_>>();
        let metadata = facts(&original);
        let strides =
            [2, 4, 6].map(|slot| f[slot].strides.iter().map(|s| s * 2).collect::<Vec<_>>());
        let mut inputs = original.clone();
        for (slot, strides) in [2, 4, 6].into_iter().zip(&strides) {
            // Nonnull, representable descriptor; actually reading it would be
            // invalid. No payload access may precede full workspace binding.
            inputs[slot].data = DevicePtr(std::ptr::dangling::<u128>().cast());
            inputs[slot].strides = strides;
        }
        let req = refine(&k, &inputs, &metadata);
        let dims = validate_runtime_shapes(&metadata, None, &k.attributes, k.formats).unwrap();
        let without_packed = k
            .execution_workspace_layout(dims, PackedProjectionPlan::for_current_pool(arithmetic), 0)
            .unwrap()
            .bytes();
        assert!(without_packed < req.bytes as usize);
        let mut backing = vec![91u128; (req.bytes as usize).div_ceil(16)];
        for bytes in [None, Some(without_packed), Some(req.bytes as usize - 1)] {
            refine(&k, &inputs, &metadata);
            let workspace =
                bytes.map(|len| WorkspaceView::new(DevicePtrMut(backing.as_mut_ptr().cast()), len));
            let mut y = Owned::f32(&[ROWS, H], &[37.; ROWS * H]);
            let error = k
                .execute_with_workspace(&inputs, &mut [y.view_mut()], workspace)
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("BlockQuantizedMoE execution workspace binding"),
                "{error}"
            );
            assert_eq!(y.to_f32(), [37.; ROWS * H]);
            assert!(backing.iter().all(|&v| v == 91));
            assert_eq!(k.weight_cache.activity(), (0, 0));
            for identity in &k.weight_identities {
                assert_eq!(
                    identity.stats(),
                    (0, 0),
                    "no content key may be hashed before admission"
                );
            }
        }
    }
}

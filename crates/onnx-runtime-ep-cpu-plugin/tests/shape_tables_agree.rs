//! The two shape-inference tables must agree.
//!
//! # Why there are two
//!
//! `onnx-runtime-shape-inference` serves the **native** session: it runs at
//! graph-build time over types and shapes, and mints a symbolic dimension
//! (`ctx.fresh_dim()`) for an extent that only an input's *values* fix. The
//! native executor is happy with that — `resolve_soft` omits a still-symbolic
//! value and the run loop sizes its buffer just-in-time.
//!
//! `onnx-runtime-ep-plugin`'s `ShapeInference` serves the **ORT plugin** path,
//! which cannot defer: the shape handed to `KernelContext_GetOutput` decides the
//! allocation, so it must be a concrete `Vec<usize>` *before* the kernel runs.
//! It compensates by running at Compute time, where it holds real `TensorView`s
//! and can read the values the native table can only name.
//!
//! So neither table subsumes the other, and deleting either one is not the fix.
//!
//! # Why they must still be checked against each other
//!
//! What *is* duplicated is the ONNX semantics of each operator — "`Tile`
//! multiplies each dim by its repeat", "`Expand` broadcasts bidirectionally".
//! Two independent encodings of one specification drift, and drift here is
//! silent: the native path and the plugin path would simply disagree about the
//! same graph, and whichever one a user exercised would look correct.
//!
//! This test pins them together on the cases where both can answer concretely.
//! It is deliberately narrow — every op below is one whose plugin rule was
//! written by reading the CPU kernel, and the point is to confirm that reading
//! against the independently-written native rule.

use std::collections::HashMap;

use onnx_runtime_ep_api::{DevicePtr, TensorView};
use onnx_runtime_ep_plugin::compute::ShapeInference;
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, ValueId};
use onnx_runtime_shape_inference::{
    DimExpr, InferenceRegistry, MergePolicy, NodeIo, ShapeData, SymbolInterner, TypeInfo,
};

/// Run the native table over one node.
fn native_try(
    node: &Node,
    inputs: Vec<NodeIo>,
    opset: u64,
) -> Result<Vec<NodeIo>, onnx_runtime_shape_inference::ShapeInferError> {
    let reg = InferenceRegistry::default_registry();
    let mut imports = HashMap::new();
    imports.insert(String::new(), opset);
    let mut interner = SymbolInterner::new(0x8000_0000);
    reg.infer_node(
        node,
        &imports,
        inputs,
        MergePolicy::Permissive,
        &mut interner,
    )
}

fn native(node: &Node, inputs: Vec<NodeIo>, opset: u64) -> Vec<NodeIo> {
    native_try(node, inputs, opset).expect("native inference should not error on these nodes")
}

/// The native answer as concrete extents, or `None` if any dim stayed symbolic.
fn native_static(outs: &[NodeIo]) -> Option<Vec<usize>> {
    let ty = outs.first()?.type_info.as_ref()?;
    ty.shape
        .iter()
        .map(|d| d.as_const().and_then(|n| usize::try_from(n).ok()))
        .collect()
}

fn typed(dtype: DataType, dims: &[i64]) -> NodeIo {
    NodeIo::typed(TypeInfo::new(
        dtype,
        dims.iter().map(|&d| DimExpr::constant(d)).collect(),
    ))
}

/// A `NodeIo` that also carries its integer *values*, which is how the native
/// table resolves a value-carried extent instead of minting a symbol.
fn typed_with_values(dims: &[i64], values: &[i64]) -> NodeIo {
    let mut io = typed(DataType::Int64, dims);
    io.shape_data = Some(ShapeData {
        dtype: DataType::Int64,
        dims: dims.iter().map(|&d| d as usize).collect(),
        elems: values.iter().map(|&v| DimExpr::constant(v)).collect(),
        float_elems: None,
    });
    io
}

fn node(op: &str, n_inputs: usize, attrs: &[(&str, i64)], opset: u64) -> Node {
    let inputs: Vec<Option<ValueId>> = (0..n_inputs).map(|i| Some(ValueId(i as u32))).collect();
    let mut n = Node::new(NodeId(0), op, inputs, vec![ValueId(100)]);
    n.version = Some(opset as i64);
    for (k, v) in attrs {
        n.attributes.insert((*k).to_string(), Attribute::Int(*v));
    }
    n
}

// ── Plugin-side driving ──────────────────────────────────────────────────────

fn view<'a>(
    dtype: DataType,
    shape: &'a [usize],
    strides: &'a [i64],
    data: *const u8,
) -> TensorView<'a> {
    TensorView::new(
        DevicePtr(data as *mut std::ffi::c_void),
        dtype,
        shape,
        strides,
        onnx_runtime_ir::DeviceId::cpu(),
    )
}

/// Both tables, same node, same inputs — assert they agree.
///
/// `plugin_inputs` carries real bytes because the plugin rule reads values;
/// `native_inputs` carries the same values through `ShapeData`.
fn assert_agree(
    what: &str,
    node: &Node,
    opset: u64,
    plugin_rule: ShapeInference,
    plugin_inputs: &[TensorView<'_>],
    native_inputs: Vec<NodeIo>,
) {
    let plugin =
        onnx_runtime_ep_plugin::compute::infer_shapes_for_test(&plugin_rule, plugin_inputs)
            .unwrap_or_else(|e| panic!("{what}: plugin rule failed: {e}"));
    let nat = native(node, native_inputs, opset);
    let Some(native_dims) = native_static(&nat) else {
        panic!(
            "{what}: the native table left a symbolic dim, so this case cannot \
             be compared — either give it the values it needs via ShapeData, or \
             move this op out of this test"
        );
    };
    assert_eq!(
        plugin[0], native_dims,
        "{what}: the two shape tables disagree. Plugin says {:?}, native says \
         {native_dims:?}. One of them encodes the ONNX semantics wrongly, and \
         whichever path a user exercised would look correct.",
        plugin[0]
    );

    let input_shapes: Vec<Vec<Option<usize>>> = plugin_inputs
        .iter()
        .map(|input| input.shape.iter().copied().map(Some).collect())
        .collect();
    let strategy = ShapeInference::for_node(node, &input_shapes, node.outputs.len());
    let ShapeInference::SharedNative { fallback, .. } = &strategy else {
        panic!(
            "{what}: the production registry stopped routing this op through the shared adapter"
        );
    };
    assert!(
        !matches!(fallback.as_ref(), ShapeInference::SharedNative { .. }),
        "{what}: a shared rule's fallback must not recurse into the shared adapter"
    );
    let production =
        onnx_runtime_ep_plugin::compute::infer_shapes_for_test(&strategy, plugin_inputs)
            .unwrap_or_else(|e| panic!("{what}: production shared rule failed: {e}"));
    assert_eq!(production, vec![native_dims]);
}

fn tile_agrees() {
    let buf = [0u8; 24];
    let data = view(DataType::Float32, &[2, 3], &[3, 1], buf.as_ptr());
    let reps: [i64; 2] = [3, 2];
    let r = view(DataType::Int64, &[2], &[1], reps.as_ptr().cast());
    assert_agree(
        "Tile",
        &node("Tile", 2, &[], 13),
        13,
        ShapeInference::Tile,
        &[data, r],
        vec![
            typed(DataType::Float32, &[2, 3]),
            typed_with_values(&[2], &reps),
        ],
    );
}

fn expand_agrees_on_bidirectional_broadcast() {
    // The case a "just take the target" implementation gets wrong. If the two
    // tables ever diverge, this is where it shows.
    let buf = [0u8; 12];
    let data = view(DataType::Float32, &[3, 1], &[1, 1], buf.as_ptr());
    let want: [i64; 2] = [1, 4];
    let s = view(DataType::Int64, &[2], &[1], want.as_ptr().cast());
    assert_agree(
        "Expand",
        &node("Expand", 2, &[], 13),
        13,
        ShapeInference::Expand,
        &[data, s],
        vec![
            typed(DataType::Float32, &[3, 1]),
            typed_with_values(&[2], &want),
        ],
    );
}

fn constant_of_shape_agrees() {
    let dims: [i64; 3] = [2, 3, 4];
    let t = view(DataType::Int64, &[3], &[1], dims.as_ptr().cast());
    assert_agree(
        "ConstantOfShape",
        &node("ConstantOfShape", 1, &[], 9),
        9,
        ShapeInference::ConstantOfShape,
        &[t],
        vec![typed_with_values(&[3], &dims)],
    );
}

fn dft_opset17_default_axis_agrees() {
    let data = [0.0f32; 48];
    let input = view(
        DataType::Float32,
        &[1, 8, 6, 1],
        &[48, 6, 1, 1],
        data.as_ptr().cast(),
    );
    assert_agree(
        "DFT opset 17 default axis",
        &node("DFT", 1, &[("onesided", 1)], 17),
        17,
        ShapeInference::Dft {
            onesided: true,
            axis_attr: None,
            default_axis: 1,
        },
        &[input],
        vec![typed(DataType::Float32, &[1, 8, 6, 1])],
    );
}

fn einsum_agrees_on_canonical_gemm_shape() {
    let mut einsum = node("Einsum", 2, &[], 24);
    einsum
        .attributes
        .insert("equation".into(), Attribute::String(b"ik,kj->ij".to_vec()));
    let left = [0.0f32; 6];
    let right = [0.0f32; 12];
    let left_view = view(DataType::Float32, &[2, 3], &[3, 1], left.as_ptr().cast());
    let right_view = view(DataType::Float32, &[3, 4], &[4, 1], right.as_ptr().cast());
    let input_shapes = vec![vec![Some(2), Some(3)], vec![Some(3), Some(4)]];
    let plugin_rule = ShapeInference::for_node(&einsum, &input_shapes, 1);
    assert_agree(
        "Einsum canonical GEMM",
        &einsum,
        24,
        plugin_rule,
        &[left_view, right_view],
        vec![
            typed(DataType::Float32, &[2, 3]),
            typed(DataType::Float32, &[3, 4]),
        ],
    );
}

#[test]
fn migrated_shared_rules_agree() {
    const EXPECTED_RULES: &[&str] = &["ConstantOfShape", "DFT", "Einsum", "Expand", "STFT", "Tile"];
    let rules = onnx_runtime_ep_plugin::compute::shared_native_rule_names_for_test();
    assert_eq!(
        rules, EXPECTED_RULES,
        "the production shared-rule census changed; update the explicit agreement fixtures in the same commit"
    );
    let mut compared = 0;
    for rule in rules {
        match rule {
            "ConstantOfShape" => constant_of_shape_agrees(),
            "DFT" => dft_opset17_default_axis_agrees(),
            "Einsum" => einsum_agrees_on_canonical_gemm_shape(),
            "Expand" => expand_agrees_on_bidirectional_broadcast(),
            "STFT" => stft_agrees_on_overlapping_frames_and_onesided_bins(),
            "Tile" => tile_agrees(),
            other => panic!("shared rule {other} has no agreement fixture"),
        }
        compared += 1;
    }
    assert_eq!(compared, EXPECTED_RULES.len());
}

#[test]
fn migrated_shared_rules_agree_on_edge_extents() {
    let empty: [i64; 0] = [];
    let empty_shape = view(DataType::Int64, &[0], &[1], empty.as_ptr().cast());
    assert_agree(
        "ConstantOfShape empty shape",
        &node("ConstantOfShape", 1, &[], 9),
        9,
        ShapeInference::ConstantOfShape,
        &[empty_shape],
        vec![typed_with_values(&[0], &empty)],
    );

    let scalar = [0.0f32; 1];
    let data = view(DataType::Float32, &[1], &[1], scalar.as_ptr().cast());
    let zero_target = [0i64];
    let target = view(DataType::Int64, &[1], &[1], zero_target.as_ptr().cast());
    assert_agree(
        "Expand zero target extent",
        &node("Expand", 2, &[], 8),
        8,
        ShapeInference::Expand,
        &[data, target],
        vec![
            typed(DataType::Float32, &[1]),
            typed_with_values(&[1], &zero_target),
        ],
    );

    let tile_data = [0.0f32; 6];
    let data = view(
        DataType::Float32,
        &[2, 3],
        &[3, 1],
        tile_data.as_ptr().cast(),
    );
    let zero_repeat = [0i64, 2];
    let repeats = view(DataType::Int64, &[2], &[1], zero_repeat.as_ptr().cast());
    assert_agree(
        "Tile zero repeat",
        &node("Tile", 2, &[], 13),
        13,
        ShapeInference::Tile,
        &[data, repeats],
        vec![
            typed(DataType::Float32, &[2, 3]),
            typed_with_values(&[2], &zero_repeat),
        ],
    );
}

fn stft_agrees_on_overlapping_frames_and_onesided_bins() {
    let signal_buf = [0.0f32; 128];
    let signal = view(
        DataType::Float32,
        &[2, 64, 1],
        &[64, 1, 1],
        signal_buf.as_ptr().cast(),
    );
    let step_value = [4i64];
    let step = view(DataType::Int64, &[], &[], step_value.as_ptr().cast());
    let frame_value = [16i64];
    let frame = view(DataType::Int64, &[], &[], frame_value.as_ptr().cast());
    let absent_window = TensorView::absent(DataType::Float32);
    let mut stft = node("STFT", 4, &[], 17);
    stft.inputs[2] = None;

    let plugin_inputs = [signal, step, absent_window, frame];
    let native_inputs = vec![
        typed(DataType::Float32, &[2, 64, 1]),
        typed_with_values(&[], &step_value),
        NodeIo::default(),
        typed_with_values(&[], &frame_value),
    ];
    let native_dims = native_static(&native(&stft, native_inputs, 17))
        .expect("STFT native rule must resolve concrete dimensions");
    let strategy = ShapeInference::for_node(
        &stft,
        vec![vec![Some(2), Some(64), Some(1)], vec![], vec![], vec![]].as_slice(),
        1,
    );
    assert!(
        matches!(&strategy, ShapeInference::SharedNative { .. }),
        "STFT must use the shared native shape adapter"
    );
    let plugin = onnx_runtime_ep_plugin::compute::infer_shapes_for_test(&strategy, &plugin_inputs)
        .expect("STFT shared plugin shape inference");
    assert_eq!(plugin, vec![native_dims]);
}

// ── Sweep ────────────────────────────────────────────────────────────────────

/// Every plain shape-preserving op added by #2049 must agree.
///
/// The four cases above are hand-built because they need *values*. These do
/// not: one input, output shape equals it. Sweeping them costs nothing and
/// catches the case where one table quietly stops preserving the shape — a
/// `DequantizeLinear` that started broadcasting against its scale, say.
///
/// Unlike the migrated cases above, these not-yet-shared rules have no
/// production descriptor to enumerate. Their explicit list is temporary
/// duplication and should disappear as later slices move them into
/// [`SharedNativeShapeRule`].
#[test]
fn shape_preserving_ops_agree() {
    // (op, opset, extra input count) — the companions are dtype/scale operands
    // that must not affect the extent.
    const CASES: &[(&str, u64, usize)] = &[
        ("BitwiseNot", 18, 0),
        ("CastLike", 15, 1),
        ("CumSum", 14, 1),
        ("DequantizeLinear", 13, 1),
        ("QuantizeLinear", 13, 1),
        ("EyeLike", 9, 0),
    ];

    let buf = [0u8; 64];
    let mut compared = 0;
    for &(op, opset, extras) in CASES {
        let dims: [usize; 2] = [3, 4];
        let strides: [i64; 2] = [4, 1];
        let mut plugin_inputs = vec![view(DataType::Float32, &dims, &strides, buf.as_ptr())];
        let mut native_inputs = vec![typed(DataType::Float32, &[3, 4])];
        for _ in 0..extras {
            // A rank-1 companion sized to match axis 0, so it is *valid* for the
            // per-axis quantization spellings. If either table broadcast against
            // it the answer would widen, which is the mistake to catch.
            //
            // Sizing it to 3 rather than 4 is not arbitrary: the native table
            // validates a per-axis scale against the axis it applies to and
            // rejects a mismatch outright, while the plugin's `SameAsInput(0)`
            // does not look at the companion at all. That asymmetry is real and
            // worth knowing — the native table is stricter — but it is not a
            // shape disagreement, so this test feeds inputs both accept.
            plugin_inputs.push(view(
                DataType::Float32,
                &dims[..1],
                &strides[..1],
                buf.as_ptr(),
            ));
            native_inputs.push(typed(DataType::Float32, &[3]));
        }
        let attrs: &[(&str, i64)] = if matches!(op, "DequantizeLinear" | "QuantizeLinear") {
            &[("axis", 0)]
        } else {
            &[]
        };
        let n = node(op, 1 + extras, attrs, opset);
        let nat = native_try(&n, native_inputs, opset)
            .unwrap_or_else(|error| panic!("{op}: expected native inference to resolve: {error}"));
        let native_dims =
            native_static(&nat).unwrap_or_else(|| panic!("{op}: expected a concrete native shape"));
        let plugin = onnx_runtime_ep_plugin::compute::infer_shapes_for_test(
            &ShapeInference::SameAsInput(0),
            &plugin_inputs,
        )
        .unwrap_or_else(|e| panic!("{op}: plugin rule failed: {e}"));
        assert_eq!(
            plugin[0], native_dims,
            "{op}: the two shape tables disagree. Plugin says {:?}, native says \
             {native_dims:?}.",
            plugin[0]
        );
        compared += 1;
    }
    assert_eq!(
        compared,
        CASES.len(),
        "every expected shape-preserving fixture must reach the comparison"
    );
}

#[test]
fn malformed_dequantize_preserves_the_plugin_permissiveness_boundary() {
    let data_buf = [0.0f32; 12];
    let scale_buf = [1.0f32; 3];
    let data = view(
        DataType::Float32,
        &[3, 4],
        &[4, 1],
        data_buf.as_ptr().cast(),
    );
    let scale = view(DataType::Float32, &[3], &[1], scale_buf.as_ptr().cast());
    let n = node("DequantizeLinear", 2, &[], 13);

    let native_error = native_try(
        &n,
        vec![
            typed(DataType::Float32, &[3, 4]),
            typed(DataType::Float32, &[3]),
        ],
        13,
    )
    .expect_err("native inference validates the per-axis scale length");
    assert!(native_error.to_string().contains("scale length 3"));
    let strategy = ShapeInference::for_node(&n, &[vec![Some(3), Some(4)], vec![Some(3)]], 1);
    assert!(
        matches!(strategy, ShapeInference::SameAsInput(0)),
        "DequantizeLinear is intentionally outside the first shared slice"
    );
    assert_eq!(
        onnx_runtime_ep_plugin::compute::infer_shapes_for_test(&strategy, &[data, scale]).unwrap(),
        vec![vec![3, 4]],
        "the plugin's pre-existing permissive sizing contract must remain unchanged"
    );
}

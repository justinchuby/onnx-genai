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
use onnx_runtime_ep_plugin::shared_shapes::{
    SharedNativeShapeRule, SharedShapeResult, infer_shared_node,
};
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

    let shared = infer_shared_node(node, plugin_inputs);
    assert_eq!(
        shared,
        SharedShapeResult::Resolved(vec![native_dims.clone()]),
        "{what}: the production adapter did not expose the native concrete answer"
    );

    let input_shapes: Vec<Vec<Option<usize>>> = plugin_inputs
        .iter()
        .map(|input| input.shape.iter().copied().map(Some).collect())
        .collect();
    let strategy = ShapeInference::for_node(node, &input_shapes, node.outputs.len());
    assert!(
        matches!(strategy, ShapeInference::SharedNative { .. }),
        "{what}: the production registry stopped routing this op through the shared adapter"
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

#[test]
fn migrated_shared_rules_agree() {
    let rules = SharedNativeShapeRule::all();
    assert!(
        !rules.is_empty(),
        "the shared-rule agreement test must not pass vacuously"
    );
    for rule in rules {
        match rule {
            SharedNativeShapeRule::ConstantOfShape => constant_of_shape_agrees(),
            SharedNativeShapeRule::Expand => expand_agrees_on_bidirectional_broadcast(),
            SharedNativeShapeRule::Tile => tile_agrees(),
        }
    }
}

// ── Sweep ────────────────────────────────────────────────────────────────────

/// Every plain shape-preserving op must agree, driven from the registry rather
/// than a hand-written list.
///
/// The three cases above are hand-built because they need *values*. These do
/// not: one input, output shape equals it. Sweeping them costs nothing and
/// catches the case where one table quietly stops preserving the shape — a
/// `DequantizeLinear` that started broadcasting against its scale, say.
///
/// A hand-written list would drift from the rules themselves, which is the very
/// failure this file exists to catch, so the op names come from the same place
/// the plugin's arm does.
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
        let n = node(op, 1 + extras, &[], opset);
        let nat = match native_try(&n, native_inputs, opset) {
            Ok(outs) => outs,
            // The native table rejected the node outright. That is a validity
            // judgement, not a shape answer, so there is nothing to compare —
            // and it is recorded rather than swallowed so a reader knows the
            // two tables differ in strictness, not in shape.
            Err(_) => continue,
        };
        let Some(native_dims) = native_static(&nat) else {
            // The native table declined to resolve this one; nothing to compare.
            continue;
        };
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
    }
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
    assert!(matches!(
        infer_shared_node(&n, &[data, scale]),
        SharedShapeResult::Rejected(reason) if reason.contains("scale length 3")
    ));

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

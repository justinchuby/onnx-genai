//! Inventory test: every op the CPU EP registers a kernel for must have a
//! shape rule, or be on an explicit allowlist of ops that do not.
//!
//! `GetCapability` runs a fail-closed filter that drops any claim containing a
//! node whose [`ShapeInference::for_node`] returns `Declined`, and that match
//! ends in `_ => Declined`. So an op we register a kernel for, but which is
//! absent from the table, is silently handed to ORT's CPU EP no matter what
//! `supports_op` answers — the mechanism that made the `com.microsoft`
//! activations unreachable until #1082 and the trigonometric family
//! unreachable until #1097.
//!
//! The per-op assignment fixtures in `plugin_ort_e2e.rs` cannot catch this
//! class of bug: they only prove the ops they name are assigned, and a
//! newly-registered op has no fixture by construction. This test closes that
//! hole by enumerating the registry itself.
//!
//! The invariant is not yet universally true — see
//! `docs/performance/CPU_ACTIVATION_GAPS.md`. Until it is, the value here is
//! drift detection: the allowlist is asserted *exactly*, so registering a new
//! op without a shape rule fails, and adding a shape rule without removing the
//! op from the allowlist also fails. Neither direction can pass silently.

use onnx_runtime_ep_cpu::kernels::build_cpu_registry;
use onnx_runtime_ep_plugin::compute::ShapeInference;
use onnx_runtime_ir::{Node, NodeId, ValueId};

/// Ops that resolve to `Declined` under the probe below, and therefore reach
/// ORT's CPU EP rather than ours when they appear in a graph.
///
/// Three groups, and only the third is work we intend to do.
const DECLINED: &[(&str, &str)] = &[
    // ── 1. Data-dependent. The output shape is a function of an input's
    //       *values*, not its shape, so it cannot be inferred at capability
    //       time. Correctly declined; not a gap.
    ("", "Compress"),          // output length = count of true in condition
    ("", "NonMaxSuppression"), // output length = number of boxes kept
    ("", "NonZero"),           // output length = count of non-zeros
    ("", "Unique"),            // output length = number of distinct values
    ("", "BlackmanWindow"),    // size comes from the input tensor's value
    ("", "HammingWindow"),
    ("", "HannWindow"),
    ("", "ConstantOfShape"), // shape comes from input[0]'s values
    ("", "Expand"),          // shape comes from input[1]'s values
    ("", "Range"),           // length = ceil((limit - start) / delta)
    ("", "Tile"),            // repeats come from input[1]'s values
    ("", "OneHot"),          // depth comes from input[1]'s value
    ("", "Pad"),             // pads come from input[1]'s values (opset 11+)
    ("", "TopK"),            // K comes from input[1]'s value (opset 10+)
    ("", "Split"),           // split sizes come from input[1] (opset 13+)
    ("", "Unsqueeze"),       // axes come from input[1] (opset 13+)
    ("", "Resize"),          // scales/sizes come from input[2]/input[3]
    ("", "AffineGrid"),      // output size comes from input[1]'s values
    ("", "Col2Im"),          // image_shape comes from input[1]'s values
    ("", "CenterCropPad"),   // target shape comes from input[1]'s values
    // Several of these carry a *constant initializer* in practice, so a
    // future pass that resolves initializer values at capability time could
    // claim them. Until such a pass exists, declining is the honest answer.

    // ── 2. Internal ops emitted by our own fusion passes. They are created
    //       after capability, so they are never candidates at capability time.
    ("com.microsoft", "FusedAttention"),
    ("com.microsoft", "FusedGemm"),
    ("com.microsoft", "FusedMatMulBias"),
    ("pkg.nxrt", "BlockQuantizedMatMul"),
    ("pkg.nxrt", "BlockQuantizedMoE"),
    ("pkg.nxrt", "CompressedSparseAttention"),
    ("pkg.nxrt", "IndexShare"),
    ("pkg.nxrt", "PackedVarlenAttention"),
    ("pkg.nxrt", "SparseKvGather"),
    ("pkg.nxrt", "VarlenAttention"),
    // ── 3. Inferrable but unwritten. These appear in real input graphs, we
    //       have a kernel, and we hand them to ORT anyway. This is the work.
    //
    //       Shape-preserving — output shape == input[0].shape. One line each.
    ("", "BitwiseNot"),
    ("", "GroupNormalization"),
    ("", "CastLike"),
    ("", "CumProd"),
    ("", "CumSum"),
    ("", "DequantizeLinear"),
    ("", "EyeLike"),
    ("", "QuantizeLinear"),
    ("", "ScatterElements"),
    ("", "ScatterND"),
    ("", "Trilu"),
    //       Pooling and CNN geometry: inferrable from `kernel_shape`,
    //       `strides`, `pads`, `dilations` and `ceil_mode`, exactly as
    //       `build_conv` already does for `Conv`.
    ("", "AveragePool"),
    ("", "ConvTranspose"),
    ("", "GlobalAveragePool"),
    ("", "GlobalLpPool"),
    ("", "GlobalMaxPool"),
    ("", "GridSample"),
    ("", "LpPool"),
    ("", "MaxPool"),
    ("", "SpaceToDepth"),
    //       Inferrable from attributes or a fixed rule.
    ("", "ArgMax"), // reduce over `axis`, honouring `keepdims`
    ("", "ArgMin"),
    ("", "Constant"), // shape of the `value` attribute's tensor
    ("", "DFT"),
    ("", "DynamicQuantizeLinear"), // y == input, scale/zero_point scalar
    ("", "Flatten"),               // 2-D, split at `axis` (default 1)
    ("", "GatherElements"),        // output shape == indices shape
    ("", "QLinearMatMul"),         // MatMul semantics on the quantised operands
    ("", "Size"),                  // scalar
    //       Contrib ops that need a real rule written.
    ("com.microsoft", "Attention"), // packed QKV; a different signature from
    //                                 ai.onnx::Attention, so the opset-23 arm
    //                                 deliberately does not cover it
    //       Qwen3.5 / Qwen3-Next hybrid linear-attention primitives, read
    //       from exported models rather than produced by our fusion passes.
    //       ORT has no kernel for these at all, so handing them over does not
    //       get a faster implementation — it gets a load failure.
    ("", "LinearAttention"),
    ("com.microsoft", "CausalConvWithState"),
    ("com.microsoft", "LinearAttention"),
    ("com.microsoft", "MoE"),
    ("com.microsoft", "PackedMultiHeadAttention"),
    ("com.microsoft", "QMoE"),
];

/// Ops registered only so a test can exercise the registry machinery. They are
/// not real ONNX ops and must never be expected in the shape table.
const TEST_ONLY: &[&str] = &["TotallyFakeOp"];

/// Does this op decline shape inference for *every* plausible node?
///
/// A single probe is not enough. `Conv` reads `input_shapes[1][0]` for its
/// output channel count and needs rank >= 3, so a one-input rank-2 probe
/// reports it as declining when it is in fact covered. Sweep a matrix of
/// arities and ranks instead, and count an op as declining only when no shape
/// in the matrix produces a rule.
///
/// Opsets matter too: several arms switch behaviour at an opset boundary
/// (`Unsqueeze`'s axes moved from attribute to input at 13, `ReductionFromInput`
/// at 18). Sweeping them keeps the allowlist from encoding one arbitrary point.
///
/// No attributes are supplied. That is deliberate: an attribute-dependent rule
/// must fall back to the ONNX default when the attribute is absent, because a
/// spec-legal graph may rely on that default. `for_node` already does this
/// (`int_attr("axis").unwrap_or(..)`), so an op that declines only for want of
/// an attribute has a rule that would decline on a real graph too.
fn declines(op_type: &str, domain: &str) -> bool {
    const OPSETS: &[i64] = &[1, 13, 18, 22, 23];
    const SHAPES: &[&[usize]] = &[&[8], &[4, 8], &[2, 3, 8], &[1, 3, 8, 8]];

    for &opset in OPSETS {
        for arity in 1..=4usize {
            for shape in SHAPES {
                let inputs: Vec<Option<ValueId>> =
                    (0..arity).map(|i| Some(ValueId(i as u32))).collect();
                let mut node = Node::new(NodeId(0), op_type, inputs, vec![ValueId(100)]);
                node.domain = domain.to_string();
                node.version = Some(opset);
                let input_shapes: Vec<Vec<Option<usize>>> = (0..arity)
                    .map(|_| shape.iter().map(|&d| Some(d)).collect())
                    .collect();
                if !matches!(
                    ShapeInference::for_node(&node, &input_shapes, 1),
                    ShapeInference::Declined { .. }
                ) {
                    return false;
                }
            }
        }
    }
    true
}

/// Enumerate from `OpRegistry::keys()`, not from
/// `build_cpu_registry_with_descriptors()`.
///
/// The descriptor list is *not* the registry: `register_cnn_ops` writes
/// straight to the inner `OpRegistry`, so `Resize`, `GridSample`,
/// `ConvTranspose`, `MaxPool`, the pooling family and `GroupNormalization`
/// never appear in the descriptors. `supports_op` keys off the registry, so
/// those ops are claimed and then dropped by the shape filter — invisible to
/// an inventory built on descriptors. `keys()` is the same set `supports_op`
/// consults, which is what makes it the right source of truth here.
fn registered_ops() -> Vec<(String, String)> {
    let registry = build_cpu_registry();
    let mut ops: Vec<(String, String)> = registry
        .keys()
        .filter(|k| !TEST_ONLY.contains(&k.op_type.as_str()))
        .map(|k| (k.domain.clone(), k.op_type.clone()))
        .collect();
    ops.sort();
    ops.dedup();
    ops
}

#[test]
fn every_registered_op_has_a_shape_rule_or_is_a_known_gap() {
    let actual: Vec<(String, String)> = registered_ops()
        .into_iter()
        .filter(|(domain, op)| declines(op, domain))
        .collect();

    let mut expected: Vec<(String, String)> = DECLINED
        .iter()
        .map(|(d, o)| ((*d).to_string(), (*o).to_string()))
        .collect();
    expected.sort();

    let newly_declined: Vec<_> = actual.iter().filter(|e| !expected.contains(e)).collect();
    let newly_covered: Vec<_> = expected.iter().filter(|e| !actual.contains(e)).collect();

    assert!(
        newly_declined.is_empty(),
        "these registered ops have no shape rule, so `GetCapability` will \
         silently hand them to ORT's CPU EP: {newly_declined:?}. Write the \
         rule in `ShapeInference::for_node`, or — if the output shape is \
         genuinely data-dependent — add it to `DECLINED` with which group it \
         belongs to."
    );
    assert!(
        newly_covered.is_empty(),
        "these ops now have a shape rule but are still listed as declined: \
         {newly_covered:?}. Remove them from `DECLINED` so the list keeps \
         describing reality."
    );
}

/// Every activation and normalisation op this EP owns must be assigned to it.
///
/// These are the families #1082, #1093 and #1097 made reachable. A regression
/// here means real transformer nodes silently execute on ORT again.
///
/// The list is asserted in both directions. An op that stops being registered
/// fails just as loudly as one that stops having a shape rule: renaming or
/// dropping a kernel does not get to quietly hand the op back to ORT, and a
/// filter of the form `registered.contains(op) && declines(op)` would let
/// exactly that through.
#[test]
fn no_activation_or_norm_op_is_left_to_ort() {
    const REQUIRED: &[(&str, &str)] = &[
        ("", "Relu"),
        ("", "Sigmoid"),
        ("", "Tanh"),
        ("", "Erf"),
        ("", "Exp"),
        ("", "Log"),
        ("", "Sqrt"),
        ("", "Gelu"),
        ("", "HardSigmoid"),
        ("", "LeakyRelu"),
        ("", "Elu"),
        ("", "Selu"),
        ("", "Softplus"),
        ("", "Softsign"),
        ("", "PRelu"),
        ("", "ThresholdedRelu"),
        ("", "Softmax"),
        ("", "LogSoftmax"),
        ("", "Swish"),
        ("", "Sin"),
        ("", "Cos"),
        ("", "Tan"),
        ("", "Asin"),
        ("", "Acos"),
        ("", "Atan"),
        ("", "Sinh"),
        ("", "Cosh"),
        ("", "Asinh"),
        ("", "Acosh"),
        ("", "Atanh"),
        ("", "LayerNormalization"),
        ("", "RMSNormalization"),
        ("com.microsoft", "FastGelu"),
        ("com.microsoft", "QuickGelu"),
        ("com.microsoft", "BiasGelu"),
        ("com.microsoft", "Silu"),
        ("com.microsoft", "SkipLayerNormalization"),
        ("com.microsoft", "SkipSimplifiedLayerNormalization"),
        ("com.microsoft", "SimplifiedLayerNormalization"),
    ];

    let registered = registered_ops();
    let unregistered: Vec<_> = REQUIRED
        .iter()
        .filter(|(domain, op)| !registered.contains(&((*domain).to_string(), (*op).to_string())))
        .collect();
    assert!(
        unregistered.is_empty(),
        "these ops are no longer registered by the CPU EP, so ORT will execute \
         them: {unregistered:?}. Restore the registration, or — if the kernel \
         was deliberately dropped — remove the op from REQUIRED and say why."
    );

    let declining: Vec<_> = REQUIRED
        .iter()
        .filter(|(domain, op)| declines(op, domain))
        .collect();
    assert!(
        declining.is_empty(),
        "these activation/norm ops are registered but decline shape \
         inference, so they run on ORT's CPU EP: {declining:?}"
    );
}

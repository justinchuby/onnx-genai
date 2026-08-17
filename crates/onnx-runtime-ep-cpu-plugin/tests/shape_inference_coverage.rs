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

use onnx_runtime_ep_cpu::kernels::build_cpu_registry_with_descriptors;
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
    // Several of these carry a *constant initializer* in practice, so a
    // future pass that resolves initializer values at capability time could
    // claim them. Until such a pass exists, declining is the honest answer.

    // ── 2. Internal ops emitted by our own fusion passes. They are created
    //       after capability, so they are never candidates at capability time.
    ("", "LinearAttention"),
    ("com.microsoft", "CausalConvWithState"),
    ("com.microsoft", "FusedAttention"),
    ("com.microsoft", "FusedGemm"),
    ("com.microsoft", "FusedMatMulBias"),
    ("com.microsoft", "LinearAttention"),
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
    ("", "CastLike"),
    ("", "CumProd"),
    ("", "CumSum"),
    ("", "DequantizeLinear"),
    ("", "EyeLike"),
    ("", "QuantizeLinear"),
    ("", "ScatterElements"),
    ("", "ScatterND"),
    ("", "Trilu"),
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
    ("com.microsoft", "MoE"),
    ("com.microsoft", "PackedMultiHeadAttention"),
    ("com.microsoft", "QMoE"),
];

/// Ops registered only so a test can exercise the registry machinery. They are
/// not real ONNX ops and must never be expected in the shape table.
const TEST_ONLY: &[&str] = &["TotallyFakeOp"];

/// Probe an op the way `GetCapability` does: a bare node carrying only its
/// identity, one rank-2 input, one output.
///
/// Attribute-dependent rules must therefore be written to fall back to the
/// ONNX default when the attribute is absent — which is what `for_node`
/// already does (`int_attr("axis").unwrap_or(..)`). An op that declines only
/// because this probe supplies no attributes would be a shape rule that
/// crashes on a spec-legal graph relying on defaults, so counting it as
/// declined is correct rather than a false positive.
fn declines(op_type: &str, domain: &str) -> bool {
    let mut node = Node::new(NodeId(0), op_type, vec![Some(ValueId(0))], vec![ValueId(1)]);
    node.domain = domain.to_string();
    node.version = Some(if domain.is_empty() { 22 } else { 1 });
    matches!(
        ShapeInference::for_node(&node, &[vec![Some(4), Some(8)]], 1),
        ShapeInference::Declined { .. }
    )
}

fn registered_ops() -> Vec<(String, String)> {
    let (_registry, descriptors) = build_cpu_registry_with_descriptors();
    let mut ops: Vec<(String, String)> = descriptors
        .into_iter()
        .filter(|d| !TEST_ONLY.contains(&d.op_type.as_str()))
        .map(|d| (d.domain, d.op_type))
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
/// These are the families #1082, #1093 and #1097 made reachable; a regression
/// here means real transformer nodes silently execute on ORT again.
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
        ("", "HardSwish"),
        ("", "LeakyRelu"),
        ("", "Elu"),
        ("", "Selu"),
        ("", "Softplus"),
        ("", "Softsign"),
        ("", "PRelu"),
        ("", "ThresholdedRelu"),
        ("", "Softmax"),
        ("", "LogSoftmax"),
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
        ("com.microsoft", "Swish"),
        ("com.microsoft", "SkipLayerNormalization"),
        ("com.microsoft", "SkipSimplifiedLayerNormalization"),
        ("com.microsoft", "SimplifiedLayerNormalization"),
    ];

    let registered = registered_ops();
    let missing: Vec<_> = REQUIRED
        .iter()
        .filter(|(domain, op)| {
            registered.contains(&((*domain).to_string(), (*op).to_string())) && declines(op, domain)
        })
        .collect();
    assert!(
        missing.is_empty(),
        "these activation/norm ops are registered but decline shape \
         inference, so they run on ORT's CPU EP: {missing:?}"
    );
}

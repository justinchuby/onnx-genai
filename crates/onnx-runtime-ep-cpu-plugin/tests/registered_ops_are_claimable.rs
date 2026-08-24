//! A registered kernel that the plugin cannot claim must fail loudly.
//!
//! # The failure this exists to stop
//!
//! Registering an operator on an execution provider is a multi-place contract:
//! a factory entry, a dtype arm, a coverage-list entry, **and** a rule in
//! [`ShapeInference::for_node`]. `GetCapability` applies a fail-closed shape
//! filter, so a node whose rule resolves to `Declined` is dropped from the
//! claim — silently, and by design, because over-claiming an op we cannot shape
//! correctly is worse.
//!
//! The silence is right for an op we never registered. It is wrong for one we
//! did: somebody wrote a kernel, wired it into the registry, and the plugin
//! quietly never uses it. Nothing fails. The op runs somewhere else and the
//! only symptom is that it is slower than the author believes.
//!
//! `ai.onnx::Compress` was in exactly that state — a CPU kernel, a CUDA kernel,
//! a `CUDA_COVERED_OPS` entry, and no shape rule.
//!
//! # What this asserts
//!
//! Every `(op_type, domain)` in the built CPU registry resolves to something
//! other than `Declined { reason: Unmodelled }` — that is, the table has an arm
//! for the op name.
//!
//! It deliberately does **not** assert that every node shapes successfully.
//! `DeclineReason::NodeNotShapeable` is the table working as designed: a
//! `Conv` whose attributes do not determine an output, or an opset-13
//! `Unsqueeze` whose axes arrive as an input, is correctly declined. Counting
//! those as failures would report ~50 ops and the guard would be ignored, which
//! is how a noisy check becomes no check.
//!
//! # Why the CPU registry and not the CUDA one
//!
//! `build_cuda_registry` needs a live `CudaRuntime`, so a CUDA equivalent would
//! only run on a GPU lane. The rule table is shared, and the CPU registry is a
//! superset of the CUDA one for standard-domain ops, so this catches the same
//! defect on every lane. A CUDA-side companion belongs on the GPU lane.

use onnx_runtime_ep_plugin::compute::{DeclineReason, ShapeInference};
use onnx_runtime_ir::{Node, NodeId};

/// Registered operators that deliberately have no rule at all.
///
/// Empty on purpose. An entry here is a decision to ship a kernel the plugin
/// will never dispatch to, and needs a reason a reviewer can weigh — not a
/// silent hole. Prefer writing the rule; a data-dependent extent is expressible
/// as `None` for that dimension, which is what the rule table already does for
/// dynamic dims.
/// Registered operators the plugin cannot claim today, because
/// `ShapeInference::for_node` has no arm for them.
///
/// **This list is a ratchet, not an approval.** Every entry is a kernel the ORT
/// plugin path can never dispatch to: `GetCapability` drops any claim containing
/// the node, the op runs on another provider, and nothing anywhere says so. The
/// list existing is what turns that from silent into visible — it was 55 when
/// first measured, and no one had counted.
///
/// It may only shrink. Adding an entry means shipping another kernel the plugin
/// will not use, and needs a reason in review.
const NOT_CLAIMABLE: &[(&str, &str)] = &[
    ("AffineGrid", ""),
    ("ArgMax", ""),
    ("ArgMin", ""),
    ("AveragePool", ""),
    ("BitwiseNot", ""),
    ("BlackmanWindow", ""),
    ("BlockQuantizedMatMul", "pkg.nxrt"),
    ("BlockQuantizedMoE", "pkg.nxrt"),
    ("CastLike", ""),
    ("CenterCropPad", ""),
    ("Col2Im", ""),
    ("CompressedSparseAttention", "pkg.nxrt"),
    ("Constant", ""),
    ("ConstantOfShape", ""),
    ("ConvTranspose", ""),
    ("CumProd", ""),
    ("CumSum", ""),
    ("DFT", ""),
    ("DequantizeLinear", ""),
    ("DynamicQuantizeLinear", ""),
    ("Expand", ""),
    ("EyeLike", ""),
    ("Flatten", ""),
    ("FusedAttention", "com.microsoft"),
    ("FusedGemm", "com.microsoft"),
    ("FusedMatMulBias", "com.microsoft"),
    ("GatherElements", ""),
    ("GlobalAveragePool", ""),
    ("GlobalLpPool", ""),
    ("GlobalMaxPool", ""),
    ("GridSample", ""),
    ("HammingWindow", ""),
    ("HannWindow", ""),
    ("IndexShare", "pkg.nxrt"),
    ("LinearAttention", ""),
    ("LinearAttention", "com.microsoft"),
    ("LpPool", ""),
    ("MaxPool", ""),
    ("NonMaxSuppression", ""),
    ("NonZero", ""),
    ("OneHot", ""),
    ("PackedVarlenAttention", "pkg.nxrt"),
    ("Pad", ""),
    ("QuantizeLinear", ""),
    ("Range", ""),
    ("Resize", ""),
    ("Size", ""),
    ("SpaceToDepth", ""),
    ("SparseKvGather", "pkg.nxrt"),
    ("Split", ""),
    ("Tile", ""),
    ("TopK", ""),
    ("Unique", ""),
    ("VarlenAttention", "pkg.nxrt"),
];

fn resolves(op_type: &str, domain: &str, num_outputs: usize) -> ShapeInference {
    let mut node = Node::new(NodeId(0), op_type, vec![], vec![]);
    node.domain = domain.to_string();
    // One rank-2 input covers the common case. A rule that needs more resolves
    // to `NodeNotShapeable`, which this test does not treat as a failure — the
    // claim here is that the op *name* is modelled.
    let inputs = vec![vec![Some(1usize), Some(1usize)]; 4];
    ShapeInference::for_node(&node, &inputs, num_outputs.max(1))
}

#[test]
fn every_registered_op_has_a_shape_rule() {
    let registry = onnx_runtime_ep_cpu::kernels::build_cpu_registry();

    let mut ops: Vec<(String, String)> = registry
        .keys()
        .map(|key| (key.op_type.clone(), key.domain.clone()))
        .collect();
    ops.sort();
    ops.dedup();

    // A registry that failed to build would make every assertion below vacuous.
    assert!(
        ops.len() >= 100,
        "CPU registry has only {} operators, so this test would pass by \
         checking almost nothing",
        ops.len()
    );

    let allowed =
        |op: &str, domain: &str| NOT_CLAIMABLE.iter().any(|(o, d)| *o == op && *d == domain);

    let mut unclaimable: Vec<String> = Vec::new();
    for (op, domain) in &ops {
        if allowed(op, domain) {
            continue;
        }
        if matches!(
            resolves(op, domain, 1),
            ShapeInference::Declined {
                reason: DeclineReason::Unmodelled,
                ..
            }
        ) {
            let shown = if domain.is_empty() {
                format!("ai.onnx::{op}")
            } else {
                format!("{domain}::{op}")
            };
            unclaimable.push(shown);
        }
    }

    assert!(
        unclaimable.is_empty(),
        "these operators have a registered kernel but no arm at all in \
         ShapeInference::for_node, so GetCapability's fail-closed filter drops \
         every claim containing them and the kernel is never dispatched to. \
         Nothing fails today; the op just quietly runs somewhere else: \
         {unclaimable:?}\n\
         Add the rule (a data-dependent extent is `None` for that dimension), \
         or add an entry to NOT_CLAIMABLE with a reason."
    );
}

//! Which nodes this EP asks a plugin host (ORT) to hand over, as opposed to
//! which nodes it is *able* to run.
//!
//! # Why this is separate from `supports_op`
//!
//! [`crate::CpuExecutionProvider::supports_op`] answers "can this EP produce a
//! correct result?". It must keep saying yes for everything we implement,
//! because `onnx-runtime-session`'s executor build turns a statically-shaped
//! `KernelMatch::Unsupported` into a hard `SessionError::unsupported_op` — a
//! kernel we decline to *advertise to ORT* still has to be reachable when this
//! crate is the whole runtime.
//!
//! This module answers the different question ORT actually asks at
//! `GetCapability` time: "**should** we take this node away from ORT's own CPU
//! kernel?" Advertising a node we then run slower than ORT would is a
//! regression to the user no matter how correct the result is, so the answer is
//! no whenever the measured evidence says ORT is faster — or whenever we have
//! no evidence at all.
//!
//! # Evidence
//!
//! Every threshold below comes from a session-level interleaved A/B on this
//! host: the same `.onnx` model run through ORT's CPU EP and through this
//! plugin in the same process, alternating, with `ORT_DISABLE_ALL` so no fusion
//! rewrites the graph, p50 and p90 of whole-`Run` latency. Session latency —
//! not kernel time — is the metric, because per-node dispatch overhead on both
//! sides is part of what the user pays.
//!
//! Two structural facts drive almost every decision:
//!
//! 1. **A fixed per-node plugin overhead of roughly 1.2 µs** sits on top of
//!    every claim, independent of the op. It is visible as a flat ~0.75-0.8x
//!    ratio at `n = 1`, where neither side does any arithmetic. Below roughly
//!    10 K elements no elementwise kernel can earn that back, so the small-size
//!    answer is "decline" for every op regardless of how good the kernel is.
//! 2. **Our elementwise kernels are single-threaded; ORT's are not.** At large
//!    sizes ORT splits the tensor across its intra-op thread pool. A threshold
//!    tuned at `intra_op_num_threads = 1` therefore inverts on a threaded
//!    session — measured for f32 `Sqrt`, which wins 1.9x single-threaded at
//!    1 M elements but loses at 0.30x with 16 threads. The EP cannot see the
//!    thread count at capability time, so any claim has to hold across thread
//!    counts, which caps the claimable range from above as well as below.
//!
//! The margin rule is **>= 5% repeatable win at every measured thread count**;
//! anything inside the noise band, and anything unmeasured, defers to ORT.

use onnx_runtime_ep_api::ClaimPreference;
use onnx_runtime_ir::{DataType, Node, Shape};

/// Ops whose ORT-vs-us performance has been measured and whose claim is
/// therefore governed by this module.
///
/// An op absent from this list keeps the historical "support implies claim"
/// behaviour — this module only ever *removes* claims it has evidence against,
/// it never adds one.
fn is_governed(op: &Node) -> bool {
    matches!(
        (op.domain.as_str(), op.op_type.as_str()),
        ("" | "ai.onnx", "Tanh" | "Sigmoid" | "Gelu" | "Sqrt" | "Erf")
    )
}

/// Upper bound (inclusive) on element count for the f16 `Gelu(approximate=tanh)`
/// claim.
///
/// Measured f16 `Gelu` ratios (ORT ns / ours, >1 means we win):
///
/// | elements | 1 thread | 8 threads | 16 threads |
/// |---------:|---------:|----------:|-----------:|
/// | 512      | 1.49     | 1.56      | -          |
/// | 3072     | 1.47     | 1.57      | -          |
/// | 16384    | 1.50     | 1.64      | -          |
/// | 65536    | 1.50     | 1.60      | 1.66       |
/// | 262144   | 1.81     | 1.20      | 1.87       |
/// | 524288   | -        | -         | 1.01       |
/// | 1048576  | 1.50     | 0.89      | -          |
///
/// The win survives threading up to 262144 elements and stops surviving it
/// after that, so that is the cap.
const F16_GELU_TANH_MAX_ELEMENTS: usize = 262_144;

/// Total element count of the first input, or `None` when any dimension is
/// dynamic.
fn static_elements(shapes: &[Shape]) -> Option<usize> {
    let shape = shapes.first()?;
    let mut n = 1usize;
    for dim in shape.iter() {
        n = n.checked_mul(dim.as_static()?)?;
    }
    Some(n)
}

/// Whether a `Gelu` node selects the tanh approximation.
fn is_tanh_gelu(op: &Node) -> bool {
    op.attr("approximate")
        .and_then(|a| a.as_str())
        .map(|s| s == "tanh")
        .unwrap_or(false)
}

/// Whether this EP wants `op` handed to it by a plugin host, or would rather
/// the host ran its own kernel.
pub fn claim_preference(
    op: &Node,
    _opset: u64,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> ClaimPreference {
    if !is_governed(op) {
        return ClaimPreference::Claim;
    }
    let dtype = match input_dtypes.first() {
        Some(dt) => *dt,
        None => return ClaimPreference::Claim,
    };

    match dtype {
        // ORT's CPU EP has no bfloat16 kernel for any of these ops: without
        // this plugin the session fails to create at all
        // (`NOT_IMPLEMENTED: Could not find an implementation for Sqrt(13)`).
        // The claim here is a capability, not a performance bet, so it is
        // unconditional — declining would turn a working session into a
        // load-time failure.
        DataType::BFloat16 => ClaimPreference::Claim,

        DataType::Float16 => {
            if op.op_type == "Gelu" && is_tanh_gelu(op) {
                match static_elements(shapes) {
                    Some(n) if n <= F16_GELU_TANH_MAX_ELEMENTS => ClaimPreference::Claim,
                    Some(n) => ClaimPreference::defer(format!(
                        "float16 Gelu(tanh) is measured faster than ORT only up to \
                         {F16_GELU_TANH_MAX_ELEMENTS} elements (this node has {n}); above that \
                         ORT's multi-threaded kernel wins (0.89x at 1048576 with 8 threads)"
                    )),
                    // A dynamic shape could be any size, including the sizes
                    // where we lose. Fail conservative: there is no run-time
                    // re-decision once ORT has committed the partition.
                    None => ClaimPreference::defer(
                        "float16 Gelu(tanh) is only faster than ORT up to a measured element \
                         count, and this node's shape is not static at capability time",
                    ),
                }
            } else {
                ClaimPreference::defer(
                    "measured slower than ORT's float16 CPU kernel at every size on x86-64 \
                     AVX2 (0.59-0.96x); only Gelu(approximate=tanh) is competitive in float16",
                )
            }
        }

        // Float32 is where ORT is strongest: MLAS activation kernels plus
        // intra-op threading. Measured session-level ratios on AVX2 with one
        // thread: Tanh 0.60-0.82x, Sigmoid 0.62-0.81x, Gelu(tanh) 0.64-0.79x,
        // Gelu(none) 0.036-0.79x, Erf 0.023-0.77x. `Sqrt` is now 1.1-1.9x
        // single-threaded above ~8 K elements, but inverts to 0.30x at 16
        // threads, and the thread count is not visible here — so it loses under
        // the "must hold across thread counts" rule too.
        DataType::Float32 => ClaimPreference::defer(
            "ORT's float32 MLAS kernel for this op is measured faster on x86-64 AVX2 at every \
             size (0.02-0.87x session latency); ORT also threads elementwise work across the \
             intra-op pool while this kernel is single-threaded",
        ),

        // Everything else (float64, and any integral dtype a future opset
        // admits) is unmeasured. Unmeasured means defer: the cost of wrongly
        // keeping a claim is a silent latency regression, the cost of wrongly
        // deferring is that ORT runs an op it already has a kernel for.
        other => ClaimPreference::defer(format!(
            "this EP has no measured evidence that it beats ORT's CPU kernel for {} in {other:?}, \
             and an unmeasured claim risks a silent latency regression",
            op.op_type
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, Dim, NodeId, SymbolId, static_shape};

    fn node(op_type: &str, attrs: &[(&str, &str)]) -> Node {
        let mut n = Node::new(NodeId(0), op_type, vec![], vec![]);
        for (k, v) in attrs {
            n.attributes.insert(
                (*k).to_string(),
                Attribute::String((*v).as_bytes().to_vec()),
            );
        }
        n
    }

    fn shape(dims: &[usize]) -> Shape {
        static_shape(dims.iter().copied())
    }

    fn dynamic_shape() -> Shape {
        vec![Dim::Static(1), Dim::Symbolic(SymbolId(0))]
    }

    #[test]
    fn bfloat16_is_always_claimed_because_ort_has_no_kernel() {
        for op in ["Tanh", "Sigmoid", "Gelu", "Sqrt", "Erf"] {
            for shp in [shape(&[1, 1]), shape(&[1, 4_000_000]), dynamic_shape()] {
                let pref = claim_preference(&node(op, &[]), 22, &[shp], &[DataType::BFloat16]);
                assert!(
                    pref.is_claim(),
                    "{op} bfloat16 must stay claimed — ORT cannot run it at all"
                );
            }
        }
    }

    #[test]
    fn float32_activations_are_never_claimed() {
        for op in ["Tanh", "Sigmoid", "Gelu", "Sqrt", "Erf"] {
            for n in [1usize, 32, 3072, 16384, 262_144, 4_000_000] {
                let pref =
                    claim_preference(&node(op, &[]), 22, &[shape(&[1, n])], &[DataType::Float32]);
                assert!(
                    !pref.is_claim(),
                    "{op} float32 n={n} must defer to ORT, got {pref:?}"
                );
                assert!(pref.reason().unwrap().contains("MLAS"));
            }
        }
    }

    #[test]
    fn float16_claims_only_tanh_gelu_below_the_measured_cap() {
        let g = node("Gelu", &[("approximate", "tanh")]);
        for n in [1usize, 512, 3072, 65536, F16_GELU_TANH_MAX_ELEMENTS] {
            assert!(
                claim_preference(&g, 22, &[shape(&[1, n])], &[DataType::Float16]).is_claim(),
                "float16 Gelu(tanh) n={n} is measured >=1.2x and must be claimed"
            );
        }
        for n in [F16_GELU_TANH_MAX_ELEMENTS + 1, 1_048_576] {
            let pref = claim_preference(&g, 22, &[shape(&[1, n])], &[DataType::Float16]);
            assert!(!pref.is_claim(), "float16 Gelu(tanh) n={n} must defer");
        }
    }

    #[test]
    fn float16_exact_gelu_and_the_other_activations_defer() {
        let exact = node("Gelu", &[("approximate", "none")]);
        assert!(
            !claim_preference(&exact, 22, &[shape(&[1, 3072])], &[DataType::Float16]).is_claim()
        );
        // No `approximate` attribute at all means the exact erf path.
        let bare = node("Gelu", &[]);
        assert!(
            !claim_preference(&bare, 22, &[shape(&[1, 3072])], &[DataType::Float16]).is_claim()
        );
        for op in ["Tanh", "Sigmoid", "Sqrt", "Erf"] {
            assert!(
                !claim_preference(
                    &node(op, &[]),
                    22,
                    &[shape(&[1, 3072])],
                    &[DataType::Float16]
                )
                .is_claim(),
                "{op} float16 must defer"
            );
        }
    }

    #[test]
    fn dynamic_shapes_fail_conservative_for_the_size_gated_claim() {
        let g = node("Gelu", &[("approximate", "tanh")]);
        let pref = claim_preference(&g, 22, &[dynamic_shape()], &[DataType::Float16]);
        assert!(!pref.is_claim());
        assert!(pref.reason().unwrap().contains("not static"));
    }

    #[test]
    fn unmeasured_dtypes_defer() {
        for dt in [DataType::Float64, DataType::Int32] {
            let pref = claim_preference(&node("Tanh", &[]), 22, &[shape(&[1, 3072])], &[dt]);
            assert!(!pref.is_claim(), "{dt:?} is unmeasured and must defer");
        }
    }

    #[test]
    fn ungoverned_ops_are_untouched() {
        for op in ["MatMul", "Add", "Relu", "LayerNormalization", "Softmax"] {
            for dt in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
                assert!(
                    claim_preference(&node(op, &[]), 22, &[shape(&[1, 3072])], &[dt]).is_claim(),
                    "{op} is not governed by this policy and must keep its claim"
                );
            }
        }
    }

    #[test]
    fn contrib_domain_ops_of_the_same_name_are_untouched() {
        // `com.microsoft::FastGelu`/`QuickGelu` are separate ops with their own
        // measurements; a bare op-type match must not catch them.
        let mut n = node("Gelu", &[]);
        n.domain = "com.microsoft".to_string();
        assert!(claim_preference(&n, 1, &[shape(&[1, 3072])], &[DataType::Float32]).is_claim());
    }

    #[test]
    fn a_missing_input_dtype_does_not_silently_decline() {
        let pref = claim_preference(&node("Tanh", &[]), 22, &[], &[]);
        assert!(pref.is_claim());
    }
}

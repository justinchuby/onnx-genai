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
//!
//! # Deferring is only sound when the host has a kernel
//!
//! Two ways that assumption fails, both found by measurement rather than
//! reasoning:
//!
//! 1. **The host has no kernel for the dtype.** ORT's CPU EP cannot run any of
//!    these ops in bfloat16; declining turns a working session into a
//!    `NOT_IMPLEMENTED` load failure.
//! 2. **The host has no kernel for the *op*, and inlines it instead.** `Gelu`
//!    is an ONNX function. Outside float32 ORT has no `Gelu` kernel, so
//!    declining makes it inline the function body into
//!    `Cast`/`Pow`/`Mul`/`Sum`/`Tanh`/`Sqrt` — and this EP then claims those
//!    *ungoverned* constituents, which in float16 are far slower than ORT's.
//!    Measured end-to-end, that path is **0.014x** for `Gelu(tanh)` and
//!    0.024-0.049x for `Gelu(none)`, against 0.89x worst case for just keeping
//!    the claim. An element-count cap on float16 `Gelu` was written, tested
//!    that way, and deleted.
//!
//! So a deferral is only added once the profile shows the node actually landing
//! on `CPUExecutionProvider` as a single node afterwards.
//!
//! # When deferral is unsound
//!
//! [`ClaimPreference::DeferToHost`] is a bet that the host has a kernel to run
//! the node with. Under `session.disable_cpu_ep_fallback=1` it does not: a
//! node no EP claims is unassignable and ORT fails session creation. The
//! plugin therefore reads that session option at `CreateEp` and switches this
//! gate off entirely for such sessions (`onnx-runtime-ep-plugin`'s
//! `ExportedEp::host_fallback_available`), so the worst case is a slower
//! kernel rather than a session that will not load. Anything added here may
//! assume a host fallback exists.

use onnx_runtime_ep_api::ClaimPreference;
use onnx_runtime_ir::{DataType, Node, Shape};

/// Ops whose ORT-vs-us performance has been measured and whose claim is
/// therefore governed by this module.
///
/// An op absent from this list keeps the historical "support implies claim"
/// behaviour — this module only ever *removes* claims it has evidence against,
/// it never adds one.
///
/// Exposed so callers can skip materializing shape/dtype metadata for the
/// nodes this module will wave through unconditionally.
pub fn governs(op: &Node) -> bool {
    matches!(
        (op.domain.as_str(), op.op_type.as_str()),
        (
            "" | "ai.onnx",
            "Tanh" | "Sigmoid" | "Gelu" | "Sqrt" | "Erf" | "Softmax"
        ) | ("com.microsoft", "RotaryEmbedding")
    )
}

/// Measured f16 `Gelu` ratios with the claim kept (ORT ns / ours, >1 means we
/// win):
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
/// A cap at 262144 — the last size whose win survives every thread count — was
/// tried and **falsified end-to-end**, which is why this module has no size
/// gate at all. See the function-inlining note below.
///
/// Whether this EP wants `op` handed to it by a plugin host, or would rather
/// the host ran its own kernel.
pub fn claim_preference(
    op: &Node,
    _opset: u64,
    _shapes: &[Shape],
    input_dtypes: &[DataType],
) -> ClaimPreference {
    if !governs(op) {
        return ClaimPreference::Claim;
    }
    let dtype = match input_dtypes.first() {
        Some(dt) => *dt,
        None => return ClaimPreference::Claim,
    };

    // `Gelu` is an ONNX *function*, and ORT's CPU EP only has a real kernel for
    // it in float32. In every other dtype, declining does not hand the node to
    // an ORT kernel — ORT inlines the function body into
    // `Cast`/`Pow`/`Mul`/`Sum`/`Tanh`/`Sqrt`, and this EP then claims the
    // ungoverned constituents, which in float16 are far slower than ORT's.
    // Profiled at 1048576 float16 elements, deferring `Gelu(tanh)` cost 553 ms
    // of claimed `Mul`/`Pow`/`Sum` against ORT's 10.8 ms for the same nodes — a
    // session ratio of 0.014x, against 0.89x worst case for simply keeping the
    // claim. `Gelu(none)` measured 0.024-0.049x the same way. So outside
    // float32 this is a capability claim like bfloat16's, not a performance
    // bet: there is no faster host kernel to defer to.
    if op.op_type == "Gelu" && dtype != DataType::Float32 {
        return ClaimPreference::Claim;
    }

    // `Softmax` and `RotaryEmbedding` are the two attention-adjacent ops with a
    // full measured grid against ORT's CPU EP, and both lose in **every** cell.
    //
    // Session-level interleaved A/B, AMD EPYC 9V74 16C/32T, AVX2+FMA, 5 trials
    // x 9 runs x 3 warmups, ratio = ours/ORT p50, lower is better for us:
    //
    // | model                | t=1  | t=8   | t=16  |
    // |----------------------|-----:|------:|------:|
    // | sm_decode_h32_kv1024 | 1.28 |  2.05 |  2.11 |
    // | sm_decode_h32_kv4096 | 1.34 |  5.06 |  4.60 |
    // | sm_decode_h32_kv8192 | 1.32 |  6.99 |  7.02 |
    // | sm_prefill_h32_s512  | 1.76 |  5.08 |  4.58 |
    // | sm_bert_b8_s128      | 1.56 |  4.89 |  8.83 |
    // | sm_whisper_cross     | 1.73 |  5.17 |  4.67 |
    // | rope_llama3_s1       | 9.70 |  6.93 |  6.89 |
    // | rope_llama3_b8_s1    | 5.90 |  5.77 |  3.95 |
    // | rope_llama3_s128     | 2.79 | 12.59 | 17.21 |
    // | rope_llama3_s512     | 1.53 |  5.78 |  8.83 |
    //
    // That is after this crate's own softmax/RoPE vectorization and
    // parallelization (#1061) and after the host large-allocation cache
    // (#1075), which together took the worst softmax cell from 124x to 7.0x and
    // the worst RoPE cell from 83.8x to 17.2x. The remaining gap is not
    // arithmetic: ORT reaches these through fused attention kernels and splits
    // the work across its intra-op pool at a granularity we do not match. The
    // margin rule is ">= 5% repeatable win at every measured thread count", and
    // the best cell here is 1.28x *against* us, so there is no dtype and no
    // size at which the claim is defensible.
    //
    // Deferring is sound for both: ORT's CPU EP has a float32 `Softmax` kernel
    // (MLAS) and a float32 `com.microsoft::RotaryEmbedding` contrib kernel, and
    // both are single nodes after deferral rather than inlined functions -- the
    // failure mode that killed the float16 `Gelu` size gate does not apply.
    // Non-float32 keeps the claim, for the same reason bfloat16 does below:
    // unmeasured, and in bfloat16 ORT has nothing to defer to.
    if matches!(op.op_type.as_str(), "Softmax" | "RotaryEmbedding") {
        return if dtype == DataType::Float32 {
            ClaimPreference::defer(
                "measured slower than ORT's float32 CPU kernel in every cell of a session-level \
                 A/B grid (1.28-17.2x ours/ORT across 1/8/16 threads, decode and prefill \
                 shapes); ORT has a float32 kernel for this op to defer to",
            )
        } else {
            ClaimPreference::Claim
        };
    }

    match dtype {
        // ORT's CPU EP has no bfloat16 kernel for any of these ops: without
        // this plugin the session fails to create at all
        // (`NOT_IMPLEMENTED: Could not find an implementation for Sqrt(13)`).
        // The claim here is a capability, not a performance bet, so it is
        // unconditional — declining would turn a working session into a
        // load-time failure.
        DataType::BFloat16 => ClaimPreference::Claim,

        // ORT has float16 kernels for these primitives (it casts to float32
        // around them) and runs them faster: measured 0.59-0.96x before this
        // policy, ~1.00x after. Deferring here is a real handover.
        DataType::Float16 => ClaimPreference::defer(
            "measured slower than ORT's float16 CPU kernel at every size on x86-64 AVX2 \
             (0.59-0.96x), and ORT has a float16 kernel for this op to defer to",
        ),

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
    fn float16_gelu_is_claimed_at_every_size_because_ort_would_inline_it() {
        // ORT has no float16 `Gelu` kernel of either approximation. Declining
        // makes it inline the function body, after which this EP claims the
        // ungoverned float16 constituents at 0.014-0.049x. Size and shape are
        // irrelevant to that: there is no host kernel to hand the node to.
        for attrs in [
            &[("approximate", "tanh")][..],
            &[("approximate", "none")][..],
            &[][..],
        ] {
            let g = node("Gelu", attrs);
            for shp in [
                shape(&[1, 1]),
                shape(&[1, 262_144]),
                shape(&[1, 1_048_576]),
                shape(&[1, 64_000_000]),
                dynamic_shape(),
                Shape::new(),
            ] {
                assert!(
                    claim_preference(&g, 22, &[shp], &[DataType::Float16]).is_claim(),
                    "float16 Gelu {attrs:?} must be claimed — deferring costs 20-70x"
                );
            }
        }
    }

    #[test]
    fn float16_primitives_defer_because_ort_really_does_run_them() {
        // Unlike `Gelu`, these are ops ORT's CPU EP has a float16 kernel for,
        // so the profile shows a single node landing on `CPUExecutionProvider`
        // and the measured session ratio goes 0.59-0.96x -> ~1.00x.
        for op in ["Tanh", "Sigmoid", "Sqrt", "Erf"] {
            for shp in [shape(&[1, 3072]), shape(&[1, 1_048_576]), dynamic_shape()] {
                assert!(
                    !claim_preference(&node(op, &[]), 22, &[shp], &[DataType::Float16]).is_claim(),
                    "{op} float16 must defer"
                );
            }
        }
    }

    #[test]
    fn gelu_is_only_ever_deferred_in_float32() {
        // float32 is the one dtype ORT has a real `Gelu` kernel for, verified
        // by profile: a deferred f32 `Gelu` shows up as one `Gelu` node on
        // `CPUExecutionProvider`, not as an inlined subgraph.
        let g = node("Gelu", &[("approximate", "tanh")]);
        assert!(!claim_preference(&g, 22, &[shape(&[1, 3072])], &[DataType::Float32]).is_claim());
        for dt in [
            DataType::Float16,
            DataType::BFloat16,
            DataType::Float64,
            DataType::Int32,
        ] {
            assert!(
                claim_preference(&g, 22, &[shape(&[1, 3072])], &[dt]).is_claim(),
                "{dt:?} Gelu must be claimed — ORT would inline the function instead"
            );
        }
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
        for op in ["MatMul", "Add", "Relu", "LayerNormalization", "Mul"] {
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

    #[test]
    fn no_decision_depends_on_shape() {
        // Every threshold this module once had was falsified by end-to-end
        // measurement, so the predicate is shape-blind by construction. Pinning
        // that here keeps the ambiguity between a rank-0 scalar and an
        // unknown-rank value (both are the empty `Vec<Dim>`) from ever becoming
        // load-bearing again.
        for op in ["Tanh", "Sigmoid", "Gelu", "Sqrt", "Erf"] {
            for dt in [
                DataType::Float32,
                DataType::Float16,
                DataType::BFloat16,
                DataType::Float64,
            ] {
                let n = node(op, &[]);
                let baseline = claim_preference(&n, 22, &[shape(&[1, 3072])], &[dt]).is_claim();
                for shp in [
                    Shape::new(),
                    shape(&[]),
                    shape(&[1]),
                    shape(&[1, 64_000_000]),
                    dynamic_shape(),
                ] {
                    assert_eq!(
                        claim_preference(&n, 22, std::slice::from_ref(&shp), &[dt]).is_claim(),
                        baseline,
                        "{op}/{dt:?} changed its answer for shape {shp:?}"
                    );
                }
                // No shapes supplied at all must agree too.
                assert_eq!(claim_preference(&n, 22, &[], &[dt]).is_claim(), baseline);
            }
        }
    }

    #[test]
    fn governs_matches_exactly_the_ops_the_policy_can_decline() {
        for op in ["Tanh", "Sigmoid", "Gelu", "Sqrt", "Erf", "Softmax"] {
            assert!(governs(&node(op, &[])), "{op} must be governed");
        }
        for op in ["MatMul", "Add", "Relu", "Mul", "LayerNormalization"] {
            assert!(!governs(&node(op, &[])), "{op} must not be governed");
        }
        // `RotaryEmbedding` is governed only in the contrib domain.
        let mut rope = Node::new(NodeId(0), "RotaryEmbedding", vec![], vec![]);
        assert!(
            !governs(&rope),
            "default-domain RotaryEmbedding must not be governed"
        );
        rope.domain = "com.microsoft".to_string();
        assert!(
            governs(&rope),
            "com.microsoft::RotaryEmbedding must be governed"
        );
        let mut contrib = node("Gelu", &[]);
        contrib.domain = "com.microsoft".to_string();
        assert!(!governs(&contrib));

        // The cheap pre-filter callers use to skip metadata materialization
        // must never wave through a node `claim_preference` would decline.
        for op in ["Tanh", "Sigmoid", "Gelu", "Sqrt", "Erf", "Add", "MatMul"] {
            for dt in [
                DataType::Float32,
                DataType::Float16,
                DataType::BFloat16,
                DataType::Float64,
            ] {
                let n = node(op, &[]);
                if !claim_preference(&n, 22, &[shape(&[1, 3072])], &[dt]).is_claim() {
                    assert!(
                        governs(&n),
                        "{op}/{dt:?} is declined but `governs` says it can be skipped"
                    );
                }
            }
        }
    }
    fn ms_node(op_type: &str) -> Node {
        let mut n = Node::new(NodeId(0), op_type, vec![], vec![]);
        n.domain = "com.microsoft".to_string();
        n
    }

    /// Both ops lost every cell of the measured grid, so no shape may keep the
    /// float32 claim. A size gate reintroduced later would fail here.
    #[test]
    fn float32_softmax_and_rope_defer_at_every_shape() {
        let cases: [(Node, &str); 2] = [
            (node("Softmax", &[]), "Softmax"),
            (ms_node("RotaryEmbedding"), "RotaryEmbedding"),
        ];
        for (n, name) in cases {
            for shp in [
                shape(&[1, 1]),
                shape(&[1, 32, 1, 8192]),
                shape(&[8, 32, 512, 512]),
                shape(&[1, 4_000_000]),
                dynamic_shape(),
            ] {
                let pref = claim_preference(&n, 22, &[shp.clone()], &[DataType::Float32]);
                assert!(
                    !pref.is_claim(),
                    "float32 {name} must defer at {shp:?} — it lost every measured cell"
                );
            }
        }
    }

    /// The deferral is a *performance* bet and only float32 was measured.
    /// bfloat16 additionally has no ORT kernel to defer to, so deferring it
    /// would turn a working session into a load failure.
    #[test]
    fn non_float32_softmax_and_rope_keep_the_claim() {
        for dt in [DataType::BFloat16, DataType::Float16, DataType::Float64] {
            for n in [node("Softmax", &[]), ms_node("RotaryEmbedding")] {
                let pref = claim_preference(&n, 22, &[shape(&[1, 32, 1, 4096])], &[dt]);
                assert!(
                    pref.is_claim(),
                    "{} in {dt:?} is unmeasured against ORT and must keep the claim",
                    n.op_type
                );
            }
        }
    }

    /// `RotaryEmbedding` is only governed in the `com.microsoft` domain. A
    /// same-named op in another domain is a different operator and must not
    /// inherit this evidence.
    #[test]
    fn rope_outside_the_microsoft_domain_is_not_governed() {
        let mut n = Node::new(NodeId(0), "RotaryEmbedding", vec![], vec![]);
        n.domain = "com.example".to_string();
        assert!(!governs(&n));
        assert!(
            claim_preference(&n, 22, &[shape(&[1, 32, 1, 4096])], &[DataType::Float32]).is_claim()
        );
    }

    /// Adding `Softmax` to `governs` must not change any pre-existing answer.
    #[test]
    fn previously_governed_ops_are_unchanged() {
        for op in ["Tanh", "Sigmoid", "Gelu", "Sqrt", "Erf"] {
            assert!(governs(&node(op, &[])), "{op} must stay governed");
            assert!(
                claim_preference(
                    &node(op, &[]),
                    22,
                    &[shape(&[1, 1024])],
                    &[DataType::BFloat16]
                )
                .is_claim(),
                "{op} bfloat16 answer changed"
            );
        }
    }
}

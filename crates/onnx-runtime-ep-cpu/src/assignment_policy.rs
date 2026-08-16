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
//! The full matmul-family matrix -- every shape, thread count and ratio behind
//! the decisions in [`matmul_family_preference`] -- is in
//! `docs/performance/CPU_MATMUL_ASSIGNMENT.md`.
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
        ("" | "ai.onnx", "Tanh" | "Sigmoid" | "Gelu" | "Sqrt" | "Erf")
            | ("" | "ai.onnx", "MatMul" | "Gemm" | "QLinearMatMul")
            | ("com.microsoft", "MatMulNBits")
    )
}

/// Smallest `K * N` at which the matmul-family measurements below were taken.
///
/// Every matmul ratio in this module comes from `K = N = 3584` (Qwen3-8B's
/// hidden size), i.e. 12.8 M weights. Below some size the picture must invert —
/// a 2x3 by 3x2 `MatMul` is dominated by per-node dispatch, not arithmetic, and
/// handing it to ORT costs a partition boundary rather than saving time. This
/// module only ever removes a claim it has evidence against, so the deferrals
/// are scoped to the measured region and small matmuls keep the historical
/// behaviour.
///
/// 1 M was chosen as the smallest power of two more than a decimal order of
/// magnitude below the measured point, so the extrapolation is bounded. It is
/// *not* itself a measured crossover: the honest statement is "we know we lose
/// at 12.8 M and we have not measured 1 M", and the [`DECODE_PARALLEL_NOTE`]
/// explains why the loss is expected to grow, not shrink, with size.
const MEASURED_MIN_WEIGHTS: usize = 1 << 20;

/// The single root cause behind almost every matmul deferral here.
///
/// At one thread this EP is at parity with ORT on the dense and int4 paths —
/// `MatMul` f32 M=1 1.00, M=128 0.97; `MatMulNBits` int4 M=1 1.00, M=128 0.99
/// (ratios are ours/ORT, lower is better). The gap opens as threads are added
/// and closes again only at 32, where ORT's own scaling saturates:
///
/// | threads                    | 1    | 2    | 4    | 8    | 16   | 32   |
/// |----------------------------|-----:|-----:|-----:|-----:|-----:|-----:|
/// | `MatMul` f32 M=128         | 0.97 | 2.11 | 1.77 | 1.38 | 1.65 | 0.67 |
/// | `MatMulNBits` int4 M=128   | 0.99 | 1.79 | 2.35 | 2.41 | 2.38 | 3.79 |
/// | `MatMulNBits` int4 M=1     | 1.00 | 1.52 | 1.74 | 2.23 | 2.21 | 4.28 |
///
/// So these are not slow kernels — they are kernels that realise roughly half
/// of ORT's parallel speedup. Fixing that is a threadpool/partitioning problem
/// (#1054 removed one cause, the 8-worker cap on the standalone MLAS pool), not
/// a kernel-tuning one, and until it is fixed the honest answer at capability
/// time is to let ORT run these nodes. The thread count is not visible here, so
/// a claim would have to hold at *every* count; none of these do.
const DECODE_PARALLEL_NOTE: &str = "measured slower than ORT's CPU kernel at every thread count \
     from 2 to 16 on x86-64 AVX2 (this EP realises about half of ORT's parallel speedup; at one \
     thread it is at parity)";

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
    if let Some(preference) = matmul_family_preference(op, _shapes, input_dtypes) {
        return preference;
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

/// Weight-element count for a matmul-family node, or `None` when the shapes are
/// not statically known.
///
/// Dynamic shapes are deliberately *not* treated as "large". A symbolic `K` or
/// `N` could be anything, and the deferrals below are scoped to a measured
/// region; extending them to shapes we cannot size would defer on no evidence.
fn matmul_weight_elements(op: &Node, shapes: &[Shape]) -> Option<usize> {
    if op.op_type == "MatMulNBits" {
        // K and N are attributes, not shapes, so this is exact even when the
        // activation is dynamic -- which it always is at decode.
        let k = op.attributes.get("K")?.as_int()?;
        let n = op.attributes.get("N")?.as_int()?;
        return usize::try_from(k)
            .ok()?
            .checked_mul(usize::try_from(n).ok()?);
    }
    // `MatMul`/`Gemm` take B as input 1, `QLinearMatMul` as input 3.
    let b = shapes.get(if op.op_type == "QLinearMatMul" { 3 } else { 1 })?;
    b.iter()
        .map(|dim| dim.as_static())
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim?))
}

/// The measured assignment matrix for `MatMul`/`Gemm`/`MatMulNBits`/
/// `QLinearMatMul`, or `None` when `op` is not one of them.
///
/// All ratios are **ours/ORT**, lower is better, p50 of an interleaved A/B
/// against ONNX Runtime 1.27 on AMD EPYC 9V74 (AVX2+FMA+F16C, no AVX-512/VNNI),
/// `K = N = 3584`, weights prepacked (steady state), parity asserted on every
/// rep.
fn matmul_family_preference(
    op: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<ClaimPreference> {
    let is_matmul_family = matches!(
        (op.domain.as_str(), op.op_type.as_str()),
        ("" | "ai.onnx", "MatMul" | "Gemm" | "QLinearMatMul") | ("com.microsoft", "MatMulNBits")
    );
    if !is_matmul_family {
        return None;
    }

    // `QLinearMatMul` is the one case that does not need a size gate: it loses
    // by more than an order of magnitude at the smallest shape measured, and it
    // loses for a *structural* reason rather than a tuning one. #1058 bound
    // MLAS's integer QGEMM and took u8 x u8 from 27-119x down to 3-4x, but ORT
    // pre-packs the constant B once at session init while this kernel packs
    // inside every call -- at M=1 a 12.8 MB pack dominates a 1.7 ms call, which
    // is the whole of the remaining 22x. Signed x signed is worse still (3.0x)
    // because MLAS documents `AIsSigned` as unsupported off ARM, so on x86 only
    // u8 x u8 reaches the fast path at all.
    //
    // | shape (K=N=3584, 8 threads) | u8 x u8 | i8 x i8 |
    // |-----------------------------|--------:|--------:|
    // | M=1                         | 22.04   | 2.23    |
    // | M=128                       | 4.19    | 3.02    |
    // | M=512                       | 4.05    | 3.07    |
    //
    // Closing it needs a prepack hook for constant initializers, with its own
    // correctness surface (weight identity, address reuse, dynamic weights).
    // Until that exists ORT should run these nodes.
    if op.op_type == "QLinearMatMul" {
        return Some(ClaimPreference::defer(
            "measured 2.2-22x slower than ORT's CPU QLinearMatMul on x86-64 AVX2: ORT pre-packs \
             the constant B once at session init while this kernel packs inside every call, and \
             MLAS's integer GEMM does not accept signed activations off ARM",
        ));
    }

    let weights = matmul_weight_elements(op, shapes);
    let measured_region = weights.is_some_and(|elements| elements >= MEASURED_MIN_WEIGHTS);
    if !measured_region {
        // Small or dynamically-shaped: unmeasured, and the historical claim
        // stands. Deferring here would trade a partition boundary for nothing.
        return Some(ClaimPreference::Claim);
    }

    if op.op_type == "MatMulNBits" {
        let bits = op.attributes.get("bits").and_then(|a| a.as_int());
        // bits=8 is the one matmul-family range this EP wins outright, and it
        // wins at every thread count measured: 0.15 / 0.36 / 0.25 / 0.25 / 0.23
        // (ours/ORT) at 1 / 2 / 4 / 8 / 16 threads for M=1, and 0.75 at M=128,
        // i.e. 1.3x to 6.7x faster. That is far outside the >= 5% margin rule
        // and it survives the "must hold across thread counts" test, so the
        // claim is kept.
        //
        // It is also *honest* only as of the 8-bit precision gate: this path
        // previously ran an int16-activation GEMV at `accuracy_level = 0`,
        // which bought 6-18% by being ~55x less accurate than ORT and failed
        // output parity. With fp32 activations restored the win above is
        // measured at parity=PASS.
        if bits == Some(8) {
            return Some(ClaimPreference::Claim);
        }
        // int4 is the decode workhorse and this EP still loses it above 1
        // thread -- see `DECODE_PARALLEL_NOTE`. Both accuracy levels measured
        // lose: acc0 M=1 2.23 / M=128 2.41, acc4 M=1 1.78 / M=128 2.11 at 8
        // threads.
        return Some(ClaimPreference::defer(format!(
            "MatMulNBits {}-bit: {DECODE_PARALLEL_NOTE}",
            bits.unwrap_or(4)
        )));
    }

    // `MatMul` / `Gemm`.
    match input_dtypes.first() {
        // float16 is the worst dense region and, unlike f32/int4, it is a
        // kernel problem rather than a scaling one: it loses 2.47x at ONE
        // thread and 5.34-7.77x at 2-32. ORT casts around MLAS's f32 kernels;
        // this EP's f16 path does not reach the same primitive.
        Some(DataType::Float16) => Some(ClaimPreference::defer(
            "measured 2.5x slower than ORT's CPU MatMul at one thread and 5.3-7.8x at 2-32 \
             threads on x86-64 AVX2 (K=N=3584); this is a kernel gap, not a threading one",
        )),
        Some(DataType::Float32) => Some(ClaimPreference::defer(format!(
            "MatMul/Gemm float32: {DECODE_PARALLEL_NOTE}"
        ))),
        // Every other dense dtype is unmeasured for these ops. Unlike the
        // elementwise rule above, the historical claim is kept rather than
        // deferred: `MatMul` is reachable in dtypes ORT's CPU EP has no kernel
        // for, and a deferral there is a load failure, not a slow session.
        _ => Some(ClaimPreference::Claim),
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

    fn nbits_node(bits: i64, k: i64, n: i64) -> Node {
        let mut node = Node::new(NodeId(0), "MatMulNBits", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        node.attributes.insert("bits".into(), Attribute::Int(bits));
        node.attributes.insert("K".into(), Attribute::Int(k));
        node.attributes.insert("N".into(), Attribute::Int(n));
        node
    }

    fn dense_node(op_type: &str) -> Node {
        Node::new(NodeId(0), op_type, vec![], vec![])
    }

    /// The one matmul-family range this EP is measured to win, and it must stay
    /// claimed. If this ever flips, the EP has stopped contributing anything to
    /// the matmul family under a plugin host.
    #[test]
    fn eight_bit_matmulnbits_is_the_range_we_keep() {
        let pref = claim_preference(&nbits_node(8, 3584, 3584), 22, &[], &[DataType::Float32]);
        assert!(
            pref.is_claim(),
            "bits=8 MatMulNBits is 1.3-6.7x faster than ORT at every measured thread count"
        );
    }

    /// int4 `MatMulNBits`, float32 `MatMul`/`Gemm` and float16 `MatMul` are all
    /// measured slower than ORT above one thread, so a plugin host must be left
    /// to run them.
    #[test]
    fn measured_losing_matmul_ranges_defer_to_the_host() {
        let cases: [(Node, Vec<Shape>, DataType, &str); 4] = [
            (
                nbits_node(4, 3584, 3584),
                vec![],
                DataType::Float32,
                "int4 decode",
            ),
            (
                dense_node("MatMul"),
                vec![shape(&[1, 3584]), shape(&[3584, 3584])],
                DataType::Float32,
                "f32 dense",
            ),
            (
                dense_node("MatMul"),
                vec![shape(&[1, 3584]), shape(&[3584, 3584])],
                DataType::Float16,
                "f16 dense",
            ),
            (
                dense_node("Gemm"),
                vec![shape(&[128, 3584]), shape(&[3584, 3584])],
                DataType::Float32,
                "f32 Gemm",
            ),
        ];
        for (node, shapes, dtype, what) in cases {
            let pref = claim_preference(&node, 22, &shapes, &[dtype]);
            assert!(
                !pref.is_claim(),
                "{what}: measured slower than ORT, so the host must be allowed to run it"
            );
        }
    }

    /// `QLinearMatMul` loses at every shape measured, including the smallest,
    /// and for a structural reason (ORT pre-packs B once; this kernel packs per
    /// call). It defers regardless of size.
    #[test]
    fn qlinear_matmul_defers_at_every_size() {
        for b in [shape(&[8, 8]), shape(&[3584, 3584])] {
            let shapes = vec![
                shape(&[1, 8]),
                shape(&[]),
                shape(&[]),
                b,
                shape(&[]),
                shape(&[]),
                shape(&[]),
                shape(&[]),
            ];
            let pref = claim_preference(
                &dense_node("QLinearMatMul"),
                22,
                &shapes,
                &[DataType::Uint8],
            );
            assert!(!pref.is_claim(), "QLinearMatMul is 2.2-22x slower than ORT");
        }
    }

    /// The deferrals are scoped to the region that was actually measured.
    ///
    /// A 2x3 by 3x2 `MatMul` is dominated by per-node dispatch, not arithmetic;
    /// deferring it buys a partition boundary and no measured time. This also
    /// keeps the plugin conformance suite -- which asserts tiny `MatMul` nodes
    /// land on this EP -- meaningful rather than trivially satisfied.
    #[test]
    fn small_and_dynamic_matmuls_are_outside_the_measured_region() {
        let small = claim_preference(
            &dense_node("MatMul"),
            22,
            &[shape(&[2, 3]), shape(&[3, 2])],
            &[DataType::Float32],
        );
        assert!(
            small.is_claim(),
            "a 3x2 MatMul is below the measured region"
        );

        let dynamic = claim_preference(
            &dense_node("MatMul"),
            22,
            &[shape(&[1, 3584]), dynamic_shape()],
            &[DataType::Float32],
        );
        assert!(
            dynamic.is_claim(),
            "a symbolic N could be any size; deferring it would be a guess"
        );

        // The gate is on the *weight* operand, so a huge activation against a
        // small weight is still outside the region.
        let thin = claim_preference(
            &dense_node("MatMul"),
            22,
            &[shape(&[100_000, 4]), shape(&[4, 4])],
            &[DataType::Float32],
        );
        assert!(thin.is_claim(), "K*N = 16 is far below the measured region");
    }

    /// `MatMulNBits` sizes itself from its `K`/`N` attributes, not its shapes,
    /// because the activation is dynamic at decode. Falsifies the alternative
    /// implementation that reads shapes and would therefore never fire.
    #[test]
    fn matmulnbits_is_sized_from_attributes_not_shapes() {
        assert_eq!(
            matmul_weight_elements(&nbits_node(4, 3584, 3584), &[]),
            Some(3584 * 3584),
            "K/N attributes must size the node with no shapes at all"
        );
        // Below the region even with the attributes present.
        let small = claim_preference(&nbits_node(4, 32, 32), 22, &[], &[DataType::Float32]);
        assert!(small.is_claim(), "K*N = 1024 is below the measured region");
    }

    /// The matmul rules must not disturb the elementwise ones, and vice versa.
    #[test]
    fn matmul_rules_do_not_leak_into_the_elementwise_policy() {
        assert!(governs(&dense_node("MatMul")));
        assert!(governs(&nbits_node(4, 3584, 3584)));
        assert!(!governs(&dense_node("Conv")));
        assert!(
            matmul_family_preference(&node("Tanh", &[]), &[], &[DataType::Float32]).is_none(),
            "the matmul matrix must have no opinion about elementwise ops"
        );
        // bfloat16 elementwise is still an unconditional capability claim.
        assert!(
            claim_preference(
                &node("Sqrt", &[]),
                22,
                &[shape(&[1, 1])],
                &[DataType::BFloat16]
            )
            .is_claim()
        );
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
        for op in ["Tanh", "Sigmoid", "Gelu", "Sqrt", "Erf"] {
            assert!(governs(&node(op, &[])), "{op} must be governed");
        }
        for op in ["MatMul", "Gemm", "QLinearMatMul"] {
            assert!(governs(&node(op, &[])), "{op} must be governed");
        }
        assert!(
            governs(&nbits_node(4, 3584, 3584)),
            "MatMulNBits must be governed"
        );
        for op in ["Add", "Relu", "Softmax", "LayerNormalization"] {
            assert!(!governs(&node(op, &[])), "{op} must not be governed");
        }
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
}

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
            | ("com.microsoft", "MatMulNBits" | "RotaryEmbedding")
            | ("com.microsoft", "FastGelu" | "QuickGelu" | "BiasGelu")
    )
}

/// The `com.microsoft` activations, whose claim depends only on dtype.
///
/// # Why these could not be measured until now
///
/// `onnx-runtime-ep-plugin`'s `GetCapability` runs a fail-closed filter that
/// drops any claim containing a node its own `ShapeInference` table declines.
/// That table had no entry for `FastGelu`, `QuickGelu` or `BiasGelu`, so every
/// claim on them was dropped *after* `supports_op` said yes and after this
/// module said `Claim`. The EP simply never ran them under ORT.
///
/// The consequence was not neutral. **None** of these three has a float16 CPU
/// kernel in ORT — the registry lists `tensor(float)` only for all of them — so
/// what happens to an unclaimed float16 node is decided by something else
/// entirely: whether ORT *inlines* the contrib function body. Verified from
/// ORT's own node-level profile with this plugin loaded:
///
/// * `FastGelu` -> inlined into `Identity`/`Cast`/`Mul`/`Add`/`Tanh`; this EP
///   claims a fragment of that body at **0.115-0.464x** of plain ORT.
/// * `QuickGelu` -> inlined into `Cast`/`Sigmoid`/`Cast`/`Mul`; this EP claims
///   *two* fragments, at **0.093-0.220x**.
/// * `BiasGelu` -> **not** inlined. It stays one node that ORT cast-wraps onto
///   its float32 kernel (`Cast`/`Cast`/`BiasGelu`/`Cast`, all on
///   `CPUExecutionProvider`) and runs at 0.96-1.03x of us.
///
/// So the rule is not "claim what ORT cannot run". It is **claim what ORT would
/// otherwise shred into fragments we end up owning anyway**.
///
/// Measured session ratios (ORT ns / ours, >1 = we win), one thread, AVX2,
/// interleaved A/B, reps>=9, 3072 / 65536 / 1048576 elements:
///
/// | op | float32 | float16 claimed | float16 deferred |
/// |---|---|---|---|
/// | `FastGelu` (1 input) | 0.72 / 0.63 / 0.68 | **1.39 / 1.21 / 1.18** | 0.226 / 0.118 / 0.117 |
/// | `FastGelu` (+ bias)  | 0.74 / 0.75 / 0.74 | **1.39 / 1.11 / 0.97** | 0.239 / 0.131 / 0.132 |
/// | `QuickGelu`          | 0.80 / 0.95 / 1.01 | **1.06\* / 0.88 / 0.81** | 0.220 / 0.103 / 0.093 |
/// | `BiasGelu`           | 0.72 / 0.74 / 0.74 | 0.79 / 0.70 / 0.68 | **1.03 / 0.96 / 0.98** |
///
/// \* `QuickGelu` float16 is 1.057 at 512 elements and falls below 1.0 above
/// ~3072. It is claimed anyway because the honest comparison is against the
/// *deferred* column — 0.09-0.22x — not against 1.0. `BiasGelu` is the one op
/// here whose deferred column is a real handover, so it is the one this module
/// declines in float16 as well as float32.
fn contrib_activation(op: &Node) -> bool {
    op.domain == "com.microsoft"
        && matches!(op.op_type.as_str(), "FastGelu" | "QuickGelu" | "BiasGelu")
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
/// The `approximate="none"` variant used to be the outlier of this table at
/// 0.059-0.414x, because it evaluated `libm::erf` in `f64` per element. With
/// MLAS's `erf` polynomial ported into the AVX2 path it measures 0.898x /
/// 1.064x / 0.908x / 0.917x / 0.927x / 0.898x at 512 / 3072 / 16384 / 65536 /
/// 262144 / 1048576 float16 elements on one thread — a 2.9-15.2x lift on a
/// range this module *does* claim, and the reason that claim is no longer an
/// embarrassment. It is still slightly under 1.0 above 3072 elements, but the
/// alternative measured 0.024-0.049x, so the claim remains the better of the
/// two available options rather than a win in its own right.
///
/// Whether this EP wants `op` handed to it by a plugin host, or would rather
/// the host ran its own kernel.
pub fn claim_preference(
    op: &Node,
    _opset: u64,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> ClaimPreference {
    if !governs(op) {
        return ClaimPreference::Claim;
    }
    if let Some(preference) = matmul_family_preference(op, shapes, input_dtypes) {
        return preference;
    }
    let dtype = match input_dtypes.first() {
        Some(dt) => *dt,
        None => return ClaimPreference::Claim,
    };

    if contrib_activation(op) {
        return match dtype {
            // ORT's own float32 contrib kernels win at every measured size, and
            // a declined node stays a single `FastGelu`/`QuickGelu`/`BiasGelu`
            // node on `CPUExecutionProvider` — verified by profile, not assumed.
            DataType::Float32 => ClaimPreference::defer(
                "ORT's float32 com.microsoft activation kernel is measured faster at every size \
                 on x86-64 AVX2 (0.57-0.75x for FastGelu, 0.72-0.74x for BiasGelu, 0.75-1.02x \
                 for QuickGelu), and declining leaves the node intact on ORT's CPU EP",
            ),
            // The one float16 op in this family ORT does not inline.
            DataType::Float16 if op.op_type == "BiasGelu" => ClaimPreference::defer(
                "ORT does not inline float16 BiasGelu — it stays one node cast-wrapped onto \
                 ORT's float32 kernel and runs 1.27-1.47x faster than this EP does, so \
                 declining is a real handover rather than a fragmentation",
            ),
            // ORT inlines these two instead, and this EP then owns slower
            // ungoverned constituents of the function body at 0.09-0.24x.
            DataType::Float16 | DataType::BFloat16 => ClaimPreference::Claim,
            other => ClaimPreference::defer(format!(
                "this EP has no measured evidence that it beats ORT's CPU kernel for {} in \
                 {other:?}, and an unmeasured claim risks a silent latency regression",
                op.op_type
            )),
        };
    }

    // `com.microsoft::RotaryEmbedding` loses to ORT's CPU contrib kernel in
    // **every** cell of a session-level grid. Interleaved A/B, AMD EPYC 9V74
    // 16C/32T AVX2+FMA, 5 trials x 9 runs x 3 warmups, ratio = ours/ORT p50
    // (lower is better for us):
    //
    // | model             | t=1  | t=8   | t=16  |
    // |-------------------|-----:|------:|------:|
    // | rope_llama3_s1    | 9.70 |  6.93 |  6.89 |
    // | rope_llama3_b8_s1 | 5.90 |  5.77 |  3.95 |
    // | rope_llama3_s128  | 2.79 | 12.59 | 17.21 |
    // | rope_llama3_s512  | 1.53 |  5.78 |  8.83 |
    //
    // That is *after* this crate's own RoPE vectorization and parallelization,
    // which took the worst cell from 83.75x to 17.21x, and after the host
    // large-allocation cache. The best cell is still 1.53x against us, so under
    // this module's ">= 5% win at every measured thread count" rule there is no
    // size and no thread count at which the claim survives.
    //
    // Deferring is a real handover here, not the `Gelu` inlining trap: ORT's
    // CPU EP has a genuine `com.microsoft::RotaryEmbedding` contrib kernel
    // (`contrib_ops/cpu/bert/rotary_embedding.cc`), and the node stays a single
    // node afterwards. `input_dtypes.first()` is input 0, the float data --
    // `position_ids` is input 1 and int64, so it cannot be mistaken for the
    // governing dtype. Only float32 defers; float16 and bfloat16 are unmeasured
    // here and keep the claim, which is also what keeps a bfloat16 session
    // loading at all.
    //
    // `Softmax` was measured the same way and loses just as badly (1.28x-8.83x
    // across the same grid), but is deliberately **not** governed. It is the
    // anchor of this repo's own attention fusion --
    // `MatMul -> (Mul|Div) -> [Add(mask)] -> Softmax -> MatMul` collapses into a
    // fused SDPA kernel in `onnx-runtime-optimizer`'s fusion pass, which runs on
    // plugin-claimed subgraphs. Deferring the standalone node would remove the
    // anchor and fragment the SDPA core across the EP boundary (claim QK^T,
    // hand the scores to ORT, claim the PV matmul), which is worse than either
    // the fused kernel or deferring the whole block. The grid above was measured
    // on isolated single-op graphs under `ORT_DISABLE_ALL`, so it does not
    // predict that case, and a claim to defer it needs a fused-graph
    // measurement this module does not have.
    if op.op_type == "RotaryEmbedding" {
        return if dtype == DataType::Float32 {
            ClaimPreference::defer(
                "measured slower than ORT's float32 CPU contrib kernel in every cell of a \
                 session-level A/B grid (1.53-17.2x ours/ORT across 1/8/16 threads, decode and \
                 prefill shapes); ORT has a float32 kernel for this op to defer to",
            )
        } else {
            ClaimPreference::Claim
        };
    }

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
        // Gelu(none) 0.67-0.76x, Erf 0.66-0.74x. `Sqrt` is now 1.1-1.9x
        // single-threaded above ~8 K elements, but inverts to 0.30x at 16
        // threads, and the thread count is not visible here — so it loses under
        // the "must hold across thread counts" rule too.
        //
        // `Erf` and `Gelu(none)` used to sit at 0.022-0.25x, an order of
        // magnitude worse than the rest, because both evaluated `libm::erf` in
        // `f64` per element. Porting MLAS's own `erf` polynomial lifted them by
        // 3-28x onto the same ~0.7x plateau as every other activation here.
        // That plateau is not the transcendental any more: it is this kernel
        // being single-threaded while ORT spreads the same elementwise work
        // over its intra-op pool. Until that changes, deferring stays correct.
        DataType::Float32 => ClaimPreference::defer(
            "ORT's float32 MLAS kernel for this op is measured faster on x86-64 AVX2 at every \
             size (0.58-0.87x session latency); ORT also threads elementwise work across the \
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
/// The largest statically-known row count where this EP still beats ORT at
/// 8-bit `MatMulNBits`. Measured crossover is between 128 and 256 rows.
const MEASURED_MAX_CLAIMED_ROWS: usize = 256;

/// The row count `M` of the activation, when it is statically known.
///
/// The activation is input 0 and is `[.., M, K]`, so `M` is the second-to-last
/// dimension. Returns `None` for a dynamic or rank-deficient activation, which
/// is the decode case and must not be mistaken for a small one.
fn matmul_rows(shapes: &[Shape]) -> Option<usize> {
    let activation = shapes.first()?;
    // The GEMM row count of `[.., M, K]` is the product of every dim but the
    // last, so a statically batched `[4, 100, K]` is 400 rows and belongs in
    // the wide-prefill region even though no single dim reaches the threshold.
    let leading = activation.len().checked_sub(1)?;
    if leading == 0 {
        return None;
    }
    activation[..leading]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim.as_static()?))
}

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
    // `MatMul`/`Gemm` take B as input 1. `QLinearMatMul` never reaches here:
    // it loses at every size measured, so it is decided before the size gate.
    let b = shapes.get(1)?;
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
        // Deferring is only sound when the host actually has a kernel, so this
        // gate comes before *every* MatMulNBits deferral below. This EP accepts
        // any power-of-two `block_size` >= 16, but ORT's CPU MatMulNBits
        // `ORT_ENFORCE`s `block_size` in {16, 32, 64, 128, 256} and that check
        // throws at *kernel construction*, so handing it a 512-wide block turns
        // a working session into a load failure rather than a slow one. Keep
        // the claim outside ORT's accepted set, exactly as the dense arm keeps
        // dtypes ORT has no kernel for. `bits` needs no equivalent guard: both
        // runtimes accept exactly {2, 4, 8}.
        let block_size = op.attributes.get("block_size").and_then(|a| a.as_int());
        if !matches!(block_size, Some(16 | 32 | 64 | 128 | 256)) {
            return Some(ClaimPreference::Claim);
        }
        // bits=8 is the one matmul-family range this EP wins outright, and it
        // wins at every thread count measured: 0.15 / 0.36 / 0.25 / 0.25 / 0.23
        // (ours/ORT) at 1 / 2 / 4 / 8 / 16 threads for M=1, i.e. 2.8x to 6.7x
        // faster at decode. That is far outside the >= 5% margin rule
        // and it survives the "must hold across thread counts" test, so the
        // claim is kept.
        //
        // It is also *honest* only as of the 8-bit precision gate: this path
        // previously ran an int16-activation GEMV at `accuracy_level = 0`,
        // which bought 6-18% by being ~55x less accurate than ORT and failed
        // output parity. With fp32 activations restored the win above is
        // measured at parity=PASS.
        if bits == Some(8) {
            // ...but only up to M=128. The win erodes with the row count and
            // crosses over between 128 and 256 (ours/ORT, K=N=3584, p50, and
            // these are the *low*-dispersion points, spread 0.01-0.04):
            //
            // | rows | T=2  | T=4  | T=8  |
            // |------|------|------|------|
            // | 128  | 0.90 | 0.87 | 0.94 |
            // | 256  | 1.17 | 1.15 | 0.99 |
            // | 512  | 1.41 | 1.39 | 1.25 |
            //
            // A node whose row count is *statically* >= 256 is pure prefill:
            // there is no decode traffic on it to amortise the loss, so it
            // goes to the host.
            //
            // A dynamic row count is the LLM case, where one node serves both
            // decode and prefill and the choice cannot be split. Claiming is
            // right there by a wide margin: at 8 threads decode saves 6.06 ms
            // per token (1.78 vs 7.84 ms) while a 512-row prefill costs 7.69 ms
            // once (38.58 vs 30.89), so the *second* generated token has
            // already repaid the whole prefill loss.
            if matmul_rows(shapes).is_some_and(|rows| rows >= MEASURED_MAX_CLAIMED_ROWS) {
                return Some(ClaimPreference::defer(
                    "8-bit MatMulNBits wins at decode but loses on wide prefill: 1.15-1.17 \
                     (256 rows) and 1.25-1.41 (512 rows) ours/ORT at 2-8 threads, and a \
                     statically wide node has no decode traffic to amortise that against",
                ));
            }
            return Some(ClaimPreference::Claim);
        }
        // int4 is the decode workhorse and this EP still loses it above 1
        // thread -- see `DECODE_PARALLEL_NOTE`. Both accuracy levels measured
        // lose: acc0 M=1 2.23 / M=128 2.41, acc4 M=1 1.78 / M=128 2.11 at 8
        // threads.
        // 4-bit is measured; 2-bit shares the same dequant-then-GEMM structure
        // and the same threadpool, so it is deferred by extrapolation and says
        // so rather than borrowing 4-bit's numbers.
        return Some(match bits {
            Some(2) => ClaimPreference::defer(
                "MatMulNBits 2-bit: not measured directly, but it shares the dequant-then-GEMM \
                 structure and threadpool with 4-bit, which is measured slower than ORT at \
                 every thread count above 1",
            ),
            other => ClaimPreference::defer(format!(
                "MatMulNBits {}-bit: {DECODE_PARALLEL_NOTE}",
                other.unwrap_or(4)
            )),
        });
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
        nbits_node_blocked(bits, k, n, 32)
    }

    fn nbits_node_blocked(bits: i64, k: i64, n: i64, block_size: i64) -> Node {
        let mut node = Node::new(NodeId(0), "MatMulNBits", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        node.attributes.insert("bits".into(), Attribute::Int(bits));
        node.attributes.insert("K".into(), Attribute::Int(k));
        node.attributes.insert("N".into(), Attribute::Int(n));
        node.attributes
            .insert("block_size".into(), Attribute::Int(block_size));
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
            "bits=8 MatMulNBits is 2.8-6.7x faster than ORT at decode at every measured thread \
             count"
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

    /// The 8-bit keep is bounded by row count, not unconditional. A statically
    /// wide node is pure prefill, where this EP is measured to lose; a dynamic
    /// row count is decode, where it wins by 4-6x, and must stay claimed.
    #[test]
    fn eight_bit_is_kept_for_decode_and_declined_for_static_wide_prefill() {
        let act = |rows: Dim| vec![vec![rows, Dim::Static(3584)], vec![]];
        for rows in [1, 32, 128, 255] {
            let pref = claim_preference(
                &nbits_node(8, 3584, 3584),
                22,
                &act(Dim::Static(rows)),
                &[DataType::Float32],
            );
            assert!(
                pref.is_claim(),
                "{rows} rows is inside the measured 8-bit win"
            );
        }
        for rows in [256, 512, 4096] {
            let pref = claim_preference(
                &nbits_node(8, 3584, 3584),
                22,
                &act(Dim::Static(rows)),
                &[DataType::Float32],
            );
            assert!(
                !pref.is_claim(),
                "{rows} rows is measured 1.15-1.41 ours/ORT with no decode traffic to amortise it"
            );
        }
        // Decode: the row count is symbolic and one node serves both phases.
        let pref = claim_preference(
            &nbits_node(8, 3584, 3584),
            22,
            &act(Dim::Symbolic(SymbolId(0))),
            &[DataType::Float32],
        );
        assert!(
            pref.is_claim(),
            "a dynamic row count is the decode case, worth 6.06 ms/token"
        );
    }

    /// Deferring is only sound when ORT has a kernel. ORT's CPU `MatMulNBits`
    /// `ORT_ENFORCE`s `block_size` in {16, 32, 64, 128, 256} at kernel
    /// construction, while this EP accepts any power-of-two >= 16. Handing ORT
    /// a 512-wide block would turn a working session into a load failure, so
    /// those must stay claimed even though they are in the losing range.
    #[test]
    fn block_sizes_ort_cannot_build_stay_claimed() {
        for block_size in [16, 32, 64, 128, 256] {
            let pref = claim_preference(
                &nbits_node_blocked(4, 3584, 3584, block_size),
                22,
                &[],
                &[DataType::Float32],
            );
            assert!(
                !pref.is_claim(),
                "block_size={block_size} is inside ORT's accepted set and loses, so it must defer"
            );
        }
        for block_size in [512, 1024] {
            let pref = claim_preference(
                &nbits_node_blocked(4, 3584, 3584, block_size),
                22,
                &[],
                &[DataType::Float32],
            );
            assert!(
                pref.is_claim(),
                "block_size={block_size} would fail ORT_ENFORCE, so deferring it is a load failure"
            );
        }
        // A missing `block_size` is a malformed node this EP would reject
        // anyway; treat it as outside ORT's set rather than guessing.
        let mut node = nbits_node(4, 3584, 3584);
        node.attributes.remove("block_size");
        assert!(
            claim_preference(&node, 22, &[], &[DataType::Float32]).is_claim(),
            "an absent block_size must not be assumed to be one ORT accepts"
        );
    }

    /// Deferral reasons are surfaced to users, so a literal that lost its `\`
    /// continuations (and therefore carries runs of indentation whitespace)
    /// must not ship. Falsifies exactly the defect that shipped in `b36e55a72`.
    #[test]
    fn deferral_reasons_are_single_spaced_and_substantive() {
        let mut seen = 0usize;
        let mut check = |pref: ClaimPreference| {
            if let ClaimPreference::DeferToHost { reason } = pref {
                assert!(
                    !reason.contains("  "),
                    "deferral reason has a whitespace run, so it lost a line continuation: \
                     {reason:?}"
                );
                assert!(
                    reason.len() > 40 && !reason.contains('\n'),
                    "deferral reason must be one substantive line: {reason:?}"
                );
                seen += 1;
            }
        };
        for bits in [2, 4] {
            check(claim_preference(
                &nbits_node(bits, 3584, 3584),
                22,
                &[],
                &[DataType::Float32],
            ));
        }
        check(claim_preference(
            &nbits_node(8, 3584, 3584),
            22,
            &[shape(&[512, 3584]), vec![]],
            &[DataType::Float32],
        ));
        check(claim_preference(
            &dense_node("QLinearMatMul"),
            21,
            &[shape(&[1, 3584]), shape(&[3584, 3584])],
            &[DataType::Uint8],
        ));
        for dtype in [DataType::Float32, DataType::Float16] {
            check(claim_preference(
                &dense_node("MatMul"),
                21,
                &[shape(&[128, 3584]), shape(&[3584, 3584])],
                &[dtype],
            ));
        }
        assert!(
            seen >= 5,
            "expected to exercise several deferral reasons, saw {seen}"
        );
    }

    /// Regression for the interaction of the two gates: an 8-bit node that is
    /// statically wide *and* carries a block size ORT cannot build must stay
    /// claimed. The row gate alone would defer it, and the host would then fail
    /// `ORT_ENFORCE` at kernel construction -- a load failure, not a slow node.
    /// This is why the block-size keep is evaluated before every deferral.
    #[test]
    fn wide_eight_bit_with_a_block_size_ort_rejects_stays_claimed() {
        let wide = vec![vec![Dim::Static(512), Dim::Static(3584)], vec![]];
        for block_size in [512, 1024] {
            assert!(
                claim_preference(
                    &nbits_node_blocked(8, 3584, 3584, block_size),
                    22,
                    &wide,
                    &[DataType::Float32],
                )
                .is_claim(),
                "bits=8, {block_size}-wide blocks, 512 rows: deferring is a session-load failure"
            );
        }
        // Same shape, a block size ORT *can* build: now the row gate governs.
        assert!(
            !claim_preference(
                &nbits_node_blocked(8, 3584, 3584, 128),
                22,
                &wide,
                &[DataType::Float32],
            )
            .is_claim(),
            "with a buildable block size the wide-prefill loss must still defer"
        );
    }

    /// Rows are the product of every dim but the last, so a statically batched
    /// activation lands in the wide-prefill region even when no single dim
    /// reaches the threshold. Falsifies reading only the second-to-last dim.
    #[test]
    fn batched_rows_are_folded_not_read_from_one_dim() {
        let batched = vec![
            vec![Dim::Static(4), Dim::Static(100), Dim::Static(3584)],
            vec![],
        ];
        assert_eq!(matmul_rows(&batched), Some(400));
        assert!(
            !claim_preference(
                &nbits_node(8, 3584, 3584),
                22,
                &batched,
                &[DataType::Float32]
            )
            .is_claim(),
            "4 x 100 = 400 rows is wide prefill even though no single dim is >= 256"
        );
        // One symbolic dim anywhere in the batch makes the whole count dynamic.
        let partly_dynamic = vec![
            vec![
                Dim::Symbolic(SymbolId(0)),
                Dim::Static(100),
                Dim::Static(3584),
            ],
            vec![],
        ];
        assert_eq!(matmul_rows(&partly_dynamic), None);
        assert!(
            claim_preference(
                &nbits_node(8, 3584, 3584),
                22,
                &partly_dynamic,
                &[DataType::Float32]
            )
            .is_claim(),
            "an unknown batch size is the decode case and must stay claimed"
        );
        // A rank-1 activation has no row dim to fold.
        assert_eq!(matmul_rows(&[vec![Dim::Static(3584)], vec![]]), None);
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
    fn contrib_activations_defer_in_float32_and_claim_in_float16() {
        // float32: ORT's own contrib kernel wins and a declined node stays a
        // single node on `CPUExecutionProvider`, so deferring is a real
        // handover rather than a function inlining.
        for op in ["FastGelu", "QuickGelu", "BiasGelu"] {
            let mut n = node(op, &[]);
            n.domain = "com.microsoft".to_string();
            for shp in [shape(&[1, 3072]), shape(&[1, 1_048_576]), dynamic_shape()] {
                let pref =
                    claim_preference(&n, 1, std::slice::from_ref(&shp), &[DataType::Float32]);
                assert!(!pref.is_claim(), "float32 {op} must defer, got {pref:?}");
            }
        }

        // float16: no op here has a float16 ORT kernel, but ORT *inlines*
        // `FastGelu`/`QuickGelu` when they go unclaimed, and this EP then picks
        // up the slower pieces at 0.09-0.24x. `BiasGelu` is the exception — ORT
        // keeps it whole (cast-wrapped onto its float32 kernel) and wins.
        for op in ["FastGelu", "QuickGelu"] {
            let mut n = node(op, &[]);
            n.domain = "com.microsoft".to_string();
            for shp in [shape(&[1, 512]), shape(&[1, 1_048_576]), dynamic_shape()] {
                assert!(
                    claim_preference(&n, 1, std::slice::from_ref(&shp), &[DataType::Float16])
                        .is_claim(),
                    "float16 {op} must be claimed — deferring inlines it at 0.09-0.24x"
                );
            }
        }
        let mut bias_gelu = node("BiasGelu", &[]);
        bias_gelu.domain = "com.microsoft".to_string();
        assert!(
            !claim_preference(&bias_gelu, 1, &[shape(&[1, 65536])], &[DataType::Float16])
                .is_claim(),
            "float16 BiasGelu must defer — ORT keeps it whole and runs it 1.27-1.47x faster"
        );
    }

    #[test]
    fn contrib_activations_keep_the_bfloat16_capability_claim() {
        // ORT's CPU EP has no bfloat16 kernel for any of these, and no
        // bfloat16 function body to inline either, so declining would turn a
        // working session into a load failure.
        for op in ["FastGelu", "QuickGelu", "BiasGelu"] {
            let mut n = node(op, &[]);
            n.domain = "com.microsoft".to_string();
            assert!(
                claim_preference(&n, 1, &[shape(&[1, 3072])], &[DataType::BFloat16]).is_claim(),
                "bfloat16 {op} must stay claimed — ORT cannot run it at all"
            );
        }
    }

    #[test]
    fn the_contrib_activation_rule_is_domain_scoped() {
        // `ai.onnx` has no op of these names, but a model is free to declare
        // one, and it would not be the contrib kernel this policy measured.
        for op in ["FastGelu", "QuickGelu", "BiasGelu"] {
            let n = node(op, &[]);
            assert!(
                claim_preference(&n, 22, &[shape(&[1, 3072])], &[DataType::Float32]).is_claim(),
                "default-domain {op} is not what this policy measured and must be untouched"
            );
            assert!(!governs(&n), "default-domain {op} must not be governed");
        }
        // `com.microsoft::Gelu` keeps its unconditional claim: it is a
        // different kernel with its own (already-measured) behaviour.
        let mut g = node("Gelu", &[]);
        g.domain = "com.microsoft".to_string();
        assert!(claim_preference(&g, 1, &[shape(&[1, 3072])], &[DataType::Float32]).is_claim());
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
    fn ms_node(op_type: &str) -> Node {
        let mut n = Node::new(NodeId(0), op_type, vec![], vec![]);
        n.domain = "com.microsoft".to_string();
        n
    }

    /// RoPE lost every cell of the measured grid, so no shape may keep the
    /// float32 claim. A size gate reintroduced later fails here.
    #[test]
    fn float32_rope_defers_at_every_shape() {
        let n = ms_node("RotaryEmbedding");
        for shp in [
            shape(&[1, 1, 32, 128]),
            shape(&[8, 1, 32, 128]),
            shape(&[1, 512, 32, 128]),
            shape(&[1, 4_000_000]),
            dynamic_shape(),
        ] {
            let pref = claim_preference(&n, 22, std::slice::from_ref(&shp), &[DataType::Float32]);
            assert!(
                !pref.is_claim(),
                "float32 RotaryEmbedding must defer at {shp:?} — it lost every measured cell"
            );
        }
    }

    /// The deferral is a performance bet and only float32 was measured.
    /// bfloat16 additionally has no ORT kernel in some builds, so deferring it
    /// could turn a working session into a load failure.
    #[test]
    fn non_float32_rope_keeps_the_claim() {
        for dt in [DataType::BFloat16, DataType::Float16, DataType::Float64] {
            let pref = claim_preference(
                &ms_node("RotaryEmbedding"),
                22,
                &[shape(&[1, 512, 32, 128])],
                &[dt],
            );
            assert!(
                pref.is_claim(),
                "RotaryEmbedding in {dt:?} is unmeasured against ORT and must keep the claim"
            );
        }
    }

    /// `RotaryEmbedding` is governed only in the `com.microsoft` domain; a
    /// same-named op elsewhere is a different operator.
    #[test]
    fn rope_outside_the_microsoft_domain_is_not_governed() {
        let n = Node::new(NodeId(0), "RotaryEmbedding", vec![], vec![]);
        assert!(!governs(&n));
        assert!(
            claim_preference(&n, 22, &[shape(&[1, 512, 32, 128])], &[DataType::Float32]).is_claim()
        );
    }

    /// `Softmax` measured just as badly as RoPE but is deliberately left
    /// ungoverned: it anchors this repo's attention fusion, and deferring the
    /// standalone node would fragment the fused SDPA core across the EP
    /// boundary. This test is the reason a future "Softmax loses, defer it"
    /// change has to argue with a fused-graph measurement first.
    #[test]
    fn softmax_stays_claimed_because_it_anchors_attention_fusion() {
        assert!(!governs(&node("Softmax", &[])));
        for dt in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            assert!(
                claim_preference(
                    &node("Softmax", &[]),
                    22,
                    &[shape(&[1, 32, 1, 4096])],
                    &[dt]
                )
                .is_claim(),
                "Softmax must keep its claim in {dt:?} — it is the attention-fusion anchor"
            );
        }
    }

    /// Adding `RotaryEmbedding` must not perturb any pre-existing answer,
    /// including the whole matmul family this module also governs.
    #[test]
    fn previously_governed_ops_are_unchanged_by_the_rope_addition() {
        for op in [
            "Tanh",
            "Sigmoid",
            "Gelu",
            "Sqrt",
            "Erf",
            "MatMul",
            "Gemm",
            "QLinearMatMul",
        ] {
            assert!(governs(&node(op, &[])), "{op} must stay governed");
        }
        assert!(
            governs(&ms_node("MatMulNBits")),
            "MatMulNBits must stay governed"
        );
        for op in ["Tanh", "Sigmoid", "Gelu", "Sqrt", "Erf"] {
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

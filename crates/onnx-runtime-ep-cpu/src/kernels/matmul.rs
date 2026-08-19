//! `MatMul`: numpy-style matrix multiplication for floating-point tensors,
//! including batched and broadcast leading dimensions and 1-D vector operands
//! (`docs/architecture/ORT2.md` §4.4).
//!
//! ## Perf seam (Phase-1.5)
//!
//! The 2-D tile GEMM ([`gemm`]) dispatches on [`CpuBackend::auto_detect`]
//! (`docs/architecture/ORT2.md` §25.2):
//!
//! * **Generic** (default fallback, always compiled, offline): a blocked,
//!   register-tiled, rayon-parallelized pure-Rust f32 GEMM ([`gemm_generic`]).
//!   It is the correctness baseline and contains no `unsafe`.
//! * **`SimdX86`** (default on AVX2/FMA x86-64, runtime-detected): an
//!   MLAS-style packed SIMD f32 SGEMM ([`x86_sgemm`]) — panel packing + a
//!   `6×16` AVX2/FMA register microkernel + K/N cache blocking, parallelized
//!   over column strips. Selected automatically with no cargo feature; falls
//!   back to Generic when AVX2/FMA is absent.
//! * **`Mlas`** (opt-in `mlas` feature on x86-64): vendored MLAS f32 SGEMM,
//!   selected only with `NXRT_CPU_GEMM_BACKEND=mlas`. Multi-threaded — MLAS
//!   partitions the GEMM and runs the tiles across the process Rayon pool — but
//!   kept opt-in (not an automatic default) pending a later slice.
//!
//! The batched / broadcast / 1-D-vector handling in [`matmul_dense`] is
//! backend-agnostic; only the inner 2-D tile GEMM changes. The session also
//! marks graph-initializer inputs so this kernel can safely prepack constants.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, broadcast_shapes, compute_contiguous_strides,
};
use rayon::prelude::*;

use super::check_arity;
use super::governed_weight_cache::{CacheVerdict, GovernedWeightCache};
use super::half_gemm::{self, HalfFormat, MatrixLayout};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::half_gemv;
use super::weight_transpose::{self, WeightTransposeKey};
use crate::backend::CpuBackend;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::{next_index, numel};

// MLAS-style packed SIMD f32 GEMM (the `SimdX86` backend). Kept in a sibling
// file but included here so `kernels/mod.rs` needs no edit; it is an internal
// perf detail of the MatMul hot path, not a new op.
#[path = "x86_sgemm.rs"]
mod x86_sgemm;

// Native BF16×BF16→FP32 GEMM (`_mm512_dpbf16_ps`) for avx512_bf16 hosts. It is
// runtime-detected and otherwise falls back to the portable blocked half GEMM.
#[path = "x86_bf16.rs"]
mod x86_bf16;

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[path = "accelerate_gemm.rs"]
mod accelerate_gemm;

/// Re-export the f16 GEMV for use by sibling kernels (FusedMatMulBias).
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) use accelerate_gemm::neon_gemv_f16_col_parallel;

/// Test-only counter: incremented each time the FP16 GEMV decode path is reached
/// inside [`MatMulKernel::execute_with_backend`]. Guards against dispatch
/// regressions where a broader half-precision GEMM intercepts M=1 decode before
/// the bandwidth-optimal GEMV.
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "ios")
))]
static GEMV_F16_TEST_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Test-only counter: incremented each time the column-major GEMV decode path
/// is reached (M=1, constant column-major B, zero-copy from mmap'd data).
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "ios")
))]
static GEMV_F16_COLMAJ_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Test-only counter: incremented each time the thin-M GEMM path is reached
/// (M=2..16, large N×K, f32 on macOS with pre-transposed B).
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "ios")
))]
static THIN_M_GEMM_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Test-only counter: incremented each time the non-contiguous f16 rescue block
/// is reached (M≥2, non-contiguous constant B on macOS).
#[cfg(all(test, any(target_os = "macos", target_os = "ios")))]
static NONCONTIG_RESCUE_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Test-only counter: incremented each time the BNNS fp16→f32 prefill path is
/// reached inside [`try_matmul_half`]. Guards against dispatch regressions where
/// the portable half GEMM intercepts M≥2 fp16 on macOS before the AMX-backed
/// BNNS path.
#[cfg(all(test, any(target_os = "macos", target_os = "ios")))]
static BNNS_F16_TEST_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Always-on counter: tracks total BNNS fp16 prefill calls for diagnostics.
/// Queryable via [`bnns_prefill_stats`].
#[cfg(any(target_os = "macos", target_os = "ios"))]
static BNNS_PREFILL_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Cumulative nanoseconds spent in BNNS fp16 prefill calls.
#[cfg(any(target_os = "macos", target_os = "ios"))]
static BNNS_PREFILL_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Returns (call_count, cumulative_nanos) for BNNS fp16 prefill calls.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn bnns_prefill_stats() -> (usize, u64) {
    use std::sync::atomic::Ordering;
    (
        BNNS_PREFILL_CALLS.load(Ordering::Relaxed),
        BNNS_PREFILL_NANOS.load(Ordering::Relaxed),
    )
}

/// Returns the number of entries in the process-global weight-transpose caches.
/// (f16_entries, f32_entries). Used by benchmarks to verify cache reuse across turns.
///
/// For memory questions use [`weight_transpose_cache_bytes`] instead: an entry
/// count cannot distinguish a cache holding kilobytes from one holding gigabytes.
pub fn weight_transpose_cache_sizes() -> (usize, usize) {
    weight_transpose::cache_sizes()
}

/// Bytes of transposed weight held by the process-global weight-transpose caches.
///
/// Each entry is a full `K x N` copy of a constant weight kept for the session,
/// so this scales with model size and belongs in the memory plan (#1056).
pub fn weight_transpose_cache_bytes() -> usize {
    weight_transpose::cache_bytes()
}

/// Admit (`true`) or decline (`false`) the process-global weight-transpose
/// caches for the rest of the session (#1056).
///
/// When declined, the `MatMul`/`Gemm` kernels compute each constant-weight
/// transpose per call and retain nothing, so the resident, session-lifetime,
/// weight-scaled `K x N` copies never accrue. The engine calls this once at
/// load from the memory-strategy plan's verdict, beside the identical gates for
/// the resident dequant f32 cache (#987) and the MLAS SQNBit packed buffer
/// (#1051), so one admission decision governs all three buffers. Declining is a
/// pure performance tradeoff: the transpose is byte-identical whether cached or
/// recomputed, so generated tokens are unchanged.
pub fn set_weight_transpose_cache_enabled(enabled: bool) {
    weight_transpose::set_cache_enabled(enabled);
}

/// Predicted bytes the process-global weight-transpose caches will hold for
/// `graph` once it has run — the *prediction* the memory-strategy plan budgets
/// for, whose accuracy [`weight_transpose_cache_bytes`] measures after the fact.
///
/// # What actually populates the cache
///
/// The cache holds one f32/f16 `N x K` transpose per **constant** (initializer)
/// weight that a kernel feeds through [`weight_transpose::cached_transpose_f32`]
/// / [`weight_transpose::cached_transpose_f16`]. The precise callers, and hence
/// this predictor, are platform-split because the transpose is only consumed by
/// the CPU EP's fast paths:
///
/// * **All platforms — `Gemm` with `transB != 0`** (`gemm.rs`
///   `GemmKernel::execute`, the `transposed_b(&b, n, k)` call added in #1035):
///   a constant `B` stored `[N, K]` is transposed once to the `[K, N]` the
///   shared GEMM consumes and cached as **f32** (`gemm.rs` first widens `B` to
///   dense f32 via `MatMulPrepack::dense`). Cost `N * K * 4` bytes. A
///   non-constant `B`, or `transB == 0`, never caches. When both operands are
///   f16/bf16 the node takes the half path (`try_half_gemm`) and does not
///   transpose; this predictor still counts it (over-predicts, never under —
///   the #1056-mandated safe direction), because whether `A` is half is not a
///   graph-static property here.
///
/// * **Apple only — `MatMul` / `FusedMatMulBias` with a constant `B`**
///   (`matmul.rs` `execute_with_backend` and `fused_matmul_bias.rs`, all guarded
///   by `#[cfg(any(target_os = "macos", target_os = "ios"))]`): the Accelerate
///   GEMV / thin-M paths cache the transpose as **f16** (`N * K * 2`, raw `u16`
///   bit patterns) for a contiguous Float16 `B`, else **f32** (`N * K * 4`).
///   These call sites are compiled out on x86/Windows, so this predictor's
///   Apple arm is likewise `cfg`-gated: a binary predicts exactly what its own
///   kernels will allocate.
///
/// The shape-keyed kernel cache instantiates a node once per activation shape
/// (prefill `m > 1`, decode `m == 1`), but every instance keys the *global*
/// cache on `(weight address, K, N)`, so the second instantiation hits the
/// existing entry and allocates nothing extra. Unlike the MLAS packed buffer
/// (#1051), the transpose is therefore held once per weight, not once per
/// instantiation — no per-copy multiplier applies.
pub fn weight_transpose_cache_predicted_bytes(graph: &Graph) -> u64 {
    let mut total = 0_u64;
    for node in graph.nodes.values() {
        total = total.saturating_add(node_weight_transpose_cache_bytes(node, graph));
    }
    total
}

/// Element count of a node's constant 2-D `B` initializer (input index 1), or
/// `None` when `B` is absent, non-constant, or not rank-2. Shared by both arms
/// of [`weight_transpose_cache_predicted_bytes`].
fn constant_b_numel(node: &Node, graph: &Graph) -> Option<(u64, DataType)> {
    let b_value = (*node.inputs.get(1)?)?;
    let weight = graph.initializers.get(&b_value)?;
    let dims = weight.dims();
    if dims.len() != 2 {
        return None;
    }
    let numel = (dims[0] as u64).checked_mul(dims[1] as u64)?;
    Some((numel, weight.dtype()))
}

/// Per-node contribution to [`weight_transpose_cache_predicted_bytes`]. Mirrors
/// exactly the kernel call sites documented there.
fn node_weight_transpose_cache_bytes(node: &Node, graph: &Graph) -> u64 {
    if !node.is_default_domain() {
        return 0;
    }
    // All platforms: `Gemm` with `transB != 0` transposes a constant `B[N,K]`
    // to `[K,N]` and caches it as f32 (`gemm.rs:119`).
    if node.op_type == "Gemm" {
        let trans_b = node.attr("transB").and_then(Attribute::as_int).unwrap_or(0) != 0;
        if !trans_b {
            return 0;
        }
        let Some((numel, _dtype)) = constant_b_numel(node, graph) else {
            return 0;
        };
        return numel.saturating_mul(4);
    }
    // Apple only: `MatMul` / `FusedMatMulBias` cache a constant `B`'s transpose
    // in the Accelerate paths (see the function docs). Compiled out elsewhere so
    // the predictor matches the kernels actually present in this binary.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if node.op_type == "MatMul" || node.op_type == "FusedMatMulBias" {
        let Some((numel, dtype)) = constant_b_numel(node, graph) else {
            return 0;
        };
        let elem = if dtype == DataType::Float16 { 2 } else { 4 };
        return numel.saturating_mul(elem);
    }
    0
}

/// Evict all entries from the global weight-transpose caches.
///
/// **Must** be called when an Executor drops: the caches are keyed by
/// (address, K, N), which makes a stale entry impossible for a *different*
/// tensor, but a later model whose mmap places a **same-shaped** weight at a
/// recycled address would still match. Clearing on Executor drop closes that
/// remaining lifetime window and bounds cache growth across model lifetimes.
pub fn clear_weight_transpose_caches() {
    weight_transpose::clear_all();
}

// ---------------------------------------------------------------------------
// #1056: governance for the per-kernel `MatMulPrepack::dense` widened-f32 cache.
// ---------------------------------------------------------------------------
//
// `MatMulPrepack::dense` keeps a session-lifetime `4 * K * N` f32 copy of a
// **constant** operand whenever that operand is not already a contiguous f32
// view -- i.e. whenever `to_dense_f32_widen` has to allocate (f16/bf16/f64, or a
// strided f32). A contiguous f32 constant is borrowed zero-copy and nothing is
// retained, which is why this buffer is dormant on the int4 / f32-contiguous
// models we exercise most and was never reported to the plan. It is nonetheless
// a resident, weight-scaled buffer and so must be declinable (#1056), exactly
// like the resident dequant f32 cache (#987), the MLAS SQNBit packed buffer
// (#1051), and the weight-transpose cache (#1079).
//
// The admission verdict is a process-global that production writes once, at
// load, from the memory-strategy plan. Tests must never write it (that is the
// #983 / #1033 / #1079 "passes alone, fails in company" trap); they use the
// thread-local [`DenseCacheEnabledScope`] instead, which leaves the global
// untouched and restores on drop even on panic.

/// Admission verdict for the per-kernel `MatMulPrepack::dense` widened-f32
/// caches. Defaults to enabled so the out-of-box path is unchanged.
static MATMUL_DENSE_CACHE_ENABLED: AtomicBool = AtomicBool::new(true);

thread_local! {
    /// Test-only, per-thread override of the dense-cache admission verdict.
    /// `None` defers to the process-global; `Some(v)` forces `v` on **this
    /// thread only**, so one test's decline cannot leak into another test
    /// running concurrently on a different worker thread. Set exclusively
    /// through [`DenseCacheEnabledScope`]. Production never touches this.
    static MATMUL_DENSE_CACHE_ENABLED_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Admit (`true`) or decline (`false`) the per-kernel `MatMulPrepack::dense`
/// widened-f32 caches for the rest of the session (#1056).
///
/// When declined, `MatMulPrepack::dense` widens the constant operand transiently
/// on each call and frees it, so the resident, session-lifetime `4 * K * N`
/// copies never accrue. The engine calls this once at load from the
/// memory-strategy plan's verdict, beside the identical gates for the resident
/// dequant f32 cache (#987), the MLAS SQNBit packed buffer (#1051), and the
/// weight-transpose cache (#1079), so one admission decision governs all four
/// buffers. Declining is a pure performance tradeoff: the widened f32 is
/// byte-identical whether cached or recomputed, so generated tokens are
/// unchanged.
///
/// This is the **production** entry point. Because it writes a process-global
/// every worker thread reads, tests must not call it; they use
/// [`DenseCacheEnabledScope`].
pub fn set_matmul_dense_cache_enabled(enabled: bool) {
    MATMUL_DENSE_CACHE_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether the `MatMulPrepack::dense` caches are currently admitted, honouring a
/// test-only thread-local override ([`DenseCacheEnabledScope`]) over the global.
pub(crate) fn matmul_dense_cache_enabled() -> bool {
    if let Some(forced) = MATMUL_DENSE_CACHE_ENABLED_OVERRIDE.with(std::cell::Cell::get) {
        return forced;
    }
    MATMUL_DENSE_CACHE_ENABLED.load(Ordering::Relaxed)
}

/// The verdict a freshly-built [`MatMulPrepack`] slot carries, derived from the
/// current admission gate. The predicted byte figure is per-*graph* (see
/// [`matmul_dense_cache_predicted_bytes`]) rather than per-slot, because a
/// prepack does not know `K`/`N` until an operand arrives; the slot therefore
/// carries only the admit/decline decision, and [`GovernedWeightCache::live_bytes`]
/// reports what it actually holds so a prediction can be checked against reality.
fn dense_cache_verdict() -> CacheVerdict {
    if matmul_dense_cache_enabled() {
        CacheVerdict::admit(0)
    } else {
        CacheVerdict::decline(0)
    }
}

/// RAII, thread-local scoping of the dense-cache admission verdict for tests
/// (#1056). Constructing one forces [`matmul_dense_cache_enabled`] to `enabled`
/// on the current thread until dropped, restoring the previous value even on
/// panic. Because the override is thread-local, a test scoping a decline here
/// does not race the other `MatMulPrepack` tests running concurrently.
///
/// The verdict is read at `MatMulPrepack` construction, so a test must build its
/// kernel *inside* the scope for the decline to take effect.
#[cfg(test)]
pub(crate) struct DenseCacheEnabledScope {
    prev: Option<bool>,
}

#[cfg(test)]
impl DenseCacheEnabledScope {
    pub(crate) fn new(enabled: bool) -> Self {
        let prev = MATMUL_DENSE_CACHE_ENABLED_OVERRIDE.with(|cell| cell.replace(Some(enabled)));
        Self { prev }
    }
}

#[cfg(test)]
impl Drop for DenseCacheEnabledScope {
    fn drop(&mut self) {
        MATMUL_DENSE_CACHE_ENABLED_OVERRIDE.with(|cell| cell.set(self.prev));
    }
}

/// How many times an autoregressive decode session materializes each populated
/// `MatMulPrepack::dense` copy -- the shape-keyed instantiation multiplier that
/// the first cut of this accounting (like #1051's before it) missed.
///
/// The executor's kernel cache is keyed by `(node, resolved_input_shapes)`, so a
/// decoder compiles a `MatMul` node once for the prefill shape (`m > 1`) and
/// once for the decode shape (`m == 1`), each a separate `MatMulPrepack` holding
/// its own `dense`. The only operand that reaches `dense` is a constant non-f32
/// `B` paired with an f32 `A`, and in that case both the prefill and the decode
/// instance take the generic/direct-f32 GEMM that widens `B`, so **both** retain
/// a `4 * numel` copy for the whole session. Measured directly by
/// [`predicted_dense_bytes_equal_actual_after_matmul_execution`]: a prefill
/// instance plus a decode instance of one f16-`B` node sum to exactly two
/// copies. Accounting one copy under-reports the resident footprint by 2x -- the
/// dangerous (under-reporting) direction for an admission gate. A prefill-only
/// run holds one copy, so this over-estimates there, which is the safe direction
/// (the gate declines sooner). Matches #1051's `MLAS_PACKED_DECODE_INSTANTIATIONS`.
const MATMUL_DENSE_DECODE_INSTANTIATIONS: u64 = 2;

/// Predicted bytes the per-kernel `MatMulPrepack::dense` caches will hold for
/// `graph` once it has run -- the *prediction* the memory-strategy plan budgets
/// for, whose accuracy the summed [`GovernedWeightCache::live_bytes`] of the
/// executed kernels measures after the fact.
///
/// # What actually populates the cache
///
/// `MatMulPrepack::dense(index, view)` caches iff the operand is **constant**
/// (a graph initializer, tracked by `set_constant_inputs`) **and**
/// `to_dense_f32_widen` returns an owned buffer -- that is, the operand is *not*
/// an already-contiguous f32 view (see `dtype::to_dense_f32_widen`). The cached
/// buffer is a `Vec<f32>` of `numel` elements, so it costs `4 * numel` bytes.
///
/// A graph initializer's stored layout is contiguous (`WeightRef` carries only
/// dtype and dims, no strides), so from the graph the caching condition reduces
/// to: **a constant operand whose dtype is a non-f32 float** (f16 / bf16 / f64).
/// A contiguous f32 constant is borrowed zero-copy and contributes nothing.
///
/// # The per-shape instantiation multiplier
///
/// The executor's kernel cache is **shape-keyed** (`KernelKey { node, shapes }`
/// in `onnx-runtime-session/src/executor/kernel_cache.rs`): a `MatMul` kernel is
/// compiled and cached *per distinct resolved activation shape*, and each
/// instance is a *separate* `MatMulPrepack` with its own `dense` slot. Unlike
/// the weight-transpose cache -- process-global and keyed on the weight address,
/// so a second instantiation reuses the first -- the dense cache is per-instance,
/// so its resident footprint is multiplied by the number of instantiations that
/// widen the constant `B`.
///
/// An autoregressive decoder presents exactly two activation shapes to each such
/// node: the prefill shape (`m > 1`) and the decode shape (`m == 1`). The only
/// case that populates `dense` is a constant non-f32 `B` paired with an f32 `A`
/// (a same-half `B`+`A` takes `try_matmul_half`, and both the `m == 1` GEMV and
/// the MLAS half-prefill fast paths require `A` to be half, so they are skipped
/// when `A` is f32) -- and in that case **both** the prefill instance (via the
/// generic/direct-f32 GEMM) and the decode instance (via the same paths at
/// `m == 1`) widen `B` and each retain their own `4 * numel` copy. So the
/// session holds [`MATMUL_DENSE_DECODE_INSTANTIATIONS`] copies, not one. This is
/// the same shape-keyed multiplier the MLAS SQNBit accounting missed and #1051
/// corrected; counting one copy here would under-report the footprint by 2x --
/// the dangerous (under-reporting) direction for an admission gate.
///
/// # Direction of error
///
/// Per #1056 this predictor **over-predicts, never under-predicts**:
///
/// * When both operands are the *same* half dtype and contiguous, the node
///   takes the packed half path (`try_matmul_half`) and never calls `dense`, so
///   the true cost is zero while this still counts it. Whether the other operand
///   is half is not a graph-static property of the constant one, so counting is
///   the safe direction.
/// * A prefill-only run (no `m == 1` step) instantiates one copy, so accounting
///   [`MATMUL_DENSE_DECODE_INSTANTIATIONS`] over-estimates for that workload,
///   which only makes the gate decline sooner (safe). Conversely, a session
///   that presents *more* than two distinct activation shapes to a node (e.g.
///   several prompt lengths across `generate` calls) could instantiate more than
///   two copies; the multiplier follows #1051's convention of bounding the
///   autoregressive decode workload at prefill + decode, and that residual is
///   the same documented class as #1051's.
/// * The one residual gap that cannot be closed from the graph is a **non-
///   contiguous f32** constant (a column-major weight): the kernel *would* cache
///   it (`to_dense_f32_widen` allocates), but `WeightRef` exposes no strides, so
///   it is indistinguishable from a contiguous f32 constant here and is *not*
///   counted -- a potential under-prediction. No such operand occurs in the
///   models this repo exercises (the documented column-major weight, the lm_head
///   projection, is f16 and so *is* counted via the dtype rule); if f32
///   column-major MatMul weights ever appear, the loader must expose their
///   layout for this predictor to see them. Documented rather than silently
///   over-counting every f32 weight, which would inflate the budget on the
///   overwhelmingly common contiguous-f32 case.
///
/// The predicted-equals-actual test drives *both* a prefill and a decode
/// instantiation of one node and sums their `live_bytes`, so the ratio it
/// asserts is 1.00 against the multiplied prediction -- a test that saw only one
/// instantiation could not defend against a dropped multiplier.
pub fn matmul_dense_cache_predicted_bytes(graph: &Graph) -> u64 {
    let mut total = 0_u64;
    for node in graph.nodes.values() {
        total = total.saturating_add(node_matmul_dense_cache_bytes(node, graph));
    }
    total
}

/// Per-node contribution to [`matmul_dense_cache_predicted_bytes`]. Mirrors the
/// caching condition in [`MatMulPrepack::dense`] exactly.
fn node_matmul_dense_cache_bytes(node: &Node, graph: &Graph) -> u64 {
    // Only the f32 `MatMul` kernel and its bias-fused variant own a
    // `MatMulPrepack`; every other op either has no dense cache or (MatMulNBits)
    // is accounted separately. FusedMatMulBias is included because the optimiser
    // may already have fused a `MatMul + Add` before this predictor runs, and it
    // holds the identical `dense` cache.
    if !node.is_default_domain() || (node.op_type != "MatMul" && node.op_type != "FusedMatMulBias")
    {
        return 0;
    }
    let mut total = 0_u64;
    for index in 0..2 {
        total = total.saturating_add(dense_operand_cache_bytes(node, graph, index));
    }
    // The shape-keyed kernel cache instantiates this node once per activation
    // shape (prefill + decode), each holding its own `dense` copy; see
    // [`MATMUL_DENSE_DECODE_INSTANTIATIONS`].
    total.saturating_mul(MATMUL_DENSE_DECODE_INSTANTIATIONS)
}

/// Bytes `MatMulPrepack::dense(index, ..)` retains for one operand of `node`:
/// `4 * numel` for a constant non-f32-float initializer, else 0. See
/// [`matmul_dense_cache_predicted_bytes`] for the reasoning.
fn dense_operand_cache_bytes(node: &Node, graph: &Graph, index: usize) -> u64 {
    let Some(Some(value)) = node.inputs.get(index) else {
        return 0;
    };
    let Some(weight) = graph.initializers.get(value) else {
        return 0;
    };
    // A contiguous f32 constant is borrowed zero-copy (no cache). Only float
    // dtypes reach `dense` (`to_dense_f32_widen` rejects non-float), and every
    // non-f32 float widens to an owned `4 * numel` f32 copy.
    let widens_to_owned = matches!(
        weight.dtype(),
        DataType::Float16 | DataType::BFloat16 | DataType::Float64
    );
    if !widens_to_owned {
        return 0;
    }
    let numel = weight
        .dims()
        .iter()
        .try_fold(1_u64, |acc, &dim| acc.checked_mul(dim as u64));
    match numel {
        Some(numel) => numel.saturating_mul(4),
        None => u64::MAX,
    }
}

/// Pre-compute and cache the f16 transpose of a weight matrix.
///
/// Called during model load to move the ~1 s first-decode-step transpose cost
/// into the model-load budget (we have ~1.5 s of headroom vs ORT). The global
/// cache ensures subsequent kernel-cache shape misses (prefill M=40 → decode
/// M=1) find the transpose immediately.
///
/// # Safety
/// `data_ptr` must point to a valid, aligned, contiguous array of `k * n`
/// `u16` values that remains live for the duration of this call.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub unsafe fn precompute_f16_weight_transpose(data_ptr: *const u16, k: usize, n: usize) {
    let Some(numel) = k.checked_mul(n) else {
        return;
    };
    if numel == 0 {
        return;
    }
    // SAFETY: delegated to this function's contract — `data_ptr` addresses
    // `k * n` live, aligned `u16` values for the duration of the call.
    let src = unsafe { std::slice::from_raw_parts(data_ptr, numel) };
    let _ = weight_transpose::cached_transpose_f16(src, k, n);
}

/// Pre-compute and cache the f32 transpose of a weight matrix.
///
/// Called during model load for weights eligible for the thin-M GEMM path
/// (K*N > THIN_M_LARGE_B_THRESHOLD). Moves the transpose cost into model load
/// so that TTFT is not penalized on the first inference after Engine creation.
///
/// # Safety
/// `data_ptr` must point to a valid, aligned, contiguous array of `k * n`
/// `f32` values that remains live for the duration of this call.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub unsafe fn precompute_f32_weight_transpose(data_ptr: *const f32, k: usize, n: usize) {
    use accelerate_gemm::THIN_M_LARGE_B_THRESHOLD;
    if (k as u64) * (n as u64) <= THIN_M_LARGE_B_THRESHOLD as u64 {
        return;
    }
    let Some(numel) = k.checked_mul(n) else {
        return;
    };
    if numel == 0 {
        return;
    }
    // SAFETY: delegated to this function's contract — `data_ptr` addresses
    // `k * n` live, aligned `f32` values for the duration of the call.
    let src = unsafe { std::slice::from_raw_parts(data_ptr, numel) };
    let _ = weight_transpose::cached_transpose_f32(src, k, n);
}

/// Per-kernel cache for immutable MatMul operands that require materialization.
///
/// Contiguous f32 constants already have the ideal representation, so they stay
/// zero-copy and need no owned cache entry.
///
/// Weight transpose memos (`transposed_b`, `transposed_b_f16`) use `Arc` so the
/// data is shared with the process-global caches in
/// [`weight_transpose`](crate::kernels::weight_transpose). This ensures that a
/// kernel-cache shape miss (e.g. prefill M=40 → decode M=1) finds the transpose
/// in the global cache rather than recomputing it. Each memo stores the
/// [`WeightTransposeKey`] it was filled for, so it is validated against the
/// operand actually being multiplied on every call (#845).
pub(crate) struct MatMulPrepack {
    constant_inputs: [bool; 2],
    /// Session-lifetime widened-f32 copy of a constant operand, governed by the
    /// memory plan (#1056): filled only under an admitting verdict, and reports
    /// its own held bytes via [`GovernedWeightCache::live_bytes`]. Only populated
    /// when the operand is not already contiguous f32 (see [`Self::dense`]).
    dense: [GovernedWeightCache<f32>; 2],
    #[cfg(feature = "mlas")]
    packed_b: OnceLock<mlas_sys::PackedB>,
    /// MLAS-packed f32 panel built once from a constant **f16** B weight.
    ///
    /// The widened f32 copy is a temporary: it is consumed by `PackedB::new`
    /// and dropped, so this costs one packed panel rather than the packed panel
    /// plus a permanent 4*K*N dense copy the way `dense[1]` + `packed_b` would.
    #[cfg(feature = "mlas")]
    packed_b_from_half: OnceLock<Option<(WeightTransposeKey, mlas_sys::PackedB)>>,
    /// Lazily-computed transpose of the B weight matrix for the Accelerate
    /// column-parallel GEMV path, with the key it was computed for. Only
    /// populated for constant (model weight) inputs.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "ios")),
        allow(dead_code, reason = "consumed only by the Apple Accelerate paths")
    )]
    transposed_b: OnceLock<(WeightTransposeKey, Arc<Vec<f32>>)>,
    /// Lazily-computed f16 transpose of the B weight matrix, with the key it
    /// was computed for. Stores the raw u16 bit patterns of half::f16 in N×K
    /// layout, read directly from the mmap'd model file without widening to
    /// f32. Only populated when B is a constant Float16 input.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "ios")),
        allow(dead_code, reason = "consumed only by the Apple Accelerate paths")
    )]
    transposed_b_f16: OnceLock<(WeightTransposeKey, Arc<Vec<u16>>)>,
    /// Lazily-computed contiguous f16 copy of B for non-contiguous weight
    /// matrices (e.g. lm_head vocab projection stored column-major in the ONNX
    /// model). Stores raw u16 bit patterns in row-major K×N layout. Only
    /// populated for constant Float16 inputs where the original is
    /// non-contiguous.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "ios")),
        allow(dead_code, reason = "consumed only by the Apple Accelerate paths")
    )]
    contiguous_b_f16: OnceLock<Arc<Vec<u16>>>,
}

impl Default for MatMulPrepack {
    /// Build a prepack whose `dense` slots carry the current admission verdict
    /// (#1056). Manual rather than derived because [`GovernedWeightCache`] has no
    /// `Default`: an unaccounted cache must be impossible to construct by
    /// accident, so every slot is stamped with the plan's admit/decline decision
    /// at birth (read here from the process-global gate the engine set at load).
    fn default() -> Self {
        let verdict = dense_cache_verdict();
        Self {
            constant_inputs: [false; 2],
            dense: [
                GovernedWeightCache::new(verdict),
                GovernedWeightCache::new(verdict),
            ],
            #[cfg(feature = "mlas")]
            packed_b: OnceLock::new(),
            #[cfg(feature = "mlas")]
            packed_b_from_half: OnceLock::new(),
            transposed_b: OnceLock::new(),
            transposed_b_f16: OnceLock::new(),
            contiguous_b_f16: OnceLock::new(),
        }
    }
}

impl MatMulPrepack {
    /// Total bytes the `dense` widened-f32 caches currently hold across both
    /// operand slots -- the figure a prediction is checked against (#1056).
    #[cfg(test)]
    pub(crate) fn dense_live_bytes(&self) -> u64 {
        self.dense[0]
            .live_bytes()
            .saturating_add(self.dense[1].live_bytes())
    }

    pub(crate) fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        for (index, is_constant) in self.constant_inputs.iter_mut().enumerate() {
            *is_constant = constant_inputs.get(index).copied().unwrap_or(false);
        }
    }

    /// Widen a MatMul operand to dense f32, reusing a session-lifetime copy for
    /// a constant operand when the memory plan admits it (#1056).
    ///
    /// A non-constant operand, or one already stored as contiguous f32, is
    /// handled exactly as before: `to_dense_f32_widen` borrows contiguous f32
    /// zero-copy and only allocates a transient for the widening cases.
    ///
    /// For a constant operand that must be widened (`Cow::Owned`), the widened
    /// buffer is retained in the governed cache **iff the plan admitted it**.
    /// When declined, [`GovernedWeightCache`] stores nothing, and this returns
    /// the freshly-widened `Cow::Owned` for the caller to use and drop at the end
    /// of the call -- byte-identical output, only recomputed each time.
    pub(crate) fn dense<'a>(
        &'a self,
        index: usize,
        view: &'a TensorView<'_>,
    ) -> Result<Cow<'a, [f32]>> {
        if !self.constant_inputs[index] {
            return to_dense_f32_widen("MatMul", view);
        }
        if let Some(cached) = self.dense[index].filled() {
            return Ok(Cow::Borrowed(cached));
        }

        match to_dense_f32_widen("MatMul", view)? {
            Cow::Borrowed(dense) => Ok(Cow::Borrowed(dense)),
            Cow::Owned(dense) => {
                if self.dense[index].verdict().is_admitted() {
                    // Admitted: retain the widened copy for the session and hand
                    // back a borrow of it. `get_or_fill` builds exactly once.
                    let cached = self.dense[index]
                        .get_or_fill(|| dense)
                        .expect("an admitted governed cache fills its buffer");
                    Ok(Cow::Borrowed(cached))
                } else {
                    // Declined: keep nothing resident; the caller owns and frees
                    // this transient widen at the end of the call.
                    Ok(Cow::Owned(dense))
                }
            }
        }
    }

    /// Returns a cached transpose of B[K,N] -> B_T[N,K] row-major.
    ///
    /// Only transposes constant (model weight) inputs. Uses a process-global
    /// cache so the transpose survives kernel-cache shape evictions (e.g.
    /// prefill M=40 → decode M=1).
    ///
    /// Returns `None` — never a buffer of the wrong length — when B is an
    /// activation, when `b.len()` is not exactly `k * n`, or when this prepack's
    /// memo was filled for a different operand or geometry. The callers index
    /// the returned slice unchecked, so every `Some` is exactly `n * k` long
    /// (#845); a `None` falls back to the untransposed kernel.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "ios")),
        allow(dead_code, reason = "consumed only by the Apple Accelerate paths")
    )]
    pub(crate) fn transposed_b(&self, b: &[f32], k: usize, n: usize) -> Option<&[f32]> {
        if !self.constant_inputs[1] {
            return None;
        }
        // #1056: when the plan declines the transpose cache, hand back `None` so
        // the caller recomputes a transient transpose per call (freed at the end
        // of the call) rather than memoizing an `Arc` in this kernel instance —
        // that per-instance memo would survive the session and, multiplied by
        // the shape-keyed kernel-cache instantiations, is exactly the resident
        // footprint the decline is meant to shed.
        if !weight_transpose::cache_enabled() {
            return None;
        }
        let key = WeightTransposeKey::new(b.as_ptr(), k, n);
        // Validate before touching the memo so a mismatched call can neither
        // install a wrong-length entry nor poison a correct one.
        if key.numel() != Some(b.len()) {
            return None;
        }
        let (cached_key, bt) = self.transposed_b.get_or_init(|| {
            let bt = weight_transpose::cached_transpose_f32(b, k, n)
                .expect("length was validated against [k, n] above");
            (key, bt)
        });
        (*cached_key == key).then(|| bt.as_slice())
    }

    /// Returns a cached f16 transpose of B[K,N] → B_T[N,K] row-major.
    ///
    /// Like [`transposed_b`](Self::transposed_b) but preserves the original f16
    /// storage format (as raw u16 bit patterns), reading directly from the
    /// mmap'd model buffer. Uses a process-global cache so the transpose
    /// survives kernel-cache shape evictions. The same total-identity rules as
    /// [`transposed_b`](Self::transposed_b) apply: every `Some` is exactly
    /// `n * k` elements long.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "ios")),
        allow(dead_code, reason = "consumed only by the Apple Accelerate paths")
    )]
    pub(crate) fn transposed_b_f16(
        &self,
        b_view: &TensorView,
        k: usize,
        n: usize,
    ) -> Option<&[u16]> {
        use onnx_runtime_ir::DataType;
        if !self.constant_inputs[1] || b_view.dtype != DataType::Float16 || !b_view.is_contiguous()
        {
            return None;
        }
        // #1056: declined cache -> recompute per call, retain nothing.
        if !weight_transpose::cache_enabled() {
            return None;
        }
        // The caller derives `k` from A and `n` from B, so operands that
        // disagree would otherwise build a slice longer than the weight and
        // transpose out of bounds.
        let numel = k.checked_mul(n)?;
        if b_view.numel() != numel {
            return None;
        }
        // SAFETY: `b_view` is a contiguous Float16 view whose element count was
        // just checked to equal `k * n`, and it stays live for this call.
        let src = unsafe { std::slice::from_raw_parts(b_view.data_ptr::<u16>(), numel) };
        let key = WeightTransposeKey::new(src.as_ptr(), k, n);
        let (cached_key, bt) = self.transposed_b_f16.get_or_init(|| {
            let bt = weight_transpose::cached_transpose_f16(src, k, n)
                .expect("length was validated against [k, n] above");
            (key, bt)
        });
        (*cached_key == key).then(|| bt.as_slice())
    }

    /// Returns a cached contiguous f16 copy of a non-contiguous B weight.
    ///
    /// When a model stores a weight matrix with non-row-major strides (e.g.
    /// the lm_head vocab projection stored column-major), `try_matmul_half`
    /// skips it because BNNS requires contiguous input.  This method
    /// materialises the contiguous K×N row-major copy once and caches it
    /// for the session lifetime, so subsequent prefill calls avoid both:
    ///   - the element-by-element `to_dense_f32_widen` (~1 s at 136 M elements)
    ///   - the 2× memory of f16→f32 widening
    ///
    /// The memo is validated against the current view's element count before it
    /// is served: the consumers read it as a dense `K×N` buffer, so a memo
    /// filled for a differently-sized operand must not be handed out (#845).
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "ios")),
        allow(dead_code, reason = "consumed only by the Apple Accelerate paths")
    )]
    pub(crate) fn contiguous_b_f16(&self, b_view: &TensorView) -> Option<&[u16]> {
        use onnx_runtime_ir::DataType;
        if !self.constant_inputs[1] || b_view.dtype != DataType::Float16 || b_view.is_contiguous() {
            return None; // already contiguous or not cacheable
        }
        let cached = self.contiguous_b_f16.get_or_init(|| {
            let numel = b_view.numel();
            let shape = b_view.shape;
            let strides = b_view.strides;
            let base = b_view.data_ptr::<u16>();
            let mut out = vec![0u16; numel];

            // Optimised 2-D path (covers all weight matrices).
            if shape.len() == 2 {
                let (rows, cols) = (shape[0], shape[1]);
                let (sr, sc) = (strides[0] as isize, strides[1] as isize);
                use rayon::prelude::*;
                let threads = rayon::current_num_threads();
                let rows_per_thread = rows.div_ceil(threads).max(1);
                // SAFETY: `base` points into the model's mmap'd weight
                // buffer which is immutable for the session lifetime.
                // Each Rayon task reads a disjoint set of source rows
                // and writes a disjoint output chunk.
                let base_usize = base as usize;
                out.par_chunks_mut(rows_per_thread * cols)
                    .enumerate()
                    .for_each(|(t, chunk)| {
                        let base = base_usize as *const u16;
                        let i0 = t * rows_per_thread;
                        let i_end = (i0 + rows_per_thread).min(rows);
                        for i in i0..i_end {
                            let row_base = unsafe { base.offset(i as isize * sr) };
                            let dst = &mut chunk[(i - i0) * cols..(i - i0 + 1) * cols];
                            for (j, d) in dst.iter_mut().enumerate() {
                                *d = unsafe { *row_base.offset(j as isize * sc) };
                            }
                        }
                    });
            } else {
                // General N-D fallback (rare).
                let mut idx = vec![0usize; shape.len()];
                for o in out.iter_mut() {
                    let off: isize = idx
                        .iter()
                        .zip(strides.iter())
                        .map(|(&i, &s)| i as isize * s as isize)
                        .sum();
                    *o = unsafe { *base.offset(off) };
                    for d in (0..shape.len()).rev() {
                        idx[d] += 1;
                        if idx[d] < shape[d] {
                            break;
                        }
                        idx[d] = 0;
                    }
                }
            }
            Arc::new(out)
        });
        (cached.len() == b_view.numel()).then(|| cached.as_slice())
    }

    /// MLAS-packed B built by widening a constant f16 weight to f32 once.
    ///
    /// Returns `None` for a non-constant B: an activation would have to be
    /// re-widened and re-packed on every call, which is the cost this exists to
    /// remove, and caching it would be wrong the moment the activation changed.
    #[cfg(feature = "mlas")]
    fn packed_b_from_half(
        &self,
        view: &TensorView<'_>,
        k: usize,
        n: usize,
    ) -> Result<Option<&mlas_sys::PackedB>> {
        if !self.constant_inputs[1] {
            return Ok(None);
        }
        // Defence in depth, mirroring `transposed_b`: a memo is only served
        // back for the exact weight it was built from. Today the executor's
        // kernel cache keys on node and input shapes and a constant input is a
        // graph initializer whose address is stable, so the key can only ever
        // match; the check exists so that broadening "constant" later fails
        // loudly instead of silently returning another tensor's pack.
        let key = WeightTransposeKey::new(view.data_ptr::<u16>(), k, n);
        if let Some(cached) = self.packed_b_from_half.get() {
            return Ok(cached
                .as_ref()
                .and_then(|(cached_key, packed)| (*cached_key == key).then_some(packed)));
        }
        let widened = to_dense_f32_widen("MatMul", view)?;
        if key.numel() != Some(widened.len()) {
            return Ok(None);
        }
        let packed = mlas_sys::PackedB::new(n, k, &widened);
        drop(widened);
        Ok(self
            .packed_b_from_half
            .get_or_init(|| Some((key, packed)))
            .as_ref()
            .and_then(|(cached_key, packed)| (*cached_key == key).then_some(packed)))
    }

    /// Whether the once-widened MLAS pack for a constant f16 B has been built.
    ///
    /// Lets sibling modules' tests assert which route served a call; the field
    /// itself stays private to this module.
    #[cfg(test)]
    pub(crate) fn half_pack_is_built(&self) -> bool {
        #[cfg(feature = "mlas")]
        {
            self.packed_b_from_half.get().is_some_and(Option::is_some)
        }
        #[cfg(not(feature = "mlas"))]
        false
    }

    #[cfg(feature = "mlas")]
    fn packed_b(&self, b: &[f32], k: usize, n: usize) -> Option<&mlas_sys::PackedB> {
        self.constant_inputs[1].then(|| {
            self.packed_b
                .get_or_init(|| mlas_sys::PackedB::new(n, k, b))
        })
    }
}

/// f32 MatMul kernel with initializer-only operand prepacking.
#[derive(Default)]
pub struct MatMulKernel {
    prepack: MatMulPrepack,
    /// Structural FLOPs (`2*batch*M*N*K`) when both operand shapes were static
    /// at build time; `None` otherwise (issue #995 — never fabricated).
    flops: Option<u64>,
}

/// Factory for [`MatMulKernel`] (no attributes).
pub struct MatMulFactory;

impl KernelFactory for MatMulFactory {
    fn create(&self, _node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let flops = match (input_shapes.first(), input_shapes.get(1)) {
            (Some(a), Some(b)) => super::flops::matmul_flops(a, b),
            _ => None,
        };
        Ok(Box::new(MatMulKernel {
            flops,
            ..MatMulKernel::default()
        }))
    }
}

/// 2-D tile GEMM dispatch: `c[m,n] = sum_k a[m,k] * b[k,n]` (overwrite).
///
/// `a` is `m*k` row-major, `b` is `k*n` row-major, `c` is `m*n` row-major.
/// Picks the backend via [`CpuBackend::auto_detect`] (`docs/architecture/ORT2.md` §25.2):
/// `SimdX86` when supported by the host, otherwise the pure-Rust blocked GEMM.
/// The result is bit-plausible across backends within f32 tolerance.
pub(crate) fn gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    gemm_with_backend(CpuBackend::auto_detect(), a, b, c, m, k, n)
}

#[cfg(feature = "mlas")]
fn gemm_packed(
    a: &[f32],
    packed: &mlas_sys::PackedB,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    assert_eq!(packed.dimensions(), (k, n));
    mlas_sys::sgemm_nn_packed(m, a, packed, c);
    Ok(())
}

/// f16 prefill through MLAS SGEMM on a once-widened, once-packed B.
///
/// The portable blocked half GEMM re-packs B into cache-sized panels on *every*
/// call and drives them from a 4x8 microkernel on the global rayon pool. For a
/// constant weight all of that is avoidable: ORT widens f16 to f32 and runs the
/// same tuned SGEMM it uses for float, and the difference is not small.
/// `128x3584x3584` MatMul measured 2.44x slower than ORT at one thread and
/// 8.24x at eight, because our scaling was 2.2x where ORT's was 7.5x.
///
/// Widening B is paid once and the f32 copy is dropped immediately, so the
/// steady-state footprint is one MLAS panel rather than a panel plus a
/// permanent dense copy. Only A is widened per call: `M*K` against B's `K*N`.
///
/// Returns `None` — leaving the caller's existing path untouched — for a
/// non-MLAS backend, a non-f16 operand, a non-contiguous or wrongly-sized B, an
/// activation B, or `M <= 1`. `M = 1` is excluded because packing needs row
/// reuse to pay for itself and the decode GEMV is already at parity with ORT.
#[cfg(feature = "mlas")]
pub(crate) fn try_packed_half_prefill(
    prepack: &MatMulPrepack,
    backend: CpuBackend,
    a: &TensorView,
    b: &TensorView,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Option<Vec<f32>>> {
    if backend != CpuBackend::Mlas
        || m <= 1
        || a.dtype != onnx_runtime_ir::DataType::Float16
        || b.dtype != onnx_runtime_ir::DataType::Float16
        || !b.is_contiguous()
        || b.numel() != k.saturating_mul(n)
    {
        return Ok(None);
    }
    b.validate()?;
    let Some(packed) = prepack.packed_b_from_half(b, k, n)? else {
        return Ok(None);
    };
    let a_dense = prepack.dense(0, a)?;
    if a_dense.len() != m.saturating_mul(k) {
        return Ok(None);
    }
    let mut result = vec![0.0f32; m * n];
    gemm_packed(&a_dense, packed, &mut result, m, k, n)?;
    Ok(Some(result))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_with_backend(
    backend: CpuBackend,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    match backend {
        #[cfg(feature = "mlas")]
        CpuBackend::Mlas => {
            mlas_sys::sgemm_nn(m, n, k, a, b, c);
            Ok(())
        }
        // Built-in MLAS-style packed SIMD backend for AVX2/FMA x86-64 hosts.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        CpuBackend::SimdX86 => {
            x86_sgemm::sgemm_simd(a, b, c, m, k, n);
            Ok(())
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        CpuBackend::Accelerate => {
            if m == 1 {
                accelerate_gemm::neon_gemv_parallel(a, b, c, k, n);
            } else {
                accelerate_gemm::sgemm(a, b, c, m, k, n);
            }
            Ok(())
        }
        _ => {
            gemm_generic(a, b, c, m, k, n);
            Ok(())
        }
    }
}

// Register microkernel tile: MR rows x NR cols of C accumulated in registers.
const MR: usize = 4;
const NR: usize = 4;
// Cache block over the K dimension so a panel of B stays resident in L1/L2
// while a strip of C accumulates across it.
const KC: usize = 256;
const MAX_MC: usize = 64;

/// Pure-Rust blocked, register-tiled, rayon-parallelized f32 GEMM (the Generic
/// backend). Overwrites `c` with `a @ b`.
///
/// Strategy: the outer M dimension is split into pool-sized row blocks
/// distributed across Rayon. Each task blocks over K in `KC`-wide panels and
/// walks N in `NR`-wide strips, accumulating an `MR x NR` register tile so the
/// hot inner loop over the N strip autovectorizes. Contains no `unsafe`.
fn gemm_generic(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || n == 0 {
        return;
    }
    let threads = rayon::current_num_threads();
    if threads > 1 && m < threads && n > 1 {
        gemm_generic_col_parallel(a, b, c, m, k, n, threads);
        return;
    }
    let mc = if threads <= 1 {
        MAX_MC.min(m)
    } else {
        let target_tasks = threads.saturating_mul(2);
        let rows = m.div_ceil(target_tasks).clamp(1, MAX_MC);
        if rows == 1 {
            1
        } else {
            rows.div_ceil(MR).saturating_mul(MR).min(MAX_MC)
        }
    };
    // Parallelize over row blocks of C; each block owns a disjoint slice of `c`
    // and reads shared, immutable `a`/`b`, so there is no aliasing. Size the
    // blocks from the Rayon pool: prefill commonly has fewer rows than cores,
    // while large matrices retain MR-row reuse and bounded task counts.
    // `for_each_init` rather than `for_each` so a worker-lane span covers this
    // worker's whole row block instead of being reopened per block.
    c.par_chunks_mut(mc * n).enumerate().for_each_init(
        || crate::trace::worker_span("MatMul.row_block"),
        |_span, (blk, c_block)| {
            let i0 = blk * mc;
            let rows = c_block.len() / n; // last block may be short
            let a_block = &a[i0 * k..i0 * k + rows * k];
            gemm_block(a_block, b, c_block, rows, k, n);
        },
    );
}

fn gemm_generic_col_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) {
    use rayon::prelude::*;
    let cols_per_strip = n.div_ceil(threads).max(1);
    let strips: Vec<(usize, usize)> = (0..n)
        .step_by(cols_per_strip)
        .map(|j0| (j0, cols_per_strip.min(n - j0)))
        .collect();
    let results: Vec<(usize, usize, Vec<f32>)> = strips
        .into_par_iter()
        .map(|(j0, strip_n)| {
            let mut b_strip = vec![0.0f32; k * strip_n];
            for row in 0..k {
                b_strip[row * strip_n..row * strip_n + strip_n]
                    .copy_from_slice(&b[row * n + j0..row * n + j0 + strip_n]);
            }
            let mut c_strip = vec![0.0f32; m * strip_n];
            gemm_block(a, &b_strip, &mut c_strip, m, k, strip_n);
            (j0, strip_n, c_strip)
        })
        .collect();
    for (j0, strip_n, c_strip) in results {
        for row in 0..m {
            c[row * n + j0..row * n + j0 + strip_n]
                .copy_from_slice(&c_strip[row * strip_n..row * strip_n + strip_n]);
        }
    }
}

/// Compute `c_block[rows,n] = a_block[rows,k] @ b[k,n]` (overwrite) for one row
/// block, blocking over K and register-tiling MR x NR.
fn gemm_block(a: &[f32], b: &[f32], c: &mut [f32], rows: usize, k: usize, n: usize) {
    for v in c.iter_mut() {
        *v = 0.0;
    }
    let mut kk = 0;
    while kk < k {
        let kc = KC.min(k - kk);
        let mut i = 0;
        while i < rows {
            let mr = MR.min(rows - i);
            let mut j = 0;
            while j < n {
                let nr = NR.min(n - j);
                micro_kernel(a, b, c, k, n, i, j, kk, kc, mr, nr);
                j += NR;
            }
            i += MR;
        }
        kk += KC;
    }
}

/// Accumulate an `mr x nr` (≤ `MR x NR`) tile of C over the K-panel
/// `[kk, kk+kc)`, adding into the existing `c` contents.
#[inline]
#[allow(clippy::too_many_arguments)]
fn micro_kernel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    k: usize,
    n: usize,
    i: usize,
    j: usize,
    kk: usize,
    kc: usize,
    mr: usize,
    nr: usize,
) {
    let mut acc = [[0.0f32; NR]; MR];
    for p in kk..kk + kc {
        let brow = &b[p * n + j..p * n + j + nr];
        for (ii, acc_row) in acc.iter_mut().enumerate().take(mr) {
            let aik = a[(i + ii) * k + p];
            for (jj, acc_v) in acc_row.iter_mut().enumerate().take(nr) {
                *acc_v += aik * brow[jj];
            }
        }
    }
    for (ii, acc_row) in acc.iter().enumerate().take(mr) {
        let c_row = &mut c[(i + ii) * n + j..(i + ii) * n + j + nr];
        for (jj, cv) in c_row.iter_mut().enumerate().take(nr) {
            *cv += acc_row[jj];
        }
    }
}

/// Layout-aware 2-D GEMM: `c[m,n] = a[m,k] * bt[n,k]ᵀ` (overwrite), i.e.
/// `c[i][j] = Σ_p a[i*k + p] * bt[j*k + p]`.
///
/// `a` is `m*k` row-major, **`bt` is `n*k` row-major** — B stored transposed as
/// `[n][k]` — and `c` is `m*n` row-major. This is exactly the layout MoE expert
/// weights already have (`[out_features][in_features]`), so the caller never has
/// to materialize the `[k][n]` copy that plain [`gemm`] would need. It is also a
/// *better* layout for the dot-product formulation: both operands are contiguous
/// along `k`, so the inner loop is a pure contiguous dot product.
///
/// Parallelized identically to [`gemm_generic`]: row blocks of `c` across the
/// process Rayon pool, or column strips when `m` is smaller than the pool. The
/// per-block inner kernel dispatches to an AVX2+FMA path at runtime (with a
/// portable scalar fallback), mirroring the softmax kernel's structure. Result
/// is bit-plausible against a scalar reference within f32 tolerance.
#[cfg(not(feature = "mlas"))]
pub(crate) fn gemm_bt(
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    let threads = rayon::current_num_threads();
    if threads > 1 && m < threads && n > 1 {
        gemm_bt_col_parallel(a, bt, c, m, k, n, threads);
        return Ok(());
    }
    let mc = if threads <= 1 {
        MAX_MC.min(m)
    } else {
        let target_tasks = threads.saturating_mul(2);
        let rows = m.div_ceil(target_tasks).clamp(1, MAX_MC);
        if rows == 1 {
            1
        } else {
            rows.div_ceil(MR).saturating_mul(MR).min(MAX_MC)
        }
    };
    // Parallelize over disjoint row blocks of `c`; each block reads shared,
    // immutable `a`/`bt`. `for_each_init` opens one worker-lane span per worker
    // rather than reopening it per block, matching [`gemm_generic`].
    c.par_chunks_mut(mc * n).enumerate().for_each_init(
        || crate::trace::worker_span("MatMul.gemm_bt.row_block"),
        |_span, (blk, c_block)| {
            let i0 = blk * mc;
            let rows = c_block.len() / n; // last block may be short
            let a_block = &a[i0 * k..i0 * k + rows * k];
            gemm_bt_block(a_block, bt, c_block, rows, k, n);
        },
    );
    Ok(())
}

/// Column-strip parallel counterpart of [`gemm_bt`], selected when `m` is below
/// the pool size so row-block parallelism would under-fill it. Unlike
/// [`gemm_generic_col_parallel`], no strided B copy is needed: a strip of output
/// columns `[j0, j0+strip)` is exactly the contiguous `bt` rows
/// `bt[j0*k .. (j0+strip)*k]`, so each task borrows its slice directly and only
/// the small result scatter is serial.
#[cfg(not(feature = "mlas"))]
fn gemm_bt_col_parallel(
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) {
    let cols_per_strip = n.div_ceil(threads).max(1);
    let strips: Vec<(usize, usize)> = (0..n)
        .step_by(cols_per_strip)
        .map(|j0| (j0, cols_per_strip.min(n - j0)))
        .collect();
    let results: Vec<(usize, usize, Vec<f32>)> = strips
        .into_par_iter()
        .map(|(j0, strip_n)| {
            let bt_strip = &bt[j0 * k..(j0 + strip_n) * k];
            let mut c_strip = vec![0.0f32; m * strip_n];
            gemm_bt_block(a, bt_strip, &mut c_strip, m, k, strip_n);
            (j0, strip_n, c_strip)
        })
        .collect();
    for (j0, strip_n, c_strip) in results {
        for row in 0..m {
            c[row * n + j0..row * n + j0 + strip_n]
                .copy_from_slice(&c_strip[row * strip_n..row * strip_n + strip_n]);
        }
    }
}

/// Compute `c[rows,n] = a[rows,k] * bt[n,k]ᵀ` (overwrite) for one row block,
/// dispatching to the AVX2+FMA kernel when the host supports it.
#[cfg(not(feature = "mlas"))]
fn gemm_bt_block(a: &[f32], bt: &[f32], c: &mut [f32], rows: usize, k: usize, n: usize) {
    if rows == 0 || n == 0 {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if gemm_bt_avx2::available() {
            // SAFETY: the branch proves the running CPU has AVX2+FMA. `a` is
            // `rows*k`, `bt` is `n*k`, `c` is `rows*n`, all row-major, so every
            // pointer the kernel forms from `(rows, k, n)` is in bounds.
            unsafe {
                gemm_bt_avx2::block(a.as_ptr(), bt.as_ptr(), c.as_mut_ptr(), rows, k, n);
            }
            return;
        }
    }
    gemm_bt_block_scalar(a, bt, c, rows, k, n);
}

/// Portable scalar reference block: the fallback for non-x86 targets and for x86
/// without AVX2+FMA. Contains no `unsafe`. Overwrites each `c[i][j]` with the
/// contiguous dot product `Σ_p a[i][p] * bt[j][p]` (so `k == 0` writes 0).
#[cfg(not(feature = "mlas"))]
fn gemm_bt_block_scalar(a: &[f32], bt: &[f32], c: &mut [f32], rows: usize, k: usize, n: usize) {
    for i in 0..rows {
        let arow = &a[i * k..i * k + k];
        let crow = &mut c[i * n..i * n + n];
        for (j, cv) in crow.iter_mut().enumerate() {
            let brow = &bt[j * k..j * k + k];
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += arow[p] * brow[p];
            }
            *cv = sum;
        }
    }
}

/// AVX2+FMA inner kernel for [`gemm_bt`]: `c[i][j] = Σ_p a[i][p] * bt[j][p]` on
/// eight `k`-lanes at once. Runtime-detected exactly like the softmax kernel,
/// with `loadu`/`storeu`-style unaligned access (no alignment assumed) and a
/// scalar tail for `k % 8`.
///
/// Interior `4×2` tiles of `c` are held in eight `__m256` accumulators (four
/// A rows × two B rows), reusing each loaded B vector across the four rows and
/// each loaded A vector across the two columns; the eight independent
/// accumulators hide FMA latency. Edge rows (`rows % 4`, which includes the
/// `m == 1` decode case) and the odd trailing column (`n % 2`) fall to a
/// four-accumulator contiguous dot product — itself a good GEMV.
#[cfg(all(
    not(feature = "mlas"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod gemm_bt_avx2 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Runtime detection: AVX2 (256-bit float ops) and FMA (the accumulation).
    #[inline]
    pub fn available() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    /// Horizontal sum of the eight lanes.
    ///
    /// # Safety
    /// AVX must be available (implied by AVX2 on the calling path).
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn hsum(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
        _mm_cvtss_f32(s)
    }

    /// Contiguous dot product `Σ_{p<k} a[p] * b[p]` with four independent
    /// accumulators (32-wide unroll) to hide FMA latency, then a scalar tail.
    ///
    /// # Safety
    /// AVX2+FMA available; `a` and `b` are each valid for `k` f32 reads.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn dot(a: *const f32, b: *const f32, k: usize) -> f32 {
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut p = 0usize;
            while p + 32 <= k {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(p)), _mm256_loadu_ps(b.add(p)), acc0);
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.add(p + 8)),
                    _mm256_loadu_ps(b.add(p + 8)),
                    acc1,
                );
                acc2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.add(p + 16)),
                    _mm256_loadu_ps(b.add(p + 16)),
                    acc2,
                );
                acc3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.add(p + 24)),
                    _mm256_loadu_ps(b.add(p + 24)),
                    acc3,
                );
                p += 32;
            }
            while p + 8 <= k {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(p)), _mm256_loadu_ps(b.add(p)), acc0);
                p += 8;
            }
            let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            let mut sum = hsum(acc);
            while p < k {
                sum += *a.add(p) * *b.add(p);
                p += 1;
            }
            sum
        }
    }

    /// One interior `4×2` tile: `c[i+ii][j+jj] = Σ_p a[i+ii][p] * bt[j+jj][p]`
    /// for `ii<4`, `jj<2`, held in eight `__m256` accumulators over the full
    /// `k`-loop and finished with a scalar tail (`k % 8`). Each B vector is
    /// reused across the four A rows and each A vector across the two B rows, so
    /// the tile issues eight independent FMAs per eight-lane step — enough
    /// in-flight accumulators to hide FMA latency — and pays exactly one
    /// horizontal reduction per output cell. Because the reduction is along `k`
    /// (the SIMD lanes hold `k`-partials, not `C`-partials), `k` is *not* cache
    /// blocked here; the reuse of a large `bt` is instead recovered by column
    /// blocking in [`block`], which keeps one reduction per tile.
    ///
    /// # Safety
    /// AVX2+FMA available. Rows `i..i+4` of `a` (`a` is `_*k`), rows `j..j+2` of
    /// `bt` (`bt` is `_*k`) and elements `(i+ii)*n + j+jj` of `c` are all valid.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn tile_4x2(
        a: *const f32,
        bt: *const f32,
        c: *mut f32,
        k: usize,
        n: usize,
        i: usize,
        j: usize,
    ) {
        unsafe {
            let a0 = a.add(i * k);
            let a1 = a.add((i + 1) * k);
            let a2 = a.add((i + 2) * k);
            let a3 = a.add((i + 3) * k);
            let b0 = bt.add(j * k);
            let b1 = bt.add((j + 1) * k);

            let mut c00 = _mm256_setzero_ps();
            let mut c01 = _mm256_setzero_ps();
            let mut c10 = _mm256_setzero_ps();
            let mut c11 = _mm256_setzero_ps();
            let mut c20 = _mm256_setzero_ps();
            let mut c21 = _mm256_setzero_ps();
            let mut c30 = _mm256_setzero_ps();
            let mut c31 = _mm256_setzero_ps();

            let mut p = 0usize;
            while p + 8 <= k {
                let vb0 = _mm256_loadu_ps(b0.add(p));
                let vb1 = _mm256_loadu_ps(b1.add(p));
                let va = _mm256_loadu_ps(a0.add(p));
                c00 = _mm256_fmadd_ps(va, vb0, c00);
                c01 = _mm256_fmadd_ps(va, vb1, c01);
                let va = _mm256_loadu_ps(a1.add(p));
                c10 = _mm256_fmadd_ps(va, vb0, c10);
                c11 = _mm256_fmadd_ps(va, vb1, c11);
                let va = _mm256_loadu_ps(a2.add(p));
                c20 = _mm256_fmadd_ps(va, vb0, c20);
                c21 = _mm256_fmadd_ps(va, vb1, c21);
                let va = _mm256_loadu_ps(a3.add(p));
                c30 = _mm256_fmadd_ps(va, vb0, c30);
                c31 = _mm256_fmadd_ps(va, vb1, c31);
                p += 8;
            }

            let mut s00 = hsum(c00);
            let mut s01 = hsum(c01);
            let mut s10 = hsum(c10);
            let mut s11 = hsum(c11);
            let mut s20 = hsum(c20);
            let mut s21 = hsum(c21);
            let mut s30 = hsum(c30);
            let mut s31 = hsum(c31);
            while p < k {
                let x0 = *b0.add(p);
                let x1 = *b1.add(p);
                let y0 = *a0.add(p);
                s00 += y0 * x0;
                s01 += y0 * x1;
                let y1 = *a1.add(p);
                s10 += y1 * x0;
                s11 += y1 * x1;
                let y2 = *a2.add(p);
                s20 += y2 * x0;
                s21 += y2 * x1;
                let y3 = *a3.add(p);
                s30 += y3 * x0;
                s31 += y3 * x1;
                p += 1;
            }

            *c.add(i * n + j) = s00;
            *c.add(i * n + j + 1) = s01;
            *c.add((i + 1) * n + j) = s10;
            *c.add((i + 1) * n + j + 1) = s11;
            *c.add((i + 2) * n + j) = s20;
            *c.add((i + 2) * n + j + 1) = s21;
            *c.add((i + 3) * n + j) = s30;
            *c.add((i + 3) * n + j + 1) = s31;
        }
    }

    /// Compute `c[rows,n] = a[rows,k] * bt[n,k]ᵀ` (overwrite).
    ///
    /// Interior `4×2` tiles are swept **column-block first**: a panel of `NC_KB`
    /// KiB worth of `bt` columns (`nc` rows of the `[n][k]` matrix) is pinned in
    /// cache while all row bands reuse it, then the panel advances. Without this,
    /// each row band re-streams the whole `[n × k]` `bt` from DRAM, so for a
    /// `bt` larger than L3 (e.g. the 52 MiB Phi-3.5-MoE experts) large-token
    /// prefill becomes DRAM-bound and loses to a one-shot transpose+GEMM. The
    /// tile still reduces along `k` in one pass (one horizontal sum per cell),
    /// so — unlike K blocking — column blocking adds no reduction overhead. The
    /// odd trailing column (`n % 2`) and edge rows (`rows % 4`, incl. the
    /// `rows == 1` decode case) then run as full-`k` [`dot`]s.
    ///
    /// # Safety
    /// AVX2+FMA available. `a` is valid for `rows*k` reads, `bt` for `n*k`
    /// reads, and `c` for `rows*n` writes, all row-major.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn block(
        a: *const f32,
        bt: *const f32,
        c: *mut f32,
        rows: usize,
        k: usize,
        n: usize,
    ) {
        unsafe {
            let r4 = rows - rows % 4;
            let n2 = n - n % 2;

            // Column-block width: keep one `bt` panel near L2-resident so the
            // row-band sweep reuses it instead of re-reading all of `bt` per
            // band. Round down to an even count (the tile is 2 columns wide) and
            // keep at least one tile.
            const NC_KB: usize = 256;
            let bytes_per_col = k.saturating_mul(4).max(1);
            let mut nc = (NC_KB * 1024) / bytes_per_col;
            nc &= !1;
            if nc < 2 {
                nc = 2;
            }

            // Interior tiles: column panel outer, row band inner, so the pinned
            // `bt` panel is reused across every row band before it is evicted.
            let mut j0 = 0usize;
            while j0 < n2 {
                let jend = (j0 + nc).min(n2);
                let mut i = 0usize;
                while i < r4 {
                    let mut j = j0;
                    while j < jend {
                        tile_4x2(a, bt, c, k, n, i, j);
                        j += 2;
                    }
                    i += 4;
                }
                j0 = jend;
            }

            // Odd trailing column of the interior rows (single full-k dot each).
            if n2 < n {
                let mut i = 0usize;
                while i < r4 {
                    for ii in 0..4 {
                        *c.add((i + ii) * n + n2) = dot(a.add((i + ii) * k), bt.add(n2 * k), k);
                    }
                    i += 4;
                }
            }

            // Edge rows (`rows % 4`, incl. the `rows == 1` decode case): a
            // full-k dot for every column.
            let mut i = r4;
            while i < rows {
                for j in 0..n {
                    *c.add(i * n + j) = dot(a.add(i * k), bt.add(j * k), k);
                }
                i += 1;
            }
        }
    }
}

impl Kernel for MatMulKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.prepack.set_constant_inputs(constant_inputs);
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.execute_with_backend(inputs, outputs, CpuBackend::auto_detect())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn estimated_flops(&self) -> Option<u64> {
        self.flops
    }
}

impl MatMulKernel {
    fn execute_with_backend(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        backend: CpuBackend,
    ) -> Result<()> {
        check_arity("MatMul", inputs, outputs, 2, 2, 1)?;
        let geom = matmul_geometry(&inputs[0], &inputs[1])?;
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            (numel(&geom.batch_shape) as u64)
                .saturating_mul(geom.m as u64)
                .saturating_mul(geom.n as u64)
                .saturating_mul(geom.k as u64)
                .saturating_mul(2)
        });

        // FP16 storage GEMV: when B is a constant Float16 weight and M=1
        // (decode), GEMV directly from the f16 mmap'd data — reading 2 bytes
        // per weight instead of 4, halving memory bandwidth. This MUST be
        // checked before try_matmul_half: the blocked GEMM is ~4× slower than
        // the bandwidth-optimal NEON GEMV at M=1 decode shapes.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if backend == CpuBackend::Accelerate
            && geom.m == 1
            && numel(&geom.batch_shape) <= 1
            && geom.b_promoted_rank == 2
            && inputs[1].dtype == onnx_runtime_ir::DataType::Float16
        {
            // Try the transposed-B cache first (contiguous weights).
            if let Some(bt_f16) = self.prepack.transposed_b_f16(&inputs[1], geom.k, geom.n) {
                #[cfg(all(test, target_arch = "aarch64"))]
                GEMV_F16_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let a_dense = self.prepack.dense(0, &inputs[0])?;
                let mut result = vec![0.0f32; geom.n];
                accelerate_gemm::neon_gemv_f16_col_parallel(
                    &a_dense,
                    bt_f16,
                    &mut result,
                    geom.k,
                    geom.n,
                );
                return write_dense_f32_narrow("MatMul", &mut outputs[0], &result);
            }
            // Column-major B[K,N] with strides [1, K]: memory is already
            // B^T[N,K] row-major — exactly what the GEMV expects. Use the raw
            // mmap'd data directly (zero-copy, no transpose needed). This
            // avoids a ~960 ms f32 densification on the 896×151936 lm_head
            // weight that would otherwise dominate the first decode step.
            let b_view = &inputs[1];
            if self.prepack.constant_inputs[1]
                && b_view.shape.len() == 2
                && b_view.strides.len() == 2
                && b_view.strides[0] == 1
                && b_view.strides[1] == b_view.shape[0] as i64
            {
                #[cfg(all(test, target_arch = "aarch64"))]
                GEMV_F16_COLMAJ_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                #[cfg(all(test, target_arch = "aarch64"))]
                GEMV_F16_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let a_dense = self.prepack.dense(0, &inputs[0])?;
                let bt_f16 = unsafe {
                    std::slice::from_raw_parts(b_view.data_ptr::<u16>(), geom.k * geom.n)
                };
                let mut result = vec![0.0f32; geom.n];
                accelerate_gemm::neon_gemv_f16_col_parallel(
                    &a_dense,
                    bt_f16,
                    &mut result,
                    geom.k,
                    geom.n,
                );
                return write_dense_f32_narrow("MatMul", &mut outputs[0], &result);
            }
        }

        // x86 16-bit storage GEMV: the mirror of the Accelerate path above.
        // `try_matmul_half` packs both operands into cache-sized panels, which
        // pays for itself only when a panel of B is reused across several rows
        // of A. At M=1 there is no reuse, so the packing is pure overhead on a
        // memory-bound problem. Must precede `try_matmul_half` for the same
        // reason the Accelerate block does.
        //
        // Serves bf16 as well as f16. bf16 had no decode GEMV at all, so a
        // single-token decode was widening and packing the entire weight to
        // multiply it by one row.
        //
        // B is read in place, in its stored [K, N] order — deliberately *not*
        // through `transposed_b_f16`. `try_matmul_half` allocates no weight
        // copy, so a transpose cache here would add a permanent 2*K*N bytes
        // (272 MB for a 896x151936 lm_head) that the path it replaces never
        // paid. Reading in place also keeps this available for a
        // non-constant B, which a prepacked transpose could not serve.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if geom.m == 1
            && numel(&geom.batch_shape) <= 1
            && geom.b_promoted_rank == 2
            && inputs[1].is_contiguous()
            && inputs[1].numel() == geom.k.saturating_mul(geom.n)
            && let Some(format) = half_storage_format(inputs[0].dtype, inputs[1].dtype)
            && half_gemv::simd_available(format)
            && half_decode_gemv_enabled()
            && !half_decode_prefers_gebp(format, geom.k, geom.n)
        {
            inputs[1].validate()?;
            let a_dense = self.prepack.dense(0, &inputs[0])?;
            if a_dense.len() == geom.k {
                // SAFETY: `inputs[1]` was just validated as a contiguous
                // Float16/BFloat16 view whose element count equals `k * n`.
                // Both are transparent over `u16`, so reading their storage as
                // raw bit patterns is sound, and the view outlives this call.
                let b_bits = unsafe {
                    std::slice::from_raw_parts(inputs[1].data_ptr::<u16>(), geom.k * geom.n)
                };
                let mut result = vec![0.0f32; geom.n];
                count_half_decode_gemv();
                half_gemv::gemv_half_kn(format, &a_dense, b_bits, &mut result, geom.k, geom.n);
                return write_dense_f32_narrow("MatMul", &mut outputs[0], &result);
            }
        }

        // f16 prefill through MLAS SGEMM on a once-widened, once-packed B.
        // Shared with `Gemm`; see `try_packed_half_prefill`.
        #[cfg(feature = "mlas")]
        if numel(&geom.batch_shape) <= 1
            && geom.b_promoted_rank == 2
            && let Some(result) = try_packed_half_prefill(
                &self.prepack,
                backend,
                &inputs[0],
                &inputs[1],
                geom.m,
                geom.k,
                geom.n,
            )?
        {
            return write_dense_f32_narrow("MatMul", &mut outputs[0], &result);
        }

        // Dedicated half-precision path: contiguous f16/bf16 operands stay in
        // 16-bit storage and are packed in cache-sized panels for f32
        // accumulation. Bf16 may use the runtime-gated AVX-512 BF16 kernel;
        // every other host uses the portable blocked implementation.
        if let Some(result) = try_matmul_half(&inputs[0], &inputs[1], &geom, backend)? {
            return write_dense_f32_narrow("MatMul", &mut outputs[0], &result);
        }

        // Non-contiguous f16 weight rescue: when B is a constant Float16 weight
        // stored column-major (e.g. lm_head vocab projection), route through
        // BNNS with `trans_b: true` to avoid materialising a contiguous copy.
        //
        // Column-major B[K,N] in memory is equivalent to row-major B^T[N,K].
        // BNNS `trans_b` handles the transpose internally via AMX, eliminating
        // a 272 MB strided copy that previously cost ~280 ms.
        //
        // For non-column-major non-contiguous layouts, fall back to the
        // contiguous copy path (cached via OnceLock).
        //
        // The `constant_inputs[1]` guard is essential: non-constant activations
        // (e.g. a Transpose view of another op's output) must NOT enter this
        // block because `contiguous_b_f16()` returns None for non-constants,
        // which would leave `out` as all zeros — a silent correctness bug.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if inputs[0].dtype == onnx_runtime_ir::DataType::Float16
            && inputs[1].dtype == onnx_runtime_ir::DataType::Float16
            && !inputs[1].is_contiguous()
            && self.prepack.constant_inputs[1]
            && geom.m >= 2
            && inputs[0].is_contiguous()
        {
            #[cfg(test)]
            NONCONTIG_RESCUE_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let b_view = &inputs[1];
            let b_shape = b_view.shape;
            let b_strides = b_view.strides;

            let a_len = inputs[0].numel();
            let a_bits = unsafe { std::slice::from_raw_parts(inputs[0].data_ptr::<u16>(), a_len) };
            let b_len = b_view.numel();
            let mut out = vec![0.0f32; geom.result_len];

            // Check for column-major: 2-D with strides [1, K] (stride-0 = 1,
            // stride-1 = rows). This is exactly B^T stored row-major.
            let is_column_major = b_shape.len() == 2
                && b_strides.len() == 2
                && b_strides[0] == 1
                && b_strides[1] == b_shape[0] as i64;

            // A trivial batch (all dims are 1) behaves identically to no batch.
            let trivial_batch =
                geom.batch_shape.is_empty() || geom.batch_shape.iter().all(|&d| d == 1);

            if !out.is_empty() && geom.k > 0 {
                if is_column_major && trivial_batch {
                    // B is column-major [K,N]: memory is row-major B^T[N,K].
                    // Pass the raw data and let BNNS transpose internally.
                    let bt_bits =
                        unsafe { std::slice::from_raw_parts(b_view.data_ptr::<u16>(), b_len) };
                    if !accelerate_gemm::bnns_matmul_f16_trans_b(
                        a_bits, bt_bits, &mut out, geom.m, geom.k, geom.n,
                    ) {
                        // Fallback: materialise contiguous copy
                        if let Some(b_contig) = self.prepack.contiguous_b_f16(b_view) {
                            half_gemm_tile(
                                HalfFormat::F16,
                                a_bits,
                                b_contig,
                                &mut out,
                                geom.m,
                                geom.k,
                                geom.n,
                            );
                        }
                    }
                } else if let Some(b_contig) = self.prepack.contiguous_b_f16(b_view) {
                    // Non-column-major or batched: use cached contiguous copy
                    if trivial_batch {
                        if !accelerate_gemm::bnns_matmul_f16(
                            a_bits, b_contig, &mut out, geom.m, geom.k, geom.n,
                        ) {
                            half_gemm_tile(
                                HalfFormat::F16,
                                a_bits,
                                b_contig,
                                &mut out,
                                geom.m,
                                geom.k,
                                geom.n,
                            );
                        }
                    } else {
                        bnns_half_dense_into(a_bits, b_contig, &geom, &mut out);
                    }
                }
            }
            return write_dense_f32_narrow("MatMul", &mut outputs[0], &out);
        }

        // Direct f32 output fast path: when the output is a contiguous Float32
        // CPU tensor that does not alias either input, GEMM writes straight into
        // its backing buffer, skipping both the intermediate result `Vec<f32>`
        // and the narrowing copy performed by `write_dense_f32_narrow`. Every
        // other case (f16/bf16/f64, strided/non-contiguous, or a possibly
        // aliasing output) uses the owned-buffer fallback below unchanged.
        if output_is_direct_f32_eligible(&inputs[0], &inputs[1], &outputs[0]) {
            let out = &mut outputs[0];
            // `validate()` confirms rank/dtype/offset invariants; it does NOT
            // prove backing-buffer bounds — that is the executor's `view_bounds`
            // contract for this SSA output. We still gate the pointer-slice on a
            // logical length match against the computed result length so a
            // mismatched shape errors BEFORE any GEMM write.
            out.validate()?;
            let numel = out.numel();
            if numel != geom.result_len {
                return Err(EpError::KernelFailed(format!(
                    "MatMul: output element count {numel} does not match result length {}",
                    geom.result_len
                )));
            }
            // A zero-sized result writes nothing; return before forming a slice
            // from a possibly-dangling zero-length output pointer.
            if numel == 0 {
                return Ok(());
            }
            let ptr = out.data_ptr_mut::<f32>();
            // SAFETY: the eligibility check proved `out` is a CPU, Float32,
            // row-major-contiguous view that does not alias either input's byte
            // range, so no live input slice overlaps this buffer. `data_ptr_mut`
            // applies `byte_offset` to select the element origin, and the
            // executor's bounds contract guarantees `numel` initialized f32 slots
            // exist there; `numel == geom.result_len` was just verified, so the
            // GEMM writes exactly within bounds. The slice is the sole mutable
            // borrow of this storage for the duration of the call.
            let out_slice = unsafe { std::slice::from_raw_parts_mut(ptr, numel) };
            return matmul_dense_into_with_backend(
                &self.prepack.dense(0, &inputs[0])?,
                &self.prepack.dense(1, &inputs[1])?,
                &geom,
                backend,
                Some(&self.prepack),
                out_slice,
            );
        }

        let out =
            matmul_dense_prepacked_with_backend(&inputs[0], &inputs[1], &self.prepack, backend)?;
        // If either operand was 1-D, the corresponding size-1 axis is squeezed
        // out of the result; the narrowing writer uses the output view's own
        // shape and dtype (f32/f16/bf16/f64), so the buffer matches element for
        // element and rounds to the requested precision.
        write_dense_f32_narrow("MatMul", &mut outputs[0], &out)
    }
}

/// Whether `MatMulKernel::execute` may GEMM directly into `out`'s backing
/// buffer instead of the intermediate-vector + narrowing fallback.
///
/// Requires a CPU, Float32, row-major-contiguous output whose byte range does
/// not overlap either input. The overlap check mirrors the in-place fast-path
/// convention used elsewhere in this crate (`kernels/activations.rs`,
/// `kernels/elementwise.rs`): a same-device pointer-range test on the element
/// origins. It is sound here because a zero-copy input operand
/// (`to_dense_f32_widen` borrows only contiguous Float32 views) is read
/// straight from its own contiguous `numel * 4` byte range, which this test
/// covers exactly; any other input dtype/layout is materialized into a fresh
/// owned buffer before the GEMM, so it cannot alias the output at all.
/// Whether an output tensor is eligible for direct f32 GEMM writes: must be
/// contiguous Float32 on CPU, not aliasing either input.
pub(crate) fn output_is_direct_f32_eligible(
    a: &TensorView,
    b: &TensorView,
    out: &TensorMut,
) -> bool {
    use onnx_runtime_ir::DataType;
    use onnx_runtime_ir::DeviceType;

    if out.device.device_type != DeviceType::Cpu
        || out.dtype != DataType::Float32
        || !out.is_contiguous()
    {
        return false;
    }

    // Row-major-contiguous Float32: the whole logical extent is one dense range
    // starting at the element origin.
    let out_origin = (out.data.0 as *const u8).wrapping_add(out.byte_offset) as usize;
    let out_end = out_origin.saturating_add(out.byte_size());

    !std::iter::once(a)
        .chain(std::iter::once(b))
        .any(|input| output_overlaps_input(out_origin, out_end, input, out.device))
}

/// Pointer-range overlap test between the output byte range `[out_origin,
/// out_end)` and one input's element-origin byte range, on the same device.
/// Absent (null) inputs never overlap.
fn output_overlaps_input(
    out_origin: usize,
    out_end: usize,
    input: &TensorView,
    out_device: onnx_runtime_ir::DeviceId,
) -> bool {
    if input.is_absent() || input.device != out_device {
        return false;
    }
    let in_start = input.data_ptr::<u8>() as usize;
    let in_end = in_start.saturating_add(input.byte_size());
    out_origin < in_end && in_start < out_end
}

/// Minimum `M` at which widening a contiguous `f16` GEMM to `f32` and running
/// the tuned SGEMM beats the portable blocked half GEMM.
///
/// ORT reaches its `f16` MatMul time by widening to `f32` and reusing the same
/// tuned SGEMM it uses for `f32`: measured on this host at `M=128, K=N=2048`,
/// ORT spends 14.27 ms on `f16` and 14.39 ms on `f32` -- the same kernel. Our
/// blocked half GEMM keeps operands in 16-bit storage to save bandwidth, which
/// wins while the operands dominate, but it has no tuned microkernel, so once
/// `M` grows enough for the GEMM to become compute-bound it loses badly.
///
/// `bench_f16_half_vs_widen`, pinned to 16 physical cores, `K=N=2048`, median
/// of 5, two independent runs. Ratio is `half / widen`, so > 1 means widening
/// wins:
///
/// | M | T=1 | T=16 |
/// |---|---|---|
/// | 2 | 0.67x, 0.61x (half wins) | 0.93x, 0.83x (half wins) |
/// | 8 | 1.01x, 1.05x (tie) | 1.26x, 1.38x |
/// | **16** | **1.33x, 1.30x** | **2.14x, 1.85x** |
/// | 32 | 1.56x, 1.63x | 3.14x, 3.32x |
/// | 128 | 1.90x, 2.00x | 3.47x, 3.30x |
/// | 256 | 1.94x, 2.03x | 2.88x, 2.67x |
///
/// `M=16` is the first row that wins repeatably at *both* thread counts, so it
/// is the threshold. `M=8` is a tie at `T=1` (1.01x/1.05x, inside noise) and is
/// left on the half path rather than claimed. Below the threshold the blocked
/// half GEMM is kept, so decode (`M=1`) is bit-for-bit unchanged.
#[cfg(feature = "mlas")]
const HALF_WIDEN_MIN_M: usize = 16;

/// Minimum `B` size, in **elements** (`K * N`, not bytes), before widening is
/// considered.
///
/// Widening `B` costs a `4 * K * N` byte transient on every call when `B` is
/// not a session constant, so the intuition is that a small `B` cannot repay
/// it. The sweep says otherwise: at `M = 16`, `half/widen` (>1 means widening
/// wins) is
///
/// | K x N | elements | T=1 | T=16 |
/// |---|---|---|---|
/// | 8 x 8 | 64 | 0.88x (half wins) | 275x |
/// | 16 x 16 | 256 | 1.16x | 229x |
/// | 32 x 32 | 1024 | 1.21x | 35.9x |
/// | 64 x 64 | 4096 | 1.42x | 33.2x |
/// | 128 x 128 | 16384 | 1.48x | 16.3x |
/// | 256 x 256 | 65536 | 1.52x | 7.8x |
///
/// so this is set to 256 elements: the smallest size that wins repeatably at
/// *both* thread counts. Only the 64-element case loses, and only at `T=1`.
///
/// The three-digit `T=16` ratios are noise-dominated in magnitude -- an
/// independent run of the same sweep put the two smallest at 47x and 28x --
/// but the direction is robust across runs. They are also not a widening win:
/// they are the blocked half GEMM dispatching parallel work for a problem far
/// too small to repay a fork/join (`gemm_impl` splits with `par_chunks_mut`
/// whenever threads > 1, with no small-work guard). Widening sidesteps that;
/// fixing the half path's own threshold is separate work.
#[cfg(feature = "mlas")]
const HALF_WIDEN_MIN_WEIGHT: usize = 256;

#[cfg(test)]
thread_local! {
    /// Test-only count of calls where [`try_matmul_half`] declined in favour of
    /// the widened-`f32` SGEMM. Lets a test assert *which route* ran rather
    /// than only that the numbers came out right.
    static HALF_YIELDED_TO_WIDENED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[inline]
fn count_half_yielded_to_widened() {
    #[cfg(test)]
    HALF_YIELDED_TO_WIDENED.with(|c| c.set(c.get() + 1));
}

/// Number of times the current thread took the widened-`f32` SGEMM instead of
/// the blocked half GEMM.
#[cfg(all(test, feature = "mlas"))]
pub(crate) fn half_yielded_to_widened_calls() -> u64 {
    HALF_YIELDED_TO_WIDENED.with(|c| c.get())
}

#[cfg(all(test, feature = "mlas"))]
pub(crate) fn reset_half_yielded_to_widened_calls() {
    HALF_YIELDED_TO_WIDENED.with(|c| c.set(0));
}

/// Whether a contiguous half GEMM is better served by widening to dense `f32`
/// and using the tuned SGEMM (see [`HALF_WIDEN_MIN_M`]).
///
/// Restricted to `f16`. `bf16` is left on the blocked half GEMM: it has a
/// native AVX512-BF16 kernel whose crossover has not been measured, and
/// widening it is a different (cheaper, shift-only) operation, so the `f16`
/// measurement does not transfer.
///
/// Restricted to [`CpuBackend::Mlas`] because that is the only backend whose
/// SGEMM was measured to beat the half path; every other backend keeps
/// today's behaviour. `auto_detect` returns `Mlas` on x86-64 whenever the
/// feature is compiled in, so this is the default path there.
///
/// End to end through the plugin (`bench_matmul_f16_m128`, `M=128, K=N=2048`,
/// non-constant `B`, 5 interleaved rounds of before/after/ORT in one process,
/// p50 ms, pinned to 16 physical cores), as `ours/ORT` so >1 means we lose.
/// Ranges span three independent measurements, one of them by a reviewer on a
/// separate build:
///
/// | threads | before | after | gain |
/// |---|---|---|---|
/// | 1 | 1.97x--1.99x | **1.05x--1.08x** | 1.85x--1.87x |
/// | 16 | 3.27x--3.29x | **2.03x--2.28x** | 1.44x--1.63x |
///
/// `M=1` decode cannot reach this gate and measured unchanged (0.356 vs 0.353
/// ms at `T=1`, inside run-to-run noise).
///
/// `T=16` is still ~2x off ORT and is **not** claimed as competitive. The
/// residual is the per-call widen of a non-constant `B`: serial here, parallel
/// in ORT. Evidence: our `f16` costs ~2.2 ms more than our own `f32` at `T=16`
/// but only ~1.6 ms more at `T=1`, so the conversion gets *worse* with threads.
/// That is a separate change and this gate does not address it.
fn widened_sgemm_beats_half_gemm(
    format: HalfFormat,
    geom: &MatMulGeometry,
    backend: CpuBackend,
) -> bool {
    // `CpuBackend::Mlas` only exists with the feature compiled in; without it
    // there is no measured-faster SGEMM to yield to, so keep the half GEMM.
    #[cfg(feature = "mlas")]
    {
        format == HalfFormat::F16
            && backend == CpuBackend::Mlas
            && geom.m >= HALF_WIDEN_MIN_M
            && geom.k.saturating_mul(geom.n) >= HALF_WIDEN_MIN_WEIGHT
    }
    #[cfg(not(feature = "mlas"))]
    {
        let _ = (format, geom, backend);
        false
    }
}

/// Attempt the dedicated portable half GEMM path. Both operands must be
/// contiguous and have the same `Float16` or `BFloat16` dtype. The operands stay
/// in 16-bit storage until cache-panel packing, accumulation is always `f32`,
/// and the caller narrows once into the requested output dtype.
///
/// Declines for a large-`M` `f16` GEMM, where widening to `f32` and using the
/// tuned SGEMM is measurably faster -- see [`widened_sgemm_beats_half_gemm`].
/// Every caller already falls through to that widened path.
/// Minimum `B` size, in **elements** (`K * N`), before an `M == 1` decode
/// gives up the GEMV and lets the fused widen-pack GEBP serve it.
///
/// Far above [`HALF_PREFILL_GEBP_MIN_WEIGHT`], and deliberately so. The GEMV
/// reads the weight once and packs nothing, which is the least traffic any
/// route can issue; the packing only wins once the weight is so much larger
/// than any cache that the GEMV's stripe parallelism, not its traffic, is the
/// limit -- `n / STRIPE` workers, each walking the whole `k` in strided
/// visits. Measured on 32 vCPU / 16 AVX2 cores, `M == 1`, steady p50 of 7-9
/// interleaved repetitions (a shared host, so the minimum of the same samples
/// is given where the two disagree):
///
/// | `K x N` | elements | GEMV | fused GEBP |
/// |---|---|---|---|
/// | f16 4096x1024 | 4.2M | **0.39 ms** | 0.59 ms |
/// | bf16 4096x2048 | 8.4M | **0.86 ms** | 0.71 ms (min 0.59 vs 0.51) |
/// | f16 4096x4096 | 16.8M | 1.13 ms | **1.06 ms** |
/// | f16 4096x8192 | 33.6M | 1.92 ms | 1.91 ms |
/// | bf16 4096x8192 | 33.6M | 1.87 ms | 2.01 ms |
/// | f16 4096x11008 | 45.1M | 2.47 ms | **2.46 ms** (min 2.30 vs 2.16) |
/// | bf16 4096x11008 | 45.1M | 3.32 ms | **2.45 ms** |
/// | f16 896x151936 | 136M | 6.94 ms | **5.84 ms** |
/// | bf16 896x151936 | 136M | 6.61 ms | **5.56 ms** |
///
/// `1 << 25` is the *wash*, not the win: at exactly 33.6M the two are within
/// run-to-run noise in both formats, and above it the GEBP pulls 6%-26%
/// ahead. Placing the threshold at the wash rather than at the first clear win
/// means no shape is left on the slower route, and no shape is moved onto the
/// allocating one for a gain that could not be measured.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const HALF_DECODE_GEBP_MIN_WEIGHT: usize = 1 << 25;

/// Whether an `M == 1` decode should skip the GEMV and let the fused
/// widen-pack GEBP serve it -- see [`HALF_DECODE_GEBP_MIN_WEIGHT`].
///
/// Mirrors [`half_gemm_tile`]'s precedence: if the AVX-512 BF16 kernel would
/// claim the call, or the GEBP is switched off, this must not divert a decode
/// away from the GEMV into the row-blocked GEMM.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn half_decode_prefers_gebp(format: HalfFormat, k: usize, n: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    if format == HalfFormat::Bf16 && x86_bf16::native_available() {
        return false;
    }
    let _ = format;
    k.saturating_mul(n) >= HALF_DECODE_GEBP_MIN_WEIGHT && half_prefill_gebp_enabled()
}

/// Whether the `M == 1` decode GEMV is used. On by default;
/// `ONNX_GENAI_CPU_MM_HALF_GEMV=0` (or `off`) sends decode back through the
/// blocking path for the whole process, so a regression can be bisected in the
/// field without a rebuild, and so the A/B bench can measure both arms of the
/// shipped binary. Read once and cached, like `half_prefill_gebp_enabled`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn half_decode_gemv_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ONNX_GENAI_CPU_MM_HALF_GEMV")
            .ok()
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("off"))
            })
            .unwrap_or(true)
    })
}

/// The 16-bit storage format both operands share, if they share one.
///
/// Mixed and non-16-bit pairs return `None`: every kernel behind this reads
/// both operands as raw `u16` and widens them the same way, so a mismatch
/// would silently reinterpret one of them.
fn half_storage_format(
    a: onnx_runtime_ir::DataType,
    b: onnx_runtime_ir::DataType,
) -> Option<HalfFormat> {
    use onnx_runtime_ir::DataType;
    match (a, b) {
        (DataType::Float16, DataType::Float16) => Some(HalfFormat::F16),
        (DataType::BFloat16, DataType::BFloat16) => Some(HalfFormat::Bf16),
        _ => None,
    }
}

fn try_matmul_half(
    a: &TensorView,
    b: &TensorView,
    geom: &MatMulGeometry,
    backend: CpuBackend,
) -> Result<Option<Vec<f32>>> {
    let Some(format) = half_storage_format(a.dtype, b.dtype) else {
        return Ok(None);
    };
    if !a.is_contiguous() || !b.is_contiguous() {
        return Ok(None);
    }
    if widened_sgemm_beats_half_gemm(format, geom, backend) {
        count_half_yielded_to_widened();
        return Ok(None);
    }
    a.validate()?;
    b.validate()?;
    let mut out = vec![0.0f32; geom.result_len];
    if out.is_empty() || geom.k == 0 {
        return Ok(Some(out));
    }
    let a_len = a.numel();
    let b_len = b.numel();
    // SAFETY: validated contiguous Float16/BFloat16 views address exactly
    // `a_len`/`b_len` two-byte elements. Both half types are transparent over
    // `u16`, so reading their storage as raw bit patterns is sound.
    let a_bits = unsafe { std::slice::from_raw_parts(a.data_ptr::<u16>(), a_len) };
    let b_bits = unsafe { std::slice::from_raw_parts(b.data_ptr::<u16>(), b_len) };

    // BNNS fp16→f32 path for M≥2 on macOS: reaches AMX, ~15–25× faster than
    // the portable NEON blocked GEMM at prefill shapes. Only for f16 (not bf16).
    // Called from the dispatch level, NOT from inside a Rayon parallel region.
    //
    // The M≥2 threshold is categorical (GEMV at M=1 vs GEMM at M≥2), not a
    // tuned crossover: there is no integer between 1 and 2, so this carries no
    // machine-specific assumption and should not be "tuned" to a higher value.
    // At M=1 the bandwidth-optimal NEON GEMV path (above) dominates; at M=2+
    // the arithmetic intensity is high enough for AMX to outperform even at
    // small M despite the ~50 µs GCD dispatch overhead per call.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if format == HalfFormat::F16 && geom.m >= 2 && accelerate_gemm::bnns_matmul_available() {
        #[cfg(test)]
        BNNS_F16_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        bnns_half_dense_into(a_bits, b_bits, geom, &mut out);
        let elapsed = t0.elapsed();
        BNNS_PREFILL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        BNNS_PREFILL_NANOS.fetch_add(
            elapsed.as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        return Ok(Some(out));
    }

    half_dense_into(format, a_bits, b_bits, geom, &mut out);
    Ok(Some(out))
}

/// Drive the half GEMM over a single, batched, or broadcast operand pair into
/// `out` (row-major f32), reusing the shared batch-broadcast walk.
fn half_dense_into(
    format: HalfFormat,
    a: &[u16],
    b: &[u16],
    geom: &MatMulGeometry,
    out: &mut [f32],
) {
    let (m, k, n) = (geom.m, geom.k, geom.n);
    let (a_mat, b_mat, c_mat) = (geom.a_mat, geom.b_mat, geom.c_mat);
    if geom.batch_shape.is_empty() {
        half_gemm_tile(format, a, b, out, m, k, n);
        return;
    }
    let mut bidx = vec![0usize; geom.batch_shape.len()];
    let mut b_out = 0usize;
    loop {
        let a_off = broadcast_offset(&bidx, &geom.a_batch, &geom.a_batch_strides) * a_mat;
        let b_off = broadcast_offset(&bidx, &geom.b_batch, &geom.b_batch_strides) * b_mat;
        half_gemm_tile(
            format,
            &a[a_off..a_off + a_mat],
            &b[b_off..b_off + b_mat],
            &mut out[b_out * c_mat..b_out * c_mat + c_mat],
            m,
            k,
            n,
        );
        b_out += 1;
        if !next_index(&geom.batch_shape, &mut bidx) {
            break;
        }
    }
}

/// Drive BNNS fp16→f32 matmul over a single, batched, or broadcast operand pair.
/// Each per-tile call is made from the dispatch level (not inside a Rayon parallel
/// region) to avoid oversubscribing GCD threads. Falls back to the portable
/// half GEMM if BNNS fails for a particular tile.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn bnns_half_dense_into(a: &[u16], b: &[u16], geom: &MatMulGeometry, out: &mut [f32]) {
    let (m, k, n) = (geom.m, geom.k, geom.n);
    let (a_mat, b_mat, c_mat) = (geom.a_mat, geom.b_mat, geom.c_mat);
    if geom.batch_shape.is_empty() {
        if !accelerate_gemm::bnns_matmul_f16(a, b, out, m, k, n) {
            half_gemm_tile(HalfFormat::F16, a, b, out, m, k, n);
        }
        return;
    }
    let mut bidx = vec![0usize; geom.batch_shape.len()];
    let mut b_out = 0usize;
    loop {
        let a_off = broadcast_offset(&bidx, &geom.a_batch, &geom.a_batch_strides) * a_mat;
        let b_off = broadcast_offset(&bidx, &geom.b_batch, &geom.b_batch_strides) * b_mat;
        let a_tile = &a[a_off..a_off + a_mat];
        let b_tile = &b[b_off..b_off + b_mat];
        let c_tile = &mut out[b_out * c_mat..b_out * c_mat + c_mat];
        if !accelerate_gemm::bnns_matmul_f16(a_tile, b_tile, c_tile, m, k, n) {
            half_gemm_tile(HalfFormat::F16, a_tile, b_tile, c_tile, m, k, n);
        }
        b_out += 1;
        if !next_index(&geom.batch_shape, &mut bidx) {
            break;
        }
    }
}

/// Select the runtime-gated AVX-512 BF16 microkernel when available; otherwise
/// use the fused widen-pack prefill GEBP (x86-64, AVX2/FMA, `m` above the
/// measured crossover) or the portable blocked half GEMM.
fn half_gemm_tile(
    format: HalfFormat,
    a: &[u16],
    b: &[u16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if format == HalfFormat::Bf16 && x86_bf16::native_available() {
        x86_bf16::gemm(a, b, c, m, k, n);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if half_prefill_gebp_selected(format, m, k, n) && half_prefill_gebp_enabled() {
        count_half_prefill_gebp();
        x86_sgemm::half_prefill_gebp(format, a, b, c, m, k, n);
        return;
    }

    half_gemm::gemm(
        format,
        a,
        MatrixLayout::row_major(k),
        b,
        MatrixLayout::row_major(n),
        c,
        m,
        k,
        n,
    );
}

/// Whether a contiguous half GEMM tile is better served by the fused
/// widen-pack GEBP ([`x86_sgemm::half_prefill_gebp`]) than by the blocked half
/// GEMM.
///
/// The blocked half GEMM splits only over rows of C and re-widens/re-packs the
/// whole of `B` per row block, so its weight traffic scales with `m`: at
/// `m = 64` on this 32-thread host its own block size collapses to one row,
/// i.e. 64 full passes over the weight. The GEBP traverses `B` once per column
/// strip whatever `m` is, so the question is only whether a single widen-pack
/// of `B` is repaid — which is why the gate is on the *weight* size and not on
/// `m * k * n`.
///
/// Requires AVX2 + FMA because it reuses the f32 microkernel.
#[cfg(target_arch = "x86_64")]
fn half_prefill_gebp_selected(format: HalfFormat, m: usize, k: usize, n: usize) -> bool {
    if !crate::backend::has_simd_x86() {
        return false;
    }
    if k.saturating_mul(n) < HALF_PREFILL_GEBP_MIN_WEIGHT {
        return false;
    }
    // `m == 1` reaches here two ways. A `bf16` tile that never sees the decode
    // GEMV -- batched, or a host without the GEMV's features -- would
    // otherwise pay a full widen-pack of `B` on the blocked kernel for one
    // row. And either format, once the weight clears
    // `HALF_DECODE_GEBP_MIN_WEIGHT`, is deliberately routed here *by*
    // `half_decode_prefers_gebp` declining the GEMV, so this gate must accept
    // it or that decode would land on the blocked kernel instead.
    m >= HALF_PREFILL_GEBP_MIN_ROWS
        || format == HalfFormat::Bf16
        || k.saturating_mul(n) >= HALF_DECODE_GEBP_MIN_WEIGHT
}

/// Minimum `M` at which the fused widen-pack GEBP replaces the blocked half
/// GEMM (`bf16` also takes it at `m == 1`; see
/// [`half_prefill_gebp_selected`]).
///
/// Every `m >= 2` wins once the weight clears
/// [`HALF_PREFILL_GEBP_MIN_WEIGHT`], so this is categorical rather than a
/// tuned crossover, and carries no machine-specific assumption.
///
/// `M = 1` is not a row-count decision at all: contiguous 2-D decode is
/// intercepted upstream by the 16-bit GEMV, and only comes here when
/// [`HALF_DECODE_GEBP_MIN_WEIGHT`] says the weight is large enough that the
/// packing wins, or when the tile could never reach that GEMV in the first
/// place (batched, or a `bf16` host without it). Both of those are spelled out
/// in [`half_prefill_gebp_selected`] rather than folded into this constant.
#[cfg(target_arch = "x86_64")]
const HALF_PREFILL_GEBP_MIN_ROWS: usize = 2;

/// Minimum `B` size, in **elements** (`K * N`), before the fused widen-pack
/// GEBP is used.
///
/// The GEBP widens and packs `B` once per column strip; below some weight size
/// that single pass, plus the fork/join over strips, costs more than the
/// re-packing it removes.
///
/// [`bench_half_prefill_gebp_crossover`], release build, p50 of 5, both routes
/// interleaved rep-by-rep. Ratio is `blocked / gebp`, so > 1 means the GEBP
/// wins; each cell is `T=32 / T=4`:
///
/// | K x N | elements | m=2 | m=4 | m=8 | m=16 |
/// |---|---|---|---|---|---|
/// | 256 x 256 | 65_536 | 0.60 / 0.77 | 0.59 / 0.86 | 0.56 / 1.02 | 3.42 / 1.09 |
/// | 512 x 512 | 262_144 | 1.18 / 1.84 | 3.76 / 2.39 | 3.59 / 3.18 | 4.21 / 1.29 |
/// | 768 x 768 | 589_824 | 4.42 / 3.08 | 5.44 / 3.39 | 3.84 / 3.71 | 5.76 / 1.64 |
/// | **1024 x 1024** | **1_048_576** | **5.01 / 3.36** | **7.69 / 3.42** | **6.61 / 4.24** | **4.67 / 1.95** |
/// | 1536 x 1536 | 2_359_296 | 6.79 / 3.59 | 6.63 / 5.53 | 7.67 / 5.53 | 7.38 / 2.47 |
/// | 2048 x 2048 | 4_194_304 | 6.58 / 3.47 | 7.37 / 3.15 | 6.39 / 6.79 | 8.69 / 3.53 |
///
/// (`f16`; the `bf16` half of the same run agrees within noise.)
///
/// That harness times the two routes directly. End to end through the
/// production kernel (`bench half_prefill_route_ab`, which also pays the
/// dispatch and the narrowing of the output) the ratios are damped and
/// `512 x 512` *loses* at `T=32` for `m = 2..8` (0.63x-0.98x) while still
/// winning at `T=4`. `1_048_576` is the smallest weight that wins in **both**
/// harnesses at both thread counts and in both formats, so that is the gate;
/// `512 x 512` and `768 x 768` are left on the blocked route rather than
/// claimed. Below the gate the loss is real, not noise: 0.56x at `256 x 256`.
///
/// Deliberately the same shape of rule -- and the same value -- as the sibling
/// guard `half_gemm::PARALLEL_MIN_WORK`, which exists because the same
/// fork/join stops paying at the same scale.
#[cfg(target_arch = "x86_64")]
const HALF_PREFILL_GEBP_MIN_WEIGHT: usize = 1_048_576;

/// Whether the fused widen-pack prefill GEBP is enabled. On by default;
/// `ONNX_GENAI_CPU_MM_HALF_GEBP=0` (or `off`) restores the blocked half GEMM
/// for the whole process, so a regression can be bisected in the field without
/// a rebuild. Read-only env probe -- production never mutates it at runtime.
///
/// Read once and cached, like `x86_bf16::native_available`: the doc says "for
/// the whole process", and a `OnceLock` is what makes that literally true
/// rather than "for every call that happens to read the same value".
#[cfg(target_arch = "x86_64")]
fn half_prefill_gebp_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ONNX_GENAI_CPU_MM_HALF_GEBP")
            .ok()
            .map(|value| {
                let value = value.trim();
                value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("off"))
            })
            .unwrap_or(true)
    })
}

#[cfg(all(test, target_arch = "x86_64"))]
thread_local! {
    /// Test-only count of half GEMM tiles served by the fused widen-pack GEBP,
    /// so a test can assert *which route* ran rather than only the numbers.
    static HALF_PREFILL_GEBP_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn count_half_prefill_gebp() {
    #[cfg(test)]
    HALF_PREFILL_GEBP_CALLS.with(|c| c.set(c.get() + 1));
}

#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn half_prefill_gebp_calls() -> u64 {
    HALF_PREFILL_GEBP_CALLS.with(std::cell::Cell::get)
}

#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn reset_half_prefill_gebp_calls() {
    HALF_PREFILL_GEBP_CALLS.with(|c| c.set(0));
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
thread_local! {
    /// Test-only count of decodes served by the 16-bit GEMV, so a test can
    /// assert *which route* ran rather than only the numbers. The two routes
    /// agree to half-precision rounding, so numbers alone cannot tell them
    /// apart.
    static HALF_DECODE_GEMV_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn count_half_decode_gemv() {
    #[cfg(test)]
    HALF_DECODE_GEMV_CALLS.with(|c| c.set(c.get() + 1));
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn half_decode_gemv_calls() -> u64 {
    HALF_DECODE_GEMV_CALLS.with(std::cell::Cell::get)
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn reset_half_decode_gemv_calls() {
    HALF_DECODE_GEMV_CALLS.with(|c| c.set(0));
}

/// Compute `A @ B` (numpy semantics: batched, broadcast leading dims, 1-D
/// operand promotion) into a dense row-major `Vec<f32>`.
///
/// Operands may be any float dtype (`f32`/`f16`/`bf16`/`f64`). Contiguous half
/// inputs use the blocked half GEMM; other low/medium precision layouts widen to
/// dense `f32`. Both routes accumulate in `f32`. Shared by [`MatMulKernel`] and
/// the fused `FusedMatMulBias` kernel.
pub(crate) fn matmul_dense(a: &TensorView, b: &TensorView) -> Result<Vec<f32>> {
    let geom = matmul_geometry(a, b)?;
    if let Some(result) = try_matmul_half(a, b, &geom, CpuBackend::auto_detect())? {
        return Ok(result);
    }
    matmul_dense_impl_with_geom(
        to_dense_f32_widen("MatMul", a)?,
        to_dense_f32_widen("MatMul", b)?,
        &geom,
        CpuBackend::auto_detect(),
        None,
    )
}

pub(crate) fn matmul_dense_prepacked(
    a: &TensorView,
    b: &TensorView,
    prepack: &MatMulPrepack,
) -> Result<Vec<f32>> {
    matmul_dense_prepacked_with_backend(a, b, prepack, CpuBackend::auto_detect())
}

/// Compute `MatMul(A, B)` directly into a caller-supplied output slice,
/// skipping the intermediate `Vec<f32>` allocation. Returns the number of
/// elements written (equal to the result length from matmul_geometry).
/// Used by FusedMatMulBias's direct-output path.
pub(crate) fn matmul_dense_prepacked_into(
    a: &TensorView,
    b: &TensorView,
    prepack: &MatMulPrepack,
    out: &mut [f32],
) -> Result<usize> {
    let backend = CpuBackend::auto_detect();
    let geom = matmul_geometry(a, b)?;
    if out.len() < geom.result_len {
        return Err(EpError::KernelFailed(format!(
            "FusedMatMulBias direct: output buffer {} < result length {}",
            out.len(),
            geom.result_len
        )));
    }
    let out_slice = &mut out[..geom.result_len];
    matmul_dense_into_with_backend(
        &prepack.dense(0, a)?,
        &prepack.dense(1, b)?,
        &geom,
        backend,
        Some(prepack),
        out_slice,
    )?;
    Ok(geom.result_len)
}

fn matmul_dense_prepacked_with_backend(
    a: &TensorView,
    b: &TensorView,
    prepack: &MatMulPrepack,
    backend: CpuBackend,
) -> Result<Vec<f32>> {
    let geom = matmul_geometry(a, b)?;
    if let Some(result) = try_matmul_half(a, b, &geom, backend)? {
        return Ok(result);
    }
    matmul_dense_impl_with_geom(
        prepack.dense(0, a)?,
        prepack.dense(1, b)?,
        &geom,
        backend,
        Some(prepack),
    )
}

/// Shared owned-vector GEMM: allocate the result buffer and GEMM into it.
/// Geometry is pre-computed by the caller to avoid redundant derivation.
fn matmul_dense_impl_with_geom(
    a_dense: Cow<'_, [f32]>,
    b_dense: Cow<'_, [f32]>,
    geom: &MatMulGeometry,
    backend: CpuBackend,
    prepack: Option<&MatMulPrepack>,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; geom.result_len];
    matmul_dense_into_with_backend(&a_dense, &b_dense, geom, backend, prepack, &mut out)?;
    Ok(out)
}

/// Precomputed MatMul dimensions: 1-D promotion, inner-dim agreement, batch
/// broadcast, and per-tile element counts. Computed once so both the owned and
/// direct-output paths share exactly one geometry derivation.
struct MatMulGeometry {
    m: usize,
    k: usize,
    n: usize,
    a_mat: usize,
    b_mat: usize,
    c_mat: usize,
    a_batch: Vec<usize>,
    b_batch: Vec<usize>,
    a_batch_strides: Vec<i64>,
    b_batch_strides: Vec<i64>,
    batch_shape: Vec<usize>,
    /// Promoted rank of B; the MLAS PackedB path applies only to a 2-D B.
    #[cfg_attr(not(feature = "mlas"), allow(dead_code))]
    b_promoted_rank: usize,
    /// Total elements in the result: `batch_count * c_mat`.
    result_len: usize,
}

/// Derive [`MatMulGeometry`] from the two operand views (numpy matmul
/// semantics: 1-D promotion, inner-dim check, broadcast leading dims).
fn matmul_geometry(a: &TensorView, b: &TensorView) -> Result<MatMulGeometry> {
    // Promote 1-D operands per numpy matmul: a [K] -> [1,K] (drop row after),
    // b [K] -> [K,1] (drop col after).
    let a_raw = a.shape;
    let b_raw = b.shape;
    let a_1d = a_raw.len() == 1;
    let b_1d = b_raw.len() == 1;
    let a_shape: Vec<usize> = if a_1d {
        vec![1, a_raw[0]]
    } else {
        a_raw.to_vec()
    };
    let b_shape: Vec<usize> = if b_1d {
        vec![b_raw[0], 1]
    } else {
        b_raw.to_vec()
    };

    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(EpError::KernelFailed(
            "MatMul: operands must be at least 1-D".into(),
        ));
    }

    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let k2 = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    if k != k2 {
        return Err(EpError::KernelFailed(format!(
            "MatMul: inner dims disagree ({k} vs {k2})"
        )));
    }

    // Broadcast the batch (leading) dimensions.
    let a_batch = a_shape[..a_shape.len() - 2].to_vec();
    let b_batch = b_shape[..b_shape.len() - 2].to_vec();
    let batch_shape = broadcast_shapes(&a_batch, &b_batch)?;
    let batch_count = numel(&batch_shape);

    let a_batch_strides = compute_contiguous_strides(&a_batch);
    let b_batch_strides = compute_contiguous_strides(&b_batch);
    let a_mat = m * k;
    let b_mat = k * n;
    let c_mat = m * n;

    Ok(MatMulGeometry {
        m,
        k,
        n,
        a_mat,
        b_mat,
        c_mat,
        a_batch,
        b_batch,
        a_batch_strides,
        b_batch_strides,
        batch_shape,
        b_promoted_rank: b_shape.len(),
        result_len: batch_count * c_mat,
    })
}

/// Run the GEMM (single, batched, or broadcast) into the caller-supplied
/// row-major `out` slice. `out.len()` MUST equal `geom.result_len`; a mismatch
/// returns an `EpError` before any write. This is the single code path shared by
/// the owned-vector wrapper and the direct-output fast path.
fn matmul_dense_into_with_backend(
    a_dense: &[f32],
    b_dense: &[f32],
    geom: &MatMulGeometry,
    backend: CpuBackend,
    prepack: Option<&MatMulPrepack>,
    out: &mut [f32],
) -> Result<()> {
    if out.len() != geom.result_len {
        return Err(EpError::KernelFailed(format!(
            "MatMul: output buffer length {} does not match result length {}",
            out.len(),
            geom.result_len
        )));
    }

    // Any zero dimension (batch, M, or N) yields an empty result — matching
    // numpy/ONNX reference semantics. Return before the compute loop, which
    // otherwise runs once even for a zero-sized batch (a `loop { … } while`) and
    // would index into empty operand slices.
    if out.is_empty() {
        return Ok(());
    }

    let (m, k, n) = (geom.m, geom.k, geom.n);
    let (a_mat, b_mat, c_mat) = (geom.a_mat, geom.b_mat, geom.c_mat);

    // `prepack` is consumed on macOS/iOS (Accelerate transposed_b) and with the
    // `mlas` feature (packed_b). On other platforms it is passed through for API
    // consistency but not yet consumed.
    #[cfg(not(any(feature = "mlas", target_os = "macos", target_os = "ios")))]
    let _ = &prepack;

    #[cfg(feature = "mlas")]
    let packed_b = if backend == CpuBackend::Mlas && geom.b_promoted_rank == 2 {
        prepack.and_then(|prepack| prepack.packed_b(b_dense, k, n))
    } else {
        None
    };

    // Accelerate hybrid M=1 fast path (macOS/iOS):
    //
    // Column-parallel NEON GEMV on pre-transposed B_T, parallelized via the
    // Rayon decode pool. Single-threaded for small matrices (dispatch > compute).
    // Accelerate sgemv was measured at ~30-50 µs GCD wake-up overhead per call,
    // making it SLOWER than single-threaded NEON for L2-resident matrices (the
    // wake-up dominates the compute saving). Accelerate sgemm is retained for
    // M>1 prefill where AMX provides genuine compute benefit.
    //
    // Treat batch_shape with product ≤ 1 as non-batched: during decode the
    // activation is typically [1, 1, K] against a [K, N] weight, giving
    // batch_shape = [1]. The single-element batch is equivalent to no batch
    // and must not skip this optimized path.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if backend == CpuBackend::Accelerate
        && m == 1
        && numel(&geom.batch_shape) <= 1
        && geom.b_promoted_rank == 2
    {
        if let Some(bt) = prepack.and_then(|p| p.transposed_b(b_dense, k, n)) {
            accelerate_gemm::neon_gemv_col_parallel(a_dense, bt, out, k, n);
        } else {
            accelerate_gemm::neon_gemv_parallel(a_dense, b_dense, out, k, n);
        }
        return Ok(());
    }

    // Thin-M GEMM: column-parallel NEON for M=2..16 with large B (f32 on macOS).
    //
    // When B is too large for the SLC and M is small, `cblas_sgemm`'s panel
    // tiling reads B inefficiently. The column-parallel approach streams B_T
    // once per thread (each column's data stays L1-hot across M rows),
    // achieving 2–3× speedup for the lm_head projection [7,768]×[768,50257].
    //
    // Requires pre-transposed B (constant weight). Non-constant B falls through
    // to cblas_sgemm. Does NOT affect the fp16 BNNS path (handled earlier).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if backend == CpuBackend::Accelerate
        && accelerate_gemm::thin_m_gemm_eligible(m, k, n)
        && numel(&geom.batch_shape) <= 1
        && geom.b_promoted_rank == 2
    {
        if let Some(bt) = prepack.and_then(|p| p.transposed_b(b_dense, k, n)) {
            #[cfg(all(test, target_arch = "aarch64"))]
            THIN_M_GEMM_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Column-parallel thin-M GEMM: process strips of 4 B_T columns at
            // a time, computing all M rows per strip while data is L1-hot.
            // This is ~2.6× faster than cblas_sgemm for these shapes.
            accelerate_gemm::neon_thin_m_gemm_col_parallel(a_dense, bt, out, m, k, n);
            return Ok(());
        }
    }

    // A batch product of 1 (e.g. batch_shape = [1] from a [1,1,K] activation
    // against a [K,N] weight) is equivalent to no batch. Use the non-batched
    // path to pick up packed_b, avoid the per-batch iteration overhead, and
    // stay consistent with the Accelerate M=1 fast path above.
    if numel(&geom.batch_shape) <= 1 {
        // No effective batch dims: a single matmul.
        #[cfg(feature = "mlas")]
        if let Some(packed_b) = packed_b {
            gemm_packed(a_dense, packed_b, out, m, k, n)?;
        } else {
            gemm_with_backend(backend, a_dense, b_dense, out, m, k, n)?;
        }
        #[cfg(not(feature = "mlas"))]
        gemm_with_backend(backend, a_dense, b_dense, out, m, k, n)?;
    } else {
        let mut bidx = vec![0usize; geom.batch_shape.len()];
        let mut b_out = 0usize;
        loop {
            let a_off = broadcast_offset(&bidx, &geom.a_batch, &geom.a_batch_strides) * a_mat;
            let b_off = broadcast_offset(&bidx, &geom.b_batch, &geom.b_batch_strides) * b_mat;
            let a_tile = &a_dense[a_off..a_off + a_mat];
            let c_tile = &mut out[b_out * c_mat..b_out * c_mat + c_mat];
            #[cfg(feature = "mlas")]
            if let Some(packed_b) = packed_b {
                gemm_packed(a_tile, packed_b, c_tile, m, k, n)?;
            } else {
                gemm_with_backend(
                    backend,
                    a_tile,
                    &b_dense[b_off..b_off + b_mat],
                    c_tile,
                    m,
                    k,
                    n,
                )?;
            }
            #[cfg(not(feature = "mlas"))]
            gemm_with_backend(
                backend,
                a_tile,
                &b_dense[b_off..b_off + b_mat],
                c_tile,
                m,
                k,
                n,
            )?;
            b_out += 1;
            if !next_index(&geom.batch_shape, &mut bidx) {
                break;
            }
        }
    }

    Ok(())
}

/// Element offset of batch index `bidx` into a batch of shape `batch`,
/// broadcasting any size-1 axis (stride 0). `bidx` is indexed over the
/// broadcast (output) batch shape, right-aligned onto `batch`.
fn broadcast_offset(bidx: &[usize], batch: &[usize], batch_strides: &[i64]) -> usize {
    let out_rank = bidx.len();
    let mut off = 0i64;
    for axis in 0..batch.len() {
        let out_axis = axis + (out_rank - batch.len());
        let i = if batch[axis] == 1 { 0 } else { bidx[out_axis] };
        off += batch_strides[axis] * i as i64;
    }
    off as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    // --- #1056: MatMulPrepack::dense governance ---------------------------

    /// A single-`MatMul` graph whose `B` (input index 1) is a `[k, n]`
    /// initializer of `dtype` when `constant`, else a plain activation input,
    /// for driving [`matmul_dense_cache_predicted_bytes`].
    fn dense_matmul_graph(k: usize, n: usize, dtype: DataType, constant: bool) -> Graph {
        use onnx_runtime_ir::{NodeId, TensorData, WeightRef, static_shape};
        let mut graph = Graph::new();
        let a = graph.create_named_value("A", DataType::Float32, static_shape([1, k]));
        let b = graph.create_named_value("B", dtype, static_shape([k, n]));
        let y = graph.create_named_value("Y", DataType::Float32, static_shape([1, n]));
        graph.add_input(a);
        graph.add_input(b);
        let node = Node::new(NodeId(0), "MatMul", vec![Some(a), Some(b)], vec![y]);
        graph.insert_node(node);
        graph.add_output(y);
        if constant {
            let elem = dtype.byte_size();
            graph.set_initializer(
                b,
                WeightRef::Inline(TensorData::from_raw(
                    dtype,
                    vec![k, n],
                    vec![0u8; k * n * elem],
                )),
            );
        }
        graph
    }

    /// The predictor counts a constant operand iff it widens to an owned f32
    /// copy: f16/bf16 constants cost `4*k*n` per instantiation and the
    /// shape-keyed kernel cache holds [`MATMUL_DENSE_DECODE_INSTANTIATIONS`] of
    /// them (prefill + decode), a contiguous f32 constant is borrowed zero-copy
    /// (0), and a non-constant operand is never cached (0).
    #[test]
    fn dense_cache_predictor_counts_only_constant_non_f32_operands() {
        let (k, n) = (10usize, 7usize);
        let bytes = (k * n * 4) as u64 * MATMUL_DENSE_DECODE_INSTANTIATIONS;
        assert_eq!(
            matmul_dense_cache_predicted_bytes(&dense_matmul_graph(k, n, DataType::Float16, true)),
            bytes,
            "f16 constant widens to a 4*k*n f32 copy"
        );
        assert_eq!(
            matmul_dense_cache_predicted_bytes(&dense_matmul_graph(k, n, DataType::BFloat16, true)),
            bytes,
            "bf16 constant widens to a 4*k*n f32 copy"
        );
        assert_eq!(
            matmul_dense_cache_predicted_bytes(&dense_matmul_graph(k, n, DataType::Float32, true)),
            0,
            "a contiguous f32 constant is borrowed zero-copy, nothing cached"
        );
        assert_eq!(
            matmul_dense_cache_predicted_bytes(&dense_matmul_graph(k, n, DataType::Float16, false)),
            0,
            "a non-constant operand is never memoised as a weight"
        );
    }

    /// #1056 acceptance criterion 1: the predicted dense-cache bytes must equal
    /// the bytes actually held after a real run, ratio 1.00 — proven in one run
    /// by comparing the plan's graph prediction against the *summed*
    /// [`MatMulPrepack::dense_live_bytes`] of every kernel instantiation.
    ///
    /// The operand is a constant **f16** `B` with an **f32** `A`: `try_matmul_half`
    /// only fires when both operands share a half dtype, so this deterministically
    /// takes the generic f32 path that widens `B` through `dense(1)`, genuinely
    /// populating the cache (unlike the int4 / same-half models, which never do).
    ///
    /// Crucially it drives **two** instantiations — a prefill (`m > 1`) and a
    /// decode (`m == 1`) — the way the shape-keyed kernel cache does, each a
    /// separate `MatMulKernel` with its own `dense`. Both retain a copy, so the
    /// live total is [`MATMUL_DENSE_DECODE_INSTANTIATIONS`] × `4*k*n`, and the
    /// predictor must account for that multiplier or this ratio breaks. A test
    /// that drove a single instantiation could not catch a dropped multiplier —
    /// which is exactly #1051's defect.
    #[test]
    fn predicted_dense_bytes_equal_actual_after_matmul_execution() {
        // Thread-local admit so the process-global the concurrent tests read is
        // never touched (#1056 isolation). The verdict is read at construction,
        // so each kernel is built inside the scope.
        let _admit = DenseCacheEnabledScope::new(true);

        let (k, n) = (48usize, 33usize);
        let graph = dense_matmul_graph(k, n, DataType::Float16, true);
        let predicted = matmul_dense_cache_predicted_bytes(&graph);
        // The prediction budgets for both shape-keyed instantiations.
        assert_eq!(
            predicted,
            (k as u64) * (n as u64) * 4 * MATMUL_DENSE_DECODE_INSTANTIATIONS
        );

        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.013).cos()).collect();
        let b = Owned::f16(&[k, n], &b_data);

        // Mirror the executor's shape-keyed kernel cache: the SAME node compiled
        // once per resolved activation shape (prefill m>1, decode m==1) yields a
        // SEPARATE `MatMulKernel`/`MatMulPrepack`, each widening `B` into its own
        // `dense`. We instantiate both and sum their live bytes.
        let mut instances = 0u64;
        let mut actual = 0u64;
        for m in [4usize, 1usize] {
            let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).sin()).collect();
            let a = Owned::f32(&[m, k], &a_data);
            let mut out = Owned::zeros_f32(&[m, n]);
            let mut kernel = MatMulKernel::default();
            kernel.set_constant_inputs(&[false, true]);
            kernel
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();
            assert!(
                kernel.prepack.dense[1].is_filled(),
                "the constant f16 B must be widened into the governed cache at m={m}"
            );
            assert!(
                !kernel.prepack.dense[0].is_filled(),
                "the contiguous f32 A is borrowed, never cached (m={m})"
            );
            actual += kernel.prepack.dense_live_bytes();
            instances += 1;
        }

        // Empirically: both the prefill and decode instances retain a copy.
        assert_eq!(
            instances, MATMUL_DENSE_DECODE_INSTANTIATIONS,
            "prefill + decode = two shape-keyed instantiations"
        );
        assert_eq!(
            actual, predicted,
            "predicted dense-cache bytes must equal the summed bytes actually \
             held across all instantiations (ratio 1.00)"
        );
    }

    /// #1056 decline contract: when the cache is declined, a constant operand
    /// retains **nothing** (widened per call and freed), and the result is
    /// byte-identical to the admitted run — a pure performance tradeoff, never a
    /// numerical one.
    #[test]
    fn declined_dense_cache_retains_nothing_and_is_byte_identical() {
        let (m, k, n) = (4usize, 40usize, 24usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.011).cos()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.019).sin()).collect();

        let run = |enabled: bool| -> (Vec<f32>, u64) {
            // Thread-local scope; the kernel reads the verdict at construction.
            let _scope = DenseCacheEnabledScope::new(enabled);
            let a = Owned::f32(&[m, k], &a_data);
            let b = Owned::f16(&[k, n], &b_data);
            let mut out = Owned::zeros_f32(&[m, n]);
            let mut kernel = MatMulKernel::default();
            kernel.set_constant_inputs(&[false, true]);
            kernel
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();
            (out.to_f32(), kernel.prepack.dense_live_bytes())
        };

        let (admitted_out, admitted_bytes) = run(true);
        let (declined_out, declined_bytes) = run(false);

        assert_eq!(
            admitted_bytes,
            (k * n * 4) as u64,
            "admitted holds exactly one 4*k*n f32 copy"
        );
        assert_eq!(declined_bytes, 0, "declined retains nothing resident");
        assert_eq!(
            admitted_out, declined_out,
            "output must be byte-identical whether the cache is admitted or declined"
        );
    }

    #[test]
    fn matmul_zero_batch_returns_empty_without_panicking() {
        // Regression: a zero-sized batch dim (broadcast to a 0-length result)
        // used to run the compute loop once and index empty operand slices,
        // panicking. It must return an empty buffer instead (numpy/ONNX
        // reference semantics).
        let a = Owned::f32(&[0, 1, 1], &[]);
        let b = Owned::f32(&[0, 1, 1], &[]);
        let out = matmul_dense(&a.view(), &b.view()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn matmul_2x3_times_3x2() {
        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        // C = [[58,64],[139,154]]
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![58., 64., 139., 154.]);
    }

    #[cfg(feature = "tracing")]
    #[test]
    fn matmul_populates_active_trace_span_metrics() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        let (trace, events) = onnx_runtime_tracer::TraceContext::in_memory();
        {
            let _span = trace.span("MatMul", "compute");
            MatMulKernel::default()
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();
        }

        let events = events.events();
        let args = events[0].args.as_ref().expect("MatMul trace args");
        assert_eq!(args["bytes"], 64);
        assert_eq!(args["flops"], 24);
    }

    // ── #845: weight-transpose cache identity ────────────────────────────
    //
    // These run on every target: the memos they exercise are compiled
    // everywhere even though only the Apple Accelerate kernels consume them.
    // The defect they cover reached CI precisely because the affected code
    // could not be tested off macOS.

    /// Reference transpose, independent of the production tiled/parallel code.
    fn reference_transpose(src: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * k];
        for j in 0..n {
            for i in 0..k {
                out[j * k + i] = src[i * n + j];
            }
        }
        out
    }

    fn constant_b_prepack() -> MatMulPrepack {
        let mut prepack = MatMulPrepack::default();
        prepack.set_constant_inputs(&[false, true]);
        prepack
    }

    /// The exact CI failure of #845: two independent kernels, one recycled
    /// buffer address, incompatible shapes. Kernel A caches the transpose of a
    /// `[k1, n1]` weight; kernel B — a *fresh* prepack, so its own memo is
    /// empty — then asks for `[k2, n2]` at the same address and must never be
    /// handed A's transpose.
    ///
    /// Grow (`n2 * k2` larger than the stale entry) is the release-mode
    /// out-of-bounds read; shrink is the silently-wrong-logits variant. Both
    /// orderings are covered.
    #[test]
    fn transposed_b_at_a_reused_address_never_serves_another_tensors_transpose() {
        // One live allocation stands in for the recycled address: the two
        // logical tensors below share a base pointer by construction, so the
        // collision is deterministic instead of allocator-dependent.
        let mut storage = vec![0.0f32; 96 * 64];
        for (i, v) in storage.iter_mut().enumerate() {
            *v = i as f32 * 0.5;
        }

        for &((k1, n1), (k2, n2)) in &[
            ((4usize, 6usize), (96usize, 64usize)), // grow: stale entry too short
            ((96, 64), (4, 6)),                     // shrink: stale entry too long
        ] {
            let first = constant_b_prepack();
            let bt1 = first
                .transposed_b(&storage[..k1 * n1], k1, n1)
                .expect("constant B of matching length must be transposed");
            assert_eq!(bt1.len(), n1 * k1);
            assert_eq!(bt1, reference_transpose(&storage[..k1 * n1], k1, n1));

            let second = constant_b_prepack();
            let bt2 = second
                .transposed_b(&storage[..k2 * n2], k2, n2)
                .expect("a second kernel at the same address must still be served");
            assert_eq!(
                bt2.len(),
                n2 * k2,
                "[{k1},{n1}] then [{k2},{n2}] at one address: served a {}-element \
                 transpose where {} are indexed — release builds read past the end",
                bt2.len(),
                n2 * k2
            );
            assert_eq!(
                bt2,
                reference_transpose(&storage[..k2 * n2], k2, n2),
                "[{k1},{n1}] then [{k2},{n2}] at one address: served another tensor's data"
            );
        }
    }

    /// Two kernels that legitimately share one weight (same address, same
    /// shape) still share one allocation — the global cache must not be
    /// defeated by the wider key.
    #[test]
    fn transposed_b_is_shared_across_kernels_for_the_same_weight() {
        let mut storage = vec![0.0f32; 8 * 12];
        for (i, v) in storage.iter_mut().enumerate() {
            *v = i as f32 * 0.25;
        }
        let first = constant_b_prepack();
        let second = constant_b_prepack();
        let a = first.transposed_b(&storage, 8, 12).expect("first");
        let b = second.transposed_b(&storage, 8, 12).expect("second");
        assert_eq!(
            a.as_ptr(),
            b.as_ptr(),
            "a second kernel on the same weight must reuse the cached transpose, \
             not recompute it"
        );
    }

    /// Geometry that disagrees with the operand fails closed instead of
    /// transposing (or serving) out of bounds.
    #[test]
    fn transposed_b_rejects_geometry_that_does_not_match_the_operand() {
        let prepack = constant_b_prepack();
        let b = vec![1.0f32; 12];
        assert!(prepack.transposed_b(&b, 3, 5).is_none(), "too long");
        assert!(prepack.transposed_b(&b, 2, 5).is_none(), "too short");
        assert!(
            prepack.transposed_b(&b, usize::MAX, 3).is_none(),
            "k * n overflow must fail closed"
        );
        assert!(
            prepack.transposed_b(&b, 3, 4).is_some(),
            "matching geometry"
        );

        // Non-constant B is never transposed.
        let activation = MatMulPrepack::default();
        assert!(activation.transposed_b(&b, 3, 4).is_none());
    }

    /// A prepack whose memo was filled for one geometry never serves it for
    /// another, whatever the reason the geometry changed.
    #[test]
    fn transposed_b_memo_is_validated_against_the_current_call() {
        let mut storage = vec![0.0f32; 24];
        for (i, v) in storage.iter_mut().enumerate() {
            *v = i as f32;
        }
        let prepack = constant_b_prepack();
        let first = prepack.transposed_b(&storage[..12], 3, 4).expect("first");
        assert_eq!(first.len(), 12);

        // Same prepack, same base address, different shape: the memo must not
        // be served. Falling back to `None` (the untransposed kernel) is
        // correct; serving a wrong-length slice is not.
        match prepack.transposed_b(&storage, 4, 6) {
            None => {}
            Some(bt) => {
                assert_eq!(bt.len(), 24, "memo served a wrong-length transpose");
                assert_eq!(bt, reference_transpose(&storage, 4, 6));
            }
        }

        // The original geometry keeps hitting the memo.
        let again = prepack.transposed_b(&storage[..12], 3, 4).expect("again");
        assert_eq!(again.as_ptr(), first.as_ptr());
    }

    /// Zero-element weights are handled without panicking and without growing
    /// the cache.
    #[test]
    fn transposed_b_handles_zero_sized_weights() {
        let empty: Vec<f32> = Vec::new();
        for (k, n) in [(0usize, 0usize), (0, 8), (8, 0)] {
            let prepack = constant_b_prepack();
            let bt = prepack
                .transposed_b(&empty, k, n)
                .unwrap_or_else(|| panic!("zero-size [{k}, {n}] must be served"));
            assert!(bt.is_empty());
        }
        // One prepack reached with several zero geometries keeps its first memo
        // and declines the rest — never a non-empty or wrong-shaped slice.
        let prepack = constant_b_prepack();
        assert_eq!(prepack.transposed_b(&empty, 0, 0), Some(&[][..]));
        for (k, n) in [(0usize, 8usize), (8, 0)] {
            match prepack.transposed_b(&empty, k, n) {
                None => {}
                Some(bt) => assert!(bt.is_empty()),
            }
        }
    }

    /// The f16 memo enforces the same identity, and rejects a `k`/`n` pair that
    /// does not match the view — which would otherwise build a slice longer
    /// than the weight and read out of bounds.
    #[test]
    fn transposed_b_f16_validates_the_view_geometry() {
        let bits: Vec<u16> = (0..24u16).map(|i| 0x3C00 + i).collect();
        let owned = Owned::f16_bits(&[4, 6], &bits);
        let prepack = constant_b_prepack();

        assert!(
            prepack.transposed_b_f16(&owned.view(), 4, 7).is_none(),
            "k * n larger than the view must fail closed"
        );
        assert!(
            prepack.transposed_b_f16(&owned.view(), 2, 6).is_none(),
            "k * n smaller than the view must fail closed"
        );

        let bt = prepack
            .transposed_b_f16(&owned.view(), 4, 6)
            .expect("matching geometry");
        assert_eq!(bt.len(), 24);
        for j in 0..6 {
            for i in 0..4 {
                assert_eq!(bt[j * 4 + i], bits[i * 6 + j], "B_T[{j},{i}]");
            }
        }

        // Same buffer address, different shape: never a wrong-length slice.
        let reshaped = owned.with_view(&[3, 8], &[8, 1]);
        match prepack.transposed_b_f16(&reshaped.view(), 3, 8) {
            None => {}
            Some(other) => assert_eq!(other.len(), 24),
        }
    }

    /// f16 and f32 weights that happen to share an address stay in separate
    /// caches, so no cross-dtype collision is possible.
    #[test]
    fn f16_and_f32_transposes_do_not_collide() {
        let f32_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bits: Vec<u16> = (0..6u16).map(|i| 0x3C00 + i).collect();
        let owned = Owned::f16_bits(&[2, 3], &bits);
        let prepack = constant_b_prepack();

        let f32_bt = prepack.transposed_b(&f32_data, 2, 3).expect("f32");
        assert_eq!(f32_bt, reference_transpose(&f32_data, 2, 3));

        let f16_prepack = constant_b_prepack();
        let f16_bt = f16_prepack
            .transposed_b_f16(&owned.view(), 2, 3)
            .expect("f16");
        assert_eq!(
            f16_bt,
            [bits[0], bits[3], bits[1], bits[4], bits[2], bits[5]]
        );
    }

    #[test]
    fn matmul_with_transposed_b_view() {
        // B stored as [2,3] row-major, exposed transposed as [3,2] strides [1,3].
        // A[2,3] @ Bt[3,2] where Bt = B.T.
        // B = [[7,9,11],[8,10,12]] stored; Bt = [[7,8],[9,10],[11,12]].
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[2, 3], &[7., 9., 11., 8., 10., 12.]).with_view(&[3, 2], &[1, 3]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // Same as the contiguous case above.
        assert_eq!(out.to_f32(), vec![58., 64., 139., 154.]);
    }

    #[test]
    fn matmul_batched() {
        // Two independent [2,2] matmuls.
        let a = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::f32(&[2, 2, 2], &[1., 0., 0., 1., 2., 0., 0., 2.]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // batch0: A@I = A; batch1: [[5,6],[7,8]]*2 = [[10,12],[14,16]]
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 10., 12., 14., 16.]);
    }

    #[test]
    fn matmul_broadcast_batch() {
        // A [2,2,2] @ B [2,2] (broadcast B over batch)
        let a = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::f32(&[2, 2], &[1., 0., 0., 1.]); // identity
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn matmul_vector_times_matrix() {
        // a [3] @ B [3,2] -> [2]
        let a = Owned::f32(&[3], &[1., 2., 3.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        assert_eq!(out.to_f32(), vec![58., 64.]);
    }

    #[test]
    fn matmul_f16_accumulates_in_f32() {
        // A[2,3] @ B[3,2] in f16; compute widens to f32, result rounds to f16.
        let a = Owned::f16(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f16(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Float16, &[2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f16_as_f32(), vec![58., 64., 139., 154.]);
    }

    /// Falsifier for the [`HALF_WIDEN_MIN_M`] gate: asserts *which route* each
    /// shape takes, not merely that the numbers are right. Both routes produce
    /// near-identical results, so a value-only test cannot detect the gate
    /// being mis-wired, inverted, or silently dead.
    #[cfg(all(target_arch = "x86_64", feature = "mlas"))]
    #[test]
    fn f16_prefill_yields_to_widened_sgemm_only_above_the_measured_crossover() {
        let backend = CpuBackend::auto_detect();
        assert_eq!(
            backend,
            CpuBackend::Mlas,
            "x86_64 + mlas must auto-detect Mlas, else this gate is dead code"
        );

        // Pin the constants to the numbers their doc tables were measured at.
        // Everything else here follows the constants, so without this a silent
        // retune would move the dispatch boundary while every test still
        // passed. If this fails, re-run `bench_f16_half_vs_widen` and update
        // the tables rather than just bumping the expectation.
        assert_eq!(
            HALF_WIDEN_MIN_M, 16,
            "M threshold moved off its measurement"
        );
        assert_eq!(
            HALF_WIDEN_MIN_WEIGHT, 256,
            "weight threshold moved off its measurement"
        );

        // Big enough that HALF_WIDEN_MIN_WEIGHT is satisfied, so M alone decides.
        let (k, n) = (512usize, 512usize);
        assert!(k * n >= HALF_WIDEN_MIN_WEIGHT);

        for (m, expect_widen) in [
            (1usize, false),
            (HALF_WIDEN_MIN_M - 1, false),
            (HALF_WIDEN_MIN_M, true),
            (HALF_WIDEN_MIN_M + 1, true),
        ] {
            let a_src: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.01).collect();
            let b_src: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.01).collect();
            let a = Owned::f16(&[m, k], &a_src);
            let b = Owned::f16(&[k, n], &b_src);
            let geom = matmul_geometry(&a.view(), &b.view()).unwrap();

            reset_half_yielded_to_widened_calls();
            let took_half = try_matmul_half(&a.view(), &b.view(), &geom, backend)
                .unwrap()
                .is_some();
            assert_eq!(
                took_half, !expect_widen,
                "M={m}: expected widen={expect_widen}, but half GEMM ran={took_half}"
            );
            assert_eq!(
                half_yielded_to_widened_calls(),
                u64::from(expect_widen),
                "M={m}: yield counter disagrees with the taken route"
            );
        }

        // bf16 is deliberately excluded: its crossover was never measured.
        let m = HALF_WIDEN_MIN_M + 8;
        let a_src: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.01).collect();
        let b_src: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.01).collect();
        let a = Owned::bf16(&[m, k], &a_src);
        let b = Owned::bf16(&[k, n], &b_src);
        let geom = matmul_geometry(&a.view(), &b.view()).unwrap();
        reset_half_yielded_to_widened_calls();
        assert!(
            try_matmul_half(&a.view(), &b.view(), &geom, backend)
                .unwrap()
                .is_some(),
            "bf16 must stay on the blocked half GEMM"
        );
        assert_eq!(half_yielded_to_widened_calls(), 0);

        // A non-Mlas backend keeps today's behaviour at every M.
        let a = Owned::f16(&[m, k], &a_src);
        let b = Owned::f16(&[k, n], &b_src);
        let geom = matmul_geometry(&a.view(), &b.view()).unwrap();
        reset_half_yielded_to_widened_calls();
        assert!(
            try_matmul_half(&a.view(), &b.view(), &geom, CpuBackend::Generic)
                .unwrap()
                .is_some(),
            "a backend with no tuned SGEMM must keep the half GEMM"
        );
        assert_eq!(half_yielded_to_widened_calls(), 0);

        // A small weight stays on the half path however large M is: the tuned
        // SGEMM has too little work to repay widening B.
        let (k, n) = (8usize, 8usize);
        assert!(k * n < HALF_WIDEN_MIN_WEIGHT);
        let a_src: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.01).collect();
        let b_src: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.01).collect();
        let a = Owned::f16(&[m, k], &a_src);
        let b = Owned::f16(&[k, n], &b_src);
        let geom = matmul_geometry(&a.view(), &b.view()).unwrap();
        reset_half_yielded_to_widened_calls();
        assert!(
            try_matmul_half(&a.view(), &b.view(), &geom, backend)
                .unwrap()
                .is_some(),
            "a small B must keep the half GEMM"
        );
        assert_eq!(half_yielded_to_widened_calls(), 0);
    }

    /// The widened route must stay numerically sound across the crossover.
    /// Widening `f16` to `f32` is exact, so both routes see identical inputs
    /// and differ only in `f32` summation order; each is compared against an
    /// `f64` oracle so neither route can drift without being caught.
    #[cfg(all(target_arch = "x86_64", feature = "mlas"))]
    #[test]
    fn f16_widened_route_matches_an_f64_oracle_across_the_crossover() {
        let backend = CpuBackend::auto_detect();
        let (k, n) = (521usize, 517usize);
        assert!(k * n >= HALF_WIDEN_MIN_WEIGHT, "shape must cross the gate");

        let mut state = 0x0F16_51DEu32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.5
        };

        // Straddle the threshold, including odd M and a tail-heavy K/N.
        for m in [HALF_WIDEN_MIN_M - 1, HALF_WIDEN_MIN_M, HALF_WIDEN_MIN_M + 3] {
            let a_src: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let b_src: Vec<f32> = (0..k * n).map(|_| next()).collect();
            let a = Owned::f16(&[m, k], &a_src);
            let b = Owned::f16(&[k, n], &b_src);

            // Oracle over the *rounded* f16 values, accumulated in f64.
            let a_w: Vec<f64> = a_src
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32() as f64)
                .collect();
            let b_w: Vec<f64> = b_src
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32() as f64)
                .collect();
            let mut oracle = vec![0.0f64; m * n];
            for row in 0..m {
                for depth in 0..k {
                    let av = a_w[row * k + depth];
                    for column in 0..n {
                        oracle[row * n + column] += av * b_w[depth * n + column];
                    }
                }
            }

            let geom = matmul_geometry(&a.view(), &b.view()).unwrap();
            let got = matmul_dense(&a.view(), &b.view()).unwrap();
            assert_eq!(got.len(), m * n);
            // Confirm this shape really exercised the widened route.
            reset_half_yielded_to_widened_calls();
            assert_eq!(
                try_matmul_half(&a.view(), &b.view(), &geom, backend)
                    .unwrap()
                    .is_some(),
                m < HALF_WIDEN_MIN_M
            );

            let max_error = got
                .iter()
                .zip(oracle.iter())
                .map(|(&actual, &expected)| (actual - expected as f32).abs())
                .fold(0.0f32, f32::max);
            // Measured max error here is 5.4e-7 / 1.2e-6 / 1.8e-6 for
            // M=15/16/19, i.e. pure f32 accumulation noise over K=521 terms.
            // 1e-4 leaves ~55x headroom for a different SGEMM tile order while
            // staying far below the ~1e-2 a genuine defect would produce (f16
            // accumulation, a dropped tail, a wrong tile) -- f16 inputs are
            // only granular to ~1e-3, so anything real is orders of magnitude
            // above this bound.
            const MAX_ACCUMULATION_ERROR: f32 = 1e-4;
            assert!(
                max_error <= MAX_ACCUMULATION_ERROR,
                "M={m}: max error {max_error:e} exceeds {MAX_ACCUMULATION_ERROR:e}"
            );
        }
    }

    #[test]
    fn matmul_half_dispatch_matches_widened_reference_across_irregular_shapes() {
        use onnx_runtime_ir::DataType;

        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 127, 65),
            (3, 5, 7),
            (17, 130, 11),
            (5, 257, 2),
            (2, 0, 3),
            // Exactly on each widening threshold, and one step below each.
            (16, 16, 16),
            (15, 16, 16),
            (16, 15, 17),
        ];
        let mut state = 0x51A7_CAFE_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.5
        };

        for dtype in [DataType::Float16, DataType::BFloat16] {
            for &(m, k, n) in SHAPES {
                let a_source: Vec<f32> = (0..m * k).map(|_| next()).collect();
                let b_source: Vec<f32> = (0..k * n).map(|_| next()).collect();
                let round = |value: f32| match dtype {
                    DataType::Float16 => half::f16::from_f32(value).to_f32(),
                    DataType::BFloat16 => half::bf16::from_f32(value).to_f32(),
                    _ => unreachable!(),
                };
                let a_wide: Vec<f32> = a_source.iter().copied().map(round).collect();
                let b_wide: Vec<f32> = b_source.iter().copied().map(round).collect();
                let a = match dtype {
                    DataType::Float16 => Owned::f16(&[m, k], &a_source),
                    DataType::BFloat16 => Owned::bf16(&[m, k], &a_source),
                    _ => unreachable!(),
                };
                let b = match dtype {
                    DataType::Float16 => Owned::f16(&[k, n], &b_source),
                    DataType::BFloat16 => Owned::bf16(&[k, n], &b_source),
                    _ => unreachable!(),
                };
                let geometry = matmul_geometry(&a.view(), &b.view()).unwrap();
                // f16 above the widening crossover deliberately declines here
                // and is served by the tuned SGEMM instead; the value checks
                // below are route-independent and cover both.
                let took_half =
                    try_matmul_half(&a.view(), &b.view(), &geometry, CpuBackend::auto_detect())
                        .unwrap()
                        .is_some();
                #[cfg(all(target_arch = "x86_64", feature = "mlas"))]
                let expect_widen = dtype == DataType::Float16
                    && m >= HALF_WIDEN_MIN_M
                    && k * n >= HALF_WIDEN_MIN_WEIGHT;
                #[cfg(not(all(target_arch = "x86_64", feature = "mlas")))]
                let expect_widen = false;
                assert_eq!(
                    took_half, !expect_widen,
                    "{dtype:?} {m}x{k}x{n}: unexpected route (half={took_half})"
                );

                let mut expected = vec![0.0f32; m * n];
                for row in 0..m {
                    for column in 0..n {
                        for depth in 0..k {
                            expected[row * n + column] +=
                                a_wide[row * k + depth] * b_wide[depth * n + column];
                        }
                        expected[row * n + column] = round(expected[row * n + column]);
                    }
                }

                let mut first = Owned::zeros(dtype, &[m, n]);
                let mut second = Owned::zeros(dtype, &[m, n]);
                MatMulKernel::default()
                    .execute(&[a.view(), b.view()], &mut [first.view_mut()])
                    .unwrap();
                MatMulKernel::default()
                    .execute(&[a.view(), b.view()], &mut [second.view_mut()])
                    .unwrap();
                let first = match dtype {
                    DataType::Float16 => first.to_f16_as_f32(),
                    DataType::BFloat16 => first.to_bf16_as_f32(),
                    _ => unreachable!(),
                };
                let second = match dtype {
                    DataType::Float16 => second.to_f16_as_f32(),
                    DataType::BFloat16 => second.to_bf16_as_f32(),
                    _ => unreachable!(),
                };
                assert_eq!(first, second, "{dtype:?} {m}x{k}x{n} was not deterministic");

                let tolerance = match dtype {
                    DataType::Float16 => 2e-3,
                    DataType::BFloat16 => 3e-2,
                    _ => unreachable!(),
                };
                let max_error = first
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_error <= tolerance,
                    "{dtype:?} {m}x{k}x{n}: max error {max_error} exceeds {tolerance}"
                );
            }
        }
    }

    #[test]
    fn matmul_f16_preserves_near_tie_argmax_after_f32_accumulation() {
        // The f32 reference margin (0.01171875) is below one f16 ULP at 20.
        // Widened accumulation must still retain the correct first-column winner
        // before the single final f16 narrowing.
        let a = Owned::f16(&[1, 4], &[1., 1., 1., 1.]);
        // Exact f16 representable input used to exercise the near-tie boundary.
        #[allow(clippy::excessive_precision)]
        let b = Owned::f16(
            &[4, 2],
            &[5.00390625, 5.0, 5.00390625, 5.0, 5.00390625, 5.0, 5.0, 5.0],
        );
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Float16, &[1, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f16_as_f32(), vec![20.015625, 20.0]);
    }

    #[test]
    fn matmul_bf16_batched() {
        let a = Owned::bf16(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::bf16(&[2, 2, 2], &[1., 0., 0., 1., 2., 0., 0., 2.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_bf16_as_f32(),
            vec![1., 2., 3., 4., 10., 12., 14., 16.]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn matmul_bf16_native_matches_f64_reference_within_bf16_tolerance() {
        // Native `_mm512_dpbf16_ps` GEMM must (a) match an f64 reference over the
        // SAME bf16-rounded operands within bf16 tolerance, and (b) be no worse
        // than the widen-to-f32 upcast path — bf16-input products are exact in
        // f32, so both paths differ from f64 only by f32 summation rounding.
        if !x86_bf16::native_available() {
            eprintln!("skipping: host lacks avx512_bf16");
            return;
        }
        // Shapes exercise m=1 decode, general prefill, and K/N-32 tails.
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 2048, 512), // decode GEMV
            (1, 100, 40),   // decode, K & N not multiples of 32
            (32, 256, 64),  // prefill, aligned
            (17, 130, 50),  // prefill, ragged M/K/N
            (4, 33, 3),     // tiny K tail (33 = 32 + 1)
        ];

        let mut state = 0x9E37_79B9_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0 // ~[-1, 1]
        };

        let mut worst_native_rel = 0.0f64;
        let mut worst_ratio = 0.0f64;
        for &(m, k, n) in SHAPES {
            let a_f32: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let b_f32: Vec<f32> = (0..k * n).map(|_| next()).collect();

            // Round operands to bf16; both compute paths and the reference share
            // these exact bf16 values so the only difference is accumulation.
            let a_bf: Vec<half::bf16> = a_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let b_bf: Vec<half::bf16> = b_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let a_bits: Vec<u16> = a_bf.iter().map(|v| v.to_bits()).collect();
            let b_bits: Vec<u16> = b_bf.iter().map(|v| v.to_bits()).collect();
            let a_wide: Vec<f32> = a_bf.iter().map(|v| v.to_f32()).collect();
            let b_wide: Vec<f32> = b_bf.iter().map(|v| v.to_f32()).collect();

            // f64 reference over the bf16-rounded values.
            let mut reference = vec![0.0f64; m * n];
            // Upcast reference: widen bf16 -> f32, accumulate in f32.
            let mut upcast = vec![0.0f32; m * n];
            for row in 0..m {
                for col in 0..n {
                    let mut acc64 = 0.0f64;
                    let mut acc32 = 0.0f32;
                    for depth in 0..k {
                        acc64 += a_wide[row * k + depth] as f64 * b_wide[depth * n + col] as f64;
                        acc32 += a_wide[row * k + depth] * b_wide[depth * n + col];
                    }
                    reference[row * n + col] = acc64;
                    upcast[row * n + col] = acc32;
                }
            }

            // Native bf16 path.
            let mut native = vec![0.0f32; m * n];
            x86_bf16::gemm(&a_bits, &b_bits, &mut native, m, k, n);

            let rel = |got: f32, want: f64| -> f64 {
                let denom = want.abs().max(1.0);
                (got as f64 - want).abs() / denom
            };
            let mut max_native = 0.0f64;
            let mut max_upcast = 0.0f64;
            for idx in 0..m * n {
                max_native = max_native.max(rel(native[idx], reference[idx]));
                max_upcast = max_upcast.max(rel(upcast[idx], reference[idx]));
            }
            worst_native_rel = worst_native_rel.max(max_native);
            // Native must not be materially worse than the upcast reference.
            let ratio = max_native / max_upcast.max(1e-9);
            worst_ratio = worst_ratio.max(ratio);
            assert!(
                max_native <= max_upcast * 4.0 + 1e-4,
                "{m}x{k}@{k}x{n}: native rel {max_native} worse than upcast {max_upcast}"
            );
            // bf16 accumulation over K stays within a loose bf16 tolerance.
            assert!(
                max_native <= 5e-2,
                "{m}x{k}@{k}x{n}: native rel {max_native} exceeds bf16 tolerance"
            );
        }
        println!(
            "native bf16 GEMM: worst native-vs-f64 rel {worst_native_rel:.3e}, \
             worst native/upcast ratio {worst_ratio:.3}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn matmul_bf16_native_handles_k_tail_and_kernel_matches_kernel_path() {
        // The public MatMul kernel (which auto-routes to the native path on this
        // host) must produce a bf16 result matching a direct f64 reference for a
        // K that is not a multiple of the 32-lane bf16 width.
        if !x86_bf16::native_available() {
            eprintln!("skipping: host lacks avx512_bf16");
            return;
        }
        let (m, k, n) = (3usize, 70usize, 5usize); // 70 = 64 + 6 tail
        let mut state = 0x1357_9BDF_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 1.0
        };
        let a_f32: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|_| next()).collect();
        let a = Owned::bf16(&[m, k], &a_f32);
        let b = Owned::bf16(&[k, n], &b_f32);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[m, n]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();

        // f64 reference over bf16-rounded operands, then rounded to bf16 output.
        let a_w: Vec<f32> = a_f32
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let b_w: Vec<f32> = b_f32
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let got = out.to_bf16_as_f32();
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for depth in 0..k {
                    acc += a_w[row * k + depth] as f64 * b_w[depth * n + col] as f64;
                }
                let want = half::bf16::from_f32(acc as f32).to_f32();
                let denom = want.abs().max(1.0);
                let rel = (got[row * n + col] - want).abs() / denom;
                assert!(rel <= 3e-2, "K-tail mismatch at ({row},{col}): rel {rel}");
            }
        }
    }

    /// Sweep that sets [`HALF_PREFILL_GEBP_MIN_WEIGHT`]: blocked half GEMM vs
    /// the fused widen-pack GEBP, interleaved rep-by-rep so both routes see the
    /// same machine load. Both arms call the exact function the gate selects
    /// between, so the crossover this prints is the crossover the gate must
    /// encode -- and no environment variable is touched, which keeps it safe to
    /// run inside the parallel test binary.
    ///
    /// Pin the run (`taskset -c 0-15`) and set `RAYON_NUM_THREADS`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "microbench: run explicitly with --ignored --nocapture"]
    fn bench_half_prefill_gebp_crossover() {
        use std::time::Instant;

        fn median5(mut f: impl FnMut()) -> f64 {
            let mut v: Vec<f64> = (0..5)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v[2]
        }

        println!("threads={}", rayon::current_num_threads());
        println!("format,m,k,n,weight,blocked_ms,gebp_ms,blocked/gebp");
        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            for &(k, n) in &[
                (256usize, 256usize), //    65_536
                (512, 512),           //   262_144
                (768, 768),           //   589_824
                (1024, 1024),         // 1_048_576
                (1536, 1536),         // 2_359_296
                (2048, 2048),         // 4_194_304
            ] {
                let b: Vec<u16> = (0..k * n)
                    .map(|i| narrow_bits(format, ((i % 31) as f32 - 15.0) / 16.0))
                    .collect();
                for &m in &[1usize, 2, 4, 8, 16] {
                    let a: Vec<u16> = (0..m * k)
                        .map(|i| narrow_bits(format, ((i % 23) as f32 - 11.0) / 16.0))
                        .collect();
                    let mut c = vec![0.0f32; m * n];
                    let mut blocked = || {
                        half_gemm::gemm(
                            format,
                            &a,
                            MatrixLayout::row_major(k),
                            &b,
                            MatrixLayout::row_major(n),
                            &mut c,
                            m,
                            k,
                            n,
                        );
                    };
                    blocked();
                    let mut gebp_c = vec![0.0f32; m * n];
                    let mut gebp =
                        || x86_sgemm::half_prefill_gebp(format, &a, &b, &mut gebp_c, m, k, n);
                    gebp();

                    let blocked_ms = median5(&mut blocked);
                    let gebp_ms = median5(&mut gebp);
                    let weight = k * n;
                    println!(
                        "{format:?},{m},{k},{n},{weight},{blocked_ms:.4},{gebp_ms:.4},{:.2}",
                        blocked_ms / gebp_ms
                    );
                }
            }
        }
    }

    /// Round an `f32` into the given 16-bit format and return its bits.
    #[cfg(target_arch = "x86_64")]
    fn narrow_bits(format: HalfFormat, value: f32) -> u16 {
        match format {
            HalfFormat::F16 => half::f16::from_f32(value).to_bits(),
            HalfFormat::Bf16 => half::bf16::from_f32(value).to_bits(),
        }
    }

    /// Sweep that sets [`HALF_WIDEN_MIN_M`]: blocked half GEMM vs widening to
    /// `f32` and running the tuned SGEMM, over the `M` grid at the ambient
    /// thread count. Both arms run the exact code the two routes run, so the
    /// crossover this prints is the crossover the gate must encode.
    ///
    /// Pin the run (`taskset -c 0-15`) and set `RAYON_NUM_THREADS`; an
    /// oversubscribed pool inflates the parallel arm by ~2.6x on this host.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "microbench: run explicitly with --ignored --nocapture"]
    fn bench_f16_half_vs_widen() {
        use std::time::Instant;

        fn median5(mut f: impl FnMut()) -> f64 {
            f();
            let mut v: Vec<f64> = (0..5)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v[2]
        }

        let threads = rayon::current_num_threads();
        println!("threads={threads}");
        println!("M,K,N,half_ms,widen_ms,half/widen");
        for &(m, k, n) in &[
            (2usize, 2048usize, 2048usize),
            (8, 2048, 2048),
            (16, 2048, 2048),
            (32, 2048, 2048),
            (64, 2048, 2048),
            (128, 2048, 2048),
            (256, 2048, 2048),
            (128, 3584, 3584),
            // Weight sweep at the minimum claimed M, to site the weight guard.
            (16, 8, 8),
            (16, 16, 16),
            (16, 32, 32),
            (16, 48, 48),
            (16, 64, 64),
            (16, 96, 96),
            (16, 128, 128),
            (16, 130, 11),
            (16, 256, 256),
        ] {
            let a_src: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.01 - 0.06).collect();
            let b_src: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.01 - 0.05).collect();
            let a = Owned::f16(&[m, k], &a_src);
            let b = Owned::f16(&[k, n], &b_src);
            let (av, bv) = (a.view(), b.view());
            let geom = matmul_geometry(&av, &bv).unwrap();

            // Arm 1: the blocked half GEMM, forced by asking for a backend the
            // gate never yields for.
            let half = median5(|| {
                let out = try_matmul_half(&av, &bv, &geom, CpuBackend::Generic)
                    .unwrap()
                    .unwrap();
                std::hint::black_box(&out);
            });
            // Arm 2: what every caller falls through to once the gate declines.
            let widen = median5(|| {
                let out = matmul_dense_impl_with_geom(
                    to_dense_f32_widen("MatMul", &av).unwrap(),
                    to_dense_f32_widen("MatMul", &bv).unwrap(),
                    &geom,
                    CpuBackend::auto_detect(),
                    None,
                )
                .unwrap();
                std::hint::black_box(&out);
            });
            println!(
                "{m},{k},{n},{half:.4},{widen:.4},{:.2}",
                half / widen.max(f64::MIN_POSITIVE)
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "microbench: run explicitly with --ignored --nocapture"]
    fn bench_bf16_native_vs_upcast() {
        use std::time::Instant;
        if !x86_bf16::native_available() {
            eprintln!("skipping bench: host lacks avx512_bf16");
            return;
        }
        // Decode (m=1 GEMV) and prefill (MxKxN) shapes, LLM-representative.
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 4096, 4096),   // decode GEMV
            (1, 4096, 11008),  // decode MLP up-proj
            (128, 4096, 4096), // prefill attention proj
            (256, 2048, 8192), // prefill MLP
        ];
        let mut state = 0x5DEE_CE66_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
        };
        let median3 = |mut f: Box<dyn FnMut() -> f64>| {
            let mut t = [f(), f(), f()];
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[1]
        };

        println!("bf16 GEMM microbench (native _mm512_dpbf16_ps vs widen-to-f32 + SGEMM)");
        for &(m, k, n) in SHAPES {
            let a_bits: Vec<u16> = (0..m * k)
                .map(|_| half::bf16::from_f32(next()).to_bits())
                .collect();
            let b_bits: Vec<u16> = (0..k * n)
                .map(|_| half::bf16::from_f32(next()).to_bits())
                .collect();
            let flops = 2.0 * m as f64 * k as f64 * n as f64;

            // Native bf16 path.
            let native_ms = {
                let (a, b) = (a_bits.clone(), b_bits.clone());
                median3(Box::new(move || {
                    let mut c = vec![0.0f32; m * n];
                    let t = Instant::now();
                    x86_bf16::gemm(&a, &b, &mut c, m, k, n);
                    std::hint::black_box(&c);
                    t.elapsed().as_secs_f64() * 1e3
                }))
            };

            // Upcast reference: widen bf16 -> f32, then use the crate's f32
            // SGEMM.
            let upcast_ms = {
                let (a, b) = (a_bits.clone(), b_bits.clone());
                median3(Box::new(move || {
                    let t = Instant::now();
                    let a_f: Vec<f32> = a
                        .iter()
                        .map(|&x| half::bf16::from_bits(x).to_f32())
                        .collect();
                    let b_f: Vec<f32> = b
                        .iter()
                        .map(|&x| half::bf16::from_bits(x).to_f32())
                        .collect();
                    let mut c = vec![0.0f32; m * n];
                    gemm(&a_f, &b_f, &mut c, m, k, n).unwrap();
                    std::hint::black_box(&c);
                    t.elapsed().as_secs_f64() * 1e3
                }))
            };

            let g = |ms: f64| flops / (ms * 1e-3) / 1e9;
            println!(
                "  {m:>4}x{k}x{n}: native {native_ms:>8.3} ms ({:>7.1} GFLOP/s)  \
                 upcast {upcast_ms:>8.3} ms ({:>7.1} GFLOP/s)  speedup {:.2}x",
                g(native_ms),
                g(upcast_ms),
                upcast_ms / native_ms,
            );
        }
    }

    #[test]
    fn matmul_rejects_integer_dtype_with_rule1() {
        let a = Owned::i32(&[2, 2], &[1, 2, 3, 4]);
        let b = Owned::i32(&[2, 2], &[1, 0, 0, 1]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Int32, &[2, 2]);
        let err = MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap_err();
        assert!(format!("{err}").contains("WHAT"));
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn matmul_generic_block_boundaries_match_naive_reference() {
        const SHAPES: &[(usize, usize, usize)] = &[
            (65, 257, 70),
            (128, 300, 200),
            (100, 64, 4),
            (4, 256, 4),
            (1, 512, 1),
            (200, 1, 200),
        ];
        const ABS_TOLERANCE: f32 = 1e-3;

        let mut state = 0x1234_5678_u32;
        let mut next_f32 = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.25
        };

        let mut overall_max_abs_error = 0.0f32;
        for &(m, k, n) in SHAPES {
            let a_data: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();
            let b_data: Vec<f32> = (0..k * n).map(|_| next_f32()).collect();

            let mut reference = vec![0.0f32; m * n];
            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0f32;
                    for depth in 0..k {
                        sum += a_data[row * k + depth] * b_data[depth * n + col];
                    }
                    reference[row * n + col] = sum;
                }
            }

            let a = Owned::f32(&[m, k], &a_data);
            let b = Owned::f32(&[k, n], &b_data);
            let mut out = Owned::zeros_f32(&[m, n]);
            MatMulKernel::default()
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();

            let actual = out.to_f32();
            let max_abs_error = actual
                .iter()
                .zip(&reference)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            overall_max_abs_error = overall_max_abs_error.max(max_abs_error);
            assert!(
                max_abs_error <= ABS_TOLERANCE,
                "{m}x{k} @ {k}x{n}: max abs error {max_abs_error} exceeds {ABS_TOLERANCE}"
            );
        }

        println!("generic MatMul max abs error: {overall_max_abs_error}");
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn default_gemm_backend_matches_generic_reference() {
        // Locks the auto-detected default GEMM path: after flipping the f32
        // default to MLAS, `gemm()` (which dispatches on `CpuBackend::auto_detect`)
        // must still match the generic reference within f32 tolerance, so the
        // faster default never changes decode outputs (token-ID parity).
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 2048, 2048), // decode GEMV (m=1), the dense hot path
            (1, 2304, 9216), // GeGLU up/gate projection tile
            (1, 9216, 2304), // GeGLU down projection tile
            (5, 128, 256),
            (32, 512, 512),
        ];
        assert_eq!(CpuBackend::auto_detect(), CpuBackend::Mlas);
        let mut state = 0x1234_abcd_u32;
        let mut next_f32 = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.25
        };
        for &(m, k, n) in SHAPES {
            let a: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| next_f32()).collect();
            let mut expected = vec![0.0; m * n];
            let mut actual = vec![0.0; m * n];
            gemm_generic(&a, &b, &mut expected, m, k, n);
            gemm(&a, &b, &mut actual, m, k, n).unwrap();
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 1e-3,
                "{m}x{k} @ {k}x{n}: default-backend max error {max_error} exceeds tolerance"
            );
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_gemm_matches_generic_for_matrix_and_batched_vector_tiles() {
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 1, 1),
            (7, 13, 5),
            (32, 512, 512),
            (97, 11, 3),
            // Each tile below is how batched and vector MatMul route through gemm.
            (1, 13, 5),
            (3, 13, 1),
        ];
        let mut state = 0x5eed_1234_u32;
        let mut next_f32 = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.25
        };

        for &(m, k, n) in SHAPES {
            let a: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| next_f32()).collect();
            let mut expected = vec![0.0; m * n];
            let mut actual = vec![0.0; m * n];
            gemm_generic(&a, &b, &mut expected, m, k, n);
            gemm_with_backend(CpuBackend::Mlas, &a, &b, &mut actual, m, k, n).unwrap();
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 1e-3,
                "{m}x{k} @ {k}x{n}: MLAS max error {max_error} exceeds tolerance"
            );
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_constant_b_packed_kernel_matches_unpacked_and_generic() {
        for (m, k, n) in [(5usize, 17usize, 9usize), (33, 64, 48)] {
            let a_data: Vec<f32> = (0..m * k)
                .map(|i| ((i as f32 * 0.037).sin()) * 0.25)
                .collect();
            let b_data: Vec<f32> = (0..k * n)
                .map(|i| ((i as f32 * 0.021 + 0.3).cos()) * 0.25)
                .collect();
            let a = Owned::f32(&[m, k], &a_data);
            let b = Owned::f32(&[k, n], &b_data);
            let mut out = Owned::zeros_f32(&[m, n]);
            let mut kernel = MatMulKernel::default();
            kernel.set_constant_inputs(&[false, true]);
            kernel
                .execute_with_backend(
                    &[a.view(), b.view()],
                    &mut [out.view_mut()],
                    CpuBackend::Mlas,
                )
                .unwrap();

            let mut unpacked = vec![0.0; m * n];
            let mut generic = vec![0.0; m * n];
            gemm_with_backend(CpuBackend::Mlas, &a_data, &b_data, &mut unpacked, m, k, n).unwrap();
            gemm_with_backend(CpuBackend::Generic, &a_data, &b_data, &mut generic, m, k, n)
                .unwrap();

            let packed = out.to_f32();
            for (index, ((packed, unpacked), generic)) in
                packed.iter().zip(&unpacked).zip(&generic).enumerate()
            {
                assert!(
                    (packed - unpacked).abs() <= 1e-4,
                    "{m}x{k}x{n} packed/unpacked mismatch at {index}: {packed} vs {unpacked}"
                );
                assert!(
                    (packed - generic).abs() <= 1e-3,
                    "{m}x{k}x{n} packed/generic mismatch at {index}: {packed} vs {generic}"
                );
            }
            assert!(kernel.prepack.packed_b.get().is_some());
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_constant_b_packed_buffer_is_reused() {
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        let weight_data: Vec<f32> = (0..17 * 9)
            .map(|i| ((i as f32 * 0.031).sin()) * 0.5)
            .collect();
        let weight = Owned::f16(&[17, 9], &weight_data);

        let a1_data: Vec<f32> = (0..5 * 17).map(|i| i as f32 * 0.01).collect();
        let a1 = Owned::f32(&[5, 17], &a1_data);
        let mut out1 = Owned::zeros_f32(&[5, 9]);
        kernel
            .execute_with_backend(
                &[a1.view(), weight.view()],
                &mut [out1.view_mut()],
                CpuBackend::Mlas,
            )
            .unwrap();
        let packed_ptr = kernel.prepack.packed_b.get().unwrap() as *const mlas_sys::PackedB;
        let dense_ptr = kernel.prepack.dense[1].filled().unwrap().as_ptr();

        let a2_data: Vec<f32> = (0..5 * 17)
            .map(|i| ((i as f32 * 0.07).cos()) * 0.2)
            .collect();
        let a2 = Owned::f32(&[5, 17], &a2_data);
        let mut out2 = Owned::zeros_f32(&[5, 9]);
        kernel
            .execute_with_backend(
                &[a2.view(), weight.view()],
                &mut [out2.view_mut()],
                CpuBackend::Mlas,
            )
            .unwrap();

        assert_eq!(
            kernel.prepack.packed_b.get().unwrap() as *const mlas_sys::PackedB,
            packed_ptr
        );
        assert_eq!(
            kernel.prepack.dense[1].filled().unwrap().as_ptr(),
            dense_ptr
        );
        assert!(!kernel.prepack.dense[0].is_filled());
        assert_ne!(out1.to_f32(), out2.to_f32());
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_packed_cache_requires_mlas_constant_unbatched_b() {
        let (m, k, n) = (5usize, 17usize, 9usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 * 0.02).sin()) * 0.1)
            .collect();
        let a = Owned::f32(&[m, k], &a_data);
        let b = Owned::f32(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[m, n]);
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, false]);
        kernel
            .execute_with_backend(
                &[a.view(), b.view()],
                &mut [out.view_mut()],
                CpuBackend::Mlas,
            )
            .unwrap();

        let mut expected = vec![0.0; m * n];
        gemm_generic(&a_data, &b_data, &mut expected, m, k, n);
        assert!(kernel.prepack.packed_b.get().is_none());
        for (actual, expected) in out.to_f32().iter().zip(&expected) {
            assert!((actual - expected).abs() <= 1e-3);
        }

        let mut generic_kernel = MatMulKernel::default();
        generic_kernel.set_constant_inputs(&[false, true]);
        let mut generic_out = Owned::zeros_f32(&[m, n]);
        generic_kernel
            .execute_with_backend(
                &[a.view(), b.view()],
                &mut [generic_out.view_mut()],
                CpuBackend::Generic,
            )
            .unwrap();
        assert!(generic_kernel.prepack.packed_b.get().is_none());
        assert_eq!(generic_out.to_f32(), expected);

        let batched_b_data = [b_data.clone(), b_data].concat();
        let batched_a_data = [a_data.clone(), a_data].concat();
        let batched_a = Owned::f32(&[2, m, k], &batched_a_data);
        let batched_b = Owned::f32(&[2, k, n], &batched_b_data);
        let mut batched_out = Owned::zeros_f32(&[2, m, n]);
        let mut batched_kernel = MatMulKernel::default();
        batched_kernel.set_constant_inputs(&[false, true]);
        batched_kernel
            .execute_with_backend(
                &[batched_a.view(), batched_b.view()],
                &mut [batched_out.view_mut()],
                CpuBackend::Mlas,
            )
            .unwrap();
        assert!(batched_kernel.prepack.packed_b.get().is_none());
        for (actual, expected) in batched_out.to_f32().iter().zip(expected.iter().cycle()) {
            assert!((actual - expected).abs() <= 1e-3);
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_selects_a_float_kernel_on_x86_64() {
        assert_ne!(mlas_sys::selected_float_kernel(), 0);
    }

    #[test]
    fn constant_weight_prepack_reuses_weight_and_keeps_activation_live() {
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        let weight = Owned::f16(&[2, 2], &[2., 0., 0., 3.]);

        let a1 = Owned::f32(&[1, 2], &[1., 2.]);
        let mut out1 = Owned::zeros_f32(&[1, 2]);
        kernel
            .execute(&[a1.view(), weight.view()], &mut [out1.view_mut()])
            .unwrap();
        assert_eq!(out1.to_f32(), vec![2., 6.]);
        // On macOS with M=1 and f16 weights, the f16 GEMV path reads weights
        // directly as f16 (via transposed_b_f16) and never widens to f32 —
        // so dense[1] is NOT populated. On other platforms, dense[1] IS populated.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        assert!(kernel.prepack.transposed_b_f16.get().is_some());
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        assert!(kernel.prepack.dense[1].is_filled());
        assert!(!kernel.prepack.dense[0].is_filled());

        // Capture the cache pointer *before* the second execute so the
        // comparison below is a real guard: it proves the first call populated
        // the cache and the second reused it (rather than repopulating).
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let ptr_before = kernel.prepack.transposed_b_f16.get().unwrap().1.as_ptr();
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let ptr_before = kernel.prepack.dense[1].filled().unwrap().as_ptr();

        let a2 = Owned::f32(&[1, 2], &[4., 5.]);
        let mut out2 = Owned::zeros_f32(&[1, 2]);
        kernel
            .execute(&[a2.view(), weight.view()], &mut [out2.view_mut()])
            .unwrap();
        assert_eq!(out2.to_f32(), vec![8., 15.]);

        // The pointer must be unchanged — the OnceLock was already populated.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        assert_eq!(
            kernel.prepack.transposed_b_f16.get().unwrap().1.as_ptr(),
            ptr_before,
            "transposed_b_f16 cache was reallocated on the second execute"
        );
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        assert_eq!(
            kernel.prepack.dense[1].filled().unwrap().as_ptr(),
            ptr_before,
            "dense[1] cache was reallocated on the second execute"
        );
    }

    /// The f16 prefill route through MLAS SGEMM must agree with the widened
    /// reference on shapes whose edges the packed kernel has to handle: `K` and
    /// `N` are both odd and `N` is below a SIMD step, so the tails dominate.
    ///
    /// Also proves the route is actually taken. Without the `packed_b_from_half`
    /// assertion this test would keep passing if the optimisation silently
    /// stopped applying and the slow blocked path served the call instead.
    #[cfg(feature = "mlas")]
    #[test]
    fn f16_prefill_through_mlas_matches_the_widened_reference() {
        let (m, k, n) = (6usize, 37usize, 5usize);
        // Multiples of 1/16 are exact in f16, so any mismatch is the kernel's
        // arithmetic rather than the operands' rounding.
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 16.0)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) / 16.0)
            .collect();
        let expected = naive_matmul(&a_data, &b_data, m, k, n);

        let a = Owned::f16(&[m, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[m, n]);
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();

        // `execute` resolves its backend by auto-detection, and only MLAS has a
        // pack to build; elsewhere the blocked path serves the call correctly.
        assert_eq!(
            kernel.prepack.packed_b_from_half.get().is_some(),
            crate::backend::CpuBackend::auto_detect() == crate::backend::CpuBackend::Mlas,
            "a constant f16 B at M>1 must be widened and packed once on MLAS, \
             not repacked per call"
        );
        for (actual, want) in out.to_f32().iter().zip(expected.iter()) {
            assert!(
                (actual - want).abs() <= 1e-3,
                "f16 prefill disagreed: got {actual}, want {want}"
            );
        }
    }

    /// The pack is built once and reused. A `OnceLock` that were re-initialised
    /// per call would still be correct but would reintroduce the entire cost
    /// this path exists to remove, and no numerical test would notice.
    #[cfg(feature = "mlas")]
    #[test]
    fn f16_prefill_pack_is_built_once_across_calls() {
        let (m, k, n) = (4usize, 16usize, 8usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32 - 4.0) / 8.0).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect();
        let a = Owned::f16(&[m, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut prepack = MatMulPrepack::default();
        prepack.set_constant_inputs(&[false, true]);
        // Forced dispatch: whether the pack is reused is a property of the
        // memo, not of which backend auto-detection happens to pick here.
        let backend = crate::backend::CpuBackend::Mlas;

        let first = try_packed_half_prefill(&prepack, backend, &a.view(), &b.view(), m, k, n)
            .unwrap()
            .expect("a constant f16 B at M>1 must take the packed route");
        let packed_before = prepack
            .packed_b_from_half
            .get()
            .expect("the first call must populate the pack")
            .as_ref()
            .expect("a constant B must pack") as *const _;

        let second = try_packed_half_prefill(&prepack, backend, &a.view(), &b.view(), m, k, n)
            .unwrap()
            .expect("the second call must also take the packed route");
        let packed_after = prepack.packed_b_from_half.get().unwrap().as_ref().unwrap() as *const _;

        assert_eq!(
            packed_before, packed_after,
            "the MLAS pack was rebuilt on the second call"
        );
        assert_eq!(first, second, "repeated calls diverged");
    }

    /// An activation B must never be packed: the pack is only valid because a
    /// constant weight never changes, and caching an activation would return a
    /// stale product the moment the upstream op produced new values.
    #[cfg(feature = "mlas")]
    #[test]
    fn f16_prefill_with_a_non_constant_b_is_correct_and_unpacked() {
        let (m, k, n) = (3usize, 12usize, 6usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32 - 4.0) / 8.0).collect();
        let b1: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect();
        let b2: Vec<f32> = b1.iter().map(|v| v + 0.25).collect();

        let a = Owned::f16(&[m, k], &a_data);
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, false]);

        for b_data in [&b1, &b2] {
            let b = Owned::f16(&[k, n], b_data);
            let mut out = Owned::zeros_f32(&[m, n]);
            kernel
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();
            let expected = naive_matmul(&a_data, b_data, m, k, n);
            for (actual, want) in out.to_f32().iter().zip(expected.iter()) {
                assert!(
                    (actual - want).abs() <= 1e-3,
                    "non-constant f16 B disagreed: got {actual}, want {want}"
                );
            }
            assert!(
                kernel.prepack.packed_b_from_half.get().is_none(),
                "an activation B must never be packed as a weight"
            );
        }
    }

    /// M=1 keeps the GEMV. Packing needs row reuse to pay for itself, and the
    /// decode GEMV is memory-bound and already at parity with ORT, so building
    /// a pack there would add cost and win nothing.
    /// A second, different weight at the same `[k, n]` must never be served
    /// the first weight's pack.
    ///
    /// The executor keys its kernel cache on node and input shapes and a
    /// constant input is a graph initializer with a stable address, so this
    /// cannot happen today. The guard is defence in depth: it makes any future
    /// broadening of "constant" degrade to the slow-but-correct blocked path
    /// rather than silently returning another tensor's product.
    #[cfg(feature = "mlas")]
    #[test]
    fn a_different_weight_is_never_served_the_cached_pack() {
        let (m, k, n) = (3usize, 12usize, 4usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32 - 2.0) / 8.0).collect();
        let first: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect();
        let second: Vec<f32> = first.iter().map(|value| value + 1.0).collect();
        let a = Owned::f16(&[m, k], &a_data);
        let b_first = Owned::f16(&[k, n], &first);
        let b_second = Owned::f16(&[k, n], &second);
        let mut prepack = MatMulPrepack::default();
        prepack.set_constant_inputs(&[false, true]);

        // Forced dispatch: the guard is hardware-independent, so assert it on
        // every target rather than only where auto-detection picks MLAS.
        let backend = crate::backend::CpuBackend::Mlas;
        let served =
            try_packed_half_prefill(&prepack, backend, &a.view(), &b_first.view(), m, k, n)
                .unwrap()
                .expect("the first weight builds and uses the pack");
        let want_first = naive_matmul(&a_data, &first, m, k, n);
        for (actual, want) in served.iter().zip(want_first.iter()) {
            assert!((actual - want).abs() <= 1e-3, "first weight disagreed");
        }

        let reused =
            try_packed_half_prefill(&prepack, backend, &a.view(), &b_second.view(), m, k, n)
                .unwrap();
        assert!(
            reused.is_none(),
            "a weight the pack was not built for must be declined, not served \
             the cached pack; got a result that would be the first weight's"
        );
    }

    /// Calls the shared helper directly at `M = 1`, bypassing the decode GEMV
    /// that both kernels try first.
    ///
    /// `f16_decode_does_not_build_the_prefill_pack` and its `Gemm` twin only
    /// fail under a compound injection, because at `M = 1` the GEMV returns
    /// before the helper is ever reached. This one falsifies the `m <= 1` gate
    /// on its own: remove it and the assertion below sees `Some`.
    #[cfg(feature = "mlas")]
    #[test]
    fn the_packed_prefill_helper_declines_a_single_row() {
        let (k, n) = (24usize, 8usize);
        let a = Owned::f16(
            &[1, k],
            &(0..k).map(|i| (i % 5) as f32 / 8.0).collect::<Vec<_>>(),
        );
        let b = Owned::f16(
            &[k, n],
            &(0..k * n).map(|i| (i % 7) as f32 / 8.0).collect::<Vec<_>>(),
        );
        let mut prepack = MatMulPrepack::default();
        prepack.set_constant_inputs(&[false, true]);
        let single_row = try_packed_half_prefill(
            &prepack,
            crate::backend::CpuBackend::Mlas,
            &a.view(),
            &b.view(),
            1,
            k,
            n,
        )
        .unwrap();
        assert!(
            single_row.is_none(),
            "packing needs row reuse to pay for itself, so M=1 must be declined"
        );
        assert!(
            !prepack.half_pack_is_built(),
            "a declined call must not leave a pack behind"
        );

        // The same weight at M=2 is accepted, so the decline above is the row
        // count and not some unrelated rejection.
        let a2 = Owned::f16(
            &[2, k],
            &(0..2 * k).map(|i| (i % 5) as f32 / 8.0).collect::<Vec<_>>(),
        );
        let two_rows = try_packed_half_prefill(
            &prepack,
            crate::backend::CpuBackend::Mlas,
            &a2.view(),
            &b.view(),
            2,
            k,
            n,
        )
        .unwrap();
        assert!(two_rows.is_some(), "M=2 must take the packed route");
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn f16_decode_does_not_build_the_prefill_pack() {
        let (k, n) = (24usize, 8usize);
        let a_data: Vec<f32> = (0..k).map(|i| ((i % 9) as f32 - 4.0) / 8.0).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect();
        let a = Owned::f16(&[1, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[1, n]);
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();

        assert!(
            kernel.prepack.packed_b_from_half.get().is_none(),
            "M=1 must stay on the GEMV rather than building a prefill pack"
        );
        let expected = naive_matmul(&a_data, &b_data, 1, k, n);
        for (actual, want) in out.to_f32().iter().zip(expected.iter()) {
            assert!((actual - want).abs() <= 1e-3, "decode GEMV disagreed");
        }
    }

    /// The x86 f16 decode GEMV must produce the same values as the blocked
    /// half GEMM it now preempts.
    ///
    /// `N = 5` is below the kernel's 8-lane SIMD step, so the whole stripe
    /// runs through the scalar tail; `K = 137` is a long odd contraction.
    /// Together they exercise the edges the blocked kernel never sees.
    /// The fused widen-pack prefill GEBP must agree with the blocked half GEMM
    /// it replaces, *and* the assertion must be about the route that actually
    /// ran -- so the call count is checked too, not inferred from the numbers.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn half_prefill_gebp_agrees_with_the_blocked_half_gemm_and_is_the_route() {
        use onnx_runtime_ir::DataType;

        // `k * n` must clear `HALF_PREFILL_GEBP_MIN_WEIGHT` for the route to be
        // selected at all.
        let (k, n) = (1024usize, 1024usize);
        assert!(k * n >= HALF_PREFILL_GEBP_MIN_WEIGHT);
        // Multiples of 1/16 in a small range are exact in both f16 and bf16, so
        // any mismatch is the kernel's rather than the operands' rounding.
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) / 16.0)
            .collect();

        for dtype in [DataType::Float16, DataType::BFloat16] {
            for m in [2usize, 5, 17] {
                let a_data: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 7 % 23) as f32 - 11.0) / 16.0)
                    .collect();
                let (a, b) = match dtype {
                    DataType::Float16 => {
                        (Owned::f16(&[m, k], &a_data), Owned::f16(&[k, n], &b_data))
                    }
                    _ => (Owned::bf16(&[m, k], &a_data), Owned::bf16(&[k, n], &b_data)),
                };
                let mut out = Owned::zeros_f32(&[m, n]);

                reset_half_prefill_gebp_calls();
                let mut kernel = MatMulKernel::default();
                kernel.set_constant_inputs(&[false, true]);
                kernel
                    .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                    .unwrap();
                if !crate::backend::has_simd_x86() {
                    continue; // No AVX2/FMA: the blocked half GEMM still runs.
                }
                assert_eq!(
                    half_prefill_gebp_calls(),
                    1,
                    "{dtype:?} m={m}: prefill did not take the fused widen-pack GEBP"
                );

                let expected = naive_matmul(&a_data, &b_data, m, k, n);
                for (actual, want) in out.to_f32().iter().zip(expected.iter()) {
                    // Reordered reductions over 1024 terms need a scaled
                    // tolerance; a fixed epsilon would be meaningless at these
                    // magnitudes.
                    let tol = 1e-3 * (1.0 + want.abs());
                    assert!(
                        (actual - want).abs() <= tol,
                        "{dtype:?} m={m}: got {actual}, want {want}"
                    );
                }
            }
        }
    }

    /// Small weights must stay on the blocked half GEMM: the fused route pays a
    /// widen-pack of `B` and a fork/join that a small `B` cannot repay (0.11x
    /// at 256x256; see `HALF_PREFILL_GEBP_MIN_WEIGHT`).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn half_prefill_gebp_declines_a_weight_below_the_gate() {
        let (k, n) = (64usize, 64usize);
        assert!(k * n < HALF_PREFILL_GEBP_MIN_WEIGHT);
        let a_data: Vec<f32> = (0..8 * k).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 19) as f32 - 9.0) / 16.0).collect();

        reset_half_prefill_gebp_calls();
        let a = Owned::f16(&[8, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[8, n]);
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            half_prefill_gebp_calls(),
            0,
            "a {k}x{n} weight must stay on the blocked half GEMM"
        );
    }

    /// Every decode the GEMV declines must be one the fused GEBP accepts.
    ///
    /// The two gates live in different functions with different thresholds, so
    /// nothing but a test stops them drifting apart -- and the failure mode is
    /// silent: a decode would land on the row-blocked GEMM, which is 16x-21x
    /// slower at these shapes, with identical numbers. Checked as a decision,
    /// not an execution, because the smallest weight that reaches this
    /// threshold is 64 MiB.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn no_decode_is_handed_to_the_blocked_gemm() {
        if !crate::backend::has_simd_x86() {
            return;
        }
        let shapes = [
            (512usize, 512usize),
            (1024, 768),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
            (4096, 8192),
            (4096, 11008),
            (896, 151936),
            (11008, 4096),
        ];
        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            if format == HalfFormat::Bf16 && x86_bf16::native_available() {
                continue;
            }
            for (k, n) in shapes {
                if !half_decode_prefers_gebp(format, k, n) {
                    continue;
                }
                assert!(
                    half_prefill_gebp_selected(format, 1, k, n),
                    "{format:?} {k}x{n}: the GEMV declines but the GEBP does not accept"
                );
            }
        }
    }

    /// The decode threshold is where the measurement put it.
    ///
    /// A regression here is a silent performance change, so the boundary is
    /// pinned rather than left to the constant's definition.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn the_decode_threshold_sits_at_the_measured_wash() {
        if !crate::backend::has_simd_x86() || x86_bf16::native_available() {
            return;
        }
        let k = 4096;
        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            // 4096x4096 = 16.8M elements: the GEMV is within noise and keeps it.
            assert!(
                !half_decode_prefers_gebp(format, k, 4096),
                "{format:?}: a 16.8M weight must stay on the GEMV"
            );
            // 4096x8192 = 33.6M elements: the measured wash, and the first
            // size handed over.
            assert!(
                half_decode_prefers_gebp(format, k, 8192),
                "{format:?}: a 33.6M weight must go to the fused GEBP"
            );
        }
    }

    /// Decode below the weight threshold keeps the GEMV, in both formats.
    ///
    /// The packing only pays for itself on a weight big enough to keep every
    /// worker busy; under that the GEMV is 2.6x-5.6x ahead of the blocking
    /// path, so this is where `bf16` decode actually changed.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn decode_on_a_small_weight_keeps_the_gemv() {
        let (k, n) = (512usize, 512usize);
        assert!(
            k * n < HALF_PREFILL_GEBP_MIN_WEIGHT,
            "the shape must stay under the weight gate"
        );
        let a_data: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 19) as f32 - 9.0) / 16.0).collect();

        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            if !half_gemv::simd_available(format) {
                continue;
            }
            reset_half_prefill_gebp_calls();
            reset_half_decode_gemv_calls();
            let (a, b) = match format {
                HalfFormat::F16 => (Owned::f16(&[1, k], &a_data), Owned::f16(&[k, n], &b_data)),
                HalfFormat::Bf16 => (Owned::bf16(&[1, k], &a_data), Owned::bf16(&[k, n], &b_data)),
            };
            let mut out = Owned::zeros_f32(&[1, n]);
            let mut kernel = MatMulKernel::default();
            kernel.set_constant_inputs(&[false, true]);
            kernel
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();
            assert_eq!(
                half_decode_gemv_calls(),
                1,
                "{format:?} M=1 on a {k}x{n} weight must take the GEMV"
            );
            assert_eq!(
                half_prefill_gebp_calls(),
                0,
                "{format:?} M=1 on a {k}x{n} weight must not pack"
            );
            // The GEMV's whole justification is reading B in place: a widened
            // or transposed copy would defeat it.
            assert!(
                !kernel.prepack.dense[1].is_filled(),
                "{format:?} decode must not widen B to f32"
            );
            assert!(
                kernel.prepack.transposed_b_f16.get().is_none(),
                "{format:?} decode must not materialise a transposed weight"
            );

            let want = naive_matmul(&a_data, &b_data, 1, k, n);
            for (index, (g, w)) in out.to_f32().iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 2e-2 * (1.0 + w.abs()),
                    "{format:?} column {index}: {g} != {w}"
                );
            }
        }
    }

    #[test]
    fn f16_decode_gemv_agrees_with_the_blocked_half_gemm() {
        let k = 137usize;
        let n = 5usize;
        // Multiples of 1/16 are exact in f16, so any mismatch is the kernel's
        // rather than the operands' rounding.
        let a_data: Vec<f32> = (0..k)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 16.0)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) / 16.0)
            .collect();

        let expected = naive_matmul(&a_data, &b_data, 1, k, n);

        let a = Owned::f16(&[1, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[1, n]);
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();

        for (actual, want) in out.to_f32().iter().zip(expected.iter()) {
            assert!(
                (actual - want).abs() <= 1e-3,
                "f16 decode GEMV disagreed: got {actual}, want {want}"
            );
        }
    }

    /// A non-constant f16 B must still be handled correctly. Unlike the
    /// Accelerate GEMV, this kernel reads B in place rather than through a
    /// weight cache, so an activation B is a *supported* input here, not a
    /// fallthrough — and it must never be memoised as a weight.
    #[test]
    fn f16_decode_with_a_non_constant_weight_is_correct_and_uncached() {
        let k = 40usize;
        let n = 3usize;
        let a_data: Vec<f32> = (0..k).map(|i| ((i % 9) as f32 - 4.0) / 8.0).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect();
        let expected = naive_matmul(&a_data, &b_data, 1, k, n);

        let a = Owned::f16(&[1, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[1, n]);
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, false]);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();

        assert!(
            kernel.prepack.transposed_b_f16.get().is_none(),
            "an activation B must never be memoised as a weight transpose"
        );
        assert!(
            !kernel.prepack.dense[1].is_filled(),
            "an activation B must never be memoised as a dense weight"
        );
        for (actual, want) in out.to_f32().iter().zip(expected.iter()) {
            assert!((actual - want).abs() <= 1e-3, "got {actual}, want {want}");
        }
    }

    /// A non-contiguous B must be declined by the dispatch guard: the kernel
    /// reads `B` as a flat `k * n` slice, so a strided view would be read with
    /// the wrong layout. Pinned as a negative test because the guard is the
    /// only thing standing between a strided weight and a silently wrong
    /// answer.
    #[test]
    fn f16_decode_declines_a_non_contiguous_weight_and_stays_correct() {
        let k = 6usize;
        let n = 4usize;
        // Build B as the transpose of a [n, k] buffer so the [k, n] view is
        // genuinely strided rather than merely offset.
        let mut bt_data = vec![0.0f32; n * k];
        let mut b_data = vec![0.0f32; k * n];
        for p in 0..k {
            for j in 0..n {
                let v = ((p * n + j) % 9) as f32 / 8.0 - 0.5;
                b_data[p * n + j] = v;
                bt_data[j * k + p] = v;
            }
        }
        let a_data: Vec<f32> = (0..k).map(|i| (i % 5) as f32 / 4.0 - 0.5).collect();
        let expected = naive_matmul(&a_data, &b_data, 1, k, n);

        let a = Owned::f16(&[1, k], &a_data);
        let bt = Owned::f16(&[n, k], &bt_data);
        let mut b_view = bt.view();
        let shape = [k, n];
        let strided = [1i64, k as i64];
        b_view.shape = &shape;
        b_view.strides = &strided;
        assert!(
            !b_view.is_contiguous(),
            "test must present a genuinely strided B"
        );

        let mut out = Owned::zeros_f32(&[1, n]);
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        kernel
            .execute(&[a.view(), b_view], &mut [out.view_mut()])
            .unwrap();

        for (actual, want) in out.to_f32().iter().zip(expected.iter()) {
            assert!(
                (actual - want).abs() <= 1e-3,
                "strided B gave {actual}, want {want}"
            );
        }
    }

    /// The GEMV must not allocate a weight copy: it reads B straight from the
    /// stored `[K, N]` layout. Pinned because a transposed variant would cost
    /// a permanent `2 * K * N` bytes that `try_matmul_half` never paid.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn f16_decode_gemv_does_not_cache_a_copy_of_the_weight() {
        let k = 64usize;
        let n = 8usize;
        let a_data: Vec<f32> = (0..k).map(|i| ((i % 5) as f32 - 2.0) / 4.0).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 8.0).collect();

        let a = Owned::f16(&[1, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[1, n]);
        let mut kernel = MatMulKernel::default();
        // B *is* a constant here: even so, no weight copy may be materialised.
        kernel.set_constant_inputs(&[false, true]);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();

        let expected = naive_matmul(&a_data, &b_data, 1, k, n);
        for (actual, want) in out.to_f32().iter().zip(expected.iter()) {
            assert!((actual - want).abs() <= 1e-3, "got {actual}, want {want}");
        }
        if half_gemv::simd_available(HalfFormat::F16) {
            assert!(
                kernel.prepack.transposed_b_f16.get().is_none(),
                "the GEMV must read B in place, not through a transpose cache"
            );
            assert!(
                !kernel.prepack.dense[1].is_filled(),
                "the GEMV must not widen B to f32"
            );
        }
    }

    // --- Direct f32 output path (Option A) --------------------------------

    /// Reference row-major GEMM for verifying the direct path numerically.
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += aip * b[p * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn direct_f32_eligible_for_contiguous_cpu_output() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        assert!(output_is_direct_f32_eligible(
            &a.view(),
            &b.view(),
            &out.view_mut()
        ));
    }

    #[test]
    fn direct_f32_rejects_non_f32_output() {
        // f16 output must fall back to the narrowing writer.
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Float16, &[2, 2]);
        assert!(!output_is_direct_f32_eligible(
            &a.view(),
            &b.view(),
            &out.view_mut()
        ));
    }

    #[test]
    fn direct_f32_2d_nonsquare_matches_reference() {
        // A[2,3] @ B[3,4] contiguous f32: the direct path writes into `out`.
        let a_data = [1., 2., 3., 4., 5., 6.];
        let b_data = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let a = Owned::f32(&[2, 3], &a_data);
        let b = Owned::f32(&[3, 4], &b_data);
        let mut out = Owned::zeros_f32(&[2, 4]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), naive_matmul(&a_data, &b_data, 2, 3, 4));
    }

    #[test]
    fn direct_f32_batched_and_broadcast_match_reference() {
        // Batched: two independent [2,2] matmuls into one contiguous output.
        let a = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::f32(&[2, 2, 2], &[1., 0., 0., 1., 2., 0., 0., 2.]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 10., 12., 14., 16.]);

        // Broadcast B over the batch dim.
        let a = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::f32(&[2, 2], &[1., 0., 0., 1.]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn direct_f32_matrix_times_vector() {
        // A[2,3] @ b[3] -> [2] (b promoted to [3,1], result col squeezed).
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3], &[7., 9., 11.]);
        let mut out = Owned::zeros_f32(&[2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // [1*7+2*9+3*11, 4*7+5*9+6*11] = [58, 139]
        assert_eq!(out.to_f32(), vec![58., 139.]);
    }

    #[test]
    fn direct_f32_vector_times_vector_scalar_result() {
        // a[3] @ b[3] -> scalar (shape []), a promoted [1,3], b promoted [3,1].
        let a = Owned::f32(&[3], &[1., 2., 3.]);
        let b = Owned::f32(&[3], &[4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(out.to_f32(), vec![32.]);
    }

    #[test]
    fn direct_f32_zero_sized_result_writes_nothing() {
        // A zero batch dim yields an empty result; the direct path must return
        // before any GEMM write.
        let a = Owned::f32(&[0, 2, 3], &[]);
        let b = Owned::f32(&[0, 3, 2], &[]);
        let mut out = Owned::zeros_f32(&[0, 2, 2]);
        assert!(output_is_direct_f32_eligible(
            &a.view(),
            &b.view(),
            &out.view_mut()
        ));
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert!(out.to_f32().is_empty());
    }

    #[test]
    fn strided_f32_output_takes_fallback_and_is_correct() {
        // A[2,2] @ B[2,2] into a NON-contiguous [2,2] output: row stride 3 over a
        // [2,3] backing buffer. It must NOT take the direct path; the strided
        // writer scatters into positions 0,1,3,4.
        let a = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let b = Owned::f32(&[2, 2], &[5., 6., 7., 8.]);
        let mut out = Owned::zeros_f32(&[2, 3]).with_view(&[2, 2], &[3, 1]);
        assert!(!out.view_mut().is_contiguous());
        assert!(!output_is_direct_f32_eligible(
            &a.view(),
            &b.view(),
            &out.view_mut()
        ));
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // C = [[19,22],[43,50]] scattered: buf[0]=19, buf[1]=22, buf[3]=43, buf[4]=50.
        assert_eq!(out.to_f32(), vec![19., 22., 0., 43., 50., 0.]);
    }

    #[test]
    fn mismatched_output_length_errors_before_write() {
        // A[2,3] @ B[3,2] -> [2,2] (4 elems), but the output view has 6 elems.
        // The direct path must error on the length check before any GEMM write.
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        let err = MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap_err();
        assert!(format!("{err}").contains("does not match result length"));
        // Nothing was written.
        assert_eq!(out.to_f32(), vec![0.; 6]);
    }

    #[test]
    fn output_overlaps_input_helper_detects_ranges() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::{DataType, DeviceId, DeviceType};

        let buf = [0.0f32; 8];
        let shape = [2usize, 2];
        let strides = compute_contiguous_strides(&shape);
        let base = buf.as_ptr() as usize;
        let bytes = 4 * 4; // 4 f32

        // Input covering buf[0..4].
        let input = TensorView::new(
            DevicePtr(buf.as_ptr() as *const std::ffi::c_void),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );

        // Overlapping output starting inside the input range.
        assert!(output_overlaps_input(
            base + 8,
            base + 8 + bytes,
            &input,
            DeviceId::cpu()
        ));
        // Disjoint output entirely past the input range.
        assert!(!output_overlaps_input(
            base + bytes,
            base + 2 * bytes,
            &input,
            DeviceId::cpu()
        ));
        // Absent input never overlaps.
        assert!(!output_overlaps_input(
            base,
            base + bytes,
            &TensorView::absent(DataType::Float32),
            DeviceId::cpu()
        ));
        // Different device is treated as non-overlapping (distinct address space).
        assert!(!output_overlaps_input(
            base,
            base + bytes,
            &input,
            DeviceId::new(DeviceType::Cuda, 0)
        ));
        let _ = DevicePtrMut(std::ptr::null_mut());
    }

    #[test]
    fn aliasing_output_takes_fallback_and_is_correct() {
        // DeviceIoBinding permits input/output aliasing. Construct an output that
        // shares A's backing buffer; the direct path must be rejected and the
        // owned-buffer fallback must still produce the correct result even though
        // A is read while C is written to the same memory.
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::{DataType, DeviceId};

        // A = [[1,2],[3,4]] shared with C; B = column swap so C = [[2,1],[4,3]].
        let mut shared = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_buf = [0.0f32, 1.0, 1.0, 0.0];
        let shape = vec![2usize, 2];
        let strides = compute_contiguous_strides(&shape);
        let a_ptr = shared.as_ptr() as *const std::ffi::c_void;
        let c_ptr = shared.as_mut_ptr() as *mut std::ffi::c_void;

        let a = TensorView::new(
            DevicePtr(a_ptr),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let b = TensorView::new(
            DevicePtr(b_buf.as_ptr() as *const std::ffi::c_void),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let c = TensorMut::new(
            DevicePtrMut(c_ptr),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );

        // Output aliases input A: direct path must be rejected.
        assert!(!output_is_direct_f32_eligible(&a, &b, &c));

        MatMulKernel::default().execute(&[a, b], &mut [c]).unwrap();
        // Fallback computed the full result into a temp buffer before writing.
        assert_eq!(shared, vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn accelerate_sgemm_matches_generic_for_small_shapes() {
        let shapes = [(1, 4, 4), (2, 3, 5), (4, 8, 4), (1, 16, 16), (8, 8, 8)];
        for (m, k, n) in shapes {
            let a: Vec<f32> = (0..m * k)
                .map(|i| (i as f32 * 0.7 + 0.3).sin() * 2.0)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| (i as f32 * 1.3 + 0.7).cos() * 2.0)
                .collect();
            let mut generic_out = vec![0.0f32; m * n];
            let mut accel = vec![0.0f32; m * n];
            gemm_with_backend(CpuBackend::Generic, &a, &b, &mut generic_out, m, k, n).unwrap();
            gemm_with_backend(CpuBackend::Accelerate, &a, &b, &mut accel, m, k, n).unwrap();
            for idx in 0..m * n {
                let diff = (generic_out[idx] - accel[idx]).abs();
                assert!(
                    diff < 1e-4,
                    "[{m},{k},{n}][{idx}]: g={} a={} d={}",
                    generic_out[idx],
                    accel[idx],
                    diff
                );
            }
        }
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn accelerate_decode_gemv_matches_generic_at_model_scale() {
        let shapes = [(1, 896, 896), (1, 896, 4864), (1, 4864, 896)];
        for (m, k, n) in shapes {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.31 + 0.17).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.13 + 0.71).cos()).collect();
            let mut generic_out = vec![0.0f32; m * n];
            let mut accel = vec![0.0f32; m * n];
            gemm_with_backend(CpuBackend::Generic, &a, &b, &mut generic_out, m, k, n).unwrap();
            gemm_with_backend(CpuBackend::Accelerate, &a, &b, &mut accel, m, k, n).unwrap();
            let max_rel = generic_out
                .iter()
                .zip(accel.iter())
                .map(|(g, a)| (g - a).abs() / g.abs().max(1e-8))
                .fold(0.0f32, f32::max);
            // Chew measured the worst model-scale accumulation-order drift at
            // 1.57%; 1.8% keeps modest cross-machine headroom without letting a
            // real GEMV regression hide behind the old 2% envelope.
            assert!(max_rel < 1.8e-2, "[{m},{k},{n}]: max_rel={max_rel}");
        }
    }

    #[test]
    fn col_parallel_gemm_matches_reference_for_m1() {
        let (m, k, n) = (1, 256, 512);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.41 + 0.13).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.23 + 0.57).cos()).collect();
        let mut reference = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                reference[i * n + j] = s;
            }
        }
        let mut col_par = vec![0.0f32; m * n];
        gemm_generic_col_parallel(&a, &b, &mut col_par, m, k, n, 8);
        for idx in 0..m * n {
            let diff = (reference[idx] - col_par[idx]).abs();
            assert!(
                diff < 1e-4,
                "[{m},{k},{n}][{idx}]: ref={} col={} d={}",
                reference[idx],
                col_par[idx],
                diff
            );
        }
    }

    /// Guard: an FP16 weight MatMul at M=1 (decode shape) must reach the NEON
    /// GEMV path on Apple Silicon, not the half-precision blocked GEMM.  If the
    /// dispatch order changes so `try_matmul_half` intercepts M=1 before the
    /// GEMV, decode throughput drops ~4×.  This test would have caught the
    /// `half_gemm.rs` regression that took native FP16 from 60→13 tok/s.
    ///
    /// Both A and B are Float16 to exercise the real fp16-model dispatch: the
    /// regression occurs precisely when `try_matmul_half` matches (f16, f16) at
    /// M=1 before the GEMV path can claim it.
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn fp16_m1_decode_reaches_neon_gemv_not_half_gemm() {
        use std::sync::atomic::Ordering;

        let (k, n) = (64, 32);
        let a_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        let a = Owned::f16(&[1, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[1, n]);

        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);

        let before = GEMV_F16_TEST_HITS.load(Ordering::Relaxed);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = GEMV_F16_TEST_HITS.load(Ordering::Relaxed);

        assert!(
            after > before,
            "FP16 M=1 decode did not reach neon_gemv_f16_col_parallel — \
             half_gemm.rs is likely intercepting M=1 before the GEMV path, \
             which causes a ~4× decode throughput regression"
        );
        // Sanity: output should be finite
        assert!(
            out.to_f32().iter().all(|v| v.is_finite()),
            "GEMV produced non-finite output"
        );
    }

    // ─── BNNS prefill dispatch reachability ─────────────────────────

    /// Guard: FP16 M≥2 prefill on macOS must reach the BNNS path, not
    /// the portable half_gemm. The BNNS path reaches AMX at ~2451 GFLOPS
    /// vs 52 GFLOPS for the NEON blocked GEMM.
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn fp16_m_ge2_prefill_reaches_bnns_not_half_gemm() {
        use std::sync::atomic::Ordering;

        if !accelerate_gemm::bnns_matmul_available() {
            eprintln!("BNNS not available, skipping dispatch guard");
            return;
        }

        let (m, k, n) = (4, 64, 32);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        let a = Owned::f16(&[m, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[m, n]);

        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);

        let before = BNNS_F16_TEST_HITS.load(Ordering::SeqCst);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = BNNS_F16_TEST_HITS.load(Ordering::SeqCst);

        assert!(
            after > before,
            "FP16 M≥2 prefill did not reach BNNS path — \
             the portable half_gemm.rs is likely intercepting before BNNS, \
             which would regress prefill by ~47×"
        );
        assert!(
            out.to_f32().iter().all(|v| v.is_finite()),
            "BNNS produced non-finite output"
        );
    }

    /// Guard: BF16 M≥2 must NOT reach the BNNS path (BNNS only supports f16).
    /// Verified by checking that bf16 output matches the portable half_gemm
    /// reference — if BNNS were used, it would reinterpret bf16 bits as f16,
    /// producing wildly incorrect results.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn bf16_m_ge2_does_not_reach_bnns() {
        let (m, k, n) = (4, 64, 32);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        let a = Owned::bf16(&[m, k], &a_data);
        let b = Owned::bf16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[m, n]);

        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();

        // Compute f64 reference from the actual bf16-rounded values
        let a_bf16: Vec<half::bf16> = a_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let b_bf16: Vec<half::bf16> = b_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let mut ref_c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    sum += a_bf16[i * k + p].to_f64() * b_bf16[p * n + j].to_f64();
                }
                ref_c[i * n + j] = sum;
            }
        }

        let result = out.to_f32();
        let max_rel = result
            .iter()
            .zip(&ref_c)
            .filter(|(_, r)| r.abs() > 1e-6)
            .map(|(a, r)| (((*a as f64) - r) / r).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_rel < 0.05,
            "BF16 M≥2 max relative error {max_rel:.6} — if extremely large, \
             BNNS may be reinterpreting bf16 bits as f16"
        );
    }

    /// Numerics parity: end-to-end MatMulKernel with f16 at M≥2 must match
    /// the f64 reference, exercising the BNNS dispatch path on macOS.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn bnns_f16_prefill_matches_f64_reference_via_matmul_kernel() {
        let (m, k, n) = (8, 64, 32);
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i % 997) as f32) * 0.001 - 0.5)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i % 991) as f32) * 0.001 - 0.5)
            .collect();

        let a = Owned::f16(&[m, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[m, n]);

        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();

        let a_f16: Vec<half::f16> = a_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let b_f16: Vec<half::f16> = b_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let mut ref_c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    sum += a_f16[i * k + p].to_f64() * b_f16[p * n + j].to_f64();
                }
                ref_c[i * n + j] = sum;
            }
        }

        let result = out.to_f32();
        let max_rel = result
            .iter()
            .zip(&ref_c)
            .filter(|(_, r)| r.abs() > 1e-6)
            .map(|(a, r)| (((*a as f64) - r) / r).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_rel < 0.01,
            "BNNS f16 prefill max relative error {max_rel:.6} exceeds 1%"
        );
    }

    /// Guard: Non-constant non-contiguous f16 B at M≥2 must NOT enter the
    /// rescue block (which would produce all zeros). Instead it must fall
    /// through to the generic `matmul_dense_prepacked_with_backend` path and
    /// produce correct results via f32 widening.
    ///
    /// This is the test that would have caught Blocking Bug #1 — the rescue
    /// block originally lacked the `constant_inputs[1]` guard.
    #[test]
    fn f16_non_constant_non_contiguous_b_produces_correct_result() {
        let (m, k, n) = (4, 32, 16);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
        // B stored as [N, K] (transposed layout), viewed as [K, N] with
        // column-major strides [1, K] — this is non-contiguous for a [K,N]
        // shape but valid memory.
        let b_data_transposed: Vec<f32> =
            (0..n * k).map(|i| ((i % 89) as f32) * 0.01 - 0.5).collect();

        let a = Owned::f16(&[m, k], &a_data);
        // Physical layout is [N, K] row-major; view as [K, N] column-major
        let b_physical = Owned::f16(&[n, k], &b_data_transposed);
        let b = b_physical.with_view(&[k, n], &[1, k as i64]);
        let mut out = Owned::zeros_f32(&[m, n]);

        let mut kernel = MatMulKernel::default();
        // Crucially: B is NOT constant — simulates an activation from another op
        kernel.set_constant_inputs(&[false, false]);

        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();

        // Compute f64 reference: C[i,j] = sum_p A[i,p] * B[p,j]
        // B[p,j] in column-major with strides [1,K] means element [p,j] is at
        // physical offset j*K + p, which is b_data_transposed[j*k + p].
        let a_f16: Vec<half::f16> = a_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let b_f16: Vec<half::f16> = b_data_transposed
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let mut ref_c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    sum += a_f16[i * k + p].to_f64() * b_f16[j * k + p].to_f64();
                }
                ref_c[i * n + j] = sum;
            }
        }

        let result = out.to_f32();
        // Must not be all zeros (that was the bug)
        assert!(
            result.iter().any(|&v| v != 0.0),
            "Non-constant non-contiguous B produced all-zero output — \
             the rescue block is being entered without constant_inputs[1] guard"
        );
        let max_rel = result
            .iter()
            .zip(&ref_c)
            .filter(|(_, r)| r.abs() > 1e-6)
            .map(|(a, r)| (((*a as f64) - r) / r).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_rel < 0.02,
            "Non-constant non-contiguous B: max relative error {max_rel:.6} exceeds 2%"
        );
    }

    /// Constant non-contiguous f16 B at M≥2 correctly enters the rescue block
    /// and uses the cached contiguous copy or BNNS trans_b path.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn f16_constant_non_contiguous_b_enters_rescue_block() {
        use std::sync::atomic::Ordering;

        let (m, k, n) = (4, 32, 16);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
        let b_data_transposed: Vec<f32> =
            (0..n * k).map(|i| ((i % 89) as f32) * 0.01 - 0.5).collect();

        let a = Owned::f16(&[m, k], &a_data);
        let b_physical = Owned::f16(&[n, k], &b_data_transposed);
        let b = b_physical.with_view(&[k, n], &[1, k as i64]);
        let mut out = Owned::zeros_f32(&[m, n]);

        let mut kernel = MatMulKernel::default();
        // B IS constant — should enter rescue block
        kernel.set_constant_inputs(&[false, true]);

        let before = NONCONTIG_RESCUE_TEST_HITS.load(Ordering::SeqCst);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = NONCONTIG_RESCUE_TEST_HITS.load(Ordering::SeqCst);

        assert!(
            after > before,
            "Constant non-contiguous B did not enter the rescue block — \
             dispatch may be falling through to the slow f32-widen path"
        );

        // Verify correctness regardless of path taken
        let a_f16: Vec<half::f16> = a_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let b_f16: Vec<half::f16> = b_data_transposed
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let mut ref_c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    sum += a_f16[i * k + p].to_f64() * b_f16[j * k + p].to_f64();
                }
                ref_c[i * n + j] = sum;
            }
        }

        let result = out.to_f32();
        assert!(
            result.iter().any(|&v| v != 0.0),
            "Constant non-contiguous B produced all-zero output"
        );
        let max_rel = result
            .iter()
            .zip(&ref_c)
            .filter(|(_, r)| r.abs() > 1e-6)
            .map(|(a, r)| (((*a as f64) - r) / r).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_rel < 0.02,
            "Constant non-contiguous B: max relative error {max_rel:.6} exceeds 2%"
        );
    }

    // ─── Dispatch-reachability coverage: column-major GEMV (M=1) ────────

    /// Guard: M=1 decode with a constant column-major f16 B must reach the
    /// zero-copy column-major GEMV path (no transpose needed).
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn fp16_m1_column_major_b_reaches_colmaj_gemv() {
        use std::sync::atomic::Ordering;

        let (k, n) = (64, 32);
        let a_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        // B stored as [N,K] row-major; viewed as [K,N] column-major strides [1,K]
        let b_transposed: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.001).collect();

        let a = Owned::f16(&[1, k], &a_data);
        let b_phys = Owned::f16(&[n, k], &b_transposed);
        let b = b_phys.with_view(&[k, n], &[1, k as i64]);
        let mut out = Owned::zeros_f32(&[1, n]);

        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);

        let before = GEMV_F16_COLMAJ_TEST_HITS.load(Ordering::Relaxed);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = GEMV_F16_COLMAJ_TEST_HITS.load(Ordering::Relaxed);

        assert!(
            after > before,
            "FP16 M=1 column-major B did not reach the zero-copy GEMV path"
        );

        // Verify correctness: C[j] = sum_p A[p] * B[p,j]
        // B[p,j] with strides [1,K] ⇒ physical offset j*K + p
        let a_f16: Vec<half::f16> = a_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let b_f16: Vec<half::f16> = b_transposed
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let mut ref_c = vec![0.0f64; n];
        for j in 0..n {
            for p in 0..k {
                ref_c[j] += a_f16[p].to_f64() * b_f16[j * k + p].to_f64();
            }
        }
        let result = out.to_f32();
        assert!(
            result.iter().any(|&v| v != 0.0),
            "Column-major GEMV produced zeros"
        );
        let max_rel = result
            .iter()
            .zip(&ref_c)
            .filter(|(_, r)| r.abs() > 1e-6)
            .map(|(a, r)| (((*a as f64) - r) / r).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_rel < 0.01,
            "Column-major GEMV: max relative error {max_rel:.6} exceeds 1%"
        );
    }

    // ─── Dispatch-reachability: non-constant non-contiguous (the bug path) ──

    /// Guard: non-constant non-contiguous f16 B at M=1 must NOT enter the
    /// column-major GEMV (which requires constant_inputs[1]) — it must fall
    /// through to the generic path and produce correct results.
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn fp16_m1_non_constant_colmaj_b_does_not_reach_gemv() {
        use std::sync::atomic::Ordering;

        let (k, n) = (64, 32);
        let a_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let b_transposed: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.001).collect();

        let a = Owned::f16(&[1, k], &a_data);
        let b_phys = Owned::f16(&[n, k], &b_transposed);
        let b = b_phys.with_view(&[k, n], &[1, k as i64]);
        let mut out = Owned::zeros_f32(&[1, n]);

        let mut kernel = MatMulKernel::default();
        // Non-constant B — should NOT reach the GEMV path
        kernel.set_constant_inputs(&[false, false]);

        let before = GEMV_F16_COLMAJ_TEST_HITS.load(Ordering::Relaxed);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = GEMV_F16_COLMAJ_TEST_HITS.load(Ordering::Relaxed);

        assert_eq!(
            before, after,
            "Non-constant B incorrectly reached column-major GEMV"
        );
        // Still must produce correct output (via generic path)
        let result = out.to_f32();
        assert!(
            result.iter().any(|&v| v != 0.0),
            "Non-constant M=1 non-contiguous B produced all zeros"
        );
    }

    // ─── Dispatch-reachability: non-contiguous rescue block variations ───

    /// Guard: M≥2 with non-contiguous non-constant f16 B must NOT enter the
    /// rescue block. This is the exact bug scenario from PR #275 review.
    /// The test proves the `constant_inputs[1]` guard is effective.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn f16_m_ge2_non_constant_non_contiguous_b_does_not_enter_rescue() {
        use std::sync::atomic::Ordering;

        let (m, k, n) = (4, 32, 16);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
        let b_data_transposed: Vec<f32> =
            (0..n * k).map(|i| ((i % 89) as f32) * 0.01 - 0.5).collect();

        let a = Owned::f16(&[m, k], &a_data);
        let b_physical = Owned::f16(&[n, k], &b_data_transposed);
        let b = b_physical.with_view(&[k, n], &[1, k as i64]);
        let mut out = Owned::zeros_f32(&[m, n]);

        let mut kernel = MatMulKernel::default();
        // Non-constant — must NOT enter rescue block
        kernel.set_constant_inputs(&[false, false]);

        let before = NONCONTIG_RESCUE_TEST_HITS.load(Ordering::SeqCst);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = NONCONTIG_RESCUE_TEST_HITS.load(Ordering::SeqCst);

        assert_eq!(
            before, after,
            "Non-constant non-contiguous B incorrectly entered rescue block — \
             this would produce all-zero output (the exact PR #275 bug)"
        );
        // Must produce correct (non-zero) output via the generic path
        let result = out.to_f32();
        assert!(
            result.iter().any(|&v| v != 0.0),
            "Non-constant non-contiguous B produced all-zero output"
        );
    }

    /// Guard: M≥2 with constant non-contiguous f16 B (non-column-major layout,
    /// e.g. a permuted 3D weight) enters the rescue block and uses the
    /// contiguous-copy fallback.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn f16_constant_non_contiguous_non_colmaj_b_enters_rescue() {
        use std::sync::atomic::Ordering;

        let (m, k, n) = (4, 16, 8);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
        // Non-column-major, non-contiguous: e.g. shape [K,N] with strides [2, 1]
        // (stride-0 = 2 ≠ N=8, so not row-major; stride-0 ≠ 1, so not col-major)
        // Physical buffer must be large enough: need (K-1)*2 + (N-1)*1 + 1 elements
        // Keep `* 1` for symmetry with `* 2` — mirrors stride formula (dim-1)*stride.
        #[allow(clippy::identity_op)]
        let phys_len = (k - 1) * 2 + (n - 1) * 1 + 1;
        let b_phys_data: Vec<f32> = (0..phys_len)
            .map(|i| ((i % 89) as f32) * 0.01 - 0.5)
            .collect();

        let a = Owned::f16(&[m, k], &a_data);
        // Build a physically larger buffer viewed with stride [2, 1] for shape [K, N]
        let b_physical = Owned::f16(&[phys_len], &b_phys_data);
        let b = b_physical.with_view(&[k, n], &[2, 1]);
        let mut out = Owned::zeros_f32(&[m, n]);

        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);

        let before = NONCONTIG_RESCUE_TEST_HITS.load(Ordering::SeqCst);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = NONCONTIG_RESCUE_TEST_HITS.load(Ordering::SeqCst);

        assert!(
            after > before,
            "Constant non-contiguous non-column-major B did not enter rescue block"
        );

        // Verify correctness: B[p,j] is at physical offset p*2 + j*1
        let b_f16: Vec<half::f16> = b_phys_data
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let a_f16: Vec<half::f16> = a_data.iter().map(|&v| half::f16::from_f32(v)).collect();
        let mut ref_c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    let b_idx = p * 2 + j;
                    sum += a_f16[i * k + p].to_f64() * b_f16[b_idx].to_f64();
                }
                ref_c[i * n + j] = sum;
            }
        }
        let result = out.to_f32();
        assert!(
            result.iter().any(|&v| v != 0.0),
            "Constant non-contiguous non-column-major B produced all zeros"
        );
        let max_rel = result
            .iter()
            .zip(&ref_c)
            .filter(|(_, r)| r.abs() > 1e-6)
            .map(|(a, r)| (((*a as f64) - r) / r).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_rel < 0.02,
            "Non-column-major rescue: max relative error {max_rel:.6} exceeds 2%"
        );
    }

    // ─── Dispatch-reachability: f32 fallback for non-f16 inputs ─────────

    /// Guard: f32 inputs at M≥2 must NOT enter any half-precision path.
    /// Verifies the dtype check at each dispatch branch.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn f32_m_ge2_does_not_enter_half_or_rescue_paths() {
        use std::sync::atomic::Ordering;

        let (m, k, n) = (4, 32, 16);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        let a = Owned::f32(&[m, k], &a_data);
        let b = Owned::f32(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[m, n]);

        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);

        let rescue_before = NONCONTIG_RESCUE_TEST_HITS.load(Ordering::SeqCst);
        let bnns_before = BNNS_F16_TEST_HITS.load(Ordering::SeqCst);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let rescue_after = NONCONTIG_RESCUE_TEST_HITS.load(Ordering::SeqCst);
        let bnns_after = BNNS_F16_TEST_HITS.load(Ordering::SeqCst);

        assert_eq!(
            rescue_before, rescue_after,
            "f32 input entered rescue block"
        );
        assert_eq!(bnns_before, bnns_after, "f32 input entered BNNS f16 path");
        // Correctness check
        let result = out.to_f32();
        assert!(
            result.iter().any(|&v| v != 0.0),
            "f32 matmul produced zeros"
        );
    }

    /// Guard: bf16 M≥2 must NOT enter the non-contiguous rescue block
    /// (which is f16-only).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn bf16_non_contiguous_does_not_enter_f16_rescue() {
        use std::sync::atomic::Ordering;

        let (m, k, n) = (4, 32, 16);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
        let b_data_transposed: Vec<f32> =
            (0..n * k).map(|i| ((i % 89) as f32) * 0.01 - 0.5).collect();

        let a = Owned::bf16(&[m, k], &a_data);
        let b_physical = Owned::bf16(&[n, k], &b_data_transposed);
        let b = b_physical.with_view(&[k, n], &[1, k as i64]);
        let mut out = Owned::zeros_f32(&[m, n]);

        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);

        let before = NONCONTIG_RESCUE_TEST_HITS.load(Ordering::SeqCst);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = NONCONTIG_RESCUE_TEST_HITS.load(Ordering::SeqCst);

        assert_eq!(
            before, after,
            "bf16 input incorrectly entered f16-only rescue block"
        );
        let result = out.to_f32();
        assert!(
            result.iter().any(|&v| v != 0.0),
            "bf16 non-contiguous B produced all zeros"
        );
    }

    // ─── Thin-M GEMM dispatch reachability ──────────────────────────

    /// Guard: f32 thin-M prefill (M=2..16, large K×N) on macOS must reach the
    /// column-parallel NEON path, not fall through to cblas_sgemm.
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn f32_thin_m_prefill_reaches_neon_col_parallel() {
        use std::sync::atomic::Ordering;

        // Use M=7, K=64, N=65536 to trigger the thin-M path (K*N = 4.2M > 4M threshold).
        let (m, k, n) = (7, 64, 65536);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        let a = Owned::f32(&[m, k], &a_data);
        let b = Owned::f32(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[m, n]);

        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);

        let before = THIN_M_GEMM_TEST_HITS.load(Ordering::Relaxed);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = THIN_M_GEMM_TEST_HITS.load(Ordering::Relaxed);

        assert!(
            after > before,
            "f32 thin-M (M={m}, K={k}, N={n}) did not reach column-parallel NEON — \
             cblas_sgemm is likely intercepting this shape, which causes a ~2.5× \
             TTFT regression on large-vocab prefill"
        );
    }

    /// Numerics parity: thin-M NEON path produces the same results as cblas_sgemm
    /// within f32 tolerance.
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn f32_thin_m_numerics_match_cblas_reference() {
        // Shapes covering the lm_head (large N) pattern.
        let cases = [(7, 64, 65536), (4, 128, 50257), (16, 32, 100000)];

        for (m, k, n) in cases {
            let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin()).collect();
            let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.0071).cos()).collect();

            // Compute via thin-M NEON path (constant B triggers transpose cache).
            let a = Owned::f32(&[m, k], &a_data);
            let b = Owned::f32(&[k, n], &b_data);
            let mut out_neon = Owned::zeros_f32(&[m, n]);
            let mut kernel = MatMulKernel::default();
            kernel.set_constant_inputs(&[false, true]);
            kernel
                .execute(&[a.view(), b.view()], &mut [out_neon.view_mut()])
                .unwrap();

            // Reference: naive f64 accumulation.
            let mut ref_out = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for p in 0..k {
                        sum += a_data[i * k + p] as f64 * b_data[p * n + j] as f64;
                    }
                    ref_out[i * n + j] = sum;
                }
            }

            let neon_result = out_neon.to_f32();
            let max_ref: f64 = ref_out.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            let max_err: f64 = neon_result
                .iter()
                .zip(ref_out.iter())
                .map(|(&a, &b)| (a as f64 - b).abs())
                .fold(0.0f64, f64::max);
            let rel_err = if max_ref > 0.0 {
                max_err / max_ref
            } else {
                0.0
            };

            assert!(
                rel_err < 1e-5,
                "thin-M NEON path at [{m},{k}]×[{k},{n}]: relative error {rel_err:.2e} exceeds 1e-5"
            );
        }
    }

    /// #1091 measurement harness: same-binary A/B for the dense f32 GEMM.
    ///
    /// Drives [`gemm_with_backend`] with the vendored `Mlas` kernel (#1045) and
    /// the built-in `SimdX86` microkernel ([`x86_sgemm::sgemm_simd`]) over
    /// LLM-representative f32 GEMM shapes so the kernel gap on *this* host can be
    /// measured directly, rather than inherited from #1045's EPYC number. No
    /// int4 model exercises this path (they route through `MatMulNBits`), so a
    /// synthetic driver is the only way to exercise the f32 SGEMM at model
    /// scale.
    ///
    /// One backend per process (env `GEMM_AB_ARM=mlas|simd_packed|simd_gemv|
    /// generic`) so an external poller can attribute process CPU time and peak
    /// RSS to a single arm. The default `both` interleaves mlas + simd_packed +
    /// simd_gemv **in one process**, back to back per shape, so every arm shares
    /// the same machine conditions; it prints the ratio table and a control
    /// check. `GEMM_AB_ITERS` (default 30) is the min-timing repeat count;
    /// `GEMM_AB_SHAPES="m,k,n;m,k,n"` overrides the preset; `GEMM_AB_GENERIC=1`
    /// adds the (slow) Generic arm; `GEMM_AB_CONTROL_PCT` (default 5) sets the
    /// control drift threshold.
    ///
    /// ## The M>=2 rows are a built-in control
    ///
    /// The M=1 GEMV route only fires for `m == 1`; for `m >= 2` both the
    /// `simd_packed` and `simd_gemv` arms execute the *identical* packed kernel
    /// (`sgemm_simd_variant` ignores the flag). So on the M>=2 rows the
    /// `gemv/packed` ratio is a direct, zero-cost measurement of run-to-run
    /// noise: it must be ~1.0. If it drifts past `GEMM_AB_CONTROL_PCT`, the
    /// machine was not quiet and **no conclusion may be drawn from that run** —
    /// this makes "the box was busy" a measurement taken before concluding
    /// rather than an excuse offered after. Adopt this pattern for every A/B
    /// harness in the repo.
    ///
    /// Run: `cargo test -p onnx-runtime-ep-cpu --lib --release --features mlas \
    ///   bench_f32_gemm_ab -- --ignored --nocapture`.
    #[test]
    #[ignore = "manual perf harness (#1091); run explicitly"]
    fn bench_f32_gemm_ab() {
        use std::time::Instant;

        // Dispatch one kernel arm. Returns false if the arm is not compiled into
        // this build (e.g. `mlas` without the feature), so the harness degrades
        // cleanly. For `m >= 2`, `simd_packed` and `simd_gemv` are byte-for-byte
        // the same call — that identity is what makes the control valid.
        #[allow(unused_variables)]
        fn run_kind(
            kind: &str,
            a: &[f32],
            b: &[f32],
            c: &mut [f32],
            m: usize,
            k: usize,
            n: usize,
        ) -> bool {
            match kind {
                "generic" => {
                    gemm_with_backend(CpuBackend::Generic, a, b, c, m, k, n).unwrap();
                    true
                }
                #[cfg(feature = "mlas")]
                "mlas" => {
                    gemm_with_backend(CpuBackend::Mlas, a, b, c, m, k, n).unwrap();
                    true
                }
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                "simd_packed" => {
                    super::x86_sgemm::sgemm_simd_variant(a, b, c, m, k, n, false);
                    true
                }
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                "simd_gemv" => {
                    super::x86_sgemm::sgemm_simd_variant(a, b, c, m, k, n, true);
                    true
                }
                _ => false,
            }
        }

        // Qwen2.5-14B f32 shapes (hidden 5120, inter 13824, kv 1024, vocab
        // 152064). The M=1 rows are decode GEMVs; the M=128 rows are the prefill
        // control (unaffected by the M=1 route — see the doc comment).
        let default_shapes: Vec<(usize, usize, usize)> = vec![
            (1, 5120, 7168),    // decode: fused QKV proj (q 5120 + kv 2*1024)
            (1, 5120, 5120),    // decode: o_proj
            (1, 5120, 13824),   // decode: gate/up proj
            (1, 13824, 5120),   // decode: down proj
            (1, 5120, 152064),  // decode: lm_head
            (128, 5120, 5120),  // CONTROL prefill: o_proj
            (128, 5120, 13824), // CONTROL prefill: gate/up proj
            (128, 13824, 5120), // CONTROL prefill: down proj
        ];
        let shapes: Vec<(usize, usize, usize)> = match std::env::var("GEMM_AB_SHAPES") {
            Ok(spec) if !spec.trim().is_empty() => spec
                .split(';')
                .filter(|s| !s.trim().is_empty())
                .map(|s| {
                    let parts: Vec<usize> =
                        s.split(',').map(|p| p.trim().parse().unwrap()).collect();
                    (parts[0], parts[1], parts[2])
                })
                .collect(),
            _ => default_shapes,
        };
        let iters: usize = std::env::var("GEMM_AB_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);
        let control_pct: f64 = std::env::var("GEMM_AB_CONTROL_PCT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5.0);
        let arm = std::env::var("GEMM_AB_ARM").unwrap_or_else(|_| "both".to_string());
        let want_generic = std::env::var("GEMM_AB_GENERIC").is_ok();

        let mut state = 0x1234_5678_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
        };

        // Arm order (interleaved per shape). Isolated single-arm mode selects
        // just one, for external RSS / CPU-time attribution.
        let arms: Vec<&str> = if arm == "both" {
            let mut v = vec!["mlas", "simd_packed", "simd_gemv"];
            if want_generic {
                v.push("generic");
            }
            v
        } else {
            vec![arm.as_str()]
        };

        println!(
            "f32 GEMM A/B (#1091): threads={} iters={} arm={} control_pct={}",
            rayon::current_num_threads(),
            iters,
            arm,
            control_pct
        );

        // Collected control drift (|gemv/packed - 1|) over the M>=2 rows.
        let mut worst_control_drift = 0.0f64;
        let mut control_rows = 0usize;

        for &(m, k, n) in &shapes {
            let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| next()).collect();
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            // Interleave arms at the *iteration* level, not the arm level: run
            // every available arm once per round and keep each arm's min. A
            // clean moment (or a load spike) then lands on all arms' minima
            // together, so the M>=2 control ratio stays ~1.0 unless the machine
            // is *continuously* loaded. Per-arm windows (arm A x40, then B x40)
            // let sustained load during one arm's whole window bias that arm,
            // which is exactly the noise the control is meant to catch.
            let mut avail: Vec<&str> = Vec::new();
            let mut cbufs: Vec<Vec<f32>> = Vec::new();
            for &name in &arms {
                let mut c = vec![0.0f32; m * n];
                if run_kind(name, &a, &b, &mut c, m, k, n) {
                    avail.push(name); // arm compiled in; c now warmed up
                    cbufs.push(c);
                }
            }
            let mut best = vec![f64::INFINITY; avail.len()];
            for _ in 0..iters {
                for (i, &name) in avail.iter().enumerate() {
                    let t = Instant::now();
                    run_kind(name, &a, &b, &mut cbufs[i], m, k, n);
                    std::hint::black_box(&cbufs[i]);
                    best[i] = best[i].min(t.elapsed().as_secs_f64());
                }
            }
            let times: Vec<(&str, f64)> = avail.iter().copied().zip(best.iter().copied()).collect();
            let g = |s: f64| flops / s / 1e9;
            let get = |k: &str| times.iter().find(|(n, _)| *n == k).map(|(_, s)| *s);
            let is_control = m >= 2;
            print!(
                "  {}{m:>4}x{k:>6}x{n:>6}:",
                if is_control { "[CTL] " } else { "      " }
            );
            for (name, s) in &times {
                print!("  {name} {:>8.3}ms ({:>6.1}GF/s)", s * 1e3, g(*s));
            }
            if let (Some(mlas), Some(packed), Some(gemv)) =
                (get("mlas"), get("simd_packed"), get("simd_gemv"))
            {
                print!(
                    "  | packed/mlas={:.2}x gemv/mlas={:.2}x gemv/packed={:.2}x",
                    packed / mlas,
                    gemv / mlas,
                    gemv / packed
                );
                if is_control {
                    let drift = (gemv / packed - 1.0).abs();
                    worst_control_drift = worst_control_drift.max(drift);
                    control_rows += 1;
                }
            }
            println!();
        }

        if control_rows > 0 {
            let pct = worst_control_drift * 100.0;
            println!(
                "CONTROL: {} M>=2 rows, worst |gemv/packed - 1| = {:.1}% (threshold {:.1}%) -> {}",
                control_rows,
                pct,
                control_pct,
                if pct > control_pct {
                    "RUN NOT USABLE (machine not quiet; draw no conclusions)"
                } else {
                    "OK (run usable)"
                }
            );
        }
    }

    /// Independent scalar oracle for [`gemm_bt`]: `c[i][j] = Σ_p a[i][p]*bt[j][p]`
    /// with left-to-right accumulation, sharing no code with production.
    #[cfg(not(feature = "mlas"))]
    #[allow(clippy::needless_range_loop)]
    fn naive_bt(a: &[f32], bt: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * bt[j * k + p];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// Small deterministic LCG stream in `[-0.125, 0.125)`, matching the style of
    /// [`matmul_generic_block_boundaries_match_naive_reference`].
    #[cfg(not(feature = "mlas"))]
    fn lcg_stream(seed: u32) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.25
        }
    }

    /// `gemm_bt` must match the independent naive oracle across every shape that
    /// exercises the `4×2` interior tile, the `dot` edge/GEMV path, `k % 8`
    /// tails, odd `n`, leftover rows (`rows % 4`), the `m == 1` decode case and
    /// `n == 1`.
    #[cfg(not(feature = "mlas"))]
    #[test]
    fn gemm_bt_matches_naive_reference_across_shapes() {
        // (m, k, n): full tiles, tails, odd n, decode rows, single row/col.
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 1, 1),     // smallest
            (1, 7, 1),     // m=1, n=1, k tail (7 % 8)
            (1, 8, 5),     // m=1, exact k block, odd n
            (1, 9, 5),     // m=1, k tail 1, odd n
            (1, 1024, 64), // decode: single token, large k
            (1, 300, 300), // decode: rectangular
            (4, 16, 2),    // exactly one interior tile
            (4, 17, 2),    // interior tile + k tail
            (5, 16, 2),    // 1 full row band + 1 leftover row
            (7, 33, 9),    // 3 leftover rows, k tail, odd n
            (8, 8, 8),     // two full row bands
            (13, 64, 5),   // leftover rows + odd n
            (3, 100, 1),   // all leftover rows, single column
            (6, 1, 4),     // k = 1 (tail only, no 8-block)
            (4, 3, 2),     // k < 8 (all tail)
            (2, 5, 3),
            (129, 130, 131), // large: row-block / col-parallel + all edges
        ];
        for &(m, k, n) in SHAPES {
            let mut ra = lcg_stream(0x1234_5678 ^ (m as u32).wrapping_mul(2_654_435_761));
            let mut rb = lcg_stream(0x9E37_79B9 ^ (n as u32).wrapping_mul(40_503));
            let a: Vec<f32> = (0..m * k).map(|_| ra()).collect();
            let bt: Vec<f32> = (0..n * k).map(|_| rb()).collect();
            let want = naive_bt(&a, &bt, m, k, n);
            let mut got = vec![f32::NAN; m * n];
            gemm_bt(&a, &bt, &mut got, m, k, n).unwrap();
            for (idx, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-3,
                    "shape {m}x{k}x{n} index {idx}: got {g}, want {w}"
                );
            }
        }
    }

    /// The zero-extent guards must leave a correctly sized result and never
    /// panic on the chunking math: `m == 0` and `n == 0` produce empty products
    /// and `k == 0` produces all-zero dot products.
    #[cfg(not(feature = "mlas"))]
    #[test]
    fn gemm_bt_handles_zero_extents() {
        // m == 0: nothing to write.
        let mut c: Vec<f32> = Vec::new();
        gemm_bt(&[], &[0.1, 0.2, 0.3, 0.4], &mut c, 0, 4, 1).unwrap();
        assert!(c.is_empty());

        // n == 0: nothing to write.
        let mut c: Vec<f32> = Vec::new();
        gemm_bt(&[0.1, 0.2, 0.3, 0.4], &[], &mut c, 2, 2, 0).unwrap();
        assert!(c.is_empty());

        // k == 0: every dot product is the empty sum, i.e. exactly 0.
        let mut c = vec![7.0f32; 3 * 5];
        gemm_bt(&[], &[], &mut c, 3, 0, 5).unwrap();
        assert!(c.iter().all(|&v| v == 0.0), "k == 0 must zero every output");
    }

    /// Pin the portable scalar fallback [`gemm_bt_block_scalar`] (the non-x86 /
    /// no-AVX2 path) directly against the naive oracle, since the host normally
    /// takes the AVX2 path and would never exercise it.
    #[cfg(not(feature = "mlas"))]
    #[test]
    fn gemm_bt_block_scalar_matches_naive_reference() {
        const SHAPES: &[(usize, usize, usize)] = &[(1, 7, 1), (4, 17, 3), (7, 33, 9), (13, 5, 4)];
        for &(m, k, n) in SHAPES {
            let mut ra = lcg_stream(0x0BAD_F00D ^ (k as u32));
            let mut rb = lcg_stream(0xFEED_BEEF ^ (n as u32));
            let a: Vec<f32> = (0..m * k).map(|_| ra()).collect();
            let bt: Vec<f32> = (0..n * k).map(|_| rb()).collect();
            let want = naive_bt(&a, &bt, m, k, n);
            let mut got = vec![f32::NAN; m * n];
            gemm_bt_block_scalar(&a, &bt, &mut got, m, k, n);
            for (idx, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-3,
                    "scalar shape {m}x{k}x{n} index {idx}: got {g}, want {w}"
                );
            }
        }
    }

    /// The exact oracle the task asks for: `gemm_bt` on the native `[n][k]`
    /// weight layout must equal the *removed* production path — materialize the
    /// `[k][n]` transpose, then run the plain `gemm`. Values must match within a
    /// tight f32 tolerance (only the summation order differs).
    #[cfg(not(feature = "mlas"))]
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn gemm_bt_matches_transpose_then_gemm() {
        // Small-m shapes exercise the `dot` GEMV path; the large-m shapes force
        // the row-block driver so the `4×2` tile kernel (and its `k` tail) runs.
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 96, 40),
            (5, 7, 9),
            (12, 64, 20),
            (4, 16, 2),
            (40, 130, 34),
            (64, 96, 48),
        ];
        for &(m, k, n) in SHAPES {
            let mut ra = lcg_stream(0xA5A5_1234 ^ (m as u32));
            let mut rb = lcg_stream(0x5A5A_9876 ^ (n as u32));
            let a: Vec<f32> = (0..m * k).map(|_| ra()).collect();
            // `bt` is the expert layout `[n][k]` (i.e. `[out][in]`).
            let bt: Vec<f32> = (0..n * k).map(|_| rb()).collect();

            // Reference: the old path — transpose `bt` into `[k][n]`, plain gemm.
            let mut b_kn = vec![0.0f32; k * n];
            for out in 0..n {
                for inp in 0..k {
                    b_kn[inp * n + out] = bt[out * k + inp];
                }
            }
            let mut reference = vec![0.0f32; m * n];
            gemm(&a, &b_kn, &mut reference, m, k, n).unwrap();

            let mut got = vec![f32::NAN; m * n];
            gemm_bt(&a, &bt, &mut got, m, k, n).unwrap();
            for (idx, (&g, &r)) in got.iter().zip(&reference).enumerate() {
                assert!(
                    (g - r).abs() <= 1e-3,
                    "shape {m}x{k}x{n} index {idx}: gemm_bt {g}, transpose+gemm {r}"
                );
            }
        }
    }

    /// Deterministically exercise the `4×2` tile kernel and its `k` tail. A
    /// single-thread Rayon pool forces the row-block driver (the
    /// `native-threads=1` production path), so every `m >= 4` shape runs through
    /// `tile_4x2` regardless of the ambient pool width - the ambient pool would
    /// otherwise split small `m`/`n` into one-row / one-column tasks that never
    /// reach the tile. Checked against the independent naive oracle.
    #[cfg(not(feature = "mlas"))]
    #[test]
    fn gemm_bt_tile_kernel_matches_naive_single_threaded() {
        const SHAPES: &[(usize, usize, usize)] = &[
            (8, 17, 4),   // two tile bands, k tail, even n
            (12, 130, 6), // k tail 2
            (4, 96, 2),   // exactly one tile, no tail
            (9, 33, 11),  // leftover row + odd trailing column
            (16, 256, 8),
            // Large `k`. There is no K blocking: `tile_4x2` reduces the whole
            // of `k` in one pass, so these cover a long single-pass reduction
            // and its `k % 8` scalar tail, not a panelled accumulation.
            (16, 512, 8), // long reduction, no tail
            (12, 520, 6), // long reduction, no tail
            (8, 300, 4),  // long reduction with a k%8 tail
            (8, 259, 4),  // reduction whose tail is 3 lanes, all scalar
            (13, 577, 7), // leftover rows + odd trailing column + k tail
            // The only blocking in `gemm_bt` is over columns, and it engages
            // at `n * k > NC_KB` (65536). Every shape above stays under that,
            // so without these the column-panel sweep - the change that turned
            // the largest prefill shape from a regression into a win - would
            // be covered by nothing but the 629 MiB integration fixtures.
            (8, 2000, 64),  // several full column panels
            (10, 2001, 70), // column panels with a short final panel, k tail
            (1, 2000, 40),  // the decode row, multi-panel
        ];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            for &(m, k, n) in SHAPES {
                let mut ra = lcg_stream(0x00C0_FFEE ^ (m as u32));
                let mut rb = lcg_stream(0xB16B_00B5 ^ (n as u32));
                let a: Vec<f32> = (0..m * k).map(|_| ra()).collect();
                let bt: Vec<f32> = (0..n * k).map(|_| rb()).collect();
                let want = naive_bt(&a, &bt, m, k, n);
                let mut got = vec![f32::NAN; m * n];
                gemm_bt(&a, &bt, &mut got, m, k, n).unwrap();
                for (idx, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-3,
                        "tile shape {m}x{k}x{n} index {idx}: got {g}, want {w}"
                    );
                }
            }
        });
    }
}

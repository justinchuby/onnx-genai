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

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Graph, Node, broadcast_shapes, compute_contiguous_strides};
use rayon::prelude::*;

use super::check_arity;
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
        let trans_b = node
            .attr("transB")
            .and_then(Attribute::as_int)
            .unwrap_or(0)
            != 0;
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
#[derive(Default)]
pub(crate) struct MatMulPrepack {
    constant_inputs: [bool; 2],
    dense: [OnceLock<Vec<f32>>; 2],
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

impl MatMulPrepack {
    pub(crate) fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        for (index, is_constant) in self.constant_inputs.iter_mut().enumerate() {
            *is_constant = constant_inputs.get(index).copied().unwrap_or(false);
        }
    }

    pub(crate) fn dense<'a>(
        &'a self,
        index: usize,
        view: &'a TensorView<'_>,
    ) -> Result<Cow<'a, [f32]>> {
        if !self.constant_inputs[index] {
            return to_dense_f32_widen("MatMul", view);
        }
        if let Some(cached) = self.dense[index].get() {
            return Ok(Cow::Borrowed(cached));
        }

        match to_dense_f32_widen("MatMul", view)? {
            Cow::Borrowed(dense) => Ok(Cow::Borrowed(dense)),
            Cow::Owned(dense) => {
                let _ = self.dense[index].set(dense);
                Ok(Cow::Borrowed(
                    self.dense[index]
                        .get()
                        .expect("constant MatMul prepack was just initialized"),
                ))
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

        // x86 FP16 storage GEMV: the mirror of the Accelerate path above.
        // `try_matmul_half` packs both operands into cache-sized panels, which
        // pays for itself only when a panel of B is reused across several rows
        // of A. At M=1 there is no reuse, so the packing is pure overhead on a
        // memory-bound problem. Must precede `try_matmul_half` for the same
        // reason the Accelerate block does.
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
            && inputs[0].dtype == onnx_runtime_ir::DataType::Float16
            && inputs[1].dtype == onnx_runtime_ir::DataType::Float16
            && inputs[1].is_contiguous()
            && inputs[1].numel() == geom.k.saturating_mul(geom.n)
            && half_gemv::simd_available()
        {
            inputs[1].validate()?;
            let a_dense = self.prepack.dense(0, &inputs[0])?;
            if a_dense.len() == geom.k {
                // SAFETY: `inputs[1]` was just validated as a contiguous
                // Float16 view whose element count equals `k * n`. `f16` is
                // transparent over `u16`, so reading its storage as raw bit
                // patterns is sound, and the view outlives this call.
                let b_bits = unsafe {
                    std::slice::from_raw_parts(inputs[1].data_ptr::<u16>(), geom.k * geom.n)
                };
                let mut result = vec![0.0f32; geom.n];
                half_gemv::gemv_f16_kn(&a_dense, b_bits, &mut result, geom.k, geom.n);
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
        if let Some(result) = try_matmul_half(&inputs[0], &inputs[1], &geom)? {
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

/// Attempt the dedicated portable half GEMM path. Both operands must be
/// contiguous and have the same `Float16` or `BFloat16` dtype. The operands stay
/// in 16-bit storage until cache-panel packing, accumulation is always `f32`,
/// and the caller narrows once into the requested output dtype.
fn try_matmul_half(
    a: &TensorView,
    b: &TensorView,
    geom: &MatMulGeometry,
) -> Result<Option<Vec<f32>>> {
    use onnx_runtime_ir::DataType;

    let format = match (a.dtype, b.dtype) {
        (DataType::Float16, DataType::Float16) => HalfFormat::F16,
        (DataType::BFloat16, DataType::BFloat16) => HalfFormat::Bf16,
        _ => return Ok(None),
    };
    if !a.is_contiguous() || !b.is_contiguous() {
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
/// use the portable blocked half GEMM. The f16 path is always portable today.
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

/// Compute `A @ B` (numpy semantics: batched, broadcast leading dims, 1-D
/// operand promotion) into a dense row-major `Vec<f32>`.
///
/// Operands may be any float dtype (`f32`/`f16`/`bf16`/`f64`). Contiguous half
/// inputs use the blocked half GEMM; other low/medium precision layouts widen to
/// dense `f32`. Both routes accumulate in `f32`. Shared by [`MatMulKernel`] and
/// the fused `FusedMatMulBias` kernel.
pub(crate) fn matmul_dense(a: &TensorView, b: &TensorView) -> Result<Vec<f32>> {
    let geom = matmul_geometry(a, b)?;
    if let Some(result) = try_matmul_half(a, b, &geom)? {
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
    if let Some(result) = try_matmul_half(a, b, &geom)? {
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

    #[test]
    fn matmul_half_dispatch_matches_widened_reference_across_irregular_shapes() {
        use onnx_runtime_ir::DataType;

        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 127, 65),
            (3, 5, 7),
            (17, 130, 11),
            (5, 257, 2),
            (2, 0, 3),
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
                assert!(
                    try_matmul_half(&a.view(), &b.view(), &geometry)
                        .unwrap()
                        .is_some(),
                    "{dtype:?} should select the dedicated half GEMM"
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
        let dense_ptr = kernel.prepack.dense[1].get().unwrap().as_ptr();

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
        assert_eq!(kernel.prepack.dense[1].get().unwrap().as_ptr(), dense_ptr);
        assert!(kernel.prepack.dense[0].get().is_none());
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
        assert!(kernel.prepack.dense[1].get().is_some());
        assert!(kernel.prepack.dense[0].get().is_none());

        // Capture the cache pointer *before* the second execute so the
        // comparison below is a real guard: it proves the first call populated
        // the cache and the second reused it (rather than repopulating).
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let ptr_before = kernel.prepack.transposed_b_f16.get().unwrap().1.as_ptr();
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let ptr_before = kernel.prepack.dense[1].get().unwrap().as_ptr();

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
            kernel.prepack.dense[1].get().unwrap().as_ptr(),
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
            kernel.prepack.dense[1].get().is_none(),
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
        if half_gemv::simd_available() {
            assert!(
                kernel.prepack.transposed_b_f16.get().is_none(),
                "the GEMV must read B in place, not through a transpose cache"
            );
            assert!(
                kernel.prepack.dense[1].get().is_none(),
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
}

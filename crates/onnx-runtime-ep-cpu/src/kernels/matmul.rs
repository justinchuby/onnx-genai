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

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, broadcast_shapes, compute_contiguous_strides,
};
use rayon::prelude::*;
use smallvec::SmallVec;

use super::check_arity;
use super::half_gemm::{self, HalfFormat, MatrixLayout};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::half_gemv;
#[cfg(feature = "mlas")]
use super::weight_cache::WeightIdentity;
use super::weight_cache::{GrantedWeight, WeightCache, WeightCacheKey, WeightLoad, WeightSpec};
use super::weight_transpose;
use crate::CpuMemoryResources;
use crate::backend::CpuBackend;
use crate::dtype::{
    FloatElem, to_dense_f32_widen, to_dense_f32_widen_into, validate_f32_widen_input,
    write_dense_f32_narrow,
};
use crate::strided::{next_index, numel};
use crate::{dispatch_float, dispatch_ledger};

// MLAS-style packed SIMD f32 GEMM (the `SimdX86` backend). Kept in a sibling
// file but included here so `kernels/mod.rs` needs no edit; it is an internal
// perf detail of the MatMul hot path, not a new op.
#[path = "x86_sgemm.rs"]
pub(crate) mod x86_sgemm;

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

/// Conservative transpose retention forecast for direct constant operands.
/// All payloads are node-owned and obtain actual Host grants; this estimate
/// is not admission.
///
/// # What actually populates the cache
///
/// The cache holds one f32/f16 `N x K` transpose per **constant** (initializer)
/// weight that a kernel feeds through a node-owned transpose cache. The precise callers, and hence
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
/// * **x86 — `MatMul` / `FusedMatMulBias` / `Gemm transB=0` with a constant
///   16-bit `B`**: the decode GEMV transposes the weight once to `[N, K]` and
///   caches it at its stored width, `N * K * 2`. All three route through the
///   single `try_half_decode_gemv` helper, so all three are counted by the
///   same arm — see the note there on why keying on the shared helper rather
///   than on an op-name list is what stops this going stale.
///
/// Shape variants share the node's preparation owner, so one direct constant
/// layout needs no shape multiplier. Runtime-created logical views are not
/// represented by WeightRef and rely on actual grants when prepared.
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
    // MatMul promotes a vector B[K] to B[K,1]; Gemm requires a matrix.
    // Higher-rank B never enters the single-matrix transpose consumers.
    if dims.is_empty() || dims.len() > 2 || (dims.len() == 1 && node.op_type == "Gemm") {
        return None;
    }
    let elements = dims
        .iter()
        .fold(1u64, |n, &dim| n.saturating_mul(dim as u64));
    Some((elements, weight.dtype()))
}

/// Per-node contribution to [`weight_transpose_cache_predicted_bytes`]. Mirrors
/// exactly the kernel call sites documented there.
fn node_weight_transpose_cache_bytes(node: &Node, graph: &Graph) -> u64 {
    // Domain gate. `Gemm`/`MatMul` are default-domain ops; `FusedMatMulBias` is
    // emitted by the optimizer's `MatMul + Add` fusion into the *contrib*
    // domain (`optimizer.rs`, `MICROSOFT_DOMAIN`) and its kernel is registered
    // there (`kernels/mod.rs`). A blanket `!node.is_default_domain()` bail
    // therefore silently zeroed every fused projection -- the node this
    // predictor most needs to account for after #1702 was the one node it
    // could never see. Each op below states the domain it actually ships in.
    let contrib = node.domain == "com.microsoft";
    if !node.is_default_domain() && !contrib {
        return 0;
    }
    // All platforms: `Gemm` with `transB != 0` transposes a constant `B[N,K]`
    // to `[K,N]` and caches it as f32 (`gemm.rs:119`).
    // Whether this binary contains the x86 16-bit decode GEMV, which transposes
    // a constant `[K, N]` f16/bf16 weight once so both stored orders can share
    // the `[N, K]` kernel (`half_decode_gemv_dispatch`).
    let x86_half_decode = cfg!(any(target_arch = "x86", target_arch = "x86_64"));
    let half = |dtype: DataType| matches!(dtype, DataType::Float16 | DataType::BFloat16);

    if node.op_type == "Gemm" && node.is_default_domain() {
        let trans_b = node.attr("transB").and_then(Attribute::as_int).unwrap_or(0) != 0;
        let Some((numel, dtype)) = constant_b_numel(node, graph) else {
            return 0;
        };
        // All platforms: `transB != 0` transposes a constant `B[N,K]` to `[K,N]`
        // and caches it as f32 (`gemm.rs:119`).
        if trans_b {
            return numel.saturating_mul(4);
        }
        // x86: `transB == 0` holds the weight in the layout the decode GEMV
        // wants to leave behind, so a 16-bit constant is transposed to `[N, K]`
        // and cached at its stored width. Nothing is retained for other dtypes:
        // they never reach this route.
        if x86_half_decode && half(dtype) {
            return numel.saturating_mul(2);
        }
        return 0;
    }
    // Fused MatMul+bias+Relu reaches the shared dense-f32 compute path.
    // On Apple its optional f32 transpose is a 4-byte output even when the
    // original source needs widening. Same-half execution can retain less.
    if node.op_type == "FusedGemm" && contrib {
        return if cfg!(any(target_os = "macos", target_os = "ios")) {
            constant_b_numel(node, graph).map_or(0, |(elements, _)| elements.saturating_mul(4))
        } else {
            0
        };
    }
    // `MatMul` ships in the default domain, `FusedMatMulBias` in contrib.
    if (node.op_type == "MatMul" && node.is_default_domain())
        || (node.op_type == "FusedMatMulBias" && contrib)
    {
        let Some((numel, dtype)) = constant_b_numel(node, graph) else {
            return 0;
        };
        // Apple: both ops cache a constant `B`'s transpose in the Accelerate
        // paths (see the function docs).
        if cfg!(any(target_os = "macos", target_os = "ios")) {
            let elem = if dtype == DataType::Float16 { 2 } else { 4 };
            return numel.saturating_mul(elem);
        }
        // x86 elsewhere: `MatMul` and, since #1702, `FusedMatMulBias` both
        // reach the decode GEMV through the same `try_half_decode_gemv`
        // helper, so both retain the same `2 * numel` transpose. This arm
        // deliberately does *not* name one op and exclude the other: the
        // predictor's previous `node.op_type == "MatMul"` restriction was
        // correct only for as long as the two operators had different
        // routes, and it went silently stale the moment they were unified.
        // Keying on "does this node reach the shared helper" instead of on a
        // hardcoded op name is what keeps it from going stale again.
        if x86_half_decode && half(dtype) {
            return numel.saturating_mul(2);
        }
        return 0;
    }
    0
}

/// Predicted retained f32 materialization bytes for registered MatMul-prepack
/// consumers. Count each constant non-f32 operand once per graph node: the
/// executor shares one preparation owner across that node's shape variants.
/// Both operand slots and both fused forms use the same ownership contract.
/// Contiguous f32 constants borrow original storage and claim zero bytes.
///
/// Same-half kernels can avoid widening entirely, so this remains conservative
/// for those routes. This predicts dense materialization only, not native packed
/// panels, transposes or activation/output/workspace scratch. Custom strided f32
/// views are not represented in WeightRef; their required owned copies still
/// obtain actual grants at execution even though this graph estimate sees zero.
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
    // These are the registered consumers of MatMulPrepack's dense operand
    // preparation. Scalar/transpose/epilogue attributes do not change the
    // pre-operation widening bytes. The predictor and real-factory regression
    // cover the same resource-bearing family, including both fused forms.
    let uses_dense_prepack = match node.domain.as_str() {
        "" | "ai.onnx" => matches!(node.op_type.as_str(), "MatMul" | "Gemm"),
        "com.microsoft" => matches!(node.op_type.as_str(), "FusedMatMulBias" | "FusedGemm"),
        _ => false,
    };
    if !uses_dense_prepack {
        return 0;
    }
    let mut total = 0_u64;
    for index in 0..2 {
        total = total.saturating_add(dense_operand_cache_bytes(node, graph, index));
    }
    // Shape-specialized kernels share one node owner through the executor seam.
    total
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

/// A materialized copy is a function of the complete logical source view.
/// Immutable backing bytes do not imply immutable shape/strides: runtime views
/// and shape variants may interpret the same address and extent differently.
/// The output element width is part of the cache's type, not runtime metadata.
#[derive(Clone, PartialEq, Eq, Hash)]
struct MatMulCopyKey<const ELEMENT_BYTES: usize> {
    source: usize,
    dtype: DataType,
    shape: SmallVec<[usize; 4]>,
    strides: SmallVec<[i64; 4]>,
}

impl<const ELEMENT_BYTES: usize> MatMulCopyKey<ELEMENT_BYTES> {
    fn from_view(view: &TensorView<'_>) -> Self {
        Self {
            source: view.data_ptr::<u8>() as usize,
            dtype: view.dtype,
            shape: SmallVec::from_slice(view.shape),
            strides: SmallVec::from_slice(view.strides),
        }
    }
}

impl<const ELEMENT_BYTES: usize> WeightCacheKey for MatMulCopyKey<ELEMENT_BYTES> {
    fn weight_bytes(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |elements, &dim| elements.checked_mul(dim))
            .and_then(|elements| elements.checked_mul(ELEMENT_BYTES)).ok_or_else(||
            EpError::KernelFailed(format!("MatMul constant copy of {:?} with {ELEMENT_BYTES}-byte elements overflows usize; use addressable operand dimensions", self.shape)))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct HalfTransposeKey {
    source: usize,
    dtype: DataType,
    k: usize,
    n: usize,
}
impl WeightCacheKey for HalfTransposeKey {
    fn weight_bytes(&self) -> Result<usize> {
        self.k.checked_mul(self.n).and_then(|n| n.checked_mul(2)).ok_or_else(||
            EpError::KernelFailed(format!("MatMul half transpose [{}, {}] byte count overflows usize; use addressable weight dimensions", self.k, self.n)))
    }
}

#[cfg(feature = "mlas")]
#[derive(Clone, PartialEq, Eq, Hash)]
struct SgemmPackKey {
    source: SgemmPackSource,
    k: usize,
    n: usize,
}
#[cfg(feature = "mlas")]
#[derive(Clone, PartialEq, Eq, Hash)]
enum SgemmPackSource {
    // Borrowed initializer storage lives until the node owner is dropped.
    Original(MatMulCopyKey<4>),
    // Owned conversion buffers can retire/recycle independently of the panel.
    Materialized(WeightIdentity),
}
#[cfg(feature = "mlas")]
impl WeightCacheKey for SgemmPackKey {
    fn weight_bytes(&self) -> Result<usize> {
        sgemm_pack_plan(self.k, self.n).map(|plan| plan.owned_heap_bytes())
    }
}
#[cfg(feature = "mlas")]
fn sgemm_pack_plan(k: usize, n: usize) -> Result<mlas_sys::PackedBPlan> {
    mlas_sys::PackedBPlan::new(n, k).map_err(|source| EpError::Backend {
        operation: "MatMul MLAS B packing plan",
        source: Box::new(source),
    })
}

/// A converted constant carries its Host grant through the executing reader;
/// removing cache membership or dropping the kernel cannot revoke this payload.
pub(super) struct DenseOperand<'a>(DenseOperandStorage<'a>);

#[cfg(feature = "mlas")]
impl<'a> From<Cow<'a, [f32]>> for DenseOperand<'a> {
    fn from(values: Cow<'a, [f32]>) -> Self {
        Self(DenseOperandStorage::Direct(values))
    }
}

impl<'a> DenseOperand<'a> {
    pub(super) fn scratch_bytes(&self) -> Option<usize> {
        match &self.0 {
            DenseOperandStorage::Direct(Cow::Owned(values)) => {
                values.len().checked_mul(std::mem::size_of::<f32>())
            }
            DenseOperandStorage::Direct(Cow::Borrowed(_)) | DenseOperandStorage::Funded(_) => {
                Some(0)
            }
        }
    }
    pub(super) fn borrowed(values: &'a [f32]) -> Self {
        Self(DenseOperandStorage::Direct(Cow::Borrowed(values)))
    }
    pub(super) fn funded(values: Arc<GrantedWeight>) -> Self {
        Self(DenseOperandStorage::Funded(values))
    }
}

enum DenseOperandStorage<'a> {
    Direct(Cow<'a, [f32]>),
    Funded(Arc<GrantedWeight>),
}
impl std::ops::Deref for DenseOperand<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match &self.0 {
            DenseOperandStorage::Direct(values) => values,
            DenseOperandStorage::Funded(values) => values.as_slice(),
        }
    }
}
impl AsRef<[f32]> for DenseOperand<'_> {
    fn as_ref(&self) -> &[f32] {
        self
    }
}

/// Identity is the original logical operand, never a recyclable conversion
/// allocation. A transpose can retire independently of its widened source.
#[derive(Clone, PartialEq, Eq, Hash)]
struct F32TransposeKey {
    source: MatMulCopyKey<4>,
    k: usize,
    n: usize,
}
impl WeightCacheKey for F32TransposeKey {
    fn weight_bytes(&self) -> Result<usize> {
        self.source.weight_bytes()
    }
}

struct F32TransposeHint {
    key: F32TransposeKey,
    // Flat-vector identity retains only a small Arc control block after the
    // payload and grant retire; it cannot retain an inline matrix allocation.
    reader: std::sync::Weak<GrantedWeight>,
}

fn transpose_source_geometry(source: &TensorView<'_>, k: usize, n: usize) -> bool {
    k.checked_mul(n).is_some()
        && (source.shape == [k, n].as_slice() || (n == 1 && source.shape == [k].as_slice()))
}

/// Widen directly into the transposed destination, never through a second
/// matrix. Tiled source coordinates preserve the exact set of addressed elements.
fn transpose_source_into(
    source: &TensorView<'_>,
    k: usize,
    n: usize,
    output: &mut [f32],
) -> Result<()> {
    // SAFETY: f32 and MaybeUninit<f32> have identical layouts; output is
    // exclusive and f32 has no destructor. The common fill initializes all
    // elements before returning a readable slice.
    let uninit =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), output.len()) };
    transpose_source_uninit(source, k, n, uninit).map(|_| ())
}

pub(super) fn transpose_source_uninit<'a>(
    source: &TensorView<'_>,
    k: usize,
    n: usize,
    output: &'a mut [std::mem::MaybeUninit<f32>],
) -> Result<&'a mut [f32]> {
    validate_f32_widen_input("CPU operand transpose source", source)?;
    if !transpose_source_geometry(source, k, n) || k.checked_mul(n) != Some(output.len()) {
        return Err(EpError::KernelFailed(format!(
            "CPU operand transpose [{k}, {n}] has source shape {:?} and {} output elements; supply a matching matrix or promoted vector and complete output storage",
            source.shape,
            output.len()
        )));
    }
    if output.is_empty() {
        return Ok(&mut []);
    }
    if source.dtype == DataType::Float32
        && source.is_contiguous()
        && (source.data_ptr::<u8>() as usize).is_multiple_of(std::mem::align_of::<f32>())
    {
        // SAFETY: validated host-contiguous aligned f32 storage has exactly
        // the checked matrix extent; source backing bounds are the EP contract.
        let values = unsafe { std::slice::from_raw_parts(source.data_ptr::<f32>(), output.len()) };
        return weight_transpose::transpose_map_uninit(k, n, output, |i, j| values[i * n + j]);
    }
    let (sr, sc) = if source.shape.len() == 1 {
        (source.strides[0] as isize, 0)
    } else {
        (source.strides[0] as isize, source.strides[1] as isize)
    };
    dispatch_float!(source.dtype,"CPU operand transpose conversion",T=>{
        let origin=source.data_ptr::<T>() as usize;
        weight_transpose::transpose_map_uninit(k,n,output,|i,j| {
            // SAFETY: the validated original host view addresses this logical
            // coordinate. Transposition changes traversal, never source bounds;
            // the caller supplies backing bounds under the TensorView contract.
            unsafe{(origin as *const T).offset(i as isize*sr+j as isize*sc).read_unaligned()}.to_f32()
        })
    })
}

/// Per-kernel policy over node-owned immutable MatMul preparation.
///
/// Contiguous f32 constants already have the ideal representation, so they stay
/// zero-copy and need no owned cache entry.
///
/// An Arc node owner keeps granted payloads shared across shape variants and
/// fused operators. Source identities are checked against the actual operand;
/// each prepared payload independently retains its grant through its readers.
pub(crate) struct MatMulPrepack {
    retain_caches: bool,
    constant_inputs: [bool; 2],
    weight_memory: crate::DerivedWeightMemory,
    weights: Arc<MatMulWeights>,
}

/// Mutable cache membership is shared by one session node's shape variants.
/// Kernel geometry and constant flags remain local to each compiled variant.
struct MatMulWeights {
    /// Session-lifetime widened-f32 copy of a constant operand, governed by the
    /// memory plan (#1056): filled only under an admitting verdict, and reports
    /// its own held bytes via the caches' actual retained payload bytes. Only populated
    /// when the operand is not already contiguous f32 (see [`MatMulPrepack::dense`]).
    dense: [WeightCache<MatMulCopyKey<4>>; 2],
    #[cfg(feature = "mlas")]
    packed_b: WeightCache<SgemmPackKey, mlas_sys::PackedB>,
    /// Reclaimable MLAS-packed f32 panels derived from a constant **f16** B weight.
    ///
    /// The widened f32 copy is a temporary: it is consumed by `PackedBPlan::pack`
    /// and dropped, so this costs one packed panel rather than the packed panel
    /// plus a permanent 4*K*N dense copy the way `dense[1]` + `packed_b` would.
    #[cfg(feature = "mlas")]
    packed_b_from_half: WeightCache<SgemmPackKey, mlas_sys::PackedB>,
    /// Lazily-computed transpose of the B weight matrix for the Accelerate
    /// column-parallel GEMV path, with the key it was computed for. Only
    /// populated for constant (model weight) inputs.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "ios")),
        allow(dead_code, reason = "consumed only by the Apple Accelerate paths")
    )]
    transposed_b: WeightCache<F32TransposeKey>,
    // A weak last-hit slot avoids constructing/hashing the full layout key on
    // repeated decode. It owns no payload or quota and cannot block reclamation.
    f32_transpose_hint: std::sync::Mutex<Option<F32TransposeHint>>,
    /// Reclaimable native-width half transposes keyed by original source,
    /// interpretation and matrix geometry. Executing readers keep their grant.
    #[cfg_attr(
        not(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_os = "macos",
            target_os = "ios"
        )),
        allow(
            dead_code,
            reason = "native half GEMV consumes this on x86 or Apple; portable tests cover its ownership"
        )
    )]
    half_transposes: WeightCache<HalfTransposeKey, Vec<u16>>,
    /// Lazily-computed contiguous f16 copy of B for non-contiguous weight
    /// matrices (e.g. lm_head vocab projection stored column-major in the ONNX
    /// model). Stores raw u16 bit patterns in row-major K×N layout. Only
    /// populated for constant Float16 inputs where the original is
    /// non-contiguous.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "ios")),
        allow(dead_code, reason = "consumed only by the Apple Accelerate paths")
    )]
    contiguous_b_f16: WeightCache<MatMulCopyKey<2>, Vec<u16>>,
}

struct SharedMatMulWeights {
    weights: Arc<MatMulWeights>,
    memory: crate::DerivedWeightMemory,
    constant_inputs: [bool; 2],
}

impl Default for MatMulPrepack {
    fn default() -> Self {
        Self::new(CpuMemoryResources::default())
    }
}

impl MatMulPrepack {
    /// Preserve load-time transpose warming inside the owning kernel’s policy.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) fn prepare_constants(
        &mut self,
        constants: &[Option<onnx_runtime_ep_api::KernelConstantInput<'_>>],
    ) -> Result<bool> {
        if self.retain_caches
            && self.constant_inputs[1]
            && let Some(Some(weight)) = constants.get(1)
            && weight.shape.len() == 2
        {
            let (k, n) = (weight.shape[0], weight.shape[1]);
            if k == 0 || n == 0 {
                return Ok(false);
            }
            let element_size = match weight.dtype {
                DataType::Float16 => 2,
                DataType::Float32
                    if (k as u64).saturating_mul(n as u64)
                        > accelerate_gemm::THIN_M_LARGE_B_THRESHOLD as u64 =>
                {
                    4
                }
                _ => return Ok(false),
            };
            if k.checked_mul(n)
                .and_then(|numel| numel.checked_mul(element_size))
                != Some(weight.bytes.len())
                || !(weight.bytes.as_ptr() as usize).is_multiple_of(element_size)
            {
                return Ok(false);
            }
            let Ok(n_stride) = i64::try_from(n) else {
                return Ok(false);
            };
            let strides = [n_stride, 1];
            let view = TensorView::new(
                onnx_runtime_ep_api::DevicePtr(weight.bytes.as_ptr().cast()),
                weight.dtype,
                weight.shape,
                &strides,
                onnx_runtime_ir::DeviceId::cpu(),
            );
            if weight.dtype == DataType::Float16 {
                return Ok(self
                    .transposed_b_half(&view, k, n, HalfFormat::F16)?
                    .is_some());
            } else {
                return Ok(self.transposed_b(&view, k, n)?.is_some());
            }
        }
        Ok(false)
    }

    pub(crate) fn new(resources: CpuMemoryResources) -> Self {
        let retain_caches = resources.cache_retention().is_admitted();
        // Preserve the legacy family retention verdict. The block/NBits dense
        // expansion ceiling is not an implicit new limit on ordinary MatMul.
        let limit = if retain_caches { usize::MAX } else { 0 };
        Self {
            retain_caches,
            constant_inputs: [false; 2],
            weight_memory: resources.weight_memory().clone(),
            weights: Arc::new(MatMulWeights {
                dense: [
                    WeightCache::new(resources.weight_memory().clone(), limit),
                    WeightCache::new(resources.weight_memory().clone(), limit),
                ],
                #[cfg(feature = "mlas")]
                packed_b: WeightCache::new(resources.weight_memory().clone(), limit),
                #[cfg(feature = "mlas")]
                packed_b_from_half: WeightCache::new(resources.weight_memory().clone(), limit),
                transposed_b: WeightCache::new(resources.weight_memory().clone(), limit),
                f32_transpose_hint: std::sync::Mutex::new(None),
                half_transposes: WeightCache::new(resources.weight_memory().clone(), limit),
                contiguous_b_f16: WeightCache::new(resources.weight_memory().clone(), limit),
            }),
        }
    }
}

impl MatMulPrepack {
    pub(crate) fn shareable_constant_state(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        (self.retain_caches && self.constant_inputs.iter().any(|&constant| constant)).then(|| {
            Arc::new(SharedMatMulWeights {
                weights: self.weights.clone(),
                memory: self.weight_memory.clone(),
                constant_inputs: self.constant_inputs,
            }) as Arc<dyn std::any::Any + Send + Sync>
        })
    }

    pub(crate) fn adopt_shareable_constant_state(
        &mut self,
        state: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<bool> {
        let Ok(shared) = Arc::downcast::<SharedMatMulWeights>(state) else {
            return Ok(false);
        };
        if !self.retain_caches || !self.constant_inputs.iter().any(|&constant| constant) {
            return Ok(false);
        }
        if self.constant_inputs != shared.constant_inputs {
            return Err(EpError::KernelFailed(format!(
                "MatMul preparation cannot adopt constant flags {:?} into {:?}; construct shape variants with the same immutable operand flags",
                shared.constant_inputs, self.constant_inputs
            )));
        }
        if !self.weight_memory.shares_source_with(&shared.memory) {
            return Err(EpError::KernelFailed("MatMul preparation cannot adopt weights from a different derived-weight grant source; construct this node's shape variants from clones of the same CpuMemoryResources".into()));
        }
        self.weights = shared.weights.clone();
        Ok(true)
    }

    /// Total bytes the `dense` widened-f32 caches currently hold across both
    /// operand slots -- the figure a prediction is checked against (#1056).
    #[cfg(test)]
    pub(crate) fn dense_live_bytes(&self) -> u64 {
        self.weights
            .dense
            .iter()
            .map(|cache| cache.stats().1 as u64)
            .sum()
    }

    #[cfg(test)]
    fn dense_payload(&self, index: usize) -> Option<Arc<GrantedWeight>> {
        self.weights.dense[index].test_first()
    }

    #[cfg(test)]
    fn dense_is_filled(&self, index: usize) -> bool {
        self.weights.dense[index].stats().0 != 0
    }

    pub(crate) fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        for (index, is_constant) in self.constant_inputs.iter_mut().enumerate() {
            *is_constant = constant_inputs.get(index).copied().unwrap_or(false);
        }
    }

    pub(super) fn retains_constant(&self, index: usize) -> bool {
        self.retain_caches && self.constant_inputs[index]
    }

    /// Borrow unchanged contiguous f32 storage. Materialized constants obtain
    /// their complete f32 payload grant before conversion, whether retained or
    /// call-owned. Dynamic activation conversion remains a separate scratch scope.
    pub(super) fn dense<'a>(
        &self,
        index: usize,
        view: &'a TensorView<'_>,
    ) -> Result<DenseOperand<'a>> {
        if !self.constant_inputs[index] || (view.dtype == DataType::Float32 && view.is_contiguous())
        {
            return to_dense_f32_widen("MatMul", view)
                .map(|values| DenseOperand(DenseOperandStorage::Direct(values)));
        }
        validate_f32_widen_input("MatMul constant operand", view)?;
        if view.shape.contains(&0) {
            return Ok(DenseOperand::borrowed(&[]));
        }
        let key = MatMulCopyKey::<4>::from_view(view);
        if self.retain_caches
            && let Some(hit) = self.weights.dense[index].get(&key)
        {
            return Ok(DenseOperand(DenseOperandStorage::Funded(hit)));
        }
        let spec = if self.retain_caches {
            WeightSpec::cached(&key)?
        } else {
            WeightSpec::transient(key.weight_bytes()?)
        };
        match self.weights.dense[index].load_group(&[spec], |_, output| {
            to_dense_f32_widen_into("MatMul constant operand", view, output)
        })? {
            WeightLoad::Ready(mut loaded) => Ok(DenseOperand(DenseOperandStorage::Funded(
                loaded.weights.remove(0),
            ))),
            WeightLoad::Declined(source) => Err(EpError::Backend {
                operation: if index == 0 {
                    "MatMul constant A widening admission"
                } else {
                    "MatMul constant B widening admission"
                },
                source: Box::new(source),
            }),
        }
    }

    /// Optional node-owned f32 transpose, keyed by the complete original view
    /// and requested geometry. Conversion writes directly into admitted storage;
    /// the result needs no intermediate dense copy or source-payload owner.
    pub(super) fn transposed_b(
        &self,
        source: &TensorView<'_>,
        k: usize,
        n: usize,
    ) -> Result<Option<Arc<GrantedWeight>>> {
        if !self.constant_inputs[1] || !self.retain_caches {
            return Ok(None);
        }
        validate_f32_widen_input("MatMul f32 transpose source", source)?;
        {
            let hint = self
                .weights
                .f32_transpose_hint
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(F32TransposeHint { key, reader }) = hint.as_ref()
                && key.k == k
                && key.n == n
                && key.source.source == source.data_ptr::<u8>() as usize
                && key.source.dtype == source.dtype
                && key.source.shape.as_slice() == source.shape
                && key.source.strides.as_slice() == source.strides
                && let Some(reader) = reader.upgrade()
            {
                return Ok(Some(reader));
            }
        }
        if !transpose_source_geometry(source, k, n) || k == 0 || n == 0 {
            return Ok(None);
        }
        let key = F32TransposeKey {
            source: MatMulCopyKey::from_view(source),
            k,
            n,
        };
        if let Some(hit) = self.weights.transposed_b.get(&key) {
            self.remember_f32_transpose(key, &hit);
            return Ok(Some(hit));
        }
        let loaded = self
            .weights
            .transposed_b
            .load_group(&[WeightSpec::cached(&key)?], |_, output| {
                transpose_source_into(source, k, n, output)
            })
            .map_err(|source| EpError::Backend {
                operation: "MatMul f32 transpose preparation",
                source: Box::new(source),
            })?;
        match loaded {
            WeightLoad::Ready(mut loaded) => {
                let reader = loaded.weights.remove(0);
                self.remember_f32_transpose(key, &reader);
                Ok(Some(reader))
            }
            WeightLoad::Declined(_) => Ok(None),
        }
    }

    fn remember_f32_transpose(&self, key: F32TransposeKey, reader: &Arc<GrantedWeight>) {
        let mut hint = self
            .weights
            .f32_transpose_hint
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // With one entry, recency cannot affect victim choice. Multi-layout
        // nodes use the canonical map so every hit keeps its LRU metadata.
        *hint = (self.weights.transposed_b.has_single_entry()).then(|| F32TransposeHint {
            key,
            reader: Arc::downgrade(reader),
        });
    }

    /// Gemm's retained-B route requires an owned layout. Cache refusal still
    /// funds a call-owned copy before building; dynamic/nonretained operands
    /// instead use the execution workspace.
    pub(super) fn required_transposed_b(
        &self,
        source: &TensorView<'_>,
        k: usize,
        n: usize,
    ) -> Result<Arc<GrantedWeight>> {
        if let Some(hit) = self.transposed_b(source, k, n)? {
            return Ok(hit);
        }
        validate_f32_widen_input("Gemm required B transpose", source)?;
        if !transpose_source_geometry(source, k, n) {
            return Err(EpError::KernelFailed(format!(
                "Gemm B transpose [{k}, {n}] disagrees with source shape {:?}; supply a matching matrix or promoted vector",
                source.shape
            )));
        }
        let bytes=k.checked_mul(n).and_then(|elements|elements.checked_mul(4)).ok_or_else(||EpError::KernelFailed(format!("Gemm B transpose [{k}, {n}] byte count overflows usize; use addressable matrix dimensions")))?;
        let loaded = self
            .weights
            .transposed_b
            .load_group(&[WeightSpec::transient(bytes)], |_, output| {
                transpose_source_into(source, k, n, output)
            })
            .map_err(|source| EpError::Backend {
                operation: "Gemm required B transpose preparation",
                source: Box::new(source),
            })?;
        match loaded {
            WeightLoad::Ready(mut loaded) => Ok(loaded.weights.remove(0)),
            WeightLoad::Declined(source) => Err(EpError::Backend {
                operation: "Gemm required B transpose admission",
                source: Box::new(source),
            }),
        }
    }

    #[cfg(test)]
    pub(super) fn f32_transpose_stats(&self) -> (usize, usize) {
        self.weights.transposed_b.stats()
    }

    /// Cached native-width half transposes, owned once per node and reclaimable
    /// when idle. An executing Arc reader retains its own payload and grant.
    #[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
    pub(crate) fn retained_transpose_bytes(&self) -> u64 {
        self.weights.half_transposes.stats().1 as u64
    }

    /// The requested interpretation must match the source dtype. Only complete
    /// contiguous immutable matrices are eligible; refusal runs no builder and
    /// leaves the existing untransposed computation available.
    #[cfg_attr(
        not(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_os = "macos",
            target_os = "ios"
        )),
        allow(
            dead_code,
            reason = "native half GEMV consumes this on x86 or Apple; portable tests cover its ownership"
        )
    )]
    pub(super) fn transposed_b_half(
        &self,
        view: &TensorView,
        k: usize,
        n: usize,
        format: HalfFormat,
    ) -> Result<Option<Arc<GrantedWeight<Vec<u16>>>>> {
        let expected = match format {
            HalfFormat::F16 => DataType::Float16,
            HalfFormat::Bf16 => DataType::BFloat16,
        };
        if !self.constant_inputs[1]
            || !self.retain_caches
            || view.dtype != expected
            || !view.is_contiguous()
        {
            return Ok(None);
        }
        view.validate()?;
        if !view.device.is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "MatMul constant half transpose cannot read {:?} storage on {:?}; explicitly transfer the matrix to host-accessible storage before preparing CPU weights",
                view.dtype, view.device
            )));
        }
        let Some(elements) = k.checked_mul(n) else {
            return Ok(None);
        };
        if elements == 0 || view.numel() != elements {
            return Ok(None);
        }
        let key = HalfTransposeKey {
            source: view.data_ptr::<u16>() as usize,
            dtype: view.dtype,
            k,
            n,
        };
        if let Some(hit) = self.weights.half_transposes.get(&key) {
            return Ok(Some(hit));
        }
        let loaded = self
            .weights
            .half_transposes
            .load_group(&[WeightSpec::cached(&key)?], |_, output| {
                // SAFETY: validated host-accessible contiguous half storage, with
                // exactly the checked matrix extent, stays live for this build.
                let source =
                    unsafe { std::slice::from_raw_parts(view.data_ptr::<u16>(), elements) };
                weight_transpose::transpose_row_major_into(source, k, n, output)
            })
            .map_err(|source| EpError::Backend {
                operation: "MatMul constant half transpose preparation",
                source: Box::new(source),
            })?;
        match loaded {
            WeightLoad::Ready(mut loaded) => Ok(Some(loaded.weights.remove(0))),
            WeightLoad::Declined(_) => Ok(None),
        }
    }

    /// Returns a cached contiguous f16 copy of a non-contiguous B weight.
    ///
    /// When a model stores a weight matrix with non-row-major strides (e.g.
    /// the lm_head vocab projection stored column-major), `try_matmul_half`
    /// skips it because BNNS requires contiguous input.  This method
    /// materialises a contiguous K×N row-major copy. When retention is admitted,
    /// it caches that copy for the session; otherwise the caller owns a transient.
    /// Admitted subsequent prefill calls avoid both:
    ///   - the element-by-element `to_dense_f32_widen` (~1 s at 136 M elements)
    ///   - the 2× memory of f16→f32 widening
    ///
    /// Source address, dtype, shape and strides identify each logical view of
    /// immutable backing storage at this slot (#845). Both retained
    /// and transient copies hold a Host grant until their last reader drops.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "ios")),
        allow(dead_code, reason = "consumed only by the Apple Accelerate paths")
    )]
    pub(super) fn contiguous_b_f16(
        &self,
        b_view: &TensorView,
    ) -> Result<Option<Arc<GrantedWeight<Vec<u16>>>>> {
        use onnx_runtime_ir::DataType;
        if !self.constant_inputs[1] || b_view.dtype != DataType::Float16 || b_view.is_contiguous() {
            return Ok(None); // already contiguous or not cacheable
        }
        validate_f32_widen_input("MatMul constant f16 contiguous preparation", b_view)?;
        let key = MatMulCopyKey::<2>::from_view(b_view);
        if self.retain_caches
            && let Some(hit) = self.weights.contiguous_b_f16.get(&key)
        {
            return Ok(Some(hit));
        }
        let spec = if self.retain_caches {
            WeightSpec::cached(&key)?
        } else {
            WeightSpec::transient(key.weight_bytes()?)
        };
        let loaded = self
            .weights
            .contiguous_b_f16
            .load_group(&[spec], |_, out| {
                let shape = b_view.shape;
                let strides = b_view.strides;
                let base = b_view.data_ptr::<u16>();
                if out.is_empty() {
                    return Ok(());
                }

                // Optimised 2-D path (covers all weight matrices).
                if shape.len() == 2 {
                    let (rows, cols) = (shape[0], shape[1]);
                    let (sr, sc) = (strides[0] as isize, strides[1] as isize);
                    use rayon::prelude::*;
                    let threads = rayon::current_num_threads();
                    let rows_per_thread = rows.div_ceil(threads).max(1);
                    // SAFETY: `base` points into the model's mmap'd weight
                    // buffer which is immutable for the session lifetime.
                    // Concurrent source reads are immutable (strides may alias);
                    // each Rayon task writes a disjoint output chunk.
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
                                    *d = unsafe {
                                        row_base.offset(j as isize * sc).read_unaligned()
                                    };
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
                        *o = unsafe { base.offset(off).read_unaligned() };
                        for d in (0..shape.len()).rev() {
                            idx[d] += 1;
                            if idx[d] < shape[d] {
                                break;
                            }
                            idx[d] = 0;
                        }
                    }
                }
                Ok(())
            })?;
        match loaded {
            WeightLoad::Ready(mut loaded) => Ok(Some(loaded.weights.remove(0))),
            WeightLoad::Declined(reason) => Err(EpError::Backend {
                operation: "MatMul constant f16 contiguous preparation admission",
                source: Box::new(reason),
            }),
        }
    }

    /// MLAS-packed B built by widening a constant f16 weight to f32 once.
    ///
    /// Returns `None` when retention is declined or B is non-constant.
    /// Packing retains a derived weight, so it follows the same immutable
    /// retention verdict as densification. An activation would have to be
    /// re-widened and re-packed on every call, which is the cost this exists to
    /// remove, and caching it would be wrong the moment the activation changed.
    #[cfg(feature = "mlas")]
    fn packed_b_from_half(
        &self,
        view: &TensorView<'_>,
        k: usize,
        n: usize,
    ) -> Result<Option<Arc<GrantedWeight<mlas_sys::PackedB>>>> {
        if !self.retain_caches || !self.constant_inputs[1] {
            return Ok(None);
        }
        validate_f32_widen_input("MatMul MLAS half-weight packing", view)?;
        let elements = view.shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b)).ok_or_else(|| EpError::KernelFailed("MatMul MLAS half-weight shape element count overflow; use addressable B dimensions".into()))?;
        if Some(elements) != k.checked_mul(n) {
            return Err(EpError::KernelFailed(format!(
                "MatMul MLAS half-weight packing: shape {:?} does not describe K={k} N={n}; supply one complete B matrix",
                view.shape
            )));
        }
        let key = SgemmPackKey {
            source: SgemmPackSource::Original(MatMulCopyKey::from_view(view)),
            k,
            n,
        };
        if let Some(packed) = self.weights.packed_b_from_half.get(&key) {
            return Ok(Some(packed));
        }
        let plan = sgemm_pack_plan(k, n)?;
        let bytes = elements.checked_mul(4).ok_or_else(|| EpError::KernelFailed(
            "MatMul MLAS half-weight conversion byte count overflow; use addressable B dimensions".into()
        ))?;
        let peak = plan.owned_heap_bytes().checked_add(bytes).ok_or_else(|| EpError::KernelFailed(
            "MatMul MLAS half-weight packed-plus-conversion peak overflows usize; reduce B dimensions".into()
        ))?;
        let spec = WeightSpec::cached(&key)?.with_build_peak(peak);
        match self
            .weights
            .packed_b_from_half
            .load_group_owned(&[spec], |_| {
                let mut widened = super::decoded_projection::zeroed_f32(
                    "MatMul MLAS half-weight packing",
                    "admitted conversion payload",
                    elements,
                )?;
                to_dense_f32_widen_into("MatMul MLAS half-weight packing", view, &mut widened)?;
                let packed = plan.pack(&widened).map_err(|source| EpError::Backend {
                    operation: "MatMul MLAS half-weight packing",
                    source: Box::new(source),
                })?;
                drop(widened);
                Ok(packed)
            })? {
            WeightLoad::Ready(mut loaded) => Ok(Some(loaded.weights.remove(0))),
            // Packing is optional. The existing unpacked route remains valid;
            // a declined build never allocates conversion or panel payloads.
            WeightLoad::Declined(_) => Ok(None),
        }
    }

    #[cfg(test)]
    pub(crate) fn half_pack_is_built(&self) -> bool {
        #[cfg(feature = "mlas")]
        {
            self.weights.packed_b_from_half.stats().0 != 0
        }
        #[cfg(not(feature = "mlas"))]
        {
            false
        }
    }

    #[cfg(feature = "mlas")]
    fn packed_b(
        &self,
        b: &DenseOperand<'_>,
        k: usize,
        n: usize,
    ) -> Result<Option<Arc<GrantedWeight<mlas_sys::PackedB>>>> {
        if !self.retain_caches || !self.constant_inputs[1] {
            return Ok(None);
        }
        let expected = n.checked_mul(k).ok_or_else(|| EpError::Backend {
            operation: "MatMul MLAS constant B packing",
            source: Box::new(mlas_sys::SgemmPackError::Geometry {
                n,
                k,
                reason: "logical element count overflows usize",
            }),
        })?;
        if b.len() != expected {
            return Err(EpError::Backend {
                operation: "MatMul MLAS constant B packing",
                source: Box::new(mlas_sys::SgemmPackError::Length {
                    expected,
                    actual: b.len(),
                }),
            });
        }
        let source = match &b.0 {
            DenseOperandStorage::Funded(owner) => {
                SgemmPackSource::Materialized(WeightIdentity::of(owner))
            }
            DenseOperandStorage::Direct(Cow::Borrowed(values)) => {
                let stride = i64::try_from(n).map_err(|_| EpError::KernelFailed(format!(
                    "MatMul MLAS constant B packing: N={n} cannot be represented as a logical row stride; use addressable B dimensions"
                )))?;
                SgemmPackSource::Original(MatMulCopyKey {
                    source: values.as_ptr() as usize,
                    dtype: DataType::Float32,
                    shape: SmallVec::from_slice(&[k, n]),
                    strides: SmallVec::from_slice(&[stride, 1]),
                })
            }
            // A call-owned, ungoverned conversion has no immutable lifetime
            // identity. Optional packing must use the unpacked route instead.
            DenseOperandStorage::Direct(Cow::Owned(_)) => return Ok(None),
        };
        let key = SgemmPackKey { source, k, n };
        if let Some(packed) = self.weights.packed_b.get(&key) {
            return Ok(Some(packed));
        }
        let plan = sgemm_pack_plan(k, n)?;
        match self
            .weights
            .packed_b
            .load_group_owned(&[WeightSpec::cached(&key)?], |_| {
                plan.pack(b).map_err(|source| EpError::Backend {
                    operation: "MatMul MLAS constant B packing",
                    source: Box::new(source),
                })
            })? {
            WeightLoad::Ready(mut loaded) => Ok(Some(loaded.weights.remove(0))),
            WeightLoad::Declined(_) => Ok(None),
        }
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
pub struct MatMulFactory {
    resources: CpuMemoryResources,
}
impl MatMulFactory {
    pub fn new(resources: CpuMemoryResources) -> Self {
        Self { resources }
    }
}
impl Default for MatMulFactory {
    fn default() -> Self {
        Self::new(CpuMemoryResources::default())
    }
}

impl KernelFactory for MatMulFactory {
    fn create(&self, _node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let flops = match (input_shapes.first(), input_shapes.get(1)) {
            (Some(a), Some(b)) => super::flops::matmul_flops(a, b),
            _ => None,
        };
        Ok(Box::new(MatMulKernel::new(self.resources.clone(), flops)))
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
    gemm_packed(&a_dense, packed.payload(), &mut result, m, k, n)?;
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
    dispatch_ledger::record_with(|| {
        dispatch_ledger::Observation::gemm(
            dispatch_ledger::KernelFamily::MatMulF32,
            ledger_backend(backend),
            "f32",
            m,
            n,
            k,
        )
    });
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
                accelerate_gemm::neon_gemv_parallel(a, b, c, k, n)?;
            } else {
                accelerate_gemm::sgemm(a, b, c, m, k, n)?;
            }
            Ok(())
        }
        _ => {
            gemm_generic(a, b, c, m, k, n);
            Ok(())
        }
    }
}

/// A resolved NT implementation for an available CPU backend. Private fields
/// keep unsupported backends out of callers' executable capability values.
#[derive(Clone, Copy)]
pub(crate) struct NtGemm {
    backend: CpuBackend,
    requires_retained_weights: bool,
}
impl NtGemm {
    pub(crate) fn resolve(backend: CpuBackend) -> Option<Self> {
        let reuse = match backend {
            // Apple NT's demonstrated advantage is amortized NK preparation.
            // Preserve NN for transient weights: initial real-execute probes
            // show no uniform transient benefit, including slower wall cells.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            CpuBackend::Accelerate => Some(true),
            #[cfg(feature = "mlas")]
            CpuBackend::Mlas => Some(false),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            CpuBackend::SimdX86 if crate::backend::has_simd_x86() => Some(false),
            _ => None,
        };
        reuse.map(|requires_retained_weights| Self {
            backend,
            requires_retained_weights,
        })
    }

    pub(crate) fn requires_retained_weights(self) -> bool {
        self.requires_retained_weights
    }

    /// C[M,N] = A[M,K] @ B[N,K]^T. Backend/layout reduction orders can differ;
    /// finite f64-oracle coverage does not imply bit-identical f32 outputs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        self,
        a: &[f32],
        b_nk: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        dispatch_ledger::record_with(|| {
            dispatch_ledger::Observation::gemm(
                dispatch_ledger::KernelFamily::MatMulF32,
                ledger_backend(self.backend),
                "f32",
                m,
                n,
                k,
            )
        });
        match self.backend {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            CpuBackend::Accelerate => accelerate_gemm::sgemm_nt(a, b_nk, c, m, k, n),
            #[cfg(feature = "mlas")]
            CpuBackend::Mlas => {
                mlas_sys::sgemm(false, true, m, n, k, 1.0, a, k, b_nk, k, 0.0, c, n, true);
                Ok(())
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            CpuBackend::SimdX86 => {
                x86_sgemm::sgemm_simd_nt(a, b_nk, c, m, k, n);
                Ok(())
            }
            _ => {
                let _ = (a, b_nk, c, m, k, n);
                Err(EpError::KernelFailed(format!(
                    "resolved NT GEMM contains unsupported backend {:?}; construct capabilities through NtGemm::resolve",
                    self.backend
                )))
            }
        }
    }
}

/// How a [`CpuBackend`] reads in the dispatch ledger: MLAS is the only variant
/// whose arithmetic is not ours, and it only exists in a research build.
pub(crate) fn ledger_backend(backend: CpuBackend) -> dispatch_ledger::Backend {
    match backend {
        #[cfg(feature = "mlas")]
        CpuBackend::Mlas => dispatch_ledger::Backend::Mlas,
        _ => dispatch_ledger::Backend::Native,
    }
}

// Register microkernel tile: MR rows x NR cols of C accumulated in registers.
const MR: usize = 4;
const NR: usize = 4;
// Cache block over the K dimension so a panel of B stays resident in L1/L2
// while a strip of C accumulates across it.
const KC: usize = 256;
const MAX_MC: usize = 64;

/// A borrowed column window of B. Row stride belongs to the source matrix;
/// columns belong to this window and to each output row. No element is copied.
#[derive(Clone, Copy)]
struct GemmBPanel<'a> {
    values: &'a [f32],
    row_stride: usize,
    columns: usize,
}

/// Output storage exposes disjoint rows. Dense row blocks and borrowed column
/// strips run the same arithmetic without materializing a second output matrix.
trait GemmOutputRows {
    fn row_mut(&mut self, row: usize) -> &mut [f32];
}
struct DenseGemmOutput<'a> {
    values: &'a mut [f32],
    columns: usize,
}
impl GemmOutputRows for DenseGemmOutput<'_> {
    #[inline]
    fn row_mut(&mut self, row: usize) -> &mut [f32] {
        &mut self.values[row * self.columns..(row + 1) * self.columns]
    }
}
impl GemmOutputRows for [&mut [f32]] {
    #[inline]
    fn row_mut(&mut self, row: usize) -> &mut [f32] {
        self[row]
    }
}

#[cfg(test)]
static GENERIC_BORROWED_COLS_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

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
    if k == 0 {
        c.fill(0.0);
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
            gemm_block(
                a_block,
                GemmBPanel {
                    values: b,
                    row_stride: n,
                    columns: n,
                },
                &mut DenseGemmOutput {
                    values: c_block,
                    columns: n,
                },
                rows,
                k,
            );
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
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c.fill(0.0);
        return;
    }
    let cols_per_strip = n.div_ceil(threads.max(1)).max(1);
    let strip_count = n.div_ceil(cols_per_strip);
    // Split each caller-owned output row once. Rust owns every mutable cell in
    // exactly one strip; only row references (not B/C elements) are allocated.
    let mut strips: Vec<Vec<&mut [f32]>> =
        (0..strip_count).map(|_| Vec::with_capacity(m)).collect();
    for row in c.chunks_mut(n) {
        for (strip, columns) in strips.iter_mut().zip(row.chunks_mut(cols_per_strip)) {
            strip.push(columns);
        }
    }
    strips
        .into_par_iter()
        .enumerate()
        .for_each(|(index, mut output)| {
            let start = index * cols_per_strip;
            let columns = cols_per_strip.min(n - start);
            #[cfg(test)]
            GENERIC_BORROWED_COLS_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            gemm_block(
                a,
                GemmBPanel {
                    values: &b[start..],
                    row_stride: n,
                    columns,
                },
                output.as_mut_slice(),
                m,
                k,
            );
        });
}

/// Compute `c_block[rows,n] = a_block[rows,k] @ b[k,n]` (overwrite) for one row
/// block, blocking over K and register-tiling MR x NR.
fn gemm_block<O: GemmOutputRows + ?Sized>(
    a: &[f32],
    b: GemmBPanel<'_>,
    c: &mut O,
    rows: usize,
    k: usize,
) {
    let n = b.columns;
    for row in 0..rows {
        c.row_mut(row).fill(0.0);
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
                micro_kernel(a, b, c, k, i, j, kk, kc, mr, nr);
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
fn micro_kernel<O: GemmOutputRows + ?Sized>(
    a: &[f32],
    b: GemmBPanel<'_>,
    c: &mut O,
    k: usize,
    i: usize,
    j: usize,
    kk: usize,
    kc: usize,
    mr: usize,
    nr: usize,
) {
    let mut acc = [[0.0f32; NR]; MR];
    for p in kk..kk + kc {
        let brow = &b.values[p * b.row_stride + j..p * b.row_stride + j + nr];
        for (ii, acc_row) in acc.iter_mut().enumerate().take(mr) {
            let aik = a[(i + ii) * k + p];
            for (jj, acc_v) in acc_row.iter_mut().enumerate().take(nr) {
                *acc_v += aik * brow[jj];
            }
        }
    }
    for (ii, acc_row) in acc.iter().enumerate().take(mr) {
        let c_row = &mut c.row_mut(i + ii)[j..j + nr];
        for (jj, cv) in c_row.iter_mut().enumerate().take(nr) {
            *cv += acc_row[jj];
        }
    }
}

/// Compute one transposed-B product without entering the Rayon scheduler.
/// Used inside an already partitioned operation to keep scratch ownership and
/// task counts bounded rather than nesting parallel matrix multiplies.
pub(crate) fn gemm_bt_serial(
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
    rows: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    if rows.checked_mul(k) != Some(a.len())
        || n.checked_mul(k) != Some(bt.len())
        || rows.checked_mul(n) != Some(c.len())
    {
        return Err(EpError::KernelFailed(format!(
            "serial transposed GEMM geometry mismatch: A={} B={} C={} for rows={rows}, k={k}, n={n}; supply exact contiguous matrix extents",
            a.len(),
            bt.len(),
            c.len()
        )));
    }
    if rows == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        c.fill(0.0);
        return Ok(());
    }
    #[cfg(feature = "mlas")]
    mlas_sys::sgemm(false, true, rows, n, k, 1.0, a, k, bt, k, 0.0, c, n, false);
    #[cfg(not(feature = "mlas"))]
    gemm_bt_block(a, bt, c, rows, k, n);
    Ok(())
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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if threads > 1 && gemm_bt_prefers_packed(m, k, n) && gemm_bt_packed_mt_fits(m, k, threads) {
        gemm_bt_panel_parallel(a, bt, c, m, k, n, threads);
        return Ok(());
    }
    let mc = if threads <= 1 {
        // The packed kernel pays a per-panel pack that is repaid across the row
        // bands sweeping it, so hand the single-threaded driver the whole row
        // extent instead of capping at `MAX_MC` — one pack for `m` rows rather
        // than `m.div_ceil(MAX_MC)` packs of 64 rows each.
        if gemm_bt_prefers_packed(m, k, n) {
            m
        } else {
            MAX_MC.min(m)
        }
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

/// Raw pointer to `c` wrapped so Rayon can hand disjoint column panels to
/// worker threads. Each panel writes a disjoint set of columns of every row, so
/// no two tasks touch the same element — the `Send`/`Sync` impls are sound
/// given that invariant, which [`gemm_bt_panel_parallel`] upholds. Same pattern
/// and same argument as `x86_sgemm::CPtr`.
#[cfg(not(feature = "mlas"))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
struct PanelCPtr(*mut f32);
#[cfg(not(feature = "mlas"))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for PanelCPtr {}
#[cfg(not(feature = "mlas"))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for PanelCPtr {}
#[cfg(not(feature = "mlas"))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl PanelCPtr {
    /// Take `self` by value so a closure captures the whole wrapper rather than
    /// its raw field, which is what the `Send`/`Sync` impls are written for.
    #[inline]
    fn as_ptr(self) -> *mut f32 {
        self.0
    }
}

/// Multi-threaded packed `gemm_bt`: the column panel is the unit of work, so
/// each panel is packed exactly once by whichever thread owns it.
///
/// This is the single-threaded packed algorithm with its outer loop handed to
/// the pool, and it is the multi-threaded packed path that measured well. The
/// obvious alternative — pack a panel serially and parallelise the row bands
/// under it — shares one pack across threads but pays a fork-join per panel and
/// makes the pack a partly serial term; measured across
/// `{phi35, qwen3, mixtral} x {96,148,256,365} rows x {2..32} threads` it lost
/// to the unpacked driver at every panel width tried (geomean 0.61-1.04, worse
/// the wider the pool). Distributing whole panels instead keeps the pack fully
/// parallel and the sweep barrier-free.
///
/// Ownership of the panel buffer is per-worker and explicit:
///
/// * **Identity** — the buffer is created by `for_each_init`, so it belongs to
///   one worker lane of one invocation. Nothing is shared between calls, cached,
///   or held in a static, so concurrent sessions cannot collide.
/// * **Capacity** — it is allocated once per lane at exactly `nc*k` and reused
///   across the panels that lane draws, never regrown. A final short panel uses
///   a prefix; the sweep is told how many micro-panels are live, so it never
///   reads past it.
/// * **No duplicate pack** — a lane packs only panels it owns, and panel
///   indices are disjoint across lanes, so the total pack work is exactly
///   `n16*k` element moves however many threads run. Reusing one buffer per
///   lane is what keeps that true without allocating per panel.
///
/// Both remainders (`n % PNR` columns, `m % PMR` rows) are swept after the
/// panels, in one parallel pass over output rows.
#[cfg(not(feature = "mlas"))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn gemm_bt_panel_parallel(
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) {
    use gemm_bt_avx2 as kern;
    use rayon::prelude::*;
    let pmr = kern::band_rows();
    let pnr = kern::panel_stride();
    let n16 = n - n % pnr;
    let r6 = m - m % pmr;
    let nc = gemm_bt_panel_width(k, n16, threads);
    let panels = n16.div_ceil(nc);

    let cp = PanelCPtr(c.as_mut_ptr());
    (0..panels).into_par_iter().for_each_init(
        || {
            (
                vec![0.0f32; nc * k],
                crate::trace::worker_span("MatMul.gemm_bt.panel"),
            )
        },
        |(packed, _span), pi| {
            let c_ptr = cp.as_ptr();
            let j0 = pi * nc;
            let jend = (j0 + nc).min(n16);
            // SAFETY: `available()` is checked by the caller. `bt` holds `n*k`
            // floats and `jend <= n16 <= n`; `packed` is `nc*k` long and
            // `jend - j0 <= nc`.
            unsafe { kern::pack_panel(bt.as_ptr(), packed.as_mut_ptr(), k, j0, jend) };
            // SAFETY: `a` holds `m*k` floats and `r6 <= m`. This task writes
            // only columns `j0..jend` of `c`, which no other panel index
            // covers, and the row and column remainders are swept afterwards,
            // so the aliasing `PanelCPtr` allows is never realised.
            unsafe {
                kern::sweep_band(
                    a.as_ptr(),
                    packed.as_ptr(),
                    c_ptr,
                    r6,
                    k,
                    n,
                    j0,
                    (jend - j0) / pnr,
                )
            };
        },
    );

    if n16 < n || r6 < m {
        c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            // Rows the panels covered need only their column tail; rows past
            // the band extent were never touched and need the whole row.
            let start = if i < r6 { n16 } else { 0 };
            if start < n {
                // SAFETY: `available()` is checked by the caller. `a` holds
                // `m*k` floats and `i < m`; `bt` holds `n*k`; `c_row` is this
                // closure's exclusive row of `n` floats.
                unsafe {
                    kern::dot_row_tail(
                        a.as_ptr().add(i * k),
                        bt.as_ptr(),
                        c_row.as_mut_ptr(),
                        k,
                        n,
                        start,
                    )
                }
            }
        });
    }
}

/// Panel width for [`gemm_bt_panel_parallel`]: wide enough to keep the pack
/// amortised, narrow enough that there is a panel for every thread.
///
/// The single-threaded width is a pure cache choice — as much of `b` as fits
/// beside the sweeping `a` band in a private L2. On a pool it is also the unit
/// of parallelism, so a `b` that would be two L2-sized panels cannot occupy
/// eight threads. Splitting `n16` evenly across the pool and taking whichever
/// is narrower keeps both properties.
#[cfg(not(feature = "mlas"))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn gemm_bt_panel_width(k: usize, n16: usize, threads: usize) -> usize {
    let pnr = gemm_bt_avx2::panel_stride();
    let cache = gemm_bt_avx2::panel_cols(k, n16, gemm_bt_avx2::PANEL_BYTES);
    let even = (n16 / threads.max(1) / pnr) * pnr;
    cache.min(even.max(pnr))
}

/// Whether this shape will reach the packed micro-kernel inside
/// [`gemm_bt_block`]. Used by [`gemm_bt`] to widen the single-threaded row block
/// so the pack is amortized once over the whole row extent; always `false` where
/// the packed kernel does not exist (non-x86, or x86 without AVX2+FMA).
#[cfg(not(feature = "mlas"))]
fn gemm_bt_prefers_packed(rows: usize, k: usize, n: usize) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        gemm_bt_avx2::available() && gemm_bt_avx2::packed_is_profitable(rows, k, n)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = (rows, k, n);
        false
    }
}

/// Whether the multi-threaded packed driver is the right choice for this shape.
///
/// Two independent conditions, both measured on the
/// `{phi35, qwen3, mixtral} x {96,148,256,365} rows x {2,4,8,16,32} threads`
/// grid:
///
/// * **A band for every thread.** The unpacked driver slices `m` into `MR`-row
///   blocks and can keep more threads busy on a short `m` than a `PMR`-row band
///   sweep can, so packing there would trade a real loss (idle cores) for a
///   speculative win.
/// * **Enough `k` to amortise a panel.** Below `k = 2048` the packed path lost
///   on that grid (`mixtral` `k=1024` at 0.90-1.02, `qwen3` `k=768` at
///   0.59-1.09) while every `k >= 2048` shape won. The threshold is where one
///   micro-panel reaches a quarter of the panel budget
///   (`PANEL_BYTES / (PNR * 4 * 4)`): a short `k` makes each micro-panel cheap
///   enough that the per-panel costs — the pack loop, the rayon task, the
///   `nc*k` buffer — stop being amortised, and the unpacked kernel's strided
///   `b` reads are already cache-resident at that size, so there is little left
///   for the pack to buy. Admitting `k >= 1024` scored marginally better on the
///   grid geomean (1.194 vs 1.177) but turned `mixtral`'s first GEMM into a
///   consistent small loss, which is the wrong trade for a shape that is always
///   on the critical path.
#[cfg(not(feature = "mlas"))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn gemm_bt_packed_mt_fits(m: usize, k: usize, threads: usize) -> bool {
    k >= gemm_bt_avx2::PANEL_BYTES / (gemm_bt_avx2::panel_stride() * 16)
        && m / gemm_bt_avx2::band_rows() >= threads
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
            // pointer either kernel forms from `(rows, k, n)` is in bounds.
            unsafe {
                if gemm_bt_avx2::packed_is_profitable(rows, k, n) {
                    gemm_bt_avx2::block_packed(a.as_ptr(), bt.as_ptr(), c.as_mut_ptr(), rows, k, n);
                } else {
                    gemm_bt_avx2::block(a.as_ptr(), bt.as_ptr(), c.as_mut_ptr(), rows, k, n);
                }
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

    /// Rows per micro-tile in the packed kernel: six `a` rows are broadcast
    /// against two `b` vectors, so twelve of the sixteen `ymm` registers hold
    /// accumulators and the remaining four cover the two `b` loads plus the
    /// rotating broadcast.
    const PMR: usize = 6;
    /// Columns per micro-tile in the packed kernel: two eight-lane vectors.
    const PNR: usize = 16;
    /// Target bytes for one packed `b` panel: sized to sit in a core's private
    /// L2 alongside the `PMR`-row `a` band that sweeps against it. It is a
    /// target, not a bound — one micro-panel is `PNR*k*4` bytes and the `nc`
    /// floor below never goes narrower than that, so `k > 8192` overshoots the
    /// budget. Real MoE `k` (≤ 6400 across the benchmarked experts) stays
    /// inside it.
    pub const PANEL_BYTES: usize = 512 * 1024;
    /// Row bands a packed panel must be swept by before the pack pays for
    /// itself. Packing a column panel costs `nc*k` strided element moves that
    /// are repaid only across the `rows/PMR` bands that reuse it, so a row
    /// block with too few bands pays the transpose and never earns it back.
    ///
    /// The value is chosen so that [`PACK_MIN_ROWS`] lands above `MAX_MC`,
    /// which keeps the *row-block* driver — whose blocks would each re-pack the
    /// same panel — on the unpacked path by construction, single- or
    /// multi-threaded. Threads reach the packed kernel through
    /// [`super::gemm_bt_panel_parallel`] instead, which hands whole column
    /// panels to the pool so each is still packed exactly once.
    const PACK_MIN_BANDS: usize = 12;
    /// Fewest rows the packed kernel will accept, i.e. [`PACK_MIN_BANDS`] full
    /// micro-tile bands.
    ///
    /// The gate is on the **row block**, not on `m`, and it deliberately sits
    /// above [`super::MAX_MC`]. The row-block driver slices `m` into blocks
    /// capped at `MAX_MC` and each block packs independently, so a block that
    /// could pack would re-pack a panel its neighbours already built; keeping
    /// the bar above that cap makes "the panel is packed once per sweep" a
    /// structural property rather than a lucky one. The row-block driver
    /// therefore reaches the packed kernel only single-threaded, where it is
    /// handed the whole row extent and there are no neighbours. The
    /// multi-threaded packed path is [`super::gemm_bt_panel_parallel`], which
    /// is selected before the row-block driver and distributes whole column
    /// panels, so no panel is packed twice there either.
    const PACK_MIN_ROWS: usize = PMR * PACK_MIN_BANDS;
    const _: () = assert!(
        PACK_MIN_ROWS > super::MAX_MC,
        "the packed kernel must stay out of the row-block driver's blocks, \
         which are capped at MAX_MC and would each re-pack the same panel"
    );

    /// Whether the packed kernel is worth its pack cost for this shape.
    #[inline]
    pub fn packed_is_profitable(rows: usize, k: usize, n: usize) -> bool {
        rows >= PACK_MIN_ROWS && n >= PNR && k >= 8
    }

    /// Columns per packed panel for this `k`, given a byte budget: a multiple of
    /// [`PNR`], never wider than the tiled column extent `n16`, never narrower
    /// than one micro-panel.
    #[inline]
    pub fn panel_cols(k: usize, n16: usize, panel_bytes: usize) -> usize {
        let bytes_per_col = k.saturating_mul(4).max(1);
        let mut nc = (panel_bytes / bytes_per_col / PNR) * PNR;
        if nc < PNR {
            nc = PNR;
        }
        if nc > n16 {
            nc = n16;
        }
        nc
    }

    /// Rows per micro-tile band, exposed so drivers can cut row blocks on a
    /// band boundary.
    #[inline]
    pub fn band_rows() -> usize {
        PMR
    }

    /// Columns per packed micro-panel, exposed for the same reason.
    #[inline]
    pub fn panel_stride() -> usize {
        PNR
    }

    /// Sweep one row block against an already-packed panel.
    ///
    /// `a` and `c` point at the block's own first row, so the caller may hand
    /// over a sub-slice; `n` is still the full output row stride and `j0` the
    /// panel's absolute first column. `rows` must be a multiple of [`PMR`] —
    /// the caller owns the row remainder, because with a shared panel the
    /// remainder is not per-block work.
    ///
    /// # Safety
    /// AVX2+FMA available. `a` is valid for `rows*k` reads, `panel` for
    /// `sub_panels*k*PNR` reads, and rows `0..rows`, columns
    /// `j0..j0+sub_panels*PNR` of `c` (row stride `n`) are writable.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sweep_band(
        a: *const f32,
        panel: *const f32,
        c: *mut f32,
        rows: usize,
        k: usize,
        n: usize,
        j0: usize,
        sub_panels: usize,
    ) {
        unsafe {
            let mut i = 0usize;
            while i + PMR <= rows {
                for s in 0..sub_panels {
                    micro_6x16(a, panel.add(s * k * PNR), c, k, n, i, j0 + s * PNR);
                }
                i += PMR;
            }
        }
    }

    /// Fill columns `j0..n` of one output row with plain dot products.
    ///
    /// The shared-pack driver owns both remainders (the `n % PNR` column tail
    /// and the `m % PMR` row tail) because with one panel shared by the pool
    /// neither is per-block work; this is the unit it parallelises them over.
    ///
    /// # Safety
    /// AVX2+FMA available. `a_row` is valid for `k` reads, `bt` for `n*k`
    /// reads, and `c_row` for `n` writes.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_row_tail(
        a_row: *const f32,
        bt: *const f32,
        c_row: *mut f32,
        k: usize,
        n: usize,
        j0: usize,
    ) {
        unsafe {
            for j in j0..n {
                *c_row.add(j) = dot(a_row, bt.add(j * k), k);
            }
        }
    }

    /// Copy `bt` rows `j0..jend` into micro-panel-major order:
    /// `packed[s*k*PNR + p*PNR + t] = bt[(j0 + s*PNR + t)*k + p]`.
    ///
    /// The kernel then reads each micro-panel as one contiguous `k*PNR` run, so
    /// the `b` side of the inner loop is two sequential cache lines per `k` step
    /// instead of the `k`-strided gather the unpacked layout would force. This
    /// is the only transpose in the path, it is bounded by the panel (not the
    /// whole expert bank), and it is reused by every row band in the sweep.
    ///
    /// # Safety
    /// `bt` is valid for `(jend)*k` reads; `packed` is valid for
    /// `(jend - j0) * k` writes; `jend - j0` is a multiple of [`PNR`].
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn pack_panel(bt: *const f32, packed: *mut f32, k: usize, j0: usize, jend: usize) {
        unsafe {
            let sub_panels = (jend - j0) / PNR;
            for s in 0..sub_panels {
                let dst = packed.add(s * k * PNR);
                for t in 0..PNR {
                    let src = bt.add((j0 + s * PNR + t) * k);
                    for p in 0..k {
                        *dst.add(p * PNR + t) = *src.add(p);
                    }
                }
            }
        }
    }

    /// One `PMR x PNR` micro-tile against a packed `b` micro-panel.
    ///
    /// Unlike [`tile_4x2`], the SIMD lanes hold `c` columns rather than `k`
    /// partials, so there is **no horizontal reduction at all** — the twelve
    /// accumulators are stored straight out. Per `k` step the tile issues two
    /// contiguous vector loads plus six broadcasts against twelve FMAs (1.5
    /// FMAs per load, against `tile_4x2`'s 1.33), which is what closes the gap
    /// to a packed-panel reference GEMM on prefill-shaped `m`.
    ///
    /// # Safety
    /// AVX2+FMA available. Rows `i..i+PMR` of `a` (`a` is `_*k`) are readable,
    /// `panel` is valid for `k*PNR` reads, and columns `j..j+PNR` of rows
    /// `i..i+PMR` of `c` (`c` is `_*n`) are writable.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn micro_6x16(
        a: *const f32,
        panel: *const f32,
        c: *mut f32,
        k: usize,
        n: usize,
        i: usize,
        j: usize,
    ) {
        unsafe {
            let mut acc = [_mm256_setzero_ps(); 12];
            let a0 = a.add(i * k);
            for p in 0..k {
                let b0 = _mm256_loadu_ps(panel.add(p * PNR));
                let b1 = _mm256_loadu_ps(panel.add(p * PNR + 8));
                for r in 0..PMR {
                    let av = _mm256_set1_ps(*a0.add(r * k + p));
                    acc[2 * r] = _mm256_fmadd_ps(av, b0, acc[2 * r]);
                    acc[2 * r + 1] = _mm256_fmadd_ps(av, b1, acc[2 * r + 1]);
                }
            }
            for r in 0..PMR {
                let row = c.add((i + r) * n + j);
                _mm256_storeu_ps(row, acc[2 * r]);
                _mm256_storeu_ps(row.add(8), acc[2 * r + 1]);
            }
        }
    }

    /// Packed-panel counterpart of [`block`], selected by
    /// [`packed_is_profitable`].
    ///
    /// Loop order is column panel -> row band -> micro-panel, so one packed `b`
    /// panel is built once and swept by every row band before it is evicted.
    /// The row remainder (`rows % PMR`) and the column remainder (`n % PNR`) run
    /// as full-`k` [`dot`]s, which is the same code the unpacked path uses for
    /// its edges and needs no packing to be efficient.
    ///
    /// # Safety
    /// AVX2+FMA available. `a` is valid for `rows*k` reads, `bt` for `n*k`
    /// reads, and `c` for `rows*n` writes, all row-major.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn block_packed(
        a: *const f32,
        bt: *const f32,
        c: *mut f32,
        rows: usize,
        k: usize,
        n: usize,
    ) {
        unsafe {
            let n16 = n - n % PNR;
            let r6 = rows - rows % PMR;

            let nc = panel_cols(k, n16, PANEL_BYTES);

            let mut packed = vec![0.0f32; nc * k];
            let mut j0 = 0usize;
            while j0 < n16 {
                let jend = (j0 + nc).min(n16);
                pack_panel(bt, packed.as_mut_ptr(), k, j0, jend);
                sweep_band(a, packed.as_ptr(), c, r6, k, n, j0, (jend - j0) / PNR);
                j0 = jend;
            }

            // Column remainder for the rows the micro-tile covered.
            for i in 0..r6 {
                for j in n16..n {
                    *c.add(i * n + j) = dot(a.add(i * k), bt.add(j * k), k);
                }
            }
            // Row remainder, all columns.
            for i in r6..rows {
                for j in 0..n {
                    *c.add(i * n + j) = dot(a.add(i * k), bt.add(j * k), k);
                }
            }
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

/// Decode directly from row-major half storage when optional preparation is
/// unavailable. Only the activation and output are materialized; no weight
/// widening or second weight grant is needed under Host memory pressure.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(super) fn untransposed_f16_decode(
    prepack: &MatMulPrepack,
    a: &TensorView,
    b: &TensorView,
    k: usize,
    n: usize,
) -> Result<Option<Vec<f32>>> {
    if b.dtype != DataType::Float16 || !b.is_contiguous() {
        return Ok(None);
    }
    b.validate()?;
    if !b.device.is_host_accessible() {
        return Err(EpError::KernelFailed(format!(
            "CPU half decode cannot read weights on {:?}; explicitly transfer B to host-accessible storage",
            b.device
        )));
    }
    if k.checked_mul(n) != Some(b.numel()) {
        return Ok(None);
    }
    let activation = prepack.dense(0, a)?;
    if activation.len() != k {
        return Ok(None);
    }
    // SAFETY: validated contiguous host half storage has exactly k*n elements.
    let bits = unsafe { std::slice::from_raw_parts(b.data_ptr::<u16>(), b.numel()) };
    let mut result = vec![0.0; n];
    result
        .par_chunks_mut(64)
        .enumerate()
        .for_each(|(tile, output)| {
            let start = tile * 64;
            for (p, &av) in activation.iter().enumerate() {
                let row = &bits[p * n + start..p * n + start + output.len()];
                for (slot, &weight) in output.iter_mut().zip(row) {
                    *slot = half::f16::from_bits(weight).to_f32().mul_add(av, *slot);
                }
            }
        });
    Ok(Some(result))
}

impl Kernel for MatMulKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.prepack.set_constant_inputs(constant_inputs);
    }

    fn shareable_constant_state(&self) -> Option<std::sync::Arc<dyn std::any::Any + Send + Sync>> {
        self.prepack.shareable_constant_state()
    }
    fn adopt_shareable_constant_state(
        &mut self,
        state: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<bool> {
        self.prepack.adopt_shareable_constant_state(state)
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
    pub(crate) fn new(resources: CpuMemoryResources, flops: Option<u64>) -> Self {
        Self {
            prepack: MatMulPrepack::new(resources),
            flops,
        }
    }

    pub(super) fn prepare_operand<'a>(
        &self,
        index: usize,
        original: &'a TensorView<'_>,
    ) -> Result<DenseOperand<'a>> {
        if index >= 2 {
            return Err(EpError::KernelFailed(format!(
                "MatMul operand preparation received slot {index}; use A=0 or B=1"
            )));
        }
        self.prepack.dense(index, original)
    }

    /// Consume prepared row-major f32 storage without erasing its grant identity.
    /// Source readers remain alive through optional packing and execution.
    pub(super) fn execute_prepared_operands(
        &self,
        a: &DenseOperand<'_>,
        a_shape: &[usize],
        b: &DenseOperand<'_>,
        b_shape: &[usize],
        output: &mut [f32],
    ) -> Result<()> {
        self.execute_prepared_operands_with_backend(
            a,
            a_shape,
            b,
            b_shape,
            output,
            CpuBackend::auto_detect(),
        )
    }

    fn execute_prepared_operands_with_backend(
        &self,
        a: &DenseOperand<'_>,
        a_shape: &[usize],
        b: &DenseOperand<'_>,
        b_shape: &[usize],
        output: &mut [f32],
        backend: CpuBackend,
    ) -> Result<()> {
        for (name, values, shape) in [("A", a, a_shape), ("B", b, b_shape)] {
            let expected = checked_matmul_numel(shape, name)?;
            std::alloc::Layout::array::<f32>(expected).map_err(|_| EpError::KernelFailed(format!("MatMul prepared {name} shape {shape:?} exceeds addressable f32 storage; reduce the dimensions")))?;
            if values.len() != expected {
                return Err(EpError::KernelFailed(format!(
                    "MatMul prepared {name} has {} elements but shape {shape:?} requires {expected}; preserve the complete prepared operand",
                    values.len()
                )));
            }
        }
        let geometry = matmul_geometry_from_shapes(a_shape, b_shape)?;
        std::alloc::Layout::array::<f32>(geometry.result_len).map_err(|_| EpError::KernelFailed(format!("MatMul prepared output needs {} f32 elements, exceeding addressable storage; reduce the broadcast dimensions",geometry.result_len)))?;
        #[cfg(feature = "mlas")]
        let right = b;
        #[cfg(not(feature = "mlas"))]
        let right = b.as_ref();
        matmul_dense_into_with_backend(
            a.as_ref(),
            right,
            &geometry,
            backend,
            Some(&self.prepack),
            output,
        )
    }

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
            if let Some(bt_f16) =
                self.prepack
                    .transposed_b_half(&inputs[1], geom.k, geom.n, HalfFormat::F16)?
            {
                #[cfg(all(test, target_arch = "aarch64"))]
                GEMV_F16_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let a_dense = self.prepack.dense(0, &inputs[0])?;
                let mut result = vec![0.0f32; geom.n];
                accelerate_gemm::neon_gemv_f16_col_parallel(
                    &a_dense,
                    bt_f16.as_slice(),
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

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if backend == CpuBackend::Accelerate
            && geom.m == 1
            && numel(&geom.batch_shape) <= 1
            && geom.b_promoted_rank == 2
            && let Some(result) =
                untransposed_f16_decode(&self.prepack, &inputs[0], &inputs[1], geom.k, geom.n)?
        {
            return write_dense_f32_narrow("MatMul", &mut outputs[0], &result);
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
        // Which of the two GEMVs runs is `half_decode_gemv_dispatch`'s call,
        // not this site's: a *constant* [K, N] weight is transposed once
        // (budgeted, see `node_weight_transpose_cache_bytes`) and read by
        // `gemv_half_nk`, because the in-place [K, N] walk crosses a page
        // every `p` and runs 1.56-2.98x slower. A non-constant B, or a
        // declined cache, still reads in place through `gemv_half_kn` — so
        // this site remains available to a weight a prepacked transpose could
        // not serve. The transpose carries an actual 2*K*N Host grant and
        // remains reclaimable while idle; its forecast is not an admission.
        //
        // No weight is large enough to divert this to the fused GEBP. That
        // handover existed (`half_decode_prefers_gebp`, `k * n >= 1M`) and was
        // retired: re-measured through this kernel with
        // `benches/half_decode_gemv_ab.rs` it is a loss at every shape at 8
        // threads (1.3x-5.0x) and a loss at the model shapes even at 32
        // (k = 4096 1.1x-2.4x, a 136M lm_head 1.16x).
        //
        // The GEBP earns its packing by reusing a `KC x NR` panel across the
        // rows of A, and at `m == 1` `m_panels` is 1 -- every panel is
        // consumed once, so none of the packing is repaid. What it costs is
        // ~2x the DRAM read traffic (`pack_b_half` walks B column-strip-major
        // and uses 32 bytes of each 64-byte line at the usual one panel per
        // strip) plus the widening work and a fork/join over strips that one
        // row of A cannot amortise. So there is nothing to retune: a
        // decode-specialised GEBP that skips packing B is this GEMV.
        // `docs/benchmarks/2026-08-21-half-decode-gebp-retired.md` has the matrix.
        //
        // A *batched* `m == 1` half MatMul still reaches the GEBP through
        // `try_matmul_half`, because the batch guard below keeps it off this
        // GEMV and its only other destination is the row-blocked GEMM.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(result) = try_half_decode_gemv(&self.prepack, &inputs[0], &inputs[1], &geom)? {
            return write_dense_f32_narrow("MatMul", &mut outputs[0], &result);
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
        // contiguous copy path (cached when retention is admitted, otherwise
        // owned only by this call).
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
                        if let Some(b_contig) = self.prepack.contiguous_b_f16(b_view)? {
                            half_gemm_tile(
                                HalfFormat::F16,
                                a_bits,
                                b_contig.as_slice(),
                                &mut out,
                                geom.m,
                                geom.k,
                                geom.n,
                            );
                        }
                    }
                } else if let Some(b_contig) = self.prepack.contiguous_b_f16(b_view)? {
                    // Non-column-major or batched: use the prepared copy,
                    // borrowing the cache or owning a temporary until compute ends.
                    if trivial_batch {
                        if !accelerate_gemm::bnns_matmul_f16(
                            a_bits,
                            b_contig.as_slice(),
                            &mut out,
                            geom.m,
                            geom.k,
                            geom.n,
                        ) {
                            half_gemm_tile(
                                HalfFormat::F16,
                                a_bits,
                                b_contig.as_slice(),
                                &mut out,
                                geom.m,
                                geom.k,
                                geom.n,
                            );
                        }
                    } else {
                        bnns_half_dense_into(a_bits, b_contig.as_slice(), &geom, &mut out);
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
            return matmul_prepacked_into(
                &inputs[0],
                &inputs[1],
                &geom,
                backend,
                &self.prepack,
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
#[cfg(all(test, feature = "mlas", target_arch = "x86_64"))]
pub(crate) fn half_yielded_to_widened_calls() -> u64 {
    HALF_YIELDED_TO_WIDENED.with(|c| c.get())
}

#[cfg(all(test, feature = "mlas", target_arch = "x86_64"))]
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

/// Whether the `M == 1` decode GEMV is used. On by default;
/// `ONNX_GENAI_CPU_MM_HALF_GEMV=0` (or `off`) sends decode back through the
/// blocking path for the whole process, so a regression can be bisected in the
/// field without a rebuild, and so the A/B bench can measure both arms of the
/// shipped binary. Read once and cached, like `half_prefill_gebp_enabled`.
///
/// The env var is read **once** into a `OnceLock`, which is correct for
/// production — the process is configured at launch and every worker thread
/// reads the same verdict — but toxic for a parallel test harness. A test that
/// set the variable at runtime would get one of two wrong outcomes depending on
/// which test happened to read it first: either the latch is already filled and
/// the `set_var` does nothing, so both arms of an A/B silently run the *same*
/// route and it passes while measuring nothing; or this test wins the race and
/// pins its value process-wide for every later test. Tests must therefore never
/// touch the variable; they use `HalfDecodeGemvScope`, a thread-local override
/// that leaves the latch untouched. See #1736, and #1056 for the same trap in
/// the weight-transpose cache.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn half_decode_gemv_enabled() -> bool {
    // Gated identically to the thread-local below rather than on bare `test`:
    // the two are equivalent only because this function is already x86-only, and
    // adding a non-x86 arm should not silently reference a symbol that does not
    // exist there.
    #[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
    if let Some(forced) = HALF_DECODE_GEMV_OVERRIDE.with(std::cell::Cell::get) {
        return forced;
    }
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

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
thread_local! {
    /// Test-only, per-thread override of the decode-GEMV gate. `None` defers to
    /// the process-global latch; `Some(v)` forces `v` on **this thread only**,
    /// so one test's arm cannot leak into another test running concurrently.
    /// Set exclusively through [`HalfDecodeGemvScope`]. Production never reads
    /// this — the `#[cfg(test)]` lookup in [`half_decode_gemv_enabled`] compiles
    /// out entirely.
    static HALF_DECODE_GEMV_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// RAII, thread-local scoping of the decode-GEMV gate for tests (#1736).
///
/// Constructing one forces [`half_decode_gemv_enabled`] to `enabled` on the
/// current thread; dropping it restores the previous value, including on panic,
/// so a failing test cannot leave the gate flipped for the tests that follow.
/// This replaces `set_var`/`remove_var` on `ONNX_GENAI_CPU_MM_HALF_GEMV`, which
/// could not work against a latched `OnceLock` and which is `unsafe` in edition
/// 2024 precisely because it races concurrent readers.
///
/// Being thread-local is what makes it safe under `cargo test`'s parallel
/// harness, and it is sound here only because the gate is read on the thread
/// that calls `execute`. That is not assumed: the route counters are thread-local
/// too, so if the read ever moved onto a pool worker the override would stop
/// applying *and* the count would drop to zero, failing the caller's
/// `gemv_calls == 1` assertion rather than silently reverting to the old
/// vacuous A/B.
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) struct HalfDecodeGemvScope {
    prev: Option<bool>,
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
impl HalfDecodeGemvScope {
    /// Force the decode-GEMV gate to `enabled` on this thread until dropped.
    pub(crate) fn new(enabled: bool) -> Self {
        let prev = HALF_DECODE_GEMV_OVERRIDE.with(|c| c.replace(Some(enabled)));
        Self { prev }
    }
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
impl Drop for HalfDecodeGemvScope {
    fn drop(&mut self) {
        HALF_DECODE_GEMV_OVERRIDE.with(|c| c.set(self.prev));
    }
}

/// The 16-bit storage format both operands share, if they share one.
///
/// Mixed and non-16-bit pairs return `None`: every kernel behind this reads
/// both operands as raw `u16` and widens them the same way, so a mismatch
/// would silently reinterpret one of them.
pub(crate) fn half_storage_format(
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

/// Attempt the dedicated portable half GEMM path. Both operands must be
/// contiguous and have the same `Float16` or `BFloat16` dtype. The operands stay
/// in 16-bit storage until cache-panel packing, accumulation is always `f32`,
/// and the caller narrows once into the requested output dtype.
///
/// Declines for a large-`M` `f16` GEMM, where widening to `f32` and using the
/// tuned SGEMM is measurably faster -- see [`widened_sgemm_beats_half_gemm`].
/// Every caller already falls through to that widened path.
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
        count_half_native_bf16();
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
/// The rule is therefore **weight-only**: every `m`, both formats, once
/// `k * n` clears [`HALF_PREFILL_GEBP_MIN_WEIGHT`]. `m >= 2` is settled by
/// that constant's own sweep.
///
/// `m == 1` is *not* settled by it. A non-batched half decode never arrives
/// here at all -- the GEMV in `try_matmul` takes it at every weight, because
/// the panel this kernel packs is reused across rows of `A` and at `m == 1`
/// there is one row (`docs/benchmarks/2026-08-21-half-decode-gebp-retired.md`;
/// the handover this doc used to cite was measured a 1.3x-5.0x loss at 8
/// threads). What still arrives at `m == 1` is a *batched* half MatMul, which
/// the GEMV declines and whose only other destination is the row-blocked
/// GEMM -- so this gate keeps accepting it.
///
/// `format` and `m` are taken only so the signature still documents what the
/// decision is about and so a future format- or row-dependent term has a home.
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
    let _ = (format, m);
    true
}

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
    HALF_NATIVE_BF16_CALLS.with(|c| c.set(0));
}

#[cfg(all(test, target_arch = "x86_64"))]
thread_local! {
    /// Test-only count of half GEMM tiles served by the native AVX-512 BF16
    /// kernel. `half_gemm_tile` tries that kernel *before* the fused
    /// widen-pack GEBP, so on a host with `avx512bf16` a bf16 tile never
    /// reaches the GEBP. Counting both arms lets a test assert that exactly
    /// one of them ran, which keeps the dispatch guarded on every CPU instead
    /// of only on the ones that lack AVX-512 BF16.
    static HALF_NATIVE_BF16_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn count_half_native_bf16() {
    #[cfg(test)]
    HALF_NATIVE_BF16_CALLS.with(|c| c.set(c.get() + 1));
}

#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn half_native_bf16_calls() -> u64 {
    HALF_NATIVE_BF16_CALLS.with(std::cell::Cell::get)
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

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
thread_local! {
    /// Test-only count of decodes that reached the `[N, K]` GEMV *through the
    /// transpose cache*, i.e. from a weight the model stored as `[K, N]`.
    ///
    /// Separate from `HALF_DECODE_GEMV_CALLS` on purpose: that counter says a
    /// GEMV ran, this one says *which layout kernel* served it. Only the pair
    /// distinguishes the three routes a 16-bit decode can take.
    static HALF_DECODE_TRANSPOSED_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn half_decode_transposed_calls() -> u64 {
    HALF_DECODE_TRANSPOSED_CALLS.with(std::cell::Cell::get)
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn reset_half_decode_transposed_calls() {
    HALF_DECODE_TRANSPOSED_CALLS.with(|c| c.set(0));
}

/// Serve one `m == 1` 16-bit decode from whichever layout kernel is best for
/// the weight as stored, for every operator that has one.
///
/// `trans_b` describes storage, not intent: `true` means the weight is already
/// `[N, K]`, which is the layout `gemv_half_nk` wants. When it is `false` the
/// weight is `[K, N]`, and the choice is between reading it in place with
/// `gemv_half_kn` or transposing it once into the prepack cache and using the
/// `[N, K]` kernel for the rest of the session.
///
/// Measured on this AVX2 host at `t = 1`, transposing wins on *both* axes,
/// which is why it is preferred whenever the cache admits it:
///
/// | shape (k x n)  | `kn` in place | `nk` transposed | speedup |
/// |----------------|---------------|-----------------|---------|
/// | 4096 x 6144    | 5022 us       | 1688 us         | 2.98x   |
/// | 4096 x 4096    | 1967 us       | 979 us          | 2.01x   |
/// | 4096 x 14336   | 11332 us      | 3939 us         | 2.88x   |
/// | 14336 x 4096   | 11015 us      | 7068 us         | 1.56x   |
///
/// The reason is the traversal, not the arithmetic. `[K, N]` walks a column
/// tile with an `n * 2`-byte stride between consecutive `p`; at `n = 6144`
/// that is 12 KiB, so consecutive reads land on different pages and the
/// hardware prefetcher -- which does not cross a page boundary -- cannot run
/// ahead. `[N, K]` makes every output one contiguous `k`-element dot instead.
/// The in-place kernel sustains 10.0-17.1 GB/s where the transposed one
/// reaches 16.6-34.3 GB/s over identical byte counts.
///
/// Accuracy moves the same way: `[K, N]` carries one serial accumulator per
/// column across the whole contraction, `[N, K]` carries four and combines
/// them pairwise, so the transposed kernel is *also* 2.7-9.3x closer to the
/// `f64` reference. This route therefore has no accuracy trade to weigh.
///
/// Falls back to reading `[K, N]` in place when the weight is not a constant
/// initializer (a transpose per call would cost more than it saves) or when
/// the `#1056` cache governor declines the footprint.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(
    clippy::too_many_arguments,
    reason = "one call site per operator; the operands are the GEMV's own"
)]
pub(crate) fn half_decode_gemv_dispatch(
    format: HalfFormat,
    prepack: &MatMulPrepack,
    b_view: &TensorView,
    a: &[f32],
    b_bits: &[u16],
    out: &mut [f32],
    k: usize,
    n: usize,
    trans_b: bool,
) -> Result<()> {
    if trans_b {
        half_gemv::gemv_half_nk(format, a, b_bits, out, k, n);
        return Ok(());
    }
    if let Some(bt) = prepack.transposed_b_half(b_view, k, n, format)? {
        #[cfg(test)]
        HALF_DECODE_TRANSPOSED_CALLS.with(|c| c.set(c.get() + 1));
        half_gemv::gemv_half_nk(format, a, bt.as_slice(), out, k, n);
        return Ok(());
    }
    half_gemv::gemv_half_kn(format, a, b_bits, out, k, n);
    Ok(())
}

/// The x86 16-bit decode GEMV arm, shared by every operator whose math is
/// `A @ B` with a 2-D 16-bit `B` at `m == 1`.
///
/// Returns `Some(result)` (length `n`) when the route applies, `None` when any
/// guard declines and the caller must continue down its own chain.
///
/// **This exists because a copy of these guards is how #1381 stayed open.**
/// `MatMul` reached this route and `FusedMatMulBias` — the fusion the
/// optimizer produces for the `MatMul + Add(bias)` that almost every
/// `nn.Linear` becomes — did not, so the *better-optimized* form of a
/// projection was the slower one, by 1.55x-3.02x depending on shape and dtype.
/// The divergence was invisible because the two kernels agreed on values and
/// only disagreed on which kernel produced them. Extracting the arm makes
/// "same math, same route" a property of there being one arm, rather than of
/// two sites being edited together.
///
/// Every guard here is load-bearing and is asserted by
/// `half_decode_route_tests`:
/// - `m == 1`: above it, panel reuse repays `try_matmul_half`'s packing.
/// - unbatched, 2-D promoted `B`: a batched `m == 1` belongs on the GEBP.
/// - contiguous `B` with exactly `k * n` elements: the raw-bits read below.
/// - a 16-bit storage format both operands share, with SIMD for it.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn try_half_decode_gemv(
    prepack: &MatMulPrepack,
    a: &TensorView,
    b: &TensorView,
    geom: &MatMulGeometry,
) -> Result<Option<Vec<f32>>> {
    let (k, n) = (geom.k, geom.n);
    if geom.m != 1
        || numel(&geom.batch_shape) > 1
        || geom.b_promoted_rank != 2
        || !b.is_contiguous()
        || b.numel() != k.saturating_mul(n)
        || !half_decode_gemv_enabled()
    {
        return Ok(None);
    }
    let Some(format) = half_storage_format(a.dtype, b.dtype) else {
        return Ok(None);
    };
    if !half_gemv::simd_available(format) {
        return Ok(None);
    }
    b.validate()?;
    let a_dense = prepack.dense(0, a)?;
    if a_dense.len() != k {
        return Ok(None);
    }
    // SAFETY: `b` was just validated as a contiguous Float16/BFloat16 view
    // whose element count equals `k * n`. Both are transparent over `u16`, so
    // reading their storage as raw bit patterns is sound, and the view
    // outlives this call.
    let b_bits = unsafe { std::slice::from_raw_parts(b.data_ptr::<u16>(), k * n) };
    let mut result = vec![0.0f32; n];
    count_half_decode_gemv();
    half_decode_gemv_dispatch(
        format,
        prepack,
        b,
        &a_dense,
        b_bits,
        &mut result,
        k,
        n,
        false,
    )?;
    Ok(Some(result))
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
    let a_dense = to_dense_f32_widen("MatMul", a)?;
    let b_dense = to_dense_f32_widen("MatMul", b)?;
    #[cfg(feature = "mlas")]
    let b_dense = b_dense.into();
    matmul_dense_impl_with_geom(a_dense, b_dense, &geom, CpuBackend::auto_detect(), None)
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
    matmul_prepacked_into(a, b, &geom, backend, prepack, out_slice)?;
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
    validate_f32_widen_input("MatMul prepared A", a)?;
    validate_f32_widen_input("MatMul prepared B", b)?;
    let mut out = vec![0.; geom.result_len];
    matmul_prepacked_into(a, b, &geom, backend, prepack, &mut out)?;
    Ok(out)
}

fn matmul_prepacked_into(
    a: &TensorView,
    b: &TensorView,
    geom: &MatMulGeometry,
    backend: CpuBackend,
    prepack: &MatMulPrepack,
    out: &mut [f32],
) -> Result<()> {
    if out.len() != geom.result_len {
        return Err(EpError::KernelFailed(format!(
            "MatMul prepared output has {} elements, expected {}; supply the complete result buffer",
            out.len(),
            geom.result_len
        )));
    }
    if out.is_empty() {
        validate_f32_widen_input("MatMul prepared A", a)?;
        validate_f32_widen_input("MatMul prepared B", b)?;
        return Ok(());
    }
    // Mandatory activation materialization precedes optional B admission, so
    // an optional transpose cannot consume memory needed to execute at all.
    let activation = prepack.dense(0, a)?;
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if backend == CpuBackend::Accelerate
        && numel(&geom.batch_shape) <= 1
        && geom.b_promoted_rank == 2
        && (geom.m == 1 || accelerate_gemm::thin_m_gemm_eligible(geom.m, geom.k, geom.n))
        && let Some(bt) = prepack.transposed_b(b, geom.k, geom.n)?
    {
        if geom.m == 1 {
            accelerate_gemm::neon_gemv_col_parallel(
                &activation,
                bt.as_slice(),
                out,
                geom.k,
                geom.n,
            );
        } else {
            #[cfg(all(test, target_arch = "aarch64"))]
            THIN_M_GEMM_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            accelerate_gemm::neon_thin_m_gemm_col_parallel(
                &activation,
                bt.as_slice(),
                out,
                geom.m,
                geom.k,
                geom.n,
            );
        }
        return Ok(());
    }
    matmul_dense_into_with_backend(
        &activation,
        &prepack.dense(1, b)?,
        geom,
        backend,
        Some(prepack),
        out,
    )
}

/// Shared owned-vector GEMM: allocate the result buffer and GEMM into it.
/// Geometry is pre-computed by the caller to avoid redundant derivation.
fn matmul_dense_impl_with_geom(
    a_dense: impl AsRef<[f32]>,
    #[cfg(feature = "mlas")] b_dense: DenseOperand<'_>,
    #[cfg(not(feature = "mlas"))] b_dense: impl AsRef<[f32]>,
    geom: &MatMulGeometry,
    backend: CpuBackend,
    prepack: Option<&MatMulPrepack>,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; geom.result_len];
    #[cfg(feature = "mlas")]
    let b = &b_dense;
    #[cfg(not(feature = "mlas"))]
    let b = b_dense.as_ref();
    matmul_dense_into_with_backend(a_dense.as_ref(), b, geom, backend, prepack, &mut out)?;
    Ok(out)
}

/// Precomputed MatMul dimensions: 1-D promotion, inner-dim agreement, batch
/// broadcast, and per-tile element counts. Computed once so both the owned and
/// direct-output paths share exactly one geometry derivation.
pub(crate) struct MatMulGeometry {
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
pub(crate) fn matmul_geometry(a: &TensorView, b: &TensorView) -> Result<MatMulGeometry> {
    matmul_geometry_from_shapes(a.shape, b.shape)
}

fn checked_matmul_numel(shape: &[usize], operand: &str) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize,|elements,&extent|elements.checked_mul(extent)).ok_or_else(|| EpError::KernelFailed(format!("MatMul {operand} shape {shape:?} element count overflowed usize; reduce the dimensions")))
}

fn matmul_geometry_from_shapes(a_raw: &[usize], b_raw: &[usize]) -> Result<MatMulGeometry> {
    // Promote 1-D operands per numpy matmul: a [K] -> [1,K] (drop row after),
    // b [K] -> [K,1] (drop col after).
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
    let batch_count = checked_matmul_numel(&batch_shape, "batch")?;
    // Empty results never use matrix offsets or batch strides. Do not reject
    // or overflow irrelevant large extents before the no-op can return.
    if batch_count == 0 || m == 0 || n == 0 {
        return Ok(MatMulGeometry {
            m,
            k,
            n,
            a_mat: 0,
            b_mat: 0,
            c_mat: 0,
            a_batch_strides: vec![0; a_batch.len()],
            b_batch_strides: vec![0; b_batch.len()],
            a_batch,
            b_batch,
            batch_shape,
            b_promoted_rank: b_shape.len(),
            result_len: 0,
        });
    }
    i64::try_from(batch_count).map_err(|_| EpError::KernelFailed(format!("MatMul batch shape {batch_shape:?} exceeds signed-stride addressability; reduce the batch dimensions")))?;
    let a_mat = checked_matmul_numel(&[m, k], "A matrix")?;
    let b_mat = checked_matmul_numel(&[k, n], "B matrix")?;
    let c_mat = checked_matmul_numel(&[m, n], "output matrix")?;
    let result_len = batch_count.checked_mul(c_mat).ok_or_else(|| EpError::KernelFailed(format!("MatMul broadcast result {batch_shape:?} x [{m}, {n}] element count overflowed usize; reduce the dimensions")))?;
    let a_batch_strides = compute_contiguous_strides(&a_batch);
    let b_batch_strides = compute_contiguous_strides(&b_batch);

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
        result_len,
    })
}

/// Run the GEMM (single, batched, or broadcast) into the caller-supplied
/// row-major `out` slice. `out.len()` MUST equal `geom.result_len`; a mismatch
/// returns an `EpError` before any write. This is the single code path shared by
/// the owned-vector wrapper and the direct-output fast path.
fn matmul_dense_into_with_backend(
    a_dense: &[f32],
    #[cfg(feature = "mlas")] b_operand: &DenseOperand<'_>,
    #[cfg(not(feature = "mlas"))] b_dense: &[f32],
    geom: &MatMulGeometry,
    backend: CpuBackend,
    prepack: Option<&MatMulPrepack>,
    out: &mut [f32],
) -> Result<()> {
    #[cfg(feature = "mlas")]
    let b_dense = b_operand.as_ref();
    if out.len() != geom.result_len {
        return Err(EpError::KernelFailed(format!(
            "MatMul: output buffer length {} does not match result length {}; supply the complete result buffer",
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

    if geom.k == 0 {
        out.fill(0.0);
        return Ok(());
    }

    let (m, k, n) = (geom.m, geom.k, geom.n);
    let (a_mat, b_mat, c_mat) = (geom.a_mat, geom.b_mat, geom.c_mat);

    // Original-view preparation runs before this dense compute function.
    // Only optional MLAS packing consumes the prepack here.
    #[cfg(not(feature = "mlas"))]
    let _ = &prepack;

    #[cfg(feature = "mlas")]
    let packed_b = if backend == CpuBackend::Mlas && geom.b_promoted_rank == 2 {
        prepack
            .map(|prepack| prepack.packed_b(b_operand, k, n))
            .transpose()?
            .flatten()
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
        accelerate_gemm::neon_gemv_parallel(a_dense, b_dense, out, k, n)?;
        return Ok(());
    }

    // A batch product of 1 (e.g. batch_shape = [1] from a [1,1,K] activation
    // against a [K,N] weight) is equivalent to no batch. Use the non-batched
    // path to pick up packed_b, avoid the per-batch iteration overhead, and
    // stay consistent with the Accelerate M=1 fast path above.
    if numel(&geom.batch_shape) <= 1 {
        // No effective batch dims: a single matmul.
        #[cfg(feature = "mlas")]
        if let Some(packed_b) = &packed_b {
            gemm_packed(a_dense, packed_b.payload(), out, m, k, n)?;
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
            if let Some(packed_b) = &packed_b {
                gemm_packed(a_tile, packed_b.payload(), c_tile, m, k, n)?;
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

    #[cfg(feature = "mlas")]
    #[test]
    fn packed_sgemm_distinguishes_recycled_conversion_storage_without_pinning_it() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let panel_bytes = sgemm_pack_plan(3, 2).unwrap().owned_heap_bytes() as u64;
        let books = LedgerGovernor::new(LeaseLedger::new(0, 3 * panel_bytes + 72, 0));
        let resources = CpuMemoryResources::new(
            crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
            Default::default(),
        )
        .unwrap();
        let mut prepack = MatMulPrepack::new(resources.clone());
        prepack.set_constant_inputs(&[false, true]);
        // A bounded cache retires the source while the panel remains resident.
        Arc::get_mut(&mut prepack.weights).unwrap().dense[1] =
            WeightCache::new(resources.weight_memory().clone(), 24);
        let original = Owned::f64(&[3, 2], &[0.5; 6]);
        let original_view = original.view();
        let first = prepack.dense(1, &original_view).unwrap();
        let address = first.as_ptr();
        let first_panel = prepack.packed_b(&first, 3, 2).unwrap().unwrap();
        let dummy = Owned::f64(&[3, 2], &[0.75; 6]);
        drop(prepack.dense(1, &dummy.view()).unwrap());
        let DenseOperandStorage::Funded(owner) = first.0 else {
            panic!("f64 must own a conversion");
        };
        assert_eq!(
            Arc::strong_count(&owner),
            1,
            "panel identity must not pin the source payload or its grant"
        );
        let (recycled, old_grant) = Arc::try_unwrap(owner)
            .unwrap_or_else(|_| panic!("retired conversion has no other payload owners"))
            .test_into_parts();
        // Move the allocation into a new conversion, establishing address reuse
        // without depending on the system allocator returning the same address.
        let next = Owned::f64(&[3, 2], &[1.5; 6]);
        let next_view = next.view();
        let key = MatMulCopyKey::<4>::from_view(&next_view);
        let mut recycled = Some(recycled);
        let loaded = prepack.weights.dense[1]
            .load_group_owned(&[WeightSpec::cached(&key).unwrap()], |_| {
                let mut values = recycled.take().expect("one conversion");
                to_dense_f32_widen_into("recycled constant conversion", &next_view, &mut values)?;
                Ok(values)
            })
            .unwrap();
        let WeightLoad::Ready(mut loaded) = loaded else {
            panic!("complete construction peak fits");
        };
        let second = DenseOperand(DenseOperandStorage::Funded(loaded.weights.remove(0)));
        drop(old_grant); // The new owner now funds the reused allocation.
        assert_eq!(
            second.as_ptr(),
            address,
            "the recycled-address precondition must hold on every allocator"
        );
        assert_eq!(books.used(Tier::Host), 24 + panel_bytes);
        let second_panel = prepack.packed_b(&second, 3, 2).unwrap().unwrap();
        let mut output = [0.; 4];
        gemm_packed(&[0.25; 6], second_panel.payload(), &mut output, 2, 3, 2).unwrap();
        assert_eq!(
            output, [1.125; 4],
            "a recycled conversion address must not serve the previous weight's panel"
        );
        assert!(!Arc::ptr_eq(&first_panel, &second_panel));
        let warm = prepack.packed_b(&second, 3, 2).unwrap().unwrap();
        assert!(Arc::ptr_eq(&warm, &second_panel));
        assert_eq!(prepack.weights.packed_b.activity().1, 2);
        assert_eq!(books.used(Tier::Host), 24 + 2 * panel_bytes);
        drop(warm);
        let kernel = MatMulKernel {
            prepack,
            flops: None,
        };
        let mut integrated = [73.; 4];
        kernel
            .execute_prepared_operands_with_backend(
                &DenseOperand::borrowed(&[0.25; 6]),
                &[2, 3],
                &second,
                &[3, 2],
                &mut integrated,
                CpuBackend::Mlas,
            )
            .unwrap();
        assert_eq!(integrated, [1.125; 4]);
        assert_eq!(
            kernel.prepack.weights.packed_b.activity().1,
            2,
            "typed compute must preserve the funded identity and reuse its existing panel"
        );
        assert_eq!(books.used(Tier::Host), 24 + 2 * panel_bytes);
        drop(kernel);
        assert_eq!(books.used(Tier::Host), 24 + 2 * panel_bytes);
        drop(second);
        assert_eq!(
            books.used(Tier::Host),
            2 * panel_bytes,
            "active panels do not keep retired conversion bytes charged"
        );
        gemm_packed(&[0.25; 6], first_panel.payload(), &mut output, 2, 3, 2).unwrap();
        assert_eq!(
            output, [0.375; 4],
            "an old executing panel remains independently valid"
        );
        drop(first_panel);
        drop(second_panel);
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn packed_half_sgemm_keys_the_complete_original_view() {
        let source = Owned::f16(&[24], &(1..=24).map(|n| n as f32).collect::<Vec<_>>());
        let shape = [3, 2];
        let first_strides = [4, 1];
        let next_strides = [1, 3];
        let mut first = source.view();
        first.shape = &shape;
        first.strides = &first_strides;
        let mut next = first;
        next.strides = &next_strides;
        let mut prepack = MatMulPrepack::default();
        prepack.set_constant_inputs(&[false, true]);
        let original_panel = prepack.packed_b_from_half(&first, 3, 2).unwrap().unwrap();
        let next_panel = prepack.packed_b_from_half(&next, 3, 2).unwrap().unwrap();
        let mut output = [0.; 2];
        gemm_packed(&[1.; 3], next_panel.payload(), &mut output, 1, 3, 2).unwrap();
        assert_eq!(
            output,
            [6., 15.],
            "different strides over one immutable source need different packed values"
        );
        assert!(!Arc::ptr_eq(&original_panel, &next_panel));
        let warm = prepack.packed_b_from_half(&first, 3, 2).unwrap().unwrap();
        assert!(Arc::ptr_eq(&original_panel, &warm));
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn call_owned_conversion_uses_unpacked_sgemm_without_retaining_an_address_key() {
        let mut prepack = MatMulPrepack::default();
        prepack.set_constant_inputs(&[false, true]);
        let b: DenseOperand<'_> = Cow::<[f32]>::Owned(vec![0.5; 6]).into();
        assert!(
            prepack.packed_b(&b, 3, 2).unwrap().is_none(),
            "a call-owned conversion has no stable immutable source identity"
        );
        assert_eq!(prepack.weights.packed_b.stats(), (0, 0));
        let a = Owned::f32(&[2, 3], &[0.25; 6]);
        let weight = Owned::f32(&[3, 2], &[0.5; 6]);
        let geometry = matmul_geometry(&a.view(), &weight.view()).unwrap();
        let mut output = [0.; 4];
        matmul_dense_into_with_backend(
            &[0.25; 6],
            &b,
            &geometry,
            CpuBackend::Mlas,
            Some(&prepack),
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [0.375; 4]);
        assert_eq!(prepack.weights.packed_b.stats(), (0, 0));
    }

    #[test]
    fn direct_transpose_conversion_preserves_float_bits_and_special_values() {
        let bits: Vec<u16> = (0..=u16::MAX).collect();
        let source = Owned::f16_bits(&[256, 256], &bits);
        let prepack = constant_b_prepack();
        for dtype in [DataType::Float16, DataType::BFloat16] {
            let mut view = source.view();
            view.dtype = dtype;
            let dense = to_dense_f32_widen("reference conversion", &view).unwrap();
            let expected = reference_transpose(&dense, 256, 256);
            let got = prepack.transposed_b(&view, 256, 256).unwrap().unwrap();
            for (index, (&got, &expected)) in got.as_slice().iter().zip(&expected).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "{dtype:?} transposed element {index}"
                );
            }
        }
        assert!(
            !prepack.dense_is_filled(1),
            "source conversion writes only the final transpose"
        );
        for source in [
            Owned::f64(
                &[2, 3],
                &[
                    f64::from_bits(0x7ff0000000000001),
                    f64::from_bits(0xfff0000000000001),
                    f64::from_bits(1),
                    f64::NAN,
                    -0.,
                    f64::MAX,
                ],
            ),
            Owned::f32(
                &[2, 3],
                &[
                    f32::from_bits(0x7f800001),
                    f32::from_bits(0xff800001),
                    f32::from_bits(1),
                    f32::NAN,
                    -0.,
                    f32::MAX,
                ],
            ),
        ] {
            let view = source.view();
            let dense = to_dense_f32_widen("reference conversion", &view).unwrap();
            let expected = reference_transpose(&dense, 2, 3);
            let got = prepack.transposed_b(&view, 2, 3).unwrap().unwrap();
            assert_eq!(
                got.as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn provider_decode_uses_only_final_transpose_before_and_after_source_reclamation() {
        if retention_control_in_child(
            "kernels::matmul::tests::provider_decode_uses_only_final_transpose_before_and_after_source_reclamation",
            Some("accelerate"),
        ) {
            return;
        }
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::{Attribute, NodeId};
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, MemoryGovernor};
        for prefill in [false, true] {
            for op in ["MatMul", "FusedMatMulBias", "FusedGemm", "Gemm"] {
                let gemm = op == "Gemm";
                let fused = op.starts_with("Fused");
                let books = RecordingGovernor::new(if prefill { 48 } else { 24 });
                let resources = CpuMemoryResources::new(
                    books.weight_memory(HolderId::new(13)),
                    Default::default(),
                )
                .unwrap();
                let provider = crate::CpuExecutionProvider::with_resources(resources);
                let mut node = Node::new(NodeId(0), op, vec![], vec![]);
                if fused {
                    node.domain = "com.microsoft".into();
                }
                if gemm {
                    node.attributes.insert("transB".into(), Attribute::Int(1));
                }
                let shape = if gemm { vec![2, 3] } else { vec![3, 2] };
                let b = Owned::f64(&shape, &[0.5; 6]);
                let bias = Owned::f32(&[2], &[0.125; 2]);
                let make = |rows| {
                    let mut kernel = provider
                        .get_kernel(&node, &[vec![rows, 3], shape.clone()], 17)
                        .unwrap();
                    kernel.set_constant_inputs(&[false, true]);
                    kernel
                };
                let mut kernel = make(1);
                if prefill {
                    let first = make(3);
                    let a = Owned::f32(&[3, 3], &[0.25; 9]);
                    let mut out = Owned::zeros_f32(&[3, 2]);
                    let inputs = if fused {
                        vec![a.view(), b.view(), bias.view()]
                    } else {
                        vec![a.view(), b.view()]
                    };
                    first.execute(&inputs, &mut [out.view_mut()]).unwrap();
                    assert_eq!(out.to_f32(), vec![if fused { 0.5 } else { 0.375 }; 6]);
                    assert!(
                        kernel
                            .adopt_shareable_constant_state(
                                first.shareable_constant_state().unwrap()
                            )
                            .unwrap()
                    );
                    drop(first);
                }
                let a = Owned::f32(&[1, 3], &[0.25; 3]);
                let mut out = Owned::zeros_f32(&[1, 2]);
                let inputs = if fused {
                    vec![a.view(), b.view(), bias.view()]
                } else {
                    vec![a.view(), b.view()]
                };
                let mut run = || {
                    kernel.execute(&inputs, &mut [out.view_mut()]).unwrap();
                    assert_eq!(out.to_f32(), vec![if fused { 0.5 } else { 0.375 }; 2]);
                };
                let cold = books.claims();
                run();
                assert_eq!(
                    books.claims(),
                    cold + usize::from(!prefill || !gemm),
                    "{op}: preparation must acquire only the final B representation"
                );
                assert_eq!(
                    books.used(Tier::Host),
                    if prefill && !gemm { 48 } else { 24 }
                );
                let warm = (books.claims(), books.releases());
                for _ in 0..4096 {
                    run();
                }
                assert_eq!(
                    (books.claims(), books.releases()),
                    warm,
                    "{op}: warm decode must not prepare a source copy"
                );
                if prefill {
                    let kv = books
                        .reserve(Tier::Host, 24, MemoryRole::KvCache, HolderId::new(7))
                        .unwrap();
                    assert_eq!(books.used(Tier::Host), 48);
                    let after_reclaim = (books.claims(), books.releases());
                    for _ in 0..32 {
                        run();
                    }
                    assert_eq!(
                        (books.claims(), books.releases()),
                        after_reclaim,
                        "{op}: reclaiming the unused source must not force reconstruction"
                    );
                    drop(kv);
                }
                drop(kernel);
                drop(provider);
                assert_eq!(books.used(Tier::Host), 0);
            }
        }
        // Optional preparation cannot take memory required by constant A.
        for op in ["MatMul", "FusedMatMulBias", "FusedGemm"] {
            let fused = op.starts_with("Fused");
            let books = RecordingGovernor::new(24);
            let resources =
                CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
                    .unwrap();
            let provider = crate::CpuExecutionProvider::with_resources(resources);
            let mut node = Node::new(NodeId(0), op, vec![], vec![]);
            if fused {
                node.domain = "com.microsoft".into();
            }
            let mut kernel = provider
                .get_kernel(&node, &[vec![1, 3], vec![3, 2]], 17)
                .unwrap();
            kernel.set_constant_inputs(&[true, true]);
            let a = Owned::f64(&[1, 3], &[0.25; 3]);
            let b = Owned::f32(&[3, 2], &[0.5; 6]);
            let bias = Owned::f32(&[2], &[0.125; 2]);
            let mut out = Owned::zeros_f32(&[1, 2]);
            let inputs = if fused {
                vec![a.view(), b.view(), bias.view()]
            } else {
                vec![a.view(), b.view()]
            };
            kernel.execute(&inputs, &mut [out.view_mut()]).unwrap();
            assert_eq!(out.to_f32(), vec![if fused { 0.5 } else { 0.375 }; 2]);
            assert_eq!(
                books.used(Tier::Host),
                12,
                "mandatory A fits while optional B must decline"
            );
            drop(kernel);
            drop(provider);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[test]
    fn f32_transpose_is_funded_reclaimable_and_warm_without_accounting() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, MemoryGovernor};
        let books = RecordingGovernor::new(24);
        let resources =
            CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
                .unwrap();
        let mut prepack = MatMulPrepack::new(resources);
        prepack.set_constant_inputs(&[false, true]);
        let source = Owned::f32(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let view = source.view();
        let reader = prepack.transposed_b(&view, 3, 2).unwrap().unwrap();
        assert_eq!(reader.as_slice(), [1., 3., 5., 2., 4., 6.]);
        assert_eq!(books.used(Tier::Host), 24);
        let before = (books.claims(), books.releases());
        for _ in 0..4096 {
            let warm = prepack.transposed_b(&view, 3, 2).unwrap().unwrap();
            assert!(Arc::ptr_eq(&reader, &warm));
        }
        assert_eq!((books.claims(), books.releases()), before);
        assert!(
            books
                .reserve(Tier::Host, 24, MemoryRole::KvCache, HolderId::new(7))
                .is_err()
        );
        drop(reader);
        let kv = books
            .reserve(Tier::Host, 24, MemoryRole::KvCache, HolderId::new(7))
            .unwrap();
        assert_eq!(prepack.f32_transpose_stats(), (0, 0));
        let builds = prepack.weights.transposed_b.activity().1;
        assert!(prepack.transposed_b(&view, 3, 2).unwrap().is_none());
        assert_eq!(prepack.weights.transposed_b.activity().1, builds);
        drop(kv);
        let reader = prepack.transposed_b(&view, 3, 2).unwrap().unwrap();
        assert_eq!(prepack.weights.transposed_b.activity().1, builds + 1);
        drop(prepack);
        assert_eq!(books.used(Tier::Host), 24);
        drop(reader);
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[test]
    fn f32_transpose_keys_original_layout_without_materializing_a_copy() {
        let source = Owned::f64(&[24], &(1..=24).map(|n| n as f64).collect::<Vec<_>>());
        let shape = [3, 2];
        let first_strides = [4, 1];
        let next_strides = [1, 3];
        let mut first = source.view();
        first.shape = &shape;
        first.strides = &first_strides;
        let mut next = first;
        next.strides = &next_strides;
        let prepack = constant_b_prepack();

        let first_t = prepack.transposed_b(&first, 3, 2).unwrap().unwrap();

        let next_t = prepack.transposed_b(&next, 3, 2).unwrap().unwrap();
        assert_eq!(first_t.as_slice(), [1., 5., 9., 2., 6., 10.]);
        assert_eq!(next_t.as_slice(), [1., 2., 3., 4., 5., 6.]);
        assert!(!Arc::ptr_eq(&first_t, &next_t));
        let warm = prepack.transposed_b(&first, 3, 2).unwrap().unwrap();
        assert!(Arc::ptr_eq(&first_t, &warm));
        let reversed_strides = [-4, -1];
        let mut reversed = first;
        reversed.strides = &reversed_strides;
        reversed.byte_offset = 9 * 8;
        let reversed_t = prepack.transposed_b(&reversed, 3, 2).unwrap().unwrap();
        assert_eq!(reversed_t.as_slice(), [10., 6., 2., 9., 5., 1.]);
        assert!(!prepack.dense_is_filled(1));
    }

    #[test]
    fn f32_transpose_does_not_cross_node_lifetimes_at_reused_original_addresses() {
        let mut data = vec![1., 2., 3., 4., 5., 6.];
        let address = data.as_ptr();
        let first = constant_b_prepack();
        let old = cached_f32(&first, &data, 3, 2).unwrap();
        drop(first);
        data.fill(9.);
        assert_eq!(data.as_ptr(), address);
        let next = constant_b_prepack();
        let current = cached_f32(&next, &data, 3, 2).unwrap();
        assert_eq!(old.as_slice(), [1., 3., 5., 2., 4., 6.]);
        assert_eq!(current.as_slice(), [9.; 6]);
        assert!(!Arc::ptr_eq(&old, &current));
    }

    #[test]
    fn f32_transpose_concurrent_fills_share_one_granted_payload_per_geometry() {
        for distinct in [false, true] {
            let prepack = constant_b_prepack();
            let source: Vec<f32> = (0..96).map(|v| v as f32).collect();
            let barrier = std::sync::Barrier::new(8);
            let readers = std::thread::scope(|scope| {
                let jobs: Vec<_> = (0..8)
                    .map(|i| {
                        let prepack = &prepack;
                        let source = &source;
                        let barrier = &barrier;
                        scope.spawn(move || {
                            let (k, n) = if distinct { (i + 1, 3) } else { (3, 2) };
                            barrier.wait();
                            let reader = cached_f32(prepack, &source[..k * n], k, n).unwrap();
                            assert_eq!(
                                reader.as_slice(),
                                reference_transpose(&source[..k * n], k, n)
                            );
                            reader
                        })
                    })
                    .collect();
                jobs.into_iter()
                    .map(|job| job.join().unwrap())
                    .collect::<Vec<_>>()
            });
            let expected = if distinct { 8 } else { 1 };
            assert_eq!(prepack.weights.transposed_b.activity().1, expected);
            assert_eq!(prepack.f32_transpose_stats().0, expected);
            assert_eq!(
                prepack.weights.f32_transpose_hint.lock().unwrap().is_some(),
                !distinct
            );
            if distinct {
                let hits = prepack.weights.transposed_b.activity().0;
                for i in 0..8 {
                    cached_f32(&prepack, &source[..(i + 1) * 3], i + 1, 3).unwrap();
                }
                assert_eq!(
                    prepack.weights.transposed_b.activity().0,
                    hits + 8,
                    "multi-layout hits must update canonical recency"
                );
            }
            if !distinct {
                for other in &readers {
                    assert!(Arc::ptr_eq(&readers[0], other));
                }
            }
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn provider_f32_transposes_are_funded_and_warm_without_accounting() {
        if retention_control_in_child(
            "kernels::matmul::tests::provider_f32_transposes_are_funded_and_warm_without_accounting",
            Some("accelerate"),
        ) {
            return;
        }
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::{Attribute, NodeId};
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, MemoryGovernor};
        for (op, vector) in [
            ("MatMul", false),
            ("FusedMatMulBias", false),
            ("Gemm", false),
            ("FusedGemm", false),
            ("MatMul", true),
            ("FusedMatMulBias", true),
            ("FusedGemm", true),
        ] {
            let n = if vector { 1 } else { 7 };
            let bytes = (5 * n * 4) as u64;
            let gemm = op == "Gemm";
            let fused = op.starts_with("Fused");
            let books = RecordingGovernor::new(bytes);
            let resources =
                CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
                    .unwrap();
            let provider = crate::CpuExecutionProvider::with_resources(resources);
            let mut node = Node::new(NodeId(0), op, vec![], vec![]);
            if fused {
                node.domain = "com.microsoft".into();
            }
            if gemm {
                node.attributes.insert("transB".into(), Attribute::Int(1));
            }
            let shape = if vector {
                vec![5]
            } else if gemm {
                vec![n, 5]
            } else {
                vec![5, n]
            };
            let mut graph = Graph::new();
            let a_id = graph.create_named_value(
                "A",
                DataType::Float32,
                onnx_runtime_ir::static_shape([1, 5]),
            );
            let b_id = graph.create_named_value(
                "B",
                DataType::Float32,
                onnx_runtime_ir::static_shape(shape.iter().copied()),
            );
            graph.initializers.insert(
                b_id,
                onnx_runtime_ir::WeightRef::Inline(onnx_runtime_ir::TensorData::from_raw(
                    DataType::Float32,
                    shape.clone(),
                    vec![0; bytes as usize],
                )),
            );
            let mut forecast_node = node.clone();
            forecast_node.inputs = vec![Some(a_id), Some(b_id)];
            graph.insert_node(forecast_node);
            assert_eq!(
                weight_transpose_cache_predicted_bytes(&graph),
                bytes,
                "{op}: declared transpose footprint must cover the executed f32 route"
            );
            let mut kernel = provider
                .get_kernel(&node, &[vec![1, 5], shape.clone()], 17)
                .unwrap();
            kernel.set_constant_inputs(&[false, true]);
            let a = Owned::f32(&[1, 5], &[0.25; 5]);
            let b = Owned::f32(&shape, &vec![0.5; 5 * n]);
            let bias = Owned::f32(&[n], &vec![0.125; n]);
            let inputs = if fused {
                vec![a.view(), b.view(), bias.view()]
            } else {
                vec![a.view(), b.view()]
            };
            let mut out = Owned::zeros_f32(&if vector { vec![1] } else { vec![1, n] });
            let mut run = || {
                let result = kernel.execute(&inputs, &mut [out.view_mut()]);
                if result.is_ok() {
                    assert_eq!(
                        out.to_f32(),
                        vec![if fused { 0.75 } else { 0.625 }; n],
                        "{op}"
                    );
                }
                result
            };
            run().unwrap();
            assert_eq!(
                books.used(Tier::Host),
                bytes,
                "{op}: transpose must have an actual grant"
            );
            let before = (books.claims(), books.releases());
            for _ in 0..4096 {
                run().unwrap();
            }
            assert_eq!((books.claims(), books.releases()), before, "{op}");
            let kv = books
                .reserve(Tier::Host, bytes, MemoryRole::KvCache, HolderId::new(7))
                .unwrap();
            if gemm {
                let error = run().unwrap_err().to_string();
                assert!(error.contains("required B transpose admission"), "{error}");
            } else {
                run().unwrap();
            }
            assert_eq!(books.used(Tier::Host), bytes);
            drop(kv);
            run().unwrap();
            assert_eq!(books.used(Tier::Host), bytes);
            drop(kernel);
            drop(provider);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[test]
    fn half_transpose_grants_follow_shared_node_readers_and_host_pressure() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, MemoryGovernor};
        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            let books = RecordingGovernor::new(12);
            let resources =
                CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
                    .unwrap();
            let mut first = MatMulPrepack::new(resources.clone());
            first.set_constant_inputs(&[false, true]);
            let source = Owned::f16_bits(&[2, 3], &[1, 2, 3, 4, 5, 6]);
            let mut view = source.view();
            if format == HalfFormat::Bf16 {
                view.dtype = DataType::BFloat16;
            }
            let reader = first
                .transposed_b_half(&view, 2, 3, format)
                .unwrap()
                .unwrap();
            assert_eq!(reader.as_slice(), [1, 4, 2, 5, 3, 6]);
            assert_eq!(books.used(Tier::Host), 12);
            let calls = (books.claims(), books.releases());
            let mut peer = MatMulPrepack::new(resources);
            peer.set_constant_inputs(&[false, true]);
            assert!(
                peer.adopt_shareable_constant_state(first.shareable_constant_state().unwrap())
                    .unwrap()
            );
            drop(first);
            for _ in 0..4096 {
                let warm = peer
                    .transposed_b_half(&view, 2, 3, format)
                    .unwrap()
                    .unwrap();
                assert!(Arc::ptr_eq(&reader, &warm));
            }
            assert_eq!(
                (books.claims(), books.releases()),
                calls,
                "warm half readers must not request or return quota"
            );
            assert!(
                books
                    .reserve(Tier::Host, 12, MemoryRole::KvCache, HolderId::new(7))
                    .is_err(),
                "active readers protect their payload grant"
            );
            drop(reader);
            let kv = books
                .reserve(Tier::Host, 12, MemoryRole::KvCache, HolderId::new(7))
                .unwrap();
            assert_eq!(
                peer.weights.half_transposes.stats(),
                (0, 0),
                "any Host claimant may reclaim an idle half transpose"
            );
            let builds = peer.weights.half_transposes.activity().1;
            assert!(
                peer.transposed_b_half(&view, 2, 3, format)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(
                peer.weights.half_transposes.activity().1,
                builds,
                "admission refusal runs no transpose builder"
            );
            drop(kv);
            let reader = peer
                .transposed_b_half(&view, 2, 3, format)
                .unwrap()
                .unwrap();
            assert_eq!(
                peer.weights.half_transposes.activity().1,
                builds + 1,
                "a prior refusal must not poison later admission"
            );
            drop(peer);
            drop(source);
            assert_eq!(reader.as_slice(), [1, 4, 2, 5, 3, 6]);
            assert_eq!(
                books.used(Tier::Host),
                12,
                "the last executing reader independently owns its grant"
            );
            drop(reader);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[test]
    fn half_transpose_eligibility_and_validation_precede_any_admission() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_memory_governor::HolderId;
        let books = RecordingGovernor::new(24);
        let resources =
            CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
                .unwrap();
        let source = Owned::f16(&[2, 3], &[0.5; 6]);
        let mut prepack = MatMulPrepack::new(resources.clone());
        let baseline = books.claims();
        assert!(
            prepack
                .transposed_b_half(&source.view(), 2, 3, HalfFormat::F16)
                .unwrap()
                .is_none()
        );
        prepack.set_constant_inputs(&[false, true]);
        assert!(
            prepack
                .transposed_b_half(&source.view(), 2, 3, HalfFormat::Bf16)
                .unwrap()
                .is_none()
        );
        assert!(
            prepack
                .transposed_b_half(&source.view(), 3, 3, HalfFormat::F16)
                .unwrap()
                .is_none()
        );
        let mut foreign = source.view();
        foreign.device = onnx_runtime_ir::DeviceId::cuda(0);
        let error = prepack
            .transposed_b_half(&foreign, 2, 3, HalfFormat::F16)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("half transpose") && error.contains("host-accessible"),
            "{error}"
        );
        let empty = Owned::f16(&[0, 3], &[]);
        assert!(
            prepack
                .transposed_b_half(&empty.view(), 0, 3, HalfFormat::F16)
                .unwrap()
                .is_none()
        );
        let mut disabled = MatMulPrepack::new(
            resources.with_cache_retention(crate::CpuCacheRetention::new(false)),
        );
        disabled.set_constant_inputs(&[false, true]);
        assert!(
            disabled
                .transposed_b_half(&source.view(), 2, 3, HalfFormat::F16)
                .unwrap()
                .is_none()
        );
        assert_eq!(books.claims(), baseline);
        assert_eq!(prepack.weights.half_transposes.stats(), (0, 0));
        let first = prepack
            .transposed_b_half(&source.view(), 2, 3, HalfFormat::F16)
            .unwrap()
            .unwrap();
        let mut bf16 = source.view();
        bf16.dtype = DataType::BFloat16;
        let other = prepack
            .transposed_b_half(&bf16, 2, 3, HalfFormat::Bf16)
            .unwrap()
            .unwrap();
        assert!(
            !Arc::ptr_eq(&first, &other),
            "different storage interpretations need distinct identities"
        );
        assert_eq!(prepack.weights.half_transposes.stats(), (2, 24));
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn provider_half_transpose_is_funded_and_falls_back_after_host_reclamation() {
        if retention_control_in_child(
            "kernels::matmul::tests::provider_half_transpose_is_funded_and_falls_back_after_host_reclamation",
            Some("accelerate"),
        ) {
            return;
        }
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::NodeId;
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, MemoryGovernor};
        for op in ["MatMul", "FusedMatMulBias"] {
            let books = RecordingGovernor::new(70);
            let resources =
                CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
                    .unwrap();
            let provider = crate::CpuExecutionProvider::with_resources(resources);
            let mut node = Node::new(NodeId(0), op, vec![], vec![]);
            if op == "FusedMatMulBias" {
                node.domain = "com.microsoft".into();
            }
            let mut kernel = provider
                .get_kernel(&node, &[vec![1, 5], vec![5, 7]], 17)
                .unwrap();
            kernel.set_constant_inputs(&[false, true]);
            let a = Owned::f16(&[1, 5], &[0.25; 5]);
            let b = Owned::f16(&[5, 7], &[0.5; 35]);
            let bias = Owned::f32(&[7], &[0.125; 7]);
            let inputs = if op == "MatMul" {
                vec![a.view(), b.view()]
            } else {
                vec![a.view(), b.view(), bias.view()]
            };
            let mut output = Owned::zeros_f32(&[1, 7]);
            let mut run = || {
                kernel.execute(&inputs, &mut [output.view_mut()]).unwrap();
                assert_eq!(
                    output.to_f32(),
                    vec![if op == "MatMul" { 0.625 } else { 0.75 }; 7]
                );
            };
            run();
            assert_eq!(
                books.used(Tier::Host),
                70,
                "{op}: half transpose bytes must carry an actual Host grant"
            );
            let baseline = (books.claims(), books.releases());
            for _ in 0..4096 {
                run();
            }
            assert_eq!((books.claims(), books.releases()), baseline);
            let kv = books
                .reserve(Tier::Host, 70, MemoryRole::KvCache, HolderId::new(7))
                .unwrap();
            run();
            assert_eq!(
                books.used(Tier::Host),
                70,
                "declined optional preparation must use existing untransposed arithmetic"
            );
            drop(kv);
            run();
            assert_eq!(books.used(Tier::Host), 70);
            drop(kernel);
            drop(provider);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[test]
    fn warm_provider_execution_never_calls_weight_accounting() {
        if retention_control_in_child(
            "kernels::matmul::tests::warm_provider_execution_never_calls_weight_accounting",
            None,
        ) {
            return;
        }
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::NodeId;
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, MemoryGovernor};
        for retained in [true, false] {
            let books = RecordingGovernor::new(1024 * 1024);
            // Research MLAS can also retain a native panel. Admit its checked
            // size so repeated optional refusals cannot pollute a warm-hit test.
            #[cfg(feature = "mlas")]
            let panel_bytes = if retained && CpuBackend::auto_detect() == CpuBackend::Mlas {
                sgemm_pack_plan(3, 2).unwrap().owned_heap_bytes() as u64
            } else {
                0
            };
            #[cfg(not(feature = "mlas"))]
            let panel_bytes = 0u64;
            let cold_claims = 1 + usize::from(panel_bytes != 0);
            let source = books.weight_memory(HolderId::new(13));
            let resources = CpuMemoryResources::new(source, Default::default())
                .unwrap()
                .with_cache_retention(crate::CpuCacheRetention::new(retained));
            let provider = crate::CpuExecutionProvider::with_resources(resources);
            let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
            let mut kernel = provider
                .get_kernel(&node, &[vec![2, 3], vec![3, 2]], 17)
                .unwrap();
            kernel.set_constant_inputs(&[false, true]);
            let a = Owned::f32(&[2, 3], &[0.25; 6]);
            let b = Owned::f64(&[3, 2], &[0.5; 6]);
            let mut output = Owned::zeros_f32(&[2, 2]);
            let mut execute = || {
                kernel
                    .execute(&[a.view(), b.view()], &mut [output.view_mut()])
                    .unwrap();
                assert_eq!(output.to_f32(), [0.375; 4]);
            };
            let setup_claims = books.claims();
            assert_eq!(
                setup_claims, 1,
                "only the zero-byte source anchor is created during setup"
            );
            execute();
            assert_eq!(books.claims(), setup_claims + cold_claims);
            assert_eq!(books.releases(), usize::from(!retained));
            let repeats = if retained { 4096 } else { 32 };
            for _ in 0..repeats {
                execute();
            }
            assert_eq!(
                books.claims(),
                setup_claims + if retained { cold_claims } else { repeats + 1 }
            );
            assert_eq!(
                books.releases(),
                if retained { 0 } else { repeats + 1 },
                "warm cached readers never return the owning payload's grant; transient controls must exercise the observer"
            );
            assert_eq!(
                books.used(Tier::Host),
                if retained { 24 + panel_bytes } else { 0 }
            );
            drop(kernel);
            drop(provider);
            assert_eq!(books.used(Tier::Host), 0);
            assert_eq!(
                books.releases(),
                if retained { cold_claims } else { repeats + 1 }
            );
        }
    }

    #[test]
    fn copy_key_inline_layout_preserves_arbitrary_rank_fallback() {
        let storage = Owned::f64(&[6], &[1., 2., 3., 4., 5., 6.]);
        let shape = [1, 1, 1, 1, 2, 3];
        let strides = [6, 6, 6, 6, 3, 1];
        let mut view = storage.view();
        view.shape = &shape;
        view.strides = &strides;
        let key = MatMulCopyKey::<4>::from_view(&view);
        assert!(key.shape.spilled() && key.strides.spilled());
        assert_eq!(key.weight_bytes().unwrap(), 24);
        let matrix = storage.view();
        let inline = MatMulCopyKey::<4>::from_view(&matrix);
        assert!(!inline.shape.spilled() && !inline.strides.spilled());
        let mut prepack = MatMulPrepack::default();
        prepack.set_constant_inputs(&[true, false]);
        let first = prepack.dense(0, &view).unwrap();
        let next = prepack.dense(0, &view).unwrap();
        assert_eq!(&*first, [1., 2., 3., 4., 5., 6.]);
        assert_eq!(first.as_ptr(), next.as_ptr());
    }

    #[test]
    fn materialized_constant_copies_distinguish_complete_logical_views() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let layouts = [
            ([2, 3], [4, 1], [5., 6., 7., 9., 10., 11.]),
            ([2, 3], [1, 2], [5., 7., 9., 6., 8., 10.]),
            ([3, 2], [4, 1], [5., 6., 9., 10., 13., 14.]),
            ([2, 3], [0, 1], [5., 6., 7., 5., 6., 7.]),
            ([2, 3], [-4, 1], [5., 6., 7., 1., 2., 3.]),
        ];
        for (half_copy, slot) in [(false, 0), (false, 1), (true, 1)] {
            let payload_bytes = if half_copy { 12 } else { 24 };
            let books = LedgerGovernor::new(LeaseLedger::new(0, payload_bytes * 5, 0));
            let resources = CpuMemoryResources::new(
                crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
                Default::default(),
            )
            .unwrap();
            let source = if half_copy {
                Owned::f16(&[16], &(1..=16).map(|n| n as f32).collect::<Vec<_>>())
            } else {
                Owned::f64(&[16], &(1..=16).map(|n| n as f64).collect::<Vec<_>>())
            };
            let mut owner = MatMulPrepack::new(resources.clone());
            owner.set_constant_inputs(&[slot == 0, slot == 1]);
            let mut pointers = Vec::new();
            for (index, (shape, strides, expected)) in layouts.iter().enumerate() {
                let mut view = source.view();
                view.shape = shape;
                view.strides = strides;
                view.byte_offset = if half_copy { 8 } else { 32 };
                let (values, pointer) = if half_copy {
                    let reader = owner.contiguous_b_f16(&view).unwrap().unwrap();
                    (
                        reader
                            .as_slice()
                            .iter()
                            .map(|&v| half::f16::from_bits(v).to_f32())
                            .collect::<Vec<_>>(),
                        reader.as_slice().as_ptr() as usize,
                    )
                } else {
                    let reader = owner.dense(slot, &view).unwrap();
                    (reader.to_vec(), reader.as_ptr() as usize)
                };
                assert_eq!(
                    values, expected,
                    "layout {index}, half={half_copy}, slot={slot}"
                );
                assert!(
                    !pointers.contains(&pointer),
                    "different logical views cannot share a derived payload"
                );
                pointers.push(pointer);
                assert_eq!(books.used(Tier::Host), payload_bytes * (index as u64 + 1));
            }
            let mut peer = MatMulPrepack::new(resources);
            peer.set_constant_inputs(&[slot == 0, slot == 1]);
            assert!(
                peer.adopt_shareable_constant_state(owner.shareable_constant_state().unwrap())
                    .unwrap()
            );
            drop(owner);
            for (index, (shape, strides, expected)) in layouts.iter().enumerate() {
                let mut view = source.view();
                view.shape = shape;
                view.strides = strides;
                view.byte_offset = if half_copy { 8 } else { 32 };
                let (values, pointer) = if half_copy {
                    let reader = peer.contiguous_b_f16(&view).unwrap().unwrap();
                    (
                        reader
                            .as_slice()
                            .iter()
                            .map(|&v| half::f16::from_bits(v).to_f32())
                            .collect::<Vec<_>>(),
                        reader.as_slice().as_ptr() as usize,
                    )
                } else {
                    let reader = peer.dense(slot, &view).unwrap();
                    (reader.to_vec(), reader.as_ptr() as usize)
                };
                assert_eq!(values, expected);
                assert_eq!(
                    pointer, pointers[index],
                    "an equivalent logical view must reuse its payload"
                );
                assert_eq!(books.used(Tier::Host), payload_bytes * 5);
            }
            let activity = if half_copy {
                peer.weights.contiguous_b_f16.activity()
            } else {
                peer.weights.dense[slot].activity()
            };
            assert_eq!(
                activity,
                (5, 5),
                "five builds and five warm hits, including after owner adoption"
            );
            drop(peer);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[test]
    fn provider_matmul_family_reuses_only_equivalent_constant_views() {
        if retention_control_in_child(
            "kernels::matmul::tests::provider_matmul_family_reuses_only_equivalent_constant_views",
            None,
        ) {
            return;
        }
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::NodeId;
        let source = Owned::f64(&[12], &(1..=12).map(|n| n as f64).collect::<Vec<_>>());
        let shape = [2, 3];
        for op in ["MatMul", "Gemm", "FusedMatMulBias", "FusedGemm"] {
            for slot in [0, 1] {
                let provider = crate::CpuExecutionProvider::with_resources(Default::default());
                let mut node = Node::new(NodeId(0), op, vec![], vec![]);
                if matches!(op, "FusedMatMulBias" | "FusedGemm") {
                    node.domain = "com.microsoft".into();
                }
                let (m, k, n) = if slot == 0 { (2, 3, 2) } else { (2, 2, 3) };
                let a = Owned::f64(&[m, k], &vec![1.; m * k]);
                let b = Owned::f64(&[k, n], &vec![1.; k * n]);
                let bias = Owned::f64(&[n], &vec![0.; n]);
                let mut first = provider
                    .get_kernel(&node, &[vec![m, k], vec![k, n]], 17)
                    .unwrap();
                first.set_constant_inputs(&[slot == 0, slot == 1]);
                let mut peer = provider
                    .get_kernel(&node, &[vec![m, k], vec![k, n]], 17)
                    .unwrap();
                peer.set_constant_inputs(&[slot == 0, slot == 1]);
                for (index, strides) in [[4, 1], [1, 2], [4, 1]].iter().enumerate() {
                    let mut constant = source.view();
                    constant.shape = &shape;
                    constant.strides = strides;
                    let (av, bv) = if slot == 0 {
                        (constant, b.view())
                    } else {
                        (a.view(), constant)
                    };
                    let inputs = if op == "MatMul" {
                        vec![av, bv]
                    } else {
                        vec![av, bv, bias.view()]
                    };
                    let mut output = Owned::zeros(DataType::Float64, &[m, n]);
                    let kernel = if index == 0 {
                        first.as_ref()
                    } else {
                        peer.as_ref()
                    };
                    kernel.execute(&inputs, &mut [output.view_mut()]).unwrap();
                    let expected = match (slot, index == 1) {
                        (0, false) => vec![6., 6., 18., 18.],
                        (0, true) => vec![9., 9., 12., 12.],
                        (1, false) => vec![6., 8., 10., 6., 8., 10.],
                        _ => vec![3., 7., 11., 3., 7., 11.],
                    };
                    assert_eq!(output.to_f64(), expected, "{op} slot {slot} layout {index}");
                    if index == 0 {
                        assert!(
                            peer.adopt_shareable_constant_state(
                                first.shareable_constant_state().unwrap()
                            )
                            .unwrap()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn provider_matmul_family_shape_variants_share_one_operand_grant() {
        if retention_control_in_child(
            "kernels::matmul::tests::provider_matmul_family_shape_variants_share_one_operand_grant",
            None,
        ) {
            return;
        }
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::NodeId;
        use onnx_runtime_memory_api::{MemoryError, MemoryRole, Tier};
        use onnx_runtime_memory_governor::{
            HolderId, LeaseLedger, LedgerGovernor, MemoryAuthorityId, MemoryGovernor, MemoryLease,
        };
        // No pressure callback can hide duplicate preparation by evicting a
        // sibling. Both variants must execute under exactly one payload budget.
        struct FixedBooks(LedgerGovernor);
        impl MemoryGovernor for FixedBooks {
            fn authority_id(&self, tier: Tier) -> MemoryAuthorityId {
                self.0.authority_id(tier)
            }
            fn reserve(
                &self,
                tier: Tier,
                bytes: u64,
                role: MemoryRole,
                holder: HolderId,
            ) -> std::result::Result<MemoryLease, MemoryError> {
                self.0.reserve(tier, bytes, role, holder)
            }
            fn used(&self, tier: Tier) -> u64 {
                self.0.used(tier)
            }
            fn available(&self, tier: Tier) -> u64 {
                self.0.available(tier)
            }
        }
        for op in ["MatMul", "Gemm", "FusedMatMulBias", "FusedGemm"] {
            for constant_slot in [0, 1] {
                let books = FixedBooks(LedgerGovernor::new(LeaseLedger::new(0, 24, 0)));
                let memory =
                    crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap();
                assert!(!memory.supports_pressure_reclaim());
                let resources = CpuMemoryResources::new(memory, Default::default()).unwrap();
                let provider = crate::CpuExecutionProvider::with_resources(resources);
                let mut node = Node::new(NodeId(0), op, vec![], vec![]);
                if matches!(op, "FusedMatMulBias" | "FusedGemm") {
                    node.domain = "com.microsoft".into();
                }
                let (a_shape, b_shape, next_a, next_b) = if constant_slot == 0 {
                    ([2, 3], [3, 4], [2, 3], [3, 1])
                } else {
                    ([4, 3], [3, 2], [1, 3], [3, 2])
                };
                let a = Owned::f64(&a_shape, &vec![0.25; a_shape.iter().product()]);
                let b = Owned::f64(&b_shape, &vec![0.5; b_shape.iter().product()]);
                let a2 = Owned::f64(&next_a, &vec![0.25; next_a.iter().product()]);
                let b2 = Owned::f64(&next_b, &vec![0.5; next_b.iter().product()]);
                let bias = Owned::f64(&[b_shape[1]], &vec![0.; b_shape[1]]);
                let bias2 = Owned::f64(&[next_b[1]], &vec![0.; next_b[1]]);
                let mut first = provider
                    .get_kernel(
                        &node,
                        &[a_shape.to_vec(), b_shape.to_vec(), vec![b_shape[1]]],
                        17,
                    )
                    .unwrap();
                let mut peer = provider
                    .get_kernel(
                        &node,
                        &[next_a.to_vec(), next_b.to_vec(), vec![next_b[1]]],
                        17,
                    )
                    .unwrap();
                let flags = [constant_slot == 0, constant_slot == 1, true];
                first.set_constant_inputs(&flags);
                peer.set_constant_inputs(&flags);
                let run = |kernel: &dyn Kernel,
                           a: TensorView<'_>,
                           b: TensorView<'_>,
                           bias: TensorView<'_>,
                           m: usize,
                           n: usize| {
                    let mut output = Owned::zeros(DataType::Float64, &[m, n]);
                    let inputs = if op == "MatMul" {
                        vec![a, b]
                    } else {
                        vec![a, b, bias]
                    };
                    kernel.execute(&inputs, &mut [output.view_mut()]).unwrap();
                    assert_eq!(output.to_f64(), vec![0.375; m * n]);
                };
                run(
                    first.as_ref(),
                    a.view(),
                    b.view(),
                    bias.view(),
                    a_shape[0],
                    b_shape[1],
                );
                assert_eq!(books.used(Tier::Host), 24);
                if let Some(state) = first.shareable_constant_state() {
                    assert!(peer.adopt_shareable_constant_state(state).unwrap());
                }
                let peer_a = if constant_slot == 0 {
                    a.view()
                } else {
                    a2.view()
                };
                let peer_b = if constant_slot == 1 {
                    b.view()
                } else {
                    b2.view()
                };
                run(
                    peer.as_ref(),
                    peer_a,
                    peer_b,
                    bias2.view(),
                    next_a[0],
                    next_b[1],
                );
                assert_eq!(
                    books.used(Tier::Host),
                    24,
                    "{op} operand {constant_slot}: shape variants must share one grant"
                );
                run(
                    first.as_ref(),
                    a.view(),
                    b.view(),
                    bias.view(),
                    a_shape[0],
                    b_shape[1],
                );
                drop(first);
                assert_eq!(
                    books.used(Tier::Host),
                    24,
                    "a surviving variant owns the shared preparation"
                );
                drop(peer);
                drop(provider);
                assert_eq!(books.used(Tier::Host), 0);
            }
        }
    }

    #[test]
    fn prepared_operands_keep_funded_readers_and_pack_identity_through_concurrent_compute() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, MemoryGovernor};
        let backends = vec![CpuBackend::Generic, CpuBackend::auto_detect()];
        #[cfg(feature = "mlas")]
        let backends = {
            let mut values = backends;
            values.push(CpuBackend::Mlas);
            values
        };
        for backend in backends {
            const CAP: u64 = 1 << 20;
            let books = RecordingGovernor::new_host(CAP);
            let resources =
                CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
                    .unwrap();
            let mut kernel = MatMulKernel::new(resources, None);
            kernel.set_constant_inputs(&[false, true]);
            let a_values = [1f32; 85];
            let a = DenseOperand::borrowed(&a_values);
            let scratch = DenseOperand(DenseOperandStorage::Direct(Cow::Owned(vec![1.; 153])));
            assert_eq!(scratch.scratch_bytes(), Some(612));
            let before = (books.claims(), books.releases());
            let mut output = vec![73.; 45];
            kernel
                .execute_prepared_operands_with_backend(
                    &a,
                    &[5, 17],
                    &scratch,
                    &[17, 9],
                    &mut output,
                    backend,
                )
                .unwrap();
            assert_eq!(output, [17.; 45]);
            assert_eq!(
                (books.claims(), books.releases()),
                before,
                "unfunded scratch has no immutable packing identity"
            );
            let b1 = Owned::f32(&[9, 17], &[1.; 153]).with_view(&[17, 9], &[1, 17]);
            let view1 = b1.view();
            let b2 = Owned::f32(&[9, 17], &[2.; 153]).with_view(&[17, 9], &[1, 17]);
            let view2 = b2.view();
            let first = kernel.prepare_operand(1, &view1).unwrap();
            let second = kernel.prepare_operand(1, &view2).unwrap();
            assert_eq!(first.scratch_bytes(), Some(0));
            assert_eq!(second.scratch_bytes(), Some(0));
            for (b, expected) in [(&first, 17.), (&second, 34.)] {
                kernel
                    .execute_prepared_operands_with_backend(
                        &a,
                        &[5, 17],
                        b,
                        &[17, 9],
                        &mut output,
                        backend,
                    )
                    .unwrap();
                assert_eq!(output, [expected; 45]);
            }
            #[cfg(feature = "mlas")]
            if backend == CpuBackend::Mlas {
                assert_eq!(
                    kernel.prepack.weights.packed_b.stats().0,
                    2,
                    "forced MLAS must retain two distinct funded identities"
                );
            }
            let held = books.used(Tier::Host);
            let calls = (books.claims(), books.releases());
            let entered = std::sync::Barrier::new(3);
            std::thread::scope(|scope| {
                for (b, expected) in [(&first, 17.), (&second, 34.)] {
                    let (kernel, a, entered) = (&kernel, &a, &entered);
                    scope.spawn(move || {
                        let mut output = [73.; 45];
                        entered.wait();
                        for _ in 0..64 {
                            kernel
                                .execute_prepared_operands_with_backend(
                                    a,
                                    &[5, 17],
                                    b,
                                    &[17, 9],
                                    &mut output,
                                    backend,
                                )
                                .unwrap();
                            assert_eq!(output, [expected; 45]);
                        }
                    });
                }
                entered.wait();
            });
            assert_eq!(books.used(Tier::Host), held);
            assert_eq!((books.claims(), books.releases()), calls);
            assert!(
                books
                    .reserve(Tier::Host, CAP, MemoryRole::KvCache, HolderId::new(7))
                    .is_err(),
                "active prepared readers cannot be revoked by pressure"
            );
            assert_eq!(
                books.used(Tier::Host),
                1224,
                "both active converted payloads keep their grants after cache eviction"
            );
            kernel
                .execute_prepared_operands_with_backend(
                    &a,
                    &[5, 17],
                    &first,
                    &[17, 9],
                    &mut output,
                    backend,
                )
                .unwrap();
            assert_eq!(output, [17.; 45]);
            drop(first);
            drop(second);
            let pressure = books
                .reserve(Tier::Host, CAP, MemoryRole::KvCache, HolderId::new(7))
                .unwrap();
            drop(pressure);
            assert_eq!(books.used(Tier::Host), 0);
            let rebuilt = kernel.prepare_operand(1, &view2).unwrap();
            kernel
                .execute_prepared_operands_with_backend(
                    &a,
                    &[5, 17],
                    &rebuilt,
                    &[17, 9],
                    &mut output,
                    backend,
                )
                .unwrap();
            assert_eq!(output, [34.; 45]);
            drop(rebuilt);
            drop(kernel);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[test]
    fn prepared_operands_validate_spans_and_empty_products_before_compute() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_memory_governor::HolderId;
        let books = RecordingGovernor::new_host(0);
        let resources =
            CpuMemoryResources::new(books.weight_memory(HolderId::new(13)), Default::default())
                .unwrap();
        let mut kernel = MatMulKernel::new(resources, None);
        kernel.set_constant_inputs(&[false, true]);
        let empty = DenseOperand::borrowed(&[]);
        let b = DenseOperand::borrowed(&[1.; 153]);
        let calls = (books.claims(), books.releases());
        kernel
            .execute_prepared_operands(&empty, &[0, 5, 17], &b, &[1, 17, 9], &mut [])
            .unwrap();
        kernel
            .execute_prepared_operands(
                &empty,
                &[0, usize::MAX, usize::MAX],
                &empty,
                &[1, usize::MAX, 0],
                &mut [],
            )
            .unwrap();
        let mut output = [73.; 10];
        kernel
            .execute_prepared_operands(&empty, &[5, 0], &empty, &[0, 2], &mut output)
            .unwrap();
        assert_eq!(output, [0.; 10]);
        assert_eq!(
            (books.claims(), books.releases()),
            calls,
            "empty and zero-K compute must not pack or issue zero-byte grants"
        );
        let a = DenseOperand::borrowed(&[1.; 3]);
        let right = DenseOperand::borrowed(&[1.; 6]);
        let mut output = [73.; 2];
        for (a_shape, b_shape) in [
            (&[2, 3][..], &[3, 2][..]),
            (&[usize::MAX, 2][..], &[2, 1][..]),
            (&[usize::MAX / 4, 1][..], &[1, 2][..]),
        ] {
            let error = kernel
                .execute_prepared_operands(&a, a_shape, &right, b_shape, &mut output)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("MatMul")
                    && (error.contains("dimensions")
                        || error.contains("complete prepared operand")),
                "{error}"
            );
            assert_eq!(output, [73.; 2]);
        }
        let error = kernel
            .execute_prepared_operands(&a, &[1, 3], &right, &[3, 2], &mut [73.])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("output") && error.contains("complete result buffer"),
            "{error}"
        );
    }

    #[test]
    fn matmul_geometry_checks_cross_broadcast_and_matrix_overflow() {
        for (a, b) in [
            (vec![usize::MAX, 2], vec![2, 1]),
            (vec![1, 2], vec![2, usize::MAX]),
            (vec![usize::MAX / 2, 1], vec![1, 3]),
            (vec![usize::MAX / 2, 1, 1, 1], vec![1, usize::MAX / 2, 1, 1]),
        ] {
            let error = match matmul_geometry_from_shapes(&a, &b) {
                Ok(_) => panic!("overflowed geometry must be rejected: {a:?} {b:?}"),
                Err(error) => error.to_string(),
            };
            assert!(
                error.contains("overflow") && error.contains("reduce"),
                "{error}"
            );
        }
    }

    #[test]
    fn matmul_shared_weights_reject_foreign_sources_and_constant_flags() {
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor};
        let books = LedgerGovernor::new(LeaseLedger::new(0, 1024, 0));
        let resources = CpuMemoryResources::new(
            crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
            Default::default(),
        )
        .unwrap();
        let mut owner = MatMulPrepack::new(resources.clone());
        owner.set_constant_inputs(&[false, true]);
        let state = owner.shareable_constant_state().unwrap();
        let mut peer = MatMulPrepack::new(resources.clone());
        peer.set_constant_inputs(&[false, true]);
        assert!(
            !peer
                .adopt_shareable_constant_state(Arc::new(7usize))
                .unwrap()
        );
        assert!(!Arc::ptr_eq(&owner.weights, &peer.weights));
        assert!(peer.adopt_shareable_constant_state(state.clone()).unwrap());
        assert!(Arc::ptr_eq(&owner.weights, &peer.weights));
        let mut wrong_flags = MatMulPrepack::new(resources.clone());
        wrong_flags.set_constant_inputs(&[true, false]);
        assert!(
            wrong_flags
                .adopt_shareable_constant_state(state.clone())
                .unwrap_err()
                .to_string()
                .contains("constant flags")
        );
        assert!(!Arc::ptr_eq(&owner.weights, &wrong_flags.weights));
        let mut disabled = MatMulPrepack::new(
            resources.with_cache_retention(crate::CpuCacheRetention::new(false)),
        );
        disabled.set_constant_inputs(&[false, true]);
        assert!(disabled.shareable_constant_state().is_none());
        assert!(
            !disabled
                .adopt_shareable_constant_state(state.clone())
                .unwrap()
        );
        let other = CpuMemoryResources::new(
            crate::DerivedWeightMemory::governed(&books, HolderId::new(14)).unwrap(),
            Default::default(),
        )
        .unwrap();
        let mut foreign = MatMulPrepack::new(other);
        foreign.set_constant_inputs(&[false, true]);
        assert!(
            foreign
                .adopt_shareable_constant_state(state)
                .unwrap_err()
                .to_string()
                .contains("derived-weight grant source")
        );
        assert!(!Arc::ptr_eq(&owner.weights, &foreign.weights));
        let mut dynamic = MatMulPrepack::default();
        dynamic.set_constant_inputs(&[false, false]);
        assert!(dynamic.shareable_constant_state().is_none());
    }

    #[test]
    fn matmul_funded_transpose_survives_shared_owners_and_executing_readers() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let books = LedgerGovernor::new(LeaseLedger::new(0, 48, 0));
        let resources = CpuMemoryResources::new(
            crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
            Default::default(),
        )
        .unwrap();
        let mut first = MatMulPrepack::new(resources.clone());
        first.set_constant_inputs(&[false, true]);
        let storage = Owned::f64(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let view = storage.view();
        let reader = first.dense(1, &view).unwrap();
        let transpose = first.transposed_b(&view, 3, 2).unwrap().unwrap();
        assert_eq!(transpose.as_slice(), [1., 3., 5., 2., 4., 6.]);
        assert_eq!(books.used(Tier::Host), 48);
        let mut peer = MatMulPrepack::new(resources);
        peer.set_constant_inputs(&[false, true]);
        peer.adopt_shareable_constant_state(first.shareable_constant_state().unwrap())
            .unwrap();
        drop(first);
        let again = peer.transposed_b(&view, 3, 2).unwrap().unwrap();
        assert!(Arc::ptr_eq(&transpose, &again));
        drop(again);
        drop(reader);
        drop(peer);
        assert_eq!(
            books.used(Tier::Host),
            24,
            "only the executing transpose remains"
        );
        assert_eq!(transpose.as_slice(), [1., 3., 5., 2., 4., 6.]);
        drop(transpose);
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn matmul_shared_node_reuses_native_panels_between_activation_shapes() {
        if retention_control_in_child(
            "kernels::matmul::tests::matmul_shared_node_reuses_native_panels_between_activation_shapes",
            None,
        ) {
            return;
        }
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let (k, n) = (5, 3);
        let payload = mlas_sys::PackedBPlan::new(n, k).unwrap().owned_heap_bytes() as u64;
        for half in [false, true] {
            let books =
                LedgerGovernor::new(LeaseLedger::new(0, payload + if half { 60 } else { 0 }, 0));
            let resources = CpuMemoryResources::new(
                crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
                Default::default(),
            )
            .unwrap();
            let b = if half {
                Owned::f16(&[k, n], &[0.5; 15])
            } else {
                Owned::f32(&[k, n], &[0.5; 15])
            };
            let mut shared = None;
            let mut variants = Vec::new();
            for m in [3, 7, 4] {
                let a = if half {
                    Owned::f16(&[m, k], &vec![0.25; m * k])
                } else {
                    Owned::f32(&[m, k], &vec![0.25; m * k])
                };
                let mut kernel = MatMulKernel {
                    prepack: MatMulPrepack::new(resources.clone()),
                    flops: None,
                };
                kernel.set_constant_inputs(&[false, true]);
                if let Some(state) = &shared {
                    assert!(
                        kernel
                            .adopt_shareable_constant_state(Arc::clone(state))
                            .unwrap()
                    );
                }
                let mut output = Owned::zeros_f32(&[m, n]);
                kernel
                    .execute_with_backend(
                        &[a.view(), b.view()],
                        &mut [output.view_mut()],
                        CpuBackend::Mlas,
                    )
                    .unwrap();
                assert_eq!(output.to_f32(), vec![0.625; m * n]);
                assert_eq!(books.used(Tier::Host), payload);
                let cache = if half {
                    &kernel.prepack.weights.packed_b_from_half
                } else {
                    &kernel.prepack.weights.packed_b
                };
                assert_eq!(
                    cache.activity().1,
                    1,
                    "all activation shapes use the same native panel"
                );
                if shared.is_none() {
                    shared = kernel.shareable_constant_state();
                }
                variants.push(kernel);
            }
            drop(variants);
            assert_eq!(books.used(Tier::Host), payload);
            drop(shared);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[test]
    fn dense_prediction_covers_every_resource_bearing_matmul_factory() {
        let original = dense_matmul_graph(3, 2, DataType::Float64, true);
        let expected = 3 * 2 * 4;
        for op in ["MatMul", "Gemm", "FusedMatMulBias", "FusedGemm"] {
            let mut graph = original.clone();
            let node = graph.nodes.values_mut().next().unwrap();
            node.op_type = op.into();
            node.domain = if matches!(op, "FusedMatMulBias" | "FusedGemm") {
                "com.microsoft"
            } else {
                ""
            }
            .into();
            assert_eq!(matmul_dense_cache_predicted_bytes(&graph), expected, "{op}");
            let node = graph.nodes.values_mut().next().unwrap();
            node.domain = "unregistered.example".into();
            assert_eq!(
                matmul_dense_cache_predicted_bytes(&graph),
                0,
                "foreign-domain {op} is not our factory"
            );
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn provider_strided_half_copy_uses_host_authority() {
        if retention_control_in_child(
            "kernels::matmul::tests::provider_strided_half_copy_uses_host_authority",
            None,
        ) {
            return;
        }
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::NodeId;
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let (m, k, n) = (3, 17, 9);
        let a = Owned::f16(&[m, k], &vec![0.25; m * k]);
        let b = Owned::f16(&[k, 2 * n], &vec![0.5; k * 2 * n]);
        let shape = [k, n];
        let strides = [(2 * n) as i64, 2];
        let bytes = (k * n * 2) as u64;
        for retained in [true, false] {
            let books = LedgerGovernor::new(LeaseLedger::new(0, bytes, 0));
            let resources = CpuMemoryResources::new(
                crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
                Default::default(),
            )
            .unwrap()
            .with_cache_retention(crate::CpuCacheRetention::new(retained));
            let provider = crate::CpuExecutionProvider::with_resources(resources);
            let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
            let mut kernel = provider
                .get_kernel(&node, &[vec![m, k], vec![k, n]], 17)
                .unwrap();
            kernel.set_constant_inputs(&[false, true]);
            let run = || {
                let mut view = b.view();
                view.shape = &shape;
                view.strides = &strides;
                let mut output = Owned::zeros_f32(&[m, n]);
                kernel.execute(&[a.view(), view], &mut [output.view_mut()])?;
                assert_eq!(output.to_f32(), vec![k as f32 * 0.125; m * n]);
                Ok::<_, EpError>(())
            };
            run().unwrap();
            assert_eq!(books.used(Tier::Host), if retained { bytes } else { 0 });
            let kv = books
                .reserve(Tier::Host, bytes, MemoryRole::KvCache, HolderId::new(7))
                .unwrap();
            let error = run()
                .expect_err("occupied Host capacity must refuse the copy")
                .to_string();
            assert!(
                error.contains("MatMul constant f16 contiguous preparation admission"),
                "{error}"
            );
            drop(kv);
            run().unwrap();
            drop(kernel);
            drop(provider);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[test]
    fn half_copy_rejects_invalid_sources_before_admission_and_handles_empty_layouts() {
        use onnx_runtime_ir::DeviceId;
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let books = LedgerGovernor::new(LeaseLedger::new(0, 0, 0));
        let resources = CpuMemoryResources::new(
            crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
            Default::default(),
        )
        .unwrap();
        let mut prepack = MatMulPrepack::new(resources);
        prepack.set_constant_inputs(&[false, true]);
        let storage = Owned::f16(&[2, 4], &[0.; 8]);
        let mut view = storage.view();
        let shape = [2, 2];
        let strides = [4, 2];
        view.shape = &shape;
        view.strides = &strides;
        view.byte_offset = 1;
        let error = prepack.contiguous_b_f16(&view).err().unwrap().to_string();
        assert!(error.contains("byte_offset"), "{error}");
        view.byte_offset = 0;
        view.device = DeviceId::cuda(0);
        let error = prepack.contiguous_b_f16(&view).err().unwrap().to_string();
        assert!(error.contains("host-accessible storage"), "{error}");
        assert_eq!(prepack.weights.contiguous_b_f16.activity(), (0, 0));
        assert_eq!(books.used(Tier::Host), 0);
        view.device = DeviceId::cpu();
        let shape = [2, 0];
        view.shape = &shape;
        let empty = prepack.contiguous_b_f16(&view).unwrap().unwrap();
        assert!(empty.as_slice().is_empty());
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[test]
    fn half_copy_grants_cover_readers_refusal_and_idle_reclamation() {
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let values = [0.25f32, -0.5, 1.0, 2.0, -4.0, 0.0];
        let storage: Vec<f32> = values.iter().rev().flat_map(|&v| [v, -99.]).collect();
        let weight = Owned::f16(&[3, 4], &storage);
        let mut view = weight.view();
        let shape = [3, 2];
        let strides = [-4, -2];
        view.shape = &shape;
        view.strides = &strides;
        view.byte_offset = 20;
        let expected = values.map(|v| half::f16::from_f32(v).to_bits());
        let bytes = values.len() * 2;
        for retained in [false, true] {
            let books = LedgerGovernor::new(LeaseLedger::new(0, bytes as u64, 0));
            let resources = CpuMemoryResources::new(
                crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
                Default::default(),
            )
            .unwrap()
            .with_cache_retention(crate::CpuCacheRetention::new(retained));
            let mut prepack = MatMulPrepack::new(resources);
            prepack.set_constant_inputs(&[false, true]);
            assert_eq!(books.used(Tier::Host), 0);
            let reader = prepack.contiguous_b_f16(&view).unwrap().unwrap();
            assert_eq!(reader.as_slice(), &expected);
            assert_eq!(books.used(Tier::Host), bytes as u64);
            assert_eq!(
                prepack.weights.contiguous_b_f16.stats().1 as u64,
                if retained { bytes as u64 } else { 0 }
            );
            assert!(
                books
                    .reserve(Tier::Host, 1, MemoryRole::KvCache, HolderId::new(7))
                    .is_err(),
                "a live call must retain the complete grant"
            );
            if retained {
                let warm = prepack.contiguous_b_f16(&view).unwrap().unwrap();
                assert_eq!(reader.as_slice().as_ptr(), warm.as_slice().as_ptr());
            } else {
                let error = prepack
                    .contiguous_b_f16(&view)
                    .err()
                    .expect("a second transient needs its own grant");
                assert!(
                    error
                        .to_string()
                        .contains("MatMul constant f16 contiguous preparation admission")
                );
            }
            drop(reader);
            if retained {
                let kv = books
                    .reserve(
                        Tier::Host,
                        bytes as u64,
                        MemoryRole::KvCache,
                        HolderId::new(7),
                    )
                    .unwrap();
                assert_eq!(
                    prepack.weights.contiguous_b_f16.stats().1 as u64,
                    0,
                    "ordinary KV pressure must reclaim idle derived constants"
                );
                let before = prepack.weights.contiguous_b_f16.activity();
                assert!(prepack.contiguous_b_f16(&view).is_err());
                assert_eq!(
                    prepack.weights.contiguous_b_f16.activity(),
                    before,
                    "refusal must precede conversion"
                );
                drop(kv);
            }
            let reader = prepack.contiguous_b_f16(&view).unwrap().unwrap();
            let witness = prepack.weights.contiguous_b_f16.test_first();
            assert_eq!(witness.is_some(), retained);
            drop(witness);
            drop(prepack);
            assert_eq!(
                books.used(Tier::Host),
                bytes as u64,
                "kernel drop cannot revoke an executing reader"
            );
            assert_eq!(reader.as_slice(), &expected);
            drop(reader);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[test]
    fn dense_operand_grants_cover_retained_and_call_owned_readers() {
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let values = [0.25f64, -0.5, 1.0, 2.0, -4.0, 0.0];
        let weight = Owned::f64(&[3, 2], &values);
        let view = weight.view();
        let bytes = values.len() * 4;
        for retained in [false, true] {
            let books = LedgerGovernor::new(LeaseLedger::new(0, bytes as u64, 0));
            let resources = CpuMemoryResources::new(
                crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
                Default::default(),
            )
            .unwrap()
            .with_cache_retention(crate::CpuCacheRetention::new(retained));
            let mut prepack = MatMulPrepack::new(resources);
            prepack.set_constant_inputs(&[false, true]);
            assert_eq!(books.used(Tier::Host), 0);
            let reader = prepack.dense(1, &view).unwrap();
            assert_eq!(&*reader, &values.map(|value| value as f32));
            assert_eq!(books.used(Tier::Host), bytes as u64);
            assert_eq!(
                prepack.dense_live_bytes(),
                if retained { bytes as u64 } else { 0 }
            );
            assert!(
                books
                    .reserve(Tier::Host, 1, MemoryRole::KvCache, HolderId::new(7))
                    .is_err(),
                "a live call must retain the complete grant"
            );
            if retained {
                let warm = prepack.dense(1, &view).unwrap();
                assert_eq!(reader.as_ptr(), warm.as_ptr());
            } else {
                let error = prepack
                    .dense(1, &view)
                    .err()
                    .expect("a second transient needs its own grant");
                assert!(
                    error
                        .to_string()
                        .contains("MatMul constant B widening admission")
                );
            }
            drop(reader);
            if retained {
                let kv = books
                    .reserve(
                        Tier::Host,
                        bytes as u64,
                        MemoryRole::KvCache,
                        HolderId::new(7),
                    )
                    .unwrap();
                assert_eq!(
                    prepack.dense_live_bytes(),
                    0,
                    "ordinary KV pressure must reclaim idle derived constants"
                );
                let before = prepack.weights.dense[1].activity();
                assert!(prepack.dense(1, &view).is_err());
                assert_eq!(
                    prepack.weights.dense[1].activity(),
                    before,
                    "refusal must precede conversion"
                );
                drop(kv);
            }
            let reader = prepack.dense(1, &view).unwrap();
            let witness = prepack.dense_payload(1);
            assert_eq!(witness.is_some(), retained);
            drop(witness);
            drop(prepack);
            assert_eq!(
                books.used(Tier::Host),
                bytes as u64,
                "kernel drop cannot revoke an executing reader"
            );
            assert_eq!(&*reader, &values.map(|value| value as f32));
            drop(reader);
            assert_eq!(books.used(Tier::Host), 0);
        }
        // Borrowing an unchanged constant f32 input requires no grant at all.
        let books = LedgerGovernor::new(LeaseLedger::new(0, 0, 0));
        let resources = CpuMemoryResources::new(
            crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
            Default::default(),
        )
        .unwrap();
        let mut prepack = MatMulPrepack::new(resources);
        prepack.set_constant_inputs(&[false, true]);
        let f32_weight = Owned::f32(&[3, 2], &[0.25; 6]);
        let f32_view = f32_weight.view();
        let reader = prepack.dense(1, &f32_view).unwrap();
        assert_eq!(reader.as_ptr(), f32_view.data_ptr::<f32>());
        assert_eq!(books.used(Tier::Host), 0);
        assert!(!prepack.dense_is_filled(1));
    }

    #[test]
    fn original_view_transpose_identity_does_not_pin_a_conversion() {
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let books = LedgerGovernor::new(LeaseLedger::new(0, 48, 0));
        let resources = CpuMemoryResources::new(
            crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
            Default::default(),
        )
        .unwrap();
        let mut prepack = MatMulPrepack::new(resources);
        prepack.set_constant_inputs(&[false, true]);
        let weight = Owned::f64(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let view = weight.view();
        let reader = prepack.dense(1, &view).unwrap();
        let source = match &reader.0 {
            DenseOperandStorage::Funded(owner) => Arc::downgrade(owner),
            _ => panic!("funded"),
        };
        let transpose = prepack.transposed_b(&view, 3, 2).unwrap().unwrap();
        drop(reader);
        let kv = books
            .reserve(Tier::Host, 24, MemoryRole::KvCache, HolderId::new(7))
            .unwrap();
        assert!(
            source.upgrade().is_none(),
            "original-view identity must not pin the conversion"
        );
        assert_eq!(transpose.as_slice(), [1., 3., 5., 2., 4., 6.]);
        let warm = prepack.transposed_b(&view, 3, 2).unwrap().unwrap();
        assert!(Arc::ptr_eq(&warm, &transpose));
        drop(warm);
        drop(kv);
        drop(prepack);
        assert_eq!(books.used(Tier::Host), 24);
        drop(transpose);
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[test]
    fn half_copy_concurrent_builders_share_one_granted_payload() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let books = LedgerGovernor::new(LeaseLedger::new(0, 12, 0));
        let resources = CpuMemoryResources::new(
            crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
            Default::default(),
        )
        .unwrap();
        let mut prepack = MatMulPrepack::new(resources);
        prepack.set_constant_inputs(&[false, true]);
        let weight = Owned::f16(&[3, 4], &[0.25; 12]);
        let barrier = std::sync::Barrier::new(8);
        let readers = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        let mut view = weight.view();
                        let shape = [3, 2];
                        let strides = [4, 2];
                        view.shape = &shape;
                        view.strides = &strides;
                        prepack.contiguous_b_f16(&view).unwrap().unwrap()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(
            readers
                .iter()
                .all(|reader| Arc::ptr_eq(reader, &readers[0]))
        );
        assert_eq!(prepack.weights.contiguous_b_f16.activity().1, 1);
        assert_eq!(books.used(Tier::Host), 12);
        drop(prepack);
        assert_eq!(books.used(Tier::Host), 12);
        drop(readers);
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[test]
    fn dense_operand_concurrent_builders_share_one_granted_payload() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let books = LedgerGovernor::new(LeaseLedger::new(0, 24, 0));
        let resources = CpuMemoryResources::new(
            crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
            Default::default(),
        )
        .unwrap();
        let mut prepack = MatMulPrepack::new(resources);
        prepack.set_constant_inputs(&[false, true]);
        let weight = Owned::f64(&[3, 2], &[0.25; 6]);
        let barrier = std::sync::Barrier::new(8);
        let readers = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        match prepack.dense(1, &weight.view()).unwrap().0 {
                            DenseOperandStorage::Funded(reader) => reader,
                            _ => panic!("constant widening must return a grant-carrying reader"),
                        }
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(
            readers
                .iter()
                .all(|reader| Arc::ptr_eq(reader, &readers[0]))
        );
        assert_eq!(prepack.weights.dense[1].activity().1, 1);
        assert_eq!(books.used(Tier::Host), 24);
        drop(prepack);
        assert_eq!(books.used(Tier::Host), 24);
        drop(readers);
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[test]
    fn provider_matmul_family_widening_uses_its_host_authority() {
        if retention_control_in_child(
            "kernels::matmul::tests::provider_matmul_family_widening_uses_its_host_authority",
            None,
        ) {
            return;
        }
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::NodeId;
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let a = Owned::f64(&[2, 3], &[0.25; 6]);
        let b = Owned::f64(&[3, 2], &[0.5; 6]);
        let bias = Owned::f64(&[2], &[0.; 2]);
        for op in ["MatMul", "Gemm", "FusedMatMulBias", "FusedGemm"] {
            for retained in [true, false] {
                let books = LedgerGovernor::new(LeaseLedger::new(0, 24, 0));
                let resources = CpuMemoryResources::new(
                    crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
                    Default::default(),
                )
                .unwrap()
                .with_cache_retention(crate::CpuCacheRetention::new(retained));
                let provider = crate::CpuExecutionProvider::with_resources(resources);
                let mut node = Node::new(NodeId(0), op, vec![], vec![]);
                if matches!(op, "FusedMatMulBias" | "FusedGemm") {
                    node.domain = "com.microsoft".into();
                }
                let mut kernel = provider
                    .get_kernel(&node, &[vec![2, 3], vec![3, 2], vec![2]], 17)
                    .unwrap();
                kernel.set_constant_inputs(&[false, true, true]);
                let run = || {
                    let mut out = Owned::zeros(DataType::Float64, &[2, 2]);
                    let inputs = if op == "MatMul" {
                        vec![a.view(), b.view()]
                    } else {
                        vec![a.view(), b.view(), bias.view()]
                    };
                    kernel.execute(&inputs, &mut [out.view_mut()])?;
                    assert_eq!(out.to_f64(), vec![0.375; 4]);
                    Ok::<_, EpError>(())
                };
                run().unwrap();
                assert_eq!(
                    books.used(Tier::Host),
                    if retained { 24 } else { 0 },
                    "{op}: actual factory must capture its grant source"
                );
                let kv = books
                    .reserve(Tier::Host, 24, MemoryRole::KvCache, HolderId::new(7))
                    .unwrap();
                let error = run()
                    .expect_err("full Host capacity must refuse conversion before allocation")
                    .to_string();
                assert!(
                    error.contains("MatMul constant B widening admission"),
                    "{op}: {error}"
                );
                assert_eq!(books.used(Tier::Host), 24);
                drop(kv);
                run().unwrap();
                drop(kernel);
                drop(provider);
                assert_eq!(books.used(Tier::Host), 0);
            }
        }
    }

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

    /// Every non-f32 constant contributes one node-owned dense payload;
    /// activation shapes do not multiply the retained preparation estimate.
    #[test]
    fn dense_cache_predictor_counts_only_constant_non_f32_operands() {
        let (k, n) = (10usize, 7usize);
        let bytes = (k * n * 4) as u64;
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

    /// Mutation guard: `FusedMatMulBias` ships in the optimizer's
    /// `com.microsoft` domain, never the default domain. A predictor gated on
    /// `node.is_default_domain()` alone silently zeros every fused node here —
    /// the exact defect #1702 fixed in `node_weight_transpose_cache_bytes` but
    /// which this predictor still carried until this test caught it. `MatMul`
    /// (default domain) and `FusedMatMulBias` (`com.microsoft`) with identical
    /// geometry must be budgeted identically.
    #[test]
    fn fused_matmul_bias_dense_cache_is_budgeted_in_its_shipping_domain() {
        use onnx_runtime_ir::{NodeId, TensorData, WeightRef, static_shape};
        let (k, n) = (10usize, 7usize);

        let matmul_bytes =
            matmul_dense_cache_predicted_bytes(&dense_matmul_graph(k, n, DataType::Float16, true));
        assert_ne!(matmul_bytes, 0, "sanity: MatMul itself must be budgeted");

        let mut graph = Graph::new();
        let a = graph.create_named_value("A", DataType::Float32, static_shape([1, k]));
        let b = graph.create_named_value("B", DataType::Float16, static_shape([k, n]));
        let bias = graph.create_named_value("C", DataType::Float32, static_shape([n]));
        let y = graph.create_named_value("Y", DataType::Float32, static_shape([1, n]));
        graph.add_input(a);
        graph.add_input(b);
        graph.add_input(bias);
        let mut node = Node::new(
            NodeId(0),
            "FusedMatMulBias",
            vec![Some(a), Some(b), Some(bias)],
            vec![y],
        );
        node.domain = "com.microsoft".to_string();
        graph.insert_node(node);
        graph.add_output(y);
        graph.set_initializer(
            b,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float16,
                vec![k, n],
                vec![0u8; k * n * 2],
            )),
        );

        let fused_bytes = matmul_dense_cache_predicted_bytes(&graph);
        assert_eq!(
            fused_bytes, matmul_bytes,
            "FusedMatMulBias must be budgeted exactly like MatMul in its \
             shipping (`com.microsoft`) domain -- a 0 here means the domain \
             gate silently excludes every fused node again"
        );
    }

    /// Multiple activation shapes keep one funded dense preparation per node.
    #[test]
    fn predicted_dense_bytes_equal_actual_after_matmul_execution() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let (k, n) = (7, 5);
        let graph = dense_matmul_graph(k, n, DataType::Float64, true);
        let predicted = matmul_dense_cache_predicted_bytes(&graph);
        assert_eq!(predicted, (k * n * 4) as u64);
        let books = LedgerGovernor::new(LeaseLedger::new(0, predicted, 0));
        let resources = CpuMemoryResources::new(
            crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
            Default::default(),
        )
        .unwrap();
        let b = Owned::f64(&[k, n], &vec![0.5; k * n]);
        let mut shared = None;
        let mut variants = Vec::new();
        for m in [4, 1, 7] {
            let a = Owned::f64(&[m, k], &vec![0.25; m * k]);
            let mut out = Owned::zeros(DataType::Float64, &[m, n]);
            let mut kernel = MatMulKernel {
                prepack: MatMulPrepack::new(resources.clone()),
                flops: None,
            };
            kernel.set_constant_inputs(&[false, true]);
            if let Some(state) = &shared {
                assert!(
                    kernel
                        .adopt_shareable_constant_state(Arc::clone(state))
                        .unwrap()
                );
            }
            kernel
                .execute_with_backend(
                    &[a.view(), b.view()],
                    &mut [out.view_mut()],
                    CpuBackend::Generic,
                )
                .unwrap();
            assert_eq!(out.to_f64(), vec![k as f64 * 0.125; m * n]);
            assert_eq!(books.used(Tier::Host), predicted);
            assert_eq!(kernel.prepack.dense_live_bytes(), predicted);
            if shared.is_none() {
                shared = kernel.shareable_constant_state();
            }
            variants.push(kernel);
        }
        assert_eq!(variants.len(), 3);
        assert!(
            variants
                .iter()
                .all(|kernel| Arc::ptr_eq(&kernel.prepack.weights, &variants[0].prepack.weights))
        );
        drop(variants);
        assert_eq!(
            books.used(Tier::Host),
            predicted,
            "the exported node owner keeps its preparation alive"
        );
        drop(shared);
        assert_eq!(books.used(Tier::Host), 0);
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
            // Each kernel captures its own verdict at construction.
            let a = Owned::f32(&[m, k], &a_data);
            let b = Owned::f16(&[k, n], &b_data);
            let mut out = Owned::zeros_f32(&[m, n]);
            let mut kernel = MatMulKernel::new(
                CpuMemoryResources::default()
                    .with_cache_retention(crate::CpuCacheRetention::new(enabled)),
                None,
            );
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

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn constant_preparation_warms_transpose_only_for_own_admitted_provider() {
        if retention_control_in_child(
            "kernels::matmul::tests::constant_preparation_warms_transpose_only_for_own_admitted_provider",
            None,
        ) {
            return;
        }
        use onnx_runtime_ep_api::{ExecutionProvider, KernelConstantInput};
        use onnx_runtime_ir::NodeId;
        let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
        for (admitted, constant_a) in [(false, false), (true, false), (true, true)] {
            let resources = crate::CpuMemoryResources::default()
                .with_cache_retention(crate::CpuCacheRetention::new(admitted));
            let provider = crate::CpuExecutionProvider::with_resources(resources.clone());
            let b = Owned::f16(&[5, 7], &[0.25; 35]);
            let state = provider
                .prepare_static_constants(
                    &node,
                    &[constant_a, true],
                    &[
                        None,
                        Some(KernelConstantInput {
                            dtype: b.dtype,
                            shape: &b.shape,
                            bytes: &b.bytes,
                        }),
                    ],
                )
                .unwrap();
            assert_eq!(state.is_some(), admitted);
            if let Some(state) = state {
                let prepack = state.downcast_ref::<SharedMatMulWeights>().unwrap();
                assert!(prepack.weights.half_transposes.test_first().is_some());
                let weights = prepack.weights.clone();
                let mut runtime = MatMulKernel {
                    prepack: MatMulPrepack::new(resources),
                    flops: None,
                };
                runtime.set_constant_inputs(&[constant_a, true]);
                assert!(runtime.adopt_shareable_constant_state(state).unwrap());
                assert!(Arc::ptr_eq(&runtime.prepack.weights, &weights));
                let a = Owned::f16(&[1, 5], &[0.5; 5]);
                let mut out = Owned::zeros_f32(&[1, 7]);
                runtime
                    .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                    .unwrap();
                assert_eq!(out.to_f32(), vec![0.625; 7]);
            }
        }
    }

    #[test]
    fn provider_cache_retention_survives_opposite_load_orders_and_interleaved_runs() {
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::NodeId;
        let (m, k, n) = (4usize, 40usize, 24usize);
        let a = Owned::f32(
            &[m, k],
            &(0..m * k)
                .map(|i| (i as f32 * 0.011).cos())
                .collect::<Vec<_>>(),
        );
        let b = Owned::f16(
            &[k, n],
            &(0..k * n)
                .map(|i| (i as f32 * 0.019).sin())
                .collect::<Vec<_>>(),
        );
        let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
        for first_admitted in [true, false] {
            let make = |admitted| {
                crate::CpuExecutionProvider::with_resources(
                    crate::CpuMemoryResources::default()
                        .with_cache_retention(crate::CpuCacheRetention::new(admitted)),
                )
            };
            let first = make(first_admitted);
            let second = make(!first_admitted);
            let compile = |provider: &crate::CpuExecutionProvider| {
                let mut kernel = provider
                    .get_kernel(&node, &[vec![m, k], vec![k, n]], 17)
                    .unwrap();
                kernel.set_constant_inputs(&[false, true]);
                kernel
            };
            let first_kernel = compile(&first);
            let second_kernel = compile(&second);
            let mut reference = None;
            for (kernel, admitted) in [
                (&first_kernel, first_admitted),
                (&second_kernel, !first_admitted),
                (&first_kernel, first_admitted),
                (&second_kernel, !first_admitted),
            ] {
                let mut out = Owned::zeros_f32(&[m, n]);
                kernel
                    .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                    .unwrap();
                let concrete = kernel
                    .as_ref()
                    .as_any()
                    .downcast_ref::<MatMulKernel>()
                    .unwrap();
                assert_eq!(
                    concrete.prepack.dense_live_bytes(),
                    if admitted { (k * n * 4) as u64 } else { 0 }
                );
                let values = out.to_f32();
                if let Some(expected) = &reference {
                    assert_eq!(&values, expected);
                } else {
                    reference = Some(values);
                }
            }
        }
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

    // Fixtures construct a real logical input; production callers carry the
    // original view even when its dense conversion later retires.
    fn cached_f32(
        prepack: &MatMulPrepack,
        b: &[f32],
        k: usize,
        n: usize,
    ) -> Option<Arc<GrantedWeight>> {
        let matches = k.checked_mul(n) == Some(b.len());
        let shape = if matches { vec![k, n] } else { vec![b.len()] };
        let strides = if matches { vec![n as i64, 1] } else { vec![1] };
        let view = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(b.as_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        prepack.transposed_b(&view, k, n).unwrap()
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
            ((4, 6), (6, 4)),                       // same extent, different geometry
        ] {
            let first = constant_b_prepack();
            let bt1 = cached_f32(&first, &storage[..k1 * n1], k1, n1)
                .expect("constant B of matching length must be transposed");
            assert_eq!(bt1.as_slice().len(), n1 * k1);
            assert_eq!(
                bt1.as_slice(),
                reference_transpose(&storage[..k1 * n1], k1, n1)
            );

            let second = constant_b_prepack();
            let bt2 = cached_f32(&second, &storage[..k2 * n2], k2, n2)
                .expect("a second kernel at the same address must still be served");
            assert_eq!(
                bt2.as_slice().len(),
                n2 * k2,
                "[{k1},{n1}] then [{k2},{n2}] at one address: served a {}-element \
                 transpose where {} are indexed — release builds read past the end",
                bt2.as_slice().len(),
                n2 * k2
            );
            assert_eq!(
                bt2.as_slice(),
                reference_transpose(&storage[..k2 * n2], k2, n2),
                "[{k1},{n1}] then [{k2},{n2}] at one address: served another tensor's data"
            );
        }
    }

    /// Shape kernels that explicitly adopt one node owner share one allocation.
    #[test]
    fn transposed_b_is_shared_across_kernels_for_the_same_weight() {
        let mut storage = vec![0.0f32; 8 * 12];
        for (i, v) in storage.iter_mut().enumerate() {
            *v = i as f32 * 0.25;
        }
        let first = constant_b_prepack();
        let mut second = constant_b_prepack();
        second
            .adopt_shareable_constant_state(first.shareable_constant_state().unwrap())
            .unwrap();
        let a = cached_f32(&first, &storage, 8, 12).expect("first");
        let b = cached_f32(&second, &storage, 8, 12).expect("second");
        assert_eq!(
            a.as_slice().as_ptr(),
            b.as_slice().as_ptr(),
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
        assert!(cached_f32(&prepack, &b, 3, 5).is_none(), "too long");
        assert!(cached_f32(&prepack, &b, 2, 5).is_none(), "too short");
        assert!(
            cached_f32(&prepack, &b, usize::MAX, 3).is_none(),
            "k * n overflow must fail closed"
        );
        assert!(
            cached_f32(&prepack, &b, 3, 4).is_some(),
            "matching geometry"
        );

        // Non-constant B is never transposed.
        let activation = MatMulPrepack::default();
        assert!(cached_f32(&activation, &b, 3, 4).is_none());
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
        let first = cached_f32(&prepack, &storage[..12], 3, 4).expect("first");
        assert_eq!(first.as_slice().len(), 12);

        // Same prepack, same base address, different shape: the memo must not
        // be served. Falling back to `None` (the untransposed kernel) is
        // correct; serving a wrong-length slice is not.
        match cached_f32(&prepack, &storage, 4, 6) {
            None => {}
            Some(bt) => {
                assert_eq!(
                    bt.as_slice().len(),
                    24,
                    "memo served a wrong-length transpose"
                );
                assert_eq!(bt.as_slice(), reference_transpose(&storage, 4, 6));
            }
        }

        // The original geometry keeps hitting the memo.
        let again = cached_f32(&prepack, &storage[..12], 3, 4).expect("again");
        assert_eq!(again.as_slice().as_ptr(), first.as_slice().as_ptr());
    }

    /// Zero-element weights are handled without panicking and without growing
    /// the cache.
    #[test]
    fn transposed_b_handles_zero_sized_weights() {
        let prepack = constant_b_prepack();
        for (k, n) in [(0, 0), (0, 8), (8, 0)] {
            assert!(cached_f32(&prepack, &[], k, n).is_none());
            assert_eq!(prepack.f32_transpose_stats(), (0, 0));
        }
    }

    /// The f16 memo enforces the same identity, and rejects a `k`/`n` pair that
    /// does not match the view — which would otherwise build a slice longer
    /// than the weight and read out of bounds.
    #[test]
    fn transposed_b_half_validates_the_view_geometry() {
        let bits: Vec<u16> = (0..24u16).map(|i| 0x3C00 + i).collect();
        let owned = Owned::f16_bits(&[4, 6], &bits);
        let prepack = constant_b_prepack();

        assert!(
            prepack
                .transposed_b_half(&owned.view(), 4, 7, HalfFormat::F16)
                .unwrap()
                .is_none(),
            "k * n larger than the view must fail closed"
        );
        assert!(
            prepack
                .transposed_b_half(&owned.view(), 2, 6, HalfFormat::F16)
                .unwrap()
                .is_none(),
            "k * n smaller than the view must fail closed"
        );

        let bt_owner = prepack
            .transposed_b_half(&owned.view(), 4, 6, HalfFormat::F16)
            .unwrap()
            .expect("matching geometry");
        let bt = bt_owner.as_slice();
        assert_eq!(bt.len(), 24);
        for j in 0..6 {
            for i in 0..4 {
                assert_eq!(bt[j * 4 + i], bits[i * 6 + j], "B_T[{j},{i}]");
            }
        }

        // Same buffer address, different shape: never a wrong-length slice.
        let reshaped = owned.with_view(&[3, 8], &[8, 1]);
        match prepack
            .transposed_b_half(&reshaped.view(), 3, 8, HalfFormat::F16)
            .unwrap()
        {
            None => {}
            Some(other) => assert_eq!(other.as_slice().len(), 24),
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

        let f32_bt = cached_f32(&prepack, &f32_data, 2, 3).expect("f32");
        assert_eq!(f32_bt.as_slice(), reference_transpose(&f32_data, 2, 3));

        let f16_prepack = constant_b_prepack();
        let f16_bt = f16_prepack
            .transposed_b_half(&owned.view(), 2, 3, HalfFormat::F16)
            .unwrap()
            .expect("f16");
        assert_eq!(
            f16_bt.as_slice(),
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
                let a_dense = to_dense_f32_widen("MatMul", &av).unwrap();
                let b_dense = to_dense_f32_widen("MatMul", &bv).unwrap();
                #[cfg(feature = "mlas")]
                let b_dense = b_dense.into();
                let out = matmul_dense_impl_with_geom(
                    a_dense,
                    b_dense,
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
    fn unresolved_nt_keeps_the_portable_nn_backend_available() {
        assert!(NtGemm::resolve(CpuBackend::Generic).is_none());
        let mut result = [0.; 4];
        gemm_with_backend(
            CpuBackend::Generic,
            &[1., 2., 3., 4.],
            &[5., 6., 7., 8.],
            &mut result,
            2,
            2,
            2,
        )
        .unwrap();
        assert_eq!(result, [19., 22., 43., 50.]);
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
        // The actual default (including Apple Accelerate) must agree with
        // the portable reference. Linking MLAS does not select it on every OS;
        // explicit MLAS arithmetic is exercised by the backend A/B suite.
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 2048, 2048), // decode GEMV (m=1), the dense hot path
            (1, 2304, 9216), // GeGLU up/gate projection tile
            (1, 9216, 2304), // GeGLU down projection tile
            (5, 128, 256),
            (32, 512, 512),
        ];
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
            assert!(kernel.prepack.weights.packed_b.test_first().is_some());
        }
    }

    /// Run real-kernel controls alone: platform dispatch counters are shared
    /// by the existing libtest process and negative guards observe their deltas.
    fn retention_control_in_child(name: &str, backend: Option<&str>) -> bool {
        const CHILD: &str = "NXRT_TEST_MATMUL_RETENTION_CHILD";
        if std::env::var(CHILD).as_deref() == Ok(name) {
            return false;
        }
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", name, "--nocapture"])
            .env(CHILD, name)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(backend) = backend {
            command.env("NXRT_CPU_GEMM_BACKEND", backend);
        }
        let mut child = command.spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        let timed_out = loop {
            if child.try_wait().unwrap().is_some() {
                break false;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                break true;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !timed_out
                && output.status.success()
                && stdout.contains(&format!("test {name} ... ok")),
            "named retention child must run and pass; timed_out={timed_out}; stdout={stdout}; stderr={stderr}"
        );
        true
    }

    #[test]
    fn strided_half_preparation_honors_retention() {
        if retention_control_in_child(
            "kernels::matmul::tests::strided_half_preparation_honors_retention",
            None,
        ) {
            return;
        }
        let (m, k, n) = (3usize, 17usize, 9usize);
        let weights: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 / 32.).collect();
        let backing: Vec<f32> = weights.iter().flat_map(|&value| [value, -99.]).collect();
        let storage = Owned::f16(&[k, 2 * n], &backing);
        let shape = [k, n];
        let strides = [(2 * n) as i64, 2];
        let mut b = storage.view();
        b.shape = &shape;
        b.strides = &strides;
        let expected_bits: Vec<u16> = weights
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        for admitted in [false, true] {
            use onnx_runtime_memory_api::Tier;
            use onnx_runtime_memory_governor::{
                HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor,
            };
            let bytes = (k * n * 2) as u64;
            let books = LedgerGovernor::new(LeaseLedger::new(0, bytes, 0));
            let resources = CpuMemoryResources::new(
                crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
                Default::default(),
            )
            .unwrap()
            .with_cache_retention(crate::CpuCacheRetention::new(admitted));
            let mut kernel = MatMulKernel {
                prepack: MatMulPrepack::new(resources),
                flops: None,
            };
            kernel.set_constant_inputs(&[false, true]);
            for _ in 0..3 {
                let prepared = kernel.prepack.contiguous_b_f16(&b).unwrap().unwrap();
                assert_eq!(prepared.as_slice(), &expected_bits);
                assert_eq!(books.used(Tier::Host), bytes);
                assert_eq!(
                    kernel.prepack.weights.contiguous_b_f16.stats().0 != 0,
                    admitted,
                    "noncontiguous f16 copying must follow retention permission"
                );
            }
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                let values: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 / 16.).collect();
                let a = Owned::f16(&[m, k], &values);
                let mut output = Owned::zeros_f32(&[m, n]);
                kernel
                    .execute_with_backend(
                        &[a.view(), b],
                        &mut [output.view_mut()],
                        CpuBackend::Accelerate,
                    )
                    .unwrap();
                let expected: Vec<f32> = (0..m * n)
                    .map(|i| {
                        (0..k)
                            .map(|j| values[i / n * k + j] * weights[j * n + i % n])
                            .sum()
                    })
                    .collect();
                assert_eq!(
                    output.to_f32(),
                    expected,
                    "declining reuse must not turn the rescue output into zeros"
                );
                assert_eq!(
                    kernel.prepack.weights.contiguous_b_f16.stats().0 != 0,
                    admitted
                );
                assert_eq!(kernel.prepack.dense_live_bytes(), 0);
                assert_eq!(books.used(Tier::Host), if admitted { bytes } else { 0 });
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            let _ = m;
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn sgemm_half_panel_requires_the_complete_conversion_peak_before_building() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let weight = Owned::f16(&[5, 3], &[0.5; 15]);
        let payload = mlas_sys::PackedBPlan::new(3, 5).unwrap().owned_heap_bytes() as u64;
        let peak = payload + 15 * 4;
        for capacity in [payload, peak - 1, peak] {
            let books = LedgerGovernor::new(LeaseLedger::new(0, capacity, 0));
            let resources = CpuMemoryResources::new(
                crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
                Default::default(),
            )
            .unwrap();
            let mut prepack = MatMulPrepack::new(resources);
            prepack.set_constant_inputs(&[false, true]);
            let reader = prepack.packed_b_from_half(&weight.view(), 5, 3).unwrap();
            assert_eq!(
                reader.is_some(),
                capacity == peak,
                "panel-only funding is insufficient while the f32 conversion is live"
            );
            assert_eq!(
                prepack.weights.packed_b_from_half.activity().1,
                usize::from(capacity == peak)
            );
            assert_eq!(
                books.used(Tier::Host),
                if capacity == peak { payload } else { 0 }
            );
            assert_eq!(
                prepack.dense_live_bytes(),
                0,
                "packing conversion must not become a permanent dense copy"
            );
            drop(prepack);
            drop(reader);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn sgemm_panel_grants_follow_readers_idle_pressure_and_retry() {
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let (k, n) = (5, 3);
        let bits = Owned::f16(&[k, n], &[0.5; 15]);
        let floats = [0.5; 15];
        let packed_bytes = mlas_sys::PackedBPlan::new(n, k).unwrap().owned_heap_bytes() as u64;
        for half in [false, true] {
            let capacity = packed_bytes + if half { 15 * 4 } else { 0 };
            let books = LedgerGovernor::new(LeaseLedger::new(0, capacity, 0));
            let resources = CpuMemoryResources::new(
                crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
                Default::default(),
            )
            .unwrap();
            let mut prepack = MatMulPrepack::new(resources);
            prepack.set_constant_inputs(&[false, true]);
            let build = |prepack: &MatMulPrepack| {
                if half {
                    prepack.packed_b_from_half(&bits.view(), k, n)
                } else {
                    prepack.packed_b(&Cow::Borrowed(floats.as_slice()).into(), k, n)
                }
            };
            let reader = build(&prepack).unwrap().unwrap();
            assert_eq!(reader.payload().owned_heap_bytes() as u64, packed_bytes);
            assert_eq!(
                books.used(Tier::Host),
                packed_bytes,
                "packing scratch must refund after construction"
            );
            let warm = build(&prepack).unwrap().unwrap();
            assert!(Arc::ptr_eq(&reader, &warm));
            drop(warm);
            assert!(
                books
                    .reserve(Tier::Host, capacity, MemoryRole::KvCache, HolderId::new(7))
                    .is_err(),
                "pressure cannot revoke an executing panel reader"
            );
            let mut output = [0.; 6];
            gemm_packed(&[0.25; 10], reader.payload(), &mut output, 2, k, n).unwrap();
            assert_eq!(output, [0.625; 6]);
            drop(reader);
            let kv = books
                .reserve(Tier::Host, capacity, MemoryRole::KvCache, HolderId::new(7))
                .unwrap();
            let cache = if half {
                &prepack.weights.packed_b_from_half
            } else {
                &prepack.weights.packed_b
            };
            assert_eq!(
                cache.stats(),
                (0, 0),
                "KV pressure must reclaim an idle panel"
            );
            let before = cache.activity().1;
            assert!(build(&prepack).unwrap().is_none());
            assert_eq!(
                cache.activity().1,
                before,
                "refusal must precede any native pack or half conversion"
            );
            drop(kv);
            let reader = build(&prepack).unwrap().unwrap();
            drop(prepack);
            assert_eq!(books.used(Tier::Host), packed_bytes);
            gemm_packed(&[0.25; 10], reader.payload(), &mut output, 2, k, n).unwrap();
            assert_eq!(output, [0.625; 6]);
            drop(reader);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn sgemm_panel_concurrent_builders_share_one_funded_payload() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let bits = Owned::f16(&[5, 3], &[0.5; 15]);
        let floats = [0.5; 15];
        let packed_bytes = mlas_sys::PackedBPlan::new(3, 5).unwrap().owned_heap_bytes() as u64;
        for half in [false, true] {
            let capacity = packed_bytes + if half { 60 } else { 0 };
            let books = LedgerGovernor::new(LeaseLedger::new(0, capacity, 0));
            let resources = CpuMemoryResources::new(
                crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
                Default::default(),
            )
            .unwrap();
            let mut prepack = MatMulPrepack::new(resources);
            prepack.set_constant_inputs(&[false, true]);
            let barrier = std::sync::Barrier::new(8);
            let readers = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..8)
                    .map(|_| {
                        scope.spawn(|| {
                            barrier.wait();
                            if half {
                                prepack.packed_b_from_half(&bits.view(), 5, 3)
                            } else {
                                prepack.packed_b(&Cow::Borrowed(floats.as_slice()).into(), 5, 3)
                            }
                            .unwrap()
                            .unwrap()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|thread| thread.join().unwrap())
                    .collect::<Vec<_>>()
            });
            assert!(
                readers
                    .iter()
                    .all(|reader| Arc::ptr_eq(reader, &readers[0]))
            );
            let cache = if half {
                &prepack.weights.packed_b_from_half
            } else {
                &prepack.weights.packed_b
            };
            assert_eq!(cache.activity().1, 1);
            assert_eq!(books.used(Tier::Host), packed_bytes);
            drop(prepack);
            assert_eq!(books.used(Tier::Host), packed_bytes);
            drop(readers);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn provider_sgemm_panel_admission_preserves_packed_and_unpacked_execution() {
        if retention_control_in_child(
            "kernels::matmul::tests::provider_sgemm_panel_admission_preserves_packed_and_unpacked_execution",
            Some("mlas"),
        ) {
            return;
        }
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ir::NodeId;
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        assert_eq!(CpuBackend::auto_detect(), CpuBackend::Mlas);
        let (m, k, n) = (3, 5, 3);
        let bytes = mlas_sys::PackedBPlan::new(n, k).unwrap().owned_heap_bytes() as u64;
        for dtype in [DataType::Float32, DataType::Float16] {
            let a = if dtype == DataType::Float32 {
                Owned::f32(&[m, k], &[0.25; 15])
            } else {
                Owned::f16(&[m, k], &[0.25; 15])
            };
            let b = if dtype == DataType::Float32 {
                Owned::f32(&[k, n], &[0.5; 15])
            } else {
                Owned::f16(&[k, n], &[0.5; 15])
            };
            let peak = bytes + if dtype == DataType::Float16 { 60 } else { 0 };
            for retained in [true, false] {
                for capacity in [peak, 0] {
                    let books = LedgerGovernor::new(LeaseLedger::new(0, capacity, 0));
                    let resources = CpuMemoryResources::new(
                        crate::DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
                        Default::default(),
                    )
                    .unwrap()
                    .with_cache_retention(crate::CpuCacheRetention::new(retained));
                    let provider = crate::CpuExecutionProvider::with_resources(resources);
                    let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
                    let mut kernel = provider
                        .get_kernel(&node, &[vec![m, k], vec![k, n]], 17)
                        .unwrap();
                    kernel.set_constant_inputs(&[false, true]);
                    let run = || {
                        let mut out = Owned::zeros_f32(&[m, n]);
                        kernel
                            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                            .unwrap();
                        assert_eq!(out.to_f32(), vec![0.625; m * n]);
                    };
                    run();
                    assert_eq!(
                        books.used(Tier::Host),
                        if retained && capacity != 0 { bytes } else { 0 },
                        "{dtype:?}: panel ownership must use the provider's Host ledger"
                    );
                    if capacity != 0 {
                        let kv = books
                            .reserve(Tier::Host, capacity, MemoryRole::KvCache, HolderId::new(7))
                            .unwrap();
                        run();
                        assert_eq!(
                            books.used(Tier::Host),
                            capacity,
                            "declined optional packing must use the existing unpacked route"
                        );
                        drop(kv);
                        run();
                    }
                    drop(kernel);
                    drop(provider);
                    assert_eq!(books.used(Tier::Host), 0);
                }
            }
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_pack_failure_preserves_context_and_can_retry_before_warm_validation() {
        let mut prepack = MatMulPrepack::new(CpuMemoryResources::default());
        prepack.set_constant_inputs(&[false, true]);
        let error = prepack
            .packed_b(&Cow::Borrowed([].as_slice()).into(), 5, 3)
            .err()
            .unwrap();
        assert!(error.to_string().contains("MatMul MLAS constant B packing"));
        assert!(error.to_string().contains("expected 15"));
        assert!(prepack.weights.packed_b.test_first().is_none());
        let weight = [0.25; 15];
        let first = prepack
            .packed_b(&Cow::Borrowed(weight.as_slice()).into(), 5, 3)
            .unwrap()
            .unwrap()
            .payload() as *const _;
        let second = prepack
            .packed_b(&Cow::Borrowed(weight.as_slice()).into(), 5, 3)
            .unwrap()
            .unwrap()
            .payload() as *const _;
        assert_eq!(first, second);
        assert!(
            prepack
                .packed_b(&Cow::Borrowed([].as_slice()).into(), 5, 3)
                .is_err(),
            "warm hits still validate source extent"
        );
        let changed = prepack
            .packed_b(&Cow::Borrowed(weight.as_slice()).into(), 3, 5)
            .unwrap()
            .unwrap();
        assert_eq!(changed.payload().dimensions(), (3, 5));
        assert_ne!(first, changed.payload() as *const _);
    }

    #[cfg(feature = "mlas")]
    fn assert_mlas_pack_retention(dtype: DataType) {
        let (m, k, n) = (3usize, 17usize, 9usize);
        let weights: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 / 32.).collect();
        let b = match dtype {
            DataType::Float32 => Owned::f32(&[k, n], &weights),
            DataType::Float16 => Owned::f16(&[k, n], &weights),
            _ => unreachable!("this fixture exercises the two MLAS packing sources"),
        };
        for admitted in [false, true] {
            let mut kernel = MatMulKernel::new(
                CpuMemoryResources::default()
                    .with_cache_retention(crate::CpuCacheRetention::new(admitted)),
                None,
            );
            kernel.set_constant_inputs(&[false, true]);
            for step in 0..3 {
                let values: Vec<f32> = (0..m * k).map(|i| ((i + step) % 7) as f32 / 16.).collect();
                let a = match dtype {
                    DataType::Float32 => Owned::f32(&[m, k], &values),
                    _ => Owned::f16(&[m, k], &values),
                };
                let mut output = Owned::zeros_f32(&[m, n]);
                kernel
                    .execute_with_backend(
                        &[a.view(), b.view()],
                        &mut [output.view_mut()],
                        CpuBackend::Mlas,
                    )
                    .unwrap();
                // All products and their short sums are exactly representable
                // dyadics. Exact equality here makes no broad NN/packed promise.
                let expected: Vec<f32> = (0..m * n)
                    .map(|i| {
                        (0..k)
                            .map(|j| values[i / n * k + j] * weights[j * n + i % n])
                            .sum()
                    })
                    .collect();
                assert_eq!(
                    output.to_f32(),
                    expected,
                    "dtype={dtype:?}, admitted={admitted}, step={step}"
                );
                assert_eq!(
                    kernel.prepack.weights.packed_b.test_first().is_some(),
                    admitted && dtype == DataType::Float32,
                    "dtype={dtype:?}: f32 packed allocation must follow retention permission"
                );
                assert_eq!(
                    kernel
                        .prepack
                        .weights
                        .packed_b_from_half
                        .test_first()
                        .is_some(),
                    admitted && dtype == DataType::Float16,
                    "dtype={dtype:?}: once-widened packed allocation must follow retention permission"
                );
                if !admitted {
                    assert_eq!(kernel.prepack.dense_live_bytes(), 0);
                }
            }
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_f32_packing_honors_retention() {
        if retention_control_in_child(
            "kernels::matmul::tests::mlas_f32_packing_honors_retention",
            None,
        ) {
            return;
        }
        assert_mlas_pack_retention(DataType::Float32);
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_f16_packing_honors_retention() {
        if retention_control_in_child(
            "kernels::matmul::tests::mlas_f16_packing_honors_retention",
            None,
        ) {
            return;
        }
        assert_mlas_pack_retention(DataType::Float16);
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
        let packed_ptr = kernel
            .prepack
            .weights
            .packed_b
            .test_first()
            .unwrap()
            .payload() as *const mlas_sys::PackedB;
        let dense_ptr = kernel.prepack.dense_payload(1).unwrap().as_slice().as_ptr();

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
            kernel
                .prepack
                .weights
                .packed_b
                .test_first()
                .unwrap()
                .payload() as *const mlas_sys::PackedB,
            packed_ptr
        );
        assert_eq!(
            kernel.prepack.dense_payload(1).unwrap().as_slice().as_ptr(),
            dense_ptr
        );
        assert!(!kernel.prepack.dense_is_filled(0));
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
        assert!(kernel.prepack.weights.packed_b.test_first().is_none());
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
        assert!(
            generic_kernel
                .prepack
                .weights
                .packed_b
                .test_first()
                .is_none()
        );
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
        assert!(
            batched_kernel
                .prepack
                .weights
                .packed_b
                .test_first()
                .is_none()
        );
        for (actual, expected) in batched_out.to_f32().iter().zip(expected.iter().cycle()) {
            assert!((actual - expected).abs() <= 1e-3);
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_float_kernel_report_matches_target() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert_ne!(mlas_sys::selected_float_kernel(), 0);
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        assert_eq!(
            mlas_sys::selected_float_kernel(),
            0,
            "the reference kernel ID is x86 instrumentation; arithmetic is checked independently"
        );
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
        // directly as f16 (via transposed_b_half) and never widens to f32 —
        // so dense[1] is NOT populated. On other platforms, dense[1] IS populated.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        assert!(
            kernel
                .prepack
                .weights
                .half_transposes
                .test_first()
                .is_some()
        );
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        assert!(kernel.prepack.dense_is_filled(1));
        assert!(!kernel.prepack.dense_is_filled(0));

        // Capture the cache pointer *before* the second execute so the
        // comparison below is a real guard: it proves the first call populated
        // the cache and the second reused it (rather than repopulating).
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let ptr_before = kernel
            .prepack
            .weights
            .half_transposes
            .test_first()
            .unwrap()
            .as_slice()
            .as_ptr();
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let ptr_before = kernel.prepack.dense_payload(1).unwrap().as_slice().as_ptr();

        let a2 = Owned::f32(&[1, 2], &[4., 5.]);
        let mut out2 = Owned::zeros_f32(&[1, 2]);
        kernel
            .execute(&[a2.view(), weight.view()], &mut [out2.view_mut()])
            .unwrap();
        assert_eq!(out2.to_f32(), vec![8., 15.]);

        // The pointer must be unchanged — the OnceLock was already populated.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        assert_eq!(
            kernel
                .prepack
                .weights
                .half_transposes
                .test_first()
                .unwrap()
                .as_slice()
                .as_ptr(),
            ptr_before,
            "half transpose cache was reallocated on the second execute"
        );
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        assert_eq!(
            kernel.prepack.dense_payload(1).unwrap().as_slice().as_ptr(),
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
            kernel
                .prepack
                .weights
                .packed_b_from_half
                .test_first()
                .is_some(),
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
            .weights
            .packed_b_from_half
            .test_first()
            .expect("the first call must populate the pack")
            .payload() as *const _;

        let second = try_packed_half_prefill(&prepack, backend, &a.view(), &b.view(), m, k, n)
            .unwrap()
            .expect("the second call must also take the packed route");
        let packed_after = prepack
            .weights
            .packed_b_from_half
            .test_first()
            .unwrap()
            .payload() as *const _;

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
                kernel
                    .prepack
                    .weights
                    .packed_b_from_half
                    .test_first()
                    .is_none(),
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
    /// cannot happen today. Source, dtype and geometry still bind each panel:
    /// a different source must fund its own preparation, never reuse another
    /// tensor's packed values.
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

        let changed =
            try_packed_half_prefill(&prepack, backend, &a.view(), &b_second.view(), m, k, n)
                .unwrap()
                .expect("a different weight must obtain its own preparation");
        let want_second = naive_matmul(&a_data, &second, m, k, n);
        for (actual, want) in changed.iter().zip(&want_second) {
            assert!(
                (actual - want).abs() <= 1e-3,
                "second weight received stale packed values"
            );
        }
        assert_ne!(changed, served);
        assert_eq!(prepack.weights.packed_b_from_half.activity().1, 2);
        let again = try_packed_half_prefill(&prepack, backend, &a.view(), &b_first.view(), m, k, n)
            .unwrap()
            .unwrap();
        assert_eq!(again, served);
        assert_eq!(
            prepack.weights.packed_b_from_half.activity().1,
            2,
            "the original source must still reuse its own panel"
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
            kernel
                .prepack
                .weights
                .packed_b_from_half
                .test_first()
                .is_none(),
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
                // Built with the non-default `mlas` feature, an x86 host
                // auto-detects `CpuBackend::Mlas`, and `try_packed_half_prefill`
                // claims contiguous f16 prefill before `try_matmul_half` is
                // reached. bf16 has no such interception, and the shipped
                // default artifact carries no MLAS at all -- so this is about
                // which route an opt-in build takes, not about the numbers,
                // which are still checked below either way.
                #[cfg(feature = "mlas")]
                let mlas_claims_this = dtype == DataType::Float16
                    && CpuBackend::auto_detect() == crate::backend::CpuBackend::Mlas;
                #[cfg(not(feature = "mlas"))]
                let mlas_claims_this = false;
                if !mlas_claims_this {
                    // Which of the two fast routes runs is a property of the
                    // host, not of the kernel, so derive the expectation from
                    // the very predicates `half_gemm_tile` dispatches on rather
                    // than restating a hardware assumption here. Two of them
                    // bite on real runners:
                    //
                    //   * `x86_bf16::native_available()` -- `half_gemm_tile`
                    //     tries the native AVX-512 BF16 kernel *before* the
                    //     GEBP, so on a host with `avx512bf16` a bf16 tile
                    //     legitimately never reaches the GEBP. That is the same
                    //     precedence `half_decode_prefers_gebp_when` already
                    //     encodes for decode. A bare
                    //     `half_prefill_gebp_calls() == 1` asserts a route such
                    //     a host cannot take, which is why this test was
                    //     intermittently red across GitHub's mixed Linux x64
                    //     pool rather than consistently so.
                    //
                    //   * `has_simd_x86()` -- without AVX2+FMA neither fast
                    //     route is selected and the portable blocked half GEMM
                    //     runs. That case is a legitimate `(0, 0)`, not a
                    //     reason to skip: the numeric comparison below is the
                    //     half of this test that is valuable on *every* host,
                    //     so it must stay on the unconditional path.
                    let format = match dtype {
                        DataType::Float16 => HalfFormat::F16,
                        _ => HalfFormat::Bf16,
                    };
                    let expect_native_bf16 =
                        format == HalfFormat::Bf16 && x86_bf16::native_available();
                    let expect_gebp = !expect_native_bf16
                        && half_prefill_gebp_selected(format, m, k, n)
                        && half_prefill_gebp_enabled();
                    let want = (u64::from(expect_gebp), u64::from(expect_native_bf16));
                    assert_eq!(
                        (half_prefill_gebp_calls(), half_native_bf16_calls()),
                        want,
                        "{dtype:?} m={m}: prefill took the wrong route \
                         (avx512bf16={}, simd_x86={})",
                        x86_bf16::native_available(),
                        crate::backend::has_simd_x86()
                    );
                }

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

    /// No `m == 1` half decode is diverted away from the GEMV on weight.
    ///
    /// This replaces a gate. `half_decode_prefers_gebp` used to hand every
    /// decode with `k * n >= HALF_PREFILL_GEBP_MIN_WEIGHT` to the fused
    /// widen-pack GEBP; re-measured through this kernel
    /// (`benches/half_decode_gemv_ab.rs`, arms selected by
    /// `ONNX_GENAI_CPU_MM_HALF_GEMV/GEBP`, `f32` control divided out) that
    /// divert is a 1.3x-5.0x loss at 8 threads and still a loss at the model
    /// shapes at 32 -- `k = 4096` 1.1x-2.4x, a 136M `lm_head` 1.16x. The
    /// mechanism is not tuning: the GEBP earns its packing by reusing a `B`
    /// panel across rows of `A`, and at `m == 1` `m_panels` is 1, so every
    /// panel is consumed once and none of the packing is repaid.
    /// `docs/benchmarks/2026-08-21-half-decode-gebp-retired.md` has the
    /// matrix.
    ///
    /// Asserted by *execution* rather than by asking a predicate, because the
    /// predicate is what was deleted. The counter is incremented by the GEMV
    /// arm alone, so a re-introduced divert fails here.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn no_half_decode_is_diverted_off_the_gemv_on_weight() {
        if !crate::backend::require_simd_x86("no_half_decode_is_diverted_off_the_gemv_on_weight") {
            return;
        }
        // Straddles the retired threshold (0.79M / 1.05M) and then follows the
        // decode shapes of a 7B-class graph out to 45M, every one of which the
        // divert used to claim.
        let shapes = [
            (512usize, 512usize),
            (1024, 768),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
            (4096, 11008),
        ];
        for (k, n) in shapes {
            let a_data: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect();
            let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 19) as f32 - 9.0) / 16.0).collect();
            for format in [HalfFormat::F16, HalfFormat::Bf16] {
                if !half_gemv::simd_available(format) {
                    continue;
                }
                let (a, b) = match format {
                    HalfFormat::F16 => (Owned::f16(&[1, k], &a_data), Owned::f16(&[k, n], &b_data)),
                    HalfFormat::Bf16 => {
                        (Owned::bf16(&[1, k], &a_data), Owned::bf16(&[k, n], &b_data))
                    }
                };
                let mut out = Owned::zeros_f32(&[1, n]);
                let mut kernel = MatMulKernel::default();
                kernel.set_constant_inputs(&[false, true]);
                reset_half_decode_gemv_calls();
                reset_half_prefill_gebp_calls();
                kernel
                    .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                    .unwrap();
                assert_eq!(
                    half_decode_gemv_calls(),
                    1,
                    "{format:?} {k}x{n}: an m == 1 decode must be served by the GEMV"
                );
                assert_eq!(
                    half_prefill_gebp_calls(),
                    0,
                    "{format:?} {k}x{n}: an m == 1 decode must not reach the prefill GEBP"
                );
            }
        }
    }

    /// A *batched* `m == 1` half MatMul still reaches the GEBP.
    ///
    /// The decode GEMV declines a batch outright, and the only other
    /// destination is the row-blocked half GEMM -- 16x-21x slower at these
    /// weights, with identical numbers. So retiring the decode handover must
    /// not be mistaken for "the GEBP never serves `m == 1`": this is the case
    /// that keeps `half_prefill_gebp_selected` taking `m` without a row term.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn a_batched_single_row_half_matmul_still_reaches_the_gebp() {
        if !crate::backend::has_simd_x86() || x86_bf16::native_available() {
            return;
        }
        let (batch, k, n) = (2usize, 1024usize, 1024usize);
        assert!(
            k * n >= HALF_PREFILL_GEBP_MIN_WEIGHT,
            "the weight must clear the GEBP gate for this to be a test"
        );
        let a_data: Vec<f32> = (0..batch * k)
            .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
            .collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 19) as f32 - 9.0) / 16.0).collect();
        let a = Owned::f16(&[batch, 1, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[batch, 1, n]);
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        reset_half_decode_gemv_calls();
        reset_half_prefill_gebp_calls();
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            half_decode_gemv_calls(),
            0,
            "a batched decode must not reach the non-batched GEMV"
        );
        assert!(
            half_prefill_gebp_calls() > 0,
            "a batched m == 1 decode must still be served by the GEBP"
        );
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
            // The GEMV must never widen B to f32 -- that is 4x the stored
            // weight and was the cost `try_matmul_half` existed to avoid.
            assert!(
                !kernel.prepack.dense_is_filled(1),
                "{format:?} decode must not widen B to f32"
            );
            // It *may* hold one transpose at the stored width, because the
            // `[N, K]` kernel is 1.6x-2.9x faster and 2.7x-9.3x more accurate
            // than reading `[K, N]` in place. What must never happen is that
            // copy being unbudgeted: the memory plan sizes this buffer from
            // `weight_transpose_cache_predicted_bytes`, so the prediction has
            // to cover whatever the kernel actually retained. This assertion
            // replaces an older "must not transpose at all" -- the concern it
            // encoded (a silent, weight-scaled resident buffer) is preserved,
            // and now checked against the number the plan budgets.
            if let Some(bt) = kernel.prepack.weights.half_transposes.test_first() {
                assert_eq!(
                    bt.as_slice().len(),
                    k * n,
                    "{format:?} a retained transpose must be exactly the weight"
                );
                let (node, graph) = half_matmul_node(format, k, n);
                let predicted = node_weight_transpose_cache_bytes(&node, &graph);
                assert!(
                    predicted >= (bt.as_slice().len() * 2) as u64,
                    "{format:?} retained {} bytes but the plan predicted {predicted}",
                    bt.as_slice().len() * 2
                );
            }

            let want = naive_matmul(&a_data, &b_data, 1, k, n);
            for (index, (g, w)) in out.to_f32().iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 2e-2 * (1.0 + w.abs()),
                    "{format:?} column {index}: {g} != {w}"
                );
            }
        }
    }

    /// `f16` decode at the *retired* weight gate keeps the GEMV, and its
    /// numbers are checked directly rather than by proxy.
    ///
    /// 1024x1024 is the smallest weight the prefill GEBP accepts, and this
    /// assertion has now been inverted twice: first to the GEBP on a 32-thread
    /// interleaved measurement, then back to the GEMV when the same production
    /// harness was run at 8 threads and across `k`
    /// (`docs/benchmarks/2026-08-21-half-decode-gebp-retired.md`). The second
    /// inversion is the durable one -- it has a mechanism behind it rather
    /// than a ratio: the GEBP's packing is repaid by reuse across rows of `A`,
    /// and `m == 1` has one row.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn f16_decode_at_the_retired_weight_gate_keeps_the_gemv() {
        if !crate::backend::require_simd_x86("f16_decode_at_the_retired_weight_gate_keeps_the_gemv")
        {
            return;
        }
        let (k, n) = (1024usize, 1024usize);
        assert!(
            k * n >= HALF_PREFILL_GEBP_MIN_WEIGHT,
            "the shape must be at or above the gate to be this test"
        );
        let a_data: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 19) as f32 - 9.0) / 16.0).collect();

        reset_half_prefill_gebp_calls();
        reset_half_decode_gemv_calls();
        let a = Owned::f16(&[1, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[1, n]);
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            (half_decode_gemv_calls(), half_prefill_gebp_calls()),
            (1, 0),
            "f16 M=1 decode at the retired weight gate must keep the GEMV"
        );

        // The route moved, so the numbers are checked rather than assumed.
        // Multiples of 1/16 are exact in f16, so the only rounding either
        // route can introduce is its own accumulation order.
        let want = naive_matmul(&a_data, &b_data, 1, k, n);
        for (index, (g, w)) in out.to_f32().iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 2e-2 * (1.0 + w.abs()),
                "column {index}: {g} != {w}"
            );
        }
    }

    /// A weight freed at one address must not lend its transpose to the next
    /// weight that lands there (#1726).
    ///
    /// Original addresses can be reused after a node's lifetime. Fresh node
    /// ownership must read the new values, including under the same geometry.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn a_recycled_weight_address_must_not_serve_the_previous_weights_transpose() {
        if !crate::backend::require_simd_x86(
            "a_recycled_weight_address_must_not_serve_the_previous_weights_transpose",
        ) {
            return;
        }
        let (k, n) = (1024usize, 1024usize);
        let a_data: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect();
        // Distinct weights. The second is deliberately *not* on the 1/16 grid,
        // so a stale read is arithmetically unmistakable rather than a rounding
        // argument: every honest dot product here is a multiple of 1/256.
        let first_data: Vec<f32> = (0..k * n).map(|i| ((i % 19) as f32 - 9.0) / 16.0).collect();
        let second_data: Vec<f32> = (0..k * n)
            .map(|i| ((i % 23) as f32 - 11.0) / 32.0 + 0.001_953_125)
            .collect();
        let want_first = naive_matmul(&a_data, &first_data, 1, k, n);
        let want_second = naive_matmul(&a_data, &second_data, 1, k, n);

        let mut weight = Owned::f16(&[k, n], &first_data);
        let decode = |weight: &Owned| -> Vec<f32> {
            let a = Owned::f16(&[1, k], &a_data);
            let mut out = Owned::zeros_f32(&[1, n]);
            let mut kernel = MatMulKernel::default();
            kernel.set_constant_inputs(&[false, true]);
            kernel
                .execute(&[a.view(), weight.view()], &mut [out.view_mut()])
                .unwrap();
            out.to_f32()
        };

        for (index, (got, want)) in decode(&weight).iter().zip(&want_first).enumerate() {
            assert!(
                (got - want).abs() <= 2e-2 * (1.0 + want.abs()),
                "first weight column {index}: {got} != {want}"
            );
        }

        let address = weight.bytes.as_ptr();
        for (bytes, value) in weight.bytes.chunks_exact_mut(2).zip(&second_data) {
            bytes.copy_from_slice(&half::f16::from_f32(*value).to_le_bytes());
        }
        assert_eq!(weight.bytes.as_ptr(), address);

        for (index, (got, want)) in decode(&weight).iter().zip(&want_second).enumerate() {
            assert!(
                (got - want).abs() <= 2e-2 * (1.0 + want.abs()),
                "second weight column {index}: {got} != {want} -- a transpose cached for \
                     the previous weight at this address was served to this one"
            );
        }
    }

    /// A transient shape consumer cannot retire preparation retained by its
    /// node owner or invalidate an independently executing reader.
    #[test]
    fn a_prewarmed_transpose_outlives_a_kernel_that_only_shared_it() {
        let owner = constant_b_prepack();
        let mut peer = constant_b_prepack();
        peer.adopt_shareable_constant_state(owner.shareable_constant_state().unwrap())
            .unwrap();
        let data: Vec<f32> = (0..48 * 64).map(|i| (i % 13) as f32 - 6.).collect();
        let first = cached_f32(&owner, &data, 48, 64).unwrap();
        let other = cached_f32(&peer, &data, 48, 64).unwrap();
        assert!(Arc::ptr_eq(&first, &other));
        drop(other);
        drop(peer);
        let warm = cached_f32(&owner, &data, 48, 64).unwrap();
        assert!(Arc::ptr_eq(&first, &warm));
        assert_eq!(owner.f32_transpose_stats(), (1, data.len() * 4));
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
            kernel
                .prepack
                .weights
                .half_transposes
                .test_first()
                .is_none(),
            "an activation B must never be memoised as a weight transpose"
        );
        assert!(
            !kernel.prepack.dense_is_filled(1),
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

    /// A single-node `MatMul` graph with a constant 2-D `B` of `format`'s
    /// dtype, so a test can ask the predictor what the plan would budget for
    /// the very shape it just ran.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn half_matmul_node(format: HalfFormat, k: usize, n: usize) -> (Node, Graph) {
        use onnx_runtime_ir::{NodeId, TensorData, WeightRef, static_shape};
        let dtype = match format {
            HalfFormat::F16 => DataType::Float16,
            HalfFormat::Bf16 => DataType::BFloat16,
        };
        let mut graph = Graph::new();
        let a = graph.create_named_value("A", dtype, static_shape([1, k]));
        let b = graph.create_named_value("B", dtype, static_shape([k, n]));
        let y = graph.create_named_value("Y", DataType::Float32, static_shape([1, n]));
        graph.add_input(a);
        graph.add_input(b);
        let node = Node::new(NodeId(0), "MatMul", vec![Some(a), Some(b)], vec![y]);
        graph.insert_node(node.clone());
        graph.add_output(y);
        graph.set_initializer(
            b,
            WeightRef::Inline(TensorData::from_raw(
                dtype,
                vec![k, n],
                vec![0u8; k * n * 2],
            )),
        );
        (node, graph)
    }

    /// The GEMV must never *widen* the weight, and any transpose it does keep
    /// must be one the memory plan predicted.
    ///
    /// This test used to assert that no transpose was retained at all. That
    /// policy was measured and reversed: the `[N, K]` kernel is 1.6x-2.9x
    /// faster and 2.7x-9.3x more accurate than reading `[K, N]` in place, so
    /// the `2 * K * N` copy now buys something. The property worth keeping is
    /// the one the old assertion was really protecting -- that a resident,
    /// weight-scaled buffer can never appear without the plan knowing -- so
    /// that is what it checks now, against the predictor itself.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn f16_decode_gemv_keeps_only_a_budgeted_transpose() {
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
                !kernel.prepack.dense_is_filled(1),
                "the GEMV must not widen B to f32"
            );
            let retained = kernel.prepack.weights.half_transposes.stats().1 as u64;
            let (node, graph) = half_matmul_node(HalfFormat::F16, k, n);
            let predicted = node_weight_transpose_cache_bytes(&node, &graph);
            assert!(
                predicted >= retained,
                "retained {retained} transpose bytes the plan did not predict ({predicted})"
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

    /// Independent transcription of the previous per-column K-panel sum. The
    /// panel addition boundary is observable in f32 and must not be reassociated.
    fn generic_panel_order_reference(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for column in 0..n {
                for start in (0..k).step_by(KC) {
                    let mut panel = 0.0f32;
                    for inner in start..(start + KC).min(k) {
                        panel += a[row * k + inner] * b[inner * n + column];
                    }
                    out[row * n + column] += panel;
                }
            }
        }
        out
    }

    #[test]
    fn generic_borrowed_columns_preserve_panel_order_and_output_bounds() {
        for threads in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for m in [1, 3, 7] {
                for k in [0, 1, KC - 1, KC, KC + 1, 2 * KC + 3] {
                    for n in [1, 3, NR + 1, 67] {
                        let a: Vec<f32> = (0..m * k)
                            .map(|i| ((i % 37) as f32 - 18.0) * 0.0037)
                            .collect();
                        let b: Vec<f32> = (0..k * n)
                            .map(|i| ((i % 29) as f32 - 14.0) * 0.0061)
                            .collect();
                        let expected = generic_panel_order_reference(&a, &b, m, k, n);
                        let mut storage = vec![f32::from_bits(0x7fc0_1234); m * n + 2];
                        pool.install(|| {
                            gemm_with_backend(
                                CpuBackend::Generic,
                                &a,
                                &b,
                                &mut storage[1..m * n + 1],
                                m,
                                k,
                                n,
                            )
                        })
                        .unwrap();
                        assert_eq!(storage[0].to_bits(), 0x7fc0_1234);
                        assert_eq!(storage[m * n + 1].to_bits(), 0x7fc0_1234);
                        for (index, (&got, &want)) in
                            storage[1..m * n + 1].iter().zip(&expected).enumerate()
                        {
                            assert_eq!(
                                got.to_bits(),
                                want.to_bits(),
                                "threads={threads} m={m} k={k} n={n} index={index}"
                            );
                        }
                    }
                }
            }
        }
        for (m, n) in [(0, 9), (3, 0)] {
            gemm_generic(&[], &[], &mut [], m, 7, n);
        }
    }

    #[test]
    fn generic_borrowed_columns_reach_the_native_dispatch_and_special_values() {
        use std::sync::atomic::Ordering;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let before = GENERIC_BORROWED_COLS_TEST_HITS.load(Ordering::Relaxed);
        let (m, k, n) = (3, KC + 1, 17);
        let a: Vec<f32> = (0..m * k).map(|i| [1.0, -1.0, 0.0][i / k]).collect();
        let patterns = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let b: Vec<f32> = (0..k * n)
            .map(|i| patterns[(i % n) % patterns.len()])
            .collect();
        let expected = generic_panel_order_reference(&a, &b, m, k, n);
        assert!(expected.iter().any(|x| x.is_infinite()));
        assert!(expected.iter().any(|x| x.is_nan()));
        assert!(expected.iter().any(|x| x.is_finite() && *x != 0.0));
        let mut actual = vec![42.0; m * n];
        pool.install(|| gemm_with_backend(CpuBackend::Generic, &a, &b, &mut actual, m, k, n))
            .unwrap();
        assert!(GENERIC_BORROWED_COLS_TEST_HITS.load(Ordering::Relaxed) > before);
        for (&got, &want) in actual.iter().zip(&expected) {
            if want.is_nan() {
                assert!(got.is_nan());
            } else {
                assert_eq!(got.to_bits(), want.to_bits());
            }
        }
    }

    // Frozen pre-borrowing implementation from the base commit, test-only.
    // Its original data copies and arithmetic serve as a same-binary A/B and
    // exact-output reference; no compatibility entry exists in production.
    mod copying_generic_reference {
        use super::super::{KC, MAX_MC, MR, NR};
        use rayon::prelude::*;
        pub(super) fn gemm_generic(
            a: &[f32],
            b: &[f32],
            c: &mut [f32],
            m: usize,
            k: usize,
            n: usize,
        ) {
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
    }

    #[test]
    #[ignore = "manual Generic GEMM copy/borrow diagnostic; run under the host lock"]
    fn bench_generic_borrowed_columns_ab() {
        use std::hint::black_box;
        use std::time::Instant;
        fn cpu_seconds() -> Option<f64> {
            #[cfg(unix)]
            {
                let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
                // SAFETY: the OS initializes the writable rusage on success.
                assert_eq!(
                    unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
                    0,
                    "Generic GEMM CPU-time probe failed: {}",
                    std::io::Error::last_os_error()
                );
                // SAFETY: successful getrusage initialized usage.
                let usage = unsafe { usage.assume_init() };
                Some(
                    (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) as f64
                        + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) as f64 * 1e-6,
                )
            }
            #[cfg(not(unix))]
            {
                None
            }
        }
        println!("threads,m,k,n,iterations,arm,cpu_ms,wall_ms,cpu_min_ms,cpu_max_ms");
        for threads in [1, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for (m, k, n) in [
                (1, 256, 512),
                (1, 1025, 257),
                (1, 2048, 2048),
                (3, 1025, 513),
                (7, 1025, 513),
                (15, 257, 131),
            ] {
                let a: Vec<f32> = (0..m * k)
                    .map(|i| ((i % 37) as f32 - 18.0) * 0.0037)
                    .collect();
                let b: Vec<f32> = (0..k * n)
                    .map(|i| ((i % 29) as f32 - 14.0) * 0.0061)
                    .collect();
                // Short per-call regions make process CPU-time samples mostly
                // scheduler/spin jitter. Batch a deterministic amount of work
                // in each sample, then normalize to one complete GEMM call.
                let iterations = (1usize << 24).div_ceil(m * k * n).clamp(1, 128);
                let expected = generic_panel_order_reference(&a, &b, m, k, n);
                let mut buffers = [vec![0.0; m * n], vec![0.0; m * n], vec![0.0; m * n]];
                let mut cpus: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
                let mut walls = cpus.clone();
                for repetition in 0..21 {
                    for offset in 0..3 {
                        let arm = (repetition + offset) % 3;
                        let kernel = if arm == 0 {
                            copying_generic_reference::gemm_generic
                        } else {
                            gemm_generic
                        };
                        let run = |out: &mut [f32]| {
                            pool.install(|| kernel(black_box(&a), black_box(&b), out, m, k, n))
                        };
                        run(&mut buffers[arm]);
                        assert!(
                            buffers[arm]
                                .iter()
                                .zip(&expected)
                                .all(|(x, y)| x.to_bits() == y.to_bits())
                        );
                        let cpu = cpu_seconds();
                        let wall = Instant::now();
                        for _ in 0..iterations {
                            run(black_box(&mut buffers[arm]));
                        }
                        let elapsed = wall.elapsed().as_secs_f64();
                        let used = cpu_seconds().zip(cpu).map(|(end, start)| end - start);
                        assert!(
                            buffers[arm]
                                .iter()
                                .zip(&expected)
                                .all(|(x, y)| x.to_bits() == y.to_bits())
                        );
                        if let Some(used) = used {
                            cpus[arm].push(used * 1e3 / iterations as f64);
                        }
                        walls[arm].push(elapsed * 1e3 / iterations as f64);
                    }
                }
                for arm in 0..3 {
                    cpus[arm].sort_by(f64::total_cmp);
                    walls[arm].sort_by(f64::total_cmp);
                    let cpu = if cpus[arm].is_empty() {
                        "unavailable".to_string()
                    } else {
                        format!("{:.6}", cpus[arm][10])
                    };
                    let low = cpus[arm]
                        .first()
                        .map_or_else(|| "unavailable".to_string(), |x| format!("{x:.6}"));
                    let high = cpus[arm]
                        .last()
                        .map_or_else(|| "unavailable".to_string(), |x| format!("{x:.6}"));
                    let label = ["copying", "borrowed", "borrowed_control"][arm];
                    println!(
                        "{threads},{m},{k},{n},{iterations},{label},{cpu},{:.6},{low},{high}",
                        walls[arm][10]
                    );
                }
            }
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
        #[cfg(any(target_os = "macos", target_os = "ios", test))]
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
        #[cfg(any(target_os = "macos", target_os = "ios", test))]
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
        #[cfg(any(target_os = "macos", target_os = "ios", test))]
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
        #[cfg(any(target_os = "macos", target_os = "ios", test))]
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
        #[cfg(any(target_os = "macos", target_os = "ios", test))]
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
        #[cfg(any(target_os = "macos", target_os = "ios", test))]
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
        #[cfg(any(target_os = "macos", target_os = "ios", test))]
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
        #[cfg(any(target_os = "macos", target_os = "ios", test))]
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
        #[cfg(any(target_os = "macos", target_os = "ios", test))]
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
        #[cfg(any(target_os = "macos", target_os = "ios", test))]
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

    /// Deterministically exercise the packed `6×16` micro-kernel, its pack step,
    /// its column-panel sweep and both of its remainders. Every shape here has
    /// `rows >= PACK_MIN_ROWS`, which is what selects the packed path over the
    /// `4×2` tile; the single-thread pool then widens the row block to the whole
    /// `m` (the `native-threads=1` production path), so the pack really is built
    /// once and swept by every band.
    ///
    /// The non-vacuity assertion matters more than usual here: the packed kernel
    /// is chosen by a *shape predicate*, so if that predicate is ever retuned
    /// these shapes could silently stop reaching the code under test and the
    /// oracle comparison would keep passing against the unpacked kernel.
    #[cfg(not(feature = "mlas"))]
    #[test]
    fn gemm_bt_packed_kernel_matches_naive_single_threaded() {
        const SHAPES: &[(usize, usize, usize)] = &[
            (72, 64, 16),  // 12 row bands x 1 micro-panel, no remainder at all
            (72, 64, 32),  // two micro-panels
            (73, 64, 17),  // 1 leftover row, 1 leftover column
            (77, 71, 33),  // 5 leftover rows, 1 leftover column, k tail 7
            (72, 8, 16),   // shortest k the predicate admits
            (72, 9, 16),   // k tail of 1
            (78, 128, 96), // 13 bands x 6 micro-panels
            // Large `k` shrinks the panel width, so these cross several column
            // panels - the loop that makes the pack pay for itself.
            (80, 2000, 160), // multiple full column panels
            (85, 2001, 163), // multi-panel, short final panel, both remainders
            (100, 512, 80),  // wider row extent than the MAX_MC row block
        ];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            for &(m, k, n) in SHAPES {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                assert!(
                    !gemm_bt_avx2::available() || gemm_bt_prefers_packed(m, k, n),
                    "shape {m}x{k}x{n} no longer reaches the packed kernel: \
                     this test would be measuring the unpacked path"
                );
                let mut ra = lcg_stream(0x0BAD_F00D ^ (m as u32));
                let mut rb = lcg_stream(0xDEAD_BEEF ^ (n as u32));
                let a: Vec<f32> = (0..m * k).map(|_| ra()).collect();
                let bt: Vec<f32> = (0..n * k).map(|_| rb()).collect();
                let want = naive_bt(&a, &bt, m, k, n);
                let mut got = vec![f32::NAN; m * n];
                gemm_bt(&a, &bt, &mut got, m, k, n).unwrap();
                for (idx, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-3,
                        "packed shape {m}x{k}x{n} index {idx}: got {g}, want {w}"
                    );
                }
            }
        });
    }

    /// The multi-threaded packed driver: whole column panels distributed over a
    /// real pool, each packed exactly once by the lane that owns it.
    ///
    /// Every shape here is asserted to reach that driver rather than the
    /// row-block one, at every thread count, so a gate change that quietly
    /// routes them elsewhere fails the test instead of hollowing it out. The
    /// shapes cover both remainders the driver owns (`n % 16` columns and
    /// `m % 6` rows), one-panel and many-panel column extents, panel counts
    /// both above and below the thread count, and `k` values with a tail.
    ///
    /// Every `k` is at or above the gate's threshold, which is what the
    /// production MoE experts look like; the row-block driver's own test covers
    /// the short-`k` shapes that this gate deliberately sends there.
    #[test]
    #[cfg(not(feature = "mlas"))]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn gemm_bt_packed_mt_matches_naive_across_thread_counts() {
        const SHAPES: &[(usize, usize, usize)] = &[
            (72, 2048, 16),  // 12 bands x 1 micro-panel, one panel, no remainder
            (73, 2048, 17),  // 1 leftover row, 1 leftover column
            (95, 2049, 33),  // 5 leftover rows, 1 leftover column, k tail 1
            (192, 2048, 48), // 32 bands x 3 micro-panels
            (144, 4096, 64), // narrower panels, so several of them
            (205, 2053, 51), // both remainders, k tail 5
            (121, 2048, 85), // short final panel, both remainders
            (96, 2048, 147), // short final panel at two of the widths below
            (96, 2048, 163), // short final panel at two more
            (74, 2048, 48),  // row remainder with no column remainder
        ];
        const THREADS: &[usize] = &[2, 3, 4, 8];
        // The panel width is derived from the column extent and the thread
        // count, so a shape list that happens to divide evenly at every width
        // would leave the short-final-panel path untested. Pin that it does
        // not.
        assert!(
            SHAPES
                .iter()
                .any(|&(m, _, n)| n % 16 == 0 && m % 6 != 0 && m / 6 >= *THREADS.last().unwrap()),
            "no shape has a row remainder without a column remainder: the \
             remainder pass would be reached for the wrong reason"
        );
        assert!(
            !gemm_bt_avx2::available()
                || THREADS.iter().any(|&t| {
                    SHAPES.iter().any(|&(m, k, n)| {
                        let n16 = n - n % 16;
                        m / 6 >= t && n16 % gemm_bt_panel_width(k, n16, t) != 0
                    })
                }),
            "no shape leaves a short final column panel at any tested thread \
             count: the driver's last-panel path would go unexercised"
        );
        for &threads in THREADS {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for &(m, k, n) in SHAPES {
                    if m / 6 < threads {
                        continue; // too few bands to fill this pool by design
                    }
                    assert!(
                        !gemm_bt_avx2::available()
                            || (gemm_bt_prefers_packed(m, k, n)
                                && gemm_bt_packed_mt_fits(m, k, threads)),
                        "shape {m}x{k}x{n} at {threads} threads no longer reaches \
                         the multi-threaded packed driver: this test would be \
                         measuring the row-block path"
                    );
                    let mut ra = lcg_stream(0x51DE_51DE ^ (m as u32));
                    let mut rb = lcg_stream(0xC0FF_EE00 ^ (n as u32));
                    let a: Vec<f32> = (0..m * k).map(|_| ra()).collect();
                    let bt: Vec<f32> = (0..n * k).map(|_| rb()).collect();
                    let want = naive_bt(&a, &bt, m, k, n);
                    let mut got = vec![f32::NAN; m * n];
                    gemm_bt(&a, &bt, &mut got, m, k, n).unwrap();
                    for (idx, (&g, &w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (g - w).abs() <= 1e-3,
                            "packed MT {m}x{k}x{n} at {threads} threads, \
                             index {idx} (row {}, col {}): got {g}, want {w}",
                            idx / n,
                            idx % n
                        );
                    }
                }
            });
        }
    }
}

/// #1702: keeps the weight-cache *accounting* honest, as distinct from the
/// weight-cache *governance* `.github/workflows/weight-cache-guard.yml`
/// already enforces.
///
/// # The escape this closes
///
/// The governance guard greps new `OnceLock<..Cache..>` declarations and makes
/// the author declare a retained-weight policy. It is a good guard for
/// "somebody added a new cache". It says nothing about the *other* half, which
/// is how #1702 escaped: the cache already existed and was already governed,
/// and a new **route** (`FusedMatMulBias` reaching the shared decode GEMV) was
/// pointed at it. `node_weight_transpose_cache_bytes` named `MatMul` explicitly
/// and returned 0 for `FusedMatMulBias` — a comment that was true when written
/// and false the moment the two operators were unified. Nothing failed. The
/// memory plan would simply have under-budgeted every fused projection.
///
/// A per-route hand-written reconciliation test (`gemm.rs`'s
/// `predicted_transpose_bytes_equal_actual_after_gemm_execution`) cannot close
/// this either, because the omission and the missing test are the *same*
/// omission: whoever forgets the predictor forgets its test.
///
/// # The property
///
/// The only structure that cannot be silently omitted is one derived from the
/// **real registry**. [`weight_cache_accounting_classification`] must answer
/// for every `(op_type, domain)` in `build_cpu_registry`, and the test below
/// walks `OpRegistry::keys()` to check that. Registering a kernel without
/// classifying it fails the test with the file and line to edit — the author is
/// *forced* to state whether their op retains weight-scaled bytes, at the
/// moment they add it, rather than being trusted to remember.
#[cfg(test)]
mod weight_cache_accounting {
    use super::*;
    use onnx_runtime_ir::{NodeId, TensorData, WeightRef, static_shape};

    /// Whether an operator can retain weight-scaled transpose bytes, and
    /// hence whether it owes
    /// [`node_weight_transpose_cache_bytes`] a prediction.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum WeightCacheClass {
        /// Reaches a weight-transpose cache for some input configuration. The
        /// reconciliation test below builds a graph for it and asserts
        /// `predicted == actual`.
        CachesTransposedWeight,
        /// Retains no weight-scaled transpose bytes. The `&'static str`
        /// is the reason, and exists to make a wrong classification obvious in
        /// review rather than being a bare `false`.
        NoWeightCache(&'static str),
    }

    /// The classification every registered CPU op must have.
    ///
    /// Deliberately a total function over op name rather than a `HashMap`
    /// literal: the `match` has no fallback arm that could absorb a new op, and
    /// the caller turns `None` into a failure naming the op. Adding a kernel
    /// therefore has exactly one way to proceed — decide, here, what it
    /// retains.
    fn weight_cache_accounting_classification(op_type: &str) -> Option<WeightCacheClass> {
        use WeightCacheClass::{CachesTransposedWeight, NoWeightCache};
        Some(match op_type {
            // --- the MatMul family: all three share `try_half_decode_gemv` ---
            "MatMul" | "FusedMatMulBias" | "Gemm" => CachesTransposedWeight,

            "FusedGemm" => CachesTransposedWeight,

            // Registered only under the `mlas` feature. NCHWc convolution
            // reorders weights into its own blocked layout inside the MLAS
            // path, which is never linked into the default artifact and never
            // touches the weight-transpose caches. Listed explicitly rather
            // than skipped, because `--features mlas` is exactly the
            // configuration where an unclassified op would otherwise slip
            // through: this arm is why the guard walks the registry the test
            // binary actually built.
            "NchwcConv"
            | "NchwcAveragePool"
            | "NchwcGlobalAveragePool"
            | "NchwcMaxPool"
            | "NchwcReorderToBlocked"
            | "NchwcReorderToNchw" => {
                NoWeightCache("mlas-only NCHWc blocked layout, not the transpose cache")
            }

            // Quantized/packed matmuls own their own prepack caches, which are
            // governed and accounted separately (`matmul_nbits`'s borrowed
            // packed cache, #1619/#1628). They never call the f32/f16
            // weight-transpose caches this predictor covers.
            "MatMulNBits"
            | "MatMulInteger"
            | "QLinearMatMul"
            | "DynamicQuantizeMatMul"
            | "MatMulIntegerToFloat"
            | "BlockQuantizedMatMul"
            | "BlockQuantizedMoE"
            | "GatherBlockQuantized" => {
                NoWeightCache("owns a separately governed packed-weight cache")
            }

            // Attention/MoE fusions call the shared GEMM on *activations*
            // (Q·K^T, ·V), whose operands are not graph-constant, so no
            // constant weight ever reaches a transpose cache.
            "FusedAttention"
            | "MultiHeadAttention"
            | "GroupQueryAttention"
            | "Attention"
            | "SparseAttention"
            | "MoE"
            | "QMoE"
            | "PackedMultiHeadAttention"
            | "PackedVarlenAttention"
            | "VarlenAttention"
            | "LinearAttention"
            | "CompressedSparseAttention"
            | "SparseKvGather" => {
                NoWeightCache("GEMM operands are activations, never constant weights")
            }

            // Everything else: elementwise, shape, reduction, normalization,
            // sampling and control ops. None hold weight-scaled process-global
            // state. Grouped rather than enumerated because the grouping *is*
            // the claim, and the test asserts the claim by executing the
            // predictor against them.
            other => NoWeightCache(match other {
                _ if is_known_non_caching_op(other) => "no weight-scaled transpose cache",
                _ => return None,
            }),
        })
    }

    /// The bulk of the registry: ops with no weight-scaled cache. Kept as an
    /// explicit list, not a wildcard, so a *new* op falls through to `None` and
    /// fails the test instead of being absorbed.
    fn is_known_non_caching_op(op_type: &str) -> bool {
        const NON_CACHING: &[&str] = &[
            "Abs",
            "Acos",
            "Acosh",
            "Add",
            "AffineGrid",
            "And",
            "ArgMax",
            "ArgMin",
            "Asin",
            "Asinh",
            "Atan",
            "Atanh",
            "AveragePool",
            "BatchNormalization",
            "BiasGelu",
            "BiasSplitGelu",
            "BitShift",
            "BitwiseAnd",
            "BlackmanWindow",
            "BitwiseNot",
            "BitwiseOr",
            "BitwiseXor",
            "Cast",
            "CastLike",
            "CausalConvWithState",
            "Ceil",
            "Celu",
            "CenterCropPad",
            "Clip",
            "Col2Im",
            "Compress",
            "Concat",
            "Constant",
            "ConstantOfShape",
            "Conv",
            "ConvInteger",
            "ConvTranspose",
            "Cos",
            "Cosh",
            "CumProd",
            "CumSum",
            "DFT",
            "DepthToSpace",
            "DequantizeLinear",
            "Div",
            "Dropout",
            "DsaIndexSelect",
            "DynamicQuantizeLinear",
            "Einsum",
            "Elu",
            "Equal",
            "Erf",
            "Exp",
            "Expand",
            "EyeLike",
            "FastGelu",
            "Flatten",
            "Floor",
            "Gather",
            "GatherElements",
            "GatherND",
            "Gelu",
            "GlobalAveragePool",
            "GlobalLpPool",
            "GlobalMaxPool",
            "Greater",
            "GreaterOrEqual",
            "GridSample",
            "GroupNorm",
            "GroupNormalization",
            "HammingWindow",
            "HannWindow",
            "HardSigmoid",
            "HardSwish",
            "Hardmax",
            "Identity",
            "If",
            "IndexShare",
            "InstanceNormalization",
            "IsInf",
            "IsNaN",
            "LayerNormalization",
            "LeakyRelu",
            "Less",
            "LessOrEqual",
            "Log",
            "LogSoftmax",
            "Loop",
            "LpNormalization",
            "LpPool",
            "MatMulBnb4",
            "Max",
            "MaxPool",
            "Mean",
            "MelWeightMatrix",
            "Min",
            "Mish",
            "Mod",
            "Mul",
            "Neg",
            "NegativeLogLikelihoodLoss",
            "NhwcConv",
            "NonMaxSuppression",
            "NonZero",
            "Not",
            "OneHot",
            "Or",
            "PRelu",
            "Pad",
            "Pow",
            "QLinearAdd",
            "QLinearAveragePool",
            "QLinearConcat",
            "QLinearConv",
            "QLinearGlobalAveragePool",
            "QLinearLeakyRelu",
            "QLinearMul",
            "QLinearSigmoid",
            "QLinearSoftmax",
            "QuantizeLinear",
            "QuickGelu",
            "RMSNormalization",
            "RandomNormal",
            "RandomNormalLike",
            "RandomUniform",
            "RandomUniformLike",
            "Range",
            "Reciprocal",
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceMax",
            "ReduceMean",
            "ReduceMin",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
            "Relu",
            "Reshape",
            "Resize",
            "RotaryEmbedding",
            "Round",
            "STFT",
            "ScatterElements",
            "ScatterND",
            "Selu",
            "Shape",
            "Shrink",
            "Sigmoid",
            "Sign",
            "Silu",
            "SimplifiedLayerNormalization",
            "Sin",
            "Sinh",
            "Size",
            "Skip",
            "SkipLayerNormalization",
            "SkipSimplifiedLayerNormalization",
            "Slice",
            "Softmax",
            "SoftmaxCrossEntropyLoss",
            "Softplus",
            "Softsign",
            "SpaceToDepth",
            "Split",
            "Sqrt",
            "Squeeze",
            "Sub",
            "Sum",
            "Swish",
            "Tan",
            "Tanh",
            "TensorScatter",
            "ThresholdedRelu",
            "Tile",
            "TopK",
            "Transpose",
            "Trilu",
            "Unique",
            "Unsqueeze",
            "Upsample",
            "Where",
            "Xor",
        ];
        NON_CACHING.contains(&op_type)
    }

    /// **The anti-omission property.** Walks the *real* registry, so an op
    /// cannot be added without a deliberate accounting decision.
    #[test]
    fn every_registered_cpu_op_is_classified_for_weight_cache_accounting() {
        let registry = crate::kernels::build_cpu_registry();
        let mut unclassified: Vec<String> = registry
            .keys()
            .map(|key| key.op_type.clone())
            .filter(|op| weight_cache_accounting_classification(op).is_none())
            .collect();
        unclassified.sort();
        unclassified.dedup();
        assert!(
            unclassified.is_empty(),
            "these CPU kernels are registered but have no weight-cache accounting \
             classification: {unclassified:?}\n\
             \n\
             Add each to `weight_cache_accounting_classification` in \
             `crates/onnx-runtime-ep-cpu/src/kernels/matmul.rs`. If the kernel can \
             route a *constant* weight through `MatMulPrepack::transposed_b*` \
             (directly or via `try_half_decode_gemv`), classify it \
             `CachesTransposedWeight` AND give it an arm in \
             `node_weight_transpose_cache_bytes`, or the memory plan will \
             under-budget it (#1702). Otherwise classify it `NoWeightCache` with \
             the reason it retains nothing."
        );
    }

    /// A single-node fixture for a supported transpose route: f32 for the
    /// dense-only fusion, f16 for the shared half-decode family.
    fn transpose_weight_graph(op_type: &str, domain: &str, k: usize, n: usize) -> Graph {
        let mut graph = Graph::new();
        let dtype = if op_type == "FusedGemm" {
            DataType::Float32
        } else {
            DataType::Float16
        };
        let width = if dtype == DataType::Float32 { 4 } else { 2 };
        let a = graph.create_named_value("A", dtype, static_shape([1, k]));
        let b = graph.create_named_value("B", dtype, static_shape([k, n]));
        let y = graph.create_named_value("Y", dtype, static_shape([1, n]));
        graph.add_input(a);
        graph.add_input(b);
        let mut inputs = vec![Some(a), Some(b)];
        if matches!(op_type, "FusedMatMulBias" | "FusedGemm") {
            let bias = graph.create_named_value("C", dtype, static_shape([n]));
            graph.add_input(bias);
            inputs.push(Some(bias));
            graph.set_initializer(
                bias,
                WeightRef::Inline(TensorData::from_raw(dtype, vec![n], vec![0u8; n * width])),
            );
        }
        let mut node = Node::new(NodeId(0), op_type, inputs, vec![y]);
        node.domain = domain.to_string();
        graph.insert_node(node);
        graph.add_output(y);
        graph.set_initializer(
            b,
            WeightRef::Inline(TensorData::from_raw(
                dtype,
                vec![k, n],
                vec![0u8; k * n * width],
            )),
        );
        graph
    }

    /// Every registered transpose route has its declared-width forecast.
    /// Public provider and Gemm controls separately reconcile actual Host grants.
    #[test]
    fn transpose_forecast_covers_every_registered_preparation_route() {
        // Reads only the pure `node_weight_transpose_cache_bytes` predictor, so
        // unlike `gemm.rs`'s reconciliation it touches no process-global cache
        // and needs no global admission overrides (#1056).
        // Distinctive geometry so a stray same-shaped entry cannot alias it,
        // and an odd `k` so a tail-handling bug cannot hide.
        let (k, n) = (83usize, 41usize);
        let registry = crate::kernels::build_cpu_registry();

        let mut checked = 0usize;
        for key in registry.keys() {
            if weight_cache_accounting_classification(&key.op_type)
                != Some(WeightCacheClass::CachesTransposedWeight)
            {
                continue;
            }
            // `Gemm` reaches the cache through its own `transB` arm, which
            // `gemm.rs` reconciles at f32 width; this graph builder feeds the
            // 16-bit `transB = 0` route the MatMul family shares.
            let graph = transpose_weight_graph(&key.op_type, &key.domain, k, n);
            let node = graph.nodes.values().next().expect("single-node graph");
            let predicted = node_weight_transpose_cache_bytes(node, &graph);
            // A binary predicts exactly what its own kernels allocate. The
            // shared 16-bit decode GEMV exists only on x86. Apple's Accelerate
            // paths retain the same f16 transpose for MatMul and
            // FusedMatMulBias, but Gemm does not enter those paths.
            let x86 = cfg!(any(target_arch = "x86", target_arch = "x86_64"));
            let apple_matmul =
                cfg!(any(target_os = "macos", target_os = "ios")) && key.op_type != "Gemm";
            let expected = if key.op_type == "FusedGemm" {
                if cfg!(any(target_os = "macos", target_os = "ios")) {
                    (k * n * 4) as u64
                } else {
                    0
                }
            } else if x86 || apple_matmul {
                (k * n * 2) as u64
            } else {
                0
            };
            assert_eq!(
                predicted, expected,
                "{} must predict exactly the transpose bytes this target's \
                 kernels retain; a 0 where the GEMV exists is the #1702 \
                 under-budget bug, and a non-zero where it does not exist \
                 over-budgets every projection",
                key.op_type
            );
            checked += 1;
        }
        assert!(
            checked >= 4,
            "expected MatMul, FusedMatMulBias, FusedGemm and Gemm to be reconciled, \
             got {checked} -- did the classification silently narrow?"
        );
    }

    /// **The domain-gate property, enumerated from the registry.**
    ///
    /// The Opus review of #1702 found the *same* defect a second time, in
    /// `node_matmul_dense_cache_bytes`: a blanket `!node.is_default_domain()`
    /// bail silently zeroing every `FusedMatMulBias` node. Two independent
    /// instances of one mistake is a pattern, not an accident, and the
    /// transpose-only guard above would not have caught the second one.
    ///
    /// So this checks **both** predictors, and it takes the domain from
    /// `OpRegistry::keys()` rather than from a literal. That is the whole
    /// point: the registry is the only place that knows an op ships in
    /// `com.microsoft`, so a predictor that gates on the wrong domain is
    /// caught by construction instead of by someone remembering. A third
    /// weight-scaled predictor added later gets covered by adding one line
    /// here, and until then its absence is at least visible in one place.
    #[test]
    fn no_predictor_zeroes_a_caching_op_because_of_its_shipping_domain() {
        let (k, n) = (12usize, 9usize);
        let registry = crate::kernels::build_cpu_registry();
        let mut checked = 0usize;

        for key in registry.keys() {
            if weight_cache_accounting_classification(&key.op_type)
                != Some(WeightCacheClass::CachesTransposedWeight)
            {
                continue;
            }
            // Built with the domain the registry says this op ships in. A
            // predictor gated on `is_default_domain()` returns 0 here for any
            // contrib-domain op while still looking correct in a test that
            // hardcodes `domain = ""`.
            let graph = transpose_weight_graph(&key.op_type, &key.domain, k, n);
            let node = graph.nodes.values().next().expect("single-node graph");

            // `Gemm` is excluded from the dense predictor by design (it has its
            // own f32 transpose accounting), so only the MatMul family is
            // checked there; the transpose predictor covers all three.
            if key.op_type == "MatMul" || key.op_type == "FusedMatMulBias" {
                assert_ne!(
                    node_matmul_dense_cache_bytes(node, &graph),
                    0,
                    "`node_matmul_dense_cache_bytes` returned 0 for {} in its \
                     shipping domain {:?}. If that is because of a domain gate, \
                     it is the #1702 defect again: gate each op to the domain \
                     it actually ships in, not to one shared \
                     `is_default_domain()` check.",
                    key.op_type,
                    key.domain,
                );
            }
            let x86 = cfg!(any(target_arch = "x86", target_arch = "x86_64"));
            let apple_matmul =
                cfg!(any(target_os = "macos", target_os = "ios")) && key.op_type != "Gemm";
            if (x86 && key.op_type != "FusedGemm") || apple_matmul {
                assert_ne!(
                    node_weight_transpose_cache_bytes(node, &graph),
                    0,
                    "`node_weight_transpose_cache_bytes` returned 0 for {} in \
                     its shipping domain {:?} on a target whose kernels do \
                     retain the transpose",
                    key.op_type,
                    key.domain,
                );
            }
            checked += 1;
        }
        assert!(
            checked >= 4,
            "expected MatMul, FusedMatMulBias, FusedGemm and Gemm; got {checked}"
        );
    }

    /// Mutation check: the predictor must key on *reaching the shared helper*,
    /// not on an op-name allowlist. Restoring the pre-#1702 behaviour (naming
    /// `MatMul` and excluding `FusedMatMulBias`) must be observable here.
    #[test]
    fn fused_matmul_bias_is_budgeted_exactly_like_matmul() {
        let (k, n) = (83usize, 41usize);
        let bytes = |op: &str, domain: &str| {
            let graph = transpose_weight_graph(op, domain, k, n);
            let node = graph.nodes.values().next().unwrap();
            node_weight_transpose_cache_bytes(node, &graph)
        };
        // Arch-independent: whatever `MatMul` is budgeted on this target,
        // `FusedMatMulBias` must be budgeted the same, because since #1702 they
        // are the same code path. That equality is the invariant, so the test
        // does not need to know which target it is on.
        let matmul = bytes("MatMul", "");
        // The *shipping* domain: the optimizer emits the fusion into contrib,
        // and the predictor's blanket non-default-domain bail used to zero it
        // there even once the op name was accepted. Passing "" here would make
        // this test pass against a predictor that is dead in production.
        let fused = bytes("FusedMatMulBias", "com.microsoft");
        assert_eq!(
            fused, matmul,
            "since #1702 `FusedMatMulBias` routes through the *same* \
             `try_half_decode_gemv` as `MatMul`, so it retains the same bytes \
             and must be budgeted identically. A 0 here means the predictor \
             went back to an op-name allowlist and the plan under-budgets \
             every fused projection."
        );
    }

    /// A non-constant `B` is not cacheable, so neither op may budget for it.
    /// Guards the opposite error: over-budgeting on the strength of op name
    /// alone.
    #[test]
    fn no_op_budgets_a_non_constant_weight() {
        let mut graph = Graph::new();
        let a = graph.create_named_value("A", DataType::Float16, static_shape([1, 8]));
        let b = graph.create_named_value("B", DataType::Float16, static_shape([8, 4]));
        let y = graph.create_named_value("Y", DataType::Float16, static_shape([1, 4]));
        graph.add_input(a);
        graph.add_input(b);
        let mut node = Node::new(
            NodeId(0),
            "FusedMatMulBias",
            vec![Some(a), Some(b)],
            vec![y],
        );
        node.domain = "com.microsoft".to_string();
        graph.insert_node(node);
        graph.add_output(y);
        // No `set_initializer`: `B` stays an activation.
        let node = graph.nodes.values().next().unwrap();
        assert_eq!(
            node_weight_transpose_cache_bytes(node, &graph),
            0,
            "an activation `B` is never cached, so it must never be budgeted"
        );
    }
}

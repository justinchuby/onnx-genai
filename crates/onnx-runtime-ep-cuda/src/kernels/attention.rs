//! Scaled-dot-product / grouped-query **attention** on the GPU
//! (`docs/architecture/ORT2.md` §13 + §15.5).
//!
//! Multi-token prefill uses an NVRTC-compiled tiled online-softmax kernel that
//! keeps score tiles and softmax state in SRAM/registers. It does not allocate
//! the `[B,H,Sq,Sk]` score tensor. The unfused cuBLASLt → softmax → cuBLASLt path
//! serves single-token decode and prefill shapes not selected for fusion.
//!
//! ## What it computes
//!
//! ```text
//! O = softmax( scale · Q·Kᵀ  [+ mask] , axis = keys ) · V
//! ```
//!
//! with `Q : [B, num_heads, Sq, D]`, `K,V : [B, num_kv_heads, Sk, D]`, and
//! `O : [B, num_heads, Sq, D]`, all row-major f32/f16/bf16.
//!
//! ## cuBLASLt/softmax fallback — two batched cuBLAS GEMMs around one NVRTC softmax
//!
//! 1. **Scores** `S = scale·Q·Kᵀ` via [`blas::gemm_ex`]. cuBLAS is
//!    column-major; a row-major `X[r,c]` (ld=c) is byte-identically the
//!    column-major `Xᵀ[c,r]`. We want the row-major bytes of `S[Sq,Sk]`, i.e.
//!    the column-major `Sᵀ = K·Qᵀ`, so we ask cuBLAS for
//!    `C[m=Sk, n=Sq] = opᵀ(K) · op(Q)` with `k = D` (`transa = T`,
//!    `transb = N`). The softmax `scale` folds into the GEMM `alpha` for free.
//! 2. **Softmax** over the last (keys) axis of `S`, fused with the `scale`
//!    (already applied), the optional additive `mask`, and the `causal`
//!    upper-triangular mask, in a single NVRTC-compiled kernel. Numerically
//!    stable (subtract row max). Runs in place, turning `S` into the
//!    probabilities `P`.
//! 3. **Output** `O = P·V` via [`blas::gemm_ex`]. Row-major `O[Sq,D]` bytes are
//!    the column-major `Oᵀ = Vᵀ·Pᵀ`, i.e. `C[m=D, n=Sq] = op(V) · op(P)` with
//!    `k = Sk` (`transa = N`, `transb = N`).
//!
//! All three steps submit onto the EP's single stream, so their ordering is
//! implicit. The runtime synchronization policy is applied at the end; when
//! deferred, its success is not a device-completion witness.
//!
//! ## GQA / MQA
//!
//! `num_kv_heads` may be smaller than `num_heads`; each KV head is shared by a
//! contiguous group of `num_heads / num_kv_heads` query heads. The baseline
//! iterates `(batch, query-head)` and points the QKᵀ / PV GEMMs at the KV head
//! `h / group`, so the KV broadcast costs no extra memory (no materialised
//! expansion). Per-`(b,h)` GEMMs keep the GQA pointer mapping trivially correct;
//! collapsing them into a single strided-batch call (KV stride 0 within a group)
//! remains a fallback-path throughput optimisation.
//!
//! ## Input requirements
//!
//! * Q/K/V must use f32, f16, or bf16.
//! * Q/K/V must use the explicit 4-D `[B, H, S, D]` layout.
//! * non-contiguous (strided) Q/K/V/O or mask → actionable "materialise" error.

use std::alloc::Layout;
use std::ffi::c_void;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUdeviceptr;

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView, WorkspaceLayout,
    WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node};

use crate::blas::{GemmDtype, GemmEx, WORKSPACE_BYTES, gemm_ex};
use crate::error::{driver_err, not_implemented};
use crate::kernels::kv_stride::KvCacheStrides;
use crate::runtime::{CudaRuntime, cuptr};

use super::flash_attention;

/// NVRTC source for the fused, numerically-stable softmax over the last (keys)
/// axis of the score matrix, with `causal` + optional additive `mask` folded in.
///
/// One thread block per score row (`= B · num_heads · Sq` rows); the block
/// cooperatively reduces the row max then the row sum in shared memory. The
/// `scale` is already baked into the scores by the QKᵀ GEMM `alpha`.
const SOFTMAX_SRC: &str = r#"
extern "C" __global__ void attn_softmax_f32(
    float*       scores,       // [nrows, sk] row-major, in/out
    const float* mask,         // additive mask planes, or null when mask_planes==0
    const int*   total_lengths,// optional logical key lengths [batch]
    const int*   past_lengths, // optional logical past lengths [batch]
    const int    nrows,        // B * heads * sq
    const int    sk,           // key length (softmax axis)
    const int    sq,           // query length
    const int    heads,        // num query heads
    const int    causal,       // 0/1
    const int    mask_planes,  // 0 (none), 1, batch, or batch*heads
    const int    batch,
    const int    local_window,
    const float  softcap)
{
    // NVRTC has no <math.h>: build +inf from its bit pattern.
    const float INF = __int_as_float(0x7f800000);

    const int row = blockIdx.x;
    if (row >= nrows) return;

    // row = ((b*heads) + h)*sq + i
    const int i  = row % sq;
    const int bh = row / sq;
    const int b  = bh / heads;

    float* s = scores + (size_t)row * sk;

    // Causal alignment: query i (absolute position sk-sq+i for cached decode)
    // attends to keys j <= sk-sq+i. Reduces to lower-triangular when sq==sk.
    const int causal_max = past_lengths ? past_lengths[b] + i : sk - sq + i;
    const int logical_sk = total_lengths ? total_lengths[b] : sk;
    const int local_min = local_window > 0
        ? max(0, causal_max + 1 - local_window)
        : 0;

    const float* mrow = 0;
    if (mask_planes > 0) {
        int plane = 0;
        if (mask_planes == batch)            plane = b;
        else if (mask_planes == batch*heads) plane = bh;
        // else mask_planes == 1 -> plane 0 (shared [sq,sk])
        mrow = mask + ((size_t)plane * sq + i) * sk;
    }

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    // Pass 1: apply masks, find the row max.
    float local_max = -INF;
    for (int j = tid; j < sk; j += nt) {
        float v;
        if (j >= logical_sk || (causal && j > causal_max) || j < local_min) {
            v = -INF;
        } else {
            v = s[j];
            if (softcap > 0.0f) v = softcap * tanhf(v / softcap);
            if (mrow) v += mrow[j];
        }
        s[j] = v;
        local_max = fmaxf(local_max, v);
    }
    red[tid] = local_max;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    const float row_max = red[0];
    __syncthreads();

    // Pass 2: exponentiate (stable) and sum. A fully-masked row (max == -inf)
    // yields all-zero exponentials.
    float local_sum = 0.0f;
    for (int j = tid; j < sk; j += nt) {
        const float v = s[j];
        const float e = (v == -INF) ? 0.0f : expf(v - row_max);
        s[j] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float row_sum = red[0];
    __syncthreads();

    // Pass 3: normalise (guard the degenerate fully-masked row).
    const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int j = tid; j < sk; j += nt) {
        s[j] *= inv;
    }
}
"#;

/// Half-precision softmax variants. Inputs and outputs remain in the attention
/// dtype, while every value participating in max/exp/sum arithmetic is widened
/// to f32. The f32 source above remains separate so its established path and
/// generated code are unchanged.
const SOFTMAX_HALF_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T> __device__ float load_float(T value);
template <> __device__ float load_float<__half>(__half value) { return __half2float(value); }
template <> __device__ float load_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T> __device__ T store_float(float value);
template <> __device__ __half store_float<__half>(float value) {
    return __float2half_rn(value);
}
template <> __device__ __nv_bfloat16 store_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

#define DEFINE_ATTN_SOFTMAX(TYPE, SUFFIX) \
extern "C" __global__ void attn_softmax_##SUFFIX( \
    TYPE*        scores, \
    const TYPE*  mask, \
    const int*   total_lengths, \
    const int*   past_lengths, \
    const int    nrows, \
    const int    sk, \
    const int    sq, \
    const int    heads, \
    const int    causal, \
    const int    mask_planes, \
    const int    batch, \
    const int    local_window, \
    const float  softcap) \
{ \
    const float INF = __int_as_float(0x7f800000); \
    const int row = blockIdx.x; \
    if (row >= nrows) return; \
    const int i  = row % sq; \
    const int bh = row / sq; \
    const int b  = bh / heads; \
    TYPE* s = scores + (size_t)row * sk; \
    const int causal_max = past_lengths ? past_lengths[b] + i : sk - sq + i; \
    const int logical_sk = total_lengths ? total_lengths[b] : sk; \
    const int local_min = local_window > 0 \
        ? max(0, causal_max + 1 - local_window) \
        : 0; \
    const TYPE* mrow = 0; \
    if (mask_planes > 0) { \
        int plane = 0; \
        if (mask_planes == batch)            plane = b; \
        else if (mask_planes == batch*heads) plane = bh; \
        mrow = mask + ((size_t)plane * sq + i) * sk; \
    } \
    extern __shared__ float red[]; \
    const int tid = threadIdx.x; \
    const int nt  = blockDim.x; \
    float local_max = -INF; \
    for (int j = tid; j < sk; j += nt) { \
        float v; \
        if (j >= logical_sk || (causal && j > causal_max) || j < local_min) { \
            v = -INF; \
        } else { \
            v = load_float<TYPE>(s[j]); \
            if (softcap > 0.0f) v = softcap * tanhf(v / softcap); \
            if (mrow) v += load_float<TYPE>(mrow[j]); \
        } \
        const TYPE stored = store_float<TYPE>(v); \
        s[j] = stored; \
        local_max = fmaxf(local_max, load_float<TYPE>(stored)); \
    } \
    red[tid] = local_max; \
    __syncthreads(); \
    for (int off = nt >> 1; off > 0; off >>= 1) { \
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]); \
        __syncthreads(); \
    } \
    const float row_max = red[0]; \
    __syncthreads(); \
    float local_sum = 0.0f; \
    for (int j = tid; j < sk; j += nt) { \
        const float v = load_float<TYPE>(s[j]); \
        const float e = (v == -INF) ? 0.0f : expf(v - row_max); \
        s[j] = store_float<TYPE>(e); \
        local_sum += e; \
    } \
    red[tid] = local_sum; \
    __syncthreads(); \
    for (int off = nt >> 1; off > 0; off >>= 1) { \
        if (tid < off) red[tid] += red[tid + off]; \
        __syncthreads(); \
    } \
    const float row_sum = red[0]; \
    __syncthreads(); \
    const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f; \
    for (int j = tid; j < sk; j += nt) { \
        s[j] = store_float<TYPE>(load_float<TYPE>(s[j]) * inv); \
    } \
}

DEFINE_ATTN_SOFTMAX(__half, f16)
DEFINE_ATTN_SOFTMAX(__nv_bfloat16, bf16)
"#;

/// Stable module + entry-point names for the NVRTC softmax (see
/// [`CudaRuntime::nvrtc_function`]).
const SOFTMAX_MODULE: &str = "attn_softmax_f32";
const SOFTMAX_ENTRY: &str = "attn_softmax_f32";
const SOFTMAX_HALF_MODULE: &str = "attn_softmax_half_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttentionDtype {
    F32,
    F16,
    Bf16,
}

impl AttentionDtype {
    pub(super) fn from_onnx(dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::Float32 => Ok(Self::F32),
            DataType::Float16 => Ok(Self::F16),
            DataType::BFloat16 => Ok(Self::Bf16),
            other => Err(not_implemented(format!(
                "Attention with dtype {other:?} (supported: Float32, Float16, BFloat16)"
            ))),
        }
    }

    fn gemm(self) -> GemmDtype {
        match self {
            Self::F32 => GemmDtype::F32,
            Self::F16 => GemmDtype::F16,
            Self::Bf16 => GemmDtype::Bf16,
        }
    }

    pub(super) fn element_size(self) -> u64 {
        match self {
            Self::F32 => std::mem::size_of::<f32>() as u64,
            Self::F16 | Self::Bf16 => std::mem::size_of::<u16>() as u64,
        }
    }

    fn softmax(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::F32 => (SOFTMAX_MODULE, SOFTMAX_SRC, SOFTMAX_ENTRY),
            Self::F16 => (SOFTMAX_HALF_MODULE, SOFTMAX_HALF_SRC, "attn_softmax_f16"),
            Self::Bf16 => (SOFTMAX_HALF_MODULE, SOFTMAX_HALF_SRC, "attn_softmax_bf16"),
        }
    }
}

/// Threads per block for the softmax reduction (a power of two, so the tree
/// reduction is exact); rows longer than this are handled by the strided loop.
const SOFTMAX_BLOCK: u32 = 256;

/// Alignment (bytes) of the governed cuBLASLt/softmax attention scratch blob. cuBLASLt
/// requires its workspace at a 256-byte boundary, and the `[B,H,Sq,Sk]` scores
/// buffer that precedes it is padded to the same boundary so the workspace stays
/// aligned inside one suballocation.
pub(super) const BLAS_SOFTMAX_SCRATCH_ALIGN: usize = 256;

/// The complete scores + cuBLASLt layout is shared by planning and execution.
/// Checked sizes and binding use the common workspace contract: the kernel
/// partitions an exclusive loan and never admits or owns scratch here.
pub(super) fn blas_softmax_scratch_layout(
    dtype: AttentionDtype,
    batch: usize,
    num_heads: usize,
    sq: usize,
    sk: usize,
) -> Result<WorkspaceLayout<2>> {
    WorkspaceLayout::new([
        blas_softmax_scores_layout(dtype, batch, num_heads, sq, sk)?,
        Layout::from_size_align(WORKSPACE_BYTES, BLAS_SOFTMAX_SCRATCH_ALIGN)
            .expect("fixed cuBLASLt workspace has an addressable 256-byte-aligned layout"),
    ])
}

fn blas_softmax_scores_layout(
    dtype: AttentionDtype,
    batch: usize,
    num_heads: usize,
    sq: usize,
    sk: usize,
) -> Result<Layout> {
    let rejected = || {
        EpError::KernelFailed(format!(
            "cuda_ep Attention: score workspace exceeds addressable capacity for batch={batch}, heads={num_heads}, seq_q={sq}, seq_k={sk}, dtype={dtype:?}; reduce the prepared shape"
        ))
    };
    let scores_bytes = [batch, num_heads, sq, sk, dtype.element_size() as usize]
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(rejected)?;
    Layout::from_size_align(scores_bytes, BLAS_SOFTMAX_SCRATCH_ALIGN).map_err(|_| rejected())
}

/// Resolved device pointers for the cuBLASLt/softmax scratch, carved from a governed
/// workspace blob. Holding no ownership, it is safe to hand to the compute path
/// that never frees these pointers (the executor owns the workspace lifetime).
#[derive(Clone, Copy)]
pub(super) struct BlasSoftmaxScratch {
    scores: CUdeviceptr,
    scores_bytes: usize,
    workspace: CUdeviceptr,
    workspace_bytes: usize,
}

impl BlasSoftmaxScratch {
    fn validate_geometry(
        &self,
        dtype: AttentionDtype,
        batch: usize,
        heads: usize,
        sq: usize,
        sk: usize,
    ) -> Result<()> {
        let required = blas_softmax_scores_layout(dtype, batch, heads, sq, sk)?.size();
        if self.scores_bytes < required {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: borrowed score capacity {} bytes is smaller than {required} bytes for batch={batch}, heads={heads}, seq_q={sq}, seq_k={sk}, dtype={dtype:?}; reprepare caller workspace for this invocation",
                self.scores_bytes
            )));
        }
        Ok(())
    }

    pub(super) fn bind(
        layout: &WorkspaceLayout<2>,
        workspace: Option<WorkspaceView>,
    ) -> Result<Self> {
        let [scores, blas] = layout.bind(workspace).map_err(|error| {
            EpError::KernelFailed(format!(
                "cuda_ep Attention: cannot bind prepared scores/cuBLASLt workspace: {error}; call workspace_requirement and execute_with_workspace using executor-prepared backing"
            ))
        })?;
        Ok(Self {
            scores: cuptr(scores.ptr().0.cast_const()),
            scores_bytes: scores.bytes(),
            workspace: cuptr(blas.ptr().0.cast_const()),
            workspace_bytes: blas.bytes(),
        })
    }
}

/// Factory for [`AttentionKernel`]; reads the §13.3 binding attributes.
///
/// Attributes (model-agnostic — all runtime data, RULES.md #2):
/// * `num_heads` (int, **required**) — number of query heads.
/// * `kv_num_heads` (int, optional; default `num_heads`) — GQA/MQA KV heads.
/// * `causal` (int 0/1, optional; default 0) — causal masking.
/// * `scale` (float, optional; default `1/sqrt(head_dim)`).
pub struct AttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for AttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let num_heads = node
            .attr("num_heads")
            .and_then(|a| a.as_int())
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep Attention: missing required int `num_heads` attribute".into(),
                )
            })?;
        if num_heads <= 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: `num_heads` must be positive, got {num_heads}"
            )));
        }
        let num_kv_heads = node
            .attr("kv_num_heads")
            .and_then(|a| a.as_int())
            .unwrap_or(num_heads);
        let causal = node.attr("causal").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        let scale = node.attr("scale").and_then(|a| a.as_float());

        AttentionKernel::new(
            self.runtime.clone(),
            causal,
            num_heads as usize,
            num_kv_heads as usize,
            scale,
        )
        .map(|k| Box::new(k) as Box<dyn Kernel>)
    }
}

/// SDPA/GQA attention with a fused prefill path and cuBLASLt/softmax fallback.
#[derive(Debug)]
pub struct AttentionKernel {
    runtime: Arc<CudaRuntime>,
    causal: bool,
    num_heads: usize,
    num_kv_heads: usize,
    /// Softmax scale; `None` means the default `1/sqrt(head_dim)`, resolved once
    /// `head_dim` is known from the Q shape at execute time.
    scale: Option<f32>,
    mode: AttentionMode,
    last_call_capture_safe: AtomicBool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttentionMode {
    Auto,
    Fused,
    BlasSoftmax,
}

impl AttentionKernel {
    /// Direct constructor (the testable §13.3-style entry point, independent of
    /// the not-yet-wired fusion pass). `num_kv_heads` must divide `num_heads`.
    pub fn new(
        runtime: Arc<CudaRuntime>,
        causal: bool,
        num_heads: usize,
        num_kv_heads: usize,
        scale: Option<f32>,
    ) -> Result<Self> {
        if num_heads == 0 || num_kv_heads == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep Attention: num_heads and num_kv_heads must be non-zero".into(),
            ));
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: num_heads ({num_heads}) must be a multiple of \
                 num_kv_heads ({num_kv_heads}) for grouped-query attention"
            )));
        }
        Ok(Self {
            runtime,
            causal,
            num_heads,
            num_kv_heads,
            scale,
            mode: AttentionMode::Auto,
            last_call_capture_safe: AtomicBool::new(false),
        })
    }

    /// Construct the memory-efficient fused implementation directly.
    ///
    /// The normal [`Self::new`] constructor applies a measured performance
    /// heuristic that can choose cuBLASLt/softmax for longer shapes. This
    /// constructor bypasses that heuristic for parity and memory benchmarks;
    /// unsupported shapes still fall back safely.
    pub fn new_fused(
        runtime: Arc<CudaRuntime>,
        causal: bool,
        num_heads: usize,
        num_kv_heads: usize,
        scale: Option<f32>,
    ) -> Result<Self> {
        let mut kernel = Self::new(runtime, causal, num_heads, num_kv_heads, scale)?;
        kernel.mode = AttentionMode::Fused;
        Ok(kernel)
    }

    /// Construct the retained cuBLASLt/softmax implementation directly.
    ///
    /// This is primarily a parity/benchmark oracle. Production callers should
    /// use [`Self::new`], which selects fused prefill when supported.
    pub fn new_blas_softmax(
        runtime: Arc<CudaRuntime>,
        causal: bool,
        num_heads: usize,
        num_kv_heads: usize,
        scale: Option<f32>,
    ) -> Result<Self> {
        let mut kernel = Self::new(runtime, causal, num_heads, num_kv_heads, scale)?;
        kernel.mode = AttentionMode::BlasSoftmax;
        Ok(kernel)
    }

    /// Select the fused (shared-memory) prefill path over the cuBLASLt/softmax
    /// path. Prepare-only workspace planning and execution both call this so the
    /// reserved scratch (zero for the fused path) matches what runs.
    fn use_fused(&self, dtype: DataType, sq: usize, sk: usize, d: usize) -> bool {
        let fused_supported = flash_attention::supported(sq, d);
        let measured_fused_win = sq.max(sk) <= 128
            || (dtype == DataType::Float16
                && d.is_multiple_of(16)
                && sq.max(sk) <= 512
                && self.runtime.capabilities().compute_capability().0 >= 7);
        match self.mode {
            AttentionMode::Auto => fused_supported && measured_fused_win,
            AttentionMode::Fused => fused_supported,
            AttentionMode::BlasSoftmax => false,
        }
    }

    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        if !(3..=4).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: expected 3 inputs (Q,K,V) or 4 (Q,K,V,mask) \
                 and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        let q = &inputs[0];
        let k = &inputs[1];
        let v = &inputs[2];
        let mask = inputs.get(3);

        let dtype = AttentionDtype::from_onnx(q.dtype)?;
        for (name, dt) in [
            ("Q", q.dtype),
            ("K", k.dtype),
            ("V", v.dtype),
            ("O", outputs[0].dtype),
        ] {
            if dt != q.dtype {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Attention: Q/K/V/output dtypes must match; \
                     Q is {:?}, {name} is {dt:?}",
                    q.dtype
                )));
            }
        }
        if dtype != AttentionDtype::F32 {
            self.runtime.require_nvrtc_half_headers("Attention")?;
        }

        // Explicit 4-D [B, heads, seq, head_dim] layout only.
        for (name, t) in [("Q", q.shape), ("K", k.shape), ("V", v.shape)] {
            if t.len() != 4 {
                return Err(not_implemented(format!(
                    "Attention with {name} rank {} (CUDA Attention expects 4-D \
                     [batch, heads, seq, head_dim]); reshape/transpose upstream",
                    t.len()
                )));
            }
        }

        let (batch, hq, sq, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
        let (bk, hk, sk, dk) = (k.shape[0], k.shape[1], k.shape[2], k.shape[3]);

        if hq != self.num_heads || hk != self.num_kv_heads {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: Q heads {hq} / K heads {hk} disagree with \
                 num_heads {} / num_kv_heads {}",
                self.num_heads, self.num_kv_heads
            )));
        }
        if bk != batch || dk != d {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: Q {:?} and K {:?} must share batch and head_dim",
                q.shape, k.shape
            )));
        }
        if v.shape != [batch, self.num_kv_heads, sk, d] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: V shape {:?} must be [batch {batch}, kv_heads {}, \
                 seq_k {sk}, head_dim {d}]",
                v.shape, self.num_kv_heads
            )));
        }
        if outputs[0].shape != [batch, hq, sq, d] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: output shape {:?} must be [batch {batch}, \
                 heads {hq}, seq_q {sq}, head_dim {d}]",
                outputs[0].shape
            )));
        }

        // The baseline addresses per-head slices with plain pointer arithmetic,
        // so it requires dense row-major buffers.
        for (name, contiguous) in [
            ("Q", q.is_contiguous()),
            ("K", k.is_contiguous()),
            ("V", v.is_contiguous()),
            ("O", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(not_implemented(format!(
                    "Attention with a non-contiguous (strided) {name}; \
                     materialise it (insert a copy) before the attention op"
                )));
            }
        }

        let group = self.num_heads / self.num_kv_heads;
        let scale = self.scale.unwrap_or_else(|| 1.0 / (d as f32).sqrt());

        // Optional additive mask: same dtype as Q/K/V, contiguous, element count
        // a whole number of [sq,sk] planes broadcasting over
        // {1, batch, batch*heads}.
        let (mask_ptr, mask_planes) = match mask {
            None => (0u64, 0i32),
            Some(m) => {
                if m.dtype != q.dtype {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Attention: additive mask dtype {:?} must match Q dtype {:?}",
                        m.dtype, q.dtype
                    )));
                }
                if !m.is_contiguous() {
                    return Err(not_implemented(
                        "Attention with a non-contiguous (strided) mask; materialise it first",
                    ));
                }
                let plane = sq * sk;
                let n = m.numel();
                if plane == 0 || !n.is_multiple_of(plane) {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Attention: mask has {n} elements, not a whole number of \
                         [seq_q {sq}, seq_k {sk}] planes"
                    )));
                }
                let planes = n / plane;
                if planes != 1 && planes != batch && planes != batch * self.num_heads {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Attention: mask has {planes} [seq_q,seq_k] planes; expected a \
                         broadcastable 1, batch ({batch}), or batch*heads ({})",
                        batch * self.num_heads
                    )));
                }
                (cuptr(m.data_ptr::<u8>() as *const c_void), planes as i32)
            }
        };

        let q_base = cuptr(q.data_ptr::<u8>() as *const c_void);
        let k_base = cuptr(k.data_ptr::<u8>() as *const c_void);
        let v_base = cuptr(v.data_ptr::<u8>() as *const c_void);
        let o_base = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let fused = self.use_fused(q.dtype, sq, sk, d);
        // #757: keep capture admission truthful. The fused path is capture-safe;
        // the cuBLASLt/softmax path performs an unconditional trailing synchronize (see
        // `run_attention_blas_softmax`, allowlisted in capture_sync_contract.rs) and
        // must be refused during capture. §736 governs the cuBLASLt/softmax scratch for
        // admission, but that does not make the sync capture-safe.
        self.last_call_capture_safe.store(fused, Ordering::Relaxed);
        if !fused && self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(
                "cuda_ep Attention selected the synchronizing cuBLASLt/softmax path during CUDA graph capture; select a capture-safe fused implementation"
                    .into(),
            ));
        }
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let score_elements = (batch as u64)
                .saturating_mul(self.num_heads as u64)
                .saturating_mul(sq as u64)
                .saturating_mul(sk as u64);
            let qk_flops = score_elements.saturating_mul(d as u64).saturating_mul(2);
            let pv_flops = score_elements.saturating_mul(d as u64).saturating_mul(2);
            let softmax_flops = score_elements.saturating_mul(4).saturating_add(
                (batch as u64)
                    .saturating_mul(self.num_heads as u64)
                    .saturating_mul(sq as u64),
            );
            qk_flops
                .saturating_add(pv_flops)
                .saturating_add(softmax_flops)
        });
        if fused {
            flash_attention::run(
                &self.runtime,
                q.dtype,
                self.num_heads,
                self.num_kv_heads,
                self.causal,
                batch,
                sq,
                sk,
                sk,
                d,
                group,
                scale,
                q_base,
                k_base,
                v_base,
                o_base,
                mask_ptr,
                mask_planes,
                0,
                0,
                0,
                0.0,
                &KvCacheStrides::head_major_bnsh(),
            )
        } else {
            let layout = blas_softmax_scratch_layout(dtype, batch, self.num_heads, sq, sk)?;
            let scratch = BlasSoftmaxScratch::bind(&layout, workspace)?;
            run_attention_blas_softmax(
                &self.runtime,
                dtype,
                self.num_heads,
                self.num_kv_heads,
                self.causal,
                batch,
                sq,
                sk,
                d,
                sk,
                group,
                scale,
                q_base,
                k_base,
                v_base,
                o_base,
                mask_ptr,
                mask_planes,
                0,
                0,
                0,
                0.0,
                scratch,
            )
        }
    }
}

/// Dtype-dispatched attention engine. cuBLASLt receives the native IO dtype and
/// always accumulates GEMMs in fp32; the softmax kernel likewise widens every
/// reduction value to fp32 before narrowing probabilities to the IO dtype.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_attention_blas_softmax(
    runtime: &CudaRuntime,
    dtype: AttentionDtype,
    num_heads: usize,
    num_kv_heads: usize,
    causal: bool,
    batch: usize,
    sq: usize,
    sk: usize,
    d: usize,
    kv_capacity: usize,
    group: usize,
    scale: f32,
    q_base: CUdeviceptr,
    k_base: CUdeviceptr,
    v_base: CUdeviceptr,
    o_base: CUdeviceptr,
    mask_ptr: CUdeviceptr,
    mask_planes: i32,
    total_lengths: CUdeviceptr,
    past_lengths: CUdeviceptr,
    local_window: i32,
    softcap: f32,
    scratch: BlasSoftmaxScratch,
) -> Result<()> {
    let elem_size = dtype.element_size();
    scratch.validate_geometry(dtype, batch, num_heads, sq, sk)?;
    let BlasSoftmaxScratch {
        scores: scores_buf,
        workspace,
        workspace_bytes,
        ..
    } = scratch;
    let blas = runtime.blas();
    let stream = runtime.stream_ptr();

    // Stage 1: per-head S = scale · Q·Kᵀ.  Column-major C[Sk,Sq] = Kᵀ·Q.
    for b in 0..batch {
        for h in 0..num_heads {
            let kv = h / group;
            let q_head = q_base + ((b * num_heads + h) * sq * d) as u64 * elem_size;
            let k_head = k_base + ((b * num_kv_heads + kv) * kv_capacity * d) as u64 * elem_size;
            let s_head = scores_buf + ((b * num_heads + h) * sq * sk) as u64 * elem_size;

            let p = GemmEx {
                dtype: dtype.gemm(),
                transa: true,  // op(A=K) = Kᵀ  -> [Sk, D]
                transb: false, // op(B=Q) = Q   -> [D, Sq] (col-major view)
                m: sk,
                n: sq,
                k: d,
                alpha: scale,
                beta: 0.0,
                a: k_head,
                lda: d,
                b: q_head,
                ldb: d,
                c: s_head,
                ldc: sk,
                epilogue: None,
            };
            // SAFETY: per-head pointers lie inside the validated dense Q/K
            // and checked borrowed scores buffers; `workspace` is live;
            // `s_head` (output) aliases neither operand.
            unsafe { gemm_ex(blas, stream, &p, workspace, workspace_bytes) }?;
        }
    }

    // Stage 2: fused softmax over the keys axis (scale already applied).
    let nrows = batch * num_heads * sq;
    let (softmax_module, softmax_source, softmax_entry) = dtype.softmax();
    let func = runtime.nvrtc_function(softmax_module, softmax_source, softmax_entry)?;
    let cfg = runtime.reduction_launch_config(
        &func,
        nrows as u32,
        SOFTMAX_BLOCK,
        std::mem::size_of::<f32>() as u32,
    )?;
    let nrows_i = i32::try_from(nrows).map_err(|_| {
        EpError::KernelFailed(format!("cuda_ep Attention: {nrows} score rows exceed i32"))
    })?;
    let (sk_i, sq_i, heads_i, batch_i) = (sk as i32, sq as i32, num_heads as i32, batch as i32);
    let causal_i: i32 = causal.into();
    let stream_ref = runtime.stream();
    // Device pointers are passed by value (as u64) — a CUDA pointer kernel
    // parameter is ABI-identical to a 64-bit scalar argument.
    let mut builder = stream_ref.launch_builder(&func);
    builder
        .arg(&scores_buf)
        .arg(&mask_ptr)
        .arg(&total_lengths)
        .arg(&past_lengths)
        .arg(&nrows_i)
        .arg(&sk_i)
        .arg(&sq_i)
        .arg(&heads_i)
        .arg(&causal_i)
        .arg(&mask_planes)
        .arg(&batch_i)
        .arg(&local_window)
        .arg(&softcap);
    // SAFETY: `func` is the compiled softmax entry; the argument list and
    // its ABI match the kernel signature; `scores_buf`/`mask_ptr` are live
    // device allocations sized for [nrows, sk] / the mask planes.
    unsafe { builder.launch(cfg) }
        .map_err(|e| driver_err(&format!("launch {softmax_entry}"), e))?;

    // Stage 3: per-head O = P·V.  Column-major C[D,Sq] = Vᵀ·Pᵀ.
    for b in 0..batch {
        for h in 0..num_heads {
            let kv = h / group;
            let s_head = scores_buf + ((b * num_heads + h) * sq * sk) as u64 * elem_size;
            let v_head = v_base + ((b * num_kv_heads + kv) * kv_capacity * d) as u64 * elem_size;
            let o_head = o_base + ((b * num_heads + h) * sq * d) as u64 * elem_size;

            let p = GemmEx {
                dtype: dtype.gemm(),
                transa: false, // op(A=V) = V  -> [D, Sk] (col-major view)
                transb: false, // op(B=P) = P  -> [Sk, Sq] (col-major view)
                m: d,
                n: sq,
                k: sk,
                alpha: 1.0,
                beta: 0.0,
                a: v_head,
                lda: d,
                b: s_head,
                ldb: sk,
                c: o_head,
                ldc: d,
                epilogue: None,
            };
            // SAFETY: per-head pointers lie inside the validated dense V and
            // the softmaxed scores buffer and the dense output; `workspace`
            // is live; `o_head` aliases neither operand.
            unsafe { gemm_ex(blas, stream, &p, workspace, workspace_bytes) }?;
        }
    }

    runtime.synchronize()
}

impl Kernel for AttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // Fused execution needs no external scratch. A cuBLASLt/softmax route requires
        // the caller to query its requirement and supply a prepared workspace;
        // missing admission must never become an implicit raw allocation.
        self.run(inputs, outputs, None)
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<onnx_runtime_ep_api::WorkspacePreparation> {
        // Prepare-only planning (#747): report the exact governed scratch for the
        // cuBLASLt/softmax path so the executor reserves it against the device authority
        // before request admission. The fused path uses only shared memory and
        // needs no device scratch. On any shape/dtype that `run` would reject we
        // report NONE and let execution raise the precise validation error.
        let (Some(q), Some(k)) = (inputs.first(), inputs.get(1)) else {
            return Ok(onnx_runtime_ep_api::WorkspacePreparation::Ready(
                WorkspaceRequirement::NONE,
            ));
        };
        if q.shape.len() != 4 || k.shape.len() != 4 {
            return Ok(onnx_runtime_ep_api::WorkspacePreparation::Ready(
                WorkspaceRequirement::NONE,
            ));
        }
        let Ok(dtype) = AttentionDtype::from_onnx(q.dtype) else {
            return Ok(onnx_runtime_ep_api::WorkspacePreparation::Ready(
                WorkspaceRequirement::NONE,
            ));
        };
        let (batch, hq, sq, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
        let sk = k.shape[2];
        if hq != self.num_heads {
            return Ok(onnx_runtime_ep_api::WorkspacePreparation::Ready(
                WorkspaceRequirement::NONE,
            ));
        }
        if self.use_fused(q.dtype, sq, sk, d) {
            return Ok(onnx_runtime_ep_api::WorkspacePreparation::Ready(
                WorkspaceRequirement::NONE,
            ));
        }
        let layout = blas_softmax_scratch_layout(dtype, batch, self.num_heads, sq, sk)?;
        let bytes = u64::try_from(layout.bytes()).map_err(|_| {
            EpError::KernelFailed("cuda_ep Attention: workspace does not fit u64".into())
        })?;
        Ok(onnx_runtime_ep_api::WorkspacePreparation::Ready(
            WorkspaceRequirement {
                bytes,
                alignment: BLAS_SOFTMAX_SCRATCH_ALIGN,
                lifetime: WorkspaceLifetime::StepScoped,
            },
        ))
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        // Keep layout validation in the kernel so it can report the rejected
        // view and how to materialize it.
        true
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "requires a warmed fused-attention path; the cuBLASLt/softmax fallback synchronizes the CUDA stream",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_memory_api::MemoryRole;

    fn rt() -> Option<Arc<CudaRuntime>> {
        crate::test_support::maybe_runtime()
    }

    #[test]
    fn new_rejects_indivisible_gqa_groups() {
        let Some(runtime) = rt() else {
            eprintln!("skip: no CUDA GPU");
            return;
        };
        let e = AttentionKernel::new(runtime, false, 8, 3, None).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("multiple of"), "{msg}");
    }

    #[test]
    fn new_accepts_mha_and_gqa_and_mqa() {
        let Some(runtime) = rt() else {
            eprintln!("skip: no CUDA GPU");
            return;
        };
        // MHA (8/8), GQA (8/2), MQA (8/1) all divide cleanly.
        for kv in [8usize, 2, 1] {
            AttentionKernel::new(runtime.clone(), true, 8, kv, Some(0.5)).unwrap();
        }
    }

    #[test]
    fn blas_softmax_scratch_layout_matches_byte_formula() {
        // Regression guard (§736): planning and execution size the governed
        // cuBLASLt/softmax scratch through this exact helper, so pin its formula. The
        // scores blob is `batch·heads·sq·sk·elem` padded up to the cuBLASLt
        // alignment, followed by the fixed cuBLASLt workspace.
        let (batch, heads, sq, sk) = (2usize, 4usize, 128usize, 96usize);
        let layout =
            blas_softmax_scratch_layout(AttentionDtype::F16, batch, heads, sq, sk).unwrap();
        let elem = AttentionDtype::F16.element_size() as usize;
        let scores = batch * heads * sq * sk * elem;
        assert_eq!(layout.alignment(), BLAS_SOFTMAX_SCRATCH_ALIGN);
        let storage = [0u8; BLAS_SOFTMAX_SCRATCH_ALIGN];
        let base = storage.as_ptr() as usize;
        let aligned = base.next_multiple_of(BLAS_SOFTMAX_SCRATCH_ALIGN);
        // Binding only describes raw regions; it does not access the backing.
        let backing = WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(aligned as *mut _),
            layout.bytes(),
        );
        let [scores_view, blas_view] = layout.bind(Some(backing)).unwrap();
        assert_eq!(scores_view.bytes(), scores);
        assert_eq!(scores_view.ptr().0 as usize, aligned);
        let padded = scores.next_multiple_of(BLAS_SOFTMAX_SCRATCH_ALIGN);
        assert_eq!(blas_view.ptr().0 as usize - aligned, padded);
        assert_eq!(layout.bytes(), padded + WORKSPACE_BYTES);
        assert_eq!(blas_view.bytes(), WORKSPACE_BYTES);
    }

    #[test]
    fn blas_softmax_scratch_rejects_overflow_without_allocating() {
        for (batch, heads, sq, sk) in [
            (usize::MAX, 2, 1, 1),
            (1, 1, usize::MAX, 2),
            (1, 1, 1, isize::MAX as usize),
            (1, 1, 1, (isize::MAX as usize) / 4),
        ] {
            let error =
                blas_softmax_scratch_layout(AttentionDtype::F32, batch, heads, sq, sk).unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains("workspace") && message.contains("reduce the prepared shape"),
                "{message}"
            );
        }
    }

    #[test]
    fn blas_softmax_scratch_binding_requires_prepared_exclusive_capacity() {
        use onnx_runtime_ep_api::DevicePtrMut;
        let layout = blas_softmax_scratch_layout(AttentionDtype::F32, 1, 2, 3, 5).unwrap();
        for workspace in [
            None,
            Some(WorkspaceView::new(
                DevicePtrMut(std::ptr::null_mut()),
                layout.bytes(),
            )),
            Some(WorkspaceView::new(
                DevicePtrMut(256usize as *mut _),
                layout.bytes() - 1,
            )),
            Some(WorkspaceView::new(
                DevicePtrMut(257usize as *mut _),
                layout.bytes(),
            )),
            Some(WorkspaceView::new(
                DevicePtrMut((usize::MAX - 255) as *mut _),
                layout.bytes(),
            )),
        ] {
            let message = match BlasSoftmaxScratch::bind(&layout, workspace) {
                Ok(_) => panic!("invalid prepared workspace must be rejected"),
                Err(error) => error.to_string(),
            };
            assert!(
                message.contains("cuda_ep Attention") && message.contains("execute_with_workspace"),
                "{message}"
            );
        }
        // These are opaque device addresses: binding must never dereference them.
        let scratch = BlasSoftmaxScratch::bind(
            &layout,
            Some(WorkspaceView::new(
                DevicePtrMut(256usize as *mut _),
                layout.bytes(),
            )),
        )
        .unwrap();
        assert_eq!(scratch.scores, 256);
        assert_eq!(scratch.workspace, 512);
        assert_eq!(scratch.workspace_bytes, WORKSPACE_BYTES);
    }

    #[test]
    fn blas_softmax_borrowed_scores_reject_larger_actual_geometry_without_device_access() {
        let layout = blas_softmax_scratch_layout(AttentionDtype::F32, 2, 3, 5, 7).unwrap();
        let backing = WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(0x10000usize as *mut _),
            layout.bytes(),
        );
        let scratch = BlasSoftmaxScratch::bind(&layout, Some(backing)).unwrap();
        scratch
            .validate_geometry(AttentionDtype::F32, 2, 3, 5, 7)
            .unwrap();
        scratch
            .validate_geometry(AttentionDtype::F32, 2, 3, 5, 6)
            .unwrap();
        let error = scratch
            .validate_geometry(AttentionDtype::F32, 2, 3, 5, 8)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("borrowed score capacity")
                && error.contains("reprepare caller workspace"),
            "{error}"
        );
        assert!(
            scratch
                .validate_geometry(AttentionDtype::F32, usize::MAX, 3, 5, 7)
                .is_err()
        );
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable gpu-tests on a CUDA runner"
    )]
    #[test]
    fn workspace_requirement_matches_layout_for_blas_softmax_and_is_none_for_fused() {
        let Some(runtime) = rt() else {
            eprintln!("skip: no CUDA GPU");
            return;
        };
        // A long shape forces the cuBLASLt/softmax (cuBLASLt) path: its governed
        // requirement must equal the shared layout so the executor reserves
        // exactly what execution carves — no ungoverned raw allocation remains.
        let (batch, heads, sq, sk, d) = (1usize, 8usize, 512usize, 512usize, 64usize);
        let blas_softmax =
            AttentionKernel::new_blas_softmax(runtime.clone(), true, heads, heads, None)
                .expect("blas_softmax kernel");
        let q_shape = [batch, heads, sq, d];
        let kv_shape = [batch, heads, sk, d];
        let q = TensorMetadata::new(DataType::Float16, &q_shape, true);
        let k = TensorMetadata::new(DataType::Float16, &kv_shape, true);
        let v = TensorMetadata::new(DataType::Float16, &kv_shape, true);
        let req = blas_softmax
            .workspace_requirement(&[q, k, v], &[])
            .and_then(onnx_runtime_ep_api::WorkspacePreparation::require_ready)
            .expect("workspace requirement");
        let layout =
            blas_softmax_scratch_layout(AttentionDtype::F16, batch, heads, sq, sk).unwrap();
        assert_eq!(req.bytes, layout.bytes() as u64);
        assert_eq!(req.alignment, BLAS_SOFTMAX_SCRATCH_ALIGN);
        assert_eq!(req.lifetime, WorkspaceLifetime::StepScoped);
        assert!(matches!(
            req.role(),
            MemoryRole::Workspace { step_scoped: true }
        ));

        // The fused (shared-memory) path allocates no device scratch, so it must
        // report NONE — the executor reserves nothing for it.
        let fused =
            AttentionKernel::new_fused(runtime, true, heads, heads, None).expect("fused kernel");
        let q = TensorMetadata::new(DataType::Float16, &q_shape, true);
        let k = TensorMetadata::new(DataType::Float16, &kv_shape, true);
        let v = TensorMetadata::new(DataType::Float16, &kv_shape, true);
        let fused_req = fused
            .workspace_requirement(&[q, k, v], &[])
            .and_then(onnx_runtime_ep_api::WorkspacePreparation::require_ready)
            .expect("fused workspace requirement");
        assert_eq!(fused_req.bytes, 0, "fused path needs no governed scratch");
    }
}

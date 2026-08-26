//! CUDA implementation of `com.microsoft::MultiHeadAttention` (opset 1).
//!
//! This is a **thin adapter over the shared SDPA core**
//! ([`super::attention::run_attention_phase2a`]), exactly mirroring the CPU EP's
//! `multi_head_attention.rs` (the conformance oracle): it parses/validates the
//! separate-Q/K/V MHA inputs, normalizes the Whisper-style layouts, projects the
//! optional `bias`, concatenates any past-KV cache into dense `BNSH` device
//! buffers, folds `attention_bias` and `key_padding_mask` into a single additive
//! mask, then hands the `QKᵀ → scale → +bias → +mask → softmax → ·V` math to the
//! Phase-2a cuBLASLt → softmax → cuBLASLt engine. Every layout transform runs as
//! an NVRTC kernel; only the (small, integer) `key_padding_mask` round-trips
//! through the host, matching ORT's `PrepareMask`.
//!
//! ## Semantics (byte-parity target: ORT `ComputeAttentionProbs`)
//!
//! ```text
//! scores = scale · (Q · Kᵀ)              # scale defaults to 1/sqrt(qk_head_size)
//! scores = scores + attention_bias       # optional additive float bias
//! scores = scores + mask                 # key_padding_mask → mask_filter_value,
//!                                        # causal (unidirectional & S>1) → excluded
//! probs  = softmax(scores, axis=-1)
//! out    = probs · V
//! ```
//!
//! ## Supported layouts
//!
//! * `Q_K_V_BSNH`: query `(B, S, D)`, key `(B, L, D)`, value `(B, L, D_v)`.
//! * `Q_K_V_BSNH_BNSH_BNSH` (cross-attention): query `(B, S, D)`, key/value
//!   already `(B, num_heads, L, H)`. Key/value bias is assumed zero (ORT).
//! * Optional in-op KV cache: `past_key`/`past_value` `(B, num_heads, P, H)` →
//!   `present_key`/`present_value` `(B, num_heads, P+L, H)`.
//!
//! ## Optional inputs (ORT slot order, tested with [`TensorView::is_absent`])
//!
//! `query(0)`, `key(1)`, `value(2)`, `bias(3)`, `key_padding_mask(4)`,
//! `attention_bias(5)`, `past_key(6)`, `past_value(7)`. Slots 8/9
//! (`past_sequence_length`, `cache_indirection`, `DecoderMaskedMHA`) are
//! rejected.
//!
//! ## Declined (clean claim-time / execute-time errors, never a miscompute)
//!
//! * Packed-QKV (rank-5 query) / packed-KV, and DecoderMaskedMHA extras.
//! * `v_head_size != qk_head_size`: the shared Phase-2a core assumes a single
//!   head dimension for Q/K/V/O (measured from `run_attention_phase2a`, which
//!   sizes the P·V GEMM and the output stride with the Q/K head dim `d`).
//! * `bias` / `attention_bias` / `past_*` whose dtype differs from the query
//!   dtype (the device transforms read them at the query element width).
//!
//! ## Precision note (measured)
//!
//! A *fully* key-padded row (`key_padding_mask` masking every key) adds the
//! `mask_filter_value` (`-10000`) constant to every logit; under softmax the
//! constant cancels, so ORT/the CPU oracle resolve it to `softmax(raw scores)`.
//! The shared Phase-2a softmax writes each masked logit (`raw − 10000`) back to
//! the score buffer between its two passes; at f16/bf16 the mantissa step near
//! 10000 is 8, so every value rounds to exactly `-10000` and the row collapses
//! to a uniform distribution. This matches the CPU oracle at f32 but not at
//! reduced precision — an inherent property of the shared core (which GQA and
//! `Attention` also use), not a defect of this adapter.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node, Shape};

use super::attention::{AttentionDtype, run_attention_phase2a};
use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

/// Threads per block for the memory-mover NVRTC kernels.
const BLOCK: u32 = 256;

const MODULE: &str = "multi_head_attention_v1";

fn mha_error(detail: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep MultiHeadAttention: {}", detail.into()))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize> {
    left.checked_add(right).ok_or_else(|| {
        mha_error(format!(
            "{name} addition {left} + {right} exceeds usize limits"
        ))
    })
}

fn checked_product(factors: &[usize], name: &str) -> Result<usize> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| mha_error(format!("{name} product {factors:?} exceeds usize limits")))
    })
}

fn checked_bytes(elements: usize, element_size: usize, name: &str) -> Result<usize> {
    let bytes = elements.checked_mul(element_size).ok_or_else(|| {
        mha_error(format!(
            "{name} byte size {elements} * {element_size} exceeds usize limits"
        ))
    })?;
    if bytes > isize::MAX as usize {
        return Err(mha_error(format!(
            "{name} byte size {bytes} exceeds isize::MAX"
        )));
    }
    Ok(bytes)
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| mha_error(format!("{name}={value} exceeds CUDA i32 limits")))
}

fn checked_grid(count: usize, name: &str) -> Result<u32> {
    let blocks = count.div_ceil(BLOCK as usize);
    u32::try_from(blocks).map_err(|_| {
        mha_error(format!(
            "{name} grid blocks={blocks} for element count {count} exceeds CUDA u32 limits"
        ))
    })
}

fn require_positive(value: usize, name: &str) -> Result<()> {
    if value == 0 {
        return Err(mha_error(format!("{name} must be > 0, got 0")));
    }
    Ok(())
}

fn checked_pointer_offset(base: CUdeviceptr, offset: usize, name: &str) -> Result<CUdeviceptr> {
    let offset = u64::try_from(offset).map_err(|_| {
        mha_error(format!(
            "{name} byte offset {offset} exceeds CUDA u64 limits"
        ))
    })?;
    base.checked_add(offset)
        .ok_or_else(|| mha_error(format!("{name} device pointer offset overflows u64")))
}

fn checked_index2(row: usize, column: usize, width: usize, name: &str) -> Result<usize> {
    checked_add(
        checked_product(&[row, width], &format!("{name} row offset"))?,
        column,
        name,
    )
}

fn checked_index3(
    plane: usize,
    row: usize,
    column: usize,
    rows: usize,
    columns: usize,
    name: &str,
) -> Result<usize> {
    let plane_row = checked_add(
        checked_product(&[plane, rows], &format!("{name} plane offset"))?,
        row,
        &format!("{name} row"),
    )?;
    checked_index2(plane_row, column, columns, name)
}

fn validate_static_shape(name: &str, shape: &Shape, dtype: DataType) -> Result<()> {
    if shape.is_empty() {
        return Ok(());
    }
    let mut known_product = 1usize;
    let mut fully_static = true;
    for (axis, dim) in shape.iter().enumerate() {
        match dim.as_static() {
            Some(value) => {
                require_positive(value, &format!("{name} dim {axis}"))?;
                checked_i32(value, &format!("{name} dim {axis}"))?;
                known_product = checked_product(
                    &[known_product, value],
                    &format!("{name} known-dimension element count"),
                )?;
            }
            None => fully_static = false,
        }
    }
    if fully_static {
        if dtype != DataType::Undefined {
            checked_bytes(known_product, dtype.byte_size(), name)?;
        }
        checked_grid(known_product, &format!("{name} element traversal"))?;
    }
    Ok(())
}

fn synchronize_runtime(runtime: &CudaRuntime) -> Result<()> {
    runtime.synchronize()
}

trait MhaScratchRuntime {
    fn allocate_scratch(&self, bytes: usize) -> Result<CUdeviceptr>;

    /// # Safety
    /// `ptr` must be a live allocation returned by `allocate_scratch` on this runtime.
    unsafe fn free_scratch(&self, ptr: CUdeviceptr) -> Result<()>;

    /// # Safety
    /// `dst` must cover `src.len()` bytes.
    unsafe fn upload_scratch(&self, src: &[u8], dst: CUdeviceptr) -> Result<()>;

    fn synchronize_scratch_stream(&self) -> Result<()>;
}

impl MhaScratchRuntime for CudaRuntime {
    fn allocate_scratch(&self, bytes: usize) -> Result<CUdeviceptr> {
        self.alloc_raw(bytes)
    }

    unsafe fn free_scratch(&self, ptr: CUdeviceptr) -> Result<()> {
        // SAFETY: forwarded from the trait contract.
        unsafe { self.free_raw(ptr) }
    }

    unsafe fn upload_scratch(&self, src: &[u8], dst: CUdeviceptr) -> Result<()> {
        // SAFETY: forwarded from the trait contract.
        unsafe { self.htod(src, dst) }
    }

    fn synchronize_scratch_stream(&self) -> Result<()> {
        synchronize_runtime(self)
    }
}

/// Owns every per-call MHA allocation until all stream users have completed.
struct MhaScratch<'a, R: MhaScratchRuntime + ?Sized> {
    runtime: &'a R,
    allocations: Vec<CUdeviceptr>,
    stream_may_use_allocations: bool,
}

impl<'a, R: MhaScratchRuntime + ?Sized> MhaScratch<'a, R> {
    fn new(runtime: &'a R) -> Self {
        Self {
            runtime,
            allocations: Vec::new(),
            stream_may_use_allocations: false,
        }
    }

    fn allocate(&mut self, bytes: usize) -> Result<CUdeviceptr> {
        let ptr = self.runtime.allocate_scratch(bytes.max(1))?;
        self.allocations.push(ptr);
        Ok(ptr)
    }

    fn upload(&self, src: &[u8], dst: CUdeviceptr) -> Result<()> {
        // SAFETY: callers only pass a pointer owned by this object and sized
        // from the same checked byte count as `src`.
        unsafe { self.runtime.upload_scratch(src, dst) }
    }

    fn mark_stream_use(&mut self) {
        self.stream_may_use_allocations = true;
    }

    fn finish_stream_use(&mut self) -> Result<()> {
        if self.stream_may_use_allocations {
            self.runtime.synchronize_scratch_stream()?;
            self.stream_may_use_allocations = false;
        }
        Ok(())
    }
}

impl<R: MhaScratchRuntime + ?Sized> Drop for MhaScratch<'_, R> {
    fn drop(&mut self) {
        if self.stream_may_use_allocations {
            let _ = self.runtime.synchronize_scratch_stream();
        }
        for ptr in self.allocations.drain(..).rev() {
            // SAFETY: allocation ownership is recorded only after a successful
            // allocation, drained once here, and never copied out as ownership.
            let _ = unsafe { self.runtime.free_scratch(ptr) };
        }
    }
}

/// NVRTC source: three memory-mover families, templated per storage dtype.
///
/// * `mha_build_bnsh_*` builds a dense `[B, N, past+cur, dim]` buffer from a BSH
///   or BNSH current tensor, an optional past cache prepended along the sequence
///   axis, and an optional per-`(head,dim)` bias — the Q transpose, the K/V
///   transpose, and the KV-cache concat in one pass. Accumulation is `f32`.
/// * `mha_transpose_out_*` writes the `[B, N, S, dim]` attention context back to
///   the operator's `[B, S, N·dim]` BSH output layout.
/// * `mha_build_mask_*` folds the optional `attention_bias` (broadcast over
///   `(B|1, N|1, S, T)`) and the host-resolved additive `key_padding_mask`
///   (`[B, S, T]` f32) into the `[B, N, S, T]` additive mask the softmax reads.
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define MHA_BUILD_BNSH(NAME, T, LOAD, STORE)                                    \
extern "C" __global__ void NAME(                                               \
    const T* __restrict__ cur, int cur_is_bnsh,                                \
    const T* __restrict__ past, int has_past,                                  \
    const T* __restrict__ bias, int has_bias,                                  \
    T* __restrict__ dst,                                                       \
    int batch, int heads, int cur_seq, int past_seq, int dim) {                \
  const long long total = (long long)past_seq + cur_seq;                       \
  const long long count = (long long)batch * heads * total * dim;              \
  for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;        \
       idx < count; idx += (long long)gridDim.x * blockDim.x) {                 \
    long long x = idx;                                                         \
    const int d = (int)(x % dim); x /= dim;                                     \
    const long long s = x % total; x /= total;                                 \
    const int h = (int)(x % heads); const int b = (int)(x / heads);            \
    float v;                                                                    \
    if (s < past_seq) {                                                        \
      v = LOAD(past[(((long long)(b * heads + h)) * past_seq + s) * dim + d]);  \
    } else {                                                                    \
      const long long sc = s - past_seq;                                       \
      if (cur_is_bnsh) {                                                       \
        v = LOAD(cur[(((long long)(b * heads + h)) * cur_seq + sc) * dim + d]); \
      } else {                                                                  \
        v = LOAD(cur[(((long long)(b * cur_seq + sc)) * heads + h) * dim + d]); \
      }                                                                         \
      if (has_bias) v += LOAD(bias[h * dim + d]);                               \
    }                                                                           \
    dst[idx] = STORE(v);                                                        \
  }                                                                             \
}

#define MHA_TRANSPOSE_OUT(NAME, T)                                              \
extern "C" __global__ void NAME(                                               \
    const T* __restrict__ src, T* __restrict__ dst,                            \
    int batch, int heads, int seq, int dim) {                                  \
  const long long count = (long long)batch * heads * seq * dim;                \
  for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;        \
       idx < count; idx += (long long)gridDim.x * blockDim.x) {                 \
    long long x = idx;                                                         \
    const int d = (int)(x % dim); x /= dim;                                     \
    const int s = (int)(x % seq); x /= seq;                                     \
    const int h = (int)(x % heads); const int b = (int)(x / heads);            \
    dst[(((long long)(b * seq + s)) * heads + h) * dim + d] = src[idx];         \
  }                                                                             \
}

#define MHA_BUILD_MASK(NAME, T, LOAD, STORE)                                    \
extern "C" __global__ void NAME(                                               \
    T* __restrict__ out,                                                       \
    const T* __restrict__ abias, int has_abias, int abias_d0, int abias_d1,    \
    const float* __restrict__ pad, int has_pad,                                \
    int batch, int heads, int sq, int total) {                                 \
  const long long count = (long long)batch * heads * sq * total;               \
  for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;        \
       idx < count; idx += (long long)gridDim.x * blockDim.x) {                 \
    long long x = idx;                                                         \
    const int j = (int)(x % total); x /= total;                                \
    const int i = (int)(x % sq); x /= sq;                                       \
    const int h = (int)(x % heads); const int b = (int)(x / heads);            \
    float v = 0.0f;                                                            \
    if (has_abias) {                                                           \
      const int b0 = (abias_d0 == 1) ? 0 : b;                                  \
      const int b1 = (abias_d1 == 1) ? 0 : h;                                  \
      v += LOAD(abias[(((long long)(b0 * abias_d1 + b1)) * sq + i) * total + j]);\
    }                                                                           \
    if (has_pad) v += pad[((long long)(b * sq + i)) * total + j];               \
    out[idx] = STORE(v);                                                        \
  }                                                                             \
}

#define MHA_ID(v) (v)
MHA_BUILD_BNSH(mha_build_bnsh_f32, float, MHA_ID, MHA_ID)
MHA_BUILD_BNSH(mha_build_bnsh_f16, __half, __half2float, __float2half_rn)
MHA_BUILD_BNSH(mha_build_bnsh_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16_rn)
MHA_TRANSPOSE_OUT(mha_transpose_out_f32, float)
MHA_TRANSPOSE_OUT(mha_transpose_out_f16, __half)
MHA_TRANSPOSE_OUT(mha_transpose_out_bf16, __nv_bfloat16)
MHA_BUILD_MASK(mha_build_mask_f32, float, MHA_ID, MHA_ID)
MHA_BUILD_MASK(mha_build_mask_f16, __half, __half2float, __float2half_rn)
MHA_BUILD_MASK(mha_build_mask_bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16_rn)
"#;

/// Per-dtype NVRTC entry-point stems.
fn stems(dtype: DataType) -> (&'static str, &'static str, &'static str) {
    match dtype {
        DataType::Float32 => (
            "mha_build_bnsh_f32",
            "mha_transpose_out_f32",
            "mha_build_mask_f32",
        ),
        DataType::Float16 => (
            "mha_build_bnsh_f16",
            "mha_transpose_out_f16",
            "mha_build_mask_f16",
        ),
        DataType::BFloat16 => (
            "mha_build_bnsh_bf16",
            "mha_transpose_out_bf16",
            "mha_build_mask_bf16",
        ),
        _ => ("", "", ""),
    }
}

/// `com.microsoft::MultiHeadAttention` kernel carrying the resolved attributes.
#[derive(Debug)]
pub struct MultiHeadAttentionKernel {
    runtime: Arc<CudaRuntime>,
    num_heads: usize,
    /// Explicit score scale; `None` → default `1/sqrt(qk_head_size)`.
    scale: Option<f32>,
    /// Additive fill for padding-masked positions (ORT default `-10000`).
    mask_filter_value: f32,
    /// Apply a causal (lower-triangular) mask when the query length is `> 1`.
    unidirectional: bool,
}

/// Factory for [`MultiHeadAttentionKernel`], reading the contrib-op attributes.
pub struct MultiHeadAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

/// Resolved attributes shared by the factory and the claim-time gate.
struct MhaAttributes {
    num_heads: usize,
    scale: Option<f32>,
    mask_filter_value: f32,
    unidirectional: bool,
}

fn attributes_from_node(node: &Node) -> Result<MhaAttributes> {
    let num_heads = node
        .attr("num_heads")
        .and_then(|a| a.as_int())
        .ok_or_else(|| {
            EpError::KernelFailed(
                "cuda_ep MultiHeadAttention: missing required `num_heads` attribute".into(),
            )
        })?;
    if num_heads <= 0 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep MultiHeadAttention: num_heads must be > 0, got {num_heads}"
        )));
    }
    // `> 0` is deliberately broader than ORT's `== 0` sentinel: a zero or
    // negative scale is meaningless and both kernels read it as "use the
    // 1/sqrt(head_size) default" rather than literally zeroing every score.
    let scale = node
        .attr("scale")
        .and_then(|a| a.as_float())
        .filter(|s| *s > 0.0);
    let mask_filter_value = node
        .attr("mask_filter_value")
        .and_then(|a| a.as_float())
        .unwrap_or(-10000.0);
    let unidirectional = node
        .attr("unidirectional")
        .and_then(|a| a.as_int())
        .unwrap_or(0)
        == 1;
    let num_heads = usize::try_from(num_heads).map_err(|_| {
        mha_error(format!(
            "num_heads={num_heads} cannot be represented as usize"
        ))
    })?;
    checked_i32(num_heads, "num_heads")?;
    Ok(MhaAttributes {
        num_heads,
        scale,
        mask_filter_value,
        unidirectional,
    })
}

/// Claim-time capability gate. Declines nodes this EP cannot execute *before*
/// ORT commits them to a fused partition, so a rejection is never raised from
/// `execute` (a hard session failure with no fallback). Only conditions
/// derivable from attributes and the (possibly partial) claim shapes/dtypes are
/// checked; symbolic dims are conservatively accepted and re-validated in
/// `execute`.
pub(crate) fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<String> {
    let attrs = match attributes_from_node(node) {
        Ok(attrs) => attrs,
        Err(e) => return Some(e.to_string()),
    };
    let num_heads = attrs.num_heads;

    let float_ok = |dt: DataType| {
        matches!(
            dt,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        )
    };
    let dtype_at = |i: usize| input_dtypes.get(i).copied().unwrap_or(DataType::Undefined);
    let present = |i: usize| {
        // A supplied optional input has a known (non-Undefined) dtype; ORT hands
        // omitted trailing slots as absent placeholders we never see here.
        dtype_at(i) != DataType::Undefined
    };
    let shape_at = |i: usize| shapes.get(i).map(Vec::as_slice).unwrap_or(&[][..]);

    for (slot, name) in [
        (0usize, "query"),
        (1, "key"),
        (2, "value"),
        (3, "bias"),
        (4, "key_padding_mask"),
        (5, "attention_bias"),
        (6, "past_key"),
        (7, "past_value"),
    ] {
        if let Some(shape) = shapes.get(slot)
            && let Err(error) = validate_static_shape(name, shape, dtype_at(slot))
        {
            return Some(error.to_string());
        }
    }

    let q_dtype = dtype_at(0);
    if q_dtype != DataType::Undefined && !float_ok(q_dtype) {
        return Some(format!(
            "cuda_ep MultiHeadAttention: query dtype {q_dtype:?} not supported (expected f32/f16/bf16)"
        ));
    }
    // Q/K/V and every float side input share the query element width.
    for (slot, name) in [
        (1usize, "key"),
        (2, "value"),
        (3, "bias"),
        (5, "attention_bias"),
        (6, "past_key"),
        (7, "past_value"),
    ] {
        if present(slot) && q_dtype != DataType::Undefined && dtype_at(slot) != q_dtype {
            return Some(format!(
                "cuda_ep MultiHeadAttention: {name} dtype {:?} must match query dtype {q_dtype:?}",
                dtype_at(slot)
            ));
        }
    }
    if present(4) && !matches!(dtype_at(4), DataType::Int32 | DataType::Int64) {
        return Some(format!(
            "cuda_ep MultiHeadAttention: key_padding_mask dtype {:?} must be int32 or int64",
            dtype_at(4)
        ));
    }
    // DecoderMaskedMHA extras (slots 8/9) are out of scope.
    if present(8) || present(9) {
        return Some(
            "cuda_ep MultiHeadAttention: DecoderMaskedMultiHeadAttention inputs (past_sequence_length / cache_indirection) are not supported".into(),
        );
    }

    let q = shape_at(0);
    if !q.is_empty() && q.len() != 3 {
        return Some(format!(
            "cuda_ep MultiHeadAttention: query must be rank 3 (B, S, hidden); packed-QKV layouts (rank {}) are unsupported",
            q.len()
        ));
    }
    // Packed-KV supplies a rank-5 `key` and omits `value` entirely. Both halves
    // of that shape must be declined *here*: a rejection raised from `execute`
    // arrives after ORT has already compiled the node onto this EP, which is a
    // hard session failure that no fallback recovers from. The two checks below
    // mirror the execute-time guards so the pair cannot disagree.
    if present(1) && !present(2) {
        return Some(
            "cuda_ep MultiHeadAttention: value is required when key is present; packed-KV layouts are unsupported".into(),
        );
    }
    for (slot, name) in [(1usize, "key"), (2, "value")] {
        let rank = shape_at(slot).len();
        // Rank zero is the IR's current representation for "no static shape
        // recorded"; every non-zero unsupported rank is concrete and declined.
        if rank != 0 && !matches!(rank, 3 | 4) {
            return Some(format!(
                "cuda_ep MultiHeadAttention: {name} must be rank 3 (B, T, hidden) or rank 4 (B, N, T, H); rank {rank} is unsupported"
            ));
        }
    }
    // Static extent of `shape[axis]`, or `None` when symbolic / out of range.
    let dim =
        |shape: &[onnx_runtime_ir::Dim], axis: usize| shape.get(axis).and_then(|d| d.as_static());
    if let Some(q_hidden) = dim(q, 2)
        && !q_hidden.is_multiple_of(num_heads)
    {
        return Some(format!(
            "cuda_ep MultiHeadAttention: query hidden {q_hidden} not divisible by num_heads {num_heads}"
        ));
    }
    for (slot, name) in [(1usize, "key"), (2, "value")] {
        let shape = shape_at(slot);
        if shape.len() == 3
            && let Some(hidden) = dim(shape, 2)
            && !hidden.is_multiple_of(num_heads)
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: {name} hidden {hidden} not divisible by num_heads {num_heads}"
            ));
        }
        if shape.len() == 4
            && let Some(heads) = dim(shape, 1)
            && heads != num_heads
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: rank-4 {name} dim 1 ({heads}) must equal num_heads {num_heads}"
            ));
        }
    }
    // Prove the equal-head-size requirement only when every relevant dim is
    // concretely known; a symbolic dim is conservatively accepted here and
    // re-validated in `execute`.
    if let Some(q_hidden) = dim(q, 2)
        && num_heads != 0
        && q_hidden % num_heads == 0
    {
        let head_size = q_hidden / num_heads;
        let key = shape_at(1);
        let value = shape_at(2);
        let v_head_size = match value.len() {
            3 => dim(value, 2)
                .filter(|h| h % num_heads == 0)
                .map(|h| h / num_heads),
            4 => dim(value, 3),
            _ => None,
        };
        if let Some(v_head_size) = v_head_size
            && v_head_size != head_size
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: v_head_size {v_head_size} must equal qk_head_size {head_size} (the shared attention core assumes one head dimension)"
            ));
        }
        if key.len() == 4
            && let Some(k_head) = dim(key, 3)
            && k_head != head_size
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: key head_size {k_head} must equal query head_size {head_size}"
            ));
        }
        for (shape, name) in [(shape_at(6), "past_key"), (shape_at(7), "past_value")] {
            if shape.len() == 4
                && let Some(past_head) = dim(shape, 3)
                && past_head != head_size
            {
                return Some(format!(
                    "cuda_ep MultiHeadAttention: {name} head_size {past_head} must equal query head_size {head_size}"
                ));
            }
        }
    }
    let key = shape_at(1);
    let value = shape_at(2);
    let key_seq = match key.len() {
        3 => dim(key, 1),
        4 => dim(key, 2),
        _ => None,
    };
    let value_seq = match value.len() {
        3 => dim(value, 1),
        4 => dim(value, 2),
        _ => None,
    };
    if let (Some(key_seq), Some(value_seq)) = (key_seq, value_seq)
        && key_seq != value_seq
    {
        return Some(format!(
            "cuda_ep MultiHeadAttention: key seq {key_seq} != value seq {value_seq}"
        ));
    }
    if let Some(q_batch) = dim(q, 0) {
        for (slot, name) in [
            (1usize, "key"),
            (2, "value"),
            (6, "past_key"),
            (7, "past_value"),
        ] {
            let shape = shape_at(slot);
            if !shape.is_empty()
                && let Some(input_batch) = dim(shape, 0)
                && input_batch != q_batch
            {
                return Some(format!(
                    "cuda_ep MultiHeadAttention: {name} batch {input_batch} must equal query batch {q_batch}"
                ));
            }
        }
    }
    let past_key = shape_at(6);
    let past_value = shape_at(7);
    if !past_key.is_empty() != !past_value.is_empty() {
        return Some(
            "cuda_ep MultiHeadAttention: past_key and past_value must be provided together".into(),
        );
    }
    for (shape, name) in [(past_key, "past_key"), (past_value, "past_value")] {
        if !shape.is_empty() && shape.len() != 4 {
            return Some(format!(
                "cuda_ep MultiHeadAttention: {name} must be rank 4 (B, N, P, H), got rank {}",
                shape.len()
            ));
        }
        if shape.len() == 4
            && let Some(heads) = dim(shape, 1)
            && heads != num_heads
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: {name} dim 1 ({heads}) must equal num_heads {num_heads}"
            ));
        }
    }
    if let (Some(pk_seq), Some(pv_seq)) = (dim(past_key, 2), dim(past_value, 2))
        && pk_seq != pv_seq
    {
        return Some(format!(
            "cuda_ep MultiHeadAttention: past_key seq {pk_seq} != past_value seq {pv_seq}"
        ));
    }
    let past_seq = dim(past_key, 2).unwrap_or(0);
    let total_seq = match key_seq {
        Some(cur_seq) => match checked_add(past_seq, cur_seq, "past_seq + current key seq") {
            Ok(total) => {
                if let Err(error) = checked_i32(total, "total sequence length") {
                    return Some(error.to_string());
                }
                Some(total)
            }
            Err(error) => return Some(error.to_string()),
        },
        None => None,
    };
    let bias = shape_at(3);
    if !bias.is_empty() && bias.len() != 1 {
        return Some(format!(
            "cuda_ep MultiHeadAttention: bias must be rank 1, got rank {}",
            bias.len()
        ));
    }
    if let (Some(actual), Some(q_hidden)) = (dim(bias, 0), dim(q, 2)) {
        let expected = match checked_add(
            match checked_product(&[2, q_hidden], "bias Q/K element count") {
                Ok(value) => value,
                Err(error) => return Some(error.to_string()),
            },
            q_hidden,
            "bias Q/K/V element count",
        ) {
            Ok(value) => value,
            Err(error) => return Some(error.to_string()),
        };
        if actual != expected {
            return Some(format!(
                "cuda_ep MultiHeadAttention: bias length {actual} must equal 3*query hidden = {expected}"
            ));
        }
    }
    let attention_bias = shape_at(5);
    if !attention_bias.is_empty() && attention_bias.len() != 4 {
        return Some(format!(
            "cuda_ep MultiHeadAttention: attention_bias must be rank 4, got rank {}",
            attention_bias.len()
        ));
    }
    if attention_bias.len() == 4
        && let (Some(q_batch), Some(q_seq), Some(total_seq)) = (dim(q, 0), dim(q, 1), total_seq)
    {
        let d0 = dim(attention_bias, 0);
        let d1 = dim(attention_bias, 1);
        let d2 = dim(attention_bias, 2);
        let d3 = dim(attention_bias, 3);
        if d0.is_some_and(|value| value != 1 && value != q_batch)
            || d1.is_some_and(|value| value != 1 && value != num_heads)
            || d2.is_some_and(|value| value != q_seq)
            || d3.is_some_and(|value| value != total_seq)
        {
            return Some(format!(
                "cuda_ep MultiHeadAttention: attention_bias static shape {attention_bias:?} is incompatible with B={q_batch}, N={num_heads}, S={q_seq}, T={total_seq}"
            ));
        }
    }
    let padding_mask = shape_at(4);
    if !padding_mask.is_empty() && !matches!(padding_mask.len(), 1..=3) {
        return Some(format!(
            "cuda_ep MultiHeadAttention: key_padding_mask must be rank 1, 2, or 3, got rank {}",
            padding_mask.len()
        ));
    }
    if !padding_mask.is_empty()
        && let (Some(q_batch), Some(q_seq), Some(total_seq)) = (dim(q, 0), dim(q, 1), total_seq)
    {
        let matches = match padding_mask.len() {
            1 => dim(padding_mask, 0).is_none_or(|length| {
                let packed = checked_product(&[3, q_batch], "key_padding_mask 3B")
                    .and_then(|value| checked_add(value, 2, "key_padding_mask 3B+2"));
                length == q_batch || packed.is_ok_and(|packed| length == packed)
            }),
            2 => {
                dim(padding_mask, 0).is_none_or(|value| value == q_batch)
                    && dim(padding_mask, 1).is_none_or(|value| value == total_seq)
            }
            3 => {
                dim(padding_mask, 0).is_none_or(|value| value == q_batch)
                    && dim(padding_mask, 1).is_none_or(|value| value == q_seq)
                    && dim(padding_mask, 2).is_none_or(|value| value == total_seq)
            }
            _ => false,
        };
        if !matches {
            return Some(format!(
                "cuda_ep MultiHeadAttention: key_padding_mask static shape {padding_mask:?} is incompatible with B={q_batch}, S={q_seq}, T={total_seq}"
            ));
        }
    }
    None
}

impl KernelFactory for MultiHeadAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let attrs = attributes_from_node(node)?;
        Ok(Box::new(MultiHeadAttentionKernel {
            runtime: self.runtime.clone(),
            num_heads: attrs.num_heads,
            scale: attrs.scale,
            mask_filter_value: attrs.mask_filter_value,
            unidirectional: attrs.unidirectional,
        }))
    }
}

/// Launch one memory-mover NVRTC entry (`MODULE`/`SOURCE`) over `count` linear
/// elements. A macro rather than a method so the caller supplies the argument
/// ABI inline without naming cudarc's launch-builder type.
macro_rules! mha_launch_1d {
    ($self:expr, $entry:expr, $count:expr, $builder:ident, $args:block) => {{
        let count: usize = $count;
        if count != 0 {
            let function = $self.runtime.nvrtc_function(MODULE, SOURCE, $entry)?;
            let grid = checked_grid(count, $entry)?.max(1);
            let mut $builder = $self.runtime.stream().launch_builder(&function);
            $args
            // SAFETY: each caller supplies the exact argument ABI for `$entry`;
            // every buffer is a live contiguous device allocation sized for
            // `count`, and the kernels use no shared memory.
            unsafe {
                $builder.launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| driver_err(&format!("launch {}", $entry), e))?;
        }
    }};
}

impl MultiHeadAttentionKernel {
    /// Build a dense `[B, N, past+cur, dim]` device buffer (`build_bnsh` entry).
    #[allow(clippy::too_many_arguments)]
    fn build_bnsh(
        &self,
        entry: &str,
        cur_ptr: CUdeviceptr,
        cur_is_bnsh: bool,
        past_ptr: CUdeviceptr,
        has_past: bool,
        bias_ptr: CUdeviceptr,
        has_bias: bool,
        dst: CUdeviceptr,
        batch: usize,
        heads: usize,
        cur_seq: usize,
        past_seq: usize,
        dim: usize,
    ) -> Result<()> {
        let (batch_i, heads_i, cur_i, past_i, dim_i) = (
            checked_i32(batch, "batch")?,
            checked_i32(heads, "num_heads")?,
            checked_i32(cur_seq, "current sequence length")?,
            checked_i32(past_seq, "past sequence length")?,
            checked_i32(dim, "head_size")?,
        );
        let cur_is_bnsh_i = i32::from(cur_is_bnsh);
        let has_past_i = i32::from(has_past);
        let has_bias_i = i32::from(has_bias);
        let total_seq = checked_add(past_seq, cur_seq, "past_seq + current sequence length")?;
        let count = checked_product(&[batch, heads, total_seq, dim], "BNSH build element count")?;
        mha_launch_1d!(self, entry, count, builder, {
            builder
                .arg(&cur_ptr)
                .arg(&cur_is_bnsh_i)
                .arg(&past_ptr)
                .arg(&has_past_i)
                .arg(&bias_ptr)
                .arg(&has_bias_i)
                .arg(&dst)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&cur_i)
                .arg(&past_i)
                .arg(&dim_i);
        });
        Ok(())
    }
}

impl Kernel for MultiHeadAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() < 3 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: expected at least query/key/value, got {} inputs",
                inputs.len()
            )));
        }
        if outputs.is_empty() || outputs[0].is_absent() {
            return Err(EpError::KernelFailed(
                "cuda_ep MultiHeadAttention: missing output".into(),
            ));
        }
        let optional = |i: usize| inputs.get(i).filter(|t| !t.is_absent());
        if optional(8).is_some() || optional(9).is_some() {
            return Err(EpError::KernelFailed(
                "cuda_ep MultiHeadAttention: DecoderMaskedMultiHeadAttention inputs (past_sequence_length / cache_indirection) are not supported".into(),
            ));
        }

        let query = &inputs[0];
        let key = &inputs[1];
        let value = &inputs[2];
        let dtype = query.dtype;
        let attn_dtype = AttentionDtype::from_onnx(dtype)?;
        let elem = attn_dtype.element_size() as usize;
        if dtype != DataType::Float32 {
            self.runtime
                .require_nvrtc_half_headers("MultiHeadAttention")?;
        }

        if query.shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: query must be rank 3 (B, S, hidden); packed-QKV layouts (rank {}) are unsupported",
                query.shape.len()
            )));
        }
        if key.is_absent() || value.is_absent() {
            return Err(EpError::KernelFailed(
                "cuda_ep MultiHeadAttention: separate key and value inputs are required (packed QKV/KV is unsupported)".into(),
            ));
        }
        if key.shape.len() != value.shape.len() || !(key.shape.len() == 3 || key.shape.len() == 4) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: key and value must both be rank 3 (B, L, hidden) or rank 4 (B, N, L, head_size), got key rank {}, value rank {}",
                key.shape.len(),
                value.shape.len()
            )));
        }
        for (name, v) in [("query", query), ("key", key), ("value", value)] {
            if v.dtype != dtype {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: {name} dtype {:?} must match query dtype {dtype:?}",
                    v.dtype
                )));
            }
            if !v.is_contiguous() {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: {name} must be contiguous"
                )));
            }
        }

        let num_heads = self.num_heads;
        let batch = query.shape[0];
        let q_seq = query.shape[1];
        let q_hidden = query.shape[2];
        require_positive(batch, "query batch")?;
        require_positive(q_seq, "query sequence length")?;
        require_positive(q_hidden, "query hidden")?;
        checked_i32(batch, "batch")?;
        checked_i32(num_heads, "num_heads")?;
        checked_i32(q_seq, "query sequence length")?;
        checked_i32(q_hidden, "query hidden")?;
        checked_bytes(
            checked_product(query.shape, "query element count")?,
            attn_dtype.element_size() as usize,
            "query",
        )?;
        if !q_hidden.is_multiple_of(num_heads) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: query hidden {q_hidden} not divisible by num_heads {num_heads}"
            )));
        }
        let head_size = q_hidden / num_heads;
        require_positive(head_size, "query head_size")?;
        checked_i32(head_size, "query head_size")?;
        let is_cross_bnsh = key.shape.len() == 4;

        // Resolve the current K/V geometry and validate the equal-head-size
        // requirement the shared Phase-2a core imposes on Q/K/V/O.
        let resolve_kv = |v: &TensorView, name: &str| -> Result<(bool, usize, usize)> {
            if v.shape[0] != batch {
                return Err(mha_error(format!(
                    "{name} batch {} must equal query batch {batch}",
                    v.shape[0]
                )));
            }
            match v.shape.len() {
                3 => {
                    let hidden = v.shape[2];
                    require_positive(v.shape[1], &format!("{name} sequence length"))?;
                    require_positive(hidden, &format!("{name} hidden"))?;
                    if !hidden.is_multiple_of(num_heads) {
                        return Err(EpError::KernelFailed(format!(
                            "cuda_ep MultiHeadAttention: {name} hidden {hidden} not divisible by num_heads {num_heads}"
                        )));
                    }
                    Ok((false, v.shape[1], hidden / num_heads))
                }
                4 => {
                    if v.shape[1] != num_heads {
                        return Err(EpError::KernelFailed(format!(
                            "cuda_ep MultiHeadAttention: rank-4 {name} dim 1 ({}) must equal num_heads {num_heads}",
                            v.shape[1]
                        )));
                    }
                    require_positive(v.shape[2], &format!("{name} sequence length"))?;
                    require_positive(v.shape[3], &format!("{name} head_size"))?;
                    Ok((true, v.shape[2], v.shape[3]))
                }
                other => Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: {name} rank {other} unsupported"
                ))),
            }
        };
        let (k_is_bnsh, k_seq, k_dim) = resolve_kv(key, "key")?;
        let (v_is_bnsh, v_seq, v_head_size) = resolve_kv(value, "value")?;
        for (name, view) in [("key", key), ("value", value)] {
            for (axis, &value) in view.shape.iter().enumerate() {
                checked_i32(value, &format!("{name} dim {axis}"))?;
            }
        }
        if k_dim != head_size {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: key head_size {k_dim} != query head_size {head_size}"
            )));
        }
        if k_seq != v_seq {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: key seq {k_seq} != value seq {v_seq}"
            )));
        }
        if v_head_size != head_size {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: v_head_size {v_head_size} must equal qk_head_size {head_size} (the shared attention core assumes one head dimension)"
            )));
        }
        for (name, view) in [("key", key), ("value", value)] {
            checked_bytes(
                checked_product(view.shape, &format!("{name} element count"))?,
                attn_dtype.element_size() as usize,
                name,
            )?;
        }
        let v_hidden = checked_product(&[num_heads, v_head_size], "value hidden")?;
        checked_i32(v_hidden, "value hidden")?;
        let cur_seq = k_seq;

        // Optional bias `(D + D + D_v)` split into per-projection slices.
        let bias_in = optional(3);
        if let Some(bias) = bias_in {
            if bias.dtype != dtype {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: bias dtype {:?} must match query dtype {dtype:?}",
                    bias.dtype
                )));
            }
            if bias.shape.len() != 1 {
                return Err(mha_error(format!(
                    "bias must be rank 1, got rank {}",
                    bias.shape.len()
                )));
            }
            let expected = checked_add(
                checked_product(&[2, q_hidden], "bias Q/K element count")?,
                v_hidden,
                "bias Q/K/V element count",
            )?;
            let bias_elements = checked_product(bias.shape, "bias element count")?;
            checked_bytes(bias_elements, elem, "bias")?;
            if bias_elements != expected {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: bias length {bias_elements} must equal 2*hidden + v_hidden = {expected}"
                )));
            }
            if !bias.is_contiguous() {
                return Err(EpError::KernelFailed(
                    "cuda_ep MultiHeadAttention: bias must be contiguous".into(),
                ));
            }
        }
        let bias_base = bias_in.map(|b| cuptr(b.data_ptr::<u8>() as *const c_void));
        // In the rank-4 (BNSH) key/value layout ORT applies only the query bias.
        let (q_bias, k_bias, v_bias) = match bias_base {
            None => (0, 0, 0),
            Some(base) => {
                let q_b = base;
                if is_cross_bnsh {
                    (q_b, 0, 0)
                } else {
                    let k_offset = checked_bytes(q_hidden, elem, "key bias offset")?;
                    let v_offset = checked_bytes(
                        checked_product(&[2, q_hidden], "value bias offset elements")?,
                        elem,
                        "value bias offset",
                    )?;
                    let k_b = checked_pointer_offset(base, k_offset, "key bias")?;
                    let v_b = checked_pointer_offset(base, v_offset, "value bias")?;
                    (q_b, k_b, v_b)
                }
            }
        };

        // Optional in-op KV cache (inputs 6 and 7), rank-4 (B, N, P, H).
        let past_key = optional(6);
        let past_value = optional(7);
        if past_key.is_some() != past_value.is_some() {
            return Err(EpError::KernelFailed(
                "cuda_ep MultiHeadAttention: past_key and past_value must be provided together"
                    .into(),
            ));
        }
        let past_seq = match past_key {
            Some(pk) => {
                for (name, p) in [("past_key", pk), ("past_value", past_value.unwrap())] {
                    if p.dtype != dtype {
                        return Err(EpError::KernelFailed(format!(
                            "cuda_ep MultiHeadAttention: {name} dtype {:?} must match query dtype {dtype:?}",
                            p.dtype
                        )));
                    }
                    if p.shape.len() != 4 || p.shape[0] != batch || p.shape[1] != num_heads {
                        return Err(EpError::KernelFailed(format!(
                            "cuda_ep MultiHeadAttention: {name} must be rank-4 (B={batch}, N={num_heads}, P, H), got {:?}",
                            p.shape
                        )));
                    }
                    if !p.is_contiguous() {
                        return Err(EpError::KernelFailed(format!(
                            "cuda_ep MultiHeadAttention: {name} must be contiguous"
                        )));
                    }
                    for (axis, &value) in p.shape.iter().enumerate() {
                        require_positive(value, &format!("{name} dim {axis}"))?;
                        checked_i32(value, &format!("{name} dim {axis}"))?;
                    }
                    checked_bytes(
                        checked_product(p.shape, &format!("{name} element count"))?,
                        elem,
                        name,
                    )?;
                }
                if pk.shape[3] != head_size {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep MultiHeadAttention: past_key head_size {} != query head_size {head_size}",
                        pk.shape[3]
                    )));
                }
                if past_value.unwrap().shape[3] != v_head_size {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep MultiHeadAttention: past_value head_size {} != v_head_size {v_head_size}",
                        past_value.unwrap().shape[3]
                    )));
                }
                if past_value.unwrap().shape[2] != pk.shape[2] {
                    return Err(EpError::KernelFailed(
                        "cuda_ep MultiHeadAttention: past_key and past_value seq lengths differ"
                            .into(),
                    ));
                }
                pk.shape[2]
            }
            None => 0,
        };
        let total_seq = checked_add(past_seq, cur_seq, "past_seq + current key sequence")?;
        checked_i32(past_seq, "past sequence length")?;
        checked_i32(cur_seq, "current key sequence length")?;
        checked_i32(total_seq, "total sequence length")?;

        // present_key / present_value outputs share the built cache buffers, so
        // build them directly into the output allocations when requested.
        let want_present_k = outputs.len() >= 2 && !outputs[1].is_absent();
        let want_present_v = outputs.len() >= 3 && !outputs[2].is_absent();
        for (index, name) in [(0usize, "output"), (1, "present_key"), (2, "present_value")] {
            if let Some(output) = outputs.get(index)
                && !output.is_absent()
            {
                if output.dtype != dtype {
                    return Err(mha_error(format!(
                        "{name} dtype {:?} must match query dtype {dtype:?}",
                        output.dtype
                    )));
                }
                if !output.is_contiguous() {
                    return Err(mha_error(format!("{name} must be contiguous")));
                }
            }
        }
        if want_present_k {
            check_present(
                &outputs[1],
                batch,
                num_heads,
                total_seq,
                head_size,
                "present_key",
            )?;
        }
        if want_present_v {
            check_present(
                &outputs[2],
                batch,
                num_heads,
                total_seq,
                v_head_size,
                "present_value",
            )?;
        }
        if outputs[0].shape != [batch, q_seq, v_hidden] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: output shape {:?} must be [B={batch}, S={q_seq}, v_hidden={v_hidden}]",
                outputs[0].shape
            )));
        }
        let q_elements =
            checked_product(&[batch, num_heads, q_seq, head_size], "query BNSH elements")?;
        let present_k_elements = checked_product(
            &[batch, num_heads, total_seq, head_size],
            "present_key BNSH elements",
        )?;
        let present_v_elements = checked_product(
            &[batch, num_heads, total_seq, v_head_size],
            "present_value BNSH elements",
        )?;
        let output_elements = checked_product(
            &[batch, num_heads, q_seq, head_size],
            "output BNSH elements",
        )?;
        let mask_elements = checked_product(
            &[batch, num_heads, q_seq, total_seq],
            "attention mask elements",
        )?;
        let pad_elements = checked_product(&[batch, q_seq, total_seq], "padding mask elements")?;
        let mask_planes = checked_product(&[batch, num_heads], "attention mask planes")?;
        checked_i32(mask_planes, "attention mask planes")?;
        for (elements, name) in [
            (q_elements, "query BNSH scratch"),
            (present_k_elements, "present_key storage"),
            (present_v_elements, "present_value storage"),
            (output_elements, "output BNSH scratch"),
            (mask_elements, "attention mask scratch"),
        ] {
            checked_bytes(elements, elem, name)?;
            checked_grid(elements, &format!("{name} launch"))?;
        }
        checked_bytes(
            checked_product(outputs[0].shape, "output element count")?,
            elem,
            "output",
        )?;
        if want_present_k {
            checked_bytes(
                checked_product(outputs[1].shape, "present_key output element count")?,
                elem,
                "present_key output",
            )?;
        }
        if want_present_v {
            checked_bytes(
                checked_product(outputs[2].shape, "present_value output element count")?,
                elem,
                "present_value output",
            )?;
        }
        checked_grid(mask_elements, "attention mask launch")?;

        // Resolve key_padding_mask on the host (small, integer) exactly like
        // ORT's PrepareMask, into an additive [B, S, total] f32 buffer.
        let pad_additive = match optional(4) {
            Some(m) => Some(self.resolve_pad_mask(m, batch, q_seq, total_seq)?),
            None => None,
        };

        // Optional attention_bias (input 5): additive `(B|1, N|1, S, T)`.
        let attn_bias = optional(5);
        let (abias_d0, abias_d1) = if let Some(m) = attn_bias {
            if m.dtype != dtype {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: attention_bias dtype {:?} must match query dtype {dtype:?}",
                    m.dtype
                )));
            }
            if m.shape.len() != 4 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: attention_bias must be rank 4 (B|1, N|1, S, T), got rank {}",
                    m.shape.len()
                )));
            }
            let (d0, d1, d2, d3) = (m.shape[0], m.shape[1], m.shape[2], m.shape[3]);
            if !(d0 == batch || d0 == 1)
                || !(d1 == num_heads || d1 == 1)
                || d2 != q_seq
                || d3 != total_seq
            {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: attention_bias shape {:?} incompatible with (B|1={batch}, N|1={num_heads}, S={q_seq}, T={total_seq})",
                    m.shape
                )));
            }
            if !m.is_contiguous() {
                return Err(EpError::KernelFailed(
                    "cuda_ep MultiHeadAttention: attention_bias must be contiguous".into(),
                ));
            }
            checked_bytes(
                checked_product(m.shape, "attention_bias element count")?,
                elem,
                "attention_bias",
            )?;
            checked_i32(d0, "attention_bias dim 0")?;
            checked_i32(d1, "attention_bias dim 1")?;
            (d0, d1)
        } else {
            (0, 0)
        };

        let scale = self.scale.unwrap_or(1.0 / (head_size as f32).sqrt());
        // ORT: causal only when unidirectional AND the query spans > 1 token (an
        // incremental decode step attends the whole cache).
        let causal = self.unidirectional && q_seq > 1;

        let (build_entry, transpose_entry, mask_entry) = stems(dtype);
        let want_mask = pad_additive.is_some() || attn_bias.is_some();

        let q_bytes = checked_bytes(q_elements, elem, "query BNSH scratch")?;
        let present_k_bytes = checked_bytes(present_k_elements, elem, "present_key BNSH scratch")?;
        let present_v_bytes =
            checked_bytes(present_v_elements, elem, "present_value BNSH scratch")?;
        let output_bytes = checked_bytes(output_elements, elem, "output BNSH scratch")?;
        let mask_bytes = checked_bytes(mask_elements, elem, "attention mask scratch")?;
        let pad_bytes = checked_bytes(
            pad_elements,
            std::mem::size_of::<f32>(),
            "padding mask scratch",
        )?;

        let mut scratch = MhaScratch::new(self.runtime.as_ref());
        let q_bnsh = scratch.allocate(q_bytes)?;
        let present_k = if want_present_k {
            cuptr(outputs[1].data_ptr_mut::<u8>() as *const c_void)
        } else {
            scratch.allocate(present_k_bytes)?
        };
        let present_v = if want_present_v {
            cuptr(outputs[2].data_ptr_mut::<u8>() as *const c_void)
        } else {
            scratch.allocate(present_v_bytes)?
        };
        let o_bnsh = scratch.allocate(output_bytes)?;
        let mask_ptr = if want_mask {
            scratch.allocate(mask_bytes)?
        } else {
            0
        };
        let pad_ptr = match &pad_additive {
            Some(p) => {
                let ptr = scratch.allocate(pad_bytes)?;
                let bytes = bytemuck_f32_bytes(p);
                scratch.upload(&bytes, ptr)?;
                ptr
            }
            None => 0,
        };
        let abias_ptr = attn_bias
            .map(|m| cuptr(m.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);

        let q_base = cuptr(query.data_ptr::<u8>() as *const c_void);
        let k_base = cuptr(key.data_ptr::<u8>() as *const c_void);
        let v_base = cuptr(value.data_ptr::<u8>() as *const c_void);
        let past_k_base = past_key
            .map(|p| cuptr(p.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let past_v_base = past_value
            .map(|p| cuptr(p.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let o_out = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        self.build_bnsh(
            build_entry,
            q_base,
            false,
            0,
            false,
            q_bias,
            q_bias != 0,
            q_bnsh,
            batch,
            num_heads,
            q_seq,
            0,
            head_size,
        )?;
        scratch.mark_stream_use();
        self.build_bnsh(
            build_entry,
            k_base,
            k_is_bnsh,
            past_k_base,
            past_key.is_some(),
            k_bias,
            k_bias != 0,
            present_k,
            batch,
            num_heads,
            cur_seq,
            past_seq,
            head_size,
        )?;
        scratch.mark_stream_use();
        self.build_bnsh(
            build_entry,
            v_base,
            v_is_bnsh,
            past_v_base,
            past_value.is_some(),
            v_bias,
            v_bias != 0,
            present_v,
            batch,
            num_heads,
            cur_seq,
            past_seq,
            v_head_size,
        )?;
        scratch.mark_stream_use();

        let mask_planes_i = if want_mask {
            let (batch_i, heads_i, sq_i, total_i) = (
                checked_i32(batch, "batch")?,
                checked_i32(num_heads, "num_heads")?,
                checked_i32(q_seq, "query sequence length")?,
                checked_i32(total_seq, "total sequence length")?,
            );
            let has_abias_i = i32::from(attn_bias.is_some());
            let abias_d0_i = checked_i32(abias_d0, "attention_bias dim 0")?;
            let abias_d1_i = checked_i32(abias_d1, "attention_bias dim 1")?;
            let has_pad_i = i32::from(pad_additive.is_some());
            mha_launch_1d!(self, mask_entry, mask_elements, builder, {
                builder
                    .arg(&mask_ptr)
                    .arg(&abias_ptr)
                    .arg(&has_abias_i)
                    .arg(&abias_d0_i)
                    .arg(&abias_d1_i)
                    .arg(&pad_ptr)
                    .arg(&has_pad_i)
                    .arg(&batch_i)
                    .arg(&heads_i)
                    .arg(&sq_i)
                    .arg(&total_i);
            });
            scratch.mark_stream_use();
            checked_i32(mask_planes, "attention mask planes")?
        } else {
            0
        };

        // The shared core can fail after submitting an earlier phase, so arm
        // stream-ordered cleanup before entering it.
        scratch.mark_stream_use();
        run_attention_phase2a(
            &self.runtime,
            attn_dtype,
            num_heads,
            num_heads,
            causal,
            batch,
            q_seq,
            total_seq,
            head_size,
            total_seq,
            1,
            scale,
            q_bnsh,
            present_k,
            present_v,
            o_bnsh,
            mask_ptr,
            mask_planes_i,
            0,
            0,
            0,
            0.0,
            None,
        )?;

        let (batch_i, heads_i, sq_i, dim_i) = (
            checked_i32(batch, "batch")?,
            checked_i32(num_heads, "num_heads")?,
            checked_i32(q_seq, "query sequence length")?,
            checked_i32(head_size, "head_size")?,
        );
        mha_launch_1d!(self, transpose_entry, output_elements, builder, {
            builder
                .arg(&o_bnsh)
                .arg(&o_out)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&sq_i)
                .arg(&dim_i);
        });
        scratch.mark_stream_use();

        scratch.finish_stream_use()
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        // Contiguity is validated per input; a strided view returns an actionable
        // error rather than being silently mis-read.
        true
    }

    fn capture_support(&self) -> CaptureSupport {
        // The Phase-2a core performs an unconditional trailing synchronize and
        // this kernel allocates/frees per-call scratch — neither is capturable.
        CaptureSupport::unsupported(
            "cuda_ep MultiHeadAttention uses the per-call Phase-2a workspace path (allocates scratch and synchronizes)",
        )
    }
}

impl MultiHeadAttentionKernel {
    /// Resolve `key_padding_mask` into an additive `[B, S, total]` f32 buffer,
    /// matching ORT's `PrepareMask` (`0` keeps a key, `mask_filter_value` masks
    /// it). The mask input is small and integer, so it round-trips through the
    /// host exactly as the CPU oracle reads it.
    fn resolve_pad_mask(
        &self,
        view: &TensorView,
        batch: usize,
        q_seq: usize,
        total_seq: usize,
    ) -> Result<Vec<f32>> {
        checked_i32(batch, "padding mask batch")?;
        checked_i32(q_seq, "padding mask query sequence")?;
        checked_i32(total_seq, "padding mask total sequence")?;
        let raw = self.read_i64(view)?;
        let filter = self.mask_filter_value;
        let keep_or = |keep: bool| if keep { 0.0f32 } else { filter };
        let output_elements =
            checked_product(&[batch, q_seq, total_seq], "resolved padding mask elements")?;
        checked_bytes(
            output_elements,
            std::mem::size_of::<f32>(),
            "resolved padding mask",
        )?;
        let mut out = vec![0.0f32; output_elements];
        let dims = view.shape;
        let total_i64 = i64::from(checked_i32(total_seq, "padding mask total sequence")?);
        let index = |b: usize, i: usize, j: usize| {
            checked_index3(b, i, j, q_seq, total_seq, "resolved padding mask index")
        };
        match *dims {
            [b] if b == batch => {
                for b in 0..batch {
                    let end = raw[b].clamp(0, total_i64);
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            let key = i64::from(checked_i32(j, "padding mask key index")?);
                            out[index(b, i, j)?] = keep_or(key < end);
                        }
                    }
                }
            }
            [b] if b
                == checked_add(
                    checked_product(&[3, batch], "key_padding_mask 3B")?,
                    2,
                    "key_padding_mask 3B+2",
                )? =>
            {
                for b in 0..batch {
                    let end = raw[b].clamp(0, total_i64);
                    let start = raw[checked_add(batch, b, "key_padding_mask start index")?]
                        .clamp(0, total_i64);
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            let key = i64::from(checked_i32(j, "padding mask key index")?);
                            out[index(b, i, j)?] = keep_or(key < end && key >= start);
                        }
                    }
                }
            }
            [b, t] if b == batch && t == total_seq => {
                for b in 0..batch {
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            let raw_index =
                                checked_index2(b, j, total_seq, "2-D key_padding_mask index")?;
                            out[index(b, i, j)?] = keep_or(raw[raw_index] > 0);
                        }
                    }
                }
            }
            [b, s, t] if b == batch && s == q_seq && t == total_seq => {
                for b in 0..batch {
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            let raw_index = checked_index3(
                                b,
                                i,
                                j,
                                q_seq,
                                total_seq,
                                "3-D key_padding_mask index",
                            )?;
                            out[index(b, i, j)?] = keep_or(raw[raw_index] > 0);
                        }
                    }
                }
            }
            _ => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: unsupported key_padding_mask shape {dims:?} for batch={batch}, q_seq={q_seq}, total_seq={total_seq} (expected (B,), (3B+2), (B, T) or (B, S, T))"
                )));
            }
        }
        Ok(out)
    }

    /// Copy an int32/int64 device tensor to the host as `i64`.
    fn read_i64(&self, view: &TensorView) -> Result<Vec<i64>> {
        if !view.is_contiguous() {
            return Err(EpError::KernelFailed(
                "cuda_ep MultiHeadAttention: key_padding_mask must be contiguous".into(),
            ));
        }
        let n = checked_product(view.shape, "key_padding_mask element count")?;
        let src = cuptr(view.data_ptr::<u8>() as *const c_void);
        match view.dtype {
            DataType::Int64 => {
                let byte_count = checked_bytes(n, 8, "key_padding_mask int64")?;
                let mut bytes = vec![0u8; byte_count];
                // SAFETY: `src` is a live device tensor of `n` int64 elements.
                unsafe { self.runtime.dtoh(&mut bytes, src)? };
                Ok(bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                    .collect())
            }
            DataType::Int32 => {
                let byte_count = checked_bytes(n, 4, "key_padding_mask int32")?;
                let mut bytes = vec![0u8; byte_count];
                // SAFETY: `src` is a live device tensor of `n` int32 elements.
                unsafe { self.runtime.dtoh(&mut bytes, src)? };
                Ok(bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_ne_bytes(c.try_into().unwrap()) as i64)
                    .collect())
            }
            other => Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: key_padding_mask dtype {other:?} must be int32 or int64"
            ))),
        }
    }
}

/// Reinterpret an `f32` slice as its little-endian bytes for an H2D upload.
fn bytemuck_f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// Validate a `present_key`/`present_value` output shape.
fn check_present(
    out: &TensorMut,
    batch: usize,
    num_heads: usize,
    total_seq: usize,
    dim: usize,
    name: &str,
) -> Result<()> {
    if out.shape != [batch, num_heads, total_seq, dim] {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep MultiHeadAttention: {name} shape {:?} must be [B={batch}, N={num_heads}, total={total_seq}, H={dim}]",
            out.shape
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId, static_shape};
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeMap;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[derive(Default)]
    struct FaultScratchRuntime {
        fail_allocation_at: Cell<usize>,
        allocation_attempts: Cell<usize>,
        fail_upload: Cell<bool>,
        next_pointer: Cell<CUdeviceptr>,
        committed_bytes: Cell<usize>,
        live_sizes: RefCell<BTreeMap<CUdeviceptr, usize>>,
        free_counts: RefCell<BTreeMap<CUdeviceptr, usize>>,
        events: RefCell<Vec<&'static str>>,
    }

    impl FaultScratchRuntime {
        fn failing_allocation(attempt: usize) -> Self {
            Self {
                fail_allocation_at: Cell::new(attempt),
                next_pointer: Cell::new(1),
                ..Self::default()
            }
        }

        fn failing_upload() -> Self {
            Self {
                fail_upload: Cell::new(true),
                next_pointer: Cell::new(1),
                ..Self::default()
            }
        }

        fn assert_cleaned_once(&self, successful_allocations: usize) {
            assert_eq!(
                self.committed_bytes.get(),
                0,
                "committed scratch bytes must return to the baseline"
            );
            assert!(
                self.live_sizes.borrow().is_empty(),
                "no scratch allocation may remain live"
            );
            let frees = self.free_counts.borrow();
            assert_eq!(frees.len(), successful_allocations);
            assert!(
                frees.values().all(|&count| count == 1),
                "every allocation must be deallocated exactly once: {frees:?}"
            );
        }
    }

    impl MhaScratchRuntime for FaultScratchRuntime {
        fn allocate_scratch(&self, bytes: usize) -> Result<CUdeviceptr> {
            let attempt = self.allocation_attempts.get() + 1;
            self.allocation_attempts.set(attempt);
            if attempt == self.fail_allocation_at.get() {
                return Err(mha_error(format!(
                    "injected scratch allocation failure at attempt {attempt}"
                )));
            }
            let pointer = self.next_pointer.get();
            self.next_pointer.set(pointer + 1);
            self.live_sizes.borrow_mut().insert(pointer, bytes);
            self.committed_bytes.set(self.committed_bytes.get() + bytes);
            self.events.borrow_mut().push("allocate");
            Ok(pointer)
        }

        unsafe fn free_scratch(&self, ptr: CUdeviceptr) -> Result<()> {
            let bytes = self
                .live_sizes
                .borrow_mut()
                .remove(&ptr)
                .expect("free must name a live scratch allocation");
            self.committed_bytes.set(self.committed_bytes.get() - bytes);
            *self.free_counts.borrow_mut().entry(ptr).or_default() += 1;
            self.events.borrow_mut().push("free");
            Ok(())
        }

        unsafe fn upload_scratch(&self, _src: &[u8], _dst: CUdeviceptr) -> Result<()> {
            if self.fail_upload.get() {
                return Err(mha_error("injected H2D failure"));
            }
            self.events.borrow_mut().push("upload");
            Ok(())
        }

        fn synchronize_scratch_stream(&self) -> Result<()> {
            self.events.borrow_mut().push("synchronize");
            Ok(())
        }
    }

    fn node(attrs: &[(&str, Attribute)]) -> Node {
        let mut node = Node::new(NodeId(0), "MultiHeadAttention", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        for (name, value) in attrs {
            node.attributes.insert((*name).to_string(), value.clone());
        }
        node
    }

    fn shapes(dims: &[&[usize]]) -> Vec<Shape> {
        dims.iter()
            .map(|d| static_shape(d.iter().copied()))
            .collect()
    }

    #[test]
    fn claim_requires_positive_num_heads() {
        assert!(unsupported_reason(&node(&[]), &[], &[]).is_some());
        assert!(unsupported_reason(&node(&[("num_heads", Attribute::Int(0))]), &[], &[]).is_some());
        assert!(unsupported_reason(&node(&[("num_heads", Attribute::Int(2))]), &[], &[]).is_none());
    }

    #[test]
    fn claim_declines_packed_qkv_rank5_query() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let s = shapes(&[&[1, 2, 2, 3, 4], &[1, 2, 8], &[1, 2, 8]]);
        assert!(unsupported_reason(&n, &s, &[]).is_some());
    }

    // Packed-KV is a rank-5 `key` with `value` omitted. Both halves of that
    // shape must be refused at *claim* time: this EP has no packed-KV path, and
    // a rejection from `execute` lands after ORT has compiled the node here,
    // which fails the session outright instead of falling back. These two cases
    // were previously rejected only at execute time.
    #[test]
    fn claim_declines_packed_kv_rank5_key() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let s = shapes(&[&[1, 2, 8], &[1, 2, 2, 2, 4], &[]]);
        // Query f32, key f32, value absent — the packed-KV input signature.
        let dtypes = [DataType::Float32, DataType::Float32];
        let reason = unsupported_reason(&n, &s, &dtypes).expect("packed KV declined at claim time");
        assert!(
            reason.contains("packed-KV"),
            "expected a packed-KV decline, got: {reason}"
        );
    }

    #[test]
    fn claim_declines_key_of_unsupported_rank() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let s = shapes(&[&[1, 2, 8], &[1, 2, 2, 2, 4], &[1, 2, 8]]);
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Float32];
        let reason = unsupported_reason(&n, &s, &dtypes).expect("rank-5 key declined");
        assert!(
            reason.contains("key must be rank"),
            "expected a key-rank decline, got: {reason}"
        );
    }

    // The guard must not over-decline: rank 3 and rank 4 key/value are both
    // supported layouts, and a symbolic shape (recorded as empty here) is
    // accepted at claim time and re-validated in `execute`.
    #[test]
    fn claim_accepts_supported_key_value_ranks() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Float32];
        for s in [
            shapes(&[&[1, 2, 8], &[1, 2, 8], &[1, 2, 8]]),
            shapes(&[&[1, 2, 8], &[1, 2, 2, 4], &[1, 2, 2, 4]]),
            shapes(&[&[1, 2, 8], &[], &[]]),
        ] {
            assert!(
                unsupported_reason(&n, &s, &dtypes).is_none(),
                "supported layout wrongly declined: {s:?}"
            );
        }
    }

    #[test]
    fn claim_declines_mismatched_v_head_size() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        // q hidden 8 → head_size 4; value hidden 12 → v_head_size 6 ≠ 4.
        let s = shapes(&[&[1, 2, 8], &[1, 2, 8], &[1, 2, 12]]);
        let reason = unsupported_reason(&n, &s, &[]).expect("v_head_size mismatch declined");
        assert!(reason.contains("v_head_size"));
    }

    #[test]
    fn claim_declines_non_float_query() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let dtypes = [DataType::Int32, DataType::Int32, DataType::Int32];
        assert!(unsupported_reason(&n, &[], &dtypes).is_some());
    }

    #[test]
    fn claim_declines_dtype_mismatch_between_qkv() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let dtypes = [DataType::Float16, DataType::Float32, DataType::Float16];
        assert!(unsupported_reason(&n, &[], &dtypes).is_some());
    }

    #[test]
    fn claim_accepts_plain_self_attention() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let s = shapes(&[&[1, 4, 8], &[1, 4, 8], &[1, 4, 8]]);
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Float32];
        assert!(unsupported_reason(&n, &s, &dtypes).is_none());
    }

    #[test]
    fn scratch_raii_returns_committed_bytes_after_each_late_allocation_failure() {
        const ALLOCATIONS: usize = 6;
        for failed_attempt in 2..=ALLOCATIONS {
            let runtime = FaultScratchRuntime::failing_allocation(failed_attempt);
            let result = (|| -> Result<()> {
                let mut scratch = MhaScratch::new(&runtime);
                for _ in 0..ALLOCATIONS {
                    scratch.allocate(64)?;
                }
                Ok(())
            })();
            assert!(result.is_err(), "attempt {failed_attempt} must fail");
            runtime.assert_cleaned_once(failed_attempt - 1);
        }
    }

    #[test]
    fn scratch_raii_returns_committed_bytes_after_h2d_failure() {
        let runtime = FaultScratchRuntime::failing_upload();
        let result = (|| -> Result<()> {
            let mut scratch = MhaScratch::new(&runtime);
            for _ in 0..6 {
                scratch.allocate(64)?;
            }
            scratch.upload(&[0; 64], 6)
        })();
        assert!(result.is_err());
        runtime.assert_cleaned_once(6);
    }

    #[test]
    fn scratch_raii_is_panic_safe_and_orders_stream_completion_before_free() {
        let runtime = FaultScratchRuntime {
            next_pointer: Cell::new(1),
            ..FaultScratchRuntime::default()
        };
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let mut scratch = MhaScratch::new(&runtime);
            scratch.allocate(64).unwrap();
            scratch.mark_stream_use();
            panic!("injected panic after stream submission");
        }));
        assert!(panic.is_err());
        runtime.assert_cleaned_once(1);
        assert_eq!(
            runtime.events.borrow().as_slice(),
            ["allocate", "synchronize", "free"]
        );
    }

    #[test]
    fn checked_geometry_rejects_product_and_byte_overflow() {
        let product = checked_product(&[usize::MAX, 2], "query elements").unwrap_err();
        assert!(product.to_string().contains("query elements product"));
        let bytes = checked_bytes(usize::MAX, 2, "mask").unwrap_err();
        assert!(bytes.to_string().contains("mask byte size"));
    }

    #[test]
    fn checked_cuda_conversions_reject_i32_and_u32_overflow() {
        let i32_error = checked_i32(i32::MAX as usize + 1, "past sequence length").unwrap_err();
        assert!(
            i32_error
                .to_string()
                .contains("past sequence length=2147483648")
        );
        let too_many_blocks = (u32::MAX as usize + 1) * BLOCK as usize;
        let grid_error = checked_grid(too_many_blocks, "mask launch").unwrap_err();
        assert!(grid_error.to_string().contains("mask launch grid blocks"));
    }

    #[test]
    fn claim_rejects_statically_known_zero_and_cuda_i32_geometry() {
        let n = node(&[("num_heads", Attribute::Int(1))]);
        let zero = shapes(&[&[1, 0, 4], &[1, 1, 4], &[1, 1, 4]]);
        let zero_reason = unsupported_reason(&n, &zero, &[]).unwrap();
        assert!(zero_reason.contains("query dim 1 must be > 0"));

        let oversized = shapes(&[
            &[1, 1, i32::MAX as usize + 1],
            &[1, 1, i32::MAX as usize + 1],
            &[1, 1, i32::MAX as usize + 1],
        ]);
        let oversized_reason = unsupported_reason(&n, &oversized, &[]).unwrap();
        assert!(oversized_reason.contains("query dim 2=2147483648"));
    }

    #[test]
    fn claim_rejects_rank_one_qkv() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let q_reason = unsupported_reason(&n, &shapes(&[&[8]]), &[]).unwrap();
        assert!(q_reason.contains("query must be rank 3"));

        let kv = shapes(&[&[1, 2, 8], &[8], &[1, 2, 8]]);
        let kv_reason = unsupported_reason(&n, &kv, &[]).unwrap();
        assert!(kv_reason.contains("key must be rank 3"));
    }

    #[test]
    fn claim_rejects_statically_known_batch_past_and_mask_mismatches() {
        let n = node(&[("num_heads", Attribute::Int(2))]);
        let batch_mismatch = shapes(&[&[2, 1, 8], &[1, 1, 8], &[2, 1, 8]]);
        let reason = unsupported_reason(&n, &batch_mismatch, &[]).unwrap();
        assert!(reason.contains("key batch 1 must equal query batch 2"));

        let mask_mismatch = shapes(&[&[2, 1, 8], &[2, 1, 8], &[2, 1, 8], &[], &[2, 9]]);
        let reason = unsupported_reason(&n, &mask_mismatch, &[]).unwrap();
        assert!(reason.contains("key_padding_mask static shape"));

        let past_mismatch = shapes(&[
            &[2, 1, 8],
            &[2, 1, 8],
            &[2, 1, 8],
            &[],
            &[],
            &[],
            &[2, 2, 4, 4],
            &[2, 2, 5, 4],
        ]);
        let reason = unsupported_reason(&n, &past_mismatch, &[]).unwrap();
        assert!(reason.contains("past_key seq 4 != past_value seq 5"));
    }
}

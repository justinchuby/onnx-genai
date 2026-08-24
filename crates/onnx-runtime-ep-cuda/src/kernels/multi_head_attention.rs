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
    Ok(MhaAttributes {
        num_heads: num_heads as usize,
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
    if q.len() >= 2 && q.len() != 3 {
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
        // A rank of 0 or 1 here means "no static shape recorded", not a real
        // scalar/vector input; only a concretely known rank is judged.
        if present(slot) && rank >= 2 && !matches!(rank, 3 | 4) {
            return Some(format!(
                "cuda_ep MultiHeadAttention: {name} must be rank 3 (B, T, hidden) or rank 4 (B, N, T, H); rank {rank} is unsupported"
            ));
        }
    }
    // Static extent of `shape[axis]`, or `None` when symbolic / out of range.
    let dim =
        |shape: &[onnx_runtime_ir::Dim], axis: usize| shape.get(axis).and_then(|d| d.as_static());
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
            let grid = u32::try_from(count.div_ceil(BLOCK as usize))
                .unwrap_or(u32::MAX)
                .max(1);
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
            batch as i32,
            heads as i32,
            cur_seq as i32,
            past_seq as i32,
            dim as i32,
        );
        let cur_is_bnsh_i = i32::from(cur_is_bnsh);
        let has_past_i = i32::from(has_past);
        let has_bias_i = i32::from(has_bias);
        let count = batch * heads * (past_seq + cur_seq) * dim;
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
        if outputs.is_empty() {
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
        if num_heads == 0 || !q_hidden.is_multiple_of(num_heads) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MultiHeadAttention: query hidden {q_hidden} not divisible by num_heads {num_heads}"
            )));
        }
        let head_size = q_hidden / num_heads;
        let is_cross_bnsh = key.shape.len() == 4;

        // Resolve the current K/V geometry and validate the equal-head-size
        // requirement the shared Phase-2a core imposes on Q/K/V/O.
        let resolve_kv = |v: &TensorView, name: &str| -> Result<(bool, usize, usize)> {
            match v.shape.len() {
                3 => {
                    let hidden = v.shape[2];
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
                    Ok((true, v.shape[2], v.shape[3]))
                }
                other => Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: {name} rank {other} unsupported"
                ))),
            }
        };
        let (k_is_bnsh, k_seq, k_dim) = resolve_kv(key, "key")?;
        let (v_is_bnsh, v_seq, v_head_size) = resolve_kv(value, "value")?;
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
        let v_hidden = num_heads * v_head_size;
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
            let expected = 2 * q_hidden + v_hidden;
            if bias.numel() != expected {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MultiHeadAttention: bias length {} must equal 2*hidden + v_hidden = {expected}",
                    bias.numel()
                )));
            }
            if !bias.is_contiguous() {
                return Err(EpError::KernelFailed(
                    "cuda_ep MultiHeadAttention: bias must be contiguous".into(),
                ));
            }
        }
        let elem = attn_dtype.element_size() as usize;
        let bias_base = bias_in.map(|b| cuptr(b.data_ptr::<u8>() as *const c_void));
        // In the rank-4 (BNSH) key/value layout ORT applies only the query bias.
        let (q_bias, k_bias, v_bias) = match bias_base {
            None => (0, 0, 0),
            Some(base) => {
                let q_b = base;
                if is_cross_bnsh {
                    (q_b, 0, 0)
                } else {
                    let k_b = base + (q_hidden * elem) as u64;
                    let v_b = base + (2 * q_hidden * elem) as u64;
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
        let total_seq = past_seq + cur_seq;

        // present_key / present_value outputs share the built cache buffers, so
        // build them directly into the output allocations when requested.
        let want_present_k = outputs.len() >= 2 && !outputs[1].is_absent();
        let want_present_v = outputs.len() >= 3 && !outputs[2].is_absent();
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

        // Allocate device scratch; free every self-owned block before returning.
        let mut scratch: Vec<CUdeviceptr> = Vec::new();
        let mut alloc = |bytes: usize| -> Result<CUdeviceptr> {
            let ptr = self.runtime.alloc_raw(bytes.max(1))?;
            scratch.push(ptr);
            Ok(ptr)
        };

        let q_bnsh = alloc(batch * num_heads * q_seq * head_size * elem)?;
        let present_k = if want_present_k {
            cuptr(outputs[1].data_ptr_mut::<u8>() as *const c_void)
        } else {
            alloc(batch * num_heads * total_seq * head_size * elem)?
        };
        let present_v = if want_present_v {
            cuptr(outputs[2].data_ptr_mut::<u8>() as *const c_void)
        } else {
            alloc(batch * num_heads * total_seq * v_head_size * elem)?
        };
        let o_bnsh = alloc(batch * num_heads * q_seq * head_size * elem)?;
        let mask_ptr = if want_mask {
            alloc(batch * num_heads * q_seq * total_seq * elem)?
        } else {
            0
        };
        let pad_ptr = match &pad_additive {
            Some(p) => {
                let ptr = alloc(p.len() * std::mem::size_of::<f32>())?;
                let bytes = bytemuck_f32_bytes(p);
                // SAFETY: `ptr` was just allocated to hold exactly `bytes`.
                unsafe { self.runtime.htod(&bytes, ptr)? };
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

        let result = (|| -> Result<()> {
            // Q: BSH → BNSH (+ query bias, no past).
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
            // present_key = concat(past_key, key) into BNSH (+ key bias).
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
            // present_value = concat(past_value, value) into BNSH (+ value bias).
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

            let mask_planes = if want_mask {
                let (batch_i, heads_i, sq_i, total_i) = (
                    batch as i32,
                    num_heads as i32,
                    q_seq as i32,
                    total_seq as i32,
                );
                let has_abias_i = i32::from(attn_bias.is_some());
                let abias_d0_i = abias_d0 as i32;
                let abias_d1_i = abias_d1 as i32;
                let has_pad_i = i32::from(pad_additive.is_some());
                let count = batch * num_heads * q_seq * total_seq;
                mha_launch_1d!(self, mask_entry, count, builder, {
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
                (batch * num_heads) as i32
            } else {
                0
            };

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
                mask_planes,
                0,
                0,
                0,
                0.0,
                None,
            )?;

            // BNSH context → BSH output.
            let (batch_i, heads_i, sq_i, dim_i) = (
                batch as i32,
                num_heads as i32,
                q_seq as i32,
                head_size as i32,
            );
            mha_launch_1d!(
                self,
                transpose_entry,
                batch * num_heads * q_seq * head_size,
                builder,
                {
                    builder
                        .arg(&o_bnsh)
                        .arg(&o_out)
                        .arg(&batch_i)
                        .arg(&heads_i)
                        .arg(&sq_i)
                        .arg(&dim_i);
                }
            );

            // The trailing transpose is enqueued on the EP stream; drain it
            // before the scratch it reads is returned to the allocator pool.
            self.runtime.synchronize()
        })();

        for ptr in scratch {
            // SAFETY: every pooled block was allocated above by this runtime and
            // is freed exactly once here.
            let _ = unsafe { self.runtime.free_raw(ptr) };
        }
        result
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
        let raw = self.read_i64(view)?;
        let filter = self.mask_filter_value;
        let keep_or = |keep: bool| if keep { 0.0f32 } else { filter };
        let mut out = vec![0.0f32; batch * q_seq * total_seq];
        let dims = view.shape;
        let index = |b: usize, i: usize, j: usize| (b * q_seq + i) * total_seq + j;
        match *dims {
            [b] if b == batch => {
                for b in 0..batch {
                    let end = raw[b].clamp(0, total_seq as i64);
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            out[index(b, i, j)] = keep_or((j as i64) < end);
                        }
                    }
                }
            }
            [b] if b == 3 * batch + 2 => {
                for b in 0..batch {
                    let end = raw[b].clamp(0, total_seq as i64);
                    let start = raw[batch + b].clamp(0, total_seq as i64);
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            out[index(b, i, j)] = keep_or((j as i64) < end && (j as i64) >= start);
                        }
                    }
                }
            }
            [b, t] if b == batch && t == total_seq => {
                for b in 0..batch {
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            out[index(b, i, j)] = keep_or(raw[b * total_seq + j] > 0);
                        }
                    }
                }
            }
            [b, s, t] if b == batch && s == q_seq && t == total_seq => {
                for b in 0..batch {
                    for i in 0..q_seq {
                        for j in 0..total_seq {
                            out[index(b, i, j)] = keep_or(raw[(b * q_seq + i) * total_seq + j] > 0);
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
        let n = view.numel();
        let src = cuptr(view.data_ptr::<u8>() as *const c_void);
        match view.dtype {
            DataType::Int64 => {
                let mut bytes = vec![0u8; n * 8];
                // SAFETY: `src` is a live device tensor of `n` int64 elements.
                unsafe { self.runtime.dtoh(&mut bytes, src)? };
                Ok(bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                    .collect())
            }
            DataType::Int32 => {
                let mut bytes = vec![0u8; n * 4];
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
}

//! Standard `ai.onnx::Attention` (opset 23–26): scaled dot-product attention
//! (SDPA) with multi-head / grouped-query head sharing, an optional additive or
//! boolean attention mask, causal masking, and an in-op KV cache
//! (`past_key`/`past_value` → `present_key`/`present_value`).
//!
//! This is the *standard* ONNX operator, distinct from the private
//! `com.microsoft::FusedAttention` fusion node (see [`super::fused_attention`]),
//! which only reproduces the plain `MatMul → scale → [+mask] → Softmax →
//! MatMul` core the optimizer fuses. Standard `Attention` is a richer op: it
//! reshapes 3D `(batch, seq, hidden)` inputs into heads, supports GQA/MQA head
//! sharing, concatenates a past KV cache, offset-aware causal masking, softcap,
//! and emits up to four outputs (`Y`, `present_key`, `present_value`,
//! `qk_matmul_output`).
//!
//! ## Semantics (per the spec's applied pattern)
//!
//! ```text
//! scores = (Q·√scale) · (K·√scale)ᵀ      # √scale folded into each operand so
//!                                        # extreme magnitudes don't overflow;
//!                                        # scale defaults to 1/sqrt(head_size)
//! scores = softcap · tanh(scores/softcap)  # only when softcap != 0
//! scores = scores + attn_bias            # attn_mask (add/-inf) and causal mask
//! probs  = softmax(scores, axis=-1)      # numerically stable; fully-masked → 0
//! Y      = probs · V
//! ```
//!
//! ## GPU-native execution
//!
//! Unlike a host-staged reference that copies Q/K/V to the CPU, this kernel is
//! GPU-native: Q, K, V, the attention mask, and every bulk output stay resident
//! on the device. Two NVRTC kernels do all the heavy lifting:
//!
//! * `build_kv` gathers each K/V input — handling the 3D→4D head reshape and the
//!   `past ⧺ current` cache concatenation — into a contiguous
//!   `[batch, kv_heads, total_seq, dim]` present buffer (also the `present_key`/
//!   `present_value` outputs when requested).
//! * `attention_row` runs one CUDA block per `(batch, q_head, query)` row: it
//!   computes the scaled QK scores, softcap, the composed causal/pad/attn masks,
//!   a numerically-stable softmax, and the probs·V accumulation, writing `Y`
//!   (and the optional `qk_matmul_output`) directly to device output buffers.
//!
//! Only tiny host-side control state leaves/enters the device: the per-batch
//! causal `offset` and padding-frontier arrays are built on the host and
//! uploaded as small device arrays, and `nonpad_kv_seqlen` (a per-batch scalar
//! count) is read back to compute them. Q/K/V and the score/probability tensors
//! never round-trip through host memory.
//!
//! ## Determinism
//!
//! Each score row is reduced in a fixed order: the per-row `QK` dot products and
//! the `probs·V` accumulation each sum in ascending index order within a single
//! thread (bit-identical to the CPU reference), and the softmax max/exp/sum are
//! performed sequentially by the block's lead thread. No atomics contribute to a
//! shared accumulator, so results are byte-identical run to run.
//!
//! ## Versioning (opset 23 vs 24–26)
//!
//! `Attention` was added at opset 23 and revised at opset 24 (no newer version
//! exists, so a single opset-24 kernel serves model opsets 24, 25 and 26). The
//! one semantic delta handled per registered `since_version`:
//!
//! * `nonpad_kv_seqlen` (7th input) — an external-cache per-batch valid-token
//!   count — is honored for v24+ and rejected for v23 (it did not exist there).
//!
//! `qk_matmul_output_mode` has the **same** meaning in both versions (the opset
//! 23 and 24 schema descriptions are identical): `0` = raw QK, `1` = after
//! softcap (before mask), `2` = after mask+softcap, `3` = after softmax.
//!
//! ## Supported vs. unimplemented
//!
//! * dtype: **f32, f16, and bf16** Q/K/V, cache, additive mask, and outputs.
//!   All floating inputs must use the same dtype; a boolean attention mask is
//!   exempt. Device f16/bf16 loads and stores are converted around fp32 score,
//!   softmax, and value accumulators.
//! * `qk_matmul_output_mode`: modes **0, 1, 2, 3** implemented per spec; any
//!   other value errors.

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
/// Threads per block for `attention_row` (one block services one score row).
const ROW_THREADS: u32 = 128;
const ATTENTION_MODULE: &str = "standard_attention_f32_f16_bf16_v3";
const ATTENTION_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define NEG_INF __int_as_float(0xff800000)

// dtype is 0 for f32, 1 for f16, and 2 for bf16. Keep all computation in fp32;
// only the externally visible activations and cache use the requested storage
// type.
__device__ __forceinline__ float load_float(const void* data, unsigned long long index, int dtype) {
  if (dtype == 0) {
    return ((const float*)data)[index];
  }
  if (dtype == 1) {
    return __half2float(((const __half*)data)[index]);
  }
  return __bfloat162float(((const __nv_bfloat16*)data)[index]);
}

__device__ __forceinline__ void store_float(void* data, unsigned long long index, float value, int dtype) {
  if (dtype == 0) {
    ((float*)data)[index] = value;
  } else if (dtype == 1) {
    ((__half*)data)[index] = __float2half_rn(value);
  } else {
    ((__nv_bfloat16*)data)[index] = __float2bfloat16_rn(value);
  }
}

// Gather a K/V input into a contiguous [batch, heads, total_seq, dim] present
// buffer, applying the 3D->4D head reshape and the past ++ current concat.
extern "C" __global__ void build_kv(
    const void* past, const void* cur, void* out, int dtype,
    int has_past, int cur_is_3d, int past_is_3d,
    unsigned long long batch, unsigned long long heads,
    unsigned long long past_seq, unsigned long long cur_seq,
    unsigned long long total_seq, unsigned long long dim,
    unsigned long long elements) {
  for (unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements;
       idx += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long d = idx % dim;
    unsigned long long rem = idx / dim;
    unsigned long long t = rem % total_seq;
    rem /= total_seq;
    unsigned long long h = rem % heads;
    unsigned long long b = rem / heads;
    float val;
    if (has_past && t < past_seq) {
      unsigned long long off = past_is_3d
          ? (b * past_seq + t) * (heads * dim) + h * dim + d
          : ((b * heads + h) * past_seq + t) * dim + d;
      val = load_float(past, off, dtype);
    } else {
      unsigned long long c = has_past ? (t - past_seq) : t;
      unsigned long long off = cur_is_3d
          ? (b * cur_seq + c) * (heads * dim) + h * dim + d
          : ((b * heads + h) * cur_seq + c) * dim + d;
      val = load_float(cur, off, dtype);
    }
    store_float(out, idx, val, dtype);
  }
}

// Additive mask bias for logical index (b, h, i, j), broadcasting a rank<=4
// mask right-aligned against [b, h, i, j]. Mirrors the CPU reference exactly:
// a last dim shorter than total_seq pads with -inf; bool false -> -inf.
__device__ __forceinline__ float mask_bias(
    const void* mask, int mask_kind, int mask_rank,
    unsigned long long md0, unsigned long long md1,
    unsigned long long md2, unsigned long long md3,
    unsigned long long b, unsigned long long h,
    unsigned long long i, unsigned long long j,
    unsigned long long total_seq) {
  if (mask_kind == 0) {
    return 0.0f;
  }
  unsigned long long full[4] = {b, h, i, j};
  unsigned long long md[4] = {md0, md1, md2, md3};
  unsigned long long off = 0;
  for (int a = 0; a < 4; ++a) {
    unsigned long long idx = (md[a] == 1ULL) ? 0ULL : full[a];
    off = off * md[a] + idx;
  }
  if (mask_rank > 0) {
    unsigned long long last = md3;
    if (j >= last && last < total_seq) {
      return NEG_INF;
    }
  }
  if (mask_kind == 1) {
    return ((const float*)mask)[off];
  }
  if (mask_kind == 3) {
    return __half2float(((const __half*)mask)[off]);
  }
  if (mask_kind == 4) {
    return __bfloat162float(((const __nv_bfloat16*)mask)[off]);
  }
  // Bool mask: nonzero keeps (bias 0), zero masks (-inf).
  return ((const unsigned char*)mask)[off] != 0 ? 0.0f : NEG_INF;
}

// One block per (batch, q_head, query) row. Computes scaled QK scores, softcap,
// the composed causal/pad/attn masks, a stable softmax, and probs*V.
extern "C" __global__ void attention_row(
    const void* q, const void* key, const void* value,
    const void* mask, float* scores, void* y, void* qk_out,
    const long long* offsets, const long long* pad_limits,
    unsigned long long batch, unsigned long long q_heads, unsigned long long q_seq,
    unsigned long long kv_heads, unsigned long long total_seq,
    unsigned long long head_size, unsigned long long v_head_size,
    unsigned long long group,
    int dtype, int q_is_3d, int out_is_3d, int is_causal,
    float sqrt_scale, float softcap,
    int mask_kind, int mask_rank,
    unsigned long long md0, unsigned long long md1,
    unsigned long long md2, unsigned long long md3,
    int qk_mode, int want_qk) {
  const unsigned long long row = blockIdx.x;
  const unsigned long long total_rows = batch * q_heads * q_seq;
  if (row >= total_rows) {
    return;
  }
  const unsigned long long i = row % q_seq;
  unsigned long long rem = row / q_seq;
  const unsigned long long qh = rem % q_heads;
  const unsigned long long b = rem / q_heads;
  const unsigned long long kvh = qh / group;
  const unsigned long long srow = row * total_seq;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // Base offset of this query row's head vector.
  const unsigned long long qoff = q_is_3d
      ? (b * q_seq + i) * (q_heads * head_size) + qh * head_size
      : ((b * q_heads + qh) * q_seq + i) * head_size;

  // Stage 1: scaled Q·Kᵀ scores (sqrt(scale) folded into each operand).
  for (unsigned long long j = tid; j < total_seq; j += nthreads) {
    const unsigned long long koff = ((b * kv_heads + kvh) * total_seq + j) * head_size;
    float acc = 0.0f;
    for (unsigned long long p = 0; p < head_size; ++p) {
      acc += (load_float(q, qoff + p, dtype) * sqrt_scale)
          * (load_float(key, koff + p, dtype) * sqrt_scale);
    }
    scores[srow + j] = acc;
    if (want_qk && qk_mode == 0) {
      store_float(qk_out, srow + j, acc, dtype);
    }
  }
  __syncthreads();

  // Stage 2: softcap (before mask), applied when nonzero.
  if (softcap != 0.0f) {
    for (unsigned long long j = tid; j < total_seq; j += nthreads) {
      const float s = scores[srow + j];
      scores[srow + j] = softcap * tanhf(s / softcap);
    }
    __syncthreads();
  }
  if (want_qk && qk_mode == 1) {
    for (unsigned long long j = tid; j < total_seq; j += nthreads) {
      store_float(qk_out, srow + j, scores[srow + j], dtype);
    }
    __syncthreads();
  }

  // Stage 3: attention mask + causal frontier + padding frontier.
  const long long offset = offsets[b];
  const long long pad_limit = pad_limits[b];
  const long long causal_limit = (long long)i + offset;
  for (unsigned long long j = tid; j < total_seq; j += nthreads) {
    const long long jj = (long long)j;
    if (pad_limit >= 0 && jj >= pad_limit) {
      scores[srow + j] = NEG_INF;
      continue;
    }
    if (is_causal && jj > causal_limit) {
      scores[srow + j] = NEG_INF;
      continue;
    }
    scores[srow + j] += mask_bias(mask, mask_kind, mask_rank, md0, md1, md2, md3,
                                  b, qh, i, j, total_seq);
  }
  __syncthreads();
  if (want_qk && qk_mode == 2) {
    for (unsigned long long j = tid; j < total_seq; j += nthreads) {
      store_float(qk_out, srow + j, scores[srow + j], dtype);
    }
    __syncthreads();
  }

  // Stage 4: numerically-stable softmax. The lead thread performs the max,
  // exp, and sum in a fixed ascending order to match the CPU reference and be
  // reproducible; the final normalize is embarrassingly parallel.
  __shared__ float inv_sum_sh;
  __shared__ int all_masked_sh;
  if (tid == 0) {
    float m = NEG_INF;
    for (unsigned long long j = 0; j < total_seq; ++j) {
      m = fmaxf(m, scores[srow + j]);
    }
    if (m == NEG_INF) {
      all_masked_sh = 1;
      inv_sum_sh = 0.0f;
    } else {
      all_masked_sh = 0;
      float sum = 0.0f;
      for (unsigned long long j = 0; j < total_seq; ++j) {
        const float e = expf(scores[srow + j] - m);
        scores[srow + j] = e;
        sum += e;
      }
      inv_sum_sh = 1.0f / sum;
    }
  }
  __syncthreads();
  if (all_masked_sh) {
    for (unsigned long long j = tid; j < total_seq; j += nthreads) {
      scores[srow + j] = 0.0f;
    }
  } else {
    const float inv = inv_sum_sh;
    for (unsigned long long j = tid; j < total_seq; j += nthreads) {
      scores[srow + j] *= inv;
    }
  }
  __syncthreads();
  if (want_qk && qk_mode == 3) {
    for (unsigned long long j = tid; j < total_seq; j += nthreads) {
      store_float(qk_out, srow + j, scores[srow + j], dtype);
    }
    __syncthreads();
  }

  // Stage 5: Y = probs · V. Each thread owns whole output channels and sums
  // over keys in ascending order (bit-identical to the CPU reference).
  const unsigned long long ybase = out_is_3d
      ? (b * q_seq + i) * (q_heads * v_head_size) + qh * v_head_size
      : ((b * q_heads + qh) * q_seq + i) * v_head_size;
  for (unsigned long long c = tid; c < v_head_size; c += nthreads) {
    float acc = 0.0f;
    for (unsigned long long j = 0; j < total_seq; ++j) {
      const unsigned long long voff = ((b * kv_heads + kvh) * total_seq + j) * v_head_size;
      acc += scores[srow + j] * load_float(value, voff + c, dtype);
    }
    store_float(y, ybase + c, acc, dtype);
  }
}
"#;

/// Return the claim-time denial for Attention's positional input contract.
///
/// An omitted optional input is represented by [`DataType::Undefined`]. Keep
/// that distinct from a supplied tensor so absent mask/cache/length slots are
/// accepted without weakening dtype checks for tensors that are present.
pub(crate) fn unsupported_reason(
    opset: u64,
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    let dtype_at = |index: usize| {
        input_dtypes
            .get(index)
            .copied()
            .unwrap_or(DataType::Undefined)
    };
    let floating_denial = |dtype| {
        let dtype = match dtype {
            DataType::Float16 => "f16".into(),
            DataType::BFloat16 => "bf16".into(),
            other => format!("{other:?}"),
        };
        Cow::Owned(format!(
            "Attention: dtype {dtype} not supported on CUDA (supported: f32, f16, bf16)"
        ))
    };

    for index in 0..3 {
        let dtype = dtype_at(index);
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Some(floating_denial(dtype));
        }
    }
    if dtype_at(1) != dtype_at(0) || dtype_at(2) != dtype_at(0) {
        return Some(Cow::Borrowed(
            "Attention: Q, K, and V must use the same floating dtype on CUDA",
        ));
    }

    let mask_dtype = dtype_at(3);
    if !matches!(
        mask_dtype,
        DataType::Undefined
            | DataType::Bool
            | DataType::Float32
            | DataType::Float16
            | DataType::BFloat16
    ) {
        return Some(Cow::Owned(format!(
            "Attention: attn_mask dtype {mask_dtype:?} not supported (expected bool, f32, f16, or bf16 when provided)"
        )));
    }
    if !matches!(mask_dtype, DataType::Undefined | DataType::Bool) && mask_dtype != dtype_at(0) {
        return Some(Cow::Borrowed(
            "Attention: floating attn_mask must use the same dtype as Q, K, and V on CUDA",
        ));
    }

    let past_key_dtype = dtype_at(4);
    let past_value_dtype = dtype_at(5);
    for dtype in [past_key_dtype, past_value_dtype] {
        if dtype != DataType::Undefined
            && !matches!(
                dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            )
        {
            return Some(floating_denial(dtype));
        }
        if dtype != DataType::Undefined && dtype != dtype_at(0) {
            return Some(Cow::Borrowed(
                "Attention: Q/K/V and past_key/past_value must use the same floating dtype on CUDA",
            ));
        }
    }
    let has_past_key = past_key_dtype != DataType::Undefined;
    let has_past_value = past_value_dtype != DataType::Undefined;
    if has_past_key != has_past_value {
        return Some(Cow::Borrowed(
            "Attention: past_key and past_value must be provided together",
        ));
    }

    let nonpad_dtype = dtype_at(6);
    if !matches!(nonpad_dtype, DataType::Undefined | DataType::Int64) {
        return Some(Cow::Owned(format!(
            "Attention: nonpad_kv_seqlen dtype {nonpad_dtype:?} not supported (expected int64 when provided)"
        )));
    }
    let has_nonpad = nonpad_dtype != DataType::Undefined;
    if has_nonpad && opset < 24 {
        return Some(Cow::Borrowed(
            "Attention: nonpad_kv_seqlen was added in opset 24 and is not valid for opset 23",
        ));
    }
    if has_nonpad && has_past_key {
        return Some(Cow::Borrowed(
            "Attention: nonpad_kv_seqlen must not be used together with past_key/past_value",
        ));
    }

    None
}

/// f32/f16/bf16 standard-`Attention` kernel carrying the resolved attributes.
pub struct StandardAttentionKernel {
    runtime: Arc<CudaRuntime>,
    /// Explicit score scale; `None` → default `1/sqrt(head_size)`.
    scale: Option<f32>,
    is_causal: bool,
    q_num_heads: Option<usize>,
    kv_num_heads: Option<usize>,
    qk_matmul_output_mode: i64,
    /// Softcap value; `0.0` disables it.
    softcap: f32,
    /// The registered opset version this kernel serves (23, or 24 for 24–26).
    /// Controls `nonpad_kv_seqlen` acceptance (opset 24+ only).
    since_version: u32,
}

/// Factory for [`StandardAttentionKernel`], reading the standard-`Attention`
/// attributes. `since_version` selects the opset semantics (23 vs 24–26).
pub struct StandardAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
    pub since_version: u32,
}

impl KernelFactory for StandardAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let scale = node.attr("scale").and_then(|a| a.as_float());
        let is_causal = node.attr("is_causal").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        let q_num_heads = node
            .attr("q_num_heads")
            .and_then(|a| a.as_int())
            .map(|v| v as usize);
        let kv_num_heads = node
            .attr("kv_num_heads")
            .and_then(|a| a.as_int())
            .map(|v| v as usize);
        let qk_matmul_output_mode = node
            .attr("qk_matmul_output_mode")
            .and_then(|a| a.as_int())
            .unwrap_or(0);
        let softcap = node
            .attr("softcap")
            .and_then(|a| a.as_float())
            .unwrap_or(0.0);
        if !(0..=3).contains(&qk_matmul_output_mode) {
            return Err(EpError::KernelFailed(format!(
                "Attention: qk_matmul_output_mode {qk_matmul_output_mode} is not supported \
                 (only 0, 1, 2, 3 are implemented)"
            )));
        }
        Ok(Box::new(StandardAttentionKernel {
            runtime: self.runtime.clone(),
            scale,
            is_causal,
            q_num_heads,
            kv_num_heads,
            qk_matmul_output_mode,
            softcap,
            since_version: self.since_version,
        }))
    }
}

fn check_arity(
    name: &str,
    inputs: &[TensorView],
    outputs: &[TensorMut],
    min: usize,
    max: usize,
    min_outputs: usize,
) -> Result<()> {
    if !(min..=max).contains(&inputs.len()) || outputs.len() < min_outputs {
        return Err(EpError::KernelFailed(format!(
            "{name}: expected {min}..={max} inputs and at least {min_outputs} outputs"
        )));
    }
    Ok(())
}

/// Resolved `[batch, heads, seq, dim]` view of a Q/K/V input, keeping the input
/// on the device (no host copy). A 3D `(batch, seq, heads·dim)` input records
/// `is_3d` so the on-device gather can reshape it into heads.
struct BhsdDims {
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
    is_3d: bool,
}

/// Resolve a Q/K/V input's `[batch, heads, seq, dim]` dims without copying data.
///
/// A 4D input `(batch, heads, seq, dim)` is read as-is. A 3D input
/// `(batch, seq, heads·dim)` reshapes to heads via `num_heads` (from the
/// `q_num_heads`/`kv_num_heads` attributes), which is required and must divide
/// the hidden size. Non-contiguous inputs are rejected (the kernel requests
/// contiguous inputs).
fn resolve_bhsd(view: &TensorView, name: &str, num_heads: Option<usize>) -> Result<BhsdDims> {
    if !view.is_contiguous() {
        return Err(EpError::KernelFailed(
            "Attention: non-contiguous inputs are not supported".into(),
        ));
    }
    if !matches!(
        view.dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Err(EpError::KernelFailed(format!(
            "Attention: expected f32, f16, or bf16 input, got {:?}",
            view.dtype
        )));
    }
    let shape = view.shape;
    match shape.len() {
        4 => Ok(BhsdDims {
            batch: shape[0],
            heads: shape[1],
            seq: shape[2],
            dim: shape[3],
            is_3d: false,
        }),
        3 => {
            let heads = num_heads.ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Attention: 3D {name} input requires the corresponding \
                     q_num_heads/kv_num_heads attribute"
                ))
            })?;
            if heads == 0 {
                return Err(EpError::KernelFailed(format!(
                    "Attention: {name} num_heads must be > 0"
                )));
            }
            let (batch, seq, hidden) = (shape[0], shape[1], shape[2]);
            if hidden % heads != 0 {
                return Err(EpError::KernelFailed(format!(
                    "Attention: 3D {name} hidden size {hidden} is not divisible by num_heads \
                     {heads}"
                )));
            }
            Ok(BhsdDims {
                batch,
                heads,
                seq,
                dim: hidden / heads,
                is_3d: true,
            })
        }
        other => Err(EpError::KernelFailed(format!(
            "Attention: {name} must be rank 3 or 4, got rank {other}"
        ))),
    }
}

fn dense_i64(runtime: &CudaRuntime, view: &TensorView) -> Result<Vec<i64>> {
    if view.dtype != DataType::Int64 {
        return Err(EpError::KernelFailed(
            "Attention: nonpad_kv_seqlen must be int64".into(),
        ));
    }
    if !view.is_contiguous() {
        return Err(EpError::KernelFailed(
            "Attention: non-contiguous inputs are not supported".into(),
        ));
    }
    let mut bytes = vec![0u8; view.dtype.storage_bytes(view.numel())];
    unsafe {
        runtime.dtoh(&mut bytes, cuptr(view.data_ptr::<u8>() as *const c_void))?;
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|b| i64::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}

/// Validate an output slot (contiguous requested dtype with the expected element count) and
/// return its device pointer.
fn output_ptr(output: &mut TensorMut, dtype: DataType, expected: usize) -> Result<CUdeviceptr> {
    if output.dtype != dtype || !output.is_contiguous() || output.numel() != expected {
        return Err(EpError::KernelFailed(
            "Attention: output must be contiguous and use the input dtype with the expected shape"
                .into(),
        ));
    }
    Ok(cuptr(output.data_ptr_mut::<u8>() as *const c_void))
}

/// Mask kind + right-aligned broadcast dims passed to the device kernel.
struct MaskMeta {
    ptr: CUdeviceptr,
    kind: i32,
    rank: i32,
    dims: [u64; 4],
}

impl StandardAttentionKernel {
    /// Launch `build_kv` to gather a K/V input (plus an optional past cache)
    /// into a contiguous `[batch, heads, total_seq, dim]` present buffer at
    /// `out_ptr`.
    #[allow(clippy::too_many_arguments)]
    fn launch_build_kv(
        &self,
        past_ptr: CUdeviceptr,
        cur_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        has_past: bool,
        cur_is_3d: bool,
        past_is_3d: bool,
        dtype: i32,
        batch: usize,
        heads: usize,
        past_seq: usize,
        cur_seq: usize,
        total_seq: usize,
        dim: usize,
    ) -> Result<()> {
        let elements = (batch * heads * total_seq * dim) as u64;
        if elements == 0 {
            return Ok(());
        }
        let func = self
            .runtime
            .nvrtc_function(ATTENTION_MODULE, ATTENTION_SOURCE, "build_kv")?;
        let has_past_i = i32::from(has_past);
        let cur_is_3d_i = i32::from(cur_is_3d);
        let past_is_3d_i = i32::from(past_is_3d);
        let batch = batch as u64;
        let heads = heads as u64;
        let past_seq = past_seq as u64;
        let cur_seq = cur_seq as u64;
        let total_seq = total_seq as u64;
        let dim = dim as u64;
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&past_ptr)
            .arg(&cur_ptr)
            .arg(&out_ptr)
            .arg(&dtype)
            .arg(&has_past_i)
            .arg(&cur_is_3d_i)
            .arg(&past_is_3d_i)
            .arg(&batch)
            .arg(&heads)
            .arg(&past_seq)
            .arg(&cur_seq)
            .arg(&total_seq)
            .arg(&dim)
            .arg(&elements);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch build_kv", error))
        .map(|_| ())
    }
}

impl Kernel for StandardAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Attention", inputs, outputs, 3, 7, 1)?;
        // Inputs may have been uploaded asynchronously on the EP stream.
        self.runtime.synchronize()?;

        let q_rank = inputs[0].shape.len();
        let q = resolve_bhsd(&inputs[0], "Q", self.q_num_heads)?;
        let k_cur = resolve_bhsd(&inputs[1], "K", self.kv_num_heads)?;
        let v_cur = resolve_bhsd(&inputs[2], "V", self.kv_num_heads)?;
        let dtype = inputs[0].dtype;
        if inputs[1].dtype != dtype || inputs[2].dtype != dtype {
            return Err(EpError::KernelFailed(
                "Attention: Q, K, and V must use the same floating dtype on CUDA".into(),
            ));
        }
        let dtype_code = match dtype {
            DataType::Float32 => 0,
            DataType::Float16 => 1,
            _ => 2,
        };
        let element_bytes = dtype.storage_bytes(1);

        // Optional past KV cache (inputs 4 and 5). They must be used together.
        // Presence is decided by input-slot binding (a null "absent" view for an
        // omitted optional input), NOT by an empty shape — a genuinely present
        // rank-0 tensor also has an empty shape but must not be treated as
        // absent.
        let has_past_key = inputs.len() > 4 && !inputs[4].is_absent();
        let has_past_value = inputs.len() > 5 && !inputs[5].is_absent();
        if has_past_key != has_past_value {
            return Err(EpError::KernelFailed(
                "Attention: past_key and past_value must be provided together".into(),
            ));
        }
        let past_key = if has_past_key {
            Some(resolve_bhsd(&inputs[4], "past_key", self.kv_num_heads)?)
        } else {
            None
        };
        let past_value = if has_past_value {
            Some(resolve_bhsd(&inputs[5], "past_value", self.kv_num_heads)?)
        } else {
            None
        };
        if has_past_key && (inputs[4].dtype != dtype || inputs[5].dtype != dtype) {
            return Err(EpError::KernelFailed(
                "Attention: Q/K/V and past_key/past_value must use the same floating dtype on CUDA"
                    .into(),
            ));
        }
        let key_past_seq = past_key.as_ref().map(|p| p.seq).unwrap_or(0);
        let value_past_seq = past_value.as_ref().map(|p| p.seq).unwrap_or(0);

        // Preserve the concat compatibility checks (past vs current dims).
        if let Some(past) = &past_key
            && (past.batch != k_cur.batch || past.heads != k_cur.heads || past.dim != k_cur.dim)
        {
            return Err(EpError::KernelFailed(format!(
                "Attention: past_key dims (b={},h={},d={}) incompatible with current \
                 (b={},h={},d={})",
                past.batch, past.heads, past.dim, k_cur.batch, k_cur.heads, k_cur.dim
            )));
        }
        if let Some(past) = &past_value
            && (past.batch != v_cur.batch || past.heads != v_cur.heads || past.dim != v_cur.dim)
        {
            return Err(EpError::KernelFailed(format!(
                "Attention: past_value dims (b={},h={},d={}) incompatible with current \
                 (b={},h={},d={})",
                past.batch, past.heads, past.dim, v_cur.batch, v_cur.heads, v_cur.dim
            )));
        }

        // `nonpad_kv_seqlen` (7th input, opset 24+): per-batch count of valid
        // (non-padding) KV tokens, used when the KV cache lives outside the op.
        // It shifts the causal frontier by `nonpad_kv_seqlen[b] - q_seq` and is
        // mutually exclusive with an in-op past cache.
        let has_nonpad = inputs.len() > 6 && !inputs[6].is_absent();
        if has_nonpad && self.since_version < 24 {
            return Err(EpError::KernelFailed(
                "Attention: the optional `nonpad_kv_seqlen` input was added in opset 24 and is \
                 not valid for opset 23"
                    .into(),
            ));
        }
        if has_nonpad && (has_past_key || has_past_value) {
            return Err(EpError::KernelFailed(
                "Attention: `nonpad_kv_seqlen` must not be used together with past_key/past_value \
                 (external vs. in-op KV cache)"
                    .into(),
            ));
        }
        let nonpad_kv_seqlen: Option<Vec<i64>> = if has_nonpad {
            let seqlen = dense_i64(&self.runtime, &inputs[6])?;
            if seqlen.len() != q.batch {
                return Err(EpError::KernelFailed(format!(
                    "Attention: nonpad_kv_seqlen length {} must equal batch_size {}",
                    seqlen.len(),
                    q.batch
                )));
            }
            Some(seqlen)
        } else {
            None
        };

        let batch = q.batch;
        let q_heads = q.heads;
        let q_seq = q.seq;
        let head_size = q.dim;
        let kv_heads = k_cur.heads;
        let total_seq = key_past_seq + k_cur.seq;
        let value_total_seq = value_past_seq + v_cur.seq;
        let v_head_size = v_cur.dim;

        if k_cur.dim != head_size {
            return Err(EpError::KernelFailed(format!(
                "Attention: Q head_size {head_size} != K head_size {}",
                k_cur.dim
            )));
        }
        if value_total_seq != total_seq {
            return Err(EpError::KernelFailed(format!(
                "Attention: present_key seq {total_seq} != present_value seq {value_total_seq}"
            )));
        }
        if k_cur.batch != batch || v_cur.batch != batch {
            return Err(EpError::KernelFailed(
                "Attention: Q, K, V must share the batch dimension".into(),
            ));
        }
        if kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(EpError::KernelFailed(format!(
                "Attention: q_num_heads {q_heads} must be a positive multiple of kv_num_heads \
                 {kv_heads} (MHA/GQA/MQA)"
            )));
        }
        let group = q_heads / kv_heads;

        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());
        // Fold `sqrt(scale)` into each Q and K operand so the dot product is
        // `(Q·√scale)·(K·√scale)` rather than `scale·(Q·K)`. This matches the
        // spec's `Q*sqrt(scale)`, `K*sqrt(scale)` pattern and avoids overflowing
        // an intermediate `Q·Kᵀ` for extreme magnitudes.
        let sqrt_scale = scale.sqrt();

        // Resolve the attention mask (input 3), if present. Its bytes stay on
        // the device; only the broadcast metadata is read on the host. Presence
        // is decided by input-slot binding, so a rank-0 (scalar) mask is honored
        // rather than mistaken for an omitted input.
        let mask = if inputs.len() > 3 && !inputs[3].is_absent() {
            let m = &inputs[3];
            if !m.is_contiguous() {
                return Err(EpError::KernelFailed(
                    "Attention: non-contiguous inputs are not supported".into(),
                ));
            }
            let rank = m.shape.len();
            if rank > 4 {
                return Err(EpError::KernelFailed(format!(
                    "Attention: attn_mask rank {rank} is not supported (max 4)"
                )));
            }
            // Right-align the mask dims against [b, h, i, j]; missing leading
            // axes broadcast (size 1).
            let mut dims = [1u64; 4];
            for (k, &d) in m.shape.iter().enumerate() {
                dims[4 - rank + k] = d as u64;
            }
            let kind = match m.dtype {
                DataType::Bool => 2,
                DataType::Float32 => 1,
                DataType::Float16 => 3,
                DataType::BFloat16 => 4,
                other => {
                    return Err(EpError::KernelFailed(format!(
                        "Attention: attn_mask dtype {other:?} not supported (expected bool, f32, f16, or bf16)"
                    )));
                }
            };
            MaskMeta {
                ptr: cuptr(m.data_ptr::<u8>() as *const c_void),
                kind,
                rank: rank as i32,
                dims,
            }
        } else {
            MaskMeta {
                ptr: 0,
                kind: 0,
                rank: 0,
                dims: [1u64; 4],
            }
        };

        // Validate output slots up front (before any device work) so shape
        // errors surface cleanly.
        let y_expected = if q_rank == 3 {
            batch * q_seq * q_heads * v_head_size
        } else {
            batch * q_heads * q_seq * v_head_size
        };
        let y_ptr = output_ptr(&mut outputs[0], dtype, y_expected)?;
        let want_present_key = outputs.len() >= 2;
        let want_present_value = outputs.len() >= 3;
        let want_qk = outputs.len() >= 4;
        let present_key_expected = batch * kv_heads * total_seq * head_size;
        let present_value_expected = batch * kv_heads * total_seq * v_head_size;
        let qk_expected = batch * q_heads * q_seq * total_seq;

        // Validate present/qk outputs and capture their device pointers. Split
        // the mutable borrows so each slot is checked independently.
        let (rest0, rest1) = outputs.split_at_mut(1);
        let _ = rest0;
        let (present_key_out, rest_after1) = if want_present_key {
            let (a, b) = rest1.split_at_mut(1);
            (Some(output_ptr(&mut a[0], dtype, present_key_expected)?), b)
        } else {
            (None, rest1)
        };
        let (present_value_out, rest_after2) = if want_present_value {
            let (a, b) = rest_after1.split_at_mut(1);
            (
                Some(output_ptr(&mut a[0], dtype, present_value_expected)?),
                b,
            )
        } else {
            (None, rest_after1)
        };
        let qk_ptr = if want_qk {
            output_ptr(&mut rest_after2[0], dtype, qk_expected)?
        } else {
            0
        };

        // Q/K/V device pointers (bulk data stays on the device).
        let q_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let k_cur_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let v_cur_ptr = cuptr(inputs[2].data_ptr::<u8>() as *const c_void);
        let past_key_ptr = past_key
            .as_ref()
            .map(|_| cuptr(inputs[4].data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let past_value_ptr = past_value
            .as_ref()
            .map(|_| cuptr(inputs[5].data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);

        // Per-batch causal offset and padding frontier (built on the host, then
        // uploaded as small device arrays). Query in-block index `i` attends
        // key `j` iff `j <= i + offset`. With an external cache the offset is
        // `nonpad_kv_seqlen[b] - q_seq`; with an in-op past cache it is
        // `past_seq`; otherwise 0. A negative offset fully masks leading query
        // rows (→ zero output rows). The padding frontier masks keys at
        // `j >= nonpad_kv_seqlen[b]` regardless of causal mode; `-1` disables it.
        let mut offsets = vec![0i64; batch.max(1)];
        let mut pad_limits = vec![-1i64; batch.max(1)];
        for b in 0..batch {
            offsets[b] = match &nonpad_kv_seqlen {
                Some(seqlen) => seqlen[b] - q_seq as i64,
                None => key_past_seq as i64,
            };
            pad_limits[b] = match &nonpad_kv_seqlen {
                Some(seqlen) => seqlen[b],
                None => -1,
            };
        }

        // Allocate device scratch; track owned allocations for cleanup.
        let mut owned: Vec<CUdeviceptr> = Vec::new();
        let result = (|| -> Result<()> {
            let alloc = |runtime: &CudaRuntime,
                         owned: &mut Vec<CUdeviceptr>,
                         bytes: usize|
             -> Result<CUdeviceptr> {
                let ptr = runtime.alloc_raw(bytes.max(1))?;
                owned.push(ptr);
                Ok(ptr)
            };

            // Present K/V: write directly into output slots when present, else
            // into scratch (still needed as the attention kernel's K/V source).
            let present_key_ptr = match present_key_out {
                Some(ptr) => ptr,
                None => alloc(
                    &self.runtime,
                    &mut owned,
                    present_key_expected * element_bytes,
                )?,
            };
            let present_value_ptr = match present_value_out {
                Some(ptr) => ptr,
                None => alloc(
                    &self.runtime,
                    &mut owned,
                    present_value_expected * element_bytes,
                )?,
            };

            // In-place KV growth: the decode graph binds the present K/V output
            // onto the same buffer as the past K/V input. `build_kv` rewrites the
            // whole cache with a *wider* per-head stride (total_seq > past_seq),
            // so writing directly into that buffer makes head h's current-token
            // store overlap head h+1's past load across unordered threads — a
            // data race that leaves every head beyond head 0 nondeterministic.
            // Stage the rebuild in a disjoint scratch buffer (reads the pristine
            // past), then copy the fully-formed dense cache back. General: any
            // model whose default-domain Attention grows an aliased KV cache.
            let stage_key = has_past_key && present_key_out == Some(past_key_ptr);
            let stage_value = has_past_value && present_value_out == Some(past_value_ptr);
            let key_kv_ptr = if stage_key {
                alloc(&self.runtime, &mut owned, present_key_expected * element_bytes)?
            } else {
                present_key_ptr
            };
            let value_kv_ptr = if stage_value {
                alloc(
                    &self.runtime,
                    &mut owned,
                    present_value_expected * element_bytes,
                )?
            } else {
                present_value_ptr
            };
            let scores_ptr = alloc(&self.runtime, &mut owned, qk_expected * 4)?;

            // Upload the per-batch control arrays.
            let offsets_ptr = alloc(&self.runtime, &mut owned, offsets.len() * 8)?;
            let pad_limits_ptr = alloc(&self.runtime, &mut owned, pad_limits.len() * 8)?;
            let offsets_bytes = unsafe {
                std::slice::from_raw_parts(offsets.as_ptr().cast::<u8>(), offsets.len() * 8)
            };
            let pad_bytes = unsafe {
                std::slice::from_raw_parts(pad_limits.as_ptr().cast::<u8>(), pad_limits.len() * 8)
            };
            unsafe { self.runtime.htod(offsets_bytes, offsets_ptr)? };
            unsafe { self.runtime.htod(pad_bytes, pad_limits_ptr)? };

            // Build present_key / present_value on the device.
            self.launch_build_kv(
                past_key_ptr,
                k_cur_ptr,
                key_kv_ptr,
                has_past_key,
                k_cur.is_3d,
                past_key.as_ref().map(|p| p.is_3d).unwrap_or(false),
                dtype_code,
                batch,
                kv_heads,
                key_past_seq,
                k_cur.seq,
                total_seq,
                head_size,
            )?;
            self.launch_build_kv(
                past_value_ptr,
                v_cur_ptr,
                value_kv_ptr,
                has_past_value,
                v_cur.is_3d,
                past_value.as_ref().map(|p| p.is_3d).unwrap_or(false),
                dtype_code,
                batch,
                kv_heads,
                value_past_seq,
                v_cur.seq,
                value_total_seq,
                v_head_size,
            )?;

            // When staged, publish the freshly-built dense cache back into the
            // aliased present/past buffer for the next step. Source and
            // destination are disjoint, so this copy is race-free.
            if stage_key {
                unsafe {
                    self.runtime.dtod(
                        key_kv_ptr,
                        present_key_ptr,
                        present_key_expected * element_bytes,
                    )?;
                }
            }
            if stage_value {
                unsafe {
                    self.runtime.dtod(
                        value_kv_ptr,
                        present_value_ptr,
                        present_value_expected * element_bytes,
                    )?;
                }
            }

            // Launch the main attention kernel: one block per query row.
            let func =
                self.runtime
                    .nvrtc_function(ATTENTION_MODULE, ATTENTION_SOURCE, "attention_row")?;
            let total_rows = (batch * q_heads * q_seq) as u64;
            if total_rows > 0 {
                let batch_u = batch as u64;
                let q_heads_u = q_heads as u64;
                let q_seq_u = q_seq as u64;
                let kv_heads_u = kv_heads as u64;
                let total_seq_u = total_seq as u64;
                let head_size_u = head_size as u64;
                let v_head_size_u = v_head_size as u64;
                let group_u = group as u64;
                let q_is_3d = i32::from(q.is_3d);
                let out_is_3d = i32::from(q_rank == 3);
                let is_causal = i32::from(self.is_causal);
                let mask_kind = mask.kind;
                let mask_rank = mask.rank;
                let (md0, md1, md2, md3) = (mask.dims[0], mask.dims[1], mask.dims[2], mask.dims[3]);
                let qk_mode = self.qk_matmul_output_mode as i32;
                let want_qk_i = i32::from(want_qk);
                let softcap = self.softcap;
                let mut builder = self.runtime.stream().launch_builder(&func);
                builder
                    .arg(&q_ptr)
                    .arg(&key_kv_ptr)
                    .arg(&value_kv_ptr)
                    .arg(&mask.ptr)
                    .arg(&scores_ptr)
                    .arg(&y_ptr)
                    .arg(&qk_ptr)
                    .arg(&offsets_ptr)
                    .arg(&pad_limits_ptr)
                    .arg(&batch_u)
                    .arg(&q_heads_u)
                    .arg(&q_seq_u)
                    .arg(&kv_heads_u)
                    .arg(&total_seq_u)
                    .arg(&head_size_u)
                    .arg(&v_head_size_u)
                    .arg(&group_u)
                    .arg(&dtype_code)
                    .arg(&q_is_3d)
                    .arg(&out_is_3d)
                    .arg(&is_causal)
                    .arg(&sqrt_scale)
                    .arg(&softcap)
                    .arg(&mask_kind)
                    .arg(&mask_rank)
                    .arg(&md0)
                    .arg(&md1)
                    .arg(&md2)
                    .arg(&md3)
                    .arg(&qk_mode)
                    .arg(&want_qk_i);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (total_rows.min(u32::MAX as u64).max(1) as u32, 1, 1),
                        block_dim: (ROW_THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| driver_err("launch attention_row", error))?;
            }
            self.runtime.synchronize()
        })();

        let mut free_result = Ok(());
        for ptr in owned {
            let freed = unsafe { self.runtime.free_raw(ptr) };
            if free_result.is_ok() {
                free_result = freed;
            }
        }
        result.and(free_result)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "setup synchronously uploads per-batch control arrays and reads nonpad_kv_seqlen D2H",
        )
    }
}

#[cfg(test)]
mod alias_tests {
    //! Regression tests for in-place KV-cache growth in the default-domain
    //! `Attention` kernel. DeepSeek-V2-Lite (MLA) binds the `present_key` /
    //! `present_value` outputs onto the SAME device buffer as the `past_key` /
    //! `past_value` inputs. `build_kv` re-lays-out the whole cache at a WIDER
    //! per-head stride (`total_seq` vs `past_seq`) in that shared buffer, so a
    //! head's current-token write collides with the next head's past read across
    //! unordered CUDA threads — nondeterministic for every head > 0. The fix
    //! stages the rebuild into a disjoint scratch buffer and copies it back, so
    //! the aliased result must equal a non-aliased reference and be stable across
    //! repeated runs.
    use super::*;
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
    use onnx_runtime_ir::{DeviceId, compute_contiguous_strides};
    use std::ffi::c_void;

    fn maybe_runtime() -> Option<Arc<CudaRuntime>> {
        CudaRuntime::new(0).ok().map(Arc::new)
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn bytes_f32(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // Deterministic pseudo-random fill in [-1, 1).
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9e3779b97f4a7c15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Reproduces the DeepSeek MLA decode step: multi-head KV cache grown by one
    /// token, with `present` optionally aliasing `past` in one device buffer.
    #[test]
    fn decode_kv_growth_alias_matches_reference_and_is_deterministic() {
        let Some(rt) = maybe_runtime() else {
            eprintln!("skipping: no CUDA device available");
            return;
        };
        let device = DeviceId::cuda(0);

        // MLA-flavoured shape: multi-head, asymmetric K (192-style) vs V head size.
        let heads = 4usize;
        let past = 3usize;
        let qlen = 1usize;
        let total = past + qlen;
        let kdim = 6usize; // key/query head size
        let vdim = 4usize; // value head size

        // Fixed inputs.
        let q = fill(heads * qlen * kdim, 1);
        let k_cur = fill(heads * qlen * kdim, 2);
        let v_cur = fill(heads * qlen * vdim, 3);
        let past_k = fill(heads * past * kdim, 4);
        let past_v = fill(heads * past * vdim, 5);

        let kernel = StandardAttentionKernel {
            runtime: rt.clone(),
            scale: None,
            is_causal: true,
            q_num_heads: Some(heads),
            kv_num_heads: Some(heads),
            qk_matmul_output_mode: 0,
            softcap: 0.0,
            since_version: 24,
        };

        // Runs one decode step; when `alias` the present KV outputs share the
        // past KV buffers. Returns the attention output Y as f32.
        let run = |alias: bool| -> Vec<f32> {
            let q_sh = [1usize, heads, qlen, kdim];
            let kcur_sh = [1usize, heads, qlen, kdim];
            let vcur_sh = [1usize, heads, qlen, vdim];
            let pastk_sh = [1usize, heads, past, kdim];
            let pastv_sh = [1usize, heads, past, vdim];
            let presk_sh = [1usize, heads, total, kdim];
            let presv_sh = [1usize, heads, total, vdim];
            let y_sh = [1usize, heads, qlen, vdim];

            let q_st = compute_contiguous_strides(&q_sh);
            let kcur_st = compute_contiguous_strides(&kcur_sh);
            let vcur_st = compute_contiguous_strides(&vcur_sh);
            let pastk_st = compute_contiguous_strides(&pastk_sh);
            let pastv_st = compute_contiguous_strides(&pastv_sh);
            let presk_st = compute_contiguous_strides(&presk_sh);
            let presv_st = compute_contiguous_strides(&presv_sh);
            let y_st = compute_contiguous_strides(&y_sh);

            // Cache buffers sized for the grown (total) length.
            let key_cap = heads * total * kdim * 4;
            let val_cap = heads * total * vdim * 4;
            let q_bytes = f32_bytes(&q);
            let kcur_bytes = f32_bytes(&k_cur);
            let vcur_bytes = f32_bytes(&v_cur);

            unsafe {
                let key_buf = rt.alloc_raw(key_cap).unwrap();
                let val_buf = rt.alloc_raw(val_cap).unwrap();
                let q_buf = rt.alloc_raw(q_bytes.len()).unwrap();
                let kcur_buf = rt.alloc_raw(kcur_bytes.len()).unwrap();
                let vcur_buf = rt.alloc_raw(vcur_bytes.len()).unwrap();
                // Past occupies the dense [heads, past, dim] prefix of the cache buffer.
                rt.htod(&f32_bytes(&past_k), key_buf).unwrap();
                rt.htod(&f32_bytes(&past_v), val_buf).unwrap();
                rt.htod(&q_bytes, q_buf).unwrap();
                rt.htod(&kcur_bytes, kcur_buf).unwrap();
                rt.htod(&vcur_bytes, vcur_buf).unwrap();

                let (presk_buf, presv_buf) = if alias {
                    (key_buf, val_buf)
                } else {
                    (rt.alloc_raw(key_cap).unwrap(), rt.alloc_raw(val_cap).unwrap())
                };
                let y_buf = rt.alloc_raw(heads * qlen * vdim * 4).unwrap();

                let dp = |p: CUdeviceptr| DevicePtr(p as *const c_void);
                let dpm = |p: CUdeviceptr| DevicePtrMut(p as *mut c_void);

                let inputs = [
                    TensorView::new(dp(q_buf), DataType::Float32, &q_sh, &q_st, device),
                    TensorView::new(dp(kcur_buf), DataType::Float32, &kcur_sh, &kcur_st, device),
                    TensorView::new(dp(vcur_buf), DataType::Float32, &vcur_sh, &vcur_st, device),
                    TensorView::absent(DataType::Float32),
                    TensorView::new(dp(key_buf), DataType::Float32, &pastk_sh, &pastk_st, device),
                    TensorView::new(dp(val_buf), DataType::Float32, &pastv_sh, &pastv_st, device),
                ];
                let mut outputs = [
                    TensorMut::new(dpm(y_buf), DataType::Float32, &y_sh, &y_st, device),
                    TensorMut::new(dpm(presk_buf), DataType::Float32, &presk_sh, &presk_st, device),
                    TensorMut::new(dpm(presv_buf), DataType::Float32, &presv_sh, &presv_st, device),
                ];

                kernel.execute(&inputs, &mut outputs).unwrap();

                let mut y_bytes = vec![0u8; heads * qlen * vdim * 4];
                rt.dtoh(&mut y_bytes, y_buf).unwrap();

                rt.free_raw(key_buf).unwrap();
                rt.free_raw(val_buf).unwrap();
                rt.free_raw(q_buf).unwrap();
                rt.free_raw(kcur_buf).unwrap();
                rt.free_raw(vcur_buf).unwrap();
                rt.free_raw(y_buf).unwrap();
                if !alias {
                    rt.free_raw(presk_buf).unwrap();
                    rt.free_raw(presv_buf).unwrap();
                }
                bytes_f32(&y_bytes)
            }
        };

        // Non-aliased present/past buffers give the race-free ground truth.
        let reference = run(false);
        // In-place growth must reproduce it exactly and be stable across runs.
        let aliased = run(true);
        assert_eq!(
            aliased, reference,
            "in-place KV-cache growth (present aliases past) must match the non-aliased reference"
        );
        for i in 0..4 {
            assert_eq!(
                run(true),
                aliased,
                "aliased KV-cache growth must be deterministic across runs (iteration {i})"
            );
        }
    }
}

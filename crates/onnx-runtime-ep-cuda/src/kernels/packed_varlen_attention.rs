//! Unpadded / packed variable-length attention (`pkg.nxrt::PackedVarlenAttention`
//! v1).
//!
//! This is a runtime-invented op (registered in [`onnx_runtime_ir::RUNTIME_DOMAIN`]
//! = `pkg.nxrt`, **not** `com.microsoft` or the default ONNX domain) that runs
//! scaled dot-product attention directly over *packed* — i.e. padding-free —
//! variable-length sequences. Where the standard `ai.onnx::Attention` kernel
//! lays every sequence out at a common padded length and NEG_INF-masks the
//! wasted tail, this kernel iterates only the valid tokens of each sequence, so
//! a ragged batch spends no compute or memory on padding.
//!
//! ## Packed layout & cumulative offsets
//!
//! The batch's `B` sequences are concatenated along a single token axis. Two
//! cumulative-sequence-length arrays (`cu_seqlens`, the standard flash-attention
//! varlen ABI) delimit them:
//!
//! ```text
//! cu_seqlens_q[0..=B]  = [0, L_q0, L_q0+L_q1, ..., total_q]     # int32, len B+1
//! cu_seqlens_kv[0..=B] = [0, L_kv0, L_kv0+L_kv1, ..., total_kv] # int32, len B+1
//! ```
//!
//! Sequence `b` owns query rows `[cu_seqlens_q[b], cu_seqlens_q[b+1])` and
//! key/value rows `[cu_seqlens_kv[b], cu_seqlens_kv[b+1])`. Attention is fully
//! block-diagonal: a query never attends across a sequence boundary.
//!
//! ## Relationship to ONNX Attention-24 `nonpad_kv_seqlen`
//!
//! Opset-24 `Attention` added `nonpad_kv_seqlen` — a per-batch count of valid
//! (non-padding) KV tokens for ragged batching against an external cache. The
//! packed equivalent is an *exclusive prefix sum*: `cu_seqlens_kv =
//! [0, nonpad[0], nonpad[0]+nonpad[1], ...]` once the padding is removed and the
//! KV tokens are packed. Feeding this kernel `cu_seqlens_kv` derived that way
//! reproduces the padded op's `nonpad_kv_seqlen` masking with none of the
//! padded compute. The GPU tests build the two representations from the same
//! logical batch and assert they agree.
//!
//! ## Schema (v1)
//!
//! * inputs
//!   0. `query`  — packed Q, `[total_q, num_heads, head_size]` (rank 3), or `[total_q, num_heads*head_size]` (rank 2).
//!   1. `key`    — packed K, `[total_kv, kv_num_heads, head_size]` or 2D.
//!   2. `value`  — packed V, `[total_kv, kv_num_heads, v_head_size]` or 2D.
//!   3. `cu_seqlens_q`  — int32 `[B+1]`, cumulative query offsets.
//!   4. `cu_seqlens_kv` — int32 `[B+1]`, cumulative key/value offsets.
//! * output
//!   0. `output` — packed `[total_q, num_heads, v_head_size]` (rank matches Q).
//! * attributes
//!   * `num_heads` (int, required) — query heads.
//!   * `kv_num_heads` (int, optional) — defaults to `num_heads`; must divide it (MHA/GQA/MQA head sharing).
//!   * `scale` (float, optional) — defaults `1/sqrt(head_size)`.
//!   * `is_causal` (int, optional, default 0) — tail-aligned causal mask.
//!   * `softcap` (float, optional, default 0) — `softcap·tanh(score/softcap)`.
//!
//! ## Causal alignment
//!
//! Query local position `i` (0-based within its sequence) attends key local
//! position `jk` iff `jk <= i + (L_kv - L_q)`. The `L_kv - L_q` offset
//! tail-aligns the query block against the key block, matching both flash-attn
//! varlen and the `nonpad_kv_seqlen - q_seq` offset the standard `Attention`
//! kernel uses. For self-attention prefill (`L_q == L_kv`) this is the ordinary
//! lower-triangular mask (`jk <= i`).
//!
//! ## Numerics & determinism
//!
//! One CUDA block services one `(query token, query head)` row. Scores are kept
//! in fp32 (f16/bf16 are converted on load/store around fp32 accumulators),
//! `sqrt(scale)` is folded into each Q and K operand, and a single lead thread
//! performs the softmax max/exp/sum in ascending key order — bit-identical to
//! the standard `Attention` kernel, so an all-equal-length packed batch matches
//! the padded reference exactly.
//!
//! ## Correctness-first scope
//!
//! v1 is a blocked-softmax kernel: it materializes each row's scores in a device
//! scratch buffer (sized `total_rows * max_kv_len`) rather than the register-tiled
//! streaming softmax of a full flash-attention kernel. Launch geometry is derived
//! from the live device's `multiprocessor_count` / `max_threads_per_block` (no
//! hardcoded per-GPU constants). Flash-style Q/K/V tiling to shrink the scratch
//! and lift throughput is a documented follow-up (Advances #86).

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "PackedVarlenAttention";
const DOMAIN: &str = onnx_runtime_ir::RUNTIME_DOMAIN;
const MODULE: &str = "packed_varlen_attention_f32_f16_bf16_v1";
const ENTRY: &str = "packed_varlen_attention_row";
/// Default threads per block; capped to the device's `max_threads_per_block`.
const ROW_THREADS: u32 = 128;
/// Resident blocks per SM used to size the grid before the grid-stride loop.
const BLOCKS_PER_SM: u32 = 32;

const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define NEG_INF __int_as_float(0xff800000)

// dtype is 0 for f32, 1 for f16, and 2 for bf16. All math stays in fp32; only
// the externally visible Q/K/V/output storage uses the requested type.
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

// One block per (packed query token, query head). Packed Q/K/V share a flat
// [token, head, dim] layout, so a rank-2 [token, head*dim] input indexes
// identically. Grid-stride over rows keeps the launch bounded by the device's
// resident-block budget while still covering an arbitrary token count.
extern "C" __global__ void packed_varlen_attention_row(
    const void* q, const void* key, const void* value,
    const int* cu_seqlens_q, const int* cu_seqlens_kv, const int* q_batch_ids,
    float* scores, void* y,
    unsigned long long num_heads, unsigned long long kv_heads,
    unsigned long long head_size, unsigned long long v_head_size,
    unsigned long long group, unsigned long long total_q, unsigned long long max_kv_len,
    int dtype, int is_causal, float sqrt_scale, float softcap) {
  const unsigned long long total_rows = total_q * num_heads;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  __shared__ float inv_sum_sh;
  __shared__ int all_masked_sh;

  for (unsigned long long row = blockIdx.x; row < total_rows; row += (unsigned long long)gridDim.x) {
    const unsigned long long gq = row / num_heads;
    const unsigned long long qh = row % num_heads;
    const unsigned long long b = (unsigned long long)q_batch_ids[gq];
    const long long q_start = cu_seqlens_q[b];
    const long long q_end = cu_seqlens_q[b + 1];
    const long long kv_start = cu_seqlens_kv[b];
    const long long kv_end = cu_seqlens_kv[b + 1];
    const long long q_len = q_end - q_start;
    const long long kv_len = kv_end - kv_start;
    const long long i = (long long)gq - q_start;
    const unsigned long long kvh = qh / group;
    const unsigned long long srow = row * max_kv_len;
    // Tail-aligned causal frontier: query i attends key jk iff jk <= i + offset.
    const long long causal_limit = i + (kv_len - q_len);

    // Stage 1: scaled Q.Kᵀ scores over this sequence's keys (sqrt(scale) folded
    // into each operand so extreme magnitudes don't overflow the dot product).
    const unsigned long long qoff = (gq * num_heads + qh) * head_size;
    for (long long jk = tid; jk < kv_len; jk += nthreads) {
      const unsigned long long gk = (unsigned long long)(kv_start + jk);
      const unsigned long long koff = (gk * kv_heads + kvh) * head_size;
      float acc = 0.0f;
      for (unsigned long long p = 0; p < head_size; ++p) {
        acc += (load_float(q, qoff + p, dtype) * sqrt_scale)
            * (load_float(key, koff + p, dtype) * sqrt_scale);
      }
      scores[srow + (unsigned long long)jk] = acc;
    }
    __syncthreads();

    // Stage 2: softcap (before mask), applied when nonzero.
    if (softcap != 0.0f) {
      for (long long jk = tid; jk < kv_len; jk += nthreads) {
        const float s = scores[srow + (unsigned long long)jk];
        scores[srow + (unsigned long long)jk] = softcap * tanhf(s / softcap);
      }
      __syncthreads();
    }

    // Stage 3: causal frontier. Packed sequences carry no padding, so this is
    // the only mask.
    if (is_causal) {
      for (long long jk = tid; jk < kv_len; jk += nthreads) {
        if (jk > causal_limit) {
          scores[srow + (unsigned long long)jk] = NEG_INF;
        }
      }
      __syncthreads();
    }

    // Stage 4: numerically-stable softmax. The lead thread reduces in ascending
    // key order to match the standard Attention kernel bit-for-bit; the final
    // normalize is embarrassingly parallel. A fully-masked row emits zeros.
    if (tid == 0) {
      float m = NEG_INF;
      for (long long jk = 0; jk < kv_len; ++jk) {
        m = fmaxf(m, scores[srow + (unsigned long long)jk]);
      }
      if (m == NEG_INF) {
        all_masked_sh = 1;
        inv_sum_sh = 0.0f;
      } else {
        all_masked_sh = 0;
        float sum = 0.0f;
        for (long long jk = 0; jk < kv_len; ++jk) {
          const float e = expf(scores[srow + (unsigned long long)jk] - m);
          scores[srow + (unsigned long long)jk] = e;
          sum += e;
        }
        inv_sum_sh = 1.0f / sum;
      }
    }
    __syncthreads();
    if (all_masked_sh) {
      for (long long jk = tid; jk < kv_len; jk += nthreads) {
        scores[srow + (unsigned long long)jk] = 0.0f;
      }
    } else {
      const float inv = inv_sum_sh;
      for (long long jk = tid; jk < kv_len; jk += nthreads) {
        scores[srow + (unsigned long long)jk] *= inv;
      }
    }
    __syncthreads();

    // Stage 5: Y = probs . V. Each thread owns whole output channels and sums
    // over keys in ascending order (bit-identical to the CPU reference).
    const unsigned long long ybase = (gq * num_heads + qh) * v_head_size;
    for (unsigned long long c = tid; c < v_head_size; c += nthreads) {
      float acc = 0.0f;
      for (long long jk = 0; jk < kv_len; ++jk) {
        const unsigned long long gk = (unsigned long long)(kv_start + jk);
        const unsigned long long voff = (gk * kv_heads + kvh) * v_head_size;
        acc += scores[srow + (unsigned long long)jk] * load_float(value, voff + c, dtype);
      }
      store_float(y, ybase + c, acc, dtype);
    }
    __syncthreads();
  }
}
"#;

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{DOMAIN}::{OP}: {}", message.into()))
}

/// Claim-time denial for the packed-varlen positional/dtype contract. Mirrors
/// the other `pkg.nxrt` ops: reject unsupported dtypes up front so the planner
/// falls back cleanly instead of failing at execute.
pub(crate) fn unsupported_reason(
    node: &Node,
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    let dtype_at = |index: usize| {
        input_dtypes
            .get(index)
            .copied()
            .unwrap_or(DataType::Undefined)
    };
    for index in 0..3 {
        let dtype = dtype_at(index);
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            let name = match dtype {
                DataType::Float16 => "f16".into(),
                DataType::BFloat16 => "bf16".into(),
                other => format!("{other:?}"),
            };
            return Some(Cow::Owned(format!(
                "PackedVarlenAttention: dtype {name} not supported on CUDA (Q/K/V must be f32, f16, or bf16)"
            )));
        }
    }
    if dtype_at(1) != dtype_at(0) || dtype_at(2) != dtype_at(0) {
        return Some(Cow::Borrowed(
            "PackedVarlenAttention: Q, K, and V must use the same floating dtype on CUDA",
        ));
    }
    for index in 3..5 {
        let dtype = dtype_at(index);
        if dtype != DataType::Undefined && dtype != DataType::Int32 {
            return Some(Cow::Owned(format!(
                "PackedVarlenAttention: cu_seqlens input {index} dtype {dtype:?} not supported (expected int32)"
            )));
        }
    }
    if node.attr("num_heads").and_then(|a| a.as_int()).is_none() {
        return Some(Cow::Borrowed(
            "PackedVarlenAttention: missing required int attribute 'num_heads'",
        ));
    }
    None
}

/// Factory reading the packed-varlen attention attributes.
pub struct PackedVarlenAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for PackedVarlenAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let num_heads = node
            .attr("num_heads")
            .and_then(|a| a.as_int())
            .ok_or_else(|| error("missing required int attribute 'num_heads'"))?;
        if num_heads <= 0 {
            return Err(error(format!(
                "num_heads must be positive, got {num_heads}"
            )));
        }
        let kv_num_heads = node.attr("kv_num_heads").and_then(|a| a.as_int());
        if let Some(kv) = kv_num_heads
            && kv <= 0
        {
            return Err(error(format!("kv_num_heads must be positive, got {kv}")));
        }
        let scale = node.attr("scale").and_then(|a| a.as_float());
        let is_causal = node.attr("is_causal").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        let softcap = node
            .attr("softcap")
            .and_then(|a| a.as_float())
            .unwrap_or(0.0);
        Ok(Box::new(PackedVarlenAttentionKernel {
            runtime: self.runtime.clone(),
            num_heads: num_heads as usize,
            kv_num_heads: kv_num_heads.map(|v| v as usize),
            scale,
            is_causal,
            softcap,
        }))
    }
}

#[derive(Debug)]
struct PackedVarlenAttentionKernel {
    runtime: Arc<CudaRuntime>,
    num_heads: usize,
    kv_num_heads: Option<usize>,
    scale: Option<f32>,
    is_causal: bool,
    softcap: f32,
}

/// Resolved `[tokens, heads, dim]` shape of a packed Q/K/V input. A rank-2
/// `[tokens, heads*dim]` input reshapes into heads via `num_heads`.
struct PackedDims {
    tokens: usize,
    dim: usize,
}

fn resolve_packed(view: &TensorView, name: &str, heads: usize) -> Result<PackedDims> {
    if !view.is_contiguous() {
        return Err(error(format!("{name} must be contiguous on CUDA")));
    }
    if heads == 0 {
        return Err(error(format!("{name} head count must be > 0")));
    }
    match view.shape.len() {
        3 => {
            if view.shape[1] != heads {
                return Err(error(format!(
                    "{name} rank-3 head dim {} must equal head count {heads}",
                    view.shape[1]
                )));
            }
            Ok(PackedDims {
                tokens: view.shape[0],
                dim: view.shape[2],
            })
        }
        2 => {
            let hidden = view.shape[1];
            if !hidden.is_multiple_of(heads) {
                return Err(error(format!(
                    "{name} rank-2 hidden size {hidden} is not divisible by head count {heads}"
                )));
            }
            Ok(PackedDims {
                tokens: view.shape[0],
                dim: hidden / heads,
            })
        }
        other => Err(error(format!(
            "{name} must be rank 2 or 3, got rank {other}"
        ))),
    }
}

/// Read a contiguous int32 `cu_seqlens` array off the device to the host. The
/// bulk Q/K/V tensors stay resident; only these tiny per-batch offset arrays
/// (length `B+1`) round-trip so the host can size the launch and label tokens.
fn read_cu_seqlens(runtime: &CudaRuntime, view: &TensorView, name: &str) -> Result<Vec<i32>> {
    if view.dtype != DataType::Int32 {
        return Err(error(format!("{name} must be int32")));
    }
    if !view.is_contiguous() {
        return Err(error(format!("{name} must be contiguous on CUDA")));
    }
    if view.shape.len() != 1 || view.shape[0] < 2 {
        return Err(error(format!(
            "{name} must be a 1D tensor of length batch_size + 1 (>= 2), got shape {:?}",
            view.shape
        )));
    }
    let count = view.shape[0];
    let mut bytes = vec![0u8; count * 4];
    // SAFETY: `view` is a live, contiguous int32 device allocation of `count`
    // elements; the destination matches its byte length exactly.
    unsafe {
        runtime.dtoh(&mut bytes, cuptr(view.data_ptr::<u8>() as *const c_void))?;
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}

impl Kernel for PackedVarlenAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 5 || outputs.len() != 1 {
            return Err(error(format!(
                "expected 5 inputs and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        let dtype = inputs[0].dtype;
        let (dtype_code, element_bytes) = match dtype {
            DataType::Float32 => (0i32, 4usize),
            DataType::Float16 => (1, 2),
            DataType::BFloat16 => (2, 2),
            other => {
                return Err(error(format!(
                    "Q/K/V dtype {other:?} not supported (expected f32, f16, or bf16)"
                )));
            }
        };
        if inputs[1].dtype != dtype || inputs[2].dtype != dtype {
            return Err(error("Q, K, and V must use the same floating dtype"));
        }

        let num_heads = self.num_heads;
        let kv_num_heads = self.kv_num_heads.unwrap_or(num_heads);
        if kv_num_heads == 0 || !num_heads.is_multiple_of(kv_num_heads) {
            return Err(error(format!(
                "num_heads {num_heads} must be a positive multiple of kv_num_heads {kv_num_heads} (MHA/GQA/MQA)"
            )));
        }
        let group = num_heads / kv_num_heads;

        let q_dims = resolve_packed(&inputs[0], "query", num_heads)?;
        let k_dims = resolve_packed(&inputs[1], "key", kv_num_heads)?;
        let v_dims = resolve_packed(&inputs[2], "value", kv_num_heads)?;
        let total_q = q_dims.tokens;
        let total_kv = k_dims.tokens;
        let head_size = q_dims.dim;
        let v_head_size = v_dims.dim;
        if k_dims.dim != head_size {
            return Err(error(format!(
                "query head_size {head_size} != key head_size {}",
                k_dims.dim
            )));
        }
        if v_dims.tokens != total_kv {
            return Err(error(format!(
                "key token count {total_kv} != value token count {}",
                v_dims.tokens
            )));
        }

        let cu_seqlens_q = read_cu_seqlens(&self.runtime, &inputs[3], "cu_seqlens_q")?;
        let cu_seqlens_kv = read_cu_seqlens(&self.runtime, &inputs[4], "cu_seqlens_kv")?;
        if cu_seqlens_q.len() != cu_seqlens_kv.len() {
            return Err(error(format!(
                "cu_seqlens_q length {} != cu_seqlens_kv length {} (both must be batch_size + 1)",
                cu_seqlens_q.len(),
                cu_seqlens_kv.len()
            )));
        }
        let batch = cu_seqlens_q.len() - 1;
        // Validate the cumulative arrays: start at 0, non-decreasing, and end at
        // the packed token totals so no in-kernel index can fall out of bounds.
        let validate = |cu: &[i32], total: usize, name: &str| -> Result<usize> {
            if cu[0] != 0 {
                return Err(error(format!("{name}[0] must be 0, got {}", cu[0])));
            }
            let mut max_len = 0usize;
            for b in 0..batch {
                let lo = cu[b];
                let hi = cu[b + 1];
                if hi < lo {
                    return Err(error(format!(
                        "{name} must be non-decreasing, but element {b} ({lo}) > {} ({hi})",
                        b + 1
                    )));
                }
                max_len = max_len.max((hi - lo) as usize);
            }
            let end = cu[batch];
            if end as usize != total {
                return Err(error(format!(
                    "{name}[{batch}] = {end} must equal the packed token count {total}"
                )));
            }
            Ok(max_len)
        };
        validate(&cu_seqlens_q, total_q, "cu_seqlens_q")?;
        let max_kv_len = validate(&cu_seqlens_kv, total_kv, "cu_seqlens_kv")?;

        // Output must match the packed [total_q, num_heads, v_head_size] extent.
        let y_expected = total_q * num_heads * v_head_size;
        if outputs[0].dtype != dtype
            || !outputs[0].is_contiguous()
            || outputs[0].numel() != y_expected
        {
            return Err(error(
                "output must be contiguous, use the Q/K/V dtype, and have shape [total_q, num_heads, v_head_size]",
            ));
        }

        crate::trace::record_kernel_metrics(inputs, outputs, || {
            // 2 * QK + 2 * PV flops over each sequence's valid (i, j) pairs.
            let mut pairs = 0u64;
            for b in 0..batch {
                let lq = (cu_seqlens_q[b + 1] - cu_seqlens_q[b]) as u64;
                let lkv = (cu_seqlens_kv[b + 1] - cu_seqlens_kv[b]) as u64;
                pairs = pairs.saturating_add(lq.saturating_mul(lkv));
            }
            pairs
                .saturating_mul(num_heads as u64)
                .saturating_mul((head_size + v_head_size) as u64)
                .saturating_mul(2)
        });

        if total_q == 0 || y_expected == 0 {
            return Ok(());
        }

        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());
        // Fold sqrt(scale) into each Q and K operand: (Q.√scale)·(K.√scale).
        let sqrt_scale = scale.sqrt();

        let q_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let k_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let v_ptr = cuptr(inputs[2].data_ptr::<u8>() as *const c_void);
        let cu_q_ptr = cuptr(inputs[3].data_ptr::<u8>() as *const c_void);
        let cu_kv_ptr = cuptr(inputs[4].data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let _ = element_bytes;

        // Host-built per-token batch label so each block resolves its sequence in
        // O(1) without a device search. Bulk data stays on the device; only this
        // small control array is uploaded.
        let mut q_batch_ids = vec![0i32; total_q];
        for b in 0..batch {
            let lo = cu_seqlens_q[b] as usize;
            let hi = cu_seqlens_q[b + 1] as usize;
            for slot in &mut q_batch_ids[lo..hi] {
                *slot = b as i32;
            }
        }

        let total_rows = (total_q * num_heads) as u64;
        let scores_elems = total_rows.saturating_mul(max_kv_len.max(1) as u64);

        let mut owned: Vec<CUdeviceptr> = Vec::new();
        let result = (|| -> Result<()> {
            let alloc = |bytes: usize, owned: &mut Vec<CUdeviceptr>| -> Result<CUdeviceptr> {
                let ptr = self.runtime.alloc_raw(bytes.max(1))?;
                owned.push(ptr);
                Ok(ptr)
            };
            let q_batch_ids_ptr = alloc(total_q * 4, &mut owned)?;
            let scores_ptr = alloc(scores_elems as usize * 4, &mut owned)?;
            // SAFETY: `q_batch_ids_ptr` covers `total_q` int32 slots exactly.
            unsafe {
                let bytes = std::slice::from_raw_parts(
                    q_batch_ids.as_ptr().cast::<u8>(),
                    q_batch_ids.len() * 4,
                );
                self.runtime.htod(bytes, q_batch_ids_ptr)?;
            }

            let caps = self.runtime.capabilities();
            let block_threads = ROW_THREADS.min(caps.max_threads_per_block().max(1));
            // Device-cap-driven grid: enough resident blocks to fill the SMs,
            // capped by the actual row count; the kernel grid-strides the rest.
            let grid_cap = caps
                .multiprocessor_count()
                .saturating_mul(BLOCKS_PER_SM)
                .max(1);
            let grid_blocks = (total_rows.min(grid_cap as u64)).max(1) as u32;

            let func = self.runtime.nvrtc_function(MODULE, SOURCE, ENTRY)?;
            let num_heads_u = num_heads as u64;
            let kv_heads_u = kv_num_heads as u64;
            let head_size_u = head_size as u64;
            let v_head_size_u = v_head_size as u64;
            let group_u = group as u64;
            let total_q_u = total_q as u64;
            let max_kv_len_u = max_kv_len as u64;
            let is_causal_i = i32::from(self.is_causal);
            let softcap = self.softcap;
            let mut builder = self.runtime.stream().launch_builder(&func);
            builder
                .arg(&q_ptr)
                .arg(&k_ptr)
                .arg(&v_ptr)
                .arg(&cu_q_ptr)
                .arg(&cu_kv_ptr)
                .arg(&q_batch_ids_ptr)
                .arg(&scores_ptr)
                .arg(&y_ptr)
                .arg(&num_heads_u)
                .arg(&kv_heads_u)
                .arg(&head_size_u)
                .arg(&v_head_size_u)
                .arg(&group_u)
                .arg(&total_q_u)
                .arg(&max_kv_len_u)
                .arg(&dtype_code)
                .arg(&is_causal_i)
                .arg(&sqrt_scale)
                .arg(&softcap);
            // SAFETY: all pointers are live device allocations validated above and
            // the scalar ABI matches `packed_varlen_attention_row`.
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid_blocks, 1, 1),
                    block_dim: (block_threads, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|err| driver_err("launch packed_varlen_attention_row", err))?;
            Ok(())
        })();

        let sync_result = if result.is_ok() {
            self.runtime.synchronize()
        } else {
            Ok(())
        };
        let mut free_result = Ok(());
        for ptr in owned {
            // SAFETY: every pointer came from this runtime's `alloc_raw` above.
            let freed = unsafe { self.runtime.free_raw(ptr) };
            if free_result.is_ok() {
                free_result = freed;
            }
        }
        result.and(sync_result).and(free_result)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "PackedVarlenAttention reads cu_seqlens off-device and performs a trailing stream synchronize",
        )
    }
}

//! Unpadded-compute variable-length attention driven by ONNX Attention-24
//! `nonpad_kv_seqlen` (`pkg.nxrt::VarlenAttention` v1).
//!
//! This is a runtime-invented op (registered in [`onnx_runtime_ir::RUNTIME_DOMAIN`]
//! = `pkg.nxrt`, **not** `com.microsoft` or the default ONNX domain). It runs
//! scaled dot-product attention over a *padded rectangular* ragged batch
//! (`[batch, seq, ...]`, the layout continuous/ragged batching produces today)
//! but consumes the opset-24 `nonpad_kv_seqlen` per-batch valid-token count to
//! iterate **only the real KV tokens** of each sequence. Where the standard
//! `ai.onnx::Attention` kernel computes every `Q·Kᵀ` score for the full padded
//! `kv_seq` and then NEG_INF-masks the `j >= nonpad_kv_seqlen[b]` tail, this
//! kernel bounds its key loop at `nonpad_kv_seqlen[b]`, so a skewed-length batch
//! spends no compute on padded keys.
//!
//! ## Relationship to `pkg.nxrt::PackedVarlenAttention` (#86)
//!
//! [`super::packed_varlen_attention`] is the *packed* half of #86: it takes a
//! padding-free token layout with explicit `cu_seqlens` cumulative offsets. This
//! op is the *padded-in / unpadded-compute* half: it takes the rectangular batch
//! that ragged prefill already materializes and the ONNX `nonpad_kv_seqlen`
//! that describes it, and derives each sequence's valid length internally — the
//! per-batch valid KV length is the varlen descriptor. The two are complementary
//! entry points into the same unpadded attention math and both advance #86.
//!
//! ## Schema (v1)
//!
//! * inputs
//!   0. `query` — padded Q, `[batch, q_seq, num_heads, head_size]` (rank 4) or `[batch, q_seq, num_heads*head_size]` (rank 3).
//!   1. `key`   — padded K, `[batch, kv_seq, kv_num_heads, head_size]` or rank 3.
//!   2. `value` — padded V, `[batch, kv_seq, kv_num_heads, v_head_size]` or rank 3.
//!   3. `nonpad_kv_seqlen` — int64 `[batch]`, per-batch valid (non-padding) KV token count, `0 <= nonpad_kv_seqlen[b] <= kv_seq` (ONNX Attention-24 semantics).
//! * output
//!   0. `output` — `[batch, q_seq, num_heads, v_head_size]` (rank matches Q).
//! * attributes
//!   * `num_heads` (int, required) — query heads.
//!   * `kv_num_heads` (int, optional) — defaults to `num_heads`; must divide it (MHA/GQA/MQA head sharing).
//!   * `scale` (float, optional) — defaults `1/sqrt(head_size)`.
//!   * `is_causal` (int, optional, default 0) — tail-aligned causal mask.
//!   * `softcap` (float, optional, default 0) — `softcap·tanh(score/softcap)`.
//!
//! ## Causal alignment
//!
//! Query local position `i` (0-based within the batch's `q_seq`) attends key
//! position `jk` iff `jk <= i + (nonpad_kv_seqlen[b] - q_seq)`. The
//! `nonpad_kv_seqlen[b] - q_seq` offset tail-aligns the query block against the
//! valid key block, matching exactly the `offset = nonpad_kv_seqlen[b] - q_seq`
//! the standard `Attention` kernel applies. A negative frontier fully masks a
//! leading query row (→ zero output row).
//!
//! ## Numerics & determinism
//!
//! One CUDA block services one `(batch, query token, query head)` row. Scores
//! are kept in fp32 (f16/bf16 are converted on load/store around fp32
//! accumulators), `sqrt(scale)` is folded into each Q and K operand, and a
//! single lead thread performs the softmax max/exp/sum in ascending key order —
//! bit-identical to the standard `Attention` kernel, so this kernel's valid
//! output matches the padded `Attention`-with-`nonpad_kv_seqlen` reference.
//!
//! ## Correctness-first scope
//!
//! v1 materializes each row's scores in a device scratch buffer (sized
//! `total_rows * max_valid_kv`) rather than a register-tiled streaming softmax.
//! Launch geometry is derived from the live device's `multiprocessor_count` /
//! `max_threads_per_block` (no hardcoded per-GPU constants). Flash-style tiling
//! is a documented follow-up (Advances #86).

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "VarlenAttention";
const DOMAIN: &str = onnx_runtime_ir::RUNTIME_DOMAIN;
const MODULE: &str = "varlen_attention_f32_f16_bf16_v1";
const ENTRY: &str = "varlen_attention_row";
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

// One block per (batch, padded query token, query head). Q/K/V share a padded
// [batch, seq, head, dim] flat layout, so a rank-3 [batch, seq, head*dim] input
// indexes identically. `nonpad_kv` holds each batch's valid KV length, so the
// key loop bound skips padding entirely (no compute on masked-out keys).
extern "C" __global__ void varlen_attention_row(
    const void* q, const void* key, const void* value,
    const int* nonpad_kv,
    float* scores, void* y,
    unsigned long long num_heads, unsigned long long kv_heads,
    unsigned long long head_size, unsigned long long v_head_size,
    unsigned long long group, unsigned long long batch,
    unsigned long long q_seq, unsigned long long kv_seq,
    unsigned long long max_kv_len,
    int dtype, int is_causal, float sqrt_scale, float softcap) {
  const unsigned long long total_rows = batch * q_seq * num_heads;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  __shared__ float inv_sum_sh;
  __shared__ int all_masked_sh;

  for (unsigned long long row = blockIdx.x; row < total_rows; row += (unsigned long long)gridDim.x) {
    const unsigned long long gq = row / num_heads;   // global (batch, q token) index
    const unsigned long long qh = row % num_heads;   // query head
    const unsigned long long b = gq / q_seq;         // batch
    const long long i = (long long)(gq % q_seq);     // query position within its batch
    const long long valid_kv = (long long)nonpad_kv[b];
    const unsigned long long kvh = qh / group;
    const unsigned long long srow = row * max_kv_len;
    // Tail-aligned causal frontier: query i attends key jk iff jk <= i + offset,
    // with offset = nonpad_kv_seqlen[b] - q_seq (matches the standard kernel).
    const long long causal_limit = i + (valid_kv - (long long)q_seq);

    // Stage 1: scaled Q.Kᵀ scores over only this batch's VALID keys (sqrt(scale)
    // folded into each operand so extreme magnitudes don't overflow the dot).
    const unsigned long long qoff = (gq * num_heads + qh) * head_size;
    for (long long jk = tid; jk < valid_kv; jk += nthreads) {
      const unsigned long long gk = b * kv_seq + (unsigned long long)jk;
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
      for (long long jk = tid; jk < valid_kv; jk += nthreads) {
        const float s = scores[srow + (unsigned long long)jk];
        scores[srow + (unsigned long long)jk] = softcap * tanhf(s / softcap);
      }
      __syncthreads();
    }

    // Stage 3: causal frontier. Padded keys are already excluded by the valid_kv
    // loop bound, so this is the only remaining mask.
    if (is_causal) {
      for (long long jk = tid; jk < valid_kv; jk += nthreads) {
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
      for (long long jk = 0; jk < valid_kv; ++jk) {
        m = fmaxf(m, scores[srow + (unsigned long long)jk]);
      }
      if (m == NEG_INF) {
        all_masked_sh = 1;
        inv_sum_sh = 0.0f;
      } else {
        all_masked_sh = 0;
        float sum = 0.0f;
        for (long long jk = 0; jk < valid_kv; ++jk) {
          const float e = expf(scores[srow + (unsigned long long)jk] - m);
          scores[srow + (unsigned long long)jk] = e;
          sum += e;
        }
        inv_sum_sh = 1.0f / sum;
      }
    }
    __syncthreads();
    if (all_masked_sh) {
      for (long long jk = tid; jk < valid_kv; jk += nthreads) {
        scores[srow + (unsigned long long)jk] = 0.0f;
      }
    } else {
      const float inv = inv_sum_sh;
      for (long long jk = tid; jk < valid_kv; jk += nthreads) {
        scores[srow + (unsigned long long)jk] *= inv;
      }
    }
    __syncthreads();

    // Stage 5: Y = probs . V over the valid keys. Each thread owns whole output
    // channels and sums in ascending key order (bit-identical to the reference).
    // A fully-masked row (valid_kv == 0 or causal frontier below 0) writes zeros.
    const unsigned long long ybase = (gq * num_heads + qh) * v_head_size;
    for (unsigned long long c = tid; c < v_head_size; c += nthreads) {
      float acc = 0.0f;
      for (long long jk = 0; jk < valid_kv; ++jk) {
        const unsigned long long gk = b * kv_seq + (unsigned long long)jk;
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

/// Claim-time denial for the varlen positional/dtype contract. Mirrors the
/// other `pkg.nxrt` ops: reject unsupported dtypes up front so the planner falls
/// back cleanly instead of failing at execute.
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
                "VarlenAttention: dtype {name} not supported on CUDA (Q/K/V must be f32, f16, or bf16)"
            )));
        }
    }
    if dtype_at(1) != dtype_at(0) || dtype_at(2) != dtype_at(0) {
        return Some(Cow::Borrowed(
            "VarlenAttention: Q, K, and V must use the same floating dtype on CUDA",
        ));
    }
    let nonpad = dtype_at(3);
    if nonpad != DataType::Undefined && nonpad != DataType::Int64 {
        return Some(Cow::Owned(format!(
            "VarlenAttention: nonpad_kv_seqlen dtype {nonpad:?} not supported (expected int64)"
        )));
    }
    if node.attr("num_heads").and_then(|a| a.as_int()).is_none() {
        return Some(Cow::Borrowed(
            "VarlenAttention: missing required int attribute 'num_heads'",
        ));
    }
    None
}

/// Factory reading the varlen attention attributes.
pub struct VarlenAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for VarlenAttentionFactory {
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
        Ok(Box::new(VarlenAttentionKernel {
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
struct VarlenAttentionKernel {
    runtime: Arc<CudaRuntime>,
    num_heads: usize,
    kv_num_heads: Option<usize>,
    scale: Option<f32>,
    is_causal: bool,
    softcap: f32,
}

/// Resolved `[batch, seq, dim]` extents of a padded Q/K/V input. A rank-3
/// `[batch, seq, heads*dim]` input reshapes into heads via `heads`.
struct PaddedDims {
    seq: usize,
    dim: usize,
}

fn resolve_padded(view: &TensorView, name: &str, heads: usize, batch: usize) -> Result<PaddedDims> {
    if !view.is_contiguous() {
        return Err(error(format!("{name} must be contiguous on CUDA")));
    }
    if heads == 0 {
        return Err(error(format!("{name} head count must be > 0")));
    }
    let (seq, dim) = match view.shape.len() {
        4 => {
            if view.shape[2] != heads {
                return Err(error(format!(
                    "{name} rank-4 head dim {} must equal head count {heads}",
                    view.shape[2]
                )));
            }
            (view.shape[1], view.shape[3])
        }
        3 => {
            let hidden = view.shape[2];
            if !hidden.is_multiple_of(heads) {
                return Err(error(format!(
                    "{name} rank-3 hidden size {hidden} is not divisible by head count {heads}"
                )));
            }
            (view.shape[1], hidden / heads)
        }
        other => {
            return Err(error(format!(
                "{name} must be rank 3 or 4, got rank {other}"
            )));
        }
    };
    if view.shape[0] != batch {
        return Err(error(format!(
            "{name} batch dim {} must equal nonpad_kv_seqlen length {batch}",
            view.shape[0]
        )));
    }
    Ok(PaddedDims { seq, dim })
}

/// Read the contiguous int64 `nonpad_kv_seqlen` array off the device to the
/// host. Only this tiny per-batch control array (length `batch`) round-trips;
/// the bulk Q/K/V tensors stay resident.
fn read_nonpad(runtime: &CudaRuntime, view: &TensorView) -> Result<Vec<i64>> {
    if view.dtype != DataType::Int64 {
        return Err(error("nonpad_kv_seqlen must be int64"));
    }
    if !view.is_contiguous() {
        return Err(error("nonpad_kv_seqlen must be contiguous on CUDA"));
    }
    if view.shape.len() != 1 || view.shape[0] == 0 {
        return Err(error(format!(
            "nonpad_kv_seqlen must be a 1D tensor of length batch_size (>= 1), got shape {:?}",
            view.shape
        )));
    }
    let count = view.shape[0];
    let mut bytes = vec![0u8; count * 8];
    // SAFETY: `view` is a live, contiguous int64 device allocation of `count`
    // elements; the destination matches its byte length exactly.
    unsafe {
        runtime.dtoh(&mut bytes, cuptr(view.data_ptr::<u8>() as *const c_void))?;
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|b| i64::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}

impl Kernel for VarlenAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 4 || outputs.len() != 1 {
            return Err(error(format!(
                "expected 4 inputs and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        let dtype = inputs[0].dtype;
        let dtype_code = match dtype {
            DataType::Float32 => 0i32,
            DataType::Float16 => 1,
            DataType::BFloat16 => 2,
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

        let nonpad = read_nonpad(&self.runtime, &inputs[3])?;
        let batch = nonpad.len();

        let q_dims = resolve_padded(&inputs[0], "query", num_heads, batch)?;
        let k_dims = resolve_padded(&inputs[1], "key", kv_num_heads, batch)?;
        let v_dims = resolve_padded(&inputs[2], "value", kv_num_heads, batch)?;
        let q_seq = q_dims.seq;
        let kv_seq = k_dims.seq;
        let head_size = q_dims.dim;
        let v_head_size = v_dims.dim;
        if k_dims.dim != head_size {
            return Err(error(format!(
                "query head_size {head_size} != key head_size {}",
                k_dims.dim
            )));
        }
        if v_dims.seq != kv_seq {
            return Err(error(format!(
                "key kv_seq {kv_seq} != value kv_seq {}",
                v_dims.seq
            )));
        }

        // Validate the per-batch valid KV lengths and size the score scratch by
        // the widest valid sequence so no in-kernel index escapes its row.
        let mut max_valid_kv = 0usize;
        for (b, &n) in nonpad.iter().enumerate() {
            if n < 0 || n as usize > kv_seq {
                return Err(error(format!(
                    "nonpad_kv_seqlen[{b}] = {n} must be in [0, kv_seq={kv_seq}]"
                )));
            }
            max_valid_kv = max_valid_kv.max(n as usize);
        }

        let y_expected = batch * q_seq * num_heads * v_head_size;
        if outputs[0].dtype != dtype
            || !outputs[0].is_contiguous()
            || outputs[0].numel() != y_expected
        {
            return Err(error(
                "output must be contiguous, use the Q/K/V dtype, and have shape [batch, q_seq, num_heads, v_head_size]",
            ));
        }

        crate::trace::record_kernel_metrics(inputs, outputs, || {
            // 2 * QK + 2 * PV flops over each row's VALID (i, j) pairs.
            let mut pairs = 0u64;
            for &n in &nonpad {
                let valid = n.max(0) as u64;
                pairs = pairs.saturating_add((q_seq as u64).saturating_mul(valid));
            }
            pairs
                .saturating_mul(num_heads as u64)
                .saturating_mul((head_size + v_head_size) as u64)
                .saturating_mul(2)
        });

        if y_expected == 0 || batch == 0 || q_seq == 0 {
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
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        // Host-narrowed int32 valid-length control array (bounds validated above
        // to fit i32). Bulk data stays on the device; only this small array is
        // uploaded.
        let nonpad_i32: Vec<i32> = nonpad.iter().map(|&n| n as i32).collect();

        let total_rows = (batch * q_seq * num_heads) as u64;
        let stride = max_valid_kv.max(1) as u64;
        let scores_elems = total_rows.saturating_mul(stride);

        let mut owned: Vec<CUdeviceptr> = Vec::new();
        let result = (|| -> Result<()> {
            let alloc = |bytes: usize, owned: &mut Vec<CUdeviceptr>| -> Result<CUdeviceptr> {
                let ptr = self.runtime.alloc_raw(bytes.max(1))?;
                owned.push(ptr);
                Ok(ptr)
            };
            let nonpad_ptr = alloc(batch * 4, &mut owned)?;
            let scores_ptr = alloc(scores_elems as usize * 4, &mut owned)?;
            // SAFETY: `nonpad_ptr` covers `batch` int32 slots exactly.
            unsafe {
                let bytes = std::slice::from_raw_parts(
                    nonpad_i32.as_ptr().cast::<u8>(),
                    nonpad_i32.len() * 4,
                );
                self.runtime.htod(bytes, nonpad_ptr)?;
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
            let batch_u = batch as u64;
            let q_seq_u = q_seq as u64;
            let kv_seq_u = kv_seq as u64;
            let max_kv_len_u = stride;
            let is_causal_i = i32::from(self.is_causal);
            let softcap = self.softcap;
            let mut builder = self.runtime.stream().launch_builder(&func);
            builder
                .arg(&q_ptr)
                .arg(&k_ptr)
                .arg(&v_ptr)
                .arg(&nonpad_ptr)
                .arg(&scores_ptr)
                .arg(&y_ptr)
                .arg(&num_heads_u)
                .arg(&kv_heads_u)
                .arg(&head_size_u)
                .arg(&v_head_size_u)
                .arg(&group_u)
                .arg(&batch_u)
                .arg(&q_seq_u)
                .arg(&kv_seq_u)
                .arg(&max_kv_len_u)
                .arg(&dtype_code)
                .arg(&is_causal_i)
                .arg(&sqrt_scale)
                .arg(&softcap);
            // SAFETY: all pointers are live device allocations validated above and
            // the scalar ABI matches `varlen_attention_row`.
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid_blocks, 1, 1),
                    block_dim: (block_threads, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|err| driver_err("launch varlen_attention_row", err))?;
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
            "VarlenAttention reads nonpad_kv_seqlen off-device and performs a trailing stream synchronize",
        )
    }
}

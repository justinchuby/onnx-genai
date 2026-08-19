//! `com.microsoft::LinearAttention` — gated delta-rule linear attention
//! (Gated DeltaNet, as used by the Qwen3.5 / Qwen3-Next hybrid family) with a
//! per-head recurrent state matrix.
//!
//! Faithful CUDA port of the CPU EP kernel
//! (`onnx-runtime-ep-cpu/src/kernels/linear_attention.rs`, itself a port of
//! ORT's `contrib_ops/cpu/bert/linear_attention.cc`). See the design note
//! `.squad/decisions/inbox/cohaagen-linear-attention-design.md` for the full
//! contract.
//!
//! ## Parallelization
//!
//! Column `j` of the per-head state matrix `S[d_k, d_v]` evolves independently
//! of every other column (retrieval `r[j] = Σ_i S[i,j]·k[i]`, the delta/linear
//! update and the readout `o[j] = scale·Σ_i q[i]·S[i,j]` all touch only column
//! `j`). So the op is embarrassingly parallel across `(b, h_kv, j)`; each is one
//! sequential scan over `t`. One thread owns one column, keeping the column in a
//! per-thread **f32** local array so the recurrent state stays in f32 for the
//! whole scan — reproducing ORT's `float` kernel exactly regardless of the I/O
//! dtype (`f16`/`bf16` are widened on read, narrowed on write).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;

/// Largest `d_k` the per-thread f32 state column can hold. Real Qwen3.5 hybrid
/// heads use `d_k = 128`; the claim gate rejects anything larger so we never
/// claim an op this kernel cannot run.
const MAX_D_K: usize = 256;

/// Private markers set by `CudaLinearAttentionGatingFusion` (see `optimizer.rs`)
/// on a `LinearAttention` node whose standalone gate chains have been folded
/// into this kernel. Consumed only by the CUDA EP factory above, so a node the
/// pass never touched keeps the exact exported (unfused) arithmetic.
pub(crate) const FUSE_BETA_SIGMOID_ATTR: &str = "com.microsoft.cuda_fuse_beta_sigmoid";
pub(crate) const FUSE_DECAY_SOFTPLUS_ATTR: &str = "com.microsoft.cuda_fuse_decay_softplus";

const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define MAX_D_K 256

__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_f_val(float x, float*) { return x; }
__device__ __forceinline__ __half from_f_val(float x, __half*) { return __float2half_rn(x); }
__device__ __forceinline__ __nv_bfloat16 from_f_val(float x, __nv_bfloat16*) {
  return __float2bfloat16_rn(x);
}

// Round `x` through the storage dtype `T` and widen back to f32. Reproduces the
// per-op narrow rounding a standalone `Sigmoid`/`Softplus`/`Add`/`Mul` kernel
// applies at each fused gate's boundary, so a folded gate stays byte-identical
// for narrow (f16/bf16) I/O and is an exact no-op for f32.
template <typename T>
__device__ __forceinline__ float round_store(float x) {
  return to_f(from_f_val(x, (T*)0));
}

// Byte-exact port of the standalone `Sigmoid` op device function
// (`kernels/elementwise.rs::op_sigmoid`): the `exp` is evaluated in double, so a
// folded `beta = Sigmoid(x)` gate reproduces the unfused kernel bit-for-bit.
__device__ __forceinline__ float la_sigmoid(float x) {
  if (x >= 0.0f) return 1.0f / (1.0f + (float)exp((double)-x));
  const float e = (float)exp((double)x);
  return e / (1.0f + e);
}

// Byte-exact port of the standalone `Softplus` op device function
// (`kernels/pointwise.rs::op_softplus`).
__device__ __forceinline__ float la_softplus(float x) {
  return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

template <typename T>
__device__ void linear_attention_core(
    const T* q, const T* k, const T* v,
    const T* past_state, const T* decay, const T* beta,
    const T* dt_bias, const T* neg_exp_A,
    T* output, T* present_state,
    unsigned long long batch, unsigned long long seq,
    unsigned long long d_k, unsigned long long d_v,
    unsigned long long q_num_heads, unsigned long long kv_num_heads,
    unsigned long long n_k_heads, unsigned long long heads_per_group,
    unsigned long long kv_per_k_head, unsigned long long output_hidden,
    float scale, int needs_decay, int decay_per_key_dim,
    int needs_delta, int beta_per_head,
    int fuse_beta_sigmoid, int fuse_decay_softplus) {
  const unsigned long long total = batch * kv_num_heads * d_v;
  const unsigned long long stride =
      (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long tid =
           (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       tid < total; tid += stride) {
  const unsigned long long j = tid % d_v;
  const unsigned long long hk_flat = tid / d_v;  // b * kv_num_heads + h_kv
  const unsigned long long b = hk_flat / kv_num_heads;
  const unsigned long long h_kv = hk_flat % kv_num_heads;
  const unsigned long long h_k = h_kv / kv_per_k_head;
  const unsigned long long sbase = hk_flat * d_k * d_v;  // + i*d_v + j

  float sc[MAX_D_K];
  for (unsigned long long i = 0; i < d_k; ++i) {
    sc[i] = past_state ? to_f(past_state[sbase + i * d_v + j]) : 0.0f;
  }

  for (unsigned long long t = 0; t < seq; ++t) {
    const unsigned long long row = b * seq + t;

    // Step 1: decay  S *= exp(g_t)
    if (needs_decay) {
      if (decay_per_key_dim) {
        const T* g = decay + row * (kv_num_heads * d_k) + h_kv * d_k;
        if (fuse_decay_softplus) {
          const T* dtb = dt_bias + h_kv * d_k;
          const T* na = neg_exp_A + h_kv * d_k;
          for (unsigned long long i = 0; i < d_k; ++i) {
            float a = round_store<T>(to_f(g[i]) + to_f(dtb[i]));
            a = round_store<T>(la_softplus(a));
            a = round_store<T>(to_f(na[i]) * a);
            sc[i] *= expf(a);
          }
        } else {
          for (unsigned long long i = 0; i < d_k; ++i) sc[i] *= expf(to_f(g[i]));
        }
      } else {
        float g_val;
        if (fuse_decay_softplus) {
          float a = round_store<T>(
              to_f(decay[row * kv_num_heads + h_kv]) + to_f(dt_bias[h_kv]));
          a = round_store<T>(la_softplus(a));
          g_val = round_store<T>(to_f(neg_exp_A[h_kv]) * a);
        } else {
          g_val = to_f(decay[row * kv_num_heads + h_kv]);
        }
        const float eg = expf(g_val);
        for (unsigned long long i = 0; i < d_k; ++i) sc[i] *= eg;
      }
    }

    const float vt = to_f(v[row * (kv_num_heads * d_v) + h_kv * d_v + j]);
    const T* kt = k + row * (n_k_heads * d_k) + h_k * d_k;

    if (needs_delta) {
      // Step 2: retrieval r = Sᵀ k_t  (over d_k)
      float r = 0.0f;
      for (unsigned long long i = 0; i < d_k; ++i) r += sc[i] * to_f(kt[i]);
      // Step 3: delta update  S += k_t ⊗ (beta·(v_t − r))
      float bt = beta_per_head ? to_f(beta[row * kv_num_heads + h_kv]) : to_f(beta[row]);
      if (fuse_beta_sigmoid) bt = round_store<T>(la_sigmoid(bt));
      const float d = bt * (vt - r);
      for (unsigned long long i = 0; i < d_k; ++i) sc[i] += to_f(kt[i]) * d;
    } else {
      // linear / gated: S += k_t ⊗ v_t
      for (unsigned long long i = 0; i < d_k; ++i) sc[i] += to_f(kt[i]) * vt;
    }

    // Step 4: readout o_t = scale · q_tᵀ S  (updated S)
    if (heads_per_group > 0) {
      for (unsigned long long g = 0; g < heads_per_group; ++g) {
        const unsigned long long h_q = h_kv * heads_per_group + g;
        const T* qt = q + row * (q_num_heads * d_k) + h_q * d_k;
        float o = 0.0f;
        for (unsigned long long i = 0; i < d_k; ++i) o += to_f(qt[i]) * sc[i];
        output[row * output_hidden + h_q * d_v + j] = from_f_val(o * scale, output);
      }
    } else {
      // Inverse GQA: output slot is h_kv, query head h_kv·H_q/H_kv.
      const unsigned long long h_q = h_kv * q_num_heads / kv_num_heads;
      const T* qt = q + row * (q_num_heads * d_k) + h_q * d_k;
      float o = 0.0f;
      for (unsigned long long i = 0; i < d_k; ++i) o += to_f(qt[i]) * sc[i];
      output[row * output_hidden + h_kv * d_v + j] = from_f_val(o * scale, output);
    }
  }

  for (unsigned long long i = 0; i < d_k; ++i) {
    present_state[sbase + i * d_v + j] = from_f_val(sc[i], present_state);
  }
  }
}

extern "C" __global__ void linear_attention_f32(
    const float* q, const float* k, const float* v,
    const float* past_state, const float* decay, const float* beta,
    const float* dt_bias, const float* neg_exp_A,
    float* output, float* present_state,
    unsigned long long batch, unsigned long long seq,
    unsigned long long d_k, unsigned long long d_v,
    unsigned long long q_num_heads, unsigned long long kv_num_heads,
    unsigned long long n_k_heads, unsigned long long heads_per_group,
    unsigned long long kv_per_k_head, unsigned long long output_hidden,
    float scale, int needs_decay, int decay_per_key_dim,
    int needs_delta, int beta_per_head,
    int fuse_beta_sigmoid, int fuse_decay_softplus) {
  linear_attention_core<float>(
      q, k, v, past_state, decay, beta, dt_bias, neg_exp_A, output,
      present_state, batch, seq, d_k, d_v, q_num_heads, kv_num_heads, n_k_heads,
      heads_per_group, kv_per_k_head, output_hidden, scale, needs_decay,
      decay_per_key_dim, needs_delta, beta_per_head, fuse_beta_sigmoid,
      fuse_decay_softplus);
}

extern "C" __global__ void linear_attention_f16(
    const __half* q, const __half* k, const __half* v,
    const __half* past_state, const __half* decay, const __half* beta,
    const __half* dt_bias, const __half* neg_exp_A,
    __half* output, __half* present_state,
    unsigned long long batch, unsigned long long seq,
    unsigned long long d_k, unsigned long long d_v,
    unsigned long long q_num_heads, unsigned long long kv_num_heads,
    unsigned long long n_k_heads, unsigned long long heads_per_group,
    unsigned long long kv_per_k_head, unsigned long long output_hidden,
    float scale, int needs_decay, int decay_per_key_dim,
    int needs_delta, int beta_per_head,
    int fuse_beta_sigmoid, int fuse_decay_softplus) {
  linear_attention_core<__half>(
      q, k, v, past_state, decay, beta, dt_bias, neg_exp_A, output,
      present_state, batch, seq, d_k, d_v, q_num_heads, kv_num_heads, n_k_heads,
      heads_per_group, kv_per_k_head, output_hidden, scale, needs_decay,
      decay_per_key_dim, needs_delta, beta_per_head, fuse_beta_sigmoid,
      fuse_decay_softplus);
}

extern "C" __global__ void linear_attention_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* past_state, const __nv_bfloat16* decay,
    const __nv_bfloat16* beta,
    const __nv_bfloat16* dt_bias, const __nv_bfloat16* neg_exp_A,
    __nv_bfloat16* output,
    __nv_bfloat16* present_state,
    unsigned long long batch, unsigned long long seq,
    unsigned long long d_k, unsigned long long d_v,
    unsigned long long q_num_heads, unsigned long long kv_num_heads,
    unsigned long long n_k_heads, unsigned long long heads_per_group,
    unsigned long long kv_per_k_head, unsigned long long output_hidden,
    float scale, int needs_decay, int decay_per_key_dim,
    int needs_delta, int beta_per_head,
    int fuse_beta_sigmoid, int fuse_decay_softplus) {
  linear_attention_core<__nv_bfloat16>(
      q, k, v, past_state, decay, beta, dt_bias, neg_exp_A, output,
      present_state, batch, seq, d_k, d_v, q_num_heads, kv_num_heads, n_k_heads,
      heads_per_group, kv_per_k_head, output_hidden, scale, needs_decay,
      decay_per_key_dim, needs_delta, beta_per_head, fuse_beta_sigmoid,
      fuse_decay_softplus);
}
"#;

#[derive(Clone, Copy, PartialEq, Eq)]
enum UpdateRule {
    Linear,
    Gated,
    Delta,
    GatedDelta,
}

impl UpdateRule {
    fn parse(node: &Node) -> Result<Self> {
        match node.attr("update_rule").and_then(|a| a.as_str()) {
            Some("linear") => Ok(UpdateRule::Linear),
            Some("gated") => Ok(UpdateRule::Gated),
            Some("delta") => Ok(UpdateRule::Delta),
            None | Some("gated_delta") => Ok(UpdateRule::GatedDelta),
            Some(other) => Err(EpError::KernelFailed(format!(
                "LinearAttention: update_rule must be one of linear, gated, delta, \
                 gated_delta; got {other:?}"
            ))),
        }
    }
    fn needs_decay(self) -> bool {
        matches!(self, UpdateRule::Gated | UpdateRule::GatedDelta)
    }
    fn needs_delta(self) -> bool {
        matches!(self, UpdateRule::Delta | UpdateRule::GatedDelta)
    }
}

fn read_heads(node: &Node, name: &str) -> Option<usize> {
    node.attr(name)
        .and_then(|a| a.as_int())
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&v| v > 0)
}

/// Claim gate. Returns `Some(reason)` when the CUDA EP cannot run this node, so
/// placement declines cleanly instead of claiming then failing at execute time.
pub(crate) fn unsupported_reason(node: &Node, input_dtypes: &[DataType]) -> Option<String> {
    let q_num_heads = read_heads(node, "q_num_heads")?;
    let kv_num_heads = read_heads(node, "kv_num_heads")?;
    if UpdateRule::parse(node).is_err() {
        return Some("LinearAttention: unrecognized update_rule".into());
    }
    // query/key/value are required; the rest are optional trailing inputs.
    if input_dtypes.len() < 3 {
        return Some("LinearAttention: requires query, key, value inputs".into());
    }
    let dtype = input_dtypes[0];
    if !matches!(
        dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Some(format!(
            "LinearAttention dtype {dtype:?} (supported: Float32, Float16, BFloat16)"
        ));
    }
    if input_dtypes[..3.min(input_dtypes.len())]
        .iter()
        .any(|&d| d != dtype)
    {
        return Some("LinearAttention: query/key/value must share one dtype".into());
    }
    if q_num_heads >= kv_num_heads {
        if !q_num_heads.is_multiple_of(kv_num_heads) {
            return Some(format!(
                "LinearAttention: q_num_heads {q_num_heads} not a multiple of kv_num_heads \
                 {kv_num_heads}"
            ));
        }
    } else if !kv_num_heads.is_multiple_of(q_num_heads) {
        return Some(format!(
            "LinearAttention: kv_num_heads {kv_num_heads} not a multiple of q_num_heads \
             {q_num_heads} (inverse GQA)"
        ));
    }
    None
}

pub struct LinearAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for LinearAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let q_num_heads = read_heads(node, "q_num_heads").ok_or_else(|| {
            EpError::KernelFailed(
                "LinearAttention: `q_num_heads` must be a positive integer".into(),
            )
        })?;
        let kv_num_heads = read_heads(node, "kv_num_heads").ok_or_else(|| {
            EpError::KernelFailed(
                "LinearAttention: `kv_num_heads` must be a positive integer".into(),
            )
        })?;
        let update_rule = UpdateRule::parse(node)?;
        // ORT resolves a `0` (or missing) scale attribute to `1 / sqrt(d_k)`.
        let scale = match node.attr("scale").and_then(|a| a.as_float()) {
            Some(s) if s != 0.0 => Some(s),
            _ => None,
        };
        // Private markers set by `CudaLinearAttentionGatingFusion`: the CUDA EP
        // has folded the standalone `beta = Sigmoid(x)` and/or the decay
        // `exp(neg_exp_A · Softplus(a + dt_bias))` gate chain into this kernel.
        let fuse_beta_sigmoid = node
            .attr(FUSE_BETA_SIGMOID_ATTR)
            .and_then(|a| a.as_int())
            .is_some_and(|v| v != 0);
        let fuse_decay_softplus = node
            .attr(FUSE_DECAY_SOFTPLUS_ATTR)
            .and_then(|a| a.as_int())
            .is_some_and(|v| v != 0);
        Ok(Box::new(LinearAttentionKernel {
            runtime: self.runtime.clone(),
            q_num_heads,
            kv_num_heads,
            update_rule,
            scale,
            fuse_beta_sigmoid,
            fuse_decay_softplus,
        }))
    }
}

#[derive(Debug)]
struct LinearAttentionKernel {
    runtime: Arc<CudaRuntime>,
    q_num_heads: usize,
    kv_num_heads: usize,
    update_rule: UpdateRule,
    scale: Option<f32>,
    fuse_beta_sigmoid: bool,
    fuse_decay_softplus: bool,
}

impl std::fmt::Debug for UpdateRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            UpdateRule::Linear => "linear",
            UpdateRule::Gated => "gated",
            UpdateRule::Delta => "delta",
            UpdateRule::GatedDelta => "gated_delta",
        };
        f.write_str(name)
    }
}

impl Kernel for LinearAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() < 3 || inputs.len() > 8 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LinearAttention: expected 3..=8 inputs, got {}",
                inputs.len()
            )));
        }
        if outputs.len() != 2 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LinearAttention: expected 2 outputs (output, present_state), got {}",
                outputs.len()
            )));
        }

        let dtype = inputs[0].dtype;
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(not_implemented(format!(
                "LinearAttention dtype {dtype:?} (supported: Float32, Float16, BFloat16)"
            )));
        }

        let q = &inputs[0];
        let k = &inputs[1];
        let v = &inputs[2];
        if q.shape.len() != 3 || k.shape.len() != 3 || v.shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LinearAttention: query/key/value must be rank 3 [B, T, H·D], got {:?}, \
                 {:?}, {:?}",
                q.shape, k.shape, v.shape
            )));
        }
        let (batch, seq, q_hidden) = (q.shape[0], q.shape[1], q.shape[2]);
        if k.shape[0] != batch || v.shape[0] != batch || k.shape[1] != seq || v.shape[1] != seq {
            return Err(EpError::KernelFailed(
                "cuda_ep LinearAttention: query/key/value batch and sequence dims must agree"
                    .into(),
            ));
        }

        let q_num_heads = self.q_num_heads;
        let kv_num_heads = self.kv_num_heads;
        if q_num_heads == 0 || kv_num_heads == 0 || !q_hidden.is_multiple_of(q_num_heads) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LinearAttention: query hidden {q_hidden} not divisible by q_num_heads \
                 {q_num_heads}"
            )));
        }
        let d_k = q_hidden / q_num_heads;
        if d_k == 0 || d_k > MAX_D_K || !k.shape[2].is_multiple_of(d_k) {
            return Err(not_implemented(format!(
                "cuda_ep LinearAttention: d_k {d_k} unsupported (must be 1..={MAX_D_K} and divide \
                 key hidden {})",
                k.shape[2]
            )));
        }
        let n_k_heads = k.shape[2] / d_k;
        if n_k_heads == 0 || !v.shape[2].is_multiple_of(kv_num_heads) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LinearAttention: value hidden {} not divisible by kv_num_heads \
                 {kv_num_heads}",
                v.shape[2]
            )));
        }
        let d_v = v.shape[2] / kv_num_heads;

        let heads_per_group = if q_num_heads >= kv_num_heads {
            if !q_num_heads.is_multiple_of(kv_num_heads) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep LinearAttention: q_num_heads {q_num_heads} must be a multiple of \
                     kv_num_heads {kv_num_heads}"
                )));
            }
            q_num_heads / kv_num_heads
        } else {
            if !kv_num_heads.is_multiple_of(q_num_heads) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep LinearAttention: kv_num_heads {kv_num_heads} must be a multiple of \
                     q_num_heads {q_num_heads} (inverse GQA)"
                )));
            }
            0
        };
        if !kv_num_heads.is_multiple_of(n_k_heads) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LinearAttention: kv_num_heads {kv_num_heads} must be a multiple of \
                 n_k_heads {n_k_heads}"
            )));
        }
        let kv_per_k_head = kv_num_heads / n_k_heads;
        let scale = self.scale.unwrap_or_else(|| 1.0 / (d_k as f32).sqrt());

        let needs_decay = self.update_rule.needs_decay();
        let needs_delta = self.update_rule.needs_delta();

        let past_state = inputs.get(3);
        let decay = inputs.get(4);
        let beta = inputs.get(5);
        // Trailing gate-fusion operands, present only when the CUDA gating
        // fusion folded the decay `Softplus` chain into this kernel.
        let dt_bias = inputs.get(6);
        let neg_exp_a = inputs.get(7);

        if needs_decay && decay.is_none() {
            return Err(EpError::KernelFailed(
                "cuda_ep LinearAttention: decay input required for update_rule=gated/gated_delta"
                    .into(),
            ));
        }
        if needs_delta && beta.is_none() {
            return Err(EpError::KernelFailed(
                "cuda_ep LinearAttention: beta input required for update_rule=delta/gated_delta"
                    .into(),
            ));
        }
        let fuse_beta_sigmoid = self.fuse_beta_sigmoid && needs_delta;
        let fuse_decay_softplus = self.fuse_decay_softplus && needs_decay;
        if fuse_decay_softplus && (dt_bias.is_none() || neg_exp_a.is_none()) {
            return Err(EpError::KernelFailed(
                "cuda_ep LinearAttention: folded decay softplus requires dt_bias and neg_exp_A \
                 inputs"
                    .into(),
            ));
        }

        // decay layout: per-head (H_kv) or per-key-dim (H_kv · d_k).
        let decay_per_key_dim = if needs_decay {
            let s = decay.unwrap().shape;
            if s.len() != 3 || s[0] != batch || s[1] != seq {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep LinearAttention: decay must be [B={batch}, T={seq}, ...], got {s:?}"
                )));
            }
            if s[2] == kv_num_heads * d_k {
                true
            } else if s[2] == kv_num_heads {
                false
            } else {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep LinearAttention: decay last dim must be H_kv={kv_num_heads} or \
                     H_kv·d_k={}, got {}",
                    kv_num_heads * d_k,
                    s[2]
                )));
            }
        } else {
            false
        };

        // beta layout: per-head (H_kv) or shared (1).
        let beta_per_head = if needs_delta {
            let s = beta.unwrap().shape;
            if s.len() != 3 || s[0] != batch || s[1] != seq {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep LinearAttention: beta must be [B={batch}, T={seq}, ...], got {s:?}"
                )));
            }
            if s[2] == kv_num_heads {
                true
            } else if s[2] == 1 {
                false
            } else {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep LinearAttention: beta last dim must be H_kv={kv_num_heads} or 1, got {}",
                    s[2]
                )));
            }
        } else {
            false
        };

        // Folded decay operands index per-head (`dt_bias[h_kv]`) or per-key-dim
        // (`dt_bias[h_kv·d_k + i]`); require the matching element count so the
        // kernel never reads out of bounds regardless of head count / layout.
        if fuse_decay_softplus {
            let want = if decay_per_key_dim {
                kv_num_heads * d_k
            } else {
                kv_num_heads
            };
            for (name, view) in [
                ("dt_bias", dt_bias.unwrap()),
                ("neg_exp_A", neg_exp_a.unwrap()),
            ] {
                let n: usize = view.shape.iter().product();
                if n != want {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep LinearAttention: folded {name} must have {want} elements, got \
                         {:?}",
                        view.shape
                    )));
                }
            }
        }

        if let Some(view) = past_state {
            let s = view.shape;
            if s.len() != 4 || s[0] != batch || s[1] != kv_num_heads || s[2] != d_k || s[3] != d_v {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep LinearAttention: past_state must be [B={batch}, H_kv={kv_num_heads}, \
                     d_k={d_k}, d_v={d_v}], got {s:?}"
                )));
            }
        }

        let output_hidden = q_num_heads.max(kv_num_heads) * d_v;
        if outputs[0].shape != [batch, seq, output_hidden] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LinearAttention: output must be [B={batch}, T={seq}, {output_hidden}], \
                 got {:?}",
                outputs[0].shape
            )));
        }
        if outputs[1].shape != [batch, kv_num_heads, d_k, d_v] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LinearAttention: present_state must be [B={batch}, H_kv={kv_num_heads}, \
                 d_k={d_k}, d_v={d_v}], got {:?}",
                outputs[1].shape
            )));
        }

        // Uniform dtype + contiguity across every present tensor.
        let all_inputs_ok = inputs
            .iter()
            .all(|input| input.dtype == dtype && input.is_contiguous());
        if !all_inputs_ok
            || outputs[0].dtype != dtype
            || outputs[1].dtype != dtype
            || !outputs[0].is_contiguous()
            || !outputs[1].is_contiguous()
        {
            return Err(not_implemented(
                "LinearAttention requires contiguous, uniform-dtype tensors",
            ));
        }

        let total = (batch * kv_num_heads * d_v) as u64;
        if total == 0 || seq == 0 {
            // Nothing to compute; present_state is a byte-copy of past (or zeros).
            return Ok(());
        }

        let stem = match dtype {
            DataType::Float32 => "linear_attention_f32",
            DataType::Float16 => "linear_attention_f16",
            DataType::BFloat16 => "linear_attention_bf16",
            _ => unreachable!(),
        };
        if dtype != DataType::Float32 {
            self.runtime.require_nvrtc_half_headers("LinearAttention")?;
        }
        let function = self
            .runtime
            .nvrtc_function("linear_attention_v2", SOURCE, stem)?;

        let q_ptr = cuptr(q.data_ptr::<u8>() as *const c_void);
        let k_ptr = cuptr(k.data_ptr::<u8>() as *const c_void);
        let v_ptr = cuptr(v.data_ptr::<u8>() as *const c_void);
        let past_ptr = past_state
            .map(|t| cuptr(t.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let decay_ptr = if needs_decay {
            cuptr(decay.unwrap().data_ptr::<u8>() as *const c_void)
        } else {
            0
        };
        let beta_ptr = if needs_delta {
            cuptr(beta.unwrap().data_ptr::<u8>() as *const c_void)
        } else {
            0
        };
        let dt_bias_ptr = if fuse_decay_softplus {
            cuptr(dt_bias.unwrap().data_ptr::<u8>() as *const c_void)
        } else {
            0
        };
        let neg_exp_a_ptr = if fuse_decay_softplus {
            cuptr(neg_exp_a.unwrap().data_ptr::<u8>() as *const c_void)
        } else {
            0
        };
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let present_ptr = cuptr(outputs[1].data_ptr_mut::<u8>() as *const c_void);

        let batch = batch as u64;
        let seq = seq as u64;
        let d_k = d_k as u64;
        let d_v = d_v as u64;
        let q_num_heads = q_num_heads as u64;
        let kv_num_heads = kv_num_heads as u64;
        let n_k_heads = n_k_heads as u64;
        let heads_per_group = heads_per_group as u64;
        let kv_per_k_head = kv_per_k_head as u64;
        let output_hidden = output_hidden as u64;
        let needs_decay_i = i32::from(needs_decay);
        let decay_per_key_dim_i = i32::from(decay_per_key_dim);
        let needs_delta_i = i32::from(needs_delta);
        let beta_per_head_i = i32::from(beta_per_head);
        let fuse_beta_sigmoid_i = i32::from(fuse_beta_sigmoid);
        let fuse_decay_softplus_i = i32::from(fuse_decay_softplus);

        let grid = u32::try_from(total.div_ceil(BLOCK as u64))
            .unwrap_or(u32::MAX)
            .clamp(1, 65_535);
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&q_ptr)
            .arg(&k_ptr)
            .arg(&v_ptr)
            .arg(&past_ptr)
            .arg(&decay_ptr)
            .arg(&beta_ptr)
            .arg(&dt_bias_ptr)
            .arg(&neg_exp_a_ptr)
            .arg(&output_ptr)
            .arg(&present_ptr)
            .arg(&batch)
            .arg(&seq)
            .arg(&d_k)
            .arg(&d_v)
            .arg(&q_num_heads)
            .arg(&kv_num_heads)
            .arg(&n_k_heads)
            .arg(&heads_per_group)
            .arg(&kv_per_k_head)
            .arg(&output_hidden)
            .arg(&scale)
            .arg(&needs_decay_i)
            .arg(&decay_per_key_dim_i)
            .arg(&needs_delta_i)
            .arg(&beta_per_head_i)
            .arg(&fuse_beta_sigmoid_i)
            .arg(&fuse_decay_softplus_i);
        // SAFETY: argument types/order match `linear_attention_*`; every pointer
        // is a live contiguous device allocation validated above (nulls for
        // absent optionals), and each thread owns one disjoint state column so
        // there are no data races.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch LinearAttention", error))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }
}

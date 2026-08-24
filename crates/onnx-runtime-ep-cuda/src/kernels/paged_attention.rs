//! Native `com.microsoft::PagedAttention` v1 — the **LATENT** (absorbed
//! multi-head latent attention) subset used by GLM-5.2 dense MLA.
//!
//! This kernel is written from the op's equations (mirroring the CPU oracle in
//! `onnx-genai-paged-attention::oracle`), NOT copied from ORT source. It claims
//! only the subset it can prove correct; every other mode
//! (`kv_cache_layout="SEPARATE"`, quantized cache, `head_sink`, q/k-norm, packed
//! QKV, non-f16/bf16) is rejected with a typed reason via
//! [`unsupported_reason`], so the session routes those elsewhere instead of the
//! kernel silently miscomputing.
//!
//! ## LATENT semantics (single cache, `kv_num_heads == 1`)
//!
//! * **Write phase** — each new token's latent key row (input `key`) has partial
//!   RoPE applied to its `[rotary_offset, rotary_offset+rotary_dim)` suffix at
//!   the token's absolute position, then is scattered into `key_cache` at
//!   `slot_mapping[token]` (a slot of `-1` skips the write). The cache is updated
//!   **in place**; `key_cache_out`, when wired, must alias `key_cache`.
//! * **Read phase** — for every `(token, head)` the query row gets partial RoPE,
//!   then dense causal attention runs over the cached prefix addressed by
//!   `block_table`. `V` is the **leading `v_head_size` channels** of the same
//!   latent row that supplies `K` (width `head_size`).
//!
//! ## Capture safety
//!
//! Two launches (write, then attention) on the op's stream, both with launch
//! geometry fixed by tensor shapes — no host readback, no host allocation, no
//! device→host sync. Per-token `(batch, position)` and per-block physical slots
//! are resolved on device from `cumulative_sequence_length` / `past_seqlens` /
//! `block_table`. A warmed exact-shape signature gates capture, matching the
//! other attention kernels in this crate.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

/// Fixed thread-block width (power of two for the score reduction). Head
/// channels are striped across the block, so `head_size`/`v_head_size` are
/// unconstrained by this value (model-agnostic: no baked attention dims).
const ATTN_BLOCK: u32 = 128;
const WRITE_BLOCK: u32 = 256;
const PAGED_ATTENTION_MODULE: &str = "paged_attention_latent_v1";

// Schema input indices (`bert_defs.cc:1622-1729`).
const IN_QUERY: usize = 0;
const IN_KEY: usize = 1;
const IN_VALUE: usize = 2;
const IN_KEY_CACHE: usize = 3;
const IN_VALUE_CACHE: usize = 4;
const IN_CUMULATIVE_SEQLEN: usize = 5;
const IN_PAST_SEQLENS: usize = 6;
const IN_BLOCK_TABLE: usize = 7;
const IN_COS_CACHE: usize = 8;
const IN_SIN_CACHE: usize = 9;
const IN_SLOT_MAPPING: usize = 10;
const IN_HEAD_SINK: usize = 11;
const IN_Q_NORM: usize = 12;
const IN_K_NORM: usize = 13;
const IN_K_SCALE: usize = 14;
const IN_V_SCALE: usize = 15;

const PAGED_ATTENTION_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define POS_INF __int_as_float(0x7f800000)

__device__ __forceinline__ float cvt_f32(__half x) { return __half2float(x); }
__device__ __forceinline__ float cvt_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void store_f32(__half* p, float v) { *p = __float2half_rn(v); }
__device__ __forceinline__ void store_f32(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

// Locate the (batch, absolute position) of global token `t`.
__device__ __forceinline__ void locate_token(
    long long t, const int* cumseq, const int* past, int batch,
    int* b_out, long long* pos_out) {
  int b = 0;
  for (int i = 0; i < batch; ++i) {
    if (t < (long long)cumseq[i + 1]) { b = i; break; }
  }
  *b_out = b;
  *pos_out = (long long)past[b] + (t - (long long)cumseq[b]);
}

// Rotated value for channel `d` of a head row starting at `base`, absolute
// position `pos`. Mirrors oracle::apply_rope: for non-interleaved the first
// half uses -sin, the second half +sin; interleaved pairs adjacent channels.
template <typename T>
__device__ __forceinline__ float rope_channel(
    const T* base, long long d, long long rotary_offset, long long rotary_dim,
    long long pos, const T* cos_cache, const T* sin_cache, int interleaved) {
  float x = cvt_f32(base[d]);
  if (d < rotary_offset || d >= rotary_offset + rotary_dim) {
    return x;
  }
  const long long half = rotary_dim / 2;
  const long long rd = d - rotary_offset;
  long long i_idx;
  long long partner;
  float sign;
  if (interleaved) {
    i_idx = rd / 2;
    if ((rd & 1LL) == 0) { partner = d + 1; sign = -1.0f; }
    else { partner = d - 1; sign = 1.0f; }
  } else if (rd < half) {
    i_idx = rd; partner = d + half; sign = -1.0f;
  } else {
    i_idx = rd - half; partner = d - half; sign = 1.0f;
  }
  const float c = cvt_f32(cos_cache[pos * half + i_idx]);
  const float s = cvt_f32(sin_cache[pos * half + i_idx]);
  const float xp = cvt_f32(base[partner]);
  return x * c + sign * xp * s;
}

// ---- Write phase: post-RoPE latent K scattered to key_cache[slot*hs ..].
template <typename T>
__device__ void write_impl(
    const T* key, const T* cos_cache, const T* sin_cache,
    const int* slot_mapping, const int* cumseq, const int* past,
    T* key_cache,
    long long token_count, long long hs, long long rotary_offset,
    long long rotary_dim, int do_rotary, int interleaved, int batch,
    unsigned long long elements) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < elements; i += (unsigned long long)gridDim.x * blockDim.x) {
    const long long t = (long long)(i / (unsigned long long)hs);
    const long long d = (long long)(i % (unsigned long long)hs);
    const int slot = slot_mapping[t];
    if (slot < 0) continue;
    int b;
    long long pos;
    locate_token(t, cumseq, past, batch, &b, &pos);
    const T* row = key + t * hs;
    float out;
    if (do_rotary) {
      out = rope_channel(row, d, rotary_offset, rotary_dim, pos, cos_cache,
                         sin_cache, interleaved);
    } else {
      out = cvt_f32(row[d]);
    }
    store_f32(&key_cache[(unsigned long long)slot * hs + d], out);
  }
}

// ---- Read phase: dense causal attention over the block cache. One block per
// (token, head); head channels striped across the block; online softmax.
template <typename T>
__device__ void attn_impl(
    const T* query, const T* key_cache, const T* cos_cache, const T* sin_cache,
    const int* cumseq, const int* past, const int* block_table,
    T* output,
    long long token_count, long long num_heads, long long hs, long long vhs,
    long long block_size, long long max_blocks, long long rotary_offset,
    long long rotary_dim, int do_rotary, int interleaved, int batch,
    float scale, float softcap, long long window) {
  extern __shared__ float smem[];
  float* q_sh = smem;           // hs
  float* acc = q_sh + hs;       // vhs
  float* red = acc + vhs;       // blockDim
  __shared__ float sc[5];       // 0:m 1:denom 2:score 3:corr 4:p

  const long long blk = blockIdx.x;
  const long long t = blk / num_heads;
  const long long h = blk % num_heads;
  const long long q_hidden = num_heads * hs;
  int b;
  long long pos;
  locate_token(t, cumseq, past, batch, &b, &pos);

  const T* q_row = query + t * q_hidden + h * hs;
  for (long long d = threadIdx.x; d < hs; d += blockDim.x) {
    q_sh[d] = do_rotary
        ? rope_channel(q_row, d, rotary_offset, rotary_dim, pos, cos_cache,
                       sin_cache, interleaved)
        : cvt_f32(q_row[d]);
  }
  for (long long d = threadIdx.x; d < vhs; d += blockDim.x) acc[d] = 0.0f;
  if (threadIdx.x == 0) { sc[0] = -POS_INF; sc[1] = 0.0f; }
  __syncthreads();

  for (long long j = 0; j <= pos; ++j) {
    if (window >= 0 && (pos - j) > window) continue;
    const long long block_in_seq = j / block_size;
    const long long phys = (long long)block_table[b * max_blocks + block_in_seq];
    const unsigned long long slot =
        (unsigned long long)phys * block_size + (unsigned long long)(j % block_size);
    const T* k_row = key_cache + slot * hs;

    float partial = 0.0f;
    for (long long d = threadIdx.x; d < hs; d += blockDim.x) {
      partial += q_sh[d] * cvt_f32(k_row[d]);
    }
    red[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      float s = red[0] * scale;
      if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
      const float m_old = sc[0];
      const float m_new = fmaxf(m_old, s);
      const float corr = (m_old == -POS_INF) ? 0.0f : __expf(m_old - m_new);
      const float p = __expf(s - m_new);
      sc[0] = m_new;
      sc[3] = corr;
      sc[4] = p;
      sc[1] = sc[1] * corr + p;
    }
    __syncthreads();
    const float corr = sc[3];
    const float p = sc[4];
    for (long long d = threadIdx.x; d < vhs; d += blockDim.x) {
      acc[d] = acc[d] * corr + p * cvt_f32(k_row[d]);
    }
    __syncthreads();
  }

  const float denom = sc[1];
  T* out_row = output + t * num_heads * vhs + h * vhs;
  for (long long d = threadIdx.x; d < vhs; d += blockDim.x) {
    store_f32(&out_row[d], denom > 0.0f ? acc[d] / denom : 0.0f);
  }
}

extern "C" __global__ void paged_latent_write_f16(
    const __half* key, const __half* cos_cache, const __half* sin_cache,
    const int* slot_mapping, const int* cumseq, const int* past,
    __half* key_cache, long long token_count, long long hs,
    long long rotary_offset, long long rotary_dim, int do_rotary,
    int interleaved, int batch, unsigned long long elements) {
  write_impl(key, cos_cache, sin_cache, slot_mapping, cumseq, past, key_cache,
             token_count, hs, rotary_offset, rotary_dim, do_rotary, interleaved,
             batch, elements);
}

extern "C" __global__ void paged_latent_write_bf16(
    const __nv_bfloat16* key, const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache, const int* slot_mapping, const int* cumseq,
    const int* past, __nv_bfloat16* key_cache, long long token_count,
    long long hs, long long rotary_offset, long long rotary_dim, int do_rotary,
    int interleaved, int batch, unsigned long long elements) {
  write_impl(key, cos_cache, sin_cache, slot_mapping, cumseq, past, key_cache,
             token_count, hs, rotary_offset, rotary_dim, do_rotary, interleaved,
             batch, elements);
}

extern "C" __global__ void paged_latent_attn_f16(
    const __half* query, const __half* key_cache, const __half* cos_cache,
    const __half* sin_cache, const int* cumseq, const int* past,
    const int* block_table, __half* output, long long token_count,
    long long num_heads, long long hs, long long vhs, long long block_size,
    long long max_blocks, long long rotary_offset, long long rotary_dim,
    int do_rotary, int interleaved, int batch, float scale, float softcap,
    long long window) {
  attn_impl(query, key_cache, cos_cache, sin_cache, cumseq, past, block_table,
            output, token_count, num_heads, hs, vhs, block_size, max_blocks,
            rotary_offset, rotary_dim, do_rotary, interleaved, batch, scale,
            softcap, window);
}

extern "C" __global__ void paged_latent_attn_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* cos_cache, const __nv_bfloat16* sin_cache,
    const int* cumseq, const int* past, const int* block_table,
    __nv_bfloat16* output, long long token_count, long long num_heads,
    long long hs, long long vhs, long long block_size, long long max_blocks,
    long long rotary_offset, long long rotary_dim, int do_rotary,
    int interleaved, int batch, float scale, float softcap, long long window) {
  attn_impl(query, key_cache, cos_cache, sin_cache, cumseq, past, block_table,
            output, token_count, num_heads, hs, vhs, block_size, max_blocks,
            rotary_offset, rotary_dim, do_rotary, interleaved, batch, scale,
            softcap, window);
}
"#;

fn attr_int(node: &Node, name: &str, default: i64) -> i64 {
    node.attr(name)
        .and_then(onnx_runtime_ir::Attribute::as_int)
        .unwrap_or(default)
}

fn attr_float(node: &Node, name: &str) -> Option<f32> {
    node.attr(name)
        .and_then(onnx_runtime_ir::Attribute::as_float)
}

fn attr_str<'a>(node: &'a Node, name: &str) -> Option<&'a str> {
    node.attr(name).and_then(onnx_runtime_ir::Attribute::as_str)
}

fn layout_is_latent(node: &Node) -> bool {
    attr_str(node, "kv_cache_layout") == Some("LATENT")
}

fn quant_attr_present(node: &Node, name: &str) -> bool {
    match attr_str(node, name) {
        None => false,
        Some(s) => !s.is_empty() && s != "0" && !s.eq_ignore_ascii_case("none"),
    }
}

/// Returns a typed reason when the node is a `PagedAttention` the LATENT kernel
/// does not implement, so `supports_op` declines rather than miscomputing.
pub fn unsupported_reason(
    op: &Node,
    _shapes: &[onnx_runtime_ir::Shape],
    input_dtypes: &[DataType],
) -> Option<String> {
    if !layout_is_latent(op) {
        return Some(
            "cuda_ep PagedAttention: only kv_cache_layout=\"LATENT\" is implemented; \
             SEPARATE is routed elsewhere"
                .to_string(),
        );
    }
    // Quantized cache is explicitly out of scope for this slice.
    for name in [
        "k_quant_type",
        "v_quant_type",
        "k_cache_dtype",
        "v_cache_dtype",
    ] {
        if quant_attr_present(op, name) {
            return Some(format!(
                "cuda_ep PagedAttention: quantized KV cache ({name}) is not implemented in the \
                 LATENT slice"
            ));
        }
    }
    // Optional inputs the LATENT slice does not implement yet.
    let present = |i: usize| op.inputs.get(i).is_some_and(Option::is_some);
    if present(IN_HEAD_SINK) {
        return Some(
            "cuda_ep PagedAttention: head_sink is not implemented in the LATENT slice".into(),
        );
    }
    if present(IN_Q_NORM) || present(IN_K_NORM) {
        return Some(
            "cuda_ep PagedAttention: q/k-norm is not implemented in the LATENT slice".into(),
        );
    }
    if present(IN_K_SCALE) || present(IN_V_SCALE) {
        return Some(
            "cuda_ep PagedAttention: k/v_scale (quantized cache) is not implemented in the LATENT slice"
                .into(),
        );
    }
    if present(IN_VALUE) || present(IN_VALUE_CACHE) {
        return Some(
            "cuda_ep PagedAttention: LATENT has a single cache; 'value'/'value_cache' must be absent"
                .into(),
        );
    }
    // Dtype: query drives the compute type.
    match input_dtypes.first() {
        Some(DataType::Float16 | DataType::BFloat16) => {}
        Some(other) => {
            return Some(format!(
                "cuda_ep PagedAttention: LATENT kernel supports float16/bfloat16, got {other:?}"
            ));
        }
        None => return Some("cuda_ep PagedAttention: missing query input".into()),
    }
    None
}

pub struct PagedAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for PagedAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        if !layout_is_latent(node) {
            return Err(EpError::KernelFailed(
                "cuda_ep PagedAttention: only kv_cache_layout=\"LATENT\" is implemented".into(),
            ));
        }
        let num_heads = attr_int(node, "num_heads", 0);
        let kv_num_heads = attr_int(node, "kv_num_heads", 0);
        if kv_num_heads != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep PagedAttention (LATENT): kv_num_heads must be 1, got {kv_num_heads}"
            )));
        }
        if num_heads < 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep PagedAttention: num_heads must be >= 1".into(),
            ));
        }
        let v_head_size = attr_int(node, "v_head_size", 0);
        let rotary_offset = attr_int(node, "rotary_offset", 0);
        let do_rotary = attr_int(node, "do_rotary", 0) != 0;
        let interleaved = attr_int(node, "rotary_interleaved", 0) != 0;
        let softcap = attr_float(node, "softcap").unwrap_or(0.0);
        let local_window_size = attr_int(node, "local_window_size", -1);
        let scale = attr_float(node, "scale");
        Ok(Box::new(PagedAttentionLatentKernel {
            runtime: self.runtime.clone(),
            num_heads,
            v_head_size,
            rotary_offset,
            do_rotary,
            interleaved,
            softcap,
            local_window_size,
            scale,
            warmed_signature: Mutex::new(None),
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

pub struct PagedAttentionLatentKernel {
    runtime: Arc<CudaRuntime>,
    num_heads: i64,
    v_head_size: i64,
    rotary_offset: i64,
    do_rotary: bool,
    interleaved: bool,
    softcap: f32,
    local_window_size: i64,
    scale: Option<f32>,
    warmed_signature: Mutex<Option<PagedCaptureSignature>>,
    last_call_capture_safe: AtomicBool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PagedCaptureSignature {
    dtype: DataType,
    query_shape: Vec<usize>,
    key_cache_shape: Vec<usize>,
    block_table_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

fn require_present<'a>(
    inputs: &'a [TensorView<'a>],
    idx: usize,
    name: &str,
) -> Result<&'a TensorView<'a>> {
    match inputs.get(idx) {
        Some(v) if !v.is_absent() => Ok(v),
        _ => Err(EpError::KernelFailed(format!(
            "cuda_ep PagedAttention: required input '{name}' (index {idx}) is absent"
        ))),
    }
}

fn expect_dtype(view: &TensorView, want: DataType, name: &str) -> Result<()> {
    if view.dtype != want {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep PagedAttention: input '{name}' dtype {:?} != expected {want:?}",
            view.dtype
        )));
    }
    Ok(())
}

fn expect_i32(view: &TensorView, name: &str) -> Result<()> {
    if view.dtype != DataType::Int32 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep PagedAttention: index input '{name}' must be int32, got {:?}",
            view.dtype
        )));
    }
    Ok(())
}

impl Kernel for PagedAttentionLatentKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.last_call_capture_safe.store(false, Ordering::Relaxed);

        let query = require_present(inputs, IN_QUERY, "query")?;
        let key = require_present(inputs, IN_KEY, "key")?;
        let key_cache = require_present(inputs, IN_KEY_CACHE, "key_cache")?;
        let cumseq = require_present(inputs, IN_CUMULATIVE_SEQLEN, "cumulative_sequence_length")?;
        let past = require_present(inputs, IN_PAST_SEQLENS, "past_seqlens")?;
        let block_table = require_present(inputs, IN_BLOCK_TABLE, "block_table")?;
        let slot_mapping = require_present(inputs, IN_SLOT_MAPPING, "slot_mapping")?;

        let dtype = query.dtype;
        if !matches!(dtype, DataType::Float16 | DataType::BFloat16) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep PagedAttention (LATENT): query must be float16/bfloat16, got {dtype:?}"
            )));
        }
        expect_dtype(key, dtype, "key")?;
        expect_dtype(key_cache, dtype, "key_cache")?;
        for (idx, name) in [
            (IN_VALUE, "value"),
            (IN_VALUE_CACHE, "value_cache"),
            (IN_HEAD_SINK, "head_sink"),
            (IN_Q_NORM, "q_norm_weight"),
            (IN_K_NORM, "k_norm_weight"),
            (IN_K_SCALE, "k_scale"),
            (IN_V_SCALE, "v_scale"),
        ] {
            if inputs.get(idx).is_some_and(|v| !v.is_absent()) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep PagedAttention (LATENT): optional input '{name}' is not implemented in this slice"
                )));
            }
        }
        expect_i32(cumseq, "cumulative_sequence_length")?;
        expect_i32(past, "past_seqlens")?;
        expect_i32(block_table, "block_table")?;
        expect_i32(slot_mapping, "slot_mapping")?;

        // Geometry.
        let hs = *key_cache.shape.get(3).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep PagedAttention: key_cache must be 4D".into())
        })? as i64;
        let block_size = *key_cache.shape.get(1).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep PagedAttention: key_cache must be 4D".into())
        })? as i64;
        if key_cache.shape.get(2).copied() != Some(1) {
            return Err(EpError::KernelFailed(
                "cuda_ep PagedAttention (LATENT): key_cache kv_num_heads dim must be 1".into(),
            ));
        }
        if block_size < 16 || (block_size & (block_size - 1)) != 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep PagedAttention: block_size must be a power of two >= 16, got {block_size}"
            )));
        }
        let token_count = *query.shape.first().ok_or_else(|| {
            EpError::KernelFailed("cuda_ep PagedAttention: query must be rank>=1".into())
        })? as i64;
        let num_heads = self.num_heads;
        let v_head_size = if self.v_head_size == 0 {
            hs
        } else {
            self.v_head_size
        };
        if v_head_size < 1 || v_head_size > hs {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep PagedAttention (LATENT): v_head_size {v_head_size} must be in 1..=head_size {hs}"
            )));
        }
        let rotary_dim = if self.do_rotary {
            let cos = require_present(inputs, IN_COS_CACHE, "cos_cache")?;
            let _sin = require_present(inputs, IN_SIN_CACHE, "sin_cache")?;
            expect_dtype(cos, dtype, "cos_cache")?;
            (*cos.shape.get(1).ok_or_else(|| {
                EpError::KernelFailed("cuda_ep PagedAttention: cos_cache must be 2D".into())
            })? as i64)
                * 2
        } else {
            0
        };
        if self.do_rotary && self.rotary_offset + rotary_dim > hs {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep PagedAttention: rotary_offset {} + rotary_dim {rotary_dim} exceeds head_size {hs}",
                self.rotary_offset
            )));
        }
        // The op requires an explicit scale when v_head_size != head_size (the
        // LATENT case); default 1/sqrt(head_size) otherwise.
        let scale = match self.scale {
            Some(s) => s,
            None if v_head_size == hs => 1.0f32 / (hs as f32).sqrt(),
            None => {
                return Err(EpError::KernelFailed(
                    "cuda_ep PagedAttention (LATENT): explicit 'scale' attribute is required when v_head_size != head_size"
                        .into(),
                ));
            }
        };
        let max_blocks = *block_table.shape.get(1).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep PagedAttention: block_table must be 2D".into())
        })? as i64;
        let batch = *past.shape.first().ok_or_else(|| {
            EpError::KernelFailed("cuda_ep PagedAttention: past_seqlens must be rank>=1".into())
        })? as i64;

        // Output 0 shape/dtype.
        let out = outputs.first_mut().ok_or_else(|| {
            EpError::KernelFailed("cuda_ep PagedAttention: missing output".into())
        })?;
        if out.dtype != dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep PagedAttention: output dtype must match query".into(),
            ));
        }

        // In-place cache alias: when key_cache_out (output 1) is wired it must
        // alias key_cache (input 3). We update the cache in place through the
        // key_cache device buffer, so the aliased output is updated too.
        let key_cache_ptr = key_cache.data_ptr::<u8>();
        let cache_out_present = outputs.get(1).is_some_and(|c| !c.absent);
        if cache_out_present {
            let out_ptr = outputs[1].data_ptr_mut::<u8>();
            if !std::ptr::eq(out_ptr as *const u8, key_cache_ptr) {
                return Err(EpError::KernelFailed(
                    "cuda_ep PagedAttention: key_cache_out must alias key_cache (in-place update)"
                        .into(),
                ));
            }
        }

        let capturing = self.runtime.is_capturing()?;
        let signature = PagedCaptureSignature {
            dtype,
            query_shape: query.shape.to_vec(),
            key_cache_shape: key_cache.shape.to_vec(),
            block_table_shape: block_table.shape.to_vec(),
            output_shape: outputs[0].shape.to_vec(),
        };
        {
            let warmed = self
                .warmed_signature
                .lock()
                .expect("cuda_ep PagedAttention capture signature poisoned");
            if capturing && warmed.as_ref() != Some(&signature) {
                return Err(EpError::KernelFailed(
                    "cuda_ep PagedAttention: shape/dtype changed during CUDA graph capture; warm the exact signature first".into(),
                ));
            }
        }

        // Device pointers.
        let query_ptr = cuptr(query.data_ptr::<u8>() as *const c_void);
        let key_ptr = cuptr(key.data_ptr::<u8>() as *const c_void);
        let key_cache_dev = cuptr(key_cache_ptr as *const c_void);
        let cumseq_ptr = cuptr(cumseq.data_ptr::<u8>() as *const c_void);
        let past_ptr = cuptr(past.data_ptr::<u8>() as *const c_void);
        let block_table_ptr = cuptr(block_table.data_ptr::<u8>() as *const c_void);
        let slot_ptr = cuptr(slot_mapping.data_ptr::<u8>() as *const c_void);
        let out_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let (cos_ptr, sin_ptr) = if self.do_rotary {
            (
                cuptr(inputs[IN_COS_CACHE].data_ptr::<u8>() as *const c_void),
                cuptr(inputs[IN_SIN_CACHE].data_ptr::<u8>() as *const c_void),
            )
        } else {
            (
                cuptr(query.data_ptr::<u8>() as *const c_void),
                cuptr(query.data_ptr::<u8>() as *const c_void),
            )
        };

        let suffix = match dtype {
            DataType::Float16 => "f16",
            DataType::BFloat16 => "bf16",
            _ => unreachable!("dtype checked above"),
        };
        let write_entry = format!("paged_latent_write_{suffix}");
        let attn_entry = format!("paged_latent_attn_{suffix}");
        let write_func = self.runtime.nvrtc_function(
            PAGED_ATTENTION_MODULE,
            PAGED_ATTENTION_SOURCE,
            &write_entry,
        )?;
        let attn_func = self.runtime.nvrtc_function(
            PAGED_ATTENTION_MODULE,
            PAGED_ATTENTION_SOURCE,
            &attn_entry,
        )?;

        // Scalars.
        let do_rotary_i = i32::from(self.do_rotary);
        let interleaved_i = i32::from(self.interleaved);
        let batch_i = batch as i32;
        let write_elements = (token_count * hs) as u64;

        // ---- Write launch.
        {
            let mut b = self.runtime.stream().launch_builder(&write_func);
            b.arg(&key_ptr)
                .arg(&cos_ptr)
                .arg(&sin_ptr)
                .arg(&slot_ptr)
                .arg(&cumseq_ptr)
                .arg(&past_ptr)
                .arg(&key_cache_dev)
                .arg(&token_count)
                .arg(&hs)
                .arg(&self.rotary_offset)
                .arg(&rotary_dim)
                .arg(&do_rotary_i)
                .arg(&interleaved_i)
                .arg(&batch_i)
                .arg(&write_elements);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (
                        write_elements.div_ceil(WRITE_BLOCK as u64).clamp(1, 65_535) as u32,
                        1,
                        1,
                    ),
                    block_dim: (WRITE_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| driver_err(&format!("launch {write_entry}"), e))?;
        }

        // ---- Attention launch (one block per (token, head)).
        let attn_blocks = (token_count * num_heads) as u64;
        if attn_blocks > u64::from(u32::MAX) {
            return Err(EpError::KernelFailed(
                "cuda_ep PagedAttention: token_count*num_heads exceeds grid limit".into(),
            ));
        }
        let shared_floats = (hs + v_head_size) as usize + ATTN_BLOCK as usize;
        let shared_mem_bytes = (shared_floats * std::mem::size_of::<f32>()) as u32;
        {
            let mut b = self.runtime.stream().launch_builder(&attn_func);
            b.arg(&query_ptr)
                .arg(&key_cache_dev)
                .arg(&cos_ptr)
                .arg(&sin_ptr)
                .arg(&cumseq_ptr)
                .arg(&past_ptr)
                .arg(&block_table_ptr)
                .arg(&out_ptr)
                .arg(&token_count)
                .arg(&num_heads)
                .arg(&hs)
                .arg(&v_head_size)
                .arg(&block_size)
                .arg(&max_blocks)
                .arg(&self.rotary_offset)
                .arg(&rotary_dim)
                .arg(&do_rotary_i)
                .arg(&interleaved_i)
                .arg(&batch_i)
                .arg(&scale)
                .arg(&self.softcap)
                .arg(&self.local_window_size);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (attn_blocks.max(1) as u32, 1, 1),
                    block_dim: (ATTN_BLOCK, 1, 1),
                    shared_mem_bytes,
                })
            }
            .map_err(|e| driver_err(&format!("launch {attn_entry}"), e))?;
        }

        if !capturing {
            *self
                .warmed_signature
                .lock()
                .expect("cuda_ep PagedAttention capture signature poisoned") = Some(signature);
        }
        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "cuda_ep PagedAttention requires a warmed exact-shape LATENT signature",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId, ValueId};

    fn latent_node() -> Node {
        let mut node = Node::new(NodeId(0), "PagedAttention", vec![None; 11], vec![]);
        node.domain = "com.microsoft".to_string();
        node.attributes.insert(
            "kv_cache_layout".into(),
            Attribute::String(b"LATENT".to_vec()),
        );
        node.attributes
            .insert("num_heads".into(), Attribute::Int(2));
        node.attributes
            .insert("kv_num_heads".into(), Attribute::Int(1));
        node.attributes
            .insert("v_head_size".into(), Attribute::Int(128));
        node
    }

    #[test]
    fn accepts_supported_latent_fp16() {
        let node = latent_node();
        assert_eq!(
            unsupported_reason(&node, &[], &[DataType::Float16]),
            None,
            "a plain fp16 LATENT node must be accepted"
        );
        assert_eq!(unsupported_reason(&node, &[], &[DataType::BFloat16]), None);
    }

    #[test]
    fn rejects_separate_layout() {
        let mut node = latent_node();
        node.attributes.insert(
            "kv_cache_layout".into(),
            Attribute::String(b"SEPARATE".to_vec()),
        );
        let reason =
            unsupported_reason(&node, &[], &[DataType::Float16]).expect("SEPARATE rejected");
        assert!(reason.contains("LATENT"), "{reason}");
    }

    #[test]
    fn rejects_quantized_cache() {
        for attr in [
            "k_quant_type",
            "v_quant_type",
            "k_cache_dtype",
            "v_cache_dtype",
        ] {
            let mut node = latent_node();
            node.attributes
                .insert(attr.into(), Attribute::String(b"PER_TENSOR".to_vec()));
            let reason =
                unsupported_reason(&node, &[], &[DataType::Float16]).expect("quant rejected");
            assert!(reason.contains("quantized"), "{attr}: {reason}");
        }
    }

    #[test]
    fn rejects_unsupported_optional_inputs() {
        for (idx, needle) in [
            (IN_HEAD_SINK, "head_sink"),
            (IN_Q_NORM, "q/k-norm"),
            (IN_K_NORM, "q/k-norm"),
            (IN_K_SCALE, "k/v_scale"),
            (IN_VALUE, "single cache"),
            (IN_VALUE_CACHE, "single cache"),
        ] {
            let mut node = latent_node();
            let mut inputs = vec![None; idx + 1];
            inputs[idx] = Some(ValueId(idx as u32));
            node.inputs = inputs;
            let reason =
                unsupported_reason(&node, &[], &[DataType::Float16]).expect("optional rejected");
            assert!(reason.contains(needle), "idx {idx}: {reason}");
        }
    }

    #[test]
    fn rejects_non_half_dtype() {
        let node = latent_node();
        let reason = unsupported_reason(&node, &[], &[DataType::Float32]).expect("fp32 rejected");
        assert!(reason.contains("float16/bfloat16"), "{reason}");
    }
}

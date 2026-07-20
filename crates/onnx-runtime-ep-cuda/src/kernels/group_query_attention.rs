//! CUDA implementation of `com.microsoft::GroupQueryAttention`.
//!
//! BSH query/key/value inputs are prepared into BNSH buffers with NVRTC kernels,
//! including cache append and optional RoPE. Multi-token prefill then uses the
//! shared tiled online-softmax flash kernel when its measured shape gate wins;
//! decode and unsupported/slower shapes retain the existing attention baseline.
//! Present key/value outputs remain BNSH and preserve a fixed cache capacity.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

use super::attention::{AttentionDtype, run_attention_phase2a};
use super::flash_attention;

const PREP_SRC: &str = r#"
extern "C" __global__ void gqa_transpose_bsh_to_bnsh(
    const float* src, float* dst, int batch, int seq, int heads, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * heads * seq * dim;
    if (idx >= count) return;
    int x = idx;
    const int d = x % dim; x /= dim;
    const int s = x % seq; x /= seq;
    const int h = x % heads; const int b = x / heads;
    dst[idx] = src[((b * seq + s) * heads + h) * dim + d];
}

extern "C" __global__ void gqa_split_packed_qkv(
    const float* packed, float* query, float* key, float* value,
    int batch, int seq, int q_heads, int kv_heads, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int q_hidden = q_heads * dim;
    const int kv_hidden = kv_heads * dim;
    const int packed_hidden = q_hidden + 2 * kv_hidden;
    const int count = batch * seq * packed_hidden;
    if (idx >= count) return;
    const int feature = idx % packed_hidden;
    const int token = idx / packed_hidden;
    if (feature < q_hidden) {
        query[token * q_hidden + feature] = packed[idx];
    } else if (feature < q_hidden + kv_hidden) {
        key[token * kv_hidden + feature - q_hidden] = packed[idx];
    } else {
        value[token * kv_hidden + feature - q_hidden - kv_hidden] = packed[idx];
    }
}

extern "C" __global__ void gqa_prepare_metadata(
    const int* seqlens_k, int* total_lengths, int* past_lengths,
    int* query_starts, int batch, int current_key_length, int query_length,
    int past_capacity, int present_capacity, const long long* position_ids,
    int validate_positions, int cache_rows, int* error_flag)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const long long total = (long long)seqlens_k[b] + 1;
    const long long past = total - current_key_length;
    const long long query_start = total - query_length;
    int error = 0;
    if (total > 2147483647LL) error |= 1;
    if (past < 0) error |= 2;
    if (query_start < 0) error |= 4;
    if (past > past_capacity) error |= 8;
    if (total > present_capacity) error |= 16;
    if (validate_positions) {
        for (int s = 0; s < query_length; ++s) {
            const long long position = position_ids
                ? position_ids[b * query_length + s]
                : past + s;
            if (position < 0 || position >= (long long)cache_rows) {
                error |= 32;
            }
        }
    }
    if (error) {
        if (error_flag) atomicOr(error_flag, error);
        total_lengths[b] = -1;
        past_lengths[b] = -1;
        query_starts[b] = -1;
        return;
    }
    total_lengths[b] = (int)total;
    past_lengths[b] = (int)past;
    query_starts[b] = (int)query_start;
}

extern "C" __global__ void gqa_build_cache(
    const float* current, const float* past, float* present,
    const int* past_lengths, int batch, int seq, int heads, int dim,
    int past_capacity, int present_capacity)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * heads * present_capacity * dim;
    if (idx >= count) return;
    int x = idx;
    const int d = x % dim; x /= dim;
    const int s = x % present_capacity; x /= present_capacity;
    const int h = x % heads; const int b = x / heads;
    const int past_len = past_lengths[b];
    if (past_len < 0) return;
    float value = 0.0f;
    if (s < past_len && past) {
        value = past[((b * heads + h) * past_capacity + s) * dim + d];
    } else if (s >= past_len && s < past_len + seq) {
        const int current_s = s - past_len;
        value = current[((b * seq + current_s) * heads + h) * dim + d];
    }
    present[idx] = value;
}

extern "C" __global__ void gqa_append_cache(
    const float* current, float* present, const int* past_lengths,
    int batch, int seq, int heads, int dim, int present_capacity)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * heads * seq * dim;
    if (idx >= count) return;
    int x = idx;
    const int d = x % dim; x /= dim;
    const int s = x % seq; x /= seq;
    const int h = x % heads; const int b = x / heads;
    const int target_s = past_lengths[b] + s;
    if (target_s < 0 || target_s >= present_capacity) return;
    present[((b * heads + h) * present_capacity + target_s) * dim + d] =
        current[((b * seq + s) * heads + h) * dim + d];
}

extern "C" __global__ void gqa_rope_bnsh(
    float* tensor, const float* cos_cache, const float* sin_cache,
    const long long* position_ids, const int* past_lengths,
    int batch, int seq, int heads, int dim, int tensor_capacity,
    int current_offset, int cache_rows, int interleaved)
{
    const int half = dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * heads * seq * half;
    if (idx >= count) return;
    int x = idx;
    const int k = x % half; x /= half;
    const int s = x % seq; x /= seq;
    const int h = x % heads; const int b = x / heads;
    const int past = past_lengths[b];
    if (past < 0) return;
    const long long position = position_ids
        ? position_ids[b * seq + s]
        : (long long)past + s;
    if (position < 0 || position >= (long long)cache_rows) return;
    const int pos = (int)position;
    const int d0 = interleaved ? 2 * k : k;
    const int d1 = interleaved ? 2 * k + 1 : k + half;
    const int tensor_s = current_offset ? past_lengths[b] + s : s;
    const size_t base = ((size_t)(b * heads + h) * tensor_capacity + tensor_s) * dim;
    const float x0 = tensor[base + d0];
    const float x1 = tensor[base + d1];
    const float c = cos_cache[pos * half + k];
    const float sn = sin_cache[pos * half + k];
    tensor[base + d0] =
        __fsub_rn(__fmul_rn(c, x0), __fmul_rn(sn, x1));
    tensor[base + d1] =
        __fadd_rn(__fmul_rn(sn, x0), __fmul_rn(c, x1));
}

extern "C" __global__ void gqa_transpose_bnsh_to_bsh(
    const float* src, float* dst, int batch, int seq, int heads, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * seq * heads * dim;
    if (idx >= count) return;
    int x = idx;
    const int d = x % dim; x /= dim;
    const int h = x % heads; x /= heads;
    const int s = x % seq; const int b = x / seq;
    dst[idx] = src[((b * heads + h) * seq + s) * dim + d];
}

extern "C" __global__ void gqa_attention_reference_f32(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* scores,
    const int* total_lengths,
    const int batch,
    const int query_heads,
    const int kv_heads,
    const int query_seq,
    const int head_size,
    const int cache_capacity,
    const int group_size,
    const float scale,
    const int local_window,
    const float softcap)
{
    const int row = blockIdx.x;
    const int rows = batch * query_heads * query_seq;
    if (row >= rows) return;
    const int query_pos = row % query_seq;
    const int query_head = (row / query_seq) % query_heads;
    const int batch_index = row / (query_heads * query_seq);
    const int kv_head = query_head / group_size;
    const int total = total_lengths[batch_index];
    const int causal_limit = total - query_seq + query_pos;
    const int local_start =
        local_window > 0 && causal_limit + 1 > local_window
            ? causal_limit + 1 - local_window
            : 0;
    float* row_scores = scores + (long)row * cache_capacity;

    if (threadIdx.x == 0) {
        const float negative_infinity = __int_as_float(0xff800000);
        float maximum = negative_infinity;
        for (int key_pos = 0; key_pos < total; ++key_pos) {
            float score = negative_infinity;
            if (key_pos >= local_start && key_pos <= causal_limit) {
                score = 0.0f;
                const long q_base =
                    ((long)(batch_index * query_heads + query_head) * query_seq + query_pos)
                    * head_size;
                const long k_base =
                    ((long)(batch_index * kv_heads + kv_head) * cache_capacity + key_pos)
                    * head_size;
                for (int d = 0; d < head_size; ++d) {
                    score = __fadd_rn(
                        score,
                        __fmul_rn(query[q_base + d], key[k_base + d]));
                }
                score = __fmul_rn(score, scale);
                if (softcap != 0.0f) {
                    score = __fmul_rn(softcap, tanhf(score / softcap));
                }
            }
            row_scores[key_pos] = score;
            maximum = fmaxf(maximum, score);
        }
        for (int key_pos = 0; key_pos < total; ++key_pos) {
            float probability = isfinite(row_scores[key_pos])
                ? (float)exp((double)(row_scores[key_pos] - maximum))
                : 0.0f;
            row_scores[key_pos] = probability;
        }
    }
    __syncthreads();

    float sum = 0.0f;
    for (int key_pos = 0; key_pos < total; ++key_pos) {
        sum = __fadd_rn(sum, row_scores[key_pos]);
    }
    for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
        float result = 0.0f;
        for (int key_pos = 0; key_pos < total; ++key_pos) {
            const long v_index =
                ((long)(batch_index * kv_heads + kv_head) * cache_capacity + key_pos)
                * head_size + d;
            const float weighted =
                __fmul_rn(row_scores[key_pos] / sum, value[v_index]);
            result = __fadd_rn(result, weighted);
        }
        output[
            ((long)(batch_index * query_heads + query_head) * query_seq + query_pos)
                * head_size + d] = result;
    }
}
"#;

const PREP_HALF_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T> __device__ __forceinline__ float gqa_load(T value);
template <> __device__ __forceinline__ float gqa_load<__half>(__half value) {
    return __half2float(value);
}
template <> __device__ __forceinline__ float gqa_load<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
template <typename T> __device__ __forceinline__ T gqa_store(float value);
template <> __device__ __forceinline__ __half gqa_store<__half>(float value) {
    return __float2half_rn(value);
}
template <> __device__ __forceinline__ __nv_bfloat16 gqa_store<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__device__ void gqa_transpose_bsh_to_bnsh_body(
    const T* src, T* dst, int batch, int seq, int heads, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * heads * seq * dim;
    if (idx >= count) return;
    int x = idx;
    const int d = x % dim; x /= dim;
    const int s = x % seq; x /= seq;
    const int h = x % heads; const int b = x / heads;
    dst[idx] = src[((b * seq + s) * heads + h) * dim + d];
}

template <typename T>
__device__ void gqa_split_packed_qkv_body(
    const T* packed, T* query, T* key, T* value,
    int batch, int seq, int q_heads, int kv_heads, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int q_hidden = q_heads * dim;
    const int kv_hidden = kv_heads * dim;
    const int packed_hidden = q_hidden + 2 * kv_hidden;
    const int count = batch * seq * packed_hidden;
    if (idx >= count) return;
    const int feature = idx % packed_hidden;
    const int token = idx / packed_hidden;
    if (feature < q_hidden) {
        query[token * q_hidden + feature] = packed[idx];
    } else if (feature < q_hidden + kv_hidden) {
        key[token * kv_hidden + feature - q_hidden] = packed[idx];
    } else {
        value[token * kv_hidden + feature - q_hidden - kv_hidden] = packed[idx];
    }
}

template <typename T>
__device__ void gqa_build_cache_body(
    const T* current, const T* past, T* present,
    const int* past_lengths, int batch, int seq, int heads, int dim,
    int past_capacity, int present_capacity)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * heads * present_capacity * dim;
    if (idx >= count) return;
    int x = idx;
    const int d = x % dim; x /= dim;
    const int s = x % present_capacity; x /= present_capacity;
    const int h = x % heads; const int b = x / heads;
    const int past_len = past_lengths[b];
    if (past_len < 0) return;
    T result = gqa_store<T>(0.0f);
    if (s < past_len && past) {
        result = past[((b * heads + h) * past_capacity + s) * dim + d];
    } else if (s >= past_len && s < past_len + seq) {
        const int current_s = s - past_len;
        result = current[((b * seq + current_s) * heads + h) * dim + d];
    }
    present[idx] = result;
}

template <typename T>
__device__ void gqa_append_cache_body(
    const T* current, T* present, const int* past_lengths,
    int batch, int seq, int heads, int dim, int present_capacity)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * heads * seq * dim;
    if (idx >= count) return;
    int x = idx;
    const int d = x % dim; x /= dim;
    const int s = x % seq; x /= seq;
    const int h = x % heads; const int b = x / heads;
    const int target_s = past_lengths[b] + s;
    if (target_s < 0 || target_s >= present_capacity) return;
    present[((b * heads + h) * present_capacity + target_s) * dim + d] =
        current[((b * seq + s) * heads + h) * dim + d];
}

template <typename T>
__device__ void gqa_rope_bnsh_body(
    T* tensor, const float* cos_cache, const float* sin_cache,
    const long long* position_ids, const int* past_lengths,
    int batch, int seq, int heads, int dim, int tensor_capacity,
    int current_offset, int cache_rows, int interleaved)
{
    const int half = dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * heads * seq * half;
    if (idx >= count) return;
    int x = idx;
    const int k = x % half; x /= half;
    const int s = x % seq; x /= seq;
    const int h = x % heads; const int b = x / heads;
    const int past = past_lengths[b];
    if (past < 0) return;
    const long long position = position_ids
        ? position_ids[b * seq + s]
        : (long long)past + s;
    if (position < 0 || position >= (long long)cache_rows) return;
    const int pos = (int)position;
    const int d0 = interleaved ? 2 * k : k;
    const int d1 = interleaved ? 2 * k + 1 : k + half;
    const int tensor_s = current_offset ? past_lengths[b] + s : s;
    const size_t base = ((size_t)(b * heads + h) * tensor_capacity + tensor_s) * dim;
    const float x0 = gqa_load<T>(tensor[base + d0]);
    const float x1 = gqa_load<T>(tensor[base + d1]);
    const float c = cos_cache[pos * half + k];
    const float sn = sin_cache[pos * half + k];
    tensor[base + d0] = gqa_store<T>(c * x0 - sn * x1);
    tensor[base + d1] = gqa_store<T>(sn * x0 + c * x1);
}

template <typename T>
__device__ void gqa_transpose_bnsh_to_bsh_body(
    const T* src, T* dst, int batch, int seq, int heads, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * seq * heads * dim;
    if (idx >= count) return;
    int x = idx;
    const int d = x % dim; x /= dim;
    const int h = x % heads; x /= heads;
    const int s = x % seq; const int b = x / seq;
    dst[idx] = src[((b * heads + h) * seq + s) * dim + d];
}

#define DEFINE_GQA_HALF_KERNELS(TYPE, SUFFIX) \
extern "C" __global__ void gqa_transpose_bsh_to_bnsh_##SUFFIX( \
    const TYPE* src, TYPE* dst, int batch, int seq, int heads, int dim) { \
    gqa_transpose_bsh_to_bnsh_body<TYPE>(src, dst, batch, seq, heads, dim); \
} \
extern "C" __global__ void gqa_split_packed_qkv_##SUFFIX( \
    const TYPE* packed, TYPE* query, TYPE* key, TYPE* value, \
    int batch, int seq, int q_heads, int kv_heads, int dim) { \
    gqa_split_packed_qkv_body<TYPE>( \
        packed, query, key, value, batch, seq, q_heads, kv_heads, dim); \
} \
extern "C" __global__ void gqa_build_cache_##SUFFIX( \
    const TYPE* current, const TYPE* past, TYPE* present, \
    const int* past_lengths, int batch, int seq, int heads, int dim, \
    int past_capacity, int present_capacity) { \
    gqa_build_cache_body<TYPE>(current, past, present, past_lengths, batch, seq, heads, \
                               dim, past_capacity, present_capacity); \
} \
extern "C" __global__ void gqa_append_cache_##SUFFIX( \
    const TYPE* current, TYPE* present, const int* past_lengths, \
    int batch, int seq, int heads, int dim, int present_capacity) { \
    gqa_append_cache_body<TYPE>( \
        current, present, past_lengths, batch, seq, heads, dim, present_capacity); \
} \
extern "C" __global__ void gqa_rope_bnsh_##SUFFIX( \
    TYPE* tensor, const float* cos_cache, const float* sin_cache, \
    const long long* position_ids, const int* past_lengths, \
    int batch, int seq, int heads, int dim, int tensor_capacity, \
    int current_offset, int cache_rows, int interleaved) { \
    gqa_rope_bnsh_body<TYPE>(tensor, cos_cache, sin_cache, position_ids, past_lengths, \
                             batch, seq, heads, dim, tensor_capacity, current_offset, \
                             cache_rows, interleaved); \
} \
extern "C" __global__ void gqa_transpose_bnsh_to_bsh_##SUFFIX( \
    const TYPE* src, TYPE* dst, int batch, int seq, int heads, int dim) { \
    gqa_transpose_bnsh_to_bsh_body<TYPE>(src, dst, batch, seq, heads, dim); \
}

DEFINE_GQA_HALF_KERNELS(__half, f16)
DEFINE_GQA_HALF_KERNELS(__nv_bfloat16, bf16)
"#;

const PREP_MODULE: &str = "group_query_attention_prep";
const PREP_HALF_MODULE: &str = "group_query_attention_prep_half_v1";
const BLOCK: u32 = 256;
const WS_TOTALS: usize = 0;
const WS_PAST_LENGTHS: usize = 1;
const WS_QUERY_STARTS: usize = 2;
const WS_PACKED_Q: usize = 3;
const WS_PACKED_K: usize = 4;
const WS_PACKED_V: usize = 5;
const WS_Q_BNSH: usize = 6;
const WS_OUT_BNSH: usize = 7;
const WS_PRESENT_K: usize = 8;
const WS_PRESENT_V: usize = 9;
const WS_SCORES: usize = 10;
const WS_METADATA_ERROR: usize = 11;
const WS_COUNT: usize = 12;

const METADATA_ERROR_TOTAL_OVERFLOW: u32 = 1;
const METADATA_ERROR_PAST_NEGATIVE: u32 = 2;
const METADATA_ERROR_QUERY_NEGATIVE: u32 = 4;
const METADATA_ERROR_PAST_CAPACITY: u32 = 8;
const METADATA_ERROR_PRESENT_CAPACITY: u32 = 16;
const METADATA_ERROR_POSITION: u32 = 32;

pub struct GroupQueryAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for GroupQueryAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let required_heads = |name: &str| -> Result<usize> {
            let value = node.attr(name).and_then(|a| a.as_int()).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: missing required `{name}` attribute"
                ))
            })?;
            usize::try_from(value)
                .ok()
                .filter(|&v| v > 0)
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "cuda_ep GroupQueryAttention: `{name}` must be > 0"
                    ))
                })
        };
        let num_heads = required_heads("num_heads")?;
        let kv_num_heads = required_heads("kv_num_heads")?;
        if !num_heads.is_multiple_of(kv_num_heads) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: num_heads {num_heads} must be a multiple of kv_num_heads {kv_num_heads}"
            )));
        }
        for name in ["k_quant_type", "v_quant_type"] {
            if let Some(value) = node.attr(name)
                && value.as_str() != Some("NONE")
            {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: `{name}` other than NONE is not supported"
                )));
            }
        }
        for (name, message) in [
            ("kv_cache_bit_width", "quantized KV cache"),
            ("qk_output", "qk_output"),
            ("smooth_softmax", "smooth_softmax"),
        ] {
            if node.attr(name).and_then(|a| a.as_int()).unwrap_or(0) != 0 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: {message} is not supported"
                )));
            }
        }
        let softcap = node
            .attr("softcap")
            .and_then(|a| a.as_float())
            .unwrap_or(0.0);
        if softcap < 0.0 {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: softcap must be non-negative".into(),
            ));
        }
        Ok(Box::new(GroupQueryAttentionKernel::new(
            self.runtime.clone(),
            num_heads,
            kv_num_heads,
            node.attr("scale").and_then(|a| a.as_float()),
            node.attr("do_rotary").and_then(|a| a.as_int()).unwrap_or(0) != 0,
            node.attr("rotary_interleaved")
                .and_then(|a| a.as_int())
                .unwrap_or(0)
                != 0,
            node.attr("local_window_size")
                .and_then(|a| a.as_int())
                .unwrap_or(-1),
            softcap,
        )?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupQueryAttentionBackend {
    Auto,
    Fused,
    Phase2a,
}

#[derive(Debug)]
pub struct GroupQueryAttentionKernel {
    runtime: Arc<CudaRuntime>,
    num_heads: usize,
    kv_num_heads: usize,
    scale: Option<f32>,
    do_rotary: bool,
    rotary_interleaved: bool,
    local_window_size: i64,
    softcap: f32,
    backend: GroupQueryAttentionBackend,
    workspace: Mutex<GqaWorkspace>,
    last_capture_safe_signature: Mutex<Option<GqaCaptureSignature>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GqaCaptureSignature {
    dtype: DataType,
    batch: usize,
    query_sequence_length: usize,
    key_sequence_length: usize,
    q_hidden: usize,
    k_hidden: usize,
    dim: usize,
    past_capacity: usize,
    present_capacity: usize,
    packed_qkv: bool,
    explicit_positions: bool,
    cache_rows: usize,
    input_shapes: Vec<Option<Vec<usize>>>,
    output_shapes: Vec<Vec<usize>>,
    backend: GroupQueryAttentionBackend,
}

#[derive(Clone, Copy, Debug, Default)]
struct WorkspaceSlot {
    ptr: CUdeviceptr,
    bytes: usize,
}

#[derive(Debug)]
struct GqaWorkspace {
    runtime: Arc<CudaRuntime>,
    slots: [WorkspaceSlot; WS_COUNT],
    metadata_error_initialized: bool,
}

impl GqaWorkspace {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            slots: [WorkspaceSlot::default(); WS_COUNT],
            metadata_error_initialized: false,
        }
    }

    fn reserve(&mut self, index: usize, bytes: usize) -> Result<CUdeviceptr> {
        let bytes = bytes.max(1);
        let slot = self.slots[index];
        if slot.bytes >= bytes {
            return Ok(slot.ptr);
        }
        if self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: workspace slot {index} requires {bytes} bytes during CUDA graph capture; warm the fixed decode shape before capture"
            )));
        }
        let ptr = self.runtime.alloc_raw(bytes)?;
        if slot.ptr != 0 {
            // Dynamic prefill/growing-cache shapes may outgrow a slot. Preserve
            // the fixed-capacity decode fast path (which never reaches here),
            // but wait before replacing storage that queued work may still use.
            if let Err(error) = self.runtime.synchronize() {
                // SAFETY: `ptr` was allocated above and has not escaped.
                let _ = unsafe { self.runtime.free_raw(ptr) };
                return Err(error);
            }
            // SAFETY: synchronization completed all prior users of `slot.ptr`,
            // which is still exclusively owned by this workspace.
            if let Err(error) = unsafe { self.runtime.free_raw(slot.ptr) } {
                // SAFETY: `ptr` was allocated above and has not escaped.
                let _ = unsafe { self.runtime.free_raw(ptr) };
                return Err(error);
            }
        }
        self.slots[index] = WorkspaceSlot { ptr, bytes };
        Ok(ptr)
    }

    fn metadata_error_flag(&mut self) -> Result<CUdeviceptr> {
        let ptr = self.reserve(WS_METADATA_ERROR, std::mem::size_of::<u32>())?;
        if !self.metadata_error_initialized {
            // SAFETY: the metadata error slot is a live four-byte allocation.
            unsafe { self.runtime.htod(&0_u32.to_ne_bytes(), ptr) }?;
            self.metadata_error_initialized = true;
        }
        Ok(ptr)
    }

    fn take_metadata_error(&mut self) -> Result<u32> {
        if !self.metadata_error_initialized {
            return Ok(0);
        }
        let ptr = self.slots[WS_METADATA_ERROR].ptr;
        let mut bytes = [0_u8; std::mem::size_of::<u32>()];
        // SAFETY: the initialized metadata error slot contains one u32.
        unsafe { self.runtime.dtoh(&mut bytes, ptr) }?;
        let error = u32::from_ne_bytes(bytes);
        if error != 0 {
            // SAFETY: the metadata error slot remains a live four-byte allocation.
            unsafe { self.runtime.htod(&0_u32.to_ne_bytes(), ptr) }?;
        }
        Ok(error)
    }
}

impl Drop for GqaWorkspace {
    fn drop(&mut self) {
        for slot in self.slots.iter_mut().rev() {
            if slot.ptr != 0 {
                // SAFETY: every live slot pointer was allocated by this runtime
                // and remains exclusively owned by this workspace.
                let _ = unsafe { self.runtime.free_raw(slot.ptr) };
                slot.ptr = 0;
            }
        }
    }
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep GroupQueryAttention: {name} {value} exceeds i32"
        ))
    })
}

fn require_matching_capture_signature(
    runtime: &CudaRuntime,
    warmed: Option<&GqaCaptureSignature>,
    current: Option<&GqaCaptureSignature>,
) -> Result<()> {
    if runtime.is_capturing()? && (current.is_none() || warmed != current) {
        return Err(EpError::KernelFailed(
            "cuda_ep GroupQueryAttention: dtype, decode mode, or shape changed during CUDA graph capture; warm the exact f32 one-token fixed device-KV signature before capture".into(),
        ));
    }
    Ok(())
}

fn metadata_error_to_ep_error(error: u32) -> EpError {
    let mut violations = Vec::new();
    if error & METADATA_ERROR_TOTAL_OVERFLOW != 0 {
        violations.push("seqlens_k + 1 overflows int32");
    }
    if error & METADATA_ERROR_PAST_NEGATIVE != 0 {
        violations.push("seqlens_k + 1 is shorter than current key sequence");
    }
    if error & METADATA_ERROR_QUERY_NEGATIVE != 0 {
        violations.push("seqlens_k + 1 is shorter than current query sequence");
    }
    if error & METADATA_ERROR_PAST_CAPACITY != 0 {
        violations.push("effective past length exceeds past cache extent");
    }
    if error & METADATA_ERROR_PRESENT_CAPACITY != 0 {
        violations.push("valid sequence length exceeds present cache capacity");
    }
    if error & METADATA_ERROR_POSITION != 0 {
        violations.push("position_ids or implicit rotary position exceeds cache rows");
    }
    EpError::KernelFailed(format!(
        "cuda_ep GroupQueryAttention: deferred device metadata validation failed: {}",
        violations.join("; ")
    ))
}

fn require_dense(view: &TensorView, name: &str, dtype: DataType) -> Result<()> {
    if view.dtype != dtype {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep GroupQueryAttention: {name} must have dtype {dtype:?}, got {:?}",
            view.dtype
        )));
    }
    if !view.is_contiguous() {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep GroupQueryAttention: non-contiguous {name} is not supported; materialise it first"
        )));
    }
    Ok(())
}

fn read_i32(runtime: &CudaRuntime, view: &TensorView, name: &str) -> Result<Vec<i32>> {
    require_dense(view, name, DataType::Int32)?;
    let mut bytes = vec![0u8; view.numel() * 4];
    // SAFETY: the source tensor has exactly `bytes.len()` bytes.
    unsafe {
        runtime.dtoh(&mut bytes, cuptr(view.data_ptr::<u8>() as *const c_void))?;
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|x| i32::from_ne_bytes([x[0], x[1], x[2], x[3]]))
        .collect())
}

fn read_i64(runtime: &CudaRuntime, view: &TensorView, name: &str) -> Result<Vec<i64>> {
    require_dense(view, name, DataType::Int64)?;
    let mut bytes = vec![0u8; view.numel() * 8];
    // SAFETY: the source tensor has exactly `bytes.len()` bytes.
    unsafe {
        runtime.dtoh(&mut bytes, cuptr(view.data_ptr::<u8>() as *const c_void))?;
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|x| i64::from_ne_bytes([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]]))
        .collect())
}

macro_rules! launch_1d {
    ($runtime:expr, $module:expr, $source:expr, $entry:expr, $count:expr, $builder:ident, $args:block) => {{
        let function = $runtime.nvrtc_function($module, $source, $entry)?;
        let grid = u32::try_from(($count).div_ceil(BLOCK as usize)).map_err(|_| {
            EpError::KernelFailed("cuda_ep GroupQueryAttention: launch grid exceeds u32".into())
        })?;
        let mut $builder = $runtime.stream().launch_builder(&function);
        $args
        // SAFETY: each invocation supplies the argument ABI for its entry point;
        // input/output buffers outlive execution, and workspace buffers remain
        // owned by the kernel while stream-ordered work is pending.
        unsafe {
            $builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err(&format!("launch {}", $entry), e))?;
    }};
}

impl GroupQueryAttentionKernel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: Arc<CudaRuntime>,
        num_heads: usize,
        kv_num_heads: usize,
        scale: Option<f32>,
        do_rotary: bool,
        rotary_interleaved: bool,
        local_window_size: i64,
        softcap: f32,
    ) -> Result<Self> {
        if num_heads == 0
            || kv_num_heads == 0
            || !num_heads.is_multiple_of(kv_num_heads)
            || local_window_size == 0
            || local_window_size < -1
            || softcap < 0.0
        {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: invalid heads, local window, or softcap".into(),
            ));
        }
        Ok(Self {
            workspace: Mutex::new(GqaWorkspace::new(runtime.clone())),
            runtime,
            num_heads,
            kv_num_heads,
            scale,
            do_rotary,
            rotary_interleaved,
            local_window_size,
            softcap,
            backend: GroupQueryAttentionBackend::Auto,
            last_capture_safe_signature: Mutex::new(None),
        })
    }

    pub fn with_backend(mut self, backend: GroupQueryAttentionBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Resolves the configured backend using the same shape gate as execution.
    pub fn selected_backend_for_shape(
        &self,
        dtype: DataType,
        query_sequence_length: usize,
        valid_sequence_length: usize,
        head_size: usize,
    ) -> GroupQueryAttentionBackend {
        let fused_supported = flash_attention::supported(query_sequence_length, head_size);
        let measured_fused_win = valid_sequence_length <= 128
            || (dtype == DataType::Float16
                && head_size.is_multiple_of(16)
                && valid_sequence_length <= 512
                && self.runtime.capabilities().compute_capability().0 >= 7);
        if fused_supported
            && (self.backend == GroupQueryAttentionBackend::Fused
                || (self.backend == GroupQueryAttentionBackend::Auto && measured_fused_win))
        {
            GroupQueryAttentionBackend::Fused
        } else {
            GroupQueryAttentionBackend::Phase2a
        }
    }

    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let mut last_signature = self.last_capture_safe_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: capture signature lock poisoned".into(),
            )
        })?;
        let warmed_signature = last_signature.take();
        let capturing = self.runtime.is_capturing()?;
        if !capturing {
            let metadata_error = self
                .workspace
                .lock()
                .map_err(|_| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: workspace lock poisoned".into(),
                    )
                })?
                .take_metadata_error()?;
            if metadata_error != 0 {
                return Err(metadata_error_to_ep_error(metadata_error));
            }
        }
        if !(7..=14).contains(&inputs.len()) || !(1..=3).contains(&outputs.len()) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: expected 7..14 inputs and 1..3 outputs, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let packed_qkv = inputs[1].is_absent() && inputs[2].is_absent();
        if inputs[1].is_absent() != inputs[2].is_absent() {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: key and value must both be present for unpacked Q/K/V or both absent for packed QKV".into(),
            ));
        }
        for (index, feature) in [
            (10, "attention_bias"),
            (11, "head_sink"),
            (12, "quantized-cache k_scale"),
            (13, "quantized-cache v_scale"),
        ] {
            if inputs.get(index).is_some_and(|v| !v.is_absent()) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: {feature} is not supported"
                )));
            }
        }
        if self.local_window_size == 0 || self.local_window_size < -1 {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: local_window_size must be -1 or a positive integer"
                    .into(),
            ));
        }

        let q = &inputs[0];
        let dtype = AttentionDtype::from_onnx(q.dtype).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: query dtype {:?} is not supported; expected Float32, Float16, or BFloat16",
                q.dtype
            ))
        })?;
        require_dense(q, "query", q.dtype)?;
        if q.dtype != DataType::Float32 {
            self.runtime
                .require_nvrtc_half_headers("GroupQueryAttention")?;
        }
        let element_size = dtype.element_size() as usize;
        let (
            prep_module,
            prep_src,
            split_entry,
            transpose_in_entry,
            build_entry,
            append_entry,
            rope_entry,
            transpose_out_entry,
        ) = match q.dtype {
            DataType::Float32 => (
                PREP_MODULE,
                PREP_SRC,
                "gqa_split_packed_qkv",
                "gqa_transpose_bsh_to_bnsh",
                "gqa_build_cache",
                "gqa_append_cache",
                "gqa_rope_bnsh",
                "gqa_transpose_bnsh_to_bsh",
            ),
            DataType::Float16 => (
                PREP_HALF_MODULE,
                PREP_HALF_SRC,
                "gqa_split_packed_qkv_f16",
                "gqa_transpose_bsh_to_bnsh_f16",
                "gqa_build_cache_f16",
                "gqa_append_cache_f16",
                "gqa_rope_bnsh_f16",
                "gqa_transpose_bnsh_to_bsh_f16",
            ),
            DataType::BFloat16 => (
                PREP_HALF_MODULE,
                PREP_HALF_SRC,
                "gqa_split_packed_qkv_bf16",
                "gqa_transpose_bsh_to_bnsh_bf16",
                "gqa_build_cache_bf16",
                "gqa_append_cache_bf16",
                "gqa_rope_bnsh_bf16",
                "gqa_transpose_bnsh_to_bsh_bf16",
            ),
            _ => unreachable!(),
        };
        if q.shape.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: query must be rank 3 [B,S,H], got {:?}",
                q.shape
            )));
        }
        let (batch, q_seq, input_hidden) = (q.shape[0], q.shape[1], q.shape[2]);
        let (q_hidden, k_seq, k_hidden, dim) = if packed_qkv {
            let packed_heads = self.num_heads + 2 * self.kv_num_heads;
            if batch == 0
                || q_seq == 0
                || input_hidden == 0
                || !input_hidden.is_multiple_of(packed_heads)
            {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: packed query must be [B,S,(num_heads + 2*kv_num_heads)*head_size], got {:?}",
                    q.shape
                )));
            }
            let dim = input_hidden / packed_heads;
            (self.num_heads * dim, q_seq, self.kv_num_heads * dim, dim)
        } else {
            let (k, v) = (&inputs[1], &inputs[2]);
            for (view, name) in [(k, "key"), (v, "value")] {
                require_dense(view, name, q.dtype)?;
                if view.shape.len() != 3 {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep GroupQueryAttention: unpacked {name} must be rank 3 [B,S,H*D], got {:?}",
                        view.shape
                    )));
                }
            }
            let (k_batch, k_seq, k_hidden) = (k.shape[0], k.shape[1], k.shape[2]);
            if batch == 0
                || q_seq == 0
                || k_seq == 0
                || input_hidden == 0
                || k_hidden == 0
                || !input_hidden.is_multiple_of(self.num_heads)
                || !k_hidden.is_multiple_of(self.kv_num_heads)
                || v.shape != [batch, k_seq, k_hidden]
                || k_batch != batch
            {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: incompatible query/key/value batch, sequence, or hidden dimensions".into(),
                ));
            }
            let dim = input_hidden / self.num_heads;
            if k_hidden / self.kv_num_heads != dim {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: query and key/value head sizes must match".into(),
                ));
            }
            (input_hidden, k_seq, k_hidden, dim)
        };
        if outputs[0].dtype != q.dtype
            || outputs[0].shape != [batch, q_seq, q_hidden]
            || !outputs[0].is_contiguous()
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: output must be contiguous {:?} [B,S,H*D] = [{batch},{q_seq},{q_hidden}], got {:?}",
                q.dtype, outputs[0].shape
            )));
        }

        let has_past_key = !inputs[3].is_absent();
        let has_past_value = !inputs[4].is_absent();
        if has_past_key != has_past_value {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: past_key and past_value must be provided together"
                    .into(),
            ));
        }
        let past_capacity = if has_past_key {
            for (view, name) in [(&inputs[3], "past_key"), (&inputs[4], "past_value")] {
                require_dense(view, name, q.dtype)?;
                if view.shape.len() != 4
                    || view.shape[0] != batch
                    || view.shape[1] != self.kv_num_heads
                    || view.shape[3] != dim
                {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep GroupQueryAttention: {name} must be BNSH [{batch},{},{},{}], got {:?}",
                        self.kv_num_heads, view.shape[2], dim, view.shape
                    )));
                }
            }
            if inputs[3].shape != inputs[4].shape {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: past_key and past_value shapes must match".into(),
                ));
            }
            inputs[3].shape[2]
        } else {
            0
        };

        require_dense(&inputs[5], "seqlens_k", DataType::Int32)?;
        if inputs[5].shape != [batch] {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: seqlens_k must be non-negative int32 [batch_size]"
                    .into(),
            ));
        }
        require_dense(&inputs[6], "total_sequence_length", DataType::Int32)?;
        if inputs[6].numel() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: total_sequence_length must be one non-negative int32 scalar".into(),
            ));
        }
        let current_key_length = checked_i32(k_seq, "key sequence length")?;
        let query_length = checked_i32(q_seq, "query sequence length")?;
        let requested_present_capacity = outputs
            .get(1)
            .map(|output| output.shape.get(2).copied().unwrap_or(past_capacity));

        let explicit_positions = inputs.get(9).filter(|view| !view.is_absent());
        let (cos_ptr, sin_ptr, positions_ptr, cache_rows, cache_rows_usize) = if self.do_rotary {
            if !dim.is_multiple_of(2) {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: do_rotary requires an even head_size".into(),
                ));
            }
            if q_seq != k_seq {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: do_rotary requires equal query/key sequence lengths".into(),
                ));
            }
            let cos = inputs
                .get(7)
                .filter(|view| !view.is_absent())
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: do_rotary=1 requires cos_cache".into(),
                    )
                })?;
            let sin = inputs
                .get(8)
                .filter(|view| !view.is_absent())
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: do_rotary=1 requires sin_cache".into(),
                    )
                })?;
            require_dense(cos, "cos_cache", DataType::Float32)?;
            require_dense(sin, "sin_cache", DataType::Float32)?;
            if cos.shape.len() != 2 || sin.shape != cos.shape || cos.shape[1] != dim / 2 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: cos_cache/sin_cache must have shape [max_sequence_length,{}]",
                    dim / 2
                )));
            }
            let position_ptr = if let Some(position_ids) = explicit_positions {
                require_dense(position_ids, "position_ids", DataType::Int64)?;
                if position_ids.shape != [batch, q_seq] {
                    return Err(EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: position_ids must be valid non-negative int64 [batch_size, sequence_length]".into(),
                    ));
                }
                cuptr(position_ids.data_ptr::<u8>() as *const c_void)
            } else {
                0
            };
            (
                cuptr(cos.data_ptr::<u8>() as *const c_void),
                cuptr(sin.data_ptr::<u8>() as *const c_void),
                position_ptr,
                checked_i32(cos.shape[0], "rotary cache rows")?,
                cos.shape[0],
            )
        } else {
            (0, 0, 0, 0, 0)
        };

        let structurally_valid_outputs = requested_present_capacity.is_some_and(|capacity| {
            let expected = [batch, self.kv_num_heads, capacity, dim];
            outputs.len() == 3
                && outputs[1].dtype == q.dtype
                && outputs[1].shape == expected
                && outputs[1].is_contiguous()
                && outputs[2].dtype == q.dtype
                && outputs[2].shape == expected
                && outputs[2].is_contiguous()
        });
        let aliased_device_kv = structurally_valid_outputs
            && (outputs[1].data.0 as *mut u8)
                .wrapping_add(outputs[1].byte_offset)
                .cast_const()
                == inputs[3].data_ptr::<u8>()
            && (outputs[2].data.0 as *mut u8)
                .wrapping_add(outputs[2].byte_offset)
                .cast_const()
                == inputs[4].data_ptr::<u8>();
        let capture_candidate = requested_present_capacity
            .filter(|&present_capacity| {
                q.dtype == DataType::Float32
                    && q_seq == 1
                    && k_seq == 1
                    && has_past_key
                    && present_capacity >= 1
                    && past_capacity == present_capacity
                    && aliased_device_kv
                    && self.selected_backend_for_shape(q.dtype, q_seq, present_capacity, dim)
                        == GroupQueryAttentionBackend::Phase2a
            })
            .map(|present_capacity| GqaCaptureSignature {
                dtype: q.dtype,
                batch,
                query_sequence_length: q_seq,
                key_sequence_length: k_seq,
                q_hidden,
                k_hidden,
                dim,
                past_capacity,
                present_capacity,
                packed_qkv,
                explicit_positions: explicit_positions.is_some(),
                cache_rows: cache_rows_usize,
                input_shapes: inputs
                    .iter()
                    .map(|input| (!input.is_absent()).then(|| input.shape.to_vec()))
                    .collect(),
                output_shapes: outputs.iter().map(|output| output.shape.to_vec()).collect(),
                backend: GroupQueryAttentionBackend::Phase2a,
            });
        require_matching_capture_signature(
            &self.runtime,
            warmed_signature.as_ref(),
            capture_candidate.as_ref(),
        )?;
        let capture_safe_decode = capture_candidate
            .as_ref()
            .is_some_and(|candidate| warmed_signature.as_ref() == Some(candidate));

        let mut valid_sequence_length = None;
        let mut validated_past_lengths = None;
        let total_sequence_length = if capture_safe_decode {
            None
        } else {
            let seqlens = read_i32(&self.runtime, &inputs[5], "seqlens_k")?;
            if seqlens.iter().any(|&length| length < 0) {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: seqlens_k must be non-negative int32 [batch_size]"
                        .into(),
                ));
            }
            let total_scalar = read_i32(&self.runtime, &inputs[6], "total_sequence_length")?;
            if total_scalar[0] < 0 {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: total_sequence_length must be one non-negative int32 scalar".into(),
                ));
            }
            let total_sequence_length = total_scalar[0] as usize;
            let totals: Vec<i32> = seqlens
                .iter()
                .map(|&length| length.checked_add(1))
                .collect::<Option<_>>()
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: seqlens_k + 1 overflows int32".into(),
                    )
                })?;
            let maximum = totals.iter().copied().max().unwrap_or(0) as usize;
            if maximum > total_sequence_length {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: valid sequence length {maximum} exceeds physical total_sequence_length capacity {total_sequence_length}"
                )));
            }
            let mut past_lengths = Vec::with_capacity(batch);
            for &total in &totals {
                let past = total.checked_sub(current_key_length).ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: seqlens_k + 1 is shorter than current key sequence"
                            .into(),
                    )
                })?;
                total.checked_sub(query_length).ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: seqlens_k + 1 is shorter than current query sequence"
                            .into(),
                    )
                })?;
                if past as usize > past_capacity {
                    return Err(EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: effective past length exceeds past cache extent"
                            .into(),
                    ));
                }
                past_lengths.push(past);
            }
            valid_sequence_length = Some(maximum);
            validated_past_lengths = Some(past_lengths);
            Some(total_sequence_length)
        };

        if !capture_safe_decode && self.do_rotary {
            if let Some(position_ids) = explicit_positions {
                let ids = read_i64(&self.runtime, position_ids, "position_ids")?;
                let cache_rows_i64 = i64::from(cache_rows);
                if ids
                    .iter()
                    .any(|&position| position < 0 || position >= cache_rows_i64)
                {
                    return Err(EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: position_ids must be valid non-negative int64 [batch_size, sequence_length]".into(),
                    ));
                }
            } else if validated_past_lengths.as_ref().is_some_and(|lengths| {
                lengths
                    .iter()
                    .any(|&past| past as usize + q_seq > cache_rows_usize)
            }) {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: rotary position exceeds cache rows".into(),
                ));
            }
        }

        let minimum_present_capacity =
            past_capacity.max(total_sequence_length.unwrap_or(past_capacity));
        let present_capacity = requested_present_capacity.unwrap_or(minimum_present_capacity);
        if present_capacity < minimum_present_capacity {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: present cache capacity {present_capacity} is smaller than required {minimum_present_capacity}"
            )));
        }
        let expected_cache_shape = [batch, self.kv_num_heads, present_capacity, dim];
        for (index, name) in [(1, "present_key"), (2, "present_value")] {
            if let Some(output) = outputs.get(index)
                && (output.dtype != q.dtype
                    || output.shape != expected_cache_shape
                    || !output.is_contiguous())
            {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: {name} must be contiguous {:?} BNSH {:?}, got {:?}",
                    q.dtype, expected_cache_shape, output.shape
                )));
            }
        }

        let mut workspace = self.workspace.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep GroupQueryAttention: workspace lock poisoned".into())
        })?;
        let metadata_bytes = batch * std::mem::size_of::<i32>();
        let totals_gpu = workspace.reserve(WS_TOTALS, metadata_bytes)?;
        let past_lengths_gpu = workspace.reserve(WS_PAST_LENGTHS, metadata_bytes)?;
        let query_starts_gpu = workspace.reserve(WS_QUERY_STARTS, metadata_bytes)?;
        let metadata_error_gpu = if capture_candidate.is_some() {
            workspace.metadata_error_flag()?
        } else {
            0
        };
        let packed_q = packed_qkv
            .then(|| workspace.reserve(WS_PACKED_Q, batch * q_seq * q_hidden * element_size))
            .transpose()?;
        let packed_k = packed_qkv
            .then(|| workspace.reserve(WS_PACKED_K, batch * k_seq * k_hidden * element_size))
            .transpose()?;
        let packed_v = packed_qkv
            .then(|| workspace.reserve(WS_PACKED_V, batch * k_seq * k_hidden * element_size))
            .transpose()?;
        let q_bnsh = workspace.reserve(WS_Q_BNSH, batch * q_seq * q_hidden * element_size)?;
        let out_bnsh = workspace.reserve(WS_OUT_BNSH, outputs[0].numel() * element_size)?;
        let owned_present_k = (outputs.len() < 2)
            .then(|| {
                workspace.reserve(
                    WS_PRESENT_K,
                    expected_cache_shape.iter().product::<usize>() * element_size,
                )
            })
            .transpose()?;
        let owned_present_v = (outputs.len() < 3)
            .then(|| {
                workspace.reserve(
                    WS_PRESENT_V,
                    expected_cache_shape.iter().product::<usize>() * element_size,
                )
            })
            .transpose()?;
        let present_k_ptr = if let Some(output) = outputs.get_mut(1) {
            cuptr(output.data_ptr_mut::<u8>() as *const c_void)
        } else {
            *owned_present_k.as_ref().ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: internal present-key allocation missing".into(),
                )
            })?
        };
        let present_v_ptr = if let Some(output) = outputs.get_mut(2) {
            cuptr(output.data_ptr_mut::<u8>() as *const c_void)
        } else {
            *owned_present_v.as_ref().ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: internal present-value allocation missing".into(),
                )
            })?
        };

        let batch_i = checked_i32(batch, "batch")?;
        let q_seq_i = checked_i32(q_seq, "query sequence length")?;
        let k_seq_i = checked_i32(k_seq, "key sequence length")?;
        let heads_i = checked_i32(self.num_heads, "num_heads")?;
        let kv_heads_i = checked_i32(self.kv_num_heads, "kv_num_heads")?;
        let dim_i = checked_i32(dim, "head_size")?;
        let past_capacity_i = checked_i32(past_capacity, "past capacity")?;
        let present_capacity_i = checked_i32(present_capacity, "present capacity")?;
        let local_window_i = i32::try_from(self.local_window_size.max(0)).map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: local_window_size exceeds i32".into(),
            )
        })?;
        let seqlens_ptr = cuptr(inputs[5].data_ptr::<u8>() as *const c_void);
        let validate_positions_i: i32 = self.do_rotary.into();
        launch_1d!(
            self.runtime,
            PREP_MODULE,
            PREP_SRC,
            "gqa_prepare_metadata",
            batch,
            builder,
            {
                builder
                    .arg(&seqlens_ptr)
                    .arg(&totals_gpu)
                    .arg(&past_lengths_gpu)
                    .arg(&query_starts_gpu)
                    .arg(&batch_i)
                    .arg(&current_key_length)
                    .arg(&query_length)
                    .arg(&past_capacity_i)
                    .arg(&present_capacity_i)
                    .arg(&positions_ptr)
                    .arg(&validate_positions_i)
                    .arg(&cache_rows)
                    .arg(&metadata_error_gpu);
            }
        );
        let input_q_ptr = cuptr(q.data_ptr::<u8>() as *const c_void);
        let (q_ptr, k_ptr, v_ptr) = if packed_qkv {
            let q_scratch = packed_q.as_ref().ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: internal packed-query allocation missing".into(),
                )
            })?;
            let k_scratch = packed_k.as_ref().ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: internal packed-key allocation missing".into(),
                )
            })?;
            let v_scratch = packed_v.as_ref().ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: internal packed-value allocation missing".into(),
                )
            })?;
            let packed_count = q.numel();
            launch_1d!(
                self.runtime,
                prep_module,
                prep_src,
                split_entry,
                packed_count,
                builder,
                {
                    builder
                        .arg(&input_q_ptr)
                        .arg(q_scratch)
                        .arg(k_scratch)
                        .arg(v_scratch)
                        .arg(&batch_i)
                        .arg(&q_seq_i)
                        .arg(&heads_i)
                        .arg(&kv_heads_i)
                        .arg(&dim_i);
                }
            );
            (*q_scratch, *k_scratch, *v_scratch)
        } else {
            (
                input_q_ptr,
                cuptr(inputs[1].data_ptr::<u8>() as *const c_void),
                cuptr(inputs[2].data_ptr::<u8>() as *const c_void),
            )
        };
        launch_1d!(
            self.runtime,
            prep_module,
            prep_src,
            transpose_in_entry,
            batch * q_seq * q_hidden,
            builder,
            {
                builder
                    .arg(&q_ptr)
                    .arg(&q_bnsh)
                    .arg(&batch_i)
                    .arg(&q_seq_i)
                    .arg(&heads_i)
                    .arg(&dim_i);
            }
        );

        let past_k_ptr = if has_past_key {
            cuptr(inputs[3].data_ptr::<u8>() as *const c_void)
        } else {
            0
        };
        let past_v_ptr = if has_past_value {
            cuptr(inputs[4].data_ptr::<u8>() as *const c_void)
        } else {
            0
        };
        for (current, past, present) in [
            (k_ptr, past_k_ptr, present_k_ptr),
            (v_ptr, past_v_ptr, present_v_ptr),
        ] {
            if past != 0 && past == present && past_capacity == present_capacity {
                launch_1d!(
                    self.runtime,
                    prep_module,
                    prep_src,
                    append_entry,
                    batch * self.kv_num_heads * k_seq * dim,
                    builder,
                    {
                        builder
                            .arg(&current)
                            .arg(&present)
                            .arg(&past_lengths_gpu)
                            .arg(&batch_i)
                            .arg(&k_seq_i)
                            .arg(&kv_heads_i)
                            .arg(&dim_i)
                            .arg(&present_capacity_i);
                    }
                );
            } else {
                launch_1d!(
                    self.runtime,
                    prep_module,
                    prep_src,
                    build_entry,
                    expected_cache_shape.iter().product::<usize>(),
                    builder,
                    {
                        builder
                            .arg(&current)
                            .arg(&past)
                            .arg(&present)
                            .arg(&past_lengths_gpu)
                            .arg(&batch_i)
                            .arg(&k_seq_i)
                            .arg(&kv_heads_i)
                            .arg(&dim_i)
                            .arg(&past_capacity_i)
                            .arg(&present_capacity_i);
                    }
                );
            }
        }

        if self.do_rotary {
            let interleaved_i: i32 = self.rotary_interleaved.into();
            for (tensor, seq_i, heads, capacity, current_offset) in [
                (q_bnsh, q_seq_i, heads_i, q_seq_i, 0i32),
                (present_k_ptr, k_seq_i, kv_heads_i, present_capacity_i, 1i32),
            ] {
                let count = batch * (heads as usize) * (seq_i as usize) * (dim / 2);
                launch_1d!(
                    self.runtime,
                    prep_module,
                    prep_src,
                    rope_entry,
                    count,
                    builder,
                    {
                        builder
                            .arg(&tensor)
                            .arg(&cos_ptr)
                            .arg(&sin_ptr)
                            .arg(&positions_ptr)
                            .arg(&past_lengths_gpu)
                            .arg(&batch_i)
                            .arg(&seq_i)
                            .arg(&heads)
                            .arg(&dim_i)
                            .arg(&capacity)
                            .arg(&current_offset)
                            .arg(&cache_rows)
                            .arg(&interleaved_i);
                    }
                );
            }
        }

        let scale = self
            .scale
            .filter(|&scale| scale != 0.0)
            .unwrap_or_else(|| 1.0 / (dim as f32).sqrt());
        let attention_sequence_length = valid_sequence_length.unwrap_or(present_capacity);
        let use_fused =
            self.selected_backend_for_shape(q.dtype, q_seq, attention_sequence_length, dim)
                == GroupQueryAttentionBackend::Fused;
        if use_fused {
            flash_attention::run(
                &self.runtime,
                q.dtype,
                self.num_heads,
                self.kv_num_heads,
                true,
                batch,
                q_seq,
                attention_sequence_length,
                present_capacity,
                dim,
                self.num_heads / self.kv_num_heads,
                scale,
                q_bnsh,
                present_k_ptr,
                present_v_ptr,
                out_bnsh,
                0,
                0,
                totals_gpu,
                query_starts_gpu,
                local_window_i,
                self.softcap,
            )?;
        } else if q.dtype == DataType::Float32 {
            let attention_rows = batch
                .checked_mul(self.num_heads)
                .and_then(|rows| rows.checked_mul(q_seq))
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: attention row count overflow".into(),
                    )
                })?;
            let score_count = attention_rows
                .checked_mul(present_capacity)
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: score scratch size overflow".into(),
                    )
                })?;
            let score_scratch = workspace.reserve(WS_SCORES, score_count.max(1) * 4)?;
            let attention_rows_u32 = u32::try_from(attention_rows).map_err(|_| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: attention row count exceeds u32".into(),
                )
            })?;
            let kv_heads_i = checked_i32(self.kv_num_heads, "KV head count")?;
            let group_i = checked_i32(
                self.num_heads / self.kv_num_heads,
                "query-to-KV head group size",
            )?;
            let func = self.runtime.nvrtc_function(
                PREP_MODULE,
                PREP_SRC,
                "gqa_attention_reference_f32",
            )?;
            let mut builder = self.runtime.stream().launch_builder(&func);
            builder
                .arg(&q_bnsh)
                .arg(&present_k_ptr)
                .arg(&present_v_ptr)
                .arg(&out_bnsh)
                .arg(&score_scratch)
                .arg(&totals_gpu)
                .arg(&batch_i)
                .arg(&heads_i)
                .arg(&kv_heads_i)
                .arg(&q_seq_i)
                .arg(&dim_i)
                .arg(&present_capacity_i)
                .arg(&group_i)
                .arg(&scale)
                .arg(&local_window_i)
                .arg(&self.softcap);
            // SAFETY: the scratch and BNSH buffers are sized above, and the scalar
            // ABI matches `gqa_attention_reference_f32`.
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (attention_rows_u32, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| driver_err("launch GQA reference attention", error))?;
        } else {
            run_attention_phase2a(
                &self.runtime,
                dtype,
                self.num_heads,
                self.kv_num_heads,
                true,
                batch,
                q_seq,
                attention_sequence_length,
                dim,
                present_capacity,
                self.num_heads / self.kv_num_heads,
                scale,
                q_bnsh,
                present_k_ptr,
                present_v_ptr,
                out_bnsh,
                0,
                0,
                totals_gpu,
                query_starts_gpu,
                local_window_i,
                self.softcap,
            )?;
        }

        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        launch_1d!(
            self.runtime,
            prep_module,
            prep_src,
            transpose_out_entry,
            outputs[0].numel(),
            builder,
            {
                builder
                    .arg(&out_bnsh)
                    .arg(&output_ptr)
                    .arg(&batch_i)
                    .arg(&q_seq_i)
                    .arg(&heads_i)
                    .arg(&dim_i);
            }
        );
        *last_signature = capture_candidate;
        Ok(())
    }
}

impl Kernel for GroupQueryAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn cuda_graph_compatible(&self) -> bool {
        // Eligibility is tied to the exact f32 one-token, fixed-capacity,
        // in-place device-KV reference-attention signature warmed by the most
        // recent successful call.
        self.last_capture_safe_signature
            .lock()
            .is_ok_and(|signature| signature.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> Option<Arc<CudaRuntime>> {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let runtime = std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten();
        std::panic::set_hook(previous_hook);
        runtime
    }

    #[test]
    fn persistent_workspace_reuses_fixed_shape_allocation() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping CUDA GQA workspace test: CUDA runtime unavailable");
            return;
        };
        let before = runtime.allocation_counts();
        let mut workspace = GqaWorkspace::new(runtime.clone());

        let first = workspace.reserve(WS_Q_BNSH, 4096).unwrap();
        let allocated = runtime.allocation_counts();
        assert_eq!(allocated.allocations, before.allocations + 1);
        assert_eq!(allocated.frees, before.frees);

        assert_eq!(workspace.reserve(WS_Q_BNSH, 4096).unwrap(), first);
        assert_eq!(workspace.reserve(WS_Q_BNSH, 2048).unwrap(), first);
        assert_eq!(runtime.allocation_counts(), allocated);

        let grown = workspace.reserve(WS_Q_BNSH, 8192).unwrap();
        assert_ne!(grown, first);
        let grown_counts = runtime.allocation_counts();
        assert_eq!(grown_counts.allocations, before.allocations + 2);
        assert_eq!(grown_counts.frees, before.frees + 1);

        drop(workspace);
        let after = runtime.allocation_counts();
        assert_eq!(after.allocations, before.allocations + 2);
        assert_eq!(after.frees, before.frees + 2);
    }
}

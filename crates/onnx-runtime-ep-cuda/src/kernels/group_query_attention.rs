//! CUDA implementation of `com.microsoft::GroupQueryAttention`.
//!
//! BSH query/key/value inputs are prepared into descriptor-selected KV buffers
//! with NVRTC kernels, including cache append and optional RoPE. Multi-token
//! prefill then uses the shared tiled online-softmax flash kernel when its
//! measured shape gate wins; decode and unsupported/slower shapes retain the
//! existing attention baseline. Declared present shapes remain BNSH-compatible
//! while the native backend may physically store converted paths as BSNH.

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    DeviceGraphResource, EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut,
    TensorView, WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node};
use onnx_runtime_memory_governor::MemoryRole;

use crate::error::driver_err;
use crate::kernels::kv_stride::{KvCachePath, KvCacheStrides};
use crate::runtime::{CudaRuntime, GraphDeviceAllocation, cuptr};

use super::attention::{AttentionDtype, run_attention_phase2a};
use super::flash_attention;
use super::gqa_decode;
use super::gqa_decode_bf16;
use super::gqa_decode_fp16;

const PREP_SRC: &str = r#"
// Default (head-major BNSH) KV-cache index. A non-default layout prepends its
// own descriptor-generated definition before this source.
#ifndef GQA_KV_INDEX
#define GQA_KV_INDEX(b, h, slot, heads, capacity, dim) \
    ( ((long)((b) * (heads) + (h)) * (capacity) + (slot)) * (dim) )
#endif
#ifndef GQA_KV_DST
#define GQA_KV_DST(b, h, slot) \
    GQA_KV_INDEX(b, h, slot, kv_heads, present_capacity, dim)
#endif
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
    // Latch-first poison propagation: if any prior captured kernel (an earlier
    // step, an earlier replay, or an earlier layer in this replay) already
    // recorded a bounds violation, force this step into the sentinel/skip state
    // deterministically. A later in-range step can therefore never resume writes
    // over a KV row that a poisoned step skipped, so no stale hole can form.
    // `atomicOr(flag, 0)` performs a coherent global read without mutating it.
    if (error_flag && atomicOr(error_flag, 0) != 0) {
        total_lengths[b] = -1;
        past_lengths[b] = -1;
        query_starts[b] = -1;
        return;
    }
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
                : query_start + s;
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

__device__ __forceinline__ int gqa_prepare_metadata_batch1(
    const int* seqlens_k, int* total_lengths, int* past_lengths,
    int* query_starts, int past_capacity, int present_capacity,
    const long long* position_ids, int validate_positions, int cache_rows,
    int* error_flag, int write_metadata)
{
    if (error_flag && atomicOr(error_flag, 0) != 0) {
        if (write_metadata) {
            total_lengths[0] = -1;
            past_lengths[0] = -1;
            query_starts[0] = -1;
        }
        return -1;
    }
    const long long total = (long long)seqlens_k[0] + 1;
    const long long past = total - 1;
    const long long query_start = total - 1;
    int error = 0;
    if (total > 2147483647LL) error |= 1;
    if (past < 0) error |= 2;
    if (query_start < 0) error |= 4;
    if (past > past_capacity) error |= 8;
    if (total > present_capacity) error |= 16;
    if (validate_positions) {
        const long long position = position_ids ? position_ids[0] : past;
        if (position < 0 || position >= (long long)cache_rows) error |= 32;
    }
    if (error) {
        if (error_flag) atomicOr(error_flag, error);
        if (write_metadata) {
            total_lengths[0] = -1;
            past_lengths[0] = -1;
            query_starts[0] = -1;
        }
        return -1;
    }
    if (write_metadata) {
        total_lengths[0] = (int)total;
        past_lengths[0] = (int)past;
        query_starts[0] = (int)query_start;
    }
    return (int)past;
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
        value = past[GQA_KV_INDEX(b, h, s, heads, past_capacity, dim) + d];
    } else if (s >= past_len && s < past_len + seq) {
        const int current_s = s - past_len;
        value = current[((b * seq + current_s) * heads + h) * dim + d];
    }
    present[GQA_KV_INDEX(b, h, s, heads, present_capacity, dim) + d] = value;
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
    present[GQA_KV_INDEX(b, h, target_s, heads, present_capacity, dim) + d] =
        current[((b * seq + s) * heads + h) * dim + d];
}

extern "C" __global__ void gqa_rope_bnsh(
    float* tensor, const float* cos_cache, const float* sin_cache,
    const long long* position_ids, const int* past_lengths,
    int batch, int seq, int heads, int dim, int rotary_dim, int tensor_capacity,
    int current_offset, int cache_rows, int interleaved, int cache_is_half)
{
    (void)cache_is_half;
    const int half = rotary_dim / 2;
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
    const size_t base = current_offset
        ? GQA_KV_INDEX(b, h, tensor_s, heads, tensor_capacity, dim)
        : ((size_t)(b * heads + h) * tensor_capacity + tensor_s) * dim;
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
    const float softcap,
    const float* head_sink)
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

    // Learned attention sink (gpt-oss family): a per-head logit that enters the
    // softmax denominator but carries no value vector. Thread 0 folds it into
    // `maximum`, then publishes its exp() contribution so every thread adds it
    // to the shared denominator `sum`. `head_sink == nullptr` => 0 contribution
    // (byte-identical to the pre-sink path).
    __shared__ float sink_contrib;
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
        if (head_sink != nullptr) {
            maximum = fmaxf(maximum, head_sink[query_head]);
        }
        for (int key_pos = 0; key_pos < total; ++key_pos) {
            float probability = isfinite(row_scores[key_pos])
                ? (float)exp((double)(row_scores[key_pos] - maximum))
                : 0.0f;
            row_scores[key_pos] = probability;
        }
        sink_contrib = (head_sink != nullptr)
            ? (float)exp((double)(head_sink[query_head] - maximum))
            : 0.0f;
    }
    __syncthreads();

    float sum = sink_contrib;
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

// Fused single-token (Sq=1, Sk=1) decode prep: split (implicit, reads packed or
// unpacked source directly), BSH->BNSH transpose (identity for Sq=1), in-place
// KV cache append, and RoPE for Q and present-K -- all in one launch. Replaces
// the split+transpose_in+append(K,V)+rope(Q,K) chain on the aliased device-KV
// decode path. For batch 1, metadata is derived independently by thread 0 of
// every CTA and shared within that CTA; block 0 also writes the attention
// metadata arrays. This avoids a cross-CTA dependency while preserving the
// sticky error latch and sentinel-gated cache writes.
extern "C" __global__ void gqa_fuse_decode_prep(
    const float* q_src, const float* k_src, const float* v_src, int packed,
    float* q_bnsh, float* present_k, float* present_v,
    const int* seqlens_k, int* total_lengths, int* past_lengths,
    int* query_starts, int past_capacity, int* error_flag, int derive_metadata,
    const float* cos_cache, const float* sin_cache,
    const long long* position_ids, int batch, int q_heads, int kv_heads, int dim,
    int rotary_dim, int present_capacity, int cache_rows, int do_rotary,
    int interleaved, int cache_is_half)
{
    (void)cache_is_half;
    const int head_half = dim / 2;
    const int rotary_half = rotary_dim / 2;
    const int qN = q_heads * head_half;
    const int kvN = kv_heads * head_half;
    const int per_batch = qN + 2 * kvN;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int batch1_past;
    if (derive_metadata) {
        if (threadIdx.x == 0) {
            batch1_past = gqa_prepare_metadata_batch1(
                seqlens_k, total_lengths, past_lengths, query_starts,
                past_capacity, present_capacity, position_ids, do_rotary,
                cache_rows, error_flag, blockIdx.x == 0);
        }
        __syncthreads();
    }
    if (idx >= batch * per_batch) return;
    const int b = idx / per_batch;
    int local = idx - b * per_batch;
    const int past = derive_metadata ? batch1_past : past_lengths[b];
    const int q_hidden = q_heads * dim;
    const int kv_hidden = kv_heads * dim;
    const int packed_hidden = q_hidden + 2 * kv_hidden;
    const long long position = position_ids ? position_ids[b] : (long long)past;
    const int rope_ok = do_rotary && position >= 0 && position < (long long)cache_rows;
    const int pos = rope_ok ? (int)position : 0;
    int region, h, k;
    if (local < qN) { region = 0; h = local / head_half; k = local % head_half; }
    else if (local < qN + kvN) { local -= qN; region = 1; h = local / head_half; k = local % head_half; }
    else { local -= qN + kvN; region = 2; h = local / head_half; k = local % head_half; }
    const int is_rotary = k < rotary_half;
    const int tail = rotary_dim + 2 * (k - rotary_half);
    const int d0 = is_rotary ? (interleaved ? 2 * k : k) : tail;
    const int d1 = is_rotary ? (interleaved ? 2 * k + 1 : k + rotary_half) : tail + 1;
    if (region == 0) {
        if (!q_bnsh) return;
        const long src = (long)b * (packed ? packed_hidden : q_hidden) + (long)h * dim;
        const long dst = (long)(b * q_heads + h) * dim;
        const float x0 = q_src[src + d0];
        const float x1 = q_src[src + d1];
        if (rope_ok && past >= 0 && is_rotary) {
            const float c = cos_cache[pos * rotary_half + k];
            const float sn = sin_cache[pos * rotary_half + k];
            q_bnsh[dst + d0] = __fsub_rn(__fmul_rn(c, x0), __fmul_rn(sn, x1));
            q_bnsh[dst + d1] = __fadd_rn(__fmul_rn(sn, x0), __fmul_rn(c, x1));
        } else {
            q_bnsh[dst + d0] = x0;
            q_bnsh[dst + d1] = x1;
        }
        return;
    }
    if (past < 0 || past >= present_capacity) return;
    const long dst = GQA_KV_DST(b, h, past);
    if (region == 1) {
        const long src = (long)b * (packed ? packed_hidden : kv_hidden)
                       + (packed ? q_hidden : 0) + (long)h * dim;
        const float x0 = k_src[src + d0];
        const float x1 = k_src[src + d1];
        if (rope_ok && is_rotary) {
            const float c = cos_cache[pos * rotary_half + k];
            const float sn = sin_cache[pos * rotary_half + k];
            present_k[dst + d0] = __fsub_rn(__fmul_rn(c, x0), __fmul_rn(sn, x1));
            present_k[dst + d1] = __fadd_rn(__fmul_rn(sn, x0), __fmul_rn(c, x1));
        } else {
            present_k[dst + d0] = x0;
            present_k[dst + d1] = x1;
        }
    } else {
        const long src = (long)b * (packed ? packed_hidden : kv_hidden)
                       + (packed ? (q_hidden + kv_hidden) : 0) + (long)h * dim;
        present_v[dst + d0] = v_src[src + d0];
        present_v[dst + d1] = v_src[src + d1];
    }
}
"#;

const PREP_HALF_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Default (head-major BNSH) KV-cache index; a non-default layout prepends its
// descriptor-generated definition and these guards yield to it.
#ifndef GQA_KV_INDEX
#define GQA_KV_INDEX(b, h, slot, heads, capacity, dim) \
    ( ((long)((b) * (heads) + (h)) * (capacity) + (slot)) * (dim) )
#endif
#ifndef GQA_KV_DST
#define GQA_KV_DST(b, h, slot) \
    GQA_KV_INDEX(b, h, slot, kv_heads, present_capacity, dim)
#endif

__device__ __forceinline__ int gqa_prepare_metadata_batch1(
    const int* seqlens_k, int* total_lengths, int* past_lengths,
    int* query_starts, int past_capacity, int present_capacity,
    const long long* position_ids, int validate_positions, int cache_rows,
    int* error_flag, int write_metadata)
{
    if (error_flag && atomicOr(error_flag, 0) != 0) {
        if (write_metadata) {
            total_lengths[0] = -1;
            past_lengths[0] = -1;
            query_starts[0] = -1;
        }
        return -1;
    }
    const long long total = (long long)seqlens_k[0] + 1;
    const long long past = total - 1;
    const long long query_start = total - 1;
    int error = 0;
    if (total > 2147483647LL) error |= 1;
    if (past < 0) error |= 2;
    if (query_start < 0) error |= 4;
    if (past > past_capacity) error |= 8;
    if (total > present_capacity) error |= 16;
    if (validate_positions) {
        const long long position = position_ids ? position_ids[0] : past;
        if (position < 0 || position >= (long long)cache_rows) error |= 32;
    }
    if (error) {
        if (error_flag) atomicOr(error_flag, error);
        if (write_metadata) {
            total_lengths[0] = -1;
            past_lengths[0] = -1;
            query_starts[0] = -1;
        }
        return -1;
    }
    if (write_metadata) {
        total_lengths[0] = (int)total;
        past_lengths[0] = (int)past;
        query_starts[0] = (int)query_start;
    }
    return (int)past;
}

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

__device__ __forceinline__ float gqa_load_cache(
    const void* cache, int index, int cache_is_half) {
    // `cache_is_half` is a tri-state cache-dtype tag: 0 = float32, 1 = float16,
    // 2 = bfloat16. All three decode to a float rotary coefficient.
    if (cache_is_half == 1) {
        return __half2float(reinterpret_cast<const __half*>(cache)[index]);
    }
    if (cache_is_half == 2) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(cache)[index]);
    }
    return reinterpret_cast<const float*>(cache)[index];
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
        result = past[GQA_KV_INDEX(b, h, s, heads, past_capacity, dim) + d];
    } else if (s >= past_len && s < past_len + seq) {
        const int current_s = s - past_len;
        result = current[((b * seq + current_s) * heads + h) * dim + d];
    }
    present[GQA_KV_INDEX(b, h, s, heads, present_capacity, dim) + d] = result;
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
    present[GQA_KV_INDEX(b, h, target_s, heads, present_capacity, dim) + d] =
        current[((b * seq + s) * heads + h) * dim + d];
}

template <typename T>
__device__ void gqa_rope_bnsh_body(
    T* tensor, const void* cos_cache, const void* sin_cache,
    const long long* position_ids, const int* past_lengths,
    int batch, int seq, int heads, int dim, int rotary_dim, int tensor_capacity,
    int current_offset, int cache_rows, int interleaved, int cache_is_half)
{
    const int half = rotary_dim / 2;
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
    const size_t base = current_offset
        ? GQA_KV_INDEX(b, h, tensor_s, heads, tensor_capacity, dim)
        : ((size_t)(b * heads + h) * tensor_capacity + tensor_s) * dim;
    const float x0 = gqa_load<T>(tensor[base + d0]);
    const float x1 = gqa_load<T>(tensor[base + d1]);
    const float c = gqa_load_cache(cos_cache, pos * half + k, cache_is_half);
    const float sn = gqa_load_cache(sin_cache, pos * half + k, cache_is_half);
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

// Half/bf16 counterpart of `gqa_fuse_decode_prep` (see the f32 source for the
// full contract). RoPE reads/writes go through the shared float load/store and
// cache helpers so the fused result is bit-identical to the unfused
// split+transpose+append+rope chain; raw (non-rotary) writes stay direct T
// copies to avoid any extra float round-trip.
template <typename T>
__device__ void gqa_fuse_decode_prep_body(
    const T* q_src, const T* k_src, const T* v_src, int packed,
    T* q_bnsh, T* present_k, T* present_v,
    const int* seqlens_k, int* total_lengths, int* past_lengths,
    int* query_starts, int past_capacity, int* error_flag, int derive_metadata,
    const void* cos_cache, const void* sin_cache,
    const long long* position_ids, int batch, int q_heads, int kv_heads, int dim,
    int rotary_dim, int present_capacity, int cache_rows, int do_rotary,
    int interleaved, int cache_is_half)
{
    const int head_half = dim / 2;
    const int rotary_half = rotary_dim / 2;
    const int qN = q_heads * head_half;
    const int kvN = kv_heads * head_half;
    const int per_batch = qN + 2 * kvN;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int batch1_past;
    if (derive_metadata) {
        if (threadIdx.x == 0) {
            batch1_past = gqa_prepare_metadata_batch1(
                seqlens_k, total_lengths, past_lengths, query_starts,
                past_capacity, present_capacity, position_ids, do_rotary,
                cache_rows, error_flag, blockIdx.x == 0);
        }
        __syncthreads();
    }
    if (idx >= batch * per_batch) return;
    const int b = idx / per_batch;
    int local = idx - b * per_batch;
    const int past = derive_metadata ? batch1_past : past_lengths[b];
    const int q_hidden = q_heads * dim;
    const int kv_hidden = kv_heads * dim;
    const int packed_hidden = q_hidden + 2 * kv_hidden;
    const long long position = position_ids ? position_ids[b] : (long long)past;
    const int rope_ok = do_rotary && position >= 0 && position < (long long)cache_rows;
    const int pos = rope_ok ? (int)position : 0;
    int region, h, k;
    if (local < qN) { region = 0; h = local / head_half; k = local % head_half; }
    else if (local < qN + kvN) { local -= qN; region = 1; h = local / head_half; k = local % head_half; }
    else { local -= qN + kvN; region = 2; h = local / head_half; k = local % head_half; }
    const int is_rotary = k < rotary_half;
    const int tail = rotary_dim + 2 * (k - rotary_half);
    const int d0 = is_rotary ? (interleaved ? 2 * k : k) : tail;
    const int d1 = is_rotary ? (interleaved ? 2 * k + 1 : k + rotary_half) : tail + 1;
    if (region == 0) {
        if (!q_bnsh) return;
        const long src = (long)b * (packed ? packed_hidden : q_hidden) + (long)h * dim;
        const long dst = (long)(b * q_heads + h) * dim;
        if (rope_ok && past >= 0 && is_rotary) {
            const float x0 = gqa_load<T>(q_src[src + d0]);
            const float x1 = gqa_load<T>(q_src[src + d1]);
            const float c = gqa_load_cache(cos_cache, pos * rotary_half + k, cache_is_half);
            const float sn = gqa_load_cache(sin_cache, pos * rotary_half + k, cache_is_half);
            q_bnsh[dst + d0] = gqa_store<T>(c * x0 - sn * x1);
            q_bnsh[dst + d1] = gqa_store<T>(sn * x0 + c * x1);
        } else {
            q_bnsh[dst + d0] = q_src[src + d0];
            q_bnsh[dst + d1] = q_src[src + d1];
        }
        return;
    }
    if (past < 0 || past >= present_capacity) return;
    const long dst = GQA_KV_DST(b, h, past);
    if (region == 1) {
        const long src = (long)b * (packed ? packed_hidden : kv_hidden)
                       + (packed ? q_hidden : 0) + (long)h * dim;
        if (rope_ok && is_rotary) {
            const float x0 = gqa_load<T>(k_src[src + d0]);
            const float x1 = gqa_load<T>(k_src[src + d1]);
            const float c = gqa_load_cache(cos_cache, pos * rotary_half + k, cache_is_half);
            const float sn = gqa_load_cache(sin_cache, pos * rotary_half + k, cache_is_half);
            present_k[dst + d0] = gqa_store<T>(c * x0 - sn * x1);
            present_k[dst + d1] = gqa_store<T>(sn * x0 + c * x1);
        } else {
            present_k[dst + d0] = k_src[src + d0];
            present_k[dst + d1] = k_src[src + d1];
        }
    } else {
        const long src = (long)b * (packed ? packed_hidden : kv_hidden)
                       + (packed ? (q_hidden + kv_hidden) : 0) + (long)h * dim;
        present_v[dst + d0] = v_src[src + d0];
        present_v[dst + d1] = v_src[src + d1];
    }
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
    TYPE* tensor, const void* cos_cache, const void* sin_cache, \
    const long long* position_ids, const int* past_lengths, \
    int batch, int seq, int heads, int dim, int rotary_dim, int tensor_capacity, \
    int current_offset, int cache_rows, int interleaved, int cache_is_half) { \
    gqa_rope_bnsh_body<TYPE>(tensor, cos_cache, sin_cache, position_ids, past_lengths, \
                             batch, seq, heads, dim, rotary_dim, tensor_capacity, current_offset, \
                             cache_rows, interleaved, cache_is_half); \
} \
extern "C" __global__ void gqa_transpose_bnsh_to_bsh_##SUFFIX( \
    const TYPE* src, TYPE* dst, int batch, int seq, int heads, int dim) { \
    gqa_transpose_bnsh_to_bsh_body<TYPE>(src, dst, batch, seq, heads, dim); \
} \
extern "C" __global__ void gqa_fuse_decode_prep_##SUFFIX( \
    const TYPE* q_src, const TYPE* k_src, const TYPE* v_src, int packed, \
    TYPE* q_bnsh, TYPE* present_k, TYPE* present_v, \
    const int* seqlens_k, int* total_lengths, int* past_lengths, \
    int* query_starts, int past_capacity, int* error_flag, int derive_metadata, \
    const void* cos_cache, const void* sin_cache, \
    const long long* position_ids, int batch, int q_heads, int kv_heads, int dim, \
    int rotary_dim, int present_capacity, int cache_rows, int do_rotary, \
    int interleaved, int cache_is_half) { \
    gqa_fuse_decode_prep_body<TYPE>(q_src, k_src, v_src, packed, q_bnsh, present_k, \
        present_v, seqlens_k, total_lengths, past_lengths, query_starts, past_capacity, \
        error_flag, derive_metadata, cos_cache, sin_cache, position_ids, batch, \
        q_heads, kv_heads, dim, rotary_dim, present_capacity, cache_rows, do_rotary, \
        interleaved, cache_is_half); \
}

DEFINE_GQA_HALF_KERNELS(__half, f16)
DEFINE_GQA_HALF_KERNELS(__nv_bfloat16, bf16)
"#;

const PREP_MODULE: &str = "group_query_attention_prep_v4";
const PREP_HALF_MODULE: &str = "group_query_attention_prep_half_v4";
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
const WS_COUNT: usize = 11;

/// Device-pointer alignment for the governed GQA session-persistent composite
/// workspace and every sub-buffer carved from it (packed Q/K/V staging,
/// BSH↔BNSH transpose scratch, and the f32 reference score matrix, §736).
const GQA_SCORES_ALIGN: usize = 256;

/// Whether a concrete GQA dispatch materializes an `[B, H, Sq, kv]` device score
/// matrix. Only the f32 *reference* attention path does: every fused-flash and
/// capture-safe split-K decode path (fused flash, f32 `gqa_decode`, fp16
/// `gqa_decode_fp16`) streams softmax through registers/shared memory and needs
/// no score scratch, and the phase-2a path owns its own scratch inside
/// `run_attention_phase2a`. The reference branch is reachable only when the
/// query dtype is f32 and the single-token split-K decode kernel does not cover
/// the shape (`Sq > 1` prefill, or `head_dim > 128`). This is the static signal
/// (#751) that keeps the governed reservation off every path that never touches
/// scores, so no device capacity is charged for scratch it never materializes.
fn gqa_reference_scores_path(dtype: DataType, q_seq: usize, head_dim: usize) -> bool {
    dtype == DataType::Float32 && !gqa_decode::supported(q_seq, head_dim)
}

/// Byte size of the governed GQA reference score buffer for one concrete
/// geometry. Prepare-only planning (`Kernel::workspace_requirement`) and
/// execution (the reference branch of `run`) both size the reservation through
/// this identical helper so the reserved and consumed byte counts cannot drift.
/// The buffer holds `batch·num_heads·q_seq·kv_capacity` f32 scores, matching the
/// reference kernel's `score_count · sizeof(f32)` indexing (a degenerate
/// geometry still reserves one element, as the kernel's `score_count.max(1)`
/// did).
fn gqa_reference_scores_bytes(
    batch: usize,
    num_heads: usize,
    q_seq: usize,
    kv_capacity: usize,
) -> Result<usize> {
    let rows = batch
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(q_seq))
        .ok_or_else(|| {
            EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: attention row count overflow".into(),
            )
        })?;
    let elements = rows.checked_mul(kv_capacity).ok_or_else(|| {
        EpError::KernelFailed("cuda_ep GroupQueryAttention: score scratch size overflow".into())
    })?;
    elements
        .max(1)
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: score scratch byte count overflow".into(),
            )
        })
}

/// Round `bytes` up to the next multiple of `align` (a power of two), erroring
/// on overflow rather than wrapping. Used to place each governed workspace
/// sub-buffer on a `GQA_SCORES_ALIGN`-aligned device offset.
fn gqa_align_up(bytes: usize, align: usize) -> Result<usize> {
    debug_assert!(align.is_power_of_two());
    bytes
        .checked_add(align - 1)
        .map(|value| value & !(align - 1))
        .ok_or_else(|| {
            EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: workspace offset alignment overflow".into(),
            )
        })
}

/// Byte size of one packed-QKV projection-staging buffer
/// (`batch·seq·hidden·element_size`). The unfused prep path splits an
/// interleaved `[B, S, (H + 2·Hkv)·D]` query tensor into these Q/K/V scratch
/// buffers before the BSH->BNSH transpose and cache append. Prepare-only
/// planning (`Kernel::workspace_requirement`) and execution (the split branch of
/// `run`) size the reservation through this identical helper so the reserved and
/// consumed byte counts cannot drift.
fn gqa_packed_staging_bytes(
    batch: usize,
    seq: usize,
    hidden: usize,
    element_size: usize,
) -> Result<usize> {
    batch
        .checked_mul(seq)
        .and_then(|value| value.checked_mul(hidden))
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| {
            EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: packed QKV staging byte count overflow".into(),
            )
        })
}

/// Byte layout of the governed GQA session-persistent composite workspace. The
/// packed Q/K/V projection-staging buffers come first, followed by route-required
/// Q/output BNSH scratch, then the f32 reference score matrix last. The
/// capacity-dependent score matrix can therefore extend into the reserved peak
/// without perturbing shape-derived offsets. Packed `Sq==1` overlays its Q
/// staging and Q-BNSH roles because fused extraction and unfused splitting are
/// mutually exclusive. Any region a route does not populate is zero-length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GqaWorkspaceLayout {
    packed_q_offset: usize,
    packed_q_bytes: usize,
    packed_k_offset: usize,
    packed_k_bytes: usize,
    packed_v_offset: usize,
    packed_v_bytes: usize,
    q_bnsh_offset: usize,
    q_bnsh_bytes: usize,
    out_bnsh_offset: usize,
    out_bnsh_bytes: usize,
    scores_offset: usize,
    scores_bytes: usize,
    total_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GqaTransposeScratch {
    query: bool,
    output: bool,
}

/// Identify which BSH↔BNSH staging buffers a route actually populates.
///
/// For `Sq == 1`, BSH and BNSH have identical indexing. An unpacked query with
/// no RoPE can therefore be read directly by attention, while packed input
/// still needs extraction and RoPE still needs a writable transformed copy.
/// The output transpose is always unnecessary for `Sq == 1`.
fn gqa_transpose_scratch(packed_qkv: bool, do_rotary: bool, q_seq: usize) -> GqaTransposeScratch {
    GqaTransposeScratch {
        query: q_seq != 1 || packed_qkv || do_rotary,
        output: q_seq > 1,
    }
}

/// Compute the composite workspace layout for one concrete GQA geometry.
///
/// `want_staging` is true exactly when the dispatch splits a packed QKV tensor
/// (both key/value inputs absent); `want_scores` is true exactly on the f32
/// reference attention path (`gqa_reference_scores_path`). For a packed tensor
/// the key/value sequence length equals the query length, so the K/V staging
/// buffers are sized on `q_seq` and `k_hidden = kv_num_heads·head_dim`.
///
/// These governed classes are folded into one session-persistent requirement
/// because the executor reserves one slot per lifetime class and hands the
/// kernel one view. Reference prefill needs Q, output, scores, and (for packed
/// input) split staging live in the same dispatch, so non-exclusive regions
/// occupy disjoint ranges.
#[allow(clippy::too_many_arguments)]
fn gqa_workspace_layout(
    batch: usize,
    num_heads: usize,
    q_seq: usize,
    q_hidden: usize,
    k_hidden: usize,
    element_size: usize,
    want_staging: bool,
    want_q_bnsh: bool,
    want_out_bnsh: bool,
    scores_kv_capacity: usize,
    want_scores: bool,
) -> Result<GqaWorkspaceLayout> {
    let mut layout = GqaWorkspaceLayout::default();
    let mut offset = 0usize;
    if want_staging {
        let q_bytes = gqa_packed_staging_bytes(batch, q_seq, q_hidden, element_size)?;
        let kv_bytes = gqa_packed_staging_bytes(batch, q_seq, k_hidden, element_size)?;
        layout.packed_q_offset = offset;
        layout.packed_q_bytes = q_bytes;
        offset = offset
            .checked_add(gqa_align_up(q_bytes, GQA_SCORES_ALIGN)?)
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: workspace layout overflow".into(),
                )
            })?;
        layout.packed_k_offset = offset;
        layout.packed_k_bytes = kv_bytes;
        offset = offset
            .checked_add(gqa_align_up(kv_bytes, GQA_SCORES_ALIGN)?)
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: workspace layout overflow".into(),
                )
            })?;
        layout.packed_v_offset = offset;
        layout.packed_v_bytes = kv_bytes;
        offset = offset
            .checked_add(gqa_align_up(kv_bytes, GQA_SCORES_ALIGN)?)
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: workspace layout overflow".into(),
                )
            })?;
    }
    if want_q_bnsh {
        let q_bytes = gqa_packed_staging_bytes(batch, q_seq, q_hidden, element_size)?;
        if want_staging && q_seq == 1 {
            // Packed single-token routes are mutually exclusive:
            // - fused prep extracts Q directly into q_bnsh and does not split;
            // - unfused prep splits Q into packed_q, whose Sq==1 layout is
            //   already BNSH and can be consumed/rotated in place.
            // Overlay the two roles so planning reserves their peak, not their
            // impossible sum.
            layout.q_bnsh_offset = layout.packed_q_offset;
            layout.q_bnsh_bytes = q_bytes;
        } else {
            layout.q_bnsh_offset = offset;
            layout.q_bnsh_bytes = q_bytes;
            offset = offset
                .checked_add(gqa_align_up(q_bytes, GQA_SCORES_ALIGN)?)
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: workspace layout overflow".into(),
                    )
                })?;
        }
    }
    if want_out_bnsh {
        let out_bytes = gqa_packed_staging_bytes(batch, q_seq, q_hidden, element_size)?;
        layout.out_bnsh_offset = offset;
        layout.out_bnsh_bytes = out_bytes;
        offset = offset
            .checked_add(gqa_align_up(out_bytes, GQA_SCORES_ALIGN)?)
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: workspace layout overflow".into(),
                )
            })?;
    }
    // The score matrix is last and left unpadded: `offset` is already a multiple
    // of `GQA_SCORES_ALIGN` (a sum of aligned staging regions, or zero), so the
    // scores buffer starts aligned and the composite total equals exactly the
    // reference score bytes when no staging is present.
    if want_scores {
        let scores_bytes = gqa_reference_scores_bytes(batch, num_heads, q_seq, scores_kv_capacity)?;
        layout.scores_offset = offset;
        layout.scores_bytes = scores_bytes;
        offset = offset.checked_add(scores_bytes).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep GroupQueryAttention: workspace layout overflow".into())
        })?;
    }
    layout.total_bytes = offset;
    Ok(layout)
}

/// Carve a `CUdeviceptr` for the `[offset, offset + bytes)` sub-range of an
/// executor-prepared composite workspace. A view that cannot cover the range is
/// a deterministic error (the governed-slot shortfall contract, §736) rather
/// than a silent under-allocation.
fn gqa_carve(view: WorkspaceView, offset: usize, bytes: usize, what: &str) -> Result<CUdeviceptr> {
    let end = offset.checked_add(bytes).ok_or_else(|| {
        EpError::KernelFailed(format!(
            "cuda_ep GroupQueryAttention: workspace sub-range overflow carving {what}"
        ))
    })?;
    if end > view.bytes() {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep GroupQueryAttention: prepared workspace {} bytes is smaller than the {end} \
             bytes required to carve {what}",
            view.bytes()
        )));
    }
    let base = view.ptr().0 as *mut u8;
    Ok(cuptr(base.wrapping_add(offset).cast_const().cast()))
}

/// Bit flags a captured GQA prep kernel `atomicOr`s into the runtime's latching
/// capture-error word when a decode-metadata invariant is violated during graph
/// replay. Exposed so hosts (and tests) can identify which bound was breached.
pub const GQA_CAPTURE_ERROR_TOTAL_OVERFLOW: u32 = 1;
pub const GQA_CAPTURE_ERROR_PAST_NEGATIVE: u32 = 2;
pub const GQA_CAPTURE_ERROR_QUERY_NEGATIVE: u32 = 4;
pub const GQA_CAPTURE_ERROR_PAST_CAPACITY: u32 = 8;
pub const GQA_CAPTURE_ERROR_PRESENT_CAPACITY: u32 = 16;
pub const GQA_CAPTURE_ERROR_POSITION: u32 = 32;

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
        let kv_layout = node.attr("kv_layout").and_then(|a| a.as_int()).unwrap_or(0);
        // Build the KV stride descriptor the native backend's `kv_layout`
        // attribute selects. `from_attribute` rejects any value other than
        // 0 (head-major BNSH) or 1 (seq-major BSNH) — the wire attribute carries
        // only the two named layouts.
        let kv_strides = KvCacheStrides::from_attribute(kv_layout)?;
        Ok(Box::new(
            GroupQueryAttentionKernel::new(
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
            )?
            .with_kv_strides(kv_strides),
        ))
    }
}

/// Claim-time capability gate for `com.microsoft::GroupQueryAttention`.
///
/// Mirrors the attribute-based rejections in
/// [`GroupQueryAttentionFactory::create`] so the EP declines nodes it cannot
/// execute *before* ORT commits them to a fused partition. Without this,
/// `create` would reject an unsupported attribute (e.g. `smooth_softmax`) only
/// at kernel-construction time, which sinks the entire fused partition instead
/// of letting ORT route the offending node to another EP. Only
/// attribute-derivable conditions are checked here; shape- and runtime-derived
/// conditions remain enforced in `create`.
pub(crate) fn unsupported_reason(node: &Node) -> Option<Cow<'static, str>> {
    let required_head = |name: &str| -> core::result::Result<usize, Cow<'static, str>> {
        let value = node.attr(name).and_then(|a| a.as_int()).ok_or_else(|| {
            Cow::Owned(format!(
                "cuda_ep GroupQueryAttention: missing required `{name}` attribute"
            ))
        })?;
        usize::try_from(value)
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| Cow::Owned(format!("cuda_ep GroupQueryAttention: `{name}` must be > 0")))
    };
    let num_heads = match required_head("num_heads") {
        Ok(value) => value,
        Err(reason) => return Some(reason),
    };
    let kv_num_heads = match required_head("kv_num_heads") {
        Ok(value) => value,
        Err(reason) => return Some(reason),
    };
    if !num_heads.is_multiple_of(kv_num_heads) {
        return Some(Cow::Owned(format!(
            "cuda_ep GroupQueryAttention: num_heads {num_heads} must be a multiple of kv_num_heads {kv_num_heads}"
        )));
    }
    for name in ["k_quant_type", "v_quant_type"] {
        if let Some(value) = node.attr(name)
            && value.as_str() != Some("NONE")
        {
            return Some(Cow::Owned(format!(
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
            return Some(Cow::Owned(format!(
                "cuda_ep GroupQueryAttention: {message} is not supported"
            )));
        }
    }
    let softcap = node
        .attr("softcap")
        .and_then(|a| a.as_float())
        .unwrap_or(0.0);
    if softcap < 0.0 {
        return Some(Cow::Borrowed(
            "cuda_ep GroupQueryAttention: softcap must be non-negative",
        ));
    }
    let kv_layout = node.attr("kv_layout").and_then(|a| a.as_int()).unwrap_or(0);
    if !matches!(kv_layout, 0 | 1) {
        return Some(Cow::Owned(format!(
            "cuda_ep GroupQueryAttention: kv_layout {kv_layout} is not supported (expected 0 or 1)"
        )));
    }
    None
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
    prep_fusion_disabled: bool,
    /// KV cache physical layout as a stride descriptor. The default is
    /// head-major BNSH (ORT-compatible), which every reader and writer honors.
    /// The native backend sets this to seq-major BSNH on GQA nodes whose KV it
    /// exclusively owns. Converted flash-prefill and fp16 decode kernels
    /// generate their index arithmetic from this descriptor at NVRTC
    /// module-build time; every unconverted path rejects it.
    kv_strides: KvCacheStrides,
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

#[derive(Clone, Debug, Default)]
struct WorkspaceSlot {
    allocation: Option<Arc<GraphDeviceAllocation>>,
    bytes: usize,
}

#[derive(Debug)]
struct GqaWorkspace {
    runtime: Arc<CudaRuntime>,
    slots: [WorkspaceSlot; WS_COUNT],
    used: [bool; WS_COUNT],
}

impl GqaWorkspace {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            slots: std::array::from_fn(|_| WorkspaceSlot::default()),
            used: [false; WS_COUNT],
        }
    }

    fn begin_call(&mut self) {
        self.used.fill(false);
    }

    fn reserve(&mut self, index: usize, bytes: usize) -> Result<CUdeviceptr> {
        self.used[index] = true;
        let bytes = bytes.max(1);
        let slot = &self.slots[index];
        if slot.bytes >= bytes
            && let Some(allocation) = slot.allocation.as_ref()
        {
            return Ok(allocation.ptr());
        }
        if self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: workspace slot {index} requires {bytes} bytes during CUDA graph capture; warm the fixed decode shape before capture"
            )));
        }
        let allocation = GraphDeviceAllocation::allocate(&self.runtime, bytes)?;
        if slot.allocation.is_some() {
            // Dynamic prefill/growing-cache shapes may outgrow a slot. Preserve
            // the fixed-capacity decode fast path (which never reaches here),
            // but wait before replacing storage that queued work may still use.
            self.runtime.synchronize()?;
        }
        let ptr = allocation.ptr();
        self.slots[index] = WorkspaceSlot {
            allocation: Some(allocation),
            bytes,
        };
        Ok(ptr)
    }

    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        self.slots
            .iter()
            .zip(self.used)
            .filter_map(|(slot, used)| used.then_some(slot.allocation.as_ref()).flatten())
            .map(GraphDeviceAllocation::device_graph_resource)
            .collect()
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
            "cuda_ep GroupQueryAttention: dtype, decode mode, or shape changed during CUDA graph capture; warm the exact one-token fixed device-KV signature before capture".into(),
        ));
    }
    Ok(())
}

/// Human-readable description of the invariant(s) a captured GQA decode step
/// latched into the runtime capture-error word, given its raw bitmask. Returns
/// `None` for a zero (un-poisoned) mask.
pub fn gqa_capture_error_description(error: u32) -> Option<String> {
    if error == 0 {
        return None;
    }
    let mut violations = Vec::new();
    if error & GQA_CAPTURE_ERROR_TOTAL_OVERFLOW != 0 {
        violations.push("seqlens_k + 1 overflows int32");
    }
    if error & GQA_CAPTURE_ERROR_PAST_NEGATIVE != 0 {
        violations.push("seqlens_k + 1 is shorter than current key sequence");
    }
    if error & GQA_CAPTURE_ERROR_QUERY_NEGATIVE != 0 {
        violations.push("seqlens_k + 1 is shorter than current query sequence");
    }
    if error & GQA_CAPTURE_ERROR_PAST_CAPACITY != 0 {
        violations.push("effective past length exceeds past cache extent");
    }
    if error & GQA_CAPTURE_ERROR_PRESENT_CAPACITY != 0 {
        violations.push("valid sequence length exceeds present cache capacity");
    }
    if error & GQA_CAPTURE_ERROR_POSITION != 0 {
        violations.push("position_ids or implicit rotary position exceeds cache rows");
    }
    if violations.is_empty() {
        violations.push("unrecognized capture-safety violation");
    }
    Some(violations.join("; "))
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

fn validate_sequence_lengths_shape(shape: &[usize], numel: usize, batch: usize) -> Result<()> {
    if shape == [batch] || shape == [batch, 1] {
        return Ok(());
    }

    let scalar = shape.is_empty() && numel == 1;
    if scalar {
        return if batch == 1 {
            Ok(())
        } else {
            Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: scalar seqlens_k can only be promoted to [1] when batch_size is 1, got batch_size {batch}; provide contiguous int32 [batch_size] or [batch_size, 1] values for every row"
            )))
        };
    }

    Err(EpError::KernelFailed(format!(
        "cuda_ep GroupQueryAttention: seqlens_k must be non-negative contiguous int32 with shape [batch_size], [batch_size, 1], or a scalar for batch_size 1 (for batch_size {batch}: [{batch}] or [{batch}, 1]), got shape {shape:?}"
    )))
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
        let launch_count: usize = $count;
        if launch_count != 0 {
            let function = $runtime.nvrtc_function($module, $source, $entry)?;
            let grid =
                u32::try_from(launch_count.div_ceil(BLOCK as usize)).map_err(|_| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: launch grid exceeds u32".into(),
                    )
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
        }
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
            prep_fusion_disabled: false,
            kv_strides: KvCacheStrides::head_major_bnsh(),
            last_capture_safe_signature: Mutex::new(None),
        })
    }

    pub fn with_backend(mut self, backend: GroupQueryAttentionBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Select the KV cache stride descriptor. The native backend sets a
    /// non-default (seq-major) descriptor on GQA nodes whose KV it exclusively
    /// owns; converted paths are selected and hard-gated in [`Self::run`].
    pub(crate) fn with_kv_strides(mut self, kv_strides: KvCacheStrides) -> Self {
        self.kv_strides = kv_strides;
        self
    }

    /// Select the KV cache layout by its wire attribute value (0 = head-major
    /// BNSH, 1 = seq-major BSNH). A thin adapter over [`Self::with_kv_strides`]
    /// for callers that only have the integer attribute.
    pub fn with_kv_layout(self, kv_layout: i32) -> Self {
        let kv_strides = KvCacheStrides::from_attribute(i64::from(kv_layout))
            .unwrap_or_else(|_| KvCacheStrides::head_major_bnsh());
        self.with_kv_strides(kv_strides)
    }

    /// Forces the unfused per-op decode prep chain (split, transpose, append,
    /// RoPE, output transpose) instead of the single fused decode-prep launch.
    /// Exposed so tests can prove the fused kernel is bit-identical to the
    /// unfused reference path on the same inputs.
    pub fn with_prep_fusion_disabled(mut self, disabled: bool) -> Self {
        self.prep_fusion_disabled = disabled;
        self
    }

    /// Reads the internal metadata workspace for GPU parity tests.
    #[doc(hidden)]
    pub fn read_prepared_metadata_for_test(
        &self,
        batch: usize,
    ) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
        let workspace = self.workspace.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep GroupQueryAttention: workspace lock poisoned".into())
        })?;
        let read_slot = |index: usize| -> Result<Vec<i32>> {
            let slot = &workspace.slots[index];
            let bytes_len = batch
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: test metadata size overflow".into(),
                    )
                })?;
            let Some(allocation) = slot.allocation.as_ref() else {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: test metadata workspace is unavailable".into(),
                ));
            };
            if slot.bytes < bytes_len {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: test metadata workspace is unavailable".into(),
                ));
            }
            let mut bytes = vec![0u8; bytes_len];
            // SAFETY: the workspace slot is live and was reserved for at least
            // `bytes_len` bytes by the most recent successful execution.
            unsafe {
                self.runtime.dtoh(&mut bytes, allocation.ptr())?;
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|x| i32::from_ne_bytes([x[0], x[1], x[2], x[3]]))
                .collect())
        };
        Ok((
            read_slot(WS_TOTALS)?,
            read_slot(WS_PAST_LENGTHS)?,
            read_slot(WS_QUERY_STARTS)?,
        ))
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

    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        prepared: Option<WorkspaceView>,
    ) -> Result<()> {
        self.workspace
            .lock()
            .map_err(|_| {
                EpError::KernelFailed("cuda_ep GroupQueryAttention: workspace lock poisoned".into())
            })?
            .begin_call();
        let mut last_signature = self.last_capture_safe_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: capture signature lock poisoned".into(),
            )
        })?;
        let warmed_signature = last_signature.take();
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
            (12, "quantized-cache k_scale"),
            (13, "quantized-cache v_scale"),
        ] {
            if inputs.get(index).is_some_and(|v| !v.is_absent()) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: {feature} is not supported"
                )));
            }
        }
        // Optional learned attention sink (input 11): a per-query-head logit
        // vector (`[num_heads]`, f32) that joins the softmax denominator with no
        // value contribution (gpt-oss family). Supported only on the f32
        // reference and f32 split-K decode paths, which is where every f32
        // decoder with sinks lands once fused flash is disabled below. Any other
        // dtype/path with a sink present is rejected rather than silently
        // dropping the sink term.
        let head_sink = match inputs.get(11) {
            Some(v) if !v.is_absent() => {
                if v.dtype != DataType::Float32 {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep GroupQueryAttention: head_sink dtype {:?} is not supported; expected Float32",
                        v.dtype
                    )));
                }
                require_dense(v, "head_sink", v.dtype)?;
                Some(cuptr(v.data_ptr::<u8>() as *const c_void))
            }
            _ => None,
        };
        if head_sink.is_some() && inputs[0].dtype != DataType::Float32 {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: head_sink is only supported for Float32 query \
                 (reference and split-K decode paths)"
                    .into(),
            ));
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
            base_prep_module,
            base_prep_src,
            split_entry,
            transpose_in_entry,
            build_entry,
            append_entry,
            rope_entry,
            transpose_out_entry,
            fuse_entry,
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
                "gqa_fuse_decode_prep",
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
                "gqa_fuse_decode_prep_f16",
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
                "gqa_fuse_decode_prep_bf16",
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

        if inputs[5].dtype == DataType::Int32 && !inputs[5].is_contiguous() {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: non-contiguous seqlens_k was provided; expected non-negative contiguous int32 with shape [batch_size] or [batch_size, 1]"
                    .into(),
            ));
        }
        require_dense(&inputs[5], "seqlens_k", DataType::Int32)?;
        validate_sequence_lengths_shape(inputs[5].shape, inputs[5].numel(), batch)?;
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
        let (
            cos_ptr,
            sin_ptr,
            positions_ptr,
            cache_rows,
            cache_rows_usize,
            rotary_dim,
            rotary_dim_usize,
            rope_cache_is_half,
        ) = if self.do_rotary {
            if k_seq != 0 && q_seq != k_seq {
                return Err(EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: do_rotary requires equal query/key sequence lengths unless current key/value are empty".into(),
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
            require_dense(cos, "cos_cache", DataType::Float32)
                .or_else(|_| require_dense(cos, "cos_cache", DataType::Float16))
                .or_else(|_| require_dense(cos, "cos_cache", DataType::BFloat16))?;
            let cache_dtype = cos.dtype;
            // Tri-state cache-dtype tag consumed by `gqa_load_cache`:
            // 0 = Float32, 1 = Float16, 2 = BFloat16. Half/bf16 caches are only
            // valid alongside half-precision (Float16/BFloat16) queries.
            let cache_is_half = match cache_dtype {
                DataType::Float32 => 0i32,
                DataType::Float16 if matches!(q.dtype, DataType::Float16 | DataType::BFloat16) => {
                    1i32
                }
                DataType::BFloat16 if matches!(q.dtype, DataType::Float16 | DataType::BFloat16) => {
                    2i32
                }
                other => {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep GroupQueryAttention: cos_cache/sin_cache dtype {other:?} unsupported for query dtype {:?}; expected Float32, or Float16/BFloat16 with half-precision queries",
                        q.dtype
                    )));
                }
            };
            require_dense(sin, "sin_cache", cache_dtype)?;
            if cos.shape.len() != 2 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: cos_cache must be rank 2 [max_sequence_length, rotary_dim/2], got shape {:?}",
                    cos.shape
                )));
            }
            if sin.shape != cos.shape {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: sin_cache shape {:?} must exactly match cos_cache shape {:?} so both caches describe the same rotary_dim",
                    sin.shape, cos.shape
                )));
            }
            let rotary_dim_usize = cos.shape[1].checked_mul(2).ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "cuda_ep GroupQueryAttention: rotary_dim derived from cos_cache width {} overflows; use a finite cache width with 1 <= width <= head_size/2",
                        cos.shape[1]
                    ))
                })?;
            if rotary_dim_usize < 2
                || !rotary_dim_usize.is_multiple_of(2)
                || rotary_dim_usize > dim
                || cos.shape[1] != rotary_dim_usize / 2
            {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: rotary_dim derived from cos_cache width {} is {}; it must be even and satisfy 2 <= rotary_dim <= head_size={} (cache width must equal rotary_dim/2)",
                    cos.shape[1], rotary_dim_usize, dim
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
                checked_i32(rotary_dim_usize, "rotary dimension")?,
                rotary_dim_usize,
                cache_is_half,
            )
        } else {
            (0, 0, 0, 0, 0, 0, 0, 0)
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
            .and_then(|present_capacity| {
                // Structural preconditions shared by every capture-safe GQA
                // signature: an in-place, fixed-capacity, aliased device-KV cache
                // whose past and present extents match, so the prep append and
                // attention read replay against stable addresses.
                let structural = has_past_key
                    && present_capacity >= 1
                    && past_capacity == present_capacity
                    && aliased_device_kv;
                if !structural {
                    return None;
                }
                // Resolve the backend against the fixed `present_capacity` (never
                // the runtime valid length), so the warm and captured runs agree
                // on the dispatch and the signature stays stable.
                let backend =
                    self.selected_backend_for_shape(q.dtype, q_seq, present_capacity, dim);
                let eligible = match backend {
                    // Single-token device-KV decode (M=1): the existing Phase2a
                    // split-K flash-decode kernels read the valid length on-device.
                    GroupQueryAttentionBackend::Phase2a => {
                        (q.dtype == DataType::Float32
                            || (q.dtype == DataType::Float16
                                && gqa_decode_fp16::supported(q_seq, dim))
                            || (q.dtype == DataType::BFloat16
                                && gqa_decode_bf16::supported(q_seq, dim)))
                            && q_seq == 1
                            && k_seq <= 1
                    }
                    // Batched query-width > 1 (speculative-verify / prefill M=K):
                    // the fused flash kernel is capture-safe by construction
                    // (static grid, device `total_lengths`/`past_lengths`, no
                    // host read-back, no mid-launch sync). It handles the fp16/bf16
                    // shapes flash supports; f32 M>1 would take the reference-scores
                    // path (host-sized) and is left on the eager fallback.
                    GroupQueryAttentionBackend::Fused => {
                        matches!(q.dtype, DataType::Float16 | DataType::BFloat16)
                            && flash_attention::supported(q_seq, dim)
                            && q_seq > 1
                            && k_seq == q_seq
                    }
                    GroupQueryAttentionBackend::Auto => false,
                };
                eligible.then_some((present_capacity, backend))
            })
            .map(|(present_capacity, backend)| GqaCaptureSignature {
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
                    .enumerate()
                    .map(|(index, input)| {
                        (!input.is_absent()).then(|| {
                            if index == 5 {
                                vec![batch]
                            } else {
                                input.shape.to_vec()
                            }
                        })
                    })
                    .collect(),
                output_shapes: outputs.iter().map(|output| output.shape.to_vec()).collect(),
                backend,
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
        let mut validated_query_starts = None;
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
            let mut query_starts = Vec::with_capacity(batch);
            for &total in &totals {
                let past = total.checked_sub(current_key_length).ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: seqlens_k + 1 is shorter than current key sequence"
                            .into(),
                    )
                })?;
                let query_start = total.checked_sub(query_length).ok_or_else(|| {
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
                query_starts.push(query_start);
            }
            valid_sequence_length = Some(maximum);
            validated_query_starts = Some(query_starts);
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
            } else if validated_query_starts.as_ref().is_some_and(|starts| {
                starts
                    .iter()
                    .any(|&start| start as usize + q_seq > cache_rows_usize)
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

        // Single-token decode lets the trailing BNSH->BSH output transpose
        // collapse into the attention write (identical layouts when Sq==1), and
        // when the KV cache is appended in place (aliased device-KV, matching
        // past/present capacity) the whole split+transpose+append+RoPE prep
        // chain fuses into one launch. Prefill/growing-cache/non-aliased shapes
        // keep the unfused kernels.
        let single_token = q_seq == 1;
        let fuse_prep = single_token
            && k_seq == 1
            && dim.is_multiple_of(2)
            && has_past_key
            && aliased_device_kv
            && past_capacity == present_capacity
            && !self.prep_fusion_disabled;
        let fuse_metadata = fuse_prep && batch == 1;

        let attention_sequence_length = valid_sequence_length.unwrap_or(present_capacity);
        let mut selected_backend =
            self.selected_backend_for_shape(q.dtype, q_seq, attention_sequence_length, dim);
        // A learned attention sink is implemented only on the f32 reference and
        // f32 split-K decode paths (the fused-flash and phase-2a kernels have no
        // sink term). Route around fused flash whenever a sink is present so the
        // sink is never silently dropped; f32 decode/prefill then land on the
        // sink-aware kernels below.
        if head_sink.is_some() && selected_backend == GroupQueryAttentionBackend::Fused {
            selected_backend = GroupQueryAttentionBackend::Phase2a;
        }
        let use_fused = selected_backend == GroupQueryAttentionBackend::Fused;
        let head_sink_ptr = head_sink.unwrap_or(0);
        let prep_path = if fuse_prep && q.dtype == DataType::Float16 {
            KvCachePath::FusedDecodePrep
        } else if q_seq > 1 && use_fused {
            KvCachePath::FlashPrefillPrep
        } else {
            KvCachePath::UnfusedDecodePrep
        };
        self.kv_strides.require_converted_path_support(prep_path)?;
        let prep_src_owned;
        let (prep_module, prep_src): (&str, &str) = if self.kv_strides.is_head_major() {
            (base_prep_module, base_prep_src)
        } else {
            prep_src_owned = format!("{}{}", self.kv_strides.prep_prelude(), base_prep_src);
            let module = match q.dtype {
                DataType::Float32 => self.kv_strides.prep_f32_module_key()?,
                DataType::Float16 | DataType::BFloat16 => self.kv_strides.prep_half_module_key()?,
                _ => unreachable!(),
            };
            (module, prep_src_owned.as_str())
        };
        // Governed session-persistent composite workspace layout (§736). Planning
        // (`workspace_requirement`) and execution size packed Q/K/V staging,
        // BSH↔BNSH transpose scratch, and the f32 reference score matrix through
        // this identical helper, using the same route flags, so their offsets
        // and byte counts cannot drift.
        // `want_staging` mirrors planning (any packed-QKV dispatch) rather than
        // the exact `packed_qkv && !fuse_prep` split condition, so a fused decode
        // that happens not to split still carves stable offsets — it simply
        // leaves the (tiny, `q_seq==1`) staging region untouched.
        let transpose_scratch = gqa_transpose_scratch(packed_qkv, self.do_rotary, q_seq);
        let want_scores = gqa_reference_scores_path(q.dtype, q_seq, dim);
        let composite_layout = gqa_workspace_layout(
            batch,
            self.num_heads,
            q_seq,
            q_hidden,
            k_hidden,
            element_size,
            packed_qkv,
            transpose_scratch.query,
            transpose_scratch.output,
            present_capacity,
            want_scores,
        )?;
        let input_q_ptr = cuptr(q.data_ptr::<u8>() as *const c_void);
        let mut workspace = self.workspace.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep GroupQueryAttention: workspace lock poisoned".into())
        })?;
        let metadata_bytes = batch * std::mem::size_of::<i32>();
        let totals_gpu = workspace.reserve(WS_TOTALS, metadata_bytes)?;
        let past_lengths_gpu = workspace.reserve(WS_PAST_LENGTHS, metadata_bytes)?;
        let query_starts_gpu = workspace.reserve(WS_QUERY_STARTS, metadata_bytes)?;
        let metadata_error_gpu = if capture_candidate.is_some() {
            // Capture-safe decode steps latch any bounds violation into the
            // runtime-shared word so every captured GQA layer poisons the same
            // flag, and the host detects it once per step at the logits sync.
            self.runtime.capture_error_ptr()
        } else {
            0
        };
        // Packed QKV projection staging (§736). When the executor prepared the
        // composite workspace, carve the Q/K/V staging sub-buffers from it so the
        // bytes are charged against the device authority; otherwise (the
        // compatibility/opt-out `execute` path) keep them self-owned in the
        // pooled slot. The staging is materialized only when a packed tensor is
        // split on the unfused prep path (`packed_qkv && !fuse_prep`); every
        // unpacked-input and fused-decode dispatch carves nothing.
        let stage_packed = packed_qkv && !fuse_prep;
        let (packed_q, packed_k, packed_v) = if stage_packed {
            match prepared {
                Some(view) => (
                    Some(gqa_carve(
                        view,
                        composite_layout.packed_q_offset,
                        composite_layout.packed_q_bytes,
                        "packed query staging",
                    )?),
                    Some(gqa_carve(
                        view,
                        composite_layout.packed_k_offset,
                        composite_layout.packed_k_bytes,
                        "packed key staging",
                    )?),
                    Some(gqa_carve(
                        view,
                        composite_layout.packed_v_offset,
                        composite_layout.packed_v_bytes,
                        "packed value staging",
                    )?),
                ),
                None => (
                    Some(workspace.reserve(WS_PACKED_Q, batch * q_seq * q_hidden * element_size)?),
                    Some(workspace.reserve(WS_PACKED_K, batch * k_seq * k_hidden * element_size)?),
                    Some(workspace.reserve(WS_PACKED_V, batch * k_seq * k_hidden * element_size)?),
                ),
            }
        } else {
            (None, None, None)
        };
        // BSH↔BNSH transpose scratch (§736). Prepared execution carves both
        // buffers from the governed composite. The compatibility/opt-out path
        // retains the self-owned pooled slots. For Sq==1, unpacked non-RoPE Q is
        // already in the layout attention consumes and uses the input directly.
        // An unfused packed Sq==1 route similarly consumes/rotates its split Q
        // staging in place; the layout overlays that mutually-exclusive role.
        let q_bnsh = if !transpose_scratch.query {
            input_q_ptr
        } else if single_token && stage_packed {
            packed_q.ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: packed single-token query staging missing".into(),
                )
            })?
        } else {
            match prepared {
                Some(view) => gqa_carve(
                    view,
                    composite_layout.q_bnsh_offset,
                    composite_layout.q_bnsh_bytes,
                    "query BNSH transpose scratch",
                )?,
                None => workspace.reserve(
                    WS_Q_BNSH,
                    gqa_packed_staging_bytes(batch, q_seq, q_hidden, element_size)?,
                )?,
            }
        };
        let out_bnsh = if transpose_scratch.output {
            Some(match prepared {
                Some(view) => gqa_carve(
                    view,
                    composite_layout.out_bnsh_offset,
                    composite_layout.out_bnsh_bytes,
                    "output BNSH transpose scratch",
                )?,
                None => workspace.reserve(
                    WS_OUT_BNSH,
                    gqa_packed_staging_bytes(batch, q_seq, q_hidden, element_size)?,
                )?,
            })
        } else {
            None
        };
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
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        // Sq==1 makes BNSH and BSH layouts identical, so attention writes into
        // the real output tensor directly and the trailing transpose is skipped.
        let attention_out = if single_token {
            output_ptr
        } else {
            *out_bnsh.as_ref().ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: internal BNSH output allocation missing".into(),
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
        if fuse_metadata {
            onnx_runtime_ep_api::record_kernel_variant_stage!(
                "metadata",
                "gqa_prep_fused_with_metadata",
                "batch-1 fixed-capacity single-token decode derives past/total/query-start \
                 metadata inside fused prep with device-side bounds and sticky error latching"
            );
        } else {
            onnx_runtime_ep_api::record_kernel_variant_stage!(
                "metadata",
                "metadata_separate",
                "metadata remains a separate launch: batch={}, prep_fused={}; folding requires \
                 batch==1 and the eligible fixed-capacity single-token fused prep path",
                batch,
                fuse_prep
            );
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
        }
        if fuse_prep {
            let prep_variant = if fuse_metadata {
                "gqa_prep_fused_with_metadata"
            } else {
                "gqa_prep_fused"
            };
            onnx_runtime_ep_api::record_kernel_variant_stage!(
                "prep",
                prep_variant,
                "decode prep fused into one launch: Sq==1, k_seq==1, even head_dim={}, \
                 aliased device-KV, past_capacity==present_capacity={}, metadata_fused={}",
                dim,
                present_capacity,
                fuse_metadata
            );
            // Fused single-token decode prep. One launch subsumes the packed
            // split, BSH->BNSH query transpose, in-place K/V cache append, and
            // Q/present-K RoPE that the branch below performs separately. Batch
            // 1 also derives metadata per CTA, with block 0 writing the arrays;
            // larger batches consume the stream-ordered separate metadata.
            let (q_src, k_src, v_src, packed_flag) = if packed_qkv {
                (input_q_ptr, input_q_ptr, input_q_ptr, 1i32)
            } else {
                (
                    input_q_ptr,
                    cuptr(inputs[1].data_ptr::<u8>() as *const c_void),
                    cuptr(inputs[2].data_ptr::<u8>() as *const c_void),
                    0i32,
                )
            };
            let interleaved_i: i32 = self.rotary_interleaved.into();
            let do_rotary_i: i32 = self.do_rotary.into();
            let derive_metadata_i: i32 = fuse_metadata.into();
            // Unpacked non-RoPE Sq==1 attention reads Q directly. Passing a null
            // destination makes the fused prep's Q region a no-op while its K/V
            // append and metadata work remain fused.
            let fused_q_dst = if transpose_scratch.query { q_bnsh } else { 0 };
            let fused_count = batch * (self.num_heads + 2 * self.kv_num_heads) * (dim / 2);
            launch_1d!(
                self.runtime,
                prep_module,
                prep_src,
                fuse_entry,
                fused_count,
                builder,
                {
                    builder
                        .arg(&q_src)
                        .arg(&k_src)
                        .arg(&v_src)
                        .arg(&packed_flag)
                        .arg(&fused_q_dst)
                        .arg(&present_k_ptr)
                        .arg(&present_v_ptr)
                        .arg(&seqlens_ptr)
                        .arg(&totals_gpu)
                        .arg(&past_lengths_gpu)
                        .arg(&query_starts_gpu)
                        .arg(&past_capacity_i)
                        .arg(&metadata_error_gpu)
                        .arg(&derive_metadata_i)
                        .arg(&cos_ptr)
                        .arg(&sin_ptr)
                        .arg(&positions_ptr)
                        .arg(&batch_i)
                        .arg(&heads_i)
                        .arg(&kv_heads_i)
                        .arg(&dim_i)
                        .arg(&rotary_dim)
                        .arg(&present_capacity_i)
                        .arg(&cache_rows)
                        .arg(&do_rotary_i)
                        .arg(&interleaved_i)
                        .arg(&rope_cache_is_half);
                }
            );
        } else {
            onnx_runtime_ep_api::record_kernel_variant_stage!(
                "prep",
                "gqa_prep_unfused",
                "decode prep unfused (split/transpose/append/rope as separate launches): \
                 single_token={}, k_seq={}, even_head_dim={}, has_past_key={}, \
                 aliased_device_kv={}, past==present_capacity={}, prep_fusion_disabled={}",
                single_token,
                k_seq,
                dim.is_multiple_of(2),
                has_past_key,
                aliased_device_kv,
                past_capacity == present_capacity,
                self.prep_fusion_disabled
            );
            let (q_ptr, k_ptr, v_ptr) = if packed_qkv {
                let q_scratch = packed_q.as_ref().ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: internal packed-query allocation missing"
                            .into(),
                    )
                })?;
                let k_scratch = packed_k.as_ref().ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: internal packed-key allocation missing"
                            .into(),
                    )
                })?;
                let v_scratch = packed_v.as_ref().ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: internal packed-value allocation missing"
                            .into(),
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
            if q_ptr != q_bnsh {
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
            }

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
                for (tensor, positions, seq_i, heads, capacity, current_offset) in [
                    (q_bnsh, query_starts_gpu, q_seq_i, heads_i, q_seq_i, 0i32),
                    (
                        present_k_ptr,
                        past_lengths_gpu,
                        k_seq_i,
                        kv_heads_i,
                        present_capacity_i,
                        1i32,
                    ),
                ] {
                    let count =
                        batch * (heads as usize) * (seq_i as usize) * (rotary_dim_usize / 2);
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
                                .arg(&positions)
                                .arg(&batch_i)
                                .arg(&seq_i)
                                .arg(&heads)
                                .arg(&dim_i)
                                .arg(&rotary_dim)
                                .arg(&capacity)
                                .arg(&current_offset)
                                .arg(&cache_rows)
                                .arg(&interleaved_i)
                                .arg(&rope_cache_is_half);
                        }
                    );
                }
            }
        }

        let scale = self
            .scale
            .filter(|&scale| scale != 0.0)
            .unwrap_or_else(|| 1.0 / (dim as f32).sqrt());
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let score_elements = (batch as u64)
                .saturating_mul(self.num_heads as u64)
                .saturating_mul(q_seq as u64)
                .saturating_mul(attention_sequence_length as u64);
            let qk_flops = score_elements.saturating_mul(dim as u64).saturating_mul(2);
            let pv_flops = score_elements.saturating_mul(dim as u64).saturating_mul(2);
            let softmax_flops = score_elements.saturating_mul(4).saturating_add(
                (batch as u64)
                    .saturating_mul(self.num_heads as u64)
                    .saturating_mul(q_seq as u64),
            );
            qk_flops
                .saturating_add(pv_flops)
                .saturating_add(softmax_flops)
        });
        let read_path = if use_fused {
            KvCachePath::FlashPrefillRead
        } else if q.dtype == DataType::Float32 && gqa_decode::supported(q_seq, dim) {
            KvCachePath::F32DecodeRead
        } else if q.dtype == DataType::Float16 && gqa_decode_fp16::supported(q_seq, dim) {
            KvCachePath::Fp16DecodeRead
        } else if q.dtype == DataType::BFloat16 && gqa_decode_bf16::supported(q_seq, dim) {
            KvCachePath::Bf16DecodeRead
        } else if q.dtype == DataType::Float32 {
            KvCachePath::ReferenceRead
        } else {
            KvCachePath::Phase2aRead
        };
        self.kv_strides.require_converted_path_support(read_path)?;
        if use_fused {
            onnx_runtime_ep_api::record_kernel_variant!(
                "attention_flash_fused",
                "fused flash attention: backend={:?}, dtype={:?}, q_seq={}, \
                 valid_seq_len={}, head_dim={} passed the fused-support + measured-win gates",
                self.backend,
                q.dtype,
                q_seq,
                attention_sequence_length,
                dim
            );
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
                attention_out,
                0,
                0,
                totals_gpu,
                query_starts_gpu,
                local_window_i,
                self.softcap,
                &self.kv_strides,
            )?;
        } else if q.dtype == DataType::Float32 && gqa_decode::supported(q_seq, dim) {
            onnx_runtime_ep_api::record_kernel_variant!(
                "attention_gqa_decode_f32_splitk",
                "capture-safe f32 split-K single-token decode: q_seq={}, head_dim={}; \
                 active split count (1/2/4/8/16, max {}) is chosen on-device; \
                 flash backend={:?} not selected",
                q_seq,
                dim,
                gqa_decode::MAX_SPLITS,
                selected_backend
            );
            // Capture-safe split-K single-token GQA decode. Reads the valid
            // length on-device and uses fixed module-global scratch, so both
            // partial and merge launches record/replay inside a CUDA graph.
            gqa_decode::run(
                &self.runtime,
                batch,
                self.num_heads,
                self.kv_num_heads,
                q_seq,
                dim,
                present_capacity,
                self.num_heads / self.kv_num_heads,
                scale,
                q_bnsh,
                present_k_ptr,
                present_v_ptr,
                attention_out,
                totals_gpu,
                local_window_i,
                self.softcap,
                head_sink_ptr,
            )?;
        } else if q.dtype == DataType::Float16 && gqa_decode_fp16::supported(q_seq, dim) {
            onnx_runtime_ep_api::record_kernel_variant!(
                "attention_gqa_decode_fp16_splitk",
                "capture-safe fp16 split-K flash-decode: q_seq={}, even head_dim={} (<=512); \
                 active split count (up to {}) chosen on-device from the valid length \
                 and a host occupancy target that fills the multiprocessors",
                q_seq,
                dim,
                gqa_decode_fp16::MAX_SPLITS
            );
            // Capture-safe fp16 flash-decode sibling of the f32 `gqa_decode`
            // branch above. Same launcher signature/units; passes the fp16
            // device pointers for query/present-K/present-V/output. Reads the
            // valid length on-device from `totals_gpu` and allocates only
            // fixed-size dynamic shared memory, so it records/replays inside a
            // CUDA graph. Unsupported fp16 shapes (e.g. prefill Sq>1) still fall
            // through to the phase-2a path below.
            gqa_decode_fp16::run(
                &self.runtime,
                batch,
                self.num_heads,
                self.kv_num_heads,
                q_seq,
                dim,
                present_capacity,
                self.num_heads / self.kv_num_heads,
                scale,
                q_bnsh,
                present_k_ptr,
                present_v_ptr,
                attention_out,
                totals_gpu,
                local_window_i,
                self.softcap,
                &self.kv_strides,
            )?;
        } else if q.dtype == DataType::BFloat16 && gqa_decode_bf16::supported(q_seq, dim) {
            onnx_runtime_ep_api::record_kernel_variant!(
                "attention_gqa_decode_bf16_splitk",
                "capture-safe bf16 split-K flash-decode: q_seq={}, even head_dim={} (<=512); \
                 active split count (up to {}) chosen on-device from the valid length \
                 and a host occupancy target that fills the multiprocessors",
                q_seq,
                dim,
                gqa_decode_bf16::MAX_SPLITS
            );
            // Capture-safe bf16 flash-decode sibling of the fp16 branch above.
            // Same launcher signature/units and capture-safety contract; passes
            // the bf16 device pointers for query/present-K/present-V/output.
            // Reads the valid length on-device from `totals_gpu` and allocates
            // only fixed-size dynamic shared memory, so it records/replays inside
            // a CUDA graph. This is the branch that makes bfloat16 decoders (e.g.
            // Muse-Glimmer) capture-eligible: without it every bf16 GQA node
            // declined at the kernel gate and forced an eager device seam that
            // defeated whole-graph capture. Unsupported bf16 shapes (e.g. prefill
            // Sq>1) still fall through to the phase-2a path below.
            gqa_decode_bf16::run(
                &self.runtime,
                batch,
                self.num_heads,
                self.kv_num_heads,
                q_seq,
                dim,
                present_capacity,
                self.num_heads / self.kv_num_heads,
                scale,
                q_bnsh,
                present_k_ptr,
                present_v_ptr,
                attention_out,
                totals_gpu,
                local_window_i,
                self.softcap,
                &self.kv_strides,
            )?;
        } else if q.dtype == DataType::Float32 {
            onnx_runtime_ep_api::record_kernel_variant!(
                "attention_reference_f32",
                "f32 reference attention: flash backend={:?} for q_seq={}, valid_seq_len={}, \
                 head_dim={}; capture-safe gqa_decode does not support this shape",
                selected_backend,
                q_seq,
                attention_sequence_length,
                dim
            );
            let attention_rows = batch
                .checked_mul(self.num_heads)
                .and_then(|rows| rows.checked_mul(q_seq))
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: attention row count overflow".into(),
                    )
                })?;
            // The score buffer is `[B, H, Sq, present_capacity]` f32; planning
            // and execution size it through the same helper so the reserved and
            // consumed byte counts cannot drift. It occupies the score region of
            // the governed composite workspace, after the packed QKV staging.
            let scores_bytes =
                gqa_reference_scores_bytes(batch, self.num_heads, q_seq, present_capacity)?;
            let score_scratch = match prepared {
                // Governed slice (§736): the executor reserved this
                // session-persistent composite buffer against the device
                // authority during prepare-only planning, sized through the same
                // `gqa_workspace_layout`/`gqa_reference_scores_bytes` helpers.
                // Carve the score region deterministically, refusing on a
                // shortfall rather than silently under-allocating or
                // reintroducing a raw pooled allocation.
                Some(view) => gqa_carve(
                    view,
                    composite_layout.scores_offset,
                    scores_bytes,
                    "reference score matrix",
                )?,
                // Compatibility/opt-out path: no executor-prepared workspace, so
                // the score scratch stays self-owned in the pooled slot.
                None => workspace.reserve(WS_SCORES, scores_bytes)?,
            };
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
                .arg(&attention_out)
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
                .arg(&self.softcap)
                .arg(&head_sink_ptr);
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
            onnx_runtime_ep_api::record_kernel_variant!(
                "attention_phase2a",
                "phase2a general attention: dtype={:?}, q_seq={}, valid_seq_len={}, head_dim={} \
                 not selected for fused flash or a capture-safe dtype-specific decode kernel",
                q.dtype,
                q_seq,
                attention_sequence_length,
                dim
            );
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
                attention_out,
                0,
                0,
                totals_gpu,
                query_starts_gpu,
                local_window_i,
                self.softcap,
                None,
            )?;
        }

        // For Sq==1 attention already wrote the BSH output in place, so the
        // BNSH->BSH transpose is skipped. Sq>1 still materialises via `out_bnsh`.
        if !single_token {
            let out_bnsh_ptr = out_bnsh.ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: internal BNSH output allocation missing".into(),
                )
            })?;
            launch_1d!(
                self.runtime,
                prep_module,
                prep_src,
                transpose_out_entry,
                outputs[0].numel(),
                builder,
                {
                    builder
                        .arg(&out_bnsh_ptr)
                        .arg(&output_ptr)
                        .arg(&batch_i)
                        .arg(&q_seq_i)
                        .arg(&heads_i)
                        .arg(&dim_i);
                }
            );
        }
        *last_signature = capture_candidate;
        Ok(())
    }

    /// Prepare-only planning (#747, §736): report the governed session-persistent
    /// composite workspace — packed Q/K/V projection staging, BSH↔BNSH
    /// transpose scratch, and the f32 reference score matrix — so the executor
    /// reserves it against the device authority before request admission. Each
    /// region is charged only on routes that materialize it, so no device
    /// capacity is charged for scratch a dispatch never touches:
    ///
    /// - **Packed QKV staging** is reserved only when the query arrives packed
    ///   (both key/value inputs absent). When Q/K/V arrive already unpacked the
    ///   split scratch is never used, so it charges **zero**. The fp16/bf16 fused
    ///   single-token decode also skips the split, but planning cannot observe
    ///   its device-KV aliasing, so it conservatively reserves the (tiny,
    ///   `q_seq==1`) staging; every packed *prefill* genuinely splits.
    /// - **Reference scores** are reserved only on the f32 reference path
    ///   (`gqa_reference_scores_path`); fused flash, f32/fp16 split-K decode and
    ///   phase-2a materialize no device score matrix and charge zero.
    /// - **BNSH transpose scratch** reserves Q for multi-token transpose, packed
    ///   extraction, or writable RoPE, and reserves output only for `q_seq > 1`.
    ///   Unpacked non-RoPE `q_seq == 1` reads Q and writes output directly, so
    ///   both transpose regions charge zero. Seq-major BSNH changes KV strides,
    ///   not the Q/output orientation, and therefore does not change this split.
    ///
    /// On a shape/dtype this kernel would reject, report NONE and let `run` raise
    /// the precise error via the compatibility scratch.
    ///
    /// The reservation is session-persistent: these regions live in the pooled
    /// `Mutex<GqaWorkspace>` slots, grown to the largest geometry seen and
    /// retained across decode/prefill steps, so — unlike the step-scoped
    /// Attention Phase-2a scratch (#753) — the composite holds a standing claim
    /// for the session and is charged as `WorkspaceLifetime::SessionPersistent`.
    /// Because the staging (`q_seq`) and scores (`Sq · kv`) geometries are
    /// prompt-dependent, the executor grows it transactionally through a
    /// `MappedGrowthGrant` against the device authority.
    fn composite_workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
    ) -> Result<WorkspaceRequirement> {
        let Some(q) = inputs.first() else {
            return Ok(WorkspaceRequirement::NONE);
        };
        if q.shape.len() != 3 {
            return Ok(WorkspaceRequirement::NONE);
        }
        let (batch, q_seq, input_hidden) = (q.shape[0], q.shape[1], q.shape[2]);
        // Packed QKV is signalled by absent key/value inputs, mirroring `run`.
        let packed_qkv =
            inputs.get(1).is_none_or(|k| !k.present) && inputs.get(2).is_none_or(|v| !v.present);
        let head_dim = if packed_qkv {
            let packed_heads = self.num_heads + 2 * self.kv_num_heads;
            if packed_heads == 0 || input_hidden == 0 || input_hidden % packed_heads != 0 {
                return Ok(WorkspaceRequirement::NONE);
            }
            input_hidden / packed_heads
        } else {
            if self.num_heads == 0 || input_hidden == 0 || input_hidden % self.num_heads != 0 {
                return Ok(WorkspaceRequirement::NONE);
            }
            input_hidden / self.num_heads
        };
        if head_dim == 0 {
            return Ok(WorkspaceRequirement::NONE);
        }
        // Staging is charged only for packed QKV inputs, and only for dtypes this
        // kernel actually splits/stages (f32/f16/bf16). Its Q/K/V byte counts are
        // shape-derived and identical in `run`, so the layout offsets match.
        let element_size = q.dtype.byte_size();
        let want_staging = packed_qkv
            && element_size != 0
            && matches!(
                q.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            );
        let want_scores = gqa_reference_scores_path(q.dtype, q_seq, head_dim);
        let transpose_scratch = gqa_transpose_scratch(packed_qkv, self.do_rotary, q_seq);
        // KV-capacity proxy from static input metadata. The reference score
        // buffer is strided by the present-cache capacity; for GQA's
        // buffer-shared KV cache that equals the past_key capacity
        // (`inputs[3].shape[2]`), and on a pure prefill with no past it equals
        // the incoming key length. Both are static input dims. Execution
        // re-derives the exact present capacity and refuses deterministically on
        // any shortfall, so this never silently under-allocates.
        let scores_kv_capacity = if want_scores {
            let past_capacity = inputs
                .get(3)
                .filter(|past| past.present && past.shape.len() == 4)
                .map(|past| past.shape[2])
                .unwrap_or(0);
            let key_length = if packed_qkv {
                q_seq
            } else {
                inputs
                    .get(1)
                    .filter(|key| key.shape.len() == 3)
                    .map(|key| key.shape[1])
                    .unwrap_or(0)
            };
            past_capacity.max(key_length)
        } else {
            0
        };
        let layout = gqa_workspace_layout(
            batch,
            self.num_heads,
            q_seq,
            self.num_heads * head_dim,
            self.kv_num_heads * head_dim,
            element_size,
            want_staging,
            transpose_scratch.query,
            transpose_scratch.output,
            scores_kv_capacity,
            want_scores,
        )?;
        if layout.total_bytes == 0 {
            return Ok(WorkspaceRequirement::NONE);
        }
        Ok(WorkspaceRequirement {
            bytes: u64::try_from(layout.total_bytes).map_err(|_| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: composite workspace does not fit u64".into(),
                )
            })?,
            alignment: GQA_SCORES_ALIGN,
            lifetime: WorkspaceLifetime::SessionPersistent,
            role: MemoryRole::Workspace { step_scoped: false },
        })
    }
}

impl Kernel for GroupQueryAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // Compatibility/opt-out path: no executor-prepared workspace, so the f32
        // reference score scratch stays self-owned in the pooled slot.
        self.run(inputs, outputs, None)
    }

    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        self.composite_workspace_requirement(inputs)
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
        true
    }

    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        self.workspace
            .lock()
            .map(|workspace| workspace.device_graph_resources())
            .unwrap_or_default()
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        // Eligibility is tied to the exact one-token, fixed-capacity, in-place
        // device-KV decode signature warmed by the most recent successful call.
        match self.last_capture_safe_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "requires a warmed fixed-capacity aliased device-KV signature: either f32/fp16/bf16 q_seq==1 (Phase2a split-K decode) or fp16/bf16 q_seq>1 (fused flash verify/prefill); the current signature was not warmed as capture-safe",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "GroupQueryAttention capture signature is unavailable because its state lock was poisoned",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId};

    fn gqa_node(attrs: &[(&str, Attribute)]) -> Node {
        let mut node = Node::new(NodeId(0), "GroupQueryAttention", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        node
    }

    #[test]
    fn gqa_unsupported_reason_accepts_plain_grouped_attention() {
        // Qwen2 0.5B head layout: 14 query heads, 2 KV heads, no exotic attrs.
        let node = gqa_node(&[
            ("num_heads", Attribute::Int(14)),
            ("kv_num_heads", Attribute::Int(2)),
        ]);
        assert!(unsupported_reason(&node).is_none());
    }

    #[test]
    fn gqa_unsupported_reason_declines_smooth_softmax() {
        let node = gqa_node(&[
            ("num_heads", Attribute::Int(14)),
            ("kv_num_heads", Attribute::Int(2)),
            ("smooth_softmax", Attribute::Int(1)),
        ]);
        let reason =
            unsupported_reason(&node).expect("smooth_softmax must be declined at claim time");
        assert!(reason.contains("smooth_softmax"), "reason: {reason}");
    }

    #[test]
    fn gqa_unsupported_reason_declines_missing_and_misconfigured_heads() {
        // Missing required num_heads.
        let missing = gqa_node(&[("kv_num_heads", Attribute::Int(2))]);
        assert!(
            unsupported_reason(&missing).is_some_and(|r| r.contains("num_heads")),
            "missing num_heads must decline"
        );
        // num_heads not a multiple of kv_num_heads.
        let ratio = gqa_node(&[
            ("num_heads", Attribute::Int(14)),
            ("kv_num_heads", Attribute::Int(3)),
        ]);
        assert!(
            unsupported_reason(&ratio).is_some_and(|r| r.contains("multiple")),
            "non-divisible head counts must decline"
        );
    }

    #[test]
    fn sequence_lengths_shape_accepts_canonical_per_batch_forms() {
        for batch in [1, 3] {
            validate_sequence_lengths_shape(&[batch], batch, batch).unwrap();
            validate_sequence_lengths_shape(&[batch, 1], batch, batch).unwrap();
        }
    }

    #[test]
    fn sequence_lengths_shape_rejects_noncanonical_singleton_layouts_actionably() {
        for shape in [vec![1, 3], vec![3, 1, 1]] {
            let error = validate_sequence_lengths_shape(&shape, 3, 3)
                .expect_err("noncanonical seqlens_k shape must fail");
            let message = format!("{error}");
            assert!(message.contains("[batch_size], [batch_size, 1]"));
            assert!(message.contains(&format!("got shape {shape:?}")));
        }
    }

    #[test]
    fn sequence_lengths_shape_promotes_unit_batch_scalar_and_rejects_larger_batches() {
        validate_sequence_lengths_shape(&[], 1, 1).unwrap();
        let error = validate_sequence_lengths_shape(&[], 1, 2)
            .expect_err("a scalar cannot represent a multi-row batch");
        assert!(format!("{error}").contains("batch_size 2"));
    }

    fn runtime() -> Option<Arc<CudaRuntime>> {
        crate::test_support::maybe_runtime()
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

        // Growing releases the old block and takes a larger one. `frees` counts
        // *driver* calls, and the runtime's allocation pool retains released
        // blocks rather than returning them to the driver, so the released
        // 4096-byte block shows up as retained bytes instead of as a `cuMemFree`.
        // Asserting on the retained total keeps the property this test exists
        // for -- the old block is released, not leaked -- without pinning which
        // side of the pool it ends up on.
        let grown = workspace.reserve(WS_Q_BNSH, 8192).unwrap();
        assert_ne!(grown, first);
        let grown_counts = runtime.allocation_counts();
        assert_eq!(grown_counts.allocations, before.allocations + 2);
        assert!(
            runtime.raw_pool_retained_bytes() == 4096 || grown_counts.frees == before.frees + 1,
            "the 4096-byte block must be released -- retained by the pool or \
             returned to the driver, but not held by the workspace"
        );

        drop(workspace);
        let after = runtime.allocation_counts();
        assert_eq!(after.allocations, before.allocations + 2);
        assert!(
            runtime.raw_pool_retained_bytes() == 4096 + 8192 || after.frees == before.frees + 2,
            "dropping the workspace must release the 8192-byte block too"
        );
    }

    #[test]
    fn reference_scores_bytes_matches_score_count_formula() {
        // Planning and execution size the governed score buffer through this
        // exact helper (§736), so pin its formula: `batch·heads·q_seq·kv` f32
        // scores. A degenerate geometry still reserves one element, matching the
        // kernel's historical `score_count.max(1) * 4`.
        let (batch, heads, q_seq, kv) = (2usize, 4usize, 8usize, 130usize);
        let bytes = gqa_reference_scores_bytes(batch, heads, q_seq, kv).unwrap();
        assert_eq!(
            bytes,
            batch * heads * q_seq * kv * std::mem::size_of::<f32>()
        );
        assert_eq!(
            gqa_reference_scores_bytes(0, heads, q_seq, kv).unwrap(),
            std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn reference_scores_path_gates_on_dtype_and_decode_support() {
        // f32 prefill (Sq>1) has no split-K decode kernel: reference path runs
        // and materializes scores.
        assert!(gqa_reference_scores_path(DataType::Float32, 8, 64));
        // f32 single-token decode with head_dim within the gqa_decode ceiling
        // (MAX_HEAD_DIM = 512) is covered by gqa_decode, which streams softmax
        // and needs no score scratch. The ceiling was widened 128->256->512 as a
        // feature (#1438, gemma4-e2b / Gemma-3n head_dim=512); this assertion is
        // pinned to that ceiling so it flags deliberately if it moves again.
        assert!(!gqa_reference_scores_path(DataType::Float32, 1, 64));
        assert!(!gqa_reference_scores_path(DataType::Float32, 1, 192));
        assert!(!gqa_reference_scores_path(DataType::Float32, 1, 512));
        // f32 single-token with head_dim > MAX_HEAD_DIM exceeds gqa_decode: the
        // reference path runs and materializes scores.
        assert!(gqa_reference_scores_path(DataType::Float32, 1, 513));
        assert!(gqa_reference_scores_path(DataType::Float32, 1, 576));
        // fp16/bf16 never take the f32 reference path, so no score buffer is
        // charged on the common decode/prefill dtypes.
        assert!(!gqa_reference_scores_path(DataType::Float16, 8, 64));
        assert!(!gqa_reference_scores_path(DataType::BFloat16, 8, 64));
    }

    #[test]
    fn workspace_requirement_governs_reference_scores_and_transposes_by_route() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping CUDA GQA workspace requirement test: CUDA runtime unavailable");
            return;
        };
        let (num_heads, kv_heads, dim) = (4usize, 2usize, 64usize);
        let kernel = GroupQueryAttentionKernel::new(
            runtime, num_heads, kv_heads, None, false, false, -1, 0.0,
        )
        .expect("kernel");
        let (batch, q_seq, cache) = (1usize, 8usize, 256usize);
        let q_shape = [batch, q_seq, num_heads * dim];
        let kv_shape = [batch, q_seq, kv_heads * dim];
        let past_shape = [batch, kv_heads, cache, dim];
        let seqlens_shape = [batch];
        let total_shape = [1usize];
        fn meta(dtype: DataType, shape: &[usize]) -> TensorMetadata<'_> {
            TensorMetadata::new(dtype, shape, true)
        }
        let f32_inputs = [
            meta(DataType::Float32, &q_shape),
            meta(DataType::Float32, &kv_shape),
            meta(DataType::Float32, &kv_shape),
            meta(DataType::Float32, &past_shape),
            meta(DataType::Float32, &past_shape),
            meta(DataType::Int32, &seqlens_shape),
            meta(DataType::Int32, &total_shape),
        ];
        // f32 prefill routes to the reference path: governed, session-persistent,
        // and sized through the shared helper against the cache capacity.
        let req = kernel
            .workspace_requirement(&f32_inputs)
            .expect("requirement");
        let tensor_bytes = batch * q_seq * num_heads * dim * 4;
        let aligned_tensor = gqa_align_up(tensor_bytes, GQA_SCORES_ALIGN).unwrap();
        let expected = (2 * aligned_tensor
            + gqa_reference_scores_bytes(batch, num_heads, q_seq, cache).unwrap())
            as u64;
        assert_eq!(req.bytes, expected);
        assert_eq!(req.alignment, GQA_SCORES_ALIGN);
        assert_eq!(req.lifetime, WorkspaceLifetime::SessionPersistent);
        assert!(matches!(
            req.role,
            MemoryRole::Workspace { step_scoped: false }
        ));

        // The same geometry in fp16 never materializes scores, but multi-token
        // Q/output transpose scratch is still genuinely needed.
        let f16_inputs = [
            meta(DataType::Float16, &q_shape),
            meta(DataType::Float16, &kv_shape),
            meta(DataType::Float16, &kv_shape),
            meta(DataType::Float16, &past_shape),
            meta(DataType::Float16, &past_shape),
            meta(DataType::Int32, &seqlens_shape),
            meta(DataType::Int32, &total_shape),
        ];
        let f16_req = kernel
            .workspace_requirement(&f16_inputs)
            .expect("fp16 requirement");
        let f16_tensor = batch * q_seq * num_heads * dim * 2;
        assert_eq!(
            f16_req.bytes,
            (2 * gqa_align_up(f16_tensor, GQA_SCORES_ALIGN).unwrap()) as u64,
            "fp16 prefill charges only its Q/output transpose scratch"
        );

        // f32 single-token decode (head_dim<=128) is covered by the capture-safe
        // split-K kernel and reserves no scores either.
        let decode_q = [batch, 1usize, num_heads * dim];
        let decode_kv = [batch, 1usize, kv_heads * dim];
        let decode_inputs = [
            meta(DataType::Float32, &decode_q),
            meta(DataType::Float32, &decode_kv),
            meta(DataType::Float32, &decode_kv),
            meta(DataType::Float32, &past_shape),
            meta(DataType::Float32, &past_shape),
            meta(DataType::Int32, &seqlens_shape),
            meta(DataType::Int32, &total_shape),
        ];
        let decode_req = kernel
            .workspace_requirement(&decode_inputs)
            .expect("decode requirement");
        assert_eq!(
            decode_req.bytes, 0,
            "unpacked non-RoPE single-token decode uses direct Q/output and gqa_decode scores"
        );
    }

    #[test]
    fn packed_staging_bytes_matches_split_scratch_formula() {
        // Planning and execution size each packed QKV staging buffer through this
        // helper (§736): `batch·seq·hidden·element_size`, matching the pooled
        // `workspace.reserve(WS_PACKED_*, batch * seq * hidden * element_size)`.
        let (batch, seq, hidden, elem) = (2usize, 7usize, 512usize, 2usize);
        assert_eq!(
            gqa_packed_staging_bytes(batch, seq, hidden, elem).unwrap(),
            batch * seq * hidden * elem
        );
    }

    #[test]
    fn workspace_layout_places_staging_before_unpadded_scores() {
        let (batch, num_heads, kv_heads, q_seq, dim, elem) =
            (1usize, 8usize, 2usize, 16usize, 64usize, 4usize);
        let q_hidden = num_heads * dim;
        let k_hidden = kv_heads * dim;
        let kv_capacity = 128usize;

        // Both regions live (f32 packed reference prefill): staging first, the
        // score matrix last and unpadded, all sub-buffers 256-aligned.
        let both = gqa_workspace_layout(
            batch,
            num_heads,
            q_seq,
            q_hidden,
            k_hidden,
            elem,
            true,
            false,
            false,
            kv_capacity,
            true,
        )
        .unwrap();
        let q_bytes = batch * q_seq * q_hidden * elem;
        let kv_bytes = batch * q_seq * k_hidden * elem;
        let align = |b: usize| b.div_ceil(GQA_SCORES_ALIGN) * GQA_SCORES_ALIGN;
        assert_eq!(both.packed_q_offset, 0);
        assert_eq!(both.packed_q_bytes, q_bytes);
        assert_eq!(both.packed_k_offset, align(q_bytes));
        assert_eq!(both.packed_v_offset, align(q_bytes) + align(kv_bytes));
        let staging_total = align(q_bytes) + 2 * align(kv_bytes);
        assert_eq!(both.scores_offset, staging_total);
        assert_eq!(both.scores_offset % GQA_SCORES_ALIGN, 0);
        let scores_bytes =
            gqa_reference_scores_bytes(batch, num_heads, q_seq, kv_capacity).unwrap();
        assert_eq!(both.scores_bytes, scores_bytes);
        assert_eq!(both.total_bytes, staging_total + scores_bytes);

        // Staging only (fp16 packed prefill: no reference score matrix).
        let staging = gqa_workspace_layout(
            batch, num_heads, q_seq, q_hidden, k_hidden, elem, true, false, false, 0, false,
        )
        .unwrap();
        assert_eq!(staging.scores_bytes, 0);
        assert_eq!(staging.total_bytes, staging_total);

        // Scores only (unpacked f32 reference prefill): offset 0, total is the
        // exact unpadded score bytes, so the composite matches the pre-staging
        // reservation for unpacked inputs.
        let scores = gqa_workspace_layout(
            batch,
            num_heads,
            q_seq,
            q_hidden,
            k_hidden,
            elem,
            false,
            false,
            false,
            kv_capacity,
            true,
        )
        .unwrap();
        assert_eq!(scores.packed_q_bytes, 0);
        assert_eq!(scores.scores_offset, 0);
        assert_eq!(scores.total_bytes, scores_bytes);

        // No region live (fp16/bf16 decode, or unpacked non-reference).
        let none = gqa_workspace_layout(
            batch, num_heads, q_seq, q_hidden, k_hidden, elem, false, false, false, 0, false,
        )
        .unwrap();
        assert_eq!(none.total_bytes, 0);
    }

    #[test]
    fn transpose_scratch_routes_and_layout_match_use() {
        assert_eq!(
            gqa_transpose_scratch(false, false, 1),
            GqaTransposeScratch {
                query: false,
                output: false
            },
            "unpacked non-RoPE decode uses Q/output directly"
        );
        assert_eq!(
            gqa_transpose_scratch(false, true, 1),
            GqaTransposeScratch {
                query: true,
                output: false
            },
            "RoPE decode needs a writable Q copy but no output transpose"
        );
        assert_eq!(
            gqa_transpose_scratch(true, false, 1),
            GqaTransposeScratch {
                query: true,
                output: false
            },
            "packed decode needs Q extraction but no output transpose"
        );
        assert_eq!(
            gqa_transpose_scratch(false, false, 8),
            GqaTransposeScratch {
                query: true,
                output: true
            },
            "multi-token prefill needs both transposes"
        );

        let (batch, heads, kv_heads, dim, elem) = (1usize, 8usize, 2usize, 64usize, 2usize);
        let q_hidden = heads * dim;
        let k_hidden = kv_heads * dim;
        let packed_decode = gqa_workspace_layout(
            batch, heads, 1, q_hidden, k_hidden, elem, true, true, false, 0, false,
        )
        .unwrap();
        assert_eq!(packed_decode.q_bnsh_offset, packed_decode.packed_q_offset);
        assert_eq!(packed_decode.q_bnsh_bytes, packed_decode.packed_q_bytes);
        let packed_total = gqa_align_up(packed_decode.packed_q_bytes, GQA_SCORES_ALIGN).unwrap()
            + gqa_align_up(packed_decode.packed_k_bytes, GQA_SCORES_ALIGN).unwrap()
            + gqa_align_up(packed_decode.packed_v_bytes, GQA_SCORES_ALIGN).unwrap();
        assert_eq!(
            packed_decode.total_bytes, packed_total,
            "packed Sq==1 Q extraction and BNSH scratch must share one peak"
        );

        let prefill = gqa_workspace_layout(
            batch, heads, 8, q_hidden, k_hidden, elem, false, true, true, 0, false,
        )
        .unwrap();
        let tensor_bytes = batch * 8 * q_hidden * elem;
        let aligned = gqa_align_up(tensor_bytes, GQA_SCORES_ALIGN).unwrap();
        assert_eq!(prefill.q_bnsh_offset, 0);
        assert_eq!(prefill.out_bnsh_offset, aligned);
        assert_eq!(prefill.total_bytes, 2 * aligned);
    }

    #[test]
    fn transpose_workspace_shortfall_is_deterministic() {
        let view = WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(0x1000usize as *mut c_void),
            1024,
        );
        let error = gqa_carve(view, 768, 512, "query BNSH transpose scratch")
            .expect_err("short prepared workspace must fail");
        assert!(
            error
                .to_string()
                .contains("prepared workspace 1024 bytes is smaller than the 1280 bytes"),
            "unexpected shortfall error: {error}"
        );
    }

    #[test]
    fn workspace_requirement_charges_packed_staging_only_when_packed() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping CUDA GQA staging requirement test: CUDA runtime unavailable");
            return;
        };
        let (num_heads, kv_heads, dim) = (8usize, 2usize, 64usize);
        let kernel = GroupQueryAttentionKernel::new(
            runtime, num_heads, kv_heads, None, false, false, -1, 0.0,
        )
        .expect("kernel");
        let (batch, q_seq, cache) = (1usize, 32usize, 256usize);
        let packed_heads = num_heads + 2 * kv_heads;
        let packed_q_shape = [batch, q_seq, packed_heads * dim];
        let past_shape = [batch, kv_heads, cache, dim];
        let seqlens_shape = [batch];
        let total_shape = [1usize];
        fn meta(dtype: DataType, shape: &[usize], present: bool) -> TensorMetadata<'_> {
            TensorMetadata::new(dtype, shape, present)
        }
        let absent = [0usize; 0];

        // fp16 packed prefill: no reference score matrix, but the split staging
        // is charged (Q + K + V, 256-aligned). This is the new governed slice.
        let f16_packed = [
            meta(DataType::Float16, &packed_q_shape, true),
            meta(DataType::Float16, &absent, false),
            meta(DataType::Float16, &absent, false),
            meta(DataType::Float16, &past_shape, true),
            meta(DataType::Float16, &past_shape, true),
            meta(DataType::Int32, &seqlens_shape, true),
            meta(DataType::Int32, &total_shape, true),
        ];
        let req = kernel
            .workspace_requirement(&f16_packed)
            .expect("requirement");
        let q_bytes = batch * q_seq * (num_heads * dim) * 2;
        let kv_bytes = batch * q_seq * (kv_heads * dim) * 2;
        let align = |b: usize| b.div_ceil(GQA_SCORES_ALIGN) * GQA_SCORES_ALIGN;
        let transpose_total = 2 * align(q_bytes);
        let staging_total = align(q_bytes) + 2 * align(kv_bytes);
        assert_eq!(
            req.bytes,
            (staging_total + transpose_total) as u64,
            "fp16 packed prefill charges packed staging plus Q/output transpose scratch"
        );
        assert_eq!(req.lifetime, WorkspaceLifetime::SessionPersistent);

        // The same geometry with unpacked K/V present never splits, so staging
        // charges zero (the primary "size from use" finding for this slice).
        let kv_shape = [batch, q_seq, kv_heads * dim];
        let unpacked_q_shape = [batch, q_seq, num_heads * dim];
        let f16_unpacked = [
            meta(DataType::Float16, &unpacked_q_shape, true),
            meta(DataType::Float16, &kv_shape, true),
            meta(DataType::Float16, &kv_shape, true),
            meta(DataType::Float16, &past_shape, true),
            meta(DataType::Float16, &past_shape, true),
            meta(DataType::Int32, &seqlens_shape, true),
            meta(DataType::Int32, &total_shape, true),
        ];
        let unpacked_req = kernel
            .workspace_requirement(&f16_unpacked)
            .expect("unpacked requirement");
        assert_eq!(
            unpacked_req.bytes, transpose_total as u64,
            "unpacked fp16 prefill skips split staging but still needs both transposes"
        );

        // f32 packed prefill charges both the staging and the reference scores.
        let f32_packed = [
            meta(DataType::Float32, &packed_q_shape, true),
            meta(DataType::Float32, &absent, false),
            meta(DataType::Float32, &absent, false),
            meta(DataType::Float32, &past_shape, true),
            meta(DataType::Float32, &past_shape, true),
            meta(DataType::Int32, &seqlens_shape, true),
            meta(DataType::Int32, &total_shape, true),
        ];
        let f32_req = kernel
            .workspace_requirement(&f32_packed)
            .expect("f32 packed requirement");
        let f32_staging = align(batch * q_seq * (num_heads * dim) * 4)
            + 2 * align(batch * q_seq * (kv_heads * dim) * 4);
        let scores = gqa_reference_scores_bytes(batch, num_heads, q_seq, cache).unwrap();
        assert_eq!(
            f32_req.bytes,
            (f32_staging + 2 * align(batch * q_seq * (num_heads * dim) * 4) + scores) as u64,
            "f32 packed prefill charges staging, both transposes, and reference scores"
        );
    }
}

/// Regression guard for issue #736: the governed GQA score buffer (`WS_SCORES`)
/// is routed through `Kernel::workspace_requirement` + an executor-prepared
/// session-persistent workspace, consumed via `execute_with_workspace`. This
/// CPU-only test fails if the f32 reference branch reintroduces an unconditional
/// raw pooled allocation of the score slot, bypassing the device authority. It
/// runs on CI without a GPU, matching the #751 bar (a source-scan unit test, not
/// a `Select-String`-style external scan).
#[cfg(test)]
mod raw_allocation_guard {
    #[test]
    fn gqa_scores_slot_is_governed_not_raw_allocated() {
        const SOURCE: &str = include_str!("group_query_attention.rs");
        assert!(
            SOURCE.contains("fn execute_with_workspace"),
            "GroupQueryAttention must stay wired into governed workspace preparation (#736)."
        );
        assert!(
            SOURCE.contains("gqa_reference_scores_bytes"),
            "the f32 reference score buffer must be sized through the shared helper (#736)."
        );
        // The pre-#736 seam reserved the score slot from the self-owned pool
        // unconditionally. The needle is assembled at runtime so this literal
        // does not itself match when the file is scanned via `include_str!`.
        let ungoverned = ["workspace.reserve(WS_SCORES, score_count", ".max(1) * 4)"].concat();
        assert!(
            !SOURCE.contains(ungoverned.as_str()),
            "GroupQueryAttention must not reintroduce an unconditional raw pooled allocation of \
             the governed score slot (#736); carve it from the executor-prepared workspace."
        );
    }

    #[test]
    fn gqa_packed_qkv_staging_is_governed_not_raw_allocated() {
        const SOURCE: &str = include_str!("group_query_attention.rs");
        // The packed QKV projection staging is sized through the shared composite
        // layout helper and carved from the executor-prepared workspace on the
        // governed path (§736 QKV slice).
        assert!(
            SOURCE.contains("gqa_workspace_layout"),
            "packed QKV staging must be sized through the shared composite layout helper (#736)."
        );
        assert!(
            SOURCE.contains("packed query staging")
                && SOURCE.contains("packed key staging")
                && SOURCE.contains("packed value staging"),
            "packed QKV staging must be carved from the executor-prepared composite workspace \
             via `gqa_carve` on the governed path (#736)."
        );
        // The pre-#736 seam reserved the packed slots straight from the pooled
        // allocator with no executor-prepared branch. The needle is assembled at
        // runtime so this literal does not match when scanned via `include_str!`.
        let ungoverned = [
            "(packed_qkv && !fuse_prep)\n",
            "            .then(|| workspace.reserve(WS_PACKED_Q,",
        ]
        .concat();
        assert!(
            !SOURCE.contains(ungoverned.as_str()),
            "GroupQueryAttention must not reintroduce an unconditional raw pooled allocation of \
             the packed QKV staging slots (#736); carve them from the prepared workspace and keep \
             the pooled reserve only on the compatibility/opt-out `None` arm."
        );
    }

    #[test]
    fn gqa_bnsh_transpose_slots_are_governed_not_raw_allocated() {
        const SOURCE: &str = include_str!("group_query_attention.rs");
        assert!(
            SOURCE.contains("gqa_transpose_scratch")
                && SOURCE.contains("query BNSH transpose scratch")
                && SOURCE.contains("output BNSH transpose scratch"),
            "GQA transpose scratch must be route-sized and carved from the prepared composite."
        );
        let unconditional_query = [
            "let q_bnsh = workspace.reserve(WS_Q_BNSH,",
            " batch * q_seq * q_hidden * element_size)?;",
        ]
        .concat();
        let unconditional_output = [
            "then(|| workspace.reserve(WS_OUT_BNSH,",
            " outputs[0].numel() * element_size))",
        ]
        .concat();
        assert!(
            !SOURCE.contains(unconditional_query.as_str())
                && !SOURCE.contains(unconditional_output.as_str()),
            "GroupQueryAttention must not reintroduce unconditional raw pooled BNSH transpose \
             reservations; prepared execution must use the governed composite and Sq==1 direct \
             routes must remain allocation-free."
        );
    }
}

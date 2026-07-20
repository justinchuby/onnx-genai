//! NVRTC-compiled, tiled online-softmax attention for prefill.
//!
//! A block owns eight query rows from one `(batch, query_head)` plane. K/V are
//! loaded in 16-token tiles and reused by all eight rows. Scores live only in
//! shared memory; the running softmax maximum, denominator, and output numerator
//! are updated tile by tile, so no `[B,H,Sq,Sk]` allocation is required.

use std::sync::LazyLock;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_ir::DataType;

use crate::error::{driver_err, not_implemented};
use crate::runtime::CudaRuntime;

const QUERY_TILE: usize = 8;
const MAX_HEAD_DIM: usize = 128;
const BLOCK_THREADS: u32 = 256;

const FLASH_BODY: &str = r#"
#define FLASH_Q_TILE 8
#define FLASH_K_TILE 16
#define FLASH_MAX_D 128
#define FLASH_NEG_INF __int_as_float(0xff800000)

template <typename T>
__device__ __forceinline__ void flash_attention_body(
    const T* q,
    const T* key,
    const T* value,
    const T* mask,
    const int* total_lengths,
    const int* past_lengths,
    T* output,
    int batch,
    int q_heads,
    int kv_heads,
    int sq,
    int sk,
    int kv_capacity,
    int dim,
    int group,
    int causal,
    int mask_planes,
    int local_window,
    float scale,
    float softcap)
{
    __shared__ float k_tile[FLASH_K_TILE * FLASH_MAX_D];
    __shared__ float v_tile[FLASH_K_TILE * FLASH_MAX_D];
    __shared__ float scores[FLASH_Q_TILE * FLASH_K_TILE];
    __shared__ float running_m[FLASH_Q_TILE];
    __shared__ float running_l[FLASH_Q_TILE];
    __shared__ float row_alpha[FLASH_Q_TILE];
    __shared__ float row_beta[FLASH_Q_TILE];
    __shared__ float tile_m[FLASH_Q_TILE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int query_tiles = (sq + FLASH_Q_TILE - 1) / FLASH_Q_TILE;
    const int plane = blockIdx.x / query_tiles;
    const int query_tile = blockIdx.x - plane * query_tiles;
    const int h = plane % q_heads;
    const int b = plane / q_heads;
    const int kvh = h / group;
    const int qi = query_tile * FLASH_Q_TILE + warp;
    const bool valid_q = b < batch && qi < sq;
    const int logical_sk = total_lengths ? total_lengths[b] : sk;
    const int causal_max = past_lengths ? past_lengths[b] + qi : logical_sk - sq + qi;
    const int local_min =
        local_window > 0 ? max(0, causal_max + 1 - local_window) : 0;

    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (lane == 0) {
        running_m[warp] = FLASH_NEG_INF;
        running_l[warp] = 0.0f;
    }
    __syncthreads();

    const unsigned long long q_base =
        ((unsigned long long)(b * q_heads + h) * sq + qi) * dim;
    const unsigned long long kv_base =
        (unsigned long long)(b * kv_heads + kvh) * kv_capacity * dim;

    for (int key0 = 0; key0 < sk; key0 += FLASH_K_TILE) {
        const int tile_keys = min(FLASH_K_TILE, sk - key0);
        const int tile_values = FLASH_K_TILE * dim;
        for (int index = tid; index < tile_values; index += blockDim.x) {
            const int kj = index / dim;
            const int p = index - kj * dim;
            const int j = key0 + kj;
            float kval = 0.0f;
            float vval = 0.0f;
            if (j < sk) {
                const unsigned long long offset = kv_base + (unsigned long long)j * dim + p;
                kval = flash_load<T>(key[offset]);
                vval = flash_load<T>(value[offset]);
            }
            k_tile[kj * FLASH_MAX_D + p] = kval;
            v_tile[kj * FLASH_MAX_D + p] = vval;
        }
        __syncthreads();

        for (int kj = 0; kj < tile_keys; ++kj) {
            float dot = 0.0f;
            if (valid_q) {
                for (int p = lane; p < dim; p += 32) {
                    dot = fmaf(
                        flash_load<T>(q[q_base + p]),
                        k_tile[kj * FLASH_MAX_D + p],
                        dot);
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            if (lane == 0) {
                const int j = key0 + kj;
                float score = valid_q ? dot * scale : FLASH_NEG_INF;
                if (!valid_q || j >= logical_sk || (causal && j > causal_max)
                    || j < local_min) {
                    score = FLASH_NEG_INF;
                } else {
                    if (softcap > 0.0f) {
                        score = softcap * tanhf(score / softcap);
                    }
                    if (mask_planes > 0) {
                        int mask_plane = 0;
                        if (mask_planes == batch) {
                            mask_plane = b;
                        } else if (mask_planes == batch * q_heads) {
                            mask_plane = b * q_heads + h;
                        }
                        const unsigned long long mask_offset =
                            ((unsigned long long)mask_plane * sq + qi) * sk + j;
                        score += flash_load<T>(mask[mask_offset]);
                    }
                }
                scores[warp * FLASH_K_TILE + kj] = score;
            }
        }
        __syncthreads();

        if (lane == 0) {
            float maximum = FLASH_NEG_INF;
            for (int kj = 0; kj < tile_keys; ++kj) {
                maximum = fmaxf(maximum, scores[warp * FLASH_K_TILE + kj]);
            }
            float tile_sum = 0.0f;
            if (maximum != FLASH_NEG_INF) {
                for (int kj = 0; kj < tile_keys; ++kj) {
                    const float score = scores[warp * FLASH_K_TILE + kj];
                    if (score != FLASH_NEG_INF) {
                        tile_sum += expf(score - maximum);
                    }
                }
            }

            const float old_m = running_m[warp];
            const float old_l = running_l[warp];
            if (tile_sum == 0.0f) {
                row_alpha[warp] = 1.0f;
                row_beta[warp] = 0.0f;
                tile_m[warp] = FLASH_NEG_INF;
            } else {
                const float new_m = old_l > 0.0f ? fmaxf(old_m, maximum) : maximum;
                const float alpha = old_l > 0.0f ? expf(old_m - new_m) : 0.0f;
                const float beta = expf(maximum - new_m);
                running_m[warp] = new_m;
                running_l[warp] = alpha * old_l + beta * tile_sum;
                row_alpha[warp] = alpha;
                row_beta[warp] = beta;
                tile_m[warp] = maximum;
            }
        }
        __syncwarp();

        const float alpha = row_alpha[warp];
        const float beta = row_beta[warp];
        const float maximum = tile_m[warp];
        for (int slot = 0; slot < 4; ++slot) {
            const int p = lane + slot * 32;
            if (valid_q && p < dim) {
                float tile_out = 0.0f;
                if (beta != 0.0f) {
                    for (int kj = 0; kj < tile_keys; ++kj) {
                        const float score = scores[warp * FLASH_K_TILE + kj];
                        if (score != FLASH_NEG_INF) {
                            tile_out = fmaf(
                                expf(score - maximum),
                                v_tile[kj * FLASH_MAX_D + p],
                                tile_out);
                        }
                    }
                }
                out_acc[slot] = alpha * out_acc[slot] + beta * tile_out;
            }
        }
        __syncthreads();
    }

    const float denominator = running_l[warp];
    for (int slot = 0; slot < 4; ++slot) {
        const int p = lane + slot * 32;
        if (valid_q && p < dim) {
            const float result = denominator > 0.0f ? out_acc[slot] / denominator : 0.0f;
            output[q_base + p] = flash_store<T>(result);
        }
    }
}
"#;

const FLASH_F32_PREFIX: &str = r#"
template <typename T> __device__ __forceinline__ float flash_load(T value);
template <> __device__ __forceinline__ float flash_load<float>(float value) { return value; }
template <typename T> __device__ __forceinline__ T flash_store(float value);
template <> __device__ __forceinline__ float flash_store<float>(float value) { return value; }
"#;

const FLASH_F32_SUFFIX: &str = r#"
extern "C" __global__ void flash_attention_f32(
    const float* q, const float* key, const float* value, const float* mask,
    const int* total_lengths, const int* past_lengths, float* output,
    int batch, int q_heads, int kv_heads, int sq, int sk, int dim, int group,
    int kv_capacity, int causal, int mask_planes, int local_window,
    float scale, float softcap) {
    flash_attention_body<float>(q, key, value, mask, total_lengths, past_lengths, output,
                                batch, q_heads, kv_heads, sq, sk, kv_capacity, dim, group, causal,
                                mask_planes, local_window, scale, softcap);
}
"#;

const FLASH_HALF_PREFIX: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
template <typename T> __device__ __forceinline__ float flash_load(T value);
template <> __device__ __forceinline__ float flash_load<__half>(__half value) {
    return __half2float(value);
}
template <> __device__ __forceinline__ float flash_load<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
template <typename T> __device__ __forceinline__ T flash_store(float value);
template <> __device__ __forceinline__ __half flash_store<__half>(float value) {
    return __float2half_rn(value);
}
template <> __device__ __forceinline__ __nv_bfloat16 flash_store<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}
"#;

const FLASH_HALF_SUFFIX: &str = r#"
extern "C" __global__ void flash_attention_f16(
    const __half* q, const __half* key, const __half* value, const __half* mask,
    const int* total_lengths, const int* past_lengths, __half* output,
    int batch, int q_heads, int kv_heads, int sq, int sk, int dim, int group,
    int kv_capacity, int causal, int mask_planes, int local_window,
    float scale, float softcap) {
    flash_attention_body<__half>(q, key, value, mask, total_lengths, past_lengths, output,
                                 batch, q_heads, kv_heads, sq, sk, kv_capacity, dim, group, causal,
                                 mask_planes, local_window, scale, softcap);
}
extern "C" __global__ void flash_attention_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* key, const __nv_bfloat16* value,
    const __nv_bfloat16* mask, const int* total_lengths, const int* past_lengths,
    __nv_bfloat16* output,
    int batch, int q_heads, int kv_heads, int sq, int sk, int dim, int group,
    int kv_capacity, int causal, int mask_planes, int local_window,
    float scale, float softcap) {
    flash_attention_body<__nv_bfloat16>(
        q, key, value, mask, total_lengths, past_lengths, output, batch, q_heads, kv_heads,
        sq, sk, kv_capacity, dim, group, causal, mask_planes, local_window, scale, softcap);
}
"#;

const FLASH_F16_TENSOR_CORE: &str = r#"
#include <mma.h>

#define FLASH_TC_Q 16
#define FLASH_TC_K 16
#define FLASH_TC_D 128

extern "C" __global__ void flash_attention_f16_tc(
    const __half* q, const __half* key, const __half* value, const __half* mask,
    const int* total_lengths, const int* past_lengths, __half* output,
    int batch, int q_heads, int kv_heads, int sq, int sk, int dim, int group,
    int kv_capacity, int causal, int mask_planes, int local_window,
    float scale, float softcap)
{
    using namespace nvcuda;
    __shared__ __half q_tile[FLASH_TC_Q * FLASH_TC_D];
    __shared__ __half k_tile[FLASH_TC_K * FLASH_TC_D];
    __shared__ __half v_tile[FLASH_TC_K * FLASH_TC_D];
    __shared__ __half probabilities[FLASH_TC_Q * FLASH_TC_K];
    __shared__ float scores[FLASH_TC_Q * FLASH_TC_K];
    __shared__ float output_numerator[FLASH_TC_Q * FLASH_TC_D];
    __shared__ float pv_tile[FLASH_TC_Q * FLASH_TC_D];
    __shared__ float running_m[FLASH_TC_Q];
    __shared__ float running_l[FLASH_TC_Q];
    __shared__ float row_alpha[FLASH_TC_Q];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int query_tiles = (sq + FLASH_TC_Q - 1) / FLASH_TC_Q;
    const int plane = blockIdx.x / query_tiles;
    const int query_tile_index = blockIdx.x - plane * query_tiles;
    const int h = plane % q_heads;
    const int b = plane / q_heads;
    const int kvh = h / group;
    const int query0 = query_tile_index * FLASH_TC_Q;
    const int logical_sk = total_lengths ? total_lengths[b] : sk;
    const unsigned long long q_plane =
        (unsigned long long)(b * q_heads + h) * sq * dim;
    const unsigned long long kv_plane =
        (unsigned long long)(b * kv_heads + kvh) * kv_capacity * dim;

    for (int index = tid; index < FLASH_TC_Q * FLASH_TC_D; index += blockDim.x) {
        const int row = index / FLASH_TC_D;
        const int p = index - row * FLASH_TC_D;
        const int qi = query0 + row;
        q_tile[index] = qi < sq && p < dim ? q[q_plane + (unsigned long long)qi * dim + p]
                                           : __float2half(0.0f);
        output_numerator[index] = 0.0f;
    }
    if (tid < FLASH_TC_Q) {
        running_m[tid] = FLASH_NEG_INF;
        running_l[tid] = 0.0f;
    }
    __syncthreads();

    for (int key0 = 0; key0 < sk; key0 += FLASH_TC_K) {
        for (int index = tid; index < FLASH_TC_K * FLASH_TC_D; index += blockDim.x) {
            const int row = index / FLASH_TC_D;
            const int p = index - row * FLASH_TC_D;
            const int kj = key0 + row;
            if (kj < sk && p < dim) {
                const unsigned long long offset = kv_plane + (unsigned long long)kj * dim + p;
                k_tile[index] = key[offset];
                v_tile[index] = value[offset];
            } else {
                k_tile[index] = __float2half(0.0f);
                v_tile[index] = __float2half(0.0f);
            }
        }
        __syncthreads();

        if (warp == 0) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bfrag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
            wmma::fill_fragment(accumulator, 0.0f);
            for (int p = 0; p < dim; p += 16) {
                wmma::load_matrix_sync(a, q_tile + p, FLASH_TC_D);
                // K is [key, dim] row-major, byte-identical to K^T [dim,key]
                // column-major with leading dimension FLASH_TC_D.
                wmma::load_matrix_sync(bfrag, k_tile + p, FLASH_TC_D);
                wmma::mma_sync(accumulator, a, bfrag, accumulator);
            }
            wmma::store_matrix_sync(scores, accumulator, FLASH_TC_K, wmma::mem_row_major);
        }
        __syncthreads();

        if (tid < FLASH_TC_Q * FLASH_TC_K) {
            const int row = tid / FLASH_TC_K;
            const int col = tid - row * FLASH_TC_K;
            const int qi = query0 + row;
            const int kj = key0 + col;
            float score = scores[tid] * scale;
            const int causal_max =
                past_lengths ? past_lengths[b] + qi : logical_sk - sq + qi;
            const int local_min =
                local_window > 0 ? max(0, causal_max + 1 - local_window) : 0;
            if (qi >= sq || kj >= logical_sk || (causal && kj > causal_max)
                || kj < local_min) {
                score = FLASH_NEG_INF;
            } else {
                if (softcap > 0.0f) {
                    score = softcap * tanhf(score / softcap);
                }
                if (mask_planes > 0) {
                    int mask_plane = 0;
                    if (mask_planes == batch) {
                        mask_plane = b;
                    } else if (mask_planes == batch * q_heads) {
                        mask_plane = b * q_heads + h;
                    }
                    const unsigned long long mask_offset =
                        ((unsigned long long)mask_plane * sq + qi) * sk + kj;
                    score += __half2float(mask[mask_offset]);
                }
            }
            scores[tid] = score;
        }
        __syncthreads();

        if (tid < FLASH_TC_Q) {
            const int row = tid;
            float tile_maximum = FLASH_NEG_INF;
            for (int col = 0; col < FLASH_TC_K; ++col) {
                tile_maximum = fmaxf(tile_maximum, scores[row * FLASH_TC_K + col]);
            }
            float tile_sum = 0.0f;
            if (tile_maximum != FLASH_NEG_INF) {
                for (int col = 0; col < FLASH_TC_K; ++col) {
                    const float score = scores[row * FLASH_TC_K + col];
                    if (score != FLASH_NEG_INF) {
                        tile_sum += expf(score - tile_maximum);
                    }
                }
            }
            const float old_m = running_m[row];
            const float old_l = running_l[row];
            float alpha = 1.0f;
            float beta = 0.0f;
            if (tile_sum > 0.0f) {
                const float new_m = old_l > 0.0f ? fmaxf(old_m, tile_maximum) : tile_maximum;
                alpha = old_l > 0.0f ? expf(old_m - new_m) : 0.0f;
                beta = expf(tile_maximum - new_m);
                running_m[row] = new_m;
                running_l[row] = alpha * old_l + beta * tile_sum;
            }
            row_alpha[row] = alpha;
            for (int col = 0; col < FLASH_TC_K; ++col) {
                const float score = scores[row * FLASH_TC_K + col];
                const float probability =
                    beta != 0.0f && score != FLASH_NEG_INF ? beta * expf(score - tile_maximum)
                                                           : 0.0f;
                probabilities[row * FLASH_TC_K + col] = __float2half_rn(probability);
            }
        }
        __syncthreads();

        for (int index = tid; index < FLASH_TC_Q * dim; index += blockDim.x) {
            const int row = index / dim;
            const int p = index - row * dim;
            output_numerator[row * FLASH_TC_D + p] *= row_alpha[row];
        }
        __syncthreads();

        if (warp < dim / 16) {
            const int column = warp * 16;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bfrag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
            wmma::fill_fragment(accumulator, 0.0f);
            wmma::load_matrix_sync(a, probabilities, FLASH_TC_K);
            wmma::load_matrix_sync(bfrag, v_tile + column, FLASH_TC_D);
            wmma::mma_sync(accumulator, a, bfrag, accumulator);
            wmma::store_matrix_sync(
                pv_tile + column, accumulator, FLASH_TC_D, wmma::mem_row_major);
        }
        __syncthreads();

        for (int index = tid; index < FLASH_TC_Q * dim; index += blockDim.x) {
            const int row = index / dim;
            const int p = index - row * dim;
            output_numerator[row * FLASH_TC_D + p] += pv_tile[row * FLASH_TC_D + p];
        }
        __syncthreads();
    }

    for (int index = tid; index < FLASH_TC_Q * dim; index += blockDim.x) {
        const int row = index / dim;
        const int p = index - row * dim;
        const int qi = query0 + row;
        if (qi < sq) {
            const float denominator = running_l[row];
            const float result =
                denominator > 0.0f ? output_numerator[row * FLASH_TC_D + p] / denominator : 0.0f;
            output[q_plane + (unsigned long long)qi * dim + p] = __float2half_rn(result);
        }
    }
}
"#;

static FLASH_F32_SOURCE: LazyLock<String> =
    LazyLock::new(|| [FLASH_F32_PREFIX, FLASH_BODY, FLASH_F32_SUFFIX].concat());
static FLASH_HALF_SOURCE: LazyLock<String> = LazyLock::new(|| {
    [
        FLASH_HALF_PREFIX,
        FLASH_BODY,
        FLASH_HALF_SUFFIX,
        FLASH_F16_TENSOR_CORE,
    ]
    .concat()
});

pub(super) fn supported(sq: usize, head_dim: usize) -> bool {
    sq > 1 && (1..=MAX_HEAD_DIM).contains(&head_dim)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    runtime: &CudaRuntime,
    dtype: DataType,
    num_heads: usize,
    num_kv_heads: usize,
    causal: bool,
    batch: usize,
    sq: usize,
    sk: usize,
    kv_capacity: usize,
    head_dim: usize,
    group: usize,
    scale: f32,
    q: CUdeviceptr,
    k: CUdeviceptr,
    v: CUdeviceptr,
    output: CUdeviceptr,
    mask: CUdeviceptr,
    mask_planes: i32,
    total_lengths: CUdeviceptr,
    past_lengths: CUdeviceptr,
    local_window: i32,
    softcap: f32,
) -> Result<()> {
    if !supported(sq, head_dim) {
        return Err(not_implemented(format!(
            "fused Attention requires seq_q > 1 and head_dim <= {MAX_HEAD_DIM}; \
             got seq_q={sq}, head_dim={head_dim}"
        )));
    }
    let tensor_core_f16 = dtype == DataType::Float16
        && head_dim.is_multiple_of(16)
        && runtime.capabilities().compute_capability().0 >= 7;
    let (module, source, entry, query_tile) = match dtype {
        DataType::Float32 => (
            "flash_attention_f32_v1",
            FLASH_F32_SOURCE.as_str(),
            "flash_attention_f32",
            QUERY_TILE,
        ),
        DataType::Float16 if tensor_core_f16 => (
            "flash_attention_half_v2",
            FLASH_HALF_SOURCE.as_str(),
            "flash_attention_f16_tc",
            16,
        ),
        DataType::Float16 => (
            "flash_attention_half_v2",
            FLASH_HALF_SOURCE.as_str(),
            "flash_attention_f16",
            QUERY_TILE,
        ),
        DataType::BFloat16 => (
            "flash_attention_half_v2",
            FLASH_HALF_SOURCE.as_str(),
            "flash_attention_bf16",
            QUERY_TILE,
        ),
        other => {
            return Err(not_implemented(format!(
                "fused Attention dtype {other:?} (supported: Float32, Float16, BFloat16)"
            )));
        }
    };

    let as_i32 = |name: &str, value: usize| {
        i32::try_from(value).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep fused Attention: {name} {value} exceeds i32"
            ))
        })
    };
    let batch_i = as_i32("batch", batch)?;
    let heads_i = as_i32("num_heads", num_heads)?;
    let kv_heads_i = as_i32("num_kv_heads", num_kv_heads)?;
    let sq_i = as_i32("seq_q", sq)?;
    let sk_i = as_i32("seq_k", sk)?;
    let kv_capacity_i = as_i32("KV capacity", kv_capacity)?;
    let dim_i = as_i32("head_dim", head_dim)?;
    let group_i = as_i32("GQA group", group)?;
    let causal_i = i32::from(causal);
    let query_tiles = sq.div_ceil(query_tile);
    let blocks = batch
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(query_tiles))
        .ok_or_else(|| EpError::KernelFailed("cuda_ep fused Attention grid overflow".into()))?;
    let grid_x = u32::try_from(blocks).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep fused Attention requires {blocks} blocks, exceeding CUDA grid.x"
        ))
    })?;
    if grid_x == 0 || sk == 0 {
        return runtime.synchronize();
    }

    let function = runtime.nvrtc_function(module, source, entry)?;
    let mut builder = runtime.stream().launch_builder(&function);
    builder
        .arg(&q)
        .arg(&k)
        .arg(&v)
        .arg(&mask)
        .arg(&total_lengths)
        .arg(&past_lengths)
        .arg(&output)
        .arg(&batch_i)
        .arg(&heads_i)
        .arg(&kv_heads_i)
        .arg(&sq_i)
        .arg(&sk_i)
        .arg(&dim_i)
        .arg(&group_i)
        .arg(&kv_capacity_i)
        .arg(&causal_i)
        .arg(&mask_planes)
        .arg(&local_window)
        .arg(&scale)
        .arg(&softcap);
    // SAFETY: the selected entry matches the dtype-specific pointer ABI; all
    // buffers and dimensions were validated by AttentionKernel before launch.
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|error| driver_err(&format!("launch {entry}"), error))?;
    runtime.synchronize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_gate_targets_prefill_and_common_head_dims() {
        assert!(supported(2, 128));
        assert!(supported(2048, 64));
        assert!(!supported(1, 128));
        assert!(!supported(2, 129));
        assert!(!supported(2, 0));
    }
}

//! Fused **normalization** kernels on the GPU via runtime-compiled (NVRTC)
//! kernels: `LayerNormalization` (ai.onnx + `com.microsoft`),
//! `SkipLayerNormalization` and `SimplifiedLayerNormalization` /
//! `RMSNormalization` (`com.microsoft` / ai.onnx).
//!
//! ## Backend choice — custom fused NVRTC, and *why* (the "我们能优化的才自己写" case)
//!
//! A library path (cuDNN/cub reduction + several pointwise passes) reads the
//! activation from HBM multiple times. The **fused** kernel does the mean/variance
//! reduction, the normalize, and the affine (`γ·x̂ + β`) in **one** pass over a
//! single HBM read — the classic normalization fusion win, and PyTorch's own
//! `LayerNorm` CUDA kernel is fused for exactly this reason.
//! `SkipLayerNormalization` folds the residual add (`input + skip + bias`) into
//! the same kernel, saving an entire tensor round-trip. `RMSNormalization` drops
//! the mean subtraction (root-mean-square scale only) — the LLaMA-family norm.
//!
//! Numerics mirror `crates/onnx-runtime-ep-cpu/src/kernels/layernorm.rs`:
//!
//! ```text
//! LayerNorm: y = (x - mean) / sqrt(var + eps) · scale + bias
//! RMSNorm:   y = x / sqrt(mean(x²) + eps) · scale
//! ```
//!
//! with `mean`/`var` the **population** statistics (divide by N) over the
//! normalized axes `[axis..]` (LayerNorm) or the last dimension (Skip/RMS).
//!
//! ## Limits (actionable errors, never panics — RULES.md #1)
//!
//! * activation dtype other than f32/f16 → deferred (names the dtype + op).
//! * `axis`/last-dim size 0, or a `scale`/`bias`/`gamma`/`beta` length that does
//!   not match the normalized size → rejected, naming the offending length.
//! * non-contiguous (strided) operands → "materialise first" error.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

use super::softmax::resolve_axis;

/// NVRTC source for the fused f32 `LayerNormalization`. One block per group
/// (`group = prod(shape[..axis])`); the block reduces the mean then the variance
/// over `norm_size = prod(shape[axis..])` in shared memory, then writes the
/// normalized+affine output in a third pass. Optional `mean`/`inv_std` outputs
/// are written when the pointers are non-null.
const LAYERNORM_SRC: &str = r#"
#include <cuda_fp16.h>

__device__ __forceinline__ float load_layernorm_param(
    const void* values, const int is_half, const int index) {
    return is_half
        ? __half2float(((const __half*)values)[index])
        : ((const float*)values)[index];
}

extern "C" __global__ void layernorm_f32(
    const float* x,
    const float* scale,
    const float* bias,        // null when absent
    float*       y,
    float*       mean_out,    // null when not requested
    float*       invstd_out,  // null when not requested
    const int    num_groups,
    const int    norm_size,
    const int    has_bias,
    const float  epsilon)
{
    const int g = blockIdx.x;
    if (g >= num_groups) return;
    const size_t base = (size_t)g * norm_size;

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    // Pass 1: mean.
    float s = 0.0f;
    for (int j = tid; j < norm_size; j += nt) s += x[base + j];
    red[tid] = s;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float mean = red[0] / (float)norm_size;
    __syncthreads();

    // Pass 2: population variance.
    float v = 0.0f;
    for (int j = tid; j < norm_size; j += nt) {
        const float d = x[base + j] - mean;
        v += d * d;
    }
    red[tid] = v;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float var = red[0] / (float)norm_size;
    const float inv_std = 1.0f / sqrtf(var + epsilon);

    if (tid == 0) {
        if (mean_out)   mean_out[g]   = mean;
        if (invstd_out) invstd_out[g] = inv_std;
    }

    // Pass 3: normalize + affine.
    for (int j = tid; j < norm_size; j += nt) {
        const float xhat = (x[base + j] - mean) * inv_std;
        float o = xhat * scale[j];
        if (has_bias) o += bias[j];
        y[base + j] = o;
    }
}

extern "C" __global__ void layernorm_f16(
    const __half* x,
    const void*   scale,
    const void*   bias,
    __half*       y,
    float*        mean_out,
    float*        invstd_out,
    const int     num_groups,
    const int     norm_size,
    const int     scale_is_half,
    const int     bias_is_half,
    const int     has_bias,
    const float   epsilon)
{
    const int g = blockIdx.x;
    if (g >= num_groups) return;
    const size_t base = (size_t)g * norm_size;

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    float s = 0.0f;
    for (int j = tid; j < norm_size; j += nt)
        s += __half2float(x[base + j]);
    red[tid] = s;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float mean = red[0] / (float)norm_size;
    __syncthreads();

    float v = 0.0f;
    for (int j = tid; j < norm_size; j += nt) {
        const float d = __half2float(x[base + j]) - mean;
        v += d * d;
    }
    red[tid] = v;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float inv_std =
        1.0f / sqrtf(red[0] / (float)norm_size + epsilon);
    if (tid == 0) {
        if (mean_out) mean_out[g] = mean;
        if (invstd_out) invstd_out[g] = inv_std;
    }

    for (int j = tid; j < norm_size; j += nt) {
        const float xhat = (__half2float(x[base + j]) - mean) * inv_std;
        float o = xhat * load_layernorm_param(scale, scale_is_half, j);
        if (has_bias)
            o += load_layernorm_param(bias, bias_is_half, j);
        y[base + j] = __float2half_rn(o);
    }
}
"#;

/// NVRTC source for the fused f32 `RMSNormalization` /
/// `SimplifiedLayerNormalization`: no mean subtraction, scale by the inverse
/// root-mean-square.
const RMSNORM_SRC: &str = r#"
#include <cuda_fp16.h>

__device__ __forceinline__ float load_rmsnorm_scale(
    const void* values, const int is_half, const int index) {
    return is_half
        ? __half2float(((const __half*)values)[index])
        : ((const float*)values)[index];
}

extern "C" __global__ void rmsnorm_f32(
    const float* x,
    const float* scale,
    float*       y,
    float*       invstd_out,  // null when not requested
    const int    num_groups,
    const int    norm_size,
    const float  epsilon)
{
    const int g = blockIdx.x;
    if (g >= num_groups) return;
    const size_t base = (size_t)g * norm_size;

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    // Keep the correctness path in the CPU kernel's left-to-right f32 order.
    // Accuracy-level-4 MatMulNBits quantizes activations, so even a one-ulp
    // normalization difference can cross an int8 rounding boundary in decode.
    if (tid == 0) {
        float ss = 0.0f;
        for (int j = 0; j < norm_size; ++j) {
            const float xv = x[base + j];
            // Match the CPU kernel's separate multiply then add. NVRTC otherwise
            // contracts this expression to FMA and changes recurrent decode state.
            ss = __fadd_rn(ss, __fmul_rn(xv, xv));
        }
        red[0] = ss;
    }
    __syncthreads();
    const float ms = red[0] / (float)norm_size;
    const float inv_std = 1.0f / sqrtf(ms + epsilon);
    if (tid == 0 && invstd_out) invstd_out[g] = inv_std;

    for (int j = tid; j < norm_size; j += nt)
        y[base + j] = x[base + j] * inv_std * scale[j];
}

extern "C" __global__ void rmsnorm_f16(
    const __half* x,
    const void*   scale,
    __half*       y,
    float*        invstd_out,
    const int     num_groups,
    const int     norm_size,
    const int     scale_is_half,
    const float   epsilon)
{
    const int g = blockIdx.x;
    if (g >= num_groups) return;
    const size_t base = (size_t)g * norm_size;

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    float ss = 0.0f;
    for (int j = tid; j < norm_size; j += nt) {
        const float xv = __half2float(x[base + j]);
        ss += xv * xv;
    }
    red[tid] = ss;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float inv_std =
        1.0f / sqrtf(red[0] / (float)norm_size + epsilon);
    if (tid == 0 && invstd_out) invstd_out[g] = inv_std;

    for (int j = tid; j < norm_size; j += nt) {
        const float o = __half2float(x[base + j]) * inv_std
            * load_rmsnorm_scale(scale, scale_is_half, j);
        y[base + j] = __float2half_rn(o);
    }
}
"#;

/// NVRTC source for `com.microsoft::SkipSimplifiedLayerNormalization`.
/// The residual sum supports right-aligned NumPy broadcasting for `skip`.
const SKIP_RMSNORM_SRC: &str = r#"
#include <cuda_fp16.h>

__device__ __forceinline__ float load_skip_val(
    const void* values, const int is_half, const int index) {
    return is_half
        ? __half2float(((const __half*)values)[index])
        : ((const float*)values)[index];
}

extern "C" __global__ void skip_rmsnorm_f32(
    const float* input,
    const float* skip,
    const float* gamma,
    const float* bias,          // null when absent
    float*       y,
    float*       sum_out,       // null when not requested
    float*       mean_out,      // null when not requested (always zero)
    float*       invstd_out,    // null when not requested
    const unsigned long long* metadata,
    const int    rank,
    const int    num_groups,
    const int    norm_size,
    const int    has_bias,
    const float  epsilon)
{
    const int g = blockIdx.x;
    if (g >= num_groups) return;
    const size_t base = (size_t)g * norm_size;
    const unsigned long long* shape = metadata;
    const unsigned long long* skip_strides = metadata + rank;

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    for (int j = tid; j < norm_size; j += nt) {
        unsigned long long linear = (unsigned long long)base + j;
        unsigned long long skip_index = 0;
        for (int d = rank - 1; d >= 0; --d) {
            const unsigned long long coord = linear % shape[d];
            linear /= shape[d];
            skip_index += coord * skip_strides[d];
        }
        float sv = input[base + j] + skip[skip_index];
        if (has_bias) sv += bias[j];
        y[base + j] = sv;
        if (sum_out) sum_out[base + j] = sv;
    }
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        for (int j = 0; j < norm_size; ++j) {
            const float sv = y[base + j];
            // Keep the residual RMS reduction bit-identical to the CPU kernel.
            ss = __fadd_rn(ss, __fmul_rn(sv, sv));
        }
        red[0] = ss;
    }
    __syncthreads();
    const float inv_std = 1.0f / sqrtf(red[0] / (float)norm_size + epsilon);
    if (tid == 0) {
        if (mean_out) mean_out[g] = 0.0f;
        if (invstd_out) invstd_out[g] = inv_std;
    }
    __syncthreads();
    for (int j = tid; j < norm_size; j += nt)
        y[base + j] = (y[base + j] * inv_std) * gamma[j];
}

union SkipHalf4 {
    unsigned long long raw;
    __half2 pair[2];
};

// One warp covers aligned half4 chunks. The launch predicate guarantees that
// norm_size is divisible by 32 lanes * 4 halves, so every lane owns the same
// number of complete chunks and no tail handling is needed.
extern "C" __global__ void skip_rmsnorm_f16_warp_half4(
    const __half* input,
    const __half* skip,
    const void*   gamma,
    const void*   bias,
    __half*       y,
    __half*       sum_out,
    void*         mean_out,
    void*         invstd_out,
    const unsigned long long* metadata,
    const int     rank,
    const int     num_groups,
    const int     norm_size,
    const int     has_bias,
    const int     dense_skip,
    const int     gamma_is_half,
    const int     bias_is_half,
    const int     stat_is_half,
    const float   epsilon)
{
    const int g = blockIdx.x;
    if (g >= num_groups) return;
    const size_t base = (size_t)g * norm_size;
    const int lane = threadIdx.x;
    const int chunks_per_lane = norm_size / (32 * 4);
    const unsigned long long* input4 =
        (const unsigned long long*)(input + base);
    const unsigned long long* skip4 =
        (const unsigned long long*)(skip + base);
    const unsigned long long* gamma4 =
        (const unsigned long long*)gamma;
    unsigned long long* y4 = (unsigned long long*)(y + base);
    unsigned long long* sum4 =
        sum_out ? (unsigned long long*)(sum_out + base) : 0;
    float ss0 = 0.0f;
    float ss1 = 0.0f;
    float ss2 = 0.0f;
    float ss3 = 0.0f;

    for (int item = 0; item < chunks_per_lane; ++item) {
        const int chunk = lane + item * 32;
        SkipHalf4 input_v;
        SkipHalf4 skip_v;
        SkipHalf4 residual;
        input_v.raw = input4[chunk];
        skip_v.raw = skip4[chunk];
        residual.pair[0] = __hadd2(input_v.pair[0], skip_v.pair[0]);
        residual.pair[1] = __hadd2(input_v.pair[1], skip_v.pair[1]);
        y4[chunk] = residual.raw;
        if (sum4) sum4[chunk] = residual.raw;
        const float2 rounded0 = __half22float2(residual.pair[0]);
        const float2 rounded1 = __half22float2(residual.pair[1]);
        ss0 += rounded0.x * rounded0.x;
        ss1 += rounded0.y * rounded0.y;
        ss2 += rounded1.x * rounded1.x;
        ss3 += rounded1.y * rounded1.y;
    }

    float ss = (ss0 + ss1) + (ss2 + ss3);
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    }
    float inv_std = 0.0f;
    if (lane == 0) {
        inv_std = 1.0f / sqrtf(ss / (float)norm_size + epsilon);
        if (mean_out) {
            if (stat_is_half) ((__half*)mean_out)[g] = __float2half_rn(0.0f);
            else ((float*)mean_out)[g] = 0.0f;
        }
        if (invstd_out) {
            if (stat_is_half) ((__half*)invstd_out)[g] = __float2half_rn(inv_std);
            else ((float*)invstd_out)[g] = inv_std;
        }
    }
    inv_std = __shfl_sync(0xffffffffu, inv_std, 0);

    const float* gamma_f = (const float*)gamma;
    for (int item = 0; item < chunks_per_lane; ++item) {
        const int chunk = lane + item * 32;
        SkipHalf4 residual;
        SkipHalf4 output;
        residual.raw = y4[chunk];
        const float2 value0 = __half22float2(residual.pair[0]);
        const float2 value1 = __half22float2(residual.pair[1]);
        // gamma is only ever a final multiplicand (never part of the fp32
        // variance accumulation), so an fp32 gamma is loaded at full precision
        // while an fp16 gamma keeps the wide half4 load. This lets decoders that
        // export gamma in fp32 (e.g. Phi) still take the vectorized warp path.
        float scale0x, scale0y, scale1x, scale1y;
        if (gamma_is_half) {
            SkipHalf4 scale;
            scale.raw = gamma4[chunk];
            const float2 scale0 = __half22float2(scale.pair[0]);
            const float2 scale1 = __half22float2(scale.pair[1]);
            scale0x = scale0.x;
            scale0y = scale0.y;
            scale1x = scale1.x;
            scale1y = scale1.y;
        } else {
            const int j = chunk << 2;
            scale0x = gamma_f[j];
            scale0y = gamma_f[j + 1];
            scale1x = gamma_f[j + 2];
            scale1y = gamma_f[j + 3];
        }
        output.pair[0] = __floats2half2_rn(
            value0.x * inv_std * scale0x,
            value0.y * inv_std * scale0y);
        output.pair[1] = __floats2half2_rn(
            value1.x * inv_std * scale1x,
            value1.y * inv_std * scale1y);
        y4[chunk] = output.raw;
    }
}

extern "C" __global__ void skip_rmsnorm_f16(
    const __half* input,
    const __half* skip,
    const void*   gamma,
    const void*   bias,         // null when absent
    __half*       y,
    __half*       sum_out,      // null when not requested
    void*         mean_out,     // null when not requested (always zero)
    void*         invstd_out,   // null when not requested
    const unsigned long long* metadata,
    const int     rank,
    const int     num_groups,
    const int     norm_size,
    const int     has_bias,
    const int     dense_skip,
    const int     gamma_is_half,
    const int     bias_is_half,
    const int     stat_is_half,
    const float   epsilon)
{
    const int g = blockIdx.x;
    if (g >= num_groups) return;
    const size_t base = (size_t)g * norm_size;
    const unsigned long long* shape = metadata;
    const unsigned long long* skip_strides = metadata + rank;

    const int lane = threadIdx.x;

    // fp32 accumulate over the fp16-rounded residual so the RMS matches the
    // residual value stored into `sum_out` and reused by the next layer.
    float ss = 0.0f;
    const bool vectorized = dense_skip && ((base & 1) == 0);
    if (vectorized) {
        const int pairs = norm_size >> 1;
        const __half2* input2 = (const __half2*)(input + base);
        const __half2* skip2 = (const __half2*)(skip + base);
        __half2* y2 = (__half2*)(y + base);
        __half2* sum2 = sum_out ? (__half2*)(sum_out + base) : 0;
        for (int pair = lane; pair < pairs; pair += 32) {
            const float2 input_v = __half22float2(input2[pair]);
            const float2 skip_v = __half22float2(skip2[pair]);
            const int j = pair << 1;
            float sv0 = input_v.x + skip_v.x;
            float sv1 = input_v.y + skip_v.y;
            if (has_bias) {
                sv0 += load_skip_val(bias, bias_is_half, j);
                sv1 += load_skip_val(bias, bias_is_half, j + 1);
            }
            const __half svh0 = __float2half_rn(sv0);
            const __half svh1 = __float2half_rn(sv1);
            const __half2 svh = __halves2half2(svh0, svh1);
            y2[pair] = svh;
            if (sum2) sum2[pair] = svh;
            const float2 rounded = __half22float2(svh);
            ss += rounded.x * rounded.x;
            ss += rounded.y * rounded.y;
        }
        if ((norm_size & 1) && lane == 0) {
            const int j = norm_size - 1;
            float sv = __half2float(input[base + j]) + __half2float(skip[base + j]);
            if (has_bias) sv += load_skip_val(bias, bias_is_half, j);
            const __half svh = __float2half_rn(sv);
            y[base + j] = svh;
            if (sum_out) sum_out[base + j] = svh;
            const float rounded = __half2float(svh);
            ss += rounded * rounded;
        }
    } else {
        for (int j = lane; j < norm_size; j += 32) {
            unsigned long long linear = (unsigned long long)base + j;
            unsigned long long skip_index = 0;
            for (int d = rank - 1; d >= 0; --d) {
                const unsigned long long coord = linear % shape[d];
                linear /= shape[d];
                skip_index += coord * skip_strides[d];
            }
            float sv = __half2float(input[base + j]) + __half2float(skip[skip_index]);
            if (has_bias) sv += load_skip_val(bias, bias_is_half, j);
            const __half svh = __float2half_rn(sv);
            y[base + j] = svh;
            if (sum_out) sum_out[base + j] = svh;
            const float rounded = __half2float(svh);
            ss += rounded * rounded;
        }
    }
    __syncwarp();
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    }
    float inv_std = 0.0f;
    if (lane == 0) {
        inv_std = 1.0f / sqrtf(ss / (float)norm_size + epsilon);
        if (mean_out) {
            if (stat_is_half) ((__half*)mean_out)[g] = __float2half_rn(0.0f);
            else ((float*)mean_out)[g] = 0.0f;
        }
        if (invstd_out) {
            if (stat_is_half) ((__half*)invstd_out)[g] = __float2half_rn(inv_std);
            else ((float*)invstd_out)[g] = inv_std;
        }
    }
    inv_std = __shfl_sync(0xffffffffu, inv_std, 0);
    if (vectorized) {
        const int pairs = norm_size >> 1;
        __half2* y2 = (__half2*)(y + base);
        for (int pair = lane; pair < pairs; pair += 32) {
            const float2 residual = __half22float2(y2[pair]);
            const int j = pair << 1;
            const float out0 = residual.x * inv_std
                * load_skip_val(gamma, gamma_is_half, j);
            const float out1 = residual.y * inv_std
                * load_skip_val(gamma, gamma_is_half, j + 1);
            y2[pair] = __floats2half2_rn(out0, out1);
        }
        if ((norm_size & 1) && lane == 0) {
            const int j = norm_size - 1;
            const float v = __half2float(y[base + j]) * inv_std
                * load_skip_val(gamma, gamma_is_half, j);
            y[base + j] = __float2half_rn(v);
        }
    } else {
        for (int j = lane; j < norm_size; j += 32) {
            const float v = __half2float(y[base + j]) * inv_std
                * load_skip_val(gamma, gamma_is_half, j);
            y[base + j] = __float2half_rn(v);
        }
    }
}
"#;

/// NVRTC source for the fused f32 `SkipLayerNormalization` (`com.microsoft`):
/// `y = LayerNorm(input + skip + bias) · gamma + beta`. The residual sum is
/// computed once into `y` (scratch) and optionally published to `sum_out`, then
/// the standard two-pass LayerNorm runs over it.
const SKIP_LAYERNORM_SRC: &str = r#"
extern "C" __global__ void skip_layernorm_f32(
    const float* input,
    const float* skip,
    const float* gamma,
    const float* beta,        // null when absent
    const float* bias,        // null when absent (per-channel, length norm_size)
    float*       y,
    float*       sum_out,      // null when not requested
    float*       mean_out,     // null when not requested
    float*       invstd_out,   // null when not requested
    const int    num_groups,
    const int    norm_size,
    const int    has_beta,
    const int    has_bias,
    const float  epsilon)
{
    const int g = blockIdx.x;
    if (g >= num_groups) return;
    const size_t base = (size_t)g * norm_size;

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    // Residual sum s = input + skip (+ bias); stash in y and optionally sum_out.
    for (int j = tid; j < norm_size; j += nt) {
        float sv = input[base + j] + skip[base + j];
        if (has_bias) sv += bias[j];
        y[base + j] = sv;
        if (sum_out) sum_out[base + j] = sv;
    }
    __syncthreads();

    // Pass 1: mean of s.
    float s = 0.0f;
    for (int j = tid; j < norm_size; j += nt) s += y[base + j];
    red[tid] = s;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float mean = red[0] / (float)norm_size;
    __syncthreads();

    // Pass 2: population variance of s.
    float v = 0.0f;
    for (int j = tid; j < norm_size; j += nt) {
        const float d = y[base + j] - mean;
        v += d * d;
    }
    red[tid] = v;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float var = red[0] / (float)norm_size;
    const float inv_std = 1.0f / sqrtf(var + epsilon);
    if (tid == 0) {
        if (mean_out)   mean_out[g]   = mean;
        if (invstd_out) invstd_out[g] = inv_std;
    }
    __syncthreads();

    // Pass 3: normalize + affine (gamma / optional beta).
    for (int j = tid; j < norm_size; j += nt) {
        const float xhat = (y[base + j] - mean) * inv_std;
        float o = xhat * gamma[j];
        if (has_beta) o += beta[j];
        y[base + j] = o;
    }
}
"#;

const LAYERNORM_MODULE: &str = "layernorm_f16_v1";
const RMSNORM_MODULE: &str = "rmsnorm_f16_v1";
const SKIP_RMSNORM_MODULE: &str = "skip_rmsnorm_f16_warp_v5";
const SKIP_LAYERNORM_MODULE: &str = "skip_layernorm_f32";

/// Threads per block for the norm reductions (power of two → exact tree reduce).
const NORM_BLOCK: u32 = 256;
const SKIP_RMSNORM_WARP_HALF4_MULTIPLE: usize = 32 * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SkipRmsnormVariant {
    F32,
    F16Generic,
    F16WarpHalf4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SkipRmsnormSelection {
    variant: SkipRmsnormVariant,
    entry: &'static str,
    reason: &'static str,
}

/// Environment opt-out for the fp32-gamma warp-half4 path, mirroring the other
/// CUDA A/B switches. Any value other than unset/empty/`0` forces an fp16
/// activation norm with an fp32 gamma back onto the generic warp kernel (for
/// A/B measurement or rollback); an fp16 gamma is unaffected.
const FP32_GAMMA_WARP_DISABLE_ENV: &str = "ONNX_GENAI_CUDA_DISABLE_FP32_GAMMA_WARP_NORM";

fn fp32_gamma_warp_disabled() -> bool {
    std::env::var_os(FP32_GAMMA_WARP_DISABLE_ENV)
        .is_some_and(|value| value != "0" && !value.is_empty())
}

/// Select the one-warp half4 path by its actual data-layout capabilities, never
/// by a model-specific hidden dimension.
fn select_skip_rmsnorm_variant(
    is_half: bool,
    dense_skip: bool,
    norm_size: usize,
    has_bias: bool,
    gamma_is_half: bool,
) -> SkipRmsnormSelection {
    // gamma is only a final multiplicand (never part of the fp32 variance
    // accumulation), so the vectorized warp path serves an fp32 gamma at full
    // precision too. The A/B switch keeps the pre-existing fp16-gamma-only gate.
    let gamma_ok = gamma_is_half || !fp32_gamma_warp_disabled();
    if is_half
        && dense_skip
        && norm_size.is_multiple_of(SKIP_RMSNORM_WARP_HALF4_MULTIPLE)
        && !has_bias
        && gamma_ok
    {
        SkipRmsnormSelection {
            variant: SkipRmsnormVariant::F16WarpHalf4,
            entry: "skip_rmsnorm_f16_warp_half4",
            reason: if gamma_is_half {
                "variant=warp_half4;dtype=fp16;dense_skip;bias=none;gamma=fp16;\
                 hidden%128==0;one_warp"
            } else {
                "variant=warp_half4;dtype=fp16;dense_skip;bias=none;gamma=fp32;\
                 hidden%128==0;one_warp"
            },
        }
    } else if is_half {
        SkipRmsnormSelection {
            variant: SkipRmsnormVariant::F16Generic,
            entry: "skip_rmsnorm_f16",
            reason: "variant=generic;dtype=fp16;not(dense_skip & bias=none & \
                     hidden%128==0)",
        }
    } else {
        SkipRmsnormSelection {
            variant: SkipRmsnormVariant::F32,
            entry: "skip_rmsnorm_f32",
            reason: "variant=generic;dtype=fp32",
        }
    }
}

/// Reject any non-f32 tensor with an actionable, op-named error (RULES.md #1).
fn require_f32(op: &str, name: &str, dt: DataType) -> Result<()> {
    if dt != DataType::Float32 {
        return Err(not_implemented(format!(
            "{op} with {name} dtype {dt:?} (this slice is f32-only; f16/bf16 pending)"
        )));
    }
    Ok(())
}

fn require_f16_or_f32(op: &str, name: &str, dt: DataType) -> Result<()> {
    if !matches!(dt, DataType::Float16 | DataType::Float32) {
        return Err(not_implemented(format!(
            "{op} with {name} dtype {dt:?} (expected f16 or f32)"
        )));
    }
    Ok(())
}

/// Reject a strided view with a "materialise first" error.
fn require_contiguous(op: &str, name: &str, contiguous: bool) -> Result<()> {
    if !contiguous {
        return Err(not_implemented(format!(
            "{op} with a non-contiguous (strided) {name}; \
             insert an explicit copy to materialise it before the op"
        )));
    }
    Ok(())
}

fn dim_overflow(op: &str, name: &str, v: usize) -> EpError {
    EpError::KernelFailed(format!(
        "cuda_ep {op}: {name} ({v}) exceeds the i32 kernel bound"
    ))
}

// ───────────────────────────── LayerNormalization ──────────────────────────

/// Factory reading `axis` (default -1) and `epsilon` (default 1e-5).
pub struct LayerNormFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for LayerNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1);
        let epsilon = node
            .attr("epsilon")
            .and_then(|a| a.as_float())
            .unwrap_or(1e-5);
        Ok(Box::new(LayerNormKernel {
            axis,
            epsilon,
            runtime: self.runtime.clone(),
            warmed_signature: Mutex::new(None),
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

/// Fused f32/f16 LayerNormalization kernel.
#[derive(Debug)]
pub struct LayerNormKernel {
    axis: i64,
    epsilon: f32,
    runtime: Arc<CudaRuntime>,
    warmed_signature: Mutex<Option<NormCaptureSignature>>,
    last_call_capture_safe: AtomicBool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NormCaptureSignature {
    activation_dtype: DataType,
    scale_dtype: DataType,
    bias_dtype: Option<DataType>,
    input_shape: Vec<usize>,
    output_dtypes_and_shapes: Vec<(DataType, Vec<usize>)>,
}

impl LayerNormKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.last_call_capture_safe.store(false, Ordering::Relaxed);
        if !(2..=3).contains(&inputs.len()) || outputs.is_empty() || outputs.len() > 3 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LayerNormalization: expected 2-3 inputs (X, Scale[, B]) and \
                 1-3 outputs (Y[, Mean, InvStdDev]), got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let scale = &inputs[1];
        let bias = inputs.get(2);
        require_f16_or_f32("LayerNormalization", "X", x.dtype)?;
        if x.dtype == DataType::Float32 {
            require_f32("LayerNormalization", "Scale", scale.dtype)?;
        } else {
            require_f16_or_f32("LayerNormalization", "Scale", scale.dtype)?;
        }
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LayerNormalization: Y dtype {:?} must match X dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        require_contiguous("LayerNormalization", "X", x.is_contiguous())?;
        require_contiguous("LayerNormalization", "Scale", scale.is_contiguous())?;
        require_contiguous("LayerNormalization", "Y", outputs[0].is_contiguous())?;

        let rank = x.shape.len();
        let axis = resolve_axis("LayerNormalization", self.axis, rank)?;
        let norm_size: usize = x.shape[axis..].iter().product();
        let num_groups: usize = x.shape[..axis].iter().product();
        if norm_size == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep LayerNormalization: empty normalization axis".into(),
            ));
        }
        if scale.numel() != norm_size {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LayerNormalization: Scale has {} elements, expected {norm_size} \
                 (= prod(shape[axis..]))",
                scale.numel()
            )));
        }
        let bias_ptr = match bias {
            None => 0u64,
            Some(b) => {
                if x.dtype == DataType::Float32 {
                    require_f32("LayerNormalization", "B", b.dtype)?;
                } else {
                    require_f16_or_f32("LayerNormalization", "B", b.dtype)?;
                }
                require_contiguous("LayerNormalization", "B", b.is_contiguous())?;
                if b.numel() != norm_size {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep LayerNormalization: B has {} elements, expected {norm_size}",
                        b.numel()
                    )));
                }
                cuptr(b.data_ptr::<u8>() as *const c_void)
            }
        };
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LayerNormalization: Y shape {:?} must equal X shape {:?}",
                outputs[0].shape, x.shape
            )));
        }
        if num_groups == 0 {
            return Ok(());
        }

        // Optional Mean / InvStdDev outputs (per group). Validate dtype only when
        // present; their length is num_groups.
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let scale_ptr = cuptr(scale.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let (mean_ptr, invstd_ptr) = optional_stat_ptrs("LayerNormalization", outputs, num_groups)?;

        let (groups_u, norm_i) = (
            u32::try_from(num_groups)
                .map_err(|_| dim_overflow("LayerNormalization", "num_groups", num_groups))?,
            i32::try_from(norm_size)
                .map_err(|_| dim_overflow("LayerNormalization", "norm_size", norm_size))?,
        );
        let has_bias: i32 = i32::from(bias_ptr != 0);
        let eps = self.epsilon;
        let groups_i = groups_u_i32(groups_u);
        let signature = (num_groups == 1).then(|| NormCaptureSignature {
            activation_dtype: x.dtype,
            scale_dtype: scale.dtype,
            bias_dtype: bias.map(|bias| bias.dtype),
            input_shape: x.shape.to_vec(),
            output_dtypes_and_shapes: outputs
                .iter()
                .map(|output| (output.dtype, output.shape.to_vec()))
                .collect(),
        });
        let capturing = self.runtime.is_capturing()?;
        let mut warmed_signature = self
            .warmed_signature
            .lock()
            .expect("cuda_ep LayerNormalization capture signature poisoned");
        if capturing && warmed_signature.as_ref() != signature.as_ref() {
            return Err(EpError::KernelFailed(
                "cuda_ep LayerNormalization: dtype or shape changed during CUDA graph capture; warm the exact single-group signature before capture"
                    .into(),
            ));
        }

        let entry = if x.dtype == DataType::Float16 {
            "layernorm_f16"
        } else {
            "layernorm_f32"
        };
        let func = self
            .runtime
            .nvrtc_function(LAYERNORM_MODULE, LAYERNORM_SRC, entry)?;
        let cfg = self.runtime.reduction_launch_config(
            &func,
            groups_u,
            NORM_BLOCK,
            std::mem::size_of::<f32>() as u32,
        )?;
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        let scale_is_half = i32::from(scale.dtype == DataType::Float16);
        let bias_is_half = i32::from(bias.is_some_and(|bias| bias.dtype == DataType::Float16));
        builder
            .arg(&x_ptr)
            .arg(&scale_ptr)
            .arg(&bias_ptr)
            .arg(&y_ptr)
            .arg(&mean_ptr)
            .arg(&invstd_ptr)
            .arg(&groups_i)
            .arg(&norm_i);
        if x.dtype == DataType::Float16 {
            builder
                .arg(&scale_is_half)
                .arg(&bias_is_half)
                .arg(&has_bias)
                .arg(&eps);
        } else {
            builder.arg(&has_bias).arg(&eps);
        }
        // SAFETY: `func` is the compiled layernorm entry; the argument list and
        // ABI match its signature; every non-null pointer is a live device
        // allocation sized as validated above (X/Y: num_groups·norm_size;
        // scale/bias: norm_size; mean/invstd: num_groups).
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        if !capturing {
            *warmed_signature = signature.clone();
        }
        self.last_call_capture_safe
            .store(signature.is_some(), Ordering::Relaxed);
        Ok(())
    }
}

impl Kernel for LayerNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }
    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "LayerNormalization shape/dtype signature does not match the warmed single-group capture signature",
            )
        }
    }
}

// ─────────────────────── RMSNorm / SimplifiedLayerNorm ──────────────────────

/// Factory reading `axis` (default -1) and `epsilon` (default 1e-5).
pub struct RmsNormFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for RmsNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        if node
            .attr("stash_type")
            .is_some_and(|attribute| attribute.as_int() != Some(1))
        {
            return Err(EpError::KernelFailed(
                "RMSNormalization: stash_type must be 1 (float)".into(),
            ));
        }
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1);
        let epsilon = node
            .attr("epsilon")
            .and_then(|a| a.as_float())
            .unwrap_or(1e-5);
        Ok(Box::new(RmsNormKernel {
            axis,
            epsilon,
            runtime: self.runtime.clone(),
            warmed_signature: Mutex::new(None),
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

/// Fused f32/f16 RMSNormalization / SimplifiedLayerNormalization kernel.
#[derive(Debug)]
pub struct RmsNormKernel {
    axis: i64,
    epsilon: f32,
    runtime: Arc<CudaRuntime>,
    warmed_signature: Mutex<Option<NormCaptureSignature>>,
    last_call_capture_safe: AtomicBool,
}

impl RmsNormKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.last_call_capture_safe.store(false, Ordering::Relaxed);
        let op = "RMSNormalization";
        if inputs.len() != 2 || outputs.is_empty() || outputs.len() > 2 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 2 inputs (X, Scale) and 1-2 outputs \
                 (Y[, InvStdDev]), got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let scale = &inputs[1];
        require_f16_or_f32(op, "X", x.dtype)?;
        if x.dtype == DataType::Float32 {
            require_f32(op, "Scale", scale.dtype)?;
        } else {
            require_f16_or_f32(op, "Scale", scale.dtype)?;
        }
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: Y dtype {:?} must match X dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        require_contiguous(op, "X", x.is_contiguous())?;
        require_contiguous(op, "Scale", scale.is_contiguous())?;
        require_contiguous(op, "Y", outputs[0].is_contiguous())?;

        let rank = x.shape.len();
        let axis = resolve_axis(op, self.axis, rank)?;
        let norm_size: usize = x.shape[axis..].iter().product();
        let num_groups: usize = x.shape[..axis].iter().product();
        if norm_size == 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: empty normalization axis"
            )));
        }
        if scale.numel() != norm_size {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: Scale has {} elements, expected {norm_size}",
                scale.numel()
            )));
        }
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: Y shape {:?} must equal X shape {:?}",
                outputs[0].shape, x.shape
            )));
        }
        if num_groups == 0 {
            return Ok(());
        }

        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let scale_ptr = cuptr(scale.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        // Only one optional stat output (InvStdDev) for the simplified norm.
        let invstd_ptr = match outputs.get_mut(1) {
            None => 0u64,
            Some(t) => {
                require_f32(op, "InvStdDev", t.dtype)?;
                if t.numel() != num_groups {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: InvStdDev has {} elements, expected {num_groups}",
                        t.numel()
                    )));
                }
                cuptr(t.data_ptr_mut::<u8>() as *const c_void)
            }
        };

        let (groups_u, norm_i) = (
            u32::try_from(num_groups).map_err(|_| dim_overflow(op, "num_groups", num_groups))?,
            i32::try_from(norm_size).map_err(|_| dim_overflow(op, "norm_size", norm_size))?,
        );
        let eps = self.epsilon;
        let signature = (num_groups == 1).then(|| NormCaptureSignature {
            activation_dtype: x.dtype,
            scale_dtype: scale.dtype,
            bias_dtype: None,
            input_shape: x.shape.to_vec(),
            output_dtypes_and_shapes: outputs
                .iter()
                .map(|output| (output.dtype, output.shape.to_vec()))
                .collect(),
        });
        let capturing = self.runtime.is_capturing()?;
        let mut warmed_signature = self
            .warmed_signature
            .lock()
            .expect("cuda_ep RMSNormalization capture signature poisoned");
        if capturing && warmed_signature.as_ref() != signature.as_ref() {
            return Err(EpError::KernelFailed(
                "cuda_ep RMSNormalization: dtype or shape changed during CUDA graph capture; warm the exact single-group signature before capture"
                    .into(),
            ));
        }

        let entry = if x.dtype == DataType::Float16 {
            "rmsnorm_f16"
        } else {
            "rmsnorm_f32"
        };
        let func = self
            .runtime
            .nvrtc_function(RMSNORM_MODULE, RMSNORM_SRC, entry)?;
        let cfg = self.runtime.reduction_launch_config(
            &func,
            groups_u,
            NORM_BLOCK,
            std::mem::size_of::<f32>() as u32,
        )?;
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        let groups_i = groups_u_i32(groups_u);
        let scale_is_half = i32::from(scale.dtype == DataType::Float16);
        builder
            .arg(&x_ptr)
            .arg(&scale_ptr)
            .arg(&y_ptr)
            .arg(&invstd_ptr)
            .arg(&groups_i)
            .arg(&norm_i);
        if x.dtype == DataType::Float16 {
            builder.arg(&scale_is_half).arg(&eps);
        } else {
            builder.arg(&eps);
        }
        // SAFETY: `func` is the compiled rmsnorm entry; the argument list/ABI
        // match; pointers are live device allocations sized as validated.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        if !capturing {
            *warmed_signature = signature.clone();
        }
        self.last_call_capture_safe
            .store(signature.is_some(), Ordering::Relaxed);
        Ok(())
    }
}

impl Kernel for RmsNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }
    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "SimplifiedLayerNormalization/RMSNorm shape/dtype signature does not match the warmed single-group capture signature",
            )
        }
    }
}

// ───────────────────── SkipSimplifiedLayerNormalization ─────────────────────

/// Factory reading `epsilon` (default 1e-5) for the fused residual RMS norm.
pub struct SkipSimplifiedLayerNormFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for SkipSimplifiedLayerNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let epsilon = node
            .attr("epsilon")
            .and_then(|a| a.as_float())
            .unwrap_or(1e-5);
        Ok(Box::new(SkipSimplifiedLayerNormKernel {
            epsilon,
            runtime: self.runtime.clone(),
            metadata: Mutex::new(SkipBroadcastMetadataCache::new(self.runtime.clone())),
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

/// Fused f32 `SkipSimplifiedLayerNormalization` kernel (`com.microsoft`).
#[derive(Debug)]
pub struct SkipSimplifiedLayerNormKernel {
    epsilon: f32,
    runtime: Arc<CudaRuntime>,
    metadata: Mutex<SkipBroadcastMetadataCache>,
    last_call_capture_safe: AtomicBool,
}

#[derive(Debug)]
struct SkipBroadcastMetadataCache {
    runtime: Arc<CudaRuntime>,
    ptr: CUdeviceptr,
    input_shape: Vec<usize>,
    skip_shape: Vec<usize>,
}

impl SkipBroadcastMetadataCache {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            ptr: 0,
            input_shape: Vec::new(),
            skip_shape: Vec::new(),
        }
    }

    fn reserve(&mut self, input_shape: &[usize], skip_shape: &[usize]) -> Result<CUdeviceptr> {
        if self.ptr != 0 && self.input_shape == input_shape && self.skip_shape == skip_shape {
            return Ok(self.ptr);
        }
        if self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(
                "cuda_ep SkipSimplifiedLayerNormalization: broadcast metadata shape changed \
                 during CUDA graph capture; warm the fixed decode shape before capture"
                    .into(),
            ));
        }

        let metadata = skip_broadcast_metadata(input_shape, skip_shape);
        let metadata_bytes = u64_bytes(&metadata);
        let ptr = self.runtime.alloc_raw(metadata_bytes.len())?;
        // SAFETY: `ptr` exactly covers the metadata byte slice.
        if let Err(error) = unsafe { self.runtime.htod(metadata_bytes, ptr) } {
            // SAFETY: `ptr` is still exclusively owned and no launch used it.
            let _ = unsafe { self.runtime.free_raw(ptr) };
            return Err(error);
        }
        if self.ptr != 0 {
            // A dynamic shape change may replace metadata still referenced by
            // queued work. Fixed-shape decode always takes the cache-hit path.
            if let Err(error) = self.runtime.synchronize() {
                // SAFETY: `ptr` is still exclusively owned and has not escaped.
                let _ = unsafe { self.runtime.free_raw(ptr) };
                return Err(error);
            }
            // SAFETY: synchronization completed all prior users of `self.ptr`.
            if let Err(error) = unsafe { self.runtime.free_raw(self.ptr) } {
                // SAFETY: `ptr` is still exclusively owned and has not escaped.
                let _ = unsafe { self.runtime.free_raw(ptr) };
                return Err(error);
            }
        }
        self.ptr = ptr;
        self.input_shape.clear();
        self.input_shape.extend_from_slice(input_shape);
        self.skip_shape.clear();
        self.skip_shape.extend_from_slice(skip_shape);
        Ok(ptr)
    }
}

impl Drop for SkipBroadcastMetadataCache {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let _ = self.runtime.synchronize();
            // SAFETY: this cache exclusively owns the persistent allocation.
            let _ = unsafe { self.runtime.free_raw(self.ptr) };
            self.ptr = 0;
        }
    }
}

impl SkipSimplifiedLayerNormKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.last_call_capture_safe.store(false, Ordering::Relaxed);
        let op = "SkipSimplifiedLayerNormalization";
        if !(3..=4).contains(&inputs.len()) || outputs.is_empty() || outputs.len() > 4 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 3-4 inputs (input, skip, gamma[, bias]) and 1-4 outputs, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let input = &inputs[0];
        let skip = &inputs[1];
        let gamma = &inputs[2];
        let bias = inputs.get(3).filter(|bias| !bias.is_absent());
        require_f16_or_f32(op, "input", input.dtype)?;
        let is_half = input.dtype == DataType::Float16;
        if skip.dtype != input.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: skip dtype {:?} must match input dtype {:?}",
                skip.dtype, input.dtype
            )));
        }
        if outputs[0].dtype != input.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output dtype {:?} must match input dtype {:?}",
                outputs[0].dtype, input.dtype
            )));
        }
        if is_half {
            require_f16_or_f32(op, "gamma", gamma.dtype)?;
        } else {
            require_f32(op, "gamma", gamma.dtype)?;
        }
        require_contiguous(op, "input", input.is_contiguous())?;
        require_contiguous(op, "skip", skip.is_contiguous())?;
        require_contiguous(op, "gamma", gamma.is_contiguous())?;
        require_contiguous(op, "output", outputs[0].is_contiguous())?;
        if is_half {
            self.runtime.require_nvrtc_half_headers(op)?;
        }

        let rank = input.shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: input must have rank >= 1"
            )));
        }
        let norm_size = input.shape[rank - 1];
        let num_groups: usize = input.shape[..rank - 1].iter().product();
        if norm_size == 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: empty hidden (last) dimension"
            )));
        }
        if gamma.shape != [norm_size] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: gamma shape {:?} must equal [{norm_size}]",
                gamma.shape
            )));
        }
        let bias_ptr = optional_norm_vec_ptr(op, "bias", bias, norm_size, is_half)?;
        let broadcast =
            onnx_runtime_ir::broadcast_shapes(input.shape, skip.shape).map_err(EpError::Ir)?;
        if broadcast != input.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: skip shape {:?} is not broadcastable to input shape {:?}",
                skip.shape, input.shape
            )));
        }
        if outputs[0].shape != input.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?} must equal input shape {:?}",
                outputs[0].shape, input.shape
            )));
        }
        if num_groups == 0 {
            return Ok(());
        }

        // Optional Mean/InvStdDev stat outputs may be f16 in a half graph (they
        // are typically unused). Track their precision so the kernel narrows.
        let gamma_is_half = i32::from(gamma.dtype == DataType::Float16);
        let bias_is_half = i32::from(bias.is_some_and(|b| b.dtype == DataType::Float16));
        let (mean_ptr, invstd_ptr, stat_is_half) = if is_half {
            let mean = optional_half_stat_ptr(op, "Mean", outputs, 1, num_groups)?;
            let invstd = optional_half_stat_ptr(op, "InvStdDev", outputs, 2, num_groups)?;
            let stat_half = i32::from(
                outputs.get(1).is_some_and(|t| t.dtype == DataType::Float16)
                    || outputs.get(2).is_some_and(|t| t.dtype == DataType::Float16),
            );
            (mean, invstd, stat_half)
        } else {
            let (mean, invstd) = optional_stat_ptrs(op, outputs, num_groups)?;
            (mean, invstd, 0)
        };
        let sum_ptr = match outputs.get_mut(3) {
            None => 0u64,
            Some(t) => {
                if t.dtype != input.dtype {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: input_skip_bias_sum dtype {:?} must match input dtype {:?}",
                        t.dtype, input.dtype
                    )));
                }
                if t.shape != input.shape {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: input_skip_bias_sum shape {:?} must equal input shape {:?}",
                        t.shape, input.shape
                    )));
                }
                cuptr(t.data_ptr_mut::<u8>() as *const c_void)
            }
        };
        let (groups_u, norm_i) = (
            u32::try_from(num_groups).map_err(|_| dim_overflow(op, "num_groups", num_groups))?,
            i32::try_from(norm_size).map_err(|_| dim_overflow(op, "norm_size", norm_size))?,
        );
        let rank_i = i32::try_from(rank).map_err(|_| dim_overflow(op, "rank", rank))?;
        let has_bias = i32::from(bias_ptr != 0);
        let input_ptr = cuptr(input.data_ptr::<u8>() as *const c_void);
        let skip_ptr = cuptr(skip.data_ptr::<u8>() as *const c_void);
        let gamma_ptr = cuptr(gamma.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let mut metadata = self
            .metadata
            .lock()
            .expect("cuda_ep skip normalization metadata cache poisoned");
        let metadata_ptr = metadata.reserve(input.shape, skip.shape)?;
        let dense_skip = i32::from(skip.numel() == input.numel());
        let selection = select_skip_rmsnorm_variant(
            is_half,
            dense_skip != 0,
            norm_size,
            bias_ptr != 0,
            gamma_is_half != 0,
        );
        let variant_name = match selection.variant {
            SkipRmsnormVariant::F32 => "skip_rmsnorm_f32",
            SkipRmsnormVariant::F16Generic => "skip_rmsnorm_f16_generic",
            SkipRmsnormVariant::F16WarpHalf4 => "skip_rmsnorm_f16_warp_half4",
        };
        onnx_runtime_ep_api::record_kernel_variant!(
            variant_name,
            "SkipSimplifiedLayerNormalization hidden={norm_size}: {}",
            selection.reason
        );
        let func =
            self.runtime
                .nvrtc_function(SKIP_RMSNORM_MODULE, SKIP_RMSNORM_SRC, selection.entry)?;
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        let groups_i = groups_u_i32(groups_u);
        builder
            .arg(&input_ptr)
            .arg(&skip_ptr)
            .arg(&gamma_ptr)
            .arg(&bias_ptr)
            .arg(&y_ptr)
            .arg(&sum_ptr)
            .arg(&mean_ptr)
            .arg(&invstd_ptr)
            .arg(&metadata_ptr)
            .arg(&rank_i)
            .arg(&groups_i)
            .arg(&norm_i)
            .arg(&has_bias);
        if is_half {
            builder
                .arg(&dense_skip)
                .arg(&gamma_is_half)
                .arg(&bias_is_half)
                .arg(&stat_is_half)
                .arg(&self.epsilon);
        } else {
            builder.arg(&self.epsilon);
        }
        // SAFETY: all pointers reference validated device buffers; metadata has
        // two rank-length u64 arrays describing the output shape and skip strides.
        let cfg = if is_half {
            LaunchConfig {
                grid_dim: (groups_u, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            self.runtime.reduction_launch_config(
                &func,
                groups_u,
                NORM_BLOCK,
                std::mem::size_of::<f32>() as u32,
            )?
        };
        unsafe { builder.launch(cfg) }
            .map_err(|e| driver_err(&format!("launch {}", selection.entry), e))?;
        self.last_call_capture_safe
            .store(num_groups == 1, Ordering::Relaxed);
        Ok(())
    }
}

impl Kernel for SkipSimplifiedLayerNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }
    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "SkipSimplifiedLayerNormalization shape/dtype signature does not match the warmed single-group capture signature",
            )
        }
    }
}

// ─────────────────────────── SkipLayerNormalization ─────────────────────────

/// Factory reading `epsilon` (default 1e-5). SkipLayerNorm always normalizes the
/// last dimension (hidden size), so it takes no `axis`.
pub struct SkipLayerNormFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for SkipLayerNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let epsilon = node
            .attr("epsilon")
            .and_then(|a| a.as_float())
            .unwrap_or(1e-5);
        Ok(Box::new(SkipLayerNormKernel {
            epsilon,
            runtime: self.runtime.clone(),
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

/// Fused f32 SkipLayerNormalization kernel (`com.microsoft`).
///
/// Inputs: `input`, `skip`, `gamma`, optional `beta`, optional `bias`.
/// Outputs: `output`, optional `mean`, optional `inv_std_var`, optional
/// `input_skip_bias_sum` (positional slots 1..=3).
#[derive(Debug)]
pub struct SkipLayerNormKernel {
    epsilon: f32,
    runtime: Arc<CudaRuntime>,
    last_call_capture_safe: AtomicBool,
}

impl SkipLayerNormKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.last_call_capture_safe.store(false, Ordering::Relaxed);
        let op = "SkipLayerNormalization";
        if !(3..=5).contains(&inputs.len()) || outputs.is_empty() || outputs.len() > 4 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 3-5 inputs (input, skip, gamma[, beta][, bias]) \
                 and 1-4 outputs, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let input = &inputs[0];
        let skip = &inputs[1];
        let gamma = &inputs[2];
        let beta = inputs.get(3);
        let bias = inputs.get(4);
        require_f32(op, "input", input.dtype)?;
        require_f32(op, "skip", skip.dtype)?;
        require_f32(op, "gamma", gamma.dtype)?;
        require_f32(op, "output", outputs[0].dtype)?;
        require_contiguous(op, "input", input.is_contiguous())?;
        require_contiguous(op, "skip", skip.is_contiguous())?;
        require_contiguous(op, "gamma", gamma.is_contiguous())?;
        require_contiguous(op, "output", outputs[0].is_contiguous())?;

        let rank = input.shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: input must have rank >= 1"
            )));
        }
        if skip.shape != input.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: skip shape {:?} must equal input shape {:?}",
                skip.shape, input.shape
            )));
        }
        let norm_size = input.shape[rank - 1];
        let num_groups: usize = input.shape[..rank - 1].iter().product();
        if norm_size == 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: empty hidden (last) dimension"
            )));
        }
        if gamma.numel() != norm_size {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: gamma has {} elements, expected {norm_size} (hidden size)",
                gamma.numel()
            )));
        }
        let beta_ptr = optional_vec_ptr(op, "beta", beta, norm_size)?;
        let bias_ptr = optional_vec_ptr(op, "bias", bias, norm_size)?;
        if outputs[0].shape != input.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?} must equal input shape {:?}",
                outputs[0].shape, input.shape
            )));
        }
        if num_groups == 0 {
            return Ok(());
        }

        let input_ptr = cuptr(input.data_ptr::<u8>() as *const c_void);
        let skip_ptr = cuptr(skip.data_ptr::<u8>() as *const c_void);
        let gamma_ptr = cuptr(gamma.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        // Optional outputs: mean (slot 1), inv_std_var (slot 2) — length
        // num_groups; input_skip_bias_sum (slot 3) — length input.numel().
        let (mean_ptr, invstd_ptr) = optional_stat_ptrs(op, outputs, num_groups)?;
        let sum_ptr = match outputs.get_mut(3) {
            None => 0u64,
            Some(t) => {
                require_f32(op, "input_skip_bias_sum", t.dtype)?;
                if t.numel() != input.numel() {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: input_skip_bias_sum has {} elements, expected {}",
                        t.numel(),
                        input.numel()
                    )));
                }
                cuptr(t.data_ptr_mut::<u8>() as *const c_void)
            }
        };

        let (groups_u, norm_i) = (
            u32::try_from(num_groups).map_err(|_| dim_overflow(op, "num_groups", num_groups))?,
            i32::try_from(norm_size).map_err(|_| dim_overflow(op, "norm_size", norm_size))?,
        );
        let has_beta: i32 = i32::from(beta_ptr != 0);
        let has_bias: i32 = i32::from(bias_ptr != 0);
        let eps = self.epsilon;

        let func = self.runtime.nvrtc_function(
            SKIP_LAYERNORM_MODULE,
            SKIP_LAYERNORM_SRC,
            "skip_layernorm_f32",
        )?;
        let cfg = self.runtime.reduction_launch_config(
            &func,
            groups_u,
            NORM_BLOCK,
            std::mem::size_of::<f32>() as u32,
        )?;
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        let groups_i = groups_u_i32(groups_u);
        builder
            .arg(&input_ptr)
            .arg(&skip_ptr)
            .arg(&gamma_ptr)
            .arg(&beta_ptr)
            .arg(&bias_ptr)
            .arg(&y_ptr)
            .arg(&sum_ptr)
            .arg(&mean_ptr)
            .arg(&invstd_ptr)
            .arg(&groups_i)
            .arg(&norm_i)
            .arg(&has_beta)
            .arg(&has_bias)
            .arg(&eps);
        // SAFETY: `func` is the compiled skip-layernorm entry; argument list/ABI
        // match; each non-null pointer is a live device allocation sized as
        // validated (input/skip/output/sum: num_groups·norm_size; gamma/beta/
        // bias: norm_size; mean/invstd: num_groups).
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err("launch skip_layernorm_f32", e))?;
        self.last_call_capture_safe
            .store(num_groups == 1, Ordering::Relaxed);
        Ok(())
    }
}

impl Kernel for SkipLayerNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }
    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "SkipLayerNormalization shape/dtype signature does not match the warmed single-group capture signature",
            )
        }
    }
}

// ───────────────────────────────── helpers ─────────────────────────────────

/// The kernels take `num_groups` as a signed `int`; convert the validated `u32`.
fn groups_u_i32(groups: u32) -> i32 {
    groups as i32
}

/// Resolve the optional per-group `Mean` (output slot 1) and `InvStdDev` (slot 2)
/// device pointers, validating f32 dtype and `num_groups` length when present.
fn skip_broadcast_metadata(input: &[usize], skip: &[usize]) -> Vec<u64> {
    let mut metadata = input.iter().map(|&dim| dim as u64).collect::<Vec<_>>();
    let contiguous = onnx_runtime_ir::compute_contiguous_strides(skip);
    let leading = input.len() - skip.len();
    metadata.extend((0..input.len()).map(|axis| {
        if axis < leading || skip[axis - leading] == 1 {
            0
        } else {
            contiguous[axis - leading] as u64
        }
    }));
    metadata
}

fn u64_bytes(values: &[u64]) -> &[u8] {
    // SAFETY: u64 is plain data and the byte slice retains the input lifetime.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn optional_stat_ptrs(
    op: &str,
    outputs: &mut [TensorMut],
    num_groups: usize,
) -> Result<(CUdeviceptr, CUdeviceptr)> {
    let mean = optional_out_ptr(op, "Mean", outputs, 1, num_groups)?;
    let invstd = optional_out_ptr(op, "InvStdDev", outputs, 2, num_groups)?;
    Ok((mean, invstd))
}

fn optional_out_ptr(
    op: &str,
    name: &str,
    outputs: &mut [TensorMut],
    idx: usize,
    expect: usize,
) -> Result<CUdeviceptr> {
    match outputs.get_mut(idx) {
        None => Ok(0),
        Some(t) => {
            require_f32(op, name, t.dtype)?;
            if t.numel() != expect {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: {name} has {} elements, expected {expect}",
                    t.numel()
                )));
            }
            Ok(cuptr(t.data_ptr_mut::<u8>() as *const c_void))
        }
    }
}

/// Resolve an optional length-`expect` input vector (f32, contiguous) to a
/// device pointer, or 0 when absent.
fn optional_vec_ptr(
    op: &str,
    name: &str,
    t: Option<&TensorView>,
    expect: usize,
) -> Result<CUdeviceptr> {
    match t {
        None => Ok(0),
        Some(v) => {
            require_f32(op, name, v.dtype)?;
            require_contiguous(op, name, v.is_contiguous())?;
            if v.numel() != expect {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: {name} has {} elements, expected {expect}",
                    v.numel()
                )));
            }
            Ok(cuptr(v.data_ptr::<u8>() as *const c_void))
        }
    }
}

/// Optional length-`expect` input vector accepting either f16 or f32 (used by
/// the half normalization paths, which pass a per-tensor `*_is_half` flag).
fn optional_norm_vec_ptr(
    op: &str,
    name: &str,
    t: Option<&TensorView>,
    expect: usize,
    allow_half: bool,
) -> Result<CUdeviceptr> {
    match t {
        None => Ok(0),
        Some(v) => {
            if allow_half {
                require_f16_or_f32(op, name, v.dtype)?;
            } else {
                require_f32(op, name, v.dtype)?;
            }
            require_contiguous(op, name, v.is_contiguous())?;
            if v.numel() != expect {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: {name} has {} elements, expected {expect}",
                    v.numel()
                )));
            }
            Ok(cuptr(v.data_ptr::<u8>() as *const c_void))
        }
    }
}

/// Optional per-group stat output (Mean/InvStdDev) accepting f16 or f32. Half
/// graphs frequently declare these unused outputs in the model's activation
/// dtype; the kernel narrows the stat write to match.
fn optional_half_stat_ptr(
    op: &str,
    name: &str,
    outputs: &mut [TensorMut],
    idx: usize,
    expect: usize,
) -> Result<CUdeviceptr> {
    match outputs.get_mut(idx) {
        None => Ok(0),
        Some(t) => {
            require_f16_or_f32(op, name, t.dtype)?;
            if t.numel() != expect {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: {name} has {} elements, expected {expect}",
                    t.numel()
                )));
            }
            Ok(cuptr(t.data_ptr_mut::<u8>() as *const c_void))
        }
    }
}

#[cfg(test)]
mod tests {
    use half::f16;
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider};
    use onnx_runtime_ir::compute_contiguous_strides;

    use super::*;
    use crate::CudaExecutionProvider;

    #[test]
    fn sources_expose_their_entry_points() {
        assert!(LAYERNORM_SRC.contains("layernorm_f32"));
        assert!(LAYERNORM_SRC.contains("layernorm_f16"));
        assert!(RMSNORM_SRC.contains("rmsnorm_f32"));
        assert!(RMSNORM_SRC.contains("rmsnorm_f16"));
        assert!(SKIP_LAYERNORM_SRC.contains("skip_layernorm_f32"));
        assert!(SKIP_RMSNORM_SRC.contains("skip_rmsnorm_f32"));
        assert!(SKIP_RMSNORM_SRC.contains("skip_rmsnorm_f16"));
        assert!(SKIP_RMSNORM_SRC.contains("skip_rmsnorm_f16_warp_half4"));
    }

    fn skip_rmsnorm_residuals(hidden: usize) -> (Vec<f16>, Vec<f16>) {
        let residual = (0..hidden)
            .map(|index| {
                let input = f16::from_f32(((index * 37 % 101) as f32 - 50.0) / 31.0);
                let skip = f16::from_f32(((index * 17 % 67) as f32 - 33.0) / 47.0);
                let bias = f16::from_f32(((index * 11 % 29) as f32 - 14.0) / 113.0);
                f16::from_f32(input.to_f32() + skip.to_f32() + bias.to_f32())
            })
            .collect();
        let gamma = (0..hidden)
            .map(|index| f16::from_f32(0.75 + (index * 13 % 41) as f32 / 64.0))
            .collect();
        (residual, gamma)
    }

    fn normalize_f16(residual: &[f16], gamma: &[f16], sum_squares: f32) -> Vec<f16> {
        let inv_std = 1.0 / (sum_squares / residual.len() as f32 + 1e-5).sqrt();
        residual
            .iter()
            .zip(gamma)
            .map(|(residual, gamma)| f16::from_f32(residual.to_f32() * inv_std * gamma.to_f32()))
            .collect()
    }

    fn previous_shared_tree_skip_rmsnorm(residual: &[f16], gamma: &[f16]) -> Vec<f16> {
        let mut lanes = [0.0f32; NORM_BLOCK as usize];
        for (lane, sum) in lanes.iter_mut().enumerate() {
            for value in residual.iter().skip(lane).step_by(NORM_BLOCK as usize) {
                let value = value.to_f32();
                *sum += value * value;
            }
        }
        let mut offset = lanes.len() / 2;
        while offset > 0 {
            for lane in 0..offset {
                lanes[lane] += lanes[lane + offset];
            }
            offset /= 2;
        }
        normalize_f16(residual, gamma, lanes[0])
    }

    fn generic_warp_shuffle_skip_rmsnorm(residual: &[f16], gamma: &[f16]) -> Vec<f16> {
        let mut lanes = [0.0f32; 32];
        let pairs = residual.len() / 2;
        for (lane, sum) in lanes.iter_mut().enumerate() {
            for pair in (lane..pairs).step_by(32) {
                let first = residual[pair * 2].to_f32();
                let second = residual[pair * 2 + 1].to_f32();
                *sum += first * first;
                *sum += second * second;
            }
        }
        if residual.len() % 2 != 0 {
            let tail = residual[residual.len() - 1].to_f32();
            lanes[0] += tail * tail;
        }
        let mut offset = 16;
        while offset > 0 {
            let previous = lanes;
            for lane in 0..(32 - offset) {
                lanes[lane] += previous[lane + offset];
            }
            offset /= 2;
        }
        normalize_f16(residual, gamma, lanes[0])
    }

    fn half4_warp_skip_rmsnorm(residual: &[f16], gamma: &[f16]) -> Vec<f16> {
        assert!(
            residual
                .len()
                .is_multiple_of(SKIP_RMSNORM_WARP_HALF4_MULTIPLE)
        );
        let mut lanes = [0.0f32; 32];
        let chunks_per_lane = residual.len() / SKIP_RMSNORM_WARP_HALF4_MULTIPLE;
        for (lane, sum) in lanes.iter_mut().enumerate() {
            let mut ss0 = 0.0f32;
            let mut ss1 = 0.0f32;
            let mut ss2 = 0.0f32;
            let mut ss3 = 0.0f32;
            for item in 0..chunks_per_lane {
                let base = (lane + item * 32) * 4;
                let value0 = residual[base].to_f32();
                let value1 = residual[base + 1].to_f32();
                let value2 = residual[base + 2].to_f32();
                let value3 = residual[base + 3].to_f32();
                ss0 += value0 * value0;
                ss1 += value1 * value1;
                ss2 += value2 * value2;
                ss3 += value3 * value3;
            }
            *sum = (ss0 + ss1) + (ss2 + ss3);
        }
        let mut offset = 16;
        while offset > 0 {
            let previous = lanes;
            for lane in 0..(32 - offset) {
                lanes[lane] += previous[lane + offset];
            }
            offset /= 2;
        }
        normalize_f16(residual, gamma, lanes[0])
    }

    fn fixed_seven_half4_warp_skip_rmsnorm(residual: &[f16; 896], gamma: &[f16; 896]) -> Vec<f16> {
        let mut lanes = [0.0f32; 32];
        for (lane, sum) in lanes.iter_mut().enumerate() {
            let mut ss0 = 0.0f32;
            let mut ss1 = 0.0f32;
            let mut ss2 = 0.0f32;
            let mut ss3 = 0.0f32;
            for item in 0..7 {
                let base = (lane + item * 32) * 4;
                let value0 = residual[base].to_f32();
                let value1 = residual[base + 1].to_f32();
                let value2 = residual[base + 2].to_f32();
                let value3 = residual[base + 3].to_f32();
                ss0 += value0 * value0;
                ss1 += value1 * value1;
                ss2 += value2 * value2;
                ss3 += value3 * value3;
            }
            *sum = (ss0 + ss1) + (ss2 + ss3);
        }
        let mut offset = 16;
        while offset > 0 {
            let previous = lanes;
            for lane in 0..(32 - offset) {
                lanes[lane] += previous[lane + offset];
            }
            offset /= 2;
        }
        normalize_f16(residual, gamma, lanes[0])
    }

    #[test]
    fn warp_shuffle_skip_rmsnorm_matches_shared_tree_for_hidden_and_tail_sizes() {
        for hidden in [896, 1024, 2048, 4096, 5120] {
            let (residual, gamma) = skip_rmsnorm_residuals(hidden);
            let previous = previous_shared_tree_skip_rmsnorm(&residual, &gamma);
            let warp = half4_warp_skip_rmsnorm(&residual, &gamma);
            let max_error = previous
                .iter()
                .zip(&warp)
                .map(|(previous, warp)| (previous.to_f32() - warp.to_f32()).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 2.0e-3,
                "hidden={hidden} shared-tree/warp max fp16 error {max_error}"
            );
        }

        let hidden = 900;
        let (residual, gamma) = skip_rmsnorm_residuals(hidden);
        let previous = previous_shared_tree_skip_rmsnorm(&residual, &gamma);
        let generic = generic_warp_shuffle_skip_rmsnorm(&residual, &gamma);
        let max_error = previous
            .iter()
            .zip(&generic)
            .map(|(previous, generic)| (previous.to_f32() - generic.to_f32()).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 2.0e-3,
            "hidden={hidden} shared-tree/generic max fp16 error {max_error}"
        );
    }

    #[test]
    fn fp16_skip_rmsnorm_warp_selection_is_structural() {
        for hidden in [128, 256, 512, 896, 1024, 2048, 4096, 5120] {
            let selection = select_skip_rmsnorm_variant(true, true, hidden, false, true);
            assert_eq!(
                selection.variant,
                SkipRmsnormVariant::F16WarpHalf4,
                "hidden={hidden}: {}",
                selection.reason
            );
            assert!(selection.reason.contains("hidden%128==0"));
        }

        let tail = select_skip_rmsnorm_variant(true, true, 900, false, true);
        assert_eq!(tail.variant, SkipRmsnormVariant::F16Generic);
        assert!(tail.reason.contains("hidden%128==0"));
    }

    #[test]
    fn generalized_half4_warp_is_bit_identical_for_hidden_896() {
        let (residual, gamma) = skip_rmsnorm_residuals(896);
        let residual: [f16; 896] = residual.try_into().unwrap();
        let gamma: [f16; 896] = gamma.try_into().unwrap();
        let fixed = fixed_seven_half4_warp_skip_rmsnorm(&residual, &gamma);
        let generalized = half4_warp_skip_rmsnorm(&residual, &gamma);
        assert_eq!(
            fixed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            generalized
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    fn f16_bytes(values: &[f16]) -> &[u8] {
        // SAFETY: f16 is plain two-byte data and the byte slice retains the input lifetime.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    fn run_fp16_skip_rmsnorm_gpu(
        ep: &CudaExecutionProvider,
        hidden: usize,
    ) -> (Vec<f16>, Vec<f16>, Vec<f16>) {
        let shape = [1, hidden];
        let strides = compute_contiguous_strides(&shape);
        let gamma_shape = [hidden];
        let gamma_strides = compute_contiguous_strides(&gamma_shape);
        let input = (0..hidden)
            .map(|index| f16::from_f32(((index * 37 % 101) as f32 - 50.0) / 31.0))
            .collect::<Vec<_>>();
        let skip = (0..hidden)
            .map(|index| f16::from_f32(((index * 17 % 67) as f32 - 33.0) / 47.0))
            .collect::<Vec<_>>();
        let gamma = (0..hidden)
            .map(|index| f16::from_f32(0.75 + (index * 13 % 41) as f32 / 64.0))
            .collect::<Vec<_>>();
        let residual = input
            .iter()
            .zip(&skip)
            .map(|(input, skip)| f16::from_f32(input.to_f32() + skip.to_f32()))
            .collect::<Vec<_>>();

        let input_buffer = ep
            .allocate(hidden * std::mem::size_of::<f16>(), 256)
            .unwrap();
        let skip_buffer = ep
            .allocate(hidden * std::mem::size_of::<f16>(), 256)
            .unwrap();
        let gamma_buffer = ep
            .allocate(hidden * std::mem::size_of::<f16>(), 256)
            .unwrap();
        let mut output_buffer = ep
            .allocate(hidden * std::mem::size_of::<f16>(), 256)
            .unwrap();
        let runtime = ep.runtime();
        unsafe {
            runtime
                .htod(f16_bytes(&input), cuptr(input_buffer.as_ptr()))
                .unwrap();
            runtime
                .htod(f16_bytes(&skip), cuptr(skip_buffer.as_ptr()))
                .unwrap();
            runtime
                .htod(f16_bytes(&gamma), cuptr(gamma_buffer.as_ptr()))
                .unwrap();
        }

        {
            let inputs = [
                TensorView::new(
                    DevicePtr(input_buffer.as_ptr()),
                    DataType::Float16,
                    &shape,
                    &strides,
                    ep.device_id(),
                ),
                TensorView::new(
                    DevicePtr(skip_buffer.as_ptr()),
                    DataType::Float16,
                    &shape,
                    &strides,
                    ep.device_id(),
                ),
                TensorView::new(
                    DevicePtr(gamma_buffer.as_ptr()),
                    DataType::Float16,
                    &gamma_shape,
                    &gamma_strides,
                    ep.device_id(),
                ),
            ];
            let output = TensorMut::new(
                DevicePtrMut(output_buffer.as_mut_ptr()),
                DataType::Float16,
                &shape,
                &strides,
                ep.device_id(),
            );
            let kernel = SkipSimplifiedLayerNormKernel {
                epsilon: 1e-5,
                runtime: runtime.clone(),
                metadata: Mutex::new(SkipBroadcastMetadataCache::new(runtime.clone())),
                last_call_capture_safe: AtomicBool::new(false),
            };
            kernel.run(&inputs, &mut [output]).unwrap();
        }

        let mut output_bytes = vec![0u8; hidden * std::mem::size_of::<f16>()];
        unsafe {
            runtime
                .dtoh(&mut output_bytes, cuptr(output_buffer.as_ptr()))
                .unwrap();
        }
        let output = output_bytes
            .chunks_exact(2)
            .map(|raw| f16::from_bits(u16::from_ne_bytes(raw.try_into().unwrap())))
            .collect();
        ep.deallocate(input_buffer).unwrap();
        ep.deallocate(skip_buffer).unwrap();
        ep.deallocate(gamma_buffer).unwrap();
        ep.deallocate(output_buffer).unwrap();
        (output, residual, gamma)
    }

    fn f32_bytes(values: &[f32]) -> &[u8] {
        // SAFETY: f32 is plain-old-data; reinterpreting as bytes is sound.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    /// Run `SkipSimplifiedLayerNormalization` on the GPU with fp16 activations
    /// but an **fp32 gamma** (the shape Phi's cast-fold leaves behind), returning
    /// `(output, residual, gamma_f32)`.
    fn run_skip_rmsnorm_gpu_f32_gamma(
        ep: &CudaExecutionProvider,
        hidden: usize,
    ) -> (Vec<f16>, Vec<f16>, Vec<f32>) {
        let shape = [1, hidden];
        let strides = compute_contiguous_strides(&shape);
        let gamma_shape = [hidden];
        let gamma_strides = compute_contiguous_strides(&gamma_shape);
        let input = (0..hidden)
            .map(|index| f16::from_f32(((index * 37 % 101) as f32 - 50.0) / 31.0))
            .collect::<Vec<_>>();
        let skip = (0..hidden)
            .map(|index| f16::from_f32(((index * 17 % 67) as f32 - 33.0) / 47.0))
            .collect::<Vec<_>>();
        // fp32 gamma with sub-fp16 precision, so the full-precision multiply is
        // observable and an fp16 gamma round-trip would perturb the result.
        let gamma = (0..hidden)
            .map(|index| 0.7501 + (index % 41) as f32 * 0.012_345)
            .collect::<Vec<f32>>();
        let residual = input
            .iter()
            .zip(&skip)
            .map(|(input, skip)| f16::from_f32(input.to_f32() + skip.to_f32()))
            .collect::<Vec<_>>();

        let input_buffer = ep
            .allocate(hidden * std::mem::size_of::<f16>(), 256)
            .unwrap();
        let skip_buffer = ep
            .allocate(hidden * std::mem::size_of::<f16>(), 256)
            .unwrap();
        let gamma_buffer = ep
            .allocate(hidden * std::mem::size_of::<f32>(), 256)
            .unwrap();
        let mut output_buffer = ep
            .allocate(hidden * std::mem::size_of::<f16>(), 256)
            .unwrap();
        let runtime = ep.runtime();
        unsafe {
            runtime
                .htod(f16_bytes(&input), cuptr(input_buffer.as_ptr()))
                .unwrap();
            runtime
                .htod(f16_bytes(&skip), cuptr(skip_buffer.as_ptr()))
                .unwrap();
            runtime
                .htod(f32_bytes(&gamma), cuptr(gamma_buffer.as_ptr()))
                .unwrap();
        }
        {
            let inputs = [
                TensorView::new(
                    DevicePtr(input_buffer.as_ptr()),
                    DataType::Float16,
                    &shape,
                    &strides,
                    ep.device_id(),
                ),
                TensorView::new(
                    DevicePtr(skip_buffer.as_ptr()),
                    DataType::Float16,
                    &shape,
                    &strides,
                    ep.device_id(),
                ),
                TensorView::new(
                    DevicePtr(gamma_buffer.as_ptr()),
                    DataType::Float32,
                    &gamma_shape,
                    &gamma_strides,
                    ep.device_id(),
                ),
            ];
            let output = TensorMut::new(
                DevicePtrMut(output_buffer.as_mut_ptr()),
                DataType::Float16,
                &shape,
                &strides,
                ep.device_id(),
            );
            let kernel = SkipSimplifiedLayerNormKernel {
                epsilon: 1e-5,
                runtime: runtime.clone(),
                metadata: Mutex::new(SkipBroadcastMetadataCache::new(runtime.clone())),
                last_call_capture_safe: AtomicBool::new(false),
            };
            kernel.run(&inputs, &mut [output]).unwrap();
        }
        let mut output_bytes = vec![0u8; hidden * std::mem::size_of::<f16>()];
        unsafe {
            runtime
                .dtoh(&mut output_bytes, cuptr(output_buffer.as_ptr()))
                .unwrap();
        }
        let output = output_bytes
            .chunks_exact(2)
            .map(|raw| f16::from_bits(u16::from_ne_bytes(raw.try_into().unwrap())))
            .collect();
        ep.deallocate(input_buffer).unwrap();
        ep.deallocate(skip_buffer).unwrap();
        ep.deallocate(gamma_buffer).unwrap();
        ep.deallocate(output_buffer).unwrap();
        (output, residual, gamma)
    }

    /// warp_half4 reduction order (fp32, four accumulators) with an fp32 gamma
    /// applied at full precision, matching the widened kernel.
    fn half4_warp_skip_rmsnorm_f32_gamma(residual: &[f16], gamma: &[f32]) -> Vec<f16> {
        let mut lanes = [0.0f32; 32];
        let chunks_per_lane = residual.len() / SKIP_RMSNORM_WARP_HALF4_MULTIPLE;
        for (lane, sum) in lanes.iter_mut().enumerate() {
            let (mut ss0, mut ss1, mut ss2, mut ss3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for item in 0..chunks_per_lane {
                let base = (lane + item * 32) * 4;
                let v0 = residual[base].to_f32();
                let v1 = residual[base + 1].to_f32();
                let v2 = residual[base + 2].to_f32();
                let v3 = residual[base + 3].to_f32();
                ss0 += v0 * v0;
                ss1 += v1 * v1;
                ss2 += v2 * v2;
                ss3 += v3 * v3;
            }
            *sum = (ss0 + ss1) + (ss2 + ss3);
        }
        let mut offset = 16;
        while offset > 0 {
            let previous = lanes;
            for lane in 0..(32 - offset) {
                lanes[lane] += previous[lane + offset];
            }
            offset /= 2;
        }
        let inv_std = 1.0 / (lanes[0] / residual.len() as f32 + 1e-5).sqrt();
        residual
            .iter()
            .zip(gamma)
            .map(|(r, g)| f16::from_f32(r.to_f32() * inv_std * g))
            .collect()
    }

    /// Same reduction but accumulating the sum-of-squares in fp16 (the broken
    /// contract). Used only as a mutation guard: the real kernel must diverge
    /// from this.
    fn f16_accumulation_skip_rmsnorm_f32_gamma(residual: &[f16], gamma: &[f32]) -> Vec<f16> {
        let mut ss = f16::from_f32(0.0);
        for r in residual {
            ss = f16::from_f32(ss.to_f32() + (r.to_f32() * r.to_f32()));
        }
        let inv_std = 1.0 / (ss.to_f32() / residual.len() as f32 + 1e-5).sqrt();
        residual
            .iter()
            .zip(gamma)
            .map(|(r, g)| f16::from_f32(r.to_f32() * inv_std * g))
            .collect()
    }

    #[test]
    fn f32_gamma_warp_selection_is_structural_and_gated() {
        // fp32 gamma now qualifies for the vectorized warp path (default on).
        for hidden in [128usize, 3072, 4096] {
            let sel = select_skip_rmsnorm_variant(true, true, hidden, false, false);
            assert_eq!(
                sel.variant,
                SkipRmsnormVariant::F16WarpHalf4,
                "hidden={hidden} fp32-gamma should take warp_half4"
            );
            assert!(sel.reason.contains("gamma=fp32"));
        }
        // fp16 gamma is unchanged.
        let half = select_skip_rmsnorm_variant(true, true, 3072, false, true);
        assert_eq!(half.variant, SkipRmsnormVariant::F16WarpHalf4);
        assert!(half.reason.contains("gamma=fp16"));
    }

    #[test]
    fn fp32_gamma_gpu_skip_rmsnorm_matches_warp_reference_at_phi_and_qwen_dims() {
        let ep = match CudaExecutionProvider::new_default() {
            Ok(ep) => ep,
            Err(error) => {
                eprintln!("skip: no CUDA GPU/runtime available ({error})");
                return;
            }
        };
        // 128 = Qwen-class small warp; 3072 = Phi-4-mini hidden (both %128==0).
        for hidden in [128usize, 3072] {
            let (output, residual, gamma) = run_skip_rmsnorm_gpu_f32_gamma(&ep, hidden);
            let reference = half4_warp_skip_rmsnorm_f32_gamma(&residual, &gamma);
            let max_error = output
                .iter()
                .zip(&reference)
                .map(|(got, want)| (got.to_f32() - want.to_f32()).abs())
                .fold(0.0f32, f32::max);
            // fp32-accum + fp32-gamma path is ULP-tight to the reference.
            let parity_tol = 1.0e-3f32;
            assert!(
                max_error <= parity_tol,
                "hidden={hidden} fp32-gamma warp GPU max error {max_error}"
            );

            // Mutation guard: a kernel that accumulated the sum-of-squares in
            // fp16 would exceed the parity bound above, so this test would catch
            // a broken accumulation dtype (proving the fp32 contract is real).
            let broken = f16_accumulation_skip_rmsnorm_f32_gamma(&residual, &gamma);
            let broken_error = reference
                .iter()
                .zip(&broken)
                .map(|(want, bad)| (want.to_f32() - bad.to_f32()).abs())
                .fold(0.0f32, f32::max);
            assert!(
                broken_error > parity_tol,
                "hidden={hidden} fp16-accumulation guard too weak ({broken_error}); \
                 test cannot detect a broken accumulation dtype"
            );
        }
    }

    #[test]
    fn fp16_skip_rmsnorm_gpu_is_generic_across_structural_hidden_sizes() {
        let ep = match CudaExecutionProvider::new_default() {
            Ok(ep) => ep,
            Err(error) => {
                eprintln!("skip: no CUDA GPU/runtime available ({error})");
                return;
            }
        };
        for hidden in [896, 1024, 2048, 4096, 5120] {
            let selection = select_skip_rmsnorm_variant(true, true, hidden, false, true);
            assert_eq!(selection.variant, SkipRmsnormVariant::F16WarpHalf4);
            let (output, residual, gamma) = run_fp16_skip_rmsnorm_gpu(&ep, hidden);
            let reference = previous_shared_tree_skip_rmsnorm(&residual, &gamma);
            let max_error = output
                .iter()
                .zip(&reference)
                .map(|(output, reference)| (output.to_f32() - reference.to_f32()).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 2.0e-3,
                "hidden={hidden} GPU half4/shared-tree max fp16 error {max_error}"
            );
        }

        let hidden = 900;
        let selection = select_skip_rmsnorm_variant(true, true, hidden, false, true);
        assert_eq!(selection.variant, SkipRmsnormVariant::F16Generic);
        let (output, residual, gamma) = run_fp16_skip_rmsnorm_gpu(&ep, hidden);
        let reference = previous_shared_tree_skip_rmsnorm(&residual, &gamma);
        let max_error = output
            .iter()
            .zip(&reference)
            .map(|(output, reference)| (output.to_f32() - reference.to_f32()).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 2.0e-3,
            "hidden={hidden} GPU generic/shared-tree max fp16 error {max_error}"
        );
    }

    #[test]
    fn fp16_skip_rmsnorm_source_uses_one_warp_without_shared_reduction() {
        let start = SKIP_RMSNORM_SRC
            .find("extern \"C\" __global__ void skip_rmsnorm_f16")
            .unwrap();
        let source = &SKIP_RMSNORM_SRC[start..];
        assert!(source.contains("__half2"));
        assert!(source.contains("__shfl_down_sync"));
        assert!(!source.contains("extern __shared__"));
        assert!(!source.contains("__syncthreads"));
    }

    #[test]
    fn require_f32_names_op_and_dtype() {
        let e = require_f32("LayerNormalization", "Scale", DataType::Float16).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("LayerNormalization"), "{msg}");
        assert!(msg.contains("Float16"), "{msg}");
    }

    #[test]
    fn require_contiguous_is_actionable() {
        let e = require_contiguous("RMSNormalization", "X", false).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("non-contiguous"), "{msg}");
        assert!(msg.contains("materialise"), "{msg}");
    }

    #[test]
    fn norm_group_split_matches_axis() {
        // shape [4, 8], axis -1 → 4 groups of 8; last-dim norm.
        let shape = [4usize, 8];
        let axis = resolve_axis("LayerNormalization", -1, shape.len()).unwrap();
        let norm_size: usize = shape[axis..].iter().product();
        let groups: usize = shape[..axis].iter().product();
        assert_eq!((groups, norm_size), (4, 8));
    }
}

//! `com.microsoft::MatMulNBits`: decode-specialized packed INT4/INT8 GEMV plus
//! the block-wise dequantization and f32 cuBLASLt GEMM fallback used for prefill.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node};

use crate::blas::{self, GemmDtype, GemmEpilogue, GemmEpilogueKind, GemmParams, WORKSPACE_BYTES};
use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr, raw_ptr};

const DEQUANT_MODULE: &str = "matmul_nbits_dequant_f32";
const DEQUANT_ENTRY: &str = "matmul_nbits_dequant_f32";
const GEMV_MODULE: &str = "matmul_nbits_gemv";
const GEMV_F32_ENTRY: &str = "matmul_nbits_gemv_f32";
const GEMV_INT8_F32_ENTRY: &str = "matmul_nbits_gemv_int8_f32";
const QUANTIZE_ACCURACY4_ENTRY: &str = "matmul_nbits_quantize_accuracy4_block32";
const GEMV_ACCURACY4_ENTRY: &str = "matmul_nbits_gemv_accuracy4_block32";
const ACCURACY4_MODULE: &str = "matmul_nbits_accuracy4";
const ACCURACY4_ENTRY: &str = "matmul_nbits_accuracy4";
const BLOCK_THREADS: u32 = 256;
const GEMV_ACCURACY4_THREADS: u32 = 256;
const GEMV_ACCURACY4_COLUMNS_PER_BLOCK: usize = 8;
const GEMV_ACCURACY4_SHARED_BYTES: u32 = 32 * 32;
const GEMV_F16_MODULE: &str = "matmul_nbits_gemv_f16";
const GEMV_F16_ENTRY: &str = "matmul_nbits_gemv_f16";
const GEMV_INT8_F16_ENTRY: &str = "matmul_nbits_gemv_int8_f16";
/// Split-K specialization of [`GEMV_INT8_F16_ENTRY`]. ncu showed the single-warp
/// standalone int8 GEMV grid-starved on Phi's int8 down projection (grid 384,
/// ~0.48 waves/SM, ~35% occupancy, 28µs) — unlike the fused RMSNorm-prologue int8
/// kernel, it has no serial prologue, so partitioning the K reduction across
/// [`GEMV_INT8_F16_SPLITK`] cooperating warps per output column multiplies the
/// grid to fill the SMs and directly attacks the memory-latency bound. The K-slice
/// partials are summed in fp32 (a new block-sum association vs the single-warp
/// kernel), so this path is near-equal — not byte-identical — to the plain entry;
/// asymmetric-zp parity is validated against a dequant reference to tolerance. Only
/// launched when `K % 256 == 0` (whole 256-wide steps, no divergent tail) and the
/// weights carry zero points; symmetric int8 keeps the byte-identical single-warp
/// kernel.
const GEMV_INT8_F16_SPLITK_ENTRY: &str = "matmul_nbits_gemv_int8_f16_splitk";
/// Warps cooperating per output column in the split-K standalone int8 GEMV. Must
/// match `K_SPLIT` in `matmul_nbits_gemv_int8_f16_splitk`. A block keeps its
/// `blockDim.x / 32` warps but now covers `warps / K_SPLIT` columns, so the launch
/// grid grows by this factor.
const GEMV_INT8_F16_SPLITK: usize = 2;
const GEMM_F16_ENTRY: &str = "matmul_nbits_gemm_f16";
/// Model-agnostic fp16 int4 decode GEMV for any power-of-two `block_size` (16,
/// 64, 128, 256, ...). The tuned [`GEMV_F16_ENTRY`]/[`GEMV_F16_SCALES_F16_ENTRY`]
/// kernels bake in the block-32 four-lane/eight-block warp layout; this entry
/// instead derives the scale/zero-point block index from the actual
/// `block_size` (`block = depth / block_size`) so a lane's contiguous 8-element
/// chunk maps to the correct block regardless of block width.
const GEMV_F16_GENERAL_BS_ENTRY: &str = "matmul_nbits_gemv_f16_general_bs";
/// Model-agnostic fp16 int4/int8 prefill GEMM for any power-of-two `block_size`.
/// Mirrors [`GEMM_F16_ENTRY`] but walks `K` in fixed 32-wide tiles and computes
/// `block = depth / block_size`, decoupling the tile width from the block width.
const GEMM_F16_GENERAL_BS_ENTRY: &str = "matmul_nbits_gemm_f16_general_bs";
const GEMV_F16_SCALES_F16_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16";
/// Asymmetric-zero-point specialization of [`GEMV_F16_SCALES_F16_ENTRY`]. The
/// symmetric entry above is compiled with `HasZp == false`, which
/// dead-code-eliminates the per-block zero-point global load and folds the
/// subtrahend to the constant fp16 `8.0` (byte-identical to the pre-zero-point
/// path). Weights that actually carry zero points launch this `_zp` entry so
/// only the asymmetric path pays for the extra per-block load.
const GEMV_F16_SCALES_F16_ZP_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_zp";
/// Split-K specialization of [`GEMV_F16_SCALES_F16_ZP_ENTRY`] for the standalone
/// asymmetric int4 GEMV. ncu showed the plain `_zp` kernel is grid-starved
/// (~0.36 waves/SM, ~64% of the SMs idle) and memory-latency bound: partitioning
/// the K reduction across [`GEMV_F16_SCALES_F16_ZP_SPLITK`] cooperating warps per
/// output column multiplies the grid, filling the machine so the extra in-flight
/// loads hide the Long-Scoreboard latency. The K-slice partials are summed in
/// fp32 (a new block-sum association vs the single-warp kernel), so this path is
/// near-equal — not byte-identical — to the plain `_zp` kernel; the asymmetric-zp
/// parity test tracks a dequant reference to tolerance. Only launched when
/// `K % 256 == 0` (whole 256-wide steps, no divergent tail); other K fall back to
/// the plain `_zp` entry.
const GEMV_F16_SCALES_F16_ZP_SPLITK_ENTRY: &str =
    "matmul_nbits_gemv_f16_scales_f16_zp_splitk";
/// Warps cooperating per output column in the split-K asymmetric int4 GEMV. Must
/// match `K_SPLIT` in `matmul_nbits_gemv_f16_scales_f16_splitk`. A block keeps
/// its `blockDim.x / 32` warps but now covers `warps / K_SPLIT` columns, so the
/// launch grid grows by this factor.
const GEMV_F16_SCALES_F16_ZP_SPLITK: usize = 2;
/// General fp16/fp16-scales GEMV with a fused RMS-normalization prologue (see
/// [`crate::optimizer::CudaSkipRmsNormMatMulFusion`]). It normalizes the input
/// activation in-kernel — byte-identically to `skip_rmsnorm_f16_warp_half4` —
/// before the standard `scales_f16` int4 dot, folding a
/// `SkipSimplifiedLayerNormalization` normalization into the following GEMV.
const GEMV_F16_SCALES_F16_RMSNORM_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_rmsnorm";
/// Asymmetric-zero-point specialization of [`GEMV_F16_SCALES_F16_RMSNORM_ENTRY`]
/// (see [`GEMV_F16_SCALES_F16_ZP_ENTRY`] for the `HasZp` specialization scheme).
const GEMV_F16_SCALES_F16_RMSNORM_ZP_ENTRY: &str =
    "matmul_nbits_gemv_f16_scales_f16_rmsnorm_zp";
/// INT8 sibling of [`GEMV_F16_SCALES_F16_RMSNORM_ENTRY`]. Shares the RMS
/// reduction and normalized-activation staging bit-for-bit and swaps in the
/// block-32 int8 dequant dot, fusing a `SkipSimplifiedLayerNormalization` into
/// the following int8 GEMV (e.g. Phi's int8 qkv projection). Compiled in the
/// same symmetric/`_zp` `HasZp` pair as the int4 sibling so a future
/// symmetric-int8 model keeps the constant-subtrahend (no per-block load) path.
const GEMV_INT8_F16_SCALES_F16_RMSNORM_ENTRY: &str =
    "matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm";
/// Asymmetric-zero-point specialization of
/// [`GEMV_INT8_F16_SCALES_F16_RMSNORM_ENTRY`] (Phi int8 qkv carries zero points).
const GEMV_INT8_F16_SCALES_F16_RMSNORM_ZP_ENTRY: &str =
    "matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm_zp";
/// Standalone RMS-normalization prologue used by the fused GEMV's M>1 prefill
/// path (see [`MatMulNBitsKernel::launch_rmsnorm_prefill`]).
const RMSNORM_PREFILL_ENTRY: &str = "matmul_nbits_rmsnorm_f16_warp_half4";
/// One warp (32 lanes) normalizes one token row in the prefill prologue.
const RMSNORM_PREFILL_THREADS: u32 = 32;
const GEMV_F16_DOWN_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_down";
const GEMM_F16_TILE: usize = 16;
const GEMV_F16_SMALL_THREADS: u32 = 64;
const GEMV_F16_LARGE_THREADS: u32 = 256;
const GEMV_F16_SMALL_N_MAX: usize = 1152;
/// Block-quantization size the down-projection tiling assumes. It stages the
/// activation as `K/8` permuted half8 vectors and indexes them as 4 `uint4` per
/// K-block (`block*4 .. block*4+3`), i.e. exactly 32 activation elements per
/// block, and it has **no** partial-block tail. So the variant is only correct
/// when `block_size == 32` and `K` is a whole multiple of 32.
const GEMV_F16_DOWN_BLOCK_SIZE: usize = 32;
const GEMV_F16_DOWN_THREADS: u32 = 256;
const GEMV_F16_DOWN_COLUMNS_PER_BLOCK: usize = 8;
const GATE_UP_SWIGLU_ENTRY: &str = "matmul_nbits_gemv_f16_gate_up_swiglu";
const GATE_UP_SWIGLU_RMSNORM_ENTRY: &str = "matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm";
/// Asymmetric-zero-point specializations of the paired gate/up SwiGLU entries
/// (see [`GEMV_F16_SCALES_F16_ZP_ENTRY`] for the `HasZp` specialization scheme).
const GATE_UP_SWIGLU_ZP_ENTRY: &str = "matmul_nbits_gemv_f16_gate_up_swiglu_zp";
const GATE_UP_SWIGLU_RMSNORM_ZP_ENTRY: &str =
    "matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_zp";
const GATE_UP_SWIGLU_THREADS: u32 = 256;

const DEQUANT_SRC: &str = r#"
extern "C" __global__ void matmul_nbits_dequant_f32(
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const int* group_indices,
    float* weight_kn,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int bits)
{
    const long total = (long)k * n;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long)gridDim.x * blockDim.x) {
        const int depth = (int)(idx / n);
        const int output = (int)(idx % n);
        const int block = depth / block_size;
        const int within = depth - block * block_size;
        const int bit_offset = within * bits;
        const unsigned char byte =
            packed[((long)output * k_blocks + block) * blob_size + bit_offset / 8];
        const int mask = bits == 8 ? 255 : ((1 << bits) - 1);
        const int quantized = (byte >> (bit_offset & 7)) & mask;
        const int group = group_indices ? group_indices[depth] : block;
        if (group < 0 || group >= k_blocks) {
            weight_kn[idx] = 0.0f;
            continue;
        }
        int zero_point = 1 << (bits - 1);
        if (zero_points) {
            const int zp_bit_offset = group * bits;
            const unsigned char zp =
                zero_points[(long)output * zp_row_bytes + zp_bit_offset / 8];
            zero_point = (zp >> (zp_bit_offset & 7)) & mask;
        }
        weight_kn[idx] =
            ((float)quantized - (float)zero_point) * scales[(long)output * k_blocks + group];
    }
}
"#;

const GEMV_SRC: &str = r#"
__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value)
{
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < ((blockDim.x + 31) >> 5) ? warp_sums[lane] : 0.0f;
    return warp == 0 ? warp_sum(value) : 0.0f;
}

extern "C" __global__ void matmul_nbits_gemv_f32(
    const float* activation,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    const int column = (int)blockIdx.x;
    if (column >= n) {
        return;
    }

    float value = 0.0f;
    for (int depth = (int)threadIdx.x; depth < k; depth += (int)blockDim.x) {
        const int block = depth / block_size;
        const int within = depth - block * block_size;
        const unsigned char byte =
            packed[((long)column * k_blocks + block) * blob_size + within / 2];
        const int quantized = (within & 1) ? (byte >> 4) : (byte & 15);
        int zero_point = 8;
        if (zero_points) {
            const unsigned char zp =
                zero_points[(long)column * zp_row_bytes + block / 2];
            zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
        }
        value += activation[depth] * ((float)quantized - (float)zero_point)
            * scales[(long)column * k_blocks + block];
    }

    value = block_sum(value);
    if (threadIdx.x == 0) {
        output[column] = value + (bias ? bias[column] : 0.0f);
    }
}

extern "C" __global__ void matmul_nbits_gemv_int8_f32(
    const float* activation,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int k_blocks)
{
    const int column = (int)blockIdx.x;
    if (column >= n) {
        return;
    }

    float value = 0.0f;
    for (int depth = (int)threadIdx.x; depth < k; depth += (int)blockDim.x) {
        const int block = depth >> 5;
        const int within = depth & 31;
        const int quantized =
            (int)packed[((long)column * k_blocks + block) * 32 + within];
        const int zero_point =
            zero_points ? (int)zero_points[(long)column * k_blocks + block] : 128;
        value += activation[depth] * ((float)quantized - (float)zero_point)
            * scales[(long)column * k_blocks + block];
    }

    value = block_sum(value);
    if (threadIdx.x == 0) {
        output[column] = value + (bias ? bias[column] : 0.0f);
    }
}

extern "C" __global__ void matmul_nbits_quantize_accuracy4_block32(
    const float* activation,
    signed char* quantized_activation,
    float* activation_scale_out,
    const int k,
    const int padded_k)
{
    const int lane = (int)threadIdx.x;
    float max_abs = 0.0f;
    for (int depth = lane; depth < k; depth += 32) {
        max_abs = fmaxf(max_abs, fabsf(activation[depth]));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_abs = fmaxf(max_abs,
            __shfl_down_sync(0xffffffffu, max_abs, offset));
    }
    max_abs = __shfl_sync(0xffffffffu, max_abs, 0);

    const float activation_scale = max_abs == 0.0f ? 0.0f : max_abs / 127.0f;
    const float inverse_scale =
        activation_scale == 0.0f ? 0.0f : 1.0f / activation_scale;
    if (lane == 0) {
        *activation_scale_out = activation_scale;
    }
    for (int depth = lane; depth < padded_k; depth += 32) {
        int quantized = 0;
        if (depth < k && activation_scale != 0.0f) {
            quantized = (int)roundf(fminf(127.0f, fmaxf(-127.0f,
                activation[depth] * inverse_scale)));
        }
        quantized_activation[depth] = (signed char)quantized;
    }
}

__device__ __forceinline__ int unpack_int4x4(unsigned int packed, int offset)
{
    const int w0 = (int)((packed >> (offset + 0)) & 15u) - 8;
    const int w1 = (int)((packed >> (offset + 4)) & 15u) - 8;
    const int w2 = (int)((packed >> (offset + 8)) & 15u) - 8;
    const int w3 = (int)((packed >> (offset + 12)) & 15u) - 8;
    return (w0 & 255) | ((w1 & 255) << 8) | ((w2 & 255) << 16)
        | ((w3 & 255) << 24);
}

extern "C" __global__ void matmul_nbits_gemv_accuracy4_block32(
    const signed char* quantized_activation,
    const float* activation_scale_ptr,
    const unsigned char* packed,
    const float* scales,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int k_blocks)
{
    extern __shared__ signed char activation_tile[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int column = (int)blockIdx.x * 8 + warp;
    const float activation_scale = *activation_scale_ptr;
    if (activation_scale == 0.0f) {
        if (lane == 0 && column < n) {
            output[column] = bias ? bias[column] : 0.0f;
        }
        return;
    }

    float value = 0.0f;
    for (int tile_block = 0; tile_block < k_blocks; tile_block += 32) {
        const int tile_blocks = min(32, k_blocks - tile_block);
        const int tile_depths = tile_blocks * 32;
        for (int depth = tid; depth < tile_depths; depth += (int)blockDim.x) {
            activation_tile[depth] =
                quantized_activation[tile_block * 32 + depth];
        }
        __syncthreads();

        const int block = tile_block + lane;
        if (column < n && block < k_blocks) {
            const long packed_start = ((long)column * k_blocks + block) * 16;
            const uint4 packed_weights =
                *reinterpret_cast<const uint4*>(packed + packed_start);
            const unsigned int words[4] = {
                packed_weights.x, packed_weights.y, packed_weights.z, packed_weights.w
            };
            const signed char* activation_block = activation_tile + lane * 32;
            int dot = 0;
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                const int activation0 =
                    *reinterpret_cast<const int*>(activation_block + word * 8);
                const int activation1 =
                    *reinterpret_cast<const int*>(activation_block + word * 8 + 4);
                dot = __dp4a(activation0, unpack_int4x4(words[word], 0), dot);
                dot = __dp4a(activation1, unpack_int4x4(words[word], 16), dot);
            }
            const float scaled =
                __fmul_rn((float)dot, scales[(long)column * k_blocks + block]);
            value = __fadd_rn(value, scaled);
        }
        __syncthreads();
    }

    value = __fmul_rn(warp_sum(value), activation_scale);
    if (lane == 0 && column < n) {
        output[column] = bias ? __fadd_rn(value, bias[column]) : value;
    }
}
"#;

const ACCURACY4_SRC: &str = r#"
extern "C" __global__ void matmul_nbits_accuracy4(
    const float* a,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* y,
    const int m,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    const long total = (long)m * n;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long)gridDim.x * blockDim.x) {
        const int row = (int)(idx / n);
        const int output = (int)(idx % n);
        const float* activation = a + (long)row * k;

        float max_abs = 0.0f;
        for (int depth = 0; depth < k; ++depth) {
            max_abs = fmaxf(max_abs, fabsf(activation[depth]));
        }
        if (max_abs == 0.0f) {
            y[idx] = bias ? bias[output] : 0.0f;
            continue;
        }

        const float activation_scale = max_abs / 127.0f;
        const float inverse_scale = 1.0f / activation_scale;
        float value = 0.0f;
        for (int block = 0; block < k_blocks; ++block) {
            int dot = 0;
            const int begin = block * block_size;
            const int end = min(begin + block_size, k);
            int zero_point = 8;
            if (zero_points) {
                const unsigned char zp =
                    zero_points[(long)output * zp_row_bytes + block / 2];
                zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
            }
            for (int depth = begin; depth < end; ++depth) {
                int quantized_activation =
                    (int)roundf(fminf(127.0f, fmaxf(-127.0f,
                        activation[depth] * inverse_scale)));
                const int within = depth - begin;
                const unsigned char byte =
                    packed[((long)output * k_blocks + block) * blob_size + within / 2];
                const int quantized_weight =
                    (within & 1) ? (byte >> 4) : (byte & 15);
                dot += quantized_activation * (quantized_weight - zero_point);
            }
            if (m == 1 && block_size == 32 && !zero_points) {
                const float scaled =
                    __fmul_rn((float)dot, scales[(long)output * k_blocks + block]);
                value = __fadd_rn(value, scaled);
            } else {
                const float combined_scale = __fmul_rn(
                    activation_scale,
                    scales[(long)output * k_blocks + block]);
                value = __fadd_rn(value, __fmul_rn((float)dot, combined_scale));
            }
        }
        if (m == 1 && block_size == 32 && !zero_points) {
            value = __fmul_rn(value, activation_scale);
        }
        y[idx] = bias ? __fadd_rn(value, bias[output]) : value;
    }
}
"#;

// Direct fp16-activation x packed-int4 GEMV (decode M=1). Unlike the
// accuracy_level=4 path this performs NO separate int8 activation-quantization
// pass. Packed nibbles are converted in registers and multiplied by fp16
// activations directly. The common fp16-scale path uses half2 accumulation,
// matching the storage precision before an fp32 warp reduction; f32 scales use
// fp32 block accumulation. Both paths round once more to fp16 on write.
const GEMV_F16_SRC: &str = r#"
#include <cuda_fp16.h>

__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Fp16 GEMV bias epilogue. The fp32 accumulator is always rounded to fp16 for
// the base output. When a bias is present:
//   * `bias_post_round == 0` (native MatMulNBits bias): add in fp32 and round
//     once — `fp16(acc + bias)` — matching an ORT-style fused epilogue.
//   * `bias_post_round != 0` (a folded standalone `Add`): round the accumulator
//     to fp16 first, then add the fp16 bias with a second fp16 round —
//     `fp16(fp16(acc) + bias)` — reproducing the original two-op path so greedy
//     tokens stay byte-identical.
__device__ __forceinline__ __half fold_bias_f16(
    const float value,
    const __half* __restrict__ bias,
    const int column,
    const int bias_post_round)
{
    const __half rounded = __float2half(value);
    if (!bias) {
        return rounded;
    }
    const float b = __half2float(bias[column]);
    if (bias_post_round) {
        return __float2half(__half2float(rounded) + b);
    }
    return __float2half(value + b);
}

// One warp per output column. Block-32 INT8 stores one unsigned quantized
// weight byte per K element and one optional uint8 zero point per block.
extern "C" __global__ void matmul_nbits_gemv_int8_f16(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int scales_fp16,
    const int bias_post_round)
{
    // Mirrors the int4 `matmul_nbits_gemv_f16` work split: four adjacent lanes
    // cooperate on one block-32 column, eight blocks are consumed per warp step.
    // Each lane issues one aligned 8-byte packed-int8 load (uint2) and one 16-byte
    // activation load (uint4), then a four-lane shuffle reduction reconstructs the
    // block dot product before its scale is applied. This replaces the previous
    // one-byte-per-lane scalar walk that only advanced 32 K per warp step.
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;

    float value = 0.0f;
    if (column < n) {
        const int quarter = lane & 3;
        for (int block_base = 0; block_base < k_blocks; block_base += 8) {
            const int block = block_base + (lane >> 2);
            float block_partial = 0.0f;
            if (block < k_blocks) {
                const int zero_point =
                    zero_points ? (int)zero_points[(long)column * k_blocks + block] : 128;
                const int depth = block * 32 + quarter * 8;
                const long packed_start =
                    ((long)column * k_blocks + block) * 32 + quarter * 8;
                if (depth + 8 <= k) {
                    const uint2 packed_word =
                        *reinterpret_cast<const uint2*>(packed + packed_start);
                    const unsigned char* bytes =
                        reinterpret_cast<const unsigned char*>(&packed_word);
                    const uint4 act = *reinterpret_cast<const uint4*>(activation + depth);
                    const __half* acth = reinterpret_cast<const __half*>(&act);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        block_partial += ((float)(int)bytes[i] - (float)zero_point)
                            * __half2float(acth[i]);
                    }
                } else if (depth < k) {
                    const int valid = min(8, k - depth);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid) {
                            const int quantized = (int)packed[packed_start + i];
                            block_partial += ((float)quantized - (float)zero_point)
                                * __half2float(activation[depth + i]);
                        }
                    }
                }
            }
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 2, 4);
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 1, 4);
            if (quarter == 0 && block < k_blocks) {
                const float scale = scales_fp16
                    ? __half2float(reinterpret_cast<const __half*>(scales_raw)
                        [(long)column * k_blocks + block])
                    : reinterpret_cast<const float*>(scales_raw)
                        [(long)column * k_blocks + block];
                value += block_partial * scale;
            }
        }
    }
    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

// Split-K standalone int8 GEMV: K_SPLIT warps cooperate on one output column,
// each reducing a strided subset of the 8-block (256-wide) K steps, then summing
// their fp32 partials through shared memory. The launch grid is K_SPLIT x larger
// than the single-warp kernel, which fills the SMs on the grid-starved
// (~0.48 waves/SM) Phi int8 down-projection decode GEMV. This kernel has no
// serial prologue, so the added grid parallelism directly hides the
// Long-Scoreboard latency (unlike the fused RMSNorm-prologue int8 kernel, whose
// serial full-vector prologue caps any split-K benefit). The fp32 partial sum is
// a new block-sum association, so results are near-equal (not byte-identical) to
// the single-warp kernel; asymmetric-zp parity is validated against a dequant
// reference to tolerance. Requires K % 256 == 0 (whole steps, no divergent tail)
// — the launch only routes here in that case.
extern "C" __global__ void matmul_nbits_gemv_int8_f16_splitk(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int scales_fp16,
    const int bias_post_round)
{
    constexpr int K_SPLIT = 2;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int cols_per_block = warps_per_block / K_SPLIT;
    const int col_local = warp / K_SPLIT;
    const int ks = warp % K_SPLIT;
    const int column = (int)blockIdx.x * cols_per_block + col_local;

    __shared__ float partials[8][K_SPLIT];

    float value = 0.0f;
    if (column < n) {
        const int quarter = lane & 3;
        for (int block_base = ks * 8; block_base < k_blocks;
             block_base += K_SPLIT * 8) {
            const int block = block_base + (lane >> 2);
            float block_partial = 0.0f;
            if (block < k_blocks) {
                const int zero_point =
                    zero_points ? (int)zero_points[(long)column * k_blocks + block] : 128;
                const int depth = block * 32 + quarter * 8;
                const long packed_start =
                    ((long)column * k_blocks + block) * 32 + quarter * 8;
                if (depth + 8 <= k) {
                    const uint2 packed_word =
                        *reinterpret_cast<const uint2*>(packed + packed_start);
                    const unsigned char* bytes =
                        reinterpret_cast<const unsigned char*>(&packed_word);
                    const uint4 act = *reinterpret_cast<const uint4*>(activation + depth);
                    const __half* acth = reinterpret_cast<const __half*>(&act);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        block_partial += ((float)(int)bytes[i] - (float)zero_point)
                            * __half2float(acth[i]);
                    }
                } else if (depth < k) {
                    const int valid = min(8, k - depth);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid) {
                            const int quantized = (int)packed[packed_start + i];
                            block_partial += ((float)quantized - (float)zero_point)
                                * __half2float(activation[depth + i]);
                        }
                    }
                }
            }
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 2, 4);
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 1, 4);
            if (quarter == 0 && block < k_blocks) {
                const float scale = scales_fp16
                    ? __half2float(reinterpret_cast<const __half*>(scales_raw)
                        [(long)column * k_blocks + block])
                    : reinterpret_cast<const float*>(scales_raw)
                        [(long)column * k_blocks + block];
                value += block_partial * scale;
            }
        }
    }
    value = warp_sum(value);
    if (lane == 0) {
        partials[col_local][ks] = (column < n) ? value : 0.0f;
    }
    __syncthreads();
    if (ks == 0 && lane == 0 && column < n) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < K_SPLIT; ++s) {
            acc += partials[col_local][s];
        }
        output[column] = fold_bias_f16(acc, bias, column, bias_post_round);
    }
}
// block-32 activation/weight tile, so each packed weight is reused by up to 16
// prompt rows and each activation by up to 16 output columns. It deliberately
// uses only ordinary shared memory, fp32 arithmetic, and __half conversion:
// no tensor-core, async-copy, or architecture-specific PTX requirement.
extern "C" __global__ void matmul_nbits_gemm_f16(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int m,
    const int k,
    const int n,
    const int k_blocks,
    const int bits,
    const int scales_fp16,
    const int bias_post_round,
    const int bias_row_stride)
{
    __shared__ float activation_tile[16][32];
    __shared__ float weight_tile[32][16];
    const int tid = (int)threadIdx.y * 16 + (int)threadIdx.x;
    const int row = (int)blockIdx.y * 16 + (int)threadIdx.y;
    const int column = (int)blockIdx.x * 16 + (int)threadIdx.x;
    float value = 0.0f;

    for (int block = 0; block < k_blocks; ++block) {
#pragma unroll
        for (int load = tid; load < 16 * 32; load += 16 * 16) {
            const int tile_row = load >> 5;
            const int within = load & 31;
            const int depth = block * 32 + within;
            const int global_row = (int)blockIdx.y * 16 + tile_row;
            activation_tile[tile_row][within] =
                global_row < m && depth < k
                    ? __half2float(activation[(long)global_row * k + depth])
                    : 0.0f;
        }
#pragma unroll
        for (int load = tid; load < 32 * 16; load += 16 * 16) {
            const int tile_column = load >> 5;
            const int within = load & 31;
            const int global_column = (int)blockIdx.x * 16 + tile_column;
            const int depth = block * 32 + within;
            float weight = 0.0f;
            if (global_column < n && depth < k) {
                const long scale_index = (long)global_column * k_blocks + block;
                const float scale = scales_fp16
                    ? __half2float(
                        reinterpret_cast<const __half*>(scales_raw)[scale_index])
                    : reinterpret_cast<const float*>(scales_raw)[scale_index];
                int quantized;
                int zero_point;
                if (bits == 8) {
                    quantized = (int)packed[scale_index * 32 + within];
                    zero_point = zero_points ? (int)zero_points[scale_index] : 128;
                } else {
                    const unsigned char byte =
                        packed[scale_index * 16 + (within >> 1)];
                    quantized = (within & 1) ? (byte >> 4) : (byte & 15);
                    zero_point = 8;
                    if (zero_points) {
                        const int zp_row_bytes = (k_blocks + 1) >> 1;
                        const unsigned char zp =
                            zero_points[(long)global_column * zp_row_bytes + (block >> 1)];
                        zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
                    }
                }
                weight = ((float)quantized - (float)zero_point) * scale;
            }
            weight_tile[within][tile_column] = weight;
        }
        __syncthreads();

        if (row < m && column < n) {
#pragma unroll
            for (int within = 0; within < 32; ++within) {
                value += activation_tile[threadIdx.y][within]
                    * weight_tile[within][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (row < m && column < n) {
        // A folded residual epilogue binds a per-token residual (row stride N)
        // into the bias slot; a genuine broadcast bias keeps stride 0.
        const __half* row_bias = bias ? bias + (long)row * bias_row_stride : bias;
        output[(long)row * n + column] =
            fold_bias_f16(value, row_bias, column, bias_post_round);
    }
}

__device__ __forceinline__ void int4x8_to_half2x4_sub(
    const unsigned int packed,
    __half2* values,
    const unsigned int sub2)
{
    unsigned int* h = reinterpret_cast<unsigned int*>(values);
    constexpr unsigned int bottom_mask = 0x000f000f;
    constexpr unsigned int top_mask = 0x00f000f0;
    constexpr unsigned int fp16_magic = 0x64006400;
    constexpr unsigned int lop3_lut = (0xf0 & 0xcc) | 0xaa;
    const unsigned int top = packed >> 8;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(packed), "n"(bottom_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(packed), "n"(top_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top), "n"(bottom_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(top), "n"(top_mask), "n"(fp16_magic), "n"(lop3_lut));

    constexpr unsigned int fp16_1024 = 0x64006400;
    constexpr unsigned int fp16_one_sixteenth = 0x2c002c00;
    constexpr unsigned int fp16_neg64 = 0xd400d400;
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[0]) : "r"(h[0]), "r"(fp16_1024));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[1])
                 : "r"(h[1]), "r"(fp16_one_sixteenth), "r"(fp16_neg64));
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[2]) : "r"(h[2]), "r"(fp16_1024));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[3])
                 : "r"(h[3]), "r"(fp16_one_sixteenth), "r"(fp16_neg64));
    // Center each nibble by subtracting the block zero point. A symmetric int4
    // weight uses the implicit `sub2 == 8` (fp16 0x48004800), which reproduces
    // the previous fixed `- 8` byte-for-byte; an asymmetric weight passes its
    // per-block zero point instead so the dequant is `(code - zp)`.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        asm volatile("sub.f16x2 %0, %1, %2;\n"
                     : "=r"(h[i]) : "r"(h[i]), "r"(sub2));
    }
}

// Symmetric int4 dequant: `(code - 8)` in fp16, byte-identical to the historical
// hard-coded `- 8` path (the `sub2` register just carries fp16 8.0).
__device__ __forceinline__ void int4x8_to_half2x4(
    const unsigned int packed,
    __half2* values)
{
    constexpr unsigned int fp16_eight = 0x48004800;
    int4x8_to_half2x4_sub(packed, values, fp16_eight);
}

// Pack a scalar block zero point (nibble in [0, 15]) into an fp16x2 subtrahend
// for [`int4x8_to_half2x4_sub`].
__device__ __forceinline__ unsigned int int4_zero_point_sub2(const int zero_point)
{
    const __half zp = __float2half((float)zero_point);
    const __half2 zp2 = __halves2half2(zp, zp);
    return *reinterpret_cast<const unsigned int*>(&zp2);
}

// Load the block zero point for `column`/`block` from the packed nibble layout,
// or the symmetric default (8) when the weight carries no zero points.
__device__ __forceinline__ int int4_block_zero_point(
    const unsigned char* __restrict__ zero_points,
    const long column,
    const int block,
    const int zp_row_bytes)
{
    if (!zero_points) {
        return 8;
    }
    const unsigned char zp = zero_points[column * zp_row_bytes + (block >> 1)];
    return (block & 1) ? (zp >> 4) : (zp & 15);
}

// Compile-time-specialized per-block subtrahend for the vectorized int4 GEMVs.
// `HasZp == false` (symmetric weights) folds to the constant fp16 `8.0`
// subtrahend with no memory traffic, so the compiler emits the exact
// pre-zero-point instruction stream; `HasZp == true` reads the per-block
// asymmetric zero point. Keying off the template parameter — never the runtime
// pointer — keeps the symmetric decode path byte-identical and register-light.
template <bool HasZp>
__device__ __forceinline__ unsigned int block_sub2(
    const unsigned char* __restrict__ zero_points,
    const long column,
    const int block,
    const int zp_row_bytes)
{
    if (!HasZp) {
        return 0x48004800u;
    }
    return int4_zero_point_sub2(
        int4_block_zero_point(zero_points, column, block, zp_row_bytes));
}

// Scalar counterpart of [`block_sub2`] for the partial-block tail. `HasZp ==
// false` returns the symmetric default (8) with no load.
template <bool HasZp>
__device__ __forceinline__ int block_zp(
    const unsigned char* __restrict__ zero_points,
    const long column,
    const int block,
    const int zp_row_bytes)
{
    if (!HasZp) {
        return 8;
    }
    return int4_block_zero_point(zero_points, column, block, zp_row_bytes);
}

__device__ __forceinline__ float dot_int4x8_f16(
    const unsigned int packed,
    const __half* __restrict__ activation)
{
    const uint4 a = *reinterpret_cast<const uint4*>(activation);
    constexpr unsigned int low_halves = 0x5410;
    constexpr unsigned int high_halves = 0x7632;
    uint4 permuted;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.x) : "r"(a.x), "r"(a.z), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.y) : "r"(a.x), "r"(a.z), "r"(high_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.z) : "r"(a.y), "r"(a.w), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.w) : "r"(a.y), "r"(a.w), "r"(high_halves));

    __half2 q[4];
    int4x8_to_half2x4(packed, q);
    const float2 q04 = __half22float2(q[0]);
    const float2 q15 = __half22float2(q[1]);
    const float2 q26 = __half22float2(q[2]);
    const float2 q37 = __half22float2(q[3]);
    const float2 a04 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.x));
    const float2 a15 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.y));
    const float2 a26 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.z));
    const float2 a37 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.w));
    float dot = q04.x * a04.x;
    dot += q15.x * a15.x;
    dot += q26.x * a26.x;
    dot += q37.x * a37.x;
    dot += q04.y * a04.y;
    dot += q15.y * a15.y;
    dot += q26.y * a26.y;
    dot += q37.y * a37.y;
    return dot;
}

__device__ __forceinline__ uint4 permute_activation_f16x8(
    const __half* __restrict__ activation)
{
    const uint4 a = *reinterpret_cast<const uint4*>(activation);
    constexpr unsigned int low_halves = 0x5410;
    constexpr unsigned int high_halves = 0x7632;
    uint4 permuted;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.x) : "r"(a.x), "r"(a.z), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.y) : "r"(a.x), "r"(a.z), "r"(high_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.z) : "r"(a.y), "r"(a.w), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.w) : "r"(a.y), "r"(a.w), "r"(high_halves));
    return permuted;
}

__device__ __forceinline__ void accumulate_int4x8_f16_permuted(
    const unsigned int packed,
    const uint4& activation,
    const __half scale,
    __half2& sum0,
    __half2& sum1,
    __half2& sum2,
    __half2& sum3)
{
    __half2 q[4];
    int4x8_to_half2x4(packed, q);
    const __half2 scale2 = __halves2half2(scale, scale);
    sum0 = __hfma2(
        __hmul2(q[0], scale2),
        *reinterpret_cast<const __half2*>(&activation.x),
        sum0);
    sum1 = __hfma2(
        __hmul2(q[1], scale2),
        *reinterpret_cast<const __half2*>(&activation.y),
        sum1);
    sum2 = __hfma2(
        __hmul2(q[2], scale2),
        *reinterpret_cast<const __half2*>(&activation.z),
        sum2);
    sum3 = __hfma2(
        __hmul2(q[3], scale2),
        *reinterpret_cast<const __half2*>(&activation.w),
        sum3);
}

// Zero-point-aware [`accumulate_int4x8_f16_permuted`]: `sub2` centers each
// nibble by the block zero point (fp16 8.0 for symmetric weights, giving a
// byte-identical result). Used by the paired gate/up kernels, which permute the
// shared activation once and dequant each projection with its own zero point.
__device__ __forceinline__ void accumulate_int4x8_f16_permuted_zp(
    const unsigned int packed,
    const uint4& activation,
    const __half scale,
    const unsigned int sub2,
    __half2& sum0,
    __half2& sum1,
    __half2& sum2,
    __half2& sum3)
{
    __half2 q[4];
    int4x8_to_half2x4_sub(packed, q, sub2);
    const __half2 scale2 = __halves2half2(scale, scale);
    sum0 = __hfma2(__hmul2(q[0], scale2),
                   *reinterpret_cast<const __half2*>(&activation.x), sum0);
    sum1 = __hfma2(__hmul2(q[1], scale2),
                   *reinterpret_cast<const __half2*>(&activation.y), sum1);
    sum2 = __hfma2(__hmul2(q[2], scale2),
                   *reinterpret_cast<const __half2*>(&activation.z), sum2);
    sum3 = __hfma2(__hmul2(q[3], scale2),
                   *reinterpret_cast<const __half2*>(&activation.w), sum3);
}

__device__ __forceinline__ void accumulate_int4x8_dot_f16(
    const unsigned int packed,
    const uint4& activation,
    const __half2 scale2,
    __half2& sum)
{
    __half2 q[4];
    int4x8_to_half2x4(packed, q);
    sum = __hfma2(
        __hmul2(q[0], scale2),
        *reinterpret_cast<const __half2*>(&activation.x),
        sum);
    sum = __hfma2(
        __hmul2(q[1], scale2),
        *reinterpret_cast<const __half2*>(&activation.y),
        sum);
    sum = __hfma2(
        __hmul2(q[2], scale2),
        *reinterpret_cast<const __half2*>(&activation.z),
        sum);
    sum = __hfma2(
        __hmul2(q[3], scale2),
        *reinterpret_cast<const __half2*>(&activation.w),
        sum);
}

__device__ __forceinline__ float dot_int4x32_f16_permuted_scaled(
    const uint4& packed,
    const uint4& activation0,
    const uint4& activation1,
    const uint4& activation2,
    const uint4& activation3,
    const __half scale)
{
    const __half2 scale2 = __halves2half2(scale, scale);
    __half2 sum0 = __float2half2_rn(0.0f);
    __half2 sum1 = __float2half2_rn(0.0f);
    __half2 sum2 = __float2half2_rn(0.0f);
    __half2 sum3 = __float2half2_rn(0.0f);
    accumulate_int4x8_dot_f16(packed.x, activation0, scale2, sum0);
    accumulate_int4x8_dot_f16(packed.y, activation1, scale2, sum1);
    accumulate_int4x8_dot_f16(packed.z, activation2, scale2, sum2);
    accumulate_int4x8_dot_f16(packed.w, activation3, scale2, sum3);
    const float2 value0 = __half22float2(sum0);
    const float2 value1 = __half22float2(sum1);
    const float2 value2 = __half22float2(sum2);
    const float2 value3 = __half22float2(sum3);
    float value = value0.x;
    value += value1.x;
    value += value2.x;
    value += value3.x;
    value += value0.y;
    value += value1.y;
    value += value2.y;
    value += value3.y;
    return value;
}

__device__ __forceinline__ void accumulate_int4x8_f16(
    const unsigned int packed,
    const __half* __restrict__ activation,
    const __half scale,
    __half2& sum0,
    __half2& sum1,
    __half2& sum2,
    __half2& sum3)
{
    const uint4 permuted = permute_activation_f16x8(activation);
    accumulate_int4x8_f16_permuted(
        packed, permuted, scale, sum0, sum1, sum2, sum3);
}

// Zero-point-aware variant of [`accumulate_int4x8_f16`]: `sub2` is the fp16x2
// subtrahend for this block (the packed zero point, or fp16 8.0 for symmetric
// weights). With the symmetric default this is byte-identical to the plain
// accumulate, so callers can route both symmetric and asymmetric weights here.
__device__ __forceinline__ void accumulate_int4x8_f16_zp(
    const unsigned int packed,
    const __half* __restrict__ activation,
    const __half scale,
    const unsigned int sub2,
    __half2& sum0,
    __half2& sum1,
    __half2& sum2,
    __half2& sum3)
{
    const uint4 permuted = permute_activation_f16x8(activation);
    __half2 q[4];
    int4x8_to_half2x4_sub(packed, q, sub2);
    const __half2 scale2 = __halves2half2(scale, scale);
    sum0 = __hfma2(__hmul2(q[0], scale2),
                   *reinterpret_cast<const __half2*>(&permuted.x), sum0);
    sum1 = __hfma2(__hmul2(q[1], scale2),
                   *reinterpret_cast<const __half2*>(&permuted.y), sum1);
    sum2 = __hfma2(__hmul2(q[2], scale2),
                   *reinterpret_cast<const __half2*>(&permuted.z), sum2);
    sum3 = __hfma2(__hmul2(q[3], scale2),
                   *reinterpret_cast<const __half2*>(&permuted.w), sum3);
}

// One warp per output column; `columns_per_block` (== blockDim.x / 32) columns
// per CTA. Four adjacent lanes split each block-32 weight blob into aligned
// uint32 loads, so every warp issues contiguous 128-byte packed-weight
// transactions. Each lane also reads eight activations with one uint4 load.
// Register-only nibble conversion and four-lane shuffle reduction reconstruct
// each block dot product before applying its scale.
extern "C" __global__ void matmul_nbits_gemv_f16(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;

    float value = 0.0f;
    if (column < n) {
        const int quarter = lane & 3;
        for (int block_base = 0; block_base < k_blocks; block_base += 8) {
            const int block = block_base + (lane >> 2);
            float block_partial = 0.0f;
            if (block < k_blocks) {
                const int depth = block * block_size + quarter * 8;
                const long packed_start =
                    ((long)column * k_blocks + block) * blob_size + quarter * 4;
                const unsigned int packed_word =
                    *reinterpret_cast<const unsigned int*>(packed + packed_start);
                int zero_point = 8;
                if (zero_points) {
                    const unsigned char zp =
                        zero_points[(long)column * zp_row_bytes + block / 2];
                    zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
                }
                if (depth + 8 <= k) {
                    if (zero_points) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int q =
                                (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                            block_partial +=
                                (float)q * __half2float(activation[depth + i]);
                        }
                    } else {
                        block_partial = dot_int4x8_f16(packed_word, activation + depth);
                    }
                } else if (depth < k) {
                    const int valid = min(8, k - depth);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid) {
                            const int q =
                                (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                            block_partial +=
                                (float)q * __half2float(activation[depth + i]);
                        }
                    }
                }
            }
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 2, 4);
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 1, 4);
            if (quarter == 0 && block < k_blocks) {
                float scale;
                if (scales_fp16) {
                    scale = __half2float(
                        reinterpret_cast<const __half*>(scales)[(long)column * k_blocks + block]);
                } else {
                    scale =
                        reinterpret_cast<const float*>(scales)[(long)column * k_blocks + block];
                }
                value += block_partial * scale;
            }
        }
    }

    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

template <bool HasZp>
__device__ __forceinline__ void matmul_nbits_gemv_f16_scales_f16_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    (void)block_size;
    (void)scales_fp16;
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column_base = (int)blockIdx.x * columns_per_block;
    const int column = column_base + warp;

    __half2 sum0 = __float2half2_rn(0.0f);
    __half2 sum1 = __float2half2_rn(0.0f);
    __half2 sum2 = __float2half2_rn(0.0f);
    __half2 sum3 = __float2half2_rn(0.0f);
    float tail = 0.0f;
    if (column < n) {
        const int lane_depth = lane * 8;
        const __half* activation_ptr = activation + lane_depth;
        const unsigned char* packed_ptr =
            packed + (long)column * k_blocks * blob_size + lane * 4;
        const __half* scale_ptr =
            scales + (long)column * k_blocks + (lane >> 2);
        int depth_base = 0;
        for (; depth_base + lane_depth + 8 <= k; depth_base += 256) {
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed_ptr);
            // block == depth/32; each lane's 8 nibbles all sit in one block.
            const int block = (depth_base >> 5) + (lane >> 2);
            const unsigned int sub2 =
                block_sub2<HasZp>(zero_points, column, block, zp_row_bytes);
            accumulate_int4x8_f16_zp(
                packed_word,
                activation_ptr,
                *scale_ptr,
                sub2,
                sum0,
                sum1,
                sum2,
                sum3);
            activation_ptr += 256;
            packed_ptr += 128;
            scale_ptr += 8;
        }
        const int tail_depth = depth_base + lane_depth;
        if (tail_depth < k) {
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed_ptr);
            const float scale = __half2float(*scale_ptr);
            const int tail_block = (depth_base >> 5) + (lane >> 2);
            const int zero_point =
                block_zp<HasZp>(zero_points, column, tail_block, zp_row_bytes);
            const int valid = min(8, k - tail_depth);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < valid) {
                    const int q = (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                    tail += (float)q * __half2float(activation_ptr[i]) * scale;
                }
            }
        }
    }
    const float2 value04 = __half22float2(sum0);
    const float2 value15 = __half22float2(sum1);
    const float2 value26 = __half22float2(sum2);
    const float2 value37 = __half22float2(sum3);
    float value = tail + value04.x;
    value += value15.x;
    value += value26.x;
    value += value37.x;
    value += value04.y;
    value += value15.y;
    value += value26.y;
    value += value37.y;
    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    matmul_nbits_gemv_f16_scales_f16_tpl<false>(activation, packed, scales_raw, zero_points, bias, output, k, n, block_size, k_blocks, blob_size, zp_row_bytes, scales_fp16, bias_post_round);
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    matmul_nbits_gemv_f16_scales_f16_tpl<true>(activation, packed, scales_raw, zero_points, bias, output, k, n, block_size, k_blocks, blob_size, zp_row_bytes, scales_fp16, bias_post_round);
}

// Split-K asymmetric int4 GEMV: K_SPLIT warps cooperate on one output column,
// each reducing a strided subset of the 256-wide K steps, then summing their
// fp32 partials through shared memory. The launch grid is K_SPLIT x larger than
// the single-warp `_zp` kernel, which fills the SMs on this grid-starved,
// latency-bound decode GEMV. Requires K % 256 == 0 (whole steps, no divergent
// tail) — the launch only routes here in that case. The fp32 partial sum is a
// new block-sum association, so results are near-equal (not byte-identical) to
// the single-warp kernel; asymmetric-zp parity is validated against a dequant
// reference to tolerance. Always HasZp (only the zp path is grid-starved enough
// to benefit); the symmetric path keeps its byte-identical single-warp kernel.
extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_zp_splitk(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    (void)block_size;
    (void)scales_fp16;
    constexpr int K_SPLIT = 2;
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int cols_per_block = warps_per_block / K_SPLIT;
    const int col_local = warp / K_SPLIT;
    const int ks = warp % K_SPLIT;
    const int column = (int)blockIdx.x * cols_per_block + col_local;

    __shared__ float partials[8][K_SPLIT];

    __half2 sum0 = __float2half2_rn(0.0f);
    __half2 sum1 = __float2half2_rn(0.0f);
    __half2 sum2 = __float2half2_rn(0.0f);
    __half2 sum3 = __float2half2_rn(0.0f);
    if (column < n) {
        const int lane_depth = lane * 8;
        int depth_base = ks * 256;
        const __half* activation_ptr = activation + depth_base + lane_depth;
        const unsigned char* packed_ptr =
            packed + (long)column * k_blocks * blob_size +
            (long)(depth_base >> 5) * blob_size + lane * 4;
        const __half* scale_ptr =
            scales + (long)column * k_blocks + (depth_base >> 5) + (lane >> 2);
        for (; depth_base + 256 <= k; depth_base += K_SPLIT * 256) {
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed_ptr);
            const int block = (depth_base >> 5) + (lane >> 2);
            const unsigned int sub2 =
                block_sub2<true>(zero_points, column, block, zp_row_bytes);
            accumulate_int4x8_f16_zp(
                packed_word, activation_ptr, *scale_ptr, sub2, sum0, sum1, sum2, sum3);
            activation_ptr += K_SPLIT * 256;
            packed_ptr += K_SPLIT * 128;
            scale_ptr += K_SPLIT * 8;
        }
    }
    const float2 value04 = __half22float2(sum0);
    const float2 value15 = __half22float2(sum1);
    const float2 value26 = __half22float2(sum2);
    const float2 value37 = __half22float2(sum3);
    float value = value04.x;
    value += value15.x;
    value += value26.x;
    value += value37.x;
    value += value04.y;
    value += value15.y;
    value += value26.y;
    value += value37.y;
    value = warp_sum(value);
    if (lane == 0) {
        partials[col_local][ks] = (column < n) ? value : 0.0f;
    }
    __syncthreads();
    if (ks == 0 && lane == 0 && column < n) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < K_SPLIT; ++s) {
            acc += partials[col_local][s];
        }
        output[column] = fold_bias_f16(acc, bias, column, bias_post_round);
    }
}

// Half4 view matching `skip_rmsnorm_f16_warp_half4` so the fused prologue below
// reduces the activation with the exact same chunking and rounding.
union MatMulNBitsSkipHalf4 {
    unsigned long long raw;
    __half2 pair[2];
};

// Scalar RMS-norm gamma load matching `skip_rmsnorm_f16_warp_half4`: gamma is
// only ever a final multiplicand (never part of the fp32 variance
// accumulation), so an fp32 gamma is read at full precision while an fp16 gamma
// keeps the half round-trip. This lets decoders that export gamma in fp32 (e.g.
// Phi-4-mini) take the fused RMS-norm-prologue GEMV path bit-identically to the
// standalone norm + GEMV pair.
__device__ __forceinline__ float load_rmsnorm_gamma(
    const void* __restrict__ gamma,
    const int gamma_is_half,
    const int index)
{
    return gamma_is_half
        ? __half2float(reinterpret_cast<const __half*>(gamma)[index])
        : reinterpret_cast<const float*>(gamma)[index];
}

// General fp16/fp16-scales GEMV with a fused RMS-normalization prologue. The
// preceding GEMV's residual epilogue already produced the byte-identical
// residual sum that `SkipSimplifiedLayerNormalization` would emit as its
// residual output, so this kernel only has to (1) reduce that sum exactly as
// `skip_rmsnorm_f16_warp_half4` does, (2) write the normalized activation into
// shared memory with the same rounding, and (3) run the standard `scales_f16`
// int4 dot over that staged, normalized activation. Every arithmetic step
// mirrors the standalone norm + GEMV pair, so tokens stay bit-for-bit identical
// while the separate normalization kernel is removed from the decode graph.
template <bool HasZp>
__device__ __forceinline__ void matmul_nbits_gemv_f16_scales_f16_rmsnorm_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    // Normalized activation, staged 16-byte aligned so the dot below can reuse
    // the `scales_f16` `uint4` activation loads unchanged.
    extern __shared__ __align__(16) __half staged_activation[];
    __shared__ float shared_inv_std;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // --- RMS reduction, byte-identical to `skip_rmsnorm_f16_warp_half4`. ---
    if (warp == 0) {
        const int chunks_per_lane = k / (32 * 4);
        const unsigned long long* activation4 =
            reinterpret_cast<const unsigned long long*>(activation);
        float ss0 = 0.0f;
        float ss1 = 0.0f;
        float ss2 = 0.0f;
        float ss3 = 0.0f;
        for (int item = 0; item < chunks_per_lane; ++item) {
            const int chunk = lane + item * 32;
            MatMulNBitsSkipHalf4 residual;
            residual.raw = activation4[chunk];
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
        if (lane == 0) {
            shared_inv_std = 1.0f / sqrtf(ss / (float)k + epsilon);
        }
    }
    __syncthreads();
    const float inv_std = shared_inv_std;

    // --- Normalized activation, matching the norm kernel's rounded output. ---
    for (int j = tid; j < k; j += (int)blockDim.x) {
        const float residual = __half2float(activation[j]);
        const float scale = load_rmsnorm_gamma(gamma, gamma_is_half, j);
        staged_activation[j] = __float2half((residual * inv_std) * scale);
    }
    __syncthreads();

    // --- Standard `scales_f16` int4 dot over the staged, normalized input. ---
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);

    __half2 sum0 = __float2half2_rn(0.0f);
    __half2 sum1 = __float2half2_rn(0.0f);
    __half2 sum2 = __float2half2_rn(0.0f);
    __half2 sum3 = __float2half2_rn(0.0f);
    float tail = 0.0f;
    if (column < n) {
        const int lane_depth = lane * 8;
        const __half* activation_ptr = staged_activation + lane_depth;
        const unsigned char* packed_ptr =
            packed + (long)column * k_blocks * blob_size + lane * 4;
        const __half* scale_ptr =
            scales + (long)column * k_blocks + (lane >> 2);
        int depth_base = 0;
        for (; depth_base + lane_depth + 8 <= k; depth_base += 256) {
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed_ptr);
            const int block = (depth_base >> 5) + (lane >> 2);
            const unsigned int sub2 =
                block_sub2<HasZp>(zero_points, column, block, zp_row_bytes);
            accumulate_int4x8_f16_zp(
                packed_word,
                activation_ptr,
                *scale_ptr,
                sub2,
                sum0,
                sum1,
                sum2,
                sum3);
            activation_ptr += 256;
            packed_ptr += 128;
            scale_ptr += 8;
        }
        const int tail_depth = depth_base + lane_depth;
        if (tail_depth < k) {
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed_ptr);
            const float scale = __half2float(*scale_ptr);
            const int tail_block = (depth_base >> 5) + (lane >> 2);
            const int zero_point =
                block_zp<HasZp>(zero_points, column, tail_block, zp_row_bytes);
            const int valid = min(8, k - tail_depth);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < valid) {
                    const int q = (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                    tail += (float)q * __half2float(activation_ptr[i]) * scale;
                }
            }
        }
    }
    const float2 value04 = __half22float2(sum0);
    const float2 value15 = __half22float2(sum1);
    const float2 value26 = __half22float2(sum2);
    const float2 value37 = __half22float2(sum3);
    float value = tail + value04.x;
    value += value15.x;
    value += value26.x;
    value += value37.x;
    value += value04.y;
    value += value15.y;
    value += value26.y;
    value += value37.y;
    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_rmsnorm(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_scales_f16_rmsnorm_tpl<false>(activation, packed, scales_raw, zero_points, gamma, bias, output, k, n, k_blocks, blob_size, zp_row_bytes, bias_post_round, gamma_is_half, epsilon);
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_rmsnorm_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_scales_f16_rmsnorm_tpl<true>(activation, packed, scales_raw, zero_points, gamma, bias, output, k, n, k_blocks, blob_size, zp_row_bytes, bias_post_round, gamma_is_half, epsilon);
}

// Compile-time-specialized per-block int8 zero point. `HasZp == false`
// (symmetric int8) folds to the constant 128 with no load — mirroring the int4
// `block_zp` helper — so a future symmetric-int8 model keeps the constant
// subtrahend and never pays the per-block occupancy cost the int4 path shed.
template <bool HasZp>
__device__ __forceinline__ int block_zp_int8(
    const unsigned char* __restrict__ zero_points,
    const long column,
    const int block,
    const int k_blocks)
{
    if (!HasZp) {
        return 128;
    }
    return (int)zero_points[column * k_blocks + block];
}

// INT8 sibling of `matmul_nbits_gemv_f16_scales_f16_rmsnorm`. The RMS reduction
// and normalized-activation staging are byte-identical to the int4 fused kernel
// (and to the standalone `skip_rmsnorm_f16_warp_half4`); only the quantized dot
// differs, reusing the exact block-32 int8 dequant work split from
// `matmul_nbits_gemv_int8_f16` (one byte per weight, per-block uint8 zero point
// defaulting to 128, fp32 accumulation). Specialized on `HasZp` like the int4
// sibling so the symmetric case emits no per-block zero-point load.
template <bool HasZp>
__device__ __forceinline__ void matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    extern __shared__ __align__(16) __half staged_activation[];
    __shared__ float shared_inv_std;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // --- RMS reduction, byte-identical to `skip_rmsnorm_f16_warp_half4`. ---
    if (warp == 0) {
        const int chunks_per_lane = k / (32 * 4);
        const unsigned long long* activation4 =
            reinterpret_cast<const unsigned long long*>(activation);
        float ss0 = 0.0f;
        float ss1 = 0.0f;
        float ss2 = 0.0f;
        float ss3 = 0.0f;
        for (int item = 0; item < chunks_per_lane; ++item) {
            const int chunk = lane + item * 32;
            MatMulNBitsSkipHalf4 residual;
            residual.raw = activation4[chunk];
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
        if (lane == 0) {
            shared_inv_std = 1.0f / sqrtf(ss / (float)k + epsilon);
        }
    }
    __syncthreads();
    const float inv_std = shared_inv_std;

    // --- Normalized activation, matching the norm kernel's rounded output. ---
    for (int j = tid; j < k; j += (int)blockDim.x) {
        const float residual = __half2float(activation[j]);
        const float scale = load_rmsnorm_gamma(gamma, gamma_is_half, j);
        staged_activation[j] = __float2half((residual * inv_std) * scale);
    }
    __syncthreads();

    // --- INT8 dot over the staged, normalized input (mirrors the non-fused
    //     `matmul_nbits_gemv_int8_f16` work split, fp32 accumulation). ---
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);

    float value = 0.0f;
    if (column < n) {
        const int quarter = lane & 3;
        for (int block_base = 0; block_base < k_blocks; block_base += 8) {
            const int block = block_base + (lane >> 2);
            float block_partial = 0.0f;
            if (block < k_blocks) {
                const int zero_point =
                    block_zp_int8<HasZp>(zero_points, column, block, k_blocks);
                const int depth = block * 32 + quarter * 8;
                const long packed_start =
                    ((long)column * k_blocks + block) * 32 + quarter * 8;
                if (depth + 8 <= k) {
                    const uint2 packed_word =
                        *reinterpret_cast<const uint2*>(packed + packed_start);
                    const unsigned char* bytes =
                        reinterpret_cast<const unsigned char*>(&packed_word);
                    const uint4 act =
                        *reinterpret_cast<const uint4*>(staged_activation + depth);
                    const __half* acth = reinterpret_cast<const __half*>(&act);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        block_partial += ((float)(int)bytes[i] - (float)zero_point)
                            * __half2float(acth[i]);
                    }
                } else if (depth < k) {
                    const int valid = min(8, k - depth);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid) {
                            const int quantized = (int)packed[packed_start + i];
                            block_partial += ((float)quantized - (float)zero_point)
                                * __half2float(staged_activation[depth + i]);
                        }
                    }
                }
            }
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 2, 4);
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 1, 4);
            if (quarter == 0 && block < k_blocks) {
                const float scale =
                    __half2float(scales[(long)column * k_blocks + block]);
                value += block_partial * scale;
            }
        }
    }
    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

extern "C" __global__ void matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm_tpl<false>(activation, packed, scales_raw, zero_points, gamma, bias, output, k, n, k_blocks, bias_post_round, gamma_is_half, epsilon);
}

extern "C" __global__ void matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm_tpl<true>(activation, packed, scales_raw, zero_points, gamma, bias, output, k, n, k_blocks, bias_post_round, gamma_is_half, epsilon);
}

// Standalone RMS-normalization prologue for the M>1 prefill path of the fused
// GEMV. It reproduces `skip_rmsnorm_f16_warp_half4` (minus the residual add,
// which the preceding GEMV's epilogue already applied) bit-for-bit: identical
// half4 chunking, identical `(ss0+ss1)+(ss2+ss3)` reduction, identical warp
// shuffle, and identical `__floats2half2_rn` output rounding. One warp
// normalizes one token row into `normalized`, which the portable tiled GEMM
// then consumes exactly as it would the standalone norm's fp16 output.
extern "C" __global__ void matmul_nbits_rmsnorm_f16_warp_half4(
    const __half* __restrict__ activation,
    const void* __restrict__ gamma,
    __half* __restrict__ normalized,
    const int norm_size,
    const int num_groups,
    const int gamma_is_half,
    const float epsilon)
{
    const int g = (int)blockIdx.x;
    if (g >= num_groups) return;
    const long base = (long)g * norm_size;
    const int lane = (int)threadIdx.x;
    const int chunks_per_lane = norm_size / (32 * 4);
    const unsigned long long* activation4 =
        reinterpret_cast<const unsigned long long*>(activation + base);
    const unsigned long long* gamma4 =
        reinterpret_cast<const unsigned long long*>(gamma);
    unsigned long long* normalized4 =
        reinterpret_cast<unsigned long long*>(normalized + base);
    float ss0 = 0.0f;
    float ss1 = 0.0f;
    float ss2 = 0.0f;
    float ss3 = 0.0f;
    for (int item = 0; item < chunks_per_lane; ++item) {
        const int chunk = lane + item * 32;
        MatMulNBitsSkipHalf4 residual;
        residual.raw = activation4[chunk];
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
    }
    inv_std = __shfl_sync(0xffffffffu, inv_std, 0);
    for (int item = 0; item < chunks_per_lane; ++item) {
        const int chunk = lane + item * 32;
        MatMulNBitsSkipHalf4 residual;
        MatMulNBitsSkipHalf4 output;
        residual.raw = activation4[chunk];
        const float2 value0 = __half22float2(residual.pair[0]);
        const float2 value1 = __half22float2(residual.pair[1]);
        // gamma is only a final multiplicand: an fp16 gamma keeps the wide
        // half4 load, an fp32 gamma is read at full precision (matching the
        // standalone `skip_rmsnorm_f16_warp_half4`), so fp32-gamma decoders fuse.
        float scale0x, scale0y, scale1x, scale1y;
        if (gamma_is_half) {
            MatMulNBitsSkipHalf4 scale;
            scale.raw = gamma4[chunk];
            const float2 scale0 = __half22float2(scale.pair[0]);
            const float2 scale1 = __half22float2(scale.pair[1]);
            scale0x = scale0.x;
            scale0y = scale0.y;
            scale1x = scale1.x;
            scale1y = scale1.y;
        } else {
            const int j = chunk << 2;
            const float* gamma_f = reinterpret_cast<const float*>(gamma);
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
        normalized4[chunk] = output.raw;
    }
}

// SwiGLU activation, byte-identical to the standalone `op_silu` in the
// elementwise kernels: silu(x) = x * sigmoid(x), evaluated in the same
// rounding-stable form so the paired epilogue reproduces the two-op tokens.
__device__ __forceinline__ float gate_up_silu_f32(float x)
{
    if (x >= 0.0f) {
        const float denominator = __fadd_rn(1.0f, (float)exp((double)-x));
        return __fdiv_rn(x, denominator);
    }
    const float e = (float)exp((double)x);
    const float numerator = __fmul_rn(x, e);
    return __fdiv_rn(numerator, __fadd_rn(1.0f, e));
}

// Paired gate/up projection + SwiGLU. One warp computes column `column` of BOTH
// the gate and up projections (which share the same activation and the block-32
// fp16 layout of `matmul_nbits_gemv_f16_scales_f16`), then writes
// silu(gate)*up directly. The activation is permuted once per K-tile and reused
// by both accumulators, so the two GEMVs read the activation from registers
// exactly once. The epilogue reproduces the standalone two-op numerics
// (`fp16(gate_acc)`, `fp16(up_acc)`, then `fp16(silu(gate_h)*up_h)`) so greedy
// decoding stays byte-identical. Register-only + warp shuffles: no shared
// memory, so it is portable to sm_53+ and safe on small SMs (no >48KB opt-in).
template <bool HasZp>
__device__ __forceinline__ void matmul_nbits_gemv_f16_gate_up_swiglu_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;

    __half2 g0 = __float2half2_rn(0.0f);
    __half2 g1 = __float2half2_rn(0.0f);
    __half2 g2 = __float2half2_rn(0.0f);
    __half2 g3 = __float2half2_rn(0.0f);
    __half2 u0 = __float2half2_rn(0.0f);
    __half2 u1 = __float2half2_rn(0.0f);
    __half2 u2 = __float2half2_rn(0.0f);
    __half2 u3 = __float2half2_rn(0.0f);
    float gate_tail = 0.0f;
    float up_tail = 0.0f;
    if (column < n) {
        const int lane_depth = lane * 8;
        const __half* activation_ptr = activation + lane_depth;
        const unsigned char* packed_gate_ptr =
            packed_gate + (long)column * k_blocks * blob_size + lane * 4;
        const unsigned char* packed_up_ptr =
            packed_up + (long)column * k_blocks * blob_size + lane * 4;
        const __half* scale_gate_ptr =
            scales_gate + (long)column * k_blocks + (lane >> 2);
        const __half* scale_up_ptr =
            scales_up + (long)column * k_blocks + (lane >> 2);
        int depth_base = 0;
        for (; depth_base + lane_depth + 8 <= k; depth_base += 256) {
            // Permute the shared activation once; both projections reuse it.
            const uint4 permuted = permute_activation_f16x8(activation_ptr);
            const int block = (depth_base >> 5) + (lane >> 2);
            const unsigned int gate_sub2 =
                block_sub2<HasZp>(zero_points_gate, column, block, zp_row_bytes);
            const unsigned int up_sub2 =
                block_sub2<HasZp>(zero_points_up, column, block, zp_row_bytes);
            const unsigned int gate_word =
                *reinterpret_cast<const unsigned int*>(packed_gate_ptr);
            accumulate_int4x8_f16_permuted_zp(
                gate_word, permuted, *scale_gate_ptr, gate_sub2, g0, g1, g2, g3);
            const unsigned int up_word =
                *reinterpret_cast<const unsigned int*>(packed_up_ptr);
            accumulate_int4x8_f16_permuted_zp(
                up_word, permuted, *scale_up_ptr, up_sub2, u0, u1, u2, u3);
            activation_ptr += 256;
            packed_gate_ptr += 128;
            packed_up_ptr += 128;
            scale_gate_ptr += 8;
            scale_up_ptr += 8;
        }
        const int tail_depth = depth_base + lane_depth;
        if (tail_depth < k) {
            const unsigned int gate_word =
                *reinterpret_cast<const unsigned int*>(packed_gate_ptr);
            const unsigned int up_word =
                *reinterpret_cast<const unsigned int*>(packed_up_ptr);
            const float gate_scale = __half2float(*scale_gate_ptr);
            const float up_scale = __half2float(*scale_up_ptr);
            const int tail_block = (depth_base >> 5) + (lane >> 2);
            const int gate_zp =
                block_zp<HasZp>(zero_points_gate, column, tail_block, zp_row_bytes);
            const int up_zp =
                block_zp<HasZp>(zero_points_up, column, tail_block, zp_row_bytes);
            const int valid = min(8, k - tail_depth);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < valid) {
                    const float a = __half2float(activation_ptr[i]);
                    const int qg = (int)((gate_word >> (i * 4)) & 15u) - gate_zp;
                    const int qu = (int)((up_word >> (i * 4)) & 15u) - up_zp;
                    gate_tail += (float)qg * a * gate_scale;
                    up_tail += (float)qu * a * up_scale;
                }
            }
        }
    }
    // Reduce each accumulator in the exact term order of the standalone
    // `matmul_nbits_gemv_f16_scales_f16` epilogue so the pre-round sums match.
    const float2 g04 = __half22float2(g0);
    const float2 g15 = __half22float2(g1);
    const float2 g26 = __half22float2(g2);
    const float2 g37 = __half22float2(g3);
    float gate_value = gate_tail + g04.x;
    gate_value += g15.x;
    gate_value += g26.x;
    gate_value += g37.x;
    gate_value += g04.y;
    gate_value += g15.y;
    gate_value += g26.y;
    gate_value += g37.y;
    gate_value = warp_sum(gate_value);

    const float2 u04 = __half22float2(u0);
    const float2 u15 = __half22float2(u1);
    const float2 u26 = __half22float2(u2);
    const float2 u37 = __half22float2(u3);
    float up_value = up_tail + u04.x;
    up_value += u15.x;
    up_value += u26.x;
    up_value += u37.x;
    up_value += u04.y;
    up_value += u15.y;
    up_value += u26.y;
    up_value += u37.y;
    up_value = warp_sum(up_value);

    if (lane == 0 && column < n) {
        // Round each projection to fp16 first (matching the separate GEMV
        // stores), then compute silu(gate)*up and round once — identical to the
        // standalone silu_mul_f16 kernel fed by the two GEMV outputs.
        const float gate_h = __half2float(__float2half(gate_value));
        const float up_h = __half2float(__float2half(up_value));
        output[column] =
            __float2half_rn(__fmul_rn(gate_up_silu_f32(gate_h), up_h));
    }
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_swiglu(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_tpl<false>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, output, k, n, k_blocks, blob_size, zp_row_bytes);
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_swiglu_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_tpl<true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, output, k, n, k_blocks, blob_size, zp_row_bytes);
}

// Paired gate/up projection + SwiGLU with a fused RMS-normalization prologue.
// This is `matmul_nbits_gemv_f16_gate_up_swiglu` preceded by the exact prologue
// of `matmul_nbits_gemv_f16_scales_f16_rmsnorm`: the block reduces the shared
// activation (the residual sum the preceding GEMV epilogue already produced)
// once, stages the normalized activation into shared memory with the same
// rounding, and then both the gate and up GEMVs read that single staged,
// normalized activation. Doing the reduction once — rather than once per
// following GEMV — is the whole point of routing the fan-out-2 post-attention
// `SkipSimplifiedLayerNormalization` through the paired kernel. Every arithmetic
// step mirrors the standalone norm followed by the two-op gate/up SwiGLU, so
// greedy tokens stay bit-for-bit identical.
template <bool HasZp>
__device__ __forceinline__ void matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    extern __shared__ __align__(16) __half staged_activation[];
    __shared__ float shared_inv_std;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // --- RMS reduction, byte-identical to `skip_rmsnorm_f16_warp_half4`. ---
    if (warp == 0) {
        const int chunks_per_lane = k / (32 * 4);
        const unsigned long long* activation4 =
            reinterpret_cast<const unsigned long long*>(activation);
        float ss0 = 0.0f;
        float ss1 = 0.0f;
        float ss2 = 0.0f;
        float ss3 = 0.0f;
        for (int item = 0; item < chunks_per_lane; ++item) {
            const int chunk = lane + item * 32;
            MatMulNBitsSkipHalf4 residual;
            residual.raw = activation4[chunk];
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
        if (lane == 0) {
            shared_inv_std = 1.0f / sqrtf(ss / (float)k + epsilon);
        }
    }
    __syncthreads();
    const float inv_std = shared_inv_std;

    // --- Normalized activation, matching the norm kernel's rounded output. ---
    for (int j = tid; j < k; j += (int)blockDim.x) {
        const float residual = __half2float(activation[j]);
        const float scale = load_rmsnorm_gamma(gamma, gamma_is_half, j);
        staged_activation[j] = __float2half((residual * inv_std) * scale);
    }
    __syncthreads();

    // --- Paired gate/up int4 dot over the staged, normalized activation. ---
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;

    __half2 g0 = __float2half2_rn(0.0f);
    __half2 g1 = __float2half2_rn(0.0f);
    __half2 g2 = __float2half2_rn(0.0f);
    __half2 g3 = __float2half2_rn(0.0f);
    __half2 u0 = __float2half2_rn(0.0f);
    __half2 u1 = __float2half2_rn(0.0f);
    __half2 u2 = __float2half2_rn(0.0f);
    __half2 u3 = __float2half2_rn(0.0f);
    float gate_tail = 0.0f;
    float up_tail = 0.0f;
    if (column < n) {
        const int lane_depth = lane * 8;
        const __half* activation_ptr = staged_activation + lane_depth;
        const unsigned char* packed_gate_ptr =
            packed_gate + (long)column * k_blocks * blob_size + lane * 4;
        const unsigned char* packed_up_ptr =
            packed_up + (long)column * k_blocks * blob_size + lane * 4;
        const __half* scale_gate_ptr =
            scales_gate + (long)column * k_blocks + (lane >> 2);
        const __half* scale_up_ptr =
            scales_up + (long)column * k_blocks + (lane >> 2);
        int depth_base = 0;
        if constexpr (HasZp) {
            // Asymmetric (zero-point) path — the dominant Phi decode kernel, which
            // ncu shows is Long-Scoreboard/global-load-latency bound. Software-
            // pipeline the int4 gate/up weight loads: issue the next iteration's
            // two 128-byte weight words before consuming the current ones so the
            // load latency overlaps this iteration's compute. Pure scheduling
            // change (identical accumulation order/ops) → bit-identical to the
            // non-prefetched loop. Only the weight words are prefetched; also
            // prefetching the small (L1/L2-resident) scales and per-block zero
            // points pushed registers 48->56 and the occupancy loss erased the
            // latency win. The symmetric (`HasZp == false`) path below keeps its
            // exact original instruction stream so Qwen stays byte-identical with
            // no register/occupancy change.
            unsigned int gate_word_next =
                *reinterpret_cast<const unsigned int*>(packed_gate_ptr);
            unsigned int up_word_next =
                *reinterpret_cast<const unsigned int*>(packed_up_ptr);
            for (; depth_base + lane_depth + 8 <= k; depth_base += 256) {
                const uint4 permuted = permute_activation_f16x8(activation_ptr);
                const int block = (depth_base >> 5) + (lane >> 2);
                const unsigned int gate_sub2 =
                    block_sub2<HasZp>(zero_points_gate, column, block, zp_row_bytes);
                const unsigned int up_sub2 =
                    block_sub2<HasZp>(zero_points_up, column, block, zp_row_bytes);
                const unsigned int gate_word = gate_word_next;
                const unsigned int up_word = up_word_next;
                if (depth_base + 256 + lane_depth + 8 <= k) {
                    gate_word_next = *reinterpret_cast<const unsigned int*>(
                        packed_gate_ptr + 128);
                    up_word_next = *reinterpret_cast<const unsigned int*>(
                        packed_up_ptr + 128);
                }
                accumulate_int4x8_f16_permuted_zp(
                    gate_word, permuted, *scale_gate_ptr, gate_sub2, g0, g1, g2, g3);
                accumulate_int4x8_f16_permuted_zp(
                    up_word, permuted, *scale_up_ptr, up_sub2, u0, u1, u2, u3);
                activation_ptr += 256;
                packed_gate_ptr += 128;
                packed_up_ptr += 128;
                scale_gate_ptr += 8;
                scale_up_ptr += 8;
            }
        } else {
            for (; depth_base + lane_depth + 8 <= k; depth_base += 256) {
                const uint4 permuted = permute_activation_f16x8(activation_ptr);
                const int block = (depth_base >> 5) + (lane >> 2);
                const unsigned int gate_sub2 =
                    block_sub2<HasZp>(zero_points_gate, column, block, zp_row_bytes);
                const unsigned int up_sub2 =
                    block_sub2<HasZp>(zero_points_up, column, block, zp_row_bytes);
                const unsigned int gate_word =
                    *reinterpret_cast<const unsigned int*>(packed_gate_ptr);
                accumulate_int4x8_f16_permuted_zp(
                    gate_word, permuted, *scale_gate_ptr, gate_sub2, g0, g1, g2, g3);
                const unsigned int up_word =
                    *reinterpret_cast<const unsigned int*>(packed_up_ptr);
                accumulate_int4x8_f16_permuted_zp(
                    up_word, permuted, *scale_up_ptr, up_sub2, u0, u1, u2, u3);
                activation_ptr += 256;
                packed_gate_ptr += 128;
                packed_up_ptr += 128;
                scale_gate_ptr += 8;
                scale_up_ptr += 8;
            }
        }
        const int tail_depth = depth_base + lane_depth;
        if (tail_depth < k) {
            const unsigned int gate_word =
                *reinterpret_cast<const unsigned int*>(packed_gate_ptr);
            const unsigned int up_word =
                *reinterpret_cast<const unsigned int*>(packed_up_ptr);
            const float gate_scale = __half2float(*scale_gate_ptr);
            const float up_scale = __half2float(*scale_up_ptr);
            const int tail_block = (depth_base >> 5) + (lane >> 2);
            const int gate_zp =
                block_zp<HasZp>(zero_points_gate, column, tail_block, zp_row_bytes);
            const int up_zp =
                block_zp<HasZp>(zero_points_up, column, tail_block, zp_row_bytes);
            const int valid = min(8, k - tail_depth);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < valid) {
                    const float a = __half2float(activation_ptr[i]);
                    const int qg = (int)((gate_word >> (i * 4)) & 15u) - gate_zp;
                    const int qu = (int)((up_word >> (i * 4)) & 15u) - up_zp;
                    gate_tail += (float)qg * a * gate_scale;
                    up_tail += (float)qu * a * up_scale;
                }
            }
        }
    }
    const float2 g04 = __half22float2(g0);
    const float2 g15 = __half22float2(g1);
    const float2 g26 = __half22float2(g2);
    const float2 g37 = __half22float2(g3);
    float gate_value = gate_tail + g04.x;
    gate_value += g15.x;
    gate_value += g26.x;
    gate_value += g37.x;
    gate_value += g04.y;
    gate_value += g15.y;
    gate_value += g26.y;
    gate_value += g37.y;
    gate_value = warp_sum(gate_value);

    const float2 u04 = __half22float2(u0);
    const float2 u15 = __half22float2(u1);
    const float2 u26 = __half22float2(u2);
    const float2 u37 = __half22float2(u3);
    float up_value = up_tail + u04.x;
    up_value += u15.x;
    up_value += u26.x;
    up_value += u37.x;
    up_value += u04.y;
    up_value += u15.y;
    up_value += u26.y;
    up_value += u37.y;
    up_value = warp_sum(up_value);

    if (lane == 0 && column < n) {
        const float gate_h = __half2float(__float2half(gate_value));
        const float up_h = __half2float(__float2half(up_value));
        output[column] =
            __float2half_rn(__fmul_rn(gate_up_silu_f32(gate_h), up_h));
    }
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl<false>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, gamma, output, k, n, k_blocks, blob_size, zp_row_bytes, gamma_is_half, epsilon);
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl<true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, gamma, output, k, n, k_blocks, blob_size, zp_row_bytes, gamma_is_half, epsilon);
}

// Down projection specialization: a 256-thread CTA computes eight columns and
// parallelizes over block-32 K tiles. Each thread loads its assigned activation
// block directly into registers and reuses it across all eight columns.
extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_down(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    (void)block_size;
    (void)zero_points;
    (void)zp_row_bytes;
    (void)scales_fp16;
    __shared__ float warp_sums[8][8];
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int column_base = (int)blockIdx.x * 8;

    float values[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int block = tid; block < k_blocks; block += (int)blockDim.x) {
        const __half* activation_block = activation + block * 32;
        const uint4 activation0 = permute_activation_f16x8(activation_block);
        const uint4 activation1 = permute_activation_f16x8(activation_block + 8);
        const uint4 activation2 = permute_activation_f16x8(activation_block + 16);
        const uint4 activation3 = permute_activation_f16x8(activation_block + 24);
#pragma unroll
        for (int tile_column = 0; tile_column < 8; ++tile_column) {
            const int column = column_base + tile_column;
            if (column < n) {
                const long packed_start =
                    ((long)column * k_blocks + block) * blob_size;
                const uint4 packed_weights =
                    *reinterpret_cast<const uint4*>(packed + packed_start);
                const __half scale = scales[(long)column * k_blocks + block];
                values[tile_column] += dot_int4x32_f16_permuted_scaled(
                    packed_weights,
                    activation0,
                    activation1,
                    activation2,
                    activation3,
                    scale);
            }
        }
    }

#pragma unroll
    for (int tile_column = 0; tile_column < 8; ++tile_column) {
        const float value = warp_sum(values[tile_column]);
        if (lane == 0) {
            warp_sums[warp][tile_column] = value;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < 8) {
        const int column = column_base + lane;
        float value = warp_sums[0][lane];
        value += warp_sums[1][lane];
        value += warp_sums[2][lane];
        value += warp_sums[3][lane];
        value += warp_sums[4][lane];
        value += warp_sums[5][lane];
        value += warp_sums[6][lane];
        value += warp_sums[7][lane];
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

// Model-agnostic fp16 int4 decode GEMV supporting any power-of-two block_size.
// One warp per output column. Each lane owns contiguous 8-element (one packed
// uint32) K chunks and strides by 256 (= 32 lanes * 8) across the reduction.
// Unlike the tuned block-32 kernels, the scale / zero-point block index is
// derived from the real block_size (block = depth / block_size), so a lane's
// 8-element chunk always resolves to the block it belongs to for any block
// width that is a multiple of 8 (all supported power-of-two block sizes >= 16).
// fp32 accumulation is preserved; the kernel is register-only (capture-safe).
extern "C" __global__ void matmul_nbits_gemv_f16_general_bs(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;

    float value = 0.0f;
    if (column < n) {
        for (int depth = lane * 8; depth < k; depth += 256) {
            const int block = depth / block_size;
            const int within = depth - block * block_size;
            const long packed_start =
                ((long)column * k_blocks + block) * blob_size + (within >> 1);
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed + packed_start);
            int zero_point = 8;
            if (zero_points) {
                const unsigned char zp =
                    zero_points[(long)column * zp_row_bytes + (block >> 1)];
                zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
            }
            float scale;
            if (scales_fp16) {
                scale = __half2float(
                    reinterpret_cast<const __half*>(scales)[(long)column * k_blocks + block]);
            } else {
                scale =
                    reinterpret_cast<const float*>(scales)[(long)column * k_blocks + block];
            }
            const int valid = min(8, k - depth);
            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < valid) {
                    const int q = (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                    partial += (float)q * __half2float(activation[depth + i]);
                }
            }
            value += partial * scale;
        }
    }

    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

// Model-agnostic fp16 int4/int8 prefill GEMM supporting any power-of-two
// block_size. Identical 16x16 tiling and fp32 accumulation as the tuned
// block-32 GEMM, but the reduction walks K in fixed 32-wide tiles and derives
// the block index from the real block_size (block = depth / block_size), so the
// K-tile width is decoupled from the block width. For block_size == 32 this is
// numerically identical to matmul_nbits_gemm_f16 (block == tile).
extern "C" __global__ void matmul_nbits_gemm_f16_general_bs(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int m,
    const int k,
    const int n,
    const int k_blocks,
    const int bits,
    const int scales_fp16,
    const int bias_post_round,
    const int bias_row_stride,
    const int block_size,
    const int blob_size)
{
    __shared__ float activation_tile[16][32];
    __shared__ float weight_tile[32][16];
    const int tid = (int)threadIdx.y * 16 + (int)threadIdx.x;
    const int row = (int)blockIdx.y * 16 + (int)threadIdx.y;
    const int column = (int)blockIdx.x * 16 + (int)threadIdx.x;
    float value = 0.0f;

    const int k_tiles = (k + 31) / 32;
    for (int tile = 0; tile < k_tiles; ++tile) {
#pragma unroll
        for (int load = tid; load < 16 * 32; load += 16 * 16) {
            const int tile_row = load >> 5;
            const int within = load & 31;
            const int depth = tile * 32 + within;
            const int global_row = (int)blockIdx.y * 16 + tile_row;
            activation_tile[tile_row][within] =
                global_row < m && depth < k
                    ? __half2float(activation[(long)global_row * k + depth])
                    : 0.0f;
        }
#pragma unroll
        for (int load = tid; load < 32 * 16; load += 16 * 16) {
            const int tile_column = load >> 5;
            const int within = load & 31;
            const int depth = tile * 32 + within;
            const int global_column = (int)blockIdx.x * 16 + tile_column;
            float weight = 0.0f;
            if (global_column < n && depth < k) {
                const int block = depth / block_size;
                const int within_block = depth - block * block_size;
                const long scale_index = (long)global_column * k_blocks + block;
                const float scale = scales_fp16
                    ? __half2float(
                        reinterpret_cast<const __half*>(scales_raw)[scale_index])
                    : reinterpret_cast<const float*>(scales_raw)[scale_index];
                const long blob_base =
                    ((long)global_column * k_blocks + block) * blob_size;
                int quantized;
                int zero_point;
                if (bits == 8) {
                    quantized = (int)packed[blob_base + within_block];
                    zero_point = zero_points ? (int)zero_points[scale_index] : 128;
                } else {
                    const unsigned char byte =
                        packed[blob_base + (within_block >> 1)];
                    quantized = (within_block & 1) ? (byte >> 4) : (byte & 15);
                    zero_point = 8;
                    if (zero_points) {
                        const int zp_row_bytes = (k_blocks + 1) >> 1;
                        const unsigned char zp =
                            zero_points[(long)global_column * zp_row_bytes + (block >> 1)];
                        zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
                    }
                }
                weight = ((float)quantized - (float)zero_point) * scale;
            }
            weight_tile[within][tile_column] = weight;
        }
        __syncthreads();

        if (row < m && column < n) {
#pragma unroll
            for (int within = 0; within < 32; ++within) {
                value += activation_tile[threadIdx.y][within]
                    * weight_tile[within][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (row < m && column < n) {
        const __half* row_bias = bias ? bias + (long)row * bias_row_stride : bias;
        output[(long)row * n + column] =
            fold_bias_f16(value, row_bias, column, bias_post_round);
    }
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum F16GemvVariant {
    General,
    DownProjection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct F16GemvSelection {
    variant: F16GemvVariant,
    reason: &'static str,
}

/// Choose the fp16 int4 GEMV variant by **structural shape class + capability**,
/// never by a specific model's dimensions.
///
/// The specialized `DownProjection` tiling stages the entire activation in
/// shared memory once and reuses it across the 8 columns of a CTA while the full
/// 256-thread block cooperatively reduces along `K`. That wins on the
/// **tall-skinny** class — a `K > N` GEMV (reduction depth exceeds output width,
/// e.g. an MLP down-projection or attention output-projection) — where the long
/// reduction benefits from block-parallel accumulation and each thread reuses
/// its register-held activation block across the CTA's columns. It is only
/// *correct* under the tiling's
/// hard constraints, all derived from the kernel body:
///
/// * `scales_fp16` and 4-bit weights (this fp16 GEMV path is always 4-bit),
/// * no explicit zero-points (the specialized half2/down kernels encode zp=8),
/// * `block_size == 32` and `K % 32 == 0` (full K-blocks; the kernel has no
///   partial-block tail).
///
/// Every other shape (wide `N >= K` projections, non-block-32, non-multiple-of-32
/// `K`) falls back to the general per-warp GEMV. Selection is thus generic across
/// models: any architecture's down/output projection that fits the class is
/// accelerated, and nothing keys on a magic `K`/`N`.
fn select_f16_gemv_variant(
    k: usize,
    n: usize,
    block_size: usize,
    scales_fp16: bool,
    has_zero_points: bool,
) -> F16GemvSelection {
    let down_eligible = !has_zero_points
        && scales_fp16
        && block_size == GEMV_F16_DOWN_BLOCK_SIZE
        && k.is_multiple_of(GEMV_F16_DOWN_BLOCK_SIZE)
        && k > n;
    if down_eligible {
        F16GemvSelection {
            variant: F16GemvVariant::DownProjection,
            reason: "variant=down_projection;class=tall_skinny(K>N);block_size=32;\
                     scales=fp16;K%32==0",
        }
    } else {
        F16GemvSelection {
            variant: F16GemvVariant::General,
            reason: if has_zero_points {
                "variant=general;zero_points=explicit;down_projection requires symmetric zp=8"
            } else {
                "variant=general;class=not(tall_skinny K>N & block_size=32 & \
                 scales=fp16 & K%32==0)"
            },
        }
    }
}

pub struct MatMulNBitsFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for MatMulNBitsFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = required_positive_attr(node, "K")?;
        let n = required_positive_attr(node, "N")?;
        let bits = optional_int_attr(node, "bits")?.unwrap_or(4);
        if !matches!(bits, 4 | 8) {
            return Err(error(format!(
                "MatMulNBits CUDA supports bits in {{4, 8}}, got bits={bits}. Why: the native \
                 kernels implement packed int4 and block-32 int8 layouts. How to fix: export \
                 bits=4, or export bits=8 with block_size=32, or select another execution provider"
            )));
        }
        let weight_prepacked = optional_int_attr(node, "weight_prepacked")?.unwrap_or(0);
        if weight_prepacked != 0 {
            return Err(error(format!(
                "weight_prepacked={weight_prepacked} is unsupported: CUDA only supports the standard (non-prepacked) layout"
            )));
        }
        let block_size = required_positive_attr(node, "block_size")?;
        if block_size < 16 || !block_size.is_power_of_two() {
            return Err(error(format!(
                "block_size must be a power of two and at least 16, got {block_size}"
            )));
        }
        if bits == 8 && block_size != 32 {
            return Err(error(format!(
                "MatMulNBits CUDA received bits=8 with block_size={block_size}. Why: the native \
                 int8 GEMV currently implements the standard one-byte-per-weight block-32 layout. \
                 How to fix: export bits=8 with block_size=32, use bits=4, or select another \
                 execution provider"
            )));
        }
        let accuracy_level = node
            .attr("accuracy_level")
            .and_then(|value| value.as_int())
            .unwrap_or(0);

        let accuracy4_workspace = if bits == 4 && accuracy_level == 4 && block_size == 32 {
            Some(Mutex::new(Accuracy4Workspace::new(
                self.runtime.clone(),
                k,
            )?))
        } else {
            None
        };
        Ok(Box::new(MatMulNBitsKernel {
            runtime: self.runtime.clone(),
            k,
            n,
            bits: bits as usize,
            block_size,
            accuracy_level,
            accuracy4_workspace,
            fold_bias_post_round: node
                .attr(crate::optimizer::MATMUL_NBITS_FOLDED_BIAS_ATTR)
                .and_then(onnx_runtime_ir::Attribute::as_int)
                == Some(1),
            gate_up_swiglu: node
                .attr(crate::optimizer::GATE_UP_SWIGLU_FUSION_ATTR)
                .and_then(onnx_runtime_ir::Attribute::as_int)
                == Some(1),
            rmsnorm_prologue: node
                .attr(crate::optimizer::MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR)
                .and_then(onnx_runtime_ir::Attribute::as_int)
                == Some(1),
            rmsnorm_epsilon: node
                .attr(crate::optimizer::MATMUL_NBITS_RMSNORM_EPSILON_ATTR)
                .and_then(onnx_runtime_ir::Attribute::as_float)
                .unwrap_or(1e-5),
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

#[derive(Debug)]
struct Accuracy4Workspace {
    runtime: Arc<CudaRuntime>,
    quantized_activation: CUdeviceptr,
    activation_scale: CUdeviceptr,
    padded_k: usize,
}

impl Accuracy4Workspace {
    fn new(runtime: Arc<CudaRuntime>, k: usize) -> Result<Self> {
        let padded_k = k.div_ceil(32) * 32;
        let quantized_activation = runtime.alloc_raw(padded_k + std::mem::size_of::<f32>())?;
        Ok(Self {
            runtime,
            quantized_activation,
            activation_scale: quantized_activation + padded_k as CUdeviceptr,
            padded_k,
        })
    }
}

impl Drop for Accuracy4Workspace {
    fn drop(&mut self) {
        if self.quantized_activation != 0 {
            // SAFETY: this persistent buffer is exclusively owned by the kernel.
            let _ = unsafe { self.runtime.free_raw(self.quantized_activation) };
            self.quantized_activation = 0;
            self.activation_scale = 0;
        }
    }
}

#[derive(Debug)]
pub struct MatMulNBitsKernel {
    runtime: Arc<CudaRuntime>,
    k: usize,
    n: usize,
    bits: usize,
    block_size: usize,
    accuracy_level: i64,
    accuracy4_workspace: Option<Mutex<Accuracy4Workspace>>,
    /// Set when this node's bias input came from folding a standalone `Add`
    /// (see [`crate::optimizer::MATMUL_NBITS_FOLDED_BIAS_ATTR`]). The fp16 GEMV
    /// then reproduces the two-op `fp16(fp16(acc) + bias)` rounding.
    fold_bias_post_round: bool,
    /// Set on a synthetic node produced by
    /// [`crate::optimizer::CudaGateUpSwiGluFusion`]: inputs are
    /// `[x, W_gate, scales_gate, W_up, scales_up]` and the kernel writes
    /// `silu(gate) * up` directly (see [`GATE_UP_SWIGLU_ENTRY`]).
    gate_up_swiglu: bool,
    /// Set on a general fp16 GEMV whose input activation must be RMS-normalized
    /// in-kernel before the int4 dot, produced by
    /// [`crate::optimizer::CudaSkipRmsNormMatMulFusion`]. The `gamma` weight is
    /// bound at input slot 6 and the kernel reproduces
    /// `skip_rmsnorm_f16_warp_half4` bit-for-bit (see
    /// [`GEMV_F16_SCALES_F16_RMSNORM_ENTRY`]).
    rmsnorm_prologue: bool,
    /// Epsilon copied from the folded `SkipSimplifiedLayerNormalization` node so
    /// the fused prologue reproduces its `1/sqrt(mean_sq + epsilon)`.
    rmsnorm_epsilon: f32,
    last_call_capture_safe: AtomicBool,
}

impl MatMulNBitsKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.last_call_capture_safe.store(false, Ordering::Relaxed);
        let max_inputs = if self.gate_up_swiglu { 8 } else { 7 };
        if !(3..=max_inputs).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(error(format!(
                "expected 3 to {max_inputs} inputs and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        if inputs[0].dtype == DataType::Float16 {
            if self.gate_up_swiglu {
                return self.run_f16_gate_up_swiglu(inputs, outputs);
            }
            return self.run_f16(inputs, outputs);
        }
        require_dtype("A", inputs[0].dtype, DataType::Float32)?;
        require_dtype("B", inputs[1].dtype, DataType::Uint8)?;
        require_dtype("scales", inputs[2].dtype, DataType::Float32)?;
        require_dtype("Y", outputs[0].dtype, DataType::Float32)?;
        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        if outputs[0].shape != expected_output_shape {
            return Err(error(format!(
                "Y must have shape {expected_output_shape:?}, got {:?}",
                outputs[0].shape
            )));
        }

        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = self.block_size * self.bits / 8;
        require_shape("B", inputs[1].shape, &[self.n, k_blocks, blob_size])?;
        require_flat_or_matrix_shape("scales", inputs[2].shape, self.n, k_blocks)?;

        let zero_points = optional_input(inputs, 3);
        let zp_row_bytes = (k_blocks * self.bits).div_ceil(8);
        if let Some(zp) = zero_points {
            require_dtype("zero_points", zp.dtype, DataType::Uint8)?;
            require_flat_or_matrix_shape("zero_points", zp.shape, self.n, zp_row_bytes)?;
        }

        let group_indices = optional_input(inputs, 4);
        if let Some(g_idx) = group_indices {
            require_dtype("g_idx", g_idx.dtype, DataType::Int32)?;
            if !g_idx.is_contiguous() {
                return Err(error(
                    "g_idx must be contiguous on the CUDA execution provider",
                ));
            }
            let padded_k = k_blocks * self.block_size;
            if g_idx.shape != [self.k] && g_idx.shape != [padded_k] {
                return Err(error(format!(
                    "g_idx must have shape [{}] or [{padded_k}], got {:?}",
                    self.k, g_idx.shape
                )));
            }
            let mut bytes = vec![0u8; g_idx.numel() * 4];
            // SAFETY: `g_idx` is a live contiguous device tensor and `bytes`
            // exactly covers all of its i32 elements.
            unsafe {
                self.runtime
                    .dtoh(&mut bytes, cuptr(g_idx.data_ptr::<u8>() as *const c_void))?
            };
            for (index, value) in bytes.chunks_exact(4).enumerate() {
                let group = i32::from_ne_bytes([value[0], value[1], value[2], value[3]]);
                if group < 0 || group as usize >= k_blocks {
                    return Err(error(format!(
                        "g_idx[{index}]={group} is outside 0..{k_blocks}"
                    )));
                }
            }
        }

        let bias = optional_input(inputs, 5);
        if let Some(bias) = bias {
            require_dtype("bias", bias.dtype, DataType::Float32)?;
            require_shape("bias", bias.shape, &[self.n])?;
        }

        for (name, contiguous) in [
            ("A", inputs[0].is_contiguous()),
            ("B", inputs[1].is_contiguous()),
            ("scales", inputs[2].is_contiguous()),
            (
                "zero_points",
                zero_points.is_none_or(TensorView::is_contiguous),
            ),
            ("g_idx", group_indices.is_none_or(TensorView::is_contiguous)),
            ("bias", bias.is_none_or(TensorView::is_contiguous)),
            ("Y", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(error(format!(
                    "{name} must be contiguous on the CUDA execution provider"
                )));
            }
        }

        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        self.last_call_capture_safe
            .store(m == 1 && group_indices.is_none(), Ordering::Relaxed);
        if m == 1 && group_indices.is_none() {
            if self.bits == 8 {
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_int8_f32",
                    "M==1 decode: bits=8, block_size=32 → direct capture-safe f32 GEMV"
                );
                return self.launch_int8_f32_gemv(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    bias,
                    &mut outputs[0],
                    k_blocks,
                );
            }
            if self.accuracy_level == 4 && self.block_size == 32 && zero_points.is_none() {
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_accuracy4_int8",
                    "M==1 decode: accuracy_level==4, block_size==32, symmetric (no zero_points) \
                     → int8-quantized-activation capture-safe GEMV"
                );
                return self.launch_accuracy4_gemv(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    bias,
                    &mut outputs[0],
                    k_blocks,
                );
            }
            if self.accuracy_level != 4 {
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_f32",
                    "M==1 decode: accuracy_level={} (non-accuracy4) → direct f32 GEMV",
                    self.accuracy_level
                );
                return self.launch_f32_gemv(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    bias,
                    &mut outputs[0],
                    k_blocks,
                    blob_size,
                    zp_row_bytes,
                );
            }
        }
        if self.bits == 4 && self.accuracy_level == 4 && group_indices.is_none() {
            onnx_runtime_ep_api::record_kernel_variant!(
                "gemm_tiled_accuracy4",
                "M={} (GEMV requires M==1), accuracy_level==4, no g_idx → tiled accuracy4 GEMM",
                m
            );
            return self.launch_accuracy4(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                zero_points,
                bias,
                &mut outputs[0],
                m,
                k_blocks,
                blob_size,
                zp_row_bytes,
            );
        }

        onnx_runtime_ep_api::record_kernel_variant!(
            "dequant_cublas_gemm",
            "M={}, accuracy_level={}, g_idx={} → dequantize weights to f32 then cuBLAS GEMM \
             (general prefill / grouped path)",
            m,
            self.accuracy_level,
            group_indices.is_some()
        );

        let weight = self.runtime.alloc_raw(self.k * self.n * 4)?;
        let workspace = match self.runtime.alloc_raw(WORKSPACE_BYTES) {
            Ok(workspace) => workspace,
            Err(err) => {
                // SAFETY: `weight` was allocated above and has not been freed.
                let _ = unsafe { self.runtime.free_raw(weight) };
                return Err(err);
            }
        };

        let result = self
            .launch_dequant(
                &inputs[1],
                &inputs[2],
                zero_points,
                group_indices,
                weight,
                k_blocks,
                blob_size,
                zp_row_bytes,
            )
            .and_then(|()| {
                let params = GemmParams {
                    dtype: GemmDtype::F32,
                    a: cuptr(inputs[0].data_ptr::<u8>() as *const c_void),
                    b: weight,
                    c: cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void),
                    m,
                    k: self.k,
                    n: self.n,
                    batch: 1,
                    a_batch_stride: m * self.k,
                    b_batch_stride: 0,
                    epilogue: bias.map(|bias| GemmEpilogue {
                        kind: GemmEpilogueKind::Bias,
                        bias: cuptr(bias.data_ptr::<u8>() as *const c_void),
                    }),
                };
                // SAFETY: validated dense f32 A/Y and the dequantized [K,N]
                // allocation cover the complete GEMM; workspace and stream live
                // through the call and Y aliases neither input.
                unsafe {
                    blas::gemm(
                        self.runtime.blas(),
                        self.runtime.stream_ptr(),
                        &params,
                        workspace,
                        WORKSPACE_BYTES,
                    )
                }
            })
            .and_then(|()| self.runtime.synchronize());

        // SAFETY: both pointers came from `alloc_raw` and are released once,
        // after all submitted work has synchronized (or the submission failed).
        let free_workspace = unsafe { self.runtime.free_raw(workspace) };
        let free_weight = unsafe { self.runtime.free_raw(weight) };
        result.and(free_workspace).and(free_weight)
    }

    /// Direct fp16-activation x int4/int8-weight path. Scales may be fp16 or
    /// f32. M=1 uses the capture-safe decode GEMVs; M>1 uses a portable tiled
    /// CUDA-core GEMM with fp32 accumulation and fp16 output.
    fn run_f16(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        require_dtype("A", inputs[0].dtype, DataType::Float16)?;
        require_dtype("B", inputs[1].dtype, DataType::Uint8)?;
        let scales_fp16 = match inputs[2].dtype {
            DataType::Float16 => true,
            DataType::Float32 => false,
            other => {
                return Err(error(format!(
                    "scales must have dtype Float16 or Float32 for fp16 activations, got {other:?}"
                )));
            }
        };
        require_dtype("Y", outputs[0].dtype, DataType::Float16)?;

        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        if outputs[0].shape != expected_output_shape {
            return Err(error(format!(
                "Y must have shape {expected_output_shape:?}, got {:?}",
                outputs[0].shape
            )));
        }

        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = self.block_size * self.bits / 8;
        require_shape("B", inputs[1].shape, &[self.n, k_blocks, blob_size])?;
        require_flat_or_matrix_shape("scales", inputs[2].shape, self.n, k_blocks)?;

        let zero_points = optional_input(inputs, 3);
        let zp_row_bytes = (k_blocks * self.bits).div_ceil(8);
        if let Some(zero_points) = zero_points {
            require_dtype("zero_points", zero_points.dtype, DataType::Uint8)?;
            require_flat_or_matrix_shape("zero_points", zero_points.shape, self.n, zp_row_bytes)?;
        }
        let group_indices = optional_input(inputs, 4);
        let bias = optional_input(inputs, 5);
        let rows = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        if let Some(bias) = bias {
            require_dtype("bias", bias.dtype, DataType::Float16)?;
            // A folded residual epilogue binds the residual activation into this
            // same slot: `[1, 1, N]` (N elements) at decode, or `[1, S, N]`
            // (rows * N elements) at prefill. A genuine broadcast bias is `[N]`.
            if bias.numel() != self.n && bias.numel() != rows * self.n {
                return Err(error(format!(
                    "bias must have {} elements (broadcast [N]) or {} elements (per-token \
                     [1, S, N] residual), got {:?}",
                    self.n,
                    rows * self.n,
                    bias.shape
                )));
            }
        }
        let gamma = optional_input(inputs, 6);
        if self.rmsnorm_prologue {
            let gamma = gamma.ok_or_else(|| {
                error("rmsnorm_prologue fusion requires the normalization weight at input 6")
            })?;
            require_gamma_dtype(gamma.dtype)?;
            require_shape("gamma", gamma.shape, &[self.k])?;
        }

        for (name, contiguous) in [
            ("A", inputs[0].is_contiguous()),
            ("B", inputs[1].is_contiguous()),
            ("scales", inputs[2].is_contiguous()),
            (
                "zero_points",
                zero_points.is_none_or(TensorView::is_contiguous),
            ),
            ("bias", bias.is_none_or(TensorView::is_contiguous)),
            ("gamma", gamma.is_none_or(TensorView::is_contiguous)),
            ("Y", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(error(format!(
                    "{name} must be contiguous on the CUDA execution provider"
                )));
            }
        }

        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        // Non-block-32 layouts are served by the model-agnostic general-block-size
        // fp16 kernels (int4 decode GEMV + int4/int8 prefill GEMM). The tuned
        // block-32 fusions (rmsnorm prologue, gate/up SwiGLU, down-projection) are
        // gated to block_size==32 in the optimizer, so a non-block-32 node always
        // arrives here as a plain int4 GEMV/GEMM. int8 is restricted to block-32
        // at kernel construction, so bits==8 never reaches the general path.
        if group_indices.is_some() {
            return Err(error(
                "MatMulNBits CUDA fp16 activations do not support g_idx. Why: the block-32 fp16 \
                 kernels map each K block directly to its scale and zero point and do not implement \
                 group remapping. How to fix: omit g_idx, provide f32 activations, or select another \
                 execution provider",
            ));
        }

        if m > 1 {
            // SAFETY: the tiled prefill kernel itself has fixed pointers and no
            // allocation or host synchronization. We nevertheless keep the
            // advertised capture contract conservative: variable-M prefill is
            // outside the persistent M=1 decode graph and has no replay coverage.
            self.last_call_capture_safe.store(false, Ordering::Relaxed);
            // A folded residual epilogue supplies a per-token residual (rows * N
            // elements) in the bias slot; index it with row stride N. A genuine
            // broadcast bias (N elements) keeps stride 0.
            let bias_row_stride = match bias {
                Some(bias) if bias.numel() == m * self.n && m * self.n != self.n => self.n,
                _ => 0,
            };
            if self.rmsnorm_prologue {
                let gamma = gamma.ok_or_else(|| {
                    error("rmsnorm_prologue fusion requires the normalization weight at input 6")
                })?;
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemm_f16_tiled_rmsnorm",
                    "M={} prefill: RMS-normalization prologue (SkipSimplifiedLayerNormalization \
                     folded) into a per-token scratch, then portable 16x16 tiled GEMM with fp32 \
                     accumulation; not advertised as CUDA-graph capture-safe",
                    m
                );
                return self.launch_f16_gemm_rmsnorm_prefill(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    gamma,
                    bias,
                    &mut outputs[0],
                    m,
                    k_blocks,
                    bias_row_stride,
                );
            }
            onnx_runtime_ep_api::record_kernel_variant!(
                "gemm_f16_tiled",
                "M={} prefill: fp16 activation, bits={}, block_size={}, zero_points={}, \
                 scales={} → portable 16x16 CUDA-core tiled GEMM with fp32 accumulation; \
                 not advertised as CUDA-graph capture-safe",
                m,
                self.bits,
                self.block_size,
                zero_points.is_some(),
                if scales_fp16 { "fp16" } else { "fp32" }
            );
            return self.launch_f16_gemm(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                scales_fp16,
                zero_points,
                bias,
                &mut outputs[0],
                m,
                k_blocks,
                blob_size,
                bias_row_stride,
            );
        }

        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        if self.bits == 8 {
            if self.rmsnorm_prologue {
                let gamma = gamma.ok_or_else(|| {
                    error("rmsnorm_prologue fusion requires the normalization weight at input 6")
                })?;
                if !scales_fp16 {
                    return Err(error(
                        "rmsnorm_prologue fusion requires fp16 scales (the fused kernel replicates \
                         the fp16 general scales path)",
                    ));
                }
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_int8_f16_scales_f16_rmsnorm",
                    "M==1 decode: fp16 activation, bits=8, block_size=32, fp16 scales, \
                     zero_points={} → int8 GEMV with fused RMS-normalization prologue \
                     (SkipSimplifiedLayerNormalization folded)",
                    zero_points.is_some()
                );
                return self.launch_int8_f16_gemv_rmsnorm(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    gamma,
                    bias,
                    &mut outputs[0],
                    k_blocks,
                );
            }
            onnx_runtime_ep_api::record_kernel_variant!(
                "gemv_int8_f16",
                "M==1 decode: fp16 activation, bits=8, block_size=32, zero_points={} → direct \
                 capture-safe GEMV",
                zero_points.is_some()
            );
            return self.launch_int8_f16_gemv(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                scales_fp16,
                zero_points,
                bias,
                &mut outputs[0],
                k_blocks,
            );
        }
        if self.rmsnorm_prologue {
            let gamma = gamma.ok_or_else(|| {
                error("rmsnorm_prologue fusion requires the normalization weight at input 6")
            })?;
            if !scales_fp16 {
                return Err(error(
                    "rmsnorm_prologue fusion requires fp16 scales (the fused kernel replicates \
                     the fp16 general scales path)",
                ));
            }
            onnx_runtime_ep_api::record_kernel_variant!(
                "gemv_f16_scales_f16_rmsnorm",
                "M==1 decode: fp16 activation, bits=4, block_size=32, fp16 scales, \
                 zero_points={} → general GEMV with fused RMS-normalization prologue \
                 (SkipSimplifiedLayerNormalization folded)",
                zero_points.is_some()
            );
            return self.launch_f16_gemv_rmsnorm(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                zero_points,
                gamma,
                bias,
                &mut outputs[0],
                k_blocks,
                blob_size,
                zp_row_bytes,
            );
        }
        self.launch_f16_gemv(
            &inputs[0],
            &inputs[1],
            &inputs[2],
            scales_fp16,
            zero_points,
            bias,
            &mut outputs[0],
            k_blocks,
            blob_size,
            zp_row_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_f16_gemm(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
        blob_size: usize,
        bias_row_stride: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 prefill GEMM")?;
        let general_block_size = self.block_size != 32;
        let entry = if general_block_size {
            GEMM_F16_GENERAL_BS_ENTRY
        } else {
            GEMM_F16_ENTRY
        };
        let function = self
            .runtime
            .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let m_i32 = as_i32("M", m)?;
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let bits = as_i32("bits", self.bits)?;
        let scales_fp16_flag = scales_fp16 as i32;
        let bias_post_round_flag: i32 = (self.fold_bias_post_round && bias.is_some()) as i32;
        let bias_row_stride_i32 = as_i32("bias row stride", bias_row_stride)?;
        let block_size_i32 = as_i32("block_size", self.block_size)?;
        let blob_size_i32 = as_i32("block blob size", blob_size)?;
        let grid_x = u32::try_from(self.n.div_ceil(GEMM_F16_TILE))
            .map_err(|_| error(format!("N={} exceeds CUDA prefill grid limits", self.n)))?;
        let grid_y = u32::try_from(m.div_ceil(GEMM_F16_TILE))
            .map_err(|_| error(format!("M={m} exceeds CUDA prefill grid limits")))?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&m_i32)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&bits)
            .arg(&scales_fp16_flag)
            .arg(&bias_post_round_flag)
            .arg(&bias_row_stride_i32);
        // The general-block-size prefill kernel takes two extra trailing scalars
        // (`block_size`, `blob_size`) to derive the packed layout for any block
        // width; the tuned block-32 kernel bakes those in and takes neither.
        if general_block_size {
            builder.arg(&block_size_i32).arg(&blob_size_i32);
        }
        // SAFETY: dense tensors and all dimensions were validated
        // above. The 16x16 CTA uses 4 KiB of statically sized shared memory,
        // ordinary fp32 CUDA-core arithmetic, and fp16 conversions only. It has
        // no tensor-core/PTX/cp.async dependency, so the same path is the
        // portable fallback on every CUDA SM supported by this crate.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (GEMM_F16_TILE as u32, GEMM_F16_TILE as u32, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits fp16 prefill GEMM", err))
    }

    /// M>1 prefill path for the fused RMS-normalization prologue. It stages the
    /// per-token normalized activation (byte-identical to the standalone
    /// `skip_rmsnorm_f16_warp_half4` output) into scratch, then runs the
    /// portable tiled GEMM over it. Prefill is outside the persistent decode
    /// graph, so the scratch allocation here is not on any captured path.
    #[allow(clippy::too_many_arguments)]
    fn launch_f16_gemm_rmsnorm_prefill(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        gamma: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
        bias_row_stride: usize,
    ) -> Result<()> {
        let scratch = self
            .runtime
            .alloc_raw(m * self.k * std::mem::size_of::<half::f16>())?;
        let scratch_shape = [m, self.k];
        let scratch_strides = [self.k as i64, 1];
        let normalized = TensorView::new(
            DevicePtr(raw_ptr(scratch) as *const c_void),
            DataType::Float16,
            &scratch_shape,
            &scratch_strides,
            activation.device,
        );
        let result = self.launch_rmsnorm_prefill(activation, gamma, scratch, m).and_then(|()| {
            self.launch_f16_gemm(
                &normalized,
                packed,
                scales,
                true,
                zero_points,
                bias,
                output,
                m,
                k_blocks,
                self.block_size * self.bits / 8,
                bias_row_stride,
            )
        });
        // SAFETY: `scratch` came from `alloc_raw` above and is freed exactly
        // once; `cuMemFree` waits for the preceding norm + GEMM stream work.
        let free_scratch = unsafe { self.runtime.free_raw(scratch) };
        result.and(free_scratch)
    }

    /// Launches the standalone RMS-normalization prologue used by prefill. One
    /// warp normalizes one token row of `activation` into `normalized`.
    fn launch_rmsnorm_prefill(
        &self,
        activation: &TensorView,
        gamma: &TensorView,
        normalized: CUdeviceptr,
        m: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 RMS-norm prefill prologue")?;
        let function =
            self.runtime
                .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, RMSNORM_PREFILL_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let gamma_ptr = cuptr(gamma.data_ptr::<u8>() as *const c_void);
        let normalized_ptr = normalized;
        let norm_size = as_i32("K", self.k)?;
        let num_groups = as_i32("M", m)?;
        let gamma_is_half: i32 = (gamma.dtype == DataType::Float16) as i32;
        let epsilon = self.rmsnorm_epsilon;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&gamma_ptr)
            .arg(&normalized_ptr)
            .arg(&norm_size)
            .arg(&num_groups)
            .arg(&gamma_is_half)
            .arg(&epsilon);
        // SAFETY: `activation` and `gamma` are validated contiguous fp16 tensors
        // and `normalized` is a `K * M`-half scratch buffer allocated by the
        // caller. Each of the `M` one-warp blocks reads/writes only its own row
        // with the launch-predicate-guaranteed `K % 128 == 0` half4 chunking.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (m as u32, 1, 1),
                block_dim: (RMSNORM_PREFILL_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits fp16 RMS-norm prefill prologue", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_int8_f16_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits int8 fp16 GEMV")?;
        // Grid-starved standalone int8-zp GEMV (e.g. Phi's int8 down projection,
        // grid 384 / ~0.48 waves/SM): when K is a whole multiple of the 256-wide
        // step and the shape uses the 256-thread large path, take the split-K
        // entry (K_SPLIT warps/column, K_SPLIT x larger grid) to fill the SMs.
        // This kernel has no serial prologue, so the extra grid parallelism pays
        // off directly. Symmetric int8 (no zero points) keeps its byte-identical
        // single-warp kernel; the small-shape (64-thread) path lacks the warps to
        // cooperate.
        let use_splitk = zero_points.is_some()
            && self.k.is_multiple_of(256)
            && !(self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX);
        let entry = if use_splitk {
            GEMV_INT8_F16_SPLITK_ENTRY
        } else {
            GEMV_INT8_F16_ENTRY
        };
        let function = self
            .runtime
            .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let scales_fp16_flag = scales_fp16 as i32;
        let bias_post_round_flag: i32 = (self.fold_bias_post_round && bias.is_some()) as i32;
        let threads = if self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX {
            GEMV_F16_SMALL_THREADS
        } else {
            GEMV_F16_LARGE_THREADS
        };
        let columns_per_block = if use_splitk {
            (threads / 32) as usize / GEMV_INT8_F16_SPLITK
        } else {
            (threads / 32) as usize
        };
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&scales_fp16_flag)
            .arg(&bias_post_round_flag);
        // SAFETY: validation restricts this entry to dense fp16 block-32 M=1
        // tensors. The kernel uses fixed device pointers, registers, and warp
        // shuffles only, so it performs no allocation or synchronization and is
        // legal to capture and replay on every CUDA SM supported by this crate.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits int8 fp16 GEMV", err))
    }

    /// INT8 decode GEMV with a fused RMS-normalization prologue. Mirrors
    /// [`Self::launch_f16_gemv_rmsnorm`] but dispatches the int8 sibling kernel,
    /// which shares the RMS reduction / normalized-activation staging bit-for-bit
    /// and swaps in the block-32 int8 dequant dot. Restricted to fp16 scales and
    /// block-32, matching the fusion's eligibility gates.
    #[allow(clippy::too_many_arguments)]
    fn launch_int8_f16_gemv_rmsnorm(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        gamma: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits int8 fp16 RMS-norm-prologue GEMV")?;
        if bias.is_some() {
            if self.fold_bias_post_round {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "qkv_bias_fused",
                    "folded standalone Add(MatMulNBits, bias) into GEMV epilogue with \
                     fp16-after-round semantics fp16(fp16(acc)+bias) (token-identity preserved)"
                );
            } else {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "bias_native",
                    "native MatMulNBits bias: single-round epilogue fp16(acc+bias)"
                );
            }
        }
        let entry = if zero_points.is_some() {
            GEMV_INT8_F16_SCALES_F16_RMSNORM_ZP_ENTRY
        } else {
            GEMV_INT8_F16_SCALES_F16_RMSNORM_ENTRY
        };
        let function = self.runtime.nvrtc_function(
            GEMV_F16_MODULE,
            GEMV_F16_SRC,
            entry,
        )?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let gamma_ptr = cuptr(gamma.data_ptr::<u8>() as *const c_void);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let bias_post_round_flag: i32 = (self.fold_bias_post_round && bias.is_some()) as i32;
        let gamma_is_half: i32 = (gamma.dtype == DataType::Float16) as i32;
        let epsilon = self.rmsnorm_epsilon;
        let threads = if self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX {
            GEMV_F16_SMALL_THREADS
        } else {
            GEMV_F16_LARGE_THREADS
        };
        let columns_per_block = (threads / 32) as usize;
        let shared_mem_bytes = (self.k * std::mem::size_of::<half::f16>()) as u32;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&gamma_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&bias_post_round_flag)
            .arg(&gamma_is_half)
            .arg(&epsilon);
        // SAFETY: restricted to block-32 M=1 fp16 inputs with fp16 scales, all
        // dtype/shape/contiguity validated above. The kernel stages the
        // normalized activation in launch-time dynamic shared memory
        // (`K * sizeof(f16)`, bounded by the fusion's `K % 128 == 0` predicate)
        // and uses only registers, warp shuffles, and `__syncthreads` — no
        // per-call allocation or host synchronization — so it is legal to record
        // into and replay from a CUDA graph.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits int8 fp16 RMS-norm-prologue GEMV", err))
    }

    /// Paired gate/up projection + SwiGLU path (see
    /// [`crate::optimizer::CudaGateUpSwiGluFusion`]). Inputs are
    /// `[x, W_gate, scales_gate, W_up, scales_up]`. M=1 keeps the paired decode
    /// GEMV unchanged; M>1 reuses the portable tiled prefill GEMM for both
    /// projections before applying the existing fp16 `silu(gate)*up` kernel.
    fn run_f16_gate_up_swiglu(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
    ) -> Result<()> {
        // Contract from `CudaGateUpSwiGluFusion`:
        //   [x, W_gate, scales_gate, W_up, scales_up, (gamma?)@5, (zp_gate?)@6, (zp_up?)@7]
        // Slot 5 carries the RMS-norm gamma when the skip-rmsnorm prologue is
        // folded in; slots 6/7 carry per-projection asymmetric zero points
        // (both present for asymmetric weights, both absent for symmetric ones).
        if !(5..=8).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(error(format!(
                "gate/up SwiGLU fusion expects 5 to 8 inputs [x, W_gate, scales_gate, W_up, \
                 scales_up, (gamma), (zp_gate, zp_up)] and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        require_dtype("A", inputs[0].dtype, DataType::Float16)?;
        require_dtype("W_gate", inputs[1].dtype, DataType::Uint8)?;
        require_dtype("scales_gate", inputs[2].dtype, DataType::Float16)?;
        require_dtype("W_up", inputs[3].dtype, DataType::Uint8)?;
        require_dtype("scales_up", inputs[4].dtype, DataType::Float16)?;
        require_dtype("Y", outputs[0].dtype, DataType::Float16)?;
        let gamma = if self.rmsnorm_prologue {
            let gamma = optional_input(inputs, 5).ok_or_else(|| {
                error("rmsnorm_prologue fusion requires the normalization weight at input 5")
            })?;
            require_gamma_dtype(gamma.dtype)?;
            require_shape("gamma", gamma.shape, &[self.k])?;
            if !gamma.is_contiguous() {
                return Err(error(
                    "gamma must be contiguous on the CUDA execution provider".to_string(),
                ));
            }
            Some(gamma)
        } else {
            None
        };

        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        if outputs[0].shape != expected_output_shape {
            return Err(error(format!(
                "Y must have shape {expected_output_shape:?}, got {:?}",
                outputs[0].shape
            )));
        }

        if self.block_size != 32 || self.bits != 4 {
            return Err(error(format!(
                "gate/up SwiGLU fusion received bits={} and block_size={}. Why: the fused fp16 \
                 path implements the block-32 packed int4 layout. How to fix: export 4-bit \
                 MatMulNBits weights with block_size=32 or disable this fusion",
                self.bits, self.block_size
            )));
        }
        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = self.block_size / 2;
        require_shape("W_gate", inputs[1].shape, &[self.n, k_blocks, blob_size])?;
        require_shape("W_up", inputs[3].shape, &[self.n, k_blocks, blob_size])?;
        require_flat_or_matrix_shape("scales_gate", inputs[2].shape, self.n, k_blocks)?;
        require_flat_or_matrix_shape("scales_up", inputs[4].shape, self.n, k_blocks)?;

        // Optional asymmetric zero points (slots 6/7). Symmetric weights omit
        // both and the kernels apply the implicit `zp == 8` subtrahend, matching
        // the historical byte-identical path. Require them paired: mixing a
        // zero-point projection with a symmetric one is never valid.
        let zp_gate = optional_input(inputs, 6);
        let zp_up = optional_input(inputs, 7);
        if zp_gate.is_some() != zp_up.is_some() {
            return Err(error(
                "gate/up SwiGLU fusion requires zero points for both projections or neither"
                    .to_string(),
            ));
        }
        let zp_row_bytes = (k_blocks * self.bits).div_ceil(8);
        for (name, zp) in [("zp_gate", zp_gate), ("zp_up", zp_up)] {
            if let Some(zp) = zp {
                require_dtype(name, zp.dtype, DataType::Uint8)?;
                require_flat_or_matrix_shape(name, zp.shape, self.n, zp_row_bytes)?;
                if !zp.is_contiguous() {
                    return Err(error(format!(
                        "{name} must be contiguous on the CUDA execution provider"
                    )));
                }
            }
        }

        for (name, contiguous) in [
            ("A", inputs[0].is_contiguous()),
            ("W_gate", inputs[1].is_contiguous()),
            ("scales_gate", inputs[2].is_contiguous()),
            ("W_up", inputs[3].is_contiguous()),
            ("scales_up", inputs[4].is_contiguous()),
            ("Y", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(error(format!(
                    "{name} must be contiguous on the CUDA execution provider"
                )));
            }
        }

        if m == 0 {
            self.last_call_capture_safe.store(false, Ordering::Relaxed);
            onnx_runtime_ep_api::record_kernel_variant!(
                "gate_up_swiglu_empty",
                "M=0 gate/up SwiGLU has an empty output and requires no CUDA launch"
            );
            return Ok(());
        }
        if m > 1 {
            self.last_call_capture_safe.store(false, Ordering::Relaxed);
            if let Some(gamma) = gamma {
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gate_up_swiglu_rmsnorm_prefill",
                    "M={} prefill: RMS-normalization prologue into scratch, then two portable \
                     block-32 int4 fp16 tiled GEMMs with fp32 accumulation, followed by fp16 \
                     SiluMul; not advertised as CUDA-graph capture-safe",
                    m
                );
                return self.launch_gate_up_swiglu_rmsnorm_prefill(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &inputs[3],
                    &inputs[4],
                    zp_gate,
                    zp_up,
                    gamma,
                    &mut outputs[0],
                    m,
                    k_blocks,
                    zp_row_bytes,
                );
            }
            onnx_runtime_ep_api::record_kernel_variant!(
                "gate_up_swiglu_prefill",
                "M={} prefill: two portable block-32 int4 fp16 tiled GEMMs with fp32 \
                 accumulation, followed by fp16 SiluMul; not advertised as CUDA-graph \
                 capture-safe",
                m
            );
            return self.launch_gate_up_swiglu_prefill(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                &inputs[3],
                &inputs[4],
                zp_gate,
                zp_up,
                &mut outputs[0],
                m,
                k_blocks,
            );
        }

        if let Some(gamma) = gamma {
            onnx_runtime_ep_api::record_kernel_variant!(
                "gate_up_swiglu_rmsnorm_fused",
                "fp16 block-32 M==1 decode: fused RMS-normalization prologue + paired gate/up \
                 int4 GEMV + SwiGLU (silu(gate)*up) in one capture-safe kernel; the RMS \
                 reduction runs once for both projections and reproduces the standalone norm \
                 plus two-op fp16 rounding for byte-identical greedy tokens"
            );
            self.last_call_capture_safe.store(true, Ordering::Relaxed);
            return self.launch_gate_up_swiglu_rmsnorm(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                &inputs[3],
                &inputs[4],
                zp_gate,
                zp_up,
                gamma,
                &mut outputs[0],
                k_blocks,
                blob_size,
                zp_row_bytes,
            );
        }

        onnx_runtime_ep_api::record_kernel_variant!(
            "gate_up_swiglu_fused",
            "fp16 block-32 M==1 decode: fused paired gate/up int4 GEMV + SwiGLU \
             (silu(gate)*up) in one capture-safe kernel; reproduces the two-op fp16 \
             rounding for byte-identical greedy tokens"
        );

        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        self.launch_gate_up_swiglu(
            &inputs[0],
            &inputs[1],
            &inputs[2],
            &inputs[3],
            &inputs[4],
            zp_gate,
            zp_up,
            &mut outputs[0],
            k_blocks,
            blob_size,
            zp_row_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_swiglu_prefill(
        &self,
        activation: &TensorView,
        packed_gate: &TensorView,
        scales_gate: &TensorView,
        packed_up: &TensorView,
        scales_up: &TensorView,
        zp_gate: Option<&TensorView>,
        zp_up: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
    ) -> Result<()> {
        let scratch = self.runtime.alloc_raw(output.byte_size())?;
        let scratch_shape = output.shape.to_vec();
        let scratch_strides = output.strides.to_vec();
        let mut gate_output = TensorMut::new(
            DevicePtrMut(raw_ptr(scratch)),
            DataType::Float16,
            &scratch_shape,
            &scratch_strides,
            output.device,
        );

        let result = (|| {
            self.launch_f16_gemm(
                activation,
                packed_gate,
                scales_gate,
                true,
                zp_gate,
                None,
                &mut gate_output,
                m,
                k_blocks,
                self.block_size * self.bits / 8,
                0,
            )?;
            self.launch_f16_gemm(
                activation, packed_up, scales_up, true, zp_up, None, output, m, k_blocks,
                self.block_size * self.bits / 8, 0,
            )?;
            let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
            crate::kernels::elementwise::launch_silu_mul_f16_raw(
                &self.runtime,
                scratch,
                output_ptr,
                output_ptr,
                output.numel(),
            )
        })();

        // Always release the prefill gate projection. `cuMemFree` waits for
        // preceding stream work that uses this allocation, including SiluMul.
        // SAFETY: `scratch` came from `alloc_raw` above and is freed exactly once.
        let free_scratch = unsafe { self.runtime.free_raw(scratch) };
        result.and(free_scratch)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_swiglu(
        &self,
        activation: &TensorView,
        packed_gate: &TensorView,
        scales_gate: &TensorView,
        packed_up: &TensorView,
        scales_up: &TensorView,
        zp_gate: Option<&TensorView>,
        zp_up: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 gate/up SwiGLU GEMV")?;
        // Symmetric weights launch the `HasZp == false` entry, whose PTX drops the
        // per-block zero-point load entirely; only asymmetric weights pay for it.
        let entry = if zp_gate.is_some() || zp_up.is_some() {
            GATE_UP_SWIGLU_ZP_ENTRY
        } else {
            GATE_UP_SWIGLU_ENTRY
        };
        let function =
            self.runtime
                .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_gate_ptr = cuptr(packed_gate.data_ptr::<u8>() as *const c_void);
        let scales_gate_ptr = cuptr(scales_gate.data_ptr::<u8>() as *const c_void);
        let packed_up_ptr = cuptr(packed_up.data_ptr::<u8>() as *const c_void);
        let scales_up_ptr = cuptr(scales_up.data_ptr::<u8>() as *const c_void);
        let zp_gate_ptr = zp_gate
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let zp_up_ptr = zp_up
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row byte count", zp_row_bytes)?;
        let threads = GATE_UP_SWIGLU_THREADS;
        let columns_per_block = (threads / 32) as usize;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_gate_ptr)
            .arg(&scales_gate_ptr)
            .arg(&packed_up_ptr)
            .arg(&scales_up_ptr)
            .arg(&zp_gate_ptr)
            .arg(&zp_up_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes);
        // SAFETY: restricted to fp16 block-32 M=1 inputs validated above; both
        // persistent weight/scale sets and the output are fixed device pointers,
        // the scalar ABI matches the paired entry point, and the kernel uses only
        // registers + warp shuffles (no per-call alloc, shared memory, or sync),
        // so the launch is legal to record into and replay from a CUDA graph.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits fp16 gate/up SwiGLU GEMV", err))
    }

    /// Decode (M==1) paired gate/up SwiGLU with a fused RMS-normalization
    /// prologue. The block reduces the shared activation once and stages the
    /// normalized activation in launch-time dynamic shared memory (`K *
    /// sizeof(f16)`, bounded by the fusion's `K % 128 == 0` predicate), then
    /// both projections read that single staged copy. Fixing the reduction to a
    /// single pass is what recovers the double-recompute cost the fan-out-2
    /// post-attention norm otherwise pays.
    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_swiglu_rmsnorm(
        &self,
        activation: &TensorView,
        packed_gate: &TensorView,
        scales_gate: &TensorView,
        packed_up: &TensorView,
        scales_up: &TensorView,
        zp_gate: Option<&TensorView>,
        zp_up: Option<&TensorView>,
        gamma: &TensorView,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 gate/up SwiGLU RMS-norm GEMV")?;
        let entry = if zp_gate.is_some() || zp_up.is_some() {
            GATE_UP_SWIGLU_RMSNORM_ZP_ENTRY
        } else {
            GATE_UP_SWIGLU_RMSNORM_ENTRY
        };
        let function = self.runtime.nvrtc_function(
            GEMV_F16_MODULE,
            GEMV_F16_SRC,
            entry,
        )?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_gate_ptr = cuptr(packed_gate.data_ptr::<u8>() as *const c_void);
        let scales_gate_ptr = cuptr(scales_gate.data_ptr::<u8>() as *const c_void);
        let packed_up_ptr = cuptr(packed_up.data_ptr::<u8>() as *const c_void);
        let scales_up_ptr = cuptr(scales_up.data_ptr::<u8>() as *const c_void);
        let zp_gate_ptr = zp_gate
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let zp_up_ptr = zp_up
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let gamma_ptr = cuptr(gamma.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row byte count", zp_row_bytes)?;
        let gamma_is_half: i32 = (gamma.dtype == DataType::Float16) as i32;
        let epsilon = self.rmsnorm_epsilon;
        let threads = GATE_UP_SWIGLU_THREADS;
        let columns_per_block = (threads / 32) as usize;
        let shared_mem_bytes = (self.k * std::mem::size_of::<half::f16>()) as u32;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_gate_ptr)
            .arg(&scales_gate_ptr)
            .arg(&packed_up_ptr)
            .arg(&scales_up_ptr)
            .arg(&zp_gate_ptr)
            .arg(&zp_up_ptr)
            .arg(&gamma_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes)
            .arg(&gamma_is_half)
            .arg(&epsilon);
        // SAFETY: restricted to fp16 block-32 M=1 inputs validated above; the
        // weight/scale/gamma sets and the output are fixed device pointers, the
        // scalar ABI matches the paired RMS-norm entry point, and the kernel
        // stages the normalized activation in launch-time dynamic shared memory
        // (`K * sizeof(f16)`, bounded by the fusion's `K % 128 == 0` predicate)
        // using only registers, warp shuffles, and `__syncthreads` — no per-call
        // allocation or host sync — so it is legal to record into and replay
        // from a CUDA graph.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits fp16 gate/up SwiGLU RMS-norm GEMV", err))
    }

    /// Prefill (M>1) gate/up SwiGLU with an RMS-normalization prologue.
    /// Normalizes each token row into scratch (byte-identical to
    /// `skip_rmsnorm_f16_warp_half4`), then runs the standard paired gate/up
    /// SwiGLU prefill over the normalized activation. Prefill is outside the
    /// persistent decode graph, so the scratch allocation is not on any captured
    /// path.
    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_swiglu_rmsnorm_prefill(
        &self,
        activation: &TensorView,
        packed_gate: &TensorView,
        scales_gate: &TensorView,
        packed_up: &TensorView,
        scales_up: &TensorView,
        zp_gate: Option<&TensorView>,
        zp_up: Option<&TensorView>,
        gamma: &TensorView,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
        _zp_row_bytes: usize,
    ) -> Result<()> {
        let scratch = self
            .runtime
            .alloc_raw(m * self.k * std::mem::size_of::<half::f16>())?;
        let scratch_shape = [m, self.k];
        let scratch_strides = [self.k as i64, 1];
        let normalized = TensorView::new(
            DevicePtr(raw_ptr(scratch) as *const c_void),
            DataType::Float16,
            &scratch_shape,
            &scratch_strides,
            activation.device,
        );
        let result = self
            .launch_rmsnorm_prefill(activation, gamma, scratch, m)
            .and_then(|()| {
                self.launch_gate_up_swiglu_prefill(
                    &normalized,
                    packed_gate,
                    scales_gate,
                    packed_up,
                    scales_up,
                    zp_gate,
                    zp_up,
                    output,
                    m,
                    k_blocks,
                )
            });
        // SAFETY: `scratch` came from `alloc_raw` above and is freed exactly
        // once; `cuMemFree` waits for the preceding norm + GEMM stream work.
        let free_scratch = unsafe { self.runtime.free_raw(scratch) };
        result.and(free_scratch)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_f16_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let selection = select_f16_gemv_variant(
            self.k,
            self.n,
            self.block_size,
            scales_fp16,
            zero_points.is_some(),
        );
        // Non-block-32 layouts are served by the model-agnostic general_bs
        // kernel; tag them distinctly so nsys/trace timelines can tell the
        // block-size-general decode GEMV apart from a tuned block-32 general one.
        let variant_name = if self.block_size != 32 {
            "gemv_f16_general_bs"
        } else {
            match selection.variant {
                F16GemvVariant::DownProjection => "gemv_f16_down_projection",
                F16GemvVariant::General => "gemv_f16_general",
            }
        };
        onnx_runtime_ep_api::record_kernel_variant!(
            variant_name,
            "fp16-activation x int4 M==1 decode GEMV: block_size={}; zero_points={}; {}",
            self.block_size,
            zero_points.is_some(),
            selection.reason
        );
        if bias.is_some() {
            if self.fold_bias_post_round {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "qkv_bias_fused",
                    "folded standalone Add(MatMulNBits, bias) into GEMV epilogue with \
                     fp16-after-round semantics fp16(fp16(acc)+bias) (token-identity preserved)"
                );
            } else {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "bias_native",
                    "native MatMulNBits bias: single-round epilogue fp16(acc+bias)"
                );
            }
        }
        self.launch_f16_gemv_variant(
            activation,
            packed,
            scales,
            scales_fp16,
            zero_points,
            bias,
            output,
            k_blocks,
            blob_size,
            zp_row_bytes,
            selection,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_f16_gemv_variant(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
        selection: F16GemvSelection,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 GEMV")?;
        // Split-K routing for the standalone asymmetric int4 GEMV: only the
        // block-32, scales-fp16, General-variant zp path is grid-starved enough to
        // benefit, and the split-K kernel assumes whole 256-wide K steps (no tail).
        let use_scales_f16_zp_splitk = self.block_size == 32
            && scales_fp16
            && zero_points.is_some()
            && matches!(selection.variant, F16GemvVariant::General)
            && self.k.is_multiple_of(256)
            // Split-K needs >= K_SPLIT warps/block; the small-shape path uses only
            // 64 threads (2 warps), so restrict to the 256-thread large path.
            && !(self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX);
        let entry = if self.block_size != 32 {
            // Any non-block-32 layout uses the model-agnostic general kernel; the
            // tuned DownProjection / scales_f16 / general entries bake in the
            // block-32 lane→block mapping. `select_f16_gemv_variant` already
            // returns `General` for block_size != 32, so shape/thread selection
            // below (the `General` arm) applies unchanged. The general_bs kernel
            // dequantizes an optional asymmetric zero point per block, so it is
            // correct for both symmetric (zp==8) and asymmetric layouts.
            GEMV_F16_GENERAL_BS_ENTRY
        } else {
            match selection.variant {
                F16GemvVariant::DownProjection => GEMV_F16_DOWN_ENTRY,
                // The vectorized `scales_f16` kernel is compiled in two
                // specializations: the symmetric entry (`HasZp == false`) folds
                // the subtrahend to the constant fp16 8.0 with zero extra memory
                // traffic (byte-identical PTX to the pre-zero-point path), while
                // the `_zp` entry reads the per-block asymmetric zero point.
                // Symmetric weights must take the constant path so the memory-
                // bound M=1 decode GEMV does not pay for an unused per-block load.
                F16GemvVariant::General if scales_fp16 => {
                    if zero_points.is_some() {
                        // Grid-starved standalone int4 zp GEMV: when K is a whole
                        // multiple of the 256-wide step, take the split-K entry
                        // (K_SPLIT warps/column, K_SPLIT x larger grid) to fill the
                        // SMs; otherwise the plain single-warp `_zp` entry (which
                        // handles the divergent K tail).
                        if use_scales_f16_zp_splitk {
                            GEMV_F16_SCALES_F16_ZP_SPLITK_ENTRY
                        } else {
                            GEMV_F16_SCALES_F16_ZP_ENTRY
                        }
                    } else {
                        GEMV_F16_SCALES_F16_ENTRY
                    }
                }
                F16GemvVariant::General => GEMV_F16_ENTRY,
            }
        };
        let function = self
            .runtime
            .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row byte count", zp_row_bytes)?;
        let scales_fp16_flag: i32 = scales_fp16 as i32;
        let bias_post_round_flag: i32 = (self.fold_bias_post_round && bias.is_some()) as i32;
        let (threads, columns_per_block, shared_mem_bytes) = match selection.variant {
            F16GemvVariant::DownProjection => (
                GEMV_F16_DOWN_THREADS,
                GEMV_F16_DOWN_COLUMNS_PER_BLOCK,
                0,
            ),
            F16GemvVariant::General => {
                let threads = if self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX {
                    GEMV_F16_SMALL_THREADS
                } else {
                    GEMV_F16_LARGE_THREADS
                };
                // Split-K assigns K_SPLIT warps per output column, so a block of
                // `threads/32` warps now covers `warps / K_SPLIT` columns and the
                // grid grows by K_SPLIT to fill the SMs.
                let columns_per_block = if use_scales_f16_zp_splitk {
                    (threads / 32) as usize / GEMV_F16_SCALES_F16_ZP_SPLITK
                } else {
                    (threads / 32) as usize
                };
                (threads, columns_per_block, 0)
            }
        };
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&block_size)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes)
            .arg(&scales_fp16_flag)
            .arg(&bias_post_round_flag);
        // SAFETY: M=1 fp16 inputs; all tensors were dtype/shape/contiguity
        // validated above, including the optional packed per-block zero-point
        // rows. Block-32 layouts use the tuned entries; any other (power-of-two,
        // >=16) block size routes to the general_bs entry, which derives the
        // scale/zero-point block index from `block_size`. The scalar ABI is
        // shared by all these entries. Every variant uses only registers and
        // launch-time shared memory (no per-call alloc or sync), so the launch
        // is legal to record into and replay from a CUDA graph.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes,
            })
        }
        .map(|_| ())
        .map_err(|err| {
            driver_err(
                &format!("launch MatMulNBits fp16 GEMV ({})", selection.reason),
                err,
            )
        })
    }

    /// General fp16 GEMV whose input activation is RMS-normalized in-kernel
    /// (the [`GEMV_F16_SCALES_F16_RMSNORM_ENTRY`] entry). Bit-for-bit identical
    /// to the standalone `SkipSimplifiedLayerNormalization` residual output
    /// followed by the general `scales_f16` GEMV, so decode tokens are
    /// unchanged while the separate normalization launch is removed.
    #[allow(clippy::too_many_arguments)]
    fn launch_f16_gemv_rmsnorm(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        gamma: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 RMS-norm-prologue GEMV")?;
        if bias.is_some() {
            if self.fold_bias_post_round {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "qkv_bias_fused",
                    "folded standalone Add(MatMulNBits, bias) into GEMV epilogue with \
                     fp16-after-round semantics fp16(fp16(acc)+bias) (token-identity preserved)"
                );
            } else {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "bias_native",
                    "native MatMulNBits bias: single-round epilogue fp16(acc+bias)"
                );
            }
        }
        let entry = if zero_points.is_some() {
            GEMV_F16_SCALES_F16_RMSNORM_ZP_ENTRY
        } else {
            GEMV_F16_SCALES_F16_RMSNORM_ENTRY
        };
        let function = self.runtime.nvrtc_function(
            GEMV_F16_MODULE,
            GEMV_F16_SRC,
            entry,
        )?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let gamma_ptr = cuptr(gamma.data_ptr::<u8>() as *const c_void);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row byte count", zp_row_bytes)?;
        let bias_post_round_flag: i32 = (self.fold_bias_post_round && bias.is_some()) as i32;
        let gamma_is_half: i32 = (gamma.dtype == DataType::Float16) as i32;
        let epsilon = self.rmsnorm_epsilon;
        let threads = if self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX {
            GEMV_F16_SMALL_THREADS
        } else {
            GEMV_F16_LARGE_THREADS
        };
        let columns_per_block = (threads / 32) as usize;
        let shared_mem_bytes = (self.k * std::mem::size_of::<half::f16>()) as u32;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&gamma_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes)
            .arg(&bias_post_round_flag)
            .arg(&gamma_is_half)
            .arg(&epsilon);
        // SAFETY: restricted to block-32 M=1 fp16 inputs with fp16 scales and no
        // zero_points, all dtype/shape/contiguity validated above. The kernel
        // stages the normalized activation in launch-time dynamic shared memory
        // (`K * sizeof(f16)`, bounded by the fusion's `K % 128 == 0` predicate)
        // and uses only registers, warp shuffles, and `__syncthreads` — no
        // per-call allocation or host synchronization — so it is legal to record
        // into and replay from a CUDA graph.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits fp16 RMS-norm-prologue GEMV", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_int8_f32_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
    ) -> Result<()> {
        let function = self
            .runtime
            .nvrtc_function(GEMV_MODULE, GEMV_SRC, GEMV_INT8_F32_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks);
        // SAFETY: dense tensors were validated for the one-byte-per-weight
        // block-32 layout. This fixed-geometry launch uses no dynamic allocation,
        // host synchronization, or architecture-specific instructions, so it is
        // CUDA-graph-capturable and portable across supported SM versions.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n as u32, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits int8 f32 GEMV", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_f32_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let function = self
            .runtime
            .nvrtc_function(GEMV_MODULE, GEMV_SRC, GEMV_F32_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row size", zp_row_bytes)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&block_size)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes);
        // SAFETY: validated dense tensors cover the complete M=1 operation and
        // the scalar ABI matches `matmul_nbits_gemv_f32`.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n as u32, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits f32 GEMV", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_accuracy4_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
    ) -> Result<()> {
        let workspace = self
            .accuracy4_workspace
            .as_ref()
            .ok_or_else(|| error("accuracy_level=4 GEMV workspace is unavailable"))?
            .lock()
            .map_err(|_| error("accuracy_level=4 GEMV workspace lock poisoned"))?;
        let quantize_function =
            self.runtime
                .nvrtc_function(GEMV_MODULE, GEMV_SRC, QUANTIZE_ACCURACY4_ENTRY)?;
        let gemv_function =
            self.runtime
                .nvrtc_function(GEMV_MODULE, GEMV_SRC, GEMV_ACCURACY4_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let padded_k = as_i32("padded K", workspace.padded_k)?;

        let mut quantize_builder = self.runtime.stream().launch_builder(&quantize_function);
        quantize_builder
            .arg(&activation_ptr)
            .arg(&workspace.quantized_activation)
            .arg(&workspace.activation_scale)
            .arg(&k)
            .arg(&padded_k);
        // SAFETY: the persistent workspace covers padded_k int8 values plus the
        // f32 scale, and the scalar ABI matches the quantization entry point.
        unsafe {
            quantize_builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|err| driver_err("launch MatMulNBits accuracy_level=4 quantization", err))?;

        let mut gemv_builder = self.runtime.stream().launch_builder(&gemv_function);
        gemv_builder
            .arg(&workspace.quantized_activation)
            .arg(&workspace.activation_scale)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks);
        // SAFETY: this path is restricted to symmetric block-32 M=1 inputs; the
        // persistent quantized activation is initialized by the preceding stream
        // launch, and the scalar ABI matches the tiled GEMV entry point.
        unsafe {
            gemv_builder.launch(LaunchConfig {
                grid_dim: (
                    self.n.div_ceil(GEMV_ACCURACY4_COLUMNS_PER_BLOCK) as u32,
                    1,
                    1,
                ),
                block_dim: (GEMV_ACCURACY4_THREADS, 1, 1),
                shared_mem_bytes: GEMV_ACCURACY4_SHARED_BYTES,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits accuracy_level=4 GEMV", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_accuracy4(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let total = m.checked_mul(self.n).ok_or_else(|| {
            error(format!(
                "accuracy_level=4 output size {m} * {} overflows usize",
                self.n
            ))
        })?;
        let blocks = total.div_ceil(BLOCK_THREADS as usize).clamp(1, 65_535) as u32;
        let function =
            self.runtime
                .nvrtc_function(ACCURACY4_MODULE, ACCURACY4_SRC, ACCURACY4_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let m = as_i32("M", m)?;
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row size", zp_row_bytes)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&block_size)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes);
        // SAFETY: all tensors were dtype/shape/contiguity validated above and
        // the scalar ABI matches `matmul_nbits_accuracy4`.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits accuracy_level=4", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_dequant(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        group_indices: Option<&TensorView>,
        weight: cudarc::driver::sys::CUdeviceptr,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let group_indices_ptr = group_indices
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row size", zp_row_bytes)?;
        let bits = as_i32("bits", self.bits)?;
        let total = self.k * self.n;
        let blocks = total.div_ceil(BLOCK_THREADS as usize).clamp(1, 65_535) as u32;
        let function = self
            .runtime
            .nvrtc_function(DEQUANT_MODULE, DEQUANT_SRC, DEQUANT_ENTRY)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&group_indices_ptr)
            .arg(&weight)
            .arg(&k)
            .arg(&n)
            .arg(&block_size)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes)
            .arg(&bits);
        // SAFETY: argument order/types match the CUDA entry point; all device
        // buffers were shape-validated and `weight` has K*N f32 elements.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits dequant", err))
    }
}

impl Kernel for MatMulNBitsKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        // The m=1 no-g_idx GEMV uses only launch-time shared memory and a
        // shape-fixed persistent accuracy-4 activation workspace; it performs no
        // per-call allocation, D2H, or synchronization. The direct fp16 GEMV is
        // likewise capture-safe: fixed grid/block geometry from the shape
        // signature and register/launch-time-shared scratch. The allocation-free
        // fp16 prefill GEMM remains conservatively unadvertised because variable-M
        // prefill is outside the persistent decode graph and lacks replay coverage;
        // f32 prefill scratch and g_idx validation are also non-capturable.
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "requires M==1 decode GEMV without group_indices; prefill is outside the advertised capture contract and group_indices validation reads D2H",
            )
        }
    }
}

fn optional_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn required_positive_attr(node: &Node, name: &str) -> Result<usize> {
    let value = optional_int_attr(node, name)?
        .ok_or_else(|| error(format!("missing required integer attribute '{name}'")))?;
    if value <= 0 {
        return Err(error(format!(
            "attribute '{name}' must be positive, got {value}"
        )));
    }
    Ok(value as usize)
}

fn optional_int_attr(node: &Node, name: &str) -> Result<Option<i64>> {
    match node.attr(name) {
        Some(attribute) => attribute
            .as_int()
            .map(Some)
            .ok_or_else(|| error(format!("attribute '{name}' must be an integer"))),
        None => Ok(None),
    }
}

fn require_dtype(name: &str, got: DataType, expected: DataType) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have dtype {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

// The RMS-norm prologue accepts an fp16 OR fp32 gamma. Gamma is only a final
// multiplicand (never in the fp32 variance accumulation), so an fp32 gamma is
// numerically safe and lets fp32-gamma exports (e.g. Phi-4-mini) fuse. The
// fused kernels branch on `gamma_is_half` to read gamma at full precision.
fn require_gamma_dtype(got: DataType) -> Result<()> {
    if got != DataType::Float16 && got != DataType::Float32 {
        return Err(error(format!(
            "gamma must have dtype Float16 or Float, got {got:?}"
        )));
    }
    Ok(())
}

fn require_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have shape {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn require_flat_or_matrix_shape(
    name: &str,
    got: &[usize],
    rows: usize,
    columns: usize,
) -> Result<()> {
    if got != [rows * columns] && got != [rows, columns] {
        return Err(error(format!(
            "{name} must have shape [{}] or [{rows}, {columns}], got {got:?}",
            rows * columns
        )));
    }
    Ok(())
}

fn as_i32(name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| error(format!("{name}={value} exceeds i32")))
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep MatMulNBits: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use half::f16;

    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
    use onnx_runtime_ir::{DataType, DeviceId};

    use super::*;

    // Qwen2.5-0.5B down-projection shape (K=intermediate, N=hidden). Used as a
    // test fixture for the tall-skinny down variant and, transposed, as the
    // gate/up shape — the runtime code never keys on these values.
    const QWEN_DOWN_K: usize = 4864;
    const QWEN_DOWN_N: usize = 896;
    const STAGED_DOWN_REFERENCE_ENTRY: &str =
        "matmul_nbits_gemv_f16_scales_f16_down_staged_reference";
    const STAGED_DOWN_REFERENCE_SRC: &str = r#"
extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_down_staged_reference(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    (void)block_size;
    (void)zero_points;
    (void)zp_row_bytes;
    (void)scales_fp16;
    extern __shared__ uint4 activation_shared[];
    __shared__ float warp_sums[8][8];
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int column_base = (int)blockIdx.x * 8;

    for (int vector = tid; vector * 8 < k; vector += (int)blockDim.x) {
        activation_shared[vector] =
            permute_activation_f16x8(activation + vector * 8);
    }
    __syncthreads();

    float values[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int block = tid; block < k_blocks; block += (int)blockDim.x) {
        const uint4 activation0 = activation_shared[block * 4];
        const uint4 activation1 = activation_shared[block * 4 + 1];
        const uint4 activation2 = activation_shared[block * 4 + 2];
        const uint4 activation3 = activation_shared[block * 4 + 3];
#pragma unroll
        for (int tile_column = 0; tile_column < 8; ++tile_column) {
            const int column = column_base + tile_column;
            if (column < n) {
                const long packed_start =
                    ((long)column * k_blocks + block) * blob_size;
                const uint4 packed_weights =
                    *reinterpret_cast<const uint4*>(packed + packed_start);
                const __half scale = scales[(long)column * k_blocks + block];
                values[tile_column] += dot_int4x32_f16_permuted_scaled(
                    packed_weights,
                    activation0,
                    activation1,
                    activation2,
                    activation3,
                    scale);
            }
        }
    }

#pragma unroll
    for (int tile_column = 0; tile_column < 8; ++tile_column) {
        const float value = warp_sum(values[tile_column]);
        if (lane == 0) {
            warp_sums[warp][tile_column] = value;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < 8) {
        const int column = column_base + lane;
        float value = warp_sums[0][lane];
        value += warp_sums[1][lane];
        value += warp_sums[2][lane];
        value += warp_sums[3][lane];
        value += warp_sums[4][lane];
        value += warp_sums[5][lane];
        value += warp_sums[6][lane];
        value += warp_sums[7][lane];
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}
"#;

    fn runtime() -> Option<Arc<CudaRuntime>> {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let runtime = std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten();
        std::panic::set_hook(previous_hook);
        runtime
    }

    fn as_bytes<T: Copy>(values: &[T]) -> &[u8] {
        // SAFETY: reinterpreting a POD slice as raw bytes for a host->device copy.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    fn as_bytes_mut<T: Copy>(values: &mut [T]) -> &mut [u8] {
        // SAFETY: reinterpreting a POD slice as raw bytes for a device->host copy.
        unsafe {
            std::slice::from_raw_parts_mut(
                values.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(values),
            )
        }
    }

    fn device_ptr(raw: CUdeviceptr) -> DevicePtr {
        DevicePtr(raw as usize as *const c_void)
    }

    fn device_ptr_mut(raw: CUdeviceptr) -> DevicePtrMut {
        DevicePtrMut(raw as usize as *mut c_void)
    }

    /// Direct fp16 GEMV parity against an f32/f64 dequant-and-matmul oracle that
    /// is fed the **same fp16-rounded** activations and the same (fp16- or
    /// f32-) rounded scales, so the residual covers only the kernel's documented
    /// accumulation precision and fp16 output rounding — not input quantization,
    /// which both sides share.
    fn run_parity(scales_fp16: bool, with_bias: bool) -> (f32, f32, f32, bool) {
        // K spans 128 block-32 groups (contraction depth 4096, near the model's
        // widest hidden path), N covers several 8-column CTAs plus a ragged tail.
        run_parity_dims(4096, 70, scales_fp16, with_bias, false)
    }

    /// Parametrized fp16 GEMV parity harness. `k` and `n` pick the projection
    /// shape so callers can pin the exact production dims (e.g. Qwen2.5-1.5B's
    /// gate/up K=1536,N=8960 and down-projection K=8960,N=1536) that select
    /// different GEMV variants and block-count boundaries. `explicit_zp` toggles
    /// the asymmetric per-block int4 zero-point path (see
    /// [`run_parity_dims_block`]). Delegates to [`run_parity_dims_block`] with
    /// the default block-32 layout.
    fn run_parity_dims(
        k: usize,
        n: usize,
        scales_fp16: bool,
        with_bias: bool,
        explicit_zp: bool,
    ) -> (f32, f32, f32, bool) {
        run_parity_dims_block(k, n, 32, scales_fp16, with_bias, explicit_zp)
    }

    /// Parametrized fp16 GEMV parity harness with an explicit `block_size`. The
    /// tuned block-32 kernels and the model-agnostic general-block-size kernel
    /// share this oracle; passing `block_size != 32` exercises the general
    /// decode GEMV (`matmul_nbits_gemv_f16_general_bs`) against the same f64
    /// dequant-and-matmul reference. `block_size` must be a power of two >= 16
    /// and must divide `k`. `explicit_zp` supplies a non-uniform per-block int4
    /// zero-point tensor (packed two block-nibbles per byte) instead of the
    /// symmetric zp=8 default, so a zero-point indexing regression in the
    /// general kernel's dequant path is caught.
    fn run_parity_dims_block(
        k: usize,
        n: usize,
        block_size: usize,
        scales_fp16: bool,
        with_bias: bool,
        explicit_zp: bool,
    ) -> (f32, f32, f32, bool) {
        let Some(runtime) = runtime() else {
            eprintln!("skipping MatMulNBits fp16 GEMV parity test: CUDA runtime unavailable");
            return (0.0, 0.0, 0.0, true);
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping MatMulNBits fp16 GEMV parity test: fp16 NVRTC headers unavailable");
            return (0.0, 0.0, 0.0, true);
        }

        let k_blocks = k / block_size;
        let blob_size = block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);

        // Deterministic LCG so the test is reproducible without extra crates.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // fp16 activations (device input) plus their fp16-value-as-f32 twin so
        // the oracle consumes identical inputs.
        let mut activation_f16 = vec![f16::ZERO; k];
        let mut activation_ref = vec![0.0f32; k];
        for (dst_h, dst_f) in activation_f16.iter_mut().zip(activation_ref.iter_mut()) {
            let h = f16::from_f32(next());
            *dst_h = h;
            *dst_f = h.to_f32();
        }

        // int4 quant codes (0..15), packed two nibbles per byte in the exact
        // symmetric block-32 layout the kernel unpacks.
        let mut quant = vec![0u8; n * k];
        for value in quant.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for pair in 0..blob_size {
                    let low = quant[col * k + block * block_size + pair * 2] & 15;
                    let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                    packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                }
            }
        }

        // Explicit asymmetric int4 zero points: a non-uniform per-(col, block)
        // code in 0..15, packed two block-nibbles per byte exactly as the kernel
        // unpacks (`zp[col*zp_row_bytes + block/2]`, low nibble for even blocks).
        // The symmetric default (zp=8) is used when `explicit_zp` is false.
        let mut zp_codes = vec![8i32; n * k_blocks];
        let mut zp_packed = vec![0u8; n * zp_row_bytes];
        if explicit_zp {
            for i in 0..n * k_blocks {
                zp_codes[i] = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as i32;
            }
            for col in 0..n {
                for block in 0..k_blocks {
                    let code = (zp_codes[col * k_blocks + block] & 15) as u8;
                    let byte = &mut zp_packed[col * zp_row_bytes + block / 2];
                    if block & 1 == 0 {
                        *byte = (*byte & 0xf0) | code;
                    } else {
                        *byte = (*byte & 0x0f) | (code << 4);
                    }
                }
            }
        }

        // Per (col, block) scales, rounded to the storage dtype so both paths use
        // the same scale value.
        let mut scale_ref = vec![0.0f32; n * k_blocks];
        let mut scale_f16 = vec![f16::ZERO; n * k_blocks];
        let mut scale_f32 = vec![0.0f32; n * k_blocks];
        for i in 0..n * k_blocks {
            let raw = 0.015 + 0.01 * (next() * 0.5 + 0.5);
            if scales_fp16 {
                let h = f16::from_f32(raw);
                scale_f16[i] = h;
                scale_ref[i] = h.to_f32();
            } else {
                scale_f32[i] = raw;
                scale_ref[i] = raw;
            }
        }

        let mut bias_f16 = vec![f16::ZERO; n];
        let mut bias_ref = vec![0.0f32; n];
        if with_bias {
            for (h, f) in bias_f16.iter_mut().zip(bias_ref.iter_mut()) {
                let value = f16::from_f32(next());
                *h = value;
                *f = value.to_f32();
            }
        }

        // f64 dequant-and-matmul oracle over the shared fp16 activations.
        let mut expected = vec![0.0f32; n];
        for col in 0..n {
            let mut acc = 0.0f64;
            for block in 0..k_blocks {
                let scale = scale_ref[col * k_blocks + block] as f64;
                let zero_point = zp_codes[col * k_blocks + block];
                for within in 0..block_size {
                    let depth = block * block_size + within;
                    let q = quant[col * k + depth] as i32 - zero_point;
                    acc += activation_ref[depth] as f64 * q as f64 * scale;
                }
            }
            if with_bias {
                acc += bias_ref[col] as f64;
            }
            expected[col] = acc as f32;
        }

        let activation_dev = runtime.alloc_raw(activation_f16.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime
            .alloc_raw(n * k_blocks * if scales_fp16 { 2 } else { 4 })
            .unwrap();
        let zp_dev = runtime.alloc_raw(zp_packed.len().max(1)).unwrap();
        let bias_dev = runtime.alloc_raw(n * 2).unwrap();
        let output_dev = runtime.alloc_raw(n * 2).unwrap();

        // SAFETY: device buffers were sized to hold each source slice.
        unsafe {
            runtime
                .htod(as_bytes(&activation_f16), activation_dev)
                .unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            if scales_fp16 {
                runtime.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            } else {
                runtime.htod(as_bytes(&scale_f32), scales_dev).unwrap();
            }
            if explicit_zp {
                runtime.htod(&zp_packed, zp_dev).unwrap();
            }
            if with_bias {
                runtime.htod(as_bytes(&bias_f16), bias_dev).unwrap();
            }
        }

        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, zp_row_bytes];
        let zp_strides = [zp_row_bytes as i64, 1];
        let bias_shape = [n];
        let bias_strides = [1i64];
        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];

        let scales_dtype = if scales_fp16 {
            DataType::Float16
        } else {
            DataType::Float32
        };
        let device = DeviceId::cuda(0);
        let mut inputs = vec![
            TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                scales_dtype,
                &scales_shape,
                &scales_strides,
                device,
            ),
        ];
        let zp_shape = [n, zp_row_bytes];
        let zp_strides = [zp_row_bytes as i64, 1];
        let zp_view = TensorView::new(
            device_ptr(zp_dev),
            DataType::Uint8,
            &zp_shape,
            &zp_strides,
            device,
        );
        // Slots: 3 = zero_points, 4 = g_idx, 5 = bias. Fill only up to the last
        // present optional input so the kernel's `optional_input` indexing holds.
        if explicit_zp {
            inputs.push(zp_view);
        } else if with_bias {
            inputs.push(TensorView::absent(DataType::Uint8));
        }
        if with_bias {
            inputs.push(TensorView::absent(DataType::Int32));
            inputs.push(TensorView::new(
                device_ptr(bias_dev),
                DataType::Float16,
                &bias_shape,
                &bias_strides,
                device,
            ));
        }

        let mut outputs = [TensorMut::new(
            device_ptr_mut(output_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        )];

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
        };
        kernel.run(&inputs, &mut outputs).unwrap();
        runtime.synchronize().unwrap();

        assert!(
            kernel.last_call_capture_safe.load(Ordering::Relaxed),
            "fp16 decode GEMV must report capture-safe"
        );

        let mut got_f16 = vec![f16::ZERO; n];
        // SAFETY: `output_dev` holds `n` fp16 values.
        unsafe {
            runtime
                .dtoh(as_bytes_mut(&mut got_f16), output_dev)
                .unwrap();
        }

        // SAFETY: each pointer came from this runtime's `alloc_raw` and is freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(bias_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }

        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        let mut max_out = 0.0f32;
        let mut all_finite = true;
        for (g16, e) in got_f16.iter().zip(expected.iter()) {
            let g = g16.to_f32();
            if !g.is_finite() {
                all_finite = false;
            }
            let abs = (g - e).abs();
            let rel = abs / e.abs().max(1e-1);
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(rel);
            max_out = max_out.max(e.abs());
        }
        (worst_abs, worst_rel, max_out, all_finite)
    }

    /// Dequant-reference parity for the int8 (bits=8) fp16-activation decode
    /// GEMV at arbitrary `(k, n)`. Exercises the vectorised four-lane/eight-block
    /// path against an f64 oracle; `explicit_zp` toggles the symmetric zp=128
    /// default versus an explicit per-block uint8 zero point.
    fn run_int8_parity_dims(
        k: usize,
        n: usize,
        scales_fp16: bool,
        with_bias: bool,
        explicit_zp: bool,
    ) -> (f32, f32, f32, bool) {
        let Some(runtime) = runtime() else {
            eprintln!("skipping MatMulNBits int8 fp16 GEMV parity test: CUDA runtime unavailable");
            return (0.0, 0.0, 0.0, true);
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!(
                "skipping MatMulNBits int8 fp16 GEMV parity test: fp16 NVRTC headers unavailable"
            );
            return (0.0, 0.0, 0.0, true);
        }

        let block_size = 32usize;
        let k_blocks = k / block_size;
        let blob_size = block_size; // one byte per weight for bits=8

        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let mut activation_f16 = vec![f16::ZERO; k];
        let mut activation_ref = vec![0.0f32; k];
        for (dst_h, dst_f) in activation_f16.iter_mut().zip(activation_ref.iter_mut()) {
            let h = f16::from_f32(next());
            *dst_h = h;
            *dst_f = h.to_f32();
        }

        // int8 quant codes (0..255), one byte per weight in [n, k_blocks, 32].
        let mut quant = vec![0u8; n * k];
        for value in quant.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for within in 0..block_size {
                    packed[(col * k_blocks + block) * blob_size + within] =
                        quant[col * k + block * block_size + within];
                }
            }
        }

        // Explicit per-block uint8 zero points, or the symmetric 128 default.
        let mut zero_points = vec![0u8; n * k_blocks];
        if explicit_zp {
            for zp in zero_points.iter_mut() {
                *zp = ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        let zp_ref = |col: usize, block: usize| -> i32 {
            if explicit_zp {
                zero_points[col * k_blocks + block] as i32
            } else {
                128
            }
        };

        let mut scale_ref = vec![0.0f32; n * k_blocks];
        let mut scale_f16 = vec![f16::ZERO; n * k_blocks];
        let mut scale_f32 = vec![0.0f32; n * k_blocks];
        for i in 0..n * k_blocks {
            let raw = 0.015 + 0.01 * (next() * 0.5 + 0.5);
            if scales_fp16 {
                let h = f16::from_f32(raw);
                scale_f16[i] = h;
                scale_ref[i] = h.to_f32();
            } else {
                scale_f32[i] = raw;
                scale_ref[i] = raw;
            }
        }

        let mut bias_f16 = vec![f16::ZERO; n];
        let mut bias_ref = vec![0.0f32; n];
        if with_bias {
            for (h, f) in bias_f16.iter_mut().zip(bias_ref.iter_mut()) {
                let value = f16::from_f32(next());
                *h = value;
                *f = value.to_f32();
            }
        }

        let mut expected = vec![0.0f32; n];
        for col in 0..n {
            let mut acc = 0.0f64;
            for block in 0..k_blocks {
                let scale = scale_ref[col * k_blocks + block] as f64;
                let zp = zp_ref(col, block);
                for within in 0..block_size {
                    let depth = block * block_size + within;
                    let q = quant[col * k + depth] as i32 - zp;
                    acc += activation_ref[depth] as f64 * q as f64 * scale;
                }
            }
            if with_bias {
                acc += bias_ref[col] as f64;
            }
            expected[col] = acc as f32;
        }

        let activation_dev = runtime.alloc_raw(activation_f16.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime
            .alloc_raw(n * k_blocks * if scales_fp16 { 2 } else { 4 })
            .unwrap();
        let zp_dev = runtime.alloc_raw(zero_points.len()).unwrap();
        let bias_dev = runtime.alloc_raw(n * 2).unwrap();
        let output_dev = runtime.alloc_raw(n * 2).unwrap();

        // SAFETY: device buffers were sized to hold each source slice.
        unsafe {
            runtime
                .htod(as_bytes(&activation_f16), activation_dev)
                .unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            if scales_fp16 {
                runtime.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            } else {
                runtime.htod(as_bytes(&scale_f32), scales_dev).unwrap();
            }
            if explicit_zp {
                runtime.htod(&zero_points, zp_dev).unwrap();
            }
            if with_bias {
                runtime.htod(as_bytes(&bias_f16), bias_dev).unwrap();
            }
        }

        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, k_blocks];
        let zp_strides = [k_blocks as i64, 1];
        let bias_shape = [n];
        let bias_strides = [1i64];
        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];

        let scales_dtype = if scales_fp16 {
            DataType::Float16
        } else {
            DataType::Float32
        };
        let device = DeviceId::cuda(0);
        let mut inputs = vec![
            TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                scales_dtype,
                &scales_shape,
                &scales_strides,
                device,
            ),
        ];
        if explicit_zp || with_bias {
            inputs.push(if explicit_zp {
                TensorView::new(
                    device_ptr(zp_dev),
                    DataType::Uint8,
                    &zp_shape,
                    &zp_strides,
                    device,
                )
            } else {
                TensorView::absent(DataType::Uint8)
            });
        }
        if with_bias {
            inputs.push(TensorView::absent(DataType::Int32));
            inputs.push(TensorView::new(
                device_ptr(bias_dev),
                DataType::Float16,
                &bias_shape,
                &bias_strides,
                device,
            ));
        }

        let mut outputs = [TensorMut::new(
            device_ptr_mut(output_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        )];

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 8,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
        };
        kernel.run(&inputs, &mut outputs).unwrap();
        runtime.synchronize().unwrap();

        assert!(
            kernel.last_call_capture_safe.load(Ordering::Relaxed),
            "int8 decode GEMV must report capture-safe"
        );

        let mut got_f16 = vec![f16::ZERO; n];
        // SAFETY: `output_dev` holds `n` fp16 values.
        unsafe {
            runtime
                .dtoh(as_bytes_mut(&mut got_f16), output_dev)
                .unwrap();
        }

        // SAFETY: each pointer came from this runtime's `alloc_raw` and is freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(bias_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }

        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        let mut max_out = 0.0f32;
        let mut all_finite = true;
        for (g16, e) in got_f16.iter().zip(expected.iter()) {
            let g = g16.to_f32();
            if !g.is_finite() {
                all_finite = false;
            }
            let abs = (g - e).abs();
            let rel = abs / e.abs().max(1e-1);
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(rel);
            max_out = max_out.max(e.abs());
        }
        (worst_abs, worst_rel, max_out, all_finite)
    }

    #[test]
    fn fp16_down_projection_is_bit_exact_to_staged_kernel() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping down-projection GEMV parity test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping down-projection GEMV parity test: fp16 NVRTC headers unavailable");
            return;
        }

        // Prove the specialization matches the general GEMV bit-numerically for
        // the Qwen down shape AND an unrelated non-Qwen tall-skinny shape, so
        // the generalized selection is correct beyond one architecture.
        for (k, n) in [(QWEN_DOWN_K, QWEN_DOWN_N), (5632usize, 2048usize)] {
            assert_eq!(
                select_f16_gemv_variant(k, n, 32, true, false).variant,
                F16GemvVariant::DownProjection,
                "shape K={k}, N={n} must select the down variant under test"
            );
            let block_size = 32usize;
            let k_blocks = k / block_size;
            let blob_size = block_size / 2;

            let activation: Vec<f16> = (0..k)
                .map(|i| f16::from_f32(((i * 17 % 257) as f32 - 128.0) / 128.0))
                .collect();
            let packed: Vec<u8> = (0..n * k_blocks * blob_size)
                .map(|i| ((i * 29 + i / 7 + 13) & 0xff) as u8)
                .collect();
            let scales: Vec<f16> = (0..n * k_blocks)
                .map(|i| f16::from_f32(0.01 + (i % 17) as f32 * 0.0005))
                .collect();

            let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
            let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
            let scales_dev = runtime.alloc_raw(scales.len() * 2).unwrap();
            let staged_output_dev = runtime.alloc_raw(n * 2).unwrap();
            let down_output_dev = runtime.alloc_raw(n * 2).unwrap();
            // SAFETY: device buffers exactly cover their source slices.
            unsafe {
                runtime.htod(as_bytes(&activation), activation_dev).unwrap();
                runtime.htod(&packed, packed_dev).unwrap();
                runtime.htod(as_bytes(&scales), scales_dev).unwrap();
            }

            let device = DeviceId::cuda(0);
            let a_shape = [1usize, k];
            let a_strides = [k as i64, 1];
            let b_shape = [n, k_blocks, blob_size];
            let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
            let scales_shape = [n, k_blocks];
            let scales_strides = [k_blocks as i64, 1];
            let y_shape = [1usize, n];
            let y_strides = [n as i64, 1];
            let activation_view = TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            );
            let packed_view = TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_view = TensorView::new(
                device_ptr(scales_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let mut down_output = TensorMut::new(
                device_ptr_mut(down_output_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            );
            let kernel = MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits: 4,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: false,
                gate_up_swiglu: false,
                rmsnorm_prologue: false,
                rmsnorm_epsilon: 1e-5,
                last_call_capture_safe: AtomicBool::new(false),
            };
            let staged_source = format!("{GEMV_F16_SRC}\n{STAGED_DOWN_REFERENCE_SRC}");
            let staged_function = runtime
                .nvrtc_function(
                    "matmul_nbits_gemv_f16_down_staged_reference",
                    &staged_source,
                    STAGED_DOWN_REFERENCE_ENTRY,
                )
                .unwrap();
            let activation_ptr = cuptr(activation_view.data_ptr::<u8>() as *const c_void);
            let packed_ptr = cuptr(packed_view.data_ptr::<u8>() as *const c_void);
            let scales_ptr = cuptr(scales_view.data_ptr::<u8>() as *const c_void);
            let zero_points_ptr: CUdeviceptr = 0;
            let bias_ptr: CUdeviceptr = 0;
            let staged_output_ptr = staged_output_dev;
            let k_i32 = as_i32("K", k).unwrap();
            let n_i32 = as_i32("N", n).unwrap();
            let block_size_i32 = as_i32("block_size", block_size).unwrap();
            let k_blocks_i32 = as_i32("K block count", k_blocks).unwrap();
            let blob_size_i32 = as_i32("block blob size", blob_size).unwrap();
            let zp_row_bytes_i32 = as_i32("zero-point row byte count", k_blocks.div_ceil(2)).unwrap();
            let scales_fp16_flag = 1i32;
            let bias_post_round_flag = 0i32;
            let mut staged_builder = runtime.stream().launch_builder(&staged_function);
            staged_builder
                .arg(&activation_ptr)
                .arg(&packed_ptr)
                .arg(&scales_ptr)
                .arg(&zero_points_ptr)
                .arg(&bias_ptr)
                .arg(&staged_output_ptr)
                .arg(&k_i32)
                .arg(&n_i32)
                .arg(&block_size_i32)
                .arg(&k_blocks_i32)
                .arg(&blob_size_i32)
                .arg(&zp_row_bytes_i32)
                .arg(&scales_fp16_flag)
                .arg(&bias_post_round_flag);
            // SAFETY: this launches the exact pre-change down-projection entry
            // over the same validated buffers used by the replacement below.
            unsafe {
                staged_builder
                    .launch(LaunchConfig {
                        grid_dim: (n.div_ceil(GEMV_F16_DOWN_COLUMNS_PER_BLOCK) as u32, 1, 1),
                        block_dim: (GEMV_F16_DOWN_THREADS, 1, 1),
                        shared_mem_bytes: (k * std::mem::size_of::<f16>()) as u32,
                    })
                    .unwrap();
            }
            kernel
                .launch_f16_gemv_variant(
                    &activation_view,
                    &packed_view,
                    &scales_view,
                    true,
                    None,
                    None,
                    &mut down_output,
                    k_blocks,
                    blob_size,
                    k_blocks.div_ceil(2),
                    select_f16_gemv_variant(k, n, block_size, true, false),
                )
                .unwrap();
            runtime.synchronize().unwrap();

            let mut staged = vec![f16::ZERO; n];
            let mut down = vec![f16::ZERO; n];
            // SAFETY: both output allocations hold `n` fp16 values.
            unsafe {
                runtime
                    .dtoh(as_bytes_mut(&mut staged), staged_output_dev)
                    .unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut down), down_output_dev)
                    .unwrap();
                runtime.free_raw(activation_dev).unwrap();
                runtime.free_raw(packed_dev).unwrap();
                runtime.free_raw(scales_dev).unwrap();
                runtime.free_raw(staged_output_dev).unwrap();
                runtime.free_raw(down_output_dev).unwrap();
            }

            assert_eq!(
                as_bytes(&staged),
                as_bytes(&down),
                "register-loaded down projection must be bit-exact to the pre-change staged kernel \
                 at K={k}, N={n}"
            );
        }
    }

    #[test]
    fn fp16_gemv_variant_selection_is_structural() {
        // The down variant is selected by the tall-skinny (K>N) block-32 fp16
        // shape *class*, generalizing across models — not by a magic K/N.
        let qwen = select_f16_gemv_variant(QWEN_DOWN_K, QWEN_DOWN_N, 32, true, false);
        assert_eq!(qwen.variant, F16GemvVariant::DownProjection);
        assert_eq!(
            qwen.reason,
            "variant=down_projection;class=tall_skinny(K>N);block_size=32;\
             scales=fp16;K%32==0"
        );

        // Non-Qwen tall-skinny down/output projections, including contractions
        // larger than the former activation-staging limit, must also select it.
        for (k, n) in [
            (5632, 2048),
            (11008, 4096),
            (2048, 512),
            (4096, 4096 - 32),
            (32_768, 4096),
        ] {
            let selection = select_f16_gemv_variant(k, n, 32, true, false);
            assert_eq!(
                selection.variant,
                F16GemvVariant::DownProjection,
                "tall-skinny K={k}, N={n} must select the down variant"
            );
        }

        // Wide (N>=K) projections, non-multiple-of-32 K, and non-block-32 all
        // fall back.
        let general_cases = [
            (896, 4864, 32, true),    // gate/up: N > K
            (896, 896, 32, true),     // square: K == N is not tall-skinny
            (896, 151_936, 32, true), // lm_head: N >> K
            (4880, 896, 32, true),    // 4880 % 32 != 0
            (4864, 896, 64, true),    // block_size != 32
        ];
        for (k, n, block_size, scales_fp16) in general_cases {
            let selection = select_f16_gemv_variant(k, n, block_size, scales_fp16, false);
            assert_eq!(
                selection.variant,
                F16GemvVariant::General,
                "K={k}, N={n}, block_size={block_size} must retain the general GEMV"
            );
        }

        // fp32 scales are never down-eligible even for a tall-skinny shape.
        assert_eq!(
            select_f16_gemv_variant(QWEN_DOWN_K, QWEN_DOWN_N, 32, false, false).variant,
            F16GemvVariant::General,
        );

        let asymmetric = select_f16_gemv_variant(QWEN_DOWN_K, QWEN_DOWN_N, 32, true, true);
        assert_eq!(asymmetric.variant, F16GemvVariant::General);
        assert_eq!(
            asymmetric.reason,
            "variant=general;zero_points=explicit;down_projection requires symmetric zp=8"
        );
    }

    #[test]
    fn fp16_down_projection_loads_activation_directly_into_registers() {
        let start = GEMV_F16_SRC
            .find("extern \"C\" __global__ void matmul_nbits_gemv_f16_scales_f16_down")
            .expect("down-projection entry must exist");
        let body = &GEMV_F16_SRC[start..];
        let end = body
            .find("\n}\n\n// Model-agnostic fp16 int4 decode GEMV")
            .expect("down-projection entry must have a bounded body");
        let body = &body[..end];

        assert!(
            !body.contains("activation_shared"),
            "down projection must not round-trip activations through shared memory"
        );
        for offset in [
            "activation_block);",
            "activation_block + 8);",
            "activation_block + 16);",
            "activation_block + 24);",
        ] {
            assert!(
                body.contains(offset),
                "down projection must directly load the block-32 activation at {offset}"
            );
        }
    }

    #[test]
    fn fp16_gemv_matches_dequant_reference() {
        let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
            (0.0f32, 0.0f32, 0.0f32, true);
        for (scales_fp16, with_bias) in [(false, false), (true, false), (true, true)] {
            let (abs, rel, out, finite) = run_parity(scales_fp16, with_bias);
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(rel);
            max_out = max_out.max(out);
            all_finite &= finite;
        }
        // fp16 output ULP is 2^-11 (~4.9e-4) of a value's magnitude, so the
        // absolute error floor scales with the largest output component. Bound
        // the observed abs error against that magnitude with 2x headroom for the
        // fp32-vs-f64 reduction-order drift accumulated over K=4096.
        let abs_bound = (max_out * 1e-3).max(1e-3);
        eprintln!(
            "MatMulNBits fp16 GEMV parity: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} \
             max_out={max_out:.3e} abs_bound={abs_bound:.3e}"
        );
        assert!(all_finite, "fp16 GEMV produced a non-finite output");
        assert!(
            worst_abs < abs_bound,
            "fp16 GEMV diverged from dequant reference: max_abs={worst_abs:.3e} bound={abs_bound:.3e}"
        );
        // Relative error (against a 1e-1 floor so near-zero columns do not
        // explode the ratio) isolates the per-element accuracy from the output
        // magnitude and must stay well under 5e-2.
        assert!(
            worst_rel < 5e-2,
            "fp16 GEMV diverged from dequant reference: max_rel={worst_rel:.3e}"
        );
    }

    /// Regression guard for the native-vs-ORT divergence investigated on
    /// Qwen2.5-1.5B-instruct (int4, block-32). Its MLP projections use dims that
    /// no other Qwen2.5 size hits: gate/up is K=1536,N=8960 (K<N → *general*
    /// GEMV) and the down-projection is K=8960,N=1536 (K>N → *tall-skinny*
    /// specialized GEMV). K=1536 is 48 block-32 groups and N=8960 is a whole
    /// multiple of the 8-column CTA width, exercising the block-count and column
    /// tiling boundaries at the exact production shapes. Both variants must track
    /// the f64 dequant-and-matmul oracle within the fp16 accumulation floor so a
    /// future kernel change cannot silently reintroduce a decode-step logit
    /// divergence at these dims.
    #[test]
    fn fp16_gemv_matches_dequant_reference_qwen_1_5b_dims() {
        // (k, n) → (gate/up general GEMV, down-projection tall-skinny GEMV).
        for (k, n) in [(1536usize, 8960usize), (8960usize, 1536usize)] {
            let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
                (0.0f32, 0.0f32, 0.0f32, true);
            for (scales_fp16, with_bias) in [(false, false), (true, false), (true, true)] {
                let (abs, rel, out, finite) = run_parity_dims(k, n, scales_fp16, with_bias, false);
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
                max_out = max_out.max(out);
                all_finite &= finite;
            }
            // Same fp16-ULP-scaled magnitude bound as the general parity test.
            // K here runs deeper (up to 8960), but the observed error stays
            // within that floor, so no extra slack is needed.
            let abs_bound = (max_out * 1e-3).max(1e-3);
            eprintln!(
                "MatMulNBits fp16 GEMV parity K={k} N={n}: max_abs={worst_abs:.3e} \
                 max_rel={worst_rel:.3e} max_out={max_out:.3e} abs_bound={abs_bound:.3e}"
            );
            assert!(all_finite, "fp16 GEMV produced a non-finite output (K={k} N={n})");
            assert!(
                worst_abs < abs_bound,
                "fp16 GEMV diverged from dequant reference at K={k} N={n}: \
                 max_abs={worst_abs:.3e} bound={abs_bound:.3e}"
            );
            assert!(
                worst_rel < 5e-2,
                "fp16 GEMV diverged from dequant reference at K={k} N={n}: max_rel={worst_rel:.3e}"
            );
        }
    }

    /// Model-agnostic block-size guard: the general-block-size fp16 decode GEMV
    /// (`matmul_nbits_gemv_f16_general_bs`, selected for any `block_size != 32`)
    /// must track the same f64 dequant-and-matmul oracle as the tuned block-32
    /// path. Exercised at `block_size = 128` — the Qwen2.5-0.5B **v4-bs128**
    /// foundry package's layout that previously failed to load — across the
    /// exact q/k/v/o/gate/up/down projection dims (K=896, and the wide MLP
    /// K=4864), a ragged-N tail, and both fp16/fp32 scales with and without a
    /// folded bias. It also drives an explicit **asymmetric** per-block int4
    /// zero-point tensor so a zero-point indexing regression in the general
    /// kernel's dequant path is caught (plausible for zp-bearing non-32-block
    /// models). A regression in the general block-index math (scale/zp stride,
    /// K-stepping, or nibble unpack) would diverge here.
    #[test]
    fn fp16_gemv_matches_dequant_reference_block128() {
        // (block_size, k, n): block-128 covers the Qwen2.5-0.5B bs128 attention
        // and MLP projection shapes (K=896, wide MLP K=4864) plus a ragged N
        // (70) spanning several 8-column CTAs with a partial tail; block-64
        // proves the block index math generalizes beyond a single width. All K
        // are whole multiples of their block size.
        for (block_size, k, n) in [
            (128usize, 896usize, 896usize),
            (128usize, 896usize, 4864usize),
            (128usize, 4864usize, 896usize),
            (128usize, 896usize, 70usize),
            (64usize, 896usize, 896usize),
            (64usize, 896usize, 70usize),
        ] {
            // Any block_size != 32 must route through the general kernel.
            assert_eq!(
                select_f16_gemv_variant(k, n, block_size, true, false).variant,
                F16GemvVariant::General,
                "block_size={block_size} K={k} N={n} must select the general variant"
            );
            let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
                (0.0f32, 0.0f32, 0.0f32, true);
            // (scales_fp16, with_bias, explicit_zp): the last two rows exercise
            // the general kernel's asymmetric int4 zero-point dequant path.
            for (scales_fp16, with_bias, explicit_zp) in [
                (false, false, false),
                (true, false, false),
                (true, true, false),
                (false, false, true),
                (true, true, true),
            ] {
                let (abs, rel, out, finite) =
                    run_parity_dims_block(k, n, block_size, scales_fp16, with_bias, explicit_zp);
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
                max_out = max_out.max(out);
                all_finite &= finite;
            }
            // Same fp16-ULP-scaled magnitude bound as the block-32 parity tests;
            // a wider block shares one scale across more K-elements but the oracle
            // uses the identical dequant, so the accumulation floor is unchanged.
            let abs_bound = (max_out * 1e-3).max(1e-3);
            eprintln!(
                "MatMulNBits fp16 GEMV block-{block_size} parity K={k} N={n}: \
                 max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} max_out={max_out:.3e} \
                 abs_bound={abs_bound:.3e}"
            );
            assert!(
                all_finite,
                "block-{block_size} fp16 GEMV produced a non-finite output (K={k} N={n})"
            );
            assert!(
                worst_abs < abs_bound,
                "block-{block_size} fp16 GEMV diverged from dequant reference at K={k} N={n}: \
                 max_abs={worst_abs:.3e} bound={abs_bound:.3e}"
            );
            assert!(
                worst_rel < 5e-2,
                "block-{block_size} fp16 GEMV diverged from dequant reference at K={k} N={n}: \
                 max_rel={worst_rel:.3e}"
            );
        }
    }

    /// Asymmetric-zero-point int4 fp16 decode GEMV at Phi-4-mini's int4 dims
    /// must track an f64 dequant oracle that honors the per-block zero point.
    /// Phi carries explicit zero points on every MatMulNBits, so the vectorized
    /// `scales_f16` GEMV that decode routes to (o_proj K=3072,N=3072 and the
    /// gate/up projection K=3072,N=8192) must dequantize `(code - zp) * scale`,
    /// not the symmetric `(code - 8)`. This is the mutation guard for the shared
    /// `int4x8_to_half2x4_sub` primitive: a kernel that ignored the zero point
    /// (subtracting the implicit 8) would diverge from the oracle far beyond the
    /// fp16 floor and fail here.
    #[test]
    fn fp16_gemv_matches_dequant_reference_phi_int4_zp_dims() {
        // (K, N): Phi o_proj (K==N general GEMV) and gate/up (K<N general GEMV).
        for (k, n) in [(3072usize, 3072usize), (3072, 8192)] {
            let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
                (0.0f32, 0.0f32, 0.0f32, true);
            // Exercise both the plain asymmetric GEMV and the folded-bias
            // (residual) epilogue that the skip-rmsnorm fusion produces on the
            // preceding projection. Routed through `run_parity_dims` (block-32
            // default) so the `explicit_zp` delegation is actually driven.
            for (with_bias, explicit_zp) in [(false, true), (true, true)] {
                let (abs, rel, out, finite) =
                    run_parity_dims(k, n, true, with_bias, explicit_zp);
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
                max_out = max_out.max(out);
                all_finite &= finite;
            }
            let abs_bound = (max_out * 1e-3).max(1e-3);
            eprintln!(
                "MatMulNBits int4 asymmetric-zp GEMV parity K={k} N={n}: max_abs={worst_abs:.3e} \
                 max_rel={worst_rel:.3e} max_out={max_out:.3e} abs_bound={abs_bound:.3e}"
            );
            assert!(
                all_finite,
                "int4 asymmetric-zp GEMV produced a non-finite output (K={k} N={n})"
            );
            assert!(
                worst_abs < abs_bound,
                "int4 asymmetric-zp GEMV diverged from dequant reference at K={k} N={n}: \
                 max_abs={worst_abs:.3e} bound={abs_bound:.3e}"
            );
            assert!(
                worst_rel < 5e-2,
                "int4 asymmetric-zp GEMV diverged from dequant reference at K={k} N={n}: \
                 max_rel={worst_rel:.3e}"
            );
        }
    }

    /// Int8 (bits=8) fp16 decode GEMV must track an f64 dequant oracle at Phi's
    /// GEMV dims — the shapes ORT beat us on. QKV (K=3072, N=5120), down
    /// projection (K=8192, N=3072), and the lm_head slice (K=3072, wide N) all
    /// exercise the vectorised four-lane/eight-block path; a ragged N tail and
    /// an explicit-zero-point case guard the reduction and dequant edges.
    #[test]
    fn int8_fp16_gemv_matches_dequant_reference_phi_dims() {
        let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
            (0.0f32, 0.0f32, 0.0f32, true);
        // (K, N, scales_fp16, with_bias, explicit_zp)
        let cases = [
            (3072usize, 5120usize, true, false, false), // Phi QKV int8
            (8192, 3072, true, false, false),           // Phi down projection int8
            (3072, 5120, true, true, false),            // QKV with folded bias
            (3072, 5120, true, false, true),            // explicit per-block zero points
            (8192, 3072, true, false, true),            // down-proj zp: exercises split-K at K=8192
            (3072, 5121, true, false, false),           // ragged N tail (not warp-tile aligned)
            (8192, 3072, false, false, false),          // fp32 scales
        ];
        for (k, n, scales_fp16, with_bias, explicit_zp) in cases {
            let (abs, rel, out, finite) =
                run_int8_parity_dims(k, n, scales_fp16, with_bias, explicit_zp);
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(rel);
            max_out = max_out.max(out);
            all_finite &= finite;
        }
        // int8 quant codes span 0..255 so accumulated magnitudes (and thus the
        // fp16 output ULP floor) are larger than the int4 case; keep the same
        // magnitude-relative bound shape with headroom for K up to 8192.
        let abs_bound = (max_out * 2e-3).max(1e-3);
        eprintln!(
            "MatMulNBits int8 fp16 GEMV parity: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} \
             max_out={max_out:.3e} abs_bound={abs_bound:.3e}"
        );
        assert!(all_finite, "int8 fp16 GEMV produced a non-finite output");
        assert!(
            worst_abs < abs_bound,
            "int8 fp16 GEMV diverged from dequant reference: max_abs={worst_abs:.3e} \
             bound={abs_bound:.3e}"
        );
        assert!(
            worst_rel < 5e-2,
            "int8 fp16 GEMV diverged from dequant reference: max_rel={worst_rel:.3e}"
        );
    }

    /// Folding a standalone `Add(MatMulNBits, bias)` into the GEMV epilogue must
    /// stay **byte-identical** to the original two-op path so greedy decode
    /// tokens do not shift. The two-op path is `fp16(fp16(acc) + bias)`: the
    /// GEMV first rounds its accumulator to fp16, then the elementwise `Add`
    /// rounds again after an fp16 add. This reproduces that exactly by running
    /// the real kernel with no bias (the fp16 GEMV output) and adding the fp16
    /// bias on the host, then asserting the folded-bias kernel matches bit-for-
    /// bit across every output column.
    #[test]
    fn fp16_folded_bias_is_bit_exact_to_two_op_path() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping folded-bias bit-exactness test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping folded-bias bit-exactness test: fp16 NVRTC headers unavailable");
            return;
        }

        // QKV decode shape: K=896, N=1152, symmetric block-32, fp16 scales.
        let k = 896usize;
        let n = 1152usize;
        let block_size = 32usize;
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let activation: Vec<f16> = (0..k).map(|_| f16::from_f32(next())).collect();
        let mut quant = vec![0u8; n * k];
        for value in quant.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for pair in 0..blob_size {
                    let low = quant[col * k + block * block_size + pair * 2] & 15;
                    let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                    packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                }
            }
        }
        let scales: Vec<f16> = (0..n * k_blocks)
            .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
            .collect();
        // Bias with a wide magnitude range so the second fp16 round is exercised.
        let bias: Vec<f16> = (0..n).map(|_| f16::from_f32(next() * 4.0)).collect();

        let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime.alloc_raw(scales.len() * 2).unwrap();
        let bias_dev = runtime.alloc_raw(bias.len() * 2).unwrap();
        let nobias_output_dev = runtime.alloc_raw(n * 2).unwrap();
        let fused_output_dev = runtime.alloc_raw(n * 2).unwrap();
        // SAFETY: device buffers exactly cover their source slices.
        unsafe {
            runtime.htod(as_bytes(&activation), activation_dev).unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            runtime.htod(as_bytes(&scales), scales_dev).unwrap();
            runtime.htod(as_bytes(&bias), bias_dev).unwrap();
        }

        let device = DeviceId::cuda(0);
        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let bias_shape = [n];
        let bias_strides = [1i64];
        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];
        let activation_view = TensorView::new(
            device_ptr(activation_dev),
            DataType::Float16,
            &a_shape,
            &a_strides,
            device,
        );
        let packed_view = TensorView::new(
            device_ptr(packed_dev),
            DataType::Uint8,
            &b_shape,
            &b_strides,
            device,
        );
        let scales_view = TensorView::new(
            device_ptr(scales_dev),
            DataType::Float16,
            &scales_shape,
            &scales_strides,
            device,
        );
        let bias_view = TensorView::new(
            device_ptr(bias_dev),
            DataType::Float16,
            &bias_shape,
            &bias_strides,
            device,
        );
        let mut nobias_output = TensorMut::new(
            device_ptr_mut(nobias_output_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        );
        let mut fused_output = TensorMut::new(
            device_ptr_mut(fused_output_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        );

        let selection = select_f16_gemv_variant(k, n, block_size, true, false);
        let kernel_nobias = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
        };
        let kernel_fold = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: true,
            gate_up_swiglu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
        };
        kernel_nobias
            .launch_f16_gemv_variant(
                &activation_view,
                &packed_view,
                &scales_view,
                true,
                None,
                None,
                &mut nobias_output,
                k_blocks,
                blob_size,
                k_blocks.div_ceil(2),
                selection,
            )
            .unwrap();
        kernel_fold
            .launch_f16_gemv_variant(
                &activation_view,
                &packed_view,
                &scales_view,
                true,
                None,
                Some(&bias_view),
                &mut fused_output,
                k_blocks,
                blob_size,
                k_blocks.div_ceil(2),
                selection,
            )
            .unwrap();
        runtime.synchronize().unwrap();

        let mut gemv_out = vec![f16::ZERO; n];
        let mut fused_out = vec![f16::ZERO; n];
        // SAFETY: both output allocations hold `n` fp16 values.
        unsafe {
            runtime
                .dtoh(as_bytes_mut(&mut gemv_out), nobias_output_dev)
                .unwrap();
            runtime
                .dtoh(as_bytes_mut(&mut fused_out), fused_output_dev)
                .unwrap();
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(bias_dev).unwrap();
            runtime.free_raw(nobias_output_dev).unwrap();
            runtime.free_raw(fused_output_dev).unwrap();
        }

        for col in 0..n {
            // Two-op reference: fp16(fp16(acc) + bias). gemv_out is already the
            // fp16-rounded accumulator, so add the fp16 bias in f32 and round.
            let two_op = f16::from_f32(gemv_out[col].to_f32() + bias[col].to_f32());
            assert_eq!(
                fused_out[col].to_bits(),
                two_op.to_bits(),
                "folded bias diverged at column {col}: fused={:?} two_op={:?} (gemv={:?} bias={:?})",
                fused_out[col],
                two_op,
                gemv_out[col],
                bias[col]
            );
        }
    }

    // Faithful replica of the elementwise `silu_mul_f16` scalar path (which is
    // byte-identical to its half2 path): `fp16(silu(f32(g)) * f32(u))`. Used to
    // build the two-op reference the paired kernel must reproduce bit-for-bit.
    const REF_SILU_MUL_SRC: &str = r#"
#include <cuda_fp16.h>
__device__ float ref_op_silu(float x) {
    if (x >= 0.0f) {
        const float denominator = __fadd_rn(1.0f, (float)exp((double)-x));
        return __fdiv_rn(x, denominator);
    }
    const float e = (float)exp((double)x);
    const float numerator = __fmul_rn(x, e);
    return __fdiv_rn(numerator, __fadd_rn(1.0f, e));
}
extern "C" __global__ void ref_silu_mul_f16(
    const __half* g, const __half* u, __half* y, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2half_rn(
            __fmul_rn(ref_op_silu(__half2float(g[i])), __half2float(u[i])));
    }
}
"#;

    /// The paired gate/up SwiGLU path must be byte-identical to running two
    /// standalone MatMulNBits projections and then `silu_mul_f16`. Decode covers
    /// the Qwen shape and an unrelated shape; prefill covers the reported
    /// five-token Qwen case plus M/N tails on a small unrelated shape.
    #[test]
    fn fp16_gate_up_swiglu_is_bit_exact_to_two_op_path() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping gate/up SwiGLU bit-exactness test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping gate/up SwiGLU bit-exactness test: fp16 NVRTC headers unavailable");
            return;
        }

        // (M, K=hidden, N=intermediate): preserve both decode cases, then add
        // Qwen M=5 prefill and unrelated row/column tails.
        for (m, k, n) in [
            (1usize, QWEN_DOWN_N, QWEN_DOWN_K),
            (1, 2048, 5632),
            (5, QWEN_DOWN_N, QWEN_DOWN_K),
            (3, 96, 77),
        ] {
            let block_size = 32usize;
            let k_blocks = k / block_size;
            let blob_size = block_size / 2;

            let mut state = 0x0bad_c0de_dead_beefu64;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            };

            let pack = |next: &mut dyn FnMut() -> f32| -> Vec<u8> {
                let mut quant = vec![0u8; n * k];
                for value in quant.iter_mut() {
                    *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
                }
                let mut packed = vec![0u8; n * k_blocks * blob_size];
                for col in 0..n {
                    for block in 0..k_blocks {
                        for pair in 0..blob_size {
                            let low = quant[col * k + block * block_size + pair * 2] & 15;
                            let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                            packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                        }
                    }
                }
                packed
            };

            let activation: Vec<f16> = (0..m * k).map(|_| f16::from_f32(next())).collect();
            let packed_gate = pack(&mut next);
            let scales_gate: Vec<f16> = (0..n * k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();
            let packed_up = pack(&mut next);
            let scales_up: Vec<f16> = (0..n * k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();

            let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
            let packed_gate_dev = runtime.alloc_raw(packed_gate.len()).unwrap();
            let scales_gate_dev = runtime.alloc_raw(scales_gate.len() * 2).unwrap();
            let packed_up_dev = runtime.alloc_raw(packed_up.len()).unwrap();
            let scales_up_dev = runtime.alloc_raw(scales_up.len() * 2).unwrap();
            let output_elements = m * n;
            let gate_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            let up_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            let ref_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            let fused_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            // SAFETY: device buffers exactly cover their source slices.
            unsafe {
                runtime.htod(as_bytes(&activation), activation_dev).unwrap();
                runtime.htod(&packed_gate, packed_gate_dev).unwrap();
                runtime
                    .htod(as_bytes(&scales_gate), scales_gate_dev)
                    .unwrap();
                runtime.htod(&packed_up, packed_up_dev).unwrap();
                runtime.htod(as_bytes(&scales_up), scales_up_dev).unwrap();
            }

            let device = DeviceId::cuda(0);
            let a_shape = [m, k];
            let a_strides = [k as i64, 1];
            let b_shape = [n, k_blocks, blob_size];
            let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
            let scales_shape = [n, k_blocks];
            let scales_strides = [k_blocks as i64, 1];
            let y_shape = [m, n];
            let y_strides = [n as i64, 1];
            let activation_view = TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            );
            let packed_gate_view = TensorView::new(
                device_ptr(packed_gate_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_gate_view = TensorView::new(
                device_ptr(scales_gate_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let packed_up_view = TensorView::new(
                device_ptr(packed_up_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_up_view = TensorView::new(
                device_ptr(scales_up_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let mut gate_out = TensorMut::new(
                device_ptr_mut(gate_out_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            );
            let mut up_out = TensorMut::new(
                device_ptr_mut(up_out_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            );
            let fused_out = TensorMut::new(
                device_ptr_mut(fused_out_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            );

            let gemv_kernel = MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits: 4,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: false,
                gate_up_swiglu: false,
                rmsnorm_prologue: false,
                rmsnorm_epsilon: 1e-5,
                last_call_capture_safe: AtomicBool::new(false),
            };
            // Reference: two standalone MatMulNBits projections.
            if m == 1 {
                let selection = select_f16_gemv_variant(k, n, block_size, true, false);
                assert_eq!(
                    selection.variant,
                    F16GemvVariant::General,
                    "gate/up decode projections must use the general GEMV as the reference"
                );
                gemv_kernel
                    .launch_f16_gemv_variant(
                        &activation_view,
                        &packed_gate_view,
                        &scales_gate_view,
                        true,
                        None,
                        None,
                        &mut gate_out,
                        k_blocks,
                        blob_size,
                        k_blocks.div_ceil(2),
                        selection,
                    )
                    .unwrap();
                gemv_kernel
                    .launch_f16_gemv_variant(
                        &activation_view,
                        &packed_up_view,
                        &scales_up_view,
                        true,
                        None,
                        None,
                        &mut up_out,
                        k_blocks,
                        blob_size,
                        k_blocks.div_ceil(2),
                        selection,
                    )
                    .unwrap();
            } else {
                gemv_kernel
                    .launch_f16_gemm(
                        &activation_view,
                        &packed_gate_view,
                        &scales_gate_view,
                        true,
                        None,
                        None,
                        &mut gate_out,
                        m,
                        k_blocks,
                        gemv_kernel.block_size * gemv_kernel.bits / 8,
                        0,
                    )
                    .unwrap();
                gemv_kernel
                    .launch_f16_gemm(
                        &activation_view,
                        &packed_up_view,
                        &scales_up_view,
                        true,
                        None,
                        None,
                        &mut up_out,
                        m,
                        k_blocks,
                        gemv_kernel.block_size * gemv_kernel.bits / 8,
                        0,
                    )
                    .unwrap();
            }
            // Then the reference silu_mul (byte-identical to silu_mul_f16).
            let ref_function = runtime
                .nvrtc_function(
                    "matmul_nbits_ref_silu_mul",
                    REF_SILU_MUL_SRC,
                    "ref_silu_mul_f16",
                )
                .unwrap();
            let gate_out_ptr = cuptr(device_ptr(gate_out_dev).0);
            let up_out_ptr = cuptr(device_ptr(up_out_dev).0);
            let ref_out_ptr = cuptr(device_ptr(ref_out_dev).0);
            let output_elements_i32 = output_elements as i32;
            let mut ref_builder = runtime.stream().launch_builder(&ref_function);
            ref_builder
                .arg(&gate_out_ptr)
                .arg(&up_out_ptr)
                .arg(&ref_out_ptr)
                .arg(&output_elements_i32);
            // SAFETY: all three buffers hold `output_elements` fp16 values.
            unsafe {
                ref_builder.launch(LaunchConfig {
                    grid_dim: (output_elements.div_ceil(256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();

            // Subject: exercise the fused node's real M=1/M>1 dispatch.
            let inputs = [
                activation_view,
                packed_gate_view,
                scales_gate_view,
                packed_up_view,
                scales_up_view,
            ];
            let mut outputs = [fused_out];
            gemv_kernel
                .run_f16_gate_up_swiglu(&inputs, &mut outputs)
                .unwrap();
            assert_eq!(
                gemv_kernel.last_call_capture_safe.load(Ordering::Relaxed),
                m == 1,
                "only M=1 decode may be advertised capture-safe"
            );
            runtime.synchronize().unwrap();

            let mut reference = vec![f16::ZERO; output_elements];
            let mut fused = vec![f16::ZERO; output_elements];
            // SAFETY: both output allocations hold `output_elements` fp16 values.
            unsafe {
                runtime
                    .dtoh(as_bytes_mut(&mut reference), ref_out_dev)
                    .unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut fused), fused_out_dev)
                    .unwrap();
                runtime.free_raw(activation_dev).unwrap();
                runtime.free_raw(packed_gate_dev).unwrap();
                runtime.free_raw(scales_gate_dev).unwrap();
                runtime.free_raw(packed_up_dev).unwrap();
                runtime.free_raw(scales_up_dev).unwrap();
                runtime.free_raw(gate_out_dev).unwrap();
                runtime.free_raw(up_out_dev).unwrap();
                runtime.free_raw(ref_out_dev).unwrap();
                runtime.free_raw(fused_out_dev).unwrap();
            }

            for index in 0..output_elements {
                assert_eq!(
                    fused[index].to_bits(),
                    reference[index].to_bits(),
                    "paired gate/up SwiGLU diverged at M={m}, K={k}, N={n}, row={}, column={}: \
                     fused={:?} reference={:?}",
                    index / n,
                    index % n,
                    fused[index],
                    reference[index]
                );
            }
        }
    }

    /// Byte-for-byte parity of the fused gate/up SwiGLU kernel *with an RMS
    /// prologue* against the standalone two-step sequence
    /// (`RMS-normalize the activation` → `paired gate/up SwiGLU`). The reference
    /// normalizes with the production prefill norm kernel
    /// (`matmul_nbits_rmsnorm_f16_warp_half4`) and then runs the already-proven
    /// non-prologue paired kernel, so any divergence isolates the fused
    /// prologue. Exercising M==1 (the single fused decode kernel) and M>1 (the
    /// normalize-into-scratch prefill path) keeps both dispatches honest.
    #[test]
    fn fused_gate_up_swiglu_rmsnorm_is_bit_exact_to_two_step_path() {
        run_fused_gate_up_swiglu_rmsnorm_parity(DataType::Float16, false);
    }

    /// Same gate/up SwiGLU RMS-norm fusion parity, but with an fp32 gamma (as
    /// Phi-4-mini exports it). The fused decode/prefill kernels must read the
    /// fp32 gamma at full precision and stay bit-identical to the two-step path.
    #[test]
    fn fused_gate_up_swiglu_rmsnorm_fp32_gamma_is_bit_exact_to_two_step_path() {
        run_fused_gate_up_swiglu_rmsnorm_parity(DataType::Float32, false);
    }

    /// Same gate/up SwiGLU RMS-norm fusion parity, but with asymmetric int4
    /// zero points on BOTH the gate and up projections (as Phi-4-mini exports
    /// them). The fused prologue kernel and the reference non-prologue paired
    /// kernel are independently written, so byte-identity here proves both honor
    /// the per-block zero point in the packed dequant. A fused kernel that
    /// ignored the zero point would diverge from the reference and fail. fp32
    /// gamma is paired with the zero points to mirror Phi's actual export.
    #[test]
    fn fused_gate_up_swiglu_rmsnorm_zero_points_is_bit_exact_to_two_step_path() {
        run_fused_gate_up_swiglu_rmsnorm_parity(DataType::Float32, true);
    }

    fn run_fused_gate_up_swiglu_rmsnorm_parity(gamma_dtype: DataType, explicit_zp: bool) {
        let Some(runtime) = runtime() else {
            eprintln!("skipping gate/up SwiGLU RMS-norm parity test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!(
                "skipping gate/up SwiGLU RMS-norm parity test: fp16 NVRTC headers unavailable"
            );
            return;
        }

        let epsilon = 1e-5f32;
        // (M, K=hidden, N=intermediate); hidden % 128 == 0 for the warp_half4
        // reduction. Decode is the capture-safe fused kernel; M=5 prefill routes
        // through the normalize-into-scratch path.
        for (m, k, n) in [(1usize, 896usize, 2432usize), (1, 3584, 4864), (5, 896, 2432)] {
            let block_size = 32usize;
            let k_blocks = k / block_size;
            let blob_size = block_size / 2;

            let mut state = 0xf00d_1ceb_00da_5555u64;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            };

            let pack = |next: &mut dyn FnMut() -> f32| -> Vec<u8> {
                let mut quant = vec![0u8; n * k];
                for value in quant.iter_mut() {
                    *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
                }
                let mut packed = vec![0u8; n * k_blocks * blob_size];
                for col in 0..n {
                    for block in 0..k_blocks {
                        for pair in 0..blob_size {
                            let low = quant[col * k + block * block_size + pair * 2] & 15;
                            let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                            packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                        }
                    }
                }
                packed
            };

            let activation: Vec<f16> = (0..m * k).map(|_| f16::from_f32(next())).collect();
            let packed_gate = pack(&mut next);
            let scales_gate: Vec<f16> = (0..n * k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();
            let packed_up = pack(&mut next);
            let scales_up: Vec<f16> = (0..n * k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();

            // Optional asymmetric zero points, nibble-packed [n, zp_row_bytes]
            // exactly as `int4_block_zero_point` reads them. Symmetric weights
            // use the implicit zp == 8 and carry no zero-point input.
            let zp_row_bytes = k_blocks.div_ceil(2);
            let pack_zp = |next: &mut dyn FnMut() -> f32| -> Vec<u8> {
                let mut zp = vec![0u8; n * zp_row_bytes];
                for col in 0..n {
                    for block in 0..k_blocks {
                        let nibble =
                            (((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8) & 15;
                        let byte = &mut zp[col * zp_row_bytes + (block >> 1)];
                        if block & 1 == 1 {
                            *byte = (*byte & 0x0f) | (nibble << 4);
                        } else {
                            *byte = (*byte & 0xf0) | nibble;
                        }
                    }
                }
                zp
            };
            let zp_gate = pack_zp(&mut next);
            let zp_up = pack_zp(&mut next);
            let gamma_is_f32 = gamma_dtype == DataType::Float32;
            let gamma_f32: Vec<f32> = (0..k).map(|_| 0.5 + 0.5 * (next() * 0.5 + 0.5)).collect();
            let gamma_bytes: Vec<u8> = if gamma_is_f32 {
                gamma_f32.iter().flat_map(|v| v.to_le_bytes()).collect()
            } else {
                gamma_f32
                    .iter()
                    .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                    .collect()
            };

            let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
            let packed_gate_dev = runtime.alloc_raw(packed_gate.len()).unwrap();
            let scales_gate_dev = runtime.alloc_raw(scales_gate.len() * 2).unwrap();
            let packed_up_dev = runtime.alloc_raw(packed_up.len()).unwrap();
            let scales_up_dev = runtime.alloc_raw(scales_up.len() * 2).unwrap();
            let gamma_dev = runtime.alloc_raw(gamma_bytes.len()).unwrap();
            let zp_gate_dev = runtime.alloc_raw(zp_gate.len()).unwrap();
            let zp_up_dev = runtime.alloc_raw(zp_up.len()).unwrap();
            let normalized_dev = runtime.alloc_raw(m * k * 2).unwrap();
            let output_elements = m * n;
            let ref_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            let fused_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            // SAFETY: device buffers exactly cover their source slices.
            unsafe {
                runtime.htod(as_bytes(&activation), activation_dev).unwrap();
                runtime.htod(&packed_gate, packed_gate_dev).unwrap();
                runtime
                    .htod(as_bytes(&scales_gate), scales_gate_dev)
                    .unwrap();
                runtime.htod(&packed_up, packed_up_dev).unwrap();
                runtime.htod(as_bytes(&scales_up), scales_up_dev).unwrap();
                runtime.htod(&gamma_bytes, gamma_dev).unwrap();
                if explicit_zp {
                    runtime.htod(&zp_gate, zp_gate_dev).unwrap();
                    runtime.htod(&zp_up, zp_up_dev).unwrap();
                }
            }

            let device = DeviceId::cuda(0);
            let a_shape = [m, k];
            let a_strides = [k as i64, 1];
            let b_shape = [n, k_blocks, blob_size];
            let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
            let scales_shape = [n, k_blocks];
            let scales_strides = [k_blocks as i64, 1];
            let gamma_shape = [k];
            let gamma_strides = [1i64];
            let y_shape = [m, n];
            let y_strides = [n as i64, 1];

            let activation_view = TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            );
            let normalized_view = TensorView::new(
                device_ptr(normalized_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            );
            let packed_gate_view = TensorView::new(
                device_ptr(packed_gate_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_gate_view = TensorView::new(
                device_ptr(scales_gate_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let packed_up_view = TensorView::new(
                device_ptr(packed_up_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_up_view = TensorView::new(
                device_ptr(scales_up_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let gamma_view = TensorView::new(
                device_ptr(gamma_dev),
                gamma_dtype,
                &gamma_shape,
                &gamma_strides,
                device,
            );
            let zp_shape = [n, zp_row_bytes];
            let zp_strides = [zp_row_bytes as i64, 1];
            let zp_gate_view = TensorView::new(
                device_ptr(zp_gate_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            );
            let zp_up_view = TensorView::new(
                device_ptr(zp_up_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            );
            let ref_out = TensorMut::new(
                device_ptr_mut(ref_out_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            );
            let fused_out = TensorMut::new(
                device_ptr_mut(fused_out_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            );

            let plain_swiglu = MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits: 4,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: false,
                gate_up_swiglu: true,
                rmsnorm_prologue: false,
                rmsnorm_epsilon: epsilon,
                last_call_capture_safe: AtomicBool::new(false),
            };
            let fused_swiglu = MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits: 4,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: false,
                gate_up_swiglu: true,
                rmsnorm_prologue: true,
                rmsnorm_epsilon: epsilon,
                last_call_capture_safe: AtomicBool::new(false),
            };

            // Reference: normalize the activation (production prefill norm
            // kernel), then run the proven non-prologue paired gate/up SwiGLU.
            plain_swiglu
                .launch_rmsnorm_prefill(&activation_view, &gamma_view, cuptr(device_ptr(normalized_dev).0), m)
                .unwrap();
            {
                let mut ref_outputs = [ref_out];
                let ref_inputs_base = [
                    normalized_view,
                    packed_gate_view,
                    scales_gate_view,
                    packed_up_view,
                    scales_up_view,
                ];
                if explicit_zp {
                    // Slot 5 gamma absent (already normalized), slots 6/7 zp.
                    let ref_inputs = [
                        ref_inputs_base[0],
                        ref_inputs_base[1],
                        ref_inputs_base[2],
                        ref_inputs_base[3],
                        ref_inputs_base[4],
                        TensorView::absent(DataType::Float16),
                        zp_gate_view,
                        zp_up_view,
                    ];
                    plain_swiglu
                        .run_f16_gate_up_swiglu(&ref_inputs, &mut ref_outputs)
                        .unwrap();
                } else {
                    plain_swiglu
                        .run_f16_gate_up_swiglu(&ref_inputs_base, &mut ref_outputs)
                        .unwrap();
                }
            }

            // Subject: the fused prologue kernel over the raw (residual sum)
            // activation with gamma at slot 5.
            {
                let mut fused_outputs = [fused_out];
                let fused_inputs_base = [
                    activation_view,
                    packed_gate_view,
                    scales_gate_view,
                    packed_up_view,
                    scales_up_view,
                    gamma_view,
                ];
                if explicit_zp {
                    let fused_inputs = [
                        fused_inputs_base[0],
                        fused_inputs_base[1],
                        fused_inputs_base[2],
                        fused_inputs_base[3],
                        fused_inputs_base[4],
                        fused_inputs_base[5],
                        zp_gate_view,
                        zp_up_view,
                    ];
                    fused_swiglu
                        .run_f16_gate_up_swiglu(&fused_inputs, &mut fused_outputs)
                        .unwrap();
                } else {
                    fused_swiglu
                        .run_f16_gate_up_swiglu(&fused_inputs_base, &mut fused_outputs)
                        .unwrap();
                }
            }
            assert_eq!(
                fused_swiglu.last_call_capture_safe.load(Ordering::Relaxed),
                m == 1,
                "only M=1 decode may be advertised capture-safe"
            );
            runtime.synchronize().unwrap();

            let mut reference = vec![f16::ZERO; output_elements];
            let mut fused = vec![f16::ZERO; output_elements];
            // SAFETY: both output allocations hold `output_elements` fp16 values.
            unsafe {
                runtime
                    .dtoh(as_bytes_mut(&mut reference), ref_out_dev)
                    .unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut fused), fused_out_dev)
                    .unwrap();
                runtime.free_raw(activation_dev).unwrap();
                runtime.free_raw(packed_gate_dev).unwrap();
                runtime.free_raw(scales_gate_dev).unwrap();
                runtime.free_raw(packed_up_dev).unwrap();
                runtime.free_raw(scales_up_dev).unwrap();
                runtime.free_raw(gamma_dev).unwrap();
                runtime.free_raw(zp_gate_dev).unwrap();
                runtime.free_raw(zp_up_dev).unwrap();
                runtime.free_raw(normalized_dev).unwrap();
                runtime.free_raw(ref_out_dev).unwrap();
                runtime.free_raw(fused_out_dev).unwrap();
            }

            for index in 0..output_elements {
                assert_eq!(
                    fused[index].to_bits(),
                    reference[index].to_bits(),
                    "fused gate/up SwiGLU RMS prologue diverged at M={m}, K={k}, N={n}, \
                     row={}, column={}: fused={:?} reference={:?}",
                    index / n,
                    index % n,
                    fused[index],
                    reference[index]
                );
            }
        }
    }

    /// Byte-for-byte parity of the fused SkipSimplifiedLayerNormalization
    /// epilogue/prologue against the standalone three-op sequence
    /// (`preceding MatMulNBits` → `SkipSimplifiedLayerNormalization` →
    /// `following MatMulNBits`) on GPU.
    ///
    /// The reference path runs the exact production kernels: a plain preceding
    /// GEMV, the standalone `skip_rmsnorm_f16_warp_half4` kernel (producing the
    /// normalized output and the residual sum), then a plain following GEMV. The
    /// fused path folds the residual add into the preceding GEMV's bias-slot
    /// epilogue and the RMS normalization into the following GEMV's prologue. The
    /// residual sum (`preceding fused output`) must equal the standalone norm's
    /// `input_skip_bias_sum`, and the final projection must be bit-identical —
    /// for decode (M==1) and prefill (M>1), with and without a following bias.
    #[test]
    fn fused_skip_rmsnorm_is_bit_exact_to_three_op_path() {
        run_fused_skip_rmsnorm_parity(DataType::Float16, 4, false);
    }

    /// Phi-4-mini exports its `SkipSimplifiedLayerNormalization` gamma in fp32.
    /// The fused RMS-norm-prologue GEMV must accept that fp32 gamma and stay
    /// bit-identical to the standalone (fp32-gamma) norm + GEMV pair, so the
    /// fusion fires on Phi as well as on Qwen (fp16 gamma).
    #[test]
    fn fused_skip_rmsnorm_fp32_gamma_is_bit_exact_to_three_op_path() {
        run_fused_skip_rmsnorm_parity(DataType::Float32, 4, false);
    }

    /// Phi-4-mini's qkv/down projections are int8 with non-trivial asymmetric
    /// zero points. The fused int8 RMS-norm-prologue GEMV (following) and the
    /// int8 residual-fold epilogue (preceding) must stay bit-identical to the
    /// standalone int8 GEMV + skip_rmsnorm + int8 GEMV sequence at Phi's dims
    /// (down K=8192>hidden=3072, qkv hidden=3072<=N=5120), fp32 gamma. The
    /// asymmetric zero points make this a mutation guard: ignoring the zero
    /// point (or dropping to fp16 accumulation) diverges from the reference.
    #[test]
    fn fused_skip_rmsnorm_int8_asymmetric_zp_is_bit_exact_to_three_op_path() {
        run_fused_skip_rmsnorm_parity(DataType::Float32, 8, true);
    }

    fn run_fused_skip_rmsnorm_parity(gamma_dtype: DataType, bits: usize, explicit_zp: bool) {
        let Some(runtime) = runtime() else {
            eprintln!("skipping fused skip-rmsnorm parity test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping fused skip-rmsnorm parity test: fp16 NVRTC headers unavailable");
            return;
        }

        // hidden % 128 == 0 (warp_half4 gate); preceding is a down projection
        // (pre_k > hidden), the following is a general projection (hidden <= n).
        // int8 exercises Phi's actual qkv/down dims; int4 keeps the Qwen shapes.
        let (hidden, pre_k, post_n) = if bits == 8 {
            (3072usize, 8192usize, 5120usize)
        } else {
            (896usize, QWEN_DOWN_K, 1152usize)
        };
        let epsilon = 1e-5f32;
        let block_size = 32usize;
        let blob_size = block_size * bits / 8;
        let device = DeviceId::cuda(0);

        for (m, following_bias) in [(1usize, false), (1, true), (5, true)] {
            let mut state = 0x51ce_d00d_f00d_1234u64;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            };

            // Pack random int4/int8 codes into the block-32 layout the kernels
            // unpack (nibble pairs for int4, one byte per weight for int8).
            let pack = |next: &mut dyn FnMut() -> f32, n: usize, k: usize| -> Vec<u8> {
                let k_blocks = k / block_size;
                let mut packed = vec![0u8; n * k_blocks * blob_size];
                if bits == 8 {
                    for byte in packed.iter_mut() {
                        *byte = ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
                    }
                    return packed;
                }
                let mut quant = vec![0u8; n * k];
                for value in quant.iter_mut() {
                    *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
                }
                for col in 0..n {
                    for block in 0..k_blocks {
                        for pair in 0..blob_size {
                            let low = quant[col * k + block * block_size + pair * 2] & 15;
                            let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                            packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                        }
                    }
                }
                packed
            };

            // Non-uniform asymmetric zero points, one byte per block (int8) so the
            // dequant is `(code - zp) * scale`. `None` keeps the symmetric default
            // (zp == 128 for int8), preserving the byte-identical int4 path.
            let zp_bytes = |next: &mut dyn FnMut() -> f32, n: usize, k: usize| -> Vec<u8> {
                let k_blocks = k / block_size;
                (0..n * k_blocks)
                    .map(|_| (128.0 + (next() * 16.0)).round().clamp(96.0, 160.0) as u8)
                    .collect()
            };

            let pre_k_blocks = pre_k / block_size;
            let post_k_blocks = hidden / block_size;

            let activation: Vec<f16> = (0..m * pre_k).map(|_| f16::from_f32(next())).collect();
            let packed_pre = pack(&mut next, hidden, pre_k);
            let scales_pre: Vec<f16> = (0..hidden * pre_k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();
            let packed_post = pack(&mut next, post_n, hidden);
            let scales_post: Vec<f16> = (0..post_n * post_k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();
            // The residual is a plain fp16 activation, one hidden vector per token.
            let residual: Vec<f16> = (0..m * hidden).map(|_| f16::from_f32(next())).collect();
            // Gamma values are produced identically regardless of storage dtype;
            // fp16 keeps the byte-identical Qwen path, fp32 exercises Phi's export.
            let gamma_is_f32 = gamma_dtype == DataType::Float32;
            let gamma_f32: Vec<f32> = (0..hidden)
                .map(|_| 0.5 + 0.5 * (next() * 0.5 + 0.5))
                .collect();
            let gamma_bytes: Vec<u8> = if gamma_is_f32 {
                gamma_f32.iter().flat_map(|v| v.to_le_bytes()).collect()
            } else {
                gamma_f32
                    .iter()
                    .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                    .collect()
            };
            let bias_post: Vec<f16> = (0..post_n).map(|_| f16::from_f32(next())).collect();
            // Per-block asymmetric zero points (int8 only); `explicit_zp == false`
            // uses the symmetric default and omits the input entirely.
            let zp_pre: Vec<u8> = if explicit_zp {
                zp_bytes(&mut next, hidden, pre_k)
            } else {
                Vec::new()
            };
            let zp_post: Vec<u8> = if explicit_zp {
                zp_bytes(&mut next, post_n, hidden)
            } else {
                Vec::new()
            };

            // Device buffers.
            let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
            let packed_pre_dev = runtime.alloc_raw(packed_pre.len()).unwrap();
            let scales_pre_dev = runtime.alloc_raw(scales_pre.len() * 2).unwrap();
            let packed_post_dev = runtime.alloc_raw(packed_post.len()).unwrap();
            let scales_post_dev = runtime.alloc_raw(scales_post.len() * 2).unwrap();
            let residual_dev = runtime.alloc_raw(residual.len() * 2).unwrap();
            let gamma_dev = runtime.alloc_raw(gamma_bytes.len()).unwrap();
            let bias_post_dev = runtime.alloc_raw(bias_post.len() * 2).unwrap();
            let matmul_out_dev = runtime.alloc_raw(m * hidden * 2).unwrap();
            let normalized_dev = runtime.alloc_raw(m * hidden * 2).unwrap();
            let sum_dev = runtime.alloc_raw(m * hidden * 2).unwrap();
            let mean_dev = runtime.alloc_raw(m * 2).unwrap();
            let invstd_dev = runtime.alloc_raw(m * 2).unwrap();
            let y_ref_dev = runtime.alloc_raw(m * post_n * 2).unwrap();
            let pre_fused_dev = runtime.alloc_raw(m * hidden * 2).unwrap();
            let y_fused_dev = runtime.alloc_raw(m * post_n * 2).unwrap();
            let zp_pre_dev = explicit_zp.then(|| runtime.alloc_raw(zp_pre.len()).unwrap());
            let zp_post_dev = explicit_zp.then(|| runtime.alloc_raw(zp_post.len()).unwrap());

            // SAFETY: device buffers exactly cover their source slices.
            unsafe {
                runtime.htod(as_bytes(&activation), activation_dev).unwrap();
                runtime.htod(&packed_pre, packed_pre_dev).unwrap();
                runtime.htod(as_bytes(&scales_pre), scales_pre_dev).unwrap();
                runtime.htod(&packed_post, packed_post_dev).unwrap();
                runtime
                    .htod(as_bytes(&scales_post), scales_post_dev)
                    .unwrap();
                runtime.htod(as_bytes(&residual), residual_dev).unwrap();
                runtime.htod(&gamma_bytes, gamma_dev).unwrap();
                runtime.htod(as_bytes(&bias_post), bias_post_dev).unwrap();
                if let Some(dev) = zp_pre_dev {
                    runtime.htod(&zp_pre, dev).unwrap();
                }
                if let Some(dev) = zp_post_dev {
                    runtime.htod(&zp_post, dev).unwrap();
                }
            }

            // Tensor descriptors.
            let pre_a_shape = [m, pre_k];
            let pre_a_strides = [pre_k as i64, 1];
            let pre_b_shape = [hidden, pre_k_blocks, blob_size];
            let pre_b_strides = [(pre_k_blocks * blob_size) as i64, blob_size as i64, 1];
            let pre_scales_shape = [hidden, pre_k_blocks];
            let pre_scales_strides = [pre_k_blocks as i64, 1];
            let hidden_shape = [m, hidden];
            let hidden_strides = [hidden as i64, 1];
            let gamma_shape = [hidden];
            let gamma_strides = [1i64];
            let post_b_shape = [post_n, post_k_blocks, blob_size];
            let post_b_strides = [(post_k_blocks * blob_size) as i64, blob_size as i64, 1];
            let post_scales_shape = [post_n, post_k_blocks];
            let post_scales_strides = [post_k_blocks as i64, 1];
            let post_bias_shape = [post_n];
            let post_bias_strides = [1i64];
            let y_shape = [m, post_n];
            let y_strides = [post_n as i64, 1];
            let stat_shape = [m];
            let stat_strides = [1i64];

            let activation_view = TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &pre_a_shape,
                &pre_a_strides,
                device,
            );
            let packed_pre_view = TensorView::new(
                device_ptr(packed_pre_dev),
                DataType::Uint8,
                &pre_b_shape,
                &pre_b_strides,
                device,
            );
            let scales_pre_view = TensorView::new(
                device_ptr(scales_pre_dev),
                DataType::Float16,
                &pre_scales_shape,
                &pre_scales_strides,
                device,
            );
            let packed_post_view = TensorView::new(
                device_ptr(packed_post_dev),
                DataType::Uint8,
                &post_b_shape,
                &post_b_strides,
                device,
            );
            let scales_post_view = TensorView::new(
                device_ptr(scales_post_dev),
                DataType::Float16,
                &post_scales_shape,
                &post_scales_strides,
                device,
            );
            let residual_view = TensorView::new(
                device_ptr(residual_dev),
                DataType::Float16,
                &hidden_shape,
                &hidden_strides,
                device,
            );
            let gamma_view = TensorView::new(
                device_ptr(gamma_dev),
                gamma_dtype,
                &gamma_shape,
                &gamma_strides,
                device,
            );
            let bias_post_view = TensorView::new(
                device_ptr(bias_post_dev),
                DataType::Float16,
                &post_bias_shape,
                &post_bias_strides,
                device,
            );
            let matmul_out_view = TensorView::new(
                device_ptr(matmul_out_dev),
                DataType::Float16,
                &hidden_shape,
                &hidden_strides,
                device,
            );
            let normalized_input_view = TensorView::new(
                device_ptr(normalized_dev),
                DataType::Float16,
                &hidden_shape,
                &hidden_strides,
                device,
            );
            let pre_fused_input_view = TensorView::new(
                device_ptr(pre_fused_dev),
                DataType::Float16,
                &hidden_shape,
                &hidden_strides,
                device,
            );
            // Asymmetric zero-point views (int8 only), one byte per block.
            let pre_zp_shape = [hidden, pre_k_blocks];
            let pre_zp_strides = [pre_k_blocks as i64, 1];
            let post_zp_shape = [post_n, post_k_blocks];
            let post_zp_strides = [post_k_blocks as i64, 1];
            let zp_pre_view = zp_pre_dev.map(|dev| {
                TensorView::new(
                    device_ptr(dev),
                    DataType::Uint8,
                    &pre_zp_shape,
                    &pre_zp_strides,
                    device,
                )
            });
            let zp_post_view = zp_post_dev.map(|dev| {
                TensorView::new(
                    device_ptr(dev),
                    DataType::Uint8,
                    &post_zp_shape,
                    &post_zp_strides,
                    device,
                )
            });

            let make_kernel = |k: usize, n: usize, fold: bool, rmsnorm: bool| MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: fold,
                gate_up_swiglu: false,
                rmsnorm_prologue: rmsnorm,
                rmsnorm_epsilon: epsilon,
                last_call_capture_safe: AtomicBool::new(false),
            };

            // ── Reference: preceding GEMV → skip_rmsnorm → following GEMV ──
            let preceding_ref = make_kernel(pre_k, hidden, false, false);
            {
                let mut matmul_out = TensorMut::new(
                    device_ptr_mut(matmul_out_dev),
                    DataType::Float16,
                    &hidden_shape,
                    &hidden_strides,
                    device,
                );
                preceding_ref
                    .run(
                        &{
                            let mut inputs =
                                vec![activation_view, packed_pre_view, scales_pre_view];
                            if let Some(zp) = zp_pre_view {
                                inputs.push(zp);
                            }
                            inputs
                        },
                        std::slice::from_mut(&mut matmul_out),
                    )
                    .unwrap();
            }

            let mut skip_node = Node::new(
                onnx_runtime_ir::NodeId(0),
                "SkipSimplifiedLayerNormalization",
                Vec::new(),
                Vec::new(),
            );
            skip_node
                .attributes
                .insert("epsilon".into(), onnx_runtime_ir::Attribute::Float(epsilon));
            let skip_kernel = crate::kernels::normalization::SkipSimplifiedLayerNormFactory {
                runtime: runtime.clone(),
            }
            .create(&skip_node, &[])
            .unwrap();
            {
                let normalized = TensorMut::new(
                    device_ptr_mut(normalized_dev),
                    DataType::Float16,
                    &hidden_shape,
                    &hidden_strides,
                    device,
                );
                let mean = TensorMut::new(
                    device_ptr_mut(mean_dev),
                    DataType::Float16,
                    &stat_shape,
                    &stat_strides,
                    device,
                );
                let invstd = TensorMut::new(
                    device_ptr_mut(invstd_dev),
                    DataType::Float16,
                    &stat_shape,
                    &stat_strides,
                    device,
                );
                let sum = TensorMut::new(
                    device_ptr_mut(sum_dev),
                    DataType::Float16,
                    &hidden_shape,
                    &hidden_strides,
                    device,
                );
                skip_kernel
                    .execute(
                        &[matmul_out_view, residual_view, gamma_view],
                        &mut [normalized, mean, invstd, sum],
                    )
                    .unwrap();
            }

            let following_ref = make_kernel(hidden, post_n, false, false);
            {
                let mut y_ref = TensorMut::new(
                    device_ptr_mut(y_ref_dev),
                    DataType::Float16,
                    &y_shape,
                    &y_strides,
                    device,
                );
                let mut inputs = vec![
                    normalized_input_view,
                    packed_post_view,
                    scales_post_view,
                ];
                if zp_post_view.is_some() || following_bias {
                    inputs.push(zp_post_view.unwrap_or(TensorView::absent(DataType::Uint8)));
                }
                if following_bias {
                    inputs.push(TensorView::absent(DataType::Int32));
                    inputs.push(bias_post_view);
                }
                following_ref
                    .run(&inputs, std::slice::from_mut(&mut y_ref))
                    .unwrap();
            }

            // ── Fused: residual epilogue in preceding, norm prologue in following ──
            let preceding_fused = make_kernel(pre_k, hidden, true, false);
            {
                let mut pre_fused = TensorMut::new(
                    device_ptr_mut(pre_fused_dev),
                    DataType::Float16,
                    &hidden_shape,
                    &hidden_strides,
                    device,
                );
                preceding_fused
                    .run(
                        &[
                            activation_view,
                            packed_pre_view,
                            scales_pre_view,
                            zp_pre_view.unwrap_or(TensorView::absent(DataType::Uint8)),
                            TensorView::absent(DataType::Int32),
                            residual_view,
                        ],
                        std::slice::from_mut(&mut pre_fused),
                    )
                    .unwrap();
            }

            let following_fused = make_kernel(hidden, post_n, false, true);
            {
                let mut y_fused = TensorMut::new(
                    device_ptr_mut(y_fused_dev),
                    DataType::Float16,
                    &y_shape,
                    &y_strides,
                    device,
                );
                let mut inputs = vec![
                    pre_fused_input_view,
                    packed_post_view,
                    scales_post_view,
                    zp_post_view.unwrap_or(TensorView::absent(DataType::Uint8)),
                    TensorView::absent(DataType::Int32),
                ];
                if following_bias {
                    inputs.push(bias_post_view);
                } else {
                    inputs.push(TensorView::absent(DataType::Float16));
                }
                inputs.push(gamma_view);
                following_fused
                    .run(&inputs, std::slice::from_mut(&mut y_fused))
                    .unwrap();
            }

            runtime.synchronize().unwrap();

            let mut sum_host = vec![f16::ZERO; m * hidden];
            let mut pre_fused_host = vec![f16::ZERO; m * hidden];
            let mut y_ref_host = vec![f16::ZERO; m * post_n];
            let mut y_fused_host = vec![f16::ZERO; m * post_n];
            // SAFETY: host buffers match their device sources.
            unsafe {
                runtime.dtoh(as_bytes_mut(&mut sum_host), sum_dev).unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut pre_fused_host), pre_fused_dev)
                    .unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut y_ref_host), y_ref_dev)
                    .unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut y_fused_host), y_fused_dev)
                    .unwrap();
                for buffer in [
                    activation_dev,
                    packed_pre_dev,
                    scales_pre_dev,
                    packed_post_dev,
                    scales_post_dev,
                    residual_dev,
                    gamma_dev,
                    bias_post_dev,
                    matmul_out_dev,
                    normalized_dev,
                    sum_dev,
                    mean_dev,
                    invstd_dev,
                    y_ref_dev,
                    pre_fused_dev,
                    y_fused_dev,
                ] {
                    runtime.free_raw(buffer).unwrap();
                }
                for buffer in [zp_pre_dev, zp_post_dev].into_iter().flatten() {
                    runtime.free_raw(buffer).unwrap();
                }
            }

            // The preceding fused output is the residual sum (input + skip).
            for index in 0..m * hidden {
                assert_eq!(
                    pre_fused_host[index].to_bits(),
                    sum_host[index].to_bits(),
                    "residual epilogue diverged from skip_rmsnorm sum at M={m}, \
                     following_bias={following_bias}, token={}, column={}",
                    index / hidden,
                    index % hidden
                );
            }
            // The fused projection matches the three-op sequence. It is normally
            // bit-identical, EXCEPT the asymmetric int8-zp M=1 case: the three-op
            // reference's standalone int8 GEMV now routes to the split-K entry
            // (K % 256 == 0, grid-starved), which reorders the fp32 block-sum
            // association across K_SPLIT cooperating warps (fp reassociation),
            // while the fused following kernel keeps the single-warp association.
            // Both are near-equal valid computations, so that path is validated to
            // a tight magnitude-relative tolerance instead of byte-identity.
            let splitk_path = bits == 8 && explicit_zp && m == 1;
            if splitk_path {
                let mut max_abs = 0.0f32;
                let mut worst = 0.0f32;
                for index in 0..m * post_n {
                    let fused = y_fused_host[index].to_f32();
                    let reference = y_ref_host[index].to_f32();
                    assert!(
                        fused.is_finite(),
                        "split-K int8-zp GEMV produced a non-finite output at M={m}, \
                         following_bias={following_bias}, column={}",
                        index % post_n
                    );
                    max_abs = max_abs.max(reference.abs());
                    worst = worst.max((fused - reference).abs());
                }
                let bound = (max_abs * 2e-3).max(1e-3);
                assert!(
                    worst < bound,
                    "fused norm prologue diverged from split-K skip_rmsnorm + GEMV at \
                     M={m}, following_bias={following_bias}: \
                     max_abs_diff={worst:.3e} bound={bound:.3e}"
                );
            } else {
                for index in 0..m * post_n {
                    assert_eq!(
                        y_fused_host[index].to_bits(),
                        y_ref_host[index].to_bits(),
                        "fused norm prologue diverged from skip_rmsnorm + GEMV at M={m}, \
                         following_bias={following_bias}, token={}, column={}",
                        index / post_n,
                        index % post_n
                    );
                }
            }
        }
    }
}

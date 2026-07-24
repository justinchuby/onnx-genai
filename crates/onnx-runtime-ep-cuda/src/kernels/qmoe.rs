//! CUDA implementation of ORT 1.27 `com.microsoft::QMoE`.
//!
//! Expert tensors remain resident on one GPU. Decode uses the Phase-1 per-route
//! GEMV path; prefill groups routes by expert, gathers contiguous activation
//! tiles, and uses a tiled affine block-dequant GEMM when an expert has multiple
//! assigned tokens. Weight paging, asynchronous prefetch, and expert-parallel
//! sharding are intentionally deferred.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::kernels::{qmoe_gemm, qmoe_grouping};
use crate::runtime::{CudaRuntime, cuptr};

const MODULE: &str = "qmoe_affine_v1";
const ROUTE_ENTRY: &str = "qmoe_route";
const ACTIVATE_ENTRY: &str = "qmoe_activate";
const LINEAR_F32_ENTRY: &str = "qmoe_linear_f32";
const LINEAR_F16_ENTRY: &str = "qmoe_linear_f16";
const LINEAR_BF16_ENTRY: &str = "qmoe_linear_bf16";
const COMBINE_F32_ENTRY: &str = "qmoe_combine_f32";
const COMBINE_F16_ENTRY: &str = "qmoe_combine_f16";
const COMBINE_BF16_ENTRY: &str = "qmoe_combine_bf16";

const CUDA_SRC: &str = r#"
#ifndef QMOE_BITS
#define QMOE_BITS 4
#endif
#ifndef QMOE_BLOCK_SIZE
#define QMOE_BLOCK_SIZE 16
#endif
#ifndef QMOE_HAS_ZERO_POINTS
#define QMOE_HAS_ZERO_POINTS 0
#endif

#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#define QMOE_HAS_HALF 1
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

__device__ __forceinline__ int total_order_key(float value)
{
    int bits = __float_as_int(value);
    bits ^= (bits >> 31) & 0x7fffffff;
    return bits;
}

__device__ __forceinline__ bool route_value_is_better(
    float candidate, int candidate_index, float best, int best_index)
{
    const int candidate_key = total_order_key(candidate);
    const int best_key = total_order_key(best);
    return candidate_key > best_key
        || (candidate_key == best_key && candidate_index < best_index);
}

extern "C" __global__ void qmoe_route(
    const float* router_probs,
    const float* router_weights,
    int* selected_experts,
    float* selected_weights,
    const unsigned long long rows,
    const int experts,
    const int top_k,
    const int normalize)
{
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long row = first; row < rows; row += stride) {
        const float* logits = router_probs + row * (unsigned long long)experts;
        int* indices = selected_experts + row * (unsigned long long)top_k;
        float* weights = selected_weights + row * (unsigned long long)top_k;

        for (int slot = 0; slot < top_k; ++slot) {
            int best_index = -1;
            float best_value = 0.0f;
            for (int expert = 0; expert < experts; ++expert) {
                bool already_selected = false;
                for (int previous = 0; previous < slot; ++previous) {
                    already_selected |= indices[previous] == expert;
                }
                if (already_selected) {
                    continue;
                }
                const float candidate = logits[expert];
                if (best_index < 0
                    || route_value_is_better(
                        candidate, expert, best_value, best_index)) {
                    best_index = expert;
                    best_value = candidate;
                }
            }
            indices[slot] = best_index;
        }

        if (router_weights) {
            const float* aggregation =
                router_weights + row * (unsigned long long)experts;
            float denominator = 1.0f;
            if (normalize) {
                denominator = 0.0f;
                for (int slot = 0; slot < top_k; ++slot) {
                    denominator += aggregation[indices[slot]];
                }
            }
            for (int slot = 0; slot < top_k; ++slot) {
                weights[slot] = denominator == 0.0f
                    ? 0.0f
                    : aggregation[indices[slot]] / denominator;
            }
            continue;
        }

        float maximum = -__int_as_float(0x7f800000);
        for (int expert = 0; expert < experts; ++expert) {
            maximum = fmaxf(maximum, logits[expert]);
        }
        float all_sum = 0.0f;
        for (int expert = 0; expert < experts; ++expert) {
            all_sum += expf(logits[expert] - maximum);
        }
        float denominator = all_sum;
        if (normalize) {
            denominator = 0.0f;
            for (int slot = 0; slot < top_k; ++slot) {
                denominator += expf(logits[indices[slot]] - maximum);
            }
        }
        for (int slot = 0; slot < top_k; ++slot) {
            weights[slot] =
                expf(logits[indices[slot]] - maximum) / denominator;
        }
    }
}

__device__ __forceinline__ float block_sum(float value)
{
    extern __shared__ float warp_sums[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < ((blockDim.x + 31) >> 5) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
    }
    return value;
}

template <int Bits, int BlockSize, bool HasZeroPoints>
__device__ __forceinline__ float decode_affine_weight(
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const int expert,
    const int output,
    const int depth,
    const int out_features,
    const int packed_in,
    const int blocks,
    const int zero_point_bytes)
{
    constexpr int PackSize = 8 / Bits;
    const unsigned long long expert_row =
        (unsigned long long)expert * out_features + output;
    const unsigned char byte =
        packed[expert_row * packed_in + depth / PackSize];
    constexpr int Mask = Bits == 8 ? 255 : ((1 << Bits) - 1);
    const int quantized = (byte >> ((depth % PackSize) * Bits)) & Mask;
    const int block = depth / BlockSize;
    int zero_point = 1 << (Bits - 1);
    if (HasZeroPoints) {
        const unsigned char packed_zero =
            zero_points[expert_row * zero_point_bytes + block / PackSize];
        zero_point =
            (packed_zero >> ((block % PackSize) * Bits)) & Mask;
    }
    return ((float)quantized - (float)zero_point)
        * scales[expert_row * blocks + block];
}

template <typename Input>
__device__ __forceinline__ float qmoe_load(
    const Input* input, unsigned long long index);

template <typename Input, int BlockSize, bool HasZeroPoints>
__device__ __forceinline__ float qmoe_int4_chunk(
    const Input* input,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const unsigned long long input_base,
    const unsigned long long expert_row,
    const int depth,
    const int packed_in,
    const int blocks,
    const int zero_point_bytes)
{
    // Int4 rows are multiples of eight packed bytes because block sizes are
    // powers of two >= 16, and chunk depths advance by eight values.
    const unsigned int packed_values =
        *reinterpret_cast<const unsigned int*>(
            packed + expert_row * packed_in + depth / 2);
    const int block = depth / BlockSize;
    int zero_point = 8;
    if (HasZeroPoints) {
        const unsigned char packed_zero =
            zero_points[expert_row * zero_point_bytes + block / 2];
        zero_point = (packed_zero >> ((block & 1) * 4)) & 15;
    }
    const float scale = scales[expert_row * blocks + block];
    float value = 0.0f;
#pragma unroll
    for (int offset = 0; offset < 8; ++offset) {
        const int quantized = (packed_values >> (offset * 4)) & 15;
        const float weight = ((float)quantized - (float)zero_point) * scale;
        value += qmoe_load(input, input_base + depth + offset) * weight;
    }
    return value;
}

template <>
__device__ __forceinline__ float qmoe_load<float>(
    const float* input, unsigned long long index)
{
    return input[index];
}

#ifdef QMOE_HAS_HALF
template <>
__device__ __forceinline__ float qmoe_load<__half>(
    const __half* input, unsigned long long index)
{
    return __half2float(input[index]);
}

template <>
__device__ __forceinline__ float qmoe_load<__nv_bfloat16>(
    const __nv_bfloat16* input, unsigned long long index)
{
    return __bfloat162float(input[index]);
}
#endif

template <typename Input, int Bits, int BlockSize, bool HasZeroPoints>
__device__ void qmoe_linear_impl(
    const Input* input,
    const int* selected_experts,
    const unsigned long long* expert_counts,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const unsigned long long routes,
    const unsigned long long gemm_min_tokens,
    const int input_rows_are_routes,
    const int top_k,
    const int out_features,
    const int in_features,
    const int packed_in,
    const int blocks,
    const int zero_point_bytes)
{
    const unsigned long long tasks =
        routes * (unsigned long long)out_features;
    for (unsigned long long task = blockIdx.x;
         task < tasks;
         task += gridDim.x) {
        const unsigned long long route = task / out_features;
        const int output_feature = (int)(task % out_features);
        const int expert = selected_experts[route];
        if (expert_counts
            && expert_counts[expert] >= gemm_min_tokens) {
            continue;
        }
        const unsigned long long input_row =
            input_rows_are_routes ? route : route / (unsigned long long)top_k;
        float value = 0.0f;
        const unsigned long long input_base =
            input_row * (unsigned long long)in_features;
        const unsigned long long expert_row =
            (unsigned long long)expert * out_features + output_feature;
        if (Bits == 4) {
            const int chunks = in_features / 8;
            for (int chunk = (int)threadIdx.x;
                 chunk < chunks;
                 chunk += (int)blockDim.x) {
                value += qmoe_int4_chunk<Input, BlockSize, HasZeroPoints>(
                    input, packed, scales, zero_points, input_base, expert_row,
                    chunk * 8, packed_in, blocks, zero_point_bytes);
            }
        } else {
            for (int depth = (int)threadIdx.x;
                 depth < in_features;
                 depth += (int)blockDim.x) {
                value += qmoe_load(input, input_base + depth)
                    * decode_affine_weight<Bits, BlockSize, HasZeroPoints>(
                        packed, scales, zero_points, expert, output_feature, depth,
                        out_features, packed_in, blocks, zero_point_bytes);
            }
        }
        value = block_sum(value);
        if (threadIdx.x == 0) {
            const unsigned long long bias_index =
                (unsigned long long)expert * out_features + output_feature;
            output[task] = value + (bias ? bias[bias_index] : 0.0f);
        }
        __syncthreads();
    }
}

extern "C" __global__ void qmoe_linear_f32(
    const float* input,
    const int* selected_experts,
    const unsigned long long* expert_counts,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const unsigned long long routes,
    const unsigned long long gemm_min_tokens,
    const int input_rows_are_routes,
    const int top_k,
    const int out_features,
    const int in_features,
    const int packed_in,
    const int blocks,
    const int zero_point_bytes)
{
    qmoe_linear_impl<float, QMOE_BITS, QMOE_BLOCK_SIZE, QMOE_HAS_ZERO_POINTS != 0>(
        input, selected_experts, expert_counts, packed, scales, zero_points, bias,
        output, routes, gemm_min_tokens, input_rows_are_routes, top_k, out_features, in_features,
        packed_in, blocks, zero_point_bytes);
}

#ifdef QMOE_HAS_HALF
extern "C" __global__ void qmoe_linear_f16(
    const __half* input,
    const int* selected_experts,
    const unsigned long long* expert_counts,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const unsigned long long routes,
    const unsigned long long gemm_min_tokens,
    const int input_rows_are_routes,
    const int top_k,
    const int out_features,
    const int in_features,
    const int packed_in,
    const int blocks,
    const int zero_point_bytes)
{
    qmoe_linear_impl<__half, QMOE_BITS, QMOE_BLOCK_SIZE, QMOE_HAS_ZERO_POINTS != 0>(
        input, selected_experts, expert_counts, packed, scales, zero_points, bias,
        output, routes, gemm_min_tokens, input_rows_are_routes, top_k, out_features, in_features,
        packed_in, blocks, zero_point_bytes);
}

extern "C" __global__ void qmoe_linear_bf16(
    const __nv_bfloat16* input,
    const int* selected_experts,
    const unsigned long long* expert_counts,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const unsigned long long routes,
    const unsigned long long gemm_min_tokens,
    const int input_rows_are_routes,
    const int top_k,
    const int out_features,
    const int in_features,
    const int packed_in,
    const int blocks,
    const int zero_point_bytes)
{
    qmoe_linear_impl<__nv_bfloat16, QMOE_BITS, QMOE_BLOCK_SIZE, QMOE_HAS_ZERO_POINTS != 0>(
        input, selected_experts, expert_counts, packed, scales, zero_points, bias,
        output, routes, gemm_min_tokens, input_rows_are_routes, top_k, out_features, in_features,
        packed_in, blocks, zero_point_bytes);
}
#endif

__device__ __forceinline__ float stable_sigmoid(float value)
{
    if (value >= 0.0f) {
        return 1.0f / (1.0f + expf(-value));
    }
    const float exponential = expf(value);
    return exponential / (1.0f + exponential);
}

__device__ __forceinline__ float swiglu_value(
    float gate,
    float linear,
    float alpha,
    float beta,
    float limit)
{
    const float bounded_gate = fminf(gate, limit);
    const float bounded_linear =
        isnan(linear) ? linear : fminf(fmaxf(linear, -limit), limit);
    return bounded_gate * stable_sigmoid(alpha * bounded_gate)
        * (bounded_linear + beta);
}

extern "C" __global__ void qmoe_activate(
    const float* fc1,
    const float* fc3,
    float* activated,
    const unsigned long long routes,
    const int inter,
    const int activation,
    const int swiglu_fusion,
    const float alpha,
    const float beta,
    const float swiglu_limit)
{
    const unsigned long long total = routes * (unsigned long long)inter;
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long index = first; index < total; index += stride) {
        const unsigned long long route = index / inter;
        const int feature = (int)(index % inter);
        const unsigned long long base =
            route * (unsigned long long)(activation == 3 && swiglu_fusion != 0
                ? inter * 2
                : inter);
        const float value = fc1[base + feature];
        if (activation == 0) {
            activated[index] = fmaxf(value, 0.0f);
        } else if (activation == 1) {
            const double x = (double)value;
            const double inner =
                0.7978845608028654 * (x + 0.044715 * x * x * x);
            activated[index] =
                (float)(0.5 * x * (1.0 + tanh(inner)));
        } else if (activation == 2 && !fc3) {
            activated[index] = value * stable_sigmoid(value);
        } else if (activation == 4) {
            activated[index] = value;
        } else {
            float gate;
            float linear;
            if (fc3) {
                gate = value;
                linear = fc3[index];
            } else if (swiglu_fusion == 1) {
                gate = fc1[base + 2 * feature];
                linear = fc1[base + 2 * feature + 1];
            } else {
                gate = value;
                linear = fc1[base + inter + feature];
            }
            activated[index] =
                swiglu_value(gate, linear, alpha, beta, swiglu_limit);
        }
    }
}

template <typename Output>
__device__ __forceinline__ void qmoe_store(
    Output* output, unsigned long long index, float value);

template <>
__device__ __forceinline__ void qmoe_store<float>(
    float* output, unsigned long long index, float value)
{
    output[index] = value;
}

#ifdef QMOE_HAS_HALF
template <>
__device__ __forceinline__ void qmoe_store<__half>(
    __half* output, unsigned long long index, float value)
{
    output[index] = __float2half_rn(value);
}

template <>
__device__ __forceinline__ void qmoe_store<__nv_bfloat16>(
    __nv_bfloat16* output, unsigned long long index, float value)
{
    output[index] = __float2bfloat16_rn(value);
}
#endif

template <typename Output>
__device__ void qmoe_combine_impl(
    const float* route_output,
    const float* selected_weights,
    Output* output,
    const unsigned long long rows,
    const int hidden,
    const int top_k)
{
    const unsigned long long total = rows * (unsigned long long)hidden;
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long index = first; index < total; index += stride) {
        const unsigned long long row = index / hidden;
        const int feature = (int)(index % hidden);
        float value = 0.0f;
        for (int slot = 0; slot < top_k; ++slot) {
            const unsigned long long route =
                row * (unsigned long long)top_k + slot;
            value += selected_weights[route]
                * route_output[route * (unsigned long long)hidden + feature];
        }
        qmoe_store(output, index, value);
    }
}

extern "C" __global__ void qmoe_combine_f32(
    const float* route_output,
    const float* selected_weights,
    float* output,
    const unsigned long long rows,
    const int hidden,
    const int top_k)
{
    qmoe_combine_impl(
        route_output, selected_weights, output, rows, hidden, top_k);
}

#ifdef QMOE_HAS_HALF
extern "C" __global__ void qmoe_combine_f16(
    const float* route_output,
    const float* selected_weights,
    __half* output,
    const unsigned long long rows,
    const int hidden,
    const int top_k)
{
    qmoe_combine_impl(
        route_output, selected_weights, output, rows, hidden, top_k);
}

extern "C" __global__ void qmoe_combine_bf16(
    const float* route_output,
    const float* selected_weights,
    __nv_bfloat16* output,
    const unsigned long long rows,
    const int hidden,
    const int top_k)
{
    qmoe_combine_impl(
        route_output, selected_weights, output, rows, hidden, top_k);
}
#endif
"#;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct QuantLayout {
    bits: usize,
    block_size: usize,
    has_zero_points: bool,
}

fn linear_module_source(layout: QuantLayout) -> (&'static str, &'static str) {
    static SOURCES: OnceLock<Mutex<HashMap<QuantLayout, (&'static str, &'static str)>>> =
        OnceLock::new();
    let sources = SOURCES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut sources = sources.lock().expect("QMoE source cache poisoned");
    if let Some(source) = sources.get(&layout) {
        return *source;
    }

    let zero_points = usize::from(layout.has_zero_points);
    let module = Box::leak(
        format!(
            "qmoe_affine_linear_v2_bits{}_block{}_zero_points{}",
            layout.bits, layout.block_size, zero_points
        )
        .into_boxed_str(),
    );
    let source = Box::leak(
        format!(
            "#define QMOE_BITS {}\n#define QMOE_BLOCK_SIZE {}\n\
             #define QMOE_HAS_ZERO_POINTS {}\n{}",
            layout.bits, layout.block_size, zero_points, CUDA_SRC
        )
        .into_boxed_str(),
    );
    sources.insert(layout, (module, source));
    (module, source)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activation {
    Relu,
    Gelu,
    Silu,
    Swiglu,
    Identity,
}

impl Activation {
    fn parse(node: &Node) -> Result<Self> {
        let name = match node.attr("activation_type") {
            Some(value) => value
                .as_str()
                .ok_or_else(|| error("attribute activation_type must be a string"))?,
            None => "relu",
        };
        match name {
            "relu" => Ok(Self::Relu),
            "gelu" => Ok(Self::Gelu),
            "silu" => Ok(Self::Silu),
            "swiglu" => Ok(Self::Swiglu),
            "identity" => Ok(Self::Identity),
            other => Err(error(format!(
                "unsupported activation_type '{other}' (supported: relu, gelu, silu, swiglu, identity)"
            ))),
        }
    }

    fn kernel_id(self) -> i32 {
        match self {
            Self::Relu => 0,
            Self::Gelu => 1,
            Self::Silu => 2,
            Self::Swiglu => 3,
            Self::Identity => 4,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MoeAttributes {
    k: usize,
    prefill_min_tokens: usize,
    activation: Activation,
    normalize_routing_weights: bool,
    swiglu_fusion: usize,
    activation_alpha: f32,
    activation_beta: f32,
    swiglu_limit: f32,
}

impl MoeAttributes {
    fn from_node(node: &Node) -> Result<Self> {
        let k = int_attr(node, "k", 1)?;
        if k <= 0 {
            return Err(error(format!("k must be > 0, got {k}")));
        }
        let activation = Activation::parse(node)?;
        let prefill_min_tokens = int_attr(node, "prefill_min_tokens", 2)?;
        if prefill_min_tokens < 2 {
            return Err(error(format!(
                "prefill_min_tokens must be at least 2, got {prefill_min_tokens}"
            )));
        }
        let normalize_routing_weights = bool_attr(node, "normalize_routing_weights", false)?;
        if bool_attr(node, "use_sparse_mixer", false)? {
            return Err(error(
                "use_sparse_mixer=1 is unsupported by the CUDA kernel",
            ));
        }
        let swiglu_fusion = int_attr(node, "swiglu_fusion", 0)?;
        if !(0..=2).contains(&swiglu_fusion) {
            return Err(error(format!(
                "swiglu_fusion must be 0, 1, or 2, got {swiglu_fusion}"
            )));
        }
        if activation != Activation::Swiglu && swiglu_fusion != 0 {
            return Err(error(
                "swiglu_fusion is only valid when activation_type='swiglu'",
            ));
        }
        Ok(Self {
            k: usize::try_from(k).map_err(|_| error("k exceeds usize limits"))?,
            prefill_min_tokens: usize::try_from(prefill_min_tokens)
                .map_err(|_| error("prefill_min_tokens exceeds usize limits"))?,
            activation,
            normalize_routing_weights,
            swiglu_fusion: swiglu_fusion as usize,
            activation_alpha: float_attr(node, "activation_alpha", 1.0)?,
            activation_beta: float_attr(node, "activation_beta", 0.0)?,
            swiglu_limit: float_attr(node, "swiglu_limit", f32::INFINITY)?,
        })
    }

    fn fc1_size(self, inter: usize) -> Result<usize> {
        if self.activation == Activation::Swiglu && self.swiglu_fusion != 0 {
            inter
                .checked_mul(2)
                .ok_or_else(|| error("fused SwiGLU FC1 width exceeds usize limits"))
        } else {
            Ok(inter)
        }
    }

    fn uses_separate_gate(self, has_fc3: bool) -> bool {
        (self.activation == Activation::Swiglu && self.swiglu_fusion == 0)
            || (self.activation == Activation::Silu && has_fc3)
    }
}

#[derive(Clone, Copy, Debug)]
enum FloatDtype {
    F32,
    F16,
    Bf16,
}

impl FloatDtype {
    fn from_input(dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::Float32 => Ok(Self::F32),
            DataType::Float16 => Ok(Self::F16),
            DataType::BFloat16 => Ok(Self::Bf16),
            other => Err(error(format!(
                "input requires Float32, Float16, or BFloat16, got {other:?}"
            ))),
        }
    }

    fn linear_entry(self) -> &'static str {
        match self {
            Self::F32 => LINEAR_F32_ENTRY,
            Self::F16 => LINEAR_F16_ENTRY,
            Self::Bf16 => LINEAR_BF16_ENTRY,
        }
    }

    fn combine_entry(self) -> &'static str {
        match self {
            Self::F32 => COMBINE_F32_ENTRY,
            Self::F16 => COMBINE_F16_ENTRY,
            Self::Bf16 => COMBINE_BF16_ENTRY,
        }
    }

    fn gather_entry(self) -> &'static str {
        match self {
            Self::F32 => qmoe_grouping::GATHER_F32_ENTRY,
            Self::F16 => qmoe_grouping::GATHER_F16_ENTRY,
            Self::Bf16 => qmoe_grouping::GATHER_BF16_ENTRY,
        }
    }

    fn needs_half_headers(self) -> bool {
        !matches!(self, Self::F32)
    }
}

pub struct QMoEFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for QMoEFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let attributes = MoeAttributes::from_node(node)?;
        let bits = int_attr(node, "expert_weight_bits", 4)?;
        if !matches!(bits, 1 | 2 | 4 | 8) {
            return Err(error(format!(
                "expert_weight_bits must be one of {{1, 2, 4, 8}}, got {bits}"
            )));
        }
        let block_size = int_attr(node, "block_size", 0)?;
        if block_size < 16 || !(block_size as usize).is_power_of_two() {
            return Err(error(format!(
                "block_size must be a power of two and at least 16, got {block_size}"
            )));
        }
        let quant_type = match node.attr("quant_type") {
            Some(value) => value
                .as_str()
                .ok_or_else(|| error("attribute quant_type must be a string"))?,
            None => "int",
        };
        if quant_type != "int" {
            return Err(error(format!(
                "quant_type='{quant_type}' is unsupported by CUDA QMoE; this kernel accepts only \
                 ORT integer-affine quant_type='int'. Native IQ/MXFP4 block layouts are not \
                 representable by QMoE's separate scales/zero-points inputs and require a \
                 block-quantized MoE operator"
            )));
        }
        Ok(Box::new(QMoEKernel {
            runtime: self.runtime.clone(),
            attributes,
            bits: bits as usize,
            block_size: block_size as usize,
            scratch: Mutex::new(ScratchPool::default()),
            warmed: AtomicBool::new(false),
        }))
    }
}

pub(crate) fn unsupported_reason(node: &Node) -> Option<Cow<'static, str>> {
    let bits = node
        .attr("expert_weight_bits")
        .map_or(Some(4), |value| value.as_int());
    match bits {
        Some(1 | 2 | 4 | 8) => {}
        Some(bits) => {
            return Some(Cow::Owned(format!(
                "QMoE: CUDA supports expert_weight_bits 1, 2, 4, or 8, got {bits} — requantize the expert weights to a supported width"
            )));
        }
        None => {
            return Some(Cow::Borrowed(
                "QMoE: expert_weight_bits must be an integer (supported: 1, 2, 4, 8)",
            ));
        }
    }
    match node.attr("block_size") {
        Some(attribute) => match attribute.as_int() {
            Some(value) if value >= 16 && (value as usize).is_power_of_two() => {}
            Some(value) => {
                return Some(Cow::Owned(format!(
                    "QMoE: CUDA requires block_size to be a power of two at least 16, got {value} — requantize the expert weights with a supported block size"
                )));
            }
            None => {
                return Some(Cow::Borrowed(
                    "QMoE: block_size must be an integer power of two at least 16",
                ));
            }
        },
        None => {
            return Some(Cow::Borrowed(
                "QMoE: missing integer block_size — export a power-of-two block size of at least 16",
            ));
        }
    }
    match node
        .attr("quant_type")
        .map_or(Some("int"), |value| value.as_str())
    {
        Some("int") => {}
        Some(quant_type) => {
            return Some(Cow::Owned(format!(
                "QMoE: CUDA supports only quant_type='int', got '{quant_type}' — use ORT integer-affine expert weights or a block-quantized MoE operator"
            )));
        }
        None => {
            return Some(Cow::Borrowed(
                "QMoE: quant_type must be the string 'int' for CUDA integer-affine expert weights",
            ));
        }
    }
    None
}

pub struct QMoEKernel {
    runtime: Arc<CudaRuntime>,
    attributes: MoeAttributes,
    bits: usize,
    block_size: usize,
    scratch: Mutex<ScratchPool>,
    warmed: AtomicBool,
}

#[derive(Clone, Copy)]
struct QuantizedExperts<'a> {
    packed: &'a TensorView<'a>,
    scales: &'a TensorView<'a>,
    zero_points: Option<&'a TensorView<'a>>,
    bias: Option<&'a TensorView<'a>>,
    out_features: usize,
    in_features: usize,
    packed_in: usize,
    blocks: usize,
    zero_point_bytes: usize,
}

#[derive(Clone, Copy)]
struct ExpertGrouping {
    counts: CUdeviceptr,
    offsets: CUdeviceptr,
    cursors: CUdeviceptr,
    grouped_routes: CUdeviceptr,
    grouped_input: CUdeviceptr,
}

impl<'a> QuantizedExperts<'a> {
    #[allow(clippy::too_many_arguments)]
    fn validate(
        name: &str,
        packed: &'a TensorView<'a>,
        scales: &'a TensorView<'a>,
        zero_points: Option<&'a TensorView<'a>>,
        bias: Option<&'a TensorView<'a>>,
        experts: usize,
        out_features: usize,
        in_features: usize,
        bits: usize,
        block_size: usize,
    ) -> Result<Self> {
        require_dtype(
            &format!("{name}_experts_weights"),
            packed.dtype,
            DataType::Uint8,
        )?;
        require_dtype(&format!("{name}_scales"), scales.dtype, DataType::Float32)?;
        let pack_size = 8 / bits;
        if !in_features.is_multiple_of(pack_size) {
            return Err(error(format!(
                "{name} input features {in_features} must be divisible by pack_size {pack_size}"
            )));
        }
        if !in_features.is_multiple_of(block_size) {
            return Err(error(format!(
                "{name} input features {in_features} must be divisible by block_size {block_size}"
            )));
        }
        let packed_in = in_features / pack_size;
        let blocks = in_features / block_size;
        let zero_point_bytes = checked_div_ceil(
            blocks,
            pack_size,
            &format!("{name} zero-point row byte count"),
        )?;
        require_shape(
            &format!("{name}_experts_weights"),
            packed.shape,
            &[experts, out_features, packed_in],
        )?;
        require_shape(
            &format!("{name}_scales"),
            scales.shape,
            &[experts, out_features, blocks],
        )?;
        if let Some(zero_points) = zero_points {
            require_dtype(
                &format!("{name}_zero_points"),
                zero_points.dtype,
                DataType::Uint8,
            )?;
            require_shape(
                &format!("{name}_zero_points"),
                zero_points.shape,
                &[experts, out_features, zero_point_bytes],
            )?;
        }
        if let Some(bias) = bias {
            require_dtype(
                &format!("{name}_experts_bias"),
                bias.dtype,
                DataType::Float32,
            )?;
            require_shape(
                &format!("{name}_experts_bias"),
                bias.shape,
                &[experts, out_features],
            )?;
        }
        for (tensor_name, tensor) in [
            (format!("{name}_experts_weights"), Some(packed)),
            (format!("{name}_scales"), Some(scales)),
            (format!("{name}_zero_points"), zero_points),
            (format!("{name}_experts_bias"), bias),
        ] {
            if let Some(tensor) = tensor {
                checked_tensor_layout(&tensor_name, tensor.shape, tensor.dtype)?;
                if !tensor.is_contiguous() {
                    return Err(error(format!(
                        "{tensor_name} must be contiguous on the CUDA execution provider"
                    )));
                }
            }
        }
        Ok(Self {
            packed,
            scales,
            zero_points,
            bias,
            out_features,
            in_features,
            packed_in,
            blocks,
            zero_point_bytes,
        })
    }
}

impl Kernel for QMoEKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(7..=21).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(error(format!(
                "expected 7 to 21 inputs and exactly 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        for (index, name) in [
            (0, "input"),
            (1, "router_probs"),
            (2, "fc1_experts_weights"),
            (3, "fc1_scales"),
            (5, "fc2_experts_weights"),
            (6, "fc2_scales"),
        ] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{name}') is absent"
                )));
            }
        }
        if let Some((index, _)) = inputs
            .iter()
            .enumerate()
            .skip(15)
            .find(|(_, input)| !input.is_absent())
        {
            return Err(error(format!(
                "input {index} is only used by FP4/FP8 QMoE modes, which are deferred"
            )));
        }

        let dtype = FloatDtype::from_input(inputs[0].dtype)?;
        if outputs[0].dtype != inputs[0].dtype {
            return Err(error(format!(
                "output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, inputs[0].dtype
            )));
        }
        require_dtype("router_probs", inputs[1].dtype, DataType::Float32)?;
        if dtype.needs_half_headers() {
            self.runtime.require_nvrtc_half_headers("QMoE")?;
        }

        let input_shape = inputs[0].shape;
        if !matches!(input_shape.len(), 2 | 3) {
            return Err(error(format!(
                "input must be 2-D [rows, hidden] or 3-D [batch, sequence, hidden], got {input_shape:?}"
            )));
        }
        require_shape("output", outputs[0].shape, input_shape)?;
        let hidden = *input_shape
            .last()
            .ok_or_else(|| error("input rank unexpectedly empty"))?;
        let rows = checked_product(
            &input_shape[..input_shape.len() - 1],
            "flattened input row count",
        )?;
        require_rank("router_probs", inputs[1].shape, 2)?;
        if inputs[1].shape[0] != rows {
            return Err(error(format!(
                "router_probs rows {} must equal flattened input rows {rows}",
                inputs[1].shape[0]
            )));
        }
        let experts = inputs[1].shape[1];
        if self.attributes.k > experts {
            return Err(error(format!(
                "requires 0 < k <= num_experts, got k={} and num_experts={experts}",
                self.attributes.k
            )));
        }
        if !hidden.is_multiple_of(self.block_size) {
            return Err(error(format!(
                "hidden_size {hidden} must be divisible by block_size {}",
                self.block_size
            )));
        }

        require_rank("fc2_experts_weights", inputs[5].shape, 3)?;
        if inputs[5].shape[0] != experts || inputs[5].shape[1] != hidden {
            return Err(error(format!(
                "fc2_experts_weights must start with [experts={experts}, hidden={hidden}], got {:?}",
                inputs[5].shape
            )));
        }
        let pack_size = 8 / self.bits;
        let inter = inputs[5].shape[2]
            .checked_mul(pack_size)
            .ok_or_else(|| error("fc2 inter_size exceeds usize limits"))?;
        if inter == 0 || !inter.is_multiple_of(self.block_size) {
            return Err(error(format!(
                "inferred inter_size {inter} must be non-zero and divisible by block_size {}",
                self.block_size
            )));
        }
        let fc1_size = self.attributes.fc1_size(inter)?;

        let fc1 = QuantizedExperts::validate(
            "fc1",
            &inputs[2],
            &inputs[3],
            optional_input(inputs, 11),
            optional_input(inputs, 4),
            experts,
            fc1_size,
            hidden,
            self.bits,
            self.block_size,
        )?;
        let fc2 = QuantizedExperts::validate(
            "fc2",
            &inputs[5],
            &inputs[6],
            optional_input(inputs, 12),
            optional_input(inputs, 7),
            experts,
            hidden,
            inter,
            self.bits,
            self.block_size,
        )?;

        let has_fc3 = optional_input(inputs, 8).is_some();
        let uses_separate_gate = self.attributes.uses_separate_gate(has_fc3);
        let fc3 = if uses_separate_gate {
            Some(QuantizedExperts::validate(
                "fc3",
                optional_input(inputs, 8)
                    .ok_or_else(|| error("unfused swiglu requires input 8 fc3_experts_weights"))?,
                optional_input(inputs, 9)
                    .ok_or_else(|| error("fc3_experts_weights requires input 9 fc3_scales"))?,
                optional_input(inputs, 13),
                optional_input(inputs, 10),
                experts,
                inter,
                hidden,
                self.bits,
                self.block_size,
            )?)
        } else {
            for (index, name) in [
                (8, "fc3_experts_weights"),
                (9, "fc3_scales"),
                (10, "fc3_experts_bias"),
                (13, "fc3_zero_points"),
            ] {
                if optional_input(inputs, index).is_some() {
                    return Err(error(format!(
                        "{name} is only valid for unfused swiglu or silu gated-GLU"
                    )));
                }
            }
            None
        };

        if let Some(router_weights) = optional_input(inputs, 14) {
            require_dtype("router_weights", router_weights.dtype, DataType::Float32)?;
            require_shape("router_weights", router_weights.shape, &[rows, experts])?;
        }
        for (name, tensor) in [("input", &inputs[0]), ("router_probs", &inputs[1])] {
            checked_tensor_layout(name, tensor.shape, tensor.dtype)?;
            if !tensor.is_contiguous() {
                return Err(error(format!(
                    "{name} must be contiguous on the CUDA execution provider"
                )));
            }
        }
        if let Some(router_weights) = optional_input(inputs, 14) {
            checked_tensor_layout("router_weights", router_weights.shape, router_weights.dtype)?;
            if !router_weights.is_contiguous() {
                return Err(error(
                    "router_weights must be contiguous on the CUDA execution provider",
                ));
            }
        }
        checked_tensor_layout("output", outputs[0].shape, outputs[0].dtype)?;
        if !outputs[0].is_contiguous() {
            return Err(error(
                "output must be contiguous on the CUDA execution provider",
            ));
        }
        if rows == 0 || hidden == 0 {
            return Ok(());
        }

        let routes = checked_product(&[rows, self.attributes.k], "route count")?;
        let route_index_bytes = checked_bytes(routes, std::mem::size_of::<i32>(), "route indices")?;
        let route_weight_bytes =
            checked_bytes(routes, std::mem::size_of::<f32>(), "route weights")?;
        let fc1_elements = checked_product(&[routes, fc1_size], "FC1 scratch element count")?;
        let fc1_bytes = checked_bytes(fc1_elements, 4, "FC1 scratch")?;
        let activated_elements =
            checked_product(&[routes, inter], "activation scratch element count")?;
        let activated_bytes = checked_bytes(activated_elements, 4, "activation scratch")?;
        let route_output_elements =
            checked_product(&[routes, hidden], "route output element count")?;
        let route_output_bytes = checked_bytes(route_output_elements, 4, "route output scratch")?;
        let grouping_sizes = (rows > 1)
            .then(|| {
                let expert_entries = experts
                    .checked_add(1)
                    .ok_or_else(|| error("expert offset entry count exceeds usize limits"))?;
                let counts =
                    checked_bytes(experts, std::mem::size_of::<u64>(), "expert token counts")?;
                let offsets = checked_bytes(
                    expert_entries,
                    std::mem::size_of::<u64>(),
                    "expert token offsets",
                )?;
                let grouped_routes =
                    checked_bytes(routes, std::mem::size_of::<u64>(), "grouped route indices")?;
                let grouped_features = hidden.max(inter);
                let grouped_elements = checked_product(
                    &[routes, grouped_features],
                    "grouped activation element count",
                )?;
                let grouped_input =
                    checked_bytes(grouped_elements, 4, "grouped activation scratch")?;
                Ok::<_, EpError>((counts, offsets, grouped_routes, grouped_input))
            })
            .transpose()?;

        let capturing = self.runtime.is_capturing()?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| error("QMoE scratch pool mutex poisoned"))?;
        let route_indices = scratch.ensure(&self.runtime, 0, route_index_bytes, capturing)?;
        let route_weights = scratch.ensure(&self.runtime, 1, route_weight_bytes, capturing)?;
        let fc1_output = scratch.ensure(&self.runtime, 2, fc1_bytes, capturing)?;
        let fc3_output = fc3
            .map(|_| scratch.ensure(&self.runtime, 3, activated_bytes, capturing))
            .transpose()?;
        let activated = scratch.ensure(&self.runtime, 4, activated_bytes, capturing)?;
        let route_output = scratch.ensure(&self.runtime, 5, route_output_bytes, capturing)?;
        let grouping = grouping_sizes
            .map(
                |(counts_bytes, offsets_bytes, grouped_routes_bytes, grouped_input_bytes)| {
                    Ok::<_, EpError>(ExpertGrouping {
                        counts: scratch.ensure(&self.runtime, 6, counts_bytes, capturing)?,
                        offsets: scratch.ensure(&self.runtime, 7, offsets_bytes, capturing)?,
                        cursors: scratch.ensure(&self.runtime, 8, counts_bytes, capturing)?,
                        grouped_routes: scratch.ensure(
                            &self.runtime,
                            9,
                            grouped_routes_bytes,
                            capturing,
                        )?,
                        grouped_input: scratch.ensure(
                            &self.runtime,
                            10,
                            grouped_input_bytes,
                            capturing,
                        )?,
                    })
                },
            )
            .transpose()?;

        self.launch_route(
            &inputs[1],
            optional_input(inputs, 14),
            route_indices,
            route_weights,
            rows,
            experts,
        )?;
        if let Some(grouping) = grouping {
            self.launch_grouping(route_indices, grouping, routes, experts)?;
            self.launch_gather(
                dtype,
                tensor_ptr(&inputs[0]),
                grouping,
                routes,
                rows,
                hidden,
                false,
            )?;
            self.launch_grouped_linear(grouping, fc1, fc1_output, routes, experts)?;
            self.launch_linear(
                dtype,
                tensor_ptr(&inputs[0]),
                route_indices,
                Some(grouping.counts),
                fc1,
                fc1_output,
                routes,
                false,
            )?;
            if let (Some(fc3), Some(fc3_output)) = (fc3, fc3_output) {
                self.launch_grouped_linear(grouping, fc3, fc3_output, routes, experts)?;
                self.launch_linear(
                    dtype,
                    tensor_ptr(&inputs[0]),
                    route_indices,
                    Some(grouping.counts),
                    fc3,
                    fc3_output,
                    routes,
                    false,
                )?;
            }
            self.launch_activation(fc1_output, fc3_output, activated, routes, inter)?;
            self.launch_gather(
                FloatDtype::F32,
                activated,
                grouping,
                routes,
                routes,
                inter,
                true,
            )?;
            self.launch_grouped_linear(grouping, fc2, route_output, routes, experts)?;
            self.launch_linear(
                FloatDtype::F32,
                activated,
                route_indices,
                Some(grouping.counts),
                fc2,
                route_output,
                routes,
                true,
            )?;
        } else {
            self.launch_linear(
                dtype,
                tensor_ptr(&inputs[0]),
                route_indices,
                None,
                fc1,
                fc1_output,
                routes,
                false,
            )?;
            if let (Some(fc3), Some(fc3_output)) = (fc3, fc3_output) {
                self.launch_linear(
                    dtype,
                    tensor_ptr(&inputs[0]),
                    route_indices,
                    None,
                    fc3,
                    fc3_output,
                    routes,
                    false,
                )?;
            }
            self.launch_activation(fc1_output, fc3_output, activated, routes, inter)?;
            self.launch_linear(
                FloatDtype::F32,
                activated,
                route_indices,
                None,
                fc2,
                route_output,
                routes,
                true,
            )?;
        }
        self.launch_combine(
            dtype,
            route_output,
            route_weights,
            &mut outputs[0],
            rows,
            hidden,
        )?;
        let result = if capturing {
            Ok(())
        } else {
            self.runtime.synchronize()
        };
        if result.is_ok() && !capturing {
            self.warmed.store(true, Ordering::Relaxed);
        }
        result
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.warmed.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "requires a warmed fixed-shape eager QMoE pass to size the pooled scratch and \
                 compile every routed expert kernel",
            )
        }
    }
}

impl QMoEKernel {
    fn launch_route(
        &self,
        router_probs: &TensorView,
        router_weights: Option<&TensorView>,
        route_indices: CUdeviceptr,
        route_weights: CUdeviceptr,
        rows: usize,
        experts: usize,
    ) -> Result<()> {
        let function = self.runtime.nvrtc_function(MODULE, CUDA_SRC, ROUTE_ENTRY)?;
        let router_probs = tensor_ptr(router_probs);
        let router_weights = router_weights.map(tensor_ptr).unwrap_or(0);
        let rows = as_u64("row count", rows)?;
        let experts = as_i32("expert count", experts)?;
        let top_k = as_i32("top-k", self.attributes.k)?;
        let normalize = i32::from(self.attributes.normalize_routing_weights);
        let config = self.pointwise_launch_config(rows)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&router_probs)
            .arg(&router_weights)
            .arg(&route_indices)
            .arg(&route_weights)
            .arg(&rows)
            .arg(&experts)
            .arg(&top_k)
            .arg(&normalize);
        // SAFETY: tensor layouts and scratch sizes were validated, and the ABI
        // matches `qmoe_route`.
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch QMoE routing", err))
    }

    fn launch_grouping(
        &self,
        route_indices: CUdeviceptr,
        grouping: ExpertGrouping,
        routes: usize,
        experts: usize,
    ) -> Result<()> {
        let routes_u64 = as_u64("route count", routes)?;
        let experts_i32 = as_i32("expert count", experts)?;
        let expert_entries = experts
            .checked_add(1)
            .ok_or_else(|| error("expert offset entry count exceeds usize limits"))?;
        let init_total = routes.max(expert_entries);

        let init = self.runtime.nvrtc_function(
            qmoe_grouping::MODULE,
            qmoe_grouping::CUDA_SRC,
            qmoe_grouping::INIT_ENTRY,
        )?;
        let mut builder = self.runtime.stream().launch_builder(&init);
        builder
            .arg(&grouping.counts)
            .arg(&grouping.offsets)
            .arg(&grouping.cursors)
            .arg(&grouping.grouped_routes)
            .arg(&routes_u64)
            .arg(&experts_i32);
        // SAFETY: all grouping buffers have their checked counts/offsets/routes
        // sizes, and the scalar ABI matches `qmoe_group_init`.
        unsafe {
            builder.launch(self.pointwise_launch_config(as_u64(
                "group initialization element count",
                init_total,
            )?)?)
        }
        .map_err(|err| driver_err("initialize QMoE expert grouping", err))?;

        let count = self.runtime.nvrtc_function(
            qmoe_grouping::MODULE,
            qmoe_grouping::CUDA_SRC,
            qmoe_grouping::COUNT_ENTRY,
        )?;
        let mut builder = self.runtime.stream().launch_builder(&count);
        builder
            .arg(&route_indices)
            .arg(&grouping.counts)
            .arg(&routes_u64)
            .arg(&experts_i32);
        // SAFETY: route_indices covers `routes` and counts covers `experts`.
        unsafe { builder.launch(self.pointwise_launch_config(routes_u64)?) }
            .map_err(|err| driver_err("count QMoE routes by expert", err))?;

        let prefix = self.runtime.nvrtc_function(
            qmoe_grouping::MODULE,
            qmoe_grouping::CUDA_SRC,
            qmoe_grouping::PREFIX_ENTRY,
        )?;
        let mut builder = self.runtime.stream().launch_builder(&prefix);
        builder
            .arg(&grouping.counts)
            .arg(&grouping.offsets)
            .arg(&routes_u64)
            .arg(&experts_i32);
        // SAFETY: the single-thread prefix kernel reads `experts` counts and
        // writes `experts + 1` offsets.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|err| driver_err("scan QMoE expert token offsets", err))?;

        let assign = self.runtime.nvrtc_function(
            qmoe_grouping::MODULE,
            qmoe_grouping::CUDA_SRC,
            qmoe_grouping::ASSIGN_ENTRY,
        )?;
        let mut builder = self.runtime.stream().launch_builder(&assign);
        builder
            .arg(&route_indices)
            .arg(&grouping.offsets)
            .arg(&grouping.cursors)
            .arg(&grouping.grouped_routes)
            .arg(&routes_u64)
            .arg(&experts_i32);
        // SAFETY: offsets and cursors cover all experts, grouped_routes covers
        // all routes, and every device-side write is bounds guarded.
        unsafe { builder.launch(self.pointwise_launch_config(routes_u64)?) }
            .map(|_| ())
            .map_err(|err| driver_err("assign QMoE grouped routes", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gather(
        &self,
        dtype: FloatDtype,
        input: CUdeviceptr,
        grouping: ExpertGrouping,
        routes: usize,
        input_rows: usize,
        features: usize,
        input_rows_are_routes: bool,
    ) -> Result<()> {
        let function = self.runtime.nvrtc_function(
            qmoe_grouping::MODULE,
            qmoe_grouping::CUDA_SRC,
            dtype.gather_entry(),
        )?;
        let total = checked_product(&[routes, features], "grouped gather element count")?;
        let routes = as_u64("route count", routes)?;
        let input_rows = as_u64("gather input row count", input_rows)?;
        let input_rows_are_routes = i32::from(input_rows_are_routes);
        let top_k = as_i32("top-k", self.attributes.k)?;
        let features = as_i32("gather feature count", features)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input)
            .arg(&grouping.grouped_routes)
            .arg(&grouping.grouped_input)
            .arg(&routes)
            .arg(&input_rows)
            .arg(&input_rows_are_routes)
            .arg(&top_k)
            .arg(&features);
        // SAFETY: grouped_routes covers every route, grouped_input covers
        // routes*features f32 values, and source row selection is bounds guarded.
        unsafe {
            builder.launch(
                self.pointwise_launch_config(as_u64("grouped gather element count", total)?)?,
            )
        }
        .map(|_| ())
        .map_err(|err| driver_err("gather QMoE expert activation rows", err))
    }

    fn launch_grouped_linear(
        &self,
        grouping: ExpertGrouping,
        weights: QuantizedExperts<'_>,
        output: CUdeviceptr,
        routes: usize,
        experts: usize,
    ) -> Result<()> {
        let capabilities = self.runtime.capabilities();
        let preferred_threads = self.preferred_reduction_threads();
        let tile = qmoe_gemm::tile_for(
            capabilities.compute_capability(),
            preferred_threads,
            capabilities.max_shared_memory_per_block_optin(),
        );
        let (module, source) = qmoe_gemm::module_source(tile);
        let function = self
            .runtime
            .nvrtc_function(module, source, qmoe_gemm::ENTRY)?;
        let tasks = checked_product(
            &[experts, weights.out_features],
            "grouped linear expert-feature task count",
        )?;
        let config = self.runtime.reduction_launch_config(
            &function,
            self.reduction_grid(tasks)?,
            preferred_threads,
            tile.checked_mul(std::mem::size_of::<f32>() as u32)
                .ok_or_else(|| error("grouped GEMM shared-memory stride overflow"))?,
        )?;
        let packed = tensor_ptr(weights.packed);
        let scales = tensor_ptr(weights.scales);
        let zero_points = weights.zero_points.map(tensor_ptr).unwrap_or(0);
        let bias = weights.bias.map(tensor_ptr).unwrap_or(0);
        let routes = as_u64("route count", routes)?;
        let tasks = as_u64("grouped linear task count", tasks)?;
        let gemm_min_tokens = as_u64(
            "prefill GEMM token threshold",
            self.attributes.prefill_min_tokens,
        )?;
        let experts = as_i32("expert count", experts)?;
        let out_features = as_i32("output feature count", weights.out_features)?;
        let in_features = as_i32("input feature count", weights.in_features)?;
        let packed_in = as_i32("packed input width", weights.packed_in)?;
        let blocks = as_i32("block count", weights.blocks)?;
        let zero_point_bytes = as_i32("zero-point row byte count", weights.zero_point_bytes)?;
        let bits = as_i32("expert weight bits", self.bits)?;
        let block_size = as_i32("block size", self.block_size)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&grouping.grouped_input)
            .arg(&grouping.grouped_routes)
            .arg(&grouping.counts)
            .arg(&grouping.offsets)
            .arg(&packed)
            .arg(&scales)
            .arg(&zero_points)
            .arg(&bias)
            .arg(&output)
            .arg(&routes)
            .arg(&tasks)
            .arg(&gemm_min_tokens)
            .arg(&experts)
            .arg(&out_features)
            .arg(&in_features)
            .arg(&packed_in)
            .arg(&blocks)
            .arg(&zero_point_bytes)
            .arg(&bits)
            .arg(&block_size);
        // SAFETY: grouped rows, expert metadata, packed weights, and outputs all
        // have checked sizes. The kernel guards empty experts and every scatter.
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch QMoE grouped block-dequant GEMM", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_linear(
        &self,
        dtype: FloatDtype,
        input_ptr: CUdeviceptr,
        route_indices: CUdeviceptr,
        expert_counts: Option<CUdeviceptr>,
        weights: QuantizedExperts<'_>,
        output: CUdeviceptr,
        routes: usize,
        input_rows_are_routes: bool,
    ) -> Result<()> {
        let layout = QuantLayout {
            bits: self.bits,
            block_size: self.block_size,
            has_zero_points: weights.zero_points.is_some(),
        };
        let (module, source) = linear_module_source(layout);
        let function = self
            .runtime
            .nvrtc_function(module, source, dtype.linear_entry())?;
        let packed = tensor_ptr(weights.packed);
        let expert_counts = expert_counts.unwrap_or(0);
        let scales = tensor_ptr(weights.scales);
        let zero_points = weights.zero_points.map(tensor_ptr).unwrap_or(0);
        let bias = weights.bias.map(tensor_ptr).unwrap_or(0);
        let tasks = checked_product(&[routes, weights.out_features], "linear output task count")?;
        let grid_x = self.reduction_grid(tasks)?;
        let config = self.runtime.reduction_launch_config(
            &function,
            grid_x,
            self.preferred_reduction_threads(),
            std::mem::size_of::<f32>() as u32,
        )?;
        let routes = as_u64("route count", routes)?;
        let gemm_min_tokens = as_u64(
            "prefill GEMM token threshold",
            self.attributes.prefill_min_tokens,
        )?;
        let input_rows_are_routes = i32::from(input_rows_are_routes);
        let top_k = as_i32("top-k", self.attributes.k)?;
        let out_features = as_i32("output feature count", weights.out_features)?;
        let in_features = as_i32("input feature count", weights.in_features)?;
        let packed_in = as_i32("packed input width", weights.packed_in)?;
        let blocks = as_i32("block count", weights.blocks)?;
        let zero_point_bytes = as_i32("zero-point row byte count", weights.zero_point_bytes)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_ptr)
            .arg(&route_indices)
            .arg(&expert_counts)
            .arg(&packed)
            .arg(&scales)
            .arg(&zero_points)
            .arg(&bias)
            .arg(&output)
            .arg(&routes)
            .arg(&gemm_min_tokens)
            .arg(&input_rows_are_routes)
            .arg(&top_k)
            .arg(&out_features)
            .arg(&in_features)
            .arg(&packed_in)
            .arg(&blocks)
            .arg(&zero_point_bytes);
        // SAFETY: all packed tensors and scratch buffers cover the validated
        // expert-major ranges, and the scalar ABI matches `qmoe_linear_*`.
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch QMoE block-dequant expert GEMV", err))
    }

    fn launch_activation(
        &self,
        fc1: CUdeviceptr,
        fc3: Option<CUdeviceptr>,
        activated: CUdeviceptr,
        routes: usize,
        inter: usize,
    ) -> Result<()> {
        let function = self
            .runtime
            .nvrtc_function(MODULE, CUDA_SRC, ACTIVATE_ENTRY)?;
        let total = checked_product(&[routes, inter], "activation element count")?;
        let config = self.pointwise_launch_config(as_u64("activation element count", total)?)?;
        let fc3 = fc3.unwrap_or(0);
        let routes = as_u64("route count", routes)?;
        let inter = as_i32("intermediate feature count", inter)?;
        let activation = self.attributes.activation.kernel_id();
        let swiglu_fusion = as_i32("swiglu_fusion", self.attributes.swiglu_fusion)?;
        let alpha = self.attributes.activation_alpha;
        let beta = self.attributes.activation_beta;
        let limit = self.attributes.swiglu_limit;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&fc1)
            .arg(&fc3)
            .arg(&activated)
            .arg(&routes)
            .arg(&inter)
            .arg(&activation)
            .arg(&swiglu_fusion)
            .arg(&alpha)
            .arg(&beta)
            .arg(&limit);
        // SAFETY: scratch buffers cover every routed intermediate element and
        // the ABI matches `qmoe_activate`.
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch QMoE activation", err))
    }

    fn launch_combine(
        &self,
        dtype: FloatDtype,
        route_output: CUdeviceptr,
        route_weights: CUdeviceptr,
        output: &mut TensorMut,
        rows: usize,
        hidden: usize,
    ) -> Result<()> {
        let function = self
            .runtime
            .nvrtc_function(MODULE, CUDA_SRC, dtype.combine_entry())?;
        let total = checked_product(&[rows, hidden], "combined output element count")?;
        let config = self.pointwise_launch_config(as_u64("output element count", total)?)?;
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let rows = as_u64("row count", rows)?;
        let hidden = as_i32("hidden feature count", hidden)?;
        let top_k = as_i32("top-k", self.attributes.k)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&route_output)
            .arg(&route_weights)
            .arg(&output_ptr)
            .arg(&rows)
            .arg(&hidden)
            .arg(&top_k);
        // SAFETY: routed output and weights cover rows*top_k, output covers
        // rows*hidden, and the ABI matches `qmoe_combine_*`.
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch QMoE weighted combine", err))
    }

    fn preferred_reduction_threads(&self) -> u32 {
        let capabilities = self.runtime.capabilities();
        let preferred = if capabilities.compute_capability().0 >= 7 {
            256
        } else {
            128
        };
        preferred.min(capabilities.max_threads_per_block())
    }

    fn reduction_grid(&self, tasks: usize) -> Result<u32> {
        if tasks == 0 {
            return Ok(1);
        }
        let capabilities = self.runtime.capabilities();
        let saturation = u64::from(capabilities.multiprocessor_count()).saturating_mul(16);
        let grid = u64::try_from(tasks)
            .unwrap_or(u64::MAX)
            .min(saturation.max(1))
            .min(u64::from(u32::MAX));
        u32::try_from(grid).map_err(|_| error("reduction grid exceeds CUDA limits"))
    }

    fn pointwise_launch_config(&self, total: u64) -> Result<LaunchConfig> {
        let capabilities = self.runtime.capabilities();
        let preferred = if capabilities.compute_capability().0 >= 7 {
            256
        } else {
            128
        };
        let threads = preferred.min(capabilities.max_threads_per_block()).max(1);
        let blocks_needed = total.div_ceil(u64::from(threads)).max(1);
        let saturation = u64::from(capabilities.multiprocessor_count()).saturating_mul(16);
        let grid_x = blocks_needed
            .min(saturation.max(1))
            .min(u64::from(u32::MAX));
        Ok(LaunchConfig {
            grid_dim: (
                u32::try_from(grid_x).map_err(|_| error("pointwise grid exceeds CUDA limits"))?,
                1,
                1,
            ),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        })
    }
}

const SCRATCH_SLOTS: usize = 11;

#[derive(Clone, Copy, Debug, Default)]
struct ScratchSlot {
    ptr: CUdeviceptr,
    capacity: usize,
}

#[derive(Debug)]
struct ScratchPool {
    slots: [ScratchSlot; SCRATCH_SLOTS],
}

impl Default for ScratchPool {
    fn default() -> Self {
        Self {
            slots: [ScratchSlot::default(); SCRATCH_SLOTS],
        }
    }
}

impl ScratchPool {
    fn ensure(
        &mut self,
        runtime: &CudaRuntime,
        index: usize,
        bytes: usize,
        capturing: bool,
    ) -> Result<CUdeviceptr> {
        let slot = &mut self.slots[index];
        let bytes = bytes.max(1);
        if slot.ptr != 0 && slot.capacity >= bytes {
            return Ok(slot.ptr);
        }
        if capturing {
            return Err(error(format!(
                "QMoE scratch slot {index} needs {bytes} bytes but the warmed capacity is {} bytes",
                slot.capacity
            )));
        }
        let fresh = runtime.alloc_raw(bytes)?;
        if slot.ptr != 0 {
            // SAFETY: the previous pointer came from this runtime and is replaced
            // only after the new allocation succeeds.
            unsafe {
                let _ = runtime.free_raw(slot.ptr);
            }
        }
        slot.ptr = fresh;
        slot.capacity = bytes;
        Ok(fresh)
    }
}

impl Drop for QMoEKernel {
    fn drop(&mut self) {
        let scratch = self
            .scratch
            .get_mut()
            .expect("cuda_ep QMoE scratch pool poisoned");
        for slot in scratch.slots.iter_mut().rev() {
            if slot.ptr != 0 {
                // SAFETY: every non-zero pointer came from this runtime and is
                // freed exactly once when the kernel is dropped.
                let _ = unsafe { self.runtime.free_raw(slot.ptr) };
                slot.ptr = 0;
                slot.capacity = 0;
            }
        }
    }
}

fn tensor_ptr(tensor: &TensorView) -> CUdeviceptr {
    cuptr(tensor.data_ptr::<u8>() as *const c_void)
}

fn optional_input<'a, 'b>(
    inputs: &'a [TensorView<'b>],
    index: usize,
) -> Option<&'a TensorView<'b>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn int_attr(node: &Node, name: &str, default: i64) -> Result<i64> {
    match node.attr(name) {
        Some(value) => value
            .as_int()
            .ok_or_else(|| error(format!("attribute {name} must be an integer"))),
        None => Ok(default),
    }
}

fn bool_attr(node: &Node, name: &str, default: bool) -> Result<bool> {
    match int_attr(node, name, i64::from(default))? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(error(format!(
            "attribute {name} must be 0 or 1, got {value}"
        ))),
    }
}

fn float_attr(node: &Node, name: &str, default: f32) -> Result<f32> {
    match node.attr(name) {
        Some(value) => value
            .as_float()
            .ok_or_else(|| error(format!("attribute {name} must be a float"))),
        None => Ok(default),
    }
}

fn require_dtype(name: &str, got: DataType, expected: DataType) -> Result<()> {
    if got != expected {
        return Err(error(format!("{name} requires {expected:?}, got {got:?}")));
    }
    Ok(())
}

fn require_rank(name: &str, shape: &[usize], rank: usize) -> Result<()> {
    if shape.len() != rank {
        return Err(error(format!(
            "{name} must be {rank}-D, got shape {shape:?}"
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

fn checked_product(factors: &[usize], context: &str) -> Result<usize> {
    let mut product = 1usize;
    let mut has_zero = false;
    for &factor in factors {
        if factor == 0 {
            has_zero = true;
        } else {
            product = product
                .checked_mul(factor)
                .ok_or_else(|| error(format!("{context} exceeds usize limits")))?;
        }
    }
    Ok(if has_zero { 0 } else { product })
}

fn checked_bytes(elements: usize, element_size: usize, context: &str) -> Result<usize> {
    let bytes = elements
        .checked_mul(element_size)
        .ok_or_else(|| error(format!("{context} byte count exceeds usize limits")))?;
    if bytes > isize::MAX as usize {
        return Err(error(format!(
            "{context} byte count {bytes} exceeds isize::MAX"
        )));
    }
    Ok(bytes)
}

fn checked_tensor_layout(name: &str, shape: &[usize], dtype: DataType) -> Result<usize> {
    let elements = checked_product(shape, &format!("{name} element count"))?;
    checked_bytes(elements, dtype.byte_size(), name)?;
    Ok(elements)
}

fn checked_div_ceil(value: usize, divisor: usize, context: &str) -> Result<usize> {
    value
        .checked_add(divisor - 1)
        .map(|adjusted| adjusted / divisor)
        .ok_or_else(|| error(format!("{context} exceeds usize limits")))
}

fn as_i32(name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| error(format!("{name}={value} exceeds CUDA i32 limits")))
}

fn as_u64(name: &str, value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| error(format!("{name}={value} exceeds CUDA u64 limits")))
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep com.microsoft::QMoE: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId};

    fn node(attrs: &[(&str, Attribute)]) -> Node {
        let mut node = Node::new(NodeId(0), "QMoE", Vec::new(), Vec::new());
        node.domain = "com.microsoft".into();
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        node
    }

    #[test]
    fn attributes_match_cpu_activation_contract() {
        for activation in ["relu", "gelu", "silu", "swiglu", "identity"] {
            let attrs = MoeAttributes::from_node(&node(&[(
                "activation_type",
                Attribute::String(activation.as_bytes().to_vec()),
            )]))
            .unwrap();
            assert!(attrs.activation.kernel_id() >= 0);
        }
    }

    #[test]
    fn placement_accepts_byte_dividing_integer_widths_only() {
        for bits in [1, 2, 4, 8] {
            let supported = node(&[
                ("expert_weight_bits", Attribute::Int(bits)),
                ("block_size", Attribute::Int(16)),
            ]);
            assert!(unsupported_reason(&supported).is_none(), "bits={bits}");
        }
        for bits in [0, 3, 5, 16] {
            let unsupported = node(&[
                ("expert_weight_bits", Attribute::Int(bits)),
                ("block_size", Attribute::Int(16)),
            ]);
            assert!(unsupported_reason(&unsupported).is_some(), "bits={bits}");
            let reason = unsupported_reason(&unsupported).expect("unsupported bits reason");
            assert!(reason.contains("1, 2, 4, or 8"), "{reason}");
            assert!(reason.contains("requantize"), "{reason}");
        }
    }

    #[test]
    fn placement_rejects_native_iq_layouts_until_block_quantized_moe_exists() {
        for quant_type in [
            "mxfp4", "iq4_nl", "iq4_xs", "iq3_s", "iq3_xxs", "iq2_s", "iq2_xs", "iq2_xxs", "iq1_s",
            "iq1_m",
        ] {
            let unsupported = node(&[
                ("expert_weight_bits", Attribute::Int(2)),
                ("block_size", Attribute::Int(16)),
                (
                    "quant_type",
                    Attribute::String(quant_type.as_bytes().to_vec()),
                ),
            ]);
            assert!(unsupported_reason(&unsupported).is_some(), "{quant_type}");
        }
    }

    #[test]
    fn checked_product_does_not_hide_overflow_behind_zero() {
        let error = checked_product(&[0, usize::MAX, 2], "test").unwrap_err();
        assert!(error.to_string().contains("exceeds usize limits"));
    }

    #[test]
    fn launch_preferences_are_compute_capability_driven_in_source() {
        assert!(CUDA_SRC.contains("gridDim.x"));
        assert!(!CUDA_SRC.contains("sm_90"));
        assert!(!CUDA_SRC.contains("__CUDA_ARCH__ >= 900"));
    }

    #[test]
    fn linear_sources_specialize_every_quant_layout_dimension() {
        let symmetric = QuantLayout {
            bits: 4,
            block_size: 32,
            has_zero_points: false,
        };
        let affine = QuantLayout {
            has_zero_points: true,
            ..symmetric
        };
        let block_128 = QuantLayout {
            block_size: 128,
            ..affine
        };
        let int8 = QuantLayout {
            bits: 8,
            ..block_128
        };

        let variants = [symmetric, affine, block_128, int8].map(linear_module_source);
        assert_eq!(
            variants
                .map(|variant| variant.0)
                .into_iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            variants.len()
        );
        for (layout, (_, source)) in [symmetric, affine, block_128, int8]
            .into_iter()
            .zip(variants)
        {
            assert!(source.contains(&format!("#define QMOE_BITS {}", layout.bits)));
            assert!(source.contains(&format!("#define QMOE_BLOCK_SIZE {}", layout.block_size)));
            assert!(source.contains(&format!(
                "#define QMOE_HAS_ZERO_POINTS {}",
                usize::from(layout.has_zero_points)
            )));
        }
        assert!(variants[0].1.contains("qmoe_int4_chunk"));
    }
}

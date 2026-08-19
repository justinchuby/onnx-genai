//! GPU **reductions** over arbitrary axes with `keepdims`
//! (`docs/execution/CUDA_COVERAGE.md`, "Normalization & softmax" / reduce rows).
//!
//! `ReduceSum` and `ReduceMean` use `cudnnReduceTensor` for f32 (native f32
//! comp type). f16 and bf16 use the typed NVRTC block reduction, which does
//! fp16/bf16 IO with f32 register accumulation in a single capture-safe kernel.
//! For f16 this deliberately avoids cuDNN's fp16↔fp32 `op_tensor` cast +
//! fp32-ReduceSum round-trip (cuDNN rejects a half `reduceTensorCompType` and
//! forces `CUDNN_DATA_FLOAT`); bf16 cannot be reduced by cuDNN at all (rejected
//! even with an f32 comp type — `CUDNN_STATUS_NOT_SUPPORTED`). That same NVRTC
//! path also runs for f32 when cuDNN is absent.
//! `ReduceMax`/`ReduceMin` always use NVRTC.
//!
//! `cub::DeviceReduce` / `DeviceSegmentedReduce` are the vendor primitives for
//! reductions, and a segmented block reduction is exactly the shape they use.
//! We keep a self-contained NVRTC block-reduction kernel here (rather than
//! linking cub) so the crate stays toolkit-free (no `nvcc`), while matching the
//! cub segmented-reduce structure: **one block per output element**, cooperative
//! shared-memory tree reduction over that element's reduction group. It is
//! memory-bandwidth-bound, the same class as PyTorch's reduce kernels.
//!
//! ## Arbitrary axes via an exact base/delta split
//!
//! A row-major input offset is separable across axes:
//! `offset = Σ_axes coord·stride`. Splitting axes into **kept** and **reduced**,
//! `offset(o, r) = base(o) + delta(r)` where `base` depends only on the kept
//! coordinates (one per output element) and `delta` only on the reduced
//! coordinates. The host precomputes `base[O]` and `delta[R]` (§ [`ReductionPlan`])
//! and uploads them; the kernel walks `delta` for its output element `o`. This is
//! exact for **any** axis set and rank, mirroring the CPU EP's reduce-walk
//! (`crates/onnx-runtime-ep-cpu/src/kernels/reduce_ops.rs`).
//!
//! ## ONNX semantics
//!
//! Axes come from the `axes` **attribute** (opset < 13/18) or the optional second
//! **input** (opset ≥ 13 for `ReduceSum`, ≥ 18 for the rest); the input wins when
//! present. `keepdims` (default 1) retains reduced dims as size-1.
//! `noop_with_empty_axes` (default 0) makes an explicitly-empty axis set an
//! identity (per-element groups) instead of reduce-all. Negative axes wrap.
//! `Max`/`Min` propagate NaN (numpy semantics), matching the CPU EP.
//!
//! ## Limits (actionable errors — RULES.md #1)
//!
//! * unsupported input/output dtype → deferred, naming the dtype.
//! * an axes-**input** dtype other than int32/int64 → rejected, naming it.
//! * an axis out of `[-rank, rank)` → rejected, naming the axis.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUdeviceptr;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::cudnn::{CudnnBufferPair, CudnnReduceCache, CudnnReduceOp, TensorDescriptorSpec};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// NVRTC source: one block per output element, reducing over its group of
/// `reduce_count` elements addressed by `base_off[o] + delta_off[r]`.
/// `op`: 0 = sum, 1 = max, 2 = min. `is_mean` divides a sum by the group size.
/// `Max`/`Min` propagate NaN (numpy / CPU-EP semantics).
const REDUCE_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__device__ __forceinline__ float reduce_load(const T* data, size_t index);
template <>
__device__ __forceinline__ float reduce_load<float>(const float* data, size_t index) {
    return data[index];
}
template <>
__device__ __forceinline__ float reduce_load<__half>(const __half* data, size_t index) {
    return __half2float(data[index]);
}
template <>
__device__ __forceinline__ float reduce_load<__nv_bfloat16>(
    const __nv_bfloat16* data, size_t index) {
    return __bfloat162float(data[index]);
}

template <typename T>
__device__ __forceinline__ void reduce_store(T* data, size_t index, float value);
template <>
__device__ __forceinline__ void reduce_store<float>(float* data, size_t index, float value) {
    data[index] = value;
}
template <>
__device__ __forceinline__ void reduce_store<__half>(
    __half* data, size_t index, float value) {
    data[index] = __float2half_rn(value);
}
template <>
__device__ __forceinline__ void reduce_store<__nv_bfloat16>(
    __nv_bfloat16* data, size_t index, float value) {
    data[index] = __float2bfloat16_rn(value);
}

extern "C" __global__ void validate_reduce_axes_i64(
    const long long* actual,
    const long long* expected,
    const int count,
    unsigned int* capture_error)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        if (actual[i] != expected[i]) atomicOr(capture_error, 128u);
    }
}

// Base sum/mean/max/min block reduction. Accumulation is always in f32 (the
// ONNX ReduceSum/ReduceMean semantics for half inputs: accumulate in f32, cast
// the result back), so the half/bf16 instantiations load/store through the
// f32-widening `reduce_load`/`reduce_store` helpers above.
template <typename T>
__device__ void reduce_base(
    const T*         x,
    T*               y,
    const long long* base_off,     // [out_count]
    const long long* delta_off,    // [reduce_count]
    const int        out_count,
    const int        reduce_count,
    const int        op,           // 0 sum, 1 max, 2 min
    const int        is_mean,
    const unsigned int* capture_error)
{
    if (capture_error && *capture_error) return;
    const int o = blockIdx.x;
    if (o >= out_count) return;

    const float NEG_INF = __int_as_float(0xff800000);
    const float POS_INF = __int_as_float(0x7f800000);
    const float QNAN    = __int_as_float(0x7fc00000);

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const size_t base = (size_t)base_off[o];

    float acc = (op == 1) ? NEG_INF : (op == 2) ? POS_INF : 0.0f;
    for (int r = tid; r < reduce_count; r += nt) {
        const float v = reduce_load(x, base + (size_t)delta_off[r]);
        if (op == 1)      acc = (isnan(acc) || isnan(v)) ? QNAN : fmaxf(acc, v);
        else if (op == 2) acc = (isnan(acc) || isnan(v)) ? QNAN : fminf(acc, v);
        else              acc += v;
    }
    red[tid] = acc;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            const float a = red[tid], b = red[tid + off];
            if (op == 1)      red[tid] = (isnan(a) || isnan(b)) ? QNAN : fmaxf(a, b);
            else if (op == 2) red[tid] = (isnan(a) || isnan(b)) ? QNAN : fminf(a, b);
            else              red[tid] = a + b;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float out = red[0];
        if (is_mean) out /= (float)reduce_count;
        reduce_store(y, (size_t)o, out);
    }
}

#define DEFINE_REDUCE_BASE(T, suffix) \
extern "C" __global__ void reduce_##suffix( \
    const T* x, T* y, const long long* base_off, const long long* delta_off, \
    const int out_count, const int reduce_count, const int op, \
    const int is_mean, const unsigned int* capture_error) { \
    reduce_base<T>(x, y, base_off, delta_off, out_count, reduce_count, op, \
                   is_mean, capture_error); \
}

DEFINE_REDUCE_BASE(float, f32)
DEFINE_REDUCE_BASE(__half, f16)
DEFINE_REDUCE_BASE(__nv_bfloat16, bf16)

template <typename T>
__device__ void reduce_ext(
    const T*         x,
    T*               y,
    const long long* base_off,     // [out_count]
    const long long* delta_off,    // [reduce_count]
    const int        out_count,
    const int        reduce_count,
    const int        pre,          // 0 id, 1 abs, 2 square, 3 exp
    const int        combine,      // 0 add, 1 mul
    const int        post,         // 0 none, 1 sqrt, 2 ln
    const unsigned int* capture_error)
{
    if (capture_error && *capture_error) return;
    const int o = blockIdx.x;
    if (o >= out_count) return;

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const size_t base = (size_t)base_off[o];

    float acc = (combine == 1) ? 1.0f : 0.0f;
    for (int r = tid; r < reduce_count; r += nt) {
        float v = reduce_load(x, base + (size_t)delta_off[r]);
        if (pre == 1)      v = fabsf(v);
        else if (pre == 2) v = v * v;
        else if (pre == 3) v = expf(v);
        if (combine == 1) acc *= v;
        else              acc += v;
    }
    red[tid] = acc;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            if (combine == 1) red[tid] *= red[tid + off];
            else              red[tid] += red[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float out = red[0];
        if (post == 1)      out = sqrtf(out);
        else if (post == 2) out = logf(out);
        reduce_store(y, o, out);
    }
}

template <typename T>
__device__ void reduce_logsumexp(
    const T*         x,
    T*               y,
    const long long* base_off,     // [out_count]
    const long long* delta_off,    // [reduce_count]
    const int        out_count,
    const int        reduce_count,
    const unsigned int* capture_error)
{
    if (capture_error && *capture_error) return;
    const int o = blockIdx.x;
    if (o >= out_count) return;

    const float NEG_INF = __int_as_float(0xff800000);
    const float QNAN    = __int_as_float(0x7fc00000);

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const size_t base = (size_t)base_off[o];

    // Pass 1 — group max with NaN propagation (numpy / CPU-EP semantics).
    // Stabilizes `log(sum(exp(x)))` as `m + log(sum(exp(x - m)))`, matching the
    // CPU EP's max-subtraction (reduce_ops.rs:179-226).
    float m = NEG_INF;
    for (int r = tid; r < reduce_count; r += nt) {
        const float v = reduce_load(x, base + (size_t)delta_off[r]);
        m = (isnan(m) || isnan(v)) ? QNAN : fmaxf(m, v);
    }
    red[tid] = m;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            const float a = red[tid], b = red[tid + off];
            red[tid] = (isnan(a) || isnan(b)) ? QNAN : fmaxf(a, b);
        }
        __syncthreads();
    }
    const float gmax = red[0];
    __syncthreads();

    // Non-finite maxima short-circuit exactly like the CPU EP: an all `-inf`
    // group yields `-inf`, any `+inf` yields `+inf`, any NaN yields NaN. This
    // also avoids the `inf - inf = NaN` that a blind `exp(v - m)` would produce.
    if (!isfinite(gmax)) {
        if (tid == 0) reduce_store(y, o, gmax);
        return;
    }

    // Pass 2 — sum of exp(v - gmax) in the shifted frame.
    float acc = 0.0f;
    for (int r = tid; r < reduce_count; r += nt) {
        const float v = reduce_load(x, base + (size_t)delta_off[r]);
        acc += expf(v - gmax);
    }
    red[tid] = acc;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid == 0) reduce_store(y, o, gmax + logf(red[0]));
}

#define DEFINE_REDUCE_EXT(T, suffix) \
extern "C" __global__ void reduce_ext_##suffix( \
    const T* x, T* y, const long long* base_off, const long long* delta_off, \
    const int out_count, const int reduce_count, const int pre, \
    const int combine, const int post, const unsigned int* capture_error) { \
    reduce_ext<T>(x, y, base_off, delta_off, out_count, reduce_count, pre, \
                  combine, post, capture_error); \
} \
extern "C" __global__ void reduce_logsumexp_##suffix( \
    const T* x, T* y, const long long* base_off, const long long* delta_off, \
    const int out_count, const int reduce_count, \
    const unsigned int* capture_error) { \
    reduce_logsumexp<T>(x, y, base_off, delta_off, out_count, reduce_count, \
                        capture_error); \
}

DEFINE_REDUCE_EXT(float, f32)
DEFINE_REDUCE_EXT(__half, f16)
DEFINE_REDUCE_EXT(__nv_bfloat16, bf16)

extern "C" __global__ void reduce_i64(
    const long long* x,
    long long*       y,
    const long long* base_off,
    const long long* delta_off,
    const int        out_count,
    const int        reduce_count,
    const int        op,
    const unsigned int* capture_error)
{
    if (capture_error && *capture_error) return;
    const int o = blockIdx.x;
    if (o >= out_count) return;

    extern __shared__ long long red_i64[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const size_t base = (size_t)base_off[o];

    long long acc = (op == 1) ? (-9223372036854775807LL - 1LL)
                              : (op == 2) ? 9223372036854775807LL : 0LL;
    for (int r = tid; r < reduce_count; r += nt) {
        const long long v = x[base + (size_t)delta_off[r]];
        if (op == 1) acc = max(acc, v);
        else if (op == 2) acc = min(acc, v);
        else acc += v;
    }
    red_i64[tid] = acc;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            const long long v = red_i64[tid + off];
            if (op == 1) red_i64[tid] = max(red_i64[tid], v);
            else if (op == 2) red_i64[tid] = min(red_i64[tid], v);
            else red_i64[tid] += v;
        }
        __syncthreads();
    }
    if (tid == 0) y[o] = red_i64[0];
}

extern "C" __global__ void reduce_i32(
    const int* x,
    int*       y,
    const long long* base_off,
    const long long* delta_off,
    const int        out_count,
    const int        reduce_count,
    const int        op,
    const unsigned int* capture_error)
{
    if (capture_error && *capture_error) return;
    const int o = blockIdx.x;
    if (o >= out_count) return;

    extern __shared__ int red_i32[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const size_t base = (size_t)base_off[o];

    int acc = (op == 1) ? (-2147483647 - 1) : (op == 2) ? 2147483647 : 0;
    for (int r = tid; r < reduce_count; r += nt) {
        const int v = x[base + (size_t)delta_off[r]];
        if (op == 1) acc = max(acc, v);
        else if (op == 2) acc = min(acc, v);
        else acc += v;
    }
    red_i32[tid] = acc;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            const int v = red_i32[tid + off];
            if (op == 1) red_i32[tid] = max(red_i32[tid], v);
            else if (op == 2) red_i32[tid] = min(red_i32[tid], v);
            else red_i32[tid] += v;
        }
        __syncthreads();
    }
    if (tid == 0) y[o] = red_i32[0];
}
"#;

const REDUCE_MODULE: &str = "reduce_typed_v2";
const REDUCE_ENTRY: &str = "reduce_f32";
const REDUCE_F16_ENTRY: &str = "reduce_f16";
const REDUCE_BF16_ENTRY: &str = "reduce_bf16";
const REDUCE_EXT_F32_ENTRY: &str = "reduce_ext_f32";
const REDUCE_EXT_F16_ENTRY: &str = "reduce_ext_f16";
const REDUCE_EXT_BF16_ENTRY: &str = "reduce_ext_bf16";
const REDUCE_LOGSUMEXP_F32_ENTRY: &str = "reduce_logsumexp_f32";
const REDUCE_LOGSUMEXP_F16_ENTRY: &str = "reduce_logsumexp_f16";
const REDUCE_LOGSUMEXP_BF16_ENTRY: &str = "reduce_logsumexp_bf16";
const REDUCE_I64_ENTRY: &str = "reduce_i64";
const REDUCE_I32_ENTRY: &str = "reduce_i32";
const REDUCE_VALIDATE_AXES_ENTRY: &str = "validate_reduce_axes_i64";
pub const REDUCE_CAPTURE_ERROR_AXES: u32 = 128;

/// Threads per block for the reduction (power of two → exact tree reduce).
const REDUCE_BLOCK: u32 = 256;

/// The reduction to apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    /// Product of the group (`ReduceProd`).
    Prod,
    /// Sum of squares (`ReduceSumSquare`).
    SumSquare,
    /// Sum of absolute values (`ReduceL1`).
    L1,
    /// Euclidean norm `sqrt(sum(x^2))` (`ReduceL2`).
    L2,
    /// `log(sum(x))` (`ReduceLogSum`).
    LogSum,
    /// `log(sum(exp(x)))`, evaluated with max-subtraction stabilization to
    /// match the CPU EP (`reduce_ops.rs:179-226`): `m + log(sum(exp(x - m)))`
    /// where `m` is the group max. Routed through a dedicated two-pass kernel
    /// (`reduce_logsumexp_{f32,f16,bf16}`) rather than the generic
    /// `(pre,combine,post)`
    /// pipeline, since the max must be known before the exp-sum (`ReduceLogSumExp`).
    LogSumExp,
}

impl ReduceOp {
    fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "ReduceSum",
            ReduceOp::Mean => "ReduceMean",
            ReduceOp::Max => "ReduceMax",
            ReduceOp::Min => "ReduceMin",
            ReduceOp::Prod => "ReduceProd",
            ReduceOp::SumSquare => "ReduceSumSquare",
            ReduceOp::L1 => "ReduceL1",
            ReduceOp::L2 => "ReduceL2",
            ReduceOp::LogSum => "ReduceLogSum",
            ReduceOp::LogSumExp => "ReduceLogSumExp",
        }
    }

    /// (`op` tag for the base kernel, `is_mean`). Only defined for the four base
    /// reductions; the extended ops route through [`ReduceOp::ext_tags`].
    fn kernel_tags(self) -> (i32, i32) {
        match self {
            ReduceOp::Sum => (0, 0),
            ReduceOp::Mean => (0, 1),
            ReduceOp::Max => (1, 0),
            ReduceOp::Min => (2, 0),
            _ => (0, 0),
        }
    }

    /// `(pre, combine, post)` tags for the typed extended-reduce kernels:
    /// `pre` transforms each element (0 id, 1 abs, 2 square, 3 exp), `combine`
    /// folds the group (0 add, 1 mul), and `post` maps the accumulator (0 none,
    /// 1 sqrt, 2 ln). Returns `None` for the four base reductions.
    ///
    /// `LogSumExp` returns `Some(..)` only so it routes past the cudnn/identity
    /// paths; its numerics use the dedicated typed two-pass LogSumExp kernel, so
    /// the tag values themselves are inert for that op.
    fn ext_tags(self) -> Option<(i32, i32, i32)> {
        match self {
            ReduceOp::Prod => Some((0, 1, 0)),
            ReduceOp::SumSquare => Some((2, 0, 0)),
            ReduceOp::L1 => Some((1, 0, 0)),
            ReduceOp::L2 => Some((2, 0, 1)),
            ReduceOp::LogSum => Some((0, 0, 2)),
            ReduceOp::LogSumExp => Some((3, 0, 2)),
            ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Max | ReduceOp::Min => None,
        }
    }

    fn cudnn_op(self) -> Option<CudnnReduceOp> {
        match self {
            ReduceOp::Sum => Some(CudnnReduceOp::Add),
            ReduceOp::Mean => Some(CudnnReduceOp::Average),
            ReduceOp::Max
            | ReduceOp::Min
            | ReduceOp::Prod
            | ReduceOp::SumSquare
            | ReduceOp::L1
            | ReduceOp::L2
            | ReduceOp::LogSum
            | ReduceOp::LogSumExp => None,
        }
    }
}

/// A resolved reduction: which axes are reduced, plus the derived
/// `base`/`delta` offset tables and the expected output shape. Computed on the
/// host (GPU-free), so it is directly unit-testable.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ReductionPlan {
    /// Input base offset for each output element (`len == out_count`).
    pub base: Vec<i64>,
    /// Offset delta for each element of a reduction group (`len == reduce_count`).
    pub delta: Vec<i64>,
    /// Expected output shape (keepdims-aware).
    pub out_shape: Vec<usize>,
}

/// Row-major contiguous strides for `shape`.
fn contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut acc = 1i64;
    for d in (0..shape.len()).rev() {
        strides[d] = acc;
        acc *= shape[d] as i64;
    }
    strides
}

fn contiguous_strides_usize(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for d in (0..shape.len()).rev() {
        strides[d] = acc;
        acc *= shape[d];
    }
    strides
}

fn reduced_output_shape(in_shape: &[usize], reduce: &[bool], keepdims: bool) -> Vec<usize> {
    let mut out_shape = Vec::with_capacity(in_shape.len());
    for (dim, &is_reduced) in in_shape.iter().zip(reduce) {
        if is_reduced {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(*dim);
        }
    }
    out_shape
}

/// Build same-rank input/output descriptors; squeezed ONNX output dimensions
/// remain size-one in cuDNN because this preserves the same contiguous storage.
pub(crate) fn cudnn_reduce_specs(
    dtype: DataType,
    in_shape: &[usize],
    reduce: &[bool],
) -> Result<(TensorDescriptorSpec, TensorDescriptorSpec)> {
    let cudnn_out_shape: Vec<usize> = in_shape
        .iter()
        .zip(reduce)
        .map(|(&dim, &is_reduced)| if is_reduced { 1 } else { dim })
        .collect();
    let input = TensorDescriptorSpec::new(dtype, in_shape, &contiguous_strides_usize(in_shape))?;
    let output = TensorDescriptorSpec::new(
        dtype,
        &cudnn_out_shape,
        &contiguous_strides_usize(&cudnn_out_shape),
    )?;
    Ok((input, output))
}

/// Build the [`ReductionPlan`] for `in_shape`, a `reduce[d]` mask, and
/// `keepdims`. The `base`/`delta` split is exact because row-major strides are
/// independent per axis (see the module docs).
pub(crate) fn build_plan(in_shape: &[usize], reduce: &[bool], keepdims: bool) -> ReductionPlan {
    let rank = in_shape.len();
    let strides = contiguous_strides(in_shape);

    let kept_axes: Vec<usize> = (0..rank).filter(|&d| !reduce[d]).collect();
    let red_axes: Vec<usize> = (0..rank).filter(|&d| reduce[d]).collect();

    let kept_dims: Vec<usize> = kept_axes.iter().map(|&d| in_shape[d]).collect();
    let red_dims: Vec<usize> = red_axes.iter().map(|&d| in_shape[d]).collect();

    let base = enumerate_offsets(&kept_dims, &kept_axes, &strides);
    let delta = enumerate_offsets(&red_dims, &red_axes, &strides);

    // Output shape: kept dims in order; reduced dims become size-1 (keepdims) or
    // are squeezed out.
    let out_shape = reduced_output_shape(in_shape, reduce, keepdims);

    ReductionPlan {
        base,
        delta,
        out_shape,
    }
}

/// Enumerate the input offsets for every multi-index over `dims` (row-major),
/// where `axes[k]` is the input axis of `dims[k]` and `strides` are the input
/// strides. Returns `[0]` for an empty dim set (a single all-zero coordinate).
fn enumerate_offsets(dims: &[usize], axes: &[usize], strides: &[i64]) -> Vec<i64> {
    let total: usize = dims.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; dims.len()];
    loop {
        let mut off = 0i64;
        for k in 0..dims.len() {
            off += idx[k] as i64 * strides[axes[k]];
        }
        out.push(off);
        if !next_index(dims, &mut idx) {
            break;
        }
    }
    out
}

/// Increment a row-major multi-index `idx` within `dims`; returns `false` on
/// wrap (end of iteration). An empty `dims` yields a single iteration.
fn next_index(dims: &[usize], idx: &mut [usize]) -> bool {
    for d in (0..dims.len()).rev() {
        idx[d] += 1;
        if idx[d] < dims[d] {
            return true;
        }
        idx[d] = 0;
    }
    false
}

macro_rules! reduce_factory {
    ($factory:ident, $variant:expr) => {
        /// Factory reading `axes` (optional attribute), `keepdims` (default 1)
        /// and `noop_with_empty_axes` (default 0), plus the shared runtime.
        pub struct $factory {
            pub runtime: Arc<CudaRuntime>,
        }
        impl KernelFactory for $factory {
            fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                let axes_attr = node
                    .attr("axes")
                    .and_then(|a| a.as_ints())
                    .map(<[i64]>::to_vec);
                let keepdims = node.attr("keepdims").and_then(|a| a.as_int()).unwrap_or(1) != 0;
                let noop_with_empty_axes = node
                    .attr("noop_with_empty_axes")
                    .and_then(|a| a.as_int())
                    .unwrap_or(0)
                    != 0;
                Ok(Box::new(ReduceKernel {
                    op: $variant,
                    axes_attr,
                    keepdims,
                    noop_with_empty_axes,
                    runtime: self.runtime.clone(),
                    reduce_metadata: Mutex::new(ReductionMetadataCache::new(self.runtime.clone())),
                    cudnn_reduce: Mutex::new(CudnnReduceCache::new(self.runtime.stream().clone())),
                    warmed_axes: Mutex::new(None),
                    last_call_capture_safe: AtomicBool::new(false),
                }))
            }
        }
    };
}

reduce_factory!(ReduceSumFactory, ReduceOp::Sum);
reduce_factory!(ReduceMeanFactory, ReduceOp::Mean);
reduce_factory!(ReduceMaxFactory, ReduceOp::Max);
reduce_factory!(ReduceMinFactory, ReduceOp::Min);
reduce_factory!(ReduceProdFactory, ReduceOp::Prod);
reduce_factory!(ReduceSumSquareFactory, ReduceOp::SumSquare);
reduce_factory!(ReduceL1Factory, ReduceOp::L1);
reduce_factory!(ReduceL2Factory, ReduceOp::L2);
reduce_factory!(ReduceLogSumFactory, ReduceOp::LogSum);
reduce_factory!(ReduceLogSumExpFactory, ReduceOp::LogSumExp);

/// f32 reduction kernel carrying the op, the attribute `axes` (opset < 13/18),
/// `keepdims`, `noop_with_empty_axes`, and the shared runtime.
#[derive(Debug)]
pub struct ReduceKernel {
    op: ReduceOp,
    axes_attr: Option<Vec<i64>>,
    keepdims: bool,
    noop_with_empty_axes: bool,
    runtime: Arc<CudaRuntime>,
    /// Cached i64 base/delta offset tables (and axes) for the NVRTC block-reduce
    /// path. Shared by the Int64 DATA reduce and every float/bf16 reduce that
    /// falls to the NVRTC kernel (all `ReduceSumSquare`/L1/L2/Prod/LogSum…
    /// extended ops, bf16, and f16/f32 when cuDNN is absent). Caching the tables
    /// means a shape-stable decode reduce allocates nothing per call and records
    /// into a captured CUDA graph segment instead of shredding it with a
    /// per-call `alloc`/`htod`/`sync`/`free`.
    reduce_metadata: Mutex<ReductionMetadataCache>,
    /// Cached cuDNN descriptors + device workspace for the float (f32/f16)
    /// cuDNN reduce path, so a shape-stable decode reduce allocates nothing
    /// per call and can be captured into a CUDA graph.
    cudnn_reduce: Mutex<CudnnReduceCache>,
    /// Axes resolved from the optional axes **input** on the last eager call,
    /// reused during CUDA graph capture where a device read of that input is
    /// illegal. `None` until the first 2-input eager call warms it.
    warmed_axes: Mutex<Option<Vec<i64>>>,
    last_call_capture_safe: AtomicBool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReductionMetadataKey {
    input_shape: Vec<usize>,
    reduce: Vec<bool>,
    keepdims: bool,
    axes: Vec<i64>,
}

#[derive(Debug)]
struct ReductionMetadataCache {
    runtime: Arc<CudaRuntime>,
    key: Option<ReductionMetadataKey>,
    base: CUdeviceptr,
    delta: CUdeviceptr,
    axes: CUdeviceptr,
}

impl ReductionMetadataCache {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            key: None,
            base: 0,
            delta: 0,
            axes: 0,
        }
    }

    fn prepare(
        &mut self,
        input_shape: &[usize],
        reduce: &[bool],
        keepdims: bool,
        axes: &[i64],
        plan: &ReductionPlan,
    ) -> Result<(CUdeviceptr, CUdeviceptr, CUdeviceptr)> {
        let key = ReductionMetadataKey {
            input_shape: input_shape.to_vec(),
            reduce: reduce.to_vec(),
            keepdims,
            axes: axes.to_vec(),
        };
        if self.key.as_ref() == Some(&key) {
            return Ok((self.base, self.delta, self.axes));
        }
        if self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(
                "cuda_ep ReduceSum: int64 reduction metadata changed during CUDA graph capture; warm the fixed decode shape before capture".into(),
            ));
        }
        if self.base != 0 || self.delta != 0 || self.axes != 0 {
            self.runtime.synchronize()?;
        }

        let base_bytes = as_i64_bytes(&plan.base);
        let delta_bytes = as_i64_bytes(&plan.delta);
        let axes_bytes = as_i64_bytes(axes);
        let base = self.runtime.alloc_raw(base_bytes.len().max(1))?;
        let delta = match self.runtime.alloc_raw(delta_bytes.len().max(1)) {
            Ok(delta) => delta,
            Err(error) => {
                // SAFETY: `base` is fresh and has not escaped this cache.
                let _ = unsafe { self.runtime.free_raw(base) };
                return Err(error);
            }
        };
        let axes_ptr = match self.runtime.alloc_raw(axes_bytes.len().max(1)) {
            Ok(axes_ptr) => axes_ptr,
            Err(error) => {
                // SAFETY: both pointers are fresh and have not escaped.
                let _ = unsafe { self.runtime.free_raw(base) };
                let _ = unsafe { self.runtime.free_raw(delta) };
                return Err(error);
            }
        };
        let upload = (|| {
            // SAFETY: all fresh allocations cover their corresponding slices.
            unsafe { self.runtime.htod(&base_bytes, base) }?;
            unsafe { self.runtime.htod(&delta_bytes, delta) }?;
            unsafe { self.runtime.htod(&axes_bytes, axes_ptr) }
        })();
        if let Err(error) = upload {
            // SAFETY: none of the fresh pointers escaped or were launched.
            let _ = unsafe { self.runtime.free_raw(base) };
            let _ = unsafe { self.runtime.free_raw(delta) };
            let _ = unsafe { self.runtime.free_raw(axes_ptr) };
            return Err(error);
        }

        if self.base != 0 {
            // SAFETY: synchronization above completed every prior launch using
            // these cache-owned pointers.
            unsafe { self.runtime.free_raw(self.base) }?;
        }
        if self.delta != 0 {
            // SAFETY: same ownership and synchronization invariant as `base`.
            unsafe { self.runtime.free_raw(self.delta) }?;
        }
        if self.axes != 0 {
            // SAFETY: same ownership and synchronization invariant as `base`.
            unsafe { self.runtime.free_raw(self.axes) }?;
        }
        self.key = Some(key);
        self.base = base;
        self.delta = delta;
        self.axes = axes_ptr;
        Ok((base, delta, axes_ptr))
    }
}

impl Drop for ReductionMetadataCache {
    fn drop(&mut self) {
        if self.base != 0 {
            // SAFETY: this cache exclusively owns the live pointer.
            let _ = unsafe { self.runtime.free_raw(self.base) };
            self.base = 0;
        }
        if self.delta != 0 {
            // SAFETY: this cache exclusively owns the live pointer.
            let _ = unsafe { self.runtime.free_raw(self.delta) };
            self.delta = 0;
        }
        if self.axes != 0 {
            // SAFETY: this cache exclusively owns the live pointer.
            let _ = unsafe { self.runtime.free_raw(self.axes) };
            self.axes = 0;
        }
    }
}

/// Resolve the reduced-axis mask from the raw axes list (input or attribute),
/// honouring `noop_with_empty_axes`. Mirrors the CPU EP.
pub(crate) fn resolve_reduce_mask(
    op: &str,
    axes_raw: &Option<Vec<i64>>,
    rank: usize,
    noop_with_empty_axes: bool,
) -> Result<Vec<bool>> {
    let mut reduce = vec![false; rank];
    match axes_raw {
        Some(a) if a.is_empty() => {
            if !noop_with_empty_axes {
                reduce.iter_mut().for_each(|r| *r = true);
            }
        }
        Some(axes) => {
            for &a in axes {
                let ax = if a < 0 { a + rank as i64 } else { a };
                if ax < 0 || ax as usize >= rank {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: axis {a} is out of range for a rank-{rank} input; \
                         axis must lie in [-{rank}, {rank})"
                    )));
                }
                reduce[ax as usize] = true;
            }
        }
        None => {
            if !noop_with_empty_axes {
                reduce.iter_mut().for_each(|r| *r = true);
            }
        }
    }
    Ok(reduce)
}

impl ReduceKernel {
    /// Read the optional axes **input** (opset 13/18+) off the device as `i64`.
    fn read_axes_input(&self, op: &str, axes: &TensorView) -> Result<Vec<i64>> {
        if !axes.is_contiguous() {
            return Err(not_implemented(format!(
                "{op} with a non-contiguous (strided) axes input; materialise it first"
            )));
        }
        let n = axes.numel();
        let src = cuptr(axes.data_ptr::<u8>() as *const c_void);
        match axes.dtype {
            DataType::Int64 => {
                let mut bytes = vec![0u8; n * std::mem::size_of::<i64>()];
                // SAFETY: `src` is a live device allocation of `n` i64 elements
                // (contiguous, validated); `bytes` is sized to match.
                unsafe { self.runtime.dtoh(&mut bytes, src) }?;
                Ok(bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                    .collect())
            }
            DataType::Int32 => {
                let mut bytes = vec![0u8; n * std::mem::size_of::<i32>()];
                // SAFETY: as above, for `n` i32 elements.
                unsafe { self.runtime.dtoh(&mut bytes, src) }?;
                Ok(bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_ne_bytes(c.try_into().unwrap()) as i64)
                    .collect())
            }
            other => Err(not_implemented(format!(
                "{op} with axes input dtype {other:?} (expected int32 or int64)"
            ))),
        }
    }

    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.op.name();
        if !(1..=2).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 1-2 inputs (data[, axes]) and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let cudnn_op = self.op.cudnn_op();
        let supported_dtype = if matches!(self.op, ReduceOp::Sum | ReduceOp::Max | ReduceOp::Min)
            && matches!(x.dtype, DataType::Int32 | DataType::Int64)
        {
            true
        } else if cudnn_op.is_some() {
            matches!(
                x.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            )
        } else if self.op.ext_tags().is_some() {
            matches!(
                x.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            )
        } else {
            x.dtype == DataType::Float32
        };
        if !supported_dtype {
            return Err(not_implemented(format!(
                "{op} with input dtype {:?} (sum/max/min support i32/i64/f32; sum/mean and \
                 extended reductions support f32/f16/bf16)",
                x.dtype
            )));
        }
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        if !x.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented(format!(
                "{op} with a non-contiguous (strided) input/output; materialise it first"
            )));
        }
        let rank = x.shape.len();

        // Resolve axes: input 1 (opset 13/18+) beats the attribute; both absent
        // means reduce-all (unless noop_with_empty_axes selects identity).
        let capturing = self.runtime.is_capturing()?;
        let axes_raw: Option<Vec<i64>> = if inputs.len() == 2 && capturing {
            if inputs[1].dtype != DataType::Int64 {
                return Err(EpError::KernelFailed(
                    "cuda_ep ReduceSum: captured axes input must be Int64".into(),
                ));
            }
            // A device read of the axes input is illegal during capture, so
            // reuse the axes warmed on the pre-capture eager call. The NVRTC
            // reduce path (Int64 DATA and every float/bf16 block reduce) records
            // the resolved axes in its metadata cache; the float cuDNN path warms
            // `warmed_axes` on every eager 2-input call. Prefer the cached
            // metadata axes when present, else the warmed copy.
            let cached_axes = self
                .reduce_metadata
                .lock()
                .map_err(|_| {
                    EpError::KernelFailed(
                        "cuda_ep ReduceSum: metadata cache lock was poisoned".into(),
                    )
                })?
                .key
                .as_ref()
                .map(|key| key.axes.clone());
            let axes = match cached_axes {
                Some(axes) => axes,
                None => self
                    .warmed_axes
                    .lock()
                    .map_err(|_| {
                        EpError::KernelFailed(
                            "cuda_ep ReduceSum: warmed-axes cache lock was poisoned".into(),
                        )
                    })?
                    .clone()
                    .ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep ReduceSum: axes were not warmed before CUDA graph capture"
                                .into(),
                        )
                    })?,
            };
            Some(axes)
        } else if inputs.len() == 2 {
            let axes = self.read_axes_input(op, &inputs[1])?;
            *self.warmed_axes.lock().map_err(|_| {
                EpError::KernelFailed(
                    "cuda_ep ReduceSum: warmed-axes cache lock was poisoned".into(),
                )
            })? = Some(axes.clone());
            Some(axes)
        } else {
            self.axes_attr.clone()
        };
        let reduce = resolve_reduce_mask(op, &axes_raw, rank, self.noop_with_empty_axes)?;
        let expected_shape = reduced_output_shape(x.shape, &reduce, self.keepdims);

        if outputs[0].shape != expected_shape.as_slice() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?} does not match the reduced shape {:?} \
                 (axes {:?}, keepdims {})",
                outputs[0].shape, expected_shape, axes_raw, self.keepdims
            )));
        }

        if x.numel() == 0 || outputs[0].numel() == 0 {
            return Ok(());
        }

        if (!reduce.iter().any(|&axis| axis) || rank == 0) && self.op.ext_tags().is_none() {
            let src = cuptr(x.data_ptr::<u8>() as *const c_void);
            let dst = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
            if src != dst {
                // SAFETY: identity reduction has equal input/output storage size.
                unsafe { self.runtime.dtod(src, dst, x.byte_size()) }?;
            }
            return Ok(());
        }

        // cuDNN reduces f32 with a native f32 comp type — byte-identical to the
        // typed NVRTC block reduction but on cuDNN's tuned kernels — so f32 sum/
        // mean stays here. f16 does **not**: `cudnnReduceTensor` rejects a half
        // `reduceTensorCompType` and forces `CUDNN_DATA_FLOAT`, which cuDNN
        // implements as an fp16→fp32 `op_tensor` cast, an fp32 ReduceSum, and an
        // fp32→fp16 cast — a three-kernel round-trip through a full-size fp32
        // temporary. A captured-graph node trace of the qwen3.5-0.8b-hybrid
        // decode (Deckard, `.squad/decisions/inbox/deckard-fair-hybrid-gap.md`)
        // measured that fragmentation at ~20.7% of decode GPU-kernel time on the
        // SSM/linear-attention path (cudnn ReduceSum 8.1% + fp16↔fp32 op_tensor
        // casts 2.7%, firing 18× per step). The typed NVRTC block reduction
        // below already does fp16 IO with fp32 register accumulation — the exact
        // ONNX accumulate-in-f32-then-cast rule — in a **single** capture-safe
        // kernel, with no fp32 temporary and half the memory traffic, so f16
        // sum/mean is routed there instead. This is general across all shapes/
        // axes/dims (the same path already serves bf16 and every extended f16
        // reduction), not a per-model special case. bf16 cannot be reduced by
        // cuDNN at all (`CUDNN_STATUS_NOT_SUPPORTED`, cuDNN 9.10/9.20) and also
        // falls through below.
        if x.dtype == DataType::Float32
            && let Some(cudnn_op) = cudnn_op
            && self.runtime.cudnn().is_available()
        {
            let (input_spec, output_spec) = cudnn_reduce_specs(x.dtype, x.shape, &reduce)?;
            let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
            let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
            let mut cache = self.cudnn_reduce.lock().map_err(|_| {
                EpError::KernelFailed(
                    "cuda_ep ReduceSum: cuDNN reduce cache lock was poisoned".into(),
                )
            })?;
            self.runtime.cudnn().with_handle(|handle| {
                handle.reduce_cached(
                    &mut cache,
                    &input_spec,
                    &output_spec,
                    cudnn_op,
                    CudnnBufferPair {
                        input: x_ptr,
                        output: y_ptr,
                        input_numel: x.numel(),
                        output_numel: outputs[0].numel(),
                    },
                    capturing,
                )
            })?;
            drop(cache);
            // During capture a host sync is illegal and is exactly what shreds
            // the graph; in eager mode keep the EP-wide sync contract. With the
            // per-call device allocation now cached away, gating the sync makes
            // the float reduce fold into the captured segment.
            if !capturing {
                self.runtime.synchronize()?;
            }
            self.last_call_capture_safe.store(true, Ordering::Relaxed);
            return Ok(());
        }

        let plan = build_plan(x.shape, &reduce, self.keepdims);
        let out_count = plan.base.len();
        let reduce_count = plan.delta.len();
        if out_count == 0 || reduce_count == 0 {
            // Empty input (a zero dim) — nothing to compute.
            return Ok(());
        }

        // NVRTC block-reduction path (Int64 DATA reduce; f16 and bf16 sum/mean,
        // which take the single-kernel fused fp16/bf16-IO f32-accumulation path
        // here rather than cuDNN's fp32 round-trip; every extended op —
        // `ReduceSumSquare`/L1/L2/Prod/LogSum(Exp) — routed by `ext_tags`; and
        // f32 when cuDNN is absent).
        // All of these use the i64 base/delta offset tables, so they share one
        // capture-eligible metadata cache: a shape-stable decode reduce reuses
        // the cached device tables with no per-call `alloc`/`htod`/`free`, gates
        // its trailing `synchronize()` on `!capturing` (in `launch`), and marks
        // the call capture-safe so the segmenter folds it into the replayed
        // graph. A signature change mid-capture is rejected by `prepare` rather
        // than reallocating device memory inside the capture.
        let axes = axes_raw.as_deref().unwrap_or(&[]);
        let mut metadata = self.reduce_metadata.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep ReduceSum: metadata cache lock was poisoned".into())
        })?;
        let (base_buf, delta_buf, expected_axes) =
            metadata.prepare(x.shape, &reduce, self.keepdims, axes, &plan)?;
        // Only the Int64 DATA reduce reads the axes device buffer, so it alone
        // validates the captured axes against the warmed metadata. The float/
        // bf16 block reduce bakes the reduce mask into base/delta and never reads
        // the axes buffer, and any axes change flips the cache key (rejected by
        // `prepare` above during capture), so it needs no device validation.
        if capturing && inputs.len() == 2 && matches!(x.dtype, DataType::Int32 | DataType::Int64) {
            self.validate_captured_axes(&inputs[1], expected_axes)?;
        }
        self.launch(
            x,
            outputs,
            base_buf,
            delta_buf,
            out_count,
            reduce_count,
            capturing,
        )?;
        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn validate_captured_axes(&self, actual: &TensorView, expected: CUdeviceptr) -> Result<()> {
        let count = i32::try_from(actual.numel()).map_err(|_| {
            EpError::KernelFailed("cuda_ep ReduceSum: axes count exceeds i32".into())
        })?;
        let actual = cuptr(actual.data_ptr::<u8>() as *const c_void);
        let capture_error = self.runtime.capture_error_ptr();
        let func =
            self.runtime
                .nvrtc_function(REDUCE_MODULE, REDUCE_SRC, REDUCE_VALIDATE_AXES_ENTRY)?;
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&actual)
            .arg(&expected)
            .arg(&count)
            .arg(&capture_error);
        // SAFETY: both axis buffers contain `count` i64 values, and the error
        // pointer names the runtime-owned four-byte latch.
        unsafe {
            builder.launch(cudarc::driver::LaunchConfig {
                grid_dim: ((count as u32).div_ceil(REDUCE_BLOCK).max(1), 1, 1),
                block_dim: (REDUCE_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch validate_reduce_axes_i64", error))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        x: &TensorView,
        outputs: &mut [TensorMut],
        base_buf: CUdeviceptr,
        delta_buf: CUdeviceptr,
        out_count: usize,
        reduce_count: usize,
        capturing: bool,
    ) -> Result<()> {
        let op = self.op.name();

        let out_i = i32::try_from(out_count).map_err(|_| {
            EpError::KernelFailed(format!("cuda_ep {op}: {out_count} outputs exceed i32"))
        })?;
        let red_i = i32::try_from(reduce_count).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep {op}: reduction group {reduce_count} exceeds i32"
            ))
        })?;
        let grid = u32::try_from(out_count).map_err(|_| {
            EpError::KernelFailed(format!("cuda_ep {op}: {out_count} blocks exceed u32"))
        })?;
        let (op_tag, is_mean) = self.op.kernel_tags();
        let ext_tags = self.op.ext_tags();
        let is_logsumexp = self.op == ReduceOp::LogSumExp;

        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let capture_error = if capturing {
            self.runtime.capture_error_ptr()
        } else {
            0
        };

        let entry = if x.dtype == DataType::Int64 {
            REDUCE_I64_ENTRY
        } else if x.dtype == DataType::Int32 {
            REDUCE_I32_ENTRY
        } else if is_logsumexp {
            match x.dtype {
                DataType::Float32 => REDUCE_LOGSUMEXP_F32_ENTRY,
                DataType::Float16 => REDUCE_LOGSUMEXP_F16_ENTRY,
                DataType::BFloat16 => REDUCE_LOGSUMEXP_BF16_ENTRY,
                _ => unreachable!("validated extended-reduce dtype {:?}", x.dtype),
            }
        } else if ext_tags.is_some() {
            match x.dtype {
                DataType::Float32 => REDUCE_EXT_F32_ENTRY,
                DataType::Float16 => REDUCE_EXT_F16_ENTRY,
                DataType::BFloat16 => REDUCE_EXT_BF16_ENTRY,
                _ => unreachable!("validated extended-reduce dtype {:?}", x.dtype),
            }
        } else {
            match x.dtype {
                DataType::Float16 => REDUCE_F16_ENTRY,
                DataType::BFloat16 => REDUCE_BF16_ENTRY,
                _ => REDUCE_ENTRY,
            }
        };
        let func = self
            .runtime
            .nvrtc_function(REDUCE_MODULE, REDUCE_SRC, entry)?;
        let bytes_per_thread = if matches!(x.dtype, DataType::Int32 | DataType::Int64) {
            x.dtype.byte_size() as u32
        } else {
            std::mem::size_of::<f32>() as u32
        };
        let cfg =
            self.runtime
                .reduction_launch_config(&func, grid, REDUCE_BLOCK, bytes_per_thread)?;
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&x_ptr)
            .arg(&y_ptr)
            .arg(&base_buf)
            .arg(&delta_buf)
            .arg(&out_i)
            .arg(&red_i);
        let (pre, combine, post) = ext_tags.unwrap_or((0, 0, 0));
        if matches!(x.dtype, DataType::Int32 | DataType::Int64) {
            builder.arg(&op_tag).arg(&capture_error);
        } else if is_logsumexp {
            // Dedicated two-pass kernel (max, then exp-sum); no pre/combine/post.
            builder.arg(&capture_error);
        } else if ext_tags.is_some() {
            builder
                .arg(&pre)
                .arg(&combine)
                .arg(&post)
                .arg(&capture_error);
        } else {
            builder.arg(&op_tag).arg(&is_mean).arg(&capture_error);
        }
        // SAFETY: `func` is the compiled reduce entry; the argument list/ABI
        // match its signature; `x_ptr`/`y_ptr` and the base/delta buffers are
        // live device allocations sized as validated above.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        if capturing {
            Ok(())
        } else {
            self.runtime.synchronize()
        }
    }
}

/// Reinterpret an `i64` slice as native-endian bytes for an H2D upload.
fn as_i64_bytes(v: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for &x in v {
        out.extend_from_slice(&x.to_ne_bytes());
    }
    out
}

impl Kernel for ReduceKernel {
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
                "requires a warmed fixed-shape ReduceSum path with stable device-resident axes metadata",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_point_present_in_source() {
        // The base sum/mean/max/min entries are macro-generated per dtype
        // (used for bf16, which cuDNN cannot reduce, and for f16/f32 when cuDNN
        // is absent), so assert the instantiations rather than the expanded
        // symbol names.
        for definition in [
            "DEFINE_REDUCE_BASE(float, f32)",
            "DEFINE_REDUCE_BASE(__half, f16)",
            "DEFINE_REDUCE_BASE(__nv_bfloat16, bf16)",
        ] {
            assert!(
                REDUCE_SRC.contains(definition),
                "missing NVRTC definition {definition}"
            );
        }
    }

    #[test]
    fn strides_are_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn plan_reduce_last_axis_keepdims() {
        // [2,3] reduce axis 1, keepdims → out [2,1]; 2 groups of 3.
        let reduce = [false, true];
        let plan = build_plan(&[2, 3], &reduce, true);
        assert_eq!(plan.out_shape, vec![2, 1]);
        assert_eq!(plan.base, vec![0, 3]); // row starts
        assert_eq!(plan.delta, vec![0, 1, 2]); // within-row offsets
    }

    #[test]
    fn plan_reduce_axis0_no_keepdims() {
        // [2,3] reduce axis 0, keepdims=false → out [3]; 3 groups of 2.
        let reduce = [true, false];
        let plan = build_plan(&[2, 3], &reduce, false);
        assert_eq!(plan.out_shape, vec![3]);
        assert_eq!(plan.base, vec![0, 1, 2]); // column starts
        assert_eq!(plan.delta, vec![0, 3]); // stride down the column
    }

    #[test]
    fn plan_reduce_all_axes() {
        let reduce = [true, true];
        let plan = build_plan(&[2, 3], &reduce, true);
        assert_eq!(plan.out_shape, vec![1, 1]);
        assert_eq!(plan.base, vec![0]);
        assert_eq!(plan.delta, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn resolve_mask_negative_axis_and_empty_noop() {
        let m = resolve_reduce_mask("ReduceSum", &Some(vec![-1]), 3, false).unwrap();
        assert_eq!(m, vec![false, false, true]);
        // Explicitly-empty axes with noop → reduce nothing (identity).
        let m = resolve_reduce_mask("ReduceSum", &Some(vec![]), 3, true).unwrap();
        assert_eq!(m, vec![false, false, false]);
        // Explicitly-empty axes without noop → reduce all.
        let m = resolve_reduce_mask("ReduceSum", &Some(vec![]), 3, false).unwrap();
        assert_eq!(m, vec![true, true, true]);
        // No axes given → reduce all.
        let m = resolve_reduce_mask("ReduceSum", &None, 2, false).unwrap();
        assert_eq!(m, vec![true, true]);
    }

    #[test]
    fn resolve_mask_rejects_out_of_range_axis() {
        let e = resolve_reduce_mask("ReduceMax", &Some(vec![5]), 2, false).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("out of range"), "{msg}");
        assert!(msg.contains("axis 5"), "{msg}");
    }

    #[test]
    fn kernel_tags_map_ops() {
        assert_eq!(ReduceOp::Sum.kernel_tags(), (0, 0));
        assert_eq!(ReduceOp::Mean.kernel_tags(), (0, 1));
        assert_eq!(ReduceOp::Max.kernel_tags(), (1, 0));
        assert_eq!(ReduceOp::Min.kernel_tags(), (2, 0));
    }

    #[test]
    fn ext_tags_map_extended_ops_and_entry_present() {
        for definition in [
            "DEFINE_REDUCE_EXT(float, f32)",
            "DEFINE_REDUCE_EXT(__half, f16)",
            "DEFINE_REDUCE_EXT(__nv_bfloat16, bf16)",
        ] {
            assert!(
                REDUCE_SRC.contains(definition),
                "missing NVRTC definition {definition}"
            );
        }
        // Base reductions do not route through the extended kernel.
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
            assert_eq!(op.ext_tags(), None);
        }
        // (pre, combine, post): pre 0 id/1 abs/2 square/3 exp; combine 0 add/1 mul;
        // post 0 none/1 sqrt/2 ln.
        assert_eq!(ReduceOp::Prod.ext_tags(), Some((0, 1, 0)));
        assert_eq!(ReduceOp::SumSquare.ext_tags(), Some((2, 0, 0)));
        assert_eq!(ReduceOp::L1.ext_tags(), Some((1, 0, 0)));
        assert_eq!(ReduceOp::L2.ext_tags(), Some((2, 0, 1)));
        assert_eq!(ReduceOp::LogSum.ext_tags(), Some((0, 0, 2)));
        // LogSumExp keeps a Some(..) tag only to route past cudnn/identity; its
        // actual math lives in the dedicated typed two-pass LogSumExp kernel.
        assert_eq!(ReduceOp::LogSumExp.ext_tags(), Some((3, 0, 2)));
        for op in [
            ReduceOp::Prod,
            ReduceOp::SumSquare,
            ReduceOp::L1,
            ReduceOp::L2,
            ReduceOp::LogSum,
            ReduceOp::LogSumExp,
        ] {
            assert_eq!(op.cudnn_op(), None);
        }
    }

    #[test]
    fn cudnn_op_mapping_only_ports_sum_and_mean() {
        assert_eq!(ReduceOp::Sum.cudnn_op(), Some(CudnnReduceOp::Add));
        assert_eq!(ReduceOp::Mean.cudnn_op(), Some(CudnnReduceOp::Average));
        assert_eq!(ReduceOp::Max.cudnn_op(), None);
        assert_eq!(ReduceOp::Min.cudnn_op(), None);
    }

    #[test]
    fn cudnn_specs_keep_reduced_axes_as_size_one() {
        let (input, output) =
            cudnn_reduce_specs(DataType::BFloat16, &[2, 3, 4], &[true, false, true]).unwrap();
        assert_eq!(input.dims(), &[1, 2, 3, 4]);
        assert_eq!(input.strides(), &[24, 12, 4, 1]);
        assert_eq!(output.dims(), &[1, 1, 3, 1]);
        assert_eq!(output.strides(), &[3, 3, 1, 1]);
    }
}

#[cfg(test)]
mod claim_probes {
    use std::ffi::c_void;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};

    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, Kernel, TensorMut, TensorView};
    use onnx_runtime_ir::{DataType, DeviceId};

    use super::{CudnnReduceCache, ReduceKernel, ReduceOp, ReductionMetadataCache};
    use crate::runtime::CudaRuntime;

    fn maybe_runtime() -> Option<Arc<CudaRuntime>> {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let rt = std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten();
        std::panic::set_hook(previous);
        rt
    }

    fn kernel(runtime: &Arc<CudaRuntime>, op: ReduceOp) -> ReduceKernel {
        ReduceKernel {
            op,
            axes_attr: Some(vec![1]),
            keepdims: false,
            noop_with_empty_axes: false,
            runtime: runtime.clone(),
            reduce_metadata: Mutex::new(ReductionMetadataCache::new(runtime.clone())),
            cudnn_reduce: Mutex::new(CudnnReduceCache::new(runtime.stream().clone())),
            warmed_axes: Mutex::new(None),
            last_call_capture_safe: AtomicBool::new(false),
        }
    }

    fn run_i32(runtime: &Arc<CudaRuntime>, op: ReduceOp, data: &[i32]) -> Vec<i32> {
        let bytes = std::mem::size_of_val(data);
        let in_dev = runtime.alloc_raw(bytes).unwrap();
        let out_dev = runtime.alloc_raw(std::mem::size_of::<i32>() * 2).unwrap();
        let as_bytes = |v: &[i32]| unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
        };
        unsafe { runtime.htod(as_bytes(data), in_dev).unwrap() };
        let device = DeviceId::cuda(0);
        let in_shape = [2usize, 3];
        let in_strides = [3i64, 1];
        let inputs = [TensorView::new(
            DevicePtr(in_dev as usize as *const c_void),
            DataType::Int32,
            &in_shape,
            &in_strides,
            device,
        )];
        let out_shape = [2usize];
        let out_strides = [1i64];
        let mut outputs = [TensorMut::new(
            DevicePtrMut(out_dev as usize as *mut c_void),
            DataType::Int32,
            &out_shape,
            &out_strides,
            device,
        )];
        kernel(runtime, op).execute(&inputs, &mut outputs).unwrap();
        runtime.synchronize().unwrap();
        let mut out = vec![0i32; 2];
        let out_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                out.as_mut_ptr().cast::<u8>(),
                std::mem::size_of::<i32>() * 2,
            )
        };
        unsafe { runtime.dtoh(out_bytes, out_dev).unwrap() };
        unsafe {
            runtime.free_raw(in_dev).unwrap();
            runtime.free_raw(out_dev).unwrap();
        }
        out
    }

    fn run_i64(runtime: &Arc<CudaRuntime>, op: ReduceOp, data: &[i64]) -> Vec<i64> {
        let bytes = std::mem::size_of_val(data);
        let in_dev = runtime.alloc_raw(bytes).unwrap();
        let out_dev = runtime.alloc_raw(std::mem::size_of::<i64>() * 2).unwrap();
        let as_bytes = |v: &[i64]| unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
        };
        unsafe { runtime.htod(as_bytes(data), in_dev).unwrap() };
        let device = DeviceId::cuda(0);
        let in_shape = [2usize, 3];
        let in_strides = [3i64, 1];
        let inputs = [TensorView::new(
            DevicePtr(in_dev as usize as *const c_void),
            DataType::Int64,
            &in_shape,
            &in_strides,
            device,
        )];
        let out_shape = [2usize];
        let out_strides = [1i64];
        let mut outputs = [TensorMut::new(
            DevicePtrMut(out_dev as usize as *mut c_void),
            DataType::Int64,
            &out_shape,
            &out_strides,
            device,
        )];
        kernel(runtime, op).execute(&inputs, &mut outputs).unwrap();
        runtime.synchronize().unwrap();
        let mut out = vec![0i64; 2];
        let out_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                out.as_mut_ptr().cast::<u8>(),
                std::mem::size_of::<i64>() * 2,
            )
        };
        unsafe { runtime.dtoh(out_bytes, out_dev).unwrap() };
        unsafe {
            runtime.free_raw(in_dev).unwrap();
            runtime.free_raw(out_dev).unwrap();
        }
        out
    }

    #[test]
    fn i32_i64_reduce_sum_max_min_over_last_axis_on_gpu() {
        let Some(runtime) = maybe_runtime() else {
            eprintln!("skipping i32/i64 reduce GPU probe: CUDA runtime unavailable");
            return;
        };
        let data32 = [1i32, 5, 3, 9, 2, 4];
        assert_eq!(
            run_i32(&runtime, ReduceOp::Sum, &data32),
            vec![9, 15],
            "i32 ReduceSum"
        );
        assert_eq!(
            run_i32(&runtime, ReduceOp::Max, &data32),
            vec![5, 9],
            "i32 ReduceMax"
        );
        assert_eq!(
            run_i32(&runtime, ReduceOp::Min, &data32),
            vec![1, 2],
            "i32 ReduceMin"
        );

        let data64 = [-1i64, 50, 3, 90, -2, 4];
        assert_eq!(
            run_i64(&runtime, ReduceOp::Sum, &data64),
            vec![52, 92],
            "i64 ReduceSum"
        );
        assert_eq!(
            run_i64(&runtime, ReduceOp::Max, &data64),
            vec![50, 90],
            "i64 ReduceMax"
        );
        assert_eq!(
            run_i64(&runtime, ReduceOp::Min, &data64),
            vec![-1, -2],
            "i64 ReduceMin"
        );
    }
}

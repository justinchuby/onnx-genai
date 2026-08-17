//! Elementwise **unary** and **binary** ops on the GPU via runtime-compiled
//! (NVRTC) kernels (`docs/architecture/ORT2.md` §15; RULES.md #4 — a fused NVRTC elementwise
//! path is the endorsed "custom kernel" case: no NVIDIA library covers arbitrary
//! ONNX elementwise chains, and keeping them as our own kernels is what later
//! enables fusing an activation into a preceding GEMM epilogue).
//!
//! ## Scope (all limits are actionable errors, never panics)
//!
//! * **dtype:** f32/f16/bf16. Half inputs are widened to f32 for arithmetic and
//!   narrowed once on store, matching the CPU EP's compute-domain convention.
//! * **Unary** (`Relu`, `Sqrt`, `Erf`, `Tanh`, `Sigmoid`, `Gelu`): one input,
//!   one output, identical shape; strided views are rejected with a
//!   "materialise first" error.
//! * **Binary** (`Add`, `Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max`): NumPy-style
//!   right-aligned broadcasting, using zero strides for size-one/missing axes.
//!
//! Each op is one thread-per-element grid-stride kernel; the arithmetic is
//! trivially bandwidth-bound and matches a PyTorch pointwise kernel's shape.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::optimizer::SILU_MUL_FUSION_ATTR;
use crate::runtime::{CudaRuntime, cuptr};

/// Threads per block for the 1-D pointwise grids (a full warp-multiple block).
const BLOCK: u32 = 256;

const POINTWISE_SRC: &str = r#"
#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#define NXRT_HAS_CUDA_HALF_HEADERS 1
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

template <typename T> __device__ float load_float(T value);
template <> __device__ float load_float<float>(float value) { return value; }
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
template <> __device__ float load_float<__half>(__half value) { return __half2float(value); }
template <> __device__ float load_float<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }
#endif

template <typename T> __device__ T store_float(float value);
template <> __device__ float store_float<float>(float value) { return value; }
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
template <> __device__ __half store_float<__half>(float value) { return __float2half_rn(value); }
template <> __device__ __nv_bfloat16 store_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}
#endif

__device__ float op_relu(float x) { return x != x ? x : fmaxf(x, 0.0f); }
__device__ float op_sqrt(float x) { return sqrtf(x); }
__device__ float op_erf(float x) { return erff(x); }
__device__ float op_tanh(float x) { return tanhf(x); }
__device__ float op_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + (float)exp((double)-x));
    float e = (float)exp((double)x);
    return e / (1.0f + e);
}
__device__ float op_silu(float x) {
    if (x >= 0.0f) {
        const float denominator =
            __fadd_rn(1.0f, (float)exp((double)-x));
        return __fdiv_rn(x, denominator);
    }
    const float e = (float)exp((double)x);
    const float numerator = __fmul_rn(x, e);
    return __fdiv_rn(numerator, __fadd_rn(1.0f, e));
}
__device__ float op_gelu(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865475f));
}
__device__ float op_gelu_tanh(float x) {
    const float cube = x * x * x;
    const float inner = 0.7978845608028654f * (x + 0.044715f * cube);
    return x * 0.5f * (1.0f + tanhf(inner));
}

__device__ float op_add(float a, float b) { return a + b; }
__device__ float op_sub(float a, float b) { return a - b; }
__device__ float op_mul(float a, float b) { return a * b; }
__device__ float op_div(float a, float b) { return a / b; }
__device__ float op_pow(float a, float b) { return powf(a, b); }
__device__ float op_min(float a, float b) { return (a != a || b != b) ? a + b : fminf(a, b); }
__device__ float op_max(float a, float b) { return (a != a || b != b) ? a + b : fmaxf(a, b); }

#define DEFINE_UNARY(NAME, TYPE, SUFFIX) \
extern "C" __global__ void NAME##_##SUFFIX(const TYPE* x, TYPE* y, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += (unsigned long long)gridDim.x * blockDim.x) \
        y[i] = store_float<TYPE>(op_##NAME(load_float<TYPE>(x[i]))); \
}

#define DEFINE_BINARY(NAME, TYPE, SUFFIX) \
extern "C" __global__ void NAME##_##SUFFIX( \
    const TYPE* a, const TYPE* b, TYPE* y, const unsigned long long* metadata, \
    const int rank, const unsigned long long n) { \
    const unsigned long long* shape = metadata; \
    const unsigned long long* a_strides = metadata + rank; \
    const unsigned long long* b_strides = metadata + rank * 2; \
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += (unsigned long long)gridDim.x * blockDim.x) { \
        unsigned long long linear = i, ai = 0, bi = 0; \
        for (int d = rank - 1; d >= 0; --d) { \
            unsigned long long coord = linear % shape[d]; \
            linear /= shape[d]; \
            ai += coord * a_strides[d]; \
            bi += coord * b_strides[d]; \
        } \
        y[i] = store_float<TYPE>(op_##NAME(load_float<TYPE>(a[ai]), load_float<TYPE>(b[bi]))); \
    } \
}

#define DEFINE_SILU_MUL(TYPE, SUFFIX) \
extern "C" __global__ void silu_mul_##SUFFIX( \
    const TYPE* a, const TYPE* b, TYPE* y, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += (unsigned long long)gridDim.x * blockDim.x) \
        y[i] = store_float<TYPE>( \
            __fmul_rn(op_silu(load_float<TYPE>(a[i])), load_float<TYPE>(b[i]))); \
}

#define DEFINE_BINARY_I64(NAME, EXPR) \
extern "C" __global__ void NAME##_i64( \
    const long long* a, const long long* b, long long* y, \
    const unsigned long long* metadata, const int rank, const unsigned long long n) { \
    const unsigned long long* shape = metadata; \
    const unsigned long long* a_strides = metadata + rank; \
    const unsigned long long* b_strides = metadata + rank * 2; \
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += (unsigned long long)gridDim.x * blockDim.x) { \
        unsigned long long linear = i, ai = 0, bi = 0; \
        for (int d = rank - 1; d >= 0; --d) { \
            unsigned long long coord = linear % shape[d]; \
            linear /= shape[d]; \
            ai += coord * a_strides[d]; \
            bi += coord * b_strides[d]; \
        } \
        y[i] = (EXPR); \
    } \
}

#define DEFINE_BINARY_I32(NAME, EXPR) \
extern "C" __global__ void NAME##_i32( \
    const int* a, const int* b, int* y, \
    const unsigned long long* metadata, const int rank, const unsigned long long n) { \
    const unsigned long long* shape = metadata; \
    const unsigned long long* a_strides = metadata + rank; \
    const unsigned long long* b_strides = metadata + rank * 2; \
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += (unsigned long long)gridDim.x * blockDim.x) { \
        unsigned long long linear = i, ai = 0, bi = 0; \
        for (int d = rank - 1; d >= 0; --d) { \
            unsigned long long coord = linear % shape[d]; \
            linear /= shape[d]; \
            ai += coord * a_strides[d]; \
            bi += coord * b_strides[d]; \
        } \
        y[i] = (EXPR); \
    } \
}

#define DEFINE_FOR_TYPE(TYPE, SUFFIX) \
DEFINE_UNARY(relu, TYPE, SUFFIX) \
DEFINE_UNARY(sqrt, TYPE, SUFFIX) \
DEFINE_UNARY(erf, TYPE, SUFFIX) \
DEFINE_UNARY(tanh, TYPE, SUFFIX) \
DEFINE_UNARY(sigmoid, TYPE, SUFFIX) \
DEFINE_UNARY(gelu, TYPE, SUFFIX) \
DEFINE_UNARY(gelu_tanh, TYPE, SUFFIX) \
DEFINE_BINARY(add, TYPE, SUFFIX) \
DEFINE_BINARY(sub, TYPE, SUFFIX) \
DEFINE_BINARY(mul, TYPE, SUFFIX) \
DEFINE_BINARY(div, TYPE, SUFFIX) \
DEFINE_BINARY(pow, TYPE, SUFFIX) \
DEFINE_BINARY(min, TYPE, SUFFIX) \
DEFINE_BINARY(max, TYPE, SUFFIX)

DEFINE_FOR_TYPE(float, f32)
DEFINE_UNARY(silu, float, f32)
DEFINE_SILU_MUL(float, f32)
DEFINE_BINARY_I64(add, a[ai] + b[bi])
DEFINE_BINARY_I64(sub, a[ai] - b[bi])
DEFINE_BINARY_I64(mul, a[ai] * b[bi])
DEFINE_BINARY_I64(div, a[ai] / b[bi])
DEFINE_BINARY_I64(min, a[ai] < b[bi] ? a[ai] : b[bi])
DEFINE_BINARY_I64(max, a[ai] > b[bi] ? a[ai] : b[bi])
DEFINE_BINARY_I32(add, a[ai] + b[bi])
DEFINE_BINARY_I32(sub, a[ai] - b[bi])
DEFINE_BINARY_I32(mul, a[ai] * b[bi])
DEFINE_BINARY_I32(div, a[ai] / b[bi])
DEFINE_BINARY_I32(min, a[ai] < b[bi] ? a[ai] : b[bi])
DEFINE_BINARY_I32(max, a[ai] > b[bi] ? a[ai] : b[bi])
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_FOR_TYPE(__half, f16)
DEFINE_FOR_TYPE(__nv_bfloat16, bf16)
DEFINE_UNARY(silu, __half, f16)
DEFINE_UNARY(silu, __nv_bfloat16, bf16)
DEFINE_SILU_MUL(__nv_bfloat16, bf16)

extern "C" __global__ void silu_mul_f16(
    const __half* a, const __half* b, __half* y, const unsigned long long n) {
    const unsigned long long thread =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    const bool half2_aligned =
        ((((unsigned long long)a | (unsigned long long)b | (unsigned long long)y) & 3ull) == 0ull);
    if (half2_aligned) {
        const __half2* a2 = reinterpret_cast<const __half2*>(a);
        const __half2* b2 = reinterpret_cast<const __half2*>(b);
        __half2* y2 = reinterpret_cast<__half2*>(y);
        const unsigned long long pairs = n / 2;
        for (unsigned long long i = thread; i < pairs; i += stride) {
            const float2 av = __half22float2(a2[i]);
            const float2 bv = __half22float2(b2[i]);
            y2[i] = __floats2half2_rn(
                __fmul_rn(op_silu(av.x), bv.x),
                __fmul_rn(op_silu(av.y), bv.y));
        }
        if (thread == 0 && (n & 1ull) != 0ull) {
            const unsigned long long i = n - 1;
            y[i] = __float2half_rn(
                __fmul_rn(op_silu(__half2float(a[i])), __half2float(b[i])));
        }
    } else {
        for (unsigned long long i = thread; i < n; i += stride) {
            y[i] = __float2half_rn(
                __fmul_rn(op_silu(__half2float(a[i])), __half2float(b[i])));
        }
    }
}

__device__ float op_decomposed_silu_f16(float x) {
    const float sigmoid_h = __half2float(__float2half_rn(op_sigmoid(x)));
    return __half2float(__float2half_rn(__fmul_rn(x, sigmoid_h)));
}

extern "C" __global__ void decomposed_silu_f16(
    const __half* x, __half* y, const unsigned long long n) {
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        y[i] = __float2half_rn(op_decomposed_silu_f16(__half2float(x[i])));
    }
}

extern "C" __global__ void decomposed_silu_mul_f16(
    const __half* a, const __half* b, __half* y, const unsigned long long n) {
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const float silu_h = op_decomposed_silu_f16(__half2float(a[i]));
        y[i] = __float2half_rn(__fmul_rn(silu_h, __half2float(b[i])));
    }
}

// BFloat16 decomposed SwiGLU: byte-identical to the standalone two-op
// `Sigmoid`/`Mul(x, sigmoid)` bf16 graph. Sigmoid and the silu product are each
// rounded to bf16 (via `store_float<__nv_bfloat16>` = `__float2bfloat16_rn`,
// the same rounding the standalone `sigmoid_bf16` / `mul_bf16` ops use), so the
// fused epilogue reproduces the unfused graph's per-op bf16 rounding exactly.
// All intermediate math runs in fp32 (bf16 carries ~8 mantissa bits).
__device__ float op_decomposed_silu_bf16(float x) {
    const float sigmoid_b =
        load_float<__nv_bfloat16>(store_float<__nv_bfloat16>(op_sigmoid(x)));
    return load_float<__nv_bfloat16>(
        store_float<__nv_bfloat16>(__fmul_rn(x, sigmoid_b)));
}

extern "C" __global__ void decomposed_silu_mul_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y,
    const unsigned long long n) {
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const float silu_b = op_decomposed_silu_bf16(load_float<__nv_bfloat16>(a[i]));
        y[i] = store_float<__nv_bfloat16>(
            __fmul_rn(silu_b, load_float<__nv_bfloat16>(b[i])));
    }
}

extern "C" __global__ void decomposed_silu_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* y, const unsigned long long n) {
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        y[i] = store_float<__nv_bfloat16>(
            op_decomposed_silu_bf16(load_float<__nv_bfloat16>(x[i])));
    }
}
#endif
"#;

/// NVRTC module names (one module holds all unary / all binary entries so a
/// runtime compiles each source string at most once — see
/// [`CudaRuntime::nvrtc_function`]).
const POINTWISE_MODULE: &str = "elementwise_float_v5";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatDtype {
    F32,
    F16,
    Bf16,
}

impl FloatDtype {
    fn from_onnx(op: &str, name: &str, dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::Float32 => Ok(Self::F32),
            DataType::Float16 => Ok(Self::F16),
            DataType::BFloat16 => Ok(Self::Bf16),
            other => Err(not_implemented(format!(
                "{op} with {name} dtype {other:?} (supported: Float32, Float16, BFloat16)"
            ))),
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

/// A supported elementwise unary op and its NVRTC entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Relu,
    Sqrt,
    Erf,
    Tanh,
    Sigmoid,
    Silu,
    Gelu,
    GeluTanh,
}

impl UnaryOp {
    fn stem(self) -> &'static str {
        match self {
            UnaryOp::Relu => "relu",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Erf => "erf",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Sigmoid => "sigmoid",
            UnaryOp::Silu => "silu",
            UnaryOp::Gelu => "gelu",
            UnaryOp::GeluTanh => "gelu_tanh",
        }
    }

    fn entry(self, dtype: FloatDtype) -> String {
        format!("{}_{}", self.stem(), dtype.suffix())
    }

    /// ONNX op type this maps to (for error messages).
    fn op_name(self) -> &'static str {
        match self {
            UnaryOp::Relu => "Relu",
            UnaryOp::Sqrt => "Sqrt",
            UnaryOp::Erf => "Erf",
            UnaryOp::Tanh => "Tanh",
            UnaryOp::Sigmoid => "Sigmoid",
            UnaryOp::Silu => "Silu",
            UnaryOp::Gelu => "Gelu",
            UnaryOp::GeluTanh => "Gelu",
        }
    }
}

/// A supported elementwise binary op and its NVRTC entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
}

impl BinaryOp {
    fn stem(self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Pow => "pow",
            BinaryOp::Min => "min",
            BinaryOp::Max => "max",
        }
    }

    fn entry(self, dtype: FloatDtype) -> String {
        format!("{}_{}", self.stem(), dtype.suffix())
    }

    fn op_name(self) -> &'static str {
        match self {
            BinaryOp::Add => "Add",
            BinaryOp::Sub => "Sub",
            BinaryOp::Mul => "Mul",
            BinaryOp::Div => "Div",
            BinaryOp::Pow => "Pow",
            BinaryOp::Min => "Min",
            BinaryOp::Max => "Max",
        }
    }
}

/// Grid dimension for `n` elements at [`BLOCK`] threads, capped so a huge tensor
/// still fits the grid limit (the kernels are grid-stride, so a capped grid
/// still covers every element).
fn grid_for(n: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

pub(crate) fn launch_silu_mul_f16_raw(
    runtime: &CudaRuntime,
    gate: CUdeviceptr,
    up: CUdeviceptr,
    output: CUdeviceptr,
    n: usize,
    decomposed: bool,
) -> Result<()> {
    runtime.require_nvrtc_half_headers("SiluMul fp16")?;
    let n_u64 = u64::try_from(n)
        .map_err(|_| EpError::KernelFailed(format!("cuda_ep SiluMul: {n} elements exceed u64")))?;
    let entry = if decomposed {
        "decomposed_silu_mul_f16"
    } else {
        "silu_mul_f16"
    };
    let func = runtime.nvrtc_function(POINTWISE_MODULE, POINTWISE_SRC, entry)?;
    let mut builder = runtime.stream().launch_builder(&func);
    builder.arg(&gate).arg(&up).arg(&output).arg(&n_u64);
    // SAFETY: callers provide three live fp16 allocations covering `n`
    // elements. The pointwise kernel permits `up == output`: every thread loads
    // its input element before overwriting that same independent element.
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|e| driver_err(&format!("launch {entry}"), e))
}

/// Reject a strided (non-contiguous) view with a "materialise first" error.
fn require_contiguous(op: &str, name: &str, contiguous: bool) -> Result<()> {
    if !contiguous {
        return Err(not_implemented(format!(
            "{op} with a non-contiguous (strided) {name}; \
             insert an explicit copy to materialise it before the op"
        )));
    }
    Ok(())
}

/// Factory for [`UnaryKernel`]; carries the op identity and shared runtime.
pub struct UnaryFactory {
    pub op: UnaryOp,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for UnaryFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(UnaryKernel {
            op: self.op,
            runtime: self.runtime.clone(),
            decomposed_silu: self.op == UnaryOp::Silu
                && node
                    .attr(crate::optimizer::DECOMPOSED_SILU_ATTR)
                    .and_then(Attribute::as_int)
                    == Some(1),
            last_capture_safe_signature: Mutex::new(None),
            capture_seq_independent: false,
        }))
    }
}

/// Factory for standard-domain `Gelu` (since opset 20), including its
/// `approximate` attribute.
pub struct StandardGeluFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for StandardGeluFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let op = match node.attr("approximate") {
            None => UnaryOp::Gelu,
            Some(attribute) => match attribute.as_str() {
                Some("none") => UnaryOp::Gelu,
                Some("tanh") => UnaryOp::GeluTanh,
                _ => {
                    return Err(EpError::KernelFailed(
                        "cuda_ep Gelu: approximate must be 'none' or 'tanh'".into(),
                    ));
                }
            },
        };
        Ok(Box::new(UnaryKernel {
            op,
            runtime: self.runtime.clone(),
            decomposed_silu: false,
            last_capture_safe_signature: Mutex::new(None),
            capture_seq_independent: false,
        }))
    }
}

/// NVRTC-backed floating-point unary elementwise kernel.
#[derive(Debug)]
pub struct UnaryKernel {
    op: UnaryOp,
    runtime: Arc<CudaRuntime>,
    decomposed_silu: bool,
    last_capture_safe_signature: Mutex<Option<UnaryCaptureSignature>>,
    capture_seq_independent: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct UnaryCaptureSignature {
    dtype: FloatDtype,
    shape: Vec<usize>,
}

impl UnaryKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let mut last_signature = self.last_capture_safe_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep unary elementwise capture signature lock was poisoned".into(),
            )
        })?;
        let warmed_signature = last_signature.take();
        let op = self.op.op_name();
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let dtype = FloatDtype::from_onnx(op, "input", x.dtype)?;
        if dtype != FloatDtype::F32 {
            self.runtime.require_nvrtc_half_headers(op)?;
        }
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        require_contiguous(op, "input", x.is_contiguous())?;
        require_contiguous(op, "output", outputs[0].is_contiguous())?;

        if outputs[0].numel() != x.numel() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output has {} elements, expected {} (same shape as input)",
                outputs[0].numel(),
                x.numel()
            )));
        }

        let n = x.numel();
        let n_u64 = u64::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {n} elements exceed u64")))?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        if self.decomposed_silu && !matches!(dtype, FloatDtype::F16 | FloatDtype::Bf16) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: decomposed SiLU rounding requires float16 or bfloat16, got {:?}",
                x.dtype
            )));
        }
        let entry = if self.decomposed_silu {
            format!("decomposed_silu_{}", dtype.suffix())
        } else {
            self.op.entry(dtype)
        };
        if self.op == UnaryOp::Silu {
            onnx_runtime_ep_api::record_kernel_variant!(
                "silu_separate",
                "CUDA SwiGLU fusion was not selected because this Silu does not feed one eligible equal-shape Mul exclusively"
            );
        }
        let current_signature =
            capture_shape_eligible(self.capture_seq_independent, x.shape).then(|| {
                UnaryCaptureSignature {
                    dtype,
                    shape: x.shape.to_vec(),
                }
            });
        require_matching_capture_signature(
            &self.runtime,
            op,
            warmed_signature.as_ref(),
            current_signature.as_ref(),
        )?;

        let func = self
            .runtime
            .nvrtc_function(POINTWISE_MODULE, POINTWISE_SRC, &entry)?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let per_element = match self.op {
                UnaryOp::Relu => 1,
                UnaryOp::Sqrt | UnaryOp::Erf | UnaryOp::Tanh => 1,
                UnaryOp::Sigmoid | UnaryOp::Silu => 4,
                UnaryOp::Gelu => 6,
                UnaryOp::GeluTanh => 9,
            };
            (n as u64).saturating_mul(per_element)
        });
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&x_ptr).arg(&y_ptr).arg(&n_u64);
        // SAFETY: the entry's pointer types match the validated dtype and both
        // allocations cover `n` elements.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        *last_signature = current_signature;
        Ok(())
    }
}

impl Kernel for UnaryKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        // Eligibility is tied to the exact dtype and shape warmed by the most
        // recent successful call, not a reusable boolean.
        match self.last_capture_safe_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(format!(
                "{} shape/dtype signature does not match the warmed fixed-decode capture signature",
                self.op.op_name()
            )),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(format!(
                "{} capture signature is unavailable because its state lock was poisoned",
                self.op.op_name()
            )),
        }
    }

    fn set_capture_seq_independent(&mut self, seq_independent: bool) {
        self.capture_seq_independent = seq_independent;
    }
}

/// Factory for [`BinaryKernel`]; carries the op identity and shared runtime.
pub struct BinaryFactory {
    pub op: BinaryOp,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for BinaryFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        if self.op == BinaryOp::Mul
            && node.attr(SILU_MUL_FUSION_ATTR).and_then(Attribute::as_int) == Some(1)
        {
            return Ok(Box::new(SiluMulKernel {
                runtime: self.runtime.clone(),
                decomposed: node
                    .attr(crate::optimizer::DECOMPOSED_SILU_ATTR)
                    .and_then(Attribute::as_int)
                    == Some(1),
                last_capture_safe_signature: Mutex::new(None),
                capture_seq_independent: false,
            }));
        }
        Ok(Box::new(BinaryKernel {
            op: self.op,
            runtime: self.runtime.clone(),
            metadata: Mutex::new(BroadcastMetadataCache::new(self.runtime.clone())),
            last_capture_safe_signature: Mutex::new(None),
            capture_seq_independent: false,
        }))
    }
}

/// NVRTC-backed floating-point binary elementwise kernel with broadcasting.
#[derive(Debug)]
pub struct BinaryKernel {
    op: BinaryOp,
    runtime: Arc<CudaRuntime>,
    metadata: Mutex<BroadcastMetadataCache>,
    last_capture_safe_signature: Mutex<Option<BinaryCaptureSignature>>,
    capture_seq_independent: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BroadcastMetadataKey {
    pub(crate) a_shape: Vec<usize>,
    pub(crate) b_shape: Vec<usize>,
    pub(crate) out_shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BinaryCaptureSignature {
    dtype: DataType,
    shapes: BroadcastMetadataKey,
}

/// A persistent device buffer holding the right-aligned broadcast stride/shape
/// metadata for a binary op. Reused across decode steps whenever the operand
/// shapes are unchanged, so a captured kernel launch performs **no** per-step
/// host allocation, upload, free, or synchronize — the prerequisite for the op
/// to advertise [`CaptureSupport::Supported`].
#[derive(Debug)]
pub(crate) struct BroadcastMetadataCache {
    runtime: Arc<CudaRuntime>,
    key: Option<BroadcastMetadataKey>,
    ptr: CUdeviceptr,
}

impl BroadcastMetadataCache {
    pub(crate) fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            key: None,
            ptr: 0,
        }
    }

    pub(crate) fn prepare(
        &mut self,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> Result<CUdeviceptr> {
        let key = BroadcastMetadataKey {
            a_shape: a_shape.to_vec(),
            b_shape: b_shape.to_vec(),
            out_shape: out_shape.to_vec(),
        };
        if self.key.as_ref() == Some(&key) {
            return Ok(self.ptr);
        }
        if self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(
                "cuda_ep binary elementwise: broadcast metadata shape changed during CUDA graph capture; warm the fixed decode shape before capture".into(),
            ));
        }
        if self.ptr != 0 {
            self.runtime.synchronize()?;
        }

        let metadata = broadcast_metadata(a_shape, b_shape, out_shape);
        let metadata_bytes = u64_bytes(&metadata);
        let ptr = self.runtime.alloc_raw(metadata_bytes.len())?;
        // SAFETY: allocation exactly covers the metadata byte slice.
        if let Err(error) = unsafe { self.runtime.htod(metadata_bytes, ptr) } {
            // SAFETY: `ptr` is still owned by this cache and no launch used it.
            let _ = unsafe { self.runtime.free_raw(ptr) };
            return Err(error);
        }
        if self.ptr != 0 {
            // SAFETY: synchronization completed all prior launches using the old
            // pointer, which remains exclusively owned by this cache.
            if let Err(error) = unsafe { self.runtime.free_raw(self.ptr) } {
                // SAFETY: the replacement has not escaped or been launched.
                let _ = unsafe { self.runtime.free_raw(ptr) };
                return Err(error);
            }
        }
        self.key = Some(key);
        self.ptr = ptr;
        Ok(ptr)
    }
}

impl Drop for BroadcastMetadataCache {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: the live pointer was allocated by this runtime and remains
            // exclusively owned by this cache.
            let _ = unsafe { self.runtime.free_raw(self.ptr) };
            self.ptr = 0;
        }
    }
}

impl BinaryKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let mut last_signature = self.last_capture_safe_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep binary elementwise capture signature lock was poisoned".into(),
            )
        })?;
        let warmed_signature = last_signature.take();
        let op = self.op.op_name();
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let a = &inputs[0];
        let b = &inputs[1];
        let float_dtype = if matches!(a.dtype, DataType::Int32 | DataType::Int64)
            && matches!(
                self.op,
                BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Min
                    | BinaryOp::Max
            ) {
            None
        } else {
            Some(FloatDtype::from_onnx(op, "A", a.dtype)?)
        };
        if float_dtype.is_some_and(|dtype| dtype != FloatDtype::F32) {
            self.runtime.require_nvrtc_half_headers(op)?;
        }
        if b.dtype != a.dtype || outputs[0].dtype != a.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: A/B/output dtypes must match, got {:?}/{:?}/{:?}",
                a.dtype, b.dtype, outputs[0].dtype
            )));
        }
        require_contiguous(op, "A", a.is_contiguous())?;
        require_contiguous(op, "B", b.is_contiguous())?;
        require_contiguous(op, "output", outputs[0].is_contiguous())?;

        let out_shape = onnx_runtime_ir::broadcast_shapes(a.shape, b.shape).map_err(EpError::Ir)?;
        if outputs[0].shape != out_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?} must equal broadcast shape {:?}",
                outputs[0].shape, out_shape
            )));
        }

        let n = outputs[0].numel();
        let n_u64 = u64::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {n} elements exceed u64")))?;
        let rank = i32::try_from(out_shape.len()).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep {op}: rank {} exceeds i32",
                out_shape.len()
            ))
        })?;
        let entry = match float_dtype {
            Some(dtype) => self.op.entry(dtype),
            None => format!(
                "{}_{}",
                self.op.stem(),
                if a.dtype == DataType::Int32 {
                    "i32"
                } else {
                    "i64"
                }
            ),
        };
        if self.op == BinaryOp::Mul {
            onnx_runtime_ep_api::record_kernel_variant!(
                "mul_separate",
                "CUDA SwiGLU fusion was not selected because this Mul is not an eligible equal-shape, single-consumer Mul(Silu(gate), up) pattern"
            );
        }
        let current_signature = capture_shape_eligible(self.capture_seq_independent, &out_shape)
            .then(|| BinaryCaptureSignature {
                dtype: a.dtype,
                shapes: BroadcastMetadataKey {
                    a_shape: a.shape.to_vec(),
                    b_shape: b.shape.to_vec(),
                    out_shape: out_shape.clone(),
                },
            });
        require_matching_capture_signature(
            &self.runtime,
            op,
            warmed_signature.as_ref(),
            current_signature.as_ref(),
        )?;
        let func = self
            .runtime
            .nvrtc_function(POINTWISE_MODULE, POINTWISE_SRC, &entry)?;
        let mut metadata = self.metadata.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep binary elementwise metadata lock was poisoned".into())
        })?;
        let metadata_ptr = metadata.prepare(a.shape, b.shape, &out_shape)?;
        let a_ptr = cuptr(a.data_ptr::<u8>() as *const c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let per_element = match self.op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => 1,
                BinaryOp::Min | BinaryOp::Max => 1,
                BinaryOp::Pow => 2,
            };
            (n as u64).saturating_mul(per_element)
        });
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&y_ptr)
            .arg(&metadata_ptr)
            .arg(&rank)
            .arg(&n_u64);
        // SAFETY: pointer types match the dtype; metadata contains three
        // rank-length u64 arrays; broadcast strides keep all reads in bounds.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        *last_signature = current_signature;
        Ok(())
    }
}

impl Kernel for BinaryKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        // Only the exact fixed-row signature recorded by the most recent
        // successful call may enter capture, including integer metadata ops.
        match self.last_capture_safe_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(format!(
                "{} broadcast shape/dtype signature does not match the warmed capture signature",
                self.op.op_name()
            )),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(format!(
                "{} capture signature is unavailable because its state lock was poisoned",
                self.op.op_name()
            )),
        }
    }

    fn set_capture_seq_independent(&mut self, seq_independent: bool) {
        self.capture_seq_independent = seq_independent;
    }
}

/// Fused equal-shape `silu(gate) * up` pointwise kernel.
#[derive(Debug)]
struct SiluMulKernel {
    runtime: Arc<CudaRuntime>,
    decomposed: bool,
    last_capture_safe_signature: Mutex<Option<UnaryCaptureSignature>>,
    capture_seq_independent: bool,
}

impl SiluMulKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let mut last_signature = self.last_capture_safe_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep fused SiluMul capture signature lock was poisoned".into(),
            )
        })?;
        let warmed_signature = last_signature.take();
        const OP: &str = "SiluMul";
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {OP}: expected 2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let gate = &inputs[0];
        let up = &inputs[1];
        let dtype = FloatDtype::from_onnx(OP, "gate", gate.dtype)?;
        if dtype != FloatDtype::F32 {
            self.runtime.require_nvrtc_half_headers(OP)?;
        }
        if up.dtype != gate.dtype || outputs[0].dtype != gate.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {OP}: gate/up/output dtypes must match, got {:?}/{:?}/{:?}",
                gate.dtype, up.dtype, outputs[0].dtype
            )));
        }
        if gate.shape != up.shape || outputs[0].shape != gate.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {OP}: gate/up/output shapes must match exactly, got {:?}/{:?}/{:?}",
                gate.shape, up.shape, outputs[0].shape
            )));
        }
        require_contiguous(OP, "gate", gate.is_contiguous())?;
        require_contiguous(OP, "up", up.is_contiguous())?;
        require_contiguous(OP, "output", outputs[0].is_contiguous())?;

        let n = gate.numel();
        let n_u64 = u64::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {OP}: {n} elements exceed u64")))?;
        let current_signature = capture_shape_eligible(self.capture_seq_independent, gate.shape)
            .then(|| UnaryCaptureSignature {
                dtype,
                shape: gate.shape.to_vec(),
            });
        require_matching_capture_signature(
            &self.runtime,
            OP,
            warmed_signature.as_ref(),
            current_signature.as_ref(),
        )?;

        if self.decomposed && !matches!(dtype, FloatDtype::F16 | FloatDtype::Bf16) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {OP}: decomposed SiLU fusion requires float16 or bfloat16, got {:?}",
                gate.dtype
            )));
        }
        let entry = if self.decomposed {
            format!("decomposed_silu_mul_{}", dtype.suffix())
        } else {
            format!("silu_mul_{}", dtype.suffix())
        };
        let func = self
            .runtime
            .nvrtc_function(POINTWISE_MODULE, POINTWISE_SRC, &entry)?;
        let gate_ptr = cuptr(gate.data_ptr::<u8>() as *const c_void);
        let up_ptr = cuptr(up.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        crate::trace::record_kernel_metrics(inputs, outputs, || (n as u64).saturating_mul(5));
        onnx_runtime_ep_api::record_kernel_variant!(
            "silu_mul_fused",
            "equal-shape {:?} Mul(Silu(gate), up) uses one capture-safe pointwise launch; fp16 uses aligned half2 with a scalar tail",
            gate.dtype
        );
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&gate_ptr).arg(&up_ptr).arg(&y_ptr).arg(&n_u64);
        // SAFETY: all pointers cover the same validated `n` elements and the
        // selected entry matches their common floating-point dtype.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        *last_signature = current_signature;
        Ok(())
    }
}

impl Kernel for SiluMulKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.last_capture_safe_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "SiluMul shape/dtype signature does not match the warmed fixed-decode capture signature",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "SiluMul capture signature is unavailable because its state lock was poisoned",
            ),
        }
    }

    fn set_capture_seq_independent(&mut self, seq_independent: bool) {
        self.capture_seq_independent = seq_independent;
    }
}

pub(crate) fn require_matching_capture_signature<T: PartialEq>(
    runtime: &CudaRuntime,
    op: &str,
    warmed: Option<&T>,
    current: Option<&T>,
) -> Result<()> {
    if runtime.is_capturing()? && (current.is_none() || warmed != current) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: dtype or shape changed during CUDA graph capture; warm the exact fixed decode signature before capture"
        )));
    }
    Ok(())
}

/// Whether a pointwise op may be admitted to CUDA-graph capture for this call.
///
/// The build-time growing-symbol classifier
/// (`compute_capture_disqualifying_symbols` →
/// `node_capture_seq_independent` → `set_capture_seq_independent`) is the sole,
/// authoritative authority here: `seq_independent == true` means every symbol on
/// this node's output shape is *provably* pinned (fail-safe classifier) / not
/// proven growing (denylist classifier), so the captured launch geometry stays
/// valid across every decode replay. `seq_independent == false` is a definitive
/// "not provably safe" in BOTH classifiers, so capture MUST be vetoed regardless
/// of the runtime shape.
///
/// This deliberately does NOT fall back to a runtime-extent heuristic. A prior
/// version OR-ed in `product == 1` / `is_fixed_decode_shape` (any rank-1 `[N]`
/// row), which could re-admit a node the classifier had correctly disqualified —
/// e.g. a `Reshape([seq_kv*8]) → Sigmoid` whose growing extent `[320]` must be
/// `[328]` next step — baking stale geometry into the graph and silently
/// corrupting decode. Those OR terms can only ever *wrongly override* a real
/// disqualification (the classifier already treats a genuinely pinned
/// single-token decode axis — query `seq_len == 1`, static feature dims — as
/// `seq_independent = true` natively), so they are pure hazard and are dropped.
/// The `shape` argument is retained for a stable call signature and potential
/// logging; eligibility is decided entirely by the classifier's verdict.
pub(crate) fn capture_shape_eligible(seq_independent: bool, _shape: &[usize]) -> bool {
    seq_independent
}

pub(crate) fn broadcast_metadata(a: &[usize], b: &[usize], out: &[usize]) -> Vec<u64> {
    let mut metadata = out.iter().map(|&d| d as u64).collect::<Vec<_>>();
    metadata.extend(broadcast_strides(a, out));
    metadata.extend(broadcast_strides(b, out));
    if metadata.is_empty() {
        metadata.push(0);
    }
    metadata
}

pub(crate) fn broadcast_strides(input: &[usize], out: &[usize]) -> Vec<u64> {
    let contiguous = onnx_runtime_ir::compute_contiguous_strides(input);
    let leading = out.len() - input.len();
    (0..out.len())
        .map(|axis| {
            if axis < leading {
                0
            } else {
                let input_axis = axis - leading;
                if input[input_axis] == 1 {
                    0
                } else {
                    contiguous[input_axis] as u64
                }
            }
        })
        .collect()
}

pub(crate) fn u64_bytes(values: &[u64]) -> &[u8] {
    // SAFETY: u64 is plain data and the byte slice retains the input lifetime.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
    use onnx_runtime_ir::DeviceId;

    #[test]
    fn unary_entry_points_are_distinct_and_named() {
        let ops = [
            UnaryOp::Relu,
            UnaryOp::Sqrt,
            UnaryOp::Erf,
            UnaryOp::Tanh,
            UnaryOp::Sigmoid,
            UnaryOp::Silu,
            UnaryOp::Gelu,
            UnaryOp::GeluTanh,
        ];
        for op in ops {
            // Every advertised entry must be present verbatim in the NVRTC source.
            assert!(
                POINTWISE_SRC.contains(&format!("DEFINE_UNARY({},", op.stem())),
                "missing NVRTC generator for {}",
                op.op_name()
            );
        }
    }

    #[test]
    fn binary_entry_points_are_present_in_source() {
        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Pow,
            BinaryOp::Min,
            BinaryOp::Max,
        ];
        for op in ops {
            assert!(
                POINTWISE_SRC.contains(&format!("DEFINE_BINARY({},", op.stem())),
                "missing NVRTC generator for {}",
                op.op_name()
            );
        }
        assert!(POINTWISE_SRC.contains("DEFINE_SILU_MUL(float, f32)"));
        assert!(POINTWISE_SRC.contains("silu_mul_f16("));
        assert!(POINTWISE_SRC.contains("DEFINE_SILU_MUL(__nv_bfloat16, bf16)"));
        assert!(POINTWISE_SRC.contains("decomposed_silu_mul_f16("));
        assert!(POINTWISE_SRC.contains("decomposed_silu_mul_bf16("));
        assert!(POINTWISE_SRC.contains("decomposed_silu_bf16("));
        for op in ["add", "sub", "mul", "min", "max"] {
            assert!(
                POINTWISE_SRC.contains(&format!("DEFINE_BINARY_I64({op},")),
                "missing int64 NVRTC generator for {op}"
            );
        }
    }

    #[test]
    fn dtype_dispatch_accepts_half_and_rejects_non_float() {
        assert_eq!(
            FloatDtype::from_onnx("Relu", "input", DataType::Float16).unwrap(),
            FloatDtype::F16
        );
        assert_eq!(
            FloatDtype::from_onnx("Relu", "input", DataType::BFloat16).unwrap(),
            FloatDtype::Bf16
        );
        let e = FloatDtype::from_onnx("Relu", "input", DataType::Int64).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("Int64"), "{msg}");
        assert!(msg.contains("Float16"), "{msg}");
    }

    #[test]
    fn require_contiguous_rejects_strided_actionably() {
        let e = require_contiguous("Add", "A", false).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("non-contiguous"), "{msg}");
        assert!(msg.contains("materialise"), "{msg}");
    }

    #[test]
    fn grid_covers_all_elements() {
        assert_eq!(grid_for(0), 1);
        assert_eq!(grid_for(1), 1);
        assert_eq!(grid_for(BLOCK as usize), 1);
        assert_eq!(grid_for(BLOCK as usize + 1), 2);
        // Huge tensors clamp to the grid limit but stay non-zero (grid-stride).
        assert_eq!(grid_for(usize::MAX / 2), 65_535);
    }

    #[test]
    fn broadcast_strides_are_right_aligned_and_zero_for_expanded_axes() {
        assert_eq!(broadcast_strides(&[4, 1, 3], &[4, 5, 3]), [3, 0, 1]);
        assert_eq!(broadcast_strides(&[1, 5, 3], &[4, 5, 3]), [0, 3, 1]);
        assert_eq!(broadcast_strides(&[3], &[4, 5, 3]), [0, 0, 1]);
    }

    #[test]
    fn classifier_disqualified_node_is_never_capture_eligible_regardless_of_shape() {
        // `seq_independent == false` is the growing-symbol classifier's hard veto:
        // no runtime shape may re-admit the node to CUDA-graph capture. On HEAD
        // 0fd87df3 `capture_shape_eligible(false, &[320])` returned `true` (any
        // rank-1 `[N]` satisfied `is_fixed_decode_shape`), silently baking a
        // growing KV extent into the captured graph -> decode corruption. The
        // authoritative veto must return `false` here.
        assert!(!capture_shape_eligible(false, &[320]));
        assert!(!capture_shape_eligible(false, &[328]));
        // Scalar / product==1 shapes were also OR-admitted before; still vetoed.
        assert!(!capture_shape_eligible(false, &[1]));
        assert!(!capture_shape_eligible(false, &[1, 1, 1]));
        assert!(!capture_shape_eligible(false, &[]));
        // Higher-rank growing rows were always eager and remain so.
        assert!(!capture_shape_eligible(false, &[320, 8]));
    }

    #[test]
    fn classifier_pinned_node_stays_capture_eligible() {
        // A truly pinned single-token decode shape the classifier proves safe
        // (`seq_independent == true`, e.g. `[1,1,heads,dim]`) stays eligible.
        assert!(capture_shape_eligible(true, &[1, 1, 32, 128]));
        assert!(capture_shape_eligible(true, &[1]));
        assert!(capture_shape_eligible(true, &[]));
        // The verdict is shape-independent: even a large static feature row the
        // classifier proved pinned stays capturable.
        assert!(capture_shape_eligible(true, &[320]));
    }

    #[test]
    fn disqualified_growing_reshape_consumer_yields_no_capture_signature() {
        // Capture-LEVEL check mirroring `UnaryKernel::run`'s gating
        // (`capture_shape_eligible(self.capture_seq_independent, shape).then(..)`)
        // for a `Reshape([seq_kv,8],[-1]) -> seq_kv*8 -> Sigmoid` consumer that
        // the classifier disqualified (`capture_seq_independent = false`) with a
        // growing runtime extent `[320]`. A disqualified node must never produce a
        // capture-safe signature, so `capture_support()` reports Unsupported. On
        // HEAD 0fd87df3 this signature was `Some` (the silent-corruption hole).
        let seq_independent = false;
        let growing_shape = [320usize];
        let signature = capture_shape_eligible(seq_independent, &growing_shape)
            .then_some(growing_shape.to_vec());
        assert!(
            signature.is_none(),
            "a classifier-disqualified node must never produce a capture-safe signature"
        );
    }

    #[test]
    fn pinned_single_token_consumer_yields_capture_signature() {
        // Positive companion: a classifier-proven-pinned single-token decode
        // consumer still produces a capture-safe signature, so capture coverage
        // for genuinely fixed decode shapes is preserved.
        let seq_independent = true;
        let pinned_shape = [1usize, 1, 32, 128];
        let signature =
            capture_shape_eligible(seq_independent, &pinned_shape).then_some(pinned_shape.to_vec());
        assert!(
            signature.is_some(),
            "a classifier-proven-pinned node must remain capture-eligible"
        );
    }

    #[test]
    fn silu_mul_f16_matches_reference_with_half2_tail() {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let runtime = std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten();
        std::panic::set_hook(previous_hook);
        let Some(runtime) = runtime else {
            eprintln!("skipping fused SiluMul fp16 parity test: CUDA runtime unavailable");
            return;
        };
        if runtime.require_nvrtc_half_headers("SiluMul").is_err() {
            eprintln!("skipping fused SiluMul fp16 parity test: fp16 headers unavailable");
            return;
        }

        let gate = [-8.0f32, -2.0, -0.25, -0.0, 0.0, 0.125, 1.0, 3.0, 9.0].map(f16::from_f32);
        let up = [-1.5f32, 0.5, 2.0, -3.0, 4.0, -0.75, 1.25, 0.25, -2.0].map(f16::from_f32);
        let mut output = [f16::ZERO; 9];
        let bytes = std::mem::size_of_val(&gate);
        let gate_dev = runtime.alloc_raw(bytes).unwrap();
        let up_dev = runtime.alloc_raw(bytes).unwrap();
        let output_dev = runtime.alloc_raw(bytes).unwrap();
        let as_bytes = |values: &[f16]| unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        unsafe {
            runtime.htod(as_bytes(&gate), gate_dev).unwrap();
            runtime.htod(as_bytes(&up), up_dev).unwrap();
        }

        let shape = [1usize, gate.len()];
        let strides = [gate.len() as i64, 1];
        let device = DeviceId::cuda(0);
        let inputs = [
            TensorView::new(
                DevicePtr(gate_dev as usize as *const c_void),
                DataType::Float16,
                &shape,
                &strides,
                device,
            ),
            TensorView::new(
                DevicePtr(up_dev as usize as *const c_void),
                DataType::Float16,
                &shape,
                &strides,
                device,
            ),
        ];
        let mut outputs = [TensorMut::new(
            DevicePtrMut(output_dev as usize as *mut c_void),
            DataType::Float16,
            &shape,
            &strides,
            device,
        )];
        SiluMulKernel {
            runtime: runtime.clone(),
            decomposed: false,
            last_capture_safe_signature: Mutex::new(None),
            capture_seq_independent: false,
        }
        .execute(&inputs, &mut outputs)
        .unwrap();
        runtime.synchronize().unwrap();
        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                output.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(&output),
            )
        };
        unsafe { runtime.dtoh(output_bytes, output_dev).unwrap() };

        for (index, ((&a, &b), &actual)) in gate.iter().zip(&up).zip(&output).enumerate() {
            let x = a.to_f32();
            let silu = if x >= 0.0 {
                x / (1.0 + (-f64::from(x)).exp() as f32)
            } else {
                let e = f64::from(x).exp() as f32;
                (x * e) / (1.0 + e)
            };
            let expected = f16::from_f32(silu * b.to_f32()).to_f32();
            let error = (actual.to_f32() - expected).abs();
            assert!(
                error <= 2.0e-3,
                "index {index}: silu({x}) * {} expected {expected}, got {} (error {error})",
                b.to_f32(),
                actual.to_f32()
            );
        }

        unsafe {
            runtime.free_raw(gate_dev).unwrap();
            runtime.free_raw(up_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }
    }

    #[test]
    fn decomposed_silu_mul_bf16_is_byte_exact_vs_two_op_bf16_reference() {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let runtime = std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten();
        std::panic::set_hook(previous_hook);
        let Some(runtime) = runtime else {
            eprintln!("skipping decomposed SiluMul bf16 parity test: CUDA runtime unavailable");
            return;
        };
        if runtime.require_nvrtc_half_headers("SiluMul").is_err() {
            eprintln!("skipping decomposed SiluMul bf16 parity test: half headers unavailable");
            return;
        }

        // Include the fp16-rounding-sensitive small magnitudes and both signs so
        // the bf16 per-op rounding boundaries (Sigmoid → bf16, gate*sigmoid →
        // bf16, silu*up → bf16) are all exercised.
        let gate = [-9.0f32, -2.0, -0.25, -0.03125, 0.0, 0.125, 1.0, 3.0, 9.0].map(bf16::from_f32);
        let up = [-1.5f32, 0.5, 2.0, -3.0, 4.0, -0.75, 1.25, 0.25, -2.0].map(bf16::from_f32);
        let mut output = [bf16::ZERO; 9];
        let bytes = std::mem::size_of_val(&gate);
        let gate_dev = runtime.alloc_raw(bytes).unwrap();
        let up_dev = runtime.alloc_raw(bytes).unwrap();
        let output_dev = runtime.alloc_raw(bytes).unwrap();
        let as_bytes = |values: &[bf16]| unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        unsafe {
            runtime.htod(as_bytes(&gate), gate_dev).unwrap();
            runtime.htod(as_bytes(&up), up_dev).unwrap();
        }

        let shape = [1usize, gate.len()];
        let strides = [gate.len() as i64, 1];
        let device = DeviceId::cuda(0);
        let inputs = [
            TensorView::new(
                DevicePtr(gate_dev as usize as *const c_void),
                DataType::BFloat16,
                &shape,
                &strides,
                device,
            ),
            TensorView::new(
                DevicePtr(up_dev as usize as *const c_void),
                DataType::BFloat16,
                &shape,
                &strides,
                device,
            ),
        ];
        let mut outputs = [TensorMut::new(
            DevicePtrMut(output_dev as usize as *mut c_void),
            DataType::BFloat16,
            &shape,
            &strides,
            device,
        )];
        SiluMulKernel {
            runtime: runtime.clone(),
            decomposed: true,
            last_capture_safe_signature: Mutex::new(None),
            capture_seq_independent: false,
        }
        .execute(&inputs, &mut outputs)
        .unwrap();
        runtime.synchronize().unwrap();
        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                output.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(&output),
            )
        };
        unsafe { runtime.dtoh(output_bytes, output_dev).unwrap() };

        // Reference reproduces the unfused two-op bf16 graph exactly:
        //   s   = bf16(sigmoid_f32(gate))      (the Sigmoid node's bf16 output)
        //   sil = bf16(gate * s)               (the first Mul's bf16 output)
        //   y   = bf16(sil * up)               (the second Mul's bf16 output)
        // All intermediate arithmetic is fp32 (sigmoid via f64 exp → f32, exactly
        // like the device `op_sigmoid`), rounded to bf16 at each op boundary.
        for (index, ((&a, &b), &actual)) in gate.iter().zip(&up).zip(&output).enumerate() {
            let x = a.to_f32();
            let sigmoid = if x >= 0.0 {
                1.0f32 / (1.0 + (-f64::from(x)).exp() as f32)
            } else {
                let e = f64::from(x).exp() as f32;
                e / (1.0 + e)
            };
            let sigmoid_b = bf16::from_f32(sigmoid).to_f32();
            let silu_b = bf16::from_f32(x * sigmoid_b).to_f32();
            let expected = bf16::from_f32(silu_b * b.to_f32());
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "index {index}: decomposed silu bf16 mismatch for gate={x} up={} \
                 expected {} got {}",
                b.to_f32(),
                expected.to_f32(),
                actual.to_f32()
            );
        }

        unsafe {
            runtime.free_raw(gate_dev).unwrap();
            runtime.free_raw(up_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }
    }

    #[test]
    fn marked_standalone_silu_matches_decomposed_fused_path_bit_exactly() {
        let Ok(runtime) = CudaRuntime::new(0).map(Arc::new) else {
            eprintln!("skipping decomposed Silu parity test: CUDA runtime unavailable");
            return;
        };
        if runtime.require_nvrtc_half_headers("Silu").is_err() {
            eprintln!("skipping decomposed Silu parity test: fp16 headers unavailable");
            return;
        }

        let input = [-9.0f32, -2.0, -0.25, -0.0, 0.0, 0.125, 1.0, 3.0, 9.0].map(f16::from_f32);
        let ones = [f16::ONE; 9];
        let bytes = std::mem::size_of_val(&input);
        let input_device = runtime.alloc_raw(bytes).unwrap();
        let ones_device = runtime.alloc_raw(bytes).unwrap();
        let standalone_device = runtime.alloc_raw(bytes).unwrap();
        let fused_device = runtime.alloc_raw(bytes).unwrap();
        let as_bytes = |values: &[f16]| unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        unsafe {
            runtime.htod(as_bytes(&input), input_device).unwrap();
            runtime.htod(as_bytes(&ones), ones_device).unwrap();
        }

        let shape = [1usize, input.len()];
        let strides = [input.len() as i64, 1];
        let device = DeviceId::cuda(0);
        let standalone_input = [TensorView::new(
            DevicePtr(input_device as usize as *const c_void),
            DataType::Float16,
            &shape,
            &strides,
            device,
        )];
        let mut standalone_output = [TensorMut::new(
            DevicePtrMut(standalone_device as usize as *mut c_void),
            DataType::Float16,
            &shape,
            &strides,
            device,
        )];
        UnaryKernel {
            op: UnaryOp::Silu,
            runtime: runtime.clone(),
            decomposed_silu: true,
            last_capture_safe_signature: Mutex::new(None),
            capture_seq_independent: false,
        }
        .execute(&standalone_input, &mut standalone_output)
        .unwrap();
        launch_silu_mul_f16_raw(
            &runtime,
            input_device,
            ones_device,
            fused_device,
            input.len(),
            true,
        )
        .unwrap();
        runtime.synchronize().unwrap();

        let mut standalone = [f16::ZERO; 9];
        let mut fused = [f16::ZERO; 9];
        unsafe {
            runtime
                .dtoh(
                    std::slice::from_raw_parts_mut(
                        standalone.as_mut_ptr().cast::<u8>(),
                        std::mem::size_of_val(&standalone),
                    ),
                    standalone_device,
                )
                .unwrap();
            runtime
                .dtoh(
                    std::slice::from_raw_parts_mut(
                        fused.as_mut_ptr().cast::<u8>(),
                        std::mem::size_of_val(&fused),
                    ),
                    fused_device,
                )
                .unwrap();
            runtime.free_raw(input_device).unwrap();
            runtime.free_raw(ones_device).unwrap();
            runtime.free_raw(standalone_device).unwrap();
            runtime.free_raw(fused_device).unwrap();
        }
        assert_eq!(standalone, fused);
    }
}

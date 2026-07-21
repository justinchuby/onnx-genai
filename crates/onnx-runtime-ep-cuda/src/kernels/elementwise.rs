//! Elementwise **unary** and **binary** ops on the GPU via runtime-compiled
//! (NVRTC) kernels (`docs/ORT2.md` §15; RULES.md #4 — a fused NVRTC elementwise
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

#define DEFINE_FOR_TYPE(TYPE, SUFFIX) \
DEFINE_UNARY(relu, TYPE, SUFFIX) \
DEFINE_UNARY(sqrt, TYPE, SUFFIX) \
DEFINE_UNARY(erf, TYPE, SUFFIX) \
DEFINE_UNARY(tanh, TYPE, SUFFIX) \
DEFINE_UNARY(sigmoid, TYPE, SUFFIX) \
DEFINE_UNARY(gelu, TYPE, SUFFIX) \
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
#endif
"#;

/// NVRTC module names (one module holds all unary / all binary entries so a
/// runtime compiles each source string at most once — see
/// [`CudaRuntime::nvrtc_function`]).
const POINTWISE_MODULE: &str = "elementwise_float_v3";

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
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(UnaryKernel {
            op: self.op,
            runtime: self.runtime.clone(),
            last_capture_safe_signature: Mutex::new(None),
        }))
    }
}

/// NVRTC-backed floating-point unary elementwise kernel.
#[derive(Debug)]
pub struct UnaryKernel {
    op: UnaryOp,
    runtime: Arc<CudaRuntime>,
    last_capture_safe_signature: Mutex<Option<UnaryCaptureSignature>>,
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
        let entry = self.op.entry(dtype);
        if self.op == UnaryOp::Silu {
            onnx_runtime_ep_api::record_kernel_variant!(
                "silu_separate",
                "CUDA SwiGLU fusion was not selected because this Silu does not feed one eligible equal-shape Mul exclusively"
            );
        }
        let current_signature = is_fixed_decode_shape(x.shape).then(|| UnaryCaptureSignature {
            dtype,
            shape: x.shape.to_vec(),
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
                last_capture_safe_signature: Mutex::new(None),
            }));
        }
        Ok(Box::new(BinaryKernel {
            op: self.op,
            runtime: self.runtime.clone(),
            metadata: Mutex::new(BroadcastMetadataCache::new(self.runtime.clone())),
            last_capture_safe_signature: Mutex::new(None),
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BroadcastMetadataKey {
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    out_shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BinaryCaptureSignature {
    dtype: DataType,
    shapes: BroadcastMetadataKey,
}

#[derive(Debug)]
struct BroadcastMetadataCache {
    runtime: Arc<CudaRuntime>,
    key: Option<BroadcastMetadataKey>,
    ptr: CUdeviceptr,
}

impl BroadcastMetadataCache {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            key: None,
            ptr: 0,
        }
    }

    fn prepare(
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
        let float_dtype = if a.dtype == DataType::Int64
            && matches!(self.op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul)
        {
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
            None => format!("{}_i64", self.op.stem()),
        };
        if self.op == BinaryOp::Mul {
            onnx_runtime_ep_api::record_kernel_variant!(
                "mul_separate",
                "CUDA SwiGLU fusion was not selected because this Mul is not an eligible equal-shape, single-consumer Mul(Silu(gate), up) pattern"
            );
        }
        let current_signature = is_fixed_decode_shape(&out_shape).then(|| BinaryCaptureSignature {
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
}

/// Fused equal-shape `silu(gate) * up` pointwise kernel.
#[derive(Debug)]
struct SiluMulKernel {
    runtime: Arc<CudaRuntime>,
    last_capture_safe_signature: Mutex<Option<UnaryCaptureSignature>>,
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
        let current_signature = is_fixed_decode_shape(gate.shape).then(|| UnaryCaptureSignature {
            dtype,
            shape: gate.shape.to_vec(),
        });
        require_matching_capture_signature(
            &self.runtime,
            OP,
            warmed_signature.as_ref(),
            current_signature.as_ref(),
        )?;

        let entry = format!("silu_mul_{}", dtype.suffix());
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
}

fn require_matching_capture_signature<T: PartialEq>(
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

fn is_fixed_decode_shape(shape: &[usize]) -> bool {
    !shape.is_empty() && shape[..shape.len() - 1].iter().product::<usize>() == 1
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
    use half::f16;
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
            last_capture_safe_signature: Mutex::new(None),
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
}

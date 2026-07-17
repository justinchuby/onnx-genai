//! Additive pointwise ops — **unary math**, **logical**, and **comparison** —
//! on the GPU via runtime-compiled (NVRTC) `extern "C"` kernels. This is CUDA
//! Wave 3 (`docs/CUDA_COVERAGE.md`), extending the [`super::elementwise`] slice
//! with the remaining CPU-EP pointwise coverage (RULES.md #4 — pointwise chains
//! have no NVIDIA library op and stay ours so they can later fuse into a GEMM
//! epilogue or a producer→activation→add chain).
//!
//! ## Scope of this slice (all limits are actionable errors, never panics)
//!
//! * **Unary math** (`Abs`, `Neg`, `Reciprocal`, `Exp`, `Log`, `Sign`, `Floor`,
//!   `Ceil`, `Round`, `Sin`, `Cos`, `Softplus`): one input, one output, identical
//!   shape, **f32/f16/bf16** (half storage computes in f32). Each formula is matched **exactly** to the CPU EP
//!   (`crates/onnx-runtime-ep-cpu/src/kernels/unary_math.rs`) so the untestable-
//!   on-this-host kernels stay numerically identical to the reference path.
//! * **Not** (`Not`): boolean element negation, matched to the CPU EP
//!   (`logical.rs` — non-zero byte is `true`, output is canonical `1`/`0`).
//! * **Comparison** (`Equal`, `Greater`, `Less`, `GreaterOrEqual`,
//!   `LessOrEqual`): two broadcast-compatible **f32** inputs → **Bool** output.
//! * **Logical** (`And`, `Or`, `Xor`): two broadcast-compatible **Bool** inputs →
//!   **Bool** output (non-zero byte is `true`, canonical `1`/`0` out).
//!
//! `dtype`: f32/f16/bf16 for unary math, f32 for comparison, and bool for
//! logical/`Not`; other dtypes return an actionable error naming the dtype/op.
//!
//! **Broadcasting:** binary comparison/logical ops reuse the same right-aligned,
//! zero-stride metadata as [`super::elementwise`].
//!
//! Each op is one thread-per-element grid-stride kernel (bandwidth-bound,
//! PyTorch-pointwise shaped).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::elementwise::{broadcast_metadata, u64_bytes};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// Threads per block for the 1-D pointwise grids (a full warp-multiple block).
const BLOCK: u32 = 256;

/// Grid dimension for `n` elements at [`BLOCK`] threads, capped so a huge tensor
/// still fits the grid limit (the kernels are grid-stride, so a capped grid
/// still covers every element).
fn grid_for(n: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

/// Reject a dtype other than the expected one with an actionable, op-named error.
fn require_dtype(op: &str, name: &str, dt: DataType, want: DataType) -> Result<()> {
    if dt != want {
        return Err(not_implemented(format!(
            "{op} with {name} dtype {dt:?} (this slice supports {want:?} only; \
             f16/bf16 pending — see docs/CUDA_COVERAGE.md)"
        )));
    }

    Ok(())
}

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

/// `n` as `u64`, matching the kernels' `unsigned long long` count parameter.
fn count_u64(op: &str, n: usize) -> Result<u64> {
    u64::try_from(n)
        .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {n} elements exceed u64")))
}

// ===========================================================================
// Unary math (f32 → f32)
// ===========================================================================

/// NVRTC source: dtype-templated pointwise kernels for each unary math op. NVRTC
/// resolves the CUDA device intrinsics (`expf`, `logf`, `sinf`, `rintf`,
/// `log1pf`, …) with no header include. Each formula is annotated with the exact
/// CPU-EP expression it mirrors (`unary_math.rs`).
const UNARY_MATH_SRC: &str = r#"
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
template <> __device__ __nv_bfloat16 store_float<__nv_bfloat16>(float value) { return __float2bfloat16_rn(value); }
#endif

__device__ float op_abs(float x) { return fabsf(x); }
__device__ float op_neg(float x) { return -x; }
__device__ float op_reciprocal(float x) { return 1.0f / x; }
__device__ float op_exp(float x) { return expf(x); }
__device__ float op_log(float x) { return logf(x); }
__device__ float op_sign(float x) {
    return (x != x) ? x : ((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f));
}
__device__ float op_floor(float x) { return floorf(x); }
__device__ float op_ceil(float x) { return ceilf(x); }
__device__ float op_round(float x) { return rintf(x); }
__device__ float op_sin(float x) { return sinf(x); }
__device__ float op_cos(float x) { return cosf(x); }
__device__ float op_softplus(float x) { return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x))); }

#define DEFINE_UNARY(NAME, TYPE, SUFFIX) \
extern "C" __global__ void NAME##_##SUFFIX(const TYPE* x, TYPE* y, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; \
         i += (unsigned long long)gridDim.x * blockDim.x) \
        y[i] = store_float<TYPE>(op_##NAME(load_float<TYPE>(x[i]))); \
}
#define DEFINE_FOR_TYPE(TYPE, SUFFIX) \
DEFINE_UNARY(abs, TYPE, SUFFIX) \
DEFINE_UNARY(neg, TYPE, SUFFIX) \
DEFINE_UNARY(reciprocal, TYPE, SUFFIX) \
DEFINE_UNARY(exp, TYPE, SUFFIX) \
DEFINE_UNARY(log, TYPE, SUFFIX) \
DEFINE_UNARY(sign, TYPE, SUFFIX) \
DEFINE_UNARY(floor, TYPE, SUFFIX) \
DEFINE_UNARY(ceil, TYPE, SUFFIX) \
DEFINE_UNARY(round, TYPE, SUFFIX) \
DEFINE_UNARY(sin, TYPE, SUFFIX) \
DEFINE_UNARY(cos, TYPE, SUFFIX) \
DEFINE_UNARY(softplus, TYPE, SUFFIX)
DEFINE_FOR_TYPE(float, f32)
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_FOR_TYPE(__half, f16)
DEFINE_FOR_TYPE(__nv_bfloat16, bf16)
#endif
"#;

const UNARY_MATH_MODULE: &str = "pointwise_unary_math_float_v2";

/// A supported unary math op and its NVRTC entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryMathOp {
    Abs,
    Neg,
    Reciprocal,
    Exp,
    Log,
    Sign,
    Floor,
    Ceil,
    Round,
    Sin,
    Cos,
    Softplus,
}

impl UnaryMathOp {
    fn stem(self) -> &'static str {
        match self {
            UnaryMathOp::Abs => "abs",
            UnaryMathOp::Neg => "neg",
            UnaryMathOp::Reciprocal => "reciprocal",
            UnaryMathOp::Exp => "exp",
            UnaryMathOp::Log => "log",
            UnaryMathOp::Sign => "sign",
            UnaryMathOp::Floor => "floor",
            UnaryMathOp::Ceil => "ceil",
            UnaryMathOp::Round => "round",
            UnaryMathOp::Sin => "sin",
            UnaryMathOp::Cos => "cos",
            UnaryMathOp::Softplus => "softplus",
        }
    }

    fn entry(self, dtype: FloatDtype) -> String {
        format!("{}_{}", self.stem(), dtype.suffix())
    }

    fn op_name(self) -> &'static str {
        match self {
            UnaryMathOp::Abs => "Abs",
            UnaryMathOp::Neg => "Neg",
            UnaryMathOp::Reciprocal => "Reciprocal",
            UnaryMathOp::Exp => "Exp",
            UnaryMathOp::Log => "Log",
            UnaryMathOp::Sign => "Sign",
            UnaryMathOp::Floor => "Floor",
            UnaryMathOp::Ceil => "Ceil",
            UnaryMathOp::Round => "Round",
            UnaryMathOp::Sin => "Sin",
            UnaryMathOp::Cos => "Cos",
            UnaryMathOp::Softplus => "Softplus",
        }
    }
}

/// Factory for [`UnaryMathKernel`]; carries the op identity and shared runtime.
pub struct UnaryMathFactory {
    pub op: UnaryMathOp,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for UnaryMathFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(UnaryMathKernel {
            op: self.op,
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed f32/f16/bf16 unary math kernel.
#[derive(Debug)]
pub struct UnaryMathKernel {
    op: UnaryMathOp,
    runtime: Arc<CudaRuntime>,
}

impl UnaryMathKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
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
        let n_u64 = count_u64(op, n)?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let entry = self.op.entry(dtype);
        let func = self
            .runtime
            .nvrtc_function(UNARY_MATH_MODULE, UNARY_MATH_SRC, &entry)?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&x_ptr).arg(&y_ptr).arg(&n_u64);
        // SAFETY: `func` is the compiled unary-math entry; the (const float*,
        // float*, unsigned long long) argument list matches its signature;
        // `x_ptr`/`y_ptr` are live device allocations of `n` f32 elements, and
        // the u64 count and indexing cover their validated bounds without overflow.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        self.runtime.synchronize()
    }
}

impl Kernel for UnaryMathKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn cuda_graph_compatible(&self) -> bool {
        true
    }
}

// ===========================================================================
// Not (bool → bool)
// ===========================================================================

/// NVRTC source: boolean negation over raw bytes. Matches the CPU EP
/// (`logical.rs`): a non-zero byte is `true`, output is canonical `1`/`0`.
const NOT_SRC: &str = r#"
extern "C" __global__ void not_bool(const unsigned char* x, unsigned char* y, const unsigned long long n) {
    // CPU: u8::from(b == 0)
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = (x[i] == 0) ? 1 : 0;
}
"#;

const NOT_MODULE: &str = "pointwise_not_bool";

/// Factory for [`NotKernel`] (no attributes).
pub struct NotFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for NotFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(NotKernel {
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed boolean `Not` kernel.
#[derive(Debug)]
pub struct NotKernel {
    runtime: Arc<CudaRuntime>,
}

impl NotKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Not: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        require_dtype("Not", "input", x.dtype, DataType::Bool)?;
        require_dtype("Not", "output", outputs[0].dtype, DataType::Bool)?;
        require_contiguous("Not", "input", x.is_contiguous())?;
        require_contiguous("Not", "output", outputs[0].is_contiguous())?;

        if outputs[0].numel() != x.numel() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Not: output has {} elements, expected {} (same shape as input)",
                outputs[0].numel(),
                x.numel()
            )));
        }

        let n = x.numel();
        let n_u64 = count_u64("Not", n)?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let func = self
            .runtime
            .nvrtc_function(NOT_MODULE, NOT_SRC, "not_bool")?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&x_ptr).arg(&y_ptr).arg(&n_u64);
        // SAFETY: `func` is the compiled `not_bool` entry; the (const uchar*,
        // uchar*, unsigned long long) argument list matches its signature; both
        // pointers are live device allocations of `n` 1-byte bool elements, and
        // the u64 count and indexing cover their validated bounds without overflow.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err("launch not_bool", e))?;
        self.runtime.synchronize()
    }
}

impl Kernel for NotKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn cuda_graph_compatible(&self) -> bool {
        true
    }
}

// ===========================================================================
// Comparison (f32, f32 → bool) — NumPy broadcasting
// ===========================================================================

/// NVRTC source: one `extern "C"` kernel per comparison op — two f32 operands,
/// a 1-byte bool output (canonical `1`/`0`), per ONNX comparison semantics.
const CMP_SRC: &str = r#"
__device__ __forceinline__ void broadcast_indices(unsigned long long out, const unsigned long long* m, int rank, unsigned long long* ai, unsigned long long* bi) {
    *ai = 0; *bi = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
        unsigned long long coord = out % m[axis]; out /= m[axis];
        *ai += coord * m[rank + axis]; *bi += coord * m[2 * rank + axis];
    }
}
#define DEFINE_CMP(name, expr) \
extern "C" __global__ void name(const float* a, const float* b, unsigned char* y, const unsigned long long* m, int rank, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) { \
        unsigned long long ai, bi; broadcast_indices(i, m, rank, &ai, &bi); y[i] = (expr) ? 1 : 0; \
    } \
}
DEFINE_CMP(equal_f32, a[ai] == b[bi])
DEFINE_CMP(greater_f32, a[ai] > b[bi])
DEFINE_CMP(less_f32, a[ai] < b[bi])
DEFINE_CMP(greater_equal_f32, a[ai] >= b[bi])
DEFINE_CMP(less_equal_f32, a[ai] <= b[bi])
"#;

const CMP_MODULE: &str = "pointwise_compare_f32";

/// A supported comparison op (f32 operands, bool output).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Equal,
    Greater,
    Less,
    GreaterOrEqual,
    LessOrEqual,
}

impl CmpOp {
    fn entry(self) -> &'static str {
        match self {
            CmpOp::Equal => "equal_f32",
            CmpOp::Greater => "greater_f32",
            CmpOp::Less => "less_f32",
            CmpOp::GreaterOrEqual => "greater_equal_f32",
            CmpOp::LessOrEqual => "less_equal_f32",
        }
    }

    fn op_name(self) -> &'static str {
        match self {
            CmpOp::Equal => "Equal",
            CmpOp::Greater => "Greater",
            CmpOp::Less => "Less",
            CmpOp::GreaterOrEqual => "GreaterOrEqual",
            CmpOp::LessOrEqual => "LessOrEqual",
        }
    }
}

// ===========================================================================
// Logical (bool, bool → bool) — NumPy broadcasting
// ===========================================================================

/// NVRTC source: one `extern "C"` kernel per logical op — two bool operands (a
/// non-zero byte is `true`, matching the CPU `Not`), 1-byte bool output.
const LOGICAL_SRC: &str = r#"
__device__ __forceinline__ void broadcast_indices(unsigned long long out, const unsigned long long* m, int rank, unsigned long long* ai, unsigned long long* bi) {
    *ai = 0; *bi = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
        unsigned long long coord = out % m[axis]; out /= m[axis];
        *ai += coord * m[rank + axis]; *bi += coord * m[2 * rank + axis];
    }
}
#define DEFINE_LOGICAL(name, expr) \
extern "C" __global__ void name(const unsigned char* a, const unsigned char* b, unsigned char* y, const unsigned long long* m, int rank, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) { \
        unsigned long long ai, bi; broadcast_indices(i, m, rank, &ai, &bi); y[i] = (expr) ? 1 : 0; \
    } \
}
DEFINE_LOGICAL(and_bool, (a[ai] != 0) && (b[bi] != 0))
DEFINE_LOGICAL(or_bool, (a[ai] != 0) || (b[bi] != 0))
DEFINE_LOGICAL(xor_bool, (a[ai] != 0) != (b[bi] != 0))
"#;

const LOGICAL_MODULE: &str = "pointwise_logical_bool";

/// A supported logical op (bool operands, bool output).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalOp {
    And,
    Or,
    Xor,
}

impl LogicalOp {
    fn entry(self) -> &'static str {
        match self {
            LogicalOp::And => "and_bool",
            LogicalOp::Or => "or_bool",
            LogicalOp::Xor => "xor_bool",
        }
    }

    fn op_name(self) -> &'static str {
        match self {
            LogicalOp::And => "And",
            LogicalOp::Or => "Or",
            LogicalOp::Xor => "Xor",
        }
    }
}

/// The dtype contract for a binary op: operand dtype and output dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinaryKind {
    /// f32 operands, bool output (comparison).
    CompareF32,
    /// bool operands, bool output (logical).
    LogicalBool,
}

impl BinaryKind {
    fn operand_dtype(self) -> DataType {
        match self {
            BinaryKind::CompareF32 => DataType::Float32,
            BinaryKind::LogicalBool => DataType::Bool,
        }
    }
}

/// Factory for a binary comparison kernel.
pub struct CmpFactory {
    pub op: CmpOp,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for CmpFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(BinaryPredKernel {
            op_name: self.op.op_name(),
            entry: self.op.entry(),
            module: CMP_MODULE,
            src: CMP_SRC,
            kind: BinaryKind::CompareF32,
            runtime: self.runtime.clone(),
        }))
    }
}

/// Factory for a binary logical kernel.
pub struct LogicalFactory {
    pub op: LogicalOp,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for LogicalFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(BinaryPredKernel {
            op_name: self.op.op_name(),
            entry: self.op.entry(),
            module: LOGICAL_MODULE,
            src: LOGICAL_SRC,
            kind: BinaryKind::LogicalBool,
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed binary predicate kernel producing a **Bool** output. Covers both
/// comparison (f32 operands) and logical (bool operands) families via
/// [`BinaryKind`], with NumPy-style right-aligned broadcasting.
#[derive(Debug)]
pub struct BinaryPredKernel {
    op_name: &'static str,
    entry: &'static str,
    module: &'static str,
    src: &'static str,
    kind: BinaryKind,
    runtime: Arc<CudaRuntime>,
}

impl BinaryPredKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.op_name;
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let a = &inputs[0];
        let b = &inputs[1];
        let operand = self.kind.operand_dtype();
        require_dtype(op, "A", a.dtype, operand)?;
        require_dtype(op, "B", b.dtype, operand)?;
        // Comparison and logical ops always emit Bool.
        require_dtype(op, "output", outputs[0].dtype, DataType::Bool)?;
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
        let n_u64 = count_u64(op, n)?;
        let a_ptr = cuptr(a.data_ptr::<u8>() as *const c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let func = self
            .runtime
            .nvrtc_function(self.module, self.src, self.entry)?;
        let metadata = broadcast_metadata(a.shape, b.shape, &out_shape);
        let metadata_bytes = u64_bytes(&metadata);
        let metadata_ptr = self.runtime.alloc_raw(metadata_bytes.len())?;
        if let Err(error) = unsafe { self.runtime.htod(metadata_bytes, metadata_ptr) } {
            let _ = unsafe { self.runtime.free_raw(metadata_ptr) };
            return Err(error);
        }
        let rank = i32::try_from(out_shape.len())
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: rank exceeds i32")))?;
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
        // SAFETY: `func` is the compiled predicate entry; its argument list is
        // (const T*, const T*, unsigned char*, unsigned long long) where T is f32
        // or uchar per `kind` — matching the validated operand/output dtypes; all
        // pointers cover `n` elements, with a matching u64 count and indexing.
        let launch = unsafe { builder.launch(cfg) }
            .map_err(|e| driver_err(&format!("launch {}", self.entry), e));
        let sync = launch.and_then(|_| self.runtime.synchronize());
        let free = unsafe { self.runtime.free_raw(metadata_ptr) };
        sync.and(free)
    }
}

impl Kernel for BinaryPredKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn cuda_graph_compatible(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_math_entry_points_are_present_in_source() {
        for op in [
            UnaryMathOp::Abs,
            UnaryMathOp::Neg,
            UnaryMathOp::Reciprocal,
            UnaryMathOp::Exp,
            UnaryMathOp::Log,
            UnaryMathOp::Sign,
            UnaryMathOp::Floor,
            UnaryMathOp::Ceil,
            UnaryMathOp::Round,
            UnaryMathOp::Sin,
            UnaryMathOp::Cos,
            UnaryMathOp::Softplus,
        ] {
            assert!(
                UNARY_MATH_SRC.contains(&format!("DEFINE_UNARY({},", op.stem())),
                "missing NVRTC generator for {}",
                op.op_name()
            );
        }
    }

    #[test]
    fn cmp_entry_points_are_present_in_source() {
        for op in [
            CmpOp::Equal,
            CmpOp::Greater,
            CmpOp::Less,
            CmpOp::GreaterOrEqual,
            CmpOp::LessOrEqual,
        ] {
            assert!(
                CMP_SRC.contains(&format!("DEFINE_CMP({},", op.entry())),
                "missing NVRTC entry {} for {}",
                op.entry(),
                op.op_name()
            );
        }
    }

    #[test]
    fn logical_entry_points_are_present_in_source() {
        for op in [LogicalOp::And, LogicalOp::Or, LogicalOp::Xor] {
            assert!(
                LOGICAL_SRC.contains(&format!("DEFINE_LOGICAL({},", op.entry())),
                "missing NVRTC entry {} for {}",
                op.entry(),
                op.op_name()
            );
        }
        assert!(NOT_SRC.contains("void not_bool("), "missing not_bool entry");
    }

    #[test]
    fn round_uses_ties_to_even_intrinsic() {
        // ONNX Round is round-half-to-even; `roundf` (half-away-from-zero) would
        // be wrong, so the kernel must use `rintf`.
        assert!(
            UNARY_MATH_SRC.contains("op_round(float x) { return rintf(x); }"),
            "Round must use rintf"
        );
        assert!(
            !UNARY_MATH_SRC.contains("roundf("),
            "Round must not use half-away-from-zero roundf"
        );
    }

    #[test]
    fn sign_handles_nan_and_zero_like_cpu() {
        // NaN -> NaN (v != v guard) and the zero case falls through to 0.0f.
        assert!(
            UNARY_MATH_SRC.contains("(x != x) ? x"),
            "sign must guard NaN"
        );
    }

    #[test]
    fn entry_points_are_all_distinct() {
        let mut seen = std::collections::HashSet::new();
        let unary = [
            UnaryMathOp::Abs,
            UnaryMathOp::Neg,
            UnaryMathOp::Reciprocal,
            UnaryMathOp::Exp,
            UnaryMathOp::Log,
            UnaryMathOp::Sign,
            UnaryMathOp::Floor,
            UnaryMathOp::Ceil,
            UnaryMathOp::Round,
            UnaryMathOp::Sin,
            UnaryMathOp::Cos,
            UnaryMathOp::Softplus,
        ]
        .map(|o| o.entry(FloatDtype::F32));
        let cmp = [
            CmpOp::Equal,
            CmpOp::Greater,
            CmpOp::Less,
            CmpOp::GreaterOrEqual,
            CmpOp::LessOrEqual,
        ]
        .map(|o| o.entry());
        let logical = [LogicalOp::And, LogicalOp::Or, LogicalOp::Xor].map(|o| o.entry());
        for e in unary
            .into_iter()
            .chain(cmp.map(str::to_owned))
            .chain(logical.map(str::to_owned))
        {
            assert!(seen.insert(e.clone()), "duplicate entry point {e}");
        }
    }

    #[test]
    fn require_dtype_rejects_actionably() {
        let e = require_dtype("Exp", "input", DataType::Int64, DataType::Float32).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("Int64"), "{msg}");
        assert!(msg.contains("Float32"), "{msg}");
    }

    #[test]
    fn require_contiguous_rejects_strided_actionably() {
        let e = require_contiguous("And", "A", false).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("non-contiguous"), "{msg}");
        assert!(msg.contains("materialise"), "{msg}");
    }

    #[test]
    fn operand_dtype_maps_kind_correctly() {
        assert_eq!(BinaryKind::CompareF32.operand_dtype(), DataType::Float32);
        assert_eq!(BinaryKind::LogicalBool.operand_dtype(), DataType::Bool);
    }

    #[test]
    fn grid_covers_all_elements() {
        assert_eq!(grid_for(0), 1);
        assert_eq!(grid_for(1), 1);
        assert_eq!(grid_for(BLOCK as usize), 1);
        assert_eq!(grid_for(BLOCK as usize + 1), 2);
        assert_eq!(grid_for(usize::MAX / 2), 65_535);
    }

    #[test]
    fn near_i32_max_uses_u64_count_and_indexing() {
        let near_i32_max = (i32::MAX as usize) + 1;
        let count: u64 = count_u64("Exp", near_i32_max).unwrap();
        assert_eq!(count, (i32::MAX as u64) + 1);

        const LOOP: &str = "for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)";
        assert!(UNARY_MATH_SRC.contains("const unsigned long long n)"));
        assert!(UNARY_MATH_SRC.contains("for (unsigned long long i ="));
        assert!(UNARY_MATH_SRC.contains("i += (unsigned long long)gridDim.x * blockDim.x"));
        for (name, source, kernel_count) in [
            ("Not", NOT_SRC, 1),
            ("comparison macro", CMP_SRC, 1),
            ("logical macro", LOGICAL_SRC, 1),
        ] {
            assert_eq!(
                source.matches("const unsigned long long n)").count(),
                kernel_count,
                "{name} count parameters must be unsigned 64-bit"
            );
            assert_eq!(
                source.matches(LOOP).count(),
                kernel_count,
                "{name} kernels must use unsigned 64-bit grid-stride indexing"
            );
            assert!(
                !source.contains("const int n)") && !source.contains("for (int i"),
                "{name} source regressed to signed 32-bit indexing"
            );
        }
    }
}

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
//!   shape, **f32**. Each formula is matched **exactly** to the CPU EP
//!   (`crates/onnx-runtime-ep-cpu/src/kernels/unary_math.rs`) so the untestable-
//!   on-this-host kernels stay numerically identical to the reference path.
//! * **Not** (`Not`): boolean element negation, matched to the CPU EP
//!   (`logical.rs` — non-zero byte is `true`, output is canonical `1`/`0`).
//! * **Comparison** (`Equal`, `Greater`, `Less`, `GreaterOrEqual`,
//!   `LessOrEqual`): two **f32** inputs of **equal shape** → **Bool** output.
//! * **Logical** (`And`, `Or`, `Xor`): two **Bool** inputs of **equal shape** →
//!   **Bool** output (non-zero byte is `true`, canonical `1`/`0` out).
//!
//! `dtype`: f32 (math/comparison) / bool (logical/`Not`) only — f16/bf16 are
//! deferred exactly as in [`super::elementwise`] (the same NVRTC source templated
//! on the element type lands with that slice); other dtypes return a "not
//! implemented" error naming the dtype and op.
//!
//! **Broadcasting:** the binary comparison/logical ops require **equal-shape**
//! operands, matching the [`super::elementwise`] binary kernels exactly (NumPy
//! broadcasting is deferred crate-wide; a mismatch returns the same actionable
//! "broadcast/materialise upstream" error). No new broadcasting math is invented
//! here.
//!
//! Each op is one thread-per-element grid-stride kernel (bandwidth-bound,
//! PyTorch-pointwise shaped).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

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

/// NVRTC source: one `extern "C"` f32 pointwise kernel per unary math op. NVRTC
/// resolves the CUDA device intrinsics (`expf`, `logf`, `sinf`, `rintf`,
/// `log1pf`, …) with no header include. Each formula is annotated with the exact
/// CPU-EP expression it mirrors (`unary_math.rs`).
const UNARY_MATH_SRC: &str = r#"
extern "C" __global__ void abs_f32(const float* x, float* y, const unsigned long long n) {
    // CPU: x.abs()
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = fabsf(x[i]);
}
extern "C" __global__ void neg_f32(const float* x, float* y, const unsigned long long n) {
    // CPU: -x
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = -x[i];
}
extern "C" __global__ void reciprocal_f32(const float* x, float* y, const unsigned long long n) {
    // CPU: 1.0 / x
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = 1.0f / x[i];
}
extern "C" __global__ void exp_f32(const float* x, float* y, const unsigned long long n) {
    // CPU: x.exp()
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = expf(x[i]);
}
extern "C" __global__ void log_f32(const float* x, float* y, const unsigned long long n) {
    // CPU: x.ln()  (natural log)
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = logf(x[i]);
}
extern "C" __global__ void sign_f32(const float* x, float* y, const unsigned long long n) {
    // CPU sign(): NaN -> NaN, >0 -> 1, <0 -> -1, else 0 (so sign(0)=0, sign(-0)=0).
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        float v = x[i];
        y[i] = (v != v) ? v : ((v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f));
    }
}
extern "C" __global__ void floor_f32(const float* x, float* y, const unsigned long long n) {
    // CPU: x.floor()
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = floorf(x[i]);
}
extern "C" __global__ void ceil_f32(const float* x, float* y, const unsigned long long n) {
    // CPU: x.ceil()
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = ceilf(x[i]);
}
extern "C" __global__ void round_f32(const float* x, float* y, const unsigned long long n) {
    // CPU: x.round_ties_even() (banker's rounding = ONNX round-half-to-even).
    // rintf rounds to nearest, ties-to-even under the default rounding mode.
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = rintf(x[i]);
}
extern "C" __global__ void sin_f32(const float* x, float* y, const unsigned long long n) {
    // CPU: x.sin()
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = sinf(x[i]);
}
extern "C" __global__ void cos_f32(const float* x, float* y, const unsigned long long n) {
    // CPU: x.cos()
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = cosf(x[i]);
}
extern "C" __global__ void softplus_f32(const float* x, float* y, const unsigned long long n) {
    // CPU: x.max(0.0) + (-x.abs()).exp().ln_1p()  (numerically stable softplus).
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        float v = x[i];
        y[i] = fmaxf(v, 0.0f) + log1pf(expf(-fabsf(v)));
    }
}
"#;

const UNARY_MATH_MODULE: &str = "pointwise_unary_math_f32";

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
    fn entry(self) -> &'static str {
        match self {
            UnaryMathOp::Abs => "abs_f32",
            UnaryMathOp::Neg => "neg_f32",
            UnaryMathOp::Reciprocal => "reciprocal_f32",
            UnaryMathOp::Exp => "exp_f32",
            UnaryMathOp::Log => "log_f32",
            UnaryMathOp::Sign => "sign_f32",
            UnaryMathOp::Floor => "floor_f32",
            UnaryMathOp::Ceil => "ceil_f32",
            UnaryMathOp::Round => "round_f32",
            UnaryMathOp::Sin => "sin_f32",
            UnaryMathOp::Cos => "cos_f32",
            UnaryMathOp::Softplus => "softplus_f32",
        }
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

/// NVRTC-backed f32 unary math kernel.
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
        require_dtype(op, "input", x.dtype, DataType::Float32)?;
        require_dtype(op, "output", outputs[0].dtype, DataType::Float32)?;
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

        let func =
            self.runtime
                .nvrtc_function(UNARY_MATH_MODULE, UNARY_MATH_SRC, self.op.entry())?;
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
        unsafe { builder.launch(cfg) }
            .map_err(|e| driver_err(&format!("launch {}", self.op.entry()), e))?;
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
// Comparison (f32, f32 → bool) — equal shape
// ===========================================================================

/// NVRTC source: one `extern "C"` kernel per comparison op — two f32 operands,
/// a 1-byte bool output (canonical `1`/`0`), per ONNX comparison semantics.
const CMP_SRC: &str = r#"
extern "C" __global__ void equal_f32(const float* a, const float* b, unsigned char* y, const unsigned long long n) {
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = (a[i] == b[i]) ? 1 : 0;
}
extern "C" __global__ void greater_f32(const float* a, const float* b, unsigned char* y, const unsigned long long n) {
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = (a[i] > b[i]) ? 1 : 0;
}
extern "C" __global__ void less_f32(const float* a, const float* b, unsigned char* y, const unsigned long long n) {
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = (a[i] < b[i]) ? 1 : 0;
}
extern "C" __global__ void greater_equal_f32(const float* a, const float* b, unsigned char* y, const unsigned long long n) {
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = (a[i] >= b[i]) ? 1 : 0;
}
extern "C" __global__ void less_equal_f32(const float* a, const float* b, unsigned char* y, const unsigned long long n) {
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = (a[i] <= b[i]) ? 1 : 0;
}
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
// Logical (bool, bool → bool) — equal shape
// ===========================================================================

/// NVRTC source: one `extern "C"` kernel per logical op — two bool operands (a
/// non-zero byte is `true`, matching the CPU `Not`), 1-byte bool output.
const LOGICAL_SRC: &str = r#"
extern "C" __global__ void and_bool(const unsigned char* a, const unsigned char* b, unsigned char* y, const unsigned long long n) {
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = ((a[i] != 0) && (b[i] != 0)) ? 1 : 0;
}
extern "C" __global__ void or_bool(const unsigned char* a, const unsigned char* b, unsigned char* y, const unsigned long long n) {
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = ((a[i] != 0) || (b[i] != 0)) ? 1 : 0;
}
extern "C" __global__ void xor_bool(const unsigned char* a, const unsigned char* b, unsigned char* y, const unsigned long long n) {
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        y[i] = ((a[i] != 0) != (b[i] != 0)) ? 1 : 0;
}
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
/// [`BinaryKind`]; operands must be **equal shape** (broadcasting deferred,
/// matching [`super::elementwise`]).
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

        if a.shape != b.shape {
            return Err(not_implemented(format!(
                "{op} with unequal operand shapes A {:?} vs B {:?} \
                 (NumPy broadcasting is not yet wired on CUDA; broadcast/materialise \
                 the operands to a common shape upstream)",
                a.shape, b.shape
            )));
        }
        if outputs[0].shape != a.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?} must equal the operand shape {:?}",
                outputs[0].shape, a.shape
            )));
        }

        let n = a.numel();
        let n_u64 = count_u64(op, n)?;
        let a_ptr = cuptr(a.data_ptr::<u8>() as *const c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let func = self
            .runtime
            .nvrtc_function(self.module, self.src, self.entry)?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr).arg(&b_ptr).arg(&y_ptr).arg(&n_u64);
        // SAFETY: `func` is the compiled predicate entry; its argument list is
        // (const T*, const T*, unsigned char*, unsigned long long) where T is f32
        // or uchar per `kind` — matching the validated operand/output dtypes; all
        // pointers cover `n` elements, with a matching u64 count and indexing.
        unsafe { builder.launch(cfg) }
            .map_err(|e| driver_err(&format!("launch {}", self.entry), e))?;
        self.runtime.synchronize()
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
        true
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
                UNARY_MATH_SRC.contains(&format!("void {}(", op.entry())),
                "missing NVRTC entry {} for {}",
                op.entry(),
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
                CMP_SRC.contains(&format!("void {}(", op.entry())),
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
                LOGICAL_SRC.contains(&format!("void {}(", op.entry())),
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
            UNARY_MATH_SRC.contains("rintf(x[i])"),
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
            UNARY_MATH_SRC.contains("(v != v) ? v"),
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
        .map(|o| o.entry());
        let cmp = [
            CmpOp::Equal,
            CmpOp::Greater,
            CmpOp::Less,
            CmpOp::GreaterOrEqual,
            CmpOp::LessOrEqual,
        ]
        .map(|o| o.entry());
        let logical = [LogicalOp::And, LogicalOp::Or, LogicalOp::Xor].map(|o| o.entry());
        for e in unary.into_iter().chain(cmp).chain(logical) {
            assert!(seen.insert(e), "duplicate entry point {e}");
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
        for (name, source, kernel_count) in [
            ("unary", UNARY_MATH_SRC, 12),
            ("Not", NOT_SRC, 1),
            ("comparison", CMP_SRC, 5),
            ("logical", LOGICAL_SRC, 3),
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

//! Elementwise **unary** and **binary** ops on the GPU via runtime-compiled
//! (NVRTC) kernels (`docs/ORT2.md` §15; RULES.md #4 — a fused NVRTC elementwise
//! path is the endorsed "custom kernel" case: no NVIDIA library covers arbitrary
//! ONNX elementwise chains, and keeping them as our own kernels is what later
//! enables fusing an activation into a preceding GEMM epilogue).
//!
//! ## Scope of this slice (all limits are actionable errors, never panics)
//!
//! * **dtype:** f32 only. f16/bf16 land next (the same NVRTC source templated on
//!   the element type); other dtypes return a "not implemented" error naming the
//!   dtype and op.
//! * **Unary** (`Relu`, `Sqrt`, `Erf`, `Tanh`, `Sigmoid`, `Gelu`): one input,
//!   one output, identical shape; strided views are rejected with a
//!   "materialise first" error.
//! * **Binary** (`Add`, `Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max`): two inputs of
//!   **equal shape** (element-for-element). NumPy-style broadcasting is deferred
//!   — a mismatch returns an actionable error telling the caller to broadcast
//!   (materialise) the smaller operand upstream. Equal-shape is the dominant
//!   case (residual adds, gate multiplies) and keeps the kernel correct-by-
//!   construction while the broadcasting index math is added later.
//!
//! Each op is one thread-per-element grid-stride kernel; the arithmetic is
//! trivially bandwidth-bound and matches a PyTorch pointwise kernel's shape.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// Threads per block for the 1-D pointwise grids (a full warp-multiple block).
const BLOCK: u32 = 256;

/// NVRTC source: one `extern "C"` f32 pointwise kernel per unary op. NVRTC has
/// no `<math.h>`, but the CUDA device runtime provides the intrinsics
/// (`expf`, `tanhf`, `sqrtf`, `erff`) directly, so no header include is needed.
const UNARY_SRC: &str = r#"
extern "C" __global__ void relu_f32(const float* x, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = fmaxf(x[i], 0.0f);
}
extern "C" __global__ void sqrt_f32(const float* x, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = sqrtf(x[i]);
}
extern "C" __global__ void erf_f32(const float* x, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = erff(x[i]);
}
extern "C" __global__ void tanh_f32(const float* x, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = tanhf(x[i]);
}
extern "C" __global__ void sigmoid_f32(const float* x, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = 1.0f / (1.0f + expf(-x[i]));
}
extern "C" __global__ void gelu_f32(const float* x, float* y, const int n) {
    // Exact (erf) GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
    const float inv_sqrt2 = 0.7071067811865475f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = x[i];
        y[i] = v * 0.5f * (1.0f + erff(v * inv_sqrt2));
    }
}
"#;

/// NVRTC source: one `extern "C"` f32 pointwise kernel per binary op (equal
/// shape; the two operands and the output share an index).
const BINARY_SRC: &str = r#"
extern "C" __global__ void add_f32(const float* a, const float* b, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = a[i] + b[i];
}
extern "C" __global__ void sub_f32(const float* a, const float* b, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = a[i] - b[i];
}
extern "C" __global__ void mul_f32(const float* a, const float* b, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = a[i] * b[i];
}
extern "C" __global__ void div_f32(const float* a, const float* b, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = a[i] / b[i];
}
extern "C" __global__ void pow_f32(const float* a, const float* b, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = powf(a[i], b[i]);
}
extern "C" __global__ void min_f32(const float* a, const float* b, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = fminf(a[i], b[i]);
}
extern "C" __global__ void max_f32(const float* a, const float* b, float* y, const int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = fmaxf(a[i], b[i]);
}
"#;

/// NVRTC module names (one module holds all unary / all binary entries so a
/// runtime compiles each source string at most once — see
/// [`CudaRuntime::nvrtc_function`]).
const UNARY_MODULE: &str = "elementwise_unary_f32";
const BINARY_MODULE: &str = "elementwise_binary_f32";

/// A supported elementwise unary op and its NVRTC entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Relu,
    Sqrt,
    Erf,
    Tanh,
    Sigmoid,
    Gelu,
}

impl UnaryOp {
    fn entry(self) -> &'static str {
        match self {
            UnaryOp::Relu => "relu_f32",
            UnaryOp::Sqrt => "sqrt_f32",
            UnaryOp::Erf => "erf_f32",
            UnaryOp::Tanh => "tanh_f32",
            UnaryOp::Sigmoid => "sigmoid_f32",
            UnaryOp::Gelu => "gelu_f32",
        }
    }

    /// ONNX op type this maps to (for error messages).
    fn op_name(self) -> &'static str {
        match self {
            UnaryOp::Relu => "Relu",
            UnaryOp::Sqrt => "Sqrt",
            UnaryOp::Erf => "Erf",
            UnaryOp::Tanh => "Tanh",
            UnaryOp::Sigmoid => "Sigmoid",
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
    fn entry(self) -> &'static str {
        match self {
            BinaryOp::Add => "add_f32",
            BinaryOp::Sub => "sub_f32",
            BinaryOp::Mul => "mul_f32",
            BinaryOp::Div => "div_f32",
            BinaryOp::Pow => "pow_f32",
            BinaryOp::Min => "min_f32",
            BinaryOp::Max => "max_f32",
        }
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

/// Reject any dtype other than f32 with an actionable, op-named error.
fn require_f32(op: &str, name: &str, dt: DataType) -> Result<()> {
    if dt != DataType::Float32 {
        return Err(not_implemented(format!(
            "{op} with {name} dtype {dt:?} (this slice is f32-only; f16/bf16 pending)"
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
        }))
    }
}

/// NVRTC-backed f32 unary elementwise kernel.
#[derive(Debug)]
pub struct UnaryKernel {
    op: UnaryOp,
    runtime: Arc<CudaRuntime>,
}

impl UnaryKernel {
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
        require_f32(op, "input", x.dtype)?;
        require_f32(op, "output", outputs[0].dtype)?;
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
        let n_i = i32::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {n} elements exceed i32")))?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let func = self
            .runtime
            .nvrtc_function(UNARY_MODULE, UNARY_SRC, self.op.entry())?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&x_ptr).arg(&y_ptr).arg(&n_i);
        // SAFETY: `func` is the compiled unary entry; the (const float*, float*,
        // int) argument list matches its signature; `x_ptr`/`y_ptr` are live
        // device allocations of `n` f32 elements (bounds validated above).
        unsafe { builder.launch(cfg) }
            .map_err(|e| driver_err(&format!("launch {}", self.op.entry()), e))?;
        self.runtime.synchronize()
    }
}

impl Kernel for UnaryKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn cuda_graph_compatible(&self) -> bool {
        // No per-call alloc/free; a pure launch is capturable. (The one-time
        // NVRTC compile happens on the first non-captured call.)
        true
    }
}

/// Factory for [`BinaryKernel`]; carries the op identity and shared runtime.
pub struct BinaryFactory {
    pub op: BinaryOp,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for BinaryFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(BinaryKernel {
            op: self.op,
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed f32 binary elementwise kernel (equal-shape operands).
#[derive(Debug)]
pub struct BinaryKernel {
    op: BinaryOp,
    runtime: Arc<CudaRuntime>,
}

impl BinaryKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
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
        require_f32(op, "A", a.dtype)?;
        require_f32(op, "B", b.dtype)?;
        require_f32(op, "output", outputs[0].dtype)?;
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
        let n_i = i32::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {n} elements exceed i32")))?;
        let a_ptr = cuptr(a.data_ptr::<u8>() as *const c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let func = self
            .runtime
            .nvrtc_function(BINARY_MODULE, BINARY_SRC, self.op.entry())?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr).arg(&b_ptr).arg(&y_ptr).arg(&n_i);
        // SAFETY: `func` is the compiled binary entry; the (const float*, const
        // float*, float*, int) argument list matches its signature; all three
        // pointers are live device allocations of `n` f32 elements (validated).
        unsafe { builder.launch(cfg) }
            .map_err(|e| driver_err(&format!("launch {}", self.op.entry()), e))?;
        self.runtime.synchronize()
    }
}

impl Kernel for BinaryKernel {
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
    fn unary_entry_points_are_distinct_and_named() {
        let ops = [
            UnaryOp::Relu,
            UnaryOp::Sqrt,
            UnaryOp::Erf,
            UnaryOp::Tanh,
            UnaryOp::Sigmoid,
            UnaryOp::Gelu,
        ];
        for op in ops {
            // Every advertised entry must be present verbatim in the NVRTC source.
            assert!(
                UNARY_SRC.contains(op.entry()),
                "missing NVRTC entry {} for {}",
                op.entry(),
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
                BINARY_SRC.contains(op.entry()),
                "missing NVRTC entry {} for {}",
                op.entry(),
                op.op_name()
            );
        }
    }

    #[test]
    fn require_f32_rejects_other_dtypes_actionably() {
        let e = require_f32("Relu", "input", DataType::Int64).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("Int64"), "{msg}");
        assert!(msg.contains("f32-only"), "{msg}");
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
}

//! Fused **normalization** kernels on the GPU via runtime-compiled (NVRTC) f32
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
//! * dtype other than f32 → deferred (names the dtype + op).
//! * `axis`/last-dim size 0, or a `scale`/`bias`/`gamma`/`beta` length that does
//!   not match the normalized size → rejected, naming the offending length.
//! * non-contiguous (strided) operands → "materialise first" error.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{cuptr, CudaRuntime};

use super::softmax::resolve_axis;

/// NVRTC source for the fused f32 `LayerNormalization`. One block per group
/// (`group = prod(shape[..axis])`); the block reduces the mean then the variance
/// over `norm_size = prod(shape[axis..])` in shared memory, then writes the
/// normalized+affine output in a third pass. Optional `mean`/`inv_std` outputs
/// are written when the pointers are non-null.
const LAYERNORM_SRC: &str = r#"
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
"#;

/// NVRTC source for the fused f32 `RMSNormalization` /
/// `SimplifiedLayerNormalization`: no mean subtraction, scale by the inverse
/// root-mean-square.
const RMSNORM_SRC: &str = r#"
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

    // Mean of squares.
    float ss = 0.0f;
    for (int j = tid; j < norm_size; j += nt) {
        const float xv = x[base + j];
        ss += xv * xv;
    }
    red[tid] = ss;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float ms = red[0] / (float)norm_size;
    const float inv_std = 1.0f / sqrtf(ms + epsilon);
    if (tid == 0 && invstd_out) invstd_out[g] = inv_std;

    for (int j = tid; j < norm_size; j += nt)
        y[base + j] = x[base + j] * inv_std * scale[j];
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

const LAYERNORM_MODULE: &str = "layernorm_f32";
const RMSNORM_MODULE: &str = "rmsnorm_f32";
const SKIP_LAYERNORM_MODULE: &str = "skip_layernorm_f32";

/// Threads per block for the norm reductions (power of two → exact tree reduce).
const NORM_BLOCK: u32 = 256;

/// Reject any non-f32 tensor with an actionable, op-named error (RULES.md #1).
fn require_f32(op: &str, name: &str, dt: DataType) -> Result<()> {
    if dt != DataType::Float32 {
        return Err(not_implemented(format!(
            "{op} with {name} dtype {dt:?} (this slice is f32-only; f16/bf16 pending)"
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
    EpError::KernelFailed(format!("cuda_ep {op}: {name} ({v}) exceeds the i32 kernel bound"))
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
        }))
    }
}

/// Fused f32 LayerNormalization kernel.
#[derive(Debug)]
pub struct LayerNormKernel {
    axis: i64,
    epsilon: f32,
    runtime: Arc<CudaRuntime>,
}

impl LayerNormKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
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
        require_f32("LayerNormalization", "X", x.dtype)?;
        require_f32("LayerNormalization", "Scale", scale.dtype)?;
        require_f32("LayerNormalization", "Y", outputs[0].dtype)?;
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
                require_f32("LayerNormalization", "B", b.dtype)?;
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

        let func =
            self.runtime
                .nvrtc_function(LAYERNORM_MODULE, LAYERNORM_SRC, "layernorm_f32")?;
        let cfg = launch_cfg(groups_u);
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&x_ptr)
            .arg(&scale_ptr)
            .arg(&bias_ptr)
            .arg(&y_ptr)
            .arg(&mean_ptr)
            .arg(&invstd_ptr)
            .arg(&groups_i)
            .arg(&norm_i)
            .arg(&has_bias)
            .arg(&eps);
        // SAFETY: `func` is the compiled layernorm entry; the argument list and
        // ABI match its signature; every non-null pointer is a live device
        // allocation sized as validated above (X/Y: num_groups·norm_size;
        // scale/bias: norm_size; mean/invstd: num_groups).
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err("launch layernorm_f32", e))?;
        self.runtime.synchronize()
    }
}

impl Kernel for LayerNormKernel {
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

// ─────────────────────── RMSNorm / SimplifiedLayerNorm ──────────────────────

/// Factory reading `axis` (default -1) and `epsilon` (default 1e-5).
pub struct RmsNormFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for RmsNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1);
        let epsilon = node
            .attr("epsilon")
            .and_then(|a| a.as_float())
            .unwrap_or(1e-5);
        Ok(Box::new(RmsNormKernel {
            axis,
            epsilon,
            runtime: self.runtime.clone(),
        }))
    }
}

/// Fused f32 RMSNormalization / SimplifiedLayerNormalization kernel.
#[derive(Debug)]
pub struct RmsNormKernel {
    axis: i64,
    epsilon: f32,
    runtime: Arc<CudaRuntime>,
}

impl RmsNormKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
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
        require_f32(op, "X", x.dtype)?;
        require_f32(op, "Scale", scale.dtype)?;
        require_f32(op, "Y", outputs[0].dtype)?;
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

        let func = self
            .runtime
            .nvrtc_function(RMSNORM_MODULE, RMSNORM_SRC, "rmsnorm_f32")?;
        let cfg = launch_cfg(groups_u);
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        let groups_i = groups_u_i32(groups_u);
        builder
            .arg(&x_ptr)
            .arg(&scale_ptr)
            .arg(&y_ptr)
            .arg(&invstd_ptr)
            .arg(&groups_i)
            .arg(&norm_i)
            .arg(&eps);
        // SAFETY: `func` is the compiled rmsnorm entry; the argument list/ABI
        // match; pointers are live device allocations sized as validated.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err("launch rmsnorm_f32", e))?;
        self.runtime.synchronize()
    }
}

impl Kernel for RmsNormKernel {
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
}

impl SkipLayerNormKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
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
        let cfg = launch_cfg(groups_u);
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
        self.runtime.synchronize()
    }
}

impl Kernel for SkipLayerNormKernel {
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

// ───────────────────────────────── helpers ─────────────────────────────────

fn launch_cfg(groups: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (groups, 1, 1),
        block_dim: (NORM_BLOCK, 1, 1),
        shared_mem_bytes: NORM_BLOCK * std::mem::size_of::<f32>() as u32,
    }
}

/// The kernels take `num_groups` as a signed `int`; convert the validated `u32`.
fn groups_u_i32(groups: u32) -> i32 {
    groups as i32
}

/// Resolve the optional per-group `Mean` (output slot 1) and `InvStdDev` (slot 2)
/// device pointers, validating f32 dtype and `num_groups` length when present.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_expose_their_entry_points() {
        assert!(LAYERNORM_SRC.contains("layernorm_f32"));
        assert!(RMSNORM_SRC.contains("rmsnorm_f32"));
        assert!(SKIP_LAYERNORM_SRC.contains("skip_layernorm_f32"));
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

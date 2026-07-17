//! Standalone axis-reduction **Softmax** on the GPU via
//! `cudnnSoftmaxForward`, with the previous f32 NVRTC kernel retained as a
//! compatibility fallback when the dynamically-loaded cuDNN runtime is absent.
//!
//! cuDNN uses its `ACCURATE` algorithm for numerically-stable f32/f16/bf16
//! execution and supplies the broad-device compatibility required by the
//! library-first CUDA strategy. The fallback mirrors the CPU EP for f32.
//!
//! ## Arbitrary axis
//!
//! The tensor is viewed as `[outer, axis_dim, inner]` where `outer =
//! prod(shape[..axis])`, `axis_dim = shape[axis]`, `inner = prod(shape[axis+1..])`.
//! Element `(o, a, i)` lives at `o·axis_dim·inner + a·inner + i`. With
//! `inner == 1` this is a plain row-major softmax; with `inner > 1` it reduces
//! along an interior axis. One thread block per `(o, i)` group.
//!
//! * **opset ≥ 13** ([`SoftmaxFactory`], `coerce_2d = false`): normalize over the
//!   single `axis`.
//! * **opset ≤ 12** ([`SoftmaxLegacyFactory`], `coerce_2d = true`): coerce to the
//!   2-D matrix `[d_0·…·d_{axis-1}, d_axis·…·d_{n-1}]` and softmax each row over
//!   the whole flattened trailing block (`axis_dim = prod(shape[axis..])`,
//!   `inner = 1`).
//!
//! ## Limits (all actionable errors, never panics — RULES.md #1)
//!
//! * dtype other than f32 → deferred (names the dtype + `Softmax`).
//! * rank 0 → rejected (softmax needs at least one axis).
//! * `axis` out of `[-rank, rank)` → rejected, naming the offending axis.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::PushKernelArg;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::cudnn::{CudnnBufferPair, CudnnSoftmaxMode, TensorDescriptorSpec};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// NVRTC source: numerically-stable f32 softmax over the middle (`axis_dim`)
/// dimension of an `[outer, axis_dim, inner]` view. One block per `(o, i)`
/// group; the block cooperatively reduces the row max then the row sum in shared
/// memory. `NEG_INF` is built from its bit pattern (NVRTC has no `<math.h>`).
const SOFTMAX_SRC: &str = r#"
extern "C" __global__ void softmax_f32(
    const float* x,
    float*       y,
    const int    outer,
    const int    axis_dim,
    const int    inner)
{
    const float NEG_INF = __int_as_float(0xff800000);

    const int group = blockIdx.x;          // in [0, outer*inner)
    const int total = outer * inner;
    if (group >= total) return;
    const int o = group / inner;
    const int i = group % inner;
    const size_t base = (size_t)o * axis_dim * inner + i;

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    // Pass 1: row max (stable-softmax shift).
    float local_max = NEG_INF;
    for (int a = tid; a < axis_dim; a += nt)
        local_max = fmaxf(local_max, x[base + (size_t)a * inner]);
    red[tid] = local_max;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    const float row_max = red[0];
    __syncthreads();

    // Pass 2: exponentiate (stable) and accumulate the row sum.
    float local_sum = 0.0f;
    for (int a = tid; a < axis_dim; a += nt) {
        const float e = expf(x[base + (size_t)a * inner] - row_max);
        y[base + (size_t)a * inner] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float row_sum = red[0];
    __syncthreads();

    // Pass 3: normalize (guard a degenerate all-equal / empty row).
    const float invs = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int a = tid; a < axis_dim; a += nt)
        y[base + (size_t)a * inner] *= invs;
}
"#;

const SOFTMAX_MODULE: &str = "softmax_f32";
const SOFTMAX_ENTRY: &str = "softmax_f32";

/// Threads per block for the reduction (a power of two so the tree reduction is
/// exact); rows longer than this are covered by the strided per-thread loop.
const SOFTMAX_BLOCK: u32 = 256;

/// Factory for the opset ≥ 13 per-axis `Softmax` (`axis` default 1, matching the
/// CPU EP).
pub struct SoftmaxFactory {
    pub runtime: Arc<CudaRuntime>,
}

/// Factory for the legacy opset ≤ 12 coerce-to-2D `Softmax` (`axis` default 1).
pub struct SoftmaxLegacyFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for SoftmaxFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1);
        Ok(Box::new(SoftmaxKernel {
            axis,
            coerce_2d: false,
            runtime: self.runtime.clone(),
        }))
    }
}

impl KernelFactory for SoftmaxLegacyFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1);
        Ok(Box::new(SoftmaxKernel {
            axis,
            coerce_2d: true,
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed f32 softmax kernel carrying the raw `axis` and opset semantics.
#[derive(Debug)]
pub struct SoftmaxKernel {
    axis: i64,
    /// `true` for opset ≤ 12 (coerce-to-2D over the flattened trailing block);
    /// `false` for opset ≥ 13 (normalize over the single `axis`).
    coerce_2d: bool,
    runtime: Arc<CudaRuntime>,
}

/// Resolve the `[outer, axis_dim, inner]` view for a `shape` + normalized
/// `axis`, honouring the opset `coerce_2d` mode. Pure host arithmetic — unit
/// tested without a GPU.
pub(crate) fn softmax_view(shape: &[usize], axis: usize, coerce_2d: bool) -> (usize, usize, usize) {
    if coerce_2d {
        // opset ≤ 12: `[prod(shape[..axis]), prod(shape[axis..])]`; trailing
        // block is contiguous so `inner == 1`.
        let outer: usize = shape[..axis].iter().product();
        let axis_dim: usize = shape[axis..].iter().product();
        (outer, axis_dim, 1)
    } else {
        // opset ≥ 13: single-axis view `[outer, axis_dim, inner]`.
        let outer: usize = shape[..axis].iter().product();
        let axis_dim = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        (outer, axis_dim, inner)
    }
}

/// Build the explicit 4-D cuDNN view and mode for ONNX softmax semantics.
pub(crate) fn cudnn_softmax_spec(
    dtype: DataType,
    outer: usize,
    axis_dim: usize,
    inner: usize,
    coerce_2d: bool,
) -> Result<(TensorDescriptorSpec, CudnnSoftmaxMode)> {
    let dims = [outer, axis_dim, inner, 1];
    let strides = [axis_dim * inner, inner, 1, 1];
    let mode = if coerce_2d {
        CudnnSoftmaxMode::Instance
    } else {
        CudnnSoftmaxMode::Channel
    };
    Ok((TensorDescriptorSpec::new(dtype, &dims, &strides)?, mode))
}

/// Normalize a possibly-negative `axis` against `rank`, rejecting out-of-range
/// values with an actionable, op-named error (RULES.md #1).
pub(crate) fn resolve_axis(op: &str, axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a as usize >= rank {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: axis {axis} is out of range for a rank-{rank} input; \
             axis must lie in [-{rank}, {rank})"
        )));
    }
    Ok(a as usize)
}

impl SoftmaxKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Softmax: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        if !matches!(
            x.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(not_implemented(format!(
                "Softmax with input dtype {:?} (cuDNN supports f32/f16/bf16)",
                x.dtype
            )));
        }
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Softmax: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        if !x.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented(
                "Softmax with a non-contiguous (strided) input/output; \
                 insert an explicit copy to materialise it before the op",
            ));
        }
        let rank = x.shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep Softmax: input must have rank >= 1".into(),
            ));
        }
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Softmax: output shape {:?} must equal input shape {:?}",
                outputs[0].shape, x.shape
            )));
        }
        let axis = resolve_axis("Softmax", self.axis, rank)?;
        let (outer, axis_dim, inner) = softmax_view(x.shape, axis, self.coerce_2d);

        // Nothing to do for an empty tensor.
        let groups = outer * inner;
        if groups == 0 || axis_dim == 0 {
            return Ok(());
        }

        if self.runtime.cudnn().is_available() {
            let (spec, mode) = cudnn_softmax_spec(x.dtype, outer, axis_dim, inner, self.coerce_2d)?;
            let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
            let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
            self.runtime.cudnn().with_handle(|handle| {
                handle.softmax(
                    &spec,
                    mode,
                    CudnnBufferPair {
                        input: x_ptr,
                        output: y_ptr,
                        input_numel: x.numel(),
                        output_numel: outputs[0].numel(),
                    },
                )
            })?;
            return self.runtime.synchronize();
        }

        if x.dtype != DataType::Float32 {
            // Entering `with_handle` produces the backend's actionable missing
            // runtime error for dtypes that have no handwritten fallback.
            return self.runtime.cudnn().with_handle(|_| Ok(()));
        }

        self.run_nvrtc_f32(x, outputs, outer, axis_dim, inner, groups)
    }

    fn run_nvrtc_f32(
        &self,
        x: &TensorView,
        outputs: &mut [TensorMut],
        outer: usize,
        axis_dim: usize,
        inner: usize,
        groups: usize,
    ) -> Result<()> {
        let (outer_i, axis_i, inner_i) = (
            i32::try_from(outer).map_err(|_| dim_overflow("outer", outer))?,
            i32::try_from(axis_dim).map_err(|_| dim_overflow("axis_dim", axis_dim))?,
            i32::try_from(inner).map_err(|_| dim_overflow("inner", inner))?,
        );
        let groups_u = u32::try_from(groups).map_err(|_| dim_overflow("groups", groups))?;

        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let func = self
            .runtime
            .nvrtc_function(SOFTMAX_MODULE, SOFTMAX_SRC, SOFTMAX_ENTRY)?;
        let cfg = self.runtime.reduction_launch_config(
            &func,
            groups_u,
            SOFTMAX_BLOCK,
            std::mem::size_of::<f32>() as u32,
        )?;
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&x_ptr)
            .arg(&y_ptr)
            .arg(&outer_i)
            .arg(&axis_i)
            .arg(&inner_i);
        // SAFETY: `func` is the compiled softmax entry; the (const float*, float*,
        // int, int, int) argument list matches its signature; `x_ptr`/`y_ptr` are
        // live device allocations of `outer·axis_dim·inner` f32 elements.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err("launch softmax_f32", e))?;
        self.runtime.synchronize()
    }
}

fn dim_overflow(name: &str, v: usize) -> EpError {
    EpError::KernelFailed(format!(
        "cuda_ep Softmax: {name} ({v}) exceeds the i32 kernel bound"
    ))
}

impl Kernel for SoftmaxKernel {
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
    fn entry_point_present_in_source() {
        assert!(SOFTMAX_SRC.contains(SOFTMAX_ENTRY));
    }

    #[test]
    fn v13_view_is_single_axis() {
        // shape [2,3,4], axis 1 -> outer=2, axis_dim=3, inner=4.
        assert_eq!(softmax_view(&[2, 3, 4], 1, false), (2, 3, 4));
        // last-axis softmax: inner collapses to 1.
        assert_eq!(softmax_view(&[2, 3, 4], 2, false), (6, 4, 1));
    }

    #[test]
    fn legacy_view_coerces_to_2d() {
        // shape [2,3,4], axis 1 -> rows=2, cols=3*4=12, inner=1.
        assert_eq!(softmax_view(&[2, 3, 4], 1, true), (2, 12, 1));
        // The two modes coincide when axis is the last dim.
        assert_eq!(
            softmax_view(&[2, 3, 4], 2, true),
            softmax_view(&[2, 3, 4], 2, false)
        );
    }

    #[test]
    fn cudnn_layout_selects_onnx_semantics() {
        let (v13, mode) = cudnn_softmax_spec(DataType::Float16, 2, 3, 4, false).unwrap();
        assert_eq!(mode, CudnnSoftmaxMode::Channel);
        assert_eq!(v13.dims(), &[2, 3, 4, 1]);
        assert_eq!(v13.strides(), &[12, 4, 1, 1]);

        let (legacy, mode) = cudnn_softmax_spec(DataType::BFloat16, 2, 12, 1, true).unwrap();
        assert_eq!(mode, CudnnSoftmaxMode::Instance);
        assert_eq!(legacy.dims(), &[2, 12, 1, 1]);
        assert_eq!(legacy.strides(), &[12, 1, 1, 1]);
    }

    #[test]
    fn resolve_axis_handles_negatives_and_rejects_out_of_range() {
        assert_eq!(resolve_axis("Softmax", -1, 3).unwrap(), 2);
        assert_eq!(resolve_axis("Softmax", 0, 3).unwrap(), 0);
        let e = resolve_axis("Softmax", 3, 3).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("out of range"), "{msg}");
        assert!(msg.contains("axis 3"), "{msg}");
    }
}

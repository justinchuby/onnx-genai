//! Numerically-stable **`LogSoftmax`** on the GPU via a runtime-compiled (NVRTC)
//! kernel. Mirrors the CPU EP (`crates/onnx-runtime-ep-cpu/src/kernels/
//! log_softmax.rs`): the reduction slice's maximum is subtracted before the
//! `exp`, so the `logsumexp` is computed as `max + log(sum(exp(x - max)))` and
//! the result is `(x - max) - log_sum` — never the overflow-prone naive
//! `log(sum(exp(x)))` (the #266 LogSumExp lesson).
//!
//! ## Arbitrary axis
//!
//! The tensor is viewed as `[outer, axis_dim, inner]`, identical to
//! [`super::softmax`]. One thread block per `(outer, inner)` group cooperatively
//! reduces the slice maximum then the shifted-exp sum in shared memory.
//!
//! * **opset ≥ 13** ([`LogSoftmaxFactory`], default `axis = -1`): normalize over
//!   the single `axis`.
//! * **opset ≤ 12** ([`LogSoftmaxLegacyFactory`], default `axis = 1`): coerce to
//!   2-D `[prod(shape[..axis]), prod(shape[axis..])]` and reduce each row.
//!
//! Half inputs (`f16`/`bf16`) are widened to f32 for the reduction and narrowed
//! once on store, matching the CPU EP's widened compute domain.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::PushKernelArg;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::softmax::{resolve_axis, softmax_view};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// NVRTC source: a numerically-stable f32/f16/bf16 log-softmax over the middle
/// (`axis_dim`) dimension of an `[outer, axis_dim, inner]` view. One block per
/// `(o, i)` group; the block reduces the row max then the shifted-exp sum in
/// shared memory. Half storage is widened to f32 for the math and narrowed on
/// store. `NEG_INF` is built from its bit pattern (NVRTC has no `<math.h>`).
const LOG_SOFTMAX_SRC: &str = r#"
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

template <typename T>
__device__ void log_softmax_impl(const T* x, T* y, int outer, int axis_dim, int inner) {
    const float NEG_INF = __int_as_float(0xff800000);
    const int group = blockIdx.x;
    const int total = outer * inner;
    if (group >= total) return;
    const int o = group / inner;
    const int i = group % inner;
    const size_t base = (size_t)o * axis_dim * inner + i;

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    // Pass 1: row max (stable shift).
    float local_max = NEG_INF;
    for (int a = tid; a < axis_dim; a += nt)
        local_max = fmaxf(local_max, load_float<T>(x[base + (size_t)a * inner]));
    red[tid] = local_max;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    const float row_max = red[0];
    __syncthreads();

    // Pass 2: accumulate sum(exp(x - max)).
    float local_sum = 0.0f;
    for (int a = tid; a < axis_dim; a += nt)
        local_sum += expf(load_float<T>(x[base + (size_t)a * inner]) - row_max);
    red[tid] = local_sum;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float log_sum = logf(red[0]);
    __syncthreads();

    // Pass 3: (x - max) - log_sum.
    for (int a = tid; a < axis_dim; a += nt) {
        const float v = load_float<T>(x[base + (size_t)a * inner]);
        y[base + (size_t)a * inner] = store_float<T>((v - row_max) - log_sum);
    }
}

extern "C" __global__ void log_softmax_f32(const float* x, float* y, int outer, int axis_dim, int inner) {
    log_softmax_impl<float>(x, y, outer, axis_dim, inner);
}
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
extern "C" __global__ void log_softmax_f16(const __half* x, __half* y, int outer, int axis_dim, int inner) {
    log_softmax_impl<__half>(x, y, outer, axis_dim, inner);
}
extern "C" __global__ void log_softmax_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int outer, int axis_dim, int inner) {
    log_softmax_impl<__nv_bfloat16>(x, y, outer, axis_dim, inner);
}
#endif
"#;

const LOG_SOFTMAX_MODULE: &str = "log_softmax_v1";

/// Threads per block for the reduction (a power of two so the tree reduction is
/// exact); rows longer than this are covered by the strided per-thread loop.
const LOG_SOFTMAX_BLOCK: u32 = 256;

/// Factory for the opset ≥ 13 per-axis `LogSoftmax` (default `axis = -1`).
pub struct LogSoftmaxFactory {
    pub runtime: Arc<CudaRuntime>,
}

/// Factory for the legacy opset ≤ 12 coerce-to-2D `LogSoftmax` (`axis` default 1).
pub struct LogSoftmaxLegacyFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for LogSoftmaxFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1);
        Ok(Box::new(LogSoftmaxKernel {
            axis,
            coerce_2d: false,
            runtime: self.runtime.clone(),
        }))
    }
}

impl KernelFactory for LogSoftmaxLegacyFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1);
        Ok(Box::new(LogSoftmaxKernel {
            axis,
            coerce_2d: true,
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed f32/f16/bf16 log-softmax kernel carrying the raw `axis` and
/// opset semantics.
#[derive(Debug)]
pub struct LogSoftmaxKernel {
    axis: i64,
    /// `true` for opset ≤ 12 (coerce-to-2D over the flattened trailing block);
    /// `false` for opset ≥ 13 (normalize over the single `axis`).
    coerce_2d: bool,
    runtime: Arc<CudaRuntime>,
}

impl LogSoftmaxKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LogSoftmax: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let suffix = match x.dtype {
            DataType::Float32 => "f32",
            DataType::Float16 => "f16",
            DataType::BFloat16 => "bf16",
            other => {
                return Err(not_implemented(format!(
                    "LogSoftmax with input dtype {other:?} (supported: Float32, Float16, BFloat16)"
                )));
            }
        };
        if x.dtype != DataType::Float32 {
            self.runtime.require_nvrtc_half_headers("LogSoftmax")?;
        }
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LogSoftmax: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        if !x.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented(
                "LogSoftmax with a non-contiguous (strided) input/output; \
                 insert an explicit copy to materialise it before the op",
            ));
        }
        let rank = x.shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep LogSoftmax: input must have rank >= 1".into(),
            ));
        }
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LogSoftmax: output shape {:?} must equal input shape {:?}",
                outputs[0].shape, x.shape
            )));
        }
        let axis = resolve_axis("LogSoftmax", self.axis, rank)?;
        let (outer, axis_dim, inner) = softmax_view(x.shape, axis, self.coerce_2d);

        let groups = outer * inner;
        if groups == 0 || axis_dim == 0 {
            return Ok(());
        }

        let (outer_i, axis_i, inner_i) = (
            i32::try_from(outer).map_err(|_| dim_overflow("outer", outer))?,
            i32::try_from(axis_dim).map_err(|_| dim_overflow("axis_dim", axis_dim))?,
            i32::try_from(inner).map_err(|_| dim_overflow("inner", inner))?,
        );
        let groups_u = u32::try_from(groups).map_err(|_| dim_overflow("groups", groups))?;

        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let entry = format!("log_softmax_{suffix}");
        let func = self
            .runtime
            .nvrtc_function(LOG_SOFTMAX_MODULE, LOG_SOFTMAX_SRC, &entry)?;
        let cfg = self.runtime.reduction_launch_config(
            &func,
            groups_u,
            LOG_SOFTMAX_BLOCK,
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
        // SAFETY: `func` is the compiled log-softmax entry for the validated
        // dtype; the (const T*, T*, int, int, int) argument list matches its
        // signature; `x_ptr`/`y_ptr` are live device allocations of
        // `outer·axis_dim·inner` elements.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        if !self.runtime.is_capturing()? {
            self.runtime.synchronize()?;
        }
        Ok(())
    }
}

fn dim_overflow(name: &str, v: usize) -> EpError {
    EpError::KernelFailed(format!(
        "cuda_ep LogSoftmax: {name} ({v}) exceeds the i32 kernel bound"
    ))
}

impl Kernel for LogSoftmaxKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::Supported
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_points_present_in_source() {
        for entry in ["log_softmax_f32", "log_softmax_f16", "log_softmax_bf16"] {
            assert!(LOG_SOFTMAX_SRC.contains(entry), "missing {entry}");
        }
    }

    #[test]
    fn uses_stable_shifted_logsumexp_not_naive_log_sum_exp() {
        // The reduction must subtract the row max before exp and add log_sum of
        // the shifted exps (the #266 LogSumExp lesson).
        assert!(
            LOG_SOFTMAX_SRC.contains("expf(load_float<T>(x[base + (size_t)a * inner]) - row_max)")
        );
        assert!(LOG_SOFTMAX_SRC.contains("(v - row_max) - log_sum"));
    }
}

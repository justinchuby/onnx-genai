//! **`Hardmax`** on the GPU via a runtime-compiled (NVRTC) kernel. Mirrors the
//! CPU EP (`crates/onnx-runtime-ep-cpu/src/kernels/hardmax.rs`): the output is a
//! one-hot tensor selecting the **first** maximal element along `axis` (ties
//! resolve to the lowest index), every other element is `0`.
//!
//! The tensor is viewed as `[outer, axis_dim, inner]` (identical to
//! [`super::softmax`]). One thread owns a whole `(outer, inner)` slice: it scans
//! the axis to find the first argmax, then writes the one-hot row. Half inputs
//! are widened to f32 for the comparison — a lossless `f16`/`bf16` → `f32`
//! widening, so the selected index matches the CPU EP's native-precision
//! comparison exactly.
//!
//! * default `axis = -1`, matching the CPU EP (opset ≥ 13 semantics).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::softmax::{resolve_axis, softmax_view};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// NVRTC source: first-argmax one-hot over the middle (`axis_dim`) dimension of
/// an `[outer, axis_dim, inner]` view. One thread per `(o, i)` group. Half
/// storage is widened to f32 for the comparison and the `0`/`1` outputs are
/// narrowed on store. The strict `>` keeps the first (lowest-index) maximum,
/// matching the CPU EP.
const HARDMAX_SRC: &str = r#"
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
__device__ void hardmax_impl(const T* x, T* y, unsigned long long outer, int axis_dim, unsigned long long inner) {
    const unsigned long long total = outer * inner;
    for (unsigned long long group = blockIdx.x*blockDim.x + threadIdx.x; group < total;
         group += (unsigned long long)gridDim.x * blockDim.x) {
        const unsigned long long o = group / inner;
        const unsigned long long i = group % inner;
        const size_t base = (size_t)o * axis_dim * inner + i;

        int best = 0;
        float best_val = load_float<T>(x[base]);
        for (int a = 1; a < axis_dim; ++a) {
            const float v = load_float<T>(x[base + (size_t)a * inner]);
            if (v > best_val) { best_val = v; best = a; }
        }
        for (int a = 0; a < axis_dim; ++a)
            y[base + (size_t)a * inner] = store_float<T>(a == best ? 1.0f : 0.0f);
    }
}

extern "C" __global__ void hardmax_f32(const float* x, float* y, unsigned long long outer, int axis_dim, unsigned long long inner) {
    hardmax_impl<float>(x, y, outer, axis_dim, inner);
}
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
extern "C" __global__ void hardmax_f16(const __half* x, __half* y, unsigned long long outer, int axis_dim, unsigned long long inner) {
    hardmax_impl<__half>(x, y, outer, axis_dim, inner);
}
extern "C" __global__ void hardmax_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, unsigned long long outer, int axis_dim, unsigned long long inner) {
    hardmax_impl<__nv_bfloat16>(x, y, outer, axis_dim, inner);
}
#endif
"#;

const HARDMAX_MODULE: &str = "hardmax_v1";

/// Threads per block for the 1-D group grid.
const HARDMAX_BLOCK: u32 = 256;

/// Grid dimension for `groups` at [`HARDMAX_BLOCK`] threads, capped to the grid
/// limit (the kernel is grid-stride, so a capped grid still covers every group).
fn grid_for(groups: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    groups.div_ceil(HARDMAX_BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

/// Factory for `Hardmax` (default `axis = -1`, matching the CPU EP).
pub struct HardmaxFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for HardmaxFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1);
        Ok(Box::new(HardmaxKernel {
            axis,
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed f32/f16/bf16 `Hardmax` kernel.
#[derive(Debug)]
pub struct HardmaxKernel {
    axis: i64,
    runtime: Arc<CudaRuntime>,
}

impl HardmaxKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Hardmax: expected 1 input and 1 output, got {} and {}",
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
                    "Hardmax with input dtype {other:?} (supported: Float32, Float16, BFloat16)"
                )));
            }
        };
        if x.dtype != DataType::Float32 {
            self.runtime.require_nvrtc_half_headers("Hardmax")?;
        }
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Hardmax: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        if !x.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented(
                "Hardmax with a non-contiguous (strided) input/output; \
                 insert an explicit copy to materialise it before the op",
            ));
        }
        let rank = x.shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep Hardmax: input must have rank >= 1".into(),
            ));
        }
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Hardmax: output shape {:?} must equal input shape {:?}",
                outputs[0].shape, x.shape
            )));
        }
        let axis = resolve_axis("Hardmax", self.axis, rank)?;
        // Hardmax always selects along the single `axis` (no legacy coerce-2D).
        let (outer, axis_dim, inner) = softmax_view(x.shape, axis, false);

        let groups = outer * inner;
        if groups == 0 {
            return Ok(());
        }
        if axis_dim == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep Hardmax: selected axis must be non-empty".into(),
            ));
        }

        let axis_i = i32::try_from(axis_dim).map_err(|_| dim_overflow("axis_dim", axis_dim))?;
        let outer_u = u64::try_from(outer).map_err(|_| dim_overflow("outer", outer))?;
        let inner_u = u64::try_from(inner).map_err(|_| dim_overflow("inner", inner))?;

        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let entry = format!("hardmax_{suffix}");
        let func = self
            .runtime
            .nvrtc_function(HARDMAX_MODULE, HARDMAX_SRC, &entry)?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(groups), 1, 1),
            block_dim: (HARDMAX_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&x_ptr)
            .arg(&y_ptr)
            .arg(&outer_u)
            .arg(&axis_i)
            .arg(&inner_u);
        // SAFETY: `func` is the compiled hardmax entry for the validated dtype;
        // the (const T*, T*, u64, int, u64) argument list matches its signature;
        // `x_ptr`/`y_ptr` are live device allocations of `outer·axis_dim·inner`
        // elements.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        if !self.runtime.is_capturing()? {
            self.runtime.synchronize()?;
        }
        Ok(())
    }
}

fn dim_overflow(name: &str, v: usize) -> EpError {
    EpError::KernelFailed(format!(
        "cuda_ep Hardmax: {name} ({v}) exceeds the kernel bound"
    ))
}

impl Kernel for HardmaxKernel {
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
        for entry in ["hardmax_f32", "hardmax_f16", "hardmax_bf16"] {
            assert!(HARDMAX_SRC.contains(entry), "missing {entry}");
        }
    }

    #[test]
    fn selects_first_maximum_with_strict_greater() {
        // Strict `>` keeps the lowest-index maximum on ties, matching the CPU EP.
        assert!(HARDMAX_SRC.contains("if (v > best_val)"));
    }

    #[test]
    fn grid_covers_all_groups() {
        assert_eq!(grid_for(0), 1);
        assert_eq!(grid_for(HARDMAX_BLOCK as usize + 1), 2);
        assert_eq!(grid_for(usize::MAX / 2), 65_535);
    }
}

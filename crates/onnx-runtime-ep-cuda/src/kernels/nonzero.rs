//! Deterministic ONNX `NonZero` coordinate extraction.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const COORD_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
template <typename T> __device__ bool nz(T v) { return v != (T)0; }
template <> __device__ bool nz(__half v) { return __half2float(v) != 0.0f; }
template <> __device__ bool nz(__nv_bfloat16 v) { return __bfloat162float(v) != 0.0f; }
#define DEFINE_NONZERO(TYPE, SUFFIX) \
extern "C" __global__ void nonzero_##SUFFIX( \
    const TYPE* x, long long* y, const unsigned long long* strides, \
    unsigned long long n, unsigned long long rank, unsigned long long expected) { \
  if (blockIdx.x != 0 || threadIdx.x != 0) return; \
  unsigned long long found = 0; \
  for (unsigned long long linear = 0; linear < n; ++linear) { \
    if (!nz<TYPE>(x[linear])) continue; \
    if (found >= expected) return; \
    unsigned long long rem = linear; \
    for (unsigned long long axis = 0; axis < rank; ++axis) { \
      y[axis * expected + found] = (long long)(rem / strides[axis]); \
      rem %= strides[axis]; \
    } \
    ++found; \
  } \
}
DEFINE_NONZERO(float, f32)
DEFINE_NONZERO(__half, f16)
DEFINE_NONZERO(__nv_bfloat16, bf16)
DEFINE_NONZERO(unsigned char, bool_)
"#;

pub struct NonZeroFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for NonZeroFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(NonZeroKernel {
            runtime: self.runtime.clone(),
        }))
    }
}

struct NonZeroKernel {
    runtime: Arc<CudaRuntime>,
}

impl Kernel for NonZeroKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep NonZero: expected 1 input and 1 output".into(),
            ));
        }
        let input = &inputs[0];
        let output = &mut outputs[0];
        if !input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented("NonZero with strided tensors"));
        }
        let suffix = match input.dtype {
            DataType::Float32 => "f32",
            DataType::Float16 => "f16",
            DataType::BFloat16 => "bf16",
            DataType::Bool => "bool_",
            dtype => return Err(not_implemented(format!("NonZero for dtype {dtype:?}"))),
        };
        let rank = input.shape.len();
        if output.dtype != DataType::Int64 || output.shape.first() != Some(&rank) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep NonZero: output must be Int64 [{rank}, count]"
            )));
        }
        let expected = output.shape.get(1).copied().unwrap_or(0);
        if output.shape.len() != 2 {
            return Err(EpError::KernelFailed(
                "cuda_ep NonZero: output must have rank 2".into(),
            ));
        }
        if input.numel() == 0 || expected == 0 || rank == 0 {
            return Ok(());
        }
        let mut strides = vec![1u64; rank];
        for axis in (0..rank.saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(input.shape[axis + 1] as u64)
                .ok_or_else(|| EpError::KernelFailed("NonZero stride overflow".into()))?;
        }
        let bytes: Vec<u8> = strides.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let strides_ptr = self.runtime.alloc_raw(bytes.len())?;
        if let Err(error) = unsafe { self.runtime.htod(&bytes, strides_ptr) } {
            unsafe { self.runtime.free_raw(strides_ptr) }?;
            return Err(error);
        }
        let result = (|| {
            let function = self.runtime.nvrtc_function(
                "nonzero_serial_v1",
                COORD_SOURCE,
                &format!("nonzero_{suffix}"),
            )?;
            let x = cuptr(input.data_ptr::<u8>() as *const c_void);
            let y = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
            let n = input.numel() as u64;
            let rank = rank as u64;
            let expected = expected as u64;
            let mut builder = self.runtime.stream().launch_builder(&function);
            builder
                .arg(&x)
                .arg(&y)
                .arg(&strides_ptr)
                .arg(&n)
                .arg(&rank)
                .arg(&expected);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| driver_err("launch NonZero", error))?;
            self.runtime.synchronize()
        })();
        unsafe { self.runtime.free_raw(strides_ptr) }?;
        result
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "NonZero allocates and uploads per-call coordinate-stride metadata",
        )
    }
}

use std::any::Any;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::random::{Distribution, Generator};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const CAPTURE_REASON: &str = "native random generators advance node-local RNG state on each launch; execute outside CUDA graph capture to avoid replaying the same samples";

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define NXRT_RANDOM_HALF 1
#endif

// Keep the counter permutation and distribution transforms in sync with ep_api::random.
__device__ unsigned long long random_word(unsigned long long key, unsigned long long counter) {
  unsigned long long z = key + counter * 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

__device__ double open_unit(unsigned long long key, unsigned long long counter) {
  return ((double)(random_word(key, counter) >> 12) + 0.5) * (1.0 / 4503599627370496.0);
}

extern "C" __global__ void random_fill(
    void* out, unsigned long long count, unsigned long long key, unsigned long long start,
    int dtype, int normal, double offset, double scale) {
  for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < count; i += (unsigned long long)gridDim.x * blockDim.x) {
    double unit;
    if (normal) {
      unsigned long long counter = start + i * 2;
      unit = sqrt(-2.0 * log(open_unit(key, counter))) *
             cos(6.2831853071795864769 * open_unit(key, counter + 1));
    } else {
      unit = (double)(random_word(key, start + i) >> 11) * (1.0 / 9007199254740992.0);
    }
    double sample = fma(unit, scale, offset);
    switch (dtype) {
      case 1: ((float*)out)[i] = (float)sample; break;
      case 11: ((double*)out)[i] = sample; break;
#ifdef NXRT_RANDOM_HALF
      case 10: ((__half*)out)[i] = __double2half(sample); break;
      case 16: ((__nv_bfloat16*)out)[i] = __double2bfloat16(sample); break;
#endif
    }
  }
}
"#;

pub struct RandomFactory {
    pub runtime: Arc<CudaRuntime>,
    pub version: u64,
}

impl KernelFactory for RandomFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(RandomKernel {
            generator: Generator::from_node(node, self.version)?,
            runtime: self.runtime.clone(),
        }))
    }
}

struct RandomKernel {
    generator: Generator,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for RandomKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(CAPTURE_REASON.into()));
        }
        let count = self.generator.check_output(inputs, outputs)?;
        let output = &mut outputs[0];
        if !output.is_contiguous() {
            return Err(EpError::KernelFailed(
                "CUDA random generator requires a contiguous output; allocate a dense output tensor".into(),
            ));
        }
        if count == 0 {
            return Ok(());
        }
        if matches!(output.dtype, DataType::Float16 | DataType::BFloat16) {
            self.runtime
                .require_nvrtc_half_headers("random generator")?;
        }
        let function = self
            .runtime
            .nvrtc_function("random_fill", SOURCE, "random_fill")?;
        let (key, start) = self.generator.reserve(count)?;
        let (normal, offset, scale) = match self.generator.distribution {
            Distribution::Normal { mean, scale } => (1_i32, mean, scale),
            Distribution::Uniform { low, high } => (0_i32, low, high - low),
        };
        let count = count as u64;
        let dtype = output.dtype as i32;
        let pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let config = LaunchConfig {
            grid_dim: (
                count.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32,
                1,
                1,
            ),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&pointer)
            .arg(&count)
            .arg(&key)
            .arg(&start)
            .arg(&dtype)
            .arg(&normal)
            .arg(&offset)
            .arg(&scale);
        // SAFETY: the dense output holds `count` elements of the validated dtype.
        unsafe { builder.launch(config) }
            .map_err(|error| driver_err("launch native CUDA random generator", error))?;
        Ok(())
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(CAPTURE_REASON)
    }

    fn bind_node_run_state(
        &mut self,
        bind: &mut dyn FnMut(Arc<dyn Any + Send + Sync>) -> Arc<dyn Any + Send + Sync>,
    ) -> Result<()> {
        self.generator.bind_node_run_state(bind)
    }
}

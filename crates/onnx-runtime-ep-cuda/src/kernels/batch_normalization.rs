//! Inference-mode ONNX `BatchNormalization`.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void batch_normalization_f32(
    const float* x, const float* scale, const float* bias,
    const float* mean, const float* variance, float* y,
    unsigned long long n, unsigned long long spatial,
    unsigned long long channels, float epsilon) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long channel = (i / spatial) % channels;
    y[i] = (x[i] - mean[channel]) / sqrtf(variance[channel] + epsilon)
         * scale[channel] + bias[channel];
  }
}

extern "C" __global__ void batch_normalization_f16(
    const __half* x, const __half* scale, const __half* bias,
    const __half* mean, const __half* variance, __half* y,
    unsigned long long n, unsigned long long spatial,
    unsigned long long channels, float epsilon) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long channel = (i / spatial) % channels;
    const float value =
        (__half2float(x[i]) - __half2float(mean[channel]))
        / sqrtf(__half2float(variance[channel]) + epsilon)
        * __half2float(scale[channel]) + __half2float(bias[channel]);
    y[i] = __float2half_rn(value);
  }
}

extern "C" __global__ void batch_normalization_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale,
    const __nv_bfloat16* bias, const __nv_bfloat16* mean,
    const __nv_bfloat16* variance, __nv_bfloat16* y,
    unsigned long long n, unsigned long long spatial,
    unsigned long long channels, float epsilon) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long channel = (i / spatial) % channels;
    const float value =
        (__bfloat162float(x[i]) - __bfloat162float(mean[channel]))
        / sqrtf(__bfloat162float(variance[channel]) + epsilon)
        * __bfloat162float(scale[channel]) + __bfloat162float(bias[channel]);
    y[i] = __float2bfloat16_rn(value);
  }
}
"#;

pub struct BatchNormalizationFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for BatchNormalizationFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let training_mode = node
            .attr("training_mode")
            .and_then(|attribute| attribute.as_int())
            .unwrap_or(0);
        if training_mode != 0 {
            return Err(not_implemented(
                "BatchNormalization training_mode=1 (CUDA EP is inference-only)",
            ));
        }
        Ok(Box::new(BatchNormalizationKernel {
            runtime: self.runtime.clone(),
            epsilon: node
                .attr("epsilon")
                .and_then(|attribute| attribute.as_float())
                .unwrap_or(1e-5),
        }))
    }
}

struct BatchNormalizationKernel {
    runtime: Arc<CudaRuntime>,
    epsilon: f32,
}

impl Kernel for BatchNormalizationKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 5 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep BatchNormalization: expected 5 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        if x.shape.len() < 2 {
            return Err(EpError::KernelFailed(
                "cuda_ep BatchNormalization: X must have rank at least 2".into(),
            ));
        }
        if !matches!(
            x.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(not_implemented(format!(
                "BatchNormalization dtype {:?} (supported: Float32, Float16, BFloat16)",
                x.dtype
            )));
        }
        if inputs
            .iter()
            .any(|input| input.dtype != x.dtype || !input.is_contiguous())
            || outputs[0].dtype != x.dtype
            || !outputs[0].is_contiguous()
        {
            return Err(not_implemented(
                "BatchNormalization requires contiguous, same-dtype tensors",
            ));
        }
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(
                "cuda_ep BatchNormalization: output shape must match X".into(),
            ));
        }
        let channels = x.shape[1];
        let spatial = x.shape[2..].iter().product::<usize>();
        if channels == 0 || spatial == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep BatchNormalization: channel and spatial dimensions must be non-empty"
                    .into(),
            ));
        }
        for (name, input) in [
            ("scale", &inputs[1]),
            ("B", &inputs[2]),
            ("input_mean", &inputs[3]),
            ("input_var", &inputs[4]),
        ] {
            if input.shape != [channels] {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep BatchNormalization: {name} must have shape [{channels}], got {:?}",
                    input.shape
                )));
            }
        }
        let n = x.numel() as u64;
        if n == 0 {
            return Ok(());
        }
        let stem = match x.dtype {
            DataType::Float32 => "batch_normalization_f32",
            DataType::Float16 => "batch_normalization_f16",
            DataType::BFloat16 => "batch_normalization_bf16",
            _ => unreachable!(),
        };
        if x.dtype != DataType::Float32 {
            self.runtime
                .require_nvrtc_half_headers("BatchNormalization")?;
        }
        let function = self
            .runtime
            .nvrtc_function("batch_normalization_v1", SOURCE, stem)?;
        let pointers = inputs
            .iter()
            .map(|input| cuptr(input.data_ptr::<u8>() as *const c_void))
            .collect::<Vec<_>>();
        let output = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let spatial = spatial as u64;
        let channels = channels as u64;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&pointers[0])
            .arg(&pointers[1])
            .arg(&pointers[2])
            .arg(&pointers[3])
            .arg(&pointers[4])
            .arg(&output)
            .arg(&n)
            .arg(&spatial)
            .arg(&channels)
            .arg(&self.epsilon);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (n.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| driver_err("launch BatchNormalization", error))
    }
}

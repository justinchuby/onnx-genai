//! CUDA generators for ONNX `HannWindow`, `HammingWindow`, and `BlackmanWindow`.

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

template <typename Output>
__device__ Output window_store(float value);
template <>
__device__ float window_store<float>(float value) { return value; }
template <>
__device__ __half window_store<__half>(float value) {
  return __float2half_rn(value);
}
template <>
__device__ __nv_bfloat16 window_store<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename Output>
__device__ void window_f32_impl(
    Output* output, unsigned long long elements, int periodic, int kind) {
  const float denominator = periodic ? (float)elements : (float)elements - 1.0f;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += (unsigned long long)gridDim.x * blockDim.x) {
    const float n = (float)index;
    float value;
    if (kind == 0) {
      const float sine = sinf(n * 3.14159265358979323846f / denominator);
      value = sine * sine;
    } else if (kind == 1) {
      const float alpha = 25.0f / 46.0f;
      value = alpha -
          cosf(n * 6.28318530717958647692f / denominator) * (1.0f - alpha);
    } else {
      value = cosf(n * 6.28318530717958647692f / denominator) * -0.5f;
      value += cosf(n * 12.56637061435917295384f / denominator) * 0.08f;
      value += 0.42f;
    }
    output[index] = window_store<Output>(value);
  }
}

extern "C" __global__ void window_f32(
    float* output, unsigned long long elements, int periodic, int kind) {
  window_f32_impl(output, elements, periodic, kind);
}
extern "C" __global__ void window_f16(
    __half* output, unsigned long long elements, int periodic, int kind) {
  window_f32_impl(output, elements, periodic, kind);
}
extern "C" __global__ void window_bf16(
    __nv_bfloat16* output, unsigned long long elements, int periodic, int kind) {
  window_f32_impl(output, elements, periodic, kind);
}
extern "C" __global__ void window_f64(
    double* output, unsigned long long elements, int periodic, int kind) {
  const double denominator = periodic ? (double)elements : (double)elements - 1.0;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += (unsigned long long)gridDim.x * blockDim.x) {
    const double n = (double)index;
    double value;
    if (kind == 0) {
      const double sine = sin(n * 3.14159265358979323846 / denominator);
      value = sine * sine;
    } else if (kind == 1) {
      const double alpha = 25.0 / 46.0;
      value = alpha -
          cos(n * 6.28318530717958647692 / denominator) * (1.0 - alpha);
    } else {
      value = cos(n * 6.28318530717958647692 / denominator) * -0.5;
      value += cos(n * 12.56637061435917295384 / denominator) * 0.08;
      value += 0.42;
    }
    output[index] = value;
  }
}
"#;

#[derive(Clone, Copy)]
pub enum WindowKind {
    Hann,
    Hamming,
    Blackman,
}

impl WindowKind {
    fn name(self) -> &'static str {
        match self {
            Self::Hann => "HannWindow",
            Self::Hamming => "HammingWindow",
            Self::Blackman => "BlackmanWindow",
        }
    }

    fn tag(self) -> i32 {
        match self {
            Self::Hann => 0,
            Self::Hamming => 1,
            Self::Blackman => 2,
        }
    }
}

pub struct WindowFactory {
    pub kind: WindowKind,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for WindowFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let periodic = match node.attr("periodic") {
            None => true,
            Some(attribute) => match attribute.as_int() {
                Some(0) => false,
                Some(1) => true,
                _ => {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {}: periodic must be 0 or 1",
                        self.kind.name()
                    )));
                }
            },
        };
        let output_dtype = match node.attr("output_datatype") {
            None => DataType::Float32,
            Some(attribute) => {
                let value = attribute.as_int().ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "cuda_ep {}: output_datatype must be an integer",
                        self.kind.name()
                    ))
                })?;
                let value = i32::try_from(value).map_err(|_| {
                    EpError::KernelFailed(format!(
                        "cuda_ep {}: invalid output_datatype {value}",
                        self.kind.name()
                    ))
                })?;
                DataType::from_onnx(value).ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "cuda_ep {}: invalid output_datatype {value}",
                        self.kind.name()
                    ))
                })?
            }
        };
        Ok(Box::new(WindowKernel {
            kind: self.kind,
            periodic,
            output_dtype,
            runtime: self.runtime.clone(),
        }))
    }
}

struct WindowKernel {
    kind: WindowKind,
    periodic: bool,
    output_dtype: DataType,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for WindowKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.kind.name();
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 1 input and 1 output"
            )));
        }
        if self.runtime.is_capturing()? {
            return Err(not_implemented(format!(
                "{op} during CUDA graph capture because its scalar input determines output shape"
            )));
        }
        if !inputs[0].is_contiguous()
            || !inputs[0].shape.is_empty()
            || inputs[0].dtype != DataType::Int64
            || !outputs[0].is_contiguous()
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: size must be a contiguous Int64 scalar and output must be contiguous"
            )));
        }
        if !matches!(
            self.output_dtype,
            DataType::Float16 | DataType::BFloat16 | DataType::Float32 | DataType::Float64
        ) {
            return Err(not_implemented(format!(
                "{op} for output dtype {:?}",
                self.output_dtype
            )));
        }
        if outputs[0].dtype != self.output_dtype || outputs[0].shape.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output must be a vector with dtype {:?}",
                self.output_dtype
            )));
        }
        let mut size_bytes = [0_u8; 8];
        unsafe {
            self.runtime.dtoh(
                &mut size_bytes,
                cuptr(inputs[0].data_ptr::<u8>() as *const c_void),
            )?
        };
        let size = usize::try_from(i64::from_ne_bytes(size_bytes)).map_err(|_| {
            EpError::KernelFailed(format!("cuda_ep {op}: size must be non-negative"))
        })?;
        if outputs[0].shape != [size] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape must be [{size}]"
            )));
        }
        if size == 0 {
            return Ok(());
        }
        if matches!(self.output_dtype, DataType::Float16 | DataType::BFloat16) {
            self.runtime.require_nvrtc_half_headers(op)?;
        }
        let entry = match self.output_dtype {
            DataType::Float16 => "window_f16",
            DataType::BFloat16 => "window_bf16",
            DataType::Float32 => "window_f32",
            DataType::Float64 => "window_f64",
            _ => unreachable!("validated above"),
        };
        let function = self.runtime.nvrtc_function("window_ops", SOURCE, entry)?;
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let elements = size as u64;
        let periodic = i32::from(self.periodic);
        let kind = self.kind.tag();
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&output_ptr)
            .arg(&elements)
            .arg(&periodic)
            .arg(&kind);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err(&format!("launch {op}"), error))?;
        Ok(())
    }
}

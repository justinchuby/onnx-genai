//! CUDA implementation of ONNX `Range`.

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

extern "C" __global__ void range_f32(
    const float* start, const float* delta, float* output,
    unsigned long long elements) {
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += (unsigned long long)gridDim.x * blockDim.x)
    output[index] = *start + (float)index * *delta;
}
extern "C" __global__ void range_f16(
    const __half* start, const __half* delta, __half* output,
    unsigned long long elements) {
  const float start_value = __half2float(*start);
  const float delta_value = __half2float(*delta);
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += (unsigned long long)gridDim.x * blockDim.x)
    output[index] = __float2half_rn(start_value + (float)index * delta_value);
}
extern "C" __global__ void range_bf16(
    const __nv_bfloat16* start, const __nv_bfloat16* delta,
    __nv_bfloat16* output, unsigned long long elements) {
  const float start_value = __bfloat162float(*start);
  const float delta_value = __bfloat162float(*delta);
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += (unsigned long long)gridDim.x * blockDim.x)
    output[index] = __float2bfloat16_rn(start_value + (float)index * delta_value);
}
extern "C" __global__ void range_i64(
    const long long* start, const long long* delta, long long* output,
    unsigned long long elements) {
  const unsigned long long start_value = (unsigned long long)*start;
  const unsigned long long delta_value = (unsigned long long)*delta;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += (unsigned long long)gridDim.x * blockDim.x)
    output[index] = (long long)(start_value + index * delta_value);
}
"#;

pub struct RangeFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for RangeFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(RangeKernel {
            runtime: self.runtime.clone(),
        }))
    }
}

struct RangeKernel {
    runtime: Arc<CudaRuntime>,
}

fn scalar_bytes(runtime: &CudaRuntime, input: &TensorView) -> Result<Vec<u8>> {
    let mut bytes = vec![0; input.dtype.byte_size()];
    unsafe { runtime.dtoh(&mut bytes, cuptr(input.data_ptr::<u8>() as *const c_void))? };
    Ok(bytes)
}

fn float_count(start: f32, limit: f32, delta: f32) -> Result<usize> {
    if delta == 0.0 {
        return Err(EpError::KernelFailed(
            "cuda_ep Range: delta must not be zero".into(),
        ));
    }
    let count = ((limit - start) / delta).ceil().max(0.0);
    if !count.is_finite() || count >= usize::MAX as f32 {
        return Err(EpError::KernelFailed(
            "cuda_ep Range: element count exceeds addressable memory".into(),
        ));
    }
    Ok(count as usize)
}

fn int_count(start: i64, limit: i64, delta: i64) -> Result<usize> {
    if delta == 0 {
        return Err(EpError::KernelFailed(
            "cuda_ep Range: delta must not be zero".into(),
        ));
    }
    let distance = limit as i128 - start as i128;
    let step = delta as i128;
    let count = if (distance > 0 && step > 0) || (distance < 0 && step < 0) {
        let distance = distance.unsigned_abs();
        let step = step.unsigned_abs();
        distance.div_ceil(step)
    } else {
        0
    };
    usize::try_from(count).map_err(|_| {
        EpError::KernelFailed("cuda_ep Range: element count exceeds addressable memory".into())
    })
}

impl Kernel for RangeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Range: expected 3 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        if self.runtime.is_capturing()? {
            return Err(not_implemented(
                "Range during CUDA graph capture because its scalar inputs determine output shape",
            ));
        }
        // ONNX `Range` declares scalar start/limit/delta. Real exports (e.g. the
        // Qwen mrope rotary range) emit these as single-element `[1]` tensors
        // rather than rank-0 scalars; both are semantically scalar, so accept any
        // contiguous single-element input (the launch reads the first element).
        if inputs.iter().any(|input| {
            !input.is_contiguous() || input.numel() != 1 || input.dtype != inputs[0].dtype
        }) || !outputs[0].is_contiguous()
            || outputs[0].dtype != inputs[0].dtype
            || outputs[0].shape.len() != 1
        {
            return Err(EpError::KernelFailed(
                "cuda_ep Range: inputs must be same-dtype contiguous scalars (rank-0 or single-element) and output a matching vector"
                    .into(),
            ));
        }
        let dtype = inputs[0].dtype;
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16 | DataType::Int64
        ) {
            return Err(not_implemented(format!("Range for dtype {dtype:?}")));
        }
        let values = inputs
            .iter()
            .map(|input| scalar_bytes(&self.runtime, input))
            .collect::<Result<Vec<_>>>()?;
        let expected = match dtype {
            DataType::Float32 => float_count(
                f32::from_ne_bytes(values[0][..4].try_into().unwrap()),
                f32::from_ne_bytes(values[1][..4].try_into().unwrap()),
                f32::from_ne_bytes(values[2][..4].try_into().unwrap()),
            )?,
            DataType::Float16 => float_count(
                half::f16::from_bits(u16::from_ne_bytes(values[0][..2].try_into().unwrap()))
                    .to_f32(),
                half::f16::from_bits(u16::from_ne_bytes(values[1][..2].try_into().unwrap()))
                    .to_f32(),
                half::f16::from_bits(u16::from_ne_bytes(values[2][..2].try_into().unwrap()))
                    .to_f32(),
            )?,
            DataType::BFloat16 => float_count(
                half::bf16::from_bits(u16::from_ne_bytes(values[0][..2].try_into().unwrap()))
                    .to_f32(),
                half::bf16::from_bits(u16::from_ne_bytes(values[1][..2].try_into().unwrap()))
                    .to_f32(),
                half::bf16::from_bits(u16::from_ne_bytes(values[2][..2].try_into().unwrap()))
                    .to_f32(),
            )?,
            DataType::Int64 => int_count(
                i64::from_ne_bytes(values[0][..8].try_into().unwrap()),
                i64::from_ne_bytes(values[1][..8].try_into().unwrap()),
                i64::from_ne_bytes(values[2][..8].try_into().unwrap()),
            )?,
            _ => unreachable!(),
        };
        if outputs[0].numel() != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Range: output has {} elements, expected {expected}",
                outputs[0].numel()
            )));
        }
        if expected == 0 {
            return Ok(());
        }
        let entry = match dtype {
            DataType::Float32 => "range_f32",
            DataType::Float16 => "range_f16",
            DataType::BFloat16 => "range_bf16",
            DataType::Int64 => "range_i64",
            _ => unreachable!(),
        };
        let function = self.runtime.nvrtc_function("range", SOURCE, entry)?;
        let start_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let delta_ptr = cuptr(inputs[2].data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let elements = expected as u64;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&start_ptr)
            .arg(&delta_ptr)
            .arg(&output_ptr)
            .arg(&elements);
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
        .map_err(|error| driver_err("launch Range", error))?;
        Ok(())
    }
}

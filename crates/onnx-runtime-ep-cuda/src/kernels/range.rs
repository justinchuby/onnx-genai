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
extern "C" __global__ void range_i32(
    const int* start, const int* delta, int* output,
    unsigned long long elements) {
  const unsigned int start_value = (unsigned int)*start;
  const unsigned int delta_value = (unsigned int)*delta;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += (unsigned long long)gridDim.x * blockDim.x)
    output[index] = (int)(start_value + (unsigned int)index * delta_value);
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
            DataType::Float32
                | DataType::Float16
                | DataType::BFloat16
                | DataType::Int32
                | DataType::Int64
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
            DataType::Int32 => int_count(
                i32::from_ne_bytes(values[0][..4].try_into().unwrap()).into(),
                i32::from_ne_bytes(values[1][..4].try_into().unwrap()).into(),
                i32::from_ne_bytes(values[2][..4].try_into().unwrap()).into(),
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
            DataType::Int32 => "range_i32",
            _ => unreachable!(),
        };
        let function = self
            .runtime
            .nvrtc_function("range_typed_v2", SOURCE, entry)?;
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

#[cfg(test)]
mod claim_probes {
    use std::ffi::c_void;
    use std::sync::Arc;

    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, Kernel, TensorMut, TensorView};
    use onnx_runtime_ir::{DataType, DeviceId};

    use super::RangeKernel;
    use crate::runtime::CudaRuntime;

    fn maybe_runtime() -> Option<Arc<CudaRuntime>> {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let rt = std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten();
        std::panic::set_hook(previous);
        rt
    }

    #[test]
    fn i32_range_matches_reference_on_gpu() {
        let Some(runtime) = maybe_runtime() else {
            eprintln!("skipping i32 Range GPU probe: CUDA runtime unavailable");
            return;
        };
        let start = [2i32];
        let limit = [14i32];
        let delta = [3i32];
        let expected = [2i32, 5, 8, 11];

        let scalar_bytes = std::mem::size_of::<i32>();
        let start_dev = runtime.alloc_raw(scalar_bytes).unwrap();
        let limit_dev = runtime.alloc_raw(scalar_bytes).unwrap();
        let delta_dev = runtime.alloc_raw(scalar_bytes).unwrap();
        let out_dev = runtime.alloc_raw(scalar_bytes * expected.len()).unwrap();
        let as_bytes = |v: &[i32]| unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
        };
        unsafe {
            runtime.htod(as_bytes(&start), start_dev).unwrap();
            runtime.htod(as_bytes(&limit), limit_dev).unwrap();
            runtime.htod(as_bytes(&delta), delta_dev).unwrap();
        }
        let device = DeviceId::cuda(0);
        let scalar_shape = [1usize];
        let scalar_strides = [1i64];
        let mk = |ptr: u64| {
            TensorView::new(
                DevicePtr(ptr as usize as *const c_void),
                DataType::Int32,
                &scalar_shape,
                &scalar_strides,
                device,
            )
        };
        let inputs = [mk(start_dev), mk(limit_dev), mk(delta_dev)];
        let out_shape = [expected.len()];
        let out_strides = [1i64];
        let mut outputs = [TensorMut::new(
            DevicePtrMut(out_dev as usize as *mut c_void),
            DataType::Int32,
            &out_shape,
            &out_strides,
            device,
        )];
        RangeKernel {
            runtime: runtime.clone(),
        }
        .execute(&inputs, &mut outputs)
        .unwrap();
        runtime.synchronize().unwrap();
        let mut out = vec![0i32; expected.len()];
        let out_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                out.as_mut_ptr().cast::<u8>(),
                scalar_bytes * expected.len(),
            )
        };
        unsafe { runtime.dtoh(out_bytes, out_dev).unwrap() };
        unsafe {
            runtime.free_raw(start_dev).unwrap();
            runtime.free_raw(limit_dev).unwrap();
            runtime.free_raw(delta_dev).unwrap();
            runtime.free_raw(out_dev).unwrap();
        }
        assert_eq!(out, expected, "i32 Range diverged on GPU");
    }
}

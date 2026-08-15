//! Low-complexity CUDA data transforms: `Compress` and `AffineGrid`.

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

extern "C" __global__ void compress_bytes(
    const unsigned char* input, const unsigned char* condition,
    unsigned char* output, unsigned long long output_elements,
    unsigned long long axis_length, unsigned long long selected_length,
    unsigned long long condition_length, unsigned long long inner,
    unsigned long long element_size) {
  for (unsigned long long output_index =
           blockIdx.x * blockDim.x + threadIdx.x;
       output_index < output_elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long inner_index = output_index % inner;
    const unsigned long long output_axis =
        (output_index / inner) % selected_length;
    const unsigned long long outer_index =
        output_index / (inner * selected_length);
    unsigned long long seen = 0;
    unsigned long long input_axis = axis_length;
    for (unsigned long long candidate = 0;
         candidate < axis_length && candidate < condition_length; ++candidate) {
      if (condition[candidate]) {
        if (seen == output_axis) {
          input_axis = candidate;
          break;
        }
        ++seen;
      }
    }
    if (input_axis == axis_length) return;
    const unsigned long long input_index =
        (outer_index * axis_length + input_axis) * inner + inner_index;
    for (unsigned long long byte = 0; byte < element_size; ++byte)
      output[output_index * element_size + byte] =
          input[input_index * element_size + byte];
  }
}

__device__ __forceinline__ float load_grid_value(
    const void* values, int dtype, unsigned long long index) {
  if (dtype == 0) return ((const float*)values)[index];
  if (dtype == 1) return __half2float(((const __half*)values)[index]);
  return __bfloat162float(((const __nv_bfloat16*)values)[index]);
}

__device__ __forceinline__ void store_grid_value(
    void* values, int dtype, unsigned long long index, float value) {
  if (dtype == 0) ((float*)values)[index] = value;
  else if (dtype == 1) ((__half*)values)[index] = __float2half_rn(value);
  else ((__nv_bfloat16*)values)[index] = __float2bfloat16_rn(value);
}

__device__ __forceinline__ float grid_coordinate(
    unsigned long long index, unsigned long long extent, int align_corners) {
  if (align_corners)
    return extent <= 1 ? 0.0f
                       : 2.0f * (float)index / (float)(extent - 1) - 1.0f;
  return (2.0f * (float)index + 1.0f) / (float)extent - 1.0f;
}

extern "C" __global__ void affine_grid_2d(
    const void* theta, void* output, unsigned long long points,
    unsigned long long height, unsigned long long width,
    int dtype, int align_corners) {
  for (unsigned long long point = blockIdx.x * blockDim.x + threadIdx.x;
       point < points; point += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long batch_index = point / (height * width);
    const unsigned long long spatial = point % (height * width);
    const float y = grid_coordinate(spatial / width, height, align_corners);
    const float x = grid_coordinate(spatial % width, width, align_corners);
    const unsigned long long theta_base = batch_index * 6;
    store_grid_value(output, dtype, point * 2,
        load_grid_value(theta, dtype, theta_base) * x
      + load_grid_value(theta, dtype, theta_base + 1) * y
      + load_grid_value(theta, dtype, theta_base + 2));
    store_grid_value(output, dtype, point * 2 + 1,
        load_grid_value(theta, dtype, theta_base + 3) * x
      + load_grid_value(theta, dtype, theta_base + 4) * y
      + load_grid_value(theta, dtype, theta_base + 5));
  }
}

extern "C" __global__ void affine_grid_3d(
    const void* theta, void* output, unsigned long long points,
    unsigned long long depth, unsigned long long height,
    unsigned long long width, int dtype, int align_corners) {
  for (unsigned long long point = blockIdx.x * blockDim.x + threadIdx.x;
       point < points; point += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long spatial_size = depth * height * width;
    const unsigned long long batch_index = point / spatial_size;
    const unsigned long long spatial = point % spatial_size;
    const unsigned long long plane = height * width;
    const float z = grid_coordinate(spatial / plane, depth, align_corners);
    const unsigned long long within_plane = spatial % plane;
    const float y = grid_coordinate(within_plane / width, height, align_corners);
    const float x = grid_coordinate(within_plane % width, width, align_corners);
    const unsigned long long theta_base = batch_index * 12;
    for (unsigned long long component = 0; component < 3; ++component) {
      const unsigned long long base = theta_base + component * 4;
      store_grid_value(output, dtype, point * 3 + component,
          load_grid_value(theta, dtype, base) * x
        + load_grid_value(theta, dtype, base + 1) * y
        + load_grid_value(theta, dtype, base + 2) * z
        + load_grid_value(theta, dtype, base + 3));
    }
  }
}
"#;

pub struct CompressFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for CompressFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(CompressKernel {
            axis: node.attr("axis").and_then(|attribute| attribute.as_int()),
            runtime: self.runtime.clone(),
        }))
    }
}

struct CompressKernel {
    axis: Option<i64>,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for CompressKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep Compress: expected 2 inputs and 1 output".into(),
            ));
        }
        let input = &inputs[0];
        let condition = &inputs[1];
        let output = &mut outputs[0];
        if condition.dtype != DataType::Bool || condition.shape.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep Compress: condition must be a one-dimensional Bool tensor".into(),
            ));
        }
        if input.dtype != output.dtype
            || !input.is_contiguous()
            || !condition.is_contiguous()
            || !output.is_contiguous()
        {
            return Err(not_implemented(
                "Compress requires contiguous tensors and matching data/output dtypes",
            ));
        }
        let element_size = input.dtype.byte_size();
        if element_size == 0 {
            return Err(not_implemented(format!(
                "Compress dtype {:?} has no fixed-width storage",
                input.dtype
            )));
        }
        let (shape, axis) = match self.axis {
            Some(raw_axis) => {
                let rank = input.shape.len();
                let axis = if raw_axis < 0 {
                    raw_axis + rank as i64
                } else {
                    raw_axis
                };
                if axis < 0 || axis as usize >= rank {
                    return Err(EpError::KernelFailed(
                        "cuda_ep Compress: axis out of range".into(),
                    ));
                }
                (input.shape.to_vec(), axis as usize)
            }
            None => (vec![input.numel()], 0),
        };
        if output.shape.len() != shape.len()
            || output
                .shape
                .iter()
                .enumerate()
                .any(|(index, dimension)| index != axis && *dimension != shape[index])
        {
            return Err(EpError::KernelFailed(
                "cuda_ep Compress: output shape does not preserve non-selected dimensions".into(),
            ));
        }
        let selected_length = output.shape[axis];
        let output_elements = output.numel() as u64;
        if output_elements == 0 {
            return Ok(());
        }
        let axis_length = shape[axis] as u64;
        let selected_length = selected_length as u64;
        let condition_length = condition.numel() as u64;
        let inner = shape[axis + 1..].iter().product::<usize>() as u64;
        let element_size = element_size as u64;
        let function =
            self.runtime
                .nvrtc_function("data_transform_v1", SOURCE, "compress_bytes")?;
        let input_pointer = cuptr(input.data_ptr::<u8>() as *const c_void);
        let condition_pointer = cuptr(condition.data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_pointer)
            .arg(&condition_pointer)
            .arg(&output_pointer)
            .arg(&output_elements)
            .arg(&axis_length)
            .arg(&selected_length)
            .arg(&condition_length)
            .arg(&inner)
            .arg(&element_size);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    output_elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| driver_err("launch Compress", error))
    }
}

pub struct AffineGridFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for AffineGridFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(AffineGridKernel {
            align_corners: node
                .attr("align_corners")
                .and_then(|attribute| attribute.as_int())
                .unwrap_or(0)
                != 0,
            runtime: self.runtime.clone(),
        }))
    }
}

struct AffineGridKernel {
    align_corners: bool,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for AffineGridKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep AffineGrid: expected 2 inputs and 1 output".into(),
            ));
        }
        let theta = &inputs[0];
        let size = &inputs[1];
        let output = &mut outputs[0];
        let spatial_rank = match theta.shape {
            [_, 2, 3] => 2,
            [_, 3, 4] => 3,
            _ => {
                return Err(EpError::KernelFailed(
                    "cuda_ep AffineGrid: theta must have shape [N,2,3] or [N,3,4]".into(),
                ));
            }
        };
        if size.dtype != DataType::Int64
            || size.shape != [spatial_rank + 2]
            || theta.dtype != output.dtype
            || !theta.is_contiguous()
            || !size.is_contiguous()
            || !output.is_contiguous()
        {
            return Err(not_implemented(
                "AffineGrid requires contiguous theta/size/output, Int64 size, and matching float dtypes",
            ));
        }
        let dtype = match theta.dtype {
            DataType::Float32 => 0i32,
            DataType::Float16 => 1,
            DataType::BFloat16 => 2,
            other => {
                return Err(not_implemented(format!(
                    "AffineGrid theta dtype {other:?} (supported: Float32, Float16, BFloat16)"
                )));
            }
        };
        if dtype != 0 {
            self.runtime.require_nvrtc_half_headers("AffineGrid")?;
        }
        let batch = theta.shape[0];
        let expected = if spatial_rank == 2 {
            output.shape.len() == 4 && output.shape[0] == batch && output.shape[3] == 2
        } else {
            output.shape.len() == 5 && output.shape[0] == batch && output.shape[4] == 3
        };
        if !expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep AffineGrid: invalid output shape {:?} for theta {:?}",
                output.shape, theta.shape
            )));
        }
        let (depth, height, width, points, stem) = if spatial_rank == 2 {
            (
                1u64,
                output.shape[1] as u64,
                output.shape[2] as u64,
                batch
                    .saturating_mul(output.shape[1])
                    .saturating_mul(output.shape[2]) as u64,
                "affine_grid_2d",
            )
        } else {
            (
                output.shape[1] as u64,
                output.shape[2] as u64,
                output.shape[3] as u64,
                batch
                    .saturating_mul(output.shape[1])
                    .saturating_mul(output.shape[2])
                    .saturating_mul(output.shape[3]) as u64,
                "affine_grid_3d",
            )
        };
        if points == 0 {
            return Ok(());
        }
        let align_corners = i32::from(self.align_corners);
        let function = self
            .runtime
            .nvrtc_function("data_transform_v1", SOURCE, stem)?;
        let theta_pointer = cuptr(theta.data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&theta_pointer)
            .arg(&output_pointer)
            .arg(&points);
        if spatial_rank == 3 {
            builder.arg(&depth);
        }
        builder
            .arg(&height)
            .arg(&width)
            .arg(&dtype)
            .arg(&align_corners);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (points.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| driver_err("launch AffineGrid", error))
    }
}

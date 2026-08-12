//! Four-dimensional ONNX `GridSample` using a common NVRTC sampling kernel.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_value(
    const void* values, int dtype, unsigned long long index) {
  if (dtype == 0) return ((const float*)values)[index];
  if (dtype == 1) return __half2float(((const __half*)values)[index]);
  return __bfloat162float(((const __nv_bfloat16*)values)[index]);
}
__device__ __forceinline__ void store_value(
    void* values, int dtype, unsigned long long index, float value) {
  if (dtype == 0) ((float*)values)[index] = value;
  else if (dtype == 1) ((__half*)values)[index] = __float2half_rn(value);
  else ((__nv_bfloat16*)values)[index] = __float2bfloat16_rn(value);
}
__device__ __forceinline__ float reflect_coordinate(float coordinate, int size, int aligned) {
  if (size <= 1) return 0.0f;
  const float low = aligned ? 0.0f : -0.5f;
  const float high = aligned ? (float)(size - 1) : (float)size - 0.5f;
  const float span = high - low;
  float offset = fmodf(coordinate - low, 2.0f * span);
  if (offset < 0.0f) offset += 2.0f * span;
  const float reflected = offset <= span ? low + offset : high - (offset - span);
  return fminf(fmaxf(reflected, 0.0f), (float)(size - 1));
}
__device__ __forceinline__ float source_coordinate(
    float normalized, int size, int aligned, int padding) {
  float coordinate = aligned
      ? fmaf(normalized, (float)(size - 1) / 2.0f, (float)(size - 1) / 2.0f)
      : fmaf(normalized, (float)size / 2.0f, ((float)size - 1.0f) / 2.0f);
  if (padding == 1) return fminf(fmaxf(coordinate, 0.0f), (float)(size - 1));
  if (padding == 2) return reflect_coordinate(coordinate, size, aligned);
  return coordinate;
}
__device__ __forceinline__ int map_index(int index, int size, int padding, int aligned) {
  if (padding == 0) return index < 0 || index >= size ? -1 : index;
  if (padding == 1) return max(0, min(index, size - 1));
  return (int)nearbyintf(reflect_coordinate((float)index, size, aligned));
}
__device__ __forceinline__ float sample(
    const void* input, int dtype, unsigned long long base, int height, int width,
    int y, int x, int padding, int aligned) {
  y = map_index(y, height, padding, aligned);
  x = map_index(x, width, padding, aligned);
  if (y < 0 || x < 0) return 0.0f;
  return load_value(input, dtype, base + (unsigned long long)y * width + x);
}
extern "C" __global__ void grid_sample(
    const void* input, const void* grid, void* output,
    unsigned long long output_elements, unsigned long long channels,
    int height, int width, int output_height, int output_width,
    int dtype, int mode, int padding, int aligned) {
  const unsigned long long output_spatial =
      (unsigned long long)output_height * output_width;
  const unsigned long long input_spatial = (unsigned long long)height * width;
  for (unsigned long long output_index =
           blockIdx.x * blockDim.x + threadIdx.x;
       output_index < output_elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long batch = output_index / (channels * output_spatial);
    const unsigned long long within_batch = output_index % (channels * output_spatial);
    const unsigned long long channel = within_batch / output_spatial;
    const unsigned long long spatial = within_batch % output_spatial;
    const unsigned long long grid_index = (batch * output_spatial + spatial) * 2;
    const float x = source_coordinate(load_value(grid, dtype, grid_index), width, aligned, padding);
    const float y = source_coordinate(load_value(grid, dtype, grid_index + 1), height, aligned, padding);
    const unsigned long long base = (batch * channels + channel) * input_spatial;
    float value;
    if (mode == 1) {
      value = sample(input, dtype, base, height, width,
                     (int)nearbyintf(y), (int)nearbyintf(x), padding, aligned);
    } else {
      const int x0 = (int)floorf(x);
      const int y0 = (int)floorf(y);
      const float dx = x - (float)x0;
      const float dy = y - (float)y0;
      const float top =
          sample(input, dtype, base, height, width, y0, x0, padding, aligned) * (1.0f - dx)
          + sample(input, dtype, base, height, width, y0, x0 + 1, padding, aligned) * dx;
      const float bottom =
          sample(input, dtype, base, height, width, y0 + 1, x0, padding, aligned) * (1.0f - dx)
          + sample(input, dtype, base, height, width, y0 + 1, x0 + 1, padding, aligned) * dx;
      value = top * (1.0f - dy) + bottom * dy;
    }
    store_value(output, dtype, output_index, value);
  }
}
"#;

#[derive(Clone, Copy)]
enum Mode {
    Linear,
    Nearest,
}

#[derive(Clone, Copy)]
enum Padding {
    Zeros,
    Border,
    Reflection,
}

pub struct GridSampleFactory {
    pub runtime: Arc<CudaRuntime>,
    pub since_version: u32,
}

impl KernelFactory for GridSampleFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        reject_deferred(node, input_shapes)?;
        let mode = match node
            .attr("mode")
            .and_then(Attribute::as_str)
            .unwrap_or("linear")
        {
            "linear" | "bilinear" => Mode::Linear,
            "nearest" => Mode::Nearest,
            other => return Err(not_implemented(format!("GridSample mode {other:?}"))),
        };
        let padding = match node
            .attr("padding_mode")
            .and_then(Attribute::as_str)
            .unwrap_or("zeros")
        {
            "zeros" => Padding::Zeros,
            "border" => Padding::Border,
            "reflection" => Padding::Reflection,
            other => {
                return Err(not_implemented(format!(
                    "GridSample padding_mode {other:?}"
                )));
            }
        };
        let align_corners = node
            .attr("align_corners")
            .and_then(Attribute::as_int)
            .unwrap_or(0);
        if !matches!(align_corners, 0 | 1) {
            return Err(EpError::KernelFailed(
                "cuda_ep GridSample: align_corners must be 0 or 1".into(),
            ));
        }
        Ok(Box::new(GridSampleKernel {
            runtime: self.runtime.clone(),
            mode,
            padding,
            align_corners: align_corners != 0,
            _since_version: self.since_version,
        }))
    }
}

pub(crate) fn reject_deferred(node: &Node, input_shapes: &[Vec<usize>]) -> Result<()> {
    match node
        .attr("mode")
        .and_then(Attribute::as_str)
        .unwrap_or("linear")
    {
        "linear" | "bilinear" | "nearest" => {}
        value => {
            return Err(not_implemented(format!(
                "GridSample mode {value:?} is deferred"
            )));
        }
    }
    if input_shapes.first().is_some_and(|shape| shape.len() != 4) {
        return Err(not_implemented(
            "GridSample volumetric/rank-other-than-4 input is deferred",
        ));
    }
    Ok(())
}

fn dtype_code(dtype: DataType) -> Result<i32> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        other => Err(not_implemented(format!(
            "GridSample dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

struct GridSampleKernel {
    runtime: Arc<CudaRuntime>,
    mode: Mode,
    padding: Padding,
    align_corners: bool,
    _since_version: u32,
}

impl Kernel for GridSampleKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep GridSample: expected X, grid, and one output".into(),
            ));
        }
        let input = &inputs[0];
        let grid = &inputs[1];
        if input.shape.len() != 4 {
            return Err(not_implemented(
                "GridSample volumetric/rank-other-than-4 input is deferred",
            ));
        }
        if grid.shape.len() != 4
            || grid.shape[0] != input.shape[0]
            || grid.shape[3] != 2
            || input.shape[2..].contains(&0)
        {
            return Err(EpError::KernelFailed(
                "cuda_ep GridSample: expected X [N,C,H,W] and grid [N,Hout,Wout,2]".into(),
            ));
        }
        if input.dtype != grid.dtype || outputs[0].dtype != input.dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep GridSample: X, grid, and output dtypes must match".into(),
            ));
        }
        if !input.is_contiguous() || !grid.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented("GridSample with strided tensors"));
        }
        let expected = [input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]];
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GridSample: output shape {:?}, expected {expected:?}",
                outputs[0].shape
            )));
        }
        let dtype = dtype_code(input.dtype)?;
        if dtype != 0 {
            self.runtime.require_nvrtc_half_headers("GridSample")?;
        }
        let output_elements = outputs[0].numel() as u64;
        if output_elements == 0 {
            return Ok(());
        }
        let function = self
            .runtime
            .nvrtc_function("grid_sample_2d_v1", SOURCE, "grid_sample")?;
        let input_pointer = cuptr(input.data_ptr::<u8>() as *const c_void);
        let grid_pointer = cuptr(grid.data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let channels = input.shape[1] as u64;
        let height = input.shape[2] as i32;
        let width = input.shape[3] as i32;
        let output_height = grid.shape[1] as i32;
        let output_width = grid.shape[2] as i32;
        let mode = match self.mode {
            Mode::Linear => 0i32,
            Mode::Nearest => 1,
        };
        let padding = match self.padding {
            Padding::Zeros => 0i32,
            Padding::Border => 1,
            Padding::Reflection => 2,
        };
        let aligned = i32::from(self.align_corners);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_pointer)
            .arg(&grid_pointer)
            .arg(&output_pointer)
            .arg(&output_elements)
            .arg(&channels)
            .arg(&height)
            .arg(&width)
            .arg(&output_height)
            .arg(&output_width)
            .arg(&dtype)
            .arg(&mode)
            .arg(&padding)
            .arg(&aligned);
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
        .map_err(|error| driver_err("launch GridSample", error))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{NodeId, ValueId};

    fn node(mode: &str) -> Node {
        let mut node = Node::new(
            NodeId(0),
            "GridSample",
            vec![Some(ValueId(0)), Some(ValueId(1))],
            vec![ValueId(2)],
        );
        node.attributes
            .insert("mode".into(), Attribute::String(mode.as_bytes().to_vec()));
        node
    }

    #[test]
    fn cubic_and_volumetric_are_rejected_by_claim_and_factory_contract() {
        for mode in ["cubic", "bicubic"] {
            let deferred = node(mode);
            assert!(
                crate::kernels::standard_claims::unsupported_reason(
                    &deferred,
                    &[vec![1.into(), 1.into(), 2.into(), 2.into()]],
                    &[DataType::Float32, DataType::Float32],
                )
                .is_some()
            );
            assert!(reject_deferred(&deferred, &[vec![1, 1, 2, 2]]).is_err());
        }
        let volumetric = node("linear");
        let reason = crate::kernels::standard_claims::unsupported_reason(
            &volumetric,
            &[
                vec![1.into(), 1.into(), 2.into(), 2.into(), 2.into()],
                vec![1.into(), 2.into(), 2.into(), 2.into(), 3.into()],
            ],
            &[DataType::Float32, DataType::Float32],
        )
        .expect("rank-5 GridSample must be declined before CUDA placement");
        assert!(reason.contains("rank 5 unsupported"), "{reason}");
        assert!(reject_deferred(&volumetric, &[vec![1, 1, 2, 2, 2]]).is_err());
    }
}

//! Common ONNX `Resize` modes implemented as one N-D NVRTC interpolation kernel.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node, compute_contiguous_strides};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_resize_value(
    const void* values, int dtype, unsigned long long index) {
  if (dtype == 0) return ((const float*)values)[index];
  if (dtype == 1) return __half2float(((const __half*)values)[index]);
  return __bfloat162float(((const __nv_bfloat16*)values)[index]);
}

__device__ __forceinline__ void store_resize_value(
    void* values, int dtype, unsigned long long index, float value) {
  if (dtype == 0) ((float*)values)[index] = value;
  else if (dtype == 1) ((__half*)values)[index] = __float2half_rn(value);
  else ((__nv_bfloat16*)values)[index] = __float2bfloat16_rn(value);
}

__device__ __forceinline__ double source_coordinate(
    unsigned long long output_index, unsigned long long input_size,
    unsigned long long output_size, double scale, int coordinate_mode) {
  const double x = (double)output_index;
  if (coordinate_mode == 0) return (x + 0.5) / scale - 0.5;
  if (coordinate_mode == 1)
    return output_size <= 1 ? 0.0
                            : x * (double)(input_size - 1) / (double)(output_size - 1);
  return x / scale;
}

__device__ __forceinline__ unsigned long long nearest_index(
    double coordinate, unsigned long long size, int nearest_mode) {
  const double lower = floor(coordinate);
  const double fraction = coordinate - lower;
  double selected;
  if (nearest_mode == 0) selected = fraction <= 0.5 ? lower : lower + 1.0;
  else if (nearest_mode == 1) selected = fraction < 0.5 ? lower : lower + 1.0;
  else if (nearest_mode == 2) selected = lower;
  else selected = ceil(coordinate);
  selected = fmax(0.0, fmin(selected, (double)(size - 1)));
  return (unsigned long long)selected;
}

extern "C" __global__ void resize_nd(
    const void* input, void* output, const unsigned long long* integer_metadata,
    const double* scales, unsigned long long output_elements, int rank, int dtype,
    int interpolation_mode, int coordinate_mode, int nearest_mode) {
  const unsigned long long* input_dimensions = integer_metadata;
  const unsigned long long* output_dimensions = integer_metadata + rank;
  const unsigned long long* input_strides = integer_metadata + rank * 2;
  const unsigned long long* output_strides = integer_metadata + rank * 3;
  for (unsigned long long output_linear =
           blockIdx.x * blockDim.x + threadIdx.x;
       output_linear < output_elements;
       output_linear += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long remaining = output_linear;
    unsigned long long lower_offset = 0;
    unsigned long long upper_delta[16];
    double upper_weight[16];
    for (int axis = 0; axis < rank; ++axis) {
      const unsigned long long output_coordinate =
          remaining / output_strides[axis];
      remaining %= output_strides[axis];
      const double coordinate = source_coordinate(
          output_coordinate, input_dimensions[axis], output_dimensions[axis],
          scales[axis], coordinate_mode);
      if (interpolation_mode == 0) {
        lower_offset += nearest_index(coordinate, input_dimensions[axis], nearest_mode)
                        * input_strides[axis];
        upper_delta[axis] = 0;
        upper_weight[axis] = 0.0;
      } else {
        const long long lower_raw = (long long)floor(coordinate);
        const double fraction = coordinate - (double)lower_raw;
        const long long upper_raw = lower_raw + 1;
        const unsigned long long lower =
            (unsigned long long)max(0LL, min(lower_raw, (long long)input_dimensions[axis] - 1));
        const unsigned long long upper =
            (unsigned long long)max(0LL, min(upper_raw, (long long)input_dimensions[axis] - 1));
        lower_offset += lower * input_strides[axis];
        upper_delta[axis] = (upper - lower) * input_strides[axis];
        upper_weight[axis] = fraction;
      }
    }
    if (interpolation_mode == 0) {
      store_resize_value(output, dtype, output_linear,
                         load_resize_value(input, dtype, lower_offset));
      continue;
    }
    double sum = 0.0;
    const unsigned long long combinations = 1ULL << rank;
    for (unsigned long long combination = 0; combination < combinations; ++combination) {
      unsigned long long input_offset = lower_offset;
      double weight = 1.0;
      for (int axis = 0; axis < rank; ++axis) {
        if ((combination >> axis) & 1ULL) {
          input_offset += upper_delta[axis];
          weight *= upper_weight[axis];
        } else {
          weight *= 1.0 - upper_weight[axis];
        }
      }
      sum += (double)load_resize_value(input, dtype, input_offset) * weight;
    }
    store_resize_value(output, dtype, output_linear, (float)sum);
  }
}
"#;

#[derive(Clone, Copy)]
enum InterpolationMode {
    Nearest,
    Linear,
}

#[derive(Clone, Copy)]
enum CoordinateMode {
    HalfPixel,
    AlignCorners,
    Asymmetric,
}

#[derive(Clone, Copy)]
enum NearestMode {
    RoundPreferFloor,
    RoundPreferCeil,
    Floor,
    Ceil,
}

pub struct ResizeFactory {
    pub runtime: Arc<CudaRuntime>,
    pub since_version: u32,
}

impl KernelFactory for ResizeFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let mode = match node
            .attr("mode")
            .and_then(Attribute::as_str)
            .unwrap_or("nearest")
        {
            "nearest" => InterpolationMode::Nearest,
            "linear" => InterpolationMode::Linear,
            other => {
                return Err(not_implemented(format!(
                    "Resize mode {other:?} (supported: nearest, linear)"
                )));
            }
        };
        let coordinate_default = if self.since_version == 10 {
            "asymmetric"
        } else {
            "half_pixel"
        };
        let coordinate_mode = match node
            .attr("coordinate_transformation_mode")
            .and_then(Attribute::as_str)
            .unwrap_or(coordinate_default)
        {
            "half_pixel" => CoordinateMode::HalfPixel,
            "align_corners" => CoordinateMode::AlignCorners,
            "asymmetric" => CoordinateMode::Asymmetric,
            other => {
                return Err(not_implemented(format!(
                    "Resize coordinate_transformation_mode {other:?} (supported: half_pixel, align_corners, asymmetric)"
                )));
            }
        };
        let nearest_default = if self.since_version == 10 {
            "floor"
        } else {
            "round_prefer_floor"
        };
        let nearest_mode = match node
            .attr("nearest_mode")
            .and_then(Attribute::as_str)
            .unwrap_or(nearest_default)
        {
            "round_prefer_floor" => NearestMode::RoundPreferFloor,
            "round_prefer_ceil" => NearestMode::RoundPreferCeil,
            "floor" => NearestMode::Floor,
            "ceil" => NearestMode::Ceil,
            other => {
                return Err(not_implemented(format!("Resize nearest_mode {other:?}")));
            }
        };
        Ok(Box::new(ResizeKernel {
            runtime: self.runtime.clone(),
            since_version: self.since_version,
            mode,
            coordinate_mode,
            nearest_mode,
            axes: node
                .attr("axes")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec),
        }))
    }
}

struct ResizeKernel {
    runtime: Arc<CudaRuntime>,
    since_version: u32,
    mode: InterpolationMode,
    coordinate_mode: CoordinateMode,
    nearest_mode: NearestMode,
    axes: Option<Vec<i64>>,
}

fn device_bytes(runtime: &CudaRuntime, input: &TensorView) -> Result<Vec<u8>> {
    if !input.is_contiguous() {
        return Err(not_implemented("Resize with strided control input"));
    }
    let mut bytes = vec![0; input.dtype.storage_bytes(input.numel())];
    if !bytes.is_empty() {
        unsafe { runtime.dtoh(&mut bytes, cuptr(input.data_ptr::<u8>() as *const c_void))? };
    }
    Ok(bytes)
}

fn device_f32(runtime: &CudaRuntime, input: &TensorView, name: &str) -> Result<Vec<f32>> {
    if input.dtype != DataType::Float32 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Resize: {name} must be Float32"
        )));
    }
    Ok(device_bytes(runtime, input)?
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

fn device_i64(runtime: &CudaRuntime, input: &TensorView, name: &str) -> Result<Vec<i64>> {
    if input.dtype != DataType::Int64 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Resize: {name} must be Int64"
        )));
    }
    Ok(device_bytes(runtime, input)?
        .chunks_exact(8)
        .map(|bytes| i64::from_ne_bytes(bytes.try_into().expect("eight-byte chunk")))
        .collect())
}

fn upload_bytes(runtime: &CudaRuntime, bytes: &[u8]) -> Result<CUdeviceptr> {
    let pointer = runtime.alloc_raw(bytes.len())?;
    if let Err(error) = unsafe { runtime.htod(bytes, pointer) } {
        let _ = unsafe { runtime.free_raw(pointer) };
        return Err(error);
    }
    Ok(pointer)
}

fn dtype_code(dtype: DataType) -> Result<i32> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        other => Err(not_implemented(format!(
            "Resize dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

impl Kernel for ResizeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if outputs.len() != 1
            || (self.since_version == 10 && inputs.len() != 2)
            || (self.since_version >= 11 && !(1..=4).contains(&inputs.len()))
        {
            return Err(EpError::KernelFailed(
                "cuda_ep Resize: invalid input/output arity".into(),
            ));
        }
        if self.runtime.is_capturing()? {
            return Err(not_implemented(
                "Resize during CUDA graph capture because scales/sizes are device inputs",
            ));
        }
        let input = &inputs[0];
        let output = &mut outputs[0];
        if input.dtype != output.dtype || !input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(
                "Resize requires contiguous input/output with matching dtypes",
            ));
        }
        if input.shape.contains(&0) {
            return Err(not_implemented("Resize with an empty input dimension"));
        }
        let dtype = dtype_code(input.dtype)?;
        if dtype != 0 {
            self.runtime.require_nvrtc_half_headers("Resize")?;
        }
        let rank = input.shape.len();
        if rank > 16 {
            return Err(not_implemented("Resize rank greater than 16"));
        }
        let axes = normalize_axes(self.axes.as_deref(), rank)?;
        let (scales_input, sizes_input) = if self.since_version == 10 {
            (Some(&inputs[1]), None)
        } else {
            (
                inputs
                    .get(2)
                    .filter(|value| !value.is_absent() && value.numel() != 0),
                inputs
                    .get(3)
                    .filter(|value| !value.is_absent() && value.numel() != 0),
            )
        };
        if scales_input.is_some() == sizes_input.is_some() {
            return Err(EpError::KernelFailed(
                "cuda_ep Resize: exactly one non-empty scales or sizes input is required".into(),
            ));
        }
        let mut scales = vec![1.0_f64; rank];
        let mut expected = input.shape.to_vec();
        if let Some(scales_input) = scales_input {
            let values = device_f32(&self.runtime, scales_input, "scales")?;
            if values.len() != axes.len() {
                return Err(EpError::KernelFailed(
                    "cuda_ep Resize: scales length must match resize axes".into(),
                ));
            }
            for (&axis, &scale) in axes.iter().zip(&values) {
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(EpError::KernelFailed(
                        "cuda_ep Resize: scales must be positive and finite".into(),
                    ));
                }
                scales[axis] = f64::from(scale);
                expected[axis] = (input.shape[axis] as f64 * scales[axis]).floor() as usize;
            }
        } else if let Some(sizes_input) = sizes_input {
            let values = device_i64(&self.runtime, sizes_input, "sizes")?;
            if values.len() != axes.len() {
                return Err(EpError::KernelFailed(
                    "cuda_ep Resize: sizes length must match resize axes".into(),
                ));
            }
            for (&axis, &size) in axes.iter().zip(&values) {
                let size = usize::try_from(size)
                    .ok()
                    .filter(|&value| value > 0)
                    .ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep Resize: sizes must be positive and addressable".into(),
                        )
                    })?;
                expected[axis] = size;
                scales[axis] = size as f64 / input.shape[axis] as f64;
            }
        }
        if output.shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Resize: output shape {:?}, expected {expected:?}",
                output.shape
            )));
        }
        if output.numel() == 0 {
            return Ok(());
        }
        let input_strides = compute_contiguous_strides(input.shape);
        let output_strides = compute_contiguous_strides(output.shape);
        let integer_metadata = input
            .shape
            .iter()
            .chain(output.shape)
            .map(|&value| value as u64)
            .chain(input_strides.into_iter().map(|value| value as u64))
            .chain(output_strides.into_iter().map(|value| value as u64))
            .collect::<Vec<_>>();
        let integer_bytes = integer_metadata
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let scale_bytes = scales
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let integer_pointer = upload_bytes(&self.runtime, &integer_bytes)?;
        let scale_pointer = match upload_bytes(&self.runtime, &scale_bytes) {
            Ok(pointer) => pointer,
            Err(error) => {
                let _ = unsafe { self.runtime.free_raw(integer_pointer) };
                return Err(error);
            }
        };
        let function = self
            .runtime
            .nvrtc_function("resize_common_v1", SOURCE, "resize_nd")?;
        let input_pointer = cuptr(input.data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let output_elements = output.numel() as u64;
        let rank = rank as i32;
        let interpolation_mode = match self.mode {
            InterpolationMode::Nearest => 0i32,
            InterpolationMode::Linear => 1,
        };
        let coordinate_mode = match self.coordinate_mode {
            CoordinateMode::HalfPixel => 0i32,
            CoordinateMode::AlignCorners => 1,
            CoordinateMode::Asymmetric => 2,
        };
        let nearest_mode = match self.nearest_mode {
            NearestMode::RoundPreferFloor => 0i32,
            NearestMode::RoundPreferCeil => 1,
            NearestMode::Floor => 2,
            NearestMode::Ceil => 3,
        };
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_pointer)
            .arg(&output_pointer)
            .arg(&integer_pointer)
            .arg(&scale_pointer)
            .arg(&output_elements)
            .arg(&rank)
            .arg(&dtype)
            .arg(&interpolation_mode)
            .arg(&coordinate_mode)
            .arg(&nearest_mode);
        let launch = unsafe {
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
        .map_err(|error| driver_err("launch Resize", error));
        let sync = launch.and_then(|_| self.runtime.synchronize());
        let free_scales = unsafe { self.runtime.free_raw(scale_pointer) };
        let free_integer = unsafe { self.runtime.free_raw(integer_pointer) };
        sync.and(free_scales).and(free_integer)
    }
}

fn normalize_axes(raw: Option<&[i64]>, rank: usize) -> Result<Vec<usize>> {
    let Some(raw) = raw.filter(|values| !values.is_empty()) else {
        return Ok((0..rank).collect());
    };
    let mut axes = Vec::with_capacity(raw.len());
    for &raw_axis in raw {
        let axis = if raw_axis < 0 {
            raw_axis + rank as i64
        } else {
            raw_axis
        };
        if axis < 0 || axis as usize >= rank || axes.contains(&(axis as usize)) {
            return Err(EpError::KernelFailed(
                "cuda_ep Resize: axes must be unique and in range".into(),
            ));
        }
        axes.push(axis as usize);
    }
    Ok(axes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{NodeId, ValueId};

    fn resize_node(name: &str, attribute: Attribute) -> Node {
        let mut node = Node::new(
            NodeId(0),
            "Resize",
            vec![Some(ValueId(0)), Some(ValueId(1))],
            vec![ValueId(2)],
        );
        node.attributes.insert(name.into(), attribute);
        node
    }

    #[test]
    fn claim_rejects_deferred_modes_fail_closed() {
        for (name, attribute, expected) in [
            (
                "mode",
                Attribute::String(b"cubic".to_vec()),
                "mode \"cubic\" unsupported",
            ),
            (
                "coordinate_transformation_mode",
                Attribute::String(b"tf_crop_and_resize".to_vec()),
                "coordinate_transformation_mode \"tf_crop_and_resize\" unsupported",
            ),
            (
                "coordinate_transformation_mode",
                Attribute::String(b"pytorch_half_pixel".to_vec()),
                "coordinate_transformation_mode \"pytorch_half_pixel\" unsupported",
            ),
            (
                "coordinate_transformation_mode",
                Attribute::String(b"half_pixel_symmetric".to_vec()),
                "coordinate_transformation_mode \"half_pixel_symmetric\" unsupported",
            ),
            ("antialias", Attribute::Int(1), "antialias=1 unsupported"),
            (
                "keep_aspect_ratio_policy",
                Attribute::String(b"not_larger".to_vec()),
                "keep_aspect_ratio_policy \"not_larger\" unsupported",
            ),
        ] {
            let reason = crate::kernels::standard_claims::unsupported_reason(
                &resize_node(name, attribute),
                &[],
                &[DataType::Float32, DataType::Float32],
            )
            .expect("deferred Resize mode must be rejected");
            assert!(reason.contains(expected), "{reason}");
        }
    }
}

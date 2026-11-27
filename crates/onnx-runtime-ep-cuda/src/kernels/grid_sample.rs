//! ONNX `GridSample` using separable native CUDA sampling.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Dim, Node, Shape};

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
__device__ __forceinline__ int map_index(long long index, int size, int padding, int aligned) {
  if (padding == 0) return index < 0 || index >= size ? -1 : index;
  if (padding == 1) return index < 0 ? 0 : (index >= size ? size - 1 : index);
  if (size == 1) return 0;
  const long long period = 2LL * (size - aligned);
  index %= period;
  if (index < 0) index += period;
  return index < size ? index : period - index - (1 - aligned);
}

__device__ __forceinline__ float cubic_weight(float distance) {
  const float a = -0.75f;
  const float x = fabsf(distance);
  if (x <= 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
  return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
}

struct Axis {
  int index[4];
  float weight[4];
  unsigned long long stride;
};

template<int Taps>
__device__ __forceinline__ bool prepare_axis(
    Axis& axis, float normalized, int size, int padding, int aligned) {
  if (!isfinite(normalized)) return false;
  const float scale = (float)(size - aligned) / 2.0f;
  const float shift = scale - (1 - aligned) * 0.5f;
  float coordinate = fmaf(normalized, scale, shift);
  if (Taps == 4) {
    // Keep coordinates inside the sampling interval, including half-pixel
    // endpoints. Outside it, apply border padding before computing weights.
    const float low = aligned ? 0.0f : -0.5f;
    const float high = (float)(size - 1) - low;
    if (padding == 1 && (coordinate < low || coordinate > high)) {
      coordinate = fminf(fmaxf(coordinate, 0.0f), (float)(size - 1));
    }
    // Reflection is periodic for cubic taps; reduce before indexing.
    if (padding == 2) {
      if (size == 1) coordinate = 0.0f;
      else {
        const float period = 2.0f * (float)(size - aligned);
        coordinate = isfinite(coordinate) ? fmodf(coordinate, period)
                                          : fmaf(fmodf(normalized, 4.0f), scale, shift);
      }
    }
  } else {
    if (padding == 1) coordinate = fminf(fmaxf(coordinate, 0.0f), (float)(size - 1));
    if (padding == 2) {
      if (!isfinite(coordinate)) {
        coordinate = fmaf(fmodf(normalized, 4.0f), scale, shift);
      }
      coordinate = reflect_coordinate(coordinate, size, aligned);
    }
  }
  if (!isfinite(coordinate)) return false;
  if (padding == 0 && (coordinate < -2.0f || coordinate > (float)size + 2.0f)) return false;
  const long long start = Taps == 1 ? (long long)nearbyintf(coordinate)
                                   : (long long)floorf(coordinate);
  const float fraction = coordinate - (float)start;
  #pragma unroll
  for (int tap = 0; tap < Taps; ++tap) {
    const int delta = tap - (Taps == 4 ? 1 : 0);
    axis.index[tap] = map_index(start + delta, size, padding, aligned);
    axis.weight[tap] = Taps == 4 ? cubic_weight(fraction - delta)
                     : Taps == 2 ? (tap == 0 ? 1.0f - fraction : fraction)
                                 : 1.0f;
  }
  return true;
}

template<int Dims, int Taps>
struct Interpolate {
  __device__ __forceinline__ static float run(
      const void* input, int dtype, unsigned long long offset, const Axis* axes) {
    const Axis& axis = axes[Dims - 1];
    float result = 0.0f;
    #pragma unroll
    for (int tap = 0; tap < Taps; ++tap) {
      const int index = axis.index[tap];
      const float value = index < 0 ? 0.0f : Interpolate<Dims - 1, Taps>::run(
          input, dtype, offset + (unsigned long long)index * axis.stride, axes);
      if (Taps == 1) return value;
      if (tap == 0) result = value * axis.weight[tap];
      else result += value * axis.weight[tap];
    }
    return result;
  }
};

template<int Taps>
struct Interpolate<0, Taps> {
  __device__ __forceinline__ static float run(
      const void* input, int dtype, unsigned long long offset, const Axis*) {
    return load_value(input, dtype, offset);
  }
};

template<int Dims, int Taps>
__device__ __forceinline__ void sample_grid(
    const void* input, const void* grid, void* output, unsigned long long output_elements,
    unsigned long long channels, unsigned long long output_spatial,
    int width, int height, int depth, int dtype, int grid_dtype, int padding, int aligned) {
  const int sizes[] = {width, height, depth};
  for (unsigned long long output_index =
           blockIdx.x * blockDim.x + threadIdx.x;
       output_index < output_elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long input_plane = output_index / output_spatial;
    const unsigned long long batch = input_plane / channels;
    const unsigned long long spatial = output_index % output_spatial;
    const unsigned long long grid_index = (batch * output_spatial + spatial) * Dims;
    Axis axes[Dims];
    unsigned long long input_spatial = 1;
    bool valid = true;
    #pragma unroll
    for (int axis = 0; axis < Dims; ++axis) {
      axes[axis].stride = input_spatial;
      input_spatial *= sizes[axis];
      valid &= prepare_axis<Taps>(axes[axis], load_value(grid, grid_dtype, grid_index + axis),
                                  sizes[axis], padding, aligned);
    }
    const float value = valid ? Interpolate<Dims, Taps>::run(
        input, dtype, input_plane * input_spatial, axes) : 0.0f;
    store_value(output, dtype, output_index, value);
  }
}

template<int Dims>
__device__ __forceinline__ void sample_mode(
    const void* input, const void* grid, void* output, unsigned long long output_elements,
    unsigned long long channels, unsigned long long output_spatial,
    int width, int height, int depth, int dtype, int grid_dtype, int taps, int padding, int aligned) {
  if (taps == 1) sample_grid<Dims, 1>(input, grid, output, output_elements, channels,
      output_spatial, width, height, depth, dtype, grid_dtype, padding, aligned);
  else if (taps == 2) sample_grid<Dims, 2>(input, grid, output, output_elements, channels,
      output_spatial, width, height, depth, dtype, grid_dtype, padding, aligned);
  else sample_grid<Dims, 4>(input, grid, output, output_elements, channels,
      output_spatial, width, height, depth, dtype, grid_dtype, padding, aligned);
}

extern "C" __global__ void grid_sample(
    const void* input, const void* grid, void* output,
    unsigned long long output_elements, unsigned long long channels,
    unsigned long long output_spatial, int dims, int width, int height, int depth,
    int dtype, int grid_dtype, int taps, int padding, int aligned) {
  if (dims == 1) sample_mode<1>(input, grid, output, output_elements, channels,
      output_spatial, width, height, depth, dtype, grid_dtype, taps, padding, aligned);
  else if (dims == 2) sample_mode<2>(input, grid, output, output_elements, channels,
      output_spatial, width, height, depth, dtype, grid_dtype, taps, padding, aligned);
  else sample_mode<3>(input, grid, output, output_elements, channels,
      output_spatial, width, height, depth, dtype, grid_dtype, taps, padding, aligned);
}
"#;

#[derive(Clone, Copy)]
#[repr(i32)]
enum Mode {
    Nearest = 1,
    Linear = 2,
    Cubic = 4,
}

#[derive(Clone, Copy)]
#[repr(i32)]
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
        let opset = u64::from(self.since_version);
        validate_arity(node).map_err(grid_error)?;
        validate_shapes(opset, input_shapes, |dim| Some(*dim)).map_err(grid_error)?;
        Ok(Box::new(GridSampleKernel {
            runtime: self.runtime.clone(),
            options: Options::parse(node, opset).map_err(grid_error)?,
            opset,
        }))
    }
}

struct Options {
    mode: Mode,
    padding: Padding,
    align_corners: bool,
}

impl Options {
    fn parse(node: &Node, opset: u64) -> std::result::Result<Self, String> {
        let (linear, cubic) = if opset < 20 {
            ("bilinear", "bicubic")
        } else {
            ("linear", "cubic")
        };
        let mode = match string_attr(node, "mode")?.unwrap_or(linear) {
            "nearest" => Mode::Nearest,
            value if value == linear => Mode::Linear,
            value if value == cubic => Mode::Cubic,
            value => {
                return Err(format!(
                    "mode {value:?} unsupported at opset {opset}; expected {linear}, nearest, or {cubic}"
                ));
            }
        };
        let padding = match string_attr(node, "padding_mode")?.unwrap_or("zeros") {
            "zeros" => Padding::Zeros,
            "border" => Padding::Border,
            "reflection" => Padding::Reflection,
            value => {
                return Err(format!(
                    "padding_mode {value:?} unsupported; expected zeros, border, or reflection"
                ));
            }
        };
        let align_corners = match node.attr("align_corners") {
            None => false,
            Some(value) => match value.as_int() {
                Some(0) => false,
                Some(1) => true,
                _ => return Err("align_corners must be 0 or 1".into()),
            },
        };
        Ok(Self {
            mode,
            padding,
            align_corners,
        })
    }
}

fn string_attr<'a>(node: &'a Node, name: &str) -> std::result::Result<Option<&'a str>, String> {
    node.attr(name)
        .map(|value| Attribute::as_str(value).ok_or_else(|| format!("{name} must be a string")))
        .transpose()
}

fn validate_arity(node: &Node) -> std::result::Result<(), String> {
    if matches!(node.inputs.as_slice(), [Some(_), Some(_)]) && node.outputs.len() == 1 {
        Ok(())
    } else {
        Err("expected two present inputs (X and grid) and exactly one output".into())
    }
}

fn validate_shapes<T>(
    opset: u64,
    shapes: &[impl AsRef<[T]>],
    extent: impl Fn(&T) -> Option<usize>,
) -> std::result::Result<(), String> {
    let [input, grid] = shapes else {
        return Err("expected shape metadata for X and grid".into());
    };
    let input = input.as_ref();
    let grid = grid.as_ref();
    let rank = input.len();
    let ranks = if opset < 20 { 4..=4 } else { 3..=5 };
    if !ranks.contains(&rank) {
        return Err(format!(
            "X rank {rank} unsupported at opset {opset}; CUDA supports ranks {ranks:?}"
        ));
    }
    if grid.len() != rank {
        return Err(format!("grid rank {} must match X rank {rank}", grid.len()));
    }
    if let (Some(n), Some(grid_n)) = (extent(&input[0]), extent(&grid[0]))
        && n != grid_n
    {
        return Err(format!("grid batch {grid_n} must match X batch {n}"));
    }
    if let Some(coords) = extent(&grid[rank - 1])
        && coords != rank - 2
    {
        return Err(format!(
            "grid has {coords} coordinates per location; expected {}",
            rank - 2
        ));
    }
    for (axis, dim) in input.iter().enumerate().skip(2) {
        if let Some(size) = extent(dim)
            && !(1..=i32::MAX as usize).contains(&size)
        {
            return Err(format!(
                "X spatial axis {axis} size {size} must be in 1..={}",
                i32::MAX
            ));
        }
    }
    Ok(())
}

fn dtype_code(dtype: DataType, name: &str) -> std::result::Result<i32, String> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        other => Err(format!(
            "{name} dtype {other:?} unsupported on CUDA; expected Float32, Float16, or BFloat16"
        )),
    }
}

fn dtype_codes(
    opset: u64,
    input: DataType,
    grid: DataType,
) -> std::result::Result<(i32, i32), String> {
    let code = |dtype, name| {
        if opset < 22 && dtype == DataType::BFloat16 {
            return Err(format!(
                "{name} dtype BFloat16 requires opset 22 or later; got {opset}"
            ));
        }
        dtype_code(dtype, name)
    };
    Ok((code(input, "X")?, code(grid, "grid")?))
}

pub(super) fn validate_claim(
    node: &Node,
    opset: u64,
    input_shapes: &[Shape],
    input_dtypes: &[DataType],
) -> std::result::Result<(), String> {
    validate_arity(node)?;
    let [input, grid] = input_dtypes else {
        return Err("expected dtype metadata for X and grid".into());
    };
    Options::parse(node, opset)?;
    validate_shapes(opset, input_shapes, |dim: &Dim| dim.as_static())?;
    dtype_codes(opset, *input, *grid)?;
    Ok(())
}

fn grid_error(reason: String) -> EpError {
    EpError::KernelFailed(format!("cuda_ep GridSample: {reason}"))
}

struct GridSampleKernel {
    runtime: Arc<CudaRuntime>,
    options: Options,
    opset: u64,
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
        validate_shapes(self.opset, &[input.shape, grid.shape], |dim| Some(*dim))
            .map_err(grid_error)?;
        let rank = input.shape.len();
        if outputs[0].dtype != input.dtype {
            return Err(grid_error(format!(
                "output dtype {:?} must match X dtype {:?}",
                outputs[0].dtype, input.dtype
            )));
        }
        if !input.is_contiguous() || !grid.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented("GridSample with strided tensors"));
        }
        if outputs[0].shape.len() != rank
            || outputs[0].shape[..2] != input.shape[..2]
            || outputs[0].shape[2..] != grid.shape[1..rank - 1]
        {
            let expected = [&input.shape[..2], &grid.shape[1..rank - 1]].concat();
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GridSample: output shape {:?}, expected {expected:?}",
                outputs[0].shape
            )));
        }
        let (dtype, grid_dtype) =
            dtype_codes(self.opset, input.dtype, grid.dtype).map_err(grid_error)?;
        if dtype != 0 || grid_dtype != 0 {
            self.runtime.require_nvrtc_half_headers("GridSample")?;
        }
        let output_elements = outputs[0].numel() as u64;
        if output_elements == 0 {
            return Ok(());
        }
        let function =
            self.runtime
                .nvrtc_function("grid_sample_separable_v1", SOURCE, "grid_sample")?;
        let input_pointer = cuptr(input.data_ptr::<u8>() as *const c_void);
        let grid_pointer = cuptr(grid.data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let channels = input.shape[1] as u64;
        let output_spatial = grid.shape[1..rank - 1].iter().product::<usize>() as u64;
        let dims = (rank - 2) as i32;
        let mut sizes = input.shape[2..].iter().rev().map(|&size| size as i32);
        let width = sizes.next().expect("validated GridSample spatial rank");
        let height = sizes.next().unwrap_or(1);
        let depth = sizes.next().unwrap_or(1);
        let taps = self.options.mode as i32;
        let padding = self.options.padding as i32;
        let aligned = i32::from(self.options.align_corners);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_pointer)
            .arg(&grid_pointer)
            .arg(&output_pointer)
            .arg(&output_elements)
            .arg(&channels)
            .arg(&output_spatial)
            .arg(&dims)
            .arg(&width)
            .arg(&height)
            .arg(&depth)
            .arg(&dtype)
            .arg(&grid_dtype)
            .arg(&taps)
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
    fn claim_tracks_schema_mode_rank_and_grid_dtype_boundaries() {
        use DataType::{BFloat16, Float16, Float32, Float64};
        use onnx_runtime_ir::static_shape;
        let line = [static_shape([1, 2, 3]), static_shape([1, 4, 1])];
        let image = [static_shape([1, 2, 3, 4]), static_shape([1, 4, 5, 2])];
        let volume = [static_shape([1, 2, 3, 4, 5]), static_shape([1, 4, 5, 6, 3])];
        let reason = |mode: &str, opset, shapes: &[Shape], dtypes| {
            crate::kernels::standard_claims::unsupported_reason_at_opset(
                &node(mode),
                opset,
                shapes,
                &dtypes,
                crate::kernels::conv_transpose::Contract::Standalone,
            )
        };
        for (mode, opset, shapes, types) in [
            ("bilinear", 19, &image, [Float32, Float32]),
            ("bicubic", 19, &image, [Float32, Float16]),
            ("linear", 20, &line, [Float32, Float16]),
            ("nearest", 20, &volume, [Float32, Float32]),
            ("cubic", 21, &volume, [Float16, Float32]),
            ("cubic", 22, &volume, [Float16, BFloat16]),
        ] {
            assert_eq!(reason(mode, opset, shapes, types), None, "{mode}/{opset}");
        }
        for (mode, opset, shapes, types, expected) in [
            (
                "linear",
                21,
                &image,
                [BFloat16, Float32],
                "X dtype BFloat16",
            ),
            ("cubic", 19, &image, [Float32, Float32], "expected bilinear"),
            (
                "bilinear",
                20,
                &image,
                [Float32, Float32],
                "expected linear",
            ),
            ("nearest", 19, &volume, [Float32, Float32], "rank 5"),
            (
                "linear",
                21,
                &image,
                [Float32, BFloat16],
                "requires opset 22",
            ),
            (
                "linear",
                22,
                &image,
                [Float32, Float64],
                "grid dtype Float64",
            ),
        ] {
            let reason = reason(mode, opset, shapes, types).expect("must decline CUDA placement");
            assert!(reason.contains(expected), "{reason}");
        }
        let malformed = [volume[0].clone(), static_shape([1, 4, 5, 6, 2])];
        assert!(
            reason("linear", 22, &malformed, [Float32, Float32])
                .unwrap()
                .contains("expected 3")
        );
    }
}

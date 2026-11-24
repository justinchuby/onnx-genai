//! Common ONNX `Resize` modes implemented with N-D NVRTC interpolation.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::{Attribute, DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

use super::metadata_workspace::{
    MetadataWorkspace, checked_metadata_strides, checked_tensor_bytes,
};

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

template <bool Symmetric>
__device__ __forceinline__ void resize_nd_impl(
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
      double coordinate;
      if (Symmetric) {
        coordinate = scales[rank + axis] + ((double)output_coordinate + 0.5) / scales[axis] - 0.5;
      } else {
        coordinate = source_coordinate(
            output_coordinate, input_dimensions[axis], output_dimensions[axis],
            scales[axis], coordinate_mode);
      }
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

extern "C" __global__ void resize_nd(
    const void* input, void* output, const unsigned long long* metadata,
    const double* scales, unsigned long long elements, int rank, int dtype,
    int interpolation_mode, int coordinate_mode, int nearest_mode) {
  resize_nd_impl<false>(input, output, metadata, scales, elements, rank, dtype,
                        interpolation_mode, coordinate_mode, nearest_mode);
}

extern "C" __global__ void resize_nd_symmetric(
    const void* input, void* output, const unsigned long long* metadata,
    const double* scales, unsigned long long elements, int rank, int dtype,
    int interpolation_mode, int coordinate_mode, int nearest_mode) {
  resize_nd_impl<true>(input, output, metadata, scales, elements, rank, dtype,
                       interpolation_mode, coordinate_mode, nearest_mode);
}
"#;

#[derive(Clone, Copy)]
enum InterpolationMode {
    Nearest,
    Linear,
}

#[derive(Clone, Copy)]
pub(super) enum CoordinateMode {
    HalfPixel,
    AlignCorners,
    Asymmetric,
    HalfPixelSymmetric,
}

impl CoordinateMode {
    pub(super) fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "half_pixel" => Ok(Self::HalfPixel),
            "align_corners" => Ok(Self::AlignCorners),
            "asymmetric" => Ok(Self::Asymmetric),
            "half_pixel_symmetric" => Ok(Self::HalfPixelSymmetric),
            other => Err(format!(
                "coordinate_transformation_mode {other:?} unsupported; expected half_pixel, \
                 align_corners, asymmetric, or half_pixel_symmetric (Resize-19)"
            )),
        }
    }
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
        let coordinate_mode = CoordinateMode::parse(
            node.attr("coordinate_transformation_mode")
                .and_then(Attribute::as_str)
                .unwrap_or(coordinate_default),
        )
        .map_err(|reason| not_implemented(format!("Resize {reason}")))?;
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
    let mut bytes =
        vec![0; checked_tensor_bytes("Resize", "control input", input.dtype, input.shape)?];
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResizeControl {
    Scales(usize),
    Sizes(usize),
}

fn select_control(since_version: u32, inputs: &[TensorMetadata<'_>]) -> Result<ResizeControl> {
    if since_version == 10 {
        return inputs
            .get(1)
            .filter(|input| input.present)
            .map(|_| ResizeControl::Scales(1))
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep Resize-10: scales input 1 must be present".into())
            });
    }
    let nonempty = |slot: usize| -> Result<bool> {
        let Some(input) = inputs.get(slot).filter(|input| input.present) else {
            return Ok(false);
        };
        let elements = if input.shape.contains(&0) {
            Some(0)
        } else {
            onnx_runtime_ir::checked_numel(input.shape)
        };
        elements.map(|elements| elements != 0).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Resize: control input {slot} element count overflows; reduce its shape"
            ))
        })
    };
    match (nonempty(2)?, nonempty(3)?) {
        (true, false) => Ok(ResizeControl::Scales(2)),
        (false, true) => Ok(ResizeControl::Sizes(3)),
        _ => Err(EpError::KernelFailed(
            "cuda_ep Resize: exactly one non-empty scales or sizes input is required".into(),
        )),
    }
}

struct ResizeMetadata {
    workspace: MetadataWorkspace,
    control: ResizeControl,
    symmetric: bool,
    integer_bytes: usize,
}

fn resize_metadata(
    since_version: u32,
    coordinate_mode: CoordinateMode,
    inputs: &[TensorMetadata<'_>],
    outputs: &[OutputMetadata<'_>],
) -> Result<ResizeMetadata> {
    if inputs.is_empty()
        || !inputs[0].present
        || outputs.len() != 1
        || (since_version == 10 && inputs.len() != 2)
        || (since_version >= 11 && !(1..=4).contains(&inputs.len()))
    {
        return Err(EpError::KernelFailed(
            "cuda_ep Resize: invalid input/output metadata arity".into(),
        ));
    }
    let rank = inputs[0].shape.len();
    if rank > 16 {
        return Err(not_implemented("Resize rank greater than 16"));
    }
    if inputs[0].shape.contains(&0) {
        return Err(not_implemented("Resize with an empty input dimension"));
    }
    dtype_code(inputs[0].dtype)?;
    checked_tensor_bytes("Resize", "input", inputs[0].dtype, inputs[0].shape)?;
    let control = select_control(since_version, inputs)?;
    let symmetric = matches!(coordinate_mode, CoordinateMode::HalfPixelSymmetric)
        && matches!(control, ResizeControl::Scales(_));
    let integer_bytes = rank * 32; // rank <= 16; four u64 arrays.
    let bytes = integer_bytes + rank * 8 * if symmetric { 2 } else { 1 };
    Ok(ResizeMetadata {
        workspace: MetadataWorkspace::for_bytes("Resize", bytes, &outputs[0])?,
        control,
        symmetric,
        integer_bytes,
    })
}

fn scaled_extent(axis: usize, input: usize, scale: f64) -> Result<usize> {
    let extent = (input as f64 * scale).floor();
    if !scale.is_finite()
        || scale <= 0.0
        || !extent.is_finite()
        || extent < 0.0
        || extent >= isize::MAX as f64
    {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Resize: scale {scale} produces an unaddressable extent at axis {axis}; use a positive finite scale and reduce the resized shape"
        )));
    }
    Ok(extent as usize)
}

impl ResizeKernel {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
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
        let metadata: [_; 4] = std::array::from_fn(|index| {
            inputs.get(index).map_or(
                TensorMetadata::new(DataType::Undefined, &[], false),
                |input| TensorMetadata::new(input.dtype, input.shape, !input.is_absent()),
            )
        });
        let plan = resize_metadata(
            self.since_version,
            self.coordinate_mode,
            &metadata[..inputs.len()],
            &[OutputMetadata::from_view(&outputs[0])],
        )?;
        let upload = plan.workspace.bind(workspace)?;
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
        let (scales_input, sizes_input) = match plan.control {
            ResizeControl::Scales(slot) => (Some(&inputs[slot]), None),
            ResizeControl::Sizes(slot) => (None, Some(&inputs[slot])),
        };
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
                expected[axis] = scaled_extent(axis, input.shape[axis], scales[axis])?;
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
        checked_tensor_bytes("Resize", "output", output.dtype, output.shape)?;
        if output.numel() == 0 {
            return Ok(());
        }
        let symmetric = plan.symmetric;
        if symmetric {
            // Sizes specify the exact extent, so only scales can truncate it.
            // Append one offset per axis to the combined descriptor payload.
            scales.reserve(rank);
            for axis in 0..rank {
                let input_width = input.shape[axis] as f64;
                let adjustment = output.shape[axis] as f64 / (input_width * scales[axis]);
                scales.push(input_width * 0.5 * (1.0 - adjustment));
            }
        }
        let input_strides = checked_metadata_strides("Resize", "input", input.shape)?;
        let output_strides = checked_metadata_strides("Resize", "output", output.shape)?;
        let integer_metadata = input
            .shape
            .iter()
            .chain(output.shape)
            .map(|&value| value as u64)
            .chain(input_strides.into_iter().map(|value| value as u64))
            .chain(output_strides.into_iter().map(|value| value as u64))
            .collect::<Vec<_>>();
        let mut payload = Vec::with_capacity(plan.integer_bytes + scales.len() * 8);
        for value in integer_metadata {
            payload.extend_from_slice(&value.to_ne_bytes());
        }
        for value in scales {
            payload.extend_from_slice(&value.to_ne_bytes());
        }
        let entry = if symmetric {
            "resize_nd_symmetric"
        } else {
            "resize_nd"
        };
        let function = self
            .runtime
            .nvrtc_function("resize_common_v2", SOURCE, entry)?;
        let integer_pointer = upload.upload_bytes(&self.runtime, &payload)?;
        let scale_pointer = integer_pointer + plan.integer_bytes as u64;
        let input_pointer = cuptr(input.data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let output_elements = output.numel() as u64;
        let rank = rank as i32;
        let interpolation_mode = match self.mode {
            InterpolationMode::Nearest => 0i32,
            InterpolationMode::Linear => 1,
        };
        let coordinate_mode = match self.coordinate_mode {
            CoordinateMode::HalfPixel | CoordinateMode::HalfPixelSymmetric => 0i32,
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
        launch.and_then(|_| self.runtime.synchronize())
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

impl Kernel for ResizeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }
    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        Ok(WorkspacePreparation::Ready(
            resize_metadata(self.since_version, self.coordinate_mode, inputs, outputs)?
                .workspace
                .requirement(),
        ))
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Resize reads device scales/sizes and uploads borrowed metadata eagerly",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{NodeId, ValueId};

    #[test]
    fn selected_control_source_drives_one_exact_integer_double_composite() {
        let mut inputs = [
            TensorMetadata::new(DataType::Float32, &[2, 2], true),
            TensorMetadata::new(DataType::Float32, &[0], true),
            TensorMetadata::new(DataType::Float32, &[2], true),
            TensorMetadata::new(DataType::Int64, &[0], true),
        ];
        let outputs = [OutputMetadata::Unresolved {
            dtype: DataType::Float32,
        }];
        let scales =
            resize_metadata(11, CoordinateMode::HalfPixelSymmetric, &inputs, &outputs).unwrap();
        assert_eq!(scales.control, ResizeControl::Scales(2));
        assert!(scales.symmetric);
        assert_eq!(scales.integer_bytes, 64);
        assert_eq!(scales.workspace.requirement().bytes, 96);
        inputs[2].shape = &[0];
        inputs[3].shape = &[2];
        let sizes =
            resize_metadata(11, CoordinateMode::HalfPixelSymmetric, &inputs, &outputs).unwrap();
        assert_eq!(sizes.control, ResizeControl::Sizes(3));
        assert!(!sizes.symmetric);
        assert_eq!(sizes.workspace.requirement().bytes, 80);
        inputs[2].shape = &[2];
        assert!(select_control(11, &inputs).is_err());
    }
    #[test]
    fn resize_zero_rank_preserves_versioned_control_rules_and_known_empty_output() {
        let scalar = [
            TensorMetadata::new(DataType::Float32, &[], true),
            TensorMetadata::new(DataType::Float32, &[0], true),
        ];
        let plan = resize_metadata(
            10,
            CoordinateMode::Asymmetric,
            &scalar,
            &[OutputMetadata::tensor(DataType::Float32, &[])],
        )
        .unwrap();
        assert_eq!(
            plan.workspace.requirement(),
            onnx_runtime_ep_api::WorkspaceRequirement::NONE
        );
        assert!(select_control(11, &scalar).is_err());
        let rank17 = [1; 17];
        let invalid = [
            TensorMetadata::new(DataType::Float32, &rank17, true),
            scalar[1],
        ];
        assert!(
            resize_metadata(
                10,
                CoordinateMode::Asymmetric,
                &invalid,
                &[OutputMetadata::Unresolved {
                    dtype: DataType::Float32
                }]
            )
            .is_err()
        );
        let data = [
            TensorMetadata::new(DataType::Float32, &[2, 2], true),
            TensorMetadata::new(DataType::Float32, &[2], true),
        ];
        assert_eq!(
            resize_metadata(
                10,
                CoordinateMode::Asymmetric,
                &data,
                &[OutputMetadata::tensor(DataType::Float32, &[0, 0])]
            )
            .unwrap()
            .workspace
            .requirement(),
            onnx_runtime_ep_api::WorkspaceRequirement::NONE
        );
    }
    #[test]
    fn scaled_extents_reject_nonfinite_or_unaddressable_values_before_casting() {
        assert_eq!(scaled_extent(0, 2, 1.6).unwrap(), 3);
        assert_eq!(scaled_extent(0, 2, 0.1).unwrap(), 0);
        for scale in [f64::NAN, f64::INFINITY, 0.0, -1.0, 1e300] {
            assert!(
                scaled_extent(3, 2, scale)
                    .unwrap_err()
                    .to_string()
                    .contains("axis 3")
            );
        }
        assert!(scaled_extent(0, isize::MAX as usize, 2.0).is_err());
    }

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

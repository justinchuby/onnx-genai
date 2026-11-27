//! CUDA index transforms for `CenterCropPad` and `Col2Im`.

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
const CENTER_WORDS: usize = 4;
const COL2IM_WORDS: usize = 6;

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void center_crop_pad(
    const unsigned char* input, unsigned char* output,
    const unsigned long long* metadata, int rank, int element_bytes,
    unsigned long long elements) {
  const unsigned long long* input_dimensions = metadata;
  const unsigned long long* input_strides = metadata + rank;
  const unsigned long long* output_strides = metadata + rank * 2;
  const long long* offsets = (const long long*)(metadata + rank * 3);
  for (unsigned long long output_index = blockIdx.x * blockDim.x + threadIdx.x;
       output_index < elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long remaining = output_index;
    unsigned long long input_index = 0;
    bool valid = true;
    for (int axis = 0; axis < rank; ++axis) {
      const unsigned long long coordinate = remaining / output_strides[axis];
      remaining %= output_strides[axis];
      const long long source = (long long)coordinate + offsets[axis];
      if (source < 0 || source >= (long long)input_dimensions[axis]) {
        valid = false;
        break;
      }
      input_index += (unsigned long long)source * input_strides[axis];
    }
    unsigned char* destination = output + output_index * element_bytes;
    if (!valid) {
      for (int byte = 0; byte < element_bytes; ++byte) destination[byte] = 0;
    } else {
      const unsigned char* source = input + input_index * element_bytes;
      for (int byte = 0; byte < element_bytes; ++byte) destination[byte] = source[byte];
    }
  }
}

__device__ __forceinline__ float load_float_value(
    const void* values, int dtype, unsigned long long index) {
  if (dtype == 0) return ((const float*)values)[index];
  if (dtype == 1) return __half2float(((const __half*)values)[index]);
  return __bfloat162float(((const __nv_bfloat16*)values)[index]);
}

__device__ __forceinline__ void store_float_value(
    void* values, int dtype, unsigned long long index, float value) {
  if (dtype == 0) ((float*)values)[index] = value;
  else if (dtype == 1) ((__half*)values)[index] = __float2half_rn(value);
  else ((__nv_bfloat16*)values)[index] = __float2bfloat16_rn(value);
}

extern "C" __global__ void col2im(
    const void* input, void* output, const unsigned long long* metadata,
    int spatial_rank, int dtype, unsigned long long output_elements,
    unsigned long long channels, unsigned long long block_elements,
    unsigned long long locations, unsigned long long image_elements) {
  const unsigned long long* image = metadata;
  const unsigned long long* block = metadata + spatial_rank;
  const unsigned long long* dilations = metadata + spatial_rank * 2;
  const unsigned long long* strides = metadata + spatial_rank * 3;
  const unsigned long long* pads = metadata + spatial_rank * 4;
  const unsigned long long* location_shape = metadata + spatial_rank * 5;
  for (unsigned long long output_index =
           blockIdx.x * blockDim.x + threadIdx.x;
       output_index < output_elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long image_linear = output_index % image_elements;
    const unsigned long long channel_linear = output_index / image_elements;
    const unsigned long long channel = channel_linear % channels;
    const unsigned long long batch = channel_linear / channels;
    float sum = 0.0f;
    for (unsigned long long kernel = 0; kernel < block_elements; ++kernel) {
      unsigned long long image_remaining = image_linear;
      unsigned long long kernel_remaining = kernel;
      unsigned long long location = 0;
      unsigned long long location_stride = 1;
      bool valid = true;
      for (int axis = spatial_rank - 1; axis >= 0; --axis) {
        const unsigned long long image_coordinate = image_remaining % image[axis];
        image_remaining /= image[axis];
        const unsigned long long kernel_coordinate = kernel_remaining % block[axis];
        kernel_remaining /= block[axis];
        const long long numerator =
            (long long)image_coordinate + (long long)pads[axis]
            - (long long)(kernel_coordinate * dilations[axis]);
        if (numerator < 0 || numerator % (long long)strides[axis] != 0) {
          valid = false;
          break;
        }
        const unsigned long long coordinate =
            (unsigned long long)(numerator / (long long)strides[axis]);
        if (coordinate >= location_shape[axis]) {
          valid = false;
          break;
        }
        location += coordinate * location_stride;
        location_stride *= location_shape[axis];
      }
      if (valid) {
        const unsigned long long source =
            (batch * channels * block_elements + channel * block_elements + kernel)
                * locations
            + location;
        sum += load_float_value(input, dtype, source);
      }
    }
    store_float_value(output, dtype, output_index, sum);
  }
}
"#;

pub(super) fn device_i64(
    runtime: &CudaRuntime,
    input: &TensorView,
    op: &str,
    name: &str,
) -> Result<Vec<i64>> {
    if input.dtype != DataType::Int64 || !input.is_contiguous() {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: {name} must be a contiguous Int64 tensor"
        )));
    }
    let bytes = checked_tensor_bytes(op, name, DataType::Int64, input.shape)?;
    let mut bytes = vec![0; bytes];
    if !bytes.is_empty() {
        unsafe { runtime.dtoh(&mut bytes, cuptr(input.data_ptr::<u8>() as *const c_void))? };
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|value| i64::from_ne_bytes(value.try_into().expect("eight-byte chunk")))
        .collect())
}

fn control_elements(input: TensorMetadata<'_>, op: &str, name: &str) -> Result<usize> {
    if !input.present || input.dtype != DataType::Int64 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: {name} must be present Int64 metadata"
        )));
    }
    Ok(checked_tensor_bytes(op, name, DataType::Int64, input.shape)? / 8)
}

fn center_workspace(
    inputs: &[TensorMetadata<'_>],
    outputs: &[OutputMetadata<'_>],
) -> Result<MetadataWorkspace> {
    if inputs.len() != 2 || outputs.len() != 1 || !inputs[0].present {
        return Err(EpError::KernelFailed(
            "cuda_ep CenterCropPad: workspace planning requires two present inputs and one output"
                .into(),
        ));
    }
    control_elements(inputs[1], "CenterCropPad", "shape")?;
    checked_tensor_bytes("CenterCropPad", "input", inputs[0].dtype, inputs[0].shape)?;
    MetadataWorkspace::for_records(
        "CenterCropPad",
        inputs[0].shape.len(),
        CENTER_WORDS,
        &outputs[0],
    )
}

fn col2im_workspace(
    inputs: &[TensorMetadata<'_>],
    outputs: &[OutputMetadata<'_>],
) -> Result<MetadataWorkspace> {
    if inputs.len() != 3 || outputs.len() != 1 || !inputs[0].present || inputs[0].shape.len() != 3 {
        return Err(EpError::KernelFailed("cuda_ep Col2Im: workspace planning requires rank-3 data, image_shape, block_shape and one output".into()));
    }
    let rank = control_elements(inputs[1], "Col2Im", "image_shape")?;
    if rank < 2 || control_elements(inputs[2], "Col2Im", "block_shape")? != rank {
        return Err(EpError::KernelFailed("cuda_ep Col2Im: image_shape and block_shape must contain the same number of elements, at least two".into()));
    }
    checked_tensor_bytes("Col2Im", "input", inputs[0].dtype, inputs[0].shape)?;
    MetadataWorkspace::for_records("Col2Im", rank, COL2IM_WORDS, &outputs[0])
}

fn checked_product(values: &[u64], name: &str) -> Result<u64> {
    if values.contains(&0) {
        return Ok(0);
    }
    values
        .iter()
        .try_fold(1u64, |product, value| product.checked_mul(*value))
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Col2Im: {name} product overflows u64; reduce image/block dimensions"
            ))
        })
}

fn col2im_location_extent(
    axis: usize,
    image: u64,
    block: u64,
    dilation: u64,
    stride: u64,
    pads: [u64; 2],
) -> Result<u64> {
    if block == 0 || dilation == 0 || stride == 0 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Col2Im: block={block}, dilation={dilation}, stride={stride} at axis {axis} must be positive"
        )));
    }
    let receptive = dilation.checked_mul(block-1).and_then(|n| n.checked_add(1))
        .ok_or_else(|| EpError::KernelFailed(format!("cuda_ep Col2Im: receptive extent overflows at axis {axis}; reduce block_shape or dilation")))?;
    let available = image.checked_add(pads[0]).and_then(|n| n.checked_add(pads[1]))
        .ok_or_else(|| EpError::KernelFailed(format!("cuda_ep Col2Im: padded image extent overflows at axis {axis}; reduce image_shape or pads")))?;
    // The device computes image_coordinate + pad_begin - kernel*dilation
    // with signed long long. Bound both the positive intermediate and negative
    // displacement, rather than assuming checked u64 extents prove that ABI.
    let signed_max = i64::MAX as u64;
    let largest_padded_coordinate = image.saturating_sub(1).checked_add(pads[0]);
    if [image, block, dilation, stride, pads[0], pads[1]]
        .iter()
        .any(|&value| value > signed_max)
        || receptive - 1 > signed_max
        || largest_padded_coordinate.is_none_or(|coordinate| coordinate > signed_max)
    {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Col2Im: signed CUDA coordinate arithmetic exceeds i64 at axis {axis}; reduce image_shape, block_shape, dilation or pad_begin"
        )));
    }
    Ok(if available < receptive {
        0
    } else {
        (available - receptive) / stride + 1
    })
}

fn launch_config(elements: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
            1,
            1,
        ),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub struct CenterCropPadFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for CenterCropPadFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(CenterCropPadKernel {
            axes: node
                .attr("axes")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec),
            runtime: self.runtime.clone(),
        }))
    }
}

struct CenterCropPadKernel {
    axes: Option<Vec<i64>>,
    runtime: Arc<CudaRuntime>,
}

impl CenterCropPadKernel {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep CenterCropPad: expected 2 inputs and 1 output".into(),
            ));
        }
        if self.runtime.is_capturing()? {
            return Err(not_implemented(
                "CenterCropPad during CUDA graph capture because shape is a device input",
            ));
        }
        let metadata: [_; 2] = std::array::from_fn(|index| {
            TensorMetadata::new(
                inputs[index].dtype,
                inputs[index].shape,
                !inputs[index].is_absent(),
            )
        });
        let upload = center_workspace(&metadata, &[OutputMetadata::from_view(&outputs[0])])?
            .bind(workspace)?;
        let input = &inputs[0];
        let shape = device_i64(&self.runtime, &inputs[1], "CenterCropPad", "shape")?;
        let output = &mut outputs[0];
        if input.dtype != output.dtype || !input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(
                "CenterCropPad requires contiguous input/output with matching dtypes",
            ));
        }
        let element_bytes = input.dtype.byte_size();
        if element_bytes == 0 {
            return Err(not_implemented(
                "CenterCropPad for packed or variable-width dtype",
            ));
        }
        let rank = input.shape.len();
        let raw_axes = self
            .axes
            .clone()
            .unwrap_or_else(|| (0..rank as i64).collect());
        if shape.len() != raw_axes.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep CenterCropPad: shape has {} dimensions for {} axes",
                shape.len(),
                raw_axes.len()
            )));
        }
        let mut expected = input.shape.to_vec();
        let mut selected = vec![false; rank];
        for (&raw_axis, &dimension) in raw_axes.iter().zip(&shape) {
            let axis = if raw_axis < 0 {
                raw_axis + rank as i64
            } else {
                raw_axis
            };
            if axis < 0 || axis as usize >= rank || dimension < 0 {
                return Err(EpError::KernelFailed(
                    "cuda_ep CenterCropPad: axes must be in range and shape dimensions non-negative"
                        .into(),
                ));
            }
            expected[axis as usize] = dimension as usize;
            selected[axis as usize] = true;
        }
        if output.shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep CenterCropPad: output shape {:?}, expected {expected:?}",
                output.shape
            )));
        }
        checked_tensor_bytes("index transform", "output", output.dtype, output.shape)?;
        if output.numel() == 0 {
            return Ok(());
        }
        let input_strides = checked_metadata_strides("CenterCropPad", "input", input.shape)?;
        let output_strides = checked_metadata_strides("CenterCropPad", "output", output.shape)?;
        let offsets = input.shape.iter().zip(output.shape).zip(selected).map(
            |((&source, &target), selected)| {
                if selected {
                    source.saturating_sub(target) as i64 / 2
                        - target.saturating_sub(source) as i64 / 2
                } else {
                    0
                }
            },
        );
        let metadata = input
            .shape
            .iter()
            .map(|&value| value as u64)
            .chain(input_strides.into_iter().map(|value| value as u64))
            .chain(output_strides.into_iter().map(|value| value as u64))
            .chain(offsets.map(|value| u64::from_ne_bytes(value.to_ne_bytes())))
            .collect::<Vec<_>>();
        let metadata_pointer = upload.upload(&self.runtime, &metadata)?;
        let function =
            self.runtime
                .nvrtc_function("index_transform_v1", SOURCE, "center_crop_pad")?;
        let input_pointer = cuptr(input.data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let rank = i32::try_from(rank)
            .map_err(|_| EpError::KernelFailed("cuda_ep CenterCropPad: rank exceeds i32".into()))?;
        let element_bytes = i32::try_from(element_bytes).map_err(|_| {
            EpError::KernelFailed("cuda_ep CenterCropPad: element size exceeds i32".into())
        })?;
        let elements = output.numel() as u64;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_pointer)
            .arg(&output_pointer)
            .arg(&metadata_pointer)
            .arg(&rank)
            .arg(&element_bytes)
            .arg(&elements);
        let launch = unsafe { builder.launch(launch_config(elements)) }
            .map_err(|error| driver_err("launch CenterCropPad", error));
        launch.and_then(|_| self.runtime.synchronize())
    }
}

impl Kernel for CenterCropPadKernel {
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
            center_workspace(inputs, outputs)?.requirement(),
        ))
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "CenterCropPad reads device shape values and uploads borrowed metadata eagerly",
        )
    }
}

pub struct Col2ImFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for Col2ImFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Col2ImKernel {
            dilations: node
                .attr("dilations")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec)
                .unwrap_or_default(),
            pads: node
                .attr("pads")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec)
                .unwrap_or_default(),
            strides: node
                .attr("strides")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec)
                .unwrap_or_default(),
            runtime: self.runtime.clone(),
        }))
    }
}

struct Col2ImKernel {
    dilations: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
    runtime: Arc<CudaRuntime>,
}

fn positive_values(values: &[i64], rank: usize, default: i64, name: &str) -> Result<Vec<u64>> {
    let values = if values.is_empty() {
        vec![default; rank]
    } else {
        values.to_vec()
    };
    if values.len() != rank || values.iter().any(|&value| value <= 0) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Col2Im: {name} must contain {rank} positive values"
        )));
    }
    Ok(values.into_iter().map(|value| value as u64).collect())
}

fn float_dtype(dtype: DataType, op: &str) -> Result<i32> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        other => Err(not_implemented(format!(
            "{op} dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

impl Col2ImKernel {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep Col2Im: expected 3 inputs and 1 output".into(),
            ));
        }
        if self.runtime.is_capturing()? {
            return Err(not_implemented(
                "Col2Im during CUDA graph capture because image/block shapes are device inputs",
            ));
        }
        let metadata: [_; 3] = std::array::from_fn(|index| {
            TensorMetadata::new(
                inputs[index].dtype,
                inputs[index].shape,
                !inputs[index].is_absent(),
            )
        });
        let upload = col2im_workspace(&metadata, &[OutputMetadata::from_view(&outputs[0])])?
            .bind(workspace)?;
        let input = &inputs[0];
        let image = device_i64(&self.runtime, &inputs[1], "Col2Im", "image_shape")?;
        let block = device_i64(&self.runtime, &inputs[2], "Col2Im", "block_shape")?;
        let output = &mut outputs[0];
        if input.shape.len() != 3 {
            return Err(EpError::KernelFailed(
                "cuda_ep Col2Im: input must have rank 3 [N,C*K,L]".into(),
            ));
        }
        if input.dtype != output.dtype || !input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(
                "Col2Im requires contiguous input/output with matching dtypes",
            ));
        }
        let dtype = float_dtype(input.dtype, "Col2Im")?;
        if dtype != 0 {
            self.runtime.require_nvrtc_half_headers("Col2Im")?;
        }
        let rank = image.len();
        if rank < 2 || block.len() != rank {
            return Err(EpError::KernelFailed(
                "cuda_ep Col2Im: image_shape and block_shape must have equal rank >= 2".into(),
            ));
        }
        if image.iter().any(|&value| value < 0) || block.iter().any(|&value| value <= 0) {
            return Err(EpError::KernelFailed(
                "cuda_ep Col2Im: image dimensions must be non-negative and block dimensions positive"
                    .into(),
            ));
        }
        let image = image
            .into_iter()
            .map(|value| value as u64)
            .collect::<Vec<_>>();
        let block = block
            .into_iter()
            .map(|value| value as u64)
            .collect::<Vec<_>>();
        let dilations = positive_values(&self.dilations, rank, 1, "dilations")?;
        let strides = positive_values(&self.strides, rank, 1, "strides")?;
        let pads = if self.pads.is_empty() {
            vec![0_u64; rank * 2]
        } else if self.pads.len() == rank * 2 && self.pads.iter().all(|&value| value >= 0) {
            self.pads.iter().map(|&value| value as u64).collect()
        } else {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Col2Im: pads must contain {} non-negative values",
                rank * 2
            )));
        };
        let block_elements = checked_product(&block, "block_shape")?;
        if !(input.shape[1] as u64).is_multiple_of(block_elements) {
            return Err(EpError::KernelFailed(
                "cuda_ep Col2Im: input channel-block dimension is not divisible by block_shape product"
                    .into(),
            ));
        }
        let channels = input.shape[1] as u64 / block_elements;
        let mut location_shape = Vec::with_capacity(rank);
        for axis in 0..rank {
            location_shape.push(col2im_location_extent(
                axis,
                image[axis],
                block[axis],
                dilations[axis],
                strides[axis],
                [pads[axis], pads[axis + rank]],
            )?);
        }
        let locations = checked_product(&location_shape, "location shape")?;
        if input.shape[2] as u64 != locations {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Col2Im: input has {} columns, expected {locations}",
                input.shape[2]
            )));
        }
        let mut expected = vec![input.shape[0], channels as usize];
        for &dimension in &image {
            expected.push(usize::try_from(dimension).map_err(|_| {
                EpError::KernelFailed(
                    "cuda_ep Col2Im: image_shape dimension exceeds usize; reduce image dimensions"
                        .into(),
                )
            })?);
        }
        if output.shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Col2Im: output shape {:?}, expected {expected:?}",
                output.shape
            )));
        }
        checked_tensor_bytes("index transform", "output", output.dtype, output.shape)?;
        if output.numel() == 0 {
            return Ok(());
        }
        let image_elements = checked_product(&image, "image_shape")?;
        let metadata = image
            .iter()
            .chain(&block)
            .chain(&dilations)
            .chain(&strides)
            .chain(&pads[..rank])
            .chain(&location_shape)
            .copied()
            .collect::<Vec<_>>();
        let metadata_pointer = upload.upload(&self.runtime, &metadata)?;
        let function = self
            .runtime
            .nvrtc_function("index_transform_v1", SOURCE, "col2im")?;
        let input_pointer = cuptr(input.data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let spatial_rank = i32::try_from(rank)
            .map_err(|_| EpError::KernelFailed("cuda_ep Col2Im: rank exceeds i32".into()))?;
        let output_elements = output.numel() as u64;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_pointer)
            .arg(&output_pointer)
            .arg(&metadata_pointer)
            .arg(&spatial_rank)
            .arg(&dtype)
            .arg(&output_elements)
            .arg(&channels)
            .arg(&block_elements)
            .arg(&locations)
            .arg(&image_elements);
        let launch = unsafe { builder.launch(launch_config(output_elements)) }
            .map_err(|error| driver_err("launch Col2Im", error));
        launch.and_then(|_| self.runtime.synchronize())
    }
}

impl Kernel for Col2ImKernel {
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
            col2im_workspace(inputs, outputs)?.requirement(),
        ))
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Col2Im reads device image/block shapes and uploads borrowed metadata eagerly",
        )
    }
}

#[cfg(test)]
mod workspace_tests {
    use super::*;
    use onnx_runtime_ep_api::WorkspaceRequirement;
    #[test]
    fn index_metadata_planning_uses_control_lengths_without_control_values() {
        let center = [
            TensorMetadata::new(DataType::Int32, &[3, 2], true),
            TensorMetadata::new(DataType::Int64, &[2], true),
        ];
        let unresolved = [OutputMetadata::Unresolved {
            dtype: DataType::Int32,
        }];
        assert_eq!(
            center_workspace(&center, &unresolved)
                .unwrap()
                .requirement()
                .bytes,
            64
        );
        assert_eq!(
            center_workspace(&center, &[OutputMetadata::tensor(DataType::Int32, &[0, 5])])
                .unwrap()
                .requirement(),
            WorkspaceRequirement::NONE
        );
        let col = [
            TensorMetadata::new(DataType::Float32, &[1, 4, 4], true),
            TensorMetadata::new(DataType::Int64, &[1, 2], true),
            TensorMetadata::new(DataType::Int64, &[2, 1], true),
        ];
        assert_eq!(
            col2im_workspace(&col, &unresolved)
                .unwrap()
                .requirement()
                .bytes,
            96
        );
        let mut invalid = col;
        invalid[2] = TensorMetadata::new(DataType::Int64, &[3], true);
        assert!(
            col2im_workspace(&invalid, &unresolved)
                .err()
                .unwrap()
                .to_string()
                .contains("same number of elements")
        );
        invalid[2] = TensorMetadata::new(DataType::Int64, &[usize::MAX], true);
        assert!(col2im_workspace(&invalid, &unresolved).is_err());
    }
    #[test]
    fn col2im_checked_extent_and_products_reject_overflow_before_launch() {
        assert_eq!(col2im_location_extent(0, 3, 2, 1, 1, [0, 0]).unwrap(), 2);
        assert_eq!(col2im_location_extent(1, 1, 3, 1, 2, [0, 0]).unwrap(), 0);
        for (image, block, dilation, stride, pads) in [
            (3, u64::MAX, 2, 1, [0, 0]),
            (u64::MAX, 2, 1, 1, [1, 1]),
            (3, 2, 1, 0, [0, 0]),
        ] {
            assert!(
                col2im_location_extent(2, image, block, dilation, stride, pads)
                    .unwrap_err()
                    .to_string()
                    .contains("axis 2")
            );
        }
        let max = i64::MAX as u64;
        let error = col2im_location_extent(0, 2, 1, 1, max, [max, 0]).unwrap_err();
        assert!(
            error.to_string().contains("signed CUDA coordinate"),
            "{error}"
        );
        // The neighboring representable boundary must still be accepted.
        assert_eq!(
            col2im_location_extent(0, 1, 1, 1, max, [max, 0]).unwrap(),
            2
        );
        assert_eq!(
            col2im_location_extent(0, 1, 2, max, max, [0, 0]).unwrap(),
            0
        );
        assert!(
            checked_product(&[u64::MAX, 2], "block_shape")
                .unwrap_err()
                .to_string()
                .contains("block_shape")
        );
        assert_eq!(
            checked_product(&[u64::MAX, 2, 0], "image_shape").unwrap(),
            0
        );
    }
}

//! CUDA kernels for dtype-agnostic structural operators: `GatherND`,
//! `SpaceToDepth`, and `EyeLike`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    DeviceGraphResource, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::movement::PersistentMetadata;
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const MODULE: &str = "structural_ops_v1";

const SOURCE: &str = r#"
extern "C" __global__ void gather_nd_bytes(
    const unsigned char* data, const void* indices, unsigned char* output,
    const unsigned long long* dimensions, int data_rank, int batch_dims,
    unsigned long long tuples_per_batch, unsigned long long tuple_width,
    unsigned long long tail_length, unsigned long long data_batch_length,
    int element_bytes, int index_is_i64, unsigned long long elements,
    unsigned int* capture_error) {
  for (unsigned long long output_index = blockIdx.x * blockDim.x + threadIdx.x;
       output_index < elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long tail_index = output_index % tail_length;
    const unsigned long long tuple_linear = output_index / tail_length;
    const unsigned long long tuple_index = tuple_linear % tuples_per_batch;
    const unsigned long long batch = tuple_linear / tuples_per_batch;
    unsigned long long source = batch * data_batch_length;
    const unsigned long long index_base =
        (batch * tuples_per_batch + tuple_index) * tuple_width;
    bool valid = true;
    for (unsigned long long dimension = 0; dimension < tuple_width; ++dimension) {
      const long long raw = index_is_i64
          ? ((const long long*)indices)[index_base + dimension]
          : (long long)((const int*)indices)[index_base + dimension];
      const unsigned long long size = dimensions[batch_dims + dimension];
      unsigned long long coordinate;
      if (raw >= 0) {
        coordinate = (unsigned long long)raw;
        valid = coordinate < size;
      } else {
        const unsigned long long magnitude = 0ull - (unsigned long long)raw;
        valid = magnitude <= size;
        coordinate = size - magnitude;
      }
      if (!valid) break;
      unsigned long long stride = 1;
      for (int axis = batch_dims + (int)dimension + 1; axis < data_rank; ++axis)
        stride *= dimensions[axis];
      source += coordinate * stride;
    }
    if (!valid) {
      if (capture_error) atomicOr(capture_error, 4096u);
      continue;
    }
    source += tail_index;
    for (int byte = 0; byte < element_bytes; ++byte)
      output[output_index * element_bytes + byte] =
          data[source * element_bytes + byte];
  }
}

extern "C" __global__ void space_to_depth_bytes(
    const unsigned char* input, unsigned char* output,
    unsigned long long channels, unsigned long long height,
    unsigned long long width, unsigned long long block,
    int element_bytes, unsigned long long elements) {
  const unsigned long long output_height = height / block;
  const unsigned long long output_width = width / block;
  const unsigned long long output_channels = channels * block * block;
  for (unsigned long long output_index = blockIdx.x * blockDim.x + threadIdx.x;
       output_index < elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long rem = output_index;
    const unsigned long long output_x = rem % output_width;
    rem /= output_width;
    const unsigned long long output_y = rem % output_height;
    rem /= output_height;
    const unsigned long long output_channel = rem % output_channels;
    const unsigned long long batch = rem / output_channels;
    const unsigned long long channel = output_channel % channels;
    const unsigned long long block_offset = output_channel / channels;
    const unsigned long long block_y = block_offset / block;
    const unsigned long long block_x = block_offset % block;
    const unsigned long long input_y = output_y * block + block_y;
    const unsigned long long input_x = output_x * block + block_x;
    const unsigned long long input_index =
        ((batch * channels + channel) * height + input_y) * width + input_x;
    for (int byte = 0; byte < element_bytes; ++byte)
      output[output_index * element_bytes + byte] =
          input[input_index * element_bytes + byte];
  }
}

extern "C" __global__ void eye_like_bytes(
    unsigned char* output, unsigned long long columns, long long k,
    unsigned long long one_bits, int element_bytes,
    unsigned long long elements) {
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long row = index / columns;
    const unsigned long long column = index % columns;
    const bool diagonal =
        (k >= 0 && column >= row && column - row == (unsigned long long)k) ||
        (k < 0 && row > column && row - column == 0ull - (unsigned long long)k);
    for (int byte = 0; byte < element_bytes; ++byte)
      output[index * element_bytes + byte] =
          diagonal ? (unsigned char)(one_bits >> (byte * 8)) : 0;
  }
}
"#;

fn grid(elements: usize) -> u32 {
    (elements as u64).div_ceil(BLOCK as u64).clamp(1, 65_535) as u32
}

fn fixed_width(op: &str, dtype: DataType) -> Result<usize> {
    let bytes = dtype.byte_size();
    if bytes == 0 {
        Err(not_implemented(format!(
            "{op} for packed or variable-width dtype {dtype:?}"
        )))
    } else {
        Ok(bytes)
    }
}

fn require_dense(op: &str, inputs: &[TensorView], outputs: &[TensorMut]) -> Result<()> {
    if inputs.iter().any(|input| !input.is_contiguous())
        || outputs.iter().any(|output| !output.is_contiguous())
    {
        Err(not_implemented(format!("{op} with non-contiguous tensors")))
    } else {
        Ok(())
    }
}

fn product(shape: &[usize], op: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |elements, &dimension| {
        elements
            .checked_mul(dimension)
            .ok_or_else(|| EpError::KernelFailed(format!("cuda_ep {op}: shape product overflow")))
    })
}

pub struct GatherNdFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for GatherNdFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(GatherNdKernel {
            batch_dims: node
                .attr("batch_dims")
                .and_then(Attribute::as_int)
                .unwrap_or(0),
            runtime: self.runtime.clone(),
            dimensions: Mutex::new(PersistentMetadata::new(self.runtime.clone())),
        }))
    }
}

struct GatherNdKernel {
    batch_dims: i64,
    runtime: Arc<CudaRuntime>,
    dimensions: Mutex<PersistentMetadata>,
}

impl Kernel for GatherNdKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GatherND: expected 2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        require_dense("GatherND", inputs, outputs)?;
        let data = &inputs[0];
        let indices = &inputs[1];
        if indices.shape.is_empty() || data.shape.is_empty() {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherND: data and indices must have rank at least 1".into(),
            ));
        }
        if !matches!(indices.dtype, DataType::Int32 | DataType::Int64) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GatherND: indices must be Int32 or Int64, got {:?}",
                indices.dtype
            )));
        }
        if outputs[0].dtype != data.dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherND: output dtype must match data".into(),
            ));
        }
        let element_bytes = fixed_width("GatherND", data.dtype)?;
        let batch_dims = usize::try_from(self.batch_dims)
            .map_err(|_| EpError::KernelFailed("cuda_ep GatherND: invalid batch_dims".into()))?;
        if batch_dims > data.shape.len() || batch_dims >= indices.shape.len() {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherND: invalid batch_dims".into(),
            ));
        }
        if data.shape[..batch_dims] != indices.shape[..batch_dims] {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherND: batch dimensions must match".into(),
            ));
        }
        let tuple_width = *indices.shape.last().unwrap();
        if tuple_width > data.shape.len() - batch_dims {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherND: index tuple is longer than data suffix rank".into(),
            ));
        }
        let expected_shape: Vec<usize> = data.shape[..batch_dims]
            .iter()
            .chain(&indices.shape[batch_dims..indices.shape.len() - 1])
            .chain(&data.shape[batch_dims + tuple_width..])
            .copied()
            .collect();
        if outputs[0].shape != expected_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GatherND: output shape {:?}, expected {expected_shape:?}",
                outputs[0].shape
            )));
        }

        let batches = product(&data.shape[..batch_dims], "GatherND")?;
        let tuples_per_batch = product(
            &indices.shape[batch_dims..indices.shape.len() - 1],
            "GatherND",
        )?;
        let data_batch_length = product(&data.shape[batch_dims..], "GatherND")?;
        let tail_length = product(&data.shape[batch_dims + tuple_width..], "GatherND")?;
        let elements = outputs[0].numel();
        let expected_elements = batches
            .checked_mul(tuples_per_batch)
            .and_then(|value| value.checked_mul(tail_length))
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep GatherND: output size overflow".into())
            })?;
        if elements != expected_elements {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GatherND: output has {elements} elements, expected {expected_elements}"
            )));
        }
        if elements == 0 {
            return Ok(());
        }

        let capturing = self.runtime.is_capturing()?;
        if !capturing && !self.runtime.eager_sync_deferred() {
            let mut bytes = vec![0u8; indices.dtype.storage_bytes(indices.numel())];
            unsafe {
                self.runtime
                    .dtoh(&mut bytes, cuptr(indices.data_ptr::<u8>() as *const c_void))?
            };
            for (linear, raw_bytes) in bytes.chunks_exact(indices.dtype.byte_size()).enumerate() {
                let raw = match indices.dtype {
                    DataType::Int32 => i32::from_ne_bytes(raw_bytes.try_into().unwrap()) as i64,
                    DataType::Int64 => i64::from_ne_bytes(raw_bytes.try_into().unwrap()),
                    _ => unreachable!("validated above"),
                };
                let dimension = linear % tuple_width;
                let size = data.shape[batch_dims + dimension] as i64;
                let normalized = if raw < 0 { raw + size } else { raw };
                if normalized < 0 || normalized >= size {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep GatherND: index {raw} out of range at tuple dimension {dimension}"
                    )));
                }
            }
        }

        let dimensions = data
            .shape
            .iter()
            .map(|&dimension| dimension as u64)
            .collect::<Vec<_>>();
        let mut dimensions_cache = self.dimensions.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep GatherND: metadata lock poisoned".into())
        })?;
        let dimensions_candidate = dimensions_cache.stage(&dimensions, "GatherND")?;
        let dimension_ptr = dimensions_candidate.ptr("GatherND")?;

        let function = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "gather_nd_bytes")?;
        let data_ptr = cuptr(data.data_ptr::<u8>() as *const c_void);
        let indices_ptr = cuptr(indices.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let data_rank = data.shape.len() as i32;
        let batch_dims = batch_dims as i32;
        let tuples_per_batch = tuples_per_batch as u64;
        let tuple_width = tuple_width as u64;
        let tail_length = tail_length as u64;
        let data_batch_length = data_batch_length as u64;
        let element_bytes = element_bytes as i32;
        let index_is_i64 = i32::from(indices.dtype == DataType::Int64);
        let elements_u64 = elements as u64;
        let capture_error = if capturing || self.runtime.eager_sync_deferred() {
            self.runtime.capture_error_ptr()
        } else {
            0
        };
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&data_ptr)
            .arg(&indices_ptr)
            .arg(&output_ptr)
            .arg(&dimension_ptr)
            .arg(&data_rank)
            .arg(&batch_dims)
            .arg(&tuples_per_batch)
            .arg(&tuple_width)
            .arg(&tail_length)
            .arg(&data_batch_length)
            .arg(&element_bytes)
            .arg(&index_is_i64)
            .arg(&elements_u64)
            .arg(&capture_error);
        let launch = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid(elements), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch gather_nd_bytes", error));
        let sync = if capturing {
            Ok(())
        } else {
            self.runtime.synchronize()
        };
        launch.and(sync)?;
        if !capturing {
            *dimensions_cache = dimensions_candidate;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }

    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        self.dimensions
            .lock()
            .ok()
            .and_then(|dimensions| dimensions.device_graph_resource())
            .into_iter()
            .collect()
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.dimensions.lock() {
            Ok(dimensions) if dimensions.device_graph_resource().is_some() => {
                onnx_runtime_ep_api::CaptureSupport::Supported
            }
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "GatherND must warm its exact shape metadata before capture",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "GatherND metadata lock was poisoned",
            ),
        }
    }
}

pub struct SpaceToDepthFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for SpaceToDepthFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let block_size = node
            .attr("blocksize")
            .and_then(Attribute::as_int)
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep SpaceToDepth: blocksize is required".into())
            })?;
        if block_size <= 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep SpaceToDepth: blocksize must be positive".into(),
            ));
        }
        Ok(Box::new(SpaceToDepthKernel {
            block_size: block_size as usize,
            runtime: self.runtime.clone(),
        }))
    }
}

struct SpaceToDepthKernel {
    block_size: usize,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for SpaceToDepthKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep SpaceToDepth: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        require_dense("SpaceToDepth", inputs, outputs)?;
        let input = &inputs[0];
        if input.shape.len() != 4 {
            return Err(EpError::KernelFailed(
                "cuda_ep SpaceToDepth: input must have rank 4".into(),
            ));
        }
        let [batch, channels, height, width] =
            <[usize; 4]>::try_from(input.shape).expect("rank checked above");
        let block = self.block_size;
        if height % block != 0 || width % block != 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep SpaceToDepth: spatial dimensions {height}x{width} must be divisible by blocksize {block}"
            )));
        }
        let output_shape = [
            batch,
            channels * block * block,
            height / block,
            width / block,
        ];
        if outputs[0].dtype != input.dtype || outputs[0].shape != output_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep SpaceToDepth: output must have dtype {:?} and shape {output_shape:?}",
                input.dtype
            )));
        }
        let elements = outputs[0].numel();
        if elements == 0 {
            return Ok(());
        }
        let function = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "space_to_depth_bytes")?;
        let input_ptr = cuptr(input.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let channels = channels as u64;
        let height = height as u64;
        let width = width as u64;
        let block = block as u64;
        let element_bytes = fixed_width("SpaceToDepth", input.dtype)? as i32;
        let elements_u64 = elements as u64;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_ptr)
            .arg(&output_ptr)
            .arg(&channels)
            .arg(&height)
            .arg(&width)
            .arg(&block)
            .arg(&element_bytes)
            .arg(&elements_u64);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid(elements), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch space_to_depth_bytes", error))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
}

pub struct EyeLikeFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for EyeLikeFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = node.attr("k").and_then(Attribute::as_int).unwrap_or(0);
        let dtype = match node.attr("dtype") {
            None => None,
            Some(Attribute::Int(value)) => {
                let value = i32::try_from(*value).map_err(|_| {
                    EpError::KernelFailed(format!("cuda_ep EyeLike: invalid dtype {value}"))
                })?;
                Some(DataType::from_onnx(value).ok_or_else(|| {
                    EpError::KernelFailed(format!("cuda_ep EyeLike: invalid dtype {value}"))
                })?)
            }
            Some(_) => {
                return Err(EpError::KernelFailed(
                    "cuda_ep EyeLike: dtype must be an integer".into(),
                ));
            }
        };
        Ok(Box::new(EyeLikeKernel {
            k,
            dtype,
            runtime: self.runtime.clone(),
        }))
    }
}

struct EyeLikeKernel {
    k: i64,
    dtype: Option<DataType>,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for EyeLikeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep EyeLike: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        if inputs[0].shape.len() != 2 || outputs[0].shape != inputs[0].shape {
            return Err(EpError::KernelFailed(
                "cuda_ep EyeLike: input must be rank 2 and output shape must match".into(),
            ));
        }
        let dtype = self.dtype.unwrap_or(inputs[0].dtype);
        if outputs[0].dtype != dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep EyeLike: output dtype {:?}, expected {dtype:?}",
                outputs[0].dtype
            )));
        }
        let (element_bytes, one_bits) = eye_storage(dtype)?;
        let elements = outputs[0].numel();
        if elements == 0 {
            return Ok(());
        }
        let function = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "eye_like_bytes")?;
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let columns = inputs[0].shape[1] as u64;
        let element_bytes = element_bytes as i32;
        let elements_u64 = elements as u64;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&output_ptr)
            .arg(&columns)
            .arg(&self.k)
            .arg(&one_bits)
            .arg(&element_bytes)
            .arg(&elements_u64);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid(elements), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch eye_like_bytes", error))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

fn eye_storage(dtype: DataType) -> Result<(usize, u64)> {
    let storage = match dtype {
        DataType::Bool | DataType::Int8 | DataType::Uint8 => (1, 1),
        DataType::Int16 | DataType::Uint16 => (2, 1),
        DataType::Int32 | DataType::Uint32 => (4, 1),
        DataType::Int64 | DataType::Uint64 => (8, 1),
        DataType::Float16 => (2, 0x3c00),
        DataType::BFloat16 => (2, 0x3f80),
        DataType::Float32 => (4, 1.0f32.to_bits() as u64),
        DataType::Float64 => (8, 1.0f64.to_bits()),
        other => {
            return Err(not_implemented(format!(
                "EyeLike with output dtype {other:?}"
            )));
        }
    };
    Ok(storage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_storage_matches_numeric_one() {
        assert_eq!(eye_storage(DataType::Float16).unwrap(), (2, 0x3c00));
        assert_eq!(
            eye_storage(DataType::Float32).unwrap(),
            (4, 1.0f32.to_bits() as u64)
        );
        assert_eq!(eye_storage(DataType::Int64).unwrap(), (8, 1));
    }
}

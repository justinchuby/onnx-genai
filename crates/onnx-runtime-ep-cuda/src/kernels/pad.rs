//! Dtype-agnostic CUDA implementation of ONNX `Pad`.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node, compute_contiguous_strides};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
extern "C" __global__ void pad_bytes(
    const unsigned char* input, unsigned char* output,
    const unsigned char* metadata_bytes, int rank, int mode,
    int element_bytes, unsigned long long elements) {
  const long long* begin = (const long long*)metadata_bytes;
  const unsigned long long* input_dimensions =
      (const unsigned long long*)(metadata_bytes + rank * sizeof(long long));
  const unsigned long long* input_strides =
      (const unsigned long long*)(metadata_bytes + rank * 2 * sizeof(long long));
  const unsigned long long* output_strides =
      (const unsigned long long*)(metadata_bytes + rank * 3 * sizeof(long long));
  const unsigned char* fill = metadata_bytes + rank * 4 * sizeof(long long);
  for (unsigned long long output_index = blockIdx.x * blockDim.x + threadIdx.x;
       output_index < elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long remaining = output_index;
    unsigned long long input_index = 0;
    bool in_range = true;
    for (int axis = 0; axis < rank; ++axis) {
      const unsigned long long stride = output_strides[axis];
      const unsigned long long coordinate = remaining / stride;
      remaining %= stride;
      long long source = (long long)coordinate - begin[axis];
      const long long dimension = (long long)input_dimensions[axis];
      if (mode == 0) {
        if (source < 0 || source >= dimension) {
          in_range = false;
          break;
        }
      } else if (mode == 1) {
        if (dimension == 1) {
          source = 0;
        } else {
          const long long period = 2 * (dimension - 1);
          source %= period;
          if (source < 0) source += period;
          if (source >= dimension) source = period - source;
        }
      } else if (mode == 2) {
        source = source < 0 ? 0 : (source >= dimension ? dimension - 1 : source);
      } else {
        source %= dimension;
        if (source < 0) source += dimension;
      }
      input_index += (unsigned long long)source * input_strides[axis];
    }
    const unsigned char* source =
        in_range ? input + input_index * element_bytes : fill;
    for (int byte = 0; byte < element_bytes; ++byte)
      output[output_index * element_bytes + byte] = source[byte];
  }
}
"#;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PadMode {
    Constant,
    Reflect,
    Edge,
    Wrap,
}

impl PadMode {
    fn code(self) -> i32 {
        match self {
            Self::Constant => 0,
            Self::Reflect => 1,
            Self::Edge => 2,
            Self::Wrap => 3,
        }
    }
}

pub struct PadFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for PadFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let mode = match node.attr("mode").and_then(Attribute::as_str) {
            None | Some("constant") => PadMode::Constant,
            Some("reflect") => PadMode::Reflect,
            Some("edge") => PadMode::Edge,
            Some("wrap") => PadMode::Wrap,
            Some(other) => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Pad: unsupported mode {other:?}"
                )));
            }
        };
        Ok(Box::new(PadKernel {
            runtime: self.runtime.clone(),
            mode,
            pads_attr: node
                .attr("pads")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec),
            value_attr: node.attr("value").and_then(Attribute::as_float),
        }))
    }
}

struct PadKernel {
    runtime: Arc<CudaRuntime>,
    mode: PadMode,
    pads_attr: Option<Vec<i64>>,
    value_attr: Option<f32>,
}

fn device_bytes(runtime: &CudaRuntime, input: &TensorView) -> Result<Vec<u8>> {
    if !input.is_contiguous() {
        return Err(not_implemented("Pad with strided metadata input"));
    }
    let mut bytes = vec![0; input.dtype.storage_bytes(input.numel())];
    if !bytes.is_empty() {
        unsafe { runtime.dtoh(&mut bytes, cuptr(input.data_ptr::<u8>() as *const c_void))? };
    }
    Ok(bytes)
}

fn device_i64(runtime: &CudaRuntime, input: &TensorView, name: &str) -> Result<Vec<i64>> {
    if input.dtype != DataType::Int64 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Pad: {name} must have Int64 dtype"
        )));
    }
    Ok(device_bytes(runtime, input)?
        .chunks_exact(8)
        .map(|bytes| i64::from_ne_bytes(bytes.try_into().unwrap()))
        .collect())
}

fn legacy_fill(dtype: DataType, value: f32) -> Vec<u8> {
    match dtype {
        DataType::Float32 => value.to_ne_bytes().to_vec(),
        DataType::Float64 => (value as f64).to_ne_bytes().to_vec(),
        DataType::Float16 => half::f16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
        DataType::BFloat16 => half::bf16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
        DataType::Int64 => (value as i64).to_ne_bytes().to_vec(),
        DataType::Int32 => (value as i32).to_ne_bytes().to_vec(),
        DataType::Int16 => (value as i16).to_ne_bytes().to_vec(),
        DataType::Int8 => vec![value as i8 as u8],
        DataType::Uint64 => (value as u64).to_ne_bytes().to_vec(),
        DataType::Uint32 => (value as u32).to_ne_bytes().to_vec(),
        DataType::Uint16 => (value as u16).to_ne_bytes().to_vec(),
        DataType::Uint8 => vec![value as u8],
        DataType::Bool => vec![u8::from(value != 0.0)],
        other => vec![0; other.byte_size()],
    }
}

impl Kernel for PadKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.is_empty() || inputs.len() > 4 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Pad: expected 1..=4 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        if self.runtime.is_capturing()? {
            return Err(not_implemented(
                "Pad during CUDA graph capture because pads and axes are device inputs",
            ));
        }
        let data = &inputs[0];
        let output = &mut outputs[0];
        if !data.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented("Pad with non-contiguous data or output"));
        }
        if output.dtype != data.dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep Pad: output dtype must match data".into(),
            ));
        }
        let element_bytes = data.dtype.byte_size();
        if element_bytes == 0 {
            return Err(not_implemented("Pad for packed or variable-width dtype"));
        }
        let rank = data.shape.len();
        let pads = if inputs.get(1).is_some_and(|input| !input.is_absent()) {
            device_i64(&self.runtime, &inputs[1], "pads")?
        } else {
            self.pads_attr.clone().ok_or_else(|| {
                EpError::KernelFailed("cuda_ep Pad: missing pads input or attribute".into())
            })?
        };
        let axes = if inputs.get(3).is_some_and(|input| !input.is_absent()) {
            Some(device_i64(&self.runtime, &inputs[3], "axes")?)
        } else {
            None
        };
        let axis_count = axes.as_ref().map_or(rank, Vec::len);
        if pads.len() != 2 * axis_count {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Pad: pads has {} values, expected {}",
                pads.len(),
                2 * axis_count
            )));
        }
        let mut begin = vec![0_i64; rank];
        let mut end = vec![0_i64; rank];
        for index in 0..axis_count {
            let raw = axes.as_ref().map_or(index as i64, |values| values[index]);
            let normalized = if raw < 0 { raw + rank as i64 } else { raw };
            if normalized < 0 || normalized as usize >= rank {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Pad: axis {raw} is out of range for rank {rank}"
                )));
            }
            begin[normalized as usize] = pads[index];
            end[normalized as usize] = pads[axis_count + index];
        }
        let expected_shape = data
            .shape
            .iter()
            .enumerate()
            .map(|(axis, &dimension)| {
                let padded = dimension as i64 + begin[axis] + end[axis];
                if padded < 0 {
                    Err(EpError::KernelFailed(format!(
                        "cuda_ep Pad: pads crop axis {axis} past its extent"
                    )))
                } else if self.mode != PadMode::Constant && padded > 0 && dimension == 0 {
                    Err(EpError::KernelFailed(format!(
                        "cuda_ep Pad: non-constant mode cannot sample empty axis {axis}"
                    )))
                } else {
                    Ok(padded as usize)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        if output.shape != expected_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Pad: output shape {:?}, expected {expected_shape:?}",
                output.shape
            )));
        }
        if output.numel() == 0 {
            return Ok(());
        }

        let fill = if inputs.get(2).is_some_and(|input| !input.is_absent()) {
            if inputs[2].dtype != data.dtype || inputs[2].numel() != 1 {
                return Err(EpError::KernelFailed(
                    "cuda_ep Pad: constant_value must be a scalar with the data dtype".into(),
                ));
            }
            device_bytes(&self.runtime, &inputs[2])?
        } else if let Some(value) = self.value_attr {
            legacy_fill(data.dtype, value)
        } else {
            vec![0; element_bytes]
        };

        let input_strides = compute_contiguous_strides(data.shape);
        let output_strides = compute_contiguous_strides(output.shape);
        let mut metadata = Vec::with_capacity(rank * 32 + element_bytes);
        for value in &begin {
            metadata.extend_from_slice(&value.to_ne_bytes());
        }
        for &value in data.shape {
            metadata.extend_from_slice(&(value as u64).to_ne_bytes());
        }
        for value in input_strides {
            metadata.extend_from_slice(&(value as u64).to_ne_bytes());
        }
        for value in output_strides {
            metadata.extend_from_slice(&(value as u64).to_ne_bytes());
        }
        metadata.extend_from_slice(&fill[..element_bytes]);

        let function = self.runtime.nvrtc_function("pad", SOURCE, "pad_bytes")?;
        let rank = i32::try_from(rank)
            .map_err(|_| EpError::KernelFailed("cuda_ep Pad: rank exceeds i32".into()))?;
        let element_bytes = i32::try_from(element_bytes)
            .map_err(|_| EpError::KernelFailed("cuda_ep Pad: element width exceeds i32".into()))?;
        let metadata_ptr = self.runtime.alloc_raw(metadata.len())?;
        if let Err(error) = unsafe { self.runtime.htod(&metadata, metadata_ptr) } {
            let _ = unsafe { self.runtime.free_raw(metadata_ptr) };
            return Err(error);
        }
        let input_ptr = cuptr(data.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let mode = self.mode.code();
        let elements = output.numel() as u64;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_ptr)
            .arg(&output_ptr)
            .arg(&metadata_ptr)
            .arg(&rank)
            .arg(&mode)
            .arg(&element_bytes)
            .arg(&elements);
        let launch = unsafe {
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
        .map_err(|error| driver_err("launch Pad", error));
        let sync = launch.and_then(|_| self.runtime.synchronize());
        let free = unsafe { self.runtime.free_raw(metadata_ptr) };
        sync.and(free)
    }
}

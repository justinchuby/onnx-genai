//! Capture-safe CUDA implementation of standard-domain `OneHot`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const MODULE: &str = "onehot";
const SOURCE: &str = r#"
template <typename Index>
__device__ void onehot_impl(
    const Index* indices, const unsigned char* values, unsigned char* output,
    unsigned long long elements, unsigned long long inner,
    unsigned long long depth, int elem_bytes, int wrap_negative) {
  for (unsigned long long out = blockIdx.x * blockDim.x + threadIdx.x;
       out < elements;
       out += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long outer = out / (inner * depth);
    unsigned long long inner_index = out % inner;
    unsigned long long index_linear = outer * inner + inner_index;
    long long raw = (long long)indices[index_linear];
    unsigned long long selected = 0;
    bool valid;
    if (raw >= 0) {
      selected = (unsigned long long)raw;
      valid = selected < depth;
    } else if (wrap_negative) {
      unsigned long long magnitude = 0ull - (unsigned long long)raw;
      valid = magnitude <= depth;
      selected = depth - magnitude;
    } else {
      valid = false;
    }
    unsigned long long category = (out / inner) % depth;
    int value_index = valid && selected == category ? 1 : 0;
    for (int byte = 0; byte < elem_bytes; ++byte)
      output[out * elem_bytes + byte] = values[value_index * elem_bytes + byte];
  }
}

extern "C" __global__ void onehot_i32(
    const int* indices, const unsigned char* values, unsigned char* output,
    unsigned long long elements, unsigned long long inner,
    unsigned long long depth, int elem_bytes, int wrap_negative) {
  onehot_impl(indices, values, output, elements, inner, depth, elem_bytes, wrap_negative);
}

extern "C" __global__ void onehot_i64(
    const long long* indices, const unsigned char* values, unsigned char* output,
    unsigned long long elements, unsigned long long inner,
    unsigned long long depth, int elem_bytes, int wrap_negative) {
  onehot_impl(indices, values, output, elements, inner, depth, elem_bytes, wrap_negative);
}
"#;

#[derive(Clone, Debug, PartialEq, Eq)]
struct CaptureSignature {
    indices_dtype: DataType,
    values_dtype: DataType,
    indices_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

pub struct OneHotFactory {
    pub runtime: Arc<CudaRuntime>,
    pub wrap_negative: bool,
}

impl KernelFactory for OneHotFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(OneHotKernel {
            runtime: self.runtime.clone(),
            axis: node.attr("axis").and_then(Attribute::as_int).unwrap_or(-1),
            wrap_negative: self.wrap_negative,
            warmed_signature: Mutex::new(None),
        }))
    }
}

struct OneHotKernel {
    runtime: Arc<CudaRuntime>,
    axis: i64,
    wrap_negative: bool,
    warmed_signature: Mutex<Option<CaptureSignature>>,
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1_usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| EpError::KernelFailed("cuda_ep OneHot: shape product overflow".into()))
    })
}

impl Kernel for OneHotKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep OneHot: expected 3 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        if inputs.iter().any(|input| !input.is_contiguous()) || !outputs[0].is_contiguous() {
            return Err(not_implemented("OneHot with non-contiguous tensors"));
        }
        let indices = &inputs[0];
        let depth_input = &inputs[1];
        let values = &inputs[2];
        let output = &mut outputs[0];
        if !matches!(indices.dtype, DataType::Int32 | DataType::Int64)
            || !matches!(depth_input.dtype, DataType::Int32 | DataType::Int64)
        {
            return Err(EpError::KernelFailed(
                "cuda_ep OneHot: indices and depth must be Int32 or Int64".into(),
            ));
        }
        if depth_input.numel() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep OneHot: depth must be a scalar".into(),
            ));
        }
        if values.shape != [2] || output.dtype != values.dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep OneHot: values must have shape [2] and output must match its dtype".into(),
            ));
        }
        let elem_bytes = values.dtype.byte_size();
        if elem_bytes == 0 {
            return Err(not_implemented(format!(
                "OneHot values dtype {:?} is packed or variable-width",
                values.dtype
            )));
        }

        let output_rank = indices.shape.len() + 1;
        let axis = if self.axis < 0 {
            self.axis + output_rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize >= output_rank {
            return Err(EpError::KernelFailed(
                "cuda_ep OneHot: axis out of range".into(),
            ));
        }
        let axis = axis as usize;
        if output.shape.len() != output_rank
            || output.shape[..axis] != indices.shape[..axis]
            || output.shape[axis + 1..] != indices.shape[axis..]
        {
            return Err(EpError::KernelFailed(
                "cuda_ep OneHot: output shape does not match indices shape and axis".into(),
            ));
        }
        let depth = output.shape[axis];
        let inner = checked_product(&indices.shape[axis..])?;
        let elements = checked_product(output.shape)?;

        let capturing = self.runtime.is_capturing()?;
        let mut warmed = self.warmed_signature.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep OneHot: capture signature lock was poisoned".into())
        })?;
        if capturing
            && !warmed.as_ref().is_some_and(|signature| {
                signature.indices_dtype == indices.dtype
                    && signature.values_dtype == values.dtype
                    && signature.indices_shape == indices.shape
                    && signature.output_shape == output.shape
            })
        {
            return Err(EpError::KernelFailed(
                "cuda_ep OneHot: shape/dtype changed during CUDA graph capture; warm the exact signature first".into(),
            ));
        }

        if elements != 0 {
            let entry = match indices.dtype {
                DataType::Int32 => "onehot_i32",
                DataType::Int64 => "onehot_i64",
                _ => unreachable!(),
            };
            let function = self.runtime.nvrtc_function(MODULE, SOURCE, entry)?;
            let indices_ptr = cuptr(indices.data_ptr::<u8>() as *const c_void);
            let values_ptr = cuptr(values.data_ptr::<u8>() as *const c_void);
            let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
            let elements = u64::try_from(elements)
                .map_err(|_| EpError::KernelFailed("cuda_ep OneHot: elements exceed u64".into()))?;
            let inner = u64::try_from(inner).map_err(|_| {
                EpError::KernelFailed("cuda_ep OneHot: inner size exceeds u64".into())
            })?;
            let depth = u64::try_from(depth)
                .map_err(|_| EpError::KernelFailed("cuda_ep OneHot: depth exceeds u64".into()))?;
            let elem_bytes = i32::try_from(elem_bytes).unwrap();
            let wrap_negative = i32::from(self.wrap_negative);
            let config = LaunchConfig {
                grid_dim: (
                    elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = self.runtime.stream().launch_builder(&function);
            builder
                .arg(&indices_ptr)
                .arg(&values_ptr)
                .arg(&output_ptr)
                .arg(&elements)
                .arg(&inner)
                .arg(&depth)
                .arg(&elem_bytes)
                .arg(&wrap_negative);
            // SAFETY: all pointers and byte widths match the validated tensors.
            unsafe { builder.launch(config) }
                .map_err(|error| driver_err(&format!("launch {entry}"), error))?;
        }
        if !capturing {
            *warmed = Some(CaptureSignature {
                indices_dtype: indices.dtype,
                values_dtype: values.dtype,
                indices_shape: indices.shape.to_vec(),
                output_shape: output.shape.to_vec(),
            });
        }
        Ok(())
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.warmed_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "OneHot must warm its exact shape/dtype signature before capture",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "OneHot capture signature lock was poisoned",
            ),
        }
    }
}

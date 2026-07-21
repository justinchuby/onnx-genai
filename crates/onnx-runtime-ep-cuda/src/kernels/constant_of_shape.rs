//! Capture-safe CUDA implementation of standard-domain `ConstantOfShape`.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const MODULE: &str = "constant_of_shape";
const SOURCE: &str = r#"
extern "C" __global__ void fill_pattern(
    unsigned char* output, unsigned long long bytes, int pattern_bytes,
    unsigned long long pattern_lo, unsigned long long pattern_hi) {
  for (unsigned long long offset = blockIdx.x * blockDim.x + threadIdx.x;
       offset < bytes;
       offset += (unsigned long long)gridDim.x * blockDim.x) {
    int pattern_offset = (int)(offset % (unsigned long long)pattern_bytes);
    unsigned long long word = pattern_offset < 8 ? pattern_lo : pattern_hi;
    int shift = (pattern_offset & 7) * 8;
    output[offset] = (unsigned char)(word >> shift);
  }
}
"#;

#[derive(Clone, Debug, PartialEq, Eq)]
struct CaptureSignature {
    dtype: DataType,
    shape: Vec<usize>,
}

pub struct ConstantOfShapeFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for ConstantOfShapeFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let (dtype, pattern, pattern_bytes) = decode_value(node)?;
        Ok(Box::new(ConstantOfShapeKernel {
            runtime: self.runtime.clone(),
            dtype,
            pattern,
            pattern_bytes,
            warmed_signature: Mutex::new(None),
        }))
    }
}

struct ConstantOfShapeKernel {
    runtime: Arc<CudaRuntime>,
    dtype: DataType,
    pattern: [u8; 16],
    pattern_bytes: usize,
    warmed_signature: Mutex<Option<CaptureSignature>>,
}

fn is_supported_dtype(dtype: DataType) -> bool {
    dtype.is_float() || dtype.is_int() || dtype == DataType::Bool
}

fn decode_value(node: &Node) -> Result<(DataType, [u8; 16], usize)> {
    let Some(attribute) = node.attr("value") else {
        let mut pattern = [0_u8; 16];
        pattern[..4].copy_from_slice(&0_f32.to_le_bytes());
        return Ok((DataType::Float32, pattern, 4));
    };
    let Attribute::Tensor(tensor) = attribute else {
        return Err(EpError::KernelFailed(
            "cuda_ep ConstantOfShape: value must be a one-element tensor".into(),
        ));
    };
    if tensor.numel() != 1 || !is_supported_dtype(tensor.dtype) {
        return Err(not_implemented(format!(
            "ConstantOfShape value dtype {:?} or element count {} (expected one numeric/bool element)",
            tensor.dtype,
            tensor.numel()
        )));
    }
    let expected = tensor.dtype.storage_bytes(1);
    if tensor.data.len() != expected {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep ConstantOfShape: value tensor has {} bytes, expected {expected}",
            tensor.data.len()
        )));
    }

    let mut pattern = [0_u8; 16];
    if tensor.dtype.is_sub_byte() {
        let bit_size = tensor.dtype.bit_size();
        let mask = (1_u8 << bit_size) - 1;
        let scalar = tensor.data[0] & mask;
        pattern[0] = match bit_size {
            4 => scalar | (scalar << 4),
            2 => scalar * 0x55,
            _ => unreachable!("ONNX sub-byte dtypes are two or four bits"),
        };
        Ok((tensor.dtype, pattern, 1))
    } else {
        let bytes = tensor.dtype.byte_size();
        pattern[..bytes].copy_from_slice(&tensor.data);
        Ok((tensor.dtype, pattern, bytes))
    }
}

impl Kernel for ConstantOfShapeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep ConstantOfShape: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let shape = &inputs[0];
        let output = &mut outputs[0];
        if shape.dtype != DataType::Int64 || shape.shape.len() != 1 || !shape.is_contiguous() {
            return Err(EpError::KernelFailed(
                "cuda_ep ConstantOfShape: shape input must be contiguous rank-1 Int64".into(),
            ));
        }
        if output.dtype != self.dtype || !output.is_contiguous() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep ConstantOfShape: output must be contiguous {:?}, got {:?}",
                self.dtype, output.dtype
            )));
        }

        let capturing = self.runtime.is_capturing()?;
        let mut warmed = self.warmed_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep ConstantOfShape: capture signature lock was poisoned".into(),
            )
        })?;
        if capturing
            && !warmed.as_ref().is_some_and(|signature| {
                signature.dtype == output.dtype && signature.shape == output.shape
            })
        {
            return Err(EpError::KernelFailed(
                "cuda_ep ConstantOfShape: output shape/dtype changed during CUDA graph capture; warm the exact signature first".into(),
            ));
        }

        let bytes = output
            .dtype
            .checked_storage_bytes(output.numel())
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep ConstantOfShape: output storage size overflow".into(),
                )
            })?;
        if bytes != 0 {
            let entry = "fill_pattern";
            let function = self.runtime.nvrtc_function(MODULE, SOURCE, entry)?;
            let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
            let bytes = u64::try_from(bytes).map_err(|_| {
                EpError::KernelFailed("cuda_ep ConstantOfShape: byte count exceeds u64".into())
            })?;
            let pattern_bytes = i32::try_from(self.pattern_bytes).unwrap();
            let pattern_lo = u64::from_le_bytes(self.pattern[..8].try_into().unwrap());
            let pattern_hi = u64::from_le_bytes(self.pattern[8..].try_into().unwrap());
            let config = LaunchConfig {
                grid_dim: (bytes.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = self.runtime.stream().launch_builder(&function);
            builder
                .arg(&output_ptr)
                .arg(&bytes)
                .arg(&pattern_bytes)
                .arg(&pattern_lo)
                .arg(&pattern_hi);
            // SAFETY: the output allocation covers `bytes`, and all remaining
            // arguments are scalar launch metadata.
            unsafe { builder.launch(config) }
                .map_err(|error| driver_err("launch ConstantOfShape fill_pattern", error))?;
        }
        if !capturing {
            *warmed = Some(CaptureSignature {
                dtype: output.dtype,
                shape: output.shape.to_vec(),
            });
        }
        Ok(())
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.warmed_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "ConstantOfShape must warm its exact output shape/dtype before capture",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "ConstantOfShape capture signature lock was poisoned",
            ),
        }
    }
}

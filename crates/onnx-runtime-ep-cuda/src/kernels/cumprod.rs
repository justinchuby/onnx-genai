//! Deterministic per-lane CUDA implementation of ONNX `CumProd` (opset 26),
//! mirroring the `CumSum` scan (`cumsum.rs`) with a multiplicative accumulator.
//! Supports f32 and i64, honouring the `exclusive` and `reverse` attributes.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
extern "C" __global__ void cumprod_f32(
    const float* input, float* output, unsigned long long lanes,
    unsigned long long width, unsigned long long inner, int exclusive, int reverse) {
  for (unsigned long long lane = blockIdx.x * blockDim.x + threadIdx.x; lane < lanes;
       lane += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long outer = lane / inner, i = lane % inner;
    float total = 1.0f;
    for (unsigned long long n = 0; n < width; ++n) {
      unsigned long long d = reverse ? width - 1 - n : n;
      unsigned long long offset = (outer * width + d) * inner + i;
      float value = input[offset];
      if (exclusive) { output[offset] = total; total *= value; }
      else { total *= value; output[offset] = total; }
    }
  }
}
extern "C" __global__ void cumprod_i64(
    const long long* input, long long* output, unsigned long long lanes,
    unsigned long long width, unsigned long long inner, int exclusive, int reverse) {
  for (unsigned long long lane = blockIdx.x * blockDim.x + threadIdx.x; lane < lanes;
       lane += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long outer = lane / inner, i = lane % inner;
    unsigned long long total = 1;
    for (unsigned long long n = 0; n < width; ++n) {
      unsigned long long d = reverse ? width - 1 - n : n;
      unsigned long long offset = (outer * width + d) * inner + i;
      unsigned long long value = (unsigned long long)input[offset];
      if (exclusive) { output[offset] = (long long)total; total *= value; }
      else { total *= value; output[offset] = (long long)total; }
    }
  }
}
"#;

pub struct CumProdFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for CumProdFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let bool_attr = |name: &str| -> Result<bool> {
            match node.attr(name) {
                None | Some(onnx_runtime_ir::Attribute::Int(0)) => Ok(false),
                Some(onnx_runtime_ir::Attribute::Int(1)) => Ok(true),
                Some(_) => Err(EpError::KernelFailed(format!(
                    "cuda_ep CumProd: {name} must be 0 or 1"
                ))),
            }
        };
        Ok(Box::new(CumProdKernel {
            runtime: self.runtime.clone(),
            exclusive: bool_attr("exclusive")?,
            reverse: bool_attr("reverse")?,
            warmed_signature: Mutex::new(None),
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CumProdCaptureSignature {
    dtype: DataType,
    shape: Vec<usize>,
    axis: usize,
}

struct CumProdKernel {
    runtime: Arc<CudaRuntime>,
    exclusive: bool,
    reverse: bool,
    warmed_signature: Mutex<Option<CumProdCaptureSignature>>,
}

impl Kernel for CumProdKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep CumProd: expected 2 inputs and 1 output".into(),
            ));
        }
        let input = &inputs[0];
        let axis_input = &inputs[1];
        let output = &mut outputs[0];
        if !input.is_contiguous() || !axis_input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented("CumProd with non-contiguous tensors"));
        }
        if output.dtype != input.dtype || output.shape != input.shape {
            return Err(EpError::KernelFailed(
                "cuda_ep CumProd: output must match input shape and dtype".into(),
            ));
        }
        if !matches!(input.dtype, DataType::Float32 | DataType::Int64) {
            return Err(not_implemented(format!(
                "CumProd supports Float32 and Int64, got {:?}",
                input.dtype
            )));
        }
        if axis_input.dtype != DataType::Int64 || axis_input.numel() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep CumProd: axis must be an Int64 scalar".into(),
            ));
        }
        let capturing = self.runtime.is_capturing()?;
        let mut warmed_signature = self.warmed_signature.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep CumProd: capture signature lock was poisoned".into())
        })?;
        let axis = if capturing {
            let signature = warmed_signature.as_ref().ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep CumProd: capture started before the fixed axis was warmed".into(),
                )
            })?;
            if signature.dtype != input.dtype || signature.shape != input.shape {
                return Err(EpError::KernelFailed(
                    "cuda_ep CumProd: shape or dtype changed during CUDA graph capture".into(),
                ));
            }
            signature.axis
        } else {
            let mut bytes = [0_u8; 8];
            unsafe {
                self.runtime.dtoh(
                    &mut bytes,
                    cuptr(axis_input.data_ptr::<i64>() as *const c_void),
                )?
            };
            let raw = i64::from_ne_bytes(bytes);
            let rank = input.shape.len();
            let normalized = if raw < 0 { raw + rank as i64 } else { raw };
            if normalized < 0 || normalized as usize >= rank {
                return Err(EpError::KernelFailed(
                    "cuda_ep CumProd: axis out of range".into(),
                ));
            }
            normalized as usize
        };
        if output.numel() == 0 {
            if !capturing {
                *warmed_signature = Some(CumProdCaptureSignature {
                    dtype: input.dtype,
                    shape: input.shape.to_vec(),
                    axis,
                });
            }
            return Ok(());
        }
        let inner = input.shape[axis + 1..].iter().product::<usize>();
        let width = input.shape[axis];
        let outer = input.shape[..axis].iter().product::<usize>();
        let lanes = outer * inner;
        let entry = if input.dtype == DataType::Float32 {
            "cumprod_f32"
        } else {
            "cumprod_i64"
        };
        let func = self.runtime.nvrtc_function("cumprod", SOURCE, entry)?;
        let input_ptr = cuptr(input.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let lanes = lanes as u64;
        let width = width as u64;
        let inner = inner as u64;
        let exclusive = i32::from(self.exclusive);
        let reverse = i32::from(self.reverse);
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&input_ptr)
            .arg(&output_ptr)
            .arg(&lanes)
            .arg(&width)
            .arg(&inner)
            .arg(&exclusive)
            .arg(&reverse);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: ((lanes.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch CumProd", e))?;
        if !capturing {
            *warmed_signature = Some(CumProdCaptureSignature {
                dtype: input.dtype,
                shape: input.shape.to_vec(),
                axis,
            });
        }
        Ok(())
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.warmed_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "CumProd must warm its fixed axis/shape signature before capture",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "CumProd capture signature lock was poisoned",
            ),
        }
    }
}

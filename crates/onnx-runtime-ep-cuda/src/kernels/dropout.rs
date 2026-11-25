//! Deterministic ONNX `Dropout` inference kernel.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::dropout::{validate_data_type, validate_ratio, validate_training_mode};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const SOURCE: &str = r#"
extern "C" __global__ void fill_true(unsigned char* mask, unsigned long long n) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x)
    mask[i] = 1;
}
"#;

pub struct DropoutFactory {
    pub runtime: Arc<CudaRuntime>,
    pub opset: u64,
}

impl KernelFactory for DropoutFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(DropoutKernel {
            runtime: self.runtime.clone(),
            opset: self.opset,
            reads_training_mode: node.inputs.get(2).is_some_and(Option::is_some),
        }))
    }
}

struct DropoutKernel {
    runtime: Arc<CudaRuntime>,
    opset: u64,
    reads_training_mode: bool,
}

const CAPTURE_REASON: &str =
    "Dropout reads training_mode on the host; omit training_mode to capture inference";

impl Kernel for DropoutKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(1..=3).contains(&inputs.len())
            || inputs[0].is_absent()
            || !(1..=2).contains(&outputs.len())
            || outputs[0].is_absent()
        {
            return Err(EpError::KernelFailed(
                "cuda_ep Dropout: expected data, optional ratio and training_mode, and output with optional mask".into(),
            ));
        }
        let input = &inputs[0];
        let (data_outputs, mask_outputs) = outputs.split_at_mut(1);
        let output = &mut data_outputs[0];
        let mask = mask_outputs.first_mut().filter(|v| !v.is_absent());
        if !input.is_contiguous()
            || !output.is_contiguous()
            || mask.as_ref().is_some_and(|v| !v.is_contiguous())
        {
            return Err(not_implemented("Dropout with strided tensors"));
        }
        validate_data_type(self.opset, input.dtype).map_err(dropout_error)?;
        if output.dtype != input.dtype || output.shape != input.shape {
            return Err(EpError::KernelFailed(
                "cuda_ep Dropout: data output must match the input dtype and shape".into(),
            ));
        }
        if mask
            .as_ref()
            .is_some_and(|v| v.dtype != DataType::Bool || v.shape != input.shape)
        {
            return Err(dropout_error(
                "mask must be Bool with the input shape".into(),
            ));
        }
        if let Some(ratio) = inputs.get(1).filter(|v| !v.is_absent()) {
            validate_ratio(self.opset, ratio.dtype, ratio.shape.len()).map_err(dropout_error)?;
        }
        if let Some(training) = inputs.get(2).filter(|v| !v.is_absent()) {
            validate_training_mode(training.dtype, training.shape.len()).map_err(dropout_error)?;
            if self.runtime.is_capturing()? {
                return Err(not_implemented(CAPTURE_REASON));
            }
            let mut value = [0];
            // SAFETY: training is a present device Bool scalar (one byte).
            unsafe {
                self.runtime.dtoh(
                    &mut value,
                    cuptr(training.data_ptr::<u8>() as *const c_void),
                )?;
            }
            if value[0] != 0 {
                return Err(not_implemented(
                    "Dropout training_mode=true; omit training_mode or set it to false for CUDA inference",
                ));
            }
        }

        let bytes = input.dtype.storage_bytes(input.numel());
        if bytes != 0 {
            unsafe {
                self.runtime.dtod_async(
                    cuptr(input.data_ptr::<u8>() as *const c_void),
                    cuptr(output.data_ptr_mut::<u8>() as *const c_void),
                    bytes,
                )?
            };
        }
        if let Some(mask) = mask {
            let n = input.numel() as u64;
            if n != 0 {
                let function =
                    self.runtime
                        .nvrtc_function("dropout_inference_v1", SOURCE, "fill_true")?;
                let mask = cuptr(mask.data_ptr_mut::<u8>() as *const c_void);
                let mut builder = self.runtime.stream().launch_builder(&function);
                builder.arg(&mask).arg(&n);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (n.div_ceil(256).clamp(1, 65_535) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| driver_err("launch Dropout mask", error))?;
            }
        }
        Ok(())
    }

    fn capture_support(&self) -> CaptureSupport {
        if self.reads_training_mode {
            CaptureSupport::unsupported(CAPTURE_REASON)
        } else {
            CaptureSupport::Supported
        }
    }
}

fn dropout_error(reason: String) -> EpError {
    EpError::KernelFailed(format!("cuda_ep Dropout: {reason}"))
}

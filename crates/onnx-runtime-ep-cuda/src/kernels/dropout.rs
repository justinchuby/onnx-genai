//! Deterministic ONNX `Dropout` inference kernel.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
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
}

impl KernelFactory for DropoutFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(DropoutKernel {
            runtime: self.runtime.clone(),
        }))
    }
}

struct DropoutKernel {
    runtime: Arc<CudaRuntime>,
}

impl Kernel for DropoutKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || !(1..=2).contains(&outputs.len()) {
            return Err(not_implemented(
                "Dropout CUDA coverage is inference mode with the data input only",
            ));
        }
        let input = &inputs[0];
        if !input.is_contiguous() || outputs.iter().any(|v| !v.is_contiguous()) {
            return Err(not_implemented("Dropout with strided tensors"));
        }
        if input.dtype.byte_size() == 0
            || outputs[0].dtype != input.dtype
            || outputs[0].shape != input.shape
        {
            return Err(EpError::KernelFailed(
                "cuda_ep Dropout: data output must match the fixed-width input".into(),
            ));
        }
        let bytes = input.dtype.storage_bytes(input.numel());
        if bytes != 0 {
            unsafe {
                self.runtime.dtod_async(
                    cuptr(input.data_ptr::<u8>() as *const c_void),
                    cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void),
                    bytes,
                )?
            };
        }
        if outputs.len() == 2 {
            if outputs[1].dtype != DataType::Bool || outputs[1].shape != input.shape {
                return Err(EpError::KernelFailed(
                    "cuda_ep Dropout: mask must be Bool with the input shape".into(),
                ));
            }
            let n = input.numel() as u64;
            if n != 0 {
                let function =
                    self.runtime
                        .nvrtc_function("dropout_inference_v1", SOURCE, "fill_true")?;
                let mask = cuptr(outputs[1].data_ptr_mut::<u8>() as *const c_void);
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
}

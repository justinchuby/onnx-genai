//! `Size`: host-compute the input's total element count as a rank-0 `Int64`
//! scalar and synchronously upload it to the GPU. Dtype-agnostic on its input
//! (it reads only shape metadata), mirroring the CPU EP's `Size` kernel.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::runtime::{CudaRuntime, cuptr};

pub struct SizeFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for SizeFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SizeKernel {
            runtime: self.runtime.clone(),
            warmed: AtomicBool::new(false),
        }))
    }
}

#[derive(Debug)]
pub struct SizeKernel {
    runtime: Arc<CudaRuntime>,
    warmed: AtomicBool,
}

impl Kernel for SizeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Size: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let output = &mut outputs[0];
        if output.dtype != DataType::Int64 || !output.is_contiguous() {
            return Err(EpError::KernelFailed(
                "cuda_ep Size: output must be a contiguous Int64 scalar".into(),
            ));
        }
        if output.numel() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Size: output has {} elements, expected 1 (rank-0 scalar)",
                output.numel()
            )));
        }
        if self.runtime.is_capturing()? {
            // The shape-keyed kernel cache guarantees the same input geometry
            // warmed this output, so its device scalar stays valid on replay.
            return Ok(());
        }
        let count = i64::try_from(inputs[0].numel()).map_err(|_| {
            EpError::KernelFailed("cuda_ep Size: element count exceeds Int64".into())
        })?;
        // SAFETY: output is a live device allocation sized for one Int64, and the
        // 8-byte payload is exactly one element.
        unsafe {
            self.runtime.htod(
                &count.to_le_bytes(),
                cuptr(output.data_ptr_mut::<u8>() as *const u8 as _),
            )?
        };
        self.warmed.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        true
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.warmed.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "Size output scalar must be warmed into its stable device buffer before capture",
            )
        }
    }
}

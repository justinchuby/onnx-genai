use std::ffi::c_void;
use std::sync::Arc;

use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
    bit_cast::copy_size,
};
use onnx_runtime_ir::{DataType, Node, opset26::bit_cast_target};

use crate::runtime::{CudaRuntime, cuptr};

pub struct BitCastFactory(pub Arc<CudaRuntime>);

impl KernelFactory for BitCastFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(BitCastKernel {
            runtime: self.0.clone(),
            target: bit_cast_target(node).map_err(EpError::KernelFailed)?,
        }))
    }
}

struct BitCastKernel {
    runtime: Arc<CudaRuntime>,
    target: DataType,
}

impl Kernel for BitCastKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let ([input], [output]) = (inputs, outputs) else {
            return Err(EpError::KernelFailed(
                "BitCast: expected one input and one output".into(),
            ));
        };
        let bytes = copy_size(input, output, self.target)?;
        if bytes != 0 {
            let src = cuptr(input.data_ptr::<c_void>());
            let dst = cuptr(output.data_ptr_mut::<c_void>());
            if src != dst {
                // Session-owned buffers remain live until the provider stream completes.
                unsafe { self.runtime.dtod_async(src, dst, bytes)? };
            }
        }
        Ok(())
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }

    fn can_run_in_place(&self, input_index: usize) -> bool {
        input_index == 0
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::Supported
    }
}

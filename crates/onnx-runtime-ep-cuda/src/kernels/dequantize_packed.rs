//! Device-native standalone DequantizePacked-29.

use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::dequantize_packed::{Format, Metadata};
use onnx_runtime_ir::{DeviceType, Node};

use super::block_quantized_matmul::{BlockFormat, decoder_prelude};
use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const ENTRY: &str = "dequantize_packed";
const THREADS: u32 = 256;

pub(super) struct Factory(pub Arc<CudaRuntime>);

struct PackedKernel {
    runtime: Arc<CudaRuntime>,
    metadata: Metadata,
}

fn invalid(detail: String) -> EpError {
    EpError::KernelFailed(format!("cuda_ep DequantizePacked: {detail}"))
}

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(PackedKernel {
            runtime: self.0.clone(),
            metadata: Metadata::from_node(node).map_err(invalid)?,
        }))
    }
}

pub(super) fn decoder_kind(format: Format) -> i32 {
    match format {
        Format::Iq4Nl => BlockFormat::Iq4Nl,
        Format::Iq4Xs => BlockFormat::Iq4Xs,
        Format::Iq2Xxs => BlockFormat::Iq2Xxs,
        Format::Iq2Xs => BlockFormat::Iq2Xs,
        Format::Iq2S => BlockFormat::Iq2S,
        Format::Iq3Xxs => BlockFormat::Iq3Xxs,
        Format::Iq3S => BlockFormat::Iq3S,
        Format::Iq1S => BlockFormat::Iq1S,
        Format::Iq1M => BlockFormat::Iq1M,
        Format::Q2K => BlockFormat::Q2K,
        Format::Q3K => BlockFormat::Q3K,
        Format::Q5K => BlockFormat::Q5K,
        Format::Q6K => BlockFormat::Q6K,
        Format::Q8_0 => BlockFormat::Q8_0,
        Format::Q1_0
        | Format::Q2_0
        | Format::Q4_0
        | Format::Q4_1
        | Format::Q5_0
        | Format::Q5_1
        | Format::Q8_1
        | Format::Q4K
        | Format::Q8K
        | Format::Tq1_0
        | Format::Tq2_0
        | Format::Mxfp4
        | Format::Nvfp4 => return -(format as i32) - 1,
    }
    .kernel_id()
        + 32
}

pub(super) fn decoder_source() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| decoder_prelude() + include_str!("dequantize_packed.cu"))
}

impl Kernel for PackedKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let ([x], [y]) = (inputs, outputs) else {
            return Err(invalid(
                "expected one packed input and one decoded output".into(),
            ));
        };
        let m = &self.metadata;
        m.validate_input(x.dtype, x.shape).map_err(invalid)?;
        m.validate_output(y.dtype, y.shape).map_err(invalid)?;
        if x.device.device_type != DeviceType::Cuda || x.device != y.device {
            return Err(invalid(
                "input and output must be on the same CUDA device".into(),
            ));
        }
        if m.elements() == 0 {
            return Ok(());
        }
        x.validate()?;
        y.validate()?;
        if !x.is_contiguous() || !y.is_contiguous() {
            return Err(invalid(
                "input and output must use contiguous storage".into(),
            ));
        }
        let src = decoder_source();
        let func = self.runtime.nvrtc_function(ENTRY, src, ENTRY)?;
        let (width, bytes) = m.format().block();
        let width = width as i32;
        let bytes = bytes as i32;
        let total = m.elements() as u64;
        let kind = decoder_kind(m.format());
        let dtype = m.dtype().to_onnx();
        let x = cuptr(x.data_ptr::<u8>().cast::<c_void>());
        let y = cuptr(y.data_ptr_mut::<u8>().cast::<c_void>());
        let grid = total.div_ceil(u64::from(THREADS)).min(65_535) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut args = stream.launch_builder(&func);
        args.arg(&x)
            .arg(&y)
            .arg(&total)
            .arg(&width)
            .arg(&bytes)
            .arg(&kind)
            .arg(&dtype);
        // SAFETY: arguments match the device signature; validated buffers cover
        // the grid-stride loop's complete packed input and typed output ranges.
        unsafe { args.launch(cfg) }.map_err(|e| driver_err("launch DequantizePacked", e))?;
        if self.runtime.is_capturing()? {
            Ok(())
        } else {
            self.runtime.synchronize()
        }
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::Supported
    }
}

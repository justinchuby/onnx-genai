//! `Shape`: host-compute shape metadata and synchronously upload it to the GPU.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::runtime::{CudaRuntime, cuptr};

pub struct ShapeFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for ShapeFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ShapeKernel {
            runtime: self.runtime.clone(),
            start: node.attr("start").and_then(|a| a.as_int()).unwrap_or(0),
            end: node.attr("end").and_then(|a| a.as_int()),
            warmed: AtomicBool::new(false),
        }))
    }
}

#[derive(Debug)]
pub struct ShapeKernel {
    runtime: Arc<CudaRuntime>,
    start: i64,
    end: Option<i64>,
    warmed: AtomicBool,
}

impl Kernel for ShapeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Shape: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let output = &mut outputs[0];
        if output.dtype != DataType::Int64 || !output.is_contiguous() {
            return Err(EpError::KernelFailed(
                "cuda_ep Shape: output must be contiguous Int64".into(),
            ));
        }
        let rank = inputs[0].shape.len() as i64;
        let clamp =
            |value: i64| (if value < 0 { value + rank } else { value }).clamp(0, rank) as usize;
        let start = clamp(self.start);
        let end = clamp(self.end.unwrap_or(rank)).max(start);
        let dims = &inputs[0].shape[start..end];
        if output.numel() != dims.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Shape: output has {} elements, expected {}",
                output.numel(),
                dims.len()
            )));
        }
        if self.runtime.is_capturing()? {
            // The shape-keyed kernel cache guarantees the same input geometry
            // warmed this output, so its device metadata stays valid on replay.
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(dims.len() * std::mem::size_of::<i64>());
        for &dim in dims {
            let dim = i64::try_from(dim).map_err(|_| {
                EpError::KernelFailed("cuda_ep Shape: dimension exceeds Int64".into())
            })?;
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        if !bytes.is_empty() {
            // SAFETY: output is a live device allocation sized for its checked
            // Int64 shape; `bytes` has exactly one i64 per output element.
            unsafe {
                self.runtime
                    .htod(&bytes, cuptr(output.data_ptr_mut::<u8>() as *const u8 as _))?
            };
        }
        self.warmed.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        true
    }

    fn cuda_graph_compatible(&self) -> bool {
        self.warmed.load(Ordering::Relaxed)
    }
}

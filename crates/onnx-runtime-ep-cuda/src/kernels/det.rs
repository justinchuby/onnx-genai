use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, DeviceId, EpError, Kernel, KernelFactory, OutputMetadata, Result,
    TensorMetadata, TensorMut, TensorView, WorkspaceLifetime, WorkspacePreparation,
    WorkspaceRequirement, WorkspaceView, det,
};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const MAX_BLOCKS: usize = 128;
const BLOCK_THREADS: u32 = 128;
const SOURCE: &str = include_str!("det.cu");

pub struct Factory(pub Arc<CudaRuntime>);

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Det {
            runtime: self.0.clone(),
            opset: node.version.unwrap_or(11),
        }))
    }
}

struct Det {
    runtime: Arc<CudaRuntime>,
    opset: i64,
}

fn workspace_for(g: &det::Geometry) -> Result<WorkspaceRequirement> {
    if g.batch == 0 || g.order < 2 {
        return Ok(WorkspaceRequirement::NONE);
    }
    let bytes = g
        .batch
        .min(MAX_BLOCKS)
        .checked_mul(g.matrix_elements)
        .and_then(|n| n.checked_mul(size_of::<f64>()))
        .ok_or_else(|| {
            EpError::KernelFailed(
                "CUDA Det: scratch size overflows usize; reduce matrix size".into(),
            )
        })?;
    Ok(WorkspaceRequirement {
        bytes: bytes as u64,
        alignment: align_of::<f64>(),
        lifetime: WorkspaceLifetime::StepScoped,
    })
}

impl Kernel for Det {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.execute_with_workspace(inputs, outputs, None)
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        let [input] = inputs else {
            return Err(EpError::KernelFailed(
                "CUDA Det: expected one input for scratch planning".into(),
            ));
        };
        workspace_for(&det::Geometry::from_shape(input.shape)?).map(WorkspacePreparation::Ready)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let g = det::validate(inputs, outputs, self.opset)?;
        let x = &inputs[0];
        let y = &mut outputs[0];
        let device = DeviceId::cuda(self.runtime.ordinal());
        if x.device != device || y.device != device {
            return Err(EpError::KernelFailed(format!(
                "CUDA Det: expected input and output on {device:?}, got {:?} and {:?}; explicitly transfer to the execution device",
                x.device, y.device
            )));
        }
        x.validate()?;
        y.validate()?;
        if g.batch == 0 {
            return Ok(());
        }
        if !x.is_contiguous() || !y.is_contiguous() {
            return Err(EpError::KernelFailed(
                "CUDA Det: expected contiguous input and output; materialize strided tensors before execution".into(),
            ));
        }
        let needed = workspace_for(&g)?.bytes as usize;
        let scratch = if needed == 0 {
            0
        } else {
            let scratch = workspace
                .filter(|w| {
                    w.bytes() >= needed
                        && w.ptr().0.cast::<f64>().is_aligned()
                        && !w.ptr().0.is_null()
                })
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "CUDA Det: executor must supply {needed} bytes of device scratch"
                    ))
                })?;
            cuptr(scratch.ptr().0 as *const c_void)
        };
        let entry = match x.dtype {
            DataType::Float32 => "det_f32",
            DataType::Float64 => "det_f64",
            DataType::Float16 => "det_f16",
            DataType::BFloat16 => "det_bf16",
            _ => unreachable!("Det dtype was validated"),
        };
        if matches!(x.dtype, DataType::Float16 | DataType::BFloat16) {
            self.runtime.require_nvrtc_half_headers("Det")?;
        }
        let func = self.runtime.nvrtc_function("det_v1", SOURCE, entry)?;
        let x_ptr = cuptr(x.data_ptr::<u8>().cast());
        let y_ptr = cuptr(y.data_ptr_mut::<u8>().cast_const().cast());
        let n = g.order as u64;
        let batch = g.batch as u64;
        let stream = self.runtime.stream();
        let mut launch = stream.launch_builder(&func);
        launch
            .arg(&x_ptr)
            .arg(&y_ptr)
            .arg(&scratch)
            .arg(&n)
            .arg(&batch);
        // SAFETY: the tensor views cover the validated geometry; each block
        // owns one matrix-sized scratch slot, and only orders >= 2 use it.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (g.batch.min(MAX_BLOCKS) as u32, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err(&format!("launch CUDA Det ({entry})"), e))?;
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "CUDA Det synchronizes before returning its step-scoped scratch",
        )
    }
}

use std::{ffi::c_void, sync::Arc};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::rnn::{self, Config, Geometry};
use onnx_runtime_ep_api::{
    CaptureSupport, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut,
    TensorView, WorkspaceLifetime, WorkspacePreparation, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const SOURCE: &str = include_str!("rnn.cu");

pub struct Factory {
    pub runtime: Arc<CudaRuntime>,
    pub since: u64,
}

struct Rnn {
    runtime: Arc<CudaRuntime>,
    config: Config,
    since: u64,
}

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Rnn {
            runtime: self.runtime.clone(),
            config: Config::parse(node)?,
            since: self.since,
        }))
    }
}

fn workspace_bytes(g: &Geometry, dtype: DataType) -> Result<usize> {
    if g.time == 0 {
        return Ok(0);
    }
    let accum = if dtype == DataType::Float64 { 8 } else { 4 };
    [g.batch, g.directions, g.hidden, 2, accum]
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| rnn::error("CUDA recurrent state scratch size overflows"))
}

impl Kernel for Rnn {
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata],
        _: &[OutputMetadata],
    ) -> Result<WorkspacePreparation> {
        let (Some(x), Some(r)) = (
            inputs.first().filter(|x| x.present),
            inputs.get(2).filter(|r| r.present),
        ) else {
            return Ok(WorkspacePreparation::Deferred);
        };
        let g = self.config.geometry(x.shape, r.shape)?;
        Ok(WorkspacePreparation::Ready(WorkspaceRequirement {
            bytes: workspace_bytes(&g, x.dtype)? as u64,
            alignment: 8,
            lifetime: WorkspaceLifetime::StepScoped,
        }))
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.execute_with_workspace(inputs, outputs, None)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let g = self.config.validate(self.since, inputs, outputs)?;
        if inputs.iter().any(|t| !t.is_absent() && !t.is_contiguous())
            || outputs.iter().any(|t| !t.is_absent() && !t.is_contiguous())
        {
            return Err(rnn::error(
                "CUDA requires contiguous input and output tensors",
            ));
        }
        let iptr = |i| {
            inputs
                .get(i)
                .filter(|t: &&TensorView| !t.is_absent() && !t.shape.contains(&0))
                .map_or(0, |t| cuptr(t.data_ptr::<u8>() as *const c_void))
        };
        let lengths = iptr(4);
        if lengths != 0 {
            // Only validation metadata crosses the device boundary. The kernel
            // reads the original device lengths and does all recurrence on GPU.
            let size = g.batch * size_of::<i32>();
            let mut raw = Vec::new();
            raw.try_reserve_exact(size)
                .map_err(|e| rnn::error(format!("cannot stage CUDA sequence_lens: {e}")))?;
            raw.resize(size, 0);
            // SAFETY: the validated length tensor contains exactly batch i32s.
            unsafe {
                self.runtime.dtoh(&mut raw, lengths)?;
            }
            for (batch, bytes) in raw.chunks_exact(4).enumerate() {
                rnn::sequence_length(
                    g.time,
                    batch,
                    i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                )?;
            }
        }
        let mut optr = |i| {
            outputs
                .get_mut(i)
                .filter(|t: &&mut TensorMut| !t.is_absent() && !t.shape.contains(&0))
                .map_or(0, |t| cuptr(t.data_ptr_mut::<u8>() as *const c_void))
        };
        let (y, yh) = (optr(0), optr(1));
        if y == 0 && yh == 0 {
            return Ok(());
        }
        let scratch_bytes = workspace_bytes(&g, inputs[0].dtype)?;
        let scratch = if scratch_bytes == 0 {
            0
        } else {
            let w = workspace
                .filter(|w| w.bytes() >= scratch_bytes)
                .ok_or_else(|| {
                    rnn::error(format!(
                        "CUDA recurrence needs {scratch_bytes} bytes of executor workspace"
                    ))
                })?;
            cuptr(w.ptr().0 as *const c_void)
        };
        let entry = match inputs[0].dtype {
            DataType::Float16 => "rnn_f16",
            DataType::BFloat16 => "rnn_bf16",
            DataType::Float32 => "rnn_f32",
            DataType::Float64 => "rnn_f64",
            dtype => return Err(rnn::error(format!("unsupported CUDA dtype {dtype:?}"))),
        };
        let function = self.runtime.nvrtc_function("rnn", SOURCE, entry)?;
        let (x, w, r, b, initial) = (iptr(0), iptr(1), iptr(2), iptr(3), iptr(5));
        let (batch, time, input, hidden) = (
            g.batch as u64,
            g.time as u64,
            g.input as u64,
            g.hidden as u64,
        );
        let (dirs, batch_first, reverse_first) = (
            g.directions as i32,
            i32::from(g.batch_first),
            i32::from(self.config.reverse_first),
        );
        let clip = self.config.clip.unwrap_or(-1.0);
        let [a, b_act] = self.config.activations;
        let (act0, act1) = (a.kind as i32, b_act.kind as i32);
        let mut launch = self.runtime.stream().launch_builder(&function);
        launch
            .arg(&x)
            .arg(&w)
            .arg(&r)
            .arg(&b)
            .arg(&lengths)
            .arg(&initial)
            .arg(&y)
            .arg(&yh)
            .arg(&scratch)
            .arg(&batch)
            .arg(&time)
            .arg(&input)
            .arg(&hidden)
            .arg(&dirs)
            .arg(&batch_first)
            .arg(&reverse_first)
            .arg(&clip)
            .arg(&act0)
            .arg(&a.alpha)
            .arg(&a.beta)
            .arg(&act1)
            .arg(&b_act.alpha)
            .arg(&b_act.beta);
        // SAFETY: geometry validates tensor extents and optional pointers, and
        // step-scoped workspace holds both accumulator vectors for each lane.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((g.batch * g.directions).min(65535) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launching native CUDA RNN", e))?;
        Ok(())
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "RNN validates runtime sequence_lens with a device-to-host synchronization",
        )
    }
}

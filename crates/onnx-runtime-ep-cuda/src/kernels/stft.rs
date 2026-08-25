//! CUDA `STFT` backed by the shared cuFFT plan/cache foundation.
//!
//! One fused NVRTC kernel extracts every complete frame, applies the optional
//! window, promotes real samples to interleaved complex f32, and packs all
//! `(batch, frame)` transforms contiguously. One cuFFT PlanMany execution then
//! transforms the entire batch, followed by one unpack kernel.

use core::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node, Shape, TensorLayout};
use onnx_runtime_memory_governor::MemoryRole;

use crate::cufft::{CufftPlan, CufftPlanCache, CufftPlanKey, DftDirection, DftInputKind};
use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const WORKSPACE_ALIGNMENT: usize = 256;
const MODULE: &str = "stft_pack_v1";
const SOURCE: &str = r#"
extern "C" __global__ void stft_pack_window_f32(
    const float* signal, const float* window, float2* packed,
    unsigned long long signal_length, unsigned long long frame_step,
    unsigned long long frame_length, unsigned long long frames,
    unsigned long long transforms, int components, int has_window) {
  const unsigned long long total = transforms * frame_length;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long sample = index % frame_length;
    const unsigned long long transform = index / frame_length;
    const unsigned long long frame = transform % frames;
    const unsigned long long batch = transform / frames;
    const unsigned long long signal_sample =
        batch * signal_length + frame * frame_step + sample;
    const unsigned long long signal_offset = signal_sample * components;
    const float weight = has_window ? window[sample] : 1.0f;
    float2 value;
    value.x = signal[signal_offset] * weight;
    value.y = components == 2 ? signal[signal_offset + 1] * weight : 0.0f;
    packed[index] = value;
  }
}

extern "C" __global__ void stft_unpack_f32(
    const float2* packed, float* output, unsigned long long frame_length,
    unsigned long long bins, unsigned long long transforms) {
  const unsigned long long total = transforms * bins;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long transform = index / bins;
    const unsigned long long bin = index % bins;
    const float2 value = packed[transform * frame_length + bin];
    output[index * 2] = value.x;
    output[index * 2 + 1] = value.y;
  }
}
"#;

static LAST_FRAMES: AtomicU64 = AtomicU64::new(0);
static LAST_FFT_BATCH: AtomicU64 = AtomicU64::new(0);
static LAST_PACK_UNPACK_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static LAST_CUFFT_EXECUTIONS: AtomicU64 = AtomicU64::new(0);
static LAST_WORKSPACE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Geometry and governed scratch used by the most recent CUDA STFT dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StftExecutionStats {
    pub frames_per_signal: u64,
    pub fft_batch: u64,
    pub pack_unpack_launches: u64,
    pub cufft_executions: u64,
    pub workspace_bytes: u64,
}

pub fn stft_last_execution_stats() -> StftExecutionStats {
    StftExecutionStats {
        frames_per_signal: LAST_FRAMES.load(Ordering::Relaxed),
        fft_batch: LAST_FFT_BATCH.load(Ordering::Relaxed),
        pack_unpack_launches: LAST_PACK_UNPACK_LAUNCHES.load(Ordering::Relaxed),
        cufft_executions: LAST_CUFFT_EXECUTIONS.load(Ordering::Relaxed),
        workspace_bytes: LAST_WORKSPACE_BYTES.load(Ordering::Relaxed),
    }
}

pub(crate) struct StftFactory {
    pub(crate) runtime: Arc<CudaRuntime>,
    pub(crate) plans: Arc<CufftPlanCache>,
}

impl KernelFactory for StftFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let onesided = node
            .attr("onesided")
            .and_then(|attribute| attribute.as_int())
            .unwrap_or(1);
        if !matches!(onesided, 0 | 1) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep STFT: attribute 'onesided' must be 0 or 1, got {onesided}"
            )));
        }
        Ok(Box::new(StftKernel {
            runtime: self.runtime.clone(),
            plans: self.plans.clone(),
            onesided: onesided != 0,
            prepared_plan: Mutex::new(None),
        }))
    }
}

#[derive(Clone, Debug)]
struct StftSpec {
    signal_shape: [usize; 3],
    frame_step: usize,
    frame_length: usize,
    frames: usize,
    bins: usize,
    transforms: usize,
    input_kind: DftInputKind,
}

impl StftSpec {
    fn output_shape(&self) -> [usize; 4] {
        [self.signal_shape[0], self.frames, self.bins, 2]
    }

    fn key(&self, device: u32) -> CufftPlanKey {
        CufftPlanKey {
            device,
            dtype: DataType::Float32,
            input_kind: self.input_kind,
            rank: 3,
            axis: 1,
            length: self.frame_length,
            batch: self.transforms,
            direction: DftDirection::Forward,
        }
    }
}

struct WorkspaceLayout {
    work_offset: usize,
    total_bytes: usize,
}

impl WorkspaceLayout {
    fn new(spec: &StftSpec, work_bytes: usize) -> Result<Self> {
        let packed_bytes = spec
            .transforms
            .checked_mul(spec.frame_length)
            .and_then(|elements| elements.checked_mul(2 * std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep STFT: packed complex workspace size overflow".into())
            })?;
        let work_offset = align_up(packed_bytes, WORKSPACE_ALIGNMENT)?;
        let total_bytes = work_offset.checked_add(work_bytes).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep STFT: cuFFT workspace size overflow".into())
        })?;
        Ok(Self {
            work_offset,
            total_bytes,
        })
    }
}

struct StftKernel {
    runtime: Arc<CudaRuntime>,
    plans: Arc<CufftPlanCache>,
    onesided: bool,
    prepared_plan: Mutex<Option<(CufftPlanKey, Arc<Mutex<CufftPlan>>)>>,
}

impl StftKernel {
    fn resolve(&self, inputs: &[TensorView<'_>]) -> Result<StftSpec> {
        if !(2..=4).contains(&inputs.len()) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep STFT: expected 2..=4 inputs, got {}",
                inputs.len()
            )));
        }
        let signal = &inputs[0];
        if signal.is_absent() {
            return Err(EpError::KernelFailed(
                "cuda_ep STFT: required 'signal' input is absent".into(),
            ));
        }
        require_f32("signal", signal.dtype)?;
        let signal_shape: [usize; 3] = signal.shape.try_into().map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep STFT: 'signal' must have rank 3 [batch, signal_length, 1|2], got rank {}",
                signal.shape.len()
            ))
        })?;
        let input_kind = match signal_shape[2] {
            1 => DftInputKind::Real,
            2 => DftInputKind::Complex,
            components => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep STFT: 'signal' last dimension must be 1 (real) or 2 (complex), got \
                     {components}"
                )));
            }
        };
        if self.onesided && input_kind == DftInputKind::Complex {
            return Err(EpError::KernelFailed(
                "cuda_ep STFT: onesided=1 requires a real signal (last dimension 1); use \
                 onesided=0 for complex input"
                    .into(),
            ));
        }
        if !signal.is_contiguous() {
            return Err(EpError::KernelFailed(
                "cuda_ep STFT: signal must be contiguous; strided inputs must decline at placement"
                    .into(),
            ));
        }

        let frame_step = positive_scalar(&self.runtime, &inputs[1], "frame_step")?;
        let window = present_input(inputs, 2);
        let explicit_frame_length = present_input(inputs, 3)
            .map(|input| positive_scalar(&self.runtime, input, "frame_length"))
            .transpose()?;
        let window_length = if let Some(window) = window {
            require_f32("window", window.dtype)?;
            if window.shape.len() != 1 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep STFT: 'window' must have rank 1, got rank {}",
                    window.shape.len()
                )));
            }
            if window.shape[0] == 0 {
                return Err(EpError::KernelFailed(
                    "cuda_ep STFT: 'window' length must be greater than zero".into(),
                ));
            }
            if !window.is_contiguous() {
                return Err(EpError::KernelFailed(
                    "cuda_ep STFT: window must be contiguous; strided inputs must decline at \
                     placement"
                        .into(),
                ));
            }
            Some(window.shape[0])
        } else {
            None
        };
        let frame_length = match (window_length, explicit_frame_length) {
            (Some(window_length), Some(frame_length)) if window_length != frame_length => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep STFT: window length {window_length} must equal frame_length \
                     {frame_length}"
                )));
            }
            (Some(_), Some(frame_length)) | (None, Some(frame_length)) => frame_length,
            (Some(window_length), None) => window_length,
            (None, None) => {
                return Err(EpError::KernelFailed(
                    "cuda_ep STFT: either optional window or frame_length must be provided".into(),
                ));
            }
        };
        let signal_length = signal_shape[1];
        if frame_length > signal_length {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep STFT: frame length {frame_length} exceeds signal length {signal_length}; \
                 STFT uses complete unpadded frames"
            )));
        }
        let frames = (signal_length - frame_length) / frame_step + 1;
        let bins = if self.onesided {
            frame_length / 2 + 1
        } else {
            frame_length
        };
        let transforms = signal_shape[0].checked_mul(frames).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep STFT: batch × frame count overflow".into())
        })?;
        Ok(StftSpec {
            signal_shape,
            frame_step,
            frame_length,
            frames,
            bins,
            transforms,
            input_kind,
        })
    }

    fn plan(&self, spec: &StftSpec) -> Result<Arc<Mutex<CufftPlan>>> {
        let key = spec.key(self.runtime.ordinal());
        let mut prepared = self.prepared_plan.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep STFT: prepared-plan lock was poisoned".into())
        })?;
        if let Some((prepared_key, plan)) = prepared.as_ref()
            && prepared_key == &key
        {
            return Ok(plan.clone());
        }
        let plan = self
            .plans
            .get_or_create(self.runtime.clone(), key.clone())?;
        *prepared = Some((key, plan.clone()));
        Ok(plan)
    }

    fn workspace_for(&self, spec: &StftSpec) -> Result<WorkspaceRequirement> {
        if spec.transforms == 0 {
            return Ok(WorkspaceRequirement::NONE);
        }
        let plan = self.plan(spec)?;
        let work_bytes = plan
            .lock()
            .map_err(|_| {
                EpError::KernelFailed("cuda_ep STFT: cuFFT plan lock was poisoned".into())
            })?
            .work_bytes();
        let layout = WorkspaceLayout::new(spec, work_bytes)?;
        Ok(WorkspaceRequirement {
            bytes: u64::try_from(layout.total_bytes).map_err(|_| {
                EpError::KernelFailed("cuda_ep STFT: workspace size exceeds u64".into())
            })?,
            alignment: WORKSPACE_ALIGNMENT,
            lifetime: WorkspaceLifetime::StepScoped,
            role: MemoryRole::Workspace { step_scoped: true },
        })
    }

    fn run(
        &self,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        if outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep STFT: expected 1 output, got {}",
                outputs.len()
            )));
        }
        let spec = self.resolve(inputs)?;
        let output = &mut outputs[0];
        require_f32("output", output.dtype)?;
        if !output.is_contiguous() {
            return Err(EpError::KernelFailed(
                "cuda_ep STFT: output must be contiguous".into(),
            ));
        }
        if output.shape != spec.output_shape() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep STFT: output shape {:?} does not match expected {:?}",
                output.shape,
                spec.output_shape()
            )));
        }
        if spec.transforms == 0 {
            return Ok(());
        }

        let plan = self.plan(&spec)?;
        let work_bytes = plan
            .lock()
            .map_err(|_| {
                EpError::KernelFailed("cuda_ep STFT: cuFFT plan lock was poisoned".into())
            })?
            .work_bytes();
        let layout = WorkspaceLayout::new(&spec, work_bytes)?;
        let workspace = workspace.ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep STFT: executor did not provide the required {}-byte governed workspace",
                layout.total_bytes
            ))
        })?;
        if workspace.bytes() < layout.total_bytes {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep STFT: governed workspace is {} bytes, but this transform requires {} bytes",
                workspace.bytes(),
                layout.total_bytes
            )));
        }
        LAST_FRAMES.store(spec.frames as u64, Ordering::Relaxed);
        LAST_FFT_BATCH.store(spec.transforms as u64, Ordering::Relaxed);
        LAST_PACK_UNPACK_LAUNCHES.store(2, Ordering::Relaxed);
        LAST_CUFFT_EXECUTIONS.store(1, Ordering::Relaxed);
        LAST_WORKSPACE_BYTES.store(layout.total_bytes as u64, Ordering::Relaxed);
        let packed = cuptr(workspace.ptr().0 as *const c_void);
        let work = packed
            .checked_add(layout.work_offset as u64)
            .ok_or_else(|| EpError::KernelFailed("cuda_ep STFT: work pointer overflow".into()))?;
        let signal = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let window = present_input(inputs, 2)
            .map(|input| cuptr(input.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let signal_length = spec.signal_shape[1] as u64;
        let frame_step = spec.frame_step as u64;
        let frame_length = spec.frame_length as u64;
        let frames = spec.frames as u64;
        let transforms = spec.transforms as u64;
        let components = match spec.input_kind {
            DftInputKind::Real => 1i32,
            DftInputKind::Complex => 2i32,
        };
        let has_window = i32::from(window != 0);
        let packed_elements = spec
            .transforms
            .checked_mul(spec.frame_length)
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep STFT: packed element count overflow".into())
            })?;
        let pack = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "stft_pack_window_f32")?;
        let mut builder = self.runtime.stream().launch_builder(&pack);
        builder
            .arg(&signal)
            .arg(&window)
            .arg(&packed)
            .arg(&signal_length)
            .arg(&frame_step)
            .arg(&frame_length)
            .arg(&frames)
            .arg(&transforms)
            .arg(&components)
            .arg(&has_window);
        // SAFETY: validated contiguous signal/window buffers and the governed
        // packed workspace cover every element addressed by the launch.
        unsafe { builder.launch(launch_config(packed_elements)) }
            .map_err(|error| driver_err("launching STFT frame/window pack kernel", error))?;

        {
            let mut plan = plan.lock().map_err(|_| {
                EpError::KernelFailed("cuda_ep STFT: cuFFT plan lock was poisoned".into())
            })?;
            // SAFETY: `packed` covers transforms*frame_length complex values;
            // `work` covers the exact bytes reported by this plan.
            unsafe {
                plan.execute(
                    packed as usize as *mut c_void,
                    if work_bytes == 0 {
                        std::ptr::null_mut()
                    } else {
                        work as usize as *mut c_void
                    },
                )?;
            }
        }

        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let bins = spec.bins as u64;
        let output_elements = spec.transforms.checked_mul(spec.bins).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep STFT: output element count overflow".into())
        })?;
        let unpack = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "stft_unpack_f32")?;
        let mut builder = self.runtime.stream().launch_builder(&unpack);
        builder
            .arg(&packed)
            .arg(&output_ptr)
            .arg(&frame_length)
            .arg(&bins)
            .arg(&transforms);
        // SAFETY: output is contiguous `[batch, frames, bins, 2]` and packed
        // holds the full spectrum for each transform.
        unsafe { builder.launch(launch_config(output_elements)) }
            .map_err(|error| driver_err("launching STFT output unpack kernel", error))?;
        Ok(())
    }
}

impl Kernel for StftKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }

    fn workspace_requirement(
        &self,
        _inputs: &[TensorMetadata<'_>],
    ) -> Result<WorkspaceRequirement> {
        // frame_step is a required runtime scalar, so prepare-only metadata
        // cannot know the frame count. Execution computes the exact requirement.
        Ok(WorkspaceRequirement::NONE)
    }

    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        _metadata: &[TensorMetadata<'_>],
    ) -> Result<WorkspaceRequirement> {
        self.workspace_for(&self.resolve(inputs)?)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "required frame_step/frame_length scalar reads and cuFFT plan selection are not \
             CUDA-graph capture-safe",
        )
    }
}

fn require_f32(name: &str, dtype: DataType) -> Result<()> {
    if dtype == DataType::Float32 {
        Ok(())
    } else {
        Err(EpError::KernelFailed(format!(
            "cuda_ep STFT: '{name}' must be Float32, got {dtype:?}; unsupported dtypes must \
             decline at placement time"
        )))
    }
}

fn positive_scalar(runtime: &CudaRuntime, input: &TensorView<'_>, name: &str) -> Result<usize> {
    if input.is_absent() {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep STFT: required '{name}' input is absent"
        )));
    }
    if !matches!(input.dtype, DataType::Int32 | DataType::Int64)
        || !input.shape.is_empty()
        || !input.is_contiguous()
    {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep STFT: '{name}' must be a contiguous Int32 or Int64 scalar, got dtype {:?}, \
             shape {:?}",
            input.dtype, input.shape
        )));
    }
    let mut bytes = [0u8; std::mem::size_of::<i64>()];
    let width = input.dtype.byte_size();
    // SAFETY: the validated scalar covers `width` readable bytes.
    unsafe {
        runtime.dtoh(
            &mut bytes[..width],
            cuptr(input.data_ptr::<u8>() as *const c_void),
        )?;
    }
    let value = if input.dtype == DataType::Int32 {
        i32::from_ne_bytes(bytes[..4].try_into().expect("four-byte scalar")) as i64
    } else {
        i64::from_ne_bytes(bytes)
    };
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep STFT: '{name}' must be greater than zero, got {value}"
            ))
        })
}

fn present_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| EpError::KernelFailed("cuda_ep STFT: workspace alignment overflow".into()))
}

fn launch_config(elements: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            (elements as u64).div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
            1,
            1,
        ),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Claim-time validation for the f32 CUDA STFT surface.
pub(crate) fn unsupported_reason(
    node: &Node,
    input_shapes: &[Shape],
    input_dtypes: &[DataType],
    input_layouts: &[TensorLayout],
) -> Option<String> {
    let reject = || -> std::result::Result<(), String> {
        if node.outputs.len() != 1 || !(2..=4).contains(&node.inputs.len()) {
            return Err("requires 2..=4 inputs and exactly 1 output".into());
        }
        if node.inputs.first().is_none_or(Option::is_none)
            || node.inputs.get(1).is_none_or(Option::is_none)
        {
            return Err("signal and frame_step inputs must be present".into());
        }
        if input_dtypes.first() != Some(&DataType::Float32) {
            return Err(format!(
                "signal dtype must be Float32 for the cuFFT path, got {:?}",
                input_dtypes.first().copied().unwrap_or(DataType::Undefined)
            ));
        }
        if !matches!(input_dtypes.get(1), Some(DataType::Int32 | DataType::Int64)) {
            return Err(format!(
                "frame_step must be Int32 or Int64, got {:?}",
                input_dtypes.get(1).copied().unwrap_or(DataType::Undefined)
            ));
        }
        if input_shapes.get(1).is_none_or(|shape| !shape.is_empty()) {
            return Err(format!(
                "frame_step must be a scalar, got shape {:?}",
                input_shapes.get(1)
            ));
        }
        let has_window = node.inputs.get(2).is_some_and(Option::is_some);
        let has_frame_length = node.inputs.get(3).is_some_and(Option::is_some);
        if !has_window && !has_frame_length {
            return Err("either optional window or frame_length must be provided".into());
        }
        if has_window && input_dtypes.get(2) != Some(&DataType::Float32) {
            return Err(format!(
                "window dtype must match the Float32 signal, got {:?}",
                input_dtypes.get(2).copied().unwrap_or(DataType::Undefined)
            ));
        }
        if has_frame_length
            && !matches!(input_dtypes.get(3), Some(DataType::Int32 | DataType::Int64))
        {
            return Err(format!(
                "frame_length must be Int32 or Int64, got {:?}",
                input_dtypes.get(3).copied().unwrap_or(DataType::Undefined)
            ));
        }
        if has_frame_length && input_shapes.get(3).is_none_or(|shape| !shape.is_empty()) {
            return Err(format!(
                "frame_length must be a scalar, got shape {:?}",
                input_shapes.get(3)
            ));
        }
        let signal = input_shapes
            .first()
            .ok_or_else(|| "missing signal shape metadata".to_string())?;
        if signal.len() != 3 {
            return Err(format!(
                "signal rank must be 3 [batch, signal_length, 1|2], got {}",
                signal.len()
            ));
        }
        let components = signal[2].as_static();
        if !matches!(components, None | Some(1 | 2)) {
            return Err(format!(
                "signal last dimension must be 1 (real) or 2 (complex), got {components:?}"
            ));
        }
        let onesided = match node
            .attr("onesided")
            .and_then(|attribute| attribute.as_int())
        {
            None | Some(1) => true,
            Some(0) => false,
            Some(value) => return Err(format!("attribute 'onesided' must be 0 or 1, got {value}")),
        };
        if onesided && components == Some(2) {
            return Err("onesided=1 requires a real signal (last dimension 1)".into());
        }
        if signal[1].as_static() == Some(0) {
            return Err("signal length must be greater than zero".into());
        }
        if has_window {
            let window = input_shapes
                .get(2)
                .ok_or_else(|| "missing window shape metadata".to_string())?;
            if window.len() != 1 {
                return Err(format!("window rank must be 1, got {}", window.len()));
            }
            if window[0].as_static() == Some(0) {
                return Err("window length must be greater than zero".into());
            }
            if !has_frame_length
                && let (Some(signal_length), Some(window_length)) =
                    (signal[1].as_static(), window[0].as_static())
                && window_length > signal_length
            {
                return Err(format!(
                    "frame length {window_length} exceeds signal length {signal_length}; STFT \
                     uses complete unpadded frames"
                ));
            }
        }
        for index in 0..node.inputs.len() {
            if node.inputs[index].is_none() {
                continue;
            }
            if let Some(layout) = input_layouts.get(index)
                && layout.strides.is_some()
            {
                let Some(shape) = input_shapes
                    .get(index)
                    .and_then(|shape| onnx_runtime_ir::as_static_shape(shape))
                else {
                    return Err(format!(
                        "input {index} has explicit strides but symbolic shape metadata, so \
                         contiguous layout cannot be proven"
                    ));
                };
                if !layout.is_contiguous(&shape) {
                    return Err(format!("input {index} layout must be contiguous"));
                }
            }
        }
        Ok(())
    };
    reject().err().map(|reason| format!("STFT: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_batches_every_frame_into_one_plan() {
        let spec = StftSpec {
            signal_shape: [2, 16, 1],
            frame_step: 3,
            frame_length: 5,
            frames: 4,
            bins: 3,
            transforms: 8,
            input_kind: DftInputKind::Real,
        };
        let layout = WorkspaceLayout::new(&spec, 640).unwrap();
        assert_eq!(spec.key(0).batch, 8);
        assert_eq!(layout.work_offset, 512);
        assert_eq!(layout.total_bytes, 1152);
    }
}

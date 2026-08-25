//! CUDA `DFT` backed by cuFFT.
//!
//! ONNX permits the signal on any non-component axis, while cuFFT is most
//! efficient over contiguous batches. Small NVRTC pack/unpack kernels bridge
//! those layouts and also implement truncation/zero-padding, real-to-complex
//! promotion, onesided output, and inverse normalization. The packed-batch and
//! plan-cache seam is intentionally reusable by a future STFT implementation;
//! framing/windowing remains outside this module.

use core::ffi::c_void;
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
const MODULE: &str = "dft_pack_v1";
const SOURCE: &str = r#"
extern "C" __global__ void dft_pack_f32(
    const float* input, float2* packed, const unsigned long long* metadata,
    int rank, int axis, unsigned long long signal_length,
    unsigned long long dft_length, unsigned long long batch,
    int complex_components) {
  const unsigned long long* dims = metadata;
  const unsigned long long* strides = metadata + rank;
  const unsigned long long total = batch * dft_length;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long signal_index = index % dft_length;
    unsigned long long batch_index = index / dft_length;
    float2 value = make_float2(0.0f, 0.0f);
    if (signal_index < signal_length) {
      unsigned long long offset = signal_index * strides[axis];
      for (int dim = rank - 2; dim >= 0; --dim) {
        if (dim == axis) continue;
        const unsigned long long coordinate = batch_index % dims[dim];
        batch_index /= dims[dim];
        offset += coordinate * strides[dim];
      }
      value.x = input[offset];
      if (complex_components == 2) value.y = input[offset + 1];
    }
    packed[index] = value;
  }
}

extern "C" __global__ void dft_unpack_f32(
    const float2* packed, float* output, const unsigned long long* metadata,
    int rank, int axis, unsigned long long dft_length,
    unsigned long long output_length, unsigned long long batch, float scale) {
  const unsigned long long* dims = metadata + 2 * rank;
  const unsigned long long* strides = metadata + 3 * rank;
  const unsigned long long total = batch * output_length;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long signal_index = index % output_length;
    unsigned long long batch_index = index / output_length;
    unsigned long long offset = signal_index * strides[axis];
    for (int dim = rank - 2; dim >= 0; --dim) {
      if (dim == axis) continue;
      const unsigned long long coordinate = batch_index % dims[dim];
      batch_index /= dims[dim];
      offset += coordinate * strides[dim];
    }
    const float2 value = packed[(index / output_length) * dft_length + signal_index];
    output[offset] = value.x * scale;
    output[offset + 1] = value.y * scale;
  }
}
"#;

pub(crate) struct DftFactory {
    pub(crate) runtime: Arc<CudaRuntime>,
    pub(crate) plans: Arc<CufftPlanCache>,
}

impl KernelFactory for DftFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let inverse = bool_attr(node, "inverse")?;
        let onesided = bool_attr(node, "onesided")?;
        Ok(Box::new(DftKernel {
            runtime: self.runtime.clone(),
            plans: self.plans.clone(),
            inverse,
            onesided,
            axis_attr: node.attr("axis").and_then(|attribute| attribute.as_int()),
            prepared_plan: Mutex::new(None),
        }))
    }
}

fn bool_attr(node: &Node, name: &str) -> Result<bool> {
    match node.attr(name).and_then(|attribute| attribute.as_int()) {
        None | Some(0) => Ok(false),
        Some(1) => Ok(true),
        Some(value) => Err(EpError::KernelFailed(format!(
            "cuda_ep DFT: attribute '{name}' must be 0 or 1, got {value}"
        ))),
    }
}

#[derive(Clone, Debug)]
struct DftSpec {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    axis: usize,
    signal_length: usize,
    dft_length: usize,
    output_length: usize,
    batch: usize,
    input_kind: DftInputKind,
    direction: DftDirection,
}

impl DftSpec {
    fn key(&self, device: u32) -> CufftPlanKey {
        CufftPlanKey {
            device,
            dtype: DataType::Float32,
            input_kind: self.input_kind,
            rank: self.input_shape.len(),
            axis: self.axis,
            length: self.dft_length,
            batch: self.batch,
            direction: self.direction,
        }
    }
}

struct WorkspaceLayout {
    packed_offset: usize,
    work_offset: usize,
    total_bytes: usize,
}

impl WorkspaceLayout {
    fn new(spec: &DftSpec, work_bytes: usize) -> Result<Self> {
        let rank_metadata = spec
            .input_shape
            .len()
            .checked_mul(4)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u64>()))
            .ok_or_else(|| EpError::KernelFailed("cuda_ep DFT: metadata size overflow".into()))?;
        let metadata_bytes = align_up(rank_metadata, WORKSPACE_ALIGNMENT)?;
        let packed_bytes = spec
            .batch
            .checked_mul(spec.dft_length)
            .and_then(|elements| elements.checked_mul(2 * std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep DFT: packed complex workspace size overflow".into())
            })?;
        let packed_offset = metadata_bytes;
        let work_offset = align_up(
            packed_offset.checked_add(packed_bytes).ok_or_else(|| {
                EpError::KernelFailed("cuda_ep DFT: packed workspace offset overflow".into())
            })?,
            WORKSPACE_ALIGNMENT,
        )?;
        let total_bytes = work_offset.checked_add(work_bytes).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep DFT: cuFFT workspace size overflow".into())
        })?;
        Ok(Self {
            packed_offset,
            work_offset,
            total_bytes,
        })
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| EpError::KernelFailed("cuda_ep DFT: workspace alignment overflow".into()))
}

struct DftKernel {
    runtime: Arc<CudaRuntime>,
    plans: Arc<CufftPlanCache>,
    inverse: bool,
    onesided: bool,
    axis_attr: Option<i64>,
    prepared_plan: Mutex<Option<(CufftPlanKey, Arc<Mutex<CufftPlan>>)>>,
}

impl DftKernel {
    fn resolve_from_views(&self, inputs: &[TensorView]) -> Result<DftSpec> {
        if inputs.is_empty() || inputs.len() > 3 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep DFT: expected 1..=3 inputs, got {}",
                inputs.len()
            )));
        }
        let input = &inputs[0];
        if input.is_absent() {
            return Err(EpError::KernelFailed(
                "cuda_ep DFT: data input must be present".into(),
            ));
        }
        if input.dtype != DataType::Float32 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep DFT: data input must be Float32, got {:?}; unsupported dtypes must \
                 decline at placement time",
                input.dtype
            )));
        }
        let axis = if let Some(axis) = present_input(inputs, 2) {
            scalar_i64(&self.runtime, axis, "axis")?
        } else {
            self.axis_attr.unwrap_or(-2)
        };
        let normalized_axis = normalize_axis(axis, input.shape.len())?;
        let dft_length = if let Some(length) = present_input(inputs, 1) {
            scalar_i64(&self.runtime, length, "dft_length")?
        } else {
            i64::try_from(input.shape[normalized_axis]).map_err(|_| {
                EpError::KernelFailed("cuda_ep DFT: input signal length exceeds i64".into())
            })?
        };
        make_spec(input.shape, axis, dft_length, self.inverse, self.onesided)
    }

    fn resolve_from_metadata(&self, inputs: &[TensorMetadata<'_>]) -> Result<Option<DftSpec>> {
        let Some(input) = inputs.first().filter(|input| input.present) else {
            return Ok(None);
        };
        if inputs.get(1).is_some_and(|input| input.present)
            || inputs.get(2).is_some_and(|input| input.present)
        {
            return Ok(None);
        }
        let axis = self.axis_attr.unwrap_or(-2);
        let signal_length = input.shape[normalize_axis(axis, input.shape.len())?];
        let dft_length = i64::try_from(signal_length).map_err(|_| {
            EpError::KernelFailed("cuda_ep DFT: input signal length exceeds i64".into())
        })?;
        Ok(Some(make_spec(
            input.shape,
            axis,
            dft_length,
            self.inverse,
            self.onesided,
        )?))
    }

    fn plan(&self, spec: &DftSpec) -> Result<Arc<Mutex<CufftPlan>>> {
        let key = spec.key(self.runtime.ordinal());
        let mut prepared = self.prepared_plan.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep DFT: prepared-plan lock was poisoned".into())
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

    fn workspace_for(&self, spec: &DftSpec) -> Result<WorkspaceRequirement> {
        if spec.batch == 0 {
            return Ok(WorkspaceRequirement::NONE);
        }
        let plan = self.plan(spec)?;
        let work_bytes = plan
            .lock()
            .map_err(|_| EpError::KernelFailed("cuda_ep DFT: cuFFT plan lock was poisoned".into()))?
            .work_bytes();
        let layout = WorkspaceLayout::new(spec, work_bytes)?;
        Ok(WorkspaceRequirement {
            bytes: u64::try_from(layout.total_bytes).map_err(|_| {
                EpError::KernelFailed("cuda_ep DFT: workspace size exceeds u64".into())
            })?,
            alignment: WORKSPACE_ALIGNMENT,
            lifetime: WorkspaceLifetime::StepScoped,
            role: MemoryRole::Workspace { step_scoped: true },
        })
    }

    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        if outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep DFT: expected 1 output, got {}",
                outputs.len()
            )));
        }
        let spec = self.resolve_from_views(inputs)?;
        validate_io(&inputs[0], &outputs[0], &spec)?;
        if spec.batch == 0 {
            return Ok(());
        }
        let plan = self.plan(&spec)?;
        let work_bytes = plan
            .lock()
            .map_err(|_| EpError::KernelFailed("cuda_ep DFT: cuFFT plan lock was poisoned".into()))?
            .work_bytes();
        let layout = WorkspaceLayout::new(&spec, work_bytes)?;
        let workspace = workspace.ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep DFT: executor did not provide the required {}-byte governed workspace",
                layout.total_bytes
            ))
        })?;
        if workspace.bytes() < layout.total_bytes {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep DFT: governed workspace is {} bytes, but this transform requires {} bytes",
                workspace.bytes(),
                layout.total_bytes
            )));
        }

        let metadata = metadata(&spec)?;
        let base = cuptr(workspace.ptr().0 as *const c_void);
        // The metadata prefix belongs to step-scoped workspace and may be
        // reused by the next dispatch. Drain the non-blocking compute stream
        // before the synchronous default-stream upload so it cannot overwrite
        // metadata still consumed by a prior DFT. This host barrier is one
        // reason capture remains explicitly unsupported.
        self.runtime
            .stream()
            .synchronize()
            .map_err(|error| driver_err("synchronizing before DFT metadata upload", error))?;
        // SAFETY: workspace covers `layout.total_bytes`; the metadata prefix is
        // within that allocation and contains exactly `metadata.len()` bytes.
        unsafe {
            self.runtime.htod(as_bytes(&metadata), base)?;
        }
        let packed = base
            .checked_add(layout.packed_offset as u64)
            .ok_or_else(|| EpError::KernelFailed("cuda_ep DFT: packed pointer overflow".into()))?;
        let work = base
            .checked_add(layout.work_offset as u64)
            .ok_or_else(|| EpError::KernelFailed("cuda_ep DFT: work pointer overflow".into()))?;

        let pack = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "dft_pack_f32")?;
        let input = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let rank = i32::try_from(spec.input_shape.len())
            .map_err(|_| EpError::KernelFailed("cuda_ep DFT: input rank exceeds i32".into()))?;
        let axis = i32::try_from(spec.axis)
            .map_err(|_| EpError::KernelFailed("cuda_ep DFT: axis exceeds i32".into()))?;
        let signal_length = spec.signal_length as u64;
        let dft_length = spec.dft_length as u64;
        let batch = spec.batch as u64;
        let complex_components = match spec.input_kind {
            DftInputKind::Real => 1i32,
            DftInputKind::Complex => 2i32,
        };
        let packed_elements = spec.batch.checked_mul(spec.dft_length).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep DFT: packed element count overflow".into())
        })?;
        let mut builder = self.runtime.stream().launch_builder(&pack);
        builder
            .arg(&input)
            .arg(&packed)
            .arg(&base)
            .arg(&rank)
            .arg(&axis)
            .arg(&signal_length)
            .arg(&dft_length)
            .arg(&batch)
            .arg(&complex_components);
        // SAFETY: pointers and scalar extents were validated against the input
        // tensor and governed workspace above.
        unsafe { builder.launch(launch_config(packed_elements)) }
            .map_err(|error| driver_err("launching DFT pack kernel", error))?;

        {
            let mut plan = plan.lock().map_err(|_| {
                EpError::KernelFailed("cuda_ep DFT: cuFFT plan lock was poisoned".into())
            })?;
            // SAFETY: `packed` covers batch*dft_length complex f32 values and
            // `work` covers the exact work size queried from this plan.
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

        let unpack = self
            .runtime
            .nvrtc_function(MODULE, SOURCE, "dft_unpack_f32")?;
        let output = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let output_length = spec.output_length as u64;
        let scale = if self.inverse {
            1.0f32 / spec.dft_length as f32
        } else {
            1.0
        };
        let output_elements = spec.batch.checked_mul(spec.output_length).ok_or_else(|| {
            EpError::KernelFailed("cuda_ep DFT: output element count overflow".into())
        })?;
        let mut builder = self.runtime.stream().launch_builder(&unpack);
        builder
            .arg(&packed)
            .arg(&output)
            .arg(&base)
            .arg(&rank)
            .arg(&axis)
            .arg(&dft_length)
            .arg(&output_length)
            .arg(&batch)
            .arg(&scale);
        // SAFETY: packed/output pointers and metadata describe the validated
        // contiguous tensors, and every indexed output has two f32 components.
        unsafe { builder.launch(launch_config(output_elements)) }
            .map_err(|error| driver_err("launching DFT unpack kernel", error))?;
        Ok(())
    }
}

impl Kernel for DftKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }

    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        match self.resolve_from_metadata(inputs)? {
            Some(spec) => self.workspace_for(&spec),
            None => Ok(WorkspaceRequirement::NONE),
        }
    }

    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        _metadata: &[TensorMetadata<'_>],
    ) -> Result<WorkspaceRequirement> {
        self.workspace_for(&self.resolve_from_views(inputs)?)
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
            "cuFFT plan selection and runtime scalar/metadata staging are not CUDA-graph capture-safe",
        )
    }
}

fn make_spec(
    input_shape: &[usize],
    axis_raw: i64,
    dft_length_raw: i64,
    inverse: bool,
    onesided: bool,
) -> Result<DftSpec> {
    let rank = input_shape.len();
    if rank < 2 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep DFT: input rank must be at least 2 (signal plus complex component), got {rank}"
        )));
    }
    let axis = normalize_axis(axis_raw, rank)?;
    let complex_components = input_shape[rank - 1];
    let input_kind = match complex_components {
        1 => DftInputKind::Real,
        2 => DftInputKind::Complex,
        value => {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep DFT: last input dimension must be 1 (real) or 2 (complex), got {value}"
            )));
        }
    };
    if onesided && input_kind != DftInputKind::Real {
        return Err(EpError::KernelFailed(
            "cuda_ep DFT: onesided=1 is valid only for real input (last dimension 1)".into(),
        ));
    }
    let dft_length = usize::try_from(dft_length_raw)
        .ok()
        .filter(|length| *length > 0)
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep DFT: dft_length must be positive, got {dft_length_raw}"
            ))
        })?;
    let output_length = if onesided {
        dft_length / 2 + 1
    } else {
        dft_length
    };
    let batch = input_shape
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != axis && *index != rank - 1)
        .try_fold(1usize, |product, (_, &dimension)| {
            product.checked_mul(dimension).ok_or_else(|| {
                EpError::KernelFailed("cuda_ep DFT: batch dimension product overflow".into())
            })
        })?;
    let mut output_shape = input_shape.to_vec();
    output_shape[axis] = output_length;
    output_shape[rank - 1] = 2;
    Ok(DftSpec {
        input_shape: input_shape.to_vec(),
        output_shape,
        axis,
        signal_length: input_shape[axis],
        dft_length,
        output_length,
        batch,
        input_kind,
        direction: if inverse {
            DftDirection::Inverse
        } else {
            DftDirection::Forward
        },
    })
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let rank_i64 = i64::try_from(rank)
        .map_err(|_| EpError::KernelFailed("cuda_ep DFT: input rank exceeds i64".into()))?;
    let normalized = if axis < 0 { axis + rank_i64 } else { axis };
    if normalized < 0 || normalized >= rank_i64 - 1 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep DFT: axis {axis} is invalid for rank {rank}; the last dimension is the \
             complex component and cannot be transformed"
        )));
    }
    Ok(normalized as usize)
}

fn present_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn scalar_i64(runtime: &CudaRuntime, input: &TensorView, name: &str) -> Result<i64> {
    if input.dtype != DataType::Int64 || input.numel() != 1 || !input.is_contiguous() {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep DFT: {name} must be a contiguous Int64 scalar, got dtype {:?}, shape {:?}",
            input.dtype, input.shape
        )));
    }
    let mut bytes = [0u8; std::mem::size_of::<i64>()];
    // SAFETY: the validated scalar input covers exactly eight readable bytes.
    unsafe {
        runtime.dtoh(&mut bytes, cuptr(input.data_ptr::<u8>() as *const c_void))?;
    }
    Ok(i64::from_ne_bytes(bytes))
}

fn validate_io(input: &TensorView, output: &TensorMut, spec: &DftSpec) -> Result<()> {
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(EpError::KernelFailed(
            "cuda_ep DFT: input and output must be contiguous; strided layouts must decline at \
             placement time"
                .into(),
        ));
    }
    if output.dtype != DataType::Float32 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep DFT: output must be Float32, got {:?}",
            output.dtype
        )));
    }
    if output.shape != spec.output_shape {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep DFT: output shape {:?} does not match expected {:?}",
            output.shape, spec.output_shape
        )));
    }
    Ok(())
}

fn contiguous_strides(shape: &[usize]) -> Result<Vec<u64>> {
    let mut strides = vec![1u64; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1]
            .checked_mul(shape[index + 1] as u64)
            .ok_or_else(|| EpError::KernelFailed("cuda_ep DFT: stride overflow".into()))?;
    }
    Ok(strides)
}

fn metadata(spec: &DftSpec) -> Result<Vec<u64>> {
    let mut values = Vec::with_capacity(spec.input_shape.len() * 4);
    values.extend(spec.input_shape.iter().map(|&value| value as u64));
    values.extend(contiguous_strides(&spec.input_shape)?);
    values.extend(spec.output_shape.iter().map(|&value| value as u64));
    values.extend(contiguous_strides(&spec.output_shape)?);
    Ok(values)
}

fn as_bytes(values: &[u64]) -> &[u8] {
    // SAFETY: u64 is plain data and the resulting byte slice borrows `values`.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
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

/// Claim-time DFT validation. Dynamic scalar values are checked at execution,
/// but dtype, known shape, attributes, and data layout fail closed here.
pub(crate) fn unsupported_reason(
    node: &Node,
    input_shapes: &[Shape],
    input_dtypes: &[DataType],
    input_layouts: &[TensorLayout],
) -> Option<String> {
    let reject = || -> std::result::Result<(), String> {
        if node.outputs.len() != 1 || node.inputs.is_empty() || node.inputs.len() > 3 {
            return Err("requires 1..=3 inputs and exactly 1 output".into());
        }
        if node.inputs[0].is_none() {
            return Err("data input must be present".into());
        }
        if input_dtypes.first() != Some(&DataType::Float32) {
            return Err(format!(
                "data input dtype must be Float32 for the cuFFT path, got {:?}",
                input_dtypes.first().copied().unwrap_or(DataType::Undefined)
            ));
        }
        for (index, name) in [(1usize, "dft_length"), (2usize, "axis")] {
            if node.inputs.get(index).is_some_and(Option::is_some)
                && input_dtypes.get(index) != Some(&DataType::Int64)
            {
                return Err(format!(
                    "input {index} ('{name}') must be Int64, got {:?}",
                    input_dtypes
                        .get(index)
                        .copied()
                        .unwrap_or(DataType::Undefined)
                ));
            }
        }
        let shape = input_shapes
            .first()
            .ok_or_else(|| "missing data input shape metadata".to_string())?;
        if shape.len() < 2 {
            return Err(format!(
                "data input rank must be at least 2, got {}",
                shape.len()
            ));
        }
        if let Some(layout) = input_layouts.first()
            && let Some(concrete) = onnx_runtime_ir::as_static_shape(shape)
            && !layout.is_contiguous(&concrete)
        {
            return Err("data input layout must be contiguous".into());
        }
        let complex_components = shape.last().and_then(|dimension| dimension.as_static());
        if !matches!(complex_components, None | Some(1 | 2)) {
            return Err(format!(
                "last input dimension must be 1 (real) or 2 (complex), got {complex_components:?}"
            ));
        }
        let onesided = match node
            .attr("onesided")
            .and_then(|attribute| attribute.as_int())
        {
            None | Some(0) => false,
            Some(1) => true,
            Some(value) => return Err(format!("attribute 'onesided' must be 0 or 1, got {value}")),
        };
        match node
            .attr("inverse")
            .and_then(|attribute| attribute.as_int())
        {
            None | Some(0 | 1) => {}
            Some(value) => return Err(format!("attribute 'inverse' must be 0 or 1, got {value}")),
        }
        if onesided && complex_components == Some(2) {
            return Err("onesided=1 is valid only for real input (last dimension 1)".into());
        }
        if node.inputs.get(2).is_none_or(Option::is_none) {
            let axis = node
                .attr("axis")
                .and_then(|attribute| attribute.as_int())
                .unwrap_or(-2);
            normalize_axis(axis, shape.len()).map_err(|error| error.to_string())?;
        }
        Ok(())
    };
    reject().err().map(|reason| format!("DFT: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_layout_is_aligned_and_contains_all_regions() {
        let spec = make_spec(&[2, 4, 1], -2, 7, false, false).unwrap();
        let layout = WorkspaceLayout::new(&spec, 123).unwrap();
        assert_eq!(layout.packed_offset % WORKSPACE_ALIGNMENT, 0);
        assert_eq!(layout.work_offset % WORKSPACE_ALIGNMENT, 0);
        assert!(spec.input_shape.len() * 4 * std::mem::size_of::<u64>() <= layout.packed_offset);
        assert!(layout.total_bytes >= layout.work_offset + 123);
    }

    #[test]
    fn onesided_rejects_complex_input() {
        let error = make_spec(&[1, 4, 2], -2, 4, false, true).unwrap_err();
        assert!(error.to_string().contains("valid only for real input"));
    }

    #[test]
    fn output_shape_tracks_arbitrary_axis_and_onesided_plus_one() {
        let spec = make_spec(&[2, 3, 5, 1], 1, 8, false, true).unwrap();
        assert_eq!(spec.output_shape, [2, 5, 5, 2]);
        assert_eq!(spec.batch, 10);
    }
}

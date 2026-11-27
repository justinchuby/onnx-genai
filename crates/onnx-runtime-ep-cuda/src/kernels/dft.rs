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
use onnx_runtime_ir::dft::{self as dft_rules, DftMode};
use onnx_runtime_ir::{DataType, Node, Shape, TensorLayout};

use crate::cufft::{CufftPlan, CufftPlanCache, CufftPlanKey, DftDirection, DftInputKind};
use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const WORKSPACE_ALIGNMENT: usize = 256;
const MODULE: &str = "dft_pack_v2";
const SOURCE: &str = r#"
template <typename T> struct Direct {
    using Storage = T;
    __device__ static T load(T value) { return value; }
    __device__ static T store(T value) { return value; }
};

struct Half {
    using Storage = unsigned short;
    __device__ static float load(Storage value) {
        float result;
        asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
        return result;
    }
    __device__ static Storage store(float value) {
        Storage result;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
        return result;
    }
};

struct BFloat {
    using Storage = unsigned short;
    __device__ static float load(Storage value) {
        return __uint_as_float((unsigned int)value << 16);
    }
    __device__ static Storage store(float value) {
        unsigned int bits = __float_as_uint(value);
        if ((bits & 0x7fffffffu) > 0x7f800000u) return (bits >> 16) | 0x0040u;
        return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
    }
};

template <typename Codec, typename Packed>
__device__ void dft_pack(
    const typename Codec::Storage* input, Packed* packed, const unsigned long long* metadata,
    int rank, int axis, unsigned long long signal_length, unsigned long long input_bins,
    unsigned long long dft_length, unsigned long long batch,
    int complex_components, int inverse_real) {
  const unsigned long long* dims = metadata;
  const unsigned long long* strides = metadata + rank;
  const unsigned long long total = batch * dft_length;
  for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long bin = index % dft_length;
    const bool mirrored = inverse_real && bin >= input_bins;
    const unsigned long long signal_index = mirrored ? dft_length - bin : bin;
    unsigned long long batch_index = index / dft_length;
    Packed value = {};
    if (signal_index < signal_length) {
      unsigned long long offset = signal_index * strides[axis];
      for (int dim = rank - 2; dim >= 0; --dim) {
        if (dim == axis) continue;
        const unsigned long long coordinate = batch_index % dims[dim];
        batch_index /= dims[dim];
        offset += coordinate * strides[dim];
      }
      value.x = Codec::load(input[offset]);
      // DC and even Nyquist are real even when their supplied imaginary values are nonzero.
      if (complex_components == 2 &&
          (!inverse_real || (bin != 0 && bin != dft_length - bin))) {
        value.y = Codec::load(input[offset + 1]);
        if (mirrored) value.y = -value.y;
      }
    }
    packed[index] = value;
  }
}

template <typename Codec, typename Packed, typename Compute>
__device__ void dft_unpack(
    const Packed* packed, typename Codec::Storage* output, const unsigned long long* metadata,
    int rank, int axis, unsigned long long dft_length,
    unsigned long long output_length, unsigned long long batch, int output_components, Compute scale) {
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
    const Packed value = packed[(index / output_length) * dft_length + signal_index];
    output[offset] = Codec::store(value.x * scale);
    if (output_components == 2) output[offset + 1] = Codec::store(value.y * scale);
  }
}

#define DFT_KERNELS(SUFFIX, CODEC, PACKED, COMPUTE) \
extern "C" __global__ void dft_pack_##SUFFIX( \
    const CODEC::Storage* input, PACKED* packed, const unsigned long long* metadata, \
    int rank, int axis, unsigned long long signal_length, unsigned long long input_bins, \
    unsigned long long dft_length, unsigned long long batch, int components, int inverse_real) { \
    dft_pack<CODEC>(input, packed, metadata, rank, axis, signal_length, input_bins, \
                    dft_length, batch, components, inverse_real); \
} \
extern "C" __global__ void dft_unpack_##SUFFIX( \
    const PACKED* packed, CODEC::Storage* output, const unsigned long long* metadata, \
    int rank, int axis, unsigned long long dft_length, unsigned long long output_length, \
    unsigned long long batch, int components, COMPUTE scale) { \
    dft_unpack<CODEC>(packed, output, metadata, rank, axis, dft_length, output_length, \
                      batch, components, scale); \
}

DFT_KERNELS(f32, Direct<float>, float2, float)
DFT_KERNELS(f64, Direct<double>, double2, double)
DFT_KERNELS(f16, Half, float2, float)
DFT_KERNELS(bf16, BFloat, float2, float)
"#;

pub(crate) struct DftFactory {
    pub(crate) runtime: Arc<CudaRuntime>,
    pub(crate) plans: Arc<CufftPlanCache>,
}

impl KernelFactory for DftFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let opset = node.local_opset().unwrap_or(17);
        let mode = supported_mode(node, opset).map_err(dft_error)?;
        Ok(Box::new(DftKernel {
            runtime: self.runtime.clone(),
            plans: self.plans.clone(),
            mode,
            opset,
            default_axis: dft_rules::default_axis(node, opset),
            prepared_plan: Mutex::new(None),
        }))
    }
}

fn supported_mode(node: &Node, opset: u64) -> std::result::Result<DftMode, String> {
    let mode = DftMode::parse(node)?;
    if opset < 20 && mode == DftMode::RealInverse {
        return Err(
            "CUDA IRFFT requires DFT opset 20 or newer; select the CPU EP for opset 17".into(),
        );
    }
    Ok(mode)
}

#[derive(Clone, Debug)]
struct DftSpec {
    dtype: DataType,
    mode: DftMode,
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
    fn packed_dtype(&self) -> DataType {
        if self.dtype == DataType::Float64 {
            DataType::Float64
        } else {
            DataType::Float32
        }
    }

    fn key(&self, device: u32) -> CufftPlanKey {
        CufftPlanKey {
            device,
            dtype: self.packed_dtype(),
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
            .and_then(|elements| elements.checked_mul(2 * spec.packed_dtype().storage_bytes(1)))
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
    mode: DftMode,
    opset: u64,
    default_axis: i64,
    prepared_plan: Mutex<Option<(CufftPlanKey, Arc<Mutex<CufftPlan>>)>>,
}

impl DftKernel {
    fn resolve_from_views(&self, inputs: &[TensorView]) -> Result<DftSpec> {
        let max_inputs = if self.opset >= 20 { 3 } else { 2 };
        if inputs.is_empty() || inputs.len() > max_inputs {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep DFT: opset {} expects 1..={max_inputs} inputs, got {}",
                self.opset,
                inputs.len()
            )));
        }
        let input = &inputs[0];
        if input.is_absent() {
            return Err(EpError::KernelFailed(
                "cuda_ep DFT: data input must be present".into(),
            ));
        }
        let axis = if let Some(axis) = present_input(inputs, 2) {
            scalar_int(&self.runtime, axis, 2, self.opset)?
        } else {
            self.default_axis
        };
        let dft_length = present_input(inputs, 1)
            .map(|length| scalar_int(&self.runtime, length, 1, self.opset))
            .transpose()?;
        self.spec(input.shape, input.dtype, axis, dft_length)
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
        Ok(Some(self.spec(
            input.shape,
            input.dtype,
            self.default_axis,
            None,
        )?))
    }

    fn spec(
        &self,
        shape: &[usize],
        dtype: DataType,
        axis: i64,
        length: Option<i64>,
    ) -> Result<DftSpec> {
        data_kernels(self.opset, dtype).map_err(dft_error)?;
        make_spec(shape, dtype, axis, length, self.mode)
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
        // reused by the next dispatch. The synchronous H2D helper drains compute
        // before the default-stream copy, then waits for that copy, so a prior
        // DFT retires before this overwrite. That host barrier is one reason
        // capture remains explicitly unsupported.
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

        let (pack_name, unpack_name) = data_kernels(self.opset, spec.dtype).map_err(dft_error)?;
        let pack = self.runtime.nvrtc_function(MODULE, SOURCE, pack_name)?;
        let input = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let rank = i32::try_from(spec.input_shape.len())
            .map_err(|_| EpError::KernelFailed("cuda_ep DFT: input rank exceeds i32".into()))?;
        let axis = i32::try_from(spec.axis)
            .map_err(|_| EpError::KernelFailed("cuda_ep DFT: axis exceeds i32".into()))?;
        let signal_length = spec.signal_length as u64;
        let dft_length = spec.dft_length as u64;
        let input_bins = spec.mode.input_length(spec.dft_length) as u64;
        let inverse_real = i32::from(spec.mode == DftMode::RealInverse);
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
            .arg(&input_bins)
            .arg(&dft_length)
            .arg(&batch)
            .arg(&complex_components)
            .arg(&inverse_real);
        // SAFETY: pointers and scalar extents were validated against the input
        // tensor and governed workspace above.
        unsafe { builder.launch(launch_config(packed_elements)) }
            .map_err(|error| driver_err("launching DFT pack kernel", error))?;

        {
            let mut plan = plan.lock().map_err(|_| {
                EpError::KernelFailed("cuda_ep DFT: cuFFT plan lock was poisoned".into())
            })?;
            // SAFETY: `packed` covers batch*dft_length complex values of the plan's precision and
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

        let unpack = self.runtime.nvrtc_function(MODULE, SOURCE, unpack_name)?;
        let output = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let output_length = spec.output_length as u64;
        let output_components = spec.mode.output_components() as i32;
        let scale_f32 = if spec.mode.inverse() {
            1.0f32 / spec.dft_length as f32
        } else {
            1.0
        };
        let scale_f64 = if spec.mode.inverse() {
            1.0f64 / spec.dft_length as f64
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
            .arg(&output_components);
        if spec.dtype == DataType::Float64 {
            builder.arg(&scale_f64);
        } else {
            builder.arg(&scale_f32);
        }
        // SAFETY: packed/output pointers and metadata describe the validated
        // contiguous tensors, with one or two output components per sample.
        unsafe { builder.launch(launch_config(output_elements)) }
            .map_err(|error| driver_err("launching DFT unpack kernel", error))?;
        Ok(())
    }
}

impl Kernel for DftKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<onnx_runtime_ep_api::WorkspacePreparation> {
        match self.resolve_from_metadata(inputs)? {
            Some(spec) => self
                .workspace_for(&spec)
                .map(onnx_runtime_ep_api::WorkspacePreparation::Ready),
            None => Ok(onnx_runtime_ep_api::WorkspacePreparation::Deferred),
        }
    }

    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        _output_views: &[TensorMut],
        _metadata: &[TensorMetadata<'_>],
        _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
        _cached: Option<onnx_runtime_ep_api::WorkspacePreparation>,
    ) -> Result<onnx_runtime_ep_api::WorkspaceExecutionPreparation> {
        self.workspace_for(&self.resolve_from_views(inputs)?)
            .map(onnx_runtime_ep_api::WorkspaceExecutionPreparation::Ready)
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
    dtype: DataType,
    axis_raw: i64,
    length: Option<i64>,
    mode: DftMode,
) -> Result<DftSpec> {
    let rank = input_shape.len();
    if rank < 2 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep DFT: input rank must be at least 2 (signal plus complex component), got {rank}"
        )));
    }
    let axis = dft_rules::normalize_axis(axis_raw, rank).map_err(dft_error)?;
    let complex_components = input_shape[rank - 1];
    mode.validate_components(complex_components)
        .map_err(dft_error)?;
    let input_kind = if complex_components == 1 {
        DftInputKind::Real
    } else {
        DftInputKind::Complex
    };
    let explicit = length
        .map(dft_rules::explicit_length)
        .transpose()
        .map_err(dft_error)?;
    let dft_length = explicit
        .map_or_else(|| mode.default_length(input_shape[axis]), Ok)
        .map_err(dft_error)?;
    let output_length = mode.output_length(dft_length);
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
    output_shape[rank - 1] = mode.output_components();
    Ok(DftSpec {
        dtype,
        mode,
        input_shape: input_shape.to_vec(),
        output_shape,
        axis,
        signal_length: input_shape[axis],
        dft_length,
        output_length,
        batch,
        input_kind,
        direction: if mode.inverse() {
            DftDirection::Inverse
        } else {
            DftDirection::Forward
        },
    })
}

fn dft_error(reason: String) -> EpError {
    EpError::KernelFailed(format!("cuda_ep DFT: {reason}"))
}

fn data_kernels(
    opset: u64,
    dtype: DataType,
) -> std::result::Result<(&'static str, &'static str), String> {
    match (opset >= 20, dtype) {
        (_, DataType::Float32) => Ok(("dft_pack_f32", "dft_unpack_f32")),
        (true, DataType::Float64) => Ok(("dft_pack_f64", "dft_unpack_f64")),
        (true, DataType::Float16) => Ok(("dft_pack_f16", "dft_unpack_f16")),
        (true, DataType::BFloat16) => Ok(("dft_pack_bf16", "dft_unpack_bf16")),
        _ => Err(format!(
            "unsupported data dtype {dtype:?} for DFT opset {opset} on CUDA"
        )),
    }
}

fn present_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn validate_scalar_dtype(
    dtype: DataType,
    index: usize,
    opset: u64,
) -> std::result::Result<(), String> {
    let length_v20 = index == 1 && opset >= 20;
    if dtype == DataType::Int64 || (length_v20 && dft_rules::is_length_dtype(dtype)) {
        Ok(())
    } else {
        Err(format!(
            "{} must have dtype {}, got {dtype:?}",
            if index == 1 { "dft_length" } else { "axis" },
            if length_v20 {
                "Int32 or Int64"
            } else {
                "Int64"
            }
        ))
    }
}

fn scalar_int(runtime: &CudaRuntime, input: &TensorView, index: usize, opset: u64) -> Result<i64> {
    validate_scalar_dtype(input.dtype, index, opset).map_err(dft_error)?;
    if !input.shape.is_empty() || !input.is_contiguous() {
        return Err(dft_error(format!(
            "{} must be a contiguous rank-zero scalar, got shape {:?}",
            if index == 1 { "dft_length" } else { "axis" },
            input.shape
        )));
    }
    let mut bytes = [0u8; std::mem::size_of::<i64>()];
    let size = input.dtype.storage_bytes(1);
    // SAFETY: the validated scalar input covers exactly `size` readable bytes.
    unsafe {
        runtime.dtoh(
            &mut bytes[..size],
            cuptr(input.data_ptr::<u8>() as *const c_void),
        )?;
    }
    Ok(if input.dtype == DataType::Int32 {
        i64::from(i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    } else {
        i64::from_ne_bytes(bytes)
    })
}

fn validate_io(input: &TensorView, output: &TensorMut, spec: &DftSpec) -> Result<()> {
    if !input.is_contiguous() || !output.is_contiguous() {
        return Err(EpError::KernelFailed(
            "cuda_ep DFT: input and output must be contiguous; strided layouts must decline at \
             placement time"
                .into(),
        ));
    }
    if output.dtype != spec.dtype {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep DFT: output dtype must match input {:?}, got {:?}",
            spec.dtype, output.dtype
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
    opset: u64,
    input_shapes: &[Shape],
    input_dtypes: &[DataType],
    input_layouts: &[TensorLayout],
) -> Option<String> {
    let reject = || -> std::result::Result<(), String> {
        let max_inputs = if opset >= 20 { 3 } else { 2 };
        if node.outputs.len() != 1 || node.inputs.is_empty() || node.inputs.len() > max_inputs {
            return Err(format!(
                "opset {opset} requires 1..={max_inputs} inputs and exactly 1 output"
            ));
        }
        if node.inputs[0].is_none() {
            return Err("data input must be present".into());
        }
        data_kernels(
            opset,
            input_dtypes.first().copied().unwrap_or(DataType::Undefined),
        )?;
        for (index, name) in [(1usize, "dft_length"), (2usize, "axis")] {
            if node.inputs.get(index).is_some_and(Option::is_some) {
                validate_scalar_dtype(
                    input_dtypes
                        .get(index)
                        .copied()
                        .unwrap_or(DataType::Undefined),
                    index,
                    opset,
                )?;
                if let Some(shape) = input_shapes.get(index)
                    && !shape.is_empty()
                {
                    return Err(format!("{name} must be a rank-zero scalar, got {shape:?}"));
                }
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
        let mode = supported_mode(node, opset)?;
        if let Some(components) = shape.last().and_then(|dimension| dimension.as_static()) {
            mode.validate_components(components)?;
        }
        if node.inputs.get(2).is_none_or(Option::is_none) {
            let axis =
                dft_rules::normalize_axis(dft_rules::default_axis(node, opset), shape.len())?;
            if node.inputs.get(1).is_none_or(Option::is_none)
                && let Some(extent) = shape[axis].as_static()
            {
                mode.default_length(extent)?;
            }
        }
        Ok(())
    };
    reject().err().map(|reason| format!("DFT: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opset17_real_inverse_still_declines_before_runtime_access() {
        let mut node = Node::new(onnx_runtime_ir::NodeId(0), "DFT", vec![], vec![]);
        for name in ["inverse", "onesided"] {
            node.attributes
                .insert(name.into(), onnx_runtime_ir::Attribute::Int(1));
        }
        assert!(supported_mode(&node, 17).unwrap_err().contains("CPU EP"));
        assert_eq!(supported_mode(&node, 20).unwrap(), DftMode::RealInverse);
    }

    #[test]
    fn workspace_layout_is_aligned_and_contains_all_regions() {
        let spec = make_spec(&[2, 4, 1], DataType::Float32, -2, Some(7), DftMode::Forward).unwrap();
        let layout = WorkspaceLayout::new(&spec, 123).unwrap();
        assert_eq!(layout.packed_offset % WORKSPACE_ALIGNMENT, 0);
        assert_eq!(layout.work_offset % WORKSPACE_ALIGNMENT, 0);
        assert!(spec.input_shape.len() * 4 * std::mem::size_of::<u64>() <= layout.packed_offset);
        assert!(layout.total_bytes >= layout.work_offset + 123);
    }

    #[test]
    fn onesided_rejects_complex_input() {
        let error = make_spec(
            &[1, 4, 2],
            DataType::Float32,
            -2,
            Some(4),
            DftMode::RealForward,
        )
        .unwrap_err();
        assert!(error.to_string().contains("valid only for real input"));
    }

    #[test]
    fn output_shape_tracks_arbitrary_axis_and_onesided_plus_one() {
        let spec = make_spec(
            &[2, 3, 5, 1],
            DataType::Float32,
            1,
            Some(8),
            DftMode::RealForward,
        )
        .unwrap();
        assert_eq!(spec.output_shape, [2, 5, 5, 2]);
        assert_eq!(spec.batch, 10);
    }
}

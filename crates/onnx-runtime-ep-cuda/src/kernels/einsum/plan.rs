use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::LaunchConfig;
use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{DeviceGraphResource, EpError, Result};
use onnx_runtime_ir::{
    DataType, DeviceId, EinsumAxis, EinsumBinaryContractionPlan, EinsumContractionTreeStep,
    EinsumOperandPlan, EinsumPlannerQuality, EinsumPrecisionPolicy, EinsumShapePlan,
    EinsumSupportedContractionTreeCandidate, EinsumValueId, compute_contiguous_strides,
};

use super::{ArithmeticExecutionSignature, ArithmeticTensorSignature};
use crate::blas::{
    self, CaptureStridedBatchedGemmPlan, GemmDtype, StridedBatchedGemmParams, WORKSPACE_BYTES,
};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, GraphDeviceAllocation, RawCudaFunction};

const BLOCK: u32 = 256;
const WORKSPACE_ALIGNMENT: usize = 256;
pub(super) const DEFAULT_MEMORY_CEILING_BYTES: u128 = 64 * 1024 * 1024;

const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// One immutable metadata record describes one complete Einsum step:
// [out_elems, reduce_elems, iter_rank, out_rank, operands, out_dtype, out_ptr,
//  dims[iter_rank], coordinate_divisors[iter_rank], out_strides[out_rank],
//  operand_record_offsets[operands + 1], operand_ptrs[operands],
//  operand_dtypes[operands], {extent, stride, iteration_axis}[records]]
struct Step {
  const unsigned long long* words;
  unsigned long long output_elements;
  unsigned long long reduction_elements;
  int iteration_rank;
  int output_rank;
  int operands;
  int output_dtype;
  void* output;
  const unsigned long long* dimensions;
  const unsigned long long* divisors;
  const long long* output_strides;
  const unsigned long long* record_offsets;
  const unsigned long long* operand_ptrs;
  const unsigned long long* operand_dtypes;
  const unsigned long long* records;
};

__device__ __forceinline__ Step decode_step(const unsigned long long* words) {
  Step step;
  step.words = words;
  step.output_elements = words[0];
  step.reduction_elements = words[1];
  step.iteration_rank = (int)words[2];
  step.output_rank = (int)words[3];
  step.operands = (int)words[4];
  step.output_dtype = (int)words[5];
  step.output = (void*)words[6];
  step.dimensions = words + 7;
  step.divisors = step.dimensions + step.iteration_rank;
  step.output_strides = (const long long*)(step.divisors + step.iteration_rank);
  step.record_offsets =
      (const unsigned long long*)(step.output_strides + step.output_rank);
  step.operand_ptrs = step.record_offsets + step.operands + 1;
  step.operand_dtypes = step.operand_ptrs + step.operands;
  step.records = step.operand_dtypes + step.operands;
  return step;
}

__device__ __forceinline__ unsigned long long coordinate(
    const Step& step, int axis, unsigned long long output_linear,
    unsigned long long reduction_linear) {
  const unsigned long long linear =
      axis < step.output_rank ? output_linear : reduction_linear;
  const unsigned long long dim = step.dimensions[axis];
  return (linear / step.divisors[axis]) % dim;
}

__device__ __forceinline__ long long operand_offset(
    const Step& step, int operand, unsigned long long output_linear,
    unsigned long long reduction_linear) {
  long long offset = 0;
  const unsigned long long begin = step.record_offsets[operand];
  const unsigned long long end = step.record_offsets[operand + 1];
  for (unsigned long long record = begin; record < end; ++record) {
    const unsigned long long* item = step.records + record * 3;
    const unsigned long long extent = item[0];
    const long long stride = (long long)item[1];
    const int axis = (int)item[2];
    const unsigned long long index =
        extent == 1 && step.dimensions[axis] != 1
            ? 0
            : coordinate(step, axis, output_linear, reduction_linear);
    offset += (long long)index * stride;
  }
  return offset;
}

__device__ __forceinline__ long long output_offset(
    const Step& step, unsigned long long output_linear) {
  long long offset = 0;
  for (int axis = 0; axis < step.output_rank; ++axis) {
    offset += (long long)coordinate(step, axis, output_linear, 0)
              * step.output_strides[axis];
  }
  return offset;
}

__device__ __forceinline__ float load_f32(
    const void* pointer, int dtype, long long offset) {
  if (dtype == 0) return ((const float*)pointer)[offset];
  if (dtype == 1) return __half2float(((const __half*)pointer)[offset]);
  return __bfloat162float(((const __nv_bfloat16*)pointer)[offset]);
}

__device__ __forceinline__ void store_f32(
    void* pointer, int dtype, long long offset, float value) {
  if (dtype == 0) {
    ((float*)pointer)[offset] = value;
    return;
  }
  if (dtype == 2) {
    ((__nv_bfloat16*)pointer)[offset] = __float2bfloat16_rn(value);
    return;
  }
  ((__half*)pointer)[offset] = __float2half_rn(value);
}

extern "C" __global__ void einsum_f32(const unsigned long long* words) {
  const Step step = decode_step(words);
  for (unsigned long long out = blockIdx.x * blockDim.x + threadIdx.x;
       out < step.output_elements;
       out += (unsigned long long)gridDim.x * blockDim.x) {
    float sum = 0.0f;
    bool have_sum = false;
    for (unsigned long long reduction = 0;
         reduction < step.reduction_elements; ++reduction) {
      const void* first_pointer = (const void*)step.operand_ptrs[0];
      float product = load_f32(
          first_pointer, (int)step.operand_dtypes[0],
          operand_offset(step, 0, out, reduction));
      for (int operand = 1; operand < step.operands; ++operand) {
        const void* pointer =
            (const void*)step.operand_ptrs[operand];
        product *= load_f32(
            pointer, (int)step.operand_dtypes[operand],
            operand_offset(step, operand, out, reduction));
      }
      if (have_sum) sum += product;
      else {
        sum = product;
        have_sum = true;
      }
    }
    store_f32(
        step.output, step.output_dtype, output_offset(step, out), sum);
  }
}

extern "C" __global__ void einsum_f64(const unsigned long long* words) {
  const Step step = decode_step(words);
  for (unsigned long long out = blockIdx.x * blockDim.x + threadIdx.x;
       out < step.output_elements;
       out += (unsigned long long)gridDim.x * blockDim.x) {
    double sum = 0.0;
    bool have_sum = false;
    for (unsigned long long reduction = 0;
         reduction < step.reduction_elements; ++reduction) {
      const double* first_pointer =
          (const double*)step.operand_ptrs[0];
      double product =
          first_pointer[operand_offset(step, 0, out, reduction)];
      for (int operand = 1; operand < step.operands; ++operand) {
        const double* pointer =
            (const double*)step.operand_ptrs[operand];
        product *= pointer[operand_offset(step, operand, out, reduction)];
      }
      if (have_sum) sum += product;
      else {
        sum = product;
        have_sum = true;
      }
    }
    ((double*)step.output)[output_offset(step, out)] = sum;
  }
}

// Signed tensors use the same unsigned storage-width kernel: two's-complement
// multiplication/addition modulo 2^N is exactly unsigned low-N-bit arithmetic.
// WIDE is always unsigned, so every intermediate operation has defined modular
// semantics; the explicit BITS cast applies the declared-width reduction.
#define DEFINE_INTEGER_EINSUM(BITS, WIDE, NAME) \
extern "C" __global__ void einsum_##NAME(const unsigned long long* words) { \
  const Step step = decode_step(words); \
  for (unsigned long long out = blockIdx.x * blockDim.x + threadIdx.x; \
       out < step.output_elements; \
       out += (unsigned long long)gridDim.x * blockDim.x) { \
    BITS sum = (BITS)0; \
    bool have_sum = false; \
    for (unsigned long long reduction = 0; \
         reduction < step.reduction_elements; ++reduction) { \
      const BITS* first_pointer = \
          (const BITS*)step.operand_ptrs[0]; \
      BITS product = \
          first_pointer[operand_offset(step, 0, out, reduction)]; \
      for (int operand = 1; operand < step.operands; ++operand) { \
        const BITS* pointer = \
            (const BITS*)step.operand_ptrs[operand]; \
        const BITS value = \
            pointer[operand_offset(step, operand, out, reduction)]; \
        product = (BITS)((WIDE)product * (WIDE)value); \
      } \
      if (have_sum) sum = (BITS)((WIDE)sum + (WIDE)product); \
      else { \
        sum = product; \
        have_sum = true; \
      } \
    } \
    ((BITS*)step.output)[output_offset(step, out)] = sum; \
  } \
}

DEFINE_INTEGER_EINSUM(unsigned char, unsigned int, u8)
DEFINE_INTEGER_EINSUM(unsigned short, unsigned int, u16)
DEFINE_INTEGER_EINSUM(unsigned int, unsigned long long, u32)
DEFINE_INTEGER_EINSUM(unsigned long long, unsigned long long, u64)
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RequestedRoute {
    Auto,
    GenericNative,
    Optimized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaEinsumRoute {
    ViewAlias,
    ViewMaterialized,
    CudaCublas,
    GenericNative,
    OptimizedDp,
    OptimizedHeuristic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arithmetic {
    Float32,
    Float64,
    U8,
    U16,
    U32,
    U64,
}

impl Arithmetic {
    fn for_dtype(dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::Float16 | DataType::BFloat16 | DataType::Float32 => Ok(Self::Float32),
            DataType::Float64 => Ok(Self::Float64),
            DataType::Uint8 | DataType::Int8 => Ok(Self::U8),
            DataType::Uint16 | DataType::Int16 => Ok(Self::U16),
            DataType::Uint32 | DataType::Int32 => Ok(Self::U32),
            DataType::Uint64 | DataType::Int64 => Ok(Self::U64),
            other => Err(not_implemented(format!(
                "Einsum dtype {other:?}; ONNX Einsum admits homogeneous f16/f32/f64, \
                 u8/u16/u32/u64, i8/i16/i32/i64, plus bf16 in Einsum-28"
            ))),
        }
    }

    fn entry(self) -> &'static str {
        match self {
            Self::Float32 => "einsum_f32",
            Self::Float64 => "einsum_f64",
            Self::U8 => "einsum_u8",
            Self::U16 => "einsum_u16",
            Self::U32 => "einsum_u32",
            Self::U64 => "einsum_u64",
        }
    }

    fn module_key(self, dtype: DataType) -> &'static str {
        match (self, dtype) {
            (Self::Float32, DataType::Float16 | DataType::BFloat16) => "einsum_generic_half_v1",
            (Self::Float32, _) => "einsum_generic_f32_v1",
            (Self::Float64, _) => "einsum_generic_f64_v1",
            (Self::U8, _) => "einsum_generic_u8_v1",
            (Self::U16, _) => "einsum_generic_u16_v1",
            (Self::U32, _) => "einsum_generic_u32_v1",
            (Self::U64, _) => "einsum_generic_u64_v1",
        }
    }
}

fn dtype_code(dtype: DataType) -> Result<u64> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        DataType::Float64
        | DataType::Uint8
        | DataType::Int8
        | DataType::Uint16
        | DataType::Int16
        | DataType::Uint32
        | DataType::Int32
        | DataType::Uint64
        | DataType::Int64 => Ok(0),
        other => Err(not_implemented(format!("Einsum dtype {other:?}"))),
    }
}

#[derive(Clone)]
struct ValueLayout {
    pointer: CUdeviceptr,
    dtype: DataType,
    axes: Vec<EinsumAxis>,
    shape: Vec<usize>,
    strides: Vec<i64>,
}

enum PreparedLaunch {
    Generic {
        metadata_offset_bytes: usize,
        config: LaunchConfig,
    },
    Cublas {
        plan: CaptureStridedBatchedGemmPlan,
        params: StridedBatchedGemmParams,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CudaEinsumPlanSummary {
    pub route: CudaEinsumRoute,
    pub planner_quality: Option<EinsumPlannerQuality>,
    pub workspace_bytes: usize,
    pub metadata_bytes: usize,
    pub kernel_launches: usize,
    pub cublas_launches: usize,
}

/// Immutable, exact-signature CUDA execution snapshot.
///
/// All device pointers embedded in the launch metadata, all workspace storage,
/// the loaded kernel function, and launch geometry are installed atomically
/// after a successful warm execution. CUDA graph recording can therefore only
/// reuse this exact snapshot and retains every private allocation through
/// [`DeviceGraphResource`] owners.
pub(super) struct CudaEinsumPlan {
    signature: ArithmeticExecutionSignature,
    requested: RequestedRoute,
    memory_ceiling_bytes: u128,
    route: CudaEinsumRoute,
    planner_quality: Option<EinsumPlannerQuality>,
    function: Option<RawCudaFunction>,
    launches: Vec<PreparedLaunch>,
    metadata: Option<Arc<GraphDeviceAllocation>>,
    workspace: Option<Arc<GraphDeviceAllocation>>,
    cublas_workspace: Option<Arc<GraphDeviceAllocation>>,
    workspace_bytes: usize,
    metadata_bytes: usize,
}

impl CudaEinsumPlan {
    pub(super) fn build(
        semantic: &EinsumShapePlan,
        signature: ArithmeticExecutionSignature,
        runtime: &Arc<CudaRuntime>,
        requested: RequestedRoute,
        memory_ceiling_bytes: u128,
    ) -> Result<Self> {
        let dtype = signature.inputs[0].dtype;
        let arithmetic = Arithmetic::for_dtype(dtype)?;
        runtime.require_nvrtc_half_headers("Einsum")?;
        let precision =
            EinsumPrecisionPolicy::for_schema(semantic.schema(), dtype).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: dtype {dtype:?} is not admitted by {}",
                    semantic.equation(),
                    semantic.schema()
                ))
            })?;
        let dimensions = logical_dimensions(semantic)?;
        let selected = if requested != RequestedRoute::GenericNative {
            select_tree(
                semantic,
                precision.intermediate_element_size(),
                &signature.inputs,
                memory_ceiling_bytes,
            )?
        } else {
            None
        };

        let (route, planner_quality, workspace_layout, step_specs) =
            if let Some((quality, candidate)) = selected {
                let workspace_layout = workspace_layout(
                    semantic,
                    precision.intermediate_dtype(),
                    candidate,
                    &dimensions,
                    memory_ceiling_bytes,
                )?;
                let route = match quality {
                    EinsumPlannerQuality::ExactSubsetDp => CudaEinsumRoute::OptimizedDp,
                    EinsumPlannerQuality::DeterministicGreedy => {
                        CudaEinsumRoute::OptimizedHeuristic
                    }
                    EinsumPlannerQuality::GenericNativeFallback => {
                        unreachable!("generic fallback has no tree candidate")
                    }
                };
                let specs = tree_step_specs(
                    semantic,
                    precision.intermediate_dtype(),
                    &signature.inputs,
                    &signature.output,
                    candidate,
                    &dimensions,
                )?;
                (route, Some(quality), workspace_layout, specs)
            } else {
                if requested == RequestedRoute::Optimized {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Einsum `{}`: forced optimized route has no contraction-tree \
                         candidate within the {memory_ceiling_bytes}-byte workspace ceiling; \
                         force GenericNative or raise the explicit ceiling",
                        semantic.equation()
                    )));
                }
                (
                    CudaEinsumRoute::GenericNative,
                    None,
                    WorkspaceLayout::default(),
                    vec![generic_step_spec(
                        semantic,
                        dtype,
                        &signature.inputs,
                        &signature.output,
                        &dimensions,
                    )?],
                )
            };

        let workspace = if workspace_layout.bytes == 0 {
            None
        } else {
            let allocation = GraphDeviceAllocation::allocate(runtime, workspace_layout.bytes)?;
            runtime.staged_warm_cache_mutation("Einsum workspace allocation")?;
            Some(allocation)
        };
        let workspace_ptr = workspace.as_ref().map_or(0, |allocation| allocation.ptr());
        let mut planned_cublas = Vec::with_capacity(step_specs.len());
        let mut cublas_workspace_bytes = 0usize;
        for spec in &step_specs {
            let planned = if let Some(gemm) = spec.gemm.as_ref() {
                let params = StridedBatchedGemmParams {
                    dtype: GemmDtype::F32,
                    a: resolve_pointer(&gemm.left, workspace_ptr, &workspace_layout)?,
                    b: resolve_pointer(&gemm.right, workspace_ptr, &workspace_layout)?,
                    c: resolve_pointer(&gemm.output, workspace_ptr, &workspace_layout)?,
                    m: gemm.m,
                    k: gemm.k,
                    n: gemm.n,
                    batch: gemm.batch,
                    transpose_a: gemm.transpose_left,
                    transpose_b: gemm.transpose_right,
                    a_batch_stride: gemm.left_batch_stride,
                    b_batch_stride: gemm.right_batch_stride,
                };
                blas::plan_capture_strided_batched_gemm(runtime.blas(), &params)
                    .ok()
                    .map(|plan| (plan, params))
            } else {
                None
            };
            if let Some((plan, _)) = &planned {
                cublas_workspace_bytes = cublas_workspace_bytes.max(plan.workspace_bytes());
            }
            planned_cublas.push(planned);
        }
        if cublas_workspace_bytes > WORKSPACE_BYTES
            || (workspace_layout.bytes as u128).saturating_add(cublas_workspace_bytes as u128)
                > memory_ceiling_bytes
        {
            planned_cublas.iter_mut().for_each(|plan| *plan = None);
            cublas_workspace_bytes = 0;
        }
        let cublas_workspace = if cublas_workspace_bytes == 0 {
            None
        } else {
            let allocation = GraphDeviceAllocation::allocate(runtime, cublas_workspace_bytes)?;
            runtime.staged_warm_cache_mutation("Einsum cuBLASLt workspace allocation")?;
            Some(allocation)
        };
        runtime.staged_warm_cache_mutation("Einsum cuBLASLt candidate selection")?;
        let mut metadata_words = Vec::new();
        let mut launches = Vec::new();
        for (spec, cublas) in step_specs.into_iter().zip(planned_cublas) {
            if let (Some(_), Some((cublas, params))) = (spec.gemm.as_ref(), cublas) {
                launches.push(PreparedLaunch::Cublas {
                    plan: cublas,
                    params,
                });
            } else {
                let offset = metadata_words
                    .len()
                    .checked_mul(std::mem::size_of::<u64>())
                    .ok_or_else(|| {
                        EpError::KernelFailed(format!(
                            "cuda_ep Einsum `{}`: metadata byte offset overflowed",
                            semantic.equation()
                        ))
                    })?;
                encode_step(
                    semantic,
                    &spec,
                    workspace_ptr,
                    &workspace_layout,
                    &mut metadata_words,
                )?;
                launches.push(PreparedLaunch::Generic {
                    metadata_offset_bytes: offset,
                    config: LaunchConfig {
                        grid_dim: (
                            spec.output_elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
                            1,
                            1,
                        ),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    },
                });
            }
        }
        let metadata_bytes = metadata_words
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: metadata byte count overflowed",
                    semantic.equation()
                ))
            })?;
        let metadata = if metadata_bytes == 0 {
            None
        } else {
            let allocation = GraphDeviceAllocation::allocate(runtime, metadata_bytes)?;
            let bytes = metadata_words
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect::<Vec<_>>();
            // SAFETY: the fresh allocation exactly covers `bytes`.
            unsafe { runtime.htod(&bytes, allocation.ptr()) }?;
            runtime.staged_warm_cache_mutation("Einsum metadata allocation/upload")?;
            Some(allocation)
        };
        let function = if launches
            .iter()
            .any(|launch| matches!(launch, PreparedLaunch::Generic { .. }))
        {
            Some(runtime.nvrtc_raw_function(
                arithmetic.module_key(dtype),
                SOURCE,
                arithmetic.entry(),
            )?)
        } else {
            None
        };
        runtime.staged_warm_cache_mutation("Einsum algorithm/function selection")?;

        Ok(Self {
            signature,
            requested,
            memory_ceiling_bytes,
            route,
            planner_quality,
            function,
            launches,
            metadata,
            workspace,
            cublas_workspace,
            workspace_bytes: workspace_layout
                .bytes
                .checked_add(cublas_workspace_bytes)
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "cuda_ep Einsum `{}`: aggregate workspace byte count overflowed",
                        semantic.equation()
                    ))
                })?,
            metadata_bytes,
        })
    }

    pub(super) fn matches(
        &self,
        requested: RequestedRoute,
        memory_ceiling_bytes: u128,
        signature: &ArithmeticExecutionSignature,
    ) -> bool {
        self.requested == requested
            && self.memory_ceiling_bytes == memory_ceiling_bytes
            && self.signature == *signature
    }

    pub(super) fn mismatch_reason(
        &self,
        signature: &ArithmeticExecutionSignature,
    ) -> Option<String> {
        self.signature.mismatch_reason(signature)
    }

    pub(super) fn launch(
        &self,
        runtime: &CudaRuntime,
        expected_device: DeviceId,
        equation: &str,
        signature: &ArithmeticExecutionSignature,
    ) -> Result<()> {
        if let Some(reason) = self.signature.mismatch_reason(signature) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum: semantic execution signature changed after validation: {reason}"
            )));
        }
        signature.validate_device_ownership(expected_device, equation)?;
        signature.validate_alias_proof(equation)?;
        for launch in &self.launches {
            match launch {
                PreparedLaunch::Generic {
                    metadata_offset_bytes,
                    config,
                } => {
                    let function = self.function.as_ref().ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep Einsum: generic launch lost its prepared function".into(),
                        )
                    })?;
                    let metadata = self.metadata.as_ref().ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep Einsum: generic launch lost its immutable metadata owner"
                                .into(),
                        )
                    })?;
                    let pointer = metadata
                        .ptr()
                        .checked_add(u64::try_from(*metadata_offset_bytes).map_err(|_| {
                            EpError::KernelFailed(
                                "cuda_ep Einsum: metadata byte offset does not fit u64".into(),
                            )
                        })?)
                        .ok_or_else(|| {
                            EpError::KernelFailed(
                                "cuda_ep Einsum: metadata device-pointer offset overflowed".into(),
                            )
                        })?;
                    let mut pointer_arg = pointer;
                    let mut params = [(&mut pointer_arg as *mut CUdeviceptr).cast::<c_void>()];
                    // SAFETY: the immutable metadata owner and every
                    // exact-signature pointer it names remain live for the
                    // stream operation. The one-argument ABI matches the entry.
                    unsafe { function.launch(runtime.stream(), *config, &mut params) }
                        .map_err(|error| driver_err("launch generic CUDA Einsum", error))?;
                }
                PreparedLaunch::Cublas { plan, params } => {
                    // SAFETY: the immutable plan was selected for these exact
                    // pointers/layouts, and the shared workspace owner remains
                    // live for every sequential step.
                    unsafe {
                        plan.launch(
                            runtime.blas(),
                            runtime.stream_ptr(),
                            params,
                            self.cublas_workspace
                                .as_ref()
                                .map_or(0, |workspace| workspace.ptr()),
                        )
                    }?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn resources(&self) -> Vec<DeviceGraphResource> {
        let mut resources = Vec::with_capacity(2);
        if let Some(metadata) = &self.metadata {
            resources.push(GraphDeviceAllocation::device_graph_resource(metadata));
        }
        if let Some(workspace) = &self.workspace {
            resources.push(GraphDeviceAllocation::device_graph_resource(workspace));
        }
        if let Some(workspace) = &self.cublas_workspace {
            resources.push(GraphDeviceAllocation::device_graph_resource(workspace));
        }
        resources
    }

    pub(super) fn require_capture_resources(&self, runtime: &CudaRuntime) -> Result<()> {
        for resource in self.resources() {
            runtime.require_registered_address_capture(
                resource.identity(),
                "immutable CUDA Einsum plan resource",
            )?;
        }
        Ok(())
    }

    pub(super) fn summary(&self) -> CudaEinsumPlanSummary {
        CudaEinsumPlanSummary {
            route: self.route,
            planner_quality: self.planner_quality,
            workspace_bytes: self.workspace_bytes,
            metadata_bytes: self.metadata_bytes,
            kernel_launches: self.launches.len(),
            cublas_launches: self
                .launches
                .iter()
                .filter(|launch| matches!(launch, PreparedLaunch::Cublas { .. }))
                .count(),
        }
    }

    pub(super) fn workspace_ptr(&self) -> CUdeviceptr {
        self.workspace
            .as_ref()
            .or(self.cublas_workspace.as_ref())
            .map_or(0, |allocation| allocation.ptr())
    }
}

#[derive(Default)]
struct WorkspaceLayout {
    bytes: usize,
    values: BTreeMap<usize, usize>,
}

impl WorkspaceLayout {
    fn value_pointer(&self, base: CUdeviceptr, value: EinsumValueId) -> Result<CUdeviceptr> {
        let offset = *self.values.get(&value.index()).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum: contraction value {value} has no workspace slot"
            ))
        })?;
        base.checked_add(u64::try_from(offset).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum: workspace offset for contraction value {value} does not fit u64"
            ))
        })?)
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum: workspace pointer overflow for contraction value {value}"
            ))
        })
    }
}

fn align_up(value: usize, alignment: usize, equation: &str) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{equation}`: aligned workspace size overflowed"
            ))
        })
}

fn logical_dimensions(plan: &EinsumShapePlan) -> Result<BTreeMap<EinsumAxis, usize>> {
    plan.logical_axes()
        .iter()
        .map(|axis| {
            axis.dimension().as_static().map(|dimension| (axis.axis(), dimension)).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: shape-specialized plan retained a dynamic extent for {}",
                    plan.equation(),
                    axis.axis()
                ))
            })
        })
        .collect()
}

fn axes_shape(
    plan: &EinsumShapePlan,
    axes: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, usize>,
) -> Result<Vec<usize>> {
    axes.iter()
        .map(|axis| {
            dimensions.get(axis).copied().ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: execution plan lost the extent for {axis}",
                    plan.equation()
                ))
            })
        })
        .collect()
}

fn checked_product(values: &[usize], equation: &str, target: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{equation}`: {target} element count overflowed usize"
            ))
        })
    })
}

fn dense_layout(
    pointer: CUdeviceptr,
    dtype: DataType,
    axes: Vec<EinsumAxis>,
    shape: Vec<usize>,
) -> ValueLayout {
    ValueLayout {
        pointer,
        dtype,
        strides: compute_contiguous_strides(&shape),
        axes,
        shape,
    }
}

fn leaf_layout(
    input: &ArithmeticTensorSignature,
    operand: &EinsumOperandPlan,
) -> Result<ValueLayout> {
    let mut axes = Vec::with_capacity(operand.unique_axes().len());
    let mut shape = Vec::with_capacity(operand.unique_axes().len());
    let mut strides = Vec::with_capacity(operand.unique_axes().len());
    for axis in operand.unique_axes() {
        let &physical = axis.input_axes().first().ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum: operand #{} has an empty canonical axis",
                operand.input()
            ))
        })?;
        axes.push(axis.axis());
        shape.push(input.shape[physical]);
        let stride = axis.input_axes().iter().try_fold(0i64, |sum, &physical| {
            sum.checked_add(input.strides[physical]).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum: diagonal stride overflow for operand #{} axis {physical}",
                    operand.input()
                ))
            })
        })?;
        strides.push(stride);
    }
    Ok(ValueLayout {
        pointer: input.effective_pointer,
        dtype: input.dtype,
        axes,
        shape,
        strides,
    })
}

fn workspace_layout(
    plan: &EinsumShapePlan,
    intermediate_dtype: DataType,
    candidate: &EinsumSupportedContractionTreeCandidate,
    dimensions: &BTreeMap<EinsumAxis, usize>,
    ceiling: u128,
) -> Result<WorkspaceLayout> {
    let element_bytes = intermediate_dtype.byte_size();
    let mut slot_bytes = BTreeMap::<usize, usize>::new();
    let mut value_slots = BTreeMap::<usize, usize>::new();
    for temporary in candidate.temporaries() {
        let shape = axes_shape(plan, temporary.axes(), dimensions)?;
        let elements = checked_product(&shape, plan.equation(), "temporary")?;
        let bytes = elements.checked_mul(element_bytes).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: temporary byte count overflowed usize",
                plan.equation()
            ))
        })?;
        slot_bytes
            .entry(temporary.slot())
            .and_modify(|maximum| *maximum = (*maximum).max(bytes))
            .or_insert(bytes);
        value_slots.insert(temporary.value().index(), temporary.slot());
    }
    let mut slots = BTreeMap::new();
    let mut offset = 0usize;
    for (slot, bytes) in slot_bytes {
        offset = align_up(offset, WORKSPACE_ALIGNMENT, plan.equation())?;
        slots.insert(slot, (offset, bytes));
        offset = offset.checked_add(bytes).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: workspace layout overflowed usize",
                plan.equation()
            ))
        })?;
    }
    let bytes = align_up(offset, WORKSPACE_ALIGNMENT, plan.equation())?;
    if bytes as u128 > ceiling {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Einsum `{}`: optimized workspace layout needs {bytes} bytes, exceeding the \
             {ceiling}-byte ceiling",
            plan.equation()
        )));
    }
    let values = value_slots
        .into_iter()
        .map(|(value, slot)| {
            let (offset, _) = slots[&slot];
            (value, offset)
        })
        .collect();
    Ok(WorkspaceLayout { bytes, values })
}

fn select_tree<'a>(
    plan: &'a EinsumShapePlan,
    intermediate_element_size: usize,
    inputs: &[ArithmeticTensorSignature],
    ceiling: u128,
) -> Result<
    Option<(
        EinsumPlannerQuality,
        &'a EinsumSupportedContractionTreeCandidate,
    )>,
> {
    let Some(tree) = plan.semantic_plan().contraction_tree() else {
        return Ok(None);
    };
    if tree.quality() == EinsumPlannerQuality::GenericNativeFallback {
        return Ok(None);
    }
    let shapes = inputs
        .iter()
        .map(|input| input.shape.as_slice())
        .collect::<Vec<_>>();
    let concrete = plan
        .resolve_concrete_contraction_tree(&shapes, intermediate_element_size)
        .map_err(|error| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: concrete contraction planning failed: {error}",
                plan.equation()
            ))
        })?
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: semantic contraction tree disappeared during concrete planning",
                plan.equation()
            ))
        })?;
    let Some(selected) = concrete.preferred_candidate_with_memory_ceiling(ceiling) else {
        return Ok(None);
    };
    let candidate = tree
        .candidates()
        .iter()
        .find(|candidate| candidate.id() == selected.id())
        .and_then(|candidate| candidate.supported())
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: concrete planner selected missing candidate {}",
                plan.equation(),
                selected.id()
            ))
        })?;
    Ok(Some((tree.quality(), candidate)))
}

struct OperandSpec {
    pointer: PointerSpec,
    dtype: DataType,
    axes: Vec<EinsumAxis>,
    shape: Vec<usize>,
    strides: Vec<i64>,
}

#[derive(Clone, Copy)]
enum PointerSpec {
    Absolute(CUdeviceptr),
    WorkspaceValue(EinsumValueId),
}

#[derive(Clone)]
struct GemmSpec {
    left: PointerSpec,
    right: PointerSpec,
    output: PointerSpec,
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    transpose_left: bool,
    transpose_right: bool,
    left_batch_stride: usize,
    right_batch_stride: usize,
}

struct StepSpec {
    iteration_axes: Vec<EinsumAxis>,
    output_rank: usize,
    output_elements: u64,
    reduction_elements: u64,
    output_pointer: PointerSpec,
    output_dtype: DataType,
    output_strides: Vec<i64>,
    operands: Vec<OperandSpec>,
    gemm: Option<GemmSpec>,
}

fn generic_step_spec(
    plan: &EinsumShapePlan,
    dtype: DataType,
    inputs: &[ArithmeticTensorSignature],
    output: &ArithmeticTensorSignature,
    dimensions: &BTreeMap<EinsumAxis, usize>,
) -> Result<StepSpec> {
    let program = plan.generic_native().index_program();
    let iteration_axes = program.iteration_axes().to_vec();
    let iteration_shape = axes_shape(plan, &iteration_axes, dimensions)?;
    let output_rank = program.output_rank();
    let output_elements =
        checked_product(&iteration_shape[..output_rank], plan.equation(), "output")?;
    let reduction_elements = checked_product(
        &iteration_shape[output_rank..],
        plan.equation(),
        "reduction",
    )?;
    let operands = inputs
        .iter()
        .zip(program.operands())
        .map(|(input, index)| {
            if input.shape.len() != index.physical_axis_to_iteration_axis().len() {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: input #{} rank {} differs from index-program rank {}",
                    plan.equation(),
                    index.input(),
                    input.shape.len(),
                    index.physical_axis_to_iteration_axis().len()
                )));
            }
            let axes = index
                .physical_axis_to_iteration_axis()
                .iter()
                .map(|&axis| iteration_axes[axis])
                .collect();
            Ok(OperandSpec {
                pointer: PointerSpec::Absolute(input.effective_pointer),
                dtype: input.dtype,
                axes,
                shape: input.shape.to_vec(),
                strides: input.strides.to_vec(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StepSpec {
        iteration_axes,
        output_rank,
        output_elements: u64::try_from(output_elements).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: output element count does not fit u64",
                plan.equation()
            ))
        })?,
        reduction_elements: u64::try_from(reduction_elements).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: reduction element count does not fit u64",
                plan.equation()
            ))
        })?,
        output_pointer: PointerSpec::Absolute(output.effective_pointer),
        output_dtype: dtype,
        output_strides: output.strides.to_vec(),
        operands,
        gemm: None,
    })
}

fn tree_step_specs(
    plan: &EinsumShapePlan,
    intermediate_dtype: DataType,
    inputs: &[ArithmeticTensorSignature],
    output: &ArithmeticTensorSignature,
    candidate: &EinsumSupportedContractionTreeCandidate,
    dimensions: &BTreeMap<EinsumAxis, usize>,
) -> Result<Vec<StepSpec>> {
    let dtype = inputs[0].dtype;
    let mut values = BTreeMap::<usize, ValueLayout>::new();
    for (input, operand) in inputs.iter().zip(plan.operands()) {
        values.insert(operand.input(), leaf_layout(input, operand)?);
    }
    for temporary in candidate.temporaries() {
        let shape = axes_shape(plan, temporary.axes(), dimensions)?;
        values.insert(
            temporary.value().index(),
            dense_layout(0, intermediate_dtype, temporary.axes().to_vec(), shape),
        );
    }
    let final_layout = ValueLayout {
        pointer: output.effective_pointer,
        dtype,
        axes: plan.output_axes().to_vec(),
        shape: output.shape.to_vec(),
        strides: output.strides.to_vec(),
    };
    let mut specs = Vec::with_capacity(candidate.steps().len());
    for step in candidate.steps() {
        let (input_ids, reduction_axes, produced, binary) = match step {
            EinsumContractionTreeStep::UnaryReduction(unary) => (
                vec![unary.input()],
                unary.reduction_axes().to_vec(),
                unary.output(),
                None,
            ),
            EinsumContractionTreeStep::BinaryContraction(binary) => (
                vec![binary.left(), binary.right()],
                binary.contract_axes().to_vec(),
                binary.output(),
                Some(binary.as_ref()),
            ),
            _ => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: contraction tree contains a newer unrecognized step",
                    plan.equation()
                )));
            }
        };
        let output_layout = if produced == candidate.final_output() {
            final_layout.clone()
        } else {
            values.get(&produced.index()).cloned().ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: contraction value {produced} has no temporary layout",
                    plan.equation()
                ))
            })?
        };
        let mut iteration_axes = output_layout.axes.clone();
        iteration_axes.extend(reduction_axes);
        let iteration_shape = axes_shape(plan, &iteration_axes, dimensions)?;
        let output_rank = output_layout.axes.len();
        let output_elements = checked_product(
            &iteration_shape[..output_rank],
            plan.equation(),
            "tree-step output",
        )?;
        let reduction_elements = checked_product(
            &iteration_shape[output_rank..],
            plan.equation(),
            "tree-step reduction",
        )?;
        let operands = input_ids
            .into_iter()
            .map(|value| {
                let layout = values.get(&value.index()).ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "cuda_ep Einsum `{}`: contraction step references unavailable value {value}",
                        plan.equation()
                    ))
                })?;
                Ok(OperandSpec {
                    pointer: if value.index() < inputs.len() {
                        PointerSpec::Absolute(layout.pointer)
                    } else {
                        PointerSpec::WorkspaceValue(value)
                    },
                    dtype: layout.dtype,
                    axes: layout.axes.clone(),
                    shape: layout.shape.clone(),
                    strides: layout.strides.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let gemm = binary.and_then(|binary| {
            gemm_step_spec(
                dtype,
                binary,
                &operands,
                &output_layout,
                &iteration_shape,
                output_rank,
            )
        });
        specs.push(StepSpec {
            iteration_axes,
            output_rank,
            output_elements: u64::try_from(output_elements).map_err(|_| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: tree-step output element count does not fit u64",
                    plan.equation()
                ))
            })?,
            reduction_elements: u64::try_from(reduction_elements).map_err(|_| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: tree-step reduction element count does not fit u64",
                    plan.equation()
                ))
            })?,
            output_pointer: if produced == candidate.final_output() {
                PointerSpec::Absolute(final_layout.pointer)
            } else {
                PointerSpec::WorkspaceValue(produced)
            },
            output_dtype: output_layout.dtype,
            output_strides: output_layout.strides,
            operands,
            gemm,
        });
    }
    Ok(specs)
}

fn ordered_storage(
    operand: &OperandSpec,
    order: &[Option<usize>],
    batch_rank: usize,
    first_group_rank: usize,
) -> Option<bool> {
    if !onnx_runtime_ir::is_contiguous(&operand.shape, &operand.strides)
        || order.len() < batch_rank + first_group_rank
    {
        return None;
    }
    let (batch, matrix) = order.split_at(batch_rank);
    let (first, second) = matrix.split_at(first_group_rank);
    if matrix.iter().any(Option::is_none) {
        return None;
    }
    let expected = (0..operand.axes.len()).collect::<Vec<_>>();
    let flatten = |parts: &[&[Option<usize>]]| {
        parts
            .iter()
            .flat_map(|part| part.iter().copied().flatten())
            .collect::<Vec<_>>()
    };
    if flatten(&[batch, first, second]) == expected {
        Some(false)
    } else if flatten(&[batch, second, first]) == expected {
        Some(true)
    } else {
        None
    }
}

fn batch_stride(
    operand: &OperandSpec,
    order: &[Option<usize>],
    batch_shape: &[usize],
    matrix_elements: usize,
) -> Option<usize> {
    let operand_batch = order
        .iter()
        .take(batch_shape.len())
        .map(|axis| axis.map_or(1, |axis| operand.shape[axis]))
        .collect::<Vec<_>>();
    if operand_batch
        .iter()
        .zip(batch_shape)
        .any(|(&operand, &output)| operand != 1 && operand != output)
    {
        return None;
    }
    if batch_shape.iter().all(|&dim| dim == 1) || operand_batch.iter().all(|&dim| dim == 1) {
        Some(0)
    } else if operand_batch == batch_shape {
        Some(matrix_elements)
    } else {
        None
    }
}

fn product_or_none(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .try_fold(1usize, |product, &value| product.checked_mul(value))
}

fn gemm_step_spec(
    dtype: DataType,
    binary: &EinsumBinaryContractionPlan,
    operands: &[OperandSpec],
    output: &ValueLayout,
    iteration_shape: &[usize],
    output_rank: usize,
) -> Option<GemmSpec> {
    if dtype != DataType::Float32
        || operands.len() != 2
        || operands
            .iter()
            .any(|operand| operand.dtype != DataType::Float32)
        || output.dtype != DataType::Float32
        || output.axes != binary.canonical_output_axes()
        || !onnx_runtime_ir::is_contiguous(&output.shape, &output.strides)
        || binary
            .output_permutation()
            .iter()
            .copied()
            .ne(0..binary.output_permutation().len())
    {
        return None;
    }
    let batch_rank = binary.batch_axes().len();
    let left_rank = binary.left_free_axes().len();
    let right_rank = binary.right_free_axes().len();
    if output_rank != batch_rank + left_rank + right_rank {
        return None;
    }
    let transpose_left = ordered_storage(
        &operands[0],
        binary.left_axis_order(),
        batch_rank,
        left_rank,
    )?;
    let transpose_right = ordered_storage(
        &operands[1],
        binary.right_axis_order(),
        batch_rank,
        binary.contract_axes().len(),
    )?;
    let batch = product_or_none(&iteration_shape[..batch_rank])?;
    let m = product_or_none(&iteration_shape[batch_rank..batch_rank + left_rank])?;
    let n = product_or_none(&iteration_shape[batch_rank + left_rank..output_rank])?;
    let k = product_or_none(&iteration_shape[output_rank..])?;
    if output_rank == 0 && binary.canonical_output_axes().is_empty() {
        // A dot product is represented as 1xK by Kx1.
    }
    if k == 0 || product_or_none(&output.shape)? == 0 {
        return None;
    }
    let left_batch_stride = batch_stride(
        &operands[0],
        binary.left_axis_order(),
        &iteration_shape[..batch_rank],
        m.checked_mul(k)?,
    )?;
    let right_batch_stride = batch_stride(
        &operands[1],
        binary.right_axis_order(),
        &iteration_shape[..batch_rank],
        k.checked_mul(n)?,
    )?;
    Some(GemmSpec {
        left: operands[0].pointer,
        right: operands[1].pointer,
        output: if output.pointer == 0 {
            PointerSpec::WorkspaceValue(binary.output())
        } else {
            PointerSpec::Absolute(output.pointer)
        },
        m,
        k,
        n,
        batch,
        transpose_left,
        transpose_right,
        left_batch_stride,
        right_batch_stride,
    })
}

fn resolve_pointer(
    pointer: &PointerSpec,
    workspace_ptr: CUdeviceptr,
    workspace: &WorkspaceLayout,
) -> Result<CUdeviceptr> {
    match pointer {
        PointerSpec::Absolute(pointer) => Ok(*pointer),
        PointerSpec::WorkspaceValue(value) => workspace.value_pointer(workspace_ptr, *value),
    }
}

fn encode_step(
    plan: &EinsumShapePlan,
    step: &StepSpec,
    workspace_ptr: CUdeviceptr,
    workspace: &WorkspaceLayout,
    words: &mut Vec<u64>,
) -> Result<()> {
    i32::try_from(step.iteration_axes.len()).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep Einsum `{}`: iteration rank {} exceeds the CUDA kernel's i32 ABI",
            plan.equation(),
            step.iteration_axes.len()
        ))
    })?;
    i32::try_from(step.output_rank).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep Einsum `{}`: output rank {} exceeds the CUDA kernel's i32 ABI",
            plan.equation(),
            step.output_rank
        ))
    })?;
    i32::try_from(step.operands.len()).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep Einsum `{}`: operand count {} exceeds the CUDA kernel's i32 ABI",
            plan.equation(),
            step.operands.len()
        ))
    })?;
    let iteration_shape = axes_shape(plan, &step.iteration_axes, &logical_dimensions(plan)?)?;
    let mut divisors = vec![1usize; iteration_shape.len()];
    let output_rank = step.output_rank;
    let mut divisor = 1usize;
    for axis in (0..output_rank).rev() {
        divisors[axis] = divisor;
        divisor = divisor.checked_mul(iteration_shape[axis]).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: output coordinate divisor overflowed",
                plan.equation()
            ))
        })?;
    }
    divisor = 1;
    for axis in (output_rank..iteration_shape.len()).rev() {
        divisors[axis] = divisor;
        divisor = divisor.checked_mul(iteration_shape[axis]).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: reduction coordinate divisor overflowed",
                plan.equation()
            ))
        })?;
    }
    let mut record_offsets = Vec::with_capacity(step.operands.len() + 1);
    let mut records = Vec::<u64>::new();
    for operand in &step.operands {
        record_offsets.push(records.len() / 3);
        if operand.axes.len() != operand.shape.len() || operand.axes.len() != operand.strides.len()
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: operand metadata rank mismatch",
                plan.equation()
            )));
        }
        for ((axis, &extent), &stride) in operand
            .axes
            .iter()
            .zip(&operand.shape)
            .zip(&operand.strides)
        {
            let iteration_axis = step
                .iteration_axes
                .iter()
                .position(|candidate| candidate == axis)
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "cuda_ep Einsum `{}`: step operand axis {axis} is absent from its iteration program",
                        plan.equation()
                    ))
                })?;
            let iteration_extent = iteration_shape[iteration_axis];
            if extent != iteration_extent && extent != 1 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: operand extent {extent} for {axis} does not match \
                     iteration extent {iteration_extent} and is not broadcastable",
                    plan.equation()
                )));
            }
            records.extend([
                u64::try_from(extent).map_err(|_| {
                    EpError::KernelFailed(format!(
                        "cuda_ep Einsum `{}`: operand extent does not fit u64",
                        plan.equation()
                    ))
                })?,
                u64::from_ne_bytes(stride.to_ne_bytes()),
                u64::try_from(iteration_axis).map_err(|_| {
                    EpError::KernelFailed(format!(
                        "cuda_ep Einsum `{}`: iteration-axis index does not fit u64",
                        plan.equation()
                    ))
                })?,
            ]);
        }
    }
    record_offsets.push(records.len() / 3);

    words.extend([
        step.output_elements,
        step.reduction_elements,
        u64::try_from(step.iteration_axes.len()).map_err(|_| {
            EpError::KernelFailed("cuda_ep Einsum: iteration rank does not fit u64".into())
        })?,
        u64::try_from(step.output_rank).map_err(|_| {
            EpError::KernelFailed("cuda_ep Einsum: output rank does not fit u64".into())
        })?,
        u64::try_from(step.operands.len()).map_err(|_| {
            EpError::KernelFailed("cuda_ep Einsum: operand count does not fit u64".into())
        })?,
        dtype_code(step.output_dtype)?,
        resolve_pointer(&step.output_pointer, workspace_ptr, workspace)?,
    ]);
    for &value in &iteration_shape {
        words.push(u64::try_from(value).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: iteration extent does not fit u64",
                plan.equation()
            ))
        })?);
    }
    for &value in &divisors {
        words.push(u64::try_from(value).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: coordinate divisor does not fit u64",
                plan.equation()
            ))
        })?);
    }
    words.extend(
        step.output_strides
            .iter()
            .map(|&value| u64::from_ne_bytes(value.to_ne_bytes())),
    );
    for &value in &record_offsets {
        words.push(u64::try_from(value).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: operand record offset does not fit u64",
                plan.equation()
            ))
        })?);
    }
    for operand in &step.operands {
        words.push(resolve_pointer(&operand.pointer, workspace_ptr, workspace)?);
    }
    for operand in &step.operands {
        words.push(dtype_code(operand.dtype)?);
    }
    words.extend(records);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{EinsumInput, EinsumPlan};

    #[test]
    fn generic_metadata_uses_the_canonical_physical_index_program() {
        let inputs = [
            EinsumInput::new(DataType::Float32, &[2, 3, 3][..]),
            EinsumInput::new(DataType::Float32, &[1, 3][..]),
        ];
        let plan = EinsumPlan::build("...ii,...i->...", &inputs).unwrap();
        let program = plan.generic_native().index_program();
        assert_eq!(program.output_rank(), 1);
        assert_eq!(
            program.operands()[0].physical_axis_to_iteration_axis(),
            &[0, 1, 1]
        );
        assert_eq!(
            program.operands()[1].physical_axis_to_iteration_axis(),
            &[0, 1]
        );
    }

    #[test]
    fn workspace_alignment_is_checked() {
        assert_eq!(align_up(257, WORKSPACE_ALIGNMENT, "i->i").unwrap(), 512);
    }
}

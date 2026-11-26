use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::LaunchConfig;
use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{
    DeviceGraphResource, DevicePtrMut, EpError, Result, WorkspaceLifetime, WorkspaceRequirement,
    WorkspaceView,
};
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
use crate::runtime::{CudaRuntime, NativeDeviceAllocation, RawCudaFunction};

const BLOCK: u32 = 256;
const WORKSPACE_ALIGNMENT: usize = blas::WORKSPACE_ALIGNMENT;
// ActualAddress heuristics observe scratch alignment only up to this cap.
// A relocated prefix and every fixed slot therefore preserve the same proof.
const _: () = assert!(blas::MAX_REPORTED_POINTER_ALIGNMENT.is_power_of_two());
const _: () = assert!(WORKSPACE_ALIGNMENT.is_power_of_two());
const _: () =
    assert!(WORKSPACE_ALIGNMENT.is_multiple_of(blas::MAX_REPORTED_POINTER_ALIGNMENT as usize));
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

/// One host algorithm program plus its last successfully executed binding.
/// Scratch belongs exclusively to the caller; only immutable address-bearing
/// metadata has a private device owner, independently pinned by installed graphs.
pub(super) struct CudaEinsumPlan {
    program: CudaEinsumProgram,
    binding: Option<BoundExecution>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BindingKey {
    base: CUdeviceptr,
    used_bytes: usize,
}

struct BoundExecution {
    key: BindingKey,
    function: Option<RawCudaFunction>,
    launches: Vec<PreparedLaunch>,
    metadata: Option<Arc<NativeDeviceAllocation>>,
    metadata_bytes: usize,
}

/// Host-only, actual-address-selected algorithms. This type owns no scratch or
/// immutable device metadata and does not need Sync or another lock/cache.
pub(super) struct CudaEinsumProgram {
    recipe: CudaEinsumRecipe,
    cublas: Vec<Option<CaptureStridedBatchedGemmPlan>>,
    workspace: SemanticWorkspace,
}

#[derive(Clone, Copy)]
struct SemanticWorkspace {
    layout: onnx_runtime_ep_api::WorkspaceLayout<2>,
}
struct BoundWorkspace {
    key: BindingKey,
    arithmetic: CUdeviceptr,
    cublas: CUdeviceptr,
    used: WorkspaceView,
}
impl SemanticWorkspace {
    fn new(arithmetic_bytes: usize, cublas_bytes: usize) -> Result<Self> {
        let region = |bytes| {
            std::alloc::Layout::from_size_align(bytes, if bytes == 0 { 1 } else { WORKSPACE_ALIGNMENT })
                .map_err(|_| EpError::KernelFailed(format!("cuda_ep Einsum: workspace region {bytes} bytes exceeds addressable capacity; reduce the contraction")))
        };
        Ok(Self {
            layout: onnx_runtime_ep_api::WorkspaceLayout::new([
                region(arithmetic_bytes)?,
                region(cublas_bytes)?,
            ])?,
        })
    }
    fn requirement(&self) -> WorkspaceRequirement {
        WorkspaceRequirement {
            bytes: self.layout.bytes() as u64,
            alignment: WORKSPACE_ALIGNMENT,
            lifetime: WorkspaceLifetime::SessionPersistent,
        }
    }
    fn bind(&self, offered: Option<WorkspaceView>) -> Result<BoundWorkspace> {
        let [arithmetic, cublas] =
            self.layout
                .bind(offered)
                .map_err(|source| EpError::Backend {
                    operation: "bind Einsum semantic caller workspace",
                    source: Box::new(source),
                })?;
        let used_bytes = self.layout.bytes();
        let base = if used_bytes == 0 {
            0
        } else {
            offered.expect("nonzero binding was checked above").ptr().0 as CUdeviceptr
        };
        Ok(BoundWorkspace {
            key: BindingKey { base, used_bytes },
            arithmetic: if arithmetic.bytes() == 0 {
                0
            } else {
                arithmetic.ptr().0 as CUdeviceptr
            },
            cublas: if cublas.bytes() == 0 {
                0
            } else {
                cublas.ptr().0 as CUdeviceptr
            },
            used: WorkspaceView::new(DevicePtrMut(base as *mut c_void), used_bytes),
        })
    }
}

pub(super) struct CudaEinsumRecipe {
    semantic: Arc<EinsumShapePlan>,
    signature: ArithmeticExecutionSignature,
    requested: RequestedRoute,
    memory_ceiling_bytes: u128,
    dtype: DataType,
    arithmetic: Arithmetic,
    route: CudaEinsumRoute,
    planner_quality: Option<EinsumPlannerQuality>,
    workspace_layout: WorkspaceLayout,
    step_specs: Vec<StepSpec>,
}

impl CudaEinsumRecipe {
    pub(super) fn build(
        semantic_owner: Arc<EinsumShapePlan>,
        signature: ArithmeticExecutionSignature,
        requested: RequestedRoute,
        memory_ceiling_bytes: u128,
    ) -> Result<Self> {
        let semantic = semantic_owner.as_ref();
        let dtype = signature.inputs[0].dtype;
        let arithmetic = Arithmetic::for_dtype(dtype)?;
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

        let (route, planner_quality, mut workspace_layout, mut step_specs) =
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

        // ONNX empty outputs have no arithmetic consumers. Preserve route
        // eligibility/ceiling validation above, but publish no scratch or
        // metadata/device launches for an accepted empty result.
        if signature.output.shape.contains(&0) {
            workspace_layout = WorkspaceLayout::default();
            step_specs.clear();
        }
        Ok(Self {
            semantic: semantic_owner,
            signature,
            requested,
            memory_ceiling_bytes,
            dtype,
            arithmetic,
            route,
            planner_quality,
            workspace_layout,
            step_specs,
        })
    }

    pub(super) fn matches(&self, signature: &ArithmeticExecutionSignature) -> bool {
        self.signature == *signature
    }

    pub(super) fn needs_binding(&self) -> bool {
        self.step_specs.iter().any(|step| step.gemm.is_some())
    }

    pub(super) fn prefix_requirement(&self) -> WorkspaceRequirement {
        WorkspaceRequirement {
            bytes: self.workspace_layout.bytes as u64,
            alignment: WORKSPACE_ALIGNMENT,
            lifetime: WorkspaceLifetime::SessionPersistent,
        }
    }

    /// The supplied prefix is real admitted storage. Only host descriptors are
    /// created here; no allocation/upload or device submission is permitted.
    pub(super) fn prepare_algorithms(
        self,
        runtime: &Arc<CudaRuntime>,
        prefix: Option<WorkspaceView>,
    ) -> Result<CudaEinsumProgram> {
        let workspace_ptr = if self.needs_binding() {
            SemanticWorkspace::new(self.workspace_layout.bytes, 0)?
                .bind(prefix)?
                .arithmetic
        } else {
            0
        };
        let mut cublas = Vec::with_capacity(self.step_specs.len());
        let mut cublas_bytes = 0;
        for spec in &self.step_specs {
            let plan = if let Some(gemm) = &spec.gemm {
                let params = gemm.params(workspace_ptr, &self.workspace_layout)?;
                // Preserve optional BLAS selection. Binding/layout errors above
                // are never converted into an algorithm fallback.
                blas::plan_capture_strided_batched_gemm(runtime.blas(), &params).ok()
            } else {
                None
            };
            if let Some(plan) = &plan {
                cublas_bytes = cublas_bytes.max(plan.workspace_bytes());
            }
            cublas.push(plan);
        }
        let workspace = if cublas_bytes > WORKSPACE_BYTES {
            cublas.iter_mut().for_each(|plan| *plan = None);
            SemanticWorkspace::new(self.workspace_layout.bytes, 0)?
        } else {
            let candidate = SemanticWorkspace::new(self.workspace_layout.bytes, cublas_bytes)?;
            if candidate.layout.bytes() as u128 > self.memory_ceiling_bytes {
                cublas.iter_mut().for_each(|plan| *plan = None);
                SemanticWorkspace::new(self.workspace_layout.bytes, 0)?
            } else {
                candidate
            }
        };
        runtime.staged_warm_cache_mutation("Einsum cuBLASLt candidate selection")?;
        Ok(CudaEinsumProgram {
            recipe: self,
            cublas,
            workspace,
        })
    }
}

impl CudaEinsumProgram {
    pub(super) fn matches(&self, signature: &ArithmeticExecutionSignature) -> bool {
        self.recipe.matches(signature)
    }
    pub(super) fn requirement(&self) -> WorkspaceRequirement {
        self.workspace.requirement()
    }
    pub(super) fn into_plan(self) -> CudaEinsumPlan {
        CudaEinsumPlan {
            program: self,
            binding: None,
        }
    }

    fn encode_binding(
        &self,
        workspace: &BoundWorkspace,
    ) -> Result<(Vec<PreparedLaunch>, Vec<u64>)> {
        let recipe = &self.recipe;
        let semantic = recipe.semantic.as_ref();
        let mut metadata_words = Vec::new();
        let mut launches = Vec::with_capacity(recipe.step_specs.len());
        // Validate every final-address BLAS proof and encode the entire host
        // payload before the first device allocation, upload or launch.
        for (spec, cublas) in recipe.step_specs.iter().zip(&self.cublas) {
            if let Some(cublas) = cublas {
                let gemm = spec
                    .gemm
                    .as_ref()
                    .expect("BLAS plans only exist for GEMM specs");
                let params = gemm.params(workspace.arithmetic, &recipe.workspace_layout)?;
                if !cublas.supports(&params) {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Einsum `{}`: final caller base {:#x} for {} used bytes (arithmetic {:#x}, BLAS {:#x}) does not preserve selected algorithm {:?}; restore the prefix alignment/offset contract and eagerly prepare the final binding before capture",
                        semantic.equation(),
                        workspace.key.base,
                        workspace.key.used_bytes,
                        workspace.arithmetic,
                        workspace.cublas,
                        cublas.algorithm_contract()
                    )));
                }
                launches.push(PreparedLaunch::Cublas { params });
            } else {
                let metadata_offset_bytes = metadata_words.len().checked_mul(std::mem::size_of::<u64>()).ok_or_else(|| EpError::KernelFailed(format!("cuda_ep Einsum `{}`: metadata byte offset overflowed; reduce the contraction", semantic.equation())))?;
                encode_step(
                    semantic,
                    spec,
                    workspace.arithmetic,
                    &recipe.workspace_layout,
                    &mut metadata_words,
                )?;
                launches.push(PreparedLaunch::Generic {
                    metadata_offset_bytes,
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
        Ok((launches, metadata_words))
    }

    fn bind_execution(
        &self,
        runtime: &Arc<CudaRuntime>,
        workspace: &BoundWorkspace,
    ) -> Result<BoundExecution> {
        let recipe = &self.recipe;
        let semantic = recipe.semantic.as_ref();
        let (launches, metadata_words) = self.encode_binding(workspace)?;
        let metadata_bytes = metadata_words
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: metadata byte count overflowed; reduce the contraction",
                    semantic.equation()
                ))
            })?;
        let metadata = if metadata_bytes == 0 {
            None
        } else {
            let allocation = NativeDeviceAllocation::allocate_shared(
                runtime,
                metadata_bytes,
                onnx_runtime_memory_api::MemoryRole::Workspace { step_scoped: false },
            )?;
            let bytes = metadata_words
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect::<Vec<_>>();
            // SAFETY: the fresh immutable owner covers this exact synchronous upload.
            unsafe { runtime.htod(&bytes, allocation.ptr()) }?;
            runtime.staged_warm_cache_mutation("Einsum metadata allocation/upload")?;
            Some(allocation)
        };
        let function = if launches
            .iter()
            .any(|launch| matches!(launch, PreparedLaunch::Generic { .. }))
        {
            runtime.require_nvrtc_half_headers("Einsum")?;
            Some(runtime.nvrtc_raw_function(
                recipe.arithmetic.module_key(recipe.dtype),
                SOURCE,
                recipe.arithmetic.entry(),
            )?)
        } else {
            None
        };
        runtime.staged_warm_cache_mutation("Einsum algorithm/function selection")?;
        Ok(BoundExecution {
            key: workspace.key,
            function,
            launches,
            metadata,
            metadata_bytes,
        })
    }
}

impl CudaEinsumPlan {
    pub(super) fn matches(
        &self,
        requested: RequestedRoute,
        memory_ceiling_bytes: u128,
        signature: &ArithmeticExecutionSignature,
    ) -> bool {
        let recipe = &self.program.recipe;
        recipe.requested == requested
            && recipe.memory_ceiling_bytes == memory_ceiling_bytes
            && recipe.signature == *signature
    }
    pub(super) fn mismatch_reason(
        &self,
        signature: &ArithmeticExecutionSignature,
    ) -> Option<String> {
        self.program.recipe.signature.mismatch_reason(signature)
    }
    pub(super) fn requirement(&self) -> WorkspaceRequirement {
        self.program.requirement()
    }

    /// Returns whether a new final-address snapshot was published. Warm calls
    /// borrow the same exact binding without encoding/upload/allocation.
    pub(super) fn launch(
        &mut self,
        runtime: &Arc<CudaRuntime>,
        expected_device: DeviceId,
        equation: &str,
        signature: &ArithmeticExecutionSignature,
        offered: Option<WorkspaceView>,
        capturing: bool,
    ) -> Result<bool> {
        if let Some(reason) = self.program.recipe.signature.mismatch_reason(signature) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum: semantic execution signature changed after validation: {reason}"
            )));
        }
        signature.validate_device_ownership(expected_device, equation)?;
        signature.validate_alias_proof(equation)?;
        let workspace = self.program.workspace.bind(offered)?;
        let hit = self
            .binding
            .as_ref()
            .is_some_and(|binding| binding.key == workspace.key);
        if capturing && !hit {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum `{equation}`: semantic caller workspace binding changed during capture; abort capture and eagerly warm this exact workspace address/layout before recording"
            )));
        }
        let candidate = if hit {
            None
        } else {
            Some(self.program.bind_execution(runtime, &workspace)?)
        };
        let binding = candidate
            .as_ref()
            .or(self.binding.as_ref())
            .expect("a warmed or new binding exists");
        if capturing {
            if workspace.used.bytes() != 0 {
                runtime.require_registered_workspace_capture(
                    workspace.used,
                    "Einsum semantic caller workspace",
                )?;
            }
            if let Some(metadata) = &binding.metadata {
                runtime.require_captured_address_owner(
                    NativeDeviceAllocation::device_graph_resource(metadata).identity(),
                    "immutable CUDA Einsum metadata",
                )?;
            }
        }
        for (launch, plan) in binding.launches.iter().zip(&self.program.cublas) {
            match launch {
                PreparedLaunch::Generic {
                    metadata_offset_bytes,
                    config,
                } => {
                    let function = binding.function.as_ref().ok_or_else(|| EpError::KernelFailed("cuda_ep Einsum: generic launch lost its prepared function; eagerly warm the exact binding".into()))?;
                    let metadata = binding.metadata.as_ref().ok_or_else(|| EpError::KernelFailed("cuda_ep Einsum: generic launch lost its immutable metadata owner; eagerly warm the exact binding".into()))?;
                    let mut pointer = metadata.ptr().checked_add(*metadata_offset_bytes as u64).ok_or_else(|| EpError::KernelFailed("cuda_ep Einsum: metadata device-pointer offset overflowed; reduce the contraction".into()))?;
                    let mut params = [(&mut pointer as *mut CUdeviceptr).cast::<c_void>()];
                    // SAFETY: complete current I/O signature and caller span,
                    // immutable metadata owner, and one-argument kernel ABI.
                    unsafe { function.launch(runtime.stream(), *config, &mut params) }
                        .map_err(|error| driver_err("launch generic CUDA Einsum", error))?;
                }
                PreparedLaunch::Cublas { params } => {
                    let plan = plan.as_ref().expect("bound BLAS launch has its host plan");
                    // SAFETY: final actual-address support was verified before
                    // metadata publication; every serial step shares this tail.
                    unsafe {
                        plan.launch(
                            runtime.blas(),
                            runtime.stream_ptr(),
                            params,
                            workspace.cublas,
                        )
                    }?;
                }
            }
        }
        if let Some(candidate) = candidate {
            self.binding = Some(candidate);
        }
        Ok(!hit)
    }
    pub(super) fn resources(&self) -> Vec<DeviceGraphResource> {
        self.binding
            .as_ref()
            .and_then(|binding| binding.metadata.as_ref())
            .map(NativeDeviceAllocation::device_graph_resource)
            .into_iter()
            .collect()
    }
    pub(super) fn summary(&self) -> CudaEinsumPlanSummary {
        let recipe = &self.program.recipe;
        CudaEinsumPlanSummary {
            route: recipe.route,
            planner_quality: recipe.planner_quality,
            workspace_bytes: self.program.workspace.layout.bytes(),
            metadata_bytes: self
                .binding
                .as_ref()
                .map_or(0, |binding| binding.metadata_bytes),
            kernel_launches: recipe.step_specs.len(),
            cublas_launches: self
                .program
                .cublas
                .iter()
                .filter(|plan| plan.is_some())
                .count(),
        }
    }
    pub(super) fn workspace_ptr(&self) -> CUdeviceptr {
        self.binding.as_ref().map_or(0, |binding| binding.key.base)
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
) -> Result<ValueLayout> {
    Ok(ValueLayout {
        pointer,
        dtype,
        strides: compute_contiguous_strides(&shape)?,
        axes,
        shape,
    })
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

impl GemmSpec {
    fn params(
        &self,
        workspace: CUdeviceptr,
        layout: &WorkspaceLayout,
    ) -> Result<StridedBatchedGemmParams> {
        Ok(StridedBatchedGemmParams {
            dtype: GemmDtype::F32,
            a: resolve_pointer(&self.left, workspace, layout)?,
            b: resolve_pointer(&self.right, workspace, layout)?,
            c: resolve_pointer(&self.output, workspace, layout)?,
            m: self.m,
            k: self.k,
            n: self.n,
            batch: self.batch,
            transpose_a: self.transpose_left,
            transpose_b: self.transpose_right,
            a_batch_stride: self.left_batch_stride,
            b_batch_stride: self.right_batch_stride,
        })
    }
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
            dense_layout(0, intermediate_dtype, temporary.axes().to_vec(), shape)?,
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

    fn recipe(requested: RequestedRoute, ceiling: u128) -> Result<CudaEinsumRecipe> {
        // Pure address descriptions: building a recipe must not access a driver.
        let shapes: &[&[usize]] = &[&[2, 3], &[3, 4], &[4, 5]];
        let plan = EinsumShapePlan::build_for_schema(
            "ab,bc,cd->ad",
            shapes,
            onnx_runtime_ir::EinsumSchema::resolve(12).unwrap(),
        )
        .unwrap();
        let tensor = |shape: &[usize], pointer| ArithmeticTensorSignature {
            device: DeviceId::cuda(0),
            raw_base_pointer: pointer,
            byte_offset: 0,
            effective_pointer: pointer,
            dtype: DataType::Float32,
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape)
                .expect("fixture shape has representable contiguous strides"),
        };
        let signature = ArithmeticExecutionSignature {
            inputs: shapes
                .iter()
                .enumerate()
                .map(|(i, shape)| tensor(shape, (i as u64 + 1) * 0x1000))
                .collect(),
            output: tensor(&[2, 5], 0x4000),
            alias_proof: super::super::ArithmeticAliasProof::OutputDisjointFromEveryInput,
        };
        CudaEinsumRecipe::build(Arc::new(plan), signature, requested, ceiling)
    }

    #[test]
    fn accepted_empty_semantic_output_has_no_scratch_metadata_or_launch_steps() {
        let shapes: &[&[usize]] = &[&[0, 3], &[3, 4], &[4, 5]];
        let semantic = Arc::new(
            EinsumShapePlan::build_for_schema(
                "ab,bc,cd->ad",
                shapes,
                onnx_runtime_ir::EinsumSchema::resolve(12).unwrap(),
            )
            .unwrap(),
        );
        let tensor = |shape: &[usize], pointer| ArithmeticTensorSignature {
            device: DeviceId::cuda(0),
            raw_base_pointer: pointer,
            byte_offset: 0,
            effective_pointer: pointer,
            dtype: DataType::Float32,
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape)
                .expect("fixture shape has representable contiguous strides"),
        };
        let signature = ArithmeticExecutionSignature {
            inputs: shapes
                .iter()
                .enumerate()
                .map(|(i, shape)| tensor(shape, (i as u64 + 1) * 0x1000))
                .collect(),
            output: tensor(&[0, 5], 0x4000),
            alias_proof: super::super::ArithmeticAliasProof::OutputDisjointFromEveryInput,
        };
        for route in [RequestedRoute::GenericNative, RequestedRoute::Optimized] {
            let recipe = CudaEinsumRecipe::build(
                semantic.clone(),
                signature.clone(),
                route,
                DEFAULT_MEMORY_CEILING_BYTES,
            )
            .unwrap();
            assert_eq!(recipe.prefix_requirement().bytes, 0);
            assert!(recipe.step_specs.is_empty());
            assert!(!recipe.needs_binding());
        }
    }

    #[test]
    fn semantic_composite_keeps_padding_disjoint_and_zero_demand_empty() {
        let layout = SemanticWorkspace::new(12, 513).unwrap();
        assert_eq!(layout.requirement().bytes, 256 + 513);
        let used = layout
            .bind(Some(WorkspaceView::new(
                DevicePtrMut(0x1000usize as *mut _),
                4096,
            )))
            .unwrap();
        assert_eq!(
            used.key,
            BindingKey {
                base: 0x1000,
                used_bytes: 769
            }
        );
        assert_eq!((used.arithmetic, used.cublas), (0x1000, 0x1100));
        assert_eq!(used.used.bytes(), 769);
        assert_eq!(
            SemanticWorkspace::new(12, 0).unwrap().requirement().bytes,
            12
        );
        let empty = SemanticWorkspace::new(0, 0).unwrap().bind(None).unwrap();
        assert_eq!(
            (empty.key.used_bytes, empty.arithmetic, empty.cublas),
            (0, 0, 0)
        );
        assert_eq!(
            SemanticWorkspace::new(0, 513).unwrap().requirement().bytes,
            513
        );
        for offered in [
            None,
            Some(WorkspaceView::new(DevicePtrMut(0x1000usize as *mut _), 768)),
            Some(WorkspaceView::new(
                DevicePtrMut(0x1001usize as *mut _),
                4096,
            )),
        ] {
            let error = layout.bind(offered).err().unwrap().to_string();
            assert!(
                error.contains("769 bytes") && error.contains("prepare and pass"),
                "{error}"
            );
        }
        assert!(SemanticWorkspace::new(usize::MAX, 1).is_err());
    }

    #[test]
    fn host_program_encodes_only_final_prefix_addresses_after_relocation() {
        // Pure generic metadata encoding, never a BLAS heuristic with synthetic addresses.
        let recipe = recipe(RequestedRoute::Optimized, DEFAULT_MEMORY_CEILING_BYTES).unwrap();
        let offsets = recipe
            .workspace_layout
            .values
            .values()
            .copied()
            .collect::<Vec<_>>();
        let workspace = SemanticWorkspace::new(recipe.workspace_layout.bytes, 257).unwrap();
        let program = CudaEinsumProgram {
            cublas: (0..recipe.step_specs.len()).map(|_| None).collect(),
            recipe,
            workspace,
        };
        let bind = |base, capacity| {
            program
                .workspace
                .bind(Some(WorkspaceView::new(
                    DevicePtrMut(base as *mut _),
                    capacity,
                )))
                .unwrap()
        };
        let first = bind(0x10000usize, 8192);
        let final_binding = bind(0x20100usize, 16384);
        let (_, before) = program.encode_binding(&first).unwrap();
        let (_, after) = program.encode_binding(&final_binding).unwrap();
        for offset in offsets {
            assert!(before.contains(&(first.arithmetic + offset as u64)));
            assert!(after.contains(&(final_binding.arithmetic + offset as u64)));
            assert!(!after.contains(&(first.arithmetic + offset as u64)));
            assert_eq!(
                (first.arithmetic + offset as u64) % WORKSPACE_ALIGNMENT as u64,
                0
            );
            assert_eq!(
                (final_binding.arithmetic + offset as u64) % WORKSPACE_ALIGNMENT as u64,
                0
            );
        }
        assert_eq!(
            final_binding.key,
            bind(0x20100usize, 32768).key,
            "retained capacity is not a snapshot key"
        );
    }

    #[test]
    fn host_recipe_selects_tree_and_aligned_live_slots_without_cuda() {
        let recipe = recipe(RequestedRoute::Optimized, DEFAULT_MEMORY_CEILING_BYTES).unwrap();
        assert_eq!(recipe.route, CudaEinsumRoute::OptimizedDp);
        assert_eq!(recipe.step_specs.len(), 2);
        assert!(!recipe.workspace_layout.values.is_empty());
        assert!(recipe.workspace_layout.bytes > 0);
        for offset in recipe.workspace_layout.values.values() {
            assert_eq!(offset % WORKSPACE_ALIGNMENT, 0);
            assert!(*offset < recipe.workspace_layout.bytes);
        }
    }

    #[test]
    fn host_recipe_ceiling_falls_back_only_for_auto() {
        let generic = recipe(RequestedRoute::GenericNative, 0).unwrap();
        assert_eq!(generic.route, CudaEinsumRoute::GenericNative);
        assert_eq!(generic.workspace_layout.bytes, 0);
        assert_eq!(generic.step_specs.len(), 1);
        let auto = recipe(RequestedRoute::Auto, 0).unwrap();
        assert_eq!(auto.route, CudaEinsumRoute::GenericNative);
        assert_eq!(auto.workspace_layout.bytes, 0);
        let error = recipe(RequestedRoute::Optimized, 0)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("0-byte workspace ceiling")
                && error.contains("raise the explicit ceiling"),
            "{error}"
        );
    }

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

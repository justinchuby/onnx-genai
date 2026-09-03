//! Capture-safe CUDA execution for canonical ONNX `Einsum` plans.
//!
//! The equation is parsed exactly once by [`EinsumShapePlan`] when the
//! shape-specialized kernel is created. Execution consumes only the plan's axis
//! mappings. Binary contractions whose contiguous storage can be represented by
//! cuBLASLt layouts run without materialized operand transposes; view-only
//! permutations and diagonals use the executor's zero-copy view contract.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{
    CaptureSupport, DeviceGraphResource, EpError, Kernel, KernelFactory, Result, TensorMut,
    TensorView, ViewOutput,
};
use onnx_runtime_ir::{
    DataType, EinsumContractionPlan, EinsumContractionTreePlan, EinsumInput, EinsumOperandPlan,
    EinsumPermutationPlan, EinsumPlan, EinsumPlannerQuality, EinsumPlanningClassification,
    EinsumSchema, EinsumShapePlan, Node, Shape, TensorLayout,
};

use super::movement::{PersistentMetadata, launch_persistent_metadata};
use crate::blas::{
    self, CaptureStridedBatchedGemmPlan, GemmDtype, StridedBatchedGemmParams, WORKSPACE_BYTES,
};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, GraphDeviceAllocation, cuptr};

/// Counters proving which CUDA Einsum route executed and what persistent state
/// it established. Values are process-global diagnostics, so GPU tests serialize
/// around reset/read windows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EinsumExecutionStats {
    pub plan_builds: u64,
    pub plan_cache_hits: u64,
    pub view_aliases: u64,
    pub view_materializations: u64,
    pub gemm_launches: u64,
    pub canonical_gemm_launches: u64,
    pub descriptor_transpose_gemm_launches: u64,
    pub zero_fill_launches: u64,
    pub capture_recordings: u64,
    pub claim_fallbacks: u64,
    pub last_fallback_reason: Option<String>,
    pub workspace_bytes: u64,
    pub workspace_ptr: u64,
    pub setup_ns: u64,
    pub persistent_metadata_bytes: u64,
    pub materialization_bytes: u64,
}

static PLAN_BUILDS: AtomicU64 = AtomicU64::new(0);
static PLAN_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static VIEW_ALIASES: AtomicU64 = AtomicU64::new(0);
static VIEW_MATERIALIZATIONS: AtomicU64 = AtomicU64::new(0);
static GEMM_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static CANONICAL_GEMM_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static DESCRIPTOR_TRANSPOSE_GEMM_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static ZERO_FILL_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static CAPTURE_RECORDINGS: AtomicU64 = AtomicU64::new(0);
static CLAIM_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static LAST_FALLBACK_REASON: Mutex<Option<String>> = Mutex::new(None);
static WORKSPACE_BYTES_LAST: AtomicU64 = AtomicU64::new(0);
static WORKSPACE_PTR_LAST: AtomicU64 = AtomicU64::new(0);
static SETUP_NS_LAST: AtomicU64 = AtomicU64::new(0);
static PERSISTENT_METADATA_BYTES_LAST: AtomicU64 = AtomicU64::new(0);
static MATERIALIZATION_BYTES: AtomicU64 = AtomicU64::new(0);

fn contraction_tree_summary(tree: &EinsumContractionTreePlan) -> String {
    if tree.quality() == EinsumPlannerQuality::GenericNativeFallback {
        let reason = tree
            .fallback_reason()
            .expect("GenericNative planner fallback records its reason");
        format!(
            "the bounded planner selected GenericNative fallback because {reason} (work={}, \
             metadata_units={}, max_depth={})",
            tree.usage().work(),
            tree.usage().metadata_units(),
            tree.usage().max_depth()
        )
    } else {
        format!(
            "{} ordered candidate(s), quality {:?}",
            tree.candidates().len(),
            tree.quality()
        )
    }
}

pub fn einsum_execution_stats() -> EinsumExecutionStats {
    let last_fallback_reason = LAST_FALLBACK_REASON
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone();
    EinsumExecutionStats {
        plan_builds: PLAN_BUILDS.load(Ordering::Relaxed),
        plan_cache_hits: PLAN_CACHE_HITS.load(Ordering::Relaxed),
        view_aliases: VIEW_ALIASES.load(Ordering::Relaxed),
        view_materializations: VIEW_MATERIALIZATIONS.load(Ordering::Relaxed),
        gemm_launches: GEMM_LAUNCHES.load(Ordering::Relaxed),
        canonical_gemm_launches: CANONICAL_GEMM_LAUNCHES.load(Ordering::Relaxed),
        descriptor_transpose_gemm_launches: DESCRIPTOR_TRANSPOSE_GEMM_LAUNCHES
            .load(Ordering::Relaxed),
        zero_fill_launches: ZERO_FILL_LAUNCHES.load(Ordering::Relaxed),
        capture_recordings: CAPTURE_RECORDINGS.load(Ordering::Relaxed),
        claim_fallbacks: CLAIM_FALLBACKS.load(Ordering::Relaxed),
        last_fallback_reason,
        workspace_bytes: WORKSPACE_BYTES_LAST.load(Ordering::Relaxed),
        workspace_ptr: WORKSPACE_PTR_LAST.load(Ordering::Relaxed),
        setup_ns: SETUP_NS_LAST.load(Ordering::Relaxed),
        persistent_metadata_bytes: PERSISTENT_METADATA_BYTES_LAST.load(Ordering::Relaxed),
        materialization_bytes: MATERIALIZATION_BYTES.load(Ordering::Relaxed),
    }
}

pub fn reset_einsum_execution_stats() {
    PLAN_BUILDS.store(0, Ordering::Relaxed);
    PLAN_CACHE_HITS.store(0, Ordering::Relaxed);
    VIEW_ALIASES.store(0, Ordering::Relaxed);
    VIEW_MATERIALIZATIONS.store(0, Ordering::Relaxed);
    GEMM_LAUNCHES.store(0, Ordering::Relaxed);
    CANONICAL_GEMM_LAUNCHES.store(0, Ordering::Relaxed);
    DESCRIPTOR_TRANSPOSE_GEMM_LAUNCHES.store(0, Ordering::Relaxed);
    ZERO_FILL_LAUNCHES.store(0, Ordering::Relaxed);
    CAPTURE_RECORDINGS.store(0, Ordering::Relaxed);
    CLAIM_FALLBACKS.store(0, Ordering::Relaxed);
    *LAST_FALLBACK_REASON
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = None;
    WORKSPACE_BYTES_LAST.store(0, Ordering::Relaxed);
    WORKSPACE_PTR_LAST.store(0, Ordering::Relaxed);
    SETUP_NS_LAST.store(0, Ordering::Relaxed);
    PERSISTENT_METADATA_BYTES_LAST.store(0, Ordering::Relaxed);
    MATERIALIZATION_BYTES.store(0, Ordering::Relaxed);
}

fn equation(node: &Node) -> Result<&str> {
    let attribute = node.attr("equation").ok_or_else(|| {
        EpError::KernelFailed(
            "cuda_ep Einsum: required string attribute `equation` is missing".into(),
        )
    })?;
    attribute.as_str().ok_or_else(|| {
        EpError::KernelFailed(
            "cuda_ep Einsum: attribute `equation` must be a valid UTF-8 string".into(),
        )
    })
}

fn einsum_dtype(dtype: DataType) -> Result<GemmDtype> {
    match dtype {
        DataType::Float32 => Ok(GemmDtype::F32),
        DataType::Float16 => Ok(GemmDtype::F16),
        other => Err(not_implemented(format!(
            "Einsum dtype {other:?}; the staged native CUDA executor currently supports only Float32 and Float16. Einsum-28 admits BFloat16 semantically, but its CUDA execution handoff is not implemented yet"
        ))),
    }
}

fn physical_axis(operand: &EinsumOperandPlan, unique_axis: usize) -> Result<usize> {
    let axis = operand.unique_axes().get(unique_axis).ok_or_else(|| {
        EpError::KernelFailed(format!(
            "cuda_ep Einsum: canonical operand #{} references missing unique axis {unique_axis}",
            operand.input()
        ))
    })?;
    let [physical] = axis.input_axes() else {
        return Err(not_implemented(format!(
            "Einsum contraction with a diagonal on operand #{}; use a separate diagonal view before the contraction",
            operand.input()
        )));
    };
    Ok(*physical)
}

fn physical_sequence(
    operand: &EinsumOperandPlan,
    order: impl IntoIterator<Item = Option<usize>>,
) -> Result<Vec<usize>> {
    order
        .into_iter()
        .flatten()
        .map(|axis| physical_axis(operand, axis))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StorageOrder {
    Canonical,
    Transposed,
}

fn storage_order(
    operand: &EinsumOperandPlan,
    order: &[Option<usize>],
    batch_rank: usize,
    first_group_rank: usize,
) -> Result<StorageOrder> {
    if order.len() < batch_rank + first_group_rank {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Einsum: canonical axis order for operand #{} is truncated",
            operand.input()
        )));
    }
    let (batch, matrix) = order.split_at(batch_rank);
    let (first, second) = matrix.split_at(first_group_rank);
    if matrix.iter().any(Option::is_none) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Einsum: canonical matrix axes for operand #{} contain a synthetic singleton",
            operand.input()
        )));
    }

    let expected: Vec<_> = (0..operand.rank()).collect();
    let canonical = physical_sequence(
        operand,
        batch
            .iter()
            .copied()
            .chain(first.iter().copied())
            .chain(second.iter().copied()),
    )?;
    if canonical == expected {
        return Ok(StorageOrder::Canonical);
    }
    let transposed = physical_sequence(
        operand,
        batch
            .iter()
            .copied()
            .chain(second.iter().copied())
            .chain(first.iter().copied()),
    )?;
    if transposed == expected {
        return Ok(StorageOrder::Transposed);
    }
    Err(not_implemented(format!(
        "Einsum operand #{} axis permutation cannot be represented by one cuBLASLt transpose descriptor",
        operand.input()
    )))
}

fn contraction_structure_reason(
    plan: &EinsumShapePlan,
    contraction: &EinsumContractionPlan,
) -> Option<String> {
    if contraction
        .output_permutation()
        .iter()
        .copied()
        .ne(0..contraction.output_permutation().len())
    {
        return Some(format!(
            "cuda_ep Einsum `{}`: requested output permutation {:?} is not a contiguous canonical [batch..., M..., N...] result; insert an explicit Transpose after Einsum",
            plan.equation(),
            contraction.output_permutation()
        ));
    }
    let [left, right] = plan.operands() else {
        return Some(format!(
            "cuda_ep Einsum `{}`: canonical GEMM classification did not contain exactly two operands",
            plan.equation()
        ));
    };
    if let Err(error) = storage_order(
        left,
        contraction.left_axis_order(),
        contraction.batch_axes().len(),
        contraction.left_free_axes().len(),
    ) {
        return Some(error.to_string());
    }
    if let Err(error) = storage_order(
        right,
        contraction.right_axis_order(),
        contraction.batch_axes().len(),
        contraction.contract_axes().len(),
    ) {
        return Some(error.to_string());
    }
    None
}

fn layout_is_contiguous(layout: &TensorLayout, shape: &Shape) -> bool {
    if layout.strides.is_none() {
        return true;
    }
    let Some(shape) = onnx_runtime_ir::as_static_shape(shape) else {
        return false;
    };
    layout.is_contiguous(&shape)
}

/// Return an actionable claim decline for an Einsum the current CUDA lowering
/// cannot execute without an unsupported materialization.
fn unsupported_reason_impl(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    input_dtypes: &[DataType],
    layouts: &[TensorLayout],
) -> Option<String> {
    let equation = match equation(node) {
        Ok(equation) => equation,
        Err(error) => return Some(error.to_string()),
    };
    if shapes.len() != input_dtypes.len() {
        return Some(format!(
            "cuda_ep Einsum `{equation}`: received {} shapes but {} input dtypes",
            shapes.len(),
            input_dtypes.len()
        ));
    }
    if !layouts.is_empty() && layouts.len() != shapes.len() {
        return Some(format!(
            "cuda_ep Einsum `{equation}`: received {} input layouts for {} inputs",
            layouts.len(),
            shapes.len()
        ));
    }
    let inputs = shapes
        .iter()
        .zip(input_dtypes)
        .map(|(shape, &dtype)| EinsumInput::new(dtype, shape.as_slice()))
        .collect::<Vec<_>>();
    let plan = match EinsumPlan::build_for_opset(equation, &inputs, opset) {
        Ok(plan) => plan,
        Err(error) => return Some(format!("cuda_ep Einsum `{equation}`: {error}")),
    };
    if let Err(error) = einsum_dtype(plan.dtype()) {
        return Some(error.to_string());
    }

    match plan.planning_classification() {
        EinsumPlanningClassification::ViewOnlyPermutation(_)
        | EinsumPlanningClassification::DiagonalView(_) => None,
        EinsumPlanningClassification::Gemm(contraction) => {
            if let Some(reason) = contraction_structure_reason(plan.shape_plan(), contraction) {
                return Some(reason);
            }
            if layouts
                .iter()
                .zip(shapes)
                .any(|(layout, shape)| !layout_is_contiguous(layout, shape))
            {
                return Some(format!(
                    "cuda_ep Einsum `{equation}`: GEMM/BMM contractions require contiguous inputs; materialize the strided input before Einsum"
                ));
            }
            // With fully static dimensions, reject partial multi-axis batch
            // broadcasting at claim time rather than after output mutation.
            if let Some(concrete) = shapes
                .iter()
                .map(|shape| onnx_runtime_ir::as_static_shape(shape))
                .collect::<Option<Vec<_>>>()
                && let Err(error) = concrete_contraction_layout(
                    plan.shape_plan(),
                    contraction,
                    &concrete,
                    plan.dtype(),
                )
            {
                return Some(error.to_string());
            }
            None
        }
        EinsumPlanningClassification::ContractionTree(tree) => Some(format!(
            "cuda_ep Einsum `{equation}`: canonical {}-input contraction plan has {}, but CUDA \
             temporary scheduling and multi-node capture execution are not implemented",
            tree.arity(),
            contraction_tree_summary(tree)
        )),
        EinsumPlanningClassification::ReductionOrElementwise(_) => Some(format!(
            "cuda_ep Einsum `{equation}`: uncoupled reductions/elementwise products are not yet lowered; use native Reduce*/Mul nodes or CPU fallback"
        )),
        _ => Some(format!(
            "cuda_ep Einsum `{equation}`: canonical planner returned a newer classification that \
             this CUDA EP does not recognize; update claim, execution, and capture paths before \
             assigning the node"
        )),
    }
}

/// Claim-time capability check using the original Einsum-12 contract.
///
/// This compatibility wrapper intentionally does not inspect node metadata or
/// infer a schema from the operand dtypes. Model/provider paths that have an
/// effective opset must call [`unsupported_reason_for_opset`] instead.
pub fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
    layouts: &[TensorLayout],
) -> Option<String> {
    unsupported_reason_for_opset(node, 12, shapes, input_dtypes, layouts)
}

/// Claim-time capability check for a model's effective ONNX opset.
pub fn unsupported_reason_for_opset(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    input_dtypes: &[DataType],
    layouts: &[TensorLayout],
) -> Option<String> {
    let reason = unsupported_reason_impl(node, opset, shapes, input_dtypes, layouts);
    if let Some(reason) = &reason {
        CLAIM_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        *LAST_FALLBACK_REASON
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(reason.clone());
    }
    reason
}

pub struct EinsumFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for EinsumFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let equation = equation(node)?.to_owned();
        let input_shape_refs = input_shapes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let schema = EinsumSchema::resolve(node.local_opset().unwrap_or(12))
            .map_err(|error| EpError::KernelFailed(format!("cuda_ep Einsum: {error}")))?;
        let plan = EinsumShapePlan::build_for_schema(&equation, &input_shape_refs, schema)
            .map_err(|error| {
                EpError::KernelFailed(format!("cuda_ep Einsum `{equation}`: {error}"))
            })?;
        match plan.planning_classification() {
            EinsumPlanningClassification::ViewOnlyPermutation(_)
            | EinsumPlanningClassification::DiagonalView(_)
            | EinsumPlanningClassification::Gemm(_) => {}
            EinsumPlanningClassification::ContractionTree(tree) => {
                return Err(not_implemented(format!(
                    "cuda_ep Einsum `{equation}` {}-input contraction plan with {}; implement \
                     GenericNative/temporary scheduling and multi-node capture before constructing \
                     this kernel",
                    tree.arity(),
                    contraction_tree_summary(tree)
                )));
            }
            EinsumPlanningClassification::ReductionOrElementwise(_) => {
                return Err(not_implemented(format!(
                    "cuda_ep Einsum `{equation}` reduction/elementwise canonical plan"
                )));
            }
            _ => {
                return Err(not_implemented(format!(
                    "cuda_ep Einsum `{equation}` newer unrecognized canonical classification; \
                     update claim, factory, execution, and capture paths before constructing it"
                )));
            }
        }
        Ok(Box::new(EinsumKernel {
            runtime: self.runtime.clone(),
            input_shapes: input_shapes.to_vec(),
            plan,
            execution: Mutex::new(None),
            view_metadata: Mutex::new(PersistentMetadata::new(self.runtime.clone())),
            view_materialization: Mutex::new(None),
            view_alias_warmed: AtomicBool::new(false),
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ContractionLayout {
    dtype: DataType,
    output_shape: Vec<usize>,
    batch_shape: Vec<usize>,
    m: usize,
    k: usize,
    n: usize,
    left_order: StorageOrder,
    right_order: StorageOrder,
    left_batch_stride: usize,
    right_batch_stride: usize,
}

fn checked_product(values: &[usize], target: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or_else(|| {
            EpError::KernelFailed(format!("cuda_ep Einsum: {target} product overflows usize"))
        })
    })
}

fn operand_batch_stride(
    operand: &EinsumOperandPlan,
    order: &[Option<usize>],
    batch_shape: &[usize],
    input_shape: &[usize],
    matrix_elements: usize,
) -> Result<usize> {
    let mut operand_batch = Vec::with_capacity(batch_shape.len());
    for (&axis, &output_dim) in order.iter().take(batch_shape.len()).zip(batch_shape) {
        let dim = match axis {
            Some(axis) => input_shape[physical_axis(operand, axis)?],
            None => 1,
        };
        if dim != 1 && dim != output_dim {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum: operand #{} batch extent {dim} does not broadcast to {output_dim}",
                operand.input()
            )));
        }
        operand_batch.push(dim);
    }
    if batch_shape.iter().all(|&dim| dim == 1) || operand_batch.iter().all(|&dim| dim == 1) {
        return Ok(0);
    }
    if operand_batch == batch_shape {
        return Ok(matrix_elements);
    }
    Err(not_implemented(format!(
        "Einsum operand #{} uses partial multi-axis batch broadcasting {:?} -> {:?}; cuBLASLt supports this lowering only when the whole operand batch is equal or stride-zero broadcast",
        operand.input(),
        operand_batch,
        batch_shape
    )))
}

fn concrete_contraction_layout(
    plan: &EinsumShapePlan,
    contraction: &EinsumContractionPlan,
    input_shapes: &[Vec<usize>],
    dtype: DataType,
) -> Result<ContractionLayout> {
    let [left_shape, right_shape] = input_shapes else {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Einsum `{}`: GEMM/BMM lowering requires exactly two inputs",
            plan.equation()
        )));
    };
    let shapes = [left_shape.as_slice(), right_shape.as_slice()];
    let output_shape = plan
        .resolve_concrete_output_shape(&shapes)
        .map_err(|error| {
            EpError::KernelFailed(format!("cuda_ep Einsum `{}`: {error}", plan.equation()))
        })?;
    let geometry = plan
        .resolve_concrete_gemm_geometry(&shapes)
        .map_err(|error| {
            EpError::KernelFailed(format!("cuda_ep Einsum `{}`: {error}", plan.equation()))
        })?
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: canonical plan lost its GEMM geometry",
                plan.equation()
            ))
        })?;
    let [left, right] = plan.operands() else {
        unreachable!("the concrete input count was checked above");
    };
    let left_order = storage_order(
        left,
        contraction.left_axis_order(),
        contraction.batch_axes().len(),
        contraction.left_free_axes().len(),
    )?;
    let right_order = storage_order(
        right,
        contraction.right_axis_order(),
        contraction.batch_axes().len(),
        contraction.contract_axes().len(),
    )?;
    let left_matrix = geometry.m().checked_mul(geometry.k()).ok_or_else(|| {
        EpError::KernelFailed("cuda_ep Einsum: left matrix element count overflows usize".into())
    })?;
    let right_matrix = geometry.k().checked_mul(geometry.n()).ok_or_else(|| {
        EpError::KernelFailed("cuda_ep Einsum: right matrix element count overflows usize".into())
    })?;
    let left_batch_stride = operand_batch_stride(
        left,
        contraction.left_axis_order(),
        geometry.batch_shape(),
        left_shape,
        left_matrix,
    )?;
    let right_batch_stride = operand_batch_stride(
        right,
        contraction.right_axis_order(),
        geometry.batch_shape(),
        right_shape,
        right_matrix,
    )?;
    Ok(ContractionLayout {
        dtype,
        output_shape,
        batch_shape: geometry.batch_shape().to_vec(),
        m: geometry.m(),
        k: geometry.k(),
        n: geometry.n(),
        left_order,
        right_order,
        left_batch_stride,
        right_batch_stride,
    })
}

enum ExecutionKind {
    NoOp,
    ZeroFill,
    Gemm(CachedGemm),
}

struct CachedExecution {
    dtype: DataType,
    input_shapes: Vec<Vec<usize>>,
    layout: ContractionLayout,
    kind: ExecutionKind,
}

impl CachedExecution {
    fn matches(&self, inputs: &[TensorView], output: &TensorMut) -> bool {
        self.dtype == inputs[0].dtype
            && self.input_shapes.len() == inputs.len()
            && self
                .input_shapes
                .iter()
                .zip(inputs)
                .all(|(expected, actual)| expected.as_slice() == actual.shape)
            && self.layout.output_shape.as_slice() == output.shape
    }
}

struct CachedGemm {
    plan: CaptureStridedBatchedGemmPlan,
    workspace: Option<Arc<GraphDeviceAllocation>>,
}

impl CachedGemm {
    fn launch(
        &self,
        layout: &ContractionLayout,
        runtime: &CudaRuntime,
        a: CUdeviceptr,
        b: CUdeviceptr,
        c: CUdeviceptr,
    ) -> Result<()> {
        let params = StridedBatchedGemmParams {
            dtype: einsum_dtype(layout.dtype)?,
            a,
            b,
            c,
            m: layout.m,
            k: layout.k,
            n: layout.n,
            batch: checked_product(&layout.batch_shape, "batch")?,
            transpose_a: layout.left_order == StorageOrder::Transposed,
            transpose_b: layout.right_order == StorageOrder::Transposed,
            a_batch_stride: layout.left_batch_stride,
            b_batch_stride: layout.right_batch_stride,
        };
        // SAFETY: the cached layout was admitted against these exact contiguous
        // input/output shapes; aliasing was rejected by the caller; workspace
        // is the exact persistent allocation selected during warmup.
        unsafe {
            self.plan.launch(
                runtime.blas(),
                runtime.stream_ptr(),
                &params,
                self.workspace
                    .as_ref()
                    .map_or(0, |workspace| workspace.ptr()),
            )
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ViewMaterialization {
    dtype: DataType,
    input_shape: Vec<usize>,
    input_strides: Vec<i64>,
    output_shape: Vec<usize>,
    metadata: Vec<u64>,
}

impl ViewMaterialization {
    fn matches(&self, input: &TensorView, output: &TensorMut) -> bool {
        self.dtype == input.dtype
            && self.input_shape.as_slice() == input.shape
            && self.input_strides.as_slice() == input.strides
            && self.output_shape.as_slice() == output.shape
    }
}

pub struct EinsumKernel {
    runtime: Arc<CudaRuntime>,
    input_shapes: Vec<Vec<usize>>,
    plan: EinsumShapePlan,
    execution: Mutex<Option<CachedExecution>>,
    view_metadata: Mutex<PersistentMetadata>,
    view_materialization: Mutex<Option<ViewMaterialization>>,
    view_alias_warmed: AtomicBool,
    last_call_capture_safe: AtomicBool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeviceByteRange {
    start: u64,
    end: u64,
}

fn checked_nonnegative_strided_byte_range(
    base: CUdeviceptr,
    byte_offset: usize,
    dtype: DataType,
    shape: &[usize],
    strides: &[i64],
    context: &str,
) -> Result<Option<DeviceByteRange>> {
    if shape.len() != strides.len() {
        return Err(EpError::KernelFailed(format!(
            "{context}: shape rank {} does not match stride rank {}; provide a valid tensor view",
            shape.len(),
            strides.len()
        )));
    }
    if let Some((axis, stride)) = strides
        .iter()
        .copied()
        .enumerate()
        .find(|(_, stride)| *stride < 0)
    {
        return Err(not_implemented(format!(
            "{context} with negative stride {stride} on axis {axis}; execute through the zero-copy view path"
        )));
    }
    if shape.contains(&0) {
        return Ok(None);
    }
    let element_bytes = u64::try_from(dtype.byte_size()).map_err(|_| {
        EpError::KernelFailed(format!(
            "{context}: element byte size does not fit u64 device addressing"
        ))
    })?;
    if element_bytes == 0 {
        return Err(EpError::KernelFailed(format!(
            "{context}: dtype {dtype:?} has no fixed-width addressable element size"
        )));
    }
    let offset = u64::try_from(byte_offset).map_err(|_| {
        EpError::KernelFailed(format!(
            "{context}: byte_offset {byte_offset} does not fit u64 device addressing"
        ))
    })?;
    let start = base.checked_add(offset).ok_or_else(|| {
        EpError::KernelFailed(format!(
            "{context}: address range overflows u64 while adding base {base:#x} and byte_offset {byte_offset}; use a view whose byte offset, shape, and strides fit device addressing"
        ))
    })?;
    let max_element_offset = shape.iter().zip(strides).enumerate().try_fold(
        0u64,
        |offset, (axis, (&dim, &stride))| {
            let dim_extent = u64::try_from(dim - 1).map_err(|_| {
                EpError::KernelFailed(format!(
                    "{context}: axis {axis} extent {} does not fit u64 device addressing",
                    dim - 1
                ))
            })?;
            let stride = u64::try_from(stride).expect("negative strides were rejected above");
            let axis_extent = dim_extent.checked_mul(stride).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "{context}: address range overflows u64 for shape {shape:?}, strides {strides:?}, byte_offset {byte_offset} at axis {axis}; use a smaller validated view"
                ))
            })?;
            offset.checked_add(axis_extent).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "{context}: address range overflows u64 while summing shape {shape:?}, strides {strides:?}, byte_offset {byte_offset}; use a smaller validated view"
                ))
            })
        },
    )?;
    let span = max_element_offset
        .checked_mul(element_bytes)
        .and_then(|bytes| bytes.checked_add(element_bytes))
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "{context}: address range overflows u64 converting shape {shape:?} and strides {strides:?} to bytes for {dtype:?}; use a smaller validated view"
            ))
        })?;
    let end = start.checked_add(span).ok_or_else(|| {
        EpError::KernelFailed(format!(
            "{context}: address range overflows u64 from start {start:#x} with byte span {span}; use a view whose byte offset, shape, and strides fit device addressing"
        ))
    })?;
    Ok(Some(DeviceByteRange { start, end }))
}

fn overlaps(left: Option<DeviceByteRange>, right: Option<DeviceByteRange>) -> bool {
    matches!((left, right), (Some(left), Some(right)) if left.start < right.end && right.start < left.end)
}

impl EinsumKernel {
    fn validate_common(&self, inputs: &[TensorView], outputs: &[TensorMut]) -> Result<DataType> {
        if inputs.len() != self.input_shapes.len() || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: expected {} inputs and 1 output, got {} and {}",
                self.plan.equation(),
                self.input_shapes.len(),
                inputs.len(),
                outputs.len()
            )));
        }
        let dtype = inputs[0].dtype;
        einsum_dtype(dtype)?;
        for (index, (input, expected_shape)) in inputs.iter().zip(&self.input_shapes).enumerate() {
            if input.dtype != dtype {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: input #{index} dtype {:?} differs from input #0 dtype {dtype:?}",
                    self.plan.equation(),
                    input.dtype
                )));
            }
            if input.shape != expected_shape {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: input #{index} shape {:?} differs from the warmed shape {expected_shape:?}; request a shape-specialized kernel",
                    self.plan.equation(),
                    input.shape
                )));
            }
        }
        if outputs[0].dtype != dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: output dtype {:?} must equal input dtype {dtype:?}",
                self.plan.equation(),
                outputs[0].dtype
            )));
        }
        Ok(dtype)
    }

    fn compile_contraction(&self, dtype: DataType) -> Result<CachedExecution> {
        let start = Instant::now();
        let EinsumPlanningClassification::Gemm(contraction) = self.plan.planning_classification()
        else {
            unreachable!("compile_contraction is called only for GEMM plans");
        };
        if let Some(reason) = contraction_structure_reason(&self.plan, contraction) {
            return Err(not_implemented(reason));
        }
        let layout =
            concrete_contraction_layout(&self.plan, contraction, &self.input_shapes, dtype)?;
        let output_numel = checked_product(&layout.output_shape, "output")?;
        let kind = if output_numel == 0 {
            ExecutionKind::NoOp
        } else if layout.k == 0 {
            ExecutionKind::ZeroFill
        } else {
            let params = StridedBatchedGemmParams {
                dtype: einsum_dtype(dtype)?,
                a: 0,
                b: 0,
                c: 0,
                m: layout.m,
                k: layout.k,
                n: layout.n,
                batch: checked_product(&layout.batch_shape, "batch")?,
                transpose_a: layout.left_order == StorageOrder::Transposed,
                transpose_b: layout.right_order == StorageOrder::Transposed,
                a_batch_stride: layout.left_batch_stride,
                b_batch_stride: layout.right_batch_stride,
            };
            let plan = blas::plan_capture_strided_batched_gemm(self.runtime.blas(), &params)?;
            let workspace_bytes = plan.workspace_bytes();
            if workspace_bytes > WORKSPACE_BYTES {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: cuBLASLt selected {workspace_bytes} workspace bytes, above the {WORKSPACE_BYTES}-byte bound",
                    self.plan.equation()
                )));
            }
            let workspace = if workspace_bytes == 0 {
                None
            } else {
                Some(GraphDeviceAllocation::allocate(
                    &self.runtime,
                    workspace_bytes,
                )?)
            };
            WORKSPACE_BYTES_LAST.store(workspace_bytes as u64, Ordering::Relaxed);
            WORKSPACE_PTR_LAST.store(
                workspace.as_ref().map_or(0, |workspace| workspace.ptr()),
                Ordering::Relaxed,
            );
            ExecutionKind::Gemm(CachedGemm { plan, workspace })
        };
        SETUP_NS_LAST.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        PLAN_BUILDS.fetch_add(1, Ordering::Relaxed);
        Ok(CachedExecution {
            dtype,
            input_shapes: self.input_shapes.clone(),
            layout,
            kind,
        })
    }

    fn run_contraction(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let dtype = self.validate_common(inputs, outputs)?;
        if inputs.iter().any(|input| !input.is_contiguous()) || !outputs[0].is_contiguous() {
            return Err(not_implemented(format!(
                "Einsum `{}` GEMM/BMM contraction with non-contiguous input/output",
                self.plan.equation()
            )));
        }
        let output_context = format!(
            "cuda_ep Einsum `{}` contraction output",
            self.plan.equation()
        );
        let output_range = checked_nonnegative_strided_byte_range(
            cuptr(outputs[0].data.0 as *const c_void),
            outputs[0].byte_offset,
            outputs[0].dtype,
            outputs[0].shape,
            outputs[0].strides,
            &output_context,
        )?;
        for (index, input) in inputs.iter().enumerate() {
            let input_context = format!(
                "cuda_ep Einsum `{}` contraction input #{index}",
                self.plan.equation()
            );
            let input_range = checked_nonnegative_strided_byte_range(
                cuptr(input.data.0),
                input.byte_offset,
                input.dtype,
                input.shape,
                input.strides,
                &input_context,
            )?;
            if overlaps(output_range, input_range) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: contraction output byte range {output_range:?} overlaps input #{index} byte range {input_range:?}; use non-overlapping storage",
                    self.plan.equation()
                )));
            }
        }
        let capturing = self.runtime.is_capturing()?;
        let mut execution = self.execution.lock().map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: execution-plan lock was poisoned",
                self.plan.equation()
            ))
        })?;
        if execution
            .as_ref()
            .is_some_and(|cached| !cached.matches(inputs, &outputs[0]))
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: runtime dtype/shape/output changed after the immutable CUDA plan was built",
                self.plan.equation()
            )));
        }
        let compiled = if execution.is_none() {
            if capturing {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: exact dtype/shape/layout was not warmed before CUDA graph capture",
                    self.plan.equation()
                )));
            }
            Some(self.compile_contraction(dtype)?)
        } else {
            PLAN_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            None
        };
        let cached = execution
            .as_ref()
            .or(compiled.as_ref())
            .expect("an existing or staged contraction plan is present");
        if cached.layout.output_shape.as_slice() != outputs[0].shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: output shape {:?}, expected {:?}",
                self.plan.equation(),
                outputs[0].shape,
                cached.layout.output_shape
            )));
        }

        let a_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let b_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let c_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        match &cached.kind {
            ExecutionKind::NoOp => {}
            ExecutionKind::ZeroFill => {
                self.runtime.bind()?;
                // SAFETY: C covers the validated contiguous output byte length;
                // the memset is stream-ordered and capture-safe.
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        c_ptr,
                        0,
                        outputs[0].byte_size(),
                        self.runtime.stream_ptr(),
                    )
                }
                .map_err(|error| driver_err("zero-fill empty Einsum contraction", error))?;
                ZERO_FILL_LAUNCHES.fetch_add(1, Ordering::Relaxed);
            }
            ExecutionKind::Gemm(gemm) => {
                gemm.launch(&cached.layout, &self.runtime, a_ptr, b_ptr, c_ptr)?;
                GEMM_LAUNCHES.fetch_add(1, Ordering::Relaxed);
                if cached.layout.left_order == StorageOrder::Transposed
                    || cached.layout.right_order == StorageOrder::Transposed
                {
                    DESCRIPTOR_TRANSPOSE_GEMM_LAUNCHES.fetch_add(1, Ordering::Relaxed);
                } else {
                    CANONICAL_GEMM_LAUNCHES.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        if capturing {
            CAPTURE_RECORDINGS.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(compiled) = compiled {
            *execution = Some(compiled);
        }
        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn view_spec(
        &self,
        input: &TensorView,
        output_shape: &[usize],
        permutation: &EinsumPermutationPlan,
    ) -> Option<ViewOutput> {
        if input.dtype.byte_size() == 0 || permutation.input() != 0 {
            return None;
        }
        let shapes = [input.shape];
        let expected = self.plan.resolve_concrete_output_shape(&shapes).ok()?;
        if expected != output_shape {
            return None;
        }
        let operand = self.plan.operands().first()?;
        let mut strides = Vec::with_capacity(output_shape.len());
        for &unique_axis in permutation.output_to_operand_axis() {
            let axis = operand.unique_axes().get(unique_axis)?;
            let stride = axis.input_axes().iter().try_fold(0i64, |sum, &physical| {
                sum.checked_add(*input.strides.get(physical)?)
            })?;
            strides.push(stride);
        }
        Some(ViewOutput {
            input_index: 0,
            shape: output_shape.to_vec(),
            strides,
            byte_offset: input.byte_offset,
        })
    }

    fn run_view(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.validate_common(inputs, outputs)?;
        let permutation = match self.plan.planning_classification() {
            EinsumPlanningClassification::ViewOnlyPermutation(permutation)
            | EinsumPlanningClassification::DiagonalView(permutation) => permutation,
            _ => unreachable!("run_view is called only for view plans"),
        };
        let view = self
            .view_spec(&inputs[0], outputs[0].shape, permutation)
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: output is not the canonical permutation/diagonal view",
                    self.plan.equation()
                ))
            })?;
        if view.strides.iter().any(|&stride| stride < 0) {
            return Err(not_implemented(format!(
                "Einsum `{}` fallback materialization with negative source strides; execute through the zero-copy view path",
                self.plan.equation()
            )));
        }
        if !outputs[0].is_contiguous() {
            return Err(not_implemented(format!(
                "Einsum `{}` view fallback with a non-contiguous destination",
                self.plan.equation()
            )));
        }
        let input_context = format!(
            "cuda_ep Einsum `{}` materialized permutation/diagonal input",
            self.plan.equation()
        );
        let output_context = format!(
            "cuda_ep Einsum `{}` materialized permutation/diagonal output",
            self.plan.equation()
        );
        let input_range = checked_nonnegative_strided_byte_range(
            cuptr(inputs[0].data.0),
            inputs[0].byte_offset,
            inputs[0].dtype,
            inputs[0].shape,
            inputs[0].strides,
            &input_context,
        )?;
        let output_range = checked_nonnegative_strided_byte_range(
            cuptr(outputs[0].data.0 as *const c_void),
            outputs[0].byte_offset,
            outputs[0].dtype,
            outputs[0].shape,
            outputs[0].strides,
            &output_context,
        )?;
        if overlaps(input_range, output_range) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: materialized permutation/diagonal output byte range {output_range:?} overlaps input byte range {input_range:?}; use non-overlapping storage or execute through the zero-copy view path",
                self.plan.equation()
            )));
        }
        let capturing = self.runtime.is_capturing()?;
        let mut warmed = self.view_materialization.lock().map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: view-materialization lock was poisoned",
                self.plan.equation()
            ))
        })?;
        let candidate = if capturing {
            let signature = warmed.as_ref().ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: view materialization was not warmed before CUDA graph capture",
                    self.plan.equation()
                ))
            })?;
            if !signature.matches(&inputs[0], &outputs[0]) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: view materialization signature changed during CUDA graph capture",
                    self.plan.equation()
                )));
            }
            signature.clone()
        } else {
            let mut metadata = outputs[0]
                .shape
                .iter()
                .map(|&dim| dim as u64)
                .collect::<Vec<_>>();
            metadata.extend(view.strides.iter().map(|&stride| stride as u64));
            ViewMaterialization {
                dtype: inputs[0].dtype,
                input_shape: inputs[0].shape.to_vec(),
                input_strides: inputs[0].strides.to_vec(),
                output_shape: outputs[0].shape.to_vec(),
                metadata,
            }
        };
        if outputs[0].numel() == 0 {
            if !capturing {
                *warmed = Some(candidate);
            }
            self.last_call_capture_safe.store(true, Ordering::Relaxed);
            return Ok(());
        }
        let mut metadata = self.view_metadata.lock().map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: view metadata lock was poisoned",
                self.plan.equation()
            ))
        })?;
        let metadata_candidate = metadata.stage(&candidate.metadata, "Einsum view")?;
        let metadata_ptr = metadata_candidate.ptr("Einsum view")?;
        let metadata_bytes = metadata_candidate.allocation_bytes();
        launch_persistent_metadata(
            &self.runtime,
            "transpose_bytes",
            &inputs[0],
            &mut outputs[0],
            metadata_ptr,
        )?;
        VIEW_MATERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
        PERSISTENT_METADATA_BYTES_LAST.store(metadata_bytes as u64, Ordering::Relaxed);
        MATERIALIZATION_BYTES.fetch_add(outputs[0].byte_size() as u64, Ordering::Relaxed);
        if capturing {
            CAPTURE_RECORDINGS.fetch_add(1, Ordering::Relaxed);
        } else {
            *metadata = metadata_candidate;
            *warmed = Some(candidate);
        }
        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        Ok(())
    }
}

impl Kernel for EinsumKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        match self.plan.planning_classification() {
            EinsumPlanningClassification::ViewOnlyPermutation(_)
            | EinsumPlanningClassification::DiagonalView(_) => self.run_view(inputs, outputs),
            EinsumPlanningClassification::Gemm(_) => self.run_contraction(inputs, outputs),
            EinsumPlanningClassification::ContractionTree(tree) => Err(not_implemented(format!(
                "Einsum `{}` {}-input contraction plan execution with {}; CUDA must implement \
                 GenericNative and the planner's temporary schedule before executing this class",
                self.plan.equation(),
                tree.arity(),
                contraction_tree_summary(tree)
            ))),
            EinsumPlanningClassification::ReductionOrElementwise(_) => {
                Err(not_implemented(format!(
                    "Einsum `{}` reduction/elementwise canonical plan",
                    self.plan.equation()
                )))
            }
            _ => Err(not_implemented(format!(
                "Einsum `{}` newer unrecognized canonical classification; update CUDA claim, \
                 execution, and capture paths before running it",
                self.plan.equation()
            ))),
        }
    }

    fn view_outputs(
        &self,
        inputs: &[TensorView],
        output_shapes: &[Vec<usize>],
        num_outputs: usize,
    ) -> Option<Vec<ViewOutput>> {
        if inputs.len() != 1 || num_outputs != 1 || output_shapes.len() != 1 {
            return None;
        }
        einsum_dtype(inputs[0].dtype).ok()?;
        if inputs[0].shape != self.input_shapes[0] {
            return None;
        }
        let permutation = match self.plan.planning_classification() {
            EinsumPlanningClassification::ViewOnlyPermutation(permutation)
            | EinsumPlanningClassification::DiagonalView(permutation) => permutation,
            _ => return None,
        };
        let view = self.view_spec(&inputs[0], &output_shapes[0], permutation)?;
        self.view_alias_warmed.store(true, Ordering::Relaxed);
        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        VIEW_ALIASES.fetch_add(1, Ordering::Relaxed);
        Some(vec![view])
    }

    fn may_produce_views(&self) -> bool {
        matches!(
            self.plan.planning_classification(),
            EinsumPlanningClassification::ViewOnlyPermutation(_)
                | EinsumPlanningClassification::DiagonalView(_)
        )
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx == 0
            && matches!(
                self.plan.planning_classification(),
                EinsumPlanningClassification::ViewOnlyPermutation(_)
                    | EinsumPlanningClassification::DiagonalView(_)
            )
    }

    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        let mut resources = Vec::with_capacity(2);
        if let Ok(execution) = self.execution.lock()
            && let Some(CachedExecution {
                kind: ExecutionKind::Gemm(gemm),
                ..
            }) = execution.as_ref()
            && let Some(workspace) = gemm.workspace.as_ref()
        {
            resources.push(GraphDeviceAllocation::device_graph_resource(workspace));
        }
        if let Ok(metadata) = self.view_metadata.lock()
            && let Some(resource) = metadata.device_graph_resource()
        {
            resources.push(resource);
        }
        resources
    }

    fn capture_support(&self) -> CaptureSupport {
        match self.plan.planning_classification() {
            EinsumPlanningClassification::ViewOnlyPermutation(_)
            | EinsumPlanningClassification::DiagonalView(_) => {
                match self.view_materialization.lock() {
                    Ok(signature) => {
                        let materialization_warmed = signature.is_some();
                        let metadata_required = signature
                            .as_ref()
                            .is_some_and(|signature| !signature.output_shape.contains(&0));
                        let metadata_ready = !metadata_required
                            || self
                                .view_metadata
                                .lock()
                                .is_ok_and(|metadata| metadata.device_graph_resource().is_some());
                        if (self.view_alias_warmed.load(Ordering::Relaxed)
                            || materialization_warmed)
                            && metadata_ready
                        {
                            CaptureSupport::Supported
                        } else {
                            CaptureSupport::unsupported(format!(
                                "Einsum `{}` must establish its zero-copy view or exact materialization signature and persistent metadata before capture",
                                self.plan.equation()
                            ))
                        }
                    }
                    Err(_) => CaptureSupport::unsupported(format!(
                        "Einsum `{}` view-materialization lock was poisoned",
                        self.plan.equation()
                    )),
                }
            }
            EinsumPlanningClassification::Gemm(_) => match self.execution.lock() {
                Ok(execution)
                    if execution.is_some()
                        && self.last_call_capture_safe.load(Ordering::Relaxed) =>
                {
                    CaptureSupport::Supported
                }
                Ok(_) => CaptureSupport::unsupported(format!(
                    "Einsum `{}` must warm its exact dtype/shape/layout and persistent cuBLASLt workspace before capture",
                    self.plan.equation()
                )),
                Err(_) => CaptureSupport::unsupported(format!(
                    "Einsum `{}` execution-plan lock was poisoned",
                    self.plan.equation()
                )),
            },
            EinsumPlanningClassification::ContractionTree(_) => CaptureSupport::unsupported(
                "CUDA Einsum contraction-tree temporary scheduling and multi-node capture are not implemented",
            ),
            EinsumPlanningClassification::ReductionOrElementwise(_) => CaptureSupport::unsupported(
                "CUDA Einsum reduction/elementwise lowering is not implemented",
            ),
            _ => CaptureSupport::unsupported(
                "CUDA Einsum received a newer unrecognized canonical classification",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(equation: &str, shapes: &[&[usize]]) -> EinsumPlan {
        let inputs = shapes
            .iter()
            .map(|shape| EinsumInput::new(DataType::Float32, shape))
            .collect::<Vec<_>>();
        EinsumPlan::build(equation, &inputs).unwrap()
    }

    #[test]
    fn multilinear_tree_is_declined_before_cuda_execution() {
        let mut node = Node::new(onnx_runtime_ir::NodeId(0), "Einsum", vec![], vec![]);
        node.attributes.insert(
            "equation".into(),
            onnx_runtime_ir::Attribute::String(b"i,ij,j->".to_vec()),
        );
        let shapes = [
            onnx_runtime_ir::static_shape([2]),
            onnx_runtime_ir::static_shape([2, 8]),
            onnx_runtime_ir::static_shape([8]),
        ];
        let reason = unsupported_reason_impl(
            &node,
            12,
            &shapes,
            &[DataType::Float32; 3],
            &[
                TensorLayout::contiguous(),
                TensorLayout::contiguous(),
                TensorLayout::contiguous(),
            ],
        )
        .unwrap();
        assert!(reason.contains("3-input contraction plan"));
        assert!(reason.contains("temporary scheduling"));
    }

    #[test]
    fn large_arity_cuda_claim_reports_bounded_generic_fallback() {
        let arity = 256;
        let equation = format!(
            "{}->",
            std::iter::repeat_n("i", arity)
                .collect::<Vec<_>>()
                .join(",")
        );
        let mut node = Node::new(onnx_runtime_ir::NodeId(0), "Einsum", vec![], vec![]);
        node.attributes.insert(
            "equation".into(),
            onnx_runtime_ir::Attribute::String(equation.into_bytes()),
        );
        let shapes = vec![onnx_runtime_ir::static_shape([1]); arity];
        let dtypes = vec![DataType::Float32; arity];
        let layouts = vec![TensorLayout::contiguous(); arity];
        let reason = unsupported_reason_impl(&node, 12, &shapes, &dtypes, &layouts).unwrap();
        assert!(reason.contains("GenericNative fallback"), "{reason}");
        assert!(
            reason.contains("work/metadata budget was exceeded"),
            "{reason}"
        );
    }

    #[test]
    fn cuda_claim_resolves_schema_before_staged_backend_dtype_support() {
        let mut node = Node::new(onnx_runtime_ir::NodeId(0), "Einsum", vec![], vec![]);
        node.attributes.insert(
            "equation".into(),
            onnx_runtime_ir::Attribute::String(b"i->i".to_vec()),
        );
        let shapes = [onnx_runtime_ir::static_shape([2])];
        let layouts = [TensorLayout::contiguous()];

        let opset11 =
            unsupported_reason_impl(&node, 11, &shapes, &[DataType::Float32], &layouts).unwrap();
        assert!(opset11.contains("predates Einsum-12"), "{opset11}");

        let opset27 =
            unsupported_reason_impl(&node, 27, &shapes, &[DataType::BFloat16], &layouts).unwrap();
        assert!(opset27.contains("not admitted by Einsum-12"), "{opset27}");

        let opset28 =
            unsupported_reason_impl(&node, 28, &shapes, &[DataType::BFloat16], &layouts).unwrap();
        assert!(opset28.contains("Einsum-28 admits BFloat16 semantically"));
    }

    #[test]
    fn layout_accepts_descriptor_transposes_without_materialization() {
        for (equation, shapes, left, right) in [
            ("ik,kj->ij", [&[2, 3][..], &[3, 4][..]], false, false),
            ("ki,kj->ij", [&[3, 2][..], &[3, 4][..]], true, false),
            ("ik,jk->ij", [&[2, 3][..], &[4, 3][..]], false, true),
            ("ki,jk->ij", [&[3, 2][..], &[4, 3][..]], true, true),
        ] {
            let plan = plan(equation, &shapes);
            let EinsumPlanningClassification::Gemm(contraction) = plan.planning_classification()
            else {
                panic!("{equation} was not GEMM");
            };
            let concrete = shapes
                .iter()
                .map(|shape| shape.to_vec())
                .collect::<Vec<_>>();
            let layout = concrete_contraction_layout(
                plan.shape_plan(),
                contraction,
                &concrete,
                DataType::Float32,
            )
            .unwrap();
            assert_eq!(
                layout.left_order == StorageOrder::Transposed,
                left,
                "{equation}"
            );
            assert_eq!(
                layout.right_order == StorageOrder::Transposed,
                right,
                "{equation}"
            );
        }
    }

    #[test]
    fn layout_admits_whole_batch_stride_zero_and_rejects_partial_broadcast() {
        let broadcast = plan("mk,...kn->...mn", &[&[2, 3], &[6, 5, 3, 4]]);
        let EinsumPlanningClassification::Gemm(contraction) = broadcast.planning_classification()
        else {
            panic!("expected GEMM");
        };
        let layout = concrete_contraction_layout(
            broadcast.shape_plan(),
            contraction,
            &[vec![2, 3], vec![6, 5, 3, 4]],
            DataType::Float32,
        )
        .unwrap();
        assert_eq!(layout.left_batch_stride, 0);
        assert_eq!(layout.right_batch_stride, 12);

        let partial = plan("...mk,...kn->...mn", &[&[2, 1, 3, 4], &[2, 5, 4, 6]]);
        let EinsumPlanningClassification::Gemm(contraction) = partial.planning_classification()
        else {
            panic!("expected GEMM");
        };
        let error = concrete_contraction_layout(
            partial.shape_plan(),
            contraction,
            &[vec![2, 1, 3, 4], vec![2, 5, 4, 6]],
            DataType::Float32,
        )
        .unwrap_err();
        assert!(error.to_string().contains("partial multi-axis batch"));
    }

    #[test]
    fn addressed_byte_range_accounts_for_offsets_strides_and_empty_tensors() {
        let range = checked_nonnegative_strided_byte_range(
            0x1000,
            8,
            DataType::Float32,
            &[2, 3],
            &[5, 1],
            "test input",
        )
        .unwrap();
        assert_eq!(
            range,
            Some(DeviceByteRange {
                start: 0x1008,
                end: 0x1028
            })
        );
        assert_eq!(
            checked_nonnegative_strided_byte_range(
                u64::MAX,
                usize::MAX,
                DataType::Float32,
                &[0, usize::MAX],
                &[i64::MAX, i64::MAX],
                "empty test input",
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn addressed_byte_range_rejects_pointer_overflow_actionably() {
        let error = checked_nonnegative_strided_byte_range(
            u64::MAX - 1,
            4,
            DataType::Float32,
            &[1],
            &[1],
            "test input",
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("test input"));
        assert!(message.contains("address range overflows u64"));
        assert!(message.contains("byte_offset 4"));
    }
}

//! Capture-safe CUDA execution for canonical ONNX `Einsum` plans.
//!
//! The equation is parsed exactly once by [`EinsumPlan`] when the
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
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView, ViewOutput,
};
use onnx_runtime_ir::{
    DataType, EinsumClassification, EinsumContractionPlan, EinsumInput, EinsumOperandPlan,
    EinsumPermutationPlan, EinsumPlan, Node, Shape, TensorLayout,
};

use super::movement::{PersistentMetadata, launch_persistent_metadata};
use crate::blas::{
    self, CaptureStridedBatchedGemmPlan, GemmDtype, StridedBatchedGemmParams, WORKSPACE_BYTES,
};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// Counters proving which CUDA Einsum route executed and what persistent state
/// it established. Values are process-global diagnostics, so GPU tests serialize
/// around reset/read windows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EinsumExecutionStats {
    pub plan_builds: u64,
    pub plan_cache_hits: u64,
    pub view_aliases: u64,
    pub view_materializations: u64,
    pub gemm_launches: u64,
    pub zero_fill_launches: u64,
    pub capture_recordings: u64,
    pub workspace_bytes: u64,
    pub workspace_ptr: u64,
    pub setup_ns: u64,
    pub materialization_bytes: u64,
}

static PLAN_BUILDS: AtomicU64 = AtomicU64::new(0);
static PLAN_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static VIEW_ALIASES: AtomicU64 = AtomicU64::new(0);
static VIEW_MATERIALIZATIONS: AtomicU64 = AtomicU64::new(0);
static GEMM_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static ZERO_FILL_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static CAPTURE_RECORDINGS: AtomicU64 = AtomicU64::new(0);
static WORKSPACE_BYTES_LAST: AtomicU64 = AtomicU64::new(0);
static WORKSPACE_PTR_LAST: AtomicU64 = AtomicU64::new(0);
static SETUP_NS_LAST: AtomicU64 = AtomicU64::new(0);
static MATERIALIZATION_BYTES: AtomicU64 = AtomicU64::new(0);

pub fn einsum_execution_stats() -> EinsumExecutionStats {
    EinsumExecutionStats {
        plan_builds: PLAN_BUILDS.load(Ordering::Relaxed),
        plan_cache_hits: PLAN_CACHE_HITS.load(Ordering::Relaxed),
        view_aliases: VIEW_ALIASES.load(Ordering::Relaxed),
        view_materializations: VIEW_MATERIALIZATIONS.load(Ordering::Relaxed),
        gemm_launches: GEMM_LAUNCHES.load(Ordering::Relaxed),
        zero_fill_launches: ZERO_FILL_LAUNCHES.load(Ordering::Relaxed),
        capture_recordings: CAPTURE_RECORDINGS.load(Ordering::Relaxed),
        workspace_bytes: WORKSPACE_BYTES_LAST.load(Ordering::Relaxed),
        workspace_ptr: WORKSPACE_PTR_LAST.load(Ordering::Relaxed),
        setup_ns: SETUP_NS_LAST.load(Ordering::Relaxed),
        materialization_bytes: MATERIALIZATION_BYTES.load(Ordering::Relaxed),
    }
}

pub fn reset_einsum_execution_stats() {
    PLAN_BUILDS.store(0, Ordering::Relaxed);
    PLAN_CACHE_HITS.store(0, Ordering::Relaxed);
    VIEW_ALIASES.store(0, Ordering::Relaxed);
    VIEW_MATERIALIZATIONS.store(0, Ordering::Relaxed);
    GEMM_LAUNCHES.store(0, Ordering::Relaxed);
    ZERO_FILL_LAUNCHES.store(0, Ordering::Relaxed);
    CAPTURE_RECORDINGS.store(0, Ordering::Relaxed);
    WORKSPACE_BYTES_LAST.store(0, Ordering::Relaxed);
    WORKSPACE_PTR_LAST.store(0, Ordering::Relaxed);
    SETUP_NS_LAST.store(0, Ordering::Relaxed);
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
            "Einsum opset-12 dtype {other:?}; the ONNX schema and native CUDA lowering support Float32 and Float16 (BFloat16 is not an opset-12 Einsum type)"
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
    plan: &EinsumPlan,
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
pub fn unsupported_reason(
    node: &Node,
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
    if let Some(dtype) = input_dtypes.first().copied()
        && let Err(error) = einsum_dtype(dtype)
    {
        return Some(error.to_string());
    }
    let inputs = shapes
        .iter()
        .zip(input_dtypes)
        .map(|(shape, &dtype)| EinsumInput::new(dtype, shape.as_slice()))
        .collect::<Vec<_>>();
    let plan = match EinsumPlan::build(equation, &inputs) {
        Ok(plan) => plan,
        Err(error) => return Some(format!("cuda_ep Einsum `{equation}`: {error}")),
    };

    match plan.classification() {
        EinsumClassification::ViewOnlyPermutation(_) | EinsumClassification::DiagonalView(_) => {
            None
        }
        EinsumClassification::Gemm(contraction) => {
            if let Some(reason) = contraction_structure_reason(&plan, contraction) {
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
                && let Err(error) = concrete_contraction_layout(&plan, contraction, &concrete)
            {
                return Some(error.to_string());
            }
            None
        }
        EinsumClassification::ReductionOrElementwise(_) => Some(format!(
            "cuda_ep Einsum `{equation}`: uncoupled reductions/elementwise products are not yet lowered; use native Reduce*/Mul nodes or CPU fallback"
        )),
        EinsumClassification::Unsupported(reason) => Some(format!(
            "cuda_ep Einsum `{equation}`: canonical planner classified this contraction as unsupported: {reason}"
        )),
    }
}

pub struct EinsumFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for EinsumFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let equation = equation(node)?.to_owned();
        // Kernel-cache entries are shape-specialized. Build the immutable
        // structural plan once here; Float32 supplies a supported placeholder
        // type because the factory API carries shapes but not value dtypes.
        // Runtime execution separately validates the real common dtype.
        let inputs = input_shapes
            .iter()
            .map(|shape| EinsumInput::new(DataType::Float32, shape.as_slice()))
            .collect::<Vec<_>>();
        let plan = EinsumPlan::build(&equation, &inputs).map_err(|error| {
            EpError::KernelFailed(format!("cuda_ep Einsum `{equation}`: {error}"))
        })?;
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
    plan: &EinsumPlan,
    contraction: &EinsumContractionPlan,
    input_shapes: &[Vec<usize>],
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
        dtype: plan.dtype(),
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
    runtime: Arc<CudaRuntime>,
    plan: CaptureStridedBatchedGemmPlan,
    workspace: CUdeviceptr,
}

impl Drop for CachedGemm {
    fn drop(&mut self) {
        if self.workspace != 0 {
            // SAFETY: allocated once when the immutable plan was built and
            // freed after the owning kernel/captured graph can no longer launch.
            unsafe {
                let _ = self.runtime.free_raw(self.workspace);
            }
        }
    }
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
                self.workspace,
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
    plan: EinsumPlan,
    execution: Mutex<Option<CachedExecution>>,
    view_metadata: Mutex<PersistentMetadata>,
    view_materialization: Mutex<Option<ViewMaterialization>>,
    view_alias_warmed: AtomicBool,
    last_call_capture_safe: AtomicBool,
}

fn pointer_range(ptr: CUdeviceptr, bytes: usize) -> Result<Option<(u64, u64)>> {
    if bytes == 0 {
        return Ok(None);
    }
    let end = ptr.checked_add(bytes as u64).ok_or_else(|| {
        EpError::KernelFailed("cuda_ep Einsum: device pointer range overflows u64".into())
    })?;
    Ok(Some((ptr, end)))
}

fn overlaps(left: Option<(u64, u64)>, right: Option<(u64, u64)>) -> bool {
    matches!((left, right), (Some((ls, le)), Some((rs, re))) if ls < re && rs < le)
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
        let EinsumClassification::Gemm(contraction) = self.plan.classification() else {
            unreachable!("compile_contraction is called only for GEMM plans");
        };
        if let Some(reason) = contraction_structure_reason(&self.plan, contraction) {
            return Err(not_implemented(reason));
        }
        let mut layout = concrete_contraction_layout(&self.plan, contraction, &self.input_shapes)?;
        layout.dtype = dtype;
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
                0
            } else {
                self.runtime.alloc_raw(workspace_bytes)?
            };
            WORKSPACE_BYTES_LAST.store(workspace_bytes as u64, Ordering::Relaxed);
            WORKSPACE_PTR_LAST.store(workspace, Ordering::Relaxed);
            ExecutionKind::Gemm(CachedGemm {
                runtime: self.runtime.clone(),
                plan,
                workspace,
            })
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
        if execution.is_none() {
            if capturing {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: exact dtype/shape/layout was not warmed before CUDA graph capture",
                    self.plan.equation()
                )));
            }
            *execution = Some(self.compile_contraction(dtype)?);
        } else {
            PLAN_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        }
        let cached = execution.as_ref().unwrap();
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
        let output_range = pointer_range(c_ptr, outputs[0].byte_size())?;
        if overlaps(output_range, pointer_range(a_ptr, inputs[0].byte_size())?)
            || overlaps(output_range, pointer_range(b_ptr, inputs[1].byte_size())?)
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: output must not alias either contraction input",
                self.plan.equation()
            )));
        }

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
            }
        }
        if capturing {
            CAPTURE_RECORDINGS.fetch_add(1, Ordering::Relaxed);
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
        let permutation = match self.plan.classification() {
            EinsumClassification::ViewOnlyPermutation(permutation)
            | EinsumClassification::DiagonalView(permutation) => permutation,
            _ => unreachable!("run_view is called only for view plans"),
        };
        let capturing = self.runtime.is_capturing()?;
        let mut warmed = self.view_materialization.lock().map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: view-materialization lock was poisoned",
                self.plan.equation()
            ))
        })?;
        if capturing {
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
        } else {
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
            let mut metadata = outputs[0]
                .shape
                .iter()
                .map(|&dim| dim as u64)
                .collect::<Vec<_>>();
            metadata.extend(view.strides.iter().map(|&stride| stride as u64));
            *warmed = Some(ViewMaterialization {
                dtype: inputs[0].dtype,
                input_shape: inputs[0].shape.to_vec(),
                input_strides: inputs[0].strides.to_vec(),
                output_shape: outputs[0].shape.to_vec(),
                metadata,
            });
        }
        if outputs[0].numel() == 0 {
            self.last_call_capture_safe.store(true, Ordering::Relaxed);
            return Ok(());
        }
        if !outputs[0].is_contiguous() {
            return Err(not_implemented(format!(
                "Einsum `{}` view fallback with a non-contiguous destination",
                self.plan.equation()
            )));
        }
        let input_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        if input_ptr == output_ptr {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Einsum `{}`: materialized permutation/diagonal output aliases its input; execute through the zero-copy view path",
                self.plan.equation()
            )));
        }
        let metadata_ptr = self
            .view_metadata
            .lock()
            .map_err(|_| {
                EpError::KernelFailed(format!(
                    "cuda_ep Einsum `{}`: view metadata lock was poisoned",
                    self.plan.equation()
                ))
            })?
            .prepare(&warmed.as_ref().unwrap().metadata, "Einsum view")?;
        launch_persistent_metadata(
            &self.runtime,
            "transpose_bytes",
            &inputs[0],
            &mut outputs[0],
            metadata_ptr,
        )?;
        VIEW_MATERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
        MATERIALIZATION_BYTES.fetch_add(outputs[0].byte_size() as u64, Ordering::Relaxed);
        if capturing {
            CAPTURE_RECORDINGS.fetch_add(1, Ordering::Relaxed);
        }
        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        Ok(())
    }
}

impl Kernel for EinsumKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        match self.plan.classification() {
            EinsumClassification::ViewOnlyPermutation(_)
            | EinsumClassification::DiagonalView(_) => self.run_view(inputs, outputs),
            EinsumClassification::Gemm(_) => self.run_contraction(inputs, outputs),
            EinsumClassification::ReductionOrElementwise(_) => Err(not_implemented(format!(
                "Einsum `{}` reduction/elementwise canonical plan",
                self.plan.equation()
            ))),
            EinsumClassification::Unsupported(reason) => Err(not_implemented(format!(
                "Einsum `{}` canonical plan: {reason}",
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
        let permutation = match self.plan.classification() {
            EinsumClassification::ViewOnlyPermutation(permutation)
            | EinsumClassification::DiagonalView(permutation) => permutation,
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
            self.plan.classification(),
            EinsumClassification::ViewOnlyPermutation(_) | EinsumClassification::DiagonalView(_)
        )
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx == 0
            && matches!(
                self.plan.classification(),
                EinsumClassification::ViewOnlyPermutation(_)
                    | EinsumClassification::DiagonalView(_)
            )
    }

    fn capture_support(&self) -> CaptureSupport {
        match self.plan.classification() {
            EinsumClassification::ViewOnlyPermutation(_)
            | EinsumClassification::DiagonalView(_) => {
                let materialization_warmed = self
                    .view_materialization
                    .lock()
                    .is_ok_and(|signature| signature.is_some());
                if self.view_alias_warmed.load(Ordering::Relaxed) || materialization_warmed {
                    CaptureSupport::Supported
                } else {
                    CaptureSupport::unsupported(format!(
                        "Einsum `{}` must establish its zero-copy view or exact materialization signature before capture",
                        self.plan.equation()
                    ))
                }
            }
            EinsumClassification::Gemm(_) => {
                if self.last_call_capture_safe.load(Ordering::Relaxed) {
                    CaptureSupport::Supported
                } else {
                    CaptureSupport::unsupported(format!(
                        "Einsum `{}` must warm its exact dtype/shape/layout and persistent cuBLASLt workspace before capture",
                        self.plan.equation()
                    ))
                }
            }
            EinsumClassification::ReductionOrElementwise(_) => CaptureSupport::unsupported(
                "CUDA Einsum reduction/elementwise lowering is not implemented",
            ),
            EinsumClassification::Unsupported(reason) => {
                CaptureSupport::unsupported(format!("unsupported canonical Einsum plan: {reason}"))
            }
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
    fn layout_accepts_descriptor_transposes_without_materialization() {
        for (equation, shapes, left, right) in [
            ("ik,kj->ij", [&[2, 3][..], &[3, 4][..]], false, false),
            ("ki,kj->ij", [&[3, 2][..], &[3, 4][..]], true, false),
            ("ik,jk->ij", [&[2, 3][..], &[4, 3][..]], false, true),
            ("ki,jk->ij", [&[3, 2][..], &[4, 3][..]], true, true),
        ] {
            let plan = plan(equation, &shapes);
            let EinsumClassification::Gemm(contraction) = plan.classification() else {
                panic!("{equation} was not GEMM");
            };
            let concrete = shapes
                .iter()
                .map(|shape| shape.to_vec())
                .collect::<Vec<_>>();
            let layout = concrete_contraction_layout(&plan, contraction, &concrete).unwrap();
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
        let EinsumClassification::Gemm(contraction) = broadcast.classification() else {
            panic!("expected GEMM");
        };
        let layout =
            concrete_contraction_layout(&broadcast, contraction, &[vec![2, 3], vec![6, 5, 3, 4]])
                .unwrap();
        assert_eq!(layout.left_batch_stride, 0);
        assert_eq!(layout.right_batch_stride, 12);

        let partial = plan("...mk,...kn->...mn", &[&[2, 1, 3, 4], &[2, 5, 4, 6]]);
        let EinsumClassification::Gemm(contraction) = partial.classification() else {
            panic!("expected GEMM");
        };
        let error = concrete_contraction_layout(
            &partial,
            contraction,
            &[vec![2, 1, 3, 4], vec![2, 5, 4, 6]],
        )
        .unwrap_err();
        assert!(error.to_string().contains("partial multi-axis batch"));
    }
}

//! `MatMul` on the GPU via cuBLASLt (`docs/architecture/ORT2.md` §15.3).
//!
//! Supports dense rank >= 2 operands with NumPy/ONNX broadcasting across all
//! leading batch dimensions for f32 / f16 / bf16, all in true fp32
//! accumulation. Broadcast runs are expressed as cuBLASLt strided batches,
//! including stride-zero operands. The row-major → column-major mapping lives
//! in [`crate::blas`].
//!
//! ## Limits (all reported as actionable errors, never panics)
//!
//! * rank-1 operand promotion is not implemented yet
//! * non-contiguous (strided) device inputs are not implemented yet
//! * dtypes other than f32 / f16 / bf16 are not implemented yet
//! * mismatched inner dims / dtypes → a plain kernel error (a real mistake, not
//!   a missing feature)

use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView, WorkspaceLayout,
    WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node};

use crate::blas::{self, GemmDtype, GemmParams};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// NVRTC module/entry for the dense decode GEMVs.
const GEMV_F16_MODULE: &str = "matmul_dense_gemv_f16";
const GEMV_F16_ENTRY: &str = "matmul_dense_gemv_f16";
/// Upper bound on the number of distinct dense `(dtype, M, K, N)` cuBLASLt plans
/// a single `MatMul` node keeps warm simultaneously. A node realistically sees a
/// small, fixed set — decode `M==1`, a handful of prefill widths, and any
/// speculative verify width — so 8 covers the hot working set while capping the
/// per-node host-plan footprint. Eviction is LRU and follows successful eager
/// creation; capture can only reuse already-warmed host plans.
const DENSE_PLAN_CACHE_CAP: usize = 8;
/// Threads per block for the dense fp16 GEMV. One thread owns one output
/// column, so a warp reads 32 consecutive `B[k, col]` fp16 values — a fully
/// coalesced 64-byte transaction per step. 256 gives good occupancy without
/// oversubscribing shared memory.
const GEMV_F16_THREADS: u32 = 256;

/// Bandwidth-bound dense fp16 GEMV `y[1, N] = a[1, K] * B[K, N]` for the M==1
/// decode step (e.g. an fp16 language-model head).
///
/// Kernel shape: one thread per output column `col`; a block of
/// [`GEMV_F16_THREADS`] threads cooperatively stages `blockDim.x` activation
/// elements into shared memory per K-tile, then every thread reads its column's
/// `B[k, col]` fp16 weight straight from global memory. Consecutive threads read
/// consecutive `col`, so each warp issues one coalesced load, giving a single
/// streaming pass over `B` at ≈ HBM roofline. Accumulation is fp32 (matching the
/// cuBLASLt path's true-fp32 accumulate) and the result is rounded to fp16 once.
/// The tiled activation staging bounds shared memory to `blockDim.x` floats for
/// any `K`, and the `col < n` guard makes any `N` safe — no magic dimensions, so
/// this fires for every dense fp16 M==1 MatMul regardless of model.
const GEMV_F16_SRC: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void matmul_dense_gemv_f16(
    const __half* __restrict__ a,   // [K]
    const __half* __restrict__ b,   // [K, N] row-major
    __half* __restrict__ y,         // [N]
    const int k,
    const int n)
{
    extern __shared__ float a_tile[];   // blockDim.x floats
    const int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    float acc = 0.0f;
    for (int k0 = 0; k0 < k; k0 += (int)blockDim.x) {
        const int kk = k0 + (int)threadIdx.x;
        a_tile[threadIdx.x] = (kk < k) ? __half2float(a[kk]) : 0.0f;
        __syncthreads();
        const int tile = min((int)blockDim.x, k - k0);
        if (col < n) {
            for (int j = 0; j < tile; ++j) {
                acc += a_tile[j] * __half2float(b[(long)(k0 + j) * n + col]);
            }
        }
        __syncthreads();
    }
    if (col < n) {
        y[col] = __float2half(acc);
    }
}
"#;

/// Factory for [`MatMulKernel`]; carries the shared CUDA runtime.
pub struct MatMulFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for MatMulFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(MatMulKernel {
            runtime: self.runtime.clone(),
            warm_state: Mutex::new(MatMulWarmState {
                f32_gemv: None,
                dense_plans: Vec::new(),
                capture_ready: None,
                prepared: None,
            }),
        }))
    }
}

/// cuBLASLt-backed f32/f16/bf16 MatMul kernel with capturable dense f32/fp16
/// GEMV fast paths for the M==1 decode step.
pub struct MatMulKernel {
    runtime: Arc<CudaRuntime>,
    warm_state: Mutex<MatMulWarmState>,
}

struct MatMulWarmState {
    /// Pure host cuBLASLt plan preselected during the f32 M==1 warmup.
    /// Reusing the exact algorithm preserves bitwise parity with the old path
    /// while eliminating all capture-time setup and device allocation.
    f32_gemv: Option<CachedGemmPlan>,
    /// Host cuBLASLt plans keyed by the 2-D (`batch == 1`)
    /// dense shape and the selected algorithm's pointer-alignment contract.
    /// Each compatible shape the node runs keeps its own preselected algorithm,
    /// so alternating prefill/decode/verify shapes replay without a per-call
    /// heuristic query while a weaker-aligned subview is replanned safely.
    /// MRU-ordered (front = most recent); bounded by [`DENSE_PLAN_CACHE_CAP`].
    dense_plans: Vec<CachedGemmPlan>,
    /// Signature of the most recent successful capture-safe call.
    capture_ready: Option<MatMulCaptureSignature>,
    /// One exclusive refine/admit/launch transaction; candidates own only host state.
    prepared: Option<PreparedDispatch>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MatMulCaptureSignature {
    dtype: GemmDtype,
    route: MatMulCaptureRoute,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    output_shape: Vec<usize>,
    m: usize,
    k: usize,
    n: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatMulCaptureRoute {
    F16HandGemv,
    F32Gemv,
    CublasLt,
}

/// Cache entries contain algorithms and host descriptors, never device scratch.
struct CachedGemmPlan {
    dtype: GemmDtype,
    m: usize,
    k: usize,
    n: usize,
    algorithm: DenseAlgorithm,
}

enum DenseAlgorithm {
    CublasLt(blas::CaptureGemmPlan),
    F16HandGemv,
}

/// The selected plan is fixed before admission. A cold candidate enters the
/// cache only after a successful launch; quota errors cannot select a fallback.
enum PreparedPlan {
    F32Slot,
    Dense(usize),
    Candidate(CachedGemmPlan),
    F16HandGemv,
}

struct PreparedDispatch {
    signature: MatMulCaptureSignature,
    plan: PreparedPlan,
    warmed: Option<MatMulCaptureSignature>,
}

impl PreparedPlan {
    fn cached<'a>(&'a self, state: &'a MatMulWarmState) -> Result<Option<&'a CachedGemmPlan>> {
        let entry = match self {
            Self::F32Slot => state.f32_gemv.as_ref(),
            Self::Dense(index) => state.dense_plans.get(*index),
            Self::Candidate(plan) => Some(plan),
            Self::F16HandGemv => return Ok(None),
        };
        entry.map(Some).ok_or_else(|| EpError::KernelFailed("cuda_ep MatMul: the admitted host-plan cache entry disappeared; retain exclusive kernel execution ownership from refinement through admission and launch".into()))
    }
}

impl CachedGemmPlan {
    fn matches(&self, p: &GemmParams) -> bool {
        self.dtype == p.dtype
            && (self.m, self.k, self.n) == (p.m, p.k, p.n)
            && match &self.algorithm {
                DenseAlgorithm::CublasLt(plan) => plan.supports(p),
                DenseAlgorithm::F16HandGemv => true,
            }
    }

    fn new(runtime: &CudaRuntime, params: &GemmParams) -> Result<Self> {
        let algorithm = match blas::plan_capture_gemm(runtime.blas(), params) {
            Ok(plan) => DenseAlgorithm::CublasLt(plan),
            // Only planning can choose the zero-scratch fallback. Allocation or
            // launch failures must propagate, never bypass governed admission.
            Err(_) if params.dtype == GemmDtype::F16 && params.m == 1 => {
                DenseAlgorithm::F16HandGemv
            }
            Err(error) => return Err(error),
        };
        Ok(Self {
            dtype: params.dtype,
            m: params.m,
            k: params.k,
            n: params.n,
            algorithm,
        })
    }

    fn workspace_bytes(&self) -> usize {
        match &self.algorithm {
            DenseAlgorithm::CublasLt(plan) => plan.workspace_bytes(),
            DenseAlgorithm::F16HandGemv => 0,
        }
    }

    fn route(&self) -> MatMulCaptureRoute {
        match &self.algorithm {
            DenseAlgorithm::F16HandGemv => MatMulCaptureRoute::F16HandGemv,
            DenseAlgorithm::CublasLt(_) if self.dtype == GemmDtype::F32 && self.m == 1 => {
                MatMulCaptureRoute::F32Gemv
            }
            DenseAlgorithm::CublasLt(_) => MatMulCaptureRoute::CublasLt,
        }
    }
}

fn dense_workspace(bytes: usize, workspace: Option<WorkspaceView>) -> Result<WorkspaceView> {
    let region = std::alloc::Layout::from_size_align(bytes, blas::WORKSPACE_ALIGNMENT)
        .map_err(|_| EpError::KernelFailed(format!("cuda_ep MatMul: cuBLASLt scratch {bytes} bytes exceeds addressable capacity; reduce the GEMM shape")))?;
    let [used] = WorkspaceLayout::new([region])?.bind(workspace)?;
    Ok(used)
}

/// Build [`GemmParams`] for a plain 2-D (`batch == 1`) dense GEMM.
fn dense_gemm_params(
    dtype: GemmDtype,
    m: usize,
    k: usize,
    n: usize,
    a: u64,
    b: u64,
    c: u64,
) -> GemmParams {
    GemmParams {
        dtype,
        a,
        b,
        c,
        m,
        k,
        n,
        batch: 1,
        a_batch_stride: 0,
        b_batch_stride: 0,
        epilogue: None,
    }
}

/// Map an ONNX element type to a cuBLASLt GEMM dtype.
fn gemm_dtype(dt: DataType) -> Result<GemmDtype> {
    match dt {
        DataType::Float32 => Ok(GemmDtype::F32),
        DataType::Float16 => Ok(GemmDtype::F16),
        DataType::BFloat16 => Ok(GemmDtype::Bf16),
        other => Err(not_implemented(format!("MatMul with dtype {other:?}"))),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct MatMulPlan {
    batch_shape: Vec<usize>,
    a_batch_strides: Vec<usize>,
    b_batch_strides: Vec<usize>,
    m: usize,
    k: usize,
    n: usize,
}

/// One structural dispatch decision shared by execution and workspace planning.
///
/// Both single-matrix routes own any cuBLASLt scratch in their cached private
/// plan. Only the batched/broadcast route consumes executor-prepared workspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatMulExecutionRoute {
    DenseGemv,
    DenseCachedGemm,
    ExecutorWorkspaceGemm,
}

#[derive(Debug, PartialEq, Eq)]
struct BatchRun {
    a_matrix: usize,
    b_matrix: usize,
    c_matrix: usize,
    batch: usize,
    a_stride: usize,
    b_stride: usize,
}

fn broadcast_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; dims.len()];
    let mut stride = 1;
    for i in (0..dims.len()).rev() {
        strides[i] = if dims[i] == 1 { 0 } else { stride };
        stride *= dims[i];
    }
    strides
}

fn matmul_plan(a: &[usize], b: &[usize]) -> Result<MatMulPlan> {
    if a.len() < 2 || b.len() < 2 {
        return Err(not_implemented(format!(
            "MatMul with operand ranks {}D x {}D (rank-1 promotion is not supported yet)",
            a.len(),
            b.len()
        )));
    }
    let (m, k, n) = (a[a.len() - 2], a[a.len() - 1], b[b.len() - 1]);
    if b[b.len() - 2] != k {
        return Err(inner_mismatch(a, b));
    }

    let batch_rank = (a.len() - 2).max(b.len() - 2);
    let mut a_batch_dims = vec![1; batch_rank];
    let mut b_batch_dims = vec![1; batch_rank];
    a_batch_dims[batch_rank - (a.len() - 2)..].copy_from_slice(&a[..a.len() - 2]);
    b_batch_dims[batch_rank - (b.len() - 2)..].copy_from_slice(&b[..b.len() - 2]);

    let mut batch_shape = Vec::with_capacity(batch_rank);
    for (&ad, &bd) in a_batch_dims.iter().zip(&b_batch_dims) {
        if ad != bd && ad != 1 && bd != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: batch dimensions do not broadcast between A {a:?} and B {b:?}"
            )));
        }
        batch_shape.push(ad.max(bd));
    }

    Ok(MatMulPlan {
        a_batch_strides: broadcast_strides(&a_batch_dims),
        b_batch_strides: broadcast_strides(&b_batch_dims),
        batch_shape,
        m,
        k,
        n,
    })
}

impl MatMulPlan {
    fn execution_route(&self, dtype: GemmDtype) -> MatMulExecutionRoute {
        let single_matrix = self.batch_shape.iter().all(|&dim| dim == 1);
        if single_matrix
            && self.m == 1
            && self.k > 0
            && self.n > 0
            && matches!(dtype, GemmDtype::F16 | GemmDtype::F32)
        {
            MatMulExecutionRoute::DenseGemv
        } else if single_matrix {
            MatMulExecutionRoute::DenseCachedGemm
        } else {
            MatMulExecutionRoute::ExecutorWorkspaceGemm
        }
    }

    fn output_shape(&self) -> Vec<usize> {
        let mut shape = self.batch_shape.clone();
        shape.extend([self.m, self.n]);
        shape
    }

    fn batch_runs(&self) -> Vec<BatchRun> {
        if self.batch_shape.is_empty() {
            return vec![BatchRun {
                a_matrix: 0,
                b_matrix: 0,
                c_matrix: 0,
                batch: 1,
                a_stride: 0,
                b_stride: 0,
            }];
        }

        let inner = *self.batch_shape.last().unwrap();
        let outer: usize = self.batch_shape[..self.batch_shape.len() - 1]
            .iter()
            .product();
        let mut runs = Vec::with_capacity(outer);
        for outer_index in 0..outer {
            let mut remaining = outer_index;
            let mut a_matrix = 0;
            let mut b_matrix = 0;
            for axis in (0..self.batch_shape.len() - 1).rev() {
                let coord = remaining % self.batch_shape[axis];
                remaining /= self.batch_shape[axis];
                a_matrix += coord * self.a_batch_strides[axis];
                b_matrix += coord * self.b_batch_strides[axis];
            }
            let last = self.batch_shape.len() - 1;
            runs.push(BatchRun {
                a_matrix,
                b_matrix,
                c_matrix: outer_index * inner,
                batch: inner,
                a_stride: self.a_batch_strides[last],
                b_stride: self.b_batch_strides[last],
            });
        }
        runs
    }
}

fn inner_mismatch(a: &[usize], b: &[usize]) -> EpError {
    EpError::KernelFailed(format!(
        "cuda_ep MatMul: inner dimensions disagree between A {a:?} and B {b:?}"
    ))
}

/// None denotes the existing batched metadata planner, never zero demand.
fn dense_metadata_preparation(
    plan: &MatMulPlan,
    dtype: GemmDtype,
    use_cublaslt: bool,
) -> Option<onnx_runtime_ep_api::WorkspacePreparation> {
    match plan.execution_route(dtype) {
        MatMulExecutionRoute::ExecutorWorkspaceGemm => None,
        MatMulExecutionRoute::DenseGemv if dtype == GemmDtype::F16 && !use_cublaslt => Some(
            onnx_runtime_ep_api::WorkspacePreparation::Ready(WorkspaceRequirement::NONE),
        ),
        _ => Some(onnx_runtime_ep_api::WorkspacePreparation::Deferred),
    }
}

/// Whether the M==1 fp16 `lm_head` decode MatMul takes the capturable cuBLASLt
/// plan path (default) instead of the hand fp16 GEMV. Default-ON: cuBLASLt is
/// ~1.5x faster on the dense logits projection and, with the shape-keyed plan
/// cache (the cached host plan), is CUDA-graph capturable
/// across prefill/decode/verify shapes. Set `ONNX_GENAI_LMHEAD_CUBLASLT` to
/// `0`/`false`/`off` to fall back to the hand GEMV (the OFF escape hatch).
fn lmhead_cublaslt_enabled() -> bool {
    !matches!(
        std::env::var("ONNX_GENAI_LMHEAD_CUBLASLT").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

impl MatMulKernel {
    fn validate_operands(
        inputs: &[TensorView],
        outputs: &[TensorMut],
    ) -> Result<(GemmDtype, MatMulPlan)> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: expected 2 inputs and 1 output, got {} and {}; provide both operands and one destination",
                inputs.len(),
                outputs.len()
            )));
        }
        let [a, b] = inputs else { unreachable!() };
        let dtype = gemm_dtype(a.dtype)?;
        if b.dtype != a.dtype || outputs[0].dtype != a.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: mixed dtypes A={:?} B={:?} C={:?}; all must match",
                a.dtype, b.dtype, outputs[0].dtype
            )));
        }
        if !a.is_contiguous() || !b.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented(
                "MatMul with non-contiguous operands/output; materialize contiguous storage before execution",
            ));
        }
        let plan = matmul_plan(a.shape, b.shape)?;
        let expected = plan.output_shape();
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: output shape {:?}, expected {expected:?}; bind the inferred output shape",
                outputs[0].shape
            )));
        }
        Ok((dtype, plan))
    }

    fn signature(
        dtype: GemmDtype,
        plan: &MatMulPlan,
        route: MatMulCaptureRoute,
        inputs: &[TensorView],
        outputs: &[TensorMut],
    ) -> MatMulCaptureSignature {
        MatMulCaptureSignature {
            dtype,
            route,
            a_shape: inputs[0].shape.to_vec(),
            b_shape: inputs[1].shape.to_vec(),
            output_shape: outputs[0].shape.to_vec(),
            m: plan.m,
            k: plan.k,
            n: plan.n,
        }
    }

    fn prepare_execution(
        &self,
        inputs: &[TensorView],
        outputs: &[TensorMut],
        metadata: &[TensorMetadata<'_>],
        cached: Option<onnx_runtime_ep_api::WorkspacePreparation>,
    ) -> Result<WorkspaceRequirement> {
        let mut state = self.warm_state.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep MatMul: warm-state lock poisoned; recreate the kernel".into(),
            )
        })?;
        // A failed refiner or generic admission must revoke capture proof too:
        // neither failure necessarily reaches execute.
        let warmed = state.capture_ready.take();
        state.prepared = None;
        let (dtype, geometry) = Self::validate_operands(inputs, outputs)?;
        if geometry.execution_route(dtype) == MatMulExecutionRoute::ExecutorWorkspaceGemm {
            return match cached {
                Some(preparation) => preparation.require_ready(),
                None => self.workspace_requirement_for(metadata),
            };
        }
        let params = dense_gemm_params(
            dtype,
            geometry.m,
            geometry.k,
            geometry.n,
            cuptr(inputs[0].data_ptr::<u8>().cast()),
            cuptr(inputs[1].data_ptr::<u8>().cast()),
            // Inspect only the opaque element origin; output values are uninitialized.
            cuptr(
                outputs[0]
                    .data
                    .0
                    .cast::<u8>()
                    .wrapping_add(outputs[0].byte_offset)
                    .cast(),
            ),
        );
        let f32_decode = dtype == GemmDtype::F32 && geometry.m == 1;
        let selected = if matches!(
            dense_metadata_preparation(&geometry, dtype, lmhead_cublaslt_enabled()),
            Some(onnx_runtime_ep_api::WorkspacePreparation::Ready(_))
        ) {
            PreparedPlan::F16HandGemv
        } else if f32_decode && state.f32_gemv.as_ref().is_some_and(|p| p.matches(&params)) {
            PreparedPlan::F32Slot
        } else if !f32_decode
            && let Some(index) = state.dense_plans.iter().position(|p| p.matches(&params))
        {
            PreparedPlan::Dense(index)
        } else {
            // Warm hits do not ask the driver whether capture is active.
            if self.runtime.is_capturing()? {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MatMul: dtype={dtype:?} M={} K={} N={} and its address alignment were not warmed before capture; abort capture and eagerly prepare/execute the exact bindings",
                    geometry.m, geometry.k, geometry.n
                )));
            }
            let candidate = CachedGemmPlan::new(&self.runtime, &params)?;
            self.runtime
                .staged_warm_cache_mutation("MatMul dense host-plan creation")?;
            PreparedPlan::Candidate(candidate)
        };
        let (bytes, route) = selected
            .cached(&state)?
            .map_or((0, MatMulCaptureRoute::F16HandGemv), |p| {
                (p.workspace_bytes(), p.route())
            });
        state.prepared = Some(PreparedDispatch {
            signature: Self::signature(dtype, &geometry, route, inputs, outputs),
            plan: selected,
            warmed,
        });
        Ok(blas::governed_workspace_requirement(bytes))
    }

    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let mut warm_state = self.warm_state.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep MatMul: warm-state lock poisoned; recreate the kernel".into(),
            )
        })?;
        warm_state.capture_ready = None;
        let prepared = warm_state.prepared.take();
        let (dtype, plan) = Self::validate_operands(inputs, outputs)?;
        let a = &inputs[0];
        let b = &inputs[1];
        let execution_route = plan.execution_route(dtype);
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            crate::trace::product(plan.batch_shape.iter().copied())
                .saturating_mul(plan.m as u64)
                .saturating_mul(plan.n as u64)
                .saturating_mul(plan.k as u64)
                .saturating_mul(2)
        });
        let capturing = self.runtime.is_capturing()?;
        let a_ptr = cuptr(a.data_ptr::<u8>().cast());
        let b_ptr = cuptr(b.data_ptr::<u8>().cast());
        let c_ptr = cuptr(outputs[0].data_ptr_mut::<u8>().cast());
        if execution_route != MatMulExecutionRoute::ExecutorWorkspaceGemm {
            let prepared = prepared.ok_or_else(|| EpError::KernelFailed("cuda_ep MatMul: no execution-time plan was prepared; refine workspace using the real input/output views, admit its exact requirement, then execute with the borrowed workspace".into()))?;
            let signature = prepared.signature;
            if signature.dtype != dtype
                || (signature.m, signature.k, signature.n) != (plan.m, plan.k, plan.n)
                || signature.a_shape != a.shape
                || signature.b_shape != b.shape
                || signature.output_shape != outputs[0].shape
            {
                return Err(EpError::KernelFailed("cuda_ep MatMul: bindings changed after workspace refinement; repeat refinement and admission for the actual launch bindings".into()));
            }
            if capturing && prepared.warmed.as_ref() != Some(&signature) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep MatMul: signature changed or no successful warm preceded capture: current={signature:?}; abort capture and eagerly execute the exact signature"
                )));
            }
            if capturing && matches!(prepared.plan, PreparedPlan::Candidate(_)) {
                return Err(EpError::KernelFailed("cuda_ep MatMul: a cold host-plan candidate cannot be launched during capture; eagerly execute it first".into()));
            }
            let params = dense_gemm_params(dtype, plan.m, plan.k, plan.n, a_ptr, b_ptr, c_ptr);
            if let Some(cached) = prepared.plan.cached(&warm_state)? {
                if !cached.matches(&params) {
                    return Err(EpError::KernelFailed("cuda_ep MatMul: launch addresses violate the admitted plan's alignment contract; refine again with the actual addresses".into()));
                }
                let used = dense_workspace(cached.workspace_bytes(), workspace)?;
                if capturing && used.bytes() != 0 {
                    self.runtime.require_registered_workspace_capture(
                        used,
                        "MatMul dense cuBLASLt workspace",
                    )?;
                }
                let replacing = if dtype == GemmDtype::F32 && plan.m == 1 {
                    warm_state.f32_gemv.is_some()
                } else {
                    warm_state.dense_plans.len() == DENSE_PLAN_CACHE_CAP
                };
                if matches!(prepared.plan, PreparedPlan::Candidate(_)) && replacing {
                    // Retain the existing conservative retirement boundary for
                    // host descriptors which could still have queued users.
                    self.runtime.drain_for_unmap()?;
                }
                match &cached.algorithm {
                    DenseAlgorithm::CublasLt(host_plan) => {
                        // SAFETY: validated geometry and exact alignment contract;
                        // `used` covers this very plan's admitted scratch.
                        unsafe {
                            host_plan.launch(
                                self.runtime.blas(),
                                self.runtime.stream_ptr(),
                                &params,
                                cuptr(used.ptr().0.cast_const()),
                            )
                        }?;
                    }
                    DenseAlgorithm::F16HandGemv => {
                        self.launch_dense_gemv_f16(a_ptr, b_ptr, c_ptr, plan.k, plan.n)?
                    }
                }
            } else {
                self.launch_dense_gemv_f16(a_ptr, b_ptr, c_ptr, plan.k, plan.n)?;
            }
            match prepared.plan {
                PreparedPlan::Candidate(candidate) if dtype == GemmDtype::F32 && plan.m == 1 => {
                    warm_state.f32_gemv = Some(candidate)
                }
                PreparedPlan::Candidate(candidate) => {
                    warm_state.dense_plans.insert(0, candidate);
                    warm_state.dense_plans.truncate(DENSE_PLAN_CACHE_CAP);
                }
                PreparedPlan::Dense(index) if !capturing && index != 0 => {
                    let plan = warm_state.dense_plans.remove(index);
                    warm_state.dense_plans.insert(0, plan);
                }
                _ => {}
            }
            warm_state.capture_ready = Some(signature);
            return Ok(());
        }

        let elem_bytes = a.dtype.byte_size();
        let a_matrix_bytes = plan.m * plan.k * elem_bytes;
        let b_matrix_bytes = plan.k * plan.n * elem_bytes;
        let c_matrix_bytes = plan.m * plan.n * elem_bytes;

        debug_assert_eq!(execution_route, MatMulExecutionRoute::ExecutorWorkspaceGemm);
        if capturing {
            return Err(EpError::KernelFailed(
                "cuda_ep MatMul: batched/broadcast MatMul is not capture-safe. HOW: abort capture \
                 and use a warmed dense GEMV or plain 2-D GEMM signature."
                    .into(),
            ));
        }
        let runs = plan.batch_runs();
        runs.into_iter()
            .try_for_each(|run| {
                let params = GemmParams {
                    dtype,
                    a: a_ptr + (run.a_matrix * a_matrix_bytes) as u64,
                    b: b_ptr + (run.b_matrix * b_matrix_bytes) as u64,
                    c: c_ptr + (run.c_matrix * c_matrix_bytes) as u64,
                    m: plan.m,
                    k: plan.k,
                    n: plan.n,
                    batch: run.batch,
                    a_batch_stride: run.a_stride * plan.m * plan.k,
                    b_batch_stride: run.b_stride * plan.k * plan.n,
                    epilogue: None,
                };
                // SAFETY: the plan's broadcast offsets address complete matrices
                // inside A/B/Y; workspace and stream remain live for every run.
                unsafe {
                    blas::governed_gemm(
                        self.runtime.blas(),
                        self.runtime.stream_ptr(),
                        &params,
                        workspace,
                        "MatMul",
                    )
                }
            })
            .and_then(|()| self.runtime.synchronize())?;
        Ok(())
    }

    fn workspace_requirement_for(
        &self,
        inputs: &[TensorMetadata<'_>],
    ) -> Result<WorkspaceRequirement> {
        let [a, b] = inputs else {
            return Ok(WorkspaceRequirement::NONE);
        };
        if b.dtype != a.dtype {
            return Ok(WorkspaceRequirement::NONE);
        }
        let dtype = gemm_dtype(a.dtype)?;
        let plan = matmul_plan(a.shape, b.shape)?;
        if plan.execution_route(dtype) != MatMulExecutionRoute::ExecutorWorkspaceGemm {
            return Ok(WorkspaceRequirement::NONE);
        }
        let mut peak = 0usize;
        for run in plan.batch_runs() {
            let params = GemmParams {
                dtype,
                a: 1,
                b: 1,
                c: 1,
                m: plan.m,
                k: plan.k,
                n: plan.n,
                batch: run.batch,
                a_batch_stride: run.a_stride * plan.m * plan.k,
                b_batch_stride: run.b_stride * plan.k * plan.n,
                epilogue: None,
            };
            peak = peak.max(blas::gemm_workspace_bytes(self.runtime.blas(), &params)?);
        }
        Ok(blas::governed_workspace_requirement(peak))
    }

    /// Launch the dense fp16 GEMV (`GEMV_F16_SRC`) on the runtime stream.
    ///
    /// Allocation- and synchronization-free: one thread per output column,
    /// `blockDim.x` floats of launch-time shared memory, fixed grid geometry
    /// from `(k, n)`. This is legal to record into and replay from a CUDA graph.
    fn launch_dense_gemv_f16(
        &self,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        k: usize,
        n: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMul fp16 GEMV")?;
        let function =
            self.runtime
                .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, GEMV_F16_ENTRY)?;
        let k_i32 = i32::try_from(k)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep MatMul: K={k} exceeds i32")))?;
        let n_i32 = i32::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep MatMul: N={n} exceeds i32")))?;
        let shared_mem_bytes = GEMV_F16_THREADS * std::mem::size_of::<f32>() as u32;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&c_ptr)
            .arg(&k_i32)
            .arg(&n_i32);
        // SAFETY: pointers address contiguous fp16 `a[K]`, `B[K, N]`, and `y[N]`
        // buffers validated by the caller; the scalar ABI matches the entry
        // point. The launch uses only registers and launch-time shared memory,
        // with no per-call allocation or synchronization, so it is capture-safe.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: ((n as u32).div_ceil(GEMV_F16_THREADS), 1, 1),
                block_dim: (GEMV_F16_THREADS, 1, 1),
                shared_mem_bytes,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMul fp16 GEMV", err))
    }
}

impl Kernel for MatMulKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<onnx_runtime_ep_api::WorkspacePreparation> {
        if let [a, b] = inputs {
            let dtype = gemm_dtype(a.dtype)?;
            let plan = matmul_plan(a.shape, b.shape)?;
            if let Some(preparation) =
                dense_metadata_preparation(&plan, dtype, lmhead_cublaslt_enabled())
            {
                return Ok(preparation);
            }
        }
        self.workspace_requirement_for(inputs)
            .map(onnx_runtime_ep_api::WorkspacePreparation::Ready)
    }

    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        outputs: &[TensorMut],
        metadata: &[TensorMetadata<'_>],
        _output_metadata: &[onnx_runtime_ep_api::OutputMetadata<'_>],
        cached: Option<onnx_runtime_ep_api::WorkspacePreparation>,
    ) -> Result<WorkspaceRequirement> {
        self.prepare_execution(inputs, outputs, metadata, cached)
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
        // Dense inputs only (see `run`).
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        // The dense f32/fp16 M==1 GEMV fast paths and the plain 2-D (`batch==1`)
        // M>1 cached-plan path perform no per-call allocation, D2H, heuristic
        // query, or synchronization. Advertise capture only after such a call
        // has warmed the selected host algorithm and borrowed caller workspace.
        match self.warm_state.lock() {
            Ok(state) if state.capture_ready.is_some() => {
                onnx_runtime_ep_api::CaptureSupport::Supported
            }
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "requires a dense f32/fp16 GEMV (M==1) or a plain 2-D (batch==1) \
                 dense GEMM warmed at the captured shape; batched/broadcast \
                 cuBLASLt GEMMs still perform a per-call heuristic query and are \
                 not capturable",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "MatMul capture readiness is unavailable because its state lock was poisoned",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_metadata_never_substitutes_zero_for_an_address_selected_plan() {
        use onnx_runtime_ep_api::WorkspacePreparation::{Deferred, Ready};
        for (a, b) in [([1, 17], [17, 23]), ([4, 17], [17, 23])] {
            let plan = matmul_plan(&a, &b).unwrap();
            for dtype in [GemmDtype::F32, GemmDtype::F16, GemmDtype::Bf16] {
                assert_eq!(
                    dense_metadata_preparation(&plan, dtype, true),
                    Some(Deferred)
                );
                let off = dense_metadata_preparation(&plan, dtype, false).unwrap();
                if dtype == GemmDtype::F16 && a[0] == 1 {
                    assert_eq!(off, Ready(WorkspaceRequirement::NONE));
                } else {
                    assert_eq!(off, Deferred);
                }
            }
        }
        let batched = matmul_plan(&[2, 4, 17], &[2, 17, 23]).unwrap();
        assert_eq!(
            dense_metadata_preparation(&batched, GemmDtype::F32, true),
            None
        );
    }

    #[test]
    fn dense_scratch_binds_only_the_exact_admitted_span() {
        use onnx_runtime_ep_api::DevicePtrMut;
        let view = WorkspaceView::new(DevicePtrMut(0x1000usize as *mut _), 8192);
        let used = dense_workspace(1024, Some(view)).unwrap();
        assert_eq!(used.ptr().0, view.ptr().0);
        assert_eq!(
            used.bytes(),
            1024,
            "capture needs the plan's used span, not an unrelated entire arena"
        );
        for offered in [
            None,
            Some(WorkspaceView::new(view.ptr(), 1023)),
            Some(WorkspaceView::new(
                DevicePtrMut(0x1001usize as *mut _),
                8192,
            )),
        ] {
            let error = dense_workspace(1024, offered).unwrap_err().to_string();
            assert!(
                error.contains("1024 bytes") && error.contains("prepare and pass"),
                "{error}"
            );
        }
        assert_eq!(dense_workspace(0, None).unwrap().bytes(), 0);
    }

    #[test]
    fn plan_2d_ok() {
        let p = matmul_plan(&[2, 3], &[3, 4]).unwrap();
        assert_eq!((p.m, p.k, p.n), (2, 3, 4));
        assert_eq!(p.output_shape(), [2, 4]);
        assert_eq!(p.batch_runs()[0].batch, 1);
        assert_eq!(
            p.execution_route(GemmDtype::F32),
            MatMulExecutionRoute::DenseCachedGemm
        );
    }

    #[test]
    fn plan_3d_equal_batch_ok() {
        let p = matmul_plan(&[5, 2, 3], &[5, 3, 4]).unwrap();
        assert_eq!(p.output_shape(), [5, 2, 4]);
        assert_eq!(p.batch_runs()[0].batch, 5);
        assert_eq!(
            p.execution_route(GemmDtype::F32),
            MatMulExecutionRoute::ExecutorWorkspaceGemm
        );
    }

    #[test]
    fn route_uses_one_single_matrix_predicate_for_dynamic_shapes() {
        let decode = matmul_plan(&[1, 17], &[17, 23]).unwrap();
        assert_eq!(
            decode.execution_route(GemmDtype::F32),
            MatMulExecutionRoute::DenseGemv
        );

        let singleton_batch = matmul_plan(&[1, 4, 17], &[1, 17, 23]).unwrap();
        assert_eq!(
            singleton_batch.execution_route(GemmDtype::Bf16),
            MatMulExecutionRoute::DenseCachedGemm
        );

        let batched = matmul_plan(&[2, 4, 17], &[2, 17, 23]).unwrap();
        assert_eq!(
            batched.execution_route(GemmDtype::F32),
            MatMulExecutionRoute::ExecutorWorkspaceGemm
        );
    }

    #[test]
    fn plan_inner_mismatch_is_plain_error() {
        let e = matmul_plan(&[2, 3], &[4, 5]).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("inner dimensions disagree"), "{msg}");
        // A genuine mistake, not a deferred feature.
        assert!(!msg.contains("not yet implemented"), "{msg}");
    }

    #[test]
    fn plan_broadcast_batch() {
        let p = matmul_plan(&[3, 1, 2, 4], &[1, 5, 4, 6]).unwrap();
        assert_eq!(p.output_shape(), [3, 5, 2, 6]);
        assert_eq!(
            p.batch_runs(),
            [
                BatchRun {
                    a_matrix: 0,
                    b_matrix: 0,
                    c_matrix: 0,
                    batch: 5,
                    a_stride: 0,
                    b_stride: 1
                },
                BatchRun {
                    a_matrix: 1,
                    b_matrix: 0,
                    c_matrix: 5,
                    batch: 5,
                    a_stride: 0,
                    b_stride: 1
                },
                BatchRun {
                    a_matrix: 2,
                    b_matrix: 0,
                    c_matrix: 10,
                    batch: 5,
                    a_stride: 0,
                    b_stride: 1
                },
            ]
        );
    }

    #[test]
    fn plan_high_rank_equal_batch() {
        let p = matmul_plan(&[2, 3, 4, 5], &[2, 3, 5, 6]).unwrap();
        assert_eq!(p.output_shape(), [2, 3, 4, 6]);
        assert_eq!(p.batch_runs().len(), 2);
        assert!(p.batch_runs().iter().all(|run| run.batch == 3));
    }

    #[test]
    fn plan_2d_broadcast_across_4d() {
        let p = matmul_plan(&[4, 5], &[2, 3, 5, 6]).unwrap();
        assert_eq!(p.output_shape(), [2, 3, 4, 6]);
        assert!(p.batch_runs().iter().all(|run| run.a_stride == 0));
    }

    #[test]
    fn plan_rejects_rank_1_with_clear_error() {
        let e = matmul_plan(&[5], &[5, 6]).unwrap_err();
        assert!(format!("{e}").contains("rank-1 promotion"), "{e}");
    }

    #[test]
    fn dtype_mapping_and_unsupported() {
        assert_eq!(gemm_dtype(DataType::Float32).unwrap(), GemmDtype::F32);
        assert_eq!(gemm_dtype(DataType::Float16).unwrap(), GemmDtype::F16);
        assert_eq!(gemm_dtype(DataType::BFloat16).unwrap(), GemmDtype::Bf16);
        let e = gemm_dtype(DataType::Int64).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("dtype Int64"), "{msg}");
        assert!(msg.contains("not yet implemented"), "{msg}");
    }
}

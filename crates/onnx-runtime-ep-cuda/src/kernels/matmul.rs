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

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::cublaslt::{result as cublaslt, sys as cublaslt_sys};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    DeviceGraphResource, EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut,
    TensorView, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node};

use crate::blas::{self, GemmDtype, GemmParams, WORKSPACE_BYTES};
use crate::error::{cublas_err, driver_err, not_implemented};
use crate::runtime::{CudaRuntime, GraphDeviceAllocation, cuptr};

/// NVRTC module/entry for the dense decode GEMVs.
const GEMV_F16_MODULE: &str = "matmul_dense_gemv_f16";
const GEMV_F16_ENTRY: &str = "matmul_dense_gemv_f16";
/// Upper bound on the number of distinct dense `(dtype, M, K, N)` cuBLASLt plans
/// a single `MatMul` node keeps warm simultaneously. A node realistically sees a
/// small, fixed set — decode `M==1`, a handful of prefill widths, and any
/// speculative verify width — so 8 covers the hot working set while capping the
/// per-node workspace footprint. Eviction is LRU and only runs off the
/// (non-capturing) creation path, so the MRU decode plan baked into a live
/// captured graph is never freed.
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
    /// cuBLASLt objects and workspace preselected during the f32 M==1 warmup.
    /// Reusing the exact algorithm preserves bitwise parity with the old path
    /// while eliminating all capture-time setup and device allocation.
    f32_gemv: Option<F32GemvPlan>,
    /// cuBLASLt plans + persistent workspaces keyed by the 2-D (`batch == 1`)
    /// dense shape `(dtype, M, K, N)`. Each distinct shape the node runs keeps
    /// its own preselected algorithm, so alternating shapes — prefill `M>1`,
    /// decode `M==1`, and a speculative `M=K` verify width — all replay without
    /// a per-call heuristic query, keeping the plain-`MatMul` path (e.g. the
    /// logits projection `lm_head`) CUDA-graph capturable across shape changes.
    /// MRU-ordered (front = most recent); bounded by [`DENSE_PLAN_CACHE_CAP`].
    dense_plans: Vec<DenseGemmPlan>,
    /// Immutable signature and exact private workspace owner from the most
    /// recent successful capture-safe call.
    capture_ready: Option<Arc<MatMulCaptureReady>>,
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

#[derive(Clone)]
struct MatMulCaptureReady {
    signature: MatMulCaptureSignature,
    resources: Vec<DeviceGraphResource>,
}

struct F32GemvPlan {
    runtime: Arc<CudaRuntime>,
    k: usize,
    n: usize,
    handle: cublaslt_sys::cublasLtHandle_t,
    desc: cublaslt_sys::cublasLtMatmulDesc_t,
    a_layout: cublaslt_sys::cublasLtMatrixLayout_t,
    b_layout: cublaslt_sys::cublasLtMatrixLayout_t,
    c_layout: cublaslt_sys::cublasLtMatrixLayout_t,
    algo: cublaslt_sys::cublasLtMatmulAlgo_t,
    workspace: Option<Arc<GraphDeviceAllocation>>,
    workspace_bytes: usize,
}

// SAFETY: cuBLASLt handles/descriptors are context-independent host objects.
// Calls through a plan are serialized by `MatMulKernel::f32_gemv`.
unsafe impl Send for F32GemvPlan {}

impl Drop for F32GemvPlan {
    fn drop(&mut self) {
        // SAFETY: every object was created once by `F32GemvPlan::new` and is
        // destroyed exactly once after the plan can no longer be launched.
        unsafe {
            if !self.c_layout.is_null() {
                let _ = cublaslt::destroy_matrix_layout(self.c_layout);
            }
            if !self.b_layout.is_null() {
                let _ = cublaslt::destroy_matrix_layout(self.b_layout);
            }
            if !self.a_layout.is_null() {
                let _ = cublaslt::destroy_matrix_layout(self.a_layout);
            }
            if !self.desc.is_null() {
                let _ = cublaslt::destroy_matmul_desc(self.desc);
            }
            if !self.handle.is_null() {
                let _ = cublaslt::destroy_handle(self.handle);
            }
        }
    }
}

impl F32GemvPlan {
    fn new(runtime: Arc<CudaRuntime>, k: usize, n: usize) -> Result<Self> {
        let mut plan = Self {
            runtime,
            k,
            n,
            handle: std::ptr::null_mut(),
            desc: std::ptr::null_mut(),
            a_layout: std::ptr::null_mut(),
            b_layout: std::ptr::null_mut(),
            c_layout: std::ptr::null_mut(),
            // SAFETY: the algorithm is not read until the heuristic initializes it.
            algo: unsafe { std::mem::zeroed() },
            workspace: None,
            workspace_bytes: 0,
        };
        plan.handle = cublaslt::create_handle().map_err(|e| cublas_err("cublasLtCreate", e))?;
        let dt = cublaslt_sys::cudaDataType_t::CUDA_R_32F;
        plan.a_layout = cublaslt::create_matrix_layout(dt, n as u64, k as u64, n as i64)
            .map_err(|e| cublas_err("cublasLtMatrixLayoutCreate(B)", e))?;
        plan.b_layout = cublaslt::create_matrix_layout(dt, k as u64, 1, k as i64)
            .map_err(|e| cublas_err("cublasLtMatrixLayoutCreate(A)", e))?;
        plan.c_layout = cublaslt::create_matrix_layout(dt, n as u64, 1, n as i64)
            .map_err(|e| cublas_err("cublasLtMatrixLayoutCreate(C)", e))?;
        plan.desc =
            cublaslt::create_matmul_desc(cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, dt)
                .map_err(|e| cublas_err("cublasLtMatmulDescCreate", e))?;
        let pref = cublaslt::create_matmul_pref()
            .map_err(|e| cublas_err("cublasLtMatmulPreferenceCreate", e))?;
        let heuristic_result = (|| {
            unsafe {
                cublaslt::set_matmul_pref_attribute(
                    pref,
                    cublaslt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    (&WORKSPACE_BYTES) as *const usize as *const c_void,
                    std::mem::size_of::<usize>(),
                )
            }
            .map_err(|e| cublas_err("set MAX_WORKSPACE_BYTES", e))?;
            unsafe {
                cublaslt::get_matmul_algo_heuristic(
                    plan.handle,
                    plan.desc,
                    plan.a_layout,
                    plan.b_layout,
                    plan.c_layout,
                    plan.c_layout,
                    pref,
                )
            }
            .map_err(|e| cublas_err("select f32 M==1 MatMul algorithm", e))
        })();
        // SAFETY: `pref` is live and is never retained by the selected algorithm.
        unsafe {
            let _ = cublaslt::destroy_matmul_pref(pref);
        }
        let heuristic = heuristic_result?;
        plan.algo = heuristic.algo;
        plan.workspace_bytes = heuristic.workspaceSize;
        if plan.workspace_bytes > 0 {
            plan.workspace = Some(GraphDeviceAllocation::allocate(
                &plan.runtime,
                plan.workspace_bytes,
            )?);
        }
        Ok(plan)
    }

    fn launch(&self, stream: cudarc::driver::sys::CUstream, a: u64, b: u64, c: u64) -> Result<()> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            cublaslt::matmul(
                self.handle,
                self.desc,
                (&alpha) as *const f32 as *const c_void,
                (&beta) as *const f32 as *const c_void,
                b as *const c_void,
                self.a_layout,
                a as *const c_void,
                self.b_layout,
                c as *const c_void,
                self.c_layout,
                c as *mut c_void,
                self.c_layout,
                (&self.algo) as *const cublaslt_sys::cublasLtMatmulAlgo_t,
                self.workspace
                    .as_ref()
                    .map_or(0, |workspace| workspace.ptr()) as *mut c_void,
                self.workspace_bytes,
                stream as cublaslt_sys::cudaStream_t,
            )
        }
        .map_err(|e| cublas_err("cublasLtMatmul f32 M==1", e))
    }

    fn device_graph_resource(&self) -> Option<DeviceGraphResource> {
        self.workspace
            .as_ref()
            .map(GraphDeviceAllocation::device_graph_resource)
    }
}

/// Cached cuBLASLt plan + persistent workspace for a fixed 2-D (`batch == 1`)
/// dense GEMM shape at M>1. Selected once, then replayed with no heuristic
/// query, allocation, or synchronization — the M>1 analogue of [`F32GemvPlan`].
/// This makes the plain-`MatMul` M>1 path CUDA-graph capturable, closing the
/// last capture seam at a speculative M=K verify width (the logits projection).
struct DenseGemmPlan {
    runtime: Arc<CudaRuntime>,
    dtype: GemmDtype,
    m: usize,
    k: usize,
    n: usize,
    plan: blas::CaptureGemmPlan,
    workspace: Option<Arc<GraphDeviceAllocation>>,
}

// SAFETY: the plan holds only context-independent cuBLASLt host handles and a
// device workspace pointer; launches are serialized by `MatMulKernel::launch_dense_capturable`.
unsafe impl Send for DenseGemmPlan {}

impl DenseGemmPlan {
    fn matches(&self, dtype: GemmDtype, m: usize, k: usize, n: usize) -> bool {
        self.dtype == dtype && self.m == m && self.k == k && self.n == n
    }

    fn new(
        runtime: Arc<CudaRuntime>,
        dtype: GemmDtype,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Self> {
        let params = dense_gemm_params(dtype, m, k, n, 0, 0, 0);
        let plan = blas::plan_capture_gemm(runtime.blas(), &params)?;
        let workspace_bytes = plan.workspace_bytes();
        let workspace = if workspace_bytes > 0 {
            Some(GraphDeviceAllocation::allocate(&runtime, workspace_bytes)?)
        } else {
            None
        };
        Ok(Self {
            runtime,
            dtype,
            m,
            k,
            n,
            plan,
            workspace,
        })
    }

    fn launch(&self, a: u64, b: u64, c: u64) -> Result<()> {
        let params = dense_gemm_params(self.dtype, self.m, self.k, self.n, a, b, c);
        // SAFETY: `params` matches the shape the plan was selected for; a/b/c are
        // live device buffers for this call; `workspace` covers the plan's
        // requirement; the runtime stream is valid and current.
        unsafe {
            self.plan.launch(
                self.runtime.blas(),
                self.runtime.stream_ptr(),
                &params,
                self.workspace
                    .as_ref()
                    .map_or(0, |workspace| workspace.ptr()),
            )
        }
    }

    fn device_graph_resource(&self) -> Option<DeviceGraphResource> {
        self.workspace
            .as_ref()
            .map(GraphDeviceAllocation::device_graph_resource)
    }
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
    DensePrivateGemm,
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
            MatMulExecutionRoute::DensePrivateGemm
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

/// Whether the M==1 fp16 `lm_head` decode MatMul takes the capturable cuBLASLt
/// plan path (default) instead of the hand fp16 GEMV. Default-ON: cuBLASLt is
/// ~1.5x faster on the dense logits projection and, with the shape-keyed plan
/// cache ([`MatMulKernel::launch_dense_capturable`]), is CUDA-graph capturable
/// across prefill/decode/verify shapes. Set `ONNX_GENAI_LMHEAD_CUBLASLT` to
/// `0`/`false`/`off` to fall back to the hand GEMV (the OFF escape hatch).
fn lmhead_cublaslt_enabled() -> bool {
    !matches!(
        std::env::var("ONNX_GENAI_LMHEAD_CUBLASLT").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

impl MatMulKernel {
    fn validate_capture_signature(
        state: &MatMulWarmState,
        signature: &MatMulCaptureSignature,
    ) -> Result<()> {
        let ready = state.capture_ready.as_ref().ok_or_else(|| {
            EpError::KernelFailed(
                "cuda_ep MatMul: capture began without a successful warmed signature. HOW: run \
                 the exact dense MatMul signature eagerly before capture."
                    .into(),
            )
        })?;
        if ready.signature != *signature {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: signature changed during CUDA graph capture: warmed={:?}, \
                 current={signature:?}. HOW: abort capture and warm the exact replacement.",
                ready.signature
            )));
        }
        Ok(())
    }

    fn publish_capture_ready(
        state: &mut MatMulWarmState,
        signature: MatMulCaptureSignature,
        resources: Vec<DeviceGraphResource>,
    ) {
        state.capture_ready = Some(Arc::new(MatMulCaptureReady {
            signature,
            resources,
        }));
    }

    fn publish_capture_unsupported(state: &mut MatMulWarmState) {
        state.capture_ready = None;
    }

    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: expected 2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let a = &inputs[0];
        let b = &inputs[1];

        // All operands must share one supported element type.
        let dtype = gemm_dtype(a.dtype)?;
        if b.dtype != a.dtype || outputs[0].dtype != a.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: mixed dtypes A={:?} B={:?} C={:?} (all must match)",
                a.dtype, b.dtype, outputs[0].dtype
            )));
        }

        // Dense, row-major device buffers are required. Strided views (e.g. a
        // transposed input) must be materialised by the graph.
        if !a.is_contiguous() || !b.is_contiguous() {
            return Err(not_implemented(
                "MatMul with a non-contiguous (strided) input; \
                 insert an explicit copy/transpose before the MatMul",
            ));
        }
        if !outputs[0].is_contiguous() {
            return Err(not_implemented("MatMul with a non-contiguous output"));
        }

        let plan = matmul_plan(a.shape, b.shape)?;

        let expected_shape = plan.output_shape();
        if outputs[0].shape != expected_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: output shape {:?}, expected {expected_shape:?}",
                outputs[0].shape
            )));
        }
        let execution_route = plan.execution_route(dtype);
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            crate::trace::product(plan.batch_shape.iter().copied())
                .saturating_mul(plan.m as u64)
                .saturating_mul(plan.n as u64)
                .saturating_mul(plan.k as u64)
                .saturating_mul(2)
        });
        let capturing = self.runtime.is_capturing()?;
        let a_shape = a.shape.to_vec();
        let b_shape = b.shape.to_vec();
        let output_shape = outputs[0].shape.to_vec();
        let capture_signature = |route| MatMulCaptureSignature {
            dtype,
            route,
            a_shape: a_shape.clone(),
            b_shape: b_shape.clone(),
            output_shape: output_shape.clone(),
            m: plan.m,
            k: plan.k,
            n: plan.n,
        };

        // Device pointers (byte_offset applied). These are opaque CUDA
        // addresses, never dereferenced on the host.
        let a_ptr = cuptr(a.data_ptr::<u8>() as *const std::ffi::c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const std::ffi::c_void);
        let c_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const std::ffi::c_void);
        let mut warm_state = self.warm_state.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep MatMul: warm-state lock poisoned".into())
        })?;

        // Decode fast path: a single f32/fp16 `y[1, N] = a[1, K] * B[K, N]`.
        // fp16 uses the dedicated GEMV; f32 reuses a cuBLASLt algorithm and
        // workspace selected once at warmup. Neither path allocates, queries a
        // heuristic, or synchronizes while capturing. The gate is purely
        // structural, never tied to a model dimension.
        if execution_route == MatMulExecutionRoute::DenseGemv {
            let route = match dtype {
                GemmDtype::F16 if lmhead_cublaslt_enabled() => MatMulCaptureRoute::CublasLt,
                GemmDtype::F16 => MatMulCaptureRoute::F16HandGemv,
                GemmDtype::F32 => MatMulCaptureRoute::F32Gemv,
                GemmDtype::Bf16 => unreachable!("bf16 excluded by GEMV gate"),
            };
            let mut signature = capture_signature(route);
            if capturing {
                Self::validate_capture_signature(&warm_state, &signature)?;
            }
            let resources = match dtype {
                GemmDtype::F16 => {
                    if lmhead_cublaslt_enabled() {
                        match self.launch_dense_capturable(
                            &mut warm_state,
                            dtype,
                            plan.m,
                            plan.k,
                            plan.n,
                            a_ptr,
                            b_ptr,
                            c_ptr,
                        ) {
                            Ok(resources) => resources,
                            Err(_err) if !self.runtime.is_capturing()? => {
                                self.launch_dense_gemv_f16(a_ptr, b_ptr, c_ptr, plan.k, plan.n)?;
                                signature = capture_signature(MatMulCaptureRoute::F16HandGemv);
                                Vec::new()
                            }
                            Err(err) => return Err(err),
                        }
                    } else {
                        self.launch_dense_gemv_f16(a_ptr, b_ptr, c_ptr, plan.k, plan.n)?;
                        Vec::new()
                    }
                }
                GemmDtype::F32 => self.launch_dense_gemv_f32(
                    &mut warm_state,
                    a_ptr,
                    b_ptr,
                    c_ptr,
                    plan.k,
                    plan.n,
                )?,
                GemmDtype::Bf16 => unreachable!("bf16 excluded by GEMV gate"),
            };
            if !capturing {
                Self::publish_capture_ready(&mut warm_state, signature, resources);
            }
            return Ok(());
        }

        let elem_bytes = a.dtype.byte_size();
        let a_matrix_bytes = plan.m * plan.k * elem_bytes;
        let b_matrix_bytes = plan.k * plan.n * elem_bytes;
        let c_matrix_bytes = plan.m * plan.n * elem_bytes;

        // M>1 capture-safe fast path: a plain 2-D (`batch == 1`) dense GEMM
        // reuses a cuBLASLt plan + persistent workspace selected once at warmup,
        // so replays perform no heuristic query, allocation, or synchronization
        // (its own workspace is never shared, so no post-GEMM sync is needed).
        // This closes the last CUDA-graph capture seam at a speculative M=K
        // verify width — the logits projection (`lm_head`).
        if execution_route == MatMulExecutionRoute::DensePrivateGemm {
            let signature = capture_signature(MatMulCaptureRoute::CublasLt);
            if capturing {
                Self::validate_capture_signature(&warm_state, &signature)?;
            }
            let resources = self.launch_dense_capturable(
                &mut warm_state,
                dtype,
                plan.m,
                plan.k,
                plan.n,
                a_ptr,
                b_ptr,
                c_ptr,
            )?;
            if !capturing {
                Self::publish_capture_ready(&mut warm_state, signature, resources);
            }
            return Ok(());
        }

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
        Self::publish_capture_unsupported(&mut warm_state);
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

    /// Launch the dense true-fp32 GEMV on the runtime stream.
    ///
    /// Warmup selects and retains the cuBLASLt algorithm and its required
    /// workspace. Subsequent launches perform no allocation, host
    /// synchronization, or heuristic query.
    fn launch_dense_gemv_f32(
        &self,
        state: &mut MatMulWarmState,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        k: usize,
        n: usize,
    ) -> Result<Vec<DeviceGraphResource>> {
        let capturing = self.runtime.is_capturing()?;
        if state
            .f32_gemv
            .as_ref()
            .is_some_and(|candidate| candidate.k == k && candidate.n == n)
        {
            let cached = state.f32_gemv.as_ref().unwrap();
            let resource = cached.device_graph_resource();
            if capturing && let Some(resource) = &resource {
                self.runtime.require_registered_address_capture(
                    resource.identity(),
                    "MatMul f32 GEMV workspace",
                )?;
            }
            cached.launch(self.runtime.stream_ptr(), a_ptr, b_ptr, c_ptr)?;
            return Ok(resource.into_iter().collect());
        }
        if capturing {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: f32 GEMV K={k}, N={n} was not warmed before capture"
            )));
        }
        let candidate = F32GemvPlan::new(self.runtime.clone(), k, n)?;
        self.runtime
            .staged_warm_cache_mutation("MatMul f32 GEMV plan/workspace creation")?;
        if state.f32_gemv.is_some() {
            // Retire all users of the old plan without publishing or dropping
            // it; replacement still depends on the candidate launch succeeding.
            self.runtime.drain_for_unmap()?;
        }
        candidate.launch(self.runtime.stream_ptr(), a_ptr, b_ptr, c_ptr)?;
        let resource = candidate.device_graph_resource();
        state.f32_gemv = Some(candidate);
        Ok(resource.into_iter().collect())
    }

    /// Launch a plain 2-D (`batch == 1`) dense M>1 GEMM through a cached
    /// cuBLASLt plan (algorithm + persistent workspace) selected once at warmup.
    ///
    /// Reusing the heuristic-selected algorithm reproduces the per-call
    /// [`governed_gemm`](crate::blas::governed_gemm) arithmetic bit-for-bit at a
    /// fixed shape, while eliminating the capture-time heuristic query,
    /// allocation, and synchronization — so the launch is legal to record into
    /// and replay from a CUDA graph. Mirrors [`Self::launch_dense_gemv_f32`]'s
    /// warm-once / reject-cold-miss-during-capture contract.
    #[allow(clippy::too_many_arguments)]
    fn launch_dense_capturable(
        &self,
        state: &mut MatMulWarmState,
        dtype: GemmDtype,
        m: usize,
        k: usize,
        n: usize,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
    ) -> Result<Vec<DeviceGraphResource>> {
        let capturing = self.runtime.is_capturing()?;
        // Shape-keyed lookup: every distinct (dtype, M, K, N) the node runs keeps
        // its own warm cuBLASLt plan + workspace, so alternating shapes (prefill
        // M>1, decode M==1, a speculative verify width) all replay a preselected
        // algorithm with no per-call heuristic query. A hit is promoted to MRU so
        // the hot decode shape is never the LRU eviction victim.
        if let Some(idx) = state
            .dense_plans
            .iter()
            .position(|plan| plan.matches(dtype, m, k, n))
        {
            let resource = state.dense_plans[idx].device_graph_resource();
            if capturing && let Some(resource) = &resource {
                self.runtime.require_registered_address_capture(
                    resource.identity(),
                    "MatMul dense GEMM workspace",
                )?;
            }
            state.dense_plans[idx].launch(a_ptr, b_ptr, c_ptr)?;
            if !capturing && idx != 0 {
                let plan = state.dense_plans.remove(idx);
                state.dense_plans.insert(0, plan);
            }
            return Ok(resource.into_iter().collect());
        }
        // Cold miss. During capture we must not create a plan (the heuristic
        // query, allocation, and the cache mutation are all illegal inside a
        // captured region), so require the shape to have been warmed by a
        // preceding non-capturing pass — the same contract the single-shape
        // path enforced, now per shape.
        if capturing {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: dense GEMM dtype={dtype:?} M={m} K={k} N={n} \
                 was not warmed before capture"
            )));
        }
        let candidate = DenseGemmPlan::new(self.runtime.clone(), dtype, m, k, n)?;
        self.runtime
            .staged_warm_cache_mutation("MatMul dense plan/workspace creation")?;
        if state.dense_plans.len() == DENSE_PLAN_CACHE_CAP {
            // Complete all users of the eventual LRU victim, but do not evict
            // anything until the replacement launch has succeeded.
            self.runtime.drain_for_unmap()?;
        }
        candidate.launch(a_ptr, b_ptr, c_ptr)?;
        let resource = candidate.device_graph_resource();
        state.dense_plans.insert(0, candidate);
        state.dense_plans.truncate(DENSE_PLAN_CACHE_CAP);
        Ok(resource.into_iter().collect())
    }
}

impl Kernel for MatMulKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }

    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        self.workspace_requirement_for(inputs)
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

    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        self.warm_state
            .lock()
            .ok()
            .and_then(|state| {
                state
                    .capture_ready
                    .as_ref()
                    .map(|ready| ready.resources.clone())
            })
            .unwrap_or_default()
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        // The dense f32/fp16 M==1 GEMV fast paths and the plain 2-D (`batch==1`)
        // M>1 cached-plan path perform no per-call allocation, D2H, heuristic
        // query, or synchronization. Advertise capture only after such a call
        // has warmed the required persistent state (algorithm + workspace).
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
    fn plan_2d_ok() {
        let p = matmul_plan(&[2, 3], &[3, 4]).unwrap();
        assert_eq!((p.m, p.k, p.n), (2, 3, 4));
        assert_eq!(p.output_shape(), [2, 4]);
        assert_eq!(p.batch_runs()[0].batch, 1);
        assert_eq!(
            p.execution_route(GemmDtype::F32),
            MatMulExecutionRoute::DensePrivateGemm
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
            MatMulExecutionRoute::DensePrivateGemm
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

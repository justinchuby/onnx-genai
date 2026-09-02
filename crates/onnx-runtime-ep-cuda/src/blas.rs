//! cuBLASLt GEMM plumbing for the CUDA EP (`docs/architecture/ORT2.md` §15.3).
//!
//! This module owns the single hardest correctness detail in the crate: the
//! **row-major ONNX ↔ column-major cuBLAS** mapping. Everything about the
//! transpose / leading-dimension handling is documented and centralised here so
//! the kernel layer never has to reason about it.
//!
//! ## Why the `result` layer (not the `safe` layer)
//!
//! cudarc ships a `safe` `CudaBlasLT` wrapper, but its `Matmul<f32>` impl hard-
//! codes the compute type `CUBLAS_COMPUTE_32F_FAST_TF32` — i.e. it silently
//! rounds f32 inputs to **TF32** (10-bit mantissa). For an ONNX runtime whose
//! Phase-1 bar is *"GPU MatMul matches the CPU reference"*, a silent ~1e-3
//! relative error on the f32 path is a correctness regression, not an
//! optimisation. So we drop to cudarc's `result`/`sys` layer (explicitly
//! sanctioned by §15.2) and request full-precision `CUBLAS_COMPUTE_32F`. The
//! safe layer's RAII structure is mirrored here so descriptors are always freed,
//! even on the error path.
//!
//! ## The mapping (row-major → column-major), proved
//!
//! ONNX MatMul is **row-major**: we want `C[M,N] = A[M,K] · B[K,N]`, all stored
//! row-major. cuBLAS is **column-major**. The identity we exploit: a row-major
//! matrix `X[r,c]` with leading dim `c` occupies the *exact same bytes* as the
//! column-major matrix `Xᵀ[c,r]` with leading dim `c`. Therefore, reading our
//! row-major buffers as column-major matrices *for free*:
//!
//! * `A` row-major `[M,K]` **is** column-major `Aᵀ [K,M]` (ld = K)
//! * `B` row-major `[K,N]` **is** column-major `Bᵀ [N,K]` (ld = N)
//! * `C` row-major `[M,N]` **is** column-major `Cᵀ [N,M]` (ld = N)
//!
//! We want `Cᵀ`. And `Cᵀ = (A·B)ᵀ = Bᵀ · Aᵀ`. So a single **no-transpose**
//! column-major GEMM with the operands swapped produces exactly the bytes of
//! our row-major `C`:
//!
//! ```text
//!   cublas(op1 = B, op2 = A)  →  op1 · op2 = Bᵀ · Aᵀ = Cᵀ  ==  row-major C
//!   with cublas dims  m = N, n = M, k = K
//!   and leading dims  lda = N (op1=B),  ldb = K (op2=A),  ldc = N (C)
//! ```
//!
//! This is the same convention cudarc's own test uses, and it is unit-tested on
//! the GPU in `tests/matmul_gpu.rs`.
//!
//! Fused bias/activation epilogues use the descriptor's native
//! `CUBLASLT_MATMUL_DESC_EPILOGUE` and `BIAS_POINTER` attributes, so bias and
//! activation execute inside the selected GEMM kernel.

use core::ffi::c_int;
use std::ffi::c_void;

use cudarc::cublaslt::{result, sys};
use cudarc::driver::sys::CUdeviceptr;

use onnx_runtime_ep_api::{
    EpError, Result, WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_memory_governor::MemoryRole;

use crate::error::cublas_err;

/// Element type of a GEMM. Maps to a cuBLAS data type; the accumulate / scale
/// type is always f32 (see [`GemmDtype::compute_type`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmDtype {
    F32,
    F16,
    Bf16,
}

impl GemmDtype {
    fn data_type(self) -> sys::cudaDataType {
        match self {
            GemmDtype::F32 => sys::cudaDataType_t::CUDA_R_32F,
            GemmDtype::F16 => sys::cudaDataType_t::CUDA_R_16F,
            GemmDtype::Bf16 => sys::cudaDataType_t::CUDA_R_16BF,
        }
    }

    /// Full-precision f32 accumulation for every element type. For f16/bf16 this
    /// is the architecture-neutral mixed-precision mode: the cuBLASLt heuristic
    /// selects an algorithm supported by the detected device, without requesting
    /// an SM-specific tensor-core compute type. For f32 it is true IEEE fp32
    /// (NOT TF32 — see the module docs for why that matters).
    fn compute_type(self) -> sys::cublasComputeType_t {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}

/// An owned cuBLASLt handle. `Drop` frees it; `Send`/`Sync` mirror cudarc's own
/// `CudaBlasLT` (the handle is a context-independent library handle).
#[derive(Debug)]
pub struct CublasLt {
    handle: sys::cublasLtHandle_t,
}

// SAFETY: a cuBLASLt handle is not thread-affine; cudarc makes the same
// assertion for its `CudaBlasLT`. Concurrent *use* is still serialised by the
// per-execute descriptors and workspace we create below.
unsafe impl Send for CublasLt {}
unsafe impl Sync for CublasLt {}

impl CublasLt {
    /// Create a cuBLASLt handle (dlopen's `libcublasLt` on first use).
    pub fn new() -> Result<Self> {
        // Handle creation initializes cuBLASLt state on the device; see the
        // matmul sites.
        let _section = onnx_runtime_cuda_memory::capture_gate::synchronizing_section();
        let handle = result::create_handle().map_err(|e| cublas_err("cublasLtCreate", e))?;
        Ok(Self { handle })
    }
}

impl Drop for CublasLt {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: `handle` was produced by `create_handle` and is freed
            // exactly once here.
            unsafe {
                let _ = result::destroy_handle(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

/// RAII wrapper over a `cublasLtMatrixLayout_t` (freed on drop).
struct MatrixLayout(sys::cublasLtMatrixLayout_t);

impl MatrixLayout {
    fn new(dtype: sys::cudaDataType, rows: u64, cols: u64, ld: i64) -> Result<Self> {
        let h = result::create_matrix_layout(dtype, rows, cols, ld)
            .map_err(|e| cublas_err("cublasLtMatrixLayoutCreate", e))?;
        Ok(Self(h))
    }

    /// Attach strided-batch metadata (batch count + element stride between
    /// consecutive matrices in the batch).
    fn set_batch(&self, count: c_int, stride: i64) -> Result<()> {
        // SAFETY: `self.0` is a live layout; the attribute buffers point at
        // locals of the documented size for the whole call.
        unsafe {
            result::set_matrix_layout_attribute(
                self.0,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                (&count) as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            )
            .map_err(|e| cublas_err("set BATCH_COUNT", e))?;
            result::set_matrix_layout_attribute(
                self.0,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                (&stride) as *const i64 as *const c_void,
                std::mem::size_of::<i64>(),
            )
            .map_err(|e| cublas_err("set STRIDED_BATCH_OFFSET", e))?;
        }
        Ok(())
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        // SAFETY: single free of a live layout handle.
        unsafe {
            let _ = result::destroy_matrix_layout(self.0);
        }
    }
}

/// RAII wrapper over a `cublasLtMatmulDesc_t`.
struct MatmulDesc(sys::cublasLtMatmulDesc_t);

impl MatmulDesc {
    fn new(compute: sys::cublasComputeType_t, scale: sys::cudaDataType) -> Result<Self> {
        let h = result::create_matmul_desc(compute, scale)
            .map_err(|e| cublas_err("cublasLtMatmulDescCreate", e))?;
        Ok(Self(h))
    }

    fn set_epilogue(&self, epilogue: GemmEpilogue) -> Result<()> {
        let kind = epilogue.kind.as_cublas();
        let bias = epilogue.bias;
        // The bias address is execution data, not an algorithm-selection input.
        // Prepare-only planning has no tensor pointers, so a zero sentinel sets
        // the epilogue kind without a BIAS_POINTER; execution supplies the real
        // address through the same descriptor builder.
        unsafe {
            result::set_matmul_desc_attribute(
                self.0,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                (&kind) as *const sys::cublasLtEpilogue_t as *const c_void,
                std::mem::size_of::<sys::cublasLtEpilogue_t>(),
            )
            .map_err(|e| cublas_err("set MATMUL_DESC_EPILOGUE", e))?;
            if bias != 0 {
                result::set_matmul_desc_attribute(
                    self.0,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    (&bias) as *const CUdeviceptr as *const c_void,
                    std::mem::size_of::<CUdeviceptr>(),
                )
                .map_err(|e| cublas_err("set MATMUL_DESC_BIAS_POINTER", e))?;
            }
            Ok(())
        }
    }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        // SAFETY: single free of a live desc handle.
        unsafe {
            let _ = result::destroy_matmul_desc(self.0);
        }
    }
}

/// RAII wrapper over a `cublasLtMatmulPreference_t`.
struct MatmulPref(sys::cublasLtMatmulPreference_t);

impl MatmulPref {
    fn new(workspace_bytes: usize) -> Result<Self> {
        let h = result::create_matmul_pref()
            .map_err(|e| cublas_err("cublasLtMatmulPreferenceCreate", e))?;
        // SAFETY: `h` is live; the attribute buffer is a local `usize`.
        unsafe {
            result::set_matmul_pref_attribute(
                h,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&workspace_bytes) as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            )
            .map_err(|e| cublas_err("set MAX_WORKSPACE_BYTES", e))?;
        }
        Ok(Self(h))
    }
}

impl Drop for MatmulPref {
    fn drop(&mut self) {
        // SAFETY: single free of a live pref handle.
        unsafe {
            let _ = result::destroy_matmul_pref(self.0);
        }
    }
}

/// A batched, row-major GEMM request. All pointers are **device** pointers
/// (`CUdeviceptr`) into buffers this EP owns. Shapes are the logical ONNX
/// (row-major) shapes: `C[batch,M,N] = A[batch,M,K] · B[batch,K,N]`.
pub struct GemmParams {
    pub dtype: GemmDtype,
    pub a: CUdeviceptr,
    pub b: CUdeviceptr,
    pub c: CUdeviceptr,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// Number of independent matrices (1 for a plain 2-D GEMM).
    pub batch: usize,
    /// Element strides between A/B matrices. A zero stride broadcasts one
    /// matrix across the batch.
    pub a_batch_stride: usize,
    pub b_batch_stride: usize,
    /// Optional in-GEMM bias/activation epilogue.
    pub epilogue: Option<GemmEpilogue>,
}

/// A fixed-shape row-major strided-batched GEMM request used by canonical
/// `Einsum` contractions.
///
/// Logical execution is `C[batch,M,N] = A[batch,M,K] · B[batch,K,N]`. When a
/// transpose flag is set, the corresponding storage matrix has the reversed
/// shape (`A` is `[K,M]`, or `B` is `[N,K]`) and cuBLASLt applies the transpose
/// through its descriptor without materializing bytes.
pub struct StridedBatchedGemmParams {
    pub dtype: GemmDtype,
    pub a: CUdeviceptr,
    pub b: CUdeviceptr,
    pub c: CUdeviceptr,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub batch: usize,
    pub transpose_a: bool,
    pub transpose_b: bool,
    /// Element stride between stored A matrices. Zero broadcasts one matrix.
    pub a_batch_stride: usize,
    /// Element stride between stored B matrices. Zero broadcasts one matrix.
    pub b_batch_stride: usize,
}

/// cuBLASLt fused epilogue kind. All variants add a per-output-channel bias;
/// activation, when present, is evaluated by cuBLASLt before the result is
/// written to global memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmEpilogueKind {
    Bias,
    ReluBias,
    GeluBias,
}

impl GemmEpilogueKind {
    fn as_cublas(self) -> sys::cublasLtEpilogue_t {
        match self {
            Self::Bias => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
            Self::ReluBias => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
            Self::GeluBias => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
        }
    }
}

/// Device bias vector and the fused operation to apply to each GEMM output.
#[derive(Clone, Copy, Debug)]
pub struct GemmEpilogue {
    pub kind: GemmEpilogueKind,
    pub bias: CUdeviceptr,
}

/// Default cuBLASLt workspace performance budget. This is device-global scratch,
/// not per-block shared memory or an SM-specific hardware limit; cuBLASLt's
/// architecture-aware heuristic selects an algorithm using at most these bytes.
///
/// Phase 2b: pool this per-stream instead of allocating it per call.
pub const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
pub const WORKSPACE_ALIGNMENT: usize = 256;

/// Attribute exact cuBLASLt scratch to the one session-persistent shared slot.
///
/// Kernels on the runtime's single CUDA stream execute sequentially, so their
/// individual heuristic requirements merge as a peak rather than a sum.
pub const fn governed_workspace_requirement(bytes: usize) -> WorkspaceRequirement {
    if bytes == 0 {
        WorkspaceRequirement::NONE
    } else {
        WorkspaceRequirement {
            bytes: bytes as u64,
            alignment: WORKSPACE_ALIGNMENT,
            lifetime: WorkspaceLifetime::SessionPersistent,
            role: MemoryRole::Workspace { step_scoped: false },
        }
    }
}

struct PlannedMatmul {
    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
    c_layout: MatrixLayout,
    _desc: MatmulDesc,
    algo: sys::cublasLtMatmulAlgo_t,
    workspace_bytes: usize,
}

fn plan_gemm(handle: &CublasLt, p: &GemmParams) -> Result<PlannedMatmul> {
    if p.m == 0 || p.n == 0 || p.k == 0 || p.batch == 0 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep MatMul: degenerate GEMM dims M={} K={} N={} batch={}",
            p.m, p.k, p.n, p.batch
        )));
    }

    let dt = p.dtype.data_type();
    let (m, n, k) = (p.n as u64, p.m as u64, p.k as u64);
    let (lda, ldb, ldc) = (p.n as i64, p.k as i64, p.n as i64);
    let a_layout = MatrixLayout::new(dt, m, k, lda)?;
    let b_layout = MatrixLayout::new(dt, k, n, ldb)?;
    let c_layout = MatrixLayout::new(dt, m, n, ldc)?;

    if p.batch > 1 {
        let count = i32::try_from(p.batch).map_err(|_| {
            EpError::KernelFailed(format!("cuda_ep MatMul: batch {} exceeds i32", p.batch))
        })?;
        a_layout.set_batch(count, p.b_batch_stride as i64)?;
        b_layout.set_batch(count, p.a_batch_stride as i64)?;
        c_layout.set_batch(count, (p.m * p.n) as i64)?;
    }

    let desc = MatmulDesc::new(p.dtype.compute_type(), sys::cudaDataType_t::CUDA_R_32F)?;
    if let Some(epilogue) = p.epilogue {
        desc.set_epilogue(epilogue)?;
    }
    let pref = MatmulPref::new(WORKSPACE_BYTES)?;
    // SAFETY: all descriptor/layout handles are live for the duration of the call.
    let heuristic = unsafe {
        result::get_matmul_algo_heuristic(
            handle.handle,
            desc.0,
            a_layout.0,
            b_layout.0,
            c_layout.0,
            c_layout.0,
            pref.0,
        )
    }
    .map_err(|e| {
        cublas_err(
            &format!(
                "no cuBLASLt algorithm for MatMul M={} K={} N={} batch={} dtype={:?}",
                p.m, p.k, p.n, p.batch, p.dtype
            ),
            e,
        )
    })?;
    Ok(PlannedMatmul {
        a_layout,
        b_layout,
        c_layout,
        _desc: desc,
        algo: heuristic.algo,
        workspace_bytes: heuristic.workspaceSize,
    })
}

/// Exact scratch selected by the cuBLASLt heuristic under [`WORKSPACE_BYTES`].
///
/// The 32 MiB constant is a performance ceiling, not a requirement. Planning
/// and governed execution both call this same planner, then execution refuses a
/// short buffer before submitting the matmul.
pub fn gemm_workspace_bytes(handle: &CublasLt, p: &GemmParams) -> Result<usize> {
    Ok(plan_gemm(handle, p)?.workspace_bytes)
}

unsafe fn launch_planned_gemm(
    handle: &CublasLt,
    stream: cudarc::driver::sys::CUstream,
    p: &GemmParams,
    plan: &PlannedMatmul,
    workspace: CUdeviceptr,
) -> Result<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    // SAFETY: layouts/desc/algo are live; a/b/c/workspace are caller-guaranteed
    // live device allocations of the right size; stream is valid.
    unsafe {
        // cuBLASLt picks and runs an algorithm here, allocating its own
        // workspace and synchronizing internally on its own schedule. Those are
        // calls this crate never makes and so cannot gate individually; gating
        // the whole invocation is what keeps them out of another thread's CUDA
        // graph capture. See `onnx_runtime_cuda_memory::capture_gate`.
        let _section = onnx_runtime_cuda_memory::capture_gate::synchronizing_section();
        result::matmul(
            handle.handle,
            plan._desc.0,
            (&alpha) as *const f32 as *const c_void,
            (&beta) as *const f32 as *const c_void,
            p.b as *const c_void,
            plan.a_layout.0,
            p.a as *const c_void,
            plan.b_layout.0,
            p.c as *const c_void,
            plan.c_layout.0,
            p.c as *mut c_void,
            plan.c_layout.0,
            (&plan.algo) as *const sys::cublasLtMatmulAlgo_t,
            workspace as *mut c_void,
            plan.workspace_bytes,
            stream as sys::cudaStream_t,
        )
    }
    .map_err(|e| cublas_err("cublasLtMatmul", e))
}

fn governed_workspace_ptr(
    workspace: Option<WorkspaceView>,
    required: usize,
    op: &str,
) -> Result<CUdeviceptr> {
    if required == 0 {
        return Ok(0);
    }
    let workspace = workspace.ok_or_else(|| {
        EpError::KernelFailed(format!(
            "cuda_ep {op}: governed cuBLASLt workspace requires {required} bytes, but none was supplied"
        ))
    })?;
    if workspace.bytes() < required {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: governed cuBLASLt workspace requires {required} bytes, supplied {}",
            workspace.bytes()
        )));
    }
    Ok(workspace.ptr().0 as CUdeviceptr)
}

/// Execute `C = A · B` (row-major, optionally batched) on `stream` using the
/// column-major mapping documented at the top of this module.
///
/// # Safety
///
/// * `handle` must be a live cuBLASLt handle.
/// * `p.a`, `p.b`, `p.c` must be live device allocations large enough for all
///   matrices addressed by the supplied element strides and `p.dtype`.
/// * `workspace` must be a live device allocation of `workspace_bytes`.
/// * `stream` must be a valid CUDA stream; the owning context must be current on
///   the calling thread.
/// * `p.c` must not alias `p.a` or `p.b`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm(
    handle: &CublasLt,
    stream: cudarc::driver::sys::CUstream,
    p: &GemmParams,
    workspace: CUdeviceptr,
    workspace_bytes: usize,
) -> Result<()> {
    let plan = plan_gemm(handle, p)?;
    if workspace_bytes < plan.workspace_bytes || (plan.workspace_bytes != 0 && workspace == 0) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep MatMul: cuBLASLt selected {} workspace bytes, supplied {workspace_bytes}",
            plan.workspace_bytes
        )));
    }

    // SAFETY: forwarded from the caller after validating the selected size.
    unsafe { launch_planned_gemm(handle, stream, p, &plan, workspace) }
}

/// Governed [`gemm`] adapter consuming an executor-prepared shared workspace.
///
/// # Safety
///
/// The tensor-pointer and stream requirements are identical to [`gemm`].
pub unsafe fn governed_gemm(
    handle: &CublasLt,
    stream: cudarc::driver::sys::CUstream,
    p: &GemmParams,
    workspace: Option<WorkspaceView>,
    op: &str,
) -> Result<()> {
    let plan = plan_gemm(handle, p)?;
    let ptr = governed_workspace_ptr(workspace, plan.workspace_bytes, op)?;
    // SAFETY: forwarded from the caller; `ptr` covers the exact selected bytes.
    unsafe { launch_planned_gemm(handle, stream, p, &plan, ptr) }
}

/// A cuBLASLt plan (selected algorithm + its layouts/descriptor and workspace
/// requirement) for one **fixed** GEMM shape, selected once and reusable across
/// later launches — including CUDA-graph capture replays — with no further
/// heuristic query, allocation, or synchronization. The caller pins the shape
/// and supplies a persistent workspace of at least [`Self::workspace_bytes`]
/// (see the capture-safe dense path in `kernels::matmul`).
///
/// For a given shape the plan reproduces [`governed_gemm`] exactly: both select
/// the algorithm through the same [`plan_gemm`] heuristic, so caching changes
/// only *when* the algorithm is chosen, never the arithmetic.
pub struct CaptureGemmPlan(PlannedMatmul);

// SAFETY: cuBLASLt layout/descriptor handles are context-independent host
// objects; the algorithm is a plain value. Launches are serialized by the
// owning kernel's plan mutex (same rationale as `F32GemvPlan`).
unsafe impl Send for CaptureGemmPlan {}

impl CaptureGemmPlan {
    /// Workspace bytes the selected algorithm requires; the caller must supply a
    /// persistent device buffer of at least this size to [`Self::launch`].
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        self.0.workspace_bytes
    }

    /// Launch the planned GEMM for `p` (which must have the same shape/dtype the
    /// plan was selected for; only the `a`/`b`/`c` pointers may differ).
    ///
    /// # Safety
    ///
    /// Identical to [`gemm`]: `handle`/`stream` valid and current, `p.a`/`p.b`/
    /// `p.c` live device allocations for the plan's shape, `workspace` a live
    /// allocation of at least [`Self::workspace_bytes`] bytes (or 0 when that is
    /// 0), and `p.c` not aliasing `p.a`/`p.b`.
    pub unsafe fn launch(
        &self,
        handle: &CublasLt,
        stream: cudarc::driver::sys::CUstream,
        p: &GemmParams,
        workspace: CUdeviceptr,
    ) -> Result<()> {
        // SAFETY: forwarded from the caller.
        unsafe { launch_planned_gemm(handle, stream, p, &self.0, workspace) }
    }
}

/// Select (once) a reusable [`CaptureGemmPlan`] for `p`'s shape via the same
/// heuristic [`governed_gemm`] uses per call.
pub fn plan_capture_gemm(handle: &CublasLt, p: &GemmParams) -> Result<CaptureGemmPlan> {
    Ok(CaptureGemmPlan(plan_gemm(handle, p)?))
}

fn checked_layout_dim(value: usize, name: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep Einsum: cuBLASLt {name}={value} exceeds u64"
        ))
    })
}

fn checked_layout_stride(value: usize, name: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep Einsum: cuBLASLt {name}={value} exceeds i64"
        ))
    })
}

fn plan_strided_batched_gemm(
    handle: &CublasLt,
    p: &StridedBatchedGemmParams,
) -> Result<PlannedMatmul> {
    if p.m == 0 || p.n == 0 || p.k == 0 || p.batch == 0 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Einsum: degenerate cuBLASLt contraction M={} K={} N={} batch={}",
            p.m, p.k, p.n, p.batch
        )));
    }

    let dt = p.dtype.data_type();
    // Row-major C[M,N] is column-major C^T[N,M]. Therefore cuBLASLt computes
    // B^T · A^T with the operand order swapped. A row-major transposed storage
    // view is represented by toggling the matching descriptor operation.
    let (b_rows, b_cols, b_ld) = if p.transpose_b {
        (p.k, p.n, p.k)
    } else {
        (p.n, p.k, p.n)
    };
    let (a_rows, a_cols, a_ld) = if p.transpose_a {
        (p.m, p.k, p.m)
    } else {
        (p.k, p.m, p.k)
    };
    let b_layout = MatrixLayout::new(
        dt,
        checked_layout_dim(b_rows, "B rows")?,
        checked_layout_dim(b_cols, "B columns")?,
        checked_layout_stride(b_ld, "B leading dimension")?,
    )?;
    let a_layout = MatrixLayout::new(
        dt,
        checked_layout_dim(a_rows, "A rows")?,
        checked_layout_dim(a_cols, "A columns")?,
        checked_layout_stride(a_ld, "A leading dimension")?,
    )?;
    let c_layout = MatrixLayout::new(
        dt,
        checked_layout_dim(p.n, "C rows")?,
        checked_layout_dim(p.m, "C columns")?,
        checked_layout_stride(p.n, "C leading dimension")?,
    )?;

    if p.batch > 1 {
        let count = i32::try_from(p.batch).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum: cuBLASLt batch {} exceeds i32",
                p.batch
            ))
        })?;
        b_layout.set_batch(
            count,
            checked_layout_stride(p.b_batch_stride, "B batch stride")?,
        )?;
        a_layout.set_batch(
            count,
            checked_layout_stride(p.a_batch_stride, "A batch stride")?,
        )?;
        let c_stride = p.m.checked_mul(p.n).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep Einsum: output matrix stride overflows for M={} N={}",
                p.m, p.n
            ))
        })?;
        c_layout.set_batch(count, checked_layout_stride(c_stride, "C batch stride")?)?;
    }

    let desc = MatmulDesc::new(p.dtype.compute_type(), sys::cudaDataType_t::CUDA_R_32F)?;
    // `transa` belongs to the first cuBLAS operand (row-major B), and `transb`
    // to the second (row-major A), because the row-major mapping swaps them.
    desc.set_transpose(
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        p.transpose_b,
    )?;
    desc.set_transpose(
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        p.transpose_a,
    )?;
    let pref = MatmulPref::new(WORKSPACE_BYTES)?;
    // SAFETY: all descriptor/layout handles are live for the query.
    let heuristic = unsafe {
        result::get_matmul_algo_heuristic(
            handle.handle,
            desc.0,
            b_layout.0,
            a_layout.0,
            c_layout.0,
            c_layout.0,
            pref.0,
        )
    }
    .map_err(|e| {
        cublas_err(
            &format!(
                "no cuBLASLt algorithm for Einsum M={} K={} N={} batch={} \
                 transpose_a={} transpose_b={} dtype={:?}",
                p.m, p.k, p.n, p.batch, p.transpose_a, p.transpose_b, p.dtype
            ),
            e,
        )
    })?;
    Ok(PlannedMatmul {
        a_layout: b_layout,
        b_layout: a_layout,
        c_layout,
        _desc: desc,
        algo: heuristic.algo,
        workspace_bytes: heuristic.workspaceSize,
    })
}

/// Immutable cuBLASLt algorithm/layout selection for one canonical `Einsum`
/// contraction. Planning happens during warmup; later launches only update
/// tensor pointers and are legal during CUDA graph capture.
pub struct CaptureStridedBatchedGemmPlan(PlannedMatmul);

// SAFETY: identical to `CaptureGemmPlan`; the descriptor/layout handles are
// context-independent host objects and the owning kernel serializes launches.
unsafe impl Send for CaptureStridedBatchedGemmPlan {}

impl CaptureStridedBatchedGemmPlan {
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        self.0.workspace_bytes
    }

    /// Launch the warmed contraction. Only A/B/C addresses may differ from the
    /// request used to create this plan.
    ///
    /// # Safety
    ///
    /// A/B/C and workspace must be live device allocations covering the fixed
    /// shape and strides represented by this plan, and C must not overlap A/B.
    pub unsafe fn launch(
        &self,
        handle: &CublasLt,
        stream: cudarc::driver::sys::CUstream,
        p: &StridedBatchedGemmParams,
        workspace: CUdeviceptr,
    ) -> Result<()> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // SAFETY: forwarded from the caller; all plan objects remain live.
        unsafe {
            let _section = onnx_runtime_cuda_memory::capture_gate::synchronizing_section();
            result::matmul(
                handle.handle,
                self.0._desc.0,
                (&alpha) as *const f32 as *const c_void,
                (&beta) as *const f32 as *const c_void,
                p.b as *const c_void,
                self.0.a_layout.0,
                p.a as *const c_void,
                self.0.b_layout.0,
                p.c as *const c_void,
                self.0.c_layout.0,
                p.c as *mut c_void,
                self.0.c_layout.0,
                (&self.0.algo) as *const sys::cublasLtMatmulAlgo_t,
                workspace as *mut c_void,
                self.0.workspace_bytes,
                stream as sys::cudaStream_t,
            )
        }
        .map_err(|e| cublas_err("cublasLtMatmul Einsum", e))
    }
}

/// Select one reusable cuBLASLt plan for a canonical `Einsum` contraction.
pub fn plan_capture_strided_batched_gemm(
    handle: &CublasLt,
    p: &StridedBatchedGemmParams,
) -> Result<CaptureStridedBatchedGemmPlan> {
    Ok(CaptureStridedBatchedGemmPlan(plan_strided_batched_gemm(
        handle, p,
    )?))
}

/// A single (non-batched) **column-major, native cuBLAS** GEMM request:
/// `C = alpha · op(A) · op(B) + beta · C`, with all shapes and leading
/// dimensions expressed in cuBLAS's own column-major terms.
///
/// The plain-`MatMul` path in [`gemm`] realises row-major ONNX semantics via
/// the operand-swap identity and never needs an explicit transpose. The
/// attention kernel, however, forms `Q·Kᵀ` (one transposed operand) and `P·V`
/// (no transpose) directly, so it drives cuBLASLt at this lower, unambiguous
/// column-major level and computes the leading dims / transpose flags itself
/// (see `kernels::attention` for the row-major → column-major derivation of
/// each GEMM). `alpha` lets the QKᵀ stage fold in the softmax `scale` for free.
pub struct GemmEx {
    pub dtype: GemmDtype,
    /// Apply `opᵀ` to A (`CUBLAS_OP_T`) instead of `op` (`CUBLAS_OP_N`).
    pub transa: bool,
    pub transb: bool,
    /// cuBLAS column-major result dims: `C` is `m × n`, contraction `k`.
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: f32,
    pub beta: f32,
    pub a: CUdeviceptr,
    pub lda: usize,
    pub b: CUdeviceptr,
    pub ldb: usize,
    pub c: CUdeviceptr,
    pub ldc: usize,
    /// Optional in-GEMM bias/activation epilogue.
    pub epilogue: Option<GemmEpilogue>,
}

// cublasOperation_t is a plain C `enum` (4-byte int): CUBLAS_OP_N = 0, _T = 1.
// The cuBLASLt `sys` layer does not re-export it, so we pass the raw code.
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;

impl MatmulDesc {
    /// Set the `CUBLASLT_MATMUL_DESC_TRANSA` / `TRANSB` operation for an operand.
    fn set_transpose(
        &self,
        attr: sys::cublasLtMatmulDescAttributes_t,
        transpose: bool,
    ) -> Result<()> {
        let op: i32 = if transpose { CUBLAS_OP_T } else { CUBLAS_OP_N };
        // SAFETY: `self.0` is a live desc; the buffer is a local `i32` matching
        // the 4-byte `cublasOperation_t` the attribute expects.
        unsafe {
            result::set_matmul_desc_attribute(
                self.0,
                attr,
                (&op) as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
            )
            .map_err(|e| cublas_err("set MATMUL_DESC_TRANS", e))
        }
    }
}

fn plan_gemm_ex(handle: &CublasLt, p: &GemmEx) -> Result<PlannedMatmul> {
    if p.m == 0 || p.n == 0 || p.k == 0 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep attention GEMM: degenerate dims M={} N={} K={}",
            p.m, p.n, p.k
        )));
    }

    let dt = p.dtype.data_type();
    let (a_rows, a_cols) = if p.transa {
        (p.k as u64, p.m as u64)
    } else {
        (p.m as u64, p.k as u64)
    };
    let (b_rows, b_cols) = if p.transb {
        (p.n as u64, p.k as u64)
    } else {
        (p.k as u64, p.n as u64)
    };
    let a_layout = MatrixLayout::new(dt, a_rows, a_cols, p.lda as i64)?;
    let b_layout = MatrixLayout::new(dt, b_rows, b_cols, p.ldb as i64)?;
    let c_layout = MatrixLayout::new(dt, p.m as u64, p.n as u64, p.ldc as i64)?;
    let desc = MatmulDesc::new(p.dtype.compute_type(), sys::cudaDataType_t::CUDA_R_32F)?;
    desc.set_transpose(
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        p.transa,
    )?;
    desc.set_transpose(
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        p.transb,
    )?;
    if let Some(epilogue) = p.epilogue {
        desc.set_epilogue(epilogue)?;
    }
    let pref = MatmulPref::new(WORKSPACE_BYTES)?;
    // SAFETY: all descriptor/layout handles are live for the call.
    let heuristic = unsafe {
        result::get_matmul_algo_heuristic(
            handle.handle,
            desc.0,
            a_layout.0,
            b_layout.0,
            c_layout.0,
            c_layout.0,
            pref.0,
        )
    }
    .map_err(|e| {
        cublas_err(
            &format!(
                "no cuBLASLt algorithm for attention GEMM M={} N={} K={} transa={} transb={} dtype={:?}",
                p.m, p.n, p.k, p.transa, p.transb, p.dtype
            ),
            e,
        )
    })?;
    Ok(PlannedMatmul {
        a_layout,
        b_layout,
        c_layout,
        _desc: desc,
        algo: heuristic.algo,
        workspace_bytes: heuristic.workspaceSize,
    })
}

/// Exact scratch selected for a column-major GEMM under [`WORKSPACE_BYTES`].
pub fn gemm_ex_workspace_bytes(handle: &CublasLt, p: &GemmEx) -> Result<usize> {
    Ok(plan_gemm_ex(handle, p)?.workspace_bytes)
}

unsafe fn launch_planned_gemm_ex(
    handle: &CublasLt,
    stream: cudarc::driver::sys::CUstream,
    p: &GemmEx,
    plan: &PlannedMatmul,
    workspace: CUdeviceptr,
) -> Result<()> {
    let alpha = p.alpha;
    let beta = p.beta;
    // SAFETY: layouts/desc/algo live; a/b/c/workspace are caller-guaranteed
    // live device allocations of the right size; stream is valid.
    unsafe {
        // cuBLASLt picks and runs an algorithm here, allocating its own
        // workspace and synchronizing internally on its own schedule. Those are
        // calls this crate never makes and so cannot gate individually; gating
        // the whole invocation is what keeps them out of another thread's CUDA
        // graph capture. See `onnx_runtime_cuda_memory::capture_gate`.
        let _section = onnx_runtime_cuda_memory::capture_gate::synchronizing_section();
        result::matmul(
            handle.handle,
            plan._desc.0,
            (&alpha) as *const f32 as *const c_void,
            (&beta) as *const f32 as *const c_void,
            p.a as *const c_void,
            plan.a_layout.0,
            p.b as *const c_void,
            plan.b_layout.0,
            p.c as *const c_void,
            plan.c_layout.0,
            p.c as *mut c_void,
            plan.c_layout.0,
            (&plan.algo) as *const sys::cublasLtMatmulAlgo_t,
            workspace as *mut c_void,
            plan.workspace_bytes,
            stream as sys::cudaStream_t,
        )
    }
    .map_err(|e| cublas_err("cublasLtMatmul (attention)", e))
}

/// Execute one column-major `C = alpha·op(A)·op(B) + beta·C` on `stream`.
///
/// # Safety
///
/// * `handle` must be a live cuBLASLt handle.
/// * `p.a`, `p.b`, `p.c` must be live device allocations large enough for the
///   stated shapes / leading dims and `p.dtype`.
/// * `workspace` must be a live device allocation of `workspace_bytes`.
/// * `stream` must be valid and its owning context current on this thread.
/// * `p.c` must not alias `p.a` or `p.b`.
pub unsafe fn gemm_ex(
    handle: &CublasLt,
    stream: cudarc::driver::sys::CUstream,
    p: &GemmEx,
    workspace: CUdeviceptr,
    workspace_bytes: usize,
) -> Result<()> {
    let plan = plan_gemm_ex(handle, p)?;
    if workspace_bytes < plan.workspace_bytes || (plan.workspace_bytes != 0 && workspace == 0) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep attention GEMM: cuBLASLt selected {} workspace bytes, supplied {workspace_bytes}",
            plan.workspace_bytes
        )));
    }

    // SAFETY: forwarded from the caller after validating the selected size.
    unsafe { launch_planned_gemm_ex(handle, stream, p, &plan, workspace) }
}

/// Governed [`gemm_ex`] adapter consuming an executor-prepared shared workspace.
///
/// # Safety
///
/// The tensor-pointer and stream requirements are identical to [`gemm_ex`].
pub unsafe fn governed_gemm_ex(
    handle: &CublasLt,
    stream: cudarc::driver::sys::CUstream,
    p: &GemmEx,
    workspace: Option<WorkspaceView>,
    op: &str,
) -> Result<()> {
    let plan = plan_gemm_ex(handle, p)?;
    let ptr = governed_workspace_ptr(workspace, plan.workspace_bytes, op)?;
    // SAFETY: forwarded from the caller; `ptr` covers the exact selected bytes.
    unsafe { launch_planned_gemm_ex(handle, stream, p, &plan, ptr) }
}

#[cfg(test)]
mod raw_workspace_allocation_guard {
    use onnx_runtime_ep_api::DevicePtrMut;

    use super::*;

    #[test]
    fn exact_requirement_is_persistent_and_shortfall_is_deterministic() {
        let requirement = governed_workspace_requirement(96);
        assert_eq!(requirement.bytes, 96);
        assert_eq!(requirement.alignment, WORKSPACE_ALIGNMENT);
        assert_eq!(requirement.lifetime, WorkspaceLifetime::SessionPersistent);
        assert!(matches!(
            requirement.role,
            MemoryRole::Workspace { step_scoped: false }
        ));

        let short = WorkspaceView::new(DevicePtrMut(std::ptr::null_mut()), 95);
        let error = governed_workspace_ptr(Some(short), 96, "test")
            .expect_err("a short prepared slot must fail before cuBLASLt launch");
        assert!(format!("{error}").contains("requires 96 bytes, supplied 95"));
    }

    #[test]
    fn governed_gemm_sites_do_not_allocate_the_32_mib_ceiling_raw() {
        let sites = [
            ("fused_gemm.rs", include_str!("kernels/fused_gemm.rs")),
            ("gemm.rs", include_str!("kernels/gemm.rs")),
            ("matmul.rs", include_str!("kernels/matmul.rs")),
            ("matmul_nbits.rs", include_str!("kernels/matmul_nbits.rs")),
        ];
        for (name, source) in sites {
            let lines = source.lines().collect::<Vec<_>>();
            for (index, line) in lines.iter().enumerate() {
                if !line.contains("alloc_raw") {
                    continue;
                }
                let end = (index + 4).min(lines.len());
                let allocation = lines[index..end].join(" ");
                assert!(
                    !allocation.contains("WORKSPACE_BYTES"),
                    "{name} reintroduced a raw allocation of the cuBLASLt 32 MiB ceiling near line {}; route it through the governed shared workspace",
                    index + 1
                );
            }
            assert!(
                source.contains("fn workspace_requirement"),
                "{name} must report its cuBLASLt scratch during prepare-only planning"
            );
            assert!(
                source.contains("fn execute_with_workspace"),
                "{name} must consume the executor-prepared shared workspace"
            );
            assert!(
                source.contains("governed_gemm"),
                "{name} must use the shared cuBLASLt governed adapter"
            );
        }

        let bindings = include_str!("../../onnx-runtime-session/src/executor/bindings.rs");
        for op in [
            "MatMul",
            "Gemm",
            "MatMulNBits",
            "FusedMatMulBias",
            "FusedGemm",
        ] {
            assert!(
                bindings.contains(op),
                "{op} must remain in the centralized is_planned_workspace_node predicate"
            );
        }
    }
}

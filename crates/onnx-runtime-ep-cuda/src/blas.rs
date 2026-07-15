//! cuBLASLt GEMM plumbing for the CUDA EP (`docs/ORT2.md` §15.3).
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
//! ## Deferred (Phase 2b)
//!
//! Fused bias/activation epilogues (`CUBLASLT_EPILOGUE_BIAS_*`), FP8, GEMV auto-
//! tuning and FP8 are **not** wired here yet.

use core::ffi::c_int;
use std::ffi::c_void;

use cudarc::cublaslt::{result, sys};
use cudarc::driver::sys::CUdeviceptr;

use onnx_runtime_ep_api::{EpError, Result};

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
    /// is the standard "compute in fp32, store in fp16/bf16" path; for f32 it is
    /// true IEEE fp32 (NOT TF32 — see the module docs for why that matters).
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
}

/// Default cuBLASLt workspace. 32 MiB is NVIDIA's recommendation for Hopper
/// (SM90, our H100/H200 target); smaller GPUs simply use less of it.
///
/// Phase 2b: pool this per-stream instead of allocating it per call.
pub const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

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
    if p.m == 0 || p.n == 0 || p.k == 0 || p.batch == 0 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep MatMul: degenerate GEMM dims M={} K={} N={} batch={}",
            p.m, p.n, p.k, p.batch
        )));
    }

    let dt = p.dtype.data_type();
    let (m, n, k) = (p.n as u64, p.m as u64, p.k as u64); // swapped: see module docs
    let (lda, ldb, ldc) = (p.n as i64, p.k as i64, p.n as i64);

    // op1 = B [N,K] (ld=N), op2 = A [K,M] (ld=K), out = C [N,M] (ld=N), all col-major.
    let a_layout = MatrixLayout::new(dt, m, k, lda)?;
    let b_layout = MatrixLayout::new(dt, k, n, ldb)?;
    let c_layout = MatrixLayout::new(dt, m, n, ldc)?;

    if p.batch > 1 {
        let count = i32::try_from(p.batch).map_err(|_| {
            EpError::KernelFailed(format!("cuda_ep MatMul: batch {} exceeds i32", p.batch))
        })?;
        // Strides are in ELEMENTS between consecutive matrices in the batch.
        a_layout.set_batch(count, p.b_batch_stride as i64)?; // B matrices
        b_layout.set_batch(count, p.a_batch_stride as i64)?; // A matrices
        c_layout.set_batch(count, (p.m * p.n) as i64)?; // C matrices
    }

    // Full-precision fp32 accumulation; scale (alpha/beta) type is f32.
    let desc = MatmulDesc::new(p.dtype.compute_type(), sys::cudaDataType_t::CUDA_R_32F)?;
    // No transpose on either operand — the swap above already realises Cᵀ = Bᵀ·Aᵀ.

    let pref = MatmulPref::new(workspace_bytes)?;

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

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    // SAFETY: layouts/desc/algo are live; a/b/c/workspace are caller-guaranteed
    // live device allocations of the right size; stream is valid. op1=B, op2=A.
    unsafe {
        result::matmul(
            handle.handle,
            desc.0,
            (&alpha) as *const f32 as *const c_void,
            (&beta) as *const f32 as *const c_void,
            p.b as *const c_void, // op1 = B
            a_layout.0,
            p.a as *const c_void, // op2 = A
            b_layout.0,
            p.c as *const c_void, // C (input for beta*C; beta=0)
            c_layout.0,
            p.c as *mut c_void, // D (output)
            c_layout.0,
            (&heuristic.algo) as *const sys::cublasLtMatmulAlgo_t,
            workspace as *mut c_void,
            workspace_bytes,
            stream as sys::cudaStream_t,
        )
    }
    .map_err(|e| cublas_err("cublasLtMatmul", e))?;

    Ok(())
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
    if p.m == 0 || p.n == 0 || p.k == 0 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep attention GEMM: degenerate dims M={} N={} K={}",
            p.m, p.n, p.k
        )));
    }

    let dt = p.dtype.data_type();

    // Layout dims describe each operand **as stored** (before op): a
    // transposed operand is physically `k × m` (A) / `n × k` (B).
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

    let pref = MatmulPref::new(workspace_bytes)?;

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

    let alpha = p.alpha;
    let beta = p.beta;

    // SAFETY: layouts/desc/algo live; a/b/c/workspace are caller-guaranteed live
    // device allocations of the right size; stream valid; C aliases neither.
    unsafe {
        result::matmul(
            handle.handle,
            desc.0,
            (&alpha) as *const f32 as *const c_void,
            (&beta) as *const f32 as *const c_void,
            p.a as *const c_void,
            a_layout.0,
            p.b as *const c_void,
            b_layout.0,
            p.c as *const c_void,
            c_layout.0,
            p.c as *mut c_void,
            c_layout.0,
            (&heuristic.algo) as *const sys::cublasLtMatmulAlgo_t,
            workspace as *mut c_void,
            workspace_bytes,
            stream as sys::cudaStream_t,
        )
    }
    .map_err(|e| cublas_err("cublasLtMatmul (attention)", e))?;

    Ok(())
}

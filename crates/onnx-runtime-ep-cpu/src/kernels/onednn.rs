//! oneDNN (`dnnl_sgemm`) FFI backend for the 2-D tile f32 GEMM.
//!
//! Compiled only under the non-default `onednn` cargo feature. `build.rs`
//! cmake-builds oneDNN from the `third_party/onednn` submodule as a static
//! CPU-only library and bindgen-generates the C API bindings included below.
//!
//! oneDNN's `dnnl_sgemm` computes `C = alpha·op(A)·op(B) + beta·C`. Crucially,
//! oneDNN documents its gemm as **row-major** (unlike Fortran BLAS / cuBLAS),
//! so our row-major tensors map straight through with no operand swap. See
//! [`sgemm`].

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use onnx_runtime_ep_api::{EpError, Result};

mod ffi {
    include!(concat!(env!("OUT_DIR"), "/onednn_bindings.rs"));
}

/// Compute row-major `C[m,n] = Σ_k A[m,k]·B[k,n]` via oneDNN `dnnl_sgemm`
/// (overwrite semantics: `alpha = 1`, `beta = 0`).
///
/// `a` is `m*k` row-major, `b` is `k*n` row-major, `c` is `m*n` row-major.
///
/// Unlike Fortran BLAS / cuBLAS, oneDNN's `dnnl_sgemm` assumes **row-major**
/// storage (see `dnnl.h`: "The matrices are assumed to be stored in row-major
/// order"), so no `Cᵀ = Bᵀ·Aᵀ` operand swap is needed — our row-major operands
/// map straight through with `transa = transb = 'N'`, `lda = k`, `ldb = n`,
/// `ldc = n`. (Verified against the MatMul kernel tests.)
pub(crate) fn sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> Result<()> {
    // Guard the buffer/shape contract before crossing FFI (RULES #1: fail
    // closed on invalid input with an actionable message rather than reading
    // out of bounds inside C).
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(EpError::KernelFailed(format!(
            "MatMul(oneDNN): buffer/shape mismatch for {m}x{k} @ {k}x{n}: \
             expected A={} B={} C={} elements, got A={} B={} C={}. \
             This is an internal GEMM tiling bug — the dense buffers must match \
             the tile dimensions.",
            m * k,
            k * n,
            m * n,
            a.len(),
            b.len(),
            c.len(),
        )));
    }
    // dnnl_sgemm requires all of M, N, K >= 1; a degenerate tile is a no-op.
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }

    // Row-major mapping: A is m×k (lda=k), B is k×n (ldb=n), C is m×n (ldc=n).
    let (mm, nn, kk) = (m as ffi::dnnl_dim_t, n as ffi::dnnl_dim_t, k as ffi::dnnl_dim_t);
    let lda = k as ffi::dnnl_dim_t;
    let ldb = n as ffi::dnnl_dim_t;
    let ldc = n as ffi::dnnl_dim_t;

    // SAFETY: transa/transb are the ASCII bytes oneDNN documents; the pointers
    // reference row-major buffers whose lengths were validated above to be
    // exactly m*k / k*n / m*n elements, so every access dnnl_sgemm makes with
    // (mm,nn,kk,lda,ldb,ldc) is in bounds. No Rust value crosses the boundary by
    // reference beyond these slices, and dnnl_sgemm does not unwind.
    let status = unsafe {
        ffi::dnnl_sgemm(
            b'N' as core::ffi::c_char,
            b'N' as core::ffi::c_char,
            mm,
            nn,
            kk,
            1.0,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            0.0,
            c.as_mut_ptr(),
            ldc,
        )
    };

    if status != ffi::dnnl_status_t_dnnl_success {
        return Err(EpError::KernelFailed(format!(
            "MatMul(oneDNN): dnnl_sgemm failed with status {status} for {m}x{k} @ {k}x{n}. \
             Rebuild the `onednn` feature (cmake static CPU build) or run the default \
             (Generic) CPU backend, which needs no external library."
        )));
    }
    Ok(())
}

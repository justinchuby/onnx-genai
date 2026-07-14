//! Error mapping from cudarc's driver / cuBLASLt errors into the shared
//! [`onnx_runtime_ep_api::EpError`] vocabulary.
//!
//! ## KEY PROJECT RULE — actionable errors (`docs/ORT2.md` §15.1)
//!
//! Every failure on the dispatch path is turned into an [`EpError`] that states
//! **what** failed, **why**, and — for unsupported cases — that it is a
//! *CUDA-EP Phase-2a* limitation with the concrete op / shape / dtype in hand.
//! There are **no bare panics** on the execution path.

use onnx_runtime_ep_api::EpError;

/// Wrap a cudarc **driver** error (`DriverError`) with actionable context.
pub(crate) fn driver_err(context: &str, e: cudarc::driver::DriverError) -> EpError {
    EpError::KernelFailed(format!("cuda_ep: {context}: CUDA driver error: {e:?}"))
}

/// Wrap a cudarc **cuBLASLt** error (`CublasError`) with actionable context.
pub(crate) fn cublas_err(context: &str, e: cudarc::cublaslt::result::CublasError) -> EpError {
    EpError::KernelFailed(format!("cuda_ep: {context}: cuBLASLt error: {e:?}"))
}

/// Build a "not implemented in this slice" error for an op/dtype/rank the
/// CUDA EP does **not** yet cover. The message is deliberately explicit that
/// this is Phase-2a scope, so callers get an actionable next step rather than a
/// silent fallback or a panic.
pub(crate) fn not_implemented(what: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!(
        "cuda_ep: {what} is not implemented in CUDA EP Phase 2a \
         (cudarc + cuBLASLt GEMM only). Custom fused kernels (norms, RoPE, \
         softmax, attention), NVRTC elementwise, and FP8 are deferred to a \
         later slice; run this op on the CPU EP for now."
    ))
}

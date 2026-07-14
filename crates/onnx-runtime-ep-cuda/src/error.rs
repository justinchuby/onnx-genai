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

/// Wrap a cudarc **cuDNN** error with actionable context.
pub(crate) fn cudnn_err(context: &str, e: cudarc::cudnn::CudnnError) -> EpError {
    EpError::KernelFailed(format!("cuda_ep: {context}: cuDNN error: {e:?}"))
}

/// Report that the dynamically-loaded cuDNN runtime is unavailable.
pub(crate) fn cudnn_unavailable() -> EpError {
    EpError::KernelFailed(
        "cuda_ep: cuDNN (libcudnn.so.9 / cudnn64_9.dll) was not found at runtime. \
         Install it with 'pip install nvidia-cudnn-cu13' or \
         'conda install -c nvidia cudnn', or add the cuDNN library directory to \
         the platform library search path."
            .into(),
    )
}

/// Wrap a cudarc **NVRTC** compile error with actionable context. NVRTC failures
/// carry the compiler log, which is the single most useful artefact for fixing a
/// bad kernel source, so it is surfaced verbatim (RULES.md #1: what/why/how).
pub(crate) fn nvrtc_err(context: &str, e: cudarc::nvrtc::CompileError) -> EpError {
    let detail = match &e {
        cudarc::nvrtc::CompileError::CompileError { nvrtc, log, .. } => {
            format!(
                "NVRTC compilation failed ({nvrtc:?}); compiler log:\n{}",
                log.to_string_lossy()
            )
        }
        other => format!("NVRTC error: {other:?}"),
    };
    EpError::KernelFailed(format!("cuda_ep: {context}: {detail}"))
}

/// Build a "not implemented on CUDA yet" error for an op/dtype/rank/layout the
/// CUDA EP does **not** cover in the current slice. The message is deliberately
/// explicit and actionable (RULES.md #1: what/why/how) — it names the missing
/// case and points at the coverage roadmap and the CPU-EP fallback, rather than
/// silently falling back or panicking.
pub(crate) fn not_implemented(what: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!(
        "cuda_ep: {what} is not yet implemented on the CUDA EP. See \
         docs/CUDA_COVERAGE.md for the op → backend roadmap; this op/case can run \
         on the CPU EP in the meantime."
    ))
}

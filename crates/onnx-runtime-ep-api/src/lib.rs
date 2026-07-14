//! # `onnx-runtime-ep-api`
//!
//! The Execution Provider (EP) interface for the ORT 2.0 runtime
//! (see `docs/ORT2.md` §4). Every backend — CPU, CUDA, MLX, or a legacy ORT
//! plugin loaded via `dlopen` — implements the same [`ExecutionProvider`]
//! trait; only the loading mechanism differs.
//!
//! This is a **Phase 1 skeleton**: trait and type *signatures* are real and
//! reference the [`onnx_runtime_ir`] contract, but method bodies live in the
//! concrete EP crates (e.g. `onnx-runtime-ep-cpu`). Zero-copy tensor views and
//! the ORT ABI bridge require `unsafe` FFI and are stubbed here.
//!
//! ## Modules
//! * [`provider`] — [`ExecutionProvider`], [`EpConfig`], device buffers/fences.
//! * [`kernel`] — [`Kernel`] trait, [`KernelMatch`], [`Cost`].
//! * [`registry`] — [`OpRegistry`], [`OpKey`], [`KernelFactory`], [`EpRegistry`].
//! * [`tensor`] — [`TensorView`] / [`TensorMut`] zero-copy device views.
//! * [`abi`] — ORT graph ABI bridge for legacy plugin EPs (Phase 2).

pub mod abi;
pub mod kernel;
pub mod provider;
pub mod registry;
pub mod tensor;

pub use error::{EpError, Result};
pub use kernel::{Cost, Kernel, KernelMatch};
pub use provider::{
    DeviceBuffer, EpConfig, EpId, ExecutionProvider, Fence, OptimizerPass, OrtPluginExport,
};
pub use registry::{EpRegistry, KernelFactory, OpKey, OpRegistry};
pub use tensor::{DevicePtr, DevicePtrMut, TensorMut, TensorView};

// Re-export the device vocabulary from the IR so EP authors have one import.
pub use onnx_runtime_ir::{DeviceId, DeviceType};

mod error {
    use std::path::PathBuf;

    /// A convenience `Result` alias for EP operations.
    pub type Result<T> = std::result::Result<T, EpError>;

    /// Errors produced by execution providers and kernels (subset of the
    /// runtime top-level `Error`, `docs/ORT2.md` §22).
    #[derive(Debug, thiserror::Error)]
    pub enum EpError {
        #[error("no EP supports op {op_type} on any available device")]
        NoEpForOp { op_type: String },

        #[error("kernel execution failed: {0}")]
        KernelFailed(String),

        #[error("EP panicked during execution")]
        EpPanicked,

        #[error("EP plugin load failed: {path}: {reason}")]
        EpLoadFailed { path: PathBuf, reason: String },

        #[error("device OOM: requested {requested} bytes, available {available}")]
        OutOfMemory { requested: usize, available: usize },

        #[error("allocation alignment mismatch")]
        AlignmentError,

        #[error("EP not initialized")]
        NotInitialized,

        #[error(transparent)]
        Ir(#[from] onnx_runtime_ir::IrError),
    }
}

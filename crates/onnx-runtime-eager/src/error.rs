//! Error types for the eager execution engine (`docs/EAGER.md` §9, §10).
//!
//! Mirrors the `thiserror`-based `error.rs` style of the sibling runtime crates
//! (e.g. `onnx-runtime-ep-api`, `onnx-runtime-shape-inference`): a crate-local
//! [`Result`] alias plus one enum that wraps the lower-layer errors it can
//! surface (ep-api, ir, shape-inference).

use onnx_runtime_ir::DeviceId;

/// A convenience `Result` alias for eager-dispatch operations.
pub type Result<T> = std::result::Result<T, EagerError>;

/// Errors produced while dispatching a single op in eager mode.
#[derive(Debug, thiserror::Error)]
pub enum EagerError {
    /// No EP registered for the resolved device has a kernel for this op at the
    /// requested opset (`docs/EAGER.md` §10.1 step 4).
    #[error("no kernel for op '{op_type}' (domain {domain:?}) on device {device:?}")]
    NoKernel {
        op_type: String,
        domain: String,
        device: DeviceId,
    },

    /// Inputs live on more than one device. Eager mode never transfers
    /// implicitly (`docs/EAGER.md` §1.6 / Design Decision): the caller must
    /// `.to(device)` first.
    #[error("mixed-device inputs {devices:?}: {hint}")]
    MixedDeviceInputs {
        devices: Vec<DeviceId>,
        hint: String,
    },

    /// The resolved device has no execution provider registered in the context.
    #[error("no execution provider registered for device {0:?}")]
    NoEpForDevice(DeviceId),

    /// Per-op output shape/dtype inference is missing or could not resolve to a
    /// concrete, allocatable shape (`docs/EAGER.md` §9). The kernel-provided
    /// inference fallback (§9.2) is DEFERRED, so an unresolved output is an
    /// error rather than a fall-through.
    #[error("shape inference failed for op '{op_type}' (domain {domain:?}): {reason}")]
    ShapeInference {
        op_type: String,
        domain: String,
        reason: String,
    },

    /// A lower-layer execution-provider / kernel error (allocation, execution).
    #[error("execution provider error: {0}")]
    Kernel(#[from] onnx_runtime_ep_api::EpError),

    /// An error escaping the shape-inference engine itself.
    #[error("shape-inference engine error: {0}")]
    ShapeInferEngine(#[from] onnx_runtime_shape_inference::ShapeInferError),

    /// An error escaping the IR layer.
    #[error("ir error: {0}")]
    Ir(#[from] onnx_runtime_ir::IrError),
}

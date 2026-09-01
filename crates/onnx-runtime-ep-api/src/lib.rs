//! # `onnx-runtime-ep-api`
//!
//! The Execution Provider (EP) interface for the ORT 2.0 runtime
//! (see `docs/architecture/ORT2.md` §4). Every backend — CPU, CUDA, MLX, or a legacy ORT
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
//! * [`epcontext`] — [`EpContext`] runtime form + `source`-keyed [`EpContextRegistry`] (§55).
//! * [`tensor`] — [`TensorView`] / [`TensorMut`] zero-copy device views.
//! * [`weight`] — capability-negotiated lazy [`WeightHandle`] delivery.
//! * [`abi`] — ORT graph ABI bridge for legacy plugin EPs (Phase 2).
//! * [`host_parallel`] — the host runtime's own thread pool, borrowed per compute call.

// This crate hands raw pointers across an FFI boundary, so keeping a Rust
// value alive past its scope is a recurring temptation. It is almost never the
// answer: a pointer that outlives its owner needs the *owner* moved somewhere
// longer-lived, not its destructor skipped.
//
// `HostNode::fused` learned this the hard way. It built value infos, took raw
// pointers into them, and called `mem::forget` so the pointers could not
// dangle -- reasoning that the leak was "tiny". It was tiny per call, and the
// call happens once per session, so a long-lived server leaked forever. The
// fix was to give the node ownership of the boxes its pointers reference,
// which is exactly what `HostGraph` a hundred lines away had always done.
#![deny(clippy::mem_forget)]

pub mod abi;
pub mod epcontext;
pub mod host_parallel;
pub mod kernel;
pub mod provider;
pub mod registry;
pub mod tensor;
pub mod weight;

pub use abi::{LegacyOrtEp, PluginCompiledKernel, PluginExecutionPlan, SubgraphClaim};
pub use epcontext::{EpContext, EpContextRegistry, build_ep_context_registry};
pub use error::{EpError, Result};
pub use host_parallel::HostParallel;
pub use kernel::{
    ARG_BYTES, ARG_DEVICE, ARG_FLOPS, ARG_KERNEL_VARIANT, ARG_KERNEL_VARIANT_REASON,
    BlockQuantizedMoeTraffic, CAT_KERNEL_WORKER, CaptureSupport, Cost, DeviceGraphResource, Kernel,
    KernelConstantInput, KernelInput, KernelMatch, KernelSizedOutput, KernelSizedOutputMetadata,
    KernelSizedOutputPolicy, KernelVariantSelection, TensorMetadata, ViewOutput, WorkspaceLifetime,
    WorkspaceRequirement, WorkspaceView, kernel_variant_tracing_enabled, kernel_worker_span,
    record_kernel_metrics, record_kernel_variant_selection, record_kernel_variant_stage_selection,
    structural_input_bytes,
};
pub use onnx_runtime_optimizer::OptimizationPass as OptimizerPass;
pub use provider::{
    ArgmaxTieBreak, BoundBufferOwnership, CaptureRegionShapeStatus, DeviceBuffer, DeviceGraphOwner,
    DeviceGraphSlot, DeviceGraphToken, DeviceValidationOwner, DeviceValidationRegistration,
    DeviceValidationToken, EpConfig, EpId, ExecutionProvider, ExecutorArtifactGeneration,
    ExecutorArtifactPending, ExecutorArtifactPolicy, ExecutorArtifactProviderId,
    ExecutorArtifactReadinessEpoch, ExecutorArtifactReport, ExecutorArtifactState,
    ExecutorInstanceId, ExecutorKernelScope, ExecutorRouteResidencyConfig, Fence,
    HostToDeviceCopier, RawDeviceAllocationSiteStats, SealedDeviceAllocation,
    StructuralCaptureDecline, WorkspaceAllocation,
};
pub use registry::{EpRegistry, KernelFactory, OpKey, OpRegistry};
pub use tensor::{
    DevicePtr, DevicePtrMut, ExternalMmapRegion, TensorBacking, TensorMut, TensorView,
};
pub use weight::{
    AdmissionPolicyInput, EvictionClass, ExecutionProviderCapabilities, ExpertWeightGroup,
    LazyDeviceWeightBinder, LazyWeight, LazyWeightBoundary, LazyWeightCandidate, MmapRegionSource,
    NXRT_WEIGHT_PAGING_CAPABILITY, NegotiatedWeight, PagedWeight, Phase3aHostOnlyBinder,
    ResidencyDecision, ResidencyDegradationReason, ResidencyPlan, ResidencyPolicy,
    ResidencyPolicyInput, ResidencyResizeOutcome, ResidencyResizePlan, ResidencyResizeRequest,
    ResidentWeight, ResidentWeightMaterializer, ResizeDirection, ResizeRejection, ResizeSafePoint,
    RoutedResidencyCoverage, RoutedResidencyGuardHandle, RoutedResidencyProof,
    RoutedResidencyRequirement, StaticProfileResidencyPolicy, WeightHandle, WeightHandleError,
    WholeBankReason, WholeBankResidentPolicy, expert_weight_groups, lazy_weight_candidates,
    plan_residency, plan_resize, prove_routed_residency,
};

// Re-export the device vocabulary from the IR so EP authors have one import.
pub use onnx_runtime_ir::{DeviceId, DeviceType};

mod error {
    use std::path::PathBuf;

    /// A convenience `Result` alias for EP operations.
    pub type Result<T> = std::result::Result<T, EpError>;

    /// Errors produced by execution providers and kernels (subset of the
    /// runtime top-level `Error`, `docs/architecture/ORT2.md` §22).
    #[derive(Debug, thiserror::Error)]
    pub enum EpError {
        #[error("no EP handler for {domain}::{op_type} at opset {opset} on any available device")]
        NoEpForOp {
            domain: String,
            op_type: String,
            opset: u64,
        },

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

        #[error("invalid tensor view: {reason}")]
        InvalidTensorView { reason: String },

        #[error("EP not initialized")]
        NotInitialized,

        #[error("no EP is registered for EPContext source key {source_key:?}")]
        NoEpForContext { source_key: Option<String> },

        #[error("EP {ep} does not support EPContext save/load")]
        UnsupportedContext { ep: String },

        #[error(
            "duplicate EPContext source key {source_key:?}: already registered to {existing:?}, \
             cannot re-register to {new:?}"
        )]
        DuplicateContextSource {
            source_key: String,
            existing: crate::provider::EpId,
            new: crate::provider::EpId,
        },

        #[error(transparent)]
        Ir(#[from] onnx_runtime_ir::IrError),

        #[error(transparent)]
        WeightHandle(#[from] crate::weight::WeightHandleError),

        #[error("{0}")]
        Memory(#[from] onnx_runtime_memory_governor::MemoryError),
    }
}

//! Compatibility re-exports for the shared executor/EP weight-handle seam.

pub use onnx_runtime_ep_api::{
    ExecutionProviderCapabilities, LazyDeviceWeightBinder, LazyWeight, LazyWeightBoundary,
    NXRT_WEIGHT_PAGING_CAPABILITY, NegotiatedWeight, Phase3aHostOnlyBinder, ResidentWeight,
    ResidentWeightMaterializer, WeightHandle, WeightHandleError,
};

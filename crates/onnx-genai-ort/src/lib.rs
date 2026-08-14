//! Safe Rust wrapper over the ONNX Runtime C API.
//!
//! This provides a thin, safe layer over ORT's C API, giving us:
//! - Full control over IoBinding (for zero-copy KV cache passing)
//! - Latest ORT features (opset 24, tensor scatter)
//! - Support for all Execution Providers (CUDA, DirectML, QNN, CoreML, etc.)
//!
//! Design: reference the `ort` crate (pyke) for patterns, but use latest ORT directly.

#[cfg(all(
    feature = "cuda",
    not(any(
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000"
    ))
))]
compile_error!(
    "onnx-genai CUDA build: no CUDA version selected. Enable exactly one of cuda-12060 | cuda-12080 | cuda-12090 | cuda-13000."
);

#[cfg(any(
    all(feature = "cuda-12060", feature = "cuda-12080"),
    all(feature = "cuda-12060", feature = "cuda-12090"),
    all(feature = "cuda-12060", feature = "cuda-13000"),
    all(feature = "cuda-12080", feature = "cuda-12090"),
    all(feature = "cuda-12080", feature = "cuda-13000"),
    all(feature = "cuda-12090", feature = "cuda-13000")
))]
compile_error!(
    "onnx-genai CUDA build: multiple CUDA versions selected; cudarc bindings cannot compile with more than one. Enable exactly one of cuda-12060 | cuda-12080 | cuda-12090 | cuda-13000 (and set default-features = false on inter-crate deps if you override the default)."
);

pub mod allocator;
pub mod binding;
pub mod chat_template;
pub mod component;
#[cfg(feature = "cuda")]
pub mod cuda_rt;
pub mod decode;
pub mod decode_contract;
#[cfg(feature = "cuda")]
pub(crate) mod device_sampler;
pub mod eagle3;
pub mod env;
pub mod error;
#[cfg(feature = "cuda")]
pub mod fused_argmax;
pub mod governed_allocator;
pub mod io_roles;
pub mod loader;
pub mod mtp;
mod pipeline_admission;
pub mod profile;
pub mod runtime_capability;
pub mod session;
pub mod shared_kv_proposer;
pub mod tokenizer;
pub mod value;

pub use allocator::{Allocator, AllocatorType, MemoryInfo, MemoryType};
pub use binding::IoBinding;
pub use chat_template::{ChatMessage, ChatRole, ChatTemplate};
pub use component::{OrtComponentSession, OrtComponentSessionRef};
pub use decode::{
    BatchedDecodeSession, BatchedSharedBufferDecodeSession, BatchedStaticCacheDecodeSession,
    DecodeKvMode, DecodeSession, DecodeSessionOptions, DeviceSampleParams,
    SharedBufferBatchOptions, StaticCacheBindingMode, StaticCacheBufferInfo,
    StaticCacheDecodeOptions, StaticCacheDecodeSession, StaticCacheSignature,
};
pub use eagle3::{
    Eagle3DecodeOptions, Eagle3DecodeSession, Eagle3DraftKvMode, Eagle3HeadSignature,
    Eagle3StepOutput,
};
pub use env::Environment;
pub use error::{OrtError, Result};
pub use loader::{
    ModelDirectory, PipelineModelDirectory, PipelineModels, PipelineTokenizerPaths,
    graph_io_from_model_path, graph_io_from_model_path_for_kv_pairs,
    graph_io_from_model_path_for_names, model_weight_bytes,
};
pub use mtp::{
    MtpDecodeOptions, MtpDecodeSession, MtpDraftKvMode, MtpHeadSignature, MtpStepOutput,
};
pub use onnx_genai_metadata::{
    ComponentDataType, ComponentError, ComponentIo, ComponentSession, ComponentTensor,
};
pub use onnx_genai_metadata::{
    ProposalType, SpeculatorConfig, SpeculatorConfigSource, SpeculatorDescriptor,
    SpeculatorProposerKind, SpeculatorProposerStatus, SpeculatorVerifier, detect_speculator,
};
pub use onnx_genai_runtime_config::EpSelection;
pub use onnx_model_package::SelectionRequest as ModelPackageSelection;
pub use session::{
    CudaAttentionMode, EpCapabilities, GraphIo, GraphIoMetadata, HardwareKind, ResolvedEp,
    RunPhaseError, Session, SessionOptions, TensorInfo, USE_ENV_ALLOCATORS,
    available_execution_providers, capability, ep_selection, resolve_execution_provider,
    selectable_execution_providers,
};
pub use shared_kv_proposer::{
    SharedKvInput, SharedKvProposerSession, SharedKvProposerSignature, SharedKvProposerStepOutput,
    SharedKvSpec,
};
pub use tokenizer::Tokenizer;
pub use value::{DataType, Value};

/// Human-readable report of the ONNX Runtime shared library selected for this process.
///
/// Calling this resolves ORT if it has not already been loaded, so it is suitable
/// for CLI diagnostics such as `onnx-genai version`.
#[must_use]
pub fn onnxruntime_library_report() -> String {
    let previous_report_setting = onnx_genai_ort_sys::set_ort_selection_report_enabled(false);
    let load_error = onnx_genai_ort_sys::ort_load_error();
    onnx_genai_ort_sys::set_ort_selection_report_enabled(previous_report_setting);
    match load_error {
        Some(error) => format!("failed to load ({error})"),
        None => {
            let path = onnx_genai_ort_sys::loaded_ort_path()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unknown path>".to_owned());
            let version = onnx_genai_ort_sys::loaded_ort_version()
                .unwrap_or_else(|| "unknown version".to_owned());
            let api = onnx_genai_ort_sys::loaded_ort_api_version()
                .map_or_else(|| "unknown".to_owned(), |api| api.to_string());
            let reason = onnx_genai_ort_sys::loaded_ort_reason()
                .unwrap_or_else(|| "dynamic loader default search path".to_owned());
            format!("{version} (API {api}) from {path} ({reason})")
        }
    }
}

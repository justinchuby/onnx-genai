//! Safe Rust wrapper over the ONNX Runtime C API.
//!
//! This provides a thin, safe layer over ORT's C API, giving us:
//! - Full control over IoBinding (for zero-copy KV cache passing)
//! - Latest ORT features (opset 24, tensor scatter)
//! - Support for all Execution Providers (CUDA, DirectML, QNN, CoreML, etc.)
//!
//! Design: reference the `ort` crate (pyke) for patterns, but use latest ORT directly.

pub mod allocator;
pub mod binding;
pub mod chat_template;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_rt;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_argmax;
pub mod decode;
pub mod eagle3;
pub mod env;
pub mod error;
pub mod shared_kv_proposer;
pub mod loader;
pub mod mtp;
pub mod profile;
pub mod session;
pub mod tokenizer;
pub mod value;

pub use allocator::{Allocator, AllocatorType, MemoryInfo, MemoryType};
pub use binding::IoBinding;
pub use chat_template::{ChatMessage, ChatRole, ChatTemplate};
pub use decode::{
    BatchedDecodeSession, BatchedSharedBufferDecodeSession, BatchedStaticCacheDecodeSession,
    DecodeKvMode, DecodeSession, DecodeSessionOptions, SharedBufferBatchOptions,
    StaticCacheBindingMode, StaticCacheBufferInfo, StaticCacheDecodeOptions,
    StaticCacheDecodeSession, StaticCacheSignature,
};
pub use eagle3::{
    Eagle3DecodeOptions, Eagle3DecodeSession, Eagle3DraftKvMode, Eagle3HeadSignature,
    Eagle3StepOutput,
};
pub use env::Environment;
pub use error::{OrtError, Result};
pub use shared_kv_proposer::{
    SharedKvInput, SharedKvProposerSession, SharedKvProposerSignature, SharedKvProposerStepOutput,
    SharedKvSpec,
};
pub use loader::{ModelDirectory, PipelineModelDirectory, PipelineModels, PipelineTokenizerPaths};
pub use mtp::{
    MtpDecodeOptions, MtpDecodeSession, MtpDraftKvMode, MtpHeadSignature, MtpStepOutput,
};
pub use onnx_genai_metadata::{
    ProposalType, SpeculatorConfig, SpeculatorConfigSource, SpeculatorDescriptor,
    SpeculatorProposerKind, SpeculatorProposerStatus, SpeculatorVerifier, detect_speculator,
};
pub use session::{
    ExecutionProvider, Session, SessionOptions, TensorInfo, available_execution_providers,
};
pub use tokenizer::Tokenizer;
pub use value::{DataType, Value};

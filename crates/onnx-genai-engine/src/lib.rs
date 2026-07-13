//! Generation engine.
//!
//! The core orchestrator that ties together:
//! - ORT sessions (model execution)
//! - KV cache manager (memory)
//! - Scheduler (batching)
//! - Logit processors (sampling)
//! - Speculative decoding (acceleration)

pub(crate) mod batched;
pub mod config;
pub(crate) mod connector_bridge;
pub(crate) mod decode;
pub(crate) mod decode_loop;
pub mod embedding;
pub mod engine;
pub mod fim;
pub(crate) mod kv_bridge;
pub mod logits;
pub mod pipeline;
pub(crate) mod processors;
pub mod sampling;
pub(crate) mod session;
pub mod speculative;

pub use batched::{ContinuousBatchEvent, ContinuousBatchHandle, ContinuousBatchManager};
pub use embedding::{EmbeddingOptions, EmbeddingPooling};
pub use engine::{
    Eagle3Config, Engine, EngineConfig, FinishReason, GenerateConstraint,
    GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult, GenerateToken,
    GenerateTokenCallback, KvConnectorBackend, KvConnectorConfig, MtpConfig,
    PrioritizedGenerateRequest, PrioritizedGenerateResult,
    ScheduledGenerateArrival, SessionId, SharedKvBinding, SharedKvProposerConfig, SpeculativeMode,
    TokenLogprob,
};
pub use connector_bridge::{ConnectorLookupOutcome, ConnectorStats};
pub use onnx_genai_kv::{CachePriority, KvDType, LocalTieredConfig};
pub use fim::{FimConfig, FimFormat};
pub use logits::{
    Constraint, ConstraintProcessor, JsonConstraint, LogitProcessor, ProcessorChain,
    ProcessorChainBuilder, ProcessorContext, ProcessorSignal, StopSequence, TokenId,
};
pub use pipeline::{PipelineEngine, PipelineGenerateRequest, PipelineTensors};
pub use sampling::{CategoricalSampler, GreedySampler, Sampler};
pub use speculative::{
    Eagle3Proposer, LinearEmbedder, LinearLmHead, LmHead, MtpProposer, NgramProposer,
    SpeculativeAcceptContext, SpeculativeProposal, SpeculativeProposer, SpeculativeProposerContext,
    SpeculativeStats, TokenEmbedder, argmax,
};

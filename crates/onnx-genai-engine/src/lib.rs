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
#[cfg(feature = "native-backend")]
pub mod native_component;
#[cfg(feature = "native-backend")]
pub mod native_decode;
#[cfg(feature = "native-backend")]
pub(crate) mod native_speculative;
pub mod pipeline;
pub mod pipeline_cache;
pub(crate) mod processors;
#[cfg(feature = "native-backend")]
pub mod runtime_trace;
pub mod sampling;
pub(crate) mod session;
pub mod speculative;

pub use batched::{ContinuousBatchEvent, ContinuousBatchHandle, ContinuousBatchManager};
pub use connector_bridge::{ConnectorLookupOutcome, ConnectorStats};
pub use embedding::{EmbeddingOptions, EmbeddingPooling};
pub use engine::{
    DryConfig, Eagle3Config, Engine, EngineConfig, EngineConfigError, EngineDecodeBackend,
    EngineGovernorError, EngineResourceGovernor, FinishReason, GenerateConstraint, GenerateOptions,
    GeneratePrompt, GenerateRequest, GenerateResult, GenerateToken, GenerateTokenCallback,
    GenerationBudgetCap, KvConnectorBackend, KvConnectorConfig, LimitParseError, MirostatConfig,
    MirostatVersion, MtpCacheScope, MtpConfig, MtpHiddenLayout, MtpWeightSource,
    PrioritizedGenerateRequest, PrioritizedGenerateResult, RewindTokenCount, SamplingOverrides,
    ScheduledGenerateArrival, SessionCheckpoint, SessionForkCapability, SessionId, SessionPosition,
    SharedKvBinding, SharedKvProposerConfig, SpeculativeMode, TokenLogprob, XtcConfig,
    parse_resource_limit,
};
pub use fim::{FimConfig, FimFormat};
pub use logits::{
    Constraint, ConstraintProcessor, JsonConstraint, LogitProcessor, ProcessorChain,
    ProcessorChainBuilder, ProcessorContext, ProcessorSignal, StopSequence, TokenId,
};
#[cfg(feature = "native-backend")]
pub use native_component::NativeComponentSession;
#[cfg(feature = "native-backend")]
pub use native_decode::{
    CudaGraphDebugStats, CudaKvDebugStats, NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES,
    NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS, NativeDecodeCudaOptions, NativeDecodeDevice,
    NativeDecodeSession,
};
pub use onnx_genai_kv::{CachePriority, KvDType, LocalTieredConfig};
pub use onnx_genai_metadata::GenerationDefaults;
pub use onnx_genai_scheduler::{
    GovernorReconfigureOutcome, GovernorSnapshot, ResourceLimit, ResourceLimits,
};
#[cfg(feature = "native-backend")]
pub use onnx_runtime_ep_cpu::set_decode_thread_budget as set_cpu_decode_thread_budget;

/// Executor phase costs from the native runtime, as `(phase, total_ns, calls)`.
///
/// The native executor keeps its own profiler — it attributes control-flow
/// overhead (`exec_if`, `run_subgraph`, child setup) that the stage profiler
/// folds into one bucket — and cannot report through the shared registry
/// without depending on the ONNX Runtime crate. This engine depends on both, so
/// it is where the two are joined.
///
/// Empty unless the native backend is compiled in *and* `NXRT_EXEC_PHASE_PROFILE`
/// is set. It stays behind its own switch because the per-phase timing is
/// deliberately fine-grained enough to perturb what it measures.
pub fn executor_phase_stats() -> Vec<(&'static str, u128, u64)> {
    #[cfg(feature = "native-backend")]
    {
        onnx_runtime_session::exec_phase_stats()
    }
    #[cfg(not(feature = "native-backend"))]
    {
        Vec::new()
    }
}

#[cfg(feature = "native-backend")]
pub use onnx_runtime_session::DecodePrecision;
pub use pipeline::{
    ImageOutput, ImageRequest, ImageStep, ImageStepCallback, ImageStream, IterativeOverrides,
    PipelineEngine, PipelineGenerateRequest, PipelineSynthesis, PipelineTensors, Scheduler,
    SchedulerFactory, SchedulerRegistry,
};
pub use sampling::{CategoricalSampler, GreedySampler, Sampler};
pub use speculative::{
    Eagle3Proposer, LinearEmbedder, LinearLmHead, LmHead, MtpProposer, NgramProposer,
    SpeculativeAcceptContext, SpeculativeProposal, SpeculativeProposer, SpeculativeProposerContext,
    SpeculativeStats, TokenEmbedder, argmax,
};

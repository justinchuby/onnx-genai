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
pub(crate) mod kv_sizing;
pub mod logits;
mod memory_authority;
#[cfg(feature = "native-backend")]
pub mod native_decode;
pub mod native_decode_device;
#[cfg(feature = "native-backend")]
pub(crate) mod native_speculative;
pub mod pipeline;
pub mod pipeline_cache;
pub(crate) mod platform_capacity;
pub(crate) mod processors;
#[cfg(feature = "native-backend")]
pub mod runtime_trace;
pub mod sampling;
pub(crate) mod session;
pub mod speculative;

pub use onnx_genai_scheduler::SchedulerAdmissionError;

pub use batched::{
    BatchingCapability, ContinuousBatchAdmission, ContinuousBatchEvent, ContinuousBatchHandle,
    ContinuousBatchManager,
};
pub use connector_bridge::{ConnectorLookupOutcome, ConnectorStats};
pub use embedding::{EmbeddingOptions, EmbeddingPooling};
pub use engine::{
    DecisionSource, DevicePolicy, DevicePolicyParseError, DryConfig, Eagle3Config, Engine,
    EngineConfig, EngineConfigError, EngineDecodeBackend, EngineGovernorError,
    EngineResourceGovernor, FinishReason, GenerateConstraint, GenerateOptions, GeneratePrompt,
    GenerateRequest, GenerateResult, GenerateToken, GenerateTokenCallback, GenerationBudgetCap,
    KvConnectorBackend, KvConnectorConfig, LayerWeightBytes, LimitParseError,
    MemoryPolicyApplication, MemoryStrategy, MemoryStrategyDecision, MemoryStrategyPlan,
    MirostatConfig, MirostatVersion, MtpCacheScope, MtpConfig, MtpHiddenLayout, MtpWeightSource,
    PrioritizedGenerateRequest, PrioritizedGenerateResult, RewindTokenCount, SamplingOverrides,
    ScheduledGenerateArrival, SessionCheckpoint, SessionForkCapability, SessionId, SessionPosition,
    SharedKvBinding, SharedKvProposerConfig, SpeculativeMode, TokenLogprob, WeightAccessPattern,
    WeightPlacementReport, XtcConfig, parse_device_policy, parse_resource_limit,
    resolve_device_vram_limit_bytes,
};
pub use fim::{FimConfig, FimFormat};
pub use logits::{
    Constraint, ConstraintProcessor, JsonConstraint, LogitProcessor, ProcessorChain,
    ProcessorChainBuilder, ProcessorContext, ProcessorSignal, StopSequence, TokenId,
};
pub use memory_authority::{
    DeviceCompatibilityDomain, DeviceMemoryAuthority, MemoryAuthorityProvider,
};
#[cfg(feature = "native-backend")]
pub use native_decode::{
    CudaGraphDebugStats, CudaKvDebugStats, NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES,
    NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS, NativeDecodeCudaOptions, NativeDecodeDevice,
    NativeDecodeSession,
};
pub use onnx_genai_kv::{
    Applicability, CachePriority, KvDType, KvNotApplicable, KvTelemetry, KvTelemetrySnapshot,
    LocalTieredConfig,
};
pub use onnx_genai_metadata::GenerationDefaults;
pub use onnx_genai_scheduler::{
    FixedCapacity, GovernorReconfigureOutcome, GovernorSnapshot, ResourceLimit, ResourceLimits,
    resolve_limit,
};
#[cfg(feature = "native-backend")]
pub use onnx_runtime_ep_cpu::set_decode_thread_budget as set_cpu_decode_thread_budget;
pub use onnx_runtime_memory_governor::MappedGrowthMetrics;

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

/// Latest native activation-memory planner measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActivationMemoryPlanSummary {
    pub complete: bool,
    pub peak_bytes: u64,
    pub naive_bytes: u64,
    pub savings_ratio: f64,
    pub unknown_sizes: usize,
}

#[cfg(feature = "native-backend")]
impl From<onnx_runtime_session::ActivationMemoryPlanStats> for ActivationMemoryPlanSummary {
    fn from(stats: onnx_runtime_session::ActivationMemoryPlanStats) -> Self {
        Self {
            complete: stats.complete,
            peak_bytes: stats.peak_bytes as u64,
            naive_bytes: stats.naive_bytes as u64,
            savings_ratio: stats.savings_ratio,
            unknown_sizes: stats.unknown_sizes,
        }
    }
}

/// What a virtual-memory arena has done to physical memory.
///
/// Backend-agnostic on purpose. The counters currently come from the native
/// CUDA arena, but nothing in this shape is CUDA-specific: any allocator that
/// reserves address space and commits it on demand answers the same six
/// questions, so a second backend can report here without the CLI changing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VmmArenaStats {
    /// Granules mapped since the process started.
    pub commits: u64,
    /// Granules unmapped since the process started.
    pub releases: u64,
    /// Physical bytes mapped right now.
    pub committed_bytes: u64,
    /// Address space reserved right now. The gap between this and
    /// `committed_bytes` is what the approach buys.
    pub reserved_bytes: u64,
    /// High-water mark of `committed_bytes`.
    pub peak_committed_bytes: u64,
    /// Spans handed out. Many allocations per commit is granule sharing
    /// working; one commit per allocation means every small tensor costs a
    /// whole granule.
    pub allocations: u64,
    /// Times a granule was released whose reference count was already zero.
    /// **Anything but zero is a bug** in the allocator's accounting.
    pub ref_underflows: u64,
    /// Times a byte counter would have gone negative and was clamped.
    /// **Anything but zero is a bug** in the allocator's accounting.
    pub byte_underflows: u64,
    /// Committed bytes the adopted memory governor did not record.
    /// **Anything but zero is a fault** because admission sees understated use.
    pub unaccounted_committed_bytes: u64,
}

#[cfg(feature = "native-backend")]
pub use onnx_runtime_session::DecodePrecision;
pub use pipeline::{
    PipelineEngine, PipelineGenerateRequest, PipelineTensors, WorkflowSessionCheckpoint,
    validate_pipeline_backend_request,
};
pub use sampling::{CategoricalSampler, GreedySampler, Sampler};
pub use speculative::{
    Eagle3Proposer, LinearEmbedder, LinearLmHead, LmHead, MtpProposer, NgramProposer,
    SpeculativeAcceptContext, SpeculativeProposal, SpeculativeProposer, SpeculativeProposerContext,
    SpeculativeStats, TokenEmbedder, argmax,
};

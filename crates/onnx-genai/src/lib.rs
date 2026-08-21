//! # onnx-genai
//!
//! A Rust inference runtime for generative AI models built on ONNX Runtime.
//!
//! Reference implementation of the ONNX inference metadata standard
//! ([onnx/onnx#8184](https://github.com/onnx/onnx/issues/8184)).

pub mod reasoning;

pub use onnx_genai_engine as engine;
pub use onnx_genai_kv as kv;
pub use onnx_genai_metadata as metadata;
pub use onnx_genai_ort as ort;
pub use onnx_genai_preprocess as preprocess;
pub use onnx_genai_scheduler as scheduler;

pub use onnx_genai_engine::{
    CategoricalSampler, Constraint, ConstraintProcessor, DryConfig, Engine, EngineConfig,
    FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult, GenerateToken,
    GenerateTokenCallback, GenerationBudgetCap, GreedySampler, JsonConstraint, LogitProcessor,
    MirostatConfig, MirostatVersion, ProcessorChain, ProcessorChainBuilder, ProcessorContext,
    ProcessorSignal, Sampler, SamplingOverrides, SessionId, SpeculativeAcceptContext,
    SpeculativeProposal, SpeculativeProposer, SpeculativeProposerContext, StopSequence, TokenId,
    XtcConfig,
};

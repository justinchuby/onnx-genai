//! Generation engine.
//!
//! The core orchestrator that ties together:
//! - ORT sessions (model execution)
//! - KV cache manager (memory)
//! - Scheduler (batching)
//! - Logit processors (sampling)
//! - Speculative decoding (acceleration)

pub mod engine;
pub mod logits;
pub mod pipeline;
pub mod sampling;
pub mod speculative;

pub use engine::{
    Engine, EngineConfig, FinishReason, GenerateConstraint, GenerateOptions, GeneratePrompt,
    GenerateRequest, GenerateResult, GenerateToken, GenerateTokenCallback,
    PrioritizedGenerateRequest, PrioritizedGenerateResult, ScheduledGenerateArrival, SessionId,
};
pub use logits::{Constraint, JsonConstraint, StopSequence, TokenId};

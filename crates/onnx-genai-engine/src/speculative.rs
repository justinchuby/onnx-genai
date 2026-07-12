//! Speculative decoding engine.

//! Phase 3 currently wires the draft-model/greedy acceptance path directly into
//! [`crate::engine::Engine`] so the public `generate` API remains stable. When
//! `EngineConfig::draft_model` is set, greedy requests propose `K`
//! autoregressive draft tokens, verify them with one target pass, accept the
//! longest target-greedy prefix, and take the target token at the first
//! mismatch. Target paged KV is rewound before committing the target token, and
//! draft KV is rewound to the shared prefix so the correction/bonus token seeds
//! the next draft round. Rejected draft tokens never leak into subsequent
//! decoding state.

/// Speculative acceptance rule implemented by the Phase 3 engine path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptanceRule {
    /// Accept a draft token iff it matches the target model's greedy argmax.
    Greedy,
}

/// Result of a single greedy speculative verification step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreedyStep {
    /// Number of proposed draft tokens accepted before the first mismatch.
    pub accepted_prefix_len: usize,
    /// Whether every proposed draft token was accepted.
    pub fully_accepted: bool,
}

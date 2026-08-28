//! Persistent engine and draft generation session state.

use crate::config::{GenerateOptions, GenerationBudgetCap, SessionId, TokenLogprob};
use crate::decode::{DecodeState, ModelDecodePath};
use crate::kv_bridge::KvModelInfo;
use crate::logits::{ProcessorChain, TokenId};
use crate::sampling::SamplingRng;
use onnx_genai_kv::PagedKvCache;
use onnx_genai_ort::Session;
use std::sync::Arc;

pub(crate) struct EngineSession {
    /// Logical token context retained across turns.
    pub(crate) tokens: Vec<TokenId>,
    /// Prefix length currently materialized in `decode_state.past`.
    pub(crate) kv_token_count: usize,
    /// ORT-managed past tensors retained between calls.
    pub(crate) decode_state: DecodeState,
    /// Optional draft-model state aligned to this target sequence.
    pub(crate) draft: Option<DraftSession>,
    /// Latched once the device sampled fast path reports it is unavailable, so
    /// the incremental generate path (which rebuilds its transient decode-loop
    /// backend every step) stops retrying it and stays on the host sampler.
    pub(crate) sampled_fastpath_failed: bool,
}

pub(crate) struct ActiveGenerate {
    /// The in-flight interpretation of this request's declared generation loop.
    ///
    /// Held across scheduler steps because the loop is the workflow's: the
    /// prioritized drive advances it one iteration at a time through the same
    /// interpreter method the run-to-completion path drives in a `for`, rather
    /// than running a second loop that would have to restate the semantics.
    pub(crate) cursor: crate::pipeline::WorkflowGenerationCursor,
    pub(crate) session_id: SessionId,
    pub(crate) state: EngineSession,
    pub(crate) options: GenerateOptions,
    pub(crate) chain: ProcessorChain,
    pub(crate) max_context: Option<usize>,
    pub(crate) prompt_len: usize,
    pub(crate) prefix_cache_hit_len: usize,
    pub(crate) generated_tokens: Vec<TokenId>,
    pub(crate) generated_text: String,
    pub(crate) logprobs: Option<Vec<TokenLogprob>>,
    pub(crate) budget_cap: Option<GenerationBudgetCap>,
    pub(crate) step: usize,
    pub(crate) rng: SamplingRng,
    /// A stop reached by authored loop setup before the first scheduled body
    /// iteration.
    pub(crate) setup_finish: Option<crate::FinishReason>,
}

pub(crate) struct DraftModel {
    pub(crate) session: Arc<Session>,
    pub(crate) decode_path: ModelDecodePath,
    pub(crate) io: Option<onnx_genai_metadata::DecoderAbi>,
    pub(crate) kv_model: Option<KvModelInfo>,
    pub(crate) kv_cache: PagedKvCache,
}

pub(crate) struct DraftSession {
    pub(crate) seq: SessionId,
    pub(crate) tokens: Vec<TokenId>,
    pub(crate) kv_token_count: usize,
    pub(crate) decode_state: DecodeState,
}

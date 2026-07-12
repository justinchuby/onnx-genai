//! Persistent engine and draft generation session state.

use crate::config::{GenerateOptions, SessionId, TokenLogprob};
use crate::decode::{DecodeState, ModelDecodePath};
use crate::kv_bridge::KvModelInfo;
use crate::logits::{ProcessorChain, TokenId};
use crate::sampling::SamplingRng;
use onnx_genai_kv::PagedKvCache;
use onnx_genai_ort::Session;

pub(crate) struct EngineSession {
    /// Logical token context retained across turns.
    pub(crate) tokens: Vec<TokenId>,
    /// Prefix length currently materialized in `decode_state.past`.
    pub(crate) kv_token_count: usize,
    /// ORT-managed past tensors retained between calls.
    pub(crate) decode_state: DecodeState,
    /// Optional draft-model state aligned to this target sequence.
    pub(crate) draft: Option<DraftSession>,
}

pub(crate) struct ActiveGenerate {
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
    pub(crate) step: usize,
    pub(crate) rng: SamplingRng,
}

pub(crate) struct DraftModel {
    pub(crate) session: Box<Session>,
    pub(crate) decode_path: ModelDecodePath,
    pub(crate) kv_model: Option<KvModelInfo>,
    pub(crate) kv_cache: PagedKvCache,
}

pub(crate) struct DraftSession {
    pub(crate) seq: SessionId,
    pub(crate) tokens: Vec<TokenId>,
    pub(crate) kv_token_count: usize,
    pub(crate) decode_state: DecodeState,
}

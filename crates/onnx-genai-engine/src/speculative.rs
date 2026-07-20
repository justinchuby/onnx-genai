//! Speculative decoding engine.

//! Greedy requests can propose candidates with a draft model, an MTP head, or
//! model-free prompt lookup. All sources feed the same target verification,
//! longest-prefix acceptance, correction-token, and KV rewind path.

use crate::TokenId;
use crate::config::{MtpCacheScope, MtpHiddenLayout};
use crate::decode::{
    apply_paged_sliding_window, extract_logits_sequence, next_session_token_logits,
    next_session_token_logits_and_hidden, next_session_token_logits_and_hiddens,
    propose_draft_tokens, run_decode_session_logits, run_decode_step,
};
use crate::decode_loop::{
    DecodeLoopState, commit_selected_token, logprob_for_token, reached_context_limit,
};
use crate::engine::Engine;
use crate::kv_bridge::{
    common_prefix_len, mirror_present_kv_to_pages, rewind_draft_state_to_len,
    rewind_target_state_to_len, trim_overmaterialized_target_kv,
};
use crate::logits::{ProcessorChain, ProcessorContext};
use crate::processors::{ensure_constrained_finish, select_next_token_with_rng};
use crate::sampling::SamplingRng;
use crate::session::{DraftModel, DraftSession, EngineSession};
use crate::{
    FinishReason, GenerateOptions, GenerateResult, GenerateTokenCallback, SessionId,
    SpeculativeMode,
};
use anyhow::Context;
use onnx_genai_kv::KvCacheOps;
use onnx_genai_ort::{
    Eagle3DecodeOptions, Eagle3DecodeSession, MtpDecodeOptions, MtpDecodeSession, Session,
    SharedKvInput, SharedKvProposerSession,
};
use onnx_runtime_ir::{DataType as IrDataType, WeightRef};
use onnx_runtime_loader::WeightStore;
use std::path::Path;
use std::sync::Arc;

/// Produces a target-model token embedding for an MTP proposal step.
pub trait TokenEmbedder {
    fn hidden_size(&self) -> usize;
    fn embed(&self, token: TokenId, out: &mut [f32]) -> anyhow::Result<()>;
}

/// Projects a target-model hidden state to vocabulary logits.
pub trait LmHead {
    fn vocab_size(&self) -> usize;
    fn logits(&self, hidden: &[f32], out: &mut [f32]) -> anyhow::Result<()>;
}

/// Dense target embedding table in row-major `[vocab, hidden]` order.
#[derive(Debug, Clone)]
pub struct LinearEmbedder {
    weight: Vec<f32>,
    vocab: usize,
    hidden: usize,
}

impl LinearEmbedder {
    pub fn new(weight: Vec<f32>, vocab: usize, hidden: usize) -> anyhow::Result<Self> {
        if weight.len() != vocab * hidden {
            anyhow::bail!(
                "embedder weight length {} != vocab {vocab} * hidden {hidden}",
                weight.len()
            );
        }
        Ok(Self {
            weight,
            vocab,
            hidden,
        })
    }
}

impl TokenEmbedder for LinearEmbedder {
    fn hidden_size(&self) -> usize {
        self.hidden
    }

    fn embed(&self, token: TokenId, out: &mut [f32]) -> anyhow::Result<()> {
        let token = token as usize;
        if token >= self.vocab {
            anyhow::bail!("token {token} out of range for vocab {}", self.vocab);
        }
        if out.len() != self.hidden {
            anyhow::bail!(
                "embed output length {} != hidden {}",
                out.len(),
                self.hidden
            );
        }
        let start = token * self.hidden;
        out.copy_from_slice(&self.weight[start..start + self.hidden]);
        Ok(())
    }
}

/// Dense target LM-head projection in row-major `[hidden, vocab]` order.
#[derive(Debug, Clone)]
pub struct LinearLmHead {
    weight: Vec<f32>,
    hidden: usize,
    vocab: usize,
}

#[derive(Debug, Clone)]
struct TargetInitializerMatrix {
    store: Arc<WeightStore>,
    weight: WeightRef,
    rows: usize,
    cols: usize,
}

impl TargetInitializerMatrix {
    fn new(
        store: Arc<WeightStore>,
        weight: WeightRef,
        rows: usize,
        cols: usize,
    ) -> anyhow::Result<Self> {
        let dtype = weight.dtype();
        if !matches!(
            dtype,
            IrDataType::Float32 | IrDataType::Float16 | IrDataType::BFloat16
        ) {
            anyhow::bail!(
                "MTP target initializer dtype {dtype:?} is not supported; Phase 1 supports Float32, Float16, and BFloat16 shared weights"
            );
        }
        let bytes = store
            .bytes(&weight)
            .context("target initializer bytes are not available")?;
        let expected = dtype.storage_bytes(
            rows.checked_mul(cols)
                .context("target initializer element count overflow")?,
        );
        if bytes.len() != expected {
            anyhow::bail!(
                "target initializer byte length {} != expected {expected} for [{rows}, {cols}] {dtype:?}",
                bytes.len()
            );
        }
        Ok(Self {
            store,
            weight,
            rows,
            cols,
        })
    }

    fn value(&self, row: usize, col: usize) -> anyhow::Result<f32> {
        let index = row
            .checked_mul(self.cols)
            .and_then(|start| start.checked_add(col))
            .context("target initializer index overflow")?;
        let bytes = self
            .store
            .bytes(&self.weight)
            .context("target initializer bytes are not available")?;
        Ok(match self.weight.dtype() {
            IrDataType::Float32 => {
                let start = index * 4;
                f32::from_le_bytes(bytes[start..start + 4].try_into().expect("four bytes"))
            }
            IrDataType::Float16 => {
                let start = index * 2;
                half::f16::from_bits(u16::from_le_bytes(
                    bytes[start..start + 2].try_into().expect("two bytes"),
                ))
                .to_f32()
            }
            IrDataType::BFloat16 => {
                let start = index * 2;
                half::bf16::from_bits(u16::from_le_bytes(
                    bytes[start..start + 2].try_into().expect("two bytes"),
                ))
                .to_f32()
            }
            dtype => anyhow::bail!("unsupported target initializer dtype {dtype:?}"),
        })
    }
}

/// Target embedding adapter backed directly by an ONNX initializer.
#[derive(Debug, Clone)]
pub(crate) struct TargetInitializerEmbedder {
    matrix: TargetInitializerMatrix,
}

impl TokenEmbedder for TargetInitializerEmbedder {
    fn hidden_size(&self) -> usize {
        self.matrix.cols
    }

    fn embed(&self, token: TokenId, out: &mut [f32]) -> anyhow::Result<()> {
        let token = token as usize;
        if token >= self.matrix.rows {
            anyhow::bail!(
                "token {token} out of range for target initializer vocabulary {}",
                self.matrix.rows
            );
        }
        if out.len() != self.matrix.cols {
            anyhow::bail!(
                "embed output length {} != hidden {}",
                out.len(),
                self.matrix.cols
            );
        }
        for (column, value) in out.iter_mut().enumerate() {
            *value = self.matrix.value(token, column)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum LmHeadInitializerLayout {
    HiddenVocab,
    VocabHidden,
}

/// Target LM-head adapter backed directly by an ONNX initializer.
#[derive(Debug, Clone)]
pub(crate) struct TargetInitializerLmHead {
    matrix: TargetInitializerMatrix,
    layout: LmHeadInitializerLayout,
    hidden: usize,
    vocab: usize,
}

impl LmHead for TargetInitializerLmHead {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn logits(&self, hidden: &[f32], out: &mut [f32]) -> anyhow::Result<()> {
        if hidden.len() != self.hidden {
            anyhow::bail!(
                "lm-head input length {} != hidden {}",
                hidden.len(),
                self.hidden
            );
        }
        if out.len() != self.vocab {
            anyhow::bail!(
                "lm-head output length {} != vocab {}",
                out.len(),
                self.vocab
            );
        }
        for (vocab_index, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (hidden_index, &value) in hidden.iter().enumerate() {
                let weight = match self.layout {
                    LmHeadInitializerLayout::HiddenVocab => {
                        self.matrix.value(hidden_index, vocab_index)?
                    }
                    LmHeadInitializerLayout::VocabHidden => {
                        self.matrix.value(vocab_index, hidden_index)?
                    }
                };
                acc += value * weight;
            }
            *slot = acc;
        }
        Ok(())
    }
}

/// Embedding implementation selected by legacy file or package metadata.
#[derive(Debug, Clone)]
pub(crate) enum MtpEmbedder {
    Linear(LinearEmbedder),
    TargetInitializer(TargetInitializerEmbedder),
}

impl TokenEmbedder for MtpEmbedder {
    fn hidden_size(&self) -> usize {
        match self {
            Self::Linear(embedder) => embedder.hidden_size(),
            Self::TargetInitializer(embedder) => embedder.hidden_size(),
        }
    }

    fn embed(&self, token: TokenId, out: &mut [f32]) -> anyhow::Result<()> {
        match self {
            Self::Linear(embedder) => embedder.embed(token, out),
            Self::TargetInitializer(embedder) => embedder.embed(token, out),
        }
    }
}

/// LM-head implementation selected by legacy file or package metadata.
#[derive(Debug, Clone)]
pub(crate) enum MtpLmHead {
    Linear(LinearLmHead),
    TargetInitializer(TargetInitializerLmHead),
}

impl LmHead for MtpLmHead {
    fn vocab_size(&self) -> usize {
        match self {
            Self::Linear(lm_head) => lm_head.vocab_size(),
            Self::TargetInitializer(lm_head) => lm_head.vocab_size(),
        }
    }

    fn logits(&self, hidden: &[f32], out: &mut [f32]) -> anyhow::Result<()> {
        match self {
            Self::Linear(lm_head) => lm_head.logits(hidden, out),
            Self::TargetInitializer(lm_head) => lm_head.logits(hidden, out),
        }
    }
}

pub(crate) fn load_target_initializer_adapters(
    model_path: &Path,
    embedding_name: &str,
    lm_head_name: &str,
    hidden_size: usize,
) -> anyhow::Result<(MtpEmbedder, MtpLmHead, usize)> {
    let (graph, store) =
        onnx_runtime_loader::load_model_with_weights(model_path).with_context(|| {
            format!(
                "Failed to load target initializers from '{}'",
                model_path.display()
            )
        })?;
    let find_weight = |name: &str| -> anyhow::Result<WeightRef> {
        graph
            .initializers
            .iter()
            .find_map(|(&value_id, weight)| {
                (graph.value(value_id).name.as_deref() == Some(name)).then(|| weight.clone())
            })
            .with_context(|| format!("target model initializer '{name}' was not found"))
    };
    let embedding_weight = find_weight(embedding_name)?;
    let lm_head_weight = find_weight(lm_head_name)?;
    let embedding_dims = embedding_weight.dims();
    if embedding_dims.len() != 2 || embedding_dims[1] != hidden_size {
        anyhow::bail!(
            "target embedding initializer '{embedding_name}' shape {:?} must be [vocab, {hidden_size}]",
            embedding_dims
        );
    }
    let vocab_size = embedding_dims[0];
    let lm_head_dims = lm_head_weight.dims();
    let (layout, rows, cols) = if lm_head_dims == [hidden_size, vocab_size] {
        (
            LmHeadInitializerLayout::HiddenVocab,
            hidden_size,
            vocab_size,
        )
    } else if lm_head_dims == [vocab_size, hidden_size] {
        (
            LmHeadInitializerLayout::VocabHidden,
            vocab_size,
            hidden_size,
        )
    } else {
        anyhow::bail!(
            "target LM-head initializer '{lm_head_name}' shape {:?} must be [{hidden_size}, {vocab_size}] or [{vocab_size}, {hidden_size}]",
            lm_head_dims
        );
    };
    let embedder = TargetInitializerEmbedder {
        matrix: TargetInitializerMatrix::new(
            Arc::clone(&store),
            embedding_weight,
            vocab_size,
            hidden_size,
        )?,
    };
    let lm_head = TargetInitializerLmHead {
        matrix: TargetInitializerMatrix::new(store, lm_head_weight, rows, cols)?,
        layout,
        hidden: hidden_size,
        vocab: vocab_size,
    };
    Ok((
        MtpEmbedder::TargetInitializer(embedder),
        MtpLmHead::TargetInitializer(lm_head),
        vocab_size,
    ))
}

impl LinearLmHead {
    pub fn new(weight: Vec<f32>, hidden: usize, vocab: usize) -> anyhow::Result<Self> {
        if weight.len() != hidden * vocab {
            anyhow::bail!(
                "lm-head weight length {} != hidden {hidden} * vocab {vocab}",
                weight.len()
            );
        }
        Ok(Self {
            weight,
            hidden,
            vocab,
        })
    }
}

impl LmHead for LinearLmHead {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn logits(&self, hidden: &[f32], out: &mut [f32]) -> anyhow::Result<()> {
        if hidden.len() != self.hidden {
            anyhow::bail!(
                "lm-head input length {} != hidden {}",
                hidden.len(),
                self.hidden
            );
        }
        if out.len() != self.vocab {
            anyhow::bail!(
                "lm-head output length {} != vocab {}",
                out.len(),
                self.vocab
            );
        }
        for (col, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (row, &value) in hidden.iter().enumerate() {
                acc += value * self.weight[row * self.vocab + col];
            }
            *slot = acc;
        }
        Ok(())
    }
}

/// Index of the maximum logit, resolving ties to the lowest index.
pub fn argmax(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .fold(None, |best, (index, &value)| match best {
            Some((_, best_value)) if value <= best_value => best,
            _ => Some((index, value)),
        })
        .map(|(index, _)| index)
}

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

/// Inputs a proposer needs to draft speculative candidates for one verify pass.
pub struct SpeculativeProposerContext<'a> {
    pub width: usize,
    pub context_tokens: &'a [TokenId],
    pub generated_tokens: &'a [TokenId],
    pub generated_text: &'a str,
    pub first_step: usize,
    pub options: &'a GenerateOptions,
    pub chain: &'a ProcessorChain,
    /// Target decoder's last hidden state, when required by the proposer.
    pub target_hidden: Option<&'a [f32]>,
    /// Target decoder's selected low/middle/high last-token hidden states.
    ///
    /// EAGLE-3 concatenates these in order to form its `fused_hidden` input.
    pub target_hidden_layers: Option<&'a [Vec<f32>]>,
    /// Target model's unprocessed greedy next token.
    pub guaranteed_token: Option<TokenId>,
    /// Target KV slices bound to a shared-KV proposer's `shared_kv.*` inputs.
    pub shared_kv_slices: Option<&'a [SharedKvInput]>,
}

/// Aggregate diagnostics for one speculative generation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeStats {
    pub verification_steps: usize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub multi_token_accepts: usize,
}

/// Observable token decisions from one speculative verification iteration.
///
/// Target tokens stop at the first mismatch. A fully accepted proposal has one
/// additional target token only when the engine actually selected a bonus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeIterationTrace {
    /// Generated-output offset before this iteration committed any tokens.
    pub output_offset: usize,
    /// Complete family-specific proposal sent to target verification.
    pub proposal_token_ids: Vec<TokenId>,
    /// Tokens actually selected from target logits along the executed path.
    pub target_token_ids: Vec<TokenId>,
    /// Tokens actually emitted before this iteration completed or terminated.
    pub committed_token_ids: Vec<TokenId>,
}

/// Proposer family that actually entered the speculative verification loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeTraceFamily {
    DraftModel,
    PromptLookup,
    Mtp,
    Eagle3,
    SharedKv,
}

/// Opt-in token-level evidence from one generation call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeGenerationTrace {
    /// Tokenized prompt used by the generation call.
    pub prompt_token_ids: Vec<TokenId>,
    /// Final generated output, excluding prompt tokens.
    pub output_token_ids: Vec<TokenId>,
    /// Generation terminal condition.
    pub finish_reason: FinishReason,
    /// Proposer family used by the request, or `None` for target-only decode.
    pub family: Option<SpeculativeTraceFamily>,
    /// Configured proposal width excluding any guaranteed target prefix.
    pub max_additional_tokens: Option<usize>,
    /// Speculative iterations. Empty when the request used target-only decode.
    pub iterations: Vec<SpeculativeIterationTrace>,
}

#[derive(Debug, Default)]
pub(crate) struct SpeculativeTraceCaptureState {
    pub family: Option<SpeculativeTraceFamily>,
    pub max_additional_tokens: Option<usize>,
    pub iterations: Vec<SpeculativeIterationTrace>,
}

/// Candidate tokens proposed for a target-model verification pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeProposal {
    pub tokens: Vec<TokenId>,
    pub positions: Option<Vec<usize>>,
    pub tree: Option<Vec<Vec<usize>>>,
}

impl SpeculativeProposal {
    pub fn linear(tokens: Vec<TokenId>) -> Self {
        Self {
            tokens,
            positions: None,
            tree: None,
        }
    }
}

/// Outcome reported back to the proposer after verification and commit.
pub struct SpeculativeAcceptContext<'a> {
    pub accepted_prefix_len: usize,
    pub committed_tokens: &'a [TokenId],
    pub target_tokens: &'a [TokenId],
}

/// Source of speculative draft tokens.
pub trait SpeculativeProposer {
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal>;

    fn accept(&mut self, _context: &SpeculativeAcceptContext<'_>) -> anyhow::Result<()> {
        Ok(())
    }

    fn rewind(&mut self, _target_tokens: &[TokenId]) -> anyhow::Result<()> {
        Ok(())
    }

    fn name(&self) -> &str;
}

/// Model-free proposer that copies the continuation after the most recent
/// earlier occurrence of the current context suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramProposer {
    ngram: usize,
    max_tokens: usize,
}

impl NgramProposer {
    pub fn new(ngram: usize, max_tokens: usize) -> anyhow::Result<Self> {
        if ngram == 0 {
            anyhow::bail!("ngram must be greater than zero");
        }
        if max_tokens == 0 {
            anyhow::bail!("max_tokens must be greater than zero");
        }
        Ok(Self { ngram, max_tokens })
    }
}

impl SpeculativeProposer for NgramProposer {
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal> {
        let tokens = context.context_tokens;
        if tokens.len() <= self.ngram {
            return Ok(SpeculativeProposal::linear(Vec::new()));
        }

        let suffix_start = tokens.len() - self.ngram;
        let suffix = &tokens[suffix_start..];
        let Some(match_start) = (0..suffix_start).rev().find(|&start| {
            start + self.ngram < tokens.len() && &tokens[start..start + self.ngram] == suffix
        }) else {
            return Ok(SpeculativeProposal::linear(Vec::new()));
        };

        let continuation_start = match_start + self.ngram;
        let continuation_len = context
            .width
            .min(self.max_tokens)
            .min(tokens.len() - continuation_start);
        Ok(SpeculativeProposal::linear(
            tokens[continuation_start..continuation_start + continuation_len].to_vec(),
        ))
    }

    fn name(&self) -> &str {
        "prompt_lookup"
    }
}

/// Multi-token-prediction proposer backed by an ORT MTP-head session.
pub struct MtpProposer<'a, E = LinearEmbedder, L = LinearLmHead> {
    session: MtpDecodeSession<'a>,
    embedder: E,
    lm_head: L,
    cache_scope: MtpCacheScope,
}

impl<'a, E, L> MtpProposer<'a, E, L>
where
    E: TokenEmbedder,
    L: LmHead,
{
    pub fn new(
        head: &'a Session,
        options: MtpDecodeOptions,
        embedder: E,
        lm_head: L,
    ) -> anyhow::Result<Self> {
        Self::new_with_cache_scope(
            head,
            options,
            embedder,
            lm_head,
            MtpCacheScope::ProposalLocal,
        )
    }

    pub fn new_with_cache_scope(
        head: &'a Session,
        options: MtpDecodeOptions,
        embedder: E,
        lm_head: L,
        cache_scope: MtpCacheScope,
    ) -> anyhow::Result<Self> {
        let session = MtpDecodeSession::new(head, options)
            .map_err(|error| anyhow::anyhow!("Failed to create MTP decode session: {error}"))?;
        if session.signature().hidden_size != embedder.hidden_size() {
            anyhow::bail!(
                "MTP head hidden size {} does not match target embedding hidden size {}",
                session.signature().hidden_size,
                embedder.hidden_size()
            );
        }
        Ok(Self {
            session,
            embedder,
            lm_head,
            cache_scope,
        })
    }
}

impl<E, L> MtpProposer<'static, E, L>
where
    E: TokenEmbedder,
    L: LmHead,
{
    pub fn new_owned(
        head: Arc<Session>,
        options: MtpDecodeOptions,
        embedder: E,
        lm_head: L,
        cache_scope: MtpCacheScope,
    ) -> anyhow::Result<Self> {
        let session = MtpDecodeSession::new_owned(head, options)
            .map_err(|error| anyhow::anyhow!("Failed to create MTP decode session: {error}"))?;
        if session.signature().hidden_size != embedder.hidden_size() {
            anyhow::bail!(
                "MTP head hidden size {} does not match target embedding hidden size {}",
                session.signature().hidden_size,
                embedder.hidden_size()
            );
        }
        Ok(Self {
            session,
            embedder,
            lm_head,
            cache_scope,
        })
    }
}

impl<E, L> SpeculativeProposer for MtpProposer<'_, E, L>
where
    E: TokenEmbedder,
    L: LmHead,
{
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal> {
        let hidden = context
            .target_hidden
            .context("MTP proposer requires the target model's last hidden state")?;
        let guaranteed_token = context
            .guaranteed_token
            .context("MTP proposer requires the target model's greedy next token")?;
        let draft_count = context.width.saturating_sub(1);
        let expected_state_len = self
            .session
            .signature()
            .hidden_size
            .checked_mul(self.session.hc_mult())
            .context("MTP HC state width overflow")?;
        if hidden.len() != expected_state_len {
            anyhow::bail!(
                "target_hidden length {} != hc_mult {} * hidden {}",
                hidden.len(),
                self.session.hc_mult(),
                self.session.signature().hidden_size
            );
        }
        if self.cache_scope == MtpCacheScope::ProposalLocal {
            self.session.reset();
        } else if context.first_step > 0 {
            anyhow::bail!(
                "MTP accepted_prefix KV reuse is not enabled: the frozen Mobius contract does not define correction-token/cache alignment"
            );
        }
        let mut tokens = Vec::with_capacity(draft_count + 1);
        tokens.push(guaranteed_token);
        let mut running_state = hidden.to_vec();
        let mut previous_token = guaranteed_token;
        let mut embedding = vec![0.0f32; self.session.signature().hidden_size];
        let mut logits = vec![0.0f32; self.lm_head.vocab_size()];
        for draft_index in 0..draft_count {
            self.embedder.embed(previous_token, &mut embedding)?;
            let position = i64::try_from(
                context
                    .context_tokens
                    .len()
                    .checked_add(draft_index)
                    .context("MTP absolute position overflow")?,
            )
            .context("MTP position exceeds i64")?;
            let output = self
                .session
                .step_with_state(&embedding, &running_state, position)
                .map_err(|error| anyhow::anyhow!("MTP proposal step failed: {error}"))?;
            self.lm_head.logits(&output.hidden, &mut logits)?;
            let token = argmax(&logits).context("lm-head produced empty logits")? as TokenId;
            tokens.push(token);
            running_state = output.state;
            previous_token = token;
        }
        Ok(SpeculativeProposal {
            tokens,
            positions: None,
            tree: None,
        })
    }

    fn accept(&mut self, context: &SpeculativeAcceptContext<'_>) -> anyhow::Result<()> {
        if self.cache_scope == MtpCacheScope::ProposalLocal
            || self.session.mode() == onnx_genai_ort::MtpDraftKvMode::HiddenThreaded
        {
            self.session.reset();
            return Ok(());
        }
        self.session
            .rewind(context.accepted_prefix_len.saturating_sub(1))
            .map_err(|error| anyhow::anyhow!("Failed to rewind MTP proposal: {error}"))
    }

    fn rewind(&mut self, _target_tokens: &[TokenId]) -> anyhow::Result<()> {
        self.session.reset();
        Ok(())
    }

    fn name(&self) -> &str {
        "mtp"
    }
}

/// EAGLE-3 proposer backed by an autoregressive ORT draft-head session.
pub struct Eagle3Proposer<'a, E = LinearEmbedder> {
    session: Eagle3DecodeSession<'a>,
    embedder: E,
}

impl<'a, E> Eagle3Proposer<'a, E>
where
    E: TokenEmbedder,
{
    pub fn new(
        head: &'a Session,
        options: Eagle3DecodeOptions,
        embedder: E,
    ) -> anyhow::Result<Self> {
        let session = Eagle3DecodeSession::new(head, options)
            .map_err(|error| anyhow::anyhow!("Failed to create EAGLE-3 decode session: {error}"))?;
        if session.signature().hidden_size != embedder.hidden_size() {
            anyhow::bail!(
                "EAGLE-3 head hidden size {} does not match target embedding hidden size {}",
                session.signature().hidden_size,
                embedder.hidden_size()
            );
        }
        Ok(Self { session, embedder })
    }
}

impl<E> SpeculativeProposer for Eagle3Proposer<'_, E>
where
    E: TokenEmbedder,
{
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal> {
        let layers = context
            .target_hidden_layers
            .context("EAGLE-3 proposer requires low/middle/high target hidden states")?;
        let guaranteed_token = context
            .guaranteed_token
            .context("EAGLE-3 proposer requires the target model's greedy next token")?;
        let hidden_size = self.session.signature().hidden_size;
        if layers.len() != 3 || layers.iter().any(|layer| layer.len() != hidden_size) {
            anyhow::bail!(
                "EAGLE-3 requires exactly three target hidden states of width {hidden_size}"
            );
        }
        let mut fused_hidden = Vec::with_capacity(self.session.signature().fused_hidden_size);
        for layer in layers {
            fused_hidden.extend_from_slice(layer);
        }
        if fused_hidden.len() != self.session.signature().fused_hidden_size {
            anyhow::bail!(
                "fused target hidden length {} != EAGLE-3 head fused hidden {}",
                fused_hidden.len(),
                self.session.signature().fused_hidden_size
            );
        }
        let mut running_hidden = context
            .target_hidden
            .map(<[f32]>::to_vec)
            .unwrap_or_else(|| layers[2].clone());
        if running_hidden.len() != hidden_size {
            anyhow::bail!(
                "EAGLE-3 recycled target hidden length {} != hidden {hidden_size}",
                running_hidden.len()
            );
        }

        self.session.reset();
        let draft_count = context.width.saturating_sub(1);
        let mut tokens = Vec::with_capacity(draft_count + 1);
        tokens.push(guaranteed_token);
        let mut previous_token = guaranteed_token;
        let mut embedding = vec![0.0f32; hidden_size];
        for step in 0..draft_count {
            self.embedder.embed(previous_token, &mut embedding)?;
            let position = i64::try_from(context.context_tokens.len() + step)
                .context("EAGLE-3 position exceeds i64")?;
            let output = self
                .session
                .step(&embedding, &fused_hidden, &running_hidden, position)
                .map_err(|error| anyhow::anyhow!("EAGLE-3 proposal step failed: {error}"))?;
            let token = TokenId::try_from(
                argmax(&output.logits).context("EAGLE-3 head produced empty draft logits")?,
            )
            .context("EAGLE-3 token id exceeds u32 range")?;
            tokens.push(token);
            previous_token = token;
            running_hidden = output.hidden;
        }
        Ok(SpeculativeProposal::linear(tokens))
    }

    fn accept(&mut self, _context: &SpeculativeAcceptContext<'_>) -> anyhow::Result<()> {
        // Every verification pass receives a fresh target low/mid/high anchor.
        // Keeping draft-only KV across passes would make rejected features stale.
        self.session.reset();
        Ok(())
    }

    fn rewind(&mut self, _target_tokens: &[TokenId]) -> anyhow::Result<()> {
        self.session.reset();
        Ok(())
    }

    fn name(&self) -> &str {
        "eagle3"
    }
}

/// Shared-KV draft proposer (originally introduced for Gemma4 `*-assistant`).
///
/// The assistant owns no KV cache: it reads slices of the target model's paged
/// KV cache through `shared_kv.*` inputs (provided via the proposer context) and
/// carries its own internal `lm_head`. Each step consumes
/// `inputs_embeds = concat(target_input_embedding(last_token), hidden)` (each `H`
/// wide), emits full draft `logits`, and threads its `projected_state` output
/// forward as the next step's `hidden`. The first step seeds `hidden` from the
/// target's last hidden state and `last_token` from the last context token; the
/// guaranteed target token is emitted for free and its assistant step only
/// advances the threaded hidden state before real drafts begin.
pub struct SharedKvProposer<'a> {
    session: SharedKvProposerSession<'a>,
    embedder: &'a dyn TokenEmbedder,
}

impl<'a> SharedKvProposer<'a> {
    pub fn new(model: &'a Session, embedder: &'a dyn TokenEmbedder) -> anyhow::Result<Self> {
        let session = SharedKvProposerSession::new(model).map_err(|error| {
            anyhow::anyhow!("Failed to create shared-KV proposer decode session: {error}")
        })?;
        let hidden_size = session.signature().backbone_hidden_size;
        if embedder.hidden_size() != hidden_size {
            anyhow::bail!(
                "shared-KV proposer embedding hidden size {} != backbone hidden size {hidden_size}",
                embedder.hidden_size()
            );
        }
        Ok(Self { session, embedder })
    }

    /// Target backbone hidden size `H` expected by this assistant.
    pub fn hidden_size(&self) -> usize {
        self.session.signature().backbone_hidden_size
    }
}

impl SpeculativeProposer for SharedKvProposer<'_> {
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal> {
        let hidden = context
            .target_hidden
            .context("shared-KV proposer requires the target model's last hidden state")?;
        let guaranteed_token = context
            .guaranteed_token
            .context("shared-KV proposer requires the target model's greedy next token")?;
        let shared_kv = context
            .shared_kv_slices
            .context("shared-KV proposer requires target shared-KV slices")?;
        let hidden_size = self.session.signature().backbone_hidden_size;
        if hidden.len() != hidden_size {
            anyhow::bail!(
                "target_hidden length {} != shared-KV proposer hidden {hidden_size}",
                hidden.len()
            );
        }
        let seed_token = context
            .context_tokens
            .last()
            .copied()
            .context("shared-KV proposer requires at least one context token")?;

        let draft_count = context.width.saturating_sub(1);
        let mut tokens = Vec::with_capacity(draft_count + 1);
        tokens.push(guaranteed_token);

        // Reference contract (HF `SinglePositionMultiTokenCandidateGenerator`):
        // each step feeds `inputs_embeds = concat(embed(last_token), hidden)`,
        // where `embed` is the *target* model's raw input-token embedding and
        // `hidden` is the target's last hidden state (step 0) or the assistant's
        // own threaded `projected_state` (later steps). The RoPE position is held
        // constant by the exported graph (derived from the shared-KV length), so
        // the token embedding is the only per-step positional cue.
        //
        // The guaranteed target token is taken for free, but the assistant is
        // still run once to advance the threaded hidden state to the position
        // that follows it; that bootstrap step's own draft is discarded and
        // `last_token` is pinned to the guaranteed token. Subsequent steps emit
        // the real draft tokens.
        let mut last_hidden = hidden.to_vec();
        let mut last_token = seed_token;
        let mut inputs_embeds = vec![0.0f32; 2 * hidden_size];
        let position = i64::try_from(context.context_tokens.len().saturating_sub(1))
            .context("shared-KV proposer position exceeds i64")?;
        for step in 0..context.width {
            self.embedder
                .embed(last_token, &mut inputs_embeds[..hidden_size])?;
            inputs_embeds[hidden_size..].copy_from_slice(&last_hidden);
            let output = self
                .session
                .step(&inputs_embeds, position, shared_kv)
                .map_err(|error| {
                    anyhow::anyhow!("shared-KV proposer proposal step failed: {error}")
                })?;
            let drafted = TokenId::try_from(
                argmax(&output.logits)
                    .context("shared-KV proposer produced empty draft logits")?,
            )
            .context("shared-KV proposer token id exceeds u32 range")?;
            last_hidden = output.projected_state;
            if step == 0 {
                // Bootstrap: pin to the guaranteed target token so the drafts
                // that follow condition on it exactly.
                last_token = guaranteed_token;
            } else {
                tokens.push(drafted);
                last_token = drafted;
            }
        }
        Ok(SpeculativeProposal::linear(tokens))
    }

    fn name(&self) -> &str {
        "shared_kv_proposer"
    }
}

pub(crate) struct DraftModelProposer<'a> {
    draft_model: &'a mut DraftModel,
    draft_state: &'a mut DraftSession,
    rng: Option<&'a mut SamplingRng>,
}

impl<'a> DraftModelProposer<'a> {
    fn new(draft_model: &'a mut DraftModel, draft_state: &'a mut DraftSession) -> Self {
        Self {
            draft_model,
            draft_state,
            rng: None,
        }
    }

    fn with_rng(
        draft_model: &'a mut DraftModel,
        draft_state: &'a mut DraftSession,
        rng: &'a mut SamplingRng,
    ) -> Self {
        Self {
            draft_model,
            draft_state,
            rng: Some(rng),
        }
    }

    fn align_to_target_prefix(
        &mut self,
        target_tokens: &[TokenId],
        prefix_len: usize,
    ) -> anyhow::Result<()> {
        self.draft_state.tokens = target_tokens[..prefix_len].to_vec();
        if self.draft_state.kv_token_count > prefix_len {
            rewind_draft_state_to_len(self.draft_model, self.draft_state, prefix_len)?;
        }
        Ok(())
    }
}

impl SpeculativeProposer for DraftModelProposer<'_> {
    fn propose(
        &mut self,
        context: &SpeculativeProposerContext<'_>,
    ) -> anyhow::Result<SpeculativeProposal> {
        let mut fallback_rng = SamplingRng::new(context.options.seed);
        let rng = self.rng.as_deref_mut().unwrap_or(&mut fallback_rng);
        let tokens = propose_draft_tokens(
            self.draft_model,
            self.draft_state,
            context.width,
            context.generated_tokens,
            context.generated_text,
            context.first_step,
            context.options,
            context.chain,
            rng,
        )?;
        Ok(SpeculativeProposal::linear(tokens))
    }

    fn rewind(&mut self, target_tokens: &[TokenId]) -> anyhow::Result<()> {
        let common_len = common_prefix_len(&self.draft_state.tokens, target_tokens);
        if self.draft_state.kv_token_count > common_len {
            rewind_draft_state_to_len(self.draft_model, self.draft_state, common_len)?;
        }
        self.draft_state.tokens = target_tokens.to_vec();
        Ok(())
    }

    fn name(&self) -> &str {
        "draft_model"
    }
}

impl Engine {
    fn speculative_mode(&self, options: &GenerateOptions) -> SpeculativeMode {
        options
            .speculative_mode
            .clone()
            .unwrap_or_else(|| self.speculative_mode.clone())
    }

    pub(crate) fn should_use_speculative(&self, options: &GenerateOptions) -> bool {
        let mode_available = match self.speculative_mode(options) {
            SpeculativeMode::None => false,
            SpeculativeMode::DraftModel => self.draft.is_some(),
            SpeculativeMode::PromptLookup { ngram, max_tokens } => ngram > 0 && max_tokens > 0,
            SpeculativeMode::Mtp(config) => {
                self.mtp.as_ref().is_some_and(|mtp| mtp.config == config)
            }
            SpeculativeMode::Eagle3(config) => self
                .eagle3
                .as_ref()
                .is_some_and(|eagle3| eagle3.config == config),
            SpeculativeMode::SharedKv(config) => self
                .shared_kv_proposer
                .as_ref()
                .is_some_and(|assistant| assistant.config == config),
        };
        mode_available
            // Grammar processors carry per-request parser state; draft/verify
            // would need separate parser branches for speculative candidates.
            && options.constraint.is_none()
            && (options.greedy || options.temperature == 0.0)
            && self.kv_model.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_speculative_loop(
        &mut self,
        session_id: SessionId,
        state: &mut EngineSession,
        options: &GenerateOptions,
        chain: &ProcessorChain,
        max_context: Option<usize>,
        prefix_cache_hit_len: usize,
        generated_tokens: &mut Vec<TokenId>,
        generated_text: &mut String,
        generated_logprobs: &mut Option<Vec<crate::config::TokenLogprob>>,
        rng: &mut SamplingRng,
        mut callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        let speculative_mode = self.speculative_mode(options);
        let draft_width = match &speculative_mode {
            SpeculativeMode::PromptLookup { max_tokens, .. } => *max_tokens,
            SpeculativeMode::Mtp(_) => {
                self.mtp
                    .as_ref()
                    .map(|mtp| {
                        options
                            .num_speculative_tokens
                            .unwrap_or(mtp.num_speculative_tokens)
                    })
                    .context("MTP speculation requested without a loaded MTP head")?
                    + 1
            }
            SpeculativeMode::Eagle3(_) => {
                self.eagle3
                    .as_ref()
                    .map(|eagle3| {
                        options
                            .num_speculative_tokens
                            .unwrap_or(eagle3.num_speculative_tokens)
                    })
                    .context("EAGLE-3 speculation requested without a loaded EAGLE-3 head")?
                    + 1
            }
            SpeculativeMode::SharedKv(_) => {
                self.shared_kv_proposer
                    .as_ref()
                    .map(|assistant| {
                        options
                            .num_speculative_tokens
                            .unwrap_or(assistant.num_speculative_tokens)
                    })
                    .context(
                        "shared-KV proposer speculation requested without a loaded proposer model",
                    )?
                    + 1
            }
            _ => options
                .num_speculative_tokens
                .unwrap_or(self.num_speculative_tokens),
        }
        .max(1);
        if let Some(capture) = self.speculative_trace_capture.as_mut() {
            let (family, guaranteed_target_prefix) = match &speculative_mode {
                SpeculativeMode::DraftModel => (SpeculativeTraceFamily::DraftModel, 0),
                SpeculativeMode::PromptLookup { .. } => {
                    (SpeculativeTraceFamily::PromptLookup, 0)
                }
                SpeculativeMode::Mtp(_) => (SpeculativeTraceFamily::Mtp, 1),
                SpeculativeMode::Eagle3(_) => (SpeculativeTraceFamily::Eagle3, 1),
                SpeculativeMode::SharedKv(_) => (SpeculativeTraceFamily::SharedKv, 1),
                SpeculativeMode::None => unreachable!(
                    "target-only mode must not enter the speculative generation loop"
                ),
            };
            capture.family = Some(family);
            capture.max_additional_tokens =
                Some(draft_width.saturating_sub(guaranteed_target_prefix));
        }
        let mut mtp_proposer = if matches!(&speculative_mode, SpeculativeMode::Mtp(_)) {
            let mtp = self
                .mtp
                .as_ref()
                .context("MTP speculation requested without a loaded MTP head")?;
            Some(MtpProposer::new_owned(
                Arc::clone(&mtp.session),
                MtpDecodeOptions {
                    kv_mode: mtp.kv_mode,
                    batch_size: 1,
                    hc_mult: mtp.runtime_config.hc_mult,
                    hidden_state_rank4: mtp.runtime_config.target_hidden_layout
                        == MtpHiddenLayout::Bshc,
                    hidden_output: mtp.runtime_config.mtp_hidden_output.clone(),
                    state_output: mtp.runtime_config.mtp_state_output.clone(),
                },
                mtp.embedder.clone(),
                mtp.lm_head.clone(),
                mtp.runtime_config.cache_scope,
            )?)
        } else {
            None
        };
        let mut step = 0;

        loop {
            if generated_tokens.len() >= options.max_new_tokens {
                ensure_constrained_finish(options, generated_text, FinishReason::MaxTokens)?;
                return self.finish_result(
                    generated_tokens,
                    FinishReason::MaxTokens,
                    prefix_cache_hit_len,
                    generated_logprobs.as_deref(),
                );
            }
            if reached_context_limit(state.tokens.len(), max_context) {
                ensure_constrained_finish(options, generated_text, FinishReason::Length)?;
                return self.finish_result(
                    generated_tokens,
                    FinishReason::Length,
                    prefix_cache_hit_len,
                    generated_logprobs.as_deref(),
                );
            }

            let remaining_tokens = options.max_new_tokens - generated_tokens.len();
            let remaining_context = max_context
                .map(|limit| limit.saturating_sub(state.tokens.len()))
                .unwrap_or(remaining_tokens);
            let width = draft_width
                .min(remaining_tokens)
                .min(remaining_context)
                .max(1);

            let base_len = state.tokens.len();
            let base_generated_len = generated_tokens.len();
            let (mut base_logits, target_hidden, target_hidden_layers) =
                if let SpeculativeMode::Mtp(_) = &speculative_mode {
                    let hidden_output = self
                        .mtp
                        .as_ref()
                        .context("MTP speculation requested without a loaded MTP head")?
                        .hidden_output
                        .clone();
                    let (logits, hidden) = next_session_token_logits_and_hidden(
                        self.session
                            .as_deref()
                            .expect("ORT backend must own a decoder session"),
                        self.kv_model.as_ref(),
                        &mut self.kv_cache,
                        session_id,
                        state,
                        &hidden_output,
                    )?;
                    (logits, Some(hidden), None)
                } else if let SpeculativeMode::Eagle3(_) = &speculative_mode {
                    let hidden_outputs = self
                        .eagle3
                        .as_ref()
                        .context("EAGLE-3 speculation requested without a loaded EAGLE-3 head")?
                        .hidden_outputs
                        .clone();
                    let (logits, layers) = next_session_token_logits_and_hiddens(
                        self.session
                            .as_deref()
                            .expect("ORT backend must own a decoder session"),
                        self.kv_model.as_ref(),
                        &mut self.kv_cache,
                        session_id,
                        state,
                        &hidden_outputs,
                    )?;
                    let last_hidden = layers
                        .last()
                        .cloned()
                        .context("EAGLE-3 target hidden-state list was empty")?;
                    (logits, Some(last_hidden), Some(layers))
                } else if let SpeculativeMode::SharedKv(_) = &speculative_mode {
                    let hidden_output = self
                        .shared_kv_proposer
                        .as_ref()
                        .context(
                            "shared-KV proposer speculation requested without a loaded proposer model",
                        )?
                        .config
                        .target_hidden_output
                        .clone();
                    let (logits, hidden) = next_session_token_logits_and_hidden(
                        self.session
                            .as_deref()
                            .expect("ORT backend must own a decoder session"),
                        self.kv_model.as_ref(),
                        &mut self.kv_cache,
                        session_id,
                        state,
                        &hidden_output,
                    )?;
                    (logits, Some(hidden), None)
                } else {
                    (
                        next_session_token_logits(
                            self.session
                                .as_deref()
                                .expect("ORT backend must own a decoder session"),
                            self.kv_model.as_ref(),
                            &mut self.kv_cache,
                            session_id,
                            state,
                        )?,
                        None,
                        None,
                    )
                };
            let guaranteed_token = target_hidden
                .as_ref()
                .map(|_| argmax(&base_logits).context("target logits were empty"))
                .transpose()?
                .map(TokenId::try_from)
                .transpose()
                .context("target token id exceeds u32 range")?;

            // Slice the target's paged KV for the assistant's shared_kv.* inputs.
            let shared_kv_slices = if let SpeculativeMode::SharedKv(_) = &speculative_mode {
                Some(self.shared_kv_proposer_slices(session_id)?)
            } else {
                None
            };

            let proposer_context = SpeculativeProposerContext {
                width,
                context_tokens: &state.tokens,
                generated_tokens,
                generated_text,
                first_step: step,
                options,
                chain,
                target_hidden: target_hidden.as_deref(),
                target_hidden_layers: target_hidden_layers.as_deref(),
                guaranteed_token,
                shared_kv_slices: shared_kv_slices.as_deref(),
            };
            let draft_tokens = match &speculative_mode {
                SpeculativeMode::None => Vec::new(),
                SpeculativeMode::DraftModel => {
                    let draft_model = self
                        .draft
                        .as_mut()
                        .context("speculative decoding requested without a draft model")?;
                    let draft_state = state
                        .draft
                        .as_mut()
                        .context("speculative session missing draft state")?;
                    let mut proposer = DraftModelProposer::with_rng(draft_model, draft_state, rng);
                    proposer.align_to_target_prefix(&state.tokens, base_len)?;
                    proposer.propose(&proposer_context)?.tokens
                }
                SpeculativeMode::PromptLookup { ngram, max_tokens } => {
                    NgramProposer::new(*ngram, *max_tokens)?
                        .propose(&proposer_context)?
                        .tokens
                }
                SpeculativeMode::Mtp(_) => {
                    mtp_proposer
                        .as_mut()
                        .context("MTP proposer state was not initialized")?
                        .propose(&proposer_context)?
                        .tokens
                }
                SpeculativeMode::Eagle3(_) => {
                    let eagle3 = self
                        .eagle3
                        .as_ref()
                        .context("EAGLE-3 speculation requested without a loaded EAGLE-3 head")?;
                    Eagle3Proposer::new(
                        &eagle3.session,
                        Eagle3DecodeOptions {
                            kv_mode: eagle3.kv_mode,
                            batch_size: 1,
                        },
                        eagle3.embedder.clone(),
                    )?
                    .propose(&proposer_context)?
                    .tokens
                }
                SpeculativeMode::SharedKv(_) => {
                    let assistant = self.shared_kv_proposer.as_ref().context(
                        "shared-KV proposer speculation requested without a loaded proposer model",
                    )?;
                    SharedKvProposer::new(&assistant.session, &assistant.embedder)?
                        .propose(&proposer_context)?
                        .tokens
                }
            };
            self.last_speculative_stats.verification_steps += 1;
            self.last_speculative_stats.proposed_tokens += draft_tokens.len();

            state.tokens.extend_from_slice(&draft_tokens);
            let verified_logits = if draft_tokens.is_empty() {
                Vec::new()
            } else if state.decode_state.has_runner() {
                let logits =
                    run_decode_session_logits(&mut state.decode_state, &draft_tokens, base_len)?;
                self.kv_cache
                    .append(session_id, draft_tokens.len())
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to advance KV sequence {session_id}: {}", e)
                    })?;
                state.kv_token_count += draft_tokens.len();
                logits
            } else {
                let retained_base_len = state.decode_state.retained_kv_len(base_len);
                let outputs = run_decode_step(
                    self.session
                        .as_deref()
                        .expect("ORT backend must own a decoder session"),
                    &mut state.decode_state,
                    &draft_tokens,
                    base_len,
                )?;
                if state.decode_state.use_kv {
                    if let Some(kv_model) = &self.kv_model {
                        mirror_present_kv_to_pages(
                            self.session
                                .as_deref()
                                .expect("ORT backend must own a decoder session"),
                            kv_model,
                            &mut self.kv_cache,
                            session_id,
                            &outputs,
                            retained_base_len,
                            draft_tokens.len(),
                        )?;
                    } else {
                        self.kv_cache
                            .append(session_id, draft_tokens.len())
                            .map_err(|e| {
                                anyhow::anyhow!("Failed to advance KV sequence {session_id}: {}", e)
                            })?;
                    }
                    state.kv_token_count += draft_tokens.len();
                    apply_paged_sliding_window(
                        &mut self.kv_cache,
                        session_id,
                        state.decode_state.sliding_window(),
                        state.decode_state.sink_tokens(),
                    )?;
                }
                extract_logits_sequence(
                    self.session
                        .as_deref()
                        .expect("ORT backend must own a decoder session"),
                    outputs,
                )?
            };

            let mut target_logits = Vec::with_capacity(draft_tokens.len() + 1);
            target_logits.push(std::mem::take(&mut base_logits));
            target_logits.extend(verified_logits);

            let mut accepted = 0;
            let mut replacement = None;
            let mut selected_target_tokens = self
                .speculative_trace_capture
                .as_ref()
                .map(|_| Vec::with_capacity(draft_tokens.len() + 1));
            let mut candidate_logprobs = options.top_logprobs.map(|_| Vec::new());
            for idx in 0..draft_tokens.len() {
                let mut context = ProcessorContext {
                    prompt_tokens: state.tokens[..base_len].to_vec(),
                    generated_tokens: generated_tokens
                        .iter()
                        .copied()
                        .chain(draft_tokens[..idx].iter().copied())
                        .collect(),
                    generated_text: self
                        .tokenizer
                        .decode(
                            &generated_tokens
                                .iter()
                                .copied()
                                .chain(draft_tokens[..idx].iter().copied())
                                .collect::<Vec<_>>(),
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to detokenize speculative context: {}", e)
                        })?,
                    step: step + idx,
                };
                let target_token = select_next_token_with_rng(
                    &mut target_logits[idx],
                    &context,
                    options,
                    chain,
                    rng,
                );
                if let Some(tokens) = selected_target_tokens.as_mut() {
                    tokens.push(target_token);
                }
                if let (Some(top_logprobs), Some(logprobs)) =
                    (options.top_logprobs, candidate_logprobs.as_mut())
                {
                    logprobs.push(logprob_for_token(
                        &target_logits[idx],
                        target_token,
                        top_logprobs,
                    ));
                }
                if target_token == draft_tokens[idx] {
                    accepted += 1;
                } else {
                    replacement = Some(target_token);
                    context.generated_tokens.push(target_token);
                    break;
                }
            }
            self.last_speculative_stats.accepted_tokens += accepted;
            if accepted >= 2 {
                self.last_speculative_stats.multi_token_accepts += 1;
            }

            let mut commit_tokens = draft_tokens[..accepted].to_vec();
            let mut commit_logprobs = candidate_logprobs
                .as_ref()
                .map(|logprobs| logprobs[..accepted].to_vec());
            let rewind_len = base_len + accepted;
            rewind_target_state_to_len(
                self.session
                    .as_deref()
                    .expect("ORT backend must own a decoder session"),
                self.kv_model.as_ref(),
                &mut self.kv_cache,
                session_id,
                state,
                rewind_len,
            )?;

            if let Some(token) = replacement {
                commit_tokens.push(token);
                if let (Some(source), Some(commit)) =
                    (candidate_logprobs.as_ref(), commit_logprobs.as_mut())
                {
                    commit.push(source[accepted].clone());
                }
            } else if generated_tokens.len() + commit_tokens.len() < options.max_new_tokens
                && !reached_context_limit(base_len + commit_tokens.len(), max_context)
            {
                let mut context = ProcessorContext {
                    prompt_tokens: state.tokens[..base_len].to_vec(),
                    generated_tokens: generated_tokens
                        .iter()
                        .copied()
                        .chain(draft_tokens.iter().copied())
                        .collect(),
                    generated_text: self
                        .tokenizer
                        .decode(
                            &generated_tokens
                                .iter()
                                .copied()
                                .chain(draft_tokens.iter().copied())
                                .collect::<Vec<_>>(),
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to detokenize speculative context: {}", e)
                        })?,
                    step: step + draft_tokens.len(),
                };
                let token = select_next_token_with_rng(
                    target_logits
                        .last_mut()
                        .context("target verification did not produce next-token logits")?,
                    &context,
                    options,
                    chain,
                    rng,
                );
                if let Some(tokens) = selected_target_tokens.as_mut() {
                    tokens.push(token);
                }
                if let (Some(top_logprobs), Some(logprobs)) =
                    (options.top_logprobs, commit_logprobs.as_mut())
                {
                    logprobs.push(logprob_for_token(
                        target_logits
                            .last()
                            .context("target verification did not produce next-token logits")?,
                        token,
                        top_logprobs,
                    ));
                }
                context.generated_tokens.push(token);
                commit_tokens.push(token);
            }

            if matches!(&speculative_mode, SpeculativeMode::DraftModel) {
                self.notify_draft_acceptance(state, accepted, &commit_tokens)?;
            } else if let Some(proposer) = mtp_proposer.as_mut() {
                proposer.accept(&SpeculativeAcceptContext {
                    accepted_prefix_len: accepted,
                    committed_tokens: &commit_tokens,
                    target_tokens: &state.tokens,
                })?;
            }

            let mut captured_commits = selected_target_tokens
                .as_ref()
                .map(|_| Vec::with_capacity(commit_tokens.len()));
            for (commit_idx, token_id) in commit_tokens.into_iter().enumerate() {
                if generated_tokens.len() >= options.max_new_tokens
                    || (commit_idx >= accepted
                        && reached_context_limit(state.tokens.len(), max_context))
                {
                    break;
                }
                if commit_idx >= accepted {
                    state.tokens.push(token_id);
                }
                self.scheduler.advance(session_id);
                let prompt_tokens = state.tokens[..base_len.min(state.tokens.len())].to_vec();
                let mut commit_state = DecodeLoopState {
                    generated_tokens: std::mem::take(generated_tokens),
                    generated_text: std::mem::take(generated_text),
                    step,
                    prefix_cache_hit_len,
                    logprobs: generated_logprobs.take(),
                    rng: SamplingRng::new(options.seed),
                    custom_sampler: None,
                };
                if let (Some(all_logprobs), Some(step_logprobs)) =
                    (commit_state.logprobs.as_mut(), commit_logprobs.as_ref())
                {
                    all_logprobs.push(step_logprobs[commit_idx].clone());
                }
                let finish_reason = commit_selected_token(
                    &mut commit_state,
                    prompt_tokens,
                    token_id,
                    options,
                    chain,
                    &self.tokenizer,
                    callback.as_deref_mut(),
                )?;
                *generated_tokens = commit_state.generated_tokens;
                *generated_text = commit_state.generated_text;
                *generated_logprobs = commit_state.logprobs;
                step = commit_state.step;
                if let Some(tokens) = captured_commits.as_mut() {
                    tokens.push(token_id);
                }
                if let Some(finish_reason) = finish_reason {
                    trim_overmaterialized_target_kv(
                        self.session
                            .as_deref()
                            .expect("ORT backend must own a decoder session"),
                        self.kv_model.as_ref(),
                        &mut self.kv_cache,
                        session_id,
                        state,
                    )?;
                    if matches!(&speculative_mode, SpeculativeMode::DraftModel) {
                        self.sync_draft_to_target(state)?;
                    }
                    self.record_speculative_iteration(
                        base_generated_len,
                        &draft_tokens,
                        selected_target_tokens.take(),
                        captured_commits.take(),
                    );
                    return self.finish_result(
                        generated_tokens,
                        finish_reason,
                        prefix_cache_hit_len,
                        generated_logprobs.as_deref(),
                    );
                }
            }

            if matches!(&speculative_mode, SpeculativeMode::DraftModel) {
                self.sync_draft_to_target(state)?;
            }

            if generated_tokens.len() == base_generated_len {
                anyhow::bail!("speculative decoding made no progress");
            }
            self.record_speculative_iteration(
                base_generated_len,
                &draft_tokens,
                selected_target_tokens,
                captured_commits,
            );
        }
    }

    fn record_speculative_iteration(
        &mut self,
        output_offset: usize,
        proposal_token_ids: &[TokenId],
        target_token_ids: Option<Vec<TokenId>>,
        committed_token_ids: Option<Vec<TokenId>>,
    ) {
        let Some(capture) = self.speculative_trace_capture.as_mut() else {
            return;
        };
        capture.iterations.push(SpeculativeIterationTrace {
            output_offset,
            proposal_token_ids: proposal_token_ids.to_vec(),
            target_token_ids: target_token_ids
                .expect("active speculative trace must collect target tokens"),
            committed_token_ids: committed_token_ids
                .expect("active speculative trace must collect committed tokens"),
        });
    }

    /// Slice the target model's paged KV cache into per-group `shared_kv.*`
    /// tensors for the shared-KV proposer. Each configured group binds a single
    /// representative target layer (the last listed `target_layers` index); the
    /// assistant has no cache of its own, so these slices are materialized once
    /// at the current base position and reused across all draft steps.
    fn shared_kv_proposer_slices(
        &self,
        session_id: SessionId,
    ) -> anyhow::Result<Vec<SharedKvInput>> {
        let assistant = self
            .shared_kv_proposer
            .as_ref()
            .context("shared-KV slicing requested without a loaded proposer model")?;
        let materialized = self
            .kv_cache
            .materialize_sequence(session_id)
            .map_err(|e| anyhow::anyhow!("Failed to materialize target KV for shared_kv: {}", e))?;
        shared_kv_slices_from_materialized(&assistant.config.shared_kv, &materialized)
    }

    pub(crate) fn sync_draft_to_target(&mut self, state: &mut EngineSession) -> anyhow::Result<()> {
        if let (Some(draft_model), Some(draft_state)) = (&mut self.draft, &mut state.draft) {
            DraftModelProposer::new(draft_model, draft_state).rewind(&state.tokens)?;
        }
        Ok(())
    }

    fn notify_draft_acceptance(
        &mut self,
        state: &mut EngineSession,
        accepted_prefix_len: usize,
        committed_tokens: &[TokenId],
    ) -> anyhow::Result<()> {
        if let (Some(draft_model), Some(draft_state)) = (&mut self.draft, &mut state.draft) {
            DraftModelProposer::new(draft_model, draft_state).accept(
                &SpeculativeAcceptContext {
                    accepted_prefix_len,
                    committed_tokens,
                    target_tokens: &state.tokens,
                },
            )?;
        }
        Ok(())
    }
}

/// Slice a materialized target KV cache into per-group `shared_kv.*` proposer
/// inputs, reading each group's `num_kv_heads`/`head_dim` from the **specific**
/// target layer it references (the last listed `target_layers` index) rather
/// than a single global geometry. This makes heterogeneous per-layer head_dim
/// models (e.g. Gemma-4 sliding vs full) bind correctly.
pub(crate) fn shared_kv_slices_from_materialized(
    groups: &[crate::config::SharedKvBinding],
    materialized: &onnx_genai_kv::MaterializedKv,
) -> anyhow::Result<Vec<SharedKvInput>> {
    let num_layers = materialized.layers.len();
    let mut slices = Vec::with_capacity(groups.len());
    for group in groups {
        let layer_idx = group
            .target_layers
            .last()
            .copied()
            .with_context(|| format!("shared_kv group '{}' has no target layers", group.name))?;
        let layer = materialized.layers.get(layer_idx).with_context(|| {
            format!(
                "shared_kv group '{}' references target layer {} but only {} layers exist",
                group.name, layer_idx, num_layers
            )
        })?;
        slices.push(SharedKvInput {
            name: group.name.clone(),
            key: layer.key.clone(),
            value: layer.value.clone(),
            kv_heads: layer.num_kv_heads,
            kv_len: materialized.sequence_len,
            head_dim: layer.head_dim,
        });
    }
    Ok(slices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_ort::{Environment, SessionOptions};
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    /// The shared-KV slicer must read each group's `num_kv_heads`/`head_dim`
    /// from the specific target layer it references, not a single global value.
    /// With a heterogeneous cache (layer 0: 2×8 sliding, layer 1: 3×16 full),
    /// the sliding group must bind 8-dim slices and the full group 16-dim, each
    /// with its layer's own head count and buffer.
    #[test]
    fn shared_kv_slices_pick_per_layer_geometry() -> anyhow::Result<()> {
        use crate::config::SharedKvBinding;
        use onnx_genai_kv::{MaterializedKv, MaterializedLayerKv};

        let seq_len = 2;
        // Layer 0 (sliding): num_kv_heads=2, head_dim=8 → 2*2*8 = 32 floats.
        let sliding_key: Vec<f32> = (0..2 * seq_len * 8).map(|v| v as f32).collect();
        let sliding_value: Vec<f32> = (0..2 * seq_len * 8).map(|v| (v + 1000) as f32).collect();
        // Layer 1 (full): num_kv_heads=3, head_dim=16 → 3*2*16 = 96 floats.
        let full_key: Vec<f32> = (0..3 * seq_len * 16).map(|v| (v + 2000) as f32).collect();
        let full_value: Vec<f32> = (0..3 * seq_len * 16).map(|v| (v + 3000) as f32).collect();

        let materialized = MaterializedKv {
            start_position: 0,
            sink_len: 0,
            sequence_len: seq_len,
            layers: vec![
                MaterializedLayerKv {
                    key: sliding_key.clone(),
                    value: sliding_value.clone(),
                    num_kv_heads: 2,
                    head_dim: 8,
                },
                MaterializedLayerKv {
                    key: full_key.clone(),
                    value: full_value.clone(),
                    num_kv_heads: 3,
                    head_dim: 16,
                },
            ],
        };

        let groups = vec![
            SharedKvBinding {
                name: "sliding_attention".into(),
                target_layers: vec![0],
            },
            SharedKvBinding {
                name: "full_attention".into(),
                target_layers: vec![1],
            },
        ];

        let slices = shared_kv_slices_from_materialized(&groups, &materialized)?;
        assert_eq!(slices.len(), 2);

        let sliding = &slices[0];
        assert_eq!(sliding.name, "sliding_attention");
        assert_eq!(sliding.kv_heads, 2, "sliding group must use layer 0 heads");
        assert_eq!(sliding.head_dim, 8, "sliding group must use layer 0 head_dim");
        assert_eq!(sliding.kv_len, seq_len);
        assert_eq!(sliding.key, sliding_key);
        assert_eq!(sliding.value, sliding_value);

        let full = &slices[1];
        assert_eq!(full.name, "full_attention");
        assert_eq!(full.kv_heads, 3, "full group must use layer 1 heads");
        assert_eq!(full.head_dim, 16, "full group must use layer 1 head_dim");
        assert_eq!(full.kv_len, seq_len);
        assert_eq!(full.key, full_key);
        assert_eq!(full.value, full_value);

        // An out-of-range target layer is a clear error, not a silent misread.
        let bad = vec![SharedKvBinding {
            name: "oob".into(),
            target_layers: vec![9],
        }];
        assert!(shared_kv_slices_from_materialized(&bad, &materialized).is_err());
        Ok(())
    }

    struct StubProposer {
        tokens: Vec<TokenId>,
        accepted: Option<usize>,
        rewound_to: Option<Vec<TokenId>>,
    }

    impl SpeculativeProposer for StubProposer {
        fn propose(
            &mut self,
            _context: &SpeculativeProposerContext<'_>,
        ) -> anyhow::Result<SpeculativeProposal> {
            Ok(SpeculativeProposal::linear(self.tokens.clone()))
        }

        fn accept(&mut self, context: &SpeculativeAcceptContext<'_>) -> anyhow::Result<()> {
            self.accepted = Some(context.accepted_prefix_len);
            Ok(())
        }

        fn rewind(&mut self, target_tokens: &[TokenId]) -> anyhow::Result<()> {
            self.rewound_to = Some(target_tokens.to_vec());
            Ok(())
        }

        fn name(&self) -> &str {
            "stub"
        }
    }

    #[test]
    fn speculative_proposer_trait_supports_non_draft_sources() -> anyhow::Result<()> {
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let mut proposer = StubProposer {
            tokens: vec![3, 5],
            accepted: None,
            rewound_to: None,
        };

        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 2,
            context_tokens: &[1],
            generated_tokens: &[1],
            generated_text: "a",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: None,
            target_hidden_layers: None,
            guaranteed_token: None,
            shared_kv_slices: None,
        })?;
        proposer.accept(&SpeculativeAcceptContext {
            accepted_prefix_len: 1,
            committed_tokens: &[3, 4],
            target_tokens: &[1, 3, 4],
        })?;
        proposer.rewind(&[1, 3, 4])?;

        assert_eq!(proposal.tokens, vec![3, 5]);
        assert_eq!(proposer.accepted, Some(1));
        assert_eq!(proposer.rewound_to, Some(vec![1, 3, 4]));
        Ok(())
    }

    #[test]
    fn ngram_proposer_copies_most_recent_matching_continuation() -> anyhow::Result<()> {
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let mut proposer = NgramProposer::new(2, 4)?;
        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 3,
            context_tokens: &[7, 8, 9, 4, 7, 8],
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: None,
            target_hidden_layers: None,
            guaranteed_token: None,
            shared_kv_slices: None,
        })?;

        assert_eq!(proposal.tokens, vec![9, 4, 7]);
        Ok(())
    }

    #[test]
    fn ngram_proposer_validates_configuration_and_empty_matches() -> anyhow::Result<()> {
        assert_eq!(
            NgramProposer::new(0, 1).unwrap_err().to_string(),
            "ngram must be greater than zero"
        );
        assert_eq!(
            NgramProposer::new(1, 0).unwrap_err().to_string(),
            "max_tokens must be greater than zero"
        );

        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let context = |tokens| SpeculativeProposerContext {
            width: 4,
            context_tokens: tokens,
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: None,
            target_hidden_layers: None,
            guaranteed_token: None,
            shared_kv_slices: None,
        };
        let mut proposer = NgramProposer::new(2, 4)?;

        assert!(proposer.propose(&context(&[1, 2]))?.tokens.is_empty());
        assert!(proposer.propose(&context(&[1, 2, 3, 4]))?.tokens.is_empty());
        assert_eq!(proposer.name(), "prompt_lookup");
        Ok(())
    }

    #[test]
    fn ngram_proposer_respects_request_and_config_widths() -> anyhow::Result<()> {
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let tokens = [1, 2, 3, 4, 1, 2];
        let mut proposer = NgramProposer::new(2, 2)?;
        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 8,
            context_tokens: &tokens,
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: None,
            target_hidden_layers: None,
            guaranteed_token: None,
            shared_kv_slices: None,
        })?;
        assert_eq!(proposal.tokens, vec![3, 4]);

        let mut proposer = NgramProposer::new(2, 8)?;
        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 1,
            context_tokens: &tokens,
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: None,
            target_hidden_layers: None,
            guaranteed_token: None,
            shared_kv_slices: None,
        })?;
        assert_eq!(proposal.tokens, vec![3]);
        Ok(())
    }

    fn lcg_weights(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 33) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn eagle3_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn load_eagle3_head() -> anyhow::Result<Session> {
        static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
        let environment = ENVIRONMENT
            .get_or_init(|| Environment::new("engine-eagle3-test").expect("environment"));
        let head_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-eagle3/model.onnx");
        Ok(Session::new(
            environment,
            &head_path,
            SessionOptions::default().with_intra_op_threads(1),
        )?)
    }

    #[test]
    fn mtp_proposer_uses_real_head_and_returns_guaranteed_plus_k_drafts() -> anyhow::Result<()> {
        const HIDDEN: usize = 16;
        const VOCAB: usize = 32;
        static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
        let environment =
            ENVIRONMENT.get_or_init(|| Environment::new("engine-mtp-test").expect("environment"));
        let head_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen35-mtp/model.onnx");
        let head = Session::new(
            environment,
            &head_path,
            SessionOptions::default().with_intra_op_threads(1),
        )?;
        let embedder =
            LinearEmbedder::new(lcg_weights(0x1111_2222, VOCAB * HIDDEN), VOCAB, HIDDEN)?;
        let lm_head = LinearLmHead::new(lcg_weights(0x3333_4444, HIDDEN * VOCAB), HIDDEN, VOCAB)?;
        let hidden = lcg_weights(0xA5A5_1234, HIDDEN);
        let mut logits = vec![0.0; VOCAB];
        LmHead::logits(&lm_head, &hidden, &mut logits)?;
        let guaranteed = argmax(&logits).context("target logits were empty")? as TokenId;
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let mut proposer = MtpProposer::new(&head, MtpDecodeOptions::default(), embedder, lm_head)?;

        fn assert_speculative_proposer<T: SpeculativeProposer>(_proposer: &T) {}
        assert_speculative_proposer(&proposer);
        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 5,
            context_tokens: &[1],
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: Some(&hidden),
            target_hidden_layers: None,
            guaranteed_token: Some(guaranteed),
            shared_kv_slices: None,
        })?;

        assert_eq!(proposer.name(), "mtp");
        assert_eq!(guaranteed, 13);
        assert_eq!(proposal.tokens.len(), 5);
        assert_eq!(proposal.tokens.first(), Some(&guaranteed));
        assert_eq!(proposal.tokens, vec![guaranteed, 27, 11, 2, 27]);
        Ok(())
    }

    #[derive(Clone)]
    struct ConstantEmbedder;

    impl TokenEmbedder for ConstantEmbedder {
        fn hidden_size(&self) -> usize {
            2
        }

        fn embed(&self, _token: TokenId, out: &mut [f32]) -> anyhow::Result<()> {
            out.copy_from_slice(&[1.0, 0.0]);
            Ok(())
        }
    }

    #[derive(Clone)]
    struct RecordingLmHead {
        inputs: Arc<Mutex<Vec<Vec<f32>>>>,
    }

    impl LmHead for RecordingLmHead {
        fn vocab_size(&self) -> usize {
            1
        }

        fn logits(&self, hidden: &[f32], out: &mut [f32]) -> anyhow::Result<()> {
            self.inputs
                .lock()
                .map_err(|_| anyhow::anyhow!("recording LM-head lock poisoned"))?
                .push(hidden.to_vec());
            out[0] = 1.0;
            Ok(())
        }
    }

    #[test]
    fn mtp_proposer_threads_rank4_hc_state_across_drafts() -> anyhow::Result<()> {
        static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
        let environment = ENVIRONMENT
            .get_or_init(|| Environment::new("engine-mtp-hc-test").expect("environment"));
        let head_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-hc-mtp/model.onnx");
        let head = Session::new(
            environment,
            &head_path,
            SessionOptions::default().with_intra_op_threads(1),
        )?;
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let mut proposer = MtpProposer::new(
            &head,
            MtpDecodeOptions {
                kv_mode: onnx_genai_ort::MtpDraftKvMode::HiddenThreaded,
                batch_size: 1,
                hc_mult: 2,
                hidden_state_rank4: true,
                hidden_output: "mtp_hidden".into(),
                state_output: Some("mtp_state".into()),
            },
            ConstantEmbedder,
            RecordingLmHead {
                inputs: Arc::clone(&recorded),
            },
        )?;
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let target_hc = vec![0.0; 4];

        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 3,
            context_tokens: &[4, 5],
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: Some(&target_hc),
            target_hidden_layers: None,
            guaranteed_token: Some(0),
            shared_kv_slices: None,
        })?;

        assert_eq!(proposal.tokens, vec![0, 0, 0]);
        assert_eq!(
            *recorded
                .lock()
                .map_err(|_| anyhow::anyhow!("recording LM-head lock poisoned"))?,
            vec![vec![2.0, 0.0], vec![4.0, 0.0]]
        );
        Ok(())
    }

    #[test]
    fn mtp_package_references_borrow_target_initializers() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-mtp-full");
        let (embedder, lm_head, vocab_size) = load_target_initializer_adapters(
            &fixture.join("model.onnx"),
            "transformer.wte.weight",
            "lm_head.weight_t",
            16,
        )?;
        assert_eq!(vocab_size, 32);

        let mut embedded = vec![0.0; 16];
        embedder.embed(7, &mut embedded)?;
        let raw_embedding = std::fs::read(fixture.join("embedding.f32"))?;
        let expected_embedding = raw_embedding
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .collect::<Vec<_>>();
        assert_eq!(embedded, expected_embedding[7 * 16..8 * 16]);

        let hidden = lcg_weights(0xDEAD_BEEF, 16);
        let mut package_logits = vec![0.0; 32];
        lm_head.logits(&hidden, &mut package_logits)?;
        let raw_lm_head = std::fs::read(fixture.join("lm_head.f32"))?;
        let linear_lm_head = LinearLmHead::new(
            raw_lm_head
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
                .collect(),
            16,
            32,
        )?;
        let mut expected_logits = vec![0.0; 32];
        linear_lm_head.logits(&hidden, &mut expected_logits)?;
        assert_eq!(package_logits, expected_logits);
        Ok(())
    }

    #[test]
    fn eagle3_proposer_loads_fixture_and_returns_shape_correct_proposal() -> anyhow::Result<()> {
        const HIDDEN: usize = 16;
        const VOCAB: usize = 32;
        let _guard = eagle3_test_lock()
            .lock()
            .map_err(|_| anyhow::anyhow!("EAGLE-3 test lock poisoned"))?;
        let head = load_eagle3_head()?;
        let embedder =
            LinearEmbedder::new(lcg_weights(0x5555_6666, VOCAB * HIDDEN), VOCAB, HIDDEN)?;
        let layers = vec![
            lcg_weights(0x1000_0001, HIDDEN),
            lcg_weights(0x2000_0002, HIDDEN),
            lcg_weights(0x3000_0003, HIDDEN),
        ];
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let mut proposer = Eagle3Proposer::new(&head, Eagle3DecodeOptions::default(), embedder)?;
        let proposal = proposer.propose(&SpeculativeProposerContext {
            width: 4,
            context_tokens: &[1, 2],
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: Some(&layers[2]),
            target_hidden_layers: Some(&layers),
            guaranteed_token: Some(7),
            shared_kv_slices: None,
        })?;

        assert_eq!(proposer.name(), "eagle3");
        assert_eq!(proposal.tokens.len(), 4);
        assert_eq!(proposal.tokens.first(), Some(&7));
        assert!(proposal.tokens.iter().all(|&token| token < VOCAB as u32));
        assert!(proposal.positions.is_none());
        assert!(proposal.tree.is_none());
        Ok(())
    }

    #[test]
    fn eagle3_proposer_accept_then_propose_resets_draft_state() -> anyhow::Result<()> {
        const HIDDEN: usize = 16;
        const VOCAB: usize = 32;
        let _guard = eagle3_test_lock()
            .lock()
            .map_err(|_| anyhow::anyhow!("EAGLE-3 test lock poisoned"))?;
        let head = load_eagle3_head()?;
        let embedder =
            LinearEmbedder::new(lcg_weights(0x7777_8888, VOCAB * HIDDEN), VOCAB, HIDDEN)?;
        let layers = vec![
            lcg_weights(0x4000_0004, HIDDEN),
            lcg_weights(0x5000_0005, HIDDEN),
            lcg_weights(0x6000_0006, HIDDEN),
        ];
        let options = GenerateOptions::default();
        let chain = ProcessorChain::new();
        let context = SpeculativeProposerContext {
            width: 3,
            context_tokens: &[4, 5, 6],
            generated_tokens: &[],
            generated_text: "",
            first_step: 0,
            options: &options,
            chain: &chain,
            target_hidden: Some(&layers[2]),
            target_hidden_layers: Some(&layers),
            guaranteed_token: Some(9),
            shared_kv_slices: None,
        };
        let mut proposer = Eagle3Proposer::new(&head, Eagle3DecodeOptions::default(), embedder)?;

        let first = proposer.propose(&context)?;
        proposer.accept(&SpeculativeAcceptContext {
            accepted_prefix_len: 2,
            committed_tokens: &first.tokens[..2],
            target_tokens: &[4, 5, 6, first.tokens[0], first.tokens[1]],
        })?;
        let second = proposer.propose(&context)?;

        assert_eq!(first, second);
        Ok(())
    }

    #[test]
    fn mtp_mode_selects_mtp_proposer_contract() {
        let mode = SpeculativeMode::Mtp(crate::config::MtpConfig {
            head_model: "mtp.onnx".into(),
            target_hidden_output: "hidden_states".into(),
            embedding_weights: "embed.f32".into(),
            lm_head_weights: "lm_head.f32".into(),
            vocab_size: 32,
            hidden_size: 16,
            kv_mode: onnx_genai_ort::MtpDraftKvMode::HiddenThreaded,
            num_speculative_tokens: 4,
        });
        let selected = match mode {
            SpeculativeMode::Mtp(_) => "mtp",
            SpeculativeMode::Eagle3(_) => "eagle3",
            SpeculativeMode::DraftModel => "draft_model",
            SpeculativeMode::PromptLookup { .. } => "prompt_lookup",
            SpeculativeMode::SharedKv(_) => "shared_kv_proposer",
            SpeculativeMode::None => "none",
        };
        assert_eq!(selected, "mtp");
    }

    #[test]
    fn eagle3_mode_selects_eagle3_proposer_contract() {
        let mode = SpeculativeMode::Eagle3(crate::config::Eagle3Config {
            head_model: "eagle3.onnx".into(),
            target_hidden_outputs: vec!["low".into(), "mid".into(), "high".into()],
            embedding_weights: "embed.f32".into(),
            vocab_size: 32,
            hidden_size: 16,
            kv_mode: onnx_genai_ort::Eagle3DraftKvMode::HiddenThreaded,
            num_speculative_tokens: 4,
        });
        let selected = match mode {
            SpeculativeMode::Eagle3(_) => "eagle3",
            SpeculativeMode::Mtp(_) => "mtp",
            SpeculativeMode::DraftModel => "draft_model",
            SpeculativeMode::PromptLookup { .. } => "prompt_lookup",
            SpeculativeMode::SharedKv(_) => "shared_kv_proposer",
            SpeculativeMode::None => "none",
        };
        assert_eq!(selected, "eagle3");
    }
}
